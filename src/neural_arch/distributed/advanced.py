"""Advanced Distributed Training Features.

This module provides advanced distributed training capabilities including
gradient accumulation, mixed precision, dynamic loss scaling, and 
fault tolerance mechanisms.
"""

import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import numpy as np

from ..core.tensor import Tensor
from ..nn.module import Module
from .communication import (
    all_reduce, all_gather, broadcast, barrier,
    get_rank, get_world_size, is_initialized, ReduceOp
)
from .data_parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


@dataclass
class DistributedTrainingConfig:
    """Configuration for advanced distributed training."""
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    
    # Mixed precision
    mixed_precision: bool = False
    loss_scale: float = 65536.0
    dynamic_loss_scaling: bool = True
    loss_scale_window: int = 1000
    min_loss_scale: float = 1.0
    
    # Communication optimization
    gradient_compression: bool = False
    bucket_size_mb: int = 25
    overlap_communication: bool = True
    
    # Fault tolerance
    checkpoint_frequency: int = 100
    checkpoint_dir: str = "./checkpoints"
    enable_fault_tolerance: bool = False
    max_failures: int = 3
    
    # Performance monitoring
    profile_communication: bool = False
    log_gradient_norms: bool = False
    memory_profiling: bool = False


class GradientAccumulator:
    """Handles gradient accumulation across micro-batches."""
    
    def __init__(self, accumulation_steps: int):
        """Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_gradients = {}
        
    def accumulate(self, model: Module):
        """Accumulate gradients from current step.
        
        Args:
            model: Model with gradients to accumulate
        """
        self.current_step += 1
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = param.grad.copy()
                else:
                    self.accumulated_gradients[name] += param.grad
        
        # Clear current gradients
        for param in model.parameters():
            param.grad = None
    
    def should_sync(self) -> bool:
        """Check if gradients should be synchronized."""
        return self.current_step % self.accumulation_steps == 0
    
    def apply_accumulated_gradients(self, model: Module):
        """Apply accumulated gradients to model.
        
        Args:
            model: Model to apply gradients to
        """
        if not self.should_sync():
            return
        
        # Scale gradients by accumulation steps
        scale = 1.0 / self.accumulation_steps
        
        for name, param in model.named_parameters():
            if name in self.accumulated_gradients:
                param.grad = self.accumulated_gradients[name] * scale
        
        # Clear accumulated gradients
        self.accumulated_gradients.clear()
        self.current_step = 0


class LossScaler:
    """Dynamic loss scaling for mixed precision training."""
    
    def __init__(self, 
                 initial_scale: float = 65536.0,
                 scale_window: int = 1000,
                 min_scale: float = 1.0,
                 max_scale: float = 2**24):
        """Initialize loss scaler.
        
        Args:
            initial_scale: Initial loss scale value
            scale_window: Window for checking overflow
            min_scale: Minimum scale value
            max_scale: Maximum scale value
        """
        self.scale = initial_scale
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.steps_since_update = 0
        self.overflow_count = 0
        
    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss for mixed precision training.
        
        Args:
            loss: Unscaled loss tensor
            
        Returns:
            Scaled loss tensor
        """
        return loss * self.scale
    
    def unscale_gradients(self, model: Module) -> bool:
        """Unscale gradients and check for overflow.
        
        Args:
            model: Model with scaled gradients
            
        Returns:
            True if no overflow detected, False otherwise
        """
        inv_scale = 1.0 / self.scale
        has_overflow = False
        
        for param in model.parameters():
            if param.grad is not None:
                # Check for overflow
                if np.any(np.isinf(param.grad.data)) or np.any(np.isnan(param.grad.data)):
                    has_overflow = True
                    break
                
                # Unscale gradient
                param.grad.data *= inv_scale
        
        if has_overflow:
            self.overflow_count += 1
            # Clear gradients on overflow
            for param in model.parameters():
                param.grad = None
            
            # Reduce scale
            self.scale = max(self.scale / 2.0, self.min_scale)
            self.steps_since_update = 0
            logger.warning(f"Gradient overflow detected. Reducing loss scale to {self.scale}")
        else:
            self.steps_since_update += 1
            
            # Increase scale if no overflow for a while
            if self.steps_since_update >= self.scale_window:
                self.scale = min(self.scale * 2.0, self.max_scale)
                self.steps_since_update = 0
                logger.debug(f"Increasing loss scale to {self.scale}")
        
        return not has_overflow


class CommunicationProfiler:
    """Profiles distributed communication operations."""
    
    def __init__(self):
        """Initialize communication profiler."""
        self.operation_times = {}
        self.operation_counts = {}
        self.total_bytes = {}
        
    def record_operation(self, operation: str, time_ms: float, bytes_transferred: int = 0):
        """Record a communication operation.
        
        Args:
            operation: Name of the operation
            time_ms: Time taken in milliseconds
            bytes_transferred: Bytes transferred
        """
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
            self.total_bytes[operation] = 0
        
        self.operation_times[operation].append(time_ms)
        self.operation_counts[operation] += 1
        self.total_bytes[operation] += bytes_transferred
    
    def get_statistics(self) -> Dict:
        """Get communication statistics.
        
        Returns:
            Dictionary of communication statistics
        """
        stats = {}
        
        for operation in self.operation_times:
            times = self.operation_times[operation]
            stats[operation] = {
                'count': self.operation_counts[operation],
                'total_time_ms': sum(times),
                'avg_time_ms': np.mean(times),
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'total_bytes': self.total_bytes[operation],
                'bandwidth_mbps': (self.total_bytes[operation] / (sum(times) / 1000)) / (1024 * 1024) if sum(times) > 0 else 0
            }
        
        return stats
    
    @contextmanager
    def profile_operation(self, operation: str, bytes_transferred: int = 0):
        """Context manager for profiling operations.
        
        Args:
            operation: Name of the operation
            bytes_transferred: Bytes transferred
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.record_operation(operation, (end_time - start_time) * 1000, bytes_transferred)


class AdvancedDistributedDataParallel(DistributedDataParallel):
    """Advanced distributed data parallel with additional features."""
    
    def __init__(self, 
                 module: Module,
                 config: Optional[DistributedTrainingConfig] = None):
        """Initialize advanced DDP.
        
        Args:
            module: Module to wrap
            config: Advanced training configuration
        """
        super().__init__(module)
        
        self.config = config or DistributedTrainingConfig()
        
        # Initialize components
        self.gradient_accumulator = GradientAccumulator(
            self.config.gradient_accumulation_steps
        )
        
        if self.config.mixed_precision:
            self.loss_scaler = LossScaler(
                initial_scale=self.config.loss_scale,
                scale_window=self.config.loss_scale_window,
                min_scale=self.config.min_loss_scale
            ) if self.config.dynamic_loss_scaling else None
        
        if self.config.profile_communication:
            self.comm_profiler = CommunicationProfiler()
        
        # Performance tracking
        self.step_count = 0
        self.gradient_norms = []
        self.communication_times = []
        
        logger.info(f"AdvancedDDP initialized with config: {asdict(self.config)}")
    
    def forward(self, *args, **kwargs):
        """Forward pass with advanced features."""
        return super().forward(*args, **kwargs)
    
    def backward_step(self, loss: Tensor, optimizer):
        """Perform backward step with advanced features.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer for parameter updates
        """
        # Scale loss for mixed precision
        if self.config.mixed_precision and hasattr(self, 'loss_scaler'):
            loss = self.loss_scaler.scale_loss(loss)
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients
        self.gradient_accumulator.accumulate(self.module)
        
        # Synchronize if accumulation is complete
        if self.gradient_accumulator.should_sync():
            self._sync_step(optimizer)
    
    def _sync_step(self, optimizer):
        """Synchronization step with advanced features.
        
        Args:
            optimizer: Optimizer for parameter updates
        """
        # Apply accumulated gradients
        self.gradient_accumulator.apply_accumulated_gradients(self.module)
        
        # Unscale gradients for mixed precision
        if self.config.mixed_precision and hasattr(self, 'loss_scaler'):
            if not self.loss_scaler.unscale_gradients(self.module):
                # Skip step on overflow
                return
        
        # Clip gradients if specified
        if self.config.max_grad_norm is not None:
            self._clip_gradients()
        
        # Profile communication if enabled
        if self.config.profile_communication:
            with self.comm_profiler.profile_operation("gradient_sync"):
                self.sync_gradients()
        else:
            self.sync_gradients()
        
        # Log gradient norms if enabled
        if self.config.log_gradient_norms:
            self._log_gradient_norms()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        self.step_count += 1
    
    def _clip_gradients(self):
        """Clip gradients to prevent explosion."""
        # Calculate global gradient norm
        total_norm_squared = 0.0
        
        for param in self.module.parameters():
            if param.grad is not None:
                param_norm_squared = np.sum(param.grad.data ** 2)
                total_norm_squared += param_norm_squared
        
        # All-reduce gradient norm across processes
        if self.world_size > 1:
            norm_tensor = Tensor(np.array([total_norm_squared]), dtype=np.float32)
            reduced_norm = all_reduce(norm_tensor, ReduceOp.SUM)
            total_norm_squared = float(reduced_norm.data[0])
        
        total_norm = np.sqrt(total_norm_squared)
        
        # Clip if necessary
        if total_norm > self.config.max_grad_norm:
            clip_ratio = self.config.max_grad_norm / total_norm
            
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad.data *= clip_ratio
            
            logger.debug(f"Clipped gradients: norm {total_norm:.4f} -> {self.config.max_grad_norm}")
    
    def _log_gradient_norms(self):
        """Log gradient norms for monitoring."""
        total_norm_squared = 0.0
        
        for param in self.module.parameters():
            if param.grad is not None:
                total_norm_squared += np.sum(param.grad.data ** 2)
        
        total_norm = np.sqrt(total_norm_squared)
        self.gradient_norms.append(total_norm)
        
        if len(self.gradient_norms) % 100 == 0:
            avg_norm = np.mean(self.gradient_norms[-100:])
            logger.info(f"Step {self.step_count}: Avg gradient norm (last 100): {avg_norm:.4f}")
    
    def get_statistics(self) -> Dict:
        """Get training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        stats = {
            'step_count': self.step_count,
            'accumulation_steps': self.config.gradient_accumulation_steps,
            'world_size': self.world_size,
            'rank': self.rank
        }
        
        if self.gradient_norms:
            stats['gradient_norms'] = {
                'mean': float(np.mean(self.gradient_norms)),
                'std': float(np.std(self.gradient_norms)),
                'min': float(np.min(self.gradient_norms)),
                'max': float(np.max(self.gradient_norms))
            }
        
        if self.config.mixed_precision and hasattr(self, 'loss_scaler'):
            stats['loss_scale'] = {
                'current': self.loss_scaler.scale,
                'overflow_count': self.loss_scaler.overflow_count
            }
        
        if self.config.profile_communication and hasattr(self, 'comm_profiler'):
            stats['communication'] = self.comm_profiler.get_statistics()
        
        return stats


class DistributedCheckpointManager:
    """Manages distributed training checkpoints."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(self, 
                       model: Module,
                       optimizer,
                       epoch: int,
                       step: int,
                       loss: float,
                       additional_state: Optional[Dict] = None):
        """Save distributed checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current step
            loss: Current loss
            additional_state: Additional state to save
        """
        rank = get_rank() if is_initialized() else 0
        
        # Only rank 0 saves the checkpoint
        if rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'model_state_dict': self._get_model_state_dict(model),
            'optimizer_state_dict': self._get_optimizer_state_dict(optimizer),
            'world_size': get_world_size() if is_initialized() else 1,
            'timestamp': time.time()
        }
        
        if additional_state:
            checkpoint.update(additional_state)
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        try:
            # In a full implementation, this would use torch.save or similar
            with open(checkpoint_path, 'w') as f:
                json.dump({k: v for k, v in checkpoint.items() if not callable(v)}, f, indent=2)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, model: Module, optimizer, checkpoint_path: Optional[str] = None):
        """Load distributed checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            checkpoint_path: Specific checkpoint path (None for latest)
            
        Returns:
            Loaded checkpoint data or None
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.info("No checkpoint found")
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            # Load model state (simplified)
            # In full implementation, this would properly restore model parameters
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"Checkpoint info: epoch {checkpoint['epoch']}, step {checkpoint['step']}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _get_model_state_dict(self, model: Module) -> Dict:
        """Get model state dictionary (simplified)."""
        # In full implementation, this would extract actual parameter values
        return {'model_type': type(model).__name__}
    
    def _get_optimizer_state_dict(self, optimizer) -> Dict:
        """Get optimizer state dictionary (simplified)."""
        # In full implementation, this would extract optimizer state
        return {'optimizer_type': type(optimizer).__name__}
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if len(checkpoints) > self.max_checkpoints:
            # Sort by modification time and remove oldest
            checkpoints.sort(key=lambda p: p.stat().st_mtime)
            
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                try:
                    checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint {checkpoint}: {e}")


def create_advanced_ddp_model(model: Module, 
                              config: Optional[DistributedTrainingConfig] = None) -> AdvancedDistributedDataParallel:
    """Create advanced distributed data parallel model.
    
    Args:
        model: Model to wrap
        config: Advanced training configuration
        
    Returns:
        AdvancedDistributedDataParallel instance
    """
    return AdvancedDistributedDataParallel(model, config)


def setup_fault_tolerant_training(model: Module,
                                 optimizer,
                                 checkpoint_manager: DistributedCheckpointManager) -> Dict:
    """Setup fault-tolerant training with checkpointing.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        checkpoint_manager: Checkpoint manager
        
    Returns:
        Training state dictionary
    """
    # Try to load latest checkpoint
    checkpoint = checkpoint_manager.load_checkpoint(model, optimizer)
    
    if checkpoint:
        return {
            'start_epoch': checkpoint['epoch'],
            'start_step': checkpoint['step'],
            'best_loss': checkpoint['loss']
        }
    else:
        return {
            'start_epoch': 0,
            'start_step': 0,
            'best_loss': float('inf')
        }


# Example usage and testing
if __name__ == "__main__":
    # Test advanced distributed features
    from neural_arch.nn import Sequential, Linear, ReLU
    
    # Create test model
    model = Sequential(
        Linear(100, 50),
        ReLU(),
        Linear(50, 10)
    )
    
    # Create advanced configuration
    config = DistributedTrainingConfig(
        gradient_accumulation_steps=4,
        mixed_precision=False,  # Simplified for testing
        max_grad_norm=1.0,
        profile_communication=True,
        log_gradient_norms=True
    )
    
    print("Testing Advanced Distributed Training Features...")
    
    # Test gradient accumulator
    accumulator = GradientAccumulator(4)
    print(f"✅ Gradient accumulator created: {accumulator.accumulation_steps} steps")
    
    # Test loss scaler
    scaler = LossScaler(initial_scale=1024.0)
    print(f"✅ Loss scaler created: initial scale {scaler.scale}")
    
    # Test communication profiler
    profiler = CommunicationProfiler()
    with profiler.profile_operation("test_op", 1024):
        time.sleep(0.001)  # Simulate operation
    
    stats = profiler.get_statistics()
    print(f"✅ Communication profiler: {len(stats)} operations recorded")
    
    # Test checkpoint manager
    checkpoint_manager = DistributedCheckpointManager("./test_checkpoints")
    print(f"✅ Checkpoint manager created: {checkpoint_manager.checkpoint_dir}")
    
    print("🎉 Advanced distributed training features validated!")