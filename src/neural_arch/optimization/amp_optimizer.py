"""AMP-aware optimizer wrappers for mixed precision training.

This module provides enhanced optimizer wrappers that integrate seamlessly with
automatic mixed precision training, providing:
- Automatic gradient scaling and unscaling
- Overflow detection and recovery
- Performance monitoring and statistics
- Flexible integration with existing optimizers
"""

import logging
import time
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np

from ..core.base import Optimizer
from ..core.tensor import Tensor
from ..exceptions import OptimizerError
from .grad_scaler import AdvancedGradScaler, ScalerConfig

logger = logging.getLogger(__name__)


class AMPOptimizer:
    """AMP-aware optimizer wrapper that handles gradient scaling automatically.
    
    This wrapper provides automatic mixed precision support for any optimizer,
    handling gradient scaling, overflow detection, and recovery seamlessly.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        scaler: Optional[AdvancedGradScaler] = None,
        enabled: bool = True,
        clip_grad_norm: Optional[float] = None,
        skip_if_nonfinite: bool = True,
    ):
        """Initialize AMP optimizer wrapper.
        
        Args:
            optimizer: Base optimizer to wrap
            scaler: Gradient scaler (creates default if None)
            enabled: Whether AMP is enabled
            clip_grad_norm: Optional gradient clipping threshold
            skip_if_nonfinite: Whether to skip optimizer steps with non-finite gradients
        """
        self.optimizer = optimizer
        self.scaler = scaler or AdvancedGradScaler()
        self.enabled = enabled
        self.clip_grad_norm = clip_grad_norm
        self.skip_if_nonfinite = skip_if_nonfinite
        
        # Statistics
        self._total_steps = 0
        self._successful_steps = 0
        self._skipped_steps = 0
        self._clipped_steps = 0
        
        # Performance tracking
        self._step_times = []
        self._overflow_times = []
        
        logger.info(f"AMP optimizer wrapper initialized: enabled={enabled}, "
                   f"clip_grad_norm={clip_grad_norm}")
    
    def step(self, closure=None) -> bool:
        """Perform optimizer step with automatic mixed precision handling.
        
        Args:
            closure: Optional closure for computing loss
            
        Returns:
            True if step was taken, False if skipped due to overflow
        """
        start_time = time.time()
        self._total_steps += 1
        
        if not self.enabled:
            # Standard optimizer step without AMP
            if closure is not None:
                loss = closure()
            self.optimizer.step()
            self._successful_steps += 1
            step_time = time.time() - start_time
            self._step_times.append(step_time)
            return True
        
        # AMP-enabled step
        step_taken = False
        
        try:
            # Apply gradient clipping if configured
            if self.clip_grad_norm is not None:
                from .grad_scaler import clip_gradients_by_norm
                total_norm = clip_gradients_by_norm(self.optimizer, self.clip_grad_norm)
                if total_norm > self.clip_grad_norm:
                    self._clipped_steps += 1
                    logger.debug(f"Clipped gradients: norm {total_norm:.4f} -> {self.clip_grad_norm}")
            
            # Check for non-finite gradients if configured
            if self.skip_if_nonfinite:
                from .grad_scaler import check_gradients_finite
                all_finite, grad_stats = check_gradients_finite(self.optimizer)
                
                if not all_finite:
                    logger.warning(f"Non-finite gradients detected: {grad_stats}")
                    self._handle_overflow()
                    return False
            
            # Attempt scaler step
            step_taken = self.scaler.step(self.optimizer)
            
            if step_taken:
                self._successful_steps += 1
                logger.debug("Successful AMP optimizer step")
            else:
                self._skipped_steps += 1
                self._handle_overflow()
                logger.debug("Skipped AMP optimizer step due to overflow")
            
        except Exception as e:
            logger.error(f"Error during AMP optimizer step: {e}")
            self._skipped_steps += 1
            self._handle_overflow()
            step_taken = False
        
        # Update scaler
        self.scaler.update()
        
        # Track timing
        step_time = time.time() - start_time
        self._step_times.append(step_time)
        
        # Keep only recent timing data
        if len(self._step_times) > 1000:
            self._step_times = self._step_times[-1000:]
        
        return step_taken
    
    def _handle_overflow(self):
        """Handle gradient overflow by logging and updating statistics."""
        overflow_time = time.time()
        self._overflow_times.append(overflow_time)
        
        # Log overflow frequency if it's becoming a problem
        if len(self._overflow_times) > 10:
            recent_overflows = [t for t in self._overflow_times if overflow_time - t < 60.0]  # Last minute
            if len(recent_overflows) > 5:
                logger.warning(f"Frequent overflows detected: {len(recent_overflows)} in last minute")
        
        # Keep only recent overflow data
        if len(self._overflow_times) > 100:
            self._overflow_times = self._overflow_times[-100:]
    
    def zero_grad(self):
        """Zero gradients in the underlying optimizer."""
        self.optimizer.zero_grad()
    
    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss for mixed precision training.
        
        Args:
            loss: Loss tensor to scale
            
        Returns:
            Scaled loss tensor
        """
        if not self.enabled:
            return loss
        return self.scaler.scale(loss)
    
    def backward(self, loss: Tensor):
        """Perform backward pass with automatic loss scaling.
        
        Args:
            loss: Loss tensor for backward pass
        """
        if self.enabled:
            scaled_loss = self.scale_loss(loss)
            scaled_loss.backward()
        else:
            loss.backward()
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        return self.scaler.get_scale() if self.enabled else 1.0
    
    def get_lr(self) -> float:
        """Get learning rate from underlying optimizer."""
        if hasattr(self.optimizer, 'get_lr'):
            return self.optimizer.get_lr()
        elif hasattr(self.optimizer, 'lr'):
            return self.optimizer.lr
        else:
            return 0.0
    
    def set_lr(self, lr: float):
        """Set learning rate in underlying optimizer."""
        if hasattr(self.optimizer, 'set_lr'):
            self.optimizer.set_lr(lr)
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lr
        else:
            logger.warning("Cannot set learning rate on this optimizer")
    
    def enable_amp(self):
        """Enable automatic mixed precision."""
        self.enabled = True
        self.scaler.enable()
        logger.info("Enabled AMP in optimizer wrapper")
    
    def disable_amp(self):
        """Disable automatic mixed precision."""
        self.enabled = False
        self.scaler.disable()
        logger.info("Disabled AMP in optimizer wrapper")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimizer statistics.
        
        Returns:
            Dictionary with optimizer and AMP statistics
        """
        stats = {
            # Basic step statistics
            "total_steps": self._total_steps,
            "successful_steps": self._successful_steps,
            "skipped_steps": self._skipped_steps,
            "clipped_steps": self._clipped_steps,
            "success_rate": self._successful_steps / max(self._total_steps, 1),
            "skip_rate": self._skipped_steps / max(self._total_steps, 1),
            "clip_rate": self._clipped_steps / max(self._total_steps, 1),
            
            # AMP-specific statistics
            "amp_enabled": self.enabled,
            "current_scale": self.get_scale(),
            "clip_grad_norm": self.clip_grad_norm,
            
            # Performance statistics
            "avg_step_time": np.mean(self._step_times) if self._step_times else 0.0,
            "total_step_time": sum(self._step_times),
            "overflow_frequency": len(self._overflow_times) / max(self._total_steps, 1),
        }
        
        # Add scaler statistics
        if self.enabled:
            scaler_stats = self.scaler.get_statistics()
            stats.update({f"scaler_{k}": v for k, v in scaler_stats.items()})
        
        # Add underlying optimizer statistics if available
        if hasattr(self.optimizer, 'get_statistics'):
            opt_stats = self.optimizer.get_statistics()
            stats.update({f"optimizer_{k}": v for k, v in opt_stats.items()})
        
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing.
        
        Returns:
            State dictionary containing optimizer and scaler state
        """
        state = {
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.enabled else None,
            "enabled": self.enabled,
            "clip_grad_norm": self.clip_grad_norm,
            "skip_if_nonfinite": self.skip_if_nonfinite,
            "statistics": {
                "total_steps": self._total_steps,
                "successful_steps": self._successful_steps,
                "skipped_steps": self._skipped_steps,
                "clipped_steps": self._clipped_steps,
            }
        }
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from checkpoint.
        
        Args:
            state_dict: State dictionary to load
        """
        # Load optimizer state
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        
        # Load scaler state if available
        if state_dict.get("scaler_state") is not None:
            self.scaler.load_state_dict(state_dict["scaler_state"])
        
        # Load configuration
        self.enabled = state_dict.get("enabled", True)
        self.clip_grad_norm = state_dict.get("clip_grad_norm")
        self.skip_if_nonfinite = state_dict.get("skip_if_nonfinite", True)
        
        # Load statistics if available
        if "statistics" in state_dict:
            stats = state_dict["statistics"]
            self._total_steps = stats.get("total_steps", 0)
            self._successful_steps = stats.get("successful_steps", 0)
            self._skipped_steps = stats.get("skipped_steps", 0)
            self._clipped_steps = stats.get("clipped_steps", 0)
        
        logger.info("Loaded AMP optimizer state from checkpoint")
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying optimizer."""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.optimizer, name)
    
    def __repr__(self) -> str:
        """String representation of the AMP optimizer."""
        return (f"AMPOptimizer(optimizer={self.optimizer.__class__.__name__}, "
                f"enabled={self.enabled}, scale={self.get_scale():.0f})")


class AMPOptimizerFactory:
    """Factory for creating AMP-aware optimizers."""
    
    @staticmethod
    def create_amp_optimizer(
        optimizer_class: Type[Optimizer],
        parameters,
        scaler_config: Optional[ScalerConfig] = None,
        amp_config: Optional[Dict[str, Any]] = None,
        **optimizer_kwargs
    ) -> AMPOptimizer:
        """Create an AMP-aware optimizer.
        
        Args:
            optimizer_class: Optimizer class to instantiate
            parameters: Parameters to optimize
            scaler_config: Configuration for gradient scaler
            amp_config: Configuration for AMP wrapper
            **optimizer_kwargs: Arguments for optimizer initialization
            
        Returns:
            AMP-aware optimizer instance
        """
        # Create base optimizer
        optimizer = optimizer_class(parameters, **optimizer_kwargs)
        
        # Create scaler
        scaler = AdvancedGradScaler(scaler_config) if scaler_config else AdvancedGradScaler()
        
        # Create AMP wrapper
        amp_config = amp_config or {}
        amp_optimizer = AMPOptimizer(optimizer=optimizer, scaler=scaler, **amp_config)
        
        logger.info(f"Created AMP optimizer: {optimizer_class.__name__} with AMP wrapper")
        return amp_optimizer
    
    @staticmethod
    def wrap_optimizer(
        optimizer: Optimizer,
        scaler_config: Optional[ScalerConfig] = None,
        **amp_kwargs
    ) -> AMPOptimizer:
        """Wrap an existing optimizer with AMP capabilities.
        
        Args:
            optimizer: Existing optimizer to wrap
            scaler_config: Configuration for gradient scaler
            **amp_kwargs: Arguments for AMP wrapper
            
        Returns:
            AMP-aware optimizer wrapper
        """
        scaler = AdvancedGradScaler(scaler_config) if scaler_config else AdvancedGradScaler()
        amp_optimizer = AMPOptimizer(optimizer=optimizer, scaler=scaler, **amp_kwargs)
        
        logger.info(f"Wrapped {optimizer.__class__.__name__} with AMP capabilities")
        return amp_optimizer


# Context manager for temporary AMP state changes

class AMPContext:
    """Context manager for temporarily changing AMP settings."""
    
    def __init__(self, amp_optimizer: AMPOptimizer, enabled: Optional[bool] = None,
                 clip_grad_norm: Optional[float] = None):
        """Initialize AMP context manager.
        
        Args:
            amp_optimizer: AMP optimizer to modify
            enabled: Temporary AMP enabled state
            clip_grad_norm: Temporary gradient clipping threshold
        """
        self.amp_optimizer = amp_optimizer
        self.temp_enabled = enabled
        self.temp_clip_grad_norm = clip_grad_norm
        
        # Store original state
        self.original_enabled = None
        self.original_clip_grad_norm = None
    
    def __enter__(self):
        """Enter context and apply temporary settings."""
        # Store original state
        self.original_enabled = self.amp_optimizer.enabled
        self.original_clip_grad_norm = self.amp_optimizer.clip_grad_norm
        
        # Apply temporary settings
        if self.temp_enabled is not None:
            if self.temp_enabled:
                self.amp_optimizer.enable_amp()
            else:
                self.amp_optimizer.disable_amp()
        
        if self.temp_clip_grad_norm is not None:
            self.amp_optimizer.clip_grad_norm = self.temp_clip_grad_norm
        
        return self.amp_optimizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original settings."""
        # Restore original state
        if self.original_enabled:
            self.amp_optimizer.enable_amp()
        else:
            self.amp_optimizer.disable_amp()
        
        self.amp_optimizer.clip_grad_norm = self.original_clip_grad_norm


# Utility functions

def create_amp_adam(parameters, lr: float = 0.001, scaler_config: Optional[ScalerConfig] = None,
                   **kwargs) -> AMPOptimizer:
    """Create AMP-aware Adam optimizer.
    
    Args:
        parameters: Parameters to optimize
        lr: Learning rate
        scaler_config: Gradient scaler configuration
        **kwargs: Additional optimizer arguments
        
    Returns:
        AMP-aware Adam optimizer
    """
    from ..optim.adam import Adam
    return AMPOptimizerFactory.create_amp_optimizer(
        Adam, parameters, scaler_config=scaler_config, lr=lr, **kwargs
    )


def create_amp_adamw(parameters, lr: float = 0.001, scaler_config: Optional[ScalerConfig] = None,
                    **kwargs) -> AMPOptimizer:
    """Create AMP-aware AdamW optimizer.
    
    Args:
        parameters: Parameters to optimize
        lr: Learning rate
        scaler_config: Gradient scaler configuration
        **kwargs: Additional optimizer arguments
        
    Returns:
        AMP-aware AdamW optimizer
    """
    from ..optim.adamw import AdamW
    return AMPOptimizerFactory.create_amp_optimizer(
        AdamW, parameters, scaler_config=scaler_config, lr=lr, **kwargs
    )


def create_amp_sgd(parameters, lr: float = 0.01, scaler_config: Optional[ScalerConfig] = None,
                  **kwargs) -> AMPOptimizer:
    """Create AMP-aware SGD optimizer.
    
    Args:
        parameters: Parameters to optimize
        lr: Learning rate
        scaler_config: Gradient scaler configuration
        **kwargs: Additional optimizer arguments
        
    Returns:
        AMP-aware SGD optimizer
    """
    from ..optim.sgd import SGD
    return AMPOptimizerFactory.create_amp_optimizer(
        SGD, parameters, scaler_config=scaler_config, lr=lr, **kwargs
    )


def get_recommended_scaler_config(model_size: str = "medium") -> ScalerConfig:
    """Get recommended scaler configuration based on model size.
    
    Args:
        model_size: Model size category ("small", "medium", "large", "xlarge")
        
    Returns:
        Recommended scaler configuration
    """
    configs = {
        "small": ScalerConfig(
            init_scale=2**15,  # 32768
            growth_interval=1000,
            backoff_factor=0.75,
        ),
        "medium": ScalerConfig(
            init_scale=2**16,  # 65536
            growth_interval=2000,
            backoff_factor=0.5,
        ),
        "large": ScalerConfig(
            init_scale=2**14,  # 16384 - more conservative for large models
            growth_interval=3000,
            backoff_factor=0.25,
            consecutive_success_threshold=3000,
        ),
        "xlarge": ScalerConfig(
            init_scale=2**12,  # 4096 - very conservative for XL models
            growth_interval=5000,
            backoff_factor=0.125,
            consecutive_success_threshold=5000,
            strategy=ScalingStrategy.CONSERVATIVE,
        ),
    }
    
    return configs.get(model_size.lower(), configs["medium"])


# Import necessary components for the scaling strategy
from .grad_scaler import ScalingStrategy