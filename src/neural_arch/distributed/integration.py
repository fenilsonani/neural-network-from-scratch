"""Easy integration helpers for distributed training with existing models."""

import logging
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager

from ..core.tensor import Tensor
from ..nn.module import Module
from ..optim.base import Optimizer
from ..optimization_config import get_config
from .data_parallel import DistributedDataParallel
from .communication import init_process_group, is_initialized, get_world_size, get_rank

logger = logging.getLogger(__name__)


def auto_distributed_setup(backend: str = "auto") -> bool:
    """Automatically set up distributed training if multiple processes detected.
    
    Args:
        backend: Communication backend ("nccl", "gloo", "auto")
        
    Returns:
        True if distributed training was set up, False otherwise
    """
    try:
        import os
        
        # Check if we're in a distributed environment
        if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            
            if world_size > 1:
                # Auto-select backend
                if backend == "auto":
                    try:
                        import cupy
                        backend = "nccl"  # Use NCCL for GPU
                    except ImportError:
                        backend = "gloo"  # Use Gloo for CPU
                
                # Initialize process group
                init_process_group(backend=backend)
                logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}, backend={backend}")
                return True
                
    except Exception as e:
        logger.warning(f"Failed to set up distributed training: {e}")
    
    return False


def make_distributed(model: Module, auto_setup: bool = True) -> Union[Module, DistributedDataParallel]:
    """Make a model distributed-ready with automatic setup.
    
    Args:
        model: Model to make distributed
        auto_setup: Whether to automatically set up distributed environment
        
    Returns:
        Original model if not distributed, DistributedDataParallel wrapper if distributed
    """
    config = get_config()
    
    # Check if distributed training is enabled in config
    if not config.optimization.enable_distributed:
        return model
    
    # Auto-setup distributed environment if requested
    if auto_setup:
        auto_distributed_setup(backend=config.optimization.distributed_backend)
    
    # Wrap with DDP if distributed is initialized
    if is_initialized():
        logger.info("Wrapping model with DistributedDataParallel")
        return DistributedDataParallel(model)
    
    return model


@contextmanager
def distributed_training_context(model: Module, optimizer: Optimizer, auto_setup: bool = True):
    """Context manager for easy distributed training setup.
    
    Usage:
        with distributed_training_context(model, optimizer) as (ddp_model, ddp_optimizer):
            for batch in dataloader:
                outputs = ddp_model(batch)
                loss = criterion(outputs, targets)
                loss.backward()
                ddp_optimizer.step()
                ddp_optimizer.zero_grad()
    """
    # Set up distributed environment
    is_distributed = False
    if auto_setup:
        is_distributed = auto_distributed_setup()
    else:
        is_distributed = is_initialized()
    
    if is_distributed:
        # Wrap model with DDP
        ddp_model = DistributedDataParallel(model)
        
        # Create distributed-aware optimizer
        # Note: The optimizer will automatically handle gradient synchronization
        # when used with DistributedDataParallel
        ddp_optimizer = optimizer  # Optimizer doesn't need wrapping
        
        logger.info("Entered distributed training context")
        yield ddp_model, ddp_optimizer
        logger.info("Exited distributed training context")
        
    else:
        # No distributed training, return original objects
        yield model, optimizer


class DistributedModelMixin:
    """Mixin to add distributed training capabilities to any model."""
    
    def enable_distributed_training(self, backend: str = "auto"):
        """Enable distributed training for this model."""
        if hasattr(self, '_is_distributed') and self._is_distributed:
            logger.warning("Model is already distributed")
            return self
        
        # Set up distributed environment
        is_distributed = auto_distributed_setup(backend)
        
        if is_distributed:
            # Wrap the model
            self._original_forward = self.forward
            self._ddp_wrapper = DistributedDataParallel(self)
            self.forward = self._ddp_wrapper.forward
            self._is_distributed = True
            logger.info("Enabled distributed training for model")
        
        return self
    
    def disable_distributed_training(self):
        """Disable distributed training and restore original model."""
        if hasattr(self, '_is_distributed') and self._is_distributed:
            self.forward = self._original_forward
            delattr(self, '_ddp_wrapper')
            delattr(self, '_original_forward')
            self._is_distributed = False
            logger.info("Disabled distributed training for model")
        
        return self


def auto_scale_batch_size(base_batch_size: int) -> int:
    """Automatically scale batch size based on number of distributed processes.
    
    Args:
        base_batch_size: Base batch size for single process
        
    Returns:
        Scaled batch size for distributed training
    """
    if is_initialized():
        world_size = get_world_size()
        scaled_batch_size = base_batch_size * world_size
        logger.info(f"Scaled batch size from {base_batch_size} to {scaled_batch_size} for {world_size} processes")
        return scaled_batch_size
    
    return base_batch_size


def auto_scale_learning_rate(base_lr: float, scaling_rule: str = "linear") -> float:
    """Automatically scale learning rate for distributed training.
    
    Args:
        base_lr: Base learning rate for single process
        scaling_rule: Scaling rule ("linear", "sqrt")
        
    Returns:
        Scaled learning rate
    """
    if is_initialized():
        world_size = get_world_size()
        
        if scaling_rule == "linear":
            scaled_lr = base_lr * world_size
        elif scaling_rule == "sqrt":
            scaled_lr = base_lr * (world_size ** 0.5)
        else:
            raise ValueError(f"Unknown scaling rule: {scaling_rule}")
        
        logger.info(f"Scaled learning rate from {base_lr} to {scaled_lr} using {scaling_rule} scaling")
        return scaled_lr
    
    return base_lr


def get_distributed_training_info() -> Dict[str, Any]:
    """Get information about current distributed training setup."""
    if not is_initialized():
        return {
            "distributed": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0
        }
    
    import os
    
    return {
        "distributed": True,
        "world_size": get_world_size(),
        "rank": get_rank(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "backend": os.environ.get("DISTRIBUTED_BACKEND", "unknown")
    }