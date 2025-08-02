"""Data parallel training implementations.

This module provides data parallel training capabilities where the same model
is replicated across multiple devices and gradients are synchronized.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core.device import Device, DeviceType
from ..core.tensor import Tensor
from ..nn.module import Module
from .communication import (
    ReduceOp,
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    get_rank,
    get_world_size,
    is_initialized,
    reduce_scatter,
)

logger = logging.getLogger(__name__)


class DataParallel(Module):
    """Simple data parallel wrapper for single-node multi-GPU training."""

    def __init__(
        self,
        module: Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
    ):
        """Initialize data parallel module.

        Args:
            module: Module to replicate across devices
            device_ids: List of GPU device IDs to use
            output_device: Device to gather outputs on
        """
        super().__init__()
        self.module = module

        # Auto-detect GPUs if not specified
        if device_ids is None:
            try:
                import cupy as cp

                device_ids = list(range(cp.cuda.runtime.getDeviceCount()))
            except:
                device_ids = [0]  # Fallback to single device

        self.device_ids = device_ids
        self.output_device = output_device if output_device is not None else device_ids[0]

        # Create module replicas on each device
        self.replicas = {}
        for device_id in device_ids:
            # In a full implementation, this would copy module to each device
            self.replicas[device_id] = module  # Simplified - use same module

        logger.info(f"DataParallel initialized on devices: {device_ids}")

    def forward(self, *inputs, **kwargs):
        """Forward pass with data parallelism."""
        if len(self.device_ids) == 1:
            # Single device - no parallelism needed
            return self.module(*inputs, **kwargs)

        # Scatter inputs across devices
        scattered_inputs = self._scatter_inputs(inputs, kwargs)

        # Parallel forward on each device
        outputs = []
        for device_id, (device_inputs, device_kwargs) in scattered_inputs.items():
            replica = self.replicas[device_id]
            output = replica(*device_inputs, **device_kwargs)
            outputs.append(output)

        # Gather outputs to output device
        return self._gather_outputs(outputs)

    def _scatter_inputs(self, inputs: tuple, kwargs: dict) -> Dict[int, tuple]:
        """Scatter inputs across devices."""
        batch_size = inputs[0].shape[0] if inputs else 1
        chunk_size = batch_size // len(self.device_ids)

        scattered = {}
        for i, device_id in enumerate(self.device_ids):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(self.device_ids) - 1 else batch_size

            # Slice inputs for this device
            device_inputs = []
            for inp in inputs:
                if isinstance(inp, Tensor):
                    chunk = inp[start_idx:end_idx]
                    # In full implementation, move to specific device
                    device_inputs.append(chunk)
                else:
                    device_inputs.append(inp)

            # Handle kwargs similarly
            device_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, Tensor):
                    device_kwargs[k] = v[start_idx:end_idx]
                else:
                    device_kwargs[k] = v

            scattered[device_id] = (tuple(device_inputs), device_kwargs)

        return scattered

    def _gather_outputs(self, outputs: List[Tensor]) -> Tensor:
        """Gather outputs from all devices."""
        if len(outputs) == 1:
            return outputs[0]

        # Concatenate outputs along batch dimension
        from ..functional import concatenate

        return concatenate(outputs, axis=0)

    def parameters(self) -> List[Tensor]:
        """Get parameters from the base module."""
        return self.module.parameters()


class DistributedDataParallel(Module):
    """Distributed data parallel training across multiple nodes/processes."""

    def __init__(
        self,
        module: Module,
        device: Optional[Device] = None,
        gradient_compression: bool = False,
        bucket_size_mb: int = 25,
        find_unused_parameters: bool = False,
    ):
        """Initialize distributed data parallel module.

        Args:
            module: Module to train with DDP
            device: Device to place module on
            gradient_compression: Enable gradient compression
            bucket_size_mb: Size of gradient buckets for communication
            find_unused_parameters: Find and skip unused parameters
        """
        super().__init__()

        if not is_initialized():
            raise RuntimeError("Distributed process group not initialized")

        self.module = module
        self.device = device
        self.gradient_compression = gradient_compression
        self.bucket_size_mb = bucket_size_mb
        self.find_unused_parameters = find_unused_parameters

        self.world_size = get_world_size()
        self.rank = get_rank()

        # Gradient synchronization state
        self.gradients_ready = False
        self.grad_buckets = []
        self.unused_parameters = set()

        # Register backward hooks for gradient synchronization
        self._register_hooks()

        logger.info(f"DistributedDataParallel initialized: rank {self.rank}/{self.world_size}")

    def forward(self, *inputs, **kwargs):
        """Forward pass with distributed data parallelism."""
        # Mark that we're in a new forward pass
        self.gradients_ready = False
        self.unused_parameters.clear()

        # Run forward pass on local module
        outputs = self.module(*inputs, **kwargs)

        # Register for gradient synchronization if training
        if hasattr(outputs, "requires_grad") and outputs.requires_grad:
            self._prepare_for_backward(outputs)

        return outputs

    def _register_hooks(self):
        """Register backward hooks for gradient synchronization."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad: Tensor) -> Tensor:
        """Hook called when gradients are computed."""
        # Add gradient to bucket for all-reduce
        self._add_to_bucket(grad)
        return grad

    def _add_to_bucket(self, grad: Tensor):
        """Add gradient to communication bucket."""
        # In a full implementation, this would:
        # 1. Add gradient to current bucket
        # 2. Check if bucket is full
        # 3. Launch asynchronous all-reduce if full
        # 4. Handle gradient averaging

        # Simplified implementation: immediate all-reduce
        if self.world_size > 1:
            # All-reduce gradient across all processes
            reduced_grad = all_reduce(grad, ReduceOp.AVERAGE)
            # Copy reduced gradient back
            grad.data[:] = reduced_grad.data

    def _prepare_for_backward(self, outputs: Tensor):
        """Prepare for backward pass synchronization."""
        # In full implementation, this would set up hooks to detect
        # when all gradients are ready for synchronization
        pass

    def sync_gradients(self):
        """Manually synchronize gradients across all processes."""
        if self.world_size <= 1:
            return

        # Synchronize all parameter gradients
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                # All-reduce gradient
                param.grad = all_reduce(param.grad, ReduceOp.AVERAGE)

        self.gradients_ready = True
        logger.debug(f"Gradients synchronized across {self.world_size} processes")

    def parameters(self) -> List[Tensor]:
        """Get parameters from the wrapped module."""
        return self.module.parameters()

    def train(self, mode: bool = True):
        """Set training mode."""
        self.module.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        self.module.eval()
        return self


class DistributedSampler:
    """Distributed sampler that partitions dataset across processes."""

    def __init__(
        self,
        dataset_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """Initialize distributed sampler.

        Args:
            dataset_size: Size of the dataset
            num_replicas: Number of processes (world size)
            rank: Rank of current process
            shuffle: Whether to shuffle indices
            seed: Random seed for shuffling
            drop_last: Whether to drop incomplete batches
        """
        if num_replicas is None:
            if is_initialized():
                num_replicas = get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if is_initialized():
                rank = get_rank()
            else:
                rank = 0

        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate number of samples per replica
        if self.drop_last and dataset_size % self.num_replicas != 0:
            self.num_samples = dataset_size // self.num_replicas
        else:
            self.num_samples = (dataset_size + self.num_replicas - 1) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """Generate indices for this process."""
        if self.shuffle:
            # Generate shuffled indices with epoch-based seed
            generator = np.random.RandomState(self.seed + self.epoch)
            indices = generator.permutation(self.dataset_size).tolist()
        else:
            indices = list(range(self.dataset_size))

        # Pad dataset to be evenly divisible by number of replicas
        if not self.drop_last:
            # Repeat indices to make total_size
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                # Repeat the whole dataset multiple times if needed
                repeat_times = padding_size // len(indices) + 1
                indices += (indices * repeat_times)[:padding_size]
        else:
            # Remove extra samples to make it evenly divisible
            indices = indices[: self.total_size]

        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch


def gather(tensor: Tensor, dst: int = 0, async_op: bool = False) -> Optional[List[Tensor]]:
    """Gather tensors from all processes to destination process.

    Args:
        tensor: Tensor to gather
        dst: Destination rank
        async_op: Whether to perform asynchronously

    Returns:
        List of tensors from all processes (only on dst rank)
    """
    if not is_initialized():
        return [tensor]

    if get_rank() == dst:
        # Destination rank collects from all processes
        gathered = all_gather(tensor)
        return gathered
    else:
        # Other ranks just participate in gather
        all_gather(tensor)
        return None


def scatter(tensor_list: Optional[List[Tensor]], src: int = 0) -> Tensor:
    """Scatter list of tensors from source to all processes.

    Args:
        tensor_list: List of tensors to scatter (only required on src rank)
        src: Source rank

    Returns:
        Scattered tensor for this process
    """
    if not is_initialized():
        return tensor_list[0] if tensor_list else Tensor([])

    if get_rank() == src:
        if tensor_list is None or len(tensor_list) != get_world_size():
            raise ValueError("tensor_list must contain tensors for all processes")

        # Broadcast each tensor to corresponding rank
        # Simplified implementation - in practice would use more efficient scatter
        for rank, tensor in enumerate(tensor_list):
            if rank == src:
                result = tensor
            else:
                broadcast(tensor, src)

        return result
    else:
        # Receive scattered tensor
        # Create dummy tensor to receive broadcast
        dummy = Tensor(np.zeros((1,)))  # Placeholder
        return broadcast(dummy, src)


@contextmanager
def no_sync():
    """Context manager to disable gradient synchronization.

    Useful for accumulating gradients over multiple forward passes
    before synchronizing.
    """
    # In a full implementation, this would temporarily disable
    # gradient synchronization hooks
    yield


def is_distributed_training_available() -> bool:
    """Check if distributed training is available."""
    return is_initialized() and get_world_size() > 1


def get_distributed_info() -> Dict[str, Any]:
    """Get distributed training information."""
    if not is_initialized():
        return {"available": False, "world_size": 1, "rank": 0, "backend": None}

    return {
        "available": True,
        "world_size": get_world_size(),
        "rank": get_rank(),
        "backend": "distributed",  # Simplified
    }
