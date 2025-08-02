"""Advanced memory pooling system for efficient tensor memory management.

This module implements a sophisticated memory pooling system that reduces
memory allocation overhead and fragmentation by reusing tensor memory.

Key features:
- Automatic memory pool management with size-based allocation
- Memory fragmentation reduction through intelligent pooling
- Device-aware memory management (CPU/GPU)
- Memory usage statistics and monitoring
- Configurable pool sizes and policies
"""

import logging
import threading
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..core.device import Device, DeviceType
from ..core.dtype import DType
from ..core.tensor import Tensor

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""

    size: int
    dtype: DType
    device: Device
    data: Any  # Backend-specific array type
    allocated_time: float
    last_used_time: float
    ref_count: int = 0

    def __post_init__(self):
        self.allocated_time = time.time()
        self.last_used_time = self.allocated_time


class MemoryPool:
    """Memory pool for efficient tensor allocation and reuse."""

    def __init__(self, device: Device, max_pool_size: int = 1024 * 1024 * 1024):  # 1GB default
        """Initialize memory pool for a specific device.

        Args:
            device: Target device for this pool
            max_pool_size: Maximum pool size in bytes
        """
        self.device = device
        self.max_pool_size = max_pool_size
        self.current_pool_size = 0

        # Pool organized by size and dtype for efficient lookup
        # Structure: {dtype: {size: [MemoryBlock, ...]}}
        self.available_blocks: Dict[DType, Dict[int, List[MemoryBlock]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.allocated_blocks: Dict[int, MemoryBlock] = {}  # id(data) -> MemoryBlock

        # Statistics
        self.total_allocations = 0
        self.pool_hits = 0
        self.pool_misses = 0
        self.bytes_allocated = 0
        self.bytes_deallocated = 0
        self.peak_usage = 0

        # Thread safety
        self._lock = threading.RLock()

        # Cleanup tracking
        self._cleanup_threshold = 0.8  # Clean up when 80% full
        self._last_cleanup = time.time()
        self._min_cleanup_interval = 60.0  # Minimum 60 seconds between cleanups

    def allocate(self, shape: Tuple[int, ...], dtype: DType) -> Any:
        """Allocate memory from the pool.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Backend array with allocated memory
        """
        size = int(np.prod(shape)) * dtype.itemsize

        with self._lock:
            self.total_allocations += 1

            # Try to find existing block
            block = self._find_suitable_block(size, dtype)

            if block is not None:
                self.pool_hits += 1
                block.last_used_time = time.time()
                block.ref_count += 1

                # Remove from available blocks
                self.available_blocks[dtype][block.size].remove(block)
                if not self.available_blocks[dtype][block.size]:
                    del self.available_blocks[dtype][block.size]

                # Add to allocated blocks
                self.allocated_blocks[id(block.data)] = block

                # Reshape the data if needed
                reshaped_data = self._reshape_block_data(block.data, shape)
                logger.debug(f"Pool hit: allocated {size} bytes from pool")
                return reshaped_data

            # Pool miss - allocate new memory
            self.pool_misses += 1
            data = self._allocate_new_memory(shape, dtype, size)

            # Create and track new block
            block = MemoryBlock(
                size=size,
                dtype=dtype,
                device=self.device,
                data=data,
                allocated_time=time.time(),
                last_used_time=time.time(),
                ref_count=1,
            )

            self.allocated_blocks[id(data)] = block
            self.current_pool_size += size
            self.bytes_allocated += size
            self.peak_usage = max(self.peak_usage, self.current_pool_size)

            logger.debug(f"Pool miss: allocated {size} bytes, pool size: {self.current_pool_size}")
            return data

    def deallocate(self, data: Any) -> bool:
        """Return memory to the pool.

        Args:
            data: Backend array to deallocate

        Returns:
            True if memory was returned to pool, False if freed
        """
        data_id = id(data)

        with self._lock:
            if data_id not in self.allocated_blocks:
                logger.warning("Attempted to deallocate unknown memory block")
                return False

            block = self.allocated_blocks[data_id]
            block.ref_count -= 1

            if block.ref_count > 0:
                # Still has references, don't deallocate
                return True

            # Remove from allocated blocks
            del self.allocated_blocks[data_id]

            # Check if we should return to pool or free
            if self._should_return_to_pool(block):
                # Return to available pool
                self.available_blocks[block.dtype][block.size].append(block)
                block.last_used_time = time.time()
                logger.debug(f"Returned {block.size} bytes to pool")

                # Trigger cleanup if needed
                if self._should_cleanup():
                    self._cleanup_pool()

                return True
            else:
                # Free the memory
                self._free_memory(block)
                self.current_pool_size -= block.size
                self.bytes_deallocated += block.size
                logger.debug(f"Freed {block.size} bytes, pool size: {self.current_pool_size}")
                return False

    def _find_suitable_block(self, size: int, dtype: DType) -> Optional[MemoryBlock]:
        """Find a suitable memory block from the pool."""
        if dtype not in self.available_blocks:
            return None

        # Look for exact size match first
        if size in self.available_blocks[dtype] and self.available_blocks[dtype][size]:
            return self.available_blocks[dtype][size][0]

        # Look for larger blocks (up to 2x size to avoid waste)
        for available_size in sorted(self.available_blocks[dtype].keys()):
            if available_size >= size and available_size <= size * 2:
                if self.available_blocks[dtype][available_size]:
                    return self.available_blocks[dtype][available_size][0]

        return None

    def _should_return_to_pool(self, block: MemoryBlock) -> bool:
        """Determine if a block should be returned to pool or freed."""
        # Don't return if pool is too full
        if self.current_pool_size >= self.max_pool_size * self._cleanup_threshold:
            return False

        # Don't return very large blocks to avoid memory waste
        if block.size > self.max_pool_size // 10:  # No single block > 10% of pool
            return False

        return True

    def _should_cleanup(self) -> bool:
        """Check if pool cleanup is needed."""
        current_time = time.time()

        # Check if enough time has passed since last cleanup
        if current_time - self._last_cleanup < self._min_cleanup_interval:
            return False

        # Check if pool is getting full
        return self.current_pool_size >= self.max_pool_size * self._cleanup_threshold

    def _cleanup_pool(self):
        """Clean up old unused blocks from the pool."""
        current_time = time.time()
        cleanup_age_threshold = 300.0  # Clean blocks older than 5 minutes
        blocks_cleaned = 0
        bytes_cleaned = 0

        logger.debug("Starting pool cleanup")

        for dtype in list(self.available_blocks.keys()):
            for size in list(self.available_blocks[dtype].keys()):
                blocks_to_remove = []

                for block in self.available_blocks[dtype][size]:
                    if current_time - block.last_used_time > cleanup_age_threshold:
                        blocks_to_remove.append(block)

                # Remove old blocks
                for block in blocks_to_remove:
                    self.available_blocks[dtype][size].remove(block)
                    self._free_memory(block)
                    self.current_pool_size -= block.size
                    self.bytes_deallocated += block.size
                    blocks_cleaned += 1
                    bytes_cleaned += block.size

                # Clean up empty size categories
                if not self.available_blocks[dtype][size]:
                    del self.available_blocks[dtype][size]

            # Clean up empty dtype categories
            if not self.available_blocks[dtype]:
                del self.available_blocks[dtype]

        self._last_cleanup = current_time
        logger.info(
            f"Pool cleanup: removed {blocks_cleaned} blocks, freed {bytes_cleaned / (1024**2):.1f} MB"
        )

    def _allocate_new_memory(self, shape: Tuple[int, ...], dtype: DType, size: int) -> Any:
        """Allocate new memory using the appropriate backend."""
        # Get backend for this device
        if self.device.type == DeviceType.CPU:
            from ..backends import get_backend

            backend = get_backend("numpy")
        elif self.device.type == DeviceType.CUDA:
            from ..backends import get_backend

            backend = get_backend("cuda")
        else:
            from ..backends import get_backend

            backend = get_backend("numpy")  # Default fallback

        # Allocate using backend
        return backend.zeros(shape, dtype=dtype.numpy_dtype)

    def _reshape_block_data(self, data: Any, shape: Tuple[int, ...]) -> Any:
        """Reshape block data to the requested shape."""
        # Check if reshape is possible
        current_size = data.size
        requested_size = int(np.prod(shape))

        if requested_size > current_size:
            raise ValueError(f"Cannot reshape to larger size: {requested_size} > {current_size}")

        # For efficiency, we might need to create a view or slice
        if requested_size == current_size:
            return data.reshape(shape)
        else:
            # Create a view of the needed size
            flat_data = data.flatten()
            return flat_data[:requested_size].reshape(shape)

    def _free_memory(self, block: MemoryBlock):
        """Free memory block."""
        # Let garbage collection handle the memory
        # In a real implementation, this might involve explicit GPU memory freeing
        del block.data

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            hit_rate = self.pool_hits / max(1, self.total_allocations) * 100
            available_blocks_count = sum(
                len(blocks)
                for dtype_blocks in self.available_blocks.values()
                for blocks in dtype_blocks.values()
            )

            return {
                "device": str(self.device),
                "total_allocations": self.total_allocations,
                "pool_hits": self.pool_hits,
                "pool_misses": self.pool_misses,
                "hit_rate_percent": hit_rate,
                "current_pool_size_mb": self.current_pool_size / (1024**2),
                "max_pool_size_mb": self.max_pool_size / (1024**2),
                "peak_usage_mb": self.peak_usage / (1024**2),
                "bytes_allocated_mb": self.bytes_allocated / (1024**2),
                "bytes_deallocated_mb": self.bytes_deallocated / (1024**2),
                "available_blocks": available_blocks_count,
                "allocated_blocks": len(self.allocated_blocks),
            }

    def clear(self):
        """Clear all memory from the pool."""
        with self._lock:
            # Free all available blocks
            for dtype_blocks in self.available_blocks.values():
                for blocks in dtype_blocks.values():
                    for block in blocks:
                        self._free_memory(block)

            self.available_blocks.clear()
            self.current_pool_size = 0
            logger.info("Memory pool cleared")


class GlobalMemoryManager:
    """Global memory manager that coordinates memory pools across devices."""

    def __init__(self):
        self.pools: Dict[Device, MemoryPool] = {}
        self._lock = threading.RLock()
        self.enabled = True

        # Global statistics
        self.total_memory_allocated = 0
        self.total_memory_saved = 0

    def get_pool(self, device: Device) -> MemoryPool:
        """Get or create memory pool for a device."""
        with self._lock:
            if device not in self.pools:
                # Calculate appropriate pool size based on device
                if device.type == DeviceType.CUDA:
                    # For GPU, use larger pool (2GB default)
                    pool_size = 2 * 1024 * 1024 * 1024
                else:
                    # For CPU, use smaller pool (1GB default)
                    pool_size = 1024 * 1024 * 1024

                self.pools[device] = MemoryPool(device, pool_size)
                logger.info(
                    f"Created memory pool for {device} with size {pool_size / (1024**2):.0f} MB"
                )

            return self.pools[device]

    def allocate_tensor_memory(self, shape: Tuple[int, ...], dtype: DType, device: Device) -> Any:
        """Allocate memory for a tensor."""
        if not self.enabled:
            # Direct allocation without pooling
            pool = self.get_pool(device)
            return pool._allocate_new_memory(shape, dtype, int(np.prod(shape)) * dtype.itemsize)

        pool = self.get_pool(device)
        return pool.allocate(shape, dtype)

    def deallocate_tensor_memory(self, data: Any, device: Device) -> bool:
        """Deallocate tensor memory."""
        if not self.enabled:
            return False

        if device in self.pools:
            return self.pools[device].deallocate(data)
        return False

    def enable_pooling(self):
        """Enable memory pooling."""
        self.enabled = True
        logger.info("Memory pooling enabled")

    def disable_pooling(self):
        """Disable memory pooling."""
        self.enabled = False
        logger.info("Memory pooling disabled")

    def clear_all_pools(self):
        """Clear all memory pools."""
        with self._lock:
            for pool in self.pools.values():
                pool.clear()
            logger.info("All memory pools cleared")

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global memory statistics."""
        with self._lock:
            total_stats = {
                "enabled": self.enabled,
                "num_pools": len(self.pools),
                "total_allocations": 0,
                "total_hits": 0,
                "total_misses": 0,
                "total_pool_size_mb": 0,
                "total_peak_usage_mb": 0,
                "pools": {},
            }

            for device, pool in self.pools.items():
                pool_stats = pool.get_statistics()
                total_stats["total_allocations"] += pool_stats["total_allocations"]
                total_stats["total_hits"] += pool_stats["pool_hits"]
                total_stats["total_misses"] += pool_stats["pool_misses"]
                total_stats["total_pool_size_mb"] += pool_stats["current_pool_size_mb"]
                total_stats["total_peak_usage_mb"] += pool_stats["peak_usage_mb"]
                total_stats["pools"][str(device)] = pool_stats

            if total_stats["total_allocations"] > 0:
                total_stats["global_hit_rate_percent"] = (
                    total_stats["total_hits"] / total_stats["total_allocations"] * 100
                )
            else:
                total_stats["global_hit_rate_percent"] = 0.0

            return total_stats


# Global memory manager instance
_global_memory_manager = GlobalMemoryManager()


def get_memory_manager() -> GlobalMemoryManager:
    """Get the global memory manager."""
    return _global_memory_manager


def enable_memory_pooling():
    """Enable global memory pooling."""
    get_memory_manager().enable_pooling()


def disable_memory_pooling():
    """Disable global memory pooling."""
    get_memory_manager().disable_pooling()


def clear_memory_pools():
    """Clear all memory pools."""
    get_memory_manager().clear_all_pools()


def get_memory_statistics() -> Dict[str, Any]:
    """Get global memory statistics."""
    return get_memory_manager().get_global_statistics()


# Integration with Tensor class
def create_tensor_with_pooling(data, requires_grad=False, dtype=None, device=None, name=None):
    """Create a tensor using memory pooling if available.

    This is a helper function that can be used to create tensors with memory pooling.
    In a full implementation, this would be integrated into the Tensor constructor.
    """
    from ..core.device import get_default_device
    from ..core.dtype import get_default_dtype
    from ..core.tensor import Tensor

    # Set defaults
    device = device or get_default_device()
    dtype = dtype or get_default_dtype()

    # Convert data to numpy if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=dtype.numpy_dtype)

    # Try to use memory pooling for the underlying storage
    manager = get_memory_manager()
    if manager.enabled:
        try:
            pooled_data = manager.allocate_tensor_memory(data.shape, dtype, device)
            # Copy data to pooled memory
            pooled_data[:] = data
            data = pooled_data
        except Exception as e:
            logger.warning(f"Failed to use memory pooling: {e}")

    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device, name=name)


# Context manager for memory pooling
from contextlib import contextmanager


@contextmanager
def memory_pool_scope():
    """Context manager for memory pooling operations."""
    manager = get_memory_manager()
    manager.enable_pooling()
    try:
        yield manager
    finally:
        # Don't disable automatically - let user control this
        pass


@contextmanager
def no_memory_pooling():
    """Context manager to temporarily disable memory pooling."""
    manager = get_memory_manager()
    old_state = manager.enabled
    manager.disable_pooling()
    try:
        yield
    finally:
        manager.enabled = old_state
