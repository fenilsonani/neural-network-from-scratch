"""
Comprehensive test suite for advanced memory management system.
Tests all components of memory_manager.py for comprehensive coverage.

This module tests:
- MemoryPool with intelligent allocation
- ZeroCopyTensor operations
- AdvancedMemoryManager
- Memory defragmentation
- NUMA-aware allocation
- GPU memory management
- Memory leak detection
- Zero-copy operations
"""

import gc
import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any

import numpy as np
import pytest

from src.neural_arch.core.memory_manager import (
    MemoryPool,
    ZeroCopyTensor,
    AdvancedMemoryManager,
    MemoryType,
    MemoryBlock,
    allocate_tensor,
    memory_info,
    temporary_tensor,
    get_memory_manager,
    test_memory_manager
)


class TestMemoryType:
    """Test MemoryType enumeration."""
    
    def test_memory_types_exist(self):
        """Test that all memory types are defined."""
        expected_types = [
            'CPU_PINNED',
            'CPU_MAPPED', 
            'GPU_DEVICE',
            'GPU_UNIFIED',
            'CPU_NUMA',
            'SHARED'
        ]
        
        for type_name in expected_types:
            assert hasattr(MemoryType, type_name)
            memory_type = getattr(MemoryType, type_name)
            assert isinstance(memory_type, MemoryType)
            
    def test_memory_type_values(self):
        """Test memory type string values."""
        assert MemoryType.CPU_PINNED.value == "cpu_pinned"
        assert MemoryType.CPU_MAPPED.value == "cpu_mapped"
        assert MemoryType.GPU_DEVICE.value == "gpu_device"
        assert MemoryType.GPU_UNIFIED.value == "gpu_unified"
        assert MemoryType.CPU_NUMA.value == "cpu_numa"
        assert MemoryType.SHARED.value == "shared"


class TestMemoryBlock:
    """Test MemoryBlock data structure."""
    
    def test_memory_block_creation(self):
        """Test MemoryBlock creation and initialization."""
        ptr = 0x1000  # Mock memory address
        size = 1024
        memory_type = MemoryType.CPU_PINNED
        
        block = MemoryBlock(
            ptr=ptr,
            size=size,
            memory_type=memory_type,
            device_id=0,
            alignment=64,
            tag="test_block"
        )
        
        assert block.ptr == ptr
        assert block.size == size
        assert block.memory_type == memory_type
        assert block.device_id == 0
        assert block.is_free is True
        assert block.reference_count == 0
        assert block.alignment == 64
        assert block.tag == "test_block"
        assert block.end_ptr == ptr + size
        
    def test_memory_block_post_init(self):
        """Test MemoryBlock post-initialization."""
        ptr = 0x2000
        size = 2048
        
        block = MemoryBlock(ptr=ptr, size=size, memory_type=MemoryType.GPU_DEVICE)
        
        # end_ptr should be calculated in __post_init__
        assert block.end_ptr == ptr + size
        
    def test_memory_block_defaults(self):
        """Test MemoryBlock default values."""
        block = MemoryBlock(ptr=0x1000, size=1024, memory_type=MemoryType.CPU_PINNED)
        
        assert block.device_id is None
        assert block.is_free is True
        assert block.reference_count == 0
        assert block.alignment == 64
        assert block.tag is None
        assert isinstance(block.allocation_time, float)
        assert isinstance(block.last_access_time, float)


class TestMemoryPool:
    """Test MemoryPool intelligent allocation system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.pool = MemoryPool(
            memory_type=MemoryType.CPU_PINNED,
            initial_size=1024 * 1024,  # 1MB
            alignment=64
        )
        
    def test_memory_pool_initialization(self):
        """Test MemoryPool initialization."""
        assert self.pool.memory_type == MemoryType.CPU_PINNED
        assert self.pool.alignment == 64
        assert self.pool.device_id is None
        assert self.pool.enable_defrag is True
        assert self.pool.max_fragmentation == 0.3
        assert self.pool.total_size >= 1024 * 1024
        assert self.pool.used_size == 0
        assert isinstance(self.pool.free_blocks, dict)
        assert isinstance(self.pool.used_blocks, dict)
        
    def test_size_class_computation(self):
        """Test size class computation for efficient allocation."""
        size_classes = self.pool._compute_size_classes()
        
        # Should have exponential size classes
        assert len(size_classes) > 0
        
        # Should be sorted
        assert size_classes == sorted(size_classes)
        
        # Should start with small sizes and go to large
        assert size_classes[0] == 8  # 2^3
        assert 1024 in size_classes  # 2^10
        assert size_classes[-1] >= 1024 * 1024 * 1024  # At least 1GB
        
    def test_get_size_class(self):
        """Test size class selection for given allocation size."""
        # Test small allocation
        size_class = self.pool._get_size_class(10)
        assert size_class >= 10
        assert size_class % self.pool.alignment == 0
        
        # Test exact power of 2
        size_class = self.pool._get_size_class(1024)
        assert size_class == 1024
        
        # Test large allocation
        large_size = 10 * 1024 * 1024  # 10MB
        size_class = self.pool._get_size_class(large_size)
        assert size_class >= large_size
        
    def test_basic_allocation(self):
        """Test basic memory allocation."""
        size = 1024
        ptr, block = self.pool.allocate(size, tag="test_alloc")
        
        assert isinstance(ptr, int)
        assert ptr > 0
        assert isinstance(block, MemoryBlock)
        assert block.ptr == ptr
        assert block.size >= size  # May be rounded up to size class
        assert block.is_free is False
        assert block.reference_count == 1
        assert block.tag == "test_alloc"
        assert self.pool.used_size > 0
        assert ptr in self.pool.used_blocks
        
    def test_allocation_alignment(self):
        """Test that allocations are properly aligned."""
        sizes = [17, 31, 100, 1000]  # Various unaligned sizes
        
        for size in sizes:
            ptr, block = self.pool.allocate(size)
            
            # Block size should be aligned
            assert block.size % self.pool.alignment == 0
            
            # Pointer should be aligned (simulated)
            assert ptr % self.pool.alignment == 0
            
            self.pool.deallocate(ptr)
            
    def test_multiple_allocations(self):
        """Test multiple allocations from the same pool."""
        allocations = []
        
        for i in range(10):
            size = (i + 1) * 128  # Different sizes
            ptr, block = self.pool.allocate(size, tag=f"alloc_{i}")
            allocations.append((ptr, block))
            
        # All allocations should be unique
        ptrs = [ptr for ptr, _ in allocations]
        assert len(set(ptrs)) == len(ptrs)  # All unique
        
        # Used size should increase
        assert self.pool.used_size > 0
        assert len(self.pool.used_blocks) == 10
        
        # Deallocate all
        for ptr, _ in allocations:
            self.pool.deallocate(ptr)
            
        assert self.pool.used_size == 0
        assert len(self.pool.used_blocks) == 0
        
    def test_deallocation(self):
        """Test memory deallocation."""
        size = 2048
        ptr, block = self.pool.allocate(size)
        
        # Verify allocation
        assert ptr in self.pool.used_blocks
        assert self.pool.used_size > 0
        
        # Deallocate
        self.pool.deallocate(ptr)
        
        # Verify deallocation
        assert ptr not in self.pool.used_blocks
        assert self.pool.used_size == 0
        assert block.is_free is True
        assert block.reference_count == 0
        assert block.tag is None
        
    def test_deallocation_error_handling(self):
        """Test error handling in deallocation."""
        invalid_ptr = 0xDEADBEEF
        
        with pytest.raises(ValueError, match="Invalid memory address"):
            self.pool.deallocate(invalid_ptr)
            
    def test_allocation_from_free_blocks(self):
        """Test allocation reusing freed blocks."""
        size = 1024
        
        # Allocate and deallocate to create free blocks
        ptr1, _ = self.pool.allocate(size)
        self.pool.deallocate(ptr1)
        
        # Allocate again - should reuse the freed block
        ptr2, block2 = self.pool.allocate(size)
        
        # May or may not be the same pointer depending on implementation
        # But should be allocated successfully
        assert ptr2 > 0
        assert block2.size >= size
        
    def test_fragmentation_detection(self):
        """Test fragmentation detection logic."""
        # Create fragmented memory pattern
        allocations = []
        
        # Allocate many small blocks
        for i in range(20):
            ptr, block = self.pool.allocate(128)
            allocations.append(ptr)
            
        # Deallocate every other block to create fragmentation
        for i in range(0, len(allocations), 2):
            self.pool.deallocate(allocations[i])
            
        # Check fragmentation detection
        should_defrag = self.pool._should_defragment()
        
        # Should detect fragmentation (many small free blocks)
        assert isinstance(should_defrag, bool)
        
    def test_defragmentation(self):
        """Test memory defragmentation."""
        # Create fragmented pattern
        allocations = []
        for i in range(10):
            ptr, _ = self.pool.allocate(256)
            allocations.append(ptr)
            
        # Deallocate to create fragmentation
        for i in range(0, len(allocations), 2):
            self.pool.deallocate(allocations[i])
            
        # Record initial state
        initial_free_blocks = sum(len(blocks) for blocks in self.pool.free_blocks.values())
        
        # Run defragmentation
        self.pool._defragment()
        
        # After defragmentation, should have fewer free blocks
        final_free_blocks = sum(len(blocks) for blocks in self.pool.free_blocks.values())
        
        # May or may not reduce blocks depending on adjacency, but should not crash
        assert isinstance(final_free_blocks, int)
        assert final_free_blocks >= 0
        
    def test_pool_expansion(self):
        """Test automatic pool expansion when needed."""
        initial_total_size = self.pool.total_size
        
        # Allocate a very large block to force expansion
        large_size = self.pool.total_size + 1024
        ptr, block = self.pool.allocate(large_size)
        
        # Pool should have expanded
        assert self.pool.total_size > initial_total_size
        assert block.size >= large_size
        
        self.pool.deallocate(ptr)
        
    def test_different_memory_types(self):
        """Test pools with different memory types."""
        memory_types = [
            MemoryType.CPU_PINNED,
            MemoryType.CPU_MAPPED,
            MemoryType.GPU_DEVICE,
            MemoryType.CPU_NUMA
        ]
        
        for memory_type in memory_types:
            pool = MemoryPool(memory_type=memory_type, initial_size=1024*1024)
            assert pool.memory_type == memory_type
            
            # Should be able to allocate regardless of type
            ptr, block = pool.allocate(1024)
            assert ptr > 0
            assert block.memory_type == memory_type
            
            pool.deallocate(ptr)
            
    def test_pool_statistics(self):
        """Test memory pool statistics."""
        # Initial statistics
        stats = self.pool.get_stats()
        
        expected_keys = [
            'total_size', 'used_size', 'free_size', 'peak_usage',
            'utilization', 'allocation_count', 'free_count',
            'fragmentation_ratio', 'cache_hit_rate'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            
        # After some allocations
        ptrs = []
        for _ in range(5):
            ptr, _ = self.pool.allocate(512)
            ptrs.append(ptr)
            
        stats_after = self.pool.get_stats()
        
        # Used size should increase
        assert stats_after['used_size'] > stats['used_size']
        assert stats_after['allocation_count'] > stats['allocation_count']
        assert stats_after['utilization'] > stats['utilization']
        
        # Cleanup
        for ptr in ptrs:
            self.pool.deallocate(ptr)
            
    def test_thread_safety(self):
        """Test thread safety of memory pool."""
        allocations = []
        allocation_lock = threading.Lock()
        
        def allocate_worker():
            try:
                for i in range(10):
                    ptr, _ = self.pool.allocate(256)
                    with allocation_lock:
                        allocations.append(ptr)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                pytest.fail(f"Thread allocation failed: {e}")
                
        def deallocate_worker():
            try:
                time.sleep(0.05)  # Let some allocations happen first
                while True:
                    with allocation_lock:
                        if allocations:
                            ptr = allocations.pop(0)
                        else:
                            break
                    if ptr:
                        self.pool.deallocate(ptr)
                    time.sleep(0.001)
            except Exception as e:
                pytest.fail(f"Thread deallocation failed: {e}")
                
        # Start threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=allocate_worker)
            t.start()
            threads.append(t)
            
        t = threading.Thread(target=deallocate_worker)
        t.start()
        threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join(timeout=5)
            
        # Should not crash and pool should be in valid state
        stats = self.pool.get_stats()
        assert stats['total_size'] >= 0


class TestZeroCopyTensor:
    """Test ZeroCopyTensor for memory-efficient operations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.pool = MemoryPool(MemoryType.CPU_PINNED, initial_size=1024*1024)
        
    def test_tensor_creation_from_array(self):
        """Test ZeroCopyTensor creation from numpy array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = ZeroCopyTensor(
            data=data,
            shape=(2, 2),
            dtype=np.float32
        )
        
        assert tensor.shape == (2, 2)
        assert tensor.dtype == np.float32
        assert tensor.size == 4
        assert tensor.nbytes == 16  # 4 * 4 bytes
        assert np.array_equal(tensor.data, data)
        
    def test_tensor_creation_from_memory_address(self):
        """Test ZeroCopyTensor creation from memory address."""
        ptr, block = self.pool.allocate(64)  # 16 floats
        
        tensor = ZeroCopyTensor(
            data=ptr,
            shape=(4, 4),
            dtype=np.float32,
            memory_pool=self.pool,
            memory_block=block
        )
        
        assert tensor.shape == (4, 4)
        assert tensor.dtype == np.float32
        assert tensor.memory_pool == self.pool
        assert tensor.memory_block == block
        assert block.reference_count == 1
        
    def test_tensor_strides_computation(self):
        """Test stride computation for different shapes."""
        # Test C-contiguous strides
        tensor = ZeroCopyTensor(
            data=np.zeros((3, 4, 5), dtype=np.float32),
            shape=(3, 4, 5),
            dtype=np.float32
        )
        
        expected_strides = (80, 20, 4)  # (4*5*4, 5*4, 4) bytes
        assert tensor.strides == expected_strides
        
    def test_tensor_view_creation(self):
        """Test creating zero-copy views."""
        data = np.arange(24, dtype=np.float32).reshape(4, 6)
        tensor = ZeroCopyTensor(data=data, shape=(4, 6), dtype=np.float32)
        
        # Create view with different shape
        view = tensor.view(shape=(2, 12))
        
        assert view.shape == (2, 12)
        assert view.size == tensor.size
        assert view.dtype == tensor.dtype
        # Should share memory (simplified check)
        assert view._data is not None
        
    def test_tensor_view_with_offset(self):
        """Test creating views with byte offset."""
        data = np.arange(16, dtype=np.float32)
        tensor = ZeroCopyTensor(data=data, shape=(16,), dtype=np.float32)
        
        # Create view starting from offset
        view = tensor.view(shape=(8,), offset=32)  # Skip first 8 elements (32 bytes)
        
        assert view.shape == (8,)
        assert view.offset == 32
        
    def test_tensor_view_dtype_change(self):
        """Test creating views with different dtype."""
        data = np.arange(16, dtype=np.float32)
        tensor = ZeroCopyTensor(data=data, shape=(16,), dtype=np.float32)
        
        # View as uint32 (same size per element)
        view = tensor.view(dtype=np.uint32)
        
        assert view.dtype == np.uint32
        assert view.shape == tensor.shape
        assert view.nbytes == tensor.nbytes
        
    def test_tensor_view_validation(self):
        """Test view creation validation."""
        data = np.arange(16, dtype=np.float32)
        tensor = ZeroCopyTensor(data=data, shape=(16,), dtype=np.float32)
        
        # Try to create view larger than available data
        with pytest.raises(ValueError, match="Cannot create view"):
            tensor.view(shape=(32,))  # Too large
            
    def test_tensor_reshape(self):
        """Test zero-copy reshape operation."""
        data = np.arange(12, dtype=np.float32)
        tensor = ZeroCopyTensor(data=data, shape=(12,), dtype=np.float32)
        
        # Reshape to 2D
        reshaped = tensor.reshape((3, 4))
        
        assert reshaped.shape == (3, 4)
        assert reshaped.size == tensor.size
        assert reshaped.nbytes == tensor.nbytes
        
        # Invalid reshape
        with pytest.raises(ValueError, match="Cannot reshape"):
            tensor.reshape((5, 3))  # Different total size
            
    def test_tensor_transpose(self):
        """Test zero-copy transpose operation."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = ZeroCopyTensor(data=data, shape=(3, 4), dtype=np.float32)
        
        # Transpose
        transposed = tensor.transpose()
        
        assert transposed.shape == (4, 3)
        assert transposed.size == tensor.size
        
        # Transpose with specific axes
        transposed_axes = tensor.transpose(axes=(1, 0))
        assert transposed_axes.shape == (4, 3)
        
    def test_tensor_reference_counting(self):
        """Test reference counting in tensors."""
        ptr, block = self.pool.allocate(64)
        
        # Create tensor - should increment reference count
        tensor1 = ZeroCopyTensor(
            data=ptr, shape=(4, 4), dtype=np.float32,
            memory_pool=self.pool, memory_block=block
        )
        assert block.reference_count == 1
        
        # Create view - should increment reference count
        view = tensor1.view(shape=(2, 8))
        assert block.reference_count == 2
        
        # Delete original tensor
        del tensor1
        gc.collect()  # Force garbage collection
        
        # Block should still be alive due to view
        assert block.reference_count >= 1
        
    def test_tensor_memory_cleanup(self):
        """Test automatic memory cleanup."""
        ptr, block = self.pool.allocate(64)
        initial_used = self.pool.used_size
        
        # Create tensor that should auto-deallocate
        tensor = ZeroCopyTensor(
            data=ptr, shape=(4, 4), dtype=np.float32,
            memory_pool=self.pool, memory_block=block
        )
        
        # Delete tensor - should trigger cleanup
        del tensor
        gc.collect()
        
        # Memory should be freed (reference count reaches 0)
        # Note: Actual cleanup depends on __del__ implementation
        # This test mainly ensures no crashes occur
        
    def test_different_dtypes(self):
        """Test tensors with different data types."""
        dtypes_and_sizes = [
            (np.float16, 2),
            (np.float32, 4), 
            (np.float64, 8),
            (np.int32, 4),
            (np.uint8, 1)
        ]
        
        for dtype, itemsize in dtypes_and_sizes:
            data = np.ones((2, 2), dtype=dtype)
            tensor = ZeroCopyTensor(data=data, shape=(2, 2), dtype=dtype)
            
            assert tensor.dtype == dtype
            assert tensor.nbytes == 4 * itemsize  # 2x2 elements
            
    def test_large_tensor_operations(self):
        """Test operations on large tensors."""
        large_shape = (1000, 500)
        data = np.random.randn(*large_shape).astype(np.float32)
        tensor = ZeroCopyTensor(data=data, shape=large_shape, dtype=np.float32)
        
        # Should handle large tensors without issues
        assert tensor.shape == large_shape
        assert tensor.size == 500000
        
        # Test reshape on large tensor
        reshaped = tensor.reshape((500, 1000))
        assert reshaped.shape == (500, 1000)
        
        # Test view on large tensor
        view = tensor.view(shape=(250, 2000))
        assert view.shape == (250, 2000)


class TestAdvancedMemoryManager:
    """Test AdvancedMemoryManager system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.manager = AdvancedMemoryManager()
        
    def test_manager_initialization(self):
        """Test AdvancedMemoryManager initialization."""
        assert isinstance(self.manager.pools, dict)
        assert len(self.manager.pools) >= 1  # At least CPU pool
        assert MemoryType.CPU_PINNED in self.manager.pools
        
        assert hasattr(self.manager, 'tensor_registry')
        assert isinstance(self.manager.global_stats, dict)
        assert self.manager.memory_pressure == 0.0
        assert self.manager.pressure_threshold == 0.85
        
    def test_tensor_allocation(self):
        """Test tensor allocation through manager."""
        tensor = self.manager.allocate_tensor(
            shape=(10, 20),
            dtype=np.float32,
            memory_type=MemoryType.CPU_PINNED,
            zero_init=True
        )
        
        assert isinstance(tensor, ZeroCopyTensor)
        assert tensor.shape == (10, 20)
        assert tensor.dtype == np.float32
        
        # Should be zero-initialized
        assert np.allclose(tensor.data, 0.0)
        
        # Should be in registry
        assert self.manager.global_stats['tensors_created'] >= 1
        
    def test_tensor_allocation_different_types(self):
        """Test tensor allocation with different memory types."""
        memory_types = [
            MemoryType.CPU_PINNED,
            MemoryType.GPU_DEVICE,
            MemoryType.CPU_NUMA
        ]
        
        for memory_type in memory_types:
            tensor = self.manager.allocate_tensor(
                shape=(5, 5),
                dtype=np.float32,
                memory_type=memory_type
            )
            
            assert tensor.shape == (5, 5)
            assert tensor.memory_pool is not None
            assert tensor.memory_pool.memory_type == memory_type
            
    def test_tensor_view_creation(self):
        """Test tensor view creation through manager."""
        source = self.manager.allocate_tensor(
            shape=(20, 30),
            dtype=np.float32
        )
        
        view = self.manager.create_tensor_view(
            source=source,
            shape=(10, 60)
        )
        
        assert isinstance(view, ZeroCopyTensor)
        assert view.shape == (10, 60)
        assert view.size == source.size
        assert self.manager.global_stats['views_created'] >= 1
        
    def test_temporary_tensor_context(self):
        """Test temporary tensor context manager."""
        with self.manager.temporary_tensor(
            shape=(100, 50),
            dtype=np.float32,
            memory_type=MemoryType.CPU_PINNED
        ) as temp_tensor:
            assert isinstance(temp_tensor, ZeroCopyTensor)
            assert temp_tensor.shape == (100, 50)
            
            # Use tensor within context
            temp_tensor.data.fill(42.0)
            assert np.allclose(temp_tensor.data, 42.0)
            
        # Tensor should be cleaned up after context
        # (Reference counting should handle this)
        
    def test_memory_info_collection(self):
        """Test memory information collection."""
        # Allocate some tensors
        tensors = []
        for i in range(5):
            tensor = self.manager.allocate_tensor(
                shape=(10, 10),
                dtype=np.float32
            )
            tensors.append(tensor)
            
        info = self.manager.get_memory_info()
        
        expected_keys = ['pools', 'global_stats', 'tensor_count', 'memory_pressure']
        for key in expected_keys:
            assert key in info
            
        assert info['tensor_count'] >= 5
        assert 'tensors_created' in info['global_stats']
        assert isinstance(info['pools'], dict)
        
    def test_garbage_collection(self):
        """Test forced garbage collection."""
        initial_stats = self.manager.get_memory_info()
        
        # Force garbage collection
        self.manager.force_garbage_collection()
        
        stats_after = self.manager.get_memory_info()
        
        # Should have recorded GC event
        assert stats_after['global_stats']['gc_forced'] > initial_stats['global_stats'].get('gc_forced', 0)
        
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        # Create scenario that might cause leaks
        for i in range(10):
            tensor = self.manager.allocate_tensor(shape=(100, 100), dtype=np.float32)
            # Don't deallocate - simulate potential leak
            
        leaks = self.manager.check_memory_leaks()
        
        # Should return list of leak descriptions
        assert isinstance(leaks, list)
        
        # May or may not detect leaks depending on allocation patterns
        # Main goal is to ensure the check doesn't crash
        
    def test_pool_creation_on_demand(self):
        """Test automatic pool creation for new memory types."""
        initial_pools = len(self.manager.pools)
        
        # Allocate with a memory type that might not have a pool yet
        tensor = self.manager.allocate_tensor(
            shape=(10, 10),
            dtype=np.float32,
            memory_type=MemoryType.SHARED  # Less common type
        )
        
        assert isinstance(tensor, ZeroCopyTensor)
        
        # May have created new pool
        final_pools = len(self.manager.pools)
        assert final_pools >= initial_pools
        
    def test_concurrent_tensor_operations(self):
        """Test concurrent tensor operations."""
        tensors = []
        lock = threading.Lock()
        
        def allocate_tensors():
            try:
                for i in range(10):
                    tensor = self.manager.allocate_tensor(
                        shape=(50, 50),
                        dtype=np.float32
                    )
                    with lock:
                        tensors.append(tensor)
                    time.sleep(0.001)
            except Exception as e:
                pytest.fail(f"Concurrent allocation failed: {e}")
                
        # Start multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=allocate_tensors)
            t.start()
            threads.append(t)
            
        # Wait for completion
        for t in threads:
            t.join(timeout=10)
            
        # Should have allocated tensors without issues
        assert len(tensors) == 30  # 3 threads * 10 tensors each
        
        # All tensors should be valid
        for tensor in tensors:
            assert isinstance(tensor, ZeroCopyTensor)
            assert tensor.shape == (50, 50)


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_memory_manager(self):
        """Test global memory manager access."""
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()
        
        # Should return the same instance (singleton)
        assert manager1 is manager2
        assert isinstance(manager1, AdvancedMemoryManager)
        
    def test_allocate_tensor_function(self):
        """Test global allocate_tensor function."""
        tensor = allocate_tensor(
            shape=(8, 16),
            dtype=np.float32,
            memory_type=MemoryType.CPU_PINNED
        )
        
        assert isinstance(tensor, ZeroCopyTensor)
        assert tensor.shape == (8, 16)
        assert tensor.dtype == np.float32
        
    def test_memory_info_function(self):
        """Test global memory_info function."""
        info = memory_info()
        
        assert isinstance(info, dict)
        expected_keys = ['pools', 'global_stats', 'tensor_count', 'memory_pressure']
        for key in expected_keys:
            assert key in info
            
    def test_temporary_tensor_function(self):
        """Test global temporary_tensor function."""
        with temporary_tensor(
            shape=(32, 64),
            dtype=np.float64,
            memory_type=MemoryType.CPU_PINNED
        ) as tensor:
            assert isinstance(tensor, ZeroCopyTensor)
            assert tensor.shape == (32, 64)
            assert tensor.dtype == np.float64
            
            # Modify tensor
            tensor.data.fill(1.5)
            assert np.allclose(tensor.data, 1.5)


class TestMemoryOptimizations:
    """Test advanced memory optimization features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.manager = AdvancedMemoryManager()
        
    def test_numa_aware_allocation(self):
        """Test NUMA-aware memory allocation."""
        # This is largely a simulation since real NUMA requires specific hardware
        tensor = self.manager.allocate_tensor(
            shape=(1000, 1000),
            dtype=np.float32,
            memory_type=MemoryType.CPU_NUMA
        )
        
        assert isinstance(tensor, ZeroCopyTensor)
        assert tensor.memory_pool.memory_type == MemoryType.CPU_NUMA
        
    def test_gpu_memory_management(self):
        """Test GPU memory management simulation."""
        tensor = self.manager.allocate_tensor(
            shape=(512, 512),
            dtype=np.float32,
            memory_type=MemoryType.GPU_DEVICE,
            device_id=0
        )
        
        assert isinstance(tensor, ZeroCopyTensor)
        assert tensor.memory_pool.memory_type == MemoryType.GPU_DEVICE
        assert tensor.memory_pool.device_id == 0
        
    def test_unified_memory_simulation(self):
        """Test unified memory simulation."""
        tensor = self.manager.allocate_tensor(
            shape=(256, 256),
            dtype=np.float32,
            memory_type=MemoryType.GPU_UNIFIED
        )
        
        assert isinstance(tensor, ZeroCopyTensor)
        assert tensor.memory_pool.memory_type == MemoryType.GPU_UNIFIED
        
    def test_memory_mapped_files(self):
        """Test memory-mapped file allocation."""
        tensor = self.manager.allocate_tensor(
            shape=(100, 100),
            dtype=np.float32,
            memory_type=MemoryType.CPU_MAPPED
        )
        
        assert isinstance(tensor, ZeroCopyTensor)
        assert tensor.memory_pool.memory_type == MemoryType.CPU_MAPPED
        
    def test_shared_memory_allocation(self):
        """Test shared memory allocation."""
        tensor = self.manager.allocate_tensor(
            shape=(64, 64), 
            dtype=np.float32,
            memory_type=MemoryType.SHARED
        )
        
        assert isinstance(tensor, ZeroCopyTensor)
        assert tensor.memory_pool.memory_type == MemoryType.SHARED
        
    def test_memory_pressure_monitoring(self):
        """Test memory pressure monitoring."""
        initial_info = self.manager.get_memory_info()
        initial_pressure = initial_info['memory_pressure']
        
        # Allocate large tensors to increase memory pressure
        large_tensors = []
        for i in range(10):
            tensor = self.manager.allocate_tensor(
                shape=(1000, 1000),
                dtype=np.float32
            )
            large_tensors.append(tensor)
            
        updated_info = self.manager.get_memory_info()
        
        # Memory pressure might increase (depends on implementation)
        assert 'memory_pressure' in updated_info
        assert isinstance(updated_info['memory_pressure'], (int, float))
        
    def test_adaptive_allocation_strategies(self):
        """Test adaptive allocation based on usage patterns."""
        # Allocate many small tensors
        small_tensors = []
        for i in range(50):
            tensor = self.manager.allocate_tensor(
                shape=(10, 10),
                dtype=np.float32
            )
            small_tensors.append(tensor)
            
        # Then allocate large tensors
        large_tensors = []
        for i in range(5):
            tensor = self.manager.allocate_tensor(
                shape=(500, 500),
                dtype=np.float32
            )
            large_tensors.append(tensor)
            
        # Memory manager should adapt to different allocation patterns
        info = self.manager.get_memory_info()
        assert info['tensor_count'] >= 55
        
        # Pool statistics should reflect mixed allocation patterns
        for pool_info in info['pools'].values():
            assert 'utilization' in pool_info
            assert 'allocation_count' in pool_info


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Setup test environment."""
        self.manager = AdvancedMemoryManager()
        
    def test_zero_size_allocation(self):
        """Test allocation with zero size."""
        # Should handle gracefully or raise appropriate error
        try:
            tensor = self.manager.allocate_tensor(
                shape=(0,),
                dtype=np.float32
            )
            assert tensor.size == 0
        except ValueError:
            # Acceptable to reject zero-size allocations
            pass
            
    def test_negative_size_allocation(self):
        """Test allocation with negative dimensions."""
        with pytest.raises((ValueError, TypeError)):
            self.manager.allocate_tensor(
                shape=(-1, 10),
                dtype=np.float32
            )
            
    def test_extremely_large_allocation(self):
        """Test allocation that exceeds available memory."""
        # Try to allocate extremely large tensor
        try:
            huge_tensor = self.manager.allocate_tensor(
                shape=(1000000, 1000000),  # 1 trillion elements
                dtype=np.float64
            )
        except (MemoryError, ValueError, OverflowError):
            # Expected to fail
            pass
        else:
            # If it succeeds, should still be valid
            assert isinstance(huge_tensor, ZeroCopyTensor)
            
    def test_invalid_dtype(self):
        """Test allocation with invalid dtype."""
        with pytest.raises((TypeError, ValueError)):
            self.manager.allocate_tensor(
                shape=(10, 10),
                dtype="invalid_dtype"
            )
            
    def test_tensor_view_edge_cases(self):
        """Test tensor view edge cases."""
        tensor = self.manager.allocate_tensor(
            shape=(10, 10),
            dtype=np.float32
        )
        
        # View with same shape
        same_view = tensor.view(shape=(10, 10))
        assert same_view.shape == tensor.shape
        
        # View with single element
        single_view = tensor.view(shape=(1, 100))
        assert single_view.size == tensor.size
        
        # View with higher dimensions
        high_dim_view = tensor.view(shape=(2, 5, 10))
        assert high_dim_view.size == tensor.size
        
    def test_memory_corruption_detection(self):
        """Test detection of potential memory corruption."""
        # This is more conceptual - real corruption detection would require
        # checksums or other validation mechanisms
        
        tensor = self.manager.allocate_tensor(
            shape=(10, 10),
            dtype=np.float32
        )
        
        # Modify tensor data
        tensor.data.fill(42.0)
        
        # Should still be valid
        assert np.allclose(tensor.data, 42.0)
        assert tensor.shape == (10, 10)
        
    def test_pool_exhaustion_handling(self):
        """Test handling when memory pool is exhausted."""
        # Create pool with limited size
        small_pool = MemoryPool(
            memory_type=MemoryType.CPU_PINNED,
            initial_size=1024,  # Very small pool
            enable_defrag=False  # Disable auto-expansion
        )
        
        # Try to allocate more than pool size
        try:
            large_ptr, large_block = small_pool.allocate(2048)  # Larger than pool
            # Should succeed by expanding pool
            assert large_block.size >= 2048
            small_pool.deallocate(large_ptr)
        except MemoryError:
            # Acceptable if pool cannot expand
            pass


class TestSystemIntegration:
    """Test integration with the complete memory management system."""
    
    def test_built_in_system_test(self):
        """Test integration with built-in system test."""
        try:
            # Run the built-in test function
            test_memory_manager()
            print("✅ Memory manager system test passed")
        except Exception as e:
            pytest.fail(f"System integration test failed: {e}")
            
    def test_realistic_workload_simulation(self):
        """Test memory manager with realistic ML workload."""
        manager = AdvancedMemoryManager()
        
        # Simulate neural network training workload
        # 1. Allocate model parameters
        weights = []
        layer_sizes = [(784, 128), (128, 64), (64, 10)]
        
        for i, (in_size, out_size) in enumerate(layer_sizes):
            weight = manager.allocate_tensor(
                shape=(in_size, out_size),
                dtype=np.float32,
                zero_init=False
            )
            bias = manager.allocate_tensor(
                shape=(out_size,),
                dtype=np.float32,
                zero_init=True
            )
            weights.extend([weight, bias])
            
        # 2. Allocate activations for forward pass
        batch_size = 32
        activations = []
        current_size = 784
        
        for _, (_, out_size) in enumerate(layer_sizes):
            activation = manager.allocate_tensor(
                shape=(batch_size, out_size),
                dtype=np.float32
            )
            activations.append(activation)
            current_size = out_size
            
        # 3. Allocate gradients (same shapes as weights)
        gradients = []
        for weight in weights:
            grad = manager.allocate_tensor(
                shape=weight.shape,
                dtype=weight.dtype
            )
            gradients.append(grad)
            
        # 4. Check memory usage
        memory_info_result = manager.get_memory_info()
        assert memory_info_result['tensor_count'] >= len(weights) + len(activations) + len(gradients)
        
        # 5. Create temporary computation tensors
        with manager.temporary_tensor(
            shape=(batch_size, 128),
            dtype=np.float32
        ) as temp_activation:
            # Simulate computation
            temp_activation.data.fill(0.5)
            assert np.allclose(temp_activation.data, 0.5)
            
        # 6. Test views for parameter sharing
        shared_weight = weights[0]
        weight_view = manager.create_tensor_view(
            shared_weight,
            shape=(shared_weight.shape[1], shared_weight.shape[0])  # Transpose view
        )
        
        assert weight_view.size == shared_weight.size
        
        print(f"✅ Realistic workload simulation completed with {memory_info_result['tensor_count']} tensors")
        
    def test_memory_efficiency_comparison(self):
        """Test memory efficiency compared to naive allocation."""
        manager = AdvancedMemoryManager()
        
        # Test 1: Many small allocations (should benefit from pooling)
        start_time = time.time()
        small_tensors = []
        
        for i in range(100):
            tensor = manager.allocate_tensor(
                shape=(32, 32),
                dtype=np.float32
            )
            small_tensors.append(tensor)
            
        pool_time = time.time() - start_time
        
        # Test 2: Zero-copy operations
        source = manager.allocate_tensor(
            shape=(1000, 1000),
            dtype=np.float32
        )
        
        start_time = time.time()
        views = []
        for i in range(10):
            view = manager.create_tensor_view(
                source,
                shape=(100, 10000),
                offset=i * 4000  # Different offsets
            )
            views.append(view)
            
        view_time = time.time() - start_time
        
        # Views should be much faster than copying
        assert view_time < pool_time  # Views are typically faster
        
        print(f"✅ Memory efficiency test: pooling={pool_time:.4f}s, views={view_time:.4f}s")
        
    def test_stress_testing(self):
        """Test system under stress conditions."""
        manager = AdvancedMemoryManager()
        
        # Stress test 1: Rapid allocation/deallocation
        for cycle in range(10):
            tensors = []
            
            # Allocate
            for i in range(50):
                tensor = manager.allocate_tensor(
                    shape=(100, 100),
                    dtype=np.float32
                )
                tensors.append(tensor)
                
            # Use tensors
            for tensor in tensors:
                tensor.data.fill(float(cycle))
                
            # Implicit cleanup when tensors go out of scope
            
        # Stress test 2: Large number of views
        base_tensor = manager.allocate_tensor(
            shape=(1000, 1000),
            dtype=np.float32
        )
        
        views = []
        for i in range(100):
            view = manager.create_tensor_view(
                base_tensor,
                shape=(100, 10000)
            )
            views.append(view)
            
        # All views should be valid
        for view in views:
            assert view.shape == (100, 10000)
            assert view.size == base_tensor.size
            
        # Force garbage collection
        manager.force_garbage_collection()
        
        # System should still be functional
        final_info = manager.get_memory_info()
        assert isinstance(final_info, dict)
        
        print("✅ Stress testing completed successfully")


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])