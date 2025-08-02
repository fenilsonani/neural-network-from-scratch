"""Comprehensive tests for distributed communication backends.

This module tests the communication primitives and backends for distributed training,
including NCCL, Gloo, and fallback implementations.
"""

import pytest
import numpy as np
import os
import time
from unittest.mock import patch, MagicMock

from neural_arch.core import Tensor
from neural_arch.core.device import Device, DeviceType
from neural_arch.distributed.communication import (
    CommunicationBackend, NCCLBackend, GlooBackend, ReduceOp,
    init_process_group, destroy_process_group, get_backend,
    get_world_size, get_rank, barrier, all_reduce, all_gather,
    reduce_scatter, broadcast, send, recv, is_initialized, is_available
)


class TestReduceOp:
    """Test ReduceOp enumeration."""
    
    def test_reduce_op_values(self):
        """Test that ReduceOp has correct values."""
        assert ReduceOp.SUM.value == "sum"
        assert ReduceOp.PRODUCT.value == "product"
        assert ReduceOp.MIN.value == "min"
        assert ReduceOp.MAX.value == "max"
        assert ReduceOp.AVERAGE.value == "average"
    
    def test_reduce_op_enumeration(self):
        """Test that all reduce operations are defined."""
        expected_ops = {'SUM', 'PRODUCT', 'MIN', 'MAX', 'AVERAGE'}
        actual_ops = {op.name for op in ReduceOp}
        assert actual_ops == expected_ops


class TestCommunicationBackendAbstract:
    """Test abstract communication backend."""
    
    def test_communication_backend_is_abstract(self):
        """Test that CommunicationBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CommunicationBackend(rank=0, world_size=1)
    
    def test_communication_backend_interface(self):
        """Test that CommunicationBackend defines required abstract methods."""
        required_methods = {
            'init_process_group', 'destroy_process_group', 'all_reduce',
            'all_gather', 'reduce_scatter', 'broadcast', 'barrier',
            'send', 'recv'
        }
        
        # Check that all required methods are defined as abstract
        backend_methods = {method for method in dir(CommunicationBackend)
                          if not method.startswith('_')}
        assert required_methods.issubset(backend_methods)


class TestGlooBackend:
    """Test Gloo communication backend."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rank = 0
        self.world_size = 4
        self.backend = GlooBackend(self.rank, self.world_size)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.backend.initialized:
            self.backend.destroy_process_group()
    
    def test_gloo_backend_creation(self):
        """Test Gloo backend initialization."""
        assert self.backend.rank == self.rank
        assert self.backend.world_size == self.world_size
        assert not self.backend.initialized
        assert self.backend.available
    
    def test_gloo_init_process_group(self):
        """Test Gloo process group initialization."""
        self.backend.init_process_group()
        
        assert self.backend.initialized
        assert self.backend.context is not None
        assert self.backend.context['rank'] == self.rank
        assert self.backend.context['world_size'] == self.world_size
    
    def test_gloo_destroy_process_group(self):
        """Test Gloo process group destruction."""
        self.backend.init_process_group()
        assert self.backend.initialized
        
        self.backend.destroy_process_group()
        assert not self.backend.initialized
        assert self.backend.context is None
    
    def test_gloo_all_reduce_sum(self):
        """Test Gloo all-reduce with SUM operation."""
        self.backend.init_process_group()
        
        # Create test tensor
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        result = self.backend.all_reduce(tensor, ReduceOp.SUM)
        
        # Should multiply by world_size for sum simulation
        expected = data * self.world_size
        np.testing.assert_array_almost_equal(result.data, expected)
        assert result.requires_grad == tensor.requires_grad
    
    def test_gloo_all_reduce_average(self):
        """Test Gloo all-reduce with AVERAGE operation."""
        self.backend.init_process_group()
        
        data = np.array([4.0, 8.0, 12.0], dtype=np.float32)
        tensor = Tensor(data)
        
        result = self.backend.all_reduce(tensor, ReduceOp.AVERAGE)
        
        # Average should keep original values in simulation
        np.testing.assert_array_almost_equal(result.data, data)
    
    def test_gloo_all_reduce_max_min(self):
        """Test Gloo all-reduce with MAX and MIN operations."""
        self.backend.init_process_group()
        
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        # Test MAX
        result_max = self.backend.all_reduce(tensor, ReduceOp.MAX)
        assert result_max.data.shape == data.shape
        
        # Test MIN
        result_min = self.backend.all_reduce(tensor, ReduceOp.MIN)
        assert result_min.data.shape == data.shape
    
    def test_gloo_all_reduce_not_initialized(self):
        """Test all-reduce fails when not initialized."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            self.backend.all_reduce(tensor)
    
    def test_gloo_all_gather(self):
        """Test Gloo all-gather operation."""
        self.backend.init_process_group()
        
        data = np.array([1.0, 2.0], dtype=np.float32)
        tensor = Tensor(data)
        
        results = self.backend.all_gather(tensor)
        
        assert len(results) == self.world_size
        for i, result in enumerate(results):
            assert isinstance(result, Tensor)
            assert result.shape == tensor.shape
            assert result.requires_grad == tensor.requires_grad
    
    def test_gloo_reduce_scatter(self):
        """Test Gloo reduce-scatter operation."""
        self.backend.init_process_group()
        
        # Create tensor divisible by world_size
        data = np.random.randn(self.world_size * 2, 3).astype(np.float32)
        tensor = Tensor(data)
        
        result = self.backend.reduce_scatter(tensor, ReduceOp.SUM)
        
        # Should get chunk for this rank
        expected_shape = (2, 3)  # world_size * 2 / world_size = 2
        assert result.shape == expected_shape
    
    def test_gloo_reduce_scatter_invalid_size(self):
        """Test reduce-scatter with tensor size not divisible by world_size."""
        self.backend.init_process_group()
        
        # Create tensor not divisible by world_size
        data = np.random.randn(5, 3).astype(np.float32)  # 5 not divisible by 4
        tensor = Tensor(data)
        
        with pytest.raises(ValueError, match="not divisible"):
            self.backend.reduce_scatter(tensor)
    
    def test_gloo_broadcast(self):
        """Test Gloo broadcast operation."""
        self.backend.init_process_group()
        
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        result = self.backend.broadcast(tensor, src=0)
        
        np.testing.assert_array_equal(result.data, data)
        assert result.requires_grad == tensor.requires_grad
    
    def test_gloo_barrier(self):
        """Test Gloo barrier operation."""
        self.backend.init_process_group()
        
        start_time = time.time()
        self.backend.barrier()
        elapsed_time = time.time() - start_time
        
        # Should complete quickly (simulated with small delay)
        assert elapsed_time < 1.0
    
    def test_gloo_send_recv(self):
        """Test Gloo point-to-point send/recv operations."""
        self.backend.init_process_group()
        
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        # Send (no-op in simplified implementation)
        self.backend.send(tensor, dst=1, tag=0)
        
        # Recv (returns input tensor in simplified implementation)
        result = self.backend.recv(tensor, src=1, tag=0)
        assert isinstance(result, Tensor)


class TestNCCLBackend:
    """Test NCCL communication backend."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rank = 0
        self.world_size = 2
        self.backend = NCCLBackend(self.rank, self.world_size)
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.backend, 'initialized') and self.backend.initialized:
            self.backend.destroy_process_group()
    
    def test_nccl_backend_creation(self):
        """Test NCCL backend initialization."""
        assert self.backend.rank == self.rank
        assert self.backend.world_size == self.world_size
        assert not self.backend.initialized
        # NCCL availability depends on CuPy installation
    
    def test_nccl_backend_availability_check(self):
        """Test NCCL backend availability detection."""
        # Should detect if CuPy/NCCL is available
        assert isinstance(self.backend.available, bool)
    
    @patch('neural_arch.distributed.communication.NCCLBackend._broadcast_unique_id')
    @patch('neural_arch.distributed.communication.NCCLBackend._receive_unique_id')
    def test_nccl_init_process_group_without_cupy(self, mock_receive, mock_broadcast):
        """Test NCCL initialization fails gracefully without CuPy."""
        if not self.backend.available:
            with pytest.raises(RuntimeError, match="not available"):
                self.backend.init_process_group()
    
    def test_nccl_operations_require_initialization(self):
        """Test NCCL operations fail when not initialized."""
        if not self.backend.available:
            pytest.skip("NCCL not available")
        
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            self.backend.all_reduce(tensor)
    
    def test_nccl_dtype_mapping(self):
        """Test NCCL dtype mapping function."""
        if not self.backend.available:
            pytest.skip("NCCL not available")
        
        # Test that _get_nccl_dtype method exists and handles basic types
        assert hasattr(self.backend, '_get_nccl_dtype')
        # Actual testing requires CuPy/NCCL to be installed


class TestDistributedProcessGroup:
    """Test distributed process group management."""
    
    def teardown_method(self):
        """Clean up after each test."""
        if is_initialized():
            destroy_process_group()
    
    @patch.dict(os.environ, {'WORLD_SIZE': '4', 'RANK': '1'})
    def test_init_process_group_from_env(self):
        """Test process group initialization from environment variables."""
        init_process_group(backend="gloo")
        
        assert is_initialized()
        assert get_world_size() == 4
        assert get_rank() == 1
    
    def test_init_process_group_explicit_params(self):
        """Test process group initialization with explicit parameters."""
        init_process_group(backend="gloo", world_size=2, rank=0)
        
        assert is_initialized()
        assert get_world_size() == 2
        assert get_rank() == 0
    
    def test_init_process_group_auto_backend_selection(self):
        """Test automatic backend selection."""
        with patch('neural_arch.distributed.communication.cp', create=True):
            # Mock CuPy available
            init_process_group(backend="auto", world_size=1, rank=0)
            backend = get_backend()
            # Should select NCCL when CuPy is available
            assert isinstance(backend, (NCCLBackend, GlooBackend))
    
    def test_init_process_group_auto_backend_fallback(self):
        """Test automatic backend selection fallback to Gloo."""
        with patch.dict('sys.modules', {'cupy': None}):
            init_process_group(backend="auto", world_size=1, rank=0)
            backend = get_backend()
            assert isinstance(backend, GlooBackend)
    
    def test_init_process_group_invalid_backend(self):
        """Test initialization with invalid backend."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            init_process_group(backend="invalid", world_size=1, rank=0)
    
    def test_destroy_process_group(self):
        """Test process group destruction."""
        init_process_group(backend="gloo", world_size=1, rank=0)
        assert is_initialized()
        
        destroy_process_group()
        assert not is_initialized()
    
    def test_get_backend_not_initialized(self):
        """Test get_backend fails when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            get_backend()
    
    def test_distributed_operations_not_initialized(self):
        """Test distributed operations fail when not initialized."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            all_reduce(tensor)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            all_gather(tensor)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            broadcast(tensor, src=0)


class TestDistributedOperations:
    """Test high-level distributed operations."""
    
    def setup_method(self):
        """Set up distributed environment for testing."""
        init_process_group(backend="gloo", world_size=2, rank=0)
    
    def teardown_method(self):
        """Clean up distributed environment."""
        destroy_process_group()
    
    def test_all_reduce_interface(self):
        """Test all_reduce function interface."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        result = all_reduce(tensor, ReduceOp.SUM)
        assert isinstance(result, Tensor)
        assert result.shape == tensor.shape
    
    def test_all_gather_interface(self):
        """Test all_gather function interface."""
        data = np.array([1.0, 2.0], dtype=np.float32)
        tensor = Tensor(data)
        
        results = all_gather(tensor)
        assert isinstance(results, list)
        assert len(results) == get_world_size()
        assert all(isinstance(t, Tensor) for t in results)
    
    def test_reduce_scatter_interface(self):
        """Test reduce_scatter function interface."""
        # Create tensor with size divisible by world_size
        world_size = get_world_size()
        data = np.random.randn(world_size * 3, 2).astype(np.float32)
        tensor = Tensor(data)
        
        result = reduce_scatter(tensor, ReduceOp.SUM)
        assert isinstance(result, Tensor)
        assert result.shape[0] == 3  # world_size * 3 / world_size
    
    def test_broadcast_interface(self):
        """Test broadcast function interface."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        result = broadcast(tensor, src=0)
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, tensor.data)
    
    def test_barrier_interface(self):
        """Test barrier function interface."""
        start_time = time.time()
        barrier()
        elapsed_time = time.time() - start_time
        
        # Should complete quickly
        assert elapsed_time < 1.0
    
    def test_send_recv_interface(self):
        """Test send/recv function interfaces."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        
        # Send (should not raise error)
        send(tensor, dst=1, tag=0)
        
        # Recv (should return tensor)
        result = recv(tensor, src=1, tag=0)
        assert isinstance(result, Tensor)
    
    def test_is_available(self):
        """Test is_available function."""
        assert is_available() is True


class TestDistributedTensorOperations:
    """Test distributed operations with various tensor types and shapes."""
    
    def setup_method(self):
        """Set up distributed environment."""
        init_process_group(backend="gloo", world_size=4, rank=0)
    
    def teardown_method(self):
        """Clean up distributed environment."""
        destroy_process_group()
    
    def test_operations_with_different_dtypes(self):
        """Test distributed operations with different data types."""
        dtypes = [np.float32, np.float64, np.int32, np.int64]
        
        for dtype in dtypes:
            data = np.array([1, 2, 3], dtype=dtype)
            tensor = Tensor(data)
            
            # Test all_reduce
            result = all_reduce(tensor, ReduceOp.SUM)
            assert result.dtype == tensor.dtype
            
            # Test all_gather
            results = all_gather(tensor)
            assert all(r.dtype == tensor.dtype for r in results)
    
    def test_operations_with_different_shapes(self):
        """Test distributed operations with different tensor shapes."""
        shapes = [(5,), (3, 4), (2, 3, 4), (1, 2, 3, 4)]
        
        for shape in shapes:
            data = np.random.randn(*shape).astype(np.float32)
            tensor = Tensor(data)
            
            # Test all_reduce
            result = all_reduce(tensor, ReduceOp.SUM)
            assert result.shape == tensor.shape
            
            # Test broadcast
            result = broadcast(tensor, src=0)
            assert result.shape == tensor.shape
    
    def test_operations_with_requires_grad(self):
        """Test distributed operations preserve gradient requirements."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        
        # Test all_reduce preserves requires_grad
        result = all_reduce(tensor, ReduceOp.SUM)
        assert result.requires_grad == tensor.requires_grad
        
        # Test all_gather preserves requires_grad
        results = all_gather(tensor)
        assert all(r.requires_grad == tensor.requires_grad for r in results)
    
    def test_large_tensor_operations(self):
        """Test distributed operations with large tensors."""
        # Test with moderately large tensor
        data = np.random.randn(1000, 100).astype(np.float32)
        tensor = Tensor(data)
        
        # Test all_reduce
        result = all_reduce(tensor, ReduceOp.AVERAGE)
        assert result.shape == tensor.shape
        
        # Test broadcast
        result = broadcast(tensor, src=0)
        assert result.shape == tensor.shape


class TestDistributedErrorHandling:
    """Test error handling in distributed operations."""
    
    def test_operations_with_mismatched_world_size(self):
        """Test operations handle world size mismatches correctly."""
        init_process_group(backend="gloo", world_size=3, rank=0)
        
        try:
            # Create tensor not divisible by world_size for reduce_scatter
            data = np.random.randn(7, 2).astype(np.float32)  # 7 not divisible by 3
            tensor = Tensor(data)
            
            with pytest.raises(ValueError, match="not divisible"):
                reduce_scatter(tensor, ReduceOp.SUM)
        finally:
            destroy_process_group()
    
    def test_invalid_src_rank(self):
        """Test broadcast with invalid source rank."""
        init_process_group(backend="gloo", world_size=2, rank=0)
        
        try:
            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            tensor = Tensor(data)
            
            # Source rank >= world_size should be handled gracefully
            # (actual validation depends on backend implementation)
            result = broadcast(tensor, src=5)
            assert isinstance(result, Tensor)
        finally:
            destroy_process_group()
    
    def test_timeout_handling(self):
        """Test timeout handling in distributed operations."""
        init_process_group(backend="gloo", world_size=1, rank=0, timeout=1)
        
        try:
            # Test barrier with timeout
            barrier(timeout=1)
            # Should complete without error for single process
        finally:
            destroy_process_group()


class TestDistributedEnvironmentVariables:
    """Test distributed initialization with various environment configurations."""
    
    def teardown_method(self):
        """Clean up after each test."""
        if is_initialized():
            destroy_process_group()
    
    @patch.dict(os.environ, {
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '29500',
        'WORLD_SIZE': '4',
        'RANK': '2'
    })
    def test_env_variable_initialization(self):
        """Test initialization using environment variables."""
        init_process_group(backend="gloo", init_method="env://")
        
        assert get_world_size() == 4
        assert get_rank() == 2
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_env_variables(self):
        """Test initialization with missing environment variables."""
        # Should use defaults when env vars are missing
        init_process_group(backend="gloo", world_size=1, rank=0)
        
        assert get_world_size() == 1
        assert get_rank() == 0
    
    def test_explicit_params_override_env(self):
        """Test explicit parameters override environment variables."""
        with patch.dict(os.environ, {'WORLD_SIZE': '8', 'RANK': '4'}):
            init_process_group(backend="gloo", world_size=2, rank=1)
            
            # Explicit params should override env vars
            assert get_world_size() == 2
            assert get_rank() == 1


if __name__ == "__main__":
    pytest.main([__file__])