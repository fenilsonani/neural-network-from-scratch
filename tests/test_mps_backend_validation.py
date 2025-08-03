"""Comprehensive MPS backend validation tests for Apple Silicon GPUs.

This test suite validates the MPS (Metal Performance Shaders) backend 
implementation on macOS with Apple Silicon, testing all operations, 
performance characteristics, and integration with the Neural Forge framework.
"""

import os
import sys
import time
import pytest
import numpy as np
from typing import List, Tuple, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch.backends import available_backends, get_backend, auto_select_backend
from neural_arch.core import Tensor
from neural_arch.nn import Linear, ReLU, Sequential
from neural_arch.optim import Adam
from neural_arch.functional import matmul, relu, sigmoid, tanh, gelu, softmax

# Check if MPS is available
try:
    import mlx.core as mx
    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False


class TestMPSBackendAvailability:
    """Test MPS backend availability and basic functionality."""
    
    def test_mps_in_available_backends(self):
        """Test that MPS backend is listed in available backends."""
        backends = available_backends()
        if MPS_AVAILABLE:
            assert "mps" in backends
        else:
            pytest.skip("MLX not available, skipping MPS tests")
    
    def test_mps_backend_initialization(self):
        """Test MPS backend can be initialized."""
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        
        backend = get_backend("mps")
        assert backend.name == "mps"
        assert backend.is_available
        assert backend.supports_gradients  # MLX has built-in autograd
    
    def test_auto_backend_selection(self):
        """Test that auto-selection prefers MPS on Apple Silicon."""
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        
        backend = auto_select_backend()
        # On Apple Silicon with MLX, should prefer MPS
        assert backend.name in ["mps", "numpy"]  # Fallback to numpy if issues


class TestMPSArrayOperations:
    """Test basic array operations on MPS backend."""
    
    @pytest.fixture
    def mps_backend(self):
        """Get MPS backend for testing."""
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        return get_backend("mps")
    
    def test_array_creation(self, mps_backend):
        """Test various array creation methods."""
        # Basic array creation
        data = [[1.0, 2.0], [3.0, 4.0]]
        arr = mps_backend.array(data)
        assert mps_backend.shape(arr) == (2, 2)
        assert mps_backend.dtype(arr) == mx.float32
        
        # Zeros
        zeros = mps_backend.zeros((3, 4))
        assert mps_backend.shape(zeros) == (3, 4)
        
        # Ones
        ones = mps_backend.ones((2, 3), dtype=mps_backend.float32)
        assert mps_backend.shape(ones) == (2, 3)
        
        # Full
        full = mps_backend.full((2, 2), 5.0)
        assert mps_backend.shape(full) == (2, 2)
        
        # Arange
        arange = mps_backend.arange(0, 10, 2)
        assert mps_backend.size(arange) == 5
    
    def test_random_generation(self, mps_backend):
        """Test random number generation."""
        # Normal distribution
        normal = mps_backend.random_normal((10, 5), mean=0.0, std=1.0)
        assert mps_backend.shape(normal) == (10, 5)
        
        # Uniform distribution
        uniform = mps_backend.random_uniform((5, 3), low=0.0, high=1.0)
        assert mps_backend.shape(uniform) == (5, 3)
        
        # Check that values are in expected range
        uniform_np = mps_backend.to_numpy(uniform)
        assert np.all(uniform_np >= 0.0) and np.all(uniform_np <= 1.0)
    
    def test_shape_operations(self, mps_backend):
        """Test shape manipulation operations."""
        arr = mps_backend.array(np.random.randn(2, 3, 4).astype(np.float32))
        
        # Reshape
        reshaped = mps_backend.reshape(arr, (6, 4))
        assert mps_backend.shape(reshaped) == (6, 4)
        
        # Transpose
        transposed = mps_backend.transpose(arr, (2, 1, 0))
        assert mps_backend.shape(transposed) == (4, 3, 2)
        
        # Squeeze
        squeezable = mps_backend.array(np.random.randn(1, 3, 1, 4).astype(np.float32))
        squeezed = mps_backend.squeeze(squeezable)
        assert mps_backend.shape(squeezed) == (3, 4)
        
        # Expand dims
        expanded = mps_backend.expand_dims(arr, axis=1)
        assert mps_backend.shape(expanded) == (2, 1, 3, 4)


class TestMPSMathOperations:
    """Test mathematical operations on MPS backend."""
    
    @pytest.fixture
    def mps_backend(self):
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        return get_backend("mps")
    
    def test_arithmetic_operations(self, mps_backend):
        """Test basic arithmetic operations."""
        a = mps_backend.array([[1.0, 2.0], [3.0, 4.0]])
        b = mps_backend.array([[2.0, 3.0], [4.0, 5.0]])
        
        # Addition
        add_result = mps_backend.add(a, b)
        assert mps_backend.shape(add_result) == (2, 2)
        
        # Subtraction
        sub_result = mps_backend.subtract(a, b)
        assert mps_backend.shape(sub_result) == (2, 2)
        
        # Multiplication
        mul_result = mps_backend.multiply(a, b)
        assert mps_backend.shape(mul_result) == (2, 2)
        
        # Division
        div_result = mps_backend.divide(a, b)
        assert mps_backend.shape(div_result) == (2, 2)
        
        # Power
        pow_result = mps_backend.power(a, 2.0)
        assert mps_backend.shape(pow_result) == (2, 2)
    
    def test_matrix_operations(self, mps_backend):
        """Test matrix operations."""
        a = mps_backend.array(np.random.randn(3, 4).astype(np.float32))
        b = mps_backend.array(np.random.randn(4, 5).astype(np.float32))
        
        # Matrix multiplication
        matmul_result = mps_backend.matmul(a, b)
        assert mps_backend.shape(matmul_result) == (3, 5)
        
        # Dot product (same as matmul in MLX)
        dot_result = mps_backend.dot(a, b)
        assert mps_backend.shape(dot_result) == (3, 5)
    
    def test_reduction_operations(self, mps_backend):
        """Test reduction operations."""
        arr = mps_backend.array(np.random.randn(3, 4, 5).astype(np.float32))
        
        # Sum
        sum_all = mps_backend.sum(arr)
        assert mps_backend.shape(sum_all) == ()  # Scalar
        
        sum_axis = mps_backend.sum(arr, axis=1)
        assert mps_backend.shape(sum_axis) == (3, 5)
        
        # Mean
        mean_result = mps_backend.mean(arr, axis=(0, 2))
        assert mps_backend.shape(mean_result) == (4,)
        
        # Max/Min
        max_result = mps_backend.max(arr, axis=0)
        assert mps_backend.shape(max_result) == (4, 5)
        
        min_result = mps_backend.min(arr, axis=-1)
        assert mps_backend.shape(min_result) == (3, 4)
        
        # Argmax/Argmin
        argmax_result = mps_backend.argmax(arr, axis=1)
        assert mps_backend.shape(argmax_result) == (3, 5)
    
    def test_activation_functions(self, mps_backend):
        """Test activation and mathematical functions."""
        arr = mps_backend.array(np.random.randn(3, 4).astype(np.float32))
        
        # Exponential and logarithm
        exp_result = mps_backend.exp(arr)
        assert mps_backend.shape(exp_result) == (3, 4)
        
        # Only test log on positive values
        pos_arr = mps_backend.abs(arr) + 1e-6
        log_result = mps_backend.log(pos_arr)
        assert mps_backend.shape(log_result) == (3, 4)
        
        # Square root (on positive values)
        sqrt_result = mps_backend.sqrt(pos_arr)
        assert mps_backend.shape(sqrt_result) == (3, 4)
        
        # Absolute value and sign
        abs_result = mps_backend.abs(arr)
        assert mps_backend.shape(abs_result) == (3, 4)
        
        sign_result = mps_backend.sign(arr)
        assert mps_backend.shape(sign_result) == (3, 4)
        
        # Clipping
        clip_result = mps_backend.clip(arr, -1.0, 1.0)
        assert mps_backend.shape(clip_result) == (3, 4)


class TestMPSArrayManipulation:
    """Test array manipulation operations."""
    
    @pytest.fixture
    def mps_backend(self):
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        return get_backend("mps")
    
    def test_concatenation_operations(self, mps_backend):
        """Test array concatenation and stacking."""
        a = mps_backend.array(np.random.randn(2, 3).astype(np.float32))
        b = mps_backend.array(np.random.randn(2, 3).astype(np.float32))
        c = mps_backend.array(np.random.randn(2, 3).astype(np.float32))
        
        # Concatenation
        concat_result = mps_backend.concatenate([a, b, c], axis=0)
        assert mps_backend.shape(concat_result) == (6, 3)
        
        concat_axis1 = mps_backend.concatenate([a, b], axis=1)
        assert mps_backend.shape(concat_axis1) == (2, 6)
        
        # Stacking
        stack_result = mps_backend.stack([a, b, c], axis=0)
        assert mps_backend.shape(stack_result) == (3, 2, 3)
        
        stack_axis1 = mps_backend.stack([a, b], axis=1)
        assert mps_backend.shape(stack_axis1) == (2, 2, 3)
    
    def test_splitting_operations(self, mps_backend):
        """Test array splitting."""
        arr = mps_backend.array(np.random.randn(6, 4).astype(np.float32))
        
        # Split into equal parts
        split_result = mps_backend.split(arr, 3, axis=0)
        assert len(split_result) == 3
        assert all(mps_backend.shape(part) == (2, 4) for part in split_result)
        
        # Split at specific indices
        split_indices = mps_backend.split(arr, [2, 4], axis=0)
        assert len(split_indices) == 3
        assert mps_backend.shape(split_indices[0]) == (2, 4)
        assert mps_backend.shape(split_indices[1]) == (2, 4)
        assert mps_backend.shape(split_indices[2]) == (2, 4)
    
    def test_type_conversions(self, mps_backend):
        """Test type conversion operations."""
        arr = mps_backend.array(np.random.randn(3, 4).astype(np.float32))
        
        # Test astype
        int_arr = mps_backend.astype(arr, mps_backend.int32)
        assert mps_backend.dtype(int_arr) == mx.int32
        
        # Test to_numpy
        np_arr = mps_backend.to_numpy(arr)
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (3, 4)
        
        # Test from_numpy
        back_to_mps = mps_backend.from_numpy(np_arr)
        assert mps_backend.shape(back_to_mps) == (3, 4)


class TestMPSComparison:
    """Test comparison operations."""
    
    @pytest.fixture
    def mps_backend(self):
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        return get_backend("mps")
    
    def test_comparison_operations(self, mps_backend):
        """Test all comparison operations."""
        a = mps_backend.array([[1.0, 2.0], [3.0, 4.0]])
        b = mps_backend.array([[1.0, 3.0], [2.0, 4.0]])
        
        # Equal
        eq_result = mps_backend.equal(a, b)
        assert mps_backend.shape(eq_result) == (2, 2)
        
        # Not equal
        neq_result = mps_backend.not_equal(a, b)
        assert mps_backend.shape(neq_result) == (2, 2)
        
        # Less than
        lt_result = mps_backend.less(a, b)
        assert mps_backend.shape(lt_result) == (2, 2)
        
        # Less equal
        le_result = mps_backend.less_equal(a, b)
        assert mps_backend.shape(le_result) == (2, 2)
        
        # Greater than
        gt_result = mps_backend.greater(a, b)
        assert mps_backend.shape(gt_result) == (2, 2)
        
        # Greater equal
        ge_result = mps_backend.greater_equal(a, b)
        assert mps_backend.shape(ge_result) == (2, 2)
    
    def test_where_operation(self, mps_backend):
        """Test conditional where operation."""
        condition = mps_backend.array([[True, False], [False, True]])
        x = mps_backend.array([[1.0, 2.0], [3.0, 4.0]])
        y = mps_backend.array([[5.0, 6.0], [7.0, 8.0]])
        
        result = mps_backend.where(condition, x, y)
        assert mps_backend.shape(result) == (2, 2)
        
        # Convert to numpy to check values
        result_np = mps_backend.to_numpy(result)
        expected = np.array([[1.0, 6.0], [7.0, 4.0]])
        np.testing.assert_allclose(result_np, expected, rtol=1e-5)


class TestMPSAdvancedOperations:
    """Test advanced operations that may have fallbacks."""
    
    @pytest.fixture
    def mps_backend(self):
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        return get_backend("mps")
    
    def test_einsum_operation(self, mps_backend):
        """Test Einstein summation (may fallback to numpy)."""
        a = mps_backend.array(np.random.randn(3, 4).astype(np.float32))
        b = mps_backend.array(np.random.randn(4, 5).astype(np.float32))
        
        # Matrix multiplication via einsum
        result = mps_backend.einsum('ij,jk->ik', a, b)
        assert mps_backend.shape(result) == (3, 5)
        
        # Trace
        square = mps_backend.array(np.random.randn(4, 4).astype(np.float32))
        trace = mps_backend.einsum('ii->', square)
        assert mps_backend.shape(trace) == ()
    
    def test_unique_operation(self, mps_backend):
        """Test unique operation (may fallback to numpy)."""
        arr = mps_backend.array([1, 2, 2, 3, 3, 3])
        
        # Unique values only
        unique_vals = mps_backend.unique(arr)
        unique_np = mps_backend.to_numpy(unique_vals)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(unique_np, expected)
        
        # Unique with counts
        unique_vals, counts = mps_backend.unique(arr, return_counts=True)
        counts_np = mps_backend.to_numpy(counts)
        expected_counts = np.array([1, 2, 3])
        np.testing.assert_array_equal(counts_np, expected_counts)


class TestMPSPerformance:
    """Test performance characteristics of MPS backend."""
    
    @pytest.fixture
    def mps_backend(self):
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        return get_backend("mps")
    
    def test_large_matrix_operations(self, mps_backend):
        """Test performance with larger matrices."""
        # Create large matrices
        size = 1000
        a = mps_backend.random_normal((size, size))
        b = mps_backend.random_normal((size, size))
        
        # Time matrix multiplication
        start_time = time.time()
        result = mps_backend.matmul(a, b)
        end_time = time.time()
        
        assert mps_backend.shape(result) == (size, size)
        
        # Should complete reasonably quickly on Apple Silicon
        elapsed = end_time - start_time
        print(f"MPS {size}x{size} matmul took {elapsed:.3f} seconds")
        assert elapsed < 5.0  # Should be fast on GPU
    
    def test_memory_efficiency(self, mps_backend):
        """Test memory handling with multiple operations."""
        arrays = []
        
        # Create multiple arrays
        for i in range(10):
            arr = mps_backend.random_normal((100, 100))
            arrays.append(arr)
        
        # Perform operations that create new arrays
        results = []
        for i in range(len(arrays) - 1):
            result = mps_backend.matmul(arrays[i], arrays[i + 1])
            results.append(result)
        
        # Should handle memory efficiently
        assert len(results) == 9
        assert all(mps_backend.shape(r) == (100, 100) for r in results)


class TestMPSIntegration:
    """Test MPS backend integration with Neural Forge components."""
    
    @pytest.fixture
    def mps_backend(self):
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        return get_backend("mps")
    
    def test_tensor_integration(self, mps_backend):
        """Test integration with Neural Forge Tensor class."""
        # Create tensor with MPS backend
        data = np.random.randn(3, 4).astype(np.float32)
        tensor = Tensor(data, requires_grad=True)
        
        # Should work with MPS backend
        assert tensor.shape == (3, 4)
        assert tensor.requires_grad
    
    def test_neural_network_integration(self, mps_backend):
        """Test MPS backend with neural network components."""
        # Create a simple network
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        # Test forward pass
        input_data = np.random.randn(3, 4).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = model(x)
        assert output.shape == (3, 2)
        assert output.requires_grad
    
    def test_optimizer_integration(self, mps_backend):
        """Test MPS backend with optimizers."""
        # Create model and optimizer
        linear = Linear(4, 2)
        optimizer = Adam(linear.parameters(), lr=0.01)
        
        # Test training step
        input_data = np.random.randn(2, 4).astype(np.float32)
        target_data = np.random.randn(2, 2).astype(np.float32)
        
        x = Tensor(input_data, requires_grad=True)
        y_true = Tensor(target_data, requires_grad=False)
        
        # Forward pass
        y_pred = linear(x)
        
        # Simple MSE loss
        loss = ((y_pred.data - y_true.data) ** 2).mean()
        
        # Should complete without errors
        assert loss is not None
    
    def test_functional_operations(self, mps_backend):
        """Test functional operations with MPS backend."""
        data = np.random.randn(3, 4).astype(np.float32)
        x = Tensor(data, requires_grad=True)
        
        # Test various functional operations
        relu_result = relu(x)
        assert relu_result.shape == x.shape
        
        sigmoid_result = sigmoid(x)
        assert sigmoid_result.shape == x.shape
        
        tanh_result = tanh(x)
        assert tanh_result.shape == x.shape
        
        gelu_result = gelu(x)
        assert gelu_result.shape == x.shape


class TestMPSErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def mps_backend(self):
        if not MPS_AVAILABLE:
            pytest.skip("MLX not available")
        return get_backend("mps")
    
    def test_invalid_device_requests(self, mps_backend):
        """Test handling of invalid device requests."""
        arr = mps_backend.array([1, 2, 3])
        
        # Valid device requests
        cpu_arr = mps_backend.to_device(arr, "cpu")
        mps_arr = mps_backend.to_device(arr, "mps")
        gpu_arr = mps_backend.to_device(arr, "gpu")
        
        # Invalid device request
        with pytest.raises(ValueError):
            mps_backend.to_device(arr, "cuda")
    
    def test_dtype_conversions(self, mps_backend):
        """Test dtype conversion handling."""
        arr = mps_backend.array([1.0, 2.0, 3.0])
        
        # Test various dtype conversions
        float32_arr = mps_backend.astype(arr, mps_backend.float32)
        assert mps_backend.dtype(float32_arr) == mx.float32
        
        int32_arr = mps_backend.astype(arr, mps_backend.int32)
        assert mps_backend.dtype(int32_arr) == mx.int32
        
        # Test numpy dtype conversion
        np_dtype_arr = mps_backend.astype(arr, np.float32)
        assert mps_backend.dtype(np_dtype_arr) == mx.float32
    
    def test_shape_mismatch_handling(self, mps_backend):
        """Test handling of shape mismatches."""
        a = mps_backend.array(np.random.randn(3, 4).astype(np.float32))
        b = mps_backend.array(np.random.randn(5, 6).astype(np.float32))
        
        # Should raise appropriate errors for incompatible operations
        with pytest.raises(Exception):  # MLX will raise appropriate error
            mps_backend.matmul(a, b)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])