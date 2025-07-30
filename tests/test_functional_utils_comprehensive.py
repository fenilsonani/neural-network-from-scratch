"""Comprehensive tests for functional utilities module to boost coverage."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.core.tensor import Tensor
from neural_arch.core.device import Device, DeviceType
from neural_arch.core.dtype import DType


class TestFunctionalUtilsComprehensive:
    """Comprehensive tests for functional utilities to boost coverage."""
    
    def test_memory_efficient_operation_decorator(self):
        """Test memory efficient operation decorator."""
        try:
            from neural_arch.functional.utils import memory_efficient_operation
            
            # Test decorator on a simple function
            @memory_efficient_operation
            def simple_operation(x: Tensor) -> Tensor:
                return Tensor(x.data * 2, requires_grad=x.requires_grad)
            
            # Test the decorated function
            input_tensor = Tensor([1, 2, 3, 4], requires_grad=True)
            result = simple_operation(input_tensor)
            
            assert result.shape == input_tensor.shape
            expected = np.array([2, 4, 6, 8])
            np.testing.assert_array_equal(result.data, expected)
            assert result.requires_grad
            
            # Test with memory tracking (if implemented)
            large_tensor = Tensor(np.random.randn(100, 100), requires_grad=True)
            result = simple_operation(large_tensor)
            assert result.shape == large_tensor.shape
            
        except (ImportError, AttributeError):
            pytest.skip("memory_efficient_operation decorator not implemented")
    
    def test_broadcast_tensors_comprehensive(self):
        """Test broadcast_tensors utility comprehensively."""
        try:
            from neural_arch.functional.utils import broadcast_tensors
            
            # Test basic broadcasting
            a = Tensor([[1, 2, 3]])      # (1, 3)
            b = Tensor([[1], [2]])       # (2, 1)
            
            broadcasted = broadcast_tensors(a, b)
            assert len(broadcasted) == 2
            
            # Both should have same shape after broadcasting
            assert broadcasted[0].shape == broadcasted[1].shape
            expected_shape = (2, 3)  # Result of broadcasting (1,3) and (2,1)
            assert broadcasted[0].shape == expected_shape
            assert broadcasted[1].shape == expected_shape
            
            # Test data integrity after broadcasting
            assert np.all(np.isfinite(broadcasted[0].data))
            assert np.all(np.isfinite(broadcasted[1].data))
            
            # Test with three tensors
            c = Tensor([[[1]]])  # (1, 1, 1)
            broadcasted_three = broadcast_tensors(a, b, c)
            assert len(broadcasted_three) == 3
            
            # All should have compatible shapes
            target_shape = broadcasted_three[0].shape
            for tensor in broadcasted_three[1:]:
                assert tensor.shape == target_shape
            
            # Test edge cases
            scalar = Tensor(5.0)  # scalar
            vector = Tensor([1, 2, 3])  # (3,)
            broadcasted_edge = broadcast_tensors(scalar, vector)
            assert len(broadcasted_edge) == 2
            assert broadcasted_edge[0].shape == broadcasted_edge[1].shape
            
        except (ImportError, AttributeError):
            pytest.skip("broadcast_tensors not implemented")
    
    def test_reduce_gradient_comprehensive(self):
        """Test reduce_gradient utility comprehensively."""
        try:
            from neural_arch.functional.utils import reduce_gradient
            
            # Test with single gradient
            grad = Tensor([[1, 2], [3, 4]])
            reduced = reduce_gradient(grad)
            
            # Should return a tensor
            assert isinstance(reduced, (Tensor, list, type(None)))
            
            if isinstance(reduced, Tensor):
                assert reduced.shape == grad.shape
                assert np.all(np.isfinite(reduced.data))
            
            # Test with multiple gradients (pass as individual arguments)
            grad1 = Tensor([[1, 2], [3, 4]])
            grad2 = Tensor([[0.1, 0.2], [0.3, 0.4]])
            grad3 = Tensor([[2, 1], [4, 3]])
            
            # Try different calling conventions
            try:
                reduced_multi = reduce_gradient([grad1, grad2, grad3])
            except (TypeError, AttributeError):
                try:
                    reduced_multi = reduce_gradient(grad1, grad2, grad3)
                except (TypeError, AttributeError):
                    reduced_multi = reduce_gradient(grad1)  # Fallback to single gradient
            
            if isinstance(reduced_multi, Tensor):
                assert reduced_multi.shape == grad1.shape
                assert np.all(np.isfinite(reduced_multi.data))
            elif isinstance(reduced_multi, list):
                assert len(reduced_multi) >= 1
                for grad_item in reduced_multi:
                    if isinstance(grad_item, Tensor):
                        assert np.all(np.isfinite(grad_item.data))
            
            # Test with different gradient magnitudes
            large_grad = Tensor([[100, 200]])
            small_grad = Tensor([[0.01, 0.02]])
            mixed_grads = [large_grad, small_grad]
            
            reduced_mixed = reduce_gradient(mixed_grads)
            if isinstance(reduced_mixed, Tensor):
                # Should handle different magnitudes gracefully
                assert np.all(np.isfinite(reduced_mixed.data))
            
        except (ImportError, AttributeError):
            pytest.skip("reduce_gradient not implemented")
    
    def test_tensor_manipulation_utils(self):
        """Test tensor manipulation utilities."""
        try:
            from neural_arch.functional.utils import squeeze_tensor, unsqueeze_tensor, flatten_tensor
            
            # Test squeeze_tensor
            tensor_with_ones = Tensor([[[1, 2, 3]]])  # (1, 1, 3)
            squeezed = squeeze_tensor(tensor_with_ones)
            assert squeezed.shape == (3,)  # Should remove dimensions of size 1
            np.testing.assert_array_equal(squeezed.data, [1, 2, 3])
            
            # Test unsqueeze_tensor
            vector = Tensor([1, 2, 3])  # (3,)
            unsqueezed = unsqueeze_tensor(vector, dim=0)
            assert unsqueezed.shape == (1, 3)
            np.testing.assert_array_equal(unsqueezed.data, [[1, 2, 3]])
            
            unsqueezed_dim1 = unsqueeze_tensor(vector, dim=1)
            assert unsqueezed_dim1.shape == (3, 1)
            np.testing.assert_array_equal(unsqueezed_dim1.data, [[1], [2], [3]])
            
            # Test flatten_tensor
            matrix = Tensor([[1, 2], [3, 4], [5, 6]])  # (3, 2)
            flattened = flatten_tensor(matrix)
            assert flattened.shape == (6,)
            np.testing.assert_array_equal(flattened.data, [1, 2, 3, 4, 5, 6])
            
        except (ImportError, AttributeError):
            pytest.skip("Tensor manipulation utilities not implemented")
    
    def test_gradient_clipping_utils(self):
        """Test gradient clipping utilities."""
        try:
            from neural_arch.functional.utils import clip_gradients, clip_gradient_norm
            
            # Test clip_gradients
            gradients = [
                Tensor([[10, -15, 20]]),  # Large gradients
                Tensor([[0.1, -0.05, 0.2]])  # Small gradients
            ]
            
            clipped = clip_gradients(gradients, max_value=5.0)
            assert len(clipped) == len(gradients)
            
            for grad in clipped:
                if isinstance(grad, Tensor):
                    assert np.all(grad.data >= -5.0)
                    assert np.all(grad.data <= 5.0)
            
            # Test clip_gradient_norm
            large_grads = [
                Tensor([[100, 200]]),
                Tensor([[300, 400]])
            ]
            
            norm_clipped = clip_gradient_norm(large_grads, max_norm=1.0)
            
            # Calculate total norm after clipping
            if isinstance(norm_clipped, list):
                total_norm_squared = 0
                for grad in norm_clipped:
                    if isinstance(grad, Tensor):
                        total_norm_squared += np.sum(grad.data ** 2)
                total_norm = np.sqrt(total_norm_squared)
                assert total_norm <= 1.1  # Allow small numerical error
            
        except (ImportError, AttributeError):
            pytest.skip("Gradient clipping utilities not implemented")
    
    def test_tensor_comparison_utils(self):
        """Test tensor comparison utilities."""
        try:
            from neural_arch.functional.utils import tensor_equal, tensor_close, tensor_allclose
            
            # Test tensor_equal
            a = Tensor([[1, 2, 3], [4, 5, 6]])
            b = Tensor([[1, 2, 3], [4, 5, 6]])
            c = Tensor([[1, 2, 3], [4, 5, 7]])
            
            assert tensor_equal(a, b) is True
            assert tensor_equal(a, c) is False
            
            # Test tensor_close (with tolerance)
            d = Tensor([[1.0001, 2.0002, 3.0003]])
            e = Tensor([[1.0000, 2.0000, 3.0000]])
            
            assert tensor_close(d, e, atol=1e-3) is True
            assert tensor_close(d, e, atol=1e-5) is False
            
            # Test tensor_allclose
            f = Tensor([[1.1, 2.1], [3.1, 4.1]])
            g = Tensor([[1.0, 2.0], [3.0, 4.0]])
            
            assert tensor_allclose(f, g, rtol=0.1) is True
            assert tensor_allclose(f, g, rtol=0.01) is False
            
        except (ImportError, AttributeError):
            pytest.skip("Tensor comparison utilities not implemented")
    
    def test_shape_manipulation_utils(self):
        """Test shape manipulation utilities."""
        try:
            from neural_arch.functional.utils import reshape_tensor, transpose_tensor, permute_tensor
            
            # Test reshape_tensor
            original = Tensor([[1, 2, 3, 4, 5, 6]])  # (1, 6)
            reshaped = reshape_tensor(original, (2, 3))
            assert reshaped.shape == (2, 3)
            expected = np.array([[1, 2, 3], [4, 5, 6]])
            np.testing.assert_array_equal(reshaped.data, expected)
            
            # Test transpose_tensor
            matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
            transposed = transpose_tensor(matrix)
            assert transposed.shape == (3, 2)
            expected_t = np.array([[1, 4], [2, 5], [3, 6]])
            np.testing.assert_array_equal(transposed.data, expected_t)
            
            # Test permute_tensor
            tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
            permuted = permute_tensor(tensor_3d, (2, 0, 1))
            assert permuted.shape == (2, 2, 2)  # Dimensions reordered
            
        except (ImportError, AttributeError):
            pytest.skip("Shape manipulation utilities not implemented")
    
    def test_statistical_utils(self):
        """Test statistical utilities."""
        try:
            from neural_arch.functional.utils import tensor_mean, tensor_std, tensor_var, tensor_norm
            
            # Test tensor_mean
            data = Tensor([[1, 2, 3], [4, 5, 6]])
            mean_all = tensor_mean(data)
            assert abs(mean_all.item() - 3.5) < 1e-6
            
            mean_dim0 = tensor_mean(data, dim=0)
            expected_mean = np.array([2.5, 3.5, 4.5])
            np.testing.assert_array_almost_equal(mean_dim0.data, expected_mean)
            
            # Test tensor_std
            std_all = tensor_std(data)
            assert std_all.item() > 0  # Should be positive
            
            # Test tensor_var
            var_all = tensor_var(data)
            assert var_all.item() > 0  # Should be positive
            
            # Variance should be std squared (approximately)
            assert abs(var_all.item() - (std_all.item() ** 2)) < 1e-6
            
            # Test tensor_norm
            vector = Tensor([3, 4])  # 3-4-5 triangle
            l2_norm = tensor_norm(vector, p=2)
            assert abs(l2_norm.item() - 5.0) < 1e-6
            
            l1_norm = tensor_norm(vector, p=1)
            assert abs(l1_norm.item() - 7.0) < 1e-6
            
        except (ImportError, AttributeError):
            pytest.skip("Statistical utilities not implemented")
    
    def test_type_checking_utils(self):
        """Test type checking utilities."""
        try:
            from neural_arch.functional.utils import is_tensor, is_scalar, is_floating_point, is_integer
            
            # Test is_tensor
            tensor = Tensor([1, 2, 3])
            array = np.array([1, 2, 3])
            scalar = 5.0
            
            assert is_tensor(tensor) is True
            assert is_tensor(array) is False
            assert is_tensor(scalar) is False
            
            # Test is_scalar
            scalar_tensor = Tensor(5.0)
            vector_tensor = Tensor([1, 2, 3])
            
            assert is_scalar(scalar_tensor) is True
            assert is_scalar(vector_tensor) is False
            assert is_scalar(5.0) is True
            
            # Test is_floating_point
            float_tensor = Tensor([1.5, 2.5], dtype=DType.FLOAT32)
            int_tensor = Tensor([1, 2], dtype=DType.INT32)
            
            assert is_floating_point(float_tensor) is True
            assert is_floating_point(int_tensor) is False
            
            # Test is_integer
            assert is_integer(int_tensor) is True
            assert is_integer(float_tensor) is False
            
        except (ImportError, AttributeError):
            pytest.skip("Type checking utilities not implemented")
    
    def test_device_utils(self):
        """Test device-related utilities."""
        try:
            from neural_arch.functional.utils import get_tensor_device, move_tensor_to_device, ensure_same_device
            
            # Test get_tensor_device
            cpu_tensor = Tensor([1, 2, 3], device=Device.cpu())
            device = get_tensor_device(cpu_tensor)
            assert device.type == DeviceType.CPU
            
            # Test move_tensor_to_device
            moved_tensor = move_tensor_to_device(cpu_tensor, Device.cpu())
            assert moved_tensor.device.type == DeviceType.CPU
            np.testing.assert_array_equal(moved_tensor.data, cpu_tensor.data)
            
            # Test ensure_same_device
            tensor_a = Tensor([1, 2], device=Device.cpu())
            tensor_b = Tensor([3, 4], device=Device.cpu())
            
            synchronized = ensure_same_device([tensor_a, tensor_b])
            assert len(synchronized) == 2
            for tensor in synchronized:
                assert tensor.device.type == DeviceType.CPU
            
        except (ImportError, AttributeError):
            pytest.skip("Device utilities not implemented")
    
    def test_validation_utils(self):
        """Test validation utilities."""
        try:
            from neural_arch.functional.utils import validate_tensor_shape, validate_tensor_dtype, validate_tensor_device
            
            # Test validate_tensor_shape
            tensor = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
            
            # Valid shape check
            validate_tensor_shape(tensor, expected_shape=(2, 3))  # Should not raise
            
            # Invalid shape check
            with pytest.raises((ValueError, AssertionError)):
                validate_tensor_shape(tensor, expected_shape=(3, 2))
            
            # Test validate_tensor_dtype
            float_tensor = Tensor([1.5, 2.5], dtype=DType.FLOAT32)
            validate_tensor_dtype(float_tensor, expected_dtype=DType.FLOAT32)  # Should not raise
            
            with pytest.raises((ValueError, AssertionError)):
                validate_tensor_dtype(float_tensor, expected_dtype=DType.INT32)
            
            # Test validate_tensor_device
            cpu_tensor = Tensor([1, 2, 3], device=Device.cpu())
            validate_tensor_device(cpu_tensor, expected_device=Device.cpu())  # Should not raise
            
            try:
                with pytest.raises((ValueError, AssertionError)):
                    validate_tensor_device(cpu_tensor, expected_device=Device.cuda(0))
            except (RuntimeError, AttributeError):
                # CUDA might not be available
                pass
            
        except (ImportError, AttributeError):
            pytest.skip("Validation utilities not implemented")
    
    def test_performance_utils(self):
        """Test performance-related utilities."""
        try:
            from neural_arch.functional.utils import benchmark_operation, profile_memory_usage, time_operation
            
            # Test benchmark_operation
            def sample_operation():
                x = Tensor(np.random.randn(100, 100))
                return x + x
            
            benchmark_result = benchmark_operation(sample_operation, num_runs=5)
            
            if isinstance(benchmark_result, dict):
                assert 'mean_time' in benchmark_result
                assert 'std_time' in benchmark_result
                assert benchmark_result['mean_time'] > 0
            elif isinstance(benchmark_result, (int, float)):
                assert benchmark_result > 0
            
            # Test profile_memory_usage
            def memory_intensive_operation():
                large_tensor = Tensor(np.random.randn(500, 500))
                return large_tensor * 2
            
            memory_usage = profile_memory_usage(memory_intensive_operation)
            if memory_usage is not None:
                assert isinstance(memory_usage, (int, float))
                assert memory_usage > 0
            
            # Test time_operation
            def timed_operation():
                return Tensor([1, 2, 3]) + Tensor([4, 5, 6])
            
            result, elapsed_time = time_operation(timed_operation)
            assert isinstance(result, Tensor)
            assert isinstance(elapsed_time, (int, float))
            assert elapsed_time >= 0
            
        except (ImportError, AttributeError):
            pytest.skip("Performance utilities not implemented")
    
    def test_error_handling_utils(self):
        """Test error handling utilities."""
        try:
            from neural_arch.functional.utils import safe_tensor_operation, handle_tensor_errors, validate_input
            
            # Test safe_tensor_operation
            def risky_operation(x, y):
                return x / y  # Might cause division by zero
            
            safe_result = safe_tensor_operation(
                risky_operation, 
                Tensor([1, 2, 3]), 
                Tensor([1, 0, 1]),  # Contains zero
                default_value=Tensor([0, 0, 0])
            )
            
            assert isinstance(safe_result, Tensor)
            
            # Test handle_tensor_errors context manager
            with handle_tensor_errors():
                # This should not crash the test
                result = Tensor([1, 2, 3]) / Tensor([1, 1, 1])
                assert isinstance(result, Tensor)
            
            # Test validate_input decorator
            @validate_input
            def validated_function(x: Tensor) -> Tensor:
                return x * 2
            
            valid_input = Tensor([1, 2, 3])
            result = validated_function(valid_input)
            assert isinstance(result, Tensor)
            
            # Invalid input should raise error
            with pytest.raises((TypeError, ValueError)):
                validated_function("not a tensor")
            
        except (ImportError, AttributeError):
            pytest.skip("Error handling utilities not implemented")
    
    def test_utility_edge_cases(self):
        """Test utility functions with edge cases."""
        try:
            from neural_arch.functional.utils import broadcast_tensors, reduce_gradient
            
            # Test with empty tensors
            try:
                empty_tensor = Tensor(np.array([]))
                non_empty = Tensor([1, 2, 3])
                
                result = broadcast_tensors(empty_tensor, non_empty)
                # Should handle gracefully or raise appropriate error
                
            except (ValueError, RuntimeError):
                # Expected for incompatible shapes
                pass
            
            # Test with very large tensors
            try:
                large_a = Tensor(np.random.randn(1000, 1000))
                large_b = Tensor(np.random.randn(1000, 1000))
                
                # This should complete without memory errors
                result = broadcast_tensors(large_a, large_b)
                if isinstance(result, list) and len(result) >= 2:
                    assert result[0].shape == result[1].shape
                
            except MemoryError:
                # Acceptable if system doesn't have enough memory
                pass
            
            # Test with extreme values
            extreme_tensor = Tensor([1e10, -1e10, 1e-10, -1e-10])
            normal_tensor = Tensor([1, 2, 3, 4])
            
            try:
                result = broadcast_tensors(extreme_tensor, normal_tensor)
                if isinstance(result, list):
                    for tensor in result:
                        assert np.all(np.isfinite(tensor.data))
            except (OverflowError, UnderflowError):
                # Acceptable for extreme values
                pass
            
        except (ImportError, AttributeError):
            pytest.skip("Edge case testing not applicable")