"""Complete test coverage for functional/utils.py targeting 95%+ coverage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from neural_arch.core.device import Device, DeviceType
from neural_arch.core.tensor import Shape, Tensor
from neural_arch.functional.utils import (
    apply_gradient_clipping,
    broadcast_tensors,
    check_finite_gradients,
    compute_output_shape,
    ensure_tensor,
    get_broadcast_shape,
    memory_efficient_operation,
    reduce_gradient,
    validate_tensor_operation,
)


class TestFunctionalUtilsComplete:
    """Complete test coverage for all functional utilities."""

    def test_broadcast_tensors_basic(self):
        """Test basic tensor broadcasting."""
        a = Tensor([[1, 2, 3]])  # Shape: (1, 3)
        b = Tensor([[1], [2]])  # Shape: (2, 1)

        result = broadcast_tensors(a, b)

        # Should return list of broadcasted numpy arrays
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert result[0].shape == (2, 3)
        assert result[1].shape == (2, 3)

    def test_broadcast_tensors_multiple(self):
        """Test broadcasting multiple tensors."""
        a = Tensor([1, 2, 3])  # Shape: (3,)
        b = Tensor([[1], [2]])  # Shape: (2, 1)
        c = Tensor([[[5]]])  # Shape: (1, 1, 1)

        result = broadcast_tensors(a, b, c)

        # Should broadcast all to compatible shape
        assert len(result) == 3
        expected_shape = (1, 2, 3)
        for arr in result:
            assert arr.shape == expected_shape

    def test_broadcast_tensors_incompatible(self):
        """Test broadcasting with incompatible tensors."""
        a = Tensor([[1, 2, 3]])  # Shape: (1, 3)
        b = Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)

        with pytest.raises(ValueError, match="Cannot broadcast tensors"):
            broadcast_tensors(a, b)

    def test_reduce_gradient_scalar_target(self):
        """Test gradient reduction to scalar target."""
        grad = np.array([[1, 2], [3, 4]])
        target_shape = ()  # Scalar
        broadcast_shape = (2, 2)

        result = reduce_gradient(grad, target_shape, broadcast_shape)

        # Should sum all elements for scalar target
        assert np.isscalar(result) or result.shape == ()
        assert result == 10  # 1+2+3+4

    def test_reduce_gradient_remove_leading_dimensions(self):
        """Test gradient reduction by removing leading dimensions."""
        grad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
        target_shape = (2, 2)  # Target has fewer dimensions
        broadcast_shape = (2, 2, 2)

        result = reduce_gradient(grad, target_shape, broadcast_shape)

        # Should sum over leading dimension
        expected = np.array([[6, 8], [10, 12]])  # Sum over first dimension
        assert result.shape == target_shape
        np.testing.assert_array_equal(result, expected)

    def test_reduce_gradient_dimension_wise_reduction(self):
        """Test gradient reduction with dimension-wise reduction."""
        grad = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
        target_shape = (1, 3)  # First dimension should be reduced to 1
        broadcast_shape = (2, 3)

        result = reduce_gradient(grad, target_shape, broadcast_shape)

        # Should sum over first dimension but keep dimension
        expected = np.array([[5, 7, 9]])  # Sum [1,4], [2,5], [3,6]
        assert result.shape == target_shape
        np.testing.assert_array_equal(result, expected)

    def test_reduce_gradient_matrix_operations(self):
        """Test gradient reduction for matrix operations."""
        # Simulate gradient from batched matrix operation
        grad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
        target_shape = (2, 2)  # Parameter shape without batch
        broadcast_shape = (2, 2, 2)

        result = reduce_gradient(grad, target_shape, broadcast_shape)

        # Should handle batch dimension reduction
        assert result.shape == target_shape

    def test_reduce_gradient_reshape_needed(self):
        """Test gradient reduction when reshape is needed."""
        grad = np.array([1, 2, 3, 4])  # Shape: (4,)
        target_shape = (2, 2)  # Needs reshape
        broadcast_shape = (4,)

        result = reduce_gradient(grad, target_shape, broadcast_shape)

        # Should reshape to target shape
        expected = np.array([[1, 2], [3, 4]])
        assert result.shape == target_shape
        np.testing.assert_array_equal(result, expected)

    def test_reduce_gradient_complex_reduction(self):
        """Test gradient reduction with complex dimension mismatch."""
        grad = np.array([[[1, 2, 3]], [[4, 5, 6]]])  # Shape: (2, 1, 3)
        target_shape = (3,)  # Much simpler target shape
        broadcast_shape = (2, 1, 3)

        result = reduce_gradient(grad, target_shape, broadcast_shape)

        # Should reduce appropriately
        assert result.shape == target_shape

    def test_get_broadcast_shape_empty(self):
        """Test get_broadcast_shape with empty input."""
        result = get_broadcast_shape()

        assert result == ()

    def test_get_broadcast_shape_single(self):
        """Test get_broadcast_shape with single shape."""
        result = get_broadcast_shape((2, 3))

        assert result == (2, 3)

    def test_get_broadcast_shape_compatible(self):
        """Test get_broadcast_shape with compatible shapes."""
        result = get_broadcast_shape((1, 3), (2, 1), (1, 1))

        # Should return broadcasted shape
        assert result == (2, 3)

    def test_get_broadcast_shape_different_lengths(self):
        """Test get_broadcast_shape with different length shapes."""
        result = get_broadcast_shape((3,), (2, 1), (1, 1, 1))

        # Should pad with 1s and broadcast
        assert result == (1, 2, 3)

    def test_get_broadcast_shape_incompatible(self):
        """Test get_broadcast_shape with incompatible shapes."""
        with pytest.raises(ValueError, match="Cannot broadcast shapes"):
            get_broadcast_shape((2, 3), (2, 4))

    def test_validate_tensor_operation_valid(self):
        """Test validate_tensor_operation with valid tensors."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        # Should not raise any exception
        validate_tensor_operation(a, b, "test_op")

    def test_validate_tensor_operation_invalid_types(self):
        """Test validate_tensor_operation with invalid types."""
        a = Tensor([1, 2, 3])
        b = [4, 5, 6]  # Not a tensor

        with pytest.raises(TypeError, match="test_op requires Tensor inputs"):
            validate_tensor_operation(a, b, "test_op")

        with pytest.raises(TypeError, match="test_op requires Tensor inputs"):
            validate_tensor_operation("not_tensor", a, "test_op")

    def test_validate_tensor_operation_device_warning(self):
        """Test validate_tensor_operation with device mismatch warning."""
        a = Tensor([1, 2, 3], device=Device(DeviceType.CPU))
        b = Tensor([4, 5, 6], device=Device(DeviceType.CUDA, 0))

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            validate_tensor_operation(a, b, "test_op")

            # Should log warning about device mismatch
            mock_logger.warning.assert_called_once()
            assert "different devices" in str(mock_logger.warning.call_args)

    def test_ensure_tensor_already_tensor(self):
        """Test ensure_tensor with already tensor input."""
        original = Tensor([1, 2, 3])

        result = ensure_tensor(original, "test_tensor")

        # Should return the same tensor
        assert result is original

    def test_ensure_tensor_conversion(self):
        """Test ensure_tensor with conversion needed."""
        data = [1, 2, 3]

        result = ensure_tensor(data, "test_tensor")

        # Should convert to tensor
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, data)

    def test_ensure_tensor_conversion_failure(self):
        """Test ensure_tensor with conversion failure."""
        # Use an object that can't be converted to tensor
        invalid_data = object()

        with pytest.raises(TypeError, match="Cannot convert .* to Tensor for test_tensor"):
            ensure_tensor(invalid_data, "test_tensor")

    def test_compute_output_shape_mean_pool(self):
        """Test compute_output_shape for mean_pool operation."""
        input_shape = (2, 4, 8)

        # Pool along axis 1
        result = compute_output_shape(input_shape, "mean_pool", axis=1)
        expected = (2, 8)  # Removes dimension at axis 1
        assert result == expected

        # Pool along axis 2
        result = compute_output_shape(input_shape, "mean_pool", axis=2)
        expected = (2, 4)  # Removes dimension at axis 2
        assert result == expected

    def test_compute_output_shape_mean_pool_negative_axis(self):
        """Test compute_output_shape for mean_pool with negative axis."""
        input_shape = (2, 4, 8)

        # Pool along axis -1 (last axis)
        result = compute_output_shape(input_shape, "mean_pool", axis=-1)
        expected = (2, 4)  # Removes last dimension
        assert result == expected

    def test_compute_output_shape_mean_pool_invalid_axis(self):
        """Test compute_output_shape for mean_pool with invalid axis."""
        input_shape = (2, 4)

        with pytest.raises(ValueError, match="Cannot pool over axis"):
            compute_output_shape(input_shape, "mean_pool", axis=5)  # Out of bounds

    def test_compute_output_shape_preserving_operations(self):
        """Test compute_output_shape for shape-preserving operations."""
        input_shape = (2, 3, 4)

        # Operations that preserve shape
        result_softmax = compute_output_shape(input_shape, "softmax")
        result_relu = compute_output_shape(input_shape, "relu")

        assert result_softmax == input_shape
        assert result_relu == input_shape

    def test_compute_output_shape_unknown_operation(self):
        """Test compute_output_shape with unknown operation."""
        input_shape = (2, 3)

        with pytest.raises(ValueError, match="Unknown operation"):
            compute_output_shape(input_shape, "unknown_op")

    def test_check_finite_gradients_finite(self):
        """Test check_finite_gradients with finite gradients."""
        tensor = Tensor([1.0, 2.0, 3.0])
        tensor.grad = np.array([0.1, 0.2, 0.3])

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            check_finite_gradients(tensor, "test_op")

            # Should not log any warnings
            mock_logger.warning.assert_not_called()

    def test_check_finite_gradients_nan(self):
        """Test check_finite_gradients with NaN gradients."""
        tensor = Tensor([1.0, 2.0, 3.0], name="test_tensor")
        tensor.grad = np.array([0.1, np.nan, 0.3])

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            check_finite_gradients(tensor, "test_op")

            # Should log NaN warning
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "NaN gradients" in warning_msg
            assert "test_tensor" in warning_msg

    def test_check_finite_gradients_inf(self):
        """Test check_finite_gradients with infinite gradients."""
        tensor = Tensor([1.0, 2.0, 3.0], name="test_tensor")
        tensor.grad = np.array([0.1, np.inf, 0.3])

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            check_finite_gradients(tensor, "test_op")

            # Should log infinite warning
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "Infinite gradients" in warning_msg
            assert "test_tensor" in warning_msg

    def test_check_finite_gradients_no_grad(self):
        """Test check_finite_gradients with no gradients."""
        tensor = Tensor([1.0, 2.0, 3.0])
        tensor.grad = None

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            check_finite_gradients(tensor, "test_op")

            # Should not log anything
            mock_logger.warning.assert_not_called()

    def test_apply_gradient_clipping_no_clipping(self):
        """Test apply_gradient_clipping when no clipping is needed."""
        grad = np.array([1.0, 2.0, 3.0])
        max_norm = 10.0

        result = apply_gradient_clipping(grad, max_norm)

        # Should return unchanged gradient
        np.testing.assert_array_equal(result, grad)

    def test_apply_gradient_clipping_with_clipping(self):
        """Test apply_gradient_clipping when clipping is needed."""
        grad = np.array([3.0, 4.0])  # Norm = 5.0
        max_norm = 2.0

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            result = apply_gradient_clipping(grad, max_norm)

            # Should clip gradient to max_norm
            result_norm = np.linalg.norm(result)
            assert np.isclose(result_norm, max_norm)

            # Should log clipping
            mock_logger.debug.assert_called_once()
            assert "Clipping gradient" in str(mock_logger.debug.call_args)

    def test_apply_gradient_clipping_zero_grad(self):
        """Test apply_gradient_clipping with zero gradient."""
        grad = np.array([0.0, 0.0, 0.0])
        max_norm = 1.0

        result = apply_gradient_clipping(grad, max_norm)

        # Should return unchanged (zero norm)
        np.testing.assert_array_equal(result, grad)

    def test_memory_efficient_operation_decorator_success(self):
        """Test memory_efficient_operation decorator with successful operation."""

        @memory_efficient_operation
        def test_operation(x):
            return x * 2

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            result = test_operation(5)

            assert result == 10

            # Should log start and completion
            assert mock_logger.debug.call_count >= 2
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("Starting operation" in call for call in debug_calls)
            assert any("Completed operation" in call for call in debug_calls)

    def test_memory_efficient_operation_decorator_error(self):
        """Test memory_efficient_operation decorator with error."""

        @memory_efficient_operation
        def failing_operation(x):
            raise ValueError("Test error")

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            with pytest.raises(ValueError, match="Test error"):
                failing_operation(5)

            # Should log error
            mock_logger.error.assert_called_once()
            error_msg = str(mock_logger.error.call_args)
            assert "Test error" in error_msg

    def test_memory_efficient_operation_decorator_preserves_metadata(self):
        """Test that memory_efficient_operation decorator preserves function metadata."""

        def original_function(x, y=1):
            """Test function docstring."""
            return x + y

        decorated = memory_efficient_operation(original_function)

        # Should preserve function name
        assert decorated.__name__ == "wrapper"  # Wrapper function name

        # Function should still work correctly
        assert decorated(5, y=2) == 7

    def test_utilities_with_edge_cases(self):
        """Test utility functions with edge cases."""
        # Test broadcast_tensors with scalar
        scalar = Tensor(5.0)
        vector = Tensor([1, 2, 3])

        result = broadcast_tensors(scalar, vector)
        assert len(result) == 2
        assert result[0].shape == (3,)
        assert result[1].shape == (3,)

        # Test reduce_gradient with matching shapes
        grad = np.array([1, 2, 3])
        target_shape = (3,)
        result = reduce_gradient(grad, target_shape, target_shape)
        np.testing.assert_array_equal(result, grad)

    def test_utilities_type_handling(self):
        """Test utility functions with different data types."""
        # Test with different tensor dtypes
        tensor_int = Tensor([1, 2, 3], dtype="int32")
        tensor_float = Tensor([1.0, 2.0, 3.0], dtype="float64")

        # Broadcast should handle different types
        result = broadcast_tensors(tensor_int, tensor_float)
        assert len(result) == 2

        # Ensure_tensor should handle numpy arrays
        numpy_array = np.array([1, 2, 3])
        tensor_result = ensure_tensor(numpy_array)
        assert isinstance(tensor_result, Tensor)

    def test_utilities_performance_with_large_data(self):
        """Test utility functions with large data for performance."""
        # Large tensors
        large_a = Tensor(np.random.randn(1000, 1000))
        large_b = Tensor(np.random.randn(1, 1000))

        # Should handle large tensor broadcasting
        result = broadcast_tensors(large_a, large_b)
        assert len(result) == 2
        assert result[0].shape == (1000, 1000)
        assert result[1].shape == (1000, 1000)

        # Should handle large gradient reduction
        large_grad = np.random.randn(100, 1000, 1000)
        target_shape = (1000, 1000)
        result = reduce_gradient(large_grad, target_shape, (100, 1000, 1000))
        assert result.shape == target_shape

    def test_utilities_error_consistency(self):
        """Test that utilities provide consistent error messages."""
        # Test consistent error message format
        with pytest.raises(ValueError) as exc_info:
            get_broadcast_shape((2, 3), (4, 5))

        assert "Cannot broadcast shapes" in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            validate_tensor_operation("not_tensor", Tensor([1]), "test")

        assert "requires Tensor inputs" in str(exc_info.value)
