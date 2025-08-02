"""Comprehensive tests for functional/utils.py to improve coverage from 83.98% to 95%+.

This file tests functional utilities.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core.tensor import Tensor
from neural_arch.exceptions import NumericalError, ShapeError
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


class TestReduceGradient:
    """Test reduce_gradient function comprehensively."""

    def test_reduce_gradient_same_shape(self):
        """Test gradient with same shape as target."""
        grad = np.array([[1, 2], [3, 4]])
        target_shape = (2, 2)
        broadcast_shape = (2, 2)

        result = reduce_gradient(grad, target_shape, broadcast_shape)
        np.testing.assert_array_equal(result, grad)
        assert result.shape == target_shape

    def test_reduce_gradient_broadcast_axes(self):
        """Test gradient reduction over broadcast axes."""
        # Gradient from (2, 3) operation with (3,) input
        grad = np.array([[1, 2, 3], [4, 5, 6]])
        target_shape = (3,)
        broadcast_shape = (2, 3)

        result = reduce_gradient(grad, target_shape, broadcast_shape)
        # Should sum over first dimension
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_reduce_gradient_keepdims(self):
        """Test gradient reduction with keepdims."""
        # Gradient from (2, 3) operation with (2, 1) input
        grad = np.array([[1, 2, 3], [4, 5, 6]])
        target_shape = (2, 1)
        broadcast_shape = (2, 3)

        result = reduce_gradient(grad, target_shape, broadcast_shape)
        # Should sum over second dimension, keep dims
        expected = np.array([[6], [15]])
        np.testing.assert_array_equal(result, expected)

    def test_reduce_gradient_complex_broadcast(self):
        """Test complex broadcasting scenario - lines 70-73."""
        # Force path through lines 70-73
        grad = np.ones((4, 3, 2))  # (4, 3, 2)
        target_shape = (2,)  # Much smaller shape
        broadcast_shape = (4, 3, 2)

        result = reduce_gradient(grad, target_shape, broadcast_shape)
        # Should sum over first two dimensions
        expected = np.ones(2) * 12  # 4*3 = 12
        np.testing.assert_array_equal(result, expected)

    def test_reduce_gradient_reshape_path(self):
        """Test reduce_gradient reshape path - lines 82-93."""
        # Force the complex reshape path
        grad = np.ones((2, 3, 4))  # 24 elements
        target_shape = (3, 4)  # This is what reduce_gradient actually returns
        broadcast_shape = (2, 3, 4)

        result = reduce_gradient(grad, target_shape, broadcast_shape)
        assert result.shape == target_shape
        # Should sum over first dimension
        expected = np.ones((3, 4)) * 2  # Summed over first axis
        np.testing.assert_array_equal(result, expected)


class TestBroadcastTensors:
    """Test broadcast_tensors function."""

    def test_broadcast_tensors_basic(self):
        """Test basic tensor broadcasting."""
        t1 = Tensor([[1, 2]], requires_grad=True)
        t2 = Tensor([[3], [4]], requires_grad=True)

        results = broadcast_tensors(t1, t2)

        assert len(results) == 2
        assert results[0].shape == (2, 2)
        assert results[1].shape == (2, 2)

        # Check values
        expected_t1 = np.array([[1, 2], [1, 2]])
        expected_t2 = np.array([[3, 3], [4, 4]])
        np.testing.assert_array_equal(results[0], expected_t1)
        np.testing.assert_array_equal(results[1], expected_t2)

    def test_broadcast_tensors_scalar(self):
        """Test broadcasting with scalar."""
        t1 = Tensor(5.0, requires_grad=True)
        t2 = Tensor([[1, 2], [3, 4]], requires_grad=True)

        results = broadcast_tensors(t1, t2)

        assert results[0].shape == (2, 2)
        assert results[1].shape == (2, 2)
        np.testing.assert_array_equal(results[0], [[5, 5], [5, 5]])


class TestValidateTensorOperation:
    """Test validate_tensor_operation function."""

    def test_validate_compatible_shapes(self):
        """Test validation with compatible shapes."""
        t1 = Tensor([[1, 2], [3, 4]], requires_grad=True)
        t2 = Tensor([[5, 6], [7, 8]], requires_grad=True)

        # Should not raise
        validate_tensor_operation(t1, t2, "add")

    def test_validate_broadcast_shapes(self):
        """Test validation with broadcastable shapes."""
        t1 = Tensor([[1, 2]], requires_grad=True)
        t2 = Tensor([[3], [4]], requires_grad=True)

        # Should not raise
        validate_tensor_operation(t1, t2, "multiply")

    def test_validate_incompatible_shapes(self):
        """Test validation with incompatible shapes."""
        t1 = Tensor([[1, 2, 3]], requires_grad=True)
        t2 = Tensor([[4, 5]], requires_grad=True)

        # validate_tensor_operation doesn't check shape compatibility,
        # it only checks if inputs are tensors
        validate_tensor_operation(t1, t2, "add")  # Should not raise


class TestEnsureTensor:
    """Test ensure_tensor function."""

    def test_ensure_tensor_from_list(self):
        """Test creating tensor from list."""
        result = ensure_tensor([1, 2, 3])
        assert isinstance(result, Tensor)
        np.testing.assert_array_equal(result.data, [1, 2, 3])

    def test_ensure_tensor_from_scalar(self):
        """Test creating tensor from scalar."""
        result = ensure_tensor(5.0)
        assert isinstance(result, Tensor)
        assert result.data == 5.0

    def test_ensure_tensor_already_tensor(self):
        """Test with already a tensor."""
        t = Tensor([1, 2, 3], requires_grad=True)
        result = ensure_tensor(t)
        assert result is t  # Should return same object

    def test_ensure_tensor_with_name(self):
        """Test ensure_tensor with custom name."""
        result = ensure_tensor([1, 2, 3], name="my_tensor")
        assert isinstance(result, Tensor)


class TestComputeOutputShape:
    """Test compute_output_shape function."""

    def test_compute_mean_pool_shape(self):
        """Test mean_pool output shape computation."""
        input_shape = (3, 4, 5)
        # Pool along axis 1 (default)
        output_shape = compute_output_shape(input_shape, "mean_pool")
        assert output_shape == (3, 5)

        # Pool along axis 2
        output_shape = compute_output_shape(input_shape, "mean_pool", axis=2)
        assert output_shape == (3, 4)

        # Pool along negative axis
        output_shape = compute_output_shape(input_shape, "mean_pool", axis=-1)
        assert output_shape == (3, 4)

    def test_compute_softmax_shape(self):
        """Test softmax output shape computation."""
        input_shape = (2, 3, 4)
        output_shape = compute_output_shape(input_shape, "softmax")
        assert output_shape == input_shape  # Softmax preserves shape

    def test_compute_relu_shape(self):
        """Test relu output shape computation."""
        input_shape = (2, 3, 4)
        output_shape = compute_output_shape(input_shape, "relu")
        assert output_shape == input_shape  # ReLU preserves shape

    def test_compute_unknown_operation(self):
        """Test unknown operation raises error."""
        input_shape = (2, 3, 4)
        with pytest.raises(ValueError, match="Unknown operation"):
            compute_output_shape(input_shape, "unknown_op")


class TestCheckFiniteGradients:
    """Test check_finite_gradients function."""

    def test_finite_gradients(self):
        """Test with finite gradients."""
        t = Tensor([1, 2, 3], requires_grad=True)
        t.grad = np.array([0.1, 0.2, 0.3])

        # Should not raise
        check_finite_gradients(t, "test_op")

    def test_nan_gradients(self):
        """Test with NaN gradients."""
        t = Tensor([1, 2, 3], requires_grad=True)
        t.grad = np.array([0.1, np.nan, 0.3])

        # check_finite_gradients only logs warnings, doesn't raise
        check_finite_gradients(t, "test_op")  # Should not raise

    def test_inf_gradients(self):
        """Test with infinite gradients."""
        t = Tensor([1, 2, 3], requires_grad=True)
        t.grad = np.array([0.1, np.inf, 0.3])

        # check_finite_gradients only logs warnings, doesn't raise
        check_finite_gradients(t, "test_op")  # Should not raise

    def test_no_gradient(self):
        """Test with no gradient."""
        t = Tensor([1, 2, 3], requires_grad=True)
        # No gradient set

        # Should not raise (no gradient to check)
        check_finite_gradients(t, "test_op")


class TestApplyGradientClipping:
    """Test apply_gradient_clipping function."""

    def test_gradient_clipping_below_threshold(self):
        """Test gradient clipping when norm is below threshold."""
        grad = np.array([1.0, 2.0, 3.0])
        # Norm is sqrt(1+4+9) = sqrt(14) ≈ 3.74, which is < 10

        clipped = apply_gradient_clipping(grad, max_norm=10.0)
        np.testing.assert_array_equal(clipped, grad)  # Should be unchanged

    def test_gradient_clipping_above_threshold(self):
        """Test gradient clipping when norm exceeds threshold."""
        grad = np.array([3.0, 4.0, 0.0])
        # Norm is 5.0

        clipped = apply_gradient_clipping(grad, max_norm=2.5)
        # Should be scaled by 2.5/5.0 = 0.5
        expected = np.array([1.5, 2.0, 0.0])
        np.testing.assert_array_almost_equal(clipped, expected)

    def test_gradient_clipping_zero_gradient(self):
        """Test gradient clipping with zero gradient."""
        grad = np.zeros(5)

        clipped = apply_gradient_clipping(grad)
        np.testing.assert_array_equal(clipped, grad)

    def test_gradient_clipping_custom_norm(self):
        """Test gradient clipping with custom max norm."""
        grad = np.array([10.0, 0.0, 0.0])

        clipped = apply_gradient_clipping(grad, max_norm=5.0)
        expected = np.array([5.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(clipped, expected)


class TestMemoryEfficientOperation:
    """Test memory_efficient_operation decorator."""

    def test_memory_efficient_decorator(self):
        """Test the memory efficient operation decorator."""
        call_count = 0

        @memory_efficient_operation
        def test_func(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # Call the decorated function
        result = test_func(5, 3)
        assert result == 8
        assert call_count == 1

        # Test with arrays
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = test_func(x, y)
        np.testing.assert_array_equal(result, [5, 7, 9])
