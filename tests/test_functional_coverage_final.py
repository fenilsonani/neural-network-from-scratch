"""Final comprehensive test suite for functional modules targeting 95%+ coverage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_arch.core.tensor import Tensor
from neural_arch.functional.activation import (
    geglu,
    gelu,
    glu,
    leaky_relu,
    mish,
    reglu,
    relu,
    sigmoid,
    silu,
    softmax,
    swiglu,
    swish,
    tanh,
)
from neural_arch.functional.arithmetic import add, div, matmul, mul, neg, sub
from neural_arch.functional.loss import (
    cosine_embedding_loss,
    cross_entropy_loss,
    focal_loss,
    huber_loss,
    kl_divergence_loss,
    label_smoothing_cross_entropy,
    mse_loss,
    triplet_loss,
)
from neural_arch.functional.pooling import max_pool, mean_pool
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


class TestActivationFunctionsCoverage:
    """Targeted tests for activation function coverage."""

    def test_all_activation_functions_basic(self):
        """Test all activation functions with basic inputs."""
        x = Tensor([[1.0, -1.0, 0.0, 2.0]], requires_grad=True)

        # Test all activations work
        activations = [
            relu(x),
            sigmoid(x),
            tanh(x),
            gelu(x),
            gelu(x, approximate=True),
            mish(x),
            silu(x),
            leaky_relu(x),
            leaky_relu(x, negative_slope=0.2),
        ]

        for result in activations:
            assert isinstance(result, Tensor)
            assert result.requires_grad
            assert result._grad_fn is not None

    def test_gated_activations(self):
        """Test gated activation functions."""
        # Even dimension for gated functions
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

        # Test all gated activations
        results = [glu(x), reglu(x), geglu(x), swiglu(x)]

        for result in results:
            assert isinstance(result, Tensor)
            assert result.shape == (1, 2)  # Half input size
            assert result.requires_grad

    def test_gated_activations_errors(self):
        """Test gated activation error cases."""
        # Odd dimension should fail
        x_odd = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        with pytest.raises(ValueError):
            glu(x_odd)
        with pytest.raises(ValueError):
            reglu(x_odd)
        with pytest.raises(ValueError):
            geglu(x_odd)
        with pytest.raises(ValueError):
            swiglu(x_odd)

    def test_softmax_axis_parameter(self):
        """Test softmax with different axis parameters."""
        x = Tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)

        # Test different axes
        result_0 = softmax(x, axis=0)
        result_1 = softmax(x, axis=1)
        result_2 = softmax(x, axis=2)

        assert all(isinstance(r, Tensor) for r in [result_0, result_1, result_2])

    def test_activation_without_gradients(self):
        """Test activations without gradient requirements."""
        x = Tensor([[1.0, -1.0]], requires_grad=False)

        result = relu(x)
        assert not result.requires_grad
        assert result._grad_fn is None

    def test_swish_alias(self):
        """Test that swish is silu."""
        assert swish is silu


class TestLossFunctionsCoverage:
    """Targeted tests for loss function coverage."""

    def test_all_loss_functions_basic(self):
        """Test all loss functions with basic inputs."""
        predictions = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)
        targets = Tensor([0])

        # Test basic loss functions
        losses = [
            cross_entropy_loss(predictions, targets),
            mse_loss(predictions, targets),
            focal_loss(predictions, targets),
            label_smoothing_cross_entropy(predictions, targets),
            huber_loss(predictions, targets),
            kl_divergence_loss(predictions, targets),
        ]

        for loss in losses:
            assert isinstance(loss, Tensor)
            assert loss.requires_grad

    def test_loss_reduction_modes(self):
        """Test loss functions with different reduction modes."""
        predictions = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        targets = Tensor([0, 1])

        for reduction in ["mean", "sum", "none"]:
            ce_loss = cross_entropy_loss(predictions, targets, reduction=reduction)
            mse_loss_result = mse_loss(predictions, targets, reduction=reduction)

            if reduction == "none":
                assert ce_loss.shape == (2,)
                assert mse_loss_result.shape == (2, 2)
            else:
                assert ce_loss.shape == () or ce_loss.shape == (1,)

    def test_loss_invalid_reduction(self):
        """Test loss functions with invalid reduction."""
        predictions = Tensor([[1.0, 2.0]])
        targets = Tensor([0])

        with pytest.raises(ValueError, match="Unknown reduction"):
            cross_entropy_loss(predictions, targets, reduction="invalid")

    def test_embedding_losses(self):
        """Test embedding loss functions."""
        input1 = Tensor([[1.0, 2.0]], requires_grad=True)
        input2 = Tensor([[2.0, 1.0]], requires_grad=True)
        target = Tensor([1])  # Similar

        cosine_loss = cosine_embedding_loss(input1, input2, target)
        assert isinstance(cosine_loss, Tensor)

        # Test triplet loss
        anchor = Tensor([[1.0, 2.0]], requires_grad=True)
        positive = Tensor([[1.1, 2.1]], requires_grad=True)
        negative = Tensor([[3.0, 4.0]], requires_grad=True)

        triplet_loss_result = triplet_loss(anchor, positive, negative)
        assert isinstance(triplet_loss_result, Tensor)

    def test_embedding_loss_errors(self):
        """Test embedding loss error cases."""
        # Shape mismatch for cosine embedding
        input1 = Tensor([[1.0, 2.0]])
        input2 = Tensor([[1.0]])  # Different shape
        target = Tensor([1])

        with pytest.raises(ValueError, match="Input shapes must match"):
            cosine_embedding_loss(input1, input2, target)

        # Shape mismatch for triplet loss
        anchor = Tensor([[1.0, 2.0]])
        positive = Tensor([[1.0]])  # Different shape
        negative = Tensor([[3.0, 4.0]])

        with pytest.raises(ValueError, match="All input shapes must match"):
            triplet_loss(anchor, positive, negative)

    def test_specialized_loss_parameters(self):
        """Test specialized loss function parameters."""
        predictions = Tensor([[1.0, 2.0]])
        targets = Tensor([0])

        # Focal loss with different parameters
        focal_result = focal_loss(predictions, targets, alpha=0.25, gamma=2.0)
        assert isinstance(focal_result, Tensor)

        # Label smoothing with different smoothing
        ls_result = label_smoothing_cross_entropy(predictions, targets, smoothing=0.1)
        assert isinstance(ls_result, Tensor)

        # Huber loss with different delta
        huber_result = huber_loss(predictions, targets, delta=2.0)
        assert isinstance(huber_result, Tensor)

        # Triplet loss with different norms
        anchor = Tensor([[1.0, 2.0]])
        positive = Tensor([[1.1, 2.1]])
        negative = Tensor([[3.0, 4.0]])

        triplet_l1 = triplet_loss(anchor, positive, negative, p=1.0)
        triplet_l2 = triplet_loss(anchor, positive, negative, p=2.0)
        triplet_lp = triplet_loss(anchor, positive, negative, p=3.0)

        assert all(isinstance(t, Tensor) for t in [triplet_l1, triplet_l2, triplet_lp])

    def test_loss_parameter_validation(self):
        """Test loss function parameter validation."""
        predictions = Tensor([[1.0, 2.0]])
        targets = Tensor([0])

        # Invalid smoothing
        with pytest.raises(ValueError, match="Smoothing must be in"):
            label_smoothing_cross_entropy(predictions, targets, smoothing=-0.1)

        with pytest.raises(ValueError, match="Smoothing must be in"):
            label_smoothing_cross_entropy(predictions, targets, smoothing=1.5)

        # Invalid delta for Huber loss
        with pytest.raises(ValueError, match="Delta must be positive"):
            huber_loss(predictions, targets, delta=0.0)

        # Invalid p for triplet loss
        anchor = Tensor([[1.0, 2.0]])
        positive = Tensor([[1.1, 2.1]])
        negative = Tensor([[3.0, 4.0]])

        with pytest.raises(ValueError, match="Norm degree p must be positive"):
            triplet_loss(anchor, positive, negative, p=0.0)


class TestArithmeticOperationsCoverage:
    """Targeted tests for arithmetic operation coverage."""

    def test_all_arithmetic_operations(self):
        """Test all arithmetic operations."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)

        # Test all operations
        operations = [add(a, b), sub(a, b), mul(a, b), div(a, b), neg(a)]

        for result in operations:
            assert isinstance(result, Tensor)
            assert result.requires_grad

    def test_matmul_operations(self):
        """Test matrix multiplication operations."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        result = matmul(a, b)
        assert isinstance(result, Tensor)
        assert result.requires_grad
        assert result.shape == (2, 2)

    def test_arithmetic_error_cases(self):
        """Test arithmetic operation error cases."""
        # Division by zero
        a = Tensor([1.0, 2.0])
        b = Tensor([0.0, 1.0])

        with pytest.raises(ValueError, match="Division by zero"):
            div(a, b)

        # Incompatible matrix dimensions
        a = Tensor([[1, 2, 3]])  # (1, 3)
        b = Tensor([[4, 5]])  # (1, 2)

        with pytest.raises(ValueError, match="Incompatible matrix dimensions"):
            matmul(a, b)

        # 1D tensors for matmul
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        with pytest.raises(ValueError, match="matmul requires 2D\\+ tensors"):
            matmul(a, b)

    def test_tensor_conversion(self):
        """Test automatic tensor conversion in operations."""
        # Scalar + tensor
        result = add(5.0, Tensor([1.0, 2.0]))
        assert isinstance(result, Tensor)

        # Tensor + scalar
        result = mul(Tensor([1.0, 2.0]), 3.0)
        assert isinstance(result, Tensor)


class TestPoolingOperationsCoverage:
    """Targeted tests for pooling operation coverage."""

    def test_pooling_operations(self):
        """Test pooling operations with different axes."""
        x = Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)

        # Test different axes
        for axis in [0, 1, 2]:
            mean_result = mean_pool(x, axis=axis)
            max_result = max_pool(x, axis=axis)

            assert isinstance(mean_result, Tensor)
            assert isinstance(max_result, Tensor)
            assert mean_result.requires_grad
            assert max_result.requires_grad

    def test_pooling_without_gradients(self):
        """Test pooling without gradients."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)

        mean_result = mean_pool(x, axis=1)
        max_result = max_pool(x, axis=1)

        assert not mean_result.requires_grad
        assert not max_result.requires_grad
        assert mean_result._grad_fn is None
        assert max_result._grad_fn is None


class TestUtilsFunctionsCoverage:
    """Targeted tests for utility function coverage."""

    def test_broadcast_tensors(self):
        """Test broadcast_tensors function."""
        a = Tensor([[1, 2, 3]])  # (1, 3)
        b = Tensor([[1], [2]])  # (2, 1)

        result = broadcast_tensors(a, b)
        assert len(result) == 2
        assert result[0].shape == (2, 3)
        assert result[1].shape == (2, 3)

        # Test incompatible broadcasting
        a = Tensor([[1, 2, 3]])  # (1, 3)
        b = Tensor([[1, 2], [3, 4]])  # (2, 2)

        with pytest.raises(ValueError, match="Cannot broadcast tensors"):
            broadcast_tensors(a, b)

    def test_reduce_gradient(self):
        """Test reduce_gradient function."""
        # Scalar target
        grad = np.array([[1, 2], [3, 4]])
        result = reduce_gradient(grad, (), (2, 2))
        assert np.isscalar(result) or result.shape == ()

        # Shape reduction
        grad = np.array([[[1, 2]], [[3, 4]]])
        result = reduce_gradient(grad, (1, 2), (2, 1, 2))
        assert result.shape == (1, 2)

    def test_get_broadcast_shape(self):
        """Test get_broadcast_shape function."""
        # Compatible shapes
        result = get_broadcast_shape((1, 3), (2, 1))
        assert result == (2, 3)

        # Incompatible shapes
        with pytest.raises(ValueError, match="Cannot broadcast shapes"):
            get_broadcast_shape((2, 3), (2, 4))

    def test_validate_tensor_operation(self):
        """Test validate_tensor_operation function."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        # Valid tensors
        validate_tensor_operation(a, b, "test")

        # Invalid types
        with pytest.raises(TypeError, match="test requires Tensor inputs"):
            validate_tensor_operation("not_tensor", b, "test")

    def test_ensure_tensor(self):
        """Test ensure_tensor function."""
        # Already tensor
        original = Tensor([1, 2, 3])
        result = ensure_tensor(original)
        assert result is original

        # Conversion needed
        data = [1, 2, 3]
        result = ensure_tensor(data)
        assert isinstance(result, Tensor)

        # Conversion failure
        with pytest.raises(TypeError, match="Cannot convert"):
            ensure_tensor(object())

    def test_compute_output_shape(self):
        """Test compute_output_shape function."""
        # Mean pool
        result = compute_output_shape((2, 4, 8), "mean_pool", axis=1)
        assert result == (2, 8)

        # Shape preserving operations
        result = compute_output_shape((2, 3), "softmax")
        assert result == (2, 3)

        # Unknown operation
        with pytest.raises(ValueError, match="Unknown operation"):
            compute_output_shape((2, 3), "unknown")

    def test_check_finite_gradients(self):
        """Test check_finite_gradients function."""
        tensor = Tensor([1.0, 2.0], name="test")

        # Finite gradients
        tensor.grad = np.array([0.1, 0.2])
        with patch("neural_arch.functional.utils.logger") as mock_logger:
            check_finite_gradients(tensor, "test_op")
            mock_logger.warning.assert_not_called()

        # NaN gradients
        tensor.grad = np.array([0.1, np.nan])
        with patch("neural_arch.functional.utils.logger") as mock_logger:
            check_finite_gradients(tensor, "test_op")
            mock_logger.warning.assert_called()

    def test_apply_gradient_clipping(self):
        """Test apply_gradient_clipping function."""
        # No clipping needed
        grad = np.array([1.0, 2.0])
        result = apply_gradient_clipping(grad, max_norm=10.0)
        np.testing.assert_array_equal(result, grad)

        # Clipping needed
        grad = np.array([3.0, 4.0])  # Norm = 5.0
        with patch("neural_arch.functional.utils.logger") as mock_logger:
            result = apply_gradient_clipping(grad, max_norm=2.0)
            assert np.isclose(np.linalg.norm(result), 2.0)
            mock_logger.debug.assert_called_once()

    def test_memory_efficient_operation(self):
        """Test memory_efficient_operation decorator."""

        @memory_efficient_operation
        def test_op(x):
            return x * 2

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            result = test_op(5)
            assert result == 10
            # Should log start and completion
            assert mock_logger.debug.call_count >= 2

        # Test error handling
        @memory_efficient_operation
        def failing_op(x):
            raise ValueError("Test error")

        with patch("neural_arch.functional.utils.logger") as mock_logger:
            with pytest.raises(ValueError, match="Test error"):
                failing_op(5)
            mock_logger.error.assert_called_once()


class TestFunctionalIntegration:
    """Integration tests for functional operations."""

    def test_complex_function_composition(self):
        """Test complex composition of functional operations."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        # Complex computation: softmax(relu(x)) + mean_pool
        step1 = relu(x)
        step2 = softmax(step1, axis=1)
        step3 = mean_pool(step2, axis=0)

        assert isinstance(step3, Tensor)
        assert step3.requires_grad
        assert step3.shape == (2,)

    def test_gradient_flow_through_operations(self):
        """Test gradient flow through multiple operations."""
        x = Tensor([[2.0, 3.0]], requires_grad=True)
        y = Tensor([[1.0, 2.0]], requires_grad=True)

        # Operations: (x + y) * sigmoid(x)
        sum_result = add(x, y)
        sigmoid_result = sigmoid(x)
        final_result = mul(sum_result, sigmoid_result)

        assert final_result.requires_grad
        assert final_result._grad_fn is not None

    def test_mixed_operation_types(self):
        """Test mixing different types of operations."""
        # Start with tensor
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

        # Apply activation
        activated = relu(x)

        # Apply pooling
        pooled = mean_pool(activated, axis=1)

        # Apply arithmetic
        scaled = mul(pooled, 2.0)

        # Apply loss (need targets)
        targets = Tensor([1.0])
        loss = mse_loss(scaled, targets)

        assert isinstance(loss, Tensor)
        assert loss.requires_grad


class TestFunctionalEdgeCases:
    """Test edge cases for functional operations."""

    def test_extreme_values(self):
        """Test operations with extreme values."""
        # Very large values
        x_large = Tensor([[100.0, -100.0]])

        # Should handle without overflow/underflow
        sigmoid_result = sigmoid(x_large)
        assert np.all(np.isfinite(sigmoid_result.data))

        softmax_result = softmax(x_large)
        assert np.all(np.isfinite(softmax_result.data))
        assert np.isclose(np.sum(softmax_result.data), 1.0)

    def test_small_values(self):
        """Test operations with very small values."""
        x_small = Tensor([[1e-10, 1e-9, 1e-8]])

        # Should handle small values
        result = relu(x_small)
        assert np.all(np.isfinite(result.data))

        result = gelu(x_small)
        assert np.all(np.isfinite(result.data))

    def test_zero_values(self):
        """Test operations with zero values."""
        x_zeros = Tensor([[0.0, 0.0, 0.0]])

        # Test activations with zeros
        activations = [relu(x_zeros), sigmoid(x_zeros), tanh(x_zeros), gelu(x_zeros)]

        for result in activations:
            assert np.all(np.isfinite(result.data))

    def test_single_element_tensors(self):
        """Test operations with single element tensors."""
        x = Tensor([[1.0]], requires_grad=True)

        # Should work with single elements
        result = softmax(x)
        assert np.isclose(result.data[0, 0], 1.0)

        result = mean_pool(x, axis=1)
        assert result.data.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
