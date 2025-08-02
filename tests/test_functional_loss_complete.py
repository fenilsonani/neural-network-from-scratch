"""Complete test coverage for functional/loss.py targeting 95%+ coverage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from neural_arch.core.tensor import GradientFunction, Tensor
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


class TestLossFunctionsComplete:
    """Complete test coverage for all loss functions."""

    def test_cross_entropy_loss_with_backend_handling(self):
        """Test cross-entropy loss with different backend data handling."""
        predictions = Tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]], requires_grad=True)
        targets = Tensor([0, 1])

        # Mock different backend scenarios
        # Case 1: CuPy-like backend for predictions
        mock_pred_backend = MagicMock()
        mock_pred_backend.get.return_value = predictions.data
        predictions.backend_data = mock_pred_backend

        # Case 2: Standard backend for targets
        mock_target_backend = MagicMock()
        mock_target_backend.to_numpy.return_value = targets.data
        targets.backend_data = mock_target_backend
        targets.backend.to_numpy = mock_target_backend.to_numpy

        result = cross_entropy_loss(predictions, targets)

        assert isinstance(result, Tensor)
        assert result.requires_grad

        # Test backward pass
        grad_output = np.array(1.0)
        result._grad_fn.apply(grad_output)

    def test_cross_entropy_loss_gradient_scaling(self):
        """Test cross-entropy loss gradient scaling with different reductions."""
        predictions = Tensor([[2.0, 1.0], [1.0, 3.0]], requires_grad=True)
        targets = Tensor([0, 1])

        # Test with scalar gradient
        result = cross_entropy_loss(predictions, targets, reduction="mean")
        grad_output = np.array(2.0)  # Scalar gradient
        result._grad_fn.apply(grad_output)

        # Test with sum reduction
        result_sum = cross_entropy_loss(predictions, targets, reduction="sum")
        grad_output = np.array(1.0)
        result_sum._grad_fn.apply(grad_output)

        # Test with none reduction
        result_none = cross_entropy_loss(predictions, targets, reduction="none")
        grad_output = np.array([1.0, 0.5])
        result_none._grad_fn.apply(grad_output)

    def test_mse_loss_with_backend_conversion(self):
        """Test MSE loss with backend data conversion."""
        predictions = Tensor([[1.0, 2.0]], requires_grad=True)
        targets = Tensor([[0.5, 1.5]], requires_grad=True)

        # Mock backend data with to_numpy method
        mock_pred_backend = MagicMock()
        mock_pred_backend.to_numpy.return_value = predictions.data
        predictions.backend_data = mock_pred_backend
        predictions.backend.to_numpy = mock_pred_backend.to_numpy

        mock_target_backend = MagicMock()
        mock_target_backend.to_numpy.return_value = targets.data
        targets.backend_data = mock_target_backend
        targets.backend.to_numpy = mock_target_backend.to_numpy

        result = mse_loss(predictions, targets)

        assert isinstance(result, Tensor)
        assert result.requires_grad

    def test_mse_loss_gradient_with_array_output(self):
        """Test MSE loss gradient with array output."""
        predictions = Tensor([[1.0, 2.0]], requires_grad=True)
        targets = Tensor([[0.5, 1.5]], requires_grad=True)

        result = mse_loss(predictions, targets, reduction="none")

        # Test with array gradient output
        grad_output = np.array([[1.0, 0.5]])
        result._grad_fn.apply(grad_output)

    def test_focal_loss_detailed_gradient_computation(self):
        """Test focal loss detailed gradient computation paths."""
        predictions = Tensor([[2.0, 1.0], [1.0, 3.0]], requires_grad=True)
        targets = Tensor([0, 1])

        # Test with gamma=0 (weighted cross-entropy case)
        result_gamma0 = focal_loss(predictions, targets, gamma=0.0)
        grad_output = np.array(1.0)
        result_gamma0._grad_fn.apply(grad_output)

        # Test with gamma>0 (full focal loss)
        result_focal = focal_loss(predictions, targets, gamma=2.0, alpha=0.25)
        grad_output = np.array(1.0)
        result_focal._grad_fn.apply(grad_output)

    def test_label_smoothing_cross_entropy_edge_cases(self):
        """Test label smoothing cross-entropy edge cases."""
        predictions = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)
        targets = Tensor([0])

        # Test with zero smoothing
        result_zero = label_smoothing_cross_entropy(predictions, targets, smoothing=0.0)
        assert isinstance(result_zero, Tensor)

        # Test with near-maximum smoothing
        result_max = label_smoothing_cross_entropy(predictions, targets, smoothing=0.99)
        assert isinstance(result_max, Tensor)

        # Test gradient computation with array output
        result = label_smoothing_cross_entropy(
            predictions, targets, smoothing=0.1, reduction="none"
        )
        grad_output = np.array([2.0])
        result._grad_fn.apply(grad_output)

    def test_huber_loss_reduction_scaling(self):
        """Test Huber loss gradient with reduction scaling."""
        predictions = Tensor([1.0, 5.0], requires_grad=True)  # Mix of quadratic/linear
        targets = Tensor([0.5, 1.0], requires_grad=True)

        result = huber_loss(predictions, targets, delta=1.0, reduction="mean")

        # Test gradient with array output
        grad_output = np.array(3.0)
        result._grad_fn.apply(grad_output)

    def test_kl_divergence_loss_input_conversions(self):
        """Test KL divergence loss with different input types."""
        # Test with predictions as logits and targets as logits
        predictions = Tensor([[2.0, 1.0]], requires_grad=True)  # max > 0 (logits)
        targets = Tensor([[1.5, 2.5]])  # Different scale (logits)

        result_logits = kl_divergence_loss(predictions, targets)
        assert isinstance(result_logits, Tensor)

        # Test with predictions as log probs and targets as invalid probs
        predictions = Tensor([[-1.0, -2.0]], requires_grad=True)  # max <= 0 (log probs)
        targets = Tensor([[2.0, 3.0]])  # max > 1.0 or min < 0 (logits)

        result_mixed = kl_divergence_loss(predictions, targets)
        assert isinstance(result_mixed, Tensor)

    def test_kl_divergence_loss_gradient_array_handling(self):
        """Test KL divergence loss gradient with array handling."""
        predictions = Tensor([[1.0, 2.0]], requires_grad=True)
        targets = Tensor([[0.4, 0.6]])

        # Test with 1D gradient array
        result = kl_divergence_loss(predictions, targets, reduction="none")
        grad_output = np.array([2.0])
        result._grad_fn.apply(grad_output)

        # Test with higher-dimensional gradient
        predictions_2d = Tensor([[[1.0, 2.0], [0.5, 1.5]]], requires_grad=True)
        targets_2d = Tensor([[[0.4, 0.6], [0.3, 0.7]]])
        result_2d = kl_divergence_loss(predictions_2d, targets_2d, reduction="none")
        grad_output_2d = np.array([[1.0, 0.5]])
        result_2d._grad_fn.apply(grad_output_2d)

    def test_cosine_embedding_loss_gradient_edge_cases(self):
        """Test cosine embedding loss gradient edge cases."""
        input1 = Tensor([[1.0, 2.0]], requires_grad=True)
        input2 = Tensor([[2.0, 1.0]], requires_grad=True)

        # Test with dissimilar pairs and margin
        target_dissimilar = Tensor([-1])
        result = cosine_embedding_loss(input1, input2, target_dissimilar, margin=0.5)

        # Test gradient with array output
        grad_output = np.array(2.0)
        result._grad_fn.apply(grad_output)

        # Test case where cosine similarity <= margin (no gradient)
        input1_far = Tensor([[1.0, 0.0]], requires_grad=True)
        input2_far = Tensor([[-1.0, 0.0]], requires_grad=True)  # Cosine similarity = -1
        target_dissimilar = Tensor([-1])

        result_no_grad = cosine_embedding_loss(
            input1_far, input2_far, target_dissimilar, margin=0.5
        )
        grad_output = np.array(1.0)
        result_no_grad._grad_fn.apply(grad_output)

    def test_triplet_loss_edge_cases(self):
        """Test triplet loss edge cases."""
        anchor = Tensor([[1.0, 2.0]], requires_grad=True)
        positive = Tensor([[1.1, 2.1]], requires_grad=True)
        negative = Tensor([[5.0, 6.0]], requires_grad=True)

        # Test with different reduction modes
        result_mean = triplet_loss(anchor, positive, negative, reduction="mean")
        result_sum = triplet_loss(anchor, positive, negative, reduction="sum")
        result_none = triplet_loss(anchor, positive, negative, reduction="none")

        assert result_mean.shape == ()
        assert result_sum.shape == ()
        assert result_none.shape == (1,)

    def test_triplet_loss_gradient_with_scaling(self):
        """Test triplet loss gradient with different scaling."""
        anchor = Tensor([[1.0, 2.0]], requires_grad=True)
        positive = Tensor([[1.2, 2.2]], requires_grad=True)  # Close to anchor
        negative = Tensor([[1.1, 2.1]], requires_grad=True)  # Also close (active triplet)

        result = triplet_loss(anchor, positive, negative, margin=0.5, reduction="sum")

        # Test gradient with array output
        grad_output = np.array(3.0)
        result._grad_fn.apply(grad_output)

    def test_triplet_loss_lp_norm_gradients(self):
        """Test triplet loss gradients for different Lp norms."""
        anchor = Tensor([[1.0, 2.0]], requires_grad=True)
        positive = Tensor([[1.3, 2.3]], requires_grad=True)
        negative = Tensor([[1.1, 2.1]], requires_grad=True)

        # Test L1 norm gradients
        result_l1 = triplet_loss(anchor, positive, negative, p=1.0, margin=0.1)
        grad_output = np.array(1.0)
        result_l1._grad_fn.apply(grad_output)

        # Test L2 norm gradients
        result_l2 = triplet_loss(anchor, positive, negative, p=2.0, margin=0.1)
        grad_output = np.array(1.0)
        result_l2._grad_fn.apply(grad_output)

        # Test general Lp norm gradients
        result_lp = triplet_loss(anchor, positive, negative, p=3.0, margin=0.1)
        grad_output = np.array(1.0)
        result_lp._grad_fn.apply(grad_output)

    def test_loss_functions_without_gradient_chaining(self):
        """Test loss functions without gradient function chaining."""
        predictions = Tensor([[2.0, 1.0]], requires_grad=True)
        targets = Tensor([0])

        # Remove _backward method to test path without chaining
        if hasattr(predictions, "_backward"):
            delattr(predictions, "_backward")

        result = cross_entropy_loss(predictions, targets)

        # Simulate backward pass
        grad_output = np.array(1.0)
        result._grad_fn.apply(grad_output)

        # Should still work without chaining
        assert isinstance(result, Tensor)

    def test_loss_functions_memory_efficiency(self):
        """Test that loss functions use memory efficient operations."""
        # Large tensors to test memory efficiency
        large_size = 1000
        predictions = Tensor(np.random.randn(large_size, 10))
        targets = Tensor(np.random.randint(0, 10, (large_size,)))

        with patch("neural_arch.functional.loss.logger") as mock_logger:
            # These should complete without memory issues due to decorator
            ce_result = cross_entropy_loss(predictions, targets)
            mse_result = mse_loss(predictions, targets)
            focal_result = focal_loss(predictions, targets)

            # Should log operation start and completion
            assert mock_logger.debug.call_count >= 6  # 2 calls per function

    def test_loss_functions_backward_chaining(self):
        """Test loss functions with backward chaining."""
        predictions = Tensor([[2.0, 1.0]], requires_grad=True)
        targets = Tensor([[1.0]], requires_grad=True)

        # Mock _backward methods
        predictions._backward = MagicMock()
        targets._backward = MagicMock()

        result = mse_loss(predictions, targets)

        # Simulate backward pass
        grad_output = np.array(1.0)
        result._grad_fn.apply(grad_output)

        # Backward chaining should be called
        predictions._backward.assert_called_once()
        targets._backward.assert_called_once()

    def test_loss_numerical_stability_extreme_values(self):
        """Test loss functions with extreme values for numerical stability."""
        # Very large positive and negative logits
        extreme_predictions = Tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
        targets = Tensor([0, 1])

        # Test all loss functions with extreme values
        ce_result = cross_entropy_loss(extreme_predictions, targets)
        focal_result = focal_loss(extreme_predictions, targets, gamma=2.0)
        ls_result = label_smoothing_cross_entropy(extreme_predictions, targets, smoothing=0.1)

        # All should produce finite results
        assert np.isfinite(ce_result.data)
        assert np.isfinite(focal_result.data)
        assert np.isfinite(ls_result.data)

        # Test KL divergence with extreme values
        extreme_log_probs = Tensor([[-1000.0, -1000.0]])
        extreme_probs = Tensor([[0.5, 0.5]])
        kl_result = kl_divergence_loss(extreme_log_probs, extreme_probs)
        assert np.isfinite(kl_result.data)

    def test_cosine_embedding_loss_numerical_stability(self):
        """Test cosine embedding loss numerical stability."""
        # Very small vectors (near zero)
        input1 = Tensor([[1e-10, 1e-10]], requires_grad=True)
        input2 = Tensor([[1e-10, -1e-10]], requires_grad=True)
        target = Tensor([1])

        result = cosine_embedding_loss(input1, input2, target)

        # Should handle near-zero vectors due to epsilon in norms
        assert np.isfinite(result.data)

        # Test gradient computation
        grad_output = np.array(1.0)
        result._grad_fn.apply(grad_output)

    def test_triplet_loss_numerical_stability(self):
        """Test triplet loss numerical stability."""
        # Very small differences
        anchor = Tensor([[0.0, 0.0]], requires_grad=True)
        positive = Tensor([[1e-10, 1e-10]], requires_grad=True)
        negative = Tensor([[1e-9, 1e-9]], requires_grad=True)

        # Test L2 norm with very small values
        result = triplet_loss(anchor, positive, negative, p=2.0, margin=0.1)

        # Should handle small values due to epsilon in distance computation
        assert np.isfinite(result.data)

        # Test gradient computation
        grad_output = np.array(1.0)
        result._grad_fn.apply(grad_output)

    def test_loss_error_handling_in_memory_decorator(self):
        """Test error handling in memory efficient decorator for loss functions."""
        with patch("neural_arch.functional.loss.logger") as mock_logger:
            # Mock a loss function that raises an error
            def failing_loss(predictions, targets):
                raise RuntimeError("Test error")

            # Apply decorator
            from neural_arch.functional.utils import memory_efficient_operation

            decorated_fn = memory_efficient_operation(failing_loss)

            with pytest.raises(RuntimeError, match="Test error"):
                decorated_fn(Tensor([1.0]), Tensor([0]))

            # Should log error
            mock_logger.error.assert_called_once()

    def test_all_loss_functions_logging(self):
        """Test that all loss functions log debug information."""
        predictions = Tensor([[2.0, 1.0]])
        targets = Tensor([0])

        with patch("neural_arch.functional.loss.logger") as mock_logger:
            # Test all loss functions
            cross_entropy_loss(predictions, targets)
            mse_loss(predictions, targets)
            focal_loss(predictions, targets)
            label_smoothing_cross_entropy(predictions, targets)
            huber_loss(predictions, targets)
            kl_divergence_loss(predictions, targets)

            # Test embedding losses
            input1 = Tensor([[1.0, 2.0]])
            input2 = Tensor([[2.0, 1.0]])
            target = Tensor([1])
            cosine_embedding_loss(input1, input2, target)

            # Test triplet loss
            anchor = Tensor([[1.0, 2.0]])
            positive = Tensor([[1.1, 2.1]])
            negative = Tensor([[3.0, 4.0]])
            triplet_loss(anchor, positive, negative)

            # Should have many debug calls
            assert mock_logger.debug.call_count >= 8

    def test_loss_gradient_function_properties(self):
        """Test that loss gradient functions have proper properties."""
        predictions = Tensor([[2.0, 1.0]], requires_grad=True)
        targets = Tensor([0])

        result = cross_entropy_loss(predictions, targets)

        # Check gradient function properties
        assert result._grad_fn is not None
        assert hasattr(result._grad_fn, "apply")
        assert hasattr(result._grad_fn, "inputs")
        assert hasattr(result._grad_fn, "name")
        assert len(result._grad_fn.inputs) == 2  # predictions and targets
        assert result._grad_fn.name == "cross_entropy_loss"
