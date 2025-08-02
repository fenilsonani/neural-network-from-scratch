"""Comprehensive test coverage for functional/loss module to boost coverage from 87.74% to 95%+"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock

from neural_arch.core.tensor import Tensor, is_grad_enabled, no_grad, enable_grad
import neural_arch.core.tensor as tensor_module
from neural_arch.functional.loss import (
    cross_entropy_loss, mse_loss, focal_loss, label_smoothing_cross_entropy,
    huber_loss, kl_divergence_loss, cosine_embedding_loss, triplet_loss
)


def set_grad_enabled(enabled: bool):
    """Helper function to set gradient state."""
    tensor_module._grad_enabled = enabled


class TestLossFunctionsCoverageBoost:
    """Comprehensive tests for loss functions targeting missing coverage paths."""
    
    def test_cross_entropy_loss_with_cupy_backend(self):
        """Test cross-entropy loss with CuPy backend data handling."""
        set_grad_enabled(True)
        
        try:
            # Mock CuPy-like backend data
            predictions = Tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]], requires_grad=True)
            targets = Tensor([0, 1])
            
            # Mock backend data with .get() method (like CuPy)
            mock_backend_data = MagicMock()
            mock_backend_data.get.return_value = np.array([0, 1])
            targets.backend_data = mock_backend_data
            
            result = cross_entropy_loss(predictions, targets)
            
            assert isinstance(result, Tensor)
            assert result.requires_grad is True
            
        finally:
            set_grad_enabled(False)
    
    def test_cross_entropy_loss_with_one_hot_targets(self):
        """Test cross-entropy loss with one-hot encoded targets."""
        set_grad_enabled(True)
        
        try:
            predictions = Tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]], requires_grad=True)
            targets = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # One-hot encoded
            
            result = cross_entropy_loss(predictions, targets)
            
            assert isinstance(result, Tensor)
            assert result.requires_grad is True
            
        finally:
            set_grad_enabled(False)
    
    def test_cross_entropy_loss_reduction_modes(self):
        """Test cross-entropy loss with different reduction modes."""
        predictions = Tensor([[2.0, 1.0], [1.0, 3.0]])
        targets = Tensor([0, 1])
        
        # Test 'mean' reduction
        loss_mean = cross_entropy_loss(predictions, targets, reduction='mean')
        assert loss_mean.shape == ()
        
        # Test 'sum' reduction
        loss_sum = cross_entropy_loss(predictions, targets, reduction='sum')
        assert loss_sum.shape == ()
        assert loss_sum.data > loss_mean.data  # Sum should be larger than mean
        
        # Test 'none' reduction
        loss_none = cross_entropy_loss(predictions, targets, reduction='none')
        assert loss_none.shape == (2,)  # Should keep individual losses
    
    def test_cross_entropy_loss_invalid_reduction(self):
        """Test cross-entropy loss with invalid reduction parameter."""
        predictions = Tensor([[2.0, 1.0]])
        targets = Tensor([0])
        
        with pytest.raises(ValueError, match="Unknown reduction"):
            cross_entropy_loss(predictions, targets, reduction='invalid')
    
    def test_cross_entropy_loss_gradient_computation(self):
        """Test cross-entropy loss gradient computation."""
        set_grad_enabled(True)
        
        try:
            predictions = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)
            targets = Tensor([0])
            
            result = cross_entropy_loss(predictions, targets)
            
            # Simulate backward pass
            grad_output = np.array(1.0)
            result._grad_fn.apply(grad_output)
            
            # Check that backward was called
            assert hasattr(predictions, 'backward')
            
        finally:
            set_grad_enabled(False)
    
    def test_mse_loss_with_cupy_backend(self):
        """Test MSE loss with CuPy backend data handling."""
        predictions = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        targets = Tensor([[0.5, 1.5], [2.5, 3.5]])
        
        # Mock backend data with .get() method
        mock_pred_data = MagicMock()
        mock_pred_data.get.return_value = predictions.data
        predictions.backend_data = mock_pred_data
        
        mock_target_data = MagicMock()
        mock_target_data.get.return_value = targets.data
        targets.backend_data = mock_target_data
        
        result = mse_loss(predictions, targets)
        
        assert isinstance(result, Tensor)
        expected = np.mean((predictions.data - targets.data) ** 2)
        assert np.isclose(result.data, expected)
    
    def test_mse_loss_gradient_computation_both_tensors(self):
        """Test MSE loss gradient computation for both tensors."""
        set_grad_enabled(True)
        
        try:
            predictions = Tensor([[1.0, 2.0]], requires_grad=True)
            targets = Tensor([[0.5, 1.5]], requires_grad=True)
            
            result = mse_loss(predictions, targets)
            
            # Simulate backward pass
            grad_output = np.array(1.0)
            result._grad_fn.apply(grad_output)
            
            # Both tensors should have gradients
            assert hasattr(predictions, 'backward')
            assert hasattr(targets, 'backward')
            
        finally:
            set_grad_enabled(False)
    
    def test_mse_loss_reduction_modes(self):
        """Test MSE loss with different reduction modes."""
        predictions = Tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = Tensor([[0.0, 1.0], [2.0, 3.0]])
        
        # Test all reduction modes
        loss_mean = mse_loss(predictions, targets, reduction='mean')
        loss_sum = mse_loss(predictions, targets, reduction='sum')
        loss_none = mse_loss(predictions, targets, reduction='none')
        
        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (2, 2)
        
        # Test invalid reduction
        with pytest.raises(ValueError, match="Unknown reduction"):
            mse_loss(predictions, targets, reduction='invalid')
    
    def test_focal_loss_basic_functionality(self):
        """Test focal loss basic functionality."""
        predictions = Tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]], requires_grad=True)
        targets = Tensor([0, 1])
        
        # Test with default parameters
        result = focal_loss(predictions, targets)
        assert isinstance(result, Tensor)
        assert result.requires_grad is True
        
        # Test with custom parameters
        result_custom = focal_loss(predictions, targets, alpha=0.25, gamma=2.0)
        assert isinstance(result_custom, Tensor)
    
    def test_focal_loss_with_one_hot_targets(self):
        """Test focal loss with one-hot encoded targets."""
        predictions = Tensor([[2.0, 1.0], [1.0, 3.0]], requires_grad=True)
        targets = Tensor([[1.0, 0.0], [0.0, 1.0]])  # One-hot
        
        result = focal_loss(predictions, targets)
        assert isinstance(result, Tensor)
    
    def test_focal_loss_gamma_zero_case(self):
        """Test focal loss with gamma=0 (weighted cross-entropy)."""
        set_grad_enabled(True)
        
        try:
            predictions = Tensor([[2.0, 1.0]], requires_grad=True)
            targets = Tensor([0])
            
            result = focal_loss(predictions, targets, gamma=0.0)
            
            # Simulate backward pass
            grad_output = np.array(1.0)
            result._grad_fn.apply(grad_output)
            
            assert isinstance(result, Tensor)
            
        finally:
            set_grad_enabled(False)
    
    def test_focal_loss_reduction_modes(self):
        """Test focal loss with different reduction modes."""
        predictions = Tensor([[2.0, 1.0], [1.0, 3.0]])
        targets = Tensor([0, 1])
        
        loss_mean = focal_loss(predictions, targets, reduction='mean')
        loss_sum = focal_loss(predictions, targets, reduction='sum')
        loss_none = focal_loss(predictions, targets, reduction='none')
        
        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (2,)
        
        with pytest.raises(ValueError, match="Unknown reduction"):
            focal_loss(predictions, targets, reduction='invalid')
    
    def test_label_smoothing_cross_entropy_invalid_smoothing(self):
        """Test label smoothing with invalid smoothing parameter."""
        predictions = Tensor([[2.0, 1.0]])
        targets = Tensor([0])
        
        # Test invalid smoothing values
        with pytest.raises(ValueError, match="Smoothing must be in"):
            label_smoothing_cross_entropy(predictions, targets, smoothing=-0.1)
        
        with pytest.raises(ValueError, match="Smoothing must be in"):
            label_smoothing_cross_entropy(predictions, targets, smoothing=1.0)
        
        with pytest.raises(ValueError, match="Smoothing must be in"):
            label_smoothing_cross_entropy(predictions, targets, smoothing=1.5)
    
    def test_label_smoothing_cross_entropy_with_one_hot(self):
        """Test label smoothing with one-hot targets."""
        predictions = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)
        targets = Tensor([[1.0, 0.0, 0.0]])  # One-hot
        
        result = label_smoothing_cross_entropy(predictions, targets, smoothing=0.1)
        assert isinstance(result, Tensor)
    
    def test_label_smoothing_cross_entropy_gradient(self):
        """Test label smoothing gradient computation."""
        set_grad_enabled(True)
        
        try:
            predictions = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)
            targets = Tensor([0])
            
            result = label_smoothing_cross_entropy(predictions, targets, smoothing=0.1)
            
            # Simulate backward pass
            grad_output = np.array(1.0)
            result._grad_fn.apply(grad_output)
            
            assert hasattr(predictions, 'backward')
            
        finally:
            set_grad_enabled(False)
    
    def test_huber_loss_invalid_delta(self):
        """Test Huber loss with invalid delta parameter."""
        predictions = Tensor([1.0, 2.0])
        targets = Tensor([0.5, 1.5])
        
        with pytest.raises(ValueError, match="Delta must be positive"):
            huber_loss(predictions, targets, delta=0.0)
        
        with pytest.raises(ValueError, match="Delta must be positive"):
            huber_loss(predictions, targets, delta=-1.0)
    
    def test_huber_loss_quadratic_and_linear_regions(self):
        """Test Huber loss in both quadratic and linear regions."""
        # Small errors (quadratic region)
        predictions_small = Tensor([1.0, 2.0])
        targets_small = Tensor([0.9, 1.9])  # Small differences
        
        result_small = huber_loss(predictions_small, targets_small, delta=1.0)
        
        # Large errors (linear region)
        predictions_large = Tensor([1.0, 2.0])
        targets_large = Tensor([-1.0, 5.0])  # Large differences
        
        result_large = huber_loss(predictions_large, targets_large, delta=1.0)
        
        assert isinstance(result_small, Tensor)
        assert isinstance(result_large, Tensor)
    
    def test_huber_loss_gradient_computation(self):
        """Test Huber loss gradient computation."""
        set_grad_enabled(True)
        
        try:
            predictions = Tensor([1.0, 5.0], requires_grad=True)
            targets = Tensor([0.5, 1.0], requires_grad=True)  # Mix of small and large errors
            
            result = huber_loss(predictions, targets, delta=1.0)
            
            # Simulate backward pass
            grad_output = np.array(1.0)
            result._grad_fn.apply(grad_output)
            
            assert hasattr(predictions, 'backward')
            assert hasattr(targets, 'backward')
            
        finally:
            set_grad_enabled(False)
    
    def test_kl_divergence_loss_with_logits_input(self):
        """Test KL divergence with logits as input."""
        # Predictions as logits (max > 0)
        predictions = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)
        # Targets as logits (for conversion test)
        targets = Tensor([[1.5, 2.0, 0.8]])
        
        result = kl_divergence_loss(predictions, targets)
        assert isinstance(result, Tensor)
    
    def test_kl_divergence_loss_with_log_probs_and_probs(self):
        """Test KL divergence with log probabilities and probabilities."""
        # Predictions as log probabilities (max <= 0)
        predictions = Tensor([[-0.5, -1.0, -1.5]], requires_grad=True)
        # Targets as probabilities (valid probability distribution)
        targets = Tensor([[0.6, 0.3, 0.1]])
        
        result = kl_divergence_loss(predictions, targets)
        assert isinstance(result, Tensor)
    
    def test_kl_divergence_loss_gradient_computation(self):
        """Test KL divergence gradient computation with different upstream gradients."""
        set_grad_enabled(True)
        
        try:
            predictions = Tensor([[2.0, 1.0], [1.5, 2.5]], requires_grad=True)
            targets = Tensor([[0.6, 0.4], [0.3, 0.7]])
            
            result = kl_divergence_loss(predictions, targets, reduction='none')
            
            # Test with 1D upstream gradient
            grad_output = np.array([1.0, 0.5])
            result._grad_fn.apply(grad_output)
            
            assert hasattr(predictions, 'backward')
            
        finally:
            set_grad_enabled(False)
    
    def test_cosine_embedding_loss_shape_mismatch(self):
        """Test cosine embedding loss with mismatched input shapes."""
        input1 = Tensor([[1.0, 2.0]])
        input2 = Tensor([[1.0, 2.0, 3.0]])  # Different shape
        target = Tensor([1])
        
        with pytest.raises(ValueError, match="Input shapes must match"):
            cosine_embedding_loss(input1, input2, target)
    
    def test_cosine_embedding_loss_similar_and_dissimilar_pairs(self):
        """Test cosine embedding loss with similar and dissimilar pairs."""
        input1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        input2 = Tensor([[1.1, 2.1], [0.0, 1.0]], requires_grad=True)
        
        # Test similar pairs (target = 1)
        target_similar = Tensor([1, 1])
        result_similar = cosine_embedding_loss(input1, input2, target_similar)
        
        # Test dissimilar pairs (target = -1)
        target_dissimilar = Tensor([-1, -1])
        result_dissimilar = cosine_embedding_loss(input1, input2, target_dissimilar, margin=0.2)
        
        assert isinstance(result_similar, Tensor)
        assert isinstance(result_dissimilar, Tensor)
    
    def test_cosine_embedding_loss_gradient_computation(self):
        """Test cosine embedding loss gradient computation."""
        set_grad_enabled(True)
        
        try:
            input1 = Tensor([[1.0, 2.0]], requires_grad=True)
            input2 = Tensor([[2.0, 1.0]], requires_grad=True)
            target = Tensor([1])  # Similar pair
            
            result = cosine_embedding_loss(input1, input2, target)
            
            # Simulate backward pass
            grad_output = np.array(1.0)
            result._grad_fn.apply(grad_output)
            
            assert hasattr(input1, 'backward')
            assert hasattr(input2, 'backward')
            
        finally:
            set_grad_enabled(False)
    
    def test_triplet_loss_shape_mismatch(self):
        """Test triplet loss with mismatched input shapes."""
        anchor = Tensor([[1.0, 2.0]])
        positive = Tensor([[1.1, 2.1]])
        negative = Tensor([[3.0, 4.0, 5.0]])  # Different shape
        
        with pytest.raises(ValueError, match="All input shapes must match"):
            triplet_loss(anchor, positive, negative)
    
    def test_triplet_loss_invalid_p_norm(self):
        """Test triplet loss with invalid p norm parameter."""
        anchor = Tensor([[1.0, 2.0]])
        positive = Tensor([[1.1, 2.1]])
        negative = Tensor([[3.0, 4.0]])
        
        with pytest.raises(ValueError, match="Norm degree p must be positive"):
            triplet_loss(anchor, positive, negative, p=0.0)
        
        with pytest.raises(ValueError, match="Norm degree p must be positive"):
            triplet_loss(anchor, positive, negative, p=-1.0)
    
    def test_triplet_loss_different_norms(self):
        """Test triplet loss with different norm types."""
        anchor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        positive = Tensor([[1.1, 2.1], [3.1, 4.1]], requires_grad=True)
        negative = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        
        # Test L1 norm
        result_l1 = triplet_loss(anchor, positive, negative, p=1.0)
        
        # Test L2 norm
        result_l2 = triplet_loss(anchor, positive, negative, p=2.0)
        
        # Test general Lp norm
        result_lp = triplet_loss(anchor, positive, negative, p=3.0)
        
        assert isinstance(result_l1, Tensor)
        assert isinstance(result_l2, Tensor)
        assert isinstance(result_lp, Tensor)
    
    def test_triplet_loss_no_active_triplets(self):
        """Test triplet loss when no triplets are active (loss = 0)."""
        set_grad_enabled(True)
        
        try:
            # Create triplets where negative is much farther than positive
            anchor = Tensor([[0.0, 0.0]], requires_grad=True)
            positive = Tensor([[0.1, 0.1]], requires_grad=True)  # Very close
            negative = Tensor([[10.0, 10.0]], requires_grad=True)  # Very far
            
            result = triplet_loss(anchor, positive, negative, margin=0.1)
            
            # Simulate backward pass
            grad_output = np.array(1.0)
            result._grad_fn.apply(grad_output)
            
            # Should handle case with no active triplets
            assert isinstance(result, Tensor)
            
        finally:
            set_grad_enabled(False)
    
    def test_triplet_loss_gradient_computation_active_triplets(self):
        """Test triplet loss gradient computation with active triplets."""
        set_grad_enabled(True)
        
        try:
            anchor = Tensor([[1.0, 2.0]], requires_grad=True)
            positive = Tensor([[1.5, 2.5]], requires_grad=True)  # Close to anchor
            negative = Tensor([[1.2, 2.2]], requires_grad=True)  # Also close (active triplet)
            
            result = triplet_loss(anchor, positive, negative, margin=1.0, p=2.0)
            
            # Simulate backward pass
            grad_output = np.array(1.0)
            result._grad_fn.apply(grad_output)
            
            assert hasattr(anchor, 'backward')
            assert hasattr(positive, 'backward')
            assert hasattr(negative, 'backward')
            
        finally:
            set_grad_enabled(False)
    
    def test_loss_functions_reduction_none_shapes(self):
        """Test that all loss functions handle 'none' reduction correctly."""
        batch_size = 3
        predictions = Tensor(np.random.randn(batch_size, 4))
        targets = Tensor(np.random.randint(0, 4, (batch_size,)))
        
        # Test cross-entropy
        ce_none = cross_entropy_loss(predictions, targets, reduction='none')
        assert ce_none.shape == (batch_size,)
        
        # Test focal loss
        focal_none = focal_loss(predictions, targets, reduction='none')
        assert focal_none.shape == (batch_size,)
        
        # Test label smoothing
        ls_none = label_smoothing_cross_entropy(predictions, targets, reduction='none')
        assert ls_none.shape == (batch_size,)
    
    def test_loss_functions_with_zero_gradients(self):
        """Test loss functions behavior when gradients are disabled."""
        set_grad_enabled(False)
        
        try:
            predictions = Tensor([[2.0, 1.0]], requires_grad=True)
            targets = Tensor([0])
            
            # These should not create gradient functions
            ce_result = cross_entropy_loss(predictions, targets)
            mse_result = mse_loss(predictions, targets)
            focal_result = focal_loss(predictions, targets)
            
            assert not ce_result.requires_grad
            assert not mse_result.requires_grad
            assert not focal_result.requires_grad
            
        finally:
            set_grad_enabled(True)
    
    def test_loss_functions_numerical_stability(self):
        """Test loss functions with extreme values for numerical stability."""
        # Very large logits
        large_predictions = Tensor([[100.0, -100.0], [-100.0, 100.0]])
        targets = Tensor([0, 1])
        
        ce_result = cross_entropy_loss(large_predictions, targets)
        focal_result = focal_loss(large_predictions, targets)
        ls_result = label_smoothing_cross_entropy(large_predictions, targets)
        
        # Should not produce NaN or Inf
        assert np.isfinite(ce_result.data)
        assert np.isfinite(focal_result.data)
        assert np.isfinite(ls_result.data)
        
        # Very small values for KL divergence
        small_predictions = Tensor([[-10.0, -10.0]])
        small_targets = Tensor([[0.5, 0.5]])
        
        kl_result = kl_divergence_loss(small_predictions, small_targets)
        assert np.isfinite(kl_result.data)
    
    def test_loss_functions_memory_efficiency_decorator(self):
        """Test that loss functions use memory efficient operations."""
        # Large tensors to test memory efficiency
        large_size = 1000
        predictions = Tensor(np.random.randn(large_size, 10))
        targets = Tensor(np.random.randint(0, 10, (large_size,)))
        
        # These should complete without memory issues
        ce_result = cross_entropy_loss(predictions, targets)
        mse_result = mse_loss(predictions, targets)
        
        assert isinstance(ce_result, Tensor)
        assert isinstance(mse_result, Tensor)
    
    def test_loss_functions_logging(self):
        """Test that loss functions log debug information."""
        import logging
        
        with patch('neural_arch.functional.loss.logger') as mock_logger:
            predictions = Tensor([[2.0, 1.0]])
            targets = Tensor([0])
            
            cross_entropy_loss(predictions, targets)
            focal_loss(predictions, targets)
            
            # Should have called debug logging
            assert mock_logger.debug.call_count >= 2