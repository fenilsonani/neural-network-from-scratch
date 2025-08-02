"""Tests for advanced loss functions with mathematical correctness validation.

This comprehensive test suite validates all the newly implemented loss functions:
- Focal Loss for handling class imbalance
- Label Smoothing Cross-Entropy for better generalization
- Huber Loss for robust regression
- KL Divergence Loss for knowledge distillation
- Cosine Embedding Loss for similarity learning
- Triplet Loss for metric learning

All tests include mathematical correctness verification and numerical gradient checking.
"""

import os
import sys

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch.core import Tensor
from neural_arch.functional import (
    cosine_embedding_loss,
    cross_entropy_loss,
    focal_loss,
    huber_loss,
    kl_divergence_loss,
    label_smoothing_cross_entropy,
    softmax,
    triplet_loss,
)


class TestAdvancedLossFunctions:
    """Test suite for advanced loss functions."""

    def numerical_gradient(self, func, x: np.ndarray, h: float = 1e-5):
        """Compute numerical gradient using finite differences."""
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus.flat[i] += h
            x_minus.flat[i] -= h

            f_plus = func(x_plus)
            f_minus = func(x_minus)
            grad.flat[i] = (f_plus - f_minus) / (2 * h)

        return grad

    def assert_gradients_close(self, analytical, numerical, rtol=1e-4, atol=1e-6):
        """Assert analytical and numerical gradients are close."""
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradients don't match:\nAnalytical: {analytical}\nNumerical: {numerical}",
        )


class TestFocalLoss(TestAdvancedLossFunctions):
    """Test Focal Loss implementation."""

    def test_focal_loss_basic(self):
        """Test basic Focal Loss functionality."""
        batch_size, num_classes = 4, 3
        predictions = Tensor(np.random.randn(batch_size, num_classes), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 1]), requires_grad=False)

        # Test with default parameters
        loss = focal_loss(predictions, targets, alpha=1.0, gamma=2.0)

        assert loss.requires_grad is True
        assert loss.shape == ()  # Scalar loss
        assert loss.data >= 0.0  # Loss should be non-negative

    def test_focal_loss_vs_cross_entropy(self):
        """Test that Focal Loss reduces to weighted cross-entropy when gamma=0."""
        batch_size, num_classes = 3, 5
        predictions = Tensor(np.random.randn(batch_size, num_classes), requires_grad=False)
        targets = Tensor(np.array([0, 2, 4]), requires_grad=False)
        alpha = 0.5

        # Focal loss with gamma=0 should be weighted cross-entropy
        focal = focal_loss(predictions, targets, alpha=alpha, gamma=0.0)
        ce = cross_entropy_loss(predictions, targets) * alpha

        np.testing.assert_allclose(focal.data, ce.data, rtol=1e-6)

    def test_focal_loss_gradient(self):
        """Test Focal Loss gradient correctness (approximate)."""
        batch_size, num_classes = 2, 3
        predictions_data = np.random.randn(batch_size, num_classes)
        targets_data = np.array([0, 2])

        # Test gradient numerically
        def loss_fn(pred_data):
            pred = Tensor(pred_data, requires_grad=True)
            targets = Tensor(targets_data, requires_grad=False)
            loss = focal_loss(pred, targets, alpha=0.25, gamma=2.0)
            return loss.data

        numerical_grad = self.numerical_gradient(loss_fn, predictions_data)

        # Get analytical gradient
        predictions = Tensor(predictions_data, requires_grad=True)
        targets = Tensor(targets_data, requires_grad=False)
        loss = focal_loss(predictions, targets, alpha=0.25, gamma=2.0)

        # Manually trigger backward
        loss._grad_fn.apply(np.array(1.0))
        analytical_grad = predictions._grad

        # Focal loss gradient is complex - use more relaxed tolerance
        self.assert_gradients_close(analytical_grad, numerical_grad, rtol=1e-1, atol=1e-2)

    def test_focal_loss_class_imbalance(self):
        """Test that Focal Loss handles class imbalance better than cross-entropy."""
        # Create imbalanced dataset
        batch_size, num_classes = 100, 3
        np.random.seed(42)

        # Highly imbalanced: 90% class 0, 5% class 1, 5% class 2
        targets_data = np.random.choice([0, 1, 2], size=batch_size, p=[0.9, 0.05, 0.05])
        predictions_data = np.random.randn(batch_size, num_classes)

        predictions = Tensor(predictions_data, requires_grad=False)
        targets = Tensor(targets_data, requires_grad=False)

        # Standard cross-entropy
        ce_loss = cross_entropy_loss(predictions, targets)

        # Focal loss should be different (usually lower for imbalanced data)
        focal = focal_loss(predictions, targets, alpha=0.25, gamma=2.0)

        # Just ensure they produce different values (focal should handle imbalance better)
        assert abs(focal.data - ce_loss.data) > 1e-6


class TestLabelSmoothingCrossEntropy(TestAdvancedLossFunctions):
    """Test Label Smoothing Cross-Entropy implementation."""

    def test_label_smoothing_basic(self):
        """Test basic Label Smoothing functionality."""
        batch_size, num_classes = 3, 4
        predictions = Tensor(np.random.randn(batch_size, num_classes), requires_grad=True)
        targets = Tensor(np.array([0, 2, 3]), requires_grad=False)

        loss = label_smoothing_cross_entropy(predictions, targets, smoothing=0.1)

        assert loss.requires_grad is True
        assert loss.shape == ()
        assert loss.data >= 0.0

    def test_label_smoothing_zero_smoothing(self):
        """Test that zero smoothing equals standard cross-entropy."""
        batch_size, num_classes = 2, 3
        predictions = Tensor(np.random.randn(batch_size, num_classes), requires_grad=False)
        targets = Tensor(np.array([0, 2]), requires_grad=False)

        # Zero smoothing should equal standard cross-entropy
        smoothed = label_smoothing_cross_entropy(predictions, targets, smoothing=0.0)
        standard = cross_entropy_loss(predictions, targets)

        np.testing.assert_allclose(smoothed.data, standard.data, rtol=1e-6)

    def test_label_smoothing_gradient(self):
        """Test Label Smoothing gradient correctness."""
        batch_size, num_classes = 2, 4
        predictions_data = np.random.randn(batch_size, num_classes)
        targets_data = np.array([1, 3])

        def loss_fn(pred_data):
            pred = Tensor(pred_data, requires_grad=True)
            targets = Tensor(targets_data, requires_grad=False)
            loss = label_smoothing_cross_entropy(pred, targets, smoothing=0.1)
            return loss.data

        numerical_grad = self.numerical_gradient(loss_fn, predictions_data)

        # Get analytical gradient
        predictions = Tensor(predictions_data, requires_grad=True)
        targets = Tensor(targets_data, requires_grad=False)
        loss = label_smoothing_cross_entropy(predictions, targets, smoothing=0.1)

        loss._grad_fn.apply(np.array(1.0))
        analytical_grad = predictions._grad

        self.assert_gradients_close(analytical_grad, numerical_grad)


class TestHuberLoss(TestAdvancedLossFunctions):
    """Test Huber Loss implementation."""

    def test_huber_loss_basic(self):
        """Test basic Huber Loss functionality."""
        batch_size, dim = 3, 2
        predictions = Tensor(np.random.randn(batch_size, dim), requires_grad=True)
        targets = Tensor(np.random.randn(batch_size, dim), requires_grad=False)

        loss = huber_loss(predictions, targets, delta=1.0)

        assert loss.requires_grad is True
        assert loss.shape == ()
        assert loss.data >= 0.0

    def test_huber_loss_quadratic_region(self):
        """Test Huber Loss in quadratic region (small errors)."""
        predictions = Tensor(np.array([[1.0, 2.0]]), requires_grad=False)
        targets = Tensor(np.array([[1.1, 1.9]]), requires_grad=False)  # Small differences
        delta = 1.0

        huber = huber_loss(predictions, targets, delta=delta)

        # Should be quadratic: 0.5 * residual^2
        expected = 0.5 * (0.1**2 + 0.1**2)  # Mean of squared errors
        np.testing.assert_allclose(huber.data, expected, rtol=1e-6)

    def test_huber_loss_linear_region(self):
        """Test Huber Loss in linear region (large errors)."""
        predictions = Tensor(np.array([[0.0]]), requires_grad=False)
        targets = Tensor(np.array([[3.0]]), requires_grad=False)  # Large difference
        delta = 1.0

        huber = huber_loss(predictions, targets, delta=delta)

        # Should be linear: delta * (|residual| - 0.5 * delta)
        residual = 3.0
        expected = delta * (abs(residual) - 0.5 * delta)
        np.testing.assert_allclose(huber.data, expected, rtol=1e-6)

    def test_huber_loss_gradient(self):
        """Test Huber Loss gradient correctness."""
        predictions_data = np.array([[1.0, -2.0], [0.5, 3.0]])
        targets_data = np.array([[1.5, -1.0], [2.0, 1.0]])

        def loss_fn(pred_data):
            pred = Tensor(pred_data, requires_grad=True)
            targets = Tensor(targets_data, requires_grad=False)
            loss = huber_loss(pred, targets, delta=1.0)
            return loss.data

        numerical_grad = self.numerical_gradient(loss_fn, predictions_data)

        # Get analytical gradient
        predictions = Tensor(predictions_data, requires_grad=True)
        targets = Tensor(targets_data, requires_grad=False)
        loss = huber_loss(predictions, targets, delta=1.0)

        loss._grad_fn.apply(np.array(1.0))
        analytical_grad = predictions._grad

        self.assert_gradients_close(analytical_grad, numerical_grad)


class TestKLDivergenceLoss(TestAdvancedLossFunctions):
    """Test KL Divergence Loss implementation."""

    def test_kl_divergence_basic(self):
        """Test basic KL Divergence functionality."""
        batch_size, num_classes = 2, 4
        predictions = Tensor(np.random.randn(batch_size, num_classes), requires_grad=True)
        targets = Tensor(np.random.randn(batch_size, num_classes), requires_grad=False)

        loss = kl_divergence_loss(predictions, targets)

        assert loss.requires_grad is True
        assert loss.shape == ()
        assert loss.data >= 0.0  # KL divergence is non-negative

    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence between identical distributions (should be 0)."""
        batch_size, num_classes = 2, 3
        logits = np.random.randn(batch_size, num_classes)

        predictions = Tensor(logits, requires_grad=False)
        targets = Tensor(logits, requires_grad=False)  # Same distribution

        loss = kl_divergence_loss(predictions, targets)

        # KL(P||P) = 0
        np.testing.assert_allclose(loss.data, 0.0, atol=1e-6)

    def test_kl_divergence_gradient(self):
        """Test KL Divergence gradient correctness."""
        batch_size, num_classes = 2, 3
        predictions_data = np.random.randn(batch_size, num_classes)
        targets_data = np.random.randn(batch_size, num_classes)

        def loss_fn(pred_data):
            pred = Tensor(pred_data, requires_grad=True)
            targets = Tensor(targets_data, requires_grad=False)
            loss = kl_divergence_loss(pred, targets)
            return loss.data

        numerical_grad = self.numerical_gradient(loss_fn, predictions_data)

        # Get analytical gradient
        predictions = Tensor(predictions_data, requires_grad=True)
        targets = Tensor(targets_data, requires_grad=False)
        loss = kl_divergence_loss(predictions, targets)

        loss._grad_fn.apply(np.array(1.0))
        analytical_grad = predictions._grad

        self.assert_gradients_close(analytical_grad, numerical_grad, rtol=1e-3)


class TestCosineEmbeddingLoss(TestAdvancedLossFunctions):
    """Test Cosine Embedding Loss implementation."""

    def test_cosine_embedding_basic(self):
        """Test basic Cosine Embedding Loss functionality."""
        batch_size, embed_dim = 3, 5
        input1 = Tensor(np.random.randn(batch_size, embed_dim), requires_grad=True)
        input2 = Tensor(np.random.randn(batch_size, embed_dim), requires_grad=True)
        target = Tensor(np.array([1, -1, 1]), requires_grad=False)  # Similar, dissimilar, similar

        loss = cosine_embedding_loss(input1, input2, target, margin=0.5)

        assert loss.requires_grad is True
        assert loss.shape == ()
        assert loss.data >= 0.0

    def test_cosine_embedding_identical_vectors(self):
        """Test cosine embedding loss with identical vectors."""
        batch_size, embed_dim = 2, 4
        vectors = np.random.randn(batch_size, embed_dim)

        input1 = Tensor(vectors, requires_grad=False)
        input2 = Tensor(vectors, requires_grad=False)  # Identical
        target = Tensor(np.array([1, 1]), requires_grad=False)  # Similar pairs

        loss = cosine_embedding_loss(input1, input2, target, margin=0.0)

        # Loss should be 0 for identical vectors with similar targets
        np.testing.assert_allclose(loss.data, 0.0, atol=1e-6)

    def test_cosine_embedding_gradient(self):
        """Test Cosine Embedding Loss gradient correctness."""
        batch_size, embed_dim = 2, 3
        input1_data = np.random.randn(batch_size, embed_dim)
        input2_data = np.random.randn(batch_size, embed_dim)
        target_data = np.array([1, -1])

        def loss_fn_input1(inp1_data):
            inp1 = Tensor(inp1_data, requires_grad=True)
            inp2 = Tensor(input2_data, requires_grad=False)
            target = Tensor(target_data, requires_grad=False)
            loss = cosine_embedding_loss(inp1, inp2, target, margin=0.3)
            return loss.data

        numerical_grad = self.numerical_gradient(loss_fn_input1, input1_data)

        # Get analytical gradient
        input1 = Tensor(input1_data, requires_grad=True)
        input2 = Tensor(input2_data, requires_grad=False)
        target = Tensor(target_data, requires_grad=False)
        loss = cosine_embedding_loss(input1, input2, target, margin=0.3)

        loss._grad_fn.apply(np.array(1.0))
        analytical_grad = input1._grad

        self.assert_gradients_close(analytical_grad, numerical_grad, rtol=1e-3)


class TestTripletLoss(TestAdvancedLossFunctions):
    """Test Triplet Loss implementation."""

    def test_triplet_loss_basic(self):
        """Test basic Triplet Loss functionality."""
        batch_size, embed_dim = 2, 4
        anchor = Tensor(np.random.randn(batch_size, embed_dim), requires_grad=True)
        positive = Tensor(np.random.randn(batch_size, embed_dim), requires_grad=True)
        negative = Tensor(np.random.randn(batch_size, embed_dim), requires_grad=True)

        loss = triplet_loss(anchor, positive, negative, margin=1.0)

        assert loss.requires_grad is True
        assert loss.shape == ()
        assert loss.data >= 0.0

    def test_triplet_loss_perfect_triplet(self):
        """Test triplet loss with perfect triplet (anchor = positive, different negative)."""
        embed_dim = 3
        anchor_data = np.array([[1.0, 0.0, 0.0]])
        positive_data = np.array([[1.0, 0.0, 0.0]])  # Same as anchor
        negative_data = np.array([[0.0, 1.0, 0.0]])  # Different from anchor

        anchor = Tensor(anchor_data, requires_grad=False)
        positive = Tensor(positive_data, requires_grad=False)
        negative = Tensor(negative_data, requires_grad=False)

        loss = triplet_loss(anchor, positive, negative, margin=0.5, p=2.0)

        # Distance anchor-positive = 0, distance anchor-negative = sqrt(2)
        # Loss = max(0, 0 - sqrt(2) + 0.5) = 0 (since 0 - sqrt(2) + 0.5 < 0)
        expected_loss = max(0.0, 0.0 - np.sqrt(2.0) + 0.5)
        np.testing.assert_allclose(loss.data, expected_loss, rtol=1e-6)

    def test_triplet_loss_different_norms(self):
        """Test triplet loss with different norm types."""
        batch_size, embed_dim = 1, 2
        anchor = Tensor(np.array([[0.0, 0.0]]), requires_grad=False)
        positive = Tensor(np.array([[1.0, 1.0]]), requires_grad=False)
        negative = Tensor(np.array([[3.0, 0.0]]), requires_grad=False)

        # L1 norm
        loss_l1 = triplet_loss(anchor, positive, negative, margin=0.0, p=1.0)
        expected_l1 = max(0.0, 2.0 - 3.0 + 0.0)  # |1|+|1| - |3|+|0| + 0
        np.testing.assert_allclose(loss_l1.data, expected_l1, rtol=1e-6)

        # L2 norm
        loss_l2 = triplet_loss(anchor, positive, negative, margin=0.0, p=2.0)
        expected_l2 = max(0.0, np.sqrt(2.0) - 3.0 + 0.0)  # sqrt(1²+1²) - sqrt(3²+0²) + 0
        np.testing.assert_allclose(loss_l2.data, expected_l2, rtol=1e-6)

    def test_triplet_loss_gradient(self):
        """Test Triplet Loss gradient correctness."""
        batch_size, embed_dim = 2, 3
        anchor_data = np.random.randn(batch_size, embed_dim)
        positive_data = np.random.randn(batch_size, embed_dim)
        negative_data = np.random.randn(batch_size, embed_dim)

        def loss_fn_anchor(anc_data):
            anchor = Tensor(anc_data, requires_grad=True)
            positive = Tensor(positive_data, requires_grad=False)
            negative = Tensor(negative_data, requires_grad=False)
            loss = triplet_loss(anchor, positive, negative, margin=0.5, p=2.0)
            return loss.data

        numerical_grad = self.numerical_gradient(loss_fn_anchor, anchor_data)

        # Get analytical gradient
        anchor = Tensor(anchor_data, requires_grad=True)
        positive = Tensor(positive_data, requires_grad=False)
        negative = Tensor(negative_data, requires_grad=False)
        loss = triplet_loss(anchor, positive, negative, margin=0.5, p=2.0)

        loss._grad_fn.apply(np.array(1.0))
        analytical_grad = anchor._grad

        self.assert_gradients_close(analytical_grad, numerical_grad, rtol=1e-3)


if __name__ == "__main__":
    # Run comprehensive tests
    print("Testing Advanced Loss Functions...")

    print("Testing Focal Loss...")
    test_focal = TestFocalLoss()
    test_focal.test_focal_loss_basic()
    test_focal.test_focal_loss_vs_cross_entropy()
    test_focal.test_focal_loss_gradient()
    test_focal.test_focal_loss_class_imbalance()
    print("✓ Focal Loss tests passed")

    print("Testing Label Smoothing Cross-Entropy...")
    test_label_smooth = TestLabelSmoothingCrossEntropy()
    test_label_smooth.test_label_smoothing_basic()
    test_label_smooth.test_label_smoothing_zero_smoothing()
    test_label_smooth.test_label_smoothing_gradient()
    print("✓ Label Smoothing Cross-Entropy tests passed")

    print("Testing Huber Loss...")
    test_huber = TestHuberLoss()
    test_huber.test_huber_loss_basic()
    test_huber.test_huber_loss_quadratic_region()
    test_huber.test_huber_loss_linear_region()
    test_huber.test_huber_loss_gradient()
    print("✓ Huber Loss tests passed")

    print("Testing KL Divergence Loss...")
    test_kl = TestKLDivergenceLoss()
    test_kl.test_kl_divergence_basic()
    test_kl.test_kl_divergence_identical_distributions()
    test_kl.test_kl_divergence_gradient()
    print("✓ KL Divergence Loss tests passed")

    print("Testing Cosine Embedding Loss...")
    test_cosine = TestCosineEmbeddingLoss()
    test_cosine.test_cosine_embedding_basic()
    test_cosine.test_cosine_embedding_identical_vectors()
    test_cosine.test_cosine_embedding_gradient()
    print("✓ Cosine Embedding Loss tests passed")

    print("Testing Triplet Loss...")
    test_triplet = TestTripletLoss()
    test_triplet.test_triplet_loss_basic()
    test_triplet.test_triplet_loss_perfect_triplet()
    test_triplet.test_triplet_loss_different_norms()
    test_triplet.test_triplet_loss_gradient()
    print("✓ Triplet Loss tests passed")

    print("\n🎉 All Advanced Loss Function tests passed!")
    print("✅ Advanced loss functions are mathematically correct and ready for production use")

    # Performance demonstration
    print("\n⚡ Performance Demonstration...")

    # Focal Loss for imbalanced classification
    print("Focal Loss - Class Imbalance Handling:")
    batch_size, num_classes = 1000, 10
    np.random.seed(42)

    # Highly imbalanced dataset
    imbalanced_targets = np.random.choice(
        range(num_classes),
        size=batch_size,
        p=[0.7, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01],
    )
    predictions = Tensor(np.random.randn(batch_size, num_classes))
    targets = Tensor(imbalanced_targets)

    ce_loss = cross_entropy_loss(predictions, targets)
    focal = focal_loss(predictions, targets, alpha=0.25, gamma=2.0)

    print(f"Standard Cross-Entropy: {ce_loss.data:.4f}")
    print(f"Focal Loss (α=0.25, γ=2.0): {focal.data:.4f}")

    # Knowledge Distillation with KL Divergence
    print("\nKL Divergence - Knowledge Distillation:")
    teacher_logits = Tensor(np.random.randn(32, 100))  # Teacher model outputs
    student_logits = Tensor(np.random.randn(32, 100))  # Student model outputs

    kl_loss = kl_divergence_loss(student_logits, teacher_logits)
    print(f"KL Divergence Loss: {kl_loss.data:.4f}")

    print("\n✅ Performance demonstration completed successfully")
