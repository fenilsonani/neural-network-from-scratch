"""Tests for learning rate schedulers.

This module contains comprehensive tests for all learning rate scheduling
strategies to ensure correct behavior and mathematical accuracy.
"""

import os
import sys

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch.core import Parameter
from neural_arch.exceptions import OptimizerError
from neural_arch.optim import (
    AdamW,
    ChainedScheduler,
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
    WarmupLR,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class TestLRSchedulers:
    """Test suite for learning rate schedulers."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create dummy parameters
        self.weights = Parameter(np.random.randn(5, 3))
        self.bias = Parameter(np.zeros(3))
        self.parameters = {"weights": self.weights, "bias": self.bias}

        # Create optimizer
        self.optimizer = AdamW(self.parameters, lr=0.1)

    def test_step_lr(self):
        """Test StepLR scheduler."""
        scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)

        # Test initial learning rate
        assert abs(self.optimizer.lr - 0.1) < 1e-6

        # Test learning rate schedule
        expected_lrs = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025]

        for i, expected_lr in enumerate(expected_lrs):
            if i > 0:
                scheduler.step()
            assert (
                abs(self.optimizer.lr - expected_lr) < 1e-6
            ), f"Step {i}: expected {expected_lr}, got {self.optimizer.lr}"

    def test_exponential_lr(self):
        """Test ExponentialLR scheduler."""
        scheduler = ExponentialLR(self.optimizer, gamma=0.9)

        # Test exponential decay
        initial_lr = 0.1
        for epoch in range(5):
            expected_lr = initial_lr * (0.9**epoch)
            assert abs(self.optimizer.lr - expected_lr) < 1e-6
            scheduler.step()

    def test_cosine_annealing_lr(self):
        """Test CosineAnnealingLR scheduler."""
        T_max = 10
        eta_min = 0.01
        scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)

        # Test cosine annealing formula
        initial_lr = 0.1

        for epoch in range(T_max + 1):
            if epoch == 0:
                expected_lr = initial_lr
            else:
                progress = epoch / T_max
                cosine_factor = (1 + np.cos(np.pi * progress)) / 2
                expected_lr = eta_min + (initial_lr - eta_min) * cosine_factor

            assert (
                abs(self.optimizer.lr - expected_lr) < 1e-6
            ), f"Epoch {epoch}: expected {expected_lr}, got {self.optimizer.lr}"
            scheduler.step()

    def test_linear_lr(self):
        """Test LinearLR scheduler."""
        start_factor = 0.5
        end_factor = 1.0
        total_iters = 5
        scheduler = LinearLR(
            self.optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
        )

        initial_lr = 0.1

        for epoch in range(total_iters + 2):
            if epoch == 0:
                expected_lr = initial_lr * start_factor
            elif epoch >= total_iters:
                expected_lr = initial_lr * end_factor
            else:
                progress = epoch / total_iters
                factor = start_factor + (end_factor - start_factor) * progress
                expected_lr = initial_lr * factor

            assert (
                abs(self.optimizer.lr - expected_lr) < 1e-6
            ), f"Epoch {epoch}: expected {expected_lr}, got {self.optimizer.lr}"
            scheduler.step()

    def test_warmup_lr(self):
        """Test WarmupLR scheduler."""
        warmup_epochs = 3
        warmup_factor = 0.1
        scheduler = WarmupLR(
            self.optimizer, warmup_epochs=warmup_epochs, warmup_factor=warmup_factor
        )

        initial_lr = 0.1

        # Test warmup phase
        for epoch in range(warmup_epochs + 2):
            if epoch < warmup_epochs:
                alpha = epoch / warmup_epochs
                factor = warmup_factor + alpha * (1 - warmup_factor)
                expected_lr = initial_lr * factor
            else:
                expected_lr = initial_lr

            assert (
                abs(self.optimizer.lr - expected_lr) < 1e-6
            ), f"Epoch {epoch}: expected {expected_lr}, got {self.optimizer.lr}"
            scheduler.step()

    def test_polynomial_lr(self):
        """Test PolynomialLR scheduler."""
        total_iters = 5
        power = 2.0
        end_lr = 0.01
        scheduler = PolynomialLR(
            self.optimizer, total_iters=total_iters, power=power, end_lr=end_lr
        )

        initial_lr = 0.1

        for epoch in range(total_iters + 2):
            if epoch == 0:
                expected_lr = initial_lr
            elif epoch >= total_iters:
                expected_lr = end_lr
            else:
                decay_factor = (1 - epoch / total_iters) ** power
                expected_lr = (initial_lr - end_lr) * decay_factor + end_lr

            assert (
                abs(self.optimizer.lr - expected_lr) < 1e-6
            ), f"Epoch {epoch}: expected {expected_lr}, got {self.optimizer.lr}"
            scheduler.step()

    def test_reduce_lr_on_plateau(self):
        """Test ReduceLROnPlateau scheduler."""
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4
        )

        initial_lr = 0.1

        # Step with first metric to initialize best value
        scheduler.step(1.0)
        assert abs(self.optimizer.lr - initial_lr) < 1e-6

        # Test with improving metrics (no reduction)
        metrics = [0.9, 0.8, 0.7]
        for i, metric in enumerate(metrics):
            scheduler.step(metric)
            assert (
                abs(self.optimizer.lr - initial_lr) < 1e-6
            ), f"Step {i+1}: lr should not change with improving metrics"

        # Test with plateau (should reduce lr)
        for i in range(3):  # patience = 2, so should reduce on 3rd step
            scheduler.step(0.7)
            if i < 2:
                assert abs(self.optimizer.lr - initial_lr) < 1e-6
            else:
                assert abs(self.optimizer.lr - initial_lr * 0.5) < 1e-6

    def test_chained_scheduler(self):
        """Test ChainedScheduler."""
        # Create individual schedulers
        warmup_scheduler = WarmupLR(self.optimizer, warmup_epochs=2)
        step_scheduler = StepLR(self.optimizer, step_size=2, gamma=0.5)

        # Chain them
        chained = ChainedScheduler([warmup_scheduler, step_scheduler], milestones=[2])

        # Test that appropriate scheduler is used at each epoch
        for epoch in range(6):
            chained.step(epoch)
            # Just verify it doesn't crash and lr is reasonable
            assert 0 < self.optimizer.lr <= 0.1

    def test_convenience_functions(self):
        """Test convenience functions for common patterns."""
        # Test linear schedule with warmup
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=2, num_training_steps=10
        )

        # Test cosine schedule with warmup
        scheduler2 = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=2, num_training_steps=10
        )

        # Just verify they create valid schedulers
        assert isinstance(scheduler, ChainedScheduler)
        assert isinstance(scheduler2, ChainedScheduler)

    def test_scheduler_state_dict(self):
        """Test scheduler state dict functionality."""
        scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)

        # Step a few times
        scheduler.step()
        scheduler.step()

        # Save state
        state_dict = scheduler.state_dict()
        assert "last_epoch" in state_dict
        assert state_dict["last_epoch"] == 2

        # Create new scheduler and load state
        new_scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)
        new_scheduler.load_state_dict(state_dict)
        assert new_scheduler.last_epoch == 2

    def test_error_handling(self):
        """Test error handling for invalid parameters."""
        # Test invalid step_size
        with pytest.raises(OptimizerError):
            StepLR(self.optimizer, step_size=0, gamma=0.5)

        # Test invalid gamma
        with pytest.raises(OptimizerError):
            StepLR(self.optimizer, step_size=3, gamma=1.5)

        # Test invalid T_max
        with pytest.raises(OptimizerError):
            CosineAnnealingLR(self.optimizer, T_max=0)

        # Test invalid mode for ReduceLROnPlateau
        with pytest.raises(OptimizerError):
            ReduceLROnPlateau(self.optimizer, mode="invalid")

    def test_scheduler_with_different_optimizers(self):
        """Test that schedulers work with different optimizers."""
        from neural_arch.optim import Adam, Lion

        # Test with Adam
        adam_optimizer = Adam(self.parameters, lr=0.01)
        scheduler = StepLR(adam_optimizer, step_size=2, gamma=0.8)

        initial_lr = adam_optimizer.lr
        scheduler.step()
        scheduler.step()  # Should trigger step
        assert abs(adam_optimizer.lr - initial_lr * 0.8) < 1e-6

        # Test with Lion
        lion_optimizer = Lion(self.parameters, lr=0.0001)
        scheduler = CosineAnnealingLR(lion_optimizer, T_max=5, eta_min=1e-6)

        # Just verify it works without crashing
        for _ in range(5):
            scheduler.step()
            assert lion_optimizer.lr >= 1e-6  # Should be at least eta_min


if __name__ == "__main__":
    # Run comprehensive tests
    print("Testing Learning Rate Schedulers...")

    test_suite = TestLRSchedulers()

    try:
        test_suite.setup_method()
        test_suite.test_step_lr()
        print("✓ StepLR test passed")

        test_suite.setup_method()
        test_suite.test_exponential_lr()
        print("✓ ExponentialLR test passed")

        test_suite.setup_method()
        test_suite.test_cosine_annealing_lr()
        print("✓ CosineAnnealingLR test passed")

        test_suite.setup_method()
        test_suite.test_linear_lr()
        print("✓ LinearLR test passed")

        test_suite.setup_method()
        test_suite.test_warmup_lr()
        print("✓ WarmupLR test passed")

        test_suite.setup_method()
        test_suite.test_polynomial_lr()
        print("✓ PolynomialLR test passed")

        test_suite.setup_method()
        test_suite.test_reduce_lr_on_plateau()
        print("✓ ReduceLROnPlateau test passed")

        test_suite.setup_method()
        test_suite.test_chained_scheduler()
        print("✓ ChainedScheduler test passed")

        test_suite.setup_method()
        test_suite.test_convenience_functions()
        print("✓ Convenience functions test passed")

        test_suite.setup_method()
        test_suite.test_scheduler_state_dict()
        print("✓ State dict test passed")

        test_suite.setup_method()
        test_suite.test_scheduler_with_different_optimizers()
        print("✓ Different optimizers test passed")

        print("\n🎉 All Learning Rate Scheduler tests passed!")
        print("✅ Learning rate scheduling is now fully implemented and tested")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise
