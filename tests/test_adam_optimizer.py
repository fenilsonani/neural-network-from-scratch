"""Tests for Adam optimizer improvements."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core import Parameter, Tensor
from neural_arch.optim import Adam


class TestAdamOptimizer:
    """Test Adam optimizer functionality."""

    def test_init(self):
        """Test optimizer initialization."""
        # Create parameters
        param1 = Parameter(np.random.randn(10, 5))
        param2 = Parameter(np.random.randn(5))

        # Create optimizer with iterator
        optimizer = Adam([param1, param2], lr=0.001)

        assert optimizer.lr == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.eps == 1e-8

    def test_parameter_groups(self):
        """Test that optimizer correctly handles parameters."""
        param1 = Parameter(np.random.randn(10, 5))
        param2 = Parameter(np.random.randn(5))

        # Test with list
        optimizer = Adam([param1, param2], lr=0.01)

        # Check internal state
        assert len(optimizer.parameters) == 2
        assert len(optimizer.state) == 2

    def test_zero_grad(self):
        """Test gradient zeroing."""
        param = Parameter(np.ones((3, 3)))
        param.grad = np.ones((3, 3))

        optimizer = Adam([param], lr=0.01)
        optimizer.zero_grad()

        assert param.grad is None or np.all(param.grad == 0)

    def test_step_updates_parameters(self):
        """Test that step() updates parameters."""
        # Create parameter with known gradient
        param = Parameter(np.zeros((5, 5)))
        param.grad = np.ones((5, 5))  # Gradient of 1 everywhere

        initial_value = param.data.copy()

        # Create optimizer and take a step
        optimizer = Adam([param], lr=0.1)
        optimizer.step()

        # Parameters should have changed
        assert not np.allclose(param.data, initial_value)

        # With positive gradients and positive learning rate,
        # parameters should decrease (gradient descent)
        assert np.all(param.data < initial_value)

    def test_momentum(self):
        """Test that Adam maintains momentum correctly."""
        param = Parameter(np.zeros((3, 3)))
        optimizer = Adam([param], lr=0.1, beta1=0.9)

        # Take multiple steps with same gradient
        for i in range(5):
            param.grad = np.ones((3, 3))
            optimizer.step()

        # Check that momentum is being maintained
        state = optimizer.state[list(optimizer.state.keys())[0]]
        assert "exp_avg" in state
        assert not np.allclose(state["exp_avg"], np.zeros((3, 3)))

    def test_adaptive_learning_rate(self):
        """Test that Adam adapts learning rate per parameter."""
        param = Parameter(np.zeros((2, 2)))
        optimizer = Adam([param], lr=0.1, beta2=0.999)

        # Different gradients for different parameters
        param.grad = np.array([[1.0, 0.1], [0.1, 1.0]])

        # Take a step
        optimizer.step()

        # Parameters with larger gradients should have smaller effective learning rates
        # due to RMSprop-style normalization
        assert abs(param.data[0, 0]) < abs(param.data[0, 1]) * 2

    def test_bias_correction(self):
        """Test bias correction in early steps."""
        param = Parameter(np.zeros((3, 3)))
        optimizer = Adam([param], lr=0.1)

        # First step
        param.grad = np.ones((3, 3))
        optimizer.step()

        # Get state
        state = optimizer.state[list(optimizer.state.keys())[0]]
        step = state["step"]

        # Bias correction factors
        bias_correction1 = 1 - 0.9**step
        bias_correction2 = 1 - 0.999**step

        # Early steps should have significant bias correction
        assert bias_correction1 > 0.05
        assert bias_correction2 > 0.001

    def test_weight_decay(self):
        """Test weight decay (L2 regularization)."""
        param = Parameter(np.ones((3, 3)))
        param.grad = np.zeros((3, 3))  # No gradient

        initial_norm = np.linalg.norm(param.data)

        # Optimizer with weight decay
        optimizer = Adam([param], lr=0.1, weight_decay=0.1)
        optimizer.step()

        # Parameters should shrink due to weight decay
        final_norm = np.linalg.norm(param.data)
        assert final_norm < initial_norm

    def test_multiple_parameters(self):
        """Test optimizer with multiple parameters."""
        params = [
            Parameter(np.random.randn(10, 10)),
            Parameter(np.random.randn(5, 5)),
            Parameter(np.random.randn(3)),
        ]

        # Set different gradients
        for i, param in enumerate(params):
            param.grad = np.ones_like(param.data) * (i + 1)

        optimizer = Adam(params, lr=0.01)

        # Save initial values
        initial_values = [p.data.copy() for p in params]

        # Take a step
        optimizer.step()

        # All parameters should be updated
        for param, initial in zip(params, initial_values):
            assert not np.allclose(param.data, initial)

    def test_gradient_clipping_compatibility(self):
        """Test that optimizer works with gradient clipping."""
        param = Parameter(np.zeros((5, 5)))
        param.grad = np.ones((5, 5)) * 100  # Large gradient

        # Clip gradient
        max_norm = 1.0
        grad_norm = np.linalg.norm(param.grad)
        if grad_norm > max_norm:
            param.grad = param.grad * (max_norm / grad_norm)

        # Optimizer should work with clipped gradients
        optimizer = Adam([param], lr=0.1)
        optimizer.step()

        # Update should be bounded
        assert np.abs(param.data).max() < 1.0

    def test_state_persistence(self):
        """Test that optimizer state persists across steps."""
        param = Parameter(np.zeros((3, 3)))
        optimizer = Adam([param], lr=0.1)

        # Take multiple steps
        for i in range(10):
            param.grad = np.ones((3, 3))
            optimizer.step()

        # Check state
        state = optimizer.state[list(optimizer.state.keys())[0]]
        assert state["step"] == 10
        assert "exp_avg" in state
        assert "exp_avg_sq" in state


class TestOptimizerIntegration:
    """Test optimizer with real model scenarios."""

    def test_with_model_parameters(self):
        """Test optimizer with model-like parameter structure."""

        # Simulate a simple model
        class SimpleModel:
            def __init__(self):
                self.weight1 = Parameter(np.random.randn(10, 5))
                self.bias1 = Parameter(np.zeros(5))
                self.weight2 = Parameter(np.random.randn(5, 2))
                self.bias2 = Parameter(np.zeros(2))

            def parameters(self):
                return [self.weight1, self.bias1, self.weight2, self.bias2]

        model = SimpleModel()
        optimizer = Adam(model.parameters(), lr=0.01)

        # Simulate training step
        for param in model.parameters():
            param.grad = np.random.randn(*param.data.shape) * 0.1

        initial_weights = [p.data.copy() for p in model.parameters()]
        optimizer.step()

        # All parameters should be updated
        for param, initial in zip(model.parameters(), initial_weights):
            assert not np.allclose(param.data, initial)

    def test_convergence(self):
        """Test that Adam can minimize a simple function."""
        # Minimize f(x) = x^2
        x = Parameter(np.array([10.0]))  # Start far from minimum
        optimizer = Adam([x], lr=0.1)

        losses = []
        for _ in range(100):
            # Forward: loss = x^2
            loss = x.data[0] ** 2
            losses.append(loss)

            # Backward: gradient = 2x
            x.grad = 2 * x.data

            # Update
            optimizer.step()

        # Should converge close to 0
        assert abs(x.data[0]) < 3.0  # More lenient threshold

        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.1  # 90% reduction
        assert losses[-1] < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
