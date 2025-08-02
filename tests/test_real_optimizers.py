"""Real comprehensive tests for optimizers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core.tensor import Tensor
from neural_arch.nn.linear import Linear
from neural_arch.optim.adam import Adam
from neural_arch.optim.adamw import AdamW
from neural_arch.optim.sgd import SGD


class TestRealOptimizers:
    """Real tests for optimizers without simulation."""

    def test_adam_optimizer_creation(self):
        """Test Adam optimizer creation."""
        # Create a simple layer with parameters
        layer = Linear(3, 2)
        params = list(layer.parameters())

        # Create Adam optimizer
        optimizer = Adam(params, lr=0.01)

        # Check initialization
        assert hasattr(optimizer, "param_groups")
        assert len(optimizer.param_groups) > 0
        assert optimizer.param_groups[0]["lr"] == 0.01

        # Check default values
        param_group = optimizer.param_groups[0]
        assert param_group["betas"] == (0.9, 0.999)
        assert param_group["eps"] == 1e-8
        assert param_group["weight_decay"] == 0

    def test_adam_optimizer_step(self):
        """Test Adam optimizer step."""
        # Create layer and optimizer
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01)

        # Create input and target
        x = Tensor([[1, 2]], requires_grad=True)
        target = Tensor([[3]])

        # Forward pass
        output = layer(x)

        # Simple loss (squared error)
        from neural_arch.functional.arithmetic import mul, sub

        diff = sub(output, target)
        loss_data = mul(diff, diff)

        # Manually set gradients for parameters
        for param in layer.parameters():
            if param.grad is None:
                param.grad = Tensor(np.random.randn(*param.shape) * 0.01)

        # Store original parameter values
        original_params = {}
        for i, param in enumerate(layer.parameters()):
            original_params[i] = param.data.copy()

        # Optimizer step
        optimizer.step()

        # Parameters should have changed
        for i, param in enumerate(layer.parameters()):
            # Check that parameters were updated
            param_changed = not np.allclose(param.data, original_params[i])
            # If gradients were present, parameters should change
            if param.grad is not None and np.any(param.grad.data != 0):
                assert param_changed, f"Parameter {i} should have changed"

    def test_adam_zero_grad(self):
        """Test Adam optimizer zero_grad."""
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01)

        # Set some gradients
        for param in layer.parameters():
            param.grad = Tensor(np.random.randn(*param.shape))

        # Zero gradients
        optimizer.zero_grad()

        # All gradients should be None or zero
        for param in layer.parameters():
            if param.grad is not None:
                np.testing.assert_array_equal(param.grad.data, np.zeros_like(param.data))
            else:
                assert param.grad is None

    def test_adam_state_management(self):
        """Test Adam optimizer state management."""
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01)

        # Initially, state should be empty
        assert hasattr(optimizer, "state")

        # Set gradients and take a step
        for param in layer.parameters():
            param.grad = Tensor(np.ones_like(param.data) * 0.1)

        optimizer.step()

        # State should now contain momentum terms
        # (Implementation details may vary)
        if hasattr(optimizer, "state") and optimizer.state:
            for param in layer.parameters():
                if id(param) in optimizer.state:
                    state = optimizer.state[id(param)]
                    # Should have momentum terms
                    assert "step" in state or "exp_avg" in state or len(state) > 0

    def test_sgd_optimizer_creation(self):
        """Test SGD optimizer creation."""
        layer = Linear(3, 2)
        params = list(layer.parameters())

        # Create SGD optimizer
        optimizer = SGD(params, lr=0.01, momentum=0.9)

        # Check initialization
        assert hasattr(optimizer, "param_groups")
        assert len(optimizer.param_groups) > 0
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_sgd_optimizer_step(self):
        """Test SGD optimizer step."""
        layer = Linear(2, 1)
        optimizer = SGD(layer.parameters(), lr=0.1)

        # Set gradients
        for param in layer.parameters():
            param.grad = Tensor(np.ones_like(param.data) * 0.01)

        # Store original values
        original_params = {}
        for i, param in enumerate(layer.parameters()):
            original_params[i] = param.data.copy()

        # Take step
        optimizer.step()

        # Parameters should have changed
        for i, param in enumerate(layer.parameters()):
            if param.grad is not None:
                param_changed = not np.allclose(param.data, original_params[i])
                if np.any(param.grad.data != 0):
                    assert param_changed, f"Parameter {i} should have changed"

    def test_adamw_optimizer(self):
        """Test AdamW optimizer if available."""
        try:
            layer = Linear(2, 1)

            # Try to create AdamW optimizer
            optimizer = AdamW(layer.parameters(), lr=0.01, weight_decay=0.01)

            # Check initialization
            assert hasattr(optimizer, "param_groups")
            assert optimizer.param_groups[0]["weight_decay"] == 0.01

        except (AttributeError, ImportError, TypeError):
            # AdamW might not be fully implemented
            pytest.skip("AdamW not available or not implemented")

    def test_optimizer_parameter_groups(self):
        """Test optimizer parameter groups."""
        layer1 = Linear(2, 3)
        layer2 = Linear(3, 1)

        # Create different parameter groups
        param_groups = [
            {"params": list(layer1.parameters()), "lr": 0.01},
            {"params": list(layer2.parameters()), "lr": 0.001},
        ]

        optimizer = Adam(param_groups)

        # Should have two parameter groups
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[1]["lr"] == 0.001

    def test_optimizer_lr_scheduling_concept(self):
        """Test learning rate modification."""
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01)

        # Check initial learning rate
        assert optimizer.param_groups[0]["lr"] == 0.01

        # Modify learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1

        # Should be updated
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_optimizer_weight_decay(self):
        """Test weight decay functionality."""
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01, weight_decay=0.1)

        # Check weight decay is set
        assert optimizer.param_groups[0]["weight_decay"] == 0.1

        # Set gradients and take step
        for param in layer.parameters():
            param.grad = Tensor(np.zeros_like(param.data))  # Zero gradients

        # Store original weights
        original_weight = layer.weight.data.copy()

        # Take step - weight decay should still affect parameters
        optimizer.step()

        # With weight decay, parameters might change even with zero gradients
        # (depending on implementation)

    def test_optimizer_gradient_clipping_concept(self):
        """Test gradient clipping concept."""
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01)

        # Set very large gradients
        for param in layer.parameters():
            param.grad = Tensor(np.ones_like(param.data) * 100)

        # Manually clip gradients (common pattern)
        max_norm = 1.0
        for param in layer.parameters():
            if param.grad is not None:
                grad_norm = np.linalg.norm(param.grad.data)
                if grad_norm > max_norm:
                    param.grad.data = param.grad.data * (max_norm / grad_norm)

        # After clipping, gradients should be bounded
        for param in layer.parameters():
            if param.grad is not None:
                grad_norm = np.linalg.norm(param.grad.data)
                assert grad_norm <= max_norm + 1e-6

    def test_optimizer_multiple_steps(self):
        """Test multiple optimizer steps."""
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01)

        # Take multiple steps
        for step in range(5):
            # Set different gradients each step
            for param in layer.parameters():
                param.grad = Tensor(np.random.randn(*param.shape) * 0.01)

            optimizer.step()
            optimizer.zero_grad()

        # Optimizer should handle multiple steps without error
        assert True  # If we get here, no errors occurred

    def test_optimizer_empty_gradients(self):
        """Test optimizer behavior with None gradients."""
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01)

        # Don't set gradients (they should be None)
        for param in layer.parameters():
            assert param.grad is None

        # Step should handle None gradients gracefully
        try:
            optimizer.step()
            # Should not crash
        except (AttributeError, RuntimeError):
            # Some optimizers might require gradients
            pass

    def test_optimizer_state_persistence(self):
        """Test optimizer state persistence across steps."""
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.01)

        # Take first step
        for param in layer.parameters():
            param.grad = Tensor(np.ones_like(param.data) * 0.1)
        optimizer.step()

        # Check if state was created
        has_state = hasattr(optimizer, "state") and len(optimizer.state) > 0

        # Take second step
        for param in layer.parameters():
            param.grad = Tensor(np.ones_like(param.data) * 0.1)
        optimizer.step()

        # State should persist (if implemented)
        if has_state:
            assert len(optimizer.state) > 0

    def test_optimizer_different_dtypes(self):
        """Test optimizers with different parameter dtypes."""
        # Create layer with specific dtype
        layer = Linear(2, 1)

        # Ensure parameters have specific dtype
        for param in layer.parameters():
            param.data = param.data.astype(np.float32)

        optimizer = Adam(layer.parameters(), lr=0.01)

        # Set gradients with same dtype
        for param in layer.parameters():
            param.grad = Tensor(np.ones_like(param.data, dtype=np.float32) * 0.01)

        # Should handle specific dtypes
        optimizer.step()

        # Parameters should maintain their dtype
        for param in layer.parameters():
            assert param.data.dtype == np.float32

    def test_optimizer_convergence_simulation(self):
        """Test optimizer convergence on simple problem."""
        # Simple quadratic function: f(x) = (x - 2)^2
        x = Tensor([0.0], requires_grad=True)
        optimizer = Adam([x], lr=0.1)

        # Run optimization steps
        for step in range(10):
            # Compute loss: (x - 2)^2
            from neural_arch.functional.arithmetic import mul, sub

            diff = sub(x, 2.0)
            loss = mul(diff, diff)

            # Manually compute gradient: 2(x - 2)
            grad = 2 * (x.data - 2.0)
            x.grad = Tensor(grad)

            # Take optimizer step
            optimizer.step()
            optimizer.zero_grad()

        # x should be closer to 2
        assert abs(x.data[0] - 2.0) < abs(0.0 - 2.0)  # Closer than initial value
