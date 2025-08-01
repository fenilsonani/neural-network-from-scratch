"""Test suite for Lion optimizer."""

import numpy as np
import pytest
from unittest.mock import patch

from src.neural_arch.optim.lion import Lion
from src.neural_arch.core.base import Parameter
from src.neural_arch.exceptions import OptimizerError


class TestLionOptimizer:
    """Test suite for Lion optimizer implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test parameters
        self.param1 = Parameter(np.array([1.0, 2.0, 3.0]), name="weight")
        self.param2 = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]), name="bias")
        self.parameters = {"weight": self.param1, "bias": self.param2}
        
        # Set gradients
        self.param1.grad = np.array([0.1, -0.2, 0.3])
        self.param2.grad = np.array([[0.1, 0.2], [-0.1, 0.3]])
    
    def test_lion_initialization_default_params(self):
        """Test Lion optimizer initialization with default parameters."""
        optimizer = Lion(self.parameters)
        
        assert optimizer.lr == 1e-4  # Lion's default LR is much smaller than Adam
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.99
        assert optimizer.weight_decay == 0.0
        assert optimizer.maximize == False
        
        # Check state initialization
        assert len(optimizer.state) == 2
        assert "weight" in optimizer.state
        assert "bias" in optimizer.state
        
        # Check momentum buffers are initialized to zero
        for name in self.parameters:
            state = optimizer.state[name]
            assert state['step'] == 0
            assert np.allclose(state['momentum_buffer'], 0.0)
    
    def test_lion_initialization_custom_params(self):
        """Test Lion optimizer initialization with custom parameters."""
        optimizer = Lion(
            self.parameters,
            lr=3e-4,
            beta1=0.95,
            beta2=0.999,
            weight_decay=0.01,
            maximize=True
        )
        
        assert optimizer.lr == 3e-4
        assert optimizer.beta1 == 0.95
        assert optimizer.beta2 == 0.999
        assert optimizer.weight_decay == 0.01
        assert optimizer.maximize == True
    
    def test_lion_initialization_with_betas_tuple(self):
        """Test Lion optimizer initialization with PyTorch-style betas tuple."""
        optimizer = Lion(self.parameters, betas=(0.95, 0.999))
        
        assert optimizer.beta1 == 0.95
        assert optimizer.beta2 == 0.999
    
    def test_lion_initialization_invalid_params(self):
        """Test Lion optimizer with invalid parameters."""
        # Invalid learning rate
        with pytest.raises(OptimizerError):
            Lion(self.parameters, lr=-0.1)
        
        # Invalid beta1
        with pytest.raises(OptimizerError):
            Lion(self.parameters, beta1=1.5)
        
        # Invalid beta2
        with pytest.raises(OptimizerError):
            Lion(self.parameters, beta2=-0.1)
        
        # Invalid weight decay
        with pytest.raises(OptimizerError):
            Lion(self.parameters, weight_decay=-0.1)
        
        # Invalid betas tuple
        with pytest.raises(OptimizerError):
            Lion(self.parameters, betas=(0.9,))  # Wrong length
    
    def test_lion_step_basic(self):
        """Test basic Lion optimization step."""
        optimizer = Lion(self.parameters, lr=1e-3)  # Higher LR for visible changes
        
        # Store original parameter values
        original_param1 = self.param1.data.copy()
        original_param2 = self.param2.data.copy()
        
        # Perform optimization step
        optimizer.step()
        
        # Check that parameters were updated
        assert not np.allclose(self.param1.data, original_param1)
        assert not np.allclose(self.param2.data, original_param2)
        
        # Check that momentum buffers were updated
        for name in self.parameters:
            state = optimizer.state[name]
            assert state['step'] == 1
            assert not np.allclose(state['momentum_buffer'], 0.0)
        
        # Check step counter
        assert optimizer.step_count == 1
    
    def test_lion_algorithm_correctness(self):
        """Test Lion algorithm correctness against manual implementation."""
        optimizer = Lion(self.parameters, lr=1e-3, beta1=0.9, beta2=0.99)
        
        # Manual implementation
        param_data = self.param1.data.copy().astype(np.float64)
        grad = self.param1.grad.astype(np.float64)
        momentum_buffer = np.zeros_like(param_data)
        
        # Step 1: Compute interpolation
        c_t = optimizer.beta1 * momentum_buffer + (1 - optimizer.beta1) * grad
        
        # Step 2: Update parameters using sign
        update = optimizer.lr * np.sign(c_t)
        expected_param = param_data - update
        
        # Step 3: Update momentum
        expected_momentum = optimizer.beta2 * momentum_buffer + (1 - optimizer.beta2) * grad
        
        # Run optimizer step
        optimizer.step()
        
        # Compare results (allowing for dtype conversion precision differences)
        actual_param = self.param1.data.astype(np.float64)
        actual_momentum = optimizer.state["weight"]["momentum_buffer"]
        
        # Use more reasonable tolerance due to dtype conversions in implementation
        assert np.allclose(actual_param, expected_param, rtol=1e-6, atol=1e-8)
        assert np.allclose(actual_momentum, expected_momentum, rtol=1e-10)
    
    def test_lion_with_weight_decay(self):
        """Test Lion optimizer with weight decay."""
        optimizer = Lion(self.parameters, lr=1e-3, weight_decay=0.01)
        
        original_param = self.param1.data.copy()
        optimizer.step()
        
        # With weight decay, parameters should be pulled towards zero more
        # The weight decay term should be: lr * weight_decay * param
        expected_wd_term = optimizer.lr * optimizer.weight_decay * original_param.astype(np.float64)
        
        # The update should include both sign-based update and weight decay
        assert not np.allclose(self.param1.data, original_param)
    
    def test_lion_maximize_mode(self):
        """Test Lion optimizer in maximize mode."""
        optimizer = Lion(self.parameters, lr=1e-3, maximize=True)
        
        original_param = self.param1.data.copy()
        original_grad = self.param1.grad.copy()
        
        optimizer.step()
        
        # In maximize mode, gradients should be negated
        # This effectively moves parameters in the opposite direction
        assert not np.allclose(self.param1.data, original_param)
    
    def test_lion_no_gradients(self):
        """Test Lion optimizer when gradients are None."""
        # Clear gradients
        self.param1.grad = None
        self.param2.grad = None
        
        optimizer = Lion(self.parameters)
        original_param1 = self.param1.data.copy()
        original_param2 = self.param2.data.copy()
        
        # Step should not crash and parameters should remain unchanged
        optimizer.step()
        
        assert np.allclose(self.param1.data, original_param1)
        assert np.allclose(self.param2.data, original_param2)
        assert optimizer.step_count == 1  # Step counter should still increment
    
    def test_lion_zero_grad(self):
        """Test zeroing gradients."""
        optimizer = Lion(self.parameters)
        
        # Verify gradients exist
        assert self.param1.grad is not None
        assert self.param2.grad is not None
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Check gradients are None (tensor's zero_grad sets to None, not zeros)
        assert self.param1.grad is None
        assert self.param2.grad is None
    
    def test_lion_multiple_steps(self):
        """Test Lion optimizer over multiple steps."""
        optimizer = Lion(self.parameters, lr=1e-3)
        
        # Store momentum evolution
        momentum_history = []
        param_history = []
        
        for step in range(5):
            momentum_history.append(optimizer.state["weight"]["momentum_buffer"].copy())
            param_history.append(self.param1.data.copy())
            
            optimizer.step()
            
            # Reset gradients to simulate new batch
            self.param1.grad = np.array([0.1, -0.2, 0.3]) * (0.9 ** step)  # Decaying gradients
            self.param2.grad = np.array([[0.1, 0.2], [-0.1, 0.3]]) * (0.9 ** step)
        
        # Check that momentum and parameters evolved
        assert not np.allclose(momentum_history[0], momentum_history[-1])
        assert not np.allclose(param_history[0], param_history[-1])
        assert optimizer.step_count == 5
        
        # Check step counts in state
        for state in optimizer.state.values():
            assert state['step'] == 5
    
    def test_lion_state_dict_operations(self):
        """Test state dict save/load operations."""
        optimizer = Lion(self.parameters, lr=2e-4, beta1=0.95, weight_decay=0.01)
        
        # Take a few steps to build state
        for _ in range(3):
            optimizer.step()
        
        # Save state
        state_dict = optimizer.get_state_dict()
        
        # Create new optimizer
        new_optimizer = Lion(self.parameters)
        
        # Load state
        new_optimizer.load_state_dict(state_dict)
        
        # Verify state was loaded correctly
        assert new_optimizer.lr == 2e-4
        assert new_optimizer.beta1 == 0.95
        assert new_optimizer.weight_decay == 0.01
        
        # Check momentum buffers match
        for name in self.parameters:
            original_momentum = optimizer.state[name]['momentum_buffer']
            loaded_momentum = new_optimizer.state[name]['momentum_buffer']
            assert np.allclose(original_momentum, loaded_momentum)
    
    def test_lion_learning_rate_operations(self):
        """Test learning rate get/set operations."""
        optimizer = Lion(self.parameters, lr=1e-4)
        
        # Test getter
        assert optimizer.get_lr() == 1e-4
        
        # Test setter
        optimizer.set_lr(5e-4)
        assert optimizer.get_lr() == 5e-4
        assert optimizer.lr == 5e-4
        
        # Test invalid learning rate
        with pytest.raises(OptimizerError):
            optimizer.set_lr(-0.1)
    
    def test_lion_statistics(self):
        """Test optimizer statistics generation."""
        optimizer = Lion(self.parameters, lr=1e-3)
        
        # Take some steps
        for _ in range(3):
            optimizer.step()
        
        stats = optimizer.get_statistics()
        
        # Check basic stats
        assert stats['lr'] == 1e-3
        assert stats['num_parameters'] == 2
        assert stats['total_steps'] == 3
        
        # Check that gradient and parameter statistics are present
        assert 'avg_grad_norm' in stats
        assert 'avg_param_norm' in stats
        assert 'avg_momentum_norm' in stats
    
    def test_lion_numerical_stability(self):
        """Test Lion optimizer numerical stability."""
        # Create parameters with extreme values
        param_extreme = Parameter(np.array([1e6, -1e6, 1e-6]), name="extreme")
        param_extreme.grad = np.array([1e3, -1e3, 1e-3])
        
        optimizer = Lion({"extreme": param_extreme}, lr=1e-3)
        
        # Should not crash with extreme values
        optimizer.step()
        
        # Parameters should remain finite
        assert np.all(np.isfinite(param_extreme.data))
    
    @patch('src.neural_arch.optim.lion.logger')
    def test_lion_logging(self, mock_logger):
        """Test Lion optimizer logging functionality."""
        optimizer = Lion(self.parameters)
        
        # Check initialization logging
        mock_logger.info.assert_called_with(
            "Initialized Lion optimizer: lr=0.0001, beta1=0.9, beta2=0.99"
        )
        
        # Check step logging
        optimizer.step()
        mock_logger.debug.assert_called_with("Completed Lion optimization step")
        
        # Check zero_grad logging
        optimizer.zero_grad()
        mock_logger.debug.assert_called_with("Zeroed all gradients")
    
    def test_lion_repr(self):
        """Test string representation of Lion optimizer."""
        optimizer = Lion(self.parameters, lr=2e-4, beta1=0.95, weight_decay=0.01)
        
        repr_str = repr(optimizer)
        expected = "Lion(lr=0.0002, beta1=0.95, beta2=0.99, weight_decay=0.01)"
        assert repr_str == expected
    
    def test_lion_iterator_parameters(self):
        """Test Lion optimizer with parameter iterator instead of dict."""
        # Create parameter list
        param_list = [self.param1, self.param2]
        
        optimizer = Lion(param_list)
        
        # Should convert to dict internally
        assert len(optimizer.parameters) == 2
        assert "param_0" in optimizer.parameters
        assert "param_1" in optimizer.parameters
        
        # Should work normally
        optimizer.step()
        optimizer.zero_grad()
    
    def test_lion_vs_sgd_behavior(self):
        """Test that Lion behaves differently from SGD due to sign operation."""
        # Set up identical initial conditions
        param_lion = Parameter(np.array([1.0, 2.0]), name="lion_param")
        param_sgd = Parameter(np.array([1.0, 2.0]), name="sgd_param")
        
        # Different gradient magnitudes but same signs
        param_lion.grad = np.array([0.1, 0.5])  # Small and large positive gradients
        param_sgd.grad = np.array([0.1, 0.5])
        
        from src.neural_arch.optim.sgd import SGD
        
        lion_opt = Lion({"param": param_lion}, lr=0.1)
        sgd_opt = SGD({"param": param_sgd}, lr=0.1)
        
        lion_opt.step()
        sgd_opt.step()
        
        # Lion should treat both gradients equally due to sign operation
        # SGD should scale updates by gradient magnitude
        lion_update = np.array([1.0, 2.0]) - param_lion.data
        sgd_update = np.array([1.0, 2.0]) - param_sgd.data
        
        # Lion updates should have same magnitude for both dimensions
        assert np.isclose(abs(lion_update[0]), abs(lion_update[1]))
        
        # SGD updates should be proportional to gradient magnitudes
        assert abs(sgd_update[1]) > abs(sgd_update[0])


if __name__ == "__main__":
    pytest.main([__file__])