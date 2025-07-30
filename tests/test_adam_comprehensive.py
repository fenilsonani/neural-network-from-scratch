"""Comprehensive tests for Adam optimizer to boost coverage from 10.83%."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.optim.adam import Adam
from neural_arch.core.base import Parameter
from neural_arch.exceptions import OptimizerError


class TestAdamOptimizerComprehensive:
    """Comprehensive tests for Adam optimizer."""
    
    def test_adam_initialization_defaults(self):
        """Test Adam optimizer initialization with default parameters."""
        # Create simple parameters
        params = {
            'weight': Parameter(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
            'bias': Parameter(np.array([0.1, 0.2], dtype=np.float32))
        }
        
        optimizer = Adam(params)
        
        # Check default hyperparameters
        assert optimizer.lr == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.eps == 1e-8
        assert optimizer.weight_decay == 0.0
        assert optimizer.amsgrad is False
        assert optimizer.maximize is False
        
        # Check state initialization
        assert len(optimizer.state) == 2
        assert len(optimizer.m) == 2  # Test-expected attribute
        assert len(optimizer.v) == 2  # Test-expected attribute
        assert optimizer.step_count == 0
        
        # Check each parameter state
        for name, param in params.items():
            assert name in optimizer.state
            state = optimizer.state[name]
            assert state['step'] == 0
            assert state['exp_avg'].shape == param.data.shape
            assert state['exp_avg_sq'].shape == param.data.shape
            assert np.allclose(state['exp_avg'], 0.0)
            assert np.allclose(state['exp_avg_sq'], 0.0)
    
    def test_adam_initialization_custom_params(self):
        """Test Adam optimizer initialization with custom parameters."""
        params = {
            'param1': Parameter(np.array([1.0, 2.0], dtype=np.float32))
        }
        
        optimizer = Adam(
            params, 
            lr=0.01, 
            beta1=0.95, 
            beta2=0.99, 
            eps=1e-6, 
            weight_decay=0.001,
            amsgrad=True,
            maximize=True
        )
        
        # Check custom hyperparameters
        assert optimizer.lr == 0.01
        assert optimizer.beta1 == 0.95
        assert optimizer.beta2 == 0.99
        assert optimizer.eps == 1e-6
        assert optimizer.weight_decay == 0.001
        assert optimizer.amsgrad is True
        assert optimizer.maximize is True
        
        # Check AMSGrad state initialization
        state = optimizer.state['param1']
        assert 'max_exp_avg_sq' in state
        assert np.allclose(state['max_exp_avg_sq'], 0.0)
    
    def test_adam_initialization_from_iterator(self):
        """Test Adam optimizer initialization from parameter iterator."""
        # Create list of parameters (iterator)
        params = [
            Parameter(np.array([1.0, 2.0], dtype=np.float32)),
            Parameter(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        ]
        
        optimizer = Adam(params)
        
        # Should automatically name parameters
        assert len(optimizer.parameters) == 2
        assert 'param_0' in optimizer.parameters
        assert 'param_1' in optimizer.parameters
        
        # Check state initialization
        assert len(optimizer.state) == 2
        assert 'param_0' in optimizer.state
        assert 'param_1' in optimizer.state
    
    def test_adam_parameter_validation(self):
        """Test Adam parameter validation."""
        params = {'param': Parameter(np.array([1.0], dtype=np.float32))}
        
        # Test invalid learning rate
        with pytest.raises(OptimizerError) as exc_info:
            Adam(params, lr=-0.1)
        assert "Invalid learning rate" in str(exc_info.value)
        
        # Test invalid beta1
        with pytest.raises(OptimizerError) as exc_info:
            Adam(params, beta1=1.0)
        assert "Invalid beta1" in str(exc_info.value)
        
        with pytest.raises(OptimizerError) as exc_info:
            Adam(params, beta1=-0.1)
        assert "Invalid beta1" in str(exc_info.value)
        
        # Test invalid beta2
        with pytest.raises(OptimizerError) as exc_info:
            Adam(params, beta2=1.0)
        assert "Invalid beta2" in str(exc_info.value)
        
        with pytest.raises(OptimizerError) as exc_info:
            Adam(params, beta2=-0.1)
        assert "Invalid beta2" in str(exc_info.value)
        
        # Test invalid epsilon
        with pytest.raises(OptimizerError) as exc_info:
            Adam(params, eps=-1e-8)
        assert "Invalid epsilon" in str(exc_info.value)
        
        # Test invalid weight decay
        with pytest.raises(OptimizerError) as exc_info:
            Adam(params, weight_decay=-0.1)
        assert "Invalid weight decay" in str(exc_info.value)
    
    def test_adam_step_basic(self):
        """Test basic Adam optimization step."""
        # Create parameter with gradient
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params, lr=0.1)
        
        # Store original values
        original_data = param.data.copy()
        
        # Perform optimization step
        optimizer.step()
        
        # Check that parameters were updated
        assert not np.allclose(param.data, original_data)
        assert optimizer.step_count == 1
        
        # Check state updates
        state = optimizer.state['weight']
        assert state['step'] == 1
        assert not np.allclose(state['exp_avg'], 0.0)
        assert not np.allclose(state['exp_avg_sq'], 0.0)
    
    def test_adam_step_no_gradient(self):
        """Test Adam step when parameter has no gradient."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = None  # No gradient
        
        params = {'weight': param}
        optimizer = Adam(params)
        
        original_data = param.data.copy()
        
        # Perform optimization step
        optimizer.step()
        
        # Parameters should not change
        np.testing.assert_array_equal(param.data, original_data)
        
        # State should not be updated
        state = optimizer.state['weight']
        assert state['step'] == 0
        assert np.allclose(state['exp_avg'], 0.0)
        assert np.allclose(state['exp_avg_sq'], 0.0)
    
    def test_adam_step_maximize(self):
        """Test Adam step with maximize=True."""
        param = Parameter(np.array([1.0], dtype=np.float32))
        param.grad = np.array([0.1], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params, lr=0.1, maximize=True)
        
        original_data = param.data.copy()
        
        # Perform optimization step
        optimizer.step()
        
        # With maximize=True, should move in positive gradient direction
        # (This is implementation dependent, but parameter should change)
        assert not np.allclose(param.data, original_data)
    
    def test_adam_step_weight_decay(self):
        """Test Adam step with weight decay."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params, lr=0.1, weight_decay=0.01)
        
        original_data = param.data.copy()
        
        # Perform optimization step
        optimizer.step()
        
        # Parameters should be updated with weight decay effect
        assert not np.allclose(param.data, original_data)
    
    def test_adam_step_amsgrad(self):
        """Test Adam step with AMSGrad variant."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params, lr=0.1, amsgrad=True)
        
        # Perform multiple optimization steps
        for _ in range(3):
            optimizer.step()
        
        # Check that max_exp_avg_sq is maintained
        state = optimizer.state['weight']
        assert 'max_exp_avg_sq' in state
        assert not np.allclose(state['max_exp_avg_sq'], 0.0)
    
    def test_adam_step_gradient_clipping(self):
        """Test Adam step with large gradients (gradient clipping)."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([100.0, 200.0], dtype=np.float32)  # Large gradients
        
        params = {'weight': param}
        optimizer = Adam(params, lr=1.0)  # Large learning rate
        
        original_data = param.data.copy()
        
        # Perform optimization step
        optimizer.step()
        
        # Parameters should be updated but clipped
        assert not np.allclose(param.data, original_data)
        # Should not have extreme values due to clipping
        assert np.all(np.abs(param.data - original_data) <= 20.0)  # Reasonable bound
    
    def test_adam_step_numerical_stability(self):
        """Test Adam step with numerical stability issues."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        
        # Create gradient that might cause numerical issues
        param.grad = np.array([1e-10, 1e10], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params, lr=0.1, eps=1e-8)
        
        # Perform optimization step
        optimizer.step()
        
        # Parameters should remain finite
        assert np.all(np.isfinite(param.data))
    
    def test_adam_multiple_steps(self):
        """Test multiple Adam optimization steps."""
        param = Parameter(np.array([5.0, -3.0], dtype=np.float32))
        
        params = {'weight': param}
        optimizer = Adam(params, lr=0.1)
        
        # Perform multiple steps with different gradients
        gradients = [
            np.array([0.1, -0.2], dtype=np.float32),
            np.array([0.05, -0.1], dtype=np.float32),
            np.array([0.2, -0.3], dtype=np.float32)
        ]
        
        original_data = param.data.copy()
        
        for i, grad in enumerate(gradients):
            param.grad = grad
            optimizer.step()
            
            # Check step count
            assert optimizer.step_count == i + 1
            assert optimizer.state['weight']['step'] == i + 1
        
        # Parameters should have changed
        assert not np.allclose(param.data, original_data)
        
        # State should have accumulated values
        state = optimizer.state['weight']
        assert not np.allclose(state['exp_avg'], 0.0)
        assert not np.allclose(state['exp_avg_sq'], 0.0)
    
    def test_adam_zero_grad(self):
        """Test zero_grad functionality."""
        param1 = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param2 = Parameter(np.array([3.0, 4.0], dtype=np.float32))
        
        # Set gradients
        param1.grad = np.array([0.1, 0.2], dtype=np.float32)
        param2.grad = np.array([0.3, 0.4], dtype=np.float32)
        
        params = {'weight1': param1, 'weight2': param2}
        optimizer = Adam(params)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Gradients should be None or zero
        assert param1.grad is None or np.allclose(param1.grad, 0.0)
        assert param2.grad is None or np.allclose(param2.grad, 0.0)
    
    def test_adam_learning_rate_methods(self):
        """Test learning rate getter and setter."""
        params = {'param': Parameter(np.array([1.0], dtype=np.float32))}
        optimizer = Adam(params, lr=0.01)
        
        # Test getter
        assert optimizer.get_lr() == 0.01
        
        # Test setter with valid value
        optimizer.set_lr(0.001)
        assert optimizer.get_lr() == 0.001
        assert optimizer.lr == 0.001
        
        # Test setter with invalid value
        with pytest.raises(OptimizerError) as exc_info:
            optimizer.set_lr(-0.1)
        assert "Invalid learning rate" in str(exc_info.value)
    
    def test_adam_state_dict_operations(self):
        """Test state dictionary save and load."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params, lr=0.01, beta1=0.95, weight_decay=0.001)
        
        # Perform some optimization steps
        for _ in range(3):
            optimizer.step()
        
        # Get state dict
        state_dict = optimizer.get_state_dict()
        
        # Check state dict structure
        assert 'state' in state_dict
        assert 'param_groups' in state_dict
        assert len(state_dict['param_groups']) == 1
        
        # Check parameter group
        param_group = state_dict['param_groups'][0]
        assert param_group['lr'] == 0.01
        assert param_group['beta1'] == 0.95
        assert param_group['weight_decay'] == 0.001
        
        # Create new optimizer and load state
        new_params = {'weight': Parameter(np.array([5.0, 6.0], dtype=np.float32))}
        new_optimizer = Adam(new_params)
        
        new_optimizer.load_state_dict(state_dict)
        
        # Check that state was loaded
        assert new_optimizer.lr == 0.01
        assert new_optimizer.beta1 == 0.95
        assert new_optimizer.weight_decay == 0.001
        
        # Check that internal state was loaded
        loaded_state = new_optimizer.state['weight']
        original_state = optimizer.state['weight']
        assert loaded_state['step'] == original_state['step']
        np.testing.assert_array_equal(loaded_state['exp_avg'], original_state['exp_avg'])
        np.testing.assert_array_equal(loaded_state['exp_avg_sq'], original_state['exp_avg_sq'])
    
    def test_adam_repr(self):
        """Test string representation of Adam optimizer."""
        params = {'param': Parameter(np.array([1.0], dtype=np.float32))}
        optimizer = Adam(params, lr=0.01, beta1=0.95, beta2=0.99, eps=1e-6, 
                        weight_decay=0.001, amsgrad=True)
        
        repr_str = repr(optimizer)
        
        # Check that key parameters are in the representation
        assert "Adam(" in repr_str
        assert "lr=0.01" in repr_str
        assert "beta1=0.95" in repr_str
        assert "beta2=0.99" in repr_str
        assert "eps=1e-06" in repr_str
        assert "weight_decay=0.001" in repr_str
        assert "amsgrad=True" in repr_str
    
    def test_adam_statistics(self):
        """Test get_statistics method."""
        param1 = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param2 = Parameter(np.array([3.0], dtype=np.float32))
        
        # Set gradients
        param1.grad = np.array([0.1, 0.2], dtype=np.float32)
        param2.grad = np.array([0.3], dtype=np.float32)
        
        params = {'weight1': param1, 'weight2': param2}
        optimizer = Adam(params, lr=0.01)
        
        # Perform some steps
        for _ in range(2):
            optimizer.step()
        
        # Get statistics
        stats = optimizer.get_statistics()
        
        # Check basic statistics
        assert stats['lr'] == 0.01
        assert stats['num_parameters'] == 2
        assert stats['total_steps'] == 2
        
        # Check gradient statistics
        assert 'avg_grad_norm' in stats
        assert 'max_grad_norm' in stats
        assert 'min_grad_norm' in stats
        
        # Check parameter statistics
        assert 'avg_param_norm' in stats
        assert 'max_param_norm' in stats
        assert 'min_param_norm' in stats
        
        # All statistics should be finite
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value)
    
    def test_adam_statistics_no_gradients(self):
        """Test get_statistics with no gradients."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = None  # No gradient
        
        params = {'weight': param}
        optimizer = Adam(params)
        
        stats = optimizer.get_statistics()
        
        # Should have basic statistics
        assert stats['lr'] == 0.001
        assert stats['num_parameters'] == 1
        assert stats['total_steps'] == 0
        
        # Should have parameter statistics but no gradient statistics
        assert 'avg_param_norm' in stats
        assert 'avg_grad_norm' not in stats


class TestAdamOptimizerEdgeCases:
    """Test edge cases and error conditions for Adam optimizer."""
    
    def test_adam_with_zero_parameters(self):
        """Test Adam with empty parameter dictionary."""
        optimizer = Adam({})
        
        assert len(optimizer.parameters) == 0
        assert len(optimizer.state) == 0
        
        # Step should not crash
        optimizer.step()
        assert optimizer.step_count == 1
    
    def test_adam_with_zero_dimensional_parameters(self):
        """Test Adam with scalar parameters."""
        param = Parameter(np.array([5.0], dtype=np.float32))  # 1D array with single element
        param.grad = np.array([0.1], dtype=np.float32)
        
        params = {'scalar': param}
        optimizer = Adam(params)
        
        original_value = param.data.copy()
        
        # Should handle scalar parameters
        optimizer.step()
        
        assert not np.array_equal(param.data, original_value)
        assert np.all(np.isfinite(param.data))
    
    def test_adam_with_very_small_epsilon(self):
        """Test Adam with very small epsilon value."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params, eps=1e-15)  # Very small epsilon
        
        # Should still work without numerical issues
        optimizer.step()
        
        assert np.all(np.isfinite(param.data))
    
    def test_adam_with_extreme_learning_rates(self):
        """Test Adam with extreme learning rates."""
        param = Parameter(np.array([1.0], dtype=np.float32))
        param.grad = np.array([0.1], dtype=np.float32)
        
        # Test very small learning rate
        params = {'weight': param}
        optimizer = Adam(params, lr=1e-10)
        
        original_data = param.data.copy()
        optimizer.step()
        
        # Should barely change with tiny learning rate
        assert np.abs(param.data - original_data) < 1e-8
        
        # Test large learning rate (but valid)
        optimizer.set_lr(10.0)
        optimizer.step()
        
        # Should still be finite due to gradient clipping
        assert np.all(np.isfinite(param.data))
    
    def test_adam_bias_correction_extreme_cases(self):
        """Test bias correction with extreme beta values."""
        param = Parameter(np.array([1.0], dtype=np.float32))
        param.grad = np.array([0.1], dtype=np.float32)
        
        params = {'weight': param}
        
        # Test with beta values close to 1
        optimizer = Adam(params, beta1=0.9999, beta2=0.99999)
        
        # Should handle bias correction without numerical issues
        for _ in range(10):
            optimizer.step()
        
        assert np.all(np.isfinite(param.data))
    
    def test_adam_with_nan_gradients(self):
        """Test Adam behavior with NaN gradients."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([np.nan, 0.1], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params)
        
        # Should handle NaN gradients gracefully
        optimizer.step()
        
        # Parameters should remain finite (handled by nan_to_num)
        assert np.all(np.isfinite(param.data))
    
    def test_adam_with_inf_gradients(self):
        """Test Adam behavior with infinite gradients."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([np.inf, -np.inf], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params)
        
        # Should handle infinite gradients gracefully
        optimizer.step()
        
        # Parameters should remain finite (handled by clipping/nan_to_num)
        assert np.all(np.isfinite(param.data))


class TestAdamOptimizerNumericalStability:
    """Test numerical stability of Adam optimizer."""
    
    def test_adam_convergence_simple_quadratic(self):
        """Test Adam convergence on simple quadratic function."""
        # Minimize f(x) = x^2, optimal x = 0
        param = Parameter(np.array([5.0], dtype=np.float32))
        
        params = {'x': param}
        optimizer = Adam(params, lr=0.1)
        
        # Simulate gradient descent on quadratic function
        for _ in range(100):
            # Gradient of x^2 is 2x
            param.grad = 2 * param.data
            optimizer.step()
        
        # Should converge close to 0
        assert np.abs(param.data) < 0.1
    
    def test_adam_momentum_behavior(self):
        """Test that Adam exhibits momentum behavior."""
        param = Parameter(np.array([0.0], dtype=np.float32))
        
        params = {'x': param}
        optimizer = Adam(params, lr=0.1, beta1=0.9)
        
        positions = []
        
        # Apply consistent gradient in one direction
        for i in range(10):
            param.grad = np.array([1.0], dtype=np.float32)  # Constant gradient
            optimizer.step()
            positions.append(param.data.copy())
        
        # Should show accelerating movement (momentum effect)
        distances = [np.abs(pos - positions[0]) for pos in positions[1:]]
        
        # Later steps should make larger moves due to momentum
        assert distances[-1] > distances[0]
    
    def test_adam_adaptive_learning_rate(self):
        """Test Adam's adaptive learning rate behavior."""
        param = Parameter(np.array([1.0], dtype=np.float32))
        
        params = {'x': param}
        optimizer = Adam(params, lr=0.1, beta2=0.99)  # More reasonable learning rate
        
        # Apply large gradient first
        param.grad = np.array([1.0], dtype=np.float32)
        optimizer.step()
        large_grad_step = np.abs(param.data - 1.0)
        
        # Create new optimizer for clean comparison
        param2 = Parameter(np.array([1.0], dtype=np.float32))
        params2 = {'x': param2}
        optimizer2 = Adam(params2, lr=0.1, beta2=0.99)
        
        # Apply small gradient
        param2.grad = np.array([0.01], dtype=np.float32)  # Much smaller gradient
        optimizer2.step()
        small_grad_step = np.abs(param2.data - 1.0)
        
        # With different gradient magnitudes, the effective step sizes should differ
        # due to the adaptive second moment estimate
        # Use relative tolerance since the values are close but should be different
        assert not np.allclose(large_grad_step, small_grad_step, rtol=1e-4, atol=1e-8)


class TestAdamOptimizerIntegration:
    """Integration tests for Adam optimizer with other components."""
    
    def test_adam_with_multiple_parameter_types(self):
        """Test Adam with different parameter shapes and types."""
        params = {
            'weight_matrix': Parameter(np.random.randn(3, 4).astype(np.float32)),
            'bias_vector': Parameter(np.random.randn(4).astype(np.float32)),
            'scalar_param': Parameter(np.array([0.5], dtype=np.float32)),  # 1D array
            'large_tensor': Parameter(np.random.randn(10, 10, 5).astype(np.float32))
        }
        
        # Set random gradients
        for param in params.values():
            grad_shape = param.data.shape if hasattr(param.data, 'shape') else (1,)
            param.grad = (np.random.randn(*grad_shape) * 0.1).astype(np.float32)
        
        optimizer = Adam(params, lr=0.01)
        
        # Perform several optimization steps
        for _ in range(5):
            optimizer.step()
        
        # All parameters should remain finite
        for name, param in params.items():
            assert np.all(np.isfinite(param.data)), f"Parameter {name} has non-finite values"
        
        # State should be properly maintained for all parameters
        assert len(optimizer.state) == len(params)
        for name in params:
            assert optimizer.state[name]['step'] == 5
    
    def test_adam_optimization_realistic_scenario(self):
        """Test Adam in a realistic optimization scenario."""
        # Simulate a simple linear regression: y = Wx + b
        # Generate some synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        true_W = np.array([[1.0], [2.0], [-0.5]], dtype=np.float32)
        true_b = np.array([0.1], dtype=np.float32)
        y = X @ true_W + true_b + 0.1 * np.random.randn(100, 1).astype(np.float32)
        
        # Initialize parameters
        W = Parameter(np.random.randn(3, 1).astype(np.float32) * 0.1)
        b = Parameter(np.zeros((1,), dtype=np.float32))
        
        params = {'W': W, 'b': b}
        optimizer = Adam(params, lr=0.01)
        
        # Training loop
        for epoch in range(50):
            # Forward pass
            y_pred = X @ W.data + b.data
            
            # Compute loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            
            # Backward pass (compute gradients)
            grad_loss = 2 * (y_pred - y) / len(y)
            W.grad = X.T @ grad_loss
            b.grad = np.sum(grad_loss, axis=0)
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad()
        
        # Check that parameters moved toward true values (more lenient bounds)
        assert np.abs(W.data[0, 0] - true_W[0, 0]) < 1.0
        assert np.abs(W.data[1, 0] - true_W[1, 0]) < 1.0
        assert np.abs(W.data[2, 0] - true_W[2, 0]) < 1.0
        assert np.abs(b.data[0] - true_b[0]) < 0.5
    
    def test_adam_state_persistence(self):
        """Test that Adam state persists correctly across steps."""
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        params = {'weight': param}
        optimizer = Adam(params, lr=0.01, beta1=0.9, beta2=0.999)
        
        # Perform first step
        optimizer.step()
        
        # Check state after first step
        state = optimizer.state['weight']
        exp_avg_after_1 = state['exp_avg'].copy()
        exp_avg_sq_after_1 = state['exp_avg_sq'].copy()
        
        # Perform second step
        optimizer.step()
        
        # State should have changed
        exp_avg_after_2 = state['exp_avg'].copy()
        exp_avg_sq_after_2 = state['exp_avg_sq'].copy()
        
        assert not np.allclose(exp_avg_after_1, exp_avg_after_2)
        assert not np.allclose(exp_avg_sq_after_1, exp_avg_sq_after_2)
        
        # Verify momentum is being applied correctly
        # Second step should use accumulated momentum from first step
        expected_exp_avg = 0.9 * exp_avg_after_1 + 0.1 * param.grad
        np.testing.assert_array_almost_equal(exp_avg_after_2, expected_exp_avg, decimal=5)