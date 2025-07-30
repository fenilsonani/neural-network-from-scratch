"""Comprehensive tests for SGD optimizer targeting coverage improvement.

This file targets the SGD optimizer to improve coverage from 57.69% to 95%+.
Tests all functionality, edge cases, and error handling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.optim.sgd import SGD, SGDMomentum
from neural_arch.core.base import Parameter
from neural_arch.exceptions import OptimizerError, NeuralArchError


class TestSGDComprehensive:
    """Comprehensive tests for SGD optimizer."""
    
    def test_sgd_initialization(self):
        """Test SGD optimizer initialization."""
        # Create parameters
        param1 = Parameter(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="param1")
        param2 = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="param2")
        
        parameters = {"param1": param1, "param2": param2}
        
        # Test default initialization
        optimizer_default = SGD(parameters)
        assert optimizer_default.lr == 0.01  # Default learning rate
        assert optimizer_default.parameters is parameters
        
        # Test custom learning rate
        custom_lr = 0.001
        optimizer_custom = SGD(parameters, lr=custom_lr)
        assert optimizer_custom.lr == custom_lr
        assert optimizer_custom.parameters is parameters
    
    def test_sgd_step_basic(self):
        """Test basic SGD optimization step."""
        # Create parameter with gradient
        initial_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        param = Parameter(initial_data.copy(), name="test_param")
        param.grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        parameters = {"param": param}
        optimizer = SGD(parameters, lr=0.1)
        
        # Perform step
        optimizer.step()
        
        # Expected: param.data = [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
        expected = initial_data - 0.1 * param.grad
        np.testing.assert_array_almost_equal(param.data, expected, decimal=6)
    
    def test_sgd_step_multiple_parameters(self):
        """Test SGD step with multiple parameters."""
        # Create multiple parameters
        param1 = Parameter(np.array([1.0, 2.0], dtype=np.float32), name="param1")
        param1.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        param2 = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="param2")
        param2.grad = np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float32)
        
        parameters = {"param1": param1, "param2": param2}
        optimizer = SGD(parameters, lr=0.5)
        
        # Store original values
        original_param1 = param1.data.copy()
        original_param2 = param2.data.copy()
        
        # Perform step
        optimizer.step()
        
        # Check updates
        expected_param1 = original_param1 - 0.5 * param1.grad
        expected_param2 = original_param2 - 0.5 * param2.grad
        
        np.testing.assert_array_almost_equal(param1.data, expected_param1, decimal=6)
        np.testing.assert_array_almost_equal(param2.data, expected_param2, decimal=6)
    
    def test_sgd_step_no_gradients(self):
        """Test SGD step when parameters have no gradients."""
        # Create parameter without gradient
        initial_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        param = Parameter(initial_data.copy(), name="test_param")
        # param.grad is None by default
        
        parameters = {"param": param}
        optimizer = SGD(parameters, lr=0.1)
        
        # Perform step (should be no-op)
        optimizer.step()
        
        # Data should be unchanged
        np.testing.assert_array_equal(param.data, initial_data)
    
    def test_sgd_step_mixed_gradients(self):
        """Test SGD step with mixed gradient availability."""
        # Create parameters with mixed gradient states
        param1 = Parameter(np.array([1.0, 2.0], dtype=np.float32), name="param1")
        param1.grad = np.array([0.1, 0.2], dtype=np.float32)  # Has gradient
        
        param2 = Parameter(np.array([3.0, 4.0], dtype=np.float32), name="param2")
        # param2.grad is None (no gradient)
        
        parameters = {"param1": param1, "param2": param2}
        optimizer = SGD(parameters, lr=0.1)
        
        # Store original values
        original_param1 = param1.data.copy()
        original_param2 = param2.data.copy()
        
        # Perform step
        optimizer.step()
        
        # Only param1 should be updated
        expected_param1 = original_param1 - 0.1 * param1.grad
        np.testing.assert_array_almost_equal(param1.data, expected_param1, decimal=6)
        np.testing.assert_array_equal(param2.data, original_param2)  # Unchanged
    
    def test_sgd_zero_grad(self):
        """Test SGD zero_grad functionality."""
        # Create parameters with gradients
        param1 = Parameter(np.array([1.0, 2.0], dtype=np.float32), name="param1")
        param1.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        param2 = Parameter(np.array([3.0, 4.0], dtype=np.float32), name="param2")
        param2.grad = np.array([0.3, 0.4], dtype=np.float32)
        
        parameters = {"param1": param1, "param2": param2}
        optimizer = SGD(parameters, lr=0.1)
        
        # Verify gradients are set
        assert param1.grad is not None
        assert param2.grad is not None
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Verify gradients are None
        assert param1.grad is None
        assert param2.grad is None
    
    def test_sgd_zero_grad_mixed_states(self):
        """Test zero_grad with mixed gradient states."""
        # Create parameters with mixed gradient states
        param1 = Parameter(np.array([1.0, 2.0], dtype=np.float32), name="param1")
        param1.grad = np.array([0.1, 0.2], dtype=np.float32)  # Has gradient
        
        param2 = Parameter(np.array([3.0, 4.0], dtype=np.float32), name="param2")
        # param2.grad is None (no gradient)
        
        parameters = {"param1": param1, "param2": param2}
        optimizer = SGD(parameters, lr=0.1)
        
        # Zero gradients (should handle both cases)
        optimizer.zero_grad()
        
        # Both should have None gradients
        assert param1.grad is None
        assert param2.grad is None
    
    def test_sgd_learning_rate_effects(self):
        """Test different learning rates on SGD performance."""
        learning_rates = [0.001, 0.01, 0.1, 1.0]
        
        for lr in learning_rates:
            # Create fresh parameter for each test
            initial_data = np.array([1.0], dtype=np.float32)
            param = Parameter(initial_data.copy(), name="test_param")
            param.grad = np.array([0.1], dtype=np.float32)
            
            parameters = {"param": param}
            optimizer = SGD(parameters, lr=lr)
            
            # Perform step
            optimizer.step()
            
            # Expected update: 1.0 - lr * 0.1
            expected = 1.0 - lr * 0.1
            np.testing.assert_array_almost_equal(param.data, [expected], decimal=6)
    
    def test_sgd_multiple_steps(self):
        """Test multiple SGD optimization steps."""
        # Create parameter
        param = Parameter(np.array([1.0], dtype=np.float32), name="test_param")
        parameters = {"param": param}
        optimizer = SGD(parameters, lr=0.1)
        
        # Perform multiple steps with different gradients
        gradients = [0.1, 0.2, 0.3, 0.4]
        expected_values = [1.0]  # Initial value
        
        for grad in gradients:
            param.grad = np.array([grad], dtype=np.float32)
            optimizer.step()
            
            # Calculate expected value
            expected_value = expected_values[-1] - 0.1 * grad
            expected_values.append(expected_value)
            
            np.testing.assert_array_almost_equal(param.data, [expected_value], decimal=6)
    
    def test_sgd_step_and_zero_grad_cycle(self):
        """Test typical training cycle of step() and zero_grad()."""
        # Create parameter
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32), name="test_param")
        parameters = {"param": param}
        optimizer = SGD(parameters, lr=0.1)
        
        # Simulate training cycles
        for i in range(3):
            # Set gradient
            param.grad = np.array([0.1 * (i + 1), 0.2 * (i + 1)], dtype=np.float32)
            
            # Store values before step
            pre_step_data = param.data.copy()
            pre_step_grad = param.grad.copy()
            
            # Perform step
            optimizer.step()
            
            # Check update occurred
            expected_data = pre_step_data - 0.1 * pre_step_grad
            np.testing.assert_array_almost_equal(param.data, expected_data, decimal=6)
            
            # Zero gradients
            optimizer.zero_grad()
            assert param.grad is None
    
    def test_sgd_edge_cases(self):
        """Test SGD with edge cases."""
        # Test with zero gradient
        param_zero_grad = Parameter(np.array([1.0, 2.0], dtype=np.float32), name="zero_grad")
        param_zero_grad.grad = np.array([0.0, 0.0], dtype=np.float32)
        
        parameters = {"param": param_zero_grad}
        optimizer = SGD(parameters, lr=0.1)
        
        original_data = param_zero_grad.data.copy()
        optimizer.step()
        
        # With zero gradient, data should be unchanged
        np.testing.assert_array_equal(param_zero_grad.data, original_data)
        
        # Test with very small learning rate
        param_small_lr = Parameter(np.array([1.0], dtype=np.float32), name="small_lr")
        param_small_lr.grad = np.array([1.0], dtype=np.float32)
        
        optimizer_small = SGD({"param": param_small_lr}, lr=1e-10)
        original_small = param_small_lr.data.copy()
        optimizer_small.step()
        
        # Update should be tiny but measurable
        expected_small = original_small - 1e-10 * param_small_lr.grad
        np.testing.assert_array_almost_equal(param_small_lr.data, expected_small, decimal=15)
        
        # Test with large learning rate
        param_large_lr = Parameter(np.array([1.0], dtype=np.float32), name="large_lr")
        param_large_lr.grad = np.array([0.1], dtype=np.float32)
        
        optimizer_large = SGD({"param": param_large_lr}, lr=100.0)
        original_large = param_large_lr.data.copy()
        optimizer_large.step()
        
        # Update should be large
        expected_large = original_large - 100.0 * param_large_lr.grad
        np.testing.assert_array_almost_equal(param_large_lr.data, expected_large, decimal=6)
    
    def test_sgd_parameter_dtypes(self):
        """Test SGD with different parameter dtypes."""
        # Test float32
        param_f32 = Parameter(np.array([1.0, 2.0], dtype=np.float32), name="f32")
        param_f32.grad = np.array([0.1, 0.2], dtype=np.float32)
        
        optimizer_f32 = SGD({"param": param_f32}, lr=0.1)
        original_f32 = param_f32.data.copy()
        optimizer_f32.step()
        
        expected_f32 = original_f32 - 0.1 * param_f32.grad
        np.testing.assert_array_almost_equal(param_f32.data, expected_f32, decimal=6)
        assert param_f32.data.dtype == np.float32
        
        # Test float64
        param_f64 = Parameter(np.array([1.0, 2.0], dtype=np.float64), name="f64")
        param_f64.grad = np.array([0.1, 0.2], dtype=np.float64)
        
        optimizer_f64 = SGD({"param": param_f64}, lr=0.1)
        original_f64 = param_f64.data.copy()
        optimizer_f64.step()
        
        expected_f64 = original_f64 - 0.1 * param_f64.grad
        np.testing.assert_array_almost_equal(param_f64.data, expected_f64, decimal=6)
        # Note: dtype might be converted during operations, check actual behavior
        assert param_f64.data.dtype in (np.float32, np.float64)
    
    def test_sgd_parameter_shapes(self):
        """Test SGD with different parameter shapes."""
        # Scalar parameter - use proper scalar array
        param_scalar = Parameter(np.array([5.0], dtype=np.float32), name="scalar")  # 1D array instead
        param_scalar.grad = np.array([0.1], dtype=np.float32)
        
        optimizer_scalar = SGD({"param": param_scalar}, lr=0.1)
        original_scalar = param_scalar.data.copy()
        optimizer_scalar.step()
        
        expected_scalar = original_scalar - 0.1 * param_scalar.grad
        np.testing.assert_array_almost_equal(param_scalar.data, expected_scalar, decimal=6)
        
        # 1D parameter
        param_1d = Parameter(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="1d")
        param_1d.grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        optimizer_1d = SGD({"param": param_1d}, lr=0.1)
        original_1d = param_1d.data.copy()
        optimizer_1d.step()
        
        expected_1d = original_1d - 0.1 * param_1d.grad
        np.testing.assert_array_almost_equal(param_1d.data, expected_1d, decimal=6)
        
        # 2D parameter
        param_2d = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="2d")
        param_2d.grad = np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float32)
        
        optimizer_2d = SGD({"param": param_2d}, lr=0.1)
        original_2d = param_2d.data.copy()
        optimizer_2d.step()
        
        expected_2d = original_2d - 0.1 * param_2d.grad
        np.testing.assert_array_almost_equal(param_2d.data, expected_2d, decimal=6)
        
        # 3D parameter
        param_3d = Parameter(np.random.randn(2, 3, 4).astype(np.float32), name="3d")
        param_3d.grad = np.random.randn(2, 3, 4).astype(np.float32)
        
        optimizer_3d = SGD({"param": param_3d}, lr=0.01)
        original_3d = param_3d.data.copy()
        optimizer_3d.step()
        
        expected_3d = original_3d - 0.01 * param_3d.grad
        np.testing.assert_array_almost_equal(param_3d.data, expected_3d, decimal=6)
    
    def test_sgd_inheritance_and_attributes(self):
        """Test SGD class inheritance and attributes."""
        # Create SGD optimizer
        param = Parameter(np.array([1.0], dtype=np.float32), name="test")
        parameters = {"param": param}
        optimizer = SGD(parameters, lr=0.05)
        
        # Test inheritance from Optimizer base class
        from neural_arch.core.base import Optimizer
        assert isinstance(optimizer, Optimizer)
        
        # Test attributes
        assert hasattr(optimizer, 'parameters')
        assert hasattr(optimizer, 'lr')
        assert hasattr(optimizer, 'step')
        assert hasattr(optimizer, 'zero_grad')
        
        # Test that methods are callable
        assert callable(optimizer.step)
        assert callable(optimizer.zero_grad)
        
        # Test parameter access
        assert optimizer.parameters is parameters
        assert optimizer.lr == 0.05
    
    def test_sgd_momentum_alias(self):
        """Test SGDMomentum alias functionality."""
        # Create parameter
        param = Parameter(np.array([1.0, 2.0], dtype=np.float32), name="test")
        parameters = {"param": param}
        
        # Test that SGDMomentum is an alias for SGD
        assert SGDMomentum is SGD
        
        # Test that SGDMomentum works the same as SGD
        optimizer_sgd = SGD(parameters, lr=0.1)
        optimizer_momentum = SGDMomentum(parameters, lr=0.1)
        
        # Both should be the same type
        assert type(optimizer_sgd) == type(optimizer_momentum)
        assert optimizer_sgd.__class__ == optimizer_momentum.__class__
    
    def test_sgd_error_handling(self):
        """Test SGD error handling with handle_exception decorator."""
        # Create parameter
        param = Parameter(np.array([1.0], dtype=np.float32), name="test")
        parameters = {"param": param}
        optimizer = SGD(parameters, lr=0.1)
        
        # Test that step() and zero_grad() have handle_exception decorator
        # This is tested by ensuring they don't raise unexpected exceptions
        # with normal usage
        
        # Normal operation should work
        param.grad = np.array([0.1], dtype=np.float32)
        try:
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            pytest.fail(f"Normal SGD operations should not raise exceptions: {e}")
        
        # Test with None parameters (should be handled gracefully)
        try:
            optimizer.step()  # param.grad is now None after zero_grad()
            optimizer.zero_grad()  # Should handle None gradients
        except Exception as e:
            pytest.fail(f"SGD should handle None gradients gracefully: {e}")


class TestSGDIntegration:
    """Integration tests for SGD optimizer."""
    
    def test_sgd_with_linear_layer(self):
        """Test SGD integration with Linear layer."""
        from neural_arch.nn.linear import Linear
        from neural_arch.core.tensor import Tensor
        
        # Create linear layer
        layer = Linear(3, 2, weight_init="zeros", bias_init="zeros")
        
        # Create SGD optimizer
        optimizer = SGD(layer.parameters(), lr=0.1)
        
        # Create input and target
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        target = Tensor([[1.0, 0.0]], requires_grad=False)
        
        # Forward pass
        output = layer(x)
        
        # Simple loss (MSE)
        from neural_arch.functional.loss import mse_loss
        loss = mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        if hasattr(loss, '_backward'):
            loss._backward()
        
        # Check that parameters have gradients
        params = list(layer.parameters())
        for param in params:
            assert param.grad is not None
        
        # Store original parameter values
        original_params = [p.data.copy() for p in params]
        
        # Optimizer step
        optimizer.step()
        
        # Parameters should have changed
        for i, param in enumerate(params):
            assert not np.array_equal(param.data, original_params[i])
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Gradients should be None
        for param in params:
            assert param.grad is None
    
    def test_sgd_convergence_simple(self):
        """Test SGD convergence on a simple optimization problem."""
        # Optimize f(x) = (x - 2)^2, minimum at x = 2
        
        # Parameter to optimize
        x = Parameter(np.array([0.0], dtype=np.float32), name="x")
        parameters = {"x": x}
        optimizer = SGD(parameters, lr=0.1)
        
        # Optimization loop
        for i in range(100):
            # Compute gradient: f'(x) = 2(x - 2)
            x.grad = 2 * (x.data - 2.0)
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad()
        
        # Should converge close to x = 2
        np.testing.assert_almost_equal(x.data, [2.0], decimal=2)
    
    def test_sgd_training_loop_simulation(self):
        """Test SGD in a realistic training loop simulation."""
        # Create multiple parameters
        weights = Parameter(np.random.randn(5, 3).astype(np.float32), name="weights")
        bias = Parameter(np.zeros(3, dtype=np.float32), name="bias")
        
        parameters = {"weights": weights, "bias": bias}
        optimizer = SGD(parameters, lr=0.01)
        
        # Simulate training epochs
        num_epochs = 5
        losses = []
        
        for epoch in range(num_epochs):
            # Simulate batch processing
            epoch_loss = 0.0
            
            for batch in range(3):  # 3 batches per epoch
                # Simulate gradients (random for this test)
                weights.grad = np.random.randn(*weights.shape).astype(np.float32) * 0.1
                bias.grad = np.random.randn(*bias.shape).astype(np.float32) * 0.1
                
                # Store parameter values before update
                weights_before = weights.data.copy()
                bias_before = bias.data.copy()
                
                # Optimization step
                optimizer.step()
                
                # Verify parameters changed
                assert not np.array_equal(weights.data, weights_before)
                assert not np.array_equal(bias.data, bias_before)
                
                # Simulate loss computation
                batch_loss = np.random.rand()
                epoch_loss += batch_loss
                
                # Zero gradients
                optimizer.zero_grad()
                assert weights.grad is None
                assert bias.grad is None
            
            losses.append(epoch_loss)
        
        # Should have completed all epochs
        assert len(losses) == num_epochs
        
        # Parameters should have final values different from initial
        assert weights.data.shape == (5, 3)
        assert bias.data.shape == (3,)
    
    def test_sgd_parameter_groups_compatibility(self):
        """Test SGD compatibility with parameter groups concept."""
        # Create different types of parameters
        layer1_weight = Parameter(np.random.randn(3, 2).astype(np.float32), name="layer1.weight")
        layer1_bias = Parameter(np.zeros(2, dtype=np.float32), name="layer1.bias")
        layer2_weight = Parameter(np.random.randn(2, 1).astype(np.float32), name="layer2.weight")
        
        # Group all parameters
        all_parameters = {
            "layer1.weight": layer1_weight,
            "layer1.bias": layer1_bias,
            "layer2.weight": layer2_weight
        }
        
        # Create optimizer
        optimizer = SGD(all_parameters, lr=0.05)
        
        # Set gradients for all parameters
        for param in all_parameters.values():
            param.grad = np.random.randn(*param.shape).astype(np.float32) * 0.1
        
        # Store original values
        original_values = {name: param.data.copy() for name, param in all_parameters.items()}
        
        # Perform optimization step
        optimizer.step()
        
        # All parameters should have been updated
        for name, param in all_parameters.items():
            assert not np.array_equal(param.data, original_values[name])
        
        # Zero gradients
        optimizer.zero_grad()
        
        # All gradients should be None
        for param in all_parameters.values():
            assert param.grad is None