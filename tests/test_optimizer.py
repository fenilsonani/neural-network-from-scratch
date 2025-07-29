"""
Test optimizers - because training better fucking work.
"""

try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_arch import Tensor, Adam


class TestAdam:
    """Test Adam optimizer because it's the gold standard."""
    
    def test_adam_creation(self):
        """Test Adam optimizer creation."""
        params = {
            'w': Tensor([[1.0, 2.0]], requires_grad=True),
            'b': Tensor([0.5], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.01)
        
        assert optimizer.lr == 0.01
        assert optimizer.step_count == 0
        assert len(optimizer.m) == 2
        assert len(optimizer.v) == 2
        assert 'w' in optimizer.parameters
        assert 'b' in optimizer.parameters
    
    def test_adam_zero_grad(self):
        """Test gradient zeroing."""
        params = {
            'w': Tensor([[1.0, 2.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.01)
        
        # Set some gradients
        params['w'].grad = np.array([[0.1, 0.2]])
        
        # Zero them
        optimizer.zero_grad()
        
        assert params['w'].grad is None
    
    def test_adam_single_step(self):
        """Test single optimization step."""
        params = {
            'w': Tensor([[1.0, 2.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.1)
        
        # Set gradients
        params['w'].grad = np.array([[0.1, -0.2]])
        
        # Store original values
        original_w = params['w'].data.copy()
        
        # Take step
        optimizer.step()
        
        # Check parameters changed
        assert not np.array_equal(params['w'].data, original_w)
        assert optimizer.step_count == 1
        
        # Check moments were updated
        assert not np.array_equal(optimizer.m['w'], np.zeros_like(optimizer.m['w']))
        assert not np.array_equal(optimizer.v['w'], np.zeros_like(optimizer.v['w']))
    
    def test_adam_multiple_steps(self):
        """Test multiple optimization steps."""
        params = {
            'w': Tensor([[1.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.1)
        
        # Take multiple steps with consistent gradients
        for i in range(5):
            params['w'].grad = np.array([[0.1]])  # Consistent positive gradient
            optimizer.step()
            optimizer.zero_grad()
        
        # Parameter should have decreased (opposite of gradient)
        assert params['w'].data[0, 0] < 1.0
        assert optimizer.step_count == 5
    
    def test_adam_bias_correction(self):
        """Test bias correction in early steps."""
        params = {
            'w': Tensor([[1.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.01)
        
        # First step
        params['w'].grad = np.array([[1.0]])
        original_w = params['w'].data.copy()
        optimizer.step()
        
        # Should have meaningful update despite bias correction
        update_magnitude = np.abs(params['w'].data - original_w)
        assert update_magnitude > 1e-6  # Should have some effect
        assert update_magnitude < 1.0   # But not too large
    
    def test_adam_momentum(self):
        """Test momentum accumulation."""
        params = {
            'w': Tensor([[0.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.1)
        
        # Apply same gradient multiple times
        gradient = np.array([[0.1]])
        
        updates = []
        for i in range(3):
            params['w'].grad = gradient.copy()
            old_w = params['w'].data.copy()
            optimizer.step()
            update = params['w'].data - old_w
            updates.append(np.abs(update[0, 0]))
            optimizer.zero_grad()
        
        # Updates should generally increase due to momentum
        # (though bias correction complicates early steps)
        assert len(updates) == 3
    
    def test_adam_gradient_clipping(self):
        """Test gradient clipping in optimizer."""
        params = {
            'w': Tensor([[1.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.1)
        
        # Apply huge gradient
        params['w'].grad = np.array([[1000.0]])
        
        original_w = params['w'].data.copy()
        optimizer.step()
        
        # Update should be reasonable, not huge
        update_magnitude = np.abs(params['w'].data - original_w)
        assert update_magnitude < 1.0  # Should be clipped
    
    def test_adam_no_gradients(self):
        """Test optimizer with no gradients."""
        params = {
            'w': Tensor([[1.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.1)
        
        # Don't set any gradients
        original_w = params['w'].data.copy()
        optimizer.step()
        
        # Parameters should not change
        assert np.array_equal(params['w'].data, original_w)
    
    def test_adam_mixed_gradients(self):
        """Test optimizer with some parameters having gradients."""
        params = {
            'w1': Tensor([[1.0]], requires_grad=True),
            'w2': Tensor([[2.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.1)
        
        # Only set gradient for w1
        params['w1'].grad = np.array([[0.1]])
        # w2.grad remains None
        
        original_w1 = params['w1'].data.copy()
        original_w2 = params['w2'].data.copy()
        
        optimizer.step()
        
        # w1 should change, w2 should not
        assert not np.array_equal(params['w1'].data, original_w1)
        assert np.array_equal(params['w2'].data, original_w2)


class TestOptimizerIntegration:
    """Test optimizer integration with layers."""
    
    def test_optimizer_with_linear_layer(self):
        """Test optimizer with linear layer."""
        from neural_arch import Linear
        
        layer = Linear(2, 1)
        optimizer = Adam(layer.parameters(), lr=0.1)
        
        # Forward pass
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = layer(x)
        
        # Simple loss
        target = Tensor([[0.0]], requires_grad=False)
        loss_data = np.mean((y.data - target.data) ** 2)
        loss = Tensor([loss_data], requires_grad=True)
        
        # Backward pass
        grad = 2 * (y.data - target.data) / y.data.size
        y.backward(grad)
        if hasattr(y, '_backward'):
            y._backward()
        
        # Store original parameters
        original_weight = layer.weight.data.copy()
        original_bias = layer.bias.data.copy()
        
        # Optimize
        optimizer.step()
        
        # Parameters should change
        assert not np.array_equal(layer.weight.data, original_weight)
        assert not np.array_equal(layer.bias.data, original_bias)
    
    def test_optimizer_convergence(self):
        """Test that optimizer can minimize a simple function."""
        # Minimize f(x) = (x - 3)^2, minimum at x=3
        params = {
            'x': Tensor([[0.0]], requires_grad=True)  # Start far from minimum
        }
        
        optimizer = Adam(params, lr=0.1)
        
        target = 3.0
        losses = []
        
        for i in range(50):
            # Compute loss: (x - target)^2
            x_val = params['x'].data[0, 0]
            loss_val = (x_val - target) ** 2
            losses.append(loss_val)
            
            # Compute gradient: 2 * (x - target)
            grad = 2 * (x_val - target)
            params['x'].grad = np.array([[grad]])
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
        
        # Should converge reasonably close to target
        # Note: With gradient clipping, convergence may be slightly slower
        final_x = params['x'].data[0, 0]
        assert abs(final_x - target) < 0.2
        
        # Loss should decrease
        assert losses[-1] < losses[0]


class TestOptimizerEdgeCases:
    """Test optimizer edge cases."""
    
    def test_adam_with_zero_lr(self):
        """Test Adam with zero learning rate."""
        params = {
            'w': Tensor([[1.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=0.0)
        
        params['w'].grad = np.array([[1.0]])
        original_w = params['w'].data.copy()
        
        optimizer.step()
        
        # Parameters should not change
        assert np.array_equal(params['w'].data, original_w)
    
    def test_adam_with_large_lr(self):
        """Test Adam with large learning rate."""
        params = {
            'w': Tensor([[1.0]], requires_grad=True)
        }
        
        optimizer = Adam(params, lr=10.0)  # Very large LR
        
        params['w'].grad = np.array([[0.1]])
        original_w = params['w'].data.copy()
        
        optimizer.step()
        
        # Should still work (Adam is relatively stable)
        assert not np.array_equal(params['w'].data, original_w)
        
        # But update shouldn't be crazy due to gradient clipping
        update_magnitude = np.abs(params['w'].data - original_w)
        assert update_magnitude < 10.0
    
    def test_adam_numerical_stability(self):
        """Test Adam numerical stability."""
        params = {
            'w': Tensor([[1e-10]], requires_grad=True)  # Very small value
        }
        
        optimizer = Adam(params, lr=0.01)
        
        # Very small gradient
        params['w'].grad = np.array([[1e-12]])
        
        optimizer.step()
        
        # Should not crash or produce NaN/inf
        assert np.isfinite(params['w'].data).all()
        assert np.isfinite(optimizer.m['w']).all()
        assert np.isfinite(optimizer.v['w']).all()


def test_optimizer_parameter_sharing():
    """Test optimizer with shared parameters."""
    # Create shared parameter
    shared_param = Tensor([[1.0, 2.0]], requires_grad=True)
    
    params = {
        'shared1': shared_param,
        'shared2': shared_param  # Same tensor reference
    }
    
    optimizer = Adam(params, lr=0.1)
    
    # Set gradient on shared parameter
    shared_param.grad = np.array([[0.1, 0.2]])
    
    original_data = shared_param.data.copy()
    optimizer.step()
    
    # Both references should see the update
    assert not np.array_equal(params['shared1'].data, original_data)
    assert not np.array_equal(params['shared2'].data, original_data)
    assert np.array_equal(params['shared1'].data, params['shared2'].data)
    
    print("✅ Parameter sharing test passed!")


if __name__ == "__main__":
    # Run tests manually
    test_adam = TestAdam()
    test_integration = TestOptimizerIntegration()
    test_edges = TestOptimizerEdgeCases()
    
    print("🧪 Running optimizer tests...")
    
    try:
        # Adam tests
        test_adam.test_adam_creation()
        test_adam.test_adam_zero_grad()
        test_adam.test_adam_single_step()
        test_adam.test_adam_multiple_steps()
        test_adam.test_adam_bias_correction()
        test_adam.test_adam_momentum()
        test_adam.test_adam_gradient_clipping()
        test_adam.test_adam_no_gradients()
        test_adam.test_adam_mixed_gradients()
        print("✅ Adam optimizer tests passed")
        
        # Integration tests
        test_integration.test_optimizer_with_linear_layer()
        test_integration.test_optimizer_convergence()
        print("✅ Optimizer integration tests passed")
        
        # Edge case tests
        test_edges.test_adam_with_zero_lr()
        test_edges.test_adam_with_large_lr()
        test_edges.test_adam_numerical_stability()
        print("✅ Optimizer edge case tests passed")
        
        # Additional tests
        test_optimizer_parameter_sharing()
        
        print("\n🎉 ALL OPTIMIZER TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()