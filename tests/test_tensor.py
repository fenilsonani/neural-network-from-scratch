"""
Comprehensive tensor tests - because we're not fucking around.
"""

try:
    import pytest
except ImportError:
    pytest = None
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch import Tensor, add, matmul, mul, relu, softmax


class TestTensor:
    """Test the fucking tensor class."""

    def test_tensor_creation(self):
        """Test tensor creation with different inputs."""
        # From list
        t1 = Tensor([1.0, 2.0, 3.0])
        assert t1.shape == (3,)
        assert t1.data.dtype in [np.float32, np.float64]

        # From numpy array
        arr = np.array([[1, 2], [3, 4]])
        t2 = Tensor(arr)
        assert t2.shape == (2, 2)

        # From scalar
        t3 = Tensor(5.0)
        assert t3.shape == ()

        # With gradient requirement
        t4 = Tensor([1, 2, 3], requires_grad=True)
        assert t4.requires_grad is True

    def test_tensor_properties(self):
        """Test tensor properties."""
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.shape == (2, 3)
        assert t.data.dtype in [np.float32, np.float64]

    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        t = Tensor([1, 2, 3], requires_grad=True)

        # First gradient
        t.backward(np.array([1, 1, 1]))
        assert np.array_equal(t.grad, [1, 1, 1])

        # Second gradient - should accumulate
        t.backward(np.array([2, 2, 2]))
        assert np.array_equal(t.grad, [3, 3, 3])

        # Zero gradients
        t.zero_grad()
        assert t.grad is None


class TestTensorOperations:
    """Test tensor operations that better fucking work."""

    def test_addition(self):
        """Test tensor addition."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)

        c = add(a, b)
        expected = np.array([[6, 8], [10, 12]])

        assert np.array_equal(c.data, expected)
        assert c.requires_grad is True

        # Test gradients
        c.backward(np.ones_like(c.data))
        if hasattr(c, "_backward"):
            c._backward()

        assert a.grad is not None
        assert b.grad is not None

    def test_multiplication(self):
        """Test tensor multiplication."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[2, 3], [4, 5]], requires_grad=True)

        c = mul(a, b)
        expected = np.array([[2, 6], [12, 20]])

        assert np.array_equal(c.data, expected)
        assert c.requires_grad is True

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)

        c = matmul(a, b)
        expected = np.array([[19, 22], [43, 50]])

        assert np.array_equal(c.data, expected)
        assert c.requires_grad is True

    def test_relu_activation(self):
        """Test ReLU activation."""
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        y = relu(x)

        expected = np.array([0, 0, 0, 1, 2])
        assert np.array_equal(y.data, expected)
        assert y.requires_grad is True

    def test_softmax_activation(self):
        """Test softmax activation."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        y = softmax(x)

        # Check shape
        assert y.shape == (2, 3)

        # Check probabilities sum to 1
        row_sums = np.sum(y.data, axis=1)
        expected_sums = np.ones(2)
        assert np.allclose(row_sums, expected_sums)

        # Check all values are positive
        assert np.all(y.data > 0)
        assert y.requires_grad is True


class TestGradientComputation:
    """Test gradient computation because math matters."""

    def test_simple_chain(self):
        """Test simple computation chain."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)

        # z = x * y + x = x * (y + 1)
        z1 = mul(x, y)
        z2 = add(z1, x)

        # Backward pass
        z2.backward(np.array([1.0]))
        if hasattr(z2, "_backward"):
            z2._backward()

        # Expected gradients:
        # dz/dx = y + 1 = 3 + 1 = 4
        # dz/dy = x = 2
        assert np.allclose(x.grad, [4.0])
        assert np.allclose(y.grad, [2.0])

    def test_matrix_gradients(self):
        """Test matrix operation gradients."""
        a = Tensor([[1, 2]], requires_grad=True)
        b = Tensor([[3], [4]], requires_grad=True)

        c = matmul(a, b)  # Result should be [[11]]
        assert c.shape == (1, 1)
        assert np.allclose(c.data, [[11]])

        # Backward pass
        c.backward(np.array([[1.0]]))
        if hasattr(c, "_backward"):
            c._backward()

        # Check gradients exist
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape


class TestNumericalStability:
    """Test numerical stability because we're not amateur hour."""

    def test_large_numbers(self):
        """Test with large numbers."""
        x = Tensor([100, 200, 300], requires_grad=True)
        y = softmax(x)

        # Should not overflow
        assert np.all(np.isfinite(y.data))
        assert np.allclose(np.sum(y.data), 1.0)

    def test_small_numbers(self):
        """Test with small numbers."""
        x = Tensor([1e-10, 2e-10, 3e-10], requires_grad=True)
        y = relu(x)

        # Should handle small numbers
        assert np.all(np.isfinite(y.data))
        assert np.all(y.data >= 0)

    def test_zero_handling(self):
        """Test zero value handling."""
        x = Tensor([0, 0, 0], requires_grad=True)
        y = relu(x)

        assert np.array_equal(y.data, [0, 0, 0])

    def test_gradient_clipping(self):
        """Test gradient clipping in backward pass."""
        x = Tensor([1000], requires_grad=True)
        x.backward(np.array([1e10]))  # Huge gradient

        # Should be clipped to reasonable range
        assert np.abs(x.grad[0]) <= 10.0


class TestEdgeCases:
    """Test edge cases because shit happens."""

    def test_empty_tensor(self):
        """Test empty tensor."""
        # Empty tensor should be allowed now - just create a 0-dimensional array
        t = Tensor([])
        assert t.data.shape == (0,)

    def test_mismatched_shapes(self):
        """Test operations with mismatched shapes."""
        a = Tensor([[1, 2]])
        b = Tensor([[1], [2], [3]])

        # This should work due to broadcasting
        c = add(a, b)
        assert c.shape == (3, 2)

    def test_no_grad_tensor(self):
        """Test tensor without gradients."""
        a = Tensor([1, 2, 3], requires_grad=False)
        b = Tensor([4, 5, 6], requires_grad=False)

        c = add(a, b)
        assert c.requires_grad is False

        # Backward should not crash
        c.backward(np.ones_like(c.data))

    def test_mixed_grad_requirements(self):
        """Test mixed gradient requirements."""
        a = Tensor([1, 2], requires_grad=True)
        b = Tensor([3, 4], requires_grad=False)

        c = add(a, b)
        assert c.requires_grad is True


def test_comprehensive_workflow():
    """Test a complete workflow that better work."""
    # Create input
    x = Tensor([[1, 2, 3]], requires_grad=True)

    # Create weight matrix
    W = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)

    # Forward pass: y = xW
    y = matmul(x, W)

    # Apply activation
    z = relu(y)

    # Apply softmax
    probs = softmax(z)

    # Compute simple loss (negative log likelihood)
    target = 0  # Target class
    loss_val = -np.log(probs.data[0, target] + 1e-8)

    # Create loss tensor
    loss = Tensor([loss_val], requires_grad=True)

    # Check forward pass results
    assert y.shape == (1, 2)
    assert z.shape == (1, 2)
    assert probs.shape == (1, 2)
    assert np.allclose(np.sum(probs.data), 1.0)

    # Backward pass - manual gradient computation
    # loss = -log(probs[0, target])
    # d_loss/d_probs = -1/probs[0, target] for target, 0 elsewhere
    loss_grad = np.zeros_like(probs.data)
    loss_grad[0, target] = -1.0 / (probs.data[0, target] + 1e-8)

    probs.backward(loss_grad)
    if hasattr(probs, "_backward"):
        probs._backward()

    # Check gradients exist
    assert x.grad is not None
    assert W.grad is not None

    print("✅ Comprehensive workflow test passed!")


if __name__ == "__main__":
    # Run tests manually if pytest not available
    test_tensor = TestTensor()
    test_ops = TestTensorOperations()
    test_grads = TestGradientComputation()
    test_stability = TestNumericalStability()
    test_edges = TestEdgeCases()

    print("🧪 Running tensor tests...")

    try:
        # Basic tensor tests
        test_tensor.test_tensor_creation()
        test_tensor.test_tensor_properties()
        test_tensor.test_gradient_accumulation()
        print("✅ Tensor creation tests passed")

        # Operation tests
        test_ops.test_addition()
        test_ops.test_multiplication()
        test_ops.test_matrix_multiplication()
        test_ops.test_relu_activation()
        test_ops.test_softmax_activation()
        print("✅ Tensor operation tests passed")

        # Gradient tests
        test_grads.test_simple_chain()
        test_grads.test_matrix_gradients()
        print("✅ Gradient computation tests passed")

        # Stability tests
        test_stability.test_large_numbers()
        test_stability.test_small_numbers()
        test_stability.test_zero_handling()
        test_stability.test_gradient_clipping()
        print("✅ Numerical stability tests passed")

        # Edge case tests
        test_edges.test_mismatched_shapes()
        test_edges.test_no_grad_tensor()
        test_edges.test_mixed_grad_requirements()
        print("✅ Edge case tests passed")

        # Comprehensive test
        test_comprehensive_workflow()

        print("\n🎉 ALL TENSOR TESTS PASSED!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
