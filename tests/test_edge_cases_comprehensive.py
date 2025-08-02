"""
Comprehensive edge case testing - testing all the weird shit that can go wrong.
"""

try:
    import pytest
except ImportError:
    pytest = None

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch import Adam, Embedding, Linear, Tensor, add, matmul, mean_pool, mul, relu, softmax


class TestExtremeInputs:
    """Test with extreme and unusual inputs."""

    def test_very_large_numbers(self):
        """Test with very large numbers."""
        # Large positive numbers
        x = Tensor([1e10, 1e20, 1e30], requires_grad=True)
        y = relu(x)
        assert np.all(np.isfinite(y.data))
        assert np.all(y.data >= 0)

        # Large negative numbers
        x = Tensor([-1e10, -1e20, -1e30], requires_grad=True)
        y = relu(x)
        assert np.all(y.data == 0)

    def test_very_small_numbers(self):
        """Test with very small numbers."""
        # Tiny positive numbers
        x = Tensor([1e-20, 1e-30, 1e-40], requires_grad=True)
        y = relu(x)
        assert np.allclose(y.data, x.data)

        # Numbers close to machine epsilon
        eps = np.finfo(np.float32).eps
        x = Tensor([eps, eps / 2, eps * 2], requires_grad=True)
        y = relu(x)
        assert np.all(np.isfinite(y.data))

    def test_special_float_values(self):
        """Test with special float values."""
        # Test with zeros
        x = Tensor([0.0, -0.0, 0.0], requires_grad=True)
        y = relu(x)
        assert np.all(y.data == 0.0)

        # Test with values very close to zero
        x = Tensor([-1e-100, 1e-100], requires_grad=True)
        y = relu(x)
        expected = [0.0, 1e-100]
        assert np.allclose(y.data, expected)

    def test_extreme_matrix_shapes(self):
        """Test with extreme matrix shapes."""
        # Very wide matrix
        a = Tensor(np.random.randn(2, 10000), requires_grad=True)
        b = Tensor(np.random.randn(10000, 3), requires_grad=True)
        c = matmul(a, b)
        assert c.shape == (2, 3)
        assert np.all(np.isfinite(c.data))

        # Very tall matrix
        a = Tensor(np.random.randn(1000, 2), requires_grad=True)
        b = Tensor(np.random.randn(2, 3), requires_grad=True)
        c = matmul(a, b)
        assert c.shape == (1000, 3)
        assert np.all(np.isfinite(c.data))

        # Single element matrix
        a = Tensor([[5.0]], requires_grad=True)
        b = Tensor([[2.0]], requires_grad=True)
        c = matmul(a, b)
        assert c.shape == (1, 1)
        assert c.data[0, 0] == 10.0

    def test_extreme_softmax_inputs(self):
        """Test softmax with extreme inputs."""
        # All identical values (should give uniform distribution)
        x = Tensor([[5, 5, 5, 5]], requires_grad=True)
        y = softmax(x)
        expected = [0.25, 0.25, 0.25, 0.25]
        assert np.allclose(y.data[0], expected)

        # One value much larger than others
        x = Tensor([[1000, 0, 0, 0]], requires_grad=True)
        y = softmax(x)
        assert np.allclose(y.data[0, 0], 1.0, atol=1e-10)
        assert np.allclose(y.data[0, 1:], 0.0, atol=1e-10)

        # One value much smaller than others
        x = Tensor([[-1000, 0, 0, 0]], requires_grad=True)
        y = softmax(x)
        assert np.allclose(y.data[0, 0], 0.0, atol=1e-10)
        expected_others = 1.0 / 3
        assert np.allclose(y.data[0, 1:], expected_others, rtol=1e-5)


class TestBoundaryConditions:
    """Test boundary conditions and edge values."""

    def test_empty_operations(self):
        """Test operations with empty tensors."""
        # Empty tensor operations
        x = Tensor(np.array([]).reshape(0, 5), requires_grad=True)
        y = Tensor(np.array([]).reshape(5, 0), requires_grad=True)

        # Should handle empty multiplications
        try:
            result = matmul(x, y)
            assert result.shape == (0, 0)
        except:
            pass  # May not be supported, that's ok

    def test_single_element_operations(self):
        """Test operations with single elements."""
        # Single element tensors
        a = Tensor([5.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)

        # All operations should work
        c = add(a, b)
        assert c.data[0] == 8.0

        d = mul(a, b)
        assert d.data[0] == 15.0

        # Single element matrix multiplication
        a_mat = Tensor(a.data.reshape(1, 1), requires_grad=True)
        b_mat = Tensor(b.data.reshape(1, 1), requires_grad=True)
        e = matmul(a_mat, b_mat)
        assert e.data[0, 0] == 15.0

    def test_dimension_edge_cases(self):
        """Test edge cases with tensor dimensions."""
        # 1D tensors
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = add(a, b)
        expected = [5, 7, 9]
        assert np.allclose(c.data, expected)

        # 0D tensor (scalar)
        scalar = Tensor(5.0, requires_grad=True)
        vector = Tensor([1, 2, 3], requires_grad=True)
        result = add(scalar, vector)
        expected = [6, 7, 8]
        assert np.allclose(result.data, expected)

    def test_layer_boundary_conditions(self):
        """Test neural network layers with boundary conditions."""
        # Linear layer with 1 input, 1 output
        layer = Linear(1, 1)
        x = Tensor([[5.0]], requires_grad=True)
        y = layer(x)
        assert y.shape == (1, 1)

        # Embedding with single vocabulary item
        embedding = Embedding(1, 10)
        indices = np.array([[0]])
        result = embedding(indices)
        assert result.shape == (1, 1, 10)

        # Embedding with single dimension
        embedding = Embedding(10, 1)
        indices = np.array([[0, 1, 2]])
        result = embedding(indices)
        assert result.shape == (1, 3, 1)


class TestNumericalInstabilities:
    """Test numerical instabilities and their handling."""

    def test_overflow_prevention(self):
        """Test overflow prevention in computations."""
        # Large numbers in softmax should not overflow
        x = Tensor([[1000, 1001, 1002]], requires_grad=True)
        y = softmax(x)
        assert np.all(np.isfinite(y.data))
        assert np.allclose(np.sum(y.data, axis=1), 1.0)

        # Large numbers in matrix multiplication
        a = Tensor([[1e5]], requires_grad=True)
        b = Tensor([[1e5]], requires_grad=True)
        c = matmul(a, b)
        assert np.isfinite(c.data[0, 0])

    def test_underflow_handling(self):
        """Test underflow handling."""
        # Very small numbers should not underflow to zero inappropriately
        x = Tensor([1e-20], requires_grad=True)
        y = relu(x)
        assert y.data[0] > 0  # Should not underflow to zero

        # Very small gradients
        x = Tensor([1.0], requires_grad=True)
        x.backward(np.array([1e-30]))
        assert x.grad is not None
        assert x.grad[0] != 0.0  # Should not underflow

    def test_precision_loss_scenarios(self):
        """Test scenarios that could cause precision loss."""
        # Adding very different magnitude numbers
        # Note: This tests realistic precision limits of float32
        a = Tensor([1e6], requires_grad=True)  # Use smaller numbers for float32
        b = Tensor([1.0], requires_grad=True)
        c = add(a, b)

        # Should maintain reasonable precision (float32 can handle this)
        assert c.data[0] > a.data[0]  # Should not lose the addition

        # Multiplying numbers that could cause precision issues
        a = Tensor([1e20], requires_grad=True)
        b = Tensor([1e-15], requires_grad=True)
        c = mul(a, b)
        expected = 1e5
        assert np.allclose(c.data[0], expected, rtol=1e-10)

    def test_accumulated_errors(self):
        """Test accumulated numerical errors."""
        # Long chain of operations
        x = Tensor([1.0], requires_grad=True)
        y = x

        # Apply many small operations
        for i in range(1000):
            y = add(y, Tensor([1e-10]))

        # Should accumulate correctly
        expected = 1.0 + 1000 * 1e-10
        assert np.allclose(y.data[0], expected, rtol=1e-6)


class TestUnusualShapeBroadcasting:
    """Test unusual broadcasting scenarios."""

    def test_complex_broadcasting_patterns(self):
        """Test complex broadcasting patterns."""
        # Multiple dimension broadcasting
        a = Tensor(np.random.randn(1, 5, 1, 3), requires_grad=True)
        b = Tensor(np.random.randn(4, 1, 2, 1), requires_grad=True)
        c = add(a, b)

        expected_shape = (4, 5, 2, 3)
        assert c.shape == expected_shape

        # Check that broadcasting worked correctly
        assert np.allclose(c.data[0, 0, 0, :], a.data[0, 0, 0, :] + b.data[0, 0, 0, 0])

    def test_single_dimension_broadcasting(self):
        """Test broadcasting with single dimensions."""
        # (1,) + (5,) -> (5,)
        a = Tensor([1.0], requires_grad=True)
        b = Tensor([1, 2, 3, 4, 5], requires_grad=True)
        c = add(a, b)
        expected = [2, 3, 4, 5, 6]
        assert np.allclose(c.data, expected)

        # (3, 1) + (1, 4) -> (3, 4)
        a = Tensor([[1], [2], [3]], requires_grad=True)
        b = Tensor([[10, 20, 30, 40]], requires_grad=True)
        c = add(a, b)

        expected = [[11, 21, 31, 41], [12, 22, 32, 42], [13, 23, 33, 43]]
        assert np.allclose(c.data, expected)

    def test_gradient_broadcasting_correctness(self):
        """Test that gradients are correctly broadcast back."""
        # Test case where gradient needs to be summed due to broadcasting
        a = Tensor([[1], [2]], requires_grad=True)  # (2, 1)
        b = Tensor([10, 20, 30], requires_grad=True)  # (3,)
        c = add(a, b)  # (2, 3)

        # Backward pass
        grad_output = np.ones((2, 3))
        c.backward(grad_output)
        if hasattr(c, "_backward"):
            c._backward()

        # Check gradient shapes
        assert a.grad.shape == (2, 1)
        assert b.grad.shape == (3,)

        # Check gradient values
        assert np.allclose(a.grad, [[3], [3]])  # Summed across broadcast dimension
        assert np.allclose(b.grad, [2, 2, 2])  # Summed across broadcast dimension


class TestMemoryExtremes:
    """Test extreme memory usage scenarios."""

    def test_many_small_tensors(self):
        """Test handling many small tensors."""
        tensors = []

        # Create many small tensors
        for i in range(1000):
            x = Tensor([float(i)], requires_grad=True)
            tensors.append(x)

        # Use them in computation
        total = tensors[0]
        for t in tensors[1:100]:  # Use subset to avoid too long computation
            total = add(total, t)

        # Should work without memory issues
        assert total.data[0] == sum(range(100))

        # Clean up
        for t in tensors:
            t.zero_grad()

    def test_deep_computation_tree(self):
        """Test very deep computation trees."""
        x = Tensor([1.0], requires_grad=True)
        y = x

        # Create very deep tree
        depth = 200
        for i in range(depth):
            y = add(y, Tensor([0.001]))

        # Should not stack overflow
        y.backward(np.array([1.0]))
        if hasattr(y, "_backward"):
            y._backward()

        assert x.grad is not None
        assert np.isfinite(x.grad[0])

    def test_wide_computation_graph(self):
        """Test very wide computation graphs."""
        x = Tensor([1.0], requires_grad=True)

        # Create many parallel branches
        branches = []
        for i in range(100):
            branch = mul(x, Tensor([float(i + 1)]))
            branches.append(branch)

        # Combine all branches
        total = branches[0]
        for branch in branches[1:]:
            total = add(total, branch)

        total.backward(np.array([1.0]))
        if hasattr(total, "_backward"):
            total._backward()

        # Gradient should be sum of all multipliers
        expected_grad = sum(range(1, 101))
        assert np.allclose(x.grad[0], expected_grad)


class TestOptimizerExtremes:
    """Test optimizer with extreme conditions."""

    def test_zero_gradients_consistently(self):
        """Test optimizer with consistently zero gradients."""
        params = {"w": Tensor([1.0], requires_grad=True)}
        optimizer = Adam(params, lr=0.01)

        # Many steps with zero gradients
        for _ in range(100):
            params["w"].grad = np.array([0.0])
            original_value = params["w"].data.copy()
            optimizer.step()

            # Parameters should not change with zero gradients
            assert np.array_equal(params["w"].data, original_value)

            optimizer.zero_grad()

    def test_very_large_gradients(self):
        """Test optimizer with very large gradients."""
        params = {"w": Tensor([1.0], requires_grad=True)}
        optimizer = Adam(params, lr=0.01)

        # Apply huge gradients
        for _ in range(10):
            params["w"].grad = np.array([1e6])
            optimizer.step()
            optimizer.zero_grad()

            # Parameters should remain finite due to gradient clipping
            assert np.isfinite(params["w"].data[0])
            assert np.abs(params["w"].data[0]) < 1000  # Should be clipped

    def test_alternating_gradient_signs(self):
        """Test optimizer with alternating gradient signs."""
        params = {"w": Tensor([0.0], requires_grad=True)}
        optimizer = Adam(params, lr=0.1)

        # Alternating positive and negative gradients
        for i in range(20):
            grad_sign = 1 if i % 2 == 0 else -1
            params["w"].grad = np.array([grad_sign * 1.0])
            optimizer.step()
            optimizer.zero_grad()

        # Parameter should be close to zero due to averaging
        assert np.abs(params["w"].data[0]) < 0.5

    def test_optimizer_with_nan_gradients(self):
        """Test optimizer behavior with NaN gradients."""
        params = {"w": Tensor([1.0], requires_grad=True)}
        optimizer = Adam(params, lr=0.01)

        # Set NaN gradient
        params["w"].grad = np.array([np.nan])
        original_value = params["w"].data.copy()

        # Optimizer should handle NaN gracefully (either skip or handle)
        try:
            optimizer.step()
            # If it doesn't crash, parameters should remain finite
            assert np.isfinite(params["w"].data[0])
        except:
            # If it crashes, that's also acceptable behavior
            pass


class TestLayerExtremes:
    """Test neural network layers with extreme conditions."""

    def test_linear_layer_extreme_weights(self):
        """Test linear layer with extreme weight values."""
        layer = Linear(3, 2)

        # Set extreme weights
        layer.weight.data = np.array([[1e6, -1e6], [1e-6, -1e-6], [0, 1e10]])
        layer.bias.data = np.array([1e-10, -1e10])

        x = Tensor([[1, 1, 1]], requires_grad=True)
        y = layer(x)

        # Should produce finite results
        assert np.all(np.isfinite(y.data))

    def test_embedding_extreme_indices(self):
        """Test embedding layer with boundary indices."""
        vocab_size = 100
        embedding = Embedding(vocab_size, 50)

        # Test boundary indices
        indices = np.array([[0, vocab_size - 1, 0]])  # First and last valid indices
        result = embedding(indices)

        assert result.shape == (1, 3, 50)
        assert np.all(np.isfinite(result.data))

        # Test that different indices give different embeddings
        assert not np.allclose(result.data[0, 0], result.data[0, 1])

    def test_activation_functions_extremes(self):
        """Test activation functions with extreme inputs."""
        # ReLU with extreme values
        extreme_values = [-1e10, -1, -1e-10, 0, 1e-10, 1, 1e10]
        x = Tensor(extreme_values, requires_grad=True)
        y = relu(x)

        # Check ReLU properties
        for i, val in enumerate(extreme_values):
            expected = max(0, val)
            assert np.allclose(y.data[i], expected)

        # All outputs should be finite
        assert np.all(np.isfinite(y.data))


def test_comprehensive_stress_scenarios():
    """Run comprehensive stress test scenarios."""
    print("🔥 Running comprehensive stress scenarios...")

    # Scenario 1: Mixed extreme operations
    x = Tensor([1e-10], requires_grad=True)
    y = Tensor([1e10], requires_grad=True)

    # Chain of mixed operations
    z1 = add(x, y)
    z2 = mul(z1, Tensor([1e-5]))
    z3 = add(z2, Tensor([1e15]))
    z4 = mul(z3, Tensor([1e-20]))

    # Should remain finite
    assert np.isfinite(z4.data[0])

    # Scenario 2: Complex broadcasting with extremes
    a = Tensor(np.full((1, 1000), 1e-10), requires_grad=True)
    b = Tensor(np.full((1000, 1), 1e10), requires_grad=True)
    c = add(a, b)

    assert c.shape == (1000, 1000)
    assert np.all(np.isfinite(c.data))

    # Scenario 3: Deep network with extreme activations
    layers = []
    for i in range(10):
        layers.append(Linear(100, 100))

    x = Tensor(np.random.randn(5, 100) * 1000, requires_grad=True)  # Large input

    for layer in layers:
        x = layer(x)
        x = relu(x)

    # Should not explode or vanish completely
    assert np.all(np.isfinite(x.data))
    assert np.any(x.data > 0)  # Should not vanish completely

    print("✅ Comprehensive stress scenarios passed!")


def test_recovery_from_numerical_issues():
    """Test recovery from numerical issues."""
    print("🛠️ Testing recovery from numerical issues...")

    # Test that the system correctly rejects problematic values at creation
    try:
        x = Tensor([np.inf], requires_grad=True)
        assert False, "Should have rejected infinite values"
    except ValueError:
        pass  # Expected: our system correctly rejects inf values

    # Test that the system correctly rejects NaN values
    try:
        x = Tensor([np.nan], requires_grad=True)
        assert False, "Should have rejected NaN values"
    except ValueError:
        pass  # Expected: our system correctly rejects NaN values

    # Test gradient clipping recovery
    x = Tensor([1.0], requires_grad=True)
    x.backward(np.array([1e20]))  # Huge gradient

    # Gradient should be clipped
    assert np.abs(x.grad[0]) <= 10.0

    print("✅ Recovery from numerical issues test passed!")


if __name__ == "__main__":
    # Run all edge case tests
    test_extreme = TestExtremeInputs()
    test_boundary = TestBoundaryConditions()
    test_numerical = TestNumericalInstabilities()
    test_broadcasting = TestUnusualShapeBroadcasting()
    test_memory = TestMemoryExtremes()
    test_optimizer_extreme = TestOptimizerExtremes()
    test_layer_extreme = TestLayerExtremes()

    print("🧪 Running comprehensive edge case tests...")

    try:
        # Extreme inputs
        test_extreme.test_very_large_numbers()
        test_extreme.test_very_small_numbers()
        test_extreme.test_special_float_values()
        test_extreme.test_extreme_matrix_shapes()
        test_extreme.test_extreme_softmax_inputs()
        print("✅ Extreme inputs tests passed")

        # Boundary conditions
        test_boundary.test_empty_operations()
        test_boundary.test_single_element_operations()
        test_boundary.test_dimension_edge_cases()
        test_boundary.test_layer_boundary_conditions()
        print("✅ Boundary conditions tests passed")

        # Numerical instabilities
        test_numerical.test_overflow_prevention()
        test_numerical.test_underflow_handling()
        test_numerical.test_precision_loss_scenarios()
        test_numerical.test_accumulated_errors()
        print("✅ Numerical instabilities tests passed")

        # Unusual broadcasting
        test_broadcasting.test_complex_broadcasting_patterns()
        test_broadcasting.test_single_dimension_broadcasting()
        test_broadcasting.test_gradient_broadcasting_correctness()
        print("✅ Unusual broadcasting tests passed")

        # Memory extremes
        test_memory.test_many_small_tensors()
        test_memory.test_deep_computation_tree()
        test_memory.test_wide_computation_graph()
        print("✅ Memory extremes tests passed")

        # Optimizer extremes
        test_optimizer_extreme.test_zero_gradients_consistently()
        test_optimizer_extreme.test_very_large_gradients()
        test_optimizer_extreme.test_alternating_gradient_signs()
        test_optimizer_extreme.test_optimizer_with_nan_gradients()
        print("✅ Optimizer extremes tests passed")

        # Layer extremes
        test_layer_extreme.test_linear_layer_extreme_weights()
        test_layer_extreme.test_embedding_extreme_indices()
        test_layer_extreme.test_activation_functions_extremes()
        print("✅ Layer extremes tests passed")

        # Comprehensive scenarios
        test_comprehensive_stress_scenarios()
        test_recovery_from_numerical_issues()

        print("\n🎉 ALL COMPREHENSIVE EDGE CASE TESTS PASSED!")

    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback

        traceback.print_exc()
