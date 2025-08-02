"""
Advanced tensor operations and mathematical functions - extensive testing.
"""

try:
    import pytest
except ImportError:
    pytest = None

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch import Tensor, add, matmul, mean_pool, mul, relu, softmax


class TestAdvancedTensorOperations:
    """Test advanced tensor operations and mathematical functions."""

    def test_tensor_broadcasting_all_combinations(self):
        """Test all possible broadcasting combinations."""
        # Scalar + Vector
        a = Tensor([5.0], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        c = add(a, b)
        assert c.shape == (3,)
        assert np.allclose(c.data, [6, 7, 8])

        # Vector + Matrix
        a = Tensor([1, 2], requires_grad=True)
        b = Tensor([[10, 20], [30, 40]], requires_grad=True)
        c = add(a, b)
        assert c.shape == (2, 2)
        assert np.allclose(c.data, [[11, 22], [31, 42]])

        # Different broadcasting patterns
        a = Tensor([[1], [2]], requires_grad=True)  # (2, 1)
        b = Tensor([10, 20, 30], requires_grad=True)  # (3,)
        c = add(a, b)
        assert c.shape == (2, 3)
        expected = [[11, 21, 31], [12, 22, 32]]
        assert np.allclose(c.data, expected)

    def test_tensor_elementwise_operations_large_scale(self):
        """Test elementwise operations on large tensors."""
        np.random.seed(42)

        # Large matrices
        a = Tensor(np.random.randn(100, 50), requires_grad=True)
        b = Tensor(np.random.randn(100, 50), requires_grad=True)

        # Addition
        c = add(a, b)
        assert c.shape == (100, 50)
        assert np.allclose(c.data, a.data + b.data)

        # Multiplication
        d = mul(a, b)
        assert d.shape == (100, 50)
        assert np.allclose(d.data, a.data * b.data)

        # Chain operations
        e = add(mul(a, b), a)
        expected = a.data * b.data + a.data
        assert np.allclose(e.data, expected)

    def test_matrix_multiplication_various_shapes(self):
        """Test matrix multiplication with various shape combinations."""
        test_cases = [
            ((1, 5), (5, 1)),  # Vector outer product
            ((10, 3), (3, 7)),  # Standard case
            ((2, 4, 3), (3, 5)),  # Batch matrix multiplication
            ((5, 1), (1, 10)),  # Outer product variations
        ]

        for shape_a, shape_b in test_cases:
            a = Tensor(np.random.randn(*shape_a), requires_grad=True)
            b = Tensor(np.random.randn(*shape_b), requires_grad=True)

            c = matmul(a, b)
            expected_shape = shape_a[:-1] + shape_b[-1:]
            assert c.shape == expected_shape

            # Verify computation
            expected = np.matmul(a.data, b.data)
            assert np.allclose(c.data, expected)

    def test_activation_functions_edge_cases(self):
        """Test activation functions with edge cases."""
        # ReLU with various inputs
        test_inputs = [
            [-1000, -1, -1e-10, 0, 1e-10, 1, 1000],
            np.random.randn(50, 30),
            np.full((10, 10), -np.inf),
            np.full((5, 5), np.inf),
        ]

        for input_data in test_inputs:
            if np.any(np.isinf(input_data)):
                continue  # Skip inf cases for now

            x = Tensor(input_data, requires_grad=True)
            y = relu(x)

            # Check ReLU property
            assert np.all(y.data >= 0)
            expected = np.maximum(0, x.data)
            assert np.allclose(y.data, expected)

    def test_softmax_numerical_stability(self):
        """Test softmax numerical stability with extreme values."""
        # Very large values (should not overflow)
        x = Tensor([[1000, 1001, 1002]], requires_grad=True)
        y = softmax(x)
        assert np.all(np.isfinite(y.data))
        assert np.allclose(np.sum(y.data, axis=1), 1.0)

        # Very small values
        x = Tensor([[-1000, -1001, -1002]], requires_grad=True)
        y = softmax(x)
        assert np.all(np.isfinite(y.data))
        assert np.allclose(np.sum(y.data, axis=1), 1.0)

        # Mixed extreme values
        x = Tensor([[-1000, 0, 1000]], requires_grad=True)
        y = softmax(x)
        assert np.all(np.isfinite(y.data))
        assert np.allclose(np.sum(y.data, axis=1), 1.0)

        # Batch processing
        x = Tensor(np.random.randn(10, 20) * 100, requires_grad=True)
        y = softmax(x)
        assert np.all(np.isfinite(y.data))
        assert np.allclose(np.sum(y.data, axis=1), 1.0)

    def test_mean_pooling_various_axes(self):
        """Test mean pooling with different axes and shapes."""
        # 3D tensor pooling
        x = Tensor(np.random.randn(5, 10, 8), requires_grad=True)

        # Pool along sequence dimension (axis=1)
        y1 = mean_pool(x, axis=1)
        assert y1.shape == (5, 8)
        expected = np.mean(x.data, axis=1)
        assert np.allclose(y1.data, expected)

        # Pool along batch dimension (axis=0)
        y0 = mean_pool(x, axis=0)
        assert y0.shape == (10, 8)
        expected = np.mean(x.data, axis=0)
        assert np.allclose(y0.data, expected)

        # Test gradient flow
        y1.backward(np.ones_like(y1.data))
        if hasattr(y1, "_backward"):
            y1._backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def test_gradient_accumulation_precision(self):
        """Test precision in gradient accumulation."""
        x = Tensor([1.0], requires_grad=True)

        # Accumulate many small gradients
        for i in range(1000):
            x.backward(np.array([1e-6]))

        expected_grad = 1000 * 1e-6
        assert np.allclose(x.grad, [expected_grad], rtol=1e-10)

    def test_floating_point_edge_cases(self):
        """Test handling of floating point edge cases."""
        # Very small numbers
        x = Tensor([1e-20, 1e-15, 1e-10], requires_grad=True)
        y = relu(x)
        assert np.all(np.isfinite(y.data))

        # Numbers close to zero
        x = Tensor([-1e-15, 0, 1e-15], requires_grad=True)
        y = relu(x)
        expected = [0, 0, 1e-15]
        assert np.allclose(y.data, expected)

    def test_precision_in_matrix_operations(self):
        """Test precision in matrix operations."""
        # Create matrices with known properties
        n = 50
        A = Tensor(np.eye(n) * 0.1, requires_grad=True)
        B = Tensor(np.eye(n) * 10.0, requires_grad=True)

        # A @ B should equal identity
        C = matmul(A, B)
        expected = np.eye(n)
        assert np.allclose(C.data, expected, atol=1e-10)

    def test_numerical_derivatives(self):
        """Test numerical derivative computation vs analytical."""

        def finite_diff(f, x, h=1e-7):
            """Compute finite difference approximation."""
            x_plus = x + h
            x_minus = x - h
            return (f(x_plus) - f(x_minus)) / (2 * h)

        # Test on simple function: f(x) = x^2
        x_val = 3.0
        x = Tensor([x_val], requires_grad=True)
        y = mul(x, x)  # x^2

        y.backward(np.array([1.0]))
        if hasattr(y, "_backward"):
            y._backward()

        analytical_grad = x.grad[0]

        # Numerical gradient
        f = lambda val: val * val
        numerical_grad = finite_diff(f, x_val)

        # Should match analytical gradient (2x = 6)
        assert np.allclose(analytical_grad, 6.0, rtol=1e-5)
        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-5)


class TestComplexComputationGraphs:
    """Test complex computation graphs and scenarios."""

    def test_deep_computation_chain(self):
        """Test deep computation chains."""
        x = Tensor([2.0], requires_grad=True)

        # Create deep chain: ((((x + 1) * 2) + 1) * 2) ...
        y = x
        for i in range(20):
            y = add(y, Tensor([1.0]))
            y = mul(y, Tensor([1.1]))

        # Backward pass
        y.backward(np.array([1.0]))
        if hasattr(y, "_backward"):
            y._backward()

        # Should not crash and should have finite gradients
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad))

    def test_branching_computation_graph(self):
        """Test branching computation graphs."""
        x = Tensor([3.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)

        # Create branching graph
        a = mul(x, y)  # x * y
        b = add(x, y)  # x + y
        c = mul(a, b)  # (x * y) * (x + y)
        d = add(a, b)  # (x * y) + (x + y)
        e = add(c, d)  # Final result

        e.backward(np.array([1.0]))
        if hasattr(e, "_backward"):
            e._backward()

        # Both x and y should have gradients
        assert x.grad is not None
        assert y.grad is not None
        assert np.all(np.isfinite(x.grad))
        assert np.all(np.isfinite(y.grad))

    def test_parameter_sharing_gradients(self):
        """Test gradient accumulation with parameter sharing."""
        shared_param = Tensor([1.0], requires_grad=True)

        # Use parameter multiple times
        a = mul(shared_param, Tensor([2.0]))
        b = mul(shared_param, Tensor([3.0]))
        c = add(a, b)

        c.backward(np.array([1.0]))
        if hasattr(c, "_backward"):
            c._backward()

        # Gradient should be sum of partial derivatives: 2 + 3 = 5
        expected_grad = 5.0
        assert np.allclose(shared_param.grad, [expected_grad])

    def test_multiple_output_gradients(self):
        """Test handling multiple outputs from same computation."""
        x = Tensor([2.0], requires_grad=True)

        # Multiple outputs
        y1 = mul(x, Tensor([2.0]))
        y2 = mul(x, Tensor([3.0]))

        # Backward from both outputs
        y1.backward(np.array([1.0]))
        if hasattr(y1, "_backward"):
            y1._backward()

        y2.backward(np.array([1.0]))
        if hasattr(y2, "_backward"):
            y2._backward()

        # Gradients should accumulate: 2 + 3 = 5
        expected_grad = 5.0
        assert np.allclose(x.grad, [expected_grad])


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_large_tensor_operations(self):
        """Test operations on large tensors."""
        # Large tensors
        shape = (500, 300)
        a = Tensor(np.random.randn(*shape), requires_grad=True)
        b = Tensor(np.random.randn(*shape), requires_grad=True)

        # Operations should complete without memory issues
        c = add(a, b)
        d = mul(c, a)

        # Backward pass
        d.backward(np.ones_like(d.data))
        if hasattr(d, "_backward"):
            d._backward()

        # Check gradients exist and are finite
        assert a.grad is not None
        assert b.grad is not None
        assert np.all(np.isfinite(a.grad))
        assert np.all(np.isfinite(b.grad))

    def test_memory_cleanup(self):
        """Test that gradients can be properly cleaned up."""
        tensors = []

        # Create many tensors with gradients
        for i in range(100):
            x = Tensor(np.random.randn(10, 10), requires_grad=True)
            y = mul(x, x)
            y.backward(np.ones_like(y.data))
            if hasattr(y, "_backward"):
                y._backward()
            tensors.append(x)

        # Zero all gradients
        for tensor in tensors:
            tensor.zero_grad()
            assert tensor.grad is None

    def test_batch_processing_consistency(self):
        """Test consistency between single and batch processing."""
        # Single item processing
        x_single = Tensor([[1, 2, 3]], requires_grad=True)
        y_single = softmax(x_single)

        # Batch processing
        x_batch = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)
        y_batch = softmax(x_batch)

        # First row of batch should match single processing
        assert np.allclose(y_single.data[0], y_batch.data[0])


def test_comprehensive_mathematical_properties():
    """Test mathematical properties of operations."""
    np.random.seed(42)

    # Test associativity: (a + b) + c = a + (b + c)
    a = Tensor(np.random.randn(5, 5), requires_grad=True)
    b = Tensor(np.random.randn(5, 5), requires_grad=True)
    c = Tensor(np.random.randn(5, 5), requires_grad=True)

    left = add(add(a, b), c)
    right = add(a, add(b, c))

    assert np.allclose(left.data, right.data, rtol=1e-6)

    # Test commutativity: a + b = b + a
    ab = add(a, b)
    ba = add(b, a)
    assert np.allclose(ab.data, ba.data)

    # Test distributivity: a * (b + c) = a * b + a * c
    bc = add(b, c)
    left = mul(a, bc)
    right = add(mul(a, b), mul(a, c))
    assert np.allclose(left.data, right.data, rtol=1e-6)

    print("✅ Mathematical properties test passed!")


def test_stress_testing():
    """Stress test the system with extreme scenarios."""
    print("🔥 Running stress tests...")

    # Very deep computation graph
    x = Tensor([1.0], requires_grad=True)
    y = x

    for i in range(100):
        y = add(y, Tensor([0.01]))
        if i % 20 == 0:
            y = mul(y, Tensor([0.99]))  # Prevent explosion

    y.backward(np.array([1.0]))
    if hasattr(y, "_backward"):
        y._backward()

    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))

    # Many parallel operations
    base = Tensor([1.0], requires_grad=True)
    results = []

    for i in range(50):
        y = mul(base, Tensor([i + 1]))
        results.append(y)

    # Sum all results
    total = results[0]
    for r in results[1:]:
        total = add(total, r)

    total.backward(np.array([1.0]))
    if hasattr(total, "_backward"):
        total._backward()

    # Gradient should be sum of 1 to 50
    expected_grad = sum(range(1, 51))
    assert np.allclose(base.grad, [expected_grad])

    print("✅ Stress tests passed!")


if __name__ == "__main__":
    # Run tests manually
    test_advanced = TestAdvancedTensorOperations()
    test_precision = TestNumericalPrecision()
    test_graphs = TestComplexComputationGraphs()
    test_memory = TestMemoryAndPerformance()

    print("🧪 Running advanced tensor operation tests...")

    try:
        # Advanced operations
        test_advanced.test_tensor_broadcasting_all_combinations()
        test_advanced.test_tensor_elementwise_operations_large_scale()
        test_advanced.test_matrix_multiplication_various_shapes()
        test_advanced.test_activation_functions_edge_cases()
        test_advanced.test_softmax_numerical_stability()
        test_advanced.test_mean_pooling_various_axes()
        print("✅ Advanced tensor operations tests passed")

        # Numerical precision
        test_precision.test_gradient_accumulation_precision()
        test_precision.test_floating_point_edge_cases()
        test_precision.test_precision_in_matrix_operations()
        test_precision.test_numerical_derivatives()
        print("✅ Numerical precision tests passed")

        # Complex graphs
        test_graphs.test_deep_computation_chain()
        test_graphs.test_branching_computation_graph()
        test_graphs.test_parameter_sharing_gradients()
        test_graphs.test_multiple_output_gradients()
        print("✅ Complex computation graph tests passed")

        # Memory and performance
        test_memory.test_large_tensor_operations()
        test_memory.test_memory_cleanup()
        test_memory.test_batch_processing_consistency()
        print("✅ Memory and performance tests passed")

        # Comprehensive tests
        test_comprehensive_mathematical_properties()
        test_stress_testing()

        print("\n🎉 ALL ADVANCED TESTS PASSED!")

    except Exception as e:
        print(f"❌ Advanced test failed: {e}")
        import traceback

        traceback.print_exc()
