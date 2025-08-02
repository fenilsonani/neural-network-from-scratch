"""Mathematical correctness tests for neural architecture framework.

This module contains comprehensive tests to ensure all mathematical operations
are implemented correctly with proper gradients and numerical stability.
"""

import math
from typing import Callable, List, Tuple

import numpy as np
import pytest

# Import from the neural architecture framework
from neural_arch.core import Tensor
from neural_arch.functional import (
    cross_entropy_loss,
    gelu,
    mish,
    mse_loss,
    relu,
    sigmoid,
    silu,
    softmax,
    swiglu,
    tanh,
)
from neural_arch.functional.arithmetic import add, div, matmul, mul, sub
from neural_arch.nn import BatchNorm1d, BatchNorm2d, LayerNorm, Linear, RMSNorm


class TestMathematicalCorrectness:
    """Test suite for mathematical correctness."""

    def numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Compute numerical gradient using finite differences.

        Args:
            func: Function to compute gradient for
            x: Input point
            h: Step size for finite differences

        Returns:
            Numerical gradient
        """
        grad = np.zeros_like(x)

        for i in range(x.size):
            # Create perturbation
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus.flat[i] += h
            x_minus.flat[i] -= h

            # Compute finite difference
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            grad.flat[i] = (f_plus - f_minus) / (2 * h)

        return grad

    def assert_gradients_close(
        self,
        analytical_grad: np.ndarray,
        numerical_grad: np.ndarray,
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        """Assert that analytical and numerical gradients are close."""
        np.testing.assert_allclose(
            analytical_grad,
            numerical_grad,
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradients don't match:\nAnalytical: {analytical_grad}\nNumerical: {numerical_grad}",
        )

    def test_gelu_accuracy(self):
        """Test GELU implementation accuracy against exact formula."""
        x_values = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])

        for x_val in x_values:
            x = Tensor([x_val], requires_grad=True)

            # Test exact GELU
            y_exact = gelu(x, approximate=False)

            # Test approximation GELU
            y_approx = gelu(x, approximate=True)

            # Compute expected exact GELU using scipy.special.erf if available
            try:
                from scipy.special import erf

                expected = 0.5 * x_val * (1 + erf(x_val / math.sqrt(2)))

                # Exact GELU should match within machine precision
                np.testing.assert_allclose(y_exact.data, [expected], rtol=1e-10, atol=1e-12)

                # Approximation should be close but not exact
                error = abs(y_approx.data[0] - expected)
                assert error < 0.001, f"GELU approximation error too large: {error}"

            except ImportError:
                # If scipy not available, just check that exact and approx are different
                assert abs(y_exact.data[0] - y_approx.data[0]) > 1e-8

    def test_gelu_gradient_correctness(self):
        """Test GELU gradient correctness using numerical differentiation."""
        x_values = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

        for x_val in x_values:
            # Test exact GELU gradients
            x = Tensor([x_val], requires_grad=True)
            y = gelu(x, approximate=False)
            y.backward()

            # Compute numerical gradient
            def gelu_func(x_np):
                try:
                    from scipy.special import erf

                    return 0.5 * x_np[0] * (1 + erf(x_np[0] / math.sqrt(2)))
                except ImportError:
                    # Fallback to approximation for numerical gradient
                    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
                    inner = sqrt_2_over_pi * (x_np[0] + 0.044715 * x_np[0] ** 3)
                    return 0.5 * x_np[0] * (1 + math.tanh(inner))

            numerical_grad = self.numerical_gradient(gelu_func, np.array([x_val]))

            self.assert_gradients_close(x.grad, numerical_grad, rtol=1e-3, atol=1e-5)

    def test_modern_activations(self):
        """Test modern activation functions for mathematical correctness."""
        x_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        for x_val in x_values:
            x = Tensor([x_val], requires_grad=True)

            # Test SiLU/Swish
            y_silu = silu(x)
            expected_silu = x_val / (1 + math.exp(-x_val))
            np.testing.assert_allclose(y_silu.data, [expected_silu], rtol=1e-6)

            # Test Mish
            x_mish = Tensor([x_val], requires_grad=True)
            y_mish = mish(x_mish)
            softplus_val = max(x_val, 0) + math.log1p(math.exp(-abs(x_val)))
            expected_mish = x_val * math.tanh(softplus_val)
            np.testing.assert_allclose(y_mish.data, [expected_mish], rtol=1e-6)

    def test_swiglu_mathematical_properties(self):
        """Test SwiGLU mathematical properties."""
        # SwiGLU requires even dimension
        x = Tensor(np.random.randn(2, 8), requires_grad=True)  # Even last dimension
        y = swiglu(x)

        # Output should have half the input dimension
        assert y.shape == (2, 4)

        # Test with odd dimension (should raise error)
        x_odd = Tensor(np.random.randn(2, 7), requires_grad=True)
        with pytest.raises(ValueError, match="even dimension"):
            swiglu(x_odd)

    def test_layernorm_mathematical_correctness(self):
        """Test LayerNorm mathematical correctness."""
        # Create test data
        batch_size, features = 4, 8
        x = Tensor(np.random.randn(batch_size, features), requires_grad=True)

        # Test LayerNorm
        layer_norm = LayerNorm(features, eps=1e-5)
        y = layer_norm(x)

        # Verify normalization properties
        # Mean should be close to 0, std should be close to 1
        y_mean = np.mean(y.data, axis=-1)
        y_var = np.var(y.data, axis=-1, ddof=0)

        np.testing.assert_allclose(y_mean, 0.0, atol=1e-6)
        np.testing.assert_allclose(y_var, 1.0, rtol=1e-5)

        # Test gradient flow
        loss = Tensor(np.sum(y.data**2))
        loss.backward()

        assert x.grad is not None
        assert layer_norm.weight.grad is not None
        assert layer_norm.bias.grad is not None

    def test_batchnorm_running_statistics(self):
        """Test BatchNorm running statistics correctness."""
        num_features = 4
        batch_norm = BatchNorm1d(num_features, momentum=0.1)

        # Test multiple batches
        batch1 = Tensor(np.random.randn(8, num_features))
        batch2 = Tensor(np.random.randn(8, num_features))

        # Training mode
        batch_norm.train()

        # Process first batch
        y1 = batch_norm(batch1)
        running_mean_1 = batch_norm.running_mean.copy()
        running_var_1 = batch_norm.running_var.copy()

        # Process second batch
        y2 = batch_norm(batch2)
        running_mean_2 = batch_norm.running_mean.copy()
        running_var_2 = batch_norm.running_var.copy()

        # Running statistics should change
        assert not np.allclose(running_mean_1, running_mean_2)
        assert not np.allclose(running_var_1, running_var_2)
        assert batch_norm.num_batches_tracked == 2

        # Test evaluation mode
        batch_norm.eval()
        y3 = batch_norm(batch1)  # Should use running stats, not batch stats

        # Running statistics should not change in eval mode
        running_mean_3 = batch_norm.running_mean.copy()
        np.testing.assert_allclose(running_mean_2, running_mean_3)

    def test_rmsnorm_vs_layernorm(self):
        """Test RMSNorm mathematical differences from LayerNorm."""
        features = 8
        x = Tensor(np.random.randn(4, features), requires_grad=True)

        # Test RMSNorm
        rms_norm = RMSNorm(features)
        y_rms = rms_norm(x)

        # Test LayerNorm
        layer_norm = LayerNorm(features)
        y_ln = layer_norm(x)

        # Outputs should be different (RMSNorm doesn't center)
        assert not np.allclose(y_rms.data, y_ln.data, rtol=1e-3)

        # RMSNorm should have RMS = 1 (approximately)
        rms_values = np.sqrt(np.mean(y_rms.data**2, axis=-1))
        np.testing.assert_allclose(rms_values, 1.0, rtol=1e-5)

    def test_arithmetic_operations_gradients(self):
        """Test arithmetic operations gradient correctness."""
        # Test basic arithmetic operations
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)

        # Addition
        c = add(a, b)
        c.backward(np.array([1.0, 1.0]))
        np.testing.assert_allclose(a.grad, [1.0, 1.0])
        np.testing.assert_allclose(b.grad, [1.0, 1.0])

        # Reset gradients
        a.zero_grad()
        b.zero_grad()

        # Multiplication
        c = mul(a, b)
        c.backward(np.array([1.0, 1.0]))
        np.testing.assert_allclose(a.grad, b.data)  # grad_a = b
        np.testing.assert_allclose(b.grad, a.data)  # grad_b = a

    def test_matmul_gradient_correctness(self):
        """Test matrix multiplication gradient correctness."""
        # Test matmul gradients
        A = Tensor(np.random.randn(3, 4), requires_grad=True)
        B = Tensor(np.random.randn(4, 5), requires_grad=True)

        C = matmul(A, B)
        grad_output = np.random.randn(*C.shape)
        C.backward(grad_output)

        # Expected gradients
        expected_grad_A = grad_output @ B.data.T
        expected_grad_B = A.data.T @ grad_output

        np.testing.assert_allclose(A.grad, expected_grad_A, rtol=1e-6)
        np.testing.assert_allclose(B.grad, expected_grad_B, rtol=1e-6)

    def test_cross_entropy_loss_correctness(self):
        """Test cross-entropy loss mathematical correctness."""
        # Create test data
        batch_size, num_classes = 4, 5
        logits = Tensor(np.random.randn(batch_size, num_classes), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 3]))  # Class indices

        # Compute loss
        loss = cross_entropy_loss(logits, targets)

        # Manual computation for verification
        softmax_probs = np.exp(logits.data) / np.sum(np.exp(logits.data), axis=1, keepdims=True)
        expected_loss = -np.mean(
            np.log(softmax_probs[np.arange(batch_size), targets.data.astype(int)])
        )

        np.testing.assert_allclose(loss.data, expected_loss, rtol=1e-6)

        # Test gradients
        loss.backward()
        assert logits.grad is not None

        # Expected gradient: softmax - one_hot
        expected_grad = softmax_probs.copy()
        expected_grad[np.arange(batch_size), targets.data.astype(int)] -= 1
        expected_grad /= batch_size

        np.testing.assert_allclose(logits.grad, expected_grad, rtol=1e-6)

    def test_numerical_stability(self):
        """Test numerical stability of operations."""
        # Test with extreme values
        extreme_values = [-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0]

        for val in extreme_values:
            x = Tensor([val])

            # Sigmoid should be stable
            y_sigmoid = sigmoid(x)
            assert np.isfinite(y_sigmoid.data).all()
            assert 0 <= y_sigmoid.data[0] <= 1

            # Softmax should be stable
            x_2d = Tensor([[val, val - 1]])
            y_softmax = softmax(x_2d)
            assert np.isfinite(y_softmax.data).all()
            assert abs(np.sum(y_softmax.data) - 1.0) < 1e-6

    def test_linear_layer_initialization(self):
        """Test Linear layer weight initialization schemes."""
        in_features, out_features = 10, 5

        # Test different initialization schemes
        init_schemes = ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal"]

        for scheme in init_schemes:
            layer = Linear(in_features, out_features, weight_init=scheme)

            # Check weight statistics
            weight_std = np.std(layer.weight.data)

            if "xavier" in scheme:
                expected_std = math.sqrt(2.0 / (in_features + out_features))
            elif "he" in scheme:
                expected_std = math.sqrt(2.0 / in_features)

            if "uniform" in scheme:
                # For uniform distribution, adjust for variance
                expected_std *= math.sqrt(3)

            # Allow some tolerance for random initialization
            assert abs(weight_std - expected_std) < 0.2 * expected_std

    def test_gradient_flow_deep_network(self):
        """Test gradient flow through a deep network."""
        # Create a simple deep network
        layers = [Linear(10, 8), LayerNorm(8), Linear(8, 6), BatchNorm1d(6), Linear(6, 1)]

        # Forward pass
        x = Tensor(np.random.randn(4, 10), requires_grad=True)

        for layer in layers:
            if isinstance(layer, BatchNorm1d):
                layer.train()  # Ensure training mode
            x = layer(x)
            if hasattr(layer, "weight"):
                # Apply ReLU activation after layers with weights
                x = relu(x)

        # Backward pass
        loss = Tensor(np.sum(x.data**2))
        loss.backward()

        # Check that all parameters have gradients
        for layer in layers:
            if hasattr(layer, "weight") and layer.weight is not None:
                assert layer.weight.grad is not None
                assert np.isfinite(layer.weight.grad).all()
            if hasattr(layer, "bias") and layer.bias is not None:
                assert layer.bias.grad is not None
                assert np.isfinite(layer.bias.grad).all()

    def test_mathematical_properties(self):
        """Test mathematical properties of operations."""
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([4.0, 5.0, 6.0])

        # Test commutativity of addition
        result1 = add(x, y)
        result2 = add(y, x)
        np.testing.assert_allclose(result1.data, result2.data)

        # Test commutativity of multiplication
        result1 = mul(x, y)
        result2 = mul(y, x)
        np.testing.assert_allclose(result1.data, result2.data)

        # Test associativity of addition
        z = Tensor([7.0, 8.0, 9.0])
        result1 = add(add(x, y), z)
        result2 = add(x, add(y, z))
        np.testing.assert_allclose(result1.data, result2.data, rtol=1e-15)


class TestNumericalPrecision:
    """Test numerical precision and edge cases."""

    def test_precision_consistency(self):
        """Test precision consistency across operations."""
        # Test with different precisions
        x_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # Operations should maintain appropriate precision
        t_f32 = Tensor(x_f32)
        t_f64 = Tensor(x_f64)

        y_f32 = relu(t_f32)
        y_f64 = relu(t_f64)

        # Results should match within precision limits
        np.testing.assert_allclose(
            y_f32.data.astype(np.float64), y_f64.data, rtol=1e-6  # Single precision tolerance
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with zeros
        x_zero = Tensor([0.0, 0.0, 0.0])
        y_relu = relu(x_zero)
        np.testing.assert_array_equal(y_relu.data, [0.0, 0.0, 0.0])

        # Test with very small numbers
        x_small = Tensor([1e-10, 1e-15, 1e-20])
        y_small = gelu(x_small)
        assert np.isfinite(y_small.data).all()

        # Test with large numbers
        x_large = Tensor([100.0, 1000.0])
        y_large = sigmoid(x_large)
        np.testing.assert_allclose(y_large.data, [1.0, 1.0], rtol=1e-6)


if __name__ == "__main__":
    # Run specific tests if called directly
    test_suite = TestMathematicalCorrectness()

    print("Running mathematical correctness tests...")

    # Run key tests
    try:
        test_suite.test_gelu_accuracy()
        print("✓ GELU accuracy test passed")

        test_suite.test_gelu_gradient_correctness()
        print("✓ GELU gradient correctness test passed")

        test_suite.test_layernorm_mathematical_correctness()
        print("✓ LayerNorm mathematical correctness test passed")

        test_suite.test_batchnorm_running_statistics()
        print("✓ BatchNorm running statistics test passed")

        test_suite.test_cross_entropy_loss_correctness()
        print("✓ Cross-entropy loss correctness test passed")

        test_suite.test_numerical_stability()
        print("✓ Numerical stability test passed")

        print("\n🎉 All mathematical correctness tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
