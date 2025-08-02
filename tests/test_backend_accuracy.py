"""Test numerical accuracy and precision across different backends."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.backends import available_backends, get_backend


class TestNumericalAccuracy:
    """Test numerical accuracy across backends."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.backends = [get_backend(name) for name in available_backends()]

    def test_small_number_accuracy(self):
        """Test accuracy with very small numbers."""
        small_vals = np.array([1e-10, 1e-20, 1e-30], dtype=np.float32)

        for backend in self.backends:
            arr = backend.from_numpy(small_vals)

            # Test that small values are preserved
            result = backend.to_numpy(arr)
            assert np.allclose(result, small_vals, rtol=1e-6)

            # Test operations preserve precision
            doubled = backend.multiply(arr, backend.array(2.0))
            result = backend.to_numpy(doubled)
            assert np.allclose(result, small_vals * 2, rtol=1e-6)

    def test_large_number_accuracy(self):
        """Test accuracy with very large numbers."""
        large_vals = np.array([1e10, 1e20, 1e30], dtype=np.float32)

        for backend in self.backends:
            arr = backend.from_numpy(large_vals)

            # Test that large values are preserved
            result = backend.to_numpy(arr)
            assert np.allclose(result, large_vals, rtol=1e-6)

            # Test operations preserve scale
            halved = backend.divide(arr, backend.array(2.0))
            result = backend.to_numpy(halved)
            assert np.allclose(result, large_vals / 2, rtol=1e-6)

    def test_numerical_stability_softmax(self):
        """Test numerical stability of softmax-like operations."""
        # Create input that could cause overflow in naive exp
        x_np = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)

        for backend in self.backends:
            x = backend.from_numpy(x_np)

            # Stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
            x_max = backend.max(x)
            x_shifted = backend.subtract(x, x_max)
            exp_x = backend.exp(x_shifted)
            sum_exp = backend.sum(exp_x)
            softmax = backend.divide(exp_x, sum_exp)

            result = backend.to_numpy(softmax)

            # Check that result is valid probabilities
            assert np.all(result >= 0)
            assert np.all(result <= 1)
            assert np.abs(np.sum(result) - 1.0) < 1e-6

    def test_matrix_multiplication_accuracy(self):
        """Test accuracy of matrix multiplication with various sizes."""
        test_cases = [
            # (m, k, n) dimensions for A(m,k) @ B(k,n)
            (2, 3, 4),
            (10, 10, 10),
            (100, 50, 100),
            (17, 23, 31),  # Prime numbers
        ]

        for m, k, n in test_cases:
            # Create test matrices
            A_np = np.random.randn(m, k).astype(np.float32)
            B_np = np.random.randn(k, n).astype(np.float32)

            # Compute reference result
            C_ref = A_np @ B_np

            # Test each backend
            for backend in self.backends:
                A = backend.from_numpy(A_np)
                B = backend.from_numpy(B_np)
                C = backend.matmul(A, B)
                C_result = backend.to_numpy(C)

                # Check accuracy (MLX has slightly different precision characteristics)
                if backend.name == "mps":
                    rtol, atol = 1e-4, 1e-6
                else:
                    rtol, atol = 1e-5, 1e-7
                assert np.allclose(
                    C_result, C_ref, rtol=rtol, atol=atol
                ), f"MatMul accuracy failed for {backend.name} with shape ({m},{k},{n})"

    def test_reduction_accuracy(self):
        """Test accuracy of reduction operations."""
        # Create test array with known sum/mean
        shape = (10, 20, 30)
        arr_np = np.random.randn(*shape).astype(np.float32)

        # Test different axes
        test_axes = [None, 0, 1, 2, (0, 1), (1, 2), (0, 2)]

        for axis in test_axes:
            # Compute reference results
            sum_ref = np.sum(arr_np, axis=axis)
            mean_ref = np.mean(arr_np, axis=axis)
            max_ref = np.max(arr_np, axis=axis)
            min_ref = np.min(arr_np, axis=axis)

            # Test each backend
            for backend in self.backends:
                arr = backend.from_numpy(arr_np)

                # Test sum
                sum_result = backend.to_numpy(backend.sum(arr, axis=axis))
                # Use relaxed tolerance for MPS backend
                rtol = 1e-4 if backend.name == "mps" else 1e-5
                atol = 1e-6 if backend.name == "mps" else 1e-7
                assert np.allclose(
                    sum_result, sum_ref, rtol=rtol, atol=atol
                ), f"Sum accuracy failed for {backend.name} with axis={axis}"

                # Test mean
                mean_result = backend.to_numpy(backend.mean(arr, axis=axis))
                assert np.allclose(
                    mean_result, mean_ref, rtol=rtol, atol=atol
                ), f"Mean accuracy failed for {backend.name} with axis={axis}"

                # Test max
                max_result = backend.to_numpy(backend.max(arr, axis=axis))
                assert np.allclose(
                    max_result, max_ref, rtol=rtol, atol=atol
                ), f"Max accuracy failed for {backend.name} with axis={axis}"

                # Test min
                min_result = backend.to_numpy(backend.min(arr, axis=axis))
                assert np.allclose(
                    min_result, min_ref, rtol=rtol, atol=atol
                ), f"Min accuracy failed for {backend.name} with axis={axis}"

    def test_transcendental_functions_accuracy(self):
        """Test accuracy of exp, log, sqrt."""
        # Test values in safe range
        x_np = np.linspace(0.1, 10.0, 100).astype(np.float32)

        for backend in self.backends:
            x = backend.from_numpy(x_np)

            # Test exp
            exp_result = backend.to_numpy(backend.exp(x))
            exp_ref = np.exp(x_np)
            assert np.allclose(
                exp_result, exp_ref, rtol=1e-5
            ), f"Exp accuracy failed for {backend.name}"

            # Test log
            log_result = backend.to_numpy(backend.log(x))
            log_ref = np.log(x_np)
            assert np.allclose(
                log_result, log_ref, rtol=1e-5
            ), f"Log accuracy failed for {backend.name}"

            # Test sqrt
            sqrt_result = backend.to_numpy(backend.sqrt(x))
            sqrt_ref = np.sqrt(x_np)
            assert np.allclose(
                sqrt_result, sqrt_ref, rtol=1e-5
            ), f"Sqrt accuracy failed for {backend.name}"

    def test_complex_expression_accuracy(self):
        """Test accuracy of complex mathematical expressions."""
        # Create test data
        x_np = np.random.randn(50, 50).astype(np.float32)
        y_np = np.random.randn(50, 50).astype(np.float32)

        # Compute complex expression: (x^2 + y^2) / sqrt(x^2 + y^2 + 1)
        x2_ref = x_np**2
        y2_ref = y_np**2
        sum_ref = x2_ref + y2_ref
        denom_ref = np.sqrt(sum_ref + 1)
        result_ref = sum_ref / denom_ref

        for backend in self.backends:
            x = backend.from_numpy(x_np)
            y = backend.from_numpy(y_np)

            # Compute using backend operations
            x2 = backend.power(x, backend.array(2.0))
            y2 = backend.power(y, backend.array(2.0))
            sum_xy = backend.add(x2, y2)
            one = backend.ones(backend.shape(sum_xy))
            denom = backend.sqrt(backend.add(sum_xy, one))
            result = backend.divide(sum_xy, denom)

            result_np = backend.to_numpy(result)

            assert np.allclose(
                result_np, result_ref, rtol=1e-4, atol=1e-6
            ), f"Complex expression accuracy failed for {backend.name}"

    def test_edge_cases(self):
        """Test edge cases like inf, nan, zero."""
        edge_cases = [
            np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32),
            np.array([np.inf, -np.inf, 0.0, 1.0], dtype=np.float32),
            np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32),
        ]

        for test_array in edge_cases:
            for backend in self.backends:
                arr = backend.from_numpy(test_array)

                # Test that values are preserved
                result = backend.to_numpy(arr)

                # Use special comparison for nan/inf
                assert np.array_equal(
                    result, test_array, equal_nan=True
                ), f"Edge case preservation failed for {backend.name}"

    def test_broadcasting_accuracy(self):
        """Test accuracy of operations with broadcasting."""
        # Different broadcasting scenarios
        test_cases = [
            # (shape1, shape2)
            ((5, 1), (1, 3)),
            ((10, 1, 5), (1, 5, 1)),
            ((1,), (10, 10)),
            ((3, 1, 4), (3, 5, 4)),
        ]

        for shape1, shape2 in test_cases:
            a_np = np.random.randn(*shape1).astype(np.float32)
            b_np = np.random.randn(*shape2).astype(np.float32)

            # Reference results
            add_ref = a_np + b_np
            mul_ref = a_np * b_np

            for backend in self.backends:
                a = backend.from_numpy(a_np)
                b = backend.from_numpy(b_np)

                # Test addition with broadcasting
                add_result = backend.to_numpy(backend.add(a, b))
                assert np.allclose(
                    add_result, add_ref, rtol=1e-5
                ), f"Broadcasting add failed for {backend.name}"

                # Test multiplication with broadcasting
                mul_result = backend.to_numpy(backend.multiply(a, b))
                assert np.allclose(
                    mul_result, mul_ref, rtol=1e-5
                ), f"Broadcasting multiply failed for {backend.name}"


class TestBackendDeterminism:
    """Test that backends produce deterministic results."""

    def test_deterministic_operations(self):
        """Test that operations are deterministic."""
        backends = [get_backend(name) for name in available_backends()]

        # Create test data
        np.random.seed(42)
        a_np = np.random.randn(100, 100).astype(np.float32)
        b_np = np.random.randn(100, 100).astype(np.float32)

        for backend in backends:
            a = backend.from_numpy(a_np)
            b = backend.from_numpy(b_np)

            # Run operation multiple times
            results = []
            for _ in range(5):
                c = backend.matmul(a, b)
                results.append(backend.to_numpy(c))

            # All results should be identical
            for i in range(1, len(results)):
                assert np.array_equal(
                    results[0], results[i]
                ), f"Non-deterministic results for {backend.name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
