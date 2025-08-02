"""Enhanced comprehensive test coverage for NumPy backend to achieve 95%+ coverage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.backends.numpy_backend import NumpyBackend


class TestNumpyBackendEnhanced:
    """Enhanced comprehensive tests for NumPy backend targeting 95%+ coverage."""

    def setup_method(self):
        """Setup method run before each test."""
        self.backend = NumpyBackend()

    def test_properties(self):
        """Test all backend properties."""
        assert self.backend.name == "numpy"
        assert self.backend.is_available is True
        assert self.backend.supports_gradients is False
        # Test backwards compatibility alias
        assert self.backend.available is True
        assert self.backend.available == self.backend.is_available

    def test_dtype_properties(self):
        """Test dtype properties."""
        assert self.backend.float32 == np.float32
        assert self.backend.float64 == np.float64
        assert self.backend.int32 == np.int32
        assert self.backend.int64 == np.int64
        assert self.backend.bool == np.bool_

    def test_array_creation_methods(self):
        """Test all array creation methods."""
        # Test array creation with different dtypes
        result = self.backend.array([1, 2, 3], dtype=np.float64)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1, 2, 3])

        # Test array creation without dtype
        result = self.backend.array([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

        # Test zeros with explicit dtype
        result = self.backend.zeros((3, 4), dtype=np.int32)
        assert result.shape == (3, 4)
        assert result.dtype == np.int32
        assert np.all(result == 0)

        # Test zeros with default dtype
        result = self.backend.zeros((2, 2))
        assert result.dtype == np.float32
        assert np.all(result == 0)

        # Test ones with explicit dtype
        result = self.backend.ones((2, 3), dtype=np.float64)
        assert result.shape == (2, 3)
        assert result.dtype == np.float64
        assert np.all(result == 1)

        # Test ones with default dtype
        result = self.backend.ones((3, 3))
        assert result.dtype == np.float32
        assert np.all(result == 1)

        # Test full with explicit dtype
        result = self.backend.full((2, 2), 5.0, dtype=np.int32)
        assert result.shape == (2, 2)
        assert result.dtype == np.int32
        assert np.all(result == 5)

        # Test full with default dtype
        result = self.backend.full((2, 2), 3.14)
        assert result.dtype == np.float32
        assert np.all(result == 3.14)

        # Test arange with explicit dtype
        result = self.backend.arange(0, 10, 2, dtype=np.int32)
        expected = np.arange(0, 10, 2, dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

        # Test arange with default dtype
        result = self.backend.arange(0.0, 5.0, 0.5)
        assert result.dtype == np.float32
        expected = np.arange(0.0, 5.0, 0.5, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_random_operations(self):
        """Test random number generation."""
        # Test random_normal with defaults
        result = self.backend.random_normal((3, 3))
        assert result.shape == (3, 3)
        assert result.dtype == np.float32

        # Test random_normal with parameters and dtype
        result = self.backend.random_normal((2, 4), mean=1.0, std=2.0, dtype=np.float64)
        assert result.shape == (2, 4)
        assert result.dtype == np.float32  # Always converted to float32

        # Test random_normal without dtype but with parameters
        result = self.backend.random_normal((2, 2), mean=0.5, std=1.5)
        assert result.dtype == np.float32

        # Test random_uniform with defaults
        result = self.backend.random_uniform((2, 2))
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        assert np.all((result >= 0.0) & (result <= 1.0))

        # Test random_uniform with parameters and dtype
        result = self.backend.random_uniform((3, 2), low=0.1, high=0.9, dtype=np.float64)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32  # Always converted to float32
        assert np.all((result >= 0.1) & (result <= 0.9))

        # Test random_uniform without dtype but with parameters
        result = self.backend.random_uniform((2, 2), low=0.2, high=0.8)
        assert result.dtype == np.float32
        assert np.all((result >= 0.2) & (result <= 0.8))

    def test_shape_operations(self):
        """Test all shape manipulation operations."""
        x = np.random.randn(2, 3, 4).astype(np.float32)

        # Test reshape
        result = self.backend.reshape(x, (6, 4))
        assert result.shape == (6, 4)

        # Test flatten (unique to numpy backend)
        result = self.backend.flatten(x)
        assert result.shape == (24,)
        expected = x.flatten()
        np.testing.assert_array_equal(result, expected)

        # Test transpose with None
        result = self.backend.transpose(x, None)
        expected = np.transpose(x, None)
        np.testing.assert_array_equal(result, expected)

        # Test transpose with axes
        result = self.backend.transpose(x, (2, 0, 1))
        expected = np.transpose(x, (2, 0, 1))
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)

        # Test squeeze
        y = np.random.randn(2, 1, 4).astype(np.float32)
        result = self.backend.squeeze(y, axis=1)
        expected = np.squeeze(y, axis=1)
        np.testing.assert_array_equal(result, expected)

        # Test squeeze with None axis
        result = self.backend.squeeze(y, axis=None)
        expected = np.squeeze(y, axis=None)
        np.testing.assert_array_equal(result, expected)

        # Test expand_dims
        result = self.backend.expand_dims(x, axis=1)
        expected = np.expand_dims(x, axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_mathematical_operations(self):
        """Test all mathematical operations."""
        a = np.array([1, 2, 3, 4], dtype=np.float32)
        b = np.array([5, 6, 7, 8], dtype=np.float32)

        # Test add
        result = self.backend.add(a, b)
        np.testing.assert_array_equal(result, a + b)

        # Test subtract
        result = self.backend.subtract(a, b)
        np.testing.assert_array_equal(result, a - b)

        # Test multiply
        result = self.backend.multiply(a, b)
        np.testing.assert_array_equal(result, a * b)

        # Test mul (alias)
        result = self.backend.mul(a, b)
        np.testing.assert_array_equal(result, a * b)

        # Test divide
        result = self.backend.divide(a, b)
        np.testing.assert_array_equal(result, a / b)

        # Test power
        result = self.backend.power(a, np.array([1, 2, 1, 2]))
        expected = np.power(a, np.array([1, 2, 1, 2]))
        np.testing.assert_array_equal(result, expected)

        # Test matmul
        A = np.random.randn(4, 3).astype(np.float32)
        B = np.random.randn(3, 5).astype(np.float32)
        result = self.backend.matmul(A, B)
        expected = np.matmul(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test dot
        result = self.backend.dot(A, B)
        expected = np.dot(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_reduction_operations(self):
        """Test all reduction operations."""
        x = np.random.randn(4, 5, 6).astype(np.float32)

        # Test sum
        result = self.backend.sum(x, axis=1, keepdims=True)
        expected = np.sum(x, axis=1, keepdims=True)
        np.testing.assert_array_equal(result, expected)

        # Test sum without keepdims
        result = self.backend.sum(x, axis=0, keepdims=False)
        expected = np.sum(x, axis=0, keepdims=False)
        np.testing.assert_array_equal(result, expected)

        # Test sum with no axis
        result = self.backend.sum(x)
        expected = np.sum(x)
        np.testing.assert_array_equal(result, expected)

        # Test mean
        result = self.backend.mean(x, axis=2, keepdims=True)
        expected = np.mean(x, axis=2, keepdims=True)
        np.testing.assert_array_equal(result, expected)

        # Test max
        result = self.backend.max(x, axis=None, keepdims=False)
        expected = np.max(x, axis=None, keepdims=False)
        np.testing.assert_array_equal(result, expected)

        # Test min
        result = self.backend.min(x, axis=1, keepdims=False)
        expected = np.min(x, axis=1, keepdims=False)
        np.testing.assert_array_equal(result, expected)

        # Test argmax
        result = self.backend.argmax(x, axis=2)
        expected = np.argmax(x, axis=2)
        np.testing.assert_array_equal(result, expected)

        # Test argmax with no axis
        result = self.backend.argmax(x)
        expected = np.argmax(x)
        np.testing.assert_array_equal(result, expected)

        # Test argmin
        result = self.backend.argmin(x, axis=0)
        expected = np.argmin(x, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_activation_functions(self):
        """Test activation and math functions."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        # Test exp
        result = self.backend.exp(x)
        expected = np.exp(x)
        np.testing.assert_array_equal(result, expected)

        # Test log
        result = self.backend.log(x)
        expected = np.log(x)
        np.testing.assert_array_equal(result, expected)

        # Test sqrt
        result = self.backend.sqrt(x)
        expected = np.sqrt(x)
        np.testing.assert_array_equal(result, expected)

        # Test abs
        y = np.array([-1, -2, 3, -4], dtype=np.float32)
        result = self.backend.abs(y)
        expected = np.abs(y)
        np.testing.assert_array_equal(result, expected)

        # Test sign
        result = self.backend.sign(y)
        expected = np.sign(y)
        np.testing.assert_array_equal(result, expected)

        # Test clip
        result = self.backend.clip(y, -2.0, 2.0)
        expected = np.clip(y, -2.0, 2.0)
        np.testing.assert_array_equal(result, expected)

        # Test maximum
        a = np.array([1, 5, 3], dtype=np.float32)
        b = np.array([2, 4, 6], dtype=np.float32)
        result = self.backend.maximum(a, b)
        expected = np.maximum(a, b)
        np.testing.assert_array_equal(result, expected)

        # Test minimum
        result = self.backend.minimum(a, b)
        expected = np.minimum(a, b)
        np.testing.assert_array_equal(result, expected)

        # Test tanh
        result = self.backend.tanh(x)
        expected = np.tanh(x)
        np.testing.assert_array_equal(result, expected)

        # Test softmax
        result = self.backend.softmax(x, axis=0)
        # Verify softmax properties
        assert np.isclose(np.sum(result), 1.0)
        assert np.all(result >= 0)

        # Test softmax with different axis
        x_2d = np.random.randn(3, 4).astype(np.float32)
        result = self.backend.softmax(x_2d, axis=-1)
        assert np.allclose(np.sum(result, axis=-1), 1.0)

        # Test softmax with default axis
        result = self.backend.softmax(x_2d)
        assert np.allclose(np.sum(result, axis=-1), 1.0)

    def test_comparison_operations(self):
        """Test comparison operations."""
        a = np.array([1, 2, 3, 4], dtype=np.float32)
        b = np.array([2, 2, 2, 2], dtype=np.float32)

        # Test equal
        result = self.backend.equal(a, b)
        expected = np.equal(a, b)
        np.testing.assert_array_equal(result, expected)

        # Test not_equal
        result = self.backend.not_equal(a, b)
        expected = np.not_equal(a, b)
        np.testing.assert_array_equal(result, expected)

        # Test less
        result = self.backend.less(a, b)
        expected = np.less(a, b)
        np.testing.assert_array_equal(result, expected)

        # Test less_equal
        result = self.backend.less_equal(a, b)
        expected = np.less_equal(a, b)
        np.testing.assert_array_equal(result, expected)

        # Test greater
        result = self.backend.greater(a, b)
        expected = np.greater(a, b)
        np.testing.assert_array_equal(result, expected)

        # Test greater_equal
        result = self.backend.greater_equal(a, b)
        expected = np.greater_equal(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_array_manipulation(self):
        """Test array manipulation operations."""
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(2, 3).astype(np.float32)
        arrays = [a, b]

        # Test concatenate
        result = self.backend.concatenate(arrays, axis=1)
        expected = np.concatenate(arrays, axis=1)
        np.testing.assert_array_equal(result, expected)

        # Test concatenate with default axis
        result = self.backend.concatenate(arrays)
        expected = np.concatenate(arrays, axis=0)
        np.testing.assert_array_equal(result, expected)

        # Test stack
        result = self.backend.stack(arrays, axis=0)
        expected = np.stack(arrays, axis=0)
        np.testing.assert_array_equal(result, expected)

        # Test stack with different axis
        result = self.backend.stack(arrays, axis=1)
        expected = np.stack(arrays, axis=1)
        np.testing.assert_array_equal(result, expected)

        # Test split
        x = np.random.randn(6, 4).astype(np.float32)
        result = self.backend.split(x, [2, 4], axis=0)
        expected = np.split(x, [2, 4], axis=0)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            np.testing.assert_array_equal(r, e)

        # Test split with integer sections
        result = self.backend.split(x, 3, axis=0)
        expected = np.split(x, 3, axis=0)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            np.testing.assert_array_equal(r, e)

    def test_type_conversion(self):
        """Test type conversion operations."""
        x = np.array([1.5, 2.5, 3.5], dtype=np.float32)

        # Test astype
        result = self.backend.astype(x, np.int32)
        expected = x.astype(np.int32)
        np.testing.assert_array_equal(result, expected)

        # Test to_numpy (no-op for numpy backend)
        result = self.backend.to_numpy(x)
        assert result is x

        # Test from_numpy with dtype
        result = self.backend.from_numpy(x, dtype=np.float64)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, x.astype(np.float64))

        # Test from_numpy without dtype
        result = self.backend.from_numpy(x)
        assert result is x

    def test_device_operations(self):
        """Test device operations."""
        x = np.array([1, 2, 3], dtype=np.float32)

        # Test to_device with CPU (should be no-op)
        result = self.backend.to_device(x, "cpu")
        assert result is x

        # Test to_device with invalid device
        with pytest.raises(ValueError, match="NumPy backend only supports CPU"):
            self.backend.to_device(x, "cuda")

        with pytest.raises(ValueError, match="NumPy backend only supports CPU"):
            self.backend.to_device(x, "gpu")

        # Test device_of (always CPU)
        result = self.backend.device_of(x)
        assert result == "cpu"

    def test_utility_functions(self):
        """Test utility functions."""
        x = np.random.randn(3, 4, 5).astype(np.float32)

        # Test is_array
        assert self.backend.is_array(x) is True
        assert self.backend.is_array([1, 2, 3]) is False
        assert self.backend.is_array("not an array") is False
        assert self.backend.is_array(42) is False

        # Test shape
        result = self.backend.shape(x)
        assert result == (3, 4, 5)

        # Test size
        result = self.backend.size(x)
        assert result == 60

        # Test dtype
        result = self.backend.dtype(x)
        assert result == np.float32

    def test_advanced_operations(self):
        """Test advanced operations."""
        # Test einsum
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        result = self.backend.einsum("ij,jk->ik", a, b)
        expected = np.einsum("ij,jk->ik", a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test einsum with more complex operations
        c = np.random.randn(2, 3, 4).astype(np.float32)
        result = self.backend.einsum("ijk->jik", c)
        expected = np.einsum("ijk->jik", c)
        np.testing.assert_array_equal(result, expected)

        # Test where
        condition = np.array([True, False, True, False])
        x = np.array([1, 2, 3, 4], dtype=np.float32)
        y = np.array([5, 6, 7, 8], dtype=np.float32)
        result = self.backend.where(condition, x, y)
        expected = np.where(condition, x, y)
        np.testing.assert_array_equal(result, expected)

        # Test unique without counts
        x = np.array([1, 1, 2, 2, 3, 3], dtype=np.float32)
        result = self.backend.unique(x, return_counts=False)
        expected = np.unique(x, return_counts=False)
        np.testing.assert_array_equal(result, expected)

        # Test unique with counts
        result_vals, result_counts = self.backend.unique(x, return_counts=True)
        expected_vals, expected_counts = np.unique(x, return_counts=True)
        np.testing.assert_array_equal(result_vals, expected_vals)
        np.testing.assert_array_equal(result_counts, expected_counts)

    def test_edge_cases_and_error_conditions(self):
        """Test edge cases and error conditions."""
        # Test empty arrays
        empty = np.array([], dtype=np.float32)
        result = self.backend.sum(empty)
        assert result == 0.0

        # Test single element arrays
        single = np.array([42.0], dtype=np.float32)
        result = self.backend.softmax(single)
        assert np.isclose(result[0], 1.0)

        # Test very small arrays
        tiny = np.array([[1.0]], dtype=np.float32)
        result = self.backend.reshape(tiny, (1,))
        assert result.shape == (1,)

        # Test with different axis types
        x = np.random.randn(2, 3, 4).astype(np.float32)

        # Test squeeze with tuple axis
        y = np.random.randn(2, 1, 1, 4).astype(np.float32)
        result = self.backend.squeeze(y, axis=(1, 2))
        expected = np.squeeze(y, axis=(1, 2))
        np.testing.assert_array_equal(result, expected)

        # Test reduction with tuple axis
        result = self.backend.sum(x, axis=(0, 2), keepdims=True)
        expected = np.sum(x, axis=(0, 2), keepdims=True)
        np.testing.assert_array_equal(result, expected)

    def test_numpy_specific_features(self):
        """Test features specific to numpy backend."""
        # Test the flatten method (not in base backend interface)
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = self.backend.flatten(x)
        expected = np.array([1, 2, 3, 4], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test maximum and minimum element-wise operations
        a = np.array([1, 5, 3], dtype=np.float32)
        b = np.array([2, 4, 6], dtype=np.float32)

        result = self.backend.maximum(a, b)
        expected = np.array([2, 5, 6], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        result = self.backend.minimum(a, b)
        expected = np.array([1, 4, 3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test tanh function
        x = np.array([-1, 0, 1], dtype=np.float32)
        result = self.backend.tanh(x)
        expected = np.tanh(x)
        np.testing.assert_array_equal(result, expected)

    def test_backend_registration(self):
        """Test that numpy backend is properly registered."""
        # This tests the module-level registration
        from neural_arch.backends import get_backend

        # Should be able to get numpy backend
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)
        assert backend.name == "numpy"

    def test_comprehensive_dtype_handling(self):
        """Test comprehensive dtype handling across all operations."""
        # Test with various dtypes
        dtypes = [np.float32, np.float64, np.int32, np.int64]

        for dtype in dtypes:
            # Array creation
            x = self.backend.array([1, 2, 3], dtype=dtype)
            assert x.dtype == dtype

            # Shape operations preserve dtype (where applicable)
            y = self.backend.reshape(x, (3, 1))
            if dtype in [np.float32, np.float64]:
                assert y.dtype == dtype

    def test_broadcasting_compatibility(self):
        """Test operations with broadcasting."""
        # Test broadcasting in mathematical operations
        a = np.array([[1], [2], [3]], dtype=np.float32)  # (3, 1)
        b = np.array([1, 2, 3], dtype=np.float32)  # (3,)

        result = self.backend.add(a, b)
        expected = a + b
        np.testing.assert_array_equal(result, expected)

        result = self.backend.multiply(a, b)
        expected = a * b
        np.testing.assert_array_equal(result, expected)

        # Test broadcasting in comparison operations
        result = self.backend.greater(a, b)
        expected = a > b
        np.testing.assert_array_equal(result, expected)
