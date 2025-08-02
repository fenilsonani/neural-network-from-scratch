"""Comprehensive tests for backends/numpy_backend.py to improve coverage from 79.71% to 95%+.

This file tests NumpyBackend implementation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.backends.backend import get_backend, register_backend
from neural_arch.backends.numpy_backend import NumpyBackend


class TestNumpyBackendComprehensive:
    """Comprehensive tests for NumpyBackend."""

    def setup_method(self):
        """Set up test method."""
        self.backend = NumpyBackend()

    def test_backend_properties(self):
        """Test backend properties."""
        assert self.backend.name == "numpy"
        assert self.backend.is_available == True
        assert self.backend.supports_gradients == False

    def test_backend_device_methods(self):
        """Test device-related methods."""
        # to_device should be no-op for numpy
        x = np.array([1, 2, 3])
        result = self.backend.to_device(x, "cpu")
        assert result is x  # Should return same object

        # device_of should always return 'cpu'
        assert self.backend.device_of(x) == "cpu"

    def test_array_creation(self):
        """Test array creation method."""
        # Test with different dtypes
        x = self.backend.array([[1, 2], [3, 4]], dtype=np.float32)
        assert x.dtype == np.float32
        np.testing.assert_array_equal(x, [[1, 2], [3, 4]])

        # Test with default dtype (NumpyBackend doesn't have a default dtype)
        x_default = self.backend.array([1, 2, 3])
        assert x_default.dtype == np.int64  # Default numpy behavior

    def test_arange(self):
        """Test arange method."""
        # Basic usage
        x = self.backend.arange(0, 5)
        np.testing.assert_array_equal(x, [0, 1, 2, 3, 4])
        assert x.dtype == np.float32

        # With step
        x_step = self.backend.arange(0, 10, 2)
        np.testing.assert_array_equal(x_step, [0, 2, 4, 6, 8])

        # With custom dtype
        x_int = self.backend.arange(0, 5, dtype=np.int32)
        assert x_int.dtype == np.int32

    def test_random_normal(self):
        """Test random_normal method."""
        # Basic usage
        x = self.backend.random_normal((2, 3))
        assert x.shape == (2, 3)
        assert x.dtype == np.float32

        # With mean and std
        x_custom = self.backend.random_normal((100,), mean=5.0, std=2.0)
        assert x_custom.shape == (100,)
        # Check approximate mean and std (with some tolerance)
        assert abs(np.mean(x_custom) - 5.0) < 0.5
        assert abs(np.std(x_custom) - 2.0) < 0.5

        # With custom dtype
        x_double = self.backend.random_normal((2, 2), dtype=np.float64)
        assert x_double.dtype == np.float32  # Always returns float32

    def test_random_uniform(self):
        """Test random_uniform method."""
        # Basic usage
        x = self.backend.random_uniform((3, 4))
        assert x.shape == (3, 4)
        assert x.dtype == np.float32
        assert np.all(x >= 0) and np.all(x <= 1)

        # With custom range
        x_range = self.backend.random_uniform((100,), low=-5, high=5)
        assert np.all(x_range >= -5) and np.all(x_range <= 5)

        # With custom dtype
        x_double = self.backend.random_uniform((2, 2), dtype=np.float64)
        assert x_double.dtype == np.float32  # Always returns float32

    def test_mathematical_operations(self):
        """Test mathematical operations."""
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)

        # Test log
        log_x = self.backend.log(x)
        np.testing.assert_array_almost_equal(log_x, np.log(x))

        # Test clip
        clipped = self.backend.clip(x, 1.5, 3.5)
        expected = np.array([[1.5, 2], [3, 3.5]], dtype=np.float32)
        np.testing.assert_array_equal(clipped, expected)

        # Test power (not pow)
        y = np.array([[2, 2], [2, 2]], dtype=np.float32)
        powered = self.backend.power(x, y)
        np.testing.assert_array_equal(powered, x**y)

        # Test abs
        x_neg = np.array([[-1, 2], [-3, 4]], dtype=np.float32)
        abs_x = self.backend.abs(x_neg)
        np.testing.assert_array_equal(abs_x, np.abs(x_neg))

    def test_activation_functions(self):
        """Test activation function implementations."""
        x = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)

        # NumpyBackend doesn't have activation functions
        # Test exp instead
        exp_x = self.backend.exp(x)
        np.testing.assert_array_almost_equal(exp_x, np.exp(x))

        # Test sqrt
        x_pos = np.array([[1, 4, 9, 16]], dtype=np.float32)
        sqrt_x = self.backend.sqrt(x_pos)
        np.testing.assert_array_almost_equal(sqrt_x, np.sqrt(x_pos))

        # Test sign
        sign_x = self.backend.sign(x)
        np.testing.assert_array_equal(sign_x, np.sign(x))

    def test_reduction_operations(self):
        """Test reduction operations."""
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        # Test argmax
        argmax_axis1 = self.backend.argmax(x, axis=1)
        np.testing.assert_array_equal(argmax_axis1, [2, 2])

        argmax_axis0 = self.backend.argmax(x, axis=0)
        np.testing.assert_array_equal(argmax_axis0, [1, 1, 1])

        # Test argmin
        argmin_axis1 = self.backend.argmin(x, axis=1)
        np.testing.assert_array_equal(argmin_axis1, [0, 0])

    def test_comparison_operations(self):
        """Test comparison operations."""
        x = np.array([1, 2, 3], dtype=np.float32)
        y = np.array([2, 2, 2], dtype=np.float32)

        # Test equal
        eq = self.backend.equal(x, y)
        np.testing.assert_array_equal(eq, [False, True, False])

        # Test not_equal
        neq = self.backend.not_equal(x, y)
        np.testing.assert_array_equal(neq, [True, False, True])

        # Test less
        lt = self.backend.less(x, y)
        np.testing.assert_array_equal(lt, [True, False, False])

        # Test less_equal
        le = self.backend.less_equal(x, y)
        np.testing.assert_array_equal(le, [True, True, False])

        # Test greater
        gt = self.backend.greater(x, y)
        np.testing.assert_array_equal(gt, [False, False, True])

        # Test greater_equal
        ge = self.backend.greater_equal(x, y)
        np.testing.assert_array_equal(ge, [False, True, True])

    def test_special_operations(self):
        """Test special operations."""
        # Test where
        condition = np.array([True, False, True])
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        result = self.backend.where(condition, x, y)
        np.testing.assert_array_equal(result, [1, 5, 3])

        # Test concatenate
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        concat_axis0 = self.backend.concatenate([a, b], axis=0)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(concat_axis0, expected)

        # Test stack
        stack_axis0 = self.backend.stack([a, b], axis=0)
        assert stack_axis0.shape == (2, 2, 2)

        # Test split
        x = np.array([1, 2, 3, 4, 5, 6])
        splits = self.backend.split(x, 3)
        assert len(splits) == 3
        for split in splits:
            assert len(split) == 2

    def test_backend_registration(self):
        """Test backend registration and retrieval."""
        # NumpyBackend should auto-register on import
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)
        assert backend.name == "numpy"
