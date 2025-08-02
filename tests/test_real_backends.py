"""Real comprehensive tests for backends."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_arch.backends import available_backends, current_backend, get_backend, set_backend
from neural_arch.backends.numpy_backend import NumpyBackend
from neural_arch.backends.utils import auto_select_backend
from neural_arch.exceptions import DeviceError


class TestRealBackends:
    """Real tests for backend system without simulation."""

    def test_available_backends(self):
        """Test available backends functionality."""
        backends = available_backends()

        # Should return list of strings
        assert isinstance(backends, list)
        assert len(backends) > 0

        # Should always include numpy
        assert "numpy" in backends

        # All entries should be strings
        for backend_name in backends:
            assert isinstance(backend_name, str)

    def test_get_backend_numpy(self):
        """Test getting NumPy backend."""
        backend = get_backend("numpy")

        # Should return NumpyBackend instance
        assert isinstance(backend, NumpyBackend)
        assert backend.name == "numpy"
        assert backend.available is True

        # Should have required methods
        required_methods = ["array", "matmul", "add", "sub", "mul", "div"]
        for method in required_methods:
            assert hasattr(backend, method)
            assert callable(getattr(backend, method))

    def test_numpy_backend_array_creation(self):
        """Test NumPy backend array creation."""
        backend = get_backend("numpy")

        # Create array from list
        arr = backend.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1, 2, 3])

        # Create array with dtype
        arr = backend.array([1.0, 2.0, 3.0], dtype=backend.float32)
        assert arr.dtype == np.float32

        # Create 2D array
        arr = backend.array([[1, 2], [3, 4]])
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4]])

    def test_numpy_backend_arithmetic(self):
        """Test NumPy backend arithmetic operations."""
        backend = get_backend("numpy")

        a = backend.array([[1, 2], [3, 4]], dtype=backend.float32)
        b = backend.array([[5, 6], [7, 8]], dtype=backend.float32)

        # Addition
        result = backend.add(a, b)
        expected = np.array([[6, 8], [10, 12]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Subtraction
        result = backend.sub(a, b)
        expected = np.array([[-4, -4], [-4, -4]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Multiplication
        result = backend.mul(a, b)
        expected = np.array([[5, 12], [21, 32]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Division
        result = backend.div(a, b)
        expected = np.array([[1 / 5, 2 / 6], [3 / 7, 4 / 8]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_numpy_backend_matmul(self):
        """Test NumPy backend matrix multiplication."""
        backend = get_backend("numpy")

        a = backend.array([[1, 2], [3, 4]], dtype=backend.float32)
        b = backend.array([[5, 6], [7, 8]], dtype=backend.float32)

        result = backend.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_numpy_backend_dtypes(self):
        """Test NumPy backend data types."""
        backend = get_backend("numpy")

        # Check dtype attributes
        assert hasattr(backend, "float32")
        assert hasattr(backend, "float64")
        assert hasattr(backend, "int32")
        assert hasattr(backend, "int64")

        # Check dtype values
        assert backend.float32 == np.float32
        assert backend.float64 == np.float64
        assert backend.int32 == np.int32
        assert backend.int64 == np.int64

    def test_numpy_backend_shape_operations(self):
        """Test NumPy backend shape operations."""
        backend = get_backend("numpy")

        arr = backend.array([[1, 2, 3], [4, 5, 6]])

        # Reshape
        reshaped = backend.reshape(arr, (3, 2))
        assert reshaped.shape == (3, 2)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(reshaped, expected)

        # Transpose
        transposed = backend.transpose(arr, (1, 0))
        assert transposed.shape == (3, 2)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(transposed, expected)

    def test_numpy_backend_activation_functions(self):
        """Test NumPy backend activation functions."""
        backend = get_backend("numpy")

        x = backend.array([-2, -1, 0, 1, 2], dtype=backend.float32)

        # ReLU
        result = backend.relu(x)
        expected = np.array([0, 0, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Sigmoid
        result = backend.sigmoid(x)
        assert result.shape == x.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)

        # Tanh
        result = backend.tanh(x)
        assert result.shape == x.shape
        assert np.all(result >= -1)
        assert np.all(result <= 1)

    def test_numpy_backend_reduction_operations(self):
        """Test NumPy backend reduction operations."""
        backend = get_backend("numpy")

        arr = backend.array([[1, 2, 3], [4, 5, 6]], dtype=backend.float32)

        # Sum
        result = backend.sum(arr)
        assert result == 21

        # Sum along axis
        result = backend.sum(arr, axis=0)
        expected = np.array([5, 7, 9], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        result = backend.sum(arr, axis=1)
        expected = np.array([6, 15], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_numpy_backend_indexing(self):
        """Test NumPy backend indexing operations."""
        backend = get_backend("numpy")

        arr = backend.array([[1, 2, 3], [4, 5, 6]])

        # Argmax
        result = backend.argmax(arr)
        assert result == 5  # Index of maximum value

        # Argmax along axis
        result = backend.argmax(arr, axis=0)
        expected = np.array([1, 1, 1])  # Indices of max in each column
        np.testing.assert_array_equal(result, expected)

        # Argmin
        result = backend.argmin(arr)
        assert result == 0  # Index of minimum value

    def test_set_backend(self):
        """Test setting default backend."""
        # Get current backend
        original_backend = current_backend()

        # Set to numpy explicitly
        set_backend("numpy")
        current = current_backend()
        assert current.name == "numpy"

        # Restore original (if different)
        if original_backend.name != "numpy":
            set_backend(original_backend.name)

    def test_backend_error_handling(self):
        """Test backend error handling."""
        # Invalid backend name
        with pytest.raises((ValueError, KeyError, AttributeError)):
            get_backend("invalid_backend")

    def test_cuda_backend_fallback(self):
        """Test CUDA backend fallback behavior."""
        try:
            backend = get_backend("cuda")

            # If CUDA is available, should be CUDA backend
            # If not available, should fall back to numpy or raise error
            if backend.available:
                assert backend.name == "cuda"
            else:
                # Should either raise error or fallback
                assert backend.name in ["cuda", "numpy"]

        except (ImportError, RuntimeError, KeyError):
            # CUDA backend might not be available
            pass

    def test_mps_backend_fallback(self):
        """Test MPS backend fallback behavior."""
        try:
            backend = get_backend("mps")

            # If MPS is available, should be MPS backend
            # If not available, should fall back or raise error
            if backend.available:
                assert backend.name == "mps"
            else:
                assert backend.name in ["mps", "numpy"]

        except (ImportError, RuntimeError, KeyError):
            # MPS backend might not be available
            pass

    def test_auto_select_backend(self):
        """Test automatic backend selection."""
        backend = auto_select_backend()

        # Should return a valid backend
        assert backend is not None
        assert hasattr(backend, "name")
        assert hasattr(backend, "available")
        assert backend.available is True

        # Should be one of the known backends
        assert backend.name in ["numpy", "cuda", "mps"]

    def test_backend_consistency(self):
        """Test that operations produce consistent results."""
        backend = get_backend("numpy")

        # Create test data
        a = backend.array([[1, 2], [3, 4]], dtype=backend.float32)
        b = backend.array([[2, 0], [1, 2]], dtype=backend.float32)

        # Test associativity: (a + b) + c = a + (b + c)
        c = backend.array([[1, 1], [1, 1]], dtype=backend.float32)

        left = backend.add(backend.add(a, b), c)
        right = backend.add(a, backend.add(b, c))

        np.testing.assert_array_almost_equal(left, right)

    def test_backend_memory_management(self):
        """Test backend memory management."""
        backend = get_backend("numpy")

        # Create large array
        large_arr = backend.array(np.random.randn(1000, 1000))

        # Perform operations
        result = backend.add(large_arr, 1.0)

        # Should complete without memory errors
        assert result.shape == (1000, 1000)

        # Clean up (Python garbage collection will handle this)
        del large_arr, result

    def test_backend_numerical_stability(self):
        """Test backend numerical stability."""
        backend = get_backend("numpy")

        # Test with very small numbers
        small = backend.array([1e-10, 1e-20, 1e-30], dtype=backend.float32)
        result = backend.add(small, small)

        # Should handle small numbers gracefully
        assert np.all(np.isfinite(result))

        # Test with very large numbers
        large = backend.array([1e10, 1e20], dtype=backend.float32)
        result = backend.mul(large, 2.0)

        # Should handle large numbers (may overflow to inf, which is OK)
        assert result.shape == large.shape

    def test_backend_broadcasting(self):
        """Test backend broadcasting capabilities."""
        backend = get_backend("numpy")

        # Test broadcasting in addition
        a = backend.array([[1, 2, 3]], dtype=backend.float32)  # (1, 3)
        b = backend.array([[1], [2]], dtype=backend.float32)  # (2, 1)

        result = backend.add(a, b)
        assert result.shape == (2, 3)

        expected = np.array([[2, 3, 4], [3, 4, 5]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_backend_type_checking(self):
        """Test backend type checking and validation."""
        backend = get_backend("numpy")

        # Should handle different input types
        list_input = [1, 2, 3]
        arr_from_list = backend.array(list_input)
        assert isinstance(arr_from_list, np.ndarray)

        # Should handle numpy arrays
        np_input = np.array([4, 5, 6])
        arr_from_np = backend.array(np_input)
        assert isinstance(arr_from_np, np.ndarray)

        # Should handle scalars
        scalar_input = 5
        arr_from_scalar = backend.array(scalar_input)
        assert isinstance(arr_from_scalar, np.ndarray)

    def test_backend_is_array_method(self):
        """Test backend array type checking."""
        backend = get_backend("numpy")

        # NumPy array should be recognized
        np_arr = np.array([1, 2, 3])
        assert backend.is_array(np_arr)

        # List should not be recognized as backend array
        list_data = [1, 2, 3]
        assert not backend.is_array(list_data)

        # Scalar should not be recognized as array
        scalar = 5
        assert not backend.is_array(scalar)

    def test_backend_to_numpy_method(self):
        """Test backend to_numpy conversion."""
        backend = get_backend("numpy")

        # Create backend array
        backend_arr = backend.array([1, 2, 3])

        # Convert to numpy
        np_arr = backend.to_numpy(backend_arr)

        # Should be numpy array
        assert isinstance(np_arr, np.ndarray)
        np.testing.assert_array_equal(np_arr, [1, 2, 3])
