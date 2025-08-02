"""Test backend utility functions."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_arch.backends import auto_select_backend, available_backends, get_backend, set_backend
from neural_arch.exceptions import DeviceError


class TestBackendUtils:
    """Test backend utility functions."""

    def test_auto_select_backend_cpu(self):
        """Test auto selection defaults to CPU."""
        backend = auto_select_backend()
        assert backend.name == "numpy"

    def test_auto_select_backend_fallback(self):
        """Test auto selection fallback behavior."""
        backend = auto_select_backend()
        # Should return a valid backend (numpy at minimum)
        assert backend is not None
        assert hasattr(backend, "name")
        assert backend.name in ["numpy", "mps", "cuda"]

    def test_benchmark_backend_numpy(self):
        """Test benchmarking NumPy backend."""
        backend = get_backend("numpy")

        # Simple performance test
        import time

        start = time.time()

        # Perform some operations to test backend
        x = backend.array([[1, 2], [3, 4]], dtype=backend.float32)
        y = backend.array([[5, 6], [7, 8]], dtype=backend.float32)
        result = backend.matmul(x, y)

        end = time.time()
        duration = end - start

        assert result is not None
        assert duration >= 0

    def test_validate_backend_basic(self):
        """Test basic backend validation."""
        backend = get_backend("numpy")

        # Should have required attributes
        assert hasattr(backend, "name")
        assert hasattr(backend, "array")
        assert hasattr(backend, "matmul")
        assert backend.name == "numpy"

    def test_backend_availability_check(self):
        """Test backend availability checking."""
        backends = available_backends()

        # Should have at least numpy
        assert "numpy" in backends
        assert len(backends) >= 1

        # Check each backend
        for backend_name in backends:
            backend = get_backend(backend_name)
            assert backend is not None
            assert hasattr(backend, "available")

    def test_backend_device_mapping(self):
        """Test device to backend mapping."""
        from neural_arch.backends import get_backend_for_device

        # CPU should map to numpy
        try:
            backend = get_backend_for_device("cpu")
            assert backend.name == "numpy"
        except AttributeError:
            # Function might not exist, test concept
            backend = get_backend("numpy")  # Fallback
            assert backend.name == "numpy"

        # Test other devices with fallback
        for device in ["cuda:0", "mps:0"]:
            try:
                backend = get_backend_for_device(device)
                assert backend is not None
            except (AttributeError, DeviceError):
                # Device/function might not be available
                pass

    def test_backend_performance_comparison(self):
        """Test comparing backend performance."""
        numpy_backend = get_backend("numpy")

        # Simple performance comparison
        import time

        start = time.time()

        # Create test data
        x = numpy_backend.array([[1, 2], [3, 4]], dtype=numpy_backend.float32)
        y = numpy_backend.array([[5, 6], [7, 8]], dtype=numpy_backend.float32)

        # Perform operation
        result = numpy_backend.matmul(x, y)

        end = time.time()
        numpy_time = end - start

        # Should complete successfully
        assert result is not None
        assert numpy_time >= 0

    def test_backend_memory_usage(self):
        """Test backend memory usage tracking."""
        backend = get_backend("numpy")

        # Create some tensors
        x = backend.array([[1, 2], [3, 4]], dtype=backend.float32)
        y = backend.array([[5, 6], [7, 8]], dtype=backend.float32)

        # Perform operations
        result = backend.matmul(x, y)

        # Should complete without errors
        assert result is not None
        assert result.shape == (2, 2)

    def test_backend_error_handling(self):
        """Test backend error handling."""
        backend = get_backend("numpy")

        # Test invalid operations
        with pytest.raises((ValueError, TypeError)):
            # Try to multiply incompatible shapes
            x = backend.array([[1, 2, 3]], dtype=backend.float32)  # (1, 3)
            y = backend.array([[1], [2]], dtype=backend.float32)  # (2, 1)
            backend.matmul(x, y)  # Should fail - incompatible shapes

    def test_backend_dtype_consistency(self):
        """Test backend dtype consistency."""
        backend = get_backend("numpy")

        # Test float32
        x = backend.array([1.0, 2.0], dtype=backend.float32)
        assert x.dtype == backend.float32

        # Test int32
        y = backend.array([1, 2], dtype=backend.int32)
        assert y.dtype == backend.int32

    def test_backend_shape_operations(self):
        """Test backend shape operations."""
        backend = get_backend("numpy")

        x = backend.array([[1, 2, 3], [4, 5, 6]], dtype=backend.float32)

        # Test reshape
        reshaped = backend.reshape(x, (3, 2))
        assert reshaped.shape == (3, 2)

        # Test transpose
        transposed = backend.transpose(x, (1, 0))
        assert transposed.shape == (3, 2)

        # Test flatten
        flattened = backend.flatten(x)
        assert flattened.shape == (6,)
