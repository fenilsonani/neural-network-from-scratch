"""Comprehensive tests for backend implementations and GPU acceleration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import List, Tuple

import numpy as np
import pytest

from neural_arch.backends import Backend, NumpyBackend, available_backends, get_backend, set_backend

# Test configuration
TOLERANCE = 1e-5  # Numerical tolerance for comparisons
SIZES = [(10, 10), (100, 100), (1000, 1000)]  # Different matrix sizes to test


class TestBackendRegistry:
    """Test backend registration and retrieval."""

    def test_numpy_backend_available(self):
        """Test that numpy backend is always available."""
        assert "numpy" in available_backends()

    def test_get_numpy_backend(self):
        """Test getting numpy backend."""
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)
        assert backend.name == "numpy"
        assert backend.is_available

    def test_set_backend(self):
        """Test setting global backend."""
        original = get_backend()
        set_backend("numpy")
        assert get_backend().name == "numpy"

    def test_invalid_backend(self):
        """Test error on invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid_backend")


class TestBackendOperations:
    """Test all backend operations for correctness."""

    @pytest.fixture
    def backends(self) -> List[Backend]:
        """Get all available backends for testing."""
        return [get_backend(name) for name in available_backends()]

    def test_array_creation(self, backends):
        """Test array creation methods."""
        for backend in backends:
            # Test array from list
            arr = backend.array([1, 2, 3, 4])
            assert backend.shape(arr) == (4,)

            # Test zeros
            zeros = backend.zeros((3, 3))
            assert backend.shape(zeros) == (3, 3)
            assert np.allclose(backend.to_numpy(zeros), np.zeros((3, 3)))

            # Test ones
            ones = backend.ones((2, 4))
            assert backend.shape(ones) == (2, 4)
            assert np.allclose(backend.to_numpy(ones), np.ones((2, 4)))

            # Test full
            full = backend.full((3, 2), 3.14)
            assert backend.shape(full) == (3, 2)
            assert np.allclose(backend.to_numpy(full), np.full((3, 2), 3.14))

    def test_shape_operations(self, backends):
        """Test shape manipulation operations."""
        for backend in backends:
            # Create test array
            arr = backend.array(np.arange(24).reshape(2, 3, 4))

            # Test reshape
            reshaped = backend.reshape(arr, (6, 4))
            assert backend.shape(reshaped) == (6, 4)

            # Test transpose
            transposed = backend.transpose(arr, (2, 0, 1))
            assert backend.shape(transposed) == (4, 2, 3)

            # Test squeeze
            arr_sq = backend.array(np.ones((1, 3, 1, 4)))
            squeezed = backend.squeeze(arr_sq)
            assert backend.shape(squeezed) == (3, 4)

            # Test expand_dims
            expanded = backend.expand_dims(arr, 0)
            assert backend.shape(expanded) == (1, 2, 3, 4)

    def test_math_operations(self, backends):
        """Test mathematical operations."""
        for backend in backends:
            # Create test arrays
            a = backend.array([[1.0, 2.0], [3.0, 4.0]])
            b = backend.array([[5.0, 6.0], [7.0, 8.0]])

            # Test addition
            c = backend.add(a, b)
            expected = np.array([[6.0, 8.0], [10.0, 12.0]])
            assert np.allclose(backend.to_numpy(c), expected)

            # Test subtraction
            c = backend.subtract(a, b)
            expected = np.array([[-4.0, -4.0], [-4.0, -4.0]])
            assert np.allclose(backend.to_numpy(c), expected)

            # Test multiplication
            c = backend.multiply(a, b)
            expected = np.array([[5.0, 12.0], [21.0, 32.0]])
            assert np.allclose(backend.to_numpy(c), expected)

            # Test division
            c = backend.divide(b, a)
            expected = np.array([[5.0, 3.0], [7 / 3, 2.0]])
            assert np.allclose(backend.to_numpy(c), expected)

            # Test power
            c = backend.power(a, backend.array(2.0))
            expected = np.array([[1.0, 4.0], [9.0, 16.0]])
            assert np.allclose(backend.to_numpy(c), expected)

    def test_matrix_operations(self, backends):
        """Test matrix operations."""
        for backend in backends:
            # Create test matrices
            a = backend.array([[1.0, 2.0], [3.0, 4.0]])
            b = backend.array([[5.0, 6.0], [7.0, 8.0]])

            # Test matmul
            c = backend.matmul(a, b)
            expected = np.array([[19.0, 22.0], [43.0, 50.0]])
            assert np.allclose(backend.to_numpy(c), expected, rtol=TOLERANCE)

            # Test batch matmul
            a_batch = backend.array(np.random.randn(10, 3, 4))
            b_batch = backend.array(np.random.randn(10, 4, 5))
            c_batch = backend.matmul(a_batch, b_batch)
            assert backend.shape(c_batch) == (10, 3, 5)

    def test_reduction_operations(self, backends):
        """Test reduction operations."""
        for backend in backends:
            # Create test array
            arr = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

            # Test sum
            assert np.allclose(backend.to_numpy(backend.sum(arr)), 21.0)
            assert np.allclose(
                backend.to_numpy(backend.sum(arr, axis=0)), np.array([5.0, 7.0, 9.0])
            )
            assert np.allclose(backend.to_numpy(backend.sum(arr, axis=1)), np.array([6.0, 15.0]))

            # Test mean
            assert np.allclose(backend.to_numpy(backend.mean(arr)), 3.5)
            assert np.allclose(
                backend.to_numpy(backend.mean(arr, axis=0)), np.array([2.5, 3.5, 4.5])
            )

            # Test max/min
            assert np.allclose(backend.to_numpy(backend.max(arr)), 6.0)
            assert np.allclose(backend.to_numpy(backend.min(arr)), 1.0)

            # Test argmax/argmin
            argmax_result = backend.to_numpy(backend.argmax(arr))
            argmin_result = backend.to_numpy(backend.argmin(arr))
            # Handle both scalar and array results
            if argmax_result.ndim == 0:
                assert argmax_result == 5
                assert argmin_result == 0
            else:
                assert argmax_result[0] == 5
                assert argmin_result[0] == 0

    def test_activation_functions(self, backends):
        """Test activation and math functions."""
        for backend in backends:
            # Create test array
            arr = backend.array([-2.0, -1.0, 0.0, 1.0, 2.0])

            # Test exp
            exp_arr = backend.exp(arr)
            expected = np.exp([-2.0, -1.0, 0.0, 1.0, 2.0])
            assert np.allclose(backend.to_numpy(exp_arr), expected, rtol=TOLERANCE)

            # Test log
            pos_arr = backend.array([0.1, 1.0, 2.0, 10.0])
            log_arr = backend.log(pos_arr)
            expected = np.log([0.1, 1.0, 2.0, 10.0])
            assert np.allclose(backend.to_numpy(log_arr), expected, rtol=TOLERANCE)

            # Test sqrt
            sqrt_arr = backend.sqrt(backend.array([0.0, 1.0, 4.0, 9.0]))
            expected = np.array([0.0, 1.0, 2.0, 3.0])
            assert np.allclose(backend.to_numpy(sqrt_arr), expected, rtol=TOLERANCE)

            # Test abs
            abs_arr = backend.abs(arr)
            expected = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
            assert np.allclose(backend.to_numpy(abs_arr), expected)

            # Test clip
            clipped = backend.clip(arr, -1.0, 1.0)
            expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0])
            assert np.allclose(backend.to_numpy(clipped), expected)

    def test_comparison_operations(self, backends):
        """Test comparison operations."""
        for backend in backends:
            a = backend.array([1.0, 2.0, 3.0])
            b = backend.array([3.0, 2.0, 1.0])

            # Test equal
            eq = backend.equal(a, b)
            expected = np.array([False, True, False])
            assert np.array_equal(backend.to_numpy(eq), expected)

            # Test less
            lt = backend.less(a, b)
            expected = np.array([True, False, False])
            assert np.array_equal(backend.to_numpy(lt), expected)

            # Test greater
            gt = backend.greater(a, b)
            expected = np.array([False, False, True])
            assert np.array_equal(backend.to_numpy(gt), expected)

    def test_array_manipulation(self, backends):
        """Test array manipulation operations."""
        for backend in backends:
            # Create test arrays
            a = backend.array([1, 2, 3])
            b = backend.array([4, 5, 6])
            c = backend.array([7, 8, 9])

            # Test concatenate
            concat = backend.concatenate([a, b, c], axis=0)
            expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            assert np.array_equal(backend.to_numpy(concat), expected)

            # Test stack
            stacked = backend.stack([a, b, c], axis=0)
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            assert np.array_equal(backend.to_numpy(stacked), expected)

            # Test split
            arr = backend.array([1, 2, 3, 4, 5, 6])
            splits = backend.split(arr, 3)
            assert len(splits) == 3
            assert np.array_equal(backend.to_numpy(splits[0]), np.array([1, 2]))

    def test_type_conversion(self, backends):
        """Test type conversion operations."""
        for backend in backends:
            # Create test array
            arr = backend.array([1.5, 2.7, 3.9])

            # Test astype
            int_arr = backend.astype(arr, np.int32)
            # Check that dtype is int32 (backend may return its own dtype object)
            dtype = backend.dtype(int_arr)
            if backend.name == "numpy":
                assert dtype == np.int32
            else:
                # For other backends, check the numpy conversion
                np_int_arr = backend.to_numpy(int_arr)
                assert np_int_arr.dtype == np.int32

            # Test to_numpy and from_numpy
            np_arr = backend.to_numpy(arr)
            assert isinstance(np_arr, np.ndarray)

            arr2 = backend.from_numpy(np_arr)
            assert backend.is_array(arr2)
            assert np.allclose(backend.to_numpy(arr2), np_arr)


class TestBackendConsistency:
    """Test consistency across different backends."""

    def get_backend_pairs(self) -> List[Tuple[str, str]]:
        """Get all pairs of backends to compare."""
        backends = available_backends()
        pairs = []
        if len(backends) > 1:
            # Always compare against numpy as reference
            for backend in backends:
                if backend != "numpy":
                    pairs.append(("numpy", backend))
        return pairs

    def test_matmul_consistency(self):
        """Test matrix multiplication consistency across backends."""
        pairs = self.get_backend_pairs()
        if not pairs:
            pytest.skip("Only one backend available")

        for size in SIZES[:2]:  # Test smaller sizes for consistency
            # Create random matrices
            np.random.seed(42)
            a_np = np.random.randn(*size).astype(np.float32)
            b_np = np.random.randn(*size).astype(np.float32)

            for backend1_name, backend2_name in pairs:
                backend1 = get_backend(backend1_name)
                backend2 = get_backend(backend2_name)

                # Convert to backend arrays
                a1 = backend1.from_numpy(a_np)
                b1 = backend1.from_numpy(b_np)
                a2 = backend2.from_numpy(a_np)
                b2 = backend2.from_numpy(b_np)

                # Compute matmul
                c1 = backend1.matmul(a1, b1)
                c2 = backend2.matmul(a2, b2)

                # Compare results
                c1_np = backend1.to_numpy(c1)
                c2_np = backend2.to_numpy(c2)

                assert np.allclose(
                    c1_np, c2_np, rtol=1e-3, atol=1e-5
                ), f"Matmul mismatch between {backend1_name} and {backend2_name}"

    def test_reduction_consistency(self):
        """Test reduction operations consistency."""
        pairs = self.get_backend_pairs()
        if not pairs:
            pytest.skip("Only one backend available")

        # Create test array
        np.random.seed(42)
        arr_np = np.random.randn(10, 20, 30).astype(np.float32)

        for backend1_name, backend2_name in pairs:
            backend1 = get_backend(backend1_name)
            backend2 = get_backend(backend2_name)

            arr1 = backend1.from_numpy(arr_np)
            arr2 = backend2.from_numpy(arr_np)

            # Test various reductions
            for axis in [None, 0, 1, 2, (0, 1), (1, 2)]:
                # Sum
                sum1 = backend1.to_numpy(backend1.sum(arr1, axis=axis))
                sum2 = backend2.to_numpy(backend2.sum(arr2, axis=axis))
                assert np.allclose(sum1, sum2, rtol=1e-4, atol=1e-6), f"Sum mismatch on axis={axis}"

                # Mean
                mean1 = backend1.to_numpy(backend1.mean(arr1, axis=axis))
                mean2 = backend2.to_numpy(backend2.mean(arr2, axis=axis))
                assert np.allclose(
                    mean1, mean2, rtol=1e-4, atol=1e-6
                ), f"Mean mismatch on axis={axis}"

    def test_activation_consistency(self):
        """Test activation function consistency."""
        pairs = self.get_backend_pairs()
        if not pairs:
            pytest.skip("Only one backend available")

        # Create test array with various values
        np.random.seed(42)
        arr_np = np.random.randn(100).astype(np.float32)

        for backend1_name, backend2_name in pairs:
            backend1 = get_backend(backend1_name)
            backend2 = get_backend(backend2_name)

            arr1 = backend1.from_numpy(arr_np)
            arr2 = backend2.from_numpy(arr_np)

            # Test exp
            exp1 = backend1.to_numpy(backend1.exp(arr1))
            exp2 = backend2.to_numpy(backend2.exp(arr2))
            assert np.allclose(
                exp1, exp2, rtol=1e-4, atol=1e-6
            ), f"Exp mismatch between {backend1_name} and {backend2_name}"

            # Test clip
            clip1 = backend1.to_numpy(backend1.clip(arr1, -1.0, 1.0))
            clip2 = backend2.to_numpy(backend2.clip(arr2, -1.0, 1.0))
            assert np.allclose(
                clip1, clip2, rtol=1e-4, atol=1e-6
            ), f"Clip mismatch between {backend1_name} and {backend2_name}"


class TestBackendPerformance:
    """Test performance characteristics of backends."""

    @pytest.mark.benchmark
    def test_matmul_performance(self):
        """Test that GPU backends are faster for large matrices."""
        backends = available_backends()
        if len(backends) == 1:
            pytest.skip("Only one backend available")

        # Use a reasonable size for testing
        size = (500, 500)

        # Create test matrices
        np.random.seed(42)
        a_np = np.random.randn(*size).astype(np.float32)
        b_np = np.random.randn(*size).astype(np.float32)

        times = {}

        for backend_name in backends:
            backend = get_backend(backend_name)

            # Convert to backend
            a = backend.from_numpy(a_np)
            b = backend.from_numpy(b_np)

            # Warmup
            for _ in range(3):
                _ = backend.matmul(a, b)

            # Time the operation
            import time

            start = time.time()
            for _ in range(10):
                _ = backend.matmul(a, b)
            end = time.time()

            times[backend_name] = (end - start) / 10

        # GPU should be faster than CPU for large matrices
        if "numpy" in times:
            cpu_time = times["numpy"]
            for backend_name, gpu_time in times.items():
                if backend_name != "numpy":
                    # GPU should be at least somewhat faster
                    # (relaxed for CI environments)
                    assert gpu_time < cpu_time * 1.5, f"{backend_name} not faster than CPU"


class TestDeviceOperations:
    """Test device-related operations."""

    def test_device_transfer(self):
        """Test transferring arrays between devices."""
        backends = available_backends()

        for backend_name in backends:
            backend = get_backend(backend_name)

            # Create array
            arr = backend.array([1, 2, 3, 4])

            # Test device operations
            if backend_name == "numpy":
                # CPU only
                arr2 = backend.to_device(arr, "cpu")
                assert backend.device_of(arr2) == "cpu"

                # Should error on GPU
                with pytest.raises(ValueError):
                    backend.to_device(arr, "cuda")
            else:
                # GPU backends
                device = "cuda" if backend_name == "cuda" else "mps"
                arr2 = backend.to_device(arr, device)
                assert device in backend.device_of(arr2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
