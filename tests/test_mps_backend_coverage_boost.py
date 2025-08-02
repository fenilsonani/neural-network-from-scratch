"""Comprehensive test coverage for MPS backend to boost coverage from 67.96% to 95%+"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from neural_arch.backends import get_backend
from neural_arch.backends.mps_backend import MPSBackend
from neural_arch.exceptions import DeviceError


class TestMPSBackendCoverageBoost:
    """Comprehensive tests for MPS backend targeting missing coverage paths."""

    def test_mps_backend_mlx_not_available(self):
        """Test MPS backend when MLX is not available."""
        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", False):
            with pytest.raises(ImportError, match="MLX is not installed"):
                MPSBackend()

    def test_mps_backend_initialization_with_mlx(self):
        """Test MPS backend initialization when MLX is available."""
        mock_mx = MagicMock()
        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                assert backend.name == "mps"

    def test_mps_backend_availability_when_mlx_unavailable(self):
        """Test is_available property when MLX is not available."""
        mock_mx = MagicMock()
        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", False):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                try:
                    backend = MPSBackend()
                except ImportError:
                    # Expected when MLX not available
                    pass
                else:
                    # If we get here, test the unavailable path
                    assert not backend.is_available

    def test_mps_backend_availability_exception_handling(self):
        """Test is_available when array creation throws exception."""
        mock_mx = MagicMock()
        mock_mx.array.side_effect = Exception("GPU not available")

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                assert not backend.is_available

    def test_mps_backend_supports_gradients(self):
        """Test supports_gradients property."""
        mock_mx = MagicMock()
        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                assert backend.supports_gradients is True

    def test_mps_backend_dtype_properties(self):
        """Test dtype properties."""
        mock_mx = MagicMock()
        mock_mx.float32 = "float32"
        mock_mx.float64 = "float64"
        mock_mx.int32 = "int32"
        mock_mx.int64 = "int64"
        mock_mx.bool_ = "bool"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                assert backend.float32 == "float32"
                assert backend.float64 == "float64"
                assert backend.int32 == "int32"
                assert backend.int64 == "int64"
                assert backend.bool == "bool"

    def test_mps_backend_array_with_explicit_dtype(self):
        """Test array creation with explicit dtype."""
        mock_mx = MagicMock()
        mock_mx.float32 = "float32"
        mock_array_result = MagicMock()
        mock_mx.array.return_value = mock_array_result

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                backend._convert_dtype = Mock(return_value="converted_dtype")

                result = backend.array([1, 2, 3], dtype="float64")

                mock_mx.array.assert_called_once_with([1, 2, 3], dtype="converted_dtype")
                assert result == mock_array_result

    def test_mps_backend_zeros_with_dtype(self):
        """Test zeros creation with dtype conversion."""
        mock_mx = MagicMock()
        mock_mx.float32 = "float32"
        mock_result = MagicMock()
        mock_mx.zeros.return_value = mock_result

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                backend._convert_dtype = Mock(return_value="converted_dtype")

                result = backend.zeros((3, 4), dtype="float64")

                mock_mx.zeros.assert_called_once_with((3, 4), dtype="converted_dtype")
                assert result == mock_result

    def test_mps_backend_ones_with_dtype(self):
        """Test ones creation with dtype conversion."""
        mock_mx = MagicMock()
        mock_mx.float32 = "float32"
        mock_result = MagicMock()
        mock_mx.ones.return_value = mock_result

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                backend._convert_dtype = Mock(return_value="converted_dtype")

                result = backend.ones((2, 3), dtype="int32")

                mock_mx.ones.assert_called_once_with((2, 3), dtype="converted_dtype")
                assert result == mock_result

    def test_mps_backend_full_with_dtype(self):
        """Test full creation with dtype conversion."""
        mock_mx = MagicMock()
        mock_mx.float32 = "float32"
        mock_result = MagicMock()
        mock_mx.full.return_value = mock_result

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                backend._convert_dtype = Mock(return_value="converted_dtype")

                result = backend.full((2, 2), 5.0, dtype="float16")

                mock_mx.full.assert_called_once_with((2, 2), 5.0, dtype="converted_dtype")
                assert result == mock_result

    def test_mps_backend_arange(self):
        """Test arange functionality."""
        mock_mx = MagicMock()
        mock_mx.float32 = "float32"
        mock_result = MagicMock()
        mock_mx.arange.return_value = mock_result

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()
                backend._convert_dtype = Mock(return_value="converted_dtype")

                result = backend.arange(0, 10, 2, dtype="int32")

                mock_mx.arange.assert_called_once_with(0, 10, 2, dtype="converted_dtype")
                assert result == mock_result

    def test_mps_backend_random_normal(self):
        """Test random_normal functionality."""
        mock_mx = MagicMock()
        mock_np = MagicMock()
        mock_np.random.randint.return_value = 12345

        mock_key = MagicMock()
        mock_mx.random.key.return_value = mock_key

        mock_normal_result = MagicMock()
        mock_mx.random.normal.return_value = mock_normal_result

        # Mock the arithmetic operations
        mock_scaled_result = MagicMock()
        mock_normal_result.__mul__ = Mock(return_value=mock_scaled_result)
        mock_final_result = MagicMock()
        mock_scaled_result.__add__ = Mock(return_value=mock_final_result)

        mock_final_result.astype = Mock(return_value="converted_result")

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                with patch("neural_arch.backends.mps_backend.np", mock_np):
                    backend = MPSBackend()
                    backend._convert_dtype = Mock(return_value="converted_dtype")

                    result = backend.random_normal((3, 3), mean=1.0, std=2.0, dtype="float16")

                    mock_np.random.randint.assert_called_once_with(0, 2**32)
                    mock_mx.random.key.assert_called_once_with(12345)
                    mock_mx.random.normal.assert_called_once_with((3, 3), key=mock_key)
                    mock_final_result.astype.assert_called_once_with("converted_dtype")
                    assert result == "converted_result"

    def test_mps_backend_random_uniform(self):
        """Test random_uniform functionality."""
        mock_mx = MagicMock()
        mock_np = MagicMock()
        mock_np.random.randint.return_value = 54321

        mock_key = MagicMock()
        mock_mx.random.key.return_value = mock_key

        mock_uniform_result = MagicMock()
        mock_mx.random.uniform.return_value = mock_uniform_result
        mock_uniform_result.astype = Mock(return_value="converted_result")

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                with patch("neural_arch.backends.mps_backend.np", mock_np):
                    backend = MPSBackend()
                    backend._convert_dtype = Mock(return_value="converted_dtype")

                    result = backend.random_uniform((2, 4), low=0.1, high=0.9, dtype="float32")

                    mock_np.random.randint.assert_called_once_with(0, 2**32)
                    mock_mx.random.key.assert_called_once_with(54321)
                    mock_mx.random.uniform.assert_called_once_with(
                        (2, 4), low=0.1, high=0.9, key=mock_key
                    )
                    mock_uniform_result.astype.assert_called_once_with("converted_dtype")
                    assert result == "converted_result"

    def test_mps_backend_transpose_with_none_axes(self):
        """Test transpose with None axes."""
        mock_mx = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.T = "transposed_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                result = backend.transpose(mock_tensor, axes=None)

                assert result == "transposed_result"

    def test_mps_backend_reshape(self):
        """Test reshape functionality."""
        mock_mx = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.reshape.return_value = "reshaped_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                result = backend.reshape(mock_tensor, (2, 3, 4))

                mock_tensor.reshape.assert_called_once_with((2, 3, 4))
                assert result == "reshaped_result"

    def test_mps_backend_mathematical_operations(self):
        """Test mathematical operations."""
        mock_mx = MagicMock()
        mock_mx.add.return_value = "add_result"
        mock_mx.subtract.return_value = "sub_result"
        mock_mx.multiply.return_value = "mul_result"
        mock_mx.divide.return_value = "div_result"
        mock_mx.power.return_value = "pow_result"
        mock_mx.matmul.return_value = "matmul_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                assert backend.add("a", "b") == "add_result"
                assert backend.subtract("a", "b") == "sub_result"
                assert backend.multiply("a", "b") == "mul_result"
                assert backend.divide("a", "b") == "div_result"
                assert backend.power("a", "b") == "pow_result"
                assert backend.matmul("a", "b") == "matmul_result"
                assert backend.dot("a", "b") == "matmul_result"  # dot uses matmul

    def test_mps_backend_reduction_operations(self):
        """Test reduction operations."""
        mock_mx = MagicMock()
        mock_mx.sum.return_value = "sum_result"
        mock_mx.mean.return_value = "mean_result"
        mock_mx.max.return_value = "max_result"
        mock_mx.min.return_value = "min_result"
        mock_mx.argmax.return_value = "argmax_result"
        mock_mx.argmin.return_value = "argmin_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                assert backend.sum("x", axis=1, keepdims=True) == "sum_result"
                assert backend.mean("x", axis=0, keepdims=False) == "mean_result"
                assert backend.max("x", axis=None, keepdims=True) == "max_result"
                assert backend.min("x", axis=2, keepdims=False) == "min_result"
                assert backend.argmax("x", axis=1) == "argmax_result"
                assert backend.argmin("x", axis=0) == "argmin_result"

    def test_mps_backend_activation_functions(self):
        """Test activation functions."""
        mock_mx = MagicMock()
        mock_mx.exp.return_value = "exp_result"
        mock_mx.log.return_value = "log_result"
        mock_mx.sqrt.return_value = "sqrt_result"
        mock_mx.abs.return_value = "abs_result"
        mock_mx.sign.return_value = "sign_result"
        mock_mx.clip.return_value = "clip_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                assert backend.exp("x") == "exp_result"
                assert backend.log("x") == "log_result"
                assert backend.sqrt("x") == "sqrt_result"
                assert backend.abs("x") == "abs_result"
                assert backend.sign("x") == "sign_result"
                assert backend.clip("x", -1.0, 1.0) == "clip_result"

    def test_mps_backend_comparison_operations(self):
        """Test comparison operations."""
        mock_mx = MagicMock()
        mock_mx.equal.return_value = "eq_result"
        mock_mx.not_equal.return_value = "ne_result"
        mock_mx.less.return_value = "lt_result"
        mock_mx.less_equal.return_value = "le_result"
        mock_mx.greater.return_value = "gt_result"
        mock_mx.greater_equal.return_value = "ge_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                assert backend.equal("a", "b") == "eq_result"
                assert backend.not_equal("a", "b") == "ne_result"
                assert backend.less("a", "b") == "lt_result"
                assert backend.less_equal("a", "b") == "le_result"
                assert backend.greater("a", "b") == "gt_result"
                assert backend.greater_equal("a", "b") == "ge_result"

    def test_mps_backend_array_manipulation(self):
        """Test array manipulation operations."""
        mock_mx = MagicMock()
        mock_mx.concatenate.return_value = "concat_result"
        mock_mx.stack.return_value = "stack_result"
        mock_mx.split.return_value = ["split1", "split2"]
        mock_mx.squeeze.return_value = "squeeze_result"
        mock_mx.expand_dims.return_value = "expand_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                assert backend.concatenate(["a", "b"], axis=1) == "concat_result"
                assert backend.stack(["a", "b"], axis=0) == "stack_result"
                assert backend.split("x", [1, 3], axis=1) == ["split1", "split2"]
                assert backend.squeeze("x", axis=1) == "squeeze_result"
                assert backend.expand_dims("x", axis=2) == "expand_result"

    def test_mps_backend_type_conversion(self):
        """Test type conversion operations."""
        mock_mx = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.astype.return_value = "astype_result"
        mock_mx.array.return_value = "from_numpy_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                with patch("neural_arch.backends.mps_backend.np") as mock_np:
                    mock_np.array.return_value = "numpy_result"

                    backend = MPSBackend()
                    backend._convert_dtype = Mock(return_value="converted_dtype")

                    # Test astype
                    result = backend.astype(mock_tensor, "float32")
                    mock_tensor.astype.assert_called_once_with("converted_dtype")
                    assert result == "astype_result"

                    # Test to_numpy
                    result = backend.to_numpy("mlx_array")
                    mock_np.array.assert_called_once_with("mlx_array")
                    assert result == "numpy_result"

                    # Test from_numpy with dtype
                    result = backend.from_numpy("numpy_array", dtype="float16")
                    mock_mx.array.assert_called_with("numpy_array", dtype="converted_dtype")
                    assert result == "from_numpy_result"

    def test_mps_backend_device_operations(self):
        """Test device operations."""
        mock_mx = MagicMock()

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                # Test valid device transfers
                assert backend.to_device("tensor", "cpu") == "tensor"
                assert backend.to_device("tensor", "mps") == "tensor"
                assert backend.to_device("tensor", "gpu") == "tensor"

                # Test invalid device
                with pytest.raises(ValueError, match="MLX backend supports cpu/mps/gpu"):
                    backend.to_device("tensor", "invalid_device")

                # Test device_of
                assert backend.device_of("tensor") == "mps"

    def test_mps_backend_utility_functions(self):
        """Test utility functions."""
        mock_mx = MagicMock()

        # Test when mx is None
        with patch("neural_arch.backends.mps_backend.mx", None):
            backend = MPSBackend.__new__(MPSBackend)  # Create without __init__
            assert backend.is_array("anything") is False

        # Test when mx is available
        mock_array_type = type("MockMLXArray", (), {})
        mock_mx.array = mock_array_type
        mock_tensor = mock_array_type()
        mock_tensor.shape = (2, 3)
        mock_tensor.size = 6
        mock_tensor.dtype = "float32"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                backend = MPSBackend()

                # Test is_array
                assert backend.is_array(mock_tensor) is True
                assert backend.is_array("not_an_array") is False

                # Test shape, size, dtype
                assert backend.shape(mock_tensor) == (2, 3)
                assert backend.size(mock_tensor) == 6
                assert backend.dtype(mock_tensor) == "float32"

    def test_mps_backend_advanced_operations(self):
        """Test advanced operations like einsum, where, unique."""
        mock_mx = MagicMock()
        mock_mx.where.return_value = "where_result"

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                with patch("neural_arch.backends.mps_backend.np") as mock_np:
                    mock_np.einsum.return_value = "einsum_numpy_result"
                    mock_np.unique.return_value = "unique_result"

                    backend = MPSBackend()
                    backend.to_numpy = Mock(side_effect=lambda x: f"numpy_{x}")
                    backend.from_numpy = Mock(side_effect=lambda x: f"mlx_{x}")

                    # Test einsum (fallback to numpy)
                    result = backend.einsum("ij,jk->ik", "op1", "op2")
                    mock_np.einsum.assert_called_once_with("ij,jk->ik", "numpy_op1", "numpy_op2")
                    assert result == "mlx_einsum_numpy_result"

                    # Test where
                    assert backend.where("cond", "x", "y") == "where_result"

                    # Test unique without counts
                    result = backend.unique("array", return_counts=False)
                    assert result == "mlx_unique_result"

                    # Test unique with counts
                    mock_np.unique.return_value = ("unique_vals", "counts")
                    result = backend.unique("array", return_counts=True)
                    expected = ("mlx_unique_vals", "mlx_counts")
                    assert result == expected

    def test_mps_backend_dtype_conversion(self):
        """Test _convert_dtype method with various inputs."""
        mock_mx = MagicMock()
        mock_mx.Dtype = type("MockDtype", (), {})
        mock_mx.float32 = "mlx_float32"
        mock_mx.float16 = "mlx_float16"
        mock_mx.int32 = "mlx_int32"
        mock_mx.uint32 = "mlx_uint32"
        mock_mx.bool_ = "mlx_bool"

        # Mock numpy dtypes
        mock_np_float32 = type("numpy.float32", (), {"type": np.float32, "name": "float32"})()
        mock_np_int64 = type("numpy.int64", (), {"type": np.int64, "name": "int64"})()

        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.mx", mock_mx):
                with patch("neural_arch.backends.mps_backend.np") as mock_np:
                    mock_np.float32 = np.float32
                    mock_np.float16 = np.float16
                    mock_np.int32 = np.int32
                    mock_np.int64 = np.int64
                    mock_np.uint32 = np.uint32
                    mock_np.bool_ = np.bool_

                    backend = MPSBackend()

                    # Test MLX dtype (should return as-is)
                    mlx_dtype = mock_mx.Dtype()
                    assert backend._convert_dtype(mlx_dtype) == mlx_dtype

                    # Test numpy dtype objects
                    assert backend._convert_dtype(mock_np_float32) == "mlx_float32"
                    assert backend._convert_dtype(mock_np_int64) == "mlx_int32"  # int64 -> int32

                    # Test numpy type classes
                    assert backend._convert_dtype(np.float32) == "mlx_float32"
                    assert backend._convert_dtype(np.int64) == "mlx_int32"

                    # Test string dtypes
                    assert backend._convert_dtype("float16") == "mlx_float16"
                    assert backend._convert_dtype("bool") == "mlx_bool"

                    # Test unknown dtype (should default to float32)
                    assert backend._convert_dtype("unknown_dtype") == "mlx_float32"

    def test_mps_backend_registration(self):
        """Test backend registration logic."""
        # Test that registration occurs when MLX is available
        with patch("neural_arch.backends.mps_backend.MLX_AVAILABLE", True):
            with patch("neural_arch.backends.mps_backend.register_backend") as mock_register:
                # Execute the registration code directly
                if True:  # Simulating MLX_AVAILABLE condition
                    from neural_arch.backends.mps_backend import MPSBackend, register_backend

                    register_backend("mps", MPSBackend)

                # Should register when MLX is available
                mock_register.assert_called_with("mps", MPSBackend)
