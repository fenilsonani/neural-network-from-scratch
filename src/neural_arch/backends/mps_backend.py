"""Apple Metal Performance Shaders backend using MLX."""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

from .backend import Backend, register_backend

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

if TYPE_CHECKING:
    import mlx.core as mx


class MPSBackend(Backend):
    """MLX backend for Apple Silicon GPU computation."""

    def __init__(self):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Install it with: pip install mlx")

    @property
    def name(self) -> str:
        return "mps"

    @property
    def is_available(self) -> bool:
        if not MLX_AVAILABLE:
            return False
        try:
            # Test if we can create an array on GPU
            test = mx.array([1.0])
            return True
        except Exception:
            return False

    @property
    def supports_gradients(self) -> bool:
        return True  # MLX has built-in autograd

    # Dtype attributes
    @property
    def float32(self):
        return mx.float32

    @property
    def float64(self):
        return mx.float64

    @property
    def int32(self):
        return mx.int32

    @property
    def int64(self):
        return mx.int64

    @property
    def bool(self):
        return mx.bool_

    # Array creation
    def array(self, data: Any, dtype: Optional[Any] = None) -> Any:
        if dtype is None:
            dtype = mx.float32
        else:
            dtype = self._convert_dtype(dtype)
        return mx.array(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        dtype = self._convert_dtype(dtype) if dtype else mx.float32
        return mx.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        dtype = self._convert_dtype(dtype) if dtype else mx.float32
        return mx.ones(shape, dtype=dtype)

    def full(self, shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None) -> Any:
        dtype = self._convert_dtype(dtype) if dtype else mx.float32
        return mx.full(shape, fill_value, dtype=dtype)

    def arange(
        self, start: float, stop: float, step: float = 1.0, dtype: Optional[Any] = None
    ) -> Any:
        dtype = self._convert_dtype(dtype) if dtype else mx.float32
        return mx.arange(start, stop, step, dtype=dtype)

    def random_normal(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Optional[Any] = None,
    ) -> Any:
        # MLX uses a key for random generation
        key = mx.random.key(np.random.randint(0, 2**32))
        result = mx.random.normal(shape, key=key) * std + mean
        if dtype:
            result = result.astype(self._convert_dtype(dtype))
        return result

    def random_uniform(
        self,
        shape: Tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        dtype: Optional[Any] = None,
    ) -> Any:
        key = mx.random.key(np.random.randint(0, 2**32))
        result = mx.random.uniform(shape, low=low, high=high, key=key)
        if dtype:
            result = result.astype(self._convert_dtype(dtype))
        return result

    # Shape operations
    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        return x.reshape(shape)

    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        if axes is None:
            return x.T
        return mx.transpose(x, axes)

    def squeeze(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        return mx.squeeze(x, axis)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return mx.expand_dims(x, axis)

    # Math operations
    def add(self, x: Any, y: Any) -> Any:
        return mx.add(x, y)

    def subtract(self, x: Any, y: Any) -> Any:
        return mx.subtract(x, y)

    def multiply(self, x: Any, y: Any) -> Any:
        return mx.multiply(x, y)

    def divide(self, x: Any, y: Any) -> Any:
        return mx.divide(x, y)

    def power(self, x: Any, y: Any) -> Any:
        return mx.power(x, y)

    def matmul(self, x: Any, y: Any) -> Any:
        return mx.matmul(x, y)

    def dot(self, x: Any, y: Any) -> Any:
        # MLX doesn't have a separate dot, use matmul
        return mx.matmul(x, y)

    # Reduction operations
    def sum(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        return mx.sum(x, axis=axis, keepdims=keepdims)

    def mean(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        return mx.mean(x, axis=axis, keepdims=keepdims)

    def max(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        return mx.max(x, axis=axis, keepdims=keepdims)

    def min(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        return mx.min(x, axis=axis, keepdims=keepdims)

    def argmax(self, x: Any, axis: Optional[int] = None) -> Any:
        return mx.argmax(x, axis=axis)

    def argmin(self, x: Any, axis: Optional[int] = None) -> Any:
        return mx.argmin(x, axis=axis)

    # Activation functions
    def exp(self, x: Any) -> Any:
        return mx.exp(x)

    def log(self, x: Any) -> Any:
        return mx.log(x)

    def sqrt(self, x: Any) -> Any:
        return mx.sqrt(x)

    def abs(self, x: Any) -> Any:
        return mx.abs(x)

    def sign(self, x: Any) -> Any:
        return mx.sign(x)

    def clip(self, x: Any, min_val: float, max_val: float) -> Any:
        return mx.clip(x, min_val, max_val)

    # Comparison operations
    def equal(self, x: Any, y: Any) -> Any:
        return mx.equal(x, y)

    def not_equal(self, x: Any, y: Any) -> Any:
        return mx.not_equal(x, y)

    def less(self, x: Any, y: Any) -> Any:
        return mx.less(x, y)

    def less_equal(self, x: Any, y: Any) -> Any:
        return mx.less_equal(x, y)

    def greater(self, x: Any, y: Any) -> Any:
        return mx.greater(x, y)

    def greater_equal(self, x: Any, y: Any) -> Any:
        return mx.greater_equal(x, y)

    # Array manipulation
    def concatenate(self, arrays: List[Any], axis: int = 0) -> Any:
        return mx.concatenate(arrays, axis=axis)

    def stack(self, arrays: List[Any], axis: int = 0) -> Any:
        return mx.stack(arrays, axis=axis)

    def split(self, x: Any, indices_or_sections: Union[int, List[int]], axis: int = 0) -> List[Any]:
        return mx.split(x, indices_or_sections, axis=axis)

    # Type conversion
    def astype(self, x: Any, dtype: Any) -> Any:
        return x.astype(self._convert_dtype(dtype))

    def to_numpy(self, x: Any) -> np.ndarray:
        # MLX arrays can be converted to numpy
        return np.array(x)

    def from_numpy(self, x: np.ndarray, dtype: Optional[Any] = None) -> Any:
        if dtype:
            return mx.array(x, dtype=self._convert_dtype(dtype))
        return mx.array(x)

    # Device operations
    def to_device(self, x: Any, device: str) -> Any:
        # MLX uses unified memory, so no explicit device transfers
        if device not in ["cpu", "mps", "gpu"]:
            raise ValueError(f"MLX backend supports cpu/mps/gpu, got {device}")
        return x

    def device_of(self, x: Any) -> str:
        # MLX uses unified memory
        return "mps"

    # Utility functions
    def is_array(self, x: Any) -> bool:
        if mx is None:
            return False
        return hasattr(mx, "array") and isinstance(x, mx.array)

    def shape(self, x: Any) -> Tuple[int, ...]:
        return x.shape

    def size(self, x: Any) -> int:
        return x.size

    def dtype(self, x: Any) -> Any:
        return x.dtype

    # Advanced operations
    def einsum(self, equation: str, *operands) -> Any:
        # MLX doesn't have einsum yet, fallback to numpy
        np_operands = [self.to_numpy(op) for op in operands]
        result = np.einsum(equation, *np_operands)
        return self.from_numpy(result)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return mx.where(condition, x, y)

    def unique(self, x: Any, return_counts: bool = False) -> Union[Any, Tuple[Any, Any]]:
        # MLX doesn't have unique yet, fallback to numpy
        np_x = self.to_numpy(x)
        if return_counts:
            unique_vals, counts = np.unique(np_x, return_counts=True)
            return self.from_numpy(unique_vals), self.from_numpy(counts)
        return self.from_numpy(np.unique(np_x))

    def _convert_dtype(self, dtype: Any) -> Any:
        """Convert numpy dtype to MLX dtype."""
        if isinstance(dtype, mx.Dtype):
            return dtype

        # Handle numpy dtype objects
        if hasattr(dtype, "type"):
            dtype = dtype.type

        dtype_map = {
            np.float32: mx.float32,
            np.float16: mx.float16,
            np.int32: mx.int32,
            np.int64: mx.int32,  # MLX doesn't have int64, use int32
            np.uint32: mx.uint32,
            np.bool_: mx.bool_,
            "float32": mx.float32,
            "float16": mx.float16,
            "int32": mx.int32,
            "int64": mx.int32,  # MLX doesn't have int64, use int32
            "uint32": mx.uint32,
            "bool": mx.bool_,
        }

        if dtype in dtype_map:
            return dtype_map[dtype]

        # Try to convert numpy dtype
        if hasattr(dtype, "name") and dtype.name in dtype_map:
            return dtype_map[dtype.name]

        # Default to float32
        return mx.float32


# Register the MPS backend
if MLX_AVAILABLE:
    register_backend("mps", MPSBackend)
