"""NumPy backend implementation."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .backend import Backend, register_backend


class NumpyBackend(Backend):
    """NumPy backend for CPU computation."""

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def is_available(self) -> bool:
        return True  # NumPy is always available

    @property
    def supports_gradients(self) -> bool:
        return False  # NumPy doesn't have built-in autograd

    @property
    def available(self) -> bool:
        """Alias for is_available for backwards compatibility."""
        return self.is_available

    # Dtype attributes
    @property
    def float32(self):
        return np.float32

    @property
    def float64(self):
        return np.float64

    @property
    def int32(self):
        return np.int32

    @property
    def int64(self):
        return np.int64

    @property
    def bool(self):
        return np.bool_

    # Array creation
    def array(self, data: Any, dtype: Optional[Any] = None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype or np.float32)

    def full(
        self, shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> np.ndarray:
        return np.full(shape, fill_value, dtype=dtype or np.float32)

    def arange(
        self, start: float, stop: float, step: float = 1.0, dtype: Optional[Any] = None
    ) -> np.ndarray:
        return np.arange(start, stop, step, dtype=dtype or np.float32)

    def random_normal(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Optional[Any] = None,
    ) -> np.ndarray:
        result = np.random.normal(mean, std, shape)
        if dtype:
            result = result.astype(dtype)
        return result.astype(np.float32)

    def random_uniform(
        self,
        shape: Tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        dtype: Optional[Any] = None,
    ) -> np.ndarray:
        result = np.random.uniform(low, high, shape)
        if dtype:
            result = result.astype(dtype)
        return result.astype(np.float32)

    # Shape operations
    def reshape(self, x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        return x.reshape(shape)

    def flatten(self, x: np.ndarray) -> np.ndarray:
        """Flatten array to 1D."""
        return x.flatten()

    def transpose(self, x: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        return np.transpose(x, axes)

    def squeeze(
        self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        return np.squeeze(x, axis)

    def expand_dims(self, x: np.ndarray, axis: int) -> np.ndarray:
        return np.expand_dims(x, axis)

    # Math operations
    def add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.add(x, y)

    def subtract(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.subtract(x, y)

    def multiply(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.multiply(x, y)

    # Alias for compatibility
    def mul(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.multiply(x, y)

    def divide(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.divide(x, y)

    def power(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.power(x, y)

    def matmul(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.matmul(x, y)

    def dot(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.dot(x, y)

    # Reduction operations
    def sum(
        self,
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(
        self,
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(
        self,
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.max(x, axis=axis, keepdims=keepdims)

    def min(
        self,
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.min(x, axis=axis, keepdims=keepdims)

    def argmax(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.argmax(x, axis=axis)

    def argmin(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.argmin(x, axis=axis)

    # Activation functions
    def exp(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def sign(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)

    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return np.clip(x, min_val, max_val)

    def maximum(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Element-wise maximum of two arrays."""
        return np.maximum(x, y)

    def minimum(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Element-wise minimum of two arrays."""
        return np.minimum(x, y)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Compute hyperbolic tangent function."""
        return np.tanh(x)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax function."""
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        x_shifted = x - x_max
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    # Comparison operations
    def equal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.equal(x, y)

    def not_equal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.not_equal(x, y)

    def less(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.less(x, y)

    def less_equal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.less_equal(x, y)

    def greater(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.greater(x, y)

    def greater_equal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.greater_equal(x, y)

    # Array manipulation
    def concatenate(self, arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def split(
        self, x: np.ndarray, indices_or_sections: Union[int, List[int]], axis: int = 0
    ) -> List[np.ndarray]:
        return np.split(x, indices_or_sections, axis=axis)

    # Type conversion
    def astype(self, x: np.ndarray, dtype: Any) -> np.ndarray:
        return x.astype(dtype)

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return x  # Already numpy

    def from_numpy(self, x: np.ndarray, dtype: Optional[Any] = None) -> np.ndarray:
        if dtype:
            return x.astype(dtype)
        return x

    # Device operations (CPU only)
    def to_device(self, x: np.ndarray, device: str) -> np.ndarray:
        if device != "cpu":
            raise ValueError(f"NumPy backend only supports CPU, got {device}")
        return x

    def device_of(self, x: np.ndarray) -> str:
        return "cpu"

    # Utility functions
    def is_array(self, x: Any) -> bool:
        return isinstance(x, np.ndarray)

    def shape(self, x: np.ndarray) -> Tuple[int, ...]:
        return x.shape

    def size(self, x: np.ndarray) -> int:
        return x.size

    def dtype(self, x: np.ndarray) -> Any:
        return x.dtype

    # Advanced operations
    def einsum(self, equation: str, *operands) -> np.ndarray:
        return np.einsum(equation, *operands)

    def where(self, condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(condition, x, y)

    def unique(
        self, x: np.ndarray, return_counts: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if return_counts:
            return np.unique(x, return_counts=True)
        return np.unique(x)


# Register the numpy backend
register_backend("numpy", NumpyBackend)
