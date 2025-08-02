"""Abstract backend interface for compute operations."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np


class Backend(ABC):
    """Abstract interface for compute backends.

    This provides a unified interface for different compute backends
    (CPU, CUDA, MPS, etc.) allowing the same code to run on different
    hardware accelerators.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the system."""
        pass

    @property
    @abstractmethod
    def supports_gradients(self) -> bool:
        """Whether this backend supports automatic differentiation."""
        pass

    # Array creation
    @abstractmethod
    def array(self, data: Any, dtype: Optional[Any] = None) -> Any:
        """Create an array from data."""
        pass

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        """Create an array of zeros."""
        pass

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        """Create an array of ones."""
        pass

    @abstractmethod
    def full(self, shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None) -> Any:
        """Create an array filled with a value."""
        pass

    @abstractmethod
    def arange(
        self, start: float, stop: float, step: float = 1.0, dtype: Optional[Any] = None
    ) -> Any:
        """Create an array with evenly spaced values."""
        pass

    @abstractmethod
    def random_normal(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Create an array with random normal values."""
        pass

    @abstractmethod
    def random_uniform(
        self,
        shape: Tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Create an array with random uniform values."""
        pass

    # Shape operations
    @abstractmethod
    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        """Reshape an array."""
        pass

    @abstractmethod
    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        """Transpose an array."""
        pass

    @abstractmethod
    def squeeze(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """Remove single-dimensional entries."""
        pass

    @abstractmethod
    def expand_dims(self, x: Any, axis: int) -> Any:
        """Expand array dimensions."""
        pass

    # Math operations
    @abstractmethod
    def add(self, x: Any, y: Any) -> Any:
        """Element-wise addition."""
        pass

    @abstractmethod
    def subtract(self, x: Any, y: Any) -> Any:
        """Element-wise subtraction."""
        pass

    @abstractmethod
    def multiply(self, x: Any, y: Any) -> Any:
        """Element-wise multiplication."""
        pass

    @abstractmethod
    def divide(self, x: Any, y: Any) -> Any:
        """Element-wise division."""
        pass

    @abstractmethod
    def power(self, x: Any, y: Any) -> Any:
        """Element-wise power."""
        pass

    @abstractmethod
    def matmul(self, x: Any, y: Any) -> Any:
        """Matrix multiplication."""
        pass

    @abstractmethod
    def dot(self, x: Any, y: Any) -> Any:
        """Dot product."""
        pass

    # Reduction operations
    @abstractmethod
    def sum(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        """Sum array elements."""
        pass

    @abstractmethod
    def mean(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        """Mean of array elements."""
        pass

    @abstractmethod
    def max(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        """Maximum of array elements."""
        pass

    @abstractmethod
    def min(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        """Minimum of array elements."""
        pass

    @abstractmethod
    def argmax(self, x: Any, axis: Optional[int] = None) -> Any:
        """Indices of maximum values."""
        pass

    @abstractmethod
    def argmin(self, x: Any, axis: Optional[int] = None) -> Any:
        """Indices of minimum values."""
        pass

    # Activation functions
    @abstractmethod
    def exp(self, x: Any) -> Any:
        """Exponential function."""
        pass

    @abstractmethod
    def log(self, x: Any) -> Any:
        """Natural logarithm."""
        pass

    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        """Square root."""
        pass

    @abstractmethod
    def abs(self, x: Any) -> Any:
        """Absolute value."""
        pass

    @abstractmethod
    def sign(self, x: Any) -> Any:
        """Sign function."""
        pass

    @abstractmethod
    def clip(self, x: Any, min_val: float, max_val: float) -> Any:
        """Clip values to range."""
        pass

    # Comparison operations
    @abstractmethod
    def equal(self, x: Any, y: Any) -> Any:
        """Element-wise equality."""
        pass

    @abstractmethod
    def not_equal(self, x: Any, y: Any) -> Any:
        """Element-wise inequality."""
        pass

    @abstractmethod
    def less(self, x: Any, y: Any) -> Any:
        """Element-wise less than."""
        pass

    @abstractmethod
    def less_equal(self, x: Any, y: Any) -> Any:
        """Element-wise less than or equal."""
        pass

    @abstractmethod
    def greater(self, x: Any, y: Any) -> Any:
        """Element-wise greater than."""
        pass

    @abstractmethod
    def greater_equal(self, x: Any, y: Any) -> Any:
        """Element-wise greater than or equal."""
        pass

    # Array manipulation
    @abstractmethod
    def concatenate(self, arrays: List[Any], axis: int = 0) -> Any:
        """Concatenate arrays along an axis."""
        pass

    @abstractmethod
    def stack(self, arrays: List[Any], axis: int = 0) -> Any:
        """Stack arrays along a new axis."""
        pass

    @abstractmethod
    def split(self, x: Any, indices_or_sections: Union[int, List[int]], axis: int = 0) -> List[Any]:
        """Split array into multiple sub-arrays."""
        pass

    # Type conversion
    @abstractmethod
    def astype(self, x: Any, dtype: Any) -> Any:
        """Cast array to a different type."""
        pass

    @abstractmethod
    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert array to numpy array."""
        pass

    @abstractmethod
    def from_numpy(self, x: np.ndarray, dtype: Optional[Any] = None) -> Any:
        """Create array from numpy array."""
        pass

    # Device operations
    @abstractmethod
    def to_device(self, x: Any, device: str) -> Any:
        """Move array to specified device."""
        pass

    @abstractmethod
    def device_of(self, x: Any) -> str:
        """Get device of array."""
        pass

    # Utility functions
    @abstractmethod
    def is_array(self, x: Any) -> bool:
        """Check if object is an array for this backend."""
        pass

    @abstractmethod
    def shape(self, x: Any) -> Tuple[int, ...]:
        """Get shape of array."""
        pass

    @abstractmethod
    def size(self, x: Any) -> int:
        """Get total number of elements."""
        pass

    @abstractmethod
    def dtype(self, x: Any) -> Any:
        """Get data type of array."""
        pass

    # Advanced operations
    @abstractmethod
    def einsum(self, equation: str, *operands) -> Any:
        """Einstein summation."""
        pass

    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Select elements from x or y based on condition."""
        pass

    @abstractmethod
    def unique(self, x: Any, return_counts: bool = False) -> Union[Any, Tuple[Any, Any]]:
        """Find unique elements."""
        pass


# Global backend registry
_BACKENDS = {}
_CURRENT_BACKEND = None


def register_backend(name: str, backend_class: type) -> None:
    """Register a backend implementation."""
    _BACKENDS[name] = backend_class


def get_backend(name: Optional[str] = None) -> Backend:
    """Get a backend by name or return current backend."""
    global _CURRENT_BACKEND

    if name is None:
        if _CURRENT_BACKEND is None:
            # Default to numpy backend
            set_backend("numpy")
        return _CURRENT_BACKEND

    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}")

    backend_class = _BACKENDS[name]
    backend = backend_class()

    if not backend.is_available:
        raise RuntimeError(f"Backend '{name}' is not available on this system")

    return backend


def set_backend(name: str) -> None:
    """Set the current global backend."""
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = get_backend(name)


def available_backends() -> List[str]:
    """Get list of available backends on this system."""
    available = []
    for name, backend_class in _BACKENDS.items():
        try:
            backend = backend_class()
            if backend.is_available:
                available.append(name)
        except:
            pass
    return available


def current_backend() -> Optional[Backend]:
    """Get the current backend."""
    return _CURRENT_BACKEND
