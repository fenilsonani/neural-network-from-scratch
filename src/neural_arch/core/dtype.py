"""Data type management for tensors."""

from enum import Enum
from typing import Type, Union

import numpy as np


class DType(Enum):
    """Supported data types for tensors."""

    FLOAT16 = np.float16  # Half precision for mixed precision training
    FLOAT32 = np.float32
    FLOAT64 = np.float64
    INT32 = np.int32
    INT64 = np.int64
    BOOL = np.bool_

    @property
    def numpy_dtype(self) -> Type[np.number]:
        """Get the corresponding NumPy data type."""
        return self.value

    @property
    def is_floating(self) -> bool:
        """Check if this is a floating-point data type."""
        return self in (DType.FLOAT16, DType.FLOAT32, DType.FLOAT64)

    @property
    def is_integer(self) -> bool:
        """Check if this is an integer data type."""
        return self in (DType.INT32, DType.INT64)

    @property
    def bytes_per_element(self) -> int:
        """Get the number of bytes per element."""
        return np.dtype(self.value).itemsize

    @classmethod
    def from_numpy(cls, numpy_dtype: Union[np.dtype, Type[np.number], str]) -> 'DType':
        """Create DType from NumPy dtype."""
        if isinstance(numpy_dtype, str):
            numpy_dtype = np.dtype(numpy_dtype)
        elif not isinstance(numpy_dtype, np.dtype):
            numpy_dtype = np.dtype(numpy_dtype)

        for dtype in cls:
            if dtype.numpy_dtype == numpy_dtype.type:
                return dtype

        raise ValueError(f"Unsupported NumPy dtype: {numpy_dtype}")

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return f"DType.{self.name}"


# Global default dtype
_DEFAULT_DTYPE = DType.FLOAT32


def get_default_dtype() -> DType:
    """Get the default data type for new tensors."""
    return _DEFAULT_DTYPE


def set_default_dtype(dtype: DType) -> None:
    """Set the default data type for new tensors."""
    global _DEFAULT_DTYPE
    if not isinstance(dtype, DType):
        raise TypeError(f"Expected DType, got {type(dtype)}")
    _DEFAULT_DTYPE = dtype


# Type aliases for convenience
FloatDType = Union[DType]  # Could be extended for more specific float types
IntDType = Union[DType]    # Could be extended for more specific int types
