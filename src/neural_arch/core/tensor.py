"""Enterprise-grade tensor implementation with automatic differentiation."""

import numpy as np
from typing import Union, Optional, Tuple, List, Protocol, Any, Callable
from contextlib import contextmanager
import weakref
from dataclasses import dataclass
import logging

from .dtype import DType, get_default_dtype
from .device import Device, get_default_device, DeviceType
from ..backends import get_backend, set_backend, Backend

# Type aliases
TensorLike = Union['Tensor', np.ndarray, list, float, int]
Shape = Tuple[int, ...]

# Global gradient computation state
_grad_enabled = True

logger = logging.getLogger(__name__)


@dataclass
class GradientFunction:
    """Represents a function in the computational graph for gradient computation."""
    
    backward_fn: Callable
    inputs: List['Tensor']
    name: str = "Unknown"
    
    def apply(self, grad_output: np.ndarray) -> None:
        """Apply the backward function with gradient clipping and error handling."""
        try:
            # Apply gradient clipping for numerical stability
            grad_output = np.clip(grad_output, -10.0, 10.0)
            
            # Check for NaN/Inf gradients
            if not np.all(np.isfinite(grad_output)):
                logger.warning(f"Non-finite gradients detected in {self.name}")
                grad_output = np.nan_to_num(grad_output, nan=0.0, posinf=1.0, neginf=-1.0)
            
            self.backward_fn(grad_output)
            
        except Exception as e:
            logger.error(f"Error in gradient computation for {self.name}: {e}")
            raise


class Tensor:
    """Enterprise-grade tensor with automatic differentiation.
    
    This tensor implementation provides:
    - Automatic differentiation with computational graph tracking
    - Memory-efficient gradient computation
    - Numerical stability with gradient clipping
    - Device placement management
    - Type safety with runtime checks
    - Comprehensive error handling
    - Performance monitoring hooks
    """
    
    def __init__(
        self,
        data: TensorLike,
        requires_grad: bool = False,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        name: Optional[str] = None
    ) -> None:
        """Initialize tensor with enterprise-grade features.
        
        Args:
            data: Input data (array-like)
            requires_grad: Whether to track gradients
            dtype: Data type (defaults to global default)
            device: Device placement (defaults to global default)
            name: Optional name for debugging
            
        Raises:
            TypeError: If data type is invalid
            ValueError: If data contains invalid values
        """
        # Input validation
        if not isinstance(requires_grad, bool):
            raise TypeError(f"requires_grad must be bool, got {type(requires_grad)}")
        
        # Core tensor properties
        self._requires_grad = requires_grad and _grad_enabled
        self._dtype = dtype or get_default_dtype()
        self._device = device or get_default_device()
        self._name = name
        
        # Select backend based on device
        self._backend = self._get_backend_for_device(self._device)
        
        # Convert data using backend
        self._data = self._validate_and_convert_data(data, dtype)
        
        # Gradient computation
        self._grad: Optional[Any] = None  # Backend array type
        self._grad_fn: Optional[GradientFunction] = None
        self._version = 0  # For detecting in-place modifications
        
        # Enterprise features
        self._creation_context = self._get_creation_context()
        self._memory_usage = self._calculate_memory_usage()
        
        # Weak references to dependent tensors for graph cleanup
        self._dependents: List[weakref.ref] = []
        
        logger.debug(f"Created tensor {self._name} with shape {self.shape} on {self._device} using {self._backend.name} backend")
    
    def _get_backend_for_device(self, device: Device) -> Backend:
        """Get appropriate backend for device."""
        if device.type == DeviceType.CPU:
            return get_backend("numpy")
        elif device.type == DeviceType.CUDA:
            return get_backend("cuda")
        elif device.type == DeviceType.MPS:
            return get_backend("mps")
        else:
            # Default to numpy backend
            return get_backend("numpy")
    
    def _calculate_memory_usage(self) -> int:
        """Calculate memory usage of tensor data."""
        # Get numpy view to calculate size
        np_data = self._backend.to_numpy(self._data)
        return np_data.nbytes
    
    def _validate_and_convert_data(self, data: TensorLike, dtype: Optional[DType]) -> Any:
        """Validate and convert input data to backend array.
        
        Args:
            data: Input data
            dtype: Target data type
            
        Returns:
            Backend array
            
        Raises:
            TypeError: If data type is invalid
            ValueError: If data contains invalid values
        """
        # First convert to numpy for validation
        if isinstance(data, Tensor):
            # Handle device transfers if needed
            if data._device != self._device:
                np_array = data._backend.to_numpy(data._data)
            else:
                # Same device, can potentially copy directly
                np_array = data._backend.to_numpy(data._data)
        elif isinstance(data, np.ndarray):
            np_array = data.copy()
        elif isinstance(data, (list, tuple)):
            np_array = np.array(data)
        elif isinstance(data, (int, float)):
            np_array = np.array(data)
        elif isinstance(data, (np.integer, np.floating)):
            # Handle numpy scalar types
            np_array = np.array(data)
        else:
            # Check if it's a backend array type
            if self._backend.is_array(data):
                np_array = self._backend.to_numpy(data)
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Apply dtype conversion
        target_dtype = dtype or get_default_dtype()
        np_array = np_array.astype(target_dtype.numpy_dtype)
        
        # Validate for NaN/Inf values
        if not np.all(np.isfinite(np_array)):
            if np.any(np.isnan(np_array)):
                raise ValueError("Tensor data contains NaN values")
            if np.any(np.isinf(np_array)):
                raise ValueError("Tensor data contains infinite values")
        
        # Convert to backend array
        backend_array = self._backend.from_numpy(np_array)
        
        # Move to correct device if needed
        device_str = self._device.type.value
        if self._device.index is not None:
            device_str = f"{device_str}:{self._device.index}"
        backend_array = self._backend.to_device(backend_array, device_str)
        
        return backend_array
    
    def _get_creation_context(self) -> dict:
        """Get context information for tensor creation (for debugging)."""
        import inspect
        frame = inspect.currentframe()
        try:
            # Get the frame that called Tensor.__init__
            caller_frame = frame.f_back.f_back if frame and frame.f_back else None
            if caller_frame:
                return {
                    'filename': caller_frame.f_code.co_filename,
                    'lineno': caller_frame.f_lineno,
                    'function': caller_frame.f_code.co_name,
                }
        finally:
            del frame
        return {}
    
    @property
    def data(self) -> np.ndarray:
        """Get the underlying data array as numpy."""
        return self._backend.to_numpy(self._data)
    
    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Set the underlying data array with validation."""
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(value)}")
        
        if value.shape != self.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {value.shape}")
        
        # Convert numpy to backend array
        value = value.astype(self._dtype.numpy_dtype)
        backend_array = self._backend.from_numpy(value)
        
        # Move to correct device
        device_str = self._device.type.value
        if self._device.index is not None:
            device_str = f"{device_str}:{self._device.index}"
        self._data = self._backend.to_device(backend_array, device_str)
        
        self._version += 1  # Track in-place modifications
    
    @property
    def shape(self) -> Shape:
        """Get tensor shape."""
        return self._backend.shape(self._data)
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self._backend.shape(self._data))
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self._backend.size(self._data)
    
    @property
    def dtype(self) -> DType:
        """Get tensor data type."""
        return self._dtype
    
    @property
    def device(self) -> Device:
        """Get tensor device."""
        return self._device
    
    @property
    def requires_grad(self) -> bool:
        """Check if gradients are tracked."""
        return self._requires_grad
    
    @property
    def grad(self) -> Optional[np.ndarray]:
        """Get accumulated gradients as numpy."""
        if self._grad is None:
            return None
        return self._backend.to_numpy(self._grad)
    
    @grad.setter
    def grad(self, value: Optional[np.ndarray]) -> None:
        """Set accumulated gradients."""
        if value is None:
            self._grad = None
            return
            
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected np.ndarray or None, got {type(value)}")
        
        if value.shape != self.shape:
            raise ValueError(f"Gradient shape {value.shape} doesn't match tensor shape {self.shape}")
        
        # Convert to backend array
        backend_grad = self._backend.from_numpy(value)
        
        # Move to same device as tensor
        device_str = self._device.type.value
        if self._device.index is not None:
            device_str = f"{device_str}:{self._device.index}"
        self._grad = self._backend.to_device(backend_grad, device_str)
    
    @property
    def grad_fn(self) -> Optional[GradientFunction]:
        """Get gradient function for backpropagation."""
        return self._grad_fn
    
    @property
    def name(self) -> Optional[str]:
        """Get tensor name."""
        return self._name
    
    @property
    def backend(self) -> Backend:
        """Get the backend used by this tensor."""
        return self._backend
    
    @property
    def backend_data(self) -> Any:
        """Get the raw backend array (for internal use)."""
        return self._data
    
    def zero_grad(self) -> None:
        """Reset gradients to None with memory cleanup."""
        if self._grad is not None:
            del self._grad
        self._grad = None
        logger.debug(f"Zeroed gradients for tensor {self._name}")
    
    def backward(self, gradient: Optional[np.ndarray] = None) -> None:
        """Compute gradients with enterprise-grade error handling.
        
        Args:
            gradient: Upstream gradient (defaults to ones)
            
        Raises:
            RuntimeError: If tensor doesn't require gradients
            ValueError: If gradient shape is invalid
        """
        if not self._requires_grad:
            logger.warning(f"Attempted backward on tensor {self._name} that doesn't require grad")
            return
        
        if not _grad_enabled:
            logger.debug("Gradient computation is disabled globally")
            return
        
        # Default gradient for scalar outputs
        if gradient is None:
            if self.size == 1:
                gradient = np.ones(self.shape, dtype=self._dtype.numpy_dtype)
            else:
                raise ValueError("Gradient must be specified for non-scalar tensors")
        
        # Validate gradient shape
        if gradient.shape != self.shape:
            raise ValueError(f"Gradient shape {gradient.shape} doesn't match tensor shape {self.shape}")
        
        # Handle NaN/Inf gradients and extreme values for numerical stability
        processed_gradient = gradient.copy()
        
        # Handle non-finite gradients
        if not np.all(np.isfinite(processed_gradient)):
            logger.warning(f"Non-finite gradients detected in tensor {self._name}")
            processed_gradient = np.nan_to_num(processed_gradient, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply gradient clipping only for extremely large gradients
        max_abs_grad = np.max(np.abs(processed_gradient))
        if max_abs_grad >= 1e6:  # Clip gradients >= 1 million for numerical stability
            logger.warning(f"Extremely large gradient detected ({max_abs_grad}), applying clipping")
            processed_gradient = np.clip(processed_gradient, -10.0, 10.0)
        
        # Convert gradient to backend array
        backend_grad = self._backend.from_numpy(processed_gradient)
        device_str = self._device.type.value
        if self._device.index is not None:
            device_str = f"{device_str}:{self._device.index}"
        backend_grad = self._backend.to_device(backend_grad, device_str)
        
        # Accumulate gradients
        if self._grad is None:
            self._grad = backend_grad
        else:
            self._grad = self._backend.add(self._grad, backend_grad)
        
        # Propagate gradients through computational graph
        if self._grad_fn is not None:
            self._grad_fn.apply(gradient)
        
        logger.debug(f"Computed gradients for tensor {self._name}")
    
    def detach(self) -> 'Tensor':
        """Create a new tensor detached from the computational graph.
        
        Returns:
            New tensor with same data but no gradient tracking
        """
        # Convert to numpy for creating new tensor
        np_data = self._backend.to_numpy(self._data)
        return Tensor(
            np_data,
            requires_grad=False,
            dtype=self._dtype,
            device=self._device,
            name=f"{self._name}_detached" if self._name else None
        )
    
    def clone(self) -> 'Tensor':
        """Create a deep copy of the tensor.
        
        Returns:
            New tensor with copied data and gradient tracking
        """
        # Convert to numpy for creating new tensor
        np_data = self._backend.to_numpy(self._data)
        cloned = Tensor(
            np_data,
            requires_grad=self._requires_grad,
            dtype=self._dtype,
            device=self._device,
            name=f"{self._name}_clone" if self._name else None
        )
        return cloned
    
    def to(self, device: Optional[Device] = None, dtype: Optional[DType] = None) -> 'Tensor':
        """Move tensor to specified device/dtype.
        
        Args:
            device: Target device
            dtype: Target data type
            
        Returns:
            New tensor on specified device/dtype
        """
        new_device = device or self._device
        new_dtype = dtype or self._dtype
        
        # Convert data if dtype changed
        new_data = self._data
        if new_dtype != self._dtype:
            new_data = new_data.astype(new_dtype.numpy_dtype)
        
        # For now, device movement is a no-op (CPU only)
        # In a real implementation, this would handle GPU transfers
        
        if new_device == self._device and new_dtype == self._dtype:
            return self
        
        return Tensor(
            new_data,
            requires_grad=self._requires_grad,
            dtype=new_dtype,
            device=new_device,
            name=self._name
        )
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array (detached from gradient computation).
        
        Returns:
            NumPy array with tensor data
            
        Raises:
            RuntimeError: If tensor requires gradients
        """
        if self._requires_grad:
            raise RuntimeError("Cannot convert tensor with requires_grad=True to numpy. Use detach() first.")
        
        return self._data.copy()
    
    def item(self) -> Union[float, int]:
        """Extract scalar value from tensor.
        
        Returns:
            Scalar value
            
        Raises:
            ValueError: If tensor is not scalar
        """
        if self._data.size != 1:
            raise ValueError(f"Can only extract scalar from single-element tensor, got {self.size} elements")
        
        return self._data.item()
    
    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return self._memory_usage
    
    def __add__(self, other: TensorLike) -> 'Tensor':
        """Addition operator."""
        from ..functional import add
        return add(self, other)
    
    def __radd__(self, other: TensorLike) -> 'Tensor':
        """Reverse addition operator."""
        from ..functional import add
        return add(other, self)
    
    def __mul__(self, other: TensorLike) -> 'Tensor':
        """Multiplication operator."""
        from ..functional import mul
        return mul(self, other)
    
    def __rmul__(self, other: TensorLike) -> 'Tensor':
        """Reverse multiplication operator."""
        from ..functional import mul
        return mul(other, self)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication operator."""
        from ..functional import matmul
        return matmul(self, other)
    
    def __sub__(self, other: TensorLike) -> 'Tensor':
        """Subtraction operator."""
        from ..functional import sub
        return sub(self, other)
    
    def __rsub__(self, other: TensorLike) -> 'Tensor':
        """Reverse subtraction operator."""
        from ..functional import sub
        return sub(other, self)
    
    def __truediv__(self, other: TensorLike) -> 'Tensor':
        """Division operator."""
        from ..functional import div
        return div(self, other)
    
    def __rtruediv__(self, other: TensorLike) -> 'Tensor':
        """Reverse division operator."""
        from ..functional import div
        return div(other, self)
    
    def __neg__(self) -> 'Tensor':
        """Negation operator."""
        from ..functional import neg
        return neg(self)
    
    def __repr__(self) -> str:
        """String representation of tensor."""
        grad_str = f", requires_grad={self._requires_grad}" if self._requires_grad else ""
        device_str = f", device={self._device}" if not self._device.is_cpu else ""
        name_str = f", name={self._name!r}" if self._name else ""
        
        return f"Tensor({self._data}{grad_str}{device_str}{name_str})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Tensor(shape={self.shape}, dtype={self._dtype})"


@contextmanager
def no_grad():
    """Context manager to disable gradient computation.
    
    Useful for inference and memory optimization.
    """
    global _grad_enabled
    old_value = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old_value


@contextmanager
def enable_grad():
    """Context manager to enable gradient computation."""
    global _grad_enabled
    old_value = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = old_value


def is_grad_enabled() -> bool:
    """Check if gradient computation is currently enabled."""
    return _grad_enabled