"""Backend implementations for different compute devices."""

from .backend import (
    Backend, 
    get_backend, 
    set_backend, 
    available_backends,
    current_backend,
    register_backend
)
from .numpy_backend import NumpyBackend
from .utils import (
    auto_select_backend,
    get_device_for_backend,
    get_backend_for_device,
    print_available_devices
)

__all__ = [
    "Backend",
    "get_backend", 
    "set_backend",
    "available_backends",
    "current_backend",
    "register_backend",
    "NumpyBackend",
    "auto_select_backend",
    "get_device_for_backend",
    "get_backend_for_device",
    "print_available_devices",
]

# Try to import optional backends
try:
    from .mps_backend import MPSBackend
    __all__.append("MPSBackend")
except ImportError:
    pass

try:
    from .cuda_backend import CudaBackend
    __all__.append("CudaBackend")
except ImportError:
    pass

try:
    from .jax_backend import JAXBackend
    __all__.append("JAXBackend")
except ImportError:
    pass

# Import JIT backend for ultra-high performance
try:
    from .jit_backend import JITBackend
    __all__.append("JITBackend")
except ImportError:
    pass