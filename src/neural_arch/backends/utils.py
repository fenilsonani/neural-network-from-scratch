"""Backend utilities for automatic selection and configuration."""

import platform
import sys
from typing import Optional
from .backend import get_backend, set_backend, available_backends, Backend


def auto_select_backend(prefer_gpu: bool = True) -> Backend:
    """Automatically select the best available backend.
    
    Args:
        prefer_gpu: If True, prefer GPU backends over CPU.
        
    Returns:
        The selected backend instance.
    """
    available = available_backends()
    
    if not prefer_gpu:
        set_backend("numpy")
        return get_backend()
    
    # On Apple Silicon Macs, prefer MPS
    if sys.platform == "darwin" and platform.machine() == "arm64":
        if "mps" in available:
            set_backend("mps")
            return get_backend()
    
    # On other systems, prefer CUDA
    if "cuda" in available:
        set_backend("cuda")
        return get_backend()
    
    # Fallback to CPU
    set_backend("numpy")
    return get_backend()


def get_device_for_backend(backend_name: Optional[str] = None) -> str:
    """Get the appropriate device string for a backend.
    
    Args:
        backend_name: Name of the backend, or None for current.
        
    Returns:
        Device string like "cpu", "cuda", "mps".
    """
    if backend_name is None:
        backend = get_backend()
        backend_name = backend.name
    
    if backend_name == "numpy":
        return "cpu"
    elif backend_name == "cuda":
        return "cuda"
    elif backend_name == "mps":
        return "mps"
    else:
        return "cpu"


def get_backend_for_device(device: str) -> str:
    """Get the appropriate backend name for a device.
    
    Args:
        device: Device string like "cpu", "cuda", "mps".
        
    Returns:
        Backend name.
    """
    device = device.lower().split(":")[0]  # Remove device index
    
    if device == "cpu":
        return "numpy"
    elif device == "cuda":
        return "cuda"
    elif device == "mps":
        return "mps"
    else:
        return "numpy"


def print_available_devices():
    """Print information about available compute devices."""
    from ..core.device import get_device_capabilities
    
    print("Available Compute Devices:")
    print("-" * 50)
    
    caps = get_device_capabilities()
    
    # CPU info
    print(f"CPU:")
    print(f"  Architecture: {caps['cpu']['architecture']}")
    print(f"  Available: {caps['cpu']['available']}")
    
    # CUDA info
    if caps['cuda']['available']:
        print(f"\nCUDA:")
        print(f"  Available: True")
        for device in caps['cuda']['devices']:
            print(f"  Device {device['index']}: {device['name']}")
            print(f"    Memory: {device['memory'] / 1e9:.1f} GB")
            print(f"    Compute Capability: {device['compute_capability']}")
    else:
        print(f"\nCUDA: Not available")
    
    # MPS info
    if caps['mps']['available']:
        print(f"\nMPS (Metal Performance Shaders):")
        print(f"  Available: True")
        print(f"  Unified Memory: {caps['mps']['unified_memory']}")
    else:
        print(f"\nMPS: Not available")
    
    # Available backends
    print(f"\nAvailable Backends: {', '.join(available_backends())}")
    print("-" * 50)