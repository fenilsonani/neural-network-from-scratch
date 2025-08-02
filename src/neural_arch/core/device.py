"""Device management for tensor operations."""

import platform
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class DeviceType(Enum):
    """Supported device types for tensor operations."""
    
    CPU = "cpu"
    CUDA = "cuda"  # NVIDIA GPU support
    MPS = "mps"    # Apple Silicon GPU support
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Device:
    """Represents a device where tensors can be stored and operations executed."""
    
    type: DeviceType
    index: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate device configuration."""
        if self.type in (DeviceType.CUDA, DeviceType.MPS) and self.index is not None:
            if self.index < 0:
                raise ValueError(f"Device index must be non-negative, got {self.index}")
    
    @classmethod
    def cpu(cls) -> 'Device':
        """Create a CPU device."""
        return cls(DeviceType.CPU)
    
    @classmethod
    def cuda(cls, index: int = 0) -> 'Device':
        """Create a CUDA device."""
        return cls(DeviceType.CUDA, index)
    
    @classmethod
    def mps(cls, index: int = 0) -> 'Device':
        """Create an MPS (Metal Performance Shaders) device."""
        return cls(DeviceType.MPS, index)
    
    @classmethod
    def from_string(cls, device_str: str) -> 'Device':
        """Create device from string representation.
        
        Args:
            device_str: String like "cpu", "cuda", "cuda:0"
            
        Returns:
            Device instance
            
        Raises:
            ValueError: If device string format is invalid
        """
        device_str = device_str.lower().strip()
        
        if device_str == "cpu":
            return cls.cpu()
        elif device_str == "cuda":
            return cls.cuda(0)
        elif device_str.startswith("cuda:"):
            try:
                index = int(device_str.split(":")[1])
                return cls.cuda(index)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid CUDA device string: {device_str}") from e
        elif device_str == "mps":
            return cls.mps(0)
        elif device_str.startswith("mps:"):
            try:
                index = int(device_str.split(":")[1])
                return cls.mps(index)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid MPS device string: {device_str}") from e
        else:
            raise ValueError(f"Unsupported device string: {device_str}")
    
    @property
    def is_cpu(self) -> bool:
        """Check if this is a CPU device."""
        return self.type == DeviceType.CPU
    
    @property
    def is_cuda(self) -> bool:
        """Check if this is a CUDA device."""
        return self.type == DeviceType.CUDA
    
    @property
    def is_mps(self) -> bool:
        """Check if this is an MPS device."""
        return self.type == DeviceType.MPS
    
    @property
    def is_gpu(self) -> bool:
        """Check if this is any GPU device."""
        return self.type in (DeviceType.CUDA, DeviceType.MPS)
    
    def __str__(self) -> str:
        """String representation of the device."""
        if self.type == DeviceType.CPU:
            return "cpu"
        elif self.type == DeviceType.CUDA:
            if self.index is not None:
                return f"cuda:{self.index}"
            return "cuda"
        elif self.type == DeviceType.MPS:
            if self.index is not None:
                return f"mps:{self.index}"
            return "mps"
        return str(self.type.value)
    
    def __repr__(self) -> str:
        return f"Device({self.type.value!r}, {self.index!r})"


# Global default device
_DEFAULT_DEVICE = Device.cpu()


def get_default_device() -> Device:
    """Get the default device for new tensors."""
    return _DEFAULT_DEVICE


def set_default_device(device: Union[Device, str]) -> None:
    """Set the default device for new tensors."""
    global _DEFAULT_DEVICE
    
    if isinstance(device, str):
        device = Device.from_string(device)
    elif not isinstance(device, Device):
        raise TypeError(f"Expected Device or str, got {type(device)}")
    
    _DEFAULT_DEVICE = device


def get_device_capabilities() -> dict:
    """Get information about available devices and their capabilities."""
    import sys
    
    capabilities = {
        "cpu": {
            "available": True,
            "cores": "unknown",  # Could be detected with psutil
            "architecture": platform.machine(),
        },
        "cuda": {
            "available": False,
            "devices": [],
        },
        "mps": {
            "available": False,
            "unified_memory": False,
        }
    }
    
    # Check for CUDA availability
    try:
        import cupy as cp
        capabilities["cuda"]["available"] = True
        # Get CUDA device info
        for i in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                capabilities["cuda"]["devices"].append({
                    "index": i,
                    "name": props["name"].decode(),
                    "memory": props["totalGlobalMem"],
                    "compute_capability": f"{props['major']}.{props['minor']}"
                })
    except:
        pass
    
    # Check for MPS availability (Apple Silicon)
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx.core as mx

            # Test if we can create an array
            test = mx.array([1.0])
            capabilities["mps"]["available"] = True
            capabilities["mps"]["unified_memory"] = True
        except:
            pass
    
    return capabilities