"""Complete test coverage for device module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import platform
from unittest.mock import patch, MagicMock
from neural_arch.core.device import (
    DeviceType, Device, get_default_device, set_default_device,
    get_device_capabilities
)


class TestDeviceType:
    """Test DeviceType enum."""
    
    def test_device_type_values(self):
        """Test device type enum values."""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"
    
    def test_device_type_str(self):
        """Test string representation."""
        assert str(DeviceType.CPU) == "cpu"
        assert str(DeviceType.CUDA) == "cuda"
        assert str(DeviceType.MPS) == "mps"


class TestDevice:
    """Test Device class."""
    
    def test_cpu_device_creation(self):
        """Test CPU device creation."""
        device = Device.cpu()
        assert device.type == DeviceType.CPU
        assert device.index is None
        assert device.is_cpu
        assert not device.is_cuda
        assert not device.is_mps
        assert not device.is_gpu
    
    def test_cuda_device_creation(self):
        """Test CUDA device creation."""
        device = Device.cuda()
        assert device.type == DeviceType.CUDA
        assert device.index == 0
        assert not device.is_cpu
        assert device.is_cuda
        assert not device.is_mps
        assert device.is_gpu
        
        # With specific index
        device = Device.cuda(2)
        assert device.index == 2
    
    def test_mps_device_creation(self):
        """Test MPS device creation."""
        device = Device.mps()
        assert device.type == DeviceType.MPS
        assert device.index == 0
        assert not device.is_cpu
        assert not device.is_cuda
        assert device.is_mps
        assert device.is_gpu
        
        # With specific index
        device = Device.mps(1)
        assert device.index == 1
    
    def test_device_validation(self):
        """Test device validation."""
        # Valid devices
        Device(DeviceType.CPU)
        Device(DeviceType.CUDA, 0)
        Device(DeviceType.MPS, 0)
        
        # Invalid index
        with pytest.raises(ValueError, match="non-negative"):
            Device(DeviceType.CUDA, -1)
        
        with pytest.raises(ValueError, match="non-negative"):
            Device(DeviceType.MPS, -1)
    
    def test_device_from_string(self):
        """Test creating device from string."""
        # CPU
        device = Device.from_string("cpu")
        assert device.type == DeviceType.CPU
        assert device.index is None
        
        # CUDA without index
        device = Device.from_string("cuda")
        assert device.type == DeviceType.CUDA
        assert device.index == 0
        
        # CUDA with index
        device = Device.from_string("cuda:2")
        assert device.type == DeviceType.CUDA
        assert device.index == 2
        
        # MPS without index
        device = Device.from_string("mps")
        assert device.type == DeviceType.MPS
        assert device.index == 0
        
        # MPS with index
        device = Device.from_string("mps:1")
        assert device.type == DeviceType.MPS
        assert device.index == 1
    
    def test_device_from_string_errors(self):
        """Test errors in device string parsing."""
        # Invalid device type
        with pytest.raises(ValueError, match="Unsupported device"):
            Device.from_string("gpu")
        
        # Invalid CUDA format
        with pytest.raises(ValueError, match="Invalid CUDA device"):
            Device.from_string("cuda:abc")
        
        # Invalid MPS format
        with pytest.raises(ValueError, match="Invalid MPS device"):
            Device.from_string("mps:xyz")
    
    def test_device_string_repr(self):
        """Test device string representation."""
        assert str(Device.cpu()) == "cpu"
        assert str(Device.cuda()) == "cuda:0"
        assert str(Device.cuda(2)) == "cuda:2"
        assert str(Device.mps()) == "mps:0"
        assert str(Device.mps(1)) == "mps:1"
        
        # Test repr
        assert repr(Device.cpu()) == "Device('cpu', None)"
        assert repr(Device.cuda(1)) == "Device('cuda', 1)"
        assert repr(Device.mps(0)) == "Device('mps', 0)"
    
    def test_device_equality(self):
        """Test device equality comparison."""
        # Same devices
        assert Device.cpu() == Device.cpu()
        assert Device.cuda(0) == Device.cuda(0)
        assert Device.mps(1) == Device.mps(1)
        
        # Different devices
        assert Device.cpu() != Device.cuda()
        assert Device.cuda(0) != Device.cuda(1)
        assert Device.mps() != Device.cpu()
    
    def test_device_hash(self):
        """Test device hashing for use in sets/dicts."""
        devices = {Device.cpu(), Device.cuda(0), Device.mps()}
        assert len(devices) == 3
        
        # Same device should have same hash
        assert hash(Device.cpu()) == hash(Device.cpu())
        assert hash(Device.cuda(1)) == hash(Device.cuda(1))


class TestDeviceGlobals:
    """Test global device functions."""
    
    def setup_method(self):
        """Reset device before each test."""
        # Reset to CPU default
        set_default_device(Device.cpu())
    
    def test_default_device(self):
        """Test default device is CPU."""
        device = get_default_device()
        assert device.type == DeviceType.CPU
        assert device.index is None
    
    def test_set_default_device_with_device(self):
        """Test setting default device with Device object."""
        cuda_device = Device.cuda(1)
        set_default_device(cuda_device)
        
        assert get_default_device() == cuda_device
    
    def test_set_default_device_with_string(self):
        """Test setting default device with string."""
        set_default_device("cuda:2")
        
        device = get_default_device()
        assert device.type == DeviceType.CUDA
        assert device.index == 2
    
    def test_set_default_device_errors(self):
        """Test errors in setting default device."""
        with pytest.raises(TypeError, match="Expected Device or str"):
            set_default_device(123)
        
        with pytest.raises(ValueError, match="Unsupported device"):
            set_default_device("invalid")
    
    def test_reset_default_device(self):
        """Test resetting default device."""
        set_default_device("cuda:1")
        # Reset by setting back to CPU
        set_default_device(Device.cpu())
        
        device = get_default_device()
        assert device.type == DeviceType.CPU
        assert device.index is None


class TestDeviceCapabilities:
    """Test device capabilities detection."""
    
    def test_cpu_capabilities(self):
        """Test CPU is always available."""
        caps = get_device_capabilities()
        
        assert "cpu" in caps
        assert caps["cpu"]["available"] is True
        assert caps["cpu"]["architecture"] == platform.machine()
    
    @patch('neural_arch.core.device.cp')
    def test_cuda_capabilities_available(self, mock_cp):
        """Test CUDA capabilities when available."""
        # Mock CUDA availability
        mock_cp.cuda.runtime.getDeviceCount.return_value = 2
        
        mock_device = MagicMock()
        mock_props = {
            "name": b"NVIDIA GeForce RTX 3090",
            "totalGlobalMem": 24576 * 1024 * 1024,  # 24GB
            "major": 8,
            "minor": 6
        }
        mock_cp.cuda.runtime.getDeviceProperties.return_value = mock_props
        mock_cp.cuda.Device.return_value.__enter__.return_value = mock_device
        
        caps = get_device_capabilities()
        
        assert caps["cuda"]["available"] is True
        assert len(caps["cuda"]["devices"]) == 2
        assert caps["cuda"]["devices"][0]["name"] == "NVIDIA GeForce RTX 3090"
        assert caps["cuda"]["devices"][0]["compute_capability"] == "8.6"
    
    def test_cuda_capabilities_unavailable(self):
        """Test CUDA capabilities when not available."""
        # CUDA import will fail, so it's marked unavailable
        caps = get_device_capabilities()
        
        assert caps["cuda"]["available"] is False
        assert caps["cuda"]["devices"] == []
    
    @patch('neural_arch.core.device.sys')
    @patch('neural_arch.core.device.platform')
    @patch('neural_arch.core.device.mx')
    def test_mps_capabilities_available(self, mock_mx, mock_platform, mock_sys):
        """Test MPS capabilities on Apple Silicon."""
        # Mock Apple Silicon environment
        mock_sys.platform = "darwin"
        mock_platform.machine.return_value = "arm64"
        
        # Mock MLX availability
        mock_mx.array.return_value = MagicMock()
        
        caps = get_device_capabilities()
        
        assert caps["mps"]["available"] is True
        assert caps["mps"]["unified_memory"] is True
    
    @patch('neural_arch.core.device.sys')
    @patch('neural_arch.core.device.platform')
    def test_mps_capabilities_unavailable_intel(self, mock_platform, mock_sys):
        """Test MPS not available on Intel Mac."""
        # Mock Intel Mac
        mock_sys.platform = "darwin"
        mock_platform.machine.return_value = "x86_64"
        
        caps = get_device_capabilities()
        
        assert caps["mps"]["available"] is False
    
    @patch('neural_arch.core.device.sys')
    def test_mps_capabilities_unavailable_linux(self, mock_sys):
        """Test MPS not available on Linux."""
        mock_sys.platform = "linux"
        
        caps = get_device_capabilities()
        
        assert caps["mps"]["available"] is False