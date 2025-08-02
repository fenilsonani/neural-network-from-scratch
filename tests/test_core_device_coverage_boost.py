"""Comprehensive test coverage for core/device module to boost coverage from 87.41% to 95%+"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import platform
from unittest.mock import patch, MagicMock, Mock

from neural_arch.core.device import (
    DeviceType, Device, 
    get_default_device, set_default_device, get_device_capabilities
)


class TestDeviceTypeCoverageBoost:
    """Comprehensive tests for DeviceType enum."""
    
    def test_device_type_values(self):
        """Test DeviceType enum values."""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"
    
    def test_device_type_string_representation(self):
        """Test DeviceType string representation."""
        assert str(DeviceType.CPU) == "cpu"
        assert str(DeviceType.CUDA) == "cuda"
        assert str(DeviceType.MPS) == "mps"
    
    def test_device_type_enumeration(self):
        """Test DeviceType enumeration functionality."""
        device_types = list(DeviceType)
        assert len(device_types) == 3
        assert DeviceType.CPU in device_types
        assert DeviceType.CUDA in device_types
        assert DeviceType.MPS in device_types


class TestDeviceCoverageBoost:
    """Comprehensive tests for Device class targeting missing coverage paths."""
    
    def test_device_creation_basic(self):
        """Test basic device creation."""
        # CPU device
        cpu_device = Device(DeviceType.CPU)
        assert cpu_device.type == DeviceType.CPU
        assert cpu_device.index is None
        
        # CUDA device with index
        cuda_device = Device(DeviceType.CUDA, 1)
        assert cuda_device.type == DeviceType.CUDA
        assert cuda_device.index == 1
        
        # MPS device with index
        mps_device = Device(DeviceType.MPS, 0)
        assert mps_device.type == DeviceType.MPS
        assert mps_device.index == 0
    
    def test_device_post_init_validation(self):
        """Test device validation in __post_init__."""
        # Valid devices should not raise
        Device(DeviceType.CPU, None)
        Device(DeviceType.CUDA, 0)
        Device(DeviceType.MPS, 1)
        Device(DeviceType.CUDA, 5)
        
        # Invalid device index should raise
        with pytest.raises(ValueError, match="Device index must be non-negative"):
            Device(DeviceType.CUDA, -1)
        
        with pytest.raises(ValueError, match="Device index must be non-negative"):
            Device(DeviceType.MPS, -5)
        
        # CPU with index should not validate index (CPU doesn't use index)
        Device(DeviceType.CPU, 0)  # Should not raise
        Device(DeviceType.CPU, -1)  # Should not raise for CPU
    
    def test_device_cpu_class_method(self):
        """Test Device.cpu() class method."""
        cpu_device = Device.cpu()
        assert cpu_device.type == DeviceType.CPU
        assert cpu_device.index is None
        assert cpu_device.is_cpu is True
    
    def test_device_cuda_class_method(self):
        """Test Device.cuda() class method."""
        # Default index
        cuda_device = Device.cuda()
        assert cuda_device.type == DeviceType.CUDA
        assert cuda_device.index == 0
        assert cuda_device.is_cuda is True
        
        # Specific index
        cuda_device_1 = Device.cuda(1)
        assert cuda_device_1.type == DeviceType.CUDA
        assert cuda_device_1.index == 1
        
        # Negative index should fail in validation
        with pytest.raises(ValueError):
            Device.cuda(-1)
    
    def test_device_mps_class_method(self):
        """Test Device.mps() class method."""
        # Default index
        mps_device = Device.mps()
        assert mps_device.type == DeviceType.MPS
        assert mps_device.index == 0
        assert mps_device.is_mps is True
        
        # Specific index
        mps_device_1 = Device.mps(1)
        assert mps_device_1.type == DeviceType.MPS
        assert mps_device_1.index == 1
        
        # Negative index should fail in validation
        with pytest.raises(ValueError):
            Device.mps(-2)
    
    def test_device_from_string_cpu(self):
        """Test Device.from_string() with CPU strings."""
        # Basic CPU
        cpu_device = Device.from_string("cpu")
        assert cpu_device.type == DeviceType.CPU
        assert cpu_device.index is None
        
        # Case insensitive
        cpu_device_upper = Device.from_string("CPU")
        assert cpu_device_upper.type == DeviceType.CPU
        
        # With whitespace
        cpu_device_ws = Device.from_string("  cpu  ")
        assert cpu_device_ws.type == DeviceType.CPU
    
    def test_device_from_string_cuda(self):
        """Test Device.from_string() with CUDA strings."""
        # Basic CUDA
        cuda_device = Device.from_string("cuda")
        assert cuda_device.type == DeviceType.CUDA
        assert cuda_device.index == 0
        
        # CUDA with index
        cuda_device_1 = Device.from_string("cuda:1")
        assert cuda_device_1.type == DeviceType.CUDA
        assert cuda_device_1.index == 1
        
        # CUDA with higher index
        cuda_device_7 = Device.from_string("cuda:7")
        assert cuda_device_7.type == DeviceType.CUDA
        assert cuda_device_7.index == 7
        
        # Case insensitive
        cuda_device_upper = Device.from_string("CUDA:2")
        assert cuda_device_upper.type == DeviceType.CUDA
        assert cuda_device_upper.index == 2
    
    def test_device_from_string_cuda_errors(self):
        """Test Device.from_string() CUDA error cases."""
        # Invalid CUDA format
        with pytest.raises(ValueError, match="Invalid CUDA device string"):
            Device.from_string("cuda:")
        
        with pytest.raises(ValueError, match="Invalid CUDA device string"):
            Device.from_string("cuda:abc")
        
        with pytest.raises(ValueError, match="Invalid CUDA device string"):
            Device.from_string("cuda:1:2")
        
        # Negative index should fail in Device validation
        with pytest.raises(ValueError):
            Device.from_string("cuda:-1")
    
    def test_device_from_string_mps(self):
        """Test Device.from_string() with MPS strings."""
        # Basic MPS
        mps_device = Device.from_string("mps")
        assert mps_device.type == DeviceType.MPS
        assert mps_device.index == 0
        
        # MPS with index
        mps_device_1 = Device.from_string("mps:1")
        assert mps_device_1.type == DeviceType.MPS
        assert mps_device_1.index == 1
        
        # Case insensitive
        mps_device_upper = Device.from_string("MPS:0")
        assert mps_device_upper.type == DeviceType.MPS
        assert mps_device_upper.index == 0
    
    def test_device_from_string_mps_errors(self):
        """Test Device.from_string() MPS error cases."""
        # Invalid MPS format
        with pytest.raises(ValueError, match="Invalid MPS device string"):
            Device.from_string("mps:")
        
        with pytest.raises(ValueError, match="Invalid MPS device string"):
            Device.from_string("mps:xyz")
        
        # Negative index should fail in Device validation
        with pytest.raises(ValueError):
            Device.from_string("mps:-1")
    
    def test_device_from_string_unsupported(self):
        """Test Device.from_string() with unsupported strings."""
        with pytest.raises(ValueError, match="Unsupported device string"):
            Device.from_string("gpu")
        
        with pytest.raises(ValueError, match="Unsupported device string"):
            Device.from_string("tpu")
        
        with pytest.raises(ValueError, match="Unsupported device string"):
            Device.from_string("")
        
        with pytest.raises(ValueError, match="Unsupported device string"):
            Device.from_string("invalid_device")
    
    def test_device_property_methods(self):
        """Test device property methods."""
        # CPU device
        cpu_device = Device.cpu()
        assert cpu_device.is_cpu is True
        assert cpu_device.is_cuda is False
        assert cpu_device.is_mps is False
        assert cpu_device.is_gpu is False
        
        # CUDA device
        cuda_device = Device.cuda(0)
        assert cuda_device.is_cpu is False
        assert cuda_device.is_cuda is True
        assert cuda_device.is_mps is False
        assert cuda_device.is_gpu is True
        
        # MPS device
        mps_device = Device.mps(0)
        assert mps_device.is_cpu is False
        assert mps_device.is_cuda is False
        assert mps_device.is_mps is True
        assert mps_device.is_gpu is True
    
    def test_device_string_representation(self):
        """Test device string representation."""
        # CPU device
        cpu_device = Device.cpu()
        assert str(cpu_device) == "cpu"
        
        # CUDA device without index
        cuda_device = Device(DeviceType.CUDA)
        assert str(cuda_device) == "cuda"
        
        # CUDA device with index
        cuda_device_1 = Device.cuda(1)
        assert str(cuda_device_1) == "cuda:1"
        
        # MPS device without index
        mps_device = Device(DeviceType.MPS)
        assert str(mps_device) == "mps"
        
        # MPS device with index
        mps_device_2 = Device.mps(2)
        assert str(mps_device_2) == "mps:2"
    
    def test_device_repr_representation(self):
        """Test device __repr__ representation."""
        cpu_device = Device.cpu()
        assert repr(cpu_device) == "Device('cpu', None)"
        
        cuda_device = Device.cuda(1)
        assert repr(cuda_device) == "Device('cuda', 1)"
        
        mps_device = Device.mps(0)
        assert repr(mps_device) == "Device('mps', 0)"
    
    def test_device_equality_and_hashing(self):
        """Test device equality and hashing (dataclass features)."""
        # Same devices should be equal
        cpu1 = Device.cpu()
        cpu2 = Device.cpu()
        assert cpu1 == cpu2
        assert hash(cpu1) == hash(cpu2)
        
        # Different devices should not be equal
        cuda1 = Device.cuda(0)
        cuda2 = Device.cuda(1)
        assert cuda1 != cuda2
        assert hash(cuda1) != hash(cuda2)
        
        # Different types should not be equal
        cpu = Device.cpu()
        cuda = Device.cuda(0)
        assert cpu != cuda
    
    def test_device_immutability(self):
        """Test device immutability (frozen dataclass)."""
        device = Device.cuda(0)
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            device.type = DeviceType.CPU
        
        with pytest.raises(AttributeError):
            device.index = 1


class TestDeviceGlobalFunctionsCoverageBoost:
    """Test global device management functions."""
    
    def test_get_default_device(self):
        """Test get_default_device function."""
        default_device = get_default_device()
        assert isinstance(default_device, Device)
        # Should be CPU by default
        assert default_device.is_cpu is True
    
    def test_set_default_device_with_device_object(self):
        """Test set_default_device with Device object."""
        original_device = get_default_device()
        
        try:
            # Set to CUDA device
            cuda_device = Device.cuda(1)
            set_default_device(cuda_device)
            
            current_device = get_default_device()
            assert current_device == cuda_device
            assert current_device.is_cuda is True
            assert current_device.index == 1
            
        finally:
            # Restore original
            set_default_device(original_device)
    
    def test_set_default_device_with_string(self):
        """Test set_default_device with string."""
        original_device = get_default_device()
        
        try:
            # Set to MPS device via string
            set_default_device("mps:0")
            
            current_device = get_default_device()
            assert current_device.is_mps is True
            assert current_device.index == 0
            
            # Set to CPU via string
            set_default_device("cpu")
            
            current_device = get_default_device()
            assert current_device.is_cpu is True
            
        finally:
            # Restore original
            set_default_device(original_device)
    
    def test_set_default_device_invalid_type(self):
        """Test set_default_device with invalid types."""
        with pytest.raises(TypeError, match="Expected Device or str"):
            set_default_device(123)
        
        with pytest.raises(TypeError, match="Expected Device or str"):
            set_default_device(None)
        
        with pytest.raises(TypeError, match="Expected Device or str"):
            set_default_device(["cuda"])
    
    def test_set_default_device_invalid_string(self):
        """Test set_default_device with invalid string."""
        with pytest.raises(ValueError, match="Unsupported device string"):
            set_default_device("invalid_device")


class TestDeviceCapabilitiesCoverageBoost:
    """Test get_device_capabilities function."""
    
    def test_get_device_capabilities_basic_structure(self):
        """Test basic structure of device capabilities."""
        capabilities = get_device_capabilities()
        
        # Should have all expected keys
        assert "cpu" in capabilities
        assert "cuda" in capabilities
        assert "mps" in capabilities
        
        # CPU should always be available
        assert capabilities["cpu"]["available"] is True
        assert "architecture" in capabilities["cpu"]
        assert "cores" in capabilities["cpu"]
        
        # CUDA and MPS structure
        assert "available" in capabilities["cuda"]
        assert "devices" in capabilities["cuda"]
        assert "available" in capabilities["mps"]
        assert "unified_memory" in capabilities["mps"]
    
    def test_get_device_capabilities_cpu_info(self):
        """Test CPU capabilities detection."""
        capabilities = get_device_capabilities()
        
        cpu_info = capabilities["cpu"]
        assert cpu_info["available"] is True
        assert cpu_info["architecture"] == platform.machine()
        # cores might be "unknown" without psutil
        assert "cores" in cpu_info
    
    @patch('neural_arch.core.device.sys.platform', 'linux')
    def test_get_device_capabilities_non_darwin(self):
        """Test device capabilities on non-Darwin platforms."""
        # On non-Darwin platforms, MPS should not be available
        capabilities = get_device_capabilities()
        
        # MPS should not be available on non-Darwin
        assert capabilities["mps"]["available"] is False
    
    @patch('neural_arch.core.device.sys.platform', 'darwin')
    @patch('neural_arch.core.device.platform.machine', return_value='arm64')
    def test_get_device_capabilities_darwin_arm64_mlx_available(self):
        """Test device capabilities on Darwin ARM64 with MLX available."""
        mock_mx = MagicMock()
        mock_mx.array.return_value = "test_array"
        
        with patch.dict('sys.modules', {'mlx.core': mock_mx}):
            capabilities = get_device_capabilities()
            
            # MPS should be available on Darwin ARM64 with MLX
            assert capabilities["mps"]["available"] is True
            assert capabilities["mps"]["unified_memory"] is True
    
    @patch('neural_arch.core.device.sys.platform', 'darwin')
    @patch('neural_arch.core.device.platform.machine', return_value='arm64')
    def test_get_device_capabilities_darwin_arm64_mlx_unavailable(self):
        """Test device capabilities on Darwin ARM64 without MLX."""
        # Mock MLX import failure
        def mock_import(name, *args, **kwargs):
            if name == 'mlx.core':
                raise ImportError("MLX not available")
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            capabilities = get_device_capabilities()
            
            # MPS should not be available without MLX
            assert capabilities["mps"]["available"] is False
    
    @patch('neural_arch.core.device.sys.platform', 'darwin')
    @patch('neural_arch.core.device.platform.machine', return_value='x86_64')
    def test_get_device_capabilities_darwin_x86_64(self):
        """Test device capabilities on Darwin x86_64 (Intel Mac)."""
        capabilities = get_device_capabilities()
        
        # MPS should not be available on Intel Macs
        assert capabilities["mps"]["available"] is False
    
    def test_get_device_capabilities_cuda_available(self):
        """Test device capabilities with CUDA available."""
        mock_cp = MagicMock()
        mock_cp.cuda.runtime.getDeviceCount.return_value = 2
        
        # Mock device properties
        mock_props = {
            "name": b"GeForce RTX 3080",
            "totalGlobalMem": 10737418240,  # 10GB
            "major": 8,
            "minor": 6
        }
        mock_cp.cuda.runtime.getDeviceProperties.return_value = mock_props
        mock_cp.cuda.Device.return_value.__enter__ = Mock()
        mock_cp.cuda.Device.return_value.__exit__ = Mock()
        
        with patch.dict('sys.modules', {'cupy': mock_cp}):
            capabilities = get_device_capabilities()
            
            # CUDA should be available
            assert capabilities["cuda"]["available"] is True
            assert len(capabilities["cuda"]["devices"]) == 2
            
            # Check device info
            device_info = capabilities["cuda"]["devices"][0]
            assert device_info["index"] == 0
            assert device_info["name"] == "GeForce RTX 3080"
            assert device_info["memory"] == 10737418240
            assert device_info["compute_capability"] == "8.6"
    
    def test_get_device_capabilities_cuda_unavailable(self):
        """Test device capabilities with CUDA unavailable."""
        # Mock CuPy import failure
        def mock_import(name, *args, **kwargs):
            if name == 'cupy':
                raise ImportError("CuPy not available")
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            capabilities = get_device_capabilities()
            
            # CUDA should not be available
            assert capabilities["cuda"]["available"] is False
            assert capabilities["cuda"]["devices"] == []
    
    def test_get_device_capabilities_cuda_exception(self):
        """Test device capabilities when CUDA throws exception."""
        mock_cp = MagicMock()
        mock_cp.cuda.runtime.getDeviceCount.side_effect = Exception("CUDA error")
        
        with patch.dict('sys.modules', {'cupy': mock_cp}):
            capabilities = get_device_capabilities()
            
            # Should handle exception gracefully
            assert capabilities["cuda"]["available"] is False
    
    def test_get_device_capabilities_mlx_exception(self):
        """Test device capabilities when MLX throws exception."""
        mock_mx = MagicMock()
        mock_mx.array.side_effect = Exception("MLX error")
        
        with patch('neural_arch.core.device.sys.platform', 'darwin'):
            with patch('neural_arch.core.device.platform.machine', return_value='arm64'):
                with patch.dict('sys.modules', {'mlx.core': mock_mx}):
                    capabilities = get_device_capabilities()
                    
                    # Should handle exception gracefully
                    assert capabilities["mps"]["available"] is False
    
    def test_device_capabilities_complete_coverage(self):
        """Test comprehensive device capabilities scenarios."""
        # Test that the function doesn't crash with any platform combination
        original_platform = getattr(platform, 'machine', lambda: 'unknown')
        
        try:
            # Test various machine types
            for machine_type in ['x86_64', 'arm64', 'aarch64', 'i386', 'unknown']:
                with patch('neural_arch.core.device.platform.machine', return_value=machine_type):
                    capabilities = get_device_capabilities()
                    
                    # Basic structure should always be present
                    assert isinstance(capabilities, dict)
                    assert "cpu" in capabilities
                    assert "cuda" in capabilities
                    assert "mps" in capabilities
                    
        finally:
            # Restore original
            platform.machine = original_platform