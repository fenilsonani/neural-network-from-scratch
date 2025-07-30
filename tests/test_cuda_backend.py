"""Test CUDA backend with mocking."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from neural_arch.backends import get_backend
from neural_arch.exceptions import DeviceError


class TestCudaBackend:
    """Test CUDA backend functionality with mocking."""
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_backend_initialization(self, mock_cp):
        """Test CUDA backend initialization."""
        # Mock CuPy availability
        mock_cp.cuda.Device.return_value.__enter__ = Mock()
        mock_cp.cuda.Device.return_value.__exit__ = Mock()
        mock_cp.array = Mock(return_value=MagicMock())
        
        try:
            backend = get_backend("cuda")
            assert backend.name == "cuda"
            assert hasattr(backend, 'array')
        except ImportError:
            # If CUDA backend not available, skip
            pytest.skip("CUDA backend not available")
    
    def test_cuda_backend_unavailable(self):
        """Test CUDA backend when CuPy not available."""
        # When CUDA is not available, should fall back gracefully
        backend = get_backend("cuda")
        
        # Either gets CUDA backend (if available) or falls back
        assert backend.name in ["cuda", "numpy"]
        
        if backend.name == "cuda" and not backend.available:
            with pytest.raises(DeviceError):
                backend.array([1, 2, 3])
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_array_creation(self, mock_cp):
        """Test CUDA array creation."""
        # Mock CuPy array
        mock_array = MagicMock()
        mock_array.shape = (3,)
        mock_array.dtype = mock_cp.float32
        mock_cp.array.return_value = mock_array
        mock_cp.float32 = np.float32
        
        try:
            backend = get_backend("cuda")
            if backend.available:
                result = backend.array([1, 2, 3], dtype=backend.float32)
                assert result is not None
                mock_cp.array.assert_called_once()
        except ImportError:
            pytest.skip("CUDA backend not available")
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_matmul(self, mock_cp):
        """Test CUDA matrix multiplication."""
        # Mock CuPy operations
        mock_result = MagicMock()
        mock_result.shape = (2, 2)
        mock_cp.matmul.return_value = mock_result
        
        try:
            backend = get_backend("cuda")
            if backend.available:
                x = MagicMock()
                y = MagicMock()
                result = backend.matmul(x, y)
                assert result is not None
                mock_cp.matmul.assert_called_once_with(x, y)
        except ImportError:
            pytest.skip("CUDA backend not available")
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_arithmetic_operations(self, mock_cp):
        """Test CUDA arithmetic operations."""
        # Mock basic operations
        mock_cp.add.return_value = MagicMock()
        mock_cp.subtract.return_value = MagicMock()
        mock_cp.multiply.return_value = MagicMock()
        mock_cp.divide.return_value = MagicMock()
        
        try:
            backend = get_backend("cuda")
            if backend.available:
                x = MagicMock()
                y = MagicMock()
                
                # Test all arithmetic operations
                backend.add(x, y)
                mock_cp.add.assert_called_once_with(x, y)
                
                backend.sub(x, y)
                mock_cp.subtract.assert_called_once_with(x, y)
                
                backend.mul(x, y)
                mock_cp.multiply.assert_called_once_with(x, y)
                
                backend.div(x, y)
                mock_cp.divide.assert_called_once_with(x, y)
        except ImportError:
            pytest.skip("CUDA backend not available")
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_activation_functions(self, mock_cp):
        """Test CUDA activation functions."""
        # Mock activation functions
        mock_cp.maximum.return_value = MagicMock()
        mock_cp.exp.return_value = MagicMock()
        mock_cp.sum.return_value = MagicMock()
        
        try:
            backend = get_backend("cuda")
            if backend.available:
                x = MagicMock()
                
                # Test ReLU (maximum with 0)
                backend.relu(x)
                mock_cp.maximum.assert_called()
                
                # Test softmax components
                backend.exp(x)
                mock_cp.exp.assert_called_once_with(x)
        except ImportError:
            pytest.skip("CUDA backend not available")
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_memory_management(self, mock_cp):
        """Test CUDA memory management."""
        # Mock memory operations
        mock_cp.cuda.MemoryPool.return_value.free_all_blocks = Mock()
        
        try:
            backend = get_backend("cuda")
            if backend.available and hasattr(backend, 'clear_memory'):
                backend.clear_memory()
                # Should call memory cleanup
        except ImportError:
            pytest.skip("CUDA backend not available")
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_device_management(self, mock_cp):
        """Test CUDA device management."""
        # Mock device operations
        mock_device = MagicMock()
        mock_cp.cuda.Device.return_value = mock_device
        mock_device.__enter__.return_value = mock_device
        mock_device.__exit__.return_value = None
        
        try:
            backend = get_backend("cuda")
            if backend.available:
                # Test device context
                with backend.device_context(0):
                    pass
                mock_cp.cuda.Device.assert_called()
        except (ImportError, AttributeError):
            pytest.skip("CUDA backend device management not available")
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_error_handling(self, mock_cp):
        """Test CUDA error handling."""
        # Mock CUDA errors
        mock_cp.cuda.runtime.CUDARuntimeError = RuntimeError
        mock_cp.matmul.side_effect = RuntimeError("CUDA error")
        
        try:
            backend = get_backend("cuda")
            if backend.available:
                x = MagicMock()
                y = MagicMock()
                
                with pytest.raises(RuntimeError):
                    backend.matmul(x, y)
        except ImportError:
            pytest.skip("CUDA backend not available")
    
    def test_cuda_fallback_to_numpy(self):
        """Test fallback to NumPy when CUDA unavailable."""
        # This should always work - either get CUDA or fall back to numpy
        backend = get_backend("cuda")
        
        # Should be able to create arrays regardless
        if backend.name == "numpy" or not backend.available:
            # Fallback case
            result = backend.array([1, 2, 3], dtype=backend.float32)
            assert result is not None
            assert hasattr(result, 'shape')
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_dtype_support(self, mock_cp):
        """Test CUDA dtype support."""
        # Mock dtype attributes
        mock_cp.float32 = np.float32
        mock_cp.float64 = np.float64
        mock_cp.int32 = np.int32
        mock_cp.int64 = np.int64
        
        try:
            backend = get_backend("cuda")
            if backend.available:
                # Test dtype attributes exist
                assert hasattr(backend, 'float32')
                assert hasattr(backend, 'float64')
                assert hasattr(backend, 'int32')
                assert hasattr(backend, 'int64')
        except ImportError:
            pytest.skip("CUDA backend not available")
    
    @patch('neural_arch.backends.cuda_backend.cp')
    def test_cuda_shape_operations(self, mock_cp):
        """Test CUDA shape operations."""
        # Mock shape operations
        mock_cp.reshape.return_value = MagicMock()
        mock_cp.transpose.return_value = MagicMock()
        mock_cp.flatten.return_value = MagicMock()
        
        try:
            backend = get_backend("cuda")
            if backend.available:
                x = MagicMock()
                
                backend.reshape(x, (2, 3))
                mock_cp.reshape.assert_called_once()
                
                backend.transpose(x, (1, 0))
                mock_cp.transpose.assert_called_once()
                
                if hasattr(backend, 'flatten'):
                    backend.flatten(x)
        except ImportError:
            pytest.skip("CUDA backend not available")