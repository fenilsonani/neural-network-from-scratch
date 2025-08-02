"""Complete test coverage for functional/activation.py targeting 95%+ coverage."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock

from neural_arch.core.tensor import Tensor, GradientFunction
from neural_arch.functional.activation import (
    relu, softmax, sigmoid, tanh, gelu, mish, silu, swiglu, leaky_relu, 
    swish, glu, reglu, geglu
)
from neural_arch.core.device import Device, DeviceType


class TestActivationFunctionsComplete:
    """Complete test coverage for all activation functions."""
    
    def test_relu_with_backend_optimization(self):
        """Test ReLU with backend-optimized implementation."""
        x = Tensor([[1.0, -2.0, 3.0]], requires_grad=True)
        
        # Mock backend with relu method
        mock_backend = MagicMock()
        mock_backend.relu.return_value = np.array([[1.0, 0.0, 3.0]])
        mock_backend.maximum.return_value = np.array([[1.0, 0.0, 3.0]])
        x.backend = mock_backend
        x.backend_data = x.data
        
        result = relu(x)
        
        # Backend relu should be called
        mock_backend.relu.assert_called_once_with(x.data)
        assert np.allclose(result.data, [[1.0, 0.0, 3.0]])
    
    def test_relu_fallback_implementation(self):
        """Test ReLU fallback when backend optimization fails."""
        x = Tensor([[-1.0, 2.0, -3.0]], requires_grad=True)
        
        # Mock backend without relu method
        mock_backend = MagicMock()
        del mock_backend.relu  # Remove relu method
        mock_backend.array.return_value = 0
        mock_backend.maximum.return_value = np.array([[0.0, 2.0, 0.0]])
        x.backend = mock_backend
        x.backend_data = x.data
        
        result = relu(x)
        
        # Fallback maximum should be called
        mock_backend.maximum.assert_called_once()
        assert np.allclose(result.data, [[0.0, 2.0, 0.0]])
    
    def test_relu_gradient_with_cupy_backend(self):
        """Test ReLU gradient computation with CuPy-like backend."""
        x = Tensor([[-1.0, 2.0]], requires_grad=True)
        
        # Mock CuPy-like backend data
        mock_backend_data = MagicMock()
        mock_backend_data.get.return_value = x.data
        x.backend_data = mock_backend_data
        
        result = relu(x)
        
        # Simulate backward pass
        grad_output = np.array([[1.0, 1.0]])
        result._grad_fn.apply(grad_output)
        
        # Should handle CuPy backend data conversion
        mock_backend_data.get.assert_called()
    
    def test_relu_gradient_with_backward_chaining(self):
        """Test ReLU gradient with backward chaining."""
        x = Tensor([[1.0, -1.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = relu(x)
        
        # Simulate backward pass
        grad_output = np.array([[1.0, 1.0]])
        result._grad_fn.apply(grad_output)
        
        # Backward chaining should be called
        x._backward.assert_called_once()
    
    def test_softmax_with_backend_optimization(self):
        """Test softmax with backend-optimized implementation."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        
        # Mock backend with softmax method
        mock_backend = MagicMock()
        softmax_result = np.array([[0.09003057, 0.24472847, 0.66524096]])
        mock_backend.softmax.return_value = softmax_result
        x.backend = mock_backend
        x.backend_data = x.data
        
        result = softmax(x)
        
        # Backend softmax should be called
        mock_backend.softmax.assert_called_once_with(x.data, axis=-1)
        assert np.allclose(result.data, softmax_result)
    
    def test_softmax_backend_fallback(self):
        """Test softmax fallback when backend optimization fails."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        
        # Mock backend with failing softmax method
        mock_backend = MagicMock()
        mock_backend.softmax.side_effect = Exception("Backend error")
        mock_backend.max.return_value = np.array([[3.0]])
        mock_backend.exp.return_value = np.array([[0.36787944, 1.0, 2.71828183]])
        mock_backend.sum.return_value = np.array([[4.08616127]])
        mock_backend.maximum.return_value = np.array([[4.08616127]])
        x.backend = mock_backend
        x.backend_data = x.data
        
        result = softmax(x)
        
        # Fallback implementation should be used
        mock_backend.max.assert_called()
        mock_backend.exp.assert_called()
        mock_backend.sum.assert_called()
    
    def test_softmax_gradient_with_cupy_backend(self):
        """Test softmax gradient computation with CuPy-like backend."""
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        result = softmax(x)
        
        # Mock result data with CuPy-like interface
        mock_result_data = MagicMock()
        mock_result_data.get.return_value = result.data
        result.backend_data = mock_result_data
        
        # Simulate backward pass
        grad_output = np.array([[1.0, 1.0]])
        result._grad_fn.apply(grad_output)
        
        # Should handle CuPy backend data conversion
        mock_result_data.get.assert_called()
    
    def test_sigmoid_with_backend_optimization(self):
        """Test sigmoid with backend-optimized implementation."""
        x = Tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        
        # Mock backend with sigmoid method
        mock_backend = MagicMock()
        sigmoid_result = np.array([[0.5, 0.73105858, 0.26894142]])
        mock_backend.sigmoid.return_value = sigmoid_result
        x.backend = mock_backend
        x.backend_data = x.data
        
        result = sigmoid(x)
        
        # Backend sigmoid should be called
        mock_backend.sigmoid.assert_called_once_with(x.data)
        assert np.allclose(result.data, sigmoid_result)
    
    def test_sigmoid_backend_fallback(self):
        """Test sigmoid fallback when backend optimization fails."""
        x = Tensor([[0.0, 100.0, -100.0]], requires_grad=True)
        
        # Mock backend with failing sigmoid method
        mock_backend = MagicMock()
        mock_backend.sigmoid.side_effect = Exception("Backend error")
        mock_backend.where.return_value = np.array([[0.5, 1.0, 0.0]])
        mock_backend.exp.side_effect = lambda x: np.exp(x)
        x.backend = mock_backend
        x.backend_data = x.data
        
        result = sigmoid(x)
        
        # Fallback implementation should be used
        mock_backend.where.assert_called()
    
    def test_sigmoid_gradient_with_cupy_backend(self):
        """Test sigmoid gradient computation with CuPy-like backend."""
        x = Tensor([[0.0]], requires_grad=True)
        result = sigmoid(x)
        
        # Mock result data with CuPy-like interface
        mock_result_data = MagicMock()
        mock_result_data.get.return_value = result.data
        result.backend_data = mock_result_data
        
        # Simulate backward pass
        grad_output = np.array([[1.0]])
        result._grad_fn.apply(grad_output)
        
        # Should handle CuPy backend data conversion
        mock_result_data.get.assert_called()
    
    def test_tanh_gradient_with_cupy_backend(self):
        """Test tanh gradient computation with CuPy-like backend."""
        x = Tensor([[0.0]], requires_grad=True)
        result = tanh(x)
        
        # Mock result data with CuPy-like interface
        mock_result_data = MagicMock()
        mock_result_data.get.return_value = result.data
        result.backend_data = mock_result_data
        
        # Simulate backward pass
        grad_output = np.array([[1.0]])
        result._grad_fn.apply(grad_output)
        
        # Should handle CuPy backend data conversion
        mock_result_data.get.assert_called()
    
    def test_gelu_with_backend_optimization(self):
        """Test GELU with backend-optimized implementation."""
        x = Tensor([[0.0, 1.0, -1.0]], requires_grad=True, device=Device(DeviceType.CUDA))
        
        # Mock backend with gelu method
        mock_backend = MagicMock()
        mock_backend.gelu.return_value = np.array([[0.0, 0.8413, -0.1587]])
        mock_backend.array.side_effect = lambda x: np.array(x)
        mock_backend.sqrt.side_effect = lambda x: np.sqrt(x)
        mock_backend.exp.side_effect = lambda x: np.exp(x)
        mock_backend.sign.side_effect = lambda x: np.sign(x)
        mock_backend.abs.side_effect = lambda x: np.abs(x)
        x.backend = mock_backend
        x.backend_data = x.data
        
        result = gelu(x, approximate=False)
        
        # Backend gelu should be called
        mock_backend.gelu.assert_called_once_with(x.data)
        assert np.allclose(result.data, [[0.0, 0.8413, -0.1587]], atol=1e-3)
    
    def test_gelu_backend_optimization_fallback(self):
        """Test GELU backend optimization fallback."""
        x = Tensor([[0.0]], requires_grad=True, device=Device(DeviceType.CUDA))
        
        # Mock backend with failing gelu method
        mock_backend = MagicMock()
        mock_backend.gelu.side_effect = Exception("Backend optimization failed")
        x.backend = mock_backend
        x.backend_data = x.data
        
        # Should log debug message and fall back to standard implementation
        with patch('neural_arch.functional.activation.logger') as mock_logger:
            result = gelu(x, approximate=False)
            mock_logger.debug.assert_called()
    
    def test_gelu_approximate_mode(self):
        """Test GELU approximate mode."""
        x = Tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        
        result = gelu(x, approximate=True)
        
        # Should use tanh approximation
        assert result.requires_grad
        assert "gelu_approx" in result.name
    
    def test_gelu_exact_mode_without_scipy(self):
        """Test GELU exact mode without scipy."""
        x = Tensor([[0.0, 1.0]], requires_grad=True)
        
        # Mock scipy import failure
        with patch.dict('sys.modules', {'scipy.special': None}):
            with patch.dict('sys.modules', {'numpy': MagicMock()}):
                # Mock numpy erf not available
                import sys
                if 'numpy' in sys.modules:
                    del sys.modules['numpy'].erf
                
                result = gelu(x, approximate=False)
                
                # Should use manual erf approximation
                assert result.requires_grad
                assert "gelu_exact" in result.name
    
    def test_gelu_gradient_optimized_with_cupy(self):
        """Test GELU optimized gradient with CuPy backend."""
        x = Tensor([[1.0]], requires_grad=True, device=Device(DeviceType.CUDA))
        
        # Mock backend with gelu method
        mock_backend = MagicMock()
        mock_backend.gelu.return_value = np.array([[0.8413]])
        mock_backend.array.side_effect = lambda x: np.array(x) if not hasattr(x, 'backend') else x
        mock_backend.sqrt.side_effect = lambda x: np.sqrt(x)
        mock_backend.exp.side_effect = lambda x: np.exp(x)
        mock_backend.to_numpy.side_effect = lambda x: x if isinstance(x, np.ndarray) else np.array(x)
        
        # Mock CuPy-like backend data
        mock_backend_data = MagicMock()
        mock_backend_data.get.return_value = x.data
        x.backend_data = mock_backend_data
        x.backend = mock_backend
        
        result = gelu(x, approximate=False)
        
        # Simulate backward pass
        grad_output = np.array([[1.0]])
        result._grad_fn.apply(grad_output)
        
        # Should handle CuPy backend data conversion
        mock_backend_data.get.assert_called()
    
    def test_leaky_relu_comprehensive(self):
        """Test Leaky ReLU with comprehensive cases."""
        # Test with custom negative slope
        x = Tensor([[-2.0, 0.0, 2.0]], requires_grad=True)
        result = leaky_relu(x, negative_slope=0.2)
        
        expected = np.array([[-0.4, 0.0, 2.0]])
        assert np.allclose(result.data, expected)
        assert result.requires_grad
        assert "leaky_relu" in result.name
    
    def test_leaky_relu_gradient(self):
        """Test Leaky ReLU gradient computation."""
        x = Tensor([[-1.0, 0.0, 1.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = leaky_relu(x, negative_slope=0.1)
        
        # Simulate backward pass
        grad_output = np.array([[1.0, 1.0, 1.0]])
        result._grad_fn.apply(grad_output)
        
        # Backward chaining should be called
        x._backward.assert_called_once()
    
    def test_swiglu_invalid_dimension(self):
        """Test SwiGLU with invalid dimension."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # Odd dimension
        
        with pytest.raises(ValueError, match="SwiGLU requires even dimension"):
            swiglu(x)
    
    def test_swiglu_valid_computation(self):
        """Test SwiGLU valid computation."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)  # Even dimension
        
        result = swiglu(x)
        
        # Result should have half the input dimension
        assert result.shape == (1, 2)
        assert result.requires_grad
        assert "swiglu" in result.name
    
    def test_swiglu_gradient(self):
        """Test SwiGLU gradient computation."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = swiglu(x)
        
        # Simulate backward pass
        grad_output = np.array([[1.0, 1.0]])
        result._grad_fn.apply(grad_output)
        
        # Backward chaining should be called
        x._backward.assert_called_once()
    
    def test_mish_computation(self):
        """Test Mish activation computation."""
        x = Tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        
        result = mish(x)
        
        assert result.requires_grad
        assert "mish" in result.name
        assert np.all(np.isfinite(result.data))
    
    def test_mish_gradient(self):
        """Test Mish gradient computation."""
        x = Tensor([[1.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = mish(x)
        
        # Simulate backward pass
        grad_output = np.array([[1.0]])
        result._grad_fn.apply(grad_output)
        
        # Backward chaining should be called
        x._backward.assert_called_once()
    
    def test_silu_computation(self):
        """Test SiLU activation computation."""
        x = Tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        
        result = silu(x)
        
        assert result.requires_grad
        assert "silu" in result.name
        assert np.all(np.isfinite(result.data))
    
    def test_silu_gradient(self):
        """Test SiLU gradient computation."""
        x = Tensor([[1.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = silu(x)
        
        # Simulate backward pass
        grad_output = np.array([[1.0]])
        result._grad_fn.apply(grad_output)
        
        # Backward chaining should be called
        x._backward.assert_called_once()
    
    def test_swish_alias(self):
        """Test that swish is an alias for silu."""
        assert swish is silu
    
    def test_glu_invalid_dimension(self):
        """Test GLU with invalid dimension."""
        x = Tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)  # Odd last dimension
        
        with pytest.raises(ValueError, match="GLU requires even dimension"):
            glu(x)
    
    def test_glu_last_dimension(self):
        """Test GLU with last dimension split."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        result = glu(x, dim=-1)
        
        # Result should have half the input dimension
        assert result.shape == (1, 2)
        assert result.requires_grad
        assert "glu" in result.name
    
    def test_glu_general_dimension(self):
        """Test GLU with general dimension split."""
        x = Tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # Shape: (1, 2, 2)
        
        result = glu(x, dim=1)  # Split along dimension 1
        
        # Result should have half the size along dimension 1
        assert result.shape == (1, 1, 2)
        assert result.requires_grad
    
    def test_glu_gradient(self):
        """Test GLU gradient computation."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = glu(x, dim=-1)
        
        # Simulate backward pass
        grad_output = np.array([[1.0, 1.0]])
        result._grad_fn.apply(grad_output)
        
        # Backward chaining should be called
        x._backward.assert_called_once()
    
    def test_reglu_invalid_dimension(self):
        """Test ReGLU with invalid dimension."""
        x = Tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)  # Odd last dimension
        
        with pytest.raises(ValueError, match="ReGLU requires even dimension"):
            reglu(x)
    
    def test_reglu_computation(self):
        """Test ReGLU computation."""
        x = Tensor([[1.0, -1.0, 2.0, 3.0]], requires_grad=True)
        
        result = reglu(x, dim=-1)
        
        # Result should have half the input dimension
        assert result.shape == (1, 2)
        assert result.requires_grad
        assert "reglu" in result.name
    
    def test_reglu_gradient(self):
        """Test ReGLU gradient computation."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = reglu(x, dim=-1)
        
        # Simulate backward pass
        grad_output = np.array([[1.0, 1.0]])
        result._grad_fn.apply(grad_output)
        
        # Backward chaining should be called
        x._backward.assert_called_once()
    
    def test_geglu_invalid_dimension(self):
        """Test GEGLU with invalid dimension."""
        x = Tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)  # Odd last dimension
        
        with pytest.raises(ValueError, match="GEGLU requires even dimension"):
            geglu(x)
    
    def test_geglu_computation(self):
        """Test GEGLU computation."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        result = geglu(x, dim=-1)
        
        # Result should have half the input dimension
        assert result.shape == (1, 2)
        assert result.requires_grad
        assert "geglu" in result.name
    
    def test_geglu_gradient_with_scipy(self):
        """Test GEGLU gradient computation with scipy."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        with patch('scipy.special.erf') as mock_erf:
            mock_erf.return_value = np.array([0.8413, 0.9953])
            
            result = geglu(x, dim=-1)
            
            # Simulate backward pass
            grad_output = np.array([[1.0, 1.0]])
            result._grad_fn.apply(grad_output)
            
            # Backward chaining should be called
            x._backward.assert_called_once()
    
    def test_geglu_gradient_without_scipy(self):
        """Test GEGLU gradient computation without scipy."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        # Mock scipy import failure
        with patch.dict('sys.modules', {'scipy.special': None}):
            result = geglu(x, dim=-1)
            
            # Simulate backward pass
            grad_output = np.array([[1.0, 1.0]])
            result._grad_fn.apply(grad_output)
            
            # Backward chaining should be called
            x._backward.assert_called_once()
    
    def test_activation_without_gradients(self):
        """Test all activations without gradient requirements."""
        x = Tensor([[1.0, -1.0, 2.0, -2.0]], requires_grad=False)
        
        # All activations should work without gradients
        result_relu = relu(x)
        result_sigmoid = sigmoid(x)
        result_tanh = tanh(x)
        result_gelu = gelu(x)
        result_mish = mish(x)
        result_silu = silu(x)
        result_leaky_relu = leaky_relu(x)
        
        # None should have gradient functions
        for result in [result_relu, result_sigmoid, result_tanh, result_gelu, 
                      result_mish, result_silu, result_leaky_relu]:
            assert not result.requires_grad
            assert result._grad_fn is None
    
    def test_activation_logging(self):
        """Test that activations log debug information."""
        with patch('neural_arch.functional.activation.logger') as mock_logger:
            x = Tensor([[1.0, 2.0]], requires_grad=True)
            
            relu(x)
            sigmoid(x)
            tanh(x)
            gelu(x)
            
            # Should have called debug logging
            assert mock_logger.debug.call_count >= 4
    
    def test_activation_memory_efficiency(self):
        """Test that activations use memory efficient operations."""
        # Large tensor to test memory efficiency
        large_x = Tensor(np.random.randn(1000, 1000), requires_grad=True)
        
        # These should complete without memory issues due to decorator
        with patch('neural_arch.functional.activation.logger') as mock_logger:
            result = relu(large_x)
            
            # Should log operation start and completion
            assert mock_logger.debug.call_count >= 2
            assert any("Starting operation" in str(call) for call in mock_logger.debug.call_args_list)
            assert any("Completed operation" in str(call) for call in mock_logger.debug.call_args_list)
    
    def test_activation_error_handling_in_memory_decorator(self):
        """Test error handling in memory efficient decorator."""
        with patch('neural_arch.functional.activation.logger') as mock_logger:
            # Mock an activation function that raises an error
            def failing_activation(x):
                raise ValueError("Test error")
            
            # Apply decorator
            from neural_arch.functional.utils import memory_efficient_operation
            decorated_fn = memory_efficient_operation(failing_activation)
            
            with pytest.raises(ValueError, match="Test error"):
                decorated_fn(Tensor([1.0]))
            
            # Should log error
            mock_logger.error.assert_called_once()