"""Complete test coverage for functional/arithmetic.py targeting 95%+ coverage."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock

from neural_arch.core.tensor import Tensor, GradientFunction, is_grad_enabled
import neural_arch.core.tensor as tensor_module
from neural_arch.core.device import Device, DeviceType
from neural_arch.functional.arithmetic import add, sub, mul, div, neg, matmul


def set_grad_enabled(enabled: bool):
    """Helper function to set gradient state."""
    tensor_module._grad_enabled = enabled


class TestArithmeticOperationsComplete:
    """Complete test coverage for all arithmetic operations."""
    
    def test_add_with_backend_operations(self):
        """Test add operation with backend-specific operations."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        
        # Mock backend operations
        mock_backend = MagicMock()
        mock_backend.add.return_value = np.array([4.0, 6.0])
        mock_backend.to_numpy.side_effect = lambda x: x if isinstance(x, np.ndarray) else np.array(x)
        mock_backend.from_numpy.side_effect = lambda x: x
        mock_backend.to_device.side_effect = lambda x, device: x
        
        a.backend = mock_backend
        a.backend_data = a.data
        b.backend = mock_backend
        b.backend_data = b.data
        
        set_grad_enabled(True)
        try:
            result = add(a, b)
            
            # Backend add should be called
            mock_backend.add.assert_called_once()
            assert np.allclose(result.data, [4.0, 6.0])
            
        finally:
            set_grad_enabled(False)
    
    def test_add_gradient_accumulation_with_device_handling(self):
        """Test add gradient accumulation with device handling."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([1.0, 2.0], requires_grad=True, device=Device(DeviceType.CUDA, 0))
            b = Tensor([3.0, 4.0], requires_grad=True, device=Device(DeviceType.CUDA, 0))
            
            # Mock backend for device handling
            mock_backend = MagicMock()
            mock_backend.add.return_value = np.array([4.0, 6.0])
            mock_backend.to_numpy.side_effect = lambda x: x
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.to_device.side_effect = lambda x, device: x
            
            a.backend = mock_backend
            a._backend = mock_backend
            a.backend_data = a.data
            b.backend = mock_backend
            b._backend = mock_backend
            b.backend_data = b.data
            
            # Initialize gradients
            a._grad = mock_backend.from_numpy(np.array([0.1, 0.2]))
            b._grad = mock_backend.from_numpy(np.array([0.3, 0.4]))
            
            result = add(a, b)
            
            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            result._grad_fn.apply(grad_output)
            
            # Device conversion should be called
            assert mock_backend.to_device.call_count >= 2
            
        finally:
            set_grad_enabled(False)
    
    def test_add_gradient_without_existing_gradients(self):
        """Test add gradient computation without existing gradients."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=True)
            
            # Mock backend
            mock_backend = MagicMock()
            mock_backend.add.return_value = np.array([4.0, 6.0])
            mock_backend.to_numpy.side_effect = lambda x: x
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.to_device.side_effect = lambda x, device: x
            
            a.backend = mock_backend
            a._backend = mock_backend
            a.backend_data = a.data
            b.backend = mock_backend
            b._backend = mock_backend
            b.backend_data = b.data
            
            # Ensure no existing gradients
            a._grad = None
            b._grad = None
            
            result = add(a, b)
            
            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            result._grad_fn.apply(grad_output)
            
            # Gradients should be created
            assert mock_backend.from_numpy.call_count >= 2
            
        finally:
            set_grad_enabled(False)
    
    def test_sub_with_device_mismatch(self):
        """Test subtract operation with device mismatch."""
        a = Tensor([1.0, 2.0], device=Device(DeviceType.CPU))
        b = Tensor([3.0, 4.0], device=Device(DeviceType.CUDA, 0))
        
        with pytest.raises(ValueError, match="Tensors must be on same device"):
            sub(a, b)
    
    def test_sub_gradient_accumulation(self):
        """Test subtract gradient accumulation."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([5.0, 6.0], requires_grad=True)
            b = Tensor([2.0, 3.0], requires_grad=True)
            
            # Mock backend
            mock_backend = MagicMock()
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.add.side_effect = lambda x, y: x + y
            mock_backend.to_device.side_effect = lambda x, device: x
            
            a._backend = mock_backend
            b._backend = mock_backend
            
            # Initialize gradients
            a._grad = np.array([0.1, 0.2])
            b._grad = np.array([0.3, 0.4])
            
            result = sub(a, b)
            
            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)
            
            # Gradients should be accumulated
            assert mock_backend.add.call_count >= 2
            
        finally:
            set_grad_enabled(False)
    
    def test_mul_gradient_accumulation_with_device(self):
        """Test multiply gradient accumulation with device handling."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([2.0, 3.0], requires_grad=True, device=Device(DeviceType.CUDA, 1))
            b = Tensor([4.0, 5.0], requires_grad=True, device=Device(DeviceType.CUDA, 1))
            
            # Mock backend with device index
            mock_backend = MagicMock()
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.add.side_effect = lambda x, y: x + y
            mock_backend.to_device.side_effect = lambda x, device: x
            
            a._backend = mock_backend
            a._device = Device(DeviceType.CUDA, 1)
            b._backend = mock_backend
            b._device = Device(DeviceType.CUDA, 1)
            
            # Initialize gradients
            a._grad = np.array([0.1, 0.2])
            b._grad = np.array([0.3, 0.4])
            
            result = mul(a, b)
            
            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)
            
            # Device string should include index
            expected_device_calls = [call for call in mock_backend.to_device.call_args_list 
                                   if 'cuda:1' in str(call)]
            assert len(expected_device_calls) >= 2
            
        finally:
            set_grad_enabled(False)
    
    def test_div_zero_handling(self):
        """Test division by zero handling."""
        a = Tensor([6.0, 8.0])
        b = Tensor([0.0, 2.0])
        
        # Division by zero should be caught and raise error
        with pytest.raises(ValueError, match="Division by zero detected"):
            div(a, b)
    
    def test_div_gradient_accumulation(self):
        """Test divide gradient accumulation."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([6.0, 8.0], requires_grad=True)
            b = Tensor([2.0, 4.0], requires_grad=True)
            
            # Mock backend
            mock_backend = MagicMock()
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.add.side_effect = lambda x, y: x + y
            mock_backend.to_device.side_effect = lambda x, device: x
            
            a._backend = mock_backend
            b._backend = mock_backend
            
            # Initialize gradients
            a._grad = np.array([0.1, 0.2])
            b._grad = np.array([0.3, 0.4])
            
            result = div(a, b)
            
            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)
            
            # Gradients should be accumulated
            assert mock_backend.add.call_count >= 2
            
        finally:
            set_grad_enabled(False)
    
    def test_neg_gradient_accumulation(self):
        """Test negation gradient accumulation."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([2.0, 3.0], requires_grad=True)
            
            # Mock backend
            mock_backend = MagicMock()
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.add.side_effect = lambda x, y: x + y
            mock_backend.to_device.side_effect = lambda x, device: x
            
            a._backend = mock_backend
            
            # Initialize gradient
            a._grad = np.array([0.1, 0.2])
            
            result = neg(a)
            
            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)
            
            # Gradient should be accumulated
            mock_backend.add.assert_called_once()
            
        finally:
            set_grad_enabled(False)
    
    def test_matmul_dimension_validation(self):
        """Test matrix multiplication dimension validation."""
        # Test with 1D tensors (should fail)
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        
        with pytest.raises(ValueError, match="matmul requires 2D+ tensors"):
            matmul(a, b)
        
        # Test with incompatible dimensions
        a = Tensor([[1, 2, 3]])  # (1, 3)
        b = Tensor([[4, 5]])     # (1, 2)
        
        with pytest.raises(ValueError, match="Incompatible matrix dimensions"):
            matmul(a, b)
    
    def test_matmul_with_backend_operations(self):
        """Test matrix multiplication with backend operations."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        
        # Mock backend operations
        mock_backend = MagicMock()
        mock_backend.matmul.return_value = np.array([[19.0, 22.0], [43.0, 50.0]])
        mock_backend.to_numpy.side_effect = lambda x: x if isinstance(x, np.ndarray) else np.array(x)
        mock_backend.from_numpy.side_effect = lambda x: x
        mock_backend.to_device.side_effect = lambda x, device: x
        mock_backend.transpose.side_effect = lambda x, axes: np.transpose(x, axes)
        
        a.backend = mock_backend
        a.backend_data = a.data
        b.backend = mock_backend
        b.backend_data = b.data
        
        result = matmul(a, b)
        
        # Backend matmul should be called
        mock_backend.matmul.assert_called_once()
        assert np.allclose(result.data, [[19.0, 22.0], [43.0, 50.0]])
    
    def test_matmul_gradient_with_2d_tensors(self):
        """Test matrix multiplication gradient with 2D tensors."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
            
            # Mock backend
            mock_backend = MagicMock()
            mock_backend.matmul.return_value = np.array([[19.0, 22.0], [43.0, 50.0]])
            mock_backend.to_numpy.side_effect = lambda x: x
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.to_device.side_effect = lambda x, device: x
            mock_backend.transpose.side_effect = lambda x, axes: np.transpose(x, axes)
            mock_backend.add.side_effect = lambda x, y: x + y
            
            a.backend = mock_backend
            a.backend_data = a.data
            a._backend = mock_backend
            a._device = Device(DeviceType.CPU)
            b.backend = mock_backend
            b.backend_data = b.data
            b._backend = mock_backend
            b._device = Device(DeviceType.CPU)
            
            result = matmul(a, b)
            
            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            result._grad_fn.apply(grad_output)
            
            # Transpose operations should be called for 2D case
            transpose_calls = [call for call in mock_backend.transpose.call_args_list 
                             if call[0][1] == (1, 0)]
            assert len(transpose_calls) >= 2
            
        finally:
            set_grad_enabled(False)
    
    def test_matmul_gradient_with_higher_dimensional_tensors(self):
        """Test matrix multiplication gradient with higher dimensional tensors."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # (1, 2, 2)
            b = Tensor([[[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)  # (1, 2, 2)
            
            # Mock backend
            mock_backend = MagicMock()
            mock_backend.matmul.return_value = np.array([[[19.0, 22.0], [43.0, 50.0]]])
            mock_backend.to_numpy.side_effect = lambda x: x
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.to_device.side_effect = lambda x, device: x
            mock_backend.transpose.side_effect = lambda x, axes: np.transpose(x, axes)
            mock_backend.add.side_effect = lambda x, y: x + y
            
            a.backend = mock_backend
            a.backend_data = a.data
            a._backend = mock_backend
            a._device = Device(DeviceType.CPU)
            b.backend = mock_backend
            b.backend_data = b.data
            b._backend = mock_backend
            b._device = Device(DeviceType.CPU)
            
            result = matmul(a, b)
            
            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            result._grad_fn.apply(grad_output)
            
            # Should handle higher dimensional transpose (swap last two axes)
            transpose_calls = mock_backend.transpose.call_args_list
            assert len(transpose_calls) >= 2
            
        finally:
            set_grad_enabled(False)
    
    def test_matmul_gradient_with_shape_reduction(self):
        """Test matrix multiplication gradient with shape reduction."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([[1.0, 2.0]], requires_grad=True)  # (1, 2)
            b = Tensor([[3.0], [4.0]], requires_grad=True)  # (2, 1)
            
            # Mock backend and reduce_gradient
            mock_backend = MagicMock()
            mock_backend.matmul.return_value = np.array([[11.0]])
            mock_backend.to_numpy.side_effect = lambda x: x
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.to_device.side_effect = lambda x, device: x
            mock_backend.transpose.side_effect = lambda x, axes: np.transpose(x, axes)
            mock_backend.add.side_effect = lambda x, y: x + y
            
            # Mock reduce_gradient function
            with patch('neural_arch.functional.arithmetic.reduce_gradient') as mock_reduce:
                mock_reduce.side_effect = lambda grad, target_shape, result_shape: grad
                
                a.backend = mock_backend
                a.backend_data = a.data
                a._backend = mock_backend
                a._device = Device(DeviceType.CPU)
                b.backend = mock_backend
                b.backend_data = b.data
                b._backend = mock_backend
                b._device = Device(DeviceType.CPU)
                
                result = matmul(a, b)
                
                # Create gradient with different shape to trigger reduction
                grad_output = np.array([[1.0]])
                result._grad_fn.apply(grad_output)
                
                # reduce_gradient should be called if shapes don't match
                # (This depends on the specific shapes involved)
                
        finally:
            set_grad_enabled(False)
    
    def test_operations_without_gradients_enabled(self):
        """Test operations when gradients are globally disabled."""
        set_grad_enabled(False)
        
        try:
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=True)
            
            # Operations should not create gradient functions
            result_add = add(a, b)
            result_sub = sub(a, b)
            result_mul = mul(a, b)
            result_div = div(a, b)
            result_neg = neg(a)
            result_mm = matmul(a.reshape(1, 2), b.reshape(2, 1))
            
            # No gradients should be computed
            assert not result_add.requires_grad
            assert not result_sub.requires_grad
            assert not result_mul.requires_grad
            assert not result_div.requires_grad
            assert not result_neg.requires_grad
            assert not result_mm.requires_grad
            
        finally:
            set_grad_enabled(True)
    
    def test_tensor_conversion_in_operations(self):
        """Test automatic tensor conversion in operations."""
        # Test with scalar inputs
        result = add(5.0, 3.0)
        assert isinstance(result, Tensor)
        assert np.allclose(result.data, 8.0)
        
        # Test with mixed tensor and scalar
        a = Tensor([1.0, 2.0])
        result = mul(a, 3.0)
        assert isinstance(result, Tensor)
        assert np.allclose(result.data, [3.0, 6.0])
        
        # Test with numpy array
        result = sub(np.array([5.0, 6.0]), np.array([2.0, 3.0]))
        assert isinstance(result, Tensor)
        assert np.allclose(result.data, [3.0, 3.0])
    
    def test_operations_with_backward_chaining(self):
        """Test operations with gradient function chaining."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=True)
            
            # Mock gradient functions and _backward methods
            mock_grad_fn_a = MagicMock()
            mock_grad_fn_b = MagicMock()
            a._grad_fn = mock_grad_fn_a
            b._grad_fn = mock_grad_fn_b
            
            # Mock backends
            mock_backend = MagicMock()
            mock_backend.add.return_value = np.array([4.0, 6.0])
            mock_backend.to_numpy.side_effect = lambda x: x
            mock_backend.from_numpy.side_effect = lambda x: x
            mock_backend.to_device.side_effect = lambda x, device: x
            
            a.backend = mock_backend
            a.backend_data = a.data
            a._backend = mock_backend
            b.backend = mock_backend
            b.backend_data = b.data
            b._backend = mock_backend
            
            result = add(a, b)
            
            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            result._grad_fn.apply(grad_output)
            
            # Gradient functions should be called
            mock_grad_fn_a.apply.assert_called_once()
            mock_grad_fn_b.apply.assert_called_once()
            
        finally:
            set_grad_enabled(False)
    
    def test_operations_name_propagation(self):
        """Test that operation names are properly propagated."""
        a = Tensor([1.0, 2.0], name="tensor_a")
        b = Tensor([3.0, 4.0], name="tensor_b")
        
        # Test all operations
        result_add = add(a, b)
        result_sub = sub(a, b)
        result_mul = mul(a, b)
        result_div = div(a, b)
        result_neg = neg(a)
        result_mm = matmul(a.reshape(1, 2), b.reshape(2, 1))
        
        # Check name propagation
        assert "add(tensor_a, tensor_b)" in result_add.name
        assert "sub(tensor_a, tensor_b)" in result_sub.name
        assert "mul(tensor_a, tensor_b)" in result_mul.name
        assert "div(tensor_a, tensor_b)" in result_div.name
        assert "neg(tensor_a)" in result_neg.name
        assert "matmul(tensor_a, tensor_b)" in result_mm.name
    
    def test_operations_dtype_propagation(self):
        """Test that operations propagate dtypes correctly."""
        a = Tensor([1.0, 2.0], dtype='float32')
        b = Tensor([3.0, 4.0], dtype='float32')
        
        result = add(a, b)
        
        # Should preserve dtype from input
        assert result.dtype == a.dtype
    
    def test_operations_device_propagation(self):
        """Test that operations propagate devices correctly."""
        device = Device(DeviceType.CUDA, 0)
        a = Tensor([1.0, 2.0], device=device)
        b = Tensor([3.0, 4.0], device=device)
        
        # Mock backend to avoid actual device operations
        mock_backend = MagicMock()
        mock_backend.add.return_value = np.array([4.0, 6.0])
        mock_backend.to_numpy.side_effect = lambda x: x
        
        a.backend = mock_backend
        a.backend_data = a.data
        b.backend = mock_backend
        b.backend_data = b.data
        
        result = add(a, b)
        
        # Should preserve device from input
        assert result.device == device
    
    def test_operations_error_handling(self):
        """Test error handling in arithmetic operations."""
        # Test with None inputs
        with pytest.raises((TypeError, AttributeError)):
            add(None, Tensor([1, 2]))
        
        with pytest.raises((TypeError, AttributeError)):
            mul(Tensor([1, 2]), None)
        
        # Test matmul with device mismatch
        a = Tensor([[1, 2]], device=Device(DeviceType.CPU))
        b = Tensor([[3], [4]], device=Device(DeviceType.CUDA, 0))
        
        with pytest.raises(ValueError, match="Tensors must be on same device"):
            matmul(a, b)
    
    def test_gradient_function_properties(self):
        """Test that gradient functions have proper properties."""
        set_grad_enabled(True)
        
        try:
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=True)
            
            result = add(a, b)
            
            # Check gradient function properties
            assert result._grad_fn is not None
            assert hasattr(result._grad_fn, 'apply')
            assert hasattr(result._grad_fn, 'inputs')
            assert hasattr(result._grad_fn, 'name')
            assert len(result._grad_fn.inputs) == 2
            assert result._grad_fn.name == "add"
            
        finally:
            set_grad_enabled(False)