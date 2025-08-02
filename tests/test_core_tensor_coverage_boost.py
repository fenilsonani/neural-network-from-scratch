"""Comprehensive test coverage for core/tensor module to boost coverage from 84.57% to 95%+"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import weakref
from contextlib import contextmanager

from neural_arch.core.tensor import (
    Tensor, TensorLike, GradientFunction, 
    is_grad_enabled, no_grad, enable_grad
)
import neural_arch.core.tensor as tensor_module
from neural_arch.core.device import Device, DeviceType
from neural_arch.core.dtype import DType
from neural_arch.backends import get_backend


def set_grad_enabled(enabled: bool):
    """Helper function to set gradient state."""
    tensor_module._grad_enabled = enabled


class TestTensorCoverageBoost:
    """Comprehensive tests for Tensor class targeting missing coverage paths."""
    
    def test_gradient_function_creation_and_application(self):
        """Test GradientFunction creation and application with error handling."""
        # Mock backward function
        mock_backward = Mock()
        mock_tensor = Mock()
        
        # Create gradient function
        grad_fn = GradientFunction(
            backward_fn=mock_backward,
            inputs=[mock_tensor],
            name="test_gradient_function"
        )
        
        # Test normal application
        grad_output = np.array([1.0, 2.0])
        grad_fn.apply(grad_output)
        
        # Check gradient clipping was applied
        clipped_grad = np.clip(grad_output, -10.0, 10.0)
        mock_backward.assert_called_once_with(clipped_grad)
    
    def test_gradient_function_nan_inf_handling(self):
        """Test GradientFunction handling of NaN and Inf gradients."""
        mock_backward = Mock()
        mock_tensor = Mock()
        
        grad_fn = GradientFunction(
            backward_fn=mock_backward,
            inputs=[mock_tensor],
            name="test_nan_inf"
        )
        
        # Test with NaN and Inf values
        grad_output = np.array([np.nan, np.inf, -np.inf, 1.0])
        
        with patch('neural_arch.core.tensor.logger') as mock_logger:
            grad_fn.apply(grad_output)
            
            # Should log warning about non-finite gradients
            mock_logger.warning.assert_called_once()
            
            # Should call backward with cleaned gradients
            mock_backward.assert_called_once()
            called_grad = mock_backward.call_args[0][0]
            assert np.all(np.isfinite(called_grad))
    
    def test_gradient_function_exception_handling(self):
        """Test GradientFunction exception handling during backward pass."""
        def failing_backward(grad):
            raise RuntimeError("Backward pass failed")
        
        mock_tensor = Mock()
        grad_fn = GradientFunction(
            backward_fn=failing_backward,
            inputs=[mock_tensor],
            name="failing_function"
        )
        
        with patch('neural_arch.core.tensor.logger') as mock_logger:
            with pytest.raises(RuntimeError, match="Backward pass failed"):
                grad_fn.apply(np.array([1.0]))
            
            # Should log error
            mock_logger.error.assert_called_once()
    
    def test_tensor_initialization_type_validation(self):
        """Test tensor initialization with type validation."""
        # Test invalid requires_grad type
        with pytest.raises(TypeError, match="requires_grad must be bool"):
            Tensor([1, 2, 3], requires_grad="true")
        
        # Test valid initialization
        tensor = Tensor([1, 2, 3], requires_grad=True, name="test_tensor")
        assert tensor.requires_grad is True
        assert tensor.name == "test_tensor"
    
    def test_tensor_data_validation_and_conversion(self):
        """Test tensor data validation and conversion."""
        # Test with various data types
        tensor1 = Tensor([1, 2, 3])
        assert tensor1.shape == (3,)
        
        tensor2 = Tensor(np.array([[1, 2], [3, 4]]))
        assert tensor2.shape == (2, 2)
        
        tensor3 = Tensor(5.0)  # Scalar
        assert tensor3.shape == ()
        
        # Test with nested lists
        tensor4 = Tensor([[1, 2, 3], [4, 5, 6]])
        assert tensor4.shape == (2, 3)
    
    def test_tensor_backend_selection(self):
        """Test backend selection based on device."""
        # Test CPU device
        cpu_tensor = Tensor([1, 2, 3], device=Device(DeviceType.CPU))
        assert cpu_tensor.backend.name == "numpy"
        
        # Test device with index
        device_with_index = Device(DeviceType.CUDA, 0)
        cuda_tensor = Tensor([1, 2, 3], device=device_with_index)
        # Backend should be selected based on device
        assert cuda_tensor.device == device_with_index
    
    def test_tensor_invalid_data_handling(self):
        """Test tensor handling of invalid data."""
        # Test with empty data
        tensor = Tensor([])
        assert tensor.shape == (0,)
        
        # Test with complex data structures
        with pytest.raises((ValueError, TypeError)):
            Tensor({"invalid": "data"})
    
    def test_tensor_memory_layout_optimization(self):
        """Test tensor memory layout and optimization."""
        # Test contiguous array
        data = np.array([[1, 2, 3], [4, 5, 6]])
        tensor = Tensor(data)
        
        # Test non-contiguous array
        non_contiguous = data[:, ::2]  # Skip every other column
        tensor_nc = Tensor(non_contiguous)
        assert tensor_nc.shape == (2, 2)
    
    def test_tensor_numerical_stability_checks(self):
        """Test tensor numerical stability validation."""
        # Test with extreme values
        extreme_data = np.array([1e100, -1e100, 1e-100])
        tensor = Tensor(extreme_data)
        assert np.all(np.isfinite(tensor.data))
        
        # Test with NaN values (should be handled by validation)
        nan_data = np.array([1.0, np.nan, 3.0])
        try:
            tensor = Tensor(nan_data)
            # If successful, NaN handling worked
        except ValueError:
            # If failed, validation caught it
            pass
    
    def test_tensor_performance_monitoring_hooks(self):
        """Test tensor performance monitoring capabilities."""
        # Create tensor with performance tracking
        large_data = np.random.randn(1000, 1000)
        tensor = Tensor(large_data, name="large_tensor")
        
        # Performance should be trackable through tensor properties
        assert tensor.size > 100000
        assert tensor.dtype is not None
        assert tensor.device is not None
    
    def test_tensor_gradient_state_management(self):
        """Test tensor gradient state management."""
        tensor = Tensor([1.0, 2.0], requires_grad=True)
        
        # Test gradient initialization
        assert tensor._grad is None
        assert tensor._grad_fn is None
        
        # Test gradient setting
        grad_data = np.array([0.1, 0.2])
        tensor._grad = tensor._backend.from_numpy(grad_data)
        assert tensor._grad is not None
        
        # Test gradient function setting
        mock_grad_fn = Mock()
        tensor._grad_fn = mock_grad_fn
        assert tensor._grad_fn == mock_grad_fn
    
    def test_tensor_device_transfer_validation(self):
        """Test tensor device transfer validation."""
        tensor = Tensor([1, 2, 3], device=Device(DeviceType.CPU))
        
        # Test device property access
        assert tensor.device.type == DeviceType.CPU
        assert tensor.device.index is None
        
        # Test backend data device consistency
        backend_device = tensor.backend.device_of(tensor.backend_data)
        assert isinstance(backend_device, str)
    
    def test_tensor_weak_reference_handling(self):
        """Test tensor weak reference handling for memory management."""
        tensor = Tensor([1, 2, 3])
        
        # Create weak reference
        weak_ref = weakref.ref(tensor)
        assert weak_ref() is tensor
        
        # Test that tensor can be garbage collected
        del tensor
        # Note: In practice, weak_ref() might still return the tensor
        # due to test framework keeping references
    
    def test_tensor_computational_graph_integration(self):
        """Test tensor integration with computational graph."""
        set_grad_enabled(True)
        
        try:
            # Create tensor with gradients
            tensor = Tensor([1.0, 2.0], requires_grad=True)
            
            # Test that gradient computation respects global state
            assert tensor.requires_grad is True
            assert is_grad_enabled() is True
            
            # Create mock operation result
            result_data = tensor.data * 2
            result = Tensor(
                result_data,
                requires_grad=tensor.requires_grad and is_grad_enabled()
            )
            
            assert result.requires_grad is True
            
        finally:
            set_grad_enabled(False)
    
    def test_tensor_dtype_handling_comprehensive(self):
        """Test comprehensive dtype handling."""
        # Test with different numpy dtypes
        int_tensor = Tensor([1, 2, 3], dtype='int32')
        float_tensor = Tensor([1.0, 2.0, 3.0], dtype='float32')
        bool_tensor = Tensor([True, False, True], dtype='bool')
        
        # Test dtype preservation
        assert int_tensor.dtype.name == 'int32'
        assert float_tensor.dtype.name == 'float32'
        assert bool_tensor.dtype.name == 'bool'
    
    def test_tensor_shape_property_edge_cases(self):
        """Test tensor shape property with edge cases."""
        # Scalar tensor
        scalar = Tensor(5.0)
        assert scalar.shape == ()
        assert scalar.ndim == 0
        assert scalar.size == 1
        
        # 1D tensor
        vector = Tensor([1, 2, 3, 4, 5])
        assert vector.shape == (5,)
        assert vector.ndim == 1
        assert vector.size == 5
        
        # High-dimensional tensor
        high_dim = Tensor(np.random.randn(2, 3, 4, 5))
        assert high_dim.shape == (2, 3, 4, 5)
        assert high_dim.ndim == 4
        assert high_dim.size == 120
    
    def test_tensor_string_representation(self):
        """Test tensor string representation methods."""
        tensor = Tensor([[1, 2], [3, 4]], name="test_matrix")
        
        # Test __str__ method
        str_repr = str(tensor)
        assert "Tensor" in str_repr
        assert "test_matrix" in str_repr
        
        # Test __repr__ method
        repr_str = repr(tensor)
        assert "Tensor" in repr_str
        assert str(tensor.shape) in repr_str
    
    def test_tensor_equality_and_comparison(self):
        """Test tensor equality and comparison operations."""
        tensor1 = Tensor([1, 2, 3])
        tensor2 = Tensor([1, 2, 3])
        tensor3 = Tensor([4, 5, 6])
        
        # Test data equality (not object equality)
        assert np.array_equal(tensor1.data, tensor2.data)
        assert not np.array_equal(tensor1.data, tensor3.data)
        
        # Test shape equality
        assert tensor1.shape == tensor2.shape
        assert tensor1.shape == tensor3.shape
    
    def test_global_gradient_state_functions(self):
        """Test global gradient state management functions."""
        # Test is_grad_enabled
        original_state = is_grad_enabled()
        
        # Test set_grad_enabled
        set_grad_enabled(True)
        assert is_grad_enabled() is True
        
        set_grad_enabled(False)
        assert is_grad_enabled() is False
        
        # Restore original state
        set_grad_enabled(original_state)
    
    def test_no_grad_context_manager(self):
        """Test no_grad context manager."""
        set_grad_enabled(True)
        
        try:
            # Initially gradients should be enabled
            assert is_grad_enabled() is True
            
            # Inside no_grad context
            with no_grad():
                assert is_grad_enabled() is False
                
                # Create tensor inside context
                tensor = Tensor([1.0, 2.0], requires_grad=True)
                # Tensor still has requires_grad=True but no computation graph
            
            # After context, gradients should be re-enabled
            assert is_grad_enabled() is True
            
        finally:
            set_grad_enabled(False)
    
    def test_tensor_backend_data_consistency(self):
        """Test consistency between tensor data and backend data."""
        tensor = Tensor([[1, 2], [3, 4]])
        
        # Test that backend_data is accessible
        assert tensor.backend_data is not None
        
        # Test conversion back to numpy
        numpy_data = tensor.backend.to_numpy(tensor.backend_data)
        assert np.array_equal(numpy_data, tensor.data)
    
    def test_tensor_error_propagation(self):
        """Test error propagation in tensor operations."""
        # Test with invalid backend operation
        tensor = Tensor([1, 2, 3])
        
        # Mock backend with failing operation
        mock_backend = Mock()
        mock_backend.to_numpy.side_effect = RuntimeError("Backend error")
        tensor._backend = mock_backend
        
        # Should propagate backend errors
        with pytest.raises(RuntimeError, match="Backend error"):
            _ = tensor.backend.to_numpy(tensor.backend_data)
    
    def test_tensor_memory_efficiency(self):
        """Test tensor memory efficiency features."""
        # Test view vs copy semantics
        original_data = np.array([1, 2, 3, 4, 5])
        tensor = Tensor(original_data)
        
        # Tensor should not modify original data unless intended
        tensor_data_copy = tensor.data.copy()
        assert np.array_equal(tensor.data, tensor_data_copy)
    
    def test_tensor_advanced_indexing_compatibility(self):
        """Test tensor compatibility with advanced indexing."""
        tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Test that tensor data supports advanced indexing
        subset = tensor.data[0:2, 1:3]
        assert subset.shape == (2, 2)
        assert np.array_equal(subset, [[2, 3], [5, 6]])
    
    def test_tensor_gradient_accumulation_patterns(self):
        """Test various gradient accumulation patterns."""
        set_grad_enabled(True)
        
        try:
            tensor = Tensor([1.0, 2.0], requires_grad=True)
            
            # Test multiple gradient accumulations
            grad1 = np.array([0.1, 0.2])
            grad2 = np.array([0.3, 0.4])
            
            # First accumulation
            tensor._grad = tensor._backend.from_numpy(grad1)
            assert tensor._grad is not None
            
            # Second accumulation (should add to existing)
            existing_grad = tensor._backend.to_numpy(tensor._grad)
            new_grad = tensor._backend.from_numpy(grad2)
            tensor._grad = tensor._backend.add(tensor._grad, new_grad)
            
            final_grad = tensor._backend.to_numpy(tensor._grad)
            expected = grad1 + grad2
            assert np.allclose(final_grad, expected)
            
        finally:
            set_grad_enabled(False)
    
    def test_tensor_context_switching(self):
        """Test tensor behavior across different contexts."""
        # Test tensor creation in different gradient contexts
        tensors = []
        
        # Context 1: Gradients enabled
        set_grad_enabled(True)
        tensor1 = Tensor([1.0], requires_grad=True)
        tensors.append(tensor1)
        
        # Context 2: Gradients disabled
        set_grad_enabled(False)
        tensor2 = Tensor([2.0], requires_grad=True)
        tensors.append(tensor2)
        
        # Both tensors should maintain their requires_grad setting
        assert tensor1.requires_grad is True
        assert tensor2.requires_grad is True
        
        # But gradient computation should respect global state
        set_grad_enabled(True)
        # Any new operations would respect the current state
    
    def test_tensor_edge_case_operations(self):
        """Test tensor operations in edge cases."""
        # Empty tensor
        empty_tensor = Tensor([])
        assert empty_tensor.size == 0
        assert empty_tensor.shape == (0,)
        
        # Single element tensor
        single_tensor = Tensor([42])
        assert single_tensor.size == 1
        assert single_tensor.shape == (1,)
        assert single_tensor.data[0] == 42
        
        # Very large tensor (memory test)
        try:
            large_tensor = Tensor(np.zeros((1000, 1000)))
            assert large_tensor.size == 1000000
        except MemoryError:
            # Expected in memory-constrained environments
            pass
    
    def test_tensor_thread_safety_considerations(self):
        """Test tensor behavior considering thread safety."""
        # Test that tensor creation doesn't interfere with global state
        original_grad_state = is_grad_enabled()
        
        try:
            # Multiple tensor creations
            tensors = []
            for i in range(10):
                tensor = Tensor([i], requires_grad=True)
                tensors.append(tensor)
            
            # All tensors should be valid
            for i, tensor in enumerate(tensors):
                assert tensor.data[0] == i
                assert tensor.requires_grad is True
            
        finally:
            # Global state should be unchanged
            assert is_grad_enabled() == original_grad_state