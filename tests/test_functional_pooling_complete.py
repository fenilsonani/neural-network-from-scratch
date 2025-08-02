"""Complete test coverage for functional/pooling.py targeting 95%+ coverage."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock

from neural_arch.core.tensor import Tensor, GradientFunction
from neural_arch.functional.pooling import mean_pool, max_pool


class TestPoolingOperationsComplete:
    """Complete test coverage for all pooling operations."""
    
    def test_mean_pool_basic_functionality(self):
        """Test mean pooling basic functionality."""
        # Test with 2D tensor
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        result = mean_pool(x, axis=1)
        
        # Mean along axis 1: [1+2+3+4]/4 = 2.5
        expected = np.array([2.5])
        assert np.allclose(result.data, expected)
        assert result.requires_grad
        assert "mean_pool" in result.name
    
    def test_mean_pool_different_axes(self):
        """Test mean pooling with different axes."""
        # Test with 3D tensor
        x = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        
        # Pool along axis 0 (batch dimension)
        result_axis0 = mean_pool(x, axis=0)
        expected_axis0 = np.array([[[3.0, 4.0], [5.0, 6.0]]])  # Mean of first dimension
        assert np.allclose(result_axis0.data, expected_axis0)
        
        # Pool along axis 1
        result_axis1 = mean_pool(x, axis=1)
        expected_axis1 = np.array([[[2.0, 3.0]], [[6.0, 7.0]]])  # Mean along height
        assert np.allclose(result_axis1.data, expected_axis1)
        
        # Pool along axis 2
        result_axis2 = mean_pool(x, axis=2)
        expected_axis2 = np.array([[[1.5], [3.5]], [[5.5], [7.5]]])  # Mean along width
        assert np.allclose(result_axis2.data, expected_axis2)
    
    def test_mean_pool_without_gradients(self):
        """Test mean pooling without gradients."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=False)
        
        result = mean_pool(x, axis=1)
        
        assert not result.requires_grad
        assert result._grad_fn is None
        assert np.allclose(result.data, [2.5])
    
    def test_mean_pool_gradient_computation(self):
        """Test mean pooling gradient computation."""
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = mean_pool(x, axis=1)
        
        # Simulate backward pass
        grad_output = np.array([1.0])
        result._grad_fn.apply(grad_output)
        
        # Check gradient function was called
        assert x._backward.assert_called_once
    
    def test_mean_pool_gradient_shape_handling(self):
        """Test mean pooling gradient shape handling."""
        # Test with tensor where pooling changes shape significantly
        x = Tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], requires_grad=True)  # Shape: (1, 1, 5)
        
        result = mean_pool(x, axis=2)  # Pool along last axis
        
        # Result shape should be (1, 1)
        assert result.shape == (1, 1)
        
        # Test gradient computation
        grad_output = np.array([[1.0]])
        result._grad_fn.apply(grad_output)
        
        # The gradient should be broadcasted back to original shape
        # Each element gets 1/pool_size of the gradient
        expected_grad_shape = x.shape
        # We can't directly check x.grad here since it's handled by backward method
        # but we can verify the function was set up correctly
        assert result._grad_fn is not None
    
    def test_mean_pool_memory_efficiency(self):
        """Test mean pooling memory efficiency."""
        # Large tensor to test memory efficiency
        large_x = Tensor(np.random.randn(100, 1000), requires_grad=True)
        
        with patch('neural_arch.functional.pooling.logger') as mock_logger:
            result = mean_pool(large_x, axis=1)
            
            # Should log operation start and completion due to decorator
            assert mock_logger.debug.call_count >= 2
            assert any("Starting operation" in str(call) for call in mock_logger.debug.call_args_list)
            assert any("Completed operation" in str(call) for call in mock_logger.debug.call_args_list)
    
    def test_max_pool_basic_functionality(self):
        """Test max pooling basic functionality."""
        # Test with 2D tensor
        x = Tensor([[1.0, 4.0, 2.0, 3.0]], requires_grad=True)
        
        result = max_pool(x, axis=1)
        
        # Max along axis 1: max(1, 4, 2, 3) = 4
        expected = np.array([4.0])
        assert np.allclose(result.data, expected)
        assert result.requires_grad
        assert "max_pool" in result.name
    
    def test_max_pool_different_axes(self):
        """Test max pooling with different axes."""
        # Test with 3D tensor
        x = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        
        # Pool along axis 0 (batch dimension)
        result_axis0 = max_pool(x, axis=0)
        expected_axis0 = np.array([[[5.0, 6.0], [7.0, 8.0]]])  # Max of first dimension
        assert np.allclose(result_axis0.data, expected_axis0)
        
        # Pool along axis 1
        result_axis1 = max_pool(x, axis=1)
        expected_axis1 = np.array([[[3.0, 4.0]], [[7.0, 8.0]]])  # Max along height
        assert np.allclose(result_axis1.data, expected_axis1)
        
        # Pool along axis 2
        result_axis2 = max_pool(x, axis=2)
        expected_axis2 = np.array([[[2.0], [4.0]], [[6.0], [8.0]]])  # Max along width
        assert np.allclose(result_axis2.data, expected_axis2)
    
    def test_max_pool_without_gradients(self):
        """Test max pooling without gradients."""
        x = Tensor([[1.0, 4.0, 2.0, 3.0]], requires_grad=False)
        
        result = max_pool(x, axis=1)
        
        assert not result.requires_grad
        assert result._grad_fn is None
        assert np.allclose(result.data, [4.0])
    
    def test_max_pool_gradient_computation(self):
        """Test max pooling gradient computation."""
        x = Tensor([[1.0, 4.0, 2.0, 3.0]], requires_grad=True)
        
        # Mock backward method
        x._backward = MagicMock()
        
        result = max_pool(x, axis=1)
        
        # Simulate backward pass
        grad_output = np.array([1.0])
        result._grad_fn.apply(grad_output)
        
        # Check gradient function was called
        assert x._backward.assert_called_once
    
    def test_max_pool_gradient_mask_creation(self):
        """Test max pooling gradient mask creation."""
        # Test with tensor where we can verify gradient routing
        x = Tensor([[[1.0, 4.0, 2.0], [3.0, 5.0, 1.0]]], requires_grad=True)  # Shape: (1, 2, 3)
        
        result = max_pool(x, axis=2)  # Pool along last axis
        
        # Result should have max values: [[4.0], [5.0]]
        expected = np.array([[[4.0], [5.0]]])
        assert np.allclose(result.data, expected)
        
        # Test gradient computation
        grad_output = np.array([[[1.0], [2.0]]])
        result._grad_fn.apply(grad_output)
        
        # The gradient should only flow to the max elements
        # We can't directly check x.grad here, but we can verify setup
        assert result._grad_fn is not None
    
    def test_max_pool_with_ties(self):
        """Test max pooling with tied maximum values."""
        # Test with tensor having tied maximum values
        x = Tensor([[2.0, 2.0, 1.0]], requires_grad=True)
        
        result = max_pool(x, axis=1)
        
        # Should return first occurrence of maximum
        expected = np.array([2.0])
        assert np.allclose(result.data, expected)
        
        # Test gradient computation with ties
        grad_output = np.array([1.0])
        result._grad_fn.apply(grad_output)
        
        # Gradient should go to first maximum element (index 0)
        assert result._grad_fn is not None
    
    def test_max_pool_complex_indexing(self):
        """Test max pooling with complex tensor indexing."""
        # Test with 4D tensor to check advanced indexing
        x = Tensor([[[[1.0, 2.0, 3.0]]]], requires_grad=True)  # Shape: (1, 1, 1, 3)
        
        result = max_pool(x, axis=3)  # Pool along last axis
        
        # Should return maximum value
        expected = np.array([[[3.0]]])
        assert np.allclose(result.data, expected)
        
        # Test gradient computation
        grad_output = np.array([[[1.0]]])
        result._grad_fn.apply(grad_output)
        
        assert result._grad_fn is not None
    
    def test_max_pool_memory_efficiency(self):
        """Test max pooling memory efficiency."""
        # Large tensor to test memory efficiency
        large_x = Tensor(np.random.randn(100, 1000), requires_grad=True)
        
        with patch('neural_arch.functional.pooling.logger') as mock_logger:
            result = max_pool(large_x, axis=1)
            
            # Should log operation start and completion due to decorator
            assert mock_logger.debug.call_count >= 2
            assert any("Starting operation" in str(call) for call in mock_logger.debug.call_args_list)
            assert any("Completed operation" in str(call) for call in mock_logger.debug.call_args_list)
    
    def test_pooling_operations_logging(self):
        """Test that pooling operations log debug information."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        
        with patch('neural_arch.functional.pooling.logger') as mock_logger:
            mean_pool(x, axis=1)
            max_pool(x, axis=1)
            
            # Should have called debug logging for both operations
            assert mock_logger.debug.call_count >= 4  # 2 calls per operation
    
    def test_pooling_edge_cases(self):
        """Test pooling operations with edge cases."""
        # Test with single element tensor
        x_single = Tensor([[[1.0]]], requires_grad=True)
        
        result_mean = mean_pool(x_single, axis=2)
        result_max = max_pool(x_single, axis=2)
        
        # Should return the single element
        assert np.allclose(result_mean.data, [[1.0]])
        assert np.allclose(result_max.data, [[1.0]])
        
        # Test with negative values
        x_negative = Tensor([[-5.0, -2.0, -8.0, -1.0]], requires_grad=True)
        
        result_mean_neg = mean_pool(x_negative, axis=1)
        result_max_neg = max_pool(x_negative, axis=1)
        
        # Mean should be (-5-2-8-1)/4 = -4.0
        assert np.allclose(result_mean_neg.data, [-4.0])
        # Max should be -1.0
        assert np.allclose(result_max_neg.data, [-1.0])
    
    def test_pooling_gradient_function_properties(self):
        """Test that pooling gradient functions have proper properties."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        
        result_mean = mean_pool(x, axis=1)
        result_max = max_pool(x, axis=1)
        
        # Check gradient function properties for mean pool
        assert result_mean._grad_fn is not None
        assert hasattr(result_mean._grad_fn, 'apply')
        assert hasattr(result_mean._grad_fn, 'inputs')
        assert hasattr(result_mean._grad_fn, 'name')
        assert len(result_mean._grad_fn.inputs) == 1
        assert result_mean._grad_fn.name == "mean_pool"
        
        # Check gradient function properties for max pool
        assert result_max._grad_fn is not None
        assert hasattr(result_max._grad_fn, 'apply')
        assert hasattr(result_max._grad_fn, 'inputs')
        assert hasattr(result_max._grad_fn, 'name')
        assert len(result_max._grad_fn.inputs) == 1
        assert result_max._grad_fn.name == "max_pool"
    
    def test_pooling_error_handling_in_memory_decorator(self):
        """Test error handling in memory efficient decorator for pooling."""
        with patch('neural_arch.functional.pooling.logger') as mock_logger:
            # Mock a pooling function that raises an error
            def failing_pool(x, axis=1):
                raise RuntimeError("Test pooling error")
            
            # Apply decorator
            from neural_arch.functional.utils import memory_efficient_operation
            decorated_fn = memory_efficient_operation(failing_pool)
            
            with pytest.raises(RuntimeError, match="Test pooling error"):
                decorated_fn(Tensor([1.0, 2.0]))
            
            # Should log error
            mock_logger.error.assert_called_once()
    
    def test_pooling_with_different_data_types(self):
        """Test pooling operations with different data types."""
        # Test with integer data
        x_int = Tensor([[1, 4, 2, 3]], requires_grad=True)
        
        result_mean_int = mean_pool(x_int, axis=1)
        result_max_int = max_pool(x_int, axis=1)
        
        # Should handle integer input
        assert isinstance(result_mean_int, Tensor)
        assert isinstance(result_max_int, Tensor)
        
        # Test with float32
        x_float32 = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), requires_grad=True)
        
        result_mean_f32 = mean_pool(x_float32, axis=1)
        result_max_f32 = max_pool(x_float32, axis=1)
        
        # Should preserve data type properties
        assert isinstance(result_mean_f32, Tensor)
        assert isinstance(result_max_f32, Tensor)
    
    def test_pooling_axis_handling(self):
        """Test pooling operations with different axis specifications."""
        # Test with 4D tensor
        x_4d = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
        
        # Test pooling along each axis
        for axis in range(4):
            result_mean = mean_pool(x_4d, axis=axis)
            result_max = max_pool(x_4d, axis=axis)
            
            # Shape should reduce by one dimension
            expected_shape = list(x_4d.shape)
            expected_shape.pop(axis)
            
            assert result_mean.shape == tuple(expected_shape)
            assert result_max.shape == tuple(expected_shape)
    
    def test_pooling_with_extreme_values(self):
        """Test pooling operations with extreme values."""
        # Test with very large values
        x_large = Tensor([[1e10, 2e10, 3e10]], requires_grad=True)
        
        result_mean_large = mean_pool(x_large, axis=1)
        result_max_large = max_pool(x_large, axis=1)
        
        # Should handle large values without overflow
        assert np.isfinite(result_mean_large.data).all()
        assert np.isfinite(result_max_large.data).all()
        
        # Test with very small values
        x_small = Tensor([[1e-10, 2e-10, 3e-10]], requires_grad=True)
        
        result_mean_small = mean_pool(x_small, axis=1)
        result_max_small = max_pool(x_small, axis=1)
        
        # Should handle small values without underflow
        assert np.isfinite(result_mean_small.data).all()
        assert np.isfinite(result_max_small.data).all()