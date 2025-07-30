"""Comprehensive tests for nn.normalization module to improve coverage from 70.27% to 100%.

This file targets LayerNorm and BatchNorm1d classes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.nn.normalization import LayerNorm, BatchNorm1d
from neural_arch.core.tensor import Tensor
from neural_arch.core import Module, Parameter


class TestLayerNorm:
    """Comprehensive tests for LayerNorm."""
    
    def test_layer_norm_init(self):
        """Test LayerNorm initialization."""
        # Default initialization
        norm = LayerNorm(normalized_shape=128)
        
        assert norm.normalized_shape == 128
        assert norm.eps == 1e-5
        assert isinstance(norm, Module)
        
        # Check parameters
        assert isinstance(norm.gamma, Parameter)
        assert isinstance(norm.beta, Parameter)
        
        # Check parameter shapes and values
        assert norm.gamma.shape == (128,)
        assert norm.beta.shape == (128,)
        np.testing.assert_array_equal(norm.gamma.data, np.ones(128))
        np.testing.assert_array_equal(norm.beta.data, np.zeros(128))
        
        # Custom epsilon
        norm2 = LayerNorm(normalized_shape=64, eps=1e-6)
        assert norm2.eps == 1e-6
    
    def test_layer_norm_forward_basic(self):
        """Test LayerNorm forward pass with basic input."""
        norm = LayerNorm(normalized_shape=10)
        
        # Create input with known statistics
        x = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], requires_grad=True)
        
        # Forward pass
        output = norm(x)
        
        # Check output shape
        assert output.shape == (1, 10)
        assert output.requires_grad is True
        
        # Check normalization occurred
        # Mean should be close to 0, std close to 1
        normalized_data = output.data[0]
        assert abs(np.mean(normalized_data)) < 0.1
        assert abs(np.std(normalized_data) - 1.0) < 0.1
    
    def test_layer_norm_forward_with_gradient(self):
        """Test LayerNorm forward pass with gradient computation."""
        norm = LayerNorm(normalized_shape=5)
        
        # Create input
        x = Tensor([[2, 4, 6, 8, 10]], requires_grad=True)
        
        # Forward pass
        output = norm(x)
        
        # Check gradient function is set
        assert hasattr(output, '_backward')
        assert output._backward is not None
        
        # Trigger backward pass
        grad_output = np.ones_like(output.data)
        output.backward(grad_output)
        
        # Call the backward function
        output._backward()
        
        # x should have received gradients
        assert x.grad is not None
    
    def test_layer_norm_forward_no_gradient(self):
        """Test LayerNorm forward pass without gradient tracking."""
        norm = LayerNorm(normalized_shape=5)
        
        # Create input without gradient
        x = Tensor([[1, 2, 3, 4, 5]], requires_grad=False)
        
        # Forward pass
        output = norm(x)
        
        # Output should not require gradient
        assert output.requires_grad is False
        
        # Should not have backward function
        assert not hasattr(output, '_backward') or output._backward is None
    
    def test_layer_norm_2d_input(self):
        """Test LayerNorm with 2D input (batch processing)."""
        norm = LayerNorm(normalized_shape=4)
        
        # Batch of 3 samples
        x = Tensor([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]], requires_grad=True)
        
        # Forward pass
        output = norm(x)
        
        # Check output shape
        assert output.shape == (3, 4)
        
        # Each sample should be normalized independently
        for i in range(3):
            sample_output = output.data[i]
            # Check normalization per sample
            assert abs(np.mean(sample_output)) < 0.1
            assert abs(np.std(sample_output) - 1.0) < 0.1
    
    def test_layer_norm_3d_input(self):
        """Test LayerNorm with 3D input."""
        norm = LayerNorm(normalized_shape=5)
        
        # 3D input: (batch, seq, features)
        x = Tensor(np.random.randn(2, 3, 5), requires_grad=True)
        
        # Forward pass
        output = norm(x)
        
        # Check output shape
        assert output.shape == (2, 3, 5)
        
        # Check normalization along last axis
        for i in range(2):
            for j in range(3):
                normalized_slice = output.data[i, j]
                assert abs(np.mean(normalized_slice)) < 0.1
                assert abs(np.std(normalized_slice) - 1.0) < 0.1
    
    def test_layer_norm_learnable_parameters(self):
        """Test LayerNorm with modified learnable parameters."""
        norm = LayerNorm(normalized_shape=5)
        
        # Modify gamma and beta
        norm.gamma.data = np.array([2, 2, 2, 2, 2], dtype=np.float32)
        norm.beta.data = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        
        # Create input
        x = Tensor([[1, 2, 3, 4, 5]], requires_grad=True)
        
        # Forward pass
        output = norm(x)
        
        # Output should be scaled by gamma and shifted by beta
        # After normalization, multiply by 2 and add 1
        # So values should be roughly in range [-3, 5] instead of [-1.5, 1.5]
        assert np.min(output.data) > -4
        assert np.max(output.data) < 6
        assert np.mean(output.data) > 0.5  # Shifted by beta
    
    def test_layer_norm_numerical_stability(self):
        """Test LayerNorm numerical stability with small variance."""
        norm = LayerNorm(normalized_shape=3)
        
        # Input with very small variance
        x = Tensor([[1.0, 1.0, 1.00001]], requires_grad=True)
        
        # Forward pass - should not crash or produce NaN/inf
        output = norm(x)
        
        assert np.all(np.isfinite(output.data))
        
        # Test with zero variance
        x_const = Tensor([[5.0, 5.0, 5.0]], requires_grad=True)
        output_const = norm(x_const)
        
        # Should handle gracefully (eps prevents division by zero)
        assert np.all(np.isfinite(output_const.data))
    
    def test_layer_norm_parameters_list(self):
        """Test LayerNorm parameters() method."""
        norm = LayerNorm(normalized_shape=10)
        
        # Get parameters
        params = list(norm.parameters())
        
        # Should have gamma and beta
        assert len(params) == 2
        
        # Check parameters are the correct ones
        assert norm.gamma in params
        assert norm.beta in params
        
        # All should require gradients
        for param in params:
            assert param.requires_grad is True
    
    def test_layer_norm_edge_cases(self):
        """Test LayerNorm edge cases."""
        # Very small normalized shape
        norm_small = LayerNorm(normalized_shape=1)
        x_small = Tensor([[5.0]], requires_grad=True)
        output_small = norm_small(x_small)
        # With shape 1, normalization should produce 0 (then scaled/shifted)
        assert output_small.shape == (1, 1)
        
        # Large normalized shape
        norm_large = LayerNorm(normalized_shape=1000)
        x_large = Tensor(np.random.randn(1, 1000), requires_grad=True)
        output_large = norm_large(x_large)
        assert output_large.shape == (1, 1000)
        
        # Negative values
        norm_neg = LayerNorm(normalized_shape=5)
        x_neg = Tensor([[-10, -5, 0, 5, 10]], requires_grad=True)
        output_neg = norm_neg(x_neg)
        assert np.all(np.isfinite(output_neg.data))


class TestBatchNorm1d:
    """Comprehensive tests for BatchNorm1d."""
    
    def test_batch_norm_1d_init(self):
        """Test BatchNorm1d initialization."""
        num_features = 64
        bn = BatchNorm1d(num_features=num_features)
        
        assert bn.num_features == num_features
        assert isinstance(bn, Module)
    
    def test_batch_norm_1d_forward(self):
        """Test BatchNorm1d forward pass."""
        bn = BatchNorm1d(num_features=32)
        
        # Create input (batch_size, features)
        x = Tensor(np.random.randn(10, 32), requires_grad=True)
        
        # Forward pass (currently returns input unchanged)
        output = bn(x)
        
        # Should return input unchanged (placeholder implementation)
        assert output is x
        assert output.shape == (10, 32)
        assert output.requires_grad is True
    
    def test_batch_norm_1d_different_sizes(self):
        """Test BatchNorm1d with different feature sizes."""
        feature_sizes = [1, 16, 64, 128, 512]
        
        for num_features in feature_sizes:
            bn = BatchNorm1d(num_features=num_features)
            
            # Test forward pass
            x = Tensor(np.random.randn(5, num_features), requires_grad=True)
            output = bn(x)
            
            assert output is x
            assert output.shape == (5, num_features)
    
    def test_batch_norm_1d_parameters(self):
        """Test BatchNorm1d parameters."""
        bn = BatchNorm1d(num_features=128)
        
        # Currently no parameters (placeholder implementation)
        params = list(bn.parameters())
        assert len(params) == 0
    
    def test_batch_norm_1d_training_eval_modes(self):
        """Test BatchNorm1d in training and eval modes."""
        bn = BatchNorm1d(num_features=64)
        x = Tensor(np.random.randn(8, 64), requires_grad=True)
        
        # Training mode
        bn.train()
        output_train = bn(x)
        
        # Eval mode
        bn.eval()
        output_eval = bn(x)
        
        # Both should return input (placeholder implementation)
        assert output_train is x
        assert output_eval is x
    
    def test_normalization_integration(self):
        """Test integration of normalization layers."""
        # Create a simple network with normalization
        ln = LayerNorm(normalized_shape=10)
        bn = BatchNorm1d(num_features=10)
        
        # Input data
        x = Tensor(np.random.randn(4, 10), requires_grad=True)
        
        # Pass through layer norm
        x_ln = ln(x)
        
        # Pass through batch norm
        x_bn = bn(x_ln)
        
        # Should maintain shape and gradient tracking
        assert x_bn.shape == (4, 10)
        assert x_bn.requires_grad is True
        
        # Test gradient flow
        x_bn.backward(np.ones_like(x_bn.data))
        if hasattr(x_bn, '_backward'):
            x_bn._backward()
    
    def test_normalization_str_repr(self):
        """Test string representations of normalization layers."""
        ln = LayerNorm(normalized_shape=128)
        bn = BatchNorm1d(num_features=64)
        
        # Should have meaningful representations
        ln_str = str(ln)
        bn_str = str(bn)
        
        assert "LayerNorm" in ln_str or "Module" in ln_str
        assert "BatchNorm1d" in bn_str or "Module" in bn_str