"""Comprehensive tests for nn.attention module to improve coverage from 84.85% to 100%.

This file targets MultiHeadAttention and SelfAttention classes.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core import Module
from neural_arch.core.tensor import Tensor
from neural_arch.nn.attention import MultiHeadAttention, SelfAttention


class TestMultiHeadAttention:
    """Comprehensive tests for MultiHeadAttention."""

    def test_multi_head_attention_init_valid(self):
        """Test MultiHeadAttention initialization with valid parameters."""
        # Test with default parameters
        attn = MultiHeadAttention(d_model=512)
        assert attn.d_model == 512
        assert attn.num_heads == 8
        assert attn.d_k == 64  # 512 / 8
        assert abs(attn.scale - 0.125) < 1e-6  # 1 / sqrt(64)

        # Test with custom parameters
        attn2 = MultiHeadAttention(d_model=256, num_heads=4, dropout=0.2, bias=False)
        assert attn2.d_model == 256
        assert attn2.num_heads == 4
        assert attn2.d_k == 64  # 256 / 4

        # Check linear layers exist
        assert hasattr(attn, "query_proj")
        assert hasattr(attn, "key_proj")
        assert hasattr(attn, "value_proj")
        assert hasattr(attn, "out_proj")

        # Check it's a Module
        assert isinstance(attn, Module)

    def test_multi_head_attention_init_invalid_dimensions(self):
        """Test MultiHeadAttention initialization with invalid dimensions."""
        # d_model not divisible by num_heads
        with pytest.raises(ValueError) as exc_info:
            MultiHeadAttention(d_model=100, num_heads=8)

        assert "d_model (100) must be divisible by num_heads (8)" in str(exc_info.value)

        # Another invalid case
        with pytest.raises(ValueError) as exc_info:
            MultiHeadAttention(d_model=512, num_heads=7)

        assert "d_model (512) must be divisible by num_heads (7)" in str(exc_info.value)

    def test_multi_head_attention_forward_basic(self):
        """Test MultiHeadAttention forward pass with basic input."""
        batch_size = 2
        seq_len = 10
        d_model = 128

        attn = MultiHeadAttention(d_model=d_model, num_heads=8)

        # Create input tensor
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

        # Forward pass
        output = attn(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.requires_grad is True

        # Check gradient flow
        output.backward(np.ones_like(output.data))
        assert x.grad is not None

    def test_multi_head_attention_forward_with_mask(self):
        """Test MultiHeadAttention forward pass with mask."""
        batch_size = 1
        seq_len = 5
        d_model = 64

        attn = MultiHeadAttention(d_model=d_model, num_heads=4)

        # Create input and mask
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        mask = Tensor(np.ones((batch_size, seq_len, seq_len)), requires_grad=False)

        # Forward pass with mask (note: current implementation ignores mask)
        output = attn(x, mask=mask)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.requires_grad is True

    def test_multi_head_attention_different_configurations(self):
        """Test MultiHeadAttention with different configurations."""
        configs = [
            (64, 1),  # Single head
            (128, 2),  # Two heads
            (256, 16),  # Many heads
            (512, 8),  # Standard configuration
        ]

        for d_model, num_heads in configs:
            attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

            # Test forward pass
            x = Tensor(np.random.randn(1, 5, d_model), requires_grad=True)
            output = attn(x)

            assert output.shape == (1, 5, d_model)
            assert output.requires_grad is True

    def test_multi_head_attention_parameters(self):
        """Test MultiHeadAttention parameters."""
        attn = MultiHeadAttention(d_model=128, num_heads=8)

        # Get all parameters
        params = list(attn.parameters())

        # Should have parameters from 4 linear layers
        # Each linear layer has weight and bias (if bias=True)
        assert len(params) >= 4  # At least 4 weight matrices

        # All parameters should require gradients
        for param in params:
            assert param.requires_grad is True
            assert param.grad is None  # Initially no gradients

    def test_multi_head_attention_no_bias(self):
        """Test MultiHeadAttention without bias."""
        attn = MultiHeadAttention(d_model=64, num_heads=4, bias=False)

        # Forward pass should still work
        x = Tensor(np.random.randn(1, 3, 64), requires_grad=True)
        output = attn(x)

        assert output.shape == (1, 3, 64)

        # Check linear layers don't have bias
        # (Implementation detail - the Linear layers should be created with bias=False)
        params = list(attn.parameters())
        # Should have only weight parameters (no bias)
        # 4 linear layers * 1 weight each = 4 parameters
        assert len(params) == 4

    def test_multi_head_attention_gradient_flow(self):
        """Test gradient flow through MultiHeadAttention."""
        d_model = 64
        attn = MultiHeadAttention(d_model=d_model, num_heads=4)

        # Create computation graph
        x = Tensor(np.random.randn(2, 5, d_model), requires_grad=True)
        output = attn(x)

        # Simulate loss - use manual sum
        loss_value = np.sum(output.data)

        # Backward pass with gradient
        grad_output = np.ones_like(output.data)
        output.backward(grad_output)

        # Check gradients exist
        assert x.grad is not None

        # Check all layer parameters have gradients
        for param in attn.parameters():
            assert param.grad is not None

    def test_multi_head_attention_large_input(self):
        """Test MultiHeadAttention with large input."""
        batch_size = 8
        seq_len = 100
        d_model = 512

        attn = MultiHeadAttention(d_model=d_model, num_heads=16)

        # Large input
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

        # Forward should handle large input
        output = attn(x)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_multi_head_attention_training_eval_modes(self):
        """Test MultiHeadAttention in training and eval modes."""
        attn = MultiHeadAttention(d_model=128, num_heads=8)
        x = Tensor(np.random.randn(1, 10, 128), requires_grad=True)

        # Training mode
        attn.train()
        output_train = attn(x)

        # Eval mode
        attn.eval()
        output_eval = attn(x)

        # Both should produce output (dropout not implemented so should be same)
        assert output_train.shape == output_eval.shape


class TestSelfAttention:
    """Comprehensive tests for SelfAttention."""

    def test_self_attention_init(self):
        """Test SelfAttention initialization."""
        d_model = 256
        attn = SelfAttention(d_model=d_model)

        assert attn.d_model == d_model
        assert isinstance(attn, Module)

    def test_self_attention_forward(self):
        """Test SelfAttention forward pass."""
        d_model = 128
        attn = SelfAttention(d_model=d_model)

        # Create input
        batch_size = 2
        seq_len = 10
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

        # Forward pass (currently just returns input)
        output = attn(x)

        # Should return input unchanged
        assert output is x
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.requires_grad is True

    def test_self_attention_different_sizes(self):
        """Test SelfAttention with different model sizes."""
        sizes = [64, 128, 256, 512, 1024]

        for d_model in sizes:
            attn = SelfAttention(d_model=d_model)

            # Test forward
            x = Tensor(np.random.randn(1, 5, d_model), requires_grad=True)
            output = attn(x)

            assert output is x
            assert output.shape[2] == d_model

    def test_self_attention_parameters(self):
        """Test SelfAttention parameters."""
        attn = SelfAttention(d_model=256)

        # Currently no parameters (placeholder implementation)
        params = list(attn.parameters())
        assert len(params) == 0

    def test_self_attention_gradient_flow(self):
        """Test gradient flow through SelfAttention."""
        attn = SelfAttention(d_model=128)

        # Create input
        x = Tensor(np.random.randn(2, 5, 128), requires_grad=True)

        # Forward pass
        output = attn(x)

        # Simulate loss - use manual sum
        loss_value = np.sum(output.data)

        # Backward with gradient
        grad_output = np.ones_like(output.data)
        output.backward(grad_output)

        # Since output is x, gradient should flow directly
        assert x.grad is not None

    def test_self_attention_no_modification(self):
        """Test that SelfAttention doesn't modify input."""
        attn = SelfAttention(d_model=64)

        # Create input
        x_data = np.random.randn(1, 3, 64)
        x = Tensor(x_data.copy(), requires_grad=True)

        # Forward pass
        output = attn(x)

        # Data should be unchanged (within float32 precision)
        np.testing.assert_allclose(output.data, x_data, rtol=1e-6, atol=1e-6)

    def test_attention_modules_str_repr(self):
        """Test string representations of attention modules."""
        mha = MultiHeadAttention(d_model=128, num_heads=8)
        sa = SelfAttention(d_model=128)

        # Should have meaningful string representations
        mha_str = str(mha)
        sa_str = str(sa)

        assert "MultiHeadAttention" in mha_str or "Module" in mha_str
        assert "SelfAttention" in sa_str or "Module" in sa_str
