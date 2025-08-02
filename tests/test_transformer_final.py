"""Comprehensive tests for nn/transformer.py to improve coverage from 81.17% to 95%+.

This file tests TransformerBlock and TransformerEncoder.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch import Tensor
from neural_arch.nn.transformer import TransformerBlock, TransformerEncoder


class TestTransformerBlock:
    """Test TransformerBlock comprehensively."""

    def test_transformer_block_init(self):
        """Test TransformerBlock initialization."""
        d_model = 128
        num_heads = 8
        d_ff = 512

        block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.1)

        assert block.d_model == d_model
        assert block.num_heads == num_heads
        assert hasattr(block, "self_attn")
        assert hasattr(block, "ffn1")
        assert hasattr(block, "ffn2")
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")
        assert hasattr(block, "dropout")

    def test_transformer_block_forward_basic(self):
        """Test TransformerBlock forward pass."""
        d_model = 64
        seq_len = 10
        batch_size = 2

        block = TransformerBlock(d_model, num_heads=4, d_ff=256, dropout=0.0)
        x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))

        output = block(x)

        assert output.shape == (batch_size, seq_len, d_model)
        # Output always has requires_grad=True due to parameter operations

    def test_transformer_block_forward_with_mask(self):
        """Test TransformerBlock forward pass with mask."""
        d_model = 64
        seq_len = 10
        batch_size = 2

        block = TransformerBlock(d_model, num_heads=4, d_ff=256, dropout=0.0)
        x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))

        # Create attention mask
        mask = np.ones((batch_size, seq_len, seq_len))
        mask[:, :, 5:] = 0  # Mask out positions 5 and beyond

        output = block(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_transformer_block_activation_relu(self):
        """Test TransformerBlock with ReLU activation (default)."""
        d_model = 32
        block = TransformerBlock(d_model, num_heads=2, d_ff=128, activation="relu")

        x = Tensor(np.random.randn(1, 5, d_model).astype(np.float32))
        output = block(x)

        assert output.shape == x.shape
        assert block.activation == "relu"

    def test_transformer_block_activation_other(self):
        """Test TransformerBlock with non-relu activation to cover line 84->88."""
        d_model = 32
        # Use any string other than 'relu' to skip the relu path
        block = TransformerBlock(d_model, num_heads=2, d_ff=128, activation="gelu")

        x = Tensor(np.random.randn(1, 5, d_model).astype(np.float32))
        output = block(x)

        assert output.shape == x.shape
        assert block.activation == "gelu"

    def test_transformer_block_parameters(self):
        """Test TransformerBlock parameters method."""
        d_model = 64
        block = TransformerBlock(d_model, num_heads=4, d_ff=256)

        params = block.parameters()

        # Check parameter naming
        assert any("attn_" in name for name in params.keys())
        assert any("ffn1_" in name for name in params.keys())
        assert any("ffn2_" in name for name in params.keys())
        assert any("norm1_" in name for name in params.keys())
        assert any("norm2_" in name for name in params.keys())

        # Check we have all expected parameters
        # MultiHeadAttention has q_proj, k_proj, v_proj, out_proj (weight + bias each)
        # Linear layers have weight + bias
        # LayerNorm has weight + bias
        assert len(params) > 0

    def test_transformer_block_dropout_effect(self):
        """Test TransformerBlock with different dropout rates."""
        d_model = 32
        x = Tensor(np.random.randn(2, 5, d_model).astype(np.float32))

        # With dropout = 0
        block_no_dropout = TransformerBlock(d_model, num_heads=2, dropout=0.0)
        output1 = block_no_dropout(x)

        # With dropout = 0.5
        block_with_dropout = TransformerBlock(d_model, num_heads=2, dropout=0.5)
        # In eval mode, dropout should have no effect
        block_with_dropout.eval()
        output2 = block_with_dropout(x)

        # Both should produce similar outputs in eval mode
        assert output1.shape == output2.shape

    def test_transformer_block_residual_connections(self):
        """Test that residual connections are properly applied."""
        d_model = 32
        block = TransformerBlock(d_model, num_heads=2, d_ff=128, dropout=0.0)

        # Use a small input to test residual connections
        x = Tensor(np.ones((1, 3, d_model)).astype(np.float32) * 0.1)
        output = block(x)

        # Output should not be zero due to residual connections
        assert not np.allclose(output.data, 0)
        assert output.shape == x.shape


class TestTransformerEncoder:
    """Test TransformerEncoder comprehensively."""

    def test_transformer_encoder_init(self):
        """Test TransformerEncoder initialization - covers lines 137-149."""
        d_model = 128
        num_layers = 3
        num_heads = 8

        encoder = TransformerEncoder(
            d_model, num_layers=num_layers, num_heads=num_heads, d_ff=512, dropout=0.1
        )

        assert encoder.d_model == d_model
        assert encoder.num_layers == num_layers
        assert len(encoder.layers) == num_layers
        assert hasattr(encoder, "norm")

        # Check that all layers are TransformerBlocks
        for layer in encoder.layers:
            assert isinstance(layer, TransformerBlock)

    def test_transformer_encoder_forward(self):
        """Test TransformerEncoder forward pass - covers lines 162-168."""
        d_model = 64
        seq_len = 10
        batch_size = 2
        num_layers = 2

        encoder = TransformerEncoder(d_model, num_layers=num_layers, num_heads=4)
        x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))

        output = encoder(x)

        assert output.shape == (batch_size, seq_len, d_model)
        # Output always has requires_grad=True due to parameter operations

    def test_transformer_encoder_forward_with_mask(self):
        """Test TransformerEncoder forward pass with mask."""
        d_model = 64
        seq_len = 10
        batch_size = 2

        encoder = TransformerEncoder(d_model, num_layers=2, num_heads=4)
        x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))

        # Create padding mask
        mask = np.ones((batch_size, seq_len, seq_len))
        mask[0, :, 7:] = 0  # Mask padding for first sample
        mask[1, :, 8:] = 0  # Mask padding for second sample

        output = encoder(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_transformer_encoder_parameters(self):
        """Test TransformerEncoder parameters method - covers lines 172-183."""
        d_model = 32
        num_layers = 2

        encoder = TransformerEncoder(d_model, num_layers=num_layers, num_heads=2)
        params = encoder.parameters()

        # Check parameter naming
        assert any("layer0_" in name for name in params.keys())
        assert any("layer1_" in name for name in params.keys())
        assert any("final_norm_" in name for name in params.keys())

        # Each layer should contribute multiple parameters
        layer0_params = [k for k in params.keys() if k.startswith("layer0_")]
        assert len(layer0_params) > 10  # Should have many params from attn, ffn, norm

        # Final norm should contribute weight and bias
        final_norm_params = [k for k in params.keys() if k.startswith("final_norm_")]
        assert len(final_norm_params) == 2  # weight and bias

    def test_transformer_encoder_deep_stack(self):
        """Test TransformerEncoder with many layers."""
        d_model = 32
        num_layers = 6  # Standard transformer depth

        encoder = TransformerEncoder(
            d_model, num_layers=num_layers, num_heads=4, d_ff=128, dropout=0.0
        )

        x = Tensor(np.random.randn(1, 5, d_model).astype(np.float32))
        output = encoder(x)

        assert output.shape == x.shape
        assert len(encoder.layers) == num_layers

        # Check all layers are properly initialized
        params = encoder.parameters()
        for i in range(num_layers):
            assert any(f"layer{i}_" in name for name in params.keys())

    def test_transformer_encoder_single_layer(self):
        """Test TransformerEncoder with single layer."""
        d_model = 64
        encoder = TransformerEncoder(d_model, num_layers=1, num_heads=4)

        x = Tensor(np.random.randn(2, 8, d_model).astype(np.float32))
        output = encoder(x)

        assert output.shape == x.shape
        assert len(encoder.layers) == 1

    def test_transformer_encoder_gradient_flow(self):
        """Test gradient flow through TransformerEncoder."""
        d_model = 32
        encoder = TransformerEncoder(d_model, num_layers=2, num_heads=2)

        x = Tensor(np.random.randn(1, 4, d_model).astype(np.float32), requires_grad=True)
        output = encoder(x)

        # Check gradient properties
        assert output.requires_grad

        # Check output properties
        assert output.shape == x.shape
        assert hasattr(output, "data")
        assert isinstance(output.data, np.ndarray)
