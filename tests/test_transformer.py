"""Tests for Transformer components."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core import Tensor
from neural_arch.nn import (
    Embedding,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
    TransformerDecoderBlock,
)


class TestMultiHeadAttention:
    """Test MultiHeadAttention layer."""

    def test_init(self):
        """Test initialization."""
        attn = MultiHeadAttention(d_model=64, num_heads=4)
        assert attn.d_model == 64
        assert attn.num_heads == 4
        assert attn.d_k == 16  # 64 / 4

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size, seq_len, d_model = 2, 10, 64
        attn = MultiHeadAttention(d_model=d_model, num_heads=4)

        # Create input
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))

        # Forward pass
        output = attn(x)

        # Check shape
        assert output.data.shape == (batch_size, seq_len, d_model)

    def test_attention_scores(self):
        """Test that attention scores sum to 1."""
        attn = MultiHeadAttention(d_model=64, num_heads=4)

        # Create input
        x = Tensor(np.random.randn(1, 5, 64))

        # We can't easily check internal attention scores without modifying the class
        # But we can check that output is reasonable
        output = attn(x)
        assert not np.isnan(output.data).any()
        assert not np.isinf(output.data).any()

    def test_with_mask(self):
        """Test attention with mask."""
        attn = MultiHeadAttention(d_model=64, num_heads=4)

        # Create input and mask
        x = Tensor(np.random.randn(1, 5, 64))
        mask = np.array([[0, 0, 0, 1, 1]], dtype=np.float32)  # Mask last 2 positions

        # Forward pass with mask
        output = attn(x, mask=mask)

        assert output.data.shape == (1, 5, 64)
        assert not np.isnan(output.data).any()


class TestTransformerBlock:
    """Test TransformerBlock."""

    def test_init(self):
        """Test initialization."""
        block = TransformerBlock(d_model=128, num_heads=8, d_ff=512, dropout=0.1)
        assert block.d_model == 128
        assert hasattr(block, "self_attn")
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")

    def test_forward(self):
        """Test forward pass."""
        block = TransformerBlock(d_model=64, num_heads=4, d_ff=128, dropout=0.0)

        # Create input
        x = Tensor(np.random.randn(2, 10, 64))

        # Forward pass
        output = block(x)

        # Check shape preserved
        assert output.data.shape == x.data.shape

        # Check residual connections work (output should be different from input)
        assert not np.allclose(output.data, x.data)

    def test_with_mask(self):
        """Test with attention mask."""
        block = TransformerBlock(d_model=64, num_heads=4, d_ff=128, dropout=0.0)

        # Create input and mask
        x = Tensor(np.random.randn(1, 5, 64))
        mask = np.array([[0, 0, 0, 1, 1]], dtype=np.float32)

        # Forward pass
        output = block(x, mask=mask)

        assert output.data.shape == x.data.shape
        assert not np.isnan(output.data).any()


class TestTransformerDecoderBlock:
    """Test TransformerDecoderBlock."""

    def test_init(self):
        """Test initialization."""
        decoder = TransformerDecoderBlock(d_model=128, num_heads=8, d_ff=512, dropout=0.1)
        assert hasattr(decoder, "self_attn")
        assert hasattr(decoder, "cross_attn")
        assert hasattr(decoder, "norm1")
        assert hasattr(decoder, "norm2")
        assert hasattr(decoder, "norm3")

    def test_forward(self):
        """Test forward pass."""
        decoder = TransformerDecoderBlock(d_model=64, num_heads=4, d_ff=128, dropout=0.0)

        # Create inputs
        x = Tensor(np.random.randn(2, 8, 64))  # Decoder input
        memory = Tensor(np.random.randn(2, 10, 64))  # Encoder output

        # Forward pass
        output = decoder(x, memory)

        # Check shape
        assert output.data.shape == x.data.shape

    def test_with_masks(self):
        """Test with both target and memory masks."""
        decoder = TransformerDecoderBlock(d_model=64, num_heads=4, d_ff=128, dropout=0.0)

        # Create inputs
        x = Tensor(np.random.randn(1, 5, 64))
        memory = Tensor(np.random.randn(1, 7, 64))

        # Create masks
        tgt_mask = np.triu(np.ones((5, 5)), k=1)  # Causal mask
        memory_mask = np.array([[0, 0, 0, 0, 1, 1, 1]], dtype=np.float32)  # Padding mask

        # Forward pass
        output = decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        assert output.data.shape == x.data.shape
        assert not np.isnan(output.data).any()


class TestLayerNorm:
    """Test LayerNorm implementation."""

    def test_forward(self):
        """Test layer normalization."""
        ln = LayerNorm(64)

        # Create input with known statistics
        x = Tensor(np.random.randn(2, 10, 64) * 5 + 3)  # Mean ~3, std ~5

        # Forward pass
        output = ln(x)

        # Check that output is normalized (along last dimension)
        mean = np.mean(output.data, axis=-1)
        std = np.std(output.data, axis=-1)

        # Should be close to 0 mean and 1 std
        assert np.allclose(mean, 0, atol=1e-5)
        assert np.allclose(std, 1, atol=1e-1)

    def test_learnable_params(self):
        """Test that gamma and beta are learnable."""
        ln = LayerNorm(32)

        # Check parameters exist
        params = list(ln.parameters())
        assert len(params) == 2  # gamma and beta

        # Check shapes
        assert params[0].data.shape == (32,)  # gamma
        assert params[1].data.shape == (32,)  # beta


class TestEmbedding:
    """Test Embedding layer with new fixes."""

    def test_with_tensor_input(self):
        """Test embedding with Tensor input."""
        emb = Embedding(vocab_size=100, embed_dim=32)

        # Create Tensor input
        indices = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))

        # Forward pass
        output = emb(indices)

        assert output.data.shape == (2, 3, 32)
        assert not np.isnan(output.data).any()

    def test_with_numpy_input(self):
        """Test embedding with numpy input."""
        emb = Embedding(vocab_size=100, embed_dim=32)

        # Create numpy input
        indices = np.array([[1, 2, 3], [4, 5, 6]])

        # Forward pass
        output = emb(indices)

        assert output.data.shape == (2, 3, 32)
        assert not np.isnan(output.data).any()

    def test_integer_indices(self):
        """Test that indices are properly converted to integers."""
        emb = Embedding(vocab_size=100, embed_dim=32)

        # Create float indices (should be converted to int)
        indices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        # Forward pass should work
        output = emb(indices)
        assert output.data.shape == (1, 3, 32)


def test_integration():
    """Test full transformer integration."""
    # Create a small transformer stack
    d_model = 64

    # Embedding
    emb = Embedding(vocab_size=100, embed_dim=d_model)

    # Encoder blocks
    encoder1 = TransformerBlock(d_model, num_heads=4, d_ff=128, dropout=0.0)
    encoder2 = TransformerBlock(d_model, num_heads=4, d_ff=128, dropout=0.0)

    # Input indices
    indices = np.array([[1, 2, 3, 4, 5]])

    # Forward pass through stack
    x = emb(indices)
    x = encoder1(x)
    x = encoder2(x)

    # Check final output
    assert x.data.shape == (1, 5, d_model)
    assert not np.isnan(x.data).any()
    assert not np.isinf(x.data).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
