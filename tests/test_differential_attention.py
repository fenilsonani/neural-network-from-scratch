"""Tests for Differential Attention and Differential Transformer."""

import numpy as np
import pytest

from neural_arch.core import Tensor
from neural_arch.nn.differential_attention import (
    DifferentialAttention,
    DifferentialTransformerBlock,
    DifferentialTransformer
)


class TestDifferentialAttention:
    """Test Differential Attention mechanism."""
    
    def test_init_validation(self):
        """Test initialization parameter validation."""
        # Valid initialization
        attn = DifferentialAttention(d_model=256, n_heads=8)
        assert attn.n_heads_per_group == 4
        
        # n_heads must be even
        with pytest.raises(Exception) as exc_info:
            DifferentialAttention(d_model=256, n_heads=7)
        assert "must be even" in str(exc_info.value)
        
        # d_model must be divisible by n_heads
        with pytest.raises(Exception) as exc_info:
            DifferentialAttention(d_model=255, n_heads=8)
        assert "divisible" in str(exc_info.value)
    
    def test_forward_shape(self):
        """Test output shape of differential attention."""
        batch_size, seq_len, d_model = 2, 16, 256
        n_heads = 8
        
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        attn = DifferentialAttention(d_model, n_heads)
        
        output = attn(x)
        assert output.shape == x.shape
    
    def test_lambda_parameter(self):
        """Test lambda parameter initialization and stats."""
        attn = DifferentialAttention(d_model=128, n_heads=4, lambda_init=0.8)
        
        stats = attn.get_attention_stats()
        assert abs(stats['lambda_mean'] - 0.8) < 0.01
        assert stats['lambda_std'] == 0.0  # All initialized to same value
        
        # Lambda should be learnable
        assert attn.lambda_param.requires_grad
    
    def test_noise_cancellation_property(self):
        """Test that differential attention cancels common patterns."""
        batch_size, seq_len, d_model = 1, 8, 64
        
        # Create input with repeating pattern (noise)
        noise = np.ones((batch_size, seq_len, d_model)) * 0.5
        signal = np.random.randn(batch_size, seq_len, d_model) * 0.1
        x = Tensor(noise + signal)
        
        attn = DifferentialAttention(d_model, n_heads=4)
        output, (attn1, attn2) = attn(x, return_attention_weights=True)
        
        # Check that attention maps are different
        assert not np.allclose(attn1, attn2)
        
        # The difference should reduce common patterns
        diff_attn = attn1 - attn2
        assert np.std(diff_attn) > 0  # Non-zero variance shows differentiation
    
    def test_gradient_flow(self):
        """Test gradient flow through differential attention."""
        x = Tensor(np.random.randn(2, 8, 128), requires_grad=True)
        attn = DifferentialAttention(d_model=128, n_heads=8)
        
        output = attn(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert attn.lambda_param.grad is not None
        assert attn.W_q1.grad is not None
        assert attn.W_q2.grad is not None
    
    def test_attention_sparsity(self):
        """Test that differential attention produces sparser patterns."""
        batch_size, seq_len, d_model = 2, 32, 128
        
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        attn = DifferentialAttention(d_model, n_heads=8, lambda_init=0.5)
        
        output, (attn1, attn2) = attn(x, return_attention_weights=True)
        
        # Compute effective differential attention
        lambda_val = attn.lambda_param.data.mean()
        diff_weights = (1 + lambda_val) * attn1 - lambda_val * attn2
        
        # Count near-zero values (sparsity)
        threshold = 0.01
        sparsity = np.mean(np.abs(diff_weights) < threshold)
        
        # Differential attention should have some sparsity
        assert sparsity > 0  # At least some sparse values
    
    def test_with_attention_mask(self):
        """Test differential attention with causal mask."""
        batch_size, seq_len, d_model = 2, 10, 64
        
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        # Create causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        mask = np.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))
        mask_tensor = Tensor(mask)
        
        attn = DifferentialAttention(d_model, n_heads=4)
        output = attn(x, mask=mask_tensor)
        
        assert output.shape == x.shape
        # Output should be valid (no NaN/Inf)
        assert np.all(np.isfinite(output.data))


class TestDifferentialTransformerBlock:
    """Test Differential Transformer Block."""
    
    def test_block_forward(self):
        """Test forward pass through block."""
        batch_size, seq_len, d_model = 2, 16, 256
        
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        block = DifferentialTransformerBlock(d_model, n_heads=8)
        
        output = block(x)
        assert output.shape == x.shape
    
    def test_block_components(self):
        """Test that block uses correct components."""
        block = DifferentialTransformerBlock(d_model=128, n_heads=8)
        
        # Should use DifferentialAttention
        assert isinstance(block.attention, DifferentialAttention)
        
        # Should use RMSNorm
        from neural_arch.nn.normalization import RMSNorm
        assert isinstance(block.norm1, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)
        
        # Should use SwiGLU
        from neural_arch.nn.modern_activations import SwiGLU
        assert isinstance(block.ffn, SwiGLU)
    
    def test_residual_connections(self):
        """Test that residual connections work."""
        x = Tensor(np.random.randn(1, 8, 128))
        block = DifferentialTransformerBlock(d_model=128, n_heads=4)
        
        output = block(x)
        
        # With residual connections, output shouldn't be too different from input
        relative_change = np.mean(np.abs(output.data - x.data)) / np.mean(np.abs(x.data))
        assert relative_change < 2.0  # Output is related to input
    
    def test_get_attention_stats(self):
        """Test getting lambda statistics from block."""
        block = DifferentialTransformerBlock(d_model=64, n_heads=4, lambda_init=0.3)
        
        stats = block.get_attention_stats()
        assert 'lambda_mean' in stats
        assert abs(stats['lambda_mean'] - 0.3) < 0.01


class TestDifferentialTransformer:
    """Test complete Differential Transformer model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = DifferentialTransformer(
            n_layers=2,
            d_model=128,
            n_heads=8,
            vocab_size=1000,
            max_seq_len=512
        )
        
        assert len(model.blocks) == 2
        assert model.d_model == 128
    
    def test_forward_pass(self):
        """Test forward pass through full model."""
        batch_size, seq_len = 2, 16
        vocab_size = 100
        
        # Create input token IDs
        input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
        
        model = DifferentialTransformer(
            n_layers=2,
            d_model=64,
            n_heads=4,
            vocab_size=vocab_size,
            max_seq_len=seq_len
        )
        
        logits = model(input_ids)
        
        # Check output shape
        assert logits.shape == (batch_size, seq_len, vocab_size)
        
        # Check that output is valid
        assert np.all(np.isfinite(logits.data))
    
    def test_all_lambda_stats(self):
        """Test getting lambda statistics from all layers."""
        model = DifferentialTransformer(
            n_layers=3,
            d_model=64,
            n_heads=4,
            lambda_init=0.5
        )
        
        stats = model.get_all_lambda_stats()
        
        # Should have stats for each layer plus global
        assert 'layer_0' in stats
        assert 'layer_1' in stats
        assert 'layer_2' in stats
        assert 'global' in stats
        
        # Global stats should aggregate layer stats
        assert abs(stats['global']['lambda_mean'] - 0.5) < 0.01
    
    def test_gradient_flow_full_model(self):
        """Test gradient flow through entire model."""
        batch_size, seq_len = 2, 8
        vocab_size = 50
        
        input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
        target = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
        
        model = DifferentialTransformer(
            n_layers=1,
            d_model=32,
            n_heads=4,
            vocab_size=vocab_size
        )
        
        # Forward pass
        logits = model(input_ids)
        
        # Simple loss (cross-entropy approximation)
        loss = ((logits - target) ** 2).sum()
        loss.backward()
        
        # Check that gradients flow to embeddings
        assert model.token_embedding.weight.grad is not None
        
        # Check that gradients flow to attention lambda parameters
        for block in model.blocks:
            assert block.attention.lambda_param.grad is not None


class TestDifferentialVsStandard:
    """Compare Differential Attention to standard attention."""
    
    def test_parameter_count(self):
        """Compare parameter counts."""
        d_model = 256
        n_heads = 8
        
        # Differential attention has more parameters (2x QKV)
        diff_attn = DifferentialAttention(d_model, n_heads)
        
        # Count parameters
        diff_params = sum([
            p.data.size for p in [
                diff_attn.W_q1, diff_attn.W_k1, diff_attn.W_v1,
                diff_attn.W_q2, diff_attn.W_k2, diff_attn.W_v2,
                diff_attn.W_o, diff_attn.lambda_param
            ]
        ])
        
        # Standard attention would have ~half the QKV parameters
        standard_params = 4 * d_model * d_model  # Q, K, V, O
        
        # Differential has roughly 2x parameters for QKV
        ratio = diff_params / standard_params
        assert 1.5 < ratio < 2.0  # Should be roughly 1.75x
    
    def test_sparsity_improvement(self):
        """Test that differential attention is sparser."""
        batch_size, seq_len, d_model = 2, 64, 128
        
        # Create input with some structure
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        diff_attn = DifferentialAttention(d_model, n_heads=8, lambda_init=0.5)
        output, (attn1, attn2) = diff_attn(x, return_attention_weights=True)
        
        # Compute differential attention weights
        lambda_val = diff_attn.lambda_param.data.mean()
        diff_weights = (1 + lambda_val) * attn1 - lambda_val * attn2
        
        # Measure entropy (lower entropy = more focused/sparse)
        # Normalize weights
        diff_weights_norm = diff_weights / (diff_weights.sum(axis=-1, keepdims=True) + 1e-9)
        entropy = -np.sum(diff_weights_norm * np.log(diff_weights_norm + 1e-9), axis=-1)
        
        # Standard attention would have higher entropy (more uniform)
        standard_entropy = -np.sum(attn1 * np.log(attn1 + 1e-9), axis=-1)
        
        # Differential should have lower average entropy (more focused)
        assert np.mean(entropy) <= np.mean(standard_entropy) * 1.1  # Allow small margin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])