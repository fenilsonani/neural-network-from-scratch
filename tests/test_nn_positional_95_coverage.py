"""Ultra-comprehensive tests for Positional Encoding modules to achieve 95%+ test coverage.

This test suite covers all positional encoding implementations including SinusoidalPositionalEncoding,
RotaryPositionalEmbedding (RoPE), and LearnedPositionalEmbedding to ensure robust 95%+ test coverage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core.tensor import Tensor
from neural_arch.exceptions import LayerError
from neural_arch.nn.positional import (
    LearnedPE,
    LearnedPositionalEmbedding,
    RoPE,
    RotaryPositionalEmbedding,
    SinusoidalPE,
    SinusoidalPositionalEncoding,
    create_rope,
    create_sinusoidal_pe,
)


class TestSinusoidalPositionalEncoding95Coverage:
    """Comprehensive SinusoidalPositionalEncoding tests targeting 95%+ coverage."""

    def test_sinusoidal_pe_initialization_comprehensive(self):
        """Test all SinusoidalPositionalEncoding initialization parameters."""
        # Test basic initialization
        pe = SinusoidalPositionalEncoding(d_model=512)
        assert pe.d_model == 512
        assert pe.max_len == 5000  # default
        assert pe.dropout_rate == 0.0  # default
        assert pe.base == 10000.0  # default
        assert pe.pe.shape == (5000, 512)

        # Test custom parameters
        pe_custom = SinusoidalPositionalEncoding(
            d_model=256, max_len=1000, dropout=0.1, base=8000.0
        )
        assert pe_custom.d_model == 256
        assert pe_custom.max_len == 1000
        assert pe_custom.dropout_rate == 0.1
        assert pe_custom.base == 8000.0
        assert pe_custom.pe.shape == (1000, 256)

    def test_sinusoidal_pe_initialization_validation(self):
        """Test SinusoidalPositionalEncoding initialization validation."""
        # Test odd d_model (should fail)
        with pytest.raises(LayerError):
            SinusoidalPositionalEncoding(d_model=513)  # Odd number

        # Test invalid d_model
        with pytest.raises(LayerError):
            SinusoidalPositionalEncoding(d_model=0)
        with pytest.raises(LayerError):
            SinusoidalPositionalEncoding(d_model=-1)

        # Test invalid max_len
        with pytest.raises(LayerError):
            SinusoidalPositionalEncoding(d_model=512, max_len=0)
        with pytest.raises(LayerError):
            SinusoidalPositionalEncoding(d_model=512, max_len=-1)

    def test_sinusoidal_pe_precomputed_values(self):
        """Test precomputed positional encoding values."""
        pe = SinusoidalPositionalEncoding(d_model=4, max_len=10)

        # Test that even positions use sin, odd positions use cos
        pos_0 = pe.pe[0]  # Position 0
        pos_1 = pe.pe[1]  # Position 1

        # Verify alternating sin/cos pattern
        assert pos_0[0] == 0.0  # sin(0) = 0
        assert pos_0[1] == 1.0  # cos(0) = 1

        # Values should be finite and in reasonable range
        assert np.all(np.isfinite(pe.pe))
        assert np.all(np.abs(pe.pe) <= 1.0)  # Sin/cos values should be [-1, 1]

    def test_sinusoidal_pe_mathematical_correctness(self):
        """Test mathematical correctness of sinusoidal PE."""
        d_model = 8
        max_len = 5
        pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        # Manual calculation for verification
        for pos in range(max_len):
            for i in range(d_model // 2):
                div_term = np.exp(i * -(np.log(10000.0) / d_model))

                # Even indices (sin)
                expected_sin = np.sin(pos * div_term)
                actual_sin = pe.pe[pos, 2 * i]
                assert np.isclose(actual_sin, expected_sin, rtol=1e-6)

                # Odd indices (cos)
                expected_cos = np.cos(pos * div_term)
                actual_cos = pe.pe[pos, 2 * i + 1]
                assert np.isclose(actual_cos, expected_cos, rtol=1e-6)

    def test_sinusoidal_pe_forward_pass_comprehensive(self):
        """Test SinusoidalPositionalEncoding forward pass."""
        pe = SinusoidalPositionalEncoding(d_model=128, max_len=100)

        # Test different input shapes
        test_configs = [
            (1, 10, 128),  # Single sample
            (4, 20, 128),  # Small batch
            (8, 50, 128),  # Larger batch
            (2, 1, 128),  # Single token
        ]

        for batch_size, seq_len, d_model in test_configs:
            x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
            output = pe.forward(x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert isinstance(output, Tensor)
            assert np.all(np.isfinite(output.data))

            # Output should be input + positional encoding
            expected = x.data + pe.pe[:seq_len][None, :, :]
            assert np.allclose(output.data, expected, rtol=1e-6)

    def test_sinusoidal_pe_forward_with_start_pos(self):
        """Test SinusoidalPositionalEncoding with start_pos parameter."""
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)

        x = Tensor(np.random.randn(2, 10, 64).astype(np.float32))

        # Test with different start positions
        for start_pos in [0, 5, 10, 25]:
            output = pe.forward(x, start_pos=start_pos)

            assert output.shape == x.shape

            # Verify correct positional encoding slice is used
            expected_pe = pe.pe[start_pos : start_pos + 10]
            expected = x.data + expected_pe[None, :, :]
            assert np.allclose(output.data, expected, rtol=1e-6)

    def test_sinusoidal_pe_input_validation(self):
        """Test SinusoidalPositionalEncoding input validation."""
        pe = SinusoidalPositionalEncoding(d_model=128, max_len=50)

        # Test wrong d_model
        x_wrong_dim = Tensor(np.random.randn(2, 10, 64).astype(np.float32))
        with pytest.raises(LayerError):
            pe.forward(x_wrong_dim)

        # Test sequence too long
        x_too_long = Tensor(np.random.randn(2, 60, 128).astype(np.float32))
        with pytest.raises(LayerError):
            pe.forward(x_too_long)

        # Test with start_pos causing overflow
        x = Tensor(np.random.randn(2, 10, 128).astype(np.float32))
        with pytest.raises(LayerError):
            pe.forward(x, start_pos=45)  # 45 + 10 > 50 (max_len)

    def test_sinusoidal_pe_gradient_computation(self):
        """Test SinusoidalPositionalEncoding gradient computation."""
        pe = SinusoidalPositionalEncoding(d_model=64)

        x = Tensor(np.random.randn(2, 10, 64).astype(np.float32), requires_grad=True)
        output = pe.forward(x)

        assert output.requires_grad is True
        assert output._grad_fn is not None

    def test_sinusoidal_pe_with_dropout(self):
        """Test SinusoidalPositionalEncoding with dropout."""
        try:
            pe_with_dropout = SinusoidalPositionalEncoding(d_model=128, dropout=0.1)

            x = Tensor(np.random.randn(2, 10, 128).astype(np.float32))

            # Test in training mode (if dropout is available)
            if hasattr(pe_with_dropout, "dropout") and pe_with_dropout.dropout is not None:
                output = pe_with_dropout.forward(x)
                assert output.shape == x.shape
            else:
                # If dropout not implemented, should still work
                output = pe_with_dropout.forward(x)
                assert output.shape == x.shape

        except (ImportError, AttributeError):
            # Dropout might not be implemented
            pytest.skip("Dropout not available")


class TestRotaryPositionalEmbedding95Coverage:
    """Comprehensive RotaryPositionalEmbedding tests targeting 95%+ coverage."""

    def test_rope_initialization_comprehensive(self):
        """Test all RoPE initialization parameters."""
        # Test basic initialization
        rope = RotaryPositionalEmbedding(dim=64)
        assert rope.dim == 64
        assert rope.max_seq_len == 2048  # default
        assert rope.base == 10000.0  # default
        assert rope.precision == "float32"  # default
        assert rope.inv_freq.shape == (32,)  # dim // 2

        # Test custom parameters
        rope_custom = RotaryPositionalEmbedding(
            dim=128, max_seq_len=4096, base=8000.0, precision="float64"
        )
        assert rope_custom.dim == 128
        assert rope_custom.max_seq_len == 4096
        assert rope_custom.base == 8000.0
        assert rope_custom.precision == "float64"
        assert rope_custom.inv_freq.shape == (64,)  # dim // 2

    def test_rope_initialization_validation(self):
        """Test RoPE initialization validation."""
        # Test odd dimension (should fail)
        with pytest.raises(LayerError):
            RotaryPositionalEmbedding(dim=65)  # Odd number

        # Test invalid dimensions
        with pytest.raises(LayerError):
            RotaryPositionalEmbedding(dim=0)
        with pytest.raises(LayerError):
            RotaryPositionalEmbedding(dim=-1)

        # Test invalid max_seq_len
        with pytest.raises(LayerError):
            RotaryPositionalEmbedding(dim=64, max_seq_len=0)
        with pytest.raises(LayerError):
            RotaryPositionalEmbedding(dim=64, max_seq_len=-1)

    def test_rope_precomputation(self):
        """Test RoPE precomputation of rotation matrices."""
        rope = RotaryPositionalEmbedding(dim=8, max_seq_len=10)

        # Check cached values are created
        assert hasattr(rope, "cos_cached")
        assert hasattr(rope, "sin_cached")
        assert rope.cos_cached.shape == (10, 4)  # (max_seq_len, dim//2)
        assert rope.sin_cached.shape == (10, 4)

        # Check values are finite and in expected range
        assert np.all(np.isfinite(rope.cos_cached))
        assert np.all(np.isfinite(rope.sin_cached))
        assert np.all(np.abs(rope.cos_cached) <= 1.0)
        assert np.all(np.abs(rope.sin_cached) <= 1.0)

    def test_rope_rotate_half_function(self):
        """Test RoPE rotate_half helper function."""
        rope = RotaryPositionalEmbedding(dim=8)

        # Test with simple input
        x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
        rotated = rope.rotate_half(x)

        # Expected: [-2, 1, -4, 3, -6, 5, -8, 7]
        expected = np.array([[-2, 1, -4, 3, -6, 5, -8, 7]], dtype=np.float32)
        assert np.allclose(rotated, expected)

        # Test with batch
        x_batch = np.random.randn(2, 3, 8).astype(np.float32)
        rotated_batch = rope.rotate_half(x_batch)
        assert rotated_batch.shape == x_batch.shape

        # Verify pattern for each position
        for i in range(2):
            for j in range(3):
                x_slice = x_batch[i, j]
                rot_slice = rotated_batch[i, j]
                for k in range(0, 8, 2):
                    assert rot_slice[k] == -x_slice[k + 1]
                    assert rot_slice[k + 1] == x_slice[k]

    def test_rope_apply_rope_function(self):
        """Test RoPE apply_rope function."""
        rope = RotaryPositionalEmbedding(dim=8, max_seq_len=20)

        # Test with simple input
        x = np.random.randn(2, 5, 8).astype(np.float32)
        seq_len = 5

        rotated = rope.apply_rope(x, seq_len)

        assert rotated.shape == x.shape
        assert np.all(np.isfinite(rotated))

        # Test with different start positions
        rotated_start = rope.apply_rope(x, seq_len, start_pos=3)
        assert rotated_start.shape == x.shape
        assert np.all(np.isfinite(rotated_start))

        # Results should be different for different start positions
        assert not np.allclose(rotated, rotated_start)

    def test_rope_apply_rope_sequence_extension(self):
        """Test RoPE sequence extension beyond precomputed length."""
        rope = RotaryPositionalEmbedding(dim=8, max_seq_len=10)

        # Test with sequence longer than precomputed
        x = np.random.randn(1, 15, 8).astype(np.float32)

        rotated = rope.apply_rope(x, seq_len=15)

        assert rotated.shape == x.shape
        assert np.all(np.isfinite(rotated))

        # Check that cached values were extended
        assert rope.cos_cached.shape[0] >= 15
        assert rope.sin_cached.shape[0] >= 15

    def test_rope_forward_pass_comprehensive(self):
        """Test RoPE forward pass with query and key tensors."""
        rope = RotaryPositionalEmbedding(dim=64)

        # Test different input configurations
        test_configs = [
            (1, 10, 64),  # Single sample
            (4, 20, 64),  # Small batch
            (2, 50, 64),  # Longer sequence
            (8, 5, 64),  # Larger batch, short sequence
        ]

        for batch_size, seq_len, dim in test_configs:
            q = Tensor(np.random.randn(batch_size, seq_len, dim).astype(np.float32))
            k = Tensor(np.random.randn(batch_size, seq_len, dim).astype(np.float32))

            q_rot, k_rot = rope.forward(q, k)

            assert q_rot.shape == q.shape
            assert k_rot.shape == k.shape
            assert isinstance(q_rot, Tensor)
            assert isinstance(k_rot, Tensor)
            assert np.all(np.isfinite(q_rot.data))
            assert np.all(np.isfinite(k_rot.data))

    def test_rope_forward_with_start_pos(self):
        """Test RoPE forward pass with start_pos parameter."""
        rope = RotaryPositionalEmbedding(dim=32, max_seq_len=100)

        q = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
        k = Tensor(np.random.randn(2, 10, 32).astype(np.float32))

        # Test with different start positions
        for start_pos in [0, 5, 10, 25]:
            q_rot, k_rot = rope.forward(q, k, start_pos=start_pos)

            assert q_rot.shape == q.shape
            assert k_rot.shape == k.shape
            assert np.all(np.isfinite(q_rot.data))
            assert np.all(np.isfinite(k_rot.data))

    def test_rope_forward_4d_input(self):
        """Test RoPE with 4D input (batch, heads, seq_len, dim)."""
        rope = RotaryPositionalEmbedding(dim=64)

        # 4D input: (batch, num_heads, seq_len, head_dim)
        q_4d = Tensor(np.random.randn(2, 8, 20, 64).astype(np.float32))
        k_4d = Tensor(np.random.randn(2, 8, 20, 64).astype(np.float32))

        q_rot, k_rot = rope.forward(q_4d, k_4d)

        assert q_rot.shape == q_4d.shape
        assert k_rot.shape == k_4d.shape
        assert np.all(np.isfinite(q_rot.data))
        assert np.all(np.isfinite(k_rot.data))

    def test_rope_input_validation(self):
        """Test RoPE input validation."""
        rope = RotaryPositionalEmbedding(dim=64)

        q = Tensor(np.random.randn(2, 10, 64).astype(np.float32))
        k = Tensor(np.random.randn(2, 10, 64).astype(np.float32))

        # Test mismatched shapes
        k_wrong = Tensor(np.random.randn(2, 5, 64).astype(np.float32))
        with pytest.raises(LayerError):
            rope.forward(q, k_wrong)

        # Test wrong last dimension
        q_wrong_dim = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
        k_wrong_dim = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
        with pytest.raises(LayerError):
            rope.forward(q_wrong_dim, k_wrong_dim)

    def test_rope_gradient_computation(self):
        """Test RoPE gradient computation setup."""
        rope = RotaryPositionalEmbedding(dim=32)

        q = Tensor(np.random.randn(2, 10, 32).astype(np.float32), requires_grad=True)
        k = Tensor(np.random.randn(2, 10, 32).astype(np.float32), requires_grad=True)

        q_rot, k_rot = rope.forward(q, k)

        assert q_rot.requires_grad is True
        assert k_rot.requires_grad is True
        assert q_rot._grad_fn is not None
        assert k_rot._grad_fn is not None

    def test_rope_extra_repr(self):
        """Test RoPE string representation."""
        rope = RotaryPositionalEmbedding(dim=128, max_seq_len=1024, base=8000.0)
        repr_str = rope.extra_repr()

        assert "dim=128" in repr_str
        assert "max_seq_len=1024" in repr_str
        assert "base=8000.0" in repr_str


class TestLearnedPositionalEmbedding95Coverage:
    """Comprehensive LearnedPositionalEmbedding tests targeting 95%+ coverage."""

    def test_learned_pe_initialization_comprehensive(self):
        """Test all LearnedPositionalEmbedding initialization parameters."""
        # Test basic initialization
        learned_pe = LearnedPositionalEmbedding(max_len=1000, d_model=512)
        assert learned_pe.max_len == 1000
        assert learned_pe.d_model == 512
        assert learned_pe.dropout_rate == 0.0  # default
        assert learned_pe.embedding.data.shape == (1000, 512)

        # Test with dropout
        learned_pe_dropout = LearnedPositionalEmbedding(max_len=500, d_model=256, dropout=0.1)
        assert learned_pe_dropout.max_len == 500
        assert learned_pe_dropout.d_model == 256
        assert learned_pe_dropout.dropout_rate == 0.1

    def test_learned_pe_initialization_validation(self):
        """Test LearnedPositionalEmbedding initialization validation."""
        # Test invalid max_len
        with pytest.raises(LayerError):
            LearnedPositionalEmbedding(max_len=0, d_model=512)
        with pytest.raises(LayerError):
            LearnedPositionalEmbedding(max_len=-1, d_model=512)

        # Test invalid d_model
        with pytest.raises(LayerError):
            LearnedPositionalEmbedding(max_len=1000, d_model=0)
        with pytest.raises(LayerError):
            LearnedPositionalEmbedding(max_len=1000, d_model=-1)

    def test_learned_pe_forward_pass_comprehensive(self):
        """Test LearnedPositionalEmbedding forward pass."""
        learned_pe = LearnedPositionalEmbedding(max_len=100, d_model=128)

        # Test different input shapes
        test_configs = [
            (1, 10, 128),  # Single sample
            (4, 20, 128),  # Small batch
            (8, 50, 128),  # Larger batch
            (2, 1, 128),  # Single token
        ]

        for batch_size, seq_len, d_model in test_configs:
            x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
            output = learned_pe.forward(x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert isinstance(output, Tensor)
            assert np.all(np.isfinite(output.data))

            # Output should be input + learned positional embedding
            pe_slice = learned_pe.embedding.data[:seq_len]
            expected = x.data + pe_slice[None, :, :]
            assert np.allclose(output.data, expected, rtol=1e-6)

    def test_learned_pe_forward_with_start_pos(self):
        """Test LearnedPositionalEmbedding with start_pos parameter."""
        learned_pe = LearnedPositionalEmbedding(max_len=100, d_model=64)

        x = Tensor(np.random.randn(2, 10, 64).astype(np.float32))

        # Test with different start positions
        for start_pos in [0, 5, 10, 25]:
            output = learned_pe.forward(x, start_pos=start_pos)

            assert output.shape == x.shape

            # Verify correct embedding slice is used
            pe_slice = learned_pe.embedding.data[start_pos : start_pos + 10]
            expected = x.data + pe_slice[None, :, :]
            assert np.allclose(output.data, expected, rtol=1e-6)

    def test_learned_pe_input_validation(self):
        """Test LearnedPositionalEmbedding input validation."""
        learned_pe = LearnedPositionalEmbedding(max_len=50, d_model=128)

        # Test wrong d_model
        x_wrong_dim = Tensor(np.random.randn(2, 10, 64).astype(np.float32))
        with pytest.raises(LayerError):
            learned_pe.forward(x_wrong_dim)

        # Test sequence too long
        x_too_long = Tensor(np.random.randn(2, 60, 128).astype(np.float32))
        with pytest.raises(LayerError):
            learned_pe.forward(x_too_long)

        # Test with start_pos causing overflow
        x = Tensor(np.random.randn(2, 10, 128).astype(np.float32))
        with pytest.raises(LayerError):
            learned_pe.forward(x, start_pos=45)  # 45 + 10 > 50 (max_len)

    def test_learned_pe_gradient_computation(self):
        """Test LearnedPositionalEmbedding gradient computation."""
        learned_pe = LearnedPositionalEmbedding(max_len=100, d_model=64)

        x = Tensor(np.random.randn(2, 10, 64).astype(np.float32), requires_grad=True)
        output = learned_pe.forward(x)

        assert output.requires_grad is True
        assert output._grad_fn is not None

    def test_learned_pe_parameter_access(self):
        """Test LearnedPositionalEmbedding parameter access."""
        learned_pe = LearnedPositionalEmbedding(max_len=100, d_model=64)

        # Test embedding parameter
        assert hasattr(learned_pe, "embedding")
        assert learned_pe.embedding.requires_grad is True
        assert learned_pe.embedding.data.shape == (100, 64)
        assert "learned_pe.embedding" in learned_pe.embedding.name

    def test_learned_pe_extra_repr(self):
        """Test LearnedPositionalEmbedding string representation."""
        learned_pe = LearnedPositionalEmbedding(max_len=500, d_model=256)
        repr_str = learned_pe.extra_repr()

        assert "max_len=500" in repr_str
        assert "d_model=256" in repr_str


class TestPositionalEncodingAliases:
    """Test positional encoding aliases and convenience functions."""

    def test_aliases_exist(self):
        """Test that all aliases are properly defined."""
        # Test class aliases
        assert RoPE is RotaryPositionalEmbedding
        assert SinusoidalPE is SinusoidalPositionalEncoding
        assert LearnedPE is LearnedPositionalEmbedding

        # Test that aliases work
        rope = RoPE(dim=64)
        assert isinstance(rope, RotaryPositionalEmbedding)

        sin_pe = SinusoidalPE(d_model=128)
        assert isinstance(sin_pe, SinusoidalPositionalEncoding)

        learned_pe = LearnedPE(max_len=100, d_model=64)
        assert isinstance(learned_pe, LearnedPositionalEmbedding)

    def test_create_rope_function(self):
        """Test create_rope convenience function."""
        # Test basic creation
        rope = create_rope(dim=64)
        assert isinstance(rope, RotaryPositionalEmbedding)
        assert rope.dim == 64
        assert rope.max_seq_len == 2048  # default
        assert rope.base == 10000.0  # default

        # Test with custom parameters
        rope_custom = create_rope(dim=128, max_seq_len=4096, base=8000.0)
        assert rope_custom.dim == 128
        assert rope_custom.max_seq_len == 4096
        assert rope_custom.base == 8000.0

    def test_create_sinusoidal_pe_function(self):
        """Test create_sinusoidal_pe convenience function."""
        # Test basic creation
        pe = create_sinusoidal_pe(d_model=512)
        assert isinstance(pe, SinusoidalPositionalEncoding)
        assert pe.d_model == 512
        assert pe.max_len == 5000  # default

        # Test with custom parameters
        pe_custom = create_sinusoidal_pe(d_model=256, max_len=1000)
        assert pe_custom.d_model == 256
        assert pe_custom.max_len == 1000


class TestPositionalEncodingIntegration:
    """Integration tests for positional encoding modules."""

    def test_pe_modules_composition(self):
        """Test composition of different PE modules."""
        # Create different PE modules
        sin_pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
        learned_pe = LearnedPositionalEmbedding(max_len=100, d_model=64)

        x = Tensor(np.random.randn(2, 20, 64).astype(np.float32))

        # Test sequential application
        output1 = sin_pe.forward(x)
        output2 = learned_pe.forward(output1)

        assert output2.shape == x.shape
        assert np.all(np.isfinite(output2.data))

    def test_rope_with_attention_simulation(self):
        """Test RoPE in attention-like scenario."""
        rope = RotaryPositionalEmbedding(dim=64)

        # Simulate attention queries and keys
        batch_size, num_heads, seq_len, head_dim = 2, 8, 20, 64

        q = Tensor(np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32))
        k = Tensor(np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32))

        q_rot, k_rot = rope.forward(q, k)

        # Simulate attention computation (simplified)
        # In real attention: scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert np.all(np.isfinite(q_rot.data))
        assert np.all(np.isfinite(k_rot.data))

    def test_pe_numerical_stability(self):
        """Test numerical stability across PE modules."""
        modules = [
            SinusoidalPositionalEncoding(d_model=64),
            LearnedPositionalEmbedding(max_len=100, d_model=64),
        ]

        # Test with extreme input values
        extreme_values = [1e-8, 1e8, 0.0]

        for module in modules:
            for val in extreme_values:
                x = Tensor(np.full((2, 10, 64), val, dtype=np.float32))

                try:
                    output = module.forward(x)
                    assert np.all(np.isfinite(output.data))
                except (ValueError, RuntimeError):
                    # Some extreme values might cause issues
                    pass

    def test_pe_gradient_flow(self):
        """Test gradient flow through PE modules."""
        modules = [
            SinusoidalPositionalEncoding(d_model=32),
            LearnedPositionalEmbedding(max_len=50, d_model=32),
        ]

        for module in modules:
            x = Tensor(np.random.randn(2, 10, 32).astype(np.float32), requires_grad=True)
            output = module.forward(x)

            assert output.requires_grad is True
            if hasattr(output, "_grad_fn"):
                assert output._grad_fn is not None

    def test_pe_dtype_consistency(self):
        """Test data type consistency across PE modules."""
        modules = [
            SinusoidalPositionalEncoding(d_model=32),
            LearnedPositionalEmbedding(max_len=50, d_model=32),
        ]

        for module in modules:
            x = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
            output = module.forward(x)

            assert output.data.dtype == np.float32

    def test_pe_memory_efficiency(self):
        """Test memory efficiency with large sequences."""
        # Test with moderately large sequences
        sin_pe = SinusoidalPositionalEncoding(d_model=512, max_len=2048)

        # Test with increasing sequence lengths
        sequence_lengths = [100, 500, 1000, 1500]

        for seq_len in sequence_lengths:
            x = Tensor(np.random.randn(1, seq_len, 512).astype(np.float32))
            output = sin_pe.forward(x)

            assert output.shape == (1, seq_len, 512)
            assert np.all(np.isfinite(output.data))

    def test_rope_mathematical_properties(self):
        """Test mathematical properties of RoPE."""
        rope = RotaryPositionalEmbedding(dim=8, max_seq_len=20)

        # Test that rotation preserves vector norms (approximately)
        q = Tensor(np.random.randn(1, 10, 8).astype(np.float32))
        k = Tensor(np.random.randn(1, 10, 8).astype(np.float32))

        q_rot, k_rot = rope.forward(q, k)

        # Check that norms are approximately preserved
        q_norm_orig = np.linalg.norm(q.data, axis=-1)
        q_norm_rot = np.linalg.norm(q_rot.data, axis=-1)

        assert np.allclose(q_norm_orig, q_norm_rot, rtol=1e-5)

        k_norm_orig = np.linalg.norm(k.data, axis=-1)
        k_norm_rot = np.linalg.norm(k_rot.data, axis=-1)

        assert np.allclose(k_norm_orig, k_norm_rot, rtol=1e-5)
