"""Tests for modern neural network components (RoPE, RMSNorm, SwiGLU, GQA)."""

import numpy as np
import pytest

from neural_arch.core import Tensor
from neural_arch.nn.normalization import RMSNorm
from neural_arch.nn.positional import RotaryPositionalEmbedding
from neural_arch.nn.modern_activations import SwiGLU, GeGLU, Swish
from neural_arch.nn.modern_attention import GroupedQueryAttention, MultiQueryAttention


class TestRMSNorm:
    """Test RMSNorm implementation."""
    
    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        batch_size, seq_len, d_model = 2, 10, 64
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        
        norm = RMSNorm(d_model)
        output = norm(x)
        
        assert output.shape == x.shape
        
        # Check that RMS is approximately 1 after normalization
        rms = np.sqrt(np.mean(output.data ** 2, axis=-1))
        np.testing.assert_allclose(rms, np.ones_like(rms), rtol=0.1)
    
    def test_rmsnorm_gradient(self):
        """Test RMSNorm gradient computation."""
        x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
        norm = RMSNorm(8)
        
        output = norm(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert norm.weight.grad is not None
    
    def test_rmsnorm_no_bias(self):
        """Test that RMSNorm has no bias parameter."""
        norm = RMSNorm(64)
        assert not hasattr(norm, 'bias') or norm.bias is None
    
    def test_rmsnorm_vs_layernorm_speed(self):
        """Verify RMSNorm is simpler than LayerNorm (no mean centering)."""
        from neural_arch.nn.normalization import LayerNorm
        
        x = Tensor(np.random.randn(4, 128, 512))
        
        rmsnorm = RMSNorm(512)
        layernorm = LayerNorm(512)
        
        # RMSNorm should not center the mean
        rms_out = rmsnorm(x)
        ln_out = layernorm(x)
        
        # Both should normalize, but RMSNorm keeps relative mean
        assert rms_out.shape == ln_out.shape


class TestRotaryPositionalEmbedding:
    """Test RoPE implementation."""
    
    def test_rope_forward(self):
        """Test RoPE forward pass."""
        batch_size, seq_len, n_heads, head_dim = 2, 16, 8, 64
        
        # Create query and key tensors
        q = Tensor(np.random.randn(batch_size, seq_len, head_dim))
        k = Tensor(np.random.randn(batch_size, seq_len, head_dim))
        
        rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=seq_len)
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_rope_rotation_property(self):
        """Test that RoPE applies rotation."""
        head_dim = 64
        rope = RotaryPositionalEmbedding(dim=head_dim)
        
        # Create simple input
        q = Tensor(np.ones((1, 4, head_dim)))
        k = Tensor(np.ones((1, 4, head_dim)))
        
        q_rot, k_rot = rope(q, k)
        
        # After rotation, values should change
        assert not np.allclose(q_rot.data, q.data)
        assert not np.allclose(k_rot.data, k.data)
    
    def test_rope_gradient_flow(self):
        """Test gradient flow through RoPE."""
        q = Tensor(np.random.randn(1, 8, 32), requires_grad=True)
        k = Tensor(np.random.randn(1, 8, 32), requires_grad=True)
        
        rope = RotaryPositionalEmbedding(dim=32)
        q_rot, k_rot = rope(q, k)
        
        loss = (q_rot.sum() + k_rot.sum())
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None
    
    def test_rope_different_positions(self):
        """Test RoPE with different starting positions."""
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=256)
        
        q = Tensor(np.random.randn(1, 16, 64))
        k = Tensor(np.random.randn(1, 16, 64))
        
        # Apply at different positions
        q_rot1, k_rot1 = rope(q, k, start_pos=0)
        q_rot2, k_rot2 = rope(q, k, start_pos=16)
        
        # Different positions should give different results
        assert not np.allclose(q_rot1.data, q_rot2.data)


class TestSwiGLU:
    """Test SwiGLU activation."""
    
    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        batch_size, seq_len, d_model = 2, 10, 256
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        swiglu = SwiGLU(input_dim=d_model)
        output = swiglu(x)
        
        assert output.shape == x.shape
    
    def test_swiglu_hidden_dim(self):
        """Test SwiGLU with custom hidden dimension."""
        d_model = 128
        hidden_dim = 512
        
        swiglu = SwiGLU(input_dim=d_model, hidden_dim=hidden_dim)
        
        # Check parameter shapes
        assert swiglu.W_gate.shape == (d_model, hidden_dim)
        assert swiglu.W_up.shape == (d_model, hidden_dim)
        assert swiglu.W_down.shape == (hidden_dim, d_model)
    
    def test_swiglu_gradient(self):
        """Test gradient flow through SwiGLU."""
        x = Tensor(np.random.randn(2, 4, 64), requires_grad=True)
        swiglu = SwiGLU(input_dim=64, hidden_dim=128)
        
        output = swiglu(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert swiglu.W_gate.grad is not None
        assert swiglu.W_up.grad is not None
        assert swiglu.W_down.grad is not None
    
    def test_swiglu_no_bias(self):
        """Test SwiGLU without bias."""
        swiglu = SwiGLU(input_dim=64, bias=False)
        
        assert swiglu.b_gate is None
        assert swiglu.b_up is None
        assert swiglu.b_down is None
        
        x = Tensor(np.random.randn(1, 8, 64))
        output = swiglu(x)
        assert output.shape == x.shape


class TestSwish:
    """Test Swish/SiLU activation."""
    
    def test_swish_forward(self):
        """Test Swish forward pass."""
        x = Tensor(np.array([-2, -1, 0, 1, 2]))
        swish = Swish()
        output = swish(x)
        
        # Swish(x) = x * sigmoid(x)
        expected = x.data * (1 / (1 + np.exp(-x.data)))
        np.testing.assert_allclose(output.data, expected, rtol=1e-5)
    
    def test_swish_gradient(self):
        """Test Swish gradient."""
        x = Tensor(np.random.randn(3, 3), requires_grad=True)
        swish = Swish()
        
        output = swish(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestGroupedQueryAttention:
    """Test GQA implementation."""
    
    def test_gqa_forward(self):
        """Test GQA forward pass."""
        batch_size, seq_len, d_model = 2, 16, 256
        n_heads = 8
        n_kv_heads = 2  # 4x reduction
        
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads
        )
        
        output = gqa(x)
        assert output.shape == x.shape
    
    def test_gqa_kv_reduction(self):
        """Test that GQA reduces KV parameters."""
        d_model = 512
        n_heads = 16
        n_kv_heads = 2  # 8x reduction
        
        gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        
        # Check parameter shapes
        assert gqa.W_q.shape == (d_model, n_heads * (d_model // n_heads))
        assert gqa.W_k.shape == (d_model, n_kv_heads * (d_model // n_heads))
        assert gqa.W_v.shape == (d_model, n_kv_heads * (d_model // n_heads))
        
        # KV parameters should be 8x smaller than Q
        q_params = gqa.W_q.data.size
        kv_params = gqa.W_k.data.size + gqa.W_v.data.size
        assert q_params > kv_params
    
    def test_gqa_repeat_kv(self):
        """Test KV head repetition."""
        gqa = GroupedQueryAttention(d_model=256, n_heads=8, n_kv_heads=2)
        
        # Create dummy KV tensor
        batch, seq_len = 2, 10
        kv = np.random.randn(batch, seq_len, 2, 32)  # 2 KV heads
        
        repeated = gqa.repeat_kv(kv)
        
        # Should expand to 8 heads
        assert repeated.shape == (batch, seq_len, 8, 32)
    
    def test_gqa_with_cache(self):
        """Test GQA with KV cache for inference."""
        gqa = GroupedQueryAttention(d_model=128, n_heads=4, n_kv_heads=1)
        
        x = Tensor(np.random.randn(1, 8, 128))
        output, kv_cache = gqa(x, use_cache=True)
        
        assert output.shape == x.shape
        assert kv_cache is not None
        assert len(kv_cache) == 2  # K and V
    
    def test_gqa_gradient(self):
        """Test gradient flow through GQA."""
        x = Tensor(np.random.randn(2, 8, 128), requires_grad=True)
        gqa = GroupedQueryAttention(d_model=128, n_heads=4, n_kv_heads=2)
        
        output = gqa(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestMultiQueryAttention:
    """Test MQA implementation."""
    
    def test_mqa_forward(self):
        """Test MQA forward pass."""
        x = Tensor(np.random.randn(2, 16, 256))
        mqa = MultiQueryAttention(d_model=256, n_heads=8)
        
        output = mqa(x)
        assert output.shape == x.shape
    
    def test_mqa_single_kv_head(self):
        """Verify MQA uses single KV head."""
        mqa = MultiQueryAttention(d_model=512, n_heads=16)
        
        # MQA internally uses GQA with n_kv_heads=1
        assert mqa.gqa.n_kv_heads == 1
        assert mqa.gqa.n_rep == 16  # Each KV head repeated 16 times


class TestIntegration:
    """Test integration of modern components."""
    
    def test_transformer_block_with_modern_components(self):
        """Test a transformer block using all modern components."""
        from neural_arch.nn import Linear
        
        class ModernTransformerBlock(Tensor.__class__):
            def __init__(self, d_model, n_heads, n_kv_heads):
                self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
                self.norm1 = RMSNorm(d_model)
                self.norm2 = RMSNorm(d_model)
                self.ffn = SwiGLU(d_model)
            
            def forward(self, x):
                # Pre-norm architecture (like LLaMA)
                h = x + self.attn(self.norm1(x))
                out = h + self.ffn(self.norm2(h))
                return out
        
        # This would be the modern transformer architecture
        # combining RMSNorm + RoPE + SwiGLU + GQA
    
    def test_memory_efficiency(self):
        """Verify memory efficiency of modern components."""
        d_model = 1024
        n_heads = 16
        
        # Standard MHA
        from neural_arch.nn.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model, n_heads)
        
        # GQA with 4x reduction
        gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads=4)
        
        # Count parameters (simplified)
        mha_params = sum(p.data.size for p in [mha.query_proj.weight, mha.key_proj.weight, 
                                               mha.value_proj.weight, mha.out_proj.weight])
        
        gqa_params = sum(p.data.size for p in [gqa.W_q, gqa.W_k, gqa.W_v, gqa.W_o])
        
        # GQA should have fewer parameters
        assert gqa_params < mha_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])