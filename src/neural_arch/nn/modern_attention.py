"""Modern attention mechanisms including GQA, MQA, and FlashAttention concepts.

This module implements state-of-the-art attention mechanisms:
- Grouped-Query Attention (GQA): Used in Llama 2, reduces KV cache by 8x
- Multi-Query Attention (MQA): Single KV head for all Q heads
- Efficient attention patterns for long sequences

These optimizations are critical for inference efficiency in production LLMs.
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

from ..core import GradientFunction, Module, Parameter, Tensor
from ..exceptions import LayerError
from ..functional import softmax
from ..functional.utils import memory_efficient_operation
from .linear import Linear

logger = logging.getLogger(__name__)


class GroupedQueryAttention(Module):
    """Grouped-Query Attention (GQA) mechanism.
    
    GQA is a compromise between Multi-Head Attention (MHA) and Multi-Query
    Attention (MQA). It groups query heads to share key-value heads, reducing
    memory usage while maintaining most of MHA's quality.
    
    Architecture:
    - n_heads query heads
    - n_kv_heads key-value heads (typically n_heads/8)
    - Each KV head is shared by n_heads/n_kv_heads query heads
    
    Benefits:
    - 8x reduction in KV cache memory (when n_kv_heads = n_heads/8)
    - Near-identical quality to MHA
    - Much faster inference than MHA
    
    Reference: "GQA: Training Generalized Multi-Query Transformer Models from 
    Multi-Head Checkpoints" (https://arxiv.org/abs/2305.13245)
    Used in: Llama 2 70B, Mistral, and other modern LLMs
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        rope_base: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        """Initialize Grouped-Query Attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads: Number of key-value heads (defaults to n_heads for standard MHA)
            dropout: Dropout probability
            bias: Whether to use bias in projections
            rope_base: Base for RoPE positional encoding
            max_seq_len: Maximum sequence length for RoPE
        """
        super().__init__()
        
        if d_model % n_heads != 0:
            raise LayerError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        # Default to standard MHA if not specified
        if n_kv_heads is None:
            n_kv_heads = n_heads
        
        if n_heads % n_kv_heads != 0:
            raise LayerError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Repetition factor for KV heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Query projection: d_model -> n_heads * head_dim
        self.W_q = Parameter(
            np.random.randn(d_model, n_heads * self.head_dim).astype(np.float32) 
            * np.sqrt(2.0 / d_model),
            name="gqa.W_q"
        )
        
        # Key projection: d_model -> n_kv_heads * head_dim
        self.W_k = Parameter(
            np.random.randn(d_model, n_kv_heads * self.head_dim).astype(np.float32)
            * np.sqrt(2.0 / d_model),
            name="gqa.W_k"
        )
        
        # Value projection: d_model -> n_kv_heads * head_dim
        self.W_v = Parameter(
            np.random.randn(d_model, n_kv_heads * self.head_dim).astype(np.float32)
            * np.sqrt(2.0 / d_model),
            name="gqa.W_v"
        )
        
        # Output projection: n_heads * head_dim -> d_model
        self.W_o = Parameter(
            np.random.randn(n_heads * self.head_dim, d_model).astype(np.float32)
            * np.sqrt(2.0 / (n_heads * self.head_dim)),
            name="gqa.W_o"
        )
        
        if bias:
            self.b_q = Parameter(np.zeros(n_heads * self.head_dim, dtype=np.float32), name="gqa.b_q")
            self.b_k = Parameter(np.zeros(n_kv_heads * self.head_dim, dtype=np.float32), name="gqa.b_k")
            self.b_v = Parameter(np.zeros(n_kv_heads * self.head_dim, dtype=np.float32), name="gqa.b_v")
            self.b_o = Parameter(np.zeros(d_model, dtype=np.float32), name="gqa.b_o")
        else:
            self.b_q = None
            self.b_k = None
            self.b_v = None
            self.b_o = None
        
        # Optional: RoPE for positional encoding
        self.use_rope = rope_base > 0
        if self.use_rope:
            from .positional import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len, rope_base)
        else:
            self.rope = None
    
    def repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """Repeat KV heads to match number of Q heads.
        
        Args:
            x: Tensor of shape (batch, seq_len, n_kv_heads, head_dim)
            
        Returns:
            Tensor of shape (batch, seq_len, n_heads, head_dim)
        """
        if self.n_rep == 1:
            return x
        
        batch, seq_len, n_kv_heads, head_dim = x.shape
        
        # Expand: (batch, seq_len, n_kv_heads, 1, head_dim)
        x = np.expand_dims(x, axis=3)
        
        # Repeat: (batch, seq_len, n_kv_heads, n_rep, head_dim)
        x = np.repeat(x, self.n_rep, axis=3)
        
        # Reshape: (batch, seq_len, n_kv_heads * n_rep, head_dim)
        x = x.reshape(batch, seq_len, n_kv_heads * self.n_rep, head_dim)
        
        return x
    
    @memory_efficient_operation
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False
    ) -> Tuple[Tensor, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Forward pass for Grouped-Query Attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            start_pos: Starting position for RoPE (useful for inference)
            use_cache: Whether to return KV cache for inference
            
        Returns:
            Output tensor and optional KV cache
        """
        batch_size, seq_len, d_model = x.shape
        
        if d_model != self.d_model:
            raise LayerError(f"Input d_model {d_model} != expected {self.d_model}")
        
        # Compute Q, K, V projections
        Q = np.matmul(x.data, self.W_q.data)
        K = np.matmul(x.data, self.W_k.data)
        V = np.matmul(x.data, self.W_v.data)
        
        if self.b_q is not None:
            Q = Q + self.b_q.data
            K = K + self.b_k.data
            V = V + self.b_v.data
        
        # Reshape to separate heads
        # Q: (batch, seq_len, n_heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # K, V: (batch, seq_len, n_kv_heads, head_dim)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE if enabled
        if self.use_rope:
            # RoPE expects (batch, heads, seq_len, head_dim) format
            Q_rope = Q.transpose(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
            K_rope = K.transpose(0, 2, 1, 3)  # (batch, n_kv_heads, seq_len, head_dim)
            
            # Apply RoPE (simplified - full implementation would use the rope module)
            # For now, we'll skip the actual rotation for simplicity
            Q_rope = Q_rope
            K_rope = K_rope
            
            # Transpose back
            Q = Q_rope.transpose(0, 2, 1, 3)
            K = K_rope.transpose(0, 2, 1, 3)
        
        # Repeat KV heads to match Q heads
        K = self.repeat_kv(K)  # (batch, seq_len, n_heads, head_dim)
        V = self.repeat_kv(V)  # (batch, seq_len, n_heads, head_dim)
        
        # Transpose for attention computation
        # (batch, n_heads, seq_len, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        # (batch, n_heads, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Mask should be broadcastable to (batch, n_heads, seq_len, seq_len)
            scores = scores + mask.data
        
        # Apply softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # Apply attention to values
        # (batch, n_heads, seq_len, head_dim)
        attn_output = np.matmul(attn_weights, V)
        
        # Transpose and reshape
        # (batch, seq_len, n_heads, head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        
        # (batch, seq_len, n_heads * head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        
        # Output projection
        output_data = np.matmul(attn_output, self.W_o.data)
        if self.b_o is not None:
            output_data = output_data + self.b_o.data
        
        # Determine if gradients needed
        requires_grad = (
            x.requires_grad or
            self.W_q.requires_grad or
            self.W_k.requires_grad or
            self.W_v.requires_grad or
            self.W_o.requires_grad
        )
        
        result = Tensor(
            output_data,
            requires_grad=requires_grad,
            name=f"gqa({x.name or 'tensor'})"
        )
        
        # Gradient computation (simplified for brevity)
        if requires_grad:
            cached_Q = Q
            cached_K = K
            cached_V = V
            cached_attn_weights = attn_weights
            cached_attn_output = attn_output
            
            def backward_fn(grad_output: np.ndarray) -> None:
                # This would involve computing gradients through:
                # 1. Output projection
                # 2. Attention weights
                # 3. Softmax
                # 4. Q, K, V projections
                # Full implementation would be quite lengthy
                
                # Gradient w.r.t output projection
                grad_attn = np.matmul(grad_output, self.W_o.data.T)
                
                # Simplified: just propagate to input
                if x.requires_grad:
                    # This is a placeholder - full gradient computation would be more complex
                    grad_x = np.matmul(grad_attn.reshape(batch_size * seq_len, -1),
                                      self.W_q.data.T).reshape(x.shape)
                    
                    if x._grad is None:
                        x._grad = x._backend.from_numpy(grad_x)
                    else:
                        grad_backend = x._backend.from_numpy(grad_x)
                        x._grad = x._backend.add(x._grad, grad_backend)
                    
                    if x._grad_fn is not None:
                        x._grad_fn.apply(grad_x)
                
                logger.debug("GQA gradient computation simplified - full implementation needed")
            
            inputs = [x, self.W_q, self.W_k, self.W_v, self.W_o]
            result._grad_fn = GradientFunction(backward_fn, inputs, "gqa")
        
        # Return KV cache if requested (for inference optimization)
        kv_cache = (K, V) if use_cache else None
        
        logger.debug(f"GQA: {x.shape} -> {result.shape}, n_kv_heads={self.n_kv_heads}")
        
        if use_cache:
            return result, kv_cache
        return result
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_kv_heads={self.n_kv_heads}, n_rep={self.n_rep}")


class MultiQueryAttention(Module):
    """Multi-Query Attention (MQA) - single KV head for all Q heads.
    
    MQA is the extreme version of GQA with n_kv_heads=1.
    It provides maximum memory savings but may have slightly lower quality.
    
    Reference: "Fast Transformer Decoding: One Write-Head is All You Need"
    (https://arxiv.org/abs/1911.02150)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False
    ):
        """Initialize MQA as GQA with n_kv_heads=1."""
        super().__init__()
        # MQA is just GQA with n_kv_heads=1
        self.gqa = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=1,  # Single KV head
            dropout=dropout,
            bias=bias
        )
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass delegates to GQA."""
        return self.gqa(x, mask)


class FlashAttentionConcept(Module):
    """Conceptual implementation of FlashAttention ideas.
    
    FlashAttention is an IO-aware attention algorithm that:
    - Reduces memory usage from O(n²) to O(n)
    - Achieves 2-4x speedup through better GPU memory access patterns
    - Uses tiling and recomputation to minimize HBM accesses
    
    This is a simplified conceptual implementation showing the key ideas.
    Full FlashAttention requires CUDA kernel implementation.
    
    Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    (https://arxiv.org/abs/2205.14135)
    """
    
    def __init__(self, d_model: int, n_heads: int = 8):
        """Initialize FlashAttention concept."""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Standard projections
        self.qkv_proj = Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """Conceptual forward pass showing tiling idea."""
        batch_size, seq_len, d_model = x.shape
        
        # Compute QKV in one projection (more efficient)
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        qkv_data = qkv.data.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv_data = qkv_data.transpose(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, head_dim)
        
        Q, K, V = qkv_data[0], qkv_data[1], qkv_data[2]
        
        # In real FlashAttention, computation would be tiled here
        # to process blocks of the sequence at a time, minimizing memory
        
        # Standard attention (simplified)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        attn_output = np.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        
        output = self.out_proj(Tensor(attn_output, requires_grad=x.requires_grad))
        
        logger.debug("FlashAttention concept - full implementation requires CUDA kernels")
        return output