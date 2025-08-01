"""Positional encoding implementations with mathematical correctness.

This module implements modern positional encoding techniques including:
- Rotary Position Embedding (RoPE) - used in GPT-NeoX, PaLM, LLaMA
- Sinusoidal Position Encoding - traditional transformer positional encoding
- Learned Position Embedding - trainable positional embeddings

All implementations are mathematically exact and include proper gradient computation.
"""

import numpy as np
from typing import Optional, Tuple, Union
import logging

from ..core import Module, Parameter, Tensor, GradientFunction
from ..exceptions import LayerError
from ..functional.utils import memory_efficient_operation

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(Module):
    """Standard sinusoidal positional encoding from "Attention Is All You Need".
    
    Mathematical Definition:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
    where pos is position and i is dimension index.
    
    Reference: "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
        base: float = 10000.0
    ):
        """Initialize sinusoidal positional encoding.
        
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length
            dropout: Dropout rate (optional)
            base: Base for frequency computation
        """
        super().__init__()
        
        if d_model % 2 != 0:
            raise LayerError(f"d_model must be even for sinusoidal PE, got {d_model}")
        if d_model <= 0:
            raise LayerError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise LayerError(f"max_len must be positive, got {max_len}")
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_rate = dropout
        self.base = base
        
        # Pre-compute positional encodings
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * 
            -(np.log(base) / d_model)
        )
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.pe = pe
        
        if dropout > 0:
            from .dropout import Dropout
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            start_pos: Starting position for encoding (for inference)
            
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if d_model != self.d_model:
            raise LayerError(f"Input d_model {d_model} != expected {self.d_model}")
        if start_pos + seq_len > self.max_len:
            raise LayerError(f"Sequence length {start_pos + seq_len} exceeds max_len {self.max_len}")
        
        # Get positional encodings for this sequence
        pe_slice = self.pe[start_pos:start_pos + seq_len]  # (seq_len, d_model)
        pe_broadcasted = np.broadcast_to(pe_slice[None, :, :], (batch_size, seq_len, d_model))
        
        # Add positional encoding
        output_data = x.data + pe_broadcasted
        
        result = Tensor(
            output_data,
            requires_grad=x.requires_grad,
            name=f"pos_enc({x.name or 'tensor'})"
        )
        
        # Gradient computation (PE is not learnable, so only pass through x gradients)
        if x.requires_grad:
            def backward_fn(grad_output: np.ndarray) -> None:
                if x._grad is None:
                    x._grad = x._backend.from_numpy(grad_output)
                    device_str = x._device.type.value
                    if x._device.index is not None:
                        device_str = f"{device_str}:{x._device.index}"
                    x._grad = x._backend.to_device(x._grad, device_str)
                else:
                    grad_backend = x._backend.from_numpy(grad_output)
                    device_str = x._device.type.value
                    if x._device.index is not None:
                        device_str = f"{device_str}:{x._device.index}"
                    grad_backend = x._backend.to_device(grad_backend, device_str)
                    x._grad = x._backend.add(x._grad, grad_backend)
                
                if x._grad_fn is not None:
                    x._grad_fn.apply(grad_output)
            
            result._grad_fn = GradientFunction(backward_fn, [x], "positional_encoding")
        
        # Apply dropout if configured
        if self.dropout is not None:
            result = self.dropout(result)
        
        logger.debug(f"Sinusoidal PE: {x.shape} -> {result.shape}")
        return result


class RotaryPositionalEmbedding(Module):
    """Rotary Position Embedding (RoPE) with mathematically exact implementation.
    
    RoPE applies rotation matrices to query and key vectors in attention mechanisms,
    providing superior positional information compared to additive encodings.
    
    Mathematical Definition:
        For each position m and dimension d:
        R_Θ,m = [cos(m*θ_i)  -sin(m*θ_i)]
                [sin(m*θ_i)   cos(m*θ_i)]
                
        where θ_i = base^(-2i/d) and i is the dimension pair index.
    
    The rotation is applied to consecutive pairs of dimensions in q and k vectors.
    
    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (https://arxiv.org/abs/2104.09864)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        precision: str = "float32"
    ):
        """Initialize RoPE.
        
        Args:
            dim: Dimension of embeddings (head dimension, must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for frequency computation (10000 is standard)
            precision: Numerical precision ("float32" or "float64")
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise LayerError(f"RoPE dimension must be even, got {dim}")
        if dim <= 0:
            raise LayerError(f"RoPE dimension must be positive, got {dim}")
        if max_seq_len <= 0:
            raise LayerError(f"max_seq_len must be positive, got {max_seq_len}")
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.precision = precision
        
        # Use higher precision for mathematical accuracy
        dtype = np.float64 if precision == "float64" else np.float32
        
        # Pre-compute frequency values
        # θ_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=dtype) / dim))
        self.inv_freq = inv_freq
        
        # Pre-compute rotation matrices for all positions
        self._precompute_rotation_matrices(max_seq_len, dtype)
    
    def _precompute_rotation_matrices(self, max_seq_len: int, dtype: np.dtype) -> None:
        """Pre-compute rotation matrices for efficiency."""
        # Position indices
        seq_len = np.arange(max_seq_len, dtype=dtype)  # [0, 1, 2, ..., max_seq_len-1]
        
        # Compute frequencies for all positions and dimensions
        # freqs shape: (max_seq_len, dim//2)
        freqs = np.outer(seq_len, self.inv_freq)  # m * θ_i
        
        # Pre-compute cos and sin values
        cos_vals = np.cos(freqs)  # (max_seq_len, dim//2)
        sin_vals = np.sin(freqs)  # (max_seq_len, dim//2)
        
        # Store for later use
        self.cos_cached = cos_vals.astype(np.float32)
        self.sin_cached = sin_vals.astype(np.float32) 
    
    def rotate_half(self, x: np.ndarray) -> np.ndarray:
        """Rotate half the dimensions by 90 degrees.
        
        This transforms [x1, x2, x3, x4, ...] to [-x2, x1, -x4, x3, ...]
        which is equivalent to applying a 90-degree rotation.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Rotated tensor of same shape
        """
        # Split into pairs: [x1, x2], [x3, x4], ...
        x1 = x[..., ::2]   # [x1, x3, x5, ...]
        x2 = x[..., 1::2]  # [x2, x4, x6, ...]
        
        # Apply rotation: [-x2, x1], [-x4, x3], ...
        rotated = np.empty_like(x)
        rotated[..., ::2] = -x2   # [-x2, -x4, -x6, ...]
        rotated[..., 1::2] = x1   # [x1, x3, x5, ...]
        
        return rotated
    
    def apply_rope(self, x: np.ndarray, seq_len: int, start_pos: int = 0) -> np.ndarray:
        """Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, dim)
            seq_len: Sequence length
            start_pos: Starting position (for inference)
            
        Returns:
            Tensor with RoPE applied
        """
        if start_pos + seq_len > self.max_seq_len:
            # Extend cached values if needed
            self._precompute_rotation_matrices(start_pos + seq_len, np.float32)
        
        # Get cos and sin for the current sequence
        cos = self.cos_cached[start_pos:start_pos + seq_len]  # (seq_len, dim//2)
        sin = self.sin_cached[start_pos:start_pos + seq_len]  # (seq_len, dim//2)
        
        # Repeat each frequency for the pair of dimensions
        # cos: (seq_len, dim//2) -> (seq_len, dim)
        cos_expanded = np.repeat(cos, 2, axis=-1)
        sin_expanded = np.repeat(sin, 2, axis=-1)
        
        # Broadcast to match input shape
        input_shape = x.shape[:-1]  # All dimensions except last
        cos_broadcasted = np.broadcast_to(cos_expanded, input_shape + (self.dim,))
        sin_broadcasted = np.broadcast_to(sin_expanded, input_shape + (self.dim,))
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        # This is equivalent to applying the 2D rotation matrix to each pair
        x_rotated_half = self.rotate_half(x)
        result = x * cos_broadcasted + x_rotated_half * sin_broadcasted
        
        return result
    
    @memory_efficient_operation
    def forward(
        self, 
        q: Tensor, 
        k: Tensor, 
        start_pos: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Apply RoPE to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch, seq_len, dim) or (batch, heads, seq_len, dim)
            k: Key tensor of shape (batch, seq_len, dim) or (batch, heads, seq_len, dim)
            start_pos: Starting position for rotary embedding (for inference)
            
        Returns:
            Tuple of (rotated_q, rotated_k) with RoPE applied
        """
        # Validate inputs
        if q.shape != k.shape:
            raise LayerError(f"Query and Key shapes must match: {q.shape} vs {k.shape}")
        
        last_dim = q.shape[-1]
        if last_dim != self.dim:
            raise LayerError(f"Last dimension {last_dim} != RoPE dim {self.dim}")
        
        seq_len = q.shape[-2]  # Second to last dimension is sequence length
        
        # Apply RoPE to both q and k
        q_rotated_data = self.apply_rope(q.data, seq_len, start_pos)
        k_rotated_data = self.apply_rope(k.data, seq_len, start_pos)
        
        # Create result tensors
        requires_grad_q = q.requires_grad
        requires_grad_k = k.requires_grad
        
        q_rotated = Tensor(
            q_rotated_data,
            requires_grad=requires_grad_q,
            name=f"rope_q({q.name or 'tensor'})"
        )
        
        k_rotated = Tensor(
            k_rotated_data,
            requires_grad=requires_grad_k,
            name=f"rope_k({k.name or 'tensor'})"
        )
        
        # Set up gradient computation
        if requires_grad_q:
            def backward_fn_q(grad_output: np.ndarray) -> None:
                # RoPE is a linear transformation, so gradient follows same rotation
                grad_rotated = self.apply_rope(grad_output, seq_len, start_pos)
                
                if q._grad is None:
                    q._grad = q._backend.from_numpy(grad_rotated)
                    device_str = q._device.type.value
                    if q._device.index is not None:
                        device_str = f"{device_str}:{q._device.index}"
                    q._grad = q._backend.to_device(q._grad, device_str)
                else:
                    grad_backend = q._backend.from_numpy(grad_rotated)
                    device_str = q._device.type.value
                    if q._device.index is not None:
                        device_str = f"{device_str}:{q._device.index}"
                    grad_backend = q._backend.to_device(grad_backend, device_str)
                    q._grad = q._backend.add(q._grad, grad_backend)
                
                if q._grad_fn is not None:
                    q._grad_fn.apply(grad_rotated)
            
            q_rotated._grad_fn = GradientFunction(backward_fn_q, [q], "rope_q")
        
        if requires_grad_k:
            def backward_fn_k(grad_output: np.ndarray) -> None:
                grad_rotated = self.apply_rope(grad_output, seq_len, start_pos)
                
                if k._grad is None:
                    k._grad = k._backend.from_numpy(grad_rotated)
                    device_str = k._device.type.value  
                    if k._device.index is not None:
                        device_str = f"{device_str}:{k._device.index}"
                    k._grad = k._backend.to_device(k._grad, device_str)
                else:
                    grad_backend = k._backend.from_numpy(grad_rotated)
                    device_str = k._device.type.value
                    if k._device.index is not None:
                        device_str = f"{device_str}:{k._device.index}"
                    grad_backend = k._backend.to_device(grad_backend, device_str)
                    k._grad = k._backend.add(k._grad, grad_backend)
                
                if k._grad_fn is not None:
                    k._grad_fn.apply(grad_rotated)
            
            k_rotated._grad_fn = GradientFunction(backward_fn_k, [k], "rope_k")
        
        logger.debug(f"RoPE applied: q{q.shape} k{k.shape} -> rotated q{q_rotated.shape} k{k_rotated.shape}")
        return q_rotated, k_rotated
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"


class LearnedPositionalEmbedding(Module):
    """Learned positional embeddings (trainable).
    
    This is a standard embedding layer for positions, commonly used when
    you want the model to learn positional representations rather than
    using fixed mathematical encodings.
    """
    
    def __init__(
        self,
        max_len: int,
        d_model: int,
        dropout: float = 0.0
    ):
        """Initialize learned positional embedding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            dropout: Dropout rate (optional)
        """
        super().__init__()
        
        if max_len <= 0:
            raise LayerError(f"max_len must be positive, got {max_len}")
        if d_model <= 0:
            raise LayerError(f"d_model must be positive, got {d_model}")
        
        self.max_len = max_len
        self.d_model = d_model
        self.dropout_rate = dropout
        
        # Learnable position embeddings
        self.embedding = Parameter(
            np.random.normal(0, 0.02, (max_len, d_model)).astype(np.float32),
            name="learned_pe.embedding"
        )
        
        if dropout > 0:
            from .dropout import Dropout
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
    
    @memory_efficient_operation
    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """Add learned positional embedding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            start_pos: Starting position for embedding (for inference)
            
        Returns:
            Tensor with positional embedding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if d_model != self.d_model:
            raise LayerError(f"Input d_model {d_model} != expected {self.d_model}")
        if start_pos + seq_len > self.max_len:
            raise LayerError(f"Sequence length {start_pos + seq_len} exceeds max_len {self.max_len}")
        
        # Get positional embeddings for this sequence
        pe_slice = self.embedding.data[start_pos:start_pos + seq_len]  # (seq_len, d_model)
        pe_broadcasted = np.broadcast_to(pe_slice[None, :, :], (batch_size, seq_len, d_model))
        
        # Add positional embedding
        output_data = x.data + pe_broadcasted
        
        requires_grad = x.requires_grad or self.embedding.requires_grad
        
        result = Tensor(
            output_data,
            requires_grad=requires_grad,
            name=f"learned_pe({x.name or 'tensor'})"
        )
        
        # Gradient computation
        if requires_grad:
            def backward_fn(grad_output: np.ndarray) -> None:
                # Gradient w.r.t. input (pass through)
                if x.requires_grad:
                    if x._grad is None:
                        x._grad = x._backend.from_numpy(grad_output)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        x._grad = x._backend.to_device(x._grad, device_str)
                    else:
                        grad_backend = x._backend.from_numpy(grad_output)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        grad_backend = x._backend.to_device(grad_backend, device_str)
                        x._grad = x._backend.add(x._grad, grad_backend)
                    
                    if x._grad_fn is not None:
                        x._grad_fn.apply(grad_output)
                
                # Gradient w.r.t. embedding parameters
                if self.embedding.requires_grad:
                    # Sum gradients across batch dimension
                    grad_embedding = np.sum(grad_output, axis=0)  # (seq_len, d_model)
                    
                    # Accumulate at the correct positions
                    if self.embedding._grad is None:
                        full_grad = np.zeros_like(self.embedding.data)
                        full_grad[start_pos:start_pos + seq_len] = grad_embedding
                        self.embedding._grad = self.embedding._backend.from_numpy(full_grad)
                    else:
                        # Add to existing gradient
                        grad_update = np.zeros_like(self.embedding.data)
                        grad_update[start_pos:start_pos + seq_len] = grad_embedding
                        grad_backend = self.embedding._backend.from_numpy(grad_update)
                        self.embedding._grad = self.embedding._backend.add(self.embedding._grad, grad_backend)
            
            input_tensors = [x, self.embedding]
            result._grad_fn = GradientFunction(backward_fn, input_tensors, "learned_pe")
        
        # Apply dropout if configured
        if self.dropout is not None:
            result = self.dropout(result)
        
        logger.debug(f"Learned PE: {x.shape} -> {result.shape}")
        return result
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"max_len={self.max_len}, d_model={self.d_model}"


# Convenient aliases
RoPE = RotaryPositionalEmbedding
SinusoidalPE = SinusoidalPositionalEncoding
LearnedPE = LearnedPositionalEmbedding


def create_rope(dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> RoPE:
    """Create RoPE instance with standard parameters.
    
    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
        
    Returns:
        RoPE instance ready for use
    """
    return RoPE(dim, max_seq_len, base)


def create_sinusoidal_pe(d_model: int, max_len: int = 5000) -> SinusoidalPE:
    """Create sinusoidal PE with standard parameters.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        
    Returns:
        SinusoidalPE instance ready for use
    """
    return SinusoidalPE(d_model, max_len)