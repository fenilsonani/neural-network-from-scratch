"""Modern Transformer architectures with mathematical improvements.

This module implements state-of-the-art transformer architectures including:
- Pre-Norm Transformer (more stable training than post-norm)
- RoPE-enabled attention (superior positional encoding)
- Modern activations (SwiGLU, GELU)
- Advanced normalization (RMSNorm, LayerNorm)
- Mathematical correctness improvements

All implementations follow best practices from recent research and provide
superior performance compared to the original Transformer architecture.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
import logging

from ...core import Module, Tensor, Parameter
from ...nn import Linear, LayerNorm, RMSNorm, Dropout
from ...nn.positional import RotaryPositionalEmbedding, RoPE
from ...functional import gelu, swiglu, softmax, matmul
from ...exceptions import ModelError
from ..registry import register_model

logger = logging.getLogger(__name__)


class PreNormTransformerConfig:
    """Configuration for Pre-Norm Transformer."""
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        vocab_size: int = 30000,
        dropout: float = 0.1,
        activation: str = "gelu",  # "gelu", "swiglu", "relu"
        normalization: str = "layernorm",  # "layernorm", "rmsnorm"
        use_rope: bool = True,  # Use RoPE instead of sinusoidal PE
        rope_base: float = 10000.0,
        eps: float = 1e-5,
        tie_embeddings: bool = True,  # Tie input/output embeddings
        scale_embeddings: bool = True,  # Scale embeddings by sqrt(d_model)
    ):
        """Initialize configuration.
        
        Args:
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            dropout: Dropout rate
            activation: Activation function for FFN
            normalization: Normalization type
            use_rope: Whether to use RoPE
            rope_base: Base for RoPE frequency computation
            eps: Epsilon for numerical stability
            tie_embeddings: Whether to tie input/output embeddings
            scale_embeddings: Whether to scale embeddings
        """
        # Validation
        if d_model <= 0:
            raise ModelError(f"d_model must be positive, got {d_model}")
        if d_model % num_heads != 0:
            raise ModelError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        if num_layers <= 0:
            raise ModelError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ModelError(f"num_heads must be positive, got {num_heads}")
        if d_ff <= 0:
            raise ModelError(f"d_ff must be positive, got {d_ff}")
        if max_seq_len <= 0:
            raise ModelError(f"max_seq_len must be positive, got {max_seq_len}")
        if vocab_size <= 0:
            raise ModelError(f"vocab_size must be positive, got {vocab_size}")
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.activation = activation
        self.normalization = normalization
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.eps = eps
        self.tie_embeddings = tie_embeddings
        self.scale_embeddings = scale_embeddings
        
        # Derived values
        self.head_dim = d_model // num_heads


class RoPEMultiHeadAttention(Module):
    """Multi-Head Attention with RoPE support.
    
    This is an enhanced version of multi-head attention that:
    - Uses RoPE for superior positional encoding
    - Implements mathematically correct attention computation
    - Supports both training and inference modes
    - Provides numerical stability improvements
    """
    
    def __init__(self, config: PreNormTransformerConfig):
        """Initialize RoPE-enabled multi-head attention.
        
        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / np.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = Linear(config.d_model, config.d_model, bias=True)
        self.k_proj = Linear(config.d_model, config.d_model, bias=True)
        self.v_proj = Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = Linear(config.d_model, config.d_model, bias=True)
        
        # RoPE for positional encoding
        if config.use_rope:
            self.rope = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_seq_len=config.max_seq_len,
                base=config.rope_base
            )
        else:
            self.rope = None
        
        # Dropout
        self.dropout = Dropout(config.dropout)
        
        # Initialize weights with proper scaling
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Xavier/Glorot initialization for linear layers
        std = np.sqrt(2.0 / (self.d_model + self.d_model))
        
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            layer.weight.data = np.random.normal(0, std, layer.weight.shape).astype(np.float32)
            if layer.bias is not None:
                layer.bias.data = np.zeros(layer.bias.shape, dtype=np.float32)
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with RoPE-enhanced attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            start_pos: Starting position for RoPE (inference mode)
            use_cache: Whether to use KV caching (inference)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = x.shape
        
        if d_model != self.d_model:
            raise ModelError(f"Input d_model {d_model} != expected {self.d_model}")
        
        # Linear projections
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        q = self._reshape_for_multihead(q)
        k = self._reshape_for_multihead(k)
        v = self._reshape_for_multihead(v)
        
        # Apply RoPE if enabled
        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=start_pos)
        
        # Compute attention scores
        # q: (batch_size, num_heads, seq_len, head_dim)
        # k: (batch_size, num_heads, seq_len, head_dim)
        # scores: (batch_size, num_heads, seq_len, seq_len)
        k_transposed = Tensor(
            np.transpose(k.data, (0, 1, 3, 2)),  # Transpose last two dims
            requires_grad=k.requires_grad,
            name=f"k_transposed({k.name or 'tensor'})"
        )
        
        attention_scores = matmul(q, k_transposed)
        attention_scores = Tensor(
            attention_scores.data * self.scale,
            requires_grad=attention_scores.requires_grad,
            name=f"scaled_scores({attention_scores.name or 'tensor'})"
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Broadcast mask to attention scores shape
            mask_expanded = np.broadcast_to(
                attention_mask.data[:, None, None, :],  # (batch, 1, 1, seq_len)
                attention_scores.shape
            )
            # Apply large negative value where mask is 0
            masked_scores = np.where(
                mask_expanded,
                attention_scores.data,
                -1e9
            )
            attention_scores = Tensor(
                masked_scores,
                requires_grad=attention_scores.requires_grad,
                name=f"masked_scores({attention_scores.name or 'tensor'})"
            )
        
        # Apply softmax to get attention probabilities
        attention_probs = softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        # attention_probs: (batch_size, num_heads, seq_len, seq_len)
        # v: (batch_size, num_heads, seq_len, head_dim)
        # context: (batch_size, num_heads, seq_len, head_dim)
        context = matmul(attention_probs, v)
        
        # Reshape back to original format
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        context_reshaped = self._reshape_from_multihead(context)
        
        # Final linear projection
        output = self.out_proj(context_reshaped)
        
        return output, attention_probs if use_cache else None
    
    def _reshape_for_multihead(self, x: Tensor) -> Tensor:
        """Reshape tensor for multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape to separate heads
        reshaped_data = x.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        transposed_data = np.transpose(reshaped_data, (0, 2, 1, 3))
        
        return Tensor(
            transposed_data,
            requires_grad=x.requires_grad,
            name=f"multihead_reshaped({x.name or 'tensor'})"
        )
    
    def _reshape_from_multihead(self, x: Tensor) -> Tensor:
        """Reshape tensor from multi-head attention format.
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            Reshaped tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Transpose to (batch_size, seq_len, num_heads, head_dim)
        transposed_data = np.transpose(x.data, (0, 2, 1, 3))
        # Reshape to (batch_size, seq_len, d_model)
        reshaped_data = transposed_data.reshape(batch_size, seq_len, self.d_model)
        
        return Tensor(
            reshaped_data,
            requires_grad=x.requires_grad,
            name=f"multihead_unreshaped({x.name or 'tensor'})"
        )


class PreNormFeedForward(Module):
    """Feed-forward network with modern activations.
    
    This implements the feed-forward component with support for:
    - Multiple activation functions (GELU, SwiGLU, ReLU)
    - Proper weight initialization
    - Dropout for regularization
    """
    
    def __init__(self, config: PreNormTransformerConfig):
        """Initialize feed-forward network.
        
        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.config = config
        self.activation_type = config.activation
        
        if config.activation == "swiglu":
            # SwiGLU requires different architecture
            # Single projection to 2*d_ff for gating (split internally)
            self.up_proj = Linear(config.d_model, 2 * config.d_ff, bias=False)
            self.down_proj = Linear(config.d_ff, config.d_model, bias=False)
            self.gate_proj = None
        else:
            # Standard FFN with single hidden layer - enable fusion for activation
            activation_func = config.activation.lower() if config.activation.lower() in ['gelu', 'relu'] else None
            self.up_proj = Linear(config.d_model, config.d_ff, bias=True, 
                                activation=activation_func, enable_fusion=True)
            self.down_proj = Linear(config.d_ff, config.d_model, bias=True)
            self.gate_proj = None
        
        self.dropout = Dropout(config.dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Initialize with smaller std for stability
        std = np.sqrt(2.0 / self.config.d_model)
        
        layers = [self.up_proj, self.down_proj]
        
        for layer in layers:
            layer.weight.data = np.random.normal(0, std, layer.weight.shape).astype(np.float32)
            if layer.bias is not None:
                layer.bias.data = np.zeros(layer.bias.shape, dtype=np.float32)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of same shape
        """
        if self.activation_type == "swiglu":
            # SwiGLU: projects to 2*d_ff then splits and gates
            up = self.up_proj(x)  # Projects to 2*d_ff
            hidden = swiglu(up)  # Splits, applies SiLU, and multiplies
        else:
            # Standard FFN
            hidden = self.up_proj(x)
            
            # Check if activation is already fused in the linear layer
            if hasattr(self.up_proj, 'activation') and self.up_proj.activation:
                # Activation is already applied via fusion - no need to apply again
                pass
            else:
                # Apply activation manually
                if self.activation_type == "gelu":
                    hidden = gelu(hidden)
                elif self.activation_type == "relu":
                    # Apply ReLU
                    hidden = Tensor(
                        np.maximum(0, hidden.data),
                        requires_grad=hidden.requires_grad,
                        name=f"relu({hidden.name or 'tensor'})"
                    )
                else:
                    raise ModelError(f"Unknown activation: {self.activation_type}")
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Down projection
        output = self.down_proj(hidden)
        
        return output


class PreNormTransformerLayer(Module):
    """Pre-Norm Transformer layer.
    
    The key difference from post-norm is that layer normalization is applied
    BEFORE the sub-layers (attention and FFN) rather than after. This provides:
    - More stable training gradients
    - Better performance on deep networks
    - Reduced need for learning rate warmup
    """
    
    def __init__(self, config: PreNormTransformerConfig):
        """Initialize pre-norm transformer layer.
        
        Args:
            config: Transformer configuration 
        """
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = RoPEMultiHeadAttention(config)
        
        # Feed-forward network
        self.feed_forward = PreNormFeedForward(config)
        
        # Layer normalization (applied before sub-layers)
        if config.normalization == "rmsnorm":
            self.norm1 = RMSNorm(config.d_model, eps=config.eps)
            self.norm2 = RMSNorm(config.d_model, eps=config.eps)
        else:
            self.norm1 = LayerNorm(config.d_model, eps=config.eps)
            self.norm2 = LayerNorm(config.d_model, eps=config.eps)
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        start_pos: int = 0
    ) -> Tensor:
        """Forward pass through pre-norm transformer layer.
        
        Architecture:
        x -> norm1 -> attention -> residual connection
          -> norm2 -> ffn -> residual connection -> output
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            start_pos: Starting position for RoPE
            
        Returns:
            Output tensor
        """
        # Pre-norm self-attention with residual connection
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attention(x_norm, attention_mask, start_pos)
        x = Tensor(
            residual.data + attn_output.data,
            requires_grad=x.requires_grad or attn_output.requires_grad,
            name=f"attn_residual({x.name or 'tensor'})"
        )
        
        # Pre-norm feed-forward with residual connection
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.feed_forward(x_norm)
        x = Tensor(
            residual.data + ffn_output.data,
            requires_grad=x.requires_grad or ffn_output.requires_grad,
            name=f"ffn_residual({x.name or 'tensor'})"
        )
        
        return x


class PreNormTransformer(Module):
    """Modern Pre-Norm Transformer with RoPE and advanced components.
    
    This implements a state-of-the-art transformer architecture with:
    - Pre-norm architecture for stable training
    - RoPE for superior positional encoding  
    - Modern activations (GELU, SwiGLU)
    - Advanced normalization (LayerNorm, RMSNorm)
    - Proper weight initialization and scaling
    - Mathematical correctness improvements
    """
    
    def __init__(self, config: Optional[PreNormTransformerConfig] = None, **kwargs):
        """Initialize Pre-Norm Transformer.
        
        Args:
            config: Optional configuration object
            **kwargs: Configuration parameters (used if config is None)
        """
        super().__init__()
        
        if config is None:
            config = PreNormTransformerConfig(**kwargs)
        
        self.config = config
        
        # Token embeddings
        self.token_embedding = Parameter(
            np.random.normal(0, 0.02, (config.vocab_size, config.d_model)).astype(np.float32),
            name="token_embedding"
        )
        
        # Transformer layers
        self.layers = []
        for i in range(config.num_layers):
            layer = PreNormTransformerLayer(config)
            self.layers.append(layer)
            # Register as submodule
            setattr(self, f'layer_{i}', layer)
        
        # Final layer norm (important for pre-norm architecture)
        if config.normalization == "rmsnorm":
            self.final_norm = RMSNorm(config.d_model, eps=config.eps)
        else:
            self.final_norm = LayerNorm(config.d_model, eps=config.eps)
        
        # Output projection (language modeling head)
        if config.tie_embeddings:
            # Tie input and output embeddings (parameter sharing)
            self.output_projection = None
        else:
            self.output_projection = Linear(config.d_model, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized Pre-Norm Transformer: {self._count_parameters():,} parameters")
    
    def _init_weights(self):
        """Initialize all weights with proper scaling."""
        # Token embeddings
        std = 0.02
        self.token_embedding.data = np.random.normal(0, std, self.token_embedding.shape).astype(np.float32)
        
        # Output projection (if not tied)
        if self.output_projection is not None:
            self.output_projection.weight.data = np.random.normal(
                0, std, self.output_projection.weight.shape
            ).astype(np.float32)
    
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total
    
    def get_embeddings(self, input_ids: Tensor) -> Tensor:
        """Get token embeddings with optional scaling.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Gather embeddings
        embeddings_data = self.token_embedding.data[input_ids.data.astype(int)]
        
        # Scale embeddings by sqrt(d_model) if configured
        if self.config.scale_embeddings:
            embeddings_data = embeddings_data * np.sqrt(self.config.d_model)
        
        embeddings = Tensor(
            embeddings_data,
            requires_grad=self.token_embedding.requires_grad,
            name=f"token_embeddings({input_ids.name or 'tensor'})"
        )
        
        return embeddings
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        start_pos: int = 0,
        output_hidden_states: bool = False
    ) -> Dict[str, Tensor]:
        """Forward pass through the transformer.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (1 for attend, 0 for ignore)
            start_pos: Starting position for RoPE (inference mode)
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary containing:
            - 'logits': Output logits of shape (batch_size, seq_len, vocab_size)
            - 'hidden_states': All layer hidden states (if requested)
            - 'last_hidden_state': Final hidden state before output projection
        """
        # Get token embeddings
        x = self.get_embeddings(input_ids)
        
        # Apply input dropout
        x = self.dropout(x)
        
        # Store hidden states if requested
        all_hidden_states = [] if output_hidden_states else None
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(x)
            
            x = layer(x, attention_mask, start_pos)
        
        # Final layer normalization (crucial for pre-norm)
        x = self.final_norm(x)
        
        if output_hidden_states:
            all_hidden_states.append(x)
        
        # Output projection
        if self.config.tie_embeddings:
            # Use transposed token embeddings as output weights
            output_weight = Tensor(
                self.token_embedding.data.T,  # (d_model, vocab_size)  
                requires_grad=self.token_embedding.requires_grad,
                name="tied_output_weight"
            )
            logits = matmul(x, output_weight)
        else:
            logits = self.output_projection(x)
        
        result = {
            'logits': logits,
            'last_hidden_state': x,
        }
        
        if output_hidden_states:
            result['hidden_states'] = all_hidden_states
        
        return result
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return (f"d_model={self.config.d_model}, num_layers={self.config.num_layers}, "
                f"num_heads={self.config.num_heads}, vocab_size={self.config.vocab_size}")


@register_model(
    name='prenorm_transformer',
    description='Pre-Norm Transformer with RoPE and modern components',
    paper_url='https://arxiv.org/abs/2002.04745',  # Pre-norm paper
    pretrained_configs={
        'small': {'d_model': 512, 'num_layers': 6, 'num_heads': 8},
        'base': {'d_model': 768, 'num_layers': 12, 'num_heads': 12}, 
        'large': {'d_model': 1024, 'num_layers': 24, 'num_heads': 16}
    },
    default_config='base',
    tags=['transformer', 'prenorm', 'rope', 'language'],
    aliases=['prenorm', 'modern_transformer']
)
class RegisteredPreNormTransformer(PreNormTransformer):
    """Registered version of Pre-Norm Transformer."""
    
    def __init__(self, **kwargs):
        config = PreNormTransformerConfig(**kwargs)
        super().__init__(config)


# Convenience functions
def prenorm_transformer_small(**kwargs):
    """Create small Pre-Norm Transformer."""
    defaults = {'d_model': 512, 'num_layers': 6, 'num_heads': 8, 'd_ff': 2048}
    defaults.update(kwargs)
    return PreNormTransformer(PreNormTransformerConfig(**defaults))


def prenorm_transformer_base(**kwargs):
    """Create base Pre-Norm Transformer.""" 
    defaults = {'d_model': 768, 'num_layers': 12, 'num_heads': 12, 'd_ff': 3072}
    defaults.update(kwargs)
    return PreNormTransformer(PreNormTransformerConfig(**defaults))


def prenorm_transformer_large(**kwargs):
    """Create large Pre-Norm Transformer."""
    defaults = {'d_model': 1024, 'num_layers': 24, 'num_heads': 16, 'd_ff': 4096}
    defaults.update(kwargs)
    return PreNormTransformer(PreNormTransformerConfig(**defaults))


# Aliases
ModernTransformer = PreNormTransformer
RoPETransformer = PreNormTransformer