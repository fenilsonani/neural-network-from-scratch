"""Transformer components."""

from typing import Optional
import numpy as np
from ..core import Module, Tensor
from ..functional import relu
from .attention import MultiHeadAttention
from .linear import Linear
from .normalization import LayerNorm
from .dropout import Dropout


class TransformerBlock(Module):
    """Transformer block with multi-head attention and feed-forward network.
    
    This implements a standard transformer block with:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Residual connections
    - Layer normalization
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        
        # Feed-forward network
        self.ffn1 = Linear(d_model, d_ff)
        self.ffn2 = Linear(d_ff, d_model)
        
        # Normalization layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        # Activation
        self.activation = activation
    
    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=mask)
        x = self.dropout(x)
        x = Tensor(x.data + residual.data, requires_grad=x.requires_grad)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ffn1(x)
        
        # Apply activation
        if self.activation == 'relu':
            x = relu(x)
        # Could add GELU here if implemented
        
        x = self.ffn2(x)
        x = self.dropout(x)
        x = Tensor(x.data + residual.data, requires_grad=x.requires_grad)
        
        return x
    
    def parameters(self):
        """Get all parameters."""
        params = {}
        
        # Attention parameters
        for name, param in self.self_attn.parameters().items():
            params[f'attn_{name}'] = param
        
        # FFN parameters
        for name, param in self.ffn1.parameters().items():
            params[f'ffn1_{name}'] = param
        for name, param in self.ffn2.parameters().items():
            params[f'ffn2_{name}'] = param
            
        # Norm parameters
        for name, param in self.norm1.parameters().items():
            params[f'norm1_{name}'] = param
        for name, param in self.norm2.parameters().items():
            params[f'norm2_{name}'] = param
            
        return params


class TransformerEncoder(Module):
    """Stack of transformer encoder blocks."""
    
    def __init__(
        self, 
        d_model: int, 
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """Initialize transformer encoder.
        
        Args:
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads per layer
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Stack of encoder layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, num_heads, d_ff, dropout)
            )
        
        # Final normalization
        self.norm = LayerNorm(d_model)
    
    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """Forward pass through encoder.
        
        Args:
            x: Input tensor
            mask: Optional padding mask
            
        Returns:
            Encoded representation
        """
        # Pass through all layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x
    
    def parameters(self):
        """Get all parameters."""
        params = {}
        
        # Layer parameters
        for i, layer in enumerate(self.layers):
            for name, param in layer.parameters().items():
                params[f'layer{i}_{name}'] = param
                
        # Final norm parameters
        for name, param in self.norm.parameters().items():
            params[f'final_norm_{name}'] = param
            
        return params


class TransformerDecoderBlock(Module):
    """Transformer decoder block with masked self-attention and cross-attention."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """Initialize decoder block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        
        # Self-attention (masked)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        
        # Feed-forward
        self.ffn1 = Linear(d_model, d_ff)
        self.ffn2 = Linear(d_ff, d_model)
        
        # Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = Dropout(dropout)
    
    def forward(
        self, 
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[np.ndarray] = None,
        memory_mask: Optional[np.ndarray] = None
    ) -> Tensor:
        """Forward pass through decoder block.
        
        Args:
            x: Target sequence tensor
            memory: Encoder output tensor
            tgt_mask: Mask for target self-attention
            memory_mask: Mask for encoder-decoder attention
            
        Returns:
            Decoded representation
        """
        # Masked self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=tgt_mask)
        x = self.dropout(x)
        x = Tensor(x.data + residual.data, requires_grad=x.requires_grad)
        
        # Cross-attention with encoder output
        residual = x
        x = self.norm2(x)
        # For now, simplified cross-attention (would need key/value from memory)
        x = self.cross_attn(x, mask=memory_mask)
        x = self.dropout(x)
        x = Tensor(x.data + residual.data, requires_grad=x.requires_grad)
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ffn1(x)
        x = relu(x)
        x = self.ffn2(x)
        x = self.dropout(x)
        x = Tensor(x.data + residual.data, requires_grad=x.requires_grad)
        
        return x
    
    def parameters(self):
        """Get all parameters."""
        params = {}
        
        # Self-attention parameters
        for name, param in self.self_attn.parameters().items():
            params[f'self_attn_{name}'] = param
            
        # Cross-attention parameters
        for name, param in self.cross_attn.parameters().items():
            params[f'cross_attn_{name}'] = param
        
        # FFN parameters
        for name, param in self.ffn1.parameters().items():
            params[f'ffn1_{name}'] = param
        for name, param in self.ffn2.parameters().items():
            params[f'ffn2_{name}'] = param
            
        # Norm parameters
        for name, param in self.norm1.parameters().items():
            params[f'norm1_{name}'] = param
        for name, param in self.norm2.parameters().items():
            params[f'norm2_{name}'] = param
        for name, param in self.norm3.parameters().items():
            params[f'norm3_{name}'] = param
            
        return params