"""Transformer components (placeholder)."""

from ..core import Module, Tensor


class TransformerBlock(Module):
    """Transformer block (placeholder)."""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
    
    def forward(self, x: Tensor) -> Tensor:
        return x


class TransformerEncoder(Module):
    """Transformer encoder (placeholder)."""
    
    def __init__(self, d_model: int, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
    
    def forward(self, x: Tensor) -> Tensor:
        return x