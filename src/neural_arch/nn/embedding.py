"""Embedding layer implementation."""

import numpy as np
from typing import Optional, Dict

from ..core import Module, Parameter, Tensor
from ..functional import matmul
from ..exceptions import LayerError, handle_exception


class Embedding(Module):
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, embed_dim: int, name: Optional[str] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.name = name or f"Embedding({vocab_size}, {embed_dim})"
        
        # Initialize embedding matrix
        scale = 1.0 / np.sqrt(embed_dim)
        weight_data = np.random.uniform(-scale, scale, (vocab_size, embed_dim)).astype(np.float32)
        self.weight = Parameter(weight_data, name=f"{self.name}.weight")
    
    @handle_exception
    def forward(self, indices: np.ndarray) -> Tensor:
        """Forward pass through embedding layer."""
        # Simple embedding lookup
        embedded_data = self.weight.data[indices.flatten()]
        
        # Reshape to match input + embedding dimension
        output_shape = list(indices.shape) + [self.embed_dim]
        embedded_data = embedded_data.reshape(output_shape)
        
        result = Tensor(embedded_data, requires_grad=self.weight.requires_grad)
        
        # Set up gradient computation for embedding
        if self.weight.requires_grad:
            def backward():
                if result.grad is not None:
                    # Accumulate gradients for each embedded token
                    flat_indices = indices.flatten()
                    flat_grad = result.grad.reshape(-1, self.embed_dim)
                    
                    # Create gradient for weight matrix
                    weight_grad = np.zeros_like(self.weight.data)
                    for i, idx in enumerate(flat_indices):
                        weight_grad[idx] += flat_grad[i]
                    
                    self.weight.backward(weight_grad)
                    if hasattr(self.weight, '_backward'):
                        self.weight._backward()
            
            result._backward = backward
        
        return result
    
    def __call__(self, indices: np.ndarray) -> Tensor:
        """Make layer callable."""
        return self.forward(indices)