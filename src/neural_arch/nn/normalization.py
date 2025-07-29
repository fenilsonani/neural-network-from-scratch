"""Normalization layers."""

import numpy as np
from typing import Optional

from ..core import Module, Parameter, Tensor
from ..exceptions import LayerError


class LayerNorm(Module):
    """Layer normalization."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = Parameter(np.ones(normalized_shape, dtype=np.float32))
        self.beta = Parameter(np.zeros(normalized_shape, dtype=np.float32))
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Compute mean and variance
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        
        # Normalize
        normalized = (x.data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output_data = self.gamma.data * normalized + self.beta.data
        
        result = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Simplified gradient handling
        if x.requires_grad:
            def backward():
                if result.grad is not None:
                    # Simplified gradient computation
                    x.backward(result.grad)
                    if hasattr(x, '_backward'):
                        x._backward()
            result._backward = backward
        
        return result


class BatchNorm1d(Module):
    """1D Batch normalization (placeholder)."""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
    
    def forward(self, x: Tensor) -> Tensor:
        # Simplified implementation
        return x