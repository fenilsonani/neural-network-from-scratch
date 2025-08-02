"""Spatial dropout implementations for convolutional layers."""

from typing import Optional

import numpy as np

from ..core import Module, Tensor
from ..core.tensor import GradientFunction
from ..exceptions import LayerError, handle_exception


class SpatialDropout1d(Module):
    """Spatial dropout for 1D convolutional layers.
    
    Randomly sets entire channels to zero during training to prevent overfitting
    and promote feature independence. Each channel is dropped independently.
    
    Args:
        p: Probability of an element to be zeroed
        inplace: Can optionally do the operation in-place
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False, name: Optional[str] = None):
        super().__init__()
        
        if p < 0.0 or p > 1.0:
            raise LayerError(f"Dropout probability must be between 0 and 1, got {p}")
            
        self.p = p
        self.inplace = inplace
        self.name = name or f"SpatialDropout1d(p={p})"
        
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through spatial dropout layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Output tensor with spatial dropout applied
        """
        if len(x.shape) != 3:
            raise LayerError(f"Expected 3D input (batch, channels, length), got {len(x.shape)}D")
            
        if not self.training or self.p == 0.0:
            return x
            
        batch_size, channels, length = x.shape
        
        # Generate channel-wise dropout mask
        # Each channel is either completely kept (1) or completely dropped (0)
        # Same mask pattern applied across all batch samples for consistency
        keep_prob = 1.0 - self.p
        if keep_prob == 0.0:
            # When p=1.0, all channels are dropped
            channel_mask = np.zeros((1, channels, 1))
        else:
            channel_mask = np.random.binomial(1, keep_prob, (1, channels, 1)) / keep_prob
        # Broadcast to all batch samples
        mask = np.broadcast_to(channel_mask, x.shape)
        
        if self.inplace:
            x.data *= mask
            output = x
        else:
            output_data = x.data * mask
            output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            def backward_fn(grad_output):
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data)
                    x.grad += grad_output * mask
            
            output._grad_fn = GradientFunction(backward_fn, [x], "spatial_dropout")
        
        return output


class SpatialDropout2d(Module):
    """Spatial dropout for 2D convolutional layers.
    
    Randomly sets entire channels to zero during training to prevent overfitting
    and promote feature independence. Each channel is dropped independently.
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False, name: Optional[str] = None):
        super().__init__()
        
        if p < 0.0 or p > 1.0:
            raise LayerError(f"Dropout probability must be between 0 and 1, got {p}")
            
        self.p = p
        self.inplace = inplace
        self.name = name or f"SpatialDropout2d(p={p})"
        
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through spatial dropout layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor with spatial dropout applied
        """
        if len(x.shape) != 4:
            raise LayerError(f"Expected 4D input (batch, channels, height, width), got {len(x.shape)}D")
            
        if not self.training or self.p == 0.0:
            return x
            
        batch_size, channels, height, width = x.shape
        
        # Generate channel-wise dropout mask
        # Each channel is either completely kept (1) or completely dropped (0)
        # Same mask pattern applied across all batch samples for consistency
        keep_prob = 1.0 - self.p
        if keep_prob == 0.0:
            # When p=1.0, all channels are dropped
            channel_mask = np.zeros((1, channels, 1, 1))
        else:
            channel_mask = np.random.binomial(1, keep_prob, (1, channels, 1, 1)) / keep_prob
        # Broadcast to all batch samples
        mask = np.broadcast_to(channel_mask, x.shape)
        
        if self.inplace:
            x.data *= mask
            output = x
        else:
            output_data = x.data * mask
            output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            def backward_fn(grad_output):
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data)
                    x.grad += grad_output * mask
            
            output._grad_fn = GradientFunction(backward_fn, [x], "spatial_dropout")
        
        return output


class SpatialDropout3d(Module):
    """Spatial dropout for 3D convolutional layers.
    
    Randomly sets entire channels to zero during training to prevent overfitting
    and promote feature independence. Each channel is dropped independently.
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False, name: Optional[str] = None):
        super().__init__()
        
        if p < 0.0 or p > 1.0:
            raise LayerError(f"Dropout probability must be between 0 and 1, got {p}")
            
        self.p = p
        self.inplace = inplace
        self.name = name or f"SpatialDropout3d(p={p})"
        
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through spatial dropout layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
            
        Returns:
            Output tensor with spatial dropout applied
        """
        if len(x.shape) != 5:
            raise LayerError(f"Expected 5D input (batch, channels, depth, height, width), got {len(x.shape)}D")
            
        if not self.training or self.p == 0.0:
            return x
            
        batch_size, channels, depth, height, width = x.shape
        
        # Generate channel-wise dropout mask
        # Each channel is either completely kept (1) or completely dropped (0)
        # Same mask pattern applied across all batch samples for consistency
        keep_prob = 1.0 - self.p
        if keep_prob == 0.0:
            # When p=1.0, all channels are dropped
            channel_mask = np.zeros((1, channels, 1, 1, 1))
        else:
            channel_mask = np.random.binomial(1, keep_prob, (1, channels, 1, 1, 1)) / keep_prob
        # Broadcast to all batch samples
        mask = np.broadcast_to(channel_mask, x.shape)
        
        if self.inplace:
            x.data *= mask
            output = x
        else:
            output_data = x.data * mask
            output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            def backward_fn(grad_output):
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data)
                    x.grad += grad_output * mask
            
            output._grad_fn = GradientFunction(backward_fn, [x], "spatial_dropout")
        
        return output