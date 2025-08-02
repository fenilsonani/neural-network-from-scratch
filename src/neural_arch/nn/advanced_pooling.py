"""Advanced pooling layer implementations."""

from typing import Optional, Tuple, Union

import numpy as np

from ..core import Module, Tensor
from ..core.tensor import GradientFunction
from ..exceptions import LayerError, handle_exception


class AdaptiveAvgPool1d(Module):
    """1D adaptive average pooling.
    
    Performs adaptive average pooling that outputs a tensor of size output_size,
    regardless of the input size.
    
    Args:
        output_size: Target output size
    """
    
    def __init__(self, output_size: Union[int, Tuple[int]], name: Optional[str] = None):
        super().__init__()
        
        if isinstance(output_size, int):
            self.output_size = (output_size,)
        else:
            self.output_size = output_size
            
        if len(self.output_size) != 1:
            raise LayerError(f"AdaptiveAvgPool1d expects 1D output size, got {len(self.output_size)}D")
        
        if self.output_size[0] <= 0:
            raise LayerError(f"output_size must be positive, got {self.output_size[0]}")
            
        self.name = name or f"AdaptiveAvgPool1d({self.output_size[0]})"
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through adaptive average pooling 1D.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Output tensor of shape (batch_size, channels, output_size)
        """
        if len(x.shape) != 3:
            raise LayerError(f"Expected 3D input (batch, channels, length), got {len(x.shape)}D")
        
        batch_size, channels, length = x.shape
        output_length = self.output_size[0]
        
        # Calculate adaptive pooling
        stride = length / output_length
        
        output_data = np.zeros((batch_size, channels, output_length), dtype=x.data.dtype)
        
        for i in range(output_length):
            start = int(i * stride)
            end = int((i + 1) * stride)
            if end > length:
                end = length
            if start == end:
                output_data[:, :, i] = x.data[:, :, start]
            else:
                output_data[:, :, i] = np.mean(x.data[:, :, start:end], axis=2)
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            # Capture variables in closure scope to avoid reference issues
            captured_stride = stride
            captured_length = length
            captured_output_length = output_length
            
            def backward_fn(grad_output):
                if x.requires_grad:
                    # Initialize gradient if needed
                    grad_input = np.zeros_like(x.data)
                    
                    # Compute gradients for adaptive pooling
                    for i in range(captured_output_length):
                        start = int(i * captured_stride)
                        end = int((i + 1) * captured_stride)
                        if end > captured_length:
                            end = captured_length
                        if start != end:
                            pool_size = end - start
                            grad_input[:, :, start:end] += grad_output[:, :, i:i+1] / pool_size
                    
                    # Continue backward propagation
                    x.backward(grad_input)
            
            output._grad_fn = GradientFunction(backward_fn, [x], "pooling")
        
        return output


class AdaptiveAvgPool2d(Module):
    """2D adaptive average pooling.
    
    Performs adaptive average pooling that outputs a tensor of size output_size,
    regardless of the input size.
    """
    
    def __init__(self, output_size: Union[int, Tuple[int, int]], name: Optional[str] = None):
        super().__init__()
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
            
        if len(self.output_size) != 2:
            raise LayerError(f"AdaptiveAvgPool2d expects 2D output size, got {len(self.output_size)}D")
        
        if any(s <= 0 for s in self.output_size):
            raise LayerError(f"output_size must be positive, got {self.output_size}")
            
        self.name = name or f"AdaptiveAvgPool2d({self.output_size})"
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through adaptive average pooling 2D.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels, output_height, output_width)
        """
        if len(x.shape) != 4:
            raise LayerError(f"Expected 4D input (batch, channels, height, width), got {len(x.shape)}D")
        
        batch_size, channels, height, width = x.shape
        output_height, output_width = self.output_size
        
        # Calculate adaptive pooling
        stride_h = height / output_height
        stride_w = width / output_width
        
        output_data = np.zeros((batch_size, channels, output_height, output_width), dtype=x.data.dtype)
        
        for i in range(output_height):
            for j in range(output_width):
                start_h = int(i * stride_h)
                end_h = int((i + 1) * stride_h)
                start_w = int(j * stride_w)
                end_w = int((j + 1) * stride_w)
                
                if end_h > height:
                    end_h = height
                if end_w > width:
                    end_w = width
                
                if start_h == end_h or start_w == end_w:
                    output_data[:, :, i, j] = x.data[:, :, start_h, start_w]
                else:
                    output_data[:, :, i, j] = np.mean(x.data[:, :, start_h:end_h, start_w:end_w], axis=(2, 3))
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            # Capture variables in closure scope to avoid reference issues
            captured_stride_h = stride_h
            captured_stride_w = stride_w
            captured_height = height
            captured_width = width
            captured_output_height = output_height
            captured_output_width = output_width
            
            def backward_fn(grad_output):
                if x.requires_grad:
                    # Initialize gradient
                    grad_input = np.zeros_like(x.data)
                    
                    # Compute gradients for adaptive pooling
                    for i in range(captured_output_height):
                        for j in range(captured_output_width):
                            start_h = int(i * captured_stride_h)
                            end_h = int((i + 1) * captured_stride_h)
                            start_w = int(j * captured_stride_w)
                            end_w = int((j + 1) * captured_stride_w)
                            
                            if end_h > captured_height:
                                end_h = captured_height
                            if end_w > captured_width:
                                end_w = captured_width
                            
                            if start_h != end_h and start_w != end_w:
                                pool_size = (end_h - start_h) * (end_w - start_w)
                                grad_input[:, :, start_h:end_h, start_w:end_w] += grad_output[:, :, i:i+1, j:j+1] / pool_size
                    
                    # Continue backward propagation
                    x.backward(grad_input)
            
            output._grad_fn = GradientFunction(backward_fn, [x], "pooling")
        
        return output


class AdaptiveMaxPool1d(Module):
    """1D adaptive max pooling."""
    
    def __init__(self, output_size: Union[int, Tuple[int]], name: Optional[str] = None):
        super().__init__()
        
        if isinstance(output_size, int):
            self.output_size = (output_size,)
        else:
            self.output_size = output_size
            
        if len(self.output_size) != 1:
            raise LayerError(f"AdaptiveMaxPool1d expects 1D output size, got {len(self.output_size)}D")
        
        if self.output_size[0] <= 0:
            raise LayerError(f"output_size must be positive, got {self.output_size[0]}")
            
        self.name = name or f"AdaptiveMaxPool1d({self.output_size[0]})"
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through adaptive max pooling 1D."""
        if len(x.shape) != 3:
            raise LayerError(f"Expected 3D input (batch, channels, length), got {len(x.shape)}D")
        
        batch_size, channels, length = x.shape
        output_length = self.output_size[0]
        
        # Calculate adaptive pooling
        stride = length / output_length
        
        output_data = np.zeros((batch_size, channels, output_length), dtype=x.data.dtype)
        indices = np.zeros((batch_size, channels, output_length), dtype=np.int32)
        
        for i in range(output_length):
            start = int(i * stride)
            end = int((i + 1) * stride)
            if end > length:
                end = length
            if start == end:
                output_data[:, :, i] = x.data[:, :, start]
                indices[:, :, i] = start
            else:
                pool_region = x.data[:, :, start:end]
                max_indices = np.argmax(pool_region, axis=2)
                output_data[:, :, i] = np.max(pool_region, axis=2)
                indices[:, :, i] = start + max_indices
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            # Capture variables in closure scope to avoid reference issues
            captured_batch_size = batch_size
            captured_channels = channels
            captured_output_length = output_length
            captured_indices = indices.copy()
            
            def backward_fn(grad_output):
                if x.requires_grad:
                    # Initialize gradient
                    grad_input = np.zeros_like(x.data)
                    
                    # Compute gradients for max pooling
                    for b in range(captured_batch_size):
                        for c in range(captured_channels):
                            for i in range(captured_output_length):
                                idx = captured_indices[b, c, i]
                                grad_input[b, c, idx] += grad_output[b, c, i]
                    
                    # Continue backward propagation
                    x.backward(grad_input)
            
            output._grad_fn = GradientFunction(backward_fn, [x], "pooling")
        
        return output


class AdaptiveMaxPool2d(Module):
    """2D adaptive max pooling."""
    
    def __init__(self, output_size: Union[int, Tuple[int, int]], name: Optional[str] = None):
        super().__init__()
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
            
        if len(self.output_size) != 2:
            raise LayerError(f"AdaptiveMaxPool2d expects 2D output size, got {len(self.output_size)}D")
        
        if any(s <= 0 for s in self.output_size):
            raise LayerError(f"output_size must be positive, got {self.output_size}")
            
        self.name = name or f"AdaptiveMaxPool2d({self.output_size})"
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through adaptive max pooling 2D."""
        if len(x.shape) != 4:
            raise LayerError(f"Expected 4D input (batch, channels, height, width), got {len(x.shape)}D")
        
        batch_size, channels, height, width = x.shape
        output_height, output_width = self.output_size
        
        # Calculate adaptive pooling
        stride_h = height / output_height
        stride_w = width / output_width
        
        output_data = np.zeros((batch_size, channels, output_height, output_width), dtype=x.data.dtype)
        indices_h = np.zeros((batch_size, channels, output_height, output_width), dtype=np.int32)
        indices_w = np.zeros((batch_size, channels, output_height, output_width), dtype=np.int32)
        
        for i in range(output_height):
            for j in range(output_width):
                start_h = int(i * stride_h)
                end_h = int((i + 1) * stride_h)
                start_w = int(j * stride_w)
                end_w = int((j + 1) * stride_w)
                
                if end_h > height:
                    end_h = height
                if end_w > width:
                    end_w = width
                
                if start_h == end_h or start_w == end_w:
                    output_data[:, :, i, j] = x.data[:, :, start_h, start_w]
                    indices_h[:, :, i, j] = start_h
                    indices_w[:, :, i, j] = start_w
                else:
                    pool_region = x.data[:, :, start_h:end_h, start_w:end_w]
                    # Find max indices
                    flat_indices = np.argmax(pool_region.reshape(batch_size, channels, -1), axis=2)
                    h_indices = flat_indices // (end_w - start_w) + start_h
                    w_indices = flat_indices % (end_w - start_w) + start_w
                    
                    output_data[:, :, i, j] = np.max(pool_region, axis=(2, 3))
                    indices_h[:, :, i, j] = h_indices
                    indices_w[:, :, i, j] = w_indices
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            # Capture variables in closure scope to avoid reference issues
            captured_batch_size = batch_size
            captured_channels = channels
            captured_output_height = output_height
            captured_output_width = output_width
            captured_indices_h = indices_h.copy()
            captured_indices_w = indices_w.copy()
            
            def backward_fn(grad_output):
                if x.requires_grad:
                    # Initialize gradient
                    grad_input = np.zeros_like(x.data)
                    
                    # Compute gradients for max pooling
                    for b in range(captured_batch_size):
                        for c in range(captured_channels):
                            for i in range(captured_output_height):
                                for j in range(captured_output_width):
                                    h_idx = captured_indices_h[b, c, i, j]
                                    w_idx = captured_indices_w[b, c, i, j]
                                    grad_input[b, c, h_idx, w_idx] += grad_output[b, c, i, j]
                    
                    # Continue backward propagation
                    x.backward(grad_input)
            
            output._grad_fn = GradientFunction(backward_fn, [x], "pooling")
        
        return output


class GlobalAvgPool1d(Module):
    """Global average pooling for 1D inputs.
    
    Pools over the entire spatial dimension, reducing to size 1.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or "GlobalAvgPool1d()"
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through global average pooling 1D."""
        if len(x.shape) != 3:
            raise LayerError(f"Expected 3D input (batch, channels, length), got {len(x.shape)}D")
        
        # Average over spatial dimension
        output_data = np.mean(x.data, axis=2, keepdims=True)
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            def backward_fn(grad_output):
                if x.requires_grad:
                    # Distribute gradient equally across spatial dimension
                    spatial_size = x.shape[2]
                    grad_input = np.broadcast_to(grad_output / spatial_size, x.shape)
                    
                    # Continue backward propagation
                    x.backward(grad_input)
            
            output._grad_fn = GradientFunction(backward_fn, [x], "pooling")
        
        return output


class GlobalAvgPool2d(Module):
    """Global average pooling for 2D inputs.
    
    Pools over the entire spatial dimensions, reducing to size 1x1.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or "GlobalAvgPool2d()"
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through global average pooling 2D."""
        if len(x.shape) != 4:
            raise LayerError(f"Expected 4D input (batch, channels, height, width), got {len(x.shape)}D")
        
        # Average over spatial dimensions
        output_data = np.mean(x.data, axis=(2, 3), keepdims=True)
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            def backward_fn(grad_output):
                if x.requires_grad:
                    # Distribute gradient equally across spatial dimensions
                    spatial_size = x.shape[2] * x.shape[3]
                    grad_input = np.broadcast_to(grad_output / spatial_size, x.shape)
                    
                    # Continue backward propagation
                    x.backward(grad_input)
            
            output._grad_fn = GradientFunction(backward_fn, [x], "pooling")
        
        return output


class GlobalMaxPool1d(Module):
    """Global max pooling for 1D inputs."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or "GlobalMaxPool1d()"
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through global max pooling 1D."""
        if len(x.shape) != 3:
            raise LayerError(f"Expected 3D input (batch, channels, length), got {len(x.shape)}D")
        
        # Max over spatial dimension
        output_data = np.max(x.data, axis=2, keepdims=True)
        max_indices = np.argmax(x.data, axis=2)
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            # Capture variables in closure scope to avoid reference issues
            captured_max_indices = max_indices.copy()
            
            def backward_fn(grad_output):
                if x.requires_grad:
                    # Initialize gradient
                    grad_input = np.zeros_like(x.data)
                    
                    # Compute gradients for max pooling
                    batch_size, channels = captured_max_indices.shape
                    for b in range(batch_size):
                        for c in range(channels):
                            idx = captured_max_indices[b, c]
                            grad_input[b, c, idx] += grad_output[b, c, 0]
                    
                    # Continue backward propagation
                    x.backward(grad_input)
            
            output._grad_fn = GradientFunction(backward_fn, [x], "pooling")
        
        return output


class GlobalMaxPool2d(Module):
    """Global max pooling for 2D inputs."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or "GlobalMaxPool2d()"
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through global max pooling 2D."""
        if len(x.shape) != 4:
            raise LayerError(f"Expected 4D input (batch, channels, height, width), got {len(x.shape)}D")
        
        batch_size, channels, height, width = x.shape
        
        # Max over spatial dimensions
        output_data = np.max(x.data, axis=(2, 3), keepdims=True)
        
        # Find max indices
        flat_data = x.data.reshape(batch_size, channels, -1)
        flat_indices = np.argmax(flat_data, axis=2)
        max_indices_h = flat_indices // width
        max_indices_w = flat_indices % width
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            # Capture variables in closure scope to avoid reference issues
            captured_batch_size = batch_size
            captured_channels = channels
            captured_max_indices_h = max_indices_h.copy()
            captured_max_indices_w = max_indices_w.copy()
            
            def backward_fn(grad_output):
                if x.requires_grad:
                    # Initialize gradient
                    grad_input = np.zeros_like(x.data)
                    
                    # Compute gradients for max pooling
                    for b in range(captured_batch_size):
                        for c in range(captured_channels):
                            h_idx = captured_max_indices_h[b, c]
                            w_idx = captured_max_indices_w[b, c]
                            grad_input[b, c, h_idx, w_idx] += grad_output[b, c, 0, 0]
                    
                    # Continue backward propagation
                    x.backward(grad_input)
            
            output._grad_fn = GradientFunction(backward_fn, [x], "pooling")
        
        return output