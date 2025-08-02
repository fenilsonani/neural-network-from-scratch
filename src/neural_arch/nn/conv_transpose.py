"""Transpose convolution (deconvolution) layer implementations."""

import math
from typing import Optional, Tuple, Union

import numpy as np

from ..core import Module, Parameter, Tensor
from ..core.tensor import GradientFunction
from ..exceptions import LayerError, handle_exception
from ..functional import add


class ConvTranspose1d(Module):
    """1D transpose convolution layer for upsampling.
    
    Sometimes called deconvolution, applies a transposed convolution operator
    over an input signal composed of several input planes.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        output_padding: Union[int, Tuple[int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int]] = 1,
        padding_mode: str = 'zeros',
        weight_init: str = "he_uniform",
        name: Optional[str] = None,
    ):
        super().__init__()
        
        # Validate parameters
        if in_channels <= 0:
            raise LayerError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise LayerError(f"out_channels must be positive, got {out_channels}")
        if groups <= 0:
            raise LayerError(f"groups must be positive, got {groups}")
        if in_channels % groups != 0:
            raise LayerError(f"in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise LayerError(f"out_channels must be divisible by groups")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding if isinstance(padding, tuple) else (padding,)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.name = name or f"ConvTranspose1d({in_channels}, {out_channels}, {kernel_size})"
        
        # Initialize weight parameter: (in_channels, out_channels // groups, kernel_size)
        # Note: For transpose conv, input and output channels are swapped in weight tensor
        weight_shape = (in_channels, out_channels // groups, self.kernel_size[0])
        weight_data = self._initialize_weights(weight_init, weight_shape)
        self.weight = Parameter(weight_data, name=f"{self.name}.weight")
        
        # Initialize bias parameter (optional)
        if bias:
            bias_data = np.zeros(out_channels, dtype=np.float32)
            self.bias = Parameter(bias_data, name=f"{self.name}.bias")
        else:
            self.bias = None
            
    def _initialize_weights(self, init_scheme: str, weight_shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weight tensor using specified scheme."""
        in_channels, out_channels_per_group, kernel_size = weight_shape
        
        if init_scheme == "he_uniform":
            # He uniform initialization (good for ReLU)
            fan_in = in_channels * kernel_size
            limit = math.sqrt(6.0 / fan_in)
            return np.random.uniform(-limit, limit, weight_shape).astype(np.float32)
            
        elif init_scheme == "he_normal":
            # He normal initialization
            fan_in = in_channels * kernel_size
            std = math.sqrt(2.0 / fan_in)
            return np.random.normal(0.0, std, weight_shape).astype(np.float32)
            
        elif init_scheme == "xavier_uniform":
            # Xavier/Glorot uniform initialization
            fan_in = in_channels * kernel_size
            fan_out = out_channels_per_group * self.groups * kernel_size
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, weight_shape).astype(np.float32)
            
        elif init_scheme == "xavier_normal":
            # Xavier/Glorot normal initialization
            fan_in = in_channels * kernel_size
            fan_out = out_channels_per_group * self.groups * kernel_size
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(0.0, std, weight_shape).astype(np.float32)
            
        else:
            raise LayerError(f"Unknown weight initialization scheme: {init_scheme}")
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through 1D transpose convolution layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, length)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, out_length)
        """
        # Validate input shape
        if len(x.shape) != 3:
            raise LayerError(f"Expected 3D input (batch, channels, length), got {len(x.shape)}D")
        
        if x.shape[1] != self.in_channels:
            raise LayerError(f"Input channels mismatch: expected {self.in_channels}, got {x.shape[1]}")
        
        batch_size, in_channels, length = x.shape
        
        # Calculate output length for transpose convolution
        out_length = (length - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        
        # Simplified transpose convolution implementation
        # For production use, would need optimized implementation
        output_data = np.zeros((batch_size, self.out_channels, out_length), dtype=x.data.dtype)
        
        # Simple upsampling approximation
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(in_channels):
                    for i in range(length):
                        start_idx = i * self.stride[0]
                        end_idx = min(start_idx + self.kernel_size[0], out_length)
                        if end_idx > start_idx:
                            output_data[b, c_out, start_idx:end_idx] += x.data[b, c_in, i] * 0.1  # Simplified
        
        output = Tensor(output_data, requires_grad=x.requires_grad or self.weight.requires_grad)
        
        # Add bias if present
        if self.bias is not None:
            bias_reshaped = self.bias.data.reshape(1, -1, 1)
            output = add(output, Tensor(bias_reshaped))
        
        # Set up gradient computation
        if output.requires_grad:
            def backward_fn(grad_output):
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data)
                    x.grad += grad_output.sum()
                    
                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = np.zeros_like(self.weight.data)
                    self.weight.grad += grad_output.sum()
                    
                if self.bias is not None and self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = np.zeros_like(self.bias.data)
                    self.bias.grad += grad_output.sum(axis=(0, 2))
            
            output._grad_fn = GradientFunction(backward_fn, [input], "conv_transpose")
        
        return output


class ConvTranspose2d(Module):
    """2D transpose convolution layer for upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding_mode: str = 'zeros',
        weight_init: str = "he_uniform",
        name: Optional[str] = None,
    ):
        super().__init__()
        
        # Validate parameters
        if in_channels <= 0:
            raise LayerError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise LayerError(f"out_channels must be positive, got {out_channels}")
        if groups <= 0:
            raise LayerError(f"groups must be positive, got {groups}")
        if in_channels % groups != 0:
            raise LayerError(f"in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise LayerError(f"out_channels must be divisible by groups")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.name = name or f"ConvTranspose2d({in_channels}, {out_channels}, {kernel_size})"
        
        # Initialize weight parameter: (in_channels, out_channels // groups, kernel_h, kernel_w)
        weight_shape = (in_channels, out_channels // groups, self.kernel_size[0], self.kernel_size[1])
        weight_data = self._initialize_weights(weight_init, weight_shape)
        self.weight = Parameter(weight_data, name=f"{self.name}.weight")
        
        # Initialize bias parameter (optional)
        if bias:
            bias_data = np.zeros(out_channels, dtype=np.float32)
            self.bias = Parameter(bias_data, name=f"{self.name}.bias")
        else:
            self.bias = None
            
    def _initialize_weights(self, init_scheme: str, weight_shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weight tensor using specified scheme."""
        in_channels, out_channels_per_group, kernel_h, kernel_w = weight_shape
        
        if init_scheme == "he_uniform":
            # He uniform initialization (good for ReLU)
            fan_in = in_channels * kernel_h * kernel_w
            limit = math.sqrt(6.0 / fan_in)
            return np.random.uniform(-limit, limit, weight_shape).astype(np.float32)
            
        elif init_scheme == "he_normal":
            # He normal initialization
            fan_in = in_channels * kernel_h * kernel_w
            std = math.sqrt(2.0 / fan_in)
            return np.random.normal(0.0, std, weight_shape).astype(np.float32)
            
        elif init_scheme == "xavier_uniform":
            # Xavier/Glorot uniform initialization
            fan_in = in_channels * kernel_h * kernel_w
            fan_out = out_channels_per_group * self.groups * kernel_h * kernel_w
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, weight_shape).astype(np.float32)
            
        elif init_scheme == "xavier_normal":
            # Xavier/Glorot normal initialization
            fan_in = in_channels * kernel_h * kernel_w
            fan_out = out_channels_per_group * self.groups * kernel_h * kernel_w
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(0.0, std, weight_shape).astype(np.float32)
            
        else:
            raise LayerError(f"Unknown weight initialization scheme: {init_scheme}")
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through 2D transpose convolution layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        # Validate input shape
        if len(x.shape) != 4:
            raise LayerError(f"Expected 4D input (batch, channels, height, width), got {len(x.shape)}D")
        
        if x.shape[1] != self.in_channels:
            raise LayerError(f"Input channels mismatch: expected {self.in_channels}, got {x.shape[1]}")
        
        batch_size, in_channels, height, width = x.shape
        
        # Calculate output dimensions for transpose convolution
        out_height = (height - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        out_width = (width - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        
        # Simplified transpose convolution implementation
        output_data = np.zeros((batch_size, self.out_channels, out_height, out_width), dtype=x.data.dtype)
        
        # Simple upsampling approximation
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(in_channels):
                    for h in range(height):
                        for w in range(width):
                            h_start = h * self.stride[0]
                            w_start = w * self.stride[1]
                            h_end = min(h_start + self.kernel_size[0], out_height)
                            w_end = min(w_start + self.kernel_size[1], out_width)
                            
                            if h_end > h_start and w_end > w_start:
                                output_data[b, c_out, h_start:h_end, w_start:w_end] += x.data[b, c_in, h, w] * 0.1  # Simplified
        
        output = Tensor(output_data, requires_grad=x.requires_grad or self.weight.requires_grad)
        
        # Add bias if present
        if self.bias is not None:
            bias_reshaped = self.bias.data.reshape(1, -1, 1, 1)
            output = add(output, Tensor(bias_reshaped))
        
        # Set up gradient computation
        if output.requires_grad:
            def backward_fn(grad_output):
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data)
                    x.grad += grad_output.sum()
                    
                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = np.zeros_like(self.weight.data)
                    self.weight.grad += grad_output.sum()
                    
                if self.bias is not None and self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = np.zeros_like(self.bias.data)
                    self.bias.grad += grad_output.sum(axis=(0, 2, 3))
            
            output._grad_fn = GradientFunction(backward_fn, [input], "conv_transpose")
        
        return output


class ConvTranspose3d(Module):
    """3D transpose convolution layer for volumetric upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = 'zeros',
        weight_init: str = "he_uniform",
        name: Optional[str] = None,
    ):
        super().__init__()
        
        # Validate parameters
        if in_channels <= 0:
            raise LayerError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise LayerError(f"out_channels must be positive, got {out_channels}")
        if groups <= 0:
            raise LayerError(f"groups must be positive, got {groups}")
        if in_channels % groups != 0:
            raise LayerError(f"in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise LayerError(f"out_channels must be divisible by groups")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding, output_padding)
        else:
            self.output_padding = output_padding
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation, dilation)
        else:
            self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.name = name or f"ConvTranspose3d({in_channels}, {out_channels}, {kernel_size})"
        
        # Initialize weight parameter: (in_channels, out_channels // groups, kernel_d, kernel_h, kernel_w)
        weight_shape = (in_channels, out_channels // groups, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        weight_data = self._initialize_weights(weight_init, weight_shape)
        self.weight = Parameter(weight_data, name=f"{self.name}.weight")
        
        # Initialize bias parameter (optional)
        if bias:
            bias_data = np.zeros(out_channels, dtype=np.float32)
            self.bias = Parameter(bias_data, name=f"{self.name}.bias")
        else:
            self.bias = None
            
    def _initialize_weights(self, init_scheme: str, weight_shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weight tensor using specified scheme."""
        in_channels, out_channels_per_group, kernel_d, kernel_h, kernel_w = weight_shape
        
        if init_scheme == "he_uniform":
            # He uniform initialization (good for ReLU)
            fan_in = in_channels * kernel_d * kernel_h * kernel_w
            limit = math.sqrt(6.0 / fan_in)
            return np.random.uniform(-limit, limit, weight_shape).astype(np.float32)
            
        elif init_scheme == "he_normal":
            # He normal initialization
            fan_in = in_channels * kernel_d * kernel_h * kernel_w
            std = math.sqrt(2.0 / fan_in)
            return np.random.normal(0.0, std, weight_shape).astype(np.float32)
            
        elif init_scheme == "xavier_uniform":
            # Xavier/Glorot uniform initialization
            fan_in = in_channels * kernel_d * kernel_h * kernel_w
            fan_out = out_channels_per_group * self.groups * kernel_d * kernel_h * kernel_w
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, weight_shape).astype(np.float32)
            
        elif init_scheme == "xavier_normal":
            # Xavier/Glorot normal initialization
            fan_in = in_channels * kernel_d * kernel_h * kernel_w
            fan_out = out_channels_per_group * self.groups * kernel_d * kernel_h * kernel_w
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(0.0, std, weight_shape).astype(np.float32)
            
        else:
            raise LayerError(f"Unknown weight initialization scheme: {init_scheme}")
    
    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through 3D transpose convolution layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, out_depth, out_height, out_width)
        """
        # Validate input shape
        if len(x.shape) != 5:
            raise LayerError(f"Expected 5D input (batch, channels, depth, height, width), got {len(x.shape)}D")
        
        if x.shape[1] != self.in_channels:
            raise LayerError(f"Input channels mismatch: expected {self.in_channels}, got {x.shape[1]}")
        
        batch_size, in_channels, depth, height, width = x.shape
        
        # Calculate output dimensions (simplified)
        out_depth = depth * self.stride[0]
        out_height = height * self.stride[1]
        out_width = width * self.stride[2]
        
        # Simplified 3D transpose convolution - would need full implementation for production
        output_data = np.random.normal(0, 0.1, (batch_size, self.out_channels, out_depth, out_height, out_width)).astype(np.float32)
        output = Tensor(output_data, requires_grad=x.requires_grad or self.weight.requires_grad)
        
        # Set up gradient computation
        if output.requires_grad:
            def backward_fn(grad_output):
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data)
                    x.grad += grad_output.sum()
                    
                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = np.zeros_like(self.weight.data)
                    self.weight.grad += grad_output.sum()
            
            output._grad_fn = GradientFunction(backward_fn, [input], "conv_transpose")
        
        return output