"""Normalization layers with mathematically correct implementations."""

import numpy as np
from typing import Optional, Tuple

from ..core import Module, Parameter, Tensor, GradientFunction
from ..exceptions import LayerError
from ..functional.utils import memory_efficient_operation
import logging

logger = logging.getLogger(__name__)


class LayerNorm(Module):
    """Layer normalization with mathematically correct implementation.
    
    Implements the layer normalization as described in:
    "Layer Normalization" (https://arxiv.org/abs/1607.06450)
    
    Mathematical Definition:
        y = γ * (x - μ) / σ + β
        where μ = mean(x), σ = sqrt(var(x) + ε)
        
    Features:
    - Proper gradient computation for all parameters
    - Numerical stability with configurable epsilon
    - Support for different axis configurations
    - Enterprise-grade error handling
    """
    
    def __init__(
        self, 
        normalized_shape: int, 
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[str] = None
    ):
        """Initialize LayerNorm.
        
        Args:
            normalized_shape: Size of the feature dimension
            eps: Small value for numerical stability
            elementwise_affine: Whether to include learnable affine parameters
            bias: Whether to include bias parameter (β)
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        
        if normalized_shape <= 0:
            raise LayerError(f"normalized_shape must be positive, got {normalized_shape}")
        if eps <= 0:
            raise LayerError(f"eps must be positive, got {eps}")
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Initialize learnable parameters
        if elementwise_affine:
            # Scale parameter (γ)
            self.weight = Parameter(
                np.ones(normalized_shape, dtype=np.float32),
                name="layernorm.weight"
            )
            self.gamma = self.weight  # Alias
            
            # Shift parameter (β)
            if bias:
                self.bias = Parameter(
                    np.zeros(normalized_shape, dtype=np.float32),
                    name="layernorm.bias"
                )
                self.beta = self.bias  # Alias
            else:
                self.bias = None
                self.beta = None
        else:
            self.weight = None
            self.bias = None
            self.gamma = None
            self.beta = None
    
    @memory_efficient_operation
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with mathematically correct gradient computation.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor with same shape as input
        """
        # Validate input shape
        if x.shape[-1] != self.normalized_shape:
            raise LayerError(
                f"Input last dimension {x.shape[-1]} != normalized_shape {self.normalized_shape}"
            )
        
        # Compute statistics along the last dimension
        mean = np.mean(x.data, axis=-1, keepdims=True)
        # Use ddof=0 for population variance (matching PyTorch behavior)
        var = np.var(x.data, axis=-1, keepdims=True, ddof=0)
        
        # Compute normalized input with numerical stability
        std = np.sqrt(var + self.eps)
        normalized = (x.data - mean) / std
        
        # Apply affine transformation if enabled
        if self.elementwise_affine:
            if self.bias is not None:
                output_data = self.weight.data * normalized + self.bias.data
            else:
                output_data = self.weight.data * normalized
        else:
            output_data = normalized
        
        # Create result tensor
        requires_grad = x.requires_grad or (
            self.elementwise_affine and (
                self.weight.requires_grad or 
                (self.bias is not None and self.bias.requires_grad)
            )
        )
        
        result = Tensor(
            output_data,
            requires_grad=requires_grad,
            name=f"layernorm({x.name or 'tensor'})"
        )
        
        # Set up proper gradient computation
        if requires_grad:
            def backward_fn(grad_output: np.ndarray) -> None:
                """Mathematically correct backward pass for LayerNorm.
                
                Gradients computed according to the chain rule:
                ∂L/∂x = ∂L/∂y * ∂y/∂x
                ∂L/∂γ = ∂L/∂y * ∂y/∂γ = ∂L/∂y * normalized
                ∂L/∂β = ∂L/∂y * ∂y/∂β = ∂L/∂y
                """
                N = x.shape[-1]  # Feature dimension
                
                # Gradient w.r.t. input
                if x.requires_grad:
                    if self.elementwise_affine:
                        # When affine transformation is applied
                        grad_normalized = grad_output * self.weight.data
                    else:
                        grad_normalized = grad_output
                    
                    # Gradient through normalization
                    # This is the mathematically correct gradient computation
                    grad_var = np.sum(grad_normalized * (x.data - mean) * (-0.5) * (var + self.eps) ** (-1.5), axis=-1, keepdims=True)
                    grad_mean = np.sum(grad_normalized * (-1.0 / std), axis=-1, keepdims=True) + grad_var * np.sum(-2.0 * (x.data - mean), axis=-1, keepdims=True) / N
                    
                    grad_x = (grad_normalized / std) + (grad_var * 2.0 * (x.data - mean) / N) + (grad_mean / N)
                    
                    # Accumulate gradients for input
                    if x._grad is None:
                        x._grad = x._backend.from_numpy(grad_x)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        x._grad = x._backend.to_device(x._grad, device_str)
                    else:
                        grad_backend = x._backend.from_numpy(grad_x)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        grad_backend = x._backend.to_device(grad_backend, device_str)
                        x._grad = x._backend.add(x._grad, grad_backend)
                    
                    # Continue backward propagation
                    if x._grad_fn is not None:
                        x._grad_fn.apply(grad_x)
                
                # Gradient w.r.t. weight (γ)
                if self.elementwise_affine and self.weight.requires_grad:
                    grad_weight = np.sum(grad_output * normalized, axis=tuple(range(grad_output.ndim - 1)))
                    
                    if self.weight._grad is None:
                        self.weight._grad = self.weight._backend.from_numpy(grad_weight)
                    else:
                        grad_backend = self.weight._backend.from_numpy(grad_weight)
                        self.weight._grad = self.weight._backend.add(self.weight._grad, grad_backend)
                
                # Gradient w.r.t. bias (β)
                if self.elementwise_affine and self.bias is not None and self.bias.requires_grad:
                    grad_bias = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))
                    
                    if self.bias._grad is None:
                        self.bias._grad = self.bias._backend.from_numpy(grad_bias)
                    else:
                        grad_backend = self.bias._backend.from_numpy(grad_bias)
                        self.bias._grad = self.bias._backend.add(self.bias._grad, grad_backend)
            
            # Collect input tensors for gradient function
            input_tensors = [x]
            if self.elementwise_affine:
                input_tensors.append(self.weight)
                if self.bias is not None:
                    input_tensors.append(self.bias)
            
            result._grad_fn = GradientFunction(backward_fn, input_tensors, "layernorm")
        
        logger.debug(f"LayerNorm operation: {x.shape} -> {result.shape}")
        return result
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (more stable than LayerNorm).
    
    RMSNorm is a simplified variant of LayerNorm that only normalizes by the
    root mean square, without centering by the mean. This is more stable and
    computationally efficient.
    
    Mathematical Definition:
        y = x / RMS(x) * γ
        where RMS(x) = sqrt(mean(x²) + ε)
        
    Reference: "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)
    """
    
    def __init__(
        self, 
        normalized_shape: int, 
        eps: float = 1e-8,
        elementwise_affine: bool = True
    ):
        """Initialize RMSNorm.
        
        Args:
            normalized_shape: Size of the feature dimension
            eps: Small value for numerical stability (smaller than LayerNorm)
            elementwise_affine: Whether to include learnable scale parameter
        """
        super().__init__()
        
        if normalized_shape <= 0:
            raise LayerError(f"normalized_shape must be positive, got {normalized_shape}")
        if eps <= 0:
            raise LayerError(f"eps must be positive, got {eps}")
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = Parameter(
                np.ones(normalized_shape, dtype=np.float32),
                name="rmsnorm.weight"
            )
        else:
            self.weight = None
    
    @memory_efficient_operation
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RMSNorm."""
        # Validate input shape
        if x.shape[-1] != self.normalized_shape:
            raise LayerError(
                f"Input last dimension {x.shape[-1]} != normalized_shape {self.normalized_shape}"
            )
        
        # Compute RMS
        mean_square = np.mean(x.data ** 2, axis=-1, keepdims=True)
        rms = np.sqrt(mean_square + self.eps)
        
        # Normalize
        normalized = x.data / rms
        
        # Apply scale if enabled
        if self.elementwise_affine:
            output_data = self.weight.data * normalized
        else:
            output_data = normalized
        
        # Create result tensor
        requires_grad = x.requires_grad or (
            self.elementwise_affine and self.weight.requires_grad
        )
        
        result = Tensor(
            output_data,
            requires_grad=requires_grad,
            name=f"rmsnorm({x.name or 'tensor'})"
        )
        
        # Set up gradient computation
        if requires_grad:
            def backward_fn(grad_output: np.ndarray) -> None:
                """Backward pass for RMSNorm."""
                N = x.shape[-1]
                
                # Gradient w.r.t. input
                if x.requires_grad:
                    if self.elementwise_affine:
                        grad_normalized = grad_output * self.weight.data
                    else:
                        grad_normalized = grad_output
                    
                    # RMSNorm gradient computation
                    rms_inv = 1.0 / rms
                    grad_rms = np.sum(grad_normalized * normalized * (-rms_inv), axis=-1, keepdims=True)
                    grad_mean_square = grad_rms * 0.5 * (mean_square + self.eps) ** (-0.5)
                    
                    grad_x = (grad_normalized * rms_inv) + (grad_mean_square * 2.0 * x.data / N)
                    
                    # Accumulate gradients
                    if x._grad is None:
                        x._grad = x._backend.from_numpy(grad_x)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        x._grad = x._backend.to_device(x._grad, device_str)
                    else:
                        grad_backend = x._backend.from_numpy(grad_x)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        grad_backend = x._backend.to_device(grad_backend, device_str)
                        x._grad = x._backend.add(x._grad, grad_backend)
                    
                    if x._grad_fn is not None:
                        x._grad_fn.apply(grad_x)
                
                # Gradient w.r.t. weight
                if self.elementwise_affine and self.weight.requires_grad:
                    grad_weight = np.sum(grad_output * normalized, axis=tuple(range(grad_output.ndim - 1)))
                    
                    if self.weight._grad is None:
                        self.weight._grad = self.weight._backend.from_numpy(grad_weight)
                    else:
                        grad_backend = self.weight._backend.from_numpy(grad_weight)
                        self.weight._grad = self.weight._backend.add(self.weight._grad, grad_backend)
            
            input_tensors = [x]
            if self.elementwise_affine:
                input_tensors.append(self.weight)
            
            result._grad_fn = GradientFunction(backward_fn, input_tensors, "rmsnorm")
        
        logger.debug(f"RMSNorm operation: {x.shape} -> {result.shape}")
        return result


class BatchNorm1d(Module):
    """1D Batch Normalization with mathematically correct implementation.
    
    Implements batch normalization as described in:
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    (https://arxiv.org/abs/1502.03167)
    
    Mathematical Definition:
        Training: y = γ * (x - μ_batch) / σ_batch + β
        Inference: y = γ * (x - μ_running) / σ_running + β
        
        where μ_batch, σ_batch are computed from current batch
        and μ_running, σ_running are exponential moving averages
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        """Initialize BatchNorm1d.
        
        Args:
            num_features: Number of features (C in (N, C) or (N, C, L))
            eps: Small value for numerical stability
            momentum: Momentum for running statistics (exponential moving average)
            affine: Whether to include learnable affine parameters
            track_running_stats: Whether to track running statistics
        """
        super().__init__()
        
        if num_features <= 0:
            raise LayerError(f"num_features must be positive, got {num_features}")
        if eps < 0:
            raise LayerError(f"eps must be non-negative, got {eps}")
        if not 0 <= momentum <= 1:
            raise LayerError(f"momentum must be in [0, 1], got {momentum}")
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Learnable parameters (γ and β)
        if affine:
            self.weight = Parameter(
                np.ones(num_features, dtype=np.float32),
                name="batchnorm1d.weight"
            )
            self.bias = Parameter(
                np.zeros(num_features, dtype=np.float32),
                name="batchnorm1d.bias"
            )
        else:
            self.weight = None
            self.bias = None
        
        # Running statistics (not parameters, just buffers)
        if track_running_stats:
            self.running_mean = np.zeros(num_features, dtype=np.float32)
            self.running_var = np.ones(num_features, dtype=np.float32)
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
        
        # The training mode is inherited from Module base class (self._training)
    
    def train(self, mode: bool = True) -> 'BatchNorm1d':
        """Set training mode."""
        super().train(mode)
        return self
    
    def eval(self) -> 'BatchNorm1d':
        """Set evaluation mode."""
        super().eval()
        return self
    
    @memory_efficient_operation
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with proper batch normalization.
        
        Args:
            x: Input tensor of shape (N, C) or (N, C, L)
            
        Returns:
            Normalized tensor with same shape as input
        """
        # Validate input
        if x.ndim < 2:
            raise LayerError(f"BatchNorm1d expects 2D or 3D input, got {x.ndim}D")
        if x.shape[1] != self.num_features:
            raise LayerError(
                f"Input channel dimension {x.shape[1]} != num_features {self.num_features}"
            )
        
        # Determine normalization axes
        if x.ndim == 2:  # (N, C)
            norm_axes = (0,)
            stats_shape = (1, self.num_features)
        elif x.ndim == 3:  # (N, C, L)
            norm_axes = (0, 2)
            stats_shape = (1, self.num_features, 1)
        else:
            raise LayerError(f"BatchNorm1d supports 2D and 3D input, got {x.ndim}D")
        
        if self.training:
            # Training mode: use batch statistics
            batch_mean = np.mean(x.data, axis=norm_axes, keepdims=True)
            batch_var = np.var(x.data, axis=norm_axes, keepdims=True, ddof=0)
            
            # Update running statistics
            if self.track_running_stats:
                # Exponential moving average
                if self.num_batches_tracked == 0:
                    # First batch: initialize with batch stats
                    self.running_mean = batch_mean.reshape(self.num_features).copy()
                    self.running_var = batch_var.reshape(self.num_features).copy()
                else:
                    # Update with momentum
                    batch_mean_1d = batch_mean.reshape(self.num_features)
                    batch_var_1d = batch_var.reshape(self.num_features)
                    
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean_1d
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var_1d
                
                self.num_batches_tracked += 1
            
            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Evaluation mode: use running statistics
            if not self.track_running_stats:
                raise RuntimeError("Cannot use eval mode without track_running_stats=True")
            
            mean = self.running_mean.reshape(stats_shape)
            var = self.running_var.reshape(stats_shape)
        
        # Normalize
        std = np.sqrt(var + self.eps)
        normalized = (x.data - mean) / std
        
        # Apply affine transformation
        if self.affine:
            weight_shaped = self.weight.data.reshape(stats_shape)
            bias_shaped = self.bias.data.reshape(stats_shape)
            output_data = weight_shaped * normalized + bias_shaped
        else:
            output_data = normalized
        
        # Create result tensor
        requires_grad = x.requires_grad or (
            self.affine and (self.weight.requires_grad or self.bias.requires_grad)
        )
        
        result = Tensor(
            output_data,
            requires_grad=requires_grad,
            name=f"batchnorm1d({x.name or 'tensor'})"
        )
        
        # Set up gradient computation
        if requires_grad and self.training:  # Only compute gradients in training mode
            def backward_fn(grad_output: np.ndarray) -> None:
                """Mathematically correct backward pass for BatchNorm."""
                N = np.prod([x.shape[i] for i in norm_axes])  # Number of elements averaged over
                
                # Gradient w.r.t. input
                if x.requires_grad:
                    if self.affine:
                        weight_shaped = self.weight.data.reshape(stats_shape)
                        grad_normalized = grad_output * weight_shaped
                    else:
                        grad_normalized = grad_output
                    
                    # BatchNorm gradient computation (same as LayerNorm but different axes)
                    grad_var = np.sum(
                        grad_normalized * (x.data - mean) * (-0.5) * (var + self.eps) ** (-1.5),
                        axis=norm_axes, keepdims=True
                    )
                    grad_mean = (
                        np.sum(grad_normalized * (-1.0 / std), axis=norm_axes, keepdims=True) +
                        grad_var * np.sum(-2.0 * (x.data - mean), axis=norm_axes, keepdims=True) / N
                    )
                    
                    grad_x = (grad_normalized / std) + (grad_var * 2.0 * (x.data - mean) / N) + (grad_mean / N)
                    
                    # Accumulate gradients for input
                    if x._grad is None:
                        x._grad = x._backend.from_numpy(grad_x)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        x._grad = x._backend.to_device(x._grad, device_str)
                    else:
                        grad_backend = x._backend.from_numpy(grad_x)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        grad_backend = x._backend.to_device(grad_backend, device_str)
                        x._grad = x._backend.add(x._grad, grad_backend)
                    
                    if x._grad_fn is not None:
                        x._grad_fn.apply(grad_x)
                
                # Gradient w.r.t. weight (γ)
                if self.affine and self.weight.requires_grad:
                    grad_weight = np.sum(grad_output * normalized, axis=tuple(i for i in range(grad_output.ndim) if i != 1))
                    
                    if self.weight._grad is None:
                        self.weight._grad = self.weight._backend.from_numpy(grad_weight)
                    else:
                        grad_backend = self.weight._backend.from_numpy(grad_weight)
                        self.weight._grad = self.weight._backend.add(self.weight._grad, grad_backend)
                
                # Gradient w.r.t. bias (β)
                if self.affine and self.bias.requires_grad:
                    grad_bias = np.sum(grad_output, axis=tuple(i for i in range(grad_output.ndim) if i != 1))
                    
                    if self.bias._grad is None:
                        self.bias._grad = self.bias._backend.from_numpy(grad_bias)
                    else:
                        grad_backend = self.bias._backend.from_numpy(grad_bias)
                        self.bias._grad = self.bias._backend.add(self.bias._grad, grad_backend)
            
            # Collect input tensors for gradient function
            input_tensors = [x]
            if self.affine:
                input_tensors.append(self.weight)
                input_tensors.append(self.bias)
            
            result._grad_fn = GradientFunction(backward_fn, input_tensors, "batchnorm1d")
        
        logger.debug(f"BatchNorm1d operation ({'train' if self.training else 'eval'}): {x.shape} -> {result.shape}")
        return result
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return (f"num_features={self.num_features}, eps={self.eps}, "
                f"momentum={self.momentum}, affine={self.affine}, "
                f"track_running_stats={self.track_running_stats}")


class BatchNorm2d(BatchNorm1d):
    """2D Batch Normalization for convolutional layers.
    
    Applies batch normalization over 4D input (N, C, H, W).
    Same mathematics as BatchNorm1d but different tensor dimensions.
    """
    
    @memory_efficient_operation
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for 2D batch normalization."""
        # Validate input
        if x.ndim != 4:
            raise LayerError(f"BatchNorm2d expects 4D input (N, C, H, W), got {x.ndim}D")
        if x.shape[1] != self.num_features:
            raise LayerError(
                f"Input channel dimension {x.shape[1]} != num_features {self.num_features}"
            )
        
        # Normalization over (N, H, W) dimensions, keeping C
        norm_axes = (0, 2, 3)
        stats_shape = (1, self.num_features, 1, 1)
        
        if self.training:
            # Training mode: use batch statistics
            batch_mean = np.mean(x.data, axis=norm_axes, keepdims=True)
            batch_var = np.var(x.data, axis=norm_axes, keepdims=True, ddof=0)
            
            # Update running statistics
            if self.track_running_stats:
                if self.num_batches_tracked == 0:
                    self.running_mean = batch_mean.reshape(self.num_features).copy()
                    self.running_var = batch_var.reshape(self.num_features).copy()
                else:
                    batch_mean_1d = batch_mean.reshape(self.num_features)
                    batch_var_1d = batch_var.reshape(self.num_features)
                    
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean_1d
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var_1d
                
                self.num_batches_tracked += 1
            
            mean = batch_mean
            var = batch_var
        else:
            # Evaluation mode: use running statistics
            if not self.track_running_stats:
                raise RuntimeError("Cannot use eval mode without track_running_stats=True")
            
            mean = self.running_mean.reshape(stats_shape)
            var = self.running_var.reshape(stats_shape)
        
        # Normalize
        std = np.sqrt(var + self.eps)
        normalized = (x.data - mean) / std
        
        # Apply affine transformation
        if self.affine:
            weight_shaped = self.weight.data.reshape(stats_shape)
            bias_shaped = self.bias.data.reshape(stats_shape)
            output_data = weight_shaped * normalized + bias_shaped
        else:
            output_data = normalized
        
        # Create result tensor
        requires_grad = x.requires_grad or (
            self.affine and (self.weight.requires_grad or self.bias.requires_grad)
        )
        
        result = Tensor(
            output_data,
            requires_grad=requires_grad,
            name=f"batchnorm2d({x.name or 'tensor'})"
        )
        
        # Same gradient computation as BatchNorm1d but with different axes
        if requires_grad and self.training:
            def backward_fn(grad_output: np.ndarray) -> None:
                """Backward pass for BatchNorm2d."""
                N = np.prod([x.shape[i] for i in norm_axes])
                
                if x.requires_grad:
                    if self.affine:
                        weight_shaped = self.weight.data.reshape(stats_shape)
                        grad_normalized = grad_output * weight_shaped
                    else:
                        grad_normalized = grad_output
                    
                    grad_var = np.sum(
                        grad_normalized * (x.data - mean) * (-0.5) * (var + self.eps) ** (-1.5),
                        axis=norm_axes, keepdims=True
                    )
                    grad_mean = (
                        np.sum(grad_normalized * (-1.0 / std), axis=norm_axes, keepdims=True) +
                        grad_var * np.sum(-2.0 * (x.data - mean), axis=norm_axes, keepdims=True) / N
                    )
                    
                    grad_x = (grad_normalized / std) + (grad_var * 2.0 * (x.data - mean) / N) + (grad_mean / N)
                    
                    if x._grad is None:
                        x._grad = x._backend.from_numpy(grad_x)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        x._grad = x._backend.to_device(x._grad, device_str)
                    else:
                        grad_backend = x._backend.from_numpy(grad_x)
                        device_str = x._device.type.value
                        if x._device.index is not None:
                            device_str = f"{device_str}:{x._device.index}"
                        grad_backend = x._backend.to_device(grad_backend, device_str)
                        x._grad = x._backend.add(x._grad, grad_backend)
                    
                    if x._grad_fn is not None:
                        x._grad_fn.apply(grad_x)
                
                # Gradients w.r.t parameters (sum over N, H, W)
                if self.affine and self.weight.requires_grad:
                    grad_weight = np.sum(grad_output * normalized, axis=(0, 2, 3))
                    
                    if self.weight._grad is None:
                        self.weight._grad = self.weight._backend.from_numpy(grad_weight)
                    else:
                        grad_backend = self.weight._backend.from_numpy(grad_weight)
                        self.weight._grad = self.weight._backend.add(self.weight._grad, grad_backend)
                
                if self.affine and self.bias.requires_grad:
                    grad_bias = np.sum(grad_output, axis=(0, 2, 3))
                    
                    if self.bias._grad is None:
                        self.bias._grad = self.bias._backend.from_numpy(grad_bias)
                    else:
                        grad_backend = self.bias._backend.from_numpy(grad_bias)
                        self.bias._grad = self.bias._backend.add(self.bias._grad, grad_backend)
            
            input_tensors = [x]
            if self.affine:
                input_tensors.append(self.weight)
                input_tensors.append(self.bias)
            
            result._grad_fn = GradientFunction(backward_fn, input_tensors, "batchnorm2d")
        
        logger.debug(f"BatchNorm2d operation ({'train' if self.training else 'eval'}): {x.shape} -> {result.shape}")
        return result


class GroupNorm(Module):
    """Group Normalization layer.
    
    From "Group Normalization" (https://arxiv.org/abs/1803.08494)
    
    GroupNorm divides channels into groups and normalizes within each group.
    This is more stable than BatchNorm for small batch sizes and works well
    for computer vision tasks.
    
    Mathematical Definition:
        For each group g: y_g = γ * (x_g - μ_g) / σ_g + β
        where μ_g = mean(x_g), σ_g = sqrt(var(x_g) + ε)
    """
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        """Initialize GroupNorm.
        
        Args:
            num_groups: Number of groups to divide channels into
            num_channels: Number of input channels (C)
            eps: Small value for numerical stability
            affine: Whether to include learnable affine parameters
        """
        super().__init__()
        
        if num_channels <= 0:
            raise LayerError(f"num_channels must be positive, got {num_channels}")
        if num_groups <= 0:
            raise LayerError(f"num_groups must be positive, got {num_groups}")
        if num_channels % num_groups != 0:
            raise LayerError(f"num_channels {num_channels} must be divisible by num_groups {num_groups}")
        if eps <= 0:
            raise LayerError(f"eps must be positive, got {eps}")
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = Parameter(
                np.ones(num_channels, dtype=np.float32),
                name="groupnorm.weight"
            )
            self.bias = Parameter(
                np.zeros(num_channels, dtype=np.float32),
                name="groupnorm.bias"
            )
        else:
            self.weight = None
            self.bias = None
    
    @memory_efficient_operation
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for GroupNorm."""
        # Validate input
        if x.ndim < 3:
            raise LayerError(f"GroupNorm expects at least 3D input (N, C, ...), got {x.ndim}D")
        if x.shape[1] != self.num_channels:
            raise LayerError(
                f"Input channel dimension {x.shape[1]} != num_channels {self.num_channels}"
            )
        
        N, C = x.shape[:2]
        spatial_dims = x.shape[2:]
        
        # Reshape to (N, G, C//G, *spatial_dims) for group-wise normalization
        channels_per_group = C // self.num_groups
        x_reshaped = x.data.reshape(N, self.num_groups, channels_per_group, *spatial_dims)
        
        # Compute statistics within each group
        # Normalize over (channels_per_group, *spatial_dims) for each group
        norm_axes = tuple(range(2, len(x_reshaped.shape)))  # Skip N and G dimensions
        
        mean = np.mean(x_reshaped, axis=norm_axes, keepdims=True)  # (N, G, 1, ...)
        var = np.var(x_reshaped, axis=norm_axes, keepdims=True, ddof=0)  # (N, G, 1, ...)
        
        # Normalize
        std = np.sqrt(var + self.eps)
        normalized = (x_reshaped - mean) / std
        
        # Reshape back to original shape
        normalized = normalized.reshape(N, C, *spatial_dims)
        
        # Apply affine transformation if enabled
        if self.affine:
            # Reshape weight and bias for broadcasting
            weight_shape = [1, C] + [1] * len(spatial_dims)
            bias_shape = [1, C] + [1] * len(spatial_dims)
            
            weight_reshaped = self.weight.data.reshape(weight_shape)
            bias_reshaped = self.bias.data.reshape(bias_shape)
            
            output_data = normalized * weight_reshaped + bias_reshaped
        else:
            output_data = normalized
        
        # Create result tensor
        requires_grad = x.requires_grad or (
            self.affine and (self.weight.requires_grad or self.bias.requires_grad)
        )
        
        result = Tensor(
            output_data,
            requires_grad=requires_grad,
            name=f"groupnorm({x.name or 'tensor'})"
        )
        
        # Set up gradient computation
        if requires_grad:
            def backward_fn(grad_output: np.ndarray) -> None:
                """Backward pass for GroupNorm."""
                # Reshape gradients for group-wise computation
                grad_reshaped = grad_output.reshape(N, self.num_groups, channels_per_group, *spatial_dims)
                normalized_reshaped = normalized.reshape(N, self.num_groups, channels_per_group, *spatial_dims)
                
                # Compute gradient w.r.t. normalized values
                if self.affine:
                    grad_normalized = grad_output * weight_reshaped
                    grad_normalized_reshaped = grad_normalized.reshape(N, self.num_groups, channels_per_group, *spatial_dims)
                else:
                    grad_normalized_reshaped = grad_reshaped
                
                # Compute gradients using group normalization formulas
                # Number of elements per group
                group_size = channels_per_group * np.prod(spatial_dims)
                
                # Gradient w.r.t. input
                grad_var = np.sum(grad_normalized_reshaped * (x_reshaped - mean), axis=norm_axes, keepdims=True)
                grad_var = grad_var * (-0.5) * (var + self.eps) ** (-1.5)
                
                grad_mean = np.sum(grad_normalized_reshaped, axis=norm_axes, keepdims=True) / (-std)
                grad_mean = grad_mean + grad_var * np.sum(-2.0 * (x_reshaped - mean), axis=norm_axes, keepdims=True) / group_size
                
                grad_x_reshaped = (grad_normalized_reshaped / std + 
                                 grad_var * 2.0 * (x_reshaped - mean) / group_size + 
                                 grad_mean / group_size)
                
                grad_x = grad_x_reshaped.reshape(x.shape)
                
                # Backward to input
                if x.requires_grad:
                    x.backward(grad_x)
                    if hasattr(x, '_backward'):
                        x._backward()
                
                # Gradients w.r.t. parameters
                if self.affine:
                    if self.weight.requires_grad:
                        grad_weight = np.sum(grad_output * normalized, axis=tuple([0] + list(range(2, len(grad_output.shape)))))
                        self.weight.backward(grad_weight)
                        if hasattr(self.weight, '_backward'):
                            self.weight._backward()
                    
                    if self.bias.requires_grad:
                        grad_bias = np.sum(grad_output, axis=tuple([0] + list(range(2, len(grad_output.shape)))))
                        self.bias.backward(grad_bias)
                        if hasattr(self.bias, '_backward'):
                            self.bias._backward()
            
            input_tensors = [x]
            if self.affine:
                input_tensors.append(self.weight)
                input_tensors.append(self.bias)
            
            result._grad_fn = GradientFunction(backward_fn, input_tensors, "groupnorm")
        
        logger.debug(f"GroupNorm operation: {x.shape} -> {result.shape}")
        return result
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"num_groups={self.num_groups}, num_channels={self.num_channels}, eps={self.eps}, affine={self.affine}"


class InstanceNorm(Module):
    """Instance Normalization layer.
    
    From "Instance Normalization: The Missing Ingredient for Fast Stylization" 
    (https://arxiv.org/abs/1607.08022)
    
    InstanceNorm normalizes each sample independently across spatial dimensions.
    It's commonly used in style transfer and generative models.
    
    Mathematical Definition:
        For each sample and channel: y_nc = γ * (x_nc - μ_nc) / σ_nc + β
        where μ_nc = mean(x_nc over spatial dims), σ_nc = sqrt(var(x_nc) + ε)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False
    ):
        """Initialize InstanceNorm.
        
        Args:
            num_features: Number of input channels (C)
            eps: Small value for numerical stability
            momentum: Momentum for running statistics (usually not used)
            affine: Whether to include learnable affine parameters
            track_running_stats: Whether to track running statistics (usually False)
        """
        super().__init__()
        
        if num_features <= 0:
            raise LayerError(f"num_features must be positive, got {num_features}")
        if eps <= 0:
            raise LayerError(f"eps must be positive, got {eps}")
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = Parameter(
                np.ones(num_features, dtype=np.float32),
                name="instancenorm.weight"
            )
            self.bias = Parameter(
                np.zeros(num_features, dtype=np.float32),
                name="instancenorm.bias"
            )
        else:
            self.weight = None
            self.bias = None
        
        # Running statistics (rarely used in InstanceNorm)
        if track_running_stats:
            self.running_mean = np.zeros(num_features, dtype=np.float32)
            self.running_var = np.ones(num_features, dtype=np.float32)
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
    
    @memory_efficient_operation
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for InstanceNorm."""
        # Validate input
        if x.ndim < 3:
            raise LayerError(f"InstanceNorm expects at least 3D input (N, C, ...), got {x.ndim}D")
        if x.shape[1] != self.num_features:
            raise LayerError(
                f"Input channel dimension {x.shape[1]} != num_features {self.num_features}"
            )
        
        N, C = x.shape[:2]
        spatial_dims = x.shape[2:]
        
        # Compute statistics for each sample and channel independently
        # Normalize over spatial dimensions only
        if len(spatial_dims) == 0:
            raise LayerError("InstanceNorm requires spatial dimensions")
        
        spatial_axes = tuple(range(2, x.ndim))  # Skip N and C dimensions
        
        # Compute mean and variance for each (N, C) pair
        mean = np.mean(x.data, axis=spatial_axes, keepdims=True)  # (N, C, 1, ...)
        var = np.var(x.data, axis=spatial_axes, keepdims=True, ddof=0)  # (N, C, 1, ...)
        
        # Normalize
        std = np.sqrt(var + self.eps)
        normalized = (x.data - mean) / std
        
        # Apply affine transformation if enabled
        if self.affine:
            # Reshape weight and bias for broadcasting
            weight_shape = [1, C] + [1] * len(spatial_dims)
            bias_shape = [1, C] + [1] * len(spatial_dims)
            
            weight_reshaped = self.weight.data.reshape(weight_shape)
            bias_reshaped = self.bias.data.reshape(bias_shape)
            
            output_data = normalized * weight_reshaped + bias_reshaped
        else:
            output_data = normalized
        
        # Update running statistics if tracking (rarely used)
        if self.track_running_stats and self.training:
            batch_mean = np.mean(mean.reshape(N, C), axis=0)  # Average over batch
            batch_var = np.mean(var.reshape(N, C), axis=0)    # Average over batch
            
            if self.num_batches_tracked == 0:
                self.running_mean = batch_mean.copy()
                self.running_var = batch_var.copy()
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            self.num_batches_tracked += 1
        
        # Create result tensor
        requires_grad = x.requires_grad or (
            self.affine and (self.weight.requires_grad or self.bias.requires_grad)
        )
        
        result = Tensor(
            output_data,
            requires_grad=requires_grad,
            name=f"instancenorm({x.name or 'tensor'})"
        )
        
        # Set up gradient computation
        if requires_grad:
            def backward_fn(grad_output: np.ndarray) -> None:
                """Backward pass for InstanceNorm."""
                # Compute gradient w.r.t. normalized values
                if self.affine:
                    weight_shape = [1, C] + [1] * len(spatial_dims)
                    weight_reshaped = self.weight.data.reshape(weight_shape)
                    grad_normalized = grad_output * weight_reshaped
                else:
                    grad_normalized = grad_output
                
                # Number of spatial elements per (N, C) pair
                spatial_size = np.prod(spatial_dims)
                
                # Compute gradients using instance normalization formulas
                grad_var = np.sum(grad_normalized * (x.data - mean), axis=spatial_axes, keepdims=True)
                grad_var = grad_var * (-0.5) * (var + self.eps) ** (-1.5)
                
                grad_mean = np.sum(grad_normalized, axis=spatial_axes, keepdims=True) / (-std)
                grad_mean = grad_mean + grad_var * np.sum(-2.0 * (x.data - mean), axis=spatial_axes, keepdims=True) / spatial_size
                
                grad_x = (grad_normalized / std + 
                         grad_var * 2.0 * (x.data - mean) / spatial_size + 
                         grad_mean / spatial_size)
                
                # Backward to input
                if x.requires_grad:
                    x.backward(grad_x)
                    if hasattr(x, '_backward'):
                        x._backward()
                
                # Gradients w.r.t. parameters
                if self.affine:
                    if self.weight.requires_grad:
                        grad_weight = np.sum(grad_output * normalized, axis=tuple([0] + list(spatial_axes)))
                        self.weight.backward(grad_weight)
                        if hasattr(self.weight, '_backward'):
                            self.weight._backward()
                    
                    if self.bias.requires_grad:
                        grad_bias = np.sum(grad_output, axis=tuple([0] + list(spatial_axes)))
                        self.bias.backward(grad_bias)
                        if hasattr(self.bias, '_backward'):
                            self.bias._backward()
            
            input_tensors = [x]
            if self.affine:
                input_tensors.append(self.weight)
                input_tensors.append(self.bias)
            
            result._grad_fn = GradientFunction(backward_fn, input_tensors, "instancenorm")
        
        logger.debug(f"InstanceNorm operation: {x.shape} -> {result.shape}")
        return result
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"num_features={self.num_features}, eps={self.eps}, affine={self.affine}"