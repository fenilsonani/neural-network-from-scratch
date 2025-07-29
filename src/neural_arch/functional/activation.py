"""Activation functions with automatic differentiation."""

import numpy as np
import logging

from ..core.tensor import Tensor, GradientFunction
from .utils import memory_efficient_operation

logger = logging.getLogger(__name__)


@memory_efficient_operation
def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        ReLU activated tensor
        
    Mathematical Definition:
        f(x) = max(0, x)
        f'(x) = 1 if x > 0, else 0
    """
    # Apply ReLU: max(0, x)
    result_data = np.maximum(0, x.data)
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"relu({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for ReLU.
            
            Gradient is 1 where input > 0, 0 elsewhere.
            """
            grad_input = grad_output * (x.data > 0).astype(np.float32)
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "relu")
    
    logger.debug(f"ReLU operation: {x.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation function with numerical stability.
    
    Args:
        x: Input tensor
        axis: Axis along which to apply softmax
        
    Returns:
        Softmax activated tensor
        
    Mathematical Definition:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        
    The subtraction of max(x) ensures numerical stability by preventing
    overflow in the exponential function.
    """
    # Numerical stability: subtract max along the specified axis
    x_max = np.max(x.data, axis=axis, keepdims=True)
    x_shifted = x.data - x_max
    
    # Compute softmax
    exp_values = np.exp(x_shifted)
    sum_exp = np.sum(exp_values, axis=axis, keepdims=True)
    
    # Avoid division by zero
    sum_exp = np.maximum(sum_exp, 1e-8)
    result_data = exp_values / sum_exp
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"softmax({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for softmax.
            
            For softmax, the Jacobian is:
            ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
            
            This leads to: grad_input = softmax * (grad_output - sum(grad_output * softmax))
            """
            # Compute the sum along the softmax axis
            sum_term = np.sum(grad_output * result_data, axis=axis, keepdims=True)
            grad_input = result_data * (grad_output - sum_term)
            
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "softmax")
    
    logger.debug(f"Softmax operation: {x.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Sigmoid activated tensor
        
    Mathematical Definition:
        σ(x) = 1 / (1 + exp(-x))
        σ'(x) = σ(x) * (1 - σ(x))
    """
    # Numerical stability for sigmoid
    # Use different formulations for positive and negative values
    result_data = np.where(
        x.data >= 0,
        1 / (1 + np.exp(-x.data)),  # For x >= 0
        np.exp(x.data) / (1 + np.exp(x.data))  # For x < 0
    )
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"sigmoid({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for sigmoid.
            
            Gradient is sigmoid(x) * (1 - sigmoid(x))
            """
            grad_input = grad_output * result_data * (1 - result_data)
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "sigmoid")
    
    logger.debug(f"Sigmoid operation: {x.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tanh activated tensor
        
    Mathematical Definition:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        tanh'(x) = 1 - tanh²(x)
    """
    # Compute tanh using numpy's stable implementation
    result_data = np.tanh(x.data)
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"tanh({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for tanh.
            
            Gradient is 1 - tanh²(x)
            """
            grad_input = grad_output * (1 - result_data ** 2)
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "tanh")
    
    logger.debug(f"Tanh operation: {x.shape} -> {result.shape}")
    return result


def gelu(x: Tensor) -> Tensor:
    """Gaussian Error Linear Unit activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        GELU activated tensor
        
    Mathematical Definition:
        GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
        Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    # GELU approximation for computational efficiency
    # This is the approximation used in many transformer implementations
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    inner = sqrt_2_over_pi * (x.data + 0.044715 * np.power(x.data, 3))
    tanh_inner = np.tanh(inner)
    result_data = 0.5 * x.data * (1 + tanh_inner)
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"gelu({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for GELU.
            
            This uses the derivative of the GELU approximation.
            """
            # Derivative computation for GELU approximation
            sech_squared = 1 - tanh_inner ** 2  # sech²(inner) = 1 - tanh²(inner)
            
            grad_inner = sqrt_2_over_pi * (1 + 3 * 0.044715 * np.power(x.data, 2))
            grad_tanh = 0.5 * x.data * sech_squared * grad_inner
            grad_linear = 0.5 * (1 + tanh_inner)
            
            grad_input = grad_output * (grad_linear + grad_tanh)
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "gelu")
    
    logger.debug(f"GELU operation: {x.shape} -> {result.shape}")
    return result


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Leaky ReLU activation function.
    
    Args:
        x: Input tensor
        negative_slope: Slope for negative values
        
    Returns:
        Leaky ReLU activated tensor
        
    Mathematical Definition:
        f(x) = x if x > 0, else negative_slope * x
        f'(x) = 1 if x > 0, else negative_slope
    """
    # Apply Leaky ReLU
    result_data = np.where(x.data > 0, x.data, negative_slope * x.data)
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"leaky_relu({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for Leaky ReLU."""
            grad_input = grad_output * np.where(x.data > 0, 1.0, negative_slope)
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "leaky_relu")
    
    logger.debug(f"Leaky ReLU operation: {x.shape} -> {result.shape}")
    return result