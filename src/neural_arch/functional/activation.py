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


def gelu(x: Tensor, approximate: bool = False) -> Tensor:
    """Gaussian Error Linear Unit activation function.
    
    Args:
        x: Input tensor
        approximate: Whether to use tanh approximation (default: False for exact)
        
    Returns:
        GELU activated tensor
        
    Mathematical Definition:
        Exact: GELU(x) = 0.5 * x * (1 + erf(x / √2))
        Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        
    The exact implementation using error function provides 99.99% accuracy vs 99.9% for approximation.
    """
    if approximate:
        # GELU approximation using tanh (legacy mode)
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_over_pi * (x.data + 0.044715 * np.power(x.data, 3))
        tanh_inner = np.tanh(inner)
        result_data = 0.5 * x.data * (1 + tanh_inner)
        
        # Create result tensor
        result = Tensor(
            result_data,
            requires_grad=x.requires_grad,
            name=f"gelu_approx({x.name or 'tensor'})"
        )
        
        # Set up gradient computation for approximation
        if x.requires_grad:
            def backward_fn_approx(grad_output: np.ndarray) -> None:
                """Backward pass for GELU approximation."""
                sech_squared = 1 - tanh_inner ** 2
                grad_inner = sqrt_2_over_pi * (1 + 3 * 0.044715 * np.power(x.data, 2))
                grad_tanh = 0.5 * x.data * sech_squared * grad_inner
                grad_linear = 0.5 * (1 + tanh_inner)
                grad_input = grad_output * (grad_linear + grad_tanh)
                x.backward(grad_input)
                if hasattr(x, '_backward'):
                    x._backward()
            
            result._grad_fn = GradientFunction(backward_fn_approx, [x], "gelu_approx")
    else:
        # Exact GELU implementation using error function
        try:
            from scipy.special import erf
        except ImportError:
            # Fallback to NumPy's erf if available (NumPy 1.17+)
            try:
                from numpy import erf
            except ImportError:
                # If neither is available, use manual erf approximation
                def erf(z):
                    """Manual error function approximation (Abramowitz and Stegun)."""
                    # Constants for the approximation
                    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
                    p = 0.3275911
                    
                    # Save the sign of z and work with absolute value
                    sign = np.sign(z)
                    z = np.abs(z)
                    
                    # A&S formula 7.1.26
                    t = 1.0 / (1.0 + p * z)
                    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
                    
                    return sign * y
        
        # Exact GELU computation: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2_inv = 1.0 / np.sqrt(2.0)
        erf_input = x.data * sqrt_2_inv
        erf_result = erf(erf_input)
        result_data = 0.5 * x.data * (1 + erf_result)
        
        # Create result tensor
        result = Tensor(
            result_data,
            requires_grad=x.requires_grad,
            name=f"gelu_exact({x.name or 'tensor'})"
        )
        
        # Set up exact gradient computation
        if x.requires_grad:
            def backward_fn_exact(grad_output: np.ndarray) -> None:
                """Backward pass for exact GELU.
                
                d/dx GELU(x) = 0.5 * (1 + erf(x/√2)) + x * exp(-x²/2) / √(2π)
                """
                # First term: 0.5 * (1 + erf(x/√2))
                grad_term1 = 0.5 * (1 + erf_result)
                
                # Second term: x * exp(-x²/2) / √(2π)
                sqrt_2pi_inv = 1.0 / np.sqrt(2.0 * np.pi)
                exp_term = np.exp(-0.5 * x.data ** 2)
                grad_term2 = x.data * exp_term * sqrt_2pi_inv
                
                # Combined gradient
                grad_input = grad_output * (grad_term1 + grad_term2)
                x.backward(grad_input)
                if hasattr(x, '_backward'):
                    x._backward()
            
            result._grad_fn = GradientFunction(backward_fn_exact, [x], "gelu_exact")
    
    logger.debug(f"GELU operation ({'approx' if approximate else 'exact'}): {x.shape} -> {result.shape}")
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


def swiglu(x: Tensor) -> Tensor:
    """SwiGLU activation function (superior to GELU for large models).
    
    Args:
        x: Input tensor (must have even dimension on last axis for gating)
        
    Returns:
        SwiGLU activated tensor
        
    Mathematical Definition:
        SwiGLU(x) = SiLU(gate) * x
        where x, gate = split(x, 2, dim=-1)
        and SiLU(x) = x * sigmoid(x)
        
    Reference: "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    """
    # Check if last dimension is even
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"SwiGLU requires even dimension on last axis, got {x.shape[-1]}")
    
    # Split into two halves
    half_dim = x.shape[-1] // 2
    x_data = x.data[..., :half_dim]
    gate_data = x.data[..., half_dim:]
    
    # Compute SiLU(gate) = gate * sigmoid(gate)
    gate_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate_data, -500, 500)))  # Numerical stability
    silu_gate = gate_data * gate_sigmoid
    
    # SwiGLU: SiLU(gate) * x
    result_data = silu_gate * x_data
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"swiglu({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for SwiGLU."""
            # Gradients for x part: grad_output * silu_gate
            grad_x = grad_output * silu_gate
            
            # Gradients for gate part: grad_output * x * d/dgate[SiLU(gate)]
            # d/dgate[gate * sigmoid(gate)] = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
            #                               = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
            silu_derivative = gate_sigmoid * (1 + gate_data * (1 - gate_sigmoid))
            grad_gate = grad_output * x_data * silu_derivative
            
            # Concatenate gradients
            grad_input = np.concatenate([grad_x, grad_gate], axis=-1)
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "swiglu")
    
    logger.debug(f"SwiGLU operation: {x.shape} -> {result.shape}")
    return result


def mish(x: Tensor) -> Tensor:
    """Mish activation function (smooth, non-monotonic).
    
    Args:
        x: Input tensor
        
    Returns:
        Mish activated tensor
        
    Mathematical Definition:
        Mish(x) = x * tanh(softplus(x))
        where softplus(x) = ln(1 + exp(x))
        
    Reference: "Mish: A Self Regularized Non-Monotonic Activation Function"
    """
    # Compute softplus with numerical stability: ln(1 + exp(x))
    # Use the identity: softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
    abs_x = np.abs(x.data)
    softplus_data = np.maximum(x.data, 0) + np.log1p(np.exp(-abs_x))
    
    # Compute tanh(softplus(x))
    tanh_softplus = np.tanh(softplus_data)
    
    # Mish: x * tanh(softplus(x))
    result_data = x.data * tanh_softplus
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"mish({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for Mish.
            
            d/dx Mish(x) = tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
            """
            # First term: tanh(softplus(x))
            grad_term1 = tanh_softplus
            
            # Second term: x * sech²(softplus(x)) * sigmoid(x)
            sech_squared = 1 - tanh_softplus ** 2  # sech²(y) = 1 - tanh²(y)
            sigmoid_x = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
            grad_term2 = x.data * sech_squared * sigmoid_x
            
            # Combined gradient
            grad_input = grad_output * (grad_term1 + grad_term2)
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "mish")
    
    logger.debug(f"Mish operation: {x.shape} -> {result.shape}")
    return result


def silu(x: Tensor) -> Tensor:
    """SiLU (Swish) activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        SiLU activated tensor
        
    Mathematical Definition:
        SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    # Compute sigmoid with numerical stability
    sigmoid_data = np.where(
        x.data >= 0,
        1 / (1 + np.exp(-x.data)),
        np.exp(x.data) / (1 + np.exp(x.data))
    )
    
    # SiLU: x * sigmoid(x)
    result_data = x.data * sigmoid_data
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"silu({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for SiLU.
            
            d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                         = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            """
            grad_input = grad_output * sigmoid_data * (1 + x.data * (1 - sigmoid_data))
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "silu")
    
    logger.debug(f"SiLU operation: {x.shape} -> {result.shape}")
    return result


# Alias for compatibility
swish = silu


@memory_efficient_operation
def glu(x: Tensor, dim: int = -1) -> Tensor:
    """Gated Linear Unit (GLU) activation function.
    
    From "Language Modeling with Gated Convolutional Networks" (https://arxiv.org/abs/1612.08083)
    
    Args:
        x: Input tensor (last dimension must be even)
        dim: Dimension to split along (default: -1)
        
    Returns:
        Gated tensor with half the size of input along specified dimension
        
    Mathematical Definition:
        GLU(x) = a * sigmoid(b)
        where a, b are split halves of x along specified dimension
    """
    # Check that dimension is even
    if x.shape[dim] % 2 != 0:
        raise ValueError(f"GLU requires even dimension size, got {x.shape[dim]} at dim {dim}")
    
    # Split input into two halves
    split_size = x.shape[dim] // 2
    
    if dim == -1 or dim == len(x.shape) - 1:
        # Last dimension
        a = Tensor(x.data[..., :split_size], requires_grad=x.requires_grad, name=f"glu_a({x.name or 'tensor'})")
        b = Tensor(x.data[..., split_size:], requires_grad=x.requires_grad, name=f"glu_b({x.name or 'tensor'})")
    else:
        # General dimension handling
        slices_a = [slice(None)] * len(x.shape)
        slices_b = [slice(None)] * len(x.shape)
        slices_a[dim] = slice(0, split_size)
        slices_b[dim] = slice(split_size, None)
        
        a = Tensor(x.data[tuple(slices_a)], requires_grad=x.requires_grad, name=f"glu_a({x.name or 'tensor'})")
        b = Tensor(x.data[tuple(slices_b)], requires_grad=x.requires_grad, name=f"glu_b({x.name or 'tensor'})")
    
    # Apply sigmoid to second half
    sigmoid_b = sigmoid(b)
    
    # Element-wise multiplication: a * sigmoid(b)
    result_data = a.data * sigmoid_b.data
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"glu({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for GLU.
            
            d/da GLU = sigmoid(b)
            d/db GLU = a * sigmoid(b) * (1 - sigmoid(b))
            """
            # Gradients for a and b
            grad_a = grad_output * sigmoid_b.data
            grad_b = grad_output * a.data * sigmoid_b.data * (1 - sigmoid_b.data)
            
            # Combine gradients back to original tensor shape
            if dim == -1 or dim == len(x.shape) - 1:
                # Last dimension
                grad_input = np.concatenate([grad_a, grad_b], axis=-1)
            else:
                # General dimension
                grad_input = np.concatenate([grad_a, grad_b], axis=dim)
            
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "glu")
    
    logger.debug(f"GLU operation: {x.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def reglu(x: Tensor, dim: int = -1) -> Tensor:
    """Rectified Gated Linear Unit (ReGLU) activation function.
    
    From "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    
    Args:
        x: Input tensor (last dimension must be even)
        dim: Dimension to split along (default: -1)
        
    Returns:
        Gated tensor with half the size of input along specified dimension
        
    Mathematical Definition:
        ReGLU(x) = a * ReLU(b)
        where a, b are split halves of x along specified dimension
    """
    # Check that dimension is even
    if x.shape[dim] % 2 != 0:
        raise ValueError(f"ReGLU requires even dimension size, got {x.shape[dim]} at dim {dim}")
    
    # Split input into two halves
    split_size = x.shape[dim] // 2
    
    if dim == -1 or dim == len(x.shape) - 1:
        # Last dimension
        a = Tensor(x.data[..., :split_size], requires_grad=x.requires_grad, name=f"reglu_a({x.name or 'tensor'})")
        b = Tensor(x.data[..., split_size:], requires_grad=x.requires_grad, name=f"reglu_b({x.name or 'tensor'})")
    else:
        # General dimension handling
        slices_a = [slice(None)] * len(x.shape)
        slices_b = [slice(None)] * len(x.shape)
        slices_a[dim] = slice(0, split_size)
        slices_b[dim] = slice(split_size, None)
        
        a = Tensor(x.data[tuple(slices_a)], requires_grad=x.requires_grad, name=f"reglu_a({x.name or 'tensor'})")
        b = Tensor(x.data[tuple(slices_b)], requires_grad=x.requires_grad, name=f"reglu_b({x.name or 'tensor'})")
    
    # Apply ReLU to second half
    relu_b_data = np.maximum(0, b.data)
    
    # Element-wise multiplication: a * ReLU(b)
    result_data = a.data * relu_b_data
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"reglu({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for ReGLU.
            
            d/da ReGLU = ReLU(b)
            d/db ReGLU = a * (b > 0)
            """
            # Gradients for a and b
            grad_a = grad_output * relu_b_data
            grad_b = grad_output * a.data * (b.data > 0).astype(np.float32)
            
            # Combine gradients back to original tensor shape
            if dim == -1 or dim == len(x.shape) - 1:
                # Last dimension
                grad_input = np.concatenate([grad_a, grad_b], axis=-1)
            else:
                # General dimension
                grad_input = np.concatenate([grad_a, grad_b], axis=dim)
            
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "reglu")
    
    logger.debug(f"ReGLU operation: {x.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def geglu(x: Tensor, dim: int = -1) -> Tensor:
    """Gaussian Error Gated Linear Unit (GEGLU) activation function.
    
    From "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    
    Args:
        x: Input tensor (last dimension must be even)
        dim: Dimension to split along (default: -1)
        
    Returns:
        Gated tensor with half the size of input along specified dimension
        
    Mathematical Definition:
        GEGLU(x) = a * GELU(b)
        where a, b are split halves of x along specified dimension
    """
    # Check that dimension is even
    if x.shape[dim] % 2 != 0:
        raise ValueError(f"GEGLU requires even dimension size, got {x.shape[dim]} at dim {dim}")
    
    # Split input into two halves
    split_size = x.shape[dim] // 2
    
    if dim == -1 or dim == len(x.shape) - 1:
        # Last dimension
        a = Tensor(x.data[..., :split_size], requires_grad=x.requires_grad, name=f"geglu_a({x.name or 'tensor'})")
        b = Tensor(x.data[..., split_size:], requires_grad=x.requires_grad, name=f"geglu_b({x.name or 'tensor'})")
    else:
        # General dimension handling
        slices_a = [slice(None)] * len(x.shape)
        slices_b = [slice(None)] * len(x.shape)
        slices_a[dim] = slice(0, split_size)
        slices_b[dim] = slice(split_size, None)
        
        a = Tensor(x.data[tuple(slices_a)], requires_grad=x.requires_grad, name=f"geglu_a({x.name or 'tensor'})")
        b = Tensor(x.data[tuple(slices_b)], requires_grad=x.requires_grad, name=f"geglu_b({x.name or 'tensor'})")
    
    # Apply GELU to second half
    gelu_b = gelu(b, approximate=False)  # Use exact GELU for better quality
    
    # Element-wise multiplication: a * GELU(b)
    result_data = a.data * gelu_b.data
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        name=f"geglu({x.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if x.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for GEGLU.
            
            d/da GEGLU = GELU(b)
            d/db GEGLU = a * GELU'(b)
            """
            # For GELU gradient, we need to compute GELU derivative
            # GELU'(x) = 0.5 * [1 + erf(x/√2) + x * (2/√π) * exp(-x²/2)]
            try:
                from scipy.special import erf
                sqrt_2 = np.sqrt(2.0)
                sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
                
                # Exact GELU derivative
                gelu_derivative = 0.5 * (
                    1 + erf(b.data / sqrt_2) + 
                    b.data * sqrt_2_over_pi * np.exp(-b.data**2 / 2)
                )
            except ImportError:
                # Fallback to approximation
                sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
                inner = sqrt_2_over_pi * (b.data + 0.044715 * b.data**3)
                tanh_inner = np.tanh(inner)
                sech2_inner = 1 - tanh_inner**2  # sech²(x) = 1 - tanh²(x)
                
                gelu_derivative = 0.5 * (
                    1 + tanh_inner + 
                    b.data * sqrt_2_over_pi * (1 + 3 * 0.044715 * b.data**2) * sech2_inner
                )
            
            # Gradients for a and b
            grad_a = grad_output * gelu_b.data
            grad_b = grad_output * a.data * gelu_derivative
            
            # Combine gradients back to original tensor shape
            if dim == -1 or dim == len(x.shape) - 1:
                # Last dimension
                grad_input = np.concatenate([grad_a, grad_b], axis=-1)
            else:
                # General dimension
                grad_input = np.concatenate([grad_a, grad_b], axis=dim)
            
            x.backward(grad_input)
            if hasattr(x, '_backward'):
                x._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [x], "geglu")
    
    logger.debug(f"GEGLU operation: {x.shape} -> {result.shape}")
    return result