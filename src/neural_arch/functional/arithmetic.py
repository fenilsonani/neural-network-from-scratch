"""Arithmetic operations with automatic differentiation."""

import numpy as np
from typing import Union
import logging

from ..core.tensor import Tensor, TensorLike, GradientFunction
from .utils import broadcast_tensors, reduce_gradient

logger = logging.getLogger(__name__)


def add(a: TensorLike, b: TensorLike) -> Tensor:
    """Element-wise addition with broadcasting and gradient support.
    
    Args:
        a: First operand
        b: Second operand
        
    Returns:
        Result tensor
        
    Mathematical Definition:
        output = a + b
        ∂output/∂a = 1 (broadcasted to a.shape)
        ∂output/∂b = 1 (broadcasted to b.shape)
    """
    # Convert inputs to tensors
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    # Ensure tensors are on the same device
    if a.device != b.device:
        raise ValueError(f"Tensors must be on same device, got {a.device} and {b.device}")
    
    # Use backend for computation
    backend = a.backend
    result_data = backend.add(a.backend_data, b.backend_data)
    
    # Create result tensor - convert to numpy to create new tensor
    result_np = backend.to_numpy(result_data)
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(
        result_np,
        requires_grad=requires_grad,
        device=a.device,
        dtype=a.dtype,
        name=f"add({a.name or 'tensor'}, {b.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for addition."""
            if a.requires_grad:
                # Reduce gradient to match original tensor shape
                grad_a = reduce_gradient(grad_output, a.shape, result.shape)
                a.backward(grad_a)
            
            if b.requires_grad:
                # Reduce gradient to match original tensor shape
                grad_b = reduce_gradient(grad_output, b.shape, result.shape)
                b.backward(grad_b)
        
        result._grad_fn = GradientFunction(backward_fn, [a, b], "add")
    
    logger.debug(f"Add operation: {a.shape} + {b.shape} -> {result.shape}")
    return result


def sub(a: TensorLike, b: TensorLike) -> Tensor:
    """Element-wise subtraction with broadcasting and gradient support.
    
    Args:
        a: First operand (minuend)
        b: Second operand (subtrahend)
        
    Returns:
        Result tensor
        
    Mathematical Definition:
        output = a - b
        ∂output/∂a = 1 (broadcasted to a.shape)
        ∂output/∂b = -1 (broadcasted to b.shape)
    """
    # Convert inputs to tensors
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    # Broadcast tensors to compatible shapes
    a_data, b_data = np.broadcast_arrays(a.data, b.data)
    result_data = a_data - b_data
    
    # Create result tensor
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(
        result_data,
        requires_grad=requires_grad,
        name=f"sub({a.name or 'tensor'}, {b.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for subtraction."""
            if a.requires_grad:
                grad_a = reduce_gradient(grad_output, a.shape, a_data.shape)
                a.backward(grad_a)
                if hasattr(a, '_backward'):
                    a._backward()
            
            if b.requires_grad:
                # Negative gradient for subtraction
                grad_b = reduce_gradient(-grad_output, b.shape, b_data.shape)
                b.backward(grad_b)
                if hasattr(b, '_backward'):
                    b._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [a, b], "sub")
    
    logger.debug(f"Sub operation: {a.shape} - {b.shape} -> {result.shape}")
    return result


def mul(a: TensorLike, b: TensorLike) -> Tensor:
    """Element-wise multiplication with broadcasting and gradient support.
    
    Args:
        a: First operand
        b: Second operand
        
    Returns:
        Result tensor
        
    Mathematical Definition:
        output = a * b
        ∂output/∂a = b (broadcasted to a.shape)
        ∂output/∂b = a (broadcasted to b.shape)
    """
    # Convert inputs to tensors
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    # Broadcast tensors to compatible shapes
    a_data, b_data = np.broadcast_arrays(a.data, b.data)
    result_data = a_data * b_data
    
    # Create result tensor
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(
        result_data,
        requires_grad=requires_grad,
        name=f"mul({a.name or 'tensor'}, {b.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for multiplication."""
            if a.requires_grad:
                # Gradient is grad_output * b
                grad_a = reduce_gradient(grad_output * b_data, a.shape, a_data.shape)
                a.backward(grad_a)
                if hasattr(a, '_backward'):
                    a._backward()
            
            if b.requires_grad:
                # Gradient is grad_output * a
                grad_b = reduce_gradient(grad_output * a_data, b.shape, b_data.shape)
                b.backward(grad_b)
                if hasattr(b, '_backward'):
                    b._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [a, b], "mul")
    
    logger.debug(f"Mul operation: {a.shape} * {b.shape} -> {result.shape}")
    return result


def div(a: TensorLike, b: TensorLike) -> Tensor:
    """Element-wise division with broadcasting and gradient support.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result tensor
        
    Mathematical Definition:
        output = a / b
        ∂output/∂a = 1/b (broadcasted to a.shape)
        ∂output/∂b = -a/b² (broadcasted to b.shape)
        
    Raises:
        ValueError: If denominator contains zeros
    """
    # Convert inputs to tensors
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    # Check for division by zero
    if np.any(b.data == 0):
        raise ValueError("Division by zero detected")
    
    # Broadcast tensors to compatible shapes
    a_data, b_data = np.broadcast_arrays(a.data, b.data)
    result_data = a_data / b_data
    
    # Create result tensor
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(
        result_data,
        requires_grad=requires_grad,
        name=f"div({a.name or 'tensor'}, {b.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for division."""
            if a.requires_grad:
                # Gradient is grad_output / b
                grad_a = reduce_gradient(grad_output / b_data, a.shape, a_data.shape)
                a.backward(grad_a)
                if hasattr(a, '_backward'):
                    a._backward()
            
            if b.requires_grad:
                # Gradient is -grad_output * a / b²
                grad_b = reduce_gradient(-grad_output * a_data / (b_data ** 2), b.shape, b_data.shape)
                b.backward(grad_b)
                if hasattr(b, '_backward'):
                    b._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [a, b], "div")
    
    logger.debug(f"Div operation: {a.shape} / {b.shape} -> {result.shape}")
    return result


def neg(a: Tensor) -> Tensor:
    """Element-wise negation with gradient support.
    
    Args:
        a: Input tensor
        
    Returns:
        Negated tensor
        
    Mathematical Definition:
        output = -a
        ∂output/∂a = -1
    """
    result_data = -a.data
    
    # Create result tensor
    result = Tensor(
        result_data,
        requires_grad=a.requires_grad,
        name=f"neg({a.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if a.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for negation."""
            a.backward(-grad_output)
            if hasattr(a, '_backward'):
                a._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [a], "neg")
    
    logger.debug(f"Neg operation: {a.shape} -> {result.shape}")
    return result


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication with gradient support.
    
    Args:
        a: Left matrix tensor
        b: Right matrix tensor
        
    Returns:
        Matrix multiplication result
        
    Mathematical Definition:
        output = a @ b
        ∂output/∂a = grad_output @ b.T
        ∂output/∂b = a.T @ grad_output
        
    Raises:
        ValueError: If matrix dimensions are incompatible
    """
    # Ensure tensors are on the same device
    if a.device != b.device:
        raise ValueError(f"Tensors must be on same device, got {a.device} and {b.device}")
    
    # Validate matrix dimensions
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(f"matmul requires 2D+ tensors, got shapes {a.shape} and {b.shape}")
    
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Incompatible matrix dimensions: {a.shape} @ {b.shape}")
    
    # Use backend for computation
    backend = a.backend
    result_data = backend.matmul(a.backend_data, b.backend_data)
    
    # Create result tensor - convert to numpy to create new tensor
    result_np = backend.to_numpy(result_data)
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(
        result_np,
        requires_grad=requires_grad,
        device=a.device,
        dtype=a.dtype,
        name=f"matmul({a.name or 'tensor'}, {b.name or 'tensor'})"
    )
    
    # Set up gradient computation
    if requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for matrix multiplication."""
            if a.requires_grad:
                # grad_a = grad_output @ b.T
                # Convert grad_output to backend array temporarily
                grad_backend = backend.from_numpy(grad_output)
                grad_backend = backend.to_device(grad_backend, a.device.type.value)
                
                # Compute gradient using backend
                # For matrix transpose, we need to specify the axes to swap
                if b.ndim == 2:
                    b_transposed = backend.transpose(b.backend_data, (1, 0))
                else:
                    # For higher dimensional tensors, swap last two axes
                    axes = list(range(b.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    b_transposed = backend.transpose(b.backend_data, tuple(axes))
                grad_a_backend = backend.matmul(grad_backend, b_transposed)
                grad_a = backend.to_numpy(grad_a_backend)
                
                # Reduce gradient if needed (e.g., batched operations on parameters)
                if grad_a.shape != a.shape:
                    grad_a = reduce_gradient(grad_a, a.shape, grad_a.shape)
                
                a.backward(grad_a)
            
            if b.requires_grad:
                # grad_b = a.T @ grad_output
                # Convert grad_output to backend array temporarily
                grad_backend = backend.from_numpy(grad_output)
                grad_backend = backend.to_device(grad_backend, b.device.type.value)
                
                # Compute gradient using backend
                # For matrix transpose, we need to specify the axes to swap
                if a.ndim == 2:
                    a_transposed = backend.transpose(a.backend_data, (1, 0))
                else:
                    # For higher dimensional tensors, swap last two axes
                    axes = list(range(a.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    a_transposed = backend.transpose(a.backend_data, tuple(axes))
                grad_b_backend = backend.matmul(a_transposed, grad_backend)
                grad_b = backend.to_numpy(grad_b_backend)
                
                # Reduce gradient if needed (e.g., batched operations on parameters)
                if grad_b.shape != b.shape:
                    grad_b = reduce_gradient(grad_b, b.shape, grad_b.shape)
                
                b.backward(grad_b)
        
        result._grad_fn = GradientFunction(backward_fn, [a, b], "matmul")
    
    logger.debug(f"Matmul operation: {a.shape} @ {b.shape} -> {result.shape}")
    return result