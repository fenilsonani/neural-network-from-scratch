"""Pooling operations with automatic differentiation."""

import logging

import numpy as np

from ..core.tensor import GradientFunction, Tensor
from .utils import memory_efficient_operation

logger = logging.getLogger(__name__)


@memory_efficient_operation
def mean_pool(x: Tensor, axis: int = 1) -> Tensor:
    """Mean pooling operation with gradient support.

    Args:
        x: Input tensor
        axis: Axis along which to pool

    Returns:
        Pooled tensor

    Mathematical Definition:
        output = mean(x, axis=axis)
        ∂output/∂x = 1/n where n is the size of pooled dimension
    """
    # Perform mean pooling
    result_data = np.mean(x.data, axis=axis, keepdims=False)

    # Create result tensor
    result = Tensor(
        result_data, requires_grad=x.requires_grad, name=f"mean_pool({x.name or 'tensor'})"
    )

    # Set up gradient computation
    if x.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for mean pooling."""
            # Expand grad_output to match input shape
            grad_input = np.expand_dims(grad_output, axis=axis)

            # Broadcast to input shape and divide by pooled dimension size
            pool_size = x.shape[axis]
            grad_input = np.broadcast_to(grad_input, x.shape) / pool_size

            x.backward(grad_input)
            if hasattr(x, "_backward"):
                x._backward()

        result._grad_fn = GradientFunction(backward_fn, [x], "mean_pool")

    logger.debug(f"Mean pool operation: {x.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def max_pool(x: Tensor, axis: int = 1) -> Tensor:
    """Max pooling operation with gradient support.

    Args:
        x: Input tensor
        axis: Axis along which to pool

    Returns:
        Pooled tensor

    Mathematical Definition:
        output = max(x, axis=axis)
        ∂output/∂x = 1 for max element, 0 elsewhere
    """
    # Perform max pooling
    result_data = np.max(x.data, axis=axis, keepdims=False)

    # Get indices of maximum values for gradient computation
    max_indices = np.argmax(x.data, axis=axis)

    # Create result tensor
    result = Tensor(
        result_data, requires_grad=x.requires_grad, name=f"max_pool({x.name or 'tensor'})"
    )

    # Set up gradient computation
    if x.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for max pooling."""
            # Create mask for maximum values
            grad_input = np.zeros_like(x.data)

            # Expand dimensions to match input
            grad_output_expanded = np.expand_dims(grad_output, axis=axis)

            # Create indices for advanced indexing
            input_shape = list(x.shape)
            indices = [np.arange(dim) for dim in input_shape]
            indices[axis] = max_indices

            # Use advanced indexing to set gradients at max positions
            mesh_indices = np.meshgrid(*indices, indexing="ij")
            grad_input[tuple(mesh_indices)] = grad_output_expanded.flatten()

            x.backward(grad_input)
            if hasattr(x, "_backward"):
                x._backward()

        result._grad_fn = GradientFunction(backward_fn, [x], "max_pool")

    logger.debug(f"Max pool operation: {x.shape} -> {result.shape}")
    return result
