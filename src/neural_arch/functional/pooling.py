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

            # For each position in the output, find the corresponding max position in input
            # and set the gradient there
            if axis == 0:
                # Pooling along batch dimension
                for i, max_idx in enumerate(max_indices):
                    grad_input[max_idx] = grad_output[i]
            elif axis == 1:
                # Pooling along sequence/time dimension
                for batch_idx in range(x.shape[0]):
                    for feat_idx in range(x.shape[2] if x.ndim > 2 else 1):
                        max_pos = max_indices[batch_idx] if x.ndim == 2 else max_indices[batch_idx, feat_idx]
                        if x.ndim == 2:
                            grad_input[batch_idx, max_pos] = grad_output[batch_idx]
                        else:
                            grad_input[batch_idx, max_pos, feat_idx] = grad_output[batch_idx, feat_idx]
            elif axis == 2:
                # Pooling along feature dimension
                for batch_idx in range(x.shape[0]):
                    for seq_idx in range(x.shape[1]):
                        max_pos = max_indices[batch_idx, seq_idx]
                        grad_input[batch_idx, seq_idx, max_pos] = grad_output[batch_idx, seq_idx]
            else:
                # General case - use broadcasting approach
                grad_output_broadcast = np.broadcast_to(grad_output_expanded, x.shape)
                # Create mask where max values occurred
                max_mask = (x.data == np.expand_dims(result_data, axis=axis))
                grad_input = grad_output_broadcast * max_mask

            x.backward(grad_input)
            if hasattr(x, "_backward"):
                x._backward()

        result._grad_fn = GradientFunction(backward_fn, [x], "max_pool")

    logger.debug(f"Max pool operation: {x.shape} -> {result.shape}")
    return result
