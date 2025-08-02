"""Utility functions for tensor operations."""

import logging
from typing import List, Tuple

import numpy as np

from ..core.tensor import Shape, Tensor

logger = logging.getLogger(__name__)


def broadcast_tensors(*tensors: Tensor) -> List[np.ndarray]:
    """Broadcast multiple tensors to a common shape.

    Args:
        *tensors: Variable number of tensors to broadcast

    Returns:
        List of broadcasted numpy arrays

    Raises:
        ValueError: If tensors cannot be broadcasted together
    """
    try:
        arrays = [tensor.data for tensor in tensors]
        return list(np.broadcast_arrays(*arrays))
    except ValueError as e:
        shapes = [tensor.shape for tensor in tensors]
        raise ValueError(f"Cannot broadcast tensors with shapes {shapes}") from e


def reduce_gradient(grad: np.ndarray, target_shape: Shape, broadcast_shape: Shape) -> np.ndarray:
    """Reduce gradient from broadcasted shape back to original tensor shape.

    This handles gradient reduction for broadcasting operations and matrix operations.
    The key insight is that we need to sum over dimensions where the target tensor
    was smaller than the result, not rely on the broadcast_shape parameter.

    Args:
        grad: Gradient array from upstream
        target_shape: Original tensor shape to reduce to
        broadcast_shape: Shape that was used in the operation (may be ignored)

    Returns:
        Reduced gradient array matching target_shape
    """
    # Handle scalar case
    if not target_shape:
        return np.sum(grad)

    # Start with the gradient
    result = grad.copy()

    # Remove leading dimensions that were added
    while result.ndim > len(target_shape):
        result = np.sum(result, axis=0)

    # Now handle dimension-wise reduction
    # We need to sum over axes where target dimension is smaller than result dimension
    for i in range(len(target_shape)):
        if i < result.ndim:
            if target_shape[i] == 1 and result.shape[i] > 1:
                # Sum over this axis but keep dimension
                result = np.sum(result, axis=i, keepdims=True)
            elif target_shape[i] != result.shape[i]:
                # For matrix operations, we may need to sum over batch dimensions
                # or handle cases where gradients come from different operations
                if i == 0 and len(target_shape) < len(grad.shape):
                    # This is likely a batch dimension that needs summing
                    axes_to_sum = list(range(grad.ndim - len(target_shape)))
                    if axes_to_sum:
                        result = np.sum(grad, axis=tuple(axes_to_sum))
                        break

    # Final shape adjustment - reshape if needed
    if result.shape != target_shape:
        # Try to reshape or sum as needed
        if result.size == np.prod(target_shape):
            result = result.reshape(target_shape)
        else:
            # Sum over extra dimensions
            while result.ndim > len(target_shape):
                result = np.sum(result, axis=0)

            # If still doesn't match, sum over mismatched dimensions
            for i in range(min(len(target_shape), result.ndim)):
                if target_shape[i] != result.shape[i]:
                    if target_shape[i] == 1:
                        result = np.sum(result, axis=i, keepdims=True)

            # Final check - if shapes still don't match but total elements do
            if result.shape != target_shape and result.size == np.prod(target_shape):
                result = result.reshape(target_shape)

    return result


def get_broadcast_shape(*shapes: Shape) -> Shape:
    """Compute the broadcasted shape of multiple tensor shapes.

    Args:
        *shapes: Variable number of tensor shapes

    Returns:
        Broadcasted shape

    Raises:
        ValueError: If shapes cannot be broadcasted
    """
    if not shapes:
        return ()

    # Start with the first shape
    result_shape = list(shapes[0])

    for shape in shapes[1:]:
        # Pad shorter shape with 1s on the left
        shape = list(shape)
        max_len = max(len(result_shape), len(shape))
        result_shape = [1] * (max_len - len(result_shape)) + result_shape
        shape = [1] * (max_len - len(shape)) + shape

        # Check compatibility and compute result
        for i in range(max_len):
            if result_shape[i] == 1:
                result_shape[i] = shape[i]
            elif shape[i] == 1:
                pass  # Keep result_shape[i]
            elif result_shape[i] == shape[i]:
                pass  # Same size, no change needed
            else:
                raise ValueError(f"Cannot broadcast shapes: incompatible at dimension {i}")

    return tuple(result_shape)


def validate_tensor_operation(a: Tensor, b: Tensor, operation: str) -> None:
    """Validate that two tensors can be used in an operation.

    Args:
        a: First tensor
        b: Second tensor
        operation: Name of the operation for error messages

    Raises:
        TypeError: If inputs are not tensors
        ValueError: If tensors are incompatible
    """
    if not isinstance(a, Tensor):
        raise TypeError(f"{operation} requires Tensor inputs, got {type(a)} for first argument")

    if not isinstance(b, Tensor):
        raise TypeError(f"{operation} requires Tensor inputs, got {type(b)} for second argument")

    # Check for device compatibility (when we have multiple devices)
    if a.device != b.device:
        logger.warning(f"{operation}: tensors on different devices ({a.device} vs {b.device})")


def ensure_tensor(value: any, name: str = "tensor") -> Tensor:
    """Ensure a value is a Tensor, converting if necessary.

    Args:
        value: Value to convert
        name: Name for error messages

    Returns:
        Tensor instance

    Raises:
        TypeError: If value cannot be converted to tensor
    """
    if isinstance(value, Tensor):
        return value

    try:
        return Tensor(value)
    except Exception as e:
        raise TypeError(f"Cannot convert {type(value)} to Tensor for {name}") from e


def compute_output_shape(input_shape: Shape, operation: str, **kwargs) -> Shape:
    """Compute output shape for various tensor operations.

    Args:
        input_shape: Input tensor shape
        operation: Operation name
        **kwargs: Operation-specific parameters

    Returns:
        Output shape

    Raises:
        ValueError: If operation is not supported or parameters are invalid
    """
    if operation == "mean_pool":
        axis = kwargs.get("axis", 1)
        if axis < 0:
            axis = len(input_shape) + axis

        if axis >= len(input_shape):
            raise ValueError(f"Cannot pool over axis {axis} for shape {input_shape}")

        output_shape = list(input_shape)
        output_shape.pop(axis)
        return tuple(output_shape)

    elif operation == "softmax":
        return input_shape  # Softmax preserves shape

    elif operation == "relu":
        return input_shape  # ReLU preserves shape

    else:
        raise ValueError(f"Unknown operation: {operation}")


def check_finite_gradients(tensor: Tensor, operation: str) -> None:
    """Check if tensor gradients are finite and log warnings if not.

    Args:
        tensor: Tensor to check
        operation: Operation name for logging
    """
    if tensor.grad is not None:
        if not np.all(np.isfinite(tensor.grad)):
            if np.any(np.isnan(tensor.grad)):
                logger.warning(f"NaN gradients detected in {operation} for tensor {tensor.name}")
            if np.any(np.isinf(tensor.grad)):
                logger.warning(
                    f"Infinite gradients detected in {operation} for tensor {tensor.name}"
                )


def apply_gradient_clipping(grad: np.ndarray, max_norm: float = 10.0) -> np.ndarray:
    """Apply gradient clipping to prevent gradient explosion.

    Args:
        grad: Gradient array
        max_norm: Maximum allowed gradient norm

    Returns:
        Clipped gradient array
    """
    grad_norm = np.linalg.norm(grad)
    if grad_norm > max_norm:
        logger.debug(f"Clipping gradient: norm {grad_norm:.4f} -> {max_norm}")
        return grad * (max_norm / grad_norm)
    return grad


def memory_efficient_operation(func):
    """Decorator to add memory monitoring to tensor operations.

    This is a placeholder for enterprise memory monitoring.
    In a production system, this would track memory usage,
    detect memory leaks, and provide optimization suggestions.
    """

    def wrapper(*args, **kwargs):
        # Pre-operation memory check
        logger.debug(f"Starting operation: {func.__name__}")

        try:
            result = func(*args, **kwargs)

            # Post-operation memory check
            logger.debug(f"Completed operation: {func.__name__}")
            return result

        except Exception as e:
            logger.error(f"Error in operation {func.__name__}: {e}")
            raise

    return wrapper
