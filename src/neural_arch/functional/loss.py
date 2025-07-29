"""Loss functions with automatic differentiation."""

import numpy as np
import logging

from ..core.tensor import Tensor, GradientFunction
from .utils import memory_efficient_operation
from .activation import softmax

logger = logging.getLogger(__name__)


@memory_efficient_operation
def cross_entropy_loss(predictions: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    """Cross-entropy loss function.
    
    Args:
        predictions: Predicted logits of shape (batch_size, num_classes)
        targets: Target class indices of shape (batch_size,)
        reduction: How to reduce the loss ('mean', 'sum', 'none')
        
    Returns:
        Loss tensor
        
    Mathematical Definition:
        loss = -log(softmax(predictions)[targets])
    """
    # Apply softmax to get probabilities
    probs = softmax(predictions)
    
    # Convert targets to indices if needed
    if targets.data.ndim == 1:
        target_indices = targets.data.astype(int)
    else:
        target_indices = np.argmax(targets.data, axis=1)
    
    # Extract probabilities for target classes
    batch_size = predictions.shape[0]
    target_probs = probs.data[np.arange(batch_size), target_indices]
    
    # Compute cross-entropy loss (add epsilon for numerical stability)
    epsilon = 1e-8
    loss_data = -np.log(target_probs + epsilon)
    
    # Apply reduction
    if reduction == 'mean':
        loss_data = np.mean(loss_data)
    elif reduction == 'sum':
        loss_data = np.sum(loss_data)
    elif reduction == 'none':
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    # Create result tensor
    result = Tensor(
        loss_data,
        requires_grad=predictions.requires_grad,
        name="cross_entropy_loss"
    )
    
    # Set up gradient computation
    if predictions.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for cross-entropy loss."""
            # Gradient of cross-entropy w.r.t. logits
            grad_predictions = probs.data.copy()
            grad_predictions[np.arange(batch_size), target_indices] -= 1.0
            
            # Apply reduction scaling
            if reduction == 'mean':
                grad_predictions = grad_predictions / batch_size
            
            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_predictions = grad_predictions * grad_output.item()
            else:
                grad_predictions = grad_predictions * grad_output
            
            predictions.backward(grad_predictions)
            if hasattr(predictions, '_backward'):
                predictions._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [predictions, targets], "cross_entropy_loss")
    
    logger.debug(f"Cross-entropy loss: {predictions.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def mse_loss(predictions: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    """Mean squared error loss function.
    
    Args:
        predictions: Predicted values
        targets: Target values
        reduction: How to reduce the loss ('mean', 'sum', 'none')
        
    Returns:
        Loss tensor
        
    Mathematical Definition:
        loss = (predictions - targets)^2
    """
    # Compute squared differences
    diff = predictions.data - targets.data
    loss_data = diff ** 2
    
    # Apply reduction
    if reduction == 'mean':
        loss_data = np.mean(loss_data)
    elif reduction == 'sum':
        loss_data = np.sum(loss_data)
    elif reduction == 'none':
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    # Create result tensor
    result = Tensor(
        loss_data,
        requires_grad=predictions.requires_grad or targets.requires_grad,
        name="mse_loss"
    )
    
    # Set up gradient computation
    if predictions.requires_grad or targets.requires_grad:
        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for MSE loss."""
            # Gradient of MSE w.r.t. predictions and targets
            grad_factor = 2.0 * diff
            
            # Apply reduction scaling
            if reduction == 'mean':
                grad_factor = grad_factor / diff.size
            
            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_factor = grad_factor * grad_output.item()
            else:
                grad_factor = grad_factor * grad_output
            
            if predictions.requires_grad:
                predictions.backward(grad_factor)
                if hasattr(predictions, '_backward'):
                    predictions._backward()
            
            if targets.requires_grad:
                targets.backward(-grad_factor)
                if hasattr(targets, '_backward'):
                    targets._backward()
        
        result._grad_fn = GradientFunction(backward_fn, [predictions, targets], "mse_loss")
    
    logger.debug(f"MSE loss: {predictions.shape} -> {result.shape}")
    return result