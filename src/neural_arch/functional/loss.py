"""Loss functions with automatic differentiation."""

import logging

import numpy as np

from ..core.tensor import GradientFunction, Tensor
from .activation import softmax
from .utils import memory_efficient_operation

logger = logging.getLogger(__name__)


@memory_efficient_operation
def cross_entropy_loss(predictions: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
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

    # Convert targets to indices if needed using backend operations
    if targets.data.ndim == 1:
        # Convert backend data to numpy for indexing
        if hasattr(targets.backend_data, "get"):  # CuPy array
            target_indices = targets.backend_data.get().astype(int)
        else:
            target_indices = (
                targets.backend.to_numpy(targets.backend_data).astype(int)
                if hasattr(targets.backend, "to_numpy")
                else targets.data.astype(int)
            )
    else:
        # Convert backend data to numpy for argmax
        if hasattr(targets.backend_data, "get"):  # CuPy array
            target_data_np = targets.backend_data.get()
        else:
            target_data_np = (
                targets.backend.to_numpy(targets.backend_data)
                if hasattr(targets.backend, "to_numpy")
                else targets.data
            )
        target_indices = np.argmax(target_data_np, axis=1)

    # Extract probabilities for target classes
    batch_size = predictions.shape[0]
    # Convert probs.data to numpy for indexing
    if hasattr(probs.backend_data, "get"):  # CuPy array
        probs_data_np = probs.backend_data.get()
    else:
        probs_data_np = (
            probs.backend.to_numpy(probs.backend_data)
            if hasattr(probs.backend, "to_numpy")
            else probs.data
        )

    target_probs = probs_data_np[np.arange(batch_size), target_indices]

    # Compute cross-entropy loss (add epsilon for numerical stability)
    epsilon = 1e-8
    loss_data = -np.log(target_probs + epsilon)

    # Apply reduction
    if reduction == "mean":
        loss_data = np.mean(loss_data)
    elif reduction == "sum":
        loss_data = np.sum(loss_data)
    elif reduction == "none":
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Create result tensor
    result = Tensor(loss_data, requires_grad=predictions.requires_grad, name="cross_entropy_loss")

    # Set up gradient computation
    if predictions.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for cross-entropy loss."""
            # Gradient of cross-entropy w.r.t. logits
            grad_predictions = probs.data.copy()
            grad_predictions[np.arange(batch_size), target_indices] -= 1.0

            # Apply reduction scaling
            if reduction == "mean":
                grad_predictions = grad_predictions / batch_size

            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_predictions = grad_predictions * grad_output.item()
            else:
                grad_predictions = grad_predictions * grad_output

            predictions.backward(grad_predictions)
            if hasattr(predictions, "_backward"):
                predictions._backward()

        result._grad_fn = GradientFunction(
            backward_fn, [predictions, targets], "cross_entropy_loss"
        )

    logger.debug(f"Cross-entropy loss: {predictions.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def mse_loss(predictions: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
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
    # Compute squared differences using backend operations
    # Convert backend data to numpy for computation
    if hasattr(predictions.backend_data, "get"):  # CuPy array
        pred_data_np = predictions.backend_data.get()
    else:
        pred_data_np = (
            predictions.backend.to_numpy(predictions.backend_data)
            if hasattr(predictions.backend, "to_numpy")
            else predictions.data
        )

    if hasattr(targets.backend_data, "get"):  # CuPy array
        target_data_np = targets.backend_data.get()
    else:
        target_data_np = (
            targets.backend.to_numpy(targets.backend_data)
            if hasattr(targets.backend, "to_numpy")
            else targets.data
        )

    diff = pred_data_np - target_data_np
    loss_data = diff**2

    # Apply reduction
    if reduction == "mean":
        loss_data = np.mean(loss_data)
    elif reduction == "sum":
        loss_data = np.sum(loss_data)
    elif reduction == "none":
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Create result tensor
    result = Tensor(
        loss_data, requires_grad=predictions.requires_grad or targets.requires_grad, name="mse_loss"
    )

    # Set up gradient computation
    if predictions.requires_grad or targets.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for MSE loss."""
            # Gradient of MSE w.r.t. predictions and targets
            grad_factor = 2.0 * diff

            # Apply reduction scaling
            if reduction == "mean":
                grad_factor = grad_factor / diff.size

            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_factor = grad_factor * grad_output.item()
            else:
                grad_factor = grad_factor * grad_output

            if predictions.requires_grad:
                predictions.backward(grad_factor)
                if hasattr(predictions, "_backward"):
                    predictions._backward()

            if targets.requires_grad:
                targets.backward(-grad_factor)
                if hasattr(targets, "_backward"):
                    targets._backward()

        result._grad_fn = GradientFunction(backward_fn, [predictions, targets], "mse_loss")

    logger.debug(f"MSE loss: {predictions.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def focal_loss(
    predictions: Tensor,
    targets: Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """Focal Loss for addressing class imbalance.

    From "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)

    Args:
        predictions: Predicted logits of shape (batch_size, num_classes)
        targets: Target class indices of shape (batch_size,)
        alpha: Weighting factor for rare class (typically 0.25)
        gamma: Focusing parameter (typically 2.0)
        reduction: How to reduce the loss ('mean', 'sum', 'none')

    Returns:
        Focal loss tensor

    Mathematical Definition:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        where p_t is the model's estimated probability for the true class
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

    # Compute focal loss components
    epsilon = 1e-8
    target_probs_clamped = np.clip(target_probs, epsilon, 1.0 - epsilon)

    # (1 - p_t)^gamma term
    modulating_factor = (1.0 - target_probs_clamped) ** gamma

    # -log(p_t) term
    ce_loss = -np.log(target_probs_clamped)

    # Combine: FL = -α * (1-p_t)^γ * log(p_t)
    loss_data = alpha * modulating_factor * ce_loss

    # Apply reduction
    if reduction == "mean":
        loss_data = np.mean(loss_data)
    elif reduction == "sum":
        loss_data = np.sum(loss_data)
    elif reduction == "none":
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Create result tensor
    result = Tensor(loss_data, requires_grad=predictions.requires_grad, name="focal_loss")

    # Set up gradient computation
    if predictions.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for focal loss.

            Implements mathematically correct focal loss gradients using chain rule:
            ∂FL/∂x_i = ∂FL/∂p_t * ∂p_t/∂x_i

            For true class t: ∂FL/∂x_t = -α_t * (1-p_t)^γ * [-γ * log(p_t) + (1-p_t)/p_t]
            For non-true class i: ∂FL/∂x_i = α_t * (1-p_t)^(γ-1) * p_i * [-γ * log(p_t) + (1-p_t)/p_t]
            """
            grad_predictions = np.zeros_like(probs.data)

            for sample in range(batch_size):
                target_idx = target_indices[sample]
                p_t = target_probs_clamped[sample]
                p_all = probs.data[sample]  # All class probabilities for this sample

                if gamma == 0.0:
                    # Weighted cross-entropy: standard CE gradient scaled by alpha
                    grad_predictions[sample] = alpha * p_all
                    grad_predictions[sample, target_idx] = alpha * (p_t - 1.0)
                else:
                    # Implement the exact mathematical derivation step by step
                    # FL = -α * (1-p_t)^γ * log(p_t)
                    #
                    # Using chain rule: ∂FL/∂x_i = ∂FL/∂p_t * ∂p_t/∂x_i
                    #
                    # ∂FL/∂p_t = -α * [γ * (1-p_t)^(γ-1) * (-1) * log(p_t) + (1-p_t)^γ * (1/p_t)]
                    #           = α * [γ * (1-p_t)^(γ-1) * log(p_t) - (1-p_t)^γ / p_t]

                    one_minus_pt = 1.0 - p_t
                    log_pt = np.log(p_t + 1e-8)

                    # ∂FL/∂p_t
                    dFL_dpt = alpha * (
                        gamma * (one_minus_pt ** (gamma - 1)) * log_pt - (one_minus_pt**gamma) / p_t
                    )

                    # Softmax derivatives: ∂p_t/∂x_i
                    # If i == t: ∂p_t/∂x_t = p_t * (1 - p_t)
                    # If i != t: ∂p_t/∂x_i = -p_t * p_i

                    # For target class: ∂FL/∂x_t = ∂FL/∂p_t * p_t * (1 - p_t)
                    grad_predictions[sample, target_idx] = dFL_dpt * p_t * one_minus_pt

                    # For non-target classes: ∂FL/∂x_i = ∂FL/∂p_t * (-p_t * p_i)
                    for class_idx in range(len(p_all)):
                        if class_idx != target_idx:
                            grad_predictions[sample, class_idx] = dFL_dpt * (
                                -p_t * p_all[class_idx]
                            )

            # Apply reduction scaling
            if reduction == "mean":
                grad_predictions = grad_predictions / batch_size

            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_predictions = grad_predictions * grad_output.item()
            else:
                grad_predictions = grad_predictions * grad_output

            predictions.backward(grad_predictions)
            if hasattr(predictions, "_backward"):
                predictions._backward()

        result._grad_fn = GradientFunction(backward_fn, [predictions, targets], "focal_loss")

    logger.debug(f"Focal loss: {predictions.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def label_smoothing_cross_entropy(
    predictions: Tensor, targets: Tensor, smoothing: float = 0.1, reduction: str = "mean"
) -> Tensor:
    """Label Smoothing Cross-Entropy Loss for better generalization.

    From "Rethinking the Inception Architecture for Computer Vision" (https://arxiv.org/abs/1512.00567)

    Args:
        predictions: Predicted logits of shape (batch_size, num_classes)
        targets: Target class indices of shape (batch_size,)
        smoothing: Label smoothing parameter (typically 0.1)
        reduction: How to reduce the loss ('mean', 'sum', 'none')

    Returns:
        Label smoothed cross-entropy loss tensor

    Mathematical Definition:
        Smooth labels: y_smooth = (1 - ε) * y_hot + ε / K
        where ε is smoothing, y_hot is one-hot, K is num_classes
        Loss: -sum(y_smooth * log_softmax(predictions))
    """
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"Smoothing must be in [0, 1), got {smoothing}")

    batch_size, num_classes = predictions.shape

    # Apply log_softmax for numerical stability
    # log_softmax(x) = x - log(sum(exp(x)))
    max_vals = np.max(predictions.data, axis=1, keepdims=True)
    shifted_logits = predictions.data - max_vals
    exp_shifted = np.exp(shifted_logits)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    log_softmax_data = shifted_logits - np.log(sum_exp)

    # Convert targets to indices
    if targets.data.ndim == 1:
        target_indices = targets.data.astype(int)
    else:
        target_indices = np.argmax(targets.data, axis=1)

    # Create smooth labels
    smooth_labels = np.full((batch_size, num_classes), smoothing / num_classes)
    smooth_labels[np.arange(batch_size), target_indices] = 1.0 - smoothing + smoothing / num_classes

    # Compute loss: -sum(smooth_labels * log_softmax)
    loss_data = -np.sum(smooth_labels * log_softmax_data, axis=1)

    # Apply reduction
    if reduction == "mean":
        loss_data = np.mean(loss_data)
    elif reduction == "sum":
        loss_data = np.sum(loss_data)
    elif reduction == "none":
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Create result tensor
    result = Tensor(
        loss_data, requires_grad=predictions.requires_grad, name="label_smoothing_cross_entropy"
    )

    # Set up gradient computation
    if predictions.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for label smoothing cross-entropy."""
            # Gradient: softmax - smooth_labels
            softmax_data = np.exp(log_softmax_data)
            grad_predictions = softmax_data - smooth_labels

            # Apply reduction scaling
            if reduction == "mean":
                grad_predictions = grad_predictions / batch_size

            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_predictions = grad_predictions * grad_output.item()
            else:
                grad_predictions = grad_predictions * grad_output

            predictions.backward(grad_predictions)
            if hasattr(predictions, "_backward"):
                predictions._backward()

        result._grad_fn = GradientFunction(
            backward_fn, [predictions, targets], "label_smoothing_cross_entropy"
        )

    logger.debug(f"Label smoothing cross-entropy: {predictions.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def huber_loss(
    predictions: Tensor, targets: Tensor, delta: float = 1.0, reduction: str = "mean"
) -> Tensor:
    """Huber Loss for robust regression (less sensitive to outliers than MSE).

    Args:
        predictions: Predicted values
        targets: Target values
        delta: Threshold for switching between quadratic and linear loss
        reduction: How to reduce the loss ('mean', 'sum', 'none')

    Returns:
        Huber loss tensor

    Mathematical Definition:
        L_δ(a) = { 0.5 * a^2                    if |a| ≤ δ
                 { δ * (|a| - 0.5 * δ)         if |a| > δ
        where a = predictions - targets
    """
    if delta <= 0:
        raise ValueError(f"Delta must be positive, got {delta}")

    # Compute residuals
    residual = predictions.data - targets.data
    abs_residual = np.abs(residual)

    # Apply Huber loss formula
    quadratic_mask = abs_residual <= delta
    loss_data = np.where(
        quadratic_mask,
        0.5 * residual**2,  # Quadratic for small errors
        delta * (abs_residual - 0.5 * delta),  # Linear for large errors
    )

    # Apply reduction
    if reduction == "mean":
        loss_data = np.mean(loss_data)
    elif reduction == "sum":
        loss_data = np.sum(loss_data)
    elif reduction == "none":
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Create result tensor
    result = Tensor(
        loss_data,
        requires_grad=predictions.requires_grad or targets.requires_grad,
        name="huber_loss",
    )

    # Set up gradient computation
    if predictions.requires_grad or targets.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for Huber loss.

            Gradient:
            ∂L/∂a = { a              if |a| ≤ δ
                    { δ * sign(a)    if |a| > δ
            """
            # Compute gradient w.r.t. residual
            grad_residual = np.where(
                quadratic_mask,
                residual,  # Linear gradient for quadratic region
                delta * np.sign(residual),  # Constant gradient for linear region
            )

            # Apply reduction scaling
            if reduction == "mean":
                grad_residual = grad_residual / residual.size

            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_residual = grad_residual * grad_output.item()
            else:
                grad_residual = grad_residual * grad_output

            if predictions.requires_grad:
                predictions.backward(grad_residual)
                if hasattr(predictions, "_backward"):
                    predictions._backward()

            if targets.requires_grad:
                targets.backward(-grad_residual)
                if hasattr(targets, "_backward"):
                    targets._backward()

        result._grad_fn = GradientFunction(backward_fn, [predictions, targets], "huber_loss")

    logger.debug(f"Huber loss: {predictions.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def kl_divergence_loss(predictions: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Kullback-Leibler Divergence Loss for knowledge distillation.

    Args:
        predictions: Predicted log probabilities (log_softmax output)
        targets: Target probabilities (softmax output)
        reduction: How to reduce the loss ('mean', 'sum', 'none')

    Returns:
        KL divergence loss tensor

    Mathematical Definition:
        KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        where P is target distribution, Q is predicted distribution
    """
    # Ensure numerical stability
    epsilon = 1e-8

    # If predictions are logits, convert to log probabilities
    if predictions.data.max() > 0:  # Likely logits, not log probabilities
        # Convert to log probabilities using log_softmax
        max_vals = np.max(predictions.data, axis=-1, keepdims=True)
        shifted = predictions.data - max_vals
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        log_probs = shifted - np.log(sum_exp)
    else:
        log_probs = predictions.data

    # If targets are logits, convert to probabilities
    if targets.data.max() > 1.0 or targets.data.min() < 0:  # Likely logits
        # Convert to probabilities using softmax
        max_vals = np.max(targets.data, axis=-1, keepdims=True)
        shifted = targets.data - max_vals
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        target_probs = exp_shifted / sum_exp
    else:
        target_probs = targets.data

    # Clamp probabilities for numerical stability
    target_probs = np.clip(target_probs, epsilon, 1.0 - epsilon)

    # Compute KL divergence: sum(P * log(P) - P * log(Q))
    log_target_probs = np.log(target_probs)
    kl_per_sample = np.sum(target_probs * (log_target_probs - log_probs), axis=-1)

    # Apply reduction
    if reduction == "mean":
        loss_data = np.mean(kl_per_sample)
    elif reduction == "sum":
        loss_data = np.sum(kl_per_sample)
    elif reduction == "none":
        loss_data = kl_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Create result tensor
    result = Tensor(
        loss_data,
        requires_grad=predictions.requires_grad or targets.requires_grad,
        name="kl_divergence_loss",
    )

    # Set up gradient computation
    if predictions.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for KL divergence.

            Gradient w.r.t. log probabilities: -P (target probabilities)
            """
            grad_predictions = -target_probs

            # Apply reduction scaling
            if reduction == "mean":
                grad_predictions = grad_predictions / target_probs.shape[0]

            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_predictions = grad_predictions * grad_output.item()
            elif isinstance(grad_output, np.ndarray) and grad_output.ndim == 1:
                grad_predictions = grad_predictions * grad_output[:, None]
            else:
                grad_predictions = grad_predictions * grad_output

            predictions.backward(grad_predictions)
            if hasattr(predictions, "_backward"):
                predictions._backward()

        result._grad_fn = GradientFunction(
            backward_fn, [predictions, targets], "kl_divergence_loss"
        )

    logger.debug(f"KL divergence: {predictions.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def cosine_embedding_loss(
    input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0, reduction: str = "mean"
) -> Tensor:
    """Cosine Embedding Loss for similarity learning.

    Args:
        input1: First input embeddings of shape (batch_size, embedding_dim)
        input2: Second input embeddings of shape (batch_size, embedding_dim)
        target: Target labels (1 for similar, -1 for dissimilar) of shape (batch_size,)
        margin: Margin for dissimilar pairs (typically 0.0 to 1.0)
        reduction: How to reduce the loss ('mean', 'sum', 'none')

    Returns:
        Cosine embedding loss tensor

    Mathematical Definition:
        loss = { 1 - cos(x1, x2)              if y = 1
               { max(0, cos(x1, x2) - margin) if y = -1
        where cos(x1, x2) = (x1 · x2) / (||x1|| ||x2||)
    """
    if input1.shape != input2.shape:
        raise ValueError(f"Input shapes must match: {input1.shape} vs {input2.shape}")

    # Compute cosine similarity
    # Dot product
    dot_product = np.sum(input1.data * input2.data, axis=1)

    # L2 norms
    norm1 = np.sqrt(np.sum(input1.data**2, axis=1) + 1e-8)  # Add epsilon for stability
    norm2 = np.sqrt(np.sum(input2.data**2, axis=1) + 1e-8)

    # Cosine similarity
    cosine_sim = dot_product / (norm1 * norm2)

    # Compute loss based on target
    target_labels = target.data
    loss_data = np.where(
        target_labels == 1,
        1.0 - cosine_sim,  # For similar pairs: minimize 1 - cos_sim
        np.maximum(0.0, cosine_sim - margin),  # For dissimilar pairs: maximize distance from margin
    )

    # Apply reduction
    if reduction == "mean":
        loss_data = np.mean(loss_data)
    elif reduction == "sum":
        loss_data = np.sum(loss_data)
    elif reduction == "none":
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Create result tensor
    result = Tensor(
        loss_data,
        requires_grad=input1.requires_grad or input2.requires_grad,
        name="cosine_embedding_loss",
    )

    # Set up gradient computation
    if input1.requires_grad or input2.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for cosine embedding loss."""
            batch_size = input1.shape[0]

            # Compute gradients w.r.t. cosine similarity
            grad_cosine = np.where(
                target_labels == 1,
                -1.0,  # For similar pairs
                np.where(cosine_sim > margin, 1.0, 0.0),  # For dissimilar pairs
            )

            # Apply reduction scaling
            if reduction == "mean":
                grad_cosine = grad_cosine / batch_size

            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_cosine = grad_cosine * grad_output.item()

            # Compute gradients w.r.t. inputs using chain rule
            # ∂cos_sim/∂x1 = (x2 * ||x1|| ||x2|| - x1 * (x1·x2) * ||x2|| / ||x1||) / (||x1|| ||x2||)^2
            # Simplified: ∂cos_sim/∂x1 = (x2 - x1 * cos_sim) / (||x1|| ||x2||)

            norm_product = (norm1 * norm2)[:, None]  # (batch_size, 1)

            if input1.requires_grad:
                grad_input1 = grad_cosine[:, None] * (
                    input2.data / norm_product
                    - input1.data * cosine_sim[:, None] / (norm1[:, None] ** 2)
                )

                input1.backward(grad_input1)
                if hasattr(input1, "_backward"):
                    input1._backward()

            if input2.requires_grad:
                grad_input2 = grad_cosine[:, None] * (
                    input1.data / norm_product
                    - input2.data * cosine_sim[:, None] / (norm2[:, None] ** 2)
                )

                input2.backward(grad_input2)
                if hasattr(input2, "_backward"):
                    input2._backward()

        result._grad_fn = GradientFunction(
            backward_fn, [input1, input2, target], "cosine_embedding_loss"
        )

    logger.debug(f"Cosine embedding loss: {input1.shape} -> {result.shape}")
    return result


@memory_efficient_operation
def triplet_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """Triplet Loss for metric learning.

    Args:
        anchor: Anchor embeddings of shape (batch_size, embedding_dim)
        positive: Positive embeddings of shape (batch_size, embedding_dim)
        negative: Negative embeddings of shape (batch_size, embedding_dim)
        margin: Margin between positive and negative distances
        p: Norm degree for distance computation (1 for L1, 2 for L2)
        reduction: How to reduce the loss ('mean', 'sum', 'none')

    Returns:
        Triplet loss tensor

    Mathematical Definition:
        loss = max(0, ||anchor - positive||_p - ||anchor - negative||_p + margin)
    """
    if not (anchor.shape == positive.shape == negative.shape):
        raise ValueError(
            f"All input shapes must match: {anchor.shape}, {positive.shape}, {negative.shape}"
        )

    if p <= 0:
        raise ValueError(f"Norm degree p must be positive, got {p}")

    # Compute distances
    diff_pos = anchor.data - positive.data
    diff_neg = anchor.data - negative.data

    if p == 1.0:
        # L1 distance (Manhattan)
        dist_pos = np.sum(np.abs(diff_pos), axis=1)
        dist_neg = np.sum(np.abs(diff_neg), axis=1)
    elif p == 2.0:
        # L2 distance (Euclidean)
        dist_pos = np.sqrt(np.sum(diff_pos**2, axis=1) + 1e-8)
        dist_neg = np.sqrt(np.sum(diff_neg**2, axis=1) + 1e-8)
    else:
        # General Lp norm
        dist_pos = np.power(np.sum(np.power(np.abs(diff_pos), p), axis=1), 1.0 / p)
        dist_neg = np.power(np.sum(np.power(np.abs(diff_neg), p), axis=1), 1.0 / p)

    # Compute triplet loss
    loss_data = np.maximum(0.0, dist_pos - dist_neg + margin)

    # Apply reduction
    if reduction == "mean":
        loss_data = np.mean(loss_data)
    elif reduction == "sum":
        loss_data = np.sum(loss_data)
    elif reduction == "none":
        pass  # Keep individual losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Create result tensor
    result = Tensor(
        loss_data,
        requires_grad=anchor.requires_grad or positive.requires_grad or negative.requires_grad,
        name="triplet_loss",
    )

    # Set up gradient computation
    if anchor.requires_grad or positive.requires_grad or negative.requires_grad:

        def backward_fn(grad_output: np.ndarray) -> None:
            """Backward pass for triplet loss."""
            batch_size = anchor.shape[0]

            # Only compute gradients for active triplets (loss > 0)
            active_mask = (dist_pos - dist_neg + margin) > 0

            if not np.any(active_mask):
                # No active triplets, gradients are zero
                return

            # Apply reduction scaling
            grad_scale = 1.0
            if reduction == "mean":
                grad_scale = 1.0 / batch_size

            # Scale by upstream gradient
            if isinstance(grad_output, np.ndarray) and grad_output.size == 1:
                grad_scale *= grad_output.item()

            # Compute distance gradients for active triplets
            if p == 1.0:
                # L1 gradient: sign(diff)
                grad_pos_dist = np.sign(diff_pos)
                grad_neg_dist = np.sign(diff_neg)
            elif p == 2.0:
                # L2 gradient: diff / ||diff||
                grad_pos_dist = diff_pos / (dist_pos[:, None] + 1e-8)
                grad_neg_dist = diff_neg / (dist_neg[:, None] + 1e-8)
            else:
                # General Lp gradient
                sign_pos = np.sign(diff_pos)
                sign_neg = np.sign(diff_neg)
                abs_pos = np.abs(diff_pos)
                abs_neg = np.abs(diff_neg)

                grad_pos_dist = sign_pos * np.power(abs_pos, p - 1) / (dist_pos[:, None] + 1e-8)
                grad_neg_dist = sign_neg * np.power(abs_neg, p - 1) / (dist_neg[:, None] + 1e-8)

            # Apply active mask and scaling
            active_mask_expanded = active_mask[:, None]
            grad_pos_dist = grad_pos_dist * active_mask_expanded * grad_scale
            grad_neg_dist = grad_neg_dist * active_mask_expanded * grad_scale

            # Compute gradients w.r.t. inputs
            if anchor.requires_grad:
                grad_anchor = grad_pos_dist - grad_neg_dist
                anchor.backward(grad_anchor)
                if hasattr(anchor, "_backward"):
                    anchor._backward()

            if positive.requires_grad:
                grad_positive = -grad_pos_dist
                positive.backward(grad_positive)
                if hasattr(positive, "_backward"):
                    positive._backward()

            if negative.requires_grad:
                grad_negative = grad_neg_dist
                negative.backward(grad_negative)
                if hasattr(negative, "_backward"):
                    negative._backward()

        result._grad_fn = GradientFunction(
            backward_fn, [anchor, positive, negative], "triplet_loss"
        )

    logger.debug(f"Triplet loss: {anchor.shape} -> {result.shape}")
    return result
