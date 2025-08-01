"""Functional operations for tensors."""

from .arithmetic import add, sub, mul, div, neg, matmul
from .activation import relu, softmax, sigmoid, tanh, gelu, mish, silu, swiglu, leaky_relu, swish, glu, reglu, geglu
from .pooling import mean_pool, max_pool
from .loss import (
    cross_entropy_loss, mse_loss, focal_loss, label_smoothing_cross_entropy,
    huber_loss, kl_divergence_loss, cosine_embedding_loss, triplet_loss
)
from .utils import broadcast_tensors, reduce_gradient

__all__ = [
    # Arithmetic operations
    "add", "sub", "mul", "div", "neg", "matmul",
    
    # Activation functions
    "relu", "softmax", "sigmoid", "tanh", "gelu", "mish", "silu", "swiglu", "leaky_relu", "swish", "glu", "reglu", "geglu",
    
    # Pooling operations
    "mean_pool", "max_pool",
    
    # Loss functions
    "cross_entropy_loss", "mse_loss", "focal_loss", "label_smoothing_cross_entropy",
    "huber_loss", "kl_divergence_loss", "cosine_embedding_loss", "triplet_loss",
    
    # Utilities
    "broadcast_tensors", "reduce_gradient",
]