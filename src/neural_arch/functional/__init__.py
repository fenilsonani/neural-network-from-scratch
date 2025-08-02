"""Functional operations for tensors."""

from .activation import (
    geglu,
    gelu,
    glu,
    leaky_relu,
    mish,
    reglu,
    relu,
    sigmoid,
    silu,
    softmax,
    swiglu,
    swish,
    tanh,
)
from .arithmetic import add, div, matmul, mul, neg, sub
from .loss import (
    cosine_embedding_loss,
    cross_entropy_loss,
    focal_loss,
    huber_loss,
    kl_divergence_loss,
    label_smoothing_cross_entropy,
    mse_loss,
    triplet_loss,
)
from .pooling import max_pool, mean_pool
from .utils import broadcast_tensors, reduce_gradient

__all__ = [
    # Arithmetic operations
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "matmul",
    # Activation functions
    "relu",
    "softmax",
    "sigmoid",
    "tanh",
    "gelu",
    "mish",
    "silu",
    "swiglu",
    "leaky_relu",
    "swish",
    "glu",
    "reglu",
    "geglu",
    # Pooling operations
    "mean_pool",
    "max_pool",
    # Loss functions
    "cross_entropy_loss",
    "mse_loss",
    "focal_loss",
    "label_smoothing_cross_entropy",
    "huber_loss",
    "kl_divergence_loss",
    "cosine_embedding_loss",
    "triplet_loss",
    # Utilities
    "broadcast_tensors",
    "reduce_gradient",
]
