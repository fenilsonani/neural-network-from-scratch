"""Functional operations for tensors."""

from .arithmetic import add, sub, mul, div, neg, matmul
from .activation import relu, softmax, sigmoid, tanh
from .pooling import mean_pool, max_pool
from .loss import cross_entropy_loss, mse_loss
from .utils import broadcast_tensors, reduce_gradient

__all__ = [
    # Arithmetic operations
    "add", "sub", "mul", "div", "neg", "matmul",
    
    # Activation functions
    "relu", "softmax", "sigmoid", "tanh",
    
    # Pooling operations
    "mean_pool", "max_pool",
    
    # Loss functions
    "cross_entropy_loss", "mse_loss",
    
    # Utilities
    "broadcast_tensors", "reduce_gradient",
]