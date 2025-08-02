"""Activation function layers."""

from ..core import Module, Tensor
from ..functional import relu, sigmoid, softmax, tanh


class ReLU(Module):
    """ReLU activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Softmax(Module):
    """Softmax activation layer."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, axis=self.dim)


class Sigmoid(Module):
    """Sigmoid activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(Module):
    """Tanh activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class GELU(Module):
    """GELU activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        # Simplified GELU implementation
        return relu(x)  # Placeholder
