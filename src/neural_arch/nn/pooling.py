"""Pooling layers."""

from ..core import Module, Tensor
from ..functional import max_pool, mean_pool


class MeanPool(Module):
    """Mean pooling layer."""

    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return mean_pool(x, axis=self.axis)


class MaxPool(Module):
    """Max pooling layer."""

    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return max_pool(x, axis=self.axis)
