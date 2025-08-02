"""Dropout layer (placeholder)."""

from ..core import Module, Tensor


class Dropout(Module):
    """Dropout layer (placeholder)."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # In training mode, would apply dropout
        # For now, just return input
        return x
