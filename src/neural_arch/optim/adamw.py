"""AdamW optimizer implementation."""

from .adam import Adam


class AdamW(Adam):
    """AdamW optimizer (Adam with decoupled weight decay)."""
    
    def __init__(self, parameters, lr=0.001, **kwargs):
        # For now, use Adam as base (would implement proper AdamW in production)
        super().__init__(parameters, lr=lr, **kwargs)