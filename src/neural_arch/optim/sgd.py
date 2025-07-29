"""SGD optimizer implementations."""

import numpy as np
from typing import Dict
import logging

from ..core.base import Optimizer, Parameter
from ..exceptions import OptimizerError, handle_exception

logger = logging.getLogger(__name__)


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, parameters: Dict[str, Parameter], lr: float = 0.01):
        super().__init__(parameters, lr=lr)
        self.lr = lr
    
    @handle_exception
    def step(self) -> None:
        """Perform SGD step."""
        for param in self.parameters.values():
            if param.grad is not None:
                param.data = param.data - self.lr * param.grad
    
    @handle_exception  
    def zero_grad(self) -> None:
        """Zero gradients."""
        for param in self.parameters.values():
            param.zero_grad()


# Alias for compatibility
SGDMomentum = SGD