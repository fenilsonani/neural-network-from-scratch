"""SGD optimizer implementations."""

import numpy as np
from typing import Dict
import logging

from ..core.base import Optimizer, Parameter
from ..exceptions import OptimizerError, handle_exception

logger = logging.getLogger(__name__)


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and weight decay support."""
    
    def __init__(self, parameters, lr: float = 0.01, momentum: float = 0.0, 
                 weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False):
        # Convert parameters to dictionary if it's an iterator
        if hasattr(parameters, 'items'):  # Already a dict
            param_dict = parameters
        else:  # Iterator from model.parameters()
            param_dict = {f"param_{i}": param for i, param in enumerate(parameters)}
        
        super().__init__(param_dict, lr=lr)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        
        # Initialize momentum buffers
        self.momentum_buffer = {}
        for name, param in self.parameters.items():
            if momentum != 0:
                self.momentum_buffer[name] = np.zeros_like(param.data)
    
    @handle_exception
    def step(self) -> None:
        """Perform SGD step with momentum and weight decay."""
        for name, param in self.parameters.items():
            if param.grad is None:
                continue
            
            # Get gradient
            grad = param.grad
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                buf = self.momentum_buffer[name]
                buf = self.momentum * buf + (1 - self.dampening) * grad
                
                if self.nesterov:
                    grad = grad + self.momentum * buf
                else:
                    grad = buf
                
                self.momentum_buffer[name] = buf
            
            # Update parameters
            param.data = param.data - self.lr * grad
    
    @handle_exception  
    def zero_grad(self) -> None:
        """Zero gradients."""
        for param in self.parameters.values():
            param.zero_grad()


# Alias for compatibility
SGDMomentum = SGD