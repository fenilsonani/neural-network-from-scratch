"""SGD optimizer implementations."""

import logging
from typing import Dict

import numpy as np

from ..core.base import Optimizer, Parameter
from ..exceptions import OptimizerError, handle_exception

logger = logging.getLogger(__name__)


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and weight decay support."""

    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        # Convert parameters to dictionary if it's an iterator
        if hasattr(parameters, "items"):  # Already a dict
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
        
        logger.info(f"Initialized SGD optimizer: lr={lr}, momentum={momentum}")

    def step_with_mixed_precision(self, scaler=None):
        """Optimizer step with mixed precision support."""
        if scaler is not None:
            # Use provided scaler
            success = scaler.step(self)
            if success:
                scaler.update()
            return success
        else:
            # Try automatic mixed precision
            try:
                from ..optimization.mixed_precision import get_mixed_precision_manager
                
                mp_manager = get_mixed_precision_manager()
                if mp_manager.config.enabled:
                    return mp_manager.scaler.step(self)
            except Exception:
                pass
            
            # Fallback to normal step
            self.step()
            return True

    def create_amp_version(self, scaler_config=None):
        """Create an AMP-aware version of this optimizer.
        
        Args:
            scaler_config: Configuration for gradient scaler
            
        Returns:
            AMP-aware optimizer wrapper
        """
        try:
            from ..optimization.amp_optimizer import AMPOptimizerFactory
            return AMPOptimizerFactory.wrap_optimizer(self, scaler_config=scaler_config)
        except ImportError:
            logger.warning("AMP optimizer not available, returning self")
            return self

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
