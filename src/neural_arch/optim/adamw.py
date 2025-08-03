"""AdamW optimizer with proper weight decay decoupling.

This implements AdamW as described in "Decoupled Weight Decay Regularization"
with mathematically correct weight decay decoupling.
"""

import logging
from typing import Dict, Optional

import numpy as np

from ..core.base import Optimizer, Parameter
from ..exceptions import OptimizerError, handle_exception

logger = logging.getLogger(__name__)


class AdamW(Optimizer):
    """AdamW: Adam with Decoupled Weight Decay Regularization.

    This implements proper weight decay decoupling as described in:
    "Decoupled Weight Decay Regularization" (https://arxiv.org/abs/1711.05101)

    Key differences from standard Adam:
    - Weight decay is applied directly to parameters, not to gradients
    - This leads to different behavior especially with momentum
    - Better generalization performance in many cases

    Mathematical Definition:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})

    where λ is the weight decay coefficient and θ represents parameters.
    """

    def __init__(
        self,
        parameters,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        betas: Optional[tuple] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[Dict] = None,
    ) -> None:
        """Initialize AdamW optimizer.

        Args:
            parameters: Dictionary or iterator of parameters to optimize
            lr: Learning rate
            beta1: Coefficient for computing running averages of gradient
            beta2: Coefficient for computing running averages of squared gradient
            eps: Term added to denominator for numerical stability (smaller than Adam)
            weight_decay: Weight decay coefficient (decoupled from gradients)
            amsgrad: Whether to use AMSGrad variant
            maximize: Maximize objective instead of minimize
            betas: PyTorch-style tuple (beta1, beta2) - overrides individual beta params
            lr_scheduler: Learning rate scheduler type ('cosine', 'linear', 'exponential', 'step', 'warmup_cosine')
            lr_scheduler_params: Parameters for the learning rate scheduler

        Raises:
            OptimizerError: If parameters are invalid
        """
        # Handle PyTorch-style betas parameter
        if betas is not None:
            if len(betas) != 2:
                raise OptimizerError(f"betas must be a tuple of length 2, got {len(betas)}")
            beta1, beta2 = betas

        # Convert parameters to dictionary if it's an iterator
        if hasattr(parameters, "items"):  # Already a dict
            param_dict = parameters
        else:  # Iterator from model.parameters()
            param_dict = {f"param_{i}": param for i, param in enumerate(parameters)}

        super().__init__(
            param_dict,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )

        # Validate hyperparameters
        if not 0.0 <= lr:
            raise OptimizerError(f"Invalid learning rate: {lr}", learning_rate=lr)
        if not 0.0 <= beta1 < 1.0:
            raise OptimizerError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise OptimizerError(f"Invalid beta2: {beta2}")
        if not 0.0 <= eps:
            raise OptimizerError(f"Invalid epsilon: {eps}")
        if not 0.0 <= weight_decay:
            raise OptimizerError(f"Invalid weight decay: {weight_decay}")

        self.lr = lr
        self.initial_lr = lr  # Store initial learning rate for scheduling
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        # Initialize state for each parameter
        self.state = {}
        self.step_count = 0  # Global step counter

        # Test-expected attributes
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (velocity)

        for name, param in self.parameters.items():
            exp_avg = np.zeros_like(param.data, dtype=np.float64)
            exp_avg_sq = np.zeros_like(param.data, dtype=np.float64)

            self.state[name] = {
                "step": 0,
                "exp_avg": exp_avg,  # First moment estimate
                "exp_avg_sq": exp_avg_sq,  # Second moment estimate
            }

            # Test-expected attributes
            self.m[name] = exp_avg
            self.v[name] = exp_avg_sq

            if amsgrad:
                self.state[name]["max_exp_avg_sq"] = np.zeros_like(param.data)

        # Initialize learning rate scheduler
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}

        if lr_scheduler:
            self._validate_scheduler_params()

        logger.info(
            f"Initialized AdamW optimizer: lr={lr}, beta1={beta1}, beta2={beta2}, weight_decay={weight_decay}"
        )
        if lr_scheduler:
            logger.info(
                f"Learning rate scheduler: {lr_scheduler} with params {self.lr_scheduler_params}"
            )

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

    def _validate_scheduler_params(self) -> None:
        """Validate learning rate scheduler parameters."""
        if self.lr_scheduler == "cosine":
            if "T_max" not in self.lr_scheduler_params:
                raise OptimizerError("Cosine scheduler requires 'T_max' parameter")
            if "eta_min" not in self.lr_scheduler_params:
                self.lr_scheduler_params["eta_min"] = 0.0

        elif self.lr_scheduler == "linear":
            if "total_steps" not in self.lr_scheduler_params:
                raise OptimizerError("Linear scheduler requires 'total_steps' parameter")

        elif self.lr_scheduler == "exponential":
            if "gamma" not in self.lr_scheduler_params:
                raise OptimizerError("Exponential scheduler requires 'gamma' parameter")

        elif self.lr_scheduler == "step":
            if "step_size" not in self.lr_scheduler_params:
                raise OptimizerError("Step scheduler requires 'step_size' parameter")
            if "gamma" not in self.lr_scheduler_params:
                self.lr_scheduler_params["gamma"] = 0.1

        elif self.lr_scheduler == "warmup_cosine":
            if "warmup_steps" not in self.lr_scheduler_params:
                raise OptimizerError("Warmup cosine scheduler requires 'warmup_steps' parameter")
            if "T_max" not in self.lr_scheduler_params:
                raise OptimizerError("Warmup cosine scheduler requires 'T_max' parameter")
            if "eta_min" not in self.lr_scheduler_params:
                self.lr_scheduler_params["eta_min"] = 0.0

        else:
            available = ["cosine", "linear", "exponential", "step", "warmup_cosine"]
            raise OptimizerError(f"Unknown scheduler '{self.lr_scheduler}'. Available: {available}")

    def _update_learning_rate(self) -> None:
        """Update learning rate based on the scheduler."""
        if not self.lr_scheduler:
            return

        if self.lr_scheduler == "cosine":
            # Cosine annealing
            T_max = self.lr_scheduler_params["T_max"]
            eta_min = self.lr_scheduler_params["eta_min"]

            progress = min(self.step_count / T_max, 1.0)
            self.lr = eta_min + (self.initial_lr - eta_min) * 0.5 * (1 + np.cos(np.pi * progress))

        elif self.lr_scheduler == "linear":
            # Linear decay
            total_steps = self.lr_scheduler_params["total_steps"]
            progress = min(self.step_count / total_steps, 1.0)
            self.lr = self.initial_lr * (1.0 - progress)

        elif self.lr_scheduler == "exponential":
            # Exponential decay
            gamma = self.lr_scheduler_params["gamma"]
            self.lr = self.initial_lr * (gamma**self.step_count)

        elif self.lr_scheduler == "step":
            # Step decay
            step_size = self.lr_scheduler_params["step_size"]
            gamma = self.lr_scheduler_params["gamma"]
            steps_completed = self.step_count // step_size
            self.lr = self.initial_lr * (gamma**steps_completed)

        elif self.lr_scheduler == "warmup_cosine":
            # Warmup followed by cosine annealing
            warmup_steps = self.lr_scheduler_params["warmup_steps"]
            T_max = self.lr_scheduler_params["T_max"]
            eta_min = self.lr_scheduler_params["eta_min"]

            if self.step_count <= warmup_steps:
                # Linear warmup
                self.lr = self.initial_lr * (self.step_count / warmup_steps)
            else:
                # Cosine annealing after warmup
                cosine_steps = self.step_count - warmup_steps
                cosine_T_max = T_max - warmup_steps
                progress = min(cosine_steps / cosine_T_max, 1.0) if cosine_T_max > 0 else 1.0
                self.lr = eta_min + (self.initial_lr - eta_min) * 0.5 * (
                    1 + np.cos(np.pi * progress)
                )

    @handle_exception
    def step(self) -> None:
        """Perform a single optimization step with decoupled weight decay.

        The key difference from Adam is that weight decay is applied
        directly to parameters, not added to gradients.

        Raises:
            OptimizerError: If optimization step fails
        """
        self.step_count += 1  # Increment global step counter

        # Update learning rate based on scheduler
        self._update_learning_rate()

        for name, param in self.parameters.items():
            if param.grad is None:
                continue

            # Get parameter state
            state = self.state[name]

            # Get gradient (negate if maximizing)
            grad = param.grad
            if self.maximize:
                grad = -grad

            # Get state variables
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Update step count
            state["step"] += 1
            step = state["step"]

            # Exponential moving average of gradient values (use higher precision)
            exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * grad.astype(np.float64)

            # Exponential moving average of squared gradient values (use higher precision)
            exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * (grad.astype(np.float64) ** 2)

            # Update state (important for next iteration)
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq

            # Update test-expected attributes
            self.m[name] = exp_avg
            self.v[name] = exp_avg_sq

            # Bias correction (ensure we maintain precision)
            bias_correction1 = 1 - self.beta1**step
            bias_correction2 = 1 - self.beta2**step

            # Apply bias correction with higher precision
            corrected_exp_avg = exp_avg / bias_correction1
            corrected_exp_avg_sq = exp_avg_sq / bias_correction2

            # Compute denominator with careful numerical handling
            if self.amsgrad:
                # AMSGrad variant: use maximum of past squared gradients
                max_exp_avg_sq = state["max_exp_avg_sq"]
                np.maximum(max_exp_avg_sq, corrected_exp_avg_sq, out=max_exp_avg_sq)
                denom = np.sqrt(max_exp_avg_sq) + self.eps
            else:
                # Add epsilon before sqrt for better numerical stability
                denom = np.sqrt(corrected_exp_avg_sq + self.eps)

            # Compute step size
            step_size = self.lr / bias_correction1

            # Apply the update with decoupled weight decay
            # This is the key difference from Adam: weight decay is applied directly to parameters
            # AdamW: θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε)) - α * λ * θ_{t-1}
            #      = θ_{t-1} * (1 - α * λ) - α * (m̂_t / (√v̂_t + ε))

            # Standard Adam-style update
            adam_update = step_size * corrected_exp_avg / denom
            adam_update = adam_update.astype(param.data.dtype)

            # Apply gradient clipping for numerical stability
            adam_update = np.clip(adam_update, -10.0, 10.0)

            # Apply weight decay directly to parameters (decoupled)
            if self.weight_decay != 0:
                # θ_t = θ_{t-1} * (1 - lr * weight_decay) - adam_update
                param.data = param.data * (1 - self.lr * self.weight_decay) - adam_update
            else:
                # No weight decay: θ_t = θ_{t-1} - adam_update
                param.data = param.data - adam_update

            # Check for numerical issues
            if not np.all(np.isfinite(param.data)):
                logger.warning(f"Non-finite values detected in parameter {name}")
                param.data = np.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)

        logger.debug("Completed AdamW optimization step")

    def get_statistics(self) -> Dict:
        """Get optimization statistics for monitoring."""
        stats = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "num_parameters": len(self.parameters),
            "total_steps": max((state["step"] for state in self.state.values()), default=0),
        }

        # Compute gradient and parameter statistics
        grad_norms = []
        param_norms = []
        weight_decay_effects = []

        for name, param in self.parameters.items():
            if param.grad is not None:
                grad_norms.append(np.linalg.norm(param.grad))
            param_norms.append(np.linalg.norm(param.data))

            # Estimate weight decay effect
            if self.weight_decay > 0:
                weight_decay_effects.append(
                    self.lr * self.weight_decay * np.linalg.norm(param.data)
                )

        if grad_norms:
            stats.update(
                {
                    "avg_grad_norm": np.mean(grad_norms),
                    "max_grad_norm": np.max(grad_norms),
                }
            )

        if param_norms:
            stats.update(
                {
                    "avg_param_norm": np.mean(param_norms),
                    "max_param_norm": np.max(param_norms),
                }
            )

        if weight_decay_effects:
            stats.update(
                {
                    "avg_weight_decay_effect": np.mean(weight_decay_effects),
                }
            )

        return stats

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr

    def set_lr(self, lr: float) -> None:
        """Set learning rate manually (overrides scheduler)."""
        if lr < 0:
            raise OptimizerError(f"Learning rate must be non-negative, got {lr}")
        self.lr = lr
        logger.info(f"Learning rate manually set to {lr}")

    def get_scheduler_info(self) -> Dict:
        """Get information about the learning rate scheduler."""
        if not self.lr_scheduler:
            return {"scheduler": None}

        return {
            "scheduler": self.lr_scheduler,
            "params": self.lr_scheduler_params.copy(),
            "current_lr": self.lr,
            "initial_lr": self.initial_lr,
            "step_count": self.step_count,
        }

    def reset_scheduler(self) -> None:
        """Reset the learning rate scheduler to initial state."""
        self.lr = self.initial_lr
        self.step_count = 0
        logger.info("Learning rate scheduler reset to initial state")

    def preview_lr_schedule(self, steps: int = 100) -> list:
        """Preview the learning rate schedule for given number of steps.

        Args:
            steps: Number of steps to preview

        Returns:
            List of learning rates for each step
        """
        if not self.lr_scheduler:
            return [self.lr] * steps

        # Save current state
        original_lr = self.lr
        original_step_count = self.step_count

        # Simulate schedule
        lr_schedule = []
        self.step_count = 0

        for step in range(steps):
            self.step_count = step
            self._update_learning_rate()
            lr_schedule.append(self.lr)

        # Restore original state
        self.lr = original_lr
        self.step_count = original_step_count

        return lr_schedule

    def zero_grad(self) -> None:
        """Clear gradients for all parameters."""
        for param in self.parameters.values():
            param.zero_grad()
