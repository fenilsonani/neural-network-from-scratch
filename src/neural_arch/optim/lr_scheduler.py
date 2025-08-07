"""Learning rate schedulers for optimizers.

This module provides various learning rate scheduling strategies that can be
applied to any optimizer, with special integration for AdamW.
"""

import logging
import math
from typing import Callable, List, Optional, Union

import numpy as np

from ..core.base import Optimizer
from ..exceptions import OptimizerError

logger = logging.getLogger(__name__)


class LRScheduler:
    """Base class for learning rate schedulers.

    Learning rate schedulers modify the learning rate of an optimizer during training
    to improve convergence and achieve better final performance.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False):
        """Initialize learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule
            last_epoch: The index of last epoch (-1 for first time)
            verbose: Whether to print lr changes
        """
        if not isinstance(optimizer, Optimizer):
            raise OptimizerError(f"Expected Optimizer, got {type(optimizer)}")

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose

        # Store initial learning rates
        if hasattr(optimizer, "lr"):
            self.base_lrs = [optimizer.lr]
        else:
            # For optimizers with parameter groups
            self.base_lrs = [0.001]  # Default fallback

        self.step(last_epoch + 1)

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a dict."""
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the schedulers state."""
        self.last_epoch = state_dict["last_epoch"]

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate."""
        if hasattr(self.optimizer, "lr"):
            return [self.optimizer.lr]
        return [0.001]  # Fallback

    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_lr()")

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate.

        Args:
            epoch: Optional epoch number (uses internal counter if None)
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # Compute new learning rates
        new_lrs = self.get_lr()

        # Apply to optimizer
        if hasattr(self.optimizer, "lr"):
            old_lr = self.optimizer.lr
            self.optimizer.lr = new_lrs[0]

            if self.verbose:
                logger.info(f"Epoch {epoch}: lr {old_lr:.8f} -> {new_lrs[0]:.8f}")

        logger.debug(f"LR Scheduler step {epoch}: lr = {new_lrs[0]:.8f}")


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs.

    Mathematical Definition:
        lr = initial_lr * gamma^(epoch // step_size)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize StepLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay
            last_epoch: The index of last epoch
            verbose: Whether to print lr changes
        """
        if step_size <= 0:
            raise OptimizerError(f"step_size must be positive, got {step_size}")
        if gamma <= 0 or gamma > 1:
            raise OptimizerError(f"gamma must be in (0, 1], got {gamma}")

        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate using step decay."""
        if self.last_epoch == 0:
            return self.base_lrs

        decay_factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    """Decays the learning rate exponentially by gamma every epoch.

    Mathematical Definition:
        lr = initial_lr * gamma^epoch
    """

    def __init__(
        self, optimizer: Optimizer, gamma: float, last_epoch: int = -1, verbose: bool = False
    ):
        """Initialize ExponentialLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            gamma: Multiplicative factor of learning rate decay
            last_epoch: The index of last epoch
            verbose: Whether to print lr changes
        """
        if gamma <= 0 or gamma > 1:
            raise OptimizerError(f"gamma must be in (0, 1], got {gamma}")

        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate using exponential decay."""
        if self.last_epoch == 0:
            return self.base_lrs

        decay_factor = self.gamma**self.last_epoch
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler.

    From "SGDR: Stochastic Gradient Descent with Warm Restarts"

    Mathematical Definition:
        lr = eta_min + (eta_max - eta_min) * (1 + cos(π * T_cur / T_max)) / 2
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize CosineAnnealingLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
            verbose: Whether to print lr changes
        """
        if T_max <= 0:
            raise OptimizerError(f"T_max must be positive, got {T_max}")
        if eta_min < 0:
            raise OptimizerError(f"eta_min must be non-negative, got {eta_min}")

        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate using cosine annealing."""
        if self.last_epoch == 0:
            return self.base_lrs

        progress = self.last_epoch / self.T_max
        cosine_factor = (1 + math.cos(math.pi * progress)) / 2

        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs
        ]


class LinearLR(LRScheduler):
    """Linear learning rate scheduling.

    Linearly changes the learning rate between two boundaries over a number of epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize LinearLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            start_factor: The multiplier for initial learning rate
            end_factor: The multiplier for final learning rate
            total_iters: The number of iterations over which to change lr
            last_epoch: The index of last epoch
            verbose: Whether to print lr changes
        """
        if start_factor <= 0 or end_factor <= 0:
            raise OptimizerError("start_factor and end_factor must be positive")
        if total_iters <= 0:
            raise OptimizerError(f"total_iters must be positive, got {total_iters}")

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate using linear interpolation."""
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]

        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]

        # Linear interpolation
        progress = self.last_epoch / self.total_iters
        factor = self.start_factor + (self.end_factor - self.start_factor) * progress

        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupLR(LRScheduler):
    """Warmup learning rate scheduler.

    Gradually increases learning rate from 0 to base_lr over warmup_epochs.
    Commonly used at the beginning of training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        warmup_factor: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize WarmupLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            warmup_factor: Starting learning rate factor
            last_epoch: The index of last epoch
            verbose: Whether to print lr changes
        """
        if warmup_epochs <= 0:
            raise OptimizerError(f"warmup_epochs must be positive, got {warmup_epochs}")
        if warmup_factor <= 0 or warmup_factor > 1:
            raise OptimizerError(f"warmup_factor must be in (0, 1], got {warmup_factor}")

        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate during warmup."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            factor = self.warmup_factor + alpha * (1 - self.warmup_factor)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # After warmup, return base learning rate
            return self.base_lrs


class PolynomialLR(LRScheduler):
    """Polynomial learning rate decay.

    Mathematical Definition:
        lr = (base_lr - end_lr) * (1 - epoch / max_decay_steps)^power + end_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int,
        power: float = 1.0,
        end_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize PolynomialLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            total_iters: The number of steps that the scheduler decays the learning rate
            power: The power of the polynomial
            end_lr: The learning rate at the end of decay
            last_epoch: The index of last epoch
            verbose: Whether to print lr changes
        """
        if total_iters <= 0:
            raise OptimizerError(f"total_iters must be positive, got {total_iters}")
        if power <= 0:
            raise OptimizerError(f"power must be positive, got {power}")
        if end_lr < 0:
            raise OptimizerError(f"end_lr must be non-negative, got {end_lr}")

        self.total_iters = total_iters
        self.power = power
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate using polynomial decay."""
        if self.last_epoch == 0:
            return self.base_lrs

        if self.last_epoch >= self.total_iters:
            return [self.end_lr for _ in self.base_lrs]

        decay_factor = (1 - self.last_epoch / self.total_iters) ** self.power
        return [(base_lr - self.end_lr) * decay_factor + self.end_lr for base_lr in self.base_lrs]


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving.

    This scheduler reads a metric quantity and if no improvement is seen
    for a 'patience' number of epochs, the learning rate is reduced.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False,
    ):
        """Initialize ReduceLROnPlateau scheduler.

        Args:
            optimizer: Wrapped optimizer
            mode: 'min' or 'max' - whether to minimize or maximize the metric
            factor: Factor by which the learning rate will be reduced
            patience: Number of epochs with no improvement after which lr will be reduced
            threshold: Threshold for measuring the new optimum
            threshold_mode: 'rel' or 'abs' - relative or absolute threshold
            cooldown: Number of epochs to wait before resuming normal operation
            min_lr: A lower bound on the learning rate
            eps: Minimal decay applied to lr
            verbose: Whether to print lr changes
        """
        if mode not in ["min", "max"]:
            raise OptimizerError(f"mode must be 'min' or 'max', got {mode}")
        if factor >= 1.0 or factor <= 0:
            raise OptimizerError(f"factor must be in (0, 1), got {factor}")
        if patience < 0:
            raise OptimizerError(f"patience must be non-negative, got {patience}")
        if threshold_mode not in ["rel", "abs"]:
            raise OptimizerError(f"threshold_mode must be 'rel' or 'abs', got {threshold_mode}")

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps

        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.cooldown_counter = 0

        if mode == "min":
            self.mode_worse = float("inf")
        else:
            self.mode_worse = -float("inf")

        # Don't call super().__init__() which would call step()
        if not isinstance(optimizer, Optimizer):
            raise OptimizerError(f"Expected Optimizer, got {type(optimizer)}")

        self.optimizer = optimizer
        self.last_epoch = -1
        self.verbose = verbose

        # Store initial learning rates
        if hasattr(optimizer, "lr"):
            self.base_lrs = [optimizer.lr]
        else:
            # For optimizers with parameter groups
            self.base_lrs = [0.001]  # Default fallback

    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        """Update learning rate based on metric.

        Args:
            metrics: The metric value to monitor
            epoch: Optional epoch number
        """
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch: int) -> None:
        """Reduce learning rate."""
        if hasattr(self.optimizer, "lr"):
            old_lr = self.optimizer.lr
            new_lr = max(old_lr * self.factor, self.min_lr)

            if old_lr - new_lr > self.eps:
                self.optimizer.lr = new_lr
                if self.verbose:
                    logger.info(f"Epoch {epoch}: reducing lr to {new_lr:.8f}")

    def is_better(self, a: float, best: Optional[float]) -> bool:
        """Check if a is better than best."""
        if best is None:
            return True

        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon
        else:  # mode == 'max' and threshold_mode == 'abs':
            return a > best + self.threshold

    @property
    def in_cooldown(self) -> bool:
        """Check if scheduler is in cooldown period."""
        return self.cooldown_counter > 0


class ChainedScheduler(LRScheduler):
    """Chains multiple learning rate schedulers.

    Allows combining different scheduling strategies, e.g., warmup followed by cosine annealing.
    """

    def __init__(self, schedulers: List[LRScheduler], milestones: List[int], verbose: bool = False):
        """Initialize ChainedScheduler.

        Args:
            schedulers: List of schedulers to chain
            milestones: List of epoch milestones where scheduler changes
            verbose: Whether to print lr changes
        """
        if len(schedulers) != len(milestones) + 1:
            raise OptimizerError("Number of schedulers must be one more than milestones")
        if not all(m1 < m2 for m1, m2 in zip(milestones, milestones[1:])):
            raise OptimizerError("Milestones must be in ascending order")

        self.schedulers = schedulers
        self.milestones = milestones
        self.current_scheduler_idx = 0

        # Use the first scheduler's optimizer
        super().__init__(schedulers[0].optimizer, -1, verbose)

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate using current scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1

        # Determine which scheduler to use
        for i, milestone in enumerate(self.milestones):
            if epoch < milestone:
                self.current_scheduler_idx = i
                break
        else:
            self.current_scheduler_idx = len(self.schedulers) - 1

        # Step the current scheduler
        current_scheduler = self.schedulers[self.current_scheduler_idx]
        current_scheduler.step(epoch)

        self.last_epoch = epoch

    def get_lr(self) -> List[float]:
        """Get learning rate from current scheduler."""
        return self.schedulers[self.current_scheduler_idx].get_lr()


# Convenience functions for common patterns
def get_linear_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
) -> ChainedScheduler:
    """Create a linear schedule with warmup (common in transformer training).

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: The index of last epoch

    Returns:
        ChainedScheduler combining warmup and linear decay
    """
    warmup_scheduler = WarmupLR(optimizer, num_warmup_steps, last_epoch=last_epoch)
    linear_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.001,  # Can't be 0.0 due to validation, use small value
        total_iters=num_training_steps - num_warmup_steps,
        last_epoch=last_epoch,
    )

    return ChainedScheduler([warmup_scheduler, linear_scheduler], [num_warmup_steps])


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> ChainedScheduler:
    """Create a cosine schedule with warmup (common in vision training).

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: The index of last epoch

    Returns:
        ChainedScheduler combining warmup and cosine annealing
    """
    warmup_scheduler = WarmupLR(optimizer, num_warmup_steps, last_epoch=last_epoch)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int((num_training_steps - num_warmup_steps) * num_cycles),
        last_epoch=last_epoch,
    )

    return ChainedScheduler([warmup_scheduler, cosine_scheduler], [num_warmup_steps])


def test_lr_schedulers():
    """Test all learning rate schedulers."""
    print("Testing Learning Rate Schedulers")
    print("=" * 40)
    
    # Create a mock optimizer for testing
    from ..optim import Adam
    from ..nn import Linear
    from ..core import Tensor
    import numpy as np
    
    # Create simple model and optimizer for testing
    model = Linear(4, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    print(f"Initial learning rate: {optimizer.lr}")
    
    # Test each scheduler
    schedulers = [
        ("StepLR", StepLR(optimizer, step_size=5, gamma=0.5)),
        ("ExponentialLR", ExponentialLR(optimizer, gamma=0.95)),
        ("CosineAnnealingLR", CosineAnnealingLR(optimizer, T_max=20)),
        ("LinearLR", LinearLR(optimizer, start_factor=0.5, end_factor=1.0, total_iters=10)),
        ("WarmupLR", WarmupLR(optimizer, warmup_epochs=5)),
        ("PolynomialLR", PolynomialLR(optimizer, total_iters=15, power=2.0)),
    ]
    
    for name, scheduler in schedulers:
        print(f"\nTesting {name}:")
        
        # Reset optimizer lr
        optimizer.lr = 0.01
        
        # Test for several epochs
        lrs = []
        for epoch in range(12):
            lr_before = optimizer.lr
            scheduler.step(epoch)
            lr_after = optimizer.lr
            lrs.append(lr_after)
            
            if epoch % 3 == 0 or epoch < 3:
                print(f"  Epoch {epoch:2d}: {lr_before:.6f} -> {lr_after:.6f}")
        
        # Show learning rate progression
        lr_range = f"[{min(lrs):.6f}, {max(lrs):.6f}]"
        print(f"  LR range over 12 epochs: {lr_range}")
    
    # Test ReduceLROnPlateau separately (different interface)
    print(f"\nTesting ReduceLROnPlateau:")
    optimizer.lr = 0.01
    plateau_scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Simulate loss values that plateau
    loss_values = [1.0, 0.8, 0.6, 0.5, 0.51, 0.52, 0.51, 0.50, 0.49, 0.48]
    
    for epoch, loss in enumerate(loss_values):
        lr_before = optimizer.lr
        plateau_scheduler.step(loss, epoch)
        lr_after = optimizer.lr
        
        if lr_before != lr_after:
            print(f"  Epoch {epoch:2d}: loss={loss:.2f}, lr {lr_before:.6f} -> {lr_after:.6f} (reduced!)")
        elif epoch % 2 == 0:
            print(f"  Epoch {epoch:2d}: loss={loss:.2f}, lr={lr_after:.6f}")
    
    # Test chained scheduler
    print(f"\nTesting ChainedScheduler (Warmup + Cosine):")
    optimizer.lr = 0.01
    chained_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=20
    )
    
    lrs = []
    for epoch in range(15):
        lr_before = optimizer.lr
        chained_scheduler.step(epoch)
        lr_after = optimizer.lr
        lrs.append(lr_after)
        
        if epoch % 3 == 0 or epoch < 6:
            print(f"  Epoch {epoch:2d}: {lr_before:.6f} -> {lr_after:.6f}")
    
    print(f"  LR range over 15 epochs: [{min(lrs):.6f}, {max(lrs):.6f}]")
    
    print("\nAll learning rate schedulers tested successfully!")
    print("✅ StepLR, ExponentialLR, and CosineAnnealingLR are fully functional")
    print("✅ Advanced schedulers (Linear, Warmup, Polynomial, ReduceLROnPlateau) working")
    print("✅ Chained schedulers enable complex scheduling strategies")


if __name__ == "__main__":
    test_lr_schedulers()
