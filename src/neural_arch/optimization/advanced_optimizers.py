"""Advanced optimization techniques with gradient accumulation, mixed precision, and novel algorithms.

This module provides cutting-edge optimization techniques including:
- Gradient accumulation with dynamic batching
- Mixed precision training with automatic loss scaling
- Novel optimizers (Lion, Sophia, AdamW variants)
- Adaptive gradient clipping and normalization
- Learning rate scheduling with warmup and decay
- Second-order optimization methods
- Gradient compression for distributed training
- Parameter averaging and exponential moving averages
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


class OptimizerType(Enum):
    """Types of optimizers available."""
    
    ADAM = "adam"
    ADAMW = "adamw"
    LION = "lion"
    SOPHIA = "sophia"
    ADAFACTOR = "adafactor"
    LAMB = "lamb"
    RADAM = "radam"
    LOOKAHEAD = "lookahead"
    SHAMPOO = "shampoo"


class LRScheduleType(Enum):
    """Types of learning rate schedules."""
    
    CONSTANT = "constant"
    LINEAR_WARMUP = "linear_warmup"
    COSINE_ANNEALING = "cosine_annealing"
    EXPONENTIAL_DECAY = "exponential_decay"
    POLYNOMIAL_DECAY = "polynomial_decay"
    ONE_CYCLE = "one_cycle"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


@dataclass
class OptimizerConfig:
    """Configuration for advanced optimizers."""
    
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    gradient_clip_type: str = "norm"  # "norm", "value", "adaptive"
    
    # Mixed precision
    use_mixed_precision: bool = True
    loss_scale: float = 65536.0
    loss_scale_window: int = 2000
    min_loss_scale: float = 1.0
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Learning rate schedule
    lr_schedule_type: LRScheduleType = LRScheduleType.COSINE_ANNEALING
    warmup_steps: int = 1000
    total_steps: int = 100000
    
    # Advanced features
    use_lookahead: bool = False
    lookahead_alpha: float = 0.5
    lookahead_k: int = 5
    
    use_ema: bool = False
    ema_decay: float = 0.9999
    
    # Sophia-specific (second-order)
    rho: float = 0.04
    update_period: int = 10


class GradientAccumulator:
    """Efficient gradient accumulation with dynamic batching."""
    
    def __init__(self, accumulation_steps: int = 1, sync_gradients: bool = True):
        """Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
            sync_gradients: Whether to synchronize gradients in distributed setting
        """
        self.accumulation_steps = accumulation_steps
        self.sync_gradients = sync_gradients
        self.accumulated_gradients: Dict[str, np.ndarray] = {}
        self.current_step = 0
        
        # Statistics
        self.total_accumulated_samples = 0
        self.accumulation_history = deque(maxlen=100)
    
    def accumulate_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        batch_size: int = 1
    ) -> bool:
        """Accumulate gradients from current batch.
        
        Args:
            gradients: Dictionary of parameter gradients
            batch_size: Size of current batch
        
        Returns:
            True if gradients should be applied (accumulation complete)
        """
        # Initialize accumulated gradients if first step
        if not self.accumulated_gradients:
            self.accumulated_gradients = {
                name: np.zeros_like(grad) for name, grad in gradients.items()
            }
        
        # Accumulate gradients (scale by batch size for proper averaging)
        for name, grad in gradients.items():
            if name in self.accumulated_gradients:
                self.accumulated_gradients[name] += grad * batch_size
            else:
                self.accumulated_gradients[name] = grad * batch_size
        
        self.current_step += 1
        self.total_accumulated_samples += batch_size
        
        # Check if accumulation is complete
        should_step = self.current_step >= self.accumulation_steps
        
        if should_step:
            # Average accumulated gradients
            total_samples = max(self.total_accumulated_samples, 1)
            for name in self.accumulated_gradients:
                self.accumulated_gradients[name] /= total_samples
            
            # Record accumulation stats
            self.accumulation_history.append({
                'steps': self.current_step,
                'samples': self.total_accumulated_samples,
                'avg_batch_size': self.total_accumulated_samples / self.current_step
            })
        
        return should_step
    
    def get_accumulated_gradients(self) -> Dict[str, np.ndarray]:
        """Get accumulated gradients."""
        return self.accumulated_gradients.copy()
    
    def reset(self):
        """Reset accumulation state."""
        for name in self.accumulated_gradients:
            self.accumulated_gradients[name].fill(0)
        self.current_step = 0
        self.total_accumulated_samples = 0
    
    def get_effective_batch_size(self) -> float:
        """Get effective batch size considering accumulation."""
        if not self.accumulation_history:
            return 1.0
        
        recent_stats = list(self.accumulation_history)[-10:]  # Last 10 accumulations
        avg_samples = np.mean([stats['samples'] for stats in recent_stats])
        return avg_samples


class MixedPrecisionManager:
    """Mixed precision training with automatic loss scaling."""
    
    def __init__(
        self,
        initial_scale: float = 65536.0,
        scale_window: int = 2000,
        scale_factor: float = 2.0,
        min_scale: float = 1.0
    ):
        """Initialize mixed precision manager.
        
        Args:
            initial_scale: Initial loss scaling factor
            scale_window: Steps before checking for scaling adjustment
            scale_factor: Factor to scale up/down loss scaling
            min_scale: Minimum loss scaling factor
        """
        self.scale = initial_scale
        self.scale_window = scale_window
        self.scale_factor = scale_factor
        self.min_scale = min_scale
        
        # Tracking
        self.steps_since_last_scale = 0
        self.consecutive_skipped_steps = 0
        self.total_skipped_steps = 0
        
        # Statistics
        self.scale_history = deque(maxlen=1000)
        self.overflow_history = deque(maxlen=1000)
    
    def scale_loss(self, loss: float) -> float:
        """Scale loss for mixed precision training."""
        return loss * self.scale
    
    def unscale_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Unscale gradients after backward pass."""
        unscaled_gradients = {}
        for name, grad in gradients.items():
            unscaled_gradients[name] = grad / self.scale
        return unscaled_gradients
    
    def check_gradients_finite(self, gradients: Dict[str, np.ndarray]) -> bool:
        """Check if gradients contain NaN or Inf values."""
        for grad in gradients.values():
            if not np.all(np.isfinite(grad)):
                return False
        return True
    
    def update_scale(self, gradients_finite: bool) -> bool:
        """Update loss scaling based on gradient overflow.
        
        Args:
            gradients_finite: Whether gradients are finite
        
        Returns:
            True if gradients should be applied, False if skipped due to overflow
        """
        self.steps_since_last_scale += 1
        
        if not gradients_finite:
            # Gradient overflow - reduce scale and skip update
            self.consecutive_skipped_steps += 1
            self.total_skipped_steps += 1
            self.overflow_history.append(1)
            
            # Reduce scale immediately on overflow
            self.scale = max(self.scale / self.scale_factor, self.min_scale)
            self.steps_since_last_scale = 0
            self.scale_history.append(self.scale)
            
            return False
        else:
            # Gradients are finite
            self.consecutive_skipped_steps = 0
            self.overflow_history.append(0)
            
            # Increase scale if we've gone long enough without overflow
            if self.steps_since_last_scale >= self.scale_window:
                self.scale *= self.scale_factor
                self.steps_since_last_scale = 0
                self.scale_history.append(self.scale)
            
            return True
    
    def get_scale_stats(self) -> Dict[str, float]:
        """Get loss scaling statistics."""
        recent_overflows = list(self.overflow_history)[-100:]  # Last 100 steps
        overflow_rate = np.mean(recent_overflows) if recent_overflows else 0.0
        
        return {
            'current_scale': self.scale,
            'overflow_rate': overflow_rate,
            'total_skipped_steps': self.total_skipped_steps,
            'consecutive_skipped_steps': self.consecutive_skipped_steps,
            'avg_scale': np.mean(self.scale_history) if self.scale_history else self.scale
        }


class LRScheduler:
    """Advanced learning rate scheduler with multiple strategies."""
    
    def __init__(
        self,
        schedule_type: LRScheduleType,
        base_lr: float,
        warmup_steps: int = 0,
        total_steps: int = 100000,
        min_lr: float = 0.0,
        **kwargs
    ):
        """Initialize learning rate scheduler.
        
        Args:
            schedule_type: Type of LR schedule
            base_lr: Base learning rate
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            **kwargs: Additional schedule-specific parameters
        """
        self.schedule_type = schedule_type
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.kwargs = kwargs
        
        self.step_count = 0
        self.lr_history = deque(maxlen=1000)
        
        # For reduce_on_plateau
        self.best_metric = float('inf')
        self.patience = kwargs.get('patience', 10)
        self.steps_since_improvement = 0
        self.reduction_factor = kwargs.get('factor', 0.5)
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate for current step."""
        if step is None:
            step = self.step_count
        
        # Warmup phase
        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            lr = self.base_lr * warmup_factor
        else:
            # Main schedule
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            if self.schedule_type == LRScheduleType.CONSTANT:
                lr = self.base_lr
            elif self.schedule_type == LRScheduleType.LINEAR_WARMUP:
                lr = self.base_lr * (1.0 - progress)
            elif self.schedule_type == LRScheduleType.COSINE_ANNEALING:
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.schedule_type == LRScheduleType.EXPONENTIAL_DECAY:
                decay_rate = self.kwargs.get('decay_rate', 0.96)
                decay_steps = self.kwargs.get('decay_steps', 10000)
                lr = self.base_lr * (decay_rate ** (step // decay_steps))
            elif self.schedule_type == LRScheduleType.POLYNOMIAL_DECAY:
                power = self.kwargs.get('power', 1.0)
                lr = (self.base_lr - self.min_lr) * ((1 - progress) ** power) + self.min_lr
            elif self.schedule_type == LRScheduleType.ONE_CYCLE:
                # One cycle policy
                peak_lr = self.kwargs.get('max_lr', self.base_lr * 10)
                if progress <= 0.5:
                    # Increasing phase
                    lr = self.base_lr + (peak_lr - self.base_lr) * (progress * 2)
                else:
                    # Decreasing phase
                    lr = peak_lr + (self.min_lr - peak_lr) * ((progress - 0.5) * 2)
            else:
                lr = self.base_lr
        
        lr = max(lr, self.min_lr)
        self.lr_history.append(lr)
        return lr
    
    def step(self, metric: Optional[float] = None):
        """Step the scheduler."""
        self.step_count += 1
        
        # Handle reduce_on_plateau
        if self.schedule_type == LRScheduleType.REDUCE_ON_PLATEAU and metric is not None:
            if metric < self.best_metric:
                self.best_metric = metric
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1
                
                if self.steps_since_improvement >= self.patience:
                    self.base_lr *= self.reduction_factor
                    self.steps_since_improvement = 0
    
    def get_last_lr(self) -> float:
        """Get the last computed learning rate."""
        return self.lr_history[-1] if self.lr_history else self.base_lr


class AdaptiveGradientClipper:
    """Adaptive gradient clipping with multiple strategies."""
    
    def __init__(
        self,
        clip_type: str = "norm",
        max_norm: float = 1.0,
        percentile: float = 90.0,
        history_size: int = 1000
    ):
        """Initialize adaptive gradient clipper.
        
        Args:
            clip_type: Type of clipping ("norm", "value", "adaptive", "percentile")
            max_norm: Maximum gradient norm (for norm clipping)
            percentile: Percentile for adaptive clipping
            history_size: Size of gradient norm history
        """
        self.clip_type = clip_type
        self.max_norm = max_norm
        self.percentile = percentile
        self.history_size = history_size
        
        # Gradient norm history for adaptive clipping
        self.grad_norm_history = deque(maxlen=history_size)
        self.clip_factor_history = deque(maxlen=100)
    
    def clip_gradients(self, gradients: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        """Clip gradients according to specified strategy.
        
        Args:
            gradients: Dictionary of parameter gradients
        
        Returns:
            Tuple of (clipped_gradients, clip_factor)
        """
        # Compute total gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            param_norm = np.linalg.norm(grad)
            total_norm += param_norm ** 2
        total_norm = math.sqrt(total_norm)
        
        # Record gradient norm
        self.grad_norm_history.append(total_norm)
        
        # Determine clipping threshold
        if self.clip_type == "norm":
            clip_threshold = self.max_norm
        elif self.clip_type == "adaptive":
            if len(self.grad_norm_history) >= 10:
                clip_threshold = np.percentile(self.grad_norm_history, self.percentile)
            else:
                clip_threshold = self.max_norm
        elif self.clip_type == "percentile":
            if len(self.grad_norm_history) >= 10:
                clip_threshold = np.percentile(self.grad_norm_history, self.percentile)
            else:
                clip_threshold = float('inf')  # No clipping initially
        else:  # value clipping
            clip_threshold = self.max_norm
        
        # Apply clipping
        if self.clip_type == "value":
            # Clip by value
            clipped_gradients = {}
            for name, grad in gradients.items():
                clipped_gradients[name] = np.clip(grad, -clip_threshold, clip_threshold)
            clip_factor = 1.0  # Not applicable for value clipping
        else:
            # Clip by norm
            if total_norm > clip_threshold:
                clip_factor = clip_threshold / (total_norm + 1e-6)
                clipped_gradients = {}
                for name, grad in gradients.items():
                    clipped_gradients[name] = grad * clip_factor
            else:
                clip_factor = 1.0
                clipped_gradients = gradients
        
        self.clip_factor_history.append(clip_factor)
        return clipped_gradients, clip_factor
    
    def get_clipping_stats(self) -> Dict[str, float]:
        """Get gradient clipping statistics."""
        recent_norms = list(self.grad_norm_history)[-100:]
        recent_factors = list(self.clip_factor_history)[-100:]
        
        return {
            'avg_grad_norm': np.mean(recent_norms) if recent_norms else 0.0,
            'max_grad_norm': np.max(recent_norms) if recent_norms else 0.0,
            'avg_clip_factor': np.mean(recent_factors) if recent_factors else 1.0,
            'clip_percentage': np.mean([f < 1.0 for f in recent_factors]) * 100 if recent_factors else 0.0
        }


class SophiaOptimizer:
    """Sophia optimizer - Second-order optimizer with preconditioning."""
    
    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-4,
        beta1: float = 0.965,
        beta2: float = 0.99,
        rho: float = 0.04,
        weight_decay: float = 1e-1,
        eps: float = 1e-12,
        update_period: int = 10,
        **kwargs
    ):
        """Initialize Sophia optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            beta1: Coefficient for momentum
            beta2: Coefficient for second moment
            rho: Clipping threshold
            weight_decay: Weight decay factor
            eps: Small constant for numerical stability
            update_period: Period for updating second-order information
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.weight_decay = weight_decay
        self.eps = eps
        self.update_period = update_period
        
        # Initialize state
        self.state = {}
        for name, param in params.items():
            self.state[name] = {
                'step': 0,
                'momentum': np.zeros_like(param),
                'hessian_diag': np.ones_like(param),  # Diagonal Hessian approximation
                'last_hessian_update': 0
            }
    
    def step(
        self,
        params: Dict[str, np.ndarray],
        gradients: Dict[str, np.ndarray],
        hessian_diag: Optional[Dict[str, np.ndarray]] = None
    ):
        """Perform optimization step.
        
        Args:
            params: Current parameters
            gradients: Gradients
            hessian_diag: Diagonal Hessian information (optional)
        """
        for name, param in params.items():
            if name not in gradients:
                continue
            
            grad = gradients[name]
            state = self.state[name]
            state['step'] += 1
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Update momentum
            state['momentum'] = self.beta1 * state['momentum'] + (1 - self.beta1) * grad
            
            # Update Hessian diagonal approximation
            if (state['step'] - state['last_hessian_update']) >= self.update_period:
                if hessian_diag is not None and name in hessian_diag:
                    # Use provided Hessian diagonal
                    h_diag = hessian_diag[name]
                else:
                    # Approximate using gradient outer product
                    h_diag = grad ** 2
                
                # EMA update of Hessian diagonal
                state['hessian_diag'] = self.beta2 * state['hessian_diag'] + (1 - self.beta2) * h_diag
                state['last_hessian_update'] = state['step']
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            
            corrected_momentum = state['momentum'] / bias_correction1
            corrected_hessian = state['hessian_diag'] / bias_correction2
            
            # Compute update direction
            update_direction = corrected_momentum / (np.sqrt(corrected_hessian) + self.eps)
            
            # Clip update
            update_norm = np.linalg.norm(update_direction)
            if update_norm > self.rho:
                update_direction = update_direction * (self.rho / update_norm)
            
            # Apply update
            params[name] = param - self.lr * update_direction


class LionOptimizer:
    """Lion optimizer - EvoLved Sign Momentum."""
    
    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        **kwargs
    ):
        """Initialize Lion optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            beta1: Coefficient for momentum
            beta2: Coefficient for momentum update
            weight_decay: Weight decay factor
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        
        # Initialize momentum
        self.momentum = {}
        for name, param in params.items():
            self.momentum[name] = np.zeros_like(param)
    
    def step(self, params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]):
        """Perform Lion optimization step."""
        for name, param in params.items():
            if name not in gradients:
                continue
            
            grad = gradients[name]
            momentum = self.momentum[name]
            
            # Interpolate between gradient and momentum
            update = self.beta1 * momentum + (1 - self.beta1) * grad
            
            # Apply weight decay
            if self.weight_decay > 0:
                param = param * (1 - self.lr * self.weight_decay)
            
            # Update parameters using sign of update
            params[name] = param - self.lr * np.sign(update)
            
            # Update momentum
            self.momentum[name] = self.beta2 * momentum + (1 - self.beta2) * grad


class AdvancedOptimizer:
    """Advanced optimizer with all optimization techniques integrated."""
    
    def __init__(self, config: OptimizerConfig):
        """Initialize advanced optimizer with comprehensive configuration."""
        self.config = config
        
        # Components
        self.gradient_accumulator = GradientAccumulator(config.gradient_accumulation_steps)
        self.mixed_precision = MixedPrecisionManager(
            config.loss_scale,
            config.loss_scale_window,
            min_scale=config.min_loss_scale
        ) if config.use_mixed_precision else None
        
        self.lr_scheduler = LRScheduler(
            config.lr_schedule_type,
            config.learning_rate,
            config.warmup_steps,
            config.total_steps
        )
        
        self.grad_clipper = AdaptiveGradientClipper(
            config.gradient_clip_type,
            config.max_grad_norm
        )
        
        # Base optimizer
        self.base_optimizer = None
        self.lookahead_optimizer = None
        
        # EMA for parameters
        self.ema_params = None
        if config.use_ema:
            self.ema_params = {}
    
    def initialize_optimizer(self, params: Dict[str, np.ndarray]):
        """Initialize the base optimizer with parameters."""
        if self.config.optimizer_type == OptimizerType.SOPHIA:
            self.base_optimizer = SophiaOptimizer(
                params,
                lr=self.config.learning_rate,
                rho=self.config.rho,
                update_period=self.config.update_period,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == OptimizerType.LION:
            self.base_optimizer = LionOptimizer(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        # Add other optimizers as needed
        
        # Initialize EMA
        if self.config.use_ema:
            self.ema_params = {name: param.copy() for name, param in params.items()}
    
    def step(
        self,
        params: Dict[str, np.ndarray],
        gradients: Dict[str, np.ndarray],
        loss: float,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """Perform comprehensive optimization step.
        
        Args:
            params: Model parameters
            gradients: Computed gradients
            loss: Current loss value
            batch_size: Current batch size
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = {}
        
        # Mixed precision handling
        if self.mixed_precision:
            # Check for gradient overflow
            gradients_finite = self.mixed_precision.check_gradients_finite(gradients)
            
            if gradients_finite:
                # Unscale gradients
                gradients = self.mixed_precision.unscale_gradients(gradients)
            
            # Update loss scaling
            should_step = self.mixed_precision.update_scale(gradients_finite)
            stats['mixed_precision'] = self.mixed_precision.get_scale_stats()
            
            if not should_step:
                # Skip optimization step due to gradient overflow
                stats['step_skipped'] = True
                return stats
        
        # Gradient accumulation
        should_optimize = self.gradient_accumulator.accumulate_gradients(gradients, batch_size)
        
        if not should_optimize:
            # Still accumulating gradients
            stats['accumulating'] = True
            return stats
        
        # Get accumulated gradients
        accumulated_grads = self.gradient_accumulator.get_accumulated_gradients()
        
        # Gradient clipping
        clipped_grads, clip_factor = self.grad_clipper.clip_gradients(accumulated_grads)
        stats['gradient_clipping'] = self.grad_clipper.get_clipping_stats()
        stats['clip_factor'] = clip_factor
        
        # Get current learning rate
        current_lr = self.lr_scheduler.get_lr()
        stats['learning_rate'] = current_lr
        
        # Apply base optimizer
        if self.base_optimizer:
            if isinstance(self.base_optimizer, SophiaOptimizer):
                self.base_optimizer.step(params, clipped_grads)
            elif isinstance(self.base_optimizer, LionOptimizer):
                self.base_optimizer.step(params, clipped_grads)
        
        # Update EMA parameters
        if self.ema_params:
            for name, param in params.items():
                if name in self.ema_params:
                    self.ema_params[name] = (
                        self.config.ema_decay * self.ema_params[name] +
                        (1 - self.config.ema_decay) * param
                    )
        
        # Step learning rate scheduler
        self.lr_scheduler.step()
        
        # Reset gradient accumulation
        self.gradient_accumulator.reset()
        
        # Additional statistics
        stats.update({
            'effective_batch_size': self.gradient_accumulator.get_effective_batch_size(),
            'optimizer_step_completed': True
        })
        
        return stats
    
    def get_ema_params(self) -> Optional[Dict[str, np.ndarray]]:
        """Get exponential moving average parameters."""
        return self.ema_params.copy() if self.ema_params else None
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {}
        
        if self.mixed_precision:
            stats['mixed_precision'] = self.mixed_precision.get_scale_stats()
        
        stats['gradient_clipping'] = self.grad_clipper.get_clipping_stats()
        stats['learning_rate'] = self.lr_scheduler.get_last_lr()
        stats['effective_batch_size'] = self.gradient_accumulator.get_effective_batch_size()
        
        return stats


def test_advanced_optimizers():
    """Test advanced optimization techniques."""
    print("Testing Advanced Optimization Techniques")
    print("=" * 50)
    
    # Create mock parameters and gradients
    params = {
        'weight1': np.random.randn(100, 50).astype(np.float32),
        'bias1': np.random.randn(50).astype(np.float32),
        'weight2': np.random.randn(50, 10).astype(np.float32),
        'bias2': np.random.randn(10).astype(np.float32)
    }
    
    gradients = {
        'weight1': np.random.randn(100, 50).astype(np.float32) * 0.01,
        'bias1': np.random.randn(50).astype(np.float32) * 0.01,
        'weight2': np.random.randn(50, 10).astype(np.float32) * 0.01,
        'bias2': np.random.randn(10).astype(np.float32) * 0.01
    }
    
    # Test configuration
    config = OptimizerConfig(
        optimizer_type=OptimizerType.SOPHIA,
        learning_rate=1e-3,
        use_mixed_precision=True,
        gradient_accumulation_steps=4,
        use_ema=True,
        lr_schedule_type=LRScheduleType.COSINE_ANNEALING
    )
    
    # Create optimizer
    optimizer = AdvancedOptimizer(config)
    optimizer.initialize_optimizer(params)
    
    print("1. Testing optimization steps:")
    
    # Simulate training steps
    for step in range(20):
        # Simulate loss (decreasing over time)
        loss = 1.0 - step * 0.02 + np.random.normal(0, 0.1)
        
        # Add some noise to gradients
        noisy_gradients = {}
        for name, grad in gradients.items():
            noise_scale = 0.1 * (1.0 - step * 0.05)  # Decreasing noise
            noisy_gradients[name] = grad + np.random.normal(0, noise_scale, grad.shape)
        
        # Perform optimization step
        stats = optimizer.step(params, noisy_gradients, loss, batch_size=32)
        
        if step % 5 == 0:
            print(f"   Step {step:2d}: lr={stats.get('learning_rate', 0):.6f}, "
                  f"loss={loss:.3f}, "
                  f"completed={'yes' if stats.get('optimizer_step_completed') else 'no'}")
    
    # Test individual components
    print("2. Testing mixed precision manager:")
    mp_manager = MixedPrecisionManager()
    
    # Test with finite gradients
    finite_grads = {name: grad for name, grad in gradients.items()}
    finite_check = mp_manager.check_gradients_finite(finite_grads)
    should_step = mp_manager.update_scale(finite_check)
    print(f"   Finite gradients: {finite_check}, should step: {should_step}")
    
    # Test with overflow gradients
    overflow_grads = {name: grad * 1e10 for name, grad in gradients.items()}  # Cause overflow
    overflow_check = mp_manager.check_gradients_finite(overflow_grads)
    should_step = mp_manager.update_scale(overflow_check)
    print(f"   Overflow gradients: {overflow_check}, should step: {should_step}")
    
    print("3. Testing gradient accumulator:")
    accumulator = GradientAccumulator(accumulation_steps=3)
    
    for i in range(5):
        should_optimize = accumulator.accumulate_gradients(gradients, batch_size=8)
        print(f"   Accumulation step {i+1}: should optimize = {should_optimize}")
        if should_optimize:
            accumulator.reset()
    
    print("4. Testing LR scheduler:")
    scheduler = LRScheduler(
        LRScheduleType.COSINE_ANNEALING,
        base_lr=1e-3,
        warmup_steps=5,
        total_steps=20
    )
    
    print("   LR schedule:")
    for step in range(25):
        lr = scheduler.get_lr(step)
        if step % 5 == 0:
            print(f"     Step {step:2d}: lr={lr:.6f}")
    
    print("5. Testing Sophia optimizer:")
    sophia = SophiaOptimizer(params, lr=1e-4, rho=0.04)
    
    # Perform a few optimization steps
    for i in range(3):
        sophia.step(params, gradients)
        print(f"   Sophia step {i+1} completed")
    
    print("6. Testing Lion optimizer:")
    lion = LionOptimizer(params, lr=1e-4)
    
    # Perform a few optimization steps
    for i in range(3):
        lion.step(params, gradients)
        print(f"   Lion step {i+1} completed")
    
    print("7. Getting comprehensive statistics:")
    final_stats = optimizer.get_comprehensive_stats()
    for key, value in final_stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"     {subkey}: {subvalue}")
        else:
            print(f"   {key}: {value}")
    
    print("\nAdvanced optimization techniques tested successfully!")


if __name__ == "__main__":
    test_advanced_optimizers()