"""Advanced gradient scaling utilities for mixed precision training.

This module provides enterprise-grade gradient scaling functionality with:
- Dynamic loss scaling with overflow detection
- Gradient clipping integration
- Multiple scaling strategies
- Optimizer integration helpers
- Comprehensive monitoring and statistics
"""

import logging
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.base import Optimizer, Parameter
from ..core.tensor import Tensor
from ..exceptions import NumericalError

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Different scaling strategies for gradient scaling."""
    
    DYNAMIC = "dynamic"  # Dynamic scaling with growth/backoff
    FIXED = "fixed"  # Fixed scaling factor
    ADAPTIVE = "adaptive"  # Adaptive based on gradient statistics
    CONSERVATIVE = "conservative"  # Conservative with slower growth


@dataclass
class ScalerConfig:
    """Configuration for gradient scaler."""
    
    init_scale: float = 65536.0  # Initial loss scale (2^16)
    growth_factor: float = 2.0  # Factor to increase loss scale
    backoff_factor: float = 0.5  # Factor to decrease loss scale
    growth_interval: int = 2000  # Steps between loss scale increases
    max_loss_scale: float = 2**24  # Maximum loss scale
    min_loss_scale: float = 1.0  # Minimum loss scale
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    enabled: bool = True
    # Advanced settings
    consecutive_success_threshold: int = 2000  # For conservative strategy
    overflow_cooldown: int = 500  # Steps to wait after overflow before growing
    gradient_clip_threshold: float = 1.0  # Gradient clipping threshold
    stability_check_interval: int = 100  # How often to check gradient stability


class AdvancedGradScaler:
    """Advanced gradient scaler with multiple strategies and monitoring."""
    
    def __init__(self, config: Optional[ScalerConfig] = None):
        """Initialize advanced gradient scaler.
        
        Args:
            config: Scaler configuration
        """
        self.config = config or ScalerConfig()
        
        # Core state
        self._scale = self.config.init_scale
        self._growth_tracker = 0
        self._overflow_tracker = 0
        self._found_inf = False
        self._last_overflow_step = -1
        self._step_count = 0
        
        # Statistics
        self._total_overflows = 0
        self._total_successful_steps = 0
        self._gradient_norms_history = []
        self._scale_history = []
        
        # Performance tracking
        self._timing_stats = {
            "unscale_time": 0.0,
            "overflow_check_time": 0.0,
            "step_time": 0.0
        }
        
        logger.info(f"Advanced GradScaler initialized: strategy={self.config.strategy.value}, "
                   f"init_scale={self.config.init_scale}")
    
    def scale(self, loss: Tensor) -> Tensor:
        """Scale loss to prevent gradient underflow.
        
        Args:
            loss: Loss tensor to scale
            
        Returns:
            Scaled loss tensor
            
        Raises:
            TypeError: If loss is not a Tensor
            NumericalError: If loss contains invalid values
        """
        if not isinstance(loss, Tensor):
            raise TypeError(f"Expected Tensor, got {type(loss)}")
        
        if not self.config.enabled:
            return loss
        
        # Check for invalid values in loss
        if not np.all(np.isfinite(loss.data)):
            raise NumericalError("Loss contains non-finite values before scaling")
        
        scaled_data = loss.data * self._scale
        
        # Check for overflow in scaled loss
        if not np.all(np.isfinite(scaled_data)):
            logger.warning(f"Overflow detected in scaled loss with scale {self._scale}")
            # Reduce scale and try again
            self._scale = max(self._scale * self.config.backoff_factor, self.config.min_loss_scale)
            scaled_data = loss.data * self._scale
        
        scaled_loss = Tensor(
            scaled_data,
            requires_grad=loss.requires_grad,
            name=f"scaled_{loss.name or 'loss'}"
        )
        
        logger.debug(f"Scaled loss by factor {self._scale}")
        return scaled_loss
    
    def unscale_(self, optimizer: Optimizer) -> bool:
        """Unscale gradients in-place and return whether gradients are finite.
        
        Args:
            optimizer: Optimizer containing parameters to unscale
            
        Returns:
            True if all gradients are finite, False otherwise
            
        Raises:
            ValueError: If optimizer doesn't have parameters
        """
        start_time = time.time()
        
        if not self.config.enabled:
            return True
        
        self._found_inf = False
        gradient_norms = []
        
        # Get parameters from optimizer
        if hasattr(optimizer, "parameters"):
            params = optimizer.parameters
            if hasattr(params, "values"):
                param_list = list(params.values())
            else:
                param_list = list(params)
        else:
            raise ValueError("Optimizer must have parameters attribute")
        
        # Unscale gradients for all parameters
        for param in param_list:
            if hasattr(param, "grad") and param.grad is not None:
                # Unscale gradient
                unscaled_grad = param.grad.data / self._scale
                
                # Check for inf/nan
                if not np.all(np.isfinite(unscaled_grad)):
                    self._found_inf = True
                    logger.warning("Found inf/nan in gradients during unscaling")
                    break
                
                # Apply gradient clipping if configured
                if self.config.gradient_clip_threshold > 0:
                    grad_norm = np.linalg.norm(unscaled_grad)
                    gradient_norms.append(grad_norm)
                    
                    if grad_norm > self.config.gradient_clip_threshold:
                        clip_coef = self.config.gradient_clip_threshold / (grad_norm + 1e-6)
                        unscaled_grad = unscaled_grad * clip_coef
                        logger.debug(f"Clipped gradient with norm {grad_norm:.4f}")
                
                # Update gradient in place
                param.grad.data = unscaled_grad
        
        # Track gradient statistics
        if gradient_norms and len(gradient_norms) > 0:
            avg_grad_norm = np.mean(gradient_norms)
            self._gradient_norms_history.append(avg_grad_norm)
            
            # Keep only recent history
            if len(self._gradient_norms_history) > 1000:
                self._gradient_norms_history = self._gradient_norms_history[-1000:]
        
        self._timing_stats["unscale_time"] += time.time() - start_time
        return not self._found_inf
    
    def step(self, optimizer: Optimizer) -> bool:
        """Step optimizer if gradients are finite and update loss scale.
        
        Args:
            optimizer: Optimizer to step
            
        Returns:
            True if optimizer step was taken, False if skipped due to inf/nan
        """
        start_time = time.time()
        self._step_count += 1
        
        if not self.config.enabled:
            optimizer.step()
            self._total_successful_steps += 1
            return True
        
        # Unscale gradients and check for inf/nan
        grad_finite = self.unscale_(optimizer)
        
        if grad_finite:
            # Take optimizer step
            optimizer.step()
            self._total_successful_steps += 1
            self._growth_tracker += 1
            
            # Update loss scale based on strategy
            self._update_scale_on_success()
            
            step_taken = True
        else:
            # Skip optimizer step and handle overflow
            self._handle_overflow()
            step_taken = False
        
        # Track scale history
        self._scale_history.append(self._scale)
        if len(self._scale_history) > 1000:
            self._scale_history = self._scale_history[-1000:]
        
        # Periodic stability check
        if self._step_count % self.config.stability_check_interval == 0:
            self._check_gradient_stability()
        
        self._timing_stats["step_time"] += time.time() - start_time
        return step_taken
    
    def _update_scale_on_success(self):
        """Update loss scale after successful step based on strategy."""
        if self.config.strategy == ScalingStrategy.DYNAMIC:
            if self._growth_tracker >= self.config.growth_interval:
                self._scale = min(
                    self._scale * self.config.growth_factor,
                    self.config.max_loss_scale
                )
                self._growth_tracker = 0
                logger.debug(f"Increased loss scale to {self._scale}")
        
        elif self.config.strategy == ScalingStrategy.CONSERVATIVE:
            # Only grow if we've had many consecutive successes and no recent overflows
            steps_since_overflow = self._step_count - self._last_overflow_step
            if (self._growth_tracker >= self.config.consecutive_success_threshold and
                steps_since_overflow > self.config.overflow_cooldown):
                self._scale = min(
                    self._scale * self.config.growth_factor,
                    self.config.max_loss_scale
                )
                self._growth_tracker = 0
                logger.debug(f"Conservatively increased loss scale to {self._scale}")
        
        elif self.config.strategy == ScalingStrategy.ADAPTIVE:
            # Adaptive scaling based on gradient norm statistics
            if len(self._gradient_norms_history) >= 100:
                recent_norms = self._gradient_norms_history[-100:]
                avg_norm = np.mean(recent_norms)
                std_norm = np.std(recent_norms)
                
                # If gradients are stable and small, consider growing scale
                if std_norm < 0.1 * avg_norm and avg_norm < 0.1:
                    if self._growth_tracker >= self.config.growth_interval // 2:
                        self._scale = min(
                            self._scale * 1.5,  # More conservative growth
                            self.config.max_loss_scale
                        )
                        self._growth_tracker = 0
                        logger.debug(f"Adaptively increased loss scale to {self._scale}")
        
        # FIXED strategy doesn't change scale
    
    def _handle_overflow(self):
        """Handle gradient overflow by reducing scale."""
        self._total_overflows += 1
        self._last_overflow_step = self._step_count
        self._growth_tracker = 0
        
        # Reduce scale
        old_scale = self._scale
        self._scale = max(
            self._scale * self.config.backoff_factor,
            self.config.min_loss_scale
        )
        
        logger.warning(f"Gradient overflow detected. Reduced scale from {old_scale} to {self._scale}")
        
        # Additional measures for repeated overflows
        if self._total_overflows % 10 == 0:
            logger.warning(f"Frequent overflows detected ({self._total_overflows} total). "
                          f"Consider adjusting model or learning rate.")
    
    def _check_gradient_stability(self):
        """Check gradient stability and adjust strategy if needed."""
        if len(self._gradient_norms_history) < 50:
            return
        
        recent_norms = self._gradient_norms_history[-50:]
        stability_ratio = np.std(recent_norms) / (np.mean(recent_norms) + 1e-8)
        
        if stability_ratio > 2.0:  # High instability
            logger.warning(f"High gradient instability detected (ratio: {stability_ratio:.2f}). "
                          f"Consider reducing learning rate or using more conservative scaling.")
    
    def update(self):
        """Update scaler state (for compatibility with PyTorch-style API)."""
        # This method is automatically called by step(), but provided for compatibility
        pass
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        return self._scale
    
    def set_scale(self, scale: float):
        """Set loss scale manually.
        
        Args:
            scale: New loss scale value
        """
        self._scale = max(scale, self.config.min_loss_scale)
        logger.debug(f"Manually set loss scale to {self._scale}")
    
    def get_growth_tracker(self) -> int:
        """Get current growth tracker value."""
        return self._growth_tracker
    
    def is_enabled(self) -> bool:
        """Check if scaling is enabled."""
        return self.config.enabled
    
    def enable(self):
        """Enable gradient scaling."""
        self.config.enabled = True
        logger.info("Gradient scaling enabled")
    
    def disable(self):
        """Disable gradient scaling."""
        self.config.enabled = False
        logger.info("Gradient scaling disabled")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaler statistics.
        
        Returns:
            Dictionary with scaler statistics
        """
        stats = {
            "current_scale": self._scale,
            "total_steps": self._step_count,
            "successful_steps": self._total_successful_steps,
            "total_overflows": self._total_overflows,
            "overflow_rate": self._total_overflows / max(self._step_count, 1),
            "success_rate": self._total_successful_steps / max(self._step_count, 1),
            "growth_tracker": self._growth_tracker,
            "strategy": self.config.strategy.value,
            "enabled": self.config.enabled,
        }
        
        # Add gradient statistics if available
        if self._gradient_norms_history:
            recent_norms = self._gradient_norms_history[-100:]
            stats.update({
                "avg_gradient_norm": np.mean(recent_norms),
                "gradient_norm_std": np.std(recent_norms),
                "max_gradient_norm": np.max(recent_norms),
                "min_gradient_norm": np.min(recent_norms),
            })
        
        # Add scale statistics
        if self._scale_history:
            stats.update({
                "scale_history_length": len(self._scale_history),
                "max_scale_used": np.max(self._scale_history),
                "min_scale_used": np.min(self._scale_history),
                "scale_changes": len(set(self._scale_history)),
            })
        
        # Add timing statistics
        if self._step_count > 0:
            stats.update({
                "avg_unscale_time": self._timing_stats["unscale_time"] / self._step_count,
                "avg_step_time": self._timing_stats["step_time"] / self._step_count,
                "total_overhead_time": sum(self._timing_stats.values()),
            })
        
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing.
        
        Returns:
            State dictionary containing all necessary state
        """
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "overflow_tracker": self._overflow_tracker,
            "step_count": self._step_count,
            "total_overflows": self._total_overflows,
            "total_successful_steps": self._total_successful_steps,
            "last_overflow_step": self._last_overflow_step,
            "gradient_norms_history": self._gradient_norms_history[-100:],  # Keep recent history
            "scale_history": self._scale_history[-100:],
            "timing_stats": self._timing_stats.copy(),
            "config": {
                "init_scale": self.config.init_scale,
                "growth_factor": self.config.growth_factor,
                "backoff_factor": self.config.backoff_factor,
                "growth_interval": self.config.growth_interval,
                "max_loss_scale": self.config.max_loss_scale,
                "min_loss_scale": self.config.min_loss_scale,
                "strategy": self.config.strategy.value,
                "enabled": self.config.enabled,
                "consecutive_success_threshold": self.config.consecutive_success_threshold,
                "overflow_cooldown": self.config.overflow_cooldown,
                "gradient_clip_threshold": self.config.gradient_clip_threshold,
                "stability_check_interval": self.config.stability_check_interval,
            }
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from checkpoint.
        
        Args:
            state_dict: State dictionary to load
        """
        self._scale = state_dict["scale"]
        self._growth_tracker = state_dict["growth_tracker"]
        self._overflow_tracker = state_dict.get("overflow_tracker", 0)
        self._step_count = state_dict["step_count"]
        self._total_overflows = state_dict["total_overflows"]
        self._total_successful_steps = state_dict["total_successful_steps"]
        self._last_overflow_step = state_dict.get("last_overflow_step", -1)
        
        # Load history data
        self._gradient_norms_history = state_dict.get("gradient_norms_history", [])
        self._scale_history = state_dict.get("scale_history", [])
        self._timing_stats = state_dict.get("timing_stats", {
            "unscale_time": 0.0,
            "overflow_check_time": 0.0,
            "step_time": 0.0
        })
        
        # Load config if provided
        if "config" in state_dict:
            config_dict = state_dict["config"]
            strategy = ScalingStrategy(config_dict.get("strategy", "dynamic"))
            self.config = ScalerConfig(
                init_scale=config_dict.get("init_scale", 65536.0),
                growth_factor=config_dict.get("growth_factor", 2.0),
                backoff_factor=config_dict.get("backoff_factor", 0.5),
                growth_interval=config_dict.get("growth_interval", 2000),
                max_loss_scale=config_dict.get("max_loss_scale", 2**24),
                min_loss_scale=config_dict.get("min_loss_scale", 1.0),
                strategy=strategy,
                enabled=config_dict.get("enabled", True),
                consecutive_success_threshold=config_dict.get("consecutive_success_threshold", 2000),
                overflow_cooldown=config_dict.get("overflow_cooldown", 500),
                gradient_clip_threshold=config_dict.get("gradient_clip_threshold", 1.0),
                stability_check_interval=config_dict.get("stability_check_interval", 100),
            )
        
        logger.info("Loaded gradient scaler state from checkpoint")
    
    def reset(self):
        """Reset scaler to initial state."""
        self._scale = self.config.init_scale
        self._growth_tracker = 0
        self._overflow_tracker = 0
        self._found_inf = False
        self._last_overflow_step = -1
        self._step_count = 0
        self._total_overflows = 0
        self._total_successful_steps = 0
        self._gradient_norms_history = []
        self._scale_history = []
        self._timing_stats = {
            "unscale_time": 0.0,
            "overflow_check_time": 0.0,
            "step_time": 0.0
        }
        logger.info("Reset gradient scaler to initial state")


# Utility functions for gradient scaling

def check_gradients_finite(optimizer: Optimizer) -> Tuple[bool, Dict[str, float]]:
    """Check if all gradients are finite and return statistics.
    
    Args:
        optimizer: Optimizer to check gradients for
        
    Returns:
        Tuple of (all_finite, gradient_stats)
    """
    gradient_stats = {
        "total_params": 0,
        "params_with_grad": 0,
        "finite_grads": 0,
        "infinite_grads": 0,
        "nan_grads": 0,
        "max_grad_norm": 0.0,
        "avg_grad_norm": 0.0,
    }
    
    grad_norms = []
    all_finite = True
    
    # Get parameters
    if hasattr(optimizer, "parameters"):
        params = optimizer.parameters
        if hasattr(params, "values"):
            param_list = list(params.values())
        else:
            param_list = list(params)
    else:
        return False, gradient_stats
    
    for param in param_list:
        gradient_stats["total_params"] += 1
        
        if hasattr(param, "grad") and param.grad is not None:
            gradient_stats["params_with_grad"] += 1
            grad_data = param.grad.data
            
            # Check for inf/nan
            if np.any(np.isinf(grad_data)):
                gradient_stats["infinite_grads"] += 1
                all_finite = False
            elif np.any(np.isnan(grad_data)):
                gradient_stats["nan_grads"] += 1
                all_finite = False
            else:
                gradient_stats["finite_grads"] += 1
                grad_norm = np.linalg.norm(grad_data)
                grad_norms.append(grad_norm)
    
    # Calculate gradient norm statistics
    if grad_norms:
        gradient_stats["max_grad_norm"] = float(np.max(grad_norms))
        gradient_stats["avg_grad_norm"] = float(np.mean(grad_norms))
    
    return all_finite, gradient_stats


def clip_gradients_by_norm(optimizer: Optimizer, max_norm: float) -> float:
    """Clip gradients by global norm.
    
    Args:
        optimizer: Optimizer containing parameters to clip
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    # Get parameters
    if hasattr(optimizer, "parameters"):
        params = optimizer.parameters
        if hasattr(params, "values"):
            param_list = list(params.values())
        else:
            param_list = list(params)
    else:
        return 0.0
    
    # Calculate total gradient norm
    total_norm = 0.0
    for param in param_list:
        if hasattr(param, "grad") and param.grad is not None:
            param_norm = np.linalg.norm(param.grad.data)
            total_norm += param_norm ** 2
    
    total_norm = np.sqrt(total_norm)
    
    # Clip gradients if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for param in param_list:
            if hasattr(param, "grad") and param.grad is not None:
                param.grad.data = param.grad.data * clip_coef
    
    return float(total_norm)


def scale_gradients(optimizer: Optimizer, scale_factor: float):
    """Scale all gradients by a factor.
    
    Args:
        optimizer: Optimizer containing parameters to scale
        scale_factor: Factor to scale gradients by
    """
    # Get parameters
    if hasattr(optimizer, "parameters"):
        params = optimizer.parameters
        if hasattr(params, "values"):
            param_list = list(params.values())
        else:
            param_list = list(params)
    else:
        return
    
    for param in param_list:
        if hasattr(param, "grad") and param.grad is not None:
            param.grad.data = param.grad.data * scale_factor


def create_scaler(strategy: str = "dynamic", **kwargs) -> AdvancedGradScaler:
    """Create a gradient scaler with the specified strategy.
    
    Args:
        strategy: Scaling strategy ("dynamic", "fixed", "adaptive", "conservative")
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured AdvancedGradScaler instance
    """
    try:
        strategy_enum = ScalingStrategy(strategy.lower())
    except ValueError:
        logger.warning(f"Unknown strategy '{strategy}', using 'dynamic'")
        strategy_enum = ScalingStrategy.DYNAMIC
    
    config = ScalerConfig(strategy=strategy_enum, **kwargs)
    return AdvancedGradScaler(config)