"""
Comprehensive test suite for advanced optimization techniques.
Tests all components of advanced_optimizers.py for comprehensive coverage.

This module tests:
- GradientAccumulator with dynamic batching
- MixedPrecisionManager with automatic loss scaling
- LRScheduler with multiple scheduling strategies
- AdaptiveGradientClipper with various clipping methods
- SophiaOptimizer (second-order optimization)
- LionOptimizer (evolved sign momentum)
- AdvancedOptimizer (integrated optimization system)
- All enum classes and configuration
"""

import math
import tempfile
from collections import deque
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import numpy as np
import pytest

from src.neural_arch.optimization.advanced_optimizers import (
    OptimizerType,
    LRScheduleType,
    OptimizerConfig,
    GradientAccumulator,
    MixedPrecisionManager,
    LRScheduler,
    AdaptiveGradientClipper,
    SophiaOptimizer,
    LionOptimizer,
    AdvancedOptimizer,
    test_advanced_optimizers
)


class TestOptimizerType:
    """Test OptimizerType enumeration."""
    
    def test_optimizer_types(self):
        """Test all optimizer type values."""
        assert OptimizerType.ADAM.value == "adam"
        assert OptimizerType.ADAMW.value == "adamw"
        assert OptimizerType.LION.value == "lion"
        assert OptimizerType.SOPHIA.value == "sophia"
        assert OptimizerType.ADAFACTOR.value == "adafactor"
        assert OptimizerType.LAMB.value == "lamb"
        assert OptimizerType.RADAM.value == "radam"
        assert OptimizerType.LOOKAHEAD.value == "lookahead"
        assert OptimizerType.SHAMPOO.value == "shampoo"
    
    def test_optimizer_type_count(self):
        """Test total number of optimizer types."""
        assert len(OptimizerType) == 9


class TestLRScheduleType:
    """Test LRScheduleType enumeration."""
    
    def test_lr_schedule_types(self):
        """Test all LR schedule type values."""
        assert LRScheduleType.CONSTANT.value == "constant"
        assert LRScheduleType.LINEAR_WARMUP.value == "linear_warmup"
        assert LRScheduleType.COSINE_ANNEALING.value == "cosine_annealing"
        assert LRScheduleType.EXPONENTIAL_DECAY.value == "exponential_decay"
        assert LRScheduleType.POLYNOMIAL_DECAY.value == "polynomial_decay"
        assert LRScheduleType.ONE_CYCLE.value == "one_cycle"
        assert LRScheduleType.REDUCE_ON_PLATEAU.value == "reduce_on_plateau"
    
    def test_lr_schedule_type_count(self):
        """Test total number of LR schedule types."""
        assert len(LRScheduleType) == 7


class TestOptimizerConfig:
    """Test OptimizerConfig configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizerConfig()
        
        # Basic optimizer settings
        assert config.optimizer_type == OptimizerType.ADAMW
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.01
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
        assert config.epsilon == 1e-8
        
        # Gradient clipping
        assert config.max_grad_norm == 1.0
        assert config.gradient_clip_type == "norm"
        
        # Mixed precision
        assert config.use_mixed_precision is True
        assert config.loss_scale == 65536.0
        assert config.loss_scale_window == 2000
        assert config.min_loss_scale == 1.0
        
        # Gradient accumulation
        assert config.gradient_accumulation_steps == 1
        
        # Learning rate schedule
        assert config.lr_schedule_type == LRScheduleType.COSINE_ANNEALING
        assert config.warmup_steps == 1000
        assert config.total_steps == 100000
        
        # Advanced features
        assert config.use_lookahead is False
        assert config.lookahead_alpha == 0.5
        assert config.lookahead_k == 5
        assert config.use_ema is False
        assert config.ema_decay == 0.9999
        
        # Sophia-specific
        assert config.rho == 0.04
        assert config.update_period == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOPHIA,
            learning_rate=1e-4,
            weight_decay=0.1,
            beta1=0.95,
            beta2=0.9999,
            epsilon=1e-10,
            max_grad_norm=2.0,
            gradient_clip_type="adaptive",
            use_mixed_precision=False,
            loss_scale=32768.0,
            gradient_accumulation_steps=4,
            lr_schedule_type=LRScheduleType.ONE_CYCLE,
            warmup_steps=500,
            total_steps=50000,
            use_lookahead=True,
            lookahead_alpha=0.7,
            lookahead_k=3,
            use_ema=True,
            ema_decay=0.999,
            rho=0.02,
            update_period=5
        )
        
        assert config.optimizer_type == OptimizerType.SOPHIA
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.1
        assert config.beta1 == 0.95
        assert config.beta2 == 0.9999
        assert config.epsilon == 1e-10
        assert config.max_grad_norm == 2.0
        assert config.gradient_clip_type == "adaptive"
        assert config.use_mixed_precision is False
        assert config.loss_scale == 32768.0
        assert config.gradient_accumulation_steps == 4
        assert config.lr_schedule_type == LRScheduleType.ONE_CYCLE
        assert config.warmup_steps == 500
        assert config.total_steps == 50000
        assert config.use_lookahead is True
        assert config.lookahead_alpha == 0.7
        assert config.lookahead_k == 3
        assert config.use_ema is True
        assert config.ema_decay == 0.999
        assert config.rho == 0.02
        assert config.update_period == 5


class TestGradientAccumulator:
    """Test gradient accumulation with dynamic batching."""
    
    def setup_method(self):
        """Setup test environment."""
        self.accumulator = GradientAccumulator(accumulation_steps=4, sync_gradients=True)
        self.gradients = {
            'layer1.weight': np.random.randn(10, 5),
            'layer1.bias': np.random.randn(5),
            'layer2.weight': np.random.randn(5, 3)
        }
    
    def test_initialization(self):
        """Test gradient accumulator initialization."""
        assert self.accumulator.accumulation_steps == 4
        assert self.accumulator.sync_gradients is True
        assert len(self.accumulator.accumulated_gradients) == 0
        assert self.accumulator.current_step == 0
        assert self.accumulator.total_accumulated_samples == 0
        assert len(self.accumulator.accumulation_history) == 0
    
    def test_gradient_accumulation_sequence(self):
        """Test complete gradient accumulation sequence."""
        batch_sizes = [8, 12, 16, 10]  # Variable batch sizes
        
        for i, batch_size in enumerate(batch_sizes):
            should_step = self.accumulator.accumulate_gradients(self.gradients, batch_size)
            
            # First 3 steps should not trigger optimization
            if i < 3:
                assert should_step is False
                assert self.accumulator.current_step == i + 1
                assert self.accumulator.total_accumulated_samples == sum(batch_sizes[:i+1])
            else:
                # 4th step should trigger optimization
                assert should_step is True
                assert len(self.accumulator.accumulation_history) == 1
    
    def test_accumulated_gradients_averaging(self):
        """Test proper gradient averaging during accumulation."""
        # Use consistent gradients for testing averaging
        test_gradients = {
            'param1': np.ones((2, 2)),
            'param2': np.ones(3) * 2
        }
        
        accumulator = GradientAccumulator(accumulation_steps=2)
        
        # First accumulation
        should_step = accumulator.accumulate_gradients(test_gradients, batch_size=4)
        assert should_step is False
        
        # Second accumulation (should trigger averaging)
        should_step = accumulator.accumulate_gradients(test_gradients, batch_size=6)
        assert should_step is True
        
        # Check averaged gradients
        accumulated = accumulator.get_accumulated_gradients()
        
        # Should be averaged over total samples (4 + 6 = 10)
        # Original grad * batch_size / total_samples = grad * batch_size / 10
        # For param1: 1 * 4 / 10 + 1 * 6 / 10 = 1.0
        # For param2: 2 * 4 / 10 + 2 * 6 / 10 = 2.0
        np.testing.assert_array_equal(accumulated['param1'], np.ones((2, 2)))
        np.testing.assert_array_equal(accumulated['param2'], np.ones(3) * 2)
    
    def test_reset_functionality(self):
        """Test accumulator reset."""
        # Accumulate some gradients
        self.accumulator.accumulate_gradients(self.gradients, batch_size=8)
        
        # Verify state before reset
        assert self.accumulator.current_step == 1
        assert self.accumulator.total_accumulated_samples == 8
        assert len(self.accumulator.accumulated_gradients) > 0
        
        # Reset
        self.accumulator.reset()
        
        # Verify state after reset
        assert self.accumulator.current_step == 0
        assert self.accumulator.total_accumulated_samples == 0
        
        # Accumulated gradients should be zeroed but not empty
        for grad in self.accumulator.accumulated_gradients.values():
            np.testing.assert_array_equal(grad, np.zeros_like(grad))
    
    def test_get_effective_batch_size(self):
        """Test effective batch size calculation."""
        accumulator = GradientAccumulator(accumulation_steps=2)
        
        # No history yet
        assert accumulator.get_effective_batch_size() == 1.0
        
        # Add some accumulation history
        batch_sizes = [8, 12, 16, 10, 14]
        for batch_size in batch_sizes:
            should_step = accumulator.accumulate_gradients(self.gradients, batch_size)
            if should_step:
                accumulator.reset()
        
        # Should calculate average from recent history
        effective_batch_size = accumulator.get_effective_batch_size()
        assert effective_batch_size > 1.0  # Should be reasonable
    
    def test_accumulation_history_tracking(self):
        """Test accumulation history tracking."""
        accumulator = GradientAccumulator(accumulation_steps=2)
        
        # Trigger multiple accumulation cycles
        for cycle in range(3):
            for step in range(2):
                batch_size = (cycle + 1) * 4 + step * 2
                should_step = accumulator.accumulate_gradients(self.gradients, batch_size)
                if should_step:
                    # Check history entry was added
                    assert len(accumulator.accumulation_history) == cycle + 1
                    
                    recent_history = accumulator.accumulation_history[-1]
                    assert 'steps' in recent_history
                    assert 'samples' in recent_history
                    assert 'avg_batch_size' in recent_history
                    assert recent_history['steps'] == 2
                    
                    accumulator.reset()
    
    def test_variable_batch_sizes(self):
        """Test handling of variable batch sizes."""
        accumulator = GradientAccumulator(accumulation_steps=3)
        variable_batches = [1, 100, 50]  # Very different sizes
        
        for i, batch_size in enumerate(variable_batches):
            should_step = accumulator.accumulate_gradients(self.gradients, batch_size)
            
            if i < 2:
                assert should_step is False
            else:
                assert should_step is True
                # Total samples should be sum of all batches
                assert accumulator.total_accumulated_samples == sum(variable_batches)


class TestMixedPrecisionManager:
    """Test mixed precision training with automatic loss scaling."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mp_manager = MixedPrecisionManager(
            initial_scale=1024.0,
            scale_window=10,
            scale_factor=2.0,
            min_scale=1.0
        )
        self.gradients = {
            'param1': np.random.randn(5, 5),
            'param2': np.random.randn(10)
        }
    
    def test_initialization(self):
        """Test mixed precision manager initialization."""
        assert self.mp_manager.scale == 1024.0
        assert self.mp_manager.scale_window == 10
        assert self.mp_manager.scale_factor == 2.0
        assert self.mp_manager.min_scale == 1.0
        assert self.mp_manager.steps_since_last_scale == 0
        assert self.mp_manager.consecutive_skipped_steps == 0
        assert self.mp_manager.total_skipped_steps == 0
        assert len(self.mp_manager.scale_history) == 0
        assert len(self.mp_manager.overflow_history) == 0
    
    def test_loss_scaling(self):
        """Test loss scaling functionality."""
        loss = 0.5
        scaled_loss = self.mp_manager.scale_loss(loss)
        
        assert scaled_loss == loss * self.mp_manager.scale
        assert scaled_loss == 0.5 * 1024.0
    
    def test_gradient_unscaling(self):
        """Test gradient unscaling."""
        # Create scaled gradients (simulate after backward pass)
        scaled_gradients = {}
        for name, grad in self.gradients.items():
            scaled_gradients[name] = grad * self.mp_manager.scale
        
        # Unscale gradients
        unscaled_gradients = self.mp_manager.unscale_gradients(scaled_gradients)
        
        # Should match original gradients
        for name in self.gradients.keys():
            np.testing.assert_array_almost_equal(
                unscaled_gradients[name], 
                self.gradients[name]
            )
    
    def test_finite_gradient_check(self):
        """Test gradient finiteness checking."""
        # Test with finite gradients
        finite_gradients = {
            'param1': np.array([1.0, 2.0, 3.0]),
            'param2': np.array([0.5, -1.5])
        }
        assert self.mp_manager.check_gradients_finite(finite_gradients) is True
        
        # Test with NaN gradients
        nan_gradients = {
            'param1': np.array([1.0, np.nan, 3.0]),
            'param2': np.array([0.5, -1.5])
        }
        assert self.mp_manager.check_gradients_finite(nan_gradients) is False
        
        # Test with Inf gradients
        inf_gradients = {
            'param1': np.array([1.0, 2.0, np.inf]),
            'param2': np.array([0.5, -1.5])
        }
        assert self.mp_manager.check_gradients_finite(inf_gradients) is False
        
        # Test with -Inf gradients
        neg_inf_gradients = {
            'param1': np.array([1.0, 2.0, -np.inf]),
            'param2': np.array([0.5, -1.5])
        }
        assert self.mp_manager.check_gradients_finite(neg_inf_gradients) is False
    
    def test_scale_update_with_finite_gradients(self):
        """Test loss scale update with finite gradients."""
        initial_scale = self.mp_manager.scale
        
        # Update with finite gradients multiple times
        for step in range(15):  # More than scale_window (10)
            should_step = self.mp_manager.update_scale(gradients_finite=True)
            assert should_step is True  # Should always allow stepping with finite gradients
        
        # Scale should have increased after scale_window steps
        assert self.mp_manager.scale > initial_scale
        assert len(self.mp_manager.scale_history) > 0
        assert all(overflow == 0 for overflow in self.mp_manager.overflow_history)
    
    def test_scale_update_with_overflow(self):
        """Test loss scale update with gradient overflow."""
        initial_scale = self.mp_manager.scale
        
        # Simulate overflow
        should_step = self.mp_manager.update_scale(gradients_finite=False)
        
        assert should_step is False  # Should skip step due to overflow
        assert self.mp_manager.scale < initial_scale  # Scale should decrease
        assert self.mp_manager.total_skipped_steps == 1
        assert self.mp_manager.consecutive_skipped_steps == 1
        assert self.mp_manager.overflow_history[-1] == 1
    
    def test_scale_update_mixed_scenario(self):
        """Test scale update with mixed finite/overflow scenarios."""
        # Start with some finite gradients
        for _ in range(5):
            self.mp_manager.update_scale(gradients_finite=True)
        
        # Cause overflow
        should_step = self.mp_manager.update_scale(gradients_finite=False)
        assert should_step is False
        
        # Recovery with finite gradients
        for _ in range(12):  # More than scale_window
            should_step = self.mp_manager.update_scale(gradients_finite=True)
            assert should_step is True
        
        # Scale should have recovered and possibly increased
        assert self.mp_manager.consecutive_skipped_steps == 0
    
    def test_minimum_scale_limit(self):
        """Test minimum scale limit enforcement."""
        # Set scale close to minimum
        self.mp_manager.scale = 2.0
        
        # Force multiple overflows
        for _ in range(5):
            self.mp_manager.update_scale(gradients_finite=False)
        
        # Scale should not go below minimum
        assert self.mp_manager.scale >= self.mp_manager.min_scale
    
    def test_scale_statistics(self):
        """Test loss scaling statistics."""
        # Create mixed history
        finite_steps = [True] * 8 + [False] * 2  # 20% overflow rate
        
        for finite in finite_steps:
            self.mp_manager.update_scale(finite)
        
        stats = self.mp_manager.get_scale_stats()
        
        assert 'current_scale' in stats
        assert 'overflow_rate' in stats
        assert 'total_skipped_steps' in stats
        assert 'consecutive_skipped_steps' in stats
        assert 'avg_scale' in stats
        
        assert stats['current_scale'] == self.mp_manager.scale
        assert 0.0 <= stats['overflow_rate'] <= 1.0
        assert stats['total_skipped_steps'] >= 0
        assert stats['consecutive_skipped_steps'] >= 0


class TestLRScheduler:
    """Test advanced learning rate scheduler."""
    
    def test_constant_schedule(self):
        """Test constant learning rate schedule."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.CONSTANT,
            base_lr=1e-3,
            warmup_steps=5,
            total_steps=100
        )
        
        # During warmup
        for step in range(5):
            lr = scheduler.get_lr(step)
            expected_lr = 1e-3 * (step / 5)
            assert abs(lr - expected_lr) < 1e-6
        
        # After warmup (constant)
        for step in range(5, 20):
            lr = scheduler.get_lr(step)
            assert abs(lr - 1e-3) < 1e-6
    
    def test_linear_warmup_schedule(self):
        """Test linear warmup with decay schedule."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.LINEAR_WARMUP,
            base_lr=1e-3,
            warmup_steps=10,
            total_steps=100
        )
        
        # During warmup
        for step in range(10):
            lr = scheduler.get_lr(step)
            expected_warmup_lr = 1e-3 * (step / 10)
            assert abs(lr - expected_warmup_lr) < 1e-6
        
        # After warmup (linear decay)
        lr_50 = scheduler.get_lr(50)
        lr_75 = scheduler.get_lr(75)
        lr_100 = scheduler.get_lr(100)
        
        # Should decay linearly
        assert lr_50 > lr_75 > lr_100
        assert lr_100 == 0.0  # Should reach zero at total_steps
    
    def test_cosine_annealing_schedule(self):
        """Test cosine annealing schedule."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.COSINE_ANNEALING,
            base_lr=1e-3,
            warmup_steps=0,  # No warmup for clarity
            total_steps=100,
            min_lr=1e-5
        )
        
        lr_0 = scheduler.get_lr(0)
        lr_25 = scheduler.get_lr(25)
        lr_50 = scheduler.get_lr(50)
        lr_75 = scheduler.get_lr(75)
        lr_100 = scheduler.get_lr(100)
        
        # Should follow cosine curve
        assert abs(lr_0 - 1e-3) < 1e-6  # Start at base_lr
        assert abs(lr_100 - 1e-5) < 1e-6  # End at min_lr
        assert lr_50 < lr_0  # Should decrease
        assert lr_50 > lr_100  # Should be between start and end
    
    def test_exponential_decay_schedule(self):
        """Test exponential decay schedule."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.EXPONENTIAL_DECAY,
            base_lr=1e-3,
            warmup_steps=0,
            total_steps=100,
            decay_rate=0.9,
            decay_steps=10
        )
        
        lr_0 = scheduler.get_lr(0)
        lr_10 = scheduler.get_lr(10)
        lr_20 = scheduler.get_lr(20)
        
        # Should decay exponentially
        assert abs(lr_0 - 1e-3) < 1e-6
        assert abs(lr_10 - (1e-3 * 0.9)) < 1e-6
        assert abs(lr_20 - (1e-3 * 0.9 ** 2)) < 1e-6
    
    def test_polynomial_decay_schedule(self):
        """Test polynomial decay schedule."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.POLYNOMIAL_DECAY,
            base_lr=1e-3,
            warmup_steps=0,
            total_steps=100,
            min_lr=1e-5,
            power=2.0
        )
        
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)
        lr_100 = scheduler.get_lr(100)
        
        # Should follow polynomial decay
        assert abs(lr_0 - 1e-3) < 1e-6
        assert abs(lr_100 - 1e-5) < 1e-6
        assert lr_0 > lr_50 > lr_100
    
    def test_one_cycle_schedule(self):
        """Test one cycle policy schedule."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.ONE_CYCLE,
            base_lr=1e-3,
            warmup_steps=0,
            total_steps=100,
            min_lr=1e-5,
            max_lr=1e-2
        )
        
        lr_0 = scheduler.get_lr(0)
        lr_25 = scheduler.get_lr(25)
        lr_50 = scheduler.get_lr(50)
        lr_75 = scheduler.get_lr(75)
        lr_100 = scheduler.get_lr(100)
        
        # Should increase then decrease
        assert lr_0 < lr_25 < lr_50  # Increasing phase
        assert lr_50 > lr_75 > lr_100  # Decreasing phase
        assert lr_50 == 1e-2  # Peak at max_lr
    
    def test_reduce_on_plateau_schedule(self):
        """Test reduce on plateau schedule."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.REDUCE_ON_PLATEAU,
            base_lr=1e-3,
            warmup_steps=0,
            total_steps=100,
            patience=3,
            factor=0.5
        )
        
        initial_lr = scheduler.get_lr()
        
        # Simulate improving metrics (no reduction)
        for step in range(5):
            scheduler.step(metric=1.0 - step * 0.1)  # Decreasing loss
            lr = scheduler.get_lr()
            assert lr == initial_lr
        
        # Simulate plateau (should reduce)
        for step in range(5):
            scheduler.step(metric=0.5)  # Constant loss
        
        # After patience steps, LR should be reduced
        reduced_lr = scheduler.get_lr()
        assert reduced_lr == initial_lr * 0.5
    
    def test_warmup_functionality(self):
        """Test warmup functionality across different schedules."""
        schedules = [
            LRScheduleType.CONSTANT,
            LRScheduleType.COSINE_ANNEALING,
            LRScheduleType.EXPONENTIAL_DECAY
        ]
        
        for schedule_type in schedules:
            scheduler = LRScheduler(
                schedule_type=schedule_type,
                base_lr=1e-3,
                warmup_steps=10,
                total_steps=100
            )
            
            # During warmup, all schedules should behave the same
            for step in range(10):
                lr = scheduler.get_lr(step)
                expected_lr = 1e-3 * (step / 10)
                assert abs(lr - expected_lr) < 1e-6, f"Failed for {schedule_type} at step {step}"
    
    def test_scheduler_step_method(self):
        """Test scheduler step method and history tracking."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.CONSTANT,
            base_lr=1e-3,
            warmup_steps=0,
            total_steps=100
        )
        
        # Initial state
        assert scheduler.step_count == 0
        
        # Step a few times
        for i in range(5):
            scheduler.step()
            assert scheduler.step_count == i + 1
        
        # Check history
        assert len(scheduler.lr_history) > 0
    
    def test_get_last_lr(self):
        """Test get_last_lr functionality."""
        scheduler = LRScheduler(
            schedule_type=LRScheduleType.CONSTANT,
            base_lr=1e-3,
            warmup_steps=0,
            total_steps=100
        )
        
        # Before any LR computation
        assert scheduler.get_last_lr() == 1e-3
        
        # After LR computation
        scheduler.get_lr(5)
        assert scheduler.get_last_lr() == scheduler.lr_history[-1]


class TestAdaptiveGradientClipper:
    """Test adaptive gradient clipping."""
    
    def setup_method(self):
        """Setup test environment."""
        self.gradients = {
            'param1': np.random.randn(10, 5) * 0.1,
            'param2': np.random.randn(3) * 2.0,  # Larger gradients
            'param3': np.random.randn(5, 5) * 0.01
        }
    
    def test_norm_clipping(self):
        """Test gradient clipping by norm."""
        clipper = AdaptiveGradientClipper(
            clip_type="norm",
            max_norm=1.0
        )
        
        # Create gradients with large norm
        large_gradients = {
            'param1': np.ones((3, 3)) * 10,  # Large gradients
            'param2': np.ones(5) * 5
        }
        
        # Compute total norm before clipping
        total_norm_before = 0
        for grad in large_gradients.values():
            total_norm_before += np.linalg.norm(grad) ** 2
        total_norm_before = math.sqrt(total_norm_before)
        
        clipped_grads, clip_factor = clipper.clip_gradients(large_gradients)
        
        # Verify clipping occurred
        assert clip_factor < 1.0  # Should be clipped
        
        # Compute total norm after clipping
        total_norm_after = 0
        for grad in clipped_grads.values():
            total_norm_after += np.linalg.norm(grad) ** 2
        total_norm_after = math.sqrt(total_norm_after)
        
        # Should be approximately equal to max_norm
        assert abs(total_norm_after - 1.0) < 1e-5
    
    def test_value_clipping(self):
        """Test gradient clipping by value."""
        clipper = AdaptiveGradientClipper(
            clip_type="value",
            max_norm=1.0
        )
        
        # Create gradients with large values
        large_gradients = {
            'param1': np.array([5.0, -3.0, 2.0]),
            'param2': np.array([[10.0, -8.0], [1.5, -0.5]])
        }
        
        clipped_grads, clip_factor = clipper.clip_gradients(large_gradients)
        
        # All values should be clipped to [-1, 1]
        for grad in clipped_grads.values():
            assert np.all(grad <= 1.0)
            assert np.all(grad >= -1.0)
        
        # Specific checks
        np.testing.assert_array_equal(clipped_grads['param1'], [1.0, -1.0, 1.0])
        np.testing.assert_array_equal(clipped_grads['param2'], [[1.0, -1.0], [1.0, -0.5]])
    
    def test_adaptive_clipping(self):
        """Test adaptive gradient clipping."""
        clipper = AdaptiveGradientClipper(
            clip_type="adaptive",
            percentile=80.0,
            history_size=100
        )
        
        # Build up gradient norm history
        for _ in range(20):
            # Simulate gradients with varying norms
            norm = np.random.uniform(0.1, 2.0)
            dummy_grads = {'param': np.random.randn(5) * norm}
            clipper.clip_gradients(dummy_grads)
        
        # Now test with known gradients
        test_gradients = {
            'param1': np.ones(3) * 5,  # Large gradients
            'param2': np.ones(2) * 0.1  # Small gradients
        }
        
        clipped_grads, clip_factor = clipper.clip_gradients(test_gradients)
        
        # Should have been clipped based on historical percentile
        assert len(clipper.grad_norm_history) > 10
        assert clip_factor <= 1.0
    
    def test_percentile_clipping(self):
        """Test percentile-based gradient clipping."""
        clipper = AdaptiveGradientClipper(
            clip_type="percentile",
            percentile=90.0
        )
        
        # Build history with known values
        known_norms = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for norm in known_norms:
            dummy_grads = {'param': np.ones(2) * norm}
            clipper.clip_gradients(dummy_grads)
        
        # 90th percentile of [0.1, ..., 1.0] should be 0.9
        test_gradients = {'param': np.ones(2) * 2.0}  # Larger than 90th percentile
        
        clipped_grads, clip_factor = clipper.clip_gradients(test_gradients)
        
        # Should be clipped
        assert clip_factor < 1.0
    
    def test_no_clipping_needed(self):
        """Test case where no clipping is needed."""
        clipper = AdaptiveGradientClipper(
            clip_type="norm",
            max_norm=10.0  # High threshold
        )
        
        small_gradients = {
            'param1': np.ones(3) * 0.1,
            'param2': np.ones(2) * 0.05
        }
        
        clipped_grads, clip_factor = clipper.clip_gradients(small_gradients)
        
        # No clipping should occur
        assert clip_factor == 1.0
        
        # Gradients should be unchanged
        for name in small_gradients:
            np.testing.assert_array_equal(clipped_grads[name], small_gradients[name])
    
    def test_clipping_statistics(self):
        """Test gradient clipping statistics."""
        clipper = AdaptiveGradientClipper(clip_type="norm", max_norm=1.0)
        
        # Generate mixed clipping scenarios
        test_cases = [
            {'param': np.ones(2) * 0.1},  # No clipping
            {'param': np.ones(2) * 2.0},  # Clipping needed
            {'param': np.ones(2) * 0.5},  # No clipping
            {'param': np.ones(2) * 3.0},  # Clipping needed
        ]
        
        for grads in test_cases:
            clipper.clip_gradients(grads)
        
        stats = clipper.get_clipping_stats()
        
        assert 'avg_grad_norm' in stats
        assert 'max_grad_norm' in stats
        assert 'avg_clip_factor' in stats
        assert 'clip_percentage' in stats
        
        assert stats['avg_grad_norm'] > 0
        assert stats['max_grad_norm'] > 0
        assert 0 <= stats['avg_clip_factor'] <= 1
        assert 0 <= stats['clip_percentage'] <= 100
    
    def test_gradient_norm_calculation(self):
        """Test gradient norm calculation across multiple parameters."""
        clipper = AdaptiveGradientClipper(clip_type="norm", max_norm=1.0)
        
        # Known gradients for testing
        gradients = {
            'param1': np.array([3.0, 4.0]),  # Norm = 5.0
            'param2': np.array([0.0, 0.0]),  # Norm = 0.0
            'param3': np.array([1.0])        # Norm = 1.0
        }
        
        # Expected total norm = sqrt(5^2 + 0^2 + 1^2) = sqrt(26) ≈ 5.099
        expected_total_norm = math.sqrt(26)
        
        clipped_grads, clip_factor = clipper.clip_gradients(gradients)
        
        # Should be clipped since total norm > max_norm
        assert clip_factor < 1.0
        assert abs(clip_factor - (1.0 / expected_total_norm)) < 1e-6
        
        # Check final norm is approximately 1.0
        final_norm = 0
        for grad in clipped_grads.values():
            final_norm += np.linalg.norm(grad) ** 2
        final_norm = math.sqrt(final_norm)
        
        assert abs(final_norm - 1.0) < 1e-5


class TestSophiaOptimizer:
    """Test Sophia optimizer (second-order optimization)."""
    
    def setup_method(self):
        """Setup test environment."""
        self.params = {
            'layer1.weight': np.random.randn(10, 5).astype(np.float32),
            'layer1.bias': np.random.randn(5).astype(np.float32),
            'layer2.weight': np.random.randn(5, 3).astype(np.float32)
        }
        
        self.gradients = {
            'layer1.weight': np.random.randn(10, 5).astype(np.float32) * 0.01,
            'layer1.bias': np.random.randn(5).astype(np.float32) * 0.01,
            'layer2.weight': np.random.randn(5, 3).astype(np.float32) * 0.01
        }
        
        self.optimizer = SophiaOptimizer(
            self.params,
            lr=1e-4,
            beta1=0.965,
            beta2=0.99,
            rho=0.04,
            weight_decay=0.1,
            update_period=5
        )
    
    def test_initialization(self):
        """Test Sophia optimizer initialization."""
        assert self.optimizer.lr == 1e-4
        assert self.optimizer.beta1 == 0.965
        assert self.optimizer.beta2 == 0.99
        assert self.optimizer.rho == 0.04
        assert self.optimizer.weight_decay == 0.1
        assert self.optimizer.update_period == 5
        
        # Check state initialization
        for name in self.params.keys():
            assert name in self.optimizer.state
            state = self.optimizer.state[name]
            assert state['step'] == 0
            assert 'momentum' in state
            assert 'hessian_diag' in state
            assert state['last_hessian_update'] == 0
            
            # Check shapes
            assert state['momentum'].shape == self.params[name].shape
            assert state['hessian_diag'].shape == self.params[name].shape
    
    def test_single_optimization_step(self):
        """Test single optimization step."""
        # Store original parameters
        original_params = {name: param.copy() for name, param in self.params.items()}
        
        # Perform optimization step
        self.optimizer.step(self.params, self.gradients)
        
        # Check that parameters were updated
        for name in self.params.keys():
            assert not np.array_equal(self.params[name], original_params[name])
            
            # Check state updates
            state = self.optimizer.state[name]
            assert state['step'] == 1
    
    def test_momentum_update(self):
        """Test momentum update mechanism."""
        # Perform first step
        self.optimizer.step(self.params, self.gradients)
        
        # Check momentum was updated
        for name in self.params.keys():
            state = self.optimizer.state[name]
            expected_momentum = (1 - self.optimizer.beta1) * self.gradients[name]
            np.testing.assert_array_almost_equal(state['momentum'], expected_momentum)
    
    def test_hessian_update_period(self):
        """Test Hessian diagonal update period."""
        # Perform steps up to update period
        for step in range(self.optimizer.update_period):
            self.optimizer.step(self.params, self.gradients)
            
            for name in self.params.keys():
                state = self.optimizer.state[name]
                if step + 1 < self.optimizer.update_period:
                    # Hessian should not be updated yet
                    assert state['last_hessian_update'] == 0
                else:
                    # Hessian should be updated
                    assert state['last_hessian_update'] == step + 1
    
    def test_hessian_diagonal_provided(self):
        """Test optimization step with provided Hessian diagonal."""
        hessian_diag = {}
        for name, param in self.params.items():
            hessian_diag[name] = np.random.rand(*param.shape) * 0.1
        
        # Perform step with explicit Hessian
        self.optimizer.step(self.params, self.gradients, hessian_diag)
        
        # Should still work correctly
        for name in self.params.keys():
            state = self.optimizer.state[name]
            assert state['step'] == 1
    
    def test_weight_decay(self):
        """Test weight decay application."""
        # Test with zero weight decay
        optimizer_no_decay = SophiaOptimizer(
            self.params,
            weight_decay=0.0
        )
        
        # Create copy for comparison
        params_no_decay = {name: param.copy() for name, param in self.params.items()}
        params_with_decay = {name: param.copy() for name, param in self.params.items()}
        
        # Run both optimizers
        optimizer_no_decay.step(params_no_decay, self.gradients)
        self.optimizer.step(params_with_decay, self.gradients)
        
        # Weight decay should cause different parameter updates
        for name in self.params.keys():
            param_diff = np.linalg.norm(params_with_decay[name] - params_no_decay[name])
            assert param_diff > 1e-6  # Should be different due to weight decay
    
    def test_bias_correction(self):
        """Test bias correction in optimization."""
        # Perform multiple steps to test bias correction
        for step in range(10):
            self.optimizer.step(self.params, self.gradients)
        
        # Check that bias correction factors are reasonable
        for name in self.params.keys():
            state = self.optimizer.state[name]
            step_count = state['step']
            
            bias_correction1 = 1 - self.optimizer.beta1 ** step_count
            bias_correction2 = 1 - self.optimizer.beta2 ** step_count
            
            # Should approach 1 as steps increase
            assert bias_correction1 > 0.5
            assert bias_correction2 > 0.5
    
    def test_update_clipping(self):
        """Test update clipping with rho parameter."""
        # Create gradients that would cause large updates
        large_gradients = {}
        for name, param in self.params.items():
            large_gradients[name] = np.ones_like(param) * 10.0
        
        # Store original parameters
        original_params = {name: param.copy() for name, param in self.params.items()}
        
        # Perform optimization with large gradients
        self.optimizer.step(self.params, large_gradients)
        
        # Calculate actual update magnitudes
        for name in self.params.keys():
            update_magnitude = np.linalg.norm(self.params[name] - original_params[name])
            # Update should be bounded by lr * rho (approximately)
            max_expected_update = self.optimizer.lr * self.optimizer.rho * math.sqrt(self.params[name].size)
            assert update_magnitude <= max_expected_update * 2  # Allow some margin
    
    def test_numerical_stability(self):
        """Test numerical stability with small epsilon."""
        # Test with very small gradients
        tiny_gradients = {}
        for name, param in self.params.items():
            tiny_gradients[name] = np.ones_like(param) * 1e-10
        
        # Should not cause numerical issues
        try:
            self.optimizer.step(self.params, tiny_gradients)
        except (ValueError, RuntimeWarning):
            pytest.fail("Optimizer failed with small gradients")
        
        # Test with zero gradients
        zero_gradients = {}
        for name, param in self.params.items():
            zero_gradients[name] = np.zeros_like(param)
        
        try:
            self.optimizer.step(self.params, zero_gradients)
        except (ValueError, RuntimeWarning):
            pytest.fail("Optimizer failed with zero gradients")


class TestLionOptimizer:
    """Test Lion optimizer (EvoLved Sign Momentum)."""
    
    def setup_method(self):
        """Setup test environment."""
        self.params = {
            'layer1.weight': np.random.randn(8, 4).astype(np.float32),
            'layer1.bias': np.random.randn(4).astype(np.float32),
            'layer2.weight': np.random.randn(4, 2).astype(np.float32)
        }
        
        self.gradients = {
            'layer1.weight': np.random.randn(8, 4).astype(np.float32) * 0.1,
            'layer1.bias': np.random.randn(4).astype(np.float32) * 0.1,
            'layer2.weight': np.random.randn(4, 2).astype(np.float32) * 0.1
        }
        
        self.optimizer = LionOptimizer(
            self.params,
            lr=1e-4,
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.01
        )
    
    def test_initialization(self):
        """Test Lion optimizer initialization."""
        assert self.optimizer.lr == 1e-4
        assert self.optimizer.beta1 == 0.9
        assert self.optimizer.beta2 == 0.99
        assert self.optimizer.weight_decay == 0.01
        
        # Check momentum initialization
        for name in self.params.keys():
            assert name in self.optimizer.momentum
            assert self.optimizer.momentum[name].shape == self.params[name].shape
            np.testing.assert_array_equal(
                self.optimizer.momentum[name], 
                np.zeros_like(self.params[name])
            )
    
    def test_single_optimization_step(self):
        """Test single optimization step."""
        # Store original parameters
        original_params = {name: param.copy() for name, param in self.params.items()}
        
        # Perform optimization step
        self.optimizer.step(self.params, self.gradients)
        
        # Check that parameters were updated
        for name in self.params.keys():
            assert not np.array_equal(self.params[name], original_params[name])
    
    def test_sign_based_updates(self):
        """Test sign-based parameter updates (key feature of Lion)."""
        # Use controlled gradients to test sign operation
        controlled_gradients = {
            'layer1.weight': np.array([[1.0, -2.0], [3.0, -0.5]]),
            'layer1.bias': np.array([0.5, -1.5])
        }
        
        controlled_params = {
            'layer1.weight': np.zeros((2, 2)),
            'layer1.bias': np.zeros(2)
        }
        
        lion_optimizer = LionOptimizer(controlled_params, lr=0.1, beta1=0.9)
        
        # First step (momentum starts at zero)
        lion_optimizer.step(controlled_params, controlled_gradients)
        
        # Check that updates are based on sign of interpolation
        # Update = beta1 * momentum + (1 - beta1) * gradient
        # Since momentum is initially zero: update = 0.1 * gradient
        expected_update_signs = {
            'layer1.weight': np.array([[1, -1], [1, -1]]),  # Sign of gradients
            'layer1.bias': np.array([1, -1])
        }
        
        for name in controlled_params.keys():
            actual_update = controlled_params[name]  # Parameters started at zero
            expected_signs = expected_update_signs[name]
            actual_signs = np.sign(actual_update / (-0.1))  # Divide by -lr to get update direction
            
            np.testing.assert_array_equal(actual_signs, expected_signs)
    
    def test_momentum_update_mechanism(self):
        """Test momentum update mechanism."""
        # Perform first step
        self.optimizer.step(self.params, self.gradients)
        
        # Check momentum update: momentum = beta2 * momentum + (1 - beta2) * gradient
        for name in self.params.keys():
            expected_momentum = (1 - self.optimizer.beta2) * self.gradients[name]
            np.testing.assert_array_almost_equal(
                self.optimizer.momentum[name], 
                expected_momentum,
                decimal=6
            )
        
        # Store momentum after first step
        momentum_after_first = {name: mom.copy() for name, mom in self.optimizer.momentum.items()}
        
        # Perform second step with same gradients
        self.optimizer.step(self.params, self.gradients)
        
        # Check momentum update after second step
        for name in self.params.keys():
            expected_momentum = (
                self.optimizer.beta2 * momentum_after_first[name] + 
                (1 - self.optimizer.beta2) * self.gradients[name]
            )
            np.testing.assert_array_almost_equal(
                self.optimizer.momentum[name],
                expected_momentum,
                decimal=6
            )
    
    def test_weight_decay(self):
        """Test weight decay application."""
        # Test with non-zero parameters
        nonzero_params = {
            'param1': np.ones((3, 3)),
            'param2': np.ones(5) * 2.0
        }
        
        gradients = {
            'param1': np.zeros((3, 3)),  # Zero gradients to isolate weight decay effect
            'param2': np.zeros(5)
        }
        
        optimizer_with_decay = LionOptimizer(
            nonzero_params, 
            lr=0.1, 
            weight_decay=0.1
        )
        
        original_params = {name: param.copy() for name, param in nonzero_params.items()}
        
        # Perform step
        optimizer_with_decay.step(nonzero_params, gradients)
        
        # Parameters should be reduced due to weight decay
        for name in nonzero_params.keys():
            # Weight decay: param = param * (1 - lr * weight_decay)
            expected_param = original_params[name] * (1 - 0.1 * 0.1)
            np.testing.assert_array_almost_equal(
                nonzero_params[name],
                expected_param,
                decimal=6
            )
    
    def test_no_weight_decay(self):
        """Test optimizer with zero weight decay."""
        optimizer_no_decay = LionOptimizer(
            self.params,
            lr=1e-4,
            weight_decay=0.0
        )
        
        original_params = {name: param.copy() for name, param in self.params.items()}
        
        # Perform step
        optimizer_no_decay.step(self.params, self.gradients)
        
        # Should only be affected by gradient-based updates
        for name in self.params.keys():
            # Parameters should change (due to gradients) but not be scaled by weight decay
            assert not np.array_equal(self.params[name], original_params[name])
    
    def test_gradient_interpolation(self):
        """Test gradient-momentum interpolation."""
        # Set up controlled test
        controlled_params = {'param': np.zeros(3)}
        
        # First gradient
        grad1 = np.array([1.0, 0.0, -1.0])
        optimizer = LionOptimizer(controlled_params, lr=0.1, beta1=0.8, beta2=0.9)
        
        optimizer.step(controlled_params, {'param': grad1})
        
        # Momentum after first step
        expected_momentum1 = 0.1 * grad1  # (1 - beta2) * grad1
        np.testing.assert_array_almost_equal(optimizer.momentum['param'], expected_momentum1)
        
        # Second gradient (different)
        grad2 = np.array([0.0, 2.0, 1.0])
        optimizer.step(controlled_params, {'param': grad2})
        
        # Momentum should be: beta2 * momentum1 + (1 - beta2) * grad2
        expected_momentum2 = 0.9 * expected_momentum1 + 0.1 * grad2
        np.testing.assert_array_almost_equal(optimizer.momentum['param'], expected_momentum2)
    
    def test_multiple_optimization_steps(self):
        """Test multiple optimization steps for convergence behavior."""
        # Create simple quadratic loss: f(x) = x^2
        param = {'x': np.array([2.0])}  # Start away from optimum
        
        optimizer = LionOptimizer(param, lr=0.1)
        
        initial_value = param['x'].copy()
        
        # Perform multiple steps with gradient = 2x
        for _ in range(10):
            gradient = {'x': 2 * param['x']}  # Gradient of x^2
            optimizer.step(param, gradient)
        
        # Should move towards zero (though may not converge exactly due to sign updates)
        final_value = param['x']
        assert abs(final_value[0]) < abs(initial_value[0])  # Should get closer to optimum
    
    def test_missing_gradients(self):
        """Test behavior with missing gradients."""
        # Create gradient dict missing some parameters
        incomplete_gradients = {
            'layer1.weight': self.gradients['layer1.weight']
            # Missing other parameters
        }
        
        original_params = {name: param.copy() for name, param in self.params.items()}
        
        # Should handle missing gradients gracefully
        self.optimizer.step(self.params, incomplete_gradients)
        
        # Only parameters with gradients should be updated
        assert not np.array_equal(self.params['layer1.weight'], original_params['layer1.weight'])
        np.testing.assert_array_equal(self.params['layer1.bias'], original_params['layer1.bias'])
        np.testing.assert_array_equal(self.params['layer2.weight'], original_params['layer2.weight'])


class TestAdvancedOptimizer:
    """Test integrated AdvancedOptimizer system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.params = {
            'layer1.weight': np.random.randn(20, 10).astype(np.float32),
            'layer1.bias': np.random.randn(10).astype(np.float32),
            'layer2.weight': np.random.randn(10, 5).astype(np.float32),
            'layer2.bias': np.random.randn(5).astype(np.float32)
        }
        
        self.gradients = {
            'layer1.weight': np.random.randn(20, 10).astype(np.float32) * 0.01,
            'layer1.bias': np.random.randn(10).astype(np.float32) * 0.01,
            'layer2.weight': np.random.randn(10, 5).astype(np.float32) * 0.01,
            'layer2.bias': np.random.randn(5).astype(np.float32) * 0.01
        }
        
        self.config = OptimizerConfig(
            optimizer_type=OptimizerType.SOPHIA,
            learning_rate=1e-3,
            use_mixed_precision=True,
            gradient_accumulation_steps=2,
            use_ema=True,
            lr_schedule_type=LRScheduleType.COSINE_ANNEALING,
            warmup_steps=5,
            total_steps=100
        )
        
        self.optimizer = AdvancedOptimizer(self.config)
        self.optimizer.initialize_optimizer(self.params)
    
    def test_initialization(self):
        """Test AdvancedOptimizer initialization."""
        assert self.optimizer.config == self.config
        assert self.optimizer.gradient_accumulator is not None
        assert self.optimizer.mixed_precision is not None
        assert self.optimizer.lr_scheduler is not None
        assert self.optimizer.grad_clipper is not None
        assert self.optimizer.base_optimizer is not None
        assert self.optimizer.ema_params is not None
    
    def test_initialization_without_mixed_precision(self):
        """Test initialization without mixed precision."""
        config_no_mp = OptimizerConfig(use_mixed_precision=False)
        optimizer_no_mp = AdvancedOptimizer(config_no_mp)
        
        assert optimizer_no_mp.mixed_precision is None
    
    def test_initialization_without_ema(self):
        """Test initialization without EMA."""
        config_no_ema = OptimizerConfig(use_ema=False)
        optimizer_no_ema = AdvancedOptimizer(config_no_ema)
        optimizer_no_ema.initialize_optimizer(self.params)
        
        assert optimizer_no_ema.ema_params is None
    
    def test_sophia_optimizer_initialization(self):
        """Test Sophia optimizer initialization."""
        assert isinstance(self.optimizer.base_optimizer, SophiaOptimizer)
        assert self.optimizer.base_optimizer.lr == self.config.learning_rate
        assert self.optimizer.base_optimizer.rho == self.config.rho
    
    def test_lion_optimizer_initialization(self):
        """Test Lion optimizer initialization."""
        lion_config = OptimizerConfig(optimizer_type=OptimizerType.LION)
        lion_optimizer = AdvancedOptimizer(lion_config)
        lion_optimizer.initialize_optimizer(self.params)
        
        assert isinstance(lion_optimizer.base_optimizer, LionOptimizer)
        assert lion_optimizer.base_optimizer.lr == lion_config.learning_rate
    
    def test_single_optimization_step_complete(self):
        """Test complete optimization step."""
        loss = 0.5
        batch_size = 8
        
        # Step multiple times to complete accumulation
        stats1 = self.optimizer.step(self.params, self.gradients, loss, batch_size)
        stats2 = self.optimizer.step(self.params, self.gradients, loss, batch_size)
        
        # First step should be accumulating
        assert stats1.get('accumulating', False) is True
        assert stats1.get('optimizer_step_completed', False) is False
        
        # Second step should complete optimization
        assert stats2.get('optimizer_step_completed', False) is True
        assert 'learning_rate' in stats2
        assert 'gradient_clipping' in stats2
        assert 'mixed_precision' in stats2
    
    def test_gradient_overflow_handling(self):
        """Test gradient overflow handling in mixed precision."""
        # Create gradients that will overflow
        overflow_gradients = {}
        for name, grad in self.gradients.items():
            overflow_gradients[name] = grad * 1e10  # Cause overflow
        
        loss = 0.5
        batch_size = 8
        
        # Should skip optimization due to overflow
        stats = self.optimizer.step(self.params, overflow_gradients, loss, batch_size)
        
        assert stats.get('step_skipped', False) is True
        assert 'mixed_precision' in stats
    
    def test_ema_parameter_updates(self):
        """Test EMA parameter updates."""
        # Store initial EMA parameters
        initial_ema = {name: param.copy() for name, param in self.optimizer.ema_params.items()}
        
        loss = 0.1
        batch_size = 8
        
        # Complete an optimization cycle
        for _ in range(self.config.gradient_accumulation_steps):
            stats = self.optimizer.step(self.params, self.gradients, loss, batch_size)
            if stats.get('optimizer_step_completed'):
                break
        
        # EMA parameters should be updated
        for name in self.optimizer.ema_params.keys():
            assert not np.array_equal(self.optimizer.ema_params[name], initial_ema[name])
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling integration."""
        loss = 0.2
        batch_size = 8
        
        learning_rates = []
        
        # Perform multiple optimization cycles
        for cycle in range(5):
            for step in range(self.config.gradient_accumulation_steps):
                stats = self.optimizer.step(self.params, self.gradients, loss, batch_size)
                if stats.get('optimizer_step_completed'):
                    learning_rates.append(stats['learning_rate'])
                    break
        
        # Learning rates should change according to schedule
        assert len(learning_rates) == 5
        # For cosine annealing with warmup, rates should vary
        assert not all(lr == learning_rates[0] for lr in learning_rates)
    
    def test_gradient_clipping_integration(self):
        """Test gradient clipping integration."""
        # Create large gradients
        large_gradients = {}
        for name, grad in self.gradients.items():
            large_gradients[name] = grad * 100  # Large gradients
        
        loss = 1.0
        batch_size = 8
        
        # Complete optimization cycle
        for _ in range(self.config.gradient_accumulation_steps):
            stats = self.optimizer.step(self.params, large_gradients, loss, batch_size)
            if stats.get('optimizer_step_completed'):
                break
        
        # Should have clipping statistics
        assert 'gradient_clipping' in stats
        assert 'clip_factor' in stats
        assert stats['clip_factor'] <= 1.0  # Should be clipped
    
    def test_get_ema_params(self):
        """Test getting EMA parameters."""
        ema_params = self.optimizer.get_ema_params()
        
        assert ema_params is not None
        assert len(ema_params) == len(self.params)
        
        for name in self.params.keys():
            assert name in ema_params
            assert ema_params[name].shape == self.params[name].shape
    
    def test_get_ema_params_when_disabled(self):
        """Test getting EMA parameters when EMA is disabled."""
        config_no_ema = OptimizerConfig(use_ema=False)
        optimizer_no_ema = AdvancedOptimizer(config_no_ema)
        optimizer_no_ema.initialize_optimizer(self.params)
        
        ema_params = optimizer_no_ema.get_ema_params()
        assert ema_params is None
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics gathering."""
        # Perform some optimization steps first
        loss = 0.3
        batch_size = 8
        
        for _ in range(3):
            for step in range(self.config.gradient_accumulation_steps):
                stats = self.optimizer.step(self.params, self.gradients, loss, batch_size)
                if stats.get('optimizer_step_completed'):
                    break
        
        # Get comprehensive stats
        comp_stats = self.optimizer.get_comprehensive_stats()
        
        expected_keys = [
            'mixed_precision',
            'gradient_clipping', 
            'learning_rate',
            'effective_batch_size'
        ]
        
        for key in expected_keys:
            assert key in comp_stats
        
        # Check types
        assert isinstance(comp_stats['mixed_precision'], dict)
        assert isinstance(comp_stats['gradient_clipping'], dict)
        assert isinstance(comp_stats['learning_rate'], float)
        assert isinstance(comp_stats['effective_batch_size'], float)
    
    def test_optimization_cycle_completion(self):
        """Test complete optimization cycle."""
        loss = 0.4
        batch_size = 16
        
        original_params = {name: param.copy() for name, param in self.params.items()}
        
        # Complete full optimization cycle
        step_completed = False
        for attempt in range(self.config.gradient_accumulation_steps):
            stats = self.optimizer.step(self.params, self.gradients, loss, batch_size)
            if stats.get('optimizer_step_completed'):
                step_completed = True
                break
        
        assert step_completed is True
        
        # Parameters should be updated
        for name in self.params.keys():
            assert not np.array_equal(self.params[name], original_params[name])
    
    def test_mixed_precision_without_overflow(self):
        """Test mixed precision path without overflow."""
        loss = 0.1
        batch_size = 8
        
        # Use normal gradients (no overflow)
        for _ in range(self.config.gradient_accumulation_steps):
            stats = self.optimizer.step(self.params, self.gradients, loss, batch_size)
            if stats.get('optimizer_step_completed'):
                break
        
        # Should complete successfully
        assert stats.get('step_skipped', False) is False
        assert stats.get('optimizer_step_completed') is True
        
        # Mixed precision stats should show no overflow
        mp_stats = stats['mixed_precision']
        assert mp_stats['overflow_rate'] >= 0.0  # Should be low
    
    def test_effective_batch_size_tracking(self):
        """Test effective batch size tracking."""
        variable_batch_sizes = [4, 8, 12, 6]
        
        for batch_size in variable_batch_sizes:
            stats = self.optimizer.step(self.params, self.gradients, 0.2, batch_size)
            if stats.get('optimizer_step_completed'):
                # Should report effective batch size
                assert 'effective_batch_size' in stats
                assert stats['effective_batch_size'] > 0
                break


class TestIntegrationScenarios:
    """Integration tests for advanced optimization scenarios."""
    
    def test_complete_training_simulation(self):
        """Test complete training simulation with all features."""
        # Setup comprehensive configuration
        config = OptimizerConfig(
            optimizer_type=OptimizerType.LION,
            learning_rate=1e-3,
            use_mixed_precision=True,
            gradient_accumulation_steps=4,
            use_ema=True,
            lr_schedule_type=LRScheduleType.ONE_CYCLE,
            warmup_steps=10,
            total_steps=100,
            max_grad_norm=1.0
        )
        
        # Create model parameters
        params = {
            'layer1.weight': np.random.randn(50, 25).astype(np.float32),
            'layer1.bias': np.random.randn(25).astype(np.float32),
            'layer2.weight': np.random.randn(25, 10).astype(np.float32),
            'layer2.bias': np.random.randn(10).astype(np.float32)
        }
        
        optimizer = AdvancedOptimizer(config)
        optimizer.initialize_optimizer(params)
        
        # Simulate training epochs
        training_stats = []
        
        for epoch in range(5):
            epoch_losses = []
            
            for batch in range(10):
                # Simulate decreasing loss over time
                base_loss = 1.0 - (epoch * 10 + batch) * 0.01
                loss = max(base_loss + np.random.normal(0, 0.1), 0.01)
                
                # Generate gradients with some noise
                gradients = {}
                for name, param in params.items():
                    noise_scale = 0.1 * (1.0 - epoch * 0.1)  # Decreasing noise
                    gradients[name] = np.random.randn(*param.shape).astype(np.float32) * noise_scale
                
                # Optimization step
                stats = optimizer.step(params, gradients, loss, batch_size=32)
                
                if stats.get('optimizer_step_completed'):
                    epoch_losses.append(loss)
                    training_stats.append(stats)
            
            # Check epoch progression
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")
        
        # Verify training progressed
        assert len(training_stats) > 0
        
        # Check that learning rates changed (due to scheduling)
        learning_rates = [stat['learning_rate'] for stat in training_stats]
        assert len(set(learning_rates)) > 1  # Should have different learning rates
        
        # Check EMA parameters exist and are different from regular parameters
        ema_params = optimizer.get_ema_params()
        assert ema_params is not None
        
        for name in params.keys():
            # EMA and regular params should be different (but similar)
            param_diff = np.linalg.norm(ema_params[name] - params[name])
            assert 0 < param_diff < 1.0  # Should be close but not identical
    
    def test_optimizer_comparison(self):
        """Test comparison between different optimizers."""
        # Create identical parameters for fair comparison
        base_params = {
            'weight': np.random.randn(10, 5).astype(np.float32),
            'bias': np.random.randn(5).astype(np.float32)
        }
        
        # Create gradients simulating a simple quadratic loss
        def compute_gradients(params):
            # Simulate gradients for minimizing sum of squares
            gradients = {}
            for name, param in params.items():
                gradients[name] = 2 * param + np.random.randn(*param.shape) * 0.01
            return gradients
        
        # Test different optimizers
        optimizer_configs = {
            'sophia': OptimizerConfig(
                optimizer_type=OptimizerType.SOPHIA,
                learning_rate=1e-3,
                use_mixed_precision=False,
                gradient_accumulation_steps=1
            ),
            'lion': OptimizerConfig(
                optimizer_type=OptimizerType.LION,
                learning_rate=1e-3,
                use_mixed_precision=False,
                gradient_accumulation_steps=1
            )
        }
        
        results = {}
        
        for name, config in optimizer_configs.items():
            # Create separate parameter copy for each optimizer
            params = {key: value.copy() for key, value in base_params.items()}
            optimizer = AdvancedOptimizer(config)
            optimizer.initialize_optimizer(params)
            
            initial_loss = np.sum([np.sum(p**2) for p in params.values()])
            
            # Run optimization steps
            for step in range(20):
                gradients = compute_gradients(params)
                loss = np.sum([np.sum(p**2) for p in params.values()])
                stats = optimizer.step(params, gradients, loss, batch_size=1)
            
            final_loss = np.sum([np.sum(p**2) for p in params.values()])
            
            results[name] = {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement': initial_loss - final_loss
            }
        
        # Both optimizers should reduce loss
        for name, result in results.items():
            assert result['improvement'] > 0, f"{name} optimizer did not reduce loss"
            print(f"{name}: {result['initial_loss']:.4f} -> {result['final_loss']:.4f}")
    
    def test_robustness_to_extreme_conditions(self):
        """Test optimizer robustness to extreme conditions."""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOPHIA,
            use_mixed_precision=True,
            gradient_accumulation_steps=1,
            learning_rate=1e-3
        )
        
        params = {
            'param1': np.random.randn(5, 5).astype(np.float32),
            'param2': np.random.randn(10).astype(np.float32)
        }
        
        optimizer = AdvancedOptimizer(config)
        optimizer.initialize_optimizer(params)
        
        # Test extreme conditions
        test_cases = [
            {
                'name': 'zero_gradients',
                'gradients': {name: np.zeros_like(param) for name, param in params.items()},
                'loss': 0.1
            },
            {
                'name': 'very_large_gradients',
                'gradients': {name: np.ones_like(param) * 1000 for name, param in params.items()},
                'loss': 1000.0
            },
            {
                'name': 'very_small_gradients',
                'gradients': {name: np.ones_like(param) * 1e-10 for name, param in params.items()},
                'loss': 1e-10
            },
            {
                'name': 'mixed_scale_gradients',
                'gradients': {
                    'param1': np.ones_like(params['param1']) * 1e-8,
                    'param2': np.ones_like(params['param2']) * 100
                },
                'loss': 50.0
            }
        ]
        
        for test_case in test_cases:
            try:
                stats = optimizer.step(
                    params, 
                    test_case['gradients'], 
                    test_case['loss'], 
                    batch_size=1
                )
                # Should handle extreme conditions gracefully
                assert isinstance(stats, dict)
                print(f"✓ Handled {test_case['name']} successfully")
                
            except Exception as e:
                pytest.fail(f"Failed to handle {test_case['name']}: {e}")


def test_system_integration():
    """Test the complete advanced optimization system integration."""
    # This test would run the built-in test function
    try:
        # Run the built-in system test
        test_advanced_optimizers()
        print("✅ Advanced optimization system integration test passed")
    except Exception as e:
        pytest.fail(f"Advanced optimization system test failed: {e}")


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])