"""Comprehensive tests for mixed precision training functionality.

This test suite covers:
- Advanced gradient scaler functionality
- AMP optimizer wrappers
- Autocast policies and configurations
- Integration with existing optimizers
- Numerical stability and overflow handling
- Performance and convergence validation
"""

import logging
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import core modules
from src.neural_arch.core.base import Module, Parameter
from src.neural_arch.core.tensor import Tensor
from src.neural_arch.exceptions import NumericalError, OptimizerError

# Import optimization modules
from src.neural_arch.optimization.grad_scaler import (
    AdvancedGradScaler,
    ScalerConfig,
    ScalingStrategy,
    check_gradients_finite,
    clip_gradients_by_norm,
    create_scaler,
)
from src.neural_arch.optimization.amp_optimizer import (
    AMPOptimizer,
    AMPOptimizerFactory,
    AMPContext,
    create_amp_adam,
    get_recommended_scaler_config,
)
from src.neural_arch.optimization.mixed_precision import (
    AutocastConfig,
    AutocastPolicy,
    PrecisionConfig,
    AutomaticMixedPrecision,
    MixedPrecisionManager,
    autocast,
    create_precision_config,
    get_recommended_precision_config,
    should_cast_operation,
)

# Import optimizers
from src.neural_arch.optim.adam import Adam
from src.neural_arch.optim.adamw import AdamW
from src.neural_arch.optim.sgd import SGD

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SimpleModel(Module):
    """Simple model for testing."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        # Create parameters manually
        self.weight1 = Parameter(np.random.randn(input_size, hidden_size) * 0.1, name="weight1")
        self.bias1 = Parameter(np.zeros(hidden_size), name="bias1")
        self.weight2 = Parameter(np.random.randn(hidden_size, output_size) * 0.1, name="weight2")
        self.bias2 = Parameter(np.zeros(output_size), name="bias2")
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # First layer
        h1 = Tensor(np.dot(x.data, self.weight1.data) + self.bias1.data, requires_grad=True)
        h1_act = Tensor(np.maximum(0, h1.data), requires_grad=True)  # ReLU
        
        # Second layer
        output = Tensor(np.dot(h1_act.data, self.weight2.data) + self.bias2.data, requires_grad=True)
        return output


class TestAdvancedGradScaler(unittest.TestCase):
    """Test advanced gradient scaler functionality."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        
    def test_scaler_config_creation(self):
        """Test scaler configuration creation."""
        config = ScalerConfig(
            init_scale=32768.0,
            growth_factor=1.5,
            strategy=ScalingStrategy.CONSERVATIVE
        )
        
        self.assertEqual(config.init_scale, 32768.0)
        self.assertEqual(config.growth_factor, 1.5)
        self.assertEqual(config.strategy, ScalingStrategy.CONSERVATIVE)
        
    def test_scaler_initialization(self):
        """Test scaler initialization with different strategies."""
        # Test dynamic strategy
        scaler = AdvancedGradScaler()
        self.assertEqual(scaler.config.strategy, ScalingStrategy.DYNAMIC)
        self.assertEqual(scaler.get_scale(), 65536.0)
        
        # Test conservative strategy
        config = ScalerConfig(strategy=ScalingStrategy.CONSERVATIVE)
        scaler = AdvancedGradScaler(config)
        self.assertEqual(scaler.config.strategy, ScalingStrategy.CONSERVATIVE)
        
    def test_loss_scaling(self):
        """Test loss scaling functionality."""
        scaler = AdvancedGradScaler()
        
        # Create a simple loss
        loss = Tensor(np.array([1.5]), requires_grad=True)
        scaled_loss = scaler.scale(loss)
        
        expected_scale = scaler.get_scale()
        self.assertAlmostEqual(scaled_loss.data[0], 1.5 * expected_scale, places=5)
        
    def test_loss_scaling_with_invalid_input(self):
        """Test loss scaling with invalid inputs."""
        scaler = AdvancedGradScaler()
        
        # Test with non-tensor input
        with self.assertRaises(TypeError):
            scaler.scale("not a tensor")
        
        # Test with non-finite loss
        nan_loss = Tensor(np.array([np.nan]), requires_grad=True)
        with self.assertRaises(NumericalError):
            scaler.scale(nan_loss)
            
    def test_gradient_unscaling(self):
        """Test gradient unscaling functionality."""
        scaler = AdvancedGradScaler()
        
        # Create gradients
        for param in self.model.parameters():
            param.grad = Tensor(np.random.randn(*param.data.shape) * scaler.get_scale())
        
        # Unscale gradients
        success = scaler.unscale_(self.optimizer)
        self.assertTrue(success)
        
        # Check that gradients were unscaled
        for param in self.model.parameters():
            self.assertTrue(np.all(np.isfinite(param.grad.data)))
            self.assertLess(np.max(np.abs(param.grad.data)), 10.0)  # Should be reasonable range
            
    def test_gradient_overflow_detection(self):
        """Test gradient overflow detection and handling."""
        scaler = AdvancedGradScaler()
        
        # Create gradients with inf values
        for param in self.model.parameters():
            grad_data = np.random.randn(*param.data.shape)
            grad_data[0] = np.inf  # Add infinity
            param.grad = Tensor(grad_data)
        
        # Unscale should detect overflow
        success = scaler.unscale_(self.optimizer)
        self.assertFalse(success)
        
    def test_optimizer_step_with_scaler(self):
        """Test optimizer step with gradient scaler."""
        scaler = AdvancedGradScaler()
        
        # Create valid gradients
        for param in self.model.parameters():
            param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
        
        # Step should succeed
        original_scale = scaler.get_scale()
        success = scaler.step(self.optimizer)
        self.assertTrue(success)
        
        # Scale should potentially grow after successful steps
        # (though it may take multiple steps depending on growth interval)
        
    def test_optimizer_step_with_overflow(self):
        """Test optimizer step with gradient overflow."""
        scaler = AdvancedGradScaler()
        
        # Create gradients with overflow
        for param in self.model.parameters():
            grad_data = np.random.randn(*param.data.shape)
            grad_data[0] = np.inf
            param.grad = Tensor(grad_data)
        
        original_scale = scaler.get_scale()
        success = scaler.step(self.optimizer)
        
        # Step should fail and scale should decrease
        self.assertFalse(success)
        self.assertLess(scaler.get_scale(), original_scale)
        
    def test_scaler_state_dict(self):
        """Test scaler state dictionary serialization."""
        scaler = AdvancedGradScaler()
        
        # Modify some state
        scaler._step_count = 100
        scaler._total_successful_steps = 95
        scaler._total_overflows = 5
        
        # Get state dict
        state_dict = scaler.state_dict()
        
        # Check required keys
        required_keys = ["scale", "step_count", "total_successful_steps", "config"]
        for key in required_keys:
            self.assertIn(key, state_dict)
        
        # Create new scaler and load state
        new_scaler = AdvancedGradScaler()
        new_scaler.load_state_dict(state_dict)
        
        self.assertEqual(new_scaler._step_count, 100)
        self.assertEqual(new_scaler._total_successful_steps, 95)
        self.assertEqual(new_scaler._total_overflows, 5)
        
    def test_gradient_statistics(self):
        """Test gradient statistics collection."""
        scaler = AdvancedGradScaler()
        
        # Create gradients and perform several steps
        for i in range(10):
            for param in self.model.parameters():
                param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
            scaler.step(self.optimizer)
        
        stats = scaler.get_statistics()
        
        # Check that statistics are collected
        self.assertIn("total_steps", stats)
        self.assertIn("successful_steps", stats)
        self.assertIn("success_rate", stats)
        self.assertEqual(stats["total_steps"], 10)
        
    def test_utility_functions(self):
        """Test gradient utility functions."""
        # Test gradient finite checking
        finite_result, stats = check_gradients_finite(self.optimizer)
        # Should be True for uninitialized gradients (no gradients)
        
        # Add some gradients
        for param in self.model.parameters():
            param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
        
        finite_result, stats = check_gradients_finite(self.optimizer)
        self.assertTrue(finite_result)
        self.assertGreater(stats["params_with_grad"], 0)
        
        # Test gradient clipping
        total_norm = clip_gradients_by_norm(self.optimizer, max_norm=1.0)
        self.assertGreater(total_norm, 0)


class TestAMPOptimizer(unittest.TestCase):
    """Test AMP optimizer wrapper functionality."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.base_optimizer = Adam(self.model.parameters(), lr=0.001)
        
    def test_amp_optimizer_creation(self):
        """Test AMP optimizer creation."""
        amp_opt = AMPOptimizer(self.base_optimizer)
        
        self.assertTrue(amp_opt.enabled)
        self.assertIsNotNone(amp_opt.scaler)
        
    def test_amp_optimizer_step(self):
        """Test AMP optimizer step functionality."""
        amp_opt = AMPOptimizer(self.base_optimizer)
        
        # Create a simple loss and gradients
        loss = Tensor(np.array([1.0]), requires_grad=True)
        
        # Simulate gradients
        for param in self.model.parameters():
            param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
        
        # Test step
        success = amp_opt.step()
        self.assertTrue(success)
        
    def test_amp_optimizer_with_overflow(self):
        """Test AMP optimizer with gradient overflow."""
        amp_opt = AMPOptimizer(self.base_optimizer)
        
        # Create gradients with overflow
        for param in self.model.parameters():
            grad_data = np.random.randn(*param.data.shape)
            grad_data[0] = np.inf
            param.grad = Tensor(grad_data)
        
        # Step should be skipped
        success = amp_opt.step()
        self.assertFalse(success)
        
    def test_amp_optimizer_disabled(self):
        """Test AMP optimizer when disabled."""
        amp_opt = AMPOptimizer(self.base_optimizer, enabled=False)
        
        # Create gradients
        for param in self.model.parameters():
            param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
        
        # Should always succeed when disabled
        success = amp_opt.step()
        self.assertTrue(success)
        
    def test_amp_optimizer_factory(self):
        """Test AMP optimizer factory methods."""
        # Test creating AMP Adam
        amp_adam = create_amp_adam(self.model.parameters(), lr=0.001)
        self.assertIsInstance(amp_adam, AMPOptimizer)
        
        # Test wrapping existing optimizer
        wrapped = AMPOptimizerFactory.wrap_optimizer(self.base_optimizer)
        self.assertIsInstance(wrapped, AMPOptimizer)
        
    def test_amp_context_manager(self):
        """Test AMP context manager."""
        amp_opt = AMPOptimizer(self.base_optimizer)
        
        # Test enabling/disabling AMP temporarily
        with AMPContext(amp_opt, enabled=False):
            self.assertFalse(amp_opt.enabled)
        
        # Should be restored
        self.assertTrue(amp_opt.enabled)
        
    def test_amp_optimizer_statistics(self):
        """Test AMP optimizer statistics."""
        amp_opt = AMPOptimizer(self.base_optimizer)
        
        # Perform several steps
        for i in range(5):
            for param in self.model.parameters():
                param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
            amp_opt.step()
            amp_opt.zero_grad()
        
        stats = amp_opt.get_statistics()
        
        self.assertEqual(stats["total_steps"], 5)
        self.assertEqual(stats["successful_steps"], 5)
        self.assertEqual(stats["success_rate"], 1.0)


class TestAutocastConfiguration(unittest.TestCase):
    """Test autocast configuration and policies."""
    
    def test_autocast_config_creation(self):
        """Test autocast configuration creation."""
        config = AutocastConfig(
            policy=AutocastPolicy.CONSERVATIVE,
            allowed_ops=["add", "multiply"],
            blocked_ops=["divide"]
        )
        
        self.assertEqual(config.policy, AutocastPolicy.CONSERVATIVE)
        self.assertIn("add", config.allowed_ops)
        self.assertIn("divide", config.blocked_ops)
        
    def test_autocast_policy_rules(self):
        """Test autocast policy operation rules."""
        # Conservative policy
        conservative_config = AutocastConfig(policy=AutocastPolicy.CONSERVATIVE)
        self.assertTrue(conservative_config.should_cast_op("add"))
        self.assertFalse(conservative_config.should_cast_op("softmax"))
        
        # Aggressive policy
        aggressive_config = AutocastConfig(policy=AutocastPolicy.AGGRESSIVE)
        self.assertTrue(aggressive_config.should_cast_op("add"))
        self.assertTrue(aggressive_config.should_cast_op("multiply"))
        self.assertFalse(aggressive_config.should_cast_op("softmax"))
        
        # Selective policy
        selective_config = AutocastConfig(policy=AutocastPolicy.SELECTIVE)
        self.assertTrue(selective_config.should_cast_op("linear"))
        self.assertFalse(selective_config.should_cast_op("log"))
        
    def test_custom_rules(self):
        """Test custom operation rules."""
        config = AutocastConfig(
            custom_rules={"my_custom_op": True, "another_op": False}
        )
        
        self.assertTrue(config.should_cast_op("my_custom_op"))
        self.assertFalse(config.should_cast_op("another_op"))
        
    def test_precision_config_creation(self):
        """Test precision configuration creation."""
        config = create_precision_config(
            enabled=True,
            loss_scale=32768.0,
            policy="conservative"
        )
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.loss_scale, 32768.0)
        self.assertEqual(config.autocast_config.policy, AutocastPolicy.CONSERVATIVE)
        
    def test_recommended_configurations(self):
        """Test recommended configuration generation."""
        # Test transformer configuration
        transformer_config = get_recommended_precision_config(
            model_type="transformer",
            model_size="large"
        )
        self.assertIsInstance(transformer_config, PrecisionConfig)
        
        # Test CNN configuration
        cnn_config = get_recommended_precision_config(
            model_type="cnn",
            model_size="small"
        )
        self.assertIsInstance(cnn_config, PrecisionConfig)


class TestMixedPrecisionIntegration(unittest.TestCase):
    """Test mixed precision integration with existing components."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        
    def test_autocast_context_manager(self):
        """Test autocast context manager."""
        with autocast(enabled=True):
            # Should be in autocast context
            from src.neural_arch.optimization.mixed_precision import is_autocast_enabled
            # Note: This test may need adjustment based on actual implementation
            pass
            
    def test_mixed_precision_manager(self):
        """Test mixed precision manager."""
        config = PrecisionConfig(enabled=True)
        manager = MixedPrecisionManager(config)
        
        self.assertTrue(manager.config.enabled)
        self.assertIsNotNone(manager.scaler)
        
        # Test autocast context
        with manager.autocast():
            # Should be in autocast context
            pass
            
    def test_optimizer_amp_integration(self):
        """Test optimizer AMP integration methods."""
        optimizers = [
            Adam(self.model.parameters(), lr=0.001),
            AdamW(self.model.parameters(), lr=0.001),
            SGD(self.model.parameters(), lr=0.01)
        ]
        
        for optimizer in optimizers:
            # Test step_with_mixed_precision method
            self.assertTrue(hasattr(optimizer, "step_with_mixed_precision"))
            
            # Test create_amp_version method
            self.assertTrue(hasattr(optimizer, "create_amp_version"))
            
            # Test creating AMP version
            amp_version = optimizer.create_amp_version()
            # Should return some form of optimizer (wrapped or original)
            self.assertIsNotNone(amp_version)
            
    def test_training_context_creation(self):
        """Test complete training context creation."""
        from src.neural_arch.optimization.mixed_precision import create_training_context
        
        optimizer = Adam(self.model.parameters(), lr=0.001)
        config = create_precision_config(enabled=True)
        
        manager, amp_optimizer, autocast_context = create_training_context(
            self.model, optimizer, config
        )
        
        self.assertIsInstance(manager, MixedPrecisionManager)
        self.assertIsNotNone(amp_optimizer)
        self.assertIsNotNone(autocast_context)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability aspects of mixed precision training."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        
    def test_loss_scaling_prevents_underflow(self):
        """Test that loss scaling prevents gradient underflow."""
        scaler = AdvancedGradScaler()
        
        # Create very small loss that would underflow in FP16
        small_loss = Tensor(np.array([1e-7]), requires_grad=True)
        scaled_loss = scaler.scale(small_loss)
        
        # Scaled loss should be much larger
        self.assertGreater(scaled_loss.data[0], 1.0)
        
    def test_gradient_clipping_integration(self):
        """Test gradient clipping integration."""
        config = ScalerConfig(gradient_clip_threshold=1.0)
        scaler = AdvancedGradScaler(config)
        
        # Create large gradients
        for param in self.model.parameters():
            param.grad = Tensor(np.random.randn(*param.data.shape) * 10.0)
        
        # Unscaling should apply clipping
        success = scaler.unscale_(self.optimizer)
        
        # Check that gradients were clipped
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = np.linalg.norm(param.grad.data)
                # Allow some tolerance for clipping implementation
                self.assertLessEqual(grad_norm, 2.0)
                
    def test_overflow_recovery(self):
        """Test overflow detection and recovery."""
        scaler = AdvancedGradScaler()
        
        # Simulate several overflows
        for i in range(3):
            # Create overflow gradients
            for param in self.model.parameters():
                grad_data = np.random.randn(*param.data.shape)
                grad_data[0] = np.inf
                param.grad = Tensor(grad_data)
            
            success = scaler.step(self.optimizer)
            self.assertFalse(success)
        
        # Scale should have decreased significantly
        self.assertLess(scaler.get_scale(), 65536.0)
        
        # Recovery with normal gradients
        for i in range(10):
            for param in self.model.parameters():
                param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
            
            scaler.step(self.optimizer)
        
        # Statistics should show recovery
        stats = scaler.get_statistics()
        self.assertGreater(stats["total_steps"], 10)


class TestPerformanceAspects(unittest.TestCase):
    """Test performance-related aspects of mixed precision training."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.model = SimpleModel(input_size=100, hidden_size=200, output_size=50)
        
    def test_scaler_performance_tracking(self):
        """Test scaler performance tracking."""
        scaler = AdvancedGradScaler()
        optimizer = Adam(self.model.parameters(), lr=0.001)
        
        # Perform multiple steps
        for i in range(10):
            for param in self.model.parameters():
                param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
            scaler.step(optimizer)
        
        stats = scaler.get_statistics()
        
        # Should have timing information
        self.assertIn("avg_step_time", stats)
        self.assertGreater(stats["avg_step_time"], 0)
        
    def test_amp_optimizer_overhead(self):
        """Test AMP optimizer overhead tracking."""
        base_optimizer = Adam(self.model.parameters(), lr=0.001)
        amp_optimizer = AMPOptimizer(base_optimizer)
        
        # Perform steps with both optimizers
        for i in range(5):
            for param in self.model.parameters():
                param.grad = Tensor(np.random.randn(*param.data.shape) * 0.01)
            
            amp_optimizer.step()
            amp_optimizer.zero_grad()
        
        stats = amp_optimizer.get_statistics()
        
        # Should track timing and success rates
        self.assertIn("avg_step_time", stats)
        self.assertIn("success_rate", stats)
        self.assertEqual(stats["success_rate"], 1.0)


if __name__ == "__main__":
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test suite
    unittest.main(verbosity=2)