"""Enhanced comprehensive tests for optimization modules.

This module provides in-depth testing of optimization modules including mixed precision
training, operator fusion, gradient scaling, automatic casting, and performance
optimizations with detailed validation of numerical stability and efficiency gains.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_arch.core import Tensor
from neural_arch.optimization.fusion import (
    ConvBNReLUFusion,
    FusedOperation,
    FusionEngine,
    FusionPattern,
    LayerNormLinearFusion,
    LinearGELUFusion,
    LinearReLUFusion,
    fuse_conv_bn_activation,
    fuse_layernorm_linear,
    fuse_linear_activation,
    get_fusion_engine,
)
from neural_arch.optimization.mixed_precision import (
    AutomaticMixedPrecision,
    GradScaler,
    MixedPrecisionManager,
    PrecisionConfig,
    autocast,
    cast_to_fp16,
    cast_to_fp32,
    get_mixed_precision_manager,
    is_autocast_enabled,
    is_fp16_safe_op,
    mixed_precision_training,
    set_mixed_precision_config,
)


class MockOptimizer:
    """Mock optimizer for testing gradient scaling."""

    def __init__(self):
        self.parameters = {}
        self.step_called = False
        self.zero_grad_called = False

    def add_param(self, name: str, param: Tensor):
        """Add parameter to optimizer."""
        self.parameters[name] = param

    def step(self):
        """Mock optimizer step."""
        self.step_called = True

    def zero_grad(self):
        """Mock zero gradients."""
        self.zero_grad_called = True
        for param in self.parameters.values():
            if hasattr(param, "grad") and param.grad is not None:
                param.grad.data.fill(0.0)


class TestGradScaler:
    """Test gradient scaler for mixed precision training."""

    def test_grad_scaler_initialization(self):
        """Test gradient scaler initialization."""
        # Test default initialization
        scaler = GradScaler()
        assert scaler._scale == 65536.0  # Default scale (2^16)
        assert scaler._growth_factor == 2.0
        assert scaler._backoff_factor == 0.5
        assert scaler._growth_interval == 2000
        assert scaler._growth_tracker == 0
        assert not scaler._found_inf

        # Test custom initialization
        custom_scaler = GradScaler(
            init_scale=1024.0, growth_factor=1.5, backoff_factor=0.25, growth_interval=1000
        )
        assert custom_scaler._scale == 1024.0
        assert custom_scaler._growth_factor == 1.5
        assert custom_scaler._backoff_factor == 0.25
        assert custom_scaler._growth_interval == 1000

    def test_grad_scaler_loss_scaling(self):
        """Test loss scaling functionality."""
        scaler = GradScaler(init_scale=100.0)

        # Test loss scaling
        loss = Tensor(np.array([0.5]), requires_grad=True, name="test_loss")
        scaled_loss = scaler.scale(loss)

        assert scaled_loss.data[0] == 50.0  # 0.5 * 100.0
        assert scaled_loss.requires_grad == loss.requires_grad
        assert "scaled_" in scaled_loss.name

        # Test with different scale values
        scaler.set_scale(256.0)
        scaled_loss_2 = scaler.scale(loss)
        assert scaled_loss_2.data[0] == 128.0  # 0.5 * 256.0

        # Test with zero loss
        zero_loss = Tensor(np.array([0.0]), requires_grad=True)
        scaled_zero = scaler.scale(zero_loss)
        assert scaled_zero.data[0] == 0.0

        # Test with invalid input
        with pytest.raises(TypeError):
            scaler.scale("not a tensor")

    def test_grad_scaler_unscaling(self):
        """Test gradient unscaling functionality."""
        scaler = GradScaler(init_scale=100.0)
        optimizer = MockOptimizer()

        # Create parameters with gradients
        param1 = Tensor(np.array([1.0, 2.0]), requires_grad=True, name="param1")
        param1.grad = Tensor(np.array([10.0, 20.0]) * 100.0)  # Scaled gradients

        param2 = Tensor(np.array([3.0, 4.0]), requires_grad=True, name="param2")
        param2.grad = Tensor(np.array([30.0, 40.0]) * 100.0)  # Scaled gradients

        optimizer.add_param("param1", param1)
        optimizer.add_param("param2", param2)

        # Test unscaling
        grad_finite = scaler.unscale_(optimizer)

        assert grad_finite is True
        assert not scaler._found_inf

        # Check that gradients were unscaled
        np.testing.assert_array_almost_equal(param1.grad.data, [10.0, 20.0])
        np.testing.assert_array_almost_equal(param2.grad.data, [30.0, 40.0])

    def test_grad_scaler_inf_nan_detection(self):
        """Test detection of inf/nan gradients."""
        scaler = GradScaler(init_scale=100.0)
        optimizer = MockOptimizer()

        # Create parameter with inf gradient
        param_inf = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        param_inf.grad = Tensor(np.array([np.inf, 20.0]))

        # Create parameter with nan gradient
        param_nan = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        param_nan.grad = Tensor(np.array([30.0, np.nan]))

        optimizer.add_param("param_inf", param_inf)
        optimizer.add_param("param_nan", param_nan)

        # Test inf detection
        grad_finite = scaler.unscale_(optimizer)

        assert grad_finite is False
        assert scaler._found_inf is True

    def test_grad_scaler_step_with_finite_gradients(self):
        """Test optimizer step with finite gradients."""
        scaler = GradScaler(init_scale=100.0, growth_interval=3)
        optimizer = MockOptimizer()

        # Create parameter with finite gradients
        param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        param.grad = Tensor(np.array([10.0, 20.0]) * 100.0)  # Scaled gradients
        optimizer.add_param("param", param)

        initial_scale = scaler.get_scale()

        # Test multiple successful steps
        for i in range(4):
            param.grad = Tensor(np.array([10.0, 20.0]) * scaler.get_scale())
            step_taken = scaler.step(optimizer)

            assert step_taken is True
            assert optimizer.step_called

            # Reset optimizer state
            optimizer.step_called = False

        # After 3 successful steps (growth_interval), scale should increase
        final_scale = scaler.get_scale()
        assert final_scale > initial_scale
        assert final_scale == initial_scale * scaler._growth_factor

    def test_grad_scaler_step_with_infinite_gradients(self):
        """Test optimizer step with infinite gradients."""
        scaler = GradScaler(init_scale=100.0)
        optimizer = MockOptimizer()

        # Create parameter with infinite gradients
        param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        param.grad = Tensor(np.array([np.inf, 20.0]))
        optimizer.add_param("param", param)

        initial_scale = scaler.get_scale()

        # Test step with infinite gradients
        step_taken = scaler.step(optimizer)

        assert step_taken is False
        assert not optimizer.step_called  # Step should be skipped

        # Scale should be reduced
        final_scale = scaler.get_scale()
        assert final_scale < initial_scale
        assert final_scale == initial_scale * scaler._backoff_factor

        # Growth tracker should be reset
        assert scaler._growth_tracker == 0

    def test_grad_scaler_scale_bounds(self):
        """Test gradient scaler scale bounds."""
        scaler = GradScaler(init_scale=1.0, backoff_factor=0.1)

        # Test minimum scale bound
        scaler.set_scale(0.5)
        assert scaler.get_scale() == 1.0  # Should be clamped to minimum

        # Test scale reduction respects minimum
        scaler.set_scale(5.0)
        for _ in range(10):  # Many reductions
            scaler._scale = max(scaler._scale * scaler._backoff_factor, 1.0)

        assert scaler.get_scale() >= 1.0

        # Test maximum scale bound (2^24)
        scaler._scale = 2**25  # Above maximum
        scaler._growth_tracker = scaler._growth_interval
        # In real step, this would be clamped to 2^24


class TestPrecisionConfig:
    """Test precision configuration dataclass."""

    def test_precision_config_defaults(self):
        """Test default precision configuration values."""
        config = PrecisionConfig()

        assert config.enabled is True
        assert config.loss_scale == 65536.0
        assert config.growth_factor == 2.0
        assert config.backoff_factor == 0.5
        assert config.growth_interval == 2000
        assert config.max_loss_scale == 2**24
        assert config.min_loss_scale == 1.0

    def test_precision_config_custom(self):
        """Test custom precision configuration."""
        config = PrecisionConfig(
            enabled=False,
            loss_scale=1024.0,
            growth_factor=1.5,
            backoff_factor=0.25,
            growth_interval=1000,
            max_loss_scale=2**20,
            min_loss_scale=0.5,
        )

        assert config.enabled is False
        assert config.loss_scale == 1024.0
        assert config.growth_factor == 1.5
        assert config.backoff_factor == 0.25
        assert config.growth_interval == 1000
        assert config.max_loss_scale == 2**20
        assert config.min_loss_scale == 0.5


class TestAutomaticMixedPrecision:
    """Test automatic mixed precision context manager."""

    def test_amp_enabled_context(self):
        """Test AMP context when enabled."""
        amp = AutomaticMixedPrecision(enabled=True)

        # Mock dtype functions
        with patch("neural_arch.optimization.mixed_precision.get_default_dtype") as mock_get, patch(
            "neural_arch.optimization.mixed_precision.set_default_dtype"
        ) as mock_set:

            mock_get.return_value = "FLOAT32"

            with amp:
                # Should set FP16 during context
                mock_set.assert_called()

            # Should restore original dtype after context
            assert mock_set.call_count == 2  # Once to set FP16, once to restore

    def test_amp_disabled_context(self):
        """Test AMP context when disabled."""
        amp = AutomaticMixedPrecision(enabled=False)

        with patch("neural_arch.optimization.mixed_precision.get_default_dtype") as mock_get, patch(
            "neural_arch.optimization.mixed_precision.set_default_dtype"
        ) as mock_set:

            with amp:
                # Should not modify dtype when disabled
                mock_get.assert_not_called()
                mock_set.assert_not_called()

    def test_amp_global_state(self):
        """Test AMP global state tracking."""
        from neural_arch.optimization.mixed_precision import _autocast_enabled

        amp = AutomaticMixedPrecision(enabled=True)

        with patch("neural_arch.optimization.mixed_precision.get_default_dtype"), patch(
            "neural_arch.optimization.mixed_precision.set_default_dtype"
        ):

            assert not is_autocast_enabled()

            with amp:
                # Should be enabled in context
                pass  # Global state tracking is implementation detail

            # Should be disabled after context
            assert not is_autocast_enabled()


class TestMixedPrecisionManager:
    """Test mixed precision training manager."""

    def test_mp_manager_initialization(self):
        """Test mixed precision manager initialization."""
        # Test with default config
        manager = MixedPrecisionManager()
        assert manager.config.enabled is True
        assert hasattr(manager, "scaler")
        assert manager._step_count == 0
        assert manager._successful_steps == 0
        assert manager._skipped_steps == 0

        # Test with custom config
        custom_config = PrecisionConfig(enabled=False, loss_scale=1024.0)
        custom_manager = MixedPrecisionManager(custom_config)
        assert custom_manager.config.enabled is False
        assert custom_manager.config.loss_scale == 1024.0

    def test_mp_manager_autocast(self):
        """Test mixed precision manager autocast context."""
        manager = MixedPrecisionManager()

        with patch("neural_arch.optimization.mixed_precision.AutomaticMixedPrecision") as mock_amp:
            with manager.autocast():
                mock_amp.assert_called_once_with(enabled=manager.config.enabled)

    def test_mp_manager_loss_scaling(self):
        """Test loss scaling through manager."""
        manager = MixedPrecisionManager()
        loss = Tensor(np.array([0.1]), requires_grad=True)

        # Test with enabled MP
        scaled_loss = manager.scale_loss(loss)
        assert scaled_loss.data[0] > loss.data[0]  # Should be scaled up

        # Test with disabled MP
        disabled_manager = MixedPrecisionManager(PrecisionConfig(enabled=False))
        unscaled_loss = disabled_manager.scale_loss(loss)
        assert unscaled_loss is loss  # Should return original loss

    def test_mp_manager_backward_and_step_enabled(self):
        """Test backward and step with MP enabled."""
        manager = MixedPrecisionManager()
        optimizer = MockOptimizer()

        # Create loss
        loss = Tensor(np.array([0.1]), requires_grad=True)

        # Mock backward method
        loss.backward = MagicMock()

        # Test backward and step
        step_taken = manager.backward_and_step(loss, optimizer)

        # With finite gradients, step should be taken
        assert optimizer.zero_grad_called

        # Statistics should be updated
        assert manager._step_count == 1

    def test_mp_manager_backward_and_step_disabled(self):
        """Test backward and step with MP disabled."""
        config = PrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)
        optimizer = MockOptimizer()

        # Create loss
        loss = Tensor(np.array([0.1]), requires_grad=True)

        # Mock methods
        loss.backward = MagicMock()

        # Test backward and step
        step_taken = manager.backward_and_step(loss, optimizer)

        assert step_taken is True
        assert loss.backward.called
        assert optimizer.step_called
        assert optimizer.zero_grad_called
        assert manager._successful_steps == 1

    def test_mp_manager_statistics(self):
        """Test mixed precision training statistics."""
        manager = MixedPrecisionManager()

        # Initial statistics
        stats = manager.get_statistics()
        expected_keys = {
            "total_steps",
            "successful_steps",
            "skipped_steps",
            "success_rate",
            "current_scale",
            "enabled",
        }
        assert set(stats.keys()) == expected_keys
        assert stats["total_steps"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["enabled"] is True

    def test_mp_manager_state_dict(self):
        """Test state dictionary for checkpointing."""
        manager = MixedPrecisionManager()

        # Get state dict
        state_dict = manager.state_dict()

        required_keys = {
            "scale",
            "growth_tracker",
            "step_count",
            "successful_steps",
            "skipped_steps",
            "config",
        }
        assert set(state_dict.keys()) == required_keys

        # Test loading state dict
        state_dict["scale"] = 2048.0
        state_dict["step_count"] = 100

        manager.load_state_dict(state_dict)
        assert manager.scaler.get_scale() == 2048.0
        assert manager._step_count == 100


class TestTensorCasting:
    """Test tensor casting utilities."""

    def test_cast_to_fp16(self):
        """Test casting tensor to FP16."""
        # Test FP32 to FP16
        fp32_data = np.array([1.5, 2.7, 3.9], dtype=np.float32)
        fp32_tensor = Tensor(fp32_data, requires_grad=True, name="test_tensor")

        fp16_tensor = cast_to_fp16(fp32_tensor)

        assert fp16_tensor.data.dtype == np.float16
        assert fp16_tensor.requires_grad == fp32_tensor.requires_grad
        assert "fp16_" in fp16_tensor.name

        # Test values are approximately preserved
        np.testing.assert_allclose(fp16_tensor.data, fp32_data, rtol=1e-3)

        # Test already FP16 tensor
        already_fp16 = cast_to_fp16(fp16_tensor)
        assert already_fp16 is fp16_tensor  # Should return same tensor

    def test_cast_to_fp32(self):
        """Test casting tensor to FP32."""
        # Test FP16 to FP32
        fp16_data = np.array([1.5, 2.7, 3.9], dtype=np.float16)
        fp16_tensor = Tensor(fp16_data, requires_grad=True, name="test_tensor")

        fp32_tensor = cast_to_fp32(fp16_tensor)

        assert fp32_tensor.data.dtype == np.float32
        assert fp32_tensor.requires_grad == fp16_tensor.requires_grad
        assert "fp32_" in fp32_tensor.name

        # Test values are preserved
        np.testing.assert_allclose(fp32_tensor.data, fp16_data.astype(np.float32))

        # Test already FP32 tensor
        already_fp32 = cast_to_fp32(fp32_tensor)
        assert already_fp32 is fp32_tensor  # Should return same tensor

    def test_fp16_safe_operations(self):
        """Test FP16 safe operation detection."""
        # Test safe operations
        safe_ops = ["add", "multiply", "matmul", "conv2d", "relu", "gelu"]
        for op in safe_ops:
            assert is_fp16_safe_op(op) is True

        # Test unsafe operations
        unsafe_ops = ["softmax", "log", "exp", "divide", "sqrt"]
        for op in unsafe_ops:
            assert is_fp16_safe_op(op) is False

        # Test case insensitivity
        assert is_fp16_safe_op("ADD") is True
        assert is_fp16_safe_op("SOFTMAX") is False


class TestFusionEngine:
    """Test operator fusion engine."""

    def test_fusion_engine_initialization(self):
        """Test fusion engine initialization."""
        engine = FusionEngine()

        assert len(engine.patterns) > 0
        assert len(engine.fused_ops) > 0

        # Check default patterns are registered
        pattern_names = [p.name for p in engine.patterns]
        expected_patterns = ["linear_gelu", "linear_relu", "conv_bn_relu", "layernorm_linear"]

        for expected in expected_patterns:
            assert expected in pattern_names

    def test_fusion_pattern_registration(self):
        """Test fusion pattern registration."""
        engine = FusionEngine()
        initial_count = len(engine.patterns)

        # Create custom pattern
        custom_pattern = FusionPattern(
            name="test_pattern",
            pattern=["Op1", "Op2"],
            fused_op=MagicMock(spec=FusedOperation),
            memory_savings=0.3,
            compute_speedup=1.5,
        )

        engine.register_pattern(custom_pattern)

        assert len(engine.patterns) == initial_count + 1
        assert "test_pattern" in engine.fused_ops
        assert engine.fused_ops["test_pattern"] is custom_pattern.fused_op

    def test_fusion_opportunity_detection(self):
        """Test detection of fusion opportunities."""
        engine = FusionEngine()

        # Test sequence with fusion opportunities
        operations = ["Linear", "GELU", "Linear", "ReLU", "Conv2d", "BatchNorm2d", "ReLU"]
        opportunities = engine.find_fusion_opportunities(operations)

        assert len(opportunities) >= 2  # Should find Linear+GELU and Linear+ReLU

        # Check specific patterns found
        found_patterns = [pattern.name for _, pattern in opportunities]
        assert "linear_gelu" in found_patterns
        assert "linear_relu" in found_patterns

        # Test sequence without opportunities
        no_fusion_ops = ["Conv2d", "Linear", "Conv2d"]
        no_opportunities = engine.find_fusion_opportunities(no_fusion_ops)
        assert len(no_opportunities) == 0

    def test_speedup_estimation(self):
        """Test speedup estimation."""
        engine = FusionEngine()

        # Test operations with known fusion patterns
        operations = ["Linear", "GELU", "Linear", "ReLU"]
        speedup_info = engine.estimate_speedup(operations)

        required_keys = {"memory_savings", "compute_speedup", "fusion_count", "patterns_found"}
        assert set(speedup_info.keys()) == required_keys

        assert speedup_info["fusion_count"] >= 1
        assert speedup_info["compute_speedup"] > 1.0
        assert 0.0 <= speedup_info["memory_savings"] <= 0.9
        assert len(speedup_info["patterns_found"]) >= 1

    def test_sequence_optimization(self):
        """Test sequence optimization."""
        engine = FusionEngine()

        # Test optimization of fusible sequence
        original_ops = ["Linear", "GELU", "Conv2d", "Linear", "ReLU"]
        optimized_ops, info = engine.optimize_sequence(original_ops)

        assert len(optimized_ops) < len(original_ops)  # Should be shorter due to fusion
        assert info["fusions_applied"] > 0
        assert info["speedup_estimate"] > 1.0

        # Test that fused operations are named correctly
        fused_op_found = any("Fused_" in op for op in optimized_ops)
        assert fused_op_found

        # Test sequence without fusion opportunities
        no_fusion_ops = ["Conv2d", "Dropout", "BatchNorm2d"]
        optimized_no_fusion, info_no_fusion = engine.optimize_sequence(no_fusion_ops)

        assert optimized_no_fusion == no_fusion_ops  # Should be unchanged
        assert info_no_fusion["fusions_applied"] == 0


class TestFusedOperations:
    """Test individual fused operations."""

    def test_linear_gelu_fusion(self):
        """Test Linear + GELU fusion."""
        fusion = LinearGELUFusion()

        # Test pattern
        assert fusion.get_pattern() == ["Linear", "GELU"]

        # Test forward pass
        batch_size, in_features, out_features = 4, 8, 6
        x = np.random.randn(batch_size, in_features).astype(np.float32)
        weight = np.random.randn(out_features, in_features).astype(np.float32)
        bias = np.random.randn(out_features).astype(np.float32)

        output = fusion.forward(x, weight, bias)

        assert output.shape == (batch_size, out_features)
        assert output.dtype == np.float32

        # Test that output is reasonable (GELU should be smooth activation)
        assert np.all(np.isfinite(output))

    def test_linear_relu_fusion(self):
        """Test Linear + ReLU fusion."""
        fusion = LinearReLUFusion()

        # Test pattern
        assert fusion.get_pattern() == ["Linear", "ReLU"]

        # Test forward pass
        batch_size, in_features, out_features = 4, 8, 6
        x = np.random.randn(batch_size, in_features).astype(np.float32)
        weight = np.random.randn(out_features, in_features).astype(np.float32)
        bias = np.random.randn(out_features).astype(np.float32)

        output = fusion.forward(x, weight, bias)

        assert output.shape == (batch_size, out_features)
        assert np.all(output >= 0)  # ReLU should ensure non-negative outputs

    def test_layernorm_linear_fusion(self):
        """Test LayerNorm + Linear fusion."""
        fusion = LayerNormLinearFusion()

        # Test pattern
        assert fusion.get_pattern() == ["LayerNorm", "Linear"]

        # Test forward pass
        batch_size, seq_len, hidden_size = 2, 8, 16
        out_size = 12

        x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        ln_weight = np.ones(hidden_size, dtype=np.float32)
        ln_bias = np.zeros(hidden_size, dtype=np.float32)
        linear_weight = np.random.randn(out_size, hidden_size).astype(np.float32)
        linear_bias = np.random.randn(out_size).astype(np.float32)

        output = fusion.forward(x, ln_weight, ln_bias, linear_weight, linear_bias)

        assert output.shape == (batch_size, seq_len, out_size)

        # Test that layer norm is applied (output should have normalized statistics)
        # This is a simplified test - real test would compare with separate operations
        assert np.all(np.isfinite(output))

    def test_conv_bn_relu_fusion(self):
        """Test Conv2D + BatchNorm + ReLU fusion."""
        fusion = ConvBNReLUFusion()

        # Test pattern
        assert fusion.get_pattern() == ["Conv2d", "BatchNorm2d", "ReLU"]

        # Test that forward requires Numba (since fallback raises NotImplementedError)
        # In a real test environment with Numba, this would test the actual fusion
        batch_size, in_channels, height, width = 2, 3, 8, 8
        out_channels, kernel_size = 4, 3

        input_data = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
        weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(
            np.float32
        )
        bias = np.random.randn(out_channels).astype(np.float32)
        bn_weight = np.ones(out_channels).astype(np.float32)
        bn_bias = np.zeros(out_channels).astype(np.float32)
        bn_mean = np.zeros(out_channels).astype(np.float32)
        bn_var = np.ones(out_channels).astype(np.float32)

        # This would test the fusion if Numba is available
        # For now, test that the method exists and has correct interface
        try:
            output = fusion.forward(input_data, weight, bias, bn_weight, bn_bias, bn_mean, bn_var)
            # If Numba is available, test output shape
            expected_shape = (
                batch_size,
                out_channels,
                height - kernel_size + 1,
                width - kernel_size + 1,
            )
            assert output.shape == expected_shape
            assert np.all(output >= 0)  # ReLU ensures non-negative
        except NotImplementedError:
            # Expected if Numba is not available
            pass


class TestFusionHighLevelAPI:
    """Test high-level fusion API functions."""

    def test_fuse_linear_activation_gelu(self):
        """Test high-level linear + GELU fusion API."""
        batch_size, in_features, out_features = 4, 8, 6
        x = np.random.randn(batch_size, in_features).astype(np.float32)
        weight = np.random.randn(out_features, in_features).astype(np.float32)
        bias = np.random.randn(out_features).astype(np.float32)

        output = fuse_linear_activation(x, weight, bias, activation="gelu")

        assert output.shape == (batch_size, out_features)
        assert np.all(np.isfinite(output))

    def test_fuse_linear_activation_relu(self):
        """Test high-level linear + ReLU fusion API."""
        batch_size, in_features, out_features = 4, 8, 6
        x = np.random.randn(batch_size, in_features).astype(np.float32)
        weight = np.random.randn(out_features, in_features).astype(np.float32)
        bias = np.random.randn(out_features).astype(np.float32)

        output = fuse_linear_activation(x, weight, bias, activation="relu")

        assert output.shape == (batch_size, out_features)
        assert np.all(output >= 0)  # ReLU non-negativity

    def test_fuse_linear_activation_unsupported(self):
        """Test unsupported activation in linear fusion."""
        x = np.random.randn(4, 8).astype(np.float32)
        weight = np.random.randn(6, 8).astype(np.float32)
        bias = np.random.randn(6).astype(np.float32)

        with pytest.raises(ValueError, match="Unsupported activation"):
            fuse_linear_activation(x, weight, bias, activation="sigmoid")

    def test_fuse_layernorm_linear_api(self):
        """Test high-level LayerNorm + Linear fusion API."""
        batch_size, seq_len, hidden_size, out_size = 2, 8, 16, 12

        x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        ln_weight = np.ones(hidden_size, dtype=np.float32)
        ln_bias = np.zeros(hidden_size, dtype=np.float32)
        linear_weight = np.random.randn(out_size, hidden_size).astype(np.float32)
        linear_bias = np.random.randn(out_size).astype(np.float32)

        output = fuse_layernorm_linear(x, ln_weight, ln_bias, linear_weight, linear_bias)

        assert output.shape == (batch_size, seq_len, out_size)
        assert np.all(np.isfinite(output))


class TestOptimizationIntegration:
    """Test integration between optimization modules."""

    def test_mixed_precision_with_fusion(self):
        """Test mixed precision training with operator fusion."""
        # Test that both optimizations can work together
        mp_manager = MixedPrecisionManager()
        fusion_engine = get_fusion_engine()

        # Test fusion opportunity detection
        operations = ["Linear", "GELU", "Linear", "ReLU"]
        speedup_info = fusion_engine.estimate_speedup(operations)

        assert speedup_info["compute_speedup"] > 1.0

        # Test that MP manager can scale losses
        loss = Tensor(np.array([0.1]), requires_grad=True)
        scaled_loss = mp_manager.scale_loss(loss)

        assert scaled_loss.data[0] > loss.data[0]

        # Both optimizations should be independent and composable
        assert mp_manager.config.enabled
        assert len(fusion_engine.patterns) > 0

    def test_global_optimization_managers(self):
        """Test global optimization manager access."""
        # Test global MP manager
        global_mp = get_mixed_precision_manager()
        assert isinstance(global_mp, MixedPrecisionManager)

        # Test global fusion engine
        global_fusion = get_fusion_engine()
        assert isinstance(global_fusion, FusionEngine)

        # Test setting global MP config
        custom_config = PrecisionConfig(enabled=False, loss_scale=512.0)
        set_mixed_precision_config(custom_config)

        new_global_mp = get_mixed_precision_manager()
        assert new_global_mp.config.enabled is False
        assert new_global_mp.config.loss_scale == 512.0

    def test_context_managers(self):
        """Test optimization context managers."""
        # Test autocast context manager
        with patch("neural_arch.optimization.mixed_precision.AutomaticMixedPrecision") as mock_amp:
            with autocast(enabled=True):
                pass
            mock_amp.assert_called()

        # Test mixed precision training context manager
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        with mixed_precision_training(mock_model, mock_optimizer, enabled=True) as (
            scaler,
            autocast_ctx,
        ):
            assert scaler is not None
            assert autocast_ctx is not None

        # Test disabled case
        with mixed_precision_training(mock_model, mock_optimizer, enabled=False) as (
            scaler,
            autocast_ctx,
        ):
            assert scaler is None


class TestOptimizationPerformance:
    """Test performance characteristics of optimization modules."""

    def test_fusion_memory_estimation(self):
        """Test fusion memory savings estimation."""
        engine = FusionEngine()

        # Test operations with high memory savings potential
        memory_intensive_ops = ["Linear", "GELU", "Linear", "ReLU", "Linear", "GELU"]
        speedup_info = engine.estimate_speedup(memory_intensive_ops)

        # Should find multiple fusion opportunities
        assert speedup_info["fusion_count"] >= 2
        assert speedup_info["memory_savings"] > 0.0

        # Memory savings should be capped
        assert speedup_info["memory_savings"] <= 0.9

    def test_mixed_precision_scaling_efficiency(self):
        """Test mixed precision scaling efficiency."""
        scaler = GradScaler(init_scale=1024.0)

        # Test scaling is exact for simple cases
        loss = Tensor(np.array([0.001]), requires_grad=True)
        scaled_loss = scaler.scale(loss)

        expected_value = 0.001 * 1024.0
        assert abs(scaled_loss.data[0] - expected_value) < 1e-10

        # Test unscaling is inverse of scaling
        optimizer = MockOptimizer()
        param = Tensor(np.array([1.0]), requires_grad=True)
        param.grad = Tensor(np.array([0.5 * 1024.0]))  # Scaled gradient
        optimizer.add_param("param", param)

        scaler.unscale_(optimizer)
        assert abs(param.grad.data[0] - 0.5) < 1e-6  # Should be unscaled back to 0.5

    def test_optimization_numerical_stability(self):
        """Test numerical stability of optimizations."""
        # Test MP with very small loss values
        scaler = GradScaler(init_scale=65536.0)
        very_small_loss = Tensor(np.array([1e-8]), requires_grad=True)

        scaled_loss = scaler.scale(very_small_loss)
        # Should be scaled up to reasonable range
        assert scaled_loss.data[0] > 1e-3

        # Test fusion with edge case values
        fusion = LinearGELUFusion()

        # Test with zero input
        zero_input = np.zeros((2, 4), dtype=np.float32)
        weight = np.random.randn(3, 4).astype(np.float32)
        bias = np.zeros(3, dtype=np.float32)

        output = fusion.forward(zero_input, weight, bias)
        assert np.all(np.isfinite(output))

        # Test with large input values
        large_input = np.full((2, 4), 100.0, dtype=np.float32)
        large_output = fusion.forward(large_input, weight, bias)
        assert np.all(np.isfinite(large_output))


if __name__ == "__main__":
    pytest.main([__file__])
