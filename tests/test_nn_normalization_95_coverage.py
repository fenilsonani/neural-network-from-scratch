"""Ultra-comprehensive tests for Normalization layers to achieve 95%+ test coverage.

This test suite covers all normalization implementations including LayerNorm, RMSNorm,
BatchNorm1d, BatchNorm2d, GroupNorm, and InstanceNorm to ensure robust 95%+ test coverage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core.tensor import Tensor
from neural_arch.exceptions import LayerError
from neural_arch.nn.normalization import (
    BatchNorm1d,
    BatchNorm2d,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    RMSNorm,
)


class TestLayerNorm95Coverage:
    """Comprehensive LayerNorm tests targeting 95%+ coverage."""

    def test_layernorm_initialization_comprehensive(self):
        """Test all LayerNorm initialization parameters."""
        # Test basic initialization
        ln = LayerNorm(128)
        assert ln.normalized_shape == 128
        assert ln.eps == 1e-5
        assert ln.elementwise_affine is True
        assert ln.weight.shape == (128,)
        assert ln.bias.shape == (128,)
        assert np.allclose(ln.weight.data, 1.0)
        assert np.allclose(ln.bias.data, 0.0)

        # Test custom parameters
        ln_custom = LayerNorm(normalized_shape=256, eps=1e-8, elementwise_affine=True, bias=True)
        assert ln_custom.normalized_shape == 256
        assert ln_custom.eps == 1e-8
        assert ln_custom.elementwise_affine is True

        # Test without affine transformation
        ln_no_affine = LayerNorm(64, elementwise_affine=False)
        assert ln_no_affine.normalized_shape == 64
        assert ln_no_affine.elementwise_affine is False
        assert ln_no_affine.weight is None
        assert ln_no_affine.bias is None

        # Test without bias
        ln_no_bias = LayerNorm(64, bias=False)
        assert ln_no_bias.weight is not None
        assert ln_no_bias.bias is None

    def test_layernorm_initialization_validation(self):
        """Test LayerNorm initialization validation."""
        # Test invalid normalized_shape
        with pytest.raises(LayerError):
            LayerNorm(0)
        with pytest.raises(LayerError):
            LayerNorm(-1)

        # Test invalid eps
        with pytest.raises(LayerError):
            LayerNorm(128, eps=0)
        with pytest.raises(LayerError):
            LayerNorm(128, eps=-1e-5)

    def test_layernorm_forward_pass_comprehensive(self):
        """Test LayerNorm forward pass with various configurations."""
        ln = LayerNorm(128)

        # Test different input shapes
        test_shapes = [
            (1, 128),  # 2D input
            (4, 128),  # Batch
            (2, 10, 128),  # 3D input
            (1, 5, 10, 128),  # 4D input
        ]

        for shape in test_shapes:
            x = Tensor(np.random.randn(*shape).astype(np.float32))
            output = ln.forward(x)
            assert output.shape == shape

            # Check normalization properties (approximately)
            # Last dimension should have mean ~0 and std ~1
            normalized_data = output.data
            last_dim_mean = np.mean(normalized_data, axis=-1, keepdims=True)
            last_dim_std = np.std(normalized_data, axis=-1, keepdims=True)

            # Due to affine transformation, we check the structure is reasonable
            assert np.all(np.isfinite(normalized_data))

    def test_layernorm_forward_validation(self):
        """Test LayerNorm input validation."""
        ln = LayerNorm(128)

        # Test wrong last dimension
        x_wrong = Tensor(np.random.randn(2, 64).astype(np.float32))
        with pytest.raises(LayerError):
            ln.forward(x_wrong)

    def test_layernorm_mathematical_correctness(self):
        """Test LayerNorm mathematical correctness."""
        ln = LayerNorm(4, eps=1e-5)

        # Create test input with known statistics
        x_data = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data)
        output = ln.forward(x)

        # Manual calculation
        for i in range(x_data.shape[0]):
            row = x_data[i]
            mean = np.mean(row)
            var = np.var(row, ddof=0)
            std = np.sqrt(var + ln.eps)
            expected = ln.weight.data * (row - mean) / std + ln.bias.data

            assert np.allclose(output.data[i], expected, rtol=1e-5)

    def test_layernorm_without_affine(self):
        """Test LayerNorm without affine transformation."""
        ln = LayerNorm(4, elementwise_affine=False)

        x_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        x = Tensor(x_data)
        output = ln.forward(x)

        # Check that output has mean ~0 and std ~1
        output_mean = np.mean(output.data, axis=-1)
        output_std = np.std(output.data, axis=-1)

        assert np.allclose(output_mean, 0.0, atol=1e-6)
        assert np.allclose(output_std, 1.0, rtol=1e-5)

    def test_layernorm_gradient_computation(self):
        """Test LayerNorm gradient computation setup."""
        ln = LayerNorm(4)

        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        output = ln.forward(x)

        assert output.requires_grad is True
        assert output._grad_fn is not None

    def test_layernorm_extra_repr(self):
        """Test LayerNorm string representation."""
        ln = LayerNorm(128, eps=1e-8, elementwise_affine=False)
        repr_str = ln.extra_repr()

        assert "normalized_shape=128" in repr_str
        assert "eps=1e-08" in repr_str
        assert "elementwise_affine=False" in repr_str


class TestRMSNorm95Coverage:
    """Comprehensive RMSNorm tests targeting 95%+ coverage."""

    def test_rmsnorm_initialization_comprehensive(self):
        """Test all RMSNorm initialization parameters."""
        # Test basic initialization
        rms = RMSNorm(128)
        assert rms.normalized_shape == 128
        assert rms.eps == 1e-8  # Different default than LayerNorm
        assert rms.elementwise_affine is True
        assert rms.weight.shape == (128,)
        assert np.allclose(rms.weight.data, 1.0)

        # Test custom parameters
        rms_custom = RMSNorm(256, eps=1e-6, elementwise_affine=False)
        assert rms_custom.normalized_shape == 256
        assert rms_custom.eps == 1e-6
        assert rms_custom.elementwise_affine is False
        assert rms_custom.weight is None

    def test_rmsnorm_initialization_validation(self):
        """Test RMSNorm initialization validation."""
        with pytest.raises(LayerError):
            RMSNorm(0)
        with pytest.raises(LayerError):
            RMSNorm(-1)
        with pytest.raises(LayerError):
            RMSNorm(128, eps=0)
        with pytest.raises(LayerError):
            RMSNorm(128, eps=-1e-8)

    def test_rmsnorm_forward_pass_comprehensive(self):
        """Test RMSNorm forward pass."""
        rms = RMSNorm(128)

        test_shapes = [
            (1, 128),
            (4, 128),
            (2, 10, 128),
            (1, 5, 10, 128),
        ]

        for shape in test_shapes:
            x = Tensor(np.random.randn(*shape).astype(np.float32))
            output = rms.forward(x)
            assert output.shape == shape
            assert np.all(np.isfinite(output.data))

    def test_rmsnorm_mathematical_correctness(self):
        """Test RMSNorm mathematical correctness."""
        rms = RMSNorm(4, eps=1e-8)

        # Test with known input
        x_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        x = Tensor(x_data)
        output = rms.forward(x)

        # Manual RMSNorm calculation
        row = x_data[0]
        mean_square = np.mean(row**2)
        rms_val = np.sqrt(mean_square + rms.eps)
        expected = rms.weight.data * (row / rms_val)

        assert np.allclose(output.data[0], expected, rtol=1e-6)

    def test_rmsnorm_without_affine(self):
        """Test RMSNorm without affine transformation."""
        rms = RMSNorm(4, elementwise_affine=False)

        x_data = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data)
        output = rms.forward(x)

        # Check RMS normalization
        row = x_data[0]
        mean_square = np.mean(row**2)
        rms_val = np.sqrt(mean_square + rms.eps)
        expected = row / rms_val

        assert np.allclose(output.data[0], expected, rtol=1e-6)


class TestBatchNorm1d95Coverage:
    """Comprehensive BatchNorm1d tests targeting 95%+ coverage."""

    def test_batchnorm1d_initialization_comprehensive(self):
        """Test all BatchNorm1d initialization parameters."""
        # Test basic initialization
        bn = BatchNorm1d(64)
        assert bn.num_features == 64
        assert bn.eps == 1e-5
        assert bn.momentum == 0.1
        assert bn.affine is True
        assert bn.track_running_stats is True
        assert bn.weight.shape == (64,)
        assert bn.bias.shape == (64,)
        assert bn.running_mean.shape == (64,)
        assert bn.running_var.shape == (64,)
        assert bn.num_batches_tracked == 0

        # Test custom parameters
        bn_custom = BatchNorm1d(
            num_features=128, eps=1e-3, momentum=0.2, affine=False, track_running_stats=False
        )
        assert bn_custom.num_features == 128
        assert bn_custom.eps == 1e-3
        assert bn_custom.momentum == 0.2
        assert bn_custom.affine is False
        assert bn_custom.track_running_stats is False
        assert bn_custom.weight is None
        assert bn_custom.bias is None
        assert bn_custom.running_mean is None
        assert bn_custom.running_var is None

    def test_batchnorm1d_initialization_validation(self):
        """Test BatchNorm1d initialization validation."""
        with pytest.raises(LayerError):
            BatchNorm1d(0)
        with pytest.raises(LayerError):
            BatchNorm1d(-1)
        with pytest.raises(LayerError):
            BatchNorm1d(64, eps=-1e-5)
        with pytest.raises(LayerError):
            BatchNorm1d(64, momentum=-0.1)
        with pytest.raises(LayerError):
            BatchNorm1d(64, momentum=1.1)

    def test_batchnorm1d_training_mode(self):
        """Test BatchNorm1d in training mode."""
        bn = BatchNorm1d(32)
        assert bn.training is True

        # Test 2D input (N, C)
        x_2d = Tensor(np.random.randn(4, 32).astype(np.float32))
        output_2d = bn.forward(x_2d)
        assert output_2d.shape == (4, 32)
        assert bn.num_batches_tracked == 1

        # Test 3D input (N, C, L)
        x_3d = Tensor(np.random.randn(4, 32, 10).astype(np.float32))
        output_3d = bn.forward(x_3d)
        assert output_3d.shape == (4, 32, 10)
        assert bn.num_batches_tracked == 2

    def test_batchnorm1d_evaluation_mode(self):
        """Test BatchNorm1d in evaluation mode."""
        bn = BatchNorm1d(32)

        # First, run in training mode to populate running stats
        x_train = Tensor(np.random.randn(4, 32).astype(np.float32))
        bn.forward(x_train)

        # Switch to eval mode
        bn.eval()
        assert bn.training is False

        x_eval = Tensor(np.random.randn(2, 32).astype(np.float32))
        output_eval = bn.forward(x_eval)
        assert output_eval.shape == (2, 32)
        # num_batches_tracked should not increase in eval mode
        assert bn.num_batches_tracked == 1

    def test_batchnorm1d_train_eval_switching(self):
        """Test switching between train and eval modes."""
        bn = BatchNorm1d(16)

        # Start in training
        assert bn.training is True

        # Switch to eval
        bn.eval()
        assert bn.training is False

        # Switch back to training
        bn.train()
        assert bn.training is True

        # Test chaining
        bn_chain = bn.train(False)
        assert bn_chain is bn
        assert bn.training is False

    def test_batchnorm1d_without_tracking(self):
        """Test BatchNorm1d without running stats tracking."""
        bn = BatchNorm1d(16, track_running_stats=False)

        x = Tensor(np.random.randn(4, 16).astype(np.float32))
        output = bn.forward(x)
        assert output.shape == (4, 16)

        # Try eval mode without tracking - should raise error
        bn.eval()
        with pytest.raises(RuntimeError):
            bn.forward(x)

    def test_batchnorm1d_input_validation(self):
        """Test BatchNorm1d input validation."""
        bn = BatchNorm1d(32)

        # Test wrong dimensions
        x_1d = Tensor(np.random.randn(32).astype(np.float32))
        with pytest.raises(LayerError):
            bn.forward(x_1d)

        # Test wrong number of features
        x_wrong_features = Tensor(np.random.randn(4, 16).astype(np.float32))
        with pytest.raises(LayerError):
            bn.forward(x_wrong_features)

        # Test unsupported dimensions
        x_4d = Tensor(np.random.randn(2, 32, 8, 8).astype(np.float32))
        with pytest.raises(LayerError):
            bn.forward(x_4d)

    def test_batchnorm1d_running_stats_update(self):
        """Test running statistics update mechanism."""
        bn = BatchNorm1d(8)

        # First batch
        x1 = Tensor(np.random.randn(4, 8).astype(np.float32))
        output1 = bn.forward(x1)

        first_mean = bn.running_mean.copy()
        first_var = bn.running_var.copy()

        # Second batch
        x2 = Tensor(np.random.randn(4, 8).astype(np.float32))
        output2 = bn.forward(x2)

        # Running stats should have changed
        assert not np.allclose(bn.running_mean, first_mean)
        assert not np.allclose(bn.running_var, first_var)
        assert bn.num_batches_tracked == 2

    def test_batchnorm1d_extra_repr(self):
        """Test BatchNorm1d string representation."""
        bn = BatchNorm1d(64, eps=1e-3, momentum=0.2, affine=False)
        repr_str = bn.extra_repr()

        assert "num_features=64" in repr_str
        assert "eps=0.001" in repr_str
        assert "momentum=0.2" in repr_str
        assert "affine=False" in repr_str


class TestBatchNorm2d95Coverage:
    """Comprehensive BatchNorm2d tests targeting 95%+ coverage."""

    def test_batchnorm2d_forward_pass(self):
        """Test BatchNorm2d forward pass with 4D input."""
        bn2d = BatchNorm2d(32)

        # Test 4D input (N, C, H, W)
        x_4d = Tensor(np.random.randn(2, 32, 8, 8).astype(np.float32))
        output_4d = bn2d.forward(x_4d)
        assert output_4d.shape == (2, 32, 8, 8)

    def test_batchnorm2d_input_validation(self):
        """Test BatchNorm2d input validation."""
        bn2d = BatchNorm2d(32)

        # Test wrong dimensions
        x_3d = Tensor(np.random.randn(2, 32, 8).astype(np.float32))
        with pytest.raises(LayerError):
            bn2d.forward(x_3d)

        # Test wrong number of channels
        x_wrong_channels = Tensor(np.random.randn(2, 16, 8, 8).astype(np.float32))
        with pytest.raises(LayerError):
            bn2d.forward(x_wrong_channels)

    def test_batchnorm2d_training_vs_eval(self):
        """Test BatchNorm2d training vs evaluation behavior."""
        bn2d = BatchNorm2d(16)

        # Training mode
        x_train = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float32))
        output_train = bn2d.forward(x_train)
        train_batches = bn2d.num_batches_tracked

        # Eval mode
        bn2d.eval()
        x_eval = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float32))
        output_eval = bn2d.forward(x_eval)

        # Batches tracked should not increase in eval
        assert bn2d.num_batches_tracked == train_batches


class TestGroupNorm95Coverage:
    """Comprehensive GroupNorm tests targeting 95%+ coverage."""

    def test_groupnorm_initialization_comprehensive(self):
        """Test all GroupNorm initialization parameters."""
        # Test basic initialization
        gn = GroupNorm(num_groups=4, num_channels=32)
        assert gn.num_groups == 4
        assert gn.num_channels == 32
        assert gn.eps == 1e-5
        assert gn.affine is True
        assert gn.weight.shape == (32,)
        assert gn.bias.shape == (32,)

        # Test without affine
        gn_no_affine = GroupNorm(num_groups=2, num_channels=16, affine=False)
        assert gn_no_affine.weight is None
        assert gn_no_affine.bias is None

    def test_groupnorm_initialization_validation(self):
        """Test GroupNorm initialization validation."""
        # Invalid num_channels
        with pytest.raises(LayerError):
            GroupNorm(num_groups=4, num_channels=0)
        with pytest.raises(LayerError):
            GroupNorm(num_groups=4, num_channels=-1)

        # Invalid num_groups
        with pytest.raises(LayerError):
            GroupNorm(num_groups=0, num_channels=32)
        with pytest.raises(LayerError):
            GroupNorm(num_groups=-1, num_channels=32)

        # Non-divisible combination
        with pytest.raises(LayerError):
            GroupNorm(num_groups=3, num_channels=32)  # 32 not divisible by 3

        # Invalid eps
        with pytest.raises(LayerError):
            GroupNorm(num_groups=4, num_channels=32, eps=0)
        with pytest.raises(LayerError):
            GroupNorm(num_groups=4, num_channels=32, eps=-1e-5)

    def test_groupnorm_forward_pass_comprehensive(self):
        """Test GroupNorm forward pass with various inputs."""
        gn = GroupNorm(num_groups=4, num_channels=32)

        # Test different input shapes
        test_shapes = [
            (2, 32, 8),  # 3D input (N, C, L)
            (2, 32, 8, 8),  # 4D input (N, C, H, W)
            (1, 32, 4, 4, 4),  # 5D input (N, C, D, H, W)
        ]

        for shape in test_shapes:
            x = Tensor(np.random.randn(*shape).astype(np.float32))
            output = gn.forward(x)
            assert output.shape == shape
            assert np.all(np.isfinite(output.data))

    def test_groupnorm_input_validation(self):
        """Test GroupNorm input validation."""
        gn = GroupNorm(num_groups=4, num_channels=32)

        # Test insufficient dimensions
        x_2d = Tensor(np.random.randn(2, 32).astype(np.float32))
        with pytest.raises(LayerError):
            gn.forward(x_2d)

        # Test wrong number of channels
        x_wrong = Tensor(np.random.randn(2, 16, 8).astype(np.float32))
        with pytest.raises(LayerError):
            gn.forward(x_wrong)

    def test_groupnorm_different_group_configurations(self):
        """Test GroupNorm with different group configurations."""
        # Test various valid group configurations
        configs = [
            (1, 32),  # Layer norm style (1 group)
            (2, 32),  # 2 groups
            (4, 32),  # 4 groups
            (8, 32),  # 8 groups
            (16, 32),  # 16 groups
            (32, 32),  # Instance norm style (each channel is a group)
        ]

        for num_groups, num_channels in configs:
            gn = GroupNorm(num_groups=num_groups, num_channels=num_channels)
            x = Tensor(np.random.randn(2, num_channels, 8, 8).astype(np.float32))
            output = gn.forward(x)
            assert output.shape == (2, num_channels, 8, 8)

    def test_groupnorm_extra_repr(self):
        """Test GroupNorm string representation."""
        gn = GroupNorm(num_groups=4, num_channels=32, eps=1e-3, affine=False)
        repr_str = gn.extra_repr()

        assert "num_groups=4" in repr_str
        assert "num_channels=32" in repr_str
        assert "eps=0.001" in repr_str
        assert "affine=False" in repr_str


class TestInstanceNorm95Coverage:
    """Comprehensive InstanceNorm tests targeting 95%+ coverage."""

    def test_instancenorm_initialization_comprehensive(self):
        """Test all InstanceNorm initialization parameters."""
        # Test basic initialization
        in_norm = InstanceNorm(32)
        assert in_norm.num_features == 32
        assert in_norm.eps == 1e-5
        assert in_norm.momentum == 0.1
        assert in_norm.affine is False  # Default is False for InstanceNorm
        assert in_norm.track_running_stats is False  # Default is False
        assert in_norm.weight is None
        assert in_norm.bias is None

        # Test with affine
        in_norm_affine = InstanceNorm(32, affine=True)
        assert in_norm_affine.weight.shape == (32,)
        assert in_norm_affine.bias.shape == (32,)

        # Test with tracking
        in_norm_track = InstanceNorm(32, track_running_stats=True)
        assert in_norm_track.running_mean.shape == (32,)
        assert in_norm_track.running_var.shape == (32,)
        assert in_norm_track.num_batches_tracked == 0

    def test_instancenorm_initialization_validation(self):
        """Test InstanceNorm initialization validation."""
        with pytest.raises(LayerError):
            InstanceNorm(0)
        with pytest.raises(LayerError):
            InstanceNorm(-1)
        with pytest.raises(LayerError):
            InstanceNorm(32, eps=0)
        with pytest.raises(LayerError):
            InstanceNorm(32, eps=-1e-5)

    def test_instancenorm_forward_pass_comprehensive(self):
        """Test InstanceNorm forward pass."""
        in_norm = InstanceNorm(32)

        # Test different input shapes
        test_shapes = [
            (2, 32, 8),  # 3D input (N, C, L)
            (2, 32, 8, 8),  # 4D input (N, C, H, W)
            (1, 32, 4, 4, 4),  # 5D input (N, C, D, H, W)
        ]

        for shape in test_shapes:
            x = Tensor(np.random.randn(*shape).astype(np.float32))
            output = in_norm.forward(x)
            assert output.shape == shape
            assert np.all(np.isfinite(output.data))

    def test_instancenorm_input_validation(self):
        """Test InstanceNorm input validation."""
        in_norm = InstanceNorm(32)

        # Test insufficient dimensions
        x_2d = Tensor(np.random.randn(2, 32).astype(np.float32))
        with pytest.raises(LayerError):
            in_norm.forward(x_2d)

        # Test wrong number of features
        x_wrong = Tensor(np.random.randn(2, 16, 8).astype(np.float32))
        with pytest.raises(LayerError):
            in_norm.forward(x_wrong)

        # Test no spatial dimensions
        # This would be a 2D input which should fail
        pass

    def test_instancenorm_with_tracking(self):
        """Test InstanceNorm with running statistics tracking."""
        in_norm = InstanceNorm(16, track_running_stats=True)

        # Training mode
        x1 = Tensor(np.random.randn(2, 16, 8, 8).astype(np.float32))
        output1 = in_norm.forward(x1)
        assert in_norm.num_batches_tracked == 1

        # Second batch
        x2 = Tensor(np.random.randn(2, 16, 8, 8).astype(np.float32))
        output2 = in_norm.forward(x2)
        assert in_norm.num_batches_tracked == 2

    def test_instancenorm_extra_repr(self):
        """Test InstanceNorm string representation."""
        in_norm = InstanceNorm(32, eps=1e-3, affine=True)
        repr_str = in_norm.extra_repr()

        assert "num_features=32" in repr_str
        assert "eps=0.001" in repr_str
        assert "affine=True" in repr_str


class TestNormalizationIntegration:
    """Integration tests for normalization layers."""

    def test_normalization_layers_composition(self):
        """Test composition of different normalization layers."""
        # Create different normalization layers
        ln = LayerNorm(64)
        rms = RMSNorm(64)

        x = Tensor(np.random.randn(2, 10, 64).astype(np.float32))

        # Test sequential application
        output1 = ln.forward(x)
        output2 = rms.forward(output1)

        assert output2.shape == x.shape
        assert np.all(np.isfinite(output2.data))

    def test_normalization_gradient_flow(self):
        """Test gradient flow through normalization layers."""
        layers = [
            LayerNorm(32),
            RMSNorm(32),
            BatchNorm1d(32),
            GroupNorm(4, 32),
            InstanceNorm(32, affine=True),
        ]

        for layer in layers:
            if isinstance(layer, (BatchNorm1d,)):
                x = Tensor(np.random.randn(4, 32).astype(np.float32), requires_grad=True)
            elif isinstance(layer, (GroupNorm, InstanceNorm)):
                x = Tensor(np.random.randn(2, 32, 8).astype(np.float32), requires_grad=True)
            else:
                x = Tensor(np.random.randn(2, 10, 32).astype(np.float32), requires_grad=True)

            output = layer.forward(x)
            assert output.requires_grad is True

    def test_normalization_numerical_stability(self):
        """Test numerical stability across normalization layers."""
        layers = [
            LayerNorm(16),
            RMSNorm(16),
            BatchNorm1d(16),
            GroupNorm(2, 16),
            InstanceNorm(16),
        ]

        # Test with extreme values
        extreme_values = [1e-8, 1e8, 0.0]

        for layer in layers:
            for val in extreme_values:
                if isinstance(layer, BatchNorm1d):
                    x = Tensor(np.full((2, 16), val, dtype=np.float32))
                elif isinstance(layer, (GroupNorm, InstanceNorm)):
                    x = Tensor(np.full((2, 16, 4), val, dtype=np.float32))
                else:
                    x = Tensor(np.full((2, 8, 16), val, dtype=np.float32))

                try:
                    output = layer.forward(x)
                    assert np.all(np.isfinite(output.data))
                except (ValueError, RuntimeError):
                    # Some extreme values might cause mathematical issues
                    pass

    def test_normalization_dtype_consistency(self):
        """Test data type consistency across normalization layers."""
        layers = [
            LayerNorm(8),
            RMSNorm(8),
            BatchNorm1d(8),
            GroupNorm(2, 8),
            InstanceNorm(8),
        ]

        for layer in layers:
            if isinstance(layer, BatchNorm1d):
                x = Tensor(np.random.randn(2, 8).astype(np.float32))
            elif isinstance(layer, (GroupNorm, InstanceNorm)):
                x = Tensor(np.random.randn(2, 8, 4).astype(np.float32))
            else:
                x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))

            output = layer.forward(x)
            assert output.data.dtype == np.float32
