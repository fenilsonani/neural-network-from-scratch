"""Comprehensive tests for Linear layer targeting missing coverage areas.

This file specifically targets the uncovered lines to improve Linear layer coverage
from 52.14% to 85%+. Focus on initialization schemes, error handling, and advanced features.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core.base import Parameter
from neural_arch.core.tensor import Tensor
from neural_arch.exceptions import LayerError
from neural_arch.nn.linear import Linear


class TestLinearMissingCoverage:
    """Tests targeting specific missing coverage areas in Linear layer."""

    def test_all_weight_initialization_schemes(self):
        """Test all weight initialization schemes comprehensively."""
        in_features, out_features = 10, 5

        # Test xavier_uniform (default)
        layer_xavier_uniform = Linear(in_features, out_features, weight_init="xavier_uniform")
        assert layer_xavier_uniform.weight.shape == (in_features, out_features)

        # Test xavier_normal
        layer_xavier_normal = Linear(in_features, out_features, weight_init="xavier_normal")
        assert layer_xavier_normal.weight.shape == (in_features, out_features)

        # Test he_uniform
        layer_he_uniform = Linear(in_features, out_features, weight_init="he_uniform")
        assert layer_he_uniform.weight.shape == (in_features, out_features)

        # Test he_normal
        layer_he_normal = Linear(in_features, out_features, weight_init="he_normal")
        assert layer_he_normal.weight.shape == (in_features, out_features)

        # Test lecun_uniform
        layer_lecun_uniform = Linear(in_features, out_features, weight_init="lecun_uniform")
        assert layer_lecun_uniform.weight.shape == (in_features, out_features)

        # Test lecun_normal
        layer_lecun_normal = Linear(in_features, out_features, weight_init="lecun_normal")
        assert layer_lecun_normal.weight.shape == (in_features, out_features)

        # Test uniform
        layer_uniform = Linear(in_features, out_features, weight_init="uniform")
        assert layer_uniform.weight.shape == (in_features, out_features)

        # Test normal
        layer_normal = Linear(in_features, out_features, weight_init="normal")
        assert layer_normal.weight.shape == (in_features, out_features)

        # Test zeros
        layer_zeros = Linear(in_features, out_features, weight_init="zeros")
        assert layer_zeros.weight.shape == (in_features, out_features)
        np.testing.assert_array_equal(
            layer_zeros.weight.data, np.zeros((in_features, out_features))
        )

        # Test ones
        layer_ones = Linear(in_features, out_features, weight_init="ones")
        assert layer_ones.weight.shape == (in_features, out_features)
        np.testing.assert_array_equal(layer_ones.weight.data, np.ones((in_features, out_features)))

    def test_all_bias_initialization_schemes(self):
        """Test all bias initialization schemes comprehensively."""
        in_features, out_features = 5, 3

        # Test zeros (default)
        layer_zeros = Linear(in_features, out_features, bias_init="zeros")
        assert layer_zeros.bias.shape == (out_features,)
        np.testing.assert_array_equal(layer_zeros.bias.data, np.zeros(out_features))

        # Test ones
        layer_ones = Linear(in_features, out_features, bias_init="ones")
        assert layer_ones.bias.shape == (out_features,)
        np.testing.assert_array_equal(layer_ones.bias.data, np.ones(out_features))

        # Test uniform
        layer_uniform = Linear(in_features, out_features, bias_init="uniform")
        assert layer_uniform.bias.shape == (out_features,)
        assert np.all(layer_uniform.bias.data >= -0.1)
        assert np.all(layer_uniform.bias.data <= 0.1)

        # Test normal
        layer_normal = Linear(in_features, out_features, bias_init="normal")
        assert layer_normal.bias.shape == (out_features,)
        # Normal initialization should have reasonable values
        assert np.std(layer_normal.bias.data) < 1.0

    def test_invalid_weight_initialization(self):
        """Test error handling for invalid weight initialization."""
        with pytest.raises(LayerError) as exc_info:
            Linear(5, 3, weight_init="invalid_scheme")

        assert "Unknown weight initialization scheme" in str(exc_info.value)
        assert "invalid_scheme" in str(exc_info.value)

    def test_invalid_bias_initialization(self):
        """Test error handling for invalid bias initialization."""
        with pytest.raises(LayerError) as exc_info:
            Linear(5, 3, bias_init="invalid_scheme")

        assert "Unknown bias initialization scheme" in str(exc_info.value)
        assert "invalid_scheme" in str(exc_info.value)

    def test_invalid_input_features(self):
        """Test error handling for invalid input features."""
        # Test zero input features
        with pytest.raises(LayerError) as exc_info:
            Linear(0, 3)

        assert "in_features must be positive" in str(exc_info.value)
        assert "got 0" in str(exc_info.value)

        # Test negative input features
        with pytest.raises(LayerError) as exc_info:
            Linear(-5, 3)

        assert "in_features must be positive" in str(exc_info.value)
        assert "got -5" in str(exc_info.value)

    def test_invalid_output_features(self):
        """Test error handling for invalid output features."""
        # Test zero output features
        with pytest.raises(LayerError) as exc_info:
            Linear(5, 0)

        assert "out_features must be positive" in str(exc_info.value)
        assert "got 0" in str(exc_info.value)

        # Test negative output features
        with pytest.raises(LayerError) as exc_info:
            Linear(5, -3)

        assert "out_features must be positive" in str(exc_info.value)
        assert "got -3" in str(exc_info.value)

    def test_invalid_input_shape_forward(self):
        """Test error handling for invalid input shape in forward pass."""
        layer = Linear(4, 3)

        # Test input with wrong feature dimension
        x_wrong = Tensor([[1, 2, 3]], requires_grad=True)  # Only 3 features, expected 4

        # The handle_exception decorator wraps LayerError in NeuralArchError
        from neural_arch.exceptions import NeuralArchError

        with pytest.raises(NeuralArchError) as exc_info:
            layer.forward(x_wrong)

        assert "Input feature dimension mismatch" in str(exc_info.value)
        assert "expected 4" in str(exc_info.value)
        assert "got 3" in str(exc_info.value)

    def test_layer_name_functionality(self):
        """Test layer name functionality and representation."""
        # Test with custom name
        custom_name = "my_linear_layer"
        layer_named = Linear(5, 3, name=custom_name)
        assert layer_named.name == custom_name

        # Test default name generation
        layer_default = Linear(10, 7)
        assert "Linear(10, 7)" in layer_default.name

        # Note: Parameter names are currently using class name, not instance name
        # This is the actual behavior we're testing
        assert "Linear" in layer_named.weight.name
        assert "Linear" in layer_named.bias.name

    def test_reset_parameters_functionality(self):
        """Test parameter reset functionality comprehensively."""
        layer = Linear(4, 3, weight_init="zeros", bias_init="zeros")

        # Initially all parameters should be zero
        np.testing.assert_array_equal(layer.weight.data, np.zeros((4, 3)))
        np.testing.assert_array_equal(layer.bias.data, np.zeros(3))

        # Reset with different initialization
        layer.reset_parameters(weight_init="ones", bias_init="ones")

        # Now parameters should be ones
        np.testing.assert_array_equal(layer.weight.data, np.ones((4, 3)))
        np.testing.assert_array_equal(layer.bias.data, np.ones(3))

        # Reset only weights
        layer.reset_parameters(weight_init="zeros")
        np.testing.assert_array_equal(layer.weight.data, np.zeros((4, 3)))
        np.testing.assert_array_equal(layer.bias.data, np.ones(3))  # Bias unchanged

        # Reset only bias
        layer.reset_parameters(bias_init="zeros")
        np.testing.assert_array_equal(layer.weight.data, np.zeros((4, 3)))  # Weight unchanged
        np.testing.assert_array_equal(layer.bias.data, np.zeros(3))

        # Reset with None (should keep current values)
        original_weight = layer.weight.data.copy()
        original_bias = layer.bias.data.copy()
        layer.reset_parameters(weight_init=None, bias_init=None)
        np.testing.assert_array_equal(layer.weight.data, original_weight)
        np.testing.assert_array_equal(layer.bias.data, original_bias)

    def test_reset_parameters_no_bias(self):
        """Test parameter reset on layer without bias."""
        layer = Linear(3, 2, bias=False, weight_init="zeros")

        # Initially weight should be zero
        np.testing.assert_array_equal(layer.weight.data, np.zeros((3, 2)))
        assert layer.bias is None

        # Reset weight
        layer.reset_parameters(weight_init="ones")
        np.testing.assert_array_equal(layer.weight.data, np.ones((3, 2)))

        # Try to reset bias (should not cause error, just be ignored)
        layer.reset_parameters(bias_init="ones")
        assert layer.bias is None  # Should still be None

    def test_extra_repr_functionality(self):
        """Test extra_repr method for debugging."""
        # Test with bias
        layer_with_bias = Linear(5, 3, bias=True)
        extra_repr = layer_with_bias.extra_repr()

        assert "in_features=5" in extra_repr
        assert "out_features=3" in extra_repr
        assert "bias=True" in extra_repr

        # Test without bias
        layer_no_bias = Linear(5, 3, bias=False)
        extra_repr_no_bias = layer_no_bias.extra_repr()

        assert "in_features=5" in extra_repr_no_bias
        assert "out_features=3" in extra_repr_no_bias
        assert "bias=False" in extra_repr_no_bias

    def test_repr_functionality(self):
        """Test __repr__ method."""
        layer = Linear(8, 4, bias=True)
        repr_str = repr(layer)

        assert "Linear" in repr_str
        assert "in_features=8" in repr_str
        assert "out_features=4" in repr_str
        assert "bias=True" in repr_str

    def test_weight_norm_property(self):
        """Test weight_norm property."""
        layer = Linear(3, 2, weight_init="ones")

        # For ones initialization, Frobenius norm should be sqrt(total_elements)
        expected_norm = np.sqrt(3 * 2)  # sqrt(6)
        actual_norm = layer.weight_norm

        assert isinstance(actual_norm, float)
        np.testing.assert_almost_equal(actual_norm, expected_norm, decimal=6)

        # Test with zeros initialization
        layer_zeros = Linear(3, 2, weight_init="zeros")
        assert layer_zeros.weight_norm == 0.0

    def test_bias_norm_property(self):
        """Test bias_norm property."""
        # Test with bias
        layer_with_bias = Linear(3, 2, bias_init="ones")

        # For ones initialization, L2 norm should be sqrt(num_elements)
        expected_norm = np.sqrt(2)  # sqrt(2)
        actual_norm = layer_with_bias.bias_norm

        assert isinstance(actual_norm, float)
        np.testing.assert_almost_equal(actual_norm, expected_norm, decimal=6)

        # Test with zeros initialization
        layer_zeros = Linear(3, 2, bias_init="zeros")
        assert layer_zeros.bias_norm == 0.0

        # Test without bias
        layer_no_bias = Linear(3, 2, bias=False)
        assert layer_no_bias.bias_norm == 0.0

    def test_get_weight_stats_functionality(self):
        """Test get_weight_stats method comprehensively."""
        # Test with known initialization
        layer = Linear(4, 3, weight_init="ones", bias_init="zeros")

        stats = layer.get_weight_stats()

        # Check that all expected keys are present
        expected_keys = {
            "weight_mean",
            "weight_std",
            "weight_min",
            "weight_max",
            "weight_norm",
            "bias_mean",
            "bias_std",
            "bias_min",
            "bias_max",
            "bias_norm",
        }
        assert all(key in stats for key in expected_keys)

        # Check weight statistics for ones initialization
        assert stats["weight_mean"] == 1.0
        assert stats["weight_std"] == 0.0
        assert stats["weight_min"] == 1.0
        assert stats["weight_max"] == 1.0
        np.testing.assert_almost_equal(stats["weight_norm"], np.sqrt(12), decimal=6)  # sqrt(4*3)

        # Check bias statistics for zeros initialization
        assert stats["bias_mean"] == 0.0
        assert stats["bias_std"] == 0.0
        assert stats["bias_min"] == 0.0
        assert stats["bias_max"] == 0.0
        assert stats["bias_norm"] == 0.0

        # Verify all values are Python floats
        for key, value in stats.items():
            assert isinstance(value, float)

    def test_get_weight_stats_no_bias(self):
        """Test get_weight_stats method for layer without bias."""
        layer = Linear(3, 2, bias=False, weight_init="zeros")

        stats = layer.get_weight_stats()

        # Should only have weight statistics
        weight_keys = {"weight_mean", "weight_std", "weight_min", "weight_max", "weight_norm"}
        bias_keys = {"bias_mean", "bias_std", "bias_min", "bias_max", "bias_norm"}

        assert all(key in stats for key in weight_keys)
        assert all(key not in stats for key in bias_keys)

        # Check weight statistics for zeros initialization
        assert stats["weight_mean"] == 0.0
        assert stats["weight_std"] == 0.0
        assert stats["weight_min"] == 0.0
        assert stats["weight_max"] == 0.0
        assert stats["weight_norm"] == 0.0

    def test_forward_without_bias(self):
        """Test forward pass without bias thoroughly."""
        layer = Linear(3, 2, bias=False, weight_init="ones")

        x = Tensor([[1, 2, 3]], requires_grad=True)
        output = layer.forward(x)

        # Expected output: [1, 2, 3] @ [[1, 1], [1, 1], [1, 1]] = [6, 6]
        expected_output = np.array([[6, 6]], dtype=np.float32)
        np.testing.assert_array_almost_equal(output.data, expected_output)

        assert output.requires_grad is True
        assert output.shape == (1, 2)

    def test_forward_with_bias(self):
        """Test forward pass with bias thoroughly."""
        layer = Linear(3, 2, weight_init="ones", bias_init="ones")

        x = Tensor([[1, 2, 3]], requires_grad=True)
        output = layer.forward(x)

        # Expected output: [1, 2, 3] @ [[1, 1], [1, 1], [1, 1]] + [1, 1] = [7, 7]
        expected_output = np.array([[7, 7]], dtype=np.float32)
        np.testing.assert_array_almost_equal(output.data, expected_output)

        assert output.requires_grad is True
        assert output.shape == (1, 2)

    def test_initialization_statistical_properties(self):
        """Test statistical properties of different initialization schemes."""
        in_features, out_features = 100, 50  # Large enough for statistics

        # Test xavier_uniform bounds
        layer_xavier = Linear(in_features, out_features, weight_init="xavier_uniform")
        limit = np.sqrt(6.0 / (in_features + out_features))
        assert np.all(layer_xavier.weight.data >= -limit)
        assert np.all(layer_xavier.weight.data <= limit)

        # Test he_uniform bounds
        layer_he = Linear(in_features, out_features, weight_init="he_uniform")
        limit_he = np.sqrt(6.0 / in_features)
        assert np.all(layer_he.weight.data >= -limit_he)
        assert np.all(layer_he.weight.data <= limit_he)

        # Test lecun_uniform bounds
        layer_lecun = Linear(in_features, out_features, weight_init="lecun_uniform")
        limit_lecun = np.sqrt(3.0 / in_features)
        assert np.all(layer_lecun.weight.data >= -limit_lecun)
        assert np.all(layer_lecun.weight.data <= limit_lecun)

        # Test uniform bounds
        layer_uniform = Linear(in_features, out_features, weight_init="uniform")
        assert np.all(layer_uniform.weight.data >= -0.1)
        assert np.all(layer_uniform.weight.data <= 0.1)

        # Test normal initialization statistics
        layer_normal = Linear(in_features, out_features, weight_init="normal")
        weight_mean = np.mean(layer_normal.weight.data)
        weight_std = np.std(layer_normal.weight.data)
        assert abs(weight_mean) < 0.1  # Should be close to 0
        assert 0.05 < weight_std < 0.15  # Should be close to 0.1

    def test_layer_error_inheritance(self):
        """Test that LayerError is properly raised and contains correct information."""
        # Test invalid in_features with layer name
        layer_name = "test_layer"

        with pytest.raises(LayerError) as exc_info:
            Linear(-1, 3, name=layer_name)

        error = exc_info.value
        assert "in_features must be positive" in str(error)
        assert "got -1" in str(error)

        # Test the exception has proper attributes if they exist
        if hasattr(error, "layer_name"):
            assert error.layer_name == layer_name
        if hasattr(error, "layer_type"):
            assert error.layer_type == "Linear"

    def test_mathematical_correctness(self):
        """Test mathematical correctness of linear transformation."""
        # Use known values for precise testing
        layer = Linear(2, 3, weight_init="zeros", bias_init="zeros")

        # Set specific weight and bias values
        layer.weight.data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        layer.bias.data = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Test input
        x = Tensor([[2, 3]], requires_grad=True)
        output = layer(x)

        # Manual calculation: [2, 3] @ [[1, 2, 3], [4, 5, 6]] + [0.1, 0.2, 0.3]
        # = [2*1 + 3*4, 2*2 + 3*5, 2*3 + 3*6] + [0.1, 0.2, 0.3]
        # = [14, 19, 24] + [0.1, 0.2, 0.3] = [14.1, 19.2, 24.3]
        expected = np.array([[14.1, 19.2, 24.3]], dtype=np.float32)

        np.testing.assert_array_almost_equal(output.data, expected, decimal=6)

    def test_multiple_initialization_calls(self):
        """Test that multiple initialization calls work correctly."""
        layer = Linear(3, 2, weight_init="zeros", bias_init="zeros")

        # Initial state
        np.testing.assert_array_equal(layer.weight.data, np.zeros((3, 2)))
        np.testing.assert_array_equal(layer.bias.data, np.zeros(2))

        # First reset
        layer.reset_parameters(weight_init="ones", bias_init="ones")
        np.testing.assert_array_equal(layer.weight.data, np.ones((3, 2)))
        np.testing.assert_array_equal(layer.bias.data, np.ones(2))

        # Second reset
        layer.reset_parameters(weight_init="uniform", bias_init="uniform")
        assert layer.weight.shape == (3, 2)
        assert layer.bias.shape == (2,)
        # Values should be different from ones
        assert not np.array_equal(layer.weight.data, np.ones((3, 2)))
        assert not np.array_equal(layer.bias.data, np.ones(2))


class TestLinearAdvancedFeatures:
    """Test advanced Linear layer features and edge cases."""

    def test_parameter_dtypes(self):
        """Test that parameters have correct dtypes."""
        layer = Linear(3, 2)

        # Weights and biases should be float32
        assert layer.weight.data.dtype == np.float32
        assert layer.bias.data.dtype == np.float32

    def test_parameter_requires_grad(self):
        """Test that parameters require gradients by default."""
        layer = Linear(4, 3)

        assert layer.weight.requires_grad is True
        assert layer.bias.requires_grad is True

    def test_handle_exception_decorator(self):
        """Test that the handle_exception decorator works correctly."""
        layer = Linear(3, 2)

        # Test that invalid input raises NeuralArchError (which wraps LayerError)
        x_invalid = Tensor([[1, 2]], requires_grad=True)  # Wrong shape (2 features instead of 3)

        from neural_arch.exceptions import NeuralArchError

        with pytest.raises(NeuralArchError):
            layer.forward(x_invalid)

    def test_layer_with_different_names(self):
        """Test layers with different custom names."""
        layer1 = Linear(3, 2, name="layer1")
        layer2 = Linear(3, 2, name="layer2")

        assert layer1.name == "layer1"
        assert layer2.name == "layer2"

        # Parameter names currently use class name, not instance name
        # This is the actual behavior we're testing
        assert "Linear" in layer1.weight.name
        assert "Linear" in layer2.weight.name
        assert "Linear" in layer1.bias.name
        assert "Linear" in layer2.bias.name

        # But the layers themselves have different names
        assert layer1.name != layer2.name

    def test_statistics_with_random_data(self):
        """Test statistics methods with random initialized data."""
        # Use xavier_normal for predictable statistics
        layer = Linear(100, 50, weight_init="xavier_normal", bias_init="normal")

        stats = layer.get_weight_stats()

        # Xavier normal should have reasonable statistics
        assert -0.5 < stats["weight_mean"] < 0.5  # Should be close to 0
        assert 0.05 < stats["weight_std"] < 0.3  # Should have reasonable std
        assert stats["weight_norm"] > 0  # Should be non-zero

        # Bias should also have reasonable statistics
        assert -0.5 < stats["bias_mean"] < 0.5
        assert stats["bias_norm"] >= 0

    def test_edge_case_dimensions(self):
        """Test edge cases with unusual dimensions."""
        # Very small layer
        layer_tiny = Linear(1, 1)
        x_tiny = Tensor([[5]], requires_grad=True)
        output_tiny = layer_tiny(x_tiny)
        assert output_tiny.shape == (1, 1)

        # Asymmetric layer (many inputs, few outputs)
        layer_wide = Linear(100, 1)
        x_wide = Tensor(np.random.randn(1, 100), requires_grad=True)
        output_wide = layer_wide(x_wide)
        assert output_wide.shape == (1, 1)

        # Asymmetric layer (few inputs, many outputs)
        layer_tall = Linear(1, 100)
        x_tall = Tensor([[1]], requires_grad=True)
        output_tall = layer_tall(x_tall)
        assert output_tall.shape == (1, 100)
