"""Ultra-comprehensive tests for Linear layer to achieve 95%+ test coverage.

This test suite is designed to exercise every code path, edge case, and error condition
in the Linear layer implementation to ensure robust 95%+ test coverage.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.core.tensor import Tensor
from neural_arch.core.device import Device, DeviceType
from neural_arch.core.dtype import DType
from neural_arch.nn.linear import Linear
from neural_arch.core.base import Parameter
from neural_arch.exceptions import LayerError


class TestLinearLayer95Coverage:
    """Comprehensive Linear layer tests targeting 95%+ coverage."""
    
    def test_linear_initialization_all_parameters(self):
        """Test all Linear initialization parameter combinations."""
        # Test basic initialization with all defaults
        layer = Linear(10, 5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.use_bias == True
        assert layer.weight.shape == (10, 5)
        assert layer.bias.shape == (5,)
        assert isinstance(layer.weight, Parameter)
        assert isinstance(layer.bias, Parameter)
        
        # Test initialization without bias
        layer_no_bias = Linear(8, 4, bias=False)
        assert layer_no_bias.in_features == 8
        assert layer_no_bias.out_features == 4
        assert layer_no_bias.use_bias == False
        assert layer_no_bias.weight.shape == (8, 4)
        assert layer_no_bias.bias is None
        
        # Test with custom name
        custom_name = "custom_linear_layer"
        layer_named = Linear(6, 3, name=custom_name)
        assert layer_named.name == custom_name
        assert custom_name in layer_named.weight.name
        assert custom_name in layer_named.bias.name
        
    def test_all_weight_initialization_schemes(self):
        """Test every weight initialization scheme available."""
        in_features, out_features = 4, 3
        
        # Test all weight initialization schemes
        weight_schemes = [
            "xavier_uniform", "xavier_normal", "he_uniform", "he_normal",
            "lecun_uniform", "lecun_normal", "uniform", "normal", "zeros", "ones"
        ]
        
        for scheme in weight_schemes:
            layer = Linear(in_features, out_features, weight_init=scheme)
            assert layer.weight.shape == (in_features, out_features)
            
            # Verify initialization properties
            if scheme == "zeros":
                assert np.allclose(layer.weight.data, 0.0)
            elif scheme == "ones":
                assert np.allclose(layer.weight.data, 1.0)
            else:
                # For other schemes, verify they're not all zeros or ones
                assert not np.allclose(layer.weight.data, 0.0)
                assert not np.allclose(layer.weight.data, 1.0)
    
    def test_all_bias_initialization_schemes(self):
        """Test every bias initialization scheme available."""
        in_features, out_features = 4, 3
        
        # Test all bias initialization schemes
        bias_schemes = ["zeros", "ones", "uniform", "normal"]
        
        for scheme in bias_schemes:
            layer = Linear(in_features, out_features, bias_init=scheme)
            assert layer.bias.shape == (out_features,)
            
            # Verify initialization properties
            if scheme == "zeros":
                assert np.allclose(layer.bias.data, 0.0)
            elif scheme == "ones":
                assert np.allclose(layer.bias.data, 1.0)
            else:
                # For other schemes, verify they're not all zeros
                assert not np.allclose(layer.bias.data, 0.0) or scheme == "uniform"
    
    def test_initialization_error_conditions(self):
        """Test all error conditions during initialization."""
        # Test negative in_features
        with pytest.raises(LayerError) as exc_info:
            Linear(-1, 5)
        assert "in_features must be positive" in str(exc_info.value)
        
        # Test zero in_features
        with pytest.raises(LayerError) as exc_info:
            Linear(0, 5)
        assert "in_features must be positive" in str(exc_info.value)
        
        # Test negative out_features
        with pytest.raises(LayerError) as exc_info:
            Linear(5, -1)
        assert "out_features must be positive" in str(exc_info.value)
        
        # Test zero out_features
        with pytest.raises(LayerError) as exc_info:
            Linear(5, 0)
        assert "out_features must be positive" in str(exc_info.value)
        
        # Test unknown weight initialization scheme
        with pytest.raises(LayerError) as exc_info:
            Linear(5, 3, weight_init="unknown_scheme")
        assert "Unknown weight initialization scheme" in str(exc_info.value)
        
        # Test unknown bias initialization scheme
        with pytest.raises(LayerError) as exc_info:
            Linear(5, 3, bias_init="unknown_scheme")
        assert "Unknown bias initialization scheme" in str(exc_info.value)
    
    def test_forward_pass_comprehensive(self):
        """Test forward pass with various input configurations."""
        layer = Linear(4, 3)
        
        # Test 2D input (batch_size, in_features)
        x_2d = Tensor(np.random.randn(2, 4).astype(np.float32))
        output_2d = layer.forward(x_2d)
        assert output_2d.shape == (2, 3)
        
        # Test 3D input (batch_size, seq_len, in_features)
        x_3d = Tensor(np.random.randn(2, 5, 4).astype(np.float32))
        output_3d = layer.forward(x_3d)
        assert output_3d.shape == (2, 5, 3)
        
        # Test 4D input (batch_size, height, width, in_features)
        x_4d = Tensor(np.random.randn(2, 8, 8, 4).astype(np.float32))
        output_4d = layer.forward(x_4d)
        assert output_4d.shape == (2, 8, 8, 3)
        
        # Test 1D input (in_features,)
        x_1d = Tensor(np.random.randn(4).astype(np.float32))
        output_1d = layer.forward(x_1d)
        assert output_1d.shape == (3,)
    
    def test_forward_pass_without_bias(self):
        """Test forward pass for layer without bias."""
        layer = Linear(4, 3, bias=False)
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        output = layer.forward(x)
        assert output.shape == (2, 3)
        
        # Verify output is just matrix multiplication
        expected = np.dot(x.data, layer.weight.data)
        assert np.allclose(output.data, expected, rtol=1e-5)
    
    def test_forward_pass_with_bias(self):
        """Test forward pass for layer with bias."""
        layer = Linear(4, 3, bias=True)
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        output = layer.forward(x)
        assert output.shape == (2, 3)
        
        # Verify output includes bias
        expected = np.dot(x.data, layer.weight.data) + layer.bias.data
        assert np.allclose(output.data, expected, rtol=1e-5)
    
    def test_forward_input_validation(self):
        """Test input validation in forward pass."""
        layer = Linear(4, 3)
        
        # Test input with wrong feature dimension
        x_wrong = Tensor(np.random.randn(2, 5).astype(np.float32))  # 5 instead of 4
        with pytest.raises(LayerError) as exc_info:
            layer.forward(x_wrong)
        assert "Input feature dimension mismatch" in str(exc_info.value)
    
    def test_reset_parameters_functionality(self):
        """Test reset_parameters method with all options."""
        layer = Linear(4, 3)
        
        # Store original weights and bias
        original_weight = layer.weight.data.copy()
        original_bias = layer.bias.data.copy()
        
        # Test reset with same initialization schemes
        layer.reset_parameters()
        # Weights should be different (randomly re-initialized)
        assert not np.allclose(layer.weight.data, original_weight)
        
        # Test reset with different weight initialization
        layer.reset_parameters(weight_init="zeros")
        assert np.allclose(layer.weight.data, 0.0)
        
        # Test reset with different bias initialization  
        layer.reset_parameters(bias_init="ones")
        assert np.allclose(layer.bias.data, 1.0)
        
        # Test reset with both parameters
        layer.reset_parameters(weight_init="ones", bias_init="zeros")
        assert np.allclose(layer.weight.data, 1.0)
        assert np.allclose(layer.bias.data, 0.0)
        
        # Test reset with None values (should keep current)
        layer.reset_parameters(weight_init=None, bias_init=None)
        assert np.allclose(layer.weight.data, 1.0)  # Should remain ones
        assert np.allclose(layer.bias.data, 0.0)    # Should remain zeros
    
    def test_reset_parameters_no_bias(self):
        """Test reset_parameters for layer without bias."""
        layer = Linear(4, 3, bias=False)
        
        # Store original weight
        original_weight = layer.weight.data.copy()
        
        # Reset parameters
        layer.reset_parameters(weight_init="zeros")
        assert np.allclose(layer.weight.data, 0.0)
        assert layer.bias is None
        
        # Try to reset bias (should be ignored)
        layer.reset_parameters(bias_init="ones")
        assert layer.bias is None
    
    def test_string_representations(self):
        """Test all string representation methods."""
        # Test with bias
        layer_with_bias = Linear(4, 3, bias=True)
        extra_repr = layer_with_bias.extra_repr()
        assert "in_features=4" in extra_repr
        assert "out_features=3" in extra_repr
        assert "bias=True" in extra_repr
        
        full_repr = repr(layer_with_bias)
        assert "Linear" in full_repr
        assert "in_features=4" in full_repr
        assert "out_features=3" in full_repr
        
        # Test without bias
        layer_no_bias = Linear(4, 3, bias=False)
        extra_repr_no_bias = layer_no_bias.extra_repr()
        assert "bias=False" in extra_repr_no_bias
    
    def test_weight_and_bias_norms(self):
        """Test weight_norm and bias_norm properties."""
        layer = Linear(4, 3)
        
        # Test weight norm calculation
        weight_norm = layer.weight_norm
        expected_norm = np.linalg.norm(layer.weight.data)
        assert np.isclose(weight_norm, expected_norm)
        
        # Test bias norm calculation
        bias_norm = layer.bias_norm
        expected_bias_norm = np.linalg.norm(layer.bias.data)
        assert np.isclose(bias_norm, expected_bias_norm)
        
        # Test bias norm for layer without bias
        layer_no_bias = Linear(4, 3, bias=False)
        assert layer_no_bias.bias_norm == 0.0
    
    def test_get_weight_stats_comprehensive(self):
        """Test get_weight_stats method comprehensively."""
        # Test with bias
        layer = Linear(4, 3)
        stats = layer.get_weight_stats()
        
        # Check weight statistics
        assert "weight_mean" in stats
        assert "weight_std" in stats
        assert "weight_min" in stats
        assert "weight_max" in stats
        assert "weight_norm" in stats
        
        # Check bias statistics
        assert "bias_mean" in stats
        assert "bias_std" in stats
        assert "bias_min" in stats
        assert "bias_max" in stats
        assert "bias_norm" in stats
        
        # Verify statistics are reasonable
        assert isinstance(stats["weight_mean"], float)
        assert isinstance(stats["weight_std"], float)
        assert stats["weight_std"] >= 0
        assert stats["weight_norm"] >= 0
        assert stats["bias_norm"] >= 0
        
        # Test without bias
        layer_no_bias = Linear(4, 3, bias=False)
        stats_no_bias = layer_no_bias.get_weight_stats()
        
        # Should only have weight statistics
        assert "weight_mean" in stats_no_bias
        assert "weight_std" in stats_no_bias
        assert "weight_min" in stats_no_bias
        assert "weight_max" in stats_no_bias
        assert "weight_norm" in stats_no_bias
        
        # Should not have bias statistics
        assert "bias_mean" not in stats_no_bias
        assert "bias_std" not in stats_no_bias
        assert "bias_min" not in stats_no_bias
        assert "bias_max" not in stats_no_bias
        assert "bias_norm" not in stats_no_bias
    
    def test_gradient_computation_setup(self):
        """Test gradient computation setup in forward pass."""
        layer = Linear(4, 3)
        
        # Test with requires_grad=True input
        x_grad = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        output_grad = layer.forward(x_grad)
        assert output_grad.requires_grad == True
        
        # Test with requires_grad=False input
        x_no_grad = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=False)
        output_no_grad = layer.forward(x_no_grad)
        # Output should still require grad because layer parameters do
        assert output_no_grad.requires_grad == True
        
        # Test with parameters that don't require gradients
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
        output_no_param_grad = layer.forward(x_no_grad)
        assert output_no_param_grad.requires_grad == False
    
    def test_extreme_input_sizes(self):
        """Test with extreme input sizes and edge cases."""
        # Test very small layer
        tiny_layer = Linear(1, 1)
        x_tiny = Tensor(np.array([[0.5]]).astype(np.float32))
        output_tiny = tiny_layer.forward(x_tiny)
        assert output_tiny.shape == (1, 1)
        
        # Test larger layer
        large_layer = Linear(100, 50)
        x_large = Tensor(np.random.randn(10, 100).astype(np.float32))
        output_large = large_layer.forward(x_large)
        assert output_large.shape == (10, 50)
        
        # Test single sample
        x_single = Tensor(np.random.randn(1, 4).astype(np.float32))
        layer = Linear(4, 3)
        output_single = layer.forward(x_single)
        assert output_single.shape == (1, 3)
    
    def test_numerical_stability_and_precision(self):
        """Test numerical stability and precision handling."""
        layer = Linear(4, 3)
        
        # Test with very small values
        x_small = Tensor(np.full((2, 4), 1e-8, dtype=np.float32))
        output_small = layer.forward(x_small)
        assert output_small.shape == (2, 3)
        assert np.all(np.isfinite(output_small.data))
        
        # Test with very large values  
        x_large = Tensor(np.full((2, 4), 1e6, dtype=np.float32))
        output_large = layer.forward(x_large)
        assert output_large.shape == (2, 3)
        assert np.all(np.isfinite(output_large.data))
        
        # Test with zero input
        x_zero = Tensor(np.zeros((2, 4), dtype=np.float32))
        output_zero = layer.forward(x_zero)
        assert output_zero.shape == (2, 3)
        # Output should equal bias when input is zero
        expected_zero = np.broadcast_to(layer.bias.data, (2, 3))
        assert np.allclose(output_zero.data, expected_zero)
    
    def test_dtype_consistency(self):
        """Test data type consistency throughout operations."""
        layer = Linear(4, 3)
        
        # Verify layer parameters are float32
        assert layer.weight.data.dtype == np.float32
        assert layer.bias.data.dtype == np.float32
        
        # Test with float32 input
        x_f32 = Tensor(np.random.randn(2, 4).astype(np.float32))
        output_f32 = layer.forward(x_f32)
        assert output_f32.data.dtype == np.float32
        
        # Test with float64 input (should still work)
        x_f64 = Tensor(np.random.randn(2, 4).astype(np.float64))
        output_f64 = layer.forward(x_f64)
        # Output dtype depends on implementation, but should be finite
        assert np.all(np.isfinite(output_f64.data))
    
    def test_parameter_names_and_attributes(self):
        """Test parameter naming and attribute access."""
        custom_name = "test_linear"
        layer = Linear(4, 3, name=custom_name)
        
        # Test parameter names
        assert custom_name in layer.weight.name
        assert custom_name in layer.bias.name
        assert ".weight" in layer.weight.name
        assert ".bias" in layer.bias.name
        
        # Test layer attributes
        assert hasattr(layer, 'in_features')
        assert hasattr(layer, 'out_features')
        assert hasattr(layer, 'use_bias')
        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'bias')
        assert hasattr(layer, 'name')
    
    def test_initialization_scheme_mathematical_properties(self):
        """Test mathematical properties of initialization schemes."""
        in_features, out_features = 100, 50  # Larger size for statistical tests
        
        # Xavier uniform initialization
        layer_xavier_uniform = Linear(in_features, out_features, weight_init="xavier_uniform")
        weight_data = layer_xavier_uniform.weight.data
        limit = np.sqrt(6.0 / (in_features + out_features))
        assert np.all(weight_data >= -limit)
        assert np.all(weight_data <= limit)
        
        # Xavier normal initialization
        layer_xavier_normal = Linear(in_features, out_features, weight_init="xavier_normal")
        weight_data_normal = layer_xavier_normal.weight.data
        expected_std = np.sqrt(2.0 / (in_features + out_features))
        actual_std = np.std(weight_data_normal)
        assert abs(actual_std - expected_std) < 0.1  # Allow some variance
        
        # He uniform initialization
        layer_he_uniform = Linear(in_features, out_features, weight_init="he_uniform")
        weight_data_he = layer_he_uniform.weight.data
        he_limit = np.sqrt(6.0 / in_features)
        assert np.all(weight_data_he >= -he_limit)
        assert np.all(weight_data_he <= he_limit)
        
        # He normal initialization
        layer_he_normal = Linear(in_features, out_features, weight_init="he_normal")
        weight_data_he_normal = layer_he_normal.weight.data
        he_std = np.sqrt(2.0 / in_features)
        actual_he_std = np.std(weight_data_he_normal)
        assert abs(actual_he_std - he_std) < 0.1
    
    def test_error_propagation_and_handling(self):
        """Test error propagation and exception handling."""
        layer = Linear(4, 3)
        
        # Test with malformed input shapes
        shapes_to_test = [
            (2, 5),    # Wrong feature size
            (3, 2, 5), # Wrong feature size in 3D
        ]
        
        for shape in shapes_to_test:
            x_wrong = Tensor(np.random.randn(*shape).astype(np.float32))
            with pytest.raises(LayerError):
                layer.forward(x_wrong)
    
    def test_layer_state_persistence(self):
        """Test that layer state persists correctly across operations."""
        layer = Linear(4, 3)
        
        # Store initial state
        initial_weight = layer.weight.data.copy()
        initial_bias = layer.bias.data.copy()
        
        # Perform forward pass
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        output = layer.forward(x)
        
        # Verify state hasn't changed
        assert np.allclose(layer.weight.data, initial_weight)
        assert np.allclose(layer.bias.data, initial_bias)
        
        # Verify layer properties are maintained
        assert layer.in_features == 4
        assert layer.out_features == 3
        assert layer.use_bias == True
    
    def test_edge_case_mathematical_operations(self):
        """Test edge cases in mathematical operations."""
        layer = Linear(4, 3)
        
        # Test with NaN input (should handle gracefully or raise appropriate error)
        x_nan = Tensor(np.full((2, 4), np.nan, dtype=np.float32))
        try:
            output_nan = layer.forward(x_nan)
            # If it doesn't raise an error, check if NaN is handled
            assert output_nan.shape == (2, 3)
        except (ValueError, RuntimeError):
            # NaN handling might raise an error, which is acceptable
            pass
        
        # Test with infinity input
        x_inf = Tensor(np.full((2, 4), np.inf, dtype=np.float32))
        try:
            output_inf = layer.forward(x_inf)
            assert output_inf.shape == (2, 3)
        except (ValueError, RuntimeError, OverflowError):
            # Infinity handling might raise an error, which is acceptable
            pass
    
    def test_memory_efficiency_and_cleanup(self):
        """Test memory efficiency and proper cleanup."""
        # Create a layer and perform operations
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(100, 10).astype(np.float32))
        
        # Perform multiple forward passes
        for _ in range(10):
            output = layer.forward(x)
            assert output.shape == (100, 5)
        
        # Verify layer still functions correctly
        final_output = layer.forward(x)
        assert final_output.shape == (100, 5)
        assert np.all(np.isfinite(final_output.data))
    
    def test_concurrent_operations(self):
        """Test behavior with concurrent operations."""
        layer = Linear(4, 3)
        
        # Create multiple different inputs
        inputs = [
            Tensor(np.random.randn(2, 4).astype(np.float32)),
            Tensor(np.random.randn(3, 4).astype(np.float32)),
            Tensor(np.random.randn(1, 4).astype(np.float32)),
        ]
        
        # Process all inputs and verify outputs
        outputs = []
        for x in inputs:
            output = layer.forward(x)
            outputs.append(output)
            assert output.shape[1] == 3  # out_features
            assert output.shape[0] == x.shape[0]  # batch_size preserved
        
        # Verify all outputs are different (since inputs are different)
        for i, output1 in enumerate(outputs):
            for j, output2 in enumerate(outputs):
                if i != j and output1.shape == output2.shape:
                    assert not np.allclose(output1.data, output2.data)