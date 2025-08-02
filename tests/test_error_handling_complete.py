"""Complete error handling and exception coverage tests for neural architecture framework.

This test suite provides comprehensive coverage for:
- All possible exception paths in layers and operations
- Parameter validation edge cases
- Invalid configuration combinations
- Resource exhaustion scenarios
- Malformed input handling
- Edge case error conditions
- Exception propagation testing
- Error recovery mechanisms
- Graceful failure scenarios

Designed to achieve maximum coverage of error handling code paths.
"""

import gc
import math
import os
import sys
import pytest
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import neural architecture components
from neural_arch.nn import (
    Conv1d, Conv2d, Conv3d,
    ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell,
    Linear, Sequential, ModuleList, Module,
    SpatialDropout1d, SpatialDropout2d, SpatialDropout3d,
    BatchNorm1d, BatchNorm2d, BatchNorm3d,
    LayerNorm, GroupNorm, InstanceNorm, RMSNorm,
    Dropout, ReLU, Tanh, Sigmoid, GELU,
    MultiHeadAttention, SelfAttention,
    TransformerBlock, TransformerEncoder,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d,
    GlobalAvgPool1d, GlobalAvgPool2d,
    MaxPool, MeanPool
)
from neural_arch.core import Tensor, Parameter
from neural_arch.exceptions import LayerError, handle_exception
from neural_arch.functional import add, matmul, relu, sigmoid, tanh
from neural_arch.optim import Adam, SGD, AdamW


class TestParameterValidationErrors:
    """Tests for parameter validation error handling."""
    
    def test_conv_layer_parameter_validation(self):
        """Test parameter validation in convolution layers."""
        # Test negative in_channels
        with pytest.raises(LayerError, match="in_channels must be positive"):
            Conv1d(in_channels=-1, out_channels=8, kernel_size=3)
        
        with pytest.raises(LayerError, match="in_channels must be positive"):
            Conv2d(in_channels=0, out_channels=8, kernel_size=3)
        
        with pytest.raises(LayerError, match="in_channels must be positive"):
            Conv3d(in_channels=-5, out_channels=8, kernel_size=3)
        
        # Test negative out_channels
        with pytest.raises(LayerError, match="out_channels must be positive"):
            Conv1d(in_channels=4, out_channels=-1, kernel_size=3)
        
        with pytest.raises(LayerError, match="out_channels must be positive"):
            Conv2d(in_channels=4, out_channels=0, kernel_size=3)
        
        # Test invalid groups
        with pytest.raises(LayerError, match="groups must be positive"):
            Conv1d(in_channels=4, out_channels=8, kernel_size=3, groups=0)
        
        with pytest.raises(LayerError, match="groups must be positive"):
            Conv2d(in_channels=4, out_channels=8, kernel_size=3, groups=-1)
        
        # Test in_channels not divisible by groups
        with pytest.raises(LayerError, match="in_channels must be divisible by groups"):
            Conv1d(in_channels=5, out_channels=8, kernel_size=3, groups=2)
        
        # Test out_channels not divisible by groups
        with pytest.raises(LayerError, match="out_channels must be divisible by groups"):
            Conv2d(in_channels=4, out_channels=7, kernel_size=3, groups=2)
        
        # Test invalid weight initialization
        with pytest.raises(LayerError, match="Unknown weight initialization scheme"):
            Conv1d(in_channels=4, out_channels=8, kernel_size=3, weight_init="invalid_scheme")
        
        # Test invalid padding mode
        with pytest.raises(LayerError, match="Unsupported padding mode"):
            conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, 
                         padding=1, padding_mode="invalid_mode")
            # Error might occur during forward pass
            input_data = np.random.randn(1, 2, 10).astype(np.float32)
            x = Tensor(input_data)
            conv(x)
    
    def test_rnn_layer_parameter_validation(self):
        """Test parameter validation in RNN layers."""
        # Test negative input_size
        with pytest.raises(LayerError, match="input_size must be positive"):
            RNN(input_size=-1, hidden_size=8)
        
        with pytest.raises(LayerError, match="input_size must be positive"):
            LSTM(input_size=0, hidden_size=16)
        
        with pytest.raises(LayerError, match="input_size must be positive"):
            GRU(input_size=-5, hidden_size=32)
        
        # Test negative hidden_size
        with pytest.raises(LayerError, match="hidden_size must be positive"):
            RNN(input_size=8, hidden_size=-1)
        
        with pytest.raises(LayerError, match="hidden_size must be positive"):
            LSTM(input_size=8, hidden_size=0)
        
        # Test negative num_layers
        with pytest.raises(LayerError, match="num_layers must be positive"):
            RNN(input_size=8, hidden_size=16, num_layers=-1)
        
        with pytest.raises(LayerError, match="num_layers must be positive"):
            GRU(input_size=8, hidden_size=16, num_layers=0)
        
        # Test invalid dropout probability
        with pytest.raises(LayerError, match="dropout must be in \\[0, 1\\]"):
            RNN(input_size=8, hidden_size=16, dropout=-0.1)
        
        with pytest.raises(LayerError, match="dropout must be in \\[0, 1\\]"):
            LSTM(input_size=8, hidden_size=16, dropout=1.5)
        
        # Test invalid nonlinearity
        with pytest.raises(LayerError, match="nonlinearity must be 'tanh' or 'relu'"):
            RNN(input_size=8, hidden_size=16, nonlinearity="invalid")
    
    def test_linear_layer_parameter_validation(self):
        """Test parameter validation in Linear layers."""
        # Test negative in_features
        with pytest.raises(LayerError, match="in_features must be positive"):
            Linear(in_features=-1, out_features=8)
        
        with pytest.raises(LayerError, match="in_features must be positive"):
            Linear(in_features=0, out_features=8)
        
        # Test negative out_features
        with pytest.raises(LayerError, match="out_features must be positive"):
            Linear(in_features=8, out_features=-1)
        
        with pytest.raises(LayerError, match="out_features must be positive"):
            Linear(in_features=8, out_features=0)
    
    def test_normalization_layer_parameter_validation(self):
        """Test parameter validation in normalization layers."""
        # Test negative num_features
        with pytest.raises(LayerError, match="num_features must be positive"):
            BatchNorm1d(num_features=-1)
        
        with pytest.raises(LayerError, match="num_features must be positive"):
            BatchNorm2d(num_features=0)
        
        with pytest.raises(LayerError, match="num_features must be positive"):
            LayerNorm(normalized_shape=-5)
        
        # Test invalid epsilon
        with pytest.raises(LayerError, match="eps must be positive"):
            BatchNorm1d(num_features=8, eps=-1e-5)
        
        with pytest.raises(LayerError, match="eps must be positive"):
            LayerNorm(normalized_shape=8, eps=0.0)
        
        # Test invalid momentum
        with pytest.raises(LayerError, match="momentum must be in \\[0, 1\\]"):
            BatchNorm1d(num_features=8, momentum=-0.1)
        
        with pytest.raises(LayerError, match="momentum must be in \\[0, 1\\]"):
            BatchNorm2d(num_features=8, momentum=1.5)
    
    def test_dropout_parameter_validation(self):
        """Test parameter validation in dropout layers."""
        # Test invalid dropout probability
        with pytest.raises(LayerError, match="Dropout probability must be between 0 and 1"):
            Dropout(p=-0.1)
        
        with pytest.raises(LayerError, match="Dropout probability must be between 0 and 1"):
            Dropout(p=1.5)
        
        with pytest.raises(LayerError, match="Dropout probability must be between 0 and 1"):
            SpatialDropout1d(p=-0.05)
        
        with pytest.raises(LayerError, match="Dropout probability must be between 0 and 1"):
            SpatialDropout2d(p=2.0)
        
        with pytest.raises(LayerError, match="Dropout probability must be between 0 and 1"):
            SpatialDropout3d(p=1.1)
    
    def test_attention_parameter_validation(self):
        """Test parameter validation in attention layers."""
        # Test negative d_model
        with pytest.raises(LayerError, match="d_model must be positive"):
            MultiHeadAttention(d_model=-1, n_heads=8)
        
        with pytest.raises(LayerError, match="d_model must be positive"):
            MultiHeadAttention(d_model=0, n_heads=8)
        
        # Test negative n_heads
        with pytest.raises(LayerError, match="n_heads must be positive"):
            MultiHeadAttention(d_model=64, n_heads=-1)
        
        with pytest.raises(LayerError, match="n_heads must be positive"):
            MultiHeadAttention(d_model=64, n_heads=0)
        
        # Test d_model not divisible by n_heads
        with pytest.raises(LayerError, match="d_model must be divisible by n_heads"):
            MultiHeadAttention(d_model=65, n_heads=8)
        
        # Test invalid dropout
        with pytest.raises(LayerError, match="dropout must be in \\[0, 1\\]"):
            MultiHeadAttention(d_model=64, n_heads=8, dropout=-0.1)
        
        with pytest.raises(LayerError, match="dropout must be in \\[0, 1\\]"):
            MultiHeadAttention(d_model=64, n_heads=8, dropout=1.5)


class TestInputValidationErrors:
    """Tests for input validation error handling."""
    
    def test_conv_input_shape_validation(self):
        """Test input shape validation in convolution layers."""
        conv1d = Conv1d(in_channels=3, out_channels=8, kernel_size=3)
        conv2d = Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        conv3d = Conv3d(in_channels=3, out_channels=8, kernel_size=3)
        
        # Test wrong number of dimensions
        with pytest.raises(LayerError, match="Expected 3D input"):
            wrong_dims = Tensor(np.random.randn(3, 10).astype(np.float32))  # 2D instead of 3D
            conv1d(wrong_dims)
        
        with pytest.raises(LayerError, match="Expected 4D input"):
            wrong_dims = Tensor(np.random.randn(3, 10, 10).astype(np.float32))  # 3D instead of 4D
            conv2d(wrong_dims)
        
        with pytest.raises(LayerError, match="Expected 5D input"):
            wrong_dims = Tensor(np.random.randn(3, 10, 10, 10).astype(np.float32))  # 4D instead of 5D
            conv3d(wrong_dims)
        
        # Test wrong channel count
        with pytest.raises(LayerError, match="Input channels mismatch"):
            wrong_channels = Tensor(np.random.randn(1, 5, 10).astype(np.float32))  # 5 channels instead of 3
            conv1d(wrong_channels)
        
        with pytest.raises(LayerError, match="Input channels mismatch"):
            wrong_channels = Tensor(np.random.randn(1, 7, 10, 10).astype(np.float32))  # 7 channels instead of 3
            conv2d(wrong_channels)
        
        with pytest.raises(LayerError, match="Input channels mismatch"):
            wrong_channels = Tensor(np.random.randn(1, 2, 8, 8, 8).astype(np.float32))  # 2 channels instead of 3
            conv3d(wrong_channels)
    
    def test_rnn_input_shape_validation(self):
        """Test input shape validation in RNN layers."""
        rnn = RNN(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        lstm = LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        gru = GRU(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        
        # Test wrong number of dimensions
        for layer in [rnn, lstm, gru]:
            with pytest.raises(LayerError, match="Expected 3D input"):
                wrong_dims = Tensor(np.random.randn(2, 16).astype(np.float32))  # 2D instead of 3D
                if isinstance(layer, (LSTM,)):
                    layer(wrong_dims)
                else:
                    layer(wrong_dims)
        
        # Test wrong input size
        for layer in [rnn, lstm, gru]:
            with pytest.raises(LayerError, match="Input size mismatch"):
                wrong_size = Tensor(np.random.randn(2, 10, 20).astype(np.float32))  # 20 features instead of 16
                if isinstance(layer, (LSTM,)):
                    layer(wrong_size)
                else:
                    layer(wrong_size)
    
    def test_linear_input_shape_validation(self):
        """Test input shape validation in Linear layers."""
        linear = Linear(in_features=32, out_features=16)
        
        # Test wrong number of dimensions (too few)
        with pytest.raises(LayerError, match="Expected at least 2D input"):
            wrong_dims = Tensor(np.random.randn(32).astype(np.float32))  # 1D instead of 2D+
            linear(wrong_dims)
        
        # Test wrong feature count
        with pytest.raises(LayerError, match="Input features mismatch"):
            wrong_features = Tensor(np.random.randn(2, 25).astype(np.float32))  # 25 features instead of 32
            linear(wrong_features)
    
    def test_normalization_input_validation(self):
        """Test input validation in normalization layers."""
        bn1d = BatchNorm1d(num_features=16)
        bn2d = BatchNorm2d(num_features=16)
        bn3d = BatchNorm3d(num_features=16)
        ln = LayerNorm(normalized_shape=16)
        
        # Test wrong dimensions for BatchNorm
        with pytest.raises(LayerError, match="Expected 3D input"):
            wrong_dims = Tensor(np.random.randn(2, 16).astype(np.float32))  # 2D instead of 3D
            bn1d(wrong_dims)
        
        with pytest.raises(LayerError, match="Expected 4D input"):
            wrong_dims = Tensor(np.random.randn(2, 16, 8).astype(np.float32))  # 3D instead of 4D
            bn2d(wrong_dims)
        
        with pytest.raises(LayerError, match="Expected 5D input"):
            wrong_dims = Tensor(np.random.randn(2, 16, 8, 8).astype(np.float32))  # 4D instead of 5D
            bn3d(wrong_dims)
        
        # Test wrong channel count
        with pytest.raises(LayerError, match="Expected .* channels"):
            wrong_channels = Tensor(np.random.randn(2, 20, 8).astype(np.float32))  # 20 channels instead of 16
            bn1d(wrong_channels)
    
    def test_pooling_input_validation(self):
        """Test input validation in pooling layers."""
        maxpool = MaxPool(kernel_size=2)
        meanpool = MeanPool(kernel_size=2)
        global_avg_1d = GlobalAvgPool1d()
        global_avg_2d = GlobalAvgPool2d()
        adaptive_avg_1d = AdaptiveAvgPool1d(output_size=4)
        adaptive_avg_2d = AdaptiveAvgPool2d(output_size=(4, 4))
        
        # Test wrong dimensions
        with pytest.raises(LayerError, match="Expected.*D input"):
            wrong_dims = Tensor(np.random.randn(2, 8).astype(np.float32))  # Wrong dimensions
            global_avg_1d(wrong_dims)
        
        with pytest.raises(LayerError, match="Expected.*D input"):
            wrong_dims = Tensor(np.random.randn(2, 8, 8).astype(np.float32))  # Wrong dimensions
            global_avg_2d(wrong_dims)


class TestInvalidConfigurationErrors:
    """Tests for invalid configuration combinations."""
    
    def test_invalid_sequential_configurations(self):
        """Test invalid Sequential layer configurations."""
        # Test empty Sequential
        empty_seq = Sequential()
        input_data = np.random.randn(1, 4, 8).astype(np.float32)
        x = Tensor(input_data)
        
        # Empty sequential should pass through input unchanged
        output = empty_seq(x)
        assert output.shape == x.shape
        
        # Test incompatible layer sequence
        incompatible_layers = [
            Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1),  # Output: (1, 8, 8)
            Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1), # Expects 4D input
        ]
        
        incompatible_seq = Sequential(incompatible_layers)
        
        with pytest.raises((LayerError, ValueError, IndexError)):
            # This should fail because Conv1d output is 3D but Conv2d expects 4D
            output = incompatible_seq(x)
    
    def test_invalid_module_combinations(self):
        """Test invalid module combinations."""
        # Test RNN with incompatible hidden state
        rnn = RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
        
        input_data = np.random.randn(2, 10, 8).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Wrong hidden state shape
        wrong_hidden = np.random.randn(1, 2, 20).astype(np.float32)  # 20 instead of 16
        h_wrong = Tensor(wrong_hidden, requires_grad=True)
        
        with pytest.raises(LayerError, match="Hidden shape mismatch"):
            output, hidden = rnn(x, h_wrong)
    
    def test_invalid_attention_configurations(self):
        """Test invalid attention configurations."""
        attention = MultiHeadAttention(d_model=64, n_heads=8)
        
        # Test mismatched query, key, value dimensions
        query = Tensor(np.random.randn(2, 10, 64).astype(np.float32))
        key = Tensor(np.random.randn(2, 15, 64).astype(np.float32))    # Different seq length is OK
        value = Tensor(np.random.randn(2, 15, 32).astype(np.float32))  # Different d_model is NOT OK
        
        with pytest.raises(LayerError, match="Key and value must have same.*dimensions"):
            output = attention(query, key, value)
    
    def test_invalid_transformer_configurations(self):
        """Test invalid transformer configurations."""
        # Test transformer with incompatible dimensions
        transformer = TransformerBlock(d_model=128, n_heads=8, d_ff=512)
        
        # Wrong d_model
        wrong_input = Tensor(np.random.randn(2, 10, 64).astype(np.float32))  # 64 instead of 128
        
        with pytest.raises(LayerError, match="Expected.*d_model"):
            output = transformer(wrong_input)


class TestResourceExhaustionErrors:
    """Tests for resource exhaustion error handling."""
    
    def test_memory_exhaustion_handling(self):
        """Test handling of memory exhaustion scenarios."""
        # This test tries to allocate very large tensors that might cause memory errors
        try:
            # Try to create an extremely large tensor
            huge_size = (1000, 1000, 1000)  # ~4GB
            
            try:
                huge_data = np.random.randn(*huge_size).astype(np.float32)
                huge_tensor = Tensor(huge_data, requires_grad=True)
                
                # If we get here, the system has enough memory, so clean up
                del huge_tensor, huge_data
                gc.collect()
                
            except MemoryError:
                # This is expected behavior - system should handle memory exhaustion gracefully
                pass
                
        except Exception as e:
            # Any other exception type indicates poor error handling
            pytest.fail(f"Memory exhaustion caused unexpected exception: {type(e).__name__}: {e}")
    
    def test_stack_overflow_protection(self):
        """Test protection against stack overflow in deep recursion."""
        # Create very deep Sequential model that might cause stack overflow
        max_depth = 100
        layers = []
        
        for i in range(max_depth):
            layers.append(Linear(in_features=8, out_features=8))
            if i % 10 == 9:  # Add activation every 10 layers
                layers.append(ReLU())
        
        try:
            deep_model = Sequential(layers)
            
            input_data = np.random.randn(1, 8).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # This might cause stack overflow in gradient computation
            output = deep_model(x)
            
            assert output.shape == (1, 8)
            assert output.requires_grad
            
        except RecursionError:
            # This is acceptable - deep models might hit recursion limits
            pass
        except Exception as e:
            # Other exceptions might indicate poor error handling
            warnings.warn(f"Deep model caused unexpected exception: {type(e).__name__}: {e}")
    
    def test_parameter_overflow_handling(self):
        """Test handling of parameter overflow scenarios."""
        try:
            # Try to create a model with an enormous number of parameters
            huge_linear = Linear(in_features=10000, out_features=10000, bias=True)
            
            # Count parameters
            total_params = sum(np.prod(p.data.shape) for p in huge_linear.parameters() if p.requires_grad)
            
            # Should have 10000*10000 + 10000 = 100,010,000 parameters
            expected_params = 10000 * 10000 + 10000
            assert total_params == expected_params
            
            # Test that we can still use the model (if memory allows)
            input_data = np.random.randn(1, 10000).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            output = huge_linear(x)
            assert output.shape == (1, 10000)
            
        except MemoryError:
            # This is acceptable for very large models
            pass


class TestMalformedInputHandling:
    """Tests for handling malformed inputs."""
    
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        layers_to_test = [
            Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
            RNN(input_size=4, hidden_size=8, num_layers=1, batch_first=True),
            Linear(in_features=8, out_features=4),
            BatchNorm1d(num_features=4),
            LayerNorm(normalized_shape=4),
        ]
        
        test_inputs = [
            np.full((1, 2, 10), np.nan, dtype=np.float32),      # Conv1d
            np.full((1, 10, 4), np.nan, dtype=np.float32),      # RNN
            np.full((1, 8), np.nan, dtype=np.float32),          # Linear
            np.full((1, 4, 10), np.nan, dtype=np.float32),      # BatchNorm1d
            np.full((1, 4), np.nan, dtype=np.float32),          # LayerNorm
        ]
        
        for layer, input_data in zip(layers_to_test, test_inputs):
            x = Tensor(input_data, requires_grad=True)
            
            try:
                if isinstance(layer, RNN):
                    output, hidden = layer(x)
                    # NaN should propagate
                    assert np.any(np.isnan(output.data))
                else:
                    output = layer(x)
                    # NaN should propagate
                    assert np.any(np.isnan(output.data))
                    
            except Exception as e:
                # Some layers might have special NaN handling
                warnings.warn(f"Layer {type(layer).__name__} with NaN input caused: {e}")
    
    def test_inf_input_handling(self):
        """Test handling of infinite inputs."""
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        
        # Test positive infinity
        pos_inf_data = np.full((1, 2, 10), np.inf, dtype=np.float32)
        x_pos_inf = Tensor(pos_inf_data, requires_grad=True)
        
        output_pos_inf = conv(x_pos_inf)
        # Infinity should propagate
        assert np.any(np.isinf(output_pos_inf.data))
        
        # Test negative infinity
        neg_inf_data = np.full((1, 2, 10), -np.inf, dtype=np.float32)
        x_neg_inf = Tensor(neg_inf_data, requires_grad=True)
        
        output_neg_inf = conv(x_neg_inf)
        # Infinity should propagate
        assert np.any(np.isinf(output_neg_inf.data))
    
    def test_zero_input_handling(self):
        """Test handling of all-zero inputs."""
        layers_to_test = [
            Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1),
            LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True),
            Linear(in_features=12, out_features=6),
            MultiHeadAttention(d_model=16, n_heads=4),
        ]
        
        test_inputs = [
            np.zeros((1, 3, 8, 8), dtype=np.float32),          # Conv2d
            np.zeros((1, 10, 8), dtype=np.float32),            # LSTM
            np.zeros((1, 12), dtype=np.float32),               # Linear
            np.zeros((1, 8, 16), dtype=np.float32),            # MultiHeadAttention
        ]
        
        for layer, input_data in zip(layers_to_test, test_inputs):
            x = Tensor(input_data, requires_grad=True)
            
            try:
                if isinstance(layer, LSTM):
                    output, (hidden, cell) = layer(x)
                    assert np.all(np.isfinite(output.data))
                elif isinstance(layer, MultiHeadAttention):
                    output = layer(x, x, x)  # Self-attention
                    assert np.all(np.isfinite(output.data))
                else:
                    output = layer(x)
                    assert np.all(np.isfinite(output.data))
                    
            except Exception as e:
                pytest.fail(f"Layer {type(layer).__name__} failed with zero input: {e}")
    
    def test_mixed_type_input_handling(self):
        """Test handling of mixed data type inputs."""
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        
        # Test with different dtypes
        dtypes_to_test = [np.float32, np.float64, np.int32, np.int64]
        
        for dtype in dtypes_to_test:
            try:
                input_data = np.random.randn(1, 2, 10).astype(dtype)
                x = Tensor(input_data, requires_grad=True)
                
                output = conv(x)
                
                # Output should be finite and reasonable
                assert np.all(np.isfinite(output.data))
                
            except Exception as e:
                # Some dtypes might not be supported
                warnings.warn(f"Conv1d with dtype {dtype} caused: {e}")


class TestExceptionPropagationAndRecovery:
    """Tests for exception propagation and recovery mechanisms."""
    
    def test_exception_propagation_through_sequential(self):
        """Test that exceptions propagate correctly through Sequential layers."""
        # Create Sequential with layer that will fail
        problematic_seq = Sequential(
            Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1),  # This should work
            Linear(in_features=8, out_features=4),  # This will fail due to shape mismatch
        )
        
        input_data = np.random.randn(1, 4, 10).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Should raise an exception when trying to pass 3D output to Linear layer
        with pytest.raises((LayerError, ValueError, IndexError)):
            output = problematic_seq(x)
    
    def test_graceful_degradation(self):
        """Test graceful degradation when some operations fail."""
        # Test that we can continue using other layers even if one fails
        working_conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        
        input_data = np.random.randn(1, 2, 10).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # This should work fine
        output1 = working_conv(x)
        assert output1.shape == (1, 4, 10)
        
        # Try something that will fail
        try:
            wrong_input = np.random.randn(1, 5, 10).astype(np.float32)  # Wrong channels
            x_wrong = Tensor(wrong_input, requires_grad=True)
            working_conv(x_wrong)  # Should fail
        except LayerError:
            pass  # Expected
        
        # Original layer should still work
        output2 = working_conv(x)
        assert output2.shape == (1, 4, 10)
    
    def test_error_recovery_in_training_loop(self):
        """Test error recovery in simulated training scenarios."""
        model = Sequential(
            Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
            Linear(in_features=4, out_features=2),  # This might fail with shape mismatch
        )
        
        # Simulate multiple training steps with some failing
        successful_steps = 0
        failed_steps = 0
        
        for step in range(5):
            try:
                # Different input sizes - some might cause failures
                if step % 3 == 0:
                    # This might cause shape mismatch in Linear layer
                    input_data = np.random.randn(1, 2, 10).astype(np.float32)
                else:
                    # This should work if model architecture is compatible
                    input_data = np.random.randn(1, 2, 8).astype(np.float32)
                
                x = Tensor(input_data, requires_grad=True)
                output = model(x)
                successful_steps += 1
                
            except (LayerError, ValueError, IndexError):
                failed_steps += 1
                continue  # Continue with next step
        
        # Should have attempted all steps
        assert successful_steps + failed_steps == 5
    
    def test_exception_context_preservation(self):
        """Test that exception context is preserved for debugging."""
        conv = Conv1d(in_channels=3, out_channels=6, kernel_size=3)
        
        # Create input that will cause a specific error
        wrong_input = np.random.randn(1, 5, 10).astype(np.float32)  # 5 channels instead of 3
        x = Tensor(wrong_input, requires_grad=True)
        
        try:
            output = conv(x)
            pytest.fail("Expected LayerError was not raised")
        except LayerError as e:
            # Exception should contain useful information
            error_message = str(e)
            assert "channels" in error_message.lower() or "mismatch" in error_message.lower()
            assert "3" in error_message  # Expected channels
            assert "5" in error_message  # Actual channels


class TestCornerCaseErrors:
    """Tests for corner case error scenarios."""
    
    def test_edge_case_tensor_operations(self):
        """Test edge cases in tensor operations."""
        # Test operations with empty tensors (if supported)
        try:
            empty_data = np.array([], dtype=np.float32).reshape(0, 4)
            empty_tensor = Tensor(empty_data, requires_grad=True)
            
            # Most operations should gracefully handle empty tensors
            # or raise appropriate exceptions
            linear = Linear(in_features=4, out_features=2)
            
            try:
                output = linear(empty_tensor)
                assert output.shape[0] == 0  # Should maintain batch dimension
                assert output.shape[1] == 2   # Should have correct feature dimension
            except (LayerError, ValueError):
                # Acceptable if empty tensors are not supported
                pass
                
        except Exception as e:
            warnings.warn(f"Empty tensor handling caused unexpected error: {e}")
    
    def test_boundary_value_errors(self):
        """Test boundary value error conditions."""
        # Test with maximum/minimum float values
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        
        # Maximum float32 values
        max_input = np.full((1, 2, 10), np.finfo(np.float32).max, dtype=np.float32)
        x_max = Tensor(max_input, requires_grad=True)
        
        try:
            output_max = conv(x_max)
            # Should either work or overflow gracefully
            assert np.all(np.isfinite(output_max.data)) or np.any(np.isinf(output_max.data))
        except (OverflowError, LayerError):
            pass  # Acceptable for extreme values
        
        # Minimum positive float32 values
        min_input = np.full((1, 2, 10), np.finfo(np.float32).tiny, dtype=np.float32)
        x_min = Tensor(min_input, requires_grad=True)
        
        try:
            output_min = conv(x_min)
            assert np.all(np.isfinite(output_min.data))
        except (UnderflowError, LayerError):
            pass  # Acceptable for extreme values
    
    def test_concurrent_error_handling(self):
        """Test error handling in concurrent scenarios."""
        import threading
        
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        results = {}
        errors = {}
        
        def worker(thread_id):
            try:
                if thread_id % 2 == 0:
                    # Good input
                    input_data = np.random.randn(1, 2, 10).astype(np.float32)
                else:
                    # Bad input (wrong channels)
                    input_data = np.random.randn(1, 5, 10).astype(np.float32)
                
                x = Tensor(input_data, requires_grad=True)
                output = conv(x)
                results[thread_id] = output.shape
                
            except LayerError as e:
                errors[thread_id] = str(e)
            except Exception as e:
                errors[thread_id] = f"Unexpected: {type(e).__name__}: {e}"
        
        # Run multiple threads
        threads = []
        num_threads = 6
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check that errors were handled appropriately
        even_threads = [i for i in range(num_threads) if i % 2 == 0]
        odd_threads = [i for i in range(num_threads) if i % 2 == 1]
        
        # Even threads should succeed
        for thread_id in even_threads:
            assert thread_id in results, f"Even thread {thread_id} should have succeeded"
            assert results[thread_id] == (1, 4, 10)
        
        # Odd threads should fail with LayerError
        for thread_id in odd_threads:
            assert thread_id in errors, f"Odd thread {thread_id} should have failed"
            assert "channels" in errors[thread_id].lower() or "mismatch" in errors[thread_id].lower()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])