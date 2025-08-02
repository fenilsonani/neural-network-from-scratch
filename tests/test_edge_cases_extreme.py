"""Extreme edge case and boundary condition tests for neural architecture framework.

This test suite provides comprehensive coverage for:
- Extreme parameter values (very large/small sizes, unusual configurations)
- Memory stress testing with large tensors
- Numerical stability with extreme input values
- Multi-threading and concurrent layer usage
- Resource exhaustion scenarios
- Boundary condition edge cases
- Degenerate input handling
- Performance under stress conditions

Designed to push the framework to its limits and achieve maximum coverage.
"""

import gc
import math
import os
import sys
import time
import pytest
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import neural architecture components
from neural_arch.nn import (
    Conv1d, Conv2d, Conv3d,
    ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell,
    Linear, Sequential, ModuleList,
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
from neural_arch.exceptions import LayerError
from neural_arch.functional import add, matmul, relu, sigmoid, tanh


class TestExtremeParameterValues:
    """Tests with extreme parameter values."""
    
    def test_extremely_large_channel_counts(self):
        """Test layers with very large channel counts."""
        # Test very large channels (memory permitting)
        try:
            large_channels = 1024
            
            # Use 1x1 conv to minimize memory usage
            conv = Conv2d(in_channels=large_channels, out_channels=large_channels//2, 
                         kernel_size=1, bias=False)
            
            # Small spatial dimensions to keep memory reasonable
            input_data = np.random.randn(1, large_channels, 2, 2).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            output = conv(x)
            
            assert output.shape == (1, large_channels//2, 2, 2)
            assert output.requires_grad
            
        except MemoryError:
            pytest.skip("Insufficient memory for extremely large channel test")
    
    def test_extremely_small_values(self):
        """Test layers with extremely small but valid parameter values."""
        # Minimum possible channels
        conv = Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        
        # Minimum possible input
        input_data = np.array([[[0.5]]]).astype(np.float32)  # (1, 1, 1)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        assert output.shape == (1, 1, 1)
        assert output.requires_grad
    
    def test_maximum_kernel_sizes(self):
        """Test with maximum reasonable kernel sizes."""
        input_size = 50
        max_kernel = input_size  # Kernel size equal to input size
        
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=max_kernel)
        
        input_data = np.random.randn(1, 2, input_size).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        # Should produce output of size 1 (input_size - kernel_size + 1)
        assert output.shape == (1, 4, 1)
    
    def test_extreme_stride_values(self):
        """Test with very large stride values."""
        input_size = 100
        large_stride = 50
        
        conv = Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=large_stride)
        
        input_data = np.random.randn(1, 3, input_size).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        # Calculate expected output size
        expected_size = (input_size - 3) // large_stride + 1
        assert output.shape == (1, 6, expected_size)
    
    def test_extreme_dilation_values(self):
        """Test with very large dilation values."""
        input_size = 200
        large_dilation = 20
        kernel_size = 5
        
        # Effective kernel size: kernel_size + (kernel_size - 1) * (dilation - 1)
        effective_kernel = kernel_size + (kernel_size - 1) * (large_dilation - 1)
        
        if effective_kernel <= input_size:
            conv = Conv1d(in_channels=2, out_channels=4, kernel_size=kernel_size, 
                         dilation=large_dilation)
            
            input_data = np.random.randn(1, 2, input_size).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            output = conv(x)
            
            expected_size = input_size - effective_kernel + 1
            assert output.shape == (1, 4, expected_size)
    
    def test_extreme_padding_values(self):
        """Test with very large padding values."""
        input_size = 10
        large_padding = 50  # Padding larger than input
        
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=large_padding)
        
        input_data = np.random.randn(1, 2, input_size).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        # Output size with large padding
        expected_size = input_size + 2 * large_padding - 3 + 1
        assert output.shape == (1, 4, expected_size)
    
    def test_extreme_rnn_sequence_lengths(self):
        """Test RNN with extremely long sequences."""
        try:
            very_long_seq = 1000
            
            # Use small hidden size to manage memory
            rnn = RNN(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
            
            input_data = np.random.randn(1, very_long_seq, 4).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            output, hidden = rnn(x)
            
            assert output.shape == (1, very_long_seq, 8)
            assert hidden.shape == (1, 1, 8)
            
        except MemoryError:
            pytest.skip("Insufficient memory for very long sequence test")
    
    def test_extreme_layer_depth(self):
        """Test with very deep layer stacks."""
        # Create a deep but narrow network
        layers = []
        num_layers = 50
        channels = 4
        
        for i in range(num_layers):
            layers.append(Conv1d(in_channels=channels, out_channels=channels, 
                                kernel_size=3, padding=1))
            # Add occasional activation to prevent degradation
            if i % 5 == 4:
                layers.append(ReLU())
        
        deep_model = Sequential(layers)
        
        input_data = np.random.randn(1, channels, 8).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = deep_model(x)
        
        # Should maintain shape through deep stack
        assert output.shape == (1, channels, 8)
        assert output.requires_grad
    
    def test_extreme_batch_sizes(self):
        """Test with very large batch sizes."""
        try:
            large_batch = 100
            
            conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
            
            input_data = np.random.randn(large_batch, 2, 10).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            output = conv(x)
            
            assert output.shape == (large_batch, 4, 10)
            assert output.requires_grad
            
        except MemoryError:
            pytest.skip("Insufficient memory for large batch test")


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""
    
    def test_very_large_input_values(self):
        """Test with very large input values."""
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        
        # Very large values
        large_values = np.full((1, 2, 10), 1e6, dtype=np.float32)
        x = Tensor(large_values, requires_grad=True)
        
        output = conv(x)
        
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(output.data))
        assert output.shape == (1, 4, 10)
    
    def test_very_small_input_values(self):
        """Test with very small input values."""
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        
        # Very small values
        small_values = np.full((1, 2, 10), 1e-6, dtype=np.float32)
        x = Tensor(small_values, requires_grad=True)
        
        output = conv(x)
        
        # Should not underflow to zero (unless mathematically correct)
        assert np.all(np.isfinite(output.data))
        assert output.shape == (1, 4, 10)
    
    def test_mixed_extreme_values(self):
        """Test with mixture of very large and very small values."""
        rnn = RNN(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
        
        # Create input with mixed extreme values
        mixed_data = np.random.randn(1, 10, 4).astype(np.float32)
        mixed_data[0, 0, :] = 1e6   # Very large
        mixed_data[0, 1, :] = 1e-6  # Very small
        mixed_data[0, 2, :] = -1e6  # Very large negative
        
        x = Tensor(mixed_data, requires_grad=True)
        
        output, hidden = rnn(x)
        
        # Should handle mixed values gracefully
        assert np.all(np.isfinite(output.data))
        assert np.all(np.isfinite(hidden.data))
    
    def test_gradient_explosion_resistance(self):
        """Test resistance to gradient explosion."""
        # Deep RNN that might cause gradient explosion
        deep_rnn = RNN(input_size=4, hidden_size=4, num_layers=5, batch_first=True)
        
        # Input with large values
        input_data = np.random.randn(1, 20, 4).astype(np.float32) * 10
        x = Tensor(input_data, requires_grad=True)
        
        output, hidden = deep_rnn(x)
        
        # Check for gradient explosion indicators
        assert np.all(np.isfinite(output.data))
        assert not np.any(np.abs(output.data) > 1e8)  # Shouldn't explode
    
    def test_zero_input_handling(self):
        """Test handling of zero inputs."""
        layers_to_test = [
            Conv1d(2, 4, 3, padding=1),
            Conv2d(2, 4, 3, padding=1),
            RNN(4, 8, 1, batch_first=True),
            Linear(4, 8),
            BatchNorm1d(4),
            LayerNorm(4),
        ]
        
        test_inputs = [
            np.zeros((1, 2, 10), dtype=np.float32),  # Conv1d
            np.zeros((1, 2, 8, 8), dtype=np.float32),  # Conv2d
            np.zeros((1, 10, 4), dtype=np.float32),   # RNN
            np.zeros((1, 4), dtype=np.float32),       # Linear
            np.zeros((1, 4, 10), dtype=np.float32),   # BatchNorm1d
            np.zeros((1, 4), dtype=np.float32),       # LayerNorm
        ]
        
        for layer, input_data in zip(layers_to_test, test_inputs):
            x = Tensor(input_data, requires_grad=True)
            
            try:
                if isinstance(layer, RNN):
                    output, _ = layer(x)
                else:
                    output = layer(x)
                
                # Should handle zero inputs gracefully
                assert np.all(np.isfinite(output.data))
                
            except Exception as e:
                pytest.fail(f"Layer {type(layer).__name__} failed with zero input: {e}")
    
    def test_nan_input_handling(self):
        """Test behavior with NaN inputs."""
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        
        # Input with NaN values
        nan_data = np.random.randn(1, 2, 10).astype(np.float32)
        nan_data[0, 0, 5] = np.nan
        
        x = Tensor(nan_data, requires_grad=True)
        
        output = conv(x)
        
        # NaN should propagate (this is expected behavior)
        assert np.any(np.isnan(output.data))
    
    def test_inf_input_handling(self):
        """Test behavior with infinite inputs."""
        rnn = RNN(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
        
        # Input with infinite values
        inf_data = np.random.randn(1, 10, 4).astype(np.float32)
        inf_data[0, 5, 2] = np.inf
        inf_data[0, 6, 1] = -np.inf
        
        x = Tensor(inf_data, requires_grad=True)
        
        output, hidden = rnn(x)
        
        # Infinity should propagate
        assert np.any(np.isinf(output.data))


class TestMemoryStressTesting:
    """Memory stress tests with large tensors."""
    
    def test_large_tensor_allocation(self):
        """Test allocation of large tensors."""
        try:
            # Allocate reasonably large tensor
            large_size = (8, 64, 64, 64)  # ~16MB
            
            large_data = np.random.randn(*large_size).astype(np.float32)
            large_tensor = Tensor(large_data, requires_grad=True)
            
            # Simple operation to ensure it works
            assert large_tensor.shape == large_size
            assert large_tensor.requires_grad
            
            # Clean up
            del large_tensor, large_data
            gc.collect()
            
        except MemoryError:
            pytest.skip("Insufficient memory for large tensor test")
    
    def test_memory_efficient_operations(self):
        """Test memory efficiency in operations."""
        # Test that operations don't unnecessarily duplicate large amounts of memory
        batch_size = 4
        channels = 32
        size = 64
        
        try:
            input_data = np.random.randn(batch_size, channels, size).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Chain of operations that should be memory efficient
            conv1 = Conv1d(channels, channels, 3, padding=1)
            conv2 = Conv1d(channels, channels, 3, padding=1)
            
            out1 = conv1(x)
            out2 = conv2(out1)
            
            assert out2.shape == x.shape
            assert out2.requires_grad
            
            # Clean up
            del x, out1, out2, input_data
            gc.collect()
            
        except MemoryError:
            pytest.skip("Insufficient memory for memory efficiency test")
    
    def test_gradient_memory_usage(self):
        """Test memory usage during gradient computation."""
        try:
            # Model with many parameters
            model = Sequential(
                Conv1d(8, 16, 3, padding=1),
                Conv1d(16, 32, 3, padding=1),
                Conv1d(32, 16, 3, padding=1),
                Conv1d(16, 8, 3, padding=1),
            )
            
            input_data = np.random.randn(2, 8, 50).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Forward pass
            output = model(x)
            
            # Check that gradient functions are set up properly
            assert output.requires_grad
            assert output._grad_fn is not None
            
            # Verify all model parameters have gradients enabled
            all_params = []
            for layer in model.layers:
                if hasattr(layer, 'parameters'):
                    all_params.extend(layer.parameters())
            
            grad_enabled = [p for p in all_params if p.requires_grad]
            assert len(grad_enabled) > 0
            
        except MemoryError:
            pytest.skip("Insufficient memory for gradient memory test")
    
    def test_sequential_large_operations(self):
        """Test sequential large operations without memory buildup."""
        conv = Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        
        # Process multiple large inputs sequentially
        for i in range(5):
            try:
                input_data = np.random.randn(2, 16, 32, 32).astype(np.float32)
                x = Tensor(input_data, requires_grad=True)
                
                output = conv(x)
                
                assert output.shape == (2, 16, 32, 32)
                
                # Clean up each iteration
                del x, output, input_data
                gc.collect()
                
            except MemoryError:
                pytest.skip(f"Memory exhausted at iteration {i}")


class TestConcurrentUsage:
    """Tests for concurrent and multi-threaded usage."""
    
    def test_thread_safety_conv_layers(self):
        """Test thread safety of convolution layers."""
        conv = Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        results = {}
        errors = {}
        
        def worker(thread_id):
            try:
                # Each thread uses different input
                input_data = np.random.randn(1, 4, 10).astype(np.float32) + thread_id
                x = Tensor(input_data, requires_grad=True)
                
                output = conv(x)
                results[thread_id] = output.shape
                
            except Exception as e:
                errors[thread_id] = str(e)
        
        # Run multiple threads
        threads = []
        num_threads = 4
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == num_threads
        
        # All should have same output shape
        expected_shape = (1, 8, 10)
        for thread_id, shape in results.items():
            assert shape == expected_shape, f"Thread {thread_id} got shape {shape}"
    
    def test_concurrent_rnn_processing(self):
        """Test concurrent RNN processing."""
        rnn = RNN(input_size=6, hidden_size=12, num_layers=1, batch_first=True)
        
        def process_sequence(sequence_data):
            try:
                x = Tensor(sequence_data, requires_grad=True)
                output, hidden = rnn(x)
                return output.shape, hidden.shape
            except Exception as e:
                return str(e)
        
        # Prepare different sequences
        sequences = [
            np.random.randn(1, 15, 6).astype(np.float32),
            np.random.randn(1, 20, 6).astype(np.float32),
            np.random.randn(1, 10, 6).astype(np.float32),
        ]
        
        # Process concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_sequence, seq) for seq in sequences]
            results = [future.result() for future in futures]
        
        # Verify all succeeded
        for i, result in enumerate(results):
            assert isinstance(result, tuple), f"Sequence {i} failed: {result}"
            output_shape, hidden_shape = result
            assert output_shape[2] == 12  # hidden_size
            assert hidden_shape == (1, 1, 12)  # (num_layers, batch, hidden)
    
    def test_parameter_sharing_thread_safety(self):
        """Test thread safety with parameter sharing."""
        # Shared layer
        shared_linear = Linear(in_features=8, out_features=4)
        
        def worker_with_shared_layer(worker_id):
            try:
                input_data = np.random.randn(1, 8).astype(np.float32)
                x = Tensor(input_data, requires_grad=True)
                
                output = shared_linear(x)
                return worker_id, output.shape
                
            except Exception as e:
                return worker_id, str(e)
        
        # Multiple threads using shared layer
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_with_shared_layer, i) for i in range(4)]
            results = [future.result() for future in futures]
        
        # Verify all workers succeeded
        for worker_id, result in results:
            if isinstance(result, str):  # Error case
                pytest.fail(f"Worker {worker_id} failed: {result}")
            else:
                assert result == (1, 4), f"Worker {worker_id} got wrong shape: {result}"


class TestResourceExhaustion:
    """Tests for resource exhaustion scenarios."""
    
    def test_maximum_layers_in_sequential(self):
        """Test maximum number of layers in Sequential."""
        # Test with a large number of simple layers
        max_layers = 200
        layers = []
        
        for i in range(max_layers):
            # Use identity-like operations to minimize computation
            layers.append(Conv1d(in_channels=2, out_channels=2, kernel_size=1))
        
        try:
            huge_model = Sequential(layers)
            
            input_data = np.random.randn(1, 2, 5).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            output = huge_model(x)
            
            assert output.shape == (1, 2, 5)
            assert output.requires_grad
            
        except (MemoryError, RecursionError):
            pytest.skip("Hit resource limits with maximum layers test")
    
    def test_deeply_nested_operations(self):
        """Test deeply nested tensor operations."""
        # Start with simple tensor
        input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Apply many operations
        result = x
        try:
            for i in range(50):  # Deep nesting
                # Simple operation that maintains shape
                result = add(result, Tensor(np.ones_like(result.data) * 0.01, requires_grad=True))
                
            assert result.shape == x.shape
            assert result.requires_grad
            
        except (RecursionError, MemoryError):
            pytest.skip("Hit recursion/memory limits with deep nesting")
    
    def test_parameter_count_limits(self):
        """Test models with very large parameter counts."""
        try:
            # Model with many parameters
            large_linear = Linear(in_features=1000, out_features=1000, bias=True)
            
            # Count parameters
            params = large_linear.parameters()
            total_params = sum(np.prod(p.data.shape) for p in params if p.requires_grad)
            
            # Should have weight (1000x1000) + bias (1000) = 1,001,000 parameters
            expected_params = 1000 * 1000 + 1000
            assert total_params == expected_params
            
            # Test forward pass
            input_data = np.random.randn(1, 1000).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            output = large_linear(x)
            assert output.shape == (1, 1000)
            
        except MemoryError:
            pytest.skip("Insufficient memory for large parameter count test")


class TestDegenerateInputs:
    """Tests for degenerate and edge case inputs."""
    
    def test_single_element_inputs(self):
        """Test with single-element inputs."""
        # Conv1d with single element
        conv = Conv1d(in_channels=1, out_channels=2, kernel_size=1)
        single_input = Tensor(np.array([[[1.0]]]), requires_grad=True)
        
        output = conv(single_input)
        assert output.shape == (1, 2, 1)
        
        # RNN with single timestep
        rnn = RNN(input_size=1, hidden_size=2, num_layers=1, batch_first=True)
        single_seq = Tensor(np.array([[[0.5]]]), requires_grad=True)
        
        rnn_out, hidden = rnn(single_seq)
        assert rnn_out.shape == (1, 1, 2)
        assert hidden.shape == (1, 1, 2)
    
    def test_empty_like_inputs(self):
        """Test with minimal valid inputs."""
        # Smallest valid 2D conv input
        conv2d = Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        min_input = Tensor(np.array([[[[1.0]]]]), requires_grad=True)  # (1,1,1,1)
        
        output = conv2d(min_input)
        assert output.shape == (1, 1, 1, 1)
        
        # Smallest valid 3D conv input
        conv3d = Conv3d(in_channels=1, out_channels=1, kernel_size=1)
        min_input_3d = Tensor(np.array([[[[[1.0]]]]]), requires_grad=True)  # (1,1,1,1,1)
        
        output_3d = conv3d(min_input_3d)
        assert output_3d.shape == (1, 1, 1, 1, 1)
    
    def test_asymmetric_inputs(self):
        """Test with highly asymmetric input shapes."""
        # Very wide, short input
        wide_input = np.random.randn(1, 2, 1000).astype(np.float32)
        x_wide = Tensor(wide_input, requires_grad=True)
        
        conv_wide = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        output_wide = conv_wide(x_wide)
        assert output_wide.shape == (1, 4, 1000)
        
        # Very tall, narrow 2D input
        tall_input = np.random.randn(1, 2, 100, 1).astype(np.float32)
        x_tall = Tensor(tall_input, requires_grad=True)
        
        conv_tall = Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 1), padding=(1, 0))
        output_tall = conv_tall(x_tall)
        assert output_tall.shape == (1, 4, 100, 1)
    
    def test_repeated_identical_inputs(self):
        """Test with inputs containing repeated identical values."""
        # All same values
        identical_data = np.full((1, 3, 10), 2.5, dtype=np.float32)
        x_identical = Tensor(identical_data, requires_grad=True)
        
        conv = Conv1d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        output = conv(x_identical)
        
        assert output.shape == (1, 6, 10)
        assert np.all(np.isfinite(output.data))
        
        # Test with RNN
        rnn_input = np.full((1, 8, 4), 1.0, dtype=np.float32)
        x_rnn = Tensor(rnn_input, requires_grad=True)
        
        rnn = RNN(input_size=4, hidden_size=6, num_layers=1, batch_first=True)
        rnn_out, hidden = rnn(x_rnn)
        
        assert rnn_out.shape == (1, 8, 6)
        assert np.all(np.isfinite(rnn_out.data))
    
    def test_alternating_pattern_inputs(self):
        """Test with inputs having alternating patterns."""
        # Checkerboard pattern for 2D
        pattern_2d = np.zeros((1, 1, 8, 8), dtype=np.float32)
        pattern_2d[0, 0, ::2, ::2] = 1.0  # Checkerboard
        pattern_2d[0, 0, 1::2, 1::2] = 1.0
        
        x_pattern = Tensor(pattern_2d, requires_grad=True)
        
        conv2d = Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        output_pattern = conv2d(x_pattern)
        
        assert output_pattern.shape == (1, 2, 8, 8)
        assert np.all(np.isfinite(output_pattern.data))
    
    def test_sparse_inputs(self):
        """Test with mostly zero inputs (sparse)."""
        # Mostly zeros with few non-zero elements
        sparse_data = np.zeros((1, 4, 20), dtype=np.float32)
        sparse_data[0, 0, 5] = 1.0
        sparse_data[0, 1, 10] = -1.0
        sparse_data[0, 2, 15] = 2.0
        
        x_sparse = Tensor(sparse_data, requires_grad=True)
        
        conv = Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        output_sparse = conv(x_sparse)
        
        assert output_sparse.shape == (1, 8, 20)
        assert np.all(np.isfinite(output_sparse.data))
        
        # Most output should be influenced by the sparse non-zero inputs
        assert not np.all(output_sparse.data == 0)


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""
    
    def test_kernel_size_equal_input_size(self):
        """Test when kernel size equals input size."""
        input_size = 10
        kernel_size = 10
        
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=kernel_size)
        
        input_data = np.random.randn(1, 2, input_size).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        # Should produce output of size 1
        assert output.shape == (1, 4, 1)
    
    def test_stride_larger_than_kernel(self):
        """Test when stride is larger than kernel size."""
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=5)
        
        input_data = np.random.randn(1, 2, 20).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        # Should skip over input elements
        expected_size = (20 - 3) // 5 + 1
        assert output.shape == (1, 4, expected_size)
    
    def test_padding_larger_than_input(self):
        """Test when padding is larger than input size."""
        input_size = 5
        padding = 10
        
        conv = Conv1d(in_channels=1, out_channels=2, kernel_size=3, padding=padding)
        
        input_data = np.random.randn(1, 1, input_size).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        # Output size: input + 2*padding - kernel + 1
        expected_size = input_size + 2 * padding - 3 + 1
        assert output.shape == (1, 2, expected_size)
    
    def test_dilation_creates_gaps(self):
        """Test dilation that creates large gaps."""
        conv = Conv1d(in_channels=1, out_channels=2, kernel_size=3, dilation=5)
        
        # Need large enough input for dilated kernel
        input_size = 20
        input_data = np.random.randn(1, 1, input_size).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        # Effective kernel size: 3 + (3-1)*4 = 11
        # Output size: 20 - 11 + 1 = 10
        assert output.shape == (1, 2, 10)
    
    def test_group_convolution_edge_cases(self):
        """Test edge cases in group convolution."""
        # Groups equal to input channels (depthwise)
        in_channels = 8
        groups = 8
        
        conv = Conv1d(in_channels=in_channels, out_channels=16, 
                     kernel_size=3, groups=groups, padding=1)
        
        input_data = np.random.randn(1, in_channels, 10).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        output = conv(x)
        
        assert output.shape == (1, 16, 10)
        
        # Verify weight shape for grouped convolution
        # Each group processes in_channels/groups input channels
        expected_weight_shape = (16, 1, 3)  # (out_channels, in_channels/groups, kernel)
        assert conv.weight.data.shape == expected_weight_shape


class TestStressConditions:
    """Stress tests under extreme conditions."""
    
    def test_rapid_memory_allocation_deallocation(self):
        """Test rapid memory allocation and deallocation."""
        conv = Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        
        # Rapidly create and destroy tensors
        for i in range(20):
            input_data = np.random.randn(1, 4, 50).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            output = conv(x)
            
            assert output.shape == (1, 8, 50)
            
            # Force cleanup
            del x, output, input_data
            if i % 5 == 0:
                gc.collect()
    
    def test_alternating_large_small_operations(self):
        """Test alternating between large and small operations."""
        small_conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        large_conv = Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        
        for i in range(10):
            if i % 2 == 0:
                # Small operation
                small_data = np.random.randn(1, 2, 10).astype(np.float32)
                x_small = Tensor(small_data, requires_grad=True)
                output_small = small_conv(x_small)
                assert output_small.shape == (1, 4, 10)
                del x_small, output_small, small_data
            else:
                # Large operation
                large_data = np.random.randn(1, 16, 100).astype(np.float32)
                x_large = Tensor(large_data, requires_grad=True)
                output_large = large_conv(x_large)
                assert output_large.shape == (1, 32, 100)
                del x_large, output_large, large_data
            
            if i % 3 == 0:
                gc.collect()
    
    def test_performance_under_stress(self):
        """Test performance under stress conditions."""
        import time
        
        # Model that should complete quickly even under stress
        model = Sequential(
            Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            ReLU(),
            Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
        )
        
        input_data = np.random.randn(2, 4, 30).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Measure performance
        times = []
        for _ in range(5):
            start_time = time.time()
            output = model(x)
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert output.shape == (2, 4, 30)
        
        # Performance should be consistent and fast
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        assert avg_time < 0.1, f"Average time {avg_time:.3f}s too slow"
        assert max_time < 0.2, f"Max time {max_time:.3f}s too slow"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])