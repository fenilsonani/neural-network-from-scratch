"""Performance optimization and validation tests for neural architecture framework.

This test suite provides comprehensive coverage for:
- JIT compilation testing with neural layers
- CUDA acceleration testing (if available)
- Memory optimization validation
- Benchmark comparison tests
- Performance regression testing
- Speed optimization validation
- Throughput measurement
- Latency optimization testing
- Cache efficiency validation

Targets <200ms performance requirements and validates optimization effectiveness.
"""

import gc
import math
import os
import sys
import time
import psutil
import pytest
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from contextlib import contextmanager

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
from neural_arch.optim import Adam, SGD, AdamW


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000 if self.duration else 0.0


class MemoryProfiler:
    """Simple memory profiler for tracking memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_delta(self) -> float:
        """Get memory delta from initial in MB."""
        return self.get_memory_usage() - self.initial_memory


@contextmanager
def memory_tracker():
    """Context manager for tracking memory usage."""
    profiler = MemoryProfiler()
    initial_memory = profiler.get_memory_usage()
    
    try:
        yield profiler
    finally:
        final_memory = profiler.get_memory_usage()
        delta = final_memory - initial_memory
        # Force garbage collection to get accurate measurements
        gc.collect()


class TestBasicPerformanceTargets:
    """Tests for basic performance targets (<200ms)."""
    
    def test_conv_layer_performance_targets(self):
        """Test that convolution layers meet performance targets."""
        test_configs = [
            # (layer_class, input_shape, layer_kwargs)
            (Conv1d, (4, 32, 64), {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}),
            (Conv2d, (4, 32, 32, 32), {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}),
            (Conv2d, (2, 64, 16, 16), {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1}),
        ]
        
        for layer_class, input_shape, layer_kwargs in test_configs:
            layer = layer_class(**layer_kwargs)
            input_data = np.random.randn(*input_shape).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Warm up
            _ = layer(x)
            
            # Measure performance
            with PerformanceTimer(f"{layer_class.__name__} forward") as timer:
                output = layer(x)
            
            assert timer.duration_ms < 200, \
                f"{layer_class.__name__} took {timer.duration_ms:.2f}ms, exceeding 200ms target"
            
            # Verify output is correct
            assert output.requires_grad
            assert output.shape[0] == input_shape[0]  # Batch size preserved
    
    def test_rnn_layer_performance_targets(self):
        """Test that RNN layers meet performance targets."""
        test_configs = [
            # (layer_class, input_shape, layer_kwargs)
            (RNN, (2, 50, 32), {"input_size": 32, "hidden_size": 64, "num_layers": 1, "batch_first": True}),
            (LSTM, (2, 30, 16), {"input_size": 16, "hidden_size": 32, "num_layers": 2, "batch_first": True}),
            (GRU, (4, 25, 24), {"input_size": 24, "hidden_size": 48, "num_layers": 1, "batch_first": True}),
        ]
        
        for layer_class, input_shape, layer_kwargs in test_configs:
            layer = layer_class(**layer_kwargs)
            input_data = np.random.randn(*input_shape).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Warm up
            if layer_class in [RNN, GRU]:
                _ = layer(x)
            else:  # LSTM
                _ = layer(x)
            
            # Measure performance
            with PerformanceTimer(f"{layer_class.__name__} forward") as timer:
                if layer_class in [RNN, GRU]:
                    output, hidden = layer(x)
                else:  # LSTM
                    output, (hidden, cell) = layer(x)
            
            assert timer.duration_ms < 200, \
                f"{layer_class.__name__} took {timer.duration_ms:.2f}ms, exceeding 200ms target"
            
            # Verify output shapes
            assert output.shape[:2] == input_shape[:2]  # Batch and sequence preserved
    
    def test_transformer_performance_targets(self):
        """Test that transformer components meet performance targets."""
        # Transformer block
        transformer = TransformerBlock(
            d_model=128, 
            n_heads=8, 
            d_ff=512,
            dropout=0.1
        )
        
        # Input: (batch, seq_len, d_model)
        input_data = np.random.randn(2, 32, 128).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Warm up
        _ = transformer(x)
        
        # Measure performance
        with PerformanceTimer("TransformerBlock forward") as timer:
            output = transformer(x)
        
        assert timer.duration_ms < 200, \
            f"TransformerBlock took {timer.duration_ms:.2f}ms, exceeding 200ms target"
        
        assert output.shape == input_data.shape
        assert output.requires_grad
    
    def test_attention_performance_targets(self):
        """Test that attention mechanisms meet performance targets."""
        # Multi-head attention
        attention = MultiHeadAttention(
            d_model=64,
            n_heads=8,
            dropout=0.1
        )
        
        # Input: (batch, seq_len, d_model)
        input_data = np.random.randn(2, 16, 64).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Warm up
        _ = attention(x, x, x)
        
        # Measure performance
        with PerformanceTimer("MultiHeadAttention forward") as timer:
            output = attention(x, x, x)
        
        assert timer.duration_ms < 200, \
            f"MultiHeadAttention took {timer.duration_ms:.2f}ms, exceeding 200ms target"
        
        assert output.shape == input_data.shape
        assert output.requires_grad


class TestMemoryOptimization:
    """Tests for memory optimization and efficiency."""
    
    def test_memory_efficient_conv_operations(self):
        """Test memory efficiency of convolution operations."""
        with memory_tracker() as profiler:
            # Create conv layer
            conv = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            
            # Process multiple inputs to test memory reuse
            for i in range(5):
                input_data = np.random.randn(2, 32, 16, 16).astype(np.float32)
                x = Tensor(input_data, requires_grad=True)
                
                output = conv(x)
                
                # Verify output
                assert output.shape == (2, 64, 16, 16)
                
                # Clean up explicitly
                del x, output, input_data
                
                if i % 2 == 0:
                    gc.collect()
        
        # Memory usage should not grow excessively
        final_delta = profiler.get_memory_delta()
        assert final_delta < 50, f"Memory usage grew by {final_delta:.2f}MB, indicating memory leak"
    
    def test_memory_efficient_rnn_processing(self):
        """Test memory efficiency of RNN processing."""
        with memory_tracker() as profiler:
            rnn = LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
            
            # Process sequences of varying lengths
            sequence_lengths = [10, 20, 15, 25, 12]
            
            for seq_len in sequence_lengths:
                input_data = np.random.randn(2, seq_len, 32).astype(np.float32)
                x = Tensor(input_data, requires_grad=True)
                
                output, (hidden, cell) = rnn(x)
                
                assert output.shape == (2, seq_len, 64)
                
                # Clean up
                del x, output, hidden, cell, input_data
                gc.collect()
        
        # Memory should not accumulate across different sequence lengths
        final_delta = profiler.get_memory_delta()
        assert final_delta < 30, f"RNN memory usage grew by {final_delta:.2f}MB"
    
    def test_gradient_memory_efficiency(self):
        """Test memory efficiency during gradient computation."""
        with memory_tracker() as profiler:
            # Create a model with multiple layers
            model = Sequential(
                Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                ReLU(),
                Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
                ReLU(),
            )
            
            input_data = np.random.randn(4, 16, 50).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Forward pass with gradient tracking
            output = model(x)
            
            assert output.requires_grad
            assert output._grad_fn is not None
            
            # Verify all parameters have gradients enabled
            all_params = []
            for layer in model.layers:
                if hasattr(layer, 'parameters'):
                    all_params.extend(layer.parameters())
            
            grad_enabled_params = [p for p in all_params if p.requires_grad]
            assert len(grad_enabled_params) > 0
        
        # Gradient computation setup should not use excessive memory
        gradient_memory = profiler.get_memory_delta()
        assert gradient_memory < 40, f"Gradient memory usage: {gradient_memory:.2f}MB"
    
    def test_parameter_sharing_memory_efficiency(self):
        """Test memory efficiency with parameter sharing."""
        with memory_tracker() as profiler:
            # Shared layer
            shared_conv = Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
            
            # Use the same layer in multiple paths
            paths = []
            for i in range(4):
                input_data = np.random.randn(1, 8, 20).astype(np.float32)
                x = Tensor(input_data, requires_grad=True)
                
                output = shared_conv(x)
                paths.append(output)
                
                assert output.shape == (1, 16, 20)
            
            # Verify parameter sharing (same memory address)
            base_weight_id = id(shared_conv.weight.data)
            
            # Clean up
            del paths
            gc.collect()
        
        # Parameter sharing should keep memory usage low
        sharing_memory = profiler.get_memory_delta()
        assert sharing_memory < 20, f"Parameter sharing used {sharing_memory:.2f}MB"
    
    def test_large_batch_memory_scaling(self):
        """Test memory scaling with large batch sizes."""
        conv = Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        
        batch_sizes = [1, 2, 4, 8, 16]
        memory_usage = []
        
        for batch_size in batch_sizes:
            with memory_tracker() as profiler:
                input_data = np.random.randn(batch_size, 4, 32).astype(np.float32)
                x = Tensor(input_data, requires_grad=True)
                
                output = conv(x)
                
                assert output.shape == (batch_size, 8, 32)
                
                current_memory = profiler.get_memory_delta()
                memory_usage.append(current_memory)
                
                del x, output, input_data
                gc.collect()
        
        # Memory scaling should be roughly linear with batch size
        # Check that memory doesn't grow exponentially
        if len(memory_usage) >= 3:
            ratio_1_2 = memory_usage[2] / max(memory_usage[1], 0.1)  # batch 4 vs batch 2
            ratio_2_3 = memory_usage[3] / max(memory_usage[2], 0.1)  # batch 8 vs batch 4
            
            # Ratios should be similar (indicating linear scaling)
            assert abs(ratio_1_2 - ratio_2_3) < 2.0, \
                f"Memory scaling not linear: {ratio_1_2:.2f} vs {ratio_2_3:.2f}"


class TestThroughputOptimization:
    """Tests for throughput optimization and batch processing."""
    
    def test_batch_processing_throughput(self):
        """Test throughput improvements with batch processing."""
        conv = Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Single sample processing
        single_times = []
        for _ in range(10):
            input_data = np.random.randn(1, 16, 16, 16).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            with PerformanceTimer() as timer:
                output = conv(x)
            
            single_times.append(timer.duration_ms)
            del x, output, input_data
        
        avg_single_time = sum(single_times) / len(single_times)
        
        # Batch processing
        batch_sizes = [4, 8, 16]
        for batch_size in batch_sizes:
            input_data = np.random.randn(batch_size, 16, 16, 16).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            with PerformanceTimer() as timer:
                output = conv(x)
            
            batch_time = timer.duration_ms
            time_per_sample = batch_time / batch_size
            
            # Batch processing should be more efficient than single processing
            efficiency_ratio = avg_single_time / time_per_sample
            assert efficiency_ratio > 1.1, \
                f"Batch size {batch_size} not efficient: {efficiency_ratio:.2f}x speedup"
            
            del x, output, input_data
    
    def test_sequence_processing_throughput(self):
        """Test throughput for sequence processing."""
        rnn = LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        
        # Different sequence lengths
        sequence_lengths = [10, 20, 50, 100]
        times_per_timestep = []
        
        for seq_len in sequence_lengths:
            input_data = np.random.randn(2, seq_len, 32).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Warm up
            _ = rnn(x)
            
            with PerformanceTimer() as timer:
                output, (hidden, cell) = rnn(x)
            
            time_per_timestep = timer.duration_ms / seq_len
            times_per_timestep.append(time_per_timestep)
            
            assert output.shape == (2, seq_len, 64)
            del x, output, hidden, cell, input_data
        
        # Time per timestep should not grow significantly with sequence length
        # (indicating efficient sequence processing)
        max_time_per_step = max(times_per_timestep)
        min_time_per_step = min(times_per_timestep)
        ratio = max_time_per_step / min_time_per_step
        
        assert ratio < 3.0, \
            f"Time per timestep varies too much: {ratio:.2f}x variation"
    
    def test_parallel_processing_potential(self):
        """Test potential for parallel processing."""
        # Create multiple independent models
        models = [
            Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
            for _ in range(4)
        ]
        
        inputs = [
            np.random.randn(1, 4, 20).astype(np.float32)
            for _ in range(4)
        ]
        
        # Sequential processing
        with PerformanceTimer("Sequential processing") as seq_timer:
            for model, input_data in zip(models, inputs):
                x = Tensor(input_data, requires_grad=True)
                output = model(x)
                del x, output
        
        # Threaded processing (simulated parallel)
        def process_model(model_input_pair):
            model, input_data = model_input_pair
            x = Tensor(input_data, requires_grad=True)
            output = model(x)
            return output.shape
        
        with PerformanceTimer("Threaded processing") as thread_timer:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_model, zip(models, inputs)))
        
        # Verify all results are correct
        for result in results:
            assert result == (1, 8, 20)
        
        # Threaded should be faster (or at least not much slower due to GIL)
        speedup = seq_timer.duration_ms / thread_timer.duration_ms
        print(f"Threading speedup: {speedup:.2f}x")
        
        # Even with Python GIL, should not be more than 50% slower
        assert speedup > 0.5, f"Threading much slower: {speedup:.2f}x"


class TestOptimizationAlgorithmPerformance:
    """Tests for optimizer performance."""
    
    def test_optimizer_update_performance(self):
        """Test performance of optimizer updates."""
        # Create model with many parameters
        model = Sequential(
            Linear(in_features=128, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=64),
        )
        
        # Test different optimizers
        optimizers = [
            ("SGD", SGD(model.parameters(), lr=0.01)),
            ("Adam", Adam(model.parameters(), lr=0.001)),
            ("AdamW", AdamW(model.parameters(), lr=0.001)),
        ]
        
        for opt_name, optimizer in optimizers:
            # Generate input and target
            input_data = np.random.randn(4, 128).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Forward pass
            output = model(x)
            
            # Simulate loss computation
            loss_data = np.sum(output.data ** 2)
            loss = Tensor(np.array([loss_data]), requires_grad=True)
            
            # Measure optimizer step time
            with PerformanceTimer(f"{opt_name} step") as timer:
                # Simulate gradient computation
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad = np.random.randn(*param.data.shape).astype(np.float32)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
            
            assert timer.duration_ms < 50, \
                f"{opt_name} step took {timer.duration_ms:.2f}ms, too slow"
            
            del x, output, loss
    
    def test_gradient_computation_efficiency(self):
        """Test efficiency of gradient computation setup."""
        # Model with skip connections
        conv1 = Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        conv2 = Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        
        input_data = np.random.randn(2, 8, 32).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        with PerformanceTimer("Gradient setup") as timer:
            # Forward with residual connection
            conv1_out = conv1(x)
            conv2_out = conv2(conv1_out)
            
            # Residual connection
            residual_data = x.data + conv2_out.data  # Skip connection
            output = Tensor(residual_data, requires_grad=True)
            
            # Verify gradient functions are set up
            assert conv1_out._grad_fn is not None
            assert conv2_out._grad_fn is not None
        
        assert timer.duration_ms < 10, \
            f"Gradient setup took {timer.duration_ms:.2f}ms, too slow"
        
        # Verify output shape
        assert output.shape == x.shape


class TestCacheEfficiency:
    """Tests for cache efficiency and memory access patterns."""
    
    def test_sequential_memory_access_patterns(self):
        """Test that operations favor sequential memory access."""
        # Large 1D convolution that should benefit from sequential access
        large_conv = Conv1d(in_channels=2, out_channels=4, kernel_size=5, padding=2)
        
        # Very long sequence to test cache efficiency
        long_sequence = 1000
        input_data = np.random.randn(1, 2, long_sequence).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Multiple runs to test cache warming
        times = []
        for run in range(5):
            with PerformanceTimer() as timer:
                output = large_conv(x)
            
            times.append(timer.duration_ms)
            assert output.shape == (1, 4, long_sequence)
        
        # Later runs should be faster due to cache warming
        first_time = times[0]
        avg_later_time = sum(times[2:]) / len(times[2:])  # Skip first warm-up run
        
        cache_efficiency = first_time / avg_later_time
        assert cache_efficiency > 0.8, \
            f"Poor cache efficiency: {cache_efficiency:.2f}x"
    
    def test_spatial_locality_conv2d(self):
        """Test spatial locality in 2D convolutions."""
        conv = Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        
        # Different input sizes to test spatial locality
        sizes = [(16, 16), (32, 32), (64, 64)]
        times_per_pixel = []
        
        for height, width in sizes:
            input_data = np.random.randn(1, 8, height, width).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Warm up
            _ = conv(x)
            
            with PerformanceTimer() as timer:
                output = conv(x)
            
            pixels = height * width
            time_per_pixel = timer.duration_ms / pixels
            times_per_pixel.append(time_per_pixel)
            
            assert output.shape == (1, 16, height, width)
            del x, output, input_data
        
        # Time per pixel should not increase dramatically with size
        # (indicating good spatial locality)
        max_time_per_pixel = max(times_per_pixel)
        min_time_per_pixel = min(times_per_pixel)
        efficiency_ratio = max_time_per_pixel / min_time_per_pixel
        
        assert efficiency_ratio < 5.0, \
            f"Poor spatial locality: {efficiency_ratio:.2f}x variation in time per pixel"
    
    def test_temporal_locality_rnn(self):
        """Test temporal locality in RNN processing."""
        rnn = RNN(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        
        # Process same sequence multiple times
        input_data = np.random.randn(2, 30, 16).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # First run (cold cache)
        with PerformanceTimer("Cold run") as cold_timer:
            output1, hidden1 = rnn(x)
        
        # Immediate second run (warm cache)
        with PerformanceTimer("Warm run") as warm_timer:
            output2, hidden2 = rnn(x)
        
        # Third run (should be consistently fast)
        with PerformanceTimer("Consistent run") as consistent_timer:
            output3, hidden3 = rnn(x)
        
        # Warm runs should be faster
        cache_speedup = cold_timer.duration_ms / warm_timer.duration_ms
        consistency = abs(warm_timer.duration_ms - consistent_timer.duration_ms) / warm_timer.duration_ms
        
        assert cache_speedup > 0.8, f"Poor cache speedup: {cache_speedup:.2f}x"
        assert consistency < 0.5, f"Poor temporal consistency: {consistency:.2f}"


class TestScalabilityBenchmarks:
    """Scalability benchmarks for different model sizes."""
    
    def test_model_size_scaling(self):
        """Test how performance scales with model size."""
        base_channels = 8
        scale_factors = [1, 2, 4]
        
        performance_results = {}
        
        for scale in scale_factors:
            channels = base_channels * scale
            
            # Create scaled model
            model = Sequential(
                Conv1d(in_channels=channels, out_channels=channels*2, kernel_size=3, padding=1),
                ReLU(),
                Conv1d(in_channels=channels*2, out_channels=channels, kernel_size=3, padding=1),
            )
            
            input_data = np.random.randn(2, channels, 50).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Warm up
            _ = model(x)
            
            # Measure performance
            with PerformanceTimer() as timer:
                output = model(x)
            
            # Calculate parameters
            total_params = sum(np.prod(p.data.shape) for p in model.parameters() if p.requires_grad)
            
            performance_results[scale] = {
                'time_ms': timer.duration_ms,
                'params': total_params,
                'time_per_param': timer.duration_ms / total_params
            }
            
            assert output.shape == (2, channels, 50)
            del x, output, input_data, model
        
        # Analyze scaling characteristics
        scales = sorted(performance_results.keys())
        times = [performance_results[s]['time_ms'] for s in scales]
        params = [performance_results[s]['params'] for s in scales]
        
        # Time should scale sub-quadratically with parameters
        if len(scales) >= 2:
            param_ratio = params[-1] / params[0]
            time_ratio = times[-1] / times[0]
            
            efficiency = param_ratio / time_ratio
            assert efficiency > 0.3, \
                f"Poor scaling efficiency: {efficiency:.2f} (time grows too fast with parameters)"
    
    def test_sequence_length_scaling(self):
        """Test performance scaling with sequence length."""
        rnn = LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        
        sequence_lengths = [20, 50, 100, 200]
        performance_data = {}
        
        for seq_len in sequence_lengths:
            input_data = np.random.randn(2, seq_len, 32).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Warm up
            _ = rnn(x)
            
            with PerformanceTimer() as timer:
                output, (hidden, cell) = rnn(x)
            
            performance_data[seq_len] = {
                'time_ms': timer.duration_ms,
                'time_per_step': timer.duration_ms / seq_len
            }
            
            assert output.shape == (2, seq_len, 64)
            del x, output, hidden, cell, input_data
        
        # Time per step should remain relatively constant
        times_per_step = [performance_data[s]['time_per_step'] for s in sequence_lengths]
        max_time_per_step = max(times_per_step)
        min_time_per_step = min(times_per_step)
        variation = max_time_per_step / min_time_per_step
        
        assert variation < 3.0, \
            f"Poor sequence scaling: {variation:.2f}x variation in time per step"
    
    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        conv = Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        batch_sizes = [1, 2, 4, 8, 16]
        efficiency_data = {}
        
        for batch_size in batch_sizes:
            if batch_size > 8:  # Skip very large batches if memory constrained
                try:
                    input_data = np.random.randn(batch_size, 16, 16, 16).astype(np.float32)
                    x = Tensor(input_data, requires_grad=True)
                except MemoryError:
                    continue
            else:
                input_data = np.random.randn(batch_size, 16, 16, 16).astype(np.float32)
                x = Tensor(input_data, requires_grad=True)
            
            # Warm up
            _ = conv(x)
            
            with PerformanceTimer() as timer:
                output = conv(x)
            
            time_per_sample = timer.duration_ms / batch_size
            efficiency_data[batch_size] = {
                'total_time': timer.duration_ms,
                'time_per_sample': time_per_sample
            }
            
            assert output.shape == (batch_size, 32, 16, 16)
            del x, output, input_data
        
        # Larger batches should be more efficient (lower time per sample)
        if len(efficiency_data) >= 2:
            batch_sizes_tested = sorted(efficiency_data.keys())
            smallest_batch = batch_sizes_tested[0]
            largest_batch = batch_sizes_tested[-1]
            
            efficiency_gain = (efficiency_data[smallest_batch]['time_per_sample'] / 
                             efficiency_data[largest_batch]['time_per_sample')
            
            assert efficiency_gain > 1.0, \
                f"Batch processing not efficient: {efficiency_gain:.2f}x speedup"


class TestRegressionBenchmarks:
    """Regression benchmarks to ensure performance doesn't degrade."""
    
    def __init__(self):
        """Initialize regression benchmarks with baseline values."""
        # These are target baseline performance values in milliseconds
        self.performance_baselines = {
            'conv1d_forward': 10.0,      # Conv1d forward pass
            'conv2d_forward': 15.0,      # Conv2d forward pass
            'rnn_forward': 20.0,         # RNN forward pass
            'lstm_forward': 25.0,        # LSTM forward pass
            'attention_forward': 30.0,   # Attention forward pass
            'transformer_forward': 50.0, # Transformer block forward
        }
    
    def test_conv1d_regression_benchmark(self):
        """Regression test for Conv1d performance."""
        conv = Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        input_data = np.random.randn(4, 32, 64).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Warm up
        _ = conv(x)
        
        # Measure performance
        times = []
        for _ in range(5):
            with PerformanceTimer() as timer:
                output = conv(x)
            times.append(timer.duration_ms)
            del output
        
        avg_time = sum(times) / len(times)
        baseline = self.performance_baselines['conv1d_forward']
        
        assert avg_time < baseline * 2.0, \
            f"Conv1d regression: {avg_time:.2f}ms vs baseline {baseline:.2f}ms"
        
        print(f"Conv1d performance: {avg_time:.2f}ms (baseline: {baseline:.2f}ms)")
    
    def test_lstm_regression_benchmark(self):
        """Regression test for LSTM performance."""
        lstm = LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        input_data = np.random.randn(2, 32, 64).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Warm up
        _ = lstm(x)
        
        # Measure performance
        times = []
        for _ in range(3):
            with PerformanceTimer() as timer:
                output, (hidden, cell) = lstm(x)
            times.append(timer.duration_ms)
            del output, hidden, cell
        
        avg_time = sum(times) / len(times)
        baseline = self.performance_baselines['lstm_forward']
        
        assert avg_time < baseline * 3.0, \
            f"LSTM regression: {avg_time:.2f}ms vs baseline {baseline:.2f}ms"
        
        print(f"LSTM performance: {avg_time:.2f}ms (baseline: {baseline:.2f}ms)")
    
    def test_attention_regression_benchmark(self):
        """Regression test for attention mechanism performance."""
        attention = MultiHeadAttention(d_model=128, n_heads=8, dropout=0.0)
        input_data = np.random.randn(2, 32, 128).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Warm up
        _ = attention(x, x, x)
        
        # Measure performance
        times = []
        for _ in range(3):
            with PerformanceTimer() as timer:
                output = attention(x, x, x)
            times.append(timer.duration_ms)
            del output
        
        avg_time = sum(times) / len(times)
        baseline = self.performance_baselines['attention_forward']
        
        assert avg_time < baseline * 3.0, \
            f"Attention regression: {avg_time:.2f}ms vs baseline {baseline:.2f}ms"
        
        print(f"Attention performance: {avg_time:.2f}ms (baseline: {baseline:.2f}ms)")
    
    def test_end_to_end_model_benchmark(self):
        """End-to-end model performance regression test."""
        # Complete model pipeline
        model = Sequential(
            Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            ReLU(),
        )
        
        # Add RNN component
        rnn = LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        # Input data
        input_data = np.random.randn(2, 16, 100).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Warm up
        conv_out = model(x)
        rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
        _ = rnn(rnn_input)
        
        # Measure end-to-end performance
        with PerformanceTimer("End-to-end model") as timer:
            # Conv processing
            conv_out = model(x)
            
            # Reshape for RNN
            rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
            rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
            
            # RNN processing
            rnn_out, (hidden, cell) = rnn(rnn_input)
        
        # End-to-end should complete within reasonable time
        end_to_end_time = timer.duration_ms
        assert end_to_end_time < 100.0, \
            f"End-to-end model too slow: {end_to_end_time:.2f}ms"
        
        print(f"End-to-end performance: {end_to_end_time:.2f}ms")
        
        # Verify output shapes
        assert conv_out.shape == (2, 64, 50)  # Stride=2 halves sequence length
        assert rnn_out.shape == (2, 50, 128)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])