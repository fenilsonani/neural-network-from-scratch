"""
Performance benchmarks and regression tests for neural architecture.
"""

try:
    import pytest
except ImportError:
    pytest = None

import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_arch import (
    Tensor, Linear, Embedding, Adam, 
    add, mul, matmul, relu, softmax, mean_pool,
    create_text_vocab, text_to_sequences
)


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
    
    def time_operation(self, operation, *args, **kwargs):
        """Time an operation and return elapsed time."""
        start_time = time.time()
        result = operation(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        return result, elapsed
    
    def run_benchmark(self, operation, *args, num_runs=10, **kwargs):
        """Run benchmark multiple times and return statistics."""
        times = []
        
        for _ in range(num_runs):
            _, elapsed = self.time_operation(operation, *args, **kwargs)
            times.append(elapsed)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'times': times
        }


class TestTensorOperationPerformance:
    """Test performance of basic tensor operations."""
    
    def test_tensor_creation_performance(self):
        """Benchmark tensor creation speed."""
        benchmark = PerformanceBenchmark("Tensor Creation")
        
        # Test different sizes
        sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        for size in sizes:
            data = np.random.randn(*size)
            
            # Benchmark tensor creation
            stats = benchmark.run_benchmark(
                lambda: Tensor(data, requires_grad=True),
                num_runs=100
            )
            
            print(f"Tensor creation {size}: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
            
            # Performance requirement: should be fast
            assert stats['mean_time'] < 0.01  # Less than 10ms
    
    def test_matrix_multiplication_performance(self):
        """Benchmark matrix multiplication performance."""
        benchmark = PerformanceBenchmark("Matrix Multiplication")
        
        # Test different matrix sizes
        test_cases = [
            ((100, 50), (50, 100)),
            ((500, 300), (300, 200)),
            ((1000, 100), (100, 500)),
        ]
        
        for shape_a, shape_b in test_cases:
            a = Tensor(np.random.randn(*shape_a), requires_grad=True)
            b = Tensor(np.random.randn(*shape_b), requires_grad=True)
            
            # Forward pass benchmark
            stats = benchmark.run_benchmark(
                lambda: matmul(a, b),
                num_runs=20
            )
            
            print(f"MatMul {shape_a}x{shape_b}: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
            
            # Performance requirement
            expected_ops = shape_a[0] * shape_a[1] * shape_b[1]
            max_time = expected_ops / 1e6  # 1M ops per second minimum
            assert stats['mean_time'] < max_time
    
    def test_gradient_computation_performance(self):
        """Benchmark gradient computation performance."""
        benchmark = PerformanceBenchmark("Gradient Computation")
        
        # Create computation graph
        x = Tensor(np.random.randn(100, 50), requires_grad=True)
        W = Tensor(np.random.randn(50, 30), requires_grad=True)
        
        def forward_backward():
            y = matmul(x, W)
            z = relu(y)
            output = mean_pool(z, axis=0)
            
            # Backward pass
            output.backward(np.ones_like(output.data))
            if hasattr(output, '_backward'):
                output._backward()
            
            # Clear gradients for next run
            x.zero_grad()
            W.zero_grad()
            
            return output
        
        stats = benchmark.run_benchmark(forward_backward, num_runs=50)
        
        print(f"Forward-Backward pass: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
        
        # Should complete within reasonable time
        assert stats['mean_time'] < 0.1  # Less than 100ms
    
    def test_softmax_performance_large_batches(self):
        """Benchmark softmax with large batches."""
        benchmark = PerformanceBenchmark("Softmax Large Batches")
        
        batch_sizes = [10, 100, 1000]
        vocab_size = 50000  # Large vocabulary
        
        for batch_size in batch_sizes:
            x = Tensor(np.random.randn(batch_size, vocab_size), requires_grad=True)
            
            stats = benchmark.run_benchmark(
                lambda: softmax(x),
                num_runs=10
            )
            
            print(f"Softmax batch={batch_size}: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
            
            # Should scale reasonably with batch size
            max_time = batch_size * vocab_size / 1e6  # 1M elements per second
            assert stats['mean_time'] < max_time


class TestLayerPerformance:
    """Test performance of neural network layers."""
    
    def test_linear_layer_performance(self):
        """Benchmark linear layer performance."""
        benchmark = PerformanceBenchmark("Linear Layer")
        
        # Different layer sizes
        layer_configs = [
            (100, 50),
            (500, 1000),
            (1000, 100),
            (2000, 2000),
        ]
        
        for in_features, out_features in layer_configs:
            layer = Linear(in_features, out_features)
            x = Tensor(np.random.randn(32, in_features), requires_grad=True)  # Batch size 32
            
            # Forward pass benchmark
            stats = benchmark.run_benchmark(
                lambda: layer(x),
                num_runs=50
            )
            
            print(f"Linear {in_features}->{out_features}: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
            
            # Performance requirement
            expected_ops = 32 * in_features * out_features  # Batch size * input * output
            max_time = expected_ops / 1e6  # 1M ops per second
            assert stats['mean_time'] < max_time
    
    def test_embedding_layer_performance(self):
        """Benchmark embedding layer performance."""
        benchmark = PerformanceBenchmark("Embedding Layer")
        
        # Different embedding configurations
        configs = [
            (1000, 128),    # Small vocab, small dim
            (10000, 256),   # Medium vocab, medium dim
            (50000, 512),   # Large vocab, large dim
        ]
        
        for vocab_size, embed_dim in configs:
            embedding = Embedding(vocab_size, embed_dim)
            batch_size, seq_len = 32, 100
            indices = np.random.randint(0, vocab_size, (batch_size, seq_len))
            
            stats = benchmark.run_benchmark(
                lambda: embedding(indices),
                num_runs=50
            )
            
            print(f"Embedding {vocab_size}x{embed_dim}: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
            
            # Should be very fast (just lookup)
            assert stats['mean_time'] < 0.01  # Less than 10ms
    
    def test_optimizer_performance(self):
        """Benchmark optimizer performance."""
        benchmark = PerformanceBenchmark("Adam Optimizer")
        
        # Create model with many parameters
        layers = [
            Linear(1000, 500),
            Linear(500, 200),
            Linear(200, 50),
            Linear(50, 10)
        ]
        
        # Collect all parameters
        all_params = {}
        for i, layer in enumerate(layers):
            for name, param in layer.parameters().items():
                all_params[f'layer_{i}_{name}'] = param
        
        optimizer = Adam(all_params, lr=0.001)
        
        # Set random gradients
        def set_random_gradients():
            for param in all_params.values():
                param.grad = np.random.randn(*param.shape) * 0.01
        
        def optimizer_step():
            set_random_gradients()
            optimizer.step()
            optimizer.zero_grad()
        
        stats = benchmark.run_benchmark(optimizer_step, num_runs=100)
        
        print(f"Optimizer step: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
        
        # Should be reasonably fast
        assert stats['mean_time'] < 0.05  # Less than 50ms


class TestTrainingPerformance:
    """Test performance of training loops."""
    
    def test_simple_training_loop_performance(self):
        """Benchmark a simple training loop."""
        benchmark = PerformanceBenchmark("Training Loop")
        
        # Simple model
        vocab_size = 1000
        embed_dim = 128
        hidden_dim = 256
        
        embedding = Embedding(vocab_size, embed_dim)
        linear1 = Linear(embed_dim, hidden_dim)
        linear2 = Linear(hidden_dim, vocab_size)
        
        # Collect parameters
        params = {}
        params.update(embedding.parameters())
        params.update({f'linear1_{k}': v for k, v in linear1.parameters().items()})
        params.update({f'linear2_{k}': v for k, v in linear2.parameters().items()})
        
        optimizer = Adam(params, lr=0.001)
        
        # Training data
        batch_size, seq_len = 16, 50
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = np.random.randint(0, vocab_size, (batch_size,))
        
        def training_step():
            # Forward pass
            embedded = embedding(input_ids)
            pooled = mean_pool(embedded, axis=1)
            hidden = relu(linear1(pooled))
            logits = linear2(hidden)
            probs = softmax(logits)
            
            # Simple loss
            batch_indices = np.arange(batch_size)
            loss_val = -np.mean(np.log(probs.data[batch_indices, target_ids] + 1e-8))
            
            # Backward pass
            grad = np.zeros_like(probs.data)
            grad[batch_indices, target_ids] = -1.0 / (probs.data[batch_indices, target_ids] + 1e-8)
            grad /= batch_size
            
            probs.backward(grad)
            if hasattr(probs, '_backward'):
                probs._backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            return loss_val
        
        stats = benchmark.run_benchmark(training_step, num_runs=20)
        
        print(f"Training step: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
        
        # Should complete training step in reasonable time
        assert stats['mean_time'] < 0.5  # Less than 500ms
    
    def test_batch_size_scaling(self):
        """Test how performance scales with batch size."""
        batch_sizes = [1, 4, 16, 64]
        vocab_size = 1000
        embed_dim = 64
        seq_len = 20
        
        embedding = Embedding(vocab_size, embed_dim)
        linear = Linear(embed_dim, vocab_size)
        
        performance_data = []
        
        for batch_size in batch_sizes:
            input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
            
            def forward_pass():
                embedded = embedding(input_ids)
                pooled = mean_pool(embedded, axis=1)
                output = linear(pooled)
                return softmax(output)
            
            benchmark = PerformanceBenchmark(f"Batch Size {batch_size}")
            stats = benchmark.run_benchmark(forward_pass, num_runs=50)
            
            time_per_sample = stats['mean_time'] / batch_size
            performance_data.append((batch_size, stats['mean_time'], time_per_sample))
            
            print(f"Batch size {batch_size}: {stats['mean_time']*1000:.2f}ms total, {time_per_sample*1000:.2f}ms per sample")
        
        # Check that larger batches are more efficient per sample
        for i in range(len(performance_data) - 1):
            current_per_sample = performance_data[i][2]
            next_per_sample = performance_data[i + 1][2]
            
            # Allow some variance, but generally should be more efficient
            assert next_per_sample <= current_per_sample * 1.5


class TestMemoryUsage:
    """Test memory usage patterns."""
    
    def test_gradient_memory_cleanup(self):
        """Test that gradients are properly cleaned up."""
        # Create many tensors
        tensors = []
        for i in range(100):
            x = Tensor(np.random.randn(100, 100), requires_grad=True)
            y = matmul(x, x)
            y.backward(np.ones_like(y.data))
            if hasattr(y, '_backward'):
                y._backward()
            tensors.append(x)
        
        # All should have gradients
        assert all(t.grad is not None for t in tensors)
        
        # Clear gradients
        for t in tensors:
            t.zero_grad()
        
        # All gradients should be None
        assert all(t.grad is None for t in tensors)
        
        print("✅ Gradient memory cleanup test passed")
    
    def test_large_tensor_handling(self):
        """Test handling of large tensors."""
        # Create large tensors
        large_size = (2000, 1000)
        
        a = Tensor(np.random.randn(*large_size), requires_grad=True)
        b = Tensor(np.random.randn(large_size[1], 500), requires_grad=True)
        
        # Should handle large operations
        c = matmul(a, b)
        assert c.shape == (large_size[0], 500)
        
        # Backward pass should work
        c.backward(np.ones_like(c.data))
        if hasattr(c, '_backward'):
            c._backward()
        
        assert a.grad is not None
        assert b.grad is not None
        
        print("✅ Large tensor handling test passed")


class TestRegressionBenchmarks:
    """Regression tests to catch performance regressions."""
    
    def __init__(self):
        # Performance baselines (in seconds)
        self.baselines = {
            'tensor_creation_1000x1000': 0.005,
            'matmul_1000x500_500x200': 0.05,
            'softmax_1000x10000': 0.1,
            'linear_layer_1000_1000': 0.01,
            'training_step_simple': 0.3,
        }
    
    def test_performance_regression(self):
        """Test that performance hasn't regressed."""
        results = {}
        
        # Tensor creation
        data = np.random.randn(1000, 1000)
        start = time.time()
        for _ in range(10):
            Tensor(data, requires_grad=True)
        results['tensor_creation_1000x1000'] = (time.time() - start) / 10
        
        # Matrix multiplication
        a = Tensor(np.random.randn(1000, 500), requires_grad=True)
        b = Tensor(np.random.randn(500, 200), requires_grad=True)
        start = time.time()
        for _ in range(10):
            matmul(a, b)
        results['matmul_1000x500_500x200'] = (time.time() - start) / 10
        
        # Softmax
        x = Tensor(np.random.randn(1000, 10000), requires_grad=True)
        start = time.time()
        for _ in range(5):
            softmax(x)
        results['softmax_1000x10000'] = (time.time() - start) / 5
        
        # Linear layer
        layer = Linear(1000, 1000)
        x = Tensor(np.random.randn(32, 1000), requires_grad=True)
        start = time.time()
        for _ in range(20):
            layer(x)
        results['linear_layer_1000_1000'] = (time.time() - start) / 20
        
        # Check for regressions
        regressions = []
        for test_name, current_time in results.items():
            baseline = self.baselines.get(test_name)
            if baseline and current_time > baseline * 2.0:  # Allow 2x slowdown
                regressions.append((test_name, current_time, baseline))
        
        # Report results
        print("\n📊 Performance Regression Test Results:")
        for test_name, current_time in results.items():
            baseline = self.baselines.get(test_name, 0)
            status = "✅" if current_time <= baseline * 2.0 else "❌"
            print(f"{status} {test_name}: {current_time*1000:.2f}ms (baseline: {baseline*1000:.2f}ms)")
        
        if regressions:
            print(f"\n❌ Performance regressions detected in {len(regressions)} tests:")
            for test_name, current_time, baseline in regressions:
                slowdown = current_time / baseline
                print(f"  - {test_name}: {slowdown:.1f}x slower than baseline")
        
        # Fail if there are significant regressions
        assert len(regressions) == 0, f"Performance regressions detected: {regressions}"


def run_comprehensive_benchmarks():
    """Run all performance benchmarks."""
    print("🚀 Running Comprehensive Performance Benchmarks")
    print("=" * 60)
    
    # Tensor operations
    print("\n📊 Tensor Operations Performance:")
    tensor_perf = TestTensorOperationPerformance()
    tensor_perf.test_tensor_creation_performance()
    tensor_perf.test_matrix_multiplication_performance()
    tensor_perf.test_gradient_computation_performance()
    tensor_perf.test_softmax_performance_large_batches()
    
    # Layer performance
    print("\n📊 Layer Performance:")
    layer_perf = TestLayerPerformance()
    layer_perf.test_linear_layer_performance()
    layer_perf.test_embedding_layer_performance()
    layer_perf.test_optimizer_performance()
    
    # Training performance
    print("\n📊 Training Performance:")
    training_perf = TestTrainingPerformance()
    training_perf.test_simple_training_loop_performance()
    training_perf.test_batch_size_scaling()
    
    # Memory usage
    print("\n📊 Memory Usage:")
    memory_test = TestMemoryUsage()
    memory_test.test_gradient_memory_cleanup()
    memory_test.test_large_tensor_handling()
    
    # Regression tests
    print("\n📊 Regression Tests:")
    regression_test = TestRegressionBenchmarks()
    regression_test.test_performance_regression()
    
    print("\n🎉 All Performance Benchmarks Completed!")


if __name__ == "__main__":
    # Run comprehensive benchmarks
    run_comprehensive_benchmarks()