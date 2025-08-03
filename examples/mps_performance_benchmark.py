"""MPS Backend Performance Benchmark for Apple Silicon.

This script benchmarks the MPS (Metal Performance Shaders) backend 
against the NumPy CPU backend to demonstrate GPU acceleration on macOS.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch.backends import get_backend, available_backends
from neural_arch.core import Tensor
from neural_arch.nn import Linear, ReLU, Sequential
from neural_arch.optim import Adam


def benchmark_operation(backend, operation_name: str, operation_func, *args, num_runs: int = 10) -> float:
    """Benchmark a specific operation."""
    # Warmup
    for _ in range(3):
        try:
            result = operation_func(*args)
            if hasattr(result, 'data'):
                # Force evaluation for lazy backends
                _ = backend.to_numpy(result) if hasattr(backend, 'to_numpy') else result
        except Exception as e:
            print(f"Warmup failed for {operation_name}: {e}")
            return float('inf')
    
    # Actual benchmark
    start_time = time.time()
    for _ in range(num_runs):
        result = operation_func(*args)
        if hasattr(result, 'data'):
            # Force evaluation
            _ = backend.to_numpy(result) if hasattr(backend, 'to_numpy') else result
    end_time = time.time()
    
    return (end_time - start_time) / num_runs


def benchmark_matrix_operations():
    """Benchmark basic matrix operations."""
    print("🔥 Matrix Operations Benchmark")
    print("=" * 50)
    
    sizes = [100, 500, 1000, 2000]
    backends_to_test = []
    
    # Get available backends
    for backend_name in available_backends():
        try:
            backend = get_backend(backend_name)
            if backend.is_available:
                backends_to_test.append((backend_name, backend))
        except Exception as e:
            print(f"Failed to get backend {backend_name}: {e}")
    
    results = {}
    
    for size in sizes:
        print(f"\n📊 Matrix size: {size}x{size}")
        print("-" * 30)
        
        for backend_name, backend in backends_to_test:
            try:
                # Create test matrices
                a = backend.random_normal((size, size))
                b = backend.random_normal((size, size))
                
                # Test matrix multiplication
                matmul_time = benchmark_operation(
                    backend, "matmul", backend.matmul, a, b, num_runs=5
                )
                
                # Test element-wise operations
                add_time = benchmark_operation(
                    backend, "add", backend.add, a, b, num_runs=10
                )
                
                # Test reduction operations
                sum_time = benchmark_operation(
                    backend, "sum", backend.sum, a, num_runs=10
                )
                
                results[f"{backend_name}_{size}"] = {
                    'matmul': matmul_time,
                    'add': add_time,
                    'sum': sum_time
                }
                
                print(f"{backend_name:>8}: MatMul={matmul_time*1000:6.1f}ms, "
                      f"Add={add_time*1000:5.1f}ms, Sum={sum_time*1000:5.1f}ms")
                
            except Exception as e:
                print(f"{backend_name:>8}: Failed - {e}")
    
    return results


def benchmark_neural_network():
    """Benchmark neural network training."""
    print("\n🧠 Neural Network Training Benchmark")
    print("=" * 50)
    
    batch_sizes = [32, 128, 512]
    layer_sizes = [784, 256, 128, 10]  # MNIST-like network
    
    backends_to_test = []
    for backend_name in available_backends():
        try:
            backend = get_backend(backend_name)
            if backend.is_available:
                backends_to_test.append((backend_name, backend))
        except Exception:
            continue
    
    for batch_size in batch_sizes:
        print(f"\n📈 Batch size: {batch_size}")
        print("-" * 30)
        
        for backend_name, backend in backends_to_test:
            try:
                # Create model
                model = Sequential(
                    Linear(layer_sizes[0], layer_sizes[1]),
                    ReLU(),
                    Linear(layer_sizes[1], layer_sizes[2]),
                    ReLU(),
                    Linear(layer_sizes[2], layer_sizes[3])
                )
                
                optimizer = Adam(model.parameters(), lr=0.001)
                
                # Create test data
                input_data = np.random.randn(batch_size, layer_sizes[0]).astype(np.float32)
                target_data = np.random.randint(0, layer_sizes[3], (batch_size,))
                
                def training_step():
                    x = Tensor(input_data, requires_grad=True)
                    y_true = Tensor(target_data, requires_grad=False)
                    
                    # Forward pass
                    y_pred = model(x)
                    
                    # Simple MSE loss for benchmarking
                    target_one_hot = np.eye(layer_sizes[3])[target_data]
                    loss = ((y_pred.data - target_one_hot) ** 2).mean()
                    
                    # Simulate backward pass
                    for param in model.parameters():
                        if param.requires_grad:
                            param.grad = np.random.randn(*param.data.shape).astype(np.float32) * 0.01
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    return loss
                
                # Benchmark training step
                train_time = benchmark_operation(
                    backend, "training_step", training_step, num_runs=5
                )
                
                print(f"{backend_name:>8}: {train_time*1000:6.1f}ms per training step")
                
            except Exception as e:
                print(f"{backend_name:>8}: Failed - {e}")


def benchmark_activation_functions():
    """Benchmark activation functions."""
    print("\n⚡ Activation Functions Benchmark")
    print("=" * 50)
    
    sizes = [(1000, 1000), (2000, 2000)]
    
    backends_to_test = []
    for backend_name in available_backends():
        try:
            backend = get_backend(backend_name)
            if backend.is_available:
                backends_to_test.append((backend_name, backend))
        except Exception:
            continue
    
    for size in sizes:
        print(f"\n🎯 Tensor size: {size[0]}x{size[1]}")
        print("-" * 30)
        
        for backend_name, backend in backends_to_test:
            try:
                # Create test data
                x = backend.random_normal(size)
                
                # Test different activations
                exp_time = benchmark_operation(
                    backend, "exp", backend.exp, x, num_runs=10
                )
                
                sqrt_time = benchmark_operation(
                    backend, "sqrt", backend.abs, x, num_runs=10  # Use abs to ensure positive values
                )
                sqrt_time += benchmark_operation(
                    backend, "sqrt", backend.sqrt, backend.abs(x), num_runs=10
                )
                sqrt_time /= 2
                
                abs_time = benchmark_operation(
                    backend, "abs", backend.abs, x, num_runs=10
                )
                
                print(f"{backend_name:>8}: Exp={exp_time*1000:5.1f}ms, "
                      f"Sqrt={sqrt_time*1000:5.1f}ms, Abs={abs_time*1000:5.1f}ms")
                
            except Exception as e:
                print(f"{backend_name:>8}: Failed - {e}")


def calculate_speedup(results: Dict) -> None:
    """Calculate and display speedup ratios."""
    print("\n🚀 Performance Summary")
    print("=" * 50)
    
    # Find numpy baseline results
    numpy_results = {}
    mps_results = {}
    
    for key, values in results.items():
        if key.startswith('numpy_'):
            size = key.split('_')[1]
            numpy_results[size] = values
        elif key.startswith('mps_'):
            size = key.split('_')[1]
            mps_results[size] = values
    
    if not numpy_results or not mps_results:
        print("Incomplete results for speedup calculation")
        return
    
    print("\n📊 MPS vs NumPy Speedup Ratios:")
    print("-" * 40)
    
    for size in numpy_results.keys():
        if size in mps_results:
            print(f"\nMatrix size {size}x{size}:")
            
            for operation in ['matmul', 'add', 'sum']:
                if operation in numpy_results[size] and operation in mps_results[size]:
                    numpy_time = numpy_results[size][operation]
                    mps_time = mps_results[size][operation]
                    
                    if mps_time > 0 and numpy_time > 0:
                        speedup = numpy_time / mps_time
                        print(f"  {operation:>6}: {speedup:5.2f}x faster")
                    else:
                        print(f"  {operation:>6}: Invalid timing")


def system_info():
    """Display system information."""
    print("🖥️  System Information")
    print("=" * 50)
    
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    
    try:
        import mlx.core as mx
        print(f"MLX version: {mx.__version__}")
    except ImportError:
        print("MLX: Not available")
    
    print(f"NumPy version: {np.__version__}")
    print(f"Available backends: {available_backends()}")
    
    # Check if running on Apple Silicon
    if platform.processor() == 'arm':
        print("✅ Running on Apple Silicon - MPS acceleration available")
    else:
        print("⚠️  Not running on Apple Silicon - MPS may not be optimal")


def main():
    """Run comprehensive MPS performance benchmarks."""
    print("🔥 Neural Forge MPS Backend Performance Benchmark")
    print("=" * 60)
    
    system_info()
    
    # Run benchmarks
    matrix_results = benchmark_matrix_operations()
    benchmark_neural_network()
    benchmark_activation_functions()
    
    # Calculate speedups
    calculate_speedup(matrix_results)
    
    print("\n✅ Benchmark completed!")
    print("\n💡 Tips for optimal MPS performance:")
    print("   - Use larger batch sizes and matrices")
    print("   - Prefer float32 over float64")
    print("   - MLX automatically optimizes for Apple Silicon")
    print("   - GPU acceleration improves with computational complexity")


if __name__ == "__main__":
    main()