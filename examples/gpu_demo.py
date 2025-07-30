"""Demo script showing GPU acceleration with the neural architecture."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
from neural_arch.backends import (
    print_available_devices, 
    auto_select_backend,
    available_backends,
    set_backend,
    get_backend
)
from neural_arch.core import Tensor
from neural_arch.nn import Linear
from neural_arch.functional import relu, softmax


def benchmark_matrix_multiplication(backend_name: str, size: int = 1000):
    """Benchmark matrix multiplication on different backends."""
    print(f"\nBenchmarking {backend_name} backend (matrix size: {size}x{size})...")
    
    # Set backend
    set_backend(backend_name)
    backend = get_backend()
    
    # Create random matrices
    np.random.seed(42)
    a_data = np.random.randn(size, size).astype(np.float32)
    b_data = np.random.randn(size, size).astype(np.float32)
    
    # Convert to backend arrays
    a = backend.from_numpy(a_data)
    b = backend.from_numpy(b_data)
    
    # Warmup
    for _ in range(3):
        _ = backend.matmul(a, b)
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        result = backend.matmul(a, b)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    gflops = (2 * size**3) / (avg_time * 1e9)
    
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    
    return avg_time


def benchmark_neural_network(backend_name: str, batch_size: int = 64):
    """Benchmark a simple neural network forward pass."""
    print(f"\nBenchmarking neural network on {backend_name} backend...")
    
    # Set backend
    set_backend(backend_name)
    backend = get_backend()
    
    # Create a simple model (currently tensors don't use backends yet)
    # This is just to show the structure
    input_size = 784  # MNIST-like
    hidden_size = 256
    output_size = 10
    
    # Create random data
    np.random.seed(42)
    x_data = np.random.randn(batch_size, input_size).astype(np.float32)
    
    # Convert to backend array
    x = backend.from_numpy(x_data)
    
    # Create weight matrices
    w1_data = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.01
    b1_data = np.zeros(hidden_size, dtype=np.float32)
    w2_data = np.random.randn(hidden_size, output_size).astype(np.float32) * 0.01
    b2_data = np.zeros(output_size, dtype=np.float32)
    
    # Convert to backend
    w1 = backend.from_numpy(w1_data)
    b1 = backend.from_numpy(b1_data)
    w2 = backend.from_numpy(w2_data)
    b2 = backend.from_numpy(b2_data)
    
    # Define forward pass using backend operations
    def forward(x):
        # First layer
        h = backend.matmul(x, w1)
        h = backend.add(h, b1)
        h = backend.clip(h, 0, float('inf'))  # ReLU
        
        # Second layer
        out = backend.matmul(h, w2)
        out = backend.add(out, b2)
        
        # Softmax (simplified)
        exp_out = backend.exp(out - backend.max(out, axis=-1, keepdims=True))
        out = backend.divide(exp_out, backend.sum(exp_out, axis=-1, keepdims=True))
        
        return out
    
    # Warmup
    for _ in range(3):
        _ = forward(x)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        output = forward(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    
    print(f"  Batch size: {batch_size}")
    print(f"  Average forward pass time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {batch_size/avg_time:.0f} samples/sec")
    
    return avg_time


def main():
    """Main demo function."""
    print("=" * 60)
    print("Neural Architecture GPU Acceleration Demo")
    print("=" * 60)
    
    # Show available devices
    print_available_devices()
    
    # Get available backends
    backends = available_backends()
    print(f"\nTesting backends: {backends}")
    
    # Matrix multiplication benchmark
    print("\n" + "="*60)
    print("Matrix Multiplication Benchmark")
    print("="*60)
    
    results = {}
    for backend_name in backends:
        try:
            time_taken = benchmark_matrix_multiplication(backend_name, size=1000)
            results[backend_name] = time_taken
        except Exception as e:
            print(f"  Error: {e}")
    
    # Show speedup
    if "numpy" in results and len(results) > 1:
        print("\nSpeedup vs CPU:")
        cpu_time = results["numpy"]
        for backend_name, time_taken in results.items():
            if backend_name != "numpy":
                speedup = cpu_time / time_taken
                print(f"  {backend_name}: {speedup:.2f}x faster")
    
    # Neural network benchmark
    print("\n" + "="*60)
    print("Neural Network Forward Pass Benchmark")
    print("="*60)
    
    nn_results = {}
    for backend_name in backends:
        try:
            time_taken = benchmark_neural_network(backend_name, batch_size=128)
            nn_results[backend_name] = time_taken
        except Exception as e:
            print(f"  Error: {e}")
    
    # Show speedup
    if "numpy" in nn_results and len(nn_results) > 1:
        print("\nSpeedup vs CPU:")
        cpu_time = nn_results["numpy"]
        for backend_name, time_taken in nn_results.items():
            if backend_name != "numpy":
                speedup = cpu_time / time_taken
                print(f"  {backend_name}: {speedup:.2f}x faster")
    
    # Auto-selection demo
    print("\n" + "="*60)
    print("Automatic Backend Selection")
    print("="*60)
    
    backend = auto_select_backend(prefer_gpu=True)
    print(f"Auto-selected backend: {backend.name}")
    
    # Example of how it would work with Tensors (future)
    print("\n" + "="*60)
    print("Future Integration with Tensors")
    print("="*60)
    print("""
# Once Tensor class is updated to use backends:

# Automatically use GPU if available
x = Tensor(data, device="cuda")  # or device="mps" on Mac

# Or let the system choose
x = Tensor(data)  # Uses default device based on backend

# Move between devices
x_gpu = x.to("cuda")
x_cpu = x_gpu.to("cpu")

# All operations automatically use the right backend
y = x @ w + b  # Matrix multiply on GPU
z = relu(y)    # Activation on GPU
""")


if __name__ == "__main__":
    main()