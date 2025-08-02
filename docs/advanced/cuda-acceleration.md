# CUDA Acceleration Guide

This guide covers CUDA acceleration support in the neural architecture framework using custom CUDA kernels for improved GPU performance.

## Overview

The neural architecture framework includes a CUDA backend with custom kernel implementations for operations like GELU activation, fused linear operations, layer normalization, and Flash Attention. These kernels are designed to provide better performance than standard implementations when properly configured.

## Dependencies and Requirements

### Required Dependencies

**CuPy is required** but not installed by default. You must install it separately:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x  
pip install cupy-cuda12x

# For specific CUDA versions, see: https://docs.cupy.dev/en/stable/install.html
```

### System Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit installed (version 11.x or 12.x)
- Compatible NVIDIA drivers
- Sufficient GPU memory for your workloads

### Verification

Check if your setup is ready:

```python
try:
    import cupy as cp
    print("✅ CuPy installed")
    print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"GPU device: {cp.cuda.Device().name}")
except ImportError:
    print("❌ CuPy not installed")
except Exception as e:
    print(f"❌ CUDA setup issue: {e}")
```

## Implementation Status

### What's Implemented

The framework includes custom CUDA kernels for:

1. **GELU Activation** - Custom kernel with tanh-based approximation
2. **Fused Linear + GELU** - Combined linear transformation and GELU activation  
3. **Layer Normalization** - Optimized normalization with shared memory usage
4. **Flash Attention** - Memory-efficient attention computation

### Current Limitations

- **Performance claims are unverified** - actual speedup depends on hardware, data sizes, and specific use cases
- **No comprehensive benchmarking** has been performed across different GPU architectures  
- **Kernel compilation** may fail on some CUDA/GPU combinations
- **Fallback behavior** is implemented but may introduce performance overhead

## Usage Examples

### Basic CUDA Backend Usage

```python
from neural_arch.backends import get_backend
import cupy as cp

# Initialize CUDA backend (requires CuPy)
try:
    cuda_backend = get_backend("cuda")
    print(f"CUDA backend available: {cuda_backend.is_available}")
    print(f"Custom kernels available: {cuda_backend.has_custom_kernels}")
except ImportError:
    print("CuPy not installed - install with: pip install cupy-cuda11x")
    exit(1)
```

### GELU Activation

```python
# Create test data
x_gpu = cp.random.randn(1024, 512).astype(cp.float32)

# Apply GELU (uses custom kernel if available, falls back to standard implementation)
result = cuda_backend.gelu(x_gpu)

print(f"GELU applied to {x_gpu.shape} tensor")
print(f"Output shape: {result.shape}")
```

### Fused Linear + GELU Operations

```python
# Create test data for fused linear + GELU
input_gpu = cp.random.randn(128, 512).astype(cp.float32)
weight_gpu = cp.random.randn(768, 512).astype(cp.float32)  
bias_gpu = cp.random.randn(768).astype(cp.float32)

# Fused operation (if custom kernels available, otherwise falls back to standard)
output = cuda_backend.fused_linear_gelu(input_gpu, weight_gpu, bias_gpu)

print(f"Fused linear+GELU output: {output.shape}")
```

### Flash Attention

```python
# Flash Attention example
batch_size, num_heads, seq_len, head_dim = 4, 8, 512, 64

q = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
k = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
v = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)

scale = 1.0 / cp.sqrt(head_dim)

# Flash Attention (uses custom kernel if available)
attention_output = cuda_backend.flash_attention(q, k, v, scale, block_size=64)

print(f"Flash Attention output: {attention_output.shape}")
```

### Layer Normalization

```python
# Layer normalization example
batch_size, hidden_size = 128, 768

input_gpu = cp.random.randn(batch_size, hidden_size).astype(cp.float32)
weight_gpu = cp.ones(hidden_size, dtype=cp.float32)
bias_gpu = cp.zeros(hidden_size, dtype=cp.float32)

# Apply layer normalization
output = cuda_backend.layernorm(input_gpu, weight_gpu, bias_gpu, eps=1e-5)

print(f"LayerNorm output: {output.shape}")
```

## Advanced Features

### Kernel Manager

Access the underlying kernel manager for more control:

```python
from neural_arch.backends.cuda_kernels import get_cuda_kernel_manager

# Get kernel manager
kernel_manager = get_cuda_kernel_manager()

print(f"CUDA available: {kernel_manager.is_available()}")
print(f"Compiled kernels: {len(kernel_manager.compiled_kernels)}")

# Direct kernel usage (advanced)
if kernel_manager.is_available():
    x = cp.random.randn(1000, 1000).astype(cp.float32)
    result = kernel_manager.gelu_forward(x)
    print(f"Direct kernel result: {result.shape}")

```

### Performance Benchmarking

Benchmark kernel performance (if available):

```python
# Benchmark specific kernel (returns time in milliseconds)
if cuda_backend.has_custom_kernels:
    test_tensor = cp.random.randn(1024, 1024).astype(cp.float32)
    gelu_time = cuda_backend.benchmark_kernel("gelu", test_tensor, num_runs=10)
    print(f"Average GELU kernel time: {gelu_time:.4f}ms")
else:
    print("Custom kernels not available for benchmarking")
```

### Memory Usage Monitoring

Monitor GPU memory usage:

```python
def monitor_gpu_memory():
    """Monitor GPU memory usage."""
    mempool = cp.get_default_memory_pool()
    used_mb = mempool.used_bytes() / (1024**2)
    total_mb = mempool.total_bytes() / (1024**2)
    print(f"GPU Memory - Used: {used_mb:.1f} MB, Total: {total_mb:.1f} MB")

# Monitor memory during operations
monitor_gpu_memory()
tensor = cp.random.randn(2048, 2048).astype(cp.float32)
result = cuda_backend.gelu(tensor)
monitor_gpu_memory()

# Clean up
del tensor, result
cp.get_default_memory_pool().free_all_blocks()
monitor_gpu_memory()
```

## Troubleshooting

### Common Issues

#### CuPy Not Available

```python
# Check CuPy installation
try:
    import cupy as cp
    print("✅ CuPy available")
except ImportError:
    print("❌ CuPy not installed")
    print("Install with: pip install cupy-cuda11x  # or cupy-cuda12x")
```

#### CUDA Backend Not Available

```python
from neural_arch.backends import get_backend

try:
    cuda_backend = get_backend("cuda")
    if not cuda_backend.is_available:
        print("❌ CUDA backend not available")
        print("Check: NVIDIA GPU, CUDA drivers, CuPy installation")
    else:
        print("✅ CUDA backend available")
except Exception as e:
    print(f"❌ Error initializing CUDA backend: {e}")
```

#### Custom Kernels Failed to Compile

```python
cuda_backend = get_backend("cuda")
if not cuda_backend.has_custom_kernels:
    print("⚠️  Custom kernels not available")
    print("- Check CUDA toolkit installation")
    print("- Verify GPU compute capability")  
    print("- Operations will fall back to standard CuPy implementations")
else:
    print("✅ Custom kernels available")
```

#### GPU Memory Issues

```python
try:
    large_tensor = cp.random.randn(10000, 10000).astype(cp.float32)
    result = cuda_backend.gelu(large_tensor)
except cp.cuda.memory.OutOfMemoryError:
    print("❌ GPU out of memory")
    print("- Reduce tensor sizes")
    print("- Clear GPU memory: cp.get_default_memory_pool().free_all_blocks()")
    print("- Use gradient checkpointing for training")
```

### Testing Kernel Correctness

Verify that custom kernels produce correct results:

```python
def test_kernel_correctness():
    """Test CUDA kernel correctness if custom kernels are available."""
    if not cuda_backend.has_custom_kernels:
        print("⚠️  No custom kernels to test")
        return
        
    # Test data
    x = cp.random.randn(100, 512).astype(cp.float32)
    
    # Reference implementation
    def reference_gelu(x):
        sqrt_2_over_pi = cp.sqrt(2.0 / cp.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + cp.tanh(inner))
    
    # Compare results
    reference_result = reference_gelu(x)
    kernel_result = cuda_backend.gelu(x)
    
    # Check correctness
    max_diff = cp.max(cp.abs(reference_result - kernel_result))
    print(f"Maximum difference: {float(max_diff):.2e}")
    
    if max_diff < 1e-5:
        print("✅ Kernel correctness validated")
    else:
        print(f"⚠️  Large difference detected: {max_diff}")

# Run test
test_kernel_correctness()
```

## Summary

The neural architecture framework provides CUDA acceleration through custom kernels for GPU operations. Key points:

**What's Available:**
- Custom CUDA kernels for GELU, fused linear+GELU, layer normalization, and Flash Attention
- Automatic fallback to standard CuPy implementations when custom kernels aren't available
- Built-in benchmarking and memory monitoring tools

**Requirements:**
- CuPy must be installed separately (`pip install cupy-cuda11x` or `cupy-cuda12x`)
- NVIDIA GPU with CUDA support
- Compatible CUDA toolkit and drivers

**Current Status:**
- Implementations are functional but performance claims are unverified
- Custom kernels may not compile on all GPU/CUDA combinations
- Comprehensive benchmarking across different hardware configurations is needed

**Getting Started:**
1. Install CuPy for your CUDA version
2. Verify CUDA backend availability with `cuda_backend.is_available`  
3. Check custom kernel availability with `cuda_backend.has_custom_kernels`
4. Operations automatically use custom kernels when available, fall back to standard implementations otherwise

For production use, thoroughly test kernel compilation and performance on your specific hardware configuration.
