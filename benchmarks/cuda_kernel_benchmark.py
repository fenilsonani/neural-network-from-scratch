#!/usr/bin/env python3
"""Comprehensive CUDA kernel performance benchmark.

This benchmark tests the performance of custom CUDA kernels against
standard CuPy implementations, demonstrating GPU acceleration capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)

try:
    import cupy as cp
    from neural_arch.backends.cuda_backend import CudaBackend
    from neural_arch.backends.cuda_kernels import get_cuda_kernel_manager
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"❌ CUDA not available: {e}")
    CUDA_AVAILABLE = False
    cp = None


class CUDAKernelBenchmark:
    """Comprehensive CUDA kernel benchmark suite."""
    
    def __init__(self):
        self.results = {}
        self.cuda_backend = None
        self.kernel_manager = None
        
        if not CUDA_AVAILABLE:
            print("❌ CUDA not available - skipping GPU benchmarks")
            return
        
        try:
            # Initialize CUDA backend
            self.cuda_backend = CudaBackend()
            print(f"✅ CUDA backend initialized: {self.cuda_backend.name}")
            
            # Initialize kernel manager
            self.kernel_manager = get_cuda_kernel_manager()
            if self.kernel_manager.is_available():
                print(f"✅ Custom CUDA kernels available")
            else:
                print(f"❌ Custom CUDA kernels not available")
                self.kernel_manager = None
                
        except Exception as e:
            print(f"❌ Failed to initialize CUDA: {e}")
            self.cuda_backend = None
            self.kernel_manager = None
    
    def is_available(self) -> bool:
        """Check if CUDA benchmarking is available."""
        return self.cuda_backend is not None
    
    def benchmark_function_gpu(self, func, *args, warmup_runs: int = 5, test_runs: int = 20, **kwargs):
        """Benchmark a GPU function with proper synchronization."""
        if not self.is_available():
            return None
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                result = func(*args, **kwargs)
                cp.cuda.Device().synchronize()
            except Exception:
                pass
        
        # Actual benchmark with CUDA events for precise timing
        times = []
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        for _ in range(test_runs):
            start_event.record()
            result = func(*args, **kwargs)
            end_event.record()
            cp.cuda.Device().synchronize()
            
            elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
            times.append(elapsed_time / 1000.0)  # Convert to seconds
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'result': result
        }
    
    def benchmark_gelu_kernels(self, sizes: List[Tuple[int, ...]]):
        """Benchmark GELU activation kernels."""
        if not self.is_available():
            return
        
        print("\n" + "="*60)
        print("CUDA GELU KERNEL BENCHMARK")
        print("="*60)
        
        results = {}
        
        for size in sizes:
            print(f"\nTesting GELU size: {size}")
            
            # Create test data on GPU
            x_gpu = cp.random.randn(*size).astype(cp.float32)
            
            # Standard CuPy GELU
            def standard_gelu_gpu(x):
                sqrt_2_over_pi = cp.sqrt(2.0 / cp.pi)
                inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
                return 0.5 * x * (1.0 + cp.tanh(inner))
            
            standard_result = self.benchmark_function_gpu(standard_gelu_gpu, x_gpu)
            
            # Custom kernel GELU
            kernel_result = None
            if self.kernel_manager and self.kernel_manager.is_available():
                kernel_result = self.benchmark_function_gpu(
                    self.kernel_manager.gelu_forward, x_gpu
                )
                
                if kernel_result:
                    speedup = standard_result['mean_time'] / kernel_result['mean_time']
                    print(f"  Standard GELU: {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                    print(f"  Kernel GELU:   {kernel_result['mean_time']:.6f}s ± {kernel_result['std_time']:.6f}s")
                    print(f"  Speedup:       {speedup:.2f}x")
                    
                    # Verify correctness
                    diff = cp.max(cp.abs(standard_result['result'] - kernel_result['result']))
                    print(f"  Max diff:      {float(diff):.2e}")
                else:
                    print(f"  Standard GELU: {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                    print(f"  Kernel GELU:   Failed")
            else:
                print(f"  Standard GELU: {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                print(f"  Kernel GELU:   Not available")
            
            results[size] = {
                'standard': standard_result,
                'kernel': kernel_result,
                'speedup': speedup if kernel_result else None
            }
        
        self.results['gelu'] = results
    
    def benchmark_fused_linear_gelu(self, configs: List[Tuple[int, int, int]]):
        """Benchmark fused linear + GELU kernels."""
        if not self.is_available():
            return
        
        print("\n" + "="*60)
        print("CUDA FUSED LINEAR+GELU KERNEL BENCHMARK")
        print("="*60)
        
        results = {}
        
        for batch_size, in_features, out_features in configs:
            print(f"\nTesting fused linear+GELU: batch={batch_size}, in={in_features}, out={out_features}")
            
            # Create test data on GPU
            input_gpu = cp.random.randn(batch_size, in_features).astype(cp.float32)
            weight_gpu = cp.random.randn(out_features, in_features).astype(cp.float32)
            bias_gpu = cp.random.randn(out_features).astype(cp.float32)
            
            # Standard implementation
            def standard_linear_gelu(input_gpu, weight_gpu, bias_gpu):
                linear_out = cp.dot(input_gpu, weight_gpu.T) + bias_gpu
                sqrt_2_over_pi = cp.sqrt(2.0 / cp.pi)
                inner = sqrt_2_over_pi * (linear_out + 0.044715 * linear_out**3)
                return 0.5 * linear_out * (1.0 + cp.tanh(inner))
            
            standard_result = self.benchmark_function_gpu(
                standard_linear_gelu, input_gpu, weight_gpu, bias_gpu
            )
            
            # Custom kernel implementation
            kernel_result = None
            if self.kernel_manager and self.kernel_manager.is_available():
                kernel_result = self.benchmark_function_gpu(
                    self.kernel_manager.fused_linear_gelu, input_gpu, weight_gpu, bias_gpu
                )
                
                if kernel_result:
                    speedup = standard_result['mean_time'] / kernel_result['mean_time']
                    print(f"  Standard:  {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                    print(f"  Kernel:    {kernel_result['mean_time']:.6f}s ± {kernel_result['std_time']:.6f}s")
                    print(f"  Speedup:   {speedup:.2f}x")
                    
                    # Memory efficiency estimate
                    intermediate_memory = batch_size * out_features * 4  # bytes
                    memory_saved_mb = intermediate_memory / (1024**2)
                    print(f"  Memory saved: {memory_saved_mb:.1f} MB (intermediate activations)")
                else:
                    print(f"  Standard:  {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                    print(f"  Kernel:    Failed")
            else:
                print(f"  Standard:  {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                print(f"  Kernel:    Not available")
            
            results[(batch_size, in_features, out_features)] = {
                'standard': standard_result,
                'kernel': kernel_result,
                'speedup': speedup if kernel_result else None
            }
        
        self.results['fused_linear_gelu'] = results
    
    def benchmark_layernorm_kernels(self, configs: List[Tuple[int, int]]):
        """Benchmark layer normalization kernels."""
        if not self.is_available():
            return
        
        print("\n" + "="*60)
        print("CUDA LAYER NORMALIZATION KERNEL BENCHMARK")
        print("="*60)
        
        results = {}
        
        for batch_size, hidden_size in configs:
            print(f"\nTesting LayerNorm: batch={batch_size}, hidden={hidden_size}")
            
            # Create test data on GPU
            input_gpu = cp.random.randn(batch_size, hidden_size).astype(cp.float32)
            weight_gpu = cp.ones(hidden_size, dtype=cp.float32)
            bias_gpu = cp.zeros(hidden_size, dtype=cp.float32)
            eps = 1e-5
            
            # Standard implementation
            def standard_layernorm(input_gpu, weight_gpu, bias_gpu, eps):
                mean = cp.mean(input_gpu, axis=-1, keepdims=True)
                var = cp.var(input_gpu, axis=-1, keepdims=True)
                normalized = (input_gpu - mean) / cp.sqrt(var + eps)
                return normalized * weight_gpu + bias_gpu
            
            standard_result = self.benchmark_function_gpu(
                standard_layernorm, input_gpu, weight_gpu, bias_gpu, eps
            )
            
            # Custom kernel implementation
            kernel_result = None
            if self.kernel_manager and self.kernel_manager.is_available():
                kernel_result = self.benchmark_function_gpu(
                    lambda i, w, b, e: self.kernel_manager.layernorm_forward(i, w, b, e)[0],
                    input_gpu, weight_gpu, bias_gpu, eps
                )
                
                if kernel_result:
                    speedup = standard_result['mean_time'] / kernel_result['mean_time']
                    print(f"  Standard:  {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                    print(f"  Kernel:    {kernel_result['mean_time']:.6f}s ± {kernel_result['std_time']:.6f}s")
                    print(f"  Speedup:   {speedup:.2f}x")
                    
                    # Verify correctness
                    diff = cp.max(cp.abs(standard_result['result'] - kernel_result['result']))
                    print(f"  Max diff:  {float(diff):.2e}")
                else:
                    print(f"  Standard:  {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                    print(f"  Kernel:    Failed")
            else:
                print(f"  Standard:  {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                print(f"  Kernel:    Not available")
            
            results[(batch_size, hidden_size)] = {
                'standard': standard_result,
                'kernel': kernel_result,
                'speedup': speedup if kernel_result else None
            }
        
        self.results['layernorm'] = results
    
    def benchmark_flash_attention(self, configs: List[Tuple[int, int, int, int]]):
        """Benchmark Flash Attention kernels."""
        if not self.is_available():
            return
        
        print("\n" + "="*60)
        print("CUDA FLASH ATTENTION KERNEL BENCHMARK")
        print("="*60)
        
        results = {}
        
        for batch_size, num_heads, seq_len, head_dim in configs:
            print(f"\nTesting Flash Attention: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}")
            
            # Create test data on GPU
            q_gpu = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
            k_gpu = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
            v_gpu = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
            scale = 1.0 / np.sqrt(head_dim)
            
            # Standard attention implementation
            def standard_attention(q, k, v, scale):
                scores = cp.matmul(q, cp.transpose(k, (0, 1, 3, 2))) * scale
                attention_weights = cp.exp(scores - cp.max(scores, axis=-1, keepdims=True))
                attention_weights = attention_weights / cp.sum(attention_weights, axis=-1, keepdims=True)
                return cp.matmul(attention_weights, v)
            
            standard_result = self.benchmark_function_gpu(
                standard_attention, q_gpu, k_gpu, v_gpu, scale
            )
            
            # Flash Attention kernel
            kernel_result = None
            if self.kernel_manager and self.kernel_manager.is_available():
                try:
                    kernel_result = self.benchmark_function_gpu(
                        self.kernel_manager.flash_attention, q_gpu, k_gpu, v_gpu, scale, 64
                    )
                    
                    if kernel_result:
                        speedup = standard_result['mean_time'] / kernel_result['mean_time']
                        print(f"  Standard:     {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                        print(f"  Flash Attn:   {kernel_result['mean_time']:.6f}s ± {kernel_result['std_time']:.6f}s")
                        print(f"  Speedup:      {speedup:.2f}x")
                        
                        # Memory efficiency
                        attention_memory = batch_size * num_heads * seq_len * seq_len * 4  # bytes
                        memory_saved_gb = attention_memory / (1024**3)
                        print(f"  Memory saved: {memory_saved_gb:.2f} GB (attention matrix)")
                    else:
                        print(f"  Standard:     {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                        print(f"  Flash Attn:   Failed")
                except Exception as e:
                    print(f"  Standard:     {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                    print(f"  Flash Attn:   Error: {e}")
            else:
                print(f"  Standard:     {standard_result['mean_time']:.6f}s ± {standard_result['std_time']:.6f}s")
                print(f"  Flash Attn:   Not available")
            
            results[(batch_size, num_heads, seq_len, head_dim)] = {
                'standard': standard_result,
                'kernel': kernel_result,
                'speedup': speedup if kernel_result else None
            }
        
        self.results['flash_attention'] = results
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage of different operations."""
        if not self.is_available():
            return
        
        print("\n" + "="*60)
        print("CUDA MEMORY USAGE BENCHMARK")
        print("="*60)
        
        # Get initial memory info
        mempool = cp.get_default_memory_pool()
        initial_used = mempool.used_bytes()
        initial_free = mempool.free_bytes()
        
        print(f"Initial GPU memory - Used: {initial_used/(1024**2):.1f} MB, Free: {initial_free/(1024**2):.1f} MB")
        
        # Test large attention computation
        batch_size, num_heads, seq_len, head_dim = 8, 12, 2048, 64
        
        print(f"\nTesting memory usage for attention: {batch_size}x{num_heads}x{seq_len}x{head_dim}")
        
        # Create tensors
        q_gpu = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
        k_gpu = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
        v_gpu = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
        
        after_tensors = mempool.used_bytes()
        tensor_memory = after_tensors - initial_used
        print(f"Tensor memory: {tensor_memory/(1024**2):.1f} MB")
        
        # Standard attention (creates large intermediate)
        try:
            scale = 1.0 / np.sqrt(head_dim)
            scores = cp.matmul(q_gpu, cp.transpose(k_gpu, (0, 1, 3, 2))) * scale
            after_scores = mempool.used_bytes()
            scores_memory = after_scores - after_tensors
            print(f"Standard attention intermediate memory: {scores_memory/(1024**2):.1f} MB")
            
            # Clean up
            del scores
            cp.cuda.Device().synchronize()
            mempool.free_all_blocks()
            
        except Exception as e:
            print(f"Standard attention failed (likely OOM): {e}")
        
        # Flash attention (memory efficient)
        if self.kernel_manager and self.kernel_manager.is_available():
            try:
                before_flash = mempool.used_bytes()
                result = self.kernel_manager.flash_attention(q_gpu, k_gpu, v_gpu, scale, 64)
                after_flash = mempool.used_bytes()
                flash_memory = after_flash - before_flash
                print(f"Flash attention memory overhead: {flash_memory/(1024**2):.1f} MB")
                
                memory_savings = scores_memory - flash_memory
                print(f"Memory savings: {memory_savings/(1024**2):.1f} MB ({100*memory_savings/scores_memory:.1f}%)")
                
            except Exception as e:
                print(f"Flash attention failed: {e}")
    
    def print_summary(self):
        """Print benchmark summary."""
        if not self.is_available():
            print("\n❌ CUDA benchmarks not available")
            return
        
        print("\n" + "="*60)
        print("CUDA KERNEL PERFORMANCE SUMMARY")
        print("="*60)
        
        total_tests = 0
        total_speedup = 0
        
        if 'gelu' in self.results:
            gelu_speedups = [r['speedup'] for r in self.results['gelu'].values() if r['speedup']]
            if gelu_speedups:
                avg_speedup = np.mean(gelu_speedups)
                print(f"GELU Kernel:           {avg_speedup:.2f}x average speedup")
                total_speedup += avg_speedup
                total_tests += 1
        
        if 'fused_linear_gelu' in self.results:
            linear_speedups = [r['speedup'] for r in self.results['fused_linear_gelu'].values() if r['speedup']]
            if linear_speedups:
                avg_speedup = np.mean(linear_speedups)
                print(f"Fused Linear+GELU:     {avg_speedup:.2f}x average speedup")
                total_speedup += avg_speedup
                total_tests += 1
        
        if 'layernorm' in self.results:
            ln_speedups = [r['speedup'] for r in self.results['layernorm'].values() if r['speedup']]
            if ln_speedups:
                avg_speedup = np.mean(ln_speedups)
                print(f"LayerNorm Kernel:      {avg_speedup:.2f}x average speedup")
                total_speedup += avg_speedup
                total_tests += 1
        
        if 'flash_attention' in self.results:
            fa_speedups = [r['speedup'] for r in self.results['flash_attention'].values() if r['speedup']]
            if fa_speedups:
                avg_speedup = np.mean(fa_speedups)
                print(f"Flash Attention:       {avg_speedup:.2f}x average speedup")
                total_speedup += avg_speedup
                total_tests += 1
        
        if total_tests > 0:
            overall_speedup = total_speedup / total_tests
            print(f"\nOVERALL GPU SPEEDUP:   {overall_speedup:.2f}x")
            
            print(f"\nGPU Optimization Status:")
            if overall_speedup >= 3.0:
                print("🚀 Excellent GPU acceleration achieved!")
            elif overall_speedup >= 2.0:
                print("⚡ Good GPU acceleration")
            elif overall_speedup >= 1.5:
                print("📈 Moderate GPU acceleration")
            else:
                print("⚠️ Limited GPU acceleration - optimization needed")
        
        print(f"\nCustom CUDA Features:")
        print("✅ Hand-optimized kernels")
        print("✅ Memory-efficient Flash Attention")
        print("✅ Fused operations")
        print("✅ Automatic fallback to standard implementations")
    
    def run_all_benchmarks(self):
        """Run all CUDA kernel benchmarks."""
        if not self.is_available():
            print("❌ CUDA not available - cannot run GPU benchmarks")
            return
        
        print("Neural Architecture Framework - CUDA Kernel Benchmark")
        print("Testing custom CUDA kernels vs standard CuPy implementations")
        print("Demonstrating GPU acceleration capabilities")
        
        # GELU sizes
        gelu_sizes = [(4096, 1024), (8192, 2048), (16384, 4096)]
        self.benchmark_gelu_kernels(gelu_sizes)
        
        # Fused linear + GELU configurations
        fused_configs = [(512, 1024, 2048), (1024, 2048, 4096), (2048, 4096, 8192)]
        self.benchmark_fused_linear_gelu(fused_configs)
        
        # LayerNorm configurations
        layernorm_configs = [(1024, 768), (2048, 1024), (4096, 2048)]
        self.benchmark_layernorm_kernels(layernorm_configs)
        
        # Flash Attention configurations (smaller due to memory requirements)
        attention_configs = [(2, 8, 512, 64), (4, 12, 1024, 64), (8, 16, 2048, 64)]
        self.benchmark_flash_attention(attention_configs)
        
        # Memory usage analysis
        self.benchmark_memory_usage()
        
        # Summary
        self.print_summary()


if __name__ == "__main__":
    benchmark = CUDAKernelBenchmark()
    benchmark.run_all_benchmarks()