"""Performance regression tests for GPU acceleration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from typing import Dict, List

import numpy as np
import pytest

from neural_arch.backends import available_backends, get_backend


class PerformanceBenchmark:
    """Helper class for performance benchmarking."""

    @staticmethod
    def time_operation(func, warmup_runs: int = 3, test_runs: int = 10) -> float:
        """Time an operation with warmup."""
        # Warmup
        for _ in range(warmup_runs):
            func()

        # Time the operation
        start = time.perf_counter()
        for _ in range(test_runs):
            func()
        end = time.perf_counter()

        return (end - start) / test_runs


class TestBackendPerformance:
    """Test performance characteristics of backends."""

    def setup_method(self):
        """Set up test data."""
        self.sizes = [
            (100, 100),
            (500, 500),
            (1000, 1000),
            (2000, 2000),
        ]
        self.batch_sizes = [1, 8, 32, 128]

    @pytest.mark.performance
    def test_matmul_scaling(self):
        """Test how matrix multiplication scales with size."""
        backends = available_backends()
        results = {}

        for backend_name in backends:
            backend = get_backend(backend_name)
            results[backend_name] = {}

            for size in self.sizes:
                # Create test matrices
                np.random.seed(42)
                a_np = np.random.randn(*size).astype(np.float32)
                b_np = np.random.randn(*size).astype(np.float32)

                # Convert to backend
                a = backend.from_numpy(a_np)
                b = backend.from_numpy(b_np)

                # Time the operation
                def matmul_op():
                    return backend.matmul(a, b)

                elapsed = PerformanceBenchmark.time_operation(matmul_op)
                results[backend_name][size] = elapsed

                # Calculate GFLOPS
                flops = 2 * size[0] * size[1] * size[1]  # 2mn^2 for m×n @ n×m
                gflops = flops / (elapsed * 1e9)

                print(f"{backend_name} - Size {size}: {elapsed*1000:.2f}ms ({gflops:.2f} GFLOPS)")

        # Verify GPU is faster for large matrices
        if len(backends) > 1 and "numpy" in backends:
            for size in self.sizes[2:]:  # Check larger sizes
                cpu_time = results["numpy"][size]
                for backend_name in backends:
                    if backend_name != "numpy":
                        gpu_time = results[backend_name][size]
                        speedup = cpu_time / gpu_time
                        print(f"  {backend_name} speedup for {size}: {speedup:.2f}x")

                        # GPU should be faster for large matrices
                        if size[0] >= 1000:
                            assert speedup > 0.8, f"{backend_name} not efficient for size {size}"

    @pytest.mark.performance
    def test_batch_operations(self):
        """Test performance with batched operations."""
        backends = available_backends()

        for backend_name in backends:
            backend = get_backend(backend_name)

            for batch_size in self.batch_sizes:
                # Create batch of matrices
                shape = (batch_size, 100, 100)
                np.random.seed(42)
                data_np = np.random.randn(*shape).astype(np.float32)

                data = backend.from_numpy(data_np)

                # Time reduction across batch
                def batch_mean():
                    return backend.mean(data, axis=0)

                elapsed = PerformanceBenchmark.time_operation(batch_mean)
                throughput = batch_size / elapsed

                print(f"{backend_name} - Batch {batch_size}: {throughput:.0f} items/sec")

    @pytest.mark.performance
    def test_memory_transfer_overhead(self):
        """Test overhead of transferring data to/from GPU."""
        backends = available_backends()

        # Different data sizes
        data_sizes = [
            (100,),  # 400 bytes
            (1000, 1000),  # 4 MB
            (2000, 2000),  # 16 MB
            (4000, 4000),  # 64 MB
        ]

        for backend_name in backends:
            if backend_name == "numpy":
                continue  # Skip CPU backend

            backend = get_backend(backend_name)
            print(f"\n{backend_name} Memory Transfer Overhead:")

            for shape in data_sizes:
                # Create test data
                data_np = np.random.randn(*shape).astype(np.float32)
                size_mb = data_np.nbytes / (1024 * 1024)

                # Time transfer to device
                def to_device():
                    return backend.from_numpy(data_np)

                to_time = PerformanceBenchmark.time_operation(to_device, warmup_runs=1, test_runs=5)

                # Time transfer from device
                device_data = backend.from_numpy(data_np)

                def from_device():
                    return backend.to_numpy(device_data)

                from_time = PerformanceBenchmark.time_operation(
                    from_device, warmup_runs=1, test_runs=5
                )

                # Calculate bandwidth
                to_bandwidth = size_mb / to_time
                from_bandwidth = size_mb / from_time

                print(f"  {shape} ({size_mb:.1f} MB):")
                print(f"    To device: {to_time*1000:.2f}ms ({to_bandwidth:.0f} MB/s)")
                print(f"    From device: {from_time*1000:.2f}ms ({from_bandwidth:.0f} MB/s)")

    @pytest.mark.performance
    def test_operation_fusion(self):
        """Test performance of fused vs separate operations."""
        backends = available_backends()

        # Create test data
        size = (1000, 1000)
        np.random.seed(42)
        x_np = np.random.randn(*size).astype(np.float32)
        w_np = np.random.randn(*size).astype(np.float32)
        b_np = np.random.randn(size[1]).astype(np.float32)

        for backend_name in backends:
            backend = get_backend(backend_name)

            # Convert to backend
            x = backend.from_numpy(x_np)
            w = backend.from_numpy(w_np)
            b = backend.from_numpy(b_np)

            # Separate operations: y = relu(x @ w + b)
            def separate_ops():
                z1 = backend.matmul(x, w)
                z2 = backend.add(z1, b)
                z3 = backend.clip(z2, 0, float("inf"))  # ReLU
                return z3

            # Time separate operations
            separate_time = PerformanceBenchmark.time_operation(separate_ops)

            # For comparison, time just matmul
            def just_matmul():
                return backend.matmul(x, w)

            matmul_time = PerformanceBenchmark.time_operation(just_matmul)

            overhead = (separate_time - matmul_time) / matmul_time * 100

            print(f"{backend_name} - Operation overhead: {overhead:.1f}%")

    @pytest.mark.performance
    def test_concurrent_operations(self):
        """Test performance with concurrent operations."""
        backends = available_backends()

        for backend_name in backends:
            if backend_name == "numpy":
                continue  # CPU is sequential

            backend = get_backend(backend_name)

            # Create multiple arrays
            size = (500, 500)
            arrays = []
            for i in range(4):
                np.random.seed(i)
                arr_np = np.random.randn(*size).astype(np.float32)
                arrays.append(backend.from_numpy(arr_np))

            # Sequential operations
            def sequential():
                results = []
                for i in range(len(arrays) - 1):
                    results.append(backend.matmul(arrays[i], arrays[i + 1]))
                return results

            seq_time = PerformanceBenchmark.time_operation(sequential)

            print(f"{backend_name} - Sequential ops: {seq_time*1000:.2f}ms")

    @pytest.mark.performance
    def test_reduction_performance(self):
        """Test performance of reduction operations."""
        backends = available_backends()

        # Different reduction scenarios
        test_cases = [
            ((1000, 1000), None),  # Full reduction
            ((1000, 1000), 0),  # Reduce along axis 0
            ((1000, 1000), 1),  # Reduce along axis 1
            ((100, 100, 100), (0, 2)),  # Multi-axis reduction
        ]

        for backend_name in backends:
            backend = get_backend(backend_name)
            print(f"\n{backend_name} Reduction Performance:")

            for shape, axis in test_cases:
                # Create test data
                np.random.seed(42)
                data_np = np.random.randn(*shape).astype(np.float32)
                data = backend.from_numpy(data_np)

                # Time different reductions
                reductions = {
                    "sum": lambda: backend.sum(data, axis=axis),
                    "mean": lambda: backend.mean(data, axis=axis),
                    "max": lambda: backend.max(data, axis=axis),
                }

                for name, func in reductions.items():
                    elapsed = PerformanceBenchmark.time_operation(func)
                    elements = np.prod(shape)
                    throughput = elements / (elapsed * 1e9)  # Gelements/sec

                    print(
                        f"  {name} {shape} axis={axis}: {elapsed*1000:.2f}ms ({throughput:.2f} GE/s)"
                    )


class TestPerformanceRegression:
    """Test for performance regressions."""

    # Expected performance baselines (in ms)
    # These should be calibrated for your specific hardware
    PERFORMANCE_BASELINES = {
        "numpy": {
            "matmul_1000": 50.0,  # 50ms for 1000x1000 matmul
            "reduction_1M": 2.0,  # 2ms for 1M element reduction
        },
        "mps": {
            "matmul_1000": 5.0,  # 5ms for 1000x1000 matmul
            "reduction_1M": 0.5,  # 0.5ms for 1M element reduction
        },
        "cuda": {
            "matmul_1000": 3.0,  # 3ms for 1000x1000 matmul
            "reduction_1M": 0.3,  # 0.3ms for 1M element reduction
        },
    }

    @pytest.mark.regression
    def test_matmul_regression(self):
        """Test that matmul performance hasn't regressed."""
        backends = available_backends()

        for backend_name in backends:
            if backend_name not in self.PERFORMANCE_BASELINES:
                continue

            backend = get_backend(backend_name)

            # Create 1000x1000 matrices
            np.random.seed(42)
            a_np = np.random.randn(1000, 1000).astype(np.float32)
            b_np = np.random.randn(1000, 1000).astype(np.float32)

            a = backend.from_numpy(a_np)
            b = backend.from_numpy(b_np)

            # Time matmul
            def matmul_op():
                return backend.matmul(a, b)

            elapsed_ms = PerformanceBenchmark.time_operation(matmul_op) * 1000
            baseline_ms = self.PERFORMANCE_BASELINES[backend_name]["matmul_1000"]

            # Allow 50% degradation as threshold
            assert (
                elapsed_ms < baseline_ms * 1.5
            ), f"{backend_name} matmul regressed: {elapsed_ms:.2f}ms vs {baseline_ms:.2f}ms baseline"

    @pytest.mark.regression
    def test_reduction_regression(self):
        """Test that reduction performance hasn't regressed."""
        backends = available_backends()

        for backend_name in backends:
            if backend_name not in self.PERFORMANCE_BASELINES:
                continue

            backend = get_backend(backend_name)

            # Create 1M element array
            np.random.seed(42)
            data_np = np.random.randn(1000, 1000).astype(np.float32)
            data = backend.from_numpy(data_np)

            # Time reduction
            def reduction_op():
                return backend.sum(data)

            elapsed_ms = PerformanceBenchmark.time_operation(reduction_op) * 1000
            baseline_ms = self.PERFORMANCE_BASELINES[backend_name]["reduction_1M"]

            # Allow 50% degradation as threshold
            assert (
                elapsed_ms < baseline_ms * 1.5
            ), f"{backend_name} reduction regressed: {elapsed_ms:.2f}ms vs {baseline_ms:.2f}ms baseline"


if __name__ == "__main__":
    # Run with: pytest test_backend_performance.py -v -m performance
    pytest.main([__file__, "-v", "-m", "performance"])
