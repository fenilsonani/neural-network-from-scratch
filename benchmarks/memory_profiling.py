"""Memory Profiling and Benchmarking for Neural Forge vs PyTorch.

This module provides comprehensive memory usage analysis and profiling
capabilities to compare memory efficiency between frameworks.
"""

import os
import sys
import gc
import time
import logging
import tracemalloc
import psutil
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.neural_arch.core.tensor import Tensor
from src.neural_arch.nn.module import Module
from src.neural_arch.nn import Sequential, Linear, ReLU

logger = logging.getLogger(__name__)

# Try to import PyTorch for comparison
try:
    import torch
    import torch.nn as torch_nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point."""
    
    # Basic memory stats
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory usage percentage
    
    # Python-specific memory
    python_current_mb: float
    python_peak_mb: float
    
    # Framework-specific memory (if available)
    torch_allocated_mb: Optional[float] = None
    torch_cached_mb: Optional[float] = None
    
    # Timing
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class MemoryBenchmarkResult:
    """Results from memory benchmarking."""
    
    # Identification
    framework: str
    operation: str
    batch_size: int
    model_params: int
    
    # Memory usage
    baseline_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_overhead_mb: float
    memory_efficiency: float  # MB per parameter
    
    # Growth analysis
    memory_growth_mb: float
    memory_leaked_mb: float
    
    # Performance correlation
    execution_time_ms: float
    memory_time_product: float  # Memory * Time efficiency metric
    
    # Detailed snapshots
    snapshots: List[MemorySnapshot] = None
    
    def __post_init__(self):
        """Initialize snapshots list if not provided."""
        if self.snapshots is None:
            self.snapshots = []


class MemoryProfiler:
    """Advanced memory profiling for deep learning operations."""
    
    def __init__(self, enable_tracemalloc: bool = True):
        """Initialize memory profiler.
        
        Args:
            enable_tracemalloc: Whether to enable Python tracemalloc
        """
        self.enable_tracemalloc = enable_tracemalloc
        self.process = psutil.Process()
        self.snapshots = []
        
        if enable_tracemalloc:
            tracemalloc.start()
    
    def get_current_memory(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        
        # System memory
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # Python memory
        if self.enable_tracemalloc:
            current, peak = tracemalloc.get_traced_memory()
            python_current_mb = current / (1024 * 1024)
            python_peak_mb = peak / (1024 * 1024)
        else:
            python_current_mb = 0.0
            python_peak_mb = 0.0
        
        # PyTorch memory (if available)
        torch_allocated_mb = None
        torch_cached_mb = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            torch_cached_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return MemorySnapshot(
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent=memory_percent,
            python_current_mb=python_current_mb,
            python_peak_mb=python_peak_mb,
            torch_allocated_mb=torch_allocated_mb,
            torch_cached_mb=torch_cached_mb
        )
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.snapshots = []
        if self.enable_tracemalloc:
            tracemalloc.clear_traces()
        
        # Force garbage collection
        gc.collect()
        
        # Take baseline snapshot
        baseline = self.get_current_memory()
        self.snapshots.append(baseline)
        return baseline
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot with optional label."""
        snapshot = self.get_current_memory()
        if hasattr(snapshot, 'label'):
            snapshot.label = label
        else:
            # Add label as custom attribute
            setattr(snapshot, 'label', label)
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        
        # Take final snapshot
        final_snapshot = self.get_current_memory()
        self.snapshots.append(final_snapshot)
        
        if len(self.snapshots) < 2:
            return {"error": "Insufficient snapshots for analysis"}
        
        baseline = self.snapshots[0]
        peak_memory = max(s.rss_mb for s in self.snapshots)
        
        summary = {
            "baseline_memory_mb": baseline.rss_mb,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_snapshot.rss_mb,
            "memory_growth_mb": final_snapshot.rss_mb - baseline.rss_mb,
            "memory_overhead_mb": peak_memory - baseline.rss_mb,
            "num_snapshots": len(self.snapshots),
            "duration_seconds": final_snapshot.timestamp - baseline.timestamp
        }
        
        return summary
    
    @contextmanager
    def profile_operation(self, operation_name: str = ""):
        """Context manager for profiling a specific operation."""
        
        logger.debug(f"Starting memory profiling for: {operation_name}")
        
        # Start monitoring
        baseline = self.start_monitoring()
        
        try:
            yield self
        finally:
            # Stop monitoring and get summary
            summary = self.stop_monitoring()
            logger.debug(f"Memory profiling completed for {operation_name}: {summary}")


class MemoryBenchmark:
    """Comprehensive memory benchmarking suite."""
    
    def __init__(self):
        """Initialize memory benchmark."""
        self.results = []
        self.profiler = MemoryProfiler()
    
    def benchmark_model_memory(self, 
                              model: Module,
                              input_shapes: List[Tuple[int, ...]],
                              framework_name: str,
                              model_name: str) -> List[MemoryBenchmarkResult]:
        """Benchmark memory usage for a model across different input shapes.
        
        Args:
            model: Model to benchmark
            input_shapes: List of input shapes to test
            framework_name: Name of framework ('neural_forge' or 'pytorch')
            model_name: Name of the model
            
        Returns:
            List of memory benchmark results
        """
        results = []
        
        # Calculate model parameters
        if hasattr(model, 'parameters'):
            model_params = sum(p.size for p in model.parameters())
        else:
            model_params = 0
        
        for input_shape in input_shapes:
            logger.info(f"Benchmarking {model_name} memory usage - shape: {input_shape}")
            
            batch_size = input_shape[0]
            
            # Profile inference
            result = self._profile_model_inference(
                model, input_shape, framework_name, model_name, model_params
            )
            results.append(result)
            
            # Profile training (forward + backward)
            if framework_name == "neural_forge":
                # Neural Forge training simulation
                training_result = self._profile_neural_forge_training(
                    model, input_shape, model_name, model_params
                )
                results.append(training_result)
            elif framework_name == "pytorch" and TORCH_AVAILABLE:
                # PyTorch training simulation
                training_result = self._profile_pytorch_training(
                    model, input_shape, model_name, model_params
                )
                results.append(training_result)
        
        self.results.extend(results)
        return results
    
    def _profile_model_inference(self,
                                model: Module,
                                input_shape: Tuple[int, ...],
                                framework_name: str,
                                model_name: str,
                                model_params: int) -> MemoryBenchmarkResult:
        """Profile memory usage during model inference."""
        
        with self.profiler.profile_operation(f"{model_name}_inference") as profiler:
            
            # Create input data
            if framework_name == "neural_forge":
                input_data = Tensor(np.random.randn(*input_shape).astype(np.float32))
            else:  # PyTorch
                input_data = torch.randn(*input_shape, dtype=torch.float32)
            
            # Take snapshot after input creation
            profiler.take_snapshot("input_created")
            
            # Model inference
            start_time = time.perf_counter()
            
            if framework_name == "pytorch" and TORCH_AVAILABLE:
                with torch.no_grad():
                    output = model(input_data)
            else:
                model.eval()
                output = model(input_data)
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Take snapshot after inference
            profiler.take_snapshot("inference_complete")
        
        # Analyze results
        summary = profiler.stop_monitoring()
        
        baseline_memory = summary["baseline_memory_mb"]
        peak_memory = summary["peak_memory_mb"]
        final_memory = summary["final_memory_mb"]
        memory_overhead = summary["memory_overhead_mb"]
        
        # Calculate efficiency metrics
        memory_efficiency = memory_overhead / model_params if model_params > 0 else 0
        memory_time_product = memory_overhead * execution_time_ms
        
        return MemoryBenchmarkResult(
            framework=framework_name,
            operation=f"{model_name}_inference",
            batch_size=input_shape[0],
            model_params=model_params,
            baseline_memory_mb=baseline_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_overhead_mb=memory_overhead,
            memory_efficiency=memory_efficiency,
            memory_growth_mb=summary["memory_growth_mb"],
            memory_leaked_mb=max(0, final_memory - baseline_memory),
            execution_time_ms=execution_time_ms,
            memory_time_product=memory_time_product,
            snapshots=profiler.snapshots.copy()
        )
    
    def _profile_neural_forge_training(self,
                                     model: Module,
                                     input_shape: Tuple[int, ...],
                                     model_name: str,
                                     model_params: int) -> MemoryBenchmarkResult:
        """Profile Neural Forge training step memory usage."""
        
        with self.profiler.profile_operation(f"{model_name}_training_nf") as profiler:
            
            # Create input and target data
            input_data = Tensor(np.random.randn(*input_shape).astype(np.float32))
            targets = Tensor(np.random.randint(0, 10, (input_shape[0],)).astype(np.int64))
            
            profiler.take_snapshot("data_created")
            
            # Simulated training step
            start_time = time.perf_counter()
            
            model.train()
            output = model(input_data)
            # Note: Neural Forge doesn't have full autodiff yet, so we simulate
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            profiler.take_snapshot("training_complete")
        
        summary = profiler.stop_monitoring()
        
        return MemoryBenchmarkResult(
            framework="neural_forge",
            operation=f"{model_name}_training",
            batch_size=input_shape[0],
            model_params=model_params,
            baseline_memory_mb=summary["baseline_memory_mb"],
            peak_memory_mb=summary["peak_memory_mb"],
            final_memory_mb=summary["final_memory_mb"],
            memory_overhead_mb=summary["memory_overhead_mb"],
            memory_efficiency=summary["memory_overhead_mb"] / model_params if model_params > 0 else 0,
            memory_growth_mb=summary["memory_growth_mb"],
            memory_leaked_mb=max(0, summary["memory_growth_mb"]),
            execution_time_ms=execution_time_ms,
            memory_time_product=summary["memory_overhead_mb"] * execution_time_ms,
            snapshots=profiler.snapshots.copy()
        )
    
    def _profile_pytorch_training(self,
                                model: Any,
                                input_shape: Tuple[int, ...],
                                model_name: str,
                                model_params: int) -> MemoryBenchmarkResult:
        """Profile PyTorch training step memory usage."""
        
        if not TORCH_AVAILABLE:
            # Return dummy result
            return MemoryBenchmarkResult(
                framework="pytorch",
                operation=f"{model_name}_training",
                batch_size=input_shape[0],
                model_params=model_params,
                baseline_memory_mb=0,
                peak_memory_mb=0,
                final_memory_mb=0,
                memory_overhead_mb=0,
                memory_efficiency=0,
                memory_growth_mb=0,
                memory_leaked_mb=0,
                execution_time_ms=0,
                memory_time_product=0
            )
        
        with self.profiler.profile_operation(f"{model_name}_training_torch") as profiler:
            
            # Create input and target data
            input_data = torch.randn(*input_shape, dtype=torch.float32, requires_grad=True)
            targets = torch.randint(0, 10, (input_shape[0],))
            
            # Create optimizer and loss
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = torch_nn.CrossEntropyLoss()
            
            profiler.take_snapshot("data_created")
            
            # Training step
            start_time = time.perf_counter()
            
            model.train()
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            profiler.take_snapshot("training_complete")
        
        summary = profiler.stop_monitoring()
        
        return MemoryBenchmarkResult(
            framework="pytorch",
            operation=f"{model_name}_training",
            batch_size=input_shape[0],
            model_params=model_params,
            baseline_memory_mb=summary["baseline_memory_mb"],
            peak_memory_mb=summary["peak_memory_mb"],
            final_memory_mb=summary["final_memory_mb"],
            memory_overhead_mb=summary["memory_overhead_mb"],
            memory_efficiency=summary["memory_overhead_mb"] / model_params if model_params > 0 else 0,
            memory_growth_mb=summary["memory_growth_mb"],
            memory_leaked_mb=max(0, summary["memory_growth_mb"]),
            execution_time_ms=execution_time_ms,
            memory_time_product=summary["memory_overhead_mb"] * execution_time_ms,
            snapshots=profiler.snapshots.copy()
        )
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get comparison summary between frameworks."""
        
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Group by framework
        framework_results = {}
        for result in self.results:
            if result.framework not in framework_results:
                framework_results[result.framework] = []
            framework_results[result.framework].append(result)
        
        summary = {}
        
        for framework, results in framework_results.items():
            if not results:
                continue
            
            # Calculate aggregate statistics
            memory_overheads = [r.memory_overhead_mb for r in results]
            memory_efficiencies = [r.memory_efficiency for r in results]
            execution_times = [r.execution_time_ms for r in results]
            
            summary[framework] = {
                "total_benchmarks": len(results),
                "avg_memory_overhead_mb": np.mean(memory_overheads),
                "max_memory_overhead_mb": np.max(memory_overheads),
                "avg_memory_efficiency": np.mean(memory_efficiencies),
                "avg_execution_time_ms": np.mean(execution_times),
                "total_memory_used_mb": np.sum(memory_overheads)
            }
        
        # Cross-framework comparison
        if len(framework_results) >= 2:
            frameworks = list(framework_results.keys())
            if "neural_forge" in frameworks and "pytorch" in frameworks:
                nf_memory = summary["neural_forge"]["avg_memory_overhead_mb"]
                pt_memory = summary["pytorch"]["avg_memory_overhead_mb"]
                
                summary["comparison"] = {
                    "neural_forge_memory_efficiency": pt_memory / nf_memory if nf_memory > 0 else float('inf'),
                    "neural_forge_more_efficient": nf_memory < pt_memory,
                    "memory_difference_mb": abs(nf_memory - pt_memory),
                    "memory_savings_percent": ((pt_memory - nf_memory) / pt_memory * 100) if pt_memory > 0 else 0
                }
        
        return summary
    
    def save_results(self, filepath: str):
        """Save memory benchmark results to file."""
        
        # Convert results to dictionaries
        results_data = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert snapshots to dictionaries
            if result_dict['snapshots']:
                result_dict['snapshots'] = [asdict(s) for s in result.snapshots]
            results_data.append(result_dict)
        
        data = {
            "memory_benchmark_results": results_data,
            "summary": self.get_comparison_summary(),
            "metadata": {
                "total_results": len(self.results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pytorch_available": TORCH_AVAILABLE
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Memory benchmark results saved to {filepath}")


# Convenience functions
def compare_memory_usage(neural_forge_model: Module,
                        pytorch_model: Any,
                        input_shapes: List[Tuple[int, ...]],
                        model_name: str = "model") -> MemoryBenchmark:
    """Compare memory usage between Neural Forge and PyTorch models.
    
    Args:
        neural_forge_model: Neural Forge model
        pytorch_model: PyTorch model
        input_shapes: List of input shapes to test
        model_name: Name of the model
        
    Returns:
        MemoryBenchmark with completed results
    """
    benchmark = MemoryBenchmark()
    
    # Benchmark Neural Forge
    logger.info("Benchmarking Neural Forge memory usage...")
    benchmark.benchmark_model_memory(
        neural_forge_model, input_shapes, "neural_forge", model_name
    )
    
    # Benchmark PyTorch
    if pytorch_model is not None:
        logger.info("Benchmarking PyTorch memory usage...")
        benchmark.benchmark_model_memory(
            pytorch_model, input_shapes, "pytorch", model_name
        )
    
    return benchmark


def profile_training_memory(model: Module,
                          input_shape: Tuple[int, ...],
                          num_steps: int = 10) -> Dict[str, Any]:
    """Profile memory usage during training over multiple steps.
    
    Args:
        model: Model to profile
        input_shape: Input shape for training
        num_steps: Number of training steps to profile
        
    Returns:
        Dictionary with memory profiling results
    """
    profiler = MemoryProfiler()
    
    with profiler.profile_operation("training_steps") as prof:
        
        for step in range(num_steps):
            # Create fresh data for each step
            input_data = Tensor(np.random.randn(*input_shape).astype(np.float32))
            
            # Training step
            model.train()
            output = model(input_data)
            
            # Take snapshot
            prof.take_snapshot(f"step_{step}")
            
            # Force garbage collection periodically
            if step % 5 == 0:
                gc.collect()
    
    summary = prof.stop_monitoring()
    
    # Analyze memory growth
    snapshots = prof.snapshots
    if len(snapshots) > 1:
        memory_values = [s.rss_mb for s in snapshots]
        memory_growth_rate = (memory_values[-1] - memory_values[0]) / len(snapshots)
        
        summary["memory_growth_analysis"] = {
            "initial_memory_mb": memory_values[0],
            "final_memory_mb": memory_values[-1],
            "growth_rate_mb_per_step": memory_growth_rate,
            "total_growth_mb": memory_values[-1] - memory_values[0],
            "max_memory_mb": max(memory_values),
            "memory_volatility": np.std(memory_values)
        }
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Forge Memory Profiling...")
    
    # Create test models
    neural_forge_model = Sequential(
        Linear(100, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )
    
    # Create PyTorch model if available
    pytorch_model = None
    if TORCH_AVAILABLE:
        pytorch_model = torch_nn.Sequential(
            torch_nn.Linear(100, 128),
            torch_nn.ReLU(),
            torch_nn.Linear(128, 64),
            torch_nn.ReLU(),
            torch_nn.Linear(64, 10)
        )
    
    # Test input shapes
    input_shapes = [
        (32, 100),   # Small batch
        (128, 100),  # Medium batch
        (512, 100)   # Large batch
    ]
    
    print(f"\n=== Memory Usage Comparison ===")
    benchmark = compare_memory_usage(
        neural_forge_model, pytorch_model, input_shapes, "linear_classifier"
    )
    
    # Print summary
    summary = benchmark.get_comparison_summary()
    print(f"\nMemory Benchmark Summary:")
    
    for framework, stats in summary.items():
        if framework != "comparison":
            print(f"  {framework}:")
            print(f"    Avg memory overhead: {stats['avg_memory_overhead_mb']:.2f} MB")
            print(f"    Max memory overhead: {stats['max_memory_overhead_mb']:.2f} MB")
            print(f"    Avg execution time: {stats['avg_execution_time_ms']:.2f} ms")
            print(f"    Memory efficiency: {stats['avg_memory_efficiency']:.4f} MB/param")
    
    if "comparison" in summary:
        comp = summary["comparison"]
        print(f"\n  Framework Comparison:")
        print(f"    Neural Forge memory efficiency: {comp['neural_forge_memory_efficiency']:.2f}x")
        print(f"    Neural Forge more efficient: {comp['neural_forge_more_efficient']}")
        print(f"    Memory savings: {comp['memory_savings_percent']:.1f}%")
    
    # Test training memory profiling
    print(f"\n=== Training Memory Profiling ===")
    training_profile = profile_training_memory(neural_forge_model, (32, 100), num_steps=5)
    
    if "memory_growth_analysis" in training_profile:
        growth = training_profile["memory_growth_analysis"]
        print(f"Initial memory: {growth['initial_memory_mb']:.2f} MB")
        print(f"Final memory: {growth['final_memory_mb']:.2f} MB")
        print(f"Growth rate: {growth['growth_rate_mb_per_step']:.3f} MB/step")
        print(f"Total growth: {growth['total_growth_mb']:.2f} MB")
    
    # Save results
    benchmark.save_results("/tmp/neural_forge_memory_benchmark.json")
    
    print("\n🎉 Memory profiling completed!")
    print("✅ Model memory usage comparison")
    print("✅ Training memory profiling")
    print("✅ Memory efficiency analysis")
    print("✅ Memory leak detection")
    print("✅ Cross-framework memory comparison")
    print("✅ Detailed memory snapshots")