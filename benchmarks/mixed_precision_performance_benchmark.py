"""Performance benchmark comparing FP32 vs FP16 mixed precision training.

This benchmark provides comprehensive evaluation of:
- Training speed improvements with mixed precision
- Memory usage reduction
- Numerical accuracy preservation
- Convergence behavior comparison
- Scaling behavior with model size
"""

import json
import logging
import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import Neural Forge components
from src.neural_arch.core.base import Module, Parameter
from src.neural_arch.core.tensor import Tensor
from src.neural_arch.nn.linear import Linear
from src.neural_arch.nn.activation import ReLU
from src.neural_arch.optim.adam import Adam
from src.neural_arch.optimization.mixed_precision import (
    AutocastPolicy,
    PrecisionConfig,
    MixedPrecisionManager,
    autocast,
    create_precision_config,
    get_recommended_precision_config,
)
from src.neural_arch.optimization.amp_optimizer import create_amp_adam

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    mode: str  # "fp32" or "fp16"
    model_size: str
    batch_size: int
    sequence_length: int
    
    # Performance metrics
    avg_step_time: float
    total_training_time: float
    steps_per_second: float
    
    # Memory metrics
    peak_memory_mb: float
    memory_reduction_percent: float
    
    # Accuracy metrics
    final_loss: float
    loss_convergence_steps: int
    numerical_stability_score: float
    
    # Mixed precision specific
    overflow_rate: float
    scaling_adjustments: int
    
    # Additional metadata
    total_steps: int
    successful_steps: int
    errors: List[str]


class BenchmarkModel(Module):
    """Configurable model for benchmarking different sizes."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        self.layers = []
        
        # Input layer
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layer = Linear(prev_size, hidden_size)
            activation = ReLU()
            
            # Store as attributes for proper parameter registration
            setattr(self, f"layer_{i}", layer)
            setattr(self, f"activation_{i}", activation)
            
            self.layers.append((layer, activation))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = Linear(prev_size, output_size)
        
        logger.info(f"Created benchmark model with {self.count_parameters()} parameters")
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model."""
        current = x
        
        for layer, activation in self.layers:
            current = layer(current)
            current = activation(current)
        
        output = self.output_layer(current)
        return output
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total


class MixedPrecisionBenchmark:
    """Comprehensive mixed precision training benchmark."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
        
        # Benchmark configurations
        self.model_configs = {
            "small": {"hidden_sizes": [128, 64], "input_size": 100, "output_size": 10},
            "medium": {"hidden_sizes": [512, 256, 128], "input_size": 200, "output_size": 20},
            "large": {"hidden_sizes": [1024, 512, 256, 128], "input_size": 500, "output_size": 50},
            "xlarge": {"hidden_sizes": [2048, 1024, 512, 256], "input_size": 1000, "output_size": 100},
        }
        
        self.training_configs = {
            "small": {"batch_size": 32, "sequence_length": 128, "num_steps": 1000},
            "medium": {"batch_size": 16, "sequence_length": 256, "num_steps": 500},
            "large": {"batch_size": 8, "sequence_length": 512, "num_steps": 200},
            "xlarge": {"batch_size": 4, "sequence_length": 1024, "num_steps": 100},
        }
    
    def create_model(self, size: str) -> BenchmarkModel:
        """Create a model of the specified size."""
        config = self.model_configs[size]
        return BenchmarkModel(**config)
    
    def generate_synthetic_data(self, batch_size: int, sequence_length: int, 
                               input_size: int) -> Tuple[Tensor, Tensor]:
        """Generate synthetic training data."""
        # Input data
        x_data = np.random.randn(batch_size, input_size).astype(np.float32)
        x = Tensor(x_data, requires_grad=False)
        
        # Target data (regression task)
        y_data = np.random.randn(batch_size, self.model_configs["small"]["output_size"]).astype(np.float32)
        y = Tensor(y_data, requires_grad=False)
        
        return x, y
    
    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute mean squared error loss."""
        diff = predictions.data - targets.data
        loss_data = np.mean(diff ** 2)
        return Tensor(np.array([loss_data]), requires_grad=True)
    
    def benchmark_fp32_training(self, model_size: str, num_steps: int = 1000) -> BenchmarkResult:
        """Benchmark FP32 training."""
        logger.info(f"Benchmarking FP32 training for {model_size} model")
        
        # Create model and optimizer
        model = self.create_model(model_size)
        optimizer = Adam(model.parameters(), lr=0.001)
        
        # Training configuration
        config = self.training_configs[model_size]
        batch_size = config["batch_size"]
        sequence_length = config["sequence_length"]
        input_size = self.model_configs[model_size]["input_size"]
        
        # Start memory tracking
        tracemalloc.start()
        
        # Training loop
        step_times = []
        losses = []
        errors = []
        
        start_time = time.time()
        
        for step in range(num_steps):
            step_start = time.time()
            
            try:
                # Generate data
                x, y = self.generate_synthetic_data(batch_size, sequence_length, input_size)
                
                # Forward pass
                predictions = model(x)
                loss = self.compute_loss(predictions, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record metrics
                step_time = time.time() - step_start
                step_times.append(step_time)
                losses.append(loss.data[0])
                
                if step % 100 == 0:
                    logger.info(f"FP32 Step {step}/{num_steps}, Loss: {loss.data[0]:.6f}, "
                               f"Time: {step_time:.4f}s")
                
            except Exception as e:
                errors.append(f"Step {step}: {str(e)}")
                logger.error(f"Error at step {step}: {e}")
        
        total_time = time.time() - start_time
        
        # Memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak / 1024 / 1024
        
        # Compute metrics
        avg_step_time = np.mean(step_times) if step_times else 0
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        final_loss = losses[-1] if losses else float('inf')
        
        # Convergence analysis
        loss_convergence_steps = self._analyze_convergence(losses)
        numerical_stability_score = self._compute_stability_score(losses)
        
        return BenchmarkResult(
            mode="fp32",
            model_size=model_size,
            batch_size=batch_size,
            sequence_length=sequence_length,
            avg_step_time=avg_step_time,
            total_training_time=total_time,
            steps_per_second=steps_per_second,
            peak_memory_mb=peak_memory_mb,
            memory_reduction_percent=0.0,  # Baseline
            final_loss=final_loss,
            loss_convergence_steps=loss_convergence_steps,
            numerical_stability_score=numerical_stability_score,
            overflow_rate=0.0,
            scaling_adjustments=0,
            total_steps=num_steps,
            successful_steps=len(step_times),
            errors=errors,
        )
    
    def benchmark_fp16_training(self, model_size: str, num_steps: int = 1000,
                               policy: AutocastPolicy = AutocastPolicy.SELECTIVE) -> BenchmarkResult:
        """Benchmark FP16 mixed precision training."""
        logger.info(f"Benchmarking FP16 training for {model_size} model with {policy.value} policy")
        
        # Create model and optimizer
        model = self.create_model(model_size)
        
        # Create AMP-aware optimizer
        try:
            optimizer = create_amp_adam(model.parameters(), lr=0.001)
            amp_enabled = True
        except Exception as e:
            logger.warning(f"Could not create AMP optimizer: {e}")
            optimizer = Adam(model.parameters(), lr=0.001)
            amp_enabled = False
        
        # Mixed precision configuration
        precision_config = get_recommended_precision_config(
            model_type="transformer",  # Generic choice
            model_size=model_size.lower(),
            training_stability="normal"
        )
        precision_config.autocast_config.policy = policy
        
        mp_manager = MixedPrecisionManager(precision_config)
        
        # Training configuration
        config = self.training_configs[model_size]
        batch_size = config["batch_size"]
        sequence_length = config["sequence_length"]
        input_size = self.model_configs[model_size]["input_size"]
        
        # Start memory tracking
        tracemalloc.start()
        
        # Training loop
        step_times = []
        losses = []
        errors = []
        
        start_time = time.time()
        
        for step in range(num_steps):
            step_start = time.time()
            
            try:
                # Generate data
                x, y = self.generate_synthetic_data(batch_size, sequence_length, input_size)
                
                # Mixed precision forward pass
                with mp_manager.autocast():
                    predictions = model(x)
                    loss = self.compute_loss(predictions, y)
                
                # Mixed precision backward pass
                if amp_enabled and hasattr(optimizer, 'backward'):
                    optimizer.zero_grad()
                    optimizer.backward(loss)
                    success = optimizer.step()
                elif hasattr(mp_manager, 'backward_and_step'):
                    success = mp_manager.backward_and_step(loss, optimizer, model)
                else:
                    # Fallback to manual mixed precision
                    optimizer.zero_grad()
                    scaled_loss = mp_manager.scale_loss(loss)
                    scaled_loss.backward()
                    success = mp_manager.scaler.step(optimizer)
                    mp_manager.scaler.update()
                
                # Record metrics only for successful steps
                if success:
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                    losses.append(loss.data[0])
                
                if step % 100 == 0:
                    scale = mp_manager.scaler.get_scale() if hasattr(mp_manager.scaler, 'get_scale') else 1.0
                    logger.info(f"FP16 Step {step}/{num_steps}, Loss: {loss.data[0]:.6f}, "
                               f"Time: {time.time() - step_start:.4f}s, Scale: {scale:.0f}")
                
            except Exception as e:
                errors.append(f"Step {step}: {str(e)}")
                logger.error(f"Error at step {step}: {e}")
        
        total_time = time.time() - start_time
        
        # Memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak / 1024 / 1024
        
        # Get mixed precision statistics
        if hasattr(mp_manager, 'get_statistics'):
            mp_stats = mp_manager.get_statistics()
            overflow_rate = 1.0 - mp_stats.get('success_rate', 1.0)
            scaling_adjustments = mp_stats.get('total_steps', 0) - mp_stats.get('successful_steps', 0)
        elif hasattr(optimizer, 'get_statistics'):
            opt_stats = optimizer.get_statistics()
            overflow_rate = opt_stats.get('skip_rate', 0.0)
            scaling_adjustments = opt_stats.get('skipped_steps', 0)
        else:
            overflow_rate = 0.0
            scaling_adjustments = 0
        
        # Compute metrics
        avg_step_time = np.mean(step_times) if step_times else 0
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        final_loss = losses[-1] if losses else float('inf')
        
        # Convergence analysis
        loss_convergence_steps = self._analyze_convergence(losses)
        numerical_stability_score = self._compute_stability_score(losses)
        
        return BenchmarkResult(
            mode="fp16",
            model_size=model_size,
            batch_size=batch_size,
            sequence_length=sequence_length,
            avg_step_time=avg_step_time,
            total_training_time=total_time,
            steps_per_second=steps_per_second,
            peak_memory_mb=peak_memory_mb,
            memory_reduction_percent=0.0,  # Will be computed later
            final_loss=final_loss,
            loss_convergence_steps=loss_convergence_steps,
            numerical_stability_score=numerical_stability_score,
            overflow_rate=overflow_rate,
            scaling_adjustments=scaling_adjustments,
            total_steps=num_steps,
            successful_steps=len(step_times),
            errors=errors,
        )
    
    def _analyze_convergence(self, losses: List[float]) -> int:
        """Analyze convergence behavior."""
        if len(losses) < 10:
            return len(losses)
        
        # Find where loss stabilizes (derivative becomes small)
        window_size = min(50, len(losses) // 4)
        
        for i in range(window_size, len(losses)):
            recent_losses = losses[i-window_size:i]
            if len(recent_losses) > 1:
                slope = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
                if abs(slope) < 1e-6:  # Converged
                    return i
        
        return len(losses)
    
    def _compute_stability_score(self, losses: List[float]) -> float:
        """Compute numerical stability score (0-1, higher is better)."""
        if len(losses) < 2:
            return 1.0
        
        # Check for NaN/Inf
        finite_losses = [l for l in losses if np.isfinite(l)]
        if len(finite_losses) != len(losses):
            return 0.0
        
        # Compute relative stability
        loss_std = np.std(finite_losses)
        loss_mean = np.mean(finite_losses)
        
        if loss_mean == 0:
            return 1.0 if loss_std == 0 else 0.0
        
        coefficient_of_variation = loss_std / abs(loss_mean)
        
        # Convert to 0-1 score (lower CV is better)
        stability_score = 1.0 / (1.0 + coefficient_of_variation)
        return stability_score
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across all configurations."""
        logger.info("Starting comprehensive mixed precision benchmark")
        
        results = {"fp32": [], "fp16": []}
        
        for model_size in ["small", "medium", "large"]:  # Skip xlarge for faster testing
            num_steps = self.training_configs[model_size]["num_steps"]
            
            # Benchmark FP32
            try:
                fp32_result = self.benchmark_fp32_training(model_size, num_steps)
                results["fp32"].append(fp32_result)
                logger.info(f"FP32 {model_size}: {fp32_result.steps_per_second:.2f} steps/sec, "
                           f"{fp32_result.peak_memory_mb:.1f} MB")
            except Exception as e:
                logger.error(f"FP32 benchmark failed for {model_size}: {e}")
            
            # Benchmark FP16 with different policies
            for policy in [AutocastPolicy.SELECTIVE, AutocastPolicy.AGGRESSIVE]:
                try:
                    fp16_result = self.benchmark_fp16_training(model_size, num_steps, policy)
                    fp16_result.mode = f"fp16_{policy.value}"
                    
                    # Compute memory reduction
                    if results["fp32"]:
                        fp32_memory = results["fp32"][-1].peak_memory_mb
                        if fp32_memory > 0:
                            fp16_result.memory_reduction_percent = (
                                (fp32_memory - fp16_result.peak_memory_mb) / fp32_memory * 100
                            )
                    
                    results["fp16"].append(fp16_result)
                    logger.info(f"FP16 {model_size} ({policy.value}): "
                               f"{fp16_result.steps_per_second:.2f} steps/sec, "
                               f"{fp16_result.peak_memory_mb:.1f} MB, "
                               f"{fp16_result.memory_reduction_percent:.1f}% memory reduction")
                except Exception as e:
                    logger.error(f"FP16 benchmark failed for {model_size} with {policy.value}: {e}")
        
        self.results = results["fp32"] + results["fp16"]
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        if not self.results:
            logger.warning("No benchmark results available")
            return {}
        
        report = {
            "summary": self._generate_summary(),
            "detailed_results": [self._result_to_dict(r) for r in self.results],
            "performance_analysis": self._analyze_performance(),
            "memory_analysis": self._analyze_memory(),
            "stability_analysis": self._analyze_stability(),
            "recommendations": self._generate_recommendations(),
        }
        
        return report
    
    def _result_to_dict(self, result: BenchmarkResult) -> Dict:
        """Convert result to dictionary."""
        return {
            "mode": result.mode,
            "model_size": result.model_size,
            "batch_size": result.batch_size,
            "avg_step_time": result.avg_step_time,
            "steps_per_second": result.steps_per_second,
            "peak_memory_mb": result.peak_memory_mb,
            "memory_reduction_percent": result.memory_reduction_percent,
            "final_loss": result.final_loss,
            "numerical_stability_score": result.numerical_stability_score,
            "overflow_rate": result.overflow_rate,
            "successful_steps": result.successful_steps,
            "total_steps": result.total_steps,
            "errors": len(result.errors),
        }
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        fp32_results = [r for r in self.results if r.mode == "fp32"]
        fp16_results = [r for r in self.results if r.mode.startswith("fp16")]
        
        summary = {
            "total_benchmarks": len(self.results),
            "fp32_benchmarks": len(fp32_results),
            "fp16_benchmarks": len(fp16_results),
        }
        
        if fp32_results and fp16_results:
            # Performance improvements
            fp32_avg_speed = np.mean([r.steps_per_second for r in fp32_results])
            fp16_avg_speed = np.mean([r.steps_per_second for r in fp16_results])
            
            summary.update({
                "avg_speedup": fp16_avg_speed / fp32_avg_speed if fp32_avg_speed > 0 else 0,
                "avg_memory_reduction": np.mean([r.memory_reduction_percent for r in fp16_results]),
                "avg_overflow_rate": np.mean([r.overflow_rate for r in fp16_results]),
            })
        
        return summary
    
    def _analyze_performance(self) -> Dict:
        """Analyze performance characteristics."""
        analysis = {}
        
        for model_size in ["small", "medium", "large"]:
            size_results = [r for r in self.results if r.model_size == model_size]
            if not size_results:
                continue
            
            fp32_results = [r for r in size_results if r.mode == "fp32"]
            fp16_results = [r for r in size_results if r.mode.startswith("fp16")]
            
            if fp32_results and fp16_results:
                fp32_speed = fp32_results[0].steps_per_second
                fp16_speeds = [r.steps_per_second for r in fp16_results]
                
                analysis[model_size] = {
                    "fp32_speed": fp32_speed,
                    "fp16_speeds": fp16_speeds,
                    "best_speedup": max(fp16_speeds) / fp32_speed if fp32_speed > 0 else 0,
                    "avg_speedup": np.mean(fp16_speeds) / fp32_speed if fp32_speed > 0 else 0,
                }
        
        return analysis
    
    def _analyze_memory(self) -> Dict:
        """Analyze memory usage patterns."""
        analysis = {}
        
        for model_size in ["small", "medium", "large"]:
            size_results = [r for r in self.results if r.model_size == model_size]
            if not size_results:
                continue
            
            fp32_results = [r for r in size_results if r.mode == "fp32"]
            fp16_results = [r for r in size_results if r.mode.startswith("fp16")]
            
            if fp32_results and fp16_results:
                fp32_memory = fp32_results[0].peak_memory_mb
                fp16_memories = [r.peak_memory_mb for r in fp16_results]
                
                analysis[model_size] = {
                    "fp32_memory_mb": fp32_memory,
                    "fp16_memory_mb": fp16_memories,
                    "memory_reduction_percent": [r.memory_reduction_percent for r in fp16_results],
                    "avg_reduction": np.mean([r.memory_reduction_percent for r in fp16_results]),
                }
        
        return analysis
    
    def _analyze_stability(self) -> Dict:
        """Analyze numerical stability."""
        fp16_results = [r for r in self.results if r.mode.startswith("fp16")]
        
        if not fp16_results:
            return {}
        
        return {
            "avg_stability_score": np.mean([r.numerical_stability_score for r in fp16_results]),
            "avg_overflow_rate": np.mean([r.overflow_rate for r in fp16_results]),
            "max_overflow_rate": max([r.overflow_rate for r in fp16_results]),
            "stable_configurations": [
                r.mode for r in fp16_results 
                if r.numerical_stability_score > 0.95 and r.overflow_rate < 0.01
            ],
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        fp16_results = [r for r in self.results if r.mode.startswith("fp16")]
        
        if not fp16_results:
            recommendations.append("Could not run FP16 benchmarks - check implementation")
            return recommendations
        
        # Speed recommendations
        best_speed_result = max(fp16_results, key=lambda r: r.steps_per_second)
        recommendations.append(
            f"Best performance: {best_speed_result.mode} on {best_speed_result.model_size} "
            f"({best_speed_result.steps_per_second:.2f} steps/sec)"
        )
        
        # Memory recommendations
        best_memory_result = max(fp16_results, key=lambda r: r.memory_reduction_percent)
        recommendations.append(
            f"Best memory efficiency: {best_memory_result.mode} on {best_memory_result.model_size} "
            f"({best_memory_result.memory_reduction_percent:.1f}% reduction)"
        )
        
        # Stability recommendations
        stable_results = [r for r in fp16_results if r.overflow_rate < 0.05]
        if stable_results:
            recommendations.append(
                f"Most stable configurations: {[r.mode for r in stable_results]}"
            )
        else:
            recommendations.append("High overflow rates detected - consider more conservative scaling")
        
        return recommendations
    
    def save_results(self, filename: str = "mixed_precision_benchmark.json"):
        """Save benchmark results to file."""
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filename}")


def main():
    """Run the mixed precision benchmark."""
    print("🚀 Neural Forge Mixed Precision Performance Benchmark")
    print("=" * 60)
    
    # Create benchmark instance
    benchmark = MixedPrecisionBenchmark()
    
    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Generate and save report
        report = benchmark.generate_report()
        benchmark.save_results("mixed_precision_benchmark_results.json")
        
        # Print summary
        print("\n📊 Benchmark Summary:")
        print(f"Total benchmarks run: {report['summary']['total_benchmarks']}")
        
        if 'avg_speedup' in report['summary']:
            print(f"Average speedup: {report['summary']['avg_speedup']:.2f}x")
            print(f"Average memory reduction: {report['summary']['avg_memory_reduction']:.1f}%")
            print(f"Average overflow rate: {report['summary']['avg_overflow_rate']:.2f}%")
        
        print("\n🎯 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\n✅ Detailed results saved to: mixed_precision_benchmark_results.json")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()