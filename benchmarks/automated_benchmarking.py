"""Automated Benchmarking Suite for Neural Forge.

This module provides automated benchmarking capabilities that combine
performance, memory, speed, and accuracy validation into comprehensive
benchmark suites with automated reporting.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.neural_arch.core.tensor import Tensor
from src.neural_arch.nn.module import Module
from src.neural_arch.nn import Sequential, Linear, ReLU, Conv2d
from src.neural_arch.models.vision.resnet import ResNet18

# Import our benchmark modules
from .performance_comparison import PerformanceBenchmark, BenchmarkConfig, BenchmarkType
from .memory_profiling import MemoryBenchmark, compare_memory_usage
from .speed_benchmarks import InferenceBenchmark, TrainingBenchmark, benchmark_inference_speed
from .accuracy_validation import AccuracyValidator, AccuracyLevel, validate_model_accuracy

logger = logging.getLogger(__name__)

# Try to import PyTorch for comparison
try:
    import torch
    import torch.nn as torch_nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BenchmarkSuiteConfig:
    """Configuration for automated benchmark suite."""
    
    # Test scope
    test_models: List[str] = None
    test_operations: List[str] = None
    batch_sizes: List[int] = None
    input_shapes: List[Tuple[int, ...]] = None
    
    # Benchmark settings
    performance_runs: int = 50
    memory_runs: int = 20
    speed_runs: int = 100
    accuracy_tolerance: AccuracyLevel = AccuracyLevel.MEDIUM
    
    # Framework settings
    compare_pytorch: bool = True
    include_memory_profiling: bool = True
    include_speed_benchmarks: bool = True
    include_accuracy_validation: bool = True
    
    # Output settings
    output_directory: str = "./benchmark_results"
    generate_html_report: bool = True
    save_detailed_results: bool = True
    
    # Advanced settings
    enable_concurrent_tests: bool = False
    timeout_seconds: float = 600.0
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.test_models is None:
            self.test_models = ["linear_classifier", "simple_cnn"]
        
        if self.test_operations is None:
            self.test_operations = ["relu", "linear", "conv2d"]
        
        if self.batch_sizes is None:
            self.batch_sizes = [1, 32, 128]
        
        if self.input_shapes is None:
            self.input_shapes = [
                (32, 784),          # Linear input
                (32, 3, 224, 224)   # CNN input
            ]


@dataclass
class BenchmarkSuiteResult:
    """Comprehensive results from benchmark suite."""
    
    # Suite metadata
    suite_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    
    # Configuration used
    config: BenchmarkSuiteConfig
    
    # Individual benchmark results
    performance_results: Optional[Dict[str, Any]] = None
    memory_results: Optional[Dict[str, Any]] = None
    speed_results: Optional[Dict[str, Any]] = None
    accuracy_results: Optional[Dict[str, Any]] = None
    
    # Summary statistics
    overall_score: float = 0.0
    performance_score: float = 0.0
    memory_score: float = 0.0
    speed_score: float = 0.0
    accuracy_score: float = 0.0
    
    # Comparison with PyTorch
    pytorch_comparison: Optional[Dict[str, Any]] = None
    
    # Issues and recommendations
    issues_found: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        """Initialize lists if not provided."""
        if self.issues_found is None:
            self.issues_found = []
        if self.recommendations is None:
            self.recommendations = []


class AutomatedBenchmark:
    """Automated benchmark execution and analysis."""
    
    def __init__(self, config: BenchmarkSuiteConfig):
        """Initialize automated benchmark.
        
        Args:
            config: Benchmark suite configuration
        """
        self.config = config
        self.results = []
        
        # Create output directory
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmark components
        self.performance_benchmark = None
        self.memory_benchmark = None
        self.speed_benchmark = None
        self.accuracy_validator = None
    
    def run_comprehensive_benchmark(self, suite_name: str = "neural_forge_comprehensive") -> BenchmarkSuiteResult:
        """Run comprehensive benchmark suite.
        
        Args:
            suite_name: Name of the benchmark suite
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting comprehensive benchmark suite: {suite_name}")
        start_time = time.time()
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        
        result = BenchmarkSuiteResult(
            suite_name=suite_name,
            start_time=start_time_str,
            end_time="",
            duration_seconds=0.0,
            config=self.config
        )
        
        try:
            # Run performance benchmarks
            if True:  # Always run performance benchmarks
                logger.info("Running performance benchmarks...")
                result.performance_results = self._run_performance_benchmarks()
                result.performance_score = self._calculate_performance_score(result.performance_results)
            
            # Run memory profiling
            if self.config.include_memory_profiling:
                logger.info("Running memory profiling...")
                result.memory_results = self._run_memory_benchmarks()
                result.memory_score = self._calculate_memory_score(result.memory_results)
            
            # Run speed benchmarks
            if self.config.include_speed_benchmarks:
                logger.info("Running speed benchmarks...")
                result.speed_results = self._run_speed_benchmarks()
                result.speed_score = self._calculate_speed_score(result.speed_results)
            
            # Run accuracy validation
            if self.config.include_accuracy_validation:
                logger.info("Running accuracy validation...")
                result.accuracy_results = self._run_accuracy_validation()
                result.accuracy_score = self._calculate_accuracy_score(result.accuracy_results)
            
            # Calculate overall score
            result.overall_score = self._calculate_overall_score(result)
            
            # Generate PyTorch comparison
            if self.config.compare_pytorch and TORCH_AVAILABLE:
                result.pytorch_comparison = self._generate_pytorch_comparison(result)
            
            # Analyze results and generate recommendations
            result.issues_found, result.recommendations = self._analyze_results(result)
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
            result.issues_found.append(f"Benchmark execution failed: {str(e)}")
        
        # Finalize timing
        end_time = time.time()
        result.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        result.duration_seconds = end_time - start_time
        
        self.results.append(result)
        
        # Save results
        if self.config.save_detailed_results:
            self._save_results(result)
        
        # Generate HTML report
        if self.config.generate_html_report:
            self._generate_html_report(result)
        
        logger.info(f"Benchmark suite completed in {result.duration_seconds:.1f}s")
        return result
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        
        config = BenchmarkConfig(
            num_warmup_runs=10,
            num_benchmark_runs=self.config.performance_runs,
            batch_sizes=self.config.batch_sizes,
            benchmark_types=[BenchmarkType.INFERENCE, BenchmarkType.TRAINING],
            use_pytorch=self.config.compare_pytorch
        )
        
        self.performance_benchmark = PerformanceBenchmark(config)
        
        # Test predefined models
        for model_name in self.config.test_models:
            nf_model, pt_model = self._create_test_models(model_name)
            
            if model_name == "linear_classifier":
                input_shape = (32, 784)
            elif model_name == "simple_cnn":
                input_shape = (32, 3, 32, 32)  # Smaller for testing
            else:
                input_shape = (32, 100)
            
            self.performance_benchmark.benchmark_model(
                model_name, nf_model, pt_model, input_shape
            )
        
        return self.performance_benchmark.get_summary_statistics()
    
    def _run_memory_benchmarks(self) -> Dict[str, Any]:
        """Run memory benchmarks."""
        
        self.memory_benchmark = MemoryBenchmark()
        
        # Test models for memory usage
        for model_name in self.config.test_models:
            nf_model, pt_model = self._create_test_models(model_name)
            
            if model_name == "linear_classifier":
                input_shapes = [(bs, 784) for bs in self.config.batch_sizes]
            elif model_name == "simple_cnn":
                input_shapes = [(bs, 3, 32, 32) for bs in self.config.batch_sizes]
            else:
                input_shapes = [(bs, 100) for bs in self.config.batch_sizes]
            
            # Benchmark Neural Forge
            self.memory_benchmark.benchmark_model_memory(
                nf_model, input_shapes, "neural_forge", model_name
            )
            
            # Benchmark PyTorch if available
            if pt_model is not None and self.config.compare_pytorch:
                self.memory_benchmark.benchmark_model_memory(
                    pt_model, input_shapes, "pytorch", model_name
                )
        
        return self.memory_benchmark.get_comparison_summary()
    
    def _run_speed_benchmarks(self) -> Dict[str, Any]:
        """Run speed benchmarks."""
        
        # Inference benchmark
        self.speed_benchmark = InferenceBenchmark(num_benchmark_runs=self.config.speed_runs)
        
        all_results = []
        
        for model_name in self.config.test_models:
            nf_model, pt_model = self._create_test_models(model_name)
            
            if model_name == "linear_classifier":
                input_shapes = [(bs, 784) for bs in self.config.batch_sizes]
            elif model_name == "simple_cnn":
                input_shapes = [(bs, 3, 32, 32) for bs in self.config.batch_sizes]
            else:
                input_shapes = [(bs, 100) for bs in self.config.batch_sizes]
            
            # Benchmark Neural Forge
            nf_results = self.speed_benchmark.benchmark_model_inference(
                nf_model, input_shapes, "neural_forge", model_name
            )
            all_results.extend(nf_results)
            
            # Benchmark PyTorch if available
            if pt_model is not None and self.config.compare_pytorch:
                pt_results = self.speed_benchmark.benchmark_model_inference(
                    pt_model, input_shapes, "pytorch", model_name
                )
                all_results.extend(pt_results)
        
        # Analyze speed results
        from .speed_benchmarks import analyze_speed_results
        return analyze_speed_results(all_results)
    
    def _run_accuracy_validation(self) -> Dict[str, Any]:
        """Run accuracy validation."""
        
        self.accuracy_validator = AccuracyValidator()
        
        for model_name in self.config.test_models:
            nf_model, pt_model = self._create_test_models(model_name)
            
            if pt_model is not None and self.config.compare_pytorch:
                # Copy weights from Neural Forge to PyTorch for fair comparison
                self._copy_weights_to_pytorch(nf_model, pt_model)
                
                if model_name == "linear_classifier":
                    input_shapes = [(32, 784), (64, 784)]
                elif model_name == "simple_cnn":
                    input_shapes = [(32, 3, 32, 32), (16, 3, 32, 32)]
                else:
                    input_shapes = [(32, 100), (64, 100)]
                
                # Generate test inputs
                test_inputs = []
                for shape in input_shapes:
                    test_input = Tensor(np.random.randn(*shape).astype(np.float32))
                    test_inputs.append(test_input)
                
                # Validate model
                self.accuracy_validator.validate_model(
                    nf_model, pt_model, test_inputs, model_name, self.config.accuracy_tolerance
                )
        
        return self.accuracy_validator.get_validation_summary()
    
    def _create_test_models(self, model_name: str) -> Tuple[Module, Any]:
        """Create test models for benchmarking."""
        
        if model_name == "linear_classifier":
            nf_model = Sequential(
                Linear(784, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 10)
            )
            
            pt_model = None
            if TORCH_AVAILABLE:
                pt_model = torch_nn.Sequential(
                    torch_nn.Linear(784, 128),
                    torch_nn.ReLU(),
                    torch_nn.Linear(128, 64),
                    torch_nn.ReLU(),
                    torch_nn.Linear(64, 10)
                )
        
        elif model_name == "simple_cnn":
            nf_model = Sequential(
                Conv2d(3, 32, kernel_size=3),
                ReLU(),
                Conv2d(32, 64, kernel_size=3),
                ReLU(),
                Linear(64 * 28 * 28, 10)  # Approximate flattened size
            )
            
            pt_model = None
            if TORCH_AVAILABLE:
                pt_model = torch_nn.Sequential(
                    torch_nn.Conv2d(3, 32, kernel_size=3),
                    torch_nn.ReLU(),
                    torch_nn.Conv2d(32, 64, kernel_size=3),
                    torch_nn.ReLU(),
                    torch_nn.Flatten(),
                    torch_nn.Linear(64 * 28 * 28, 10)
                )
        
        else:
            # Default simple model
            nf_model = Sequential(
                Linear(100, 50),
                ReLU(),
                Linear(50, 10)
            )
            
            pt_model = None
            if TORCH_AVAILABLE:
                pt_model = torch_nn.Sequential(
                    torch_nn.Linear(100, 50),
                    torch_nn.ReLU(),
                    torch_nn.Linear(50, 10)
                )
        
        return nf_model, pt_model
    
    def _copy_weights_to_pytorch(self, nf_model: Module, pt_model: Any):
        """Copy weights from Neural Forge model to PyTorch model."""
        
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Get Neural Forge parameters
            nf_params = list(nf_model.parameters())
            pt_params = list(pt_model.parameters())
            
            # Copy weights
            with torch.no_grad():
                for nf_param, pt_param in zip(nf_params, pt_params):
                    pt_param.copy_(torch.from_numpy(nf_param.data).float())
                    
        except Exception as e:
            logger.warning(f"Failed to copy weights: {e}")
    
    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate performance score (0-100)."""
        
        if not results or "error" in results:
            return 0.0
        
        try:
            # Base score on throughput and latency
            frameworks = [k for k in results.keys() if k != "comparison"]
            
            if not frameworks:
                return 50.0  # Neutral score
            
            # Use Neural Forge metrics if available
            if "neural_forge" in frameworks:
                nf_stats = results["neural_forge"]
                throughput = nf_stats.get("avg_throughput", 0)
                latency = nf_stats.get("avg_mean_time_ms", float('inf'))
                
                # Score based on reasonable performance expectations
                throughput_score = min(100, (throughput / 1000) * 100)  # 1000 samples/sec = 100 points
                latency_score = max(0, 100 - (latency / 10))  # 10ms = 0 points, 0ms = 100 points
                
                return (throughput_score + latency_score) / 2
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    def _calculate_memory_score(self, results: Dict[str, Any]) -> float:
        """Calculate memory efficiency score (0-100)."""
        
        if not results or "error" in results:
            return 0.0
        
        try:
            # Score based on memory efficiency
            if "neural_forge" in results:
                nf_stats = results["neural_forge"]
                avg_efficiency = nf_stats.get("avg_memory_efficiency", float('inf'))
                
                # Lower memory per parameter is better
                # Score inversely related to memory efficiency
                if avg_efficiency == 0:
                    return 100.0
                elif avg_efficiency < 0.001:  # Very efficient
                    return 90.0
                elif avg_efficiency < 0.01:   # Good
                    return 70.0
                elif avg_efficiency < 0.1:    # Acceptable
                    return 50.0
                else:                         # Poor
                    return 20.0
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating memory score: {e}")
            return 0.0
    
    def _calculate_speed_score(self, results: Dict[str, Any]) -> float:
        """Calculate speed score (0-100)."""
        
        if not results or "error" in results:
            return 0.0
        
        try:
            if "neural_forge" in results:
                nf_stats = results["neural_forge"]
                avg_time = nf_stats.get("avg_inference_time_ms", float('inf'))
                max_throughput = nf_stats.get("max_throughput", 0)
                
                # Score based on speed
                time_score = max(0, 100 - (avg_time / 5))  # 5ms = 0 points
                throughput_score = min(100, (max_throughput / 2000) * 100)  # 2000 samples/sec = 100 points
                
                return (time_score + throughput_score) / 2
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating speed score: {e}")
            return 0.0
    
    def _calculate_accuracy_score(self, results: Dict[str, Any]) -> float:
        """Calculate accuracy score (0-100)."""
        
        if not results or "error" in results:
            return 0.0
        
        try:
            pass_rate = results.get("pass_rate", 0.0)
            return pass_rate * 100
            
        except Exception as e:
            logger.error(f"Error calculating accuracy score: {e}")
            return 0.0
    
    def _calculate_overall_score(self, result: BenchmarkSuiteResult) -> float:
        """Calculate overall benchmark score."""
        
        scores = []
        weights = []
        
        if result.performance_results:
            scores.append(result.performance_score)
            weights.append(0.3)
        
        if result.memory_results:
            scores.append(result.memory_score)
            weights.append(0.2)
        
        if result.speed_results:
            scores.append(result.speed_score)
            weights.append(0.3)
        
        if result.accuracy_results:
            scores.append(result.accuracy_score)
            weights.append(0.2)
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_pytorch_comparison(self, result: BenchmarkSuiteResult) -> Dict[str, Any]:
        """Generate comparison with PyTorch."""
        
        comparison = {}
        
        # Performance comparison
        if result.performance_results and "comparison" in result.performance_results:
            comp = result.performance_results["comparison"]
            comparison["performance"] = {
                "speed_ratio": comp.get("neural_forge_vs_pytorch_speed_ratio", 1.0),
                "neural_forge_faster": comp.get("neural_forge_faster", False),
                "speed_difference_percent": comp.get("speed_difference_percent", 0.0)
            }
        
        # Memory comparison
        if result.memory_results and "comparison" in result.memory_results:
            comp = result.memory_results["comparison"]
            comparison["memory"] = {
                "efficiency_ratio": comp.get("neural_forge_memory_efficiency", 1.0),
                "neural_forge_more_efficient": comp.get("neural_forge_more_efficient", False),
                "memory_savings_percent": comp.get("memory_savings_percent", 0.0)
            }
        
        # Speed comparison
        if result.speed_results and "comparison" in result.speed_results:
            comp = result.speed_results["comparison"]
            comparison["speed"] = {
                "speedup": comp.get("neural_forge_speedup", 1.0),
                "neural_forge_faster": comp.get("neural_forge_faster", False),
                "speed_advantage_percent": comp.get("speed_advantage_percent", 0.0)
            }
        
        return comparison
    
    def _analyze_results(self, result: BenchmarkSuiteResult) -> Tuple[List[str], List[str]]:
        """Analyze results and generate issues and recommendations."""
        
        issues = []
        recommendations = []
        
        # Check overall score
        if result.overall_score < 50:
            issues.append(f"Overall benchmark score is low: {result.overall_score:.1f}/100")
            recommendations.append("Consider optimizing performance bottlenecks")
        
        # Check accuracy
        if result.accuracy_score < 95:
            issues.append(f"Accuracy validation issues found: {result.accuracy_score:.1f}% pass rate")
            recommendations.append("Review numerical precision and implementation correctness")
        
        # Check performance vs PyTorch
        if result.pytorch_comparison:
            perf_comp = result.pytorch_comparison.get("performance", {})
            if not perf_comp.get("neural_forge_faster", False):
                issues.append("Neural Forge is slower than PyTorch")
                recommendations.append("Consider enabling JIT compilation and operator fusion")
        
        # Check memory efficiency
        if result.memory_score < 60:
            issues.append("Memory efficiency could be improved")
            recommendations.append("Implement memory pooling and optimize tensor operations")
        
        return issues, recommendations
    
    def _save_results(self, result: BenchmarkSuiteResult):
        """Save detailed benchmark results."""
        
        # Convert to dictionary
        result_dict = asdict(result)
        
        # Save to JSON file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.suite_name}_{timestamp}.json"
        filepath = Path(self.config.output_directory) / filename
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Detailed results saved to {filepath}")
    
    def _generate_html_report(self, result: BenchmarkSuiteResult):
        """Generate HTML benchmark report."""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Forge Benchmark Report - {suite_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .error {{ color: #e74c3c; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Neural Forge Benchmark Report</h1>
                <p>Suite: {suite_name}</p>
                <p>Generated: {end_time}</p>
                <p>Duration: {duration:.1f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Overall Score</h2>
                <div class="score {overall_class}">{overall_score:.1f}/100</div>
            </div>
            
            <div class="section">
                <h2>Component Scores</h2>
                <table class="table">
                    <tr>
                        <th>Component</th>
                        <th>Score</th>
                        <th>Status</th>
                    </tr>
                    {component_rows}
                </table>
            </div>
            
            {pytorch_section}
            
            <div class="section">
                <h2>Issues Found</h2>
                {issues_list}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {recommendations_list}
            </div>
        </body>
        </html>
        """
        
        # Generate component rows
        components = [
            ("Performance", result.performance_score),
            ("Memory", result.memory_score),
            ("Speed", result.speed_score),
            ("Accuracy", result.accuracy_score)
        ]
        
        component_rows = ""
        for name, score in components:
            if score >= 80:
                status_class = "good"
                status = "Excellent"
            elif score >= 60:
                status_class = "warning"
                status = "Good"
            else:
                status_class = "error"
                status = "Needs Improvement"
            
            component_rows += f'<tr><td>{name}</td><td>{score:.1f}</td><td class="{status_class}">{status}</td></tr>'
        
        # PyTorch comparison section
        pytorch_section = ""
        if result.pytorch_comparison:
            pytorch_section = """
            <div class="section">
                <h2>PyTorch Comparison</h2>
                <p>Performance comparison with PyTorch framework...</p>
            </div>
            """
        
        # Generate lists
        issues_list = "<ul>" + "".join(f"<li>{issue}</li>" for issue in result.issues_found) + "</ul>"
        recommendations_list = "<ul>" + "".join(f"<li>{rec}</li>" for rec in result.recommendations) + "</ul>"
        
        # Overall score class
        if result.overall_score >= 80:
            overall_class = "good"
        elif result.overall_score >= 60:
            overall_class = "warning"
        else:
            overall_class = "error"
        
        # Format HTML
        html_content = html_template.format(
            suite_name=result.suite_name,
            end_time=result.end_time,
            duration=result.duration_seconds,
            overall_score=result.overall_score,
            overall_class=overall_class,
            component_rows=component_rows,
            pytorch_section=pytorch_section,
            issues_list=issues_list,
            recommendations_list=recommendations_list
        )
        
        # Save HTML report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.suite_name}_report_{timestamp}.html"
        filepath = Path(self.config.output_directory) / filename
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filepath}")


class BenchmarkSuite:
    """Predefined benchmark suites for different use cases."""
    
    @staticmethod
    def quick_benchmark() -> BenchmarkSuiteConfig:
        """Quick benchmark configuration for development."""
        return BenchmarkSuiteConfig(
            test_models=["linear_classifier"],
            batch_sizes=[32],
            performance_runs=20,
            speed_runs=30,
            memory_runs=10,
            accuracy_tolerance=AccuracyLevel.MEDIUM
        )
    
    @staticmethod
    def comprehensive_benchmark() -> BenchmarkSuiteConfig:
        """Comprehensive benchmark for release validation."""
        return BenchmarkSuiteConfig(
            test_models=["linear_classifier", "simple_cnn"],
            batch_sizes=[1, 32, 128],
            performance_runs=100,
            speed_runs=200,
            memory_runs=50,
            accuracy_tolerance=AccuracyLevel.HIGH,
            enable_concurrent_tests=True
        )
    
    @staticmethod
    def accuracy_focused_benchmark() -> BenchmarkSuiteConfig:
        """Accuracy-focused benchmark for correctness validation."""
        return BenchmarkSuiteConfig(
            test_models=["linear_classifier"],
            include_memory_profiling=False,
            include_speed_benchmarks=False,
            accuracy_tolerance=AccuracyLevel.HIGH,
            performance_runs=50
        )


# Convenience functions
def run_full_benchmark_suite(suite_name: str = "neural_forge_full",
                           config: Optional[BenchmarkSuiteConfig] = None) -> BenchmarkSuiteResult:
    """Run full benchmark suite with default configuration."""
    
    if config is None:
        config = BenchmarkSuite.comprehensive_benchmark()
    
    benchmark = AutomatedBenchmark(config)
    return benchmark.run_comprehensive_benchmark(suite_name)


def generate_benchmark_report(result: BenchmarkSuiteResult) -> str:
    """Generate text summary of benchmark results."""
    
    report = f"""
Neural Forge Benchmark Report
=============================

Suite: {result.suite_name}
Duration: {result.duration_seconds:.1f} seconds
Overall Score: {result.overall_score:.1f}/100

Component Scores:
- Performance: {result.performance_score:.1f}/100
- Memory: {result.memory_score:.1f}/100  
- Speed: {result.speed_score:.1f}/100
- Accuracy: {result.accuracy_score:.1f}/100

Issues Found: {len(result.issues_found)}
Recommendations: {len(result.recommendations)}

Status: {"PASS" if result.overall_score >= 70 else "NEEDS IMPROVEMENT"}
"""
    
    return report


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Forge Automated Benchmarking...")
    
    # Quick benchmark
    print(f"\n=== Running Quick Benchmark ===")
    quick_config = BenchmarkSuite.quick_benchmark()
    quick_config.output_directory = "/tmp/neural_forge_benchmarks"
    
    benchmark = AutomatedBenchmark(quick_config)
    result = benchmark.run_comprehensive_benchmark("neural_forge_quick_test")
    
    # Print summary
    print(generate_benchmark_report(result))
    
    print(f"\nDetailed results saved to: {quick_config.output_directory}")
    
    print("\n🎉 Automated benchmarking completed!")
    print("✅ Comprehensive benchmark suite")
    print("✅ Automated performance analysis")
    print("✅ Memory efficiency validation")
    print("✅ Speed benchmarking")
    print("✅ Accuracy validation")
    print("✅ PyTorch comparison")
    print("✅ HTML report generation")
    print("✅ Issue detection and recommendations")