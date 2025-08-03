"""Neural Forge Benchmarking Suite.

This module provides comprehensive performance benchmarking tools to compare
Neural Forge against PyTorch across various models, operations, and scenarios.
"""

__version__ = "1.0.0"

from .performance_comparison import (
    BenchmarkConfig,
    BenchmarkResult,
    PerformanceBenchmark,
    compare_frameworks,
    benchmark_models,
    benchmark_operations
)

from .memory_profiling import (
    MemoryProfiler,
    MemoryBenchmark,
    compare_memory_usage,
    profile_training_memory
)

from .speed_benchmarks import (
    SpeedBenchmark,
    InferenceBenchmark,
    TrainingBenchmark,
    benchmark_inference_speed,
    benchmark_training_speed
)

from .accuracy_validation import (
    AccuracyValidator,
    ValidationResult,
    validate_model_accuracy,
    compare_numerical_precision
)

from .automated_benchmarking import (
    AutomatedBenchmark,
    BenchmarkSuite,
    run_full_benchmark_suite,
    generate_benchmark_report
)

__all__ = [
    # Core benchmarking
    "BenchmarkConfig",
    "BenchmarkResult", 
    "PerformanceBenchmark",
    "compare_frameworks",
    "benchmark_models",
    "benchmark_operations",
    
    # Memory profiling
    "MemoryProfiler",
    "MemoryBenchmark",
    "compare_memory_usage",
    "profile_training_memory",
    
    # Speed benchmarking
    "SpeedBenchmark",
    "InferenceBenchmark", 
    "TrainingBenchmark",
    "benchmark_inference_speed",
    "benchmark_training_speed",
    
    # Accuracy validation
    "AccuracyValidator",
    "ValidationResult",
    "validate_model_accuracy",
    "compare_numerical_precision",
    
    # Automated suites
    "AutomatedBenchmark",
    "BenchmarkSuite",
    "run_full_benchmark_suite",
    "generate_benchmark_report"
]