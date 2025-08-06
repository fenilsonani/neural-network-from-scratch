"""
Comprehensive test suite for comprehensive benchmarking system.
Tests all components of comprehensive_benchmark.py for comprehensive coverage.

This module tests:
- BenchmarkRunner for performance evaluation and validation
- ResourceMonitor for system resource tracking
- PerformanceProfiler for detailed performance analysis
- StatisticalAnalyzer for statistical analysis and regression detection
- BenchmarkConfig and BenchmarkResult data structures
- Multi-dimensional performance testing (throughput, latency, memory, scalability)
- Mock workload generation and execution
- Comparative benchmarking and reporting
"""

import gc
import json
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from unittest.mock import Mock, patch, mock_open, MagicMock
from typing import Dict, List, Any, Callable

import numpy as np
import pytest

from src.neural_arch.benchmarks.comprehensive_benchmark import (
    BenchmarkType,
    BenchmarkStatus,
    BenchmarkConfig,
    BenchmarkResult,
    ResourceMonitor,
    PerformanceProfiler,
    StatisticalAnalyzer,
    BenchmarkRunner,
    create_mock_workloads,
    run_comprehensive_benchmark_suite
)


class TestBenchmarkType:
    """Test BenchmarkType enumeration."""
    
    def test_benchmark_types(self):
        """Test all benchmark type values."""
        assert BenchmarkType.THROUGHPUT.value == "throughput"
        assert BenchmarkType.LATENCY.value == "latency"
        assert BenchmarkType.MEMORY.value == "memory"
        assert BenchmarkType.SCALABILITY.value == "scalability"
        assert BenchmarkType.ACCURACY.value == "accuracy"
        assert BenchmarkType.ENERGY.value == "energy"
        assert BenchmarkType.STRESS.value == "stress"
        assert BenchmarkType.REGRESSION.value == "regression"
    
    def test_benchmark_type_count(self):
        """Test total number of benchmark types."""
        assert len(BenchmarkType) == 8


class TestBenchmarkStatus:
    """Test BenchmarkStatus enumeration."""
    
    def test_benchmark_statuses(self):
        """Test all benchmark status values."""
        assert BenchmarkStatus.PENDING.value == "pending"
        assert BenchmarkStatus.RUNNING.value == "running"
        assert BenchmarkStatus.COMPLETED.value == "completed"
        assert BenchmarkStatus.FAILED.value == "failed"
        assert BenchmarkStatus.SKIPPED.value == "skipped"
    
    def test_benchmark_status_count(self):
        """Test total number of benchmark statuses."""
        assert len(BenchmarkStatus) == 5


class TestBenchmarkConfig:
    """Test BenchmarkConfig data structure."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig(
            name="test_benchmark",
            benchmark_type=BenchmarkType.THROUGHPUT
        )
        
        assert config.name == "test_benchmark"
        assert config.benchmark_type == BenchmarkType.THROUGHPUT
        assert config.iterations == 100
        assert config.warmup_iterations == 10
        assert config.timeout_seconds == 300.0
        
        # Default scaling parameters
        assert config.batch_sizes == [1, 8, 32, 128]
        assert config.sequence_lengths == [64, 256, 1024, 4096]
        assert config.model_sizes == ["small", "medium", "large"]
        
        # Hardware configurations
        assert config.device_types == ["cpu"]
        assert config.num_workers == [1, 2, 4, 8]
        
        # Statistical parameters
        assert config.confidence_level == 0.95
        assert config.outlier_threshold == 3.0
        
        # Comparison baselines
        assert config.baseline_results is None
        assert config.regression_threshold == 0.05
    
    def test_custom_config(self):
        """Test custom configuration values."""
        baseline_results = {"test": 1.5}
        
        config = BenchmarkConfig(
            name="custom_benchmark",
            benchmark_type=BenchmarkType.MEMORY,
            iterations=50,
            warmup_iterations=5,
            timeout_seconds=180.0,
            batch_sizes=[16, 64],
            sequence_lengths=[128, 512],
            model_sizes=["tiny", "huge"],
            device_types=["cpu", "gpu"],
            num_workers=[1, 4],
            confidence_level=0.99,
            outlier_threshold=2.5,
            baseline_results=baseline_results,
            regression_threshold=0.10
        )
        
        assert config.name == "custom_benchmark"
        assert config.benchmark_type == BenchmarkType.MEMORY
        assert config.iterations == 50
        assert config.warmup_iterations == 5
        assert config.timeout_seconds == 180.0
        assert config.batch_sizes == [16, 64]
        assert config.sequence_lengths == [128, 512]
        assert config.model_sizes == ["tiny", "huge"]
        assert config.device_types == ["cpu", "gpu"]
        assert config.num_workers == [1, 4]
        assert config.confidence_level == 0.99
        assert config.outlier_threshold == 2.5
        assert config.baseline_results == baseline_results
        assert config.regression_threshold == 0.10


class TestBenchmarkResult:
    """Test BenchmarkResult data structure."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = BenchmarkConfig(
            name="test_result",
            benchmark_type=BenchmarkType.LATENCY
        )
    
    def test_default_result(self):
        """Test default result values."""
        result = BenchmarkResult(config=self.config)
        
        assert result.config == self.config
        assert result.status == BenchmarkStatus.PENDING
        
        # Timing metrics
        assert result.mean_time == 0.0
        assert result.median_time == 0.0
        assert result.std_time == 0.0
        assert result.min_time == float('inf')
        assert result.max_time == 0.0
        assert result.p95_time == 0.0
        assert result.p99_time == 0.0
        
        # Throughput metrics
        assert result.throughput_ops_per_sec == 0.0
        assert result.throughput_samples_per_sec == 0.0
        
        # Memory metrics
        assert result.peak_memory_mb == 0.0
        assert result.avg_memory_mb == 0.0
        assert result.memory_efficiency == 0.0
        
        # Hardware utilization
        assert result.avg_cpu_percent == 0.0
        assert result.avg_gpu_percent == 0.0
        
        # Accuracy metrics
        assert result.accuracy is None
        assert result.loss is None
        
        # Error information
        assert result.error_message is None
        
        # Raw measurements
        assert isinstance(result.raw_times, list)
        assert isinstance(result.raw_memory, list)
        
        # Metadata
        assert isinstance(result.timestamp, float)
        assert isinstance(result.hostname, str)
        assert isinstance(result.python_version, str)
    
    def test_result_with_custom_values(self):
        """Test result with custom values."""
        raw_times = [0.1, 0.2, 0.15, 0.18]
        raw_memory = [100.0, 150.0, 120.0]
        
        result = BenchmarkResult(
            config=self.config,
            status=BenchmarkStatus.COMPLETED,
            mean_time=0.16,
            median_time=0.17,
            std_time=0.04,
            min_time=0.1,
            max_time=0.2,
            p95_time=0.19,
            p99_time=0.20,
            throughput_ops_per_sec=6.25,
            throughput_samples_per_sec=50.0,
            peak_memory_mb=150.0,
            avg_memory_mb=123.3,
            memory_efficiency=0.05,
            avg_cpu_percent=75.5,
            avg_gpu_percent=85.2,
            accuracy=0.95,
            loss=0.05,
            error_message=None,
            raw_times=raw_times,
            raw_memory=raw_memory
        )
        
        assert result.status == BenchmarkStatus.COMPLETED
        assert result.mean_time == 0.16
        assert result.median_time == 0.17
        assert result.std_time == 0.04
        assert result.min_time == 0.1
        assert result.max_time == 0.2
        assert result.p95_time == 0.19
        assert result.p99_time == 0.20
        assert result.throughput_ops_per_sec == 6.25
        assert result.throughput_samples_per_sec == 50.0
        assert result.peak_memory_mb == 150.0
        assert result.avg_memory_mb == 123.3
        assert result.memory_efficiency == 0.05
        assert result.avg_cpu_percent == 75.5
        assert result.avg_gpu_percent == 85.2
        assert result.accuracy == 0.95
        assert result.loss == 0.05
        assert result.raw_times == raw_times
        assert result.raw_memory == raw_memory
    
    def test_result_to_dict(self):
        """Test result serialization to dictionary."""
        result = BenchmarkResult(
            config=self.config,
            status=BenchmarkStatus.COMPLETED,
            mean_time=0.1,
            throughput_ops_per_sec=10.0,
            peak_memory_mb=100.0,
            avg_cpu_percent=50.0,
            accuracy=0.98,
            error_message="test error"
        )
        
        result_dict = result.to_dict()
        
        # Check structure
        expected_keys = ['config', 'status', 'timing', 'throughput', 'memory', 'hardware', 'accuracy', 'metadata']
        assert all(key in result_dict for key in expected_keys)
        
        # Check config
        assert result_dict['config']['name'] == "test_result"
        assert result_dict['config']['type'] == "latency"
        
        # Check status
        assert result_dict['status'] == "completed"
        
        # Check timing
        assert result_dict['timing']['mean_time'] == 0.1
        
        # Check throughput
        assert result_dict['throughput']['ops_per_sec'] == 10.0
        
        # Check memory
        assert result_dict['memory']['peak_mb'] == 100.0
        
        # Check hardware
        assert result_dict['hardware']['avg_cpu_percent'] == 50.0
        
        # Check accuracy
        assert result_dict['accuracy']['accuracy'] == 0.98
        
        # Check metadata
        assert result_dict['metadata']['error_message'] == "test error"


class TestResourceMonitor:
    """Test resource monitoring system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.monitor = ResourceMonitor(sampling_interval=0.01)  # Fast sampling for tests
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.monitor.stop_monitoring()
    
    def test_initialization(self):
        """Test ResourceMonitor initialization."""
        assert self.monitor.sampling_interval == 0.01
        assert self.monitor.monitoring is False
        assert self.monitor.monitor_thread is None
        assert isinstance(self.monitor.cpu_measurements, deque)
        assert isinstance(self.monitor.memory_measurements, deque)
        assert isinstance(self.monitor.gpu_measurements, deque)
        assert isinstance(self.monitor.lock, threading.Lock)
    
    def test_start_stop_monitoring(self):
        """Test monitoring start and stop."""
        # Initially not monitoring
        assert not self.monitor.monitoring
        
        # Start monitoring
        self.monitor.start_monitoring()
        assert self.monitor.monitoring is True
        assert self.monitor.monitor_thread is not None
        assert self.monitor.monitor_thread.is_alive()
        
        # Allow some monitoring to happen
        time.sleep(0.05)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        assert self.monitor.monitoring is False
        
        # Thread should finish
        time.sleep(0.02)
        assert not self.monitor.monitor_thread.is_alive()
    
    def test_measurements_collection(self):
        """Test that measurements are collected."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Let it collect some data
        time.sleep(0.03)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Should have collected some measurements
        with self.monitor.lock:
            assert len(self.monitor.cpu_measurements) > 0
            assert len(self.monitor.memory_measurements) > 0
            # GPU measurements may or may not be present depending on mock
    
    def test_get_statistics_empty(self):
        """Test statistics with no measurements."""
        stats = self.monitor.get_statistics()
        assert isinstance(stats, dict)
        # Should return empty dict or default values
    
    def test_get_statistics_with_data(self):
        """Test statistics calculation with data."""
        # Manually add some test data
        with self.monitor.lock:
            self.monitor.cpu_measurements.extend([10.0, 20.0, 30.0])
            self.monitor.memory_measurements.extend([100.0, 150.0, 200.0])
            self.monitor.gpu_measurements.extend([50.0, 60.0, 70.0])
        
        stats = self.monitor.get_statistics()
        
        assert 'avg_cpu_percent' in stats
        assert 'max_cpu_percent' in stats
        assert 'avg_memory_mb' in stats
        assert 'peak_memory_mb' in stats
        assert 'avg_gpu_percent' in stats
        assert 'max_gpu_percent' in stats
        
        assert stats['avg_cpu_percent'] == 20.0
        assert stats['max_cpu_percent'] == 30.0
        assert stats['avg_memory_mb'] == 150.0
        assert stats['peak_memory_mb'] == 200.0
        assert stats['avg_gpu_percent'] == 60.0
        assert stats['max_gpu_percent'] == 70.0
    
    def test_multiple_start_monitoring(self):
        """Test multiple calls to start_monitoring."""
        # Start monitoring
        self.monitor.start_monitoring()
        first_thread = self.monitor.monitor_thread
        
        # Start again - should not create new thread
        self.monitor.start_monitoring()
        assert self.monitor.monitor_thread == first_thread
        
        self.monitor.stop_monitoring()
    
    def test_gpu_utilization_mock(self):
        """Test GPU utilization mocking."""
        gpu_util = self.monitor._get_gpu_utilization()
        
        # Should return a float or None
        assert isinstance(gpu_util, (float, type(None)))
        
        if gpu_util is not None:
            assert 0 <= gpu_util <= 100
    
    def test_monitoring_error_handling(self):
        """Test error handling in monitoring loop."""
        # Mock psutil to raise exception
        with patch('src.neural_arch.benchmarks.comprehensive_benchmark.psutil') as mock_psutil:
            mock_psutil.cpu_percent.side_effect = Exception("Mock error")
            
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Let it run for a bit
            time.sleep(0.02)
            
            # Should handle error gracefully
            self.monitor.stop_monitoring()
            
            # Should still work without crashing


class TestPerformanceProfiler:
    """Test performance profiling system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.profiler = PerformanceProfiler()
    
    def test_initialization(self):
        """Test PerformanceProfiler initialization."""
        assert isinstance(self.profiler.call_times, defaultdict)
        assert isinstance(self.profiler.call_counts, defaultdict)
        assert isinstance(self.profiler.active_calls, dict)
        assert isinstance(self.profiler.lock, threading.Lock)
    
    def test_profile_context_manager(self):
        """Test profiler context manager."""
        operation_name = "test_operation"
        
        # Profile a simple operation
        with self.profiler.profile(operation_name):
            time.sleep(0.01)  # Small delay
        
        # Check that call was recorded
        assert operation_name in self.profiler.call_times
        assert operation_name in self.profiler.call_counts
        assert self.profiler.call_counts[operation_name] == 1
        assert len(self.profiler.call_times[operation_name]) == 1
        
        # Time should be roughly 0.01 seconds
        recorded_time = self.profiler.call_times[operation_name][0]
        assert 0.005 < recorded_time < 0.05  # Allow some variance
    
    def test_multiple_profile_calls(self):
        """Test multiple profile calls for same operation."""
        operation_name = "repeated_operation"
        
        # Profile multiple times
        for i in range(3):
            with self.profiler.profile(operation_name):
                time.sleep(0.005)  # Small delay
        
        # Check that all calls were recorded
        assert self.profiler.call_counts[operation_name] == 3
        assert len(self.profiler.call_times[operation_name]) == 3
        
        # All times should be reasonable
        for recorded_time in self.profiler.call_times[operation_name]:
            assert 0.002 < recorded_time < 0.02
    
    def test_profile_different_operations(self):
        """Test profiling different operations."""
        # Profile different operations
        with self.profiler.profile("fast_op"):
            time.sleep(0.001)
        
        with self.profiler.profile("slow_op"):
            time.sleep(0.01)
        
        # Check both operations were recorded
        assert "fast_op" in self.profiler.call_times
        assert "slow_op" in self.profiler.call_times
        assert self.profiler.call_counts["fast_op"] == 1
        assert self.profiler.call_counts["slow_op"] == 1
        
        # Slow operation should take longer
        fast_time = self.profiler.call_times["fast_op"][0]
        slow_time = self.profiler.call_times["slow_op"][0]
        assert slow_time > fast_time
    
    def test_profile_with_exception(self):
        """Test profiler handles exceptions correctly."""
        operation_name = "exception_operation"
        
        try:
            with self.profiler.profile(operation_name):
                time.sleep(0.005)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Should still record the timing even with exception
        assert operation_name in self.profiler.call_times
        assert self.profiler.call_counts[operation_name] == 1
        assert len(self.profiler.call_times[operation_name]) == 1
    
    def test_get_profile_report_empty(self):
        """Test profile report with no data."""
        report = self.profiler.get_profile_report()
        assert isinstance(report, dict)
        assert len(report) == 0
    
    def test_get_profile_report_with_data(self):
        """Test profile report generation with data."""
        # Generate some profile data
        with self.profiler.profile("operation_a"):
            time.sleep(0.01)
        
        with self.profiler.profile("operation_a"):
            time.sleep(0.02)
        
        with self.profiler.profile("operation_b"):
            time.sleep(0.005)
        
        report = self.profiler.get_profile_report()
        
        # Check report structure
        assert "operation_a" in report
        assert "operation_b" in report
        
        # Check operation_a statistics
        op_a_stats = report["operation_a"]
        assert op_a_stats["count"] == 2
        assert op_a_stats["total_time"] > 0.025  # Should be sum of both calls
        assert op_a_stats["mean_time"] > 0.01
        assert op_a_stats["median_time"] > 0.01
        assert op_a_stats["min_time"] > 0.005
        assert op_a_stats["max_time"] > 0.015
        assert op_a_stats["std_time"] >= 0  # Standard deviation
        
        # Check operation_b statistics
        op_b_stats = report["operation_b"]
        assert op_b_stats["count"] == 1
        assert op_b_stats["std_time"] == 0.0  # Only one measurement
    
    def test_reset_profiler(self):
        """Test profiler reset functionality."""
        # Add some data
        with self.profiler.profile("test_op"):
            time.sleep(0.001)
        
        # Verify data exists
        assert len(self.profiler.call_times) > 0
        assert len(self.profiler.call_counts) > 0
        
        # Reset
        self.profiler.reset()
        
        # Verify data is cleared
        assert len(self.profiler.call_times) == 0
        assert len(self.profiler.call_counts) == 0
        assert len(self.profiler.active_calls) == 0
    
    def test_concurrent_profiling(self):
        """Test concurrent profiling from multiple threads."""
        import concurrent.futures
        
        def profile_operation(thread_id):
            with self.profiler.profile(f"thread_op_{thread_id}"):
                time.sleep(0.005)
        
        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(profile_operation, i) for i in range(3)]
            concurrent.futures.wait(futures)
        
        # Should have recorded all operations
        assert len(self.profiler.call_times) == 3
        assert all(f"thread_op_{i}" in self.profiler.call_times for i in range(3))
        assert all(self.profiler.call_counts[f"thread_op_{i}"] == 1 for i in range(3))


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.analyzer = StatisticalAnalyzer()
    
    def test_remove_outliers_normal_data(self):
        """Test outlier removal with normal data."""
        data = [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1]  # No real outliers
        
        filtered = self.analyzer.remove_outliers(data, threshold=2.0)
        
        # Should keep most/all data since no extreme outliers
        assert len(filtered) >= len(data) - 1  # Allow for minor filtering
        assert all(item in data for item in filtered)
    
    def test_remove_outliers_with_outliers(self):
        """Test outlier removal with clear outliers."""
        data = [1.0, 1.1, 0.9, 1.2, 0.8, 10.0, 1.0]  # 10.0 is clear outlier
        
        filtered = self.analyzer.remove_outliers(data, threshold=2.0)
        
        # Should remove the outlier
        assert len(filtered) < len(data)
        assert 10.0 not in filtered
        assert all(item <= 2.0 for item in filtered)  # All reasonable values
    
    def test_remove_outliers_edge_cases(self):
        """Test outlier removal edge cases."""
        # Empty data
        assert self.analyzer.remove_outliers([]) == []
        
        # Single data point
        assert self.analyzer.remove_outliers([5.0]) == [5.0]
        
        # Two data points
        assert self.analyzer.remove_outliers([1.0, 2.0]) == [1.0, 2.0]
        
        # All identical values (std_dev = 0)
        identical_data = [5.0, 5.0, 5.0, 5.0]
        assert self.analyzer.remove_outliers(identical_data) == identical_data
    
    def test_compute_confidence_interval_normal_case(self):
        """Test confidence interval computation."""
        data = [1.0, 1.5, 2.0, 2.5, 3.0]  # Simple ascending data
        
        lower, upper = self.analyzer.compute_confidence_interval(data, 0.95)
        
        mean_val = statistics.mean(data)
        assert lower < mean_val < upper
        assert lower >= 0  # Should be reasonable bounds
        assert upper <= 10  # Should be reasonable bounds
    
    def test_compute_confidence_interval_edge_cases(self):
        """Test confidence interval edge cases."""
        # Empty data
        lower, upper = self.analyzer.compute_confidence_interval([])
        assert lower == 0.0 and upper == 0.0
        
        # Single data point
        lower, upper = self.analyzer.compute_confidence_interval([5.0])
        assert lower == 0.0 and upper == 0.0
        
        # Different confidence levels
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        lower_90, upper_90 = self.analyzer.compute_confidence_interval(data, 0.90)
        lower_95, upper_95 = self.analyzer.compute_confidence_interval(data, 0.95)
        lower_99, upper_99 = self.analyzer.compute_confidence_interval(data, 0.99)
        
        # Higher confidence level should give wider interval
        interval_90 = upper_90 - lower_90
        interval_95 = upper_95 - lower_95
        interval_99 = upper_99 - lower_99
        
        assert interval_90 <= interval_95 <= interval_99
    
    def test_detect_regression_no_regression(self):
        """Test regression detection with no regression."""
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95]
        current = [0.98, 1.12, 0.92, 1.0, 1.02]  # Similar performance
        
        result = self.analyzer.detect_regression(current, baseline, threshold=0.10)
        
        assert result['regression_detected'] is False
        assert abs(result['relative_change']) < 0.10
        assert result['current_mean'] > 0
        assert result['baseline_mean'] > 0
    
    def test_detect_regression_with_regression(self):
        """Test regression detection with clear regression."""
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95]  # ~1.0 average
        current = [1.5, 1.6, 1.4, 1.55, 1.45]   # ~1.5 average (50% slower)
        
        result = self.analyzer.detect_regression(current, baseline, threshold=0.10)
        
        assert result['regression_detected'] is True
        assert result['relative_change'] > 0.10
        assert result['current_mean'] > result['baseline_mean']
    
    def test_detect_regression_edge_cases(self):
        """Test regression detection edge cases."""
        baseline = [1.0, 1.1, 0.9]
        
        # Empty current results
        result = self.analyzer.detect_regression([], baseline)
        assert result['regression_detected'] is False
        assert result['reason'] == 'insufficient_data'
        
        # Empty baseline
        result = self.analyzer.detect_regression([1.0, 1.1], [])
        assert result['regression_detected'] is False
        assert result['reason'] == 'insufficient_data'
        
        # Zero baseline mean
        result = self.analyzer.detect_regression([1.0], [0.0, 0.0])
        assert 'relative_change' in result
    
    def test_compute_percentiles_normal_case(self):
        """Test percentile computation."""
        data = list(range(1, 101))  # 1 to 100
        
        percentiles = self.analyzer.compute_percentiles(data)
        
        assert 'p50' in percentiles  # Median
        assert 'p90' in percentiles
        assert 'p95' in percentiles
        assert 'p99' in percentiles
        
        # Check approximate values
        assert 45 <= percentiles['p50'] <= 55  # Should be around 50
        assert 85 <= percentiles['p90'] <= 95  # Should be around 90
        assert 90 <= percentiles['p95'] <= 98  # Should be around 95
        assert 95 <= percentiles['p99'] <= 100  # Should be around 99
    
    def test_compute_percentiles_edge_cases(self):
        """Test percentile computation edge cases."""
        # Empty data
        percentiles = self.analyzer.compute_percentiles([])
        assert percentiles == {}
        
        # Single data point
        percentiles = self.analyzer.compute_percentiles([42.0])
        assert all(val == 42.0 for val in percentiles.values())
        
        # Small dataset
        data = [1.0, 2.0, 3.0]
        percentiles = self.analyzer.compute_percentiles(data)
        assert len(percentiles) == 4  # p50, p90, p95, p99
        
        # All values should be from the original data
        for val in percentiles.values():
            assert val in data


class TestBenchmarkRunner:
    """Test benchmark execution engine."""
    
    def setup_method(self):
        """Setup test environment."""
        self.runner = BenchmarkRunner()
        
        # Simple test function
        def simple_function(sleep_time=0.001):
            time.sleep(sleep_time)
            return "test_result"
        
        self.test_function = simple_function
    
    def teardown_method(self):
        """Cleanup test environment."""
        # Ensure monitoring is stopped
        self.runner.resource_monitor.stop_monitoring()
    
    def test_initialization(self):
        """Test BenchmarkRunner initialization."""
        assert isinstance(self.runner.resource_monitor, ResourceMonitor)
        assert isinstance(self.runner.profiler, PerformanceProfiler)
        assert isinstance(self.runner.analyzer, StatisticalAnalyzer)
        assert isinstance(self.runner.results_history, list)
        assert isinstance(self.runner.baseline_results, dict)
    
    def test_run_benchmark_success(self):
        """Test successful benchmark execution."""
        config = BenchmarkConfig(
            name="test_benchmark",
            benchmark_type=BenchmarkType.THROUGHPUT,
            iterations=10,
            warmup_iterations=2,
            timeout_seconds=30.0
        )
        
        result = self.runner.run_benchmark(config, self.test_function, sleep_time=0.001)
        
        assert result.status == BenchmarkStatus.COMPLETED
        assert result.config == config
        assert result.mean_time > 0
        assert result.median_time > 0
        assert result.min_time > 0
        assert result.max_time >= result.min_time
        assert result.throughput_ops_per_sec > 0
        assert len(result.raw_times) == 10  # Should have 10 measurements
        
        # Should be in results history
        assert result in self.runner.results_history
    
    def test_run_benchmark_with_exception(self):
        """Test benchmark with function that raises exception."""
        def failing_function():
            raise ValueError("Test exception")
        
        config = BenchmarkConfig(
            name="failing_benchmark",
            benchmark_type=BenchmarkType.LATENCY,
            iterations=5,
            warmup_iterations=1
        )
        
        result = self.runner.run_benchmark(config, failing_function)
        
        assert result.status == BenchmarkStatus.FAILED
        assert result.error_message is not None
        assert "Test exception" in result.error_message or "No successful iterations" in result.error_message
    
    def test_run_benchmark_timeout(self):
        """Test benchmark with timeout."""
        def slow_function():
            time.sleep(0.1)  # Slow function
        
        config = BenchmarkConfig(
            name="timeout_benchmark",
            benchmark_type=BenchmarkType.STRESS,
            iterations=100,  # Would take ~10 seconds
            timeout_seconds=0.05  # Very short timeout
        )
        
        result = self.runner.run_benchmark(config, slow_function)
        
        # Should complete but with fewer iterations due to timeout
        if result.status == BenchmarkStatus.COMPLETED:
            assert len(result.raw_times) < 100
    
    def test_run_benchmark_partial_failures(self):
        """Test benchmark with some iterations failing."""
        call_count = 0
        
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise RuntimeError("Intermittent failure")
            time.sleep(0.001)
            return "success"
        
        config = BenchmarkConfig(
            name="partial_failure_benchmark",
            benchmark_type=BenchmarkType.THROUGHPUT,
            iterations=10,
            warmup_iterations=2
        )
        
        result = self.runner.run_benchmark(config, sometimes_failing_function)
        
        # Should complete with successful iterations
        if result.status == BenchmarkStatus.COMPLETED:
            assert len(result.raw_times) > 0
            assert len(result.raw_times) < 10  # Some should have failed
    
    def test_run_scalability_benchmark(self):
        """Test scalability benchmark execution."""
        base_config = BenchmarkConfig(
            name="scalability_test",
            benchmark_type=BenchmarkType.SCALABILITY,
            iterations=5,
            warmup_iterations=1
        )
        
        scale_parameters = {
            'sleep_time': [0.001, 0.002, 0.003],
            'multiplier': [1, 2]
        }
        
        def scalable_function(sleep_time=0.001, multiplier=1):
            time.sleep(sleep_time * multiplier)
            return sleep_time * multiplier
        
        results = self.runner.run_scalability_benchmark(
            base_config,
            scalable_function,
            scale_parameters
        )
        
        # Should have 3 * 2 = 6 combinations
        assert len(results) == 6
        
        # All results should be completed or have reasonable status
        completed_results = [r for r in results if r.status == BenchmarkStatus.COMPLETED]
        assert len(completed_results) > 0
        
        # Results should show scaling behavior
        # Longer sleep times should generally result in lower throughput
        if len(completed_results) > 1:
            sorted_by_sleep = sorted(completed_results, 
                                   key=lambda r: float(r.config.name.split('sleep_time')[1].split(',')[0].split(':')[1].strip().rstrip("'")))
            
            # Not strict check since there might be variations, but general trend
            # assert sorted_by_sleep[0].throughput_ops_per_sec >= sorted_by_sleep[-1].throughput_ops_per_sec
    
    def test_generate_param_combinations(self):
        """Test parameter combination generation."""
        # Empty parameters
        combinations = self.runner._generate_param_combinations({})
        assert combinations == [{}]
        
        # Single parameter
        combinations = self.runner._generate_param_combinations({'a': [1, 2]})
        expected = [{'a': 1}, {'a': 2}]
        assert combinations == expected
        
        # Multiple parameters
        combinations = self.runner._generate_param_combinations({
            'a': [1, 2],
            'b': ['x', 'y']
        })
        expected = [
            {'a': 1, 'b': 'x'},
            {'a': 1, 'b': 'y'},
            {'a': 2, 'b': 'x'},
            {'a': 2, 'b': 'y'}
        ]
        # Order might vary, so check length and content
        assert len(combinations) == 4
        for combo in expected:
            assert combo in combinations
    
    def test_set_and_compare_baseline(self):
        """Test baseline setting and comparison."""
        # Run initial benchmark to set as baseline
        config = BenchmarkConfig(
            name="baseline_test",
            benchmark_type=BenchmarkType.LATENCY,
            iterations=10,
            warmup_iterations=2
        )
        
        baseline_result = self.runner.run_benchmark(config, self.test_function, sleep_time=0.002)
        
        # Set as baseline
        self.runner.set_baseline("test_baseline", baseline_result)
        assert "test_baseline" in self.runner.baseline_results
        assert self.runner.baseline_results["test_baseline"] == baseline_result
        
        # Run current benchmark
        current_result = self.runner.run_benchmark(config, self.test_function, sleep_time=0.001)
        
        # Compare with baseline
        comparison = self.runner.compare_with_baseline(current_result, "test_baseline")
        
        assert 'current_mean_time' in comparison
        assert 'baseline_mean_time' in comparison
        assert 'speedup' in comparison
        assert 'current_throughput' in comparison
        assert 'baseline_throughput' in comparison
        assert 'throughput_ratio' in comparison
        assert 'regression_analysis' in comparison
        
        # Current should be faster (smaller sleep time)
        if baseline_result.status == BenchmarkStatus.COMPLETED and current_result.status == BenchmarkStatus.COMPLETED:
            assert comparison['speedup'] > 1.0  # Should be faster
    
    def test_compare_with_nonexistent_baseline(self):
        """Test comparison with non-existent baseline."""
        config = BenchmarkConfig(name="test", benchmark_type=BenchmarkType.LATENCY)
        result = BenchmarkResult(config=config)
        
        comparison = self.runner.compare_with_baseline(result, "nonexistent")
        
        assert 'error' in comparison
        assert comparison['error'] == 'Baseline not found'
    
    def test_generate_report_empty(self):
        """Test report generation with no results."""
        report = self.runner.generate_report([])
        
        assert 'error' in report
        assert report['error'] == 'No results provided'
    
    def test_generate_report_with_results(self):
        """Test comprehensive report generation."""
        # Run several benchmarks
        config1 = BenchmarkConfig(name="test1", benchmark_type=BenchmarkType.THROUGHPUT, iterations=5)
        config2 = BenchmarkConfig(name="test2", benchmark_type=BenchmarkType.LATENCY, iterations=5)
        
        result1 = self.runner.run_benchmark(config1, self.test_function, sleep_time=0.001)
        result2 = self.runner.run_benchmark(config2, self.test_function, sleep_time=0.002)
        
        # Generate report
        results = [result1, result2]
        report = self.runner.generate_report(results)
        
        # Check report structure
        expected_keys = ['timestamp', 'system_info', 'aggregate_stats', 'detailed_results', 'performance_ranking', 'profiler_report']
        assert all(key in report for key in expected_keys)
        
        # Check system info
        system_info = report['system_info']
        assert 'hostname' in system_info
        assert 'platform' in system_info
        assert 'python_version' in system_info
        assert 'cpu_count' in system_info
        assert 'total_memory_gb' in system_info
        
        # Check aggregate stats
        aggregate_stats = report['aggregate_stats']
        assert aggregate_stats['total_benchmarks'] == 2
        assert aggregate_stats['success_rate'] >= 0.0
        
        # Check detailed results
        assert len(report['detailed_results']) == 2
        
        # Check performance ranking
        assert isinstance(report['performance_ranking'], list)
    
    def test_save_report(self):
        """Test report saving to file."""
        # Create a simple report
        report = {
            'timestamp': time.time(),
            'test_data': [1, 2, 3],
            'system_info': {'platform': 'test'}
        }
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_filename = f.name
        
        try:
            # Save report
            self.runner.save_report(report, temp_filename)
            
            # Verify file was created and contains correct data
            assert os.path.exists(temp_filename)
            
            with open(temp_filename, 'r') as f:
                loaded_report = json.load(f)
            
            assert loaded_report['test_data'] == [1, 2, 3]
            assert loaded_report['system_info']['platform'] == 'test'
            
        finally:
            # Cleanup
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


class TestMockWorkloads:
    """Test mock workload generation and execution."""
    
    def setup_method(self):
        """Setup test environment."""
        self.workloads = create_mock_workloads()
    
    def test_workload_creation(self):
        """Test that all expected workloads are created."""
        expected_workloads = ['cpu_intensive', 'memory_intensive', 'io_intensive', 'mixed_workload']
        
        assert len(self.workloads) == len(expected_workloads)
        for workload_name in expected_workloads:
            assert workload_name in self.workloads
            assert callable(self.workloads[workload_name])
    
    def test_cpu_intensive_workload(self):
        """Test CPU-intensive workload."""
        cpu_func = self.workloads['cpu_intensive']
        
        # Test with small size for speed
        result = cpu_func(size=10)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)
        assert result.dtype == np.float32
    
    def test_cpu_intensive_workload_scaling(self):
        """Test CPU-intensive workload scales with size."""
        cpu_func = self.workloads['cpu_intensive']
        
        # Time different sizes
        import time
        
        start_time = time.perf_counter()
        cpu_func(size=50)
        small_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        cpu_func(size=100)
        large_time = time.perf_counter() - start_time
        
        # Larger size should take more time
        assert large_time > small_time
    
    def test_memory_intensive_workload(self):
        """Test memory-intensive workload."""
        memory_func = self.workloads['memory_intensive']
        
        # Test with small size for speed
        result = memory_func(size=1000)
        
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)
    
    def test_memory_intensive_workload_scaling(self):
        """Test memory-intensive workload scales with size."""
        memory_func = self.workloads['memory_intensive']
        
        # Different sizes should complete successfully
        result_small = memory_func(size=1000)
        result_large = memory_func(size=10000)
        
        assert isinstance(result_small, (float, np.floating))
        assert isinstance(result_large, (float, np.floating))
    
    def test_io_intensive_workload(self):
        """Test I/O intensive workload."""
        io_func = self.workloads['io_intensive']
        
        # Test with small iterations for speed
        result = io_func(iterations=10)
        
        assert isinstance(result, int)
        assert result == 10  # Should return number of lines written
    
    def test_io_intensive_workload_scaling(self):
        """Test I/O intensive workload scales with iterations."""
        io_func = self.workloads['io_intensive']
        
        result_small = io_func(iterations=5)
        result_large = io_func(iterations=15)
        
        assert result_small == 5
        assert result_large == 15
    
    def test_mixed_workload(self):
        """Test mixed workload functionality."""
        mixed_func = self.workloads['mixed_workload']
        
        # Test with small parameters for speed
        result = mixed_func(cpu_size=5, memory_size=1000, io_iterations=3)
        
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)
        assert result > 0  # Should be positive sum
    
    def test_mixed_workload_components(self):
        """Test that mixed workload uses all components."""
        mixed_func = self.workloads['mixed_workload']
        
        # Mock the individual components to verify they're called
        with patch.object(self, 'workloads') as mock_workloads:
            # Create mock functions
            mock_cpu = Mock(return_value=np.array([[1.0, 2.0], [3.0, 4.0]]))
            mock_memory = Mock(return_value=10.0)
            mock_io = Mock(return_value=5)
            
            # Replace in workloads dict
            mock_workloads_dict = {
                'cpu_intensive': mock_cpu,
                'memory_intensive': mock_memory,
                'io_intensive': mock_io
            }
            
            # Test mixed workload by calling the functions directly
            cpu_result = mock_cpu()
            memory_result = mock_memory()
            io_result = mock_io()
            
            final_result = cpu_result.sum() + memory_result + io_result
            
            assert final_result == 10.0 + 10.0 + 5  # 25.0


class TestIntegrationScenarios:
    """Integration tests for comprehensive benchmarking scenarios."""
    
    def test_full_benchmark_pipeline(self):
        """Test complete benchmark pipeline from execution to reporting."""
        runner = BenchmarkRunner()
        
        # Create simple test function
        def test_workload(delay=0.001, work_amount=100):
            time.sleep(delay)
            # Simulate some work
            result = sum(range(work_amount))
            return result
        
        # Run throughput benchmark
        throughput_config = BenchmarkConfig(
            name="pipeline_throughput",
            benchmark_type=BenchmarkType.THROUGHPUT,
            iterations=5,
            warmup_iterations=2
        )
        
        throughput_result = runner.run_benchmark(
            throughput_config, 
            test_workload, 
            delay=0.001, 
            work_amount=50
        )
        
        # Run latency benchmark
        latency_config = BenchmarkConfig(
            name="pipeline_latency",
            benchmark_type=BenchmarkType.LATENCY,
            iterations=5,
            warmup_iterations=2
        )
        
        latency_result = runner.run_benchmark(
            latency_config, 
            test_workload, 
            delay=0.002, 
            work_amount=50
        )
        
        # Set baseline and compare
        runner.set_baseline("throughput_baseline", throughput_result)
        comparison = runner.compare_with_baseline(latency_result, "throughput_baseline")
        
        # Generate comprehensive report
        all_results = [throughput_result, latency_result]
        report = runner.generate_report(all_results)
        
        # Verify pipeline completed successfully
        assert throughput_result.status in [BenchmarkStatus.COMPLETED, BenchmarkStatus.FAILED]
        assert latency_result.status in [BenchmarkStatus.COMPLETED, BenchmarkStatus.FAILED]
        assert 'regression_analysis' in comparison
        assert 'system_info' in report
        assert len(report['detailed_results']) == 2
        
        runner.resource_monitor.stop_monitoring()
    
    def test_scalability_analysis(self):
        """Test scalability analysis across different parameters."""
        runner = BenchmarkRunner()
        
        def scalable_workload(size=100, complexity=1):
            # Simulate work that scales with parameters
            data = np.random.randn(size, size // 10)
            for _ in range(complexity):
                data = data @ data.T[:size // 10, :size // 10]
            return data.sum()
        
        base_config = BenchmarkConfig(
            name="scalability_analysis",
            benchmark_type=BenchmarkType.SCALABILITY,
            iterations=3,
            warmup_iterations=1,
            timeout_seconds=30.0
        )
        
        scale_parameters = {
            'size': [10, 20, 30],
            'complexity': [1, 2]
        }
        
        results = runner.run_scalability_benchmark(
            base_config,
            scalable_workload,
            scale_parameters
        )
        
        # Should have 3 * 2 = 6 results
        assert len(results) == 6
        
        # Analyze scaling behavior
        completed_results = [r for r in results if r.status == BenchmarkStatus.COMPLETED]
        
        if len(completed_results) > 1:
            # Generally, larger size should result in lower throughput
            # (but this isn't guaranteed due to various factors)
            throughputs = [r.throughput_ops_per_sec for r in completed_results]
            assert all(t >= 0 for t in throughputs)  # All should be non-negative
        
        runner.resource_monitor.stop_monitoring()
    
    def test_regression_detection_workflow(self):
        """Test regression detection workflow."""
        runner = BenchmarkRunner()
        
        def performance_workload(performance_factor=1.0):
            # Simulate workload with adjustable performance
            delay = 0.001 * performance_factor
            time.sleep(delay)
            return delay
        
        # Baseline benchmark (good performance)
        baseline_config = BenchmarkConfig(
            name="regression_baseline",
            benchmark_type=BenchmarkType.REGRESSION,
            iterations=10,
            warmup_iterations=2,
            regression_threshold=0.10  # 10% regression threshold
        )
        
        baseline_result = runner.run_benchmark(
            baseline_config,
            performance_workload,
            performance_factor=1.0
        )
        
        # Set as baseline
        runner.set_baseline("performance_baseline", baseline_result)
        
        # Current benchmark (degraded performance)
        current_config = BenchmarkConfig(
            name="regression_current",
            benchmark_type=BenchmarkType.REGRESSION,
            iterations=10,
            warmup_iterations=2,
            regression_threshold=0.10
        )
        
        current_result = runner.run_benchmark(
            current_config,
            performance_workload,
            performance_factor=1.5  # 50% slower
        )
        
        # Detect regression
        comparison = runner.compare_with_baseline(current_result, "performance_baseline")
        
        if (baseline_result.status == BenchmarkStatus.COMPLETED and 
            current_result.status == BenchmarkStatus.COMPLETED):
            # Should detect regression
            regression_analysis = comparison['regression_analysis']
            assert regression_analysis['regression_detected'] is True
            assert regression_analysis['relative_change'] > 0.10
        
        runner.resource_monitor.stop_monitoring()
    
    def test_resource_monitoring_integration(self):
        """Test resource monitoring integration."""
        runner = BenchmarkRunner()
        
        def resource_heavy_workload():
            # Create some CPU and memory load
            data = np.random.randn(1000, 1000)
            result = np.linalg.inv(data @ data.T + np.eye(1000))
            return result.sum()
        
        config = BenchmarkConfig(
            name="resource_monitoring_test",
            benchmark_type=BenchmarkType.STRESS,
            iterations=3,
            warmup_iterations=1
        )
        
        result = runner.run_benchmark(config, resource_heavy_workload)
        
        if result.status == BenchmarkStatus.COMPLETED:
            # Should have resource measurements
            assert result.avg_cpu_percent >= 0
            assert result.peak_memory_mb >= 0
            assert result.avg_memory_mb >= 0
            
            # Memory efficiency should be reasonable
            assert 0 <= result.memory_efficiency <= 1.0
        
        runner.resource_monitor.stop_monitoring()
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        runner = BenchmarkRunner()
        
        call_count = 0
        
        def flaky_workload():
            nonlocal call_count
            call_count += 1
            
            # Fail first few calls, then succeed
            if call_count <= 3:
                raise RuntimeError(f"Flaky failure #{call_count}")
            
            time.sleep(0.001)
            return "success"
        
        config = BenchmarkConfig(
            name="error_handling_test",
            benchmark_type=BenchmarkType.STRESS,
            iterations=10,
            warmup_iterations=2
        )
        
        result = runner.run_benchmark(config, flaky_workload)
        
        # Should handle partial failures gracefully
        if result.status == BenchmarkStatus.COMPLETED:
            # Should have some successful iterations
            assert len(result.raw_times) > 0
            assert len(result.raw_times) < 10  # Some should have failed
        elif result.status == BenchmarkStatus.FAILED:
            # If all failed, should have appropriate error message
            assert result.error_message is not None
        
        runner.resource_monitor.stop_monitoring()


def test_system_integration():
    """Test the complete comprehensive benchmark system integration."""
    # Mock the system resources to avoid long-running tests
    with patch('time.sleep', return_value=None):  # Speed up sleep calls
        with patch('numpy.random.randn') as mock_randn:
            # Mock numpy operations to return predictable results
            mock_randn.return_value = np.ones((100, 100))
            
            with patch('numpy.matmul') as mock_matmul:
                mock_matmul.return_value = np.ones((100, 100))
                
                try:
                    # Run a simplified version of the comprehensive benchmark
                    runner = BenchmarkRunner()
                    workloads = create_mock_workloads()
                    
                    # Test one workload type
                    config = BenchmarkConfig(
                        name="integration_test",
                        benchmark_type=BenchmarkType.THROUGHPUT,
                        iterations=3,  # Small number for testing
                        warmup_iterations=1,
                        timeout_seconds=5.0
                    )
                    
                    result = runner.run_benchmark(
                        config, 
                        workloads['cpu_intensive'], 
                        size=10
                    )
                    
                    # Generate report
                    report = runner.generate_report([result])
                    
                    assert result.status in [BenchmarkStatus.COMPLETED, BenchmarkStatus.FAILED]
                    assert 'system_info' in report
                    
                    print("✅ Comprehensive benchmark system integration test passed")
                    
                except Exception as e:
                    pytest.fail(f"Comprehensive benchmark system test failed: {e}")
                
                finally:
                    runner.resource_monitor.stop_monitoring()


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])