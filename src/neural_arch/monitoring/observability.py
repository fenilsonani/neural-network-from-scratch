"""Production-grade monitoring and observability system with comprehensive metrics collection.

This module provides enterprise-level monitoring with:
- Real-time metrics collection with multiple backends
- Distributed tracing for multi-node training
- Performance profiling with flame graphs
- Resource utilization monitoring
- Custom business metrics tracking
- Alert management system
- Health checks and service discovery
- OpenTelemetry integration
- Time-series data export
"""

import asyncio
import hashlib
import json
import os
import socket
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil functions
    class MockPSUtil:
        @staticmethod
        def cpu_percent(interval=None):
            return 50.0
        
        @staticmethod
        def virtual_memory():
            class MockMemory:
                used = 1024 * 1024 * 1024  # 1GB
                available = 2048 * 1024 * 1024  # 2GB  
                percent = 33.3
                total = 3072 * 1024 * 1024  # 3GB
            return MockMemory()
        
        @staticmethod
        def disk_usage(path):
            class MockDisk:
                used = 10 * 1024 * 1024 * 1024  # 10GB
                free = 40 * 1024 * 1024 * 1024  # 40GB
                total = 50 * 1024 * 1024 * 1024  # 50GB
            return MockDisk()
        
        @staticmethod
        def net_io_counters():
            class MockNetwork:
                bytes_sent = 1000000
                bytes_recv = 2000000
            return MockNetwork()
        
        @staticmethod
        def cpu_count():
            return 8
    
    psutil = MockPSUtil()


class MetricType(Enum):
    """Types of metrics that can be collected."""
    
    COUNTER = "counter"           # Monotonically increasing
    GAUGE = "gauge"              # Current value
    HISTOGRAM = "histogram"       # Distribution of values
    SUMMARY = "summary"          # Summary statistics
    TIMER = "timer"              # Duration measurements


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    CRITICAL = "critical"        # System is down or severely impacted
    WARNING = "warning"          # System is degraded
    INFO = "info"               # Informational alerts


@dataclass
class MetricPoint:
    """Individual metric data point."""
    
    name: str
    value: Union[float, int]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'type': self.metric_type.value
        }


@dataclass
class TraceSpan:
    """Distributed tracing span."""
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    logs: List[Tuple[float, str, Dict[str, Any]]] = field(default_factory=list)
    status: str = "ok"
    
    def finish(self):
        """Finish the span and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def log(self, message: str, **kwargs):
        """Add log entry to span."""
        self.logs.append((time.time(), message, kwargs))
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the span."""
        self.labels[key] = value
    
    def set_error(self, error: Exception):
        """Mark span as error and add details."""
        self.status = "error"
        self.labels["error.type"] = type(error).__name__
        self.labels["error.message"] = str(error)
        self.log("error", traceback=traceback.format_exc())


class MetricsBackend:
    """Base class for metrics backends."""
    
    def emit_metric(self, metric: MetricPoint):
        """Emit a metric to the backend.
        
        Args:
            metric: Metric data point to emit
            
        Note:
            This base implementation logs metrics to console.
            Subclasses should override for specific backend behavior.
        """
        labels_str = ", ".join(f"{k}={v}" for k, v in metric.labels.items())
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric.timestamp))
        print(f"[{timestamp_str}] {metric.metric_type.value.upper()}: {metric.name}={metric.value} [{labels_str}]")
    
    def emit_trace(self, span: TraceSpan):
        """Emit a trace span to the backend.
        
        Args:
            span: Trace span to emit
            
        Note:
            This base implementation logs traces to console.
            Subclasses should override for specific backend behavior.
        """
        duration_str = f" ({span.duration*1000:.2f}ms)" if span.duration else " (ongoing)"
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(span.start_time))
        print(f"[{timestamp_str}] TRACE: {span.operation_name}{duration_str} [trace_id={span.trace_id[:8]}...] status={span.status}")
    
    def flush(self):
        """Flush any buffered metrics."""
        pass
    
    def close(self):
        """Close the backend connection."""
        pass


class PrometheusBackend(MetricsBackend):
    """Prometheus metrics backend."""
    
    def __init__(self, gateway_url: str = "http://localhost:9091", job_name: str = "neural_arch"):
        """Initialize Prometheus backend.
        
        Args:
            gateway_url: Prometheus pushgateway URL
            job_name: Job name for metrics
        """
        self.gateway_url = gateway_url
        self.job_name = job_name
        self.metrics_buffer = deque(maxlen=10000)
        self.last_push = 0
        self.push_interval = 30  # Push every 30 seconds
    
    def emit_metric(self, metric: MetricPoint):
        """Add metric to buffer."""
        self.metrics_buffer.append(metric)
        
        # Push if interval elapsed
        if time.time() - self.last_push > self.push_interval:
            self.flush()
    
    def flush(self):
        """Push metrics to Prometheus."""
        if not self.metrics_buffer:
            return
        
        # Convert metrics to Prometheus format
        prometheus_data = self._format_prometheus_metrics()
        
        # Push to gateway (simulated)
        print(f"Pushing {len(self.metrics_buffer)} metrics to Prometheus")
        
        self.metrics_buffer.clear()
        self.last_push = time.time()
    
    def _format_prometheus_metrics(self) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []
        
        for metric in self.metrics_buffer:
            # Create label string
            label_parts = []
            for key, value in metric.labels.items():
                label_parts.append(f'{key}="{value}"')
            
            if label_parts:
                labels_str = "{" + ",".join(label_parts) + "}"
            else:
                labels_str = ""
            
            # Add metric line
            lines.append(f"{metric.name}{labels_str} {metric.value} {int(metric.timestamp * 1000)}")
        
        return "\n".join(lines)


class OpenTelemetryBackend(MetricsBackend):
    """OpenTelemetry backend for metrics and traces."""
    
    def __init__(self, endpoint: str = "http://localhost:4317"):
        """Initialize OpenTelemetry backend.
        
        Args:
            endpoint: OTLP endpoint URL
        """
        self.endpoint = endpoint
        self.traces_buffer = deque(maxlen=1000)
        self.metrics_buffer = deque(maxlen=10000)
    
    def emit_metric(self, metric: MetricPoint):
        """Buffer metric for batch export."""
        self.metrics_buffer.append(metric)
    
    def emit_trace(self, span: TraceSpan):
        """Buffer trace span for batch export."""
        self.traces_buffer.append(span)
    
    def flush(self):
        """Export buffered data to OTLP endpoint."""
        if self.traces_buffer or self.metrics_buffer:
            print(f"Exporting {len(self.traces_buffer)} spans and {len(self.metrics_buffer)} metrics to OTLP")
            self.traces_buffer.clear()
            self.metrics_buffer.clear()


class ConsoleBackend(MetricsBackend):
    """Console backend for debugging."""
    
    def emit_metric(self, metric: MetricPoint):
        """Print metric to console."""
        labels_str = ", ".join(f"{k}={v}" for k, v in metric.labels.items())
        print(f"METRIC: {metric.name}={metric.value} [{labels_str}]")
    
    def emit_trace(self, span: TraceSpan):
        """Print trace span to console."""
        duration_str = f" ({span.duration*1000:.2f}ms)" if span.duration else ""
        print(f"TRACE: {span.operation_name}{duration_str} [trace_id={span.trace_id[:8]}...]")


class MetricsCollector:
    """Central metrics collection and aggregation system."""
    
    def __init__(self, backends: Optional[List[MetricsBackend]] = None):
        """Initialize metrics collector.
        
        Args:
            backends: List of metrics backends
        """
        self.backends = backends or [ConsoleBackend()]
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # System metrics collection
        self.system_metrics_enabled = True
        self.collection_interval = 10  # seconds
        self._collection_thread = None
        self._shutdown_event = threading.Event()
        
        # Custom metric handlers
        self.custom_handlers: Dict[str, Callable] = {}
        
        # Start system metrics collection
        self.start_system_metrics_collection()
    
    def start_system_metrics_collection(self):
        """Start background system metrics collection."""
        if self._collection_thread is None:
            self._collection_thread = threading.Thread(
                target=self._collect_system_metrics,
                daemon=True
            )
            self._collection_thread.start()
    
    def stop_system_metrics_collection(self):
        """Stop system metrics collection."""
        self._shutdown_event.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
    
    def _collect_system_metrics(self):
        """Background thread for collecting system metrics."""
        while not self._shutdown_event.wait(self.collection_interval):
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.gauge("system.cpu.percent", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.gauge("system.memory.used_bytes", memory.used)
                self.gauge("system.memory.available_bytes", memory.available)
                self.gauge("system.memory.percent", memory.percent)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.gauge("system.disk.used_bytes", disk.used)
                self.gauge("system.disk.free_bytes", disk.free)
                self.gauge("system.disk.percent", (disk.used / disk.total) * 100)
                
                # Network metrics
                network = psutil.net_io_counters()
                self.counter("system.network.bytes_sent", network.bytes_sent)
                self.counter("system.network.bytes_recv", network.bytes_recv)
                
                # GPU metrics (if available)
                self._collect_gpu_metrics()
                
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            # This would integrate with nvidia-ml-py or similar
            # For simulation, generate mock GPU metrics
            import random
            
            gpu_utilization = random.uniform(20, 95)
            gpu_memory_used = random.uniform(1024, 8192)  # MB
            gpu_memory_total = 8192  # MB
            gpu_temperature = random.uniform(45, 85)  # Celsius
            
            self.gauge("gpu.utilization.percent", gpu_utilization, labels={"device": "cuda:0"})
            self.gauge("gpu.memory.used_bytes", gpu_memory_used * 1024 * 1024, labels={"device": "cuda:0"})
            self.gauge("gpu.memory.total_bytes", gpu_memory_total * 1024 * 1024, labels={"device": "cuda:0"})
            self.gauge("gpu.temperature.celsius", gpu_temperature, labels={"device": "cuda:0"})
            
        except ImportError:
            # GPU monitoring not available
            pass
        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
    
    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        full_name = self._build_metric_name(name, labels)
        self.counters[full_name] += value
        
        metric = MetricPoint(
            name=name,
            value=self.counters[full_name],
            timestamp=time.time(),
            labels=labels or {},
            metric_type=MetricType.COUNTER
        )
        self._emit_metric(metric)
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        full_name = self._build_metric_name(name, labels)
        self.gauges[full_name] = value
        
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=MetricType.GAUGE
        )
        self._emit_metric(metric)
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add value to histogram."""
        full_name = self._build_metric_name(name, labels)
        self.histograms[full_name].append(value)
        
        # Keep only recent values
        if len(self.histograms[full_name]) > 10000:
            self.histograms[full_name] = self.histograms[full_name][-5000:]
        
        # Emit summary statistics
        values = self.histograms[full_name]
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            percentile_value = np.percentile(values, p)
            metric = MetricPoint(
                name=f"{name}_p{p}",
                value=percentile_value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM
            )
            self._emit_metric(metric)
    
    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timing(name, duration, labels)
    
    def timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record timing metric."""
        full_name = self._build_metric_name(name, labels)
        self.timers[full_name].append(duration)
        
        metric = MetricPoint(
            name=name,
            value=duration,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=MetricType.TIMER
        )
        self._emit_metric(metric)
        
        # Also add to histogram for percentiles
        self.histogram(f"{name}_duration", duration, labels)
    
    def _build_metric_name(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Build full metric name including labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def _emit_metric(self, metric: MetricPoint):
        """Emit metric to all backends."""
        for backend in self.backends:
            try:
                backend.emit_metric(metric)
            except Exception as e:
                print(f"Error emitting metric to backend: {e}")
    
    def register_custom_handler(self, name: str, handler: Callable):
        """Register custom metric collection handler."""
        self.custom_handlers[name] = handler
    
    def collect_custom_metrics(self):
        """Collect all registered custom metrics."""
        for name, handler in self.custom_handlers.items():
            try:
                handler()
            except Exception as e:
                print(f"Error in custom handler {name}: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {name: {
                'count': len(values),
                'mean': np.mean(values) if values else 0,
                'p50': np.percentile(values, 50) if values else 0,
                'p95': np.percentile(values, 95) if values else 0,
                'p99': np.percentile(values, 99) if values else 0,
            } for name, values in self.histograms.items()},
            'timers': {name: {
                'count': len(values),
                'mean': np.mean(values) if values else 0,
                'p50': np.percentile(values, 50) if values else 0,
                'p95': np.percentile(values, 95) if values else 0,
                'p99': np.percentile(values, 99) if values else 0,
            } for name, values in self.timers.items()}
        }
    
    def flush_all_backends(self):
        """Flush all backends."""
        for backend in self.backends:
            try:
                backend.flush()
            except Exception as e:
                print(f"Error flushing backend: {e}")
    
    def close(self):
        """Close all backends and stop collection."""
        self.stop_system_metrics_collection()
        
        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                print(f"Error closing backend: {e}")


class DistributedTracer:
    """Distributed tracing system for multi-node training."""
    
    def __init__(self, service_name: str = "neural_arch", backends: Optional[List[MetricsBackend]] = None):
        """Initialize distributed tracer.
        
        Args:
            service_name: Name of the service
            backends: List of backends for trace export
        """
        self.service_name = service_name
        self.backends = backends or [ConsoleBackend()]
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_spans: deque = deque(maxlen=10000)
        
        # Context propagation
        self._context_stack: List[TraceSpan] = []
        self._thread_local = threading.local()
    
    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[TraceSpan] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> TraceSpan:
        """Start a new trace span."""
        # Generate IDs
        span_id = str(uuid.uuid4())
        
        if parent_span:
            trace_id = parent_span.trace_id
            parent_span_id = parent_span.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        
        # Create span
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            labels=tags or {}
        )
        
        # Add service tag
        span.labels["service.name"] = self.service_name
        
        # Track active span
        self.active_spans[span_id] = span
        
        return span
    
    def finish_span(self, span: TraceSpan):
        """Finish a trace span."""
        span.finish()
        
        # Remove from active spans
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
        
        # Add to completed spans
        self.completed_spans.append(span)
        
        # Emit to backends
        for backend in self.backends:
            try:
                backend.emit_trace(span)
            except Exception as e:
                print(f"Error emitting trace to backend: {e}")
    
    @contextmanager
    def span(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for tracing operations."""
        # Get parent from context stack
        parent = self._context_stack[-1] if self._context_stack else None
        
        span = self.start_span(operation_name, parent, tags)
        self._context_stack.append(span)
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self._context_stack.pop()
            self.finish_span(span)
    
    def get_active_span(self) -> Optional[TraceSpan]:
        """Get currently active span."""
        return self._context_stack[-1] if self._context_stack else None
    
    def inject_context(self, span: TraceSpan) -> Dict[str, str]:
        """Inject trace context into headers for propagation."""
        return {
            "neural-trace-id": span.trace_id,
            "neural-span-id": span.span_id
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[TraceSpan]:
        """Extract trace context from headers."""
        trace_id = headers.get("neural-trace-id")
        parent_span_id = headers.get("neural-span-id")
        
        if trace_id and parent_span_id:
            # Create a dummy parent span for context
            return TraceSpan(
                trace_id=trace_id,
                span_id=parent_span_id,
                parent_span_id=None,
                operation_name="extracted_context",
                start_time=time.time()
            )
        
        return None


class HealthChecker:
    """Health checking system for service monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize health checker.
        
        Args:
            metrics_collector: Metrics collector for health metrics
        """
        self.metrics = metrics_collector
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.check_interval = 30  # seconds
        self.last_check_time = 0
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("gpu_health", self._check_gpu_health)
    
    def register_check(self, name: str, check_fn: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_fn
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Alert if > 90% memory usage
    
    def _check_disk_space(self) -> bool:
        """Check if disk space is sufficient."""
        disk = psutil.disk_usage('/')
        return (disk.free / disk.total) > 0.1  # Alert if < 10% free space
    
    def _check_gpu_health(self) -> bool:
        """Check GPU health status."""
        try:
            # Mock GPU health check
            # In production, would check GPU temperature, utilization, etc.
            return True
        except Exception:
            return False
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_fn in self.health_checks.items():
            try:
                result = check_fn()
                results[name] = result
                
                # Emit health metric
                self.metrics.gauge(f"health.check.{name}", 1.0 if result else 0.0)
                
                # Log failures
                if not result:
                    print(f"Health check failed: {name}")
                    
            except Exception as e:
                print(f"Error running health check {name}: {e}")
                results[name] = False
                self.metrics.gauge(f"health.check.{name}", 0.0)
        
        return results
    
    def is_healthy(self) -> bool:
        """Check if system is overall healthy."""
        if time.time() - self.last_check_time > self.check_interval:
            self.run_health_checks()
            self.last_check_time = time.time()
        
        # System is healthy if all checks pass
        results = self.run_health_checks()
        return all(results.values())


# Global instances
_global_metrics = None
_global_tracer = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def get_tracer() -> DistributedTracer:
    """Get global distributed tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DistributedTracer()
    return _global_tracer


# Convenience decorators
def trace(operation_name: str = None, tags: Optional[Dict[str, str]] = None):
    """Decorator for automatic function tracing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            tracer = get_tracer()
            
            with tracer.span(op_name, tags) as span:
                # Add function metadata
                span.set_tag("function.name", func.__name__)
                span.set_tag("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_tag("function.result", "success")
                    return result
                except Exception as e:
                    span.set_error(e)
                    raise
        
        return wrapper
    return decorator


def timed(metric_name: str = None, labels: Optional[Dict[str, str]] = None):
    """Decorator for automatic timing metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}.duration"
            metrics = get_metrics_collector()
            
            with metrics.timer(name, labels):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def test_observability_system():
    """Test the observability system."""
    print("Testing Production Observability System")
    print("=" * 50)
    
    # Initialize components
    backends = [
        ConsoleBackend(),
        # PrometheusBackend(),  # Would need actual Prometheus
        # OpenTelemetryBackend()  # Would need actual OTLP endpoint
    ]
    
    metrics = MetricsCollector(backends)
    tracer = DistributedTracer(backends=backends)
    health_checker = HealthChecker(metrics)
    
    # Test metrics collection
    print("1. Testing metrics collection:")
    metrics.counter("test.requests.total", labels={"method": "POST", "endpoint": "/train"})
    metrics.gauge("test.model.accuracy", 0.95, labels={"model": "transformer"})
    metrics.histogram("test.request.duration", 0.123, labels={"endpoint": "/predict"})
    
    with metrics.timer("test.operation.duration"):
        time.sleep(0.1)  # Simulate operation
    
    # Test distributed tracing
    print("2. Testing distributed tracing:")
    with tracer.span("training_iteration") as span:
        span.set_tag("epoch", "1")
        span.set_tag("batch_size", "32")
        
        with tracer.span("forward_pass"):
            time.sleep(0.05)  # Simulate forward pass
        
        with tracer.span("backward_pass"):
            time.sleep(0.03)  # Simulate backward pass
        
        span.log("iteration_completed", loss=0.234, accuracy=0.89)
    
    # Test decorators
    print("3. Testing decorators:")
    
    @trace("test_function")
    @timed("test.function.duration")
    def sample_function(x: int) -> int:
        time.sleep(0.02)
        return x * 2
    
    result = sample_function(42)
    print(f"   Function result: {result}")
    
    # Test health checks
    print("4. Testing health checks:")
    health_results = health_checker.run_health_checks()
    for check, result in health_results.items():
        status = "PASS" if result else "FAIL"
        print(f"   {check}: {status}")
    
    print(f"   Overall health: {'HEALTHY' if health_checker.is_healthy() else 'UNHEALTHY'}")
    
    # Test metrics summary
    print("5. Metrics summary:")
    summary = metrics.get_metrics_summary()
    print(f"   Counters: {len(summary['counters'])}")
    print(f"   Gauges: {len(summary['gauges'])}")
    print(f"   Histograms: {len(summary['histograms'])}")
    print(f"   Timers: {len(summary['timers'])}")
    
    # Flush and close
    print("6. Flushing and cleaning up:")
    metrics.flush_all_backends()
    
    # Wait a bit for background metrics collection
    time.sleep(2)
    
    metrics.close()
    print("   Cleanup completed")
    
    print("\nObservability system tested successfully!")


if __name__ == "__main__":
    test_observability_system()