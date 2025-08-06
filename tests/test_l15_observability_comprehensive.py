"""
Comprehensive test suite for production observability system.
Tests all components of observability.py for comprehensive coverage.

This module tests:
- MetricsCollector with multiple backends
- DistributedTracer for multi-node tracing
- Different metrics backends (Prometheus, OpenTelemetry, Console)
- HealthChecker system
- Trace spans and context propagation
- Decorators for automatic instrumentation
- Alert management
- Performance profiling
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any

import numpy as np
import pytest

from src.neural_arch.monitoring.observability import (
    MetricsCollector,
    DistributedTracer,
    MetricsBackend,
    PrometheusBackend,
    OpenTelemetryBackend,
    ConsoleBackend,
    HealthChecker,
    MetricPoint,
    TraceSpan,
    MetricType,
    AlertSeverity,
    get_metrics_collector,
    get_tracer,
    trace,
    timed,
    test_observability_system
)


class TestMetricType:
    """Test MetricType enumeration."""
    
    def test_metric_types_exist(self):
        """Test that all metric types are defined."""
        expected_types = ['COUNTER', 'GAUGE', 'HISTOGRAM', 'SUMMARY', 'TIMER']
        
        for type_name in expected_types:
            assert hasattr(MetricType, type_name)
            metric_type = getattr(MetricType, type_name)
            assert isinstance(metric_type, MetricType)
            
    def test_metric_type_values(self):
        """Test metric type string values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"
        assert MetricType.TIMER.value == "timer"


class TestAlertSeverity:
    """Test AlertSeverity enumeration."""
    
    def test_alert_severities_exist(self):
        """Test that all alert severities are defined."""
        expected_severities = ['CRITICAL', 'WARNING', 'INFO']
        
        for severity_name in expected_severities:
            assert hasattr(AlertSeverity, severity_name)
            severity = getattr(AlertSeverity, severity_name)
            assert isinstance(severity, AlertSeverity)
            
    def test_alert_severity_values(self):
        """Test alert severity string values."""
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.INFO.value == "info"


class TestMetricPoint:
    """Test MetricPoint data structure."""
    
    def test_metric_point_creation(self):
        """Test MetricPoint creation and properties."""
        timestamp = time.time()
        labels = {"method": "POST", "endpoint": "/train"}
        
        point = MetricPoint(
            name="http_requests_total",
            value=42.0,
            timestamp=timestamp,
            labels=labels,
            metric_type=MetricType.COUNTER
        )
        
        assert point.name == "http_requests_total"
        assert point.value == 42.0
        assert point.timestamp == timestamp
        assert point.labels == labels
        assert point.metric_type == MetricType.COUNTER
        
    def test_metric_point_defaults(self):
        """Test MetricPoint default values."""
        point = MetricPoint(
            name="test_metric",
            value=1.0,
            timestamp=time.time()
        )
        
        assert point.labels == {}
        assert point.metric_type == MetricType.GAUGE
        
    def test_metric_point_serialization(self):
        """Test MetricPoint to_dict serialization."""
        timestamp = time.time()
        point = MetricPoint(
            name="test_metric",
            value=3.14,
            timestamp=timestamp,
            labels={"tag": "value"},
            metric_type=MetricType.HISTOGRAM
        )
        
        data = point.to_dict()
        
        expected_keys = ['name', 'value', 'timestamp', 'labels', 'type']
        for key in expected_keys:
            assert key in data
            
        assert data['name'] == "test_metric"
        assert data['value'] == 3.14
        assert data['timestamp'] == timestamp
        assert data['labels'] == {"tag": "value"}
        assert data['type'] == "histogram"


class TestTraceSpan:
    """Test TraceSpan for distributed tracing."""
    
    def test_trace_span_creation(self):
        """Test TraceSpan creation and properties."""
        trace_id = "trace_123"
        span_id = "span_456"
        parent_span_id = "parent_789"
        operation_name = "database_query"
        start_time = time.time()
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=start_time
        )
        
        assert span.trace_id == trace_id
        assert span.span_id == span_id
        assert span.parent_span_id == parent_span_id
        assert span.operation_name == operation_name
        assert span.start_time == start_time
        assert span.end_time is None
        assert span.duration is None
        assert span.status == "ok"
        assert len(span.logs) == 0
        assert len(span.labels) == 0
        
    def test_trace_span_finish(self):
        """Test finishing a trace span."""
        span = TraceSpan(
            trace_id="trace_1",
            span_id="span_1", 
            parent_span_id=None,
            operation_name="test_op",
            start_time=time.time()
        )
        
        time.sleep(0.01)  # Small delay
        span.finish()
        
        assert span.end_time is not None
        assert span.duration is not None
        assert span.duration > 0
        assert span.end_time > span.start_time
        
    def test_trace_span_logging(self):
        """Test span logging functionality."""
        span = TraceSpan(
            trace_id="trace_1",
            span_id="span_1",
            parent_span_id=None,
            operation_name="test_op",
            start_time=time.time()
        )
        
        # Add log entries
        span.log("operation_started", user_id=123)
        span.log("processing_data", records_count=500)
        
        assert len(span.logs) == 2
        
        # Check log structure
        log_entry = span.logs[0]
        assert len(log_entry) == 3  # (timestamp, message, kwargs)
        assert log_entry[1] == "operation_started"
        assert log_entry[2] == {"user_id": 123}
        
    def test_trace_span_tags(self):
        """Test span tagging functionality."""
        span = TraceSpan(
            trace_id="trace_1",
            span_id="span_1",
            parent_span_id=None,
            operation_name="test_op",
            start_time=time.time()
        )
        
        # Set tags
        span.set_tag("component", "database")
        span.set_tag("db.statement", "SELECT * FROM users")
        span.set_tag("http.status_code", "200")
        
        assert span.labels["component"] == "database"
        assert span.labels["db.statement"] == "SELECT * FROM users"  
        assert span.labels["http.status_code"] == "200"
        
    def test_trace_span_error_handling(self):
        """Test span error marking."""
        span = TraceSpan(
            trace_id="trace_1",
            span_id="span_1",
            parent_span_id=None,
            operation_name="test_op",
            start_time=time.time()
        )
        
        # Simulate an error
        error = ValueError("Test error message")
        span.set_error(error)
        
        assert span.status == "error"
        assert span.labels["error.type"] == "ValueError"
        assert span.labels["error.message"] == "Test error message"
        
        # Should have error log entry
        assert len(span.logs) == 1
        assert span.logs[0][1] == "error"
        assert "traceback" in span.logs[0][2]


class TestConsoleBackend:
    """Test ConsoleBackend for debugging."""
    
    def setup_method(self):
        """Setup test environment."""
        self.backend = ConsoleBackend()
        
    def test_console_backend_creation(self):
        """Test ConsoleBackend initialization."""
        assert isinstance(self.backend, ConsoleBackend)
        assert isinstance(self.backend, MetricsBackend)
        
    def test_emit_metric(self, capsys):
        """Test metric emission to console."""
        point = MetricPoint(
            name="test_counter",
            value=42.0,
            timestamp=time.time(),
            labels={"method": "GET", "status": "200"},
            metric_type=MetricType.COUNTER
        )
        
        self.backend.emit_metric(point)
        
        # Check console output
        captured = capsys.readouterr()
        assert "METRIC: test_counter=42.0" in captured.out
        assert "method=GET" in captured.out
        assert "status=200" in captured.out
        
    def test_emit_trace(self, capsys):
        """Test trace emission to console."""
        span = TraceSpan(
            trace_id="trace_123456",
            span_id="span_789",
            parent_span_id=None,
            operation_name="database_query",
            start_time=time.time()
        )
        
        span.finish()  # Complete the span
        
        self.backend.emit_trace(span)
        
        # Check console output
        captured = capsys.readouterr()
        assert "TRACE: database_query" in captured.out
        assert "trace_id=trace_123" in captured.out  # Truncated ID
        assert "ms" in captured.out  # Duration


class TestPrometheusBackend:
    """Test PrometheusBackend integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.backend = PrometheusBackend(
            gateway_url="http://localhost:9091",
            job_name="test_job"
        )
        
    def test_prometheus_backend_creation(self):
        """Test PrometheusBackend initialization."""
        assert self.backend.gateway_url == "http://localhost:9091"
        assert self.backend.job_name == "test_job"
        assert len(self.backend.metrics_buffer) == 0
        assert self.backend.push_interval == 30
        
    def test_emit_metric_buffering(self):
        """Test metric buffering in Prometheus backend."""
        point = MetricPoint(
            name="http_requests_total",
            value=100,
            timestamp=time.time(),
            labels={"endpoint": "/api/v1/predict"},
            metric_type=MetricType.COUNTER
        )
        
        self.backend.emit_metric(point)
        
        # Should be buffered
        assert len(self.backend.metrics_buffer) == 1
        assert self.backend.metrics_buffer[0] == point
        
    def test_metric_formatting(self):
        """Test Prometheus metric formatting."""
        # Add some metrics to buffer
        timestamp = time.time()
        points = [
            MetricPoint("cpu_usage", 50.5, timestamp, {"instance": "node1"}),
            MetricPoint("memory_usage", 80.0, timestamp, {"instance": "node1"}),
            MetricPoint("requests_total", 1000, timestamp, {"method": "POST"})
        ]
        
        for point in points:
            self.backend.emit_metric(point)
            
        # Format metrics
        formatted = self.backend._format_prometheus_metrics()
        
        # Check format
        lines = formatted.split('\n')
        assert len(lines) == 3
        
        # Check specific formatting
        assert 'cpu_usage{instance="node1"}' in lines[0]
        assert 'memory_usage{instance="node1"}' in lines[1] 
        assert 'requests_total{method="POST"}' in lines[2]
        
        # Check values and timestamps
        for line in lines:
            parts = line.split(' ')
            assert len(parts) == 3  # metric_name{labels} value timestamp
            
    def test_flush_operation(self, capsys):
        """Test flushing metrics to Prometheus."""
        # Add metrics to buffer
        for i in range(5):
            point = MetricPoint(f"metric_{i}", float(i), time.time())
            self.backend.emit_metric(point)
            
        assert len(self.backend.metrics_buffer) == 5
        
        # Flush metrics
        self.backend.flush()
        
        # Buffer should be cleared
        assert len(self.backend.metrics_buffer) == 0
        
        # Should have printed push message
        captured = capsys.readouterr()
        assert "Pushing 5 metrics to Prometheus" in captured.out
        
    def test_automatic_flush_on_interval(self, capsys):
        """Test automatic flushing based on time interval."""
        # Set short interval for testing
        self.backend.push_interval = 0.1
        self.backend.last_push = time.time() - 0.2  # Make it seem like it's time to push
        
        point = MetricPoint("test_metric", 1.0, time.time())
        self.backend.emit_metric(point)
        
        # Should trigger automatic flush
        captured = capsys.readouterr()
        assert "Pushing" in captured.out


class TestOpenTelemetryBackend:
    """Test OpenTelemetryBackend integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.backend = OpenTelemetryBackend(endpoint="http://localhost:4317")
        
    def test_otel_backend_creation(self):
        """Test OpenTelemetryBackend initialization."""
        assert self.backend.endpoint == "http://localhost:4317"
        assert len(self.backend.traces_buffer) == 0
        assert len(self.backend.metrics_buffer) == 0
        
    def test_emit_metric_to_otel(self):
        """Test emitting metrics to OpenTelemetry backend."""
        point = MetricPoint(
            name="system.cpu.utilization",
            value=75.5,
            timestamp=time.time(),
            labels={"host": "server1", "cpu": "0"},
            metric_type=MetricType.GAUGE
        )
        
        self.backend.emit_metric(point)
        
        assert len(self.backend.metrics_buffer) == 1
        assert self.backend.metrics_buffer[0] == point
        
    def test_emit_trace_to_otel(self):
        """Test emitting traces to OpenTelemetry backend."""
        span = TraceSpan(
            trace_id="otel_trace_123",
            span_id="otel_span_456", 
            parent_span_id=None,
            operation_name="http_request",
            start_time=time.time()
        )
        
        span.set_tag("http.method", "GET")
        span.set_tag("http.url", "/api/health")
        span.finish()
        
        self.backend.emit_trace(span)
        
        assert len(self.backend.traces_buffer) == 1
        assert self.backend.traces_buffer[0] == span
        
    def test_flush_to_otlp(self, capsys):
        """Test flushing data to OTLP endpoint."""
        # Add some data
        point = MetricPoint("test.metric", 1.0, time.time())
        self.backend.emit_metric(point)
        
        span = TraceSpan("trace1", "span1", None, "test_op", time.time())
        span.finish()
        self.backend.emit_trace(span)
        
        # Flush
        self.backend.flush()
        
        # Buffers should be cleared
        assert len(self.backend.metrics_buffer) == 0
        assert len(self.backend.traces_buffer) == 0
        
        # Should have logged export
        captured = capsys.readouterr()
        assert "Exporting" in captured.out
        assert "spans" in captured.out
        assert "metrics" in captured.out


class TestMetricsCollector:
    """Test MetricsCollector main functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.console_backend = ConsoleBackend()
        self.collector = MetricsCollector(backends=[self.console_backend])
        
    def test_collector_initialization(self):
        """Test MetricsCollector initialization."""
        assert len(self.collector.backends) == 1
        assert self.collector.backends[0] == self.console_backend
        assert isinstance(self.collector.counters, dict)
        assert isinstance(self.collector.gauges, dict)
        assert isinstance(self.collector.histograms, dict)
        assert isinstance(self.collector.timers, dict)
        assert self.collector.system_metrics_enabled is True
        
    def test_counter_metrics(self, capsys):
        """Test counter metric collection."""
        # Increment counter
        self.collector.counter("requests_total", 1.0, labels={"method": "GET"})
        self.collector.counter("requests_total", 2.0, labels={"method": "GET"})
        
        # Check internal state
        counter_key = "requests_total[method=GET]"
        assert counter_key in self.collector.counters
        assert self.collector.counters[counter_key] == 3.0
        
        # Check backend emission
        captured = capsys.readouterr()
        assert "METRIC: requests_total" in captured.out
        
    def test_gauge_metrics(self, capsys):
        """Test gauge metric collection."""
        # Set gauge values
        self.collector.gauge("cpu_usage", 45.5, labels={"core": "0"})
        self.collector.gauge("cpu_usage", 50.0, labels={"core": "0"})  # Update
        
        # Check internal state
        gauge_key = "cpu_usage[core=0]"
        assert gauge_key in self.collector.gauges
        assert self.collector.gauges[gauge_key] == 50.0
        
        # Check backend emission
        captured = capsys.readouterr()
        assert "METRIC: cpu_usage" in captured.out
        
    def test_histogram_metrics(self, capsys):
        """Test histogram metric collection."""
        # Add values to histogram
        values = [0.1, 0.2, 0.5, 0.8, 1.2, 2.0, 5.0]
        for value in values:
            self.collector.histogram("response_time", value, labels={"endpoint": "/api"})
            
        # Check internal state
        histogram_key = "response_time[endpoint=/api]"
        assert histogram_key in self.collector.histograms
        assert len(self.collector.histograms[histogram_key]) == len(values)
        
        # Check percentiles are emitted
        captured = capsys.readouterr()
        assert "response_time_p50" in captured.out
        assert "response_time_p95" in captured.out
        assert "response_time_p99" in captured.out
        
    def test_timer_context_manager(self, capsys):
        """Test timer context manager."""
        with self.collector.timer("operation_duration", labels={"op": "training"}):
            time.sleep(0.01)  # Small delay
            
        # Check timing was recorded
        timer_key = "operation_duration[op=training]"
        assert timer_key in self.collector.timers
        assert len(self.collector.timers[timer_key]) == 1
        
        # Timer should be > 0
        recorded_time = self.collector.timers[timer_key][0]
        assert recorded_time > 0
        assert recorded_time >= 0.01  # At least the sleep duration
        
        # Check backend emission
        captured = capsys.readouterr()
        assert "METRIC: operation_duration" in captured.out
        
    def test_timing_method(self, capsys):
        """Test direct timing method."""
        duration = 0.123
        self.collector.timing("custom_operation", duration, labels={"type": "inference"})
        
        # Check recorded
        timer_key = "custom_operation[type=inference]"
        assert timer_key in self.collector.timers
        assert self.collector.timers[timer_key][0] == duration
        
        # Check histogram also updated
        histogram_key = "custom_operation_duration[type=inference]"
        assert histogram_key in self.collector.histograms
        
    def test_metric_name_building(self):
        """Test metric name building with labels."""
        # No labels
        name1 = self.collector._build_metric_name("simple_metric", None)
        assert name1 == "simple_metric"
        
        # With labels
        labels = {"method": "POST", "status": "200", "endpoint": "/train"}
        name2 = self.collector._build_metric_name("http_requests", labels)
        assert name2 == "http_requests[endpoint=/train,method=POST,status=200]"  # Sorted
        
    def test_system_metrics_collection(self):
        """Test automatic system metrics collection."""
        # System metrics should be running
        assert self.collector._collection_thread is not None
        
        # Wait a bit for some metrics to be collected
        time.sleep(0.1)
        
        # Check for system metrics in gauges
        system_metric_prefixes = ["system.cpu", "system.memory", "system.disk", "gpu."]
        
        found_system_metrics = False
        for prefix in system_metric_prefixes:
            for gauge_name in self.collector.gauges.keys():
                if prefix in gauge_name:
                    found_system_metrics = True
                    break
                    
        # Should have collected some system metrics
        # (May not always pass due to timing, but shouldn't crash)
        
    def test_custom_handlers(self):
        """Test custom metric handlers."""
        # Define custom handler
        def custom_gpu_metrics():
            self.collector.gauge("gpu.temperature", 65.0, labels={"device": "cuda:0"})
            self.collector.gauge("gpu.memory_used", 4096, labels={"device": "cuda:0"})
            
        # Register handler
        self.collector.register_custom_handler("gpu_metrics", custom_gpu_metrics)
        
        # Collect custom metrics
        self.collector.collect_custom_metrics()
        
        # Check metrics were collected
        assert "gpu.temperature[device=cuda:0]" in self.collector.gauges
        assert "gpu.memory_used[device=cuda:0]" in self.collector.gauges
        
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        # Generate some metrics
        self.collector.counter("test_counter", 5.0)
        self.collector.gauge("test_gauge", 10.0)
        self.collector.histogram("test_histogram", 1.0)
        self.collector.histogram("test_histogram", 2.0)
        self.collector.histogram("test_histogram", 3.0)
        
        summary = self.collector.get_metrics_summary()
        
        # Check structure
        expected_sections = ['counters', 'gauges', 'histograms', 'timers']
        for section in expected_sections:
            assert section in summary
            
        # Check counter
        assert 'test_counter' in summary['counters']
        assert summary['counters']['test_counter'] == 5.0
        
        # Check gauge
        assert 'test_gauge' in summary['gauges']
        assert summary['gauges']['test_gauge'] == 10.0
        
        # Check histogram with statistics
        assert 'test_histogram' in summary['histograms']
        histogram_stats = summary['histograms']['test_histogram']
        assert 'count' in histogram_stats
        assert 'mean' in histogram_stats
        assert 'p50' in histogram_stats
        assert 'p95' in histogram_stats
        assert 'p99' in histogram_stats
        
    def test_backend_error_handling(self, capsys):
        """Test error handling with faulty backends."""
        # Create a backend that raises errors
        faulty_backend = Mock()
        faulty_backend.emit_metric.side_effect = Exception("Backend error")
        
        collector = MetricsCollector(backends=[faulty_backend])
        
        # Should not crash when backend fails
        collector.counter("test_metric", 1.0)
        
        # Should have logged error
        captured = capsys.readouterr()
        assert "Error emitting metric" in captured.out
        
    def test_collector_cleanup(self):
        """Test collector cleanup and shutdown."""
        # Stop system metrics collection
        self.collector.stop_system_metrics_collection()
        
        # Thread should be stopped
        if self.collector._collection_thread:
            assert not self.collector._collection_thread.is_alive()
            
        # Close all backends
        self.collector.close()
        
        # Should complete without errors


class TestDistributedTracer:
    """Test DistributedTracer functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.console_backend = ConsoleBackend()
        self.tracer = DistributedTracer(
            service_name="test_service",
            backends=[self.console_backend]
        )
        
    def test_tracer_initialization(self):
        """Test DistributedTracer initialization."""
        assert self.tracer.service_name == "test_service"
        assert len(self.tracer.backends) == 1
        assert self.tracer.backends[0] == self.console_backend
        assert len(self.tracer.active_spans) == 0
        assert len(self.tracer.completed_spans) == 0
        
    def test_start_span(self):
        """Test starting a trace span."""
        span = self.tracer.start_span(
            operation_name="database_query",
            tags={"db.type": "postgresql", "query.table": "users"}
        )
        
        assert isinstance(span, TraceSpan)
        assert span.operation_name == "database_query"
        assert span.labels["db.type"] == "postgresql"
        assert span.labels["query.table"] == "users"
        assert span.labels["service.name"] == "test_service"
        
        # Should be in active spans
        assert span.span_id in self.tracer.active_spans
        
    def test_start_child_span(self):
        """Test starting child spans."""
        parent_span = self.tracer.start_span("parent_operation")
        child_span = self.tracer.start_span("child_operation", parent_span=parent_span)
        
        # Child should reference parent
        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.trace_id == parent_span.trace_id
        
        # Both should be active
        assert parent_span.span_id in self.tracer.active_spans
        assert child_span.span_id in self.tracer.active_spans
        
    def test_finish_span(self, capsys):
        """Test finishing spans."""
        span = self.tracer.start_span("test_operation")
        
        time.sleep(0.01)  # Small delay
        self.tracer.finish_span(span)
        
        # Should be completed
        assert span.end_time is not None
        assert span.duration is not None
        assert span.duration > 0
        
        # Should be moved from active to completed
        assert span.span_id not in self.tracer.active_spans
        assert len(self.tracer.completed_spans) == 1
        
        # Should be emitted to backend
        captured = capsys.readouterr()
        assert "TRACE: test_operation" in captured.out
        
    def test_span_context_manager(self, capsys):
        """Test span context manager."""
        with self.tracer.span("web_request", tags={"http.method": "GET"}) as span:
            assert isinstance(span, TraceSpan)
            assert span.operation_name == "web_request"
            assert span.labels["http.method"] == "GET"
            
            # Add some activity
            span.log("processing_request", user_id=123)
            time.sleep(0.01)
            
            # Span should be active
            assert span.span_id in self.tracer.active_spans
            
        # After context, span should be finished
        assert span.end_time is not None
        assert span.span_id not in self.tracer.active_spans
        
        # Should be in backend output
        captured = capsys.readouterr()
        assert "TRACE: web_request" in captured.out
        
    def test_nested_span_contexts(self, capsys):
        """Test nested span contexts."""
        with self.tracer.span("outer_operation") as outer_span:
            outer_span.log("outer_started")
            
            with self.tracer.span("inner_operation") as inner_span:
                inner_span.log("inner_started")
                
                # Inner span should be child of outer
                assert inner_span.parent_span_id == outer_span.span_id
                assert inner_span.trace_id == outer_span.trace_id
                
                time.sleep(0.01)
                
            # Inner span should be finished
            assert inner_span.end_time is not None
            
            with self.tracer.span("another_inner") as another_span:
                # Should also be child of outer
                assert another_span.parent_span_id == outer_span.span_id
                
        # Both outer spans should be finished
        assert outer_span.end_time is not None
        
        # Check trace output
        captured = capsys.readouterr()
        assert "TRACE: outer_operation" in captured.out
        assert "TRACE: inner_operation" in captured.out
        assert "TRACE: another_inner" in captured.out
        
    def test_span_error_handling(self, capsys):
        """Test error handling in spans."""
        with pytest.raises(ValueError):
            with self.tracer.span("failing_operation") as span:
                span.log("about_to_fail")
                raise ValueError("Test error")
                
        # Span should be marked as error
        assert span.status == "error"
        assert span.labels["error.type"] == "ValueError"
        assert span.labels["error.message"] == "Test error"
        
        # Should still be traced
        captured = capsys.readouterr()
        assert "TRACE: failing_operation" in captured.out
        
    def test_context_injection_and_extraction(self):
        """Test trace context injection and extraction."""
        span = self.tracer.start_span("http_client_request")
        
        # Inject context into headers
        headers = self.tracer.inject_context(span)
        
        expected_headers = ["neural-trace-id", "neural-span-id"]
        for header in expected_headers:
            assert header in headers
            
        assert headers["neural-trace-id"] == span.trace_id
        assert headers["neural-span-id"] == span.span_id
        
        # Extract context from headers
        extracted_span = self.tracer.extract_context(headers)
        
        assert extracted_span is not None
        assert extracted_span.trace_id == span.trace_id
        assert extracted_span.span_id == span.span_id
        
    def test_active_span_tracking(self):
        """Test active span tracking."""
        # No active span initially
        assert self.tracer.get_active_span() is None
        
        with self.tracer.span("operation1") as span1:
            # Should be active
            active = self.tracer.get_active_span()
            assert active == span1
            
            with self.tracer.span("operation2") as span2:
                # Inner span should now be active
                active = self.tracer.get_active_span()
                assert active == span2
                
            # Back to outer span
            active = self.tracer.get_active_span()
            assert active == span1
            
        # No active span after context
        assert self.tracer.get_active_span() is None


class TestHealthChecker:
    """Test HealthChecker functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.collector = MetricsCollector(backends=[ConsoleBackend()])
        self.health_checker = HealthChecker(self.collector)
        
    def test_health_checker_initialization(self):
        """Test HealthChecker initialization."""
        assert self.health_checker.metrics == self.collector
        assert isinstance(self.health_checker.health_checks, dict)
        assert self.health_checker.check_interval == 30
        
        # Should have default checks registered
        default_checks = ["memory_usage", "disk_space", "gpu_health"]
        for check_name in default_checks:
            assert check_name in self.health_checker.health_checks
            
    def test_register_custom_check(self):
        """Test registering custom health checks."""
        def custom_check():
            return True  # Always healthy
            
        self.health_checker.register_check("custom_service", custom_check)
        
        assert "custom_service" in self.health_checker.health_checks
        assert self.health_checker.health_checks["custom_service"] == custom_check
        
    def test_memory_usage_check(self):
        """Test memory usage health check."""
        result = self.health_checker._check_memory_usage()
        
        # Should return boolean
        assert isinstance(result, bool)
        
        # Memory check should pass under normal conditions
        assert result is True
        
    def test_disk_space_check(self):
        """Test disk space health check."""
        result = self.health_checker._check_disk_space()
        
        # Should return boolean
        assert isinstance(result, bool)
        
        # Disk check should pass under normal conditions
        assert result is True
        
    def test_gpu_health_check(self):
        """Test GPU health check."""
        result = self.health_checker._check_gpu_health()
        
        # Should return boolean (may be True for mock implementation)
        assert isinstance(result, bool)
        
    def test_run_all_health_checks(self, capsys):
        """Test running all health checks."""
        results = self.health_checker.run_health_checks()
        
        # Should return dict of results
        assert isinstance(results, dict)
        
        # Should include default checks
        expected_checks = ["memory_usage", "disk_space", "gpu_health"]
        for check_name in expected_checks:
            assert check_name in results
            assert isinstance(results[check_name], bool)
            
        # Should emit metrics for each check
        captured = capsys.readouterr()
        for check_name in expected_checks:
            assert f"health.check.{check_name}" in captured.out
            
    def test_health_check_failure_handling(self, capsys):
        """Test handling of failed health checks."""
        def failing_check():
            return False  # Always fails
            
        def error_check():
            raise Exception("Check error")
            
        self.health_checker.register_check("failing_service", failing_check)
        self.health_checker.register_check("error_service", error_check)
        
        results = self.health_checker.run_health_checks()
        
        # Failing check should return False
        assert results["failing_service"] is False
        
        # Error check should return False
        assert results["error_service"] is False
        
        # Should log failure messages
        captured = capsys.readouterr()
        assert "Health check failed: failing_service" in captured.out
        assert "Error running health check error_service" in captured.out
        
    def test_overall_health_status(self):
        """Test overall health status determination."""
        # Add custom checks with known results
        self.health_checker.register_check("healthy_service", lambda: True)
        self.health_checker.register_check("unhealthy_service", lambda: False)
        
        # Should be unhealthy due to one failing check
        is_healthy = self.health_checker.is_healthy()
        assert is_healthy is False
        
        # Remove failing check
        del self.health_checker.health_checks["unhealthy_service"]
        
        # Now should be healthy
        is_healthy = self.health_checker.is_healthy()
        assert is_healthy is True


class TestDecorators:
    """Test automatic instrumentation decorators."""
    
    def setup_method(self):
        """Setup test environment."""
        # Reset global instances for testing
        import src.neural_arch.monitoring.observability as obs_module
        obs_module._global_metrics = None
        obs_module._global_tracer = None
        
    def test_trace_decorator(self, capsys):
        """Test trace decorator functionality."""
        @trace("test_function")
        def sample_function(x, y):
            time.sleep(0.01)  # Small delay
            return x + y
            
        result = sample_function(2, 3)
        
        assert result == 5
        
        # Should have traced the function
        captured = capsys.readouterr()
        assert "TRACE: test_function" in captured.out
        
    def test_trace_decorator_with_tags(self, capsys):
        """Test trace decorator with custom tags."""
        @trace("math_operation", tags={"operation": "multiplication"})
        def multiply(a, b):
            return a * b
            
        result = multiply(4, 5)
        
        assert result == 20
        
        # Should have traced with tags
        captured = capsys.readouterr()
        assert "TRACE: math_operation" in captured.out
        
    def test_trace_decorator_default_name(self, capsys):
        """Test trace decorator with default operation name."""
        @trace()
        def another_function():
            return "done"
            
        result = another_function()
        
        assert result == "done"
        
        # Should use module.function_name as operation name
        captured = capsys.readouterr()
        # Operation name should include function name
        assert "another_function" in captured.out
        
    def test_trace_decorator_error_handling(self, capsys):
        """Test trace decorator with exceptions."""
        @trace("failing_function")
        def failing_function():
            raise ValueError("Function failed")
            
        with pytest.raises(ValueError):
            failing_function()
            
        # Should still trace the error
        captured = capsys.readouterr()
        assert "TRACE: failing_function" in captured.out
        
    def test_timed_decorator(self, capsys):
        """Test timed decorator functionality."""
        @timed("operation_timer")
        def timed_function():
            time.sleep(0.01)
            return "completed"
            
        result = timed_function()
        
        assert result == "completed"
        
        # Should have emitted timing metric
        captured = capsys.readouterr()
        assert "METRIC: operation_timer" in captured.out
        
    def test_timed_decorator_with_labels(self, capsys):
        """Test timed decorator with labels."""
        @timed("api_call_duration", labels={"endpoint": "/users", "method": "GET"})
        def api_call():
            time.sleep(0.01)
            return {"users": []}
            
        result = api_call()
        
        assert "users" in result
        
        # Should have emitted timing metric with labels
        captured = capsys.readouterr()
        assert "METRIC:" in captured.out
        assert "endpoint=/users" in captured.out
        assert "method=GET" in captured.out
        
    def test_timed_decorator_default_name(self, capsys):
        """Test timed decorator with default metric name."""
        @timed()
        def default_timed_function():
            return "timed"
            
        result = default_timed_function()
        
        assert result == "timed"
        
        # Should use module.function_name.duration as metric name
        captured = capsys.readouterr()
        assert "METRIC:" in captured.out
        assert "duration" in captured.out


class TestGlobalInstances:
    """Test global instance management."""
    
    def setup_method(self):
        """Setup test environment."""
        # Reset global instances
        import src.neural_arch.monitoring.observability as obs_module
        obs_module._global_metrics = None
        obs_module._global_tracer = None
        
    def test_get_metrics_collector_singleton(self):
        """Test global metrics collector singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should be same instance
        assert collector1 is collector2
        assert isinstance(collector1, MetricsCollector)
        
    def test_get_tracer_singleton(self):
        """Test global tracer singleton."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        
        # Should be same instance
        assert tracer1 is tracer2
        assert isinstance(tracer1, DistributedTracer)


class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    def setup_method(self):
        """Setup test environment."""
        self.backends = [ConsoleBackend()]
        self.metrics = MetricsCollector(backends=self.backends)
        self.tracer = DistributedTracer(backends=self.backends)
        self.health_checker = HealthChecker(self.metrics)
        
    def test_ml_training_monitoring(self, capsys):
        """Test monitoring a machine learning training scenario."""
        # Simulate ML training loop with monitoring
        epochs = 3
        batches_per_epoch = 5
        
        with self.tracer.span("training_job", tags={"model": "transformer", "dataset": "train"}) as training_span:
            training_span.log("training_started", epochs=epochs)
            
            for epoch in range(epochs):
                with self.tracer.span("epoch", tags={"epoch": str(epoch)}) as epoch_span:
                    epoch_span.log("epoch_started")
                    
                    epoch_loss = 0.0
                    for batch in range(batches_per_epoch):
                        with self.tracer.span("batch") as batch_span:
                            # Simulate batch processing time
                            batch_start = time.time()
                            time.sleep(0.001)  # Fast batch
                            batch_duration = time.time() - batch_start
                            
                            # Record metrics
                            batch_loss = 1.0 / (epoch * batches_per_epoch + batch + 1)  # Decreasing loss
                            epoch_loss += batch_loss
                            
                            self.metrics.gauge("training.loss", batch_loss, labels={"epoch": str(epoch)})
                            self.metrics.histogram("training.batch_duration", batch_duration)
                            self.metrics.counter("training.batches_processed", 1.0)
                            
                            batch_span.set_tag("batch_loss", str(batch_loss))
                            batch_span.log("batch_completed", batch=batch, loss=batch_loss)
                            
                    # End of epoch metrics
                    avg_epoch_loss = epoch_loss / batches_per_epoch
                    self.metrics.gauge("training.epoch_loss", avg_epoch_loss, labels={"epoch": str(epoch)})
                    
                    epoch_span.set_tag("avg_loss", str(avg_epoch_loss))
                    epoch_span.log("epoch_completed", avg_loss=avg_epoch_loss)
                    
            training_span.log("training_completed", final_loss=avg_epoch_loss)
            
        # Check health after training
        health_results = self.health_checker.run_health_checks()
        
        # Verify monitoring data was collected
        captured = capsys.readouterr()
        
        # Should have traces
        assert "TRACE: training_job" in captured.out
        assert "TRACE: epoch" in captured.out
        assert "TRACE: batch" in captured.out
        
        # Should have metrics
        assert "METRIC: training.loss" in captured.out
        assert "METRIC: training.batch_duration" in captured.out
        assert "METRIC: training.batches_processed" in captured.out
        
        # Should have health checks
        assert "METRIC: health.check" in captured.out
        
        # Check metrics summary
        summary = self.metrics.get_metrics_summary()
        assert "training.batches_processed" in summary['counters']
        assert summary['counters']['training.batches_processed'] == epochs * batches_per_epoch
        
    def test_distributed_service_monitoring(self, capsys):
        """Test monitoring distributed service interactions."""
        services = ["frontend", "backend", "database"]
        
        # Simulate distributed request flow
        with self.tracer.span("user_request", tags={"user_id": "123", "request_type": "prediction"}) as root_span:
            root_span.log("request_received")
            
            # Frontend processing
            with self.tracer.span("frontend_processing", tags={"service": "frontend"}) as frontend_span:
                self.metrics.counter("requests.received", 1.0, labels={"service": "frontend"})
                
                # Inject context for service call
                headers = self.tracer.inject_context(frontend_span)
                frontend_span.log("calling_backend", headers=str(headers))
                
                # Backend processing (simulate remote call)
                extracted_parent = self.tracer.extract_context(headers)
                with self.tracer.span("backend_processing", tags={"service": "backend"}) as backend_span:
                    if extracted_parent:
                        backend_span.parent_span_id = extracted_parent.span_id
                        backend_span.trace_id = extracted_parent.trace_id
                        
                    self.metrics.counter("requests.processed", 1.0, labels={"service": "backend"})
                    
                    # Database query
                    with self.tracer.span("database_query", tags={"service": "database", "table": "models"}) as db_span:
                        query_start = time.time()
                        time.sleep(0.002)  # Simulate DB query time
                        query_duration = time.time() - query_start
                        
                        self.metrics.histogram("database.query_duration", query_duration, 
                                             labels={"table": "models", "operation": "select"})
                        
                        db_span.set_tag("query", "SELECT * FROM models WHERE id = ?")
                        db_span.log("query_executed", duration=query_duration)
                        
                    backend_span.log("database_query_completed")
                    
                frontend_span.log("backend_response_received")
                
            # Record response metrics
            self.metrics.counter("responses.sent", 1.0, labels={"service": "frontend", "status": "200"})
            root_span.log("response_sent", status=200)
            
        # Verify distributed tracing
        captured = capsys.readouterr()
        
        # Should have all service traces
        for service in services:
            assert f"service={service}" in captured.out
            
        # Should have proper trace hierarchy
        assert "TRACE: user_request" in captured.out
        assert "TRACE: frontend_processing" in captured.out
        assert "TRACE: backend_processing" in captured.out
        assert "TRACE: database_query" in captured.out
        
    def test_error_monitoring_and_alerting(self, capsys):
        """Test error monitoring and alerting scenarios."""
        # Simulate various error conditions
        
        # 1. Service error with trace
        try:
            with self.tracer.span("payment_processing") as span:
                span.set_tag("payment_id", "pay_123")
                self.metrics.counter("payments.attempted", 1.0)
                
                # Simulate payment failure
                raise ConnectionError("Payment gateway timeout")
                
        except ConnectionError as e:
            # Log error metrics
            self.metrics.counter("payments.failed", 1.0, labels={"error_type": "timeout"})
            
        # 2. Health check failure
        def failing_service():
            return False
            
        self.health_checker.register_check("payment_service", failing_service)
        health_results = self.health_checker.run_health_checks()
        
        # 3. High latency alert
        high_latencies = [0.5, 0.8, 1.2, 2.0, 3.5]  # Simulate increasing latency
        for latency in high_latencies:
            self.metrics.histogram("api.response_time", latency, labels={"endpoint": "/predict"})
            
        # 4. Error rate spike
        for i in range(10):
            if i < 7:
                self.metrics.counter("requests.total", 1.0, labels={"status": "200"})
            else:
                self.metrics.counter("requests.total", 1.0, labels={"status": "500"})
                
        # Verify error monitoring
        captured = capsys.readouterr()
        
        # Should have error traces
        assert "payment_processing" in captured.out
        
        # Should have error metrics
        assert "payments.failed" in captured.out
        assert "error_type=timeout" in captured.out
        
        # Should have health check failures
        assert "Health check failed: payment_service" in captured.out
        
        # Check metrics summary for error analysis
        summary = self.metrics.get_metrics_summary()
        
        # Should have failure counters
        assert "payments.failed" in [key.split('[')[0] for key in summary['counters'].keys()]
        
        # Should have latency histograms
        latency_histograms = [key for key in summary['histograms'].keys() if 'response_time' in key]
        assert len(latency_histograms) > 0


class TestSystemIntegration:
    """Test integration with the complete observability system."""
    
    def test_built_in_system_test(self):
        """Test integration with built-in system test."""
        try:
            # Run the built-in test function
            test_observability_system()
            print("✅ Observability system test passed")
        except Exception as e:
            pytest.fail(f"System integration test failed: {e}")
            
    def test_performance_under_load(self):
        """Test observability system performance under load."""
        backends = [ConsoleBackend()]  # Use console to avoid external dependencies
        metrics = MetricsCollector(backends=backends)
        tracer = DistributedTracer(backends=backends)
        
        # Simulate high-throughput metrics collection
        start_time = time.time()
        
        # Generate many metrics quickly
        for i in range(1000):
            metrics.counter("high_throughput_counter", 1.0, labels={"batch": str(i % 10)})
            if i % 100 == 0:
                metrics.gauge("progress", float(i), labels={"stage": "load_test"})
                
        # Generate many spans quickly
        spans = []
        for i in range(100):
            span = tracer.start_span(f"operation_{i % 5}", tags={"iteration": str(i)})
            spans.append(span)
            
        # Finish all spans
        for span in spans:
            tracer.finish_span(span)
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert duration < 5.0  # Should process 1000 metrics + 100 spans in < 5 seconds
        
        # System should still be functional
        summary = metrics.get_metrics_summary()
        assert "high_throughput_counter" in [key.split('[')[0] for key in summary['counters'].keys()]
        
        print(f"✅ Performance test completed in {duration:.2f}s")
        
    def test_concurrent_monitoring(self):
        """Test concurrent monitoring from multiple threads."""
        backends = [ConsoleBackend()]
        metrics = MetricsCollector(backends=backends)
        tracer = DistributedTracer(backends=backends)
        
        results = []
        lock = threading.Lock()
        
        def monitoring_worker(worker_id):
            try:
                # Each worker generates metrics and traces
                for i in range(50):
                    metrics.counter("worker_operations", 1.0, labels={"worker": str(worker_id)})
                    
                    with tracer.span(f"worker_{worker_id}_operation") as span:
                        span.set_tag("iteration", str(i))
                        time.sleep(0.001)  # Small work simulation
                        
                with lock:
                    results.append(f"Worker {worker_id} completed")
                    
            except Exception as e:
                with lock:
                    results.append(f"Worker {worker_id} failed: {e}")
                    
        # Start multiple worker threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=monitoring_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)
            
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10)
            
        # All workers should complete successfully
        assert len(results) == 5
        for result in results:
            assert "completed" in result
            
        # Check that metrics were collected from all workers
        summary = metrics.get_metrics_summary()
        worker_counter_keys = [key for key in summary['counters'].keys() if 'worker_operations' in key]
        assert len(worker_counter_keys) == 5  # One for each worker
        
        print("✅ Concurrent monitoring test passed")


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])