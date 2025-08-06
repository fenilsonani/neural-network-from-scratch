"""
Comprehensive test suite for fault tolerance and resilience system.
Tests all components of fault_tolerance.py for comprehensive coverage.

This module tests:
- FailureDetector with phi accrual failure detection
- CircuitBreaker pattern for service protection
- ElasticTrainingManager for dynamic node management
- MultiLevelCheckpointer with consistency guarantees
- SelfHealingSystem with automatic recovery
- ChaosEngineer for resilience testing
- All enum classes and data structures
- Distributed consensus and health monitoring
"""

import asyncio
import json
import os
import pickle
import tempfile
import threading
import time
from collections import deque
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import Dict, List, Any, Set

import numpy as np
import pytest

from src.neural_arch.reliability.fault_tolerance import (
    FailureType,
    NodeState,
    ServiceState,
    NodeInfo,
    FailureEvent,
    CircuitBreaker,
    FailureDetector,
    ElasticTrainingManager,
    MultiLevelCheckpointer,
    SelfHealingSystem,
    ChaosEngineer,
    test_fault_tolerance_system
)


class TestFailureType:
    """Test FailureType enumeration."""
    
    def test_failure_types(self):
        """Test all failure type values."""
        assert FailureType.NODE_FAILURE.value == "node_failure"
        assert FailureType.NETWORK_PARTITION.value == "network_partition"
        assert FailureType.STORAGE_FAILURE.value == "storage_failure"
        assert FailureType.MEMORY_ERROR.value == "memory_error"
        assert FailureType.TIMEOUT.value == "timeout"
        assert FailureType.CORRUPTION.value == "corruption"
        assert FailureType.RESOURCE_EXHAUSTION.value == "resource_exhaustion"
    
    def test_failure_type_count(self):
        """Test total number of failure types."""
        assert len(FailureType) == 7


class TestNodeState:
    """Test NodeState enumeration."""
    
    def test_node_states(self):
        """Test all node state values."""
        assert NodeState.HEALTHY.value == "healthy"
        assert NodeState.SUSPECTED.value == "suspected"
        assert NodeState.FAILED.value == "failed"
        assert NodeState.RECOVERING.value == "recovering"
        assert NodeState.REJOINING.value == "rejoining"
        assert NodeState.MAINTENANCE.value == "maintenance"
    
    def test_node_state_count(self):
        """Test total number of node states."""
        assert len(NodeState) == 6


class TestServiceState:
    """Test ServiceState enumeration."""
    
    def test_service_states(self):
        """Test all service state values."""
        assert ServiceState.CLOSED.value == "closed"
        assert ServiceState.OPEN.value == "open"
        assert ServiceState.HALF_OPEN.value == "half_open"
    
    def test_service_state_count(self):
        """Test total number of service states."""
        assert len(ServiceState) == 3


class TestNodeInfo:
    """Test NodeInfo data structure."""
    
    def test_node_info_creation(self):
        """Test NodeInfo creation with required fields."""
        node = NodeInfo(
            node_id="test_node",
            hostname="worker-1",
            port=29500,
            rank=0
        )
        
        assert node.node_id == "test_node"
        assert node.hostname == "worker-1"
        assert node.port == 29500
        assert node.rank == 0
        assert node.state == NodeState.HEALTHY  # Default
        assert node.failure_count == 0
        assert isinstance(node.capabilities, dict)
        assert node.load_factor == 1.0
        assert isinstance(node.last_heartbeat, float)
    
    def test_node_info_with_custom_values(self):
        """Test NodeInfo with custom values."""
        capabilities = {"gpu": True, "memory": "32GB"}
        custom_time = time.time() - 100
        
        node = NodeInfo(
            node_id="custom_node",
            hostname="gpu-worker",
            port=30000,
            rank=5,
            state=NodeState.SUSPECTED,
            last_heartbeat=custom_time,
            failure_count=3,
            capabilities=capabilities,
            load_factor=0.75
        )
        
        assert node.node_id == "custom_node"
        assert node.hostname == "gpu-worker"
        assert node.port == 30000
        assert node.rank == 5
        assert node.state == NodeState.SUSPECTED
        assert node.last_heartbeat == custom_time
        assert node.failure_count == 3
        assert node.capabilities == capabilities
        assert node.load_factor == 0.75
    
    def test_node_info_to_dict(self):
        """Test NodeInfo serialization to dictionary."""
        node = NodeInfo(
            node_id="serialize_test",
            hostname="test-host",
            port=25000,
            rank=2,
            state=NodeState.RECOVERING,
            failure_count=1,
            capabilities={"cpu_cores": 8},
            load_factor=0.6
        )
        
        node_dict = node.to_dict()
        
        assert node_dict["node_id"] == "serialize_test"
        assert node_dict["hostname"] == "test-host"
        assert node_dict["port"] == 25000
        assert node_dict["rank"] == 2
        assert node_dict["state"] == "recovering"
        assert node_dict["failure_count"] == 1
        assert node_dict["capabilities"] == {"cpu_cores": 8}
        assert node_dict["load_factor"] == 0.6
        assert "last_heartbeat" in node_dict


class TestFailureEvent:
    """Test FailureEvent data structure."""
    
    def test_failure_event_creation(self):
        """Test FailureEvent creation."""
        timestamp = time.time()
        event = FailureEvent(
            timestamp=timestamp,
            failure_type=FailureType.NODE_FAILURE,
            node_id="failed_node"
        )
        
        assert event.timestamp == timestamp
        assert event.failure_type == FailureType.NODE_FAILURE
        assert event.node_id == "failed_node"
        assert isinstance(event.details, dict)
        assert event.recovery_time is None
    
    def test_failure_event_with_details(self):
        """Test FailureEvent with details and recovery time."""
        details = {"error_code": 500, "message": "Connection lost"}
        recovery_time = 15.5
        
        event = FailureEvent(
            timestamp=time.time(),
            failure_type=FailureType.NETWORK_PARTITION,
            node_id="network_node",
            details=details,
            recovery_time=recovery_time
        )
        
        assert event.details == details
        assert event.recovery_time == recovery_time
    
    def test_failure_event_to_dict(self):
        """Test FailureEvent serialization to dictionary."""
        timestamp = time.time()
        details = {"severity": "high"}
        
        event = FailureEvent(
            timestamp=timestamp,
            failure_type=FailureType.STORAGE_FAILURE,
            node_id="storage_node",
            details=details,
            recovery_time=10.0
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["timestamp"] == timestamp
        assert event_dict["failure_type"] == "storage_failure"
        assert event_dict["node_id"] == "storage_node"
        assert event_dict["details"] == details
        assert event_dict["recovery_time"] == 10.0


class TestCircuitBreaker:
    """Test CircuitBreaker pattern implementation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_duration=5.0,
            success_threshold=2
        )
    
    def test_initialization(self):
        """Test CircuitBreaker initialization."""
        assert self.circuit_breaker.failure_threshold == 3
        assert self.circuit_breaker.timeout_duration == 5.0
        assert self.circuit_breaker.success_threshold == 2
        assert self.circuit_breaker.state == ServiceState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0
        assert self.circuit_breaker.last_failure_time == 0
        assert isinstance(self.circuit_breaker.lock, threading.Lock)
    
    @pytest.mark.asyncio
    async def test_successful_call_sync_function(self):
        """Test successful call to synchronous function."""
        def success_func(x, y):
            return x + y
        
        result = await self.circuit_breaker.call(success_func, 5, 10)
        assert result == 15
        assert self.circuit_breaker.state == ServiceState.CLOSED
        assert self.circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_successful_call_async_function(self):
        """Test successful call to asynchronous function."""
        async def async_success_func(x, y):
            return x * y
        
        result = await self.circuit_breaker.call(async_success_func, 3, 4)
        assert result == 12
        assert self.circuit_breaker.state == ServiceState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failure_accumulation(self):
        """Test failure accumulation leading to circuit opening."""
        def failing_func():
            raise ValueError("Test failure")
        
        # Accumulate failures
        for i in range(3):
            with pytest.raises(ValueError):
                await self.circuit_breaker.call(failing_func)
            
            # Should still be closed until threshold is reached
            if i < 2:
                assert self.circuit_breaker.state == ServiceState.CLOSED
            else:
                assert self.circuit_breaker.state == ServiceState.OPEN
        
        assert self.circuit_breaker.failure_count == 3
    
    @pytest.mark.asyncio
    async def test_circuit_open_behavior(self):
        """Test behavior when circuit is open."""
        def failing_func():
            raise ValueError("Test failure")
        
        # Trigger circuit opening
        for _ in range(3):
            with pytest.raises(ValueError):
                await self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.state == ServiceState.OPEN
        
        # Should fail fast when circuit is open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await self.circuit_breaker.call(lambda: "should not execute")
    
    @pytest.mark.asyncio
    async def test_half_open_transition(self):
        """Test transition from OPEN to HALF_OPEN state."""
        def failing_func():
            raise ValueError("Test failure")
        
        # Open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.state == ServiceState.OPEN
        
        # Wait for timeout and modify time to trigger half-open
        with patch('time.time', return_value=self.circuit_breaker.last_failure_time + 10):
            def success_func():
                return "success"
            
            result = await self.circuit_breaker.call(success_func)
            assert result == "success"
            # After first success in half-open, should still be half-open
    
    @pytest.mark.asyncio
    async def test_half_open_to_closed_recovery(self):
        """Test recovery from HALF_OPEN to CLOSED state."""
        # Manually set to half-open state
        self.circuit_breaker.state = ServiceState.HALF_OPEN
        self.circuit_breaker.success_count = 0
        
        def success_func():
            return "success"
        
        # Need success_threshold successes to close circuit
        for i in range(2):
            result = await self.circuit_breaker.call(success_func)
            assert result == "success"
            
            if i < 1:
                assert self.circuit_breaker.state == ServiceState.HALF_OPEN
            else:
                assert self.circuit_breaker.state == ServiceState.CLOSED
                assert self.circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_half_open_failure_back_to_open(self):
        """Test failure in HALF_OPEN state returns to OPEN."""
        self.circuit_breaker.state = ServiceState.HALF_OPEN
        
        def failing_func():
            raise ValueError("Still failing")
        
        with pytest.raises(ValueError):
            await self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.state == ServiceState.OPEN
    
    @pytest.mark.asyncio
    async def test_success_reduces_failure_count(self):
        """Test that success reduces failure count in CLOSED state."""
        def failing_func():
            raise ValueError("Test failure")
        
        def success_func():
            return "success"
        
        # Add some failures
        for _ in range(2):
            with pytest.raises(ValueError):
                await self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.failure_count == 2
        assert self.circuit_breaker.state == ServiceState.CLOSED
        
        # Success should reduce failure count
        await self.circuit_breaker.call(success_func)
        assert self.circuit_breaker.failure_count == 1


class TestFailureDetector:
    """Test phi accrual failure detector."""
    
    def setup_method(self):
        """Setup test environment."""
        self.detector = FailureDetector(
            heartbeat_interval=1.0,
            timeout_multiplier=3.0,
            phi_threshold=5.0
        )
    
    def test_initialization(self):
        """Test FailureDetector initialization."""
        assert self.detector.heartbeat_interval == 1.0
        assert self.detector.timeout_multiplier == 3.0
        assert self.detector.phi_threshold == 5.0
        assert isinstance(self.detector.arrival_intervals, dict)
        assert isinstance(self.detector.last_heartbeat, dict)
        assert isinstance(self.detector.suspected_nodes, set)
        assert isinstance(self.detector.lock, threading.Lock)
    
    def test_heartbeat_recording(self):
        """Test heartbeat recording and interval tracking."""
        node_id = "test_node"
        
        # Record first heartbeat
        start_time = time.time()
        self.detector.record_heartbeat(node_id)
        
        assert node_id in self.detector.last_heartbeat
        assert self.detector.last_heartbeat[node_id] >= start_time
        assert len(self.detector.arrival_intervals[node_id]) == 0  # No interval yet
        
        # Record second heartbeat
        time.sleep(0.1)
        self.detector.record_heartbeat(node_id)
        
        assert len(self.detector.arrival_intervals[node_id]) == 1
        interval = self.detector.arrival_intervals[node_id][0]
        assert 0.08 <= interval <= 0.2  # Allow some timing variance
    
    def test_phi_value_calculation_insufficient_data(self):
        """Test phi value calculation with insufficient data."""
        node_id = "new_node"
        
        # No heartbeats recorded
        phi = self.detector.get_phi_value(node_id)
        assert phi == float('inf')
        
        # Only one heartbeat recorded
        self.detector.record_heartbeat(node_id)
        phi = self.detector.get_phi_value(node_id)
        assert phi == 0.0  # Recent heartbeat, no intervals
        
        # Simulate time passing without heartbeat
        with patch('time.time', return_value=self.detector.last_heartbeat[node_id] + 10):
            phi = self.detector.get_phi_value(node_id)
            assert phi > self.detector.phi_threshold
    
    def test_phi_value_calculation_with_intervals(self):
        """Test phi value calculation with sufficient interval data."""
        node_id = "regular_node"
        
        # Build up interval history with regular heartbeats
        base_time = time.time()
        for i in range(10):
            with patch('time.time', return_value=base_time + i):
                self.detector.record_heartbeat(node_id)
        
        # Should have low phi value for recent heartbeat
        with patch('time.time', return_value=base_time + 9.5):  # Just after last heartbeat
            phi = self.detector.get_phi_value(node_id)
            assert phi < self.detector.phi_threshold
        
        # Should have high phi value for delayed heartbeat
        with patch('time.time', return_value=base_time + 15):  # Long delay
            phi = self.detector.get_phi_value(node_id)
            assert phi > self.detector.phi_threshold
    
    def test_node_suspicion_detection(self):
        """Test node suspicion detection."""
        node_id = "monitored_node"
        
        # New node should not be suspected
        assert not self.detector.is_node_suspected(node_id)
        
        # Record heartbeat
        self.detector.record_heartbeat(node_id)
        assert not self.detector.is_node_suspected(node_id)
        
        # Simulate long delay causing suspicion
        with patch.object(self.detector, 'get_phi_value', return_value=10.0):
            assert self.detector.is_node_suspected(node_id)
            assert node_id in self.detector.suspected_nodes
    
    def test_suspected_nodes_management(self):
        """Test suspected nodes set management."""
        nodes = ["node_1", "node_2", "node_3"]
        
        # Add nodes to suspected set
        for node_id in nodes:
            with patch.object(self.detector, 'get_phi_value', return_value=10.0):
                self.detector.is_node_suspected(node_id)
        
        suspected = self.detector.get_suspected_nodes()
        assert len(suspected) == 3
        assert all(node_id in suspected for node_id in nodes)
        
        # Remove suspicion from one node
        with patch.object(self.detector, 'get_phi_value', return_value=1.0):
            self.detector.is_node_suspected(nodes[0])
        
        suspected = self.detector.get_suspected_nodes()
        assert len(suspected) == 2
        assert nodes[0] not in suspected
    
    def test_heartbeat_removes_suspicion(self):
        """Test that recording heartbeat removes node from suspicion."""
        node_id = "recovering_node"
        
        # Make node suspected
        with patch.object(self.detector, 'get_phi_value', return_value=10.0):
            self.detector.is_node_suspected(node_id)
        
        assert node_id in self.detector.suspected_nodes
        
        # Record heartbeat should remove suspicion
        self.detector.record_heartbeat(node_id)
        assert node_id not in self.detector.suspected_nodes
    
    def test_concurrent_access(self):
        """Test thread-safe access to failure detector."""
        node_id = "concurrent_node"
        
        def record_heartbeats():
            for _ in range(10):
                self.detector.record_heartbeat(node_id)
                time.sleep(0.01)
        
        def check_suspicion():
            for _ in range(10):
                self.detector.is_node_suspected(node_id)
                time.sleep(0.01)
        
        # Run concurrent operations
        thread1 = threading.Thread(target=record_heartbeats)
        thread2 = threading.Thread(target=check_suspicion)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Should complete without errors
        assert node_id in self.detector.last_heartbeat


class TestElasticTrainingManager:
    """Test elastic training manager for dynamic scaling."""
    
    def setup_method(self):
        """Setup test environment."""
        self.manager = ElasticTrainingManager(
            min_nodes=2,
            max_nodes=10,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3
        )
    
    def test_initialization(self):
        """Test ElasticTrainingManager initialization."""
        assert self.manager.min_nodes == 2
        assert self.manager.max_nodes == 10
        assert self.manager.scale_up_threshold == 0.8
        assert self.manager.scale_down_threshold == 0.3
        assert isinstance(self.manager.active_nodes, dict)
        assert isinstance(self.manager.pending_nodes, dict)
        assert isinstance(self.manager.failed_nodes, dict)
        assert self.manager.last_scale_event == 0
        assert self.manager.scale_cooldown == 300
        assert isinstance(self.manager.lock, threading.RLock)
    
    def test_add_healthy_node(self):
        """Test adding healthy node to cluster."""
        node = NodeInfo(
            node_id="healthy_node",
            hostname="worker-1",
            port=29500,
            rank=0,
            state=NodeState.HEALTHY
        )
        
        self.manager.add_node(node)
        
        assert "healthy_node" in self.manager.active_nodes
        assert self.manager.active_nodes["healthy_node"] == node
        assert "healthy_node" not in self.manager.pending_nodes
    
    def test_add_pending_node(self):
        """Test adding pending node to cluster."""
        node = NodeInfo(
            node_id="pending_node",
            hostname="worker-2",
            port=29501,
            rank=1,
            state=NodeState.REJOINING
        )
        
        self.manager.add_node(node)
        
        assert "pending_node" in self.manager.pending_nodes
        assert self.manager.pending_nodes["pending_node"] == node
        assert "pending_node" not in self.manager.active_nodes
    
    def test_remove_active_node(self):
        """Test removing active node from cluster."""
        node = NodeInfo(
            node_id="remove_node",
            hostname="worker-3",
            port=29502,
            rank=2,
            state=NodeState.HEALTHY
        )
        
        self.manager.add_node(node)
        assert "remove_node" in self.manager.active_nodes
        
        self.manager.remove_node("remove_node", FailureType.NODE_FAILURE)
        
        assert "remove_node" not in self.manager.active_nodes
        assert "remove_node" in self.manager.failed_nodes
        assert self.manager.failed_nodes["remove_node"].state == NodeState.FAILED
    
    def test_remove_pending_node(self):
        """Test removing pending node from cluster."""
        node = NodeInfo(
            node_id="pending_remove",
            hostname="worker-4",
            port=29503,
            rank=3,
            state=NodeState.RECOVERING
        )
        
        self.manager.add_node(node)
        assert "pending_remove" in self.manager.pending_nodes
        
        self.manager.remove_node("pending_remove", FailureType.NETWORK_PARTITION)
        
        assert "pending_remove" not in self.manager.pending_nodes
        assert "pending_remove" in self.manager.failed_nodes
        assert self.manager.failed_nodes["pending_remove"].state == NodeState.FAILED
    
    def test_update_node_load(self):
        """Test updating node load factor."""
        node = NodeInfo(
            node_id="load_node",
            hostname="worker-5",
            port=29504,
            rank=4,
            load_factor=0.5
        )
        
        self.manager.add_node(node)
        assert self.manager.active_nodes["load_node"].load_factor == 0.5
        
        self.manager.update_node_load("load_node", 0.9)
        assert self.manager.active_nodes["load_node"].load_factor == 0.9
        
        # Should not crash for non-existent node
        self.manager.update_node_load("nonexistent", 0.7)
    
    def test_cluster_load_calculation(self):
        """Test cluster load calculation."""
        # Empty cluster
        assert self.manager.get_cluster_load() == 0.0
        
        # Add nodes with different loads
        loads = [0.2, 0.8, 0.5, 0.9]
        for i, load in enumerate(loads):
            node = NodeInfo(
                node_id=f"load_node_{i}",
                hostname=f"worker-{i}",
                port=29500 + i,
                rank=i,
                load_factor=load
            )
            self.manager.add_node(node)
        
        expected_avg = sum(loads) / len(loads)
        actual_avg = self.manager.get_cluster_load()
        assert abs(actual_avg - expected_avg) < 1e-6
    
    def test_scale_up_decisions(self):
        """Test scale-up decision logic."""
        # Add nodes with high load
        for i in range(3):
            node = NodeInfo(
                node_id=f"high_load_{i}",
                hostname=f"worker-{i}",
                port=29500 + i,
                rank=i,
                load_factor=0.9  # High load
            )
            self.manager.add_node(node)
        
        # Should want to scale up due to high load
        assert self.manager.should_scale_up() is True
        
        # At max nodes, should not scale up
        for i in range(3, 10):
            node = NodeInfo(
                node_id=f"max_node_{i}",
                hostname=f"worker-{i}",
                port=29500 + i,
                rank=i
            )
            self.manager.add_node(node)
        
        assert self.manager.should_scale_up() is False
        
        # During cooldown, should not scale up
        self.manager.last_scale_event = time.time()
        assert self.manager.should_scale_up() is False
    
    def test_scale_down_decisions(self):
        """Test scale-down decision logic."""
        # Add minimum nodes with low load
        for i in range(5):
            node = NodeInfo(
                node_id=f"low_load_{i}",
                hostname=f"worker-{i}",
                port=29500 + i,
                rank=i,
                load_factor=0.1  # Low load
            )
            self.manager.add_node(node)
        
        # Should want to scale down due to low load
        assert self.manager.should_scale_down() is True
        
        # Remove nodes to minimum
        for i in range(3):
            self.manager.remove_node(f"low_load_{i}", FailureType.RESOURCE_EXHAUSTION)
        
        # At minimum nodes, should not scale down
        assert self.manager.should_scale_down() is False
        
        # During cooldown, should not scale down
        for i in range(2, 5):  # Add back some nodes
            node = NodeInfo(
                node_id=f"cooldown_{i}",
                hostname=f"worker-{i}",
                port=29500 + i,
                rank=i,
                load_factor=0.1
            )
            self.manager.add_node(node)
        
        self.manager.last_scale_event = time.time()
        assert self.manager.should_scale_down() is False
    
    def test_node_selection_for_removal(self):
        """Test node selection for removal during scale-down."""
        # Add nodes with different characteristics
        nodes_data = [
            ("stable_node", 0, 0.5, 0),      # Low failure, medium load
            ("problematic_node", 3, 0.3, 1),  # High failure, low load
            ("busy_node", 1, 0.9, 2),        # Medium failure, high load
            ("reliable_node", 0, 0.8, 3)     # Low failure, high load
        ]
        
        for node_id, failures, load, rank in nodes_data:
            node = NodeInfo(
                node_id=node_id,
                hostname=f"host-{rank}",
                port=29500 + rank,
                rank=rank,
                failure_count=failures,
                load_factor=load
            )
            self.manager.add_node(node)
        
        # Select 2 nodes for removal
        to_remove = self.manager.select_nodes_for_removal(2)
        
        assert len(to_remove) == 2
        # Should prefer nodes with higher failure counts
        assert "problematic_node" in to_remove
    
    def test_cluster_state_reporting(self):
        """Test cluster state reporting."""
        # Add nodes in different states
        active_node = NodeInfo("active", "host1", 29500, 0, NodeState.HEALTHY)
        self.manager.add_node(active_node)
        
        pending_node = NodeInfo("pending", "host2", 29501, 1, NodeState.RECOVERING)
        self.manager.add_node(pending_node)
        
        # Manually add to failed
        failed_node = NodeInfo("failed", "host3", 29502, 2, NodeState.FAILED)
        self.manager.failed_nodes["failed"] = failed_node
        
        state = self.manager.get_cluster_state()
        
        assert state["active_nodes"] == 1
        assert state["pending_nodes"] == 1
        assert state["failed_nodes"] == 1
        assert "cluster_load" in state
        assert "should_scale_up" in state
        assert "should_scale_down" in state


class TestMultiLevelCheckpointer:
    """Test multi-level checkpointing system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpointer = MultiLevelCheckpointer(
            self.temp_dir,
            levels={"test": 5, "backup": 20}
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test MultiLevelCheckpointer initialization."""
        assert self.checkpointer.checkpoint_dir == self.temp_dir
        assert self.checkpointer.levels == {"test": 5, "backup": 20}
        assert isinstance(self.checkpointer.last_checkpoint, dict)
        assert isinstance(self.checkpointer.checkpoint_metadata, dict)
        assert os.path.exists(self.temp_dir)
        assert isinstance(self.checkpointer.lock, threading.Lock)
    
    def test_default_levels(self):
        """Test default checkpoint levels."""
        default_checkpointer = MultiLevelCheckpointer(self.temp_dir)
        
        expected_levels = {
            'frequent': 100,
            'hourly': 3600,
            'daily': 86400
        }
        
        assert default_checkpointer.levels == expected_levels
    
    def test_should_checkpoint_logic(self):
        """Test checkpoint scheduling logic."""
        # New level should checkpoint immediately
        assert self.checkpointer.should_checkpoint("test", 5) is True
        assert self.checkpointer.should_checkpoint("test", 4) is False
        
        # After checkpoint, should wait for interval
        self.checkpointer.last_checkpoint["test"] = 5
        assert self.checkpointer.should_checkpoint("test", 9) is False
        assert self.checkpointer.should_checkpoint("test", 10) is True
        
        # Unknown level should return False
        assert self.checkpointer.should_checkpoint("unknown", 100) is False
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(self):
        """Test checkpoint creation."""
        model_state = {"layer1": np.random.randn(5, 3)}
        optimizer_state = {"step": 100, "lr": 0.001}
        metadata = {"epoch": 5, "loss": 0.5}
        
        checkpoint_path = await self.checkpointer.create_checkpoint(
            "test", 10, model_state, optimizer_state, metadata
        )
        
        # Check file was created
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith(".pkl")
        
        # Check metadata was updated
        assert len(self.checkpointer.checkpoint_metadata) == 1
        
        # Check checkpoint content
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        assert data["level"] == "test"
        assert data["step"] == 10
        assert "timestamp" in data
        assert "model_state" in data
        assert "optimizer_state" in data
        assert "metadata" in data
        assert "checkpoint_id" in data
        assert "consistency_hash" in data
    
    @pytest.mark.asyncio
    async def test_checkpoint_consistency_hash(self):
        """Test checkpoint consistency hash computation."""
        model_state = {"param": np.array([1, 2, 3])}
        optimizer_state = {"step": 50}
        metadata = {"loss": 0.1}
        
        # Create checkpoint
        checkpoint_path = await self.checkpointer.create_checkpoint(
            "test", 50, model_state, optimizer_state, metadata
        )
        
        # Load and verify consistency
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        stored_hash = data.pop("consistency_hash")
        computed_hash = self.checkpointer._compute_hash(data)
        
        assert stored_hash == computed_hash
    
    def test_load_latest_checkpoint(self):
        """Test loading latest checkpoint."""
        # No checkpoints initially
        assert self.checkpointer.load_latest_checkpoint() is None
        
        # Manually create checkpoint files
        checkpoint_data_1 = {
            "level": "test",
            "step": 10,
            "timestamp": time.time() - 100,
            "model_state": {"param": np.array([1, 2])},
            "optimizer_state": {"step": 10},
            "metadata": {"loss": 0.5}
        }
        
        checkpoint_data_2 = {
            "level": "backup",
            "step": 20,
            "timestamp": time.time() - 50,
            "model_state": {"param": np.array([3, 4])},
            "optimizer_state": {"step": 20},
            "metadata": {"loss": 0.3}
        }
        
        # Add consistency hashes
        checkpoint_data_1["consistency_hash"] = self.checkpointer._compute_hash(checkpoint_data_1)
        checkpoint_data_2["consistency_hash"] = self.checkpointer._compute_hash(checkpoint_data_2)
        
        # Save checkpoints
        cp1_path = os.path.join(self.temp_dir, "test_10.pkl")
        cp2_path = os.path.join(self.temp_dir, "backup_20.pkl")
        
        with open(cp1_path, 'wb') as f:
            pickle.dump(checkpoint_data_1, f)
        with open(cp2_path, 'wb') as f:
            pickle.dump(checkpoint_data_2, f)
        
        # Update metadata
        self.checkpointer.checkpoint_metadata = {
            "test_10": {"path": cp1_path, "level": "test", "step": 10, "timestamp": checkpoint_data_1["timestamp"], "size": 100},
            "backup_20": {"path": cp2_path, "level": "backup", "step": 20, "timestamp": checkpoint_data_2["timestamp"], "size": 100}
        }
        
        # Load latest (should be backup_20 due to more recent timestamp)
        loaded = self.checkpointer.load_latest_checkpoint()
        assert loaded is not None
        assert loaded["step"] == 20
        assert loaded["level"] == "backup"
    
    def test_load_checkpoint_by_level(self):
        """Test loading checkpoint by specific level."""
        # Setup checkpoint metadata
        self.checkpointer.checkpoint_metadata = {
            "test_10": {"path": "test_path", "level": "test", "timestamp": time.time(), "size": 100},
            "backup_20": {"path": "backup_path", "level": "backup", "timestamp": time.time(), "size": 100}
        }
        
        # Mock successful load
        mock_data = {"level": "test", "step": 10}
        mock_data["consistency_hash"] = self.checkpointer._compute_hash(mock_data)
        
        with patch('builtins.open', create=True) as mock_open:
            with patch('pickle.load', return_value=mock_data):
                loaded = self.checkpointer.load_latest_checkpoint(level="test")
                assert loaded is not None
                assert loaded["level"] == "test"
    
    def test_load_checkpoint_consistency_failure(self):
        """Test handling of checkpoint consistency check failure."""
        # Create checkpoint with invalid hash
        corrupt_data = {
            "level": "test",
            "step": 10,
            "consistency_hash": "invalid_hash"
        }
        
        cp_path = os.path.join(self.temp_dir, "corrupt.pkl")
        with open(cp_path, 'wb') as f:
            pickle.dump(corrupt_data, f)
        
        self.checkpointer.checkpoint_metadata = {
            "corrupt": {"path": cp_path, "level": "test", "timestamp": time.time(), "size": 100}
        }
        
        # Should return None due to consistency check failure
        loaded = self.checkpointer.load_latest_checkpoint()
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints."""
        # Create multiple checkpoints for test level
        for i in range(10):
            model_state = {"param": np.random.randn(2)}
            optimizer_state = {"step": i * 5}
            metadata = {"iteration": i}
            
            await self.checkpointer.create_checkpoint(
                "test", i * 5, model_state, optimizer_state, metadata
            )
        
        # Should have all checkpoints initially
        assert len(self.checkpointer.checkpoint_metadata) == 10
        
        # Cleanup should keep only the most recent ones
        await self.checkpointer._cleanup_old_checkpoints("test")
        
        # Should keep default number for unknown level (10), but we created 10, so all should remain
        # Let's test with a known level that has a specific keep count
        
        # Manually trigger cleanup with a small keep count
        with patch.dict(self.checkpointer.checkpoint_metadata):
            # Simulate having many frequent checkpoints
            keep_counts = {'test': 3}  # Keep only 3
            
            test_metadata = {
                f"test_{i}": {
                    "path": f"test_{i}.pkl",
                    "level": "test",
                    "timestamp": time.time() - (10 - i),  # Older timestamps for earlier checkpoints
                    "size": 100
                }
                for i in range(7)
            }
            
            self.checkpointer.checkpoint_metadata.update(test_metadata)
            
            # Mock file removal
            with patch('os.remove'):
                with patch.dict('src.neural_arch.reliability.fault_tolerance.keep_counts', keep_counts):
                    await self.checkpointer._cleanup_old_checkpoints("test")
    
    def test_hash_computation(self):
        """Test consistency hash computation."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}  # Different order
        data3 = {"a": 1, "b": 2, "c": 4}  # Different value
        
        hash1 = self.checkpointer._compute_hash(data1)
        hash2 = self.checkpointer._compute_hash(data2)
        hash3 = self.checkpointer._compute_hash(data3)
        
        # Same data in different order should have same hash
        assert hash1 == hash2
        
        # Different data should have different hash
        assert hash1 != hash3


class TestSelfHealingSystem:
    """Test self-healing system for automatic recovery."""
    
    def setup_method(self):
        """Setup test environment."""
        with patch('src.neural_arch.reliability.fault_tolerance.MultiLevelCheckpointer'):
            self.healing_system = SelfHealingSystem()
            # Stop monitoring to avoid background thread issues in tests
            self.healing_system.stop_monitoring()
    
    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self.healing_system, '_monitoring_thread'):
            self.healing_system.stop_monitoring()
    
    def test_initialization(self):
        """Test SelfHealingSystem initialization."""
        assert isinstance(self.healing_system.failure_detector, FailureDetector)
        assert isinstance(self.healing_system.elastic_manager, ElasticTrainingManager)
        assert self.healing_system.checkpointer is not None
        
        # Check recovery strategies
        expected_strategies = {
            FailureType.NODE_FAILURE,
            FailureType.NETWORK_PARTITION,
            FailureType.MEMORY_ERROR,
            FailureType.STORAGE_FAILURE
        }
        
        assert set(self.healing_system.recovery_strategies.keys()) == expected_strategies
        
        # Check circuit breakers
        expected_breakers = {'training', 'checkpointing', 'communication'}
        assert set(self.healing_system.circuit_breakers.keys()) == expected_breakers
        
        for breaker in self.healing_system.circuit_breakers.values():
            assert isinstance(breaker, CircuitBreaker)
        
        # Check failure history
        assert isinstance(self.healing_system.failure_history, deque)
    
    @pytest.mark.asyncio
    async def test_handle_failure_with_strategy(self):
        """Test failure handling with specific recovery strategy."""
        node_id = "test_node"
        details = {"error": "connection lost"}
        
        # Mock recovery strategy
        mock_strategy = AsyncMock(return_value=True)
        self.healing_system.recovery_strategies[FailureType.NODE_FAILURE] = mock_strategy
        
        success = await self.healing_system.handle_failure(
            FailureType.NODE_FAILURE, node_id, details
        )
        
        assert success is True
        mock_strategy.assert_called_once_with(node_id, details)
        
        # Check failure was recorded
        assert len(self.healing_system.failure_history) == 1
        failure_event = self.healing_system.failure_history[0]
        assert failure_event.failure_type == FailureType.NODE_FAILURE
        assert failure_event.node_id == node_id
        assert failure_event.details == details
        assert failure_event.recovery_time is not None
    
    @pytest.mark.asyncio
    async def test_handle_failure_without_strategy(self):
        """Test failure handling without specific strategy."""
        node_id = "unknown_failure_node"
        
        # Mock default strategy
        mock_default = AsyncMock(return_value=True)
        self.healing_system._default_recovery_strategy = mock_default
        
        success = await self.healing_system.handle_failure(
            FailureType.TIMEOUT, node_id  # TIMEOUT not in recovery_strategies
        )
        
        assert success is True
        mock_default.assert_called_once_with(node_id, {})
    
    @pytest.mark.asyncio
    async def test_handle_failure_exception(self):
        """Test failure handling when recovery strategy raises exception."""
        node_id = "exception_node"
        
        # Mock strategy that raises exception
        mock_strategy = AsyncMock(side_effect=Exception("Recovery failed"))
        self.healing_system.recovery_strategies[FailureType.NODE_FAILURE] = mock_strategy
        
        success = await self.healing_system.handle_failure(
            FailureType.NODE_FAILURE, node_id
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_handle_node_failure(self):
        """Test node failure recovery strategy."""
        node_id = "failed_node"
        
        # Add node to manager first
        node = NodeInfo(node_id, "host", 29500, 0)
        self.healing_system.elastic_manager.add_node(node)
        
        success = await self.healing_system._handle_node_failure(node_id, {})
        
        assert success is True
        assert node_id in self.healing_system.elastic_manager.failed_nodes
        assert node_id not in self.healing_system.elastic_manager.active_nodes
    
    @pytest.mark.asyncio
    async def test_handle_node_failure_triggers_scaling(self):
        """Test node failure triggering cluster scaling."""
        # Set minimum nodes higher than current
        self.healing_system.elastic_manager.min_nodes = 3
        
        # Add only one node
        node = NodeInfo("lonely_node", "host", 29500, 0)
        self.healing_system.elastic_manager.add_node(node)
        
        # Mock scale up
        mock_scale_up = AsyncMock(return_value=True)
        self.healing_system._scale_up_cluster = mock_scale_up
        
        success = await self.healing_system._handle_node_failure("lonely_node", {})
        
        assert success is True
        mock_scale_up.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_network_partition_recovery(self):
        """Test network partition recovery with success."""
        node_id = "partition_node"
        
        # Mock successful recovery (random.random() > 0.3)
        with patch('random.random', return_value=0.8):
            success = await self.healing_system._handle_network_partition(node_id, {})
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_handle_network_partition_failure(self):
        """Test network partition recovery with failure."""
        node_id = "partition_fail_node"
        
        # Mock failed recovery attempts (random.random() <= 0.3)
        with patch('random.random', return_value=0.2):
            # Mock the subsequent node failure handling
            mock_node_failure = AsyncMock(return_value=False)
            self.healing_system._handle_node_failure = mock_node_failure
            
            success = await self.healing_system._handle_network_partition(node_id, {})
            
            assert success is False
            mock_node_failure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_memory_error(self):
        """Test memory error recovery."""
        node_id = "memory_node"
        details = {"memory_usage": 95}
        
        success = await self.healing_system._handle_memory_error(node_id, details)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_handle_storage_failure(self):
        """Test storage failure recovery."""
        node_id = "storage_node"
        details = {"disk_full": True}
        
        success = await self.healing_system._handle_storage_failure(node_id, details)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_default_recovery_strategy(self):
        """Test default recovery strategy."""
        success = await self.healing_system._default_recovery_strategy("any_node", {})
        assert success is True
    
    @pytest.mark.asyncio
    async def test_scale_up_cluster(self):
        """Test cluster scaling up."""
        initial_count = len(self.healing_system.elastic_manager.active_nodes)
        
        success = await self.healing_system._scale_up_cluster()
        
        assert success is True
        assert len(self.healing_system.elastic_manager.active_nodes) == initial_count + 1
        
        # Check new node was added
        new_nodes = list(self.healing_system.elastic_manager.active_nodes.values())
        if new_nodes:
            new_node = new_nodes[-1]
            assert new_node.hostname.startswith("worker-")
            assert new_node.port == 29500
    
    @pytest.mark.asyncio
    async def test_scale_down_cluster(self):
        """Test cluster scaling down."""
        # Add a node first
        node = NodeInfo("removable_node", "worker-99", 29500, 0)
        self.healing_system.elastic_manager.add_node(node)
        
        initial_count = len(self.healing_system.elastic_manager.active_nodes)
        
        success = await self.healing_system._scale_down_cluster()
        
        assert success is True
        # Node should be moved to failed nodes
        assert len(self.healing_system.elastic_manager.failed_nodes) >= 1
    
    def test_handle_suspected_node(self):
        """Test handling suspected node."""
        node_id = "suspected_node"
        
        # Add node first
        node = NodeInfo(node_id, "host", 29500, 0)
        self.healing_system.elastic_manager.add_node(node)
        
        self.healing_system._handle_suspected_node(node_id)
        
        # Node should be removed and failure recorded
        assert node_id in self.healing_system.elastic_manager.failed_nodes
        assert len(self.healing_system.failure_history) == 1
    
    def test_get_system_health(self):
        """Test system health reporting."""
        # Add some failure history
        failure = FailureEvent(
            timestamp=time.time() - 30,  # Recent failure
            failure_type=FailureType.NODE_FAILURE,
            node_id="failed_node"
        )
        self.healing_system.failure_history.append(failure)
        
        health = self.healing_system.get_system_health()
        
        assert "cluster_state" in health
        assert "recent_failures" in health
        assert "circuit_breaker_states" in health
        assert "total_failures" in health
        
        assert isinstance(health["cluster_state"], dict)
        assert health["recent_failures"] >= 0
        assert isinstance(health["circuit_breaker_states"], dict)
        assert health["total_failures"] == 1
        
        # Check circuit breaker states
        for service in ['training', 'checkpointing', 'communication']:
            assert service in health["circuit_breaker_states"]
    
    def test_monitoring_start_stop(self):
        """Test monitoring thread start and stop."""
        # Create new system to test monitoring
        with patch('src.neural_arch.reliability.fault_tolerance.MultiLevelCheckpointer'):
            system = SelfHealingSystem()
        
        # Should start monitoring by default
        assert system._monitoring_active is True
        assert system._monitoring_thread is not None
        
        # Stop monitoring
        system.stop_monitoring()
        assert system._monitoring_active is False
        
        # Cleanup
        if system._monitoring_thread:
            system._monitoring_thread.join(timeout=1)


class TestChaosEngineer:
    """Test chaos engineering for resilience testing."""
    
    def setup_method(self):
        """Setup test environment."""
        with patch('src.neural_arch.reliability.fault_tolerance.MultiLevelCheckpointer'):
            self.healing_system = SelfHealingSystem()
            self.healing_system.stop_monitoring()  # Prevent background thread issues
            
        self.chaos_engineer = ChaosEngineer(self.healing_system)
    
    def test_initialization(self):
        """Test ChaosEngineer initialization."""
        assert self.chaos_engineer.system == self.healing_system
        
        expected_experiments = {
            'kill_random_node',
            'network_partition',
            'memory_pressure',
            'storage_corruption'
        }
        
        assert set(self.chaos_engineer.chaos_experiments.keys()) == expected_experiments
    
    @pytest.mark.asyncio
    async def test_run_experiment_valid(self):
        """Test running valid chaos experiment."""
        # Add nodes for experiments
        for i in range(3):
            node = NodeInfo(f"node_{i}", f"host-{i}", 29500 + i, i)
            self.healing_system.elastic_manager.add_node(node)
        
        # Mock experiment
        mock_experiment = AsyncMock()
        self.chaos_engineer.chaos_experiments['kill_random_node'] = mock_experiment
        
        await self.chaos_engineer.run_experiment('kill_random_node', 0.5)
        
        mock_experiment.assert_called_once_with(0.5)
    
    @pytest.mark.asyncio
    async def test_run_experiment_invalid(self):
        """Test running invalid chaos experiment."""
        with pytest.raises(ValueError, match="Unknown experiment"):
            await self.chaos_engineer.run_experiment('nonexistent_experiment')
    
    @pytest.mark.asyncio
    async def test_kill_random_node_experiment(self):
        """Test kill random node chaos experiment."""
        # Add nodes
        node_ids = []
        for i in range(5):
            node_id = f"victim_node_{i}"
            node = NodeInfo(node_id, f"host-{i}", 29500 + i, i)
            self.healing_system.elastic_manager.add_node(node)
            node_ids.append(node_id)
        
        # Mock handle_failure
        mock_handle_failure = AsyncMock()
        self.healing_system.handle_failure = mock_handle_failure
        
        # Run experiment with 40% intensity (should kill 2 nodes)
        await self.chaos_engineer._kill_random_node(0.4)
        
        # Should have called handle_failure for killed nodes
        assert mock_handle_failure.call_count == 2
        
        # Verify calls were for node failures with chaos experiment flag
        for call_args in mock_handle_failure.call_args_list:
            args, kwargs = call_args
            assert args[0] == FailureType.NODE_FAILURE
            assert args[1] in node_ids
            assert args[2]['chaos_experiment'] is True
    
    @pytest.mark.asyncio
    async def test_kill_random_node_no_nodes(self):
        """Test kill random node with no active nodes."""
        # No nodes in cluster
        assert len(self.healing_system.elastic_manager.active_nodes) == 0
        
        # Should not crash
        await self.chaos_engineer._kill_random_node(0.5)
    
    @pytest.mark.asyncio
    async def test_network_partition_experiment(self):
        """Test network partition chaos experiment."""
        # Add nodes
        node_ids = []
        for i in range(6):
            node_id = f"network_node_{i}"
            node = NodeInfo(node_id, f"host-{i}", 29500 + i, i)
            self.healing_system.elastic_manager.add_node(node)
            node_ids.append(node_id)
        
        # Mock handle_failure
        mock_handle_failure = AsyncMock()
        self.healing_system.handle_failure = mock_handle_failure
        
        # Run experiment with 30% intensity (should partition ~2 nodes)
        await self.chaos_engineer._create_network_partition(0.3)
        
        # Should have called handle_failure for partitioned nodes
        assert mock_handle_failure.call_count >= 1
        
        # Verify calls were for network partition
        for call_args in mock_handle_failure.call_args_list:
            args, kwargs = call_args
            assert args[0] == FailureType.NETWORK_PARTITION
            assert args[2]['chaos_experiment'] is True
            assert 'partition_duration' in args[2]
    
    @pytest.mark.asyncio
    async def test_network_partition_insufficient_nodes(self):
        """Test network partition with insufficient nodes."""
        # Add only one node
        node = NodeInfo("single_node", "host", 29500, 0)
        self.healing_system.elastic_manager.add_node(node)
        
        # Should not partition anything
        mock_handle_failure = AsyncMock()
        self.healing_system.handle_failure = mock_handle_failure
        
        await self.chaos_engineer._create_network_partition(0.5)
        
        # Should not call handle_failure
        mock_handle_failure.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_memory_pressure_experiment(self):
        """Test memory pressure chaos experiment."""
        # Add nodes
        for i in range(4):
            node_id = f"memory_node_{i}"
            node = NodeInfo(node_id, f"host-{i}", 29500 + i, i)
            self.healing_system.elastic_manager.add_node(node)
        
        # Mock handle_failure
        mock_handle_failure = AsyncMock()
        self.healing_system.handle_failure = mock_handle_failure
        
        # Run experiment
        await self.chaos_engineer._create_memory_pressure(0.5)
        
        # Should affect some nodes
        assert mock_handle_failure.call_count >= 1
        
        # Verify calls were for memory errors
        for call_args in mock_handle_failure.call_args_list:
            args, kwargs = call_args
            assert args[0] == FailureType.MEMORY_ERROR
            assert args[2]['chaos_experiment'] is True
            assert 'memory_usage' in args[2]
    
    @pytest.mark.asyncio
    async def test_memory_pressure_no_nodes(self):
        """Test memory pressure with no nodes."""
        mock_handle_failure = AsyncMock()
        self.healing_system.handle_failure = mock_handle_failure
        
        await self.chaos_engineer._create_memory_pressure(0.5)
        
        # Should not call handle_failure
        mock_handle_failure.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_storage_corruption_experiment(self):
        """Test storage corruption chaos experiment."""
        # Add nodes
        for i in range(3):
            node_id = f"storage_node_{i}"
            node = NodeInfo(node_id, f"host-{i}", 29500 + i, i)
            self.healing_system.elastic_manager.add_node(node)
        
        # Mock handle_failure
        mock_handle_failure = AsyncMock()
        self.healing_system.handle_failure = mock_handle_failure
        
        # Run experiment
        await self.chaos_engineer._create_storage_corruption(0.5)
        
        # Should affect one node
        mock_handle_failure.assert_called_once()
        
        # Verify call was for storage failure
        args, kwargs = mock_handle_failure.call_args_list[0]
        assert args[0] == FailureType.STORAGE_FAILURE
        assert args[2]['chaos_experiment'] is True
        assert 'corruption_type' in args[2]
    
    @pytest.mark.asyncio
    async def test_storage_corruption_no_nodes(self):
        """Test storage corruption with no nodes."""
        mock_handle_failure = AsyncMock()
        self.healing_system.handle_failure = mock_handle_failure
        
        await self.chaos_engineer._create_storage_corruption(0.5)
        
        # Should not call handle_failure
        mock_handle_failure.assert_not_called()


class TestIntegrationScenarios:
    """Integration tests for fault tolerance scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_fault_tolerance_workflow(self):
        """Test complete fault tolerance workflow."""
        with patch('src.neural_arch.reliability.fault_tolerance.MultiLevelCheckpointer'):
            # Initialize system
            healing_system = SelfHealingSystem()
            healing_system.stop_monitoring()  # Prevent background issues
            
            chaos_engineer = ChaosEngineer(healing_system)
            
            # Add initial cluster
            for i in range(4):
                node = NodeInfo(f"node_{i}", f"worker-{i}", 29500 + i, i)
                healing_system.elastic_manager.add_node(node)
                healing_system.failure_detector.record_heartbeat(f"node_{i}")
            
            # Test failure detection and recovery
            failure_types = [
                FailureType.NODE_FAILURE,
                FailureType.NETWORK_PARTITION,
                FailureType.MEMORY_ERROR
            ]
            
            for i, failure_type in enumerate(failure_types):
                success = await healing_system.handle_failure(
                    failure_type, f"node_{i}", {"test": True}
                )
                assert success is True
            
            # Verify failures were recorded
            assert len(healing_system.failure_history) == 3
            
            # Test chaos engineering
            await chaos_engineer.run_experiment('memory_pressure', 0.2)
            
            # System should still be functional
            health = healing_system.get_system_health()
            assert health is not None
            assert 'cluster_state' in health
    
    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures."""
        with patch('src.neural_arch.reliability.fault_tolerance.MultiLevelCheckpointer'):
            healing_system = SelfHealingSystem()
            healing_system.stop_monitoring()
            
            # Setup cluster
            for i in range(6):
                node = NodeInfo(f"cascade_node_{i}", f"host-{i}", 29500 + i, i)
                healing_system.elastic_manager.add_node(node)
            
            # Simulate cascading failures
            failure_sequence = [
                (FailureType.NODE_FAILURE, "cascade_node_0"),
                (FailureType.NETWORK_PARTITION, "cascade_node_1"),
                (FailureType.MEMORY_ERROR, "cascade_node_2"),
                (FailureType.STORAGE_FAILURE, "cascade_node_3")
            ]
            
            recovery_results = []
            for failure_type, node_id in failure_sequence:
                success = await healing_system.handle_failure(failure_type, node_id)
                recovery_results.append(success)
                
                # Small delay between failures
                await asyncio.sleep(0.01)
            
            # All recoveries should succeed
            assert all(recovery_results)
            
            # System should maintain some active nodes
            cluster_state = healing_system.elastic_manager.get_cluster_state()
            assert cluster_state['active_nodes'] >= healing_system.elastic_manager.min_nodes
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with failure handling."""
        with patch('src.neural_arch.reliability.fault_tolerance.MultiLevelCheckpointer'):
            healing_system = SelfHealingSystem()
            healing_system.stop_monitoring()
        
        # Get circuit breaker
        training_cb = healing_system.circuit_breakers['training']
        
        # Simulate repeated training failures
        async def failing_training():
            raise Exception("Training failed")
        
        # Accumulate failures to open circuit
        for _ in range(5):
            try:
                await training_cb.call(failing_training)
            except Exception:
                pass
        
        # Circuit should be open
        assert training_cb.state == ServiceState.OPEN
        
        # Should fail fast
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await training_cb.call(failing_training)
    
    def test_thread_safety_under_load(self):
        """Test thread safety under concurrent load."""
        with patch('src.neural_arch.reliability.fault_tolerance.MultiLevelCheckpointer'):
            healing_system = SelfHealingSystem()
            healing_system.stop_monitoring()
        
        # Concurrent operations
        def add_remove_nodes():
            for i in range(10):
                node = NodeInfo(f"thread_node_{i}_{threading.current_thread().ident}", 
                              f"host-{i}", 29500 + i, i)
                healing_system.elastic_manager.add_node(node)
                healing_system.failure_detector.record_heartbeat(node.node_id)
                
                if i % 2 == 0:
                    healing_system.elastic_manager.remove_node(node.node_id, FailureType.NODE_FAILURE)
        
        def record_heartbeats():
            for i in range(20):
                healing_system.failure_detector.record_heartbeat(f"heartbeat_node_{i}")
                time.sleep(0.001)
        
        # Run concurrent threads
        threads = []
        for _ in range(3):
            thread1 = threading.Thread(target=add_remove_nodes)
            thread2 = threading.Thread(target=record_heartbeats)
            threads.extend([thread1, thread2])
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without deadlocks or data corruption
        health = healing_system.get_system_health()
        assert health is not None


@pytest.mark.asyncio
async def test_system_integration():
    """Test the complete fault tolerance system integration."""
    # This test would run the built-in test function
    try:
        # Run the built-in system test with shorter delays for testing
        with patch('asyncio.sleep', return_value=None):  # Speed up test
            with patch('time.sleep', return_value=None):
                await test_fault_tolerance_system()
        print("✅ Fault tolerance system integration test passed")
    except Exception as e:
        pytest.fail(f"Fault tolerance system test failed: {e}")


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])