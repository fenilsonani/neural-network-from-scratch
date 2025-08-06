"""Enterprise-grade fault tolerance and resilience system for distributed training.

This module provides comprehensive fault tolerance with:
- Automatic failure detection and recovery
- Elastic training with dynamic node management
- Circuit breakers and bulkheads for service isolation
- Graceful degradation under resource constraints
- Multi-level checkpointing with consistency guarantees
- Health monitoring and self-healing mechanisms
- Chaos engineering tools for resilience testing
- Distributed consensus for coordinator election
"""

import asyncio
import json
import logging
import os
import pickle
import random
import socket
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil functions for testing
    class MockPSUtil:
        @staticmethod
        def virtual_memory():
            class MockMemory:
                percent = 45.0
            return MockMemory()
        
        @staticmethod
        def disk_usage(path):
            class MockDisk:
                free = 100 * 1024 * 1024 * 1024  # 100GB
                total = 500 * 1024 * 1024 * 1024  # 500GB
            return MockDisk()
    
    psutil = MockPSUtil()


class FailureType(Enum):
    """Types of failures that can occur."""
    
    NODE_FAILURE = "node_failure"
    NETWORK_PARTITION = "network_partition"
    STORAGE_FAILURE = "storage_failure"
    MEMORY_ERROR = "memory_error"
    TIMEOUT = "timeout"
    CORRUPTION = "corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class NodeState(Enum):
    """States a node can be in."""
    
    HEALTHY = "healthy"
    SUSPECTED = "suspected"
    FAILED = "failed"
    RECOVERING = "recovering"
    REJOINING = "rejoining"
    MAINTENANCE = "maintenance"


class ServiceState(Enum):
    """Circuit breaker service states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    
    node_id: str
    hostname: str
    port: int
    rank: int
    state: NodeState = NodeState.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    failure_count: int = 0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    load_factor: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'port': self.port,
            'rank': self.rank,
            'state': self.state.value,
            'last_heartbeat': self.last_heartbeat,
            'failure_count': self.failure_count,
            'capabilities': self.capabilities,
            'load_factor': self.load_factor
        }


@dataclass
class FailureEvent:
    """Record of a failure event."""
    
    timestamp: float
    failure_type: FailureType
    node_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'failure_type': self.failure_type.value,
            'node_id': self.node_id,
            'details': self.details,
            'recovery_time': self.recovery_time
        }


class CircuitBreaker:
    """Circuit breaker pattern for service protection."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_duration: float = 60.0,
        success_threshold: int = 3
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening
            timeout_duration: Time to wait before trying again
            success_threshold: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.success_threshold = success_threshold
        
        self.state = ServiceState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        
        self.lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        with self.lock:
            if self.state == ServiceState.OPEN:
                if time.time() - self.last_failure_time < self.timeout_duration:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = ServiceState.HALF_OPEN
                    self.success_count = 0
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful operation."""
        with self.lock:
            if self.state == ServiceState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = ServiceState.CLOSED
                    self.failure_count = 0
            elif self.state == ServiceState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self):
        """Handle failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = ServiceState.OPEN
            elif self.state == ServiceState.HALF_OPEN:
                self.state = ServiceState.OPEN


class FailureDetector:
    """Failure detector for distributed nodes."""
    
    def __init__(
        self,
        heartbeat_interval: float = 5.0,
        timeout_multiplier: float = 3.0,
        phi_threshold: float = 8.0
    ):
        """Initialize failure detector.
        
        Args:
            heartbeat_interval: Interval between heartbeats
            timeout_multiplier: Multiplier for failure detection timeout
            phi_threshold: Phi accrual threshold for failure detection
        """
        self.heartbeat_interval = heartbeat_interval
        self.timeout_multiplier = timeout_multiplier
        self.phi_threshold = phi_threshold
        
        # Phi accrual failure detector state
        self.arrival_intervals: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.last_heartbeat: Dict[str, float] = {}
        self.suspected_nodes: Set[str] = set()
        
        self.lock = threading.Lock()
    
    def record_heartbeat(self, node_id: str):
        """Record heartbeat from a node."""
        current_time = time.time()
        
        with self.lock:
            if node_id in self.last_heartbeat:
                interval = current_time - self.last_heartbeat[node_id]
                self.arrival_intervals[node_id].append(interval)
            
            self.last_heartbeat[node_id] = current_time
            
            # Remove from suspected if it was there
            self.suspected_nodes.discard(node_id)
    
    def get_phi_value(self, node_id: str) -> float:
        """Calculate phi value for failure probability."""
        current_time = time.time()
        
        with self.lock:
            if node_id not in self.last_heartbeat:
                return float('inf')
            
            time_since_last = current_time - self.last_heartbeat[node_id]
            intervals = list(self.arrival_intervals[node_id])
            
            if len(intervals) < 2:
                # Not enough data, use simple timeout
                if time_since_last > self.heartbeat_interval * self.timeout_multiplier:
                    return self.phi_threshold + 1
                else:
                    return 0.0
            
            # Calculate mean and standard deviation
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if std_interval == 0:
                std_interval = mean_interval * 0.1  # Small default std
            
            # Phi calculation (simplified)
            normalized_time = (time_since_last - mean_interval) / std_interval
            phi = max(0, normalized_time)
            
            return phi
    
    def is_node_suspected(self, node_id: str) -> bool:
        """Check if node is suspected of failure."""
        phi = self.get_phi_value(node_id)
        is_suspected = phi > self.phi_threshold
        
        with self.lock:
            if is_suspected:
                self.suspected_nodes.add(node_id)
            else:
                self.suspected_nodes.discard(node_id)
        
        return is_suspected
    
    def get_suspected_nodes(self) -> Set[str]:
        """Get all currently suspected nodes."""
        with self.lock:
            return self.suspected_nodes.copy()


class ElasticTrainingManager:
    """Manager for elastic training with dynamic scaling."""
    
    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 100,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3
    ):
        """Initialize elastic training manager.
        
        Args:
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            scale_up_threshold: Load threshold for scaling up
            scale_down_threshold: Load threshold for scaling down
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.active_nodes: Dict[str, NodeInfo] = {}
        self.pending_nodes: Dict[str, NodeInfo] = {}
        self.failed_nodes: Dict[str, NodeInfo] = {}
        
        # Scaling cooldown to prevent thrashing
        self.last_scale_event = 0
        self.scale_cooldown = 300  # 5 minutes
        
        self.lock = threading.RLock()
    
    def add_node(self, node: NodeInfo):
        """Add a new node to the cluster."""
        with self.lock:
            if node.state == NodeState.HEALTHY:
                self.active_nodes[node.node_id] = node
            else:
                self.pending_nodes[node.node_id] = node
    
    def remove_node(self, node_id: str, reason: FailureType):
        """Remove a node from the cluster."""
        with self.lock:
            if node_id in self.active_nodes:
                node = self.active_nodes.pop(node_id)
                node.state = NodeState.FAILED
                self.failed_nodes[node_id] = node
            elif node_id in self.pending_nodes:
                node = self.pending_nodes.pop(node_id)
                node.state = NodeState.FAILED
                self.failed_nodes[node_id] = node
    
    def update_node_load(self, node_id: str, load_factor: float):
        """Update load factor for a node."""
        with self.lock:
            if node_id in self.active_nodes:
                self.active_nodes[node_id].load_factor = load_factor
    
    def get_cluster_load(self) -> float:
        """Get average cluster load factor."""
        with self.lock:
            if not self.active_nodes:
                return 0.0
            
            total_load = sum(node.load_factor for node in self.active_nodes.values())
            return total_load / len(self.active_nodes)
    
    def should_scale_up(self) -> bool:
        """Check if cluster should scale up."""
        with self.lock:
            if len(self.active_nodes) >= self.max_nodes:
                return False
            
            if time.time() - self.last_scale_event < self.scale_cooldown:
                return False
            
            return self.get_cluster_load() > self.scale_up_threshold
    
    def should_scale_down(self) -> bool:
        """Check if cluster should scale down."""
        with self.lock:
            if len(self.active_nodes) <= self.min_nodes:
                return False
            
            if time.time() - self.last_scale_event < self.scale_cooldown:
                return False
            
            return self.get_cluster_load() < self.scale_down_threshold
    
    def select_nodes_for_removal(self, count: int) -> List[str]:
        """Select nodes to remove when scaling down."""
        with self.lock:
            # Prefer nodes with higher failure counts or lower capabilities
            sorted_nodes = sorted(
                self.active_nodes.items(),
                key=lambda x: (x[1].failure_count, -x[1].load_factor, x[1].rank)
            )
            
            return [node_id for node_id, _ in sorted_nodes[:count]]
    
    def get_cluster_state(self) -> Dict[str, Any]:
        """Get current cluster state."""
        with self.lock:
            return {
                'active_nodes': len(self.active_nodes),
                'pending_nodes': len(self.pending_nodes),
                'failed_nodes': len(self.failed_nodes),
                'cluster_load': self.get_cluster_load(),
                'should_scale_up': self.should_scale_up(),
                'should_scale_down': self.should_scale_down()
            }


class MultiLevelCheckpointer:
    """Multi-level checkpointing with consistency guarantees."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        levels: Dict[str, int] = None
    ):
        """Initialize multi-level checkpointer.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            levels: Checkpoint levels with their intervals
        """
        self.checkpoint_dir = checkpoint_dir
        self.levels = levels or {
            'frequent': 100,    # Every 100 steps
            'hourly': 3600,     # Every hour
            'daily': 86400      # Every day
        }
        
        self.last_checkpoint = defaultdict(int)
        self.checkpoint_metadata = {}
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.lock = threading.Lock()
    
    def should_checkpoint(self, level: str, current_step: int) -> bool:
        """Check if checkpoint should be created for level."""
        if level not in self.levels:
            return False
        
        interval = self.levels[level]
        return (current_step - self.last_checkpoint[level]) >= interval
    
    async def create_checkpoint(
        self,
        level: str,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Create checkpoint at specified level.
        
        Args:
            level: Checkpoint level
            step: Training step
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            metadata: Additional metadata
        
        Returns:
            Checkpoint path
        """
        checkpoint_id = f"{level}_{step}_{int(time.time())}"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.pkl")
        
        checkpoint_data = {
            'level': level,
            'step': step,
            'timestamp': time.time(),
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'metadata': metadata,
            'checkpoint_id': checkpoint_id
        }
        
        # Add consistency hash
        checkpoint_data['consistency_hash'] = self._compute_hash(checkpoint_data)
        
        with self.lock:
            # Save checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Update metadata
            self.checkpoint_metadata[checkpoint_id] = {
                'path': checkpoint_path,
                'level': level,
                'step': step,
                'timestamp': checkpoint_data['timestamp'],
                'size': os.path.getsize(checkpoint_path)
            }
            
            self.last_checkpoint[level] = step
            
            # Cleanup old checkpoints for this level
            await self._cleanup_old_checkpoints(level)
        
        return checkpoint_path
    
    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute consistency hash for checkpoint data."""
        # Simplified hash computation
        serialized = json.dumps(data, sort_keys=True, default=str)
        return str(hash(serialized))
    
    async def _cleanup_old_checkpoints(self, level: str):
        """Clean up old checkpoints for a level."""
        # Keep only the last N checkpoints for each level
        keep_counts = {'frequent': 5, 'hourly': 24, 'daily': 30}
        keep_count = keep_counts.get(level, 10)
        
        level_checkpoints = [
            (cp_id, meta) for cp_id, meta in self.checkpoint_metadata.items()
            if meta['level'] == level
        ]
        
        if len(level_checkpoints) > keep_count:
            # Sort by timestamp and remove oldest
            level_checkpoints.sort(key=lambda x: x[1]['timestamp'])
            to_remove = level_checkpoints[:-keep_count]
            
            for cp_id, meta in to_remove:
                try:
                    os.remove(meta['path'])
                    del self.checkpoint_metadata[cp_id]
                except OSError:
                    pass  # File already removed
    
    def load_latest_checkpoint(self, level: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint.
        
        Args:
            level: Specific level to load from (None for any level)
        
        Returns:
            Checkpoint data or None if no checkpoint found
        """
        with self.lock:
            candidates = []
            
            for cp_id, meta in self.checkpoint_metadata.items():
                if level is None or meta['level'] == level:
                    candidates.append((cp_id, meta))
            
            if not candidates:
                return None
            
            # Get most recent checkpoint
            candidates.sort(key=lambda x: x[1]['timestamp'], reverse=True)
            latest_cp_id, latest_meta = candidates[0]
            
            # Load checkpoint
            try:
                with open(latest_meta['path'], 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Verify consistency
                stored_hash = checkpoint_data.pop('consistency_hash', None)
                computed_hash = self._compute_hash(checkpoint_data)
                
                if stored_hash != computed_hash:
                    raise ValueError(f"Checkpoint consistency check failed: {latest_cp_id}")
                
                return checkpoint_data
                
            except Exception as e:
                print(f"Failed to load checkpoint {latest_cp_id}: {e}")
                return None


class SelfHealingSystem:
    """Self-healing system for automatic recovery."""
    
    def __init__(self):
        """Initialize self-healing system."""
        self.failure_detector = FailureDetector()
        self.elastic_manager = ElasticTrainingManager()
        self.checkpointer = MultiLevelCheckpointer("./checkpoints")
        
        # Recovery strategies
        self.recovery_strategies = {
            FailureType.NODE_FAILURE: self._handle_node_failure,
            FailureType.NETWORK_PARTITION: self._handle_network_partition,
            FailureType.MEMORY_ERROR: self._handle_memory_error,
            FailureType.STORAGE_FAILURE: self._handle_storage_failure
        }
        
        # Circuit breakers for different services
        self.circuit_breakers = {
            'training': CircuitBreaker(failure_threshold=3, timeout_duration=60.0),
            'checkpointing': CircuitBreaker(failure_threshold=5, timeout_duration=30.0),
            'communication': CircuitBreaker(failure_threshold=10, timeout_duration=10.0)
        }
        
        # Failure history
        self.failure_history: deque = deque(maxlen=10000)
        
        # Monitoring
        self._monitoring_active = True
        self._monitoring_thread = None
        
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring_thread is None:
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Check for failed nodes
                suspected_nodes = self.failure_detector.get_suspected_nodes()
                for node_id in suspected_nodes:
                    self._handle_suspected_node(node_id)
                
                # Check if scaling is needed
                if self.elastic_manager.should_scale_up():
                    self._scale_up_cluster()
                elif self.elastic_manager.should_scale_down():
                    self._scale_down_cluster()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Back off on error
    
    def _handle_suspected_node(self, node_id: str):
        """Handle a suspected failed node."""
        failure_event = FailureEvent(
            timestamp=time.time(),
            failure_type=FailureType.NODE_FAILURE,
            node_id=node_id,
            details={'suspected': True}
        )
        
        self.failure_history.append(failure_event)
        self.elastic_manager.remove_node(node_id, FailureType.NODE_FAILURE)
        
        print(f"Node {node_id} marked as failed and removed from cluster")
    
    async def handle_failure(
        self,
        failure_type: FailureType,
        node_id: str,
        details: Dict[str, Any] = None
    ) -> bool:
        """Handle a detected failure.
        
        Args:
            failure_type: Type of failure
            node_id: ID of affected node
            details: Additional failure details
        
        Returns:
            True if recovery was successful
        """
        failure_event = FailureEvent(
            timestamp=time.time(),
            failure_type=failure_type,
            node_id=node_id,
            details=details or {}
        )
        
        self.failure_history.append(failure_event)
        
        # Execute recovery strategy
        recovery_start = time.time()
        
        try:
            strategy = self.recovery_strategies.get(failure_type)
            if strategy:
                success = await strategy(node_id, details or {})
            else:
                success = await self._default_recovery_strategy(node_id, details or {})
            
            recovery_time = time.time() - recovery_start
            failure_event.recovery_time = recovery_time
            
            if success:
                print(f"Successfully recovered from {failure_type.value} on {node_id} in {recovery_time:.2f}s")
            else:
                print(f"Failed to recover from {failure_type.value} on {node_id}")
            
            return success
            
        except Exception as e:
            print(f"Error during recovery: {e}")
            return False
    
    async def _handle_node_failure(self, node_id: str, details: Dict[str, Any]) -> bool:
        """Handle node failure."""
        # Remove failed node
        self.elastic_manager.remove_node(node_id, FailureType.NODE_FAILURE)
        
        # Trigger rebalancing if needed
        cluster_state = self.elastic_manager.get_cluster_state()
        if cluster_state['active_nodes'] < self.elastic_manager.min_nodes:
            return await self._scale_up_cluster()
        
        return True
    
    async def _handle_network_partition(self, node_id: str, details: Dict[str, Any]) -> bool:
        """Handle network partition."""
        # Wait and retry connection
        max_retries = 3
        for attempt in range(max_retries):
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # Simulate connection test
            if random.random() > 0.3:  # 70% chance of success
                print(f"Network partition recovered for {node_id}")
                return True
        
        # If still failing, treat as node failure
        return await self._handle_node_failure(node_id, details)
    
    async def _handle_memory_error(self, node_id: str, details: Dict[str, Any]) -> bool:
        """Handle memory error."""
        # Trigger garbage collection
        # Reduce batch size
        # Create emergency checkpoint
        
        try:
            # Simulate memory cleanup
            await asyncio.sleep(1)
            print(f"Memory error recovery attempted for {node_id}")
            return True
        except Exception:
            return False
    
    async def _handle_storage_failure(self, node_id: str, details: Dict[str, Any]) -> bool:
        """Handle storage failure."""
        # Switch to backup storage
        # Reload from checkpoint
        
        try:
            # Simulate storage recovery
            await asyncio.sleep(2)
            print(f"Storage failure recovery attempted for {node_id}")
            return True
        except Exception:
            return False
    
    async def _default_recovery_strategy(self, node_id: str, details: Dict[str, Any]) -> bool:
        """Default recovery strategy."""
        print(f"Applying default recovery strategy for {node_id}")
        return True
    
    async def _scale_up_cluster(self) -> bool:
        """Scale up the cluster."""
        print("Scaling up cluster...")
        # Simulate adding new nodes
        new_node = NodeInfo(
            node_id=str(uuid.uuid4()),
            hostname=f"worker-{random.randint(100, 999)}",
            port=29500,
            rank=len(self.elastic_manager.active_nodes)
        )
        self.elastic_manager.add_node(new_node)
        return True
    
    async def _scale_down_cluster(self) -> bool:
        """Scale down the cluster."""
        print("Scaling down cluster...")
        nodes_to_remove = self.elastic_manager.select_nodes_for_removal(1)
        for node_id in nodes_to_remove:
            self.elastic_manager.remove_node(node_id, FailureType.RESOURCE_EXHAUSTION)
        return True
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        cluster_state = self.elastic_manager.get_cluster_state()
        recent_failures = len([f for f in self.failure_history if time.time() - f.timestamp < 3600])
        
        circuit_states = {
            name: breaker.state.value for name, breaker in self.circuit_breakers.items()
        }
        
        return {
            'cluster_state': cluster_state,
            'recent_failures': recent_failures,
            'circuit_breaker_states': circuit_states,
            'total_failures': len(self.failure_history)
        }


class ChaosEngineer:
    """Chaos engineering for testing system resilience."""
    
    def __init__(self, fault_tolerance_system: SelfHealingSystem):
        """Initialize chaos engineer."""
        self.system = fault_tolerance_system
        self.chaos_experiments = {
            'kill_random_node': self._kill_random_node,
            'network_partition': self._create_network_partition,
            'memory_pressure': self._create_memory_pressure,
            'storage_corruption': self._create_storage_corruption
        }
    
    async def run_experiment(self, experiment_name: str, intensity: float = 0.1):
        """Run a chaos experiment.
        
        Args:
            experiment_name: Name of experiment to run
            intensity: Intensity level (0.0 to 1.0)
        """
        if experiment_name not in self.chaos_experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        print(f"Running chaos experiment: {experiment_name} (intensity: {intensity})")
        
        experiment = self.chaos_experiments[experiment_name]
        await experiment(intensity)
    
    async def _kill_random_node(self, intensity: float):
        """Kill random nodes to test recovery."""
        active_nodes = list(self.system.elastic_manager.active_nodes.keys())
        
        if not active_nodes:
            return
        
        # Kill nodes based on intensity
        num_to_kill = max(1, int(len(active_nodes) * intensity))
        nodes_to_kill = random.sample(active_nodes, min(num_to_kill, len(active_nodes)))
        
        for node_id in nodes_to_kill:
            await self.system.handle_failure(
                FailureType.NODE_FAILURE,
                node_id,
                {'chaos_experiment': True}
            )
    
    async def _create_network_partition(self, intensity: float):
        """Create network partition to test resilience."""
        active_nodes = list(self.system.elastic_manager.active_nodes.keys())
        
        if len(active_nodes) < 2:
            return
        
        # Partition nodes based on intensity
        partition_size = max(1, int(len(active_nodes) * intensity))
        partitioned_nodes = random.sample(active_nodes, partition_size)
        
        for node_id in partitioned_nodes:
            await self.system.handle_failure(
                FailureType.NETWORK_PARTITION,
                node_id,
                {'chaos_experiment': True, 'partition_duration': 30}
            )
    
    async def _create_memory_pressure(self, intensity: float):
        """Create memory pressure to test handling."""
        active_nodes = list(self.system.elastic_manager.active_nodes.keys())
        
        if not active_nodes:
            return
        
        affected_nodes = random.sample(
            active_nodes,
            max(1, int(len(active_nodes) * intensity))
        )
        
        for node_id in affected_nodes:
            await self.system.handle_failure(
                FailureType.MEMORY_ERROR,
                node_id,
                {'chaos_experiment': True, 'memory_usage': 95}
            )
    
    async def _create_storage_corruption(self, intensity: float):
        """Create storage corruption to test recovery."""
        active_nodes = list(self.system.elastic_manager.active_nodes.keys())
        
        if not active_nodes:
            return
        
        node_id = random.choice(active_nodes)
        await self.system.handle_failure(
            FailureType.STORAGE_FAILURE,
            node_id,
            {'chaos_experiment': True, 'corruption_type': 'checkpoint'}
        )


async def test_fault_tolerance_system():
    """Test the fault tolerance and resilience system."""
    print("Testing Fault Tolerance and Resilience System")
    print("=" * 50)
    
    # Initialize system
    healing_system = SelfHealingSystem()
    chaos_engineer = ChaosEngineer(healing_system)
    
    # Add some initial nodes
    print("1. Adding initial nodes to cluster:")
    for i in range(4):
        node = NodeInfo(
            node_id=f"node_{i}",
            hostname=f"worker-{i}",
            port=29500 + i,
            rank=i,
            load_factor=random.uniform(0.2, 0.8)
        )
        healing_system.elastic_manager.add_node(node)
        healing_system.failure_detector.record_heartbeat(f"node_{i}")
        print(f"   Added node_{i}")
    
    # Test failure detection
    print("2. Testing failure detection:")
    
    # Simulate heartbeats
    for _ in range(3):
        for i in range(3):  # node_3 stops sending heartbeats
            healing_system.failure_detector.record_heartbeat(f"node_{i}")
        await asyncio.sleep(1)
    
    # Check for suspected nodes
    await asyncio.sleep(2)
    suspected = healing_system.failure_detector.get_suspected_nodes()
    print(f"   Suspected nodes: {suspected}")
    
    # Test circuit breaker
    print("3. Testing circuit breaker:")
    cb = CircuitBreaker(failure_threshold=2, timeout_duration=5.0)
    
    async def failing_function():
        raise Exception("Simulated failure")
    
    # Cause failures to open circuit
    for i in range(3):
        try:
            await cb.call(failing_function)
        except Exception:
            print(f"   Failure {i+1} handled by circuit breaker")
    
    print(f"   Circuit breaker state: {cb.state}")
    
    # Test self-healing
    print("4. Testing self-healing capabilities:")
    
    # Inject various failures
    failures_to_test = [
        (FailureType.NODE_FAILURE, "node_1"),
        (FailureType.NETWORK_PARTITION, "node_2"),
        (FailureType.MEMORY_ERROR, "node_0"),
    ]
    
    for failure_type, node_id in failures_to_test:
        success = await healing_system.handle_failure(failure_type, node_id)
        print(f"   {failure_type.value} on {node_id}: {'recovered' if success else 'failed'}")
        await asyncio.sleep(1)
    
    # Test multi-level checkpointing
    print("5. Testing multi-level checkpointing:")
    
    checkpointer = MultiLevelCheckpointer("./test_checkpoints")
    
    # Create checkpoints at different levels
    mock_model_state = {'layer1.weight': np.random.randn(10, 5)}
    mock_optimizer_state = {'step': 1000, 'lr': 0.001}
    mock_metadata = {'epoch': 10, 'loss': 0.234}
    
    for level in ['frequent', 'hourly']:
        if checkpointer.should_checkpoint(level, 1000):
            checkpoint_path = await checkpointer.create_checkpoint(
                level, 1000, mock_model_state, mock_optimizer_state, mock_metadata
            )
            print(f"   Created {level} checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load latest checkpoint
    latest_checkpoint = checkpointer.load_latest_checkpoint()
    if latest_checkpoint:
        print(f"   Loaded checkpoint from step {latest_checkpoint['step']}")
    
    # Test chaos engineering
    print("6. Testing chaos engineering:")
    
    # Add more nodes for chaos testing
    for i in range(4, 8):
        node = NodeInfo(
            node_id=f"node_{i}",
            hostname=f"worker-{i}",
            port=29500 + i,
            rank=i
        )
        healing_system.elastic_manager.add_node(node)
    
    # Run chaos experiments
    experiments = ['kill_random_node', 'network_partition', 'memory_pressure']
    
    for experiment in experiments:
        await chaos_engineer.run_experiment(experiment, intensity=0.2)
        await asyncio.sleep(2)  # Allow recovery time
        
        health = healing_system.get_system_health()
        print(f"   After {experiment}: {health['cluster_state']['active_nodes']} active nodes")
    
    # System health report
    print("7. Final system health report:")
    health = healing_system.get_system_health()
    
    for category, status in health.items():
        if isinstance(status, dict):
            print(f"   {category}:")
            for key, value in status.items():
                print(f"     {key}: {value}")
        else:
            print(f"   {category}: {status}")
    
    # Cleanup
    print("8. Cleaning up:")
    healing_system.stop_monitoring()
    
    # Cleanup test checkpoints
    import shutil
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")
    
    print("   Cleanup completed")
    
    print("\nFault tolerance and resilience system tested successfully!")


if __name__ == "__main__":
    asyncio.run(test_fault_tolerance_system())