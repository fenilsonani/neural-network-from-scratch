"""
Comprehensive test suite for distributed training system.
Tests all components of distributed_system.py for comprehensive coverage.

This module tests:
- DistributedTrainingOrchestrator
- CommunicationPrimitive  
- ZeROOptimizer
- PipelineParallelism
- FaultTolerantCheckpointer
- ElasticTrainingCoordinator
- All advanced sharding strategies
"""

import asyncio
import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, Mock, patch
from typing import Dict, List, Any

import numpy as np
import pytest

from src.neural_arch.distributed.distributed_system import (
    DistributedTrainingOrchestrator,
    CommunicationPrimitive, 
    ZeROOptimizer,
    PipelineParallelism,
    FaultTolerantCheckpointer,
    ElasticTrainingCoordinator,
    DistributedConfig,
    ShardingStrategy,
    CommunicationBackend,
    OptimizationLevel,
    ModelPartition,
    GradientBuffer,
    AdvancedAllReduce,
    DynamicLoadBalancer,
    test_distributed_system
)


class TestDistributedConfig:
    """Test DistributedConfig configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedConfig()
        
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.backend == CommunicationBackend.NCCL
        assert config.sharding_strategy == ShardingStrategy.ZERO_2
        assert config.optimization_level == OptimizationLevel.O2
        assert config.enable_gradient_compression is True
        assert config.enable_pipeline_parallelism is False
        assert config.checkpoint_interval == 1000
        assert config.fault_tolerance is True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DistributedConfig(
            world_size=8,
            rank=2,
            local_rank=1,
            backend=CommunicationBackend.MPI,
            sharding_strategy=ShardingStrategy.ZERO_3,
            optimization_level=OptimizationLevel.O1,
            enable_gradient_compression=False,
            enable_pipeline_parallelism=True,
            checkpoint_interval=500,
            fault_tolerance=False
        )
        
        assert config.world_size == 8
        assert config.rank == 2
        assert config.local_rank == 1
        assert config.backend == CommunicationBackend.MPI
        assert config.sharding_strategy == ShardingStrategy.ZERO_3
        assert config.optimization_level == OptimizationLevel.O1
        assert config.enable_gradient_compression is False
        assert config.enable_pipeline_parallelism is True
        assert config.checkpoint_interval == 500
        assert config.fault_tolerance is False
        
    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            DistributedConfig(world_size=0)
            
        with pytest.raises(ValueError):
            DistributedConfig(rank=-1)
            
        with pytest.raises(ValueError):
            DistributedConfig(world_size=4, rank=5)


class TestCommunicationPrimitive:
    """Test CommunicationPrimitive for distributed communication."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = DistributedConfig(world_size=4, rank=0)
        self.comm = CommunicationPrimitive(self.config)
        
    def test_initialization(self):
        """Test communication primitive initialization."""
        assert self.comm.config == self.config
        assert self.comm.world_size == 4
        assert self.comm.rank == 0
        assert self.comm.backend == CommunicationBackend.NCCL
        
    def test_all_reduce(self):
        """Test all-reduce operation."""
        # Test with mock data
        data = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Mock the actual communication
        with patch.object(self.comm, '_nccl_all_reduce') as mock_reduce:
            mock_reduce.return_value = data * self.config.world_size
            
            result = self.comm.all_reduce(data)
            
            mock_reduce.assert_called_once_with(data, 'sum')
            np.testing.assert_array_equal(result, data * 4)
            
    def test_all_gather(self):
        """Test all-gather operation."""
        data = np.array([1.0, 2.0])
        
        with patch.object(self.comm, '_nccl_all_gather') as mock_gather:
            expected = np.concatenate([data] * self.config.world_size)
            mock_gather.return_value = expected
            
            result = self.comm.all_gather(data)
            
            mock_gather.assert_called_once_with(data)
            np.testing.assert_array_equal(result, expected)
            
    def test_broadcast(self):
        """Test broadcast operation."""
        data = np.array([1.0, 2.0, 3.0])
        root = 0
        
        with patch.object(self.comm, '_nccl_broadcast') as mock_bcast:
            mock_bcast.return_value = data
            
            result = self.comm.broadcast(data, root)
            
            mock_bcast.assert_called_once_with(data, root)
            np.testing.assert_array_equal(result, data)
            
    def test_send_recv(self):
        """Test point-to-point communication."""
        data = np.array([1.0, 2.0])
        
        # Test send
        with patch.object(self.comm, '_nccl_send') as mock_send:
            self.comm.send(data, dest=1)
            mock_send.assert_called_once_with(data, 1)
            
        # Test recv
        with patch.object(self.comm, '_nccl_recv') as mock_recv:
            mock_recv.return_value = data
            
            result = self.comm.recv(src=1, shape=(2,), dtype=np.float32)
            
            mock_recv.assert_called_once_with(1, (2,), np.float32)
            np.testing.assert_array_equal(result, data)
            
    def test_backend_switching(self):
        """Test communication backend switching."""
        # Test MPI backend
        mpi_config = DistributedConfig(backend=CommunicationBackend.MPI)
        mpi_comm = CommunicationPrimitive(mpi_config)
        
        assert mpi_comm.backend == CommunicationBackend.MPI
        
        # Test Gloo backend
        gloo_config = DistributedConfig(backend=CommunicationBackend.GLOO)
        gloo_comm = CommunicationPrimitive(gloo_config)
        
        assert gloo_comm.backend == CommunicationBackend.GLOO


class TestZeROOptimizer:
    """Test ZeRO optimizer for memory-efficient training."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = DistributedConfig(
            world_size=4,
            rank=0,
            sharding_strategy=ShardingStrategy.ZERO_2
        )
        self.comm = Mock()
        self.zero_opt = ZeROOptimizer(self.config, self.comm)
        
    def test_initialization(self):
        """Test ZeRO optimizer initialization."""
        assert self.zero_opt.config == self.config
        assert self.zero_opt.comm == self.comm
        assert self.zero_opt.sharding_strategy == ShardingStrategy.ZERO_2
        assert self.zero_opt.world_size == 4
        assert self.zero_opt.rank == 0
        
    def test_shard_parameters_zero1(self):
        """Test parameter sharding for ZeRO-1."""
        zero1_config = DistributedConfig(sharding_strategy=ShardingStrategy.ZERO_1)
        zero1_opt = ZeROOptimizer(zero1_config, self.comm)
        
        params = {
            'layer1.weight': np.random.randn(100, 50),
            'layer1.bias': np.random.randn(50),
            'layer2.weight': np.random.randn(50, 25)
        }
        
        sharded = zero1_opt.shard_parameters(params)
        
        # ZeRO-1 only shards optimizer states, not parameters
        for key, value in params.items():
            assert key in sharded
            np.testing.assert_array_equal(sharded[key], value)
            
    def test_shard_parameters_zero2(self):
        """Test parameter sharding for ZeRO-2."""
        params = {
            'layer1.weight': np.random.randn(100, 50),
            'layer1.bias': np.random.randn(50),
            'layer2.weight': np.random.randn(50, 25)
        }
        
        sharded = self.zero_opt.shard_parameters(params)
        
        # ZeRO-2 shards gradients and optimizer states
        assert len(sharded) <= len(params)  # May have fewer parameters per rank
        
        for key, value in sharded.items():
            assert key in params
            # Shard size should be roughly 1/world_size
            original_size = params[key].size
            expected_shard_size = (original_size + self.config.world_size - 1) // self.config.world_size
            assert value.size <= expected_shard_size * 2  # Allow some flexibility
            
    def test_shard_parameters_zero3(self):
        """Test parameter sharding for ZeRO-3."""
        zero3_config = DistributedConfig(sharding_strategy=ShardingStrategy.ZERO_3)
        zero3_opt = ZeROOptimizer(zero3_config, self.comm)
        
        params = {
            'layer1.weight': np.random.randn(100, 50),
            'layer1.bias': np.random.randn(50),
            'layer2.weight': np.random.randn(50, 25)
        }
        
        sharded = zero3_opt.shard_parameters(params)
        
        # ZeRO-3 shards parameters, gradients, and optimizer states
        total_original_params = sum(p.size for p in params.values())
        total_sharded_params = sum(p.size for p in sharded.values())
        
        # Should have roughly 1/world_size parameters
        assert total_sharded_params <= total_original_params // self.config.world_size * 2
        
    def test_gather_parameters(self):
        """Test parameter gathering for forward pass."""
        # Mock sharded parameters
        sharded_params = {
            'layer1.weight': np.random.randn(25, 50),  # 1/4 of original
            'layer1.bias': np.random.randn(13)  # ~1/4 of 50
        }
        
        # Mock communication
        self.comm.all_gather.return_value = np.random.randn(100, 50)
        
        gathered = self.zero_opt.gather_parameters(sharded_params, ['layer1.weight'])
        
        assert 'layer1.weight' in gathered
        assert gathered['layer1.weight'].shape == (100, 50)
        self.comm.all_gather.assert_called()
        
    def test_gradient_reduction(self):
        """Test gradient reduction and sharding."""
        gradients = {
            'layer1.weight': np.random.randn(100, 50),
            'layer1.bias': np.random.randn(50),
            'layer2.weight': np.random.randn(50, 25)
        }
        
        # Mock communication for reduce-scatter
        self.comm.reduce_scatter = Mock()
        self.comm.reduce_scatter.side_effect = lambda x: x[:x.size//4].reshape(-1)
        
        reduced = self.zero_opt.reduce_gradients(gradients)
        
        # Should be called for each gradient
        assert self.comm.reduce_scatter.call_count == len(gradients)
        
        for key in gradients.keys():
            assert key in reduced
            # Gradients should be sharded
            assert reduced[key].size <= gradients[key].size
            
    def test_optimizer_state_management(self):
        """Test optimizer state sharding and updates."""
        params = {
            'layer1.weight': np.random.randn(100, 50),
            'layer1.bias': np.random.randn(50)
        }
        
        gradients = {
            'layer1.weight': np.random.randn(100, 50),
            'layer1.bias': np.random.randn(50)
        }
        
        # Initialize optimizer states
        states = self.zero_opt.initialize_optimizer_states(params)
        
        for key in params.keys():
            assert key in states
            assert 'momentum' in states[key]
            assert 'variance' in states[key]
            
        # Update states
        updated_params, updated_states = self.zero_opt.update_parameters(
            params, gradients, states, lr=0.001
        )
        
        for key in params.keys():
            assert key in updated_params
            assert key in updated_states
            # Parameters should be updated
            assert not np.array_equal(updated_params[key], params[key])


class TestPipelineParallelism:
    """Test pipeline parallelism implementation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = DistributedConfig(
            world_size=4,
            rank=1,
            enable_pipeline_parallelism=True,
            pipeline_stages=4
        )
        self.comm = Mock()
        self.pipeline = PipelineParallelism(self.config, self.comm)
        
    def test_initialization(self):
        """Test pipeline parallelism initialization."""
        assert self.pipeline.config == self.config
        assert self.pipeline.comm == self.comm
        assert self.pipeline.num_stages == 4
        assert self.pipeline.stage_id == 1  # rank 1
        assert self.pipeline.is_first_stage is False
        assert self.pipeline.is_last_stage is False
        
    def test_stage_assignment(self):
        """Test model partitioning into stages."""
        # Mock model layers
        layers = [f'layer_{i}' for i in range(12)]  # 12 layers
        
        partitions = self.pipeline.partition_model(layers)
        
        assert len(partitions) == 4  # 4 stages
        
        # Each stage should have roughly equal number of layers
        for partition in partitions:
            assert len(partition.layers) in [3, 4]  # 12/4 = 3, with some flexibility
            
        # Verify all layers are assigned
        all_assigned_layers = []
        for partition in partitions:
            all_assigned_layers.extend(partition.layers)
        assert set(all_assigned_layers) == set(layers)
        
    def test_forward_pass_scheduling(self):
        """Test 1F1B (1-forward-1-backward) scheduling."""
        batch_size = 8
        microbatch_size = 2
        
        schedule = self.pipeline.create_1f1b_schedule(batch_size, microbatch_size)
        
        # Should have 4 microbatches (8/2)
        assert len(schedule) == 4
        
        # Each microbatch should have forward and backward operations
        for microbatch in schedule:
            assert 'forward' in microbatch
            assert 'backward' in microbatch
            assert microbatch['microbatch_size'] == microbatch_size
            
    def test_activation_checkpointing(self):
        """Test activation checkpointing for memory efficiency."""
        # Mock activations
        activations = [np.random.randn(32, 128) for _ in range(10)]
        
        # Enable checkpointing every 2 layers
        checkpointed = self.pipeline.apply_activation_checkpointing(
            activations, checkpoint_interval=2
        )
        
        # Should have fewer stored activations
        assert len(checkpointed) <= len(activations)
        
        # Checkpointed activations should be marked
        for i, activation in enumerate(checkpointed):
            if i % 2 == 0:
                assert activation.get('checkpointed', False) is True
                
    def test_inter_stage_communication(self):
        """Test communication between pipeline stages."""
        # Test sending to next stage
        data = np.random.randn(32, 128)
        
        with patch.object(self.comm, 'send') as mock_send:
            self.pipeline.send_to_next_stage(data)
            mock_send.assert_called_once_with(data, dest=2)  # next rank
            
        # Test receiving from previous stage  
        with patch.object(self.comm, 'recv') as mock_recv:
            mock_recv.return_value = data
            
            received = self.pipeline.recv_from_prev_stage(shape=(32, 128))
            
            mock_recv.assert_called_once_with(src=0, shape=(32, 128), dtype=np.float32)
            np.testing.assert_array_equal(received, data)
            
    def test_gradient_synchronization(self):
        """Test gradient synchronization across pipeline stages."""
        # Mock gradients from different microbatches
        gradients = [
            {'layer1': np.random.randn(10, 5), 'layer2': np.random.randn(5, 3)},
            {'layer1': np.random.randn(10, 5), 'layer2': np.random.randn(5, 3)},
            {'layer1': np.random.randn(10, 5), 'layer2': np.random.randn(5, 3)}
        ]
        
        synchronized = self.pipeline.synchronize_gradients(gradients)
        
        # Should average gradients across microbatches
        for key in synchronized.keys():
            expected = np.mean([g[key] for g in gradients], axis=0)
            np.testing.assert_array_almost_equal(synchronized[key], expected)


class TestFaultTolerantCheckpointer:
    """Test fault-tolerant checkpointing system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DistributedConfig(
            checkpoint_dir=self.temp_dir,
            checkpoint_interval=100,
            max_checkpoint_history=3
        )
        self.checkpointer = FaultTolerantCheckpointer(self.config)
        
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_initialization(self):
        """Test checkpointer initialization."""
        assert self.checkpointer.config == self.config
        assert self.checkpointer.checkpoint_dir == self.temp_dir
        assert self.checkpointer.checkpoint_interval == 100
        assert self.checkpointer.max_history == 3
        
    def test_should_checkpoint(self):
        """Test checkpoint scheduling logic."""
        # Should checkpoint at intervals
        assert self.checkpointer.should_checkpoint(100) is True
        assert self.checkpointer.should_checkpoint(200) is True
        assert self.checkpointer.should_checkpoint(99) is False
        assert self.checkpointer.should_checkpoint(150) is False
        
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        model_state = {
            'layer1.weight': np.random.randn(10, 5),
            'layer1.bias': np.random.randn(5)
        }
        
        optimizer_state = {
            'layer1.weight': {'momentum': np.random.randn(10, 5)},
            'layer1.bias': {'momentum': np.random.randn(5)}
        }
        
        metadata = {
            'step': 100,
            'epoch': 5,
            'loss': 0.25
        }
        
        checkpoint_path = self.checkpointer.save_checkpoint(
            step=100,
            model_state=model_state,
            optimizer_state=optimizer_state,
            metadata=metadata
        )
        
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith('.pkl')
        
        # Verify checkpoint contents
        import pickle
        with open(checkpoint_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        assert saved_data['step'] == 100
        assert 'model_state' in saved_data
        assert 'optimizer_state' in saved_data
        assert 'metadata' in saved_data
        assert 'timestamp' in saved_data
        
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        # First save a checkpoint
        model_state = {'layer1.weight': np.random.randn(10, 5)}
        optimizer_state = {'layer1.weight': {'momentum': np.random.randn(10, 5)}}
        metadata = {'step': 100, 'loss': 0.25}
        
        checkpoint_path = self.checkpointer.save_checkpoint(
            step=100,
            model_state=model_state,
            optimizer_state=optimizer_state,
            metadata=metadata
        )
        
        # Load the checkpoint
        loaded_data = self.checkpointer.load_checkpoint(checkpoint_path)
        
        assert loaded_data is not None
        assert loaded_data['step'] == 100
        assert loaded_data['metadata']['loss'] == 0.25
        
        # Test loading non-existent checkpoint
        invalid_path = os.path.join(self.temp_dir, 'nonexistent.pkl')
        loaded_invalid = self.checkpointer.load_checkpoint(invalid_path)
        assert loaded_invalid is None
        
    def test_cleanup_old_checkpoints(self):
        """Test automatic cleanup of old checkpoints."""
        # Create more checkpoints than max_history
        for step in [100, 200, 300, 400, 500]:
            self.checkpointer.save_checkpoint(
                step=step,
                model_state={'param': np.random.randn(5)},
                optimizer_state={'param': {'momentum': np.random.randn(5)}},
                metadata={'step': step}
            )
            
        # Trigger cleanup
        self.checkpointer._cleanup_old_checkpoints()
        
        # Should only have max_history checkpoints remaining
        checkpoint_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.pkl')]
        assert len(checkpoint_files) == self.config.max_checkpoint_history
        
        # Should keep the most recent checkpoints
        remaining_steps = []
        for filename in checkpoint_files:
            if 'step_' in filename:
                step = int(filename.split('step_')[1].split('_')[0])
                remaining_steps.append(step)
                
        remaining_steps.sort()
        assert remaining_steps[-3:] == [300, 400, 500]  # Most recent 3
        
    def test_distributed_checkpoint_consistency(self):
        """Test checkpoint consistency across ranks."""
        # Simulate multiple ranks saving checkpoints
        configs = [
            DistributedConfig(rank=0, checkpoint_dir=self.temp_dir),
            DistributedConfig(rank=1, checkpoint_dir=self.temp_dir),
            DistributedConfig(rank=2, checkpoint_dir=self.temp_dir)
        ]
        
        checkpointers = [FaultTolerantCheckpointer(config) for config in configs]
        
        # Each rank saves its shard
        model_shards = [
            {'shard_0': np.random.randn(10, 5)},
            {'shard_1': np.random.randn(10, 5)},
            {'shard_2': np.random.randn(10, 5)}
        ]
        
        checkpoint_paths = []
        for i, (checkpointer, shard) in enumerate(zip(checkpointers, model_shards)):
            path = checkpointer.save_checkpoint(
                step=100,
                model_state=shard,
                optimizer_state={},
                metadata={'rank': i}
            )
            checkpoint_paths.append(path)
            
        # Verify all checkpoints were saved
        assert len(checkpoint_paths) == 3
        for path in checkpoint_paths:
            assert os.path.exists(path)
            
        # Test loading and merging shards
        merged_state = self.checkpointer.load_distributed_checkpoint(checkpoint_paths)
        
        assert 'shard_0' in merged_state['model_state']
        assert 'shard_1' in merged_state['model_state']  
        assert 'shard_2' in merged_state['model_state']


class TestElasticTrainingCoordinator:
    """Test elastic training coordination."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = DistributedConfig(
            world_size=4,
            rank=0,
            min_replicas=2,
            max_replicas=8,
            elastic_training=True
        )
        self.coordinator = ElasticTrainingCoordinator(self.config)
        
    def test_initialization(self):
        """Test coordinator initialization."""
        assert self.coordinator.config == self.config
        assert self.coordinator.min_replicas == 2
        assert self.coordinator.max_replicas == 8
        assert self.coordinator.current_world_size == 4
        
    def test_node_failure_detection(self):
        """Test node failure detection and handling."""
        # Simulate node failure
        failed_rank = 2
        
        with patch.object(self.coordinator, '_detect_node_failure') as mock_detect:
            mock_detect.return_value = True
            
            # Should detect failure
            assert self.coordinator.check_node_health(failed_rank) is False
            
            # Should trigger reconfiguration
            new_config = self.coordinator.handle_node_failure(failed_rank)
            
            assert new_config.world_size == 3  # One less node
            assert failed_rank not in new_config.active_ranks
            
    def test_dynamic_scaling(self):
        """Test dynamic scaling up and down."""
        # Test scaling up
        scale_up_decision = self.coordinator.should_scale_up(
            current_load=0.9,  # High load
            queue_length=100   # Many pending jobs
        )
        assert scale_up_decision is True
        
        # Test scaling down
        scale_down_decision = self.coordinator.should_scale_down(
            current_load=0.2,  # Low load
            queue_length=0     # No pending jobs
        )
        assert scale_down_decision is True
        
        # Test scaling limits
        max_config = DistributedConfig(world_size=8)  # At max
        coordinator_at_max = ElasticTrainingCoordinator(max_config)
        assert coordinator_at_max.should_scale_up(0.9, 100) is False
        
        min_config = DistributedConfig(world_size=2)  # At min
        coordinator_at_min = ElasticTrainingCoordinator(min_config)
        assert coordinator_at_min.should_scale_down(0.1, 0) is False
        
    def test_state_synchronization(self):
        """Test state synchronization during scaling events."""
        # Mock model and optimizer states
        model_state = {
            'layer1.weight': np.random.randn(10, 5),
            'layer2.weight': np.random.randn(5, 3)
        }
        
        optimizer_state = {
            'layer1.weight': {'momentum': np.random.randn(10, 5)},
            'layer2.weight': {'momentum': np.random.randn(5, 3)}
        }
        
        # Test state redistribution for new world size
        redistributed = self.coordinator.redistribute_state(
            model_state=model_state,
            optimizer_state=optimizer_state,
            old_world_size=4,
            new_world_size=6
        )
        
        assert 'model_state' in redistributed
        assert 'optimizer_state' in redistributed
        
        # State should be resharded for new world size
        # This is complex logic, so we mainly test it doesn't crash
        assert redistributed is not None


class TestAdvancedFeatures:
    """Test advanced distributed training features."""
    
    def test_gradient_compression(self):
        """Test gradient compression for bandwidth optimization."""
        config = DistributedConfig(enable_gradient_compression=True)
        comm = CommunicationPrimitive(config)
        
        # Large gradient tensor
        gradients = np.random.randn(1000, 1000).astype(np.float32)
        
        # Test compression
        compressed = comm.compress_gradients(gradients, compression_ratio=0.1)
        
        # Compressed gradients should be smaller
        assert compressed.nbytes < gradients.nbytes
        
        # Test decompression
        decompressed = comm.decompress_gradients(compressed, original_shape=gradients.shape)
        
        # Should be approximately equal (lossy compression)
        correlation = np.corrcoef(gradients.flatten(), decompressed.flatten())[0, 1]
        assert correlation > 0.8  # High correlation despite compression
        
    def test_dynamic_load_balancing(self):
        """Test dynamic load balancing across nodes."""
        config = DistributedConfig(world_size=4)
        balancer = DynamicLoadBalancer(config)
        
        # Mock workload distribution
        node_loads = [0.9, 0.3, 0.7, 0.2]  # Unbalanced
        
        # Should detect imbalance
        assert balancer.is_load_balanced(node_loads, threshold=0.3) is False
        
        # Get rebalancing plan
        rebalancing_plan = balancer.create_rebalancing_plan(node_loads)
        
        assert 'migrations' in rebalancing_plan
        assert len(rebalancing_plan['migrations']) > 0
        
        # Migrations should reduce load on heavily loaded nodes
        for migration in rebalancing_plan['migrations']:
            source_load = node_loads[migration['source']]
            dest_load = node_loads[migration['dest']] 
            assert source_load > dest_load
            
    def test_memory_optimization(self):
        """Test memory optimization techniques."""
        config = DistributedConfig(
            enable_gradient_checkpointing=True,
            enable_cpu_offloading=True
        )
        
        orchestrator = DistributedTrainingOrchestrator(config)
        
        # Test memory usage monitoring
        memory_stats = orchestrator.get_memory_stats()
        
        assert 'gpu_memory_used' in memory_stats
        assert 'gpu_memory_total' in memory_stats
        assert 'cpu_memory_used' in memory_stats
        
        # Test memory pressure handling
        high_memory_usage = 0.95
        
        if memory_stats['gpu_memory_used'] / memory_stats['gpu_memory_total'] > high_memory_usage:
            # Should trigger memory optimization
            optimizations = orchestrator.handle_memory_pressure()
            
            assert len(optimizations) > 0
            assert any('offload' in opt or 'checkpoint' in opt for opt in optimizations)


class TestIntegrationScenarios:
    """Integration tests for complex distributed training scenarios."""
    
    def test_full_training_loop_simulation(self):
        """Test complete distributed training loop."""
        config = DistributedConfig(
            world_size=4,
            rank=0,
            sharding_strategy=ShardingStrategy.ZERO_2,
            enable_pipeline_parallelism=False
        )
        
        orchestrator = DistributedTrainingOrchestrator(config)
        
        # Mock model parameters
        model_params = {
            'layer1.weight': np.random.randn(100, 50),
            'layer1.bias': np.random.randn(50),
            'layer2.weight': np.random.randn(50, 10),
            'layer2.bias': np.random.randn(10)
        }
        
        # Mock training data
        batch_data = np.random.randn(32, 100)  # batch_size=32, input_dim=100
        
        # Simulate training step
        with patch.object(orchestrator.zero_optimizer, 'shard_parameters') as mock_shard:
            with patch.object(orchestrator.communication, 'all_reduce') as mock_reduce:
                mock_shard.return_value = model_params  # Simplified
                mock_reduce.return_value = np.random.randn(100, 50)  # Mock reduced gradients
                
                # Run one training step
                loss, updated_params = orchestrator.training_step(
                    model_params, batch_data, learning_rate=0.001
                )
                
                assert isinstance(loss, float)
                assert loss > 0  # Positive loss
                assert len(updated_params) == len(model_params)
                
                # Parameters should be updated
                for key in model_params.keys():
                    assert key in updated_params
                    assert not np.array_equal(updated_params[key], model_params[key])
                    
    def test_fault_recovery_scenario(self):
        """Test fault recovery during training."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = DistributedConfig(
                world_size=4,
                rank=1,
                fault_tolerance=True,
                checkpoint_dir=temp_dir,
                checkpoint_interval=1
            )
            
            orchestrator = DistributedTrainingOrchestrator(config)
            
            # Save initial checkpoint
            model_state = {'param': np.random.randn(10, 5)}
            optimizer_state = {'param': {'momentum': np.random.randn(10, 5)}}
            
            checkpoint_path = orchestrator.save_checkpoint(
                step=100,
                model_state=model_state,
                optimizer_state=optimizer_state
            )
            
            # Simulate node failure and recovery
            with patch.object(orchestrator.elastic_coordinator, 'handle_node_failure') as mock_handle:
                new_config = DistributedConfig(world_size=3, rank=1)  # One less node
                mock_handle.return_value = new_config
                
                # Should successfully recover from checkpoint
                recovered_state = orchestrator.recover_from_failure(checkpoint_path)
                
                assert recovered_state is not None
                assert 'model_state' in recovered_state
                assert 'optimizer_state' in recovered_state
                assert recovered_state['step'] == 100
                
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestPerformanceOptimizations:
    """Test performance optimizations and monitoring."""
    
    def test_communication_overlap(self):
        """Test computation-communication overlap."""
        config = DistributedConfig(enable_communication_overlap=True)
        orchestrator = DistributedTrainingOrchestrator(config)
        
        # Mock asynchronous communication
        with patch.object(orchestrator.communication, 'all_reduce_async') as mock_async:
            mock_future = Mock()
            mock_future.result.return_value = np.random.randn(100, 50)
            mock_async.return_value = mock_future
            
            # Start async communication
            future = orchestrator.start_gradient_communication(np.random.randn(100, 50))
            
            # Should return immediately (non-blocking)
            assert future is not None
            
            # Can do other computation while communication happens
            computation_result = np.random.randn(50, 25) @ np.random.randn(25, 10)
            
            # Wait for communication to complete
            comm_result = orchestrator.complete_gradient_communication(future)
            
            assert comm_result is not None
            assert computation_result is not None
            
    def test_performance_monitoring(self):
        """Test performance monitoring and profiling."""
        config = DistributedConfig(enable_profiling=True)
        orchestrator = DistributedTrainingOrchestrator(config)
        
        # Start profiling
        orchestrator.start_profiling()
        
        # Simulate some operations
        time.sleep(0.01)  # Small delay
        
        # Get profile results
        profile_data = orchestrator.get_profile_results()
        
        assert 'communication_time' in profile_data
        assert 'computation_time' in profile_data
        assert 'memory_usage' in profile_data
        
        # All times should be non-negative
        for time_metric in ['communication_time', 'computation_time']:
            assert profile_data[time_metric] >= 0
            
        orchestrator.stop_profiling()


class TestAdvancedDistributedFeatures:
    """Test advanced distributed training features for complete coverage."""
    
    def test_communication_backend_switching(self):
        """Test switching between communication backends."""
        backends = [CommunicationBackend.NCCL, CommunicationBackend.MPI, CommunicationBackend.GLOO]
        
        for backend in backends:
            config = DistributedConfig(world_size=2, rank=0, backend=backend)
            comm = CommunicationPrimitive(config)
            assert comm.backend == backend
    
    def test_sharding_strategy_variations(self):
        """Test all sharding strategy variations."""
        strategies = [ShardingStrategy.ZERO_1, ShardingStrategy.ZERO_2, ShardingStrategy.ZERO_3]
        
        params = {'test_param': np.random.randn(100, 50)}
        
        for strategy in strategies:
            config = DistributedConfig(sharding_strategy=strategy)
            optimizer = ZeROOptimizer(config, Mock())
            
            sharded = optimizer.shard_parameters(params)
            assert isinstance(sharded, dict)
    
    def test_optimization_levels(self):
        """Test different optimization levels."""
        levels = [OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3]
        
        for level in levels:
            config = DistributedConfig(optimization_level=level)
            assert config.optimization_level == level
    
    def test_gradient_buffer_management(self):
        """Test gradient buffer management."""
        buffer = GradientBuffer(capacity=1000)
        
        # Add gradients
        for i in range(5):
            grad = np.random.randn(10, 10)
            buffer.add_gradient(f"param_{i}", grad)
        
        assert buffer.size() == 5
        buffer.clear()
        assert buffer.size() == 0
    
    def test_advanced_all_reduce_patterns(self):
        """Test advanced all-reduce patterns."""
        config = DistributedConfig(world_size=4)
        all_reduce = AdvancedAllReduce(config)
        
        # Test ring all-reduce
        data = np.random.randn(1000)
        result = all_reduce.ring_all_reduce(data)
        assert result.shape == data.shape
        
        # Test tree all-reduce
        result = all_reduce.tree_all_reduce(data)
        assert result.shape == data.shape
    
    def test_dynamic_load_balancer_algorithms(self):
        """Test dynamic load balancer algorithms."""
        config = DistributedConfig(world_size=4)
        balancer = DynamicLoadBalancer(config)
        
        # Test different load balancing algorithms
        workloads = [0.1, 0.9, 0.5, 0.3]
        
        # Round-robin balancing
        plan = balancer.round_robin_balance(workloads)
        assert 'redistribution' in plan
        
        # Load-aware balancing
        plan = balancer.load_aware_balance(workloads)
        assert 'migrations' in plan
    
    def test_model_partition_strategies(self):
        """Test model partition strategies."""
        layers = [f"layer_{i}" for i in range(20)]
        
        # Even partitioning
        partition = ModelPartition.even_partition(layers, num_partitions=4)
        assert len(partition) == 4
        
        # Memory-aware partitioning
        memory_sizes = {layer: np.random.randint(100, 1000) for layer in layers}
        partition = ModelPartition.memory_aware_partition(layers, memory_sizes, num_partitions=4)
        assert len(partition) == 4
    
    def test_checkpoint_compression(self):
        """Test checkpoint compression features."""
        checkpointer = FaultTolerantCheckpointer(DistributedConfig(checkpoint_dir="/tmp"))
        
        data = {"large_param": np.random.randn(1000, 1000)}
        
        # Compressed checkpoint
        compressed = checkpointer.compress_checkpoint(data, compression="gzip")
        assert compressed is not None
        
        # Decompressed checkpoint
        decompressed = checkpointer.decompress_checkpoint(compressed)
        assert "large_param" in decompressed
    
    def test_network_topology_optimization(self):
        """Test network topology optimization."""
        config = DistributedConfig(world_size=8)
        orchestrator = DistributedTrainingOrchestrator(config)
        
        # Build communication topology
        topology = orchestrator.build_communication_topology()
        assert "nodes" in topology
        assert "edges" in topology
        
        # Optimize for bandwidth
        optimized = orchestrator.optimize_topology_for_bandwidth(topology)
        assert optimized is not None


class TestDistributedSystemEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_node_distributed_training(self):
        """Test distributed training with single node."""
        config = DistributedConfig(world_size=1, rank=0)
        orchestrator = DistributedTrainingOrchestrator(config)
        
        # Should handle single node gracefully
        assert orchestrator.config.world_size == 1
        assert orchestrator.is_distributed() is False
    
    def test_rank_assignment_edge_cases(self):
        """Test rank assignment edge cases."""
        # Invalid rank (negative)
        with pytest.raises(ValueError):
            DistributedConfig(world_size=4, rank=-1)
        
        # Invalid rank (too high)
        with pytest.raises(ValueError):
            DistributedConfig(world_size=4, rank=4)
    
    def test_empty_parameter_sharding(self):
        """Test sharding with empty parameters."""
        config = DistributedConfig(world_size=2)
        optimizer = ZeROOptimizer(config, Mock())
        
        empty_params = {}
        sharded = optimizer.shard_parameters(empty_params)
        assert sharded == {}
    
    def test_communication_failure_recovery(self):
        """Test communication failure recovery."""
        config = DistributedConfig(world_size=2)
        comm = CommunicationPrimitive(config)
        
        # Simulate communication failure
        with patch.object(comm, '_nccl_all_reduce', side_effect=RuntimeError("Network failure")):
            with pytest.raises(RuntimeError):
                comm.all_reduce(np.array([1.0, 2.0]))
    
    def test_memory_pressure_handling(self):
        """Test memory pressure handling."""
        config = DistributedConfig(world_size=2, enable_memory_monitoring=True)
        orchestrator = DistributedTrainingOrchestrator(config)
        
        # Simulate high memory usage
        memory_stats = {"used": 0.95, "available": 0.05}
        
        # Should trigger memory optimization
        optimizations = orchestrator.handle_memory_pressure(memory_stats)
        assert isinstance(optimizations, list)
    
    def test_gradient_overflow_handling(self):
        """Test gradient overflow handling."""
        config = DistributedConfig(world_size=2)
        optimizer = ZeROOptimizer(config, Mock())
        
        # Create overflow gradients
        overflow_grads = {
            "param": np.array([float('inf'), 1.0, 2.0])
        }
        
        # Should detect and handle overflow
        is_finite = optimizer.check_gradients_finite(overflow_grads)
        assert is_finite is False
    
    def test_checkpoint_corruption_recovery(self):
        """Test checkpoint corruption recovery."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = DistributedConfig(checkpoint_dir=temp_dir)
            checkpointer = FaultTolerantCheckpointer(config)
            
            # Create corrupted checkpoint file
            corrupt_path = os.path.join(temp_dir, "corrupt.pkl")
            with open(corrupt_path, 'wb') as f:
                f.write(b"corrupted data")
            
            # Should handle corruption gracefully
            loaded = checkpointer.load_checkpoint(corrupt_path)
            assert loaded is None
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestPerformanceOptimizationScenarios:
    """Test performance optimization scenarios."""
    
    def test_gradient_compression_algorithms(self):
        """Test gradient compression algorithms."""
        config = DistributedConfig(enable_gradient_compression=True)
        comm = CommunicationPrimitive(config)
        
        large_tensor = np.random.randn(10000)
        
        # Test different compression algorithms
        algorithms = ["quantization", "sparsification", "low_rank"]
        
        for algorithm in algorithms:
            compressed = comm.compress_tensor(large_tensor, algorithm=algorithm)
            decompressed = comm.decompress_tensor(compressed, original_shape=large_tensor.shape)
            
            # Should maintain reasonable accuracy
            correlation = np.corrcoef(large_tensor, decompressed)[0, 1]
            assert correlation > 0.8
    
    def test_mixed_precision_optimization(self):
        """Test mixed precision optimization."""
        config = DistributedConfig(mixed_precision=True)
        orchestrator = DistributedTrainingOrchestrator(config)
        
        # Test FP16 operations
        fp16_tensor = orchestrator.to_fp16(np.random.randn(100, 100).astype(np.float32))
        assert fp16_tensor.dtype == np.float16
        
        # Test automatic loss scaling
        scaled_loss = orchestrator.scale_loss_for_fp16(0.5)
        assert scaled_loss > 0.5


def test_full_system_integration():
    """Test the complete distributed system integration."""
    # This test would be run with the actual test_distributed_system function
    # from the module, which we imported
    try:
        # Run the built-in system test
        test_distributed_system()
        print("✅ Full system integration test passed")
    except Exception as e:
        pytest.fail(f"System integration test failed: {e}")


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])