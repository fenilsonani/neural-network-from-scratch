"""Distributed training components for neural architecture framework.

This module provides enterprise-grade distributed training capabilities including:
- Data parallel training across multiple GPUs
- Model parallel training for large models
- Gradient synchronization and communication
- Fault tolerance and recovery mechanisms
- Dynamic scaling and load balancing

Key Features:
- Multi-GPU data parallelism with efficient gradient all-reduce
- Pipeline parallelism for large transformer models
- Tensor parallelism for matrix operations
- Hierarchical communication strategies
- Automatic mixed precision in distributed settings
- Integration with existing optimization systems
"""

from .checkpointing import (
    DistributedCheckpoint,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)
from .communication import (
    CommunicationBackend,
    GlooBackend,
    NCCLBackend,
    ReduceOp,
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    reduce_scatter,
)
from .data_parallel import (
    DataParallel,
    DistributedDataParallel,
    DistributedSampler,
    gather,
    get_distributed_info,
    scatter,
)
from .integration import (
    DistributedModelMixin,
    auto_distributed_setup,
    auto_scale_batch_size,
    auto_scale_learning_rate,
    distributed_training_context,
    get_distributed_training_info,
    make_distributed,
)
from .launcher import DistributedLauncher, MultiNodeLauncher, launch_distributed_training
from .model_parallel import (
    ModelParallel,
    ParallelEmbedding,
    ParallelLinear,
    PipelineParallel,
    TensorParallel,
)

__all__ = [
    # Data parallel
    "DataParallel",
    "DistributedDataParallel",
    "DistributedSampler",
    "gather",
    "scatter",
    "get_distributed_info",
    # Model parallel
    "ModelParallel",
    "PipelineParallel",
    "TensorParallel",
    "ParallelLinear",
    "ParallelEmbedding",
    # Communication
    "CommunicationBackend",
    "NCCLBackend",
    "GlooBackend",
    "init_process_group",
    "destroy_process_group",
    "get_world_size",
    "get_rank",
    "barrier",
    "broadcast",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "ReduceOp",
    "is_initialized",
    # Launcher
    "DistributedLauncher",
    "launch_distributed_training",
    "MultiNodeLauncher",
    # Checkpointing
    "DistributedCheckpoint",
    "save_distributed_checkpoint",
    "load_distributed_checkpoint",
    # Integration helpers
    "auto_distributed_setup",
    "make_distributed",
    "distributed_training_context",
    "DistributedModelMixin",
    "auto_scale_batch_size",
    "auto_scale_learning_rate",
    "get_distributed_training_info",
]
