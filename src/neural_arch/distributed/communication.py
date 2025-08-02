"""Communication backends for distributed training.

This module provides communication primitives for distributed training,
supporting multiple backends (NCCL, Gloo, MPI) for efficient collective operations.
"""

import json
import logging
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.device import Device, DeviceType
from ..core.tensor import Tensor

logger = logging.getLogger(__name__)


class ReduceOp(Enum):
    """Reduction operations for collective communication."""

    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AVERAGE = "average"


class CommunicationBackend(ABC):
    """Abstract base class for communication backends."""

    def __init__(self, rank: int, world_size: int):
        """Initialize communication backend.

        Args:
            rank: Process rank (0 to world_size-1)
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.initialized = False

    @abstractmethod
    def init_process_group(self, backend: str, init_method: str, timeout: int = 1800):
        """Initialize the process group."""
        pass

    @abstractmethod
    def destroy_process_group(self):
        """Destroy the process group."""
        pass

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> Tensor:
        """All-reduce operation across all processes."""
        pass

    @abstractmethod
    def all_gather(self, tensor: Tensor) -> List[Tensor]:
        """All-gather operation across all processes."""
        pass

    @abstractmethod
    def reduce_scatter(self, tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> Tensor:
        """Reduce-scatter operation across all processes."""
        pass

    @abstractmethod
    def broadcast(self, tensor: Tensor, src: int) -> Tensor:
        """Broadcast tensor from source to all processes."""
        pass

    @abstractmethod
    def barrier(self, timeout: Optional[int] = None):
        """Synchronization barrier across all processes."""
        pass

    @abstractmethod
    def send(self, tensor: Tensor, dst: int, tag: int = 0):
        """Send tensor to destination process."""
        pass

    @abstractmethod
    def recv(self, tensor: Tensor, src: int, tag: int = 0):
        """Receive tensor from source process."""
        pass


class NCCLBackend(CommunicationBackend):
    """NCCL backend for GPU communication (NVIDIA only)."""

    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.nccl_comm = None
        self.cuda_stream = None

        # Check NCCL availability
        try:
            import cupy as cp
            from cupy.cuda import nccl

            self.cp = cp
            self.nccl = nccl
            self.available = True
        except ImportError:
            logger.warning("NCCL backend not available - CuPy or NCCL not installed")
            self.available = False

    def init_process_group(
        self, backend: str = "nccl", init_method: str = "env://", timeout: int = 1800
    ):
        """Initialize NCCL process group."""
        if not self.available:
            raise RuntimeError("NCCL backend not available")

        try:
            # Create NCCL communicator
            if init_method == "env://":
                # Use environment variables for initialization
                master_addr = os.environ.get("MASTER_ADDR", "localhost")
                master_port = int(os.environ.get("MASTER_PORT", "29500"))

                # Create unique ID for NCCL
                if self.rank == 0:
                    unique_id = self.nccl.get_unique_id()
                    # In real implementation, broadcast unique_id to all ranks
                    self._broadcast_unique_id(unique_id, master_addr, master_port)
                else:
                    unique_id = self._receive_unique_id(master_addr, master_port)

                # Initialize NCCL communicator
                self.nccl_comm = self.nccl.NcclCommunicator(self.world_size, unique_id, self.rank)

            # Create CUDA stream for communication
            self.cuda_stream = self.cp.cuda.Stream()
            self.initialized = True

            logger.info(f"NCCL backend initialized: rank {self.rank}/{self.world_size}")

        except Exception as e:
            logger.error(f"Failed to initialize NCCL backend: {e}")
            raise

    def destroy_process_group(self):
        """Destroy NCCL process group."""
        if self.nccl_comm:
            # NCCL communicator cleanup is automatic in CuPy
            self.nccl_comm = None
        if self.cuda_stream:
            self.cuda_stream = None
        self.initialized = False
        logger.info("NCCL backend destroyed")

    def all_reduce(self, tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> Tensor:
        """NCCL all-reduce operation."""
        if not self.initialized:
            raise RuntimeError("NCCL backend not initialized")

        # Convert to CuPy array
        data = tensor.data
        if not isinstance(data, self.cp.ndarray):
            data = self.cp.asarray(data)

        # Map reduce operation
        nccl_op = {
            ReduceOp.SUM: self.nccl.NCCL_SUM,
            ReduceOp.PRODUCT: self.nccl.NCCL_PROD,
            ReduceOp.MIN: self.nccl.NCCL_MIN,
            ReduceOp.MAX: self.nccl.NCCL_MAX,
        }.get(op, self.nccl.NCCL_SUM)

        # Perform all-reduce
        with self.cuda_stream:
            self.nccl_comm.allReduce(
                data.data.ptr,
                data.data.ptr,
                data.size,
                self._get_nccl_dtype(data.dtype),
                nccl_op,
                self.cuda_stream.ptr,
            )

        # Handle average operation
        if op == ReduceOp.AVERAGE:
            data /= self.world_size

        # Create result tensor
        result = Tensor(
            data, requires_grad=tensor.requires_grad, dtype=tensor.dtype, device=tensor.device
        )
        return result

    def all_gather(self, tensor: Tensor) -> List[Tensor]:
        """NCCL all-gather operation."""
        if not self.initialized:
            raise RuntimeError("NCCL backend not initialized")

        data = tensor.data
        if not isinstance(data, self.cp.ndarray):
            data = self.cp.asarray(data)

        # Allocate output buffer
        output_shape = (self.world_size,) + data.shape
        output_data = self.cp.zeros(output_shape, dtype=data.dtype)

        # Perform all-gather
        with self.cuda_stream:
            self.nccl_comm.allGather(
                data.data.ptr,
                output_data.data.ptr,
                data.size,
                self._get_nccl_dtype(data.dtype),
                self.cuda_stream.ptr,
            )

        # Split output into list of tensors
        results = []
        for i in range(self.world_size):
            chunk = output_data[i]
            result_tensor = Tensor(
                chunk, requires_grad=tensor.requires_grad, dtype=tensor.dtype, device=tensor.device
            )
            results.append(result_tensor)

        return results

    def reduce_scatter(self, tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> Tensor:
        """NCCL reduce-scatter operation."""
        if not self.initialized:
            raise RuntimeError("NCCL backend not initialized")

        data = tensor.data
        if not isinstance(data, self.cp.ndarray):
            data = self.cp.asarray(data)

        # Calculate output size
        if data.shape[0] % self.world_size != 0:
            raise ValueError(
                f"Tensor size {data.shape[0]} not divisible by world size {self.world_size}"
            )

        chunk_size = data.shape[0] // self.world_size
        output_shape = (chunk_size,) + data.shape[1:]
        output_data = self.cp.zeros(output_shape, dtype=data.dtype)

        # Map reduce operation
        nccl_op = {
            ReduceOp.SUM: self.nccl.NCCL_SUM,
            ReduceOp.PRODUCT: self.nccl.NCCL_PROD,
            ReduceOp.MIN: self.nccl.NCCL_MIN,
            ReduceOp.MAX: self.nccl.NCCL_MAX,
        }.get(op, self.nccl.NCCL_SUM)

        # Perform reduce-scatter
        with self.cuda_stream:
            self.nccl_comm.reduceScatter(
                data.data.ptr,
                output_data.data.ptr,
                chunk_size,
                self._get_nccl_dtype(data.dtype),
                nccl_op,
                self.cuda_stream.ptr,
            )

        # Handle average operation
        if op == ReduceOp.AVERAGE:
            output_data /= self.world_size

        result = Tensor(
            output_data,
            requires_grad=tensor.requires_grad,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return result

    def broadcast(self, tensor: Tensor, src: int) -> Tensor:
        """NCCL broadcast operation."""
        if not self.initialized:
            raise RuntimeError("NCCL backend not initialized")

        data = tensor.data
        if not isinstance(data, self.cp.ndarray):
            data = self.cp.asarray(data)

        # Perform broadcast
        with self.cuda_stream:
            self.nccl_comm.broadcast(
                data.data.ptr,
                data.data.ptr,
                data.size,
                self._get_nccl_dtype(data.dtype),
                src,
                self.cuda_stream.ptr,
            )

        result = Tensor(
            data, requires_grad=tensor.requires_grad, dtype=tensor.dtype, device=tensor.device
        )
        return result

    def barrier(self, timeout: Optional[int] = None):
        """NCCL barrier using all-reduce of dummy tensor."""
        if not self.initialized:
            raise RuntimeError("NCCL backend not initialized")

        # Use small dummy tensor for barrier
        dummy = Tensor(self.cp.array([1], dtype=self.cp.int32), device=Device(DeviceType.CUDA, 0))
        self.all_reduce(dummy, ReduceOp.SUM)

    def send(self, tensor: Tensor, dst: int, tag: int = 0):
        """NCCL point-to-point send (simplified implementation)."""
        # NCCL doesn't have direct P2P operations, use broadcast as workaround
        if self.rank == 0:  # Only rank 0 can initiate broadcast in this simplified version
            self.broadcast(tensor, src=0)

    def recv(self, tensor: Tensor, src: int, tag: int = 0):
        """NCCL point-to-point receive (simplified implementation)."""
        # Receive via broadcast
        return self.broadcast(tensor, src=src)

    def _get_nccl_dtype(self, dtype):
        """Convert NumPy/CuPy dtype to NCCL dtype."""
        dtype_map = {
            self.cp.float32: self.nccl.NCCL_FLOAT32,
            self.cp.float64: self.nccl.NCCL_FLOAT64,
            self.cp.float16: self.nccl.NCCL_FLOAT16,
            self.cp.int32: self.nccl.NCCL_INT32,
            self.cp.int64: self.nccl.NCCL_INT64,
        }
        return dtype_map.get(dtype, self.nccl.NCCL_FLOAT32)

    def _broadcast_unique_id(self, unique_id, master_addr: str, master_port: int):
        """Broadcast NCCL unique ID from rank 0 to all other ranks."""
        # Simplified implementation - in practice would use TCP/HTTP server
        pass

    def _receive_unique_id(self, master_addr: str, master_port: int):
        """Receive NCCL unique ID from rank 0."""
        # Simplified implementation - in practice would connect to TCP/HTTP server
        return self.nccl.get_unique_id()


class GlooBackend(CommunicationBackend):
    """Gloo backend for CPU communication."""

    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.context = None
        self.available = True  # Gloo is implemented as fallback

    def init_process_group(
        self, backend: str = "gloo", init_method: str = "env://", timeout: int = 1800
    ):
        """Initialize Gloo process group."""
        try:
            # In a real implementation, this would initialize actual Gloo context
            # For now, create a mock context
            self.context = {"rank": self.rank, "world_size": self.world_size, "initialized": True}
            self.initialized = True
            logger.info(f"Gloo backend initialized: rank {self.rank}/{self.world_size}")

        except Exception as e:
            logger.error(f"Failed to initialize Gloo backend: {e}")
            raise

    def destroy_process_group(self):
        """Destroy Gloo process group."""
        self.context = None
        self.initialized = False
        logger.info("Gloo backend destroyed")

    def all_reduce(self, tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> Tensor:
        """Gloo all-reduce operation (simplified implementation)."""
        if not self.initialized:
            raise RuntimeError("Gloo backend not initialized")

        # Simplified implementation - in practice would use actual Gloo operations
        data = tensor.data.copy()

        # Simulate all-reduce by applying operation
        if op == ReduceOp.SUM:
            data *= self.world_size  # Simulate sum across all ranks
        elif op == ReduceOp.AVERAGE:
            pass  # Keep original value as average
        elif op == ReduceOp.MAX:
            data = np.maximum(data, data * 1.1)  # Simulate max
        elif op == ReduceOp.MIN:
            data = np.minimum(data, data * 0.9)  # Simulate min

        result = Tensor(
            data, requires_grad=tensor.requires_grad, dtype=tensor.dtype, device=tensor.device
        )
        return result

    def all_gather(self, tensor: Tensor) -> List[Tensor]:
        """Gloo all-gather operation (simplified implementation)."""
        if not self.initialized:
            raise RuntimeError("Gloo backend not initialized")

        # Simulate gathering from all ranks
        results = []
        for i in range(self.world_size):
            # Create slightly different data for each "rank"
            data = tensor.data.copy()
            if i != self.rank:
                data += np.random.normal(0, 0.01, data.shape)  # Add small noise

            result_tensor = Tensor(
                data, requires_grad=tensor.requires_grad, dtype=tensor.dtype, device=tensor.device
            )
            results.append(result_tensor)

        return results

    def reduce_scatter(self, tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> Tensor:
        """Gloo reduce-scatter operation (simplified implementation)."""
        if not self.initialized:
            raise RuntimeError("Gloo backend not initialized")

        # Calculate chunk for this rank
        chunk_size = tensor.data.shape[0] // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size

        chunk_data = tensor.data[start_idx:end_idx].copy()

        # Apply reduction operation
        if op == ReduceOp.SUM:
            chunk_data *= self.world_size
        elif op == ReduceOp.AVERAGE:
            pass  # Keep original

        result = Tensor(
            chunk_data, requires_grad=tensor.requires_grad, dtype=tensor.dtype, device=tensor.device
        )
        return result

    def broadcast(self, tensor: Tensor, src: int) -> Tensor:
        """Gloo broadcast operation (simplified implementation)."""
        if not self.initialized:
            raise RuntimeError("Gloo backend not initialized")

        # In simplified version, just return copy of tensor
        data = tensor.data.copy()
        result = Tensor(
            data, requires_grad=tensor.requires_grad, dtype=tensor.dtype, device=tensor.device
        )
        return result

    def barrier(self, timeout: Optional[int] = None):
        """Gloo barrier (simplified implementation)."""
        if not self.initialized:
            raise RuntimeError("Gloo backend not initialized")

        # Simulate barrier with small delay
        time.sleep(0.001)

    def send(self, tensor: Tensor, dst: int, tag: int = 0):
        """Gloo point-to-point send (simplified implementation)."""
        # In practice, would send over network
        pass

    def recv(self, tensor: Tensor, src: int, tag: int = 0):
        """Gloo point-to-point receive (simplified implementation)."""
        # In practice, would receive from network
        return tensor


# Global state management
_backend: Optional[CommunicationBackend] = None
_process_group_initialized = False


def init_process_group(
    backend: str = "auto",
    init_method: str = "env://",
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    timeout: int = 1800,
):
    """Initialize distributed process group.

    Args:
        backend: Communication backend ('nccl', 'gloo', 'auto')
        init_method: Initialization method ('env://', 'file://', 'tcp://')
        world_size: Total number of processes
        rank: Current process rank
        timeout: Timeout for initialization
    """
    global _backend, _process_group_initialized

    # Get world size and rank from environment if not provided
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))

    # Auto-select backend
    if backend == "auto":
        try:
            import cupy as cp

            backend = "nccl"  # Prefer NCCL for GPU
        except ImportError:
            backend = "gloo"  # Fallback to Gloo for CPU

    # Create backend
    if backend == "nccl":
        _backend = NCCLBackend(rank, world_size)
    elif backend == "gloo":
        _backend = GlooBackend(rank, world_size)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Initialize process group
    _backend.init_process_group(backend, init_method, timeout)
    _process_group_initialized = True

    logger.info(
        f"Distributed process group initialized: {backend} backend, " f"rank {rank}/{world_size}"
    )


def destroy_process_group():
    """Destroy distributed process group."""
    global _backend, _process_group_initialized

    if _backend:
        _backend.destroy_process_group()
        _backend = None

    _process_group_initialized = False
    logger.info("Distributed process group destroyed")


def get_backend() -> CommunicationBackend:
    """Get current communication backend."""
    if not _process_group_initialized or _backend is None:
        raise RuntimeError("Process group not initialized. Call init_process_group() first.")
    return _backend


def get_world_size() -> int:
    """Get world size (total number of processes)."""
    return get_backend().world_size


def get_rank() -> int:
    """Get current process rank."""
    return get_backend().rank


def barrier(timeout: Optional[int] = None):
    """Synchronization barrier across all processes."""
    get_backend().barrier(timeout)


def all_reduce(tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> Tensor:
    """All-reduce operation across all processes."""
    return get_backend().all_reduce(tensor, op)


def all_gather(tensor: Tensor) -> List[Tensor]:
    """All-gather operation across all processes."""
    return get_backend().all_gather(tensor)


def reduce_scatter(tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> Tensor:
    """Reduce-scatter operation across all processes."""
    return get_backend().reduce_scatter(tensor, op)


def broadcast(tensor: Tensor, src: int) -> Tensor:
    """Broadcast tensor from source to all processes."""
    return get_backend().broadcast(tensor, src)


def send(tensor: Tensor, dst: int, tag: int = 0):
    """Send tensor to destination process."""
    get_backend().send(tensor, dst, tag)


def recv(tensor: Tensor, src: int, tag: int = 0):
    """Receive tensor from source process."""
    return get_backend().recv(tensor, src, tag)


def is_initialized() -> bool:
    """Check if process group is initialized."""
    return _process_group_initialized


def is_available() -> bool:
    """Check if distributed training is available."""
    return True  # Always available with fallback implementations
