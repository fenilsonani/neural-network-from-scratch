"""Distributed training launcher utilities.

This module provides utilities for launching distributed training jobs
across multiple processes and nodes.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class LaunchConfig:
    """Configuration for distributed training launch."""

    # Process configuration
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0

    # Network configuration
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    # Training configuration
    backend: str = "auto"
    use_env: bool = True

    # Resource configuration
    max_restarts: int = 3
    monitor_interval: float = 1.0
    start_timeout: int = 600

    # Logging
    log_dir: Optional[str] = None
    redirect_stdout: bool = True
    redirect_stderr: bool = True


class ProcessGroup:
    """Manages a group of distributed training processes."""

    def __init__(self, config: LaunchConfig):
        self.config = config
        self.processes: List[subprocess.Popen] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start_processes(self, training_script: str, script_args: List[str]) -> bool:
        """Start all training processes.

        Args:
            training_script: Path to training script
            script_args: Arguments to pass to training script

        Returns:
            True if all processes started successfully
        """
        logger.info(f"Starting {self.config.nproc_per_node} processes")

        # Calculate global ranks
        world_size = self.config.nnodes * self.config.nproc_per_node

        for local_rank in range(self.config.nproc_per_node):
            global_rank = self.config.node_rank * self.config.nproc_per_node + local_rank

            # Setup environment for this process
            env = self._setup_process_env(global_rank, world_size, local_rank)

            # Create command
            cmd = [sys.executable, training_script] + script_args

            # Setup logging
            stdout, stderr = self._setup_process_logging(global_rank)

            try:
                # Start process
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )

                self.processes.append(process)
                logger.info(f"Started process {global_rank} (PID: {process.pid})")

            except Exception as e:
                logger.error(f"Failed to start process {global_rank}: {e}")
                self._cleanup_processes()
                return False

        # Start monitoring
        self._start_monitoring()
        return True

    def wait(self) -> int:
        """Wait for all processes to complete.

        Returns:
            Exit code (0 if all processes succeeded)
        """
        logger.info("Waiting for processes to complete")

        exit_codes = []
        for i, process in enumerate(self.processes):
            try:
                exit_code = process.wait()
                exit_codes.append(exit_code)

                if exit_code == 0:
                    logger.info(f"Process {i} completed successfully")
                else:
                    logger.error(f"Process {i} failed with exit code {exit_code}")

            except KeyboardInterrupt:
                logger.info("Received interrupt, terminating processes")
                self._cleanup_processes()
                return 1

        # Stop monitoring
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        # Return overall exit code
        return max(exit_codes) if exit_codes else 0

    def _setup_process_env(self, rank: int, world_size: int, local_rank: int) -> Dict[str, str]:
        """Setup environment variables for a process."""
        env = os.environ.copy()

        if self.config.use_env:
            env.update(
                {
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": str(local_rank),
                    "MASTER_ADDR": self.config.master_addr,
                    "MASTER_PORT": str(self.config.master_port),
                    "NCCL_ASYNC_ERROR_HANDLING": "1",
                    "OMP_NUM_THREADS": "1",  # Prevent oversubscription
                }
            )

        return env

    def _setup_process_logging(self, rank: int) -> tuple:
        """Setup logging for a process."""
        if self.config.log_dir:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            if self.config.redirect_stdout:
                stdout = open(log_dir / f"rank_{rank}_stdout.log", "w")
            else:
                stdout = None

            if self.config.redirect_stderr:
                stderr = open(log_dir / f"rank_{rank}_stderr.log", "w")
            else:
                stderr = None
        else:
            stdout = None
            stderr = None

        return stdout, stderr

    def _start_monitoring(self):
        """Start process monitoring thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_processes)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_processes(self):
        """Monitor process health and handle failures."""
        restart_counts = [0] * len(self.processes)

        while self.monitoring:
            for i, process in enumerate(self.processes):
                if process.poll() is not None:  # Process has terminated
                    exit_code = process.returncode

                    if exit_code != 0 and restart_counts[i] < self.config.max_restarts:
                        logger.warning(f"Process {i} failed (exit code {exit_code}), restarting...")
                        restart_counts[i] += 1

                        # Restart process (simplified implementation)
                        # In practice, this would preserve the original command and environment
                        logger.info(
                            f"Restart {restart_counts[i]}/{self.config.max_restarts} for process {i}"
                        )

            time.sleep(self.config.monitor_interval)

    def _cleanup_processes(self):
        """Clean up all processes."""
        logger.info("Cleaning up processes")

        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Process is still running
                try:
                    # Try graceful termination first
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    logger.warning(f"Force killing process {i}")
                    process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process {i}: {e}")

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, cleaning up")
        self._cleanup_processes()
        sys.exit(1)


class DistributedLauncher:
    """Main launcher for distributed training."""

    def __init__(self, config: LaunchConfig):
        self.config = config
        self._validate_config()

    def launch(self, training_script: str, *script_args) -> int:
        """Launch distributed training.

        Args:
            training_script: Path to training script
            *script_args: Arguments to pass to training script

        Returns:
            Exit code
        """
        logger.info("Starting distributed training")
        logger.info(f"Configuration: {self.config}")

        # Create process group
        process_group = ProcessGroup(self.config)

        # Start processes
        if not process_group.start_processes(training_script, list(script_args)):
            return 1

        # Wait for completion
        return process_group.wait()

    def _validate_config(self):
        """Validate launch configuration."""
        if self.config.nproc_per_node <= 0:
            raise ValueError("nproc_per_node must be positive")

        if self.config.nnodes <= 0:
            raise ValueError("nnodes must be positive")

        if not (0 <= self.config.node_rank < self.config.nnodes):
            raise ValueError("node_rank must be in range [0, nnodes)")

        # Validate master address
        try:
            socket.inet_aton(self.config.master_addr)
        except socket.error:
            raise ValueError(f"Invalid master_addr: {self.config.master_addr}")

        # Validate port
        if not (1024 <= self.config.master_port <= 65535):
            raise ValueError("master_port must be in range [1024, 65535]")


class MultiNodeLauncher:
    """Launcher for multi-node distributed training."""

    def __init__(self, config: LaunchConfig, hostfile: Optional[str] = None):
        self.config = config
        self.hostfile = hostfile
        self.hosts = self._parse_hostfile() if hostfile else [config.master_addr]

    def launch(self, training_script: str, *script_args) -> int:
        """Launch multi-node training.

        Args:
            training_script: Path to training script
            *script_args: Arguments to pass to training script

        Returns:
            Exit code
        """
        logger.info(f"Starting multi-node training on {len(self.hosts)} nodes")

        # Launch on each node
        processes = []
        for node_rank, host in enumerate(self.hosts):
            if host == self.config.master_addr and node_rank == 0:
                # Launch locally on master node
                config = LaunchConfig(
                    nproc_per_node=self.config.nproc_per_node,
                    nnodes=self.config.nnodes,
                    node_rank=node_rank,
                    master_addr=self.config.master_addr,
                    master_port=self.config.master_port,
                    backend=self.config.backend,
                )

                launcher = DistributedLauncher(config)
                process_group = ProcessGroup(config)
                process_group.start_processes(training_script, list(script_args))
                processes.append(process_group)

            else:
                # Launch remotely via SSH (simplified)
                logger.info(f"Would launch on remote host {host} (rank {node_rank})")
                # In practice, this would use SSH to launch on remote nodes

        # Wait for all to complete
        exit_codes = []
        for process_group in processes:
            exit_code = process_group.wait()
            exit_codes.append(exit_code)

        return max(exit_codes) if exit_codes else 0

    def _parse_hostfile(self) -> List[str]:
        """Parse hostfile to get list of hosts."""
        if not self.hostfile:
            return []

        hosts = []
        try:
            with open(self.hostfile, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Parse format: hostname[:port] [slots=N]
                        parts = line.split()
                        host = parts[0]
                        hosts.append(host)
        except Exception as e:
            logger.error(f"Failed to parse hostfile {self.hostfile}: {e}")
            raise

        return hosts


def launch_distributed_training(
    training_script: str,
    nproc_per_node: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    backend: str = "auto",
    log_dir: Optional[str] = None,
    *script_args,
) -> int:
    """Convenience function to launch distributed training.

    Args:
        training_script: Path to training script
        nproc_per_node: Number of processes per node
        nnodes: Number of nodes
        node_rank: Rank of current node
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend
        log_dir: Directory for logs
        *script_args: Arguments to pass to training script

    Returns:
        Exit code
    """
    config = LaunchConfig(
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
        log_dir=log_dir,
    )

    launcher = DistributedLauncher(config)
    return launcher.launch(training_script, *script_args)


def find_free_port() -> int:
    """Find a free port for master node communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_local_ip() -> str:
    """Get local IP address."""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    # Command-line interface for distributed launcher
    import argparse

    parser = argparse.ArgumentParser(description="Distributed Training Launcher")
    parser.add_argument("training_script", help="Training script to run")
    parser.add_argument(
        "--nproc_per_node", type=int, default=1, help="Number of processes per node"
    )
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of current node")
    parser.add_argument("--master_addr", default="127.0.0.1", help="Master node address")
    parser.add_argument("--master_port", type=int, default=29500, help="Master node port")
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "nccl", "gloo"], help="Communication backend"
    )
    parser.add_argument("--log_dir", help="Directory for logs")

    args, script_args = parser.parse_known_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Launch distributed training
    exit_code = launch_distributed_training(
        args.training_script,
        nproc_per_node=args.nproc_per_node,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        backend=args.backend,
        log_dir=args.log_dir,
        *script_args,
    )

    sys.exit(exit_code)
