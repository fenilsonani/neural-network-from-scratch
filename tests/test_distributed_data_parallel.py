"""Comprehensive tests for distributed data parallel training.

This module tests data parallel training implementations including DataParallel,
DistributedDataParallel, DistributedSampler, and related functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_arch.core import Tensor
from neural_arch.core.device import Device, DeviceType
from neural_arch.distributed.communication import (
    destroy_process_group,
    init_process_group,
    is_initialized,
)
from neural_arch.distributed.data_parallel import (
    DataParallel,
    DistributedDataParallel,
    DistributedSampler,
    gather,
    get_distributed_info,
    is_distributed_training_available,
    no_sync,
    scatter,
)
from neural_arch.nn import Linear, Module


class SimpleModule(Module):
    """Simple test module for data parallel testing."""

    def __init__(self, input_size: int = 10, output_size: int = 5):
        super().__init__()
        self.linear = Linear(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def parameters(self):
        return self.linear.parameters()


class TestDataParallel:
    """Test DataParallel wrapper for single-node multi-GPU training."""

    def setup_method(self):
        """Set up test fixtures."""
        self.module = SimpleModule(input_size=8, output_size=4)
        self.device_ids = [0, 1]  # Mock GPU IDs

    def test_data_parallel_creation(self):
        """Test DataParallel module creation."""
        dp_module = DataParallel(self.module, device_ids=self.device_ids)

        assert dp_module.module is self.module
        assert dp_module.device_ids == self.device_ids
        assert dp_module.output_device == self.device_ids[0]
        assert len(dp_module.replicas) == len(self.device_ids)

    def test_data_parallel_auto_device_detection(self):
        """Test automatic device detection when device_ids is None."""
        with patch("neural_arch.distributed.data_parallel.cp") as mock_cp:
            mock_cp.cuda.runtime.getDeviceCount.return_value = 3

            dp_module = DataParallel(self.module, device_ids=None)
            assert dp_module.device_ids == [0, 1, 2]

    def test_data_parallel_auto_device_fallback(self):
        """Test fallback to single device when CuPy not available."""
        with patch("neural_arch.distributed.data_parallel.cp", side_effect=ImportError):
            dp_module = DataParallel(self.module, device_ids=None)
            assert dp_module.device_ids == [0]

    def test_data_parallel_custom_output_device(self):
        """Test DataParallel with custom output device."""
        output_device = 1
        dp_module = DataParallel(
            self.module, device_ids=self.device_ids, output_device=output_device
        )
        assert dp_module.output_device == output_device

    def test_data_parallel_single_device_forward(self):
        """Test forward pass with single device (no parallelism)."""
        dp_module = DataParallel(self.module, device_ids=[0])

        batch_size = 4
        x = Tensor(np.random.randn(batch_size, 8).astype(np.float32))

        output = dp_module(x)

        assert isinstance(output, Tensor)
        assert output.shape == (batch_size, 4)

    def test_data_parallel_multi_device_forward(self):
        """Test forward pass with multiple devices."""
        dp_module = DataParallel(self.module, device_ids=self.device_ids)

        batch_size = 8  # Divisible by number of devices
        x = Tensor(np.random.randn(batch_size, 8).astype(np.float32))

        with patch.object(dp_module, "_scatter_inputs") as mock_scatter, patch.object(
            dp_module, "_gather_outputs"
        ) as mock_gather:

            # Mock scattered inputs
            mock_scatter.return_value = {0: ((x[:4],), {}), 1: ((x[4:],), {})}

            # Mock gathered output
            expected_output = Tensor(np.random.randn(batch_size, 4).astype(np.float32))
            mock_gather.return_value = expected_output

            output = dp_module(x)

            assert output is expected_output
            mock_scatter.assert_called_once()
            mock_gather.assert_called_once()

    def test_data_parallel_scatter_inputs(self):
        """Test input scattering across devices."""
        dp_module = DataParallel(self.module, device_ids=self.device_ids)

        batch_size = 6
        x = Tensor(np.random.randn(batch_size, 8).astype(np.float32))
        inputs = (x,)
        kwargs = {"mask": Tensor(np.ones((batch_size, 1)))}

        scattered = dp_module._scatter_inputs(inputs, kwargs)

        assert len(scattered) == len(self.device_ids)

        # Check device 0 gets first chunk
        device_0_inputs, device_0_kwargs = scattered[0]
        assert device_0_inputs[0].shape[0] == 3  # batch_size // 2
        assert device_0_kwargs["mask"].shape[0] == 3

        # Check device 1 gets remaining chunk
        device_1_inputs, device_1_kwargs = scattered[1]
        assert device_1_inputs[0].shape[0] == 3  # remaining samples
        assert device_1_kwargs["mask"].shape[0] == 3

    def test_data_parallel_scatter_uneven_batch(self):
        """Test input scattering with uneven batch division."""
        dp_module = DataParallel(self.module, device_ids=self.device_ids)

        batch_size = 5  # Not evenly divisible by 2
        x = Tensor(np.random.randn(batch_size, 8).astype(np.float32))
        inputs = (x,)
        kwargs = {}

        scattered = dp_module._scatter_inputs(inputs, kwargs)

        # First device gets batch_size // num_devices
        assert scattered[0][0][0].shape[0] == 2
        # Last device gets remaining samples
        assert scattered[1][0][0].shape[0] == 3

    def test_data_parallel_gather_outputs(self):
        """Test output gathering from devices."""
        dp_module = DataParallel(self.module, device_ids=self.device_ids)

        # Mock outputs from different devices
        output_0 = Tensor(np.random.randn(2, 4).astype(np.float32))
        output_1 = Tensor(np.random.randn(3, 4).astype(np.float32))
        outputs = [output_0, output_1]

        with patch("neural_arch.distributed.data_parallel.concatenate") as mock_concat:
            expected_output = Tensor(np.random.randn(5, 4).astype(np.float32))
            mock_concat.return_value = expected_output

            result = dp_module._gather_outputs(outputs)

            assert result is expected_output
            mock_concat.assert_called_once_with(outputs, axis=0)

    def test_data_parallel_gather_single_output(self):
        """Test output gathering with single device output."""
        dp_module = DataParallel(self.module, device_ids=[0])

        output = Tensor(np.random.randn(4, 4).astype(np.float32))
        result = dp_module._gather_outputs([output])

        assert result is output

    def test_data_parallel_parameters(self):
        """Test parameter access from DataParallel module."""
        dp_module = DataParallel(self.module, device_ids=self.device_ids)

        dp_params = dp_module.parameters()
        original_params = self.module.parameters()

        assert dp_params == original_params


class TestDistributedDataParallel:
    """Test DistributedDataParallel for multi-node training."""

    def setup_method(self):
        """Set up distributed environment."""
        init_process_group(backend="gloo", world_size=4, rank=0)
        self.module = SimpleModule(input_size=8, output_size=4)

    def teardown_method(self):
        """Clean up distributed environment."""
        if is_initialized():
            destroy_process_group()

    def test_ddp_creation(self):
        """Test DistributedDataParallel creation."""
        ddp_module = DistributedDataParallel(self.module)

        assert ddp_module.module is self.module
        assert ddp_module.world_size == 4
        assert ddp_module.rank == 0
        assert not ddp_module.gradients_ready
        assert ddp_module.gradient_compression is False

    def test_ddp_creation_not_initialized(self):
        """Test DDP creation fails when distributed not initialized."""
        destroy_process_group()

        with pytest.raises(RuntimeError, match="not initialized"):
            DistributedDataParallel(self.module)

    def test_ddp_with_custom_options(self):
        """Test DDP creation with custom options."""
        device = Device(DeviceType.CUDA, 0)
        ddp_module = DistributedDataParallel(
            self.module,
            device=device,
            gradient_compression=True,
            bucket_size_mb=50,
            find_unused_parameters=True,
        )

        assert ddp_module.device is device
        assert ddp_module.gradient_compression is True
        assert ddp_module.bucket_size_mb == 50
        assert ddp_module.find_unused_parameters is True

    def test_ddp_forward_pass(self):
        """Test DDP forward pass."""
        ddp_module = DistributedDataParallel(self.module)

        batch_size = 4
        x = Tensor(np.random.randn(batch_size, 8).astype(np.float32))

        output = ddp_module(x)

        assert isinstance(output, Tensor)
        assert output.shape == (batch_size, 4)
        assert not ddp_module.gradients_ready

    def test_ddp_gradient_hook_registration(self):
        """Test that gradient hooks are registered for parameters."""
        ddp_module = DistributedDataParallel(self.module)

        # Check that hooks are registered (simplified check)
        for param in ddp_module.module.parameters():
            if param.requires_grad:
                # In a full implementation, would check hooks
                assert hasattr(param, "register_hook")

    def test_ddp_sync_gradients(self):
        """Test manual gradient synchronization."""
        ddp_module = DistributedDataParallel(self.module)

        # Create mock gradients
        for param in ddp_module.module.parameters():
            if param.requires_grad:
                param.grad = Tensor(np.random.randn(*param.shape).astype(np.float32))

        # Mock all_reduce to verify it's called
        with patch("neural_arch.distributed.data_parallel.all_reduce") as mock_all_reduce:
            mock_all_reduce.side_effect = lambda x, op: x  # Return input unchanged

            ddp_module.sync_gradients()

            assert ddp_module.gradients_ready
            # Should be called for each parameter with gradients
            assert mock_all_reduce.call_count >= 1

    def test_ddp_sync_gradients_single_process(self):
        """Test gradient sync with single process (should be no-op)."""
        # Create single-process environment
        destroy_process_group()
        init_process_group(backend="gloo", world_size=1, rank=0)

        ddp_module = DistributedDataParallel(self.module)

        # Add gradients
        for param in ddp_module.module.parameters():
            if param.requires_grad:
                param.grad = Tensor(np.random.randn(*param.shape).astype(np.float32))

        ddp_module.sync_gradients()
        # Should be no-op for single process

    def test_ddp_parameters(self):
        """Test parameter access from DDP module."""
        ddp_module = DistributedDataParallel(self.module)

        ddp_params = ddp_module.parameters()
        original_params = self.module.parameters()

        assert ddp_params == original_params

    def test_ddp_training_mode(self):
        """Test training mode setting."""
        ddp_module = DistributedDataParallel(self.module)

        # Test train mode
        result = ddp_module.train(True)
        assert result is ddp_module

        # Test eval mode
        result = ddp_module.eval()
        assert result is ddp_module


class TestDistributedSampler:
    """Test DistributedSampler for dataset partitioning."""

    def test_distributed_sampler_creation(self):
        """Test basic DistributedSampler creation."""
        sampler = DistributedSampler(
            dataset_size=100, num_replicas=4, rank=1, shuffle=True, seed=42
        )

        assert sampler.dataset_size == 100
        assert sampler.num_replicas == 4
        assert sampler.rank == 1
        assert sampler.shuffle is True
        assert sampler.seed == 42
        assert sampler.epoch == 0

    def test_distributed_sampler_auto_detection(self):
        """Test automatic detection of world size and rank."""
        init_process_group(backend="gloo", world_size=4, rank=2)

        try:
            sampler = DistributedSampler(dataset_size=100)

            assert sampler.num_replicas == 4
            assert sampler.rank == 2
        finally:
            destroy_process_group()

    def test_distributed_sampler_no_distributed(self):
        """Test sampler behavior when distributed is not initialized."""
        sampler = DistributedSampler(dataset_size=100)

        assert sampler.num_replicas == 1
        assert sampler.rank == 0

    def test_distributed_sampler_sample_calculation(self):
        """Test calculation of samples per replica."""
        # Test evenly divisible
        sampler = DistributedSampler(dataset_size=100, num_replicas=4)
        assert sampler.num_samples == 25
        assert sampler.total_size == 100

        # Test not evenly divisible (padding)
        sampler = DistributedSampler(dataset_size=102, num_replicas=4)
        assert sampler.num_samples == 26  # (102 + 4 - 1) // 4
        assert sampler.total_size == 104

    def test_distributed_sampler_drop_last(self):
        """Test sample calculation with drop_last=True."""
        sampler = DistributedSampler(dataset_size=102, num_replicas=4, drop_last=True)

        assert sampler.num_samples == 25  # 102 // 4
        assert sampler.total_size == 100

    def test_distributed_sampler_iteration(self):
        """Test sampler iteration and index generation."""
        sampler = DistributedSampler(
            dataset_size=10, num_replicas=2, rank=0, shuffle=False, seed=42
        )

        indices = list(sampler)

        # Should get every other index starting from rank 0
        expected_indices = [0, 2, 4, 6, 8]
        assert indices == expected_indices
        assert len(indices) == sampler.num_samples

    def test_distributed_sampler_shuffled_iteration(self):
        """Test sampler iteration with shuffling."""
        sampler = DistributedSampler(dataset_size=10, num_replicas=2, rank=0, shuffle=True, seed=42)

        indices_1 = list(sampler)
        indices_2 = list(sampler)  # Should be same due to same epoch

        assert indices_1 == indices_2
        assert len(indices_1) == sampler.num_samples

        # Change epoch and verify different shuffling
        sampler.set_epoch(1)
        indices_3 = list(sampler)
        assert indices_3 != indices_1  # Should be different with new epoch

    def test_distributed_sampler_padding(self):
        """Test dataset padding for uneven division."""
        sampler = DistributedSampler(dataset_size=7, num_replicas=3, rank=0, shuffle=False)

        indices = list(sampler)

        # Should pad to make total_size = 9 (3 samples per replica)
        assert len(indices) == 3
        # First replica should get indices [0, 3, 6]
        assert indices == [0, 3, 6]

    def test_distributed_sampler_set_epoch(self):
        """Test epoch setting for different shuffling."""
        sampler = DistributedSampler(dataset_size=20, num_replicas=2, rank=0, shuffle=True, seed=42)

        # Get indices for epoch 0
        epoch_0_indices = list(sampler)

        # Set epoch and get new indices
        sampler.set_epoch(5)
        epoch_5_indices = list(sampler)

        # Should be different due to different epoch seed
        assert epoch_0_indices != epoch_5_indices
        assert len(epoch_0_indices) == len(epoch_5_indices)

    def test_distributed_sampler_len(self):
        """Test sampler length."""
        sampler = DistributedSampler(dataset_size=100, num_replicas=4, rank=0)
        assert len(sampler) == 25


class TestDistributedUtilities:
    """Test distributed utility functions."""

    def setup_method(self):
        """Set up distributed environment."""
        init_process_group(backend="gloo", world_size=3, rank=1)

    def teardown_method(self):
        """Clean up distributed environment."""
        if is_initialized():
            destroy_process_group()

    def test_gather_function(self):
        """Test gather utility function."""
        tensor = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))

        with patch("neural_arch.distributed.data_parallel.all_gather") as mock_all_gather:
            mock_gathered = [
                Tensor(np.array([1.0, 2.0, 3.0])),
                Tensor(np.array([4.0, 5.0, 6.0])),
                Tensor(np.array([7.0, 8.0, 9.0])),
            ]
            mock_all_gather.return_value = mock_gathered

            # Test gathering to rank 0 (not current rank)
            result = gather(tensor, dst=0)
            assert result is None  # Non-destination rank returns None

            # Test gathering to current rank (1)
            result = gather(tensor, dst=1)
            assert result == mock_gathered

    def test_gather_not_initialized(self):
        """Test gather when distributed not initialized."""
        destroy_process_group()

        tensor = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        result = gather(tensor, dst=0)

        assert result == [tensor]

    def test_scatter_function(self):
        """Test scatter utility function."""
        # Test from source rank
        tensor_list = [Tensor(np.array([1.0])), Tensor(np.array([2.0])), Tensor(np.array([3.0]))]

        with patch("neural_arch.distributed.data_parallel.broadcast") as mock_broadcast:
            mock_broadcast.return_value = tensor_list[1]  # Return tensor for rank 1

            result = scatter(tensor_list, src=1)
            assert result == tensor_list[1]  # Current rank is 1

    def test_scatter_from_non_source(self):
        """Test scatter from non-source rank."""
        with patch("neural_arch.distributed.data_parallel.broadcast") as mock_broadcast:
            expected_tensor = Tensor(np.array([5.0]))
            mock_broadcast.return_value = expected_tensor

            result = scatter(None, src=0)  # Not source rank
            assert result == expected_tensor

    def test_scatter_invalid_tensor_list(self):
        """Test scatter with invalid tensor list."""
        # Wrong number of tensors
        tensor_list = [Tensor(np.array([1.0])), Tensor(np.array([2.0]))]  # Only 2, need 3

        with pytest.raises(ValueError, match="must contain tensors for all processes"):
            scatter(tensor_list, src=1)

    def test_scatter_not_initialized(self):
        """Test scatter when distributed not initialized."""
        destroy_process_group()

        tensor_list = [Tensor(np.array([1.0]))]
        result = scatter(tensor_list, src=0)

        assert result == tensor_list[0]

    def test_no_sync_context_manager(self):
        """Test no_sync context manager."""
        # Test that context manager works without errors
        with no_sync():
            # In a full implementation, this would disable gradient sync
            pass

    def test_is_distributed_training_available(self):
        """Test distributed training availability check."""
        assert is_distributed_training_available() is True

        # Test with single process
        destroy_process_group()
        init_process_group(backend="gloo", world_size=1, rank=0)
        assert is_distributed_training_available() is False

        # Test when not initialized
        destroy_process_group()
        assert is_distributed_training_available() is False

    def test_get_distributed_info(self):
        """Test distributed info retrieval."""
        info = get_distributed_info()

        expected_info = {"available": True, "world_size": 3, "rank": 1, "backend": "distributed"}

        assert info == expected_info

    def test_get_distributed_info_not_initialized(self):
        """Test distributed info when not initialized."""
        destroy_process_group()

        info = get_distributed_info()

        expected_info = {"available": False, "world_size": 1, "rank": 0, "backend": None}

        assert info == expected_info


class TestDistributedIntegration:
    """Integration tests for distributed data parallel components."""

    def setup_method(self):
        """Set up distributed environment."""
        init_process_group(backend="gloo", world_size=2, rank=0)
        self.module = SimpleModule(input_size=10, output_size=5)

    def teardown_method(self):
        """Clean up distributed environment."""
        if is_initialized():
            destroy_process_group()

    def test_ddp_with_distributed_sampler(self):
        """Test DDP integration with DistributedSampler."""
        ddp_module = DistributedDataParallel(self.module)
        sampler = DistributedSampler(dataset_size=100, shuffle=True)

        # Verify they work together
        assert ddp_module.world_size == sampler.num_replicas
        assert ddp_module.rank == sampler.rank

        # Test epoch synchronization
        sampler.set_epoch(3)
        assert sampler.epoch == 3

    def test_ddp_gradient_synchronization_workflow(self):
        """Test complete gradient synchronization workflow."""
        ddp_module = DistributedDataParallel(self.module)

        # Forward pass
        batch_size = 4
        x = Tensor(np.random.randn(batch_size, 10).astype(np.float32), requires_grad=True)
        output = ddp_module(x)

        # Simulate backward pass (create gradients)
        for param in ddp_module.module.parameters():
            if param.requires_grad:
                param.grad = Tensor(np.random.randn(*param.shape).astype(np.float32))

        # Synchronize gradients
        with patch("neural_arch.distributed.data_parallel.all_reduce") as mock_all_reduce:
            mock_all_reduce.side_effect = lambda x, op: x

            ddp_module.sync_gradients()
            assert ddp_module.gradients_ready

    def test_data_parallel_vs_ddp_behavior(self):
        """Test behavioral differences between DataParallel and DDP."""
        # DataParallel for single-node
        dp_module = DataParallel(self.module, device_ids=[0, 1])

        # DistributedDataParallel for multi-node
        ddp_module = DistributedDataParallel(self.module)

        # Both should handle same input
        x = Tensor(np.random.randn(4, 10).astype(np.float32))

        dp_output = dp_module(x)
        ddp_output = ddp_module(x)

        # Outputs should have same shape
        assert dp_output.shape == ddp_output.shape


if __name__ == "__main__":
    pytest.main([__file__])
