"""Comprehensive tests for nn.pooling module to improve coverage from 78.57% to 100%.

This file targets the MeanPool and MaxPool layer classes.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core import Module
from neural_arch.core.tensor import Tensor
from neural_arch.nn.pooling import MaxPool, MeanPool


class TestNNPoolingComprehensive:
    """Comprehensive tests for nn.pooling layers."""

    def test_mean_pool_layer_basic(self):
        """Test MeanPool layer basic functionality."""
        # Create layer
        layer = MeanPool(axis=1)

        # Check it's a Module
        assert isinstance(layer, Module)
        assert layer.axis == 1

        # Test forward pass
        x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]], requires_grad=True)
        output = layer(x)

        # Expected: mean along axis=1
        expected = np.array([2.5, 6.5])
        np.testing.assert_array_almost_equal(output.data, expected, decimal=6)

        # Test gradient flow
        output.backward(np.ones_like(output.data))
        assert x.grad is not None

    def test_mean_pool_layer_different_axes(self):
        """Test MeanPool layer with different axes."""
        # Test axis=0
        layer_axis0 = MeanPool(axis=0)
        assert layer_axis0.axis == 0

        x = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        output = layer_axis0(x)

        # Mean along axis=0: [3, 4]
        expected = np.array([3, 4])
        np.testing.assert_array_almost_equal(output.data, expected, decimal=6)

        # Test axis=2
        layer_axis2 = MeanPool(axis=2)
        assert layer_axis2.axis == 2

        x_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
        output_3d = layer_axis2(x_3d)

        # Mean along axis=2
        expected_3d = np.array([[1.5, 3.5], [5.5, 7.5]])
        np.testing.assert_array_almost_equal(output_3d.data, expected_3d, decimal=6)

    def test_mean_pool_layer_default_axis(self):
        """Test MeanPool layer with default axis."""
        # Default axis should be 1
        layer = MeanPool()
        assert layer.axis == 1

        # Test it works
        x = Tensor([[1, 2, 3]], requires_grad=True)
        output = layer(x)
        expected = np.array([2])
        np.testing.assert_array_almost_equal(output.data, expected, decimal=6)

    def test_max_pool_layer_basic(self):
        """Test MaxPool layer basic functionality."""
        # Create layer
        layer = MaxPool(axis=1)

        # Check it's a Module
        assert isinstance(layer, Module)
        assert layer.axis == 1

        # Test forward pass
        x = Tensor([[1, 5, 3, 2], [8, 6, 7, 4]], requires_grad=True)
        output = layer(x)

        # Expected: max along axis=1
        expected = np.array([5, 8])
        np.testing.assert_array_equal(output.data, expected)

        # Test gradient flow
        output.backward(np.ones_like(output.data))
        assert x.grad is not None

    def test_max_pool_layer_different_axes(self):
        """Test MaxPool layer with different axes."""
        # Test axis=0
        layer_axis0 = MaxPool(axis=0)
        assert layer_axis0.axis == 0

        x = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        output = layer_axis0(x)

        # Max along axis=0: [5, 6]
        expected = np.array([5, 6])
        np.testing.assert_array_equal(output.data, expected)

        # Test axis=2
        layer_axis2 = MaxPool(axis=2)
        assert layer_axis2.axis == 2

        x_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
        output_3d = layer_axis2(x_3d)

        # Max along axis=2
        expected_3d = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(output_3d.data, expected_3d)

    def test_max_pool_layer_default_axis(self):
        """Test MaxPool layer with default axis."""
        # Default axis should be 1
        layer = MaxPool()
        assert layer.axis == 1

        # Test it works
        x = Tensor([[1, 5, 3]], requires_grad=True)
        output = layer(x)
        expected = np.array([5])
        np.testing.assert_array_equal(output.data, expected)

    def test_pooling_layers_module_behavior(self):
        """Test that pooling layers behave as proper modules."""
        # Test MeanPool
        mean_layer = MeanPool(axis=2)

        # Should have Module methods
        assert hasattr(mean_layer, "forward")
        assert hasattr(mean_layer, "__call__")
        assert hasattr(mean_layer, "train")
        assert hasattr(mean_layer, "eval")
        assert hasattr(mean_layer, "parameters")

        # Parameters should be empty (no learnable params)
        params = list(mean_layer.parameters())
        assert len(params) == 0

        # Test MaxPool
        max_layer = MaxPool(axis=2)

        # Should have Module methods
        assert hasattr(max_layer, "forward")
        assert hasattr(max_layer, "__call__")
        assert hasattr(max_layer, "train")
        assert hasattr(max_layer, "eval")
        assert hasattr(max_layer, "parameters")

        # Parameters should be empty
        params = list(max_layer.parameters())
        assert len(params) == 0

    def test_pooling_layers_training_eval_modes(self):
        """Test pooling layers in training and eval modes."""
        # Create layers
        mean_layer = MeanPool(axis=1)
        max_layer = MaxPool(axis=1)

        # Test data
        x = Tensor([[1, 2, 3, 4]], requires_grad=True)

        # Training mode (default)
        mean_layer.train()
        max_layer.train()

        assert mean_layer.training is True
        assert max_layer.training is True

        # Forward should work
        mean_out_train = mean_layer(x)
        max_out_train = max_layer(x)

        # Eval mode
        mean_layer.eval()
        max_layer.eval()

        assert mean_layer.training is False
        assert max_layer.training is False

        # Forward should work and give same results (no dropout/batchnorm)
        mean_out_eval = mean_layer(x)
        max_out_eval = max_layer(x)

        np.testing.assert_array_equal(mean_out_train.data, mean_out_eval.data)
        np.testing.assert_array_equal(max_out_train.data, max_out_eval.data)

    def test_pooling_layers_negative_axis(self):
        """Test pooling layers with negative axis values."""
        # Negative axis should work
        mean_layer = MeanPool(axis=-1)
        max_layer = MaxPool(axis=-1)

        # 2D tensor, axis=-1 is the last axis
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

        mean_out = mean_layer(x)
        max_out = max_layer(x)

        # Expected results (same as axis=1 for 2D)
        expected_mean = np.array([2, 5])
        expected_max = np.array([3, 6])

        np.testing.assert_array_almost_equal(mean_out.data, expected_mean, decimal=6)
        np.testing.assert_array_equal(max_out.data, expected_max)

    def test_pooling_layers_large_tensors(self):
        """Test pooling layers with large tensors."""
        # Create layers
        mean_layer = MeanPool(axis=1)
        max_layer = MaxPool(axis=2)

        # Large tensor
        large_x = Tensor(np.random.randn(10, 50, 100), requires_grad=True)

        # Forward passes should work
        mean_out = mean_layer(large_x)
        assert mean_out.shape == (10, 100)

        max_out = max_layer(large_x)
        assert max_out.shape == (10, 50)

        # Backward should work
        mean_out.backward(np.ones_like(mean_out.data))
        assert large_x.grad is not None

        large_x.zero_grad()
        max_out.backward(np.ones_like(max_out.data))
        assert large_x.grad is not None

    def test_pooling_repr_and_str(self):
        """Test string representations of pooling layers."""
        mean_layer = MeanPool(axis=2)
        max_layer = MaxPool(axis=0)

        # Should have meaningful representations
        mean_str = str(mean_layer)
        max_str = str(max_layer)

        # Basic check - should contain class name
        assert "MeanPool" in mean_str or "Module" in mean_str
        assert "MaxPool" in max_str or "Module" in max_str
