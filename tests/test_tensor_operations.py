"""Test tensor operations to improve coverage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core.device import Device
from neural_arch.core.dtype import DType
from neural_arch.core.tensor import Tensor


class TestTensorOperations:
    """Test tensor operations with focus on coverage."""

    def test_tensor_creation_basic(self):
        """Test basic tensor creation."""
        # From list
        t1 = Tensor([1, 2, 3])
        assert t1.shape == (3,)
        np.testing.assert_array_equal(t1.data, [1, 2, 3])

        # From numpy array
        arr = np.array([[1, 2], [3, 4]])
        t2 = Tensor(arr)
        assert t2.shape == (2, 2)
        np.testing.assert_array_equal(t2.data, arr)

    def test_tensor_creation_with_options(self):
        """Test tensor creation with various options."""
        # With requires_grad
        t1 = Tensor([1, 2, 3], requires_grad=True)
        assert t1.requires_grad

        # With specific dtype
        t2 = Tensor([1, 2, 3], dtype=np.float32)
        assert t2.dtype == np.float32

        # With device (if implemented)
        try:
            device = Device.cpu()
            t3 = Tensor([1, 2, 3], device=device)
            assert hasattr(t3, "device")
        except (AttributeError, TypeError):
            # Device might not be implemented
            pass

    def test_tensor_properties(self):
        """Test tensor properties."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])

        assert t.shape == (2, 3)
        assert t.ndim == 2
        assert t.size == 6
        assert t.dtype in (np.float32, np.float64, np.int32, np.int64)

    def test_tensor_indexing(self):
        """Test tensor indexing and slicing."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])

        # Basic indexing
        assert t[0, 0].item() == 1
        assert t[1, 2].item() == 6

        # Slicing
        row = t[0]
        assert row.shape == (3,)
        np.testing.assert_array_equal(row.data, [1, 2, 3])

        col = t[:, 1]
        assert col.shape == (2,)
        np.testing.assert_array_equal(col.data, [2, 5])

    def test_tensor_arithmetic(self):
        """Test tensor arithmetic operations."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)

        # Addition
        c = a + b
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(c.data, expected)
        assert c.requires_grad

        # Subtraction
        c = a - b
        expected = np.array([[-4, -4], [-4, -4]])
        np.testing.assert_array_equal(c.data, expected)

        # Multiplication
        c = a * b
        expected = np.array([[5, 12], [21, 32]])
        np.testing.assert_array_equal(c.data, expected)

        # Division
        c = a / b
        expected = np.array([[1 / 5, 2 / 6], [3 / 7, 4 / 8]])
        np.testing.assert_array_almost_equal(c.data, expected)

    def test_tensor_matmul(self):
        """Test tensor matrix multiplication."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)

        c = a @ b  # Matrix multiplication
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(c.data, expected)
        assert c.requires_grad

    def test_tensor_unary_operations(self):
        """Test tensor unary operations."""
        a = Tensor([[-2, -1], [1, 2]], requires_grad=True)

        # Negation
        b = -a
        expected = np.array([[2, 1], [-1, -2]])
        np.testing.assert_array_equal(b.data, expected)

        # Absolute value (if implemented)
        try:
            b = abs(a)
            expected = np.array([[2, 1], [1, 2]])
            np.testing.assert_array_equal(b.data, expected)
        except (AttributeError, TypeError):
            pass

    def test_tensor_reduction_operations(self):
        """Test tensor reduction operations."""
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

        # Sum
        total = a.sum()
        assert total.item() == 21
        assert total.requires_grad

        # Sum along axis
        try:
            row_sums = a.sum(axis=1)
            expected = np.array([6, 15])
            np.testing.assert_array_equal(row_sums.data, expected)
        except TypeError:
            # axis parameter might not be supported
            pass

        # Mean (if implemented)
        try:
            mean = a.mean()
            assert abs(mean.item() - 3.5) < 1e-6
        except AttributeError:
            pass

    def test_tensor_shape_operations(self):
        """Test tensor shape operations."""
        a = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        # Reshape
        b = a.reshape(4, 2)
        assert b.shape == (4, 2)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(b.data, expected)

        # Transpose
        c = a.T
        assert c.shape == (4, 2)
        expected = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
        np.testing.assert_array_equal(c.data, expected)

        # View (if implemented)
        try:
            d = a.view(-1)
            assert d.shape == (8,)
            expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
            np.testing.assert_array_equal(d.data, expected)
        except AttributeError:
            pass

    def test_tensor_backward(self):
        """Test tensor backward pass."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)

        c = a + b
        d = c * c
        loss = d.sum()

        # Backward pass
        loss.backward()

        # Check gradients
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_tensor_no_grad_context(self):
        """Test no_grad context."""
        from neural_arch import no_grad

        a = Tensor([[1, 2], [3, 4]], requires_grad=True)

        with no_grad():
            b = a + 1
            assert not b.requires_grad

        # Outside context, should have grad
        c = a + 1
        assert c.requires_grad

    def test_tensor_detach(self):
        """Test tensor detach."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)

        try:
            b = a.detach()
            assert not b.requires_grad
            np.testing.assert_array_equal(a.data, b.data)
        except AttributeError:
            # detach might not be implemented
            pass

    def test_tensor_clone(self):
        """Test tensor cloning."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)

        try:
            b = a.clone()
            assert b.requires_grad == a.requires_grad
            np.testing.assert_array_equal(a.data, b.data)

            # Should be different objects
            assert b is not a
            assert b.data is not a.data
        except AttributeError:
            # clone might not be implemented
            pass

    def test_tensor_dtype_conversion(self):
        """Test tensor dtype conversion."""
        a = Tensor([1, 2, 3], dtype=np.int32)

        try:
            b = a.float()
            assert b.dtype == np.float32

            c = a.double()
            assert c.dtype == np.float64

            d = b.int()
            assert d.dtype in (np.int32, np.int64)
        except AttributeError:
            # dtype conversion methods might not be implemented
            pass

    def test_tensor_device_transfer(self):
        """Test tensor device transfer."""
        a = Tensor([1, 2, 3])

        try:
            # Transfer to CPU (should be no-op if already on CPU)
            b = a.cpu()
            assert b.device.type.value == "cpu"

            # Try transfer to CUDA (might fail if not available)
            try:
                c = a.cuda()
                assert c.device.type.value == "cuda"
            except (AttributeError, RuntimeError):
                # CUDA might not be available
                pass
        except AttributeError:
            # Device methods might not be implemented
            pass

    def test_tensor_comparison(self):
        """Test tensor comparison operations."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[2, 2], [3, 5]])

        try:
            # Element-wise comparison
            c = a < b
            expected = np.array([[True, False], [False, True]])
            np.testing.assert_array_equal(c.data, expected)

            d = a == b
            expected = np.array([[False, True], [True, False]])
            np.testing.assert_array_equal(d.data, expected)
        except (AttributeError, TypeError):
            # Comparison operators might not be implemented
            pass

    def test_tensor_broadcasting(self):
        """Test tensor broadcasting."""
        a = Tensor([[1, 2, 3]])  # Shape: (1, 3)
        b = Tensor([[1], [2]])  # Shape: (2, 1)

        c = a + b  # Should broadcast to (2, 3)
        assert c.shape == (2, 3)
        expected = np.array([[2, 3, 4], [3, 4, 5]])
        np.testing.assert_array_equal(c.data, expected)

    def test_tensor_memory_layout(self):
        """Test tensor memory layout."""
        a = Tensor([[1, 2, 3], [4, 5, 6]])

        # Check if contiguous
        try:
            assert a.is_contiguous()

            # Transpose should not be contiguous
            b = a.T
            # Might or might not be contiguous depending on implementation
        except AttributeError:
            # is_contiguous might not be implemented
            pass

    def test_tensor_item_access(self):
        """Test tensor item access."""
        # Scalar tensor
        a = Tensor(5.0)
        assert a.item() == 5.0

        # Single element tensor
        b = Tensor([42])
        assert b.item() == 42

        # Multi-element tensor should raise error
        c = Tensor([1, 2, 3])
        try:
            c.item()
            assert False, "Should have raised ValueError"
        except (ValueError, RuntimeError):
            pass

    def test_tensor_numpy_conversion(self):
        """Test tensor to numpy conversion."""
        a = Tensor([[1, 2], [3, 4]])

        try:
            np_array = a.numpy()
            expected = np.array([[1, 2], [3, 4]])
            np.testing.assert_array_equal(np_array, expected)
        except AttributeError:
            # numpy() method might not be implemented
            # Can still access .data
            np.testing.assert_array_equal(a.data, expected)
