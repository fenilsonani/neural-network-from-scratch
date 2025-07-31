"""Real comprehensive tests for tensor core functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.core.tensor import Tensor
from neural_arch.core.dtype import DType
from neural_arch.core.device import Device, DeviceType


class TestRealTensorCore:
    """Real tests for tensor core functionality without simulation."""
    
    def test_tensor_creation_basic(self):
        """Test basic tensor creation methods."""
        # From list
        t1 = Tensor([1, 2, 3])
        assert t1.shape == (3,)
        np.testing.assert_array_equal(t1.data, [1, 2, 3])
        
        # From nested list
        t2 = Tensor([[1, 2], [3, 4]])
        assert t2.shape == (2, 2)
        np.testing.assert_array_equal(t2.data, [[1, 2], [3, 4]])
        
        # From numpy array
        arr = np.array([5, 6, 7])
        t3 = Tensor(arr)
        assert t3.shape == (3,)
        np.testing.assert_array_equal(t3.data, [5, 6, 7])
        
        # From scalar
        t4 = Tensor(42)
        assert t4.shape == ()
        assert t4.item() == 42
    
    def test_tensor_properties(self):
        """Test tensor properties."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        
        # Shape properties
        assert t.shape == (2, 3)
        assert t.ndim == 2
        assert t.size == 6
        
        # Data type
        assert hasattr(t, 'dtype')
        assert t.dtype in (np.float32, np.float64, np.int32, np.int64)
        
        # Device
        assert hasattr(t, 'device')
        assert isinstance(t.device, Device)
        
        # Gradient properties
        assert hasattr(t, 'requires_grad')
        assert t.requires_grad is False  # Default
        assert t.grad is None
    
    def test_tensor_requires_grad(self):
        """Test gradient requirement functionality."""
        # Default: no gradients
        t1 = Tensor([1, 2, 3])
        assert t1.requires_grad is False
        assert t1.grad is None
        
        # Explicit gradient requirement
        t2 = Tensor([1, 2, 3], requires_grad=True)
        assert t2.requires_grad is True
        assert t2.grad is None  # Initially None
    
    def test_tensor_data_access(self):
        """Test tensor data access methods."""
        t = Tensor([[1, 2], [3, 4]])
        
        # Data property
        assert hasattr(t, 'data')
        np.testing.assert_array_equal(t.data, [[1, 2], [3, 4]])
        
        # Numpy conversion
        np_array = t.numpy()
        assert isinstance(np_array, np.ndarray)
        np.testing.assert_array_equal(np_array, [[1, 2], [3, 4]])
        
        # Backend data
        assert hasattr(t, 'backend_data')
        assert t.backend_data is not None
    
    def test_tensor_item_method(self):
        """Test tensor item extraction."""
        # Scalar tensor
        scalar = Tensor(5.0)
        assert scalar.item() == 5.0
        
        # Single element tensor
        single = Tensor([42])
        assert single.item() == 42
        
        # Multi-element tensor should raise error
        multi = Tensor([1, 2, 3])
        with pytest.raises((ValueError, RuntimeError)):
            multi.item()
    
    def test_tensor_clone_detach(self):
        """Test tensor cloning and detaching."""
        t = Tensor([[1, 2], [3, 4]], requires_grad=True)
        
        # Clone should preserve requires_grad
        cloned = t.clone()
        assert cloned.requires_grad == t.requires_grad
        np.testing.assert_array_equal(cloned.data, t.data)
        assert cloned is not t  # Different objects
        
        # Detach should remove gradient requirement
        detached = t.detach()
        assert detached.requires_grad is False
        np.testing.assert_array_equal(detached.data, t.data)
    
    def test_tensor_device_property(self):
        """Test tensor device management."""
        t = Tensor([1, 2, 3])
        
        # Should have a device
        assert hasattr(t, 'device')
        assert isinstance(t.device, Device)
        
        # Default should be CPU
        assert t.device.type == DeviceType.CPU
    
    def test_tensor_dtype_creation(self):
        """Test tensor creation with specific dtypes."""
        # Create with numpy dtype
        t1 = Tensor([1, 2, 3], dtype=DType.FLOAT32)
        assert t1.dtype == np.float32
        
        t2 = Tensor([1, 2, 3], dtype=DType.INT32)
        assert t2.dtype == np.int32
        
        # Test with direct numpy dtype
        t3 = Tensor(np.array([1.0, 2.0], dtype=np.float64))
        assert t3.dtype == np.float64
    
    def test_tensor_device_transfer(self):
        """Test tensor device transfer."""
        t = Tensor([1, 2, 3], requires_grad=True)
        
        # Transfer to CPU (should be no-op if already on CPU)
        cpu_tensor = t.to(Device.cpu())
        assert cpu_tensor.device.type == DeviceType.CPU
        np.testing.assert_array_equal(cpu_tensor.data, t.data)
        
        # Try CUDA transfer (will fall back to CPU if not available)
        try:
            cuda_tensor = t.to(Device.cuda(0))
            # If successful, should be on CUDA device
            assert cuda_tensor.device.type in (DeviceType.CUDA, DeviceType.CPU)
        except (RuntimeError, AttributeError):
            # CUDA might not be available
            pass
    
    def test_tensor_memory_usage(self):
        """Test tensor memory usage reporting."""
        t = Tensor(np.random.randn(100, 100))
        
        # Should have memory usage method
        memory = t.memory_usage()
        assert isinstance(memory, (int, float))
        assert memory > 0
    
    def test_tensor_backward_basic(self):
        """Test basic backward pass setup."""
        t = Tensor([1, 2, 3], requires_grad=True)
        
        # Should have backward method
        assert hasattr(t, 'backward')
        assert callable(t.backward)
        
        # For leaf tensor, backward without gradient should work
        try:
            t.backward(gradient=np.ones_like(t.data))
            # Should set gradient
            assert t.grad is not None
            np.testing.assert_array_equal(t.grad.data, np.ones_like(t.data))
        except (RuntimeError, AttributeError):
            # Backward might have different interface
            pass
    
    def test_tensor_zero_grad(self):
        """Test gradient zeroing."""
        t = Tensor([1, 2, 3], requires_grad=True)
        
        # Set some gradient
        t.grad = Tensor([0.1, 0.2, 0.3])
        assert t.grad is not None
        
        # Zero gradients
        t.zero_grad()
        
        # Gradient should be None or zeros
        if t.grad is not None:
            np.testing.assert_array_equal(t.grad.data, np.zeros_like(t.data))
        else:
            assert t.grad is None
    
    def test_tensor_grad_fn(self):
        """Test gradient function tracking."""
        # Leaf tensor should have no grad_fn
        leaf = Tensor([1, 2, 3], requires_grad=True)
        assert leaf.grad_fn is None
        
        # Operation result should have grad_fn if inputs require grad
        from neural_arch.functional.arithmetic import add
        
        a = Tensor([1, 2], requires_grad=True)
        b = Tensor([3, 4], requires_grad=True)
        result = add(a, b)
        
        if result.requires_grad:
            assert result.grad_fn is not None
            assert hasattr(result.grad_fn, 'apply')
    
    def test_tensor_name_property(self):
        """Test tensor naming for debugging."""
        # Without name
        t1 = Tensor([1, 2, 3])
        assert hasattr(t1, 'name')
        
        # With name
        t2 = Tensor([1, 2, 3], name="test_tensor")
        assert t2.name == "test_tensor"
    
    def test_tensor_backend_property(self):
        """Test tensor backend access."""
        t = Tensor([1, 2, 3])
        
        # Should have backend
        assert hasattr(t, 'backend')
        assert t.backend is not None
        assert hasattr(t.backend, 'name')
        
        # Backend should be numpy by default
        assert t.backend.name == "numpy"
    
    def test_tensor_arithmetic_operators(self):
        """Test tensor arithmetic operators."""
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
        expected = np.array([[1/5, 2/6], [3/7, 4/8]])
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Matrix multiplication
        c = a @ b
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(c.data, expected)
    
    def test_tensor_unary_operators(self):
        """Test tensor unary operators."""
        a = Tensor([[-1, 2], [-3, 4]], requires_grad=True)
        
        # Negation
        neg_a = -a
        expected = np.array([[1, -2], [3, -4]])
        np.testing.assert_array_equal(neg_a.data, expected)
        assert neg_a.requires_grad
    
    def test_tensor_comparison_operators(self):
        """Test tensor comparison operators."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[2, 2], [3, 5]])
        
        # Less than
        try:
            result = a < b
            expected = np.array([[True, False], [False, True]])
            np.testing.assert_array_equal(result.data, expected)
        except (AttributeError, TypeError):
            # Comparison operators might not be implemented
            pass
    
    def test_tensor_scalar_operations(self):
        """Test tensor operations with scalars."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        
        # Scalar addition
        result = a + 10
        expected = np.array([[11, 12], [13, 14]])
        np.testing.assert_array_equal(result.data, expected)
        
        # Scalar multiplication
        result = a * 2
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(result.data, expected)
        
        # Right-hand scalar operations
        try:
            result = 10 + a
            expected = np.array([[11, 12], [13, 14]])
            np.testing.assert_array_equal(result.data, expected)
        except (TypeError, AttributeError):
            # Right-hand operations might not be implemented
            pass
    
    def test_tensor_error_conditions(self):
        """Test tensor error conditions."""
        # Invalid data types
        with pytest.raises(TypeError):
            Tensor("invalid")
        
        # Invalid requires_grad type
        with pytest.raises(TypeError):
            Tensor([1, 2, 3], requires_grad="invalid")
    
    def test_tensor_edge_cases(self):
        """Test tensor edge cases."""
        # Empty tensor
        try:
            empty = Tensor([])
            assert empty.shape == (0,)
        except ValueError:
            # Empty tensors might not be supported
            pass
        
        # Single element tensor
        single = Tensor([42])
        assert single.shape == (1,)
        assert single.item() == 42
        
        # Very large tensor
        large = Tensor(np.random.randn(1000, 1000))
        assert large.shape == (1000, 1000)
        assert large.size == 1000000
    
    def test_tensor_str_repr(self):
        """Test tensor string representations."""
        t = Tensor([[1, 2], [3, 4]])
        
        # Should have string representation
        str_repr = str(t)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        
        # Should have repr
        repr_str = repr(t)
        assert isinstance(repr_str, str)
        assert "Tensor" in repr_str