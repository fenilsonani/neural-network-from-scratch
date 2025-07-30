"""Comprehensive tests to maximize coverage."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.core.tensor import Tensor
from neural_arch.core.device import Device, DeviceType, get_default_device, set_default_device
from neural_arch.core.dtype import DType
from neural_arch.optim import Adam, SGD
from neural_arch.nn import Linear
from neural_arch.backends import get_backend, available_backends
from neural_arch import functional as F


class TestComprehensiveCoverage:
    """Comprehensive tests to maximize code coverage."""
    
    def test_device_functionality(self):
        """Test device functionality comprehensively."""
        # Test device creation methods
        cpu_device = Device.cpu()
        assert cpu_device.type == DeviceType.CPU
        assert cpu_device.is_cpu
        assert not cpu_device.is_gpu
        
        # Test CUDA device
        cuda_device = Device.cuda(0)
        assert cuda_device.type == DeviceType.CUDA
        assert cuda_device.is_cuda
        assert cuda_device.is_gpu
        assert cuda_device.index == 0
        
        # Test MPS device
        mps_device = Device.mps(0)
        assert mps_device.type == DeviceType.MPS
        assert mps_device.is_mps
        assert mps_device.is_gpu
        
        # Test device string parsing
        device_from_str = Device.from_string("cpu")
        assert device_from_str == cpu_device
        
        cuda_from_str = Device.from_string("cuda:1")
        assert cuda_from_str.type == DeviceType.CUDA
        assert cuda_from_str.index == 1
        
        # Test device equality and hashing
        cpu1 = Device.cpu()
        cpu2 = Device.cpu()
        assert cpu1 == cpu2
        assert hash(cpu1) == hash(cpu2)
        
        # Test device string representation
        assert str(cpu_device) == "cpu"
        assert str(cuda_device) == "cuda:0"
        assert "Device" in repr(cuda_device)
        
        # Test default device management
        original_device = get_default_device()
        set_default_device(cuda_device)
        assert get_default_device() == cuda_device
        set_default_device(original_device)  # Reset
    
    def test_dtype_functionality(self):
        """Test DType functionality."""
        # Test dtype creation and properties
        float32_dtype = DType.float32()
        assert float32_dtype.numpy_dtype == np.float32
        assert float32_dtype.is_floating_point
        assert not float32_dtype.is_integer
        
        int32_dtype = DType.int32()
        assert int32_dtype.numpy_dtype == np.int32
        assert not int32_dtype.is_floating_point
        assert int32_dtype.is_integer
        
        # Test dtype string representation
        assert "float32" in str(float32_dtype)
        assert "DType" in repr(float32_dtype)
        
        # Test dtype from numpy
        dtype_from_np = DType.from_numpy(np.float64)
        assert dtype_from_np.numpy_dtype == np.float64
        
        # Test dtype equality
        float32_1 = DType.float32()
        float32_2 = DType.float32()
        assert float32_1 == float32_2
    
    def test_optimizer_functionality(self):
        """Test optimizer functionality comprehensively."""
        # Create a simple model
        layer = Linear(3, 2)
        
        # Test Adam optimizer
        adam = Adam(layer.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        
        # Forward pass
        x = Tensor([[1, 2, 3]], requires_grad=True)
        output = layer(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        adam.step()
        
        # Test optimizer state
        assert hasattr(adam, 'param_groups')
        assert len(adam.param_groups) > 0
        
        # Test zero grad
        adam.zero_grad()
        for param in layer.parameters():
            if param.grad is not None:
                assert np.allclose(param.grad.data, 0)
        
        # Test SGD optimizer if available
        try:
            sgd = SGD(layer.parameters(), lr=0.01, momentum=0.9)
            
            # Another forward/backward pass
            output = layer(x)
            loss = output.sum()
            loss.backward()
            sgd.step()
            sgd.zero_grad()
        except (AttributeError, TypeError):
            pass
    
    def test_backend_functionality(self):
        """Test backend functionality comprehensively."""
        # Test available backends
        backends = available_backends()
        assert "numpy" in backends
        assert len(backends) >= 1
        
        # Test getting backends
        numpy_backend = get_backend("numpy")
        assert numpy_backend.name == "numpy"
        assert hasattr(numpy_backend, 'array')
        assert hasattr(numpy_backend, 'matmul')
        
        # Test backend operations
        x = numpy_backend.array([[1, 2], [3, 4]], dtype=numpy_backend.float32)
        y = numpy_backend.array([[5, 6], [7, 8]], dtype=numpy_backend.float32)
        
        # Test arithmetic operations
        add_result = numpy_backend.add(x, y)
        assert add_result.shape == (2, 2)
        
        matmul_result = numpy_backend.matmul(x, y)
        assert matmul_result.shape == (2, 2)
        
        # Test backend properties
        assert hasattr(numpy_backend, 'float32')
        assert hasattr(numpy_backend, 'float64')
        assert hasattr(numpy_backend, 'available')
        assert numpy_backend.available
    
    def test_tensor_advanced_operations(self):
        """Test advanced tensor operations."""
        # Test tensor creation with various parameters
        t1 = Tensor([1, 2, 3], requires_grad=True, dtype=np.float32)
        assert t1.requires_grad
        assert t1.dtype == np.float32
        
        # Test tensor operations with gradient tracking
        t2 = Tensor([4, 5, 6], requires_grad=True)
        t3 = t1 + t2
        assert t3.requires_grad
        
        # Test complex operations
        t4 = t3 * t3  # Square
        t5 = t4.sum()  # Reduce
        
        # Backward pass
        t5.backward()
        assert t1.grad is not None
        assert t2.grad is not None
        
        # Test tensor methods
        assert t1.ndim == 1
        assert t1.size == 3
        assert t1.shape == (3,)
        
        # Test indexing and slicing
        t_2d = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t_2d[0, 1].item() == 2
        row = t_2d[1]
        assert row.shape == (3,)
        
        # Test reshape and transpose
        reshaped = t_2d.reshape(3, 2)
        assert reshaped.shape == (3, 2)
        
        transposed = t_2d.T
        assert transposed.shape == (3, 2)
    
    def test_functional_edge_cases(self):
        """Test functional operations edge cases."""
        # Test with different shapes
        a = Tensor([[1]], requires_grad=True)  # (1, 1)
        b = Tensor([2], requires_grad=True)    # (1,)
        
        # Broadcasting addition
        c = F.add(a, b)
        assert c.shape == (1, 1)
        
        # Test with scalars
        d = F.add(a, 5)
        assert d.shape == a.shape
        
        # Test activation functions with edge values
        extreme_vals = Tensor([[-1000, 0, 1000]], requires_grad=True)
        
        # ReLU should handle large values
        relu_result = F.relu(extreme_vals)
        assert relu_result.data[0, 0] == 0  # Negative clipped
        assert relu_result.data[0, 2] == 1000  # Positive preserved
        
        # Softmax should handle large values without overflow
        softmax_result = F.softmax(extreme_vals, axis=1)
        assert softmax_result.shape == extreme_vals.shape
        assert np.allclose(np.sum(softmax_result.data, axis=1), 1.0)
    
    def test_neural_network_layers(self):
        """Test neural network layers comprehensively."""
        # Test Linear layer variations
        layer_with_bias = Linear(4, 3, bias=True)
        layer_no_bias = Linear(4, 3, bias=False)
        
        assert layer_with_bias.bias is not None
        assert layer_no_bias.bias is None
        
        # Test forward pass
        x = Tensor([[1, 2, 3, 4]], requires_grad=True)
        
        out_with_bias = layer_with_bias(x)
        out_no_bias = layer_no_bias(x)
        
        assert out_with_bias.shape == (1, 3)
        assert out_no_bias.shape == (1, 3)
        
        # Test parameter collection
        params_with_bias = list(layer_with_bias.parameters())
        params_no_bias = list(layer_no_bias.parameters())
        
        assert len(params_with_bias) == 2  # weight + bias
        assert len(params_no_bias) == 1   # weight only
        
        # Test layer representation
        layer_repr = repr(layer_with_bias)
        assert "Linear" in layer_repr
        assert "4" in layer_repr
        assert "3" in layer_repr
    
    def test_gradient_computation(self):
        """Test gradient computation comprehensively."""
        # Create a more complex computation graph
        x = Tensor([[1, 2]], requires_grad=True)
        w1 = Tensor([[0.5, 0.3], [0.7, 0.1]], requires_grad=True)
        b1 = Tensor([0.1, 0.2], requires_grad=True)
        
        # Forward pass: y = x @ w1 + b1
        y = x @ w1 + b1
        
        # Apply activation
        z = F.relu(y)
        
        # Final loss
        loss = z.sum()
        
        # Backward pass
        loss.backward()
        
        # Check all gradients exist and have correct shapes
        assert x.grad is not None
        assert w1.grad is not None
        assert b1.grad is not None
        
        assert x.grad.shape == x.shape
        assert w1.grad.shape == w1.shape
        assert b1.grad.shape == b1.shape
    
    def test_error_conditions(self):
        """Test error conditions and edge cases."""
        # Test invalid tensor operations
        a = Tensor([[1, 2, 3]])  # (1, 3)
        b = Tensor([[1, 2]])     # (1, 2)
        
        # Matrix multiplication with incompatible shapes
        with pytest.raises((ValueError, RuntimeError)):
            a @ b  # Should fail
        
        # Test device validation errors
        with pytest.raises(ValueError):
            Device(DeviceType.CUDA, -1)  # Negative index
        
        # Test device string parsing errors
        with pytest.raises(ValueError):
            Device.from_string("invalid_device")
        
        # Test tensor item() on multi-element tensor
        multi_elem = Tensor([1, 2, 3])
        with pytest.raises((ValueError, RuntimeError)):
            multi_elem.item()
    
    def test_memory_management(self):
        """Test memory management and cleanup."""
        # Create tensors with gradients
        tensors = []
        for i in range(10):
            t = Tensor(np.random.randn(100, 100), requires_grad=True)
            t.sum().backward()  # Create gradients
            tensors.append(t)
        
        # Clear gradients
        for t in tensors:
            if hasattr(t, 'zero_grad'):
                t.zero_grad()
            elif t.grad is not None:
                t.grad = None
        
        # Tensors should still be valid
        for t in tensors:
            assert t.data is not None
            assert t.shape == (100, 100)
    
    def test_dtype_conversions(self):
        """Test dtype conversions."""
        # Start with int32
        t_int = Tensor([1, 2, 3], dtype=np.int32)
        
        # Convert to different types (if methods exist)
        try:
            t_float = t_int.float()
            assert t_float.dtype == np.float32
            
            t_double = t_int.double()
            assert t_double.dtype == np.float64
            
            # Convert back to int
            t_int_back = t_float.int()
            assert t_int_back.dtype in (np.int32, np.int64)
        except AttributeError:
            # Conversion methods might not be implemented
            pass
        
        # Test explicit dtype creation
        t_explicit = Tensor([1.0, 2.0, 3.0], dtype=np.float64)
        assert t_explicit.dtype == np.float64
    
    def test_context_managers(self):
        """Test context managers."""
        from neural_arch import no_grad, enable_grad, is_grad_enabled
        
        # Test no_grad context
        x = Tensor([1, 2, 3], requires_grad=True)
        
        # Check initial state
        initial_grad_enabled = is_grad_enabled()
        
        with no_grad():
            assert not is_grad_enabled()
            y = x + 1
            assert not y.requires_grad
        
        # Should restore previous state
        assert is_grad_enabled() == initial_grad_enabled
        
        # Test enable_grad context
        with no_grad():
            with enable_grad():
                assert is_grad_enabled()
                z = x + 1
                assert z.requires_grad
    
    def test_serialization_concepts(self):
        """Test serialization concepts (if implemented)."""
        layer = Linear(3, 2)
        
        try:
            # Test state dict
            state_dict = layer.state_dict()
            assert isinstance(state_dict, dict)
            assert 'weight' in state_dict
            
            # Test loading state dict
            new_layer = Linear(3, 2)
            new_layer.load_state_dict(state_dict)
            
            # Weights should be equal
            np.testing.assert_array_equal(
                layer.weight.data,
                new_layer.weight.data
            )
        except AttributeError:
            # Serialization might not be implemented
            pass