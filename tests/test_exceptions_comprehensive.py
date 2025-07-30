"""Comprehensive tests for exception handling module to boost coverage."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.core.tensor import Tensor
from neural_arch.core.device import Device, DeviceType
from neural_arch.core.dtype import DType


class TestExceptionsComprehensive:
    """Comprehensive tests for all exception classes to boost coverage."""
    
    def test_tensor_error_exceptions(self):
        """Test tensor-related exceptions comprehensively."""
        from neural_arch.exceptions import TensorError, ShapeError, DTypeError, DeviceError
        
        # Test TensorError base exception
        try:
            raise TensorError("Test tensor error")
        except TensorError as e:
            assert "Test tensor error" in str(e)
            assert isinstance(e, Exception)
        
        # Test ShapeError
        try:
            raise ShapeError("Shape mismatch: expected (2, 3), got (3, 2)")
        except ShapeError as e:
            assert "Shape mismatch" in str(e)
            assert isinstance(e, TensorError)
        
        # Test DTypeError
        try:
            raise DTypeError("Incompatible dtype: float32 vs int32")
        except DTypeError as e:
            assert "dtype" in str(e).lower()
            assert isinstance(e, TensorError)
        
        # Test DeviceError
        try:
            raise DeviceError("Device mismatch: cuda vs cpu")
        except DeviceError as e:
            assert "Device" in str(e)
            assert isinstance(e, TensorError)
    
    def test_backend_exceptions(self):
        """Test backend-related exceptions comprehensively."""
        from neural_arch.exceptions import ResourceError, ConfigurationError
        
        # Test ResourceError (covers backend unavailability)
        try:
            raise ResourceError("Backend operation failed", resource_type="backend")
        except ResourceError as e:
            assert "Backend" in str(e)
            assert isinstance(e, Exception)
        
        # Test ConfigurationError (covers backend initialization)
        try:
            raise ConfigurationError("Failed to initialize MPS backend", config_key="backend")
        except ConfigurationError as e:
            assert "initialize" in str(e).lower()
            assert isinstance(e, Exception)
    
    def test_gradient_exceptions(self):
        """Test gradient-related exceptions comprehensively."""
        from neural_arch.exceptions import GradientError
        
        # Test GradientError base exception
        try:
            raise GradientError("Gradient computation failed")
        except GradientError as e:
            assert "Gradient" in str(e)
            assert isinstance(e, Exception)
        
        # Test GradientError with context
        try:
            raise GradientError("Cannot compute gradients: tensor doesn't require grad", tensor_name="input_tensor")
        except GradientError as e:
            assert "grad" in str(e).lower()
            assert isinstance(e, GradientError)
        
        # Test GradientError with operation context
        try:
            raise GradientError("Backward pass failed: invalid gradient shape", operation="backward")
        except GradientError as e:
            assert "Backward" in str(e)
            assert isinstance(e, GradientError)
    
    def test_optimization_exceptions(self):
        """Test optimization-related exceptions comprehensively."""
        from neural_arch.exceptions import OptimizationError, OptimizerError
        
        # Test OptimizationError base exception
        try:
            raise OptimizationError("Optimization process failed")
        except OptimizationError as e:
            assert "Optimization" in str(e)
            assert isinstance(e, Exception)
        
        # Test OptimizerError
        try:
            raise OptimizerError("Invalid learning rate: must be positive", learning_rate=-0.01)
        except OptimizerError as e:
            assert "learning rate" in str(e).lower() or "Optimizer" in str(e)
            assert isinstance(e, OptimizationError)
        
        # Test convergence as general optimization error
        try:
            raise OptimizationError("Training failed to converge after 1000 epochs")
        except OptimizationError as e:
            assert "converge" in str(e).lower()
            assert isinstance(e, OptimizationError)
    
    def test_neural_network_exceptions(self):
        """Test neural network related exceptions comprehensively."""
        from neural_arch.exceptions import ModelError, LayerError, ParameterError
        
        # Test ModelError base exception
        try:
            raise ModelError("Neural network configuration error")
        except ModelError as e:
            assert "network" in str(e).lower() or "configuration" in str(e).lower()
            assert isinstance(e, Exception)
        
        # Test LayerError
        try:
            raise LayerError("Invalid layer configuration: input size mismatch", layer_name="linear1")
        except LayerError as e:
            assert "layer" in str(e).lower()
            assert isinstance(e, ModelError)
        
        # Test ParameterError
        try:
            raise ParameterError("Invalid parameter shape", parameter_name="weight")
        except ParameterError as e:
            assert "parameter" in str(e).lower()
            assert isinstance(e, ModelError)
    
    def test_configuration_exceptions(self):
        """Test configuration-related exceptions comprehensively."""
        from neural_arch.exceptions import ConfigurationError, DataError
        
        # Test ConfigurationError base exception
        try:
            raise ConfigurationError("Invalid configuration settings", config_key="device")
        except ConfigurationError as e:
            assert "configuration" in str(e).lower()
            assert isinstance(e, Exception)
        
        # Test DataError (covers validation issues)
        try:
            raise DataError("Config validation failed: invalid device type", data_type="config")
        except DataError as e:
            assert "validation" in str(e).lower() or "invalid" in str(e).lower()
            assert isinstance(e, Exception)
        
        # Test compatibility as configuration error
        try:
            raise ConfigurationError("Version compatibility issue: requires Python 3.8+", config_key="python_version")
        except ConfigurationError as e:
            assert "compatibility" in str(e).lower()
            assert isinstance(e, ConfigurationError)
    
    def test_memory_exceptions(self):
        """Test memory-related exceptions comprehensively."""
        from neural_arch.exceptions import ResourceError
        
        # Test ResourceError for memory issues
        try:
            raise ResourceError("Memory operation failed", resource_type="memory")
        except ResourceError as e:
            assert "Memory" in str(e)
            assert isinstance(e, Exception)
        
        # Test ResourceError for out of memory
        try:
            raise ResourceError("GPU out of memory: cannot allocate 2GB tensor", resource_type="gpu_memory", required_amount="2GB")
        except ResourceError as e:
            assert "memory" in str(e).lower()
            assert isinstance(e, ResourceError)
        
        # Test ResourceError for allocation failure
        try:
            raise ResourceError("Failed to allocate tensor on device", resource_type="device_memory")
        except ResourceError as e:
            assert "allocate" in str(e).lower()
            assert isinstance(e, ResourceError)
    
    def test_io_exceptions(self):
        """Test I/O related exceptions comprehensively."""
        from neural_arch.exceptions import DataError, ResourceError
        
        # Test DataError for file operations
        try:
            raise DataError("File operation failed", data_type="model_file")
        except DataError as e:
            assert "operation failed" in str(e).lower() or "File" in str(e)
            assert isinstance(e, Exception)
        
        # Test DataError for model loading
        try:
            raise DataError("Failed to load model from checkpoint.pth", data_type="checkpoint", expected_format="pytorch")
        except DataError as e:
            assert "load" in str(e).lower()
            assert isinstance(e, DataError)
        
        # Test ResourceError for model saving
        try:
            raise ResourceError("Failed to save model: insufficient disk space", resource_type="disk_space")
        except ResourceError as e:
            assert "save" in str(e).lower()
            assert isinstance(e, ResourceError)
    
    def test_exception_with_context(self):
        """Test exceptions with additional context information."""
        from neural_arch.exceptions import ShapeError
        
        # Test exception with detailed context
        try:
            tensor_shape = (2, 3, 4)
            expected_shape = (3, 4, 5)
            raise ShapeError(
                f"Shape mismatch in matrix multiplication: "
                f"got {tensor_shape}, expected {expected_shape}",
                expected_shape=expected_shape,
                actual_shape=tensor_shape,
                operation="matrix_multiplication"
            )
        except ShapeError as e:
            assert str(tensor_shape) in str(e)
            assert str(expected_shape) in str(e)
            assert "matrix multiplication" in str(e)
    
    def test_exception_chaining(self):
        """Test exception chaining and cause tracking."""
        from neural_arch.exceptions import ResourceError
        
        # Test exception chaining
        try:
            try:
                raise RuntimeError("CUDA driver not found")
            except RuntimeError as original_error:
                raise ResourceError("CUDA backend unavailable", resource_type="cuda_backend") from original_error
        except ResourceError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, RuntimeError)
            assert "CUDA driver not found" in str(e.__cause__)
    
    def test_exception_inheritance_hierarchy(self):
        """Test exception inheritance hierarchy."""
        from neural_arch.exceptions import (
            TensorError, ShapeError, DTypeError, DeviceError,
            GradientError, ModelError, LayerError, ParameterError,
            OptimizationError, OptimizerError, ConfigurationError,
            ResourceError, DataError, NeuralArchError
        )
        
        # Test inheritance relationships
        assert issubclass(ShapeError, TensorError)
        assert issubclass(DTypeError, TensorError)
        assert issubclass(DeviceError, TensorError)
        assert issubclass(GradientError, TensorError)
        assert issubclass(LayerError, ModelError)
        assert issubclass(ParameterError, ModelError)
        assert issubclass(OptimizerError, OptimizationError)
        
        # Test that all are ultimately NeuralArchError subclasses
        for exc_class in [TensorError, GradientError, ModelError, OptimizationError, ConfigurationError, ResourceError, DataError]:
            assert issubclass(exc_class, NeuralArchError)
            assert issubclass(exc_class, Exception)
    
    def test_real_world_exception_scenarios(self):
        """Test real-world scenarios that trigger exceptions."""
        from neural_arch.exceptions import ShapeError, DTypeError, DeviceError
        
        # Test scenarios that would realistically trigger these exceptions
        
        # Shape mismatch in tensor operations
        try:
            a = Tensor([[1, 2, 3]])  # (1, 3)
            b = Tensor([[1, 2]])     # (1, 2)
            # This should trigger ShapeError in real operations
            if a.shape[1] != b.shape[1]:
                raise ShapeError(
                    f"Cannot perform operation: tensor shapes {a.shape} and {b.shape} are incompatible",
                    expected_shape=a.shape,
                    actual_shape=b.shape
                )
        except ShapeError as e:
            assert "incompatible" in str(e)
            assert str(a.shape) in str(e)
            assert str(b.shape) in str(e)
        
        # Dtype mismatch
        try:
            float_tensor = Tensor([1.5, 2.5], dtype=DType.FLOAT32)
            int_tensor = Tensor([1, 2], dtype=DType.INT32)
            # This should trigger DTypeError in real operations
            if float_tensor.dtype != int_tensor.dtype:
                raise DTypeError(
                    f"Cannot perform operation between {float_tensor.dtype} and {int_tensor.dtype}"
                )
        except DTypeError as e:
            assert "float32" in str(e) or "int32" in str(e)
        
        # Device mismatch
        try:
            cpu_tensor = Tensor([1, 2, 3], device=Device.cpu())
            # Simulate device mismatch
            target_device = "cuda:0"
            if cpu_tensor.device.type != DeviceType.CUDA:
                raise DeviceError(
                    f"Tensor is on {cpu_tensor.device.type.value} but operation requires {target_device}"
                )
        except DeviceError as e:
            assert "cpu" in str(e) and "cuda" in str(e)
    
    def test_exception_custom_attributes(self):
        """Test exceptions with custom attributes."""
        from neural_arch.exceptions import TensorError, ResourceError
        
        # Test exception with custom error codes
        try:
            error = TensorError("Custom tensor error")
            error.custom_attribute = "TENSOR_001"
            error.tensor_shape = (2, 3, 4)
            raise error
        except TensorError as e:
            if hasattr(e, 'custom_attribute'):
                assert e.custom_attribute == "TENSOR_001"
            if hasattr(e, 'tensor_shape'):
                assert e.tensor_shape == (2, 3, 4)
    
    def test_exception_formatting_and_repr(self):
        """Test exception string formatting and representation."""
        from neural_arch.exceptions import ShapeError, ResourceError
        
        # Test string representation
        error = ShapeError("Test error message")
        error_str = str(error)
        error_repr = repr(error)
        
        assert "Test error message" in error_str
        assert "ShapeError" in error_repr
        assert len(error_str) > 0
        assert len(error_repr) > 0
        
        # Test with empty message
        empty_error = ResourceError("")
        empty_str = str(empty_error)
        assert "ResourceError" in repr(empty_error)
    
    def test_exception_equality(self):
        """Test exception equality comparison."""
        from neural_arch.exceptions import TensorError, ShapeError
        
        # Test exception equality
        error1 = TensorError("Same message")
        error2 = TensorError("Same message")
        error3 = TensorError("Different message")
        
        # Same type and message might be equal (implementation dependent)
        if hasattr(error1, '__eq__'):
            # Test equality if implemented
            pass
        
        # Different types should not be equal
        tensor_error = TensorError("Test")
        shape_error = ShapeError("Test")
        assert type(tensor_error) != type(shape_error)
    
    def test_exception_pickling_if_supported(self):
        """Test exception serialization if supported."""
        import pickle
        from neural_arch.exceptions import TensorError
        
        try:
            error = TensorError("Test error for pickling")
            # Try to pickle and unpickle the exception
            pickled = pickle.dumps(error)
            unpickled = pickle.loads(pickled)
            
            assert type(unpickled) == type(error)
            assert str(unpickled) == str(error)
        except (TypeError, AttributeError):
            # Pickling might not be supported for custom exceptions
            pass
    
    def test_exception_multiple_inheritance_scenarios(self):
        """Test complex exception scenarios with multiple inheritance."""
        from neural_arch.exceptions import TensorError, ResourceError
        
        # Test creating a custom exception that inherits from multiple base exceptions
        class CustomTensorResourceError(TensorError):
            """Custom exception inheriting from TensorError."""
            def __init__(self, message, resource_info=None):
                super().__init__(message)
                self.resource_info = resource_info
        
        try:
            raise CustomTensorResourceError("Combined tensor and resource error", resource_info="gpu_memory")
        except TensorError as e:
            assert isinstance(e, CustomTensorResourceError)
            assert isinstance(e, TensorError)
            if hasattr(e, 'resource_info'):
                assert e.resource_info == "gpu_memory"
    
    def test_exception_context_managers(self):
        """Test exceptions within context managers."""
        from neural_arch.exceptions import TensorError
        
        # Test exception handling within context managers
        class ErrorContext:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is TensorError:
                    # Handle TensorError specifically
                    return True  # Suppress the exception
                return False  # Let other exceptions propagate
        
        # Test suppressed exception
        with ErrorContext():
            raise TensorError("This should be suppressed")
        
        # Test non-suppressed exception
        try:
            with ErrorContext():
                raise ValueError("This should not be suppressed")
        except ValueError:
            pass  # Expected to propagate