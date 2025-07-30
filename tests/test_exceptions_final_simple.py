"""Comprehensive tests for exceptions module to improve coverage from 79.91% to 95%+.

This file tests exception classes and handle_exception decorator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from neural_arch.exceptions import (
    NeuralArchError, ShapeError, DTypeError, DeviceError,
    OptimizerError, LayerError, NumericalError,
    ConfigurationError, ResourceError, TensorError, GradientError,
    ModelError, ParameterError, OptimizationError, DataError,
    handle_exception
)


class TestExceptionClasses:
    """Test exception class initialization and properties."""
    
    def test_neural_arch_error_basic(self):
        """Test NeuralArchError basic functionality."""
        error = NeuralArchError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_shape_error_with_params(self):
        """Test ShapeError with parameters."""
        error = ShapeError(
            "Shape mismatch",
            expected_shape=(3, 4),
            actual_shape=(5, 6),
            operation="matmul"
        )
        assert "Shape mismatch" in str(error)
        assert isinstance(error, TensorError)
    
    def test_dtype_error_with_params(self):
        """Test DTypeError with parameters."""
        error = DTypeError(
            "Type mismatch",
            expected_dtype="float32",
            actual_dtype="int32"
        )
        assert "Type mismatch" in str(error)
        assert isinstance(error, TensorError)
    
    def test_device_error_with_params(self):
        """Test DeviceError with parameters."""
        error = DeviceError(
            "Device mismatch",
            expected_device="cuda",
            actual_device="cpu"
        )
        assert "Device mismatch" in str(error)
        assert isinstance(error, TensorError)
    
    def test_gradient_error(self):
        """Test GradientError."""
        error = GradientError("Gradient computation failed")
        assert "Gradient computation failed" in str(error)
        assert isinstance(error, TensorError)
    
    def test_numerical_error(self):
        """Test NumericalError."""
        error = NumericalError(
            "NaN detected",
            operation="sqrt",
            invalid_values="[-1, -2]"
        )
        assert "NaN detected" in str(error)
        assert isinstance(error, TensorError)
    
    def test_model_error(self):
        """Test ModelError."""
        error = ModelError("Model error")
        assert "Model error" in str(error)
        assert isinstance(error, NeuralArchError)
    
    def test_layer_error(self):
        """Test LayerError."""
        error = LayerError(
            "Invalid layer config",
            layer_name="Linear",
            parameter_name="in_features",
            invalid_value=0
        )
        assert "Invalid layer config" in str(error)
        assert isinstance(error, ModelError)
    
    def test_parameter_error(self):
        """Test ParameterError."""
        error = ParameterError("Invalid parameter")
        assert "Invalid parameter" in str(error)
        assert isinstance(error, ModelError)
    
    def test_optimization_error(self):
        """Test OptimizationError."""
        error = OptimizationError("Optimization failed")
        assert "Optimization failed" in str(error)
        assert isinstance(error, NeuralArchError)
    
    def test_optimizer_error(self):
        """Test OptimizerError."""
        error = OptimizerError(
            "Invalid learning rate",
            optimizer_name="Adam",
            parameter_name="lr",
            invalid_value=-0.01
        )
        assert "Invalid learning rate" in str(error)
        assert isinstance(error, OptimizationError)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            "Invalid config",
            config_key="batch_size",
            invalid_value=-1,
            expected_type="positive integer"
        )
        assert "Invalid config" in str(error)
        assert isinstance(error, NeuralArchError)
    
    def test_resource_error(self):
        """Test ResourceError."""
        error = ResourceError(
            "Out of memory",
            resource_type="GPU memory",
            requested="10GB",
            available="8GB"
        )
        assert "Out of memory" in str(error)
        assert isinstance(error, NeuralArchError)
    
    def test_data_error(self):
        """Test DataError."""
        error = DataError("Invalid data format")
        assert "Invalid data format" in str(error)
        assert isinstance(error, NeuralArchError)


class TestExceptionDecorator:
    """Test handle_exception decorator."""
    
    def test_handle_exception_shape_error(self):
        """Test handle_exception converts ValueError with 'shape' to ShapeError."""
        @handle_exception
        def func_with_shape_error():
            raise ValueError("Invalid shape for operation")
        
        with pytest.raises(ShapeError) as exc_info:
            func_with_shape_error()
        
        assert "Shape error in func_with_shape_error" in str(exc_info.value)
    
    def test_handle_exception_dtype_error_from_valueerror(self):
        """Test handle_exception converts ValueError with 'dtype' to DTypeError."""
        @handle_exception
        def func_with_dtype_error():
            raise ValueError("Invalid dtype specified")
        
        with pytest.raises(DTypeError) as exc_info:
            func_with_dtype_error()
        
        assert "Data type error in func_with_dtype_error" in str(exc_info.value)
    
    def test_handle_exception_type_error(self):
        """Test handle_exception converts TypeError to DTypeError."""
        @handle_exception
        def func_with_type_error():
            raise TypeError("Wrong type provided")
        
        with pytest.raises(DTypeError) as exc_info:
            func_with_type_error()
        
        assert "Type error in func_with_type_error" in str(exc_info.value)
    
    def test_handle_exception_memory_error(self):
        """Test handle_exception converts MemoryError to ResourceError."""
        @handle_exception
        def func_with_memory_error():
            raise MemoryError("Out of memory")
        
        with pytest.raises(ResourceError) as exc_info:
            func_with_memory_error()
        
        assert "Memory error in func_with_memory_error" in str(exc_info.value)
    
    def test_handle_exception_generic_value_error(self):
        """Test handle_exception converts generic ValueError to NumericalError."""
        @handle_exception
        def func_with_generic_error():
            raise ValueError("Some other error")
        
        with pytest.raises(NumericalError) as exc_info:
            func_with_generic_error()
        
        assert "Numerical error in func_with_generic_error" in str(exc_info.value)
    
    def test_handle_exception_preserves_neural_arch_errors(self):
        """Test handle_exception preserves NeuralArchError subclasses."""
        @handle_exception
        def func_with_shape_error():
            raise ShapeError("Already a ShapeError")
        
        with pytest.raises(ShapeError) as exc_info:
            func_with_shape_error()
        
        # Should preserve the original error
        assert str(exc_info.value) == "Already a ShapeError"
    
    def test_handle_exception_with_args(self):
        """Test handle_exception works with function arguments."""
        @handle_exception
        def func_with_args(x, y, z=None):
            if x < 0:
                raise ValueError("x must be positive")
            return x + y + (z or 0)
        
        # Normal execution
        result = func_with_args(1, 2, z=3)
        assert result == 6
        
        # Error case
        with pytest.raises(NumericalError):
            func_with_args(-1, 2)
    
    def test_handle_exception_preserves_function_metadata(self):
        """Test handle_exception preserves function metadata."""
        @handle_exception
        def documented_function(x):
            """This is a documented function."""
            return x * 2
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."
    
    def test_handle_exception_value_error_with_type(self):
        """Test handle_exception with 'type' in ValueError message."""
        @handle_exception
        def func_with_type_in_message():
            raise ValueError("Invalid type for parameter")
        
        with pytest.raises(DTypeError) as exc_info:
            func_with_type_in_message()
        
        assert "Data type error" in str(exc_info.value)


class TestExceptionInheritance:
    """Test exception inheritance relationships."""
    
    def test_tensor_error_hierarchy(self):
        """Test TensorError subclasses."""
        tensor_errors = [
            ShapeError("test"),
            DTypeError("test"),
            DeviceError("test"),
            GradientError("test"),
            NumericalError("test")
        ]
        
        for error in tensor_errors:
            assert isinstance(error, TensorError)
            assert isinstance(error, NeuralArchError)
            assert isinstance(error, Exception)
    
    def test_model_error_hierarchy(self):
        """Test ModelError subclasses."""
        model_errors = [
            LayerError("test"),
            ParameterError("test")
        ]
        
        for error in model_errors:
            assert isinstance(error, ModelError)
            assert isinstance(error, NeuralArchError)
            assert isinstance(error, Exception)
    
    def test_optimization_error_hierarchy(self):
        """Test OptimizationError subclasses."""
        opt_errors = [
            OptimizerError("test"),
            OptimizationError("test")
        ]
        
        for error in opt_errors:
            assert isinstance(error, NeuralArchError)
            assert isinstance(error, Exception)