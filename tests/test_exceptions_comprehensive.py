"""Comprehensive tests for exceptions module to maximize coverage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.exceptions import (
    ConfigurationError,
    DataError,
    DeviceError,
    DTypeError,
    GradientError,
    LayerError,
    ModelError,
    NeuralArchError,
    NumericalError,
    OptimizationError,
    OptimizerError,
    ParameterError,
    ResourceError,
    ShapeError,
    TensorError,
    handle_exception,
)


class TestNeuralArchError:
    """Test the base NeuralArchError class comprehensively."""

    def test_basic_error_creation(self):
        """Test basic error creation and attributes."""
        error = NeuralArchError("Test message")

        assert str(error) == "[NEURALARCHERROR] Test message"
        assert error.message == "Test message"
        assert error.error_code == "NEURALARCHERROR"
        assert error.context == {}
        assert error.suggestions == []
        assert error.original_exception is None
        assert error.stack_trace is not None

    def test_full_error_creation(self):
        """Test error creation with all parameters."""
        original_exc = ValueError("Original error")
        context = {"key": "value", "number": 42}
        suggestions = ["Try this", "Or that"]

        error = NeuralArchError(
            "Complex error",
            error_code="CUSTOM_ERROR",
            context=context,
            suggestions=suggestions,
            original_exception=original_exc,
        )

        assert error.message == "Complex error"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.context == context
        assert error.suggestions == suggestions
        assert error.original_exception == original_exc

    def test_error_string_formatting(self):
        """Test comprehensive error string formatting."""
        original_exc = RuntimeError("Runtime issue")
        error = NeuralArchError(
            "Main error",
            context={"operation": "test", "value": 123},
            suggestions=["Fix the operation", "Check the value"],
            original_exception=original_exc,
        )

        error_str = str(error)
        assert "[NEURALARCHERROR] Main error" in error_str
        assert "Context: {'operation': 'test', 'value': 123}" in error_str
        assert "Suggestions:" in error_str
        assert "  - Fix the operation" in error_str
        assert "  - Check the value" in error_str
        assert "Caused by: Runtime issue" in error_str

    def test_to_dict_method(self):
        """Test error serialization to dictionary."""
        original_exc = KeyError("Missing key")
        error = NeuralArchError(
            "Dict test",
            error_code="DICT_ERROR",
            context={"test": True},
            suggestions=["Add the key"],
            original_exception=original_exc,
        )

        error_dict = error.to_dict()
        expected = {
            "error_type": "NeuralArchError",
            "error_code": "DICT_ERROR",
            "message": "Dict test",
            "context": {"test": True},
            "suggestions": ["Add the key"],
            "original_exception": "'Missing key'",  # String representation includes quotes
        }

        assert error_dict == expected

    def test_to_dict_without_original_exception(self):
        """Test to_dict method without original exception."""
        error = NeuralArchError("Simple error")
        error_dict = error.to_dict()

        assert error_dict["original_exception"] is None
        assert error_dict["error_type"] == "NeuralArchError"


class TestShapeError:
    """Test ShapeError class comprehensively."""

    def test_basic_shape_error(self):
        """Test basic shape error creation."""
        error = ShapeError("Shape mismatch")

        assert isinstance(error, TensorError)
        assert isinstance(error, NeuralArchError)
        assert error.error_code == "SHAPE_MISMATCH"
        assert "Check tensor dimensions" in error.suggestions[0]

    def test_shape_error_with_details(self):
        """Test shape error with shape details."""
        error = ShapeError(
            "Matrix multiplication error",
            expected_shape=(4, 3),
            actual_shape=(4, 5),
            operation="matmul",
        )

        assert error.context["expected_shape"] == (4, 3)
        assert error.context["actual_shape"] == (4, 5)
        assert error.context["operation"] == "matmul"
        assert "reshape()" in str(error)
        assert "broadcasting" in str(error)


class TestDTypeError:
    """Test DTypeError class comprehensively."""

    def test_basic_dtype_error(self):
        """Test basic dtype error creation."""
        error = DTypeError("Data type mismatch")

        assert isinstance(error, TensorError)
        assert error.error_code == "DTYPE_MISMATCH"
        assert "tensor.to(dtype)" in str(error)

    def test_dtype_error_with_details(self):
        """Test dtype error with type details."""
        error = DTypeError("Cannot convert types", expected_dtype="float32", actual_dtype="int64")

        assert error.context["expected_dtype"] == "float32"
        assert error.context["actual_dtype"] == "int64"
        assert "type promotion" in str(error)


class TestDeviceError:
    """Test DeviceError class comprehensively."""

    def test_basic_device_error(self):
        """Test basic device error creation."""
        error = DeviceError("Device mismatch")

        assert isinstance(error, TensorError)
        assert error.error_code == "DEVICE_MISMATCH"
        assert "tensor.to(device)" in str(error)

    def test_device_error_with_details(self):
        """Test device error with device details."""
        error = DeviceError(
            "Tensors on different devices", expected_device="cuda:0", actual_device="cpu"
        )

        assert error.context["expected_device"] == "cuda:0"
        assert error.context["actual_device"] == "cpu"
        assert "get_default_device" in str(error)


class TestHandleExceptionDecorator:
    """Test the handle_exception decorator comprehensively."""

    def test_successful_function_execution(self):
        """Test decorator doesn't interfere with successful execution."""

        @handle_exception
        def successful_function(x, y):
            return x + y

        result = successful_function(5, 3)
        assert result == 8

    def test_value_error_with_shape(self):
        """Test ValueError with 'shape' keyword gets converted to ShapeError."""

        @handle_exception
        def shape_error_function():
            raise ValueError("Shape mismatch: (3, 4) vs (3, 5)")

        with pytest.raises(ShapeError) as exc_info:
            shape_error_function()

        assert "Shape error in shape_error_function" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "shape_error_function"

    def test_value_error_with_dtype(self):
        """Test ValueError with 'dtype' keyword gets converted to DTypeError."""

        @handle_exception
        def dtype_error_function():
            raise ValueError("Data type mismatch: expected float32")

        with pytest.raises(DTypeError) as exc_info:
            dtype_error_function()

        assert "Data type error in dtype_error_function" in str(exc_info.value)

    def test_generic_value_error(self):
        """Test generic ValueError gets converted to NumericalError."""

        @handle_exception
        def numerical_error_function():
            raise ValueError("Division by zero")

        with pytest.raises(NumericalError) as exc_info:
            numerical_error_function()

        assert "Numerical error in numerical_error_function" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "numerical_error_function"

    def test_type_error_conversion(self):
        """Test TypeError gets converted to DTypeError."""

        @handle_exception
        def type_error_function():
            raise TypeError("unsupported operand type(s)")

        with pytest.raises(DTypeError) as exc_info:
            type_error_function()

        assert "Type error in type_error_function" in str(exc_info.value)

    def test_memory_error_conversion(self):
        """Test MemoryError gets converted to ResourceError."""

        @handle_exception
        def memory_error_function():
            raise MemoryError("Unable to allocate memory")

        with pytest.raises(ResourceError) as exc_info:
            memory_error_function()

        assert "Memory error in memory_error_function" in str(exc_info.value)
        assert exc_info.value.context["resource_type"] == "memory"

    def test_generic_exception_conversion(self):
        """Test generic Exception gets converted to NeuralArchError."""

        @handle_exception
        def generic_error_function():
            raise RuntimeError("Something went wrong")

        with pytest.raises(NeuralArchError) as exc_info:
            generic_error_function()

        assert "Unexpected error in generic_error_function" in str(exc_info.value)
        assert exc_info.value.context["function"] == "generic_error_function"
        assert isinstance(exc_info.value.original_exception, RuntimeError)


class TestAllErrorTypes:
    """Test all error types for basic functionality."""

    def test_all_error_classes_creation(self):
        """Test that all error classes can be created."""
        error_classes = [
            (GradientError, "Gradient failed"),
            (NumericalError, "Numerical issue"),
            (LayerError, "Layer failed"),
            (ParameterError, "Parameter error"),
            (OptimizerError, "Optimizer failed"),
            (ConfigurationError, "Config error"),
            (ResourceError, "Resource error"),
            (DataError, "Data error"),
        ]

        for error_class, message in error_classes:
            error = error_class(message)
            assert isinstance(error, NeuralArchError)
            assert error.message == message
            assert error.error_code is not None
            assert isinstance(error.suggestions, list)
            assert len(error.suggestions) > 0
