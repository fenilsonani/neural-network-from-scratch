"""Test exceptions module - simplified version."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from neural_arch.exceptions import (
    ConfigurationError,
    DeviceError,
    DTypeError,
    GradientError,
    NeuralArchError,
    OptimizerError,
    ShapeError,
    TensorError,
)


class TestSimpleExceptions:
    """Test simplified exception classes."""

    def test_base_neural_arch_error(self):
        """Test base NeuralArchError."""
        with pytest.raises(NeuralArchError) as exc_info:
            raise NeuralArchError("Test error message")

        assert "Test error message" in str(exc_info.value)
        assert isinstance(exc_info.value, Exception)

    def test_neural_arch_error_with_context(self):
        """Test NeuralArchError with context."""
        error = NeuralArchError("Test error", context={"key": "value"})

        assert "Test error" in str(error)
        assert hasattr(error, "context")
        assert error.context["key"] == "value"

    def test_tensor_error(self):
        """Test TensorError."""
        with pytest.raises(TensorError) as exc_info:
            raise TensorError("Invalid tensor operation")

        assert "Invalid tensor operation" in str(exc_info.value)
        assert isinstance(exc_info.value, NeuralArchError)

    def test_shape_error(self):
        """Test ShapeError."""
        error = ShapeError("Shape mismatch", context={"expected": (2, 3), "actual": (2, 4)})

        assert "Shape mismatch" in str(error)
        assert error.context["expected"] == (2, 3)
        assert error.context["actual"] == (2, 4)

    def test_dtype_error(self):
        """Test DTypeError."""
        with pytest.raises(DTypeError) as exc_info:
            raise DTypeError("Type mismatch", context={"expected": "float32", "actual": "int32"})

        assert "Type mismatch" in str(exc_info.value)
        assert exc_info.value.context["expected"] == "float32"
        assert exc_info.value.context["actual"] == "int32"

    def test_device_error(self):
        """Test DeviceError."""
        error = DeviceError("Device not available", context={"device": "cuda:0"})

        assert "Device not available" in str(error)
        assert error.context["device"] == "cuda:0"

    def test_gradient_error(self):
        """Test GradientError."""
        with pytest.raises(GradientError) as exc_info:
            raise GradientError("Gradient computation failed")

        assert "Gradient computation failed" in str(exc_info.value)
        assert isinstance(exc_info.value, NeuralArchError)

    def test_optimizer_error(self):
        """Test OptimizerError."""
        error = OptimizerError("Invalid learning rate", context={"lr": -0.01})

        assert "Invalid learning rate" in str(error)
        assert error.context["lr"] == -0.01

    def test_configuration_error(self):
        """Test ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Invalid configuration")

        assert "Invalid configuration" in str(exc_info.value)
        assert isinstance(exc_info.value, NeuralArchError)

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from NeuralArchError."""
        exception_classes = [
            TensorError,
            ShapeError,
            DTypeError,
            DeviceError,
            GradientError,
            OptimizerError,
            ConfigurationError,
        ]

        for exc_class in exception_classes:
            assert issubclass(exc_class, NeuralArchError)

            # Test instantiation
            exc = exc_class("Test message")
            assert isinstance(exc, NeuralArchError)
            assert "Test message" in str(exc)

    def test_exception_with_error_code(self):
        """Test exception with error code."""
        error = NeuralArchError("Test error", error_code="TEST_001")

        assert hasattr(error, "error_code")
        assert error.error_code == "TEST_001"

    def test_exception_with_suggestions(self):
        """Test exception with suggestions."""
        suggestions = ["Try using a different device", "Check your input data"]
        error = NeuralArchError("Device error", suggestions=suggestions)

        assert hasattr(error, "suggestions")
        assert error.suggestions == suggestions

    def test_exception_chaining(self):
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise TensorError("Tensor operation failed") from e
        except TensorError as exc:
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, ValueError)
            assert "Original error" in str(exc.__cause__)

    def test_exception_context_preservation(self):
        """Test that exception context is preserved."""
        context = {
            "operation": "matmul",
            "input_shapes": [(2, 3), (4, 5)],
            "expected_output": (2, 5),
        }

        error = TensorError("Matrix multiplication failed", context=context)

        assert error.context == context
        assert error.context["operation"] == "matmul"
        assert len(error.context["input_shapes"]) == 2

    def test_exception_str_formatting(self):
        """Test exception string formatting."""
        error = NeuralArchError("Test error", error_code="ERR_001", context={"param": "value"})

        error_str = str(error)
        assert "ERR_001" in error_str
        assert "Test error" in error_str

    def test_exception_repr(self):
        """Test exception repr."""
        error = TensorError("Test tensor error")

        repr_str = repr(error)
        assert "TensorError" in repr_str
        assert "Test tensor error" in repr_str

    def test_multiple_exceptions(self):
        """Test raising multiple different exceptions."""
        exceptions_to_test = [
            (TensorError, "Tensor error"),
            (ShapeError, "Shape error"),
            (DTypeError, "DType error"),
            (DeviceError, "Device error"),
            (GradientError, "Gradient error"),
            (OptimizerError, "Optimizer error"),
            (ConfigurationError, "Config error"),
        ]

        for exc_class, message in exceptions_to_test:
            with pytest.raises(exc_class) as exc_info:
                raise exc_class(message)

            assert message in str(exc_info.value)
            assert isinstance(exc_info.value, NeuralArchError)

    def test_exception_with_original_exception(self):
        """Test exception with original exception reference."""
        original = ValueError("Original problem")
        error = NeuralArchError("Wrapped error", original_exception=original)

        assert hasattr(error, "original_exception")
        assert error.original_exception is original

    def test_exception_stack_trace(self):
        """Test exception stack trace capture."""
        error = NeuralArchError("Test error")

        # Should have stack trace information
        assert hasattr(error, "stack_trace")
        assert isinstance(error.stack_trace, list)
        assert len(error.stack_trace) > 0
