"""Test custom exceptions."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
)


class TestExceptions:
    """Test all custom exceptions."""

    def test_base_exception(self):
        """Test base NeuralArchError."""
        with pytest.raises(NeuralArchError) as exc_info:
            raise NeuralArchError("Test error")

        assert "Test error" in str(exc_info.value)
        assert hasattr(exc_info.value, "context")

    def test_base_exception_with_context(self):
        """Test base exception with context."""
        context = {"param": "value", "code": 123}
        with pytest.raises(NeuralArchError) as exc_info:
            raise NeuralArchError("Test error", context=context)

        assert "Test error" in str(exc_info.value)
        assert exc_info.value.context == context

    def test_tensor_error(self):
        """Test TensorError."""
        with pytest.raises(TensorError) as exc_info:
            raise TensorError(
                "Invalid tensor operation", context={"shape": (2, 3), "dtype": "float32"}
            )

        assert "Invalid tensor operation" in str(exc_info.value)
        assert exc_info.value.context["shape"] == (2, 3)

    def test_shape_error(self):
        """Test ShapeError."""
        with pytest.raises(ShapeError) as exc_info:
            raise ShapeError("Shape mismatch", context={"expected": (10, 20), "actual": (10, 30)})

        error_str = str(exc_info.value)
        assert "Shape mismatch" in error_str
        assert exc_info.value.context["expected"] == (10, 20)
        assert exc_info.value.context["actual"] == (10, 30)

    def test_dtype_error(self):
        """Test DTypeError."""
        with pytest.raises(DTypeError) as exc_info:
            raise DTypeError("Type mismatch", context={"expected": "float32", "actual": "int32"})

        error_str = str(exc_info.value)
        assert "Type mismatch" in error_str
        assert exc_info.value.context["expected"] == "float32"
        assert exc_info.value.context["actual"] == "int32"

    def test_device_error(self):
        """Test DeviceError."""
        with pytest.raises(DeviceError) as exc_info:
            raise DeviceError(
                "Device not available", context={"device": "cuda:0", "available": ["cpu", "mps"]}
            )

        error_str = str(exc_info.value)
        assert "Device not available" in error_str
        assert exc_info.value.context["device"] == "cuda:0"
        assert exc_info.value.context["available"] == ["cpu", "mps"]

    def test_gradient_error(self):
        """Test GradientError."""
        with pytest.raises(GradientError) as exc_info:
            raise GradientError(
                "Gradient computation failed",
                context={"operation": "matmul", "reason": "NaN values detected"},
            )

        error_str = str(exc_info.value)
        assert "Gradient computation failed" in error_str
        assert exc_info.value.context["operation"] == "matmul"
        assert exc_info.value.context["reason"] == "NaN values detected"

    def test_optimizer_error(self):
        """Test OptimizerError."""
        with pytest.raises(OptimizerError) as exc_info:
            raise OptimizerError(
                "Invalid learning rate",
                context={"optimizer": "Adam", "parameter": "lr", "value": -0.01},
            )

        error_str = str(exc_info.value)
        assert "Invalid learning rate" in error_str
        assert exc_info.value.context["optimizer"] == "Adam"
        assert exc_info.value.context["parameter"] == "lr"
        assert exc_info.value.context["value"] == -0.01

    def test_numerical_error(self):
        """Test NumericalError."""
        with pytest.raises(NumericalError) as exc_info:
            raise NumericalError(
                "Numerical instability detected",
                context={"operation": "softmax", "values": "[inf, -inf, nan]"},
            )

        error_str = str(exc_info.value)
        assert "Numerical instability" in error_str
        assert exc_info.value.context["operation"] == "softmax"
        assert exc_info.value.context["values"] == "[inf, -inf, nan]"

    def test_configuration_error(self):
        """Test ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(
                "Invalid configuration",
                context={"key": "batch_size", "value": "invalid", "expected_type": "int"},
            )

        error_str = str(exc_info.value)
        assert "Invalid configuration" in error_str
        assert exc_info.value.context["key"] == "batch_size"
        assert exc_info.value.context["value"] == "invalid"
        assert exc_info.value.context["expected_type"] == "int"

    def test_resource_error(self):
        """Test ResourceError."""
        with pytest.raises(ResourceError) as exc_info:
            raise ResourceError(
                "Out of memory", context={"required": "8GB", "available": "4GB", "device": "cuda:0"}
            )

        error_str = str(exc_info.value)
        assert "Out of memory" in error_str
        assert exc_info.value.context["required"] == "8GB"
        assert exc_info.value.context["available"] == "4GB"
        assert exc_info.value.context["device"] == "cuda:0"

    def test_data_error(self):
        """Test DataError."""
        with pytest.raises(DataError) as exc_info:
            raise DataError("Invalid data format", context={"expected": "tensor", "actual": "list"})

        error_str = str(exc_info.value)
        assert "Invalid data format" in error_str
        assert exc_info.value.context["expected"] == "tensor"
        assert exc_info.value.context["actual"] == "list"

    def test_layer_error(self):
        """Test LayerError."""
        with pytest.raises(LayerError) as exc_info:
            raise LayerError(
                "Layer initialization failed", context={"layer_type": "Linear", "input_size": 128}
            )

        error_str = str(exc_info.value)
        assert "Layer initialization failed" in error_str
        assert exc_info.value.context["layer_type"] == "Linear"
        assert exc_info.value.context["input_size"] == 128

    def test_model_error(self):
        """Test ModelError."""
        with pytest.raises(ModelError) as exc_info:
            raise ModelError(
                "Model compilation failed", context={"model_type": "Transformer", "num_layers": 12}
            )

        error_str = str(exc_info.value)
        assert "Model compilation failed" in error_str
        assert exc_info.value.context["model_type"] == "Transformer"
        assert exc_info.value.context["num_layers"] == 12

    def test_optimization_error(self):
        """Test OptimizationError."""
        with pytest.raises(OptimizationError) as exc_info:
            raise OptimizationError(
                "Optimization failed", context={"algorithm": "Adam", "step": 1000}
            )

        error_str = str(exc_info.value)
        assert "Optimization failed" in error_str
        assert exc_info.value.context["algorithm"] == "Adam"
        assert exc_info.value.context["step"] == 1000

    def test_parameter_error(self):
        """Test ParameterError."""
        with pytest.raises(ParameterError) as exc_info:
            raise ParameterError(
                "Invalid parameter", context={"param_name": "learning_rate", "value": -0.1}
            )

        error_str = str(exc_info.value)
        assert "Invalid parameter" in error_str
        assert exc_info.value.context["param_name"] == "learning_rate"
        assert exc_info.value.context["value"] == -0.1

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
            NumericalError,
            ResourceError,
            DataError,
            LayerError,
            ModelError,
            OptimizationError,
            ParameterError,
        ]

        for exc_class in exception_classes:
            assert issubclass(exc_class, NeuralArchError)

    def test_exception_with_cause(self):
        """Test exception chaining."""
        try:
            try:
                # Simulate an error
                1 / 0
            except ZeroDivisionError as e:
                raise NumericalError("Division failed", context={"operation": "divide"}) from e
        except NumericalError as exc:
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, ZeroDivisionError)
