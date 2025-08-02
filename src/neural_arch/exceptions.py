"""Enterprise-grade exception hierarchy for neural architecture."""

import traceback
from typing import Any, List, Optional


class NeuralArchError(Exception):
    """Base exception for all neural architecture errors.

    This provides enterprise-grade error handling with:
    - Structured error information
    - Context preservation
    - Debugging support
    - Error recovery suggestions
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict] = None,
        suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        """Initialize neural architecture error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            context: Additional context information
            suggestions: Suggested solutions
            original_exception: Original exception if this is a wrapper
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.suggestions = suggestions or []
        self.original_exception = original_exception

        # Capture stack trace for debugging
        self.stack_trace = traceback.format_stack()

    def __str__(self) -> str:
        """Format error for display."""
        error_str = f"[{self.error_code}] {self.message}"

        if self.context:
            error_str += f"\nContext: {self.context}"

        if self.suggestions:
            error_str += f"\nSuggestions:\n" + "\n".join(f"  - {s}" for s in self.suggestions)

        if self.original_exception:
            error_str += f"\nCaused by: {self.original_exception}"

        return error_str

    def to_dict(self) -> dict:
        """Convert error to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "suggestions": self.suggestions,
            "original_exception": str(self.original_exception) if self.original_exception else None,
        }


class TensorError(NeuralArchError):
    """Base class for tensor-related errors."""

    pass


class ShapeError(TensorError):
    """Raised when tensor shapes are incompatible for an operation."""

    def __init__(
        self,
        message: str,
        expected_shape: Optional[tuple] = None,
        actual_shape: Optional[tuple] = None,
        operation: Optional[str] = None,
    ) -> None:
        context = {}
        if expected_shape is not None:
            context["expected_shape"] = expected_shape
        if actual_shape is not None:
            context["actual_shape"] = actual_shape
        if operation is not None:
            context["operation"] = operation

        suggestions = [
            "Check tensor dimensions before the operation",
            "Use reshape() or view() to adjust tensor shape",
            "Verify broadcasting rules for the operation",
        ]

        super().__init__(
            message, error_code="SHAPE_MISMATCH", context=context, suggestions=suggestions
        )


class DTypeError(TensorError):
    """Raised when tensor data types are incompatible."""

    def __init__(
        self, message: str, expected_dtype: Optional[str] = None, actual_dtype: Optional[str] = None
    ) -> None:
        context = {}
        if expected_dtype is not None:
            context["expected_dtype"] = expected_dtype
        if actual_dtype is not None:
            context["actual_dtype"] = actual_dtype

        suggestions = [
            "Use tensor.to(dtype) to convert data type",
            "Check that all inputs have compatible data types",
            "Consider using automatic type promotion",
        ]

        super().__init__(
            message, error_code="DTYPE_MISMATCH", context=context, suggestions=suggestions
        )


class DeviceError(TensorError):
    """Raised when tensors are on incompatible devices."""

    def __init__(
        self,
        message: str,
        expected_device: Optional[str] = None,
        actual_device: Optional[str] = None,
    ) -> None:
        context = {}
        if expected_device is not None:
            context["expected_device"] = expected_device
        if actual_device is not None:
            context["actual_device"] = actual_device

        suggestions = [
            "Move tensors to the same device using tensor.to(device)",
            "Check device availability before operations",
            "Use get_default_device() for consistent device placement",
        ]

        super().__init__(
            message, error_code="DEVICE_MISMATCH", context=context, suggestions=suggestions
        )


class GradientError(TensorError):
    """Raised when gradient computation fails."""

    def __init__(
        self, message: str, tensor_name: Optional[str] = None, operation: Optional[str] = None
    ) -> None:
        context = {}
        if tensor_name is not None:
            context["tensor_name"] = tensor_name
        if operation is not None:
            context["operation"] = operation

        suggestions = [
            "Check if requires_grad=True for input tensors",
            "Verify computational graph is properly constructed",
            "Use tensor.detach() if gradient computation is not needed",
            "Check for in-place operations that break gradient flow",
        ]

        super().__init__(
            message, error_code="GRADIENT_ERROR", context=context, suggestions=suggestions
        )


class NumericalError(TensorError):
    """Raised when numerical computation fails or produces invalid results."""

    def __init__(
        self, message: str, operation: Optional[str] = None, invalid_values: Optional[str] = None
    ) -> None:
        context = {}
        if operation is not None:
            context["operation"] = operation
        if invalid_values is not None:
            context["invalid_values"] = invalid_values

        suggestions = [
            "Check input values for NaN or infinity",
            "Use gradient clipping to prevent numerical instability",
            "Consider using a smaller learning rate",
            "Add numerical stability checks (epsilon values)",
        ]

        super().__init__(
            message, error_code="NUMERICAL_ERROR", context=context, suggestions=suggestions
        )


class ModelError(NeuralArchError):
    """Base class for neural network model errors."""

    pass


class LayerError(ModelError):
    """Raised when neural network layer operations fail."""

    def __init__(
        self, message: str, layer_name: Optional[str] = None, layer_type: Optional[str] = None
    ) -> None:
        context = {}
        if layer_name is not None:
            context["layer_name"] = layer_name
        if layer_type is not None:
            context["layer_type"] = layer_type

        suggestions = [
            "Check layer input dimensions and parameters",
            "Verify layer is properly initialized",
            "Ensure forward() method is correctly implemented",
        ]

        super().__init__(
            message, error_code="LAYER_ERROR", context=context, suggestions=suggestions
        )


class ParameterError(ModelError):
    """Raised when model parameter operations fail."""

    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        expected_shape: Optional[tuple] = None,
        actual_shape: Optional[tuple] = None,
    ) -> None:
        context = {}
        if parameter_name is not None:
            context["parameter_name"] = parameter_name
        if expected_shape is not None:
            context["expected_shape"] = expected_shape
        if actual_shape is not None:
            context["actual_shape"] = actual_shape

        suggestions = [
            "Check parameter initialization",
            "Verify state_dict compatibility",
            "Ensure parameter shapes match model architecture",
        ]

        super().__init__(
            message, error_code="PARAMETER_ERROR", context=context, suggestions=suggestions
        )


class OptimizationError(NeuralArchError):
    """Base class for optimization-related errors."""

    pass


class OptimizerError(OptimizationError):
    """Raised when optimizer operations fail."""

    def __init__(
        self,
        message: str,
        optimizer_type: Optional[str] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        context = {}
        if optimizer_type is not None:
            context["optimizer_type"] = optimizer_type
        if learning_rate is not None:
            context["learning_rate"] = learning_rate

        suggestions = [
            "Check learning rate and other hyperparameters",
            "Verify all parameters have gradients",
            "Ensure optimizer.zero_grad() is called",
            "Check for parameter updates in training loop",
        ]

        super().__init__(
            message, error_code="OPTIMIZER_ERROR", context=context, suggestions=suggestions
        )


class ConfigurationError(NeuralArchError):
    """Raised when configuration is invalid or inconsistent."""

    def __init__(
        self, message: str, config_key: Optional[str] = None, config_value: Optional[Any] = None
    ) -> None:
        context = {}
        if config_key is not None:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = str(config_value)

        suggestions = [
            "Check configuration file syntax",
            "Verify all required configuration keys are present",
            "Validate configuration values against expected types",
            "Use default configuration as a starting point",
        ]

        super().__init__(
            message, error_code="CONFIG_ERROR", context=context, suggestions=suggestions
        )


class ResourceError(NeuralArchError):
    """Raised when system resources are insufficient or unavailable."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        required_amount: Optional[str] = None,
        available_amount: Optional[str] = None,
    ) -> None:
        context = {}
        if resource_type is not None:
            context["resource_type"] = resource_type
        if required_amount is not None:
            context["required_amount"] = required_amount
        if available_amount is not None:
            context["available_amount"] = available_amount

        suggestions = [
            "Reduce batch size or model size",
            "Use gradient checkpointing for memory efficiency",
            "Consider using a different device or cloud instance",
            "Monitor resource usage and optimize accordingly",
        ]

        super().__init__(
            message, error_code="RESOURCE_ERROR", context=context, suggestions=suggestions
        )


class DataError(NeuralArchError):
    """Raised when data processing or validation fails."""

    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
    ) -> None:
        context = {}
        if data_type is not None:
            context["data_type"] = data_type
        if expected_format is not None:
            context["expected_format"] = expected_format
        if actual_format is not None:
            context["actual_format"] = actual_format

        suggestions = [
            "Validate data format and structure",
            "Check for missing or corrupted data",
            "Verify data preprocessing steps",
            "Use data validation utilities",
        ]

        super().__init__(message, error_code="DATA_ERROR", context=context, suggestions=suggestions)


def handle_exception(func):
    """Decorator to provide enterprise-grade exception handling.

    This decorator:
    - Catches and wraps common exceptions
    - Adds context information
    - Provides recovery suggestions
    - Enables structured error logging
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except ValueError as e:
            if "shape" in str(e).lower():
                raise ShapeError(
                    f"Shape error in {func.__name__}: {e}", operation=func.__name__
                ) from e
            elif "dtype" in str(e).lower() or "type" in str(e).lower():
                raise DTypeError(
                    f"Data type error in {func.__name__}: {e}",
                ) from e
            else:
                raise NumericalError(
                    f"Numerical error in {func.__name__}: {e}", operation=func.__name__
                ) from e

        except TypeError as e:
            raise DTypeError(
                f"Type error in {func.__name__}: {e}",
            ) from e

        except MemoryError as e:
            raise ResourceError(
                f"Memory error in {func.__name__}: {e}", resource_type="memory"
            ) from e

        except Exception as e:
            # Catch-all for unexpected errors
            raise NeuralArchError(
                f"Unexpected error in {func.__name__}: {e}",
                context={"function": func.__name__},
                original_exception=e,
            ) from e

    return wrapper
