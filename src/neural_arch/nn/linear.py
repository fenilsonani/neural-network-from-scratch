"""Linear (fully connected) layer implementation."""

import math
from typing import Optional

import numpy as np

from ..core import Module, Parameter, Tensor
from ..exceptions import LayerError, handle_exception
from ..functional import add, matmul


class Linear(Module):
    """Fully connected (linear) layer with enterprise-grade features.

    This layer performs a linear transformation: y = xW + b

    Features:
    - Multiple weight initialization schemes
    - Bias term (optional)
    - Gradient tracking and backpropagation
    - Memory-efficient implementation
    - Comprehensive error handling
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
        name: Optional[str] = None,
    ) -> None:
        """Initialize linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
            weight_init: Weight initialization scheme
            bias_init: Bias initialization scheme
            name: Optional layer name for debugging

        Raises:
            LayerError: If layer parameters are invalid
        """
        super().__init__()

        # Validate parameters
        if in_features <= 0:
            raise LayerError(
                f"in_features must be positive, got {in_features}",
                layer_name=name,
                layer_type="Linear",
            )

        if out_features <= 0:
            raise LayerError(
                f"out_features must be positive, got {out_features}",
                layer_name=name,
                layer_type="Linear",
            )

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.name = name or f"Linear({in_features}, {out_features})"

        # Initialize weight parameter
        weight_data = self._initialize_weights(weight_init, in_features, out_features)
        self.weight = Parameter(weight_data, name=f"{self.name}.weight")

        # Initialize bias parameter (optional)
        if bias:
            bias_data = self._initialize_bias(bias_init, out_features)
            self.bias = Parameter(bias_data, name=f"{self.name}.bias")
        else:
            self.bias = None

    def _initialize_weights(self, init_scheme: str, fan_in: int, fan_out: int) -> np.ndarray:
        """Initialize weight matrix using specified scheme.

        Args:
            init_scheme: Initialization scheme name
            fan_in: Number of input features
            fan_out: Number of output features

        Returns:
            Initialized weight matrix

        Raises:
            LayerError: If initialization scheme is unknown
        """
        if init_scheme == "xavier_uniform":
            # Xavier/Glorot uniform initialization
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        elif init_scheme == "xavier_normal":
            # Xavier/Glorot normal initialization
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

        elif init_scheme == "he_uniform":
            # He uniform initialization (good for ReLU)
            limit = math.sqrt(6.0 / fan_in)
            return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        elif init_scheme == "he_normal":
            # He normal initialization (good for ReLU)
            std = math.sqrt(2.0 / fan_in)
            return np.random.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

        elif init_scheme == "lecun_uniform":
            # LeCun uniform initialization
            limit = math.sqrt(3.0 / fan_in)
            return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        elif init_scheme == "lecun_normal":
            # LeCun normal initialization
            std = math.sqrt(1.0 / fan_in)
            return np.random.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

        elif init_scheme == "uniform":
            # Simple uniform initialization
            return np.random.uniform(-0.1, 0.1, (fan_in, fan_out)).astype(np.float32)

        elif init_scheme == "normal":
            # Simple normal initialization
            return np.random.normal(0.0, 0.1, (fan_in, fan_out)).astype(np.float32)

        elif init_scheme == "zeros":
            # Zero initialization (rarely used for weights)
            return np.zeros((fan_in, fan_out), dtype=np.float32)

        elif init_scheme == "ones":
            # Ones initialization (rarely used for weights)
            return np.ones((fan_in, fan_out), dtype=np.float32)

        else:
            raise LayerError(
                f"Unknown weight initialization scheme: {init_scheme}",
                layer_name=self.name,
                layer_type="Linear",
            )

    def _initialize_bias(self, init_scheme: str, size: int) -> np.ndarray:
        """Initialize bias vector using specified scheme.

        Args:
            init_scheme: Initialization scheme name
            size: Bias vector size

        Returns:
            Initialized bias vector

        Raises:
            LayerError: If initialization scheme is unknown
        """
        if init_scheme == "zeros":
            return np.zeros(size, dtype=np.float32)

        elif init_scheme == "ones":
            return np.ones(size, dtype=np.float32)

        elif init_scheme == "uniform":
            return np.random.uniform(-0.1, 0.1, size).astype(np.float32)

        elif init_scheme == "normal":
            return np.random.normal(0.0, 0.1, size).astype(np.float32)

        else:
            raise LayerError(
                f"Unknown bias initialization scheme: {init_scheme}",
                layer_name=self.name,
                layer_type="Linear",
            )

    @handle_exception
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through linear layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)

        Raises:
            LayerError: If input shape is incompatible
        """
        # Validate input shape
        if x.shape[-1] != self.in_features:
            raise LayerError(
                f"Input feature dimension mismatch: expected {self.in_features}, got {x.shape[-1]}",
                layer_name=self.name,
                layer_type="Linear",
            )

        # Linear transformation: y = xW + b
        output = matmul(x, self.weight)

        if self.bias is not None:
            output = add(output, self.bias)

        return output

    def reset_parameters(
        self, weight_init: Optional[str] = None, bias_init: Optional[str] = None
    ) -> None:
        """Reset layer parameters with new initialization.

        Args:
            weight_init: New weight initialization scheme (keeps current if None)
            bias_init: New bias initialization scheme (keeps current if None)
        """
        if weight_init is not None:
            weight_data = self._initialize_weights(weight_init, self.in_features, self.out_features)
            self.weight.data = weight_data

        if bias_init is not None and self.bias is not None:
            bias_data = self._initialize_bias(bias_init, self.out_features)
            self.bias.data = bias_data

    def extra_repr(self) -> str:
        """Return extra string representation for debugging."""
        bias_str = f", bias={self.use_bias}"
        return f"in_features={self.in_features}, out_features={self.out_features}{bias_str}"

    def __repr__(self) -> str:
        """String representation of the linear layer."""
        return f"{self.__class__.__name__}({self.extra_repr()})"

    @property
    def weight_norm(self) -> float:
        """Compute the Frobenius norm of the weight matrix."""
        return float(np.linalg.norm(self.weight.data))

    @property
    def bias_norm(self) -> float:
        """Compute the L2 norm of the bias vector."""
        if self.bias is None:
            return 0.0
        return float(np.linalg.norm(self.bias.data))

    def get_weight_stats(self) -> dict:
        """Get statistics about layer weights for monitoring.

        Returns:
            Dictionary with weight statistics
        """
        weight_data = self.weight.data
        stats = {
            "weight_mean": float(np.mean(weight_data)),
            "weight_std": float(np.std(weight_data)),
            "weight_min": float(np.min(weight_data)),
            "weight_max": float(np.max(weight_data)),
            "weight_norm": self.weight_norm,
        }

        if self.bias is not None:
            bias_data = self.bias.data
            stats.update(
                {
                    "bias_mean": float(np.mean(bias_data)),
                    "bias_std": float(np.std(bias_data)),
                    "bias_min": float(np.min(bias_data)),
                    "bias_max": float(np.max(bias_data)),
                    "bias_norm": self.bias_norm,
                }
            )

        return stats
