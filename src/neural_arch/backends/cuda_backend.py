"""NVIDIA CUDA backend using CuPy with custom kernel integration."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .backend import Backend, register_backend

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import custom CUDA kernels
try:
    from .cuda_kernels import (
        cuda_flash_attention,
        cuda_fused_linear_gelu,
        cuda_gelu,
        cuda_layernorm,
        get_cuda_kernel_manager,
    )

    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False


class CudaBackend(Backend):
    """CuPy backend for NVIDIA GPU computation with custom kernel acceleration."""

    def __init__(self):
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install it with: pip install cupy-cuda11x")

        # Initialize custom kernel manager if available
        self._kernel_manager = None
        if CUDA_KERNELS_AVAILABLE:
            try:
                self._kernel_manager = get_cuda_kernel_manager()
                if self._kernel_manager.is_available():
                    print(f"✅ Custom CUDA kernels initialized")
                else:
                    print(f"⚠️ Custom CUDA kernels unavailable")
            except Exception as e:
                print(f"⚠️ Failed to initialize custom CUDA kernels: {e}")
                self._kernel_manager = None

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def is_available(self) -> bool:
        if not CUPY_AVAILABLE:
            return False
        try:
            # Test if CUDA is available
            cp.cuda.Device(0).compute_capability
            return True
        except:
            return False

    @property
    def supports_gradients(self) -> bool:
        return False  # CuPy doesn't have built-in autograd

    # Dtype attributes
    @property
    def float32(self):
        return cp.float32

    @property
    def float64(self):
        return cp.float64

    @property
    def int32(self):
        return cp.int32

    @property
    def int64(self):
        return cp.int64

    @property
    def bool(self):
        return cp.bool_

    # Array creation
    def array(self, data: Any, dtype: Optional[Any] = None) -> Any:
        return cp.array(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        return cp.zeros(shape, dtype=dtype or cp.float32)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        return cp.ones(shape, dtype=dtype or cp.float32)

    def full(self, shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None) -> Any:
        return cp.full(shape, fill_value, dtype=dtype or cp.float32)

    def arange(
        self, start: float, stop: float, step: float = 1.0, dtype: Optional[Any] = None
    ) -> Any:
        return cp.arange(start, stop, step, dtype=dtype or cp.float32)

    def random_normal(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Optional[Any] = None,
    ) -> Any:
        result = cp.random.normal(mean, std, shape)
        if dtype:
            result = result.astype(dtype)
        return result.astype(cp.float32)

    def random_uniform(
        self,
        shape: Tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        dtype: Optional[Any] = None,
    ) -> Any:
        result = cp.random.uniform(low, high, shape)
        if dtype:
            result = result.astype(dtype)
        return result.astype(cp.float32)

    # Shape operations
    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        return x.reshape(shape)

    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        return cp.transpose(x, axes)

    def squeeze(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        return cp.squeeze(x, axis)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return cp.expand_dims(x, axis)

    # Math operations
    def add(self, x: Any, y: Any) -> Any:
        return cp.add(x, y)

    def subtract(self, x: Any, y: Any) -> Any:
        return cp.subtract(x, y)

    def multiply(self, x: Any, y: Any) -> Any:
        return cp.multiply(x, y)

    def divide(self, x: Any, y: Any) -> Any:
        return cp.divide(x, y)

    def power(self, x: Any, y: Any) -> Any:
        return cp.power(x, y)

    def matmul(self, x: Any, y: Any) -> Any:
        return cp.matmul(x, y)

    def dot(self, x: Any, y: Any) -> Any:
        return cp.dot(x, y)

    # Reduction operations
    def sum(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        return cp.sum(x, axis=axis, keepdims=keepdims)

    def mean(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        return cp.mean(x, axis=axis, keepdims=keepdims)

    def max(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        return cp.max(x, axis=axis, keepdims=keepdims)

    def min(
        self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Any:
        return cp.min(x, axis=axis, keepdims=keepdims)

    def argmax(self, x: Any, axis: Optional[int] = None) -> Any:
        return cp.argmax(x, axis=axis)

    def argmin(self, x: Any, axis: Optional[int] = None) -> Any:
        return cp.argmin(x, axis=axis)

    # Activation functions
    def exp(self, x: Any) -> Any:
        return cp.exp(x)

    def log(self, x: Any) -> Any:
        return cp.log(x)

    def sqrt(self, x: Any) -> Any:
        return cp.sqrt(x)

    def abs(self, x: Any) -> Any:
        return cp.abs(x)

    def sign(self, x: Any) -> Any:
        return cp.sign(x)

    def clip(self, x: Any, min_val: float, max_val: float) -> Any:
        return cp.clip(x, min_val, max_val)

    # Comparison operations
    def equal(self, x: Any, y: Any) -> Any:
        return cp.equal(x, y)

    def not_equal(self, x: Any, y: Any) -> Any:
        return cp.not_equal(x, y)

    def less(self, x: Any, y: Any) -> Any:
        return cp.less(x, y)

    def less_equal(self, x: Any, y: Any) -> Any:
        return cp.less_equal(x, y)

    def greater(self, x: Any, y: Any) -> Any:
        return cp.greater(x, y)

    def greater_equal(self, x: Any, y: Any) -> Any:
        return cp.greater_equal(x, y)

    # Array manipulation
    def concatenate(self, arrays: List[Any], axis: int = 0) -> Any:
        return cp.concatenate(arrays, axis=axis)

    def stack(self, arrays: List[Any], axis: int = 0) -> Any:
        return cp.stack(arrays, axis=axis)

    def split(self, x: Any, indices_or_sections: Union[int, List[int]], axis: int = 0) -> List[Any]:
        return cp.split(x, indices_or_sections, axis=axis)

    # Type conversion
    def astype(self, x: Any, dtype: Any) -> Any:
        return x.astype(dtype)

    def to_numpy(self, x: Any) -> np.ndarray:
        return cp.asnumpy(x)

    def from_numpy(self, x: np.ndarray, dtype: Optional[Any] = None) -> Any:
        result = cp.asarray(x)
        if dtype:
            result = result.astype(dtype)
        return result

    # Device operations
    def to_device(self, x: Any, device: str) -> Any:
        if device == "cpu":
            # Return numpy array for CPU
            return self.to_numpy(x)
        elif device.startswith("cuda"):
            # Parse device index
            if device == "cuda":
                device_id = 0
            else:
                try:
                    device_id = int(device.split(":")[1])
                except:
                    device_id = 0

            # Move to specified GPU
            with cp.cuda.Device(device_id):
                return cp.asarray(x)
        else:
            raise ValueError(f"CuPy backend supports cpu/cuda:N, got {device}")

    def device_of(self, x: Any) -> str:
        if isinstance(x, np.ndarray):
            return "cpu"
        return f"cuda:{x.device.id}"

    # Utility functions
    def is_array(self, x: Any) -> bool:
        return isinstance(x, (cp.ndarray, np.ndarray))

    def shape(self, x: Any) -> Tuple[int, ...]:
        return x.shape

    def size(self, x: Any) -> int:
        return x.size

    def dtype(self, x: Any) -> Any:
        return x.dtype

    # Advanced operations
    def einsum(self, equation: str, *operands) -> Any:
        return cp.einsum(equation, *operands)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return cp.where(condition, x, y)

    def unique(self, x: Any, return_counts: bool = False) -> Union[Any, Tuple[Any, Any]]:
        if return_counts:
            return cp.unique(x, return_counts=True)
        return cp.unique(x)

    # Custom kernel methods for ultra-high performance
    def gelu(self, x: Any) -> Any:
        """Ultra-fast GELU activation using custom CUDA kernel."""
        if self._kernel_manager and self._kernel_manager.is_available():
            try:
                if isinstance(x, cp.ndarray):
                    return self._kernel_manager.gelu_forward(x)
                else:
                    # Fallback to standard implementation
                    pass
            except Exception:
                # Fallback to standard implementation
                pass

        # Standard GELU implementation
        sqrt_2_over_pi = cp.sqrt(2.0 / cp.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + cp.tanh(inner))

    def fused_linear_gelu(self, input_gpu: Any, weight_gpu: Any, bias_gpu: Any) -> Any:
        """Fused linear + GELU operation using custom CUDA kernel."""
        if self._kernel_manager and self._kernel_manager.is_available():
            try:
                if all(isinstance(arr, cp.ndarray) for arr in [input_gpu, weight_gpu, bias_gpu]):
                    return self._kernel_manager.fused_linear_gelu(input_gpu, weight_gpu, bias_gpu)
            except Exception:
                # Fallback to standard implementation
                pass

        # Standard implementation: linear followed by GELU
        linear_out = cp.dot(input_gpu, weight_gpu.T) + bias_gpu
        return self.gelu(linear_out)

    def layernorm(self, input_gpu: Any, weight_gpu: Any, bias_gpu: Any, eps: float = 1e-5) -> Any:
        """Layer normalization using custom CUDA kernel."""
        if self._kernel_manager and self._kernel_manager.is_available():
            try:
                if all(isinstance(arr, cp.ndarray) for arr in [input_gpu, weight_gpu, bias_gpu]):
                    output, _, _ = self._kernel_manager.layernorm_forward(
                        input_gpu, weight_gpu, bias_gpu, eps
                    )
                    return output
            except Exception:
                # Fallback to standard implementation
                pass

        # Standard layer normalization
        mean = cp.mean(input_gpu, axis=-1, keepdims=True)
        var = cp.var(input_gpu, axis=-1, keepdims=True)
        normalized = (input_gpu - mean) / cp.sqrt(var + eps)
        return normalized * weight_gpu + bias_gpu

    def flash_attention(
        self, q_gpu: Any, k_gpu: Any, v_gpu: Any, scale: float, block_size: int = 64
    ) -> Any:
        """Flash Attention for memory-efficient attention computation."""
        if self._kernel_manager and self._kernel_manager.is_available():
            try:
                if all(isinstance(arr, cp.ndarray) for arr in [q_gpu, k_gpu, v_gpu]):
                    return self._kernel_manager.flash_attention(
                        q_gpu, k_gpu, v_gpu, scale, block_size
                    )
            except Exception:
                # Fallback to standard implementation
                pass

        # Standard attention implementation (memory-intensive)
        # Q @ K^T
        scores = cp.matmul(q_gpu, cp.transpose(k_gpu, (0, 1, 3, 2))) * scale
        # Softmax
        attention_weights = cp.exp(scores - cp.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / cp.sum(attention_weights, axis=-1, keepdims=True)
        # Attention @ V
        return cp.matmul(attention_weights, v_gpu)

    def benchmark_kernel(self, kernel_name: str, *args, num_runs: int = 100) -> float:
        """Benchmark a specific custom kernel."""
        if self._kernel_manager and self._kernel_manager.is_available():
            return self._kernel_manager.benchmark_kernel(kernel_name, *args, num_runs=num_runs)
        return float("inf")

    @property
    def has_custom_kernels(self) -> bool:
        """Check if custom CUDA kernels are available."""
        return self._kernel_manager is not None and self._kernel_manager.is_available()


# Register the CUDA backend
if CUPY_AVAILABLE:
    register_backend("cuda", CudaBackend)
