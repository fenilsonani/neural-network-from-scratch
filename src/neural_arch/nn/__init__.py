"""Neural network layers and modules."""

# Import optimized linear layer as default
try:
    from .optimized import OptimizedLinear as Linear

    _USING_OPTIMIZED_LINEAR = True
except ImportError:
    # Fallback to standard linear if optimized not available
    from .linear import Linear

    _USING_OPTIMIZED_LINEAR = False
from .activation import GELU, ReLU, Sigmoid, Softmax, Tanh
from .attention import MultiHeadAttention, SelfAttention
from .container import ModuleList, Sequential
from .dropout import Dropout
from .embedding import Embedding
from .normalization import BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm, LayerNorm, RMSNorm
from .pooling import MaxPool, MeanPool
from .positional import (
    LearnedPositionalEmbedding,
    RoPE,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
    create_rope,
)
from .transformer import TransformerBlock, TransformerDecoderBlock, TransformerEncoder

# Import standard linear as fallback
try:
    from .linear import Linear as StandardLinear

    __all_linear__ = ["Linear", "StandardLinear"]
except ImportError:
    __all_linear__ = ["Linear"]

__all__ = [
    # Core layers
    "Linear",
    "StandardLinear",  # Fallback to standard linear if needed
    "Embedding",
    # Normalization layers
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "RMSNorm",
    "GroupNorm",
    "InstanceNorm",
    # Activation layers
    "ReLU",
    "Softmax",
    "Sigmoid",
    "Tanh",
    "GELU",
    # Attention layers
    "MultiHeadAttention",
    "SelfAttention",
    # Transformer components
    "TransformerBlock",
    "TransformerEncoder",
    "TransformerDecoderBlock",
    # Positional encodings
    "RotaryPositionalEmbedding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEmbedding",
    "RoPE",
    "create_rope",
    # Regularization
    "Dropout",
    # Pooling layers
    "MeanPool",
    "MaxPool",
    # Container modules
    "ModuleList",
    "Sequential",
]
