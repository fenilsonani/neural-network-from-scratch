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
from .module import Module
from .conv import Conv1d, Conv2d, Conv3d
from .conv_transpose import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from .dropout import Dropout
from .embedding import Embedding
from .gru import GRU, GRUCell
from .lstm import LSTM, LSTMCell
from .rnn import RNN, RNNCell
from .spatial_dropout import SpatialDropout1d, SpatialDropout2d, SpatialDropout3d
from .normalization import BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, InstanceNorm, LayerNorm, RMSNorm
from .advanced_pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    GlobalAvgPool1d,
    GlobalAvgPool2d,
    GlobalMaxPool1d,
    GlobalMaxPool2d,
)
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
    # Core components
    "Module",
    # Core layers
    "Linear",
    "StandardLinear",  # Fallback to standard linear if needed
    "Embedding",
    # Convolution layers
    "Conv1d",
    "Conv2d", 
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    # RNN layers
    "RNN",
    "RNNCell",
    "LSTM", 
    "LSTMCell",
    "GRU",
    "GRUCell",
    # Normalization layers
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
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
    "SpatialDropout1d",
    "SpatialDropout2d", 
    "SpatialDropout3d",
    # Pooling layers
    "MeanPool",
    "MaxPool",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d", 
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "GlobalAvgPool1d",
    "GlobalAvgPool2d",
    "GlobalMaxPool1d",
    "GlobalMaxPool2d",
    # Container modules
    "ModuleList",
    "Sequential",
]
