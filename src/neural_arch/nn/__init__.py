"""Neural network layers and modules."""

# Import optimized linear layer as default
try:
    from .optimized import OptimizedLinear as Linear
    _USING_OPTIMIZED_LINEAR = True
except ImportError:
    # Fallback to standard linear if optimized not available
    from .linear import Linear
    _USING_OPTIMIZED_LINEAR = False
from .embedding import Embedding
from .normalization import LayerNorm, BatchNorm1d, BatchNorm2d, RMSNorm, GroupNorm, InstanceNorm
from .activation import ReLU, Softmax, Sigmoid, Tanh, GELU
from .attention import MultiHeadAttention, SelfAttention
from .transformer import TransformerBlock, TransformerEncoder, TransformerDecoderBlock
from .positional import RotaryPositionalEmbedding, SinusoidalPositionalEncoding, LearnedPositionalEmbedding, RoPE, create_rope
from .dropout import Dropout
from .pooling import MeanPool, MaxPool
from .container import ModuleList, Sequential

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