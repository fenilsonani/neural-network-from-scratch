"""Neural network optimization and acceleration utilities.

This package provides enterprise-grade optimization tools including:
- Operator fusion for 2-5x performance improvements
- Mixed precision training with automatic loss scaling
- JIT compilation and kernel optimization
- Memory optimization and gradient checkpointing
"""

from .fusion import (
    FusionEngine,
    fuse_conv_bn_activation,
    fuse_layernorm_linear,
    fuse_linear_activation,
    get_fusion_engine,
)
from .mixed_precision import (
    AutomaticMixedPrecision,
    GradScaler,
    MixedPrecisionManager,
    get_mixed_precision_manager,
)

__all__ = [
    # Operator fusion
    "FusionEngine",
    "get_fusion_engine",
    "fuse_linear_activation",
    "fuse_conv_bn_activation",
    "fuse_layernorm_linear",
    # Mixed precision training
    "MixedPrecisionManager",
    "AutomaticMixedPrecision",
    "GradScaler",
    "get_mixed_precision_manager",
]
