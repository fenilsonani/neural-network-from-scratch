"""Neural Forge Model Compression Module.

This module provides state-of-the-art model compression techniques including:
- Structured and unstructured pruning
- Post-training and quantization-aware training
- Knowledge distillation frameworks
- Sparse tensor optimizations
- Model deployment optimizations

Key Features:
- Multiple pruning strategies (magnitude, gradient-based, structured)
- INT8/INT16 quantization with calibration
- Teacher-student knowledge distillation
- Automatic compression pipeline
- Performance benchmarking and analysis
"""

from .pruning import (
    PruningStrategy,
    MagnitudePruner,
    GradientPruner,
    StructuredPruner,
    GlobalMagnitudePruner,
    LayerWisePruner,
    prune_model,
    get_sparsity_info,
)

from .quantization import (
    QuantizationConfig,
    PostTrainingQuantizer,
    QuantizationAwareTraining,
    DynamicQuantizer,
    StaticQuantizer,
    quantize_model,
    calibrate_quantization,
)

from .distillation import (
    KnowledgeDistillationConfig,
    DistillationLoss,
    TeacherStudentTrainer,
    FeatureDistillation,
    AttentionDistillation,
    ResponseDistillation,
    distill_model,
)

from .optimization import (
    ModelOptimizer,
    CompressionPipeline,
    AutoCompressionConfig,
    SparseLinear,
    QuantizedLinear,
    optimize_for_inference,
    benchmark_compression,
)

from .utils import (
    calculate_model_size,
    calculate_flops,
    measure_inference_time,
    analyze_compression_tradeoffs,
    export_compressed_model,
    load_compressed_model,
)

__all__ = [
    # Pruning
    "PruningStrategy",
    "MagnitudePruner", 
    "GradientPruner",
    "StructuredPruner",
    "GlobalMagnitudePruner",
    "LayerWisePruner",
    "prune_model",
    "get_sparsity_info",
    
    # Quantization
    "QuantizationConfig",
    "PostTrainingQuantizer",
    "QuantizationAwareTraining", 
    "DynamicQuantizer",
    "StaticQuantizer",
    "quantize_model",
    "calibrate_quantization",
    
    # Knowledge Distillation
    "KnowledgeDistillationConfig",
    "DistillationLoss",
    "TeacherStudentTrainer",
    "FeatureDistillation",
    "AttentionDistillation", 
    "ResponseDistillation",
    "distill_model",
    
    # Optimization
    "ModelOptimizer",
    "CompressionPipeline",
    "AutoCompressionConfig", 
    "SparseLinear",
    "QuantizedLinear",
    "optimize_for_inference",
    "benchmark_compression",
    
    # Utils
    "calculate_model_size",
    "calculate_flops",
    "measure_inference_time",
    "analyze_compression_tradeoffs",
    "export_compressed_model",
    "load_compressed_model",
]