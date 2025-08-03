"""Neural Forge Visualization Module.

This module provides comprehensive visualization tools for:
- Model architectures and computational graphs
- Training metrics and loss curves
- Feature maps and activation patterns
- Performance benchmarks and comparisons
- Interactive model exploration
"""

from .architecture import (
    ModelVisualizer,
    plot_model_architecture,
    plot_computational_graph,
    visualize_layer_details,
    create_architecture_diagram,
)

from .training import (
    TrainingVisualizer,
    plot_training_curves,
    plot_loss_history,
    plot_metrics_dashboard,
    create_training_report,
)

from .features import (
    FeatureVisualizer,
    plot_feature_maps,
    plot_activations,
    visualize_attention_weights,
    plot_weight_distributions,
)

from .performance import (
    PerformanceVisualizer,
    plot_benchmark_results,
    compare_backend_performance,
    plot_memory_usage,
    create_performance_dashboard,
)

from .interactive import (
    InteractiveVisualizer,
    create_streamlit_dashboard,
    launch_model_explorer,
    create_jupyter_widgets,
)

__all__ = [
    # Architecture visualization
    "ModelVisualizer",
    "plot_model_architecture", 
    "plot_computational_graph",
    "visualize_layer_details",
    "create_architecture_diagram",
    
    # Training visualization
    "TrainingVisualizer",
    "plot_training_curves",
    "plot_loss_history", 
    "plot_metrics_dashboard",
    "create_training_report",
    
    # Feature visualization
    "FeatureVisualizer",
    "plot_feature_maps",
    "plot_activations",
    "visualize_attention_weights",
    "plot_weight_distributions",
    
    # Performance visualization
    "PerformanceVisualizer", 
    "plot_benchmark_results",
    "compare_backend_performance",
    "plot_memory_usage",
    "create_performance_dashboard",
    
    # Interactive visualization
    "InteractiveVisualizer",
    "create_streamlit_dashboard",
    "launch_model_explorer",
    "create_jupyter_widgets",
]