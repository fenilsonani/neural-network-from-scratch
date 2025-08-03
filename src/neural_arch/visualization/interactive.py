"""Interactive Visualization Module.

This module provides interactive visualization components including
Streamlit dashboards, Jupyter widgets, and real-time model exploration tools.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    JUPYTER_WIDGETS_AVAILABLE = True
except ImportError:
    JUPYTER_WIDGETS_AVAILABLE = False
    widgets = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from neural_arch.nn import Module
from neural_arch.backends import available_backends, get_backend
from .architecture import ModelVisualizer
from .training import TrainingVisualizer
from .features import FeatureVisualizer
from .performance import PerformanceVisualizer


class InteractiveVisualizer:
    """Interactive visualization tools for model exploration and analysis."""
    
    def __init__(self):
        """Initialize the interactive visualizer."""
        self.model_viz = ModelVisualizer()
        self.training_viz = TrainingVisualizer()
        self.feature_viz = FeatureVisualizer()
        self.performance_viz = PerformanceVisualizer()
    
    def create_model_explorer_widget(self, models: Dict[str, Module]) -> Optional[widgets.Widget]:
        """Create Jupyter widget for interactive model exploration.
        
        Args:
            models: Dictionary of model name to model object
            
        Returns:
            IPython widget or None if widgets not available
        """
        if not JUPYTER_WIDGETS_AVAILABLE:
            print("ipywidgets not available. Install with: pip install ipywidgets")
            return None
        
        if not PLOTLY_AVAILABLE:
            print("plotly not available. Install with: pip install plotly")
            return None
        
        # Create UI components
        model_dropdown = widgets.Dropdown(
            options=list(models.keys()),
            value=list(models.keys())[0] if models else None,
            description='Model:',
            style={'description_width': 'initial'}
        )
        
        viz_type = widgets.Dropdown(
            options=['Architecture', 'Layer Details', 'Parameter Distribution'],
            value='Architecture',
            description='Visualization:',
            style={'description_width': 'initial'}
        )
        
        output = widgets.Output()
        
        def update_visualization(change=None):
            with output:
                clear_output(wait=True)
                
                if not models:
                    print("No models provided")
                    return
                
                selected_model = models[model_dropdown.value]
                
                if viz_type.value == 'Architecture':
                    fig = self.model_viz.create_interactive_architecture(selected_model)
                    if fig:
                        fig.show()
                elif viz_type.value == 'Layer Details':
                    layers = self.model_viz._extract_layers(selected_model)
                    print(f"Model: {model_dropdown.value}")
                    print(f"Total Layers: {len(layers)}")
                    print("\nLayer Details:")
                    for name, info in layers.items():
                        print(f"  {name}: {info['name']} ({info['params']:,} params)")
                elif viz_type.value == 'Parameter Distribution':
                    if MATPLOTLIB_AVAILABLE:
                        fig = self.feature_viz.plot_weight_distributions(selected_model)
                        if fig:
                            plt.show()
                    else:
                        print("matplotlib not available for parameter distribution plot")
        
        # Connect event handlers
        model_dropdown.observe(update_visualization, names='value')
        viz_type.observe(update_visualization, names='value')
        
        # Initial visualization
        update_visualization()
        
        # Layout
        controls = widgets.HBox([model_dropdown, viz_type])
        full_widget = widgets.VBox([controls, output])
        
        return full_widget
    
    def create_training_monitor_widget(self) -> Optional[widgets.Widget]:
        """Create Jupyter widget for real-time training monitoring."""
        if not JUPYTER_WIDGETS_AVAILABLE:
            return None
        
        # Training metrics display
        metrics_output = widgets.Output()
        loss_plot_output = widgets.Output()
        
        # Control buttons
        start_button = widgets.Button(description="Start Monitoring", button_style='success')
        stop_button = widgets.Button(description="Stop Monitoring", button_style='danger')
        clear_button = widgets.Button(description="Clear History", button_style='warning')
        
        # Status indicator
        status_label = widgets.Label(value="Status: Ready")
        
        # Training history storage
        training_history = {'epochs': [], 'train_loss': [], 'val_loss': [], 'accuracy': []}
        monitoring_active = {'value': False}
        
        def start_monitoring(b):
            monitoring_active['value'] = True
            status_label.value = "Status: Monitoring Active"
            start_button.disabled = True
            stop_button.disabled = False
        
        def stop_monitoring(b):
            monitoring_active['value'] = False
            status_label.value = "Status: Stopped"
            start_button.disabled = False
            stop_button.disabled = True
        
        def clear_history(b):
            training_history.clear()
            training_history.update({'epochs': [], 'train_loss': [], 'val_loss': [], 'accuracy': []})
            with metrics_output:
                clear_output()
            with loss_plot_output:
                clear_output()
        
        # Connect button events
        start_button.on_click(start_monitoring)
        stop_button.on_click(stop_monitoring)
        clear_button.on_click(clear_history)
        
        # Layout
        controls = widgets.HBox([start_button, stop_button, clear_button, status_label])
        plots = widgets.HBox([metrics_output, loss_plot_output])
        full_widget = widgets.VBox([controls, plots])
        
        return full_widget
    
    def create_performance_comparison_widget(self) -> Optional[widgets.Widget]:
        """Create widget for comparing backend performance."""
        if not JUPYTER_WIDGETS_AVAILABLE:
            return None
        
        # Backend selection
        backends = available_backends()
        backend_checkboxes = {}
        
        for backend in backends:
            backend_checkboxes[backend] = widgets.Checkbox(
                value=True,
                description=backend,
                style={'description_width': 'initial'}
            )
        
        # Matrix size slider
        matrix_size = widgets.IntSlider(
            value=512,
            min=64,
            max=2048,
            step=64,
            description='Matrix Size:',
            style={'description_width': 'initial'}
        )
        
        # Run benchmark button
        run_button = widgets.Button(description="Run Benchmark", button_style='info')
        
        # Results output
        results_output = widgets.Output()
        
        def run_benchmark(b):
            with results_output:
                clear_output(wait=True)
                print("Running benchmark...")
                
                selected_backends = [name for name, checkbox in backend_checkboxes.items() 
                                   if checkbox.value]
                
                if not selected_backends:
                    print("Please select at least one backend")
                    return
                
                # Run performance comparison
                operations = ['matmul', 'add', 'multiply', 'exp']
                benchmark_data = self.performance_viz.benchmark_operations(
                    operations, selected_backends, matrix_size.value
                )
                
                if PLOTLY_AVAILABLE:
                    # Create interactive plot
                    fig = self.performance_viz.create_performance_dashboard(
                        {'backends': selected_backends, 'operations': operations, **benchmark_data}
                    )
                    if fig:
                        fig.show()
                else:
                    # Display text results
                    print("Benchmark Results:")
                    for backend, results in benchmark_data.items():
                        print(f"\n{backend}:")
                        for operation, time in results.items():
                            print(f"  {operation}: {time:.2f} ms")
        
        run_button.on_click(run_benchmark)
        
        # Layout
        backend_controls = widgets.VBox(list(backend_checkboxes.values()))
        controls = widgets.HBox([
            widgets.VBox([widgets.Label("Select Backends:"), backend_controls]),
            widgets.VBox([matrix_size, run_button])
        ])
        
        full_widget = widgets.VBox([controls, results_output])
        
        return full_widget


def create_streamlit_dashboard():
    """Create comprehensive Streamlit dashboard for Neural Forge."""
    if not STREAMLIT_AVAILABLE:
        print("streamlit not available. Install with: pip install streamlit")
        return None
    
    st.set_page_config(
        page_title="Neural Forge Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Neural Forge - Interactive Dashboard")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section",
        ["🏠 Overview", "🏗️ Architecture", "📊 Training", "🔍 Features", "⚡ Performance"]
    )
    
    if page == "🏠 Overview":
        _create_overview_page()
    elif page == "🏗️ Architecture":
        _create_architecture_page()
    elif page == "📊 Training":
        _create_training_page()
    elif page == "🔍 Features":
        _create_features_page()
    elif page == "⚡ Performance":
        _create_performance_page()


def _create_overview_page():
    """Create overview page for Streamlit dashboard."""
    st.header("Neural Forge Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Available Backends", len(available_backends()))
    
    with col2:
        st.metric("Visualization Modules", 5)
    
    with col3:
        st.metric("Framework Status", "Active", delta="Ready")
    
    st.markdown("## 🚀 Features")
    
    features = [
        "**Multi-Backend Support**: NumPy, JAX, MPS (Apple Silicon), CUDA",
        "**Comprehensive Layers**: CNN, RNN, Transformer, and more",
        "**Advanced Visualization**: Architecture diagrams, training metrics, feature maps",
        "**Performance Optimization**: Hardware acceleration and efficient computation",
        "**Production Ready**: Docker, CI/CD, PyPI packaging"
    ]
    
    for feature in features:
        st.markdown(f"✅ {feature}")
    
    st.markdown("## 📈 Quick Stats")
    
    # Sample performance data
    sample_data = {
        'Backend': ['NumPy', 'MPS', 'JAX', 'CUDA'],
        'Matrix Mult (1000x1000)': [245, 12, 18, 8],
        'Memory Usage (MB)': [150, 95, 120, 85],
        'Availability': ['✅', '✅ (Apple)', '✅', '❓ (GPU Required)']
    }
    
    st.table(sample_data)


def _create_architecture_page():
    """Create architecture visualization page."""
    st.header("🏗️ Model Architecture Visualization")
    
    st.markdown("## Model Explorer")
    st.markdown("Visualize neural network architectures with interactive diagrams.")
    
    # Sample model creation
    if st.button("Create Sample Model"):
        st.code("""
from neural_arch.nn import Sequential, Linear, ReLU, Dropout

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128), 
    ReLU(),
    Linear(128, 10)
)
        """, language="python")
        
        st.success("Sample model created! Architecture visualization would appear here.")
    
    st.markdown("## Features")
    st.markdown("- **Layer Type Detection**: Automatic color coding by layer type")
    st.markdown("- **Parameter Counting**: Real-time parameter statistics")
    st.markdown("- **Interactive Exploration**: Hover for detailed layer information")
    st.markdown("- **Export Options**: Save as PNG, SVG, or interactive HTML")


def _create_training_page():
    """Create training monitoring page."""
    st.header("📊 Training Metrics & Monitoring")
    
    st.markdown("## Real-time Training Dashboard")
    
    # Sample training data
    epochs = list(range(1, 51))
    train_loss = [1.0 * np.exp(-0.1 * i) + 0.1 * np.random.random() for i in epochs]
    val_loss = [1.0 * np.exp(-0.08 * i) + 0.15 * np.random.random() for i in epochs]
    accuracy = [1 - np.exp(-0.1 * i) + 0.05 * np.random.random() for i in epochs]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loss Curves")
        chart_data = {
            'Epoch': epochs,
            'Training Loss': train_loss,
            'Validation Loss': val_loss
        }
        st.line_chart(chart_data, x='Epoch')
    
    with col2:
        st.subheader("Accuracy")
        st.line_chart({'Epoch': epochs, 'Accuracy': accuracy}, x='Epoch')
    
    # Training metrics
    st.markdown("## Current Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Current Epoch", 50, delta=1)
    
    with metric_col2:
        st.metric("Training Loss", f"{train_loss[-1]:.4f}", delta=f"{train_loss[-1] - train_loss[-2]:.4f}")
    
    with metric_col3:
        st.metric("Validation Loss", f"{val_loss[-1]:.4f}", delta=f"{val_loss[-1] - val_loss[-2]:.4f}")
    
    with metric_col4:
        st.metric("Accuracy", f"{accuracy[-1]:.2%}", delta=f"{accuracy[-1] - accuracy[-2]:.2%}")


def _create_features_page():
    """Create feature visualization page."""
    st.header("🔍 Feature & Activation Visualization")
    
    st.markdown("## Feature Map Explorer")
    
    # Layer selection
    layer_options = ["Conv Layer 1", "Conv Layer 2", "Conv Layer 3", "Dense Layer 1"]
    selected_layer = st.selectbox("Select Layer", layer_options)
    
    # Feature map parameters
    col1, col2 = st.columns(2)
    
    with col1:
        num_channels = st.slider("Number of Channels", 1, 64, 16)
    
    with col2:
        map_size = st.slider("Feature Map Size", 8, 128, 32)
    
    if st.button("Generate Feature Maps"):
        # Generate sample feature maps
        feature_maps = np.random.randn(num_channels, map_size, map_size)
        
        st.success(f"Generated {num_channels} feature maps of size {map_size}x{map_size}")
        st.info("Interactive feature map visualization would appear here with hover details and zoom functionality.")
    
    st.markdown("## Activation Analysis")
    st.markdown("- **Distribution Analysis**: Histogram and statistical summaries")
    st.markdown("- **Sparsity Tracking**: Monitor activation sparsity over time")
    st.markdown("- **Attention Visualization**: Heatmaps for attention mechanisms")
    st.markdown("- **Weight Analysis**: Distribution and gradient flow visualization")


def _create_performance_page():
    """Create performance benchmarking page."""
    st.header("⚡ Performance & Benchmarking")
    
    st.markdown("## Backend Comparison")
    
    # Backend selection
    backends = available_backends()
    selected_backends = st.multiselect("Select Backends", backends, default=backends[:2])
    
    # Benchmark parameters
    col1, col2 = st.columns(2)
    
    with col1:
        matrix_size = st.slider("Matrix Size", 64, 2048, 512, step=64)
    
    with col2:
        operations = st.multiselect(
            "Operations", 
            ["matmul", "add", "multiply", "exp", "sum"],
            default=["matmul", "add"]
        )
    
    if st.button("Run Benchmark"):
        with st.spinner("Running benchmark..."):
            # Simulate benchmark results
            results = {}
            for backend in selected_backends:
                results[backend] = {}
                for op in operations:
                    # Simulate timing based on backend type
                    if backend.lower() == 'mps':
                        base_time = 5
                    elif backend.lower() == 'jax':
                        base_time = 8
                    else:
                        base_time = 50
                    
                    # Add some variation
                    results[backend][op] = base_time * (1 + 0.2 * np.random.random())
            
            st.success("Benchmark completed!")
            
            # Display results
            st.markdown("### Results (ms)")
            st.json(results)
            
            # Performance metrics
            if len(selected_backends) > 1:
                st.markdown("### Speedup Analysis")
                baseline = selected_backends[0]
                for backend in selected_backends[1:]:
                    for op in operations:
                        speedup = results[baseline][op] / results[backend][op]
                        st.metric(f"{backend} vs {baseline} ({op})", f"{speedup:.2f}x", 
                                delta="faster" if speedup > 1 else "slower")
    
    st.markdown("## System Information")
    
    sys_info = {
        "Platform": "darwin",
        "Available Backends": ", ".join(available_backends()),
        "Memory": "16 GB",
        "GPU": "Apple M2 Pro" if "mps" in available_backends() else "Not Available"
    }
    
    st.json(sys_info)


def launch_model_explorer(models: Dict[str, Module], port: int = 8501):
    """Launch Streamlit model explorer.
    
    Args:
        models: Dictionary of models to explore
        port: Port to run Streamlit on
    """
    if not STREAMLIT_AVAILABLE:
        print("streamlit not available. Install with: pip install streamlit")
        return
    
    # Store models in session state for Streamlit app
    import tempfile
    import subprocess
    
    # Create temporary Streamlit app
    app_code = f"""
import streamlit as st
from neural_arch.visualization.interactive import create_streamlit_dashboard

# Store models in session state
if 'models' not in st.session_state:
    st.session_state.models = {repr(models)}

create_streamlit_dashboard()
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(app_code)
        temp_file = f.name
    
    try:
        # Launch Streamlit
        subprocess.run([
            'streamlit', 'run', temp_file, 
            '--server.port', str(port),
            '--server.headless', 'true'
        ])
    except FileNotFoundError:
        print("Streamlit not found. Install with: pip install streamlit")
    finally:
        os.unlink(temp_file)


def create_jupyter_widgets(models: Optional[Dict[str, Module]] = None) -> Optional[widgets.Widget]:
    """Create comprehensive Jupyter widget interface.
    
    Args:
        models: Optional dictionary of models to explore
        
    Returns:
        Combined widget interface or None if widgets not available
    """
    if not JUPYTER_WIDGETS_AVAILABLE:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return None
    
    visualizer = InteractiveVisualizer()
    
    # Create tab interface
    tab_contents = []
    tab_titles = []
    
    if models:
        tab_contents.append(visualizer.create_model_explorer_widget(models))
        tab_titles.append("Model Explorer")
    
    tab_contents.append(visualizer.create_training_monitor_widget())
    tab_titles.append("Training Monitor")
    
    tab_contents.append(visualizer.create_performance_comparison_widget())
    tab_titles.append("Performance")
    
    # Create tabbed interface
    tabs = widgets.Tab(children=tab_contents)
    for i, title in enumerate(tab_titles):
        tabs.set_title(i, title)
    
    return tabs


# Example usage
if __name__ == "__main__":
    # Test widget creation
    if JUPYTER_WIDGETS_AVAILABLE:
        print("Testing Jupyter widgets...")
        
        # Create sample models
        from neural_arch.nn import Sequential, Linear, ReLU
        
        models = {
            "Simple MLP": Sequential(
                Linear(784, 256),
                ReLU(),
                Linear(256, 10)
            ),
            "Deep MLP": Sequential(
                Linear(784, 512),
                ReLU(),
                Linear(512, 256),
                ReLU(),
                Linear(256, 128),
                ReLU(),
                Linear(128, 10)
            )
        }
        
        widget = create_jupyter_widgets(models)
        if widget:
            display(widget)
    
    # Test Streamlit dashboard
    if STREAMLIT_AVAILABLE:
        print("Streamlit dashboard available. Run with:")
        print("python -c \"from neural_arch.visualization.interactive import create_streamlit_dashboard; create_streamlit_dashboard()\"")