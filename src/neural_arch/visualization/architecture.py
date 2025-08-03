"""Model Architecture Visualization.

This module provides tools for visualizing neural network architectures,
computational graphs, and layer details with professional diagrams.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

from neural_arch.nn import Module, Sequential, Linear, Conv1d, Conv2d, Conv3d
from neural_arch.nn import RNN, LSTM, GRU, MultiHeadAttention, TransformerBlock
from neural_arch.nn import ReLU, Sigmoid, Tanh, GELU, Softmax, Dropout, LayerNorm


class ModelVisualizer:
    """Professional model architecture visualization with multiple output formats."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "modern"):
        """Initialize the model visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
            style: Visualization style ('modern', 'classic', 'minimal')
        """
        self.figsize = figsize
        self.style = style
        self.colors = self._get_color_scheme()
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8' if style == 'modern' else 'default')
            sns.set_palette("husl")
    
    def _get_color_scheme(self) -> Dict[str, str]:
        """Get color scheme based on style."""
        schemes = {
            'modern': {
                'linear': '#3498db',
                'conv': '#e74c3c', 
                'rnn': '#2ecc71',
                'attention': '#f39c12',
                'activation': '#9b59b6',
                'normalization': '#1abc9c',
                'dropout': '#95a5a6',
                'embedding': '#e67e22',
                'pooling': '#34495e',
            },
            'classic': {
                'linear': '#1f77b4',
                'conv': '#ff7f0e',
                'rnn': '#2ca02c', 
                'attention': '#d62728',
                'activation': '#9467bd',
                'normalization': '#8c564b',
                'dropout': '#e377c2',
                'embedding': '#7f7f7f',
                'pooling': '#bcbd22',
            },
            'minimal': {
                'linear': '#333333',
                'conv': '#666666',
                'rnn': '#999999',
                'attention': '#222222',
                'activation': '#444444',
                'normalization': '#777777',
                'dropout': '#aaaaaa',
                'embedding': '#555555',
                'pooling': '#888888',
            }
        }
        return schemes.get(self.style, schemes['modern'])
    
    def _get_layer_type(self, layer: Module) -> str:
        """Determine the type of a layer for visualization."""
        layer_type = type(layer).__name__.lower()
        
        if any(name in layer_type for name in ['linear', 'dense']):
            return 'linear'
        elif any(name in layer_type for name in ['conv', 'convolution']):
            return 'conv'
        elif any(name in layer_type for name in ['rnn', 'lstm', 'gru']):
            return 'rnn'
        elif any(name in layer_type for name in ['attention', 'transformer']):
            return 'attention'
        elif any(name in layer_type for name in ['relu', 'sigmoid', 'tanh', 'gelu', 'softmax']):
            return 'activation'
        elif any(name in layer_type for name in ['norm', 'batch', 'layer']):
            return 'normalization'
        elif 'dropout' in layer_type:
            return 'dropout'
        elif 'embedding' in layer_type:
            return 'embedding'
        elif any(name in layer_type for name in ['pool', 'pooling']):
            return 'pooling'
        else:
            return 'linear'  # Default
    
    def _get_layer_info(self, layer: Module) -> Dict[str, Any]:
        """Extract detailed information about a layer."""
        info = {
            'name': type(layer).__name__,
            'type': self._get_layer_type(layer),
            'params': 0,
            'input_shape': None,
            'output_shape': None,
            'details': {}
        }
        
        # Count parameters
        try:
            for param in layer.parameters():
                if hasattr(param, 'data'):
                    info['params'] += param.data.size
        except:
            pass
        
        # Extract layer-specific details
        if isinstance(layer, Linear):
            info['details'] = {
                'in_features': getattr(layer, 'in_features', 'Unknown'),
                'out_features': getattr(layer, 'out_features', 'Unknown'),
                'bias': getattr(layer, 'bias', None) is not None,
            }
        elif isinstance(layer, (Conv1d, Conv2d, Conv3d)):
            info['details'] = {
                'in_channels': getattr(layer, 'in_channels', 'Unknown'),
                'out_channels': getattr(layer, 'out_channels', 'Unknown'),
                'kernel_size': getattr(layer, 'kernel_size', 'Unknown'),
                'stride': getattr(layer, 'stride', 1),
                'padding': getattr(layer, 'padding', 0),
            }
        elif isinstance(layer, (RNN, LSTM, GRU)):
            info['details'] = {
                'input_size': getattr(layer, 'input_size', 'Unknown'),
                'hidden_size': getattr(layer, 'hidden_size', 'Unknown'),
                'num_layers': getattr(layer, 'num_layers', 1),
                'bidirectional': getattr(layer, 'bidirectional', False),
            }
        elif isinstance(layer, MultiHeadAttention):
            info['details'] = {
                'd_model': getattr(layer, 'd_model', 'Unknown'),
                'num_heads': getattr(layer, 'num_heads', 'Unknown'),
                'dropout': getattr(layer, 'dropout', 0.0),
            }
        
        return info
    
    def plot_model_architecture(self, model: Module, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Create a comprehensive model architecture diagram."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib seaborn")
            return None
        
        # Extract model structure
        layers = self._extract_layers(model)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(layers) + 1)
        
        # Draw layers
        y_positions = np.linspace(len(layers), 1, len(layers))
        
        for i, (layer_name, layer_info) in enumerate(layers.items()):
            y = y_positions[i]
            
            # Draw layer box
            color = self.colors[layer_info['type']]
            box = FancyBboxPatch(
                (1, y-0.3), 8, 0.6,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                alpha=0.8
            )
            ax.add_patch(box)
            
            # Add layer text
            ax.text(5, y, f"{layer_info['name']}", ha='center', va='center', 
                   fontweight='bold', fontsize=10)
            
            # Add parameter count
            if layer_info['params'] > 0:
                ax.text(9.5, y, f"{layer_info['params']:,}", ha='right', va='center',
                       fontsize=8, style='italic')
            
            # Add details
            details_text = self._format_layer_details(layer_info['details'])
            if details_text:
                ax.text(0.5, y, details_text, ha='left', va='center',
                       fontsize=8, style='italic')
            
            # Draw connections
            if i < len(layers) - 1:
                arrow = ConnectionPatch(
                    (5, y-0.3), (5, y_positions[i+1]+0.3),
                    "data", "data",
                    arrowstyle="->", shrinkA=5, shrinkB=5,
                    mutation_scale=20, fc="black"
                )
                ax.add_patch(arrow)
        
        # Styling
        ax.set_title(f"Neural Network Architecture - {type(model).__name__}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        self._add_legend(ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _extract_layers(self, model: Module) -> Dict[str, Dict]:
        """Extract layers from a model recursively."""
        layers = {}
        
        def extract_recursive(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, Sequential):
                    extract_recursive(child, full_name)
                else:
                    layers[full_name] = self._get_layer_info(child)
        
        if isinstance(model, Sequential):
            for i, layer in enumerate(model._modules_list):
                layers[f"layer_{i}"] = self._get_layer_info(layer)
        else:
            extract_recursive(model)
        
        return layers
    
    def _format_layer_details(self, details: Dict) -> str:
        """Format layer details for display."""
        if not details:
            return ""
        
        formatted = []
        for key, value in details.items():
            if isinstance(value, bool):
                if value:
                    formatted.append(key)
            else:
                formatted.append(f"{key}={value}")
        
        return ", ".join(formatted)
    
    def _add_legend(self, ax):
        """Add a legend showing layer types and colors."""
        legend_elements = []
        for layer_type, color in self.colors.items():
            legend_elements.append(
                patches.Patch(color=color, label=layer_type.capitalize())
            )
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def create_interactive_architecture(self, model: Module) -> Optional[go.Figure]:
        """Create an interactive model architecture using Plotly."""
        if not PLOTLY_AVAILABLE:
            print("plotly not available. Install with: pip install plotly")
            return None
        
        layers = self._extract_layers(model)
        
        # Create network graph
        fig = go.Figure()
        
        y_positions = list(range(len(layers), 0, -1))
        
        # Add nodes (layers)
        for i, (layer_name, layer_info) in enumerate(layers.items()):
            fig.add_trace(go.Scatter(
                x=[1], 
                y=[y_positions[i]],
                mode='markers+text',
                marker=dict(
                    size=50,
                    color=self.colors[layer_info['type']],
                    line=dict(width=2, color='black')
                ),
                text=layer_info['name'],
                textposition="middle center",
                name=layer_name,
                hovertemplate=(
                    f"<b>{layer_info['name']}</b><br>"
                    f"Type: {layer_info['type']}<br>"
                    f"Parameters: {layer_info['params']:,}<br>"
                    f"Details: {self._format_layer_details(layer_info['details'])}"
                    "<extra></extra>"
                )
            ))
        
        # Add edges (connections)
        for i in range(len(layers) - 1):
            fig.add_trace(go.Scatter(
                x=[1, 1],
                y=[y_positions[i], y_positions[i+1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f"Interactive Neural Network Architecture - {type(model).__name__}",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            showlegend=False,
            height=600,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig


def plot_model_architecture(model: Module, style: str = "modern", 
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to plot model architecture.
    
    Args:
        model: Neural network model to visualize
        style: Visualization style ('modern', 'classic', 'minimal')
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure object or None if matplotlib not available
    """
    visualizer = ModelVisualizer(style=style)
    return visualizer.plot_model_architecture(model, save_path)


def plot_computational_graph(model: Module, input_shape: Tuple[int, ...], 
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot computational graph with forward pass flow."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None
    
    # This would require implementing forward pass tracing
    # For now, return basic architecture plot
    return plot_model_architecture(model, save_path=save_path)


def visualize_layer_details(layer: Module, save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Create detailed visualization of a single layer."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None
    
    visualizer = ModelVisualizer()
    layer_info = visualizer._get_layer_info(layer)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create detailed layer visualization
    ax.text(0.5, 0.8, f"Layer: {layer_info['name']}", ha='center', va='center',
           transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    ax.text(0.5, 0.6, f"Type: {layer_info['type'].capitalize()}", ha='center', va='center',
           transform=ax.transAxes, fontsize=12)
    
    ax.text(0.5, 0.4, f"Parameters: {layer_info['params']:,}", ha='center', va='center',
           transform=ax.transAxes, fontsize=12)
    
    details_text = visualizer._format_layer_details(layer_info['details'])
    if details_text:
        ax.text(0.5, 0.2, f"Details: {details_text}", ha='center', va='center',
               transform=ax.transAxes, fontsize=10, style='italic')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_architecture_diagram(model: Module, format: str = "matplotlib") -> Optional[Any]:
    """Create architecture diagram in specified format.
    
    Args:
        model: Neural network model
        format: Output format ('matplotlib', 'plotly', 'both')
        
    Returns:
        Figure object(s) based on format
    """
    visualizer = ModelVisualizer()
    
    if format == "matplotlib":
        return visualizer.plot_model_architecture(model)
    elif format == "plotly":
        return visualizer.create_interactive_architecture(model)
    elif format == "both":
        return {
            'matplotlib': visualizer.plot_model_architecture(model),
            'plotly': visualizer.create_interactive_architecture(model)
        }
    else:
        raise ValueError(f"Unknown format: {format}")


# Example usage and testing
if __name__ == "__main__":
    # Test with a simple model
    from neural_arch.nn import Sequential, Linear, ReLU
    
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10)
    )
    
    # Create visualizations
    visualizer = ModelVisualizer()
    
    if MATPLOTLIB_AVAILABLE:
        fig = visualizer.plot_model_architecture(model)
        if fig:
            plt.show()
    
    if PLOTLY_AVAILABLE:
        fig = visualizer.create_interactive_architecture(model)
        if fig:
            fig.show()