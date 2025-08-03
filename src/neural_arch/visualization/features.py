"""Feature Visualization Module.

This module provides tools for visualizing feature maps, activations,
attention weights, and weight distributions in neural networks.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
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

from neural_arch.core import Tensor
from neural_arch.nn import Module


class FeatureVisualizer:
    """Advanced feature and activation visualization."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "modern"):
        """Initialize the feature visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
            style: Visualization style ('modern', 'classic', 'scientific')
        """
        self.figsize = figsize
        self.style = style
        
        if MATPLOTLIB_AVAILABLE:
            if style == "modern":
                plt.style.use('seaborn-v0_8-darkgrid')
                sns.set_palette("viridis")
            elif style == "scientific":
                plt.style.use('seaborn-v0_8-whitegrid')
                sns.set_palette("rocket")
    
    def plot_feature_maps(self, feature_maps: Union[Tensor, np.ndarray], 
                         layer_name: str = "Layer", max_maps: int = 16,
                         save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Visualize feature maps from convolutional layers.
        
        Args:
            feature_maps: Feature maps tensor (C, H, W) or (N, C, H, W)
            layer_name: Name of the layer for the title
            max_maps: Maximum number of feature maps to display
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib seaborn")
            return None
        
        # Convert to numpy if needed
        if isinstance(feature_maps, Tensor):
            data = feature_maps.data
        else:
            data = feature_maps
        
        # Handle different input shapes
        if data.ndim == 4:  # (N, C, H, W) - take first batch
            data = data[0]
        elif data.ndim == 3:  # (C, H, W)
            pass
        else:
            print(f"Unsupported feature map shape: {data.shape}")
            return None
        
        num_channels = min(data.shape[0], max_maps)
        
        # Calculate grid layout
        grid_size = int(np.ceil(np.sqrt(num_channels)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=self.figsize)
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(num_channels):
            ax = axes[i]
            feature_map = data[i]
            
            # Normalize for better visualization
            if feature_map.max() != feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            
            im = ax.imshow(feature_map, cmap='viridis', aspect='auto')
            ax.set_title(f'Channel {i}', fontsize=10)
            ax.axis('off')
            
            # Add colorbar for the first few maps
            if i < 4:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Feature Maps - {layer_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_activations(self, activations: Union[Tensor, np.ndarray], 
                        layer_name: str = "Layer", activation_type: str = "histogram",
                        save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Visualize activation distributions and patterns.
        
        Args:
            activations: Activation tensor
            layer_name: Name of the layer
            activation_type: Type of visualization ('histogram', 'distribution', 'heatmap')
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Convert to numpy if needed
        if isinstance(activations, Tensor):
            data = activations.data.flatten()
        else:
            data = activations.flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        # Histogram
        axes[0].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Activation Histogram')
        axes[0].set_xlabel('Activation Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(data, vert=True)
        axes[1].set_title('Activation Distribution')
        axes[1].set_ylabel('Activation Value')
        axes[1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_data = np.sort(data)
        y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[2].plot(sorted_data, y_values, linewidth=2)
        axes[2].set_title('Cumulative Distribution')
        axes[2].set_xlabel('Activation Value')
        axes[2].set_ylabel('Cumulative Probability')
        axes[2].grid(True, alpha=0.3)
        
        # Statistics summary
        axes[3].axis('off')
        stats_text = f"""
        Statistics for {layer_name}:
        
        Mean: {np.mean(data):.4f}
        Std:  {np.std(data):.4f}
        Min:  {np.min(data):.4f}
        Max:  {np.max(data):.4f}
        
        Percentiles:
        25%:  {np.percentile(data, 25):.4f}
        50%:  {np.percentile(data, 50):.4f}
        75%:  {np.percentile(data, 75):.4f}
        
        Sparsity: {np.mean(data == 0):.2%}
        """
        
        axes[3].text(0.1, 0.9, stats_text, transform=axes[3].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Activation Analysis - {layer_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_weights(self, attention_weights: Union[Tensor, np.ndarray],
                                  head_idx: Optional[int] = None, max_seq_len: int = 50,
                                  save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Visualize attention weight matrices.
        
        Args:
            attention_weights: Attention weights (num_heads, seq_len, seq_len) or (seq_len, seq_len)
            head_idx: Specific attention head to visualize (None for all)
            max_seq_len: Maximum sequence length to display
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Convert to numpy if needed
        if isinstance(attention_weights, Tensor):
            data = attention_weights.data
        else:
            data = attention_weights
        
        # Handle different shapes
        if data.ndim == 3:  # (num_heads, seq_len, seq_len)
            num_heads = data.shape[0]
            if head_idx is not None:
                data = data[head_idx:head_idx+1]
                num_heads = 1
        elif data.ndim == 2:  # (seq_len, seq_len)
            data = data[np.newaxis, :, :]  # Add head dimension
            num_heads = 1
        else:
            print(f"Unsupported attention weight shape: {data.shape}")
            return None
        
        # Limit sequence length for visualization
        seq_len = min(data.shape[1], max_seq_len)
        data = data[:, :seq_len, :seq_len]
        
        # Calculate grid layout
        if num_heads == 1:
            fig, ax = plt.subplots(figsize=self.figsize)
            axes = [ax]
        else:
            grid_size = int(np.ceil(np.sqrt(num_heads)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=self.figsize)
            if grid_size == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
        
        for i in range(num_heads):
            ax = axes[i] if num_heads > 1 else axes[0]
            
            # Create heatmap
            im = ax.imshow(data[i], cmap='Blues', aspect='auto', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', rotation=270, labelpad=15)
            
            ax.set_title(f'Attention Head {i}' if num_heads > 1 else 'Attention Weights')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add grid
            ax.set_xticks(np.arange(seq_len))
            ax.set_yticks(np.arange(seq_len))
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        if num_heads > 1:
            for i in range(num_heads, len(axes)):
                axes[i].set_visible(False)
        
        plt.suptitle('Attention Weight Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_weight_distributions(self, model: Module, layer_types: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot weight distributions across different layers.
        
        Args:
            model: Neural network model
            layer_types: Types of layers to include (None for all)
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Collect weights from model
        layer_weights = {}
        
        def collect_weights(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if this layer type should be included
                if layer_types is not None:
                    if not any(layer_type.lower() in type(child).__name__.lower() 
                             for layer_type in layer_types):
                        continue
                
                # Collect weights
                if hasattr(child, 'weight') and child.weight is not None:
                    layer_weights[f"{full_name}_weight"] = child.weight.data.flatten()
                
                if hasattr(child, 'bias') and child.bias is not None:
                    layer_weights[f"{full_name}_bias"] = child.bias.data.flatten()
                
                # Recurse into child modules
                collect_weights(child, full_name)
        
        collect_weights(model)
        
        if not layer_weights:
            print("No weights found in the model")
            return None
        
        # Create subplots
        n_layers = len(layer_weights)
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
        if n_layers == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (layer_name, weights) in enumerate(layer_weights.items()):
            ax = axes[i]
            
            # Plot histogram
            ax.hist(weights, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(layer_name, fontsize=10)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(weights)
            std_val = np.std(weights)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Weight Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_feature_map(self, feature_maps: Union[Tensor, np.ndarray],
                                     layer_name: str = "Layer") -> Optional[go.Figure]:
        """Create interactive feature map visualization using Plotly.
        
        Args:
            feature_maps: Feature maps tensor
            layer_name: Name of the layer
            
        Returns:
            Plotly Figure object or None
        """
        if not PLOTLY_AVAILABLE:
            print("plotly not available. Install with: pip install plotly")
            return None
        
        # Convert to numpy if needed
        if isinstance(feature_maps, Tensor):
            data = feature_maps.data
        else:
            data = feature_maps
        
        # Handle different input shapes
        if data.ndim == 4:  # (N, C, H, W) - take first batch
            data = data[0]
        elif data.ndim == 3:  # (C, H, W)
            pass
        else:
            print(f"Unsupported feature map shape: {data.shape}")
            return None
        
        num_channels = data.shape[0]
        
        # Create subplot grid
        grid_size = int(np.ceil(np.sqrt(min(num_channels, 16))))
        
        fig = make_subplots(
            rows=grid_size, cols=grid_size,
            subplot_titles=[f'Channel {i}' for i in range(min(num_channels, 16))],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        for i in range(min(num_channels, 16)):
            row = i // grid_size + 1
            col = i % grid_size + 1
            
            feature_map = data[i]
            
            # Normalize for better visualization
            if feature_map.max() != feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            
            fig.add_trace(
                go.Heatmap(
                    z=feature_map,
                    colorscale='Viridis',
                    showscale=i == 0,  # Only show colorbar for first subplot
                    hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f'Interactive Feature Maps - {layer_name}',
            height=800,
        )
        
        # Hide axis labels for cleaner look
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig


def plot_feature_maps(feature_maps: Union[Tensor, np.ndarray], layer_name: str = "Layer",
                     max_maps: int = 16, save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to plot feature maps."""
    visualizer = FeatureVisualizer()
    return visualizer.plot_feature_maps(feature_maps, layer_name, max_maps, save_path)


def plot_activations(activations: Union[Tensor, np.ndarray], layer_name: str = "Layer",
                    save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to plot activation distributions."""
    visualizer = FeatureVisualizer()
    return visualizer.plot_activations(activations, layer_name, save_path=save_path)


def visualize_attention_weights(attention_weights: Union[Tensor, np.ndarray],
                              head_idx: Optional[int] = None,
                              save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to visualize attention weights."""
    visualizer = FeatureVisualizer()
    return visualizer.visualize_attention_weights(attention_weights, head_idx, save_path=save_path)


def plot_weight_distributions(model: Module, layer_types: Optional[List[str]] = None,
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to plot weight distributions."""
    visualizer = FeatureVisualizer()
    return visualizer.plot_weight_distributions(model, layer_types, save_path)


# Example usage
if __name__ == "__main__":
    # Test with synthetic data
    if MATPLOTLIB_AVAILABLE:
        # Create synthetic feature maps
        feature_maps = np.random.randn(16, 32, 32)
        
        visualizer = FeatureVisualizer()
        
        # Test feature map visualization
        fig1 = visualizer.plot_feature_maps(feature_maps, "Conv Layer 1")
        if fig1:
            plt.show()
        
        # Test activation visualization
        activations = np.random.randn(1000)
        fig2 = visualizer.plot_activations(activations, "ReLU Layer")
        if fig2:
            plt.show()
        
        # Test attention visualization
        attention = np.random.rand(8, 20, 20)  # 8 heads, 20x20 attention matrix
        fig3 = visualizer.visualize_attention_weights(attention)
        if fig3:
            plt.show()
    
    if PLOTLY_AVAILABLE:
        # Test interactive feature maps
        feature_maps = np.random.randn(16, 32, 32)
        visualizer = FeatureVisualizer()
        
        fig = visualizer.create_interactive_feature_map(feature_maps, "Interactive Layer")
        if fig:
            fig.show()