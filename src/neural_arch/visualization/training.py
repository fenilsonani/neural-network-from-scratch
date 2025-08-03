"""Training Metrics Visualization.

This module provides comprehensive visualization tools for training metrics,
loss curves, performance analysis, and training progress monitoring.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
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

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


class TrainingVisualizer:
    """Comprehensive training metrics visualization and analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "modern"):
        """Initialize the training visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
            style: Visualization style ('modern', 'seaborn', 'classic')
        """
        self.figsize = figsize
        self.style = style
        self.metrics_history = {}
        
        if MATPLOTLIB_AVAILABLE:
            if style == "modern":
                plt.style.use('seaborn-v0_8-darkgrid')
                sns.set_palette("husl")
            elif style == "seaborn":
                sns.set_style("whitegrid")
                sns.set_palette("deep")
    
    def add_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Add metrics for a specific epoch.
        
        Args:
            epoch: Training epoch number
            metrics: Dictionary of metric names and values
        """
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = {'epochs': [], 'values': []}
            
            self.metrics_history[metric_name]['epochs'].append(epoch)
            self.metrics_history[metric_name]['values'].append(value)
    
    def plot_training_curves(self, metrics: Optional[List[str]] = None, 
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot training curves for specified metrics."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib seaborn")
            return None
        
        if not self.metrics_history:
            print("No metrics history available. Use add_metrics() first.")
            return None
        
        # Default to all metrics if none specified
        if metrics is None:
            metrics = list(self.metrics_history.keys())
        
        # Determine subplot layout
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric not in self.metrics_history:
                continue
            
            ax = axes[i]
            data = self.metrics_history[metric]
            
            ax.plot(data['epochs'], data['values'], marker='o', linewidth=2, markersize=4)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(data['epochs']) > 2:
                z = np.polyfit(data['epochs'], data['values'], 1)
                p = np.poly1d(z)
                ax.plot(data['epochs'], p(data['epochs']), "--", alpha=0.7, color='red')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_loss_comparison(self, train_loss: List[float], val_loss: Optional[List[float]] = None,
                           epochs: Optional[List[int]] = None, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot training vs validation loss comparison."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if epochs is None:
            epochs = list(range(1, len(train_loss) + 1))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2)
        
        if val_loss is not None:
            ax.plot(epochs[:len(val_loss)], val_loss, label='Validation Loss', marker='s', linewidth=2)
            
            # Highlight best validation loss
            best_val_idx = np.argmin(val_loss)
            ax.axvline(x=epochs[best_val_idx], color='red', linestyle='--', alpha=0.7)
            ax.text(epochs[best_val_idx], val_loss[best_val_idx], 
                   f'Best: {val_loss[best_val_idx]:.4f}', 
                   verticalalignment='bottom', fontsize=10)
        
        ax.set_title('Training Progress', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_metrics_dashboard(self, metrics_data: Dict[str, Dict]) -> Optional[go.Figure]:
        """Create interactive dashboard with multiple metrics."""
        if not PLOTLY_AVAILABLE:
            print("plotly not available. Install with: pip install plotly")
            return None
        
        # Determine subplot layout
        n_metrics = len(metrics_data)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        subplot_titles = list(metrics_data.keys())
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (metric_name, data) in enumerate(metrics_data.items()):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            if 'train' in data:
                fig.add_trace(
                    go.Scatter(
                        x=data.get('epochs', list(range(len(data['train'])))),
                        y=data['train'],
                        mode='lines+markers',
                        name=f'{metric_name} (Train)',
                        line=dict(color=colors[i % len(colors)]),
                        showlegend=True
                    ),
                    row=row, col=col
                )
            
            if 'val' in data:
                fig.add_trace(
                    go.Scatter(
                        x=data.get('epochs', list(range(len(data['val'])))),
                        y=data['val'],
                        mode='lines+markers',
                        name=f'{metric_name} (Val)',
                        line=dict(color=colors[i % len(colors)], dash='dash'),
                        showlegend=True
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Training Metrics Dashboard",
            height=400 * n_rows,
            showlegend=True
        )
        
        return fig
    
    def plot_learning_rate_schedule(self, learning_rates: List[float], 
                                  epochs: Optional[List[int]] = None,
                                  save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot learning rate schedule over training."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if epochs is None:
            epochs = list(range(1, len(learning_rates) + 1))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(epochs, learning_rates, marker='o', linewidth=2, color='green')
        ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for significant changes
        if len(learning_rates) > 1:
            changes = np.diff(learning_rates)
            significant_changes = np.where(np.abs(changes) > 0.1 * learning_rates[0])[0]
            
            for change_idx in significant_changes:
                ax.annotate(f'LR: {learning_rates[change_idx+1]:.6f}',
                           xy=(epochs[change_idx+1], learning_rates[change_idx+1]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_training_report(self, model_name: str, training_config: Dict,
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Generate comprehensive training report."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5], hspace=0.3, wspace=0.3)
        
        # Main loss curves
        ax1 = fig.add_subplot(gs[0, :])
        if 'loss' in self.metrics_history:
            data = self.metrics_history['loss']
            ax1.plot(data['epochs'], data['values'], marker='o', linewidth=2)
        ax1.set_title(f'{model_name} - Training Loss', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy if available
        ax2 = fig.add_subplot(gs[1, 0])
        if 'accuracy' in self.metrics_history:
            data = self.metrics_history['accuracy']
            ax2.plot(data['epochs'], data['values'], marker='s', linewidth=2, color='green')
            ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True, alpha=0.3)
        
        # Additional metrics
        ax3 = fig.add_subplot(gs[1, 1])
        other_metrics = [k for k in self.metrics_history.keys() if k not in ['loss', 'accuracy']]
        if other_metrics:
            for i, metric in enumerate(other_metrics[:3]):  # Show up to 3 additional metrics
                data = self.metrics_history[metric]
                ax3.plot(data['epochs'], data['values'], marker='o', linewidth=2, 
                        label=metric, alpha=0.8)
            ax3.set_title('Other Metrics', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Training summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
        Model: {model_name}
        Total Epochs: {max([max(data['epochs']) for data in self.metrics_history.values()]) if self.metrics_history else 0}
        
        Configuration:
        """
        
        for key, value in training_config.items():
            summary_text += f"        {key}: {value}\n"
        
        # Final metrics
        if self.metrics_history:
            summary_text += "\n        Final Metrics:\n"
            for metric, data in self.metrics_history.items():
                if data['values']:
                    summary_text += f"        {metric}: {data['values'][-1]:.4f}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Training Report - {model_name}', fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_training_curves(train_loss: List[float], val_loss: Optional[List[float]] = None,
                        metrics: Optional[Dict[str, List[float]]] = None,
                        save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to plot training curves.
    
    Args:
        train_loss: List of training loss values
        val_loss: Optional list of validation loss values
        metrics: Optional dictionary of additional metrics
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure object or None
    """
    visualizer = TrainingVisualizer()
    
    # Add loss data
    epochs = list(range(1, len(train_loss) + 1))
    for i, loss in enumerate(train_loss):
        visualizer.add_metrics(epochs[i], {'train_loss': loss})
    
    if val_loss:
        for i, loss in enumerate(val_loss):
            if i < len(epochs):
                visualizer.add_metrics(epochs[i], {'val_loss': loss})
    
    # Add additional metrics
    if metrics:
        for metric_name, values in metrics.items():
            for i, value in enumerate(values):
                if i < len(epochs):
                    visualizer.add_metrics(epochs[i], {metric_name: value})
    
    return visualizer.plot_training_curves()


def plot_loss_history(loss_history: Dict[str, List[float]], 
                     save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot loss history with train/validation comparison."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    visualizer = TrainingVisualizer()
    return visualizer.plot_loss_comparison(
        train_loss=loss_history.get('train', []),
        val_loss=loss_history.get('val'),
        save_path=save_path
    )


def plot_metrics_dashboard(metrics_data: Dict[str, Dict]) -> Optional[go.Figure]:
    """Create interactive metrics dashboard."""
    visualizer = TrainingVisualizer()
    return visualizer.create_metrics_dashboard(metrics_data)


def create_training_report(model_name: str, metrics_history: Dict[str, List[float]],
                         training_config: Dict, save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Generate comprehensive training report."""
    visualizer = TrainingVisualizer()
    
    # Add metrics to visualizer
    for metric_name, values in metrics_history.items():
        for i, value in enumerate(values):
            visualizer.add_metrics(i + 1, {metric_name: value})
    
    return visualizer.create_training_report(model_name, training_config, save_path)


# Example usage
if __name__ == "__main__":
    # Generate sample training data
    epochs = 50
    train_loss = [1.0 * np.exp(-0.1 * i) + 0.1 * np.random.random() for i in range(epochs)]
    val_loss = [1.0 * np.exp(-0.08 * i) + 0.15 * np.random.random() for i in range(epochs)]
    accuracy = [1 - np.exp(-0.1 * i) + 0.05 * np.random.random() for i in range(epochs)]
    
    # Test visualizations
    visualizer = TrainingVisualizer()
    
    # Add sample data
    for i in range(epochs):
        visualizer.add_metrics(i + 1, {
            'train_loss': train_loss[i],
            'val_loss': val_loss[i],
            'accuracy': accuracy[i]
        })
    
    if MATPLOTLIB_AVAILABLE:
        # Plot training curves
        fig = visualizer.plot_training_curves()
        if fig:
            plt.show()
        
        # Create training report
        config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'Adam',
            'model_size': '10M parameters'
        }
        
        report_fig = visualizer.create_training_report('TestModel', config)
        if report_fig:
            plt.show()
    
    if PLOTLY_AVAILABLE:
        # Create interactive dashboard
        metrics_data = {
            'loss': {'train': train_loss, 'val': val_loss},
            'accuracy': {'train': accuracy}
        }
        
        dashboard = visualizer.create_metrics_dashboard(metrics_data)
        if dashboard:
            dashboard.show()