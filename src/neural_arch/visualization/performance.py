"""Performance Visualization Module.

This module provides tools for visualizing performance benchmarks,
backend comparisons, memory usage, and computational efficiency.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.sankey import Sankey
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
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from neural_arch.backends import available_backends, get_backend


class PerformanceVisualizer:
    """Advanced performance and benchmark visualization."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "modern"):
        """Initialize the performance visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
            style: Visualization style ('modern', 'professional', 'minimal')
        """
        self.figsize = figsize
        self.style = style
        
        if MATPLOTLIB_AVAILABLE:
            if style == "modern":
                plt.style.use('seaborn-v0_8-darkgrid')
                sns.set_palette("Set2")
            elif style == "professional":
                plt.style.use('seaborn-v0_8-whitegrid')
                sns.set_palette("deep")
    
    def plot_benchmark_results(self, benchmark_data: Dict[str, Dict[str, float]],
                             benchmark_name: str = "Performance Benchmark",
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot comprehensive benchmark results.
        
        Args:
            benchmark_data: Dictionary with format {backend: {operation: time}}
            benchmark_name: Name of the benchmark
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib seaborn")
            return None
        
        backends = list(benchmark_data.keys())
        operations = list(benchmark_data[backends[0]].keys()) if backends else []
        
        if not backends or not operations:
            print("No benchmark data provided")
            return None
        
        # Create subplots: one for absolute times, one for speedup ratios
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # Plot 1: Absolute execution times
        x = np.arange(len(operations))
        width = 0.8 / len(backends)
        
        colors = sns.color_palette("Set2", len(backends))
        
        for i, backend in enumerate(backends):
            times = [benchmark_data[backend].get(op, 0) for op in operations]
            ax1.bar(x + i * width, times, width, label=backend, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Operations', fontsize=12)
        ax1.set_ylabel('Execution Time (ms)', fontsize=12)
        ax1.set_title(f'{benchmark_name} - Absolute Performance', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(backends) - 1) / 2)
        ax1.set_xticklabels(operations, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better comparison
        
        # Plot 2: Speedup ratios (relative to first backend)
        if len(backends) > 1:
            baseline_backend = backends[0]
            
            for i, backend in enumerate(backends[1:], 1):
                speedups = []
                for op in operations:
                    baseline_time = benchmark_data[baseline_backend].get(op, 1)
                    backend_time = benchmark_data[backend].get(op, 1)
                    speedup = baseline_time / backend_time if backend_time > 0 else 0
                    speedups.append(speedup)
                
                ax2.bar(x + i * width, speedups, width, label=f'{backend} vs {baseline_backend}', 
                       color=colors[i], alpha=0.8)
            
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
            ax2.set_xlabel('Operations', fontsize=12)
            ax2.set_ylabel('Speedup Ratio', fontsize=12)
            ax2.set_title(f'{benchmark_name} - Speedup Ratios', fontsize=14, fontweight='bold')
            ax2.set_xticks(x + width * len(backends) / 2)
            ax2.set_xticklabels(operations, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Need at least 2 backends for speedup comparison',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Speedup Comparison (N/A)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_backend_performance(self, matrix_sizes: List[int], 
                                  backends: Optional[List[str]] = None,
                                  save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Compare backend performance across different matrix sizes.
        
        Args:
            matrix_sizes: List of matrix sizes to test
            backends: List of backend names to compare (None for all available)
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if backends is None:
            backends = available_backends()
        
        # Benchmark matrix multiplication for different sizes
        results = {backend: {'sizes': [], 'times': []} for backend in backends}
        
        for size in matrix_sizes:
            print(f"Benchmarking matrix size {size}x{size}...")
            
            for backend_name in backends:
                try:
                    backend = get_backend(backend_name)
                    if not backend.is_available:
                        continue
                    
                    # Create test matrices
                    a = backend.random_normal((size, size))
                    b = backend.random_normal((size, size))
                    
                    # Warmup
                    for _ in range(3):
                        _ = backend.matmul(a, b)
                    
                    # Benchmark
                    start_time = time.time()
                    for _ in range(5):
                        result = backend.matmul(a, b)
                        # Force computation completion
                        if hasattr(result, 'data'):
                            _ = backend.to_numpy(result) if hasattr(backend, 'to_numpy') else result
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 5 * 1000  # Convert to ms
                    
                    results[backend_name]['sizes'].append(size)
                    results[backend_name]['times'].append(avg_time)
                    
                except Exception as e:
                    print(f"Error benchmarking {backend_name}: {e}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))
        
        colors = sns.color_palette("Set1", len(backends))
        
        # Plot 1: Absolute performance
        for i, backend_name in enumerate(backends):
            if results[backend_name]['sizes']:
                ax1.loglog(results[backend_name]['sizes'], results[backend_name]['times'], 
                          marker='o', linewidth=2, label=backend_name, color=colors[i])
        
        ax1.set_xlabel('Matrix Size', fontsize=12)
        ax1.set_ylabel('Execution Time (ms)', fontsize=12)
        ax1.set_title('Backend Performance vs Matrix Size', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory efficiency (FLOPS/ms)
        for i, backend_name in enumerate(backends):
            if results[backend_name]['sizes']:
                sizes = np.array(results[backend_name]['sizes'])
                times = np.array(results[backend_name]['times'])
                
                # Calculate FLOPS (2 * n^3 for matrix multiplication)
                flops = 2 * sizes ** 3
                efficiency = flops / (times * 1e6)  # GFLOPS
                
                ax2.semilogx(sizes, efficiency, marker='s', linewidth=2, 
                           label=backend_name, color=colors[i])
        
        ax2.set_xlabel('Matrix Size', fontsize=12)
        ax2.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax2.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_memory_usage(self, memory_data: Dict[str, List[float]], 
                         time_points: Optional[List[float]] = None,
                         save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot memory usage over time.
        
        Args:
            memory_data: Dictionary with format {metric: [values]}
            time_points: Time points for x-axis (None for sequential)
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if time_points is None:
            max_len = max(len(values) for values in memory_data.values())
            time_points = list(range(max_len))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        colors = sns.color_palette("Set3", len(memory_data))
        
        # Plot 1: Memory usage over time
        for i, (metric, values) in enumerate(memory_data.items()):
            t_points = time_points[:len(values)]
            ax1.plot(t_points, values, marker='o', linewidth=2, label=metric, color=colors[i])
        
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax1.set_title('Memory Usage Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory distribution (if system info available)
        if PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            labels = ['Used', 'Available', 'Cached']
            sizes = [
                memory_info.used / (1024**3),  # GB
                memory_info.available / (1024**3),
                getattr(memory_info, 'cached', 0) / (1024**3)
            ]
            colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
            
            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax2.set_title('System Memory Distribution', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'System memory info not available\n(install psutil)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_dashboard(self, benchmark_results: Dict,
                                   memory_data: Optional[Dict] = None) -> Optional[go.Figure]:
        """Create interactive performance dashboard using Plotly.
        
        Args:
            benchmark_results: Benchmark data
            memory_data: Optional memory usage data
            
        Returns:
            Plotly Figure object or None
        """
        if not PLOTLY_AVAILABLE:
            print("plotly not available. Install with: pip install plotly")
            return None
        
        # Create subplots
        if memory_data:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Execution Times', 'Speedup Ratios', 
                    'Memory Usage', 'Performance Metrics'
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "pie"}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Execution Times', 'Speedup Ratios']
            )
        
        # Add benchmark data
        if 'backends' in benchmark_results and 'operations' in benchmark_results:
            backends = benchmark_results['backends']
            operations = benchmark_results['operations']
            
            colors = px.colors.qualitative.Set1
            
            for i, backend in enumerate(backends):
                times = benchmark_results.get(backend, [])
                
                fig.add_trace(
                    go.Bar(
                        x=operations,
                        y=times,
                        name=backend,
                        marker_color=colors[i % len(colors)],
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add memory data if available
        if memory_data:
            for metric, values in memory_data.items():
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(values))),
                        y=values,
                        mode='lines+markers',
                        name=metric,
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="Performance Dashboard",
            height=800 if memory_data else 400,
            showlegend=True
        )
        
        return fig
    
    def benchmark_operations(self, operations: List[str], backends: Optional[List[str]] = None,
                           matrix_size: int = 1000) -> Dict[str, Dict[str, float]]:
        """Benchmark specific operations across backends.
        
        Args:
            operations: List of operations to benchmark
            backends: List of backend names (None for all available)
            matrix_size: Size of matrices to use for benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        if backends is None:
            backends = available_backends()
        
        results = {backend: {} for backend in backends}
        
        for backend_name in backends:
            try:
                backend = get_backend(backend_name)
                if not backend.is_available:
                    continue
                
                print(f"Benchmarking {backend_name}...")
                
                # Create test data
                a = backend.random_normal((matrix_size, matrix_size))
                b = backend.random_normal((matrix_size, matrix_size))
                
                for operation in operations:
                    try:
                        # Warmup
                        for _ in range(3):
                            if operation == 'matmul':
                                _ = backend.matmul(a, b)
                            elif operation == 'add':
                                _ = backend.add(a, b)
                            elif operation == 'multiply':
                                _ = backend.multiply(a, b)
                            elif operation == 'sum':
                                _ = backend.sum(a)
                            elif operation == 'exp':
                                _ = backend.exp(a)
                        
                        # Benchmark
                        start_time = time.time()
                        for _ in range(5):
                            if operation == 'matmul':
                                result = backend.matmul(a, b)
                            elif operation == 'add':
                                result = backend.add(a, b)
                            elif operation == 'multiply':
                                result = backend.multiply(a, b)
                            elif operation == 'sum':
                                result = backend.sum(a)
                            elif operation == 'exp':
                                result = backend.exp(a)
                            
                            # Force computation
                            if hasattr(result, 'data'):
                                _ = backend.to_numpy(result) if hasattr(backend, 'to_numpy') else result
                        
                        end_time = time.time()
                        avg_time = (end_time - start_time) / 5 * 1000  # ms
                        
                        results[backend_name][operation] = avg_time
                        
                    except Exception as e:
                        print(f"Error benchmarking {operation} on {backend_name}: {e}")
                        results[backend_name][operation] = float('inf')
                
            except Exception as e:
                print(f"Error with backend {backend_name}: {e}")
        
        return results


def plot_benchmark_results(benchmark_data: Dict[str, Dict[str, float]], 
                          benchmark_name: str = "Performance Benchmark",
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to plot benchmark results."""
    visualizer = PerformanceVisualizer()
    return visualizer.plot_benchmark_results(benchmark_data, benchmark_name, save_path)


def compare_backend_performance(matrix_sizes: List[int], backends: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to compare backend performance."""
    visualizer = PerformanceVisualizer()
    return visualizer.compare_backend_performance(matrix_sizes, backends, save_path)


def plot_memory_usage(memory_data: Dict[str, List[float]], 
                     time_points: Optional[List[float]] = None,
                     save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Convenience function to plot memory usage."""
    visualizer = PerformanceVisualizer()
    return visualizer.plot_memory_usage(memory_data, time_points, save_path)


def create_performance_dashboard(benchmark_results: Dict,
                               memory_data: Optional[Dict] = None) -> Optional[go.Figure]:
    """Convenience function to create performance dashboard."""
    visualizer = PerformanceVisualizer()
    return visualizer.create_performance_dashboard(benchmark_results, memory_data)


# Example usage
if __name__ == "__main__":
    # Test performance visualization
    visualizer = PerformanceVisualizer()
    
    # Generate sample benchmark data
    sample_data = {
        'numpy': {'matmul': 50, 'add': 5, 'exp': 10},
        'mps': {'matmul': 2, 'add': 0.5, 'exp': 1},
        'jit': {'matmul': 25, 'add': 3, 'exp': 7}
    }
    
    if MATPLOTLIB_AVAILABLE:
        # Test benchmark plotting
        fig1 = visualizer.plot_benchmark_results(sample_data, "Sample Benchmark")
        if fig1:
            plt.show()
        
        # Test memory plotting
        memory_data = {
            'GPU Memory': [100, 150, 200, 250, 300, 280, 260],
            'CPU Memory': [500, 520, 540, 560, 580, 600, 620]
        }
        
        fig2 = visualizer.plot_memory_usage(memory_data)
        if fig2:
            plt.show()
    
    if PLOTLY_AVAILABLE:
        # Test interactive dashboard
        benchmark_results = {
            'backends': ['numpy', 'mps', 'jit'],
            'operations': ['matmul', 'add', 'exp'],
            'numpy': [50, 5, 10],
            'mps': [2, 0.5, 1],
            'jit': [25, 3, 7]
        }
        
        fig = visualizer.create_performance_dashboard(benchmark_results)
        if fig:
            fig.show()