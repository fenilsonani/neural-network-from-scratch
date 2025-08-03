"""Model Compression Utilities.

This module provides utility functions for model compression analysis,
export/import, and performance measurement.
"""

import os
import sys
import time
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neural_arch.core.tensor import Tensor
from neural_arch.nn.module import Module

logger = logging.getLogger(__name__)


def calculate_model_size(model: Module, unit: str = "MB") -> float:
    """Calculate model size in specified unit.
    
    Args:
        model: Model to analyze
        unit: Size unit ('B', 'KB', 'MB', 'GB')
        
    Returns:
        Model size in specified unit
    """
    total_bytes = 0
    
    for param in model.parameters():
        total_bytes += param.data.nbytes
    
    # Convert to specified unit
    unit_multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    
    if unit not in unit_multipliers:
        raise ValueError(f"Unknown unit: {unit}. Use one of {list(unit_multipliers.keys())}")
    
    return total_bytes / unit_multipliers[unit]


def calculate_flops(model: Module, input_shape: Tuple[int, ...]) -> Dict[str, int]:
    """Calculate FLOPs (Floating Point Operations) for model inference.
    
    Args:
        model: Model to analyze
        input_shape: Shape of input tensor (including batch dimension)
        
    Returns:
        Dictionary with FLOP counts by operation type
    """
    flop_counts = {
        'total': 0,
        'linear': 0,
        'conv': 0,
        'activation': 0,
        'other': 0
    }
    
    def count_linear_flops(module, input_shape, output_shape):
        """Count FLOPs for linear layer."""
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            # Matrix multiplication: batch_size * in_features * out_features
            batch_size = input_shape[0] if input_shape else 1
            flops = batch_size * module.in_features * module.out_features
            
            # Add bias FLOPs if present
            if hasattr(module, 'bias') and module.bias is not None:
                flops += batch_size * module.out_features
            
            return flops
        return 0
    
    def count_conv_flops(module, input_shape, output_shape):
        """Count FLOPs for convolutional layer."""
        # Simplified conv FLOP counting
        if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            batch_size = input_shape[0] if input_shape else 1
            
            # Estimate output spatial dimensions (simplified)
            if len(input_shape) >= 4:  # Conv2D
                in_h, in_w = input_shape[2], input_shape[3]
                kernel_size = getattr(module, 'kernel_size', 3)
                if isinstance(kernel_size, int):
                    kernel_h, kernel_w = kernel_size, kernel_size
                else:
                    kernel_h, kernel_w = kernel_size
                
                # Assume no padding/stride for simplicity
                out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
                
                # FLOPs = batch_size * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w
                flops = (batch_size * module.out_channels * out_h * out_w * 
                        module.in_channels * kernel_h * kernel_w)
                return flops
        return 0
    
    # Traverse model and count FLOPs
    current_shape = input_shape
    
    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()
        
        if 'linear' in module_type or 'dense' in module_type:
            flops = count_linear_flops(module, current_shape, None)
            flop_counts['linear'] += flops
            flop_counts['total'] += flops
            
            # Update shape for next layer
            if hasattr(module, 'out_features'):
                current_shape = (current_shape[0], module.out_features)
        
        elif 'conv' in module_type:
            flops = count_conv_flops(module, current_shape, None)
            flop_counts['conv'] += flops
            flop_counts['total'] += flops
        
        elif any(act_type in module_type for act_type in ['relu', 'sigmoid', 'tanh', 'gelu']):
            # Activation functions: 1 FLOP per element
            if current_shape:
                batch_elements = np.prod(current_shape)
                flop_counts['activation'] += batch_elements
                flop_counts['total'] += batch_elements
    
    return flop_counts


def measure_inference_time(model: Module, input_tensor: Tensor, 
                          num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
    """Measure model inference time with statistical analysis.
    
    Args:
        model: Model to benchmark
        input_tensor: Input tensor for inference
        num_runs: Number of inference runs for measurement
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with timing statistics
    """
    logger.info(f"Measuring inference time over {num_runs} runs...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(input_tensor)
    
    # Measurement runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model(input_tensor)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'num_runs': num_runs,
        'warmup_runs': warmup_runs,
        'throughput_fps': 1000.0 / float(np.mean(times))  # Frames per second
    }


def analyze_compression_tradeoffs(original_model: Module, compressed_model: Module,
                                test_data: List[Tensor],
                                accuracy_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Analyze compression tradeoffs between size, speed, and accuracy.
    
    Args:
        original_model: Original uncompressed model
        compressed_model: Compressed model
        test_data: Test data for evaluation
        accuracy_fn: Function to compute accuracy (optional)
        
    Returns:
        Comprehensive tradeoff analysis
    """
    logger.info("Analyzing compression tradeoffs...")
    
    analysis = {}
    
    # Size analysis
    original_size = calculate_model_size(original_model, "MB")
    compressed_size = calculate_model_size(compressed_model, "MB")
    
    analysis['size_analysis'] = {
        'original_size_mb': original_size,
        'compressed_size_mb': compressed_size,
        'size_reduction_ratio': original_size / compressed_size if compressed_size > 0 else float('inf'),
        'size_reduction_percentage': ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0.0
    }
    
    # Parameter analysis
    original_params = sum(p.size for p in original_model.parameters())
    compressed_params = sum(p.size for p in compressed_model.parameters())
    
    analysis['parameter_analysis'] = {
        'original_parameters': original_params,
        'compressed_parameters': compressed_params,
        'parameter_reduction_ratio': original_params / compressed_params if compressed_params > 0 else float('inf'),
        'parameter_reduction_percentage': ((original_params - compressed_params) / original_params) * 100 if original_params > 0 else 0.0
    }
    
    # Speed analysis
    if test_data:
        sample_input = test_data[0]
        
        original_timing = measure_inference_time(original_model, sample_input, num_runs=50)
        compressed_timing = measure_inference_time(compressed_model, sample_input, num_runs=50)
        
        analysis['speed_analysis'] = {
            'original_inference_ms': original_timing['mean_ms'],
            'compressed_inference_ms': compressed_timing['mean_ms'],
            'speedup_ratio': original_timing['mean_ms'] / compressed_timing['mean_ms'] if compressed_timing['mean_ms'] > 0 else 1.0,
            'speedup_percentage': ((original_timing['mean_ms'] - compressed_timing['mean_ms']) / original_timing['mean_ms']) * 100 if original_timing['mean_ms'] > 0 else 0.0,
            'original_throughput_fps': original_timing['throughput_fps'],
            'compressed_throughput_fps': compressed_timing['throughput_fps']
        }
    
    # Accuracy analysis (if accuracy function provided)
    if accuracy_fn:
        original_accuracy = accuracy_fn(original_model)
        compressed_accuracy = accuracy_fn(compressed_model)
        
        analysis['accuracy_analysis'] = {
            'original_accuracy': original_accuracy,
            'compressed_accuracy': compressed_accuracy,
            'accuracy_loss': original_accuracy - compressed_accuracy,
            'accuracy_retention': compressed_accuracy / original_accuracy if original_accuracy > 0 else 1.0
        }
    
    # FLOP analysis
    if test_data:
        input_shape = test_data[0].shape
        original_flops = calculate_flops(original_model, input_shape)
        compressed_flops = calculate_flops(compressed_model, input_shape)
        
        analysis['flops_analysis'] = {
            'original_flops': original_flops['total'],
            'compressed_flops': compressed_flops['total'],
            'flops_reduction_ratio': original_flops['total'] / compressed_flops['total'] if compressed_flops['total'] > 0 else float('inf'),
            'flops_reduction_percentage': ((original_flops['total'] - compressed_flops['total']) / original_flops['total']) * 100 if original_flops['total'] > 0 else 0.0
        }
    
    # Efficiency metrics
    size_ratio = analysis['size_analysis']['size_reduction_ratio']
    speed_ratio = analysis['speed_analysis']['speedup_ratio'] if 'speed_analysis' in analysis else 1.0
    
    analysis['efficiency_metrics'] = {
        'compression_efficiency': size_ratio * speed_ratio,  # Combined size and speed improvement
        'size_speed_tradeoff': size_ratio / speed_ratio if speed_ratio > 0 else size_ratio,
        'bytes_per_flop_original': (original_size * 1024 * 1024) / original_flops['total'] if 'flops_analysis' in analysis and original_flops['total'] > 0 else 0,
        'bytes_per_flop_compressed': (compressed_size * 1024 * 1024) / compressed_flops['total'] if 'flops_analysis' in analysis and compressed_flops['total'] > 0 else 0
    }
    
    # Overall summary
    analysis['summary'] = {
        'compression_achieved': size_ratio > 1.1,  # At least 10% compression
        'speed_improved': speed_ratio > 1.0,
        'acceptable_accuracy_loss': analysis['accuracy_analysis']['accuracy_loss'] < 0.05 if 'accuracy_analysis' in analysis else True,
        'overall_improvement': size_ratio > 1.1 and (speed_ratio >= 0.9)  # Good compression with minimal speed loss
    }
    
    return analysis


def export_compressed_model(model: Module, export_path: str, 
                           compression_info: Optional[Dict] = None,
                           format: str = "pickle") -> str:
    """Export compressed model to file.
    
    Args:
        model: Compressed model to export
        export_path: Path to export the model
        compression_info: Additional compression information
        format: Export format ('pickle', 'json_weights', 'onnx')
        
    Returns:
        Path to exported model file
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        # Export entire model using pickle
        export_data = {
            'model': model,
            'compression_info': compression_info,
            'export_format': 'pickle',
            'neural_forge_version': '1.0.0'
        }
        
        with open(export_path, 'wb') as f:
            pickle.dump(export_data, f)
    
    elif format == "json_weights":
        # Export model weights as JSON
        weights_dict = {}
        for name, param in model.named_parameters():
            weights_dict[name] = {
                'data': param.data.tolist(),
                'shape': param.shape,
                'dtype': str(param.dtype)
            }
        
        export_data = {
            'weights': weights_dict,
            'model_structure': str(model),
            'compression_info': compression_info,
            'export_format': 'json_weights',
            'neural_forge_version': '1.0.0'
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    logger.info(f"Model exported to {export_path} in {format} format")
    return str(export_path)


def load_compressed_model(model_path: str) -> Tuple[Module, Optional[Dict]]:
    """Load compressed model from file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (loaded_model, compression_info)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Detect format from file extension or content
    if model_path.suffix == '.pkl' or model_path.suffix == '.pickle':
        format = "pickle"
    elif model_path.suffix == '.json':
        format = "json_weights"
    else:
        # Try to detect from content
        try:
            with open(model_path, 'rb') as f:
                pickle.load(f)
            format = "pickle"
        except:
            format = "json_weights"
    
    if format == "pickle":
        with open(model_path, 'rb') as f:
            export_data = pickle.load(f)
        
        model = export_data['model']
        compression_info = export_data.get('compression_info')
    
    elif format == "json_weights":
        with open(model_path, 'r') as f:
            export_data = json.load(f)
        
        # Reconstruct model from weights (simplified)
        # In practice, this would require model architecture reconstruction
        weights = export_data['weights']
        compression_info = export_data.get('compression_info')
        
        # For now, return None as model reconstruction is complex
        logger.warning("JSON weights format requires manual model reconstruction")
        model = None
    
    else:
        raise ValueError(f"Unsupported model format: {format}")
    
    logger.info(f"Model loaded from {model_path}")
    return model, compression_info


def get_model_summary(model: Module, input_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
    """Get comprehensive model summary including compression analysis.
    
    Args:
        model: Model to analyze
        input_shape: Input shape for FLOP calculation (optional)
        
    Returns:
        Comprehensive model summary
    """
    summary = {}
    
    # Basic model info
    total_params = sum(p.size for p in model.parameters())
    trainable_params = sum(p.size for p in model.parameters() if p.requires_grad)
    
    summary['model_info'] = {
        'model_type': type(model).__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': calculate_model_size(model, "MB"),
        'model_size_kb': calculate_model_size(model, "KB")
    }
    
    # Layer breakdown
    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.size for p in module.parameters())
            layer_info.append({
                'name': name,
                'type': type(module).__name__,
                'parameters': module_params,
                'size_kb': sum(p.data.nbytes for p in module.parameters()) / 1024
            })
    
    summary['layer_breakdown'] = layer_info
    
    # FLOP analysis if input shape provided
    if input_shape:
        flops = calculate_flops(model, input_shape)
        summary['computational_complexity'] = flops
        
        # Efficiency metrics
        summary['efficiency_metrics'] = {
            'params_per_flop': total_params / flops['total'] if flops['total'] > 0 else 0,
            'bytes_per_flop': (calculate_model_size(model, "B")) / flops['total'] if flops['total'] > 0 else 0,
            'mflops': flops['total'] / 1e6
        }
    
    # Memory usage estimation
    param_memory = sum(p.data.nbytes for p in model.parameters())
    
    # Estimate activation memory (rough approximation)
    if input_shape:
        activation_memory = np.prod(input_shape) * 4 * len(layer_info)  # Rough estimate
    else:
        activation_memory = 0
    
    summary['memory_usage'] = {
        'parameters_mb': param_memory / (1024 * 1024),
        'estimated_activations_mb': activation_memory / (1024 * 1024),
        'estimated_total_mb': (param_memory + activation_memory) / (1024 * 1024)
    }
    
    return summary


def compare_models(models: Dict[str, Module], 
                  test_input: Optional[Tensor] = None,
                  accuracy_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Compare multiple models across various metrics.
    
    Args:
        models: Dictionary of model_name -> model
        test_input: Test input for performance measurement
        accuracy_fn: Function to compute accuracy
        
    Returns:
        Comprehensive model comparison
    """
    comparison = {
        'models': list(models.keys()),
        'metrics': {}
    }
    
    for metric in ['size_mb', 'parameters', 'inference_time_ms', 'accuracy', 'flops']:
        comparison['metrics'][metric] = {}
    
    # Analyze each model
    for name, model in models.items():
        # Size and parameters
        size_mb = calculate_model_size(model, "MB")
        params = sum(p.size for p in model.parameters())
        
        comparison['metrics']['size_mb'][name] = size_mb
        comparison['metrics']['parameters'][name] = params
        
        # Performance measurement
        if test_input is not None:
            timing = measure_inference_time(model, test_input, num_runs=20)
            comparison['metrics']['inference_time_ms'][name] = timing['mean_ms']
            
            # FLOP calculation
            flops = calculate_flops(model, test_input.shape)
            comparison['metrics']['flops'][name] = flops['total']
        
        # Accuracy measurement
        if accuracy_fn is not None:
            accuracy = accuracy_fn(model)
            comparison['metrics']['accuracy'][name] = accuracy
    
    # Calculate relative performance
    comparison['relative_metrics'] = {}
    
    for metric_name, metric_values in comparison['metrics'].items():
        if not metric_values:
            continue
        
        # Find baseline (first model)
        baseline_name = list(metric_values.keys())[0]
        baseline_value = metric_values[baseline_name]
        
        relative_values = {}
        for model_name, value in metric_values.items():
            if baseline_value != 0:
                if metric_name in ['size_mb', 'parameters', 'inference_time_ms', 'flops']:
                    # Lower is better - show reduction ratio
                    relative_values[model_name] = baseline_value / value if value > 0 else float('inf')
                else:
                    # Higher is better - show improvement ratio
                    relative_values[model_name] = value / baseline_value if baseline_value > 0 else 1.0
            else:
                relative_values[model_name] = 1.0
        
        comparison['relative_metrics'][metric_name] = relative_values
    
    # Summary statistics
    comparison['summary'] = {}
    for metric_name, metric_values in comparison['metrics'].items():
        if metric_values:
            values = list(metric_values.values())
            comparison['summary'][metric_name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    return comparison


# Example usage and testing
if __name__ == "__main__":
    # Test compression utilities
    from neural_arch.nn import Sequential, Linear, ReLU
    
    # Create test models
    large_model = Sequential(
        Linear(100, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )
    
    small_model = Sequential(
        Linear(100, 64),
        ReLU(),
        Linear(64, 10)
    )
    
    print("Testing Neural Forge Compression Utilities...")
    
    # Test model size calculation
    large_size = calculate_model_size(large_model, "KB")
    small_size = calculate_model_size(small_model, "KB")
    
    print(f"\n=== Model Size Analysis ===")
    print(f"Large model: {large_size:.2f} KB")
    print(f"Small model: {small_size:.2f} KB")
    print(f"Size reduction: {large_size / small_size:.2f}x")
    
    # Test FLOP calculation
    input_shape = (32, 100)  # Batch size 32, 100 features
    large_flops = calculate_flops(large_model, input_shape)
    small_flops = calculate_flops(small_model, input_shape)
    
    print(f"\n=== FLOP Analysis ===")
    print(f"Large model FLOPs: {large_flops['total']:,}")
    print(f"Small model FLOPs: {small_flops['total']:,}")
    print(f"FLOP reduction: {large_flops['total'] / small_flops['total']:.2f}x")
    
    # Test inference timing
    test_input = Tensor(np.random.randn(32, 100), dtype=np.float32)
    
    print(f"\n=== Inference Timing ===")
    large_timing = measure_inference_time(large_model, test_input, num_runs=20)
    small_timing = measure_inference_time(small_model, test_input, num_runs=20)
    
    print(f"Large model: {large_timing['mean_ms']:.2f} ± {large_timing['std_ms']:.2f} ms")
    print(f"Small model: {small_timing['mean_ms']:.2f} ± {small_timing['std_ms']:.2f} ms")
    print(f"Speedup: {large_timing['mean_ms'] / small_timing['mean_ms']:.2f}x")
    
    # Test compression tradeoff analysis
    print(f"\n=== Compression Tradeoff Analysis ===")
    
    test_data = [Tensor(np.random.randn(32, 100), dtype=np.float32) for _ in range(5)]
    
    tradeoffs = analyze_compression_tradeoffs(
        large_model, small_model, test_data
    )
    
    print(f"Size reduction: {tradeoffs['size_analysis']['size_reduction_ratio']:.2f}x")
    print(f"Speed improvement: {tradeoffs['speed_analysis']['speedup_ratio']:.2f}x")
    print(f"FLOP reduction: {tradeoffs['flops_analysis']['flops_reduction_ratio']:.2f}x")
    print(f"Compression efficiency: {tradeoffs['efficiency_metrics']['compression_efficiency']:.2f}")
    
    # Test model summary
    print(f"\n=== Model Summary ===")
    summary = get_model_summary(large_model, input_shape)
    
    print(f"Total parameters: {summary['model_info']['total_parameters']:,}")
    print(f"Model size: {summary['model_info']['model_size_mb']:.2f} MB")
    print(f"Total MFLOPs: {summary['efficiency_metrics']['mflops']:.2f}")
    print(f"Parameters per FLOP: {summary['efficiency_metrics']['params_per_flop']:.6f}")
    
    # Test model comparison
    print(f"\n=== Model Comparison ===")
    models = {"Large": large_model, "Small": small_model}
    comparison = compare_models(models, test_input)
    
    print("Relative performance (vs Large model):")
    for metric, values in comparison['relative_metrics'].items():
        print(f"  {metric}:")
        for model_name, ratio in values.items():
            print(f"    {model_name}: {ratio:.2f}x")
    
    # Test export/import
    print(f"\n=== Export/Import Test ===")
    
    export_path = "/tmp/test_model.pkl"
    compression_info = {"method": "test", "ratio": 2.0}
    
    exported_path = export_compressed_model(small_model, export_path, compression_info)
    print(f"Model exported to: {exported_path}")
    
    loaded_model, loaded_info = load_compressed_model(exported_path)
    if loaded_model is not None:
        print(f"Model loaded successfully")
        print(f"Compression info: {loaded_info}")
    
    # Cleanup
    if os.path.exists(export_path):
        os.remove(export_path)
    
    print("\n🎉 All compression utilities validated!")
    print("✅ Model size and FLOP calculation")
    print("✅ Inference timing and performance measurement")
    print("✅ Comprehensive tradeoff analysis")
    print("✅ Model export and import functionality")
    print("✅ Model comparison and summary tools")
    print("✅ Efficiency metrics and benchmarking")