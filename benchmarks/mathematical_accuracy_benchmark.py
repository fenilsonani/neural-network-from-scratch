"""Benchmark mathematical accuracy against reference implementations.

This script compares the accuracy of our mathematical implementations
against reference implementations to validate correctness.
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Any

try:
    import scipy.special
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import from neural architecture framework
from neural_arch.core import Tensor
from neural_arch.functional import gelu, relu, sigmoid, tanh, softmax, mish, silu
from neural_arch.nn import LayerNorm, BatchNorm1d


class MathematicalAccuracyBenchmark:
    """Benchmark mathematical accuracy of implementations."""
    
    def __init__(self):
        self.results = {}
        self.test_sizes = [100, 1000, 10000]
        self.tolerance_levels = {
            'high': 1e-12,      # Machine precision
            'medium': 1e-8,     # High accuracy
            'low': 1e-5         # Practical accuracy
        }
    
    def generate_test_data(self, size: int, range_min: float = -5.0, range_max: float = 5.0) -> np.ndarray:
        """Generate test data with various characteristics."""
        # Mix of different value ranges for comprehensive testing
        data = []
        
        # Normal values
        data.extend(np.random.uniform(range_min, range_max, size // 3))
        
        # Small values (test numerical stability)
        data.extend(np.random.uniform(-1e-3, 1e-3, size // 3))
        
        # Large values (test overflow handling)
        large_vals = np.random.uniform(-50, 50, size - 2 * (size // 3))
        data.extend(large_vals)
        
        return np.array(data)
    
    def benchmark_gelu_accuracy(self) -> Dict[str, Any]:
        """Benchmark GELU accuracy against exact implementation."""
        print("Benchmarking GELU accuracy...")
        
        results = {
            'exact_available': HAS_SCIPY,
            'accuracy_tests': {},
            'performance_tests': {}
        }
        
        for size in self.test_sizes:
            x_data = self.generate_test_data(size, -3.0, 3.0)
            x_tensor = Tensor(x_data)
            
            # Our exact implementation
            start_time = time.time()
            y_exact = gelu(x_tensor, approximate=False)
            exact_time = time.time() - start_time
            
            # Our approximation
            start_time = time.time()
            y_approx = gelu(x_tensor, approximate=True)
            approx_time = time.time() - start_time
            
            if HAS_SCIPY:
                # Reference implementation using scipy
                reference = 0.5 * x_data * (1 + scipy.special.erf(x_data / math.sqrt(2)))
                
                # Compare accuracy
                exact_error = np.max(np.abs(y_exact.data - reference))
                approx_error = np.max(np.abs(y_approx.data - reference))
                
                exact_rmse = np.sqrt(np.mean((y_exact.data - reference) ** 2))
                approx_rmse = np.sqrt(np.mean((y_approx.data - reference) ** 2))
                
                results['accuracy_tests'][size] = {
                    'exact_max_error': float(exact_error),
                    'approx_max_error': float(approx_error),
                    'exact_rmse': float(exact_rmse),
                    'approx_rmse': float(approx_rmse),
                    'accuracy_improvement': float(approx_error / exact_error) if exact_error > 0 else float('inf')
                }
            
            results['performance_tests'][size] = {
                'exact_time': exact_time,
                'approx_time': approx_time,
                'speedup_ratio': approx_time / exact_time if exact_time > 0 else 1.0
            }
        
        return results
    
    def benchmark_activation_functions(self) -> Dict[str, Any]:
        """Benchmark various activation functions."""
        print("Benchmarking activation functions...")
        
        results = {}
        test_data = self.generate_test_data(1000, -5.0, 5.0)
        
        activations = {
            'relu': (relu, lambda x: np.maximum(0, x)),
            'sigmoid': (sigmoid, lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))),
            'tanh': (tanh, lambda x: np.tanh(x)),
            'silu': (silu, lambda x: x / (1 + np.exp(-np.clip(x, -500, 500)))),
        }
        
        for name, (our_func, ref_func) in activations.items():
            x_tensor = Tensor(test_data)
            
            # Our implementation
            start_time = time.time()
            y_ours = our_func(x_tensor)
            our_time = time.time() - start_time
            
            # Reference implementation
            start_time = time.time()
            y_ref = ref_func(test_data)
            ref_time = time.time() - start_time
            
            # Compare accuracy
            max_error = np.max(np.abs(y_ours.data - y_ref))
            rmse = np.sqrt(np.mean((y_ours.data - y_ref) ** 2))
            
            # Classify accuracy
            accuracy_level = 'low'
            for level, tolerance in self.tolerance_levels.items():
                if max_error <= tolerance:
                    accuracy_level = level
                    break
            
            results[name] = {
                'max_error': float(max_error),
                'rmse': float(rmse),
                'accuracy_level': accuracy_level,
                'our_time': our_time,
                'ref_time': ref_time,
                'performance_ratio': our_time / ref_time if ref_time > 0 else 1.0
            }
        
        return results
    
    def benchmark_normalization_layers(self) -> Dict[str, Any]:
        """Benchmark normalization layer accuracy."""
        print("Benchmarking normalization layers...")
        
        results = {}
        
        # Test LayerNorm
        batch_size, features = 32, 128
        x_data = np.random.randn(batch_size, features)
        x_tensor = Tensor(x_data, requires_grad=True)
        
        # Our LayerNorm
        layer_norm = LayerNorm(features, eps=1e-5)
        start_time = time.time()
        y_ours = layer_norm(x_tensor)
        our_time = time.time() - start_time
        
        # Reference LayerNorm (manual implementation)
        mean = np.mean(x_data, axis=-1, keepdims=True)
        var = np.var(x_data, axis=-1, keepdims=True, ddof=0)
        y_ref = (x_data - mean) / np.sqrt(var + 1e-5)
        
        # Compare normalization properties
        our_mean = np.mean(y_ours.data, axis=-1)
        our_var = np.var(y_ours.data, axis=-1, ddof=0)
        ref_mean = np.mean(y_ref, axis=-1)
        ref_var = np.var(y_ref, axis=-1, ddof=0)
        
        results['layernorm'] = {
            'mean_error': float(np.max(np.abs(our_mean - ref_mean))),
            'var_error': float(np.max(np.abs(our_var - ref_var))),
            'output_error': float(np.max(np.abs(y_ours.data - y_ref))),
            'computation_time': our_time,
            'mean_close_to_zero': float(np.max(np.abs(our_mean))),
            'var_close_to_one': float(np.max(np.abs(our_var - 1.0)))
        }
        
        # Test BatchNorm1d
        batch_norm = BatchNorm1d(features)
        batch_norm.train()
        
        start_time = time.time()
        y_bn = batch_norm(x_tensor)
        bn_time = time.time() - start_time
        
        # BatchNorm should normalize across batch dimension
        bn_mean = np.mean(y_bn.data, axis=0)
        bn_var = np.var(y_bn.data, axis=0, ddof=0)
        
        results['batchnorm1d'] = {
            'mean_close_to_zero': float(np.max(np.abs(bn_mean))),
            'var_close_to_one': float(np.max(np.abs(bn_var - 1.0))),
            'computation_time': bn_time,
            'running_stats_updated': batch_norm.num_batches_tracked == 1
        }
        
        return results
    
    def benchmark_gradient_accuracy(self) -> Dict[str, Any]:
        """Benchmark gradient computation accuracy."""
        print("Benchmarking gradient accuracy...")
        
        results = {}
        
        def numerical_gradient(func, x, h=1e-5):
            """Compute numerical gradient."""
            grad = np.zeros_like(x)
            for i in range(x.size):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus.flat[i] += h
                x_minus.flat[i] -= h
                grad.flat[i] = (func(x_plus) - func(x_minus)) / (2 * h)
            return grad
        
        # Test gradient for various functions
        test_functions = {
            'gelu_exact': lambda x: gelu(Tensor([x]), approximate=False).data[0],
            'gelu_approx': lambda x: gelu(Tensor([x]), approximate=True).data[0],
            'relu': lambda x: relu(Tensor([x])).data[0],
            'sigmoid': lambda x: sigmoid(Tensor([x])).data[0],
        }
        
        test_points = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        
        for func_name, func in test_functions.items():
            errors = []
            
            for x_val in test_points:
                # Skip points where function might not be differentiable
                if func_name == 'relu' and abs(x_val) < 1e-10:
                    continue
                
                # Analytical gradient
                x_tensor = Tensor([x_val], requires_grad=True)
                if 'gelu' in func_name:
                    y = gelu(x_tensor, approximate='approx' in func_name)
                elif func_name == 'relu':
                    y = relu(x_tensor)
                elif func_name == 'sigmoid':
                    y = sigmoid(x_tensor)
                
                y.backward()
                analytical_grad = x_tensor.grad[0]
                
                # Numerical gradient
                numerical_grad = numerical_gradient(func, np.array([x_val]))[0]
                
                # Compare
                error = abs(analytical_grad - numerical_grad)
                errors.append(error)
            
            results[func_name] = {
                'max_gradient_error': float(max(errors)) if errors else 0.0,
                'mean_gradient_error': float(np.mean(errors)) if errors else 0.0,
                'points_tested': len(errors)
            }
        
        return results
    
    def benchmark_numerical_stability(self) -> Dict[str, Any]:
        """Benchmark numerical stability with extreme values."""
        print("Benchmarking numerical stability...")
        
        results = {}
        
        # Test with extreme values
        extreme_values = {
            'very_small': [-1e-10, 1e-10],
            'small': [-1e-3, 1e-3],
            'normal': [-1.0, 1.0],
            'large': [-100.0, 100.0],
            'very_large': [-1000.0, 1000.0]
        }
        
        functions_to_test = {
            'sigmoid': sigmoid,
            'tanh': tanh,
            'gelu_exact': lambda x: gelu(x, approximate=False),
            'gelu_approx': lambda x: gelu(x, approximate=True),
            'softmax': lambda x: softmax(x.reshape(1, -1)).reshape(-1) if x.size > 1 else softmax(x.reshape(1, -1))
        }
        
        for range_name, (min_val, max_val) in extreme_values.items():
            range_results = {}
            test_data = np.linspace(min_val, max_val, 100)
            
            for func_name, func in functions_to_test.items():
                try:
                    x_tensor = Tensor(test_data)
                    y = func(x_tensor)
                    
                    # Check for numerical issues
                    has_nan = np.any(np.isnan(y.data))
                    has_inf = np.any(np.isinf(y.data))
                    all_finite = np.all(np.isfinite(y.data))
                    
                    # Check reasonable ranges for specific functions
                    reasonable_range = True
                    if func_name == 'sigmoid':
                        reasonable_range = np.all((y.data >= 0) & (y.data <= 1))
                    elif func_name == 'tanh':
                        reasonable_range = np.all((y.data >= -1) & (y.data <= 1))
                    elif func_name.startswith('softmax'):
                        reasonable_range = np.all(y.data >= 0) and abs(np.sum(y.data) - y.data.size) < 1e-6
                    
                    range_results[func_name] = {
                        'stable': all_finite and not has_nan and not has_inf,
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'reasonable_range': reasonable_range,
                        'min_output': float(np.min(y.data)),
                        'max_output': float(np.max(y.data))
                    }
                    
                except Exception as e:
                    range_results[func_name] = {
                        'stable': False,
                        'error': str(e)
                    }
            
            results[range_name] = range_results
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all mathematical accuracy benchmarks."""
        print("=" * 60)
        print("Neural Architecture Framework - Mathematical Accuracy Benchmark")
        print("=" * 60)
        
        all_results = {}
        
        # Run individual benchmarks
        all_results['gelu_accuracy'] = self.benchmark_gelu_accuracy()
        all_results['activation_functions'] = self.benchmark_activation_functions()
        all_results['normalization_layers'] = self.benchmark_normalization_layers()
        all_results['gradient_accuracy'] = self.benchmark_gradient_accuracy()
        all_results['numerical_stability'] = self.benchmark_numerical_stability()
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("MATHEMATICAL ACCURACY BENCHMARK SUMMARY")
        print("=" * 60)
        
        # GELU accuracy summary
        if 'gelu_accuracy' in results:
            gelu_res = results['gelu_accuracy']
            print(f"\n🔍 GELU Accuracy:")
            print(f"   Reference available: {gelu_res['exact_available']}")
            
            if gelu_res['exact_available'] and gelu_res['accuracy_tests']:
                latest_test = list(gelu_res['accuracy_tests'].values())[-1]
                print(f"   Exact implementation error: {latest_test['exact_rmse']:.2e}")
                print(f"   Approximation error: {latest_test['approx_rmse']:.2e}")
                print(f"   Accuracy improvement: {latest_test['accuracy_improvement']:.1f}x")
        
        # Activation functions summary
        if 'activation_functions' in results:
            print(f"\n⚡ Activation Functions:")
            act_res = results['activation_functions']
            for name, data in act_res.items():
                print(f"   {name.upper()}: {data['accuracy_level']} accuracy (error: {data['max_error']:.2e})")
        
        # Normalization layers summary
        if 'normalization_layers' in results:
            print(f"\n📊 Normalization Layers:")
            norm_res = results['normalization_layers']
            if 'layernorm' in norm_res:
                ln_res = norm_res['layernorm']
                print(f"   LayerNorm: mean error {ln_res['mean_error']:.2e}, var error {ln_res['var_error']:.2e}")
            if 'batchnorm1d' in norm_res:
                bn_res = norm_res['batchnorm1d']
                print(f"   BatchNorm1d: stats updated {bn_res['running_stats_updated']}")
        
        # Gradient accuracy summary
        if 'gradient_accuracy' in results:
            print(f"\n∇ Gradient Accuracy:")
            grad_res = results['gradient_accuracy']
            for name, data in grad_res.items():
                print(f"   {name}: max error {data['max_gradient_error']:.2e}")
        
        # Numerical stability summary
        if 'numerical_stability' in results:
            print(f"\n🛡️  Numerical Stability:")
            stab_res = results['numerical_stability']
            stable_count = 0
            total_count = 0
            
            for range_name, range_data in stab_res.items():
                for func_name, func_data in range_data.items():
                    total_count += 1
                    if func_data.get('stable', False):
                        stable_count += 1
            
            stability_rate = (stable_count / total_count) * 100 if total_count > 0 else 0
            print(f"   Overall stability: {stability_rate:.1f}% ({stable_count}/{total_count})")
        
        print("\n✅ Benchmark completed successfully!")


def main():
    """Run the mathematical accuracy benchmark."""
    benchmark = MathematicalAccuracyBenchmark()
    
    try:
        results = benchmark.run_all_benchmarks()
        benchmark.print_summary(results)
        
        # Save results to file
        import json
        with open('mathematical_accuracy_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"\n📄 Detailed results saved to: mathematical_accuracy_results.json")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()