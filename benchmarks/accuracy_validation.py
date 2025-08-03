"""Accuracy Validation and Numerical Precision Testing.

This module provides comprehensive accuracy validation tools to ensure
Neural Forge produces numerically correct results compared to PyTorch.
"""

import os
import sys
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.neural_arch.core.tensor import Tensor
from src.neural_arch.nn.module import Module
from src.neural_arch.nn import Sequential, Linear, ReLU, Conv2d
from src.neural_arch.functional import relu, sigmoid, tanh, softmax

logger = logging.getLogger(__name__)

# Try to import PyTorch for comparison
try:
    import torch
    import torch.nn as torch_nn
    import torch.nn.functional as torch_F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AccuracyLevel(Enum):
    """Accuracy levels for numerical comparisons."""
    EXACT = "exact"          # Exact match (for integer operations)
    HIGH = "high"            # 1e-6 tolerance
    MEDIUM = "medium"        # 1e-4 tolerance  
    LOW = "low"             # 1e-2 tolerance
    LOOSE = "loose"         # 1e-1 tolerance


@dataclass
class ValidationResult:
    """Results from accuracy validation."""
    
    # Test identification
    test_name: str
    framework_comparison: str  # e.g., "neural_forge_vs_pytorch"
    operation: str
    
    # Accuracy metrics
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    mean_relative_error: float
    
    # Statistical measures
    correlation_coefficient: float
    cosine_similarity: float
    normalized_rmse: float
    
    # Pass/fail status
    accuracy_level: AccuracyLevel
    tolerance_used: float
    passed: bool
    
    # Shape and data info
    output_shape: Tuple[int, ...]
    data_type: str
    num_elements: int
    
    # Additional analysis
    error_distribution: Optional[Dict[str, float]] = None
    problematic_indices: Optional[List[int]] = None
    
    # Metadata
    timestamp: str = ""
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == "":
            import time
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


class AccuracyValidator:
    """Comprehensive accuracy validation suite."""
    
    def __init__(self):
        """Initialize accuracy validator."""
        self.results = []
        self.tolerance_map = {
            AccuracyLevel.EXACT: 0.0,
            AccuracyLevel.HIGH: 1e-6,
            AccuracyLevel.MEDIUM: 1e-4,
            AccuracyLevel.LOW: 1e-2,
            AccuracyLevel.LOOSE: 1e-1
        }
    
    def compare_outputs(self,
                       neural_forge_output: Tensor,
                       pytorch_output: Any,
                       test_name: str,
                       operation: str,
                       expected_accuracy: AccuracyLevel = AccuracyLevel.MEDIUM) -> ValidationResult:
        """Compare outputs between Neural Forge and PyTorch.
        
        Args:
            neural_forge_output: Output from Neural Forge
            pytorch_output: Output from PyTorch
            test_name: Name of the test
            operation: Operation being tested
            expected_accuracy: Expected accuracy level
            
        Returns:
            ValidationResult with detailed comparison
        """
        
        if not TORCH_AVAILABLE:
            return self._create_skip_result(test_name, operation, "PyTorch not available")
        
        # Convert PyTorch tensor to numpy for comparison
        if hasattr(pytorch_output, 'detach'):
            pytorch_data = pytorch_output.detach().cpu().numpy()
        else:
            pytorch_data = pytorch_output
        
        neural_forge_data = neural_forge_output.data
        
        # Ensure same shape
        if neural_forge_data.shape != pytorch_data.shape:
            logger.error(f"Shape mismatch: NF {neural_forge_data.shape} vs PT {pytorch_data.shape}")
            return self._create_fail_result(test_name, operation, "Shape mismatch")
        
        # Calculate error metrics
        absolute_errors = np.abs(neural_forge_data - pytorch_data)
        max_abs_error = np.max(absolute_errors)
        mean_abs_error = np.mean(absolute_errors)
        
        # Relative errors (avoid division by zero)
        pytorch_abs = np.abs(pytorch_data)
        relative_errors = np.where(
            pytorch_abs > 1e-12,
            absolute_errors / pytorch_abs,
            0.0
        )
        max_rel_error = np.max(relative_errors)
        mean_rel_error = np.mean(relative_errors)
        
        # Statistical measures
        correlation = self._calculate_correlation(neural_forge_data, pytorch_data)
        cosine_sim = self._calculate_cosine_similarity(neural_forge_data, pytorch_data)
        normalized_rmse = self._calculate_normalized_rmse(neural_forge_data, pytorch_data)
        
        # Determine if test passed
        tolerance = self.tolerance_map[expected_accuracy]
        passed = max_abs_error <= tolerance
        
        # Error distribution analysis
        error_distribution = self._analyze_error_distribution(absolute_errors)
        
        # Find problematic indices
        problematic_indices = self._find_problematic_indices(absolute_errors, tolerance)
        
        result = ValidationResult(
            test_name=test_name,
            framework_comparison="neural_forge_vs_pytorch",
            operation=operation,
            max_absolute_error=max_abs_error,
            mean_absolute_error=mean_abs_error,
            max_relative_error=max_rel_error,
            mean_relative_error=mean_rel_error,
            correlation_coefficient=correlation,
            cosine_similarity=cosine_sim,
            normalized_rmse=normalized_rmse,
            accuracy_level=expected_accuracy,
            tolerance_used=tolerance,
            passed=passed,
            output_shape=neural_forge_data.shape,
            data_type=str(neural_forge_data.dtype),
            num_elements=neural_forge_data.size,
            error_distribution=error_distribution,
            problematic_indices=problematic_indices
        )
        
        self.results.append(result)
        
        if not passed:
            logger.warning(f"Accuracy test FAILED: {test_name} - Max error: {max_abs_error:.2e} > {tolerance:.2e}")
        else:
            logger.debug(f"Accuracy test PASSED: {test_name} - Max error: {max_abs_error:.2e}")
        
        return result
    
    def validate_operation(self,
                          operation_name: str,
                          neural_forge_op: Callable,
                          pytorch_op: Callable,
                          test_inputs: List[Any],
                          expected_accuracy: AccuracyLevel = AccuracyLevel.MEDIUM) -> List[ValidationResult]:
        """Validate a specific operation across multiple inputs.
        
        Args:
            operation_name: Name of the operation
            neural_forge_op: Neural Forge operation function
            pytorch_op: PyTorch operation function
            test_inputs: List of test inputs
            expected_accuracy: Expected accuracy level
            
        Returns:
            List of validation results
        """
        
        results = []
        
        for i, test_input in enumerate(test_inputs):
            test_name = f"{operation_name}_test_{i}"
            
            try:
                # Run Neural Forge operation
                if isinstance(test_input, tuple):
                    nf_input = tuple(Tensor(inp) if isinstance(inp, np.ndarray) else inp for inp in test_input)
                    nf_output = neural_forge_op(*nf_input)
                else:
                    nf_input = Tensor(test_input) if isinstance(test_input, np.ndarray) else test_input
                    nf_output = neural_forge_op(nf_input)
                
                # Run PyTorch operation
                if TORCH_AVAILABLE:
                    if isinstance(test_input, tuple):
                        pt_input = tuple(torch.from_numpy(inp) if isinstance(inp, np.ndarray) else inp for inp in test_input)
                        pt_output = pytorch_op(*pt_input)
                    else:
                        pt_input = torch.from_numpy(test_input) if isinstance(test_input, np.ndarray) else test_input
                        pt_output = pytorch_op(pt_input)
                    
                    # Compare outputs
                    result = self.compare_outputs(nf_output, pt_output, test_name, operation_name, expected_accuracy)
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Validation failed for {test_name}: {e}")
                result = self._create_fail_result(test_name, operation_name, str(e))
                results.append(result)
        
        return results
    
    def validate_model(self,
                      neural_forge_model: Module,
                      pytorch_model: Any,
                      test_inputs: List[Tensor],
                      model_name: str,
                      expected_accuracy: AccuracyLevel = AccuracyLevel.MEDIUM) -> List[ValidationResult]:
        """Validate a complete model."""
        
        results = []
        
        for i, test_input in enumerate(test_inputs):
            test_name = f"{model_name}_model_test_{i}"
            
            try:
                # Neural Forge inference
                neural_forge_model.eval()
                nf_output = neural_forge_model(test_input)
                
                # PyTorch inference
                if TORCH_AVAILABLE and pytorch_model is not None:
                    pytorch_model.eval()
                    
                    # Convert input to PyTorch tensor
                    pt_input = torch.from_numpy(test_input.data).float()
                    
                    with torch.no_grad():
                        pt_output = pytorch_model(pt_input)
                    
                    # Compare outputs
                    result = self.compare_outputs(nf_output, pt_output, test_name, f"{model_name}_inference", expected_accuracy)
                    results.append(result)
                    
                else:
                    logger.warning("PyTorch model not available for comparison")
                    
            except Exception as e:
                logger.error(f"Model validation failed for {test_name}: {e}")
                result = self._create_fail_result(test_name, f"{model_name}_inference", str(e))
                results.append(result)
        
        return results
    
    def _calculate_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate correlation coefficient between two arrays."""
        try:
            a_flat = a.flatten()
            b_flat = b.flatten()
            
            if len(a_flat) < 2:
                return 1.0
            
            corr_matrix = np.corrcoef(a_flat, b_flat)
            return float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        except:
            return 0.0
    
    def _calculate_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two arrays."""
        try:
            a_flat = a.flatten()
            b_flat = b.flatten()
            
            norm_a = np.linalg.norm(a_flat)
            norm_b = np.linalg.norm(b_flat)
            
            if norm_a == 0 or norm_b == 0:
                return 1.0 if np.allclose(a_flat, b_flat) else 0.0
            
            return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
        except:
            return 0.0
    
    def _calculate_normalized_rmse(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate normalized root mean square error."""
        try:
            rmse = np.sqrt(np.mean((a - b) ** 2))
            data_range = np.max(b) - np.min(b)
            
            if data_range == 0:
                return 0.0 if np.allclose(a, b) else float('inf')
            
            return float(rmse / data_range)
        except:
            return float('inf')
    
    def _analyze_error_distribution(self, errors: np.ndarray) -> Dict[str, float]:
        """Analyze distribution of errors."""
        try:
            return {
                "min": float(np.min(errors)),
                "max": float(np.max(errors)),
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "median": float(np.median(errors)),
                "p95": float(np.percentile(errors, 95)),
                "p99": float(np.percentile(errors, 99))
            }
        except:
            return {}
    
    def _find_problematic_indices(self, errors: np.ndarray, tolerance: float) -> List[int]:
        """Find indices where errors exceed tolerance."""
        try:
            problematic = np.where(errors > tolerance)[0]
            return problematic.tolist()[:10]  # Return first 10 problematic indices
        except:
            return []
    
    def _create_skip_result(self, test_name: str, operation: str, reason: str) -> ValidationResult:
        """Create a result for skipped tests."""
        return ValidationResult(
            test_name=test_name,
            framework_comparison="neural_forge_vs_pytorch",
            operation=operation,
            max_absolute_error=0.0,
            mean_absolute_error=0.0,
            max_relative_error=0.0,
            mean_relative_error=0.0,
            correlation_coefficient=1.0,
            cosine_similarity=1.0,
            normalized_rmse=0.0,
            accuracy_level=AccuracyLevel.EXACT,
            tolerance_used=0.0,
            passed=False,
            output_shape=(),
            data_type="unknown",
            num_elements=0
        )
    
    def _create_fail_result(self, test_name: str, operation: str, error_message: str) -> ValidationResult:
        """Create a result for failed tests."""
        return ValidationResult(
            test_name=test_name,
            framework_comparison="neural_forge_vs_pytorch",
            operation=operation,
            max_absolute_error=float('inf'),
            mean_absolute_error=float('inf'),
            max_relative_error=float('inf'),
            mean_relative_error=float('inf'),
            correlation_coefficient=0.0,
            cosine_similarity=0.0,
            normalized_rmse=float('inf'),
            accuracy_level=AccuracyLevel.LOOSE,
            tolerance_used=1e-1,
            passed=False,
            output_shape=(),
            data_type="error",
            num_elements=0
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        
        if not self.results:
            return {"error": "No validation results available"}
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        summary = {
            "total_tests": len(self.results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "pass_rate": len(passed_tests) / len(self.results) if self.results else 0.0,
            
            # Accuracy statistics
            "accuracy_stats": {
                "max_absolute_errors": [r.max_absolute_error for r in self.results if r.max_absolute_error != float('inf')],
                "mean_absolute_errors": [r.mean_absolute_error for r in self.results if r.mean_absolute_error != float('inf')],
                "correlations": [r.correlation_coefficient for r in self.results],
                "cosine_similarities": [r.cosine_similarity for r in self.results]
            },
            
            # Failed test details
            "failed_test_details": [
                {
                    "test_name": r.test_name,
                    "operation": r.operation,
                    "max_error": r.max_absolute_error,
                    "tolerance": r.tolerance_used
                }
                for r in failed_tests
            ]
        }
        
        # Calculate aggregate statistics
        if summary["accuracy_stats"]["max_absolute_errors"]:
            max_errors = summary["accuracy_stats"]["max_absolute_errors"]
            summary["aggregate_accuracy"] = {
                "worst_max_error": max(max_errors),
                "best_max_error": min(max_errors),
                "avg_max_error": np.mean(max_errors),
                "median_max_error": np.median(max_errors)
            }
        
        return summary
    
    def save_results(self, filepath: str):
        """Save validation results to file."""
        
        # Convert results to dictionaries
        results_data = [asdict(result) for result in self.results]
        
        data = {
            "validation_results": results_data,
            "summary": self.get_validation_summary(),
            "metadata": {
                "total_results": len(self.results),
                "pytorch_available": TORCH_AVAILABLE,
                "timestamp": self.results[0].timestamp if self.results else ""
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Validation results saved to {filepath}")


# Convenience functions
def validate_model_accuracy(neural_forge_model: Module,
                          pytorch_model: Any,
                          input_shapes: List[Tuple[int, ...]],
                          model_name: str = "model",
                          expected_accuracy: AccuracyLevel = AccuracyLevel.MEDIUM) -> AccuracyValidator:
    """Validate model accuracy across multiple input shapes."""
    
    validator = AccuracyValidator()
    
    # Generate test inputs
    test_inputs = []
    for shape in input_shapes:
        test_input = Tensor(np.random.randn(*shape).astype(np.float32))
        test_inputs.append(test_input)
    
    # Validate model
    validator.validate_model(neural_forge_model, pytorch_model, test_inputs, model_name, expected_accuracy)
    
    return validator


def compare_numerical_precision(operations: List[Dict[str, Any]],
                              expected_accuracy: AccuracyLevel = AccuracyLevel.MEDIUM) -> AccuracyValidator:
    """Compare numerical precision of specific operations."""
    
    validator = AccuracyValidator()
    
    for op_config in operations:
        op_name = op_config["name"]
        nf_op = op_config["neural_forge_op"]
        pt_op = op_config["pytorch_op"]
        test_inputs = op_config["test_inputs"]
        
        validator.validate_operation(op_name, nf_op, pt_op, test_inputs, expected_accuracy)
    
    return validator


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Forge Accuracy Validation...")
    
    # Test basic operations
    print(f"\n=== Testing Basic Operations ===")
    
    # Test ReLU activation
    test_inputs = [
        np.random.randn(32, 100).astype(np.float32),
        np.random.randn(64, 256).astype(np.float32),
        np.array([[-1, 0, 1, 2, -2]]).astype(np.float32)  # Known values
    ]
    
    operations = [
        {
            "name": "relu_activation",
            "neural_forge_op": relu,
            "pytorch_op": torch_F.relu if TORCH_AVAILABLE else lambda x: x,
            "test_inputs": test_inputs
        }
    ]
    
    if TORCH_AVAILABLE:
        precision_validator = compare_numerical_precision(operations, AccuracyLevel.HIGH)
        
        print("ReLU Validation Results:")
        for result in precision_validator.results:
            print(f"  {result.test_name}: {'PASS' if result.passed else 'FAIL'}")
            print(f"    Max error: {result.max_absolute_error:.2e}")
            print(f"    Correlation: {result.correlation_coefficient:.6f}")
    
    # Test model validation
    print(f"\n=== Testing Model Validation ===")
    
    neural_forge_model = Sequential(
        Linear(100, 50),
        ReLU(),
        Linear(50, 10)
    )
    
    pytorch_model = None
    if TORCH_AVAILABLE:
        pytorch_model = torch_nn.Sequential(
            torch_nn.Linear(100, 50),
            torch_nn.ReLU(),
            torch_nn.Linear(50, 10)
        )
        
        # Copy weights to ensure same initialization
        with torch.no_grad():
            pytorch_model[0].weight.copy_(torch.from_numpy(neural_forge_model.layers[0].weight.data))
            pytorch_model[0].bias.copy_(torch.from_numpy(neural_forge_model.layers[0].bias.data))
            pytorch_model[2].weight.copy_(torch.from_numpy(neural_forge_model.layers[2].weight.data))
            pytorch_model[2].bias.copy_(torch.from_numpy(neural_forge_model.layers[2].bias.data))
    
    input_shapes = [(32, 100), (64, 100)]
    model_validator = validate_model_accuracy(
        neural_forge_model, pytorch_model, input_shapes, "linear_model", AccuracyLevel.MEDIUM
    )
    
    # Print validation summary
    summary = model_validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Passed tests: {summary['passed_tests']}")
    print(f"  Pass rate: {summary['pass_rate']:.1%}")
    
    if summary.get('aggregate_accuracy'):
        agg = summary['aggregate_accuracy']
        print(f"  Worst max error: {agg['worst_max_error']:.2e}")
        print(f"  Average max error: {agg['avg_max_error']:.2e}")
    
    # Save results
    model_validator.save_results("/tmp/neural_forge_accuracy_validation.json")
    
    print("\n🎉 Accuracy validation completed!")
    print("✅ Operation-level numerical precision testing")
    print("✅ Model-level accuracy validation")
    print("✅ Statistical accuracy analysis")
    print("✅ Error distribution analysis")
    print("✅ Cross-framework numerical comparison")
    print("✅ Comprehensive accuracy reporting")