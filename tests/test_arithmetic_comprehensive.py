"""Comprehensive tests for arithmetic operations to boost coverage from 5.06%."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.core.tensor import Tensor
from neural_arch.functional.arithmetic import add, sub, mul, div, neg, matmul


class TestArithmeticOperationsComprehensive:
    """Comprehensive tests for all arithmetic operations."""
    
    def test_add_comprehensive(self):
        """Comprehensive tests for addition operation."""
        # Test basic addition
        a = Tensor([[1, 2, 3]], requires_grad=True)
        b = Tensor([[4, 5, 6]], requires_grad=True)
        
        result = add(a, b)
        
        # Check properties
        assert result.requires_grad is True
        assert "add" in result.name
        assert result._grad_fn is not None
        assert result._grad_fn.name == "add"
        
        # Check computation
        expected = np.array([[5, 7, 9]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
        
        # Test broadcasting
        a_broadcast = Tensor([[1, 2, 3]], requires_grad=True)
        b_scalar = Tensor([10], requires_grad=True)
        
        result_broadcast = add(a_broadcast, b_scalar)
        expected_broadcast = np.array([[11, 12, 13]], dtype=np.float32)
        np.testing.assert_array_equal(result_broadcast.data, expected_broadcast)
        
        # Test with different shapes
        a_2d = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b_1d = Tensor([10, 20], requires_grad=True)
        
        result_2d = add(a_2d, b_1d)
        expected_2d = np.array([[11, 22], [13, 24]], dtype=np.float32)
        np.testing.assert_array_equal(result_2d.data, expected_2d)
    
    def test_add_backward_pass(self):
        """Test addition backward pass."""
        a = Tensor([[1, 2]], requires_grad=True)
        b = Tensor([[3, 4]], requires_grad=True)
        
        result = add(a, b)
        
        # Backward pass
        grad_output = np.ones_like(result.data)
        result.backward(grad_output)
        
        # Gradients should be ones (derivative of addition)
        expected_grad = np.ones_like(a.data)
        np.testing.assert_array_equal(a.grad, expected_grad)
        np.testing.assert_array_equal(b.grad, expected_grad)
    
    def test_add_non_tensor_inputs(self):
        """Test addition with non-tensor inputs."""
        # Tensor + scalar
        a = Tensor([[1, 2, 3]], requires_grad=True)
        result = add(a, 5)
        expected = np.array([[6, 7, 8]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
        
        # Scalar + tensor
        result2 = add(10, a)
        expected2 = np.array([[11, 12, 13]], dtype=np.float32)
        np.testing.assert_array_equal(result2.data, expected2)
        
        # List + tensor
        result3 = add([1, 2, 3], a)
        expected3 = np.array([[2, 4, 6]], dtype=np.float32)
        np.testing.assert_array_equal(result3.data, expected3)
    
    def test_add_device_mismatch(self):
        """Test addition with mismatched devices."""
        a = Tensor([[1, 2]], requires_grad=True)
        b = Tensor([[3, 4]], requires_grad=True)
        
        # This should work since both are on CPU by default
        result = add(a, b)
        assert result.device == a.device
    
    def test_sub_comprehensive(self):
        """Comprehensive tests for subtraction operation."""
        # Test basic subtraction
        a = Tensor([[5, 7, 9]], requires_grad=True)
        b = Tensor([[1, 2, 3]], requires_grad=True)
        
        result = sub(a, b)
        
        # Check properties
        assert result.requires_grad is True
        assert "sub" in result.name
        assert result._grad_fn is not None
        assert result._grad_fn.name == "sub"
        
        # Check computation
        expected = np.array([[4, 5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
        
        # Test broadcasting
        a_2d = Tensor([[10, 20], [30, 40]], requires_grad=True)
        b_scalar = Tensor([5], requires_grad=True)
        
        result_broadcast = sub(a_2d, b_scalar)
        expected_broadcast = np.array([[5, 15], [25, 35]], dtype=np.float32)
        np.testing.assert_array_equal(result_broadcast.data, expected_broadcast)
    
    def test_sub_backward_pass(self):
        """Test subtraction backward pass."""
        a = Tensor([[5, 6]], requires_grad=True)
        b = Tensor([[2, 3]], requires_grad=True)
        
        result = sub(a, b)
        
        # Backward pass
        grad_output = np.ones_like(result.data)
        result.backward(grad_output)
        
        # Gradient for a should be +1, for b should be -1
        expected_grad_a = np.ones_like(a.data)
        expected_grad_b = -np.ones_like(b.data)
        np.testing.assert_array_equal(a.grad, expected_grad_a)
        np.testing.assert_array_equal(b.grad, expected_grad_b)
    
    def test_mul_comprehensive(self):
        """Comprehensive tests for multiplication operation."""
        # Test basic multiplication
        a = Tensor([[2, 3, 4]], requires_grad=True)
        b = Tensor([[5, 6, 7]], requires_grad=True)
        
        result = mul(a, b)
        
        # Check properties
        assert result.requires_grad is True
        assert "mul" in result.name
        assert result._grad_fn is not None
        assert result._grad_fn.name == "mul"
        
        # Check computation
        expected = np.array([[10, 18, 28]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
        
        # Test broadcasting
        a_2d = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b_scalar = Tensor([10], requires_grad=True)
        
        result_broadcast = mul(a_2d, b_scalar)
        expected_broadcast = np.array([[10, 20], [30, 40]], dtype=np.float32)
        np.testing.assert_array_equal(result_broadcast.data, expected_broadcast)
    
    def test_mul_backward_pass(self):
        """Test multiplication backward pass."""
        a = Tensor([[2, 3]], requires_grad=True)
        b = Tensor([[5, 7]], requires_grad=True)
        
        result = mul(a, b)
        
        # Backward pass
        grad_output = np.ones_like(result.data)
        result.backward(grad_output)
        
        # Gradient for a should be b, for b should be a
        np.testing.assert_array_equal(a.grad, b.data)
        np.testing.assert_array_equal(b.grad, a.data)
    
    def test_div_comprehensive(self):
        """Comprehensive tests for division operation."""
        # Test basic division
        a = Tensor([[10, 15, 20]], requires_grad=True)
        b = Tensor([[2, 3, 4]], requires_grad=True)
        
        result = div(a, b)
        
        # Check properties
        assert result.requires_grad is True
        assert "div" in result.name
        assert result._grad_fn is not None
        assert result._grad_fn.name == "div"
        
        # Check computation
        expected = np.array([[5, 5, 5]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
        
        # Test broadcasting
        a_2d = Tensor([[12, 18], [24, 30]], requires_grad=True)
        b_scalar = Tensor([6], requires_grad=True)
        
        result_broadcast = div(a_2d, b_scalar)
        expected_broadcast = np.array([[2, 3], [4, 5]], dtype=np.float32)
        np.testing.assert_array_equal(result_broadcast.data, expected_broadcast)
    
    def test_div_backward_pass(self):
        """Test division backward pass."""
        a = Tensor([[12, 8]], requires_grad=True)
        b = Tensor([[3, 2]], requires_grad=True)
        
        result = div(a, b)
        
        # Backward pass
        grad_output = np.ones_like(result.data)
        result.backward(grad_output)
        
        # Gradient for a should be 1/b, for b should be -a/b²
        expected_grad_a = 1.0 / b.data
        expected_grad_b = -a.data / (b.data ** 2)
        np.testing.assert_array_almost_equal(a.grad, expected_grad_a, decimal=6)
        np.testing.assert_array_almost_equal(b.grad, expected_grad_b, decimal=6)
    
    def test_div_zero_error(self):
        """Test division by zero error."""
        a = Tensor([[1, 2, 3]], requires_grad=True)
        b = Tensor([[1, 0, 2]], requires_grad=True)  # Contains zero
        
        with pytest.raises(ValueError) as exc_info:
            div(a, b)
        assert "Division by zero detected" in str(exc_info.value)
    
    def test_neg_comprehensive(self):
        """Comprehensive tests for negation operation."""
        # Test basic negation
        a = Tensor([[1, -2, 3]], requires_grad=True)
        
        result = neg(a)
        
        # Check properties
        assert result.requires_grad is True
        assert "neg" in result.name
        assert result._grad_fn is not None
        assert result._grad_fn.name == "neg"
        
        # Check computation
        expected = np.array([[-1, 2, -3]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
        
        # Test with different shapes
        a_2d = Tensor([[1, -2], [-3, 4]], requires_grad=True)
        result_2d = neg(a_2d)
        expected_2d = np.array([[-1, 2], [3, -4]], dtype=np.float32)
        np.testing.assert_array_equal(result_2d.data, expected_2d)
    
    def test_neg_backward_pass(self):
        """Test negation backward pass."""
        a = Tensor([[1, -2, 3]], requires_grad=True)
        
        result = neg(a)
        
        # Backward pass
        grad_output = np.ones_like(result.data)
        result.backward(grad_output)
        
        # Gradient should be negative of input gradient
        expected_grad = -np.ones_like(a.data)
        np.testing.assert_array_equal(a.grad, expected_grad)
    
    def test_matmul_comprehensive(self):
        """Comprehensive tests for matrix multiplication."""
        # Test basic 2D matrix multiplication
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)  # 2x2
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)  # 2x2
        
        result = matmul(a, b)
        
        # Check properties
        assert result.requires_grad is True
        assert "matmul" in result.name
        assert result._grad_fn is not None
        assert result._grad_fn.name == "matmul"
        
        # Check computation: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
        
        # Test with different shapes
        a_rect = Tensor([[1, 2, 3]], requires_grad=True)  # 1x3
        b_rect = Tensor([[4], [5], [6]], requires_grad=True)  # 3x1
        
        result_rect = matmul(a_rect, b_rect)
        expected_rect = np.array([[32]], dtype=np.float32)  # 1*4 + 2*5 + 3*6 = 32
        np.testing.assert_array_equal(result_rect.data, expected_rect)
    
    def test_matmul_backward_pass(self):
        """Test matrix multiplication backward pass."""
        a = Tensor([[1, 2]], requires_grad=True)  # 1x2
        b = Tensor([[3], [4]], requires_grad=True)  # 2x1
        
        result = matmul(a, b)  # 1x1
        
        # Backward pass (result is scalar)
        result.backward()
        
        # grad_a = grad_output @ b.T = [[1]] @ [[3, 4]] = [[3, 4]]
        # grad_b = a.T @ grad_output = [[1], [2]] @ [[1]] = [[1], [2]]
        expected_grad_a = np.array([[3, 4]], dtype=np.float32)
        expected_grad_b = np.array([[1], [2]], dtype=np.float32)
        
        np.testing.assert_array_equal(a.grad, expected_grad_a)
        np.testing.assert_array_equal(b.grad, expected_grad_b)
    
    def test_matmul_shape_errors(self):
        """Test matrix multiplication shape validation."""
        # Test incompatible dimensions
        a = Tensor([[1, 2, 3]], requires_grad=True)  # 1x3
        b = Tensor([[4, 5]], requires_grad=True)  # 1x2 (incompatible)
        
        with pytest.raises(ValueError) as exc_info:
            matmul(a, b)
        assert "Incompatible matrix dimensions" in str(exc_info.value)
        
        # Test 1D tensors
        a_1d = Tensor([1, 2, 3], requires_grad=True)
        b_2d = Tensor([[4], [5], [6]], requires_grad=True)
        
        with pytest.raises(ValueError) as exc_info:
            matmul(a_1d, b_2d)
        assert "matmul requires 2D+ tensors" in str(exc_info.value)
    
    def test_matmul_higher_dimensions(self):
        """Test matrix multiplication with higher dimensional tensors."""
        # Test 3D tensors (batch matrix multiplication)
        batch_size = 2
        a = Tensor(np.random.randn(batch_size, 2, 3), requires_grad=True)
        b = Tensor(np.random.randn(batch_size, 3, 4), requires_grad=True)
        
        result = matmul(a, b)
        
        # Check shape
        assert result.shape == (batch_size, 2, 4)
        assert result.requires_grad is True
        
        # Manual verification for first batch
        expected_0 = np.matmul(a.data[0], b.data[0])
        np.testing.assert_array_almost_equal(result.data[0], expected_0, decimal=5)


class TestArithmeticEdgeCases:
    """Test edge cases for arithmetic operations."""
    
    def test_operations_without_gradients(self):
        """Test arithmetic operations when gradients are not required."""
        a = Tensor([[1, 2]], requires_grad=False)
        b = Tensor([[3, 4]], requires_grad=False)
        
        # All operations should work without gradients
        result_add = add(a, b)
        result_sub = sub(a, b)
        result_mul = mul(a, b)
        result_div = div(a, b)
        
        # Results should not require gradients
        assert result_add.requires_grad is False
        assert result_sub.requires_grad is False
        assert result_mul.requires_grad is False
        assert result_div.requires_grad is False
        
        # Gradient functions should be None
        assert result_add._grad_fn is None
        assert result_sub._grad_fn is None
        assert result_mul._grad_fn is None
        assert result_div._grad_fn is None
    
    def test_mixed_gradient_requirements(self):
        """Test operations with mixed gradient requirements."""
        a = Tensor([[1, 2]], requires_grad=True)
        b = Tensor([[3, 4]], requires_grad=False)
        
        result = add(a, b)
        
        # Result should require gradients if any input does
        assert result.requires_grad is True
        assert result._grad_fn is not None
        
        # Test backward pass - only a should receive gradients
        grad_output = np.ones_like(result.data)
        result.backward(grad_output)
        
        assert a.grad is not None
        assert b.grad is None
    
    def test_operations_with_extreme_values(self):
        """Test arithmetic operations with extreme values."""
        # Large values
        a_large = Tensor([[1e6, 2e6]], requires_grad=True)
        b_large = Tensor([[3e6, 4e6]], requires_grad=True)
        
        result_add = add(a_large, b_large)
        assert np.all(np.isfinite(result_add.data))
        
        # Small values
        a_small = Tensor([[1e-6, 2e-6]], requires_grad=True)
        b_small = Tensor([[3e-6, 4e-6]], requires_grad=True)
        
        result_mul = mul(a_small, b_small)
        assert np.all(np.isfinite(result_mul.data))
    
    def test_operations_with_zero_tensors(self):
        """Test arithmetic operations with zero tensors."""
        a_zero = Tensor([[0, 0]], requires_grad=True)
        b = Tensor([[1, 2]], requires_grad=True)
        
        # Addition with zero
        result_add = add(a_zero, b)
        np.testing.assert_array_equal(result_add.data, b.data)
        
        # Multiplication with zero
        result_mul = mul(a_zero, b)
        np.testing.assert_array_equal(result_mul.data, np.zeros_like(b.data))
        
        # Division of zero
        result_div = div(a_zero, b)
        np.testing.assert_array_equal(result_div.data, np.zeros_like(b.data))
    
    def test_complex_broadcasting_scenarios(self):
        """Test complex broadcasting scenarios."""
        # Different broadcasting patterns
        a = Tensor([[[1, 2]]], requires_grad=True)  # (1, 1, 2)
        b = Tensor([[3], [4]], requires_grad=True)  # (2, 1)
        
        result = add(a, b)
        
        # Should broadcast to (1, 2, 2)
        expected_shape = (1, 2, 2)
        assert result.shape == expected_shape
        
        # Check values
        expected = np.array([[[4, 5], [5, 6]]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation in complex expressions."""
        a = Tensor([[2, 3]], requires_grad=True)
        b = Tensor([[4, 5]], requires_grad=True)
        
        # Complex expression: (a + b) * (a - b)
        sum_ab = add(a, b)
        diff_ab = sub(a, b)
        result = mul(sum_ab, diff_ab)
        
        # Backward pass
        grad_output = np.ones_like(result.data)
        result.backward(grad_output)
        
        # Gradients should accumulate properly
        assert a.grad is not None
        assert b.grad is not None
        assert np.all(np.isfinite(a.grad))
        assert np.all(np.isfinite(b.grad))


class TestArithmeticGradientComputation:
    """Test gradient computation for arithmetic operations."""
    
    def test_chain_rule_simple(self):
        """Test chain rule with simple arithmetic operations."""
        x = Tensor([[2]], requires_grad=True)
        
        # y = x^2 (implemented as x * x)
        y = mul(x, x)
        
        # Backward pass
        y.backward()
        
        # dy/dx = 2x = 2*2 = 4
        expected_grad = np.array([[4]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad, expected_grad)
    
    def test_chain_rule_complex(self):
        """Test chain rule with complex arithmetic operations."""
        x = Tensor([[3]], requires_grad=True)
        
        # y = (x + 1) * (x - 1) = x^2 - 1
        x_plus_1 = add(x, 1)
        x_minus_1 = sub(x, 1)
        y = mul(x_plus_1, x_minus_1)
        
        # Backward pass
        y.backward()
        
        # dy/dx = 2x = 2*3 = 6
        expected_grad = np.array([[6]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad, expected_grad)
    
    def test_multiple_operations_same_tensor(self):
        """Test gradient accumulation when same tensor is used multiple times."""
        x = Tensor([[2]], requires_grad=True)
        
        # y = x + x + x = 3x
        y1 = add(x, x)
        y2 = add(y1, x)
        
        # Backward pass
        y2.backward()
        
        # dy/dx = 3
        expected_grad = np.array([[3]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad, expected_grad)
    
    def test_matrix_multiplication_gradients(self):
        """Test gradients for matrix multiplication in complex scenarios."""
        # Weight matrix and input
        W = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # 2x3
        x = Tensor([[1], [2], [3]], requires_grad=True)  # 3x1
        
        # Forward pass: y = W @ x
        y = matmul(W, x)  # 2x1
        
        # Loss: sum of outputs
        loss_data = np.sum(y.data)
        loss = Tensor([loss_data], requires_grad=True)
        
        # Backward pass (simulate loss.backward())
        grad_output = np.ones_like(y.data)
        y.backward(grad_output)
        
        # Check gradients
        assert W.grad is not None
        assert x.grad is not None
        
        # grad_W should be grad_output @ x.T
        expected_grad_W = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        np.testing.assert_array_equal(W.grad, expected_grad_W)
        
        # grad_x should be W.T @ grad_output
        expected_grad_x = np.array([[5], [7], [9]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad, expected_grad_x)


class TestArithmeticNumericalStability:
    """Test numerical stability of arithmetic operations."""
    
    def test_addition_numerical_stability(self):
        """Test addition with numerical stability concerns."""
        # Very large numbers
        a = Tensor([[1e10, 2e10]], requires_grad=True)
        b = Tensor([[1e-10, 2e-10]], requires_grad=True)
        
        result = add(a, b)
        
        # Result should be finite and approximately equal to a
        assert np.all(np.isfinite(result.data))
        np.testing.assert_array_almost_equal(result.data, a.data, decimal=5)
    
    def test_multiplication_numerical_stability(self):
        """Test multiplication with numerical stability concerns."""
        # Numbers that could cause overflow
        a = Tensor([[1e5, 2e5]], requires_grad=True)
        b = Tensor([[1e5, 2e5]], requires_grad=True)
        
        result = mul(a, b)
        
        # Result should be finite
        assert np.all(np.isfinite(result.data))
        
        # Check expected values
        expected = np.array([[1e10, 4e10]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.data, expected, decimal=2)
    
    def test_division_numerical_stability(self):
        """Test division with numerical stability concerns."""
        # Very small denominators
        a = Tensor([[1, 2]], requires_grad=True)
        b = Tensor([[1e-7, 2e-7]], requires_grad=True)
        
        result = div(a, b)
        
        # Result should be finite and very large
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data > 1e6)
    
    def test_matmul_numerical_stability(self):
        """Test matrix multiplication numerical stability."""
        # Large matrices with controlled values
        np.random.seed(42)  # For reproducibility
        a = Tensor(np.random.randn(10, 10) * 100, requires_grad=True)
        b = Tensor(np.random.randn(10, 10) * 100, requires_grad=True)
        
        result = matmul(a, b)
        
        # Result should be finite
        assert np.all(np.isfinite(result.data))
        assert result.shape == (10, 10)
        
        # Test gradient computation doesn't overflow
        grad_output = np.ones_like(result.data)
        result.backward(grad_output)
        
        assert np.all(np.isfinite(a.grad))
        assert np.all(np.isfinite(b.grad))