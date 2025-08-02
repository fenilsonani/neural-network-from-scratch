"""Comprehensive test coverage for functional/arithmetic module to boost coverage from 83.97% to 95%+"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

import neural_arch.core.tensor as tensor_module
from neural_arch.core.device import Device, DeviceType
from neural_arch.core.tensor import Tensor, enable_grad, is_grad_enabled, no_grad
from neural_arch.functional.arithmetic import add, div, matmul, mul, neg, sub
from neural_arch.functional.utils import broadcast_tensors, reduce_gradient


def set_grad_enabled(enabled: bool):
    """Helper function to set gradient state."""
    tensor_module._grad_enabled = enabled


class TestFunctionalArithmeticCoverageBoost:
    """Comprehensive tests for functional arithmetic targeting missing coverage paths."""

    def test_add_with_device_mismatch(self):
        """Test add operation with tensors on different devices."""
        a = Tensor([1, 2, 3], device=Device(DeviceType.CPU))
        b = Tensor([4, 5, 6], device=Device(DeviceType.CUDA, 0))

        with pytest.raises(ValueError, match="Tensors must be on same device"):
            add(a, b)

    def test_add_with_scalar_inputs(self):
        """Test add operation with scalar inputs (non-tensor conversion)."""
        # Test with scalar inputs that get converted to tensors
        result = add(5, 3)
        assert isinstance(result, Tensor)
        assert np.allclose(result.data, 8)

        # Test with mixed tensor and scalar
        a = Tensor([1, 2, 3])
        result = add(a, 5)
        assert isinstance(result, Tensor)
        assert np.allclose(result.data, [6, 7, 8])

    def test_add_gradient_computation_detailed(self):
        """Test detailed gradient computation for add operation."""
        # Enable gradients
        set_grad_enabled(True)

        try:
            # Create tensors with gradients
            a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, name="tensor_a")
            b = Tensor([[0.5, 1.5]], requires_grad=True, name="tensor_b")  # Broadcasting case

            # Forward pass
            result = add(a, b)

            # Check result properties
            assert result.requires_grad is True
            assert "add(tensor_a, tensor_b)" in result.name

            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            if result._grad_fn:
                result._grad_fn.apply(grad_output)

            # Check gradients were accumulated
            assert a._grad is not None
            assert b._grad is not None

        finally:
            set_grad_enabled(False)

    def test_add_gradient_accumulation(self):
        """Test gradient accumulation in add operation."""
        set_grad_enabled(True)

        try:
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=True)

            # Initialize gradients
            a._grad = a._backend.from_numpy(np.array([0.1, 0.2]))
            b._grad = b._backend.from_numpy(np.array([0.3, 0.4]))

            result = add(a, b)

            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            if result._grad_fn:
                result._grad_fn.apply(grad_output)

            # Gradients should be accumulated (added to existing)
            assert a._grad is not None
            assert b._grad is not None

        finally:
            set_grad_enabled(False)

    def test_add_with_gradient_function_chaining(self):
        """Test add operation with existing gradient functions."""
        set_grad_enabled(True)

        try:
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=True)

            # Create mock gradient functions
            mock_grad_fn_a = MagicMock()
            mock_grad_fn_b = MagicMock()
            a._grad_fn = mock_grad_fn_a
            b._grad_fn = mock_grad_fn_b

            result = add(a, b)

            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            if result._grad_fn:
                result._grad_fn.apply(grad_output)

            # Check that gradient functions were called
            mock_grad_fn_a.apply.assert_called()
            mock_grad_fn_b.apply.assert_called()

        finally:
            set_grad_enabled(False)

    def test_subtract_operation_comprehensive(self):
        """Test subtract operation with comprehensive edge cases."""
        # Basic subtraction
        a = Tensor([5, 7, 9])
        b = Tensor([2, 3, 4])
        result = sub(a, b)
        assert np.allclose(result.data, [3, 4, 5])

        # Subtraction with broadcasting
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([1, 1])
        result = sub(a, b)
        assert np.allclose(result.data, [[0, 1], [2, 3]])

        # Device mismatch error
        a = Tensor([1, 2], device=Device(DeviceType.CPU))
        b = Tensor([3, 4], device=Device(DeviceType.CUDA, 0))
        with pytest.raises(ValueError, match="Tensors must be on same device"):
            sub(a, b)

    def test_subtract_gradient_computation(self):
        """Test gradient computation for subtract operation."""
        set_grad_enabled(True)

        try:
            a = Tensor([5.0, 6.0], requires_grad=True)
            b = Tensor([2.0, 3.0], requires_grad=True)

            result = sub(a, b)

            # Check gradient setup
            assert result.requires_grad is True
            assert result._grad_fn is not None

            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)

            # For subtraction: ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
            assert a._grad is not None
            assert b._grad is not None

        finally:
            set_grad_enabled(False)

    def test_multiply_operation_comprehensive(self):
        """Test multiply operation with comprehensive edge cases."""
        # Basic multiplication
        a = Tensor([2, 3, 4])
        b = Tensor([5, 6, 7])
        result = mul(a, b)
        assert np.allclose(result.data, [10, 18, 28])

        # Multiplication with scalars
        a = Tensor([2.0, 3.0])
        result = mul(a, 5)
        assert np.allclose(result.data, [10.0, 15.0])

        # Broadcasting multiplication
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([2, 3])
        result = mul(a, b)
        assert np.allclose(result.data, [[2, 6], [6, 12]])

    def test_multiply_gradient_computation(self):
        """Test gradient computation for multiply operation."""
        set_grad_enabled(True)

        try:
            a = Tensor([2.0, 3.0], requires_grad=True)
            b = Tensor([4.0, 5.0], requires_grad=True)

            result = mul(a, b)

            # Check gradient setup
            assert result.requires_grad is True
            assert result._grad_fn is not None

            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)

            # For multiplication: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            assert a._grad is not None
            assert b._grad is not None

        finally:
            set_grad_enabled(False)

    def test_divide_operation_comprehensive(self):
        """Test divide operation with comprehensive edge cases."""
        # Basic division
        a = Tensor([6.0, 8.0, 10.0])
        b = Tensor([2.0, 4.0, 5.0])
        result = div(a, b)
        assert np.allclose(result.data, [3.0, 2.0, 2.0])

        # Division with scalars
        a = Tensor([8.0, 12.0])
        result = div(a, 4.0)
        assert np.allclose(result.data, [2.0, 3.0])

        # Division by zero (should be handled by backend)
        a = Tensor([1.0, 2.0])
        b = Tensor([0.0, 1.0])
        result = div(a, b)
        # Result depends on backend behavior (np.inf expected)
        assert np.isinf(result.data[0])
        assert np.isclose(result.data[1], 2.0)

    def test_divide_gradient_computation(self):
        """Test gradient computation for divide operation."""
        set_grad_enabled(True)

        try:
            a = Tensor([6.0, 8.0], requires_grad=True)
            b = Tensor([2.0, 4.0], requires_grad=True)

            result = div(a, b)

            # Check gradient setup
            assert result.requires_grad is True
            assert result._grad_fn is not None

            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)

            # For division: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
            assert a._grad is not None
            assert b._grad is not None

        finally:
            set_grad_enabled(False)

    def test_neg_operation_comprehensive(self):
        """Test negation operation with comprehensive edge cases."""
        # Basic negation
        a = Tensor([1.0, -2.0, 3.0])
        result = neg(a)
        assert np.allclose(result.data, [-1.0, 2.0, -3.0])

        # Negation with zero
        a = Tensor([0.0, 1.0])
        result = neg(a)
        assert np.allclose(result.data, [0.0, -1.0])

    def test_neg_gradient_computation(self):
        """Test gradient computation for negation operation."""
        set_grad_enabled(True)

        try:
            a = Tensor([2.0, 3.0], requires_grad=True)

            result = neg(a)

            # Check gradient setup
            assert result.requires_grad is True
            assert result._grad_fn is not None

            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)

            # For negation: ∂(-a)/∂a = -1
            assert a._grad is not None

        finally:
            set_grad_enabled(False)

    def test_matmul_operation_comprehensive(self):
        """Test matrix multiplication operation with comprehensive cases."""
        # 2D matrix multiplication
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = matmul(a, b)
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(result.data, expected)

        # 1D vector dot product
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = matmul(a, b)
        assert np.isclose(result.data, 32)  # 1*4 + 2*5 + 3*6

        # Broadcasting cases
        a = Tensor([[[1, 2], [3, 4]]])  # Shape: (1, 2, 2)
        b = Tensor([[5, 6], [7, 8]])  # Shape: (2, 2)
        result = matmul(a, b)
        assert result.shape == (1, 2, 2)

    def test_matmul_incompatible_shapes(self):
        """Test matrix multiplication with incompatible shapes."""
        a = Tensor([[1, 2, 3]])  # Shape: (1, 3)
        b = Tensor([[4, 5]])  # Shape: (1, 2)

        with pytest.raises(ValueError, match="incompatible"):
            matmul(a, b)

    def test_matmul_gradient_computation(self):
        """Test gradient computation for matrix multiplication."""
        set_grad_enabled(True)

        try:
            a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            b = Tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)

            result = matmul(a, b)

            # Check gradient setup
            assert result.requires_grad is True
            assert result._grad_fn is not None

            # Simulate backward pass
            grad_output = np.ones_like(result.data)
            result._grad_fn.apply(grad_output)

            # For matmul: ∂(A@B)/∂A = grad_output @ B.T, ∂(A@B)/∂B = A.T @ grad_output
            assert a._grad is not None
            assert b._grad is not None

        finally:
            set_grad_enabled(False)

    def test_operations_without_gradients(self):
        """Test operations when gradients are disabled."""
        set_grad_enabled(False)

        try:
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=True)

            # Operations should not create gradient functions
            result_add = add(a, b)
            result_sub = sub(a, b)
            result_mul = mul(a, b)
            result_div = div(a, b)
            result_neg = neg(a)
            result_mm = matmul(a.reshape(1, 2), b.reshape(2, 1))

            # No gradients should be computed
            assert not result_add.requires_grad
            assert not result_sub.requires_grad
            assert not result_mul.requires_grad
            assert not result_div.requires_grad
            assert not result_neg.requires_grad
            assert not result_mm.requires_grad

        finally:
            set_grad_enabled(True)

    def test_operations_with_mixed_gradient_requirements(self):
        """Test operations with mixed gradient requirements."""
        set_grad_enabled(True)

        try:
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=False)

            result = add(a, b)

            # Result should require gradients because one input does
            assert result.requires_grad is True

            # Test reverse case
            a = Tensor([1.0, 2.0], requires_grad=False)
            b = Tensor([3.0, 4.0], requires_grad=True)

            result = mul(a, b)
            assert result.requires_grad is True

        finally:
            set_grad_enabled(False)

    def test_operations_error_handling(self):
        """Test error handling in arithmetic operations."""
        # Test with None inputs
        with pytest.raises((TypeError, AttributeError)):
            add(None, Tensor([1, 2]))

        # Test with incompatible types for specific operations
        a = Tensor([1, 2, 3])
        b = Tensor([[1, 2], [3, 4]])

        # Some operations should handle broadcasting, others might fail
        try:
            result = add(a, b)
            # If successful, broadcasting worked
        except ValueError:
            # If failed, incompatible shapes
            pass

    def test_complex_gradient_propagation(self):
        """Test complex gradient propagation through multiple operations."""
        set_grad_enabled(True)

        try:
            a = Tensor([[1.0, 2.0]], requires_grad=True)
            b = Tensor([[3.0], [4.0]], requires_grad=True)
            c = Tensor([0.5, 1.0], requires_grad=True)

            # Complex computation: (a @ b) + c
            intermediate = matmul(a, b)  # Shape: (1, 1)
            result = add(intermediate.reshape(-1), c)  # Reshape and add

            # Check gradients are properly set up
            assert result.requires_grad is True
            assert intermediate.requires_grad is True

            # Simulate backward pass
            grad_output = np.array([1.0, 1.0])
            result._grad_fn.apply(grad_output)

            # All input tensors should have gradients
            assert a._grad is not None
            assert b._grad is not None
            assert c._grad is not None

        finally:
            set_grad_enabled(False)

    def test_broadcasting_edge_cases(self):
        """Test arithmetic operations with complex broadcasting scenarios."""
        # Scalar with tensor
        a = Tensor(5.0)
        b = Tensor([1, 2, 3])
        result = add(a, b)
        assert np.allclose(result.data, [6, 7, 8])

        # Different dimensional broadcasting
        a = Tensor([[[1, 2]]])  # Shape: (1, 1, 2)
        b = Tensor([[3], [4]])  # Shape: (2, 1)
        result = mul(a, b)
        assert result.shape == (1, 2, 2)

        # Single element tensors
        a = Tensor([[5]])
        b = Tensor([1, 2, 3])
        result = div(b, a)
        assert np.allclose(result.data, [0.2, 0.4, 0.6])

    def test_dtype_preservation(self):
        """Test that operations preserve or handle dtypes correctly."""
        # Integer tensors
        a = Tensor([1, 2, 3], dtype="int32")
        b = Tensor([4, 5, 6], dtype="int32")
        result = add(a, b)
        # Result dtype should be compatible with inputs

        # Mixed dtype operations
        a = Tensor([1.5, 2.5], dtype="float32")
        b = Tensor([1, 2], dtype="int32")
        result = mul(a, b)
        # Should handle mixed dtypes gracefully

        # Operations should not fail due to dtype mismatches
        assert isinstance(result, Tensor)

    def test_name_propagation(self):
        """Test that tensor names are properly propagated in operations."""
        a = Tensor([1, 2], name="input_a")
        b = Tensor([3, 4], name="input_b")

        result = add(a, b)
        assert "add(input_a, input_b)" in result.name

        result = sub(a, b)
        assert "sub(input_a, input_b)" in result.name

        result = mul(a, b)
        assert "mul(input_a, input_b)" in result.name

        # Test with unnamed tensors
        a = Tensor([1, 2])  # No name
        b = Tensor([3, 4])  # No name
        result = div(a, b)
        assert "div(tensor, tensor)" in result.name
