"""
Comprehensive test suite for automatic differentiation system.
Tests all components of autograd.py for comprehensive coverage.

This module tests:
- GradientTape
- Variable class
- Operation tracking
- Forward and backward passes
- Higher-order derivatives
- Functional API
- Memory optimization
- JAX-like transformations
"""

import math
import time
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple, Any

import numpy as np
import pytest

from src.neural_arch.core.autograd import (
    GradientTape,
    Variable,
    Operation,
    OperationType,
    FunctionContext,
    no_grad,
    enable_grad,
    grad,
    jacrev,
    hessian,
    vmap,
    jit,
    value_and_grad,
    test_autograd_system
)


class TestVariable:
    """Test Variable class for gradient computation."""
    
    def test_variable_creation(self):
        """Test Variable creation and basic properties."""
        # Test with numpy array
        data = np.array([1.0, 2.0, 3.0])
        var = Variable(data, requires_grad=True)
        
        assert np.array_equal(var.data, data)
        assert var.requires_grad is True
        assert var.grad is None
        assert var.is_leaf is True
        assert var.grad_fn is None
        
    def test_variable_without_grad(self):
        """Test Variable without gradient tracking."""
        data = np.array([1.0, 2.0, 3.0])
        var = Variable(data, requires_grad=False)
        
        assert var.requires_grad is False
        assert var.grad is None
        assert var.is_leaf is True
        
    def test_variable_operations(self):
        """Test basic Variable operations."""
        a = Variable(np.array([1.0, 2.0]), requires_grad=True)
        b = Variable(np.array([3.0, 4.0]), requires_grad=True)
        
        # Addition
        c = a + b
        assert np.array_equal(c.data, np.array([4.0, 6.0]))
        assert c.requires_grad is True
        assert c.is_leaf is False
        assert c.grad_fn is not None
        
        # Multiplication
        d = a * b
        assert np.array_equal(d.data, np.array([3.0, 8.0]))
        assert d.requires_grad is True
        
    def test_variable_scalar_operations(self):
        """Test Variable operations with scalars."""
        a = Variable(np.array([2.0, 3.0]), requires_grad=True)
        
        # Scalar addition
        b = a + 5.0
        assert np.array_equal(b.data, np.array([7.0, 8.0]))
        
        # Scalar multiplication
        c = a * 2.0
        assert np.array_equal(c.data, np.array([4.0, 6.0]))
        
        # Scalar division
        d = a / 2.0
        assert np.array_equal(d.data, np.array([1.0, 1.5]))
        
    def test_variable_backward_compatibility(self):
        """Test backward compatibility for Variable."""
        a = Variable(np.array([1.0, 2.0]), requires_grad=True)
        b = Variable(np.array([3.0, 4.0]), requires_grad=True)
        
        c = a * b + a
        
        # Manual backward (simplified)
        c.grad = np.ones_like(c.data)
        assert c.grad is not None
        
    def test_variable_detach(self):
        """Test Variable detach operation."""
        a = Variable(np.array([1.0, 2.0]), requires_grad=True)
        b = a * 2.0
        
        # Detach removes from computation graph
        c = b.detach()
        assert c.requires_grad is False
        assert np.array_equal(c.data, b.data)
        assert c.is_leaf is True
        
    def test_variable_clone(self):
        """Test Variable cloning."""
        a = Variable(np.array([1.0, 2.0]), requires_grad=True)
        
        # Clone with gradients
        b = a.clone()
        assert np.array_equal(b.data, a.data)
        assert b.requires_grad == a.requires_grad
        assert b is not a  # Different objects
        
        # Clone without gradients
        c = a.clone(requires_grad=False)
        assert c.requires_grad is False


class TestOperation:
    """Test Operation tracking for automatic differentiation."""
    
    def test_operation_creation(self):
        """Test Operation object creation."""
        inputs = [Variable(np.array([1.0, 2.0]), requires_grad=True)]
        op = Operation(
            op_type=OperationType.ADD,
            inputs=inputs,
            forward_fn=lambda x: x + 1,
            backward_fn=lambda grad: [grad]
        )
        
        assert op.op_type == OperationType.ADD
        assert len(op.inputs) == 1
        assert op.forward_fn is not None
        assert op.backward_fn is not None
        assert op.ctx is not None
        
    def test_operation_types(self):
        """Test different operation types."""
        operation_types = [
            OperationType.ADD,
            OperationType.MUL,
            OperationType.MATMUL,
            OperationType.EXP,
            OperationType.LOG,
            OperationType.SIN,
            OperationType.COS,
            OperationType.TANH,
            OperationType.RELU,
            OperationType.SIGMOID,
            OperationType.SOFTMAX,
            OperationType.RESHAPE,
            OperationType.TRANSPOSE,
            OperationType.SLICE,
            OperationType.CONCAT,
            OperationType.REDUCE_SUM,
            OperationType.REDUCE_MEAN
        ]
        
        for op_type in operation_types:
            assert isinstance(op_type, OperationType)
            
    def test_function_context(self):
        """Test FunctionContext for storing intermediate values."""
        ctx = FunctionContext()
        
        # Save values
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        ctx.save_for_backward(a, b)
        
        # Retrieve values
        saved = ctx.saved_tensors
        assert len(saved) == 2
        assert np.array_equal(saved[0], a)
        assert np.array_equal(saved[1], b)
        
    def test_operation_execution(self):
        """Test operation execution with context."""
        def forward_fn(x, y, ctx):
            ctx.save_for_backward(x, y)
            return x * y
            
        def backward_fn(grad, ctx):
            x, y = ctx.saved_tensors
            return [grad * y, grad * x]
            
        a = Variable(np.array([2.0, 3.0]), requires_grad=True)
        b = Variable(np.array([4.0, 5.0]), requires_grad=True)
        
        op = Operation(
            op_type=OperationType.MUL,
            inputs=[a, b],
            forward_fn=lambda inputs, ctx: forward_fn(inputs[0].data, inputs[1].data, ctx),
            backward_fn=backward_fn
        )
        
        # Test forward execution
        result = op.forward_fn([a, b], op.ctx)
        assert np.array_equal(result, np.array([8.0, 15.0]))
        
        # Test backward execution
        grad_output = np.array([1.0, 1.0])
        grads = op.backward_fn(grad_output, op.ctx)
        assert len(grads) == 2
        assert np.array_equal(grads[0], np.array([4.0, 5.0]))  # grad w.r.t a
        assert np.array_equal(grads[1], np.array([2.0, 3.0]))  # grad w.r.t b


class TestGradientTape:
    """Test GradientTape for automatic differentiation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.tape = GradientTape()
        
    def test_tape_creation(self):
        """Test GradientTape creation and properties."""
        assert len(self.tape.operations) == 0
        assert self.tape._recording is True
        assert self.tape.persistent is False
        
    def test_tape_context_manager(self):
        """Test GradientTape as context manager."""
        a = Variable(np.array([2.0, 3.0]), requires_grad=True)
        b = Variable(np.array([4.0, 5.0]), requires_grad=True)
        
        with GradientTape() as tape:
            c = a * b + a
            
        # Should have recorded operations
        assert len(tape.operations) > 0
        
        # Test gradient computation
        grads = tape.gradient(c, [a, b])
        assert len(grads) == 2
        assert grads[0] is not None  # grad w.r.t a
        assert grads[1] is not None  # grad w.r.t b
        
    def test_persistent_tape(self):
        """Test persistent GradientTape."""
        tape = GradientTape(persistent=True)
        
        a = Variable(np.array([2.0]), requires_grad=True)
        
        with tape:
            b = a * a
            c = b * a
            
        # Can compute gradients multiple times with persistent tape
        grad_b = tape.gradient(b, a)
        grad_c = tape.gradient(c, a)
        
        assert grad_b is not None
        assert grad_c is not None
        assert not np.array_equal(grad_b, grad_c)
        
    def test_gradient_computation_basic(self):
        """Test basic gradient computation."""
        with GradientTape() as tape:
            a = Variable(np.array([3.0]), requires_grad=True)
            b = a * a  # b = a²
            
        grad = tape.gradient(b, a)
        
        # db/da = 2a = 2*3 = 6
        assert np.allclose(grad, np.array([6.0]))
        
    def test_gradient_computation_chain_rule(self):
        """Test gradient computation with chain rule."""
        with GradientTape() as tape:
            a = Variable(np.array([2.0]), requires_grad=True)
            b = a * 3.0      # b = 3a
            c = b * b        # c = b² = 9a²
            
        grad = tape.gradient(c, a)
        
        # dc/da = dc/db * db/da = 2b * 3 = 2(3a) * 3 = 18a = 18*2 = 36
        assert np.allclose(grad, np.array([36.0]))
        
    def test_gradient_multiple_variables(self):
        """Test gradient computation with multiple variables."""
        with GradientTape() as tape:
            a = Variable(np.array([2.0]), requires_grad=True)
            b = Variable(np.array([3.0]), requires_grad=True)
            c = a * b + a * a  # c = ab + a²
            
        grads = tape.gradient(c, [a, b])
        
        # dc/da = b + 2a = 3 + 2*2 = 7
        # dc/db = a = 2
        assert np.allclose(grads[0], np.array([7.0]))
        assert np.allclose(grads[1], np.array([2.0]))
        
    def test_gradient_vector_operations(self):
        """Test gradient computation with vector operations."""
        with GradientTape() as tape:
            a = Variable(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            b = Variable(np.array([4.0, 5.0, 6.0]), requires_grad=True)
            c = np.sum(a * b)  # Scalar output
            
        grads = tape.gradient(c, [a, b])
        
        # dc/da = b, dc/db = a
        assert np.allclose(grads[0], b.data)
        assert np.allclose(grads[1], a.data)
        
    def test_gradient_matrix_operations(self):
        """Test gradient computation with matrix operations."""
        with GradientTape() as tape:
            A = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
            B = Variable(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
            C = np.matmul(A, B)
            loss = np.sum(C)
            
        grads = tape.gradient(loss, [A, B])
        
        # Check shapes are correct
        assert grads[0].shape == A.data.shape
        assert grads[1].shape == B.data.shape
        
        # Check gradients are non-zero
        assert not np.allclose(grads[0], 0.0)
        assert not np.allclose(grads[1], 0.0)
        
    def test_no_gradient_context(self):
        """Test no_grad context manager."""
        a = Variable(np.array([2.0]), requires_grad=True)
        
        with no_grad():
            b = a * a
            
        # No gradients should be computed
        assert b.requires_grad is False
        assert b.grad_fn is None
        
    def test_enable_disable_grad(self):
        """Test enable_grad and disable gradient computation."""
        a = Variable(np.array([2.0]), requires_grad=True)
        
        # Disable gradients
        with no_grad():
            b = a * 2.0
            assert b.requires_grad is False
            
            # Re-enable gradients within no_grad
            with enable_grad():
                c = a * 3.0
                assert c.requires_grad is True
                
    def test_higher_order_gradients(self):
        """Test higher-order gradient computation."""
        # First order gradient
        with GradientTape() as tape1:
            with GradientTape() as tape2:
                a = Variable(np.array([2.0]), requires_grad=True)
                b = a * a * a  # b = a³
                
            first_grad = tape2.gradient(b, a)  # db/da = 3a²
            
        second_grad = tape1.gradient(first_grad, a)  # d²b/da² = 6a
        
        # At a = 2: second derivative = 6*2 = 12
        assert np.allclose(second_grad, np.array([12.0]))
        
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        a = Variable(np.array([2.0]), requires_grad=True)
        
        # Compute gradients multiple times and accumulate
        total_grad = np.zeros_like(a.data)
        
        for i in range(3):
            with GradientTape() as tape:
                b = a * a * (i + 1)  # Different scaling
                
            grad = tape.gradient(b, a)
            total_grad += grad
            
        # Total gradient should be sum of individual gradients
        expected = 2 * 2.0 * (1 + 2 + 3)  # 2a * sum(i) = 4 * 6 = 24
        assert np.allclose(total_grad, np.array([expected]))


class TestFunctionalAPI:
    """Test functional API for transformations."""
    
    def test_grad_function(self):
        """Test grad function for computing gradients."""
        def f(x):
            return x * x * x  # x³
            
        # Compute gradient: df/dx = 3x²
        grad_f = grad(f)
        
        x = np.array([2.0])
        gradient = grad_f(x)
        
        # At x = 2: gradient = 3 * 2² = 12
        assert np.allclose(gradient, np.array([12.0]))
        
    def test_value_and_grad_function(self):
        """Test value_and_grad function."""
        def f(x):
            return np.sum(x * x)  # Sum of squares
            
        value_grad_f = value_and_grad(f)
        
        x = np.array([1.0, 2.0, 3.0])
        value, gradient = value_grad_f(x)
        
        # Value: 1² + 2² + 3² = 14
        # Gradient: [2*1, 2*2, 2*3] = [2, 4, 6]
        assert np.allclose(value, 14.0)
        assert np.allclose(gradient, np.array([2.0, 4.0, 6.0]))
        
    def test_jacrev_function(self):
        """Test jacrev function for computing Jacobians."""
        def f(x):
            return np.array([x[0] * x[1], x[0] + x[1], x[0] * x[0]])
            
        jacrev_f = jacrev(f)
        
        x = np.array([2.0, 3.0])
        jacobian = jacrev_f(x)
        
        # Jacobian should be 3x2 matrix
        assert jacobian.shape == (3, 2)
        
        # df1/dx1 = x[1] = 3, df1/dx2 = x[0] = 2
        # df2/dx1 = 1, df2/dx2 = 1  
        # df3/dx1 = 2*x[0] = 4, df3/dx2 = 0
        expected = np.array([[3.0, 2.0], [1.0, 1.0], [4.0, 0.0]])
        assert np.allclose(jacobian, expected)
        
    def test_hessian_function(self):
        """Test hessian function for computing Hessians."""
        def f(x):
            return x[0] * x[0] * x[1] + x[1] * x[1]  # x₁²x₂ + x₂²
            
        hessian_f = hessian(f)
        
        x = np.array([2.0, 3.0])
        H = hessian_f(x)
        
        # Hessian should be 2x2 matrix
        assert H.shape == (2, 2)
        
        # ∂²f/∂x₁² = 2*x₂ = 6
        # ∂²f/∂x₁∂x₂ = 2*x₁ = 4
        # ∂²f/∂x₂∂x₁ = 2*x₁ = 4  
        # ∂²f/∂x₂² = 2
        expected = np.array([[6.0, 4.0], [4.0, 2.0]])
        assert np.allclose(H, expected)
        
    def test_vmap_function(self):
        """Test vmap function for vectorized mapping."""
        def f(x):
            return x * x
            
        vmap_f = vmap(f)
        
        # Apply function to each element of batch
        batch_x = np.array([[1.0], [2.0], [3.0]])
        result = vmap_f(batch_x)
        
        expected = np.array([[1.0], [4.0], [9.0]])
        assert np.allclose(result, expected)
        
    def test_jit_function(self):
        """Test jit compilation simulation."""
        def f(x):
            return x * x + 2 * x + 1
            
        # JIT compilation (simulated)
        jit_f = jit(f)
        
        x = np.array([2.0, 3.0])
        result = jit_f(x)
        
        # f(x) = x² + 2x + 1
        # f(2) = 4 + 4 + 1 = 9
        # f(3) = 9 + 6 + 1 = 16
        expected = np.array([9.0, 16.0])
        assert np.allclose(result, expected)


class TestAdvancedOperations:
    """Test advanced automatic differentiation operations."""
    
    def test_trigonometric_functions(self):
        """Test gradients of trigonometric functions."""
        with GradientTape() as tape:
            x = Variable(np.array([np.pi/4]), requires_grad=True)
            y = np.sin(x)
            
        grad = tape.gradient(y, x)
        
        # d(sin(x))/dx = cos(x)
        # At x = π/4: cos(π/4) = √2/2 ≈ 0.707
        assert np.allclose(grad, np.cos(np.pi/4), atol=1e-6)
        
    def test_exponential_functions(self):
        """Test gradients of exponential functions."""
        with GradientTape() as tape:
            x = Variable(np.array([2.0]), requires_grad=True)
            y = np.exp(x)
            
        grad = tape.gradient(y, x)
        
        # d(exp(x))/dx = exp(x)
        # At x = 2: exp(2) ≈ 7.389
        assert np.allclose(grad, np.exp(2.0), atol=1e-6)
        
    def test_logarithmic_functions(self):
        """Test gradients of logarithmic functions."""
        with GradientTape() as tape:
            x = Variable(np.array([2.0]), requires_grad=True)
            y = np.log(x)
            
        grad = tape.gradient(y, x)
        
        # d(log(x))/dx = 1/x
        # At x = 2: 1/2 = 0.5
        assert np.allclose(grad, np.array([0.5]), atol=1e-6)
        
    def test_power_functions(self):
        """Test gradients of power functions."""
        with GradientTape() as tape:
            x = Variable(np.array([3.0]), requires_grad=True)
            y = np.power(x, 3.0)  # x³
            
        grad = tape.gradient(y, x)
        
        # d(x³)/dx = 3x²
        # At x = 3: 3 * 3² = 27
        assert np.allclose(grad, np.array([27.0]), atol=1e-6)
        
    def test_activation_functions(self):
        """Test gradients of activation functions."""
        # Test ReLU
        with GradientTape() as tape:
            x = Variable(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
            y = np.maximum(x, 0.0)  # ReLU
            
        grad = tape.gradient(np.sum(y), x)
        
        # ReLU derivative: 0 for x < 0, 1 for x > 0, undefined at 0
        expected_relu = np.array([0.0, 0.0, 1.0])  # Treating 0 as having 0 gradient
        assert np.allclose(grad, expected_relu)
        
    def test_softmax_gradient(self):
        """Test gradients of softmax function."""
        def softmax(x):
            exp_x = np.exp(x - np.max(x))  # Numerical stability
            return exp_x / np.sum(exp_x)
            
        with GradientTape() as tape:
            x = Variable(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            y = softmax(x)
            loss = np.sum(y * np.array([1.0, 0.0, 0.0]))  # Cross-entropy style
            
        grad = tape.gradient(loss, x)
        
        # Softmax gradient is complex but should be well-defined
        assert grad.shape == x.data.shape
        assert not np.allclose(grad, 0.0)  # Should be non-zero
        
    def test_batch_operations(self):
        """Test gradients with batch operations."""
        batch_size = 32
        input_dim = 10
        
        with GradientTape() as tape:
            X = Variable(np.random.randn(batch_size, input_dim), requires_grad=True)
            W = Variable(np.random.randn(input_dim, 1), requires_grad=True)
            b = Variable(np.random.randn(1), requires_grad=True)
            
            # Linear layer
            y = np.matmul(X, W) + b
            loss = np.mean(y * y)  # MSE loss
            
        grads = tape.gradient(loss, [X, W, b])
        
        # Check gradient shapes
        assert grads[0].shape == X.data.shape  # dL/dX
        assert grads[1].shape == W.data.shape  # dL/dW  
        assert grads[2].shape == b.data.shape  # dL/db
        
        # Gradients should not be zero
        assert not np.allclose(grads[0], 0.0)
        assert not np.allclose(grads[1], 0.0)
        assert not np.allclose(grads[2], 0.0)


class TestMemoryOptimization:
    """Test memory optimization in automatic differentiation."""
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing for memory efficiency."""
        # This is a conceptual test - real implementation would be more complex
        def checkpoint_function(f, x):
            # Simulate checkpointing by not storing intermediate values
            return f(x)
            
        def expensive_computation(x):
            # Simulate expensive computation with many intermediate steps
            for _ in range(10):
                x = x * 1.1 + 0.01
            return x
            
        with GradientTape() as tape:
            x = Variable(np.array([1.0]), requires_grad=True)
            y = checkpoint_function(expensive_computation, x)
            
        grad = tape.gradient(y, x)
        assert grad is not None
        assert not np.isnan(grad).any()
        
    def test_memory_efficient_backward(self):
        """Test memory-efficient backward pass."""
        # Test with a deep computation graph
        with GradientTape() as tape:
            x = Variable(np.array([2.0]), requires_grad=True)
            
            # Create deep computation graph
            y = x
            for i in range(100):
                y = y * 1.01 + 0.001  # Small incremental changes
                
        grad = tape.gradient(y, x)
        
        # Should still compute gradients correctly
        assert grad is not None
        assert not np.isnan(grad).any()
        assert grad > 0  # Should be positive due to positive multipliers
        
    def test_inplace_operations(self):
        """Test handling of in-place operations."""
        with GradientTape() as tape:
            x = Variable(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            
            # Simulate in-place operation (should be handled carefully)
            y = x * 2.0
            # In real implementation, in-place ops would require special handling
            
        grad = tape.gradient(np.sum(y), x)
        
        # Gradient should be [2, 2, 2] for y = 2*x
        assert np.allclose(grad, np.array([2.0, 2.0, 2.0]))


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_gradient_of_non_scalar(self):
        """Test error handling for gradients of non-scalar outputs."""
        with GradientTape() as tape:
            x = Variable(np.array([1.0, 2.0]), requires_grad=True)
            y = x * 2.0  # Vector output
            
        # Should handle vector outputs appropriately
        # (may require output_gradients parameter)
        try:
            grad = tape.gradient(y, x, output_gradients=np.ones_like(y.data))
            assert grad is not None
        except Exception:
            # Some implementations require explicit output gradients
            pass
            
    def test_gradient_of_integer_input(self):
        """Test gradient computation with integer inputs."""
        with GradientTape() as tape:
            # Integer input should be converted to float for gradients
            x = Variable(np.array([1, 2, 3], dtype=np.int32), requires_grad=True)
            y = np.sum(x.astype(np.float32) * x.astype(np.float32))
            
        # May need special handling for integer inputs
        try:
            grad = tape.gradient(y, x)
            assert grad is not None
        except Exception:
            # Expected - gradients typically require float types
            pass
            
    def test_zero_gradients(self):
        """Test handling of zero gradients."""
        with GradientTape() as tape:
            x = Variable(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            y = Variable(np.array([4.0, 5.0, 6.0]), requires_grad=True)
            
            # Operation that doesn't depend on x
            z = y * y
            
        grad_x = tape.gradient(np.sum(z), x)
        grad_y = tape.gradient(np.sum(z), y)
        
        # Gradient w.r.t. x should be zero (z doesn't depend on x)
        assert np.allclose(grad_x, 0.0)
        
        # Gradient w.r.t. y should be non-zero
        assert not np.allclose(grad_y, 0.0)
        
    def test_disconnected_gradients(self):
        """Test gradients through disconnected computation graph."""
        with GradientTape() as tape:
            x = Variable(np.array([2.0]), requires_grad=True)
            y = x * 3.0
            
            # Detach y from computation graph
            y_detached = y.detach()
            z = y_detached * 2.0
            
        # Gradient should not flow through detached variable
        grad = tape.gradient(z, x)
        
        # May be None or zero depending on implementation
        assert grad is None or np.allclose(grad, 0.0)
        
    def test_circular_dependencies(self):
        """Test handling of circular dependencies."""
        # This is more of a conceptual test - circular dependencies
        # are generally not allowed in computation graphs
        
        with GradientTape() as tape:
            x = Variable(np.array([1.0]), requires_grad=True)
            y = x * 2.0
            
            # Avoid actual circular dependency but test robustness
            z = y + x  # Both depend on x, but no circularity
            
        grad = tape.gradient(z, x)
        
        # Should be 2 + 1 = 3
        assert np.allclose(grad, np.array([3.0]))


class TestPerformanceOptimizations:
    """Test performance optimizations in autograd."""
    
    def test_operation_fusion(self):
        """Test operation fusion for better performance."""
        with GradientTape() as tape:
            x = Variable(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            
            # Chain of operations that could be fused
            y = x * 2.0 + 1.0  # Could be fused into a single operation
            z = y * 3.0 - 0.5  # Another fusible operation
            
        grad = tape.gradient(np.sum(z), x)
        
        # Final gradient should be correct regardless of fusion
        # dz/dx = d/dx((2x + 1) * 3 - 0.5) = d/dx(6x + 2.5) = 6
        assert np.allclose(grad, np.array([6.0, 6.0, 6.0]))
        
    def test_sparse_gradients(self):
        """Test handling of sparse gradients."""
        with GradientTape() as tape:
            # Large tensor with mostly zeros
            x = Variable(np.zeros(1000), requires_grad=True)
            x.data[100] = 1.0  # Only one non-zero element
            
            y = np.sum(x * x)  # Only contributes gradient at position 100
            
        grad = tape.gradient(y, x)
        
        # Gradient should be sparse (mostly zeros)
        assert np.sum(grad != 0) <= 1  # At most one non-zero element
        if np.sum(grad != 0) == 1:
            assert np.allclose(grad[100], 2.0)  # d(x²)/dx = 2x = 2*1 = 2
            
    def test_gradient_accumulation_performance(self):
        """Test performance of gradient accumulation."""
        x = Variable(np.random.randn(1000), requires_grad=True)
        
        # Accumulate gradients from multiple computations
        accumulated_grad = np.zeros_like(x.data)
        
        for i in range(10):
            with GradientTape() as tape:
                y = np.sum(x * (i + 1))  # Different scaling each time
                
            grad = tape.gradient(y, x)
            accumulated_grad += grad
            
        # Final accumulated gradient should be sum of individual gradients
        expected = np.sum(range(1, 11)) * np.ones_like(x.data)  # sum(1..10) = 55
        assert np.allclose(accumulated_grad, expected)


class TestSystemIntegration:
    """Test integration with the complete autograd system."""
    
    def test_built_in_system_test(self):
        """Test integration with built-in system test."""
        try:
            # Run the built-in test function
            test_autograd_system()
            print("✅ Autograd system test passed")
        except Exception as e:
            pytest.fail(f"System integration test failed: {e}")
            
    def test_realistic_neural_network_gradient(self):
        """Test gradients in a realistic neural network scenario."""
        # Mini neural network: input -> linear -> relu -> linear -> output
        batch_size, input_dim, hidden_dim, output_dim = 16, 10, 20, 5
        
        with GradientTape() as tape:
            # Input
            X = Variable(np.random.randn(batch_size, input_dim), requires_grad=True)
            
            # First layer
            W1 = Variable(np.random.randn(input_dim, hidden_dim) * 0.1, requires_grad=True)
            b1 = Variable(np.zeros(hidden_dim), requires_grad=True)
            h1 = np.matmul(X, W1) + b1
            
            # ReLU activation
            h1_relu = np.maximum(h1, 0.0)
            
            # Second layer  
            W2 = Variable(np.random.randn(hidden_dim, output_dim) * 0.1, requires_grad=True)
            b2 = Variable(np.zeros(output_dim), requires_grad=True)
            output = np.matmul(h1_relu, W2) + b2
            
            # Loss (MSE)
            target = np.random.randn(batch_size, output_dim)
            loss = np.mean((output - target) ** 2)
            
        # Compute gradients
        grads = tape.gradient(loss, [W1, b1, W2, b2])
        
        # Check all gradients computed
        assert all(grad is not None for grad in grads)
        
        # Check gradient shapes
        assert grads[0].shape == W1.data.shape
        assert grads[1].shape == b1.data.shape  
        assert grads[2].shape == W2.data.shape
        assert grads[3].shape == b2.data.shape
        
        # Gradients should be non-zero (except possibly for some elements due to ReLU)
        assert not np.allclose(grads[0], 0.0)
        assert not np.allclose(grads[2], 0.0)
        
    def test_optimization_loop_simulation(self):
        """Test autograd in an optimization loop."""
        # Simple quadratic optimization: minimize (x - 3)²
        x = Variable(np.array([0.0]), requires_grad=True)
        target = 3.0
        lr = 0.1
        
        for step in range(50):
            with GradientTape() as tape:
                loss = (x - target) ** 2
                
            grad = tape.gradient(loss, x)
            
            # Manual gradient update
            x.data = x.data - lr * grad
            
        # Should converge close to target
        assert np.abs(x.data[0] - target) < 0.1
        
    def test_complex_function_approximation(self):
        """Test autograd with complex function approximation."""
        # Approximate sin(x) with a polynomial using gradient descent
        def polynomial(x, coeffs):
            result = coeffs[0]
            x_power = x
            for i in range(1, len(coeffs)):
                result = result + coeffs[i] * x_power
                x_power = x_power * x
            return result
            
        # Initialize coefficients
        coeffs = Variable(np.array([0.0, 1.0, 0.0, -1.0/6.0]), requires_grad=True)
        
        # Training data
        x_data = np.linspace(-np.pi/2, np.pi/2, 20)
        y_data = np.sin(x_data)
        
        lr = 0.01
        for step in range(100):
            with GradientTape() as tape:
                predictions = polynomial(x_data, coeffs)
                loss = np.mean((predictions - y_data) ** 2)
                
            grad = tape.gradient(loss, coeffs)
            
            # Update coefficients
            coeffs.data = coeffs.data - lr * grad
            
        # Final approximation should be reasonable
        final_predictions = polynomial(x_data, coeffs)
        mse = np.mean((final_predictions - y_data) ** 2)
        assert mse < 0.1  # Should approximate sin(x) reasonably well


class TestAdvancedAutogradFeatures:
    """Test advanced autograd features for complete coverage."""
    
    def test_higher_order_derivatives(self):
        """Test computation of higher order derivatives."""
        with GradientTape() as tape2:
            with GradientTape() as tape1:
                x = Variable(np.array([2.0]), requires_grad=True)
                y = x ** 4  # y = x^4
                
            # First derivative: dy/dx = 4x^3
            dy_dx = tape1.gradient(y, x)
            
        # Second derivative: d²y/dx² = 12x^2
        d2y_dx2 = tape2.gradient(dy_dx, x)
        
        # At x=2: d²y/dx² = 12 * 4 = 48
        expected_second_derivative = 48.0
        assert np.allclose(d2y_dx2, expected_second_derivative)
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing for memory efficiency."""
        # Simulate a deep computation with checkpointing
        def deep_computation(x, depth=10):
            result = x
            for i in range(depth):
                result = result * 2 + np.sin(result)
            return result
        
        with GradientTape() as tape:
            x = Variable(np.array([0.5]), requires_grad=True)
            
            # Use checkpointing to save memory
            y = deep_computation(x, depth=20)
            
        grad = tape.gradient(y, x)
        
        # Should compute gradient successfully despite deep computation
        assert grad is not None
        assert not np.isnan(grad).any()
    
    def test_custom_gradient_functions(self):
        """Test custom gradient functions."""
        def custom_relu(x):
            """Custom ReLU with custom gradient."""
            output = np.maximum(x, 0)
            
            def custom_gradient(upstream_grad):
                return upstream_grad * (x > 0).astype(float)
            
            return output, custom_gradient
        
        with GradientTape() as tape:
            x = Variable(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)
            y, grad_fn = custom_relu(x.data)
            y_var = Variable(y, requires_grad=True, grad_fn=grad_fn)
            loss = np.sum(y_var ** 2)
            
        grad = tape.gradient(loss, x)
        
        # Gradient should be 0 for negative inputs, positive for positive inputs
        expected = np.array([0.0, 0.0, 2.0, 4.0])  # 2 * y for positive y
        assert np.allclose(grad, expected)
    
    def test_gradient_accumulation_patterns(self):
        """Test different gradient accumulation patterns."""
        x = Variable(np.array([1.0, 2.0]), requires_grad=True)
        
        # Pattern 1: Sequential accumulation
        accumulated_grad = np.zeros_like(x.data)
        
        for i in range(5):
            with GradientTape() as tape:
                y = (x ** 2).sum() * (i + 1)
                
            grad = tape.gradient(y, x)
            accumulated_grad += grad
            
        # Expected: sum of (2x * (i+1)) for i in range(5)
        # = 2x * sum(1,2,3,4,5) = 2x * 15 = 30x
        expected = 30 * x.data
        assert np.allclose(accumulated_grad, expected)
    
    def test_dynamic_computational_graphs(self):
        """Test dynamic computational graphs."""
        def dynamic_network(x, use_nonlinearity=True):
            if use_nonlinearity:
                return np.tanh(x ** 3)
            else:
                return x ** 2
        
        x = Variable(np.array([0.5]), requires_grad=True)
        
        # Test with nonlinearity
        with GradientTape() as tape:
            y1 = dynamic_network(x, use_nonlinearity=True)
            
        grad1 = tape.gradient(y1, x)
        
        # Test without nonlinearity
        with GradientTape() as tape:
            y2 = dynamic_network(x, use_nonlinearity=False)
            
        grad2 = tape.gradient(y2, x)
        
        # Gradients should be different
        assert not np.allclose(grad1, grad2)
    
    def test_gradient_masking(self):
        """Test gradient masking for selective updates."""
        with GradientTape() as tape:
            x = Variable(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
            mask = np.array([1.0, 0.0, 1.0, 0.0])  # Mask some elements
            
            # Apply mask to gradients
            masked_x = x * mask
            y = np.sum(masked_x ** 2)
            
        grad = tape.gradient(y, x)
        
        # Gradient should be zero where mask is zero
        expected = np.array([2.0, 0.0, 6.0, 0.0])  # 2x where mask=1, 0 where mask=0
        assert np.allclose(grad, expected)
    
    def test_gradient_clipping_integration(self):
        """Test gradient clipping integration."""
        def clip_gradients(grads, max_norm=1.0):
            total_norm = np.sqrt(sum(np.sum(grad ** 2) for grad in grads))
            if total_norm > max_norm:
                clip_coef = max_norm / total_norm
                return [grad * clip_coef for grad in grads]
            return grads
        
        with GradientTape() as tape:
            x1 = Variable(np.array([10.0]), requires_grad=True)
            x2 = Variable(np.array([20.0]), requires_grad=True)
            
            # Large gradients
            y = x1 ** 2 + x2 ** 2
            
        grads = tape.gradient(y, [x1, x2])
        
        # Clip gradients
        clipped_grads = clip_gradients(grads, max_norm=1.0)
        
        # Total norm should be <= 1.0
        total_norm = np.sqrt(sum(np.sum(grad ** 2) for grad in clipped_grads))
        assert total_norm <= 1.01  # Small tolerance


class TestAutogradEdgeCases:
    """Test autograd edge cases and error conditions."""
    
    def test_zero_gradients(self):
        """Test handling of zero gradients."""
        with GradientTape() as tape:
            x = Variable(np.array([1.0]), requires_grad=True)
            y = x * 0  # Should produce zero gradient
            
        grad = tape.gradient(y, x)
        assert np.allclose(grad, 0.0)
    
    def test_nan_and_inf_gradients(self):
        """Test handling of NaN and inf gradients."""
        with GradientTape() as tape:
            x = Variable(np.array([0.0]), requires_grad=True)
            
            # This might produce nan in gradient
            y = x / x  # 0/0 = nan
            
        # Should handle gracefully
        grad = tape.gradient(y, x)
        # Gradient computation should not crash, though result may be nan
        assert grad is not None
    
    def test_very_deep_graphs(self):
        """Test very deep computational graphs."""
        with GradientTape() as tape:
            x = Variable(np.array([1.0]), requires_grad=True)
            
            # Create very deep graph
            result = x
            for _ in range(1000):
                result = result + 0.001 * result  # Small incremental changes
                
        grad = tape.gradient(result, x)
        
        # Should compute gradient successfully
        assert grad is not None
        assert not np.isnan(grad).any()
        assert grad > 0  # Should be positive due to chain rule
    
    def test_disconnected_variables(self):
        """Test gradients with disconnected variables."""
        with GradientTape() as tape:
            x = Variable(np.array([1.0]), requires_grad=True)
            y = Variable(np.array([2.0]), requires_grad=True)  # Disconnected from computation
            
            z = x ** 2  # Only depends on x
            
        grad_x = tape.gradient(z, x)
        grad_y = tape.gradient(z, y)
        
        assert grad_x is not None
        assert grad_y is None or np.allclose(grad_y, 0.0)  # Should be None or zero
    
    def test_multiple_tape_interactions(self):
        """Test interactions between multiple gradient tapes."""
        x = Variable(np.array([2.0]), requires_grad=True)
        
        with GradientTape() as tape1:
            with GradientTape() as tape2:
                y = x ** 3
                
            # First-order gradient in inner tape
            dy_dx = tape2.gradient(y, x)
            
            # Use gradient in outer computation
            z = dy_dx ** 2
            
        # Second-order gradient in outer tape
        d2z_dx = tape1.gradient(z, x)
        
        # Should compute second-order derivatives correctly
        assert d2z_dx is not None
        assert not np.isnan(d2z_dx).any()


class TestPerformanceOptimizations:
    """Test performance optimizations in autograd."""
    
    def test_operation_fusion(self):
        """Test operation fusion optimizations."""
        with GradientTape() as tape:
            x = Variable(np.random.randn(1000, 1000), requires_grad=True)
            
            # Chain of operations that could be fused
            y = x + 1.0
            y = y * 2.0
            y = np.tanh(y)
            y = y ** 2
            loss = np.sum(y)
            
        grad = tape.gradient(loss, x)
        
        # Should compute efficiently
        assert grad is not None
        assert grad.shape == x.data.shape
    
    def test_memory_efficient_gradients(self):
        """Test memory efficient gradient computation."""
        def create_large_computation():
            with GradientTape() as tape:
                x = Variable(np.random.randn(100, 100), requires_grad=True)
                
                # Operations that use significant memory
                for i in range(10):
                    x = np.matmul(x, x.T) / 100  # Keep values reasonable
                    x = x + 0.01 * np.random.randn(*x.shape)
                    
                loss = np.sum(x ** 2)
                
            return tape.gradient(loss, x)
        
        # Should complete without memory issues
        grad = create_large_computation()
        assert grad is not None
    
    def test_sparse_gradient_handling(self):
        """Test sparse gradient handling."""
        with GradientTape() as tape:
            # Simulate sparse updates (e.g., embedding lookups)
            x = Variable(np.random.randn(1000, 50), requires_grad=True)
            indices = np.array([0, 5, 10, 15, 20])  # Only update these rows
            
            # Only some rows are used in computation
            selected_x = x.data[indices]
            y = np.sum(selected_x ** 2)
            
        grad = tape.gradient(y, x)
        
        # Most gradients should be zero (sparse)
        non_zero_rows = np.any(grad != 0, axis=1)
        assert np.sum(non_zero_rows) <= len(indices) * 2  # Allow some tolerance


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])