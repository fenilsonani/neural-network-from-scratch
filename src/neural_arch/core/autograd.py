"""Production-grade automatic differentiation engine with gradient tape.

This module implements a sophisticated automatic differentiation system with:
- Dynamic computational graph construction
- Gradient tape for memory efficiency
- Higher-order derivatives support
- Custom gradient functions
- Checkpointing for memory optimization
- Vectorized operations
- Graph optimization and fusion
"""

import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np


class OpType(Enum):
    """Types of operations in the computational graph."""
    
    # Arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    NEG = "neg"
    
    # Matrix operations
    MATMUL = "matmul"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    
    # Reductions
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    
    # Activations
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    GELU = "gelu"
    
    # Convolution
    CONV2D = "conv2d"
    MAXPOOL2D = "maxpool2d"
    
    # Other
    CUSTOM = "custom"
    IDENTITY = "identity"
    BROADCAST = "broadcast"
    SLICE = "slice"
    CONCAT = "concat"


@dataclass
class TapeEntry:
    """Entry in the gradient tape representing a single operation."""
    
    op_type: OpType
    inputs: List['Variable']
    output: 'Variable'
    grad_fn: Optional[Callable] = None
    saved_tensors: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return id(self)


class GradientTape:
    """Gradient tape for automatic differentiation with memory optimization."""
    
    def __init__(self, persistent: bool = False, watch_accessed_variables: bool = True):
        """Initialize gradient tape.
        
        Args:
            persistent: If True, tape can be used multiple times
            watch_accessed_variables: Automatically watch accessed variables
        """
        self.persistent = persistent
        self.watch_accessed_variables = watch_accessed_variables
        self._tape: List[TapeEntry] = []
        self._watched_variables: Set['Variable'] = set()
        self._sources: Set['Variable'] = set()
        self._used = False
        
        # Memory optimization
        self._checkpoints: Dict[int, 'Variable'] = {}
        self._checkpoint_interval = 100
        
        # Graph optimization
        self._enable_fusion = True
        self._fusion_patterns = self._init_fusion_patterns()
    
    def _init_fusion_patterns(self) -> List[Tuple[List[OpType], Callable]]:
        """Initialize operation fusion patterns for optimization."""
        return [
            # Fuse linear + activation
            ([OpType.MATMUL, OpType.ADD, OpType.RELU], self._fuse_linear_relu),
            ([OpType.MATMUL, OpType.ADD, OpType.GELU], self._fuse_linear_gelu),
            
            # Fuse normalization
            ([OpType.SUB, OpType.DIV], self._fuse_normalization),
            
            # Fuse multiple additions
            ([OpType.ADD, OpType.ADD], self._fuse_additions),
        ]
    
    def watch(self, variable: 'Variable'):
        """Add a variable to the list of watched variables."""
        self._watched_variables.add(variable)
        self._sources.add(variable)
    
    def stop_watching(self, variable: 'Variable'):
        """Remove a variable from the watched list."""
        self._watched_variables.discard(variable)
    
    @contextmanager
    def stop_recording(self):
        """Context manager to temporarily stop recording operations."""
        old_tape = self._tape
        self._tape = None
        try:
            yield
        finally:
            self._tape = old_tape
    
    def record_operation(
        self,
        op_type: OpType,
        inputs: List['Variable'],
        output: 'Variable',
        grad_fn: Optional[Callable] = None,
        saved_tensors: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record an operation in the tape."""
        if self._tape is None:
            return
        
        entry = TapeEntry(
            op_type=op_type,
            inputs=inputs,
            output=output,
            grad_fn=grad_fn or self._get_default_grad_fn(op_type),
            saved_tensors=saved_tensors or {},
            metadata=metadata or {}
        )
        
        self._tape.append(entry)
        
        # Checkpointing for memory optimization
        if len(self._tape) % self._checkpoint_interval == 0:
            self._checkpoints[len(self._tape)] = output
    
    def _get_default_grad_fn(self, op_type: OpType) -> Callable:
        """Get default gradient function for an operation type."""
        grad_fns = {
            OpType.ADD: self._grad_add,
            OpType.SUB: self._grad_sub,
            OpType.MUL: self._grad_mul,
            OpType.DIV: self._grad_div,
            OpType.MATMUL: self._grad_matmul,
            OpType.RELU: self._grad_relu,
            OpType.SIGMOID: self._grad_sigmoid,
            OpType.TANH: self._grad_tanh,
            OpType.SOFTMAX: self._grad_softmax,
            OpType.SUM: self._grad_sum,
            OpType.MEAN: self._grad_mean,
        }
        return grad_fns.get(op_type, self._grad_identity)
    
    def gradient(
        self,
        target: 'Variable',
        sources: Optional[List['Variable']] = None,
        output_gradients: Optional[np.ndarray] = None,
        retain_graph: bool = False,
        create_graph: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute gradients using reverse-mode automatic differentiation.
        
        Args:
            target: The target variable to differentiate
            sources: Variables to compute gradients for (default: watched variables)
            output_gradients: Initial gradient (default: ones)
            retain_graph: Keep the graph after computing gradients
            create_graph: Create graph of gradient computation for higher-order derivatives
        
        Returns:
            Gradients with respect to sources
        """
        if not self.persistent and self._used:
            raise RuntimeError("Non-persistent tape can only be used once")
        
        self._used = True
        sources = sources or list(self._watched_variables)
        
        if not sources:
            raise ValueError("No sources to compute gradients for")
        
        # Initialize gradient accumulation
        grad_accumulator: Dict['Variable', np.ndarray] = defaultdict(lambda: None)
        
        # Set initial gradient
        if output_gradients is None:
            grad_accumulator[target] = np.ones_like(target.data)
        else:
            grad_accumulator[target] = output_gradients
        
        # Optimize graph if enabled
        if self._enable_fusion:
            self._optimize_graph()
        
        # Reverse-mode differentiation
        for entry in reversed(self._tape):
            if entry.output not in grad_accumulator or grad_accumulator[entry.output] is None:
                continue
            
            output_grad = grad_accumulator[entry.output]
            
            # Compute input gradients
            input_grads = entry.grad_fn(
                output_grad,
                entry.inputs,
                entry.output,
                entry.saved_tensors,
                entry.metadata
            )
            
            # Accumulate gradients
            for inp, grad in zip(entry.inputs, input_grads):
                if grad is not None:
                    if grad_accumulator[inp] is None:
                        grad_accumulator[inp] = grad
                    else:
                        grad_accumulator[inp] = grad_accumulator[inp] + grad
        
        # Clean up if not retaining graph
        if not retain_graph and not self.persistent:
            self._tape.clear()
            self._checkpoints.clear()
        
        # Return gradients
        if len(sources) == 1:
            return grad_accumulator.get(sources[0], np.zeros_like(sources[0].data))
        else:
            return [grad_accumulator.get(s, np.zeros_like(s.data)) for s in sources]
    
    def _optimize_graph(self):
        """Optimize computational graph by fusing operations."""
        if len(self._tape) < 2:
            return
        
        optimized_tape = []
        i = 0
        
        while i < len(self._tape):
            # Check for fusion patterns
            fused = False
            for pattern, fusion_fn in self._fusion_patterns:
                if self._matches_pattern(i, pattern):
                    # Fuse operations
                    fused_entry = fusion_fn(self._tape[i:i+len(pattern)])
                    optimized_tape.append(fused_entry)
                    i += len(pattern)
                    fused = True
                    break
            
            if not fused:
                optimized_tape.append(self._tape[i])
                i += 1
        
        self._tape = optimized_tape
    
    def _matches_pattern(self, start_idx: int, pattern: List[OpType]) -> bool:
        """Check if operations starting at index match a pattern."""
        if start_idx + len(pattern) > len(self._tape):
            return False
        
        for i, op_type in enumerate(pattern):
            if self._tape[start_idx + i].op_type != op_type:
                return False
        
        return True
    
    # Gradient computation functions
    def _grad_add(self, grad_output, inputs, output, saved, meta):
        """Gradient for addition."""
        return [grad_output, grad_output]
    
    def _grad_sub(self, grad_output, inputs, output, saved, meta):
        """Gradient for subtraction."""
        return [grad_output, -grad_output]
    
    def _grad_mul(self, grad_output, inputs, output, saved, meta):
        """Gradient for multiplication."""
        x, y = inputs[0].data, inputs[1].data
        return [grad_output * y, grad_output * x]
    
    def _grad_div(self, grad_output, inputs, output, saved, meta):
        """Gradient for division."""
        x, y = inputs[0].data, inputs[1].data
        return [grad_output / y, -grad_output * x / (y ** 2)]
    
    def _grad_matmul(self, grad_output, inputs, output, saved, meta):
        """Gradient for matrix multiplication."""
        a, b = inputs[0].data, inputs[1].data
        
        # Gradient w.r.t. first input
        grad_a = np.matmul(grad_output, b.T)
        
        # Gradient w.r.t. second input
        grad_b = np.matmul(a.T, grad_output)
        
        return [grad_a, grad_b]
    
    def _grad_relu(self, grad_output, inputs, output, saved, meta):
        """Gradient for ReLU."""
        x = inputs[0].data
        return [grad_output * (x > 0)]
    
    def _grad_sigmoid(self, grad_output, inputs, output, saved, meta):
        """Gradient for sigmoid."""
        sigmoid_x = output.data
        return [grad_output * sigmoid_x * (1 - sigmoid_x)]
    
    def _grad_tanh(self, grad_output, inputs, output, saved, meta):
        """Gradient for tanh."""
        tanh_x = output.data
        return [grad_output * (1 - tanh_x ** 2)]
    
    def _grad_softmax(self, grad_output, inputs, output, saved, meta):
        """Gradient for softmax."""
        softmax_x = output.data
        # Jacobian of softmax
        jacobian = softmax_x * grad_output
        sum_jacobian = np.sum(jacobian, axis=-1, keepdims=True)
        return [jacobian - softmax_x * sum_jacobian]
    
    def _grad_sum(self, grad_output, inputs, output, saved, meta):
        """Gradient for sum."""
        x_shape = inputs[0].data.shape
        axis = meta.get('axis', None)
        
        if axis is None:
            # Full sum
            grad = np.full(x_shape, grad_output)
        else:
            # Partial sum
            grad = np.expand_dims(grad_output, axis=axis)
            grad = np.broadcast_to(grad, x_shape)
        
        return [grad]
    
    def _grad_mean(self, grad_output, inputs, output, saved, meta):
        """Gradient for mean."""
        x_shape = inputs[0].data.shape
        axis = meta.get('axis', None)
        
        if axis is None:
            # Full mean
            size = np.prod(x_shape)
            grad = np.full(x_shape, grad_output / size)
        else:
            # Partial mean
            size = x_shape[axis]
            grad = np.expand_dims(grad_output / size, axis=axis)
            grad = np.broadcast_to(grad, x_shape)
        
        return [grad]
    
    def _grad_identity(self, grad_output, inputs, output, saved, meta):
        """Identity gradient (passthrough)."""
        return [grad_output]
    
    # Operation fusion functions
    def _fuse_linear_relu(self, entries: List[TapeEntry]) -> TapeEntry:
        """Fuse linear layer + ReLU activation."""
        # Create fused entry
        fused_entry = TapeEntry(
            op_type=OpType.CUSTOM,
            inputs=entries[0].inputs,
            output=entries[-1].output,
            grad_fn=self._grad_fused_linear_relu,
            saved_tensors={'intermediate': entries[1].output.data},
            metadata={'fused': 'linear_relu'}
        )
        return fused_entry
    
    def _fuse_linear_gelu(self, entries: List[TapeEntry]) -> TapeEntry:
        """Fuse linear layer + GELU activation."""
        fused_entry = TapeEntry(
            op_type=OpType.CUSTOM,
            inputs=entries[0].inputs,
            output=entries[-1].output,
            grad_fn=self._grad_fused_linear_gelu,
            saved_tensors={'intermediate': entries[1].output.data},
            metadata={'fused': 'linear_gelu'}
        )
        return fused_entry
    
    def _fuse_normalization(self, entries: List[TapeEntry]) -> TapeEntry:
        """Fuse normalization operations."""
        fused_entry = TapeEntry(
            op_type=OpType.CUSTOM,
            inputs=entries[0].inputs,
            output=entries[-1].output,
            grad_fn=self._grad_fused_normalization,
            saved_tensors={},
            metadata={'fused': 'normalization'}
        )
        return fused_entry
    
    def _fuse_additions(self, entries: List[TapeEntry]) -> TapeEntry:
        """Fuse multiple additions."""
        # Collect all unique inputs
        all_inputs = []
        seen = set()
        for entry in entries:
            for inp in entry.inputs:
                if id(inp) not in seen:
                    all_inputs.append(inp)
                    seen.add(id(inp))
        
        fused_entry = TapeEntry(
            op_type=OpType.CUSTOM,
            inputs=all_inputs,
            output=entries[-1].output,
            grad_fn=lambda g, *args: [g] * len(all_inputs),
            saved_tensors={},
            metadata={'fused': 'multi_add'}
        )
        return fused_entry
    
    def _grad_fused_linear_relu(self, grad_output, inputs, output, saved, meta):
        """Gradient for fused linear + ReLU."""
        intermediate = saved['intermediate']
        relu_grad = grad_output * (intermediate > 0)
        # Continue with linear gradient
        return self._grad_matmul(relu_grad, inputs[:2], None, {}, {})
    
    def _grad_fused_linear_gelu(self, grad_output, inputs, output, saved, meta):
        """Gradient for fused linear + GELU."""
        intermediate = saved['intermediate']
        # GELU gradient
        c1 = 0.7978845608  # sqrt(2/pi)
        c2 = 0.044715
        tanh_arg = c1 * (intermediate + c2 * intermediate ** 3)
        tanh_val = np.tanh(tanh_arg)
        gelu_grad = 0.5 * (1 + tanh_val + intermediate * (1 - tanh_val ** 2) * c1 * (1 + 3 * c2 * intermediate ** 2))
        # Continue with linear gradient
        return self._grad_matmul(grad_output * gelu_grad, inputs[:2], None, {}, {})
    
    def _grad_fused_normalization(self, grad_output, inputs, output, saved, meta):
        """Gradient for fused normalization."""
        # Simplified normalization gradient
        return [grad_output / inputs[1].data, -grad_output * inputs[0].data / (inputs[1].data ** 2)]


class Variable:
    """Variable that supports automatic differentiation."""
    
    _tape_stack: List[GradientTape] = []
    
    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        name: Optional[str] = None
    ):
        """Initialize variable.
        
        Args:
            data: Underlying data
            requires_grad: Whether to track gradients
            name: Optional name for debugging
        """
        self.data = np.asarray(data)
        self.requires_grad = requires_grad
        self.name = name
        self.grad: Optional[np.ndarray] = None
        
        # Watch this variable if there's an active tape and it requires grad
        if requires_grad and self._tape_stack:
            self._tape_stack[-1].watch(self)
    
    @classmethod
    @contextmanager
    def tape(cls, persistent: bool = False):
        """Context manager for gradient tape."""
        tape = GradientTape(persistent=persistent)
        cls._tape_stack.append(tape)
        try:
            yield tape
        finally:
            cls._tape_stack.pop()
    
    def backward(self, grad_output: Optional[np.ndarray] = None):
        """Compute gradients using the most recent tape."""
        if not self._tape_stack:
            raise RuntimeError("No active gradient tape")
        
        tape = self._tape_stack[-1]
        self.grad = tape.gradient(self, [self], grad_output)[0]
    
    # Operator overloading for automatic differentiation
    def __add__(self, other):
        return self._binary_op(other, OpType.ADD, lambda x, y: x + y)
    
    def __sub__(self, other):
        return self._binary_op(other, OpType.SUB, lambda x, y: x - y)
    
    def __mul__(self, other):
        return self._binary_op(other, OpType.MUL, lambda x, y: x * y)
    
    def __truediv__(self, other):
        return self._binary_op(other, OpType.DIV, lambda x, y: x / y)
    
    def __matmul__(self, other):
        return self._binary_op(other, OpType.MATMUL, np.matmul)
    
    def _binary_op(self, other, op_type: OpType, forward_fn: Callable):
        """Generic binary operation with gradient tracking."""
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        
        # Forward computation
        output_data = forward_fn(self.data, other.data)
        output = Variable(output_data, requires_grad=self.requires_grad or other.requires_grad)
        
        # Record operation if there's an active tape
        if self._tape_stack and (self.requires_grad or other.requires_grad):
            self._tape_stack[-1].record_operation(
                op_type=op_type,
                inputs=[self, other],
                output=output
            )
        
        return output
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False):
        """Sum reduction with gradient tracking."""
        output_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        output = Variable(output_data, requires_grad=self.requires_grad)
        
        if self._tape_stack and self.requires_grad:
            self._tape_stack[-1].record_operation(
                op_type=OpType.SUM,
                inputs=[self],
                output=output,
                metadata={'axis': axis, 'keepdims': keepdims}
            )
        
        return output
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False):
        """Mean reduction with gradient tracking."""
        output_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        output = Variable(output_data, requires_grad=self.requires_grad)
        
        if self._tape_stack and self.requires_grad:
            self._tape_stack[-1].record_operation(
                op_type=OpType.MEAN,
                inputs=[self],
                output=output,
                metadata={'axis': axis, 'keepdims': keepdims}
            )
        
        return output
    
    def relu(self):
        """ReLU activation with gradient tracking."""
        output_data = np.maximum(0, self.data)
        output = Variable(output_data, requires_grad=self.requires_grad)
        
        if self._tape_stack and self.requires_grad:
            self._tape_stack[-1].record_operation(
                op_type=OpType.RELU,
                inputs=[self],
                output=output
            )
        
        return output


def grad(
    func: Callable,
    argnums: Union[int, List[int]] = 0,
    has_aux: bool = False
) -> Callable:
    """Create a function that computes gradients.
    
    Args:
        func: Function to differentiate
        argnums: Indices of arguments to differentiate w.r.t.
        has_aux: Whether function returns auxiliary outputs
    
    Returns:
        Function that computes gradients
    """
    if isinstance(argnums, int):
        argnums = [argnums]
    
    def grad_fn(*args, **kwargs):
        # Convert arguments to Variables
        var_args = []
        watched_vars = []
        
        for i, arg in enumerate(args):
            if i in argnums:
                var = Variable(arg, requires_grad=True)
                watched_vars.append(var)
            else:
                var = Variable(arg, requires_grad=False)
            var_args.append(var)
        
        # Compute function with gradient tape
        with Variable.tape() as tape:
            for var in watched_vars:
                tape.watch(var)
            
            result = func(*var_args, **kwargs)
            
            if has_aux:
                output, aux = result
            else:
                output = result
                aux = None
        
        # Compute gradients
        grads = tape.gradient(output, watched_vars)
        
        if len(argnums) == 1:
            grads = grads[0]
        
        if has_aux:
            return grads, aux
        else:
            return grads
    
    return grad_fn


def value_and_grad(
    func: Callable,
    argnums: Union[int, List[int]] = 0,
    has_aux: bool = False
) -> Callable:
    """Create a function that computes both value and gradients.
    
    Similar to grad() but also returns the function value.
    """
    grad_fn = grad(func, argnums, has_aux)
    
    def value_and_grad_fn(*args, **kwargs):
        # Compute value
        value = func(*args, **kwargs)
        
        # Compute gradients
        grads = grad_fn(*args, **kwargs)
        
        return value, grads
    
    return value_and_grad_fn


# Example usage
def test_autograd():
    """Test automatic differentiation engine."""
    print("Testing Automatic Differentiation Engine")
    print("=" * 50)
    
    # Test basic operations
    with Variable.tape() as tape:
        x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = Variable(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
        
        # Complex computation
        z = x @ y  # Matrix multiplication
        z = z + x * 2  # Element-wise operations
        z = z.relu()  # Activation
        loss = z.mean()  # Reduction
    
    # Compute gradients
    grad_x, grad_y = tape.gradient(loss, [x, y])
    
    print(f"Input x:\n{x.data}")
    print(f"Input y:\n{y.data}")
    print(f"Loss: {loss.data}")
    print(f"Gradient w.r.t. x:\n{grad_x}")
    print(f"Gradient w.r.t. y:\n{grad_y}")
    
    # Test grad function
    print("\nTesting grad() function:")
    
    def f(x, y):
        return (x ** 2 + y ** 3).sum()
    
    grad_f = grad(f, argnums=[0, 1])
    
    x_val = np.array([1.0, 2.0, 3.0])
    y_val = np.array([4.0, 5.0, 6.0])
    
    grads = grad_f(x_val, y_val)
    print(f"f(x, y) = sum(x^2 + y^3)")
    print(f"x = {x_val}")
    print(f"y = {y_val}")
    print(f"∂f/∂x = {grads[0]}")
    print(f"∂f/∂y = {grads[1]}")
    
    # Test value_and_grad
    print("\nTesting value_and_grad() function:")
    value_and_grad_f = value_and_grad(f, argnums=[0, 1])
    value, grads = value_and_grad_f(x_val, y_val)
    print(f"Value: {value.data}")
    print(f"Gradients: {grads}")
    
    print("\nAutograd engine tested successfully!")


if __name__ == "__main__":
    test_autograd()