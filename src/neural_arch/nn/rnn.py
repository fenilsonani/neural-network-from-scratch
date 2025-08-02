"""Recurrent Neural Network (RNN) layer implementations."""

import math
from typing import Optional, Tuple, Union

import numpy as np

from ..core import Parameter, Tensor
from .module import Module
from ..core.tensor import GradientFunction
from ..exceptions import LayerError, handle_exception
from ..functional import add, matmul, tanh, sigmoid


class RNNCell(Module):
    """Basic RNN cell with configurable activation function.
    
    Implements the basic RNN cell computation:
        h_new = activation(input @ W_ih + h @ W_hh + b_ih + b_hh)
    
    Args:
        input_size: Size of the input features
        hidden_size: Size of the hidden state
        bias: Whether to use bias parameters
        nonlinearity: Type of nonlinearity ('tanh' or 'relu')
        weight_init: Weight initialization scheme
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        weight_init: str = "xavier_uniform",
        name: Optional[str] = None,
    ):
        super().__init__()
        
        # Validate parameters
        if input_size <= 0:
            raise LayerError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise LayerError(f"hidden_size must be positive, got {hidden_size}")
        if nonlinearity not in ["tanh", "relu"]:
            raise LayerError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity}")
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias
        self.nonlinearity = nonlinearity
        self.name = name or f"RNNCell({input_size}, {hidden_size})"
        
        # Initialize weight parameters
        # Input-to-hidden weights
        self.weight_ih = Parameter(
            self._initialize_weights(weight_init, (hidden_size, input_size)),
            name=f"{self.name}.weight_ih"
        )
        
        # Hidden-to-hidden weights  
        self.weight_hh = Parameter(
            self._initialize_weights(weight_init, (hidden_size, hidden_size)),
            name=f"{self.name}.weight_hh"
        )
        
        # Bias parameters (optional)
        if bias:
            self.bias_ih = Parameter(
                np.zeros(hidden_size, dtype=np.float32),
                name=f"{self.name}.bias_ih"
            )
            self.bias_hh = Parameter(
                np.zeros(hidden_size, dtype=np.float32),
                name=f"{self.name}.bias_hh"
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
    
    def _initialize_weights(self, init_scheme: str, shape: Tuple[int, int]) -> np.ndarray:
        """Initialize weight matrix using specified scheme."""
        fan_out, fan_in = shape
        
        if init_scheme == "xavier_uniform":
            # Xavier/Glorot uniform initialization
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, shape).astype(np.float32)
            
        elif init_scheme == "xavier_normal":
            # Xavier/Glorot normal initialization
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(0.0, std, shape).astype(np.float32)
            
        elif init_scheme == "he_uniform":
            # He uniform initialization (good for ReLU)
            limit = math.sqrt(6.0 / fan_in)
            return np.random.uniform(-limit, limit, shape).astype(np.float32)
            
        elif init_scheme == "he_normal":
            # He normal initialization
            std = math.sqrt(2.0 / fan_in)
            return np.random.normal(0.0, std, shape).astype(np.float32)
            
        else:
            raise LayerError(f"Unknown weight initialization scheme: {init_scheme}")
    
    def _apply_activation(self, x: Tensor) -> Tensor:
        """Apply the specified activation function."""
        if self.nonlinearity == "tanh":
            return tanh(x)
        elif self.nonlinearity == "relu":
            # Simple ReLU implementation
            relu_data = np.maximum(0, x.data)
            output = Tensor(relu_data, requires_grad=x.requires_grad)
            
            if output.requires_grad:
                def relu_backward(grad_output):
                    if x.requires_grad:
                        if x.grad is None:
                            x.grad = np.zeros_like(x.data)
                        x.grad += grad_output * (x.data > 0).astype(np.float32)
                
                output._grad_fn = GradientFunction(relu_backward, [input], "rnn")
            
            return output
        else:
            raise LayerError(f"Unsupported activation: {self.nonlinearity}")
    
    @handle_exception
    def forward(self, input: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """Forward pass through RNN cell.
        
        Args:
            input: Input tensor of shape (batch_size, input_size)
            hidden: Hidden state tensor of shape (batch_size, hidden_size).
                   If None, initialized to zeros.
                   
        Returns:
            new_hidden: New hidden state of shape (batch_size, hidden_size)
        """
        # Validate input shape
        if len(input.shape) != 2:
            raise LayerError(f"Expected 2D input (batch, input_size), got {len(input.shape)}D")
        
        if input.shape[1] != self.input_size:
            raise LayerError(f"Input size mismatch: expected {self.input_size}, got {input.shape[1]}")
        
        batch_size = input.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden_data = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
            hidden = Tensor(hidden_data, requires_grad=input.requires_grad)
        else:
            # Validate hidden shape
            if hidden.shape != (batch_size, self.hidden_size):
                raise LayerError(
                    f"Hidden shape mismatch: expected ({batch_size}, {self.hidden_size}), "
                    f"got {hidden.shape}"
                )
        
        # Compute input-to-hidden transformation: input @ W_ih^T
        ih_output = matmul(input, self.weight_ih.T)
        
        # Compute hidden-to-hidden transformation: hidden @ W_hh^T
        hh_output = matmul(hidden, self.weight_hh.T)
        
        # Add bias terms if present
        if self.bias_ih is not None:
            ih_output = add(ih_output, self.bias_ih)
        if self.bias_hh is not None:
            hh_output = add(hh_output, self.bias_hh)
        
        # Combine and apply activation
        combined = add(ih_output, hh_output)
        new_hidden = self._apply_activation(combined)
        
        return new_hidden


class RNN(Module):
    """Multi-layer RNN with support for bidirectional processing.
    
    Args:
        input_size: Size of the input features
        hidden_size: Size of the hidden state
        num_layers: Number of RNN layers
        bias: Whether to use bias parameters
        batch_first: If True, input shape is (batch, seq, features)
        dropout: Dropout probability between layers (if num_layers > 1)
        bidirectional: Whether to use bidirectional RNN
        nonlinearity: Type of nonlinearity ('tanh' or 'relu')
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
        weight_init: str = "xavier_uniform",
        name: Optional[str] = None,
    ):
        super().__init__()
        
        # Validate parameters
        if input_size <= 0:
            raise LayerError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise LayerError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise LayerError(f"num_layers must be positive, got {num_layers}")
        if not 0.0 <= dropout <= 1.0:
            raise LayerError(f"dropout must be in [0, 1], got {dropout}")
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.name = name or f"RNN({input_size}, {hidden_size})"
        
        # Create RNN cells for each layer and direction
        self.cells = []
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            # Forward direction
            forward_cell = RNNCell(
                layer_input_size, 
                hidden_size, 
                bias=bias,
                nonlinearity=nonlinearity,
                weight_init=weight_init,
                name=f"{self.name}.layer_{layer}_forward"
            )
            self.cells.append(forward_cell)
            
            # Backward direction (if bidirectional)
            if bidirectional:
                backward_cell = RNNCell(
                    layer_input_size,
                    hidden_size,
                    bias=bias, 
                    nonlinearity=nonlinearity,
                    weight_init=weight_init,
                    name=f"{self.name}.layer_{layer}_backward"
                )
                self.cells.append(backward_cell)
        
        # Register cells as submodules
        for i, cell in enumerate(self.cells):
            self.register_module(f"cell_{i}", cell)
    
    def _apply_dropout(self, x: Tensor, training: bool) -> Tensor:
        """Apply dropout if specified and in training mode."""
        if self.dropout > 0.0 and training:
            keep_prob = 1.0 - self.dropout
            mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
            return Tensor(x.data * mask, requires_grad=x.requires_grad)
        return x
    
    @handle_exception  
    def forward(
        self, 
        input: Tensor, 
        h_0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through multi-layer RNN.
        
        Args:
            input: Input tensor. Shape depends on batch_first:
                  - If batch_first=True: (batch, seq_len, input_size)
                  - If batch_first=False: (seq_len, batch, input_size)
            h_0: Initial hidden state of shape (num_layers * num_directions, batch, hidden_size)
                If None, initialized to zeros.
                
        Returns:
            output: Output tensor with all timesteps
            h_n: Final hidden state  
        """
        # Handle input shape based on batch_first
        if self.batch_first:
            # Convert to (seq_len, batch, input_size)
            input_data = np.transpose(input.data, (1, 0, 2))
        else:
            input_data = input.data
            
        seq_len, batch_size, _ = input_data.shape
        
        # Initialize hidden states if not provided
        num_directions = 2 if self.bidirectional else 1
        if h_0 is None:
            h_0_data = np.zeros(
                (self.num_layers * num_directions, batch_size, self.hidden_size), 
                dtype=np.float32
            )
            h_0 = Tensor(h_0_data, requires_grad=input.requires_grad)
        
        # Split initial hidden states by layer and direction
        hidden_states = []
        for i in range(self.num_layers * num_directions):
            hidden_states.append(Tensor(h_0.data[i], requires_grad=h_0.requires_grad))
        
        # Process through layers
        layer_input = input_data
        layer_outputs = []
        final_hidden_states = []
        
        for layer in range(self.num_layers):
            layer_output_forward = []
            layer_output_backward = []
            
            # Forward direction
            forward_cell_idx = layer * num_directions
            forward_cell = self.cells[forward_cell_idx]
            forward_hidden = hidden_states[forward_cell_idx]
            
            for t in range(seq_len):
                step_input = Tensor(layer_input[t], requires_grad=input.requires_grad)
                forward_hidden = forward_cell(step_input, forward_hidden)
                layer_output_forward.append(forward_hidden.data)
            
            final_hidden_states.append(forward_hidden)
            
            # Backward direction (if bidirectional)
            if self.bidirectional:
                backward_cell_idx = layer * num_directions + 1
                backward_cell = self.cells[backward_cell_idx]
                backward_hidden = hidden_states[backward_cell_idx]
                
                for t in range(seq_len - 1, -1, -1):
                    step_input = Tensor(layer_input[t], requires_grad=input.requires_grad)
                    backward_hidden = backward_cell(step_input, backward_hidden)
                    layer_output_backward.insert(0, backward_hidden.data)
                
                final_hidden_states.append(backward_hidden)
                
                # Concatenate forward and backward outputs
                layer_output = []
                for t in range(seq_len):
                    concatenated = np.concatenate([
                        layer_output_forward[t], 
                        layer_output_backward[t]
                    ], axis=1)
                    layer_output.append(concatenated)
                layer_input = np.array(layer_output)
            else:
                layer_input = np.array(layer_output_forward)
            
            # Apply dropout between layers (except last layer)
            if layer < self.num_layers - 1 and self.dropout > 0.0:
                for t in range(seq_len):
                    step_tensor = Tensor(layer_input[t], requires_grad=input.requires_grad)
                    step_tensor = self._apply_dropout(step_tensor, self.training)
                    layer_input[t] = step_tensor.data
        
        # Prepare output
        if self.batch_first:
            # Convert back to (batch, seq_len, features)
            output_data = np.transpose(layer_input, (1, 0, 2))
        else:
            output_data = layer_input
            
        output = Tensor(output_data, requires_grad=input.requires_grad)
        
        # Stack final hidden states
        final_hidden_data = np.stack([h.data for h in final_hidden_states], axis=0)
        h_n = Tensor(final_hidden_data, requires_grad=input.requires_grad)
        
        return output, h_n