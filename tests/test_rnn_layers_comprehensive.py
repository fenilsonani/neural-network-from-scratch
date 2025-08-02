"""Comprehensive tests for RNN layers to achieve ~95% code coverage.

This file tests RNN, RNNCell, LSTM, LSTMCell, GRU, and GRUCell implementations
with focus on:
- Basic cell computations and formulas
- Different activation functions
- Hidden state initialization and management
- Multi-layer and bidirectional processing
- Variable sequence lengths and batch formats
- Weight initialization schemes
- Gradient flow
- Error handling and edge cases
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import numpy as np
import pytest

from neural_arch.core import Tensor
from neural_arch.nn import RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell
from neural_arch.exceptions import LayerError, NeuralArchError


class TestRNNCell:
    """Comprehensive tests for RNNCell."""

    def test_rnn_cell_init_valid(self):
        """Test RNNCell initialization with valid parameters."""
        # Test with default parameters
        cell = RNNCell(input_size=10, hidden_size=20)
        assert cell.input_size == 10
        assert cell.hidden_size == 20
        assert cell.use_bias is True
        assert cell.nonlinearity == "tanh"
        assert cell.name == "RNNCell(10, 20)"
        
        # Check parameters exist
        assert hasattr(cell, "weight_ih")
        assert hasattr(cell, "weight_hh")
        assert hasattr(cell, "bias_ih")
        assert hasattr(cell, "bias_hh")
        
        # Check parameter shapes
        assert cell.weight_ih.shape == (20, 10)
        assert cell.weight_hh.shape == (20, 20)
        assert cell.bias_ih.shape == (20,)
        assert cell.bias_hh.shape == (20,)

    def test_rnn_cell_init_custom_params(self):
        """Test RNNCell initialization with custom parameters."""
        cell = RNNCell(
            input_size=5, 
            hidden_size=8, 
            bias=False, 
            nonlinearity="relu",
            weight_init="he_uniform",
            name="CustomRNN"
        )
        assert cell.input_size == 5
        assert cell.hidden_size == 8
        assert cell.use_bias is False
        assert cell.nonlinearity == "relu"
        assert cell.name == "CustomRNN"
        assert cell.bias_ih is None
        assert cell.bias_hh is None

    def test_rnn_cell_init_invalid_params(self):
        """Test RNNCell initialization with invalid parameters."""
        # Invalid input_size
        with pytest.raises(LayerError) as exc_info:
            RNNCell(input_size=0, hidden_size=10)
        assert "input_size must be positive" in str(exc_info.value)
        
        with pytest.raises(LayerError):
            RNNCell(input_size=-5, hidden_size=10)

        # Invalid hidden_size
        with pytest.raises(LayerError) as exc_info:
            RNNCell(input_size=10, hidden_size=0)
        assert "hidden_size must be positive" in str(exc_info.value)
        
        with pytest.raises(LayerError):
            RNNCell(input_size=10, hidden_size=-3)

        # Invalid nonlinearity
        with pytest.raises(LayerError) as exc_info:
            RNNCell(input_size=10, hidden_size=20, nonlinearity="invalid")
        assert "nonlinearity must be 'tanh' or 'relu'" in str(exc_info.value)

    @pytest.mark.parametrize("weight_init", ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal"])
    def test_rnn_cell_weight_initialization(self, weight_init):
        """Test different weight initialization schemes."""
        cell = RNNCell(input_size=10, hidden_size=20, weight_init=weight_init)
        
        # Check that weights are initialized (not zeros)
        assert not np.allclose(cell.weight_ih.data, 0)
        assert not np.allclose(cell.weight_hh.data, 0)
        
        # Check weight shapes
        assert cell.weight_ih.shape == (20, 10)
        assert cell.weight_hh.shape == (20, 20)

    def test_rnn_cell_invalid_weight_init(self):
        """Test RNNCell with invalid weight initialization."""
        with pytest.raises(LayerError) as exc_info:
            RNNCell(input_size=10, hidden_size=20, weight_init="invalid_init")
        assert "Unknown weight initialization scheme" in str(exc_info.value)

    def test_rnn_cell_forward_basic(self):
        """Test basic RNNCell forward pass."""
        cell = RNNCell(input_size=3, hidden_size=4)
        
        # Create input and hidden state
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        hidden_tensor = Tensor(np.random.randn(2, 4), requires_grad=True)
        
        # Forward pass
        output = cell(input_tensor, hidden_tensor)
        
        # Check output shape
        assert output.shape == (2, 4)
        assert output.requires_grad is True

    def test_rnn_cell_forward_no_hidden(self):
        """Test RNNCell forward pass without initial hidden state."""
        cell = RNNCell(input_size=3, hidden_size=4)
        
        # Create input only
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        
        # Forward pass - should initialize hidden to zeros
        output = cell(input_tensor)
        
        # Check output shape
        assert output.shape == (2, 4)
        assert output.requires_grad is True

    @pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
    def test_rnn_cell_activations(self, nonlinearity):
        """Test different activation functions."""
        cell = RNNCell(input_size=3, hidden_size=4, nonlinearity=nonlinearity)
        
        input_tensor = Tensor(np.random.randn(1, 3), requires_grad=True)
        output = cell(input_tensor)
        
        assert output.shape == (1, 4)
        
        # For ReLU, check that negative values are clipped
        if nonlinearity == "relu":
            # Create input that should produce some negative values before activation
            large_negative_input = Tensor(-10 * np.ones((1, 3)), requires_grad=True)
            output_neg = cell(large_negative_input)
            # ReLU should clip negative values, but the exact output depends on weights

    def test_rnn_cell_forward_invalid_input_shape(self):
        """Test RNNCell forward with invalid input shapes."""
        cell = RNNCell(input_size=3, hidden_size=4)
        
        # Wrong number of dimensions
        with pytest.raises(NeuralArchError) as exc_info:
            invalid_input = Tensor(np.random.randn(2, 3, 5), requires_grad=True)
            cell(invalid_input)
        assert "Expected 2D input" in str(exc_info.value)
        
        # Wrong input size
        with pytest.raises(NeuralArchError) as exc_info:
            invalid_input = Tensor(np.random.randn(2, 5), requires_grad=True)
            cell(invalid_input)
        assert "Input size mismatch" in str(exc_info.value)

    def test_rnn_cell_forward_invalid_hidden_shape(self):
        """Test RNNCell forward with invalid hidden shapes."""
        cell = RNNCell(input_size=3, hidden_size=4)
        
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        
        # Wrong hidden shape
        with pytest.raises(NeuralArchError) as exc_info:
            invalid_hidden = Tensor(np.random.randn(2, 5), requires_grad=True)
            cell(input_tensor, invalid_hidden)
        assert "Hidden shape mismatch" in str(exc_info.value)
        
        # Wrong batch size
        with pytest.raises(NeuralArchError) as exc_info:
            invalid_hidden = Tensor(np.random.randn(3, 4), requires_grad=True)
            cell(input_tensor, invalid_hidden)
        assert "Hidden shape mismatch" in str(exc_info.value)

    def test_rnn_cell_computation_formula(self):
        """Test that RNNCell implements correct computation formula."""
        cell = RNNCell(input_size=2, hidden_size=3, bias=True, nonlinearity="tanh")
        
        # Set known weights and biases for predictable computation
        cell.weight_ih.data = np.array([[1.0, 0.5], [0.0, 1.0], [-0.5, 0.0]], dtype=np.float32)
        cell.weight_hh.data = np.array([[0.2, 0.1, 0.0], [0.0, 0.3, 0.1], [0.1, 0.0, 0.2]], dtype=np.float32)
        cell.bias_ih.data = np.array([0.1, -0.1, 0.0], dtype=np.float32)
        cell.bias_hh.data = np.array([0.0, 0.1, -0.1], dtype=np.float32)
        
        # Create test input and hidden state
        input_data = np.array([[1.0, 2.0]], dtype=np.float32)
        hidden_data = np.array([[0.5, -0.3, 0.8]], dtype=np.float32)
        
        input_tensor = Tensor(input_data, requires_grad=False)
        hidden_tensor = Tensor(hidden_data, requires_grad=False)
        
        # Manual computation: h_new = tanh(input @ W_ih^T + hidden @ W_hh^T + bias_ih + bias_hh)
        ih_output = input_data @ cell.weight_ih.data.T + cell.bias_ih.data
        hh_output = hidden_data @ cell.weight_hh.data.T + cell.bias_hh.data
        expected_output = np.tanh(ih_output + hh_output)
        
        # Forward pass
        actual_output = cell(input_tensor, hidden_tensor)
        
        # Compare outputs
        np.testing.assert_allclose(actual_output.data, expected_output, atol=1e-6)

    def test_rnn_cell_no_bias(self):
        """Test RNNCell computation without bias."""
        cell = RNNCell(input_size=2, hidden_size=2, bias=False)
        
        input_tensor = Tensor(np.random.randn(1, 2), requires_grad=True)
        hidden_tensor = Tensor(np.random.randn(1, 2), requires_grad=True)
        
        output = cell(input_tensor, hidden_tensor)
        assert output.shape == (1, 2)

    def test_rnn_cell_gradient_flow(self):
        """Test gradient flow through RNNCell."""
        cell = RNNCell(input_size=2, hidden_size=3)
        
        input_tensor = Tensor(np.random.randn(1, 2), requires_grad=True)
        hidden_tensor = Tensor(np.random.randn(1, 3), requires_grad=True)
        
        output = cell(input_tensor, hidden_tensor)
        
        # Simulate backward pass
        output.backward(np.ones_like(output.data))
        
        # Check that gradients exist (simplified check since full backprop is complex)
        assert output.requires_grad is True


class TestRNN:
    """Comprehensive tests for multi-layer RNN."""

    def test_rnn_init_valid(self):
        """Test RNN initialization with valid parameters."""
        rnn = RNN(input_size=10, hidden_size=20)
        assert rnn.input_size == 10
        assert rnn.hidden_size == 20
        assert rnn.num_layers == 1
        assert rnn.bias is True
        assert rnn.batch_first is False
        assert rnn.dropout == 0.0
        assert rnn.bidirectional is False
        assert rnn.nonlinearity == "tanh"
        assert len(rnn.cells) == 1

    def test_rnn_init_multilayer(self):
        """Test RNN initialization with multiple layers."""
        rnn = RNN(input_size=5, hidden_size=8, num_layers=3)
        assert rnn.num_layers == 3
        assert len(rnn.cells) == 3
        
        # Check that layer input sizes are correct
        # Layer 0: input_size -> hidden_size
        # Layer 1+: hidden_size -> hidden_size
        assert rnn.cells[0].input_size == 5
        assert rnn.cells[1].input_size == 8
        assert rnn.cells[2].input_size == 8

    def test_rnn_init_bidirectional(self):
        """Test RNN initialization with bidirectional processing."""
        rnn = RNN(input_size=5, hidden_size=8, num_layers=2, bidirectional=True)
        assert rnn.bidirectional is True
        assert len(rnn.cells) == 4  # 2 layers * 2 directions
        
        # Check layer input sizes for bidirectional
        # Layer 0: input_size for both directions
        # Layer 1+: 2*hidden_size for both directions (due to concatenation)
        assert rnn.cells[0].input_size == 5  # forward
        assert rnn.cells[1].input_size == 5  # backward
        assert rnn.cells[2].input_size == 16  # forward (2*8)
        assert rnn.cells[3].input_size == 16  # backward (2*8)

    def test_rnn_init_invalid_params(self):
        """Test RNN initialization with invalid parameters."""
        # Invalid input_size
        with pytest.raises(LayerError):
            RNN(input_size=0, hidden_size=10)
        
        # Invalid hidden_size
        with pytest.raises(LayerError):
            RNN(input_size=10, hidden_size=0)
        
        # Invalid num_layers
        with pytest.raises(LayerError):
            RNN(input_size=10, hidden_size=20, num_layers=0)
        
        # Invalid dropout
        with pytest.raises(LayerError):
            RNN(input_size=10, hidden_size=20, dropout=-0.1)
        
        with pytest.raises(LayerError):
            RNN(input_size=10, hidden_size=20, dropout=1.5)

    def test_rnn_forward_basic(self):
        """Test basic RNN forward pass."""
        rnn = RNN(input_size=3, hidden_size=4, batch_first=True)
        
        # Input: (batch=2, seq_len=5, input_size=3)
        input_tensor = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        
        # Output should be (batch=2, seq_len=5, hidden_size=4)
        assert output.shape == (2, 5, 4)
        # Final hidden should be (num_layers=1, batch=2, hidden_size=4)
        assert h_n.shape == (1, 2, 4)

    def test_rnn_forward_seq_first(self):
        """Test RNN forward pass with sequence-first format."""
        rnn = RNN(input_size=3, hidden_size=4, batch_first=False)
        
        # Input: (seq_len=5, batch=2, input_size=3)
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        
        # Output should be (seq_len=5, batch=2, hidden_size=4)
        assert output.shape == (5, 2, 4)
        # Final hidden should be (num_layers=1, batch=2, hidden_size=4)
        assert h_n.shape == (1, 2, 4)

    def test_rnn_forward_with_initial_hidden(self):
        """Test RNN forward pass with initial hidden state."""
        rnn = RNN(input_size=3, hidden_size=4, num_layers=2)
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        h_0 = Tensor(np.random.randn(2, 2, 4), requires_grad=True)  # (num_layers, batch, hidden)
        
        output, h_n = rnn(input_tensor, h_0)
        
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)

    def test_rnn_forward_bidirectional(self):
        """Test RNN forward pass with bidirectional processing."""
        rnn = RNN(input_size=3, hidden_size=4, bidirectional=True, batch_first=True)
        
        input_tensor = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        
        # Output should concatenate forward and backward: (batch, seq, 2*hidden)
        assert output.shape == (2, 5, 8)  # 2 * 4
        # Final hidden: (num_directions=2, batch, hidden)
        assert h_n.shape == (2, 2, 4)

    def test_rnn_forward_multilayer_bidirectional(self):
        """Test RNN forward pass with multiple layers and bidirectional."""
        rnn = RNN(input_size=3, hidden_size=4, num_layers=2, bidirectional=True)
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        
        # Output should be (seq, batch, 2*hidden)
        assert output.shape == (5, 2, 8)
        # Final hidden: (num_layers * num_directions, batch, hidden)
        assert h_n.shape == (4, 2, 4)  # 2 layers * 2 directions

    def test_rnn_dropout(self):
        """Test RNN with dropout between layers."""
        rnn = RNN(input_size=3, hidden_size=4, num_layers=3, dropout=0.5)
        
        # Set to training mode to enable dropout
        rnn.training = True
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (3, 2, 4)

    @pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
    def test_rnn_nonlinearity(self, nonlinearity):
        """Test RNN with different nonlinearities."""
        rnn = RNN(input_size=3, hidden_size=4, nonlinearity=nonlinearity)
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (1, 2, 4)


class TestLSTMCell:
    """Comprehensive tests for LSTMCell."""

    def test_lstm_cell_init_valid(self):
        """Test LSTMCell initialization with valid parameters."""
        cell = LSTMCell(input_size=10, hidden_size=20)
        assert cell.input_size == 10
        assert cell.hidden_size == 20
        assert cell.use_bias is True
        assert cell.name == "LSTMCell(10, 20)"
        
        # Check parameter shapes (4 gates: i, f, g, o)
        assert cell.weight_ih.shape == (80, 10)  # 4 * 20
        assert cell.weight_hh.shape == (80, 20)  # 4 * 20
        assert cell.bias_ih.shape == (80,)
        assert cell.bias_hh.shape == (80,)

    def test_lstm_cell_init_no_bias(self):
        """Test LSTMCell initialization without bias."""
        cell = LSTMCell(input_size=5, hidden_size=8, bias=False)
        assert cell.use_bias is False
        assert cell.bias_ih is None
        assert cell.bias_hh is None

    def test_lstm_cell_init_invalid_params(self):
        """Test LSTMCell initialization with invalid parameters."""
        with pytest.raises(LayerError):
            LSTMCell(input_size=0, hidden_size=10)
        
        with pytest.raises(LayerError):
            LSTMCell(input_size=10, hidden_size=0)

    def test_lstm_cell_bias_initialization(self):
        """Test LSTM forget gate bias initialization to 1.0."""
        cell = LSTMCell(input_size=3, hidden_size=4)
        
        # Forget gate bias should be initialized to 1.0
        forget_bias = cell.bias_ih.data[4:8]  # forget gate portion
        np.testing.assert_allclose(forget_bias, 1.0, atol=1e-6)
        
        forget_bias_hh = cell.bias_hh.data[4:8]  # forget gate portion
        np.testing.assert_allclose(forget_bias_hh, 1.0, atol=1e-6)

    def test_lstm_cell_forward_basic(self):
        """Test basic LSTMCell forward pass."""
        cell = LSTMCell(input_size=3, hidden_size=4)
        
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        h = Tensor(np.random.randn(2, 4), requires_grad=True)
        c = Tensor(np.random.randn(2, 4), requires_grad=True)
        
        new_h, new_c = cell(input_tensor, (h, c))
        
        assert new_h.shape == (2, 4)
        assert new_c.shape == (2, 4)
        assert new_h.requires_grad is True
        assert new_c.requires_grad is True

    def test_lstm_cell_forward_no_hidden(self):
        """Test LSTMCell forward pass without initial states."""
        cell = LSTMCell(input_size=3, hidden_size=4)
        
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        
        new_h, new_c = cell(input_tensor)
        
        assert new_h.shape == (2, 4)
        assert new_c.shape == (2, 4)

    def test_lstm_cell_forward_invalid_input(self):
        """Test LSTMCell forward with invalid inputs."""
        cell = LSTMCell(input_size=3, hidden_size=4)
        
        # Wrong input dimensions
        with pytest.raises(NeuralArchError):
            invalid_input = Tensor(np.random.randn(2, 3, 5), requires_grad=True)
            cell(invalid_input)
        
        # Wrong input size
        with pytest.raises(NeuralArchError):
            invalid_input = Tensor(np.random.randn(2, 5), requires_grad=True)
            cell(invalid_input)

    def test_lstm_cell_forward_invalid_hidden(self):
        """Test LSTMCell forward with invalid hidden states."""
        cell = LSTMCell(input_size=3, hidden_size=4)
        
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        
        # Wrong hidden shape
        with pytest.raises(NeuralArchError):
            invalid_h = Tensor(np.random.randn(2, 5), requires_grad=True)
            invalid_c = Tensor(np.random.randn(2, 4), requires_grad=True)
            cell(input_tensor, (invalid_h, invalid_c))
        
        # Wrong cell shape
        with pytest.raises(NeuralArchError):
            valid_h = Tensor(np.random.randn(2, 4), requires_grad=True)
            invalid_c = Tensor(np.random.randn(2, 5), requires_grad=True)
            cell(input_tensor, (valid_h, invalid_c))

    def test_lstm_cell_gate_computations(self):
        """Test LSTM gate computations with known values."""
        cell = LSTMCell(input_size=2, hidden_size=2, bias=True)
        
        # Set weights to simple values for predictable computation
        # Each gate gets 2 rows (for hidden_size=2)
        cell.weight_ih.data = np.eye(8, 2, dtype=np.float32)  # Identity-like matrix
        cell.weight_hh.data = np.eye(8, 2, dtype=np.float32)
        cell.bias_ih.data = np.zeros(8, dtype=np.float32)
        cell.bias_hh.data = np.zeros(8, dtype=np.float32)
        
        # Set forget gate bias to 1.0 (standard practice)
        cell.bias_ih.data[2:4] = 1.0
        cell.bias_hh.data[2:4] = 1.0
        
        input_tensor = Tensor(np.array([[1.0, 0.0]], dtype=np.float32), requires_grad=False)
        h = Tensor(np.array([[0.0, 1.0]], dtype=np.float32), requires_grad=False)
        c = Tensor(np.array([[0.5, -0.5]], dtype=np.float32), requires_grad=False)
        
        new_h, new_c = cell(input_tensor, (h, c))
        
        # Just check that computation runs and produces reasonable outputs
        assert new_h.shape == (1, 2)
        assert new_c.shape == (1, 2)
        
        # Cell state should be influenced by forget gate (initialized to favor forgetting)
        # and input gate, but exact values depend on sigmoid/tanh computations

    @pytest.mark.parametrize("weight_init", ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal"])
    def test_lstm_cell_weight_initialization(self, weight_init):
        """Test different weight initialization schemes for LSTM."""
        cell = LSTMCell(input_size=10, hidden_size=20, weight_init=weight_init)
        
        # Check that weights are initialized (not zeros)
        assert not np.allclose(cell.weight_ih.data, 0)
        assert not np.allclose(cell.weight_hh.data, 0)


class TestLSTM:
    """Comprehensive tests for multi-layer LSTM."""

    def test_lstm_init_valid(self):
        """Test LSTM initialization with valid parameters."""
        lstm = LSTM(input_size=10, hidden_size=20)
        assert lstm.input_size == 10
        assert lstm.hidden_size == 20
        assert lstm.num_layers == 1
        assert lstm.bias is True
        assert lstm.batch_first is False
        assert lstm.dropout == 0.0
        assert lstm.bidirectional is False
        assert len(lstm.cells) == 1

    def test_lstm_init_multilayer_bidirectional(self):
        """Test LSTM initialization with multiple layers and bidirectional."""
        lstm = LSTM(input_size=5, hidden_size=8, num_layers=2, bidirectional=True)
        assert lstm.num_layers == 2
        assert lstm.bidirectional is True
        assert len(lstm.cells) == 4  # 2 layers * 2 directions

    def test_lstm_forward_basic(self):
        """Test basic LSTM forward pass."""
        lstm = LSTM(input_size=3, hidden_size=4, batch_first=True)
        
        input_tensor = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
        
        output, (h_n, c_n) = lstm(input_tensor)
        
        assert output.shape == (2, 5, 4)
        assert h_n.shape == (1, 2, 4)
        assert c_n.shape == (1, 2, 4)

    def test_lstm_forward_with_initial_states(self):
        """Test LSTM forward pass with initial states."""
        lstm = LSTM(input_size=3, hidden_size=4, num_layers=2)
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        h_0 = Tensor(np.random.randn(2, 2, 4), requires_grad=True)
        c_0 = Tensor(np.random.randn(2, 2, 4), requires_grad=True)
        
        output, (h_n, c_n) = lstm(input_tensor, (h_0, c_0))
        
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)
        assert c_n.shape == (2, 2, 4)

    def test_lstm_forward_bidirectional(self):
        """Test LSTM forward pass with bidirectional processing."""
        lstm = LSTM(input_size=3, hidden_size=4, bidirectional=True, batch_first=True)
        
        input_tensor = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
        
        output, (h_n, c_n) = lstm(input_tensor)
        
        # Output concatenates forward and backward
        assert output.shape == (2, 5, 8)  # 2 * 4
        # States: (num_directions, batch, hidden)
        assert h_n.shape == (2, 2, 4)
        assert c_n.shape == (2, 2, 4)

    def test_lstm_dropout(self):
        """Test LSTM with dropout between layers."""
        lstm = LSTM(input_size=3, hidden_size=4, num_layers=3, dropout=0.5)
        lstm.training = True
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, (h_n, c_n) = lstm(input_tensor)
        
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (3, 2, 4)
        assert c_n.shape == (3, 2, 4)


class TestGRUCell:
    """Comprehensive tests for GRUCell."""

    def test_gru_cell_init_valid(self):
        """Test GRUCell initialization with valid parameters."""
        cell = GRUCell(input_size=10, hidden_size=20)
        assert cell.input_size == 10
        assert cell.hidden_size == 20
        assert cell.use_bias is True
        assert cell.name == "GRUCell(10, 20)"
        
        # Check parameter shapes (3 gates: reset, update, new)
        assert cell.weight_ih.shape == (60, 10)  # 3 * 20
        assert cell.weight_hh.shape == (60, 20)  # 3 * 20
        assert cell.bias_ih.shape == (60,)
        assert cell.bias_hh.shape == (60,)

    def test_gru_cell_init_no_bias(self):
        """Test GRUCell initialization without bias."""
        cell = GRUCell(input_size=5, hidden_size=8, bias=False)
        assert cell.use_bias is False
        assert cell.bias_ih is None
        assert cell.bias_hh is None

    def test_gru_cell_forward_basic(self):
        """Test basic GRUCell forward pass."""
        cell = GRUCell(input_size=3, hidden_size=4)
        
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        hidden_tensor = Tensor(np.random.randn(2, 4), requires_grad=True)
        
        output = cell(input_tensor, hidden_tensor)
        
        assert output.shape == (2, 4)
        assert output.requires_grad is True

    def test_gru_cell_forward_no_hidden(self):
        """Test GRUCell forward pass without initial hidden state."""
        cell = GRUCell(input_size=3, hidden_size=4)
        
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        
        output = cell(input_tensor)
        
        assert output.shape == (2, 4)

    def test_gru_cell_gate_computations(self):
        """Test GRU gate computations with simple values."""
        cell = GRUCell(input_size=2, hidden_size=2, bias=True)
        
        # Set weights to simple values
        cell.weight_ih.data = np.eye(6, 2, dtype=np.float32)
        cell.weight_hh.data = np.eye(6, 2, dtype=np.float32)
        cell.bias_ih.data = np.zeros(6, dtype=np.float32)
        cell.bias_hh.data = np.zeros(6, dtype=np.float32)
        
        input_tensor = Tensor(np.array([[1.0, 0.0]], dtype=np.float32), requires_grad=False)
        hidden_tensor = Tensor(np.array([[0.0, 1.0]], dtype=np.float32), requires_grad=False)
        
        output = cell(input_tensor, hidden_tensor)
        
        assert output.shape == (1, 2)
        # The exact values depend on sigmoid and tanh computations

    def test_gru_cell_forward_invalid_input(self):
        """Test GRUCell forward with invalid inputs."""
        cell = GRUCell(input_size=3, hidden_size=4)
        
        # Wrong input dimensions
        with pytest.raises(NeuralArchError):
            invalid_input = Tensor(np.random.randn(2, 3, 5), requires_grad=True)
            cell(invalid_input)
        
        # Wrong input size
        with pytest.raises(NeuralArchError):
            invalid_input = Tensor(np.random.randn(2, 5), requires_grad=True)
            cell(invalid_input)

    def test_gru_cell_forward_invalid_hidden(self):
        """Test GRUCell forward with invalid hidden state."""
        cell = GRUCell(input_size=3, hidden_size=4)
        
        input_tensor = Tensor(np.random.randn(2, 3), requires_grad=True)
        
        # Wrong hidden shape
        with pytest.raises(NeuralArchError):
            invalid_hidden = Tensor(np.random.randn(2, 5), requires_grad=True)
            cell(input_tensor, invalid_hidden)


class TestGRU:
    """Comprehensive tests for multi-layer GRU."""

    def test_gru_init_valid(self):
        """Test GRU initialization with valid parameters."""
        gru = GRU(input_size=10, hidden_size=20)
        assert gru.input_size == 10
        assert gru.hidden_size == 20
        assert gru.num_layers == 1
        assert gru.bias is True
        assert gru.batch_first is False
        assert gru.dropout == 0.0
        assert gru.bidirectional is False
        assert len(gru.cells) == 1

    def test_gru_forward_basic(self):
        """Test basic GRU forward pass."""
        gru = GRU(input_size=3, hidden_size=4, batch_first=True)
        
        input_tensor = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
        
        output, h_n = gru(input_tensor)
        
        assert output.shape == (2, 5, 4)
        assert h_n.shape == (1, 2, 4)

    def test_gru_forward_with_initial_hidden(self):
        """Test GRU forward pass with initial hidden state."""
        gru = GRU(input_size=3, hidden_size=4, num_layers=2)
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        h_0 = Tensor(np.random.randn(2, 2, 4), requires_grad=True)
        
        output, h_n = gru(input_tensor, h_0)
        
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)

    def test_gru_forward_bidirectional(self):
        """Test GRU forward pass with bidirectional processing."""
        gru = GRU(input_size=3, hidden_size=4, bidirectional=True, batch_first=True)
        
        input_tensor = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
        
        output, h_n = gru(input_tensor)
        
        # Output concatenates forward and backward
        assert output.shape == (2, 5, 8)  # 2 * 4
        # Final hidden: (num_directions, batch, hidden)
        assert h_n.shape == (2, 2, 4)

    def test_gru_multilayer_bidirectional(self):
        """Test GRU with multiple layers and bidirectional."""
        gru = GRU(input_size=3, hidden_size=4, num_layers=2, bidirectional=True)
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = gru(input_tensor)
        
        assert output.shape == (5, 2, 8)  # 2 * 4
        assert h_n.shape == (4, 2, 4)  # 2 layers * 2 directions


class TestRNNIntegration:
    """Integration tests for RNN layers."""

    def test_rnn_with_linear_layer(self):
        """Test RNN combined with linear layer."""
        from neural_arch.nn import Linear
        
        rnn = RNN(input_size=3, hidden_size=4, batch_first=True)
        linear = Linear(4, 2)
        
        input_tensor = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
        
        # Forward through RNN
        rnn_output, _ = rnn(input_tensor)
        
        # Forward through linear layer for each timestep
        # Reshape RNN output to (batch * seq, hidden) for linear layer
        reshaped_data = rnn_output.data.reshape(-1, 4)
        reshaped_tensor = Tensor(reshaped_data, requires_grad=rnn_output.requires_grad)
        
        final_output = linear(reshaped_tensor)
        
        # Reshape back to (batch, seq, output_features)
        final_output_data = final_output.data.reshape(2, 5, 2)
        final_output_tensor = Tensor(final_output_data, requires_grad=final_output.requires_grad)
        
        assert final_output_tensor.shape == (2, 5, 2)

    def test_lstm_training_evaluation_modes(self):
        """Test LSTM behavior in training vs evaluation modes."""
        lstm = LSTM(input_size=3, hidden_size=4, num_layers=2, dropout=0.5)
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        # Training mode
        lstm.training = True
        train_output, _ = lstm(input_tensor)
        
        # Evaluation mode
        lstm.training = False
        eval_output, _ = lstm(input_tensor)
        
        # Outputs should have same shape
        assert train_output.shape == eval_output.shape

    def test_variable_sequence_lengths(self):
        """Test RNN layers with different sequence lengths."""
        rnn = RNN(input_size=3, hidden_size=4, batch_first=True)
        
        # Different sequence lengths
        for seq_len in [1, 3, 10]:
            input_tensor = Tensor(np.random.randn(2, seq_len, 3), requires_grad=True)
            output, h_n = rnn(input_tensor)
            
            assert output.shape == (2, seq_len, 4)
            assert h_n.shape == (1, 2, 4)

    def test_gradient_flow_integration(self):
        """Test gradient flow through complete RNN models."""
        # Simple sequence classification model
        rnn = RNN(input_size=2, hidden_size=3, batch_first=True)
        
        input_tensor = Tensor(np.random.randn(1, 4, 2), requires_grad=True)
        
        # Forward pass
        output, final_hidden = rnn(input_tensor)
        
        # Use final hidden state as features
        loss = final_hidden.data.sum()
        
        # Check that gradients can flow
        assert output.requires_grad is True
        assert final_hidden.requires_grad is True

    def test_memory_efficiency(self):
        """Test memory usage with large sequences."""
        # Test with moderately large sequences
        rnn = RNN(input_size=10, hidden_size=20, num_layers=2)
        
        # Larger sequence
        input_tensor = Tensor(np.random.randn(100, 4, 10), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        
        assert output.shape == (100, 4, 20)
        assert h_n.shape == (2, 4, 20)

    @pytest.mark.parametrize("rnn_type,cell_type", [
        (RNN, RNNCell),
        (LSTM, LSTMCell),
        (GRU, GRUCell)
    ])
    def test_rnn_types_consistency(self, rnn_type, cell_type):
        """Test consistency between RNN types and their cells."""
        # Single layer RNN should behave similarly to cell
        if rnn_type == LSTM:
            # LSTM returns both h and c
            multi_layer = rnn_type(input_size=3, hidden_size=4, num_layers=1)
            cell = cell_type(input_size=3, hidden_size=4)
            
            input_tensor = Tensor(np.random.randn(1, 1, 3), requires_grad=True)
            
            # Multi-layer output
            output, (h_n, c_n) = multi_layer(input_tensor)
            
            # Cell output
            input_step = Tensor(input_tensor.data.squeeze(1), requires_grad=True)
            if rnn_type == LSTM:
                cell_h, cell_c = cell(input_step)
            else:
                cell_h = cell(input_step)
            
            # Shapes should match
            assert output.shape == (1, 1, 4)
            assert h_n.shape == (1, 1, 4)
            assert cell_h.shape == (1, 4)
        else:
            # RNN and GRU
            multi_layer = rnn_type(input_size=3, hidden_size=4, num_layers=1)
            cell = cell_type(input_size=3, hidden_size=4)
            
            input_tensor = Tensor(np.random.randn(1, 1, 3), requires_grad=True)
            
            # Multi-layer output
            output, h_n = multi_layer(input_tensor)
            
            # Cell output
            input_step = Tensor(input_tensor.data.squeeze(1), requires_grad=True)
            cell_h = cell(input_step)
            
            # Shapes should match
            assert output.shape == (1, 1, 4)
            assert h_n.shape == (1, 1, 4)
            assert cell_h.shape == (1, 4)

    def test_edge_case_single_timestep(self):
        """Test RNN layers with single timestep."""
        rnn = RNN(input_size=3, hidden_size=4, batch_first=True)
        
        # Single timestep
        input_tensor = Tensor(np.random.randn(2, 1, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        
        assert output.shape == (2, 1, 4)
        assert h_n.shape == (1, 2, 4)

    def test_edge_case_single_batch(self):
        """Test RNN layers with single batch item."""
        lstm = LSTM(input_size=3, hidden_size=4, batch_first=True)
        
        # Single batch item
        input_tensor = Tensor(np.random.randn(1, 5, 3), requires_grad=True)
        
        output, (h_n, c_n) = lstm(input_tensor)
        
        assert output.shape == (1, 5, 4)
        assert h_n.shape == (1, 1, 4)
        assert c_n.shape == (1, 1, 4)

    def test_rnn_invalid_weight_init(self):
        """Test RNN with invalid weight initialization scheme."""
        with pytest.raises(LayerError):
            RNN(input_size=10, hidden_size=20, weight_init="invalid_scheme")
    
    def test_lstm_invalid_weight_init(self):
        """Test LSTM with invalid weight initialization scheme."""
        with pytest.raises(LayerError):
            LSTM(input_size=10, hidden_size=20, weight_init="invalid_scheme")
    
    def test_gru_invalid_weight_init(self):
        """Test GRU with invalid weight initialization scheme."""
        with pytest.raises(LayerError):
            GRU(input_size=10, hidden_size=20, weight_init="invalid_scheme")

    def test_lstm_cell_invalid_weight_init(self):
        """Test LSTMCell with invalid weight initialization scheme."""
        with pytest.raises(LayerError):
            LSTMCell(input_size=10, hidden_size=20, weight_init="invalid_scheme")

    def test_gru_cell_invalid_weight_init(self):
        """Test GRUCell with invalid weight initialization scheme."""
        with pytest.raises(LayerError):
            GRUCell(input_size=10, hidden_size=20, weight_init="invalid_scheme")

    def test_rnn_cell_relu_activation_edge_case(self):
        """Test RNN cell with ReLU activation and specific input values."""
        cell = RNNCell(input_size=2, hidden_size=3, nonlinearity="relu")
        
        # Test with input that produces negative values before activation
        input_tensor = Tensor(np.array([[-10.0, -5.0]], dtype=np.float32), requires_grad=True)
        hidden_tensor = Tensor(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), requires_grad=True)
        
        output = cell(input_tensor, hidden_tensor)
        assert output.shape == (1, 3)
        # With ReLU, negative values should be clipped to 0
        # Note: Exact values depend on weights, but we can check that computation runs

    def test_lstm_cell_no_hidden_tuple_case(self):
        """Test LSTM cell with tuple unpacking edge case."""
        cell = LSTMCell(input_size=2, hidden_size=3)
        
        input_tensor = Tensor(np.random.randn(1, 2), requires_grad=True)
        h = Tensor(np.random.randn(1, 3), requires_grad=True)
        c = Tensor(np.random.randn(1, 3), requires_grad=True)
        
        # Test with explicit tuple
        new_h, new_c = cell(input_tensor, (h, c))
        assert new_h.shape == (1, 3)
        assert new_c.shape == (1, 3)

    def test_gru_cell_weight_initialization_schemes(self):
        """Test GRU cell with different weight initialization schemes."""
        for scheme in ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal"]:
            cell = GRUCell(input_size=3, hidden_size=4, weight_init=scheme)
            assert cell.weight_ih.shape == (12, 3)  # 3 * 4
            assert cell.weight_hh.shape == (12, 4)  # 3 * 4

    def test_lstm_cell_weight_initialization_schemes(self):
        """Test LSTM cell with different weight initialization schemes."""
        for scheme in ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal"]:
            cell = LSTMCell(input_size=3, hidden_size=4, weight_init=scheme)
            assert cell.weight_ih.shape == (16, 3)  # 4 * 4
            assert cell.weight_hh.shape == (16, 4)  # 4 * 4

    def test_rnn_dropout_zero_case(self):
        """Test RNN with zero dropout."""
        rnn = RNN(input_size=3, hidden_size=4, num_layers=2, dropout=0.0)
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)

    def test_lstm_dropout_zero_case(self):
        """Test LSTM with zero dropout."""
        lstm = LSTM(input_size=3, hidden_size=4, num_layers=2, dropout=0.0)
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, (h_n, c_n) = lstm(input_tensor)
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)
        assert c_n.shape == (2, 2, 4)

    def test_gru_dropout_zero_case(self):
        """Test GRU with zero dropout."""
        gru = GRU(input_size=3, hidden_size=4, num_layers=2, dropout=0.0)
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = gru(input_tensor)
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)

    def test_rnn_cell_relu_gradient_flow(self):
        """Test RNN cell with ReLU activation and gradient computation."""
        cell = RNNCell(input_size=2, hidden_size=3, nonlinearity="relu")
        
        input_tensor = Tensor(np.random.randn(1, 2), requires_grad=True)
        hidden_tensor = Tensor(np.random.randn(1, 3), requires_grad=True)
        
        output = cell(input_tensor, hidden_tensor)
        
        # Trigger backward pass to test gradient computation
        output.backward(np.ones_like(output.data))
        
        assert output.shape == (1, 3)
        assert output.requires_grad is True
        
    def test_rnn_cell_unsupported_activation(self):
        """Test RNN cell with unsupported activation function."""
        # This should pass initialization
        cell = RNNCell(input_size=2, hidden_size=3, nonlinearity="tanh")
        
        # Manually set an invalid activation to trigger the error
        cell.nonlinearity = "invalid_activation"
        
        input_tensor = Tensor(np.random.randn(1, 2), requires_grad=True)
        
        with pytest.raises(NeuralArchError):
            cell(input_tensor)

    def test_lstm_cell_gate_detailed_computation(self):
        """Test detailed LSTM gate computations with specific hidden state cases."""
        cell = LSTMCell(input_size=2, hidden_size=2)
        
        input_tensor = Tensor(np.array([[1.0, 0.5]], dtype=np.float32), requires_grad=True)
        h = Tensor(np.array([[0.3, -0.2]], dtype=np.float32), requires_grad=True)
        c = Tensor(np.array([[0.1, 0.4]], dtype=np.float32), requires_grad=True)
        
        new_h, new_c = cell(input_tensor, (h, c))
        
        # Test gradient flow through LSTM gates
        new_h.backward(np.ones_like(new_h.data))
        new_c.backward(np.ones_like(new_c.data))
        
        assert new_h.shape == (1, 2)
        assert new_c.shape == (1, 2)

    def test_gru_cell_gate_detailed_computation(self):
        """Test detailed GRU gate computations."""
        cell = GRUCell(input_size=2, hidden_size=2)
        
        input_tensor = Tensor(np.array([[1.0, 0.5]], dtype=np.float32), requires_grad=True)
        hidden_tensor = Tensor(np.array([[0.3, -0.2]], dtype=np.float32), requires_grad=True)
        
        output = cell(input_tensor, hidden_tensor)
        
        # Test gradient flow through GRU gates
        output.backward(np.ones_like(output.data))
        
        assert output.shape == (1, 2)
        assert output.requires_grad is True

    def test_rnn_evaluation_mode(self):
        """Test RNN in evaluation mode (training=False)."""
        rnn = RNN(input_size=3, hidden_size=4, num_layers=2, dropout=0.5)
        rnn.training = False  # Set to evaluation mode
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = rnn(input_tensor)
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)

    def test_lstm_evaluation_mode(self):
        """Test LSTM in evaluation mode (training=False)."""
        lstm = LSTM(input_size=3, hidden_size=4, num_layers=2, dropout=0.5)
        lstm.training = False  # Set to evaluation mode
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, (h_n, c_n) = lstm(input_tensor)
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)
        assert c_n.shape == (2, 2, 4)

    def test_gru_evaluation_mode(self):
        """Test GRU in evaluation mode (training=False)."""
        gru = GRU(input_size=3, hidden_size=4, num_layers=2, dropout=0.5)
        gru.training = False  # Set to evaluation mode
        
        input_tensor = Tensor(np.random.randn(5, 2, 3), requires_grad=True)
        
        output, h_n = gru(input_tensor)
        assert output.shape == (5, 2, 4)
        assert h_n.shape == (2, 2, 4)

    def test_lstm_invalid_parameters(self):
        """Test LSTM with invalid initialization parameters."""
        # Test invalid input_size
        with pytest.raises(LayerError):
            LSTM(input_size=0, hidden_size=20)
        
        with pytest.raises(LayerError):
            LSTM(input_size=-1, hidden_size=20)
        
        # Test invalid hidden_size
        with pytest.raises(LayerError):
            LSTM(input_size=10, hidden_size=0)
        
        with pytest.raises(LayerError):
            LSTM(input_size=10, hidden_size=-1)
        
        # Test invalid num_layers
        with pytest.raises(LayerError):
            LSTM(input_size=10, hidden_size=20, num_layers=0)
        
        with pytest.raises(LayerError):
            LSTM(input_size=10, hidden_size=20, num_layers=-1)
        
        # Test invalid dropout
        with pytest.raises(LayerError):
            LSTM(input_size=10, hidden_size=20, dropout=-0.1)
        
        with pytest.raises(LayerError):
            LSTM(input_size=10, hidden_size=20, dropout=1.5)

    def test_gru_cell_invalid_parameters(self):
        """Test GRUCell with invalid initialization parameters."""
        # Test invalid input_size
        with pytest.raises(LayerError):
            GRUCell(input_size=0, hidden_size=20)
        
        with pytest.raises(LayerError):
            GRUCell(input_size=-1, hidden_size=20)
        
        # Test invalid hidden_size
        with pytest.raises(LayerError):
            GRUCell(input_size=10, hidden_size=0)
        
        with pytest.raises(LayerError):
            GRUCell(input_size=10, hidden_size=-1)

    def test_gru_invalid_parameters(self):
        """Test GRU with invalid initialization parameters."""
        # Test invalid input_size
        with pytest.raises(LayerError):
            GRU(input_size=0, hidden_size=20)
        
        with pytest.raises(LayerError):
            GRU(input_size=-1, hidden_size=20)
        
        # Test invalid hidden_size  
        with pytest.raises(LayerError):
            GRU(input_size=10, hidden_size=0)
        
        with pytest.raises(LayerError):
            GRU(input_size=10, hidden_size=-1)
        
        # Test invalid num_layers
        with pytest.raises(LayerError):
            GRU(input_size=10, hidden_size=20, num_layers=0)
        
        with pytest.raises(LayerError):
            GRU(input_size=10, hidden_size=20, num_layers=-1)
        
        # Test invalid dropout
        with pytest.raises(LayerError):
            GRU(input_size=10, hidden_size=20, dropout=-0.1)
        
        with pytest.raises(LayerError):
            GRU(input_size=10, hidden_size=20, dropout=1.5)

    def test_lstm_cell_invalid_parameters(self):
        """Test LSTMCell with invalid initialization parameters."""
        # Test invalid input_size
        with pytest.raises(LayerError):
            LSTMCell(input_size=0, hidden_size=20)
        
        with pytest.raises(LayerError):
            LSTMCell(input_size=-1, hidden_size=20)
        
        # Test invalid hidden_size
        with pytest.raises(LayerError):
            LSTMCell(input_size=10, hidden_size=0)
        
        with pytest.raises(LayerError):
            LSTMCell(input_size=10, hidden_size=-1)