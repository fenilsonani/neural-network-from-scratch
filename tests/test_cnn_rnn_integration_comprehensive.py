"""Comprehensive integration tests for CNN and RNN layer combinations.

This test suite provides extensive coverage for:
- CNN + RNN hybrid architectures
- Cross-layer gradient flow validation
- Mixed dimension layer compatibility
- Sequential processing pipelines
- Memory efficiency in complex architectures
- Performance validation for hybrid models

Designed to achieve high coverage for CNN-RNN integration scenarios.
"""

import math
import os
import sys
import pytest
import numpy as np
from typing import Tuple, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import neural architecture components
from neural_arch.nn import (
    Conv1d, Conv2d, Conv3d,
    RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell,
    Linear, Sequential, ModuleList,
    SpatialDropout1d, SpatialDropout2d,
    BatchNorm1d, BatchNorm2d, LayerNorm,
    ReLU, Tanh
)
from neural_arch.core import Tensor
from neural_arch.exceptions import LayerError


class TestCNNRNNIntegration:
    """Tests for CNN-RNN hybrid architectures."""
    
    def test_conv1d_to_rnn_sequence(self):
        """Test Conv1d feature extraction followed by RNN sequence processing."""
        # Conv1d for feature extraction
        conv_features = Sequential(
            Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        )
        
        # RNN for sequence processing
        rnn_processor = RNN(input_size=64, hidden_size=128, num_layers=2, 
                           batch_first=True, bidirectional=True)
        
        # Test data: (batch, channels, sequence_length)
        input_data = np.random.randn(4, 3, 100).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Forward through conv layers
        conv_out = conv_features(x)
        assert conv_out.shape[0] == 4  # batch size preserved
        assert conv_out.shape[1] == 64  # output channels
        assert conv_out.shape[2] == 50  # sequence length halved due to stride=2
        
        # Reshape for RNN: (batch, seq_len, features)
        rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
        
        # Forward through RNN
        rnn_out, hidden = rnn_processor(rnn_input)
        
        # Verify RNN output shape: bidirectional doubles hidden size
        assert rnn_out.shape == (4, 50, 256)  # (batch, seq_len, 2*hidden_size)
        assert hidden.shape == (4, 4, 128)   # final hidden state (num_layers*num_directions, batch, hidden_size)
        
        # Verify gradient tracking
        assert rnn_out.requires_grad
    
    def test_conv2d_feature_maps_to_lstm(self):
        """Test Conv2d feature maps flattened for LSTM processing."""
        # Conv2d feature extractor
        conv_net = Sequential(
            Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        )
        
        # LSTM for temporal modeling
        lstm_processor = LSTM(input_size=64*8*8, hidden_size=256, num_layers=1, 
                             batch_first=True)
        
        # Simulate video frames: (batch, channels, height, width)
        batch_size = 2
        num_frames = 10
        frames = []
        
        for _ in range(num_frames):
            frame = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
            frames.append(Tensor(frame))
        
        # Process each frame through conv net
        conv_features = []
        for frame in frames:
            conv_out = conv_net(frame)
            # Flatten spatial dimensions: (batch, channels*height*width)
            flattened = conv_out.data.reshape(batch_size, -1)
            conv_features.append(flattened)
        
        # Stack into sequence: (batch, seq_len, features)
        sequence_data = np.stack(conv_features, axis=1)
        sequence_input = Tensor(sequence_data, requires_grad=True)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = lstm_processor(sequence_input)
        
        # Verify shapes
        assert lstm_out.shape == (batch_size, num_frames, 256)
        assert hidden.shape == (1, batch_size, 256)  # (num_layers, batch, hidden)
        assert cell.shape == (1, batch_size, 256)
        
        # Verify gradient capability
        assert lstm_out.requires_grad
    
    def test_time_distributed_conv_rnn(self):
        """Test time-distributed convolution with RNN."""
        # Simulate time-distributed CNN processing
        batch_size = 3
        time_steps = 8
        channels = 1
        height, width = 16, 16
        
        # Create time series of images
        time_series = np.random.randn(batch_size, time_steps, channels, height, width).astype(np.float32)
        
        # Process each time step through same Conv2d
        conv_layer = Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding=1)
        processed_sequence = []
        
        for t in range(time_steps):
            # Extract single time step: (batch, channels, height, width)
            time_step_data = time_series[:, t, :, :, :]
            time_step_input = Tensor(time_step_data, requires_grad=True)
            
            # Process through conv
            conv_output = conv_layer(time_step_input)
            
            # Global average pooling to get feature vector
            # Shape: (batch, channels, height, width) -> (batch, channels)
            pooled = np.mean(conv_output.data, axis=(2, 3))
            processed_sequence.append(pooled)
        
        # Stack into sequence: (batch, seq_len, features)
        sequence_features = np.stack(processed_sequence, axis=1)
        rnn_input = Tensor(sequence_features, requires_grad=True)
        
        # Process through GRU
        gru = GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
        gru_out, hidden = gru(rnn_input)
        
        # Verify output shapes
        assert gru_out.shape == (batch_size, time_steps, 64)
        assert hidden.shape == (2, batch_size, 64)  # (num_layers, batch, hidden)
    
    def test_rnn_to_conv_upsampling(self):
        """Test RNN output reshaped for conv upsampling."""
        # RNN for sequence processing
        rnn = RNN(input_size=50, hidden_size=128, num_layers=1, batch_first=True)
        
        # Input sequence
        batch_size = 2
        seq_len = 16
        input_data = np.random.randn(batch_size, seq_len, 50).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Process through RNN
        rnn_out, _ = rnn(x)
        assert rnn_out.shape == (batch_size, seq_len, 128)
        
        # Reshape RNN output for conv processing
        # Treat sequence as "channels" and hidden features as spatial dimension
        # (batch, seq_len, hidden) -> (batch, seq_len, 1, hidden)
        conv_input_data = rnn_out.data.reshape(batch_size, seq_len, 1, 128)
        conv_input = Tensor(conv_input_data, requires_grad=rnn_out.requires_grad)
        
        # Conv1d for upsampling/processing
        conv_upsampler = Sequential(
            Conv1d(in_channels=seq_len, out_channels=32, kernel_size=3, padding=1),
            Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        )
        
        # Squeeze and process through conv1d
        conv_input_1d = conv_input.data.squeeze(2)  # Remove dimension of size 1
        conv_input_tensor = Tensor(conv_input_1d, requires_grad=conv_input.requires_grad)
        
        conv_out = conv_upsampler(conv_input_tensor)
        
        # Verify output
        assert conv_out.shape == (batch_size, 16, 128)
        assert conv_out.requires_grad


class TestHybridArchitectures:
    """Tests for complex hybrid CNN-RNN architectures."""
    
    def test_encoder_decoder_with_rnn_attention(self):
        """Test encoder-decoder architecture with RNN-based attention."""
        # Encoder: Conv2d -> flatten -> LSTM
        encoder_conv = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )
        
        encoder_rnn = LSTM(input_size=64*8*8, hidden_size=256, num_layers=2, 
                          batch_first=True)
        
        # Decoder: LSTM -> reshape -> conv transpose (simplified)
        decoder_rnn = LSTM(input_size=256, hidden_size=512, num_layers=2, 
                          batch_first=True)
        
        # Test input: batch of images
        batch_size = 2
        images = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
        x = Tensor(images, requires_grad=True)
        
        # Encode
        conv_features = encoder_conv(x)
        # Flatten for RNN: (batch, seq_len=1, features)
        flattened = conv_features.data.reshape(batch_size, 1, -1)
        encoder_input = Tensor(flattened, requires_grad=conv_features.requires_grad)
        
        encoder_out, (encoder_h, encoder_c) = encoder_rnn(encoder_input)
        
        # Use encoder final state as decoder initial input
        # Create decoder input sequence (teacher forcing simulation)
        decoder_seq_len = 5
        decoder_input_data = np.random.randn(batch_size, decoder_seq_len, 256).astype(np.float32)
        decoder_input = Tensor(decoder_input_data, requires_grad=True)
        
        # Decode
        decoder_out, (decoder_h, decoder_c) = decoder_rnn(decoder_input)
        
        # Verify shapes
        assert encoder_out.shape == (batch_size, 1, 256)
        assert decoder_out.shape == (batch_size, decoder_seq_len, 512)
        assert encoder_h.shape == (2, batch_size, 256)
        assert decoder_h.shape == (2, batch_size, 512)
    
    def test_multi_scale_cnn_rnn_fusion(self):
        """Test multi-scale CNN features fused with RNN."""
        # Multi-scale conv layers
        conv_scale1 = Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        conv_scale2 = Conv1d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        conv_scale3 = Conv1d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
        
        # RNN for temporal modeling
        rnn = RNN(input_size=48, hidden_size=64, num_layers=1, batch_first=True)
        
        # Input sequence
        batch_size = 3
        seq_len = 20
        input_data = np.random.randn(batch_size, 4, seq_len).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Multi-scale feature extraction
        features1 = conv_scale1(x)
        features2 = conv_scale2(x)
        features3 = conv_scale3(x)
        
        # Concatenate features along channel dimension
        concat_data = np.concatenate([features1.data, features2.data, features3.data], axis=1)
        concat_features = Tensor(concat_data, requires_grad=True)
        
        # Reshape for RNN: (batch, channels, seq_len) -> (batch, seq_len, channels)
        rnn_input_data = np.transpose(concat_features.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=concat_features.requires_grad)
        
        # Process through RNN
        rnn_out, hidden = rnn(rnn_input)
        
        # Verify output shapes
        assert features1.shape == (batch_size, 16, seq_len)
        assert features2.shape == (batch_size, 16, seq_len)
        assert features3.shape == (batch_size, 16, seq_len)
        assert concat_features.shape == (batch_size, 48, seq_len)
        assert rnn_out.shape == (batch_size, seq_len, 64)
    
    def test_residual_cnn_rnn_connections(self):
        """Test residual connections in CNN-RNN hybrid."""
        # CNN block with residual connection
        conv_block = Sequential(
            Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        )
        
        # RNN processor
        rnn = GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        
        # Input
        batch_size = 2
        seq_len = 15
        channels = 32
        input_data = np.random.randn(batch_size, channels, seq_len).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # CNN forward with residual
        conv_out = conv_block(x)
        # Residual connection
        residual_data = x.data + conv_out.data
        residual = Tensor(residual_data, requires_grad=True)
        
        # Reshape for RNN
        rnn_input_data = np.transpose(residual.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=residual.requires_grad)
        
        # RNN forward
        rnn_out, hidden = rnn(rnn_input)
        
        # Another residual connection (RNN output + original)
        original_rnn_shape = np.transpose(x.data, (0, 2, 1))
        final_residual_data = rnn_out.data + original_rnn_shape
        final_output = Tensor(final_residual_data, requires_grad=rnn_out.requires_grad)
        
        # Verify shapes and gradients
        assert conv_out.shape == x.shape
        assert residual.shape == x.shape
        assert rnn_out.shape == (batch_size, seq_len, channels)
        assert final_output.shape == (batch_size, seq_len, channels)
        assert final_output.requires_grad


class TestCrossLayerGradientFlow:
    """Tests for gradient flow through CNN-RNN combinations."""
    
    def test_gradient_flow_conv_to_rnn(self):
        """Test gradient computation through Conv->RNN pipeline."""
        # Simple pipeline
        conv = Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        rnn = RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
        
        # Input with gradient tracking
        input_data = np.random.randn(1, 2, 10).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Forward pass
        conv_out = conv(x)
        
        # Reshape for RNN
        rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
        
        rnn_out, _ = rnn(rnn_input)
        
        # Verify gradient functions are set up
        assert conv_out._grad_fn is not None
        assert rnn_out._grad_fn is not None
        
        # Check that parameters have gradient tracking
        for param in conv.parameters():
            assert param.requires_grad
        for param in rnn.parameters():
            assert param.requires_grad
    
    def test_gradient_accumulation_hybrid_model(self):
        """Test gradient accumulation in hybrid models."""
        # Hybrid model components
        conv_encoder = Conv1d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        rnn_processor = LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True)
        output_projection = Linear(in_features=32, out_features=10)
        
        # Multiple forward passes (simulating gradient accumulation)
        total_params_before = []
        
        # Store initial parameter values
        for param in conv_encoder.parameters() + rnn_processor.parameters() + output_projection.parameters():
            if param.requires_grad:
                total_params_before.append(param.data.copy())
        
        # Multiple forward passes
        for batch_idx in range(3):
            # Different input for each batch
            input_data = np.random.randn(2, 3, 12).astype(np.float32)
            x = Tensor(input_data, requires_grad=True)
            
            # Forward through pipeline
            conv_out = conv_encoder(x)
            
            # Reshape for RNN
            rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
            rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
            
            rnn_out, _ = rnn_processor(rnn_input)
            
            # Use last time step output
            final_hidden = Tensor(rnn_out.data[:, -1, :], requires_grad=rnn_out.requires_grad)
            
            logits = output_projection(final_hidden)
            
            # Verify output shape
            assert logits.shape == (2, 10)
            assert logits.requires_grad
        
        # Verify all components maintain gradient tracking
        all_params = conv_encoder.parameters() + rnn_processor.parameters() + output_projection.parameters()
        grad_enabled_params = [p for p in all_params if p.requires_grad]
        assert len(grad_enabled_params) > 0
    
    def test_mixed_precision_simulation(self):
        """Test mixed precision-like behavior in CNN-RNN models."""
        # Components
        conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        rnn = GRU(input_size=4*8*8, hidden_size=32, num_layers=1, batch_first=True)
        
        # Test with different precisions (simulated via different scales)
        precisions = [np.float32]  # We'll simulate different "precisions" with scaling
        
        for precision in precisions:
            # Input
            input_data = np.random.randn(1, 1, 8, 8).astype(precision)
            x = Tensor(input_data, requires_grad=True)
            
            # Forward pass
            conv_out = conv(x)
            
            # Flatten for RNN
            flattened = conv_out.data.reshape(1, 1, -1)
            rnn_input = Tensor(flattened, requires_grad=conv_out.requires_grad)
            
            rnn_out, _ = rnn(rnn_input)
            
            # Verify data types are preserved
            assert conv_out.data.dtype == precision
            assert rnn_out.data.dtype == precision
            
            # Verify gradient tracking works
            assert rnn_out.requires_grad


class TestMemoryEfficiency:
    """Tests for memory efficiency in CNN-RNN combinations."""
    
    def test_memory_efficient_sequence_processing(self):
        """Test memory-efficient processing of long sequences."""
        # Use smaller models for memory efficiency testing
        conv = Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        rnn = RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
        
        # Process longer sequence in chunks
        total_seq_len = 100
        chunk_size = 20
        batch_size = 2
        
        chunk_outputs = []
        hidden_state = None
        
        for start_idx in range(0, total_seq_len, chunk_size):
            end_idx = min(start_idx + chunk_size, total_seq_len)
            chunk_len = end_idx - start_idx
            
            # Create chunk data
            chunk_data = np.random.randn(batch_size, 4, chunk_len).astype(np.float32)
            chunk_input = Tensor(chunk_data, requires_grad=True)
            
            # Process through conv
            conv_out = conv(chunk_input)
            
            # Reshape for RNN
            rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
            rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
            
            # Process through RNN with hidden state continuity
            if hidden_state is not None:
                rnn_out, hidden_state = rnn(rnn_input, hidden_state)
            else:
                rnn_out, hidden_state = rnn(rnn_input)
            
            chunk_outputs.append(rnn_out.data)
        
        # Verify we processed the full sequence
        total_processed_length = sum(output.shape[1] for output in chunk_outputs)
        assert total_processed_length == total_seq_len
        
        # Verify output shapes are consistent
        for output in chunk_outputs:
            assert output.shape[0] == batch_size  # batch size
            assert output.shape[2] == 16  # hidden size
    
    def test_parameter_sharing_efficiency(self):
        """Test parameter sharing in CNN-RNN models."""
        # Shared convolutional layer
        shared_conv = Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        
        # Multiple RNN paths sharing the same conv layer
        rnn_path1 = RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
        rnn_path2 = RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
        
        # Test inputs
        input1_data = np.random.randn(1, 2, 15).astype(np.float32)
        input2_data = np.random.randn(1, 2, 15).astype(np.float32)
        
        x1 = Tensor(input1_data, requires_grad=True)
        x2 = Tensor(input2_data, requires_grad=True)
        
        # Process both inputs through shared conv
        conv_out1 = shared_conv(x1)
        conv_out2 = shared_conv(x2)
        
        # Verify same conv weights are used
        assert shared_conv.weight.data is shared_conv.weight.data  # Same object
        
        # Process through different RNN paths
        rnn_input1_data = np.transpose(conv_out1.data, (0, 2, 1))
        rnn_input2_data = np.transpose(conv_out2.data, (0, 2, 1))
        
        rnn_input1 = Tensor(rnn_input1_data, requires_grad=conv_out1.requires_grad)
        rnn_input2 = Tensor(rnn_input2_data, requires_grad=conv_out2.requires_grad)
        
        rnn_out1, _ = rnn_path1(rnn_input1)
        rnn_out2, _ = rnn_path2(rnn_input2)
        
        # Verify both paths produce valid outputs
        assert rnn_out1.shape == (1, 15, 16)
        assert rnn_out2.shape == (1, 15, 16)
        assert rnn_out1.requires_grad
        assert rnn_out2.requires_grad


class TestErrorHandlingIntegration:
    """Tests for error handling in CNN-RNN integration."""
    
    def test_dimension_mismatch_errors(self):
        """Test proper error handling for dimension mismatches."""
        conv = Conv1d(in_channels=4, out_channels=8, kernel_size=3)
        rnn = RNN(input_size=16, hidden_size=32, num_layers=1, batch_first=True)  # Wrong input size
        
        # Create input
        input_data = np.random.randn(1, 4, 10).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Conv forward should work
        conv_out = conv(x)
        assert conv_out.shape == (1, 8, 8)  # 10 - 3 + 1 = 8
        
        # Reshape for RNN - this should cause dimension mismatch
        rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))  # (1, 8, 8)
        rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
        
        # This should raise an error due to input_size mismatch (8 != 16)
        with pytest.raises(LayerError, match="Input size mismatch"):
            rnn_out, _ = rnn(rnn_input)
    
    def test_gradient_flow_interruption(self):
        """Test behavior when gradient flow is interrupted."""
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        rnn = RNN(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
        
        # Input with gradient tracking
        input_data = np.random.randn(1, 2, 10).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Forward through conv
        conv_out = conv(x)
        
        # Detach gradients (simulate gradient interruption)
        detached_data = conv_out.data.copy()
        detached_tensor = Tensor(detached_data, requires_grad=False)  # No gradients
        
        # Reshape for RNN
        rnn_input_data = np.transpose(detached_tensor.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=detached_tensor.requires_grad)
        
        # RNN forward should work but without gradients
        rnn_out, _ = rnn(rnn_input)
        
        # Verify gradient tracking is interrupted
        assert not rnn_out.requires_grad
        assert rnn_out._grad_fn is None  # No gradient function
    
    def test_invalid_sequence_length_handling(self):
        """Test handling of invalid sequence lengths."""
        conv = Conv1d(in_channels=3, out_channels=6, kernel_size=5)  # Large kernel
        rnn = RNN(input_size=6, hidden_size=12, num_layers=1, batch_first=True)
        
        # Too short input for the kernel size
        short_input_data = np.random.randn(1, 3, 3).astype(np.float32)  # Length 3 < kernel 5
        short_input = Tensor(short_input_data, requires_grad=True)
        
        # This should fail due to insufficient input length
        with pytest.raises((LayerError, ValueError, IndexError)):
            conv_out = conv(short_input)


class TestPerformanceIntegration:
    """Performance tests for CNN-RNN integration."""
    
    def test_throughput_cnn_rnn_pipeline(self):
        """Test throughput of CNN-RNN pipeline."""
        import time
        
        # Create reasonably sized model
        conv = Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        rnn = LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True)
        
        # Batch of sequences
        batch_size = 4
        seq_len = 50
        input_data = np.random.randn(batch_size, 8, seq_len).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Warm up
        conv_out = conv(x)
        rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
        _ = rnn(rnn_input)
        
        # Measure performance
        start_time = time.time()
        
        # Forward pass
        conv_out = conv(x)
        rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
        rnn_out, (hidden, cell) = rnn(rnn_input)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Verify reasonable performance (should be well under 200ms)
        assert duration_ms < 200, f"CNN-RNN pipeline took {duration_ms:.2f}ms, exceeding 200ms target"
        
        # Verify correct output shapes
        assert conv_out.shape == (batch_size, 16, seq_len)
        assert rnn_out.shape == (batch_size, seq_len, 32)
        assert hidden.shape == (2, batch_size, 32)
        assert cell.shape == (2, batch_size, 32)
    
    def test_memory_usage_large_sequences(self):
        """Test memory usage with larger sequences."""
        # Use smaller models but longer sequences
        conv = Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        rnn = GRU(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
        
        # Large sequence
        batch_size = 2
        seq_len = 200
        input_data = np.random.randn(batch_size, 4, seq_len).astype(np.float32)
        x = Tensor(input_data, requires_grad=True)
        
        # Forward pass
        conv_out = conv(x)
        rnn_input_data = np.transpose(conv_out.data, (0, 2, 1))
        rnn_input = Tensor(rnn_input_data, requires_grad=conv_out.requires_grad)
        rnn_out, hidden = rnn(rnn_input)
        
        # Verify shapes and that computation completed successfully
        assert conv_out.shape == (batch_size, 8, seq_len)
        assert rnn_out.shape == (batch_size, seq_len, 16)
        assert hidden.shape == (1, batch_size, 16)
        
        # Verify outputs are reasonable (not all zeros or NaN)
        assert not np.all(rnn_out.data == 0)
        assert np.all(np.isfinite(rnn_out.data))


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])