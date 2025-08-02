#!/usr/bin/env python3
"""
Simple test for new CNN and RNN layers
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from neural_arch.core import Tensor
    from neural_arch.nn import Conv2d, BatchNorm2d, RNN, LSTM, GRU
    from neural_arch.optimization_config import configure
    
    print("✅ All imports successful")
    
    # Enable optimizations
    configure(
        enable_fusion=True,
        enable_jit=True,
        auto_backend_selection=True
    )
    
    # Test Conv2d
    print("\n🔬 Testing Conv2d...")
    conv = Conv2d(3, 16, kernel_size=3, padding=1)
    test_input = Tensor(np.random.randn(2, 3, 32, 32), requires_grad=True)
    output = conv(test_input)
    print(f"Conv2d input: {test_input.shape}, output: {output.shape}")
    
    # Test BatchNorm2d
    print("\n🔬 Testing BatchNorm2d...")
    bn = BatchNorm2d(16)
    output = bn(output)
    print(f"BatchNorm2d output: {output.shape}")
    
    # Test RNN
    print("\n🔬 Testing RNN...")
    rnn = RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    rnn_input = Tensor(np.random.randn(3, 15, 10), requires_grad=True)
    rnn_output, hidden = rnn(rnn_input)
    print(f"RNN input: {rnn_input.shape}, output: {rnn_output.shape}, hidden: {hidden.shape}")
    
    # Test LSTM  
    print("\n🔬 Testing LSTM...")
    lstm = LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    lstm_output, (h_n, c_n) = lstm(rnn_input)
    print(f"LSTM input: {rnn_input.shape}, output: {lstm_output.shape}, h_n: {h_n.shape}, c_n: {c_n.shape}")
    
    # Test GRU
    print("\n🔬 Testing GRU...")
    gru = GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    gru_output, h_n = gru(rnn_input)
    print(f"GRU input: {rnn_input.shape}, output: {gru_output.shape}, h_n: {h_n.shape}")
    
    print("\n🎉 All basic layer tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)