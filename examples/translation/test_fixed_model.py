"""Test the fixed model to ensure parameters work correctly."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
from neural_arch.core import Tensor
from neural_arch.optim import Adam
from neural_arch.functional import cross_entropy_loss
from model_fixed import TranslationTransformer

def test_model_parameters():
    """Test that model parameters are accessible correctly."""
    print("🔍 Testing fixed model parameters...")
    
    # Create model
    model = TranslationTransformer(
        src_vocab_size=10,
        tgt_vocab_size=10,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0
    )
    
    # Check parameters
    print("\n1. Checking parameters():")
    params = model.parameters()
    param_list = list(params)
    print(f"Number of parameters: {len(param_list)}")
    
    # Check first few parameters
    for i, param in enumerate(param_list[:5]):
        if hasattr(param, 'data'):
            print(f"  Param {i}: shape={param.data.shape}, type={type(param)}")
        else:
            print(f"  Param {i}: type={type(param)}, value={param}")
    
    return model

def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\n🔍 Testing gradient flow...")
    
    # Create model
    model = TranslationTransformer(
        src_vocab_size=10,
        tgt_vocab_size=10,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0
    )
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Save initial parameter values
    initial_values = []
    for param in model.parameters():
        if hasattr(param, 'data'):
            initial_values.append(param.data.copy())
    
    # Create dummy data
    src = Tensor(np.array([[1, 2, 3, 0, 0]]), requires_grad=False)
    tgt_in = Tensor(np.array([[1, 2, 3]]), requires_grad=False)
    tgt_out = Tensor(np.array([[2, 3, 4]]), requires_grad=False)
    
    # Forward pass
    output = model(src, tgt_in)
    print(f"Output shape: {output.data.shape}")
    
    # Calculate loss
    output_flat = output.data.reshape(-1, output.data.shape[-1])
    target_flat = tgt_out.data.reshape(-1)
    
    output_tensor = Tensor(output_flat, requires_grad=True)
    target_tensor = Tensor(target_flat, requires_grad=False)
    
    loss = cross_entropy_loss(output_tensor, target_tensor)
    print(f"Loss: {loss.data}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Connect gradients
    if hasattr(output, 'grad'):
        output.backward(output_tensor.grad.reshape(output.data.shape))
    
    # Check if parameters have gradients
    grad_count = 0
    for param in model.parameters():
        if hasattr(param, 'grad') and param.grad is not None:
            grad_count += 1
    print(f"Parameters with gradients: {grad_count}/{len(list(model.parameters()))}")
    
    # Update parameters
    optimizer.step()
    
    # Check if parameters changed
    changes = []
    current_params = list(model.parameters())
    for i, (param, initial) in enumerate(zip(current_params, initial_values)):
        if hasattr(param, 'data'):
            change = np.linalg.norm(param.data - initial)
            changes.append(change)
            if change > 1e-8:
                print(f"✅ Param {i} changed by {change:.6f}")
    
    total_change = sum(changes)
    if total_change > 1e-8:
        print(f"\n✅ SUCCESS: Total parameter change: {total_change:.6f}")
    else:
        print("\n❌ FAILURE: No parameters were updated!")
    
    return total_change > 1e-8

def main():
    """Main test function."""
    print("🧪 Testing Fixed Translation Model")
    print("=" * 50)
    
    # Test 1: Parameter access
    model = test_model_parameters()
    
    # Test 2: Gradient flow
    success = test_gradient_flow()
    
    if success:
        print("\n✅ All tests passed! The model is working correctly.")
    else:
        print("\n❌ Tests failed! The model still has issues.")

if __name__ == "__main__":
    main()