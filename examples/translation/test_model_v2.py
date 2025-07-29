"""Test model v2 with fixed parameters."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
from neural_arch.core import Tensor, Parameter
from neural_arch.optim import Adam
from neural_arch.functional import cross_entropy_loss
from model_v2 import TranslationTransformer

def test_parameters_and_training():
    """Test that parameters work and training updates them."""
    print("🧪 Testing Model V2")
    print("=" * 50)
    
    # Create small model
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
    print("\n1. Checking parameters:")
    param_list = list(model.parameters())
    print(f"Total parameters: {len(param_list)}")
    
    # Verify all are Parameter objects
    all_params = True
    for i, param in enumerate(param_list[:5]):
        if isinstance(param, Parameter):
            print(f"  Param {i}: ✅ Parameter object, shape={param.data.shape}")
        else:
            print(f"  Param {i}: ❌ Not a Parameter! type={type(param)}")
            all_params = False
    
    if not all_params:
        print("\n❌ Some parameters are not Parameter objects!")
        return False
    
    # Create optimizer
    print("\n2. Creating optimizer...")
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Save initial values
    initial_values = [p.data.copy() for p in model.parameters()]
    
    # Do a training step
    print("\n3. Performing training step...")
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
    print(f"Loss: {loss.data:.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Connect gradients back to model
    if hasattr(output, 'backward'):
        grad_reshaped = output_tensor.grad.reshape(output.data.shape)
        output.backward(grad_reshaped)
    
    # Update
    optimizer.step()
    
    # Check if parameters changed
    print("\n4. Checking parameter updates:")
    changed_count = 0
    total_change = 0.0
    
    current_params = list(model.parameters())
    for i, (param, initial) in enumerate(zip(current_params, initial_values)):
        change = np.linalg.norm(param.data - initial)
        total_change += change
        if change > 1e-8:
            changed_count += 1
            if i < 5:  # Show first 5
                print(f"  Param {i}: changed by {change:.6f}")
    
    print(f"\nParameters changed: {changed_count}/{len(param_list)}")
    print(f"Total change: {total_change:.6f}")
    
    if changed_count > 0:
        print("\n✅ SUCCESS: Model parameters are updating correctly!")
        return True
    else:
        print("\n❌ FAILURE: No parameters were updated!")
        return False

def main():
    """Main test function."""
    success = test_parameters_and_training()
    
    if success:
        print("\n🎉 Model V2 is working correctly! Ready for training.")
    else:
        print("\n😞 Model V2 still has issues.")

if __name__ == "__main__":
    main()