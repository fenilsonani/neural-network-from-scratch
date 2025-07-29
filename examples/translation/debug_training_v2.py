"""Debug script to diagnose training issues (v2)."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
from neural_arch.core import Tensor
from neural_arch.optim import Adam
from neural_arch.functional import cross_entropy_loss
from vocabulary import Vocabulary
from model import TranslationTransformer

def simple_gradient_test():
    """Test if gradients flow in a simple case."""
    print("\n🔍 Testing gradient flow with simple example...")
    
    # Create simple input/output
    logits = np.random.randn(10, 50) * 0.1
    targets = np.random.randint(0, 50, size=10)
    
    # Create tensors
    logits_tensor = Tensor(logits, requires_grad=True)
    targets_tensor = Tensor(targets, requires_grad=False)
    
    # Calculate loss
    loss = cross_entropy_loss(logits_tensor, targets_tensor)
    print(f"Loss: {loss.data}")
    
    # Backward
    loss.backward()
    
    # Check gradient
    if logits_tensor.grad is not None:
        grad_norm = np.linalg.norm(logits_tensor.grad)
        print(f"✅ Gradient exists! Norm: {grad_norm:.6f}")
    else:
        print("❌ No gradient!")

def test_model_forward():
    """Test model forward pass."""
    print("\n🔍 Testing model forward pass...")
    
    # Create tiny model
    model = TranslationTransformer(
        src_vocab_size=10,
        tgt_vocab_size=10,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0
    )
    
    # Create dummy input
    src = Tensor(np.array([[1, 2, 3, 0, 0]]), requires_grad=False)
    tgt = Tensor(np.array([[1, 2, 3, 4]]), requires_grad=False)
    
    # Forward pass
    try:
        output = model(src, tgt)
        print(f"✅ Forward pass successful! Output shape: {output.data.shape}")
        
        # Check output values
        print(f"Output min: {np.min(output.data):.4f}")
        print(f"Output max: {np.max(output.data):.4f}")
        print(f"Output mean: {np.mean(output.data):.4f}")
        print(f"Output std: {np.std(output.data):.4f}")
        
        # Check if output is reasonable (not all same values)
        unique_values = len(np.unique(output.data.round(decimals=4)))
        print(f"Unique values (rounded to 4 decimals): {unique_values}")
        
        if unique_values < 10:
            print("⚠️  WARNING: Very few unique output values!")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_parameter_access():
    """Test if we can access model parameters correctly."""
    print("\n🔍 Testing parameter access...")
    
    model = TranslationTransformer(
        src_vocab_size=10,
        tgt_vocab_size=10,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0
    )
    
    # Try different ways to access parameters
    print("\nTrying model.parameters()...")
    params = model.parameters()
    param_list = list(params)
    print(f"Number of parameters: {len(param_list)}")
    
    # Check each parameter
    for i, param in enumerate(param_list):
        if hasattr(param, 'data') and hasattr(param, 'requires_grad'):
            print(f"  Param {i}: shape={param.data.shape}, requires_grad={param.requires_grad}")
        else:
            print(f"  Param {i}: type={type(param)}, value={param}")

def test_optimizer_step():
    """Test if optimizer can update parameters."""
    print("\n🔍 Testing optimizer step...")
    
    # Create simple model
    model = TranslationTransformer(
        src_vocab_size=10,
        tgt_vocab_size=10,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0
    )
    
    # Get parameters and save initial values
    params = list(model.parameters())
    initial_values = []
    for param in params:
        if hasattr(param, 'data'):
            initial_values.append(param.data.copy())
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Do forward pass
    src = Tensor(np.array([[1, 2, 3, 0, 0]]), requires_grad=False)
    tgt_in = Tensor(np.array([[1, 2, 3]]), requires_grad=False)
    tgt_out = Tensor(np.array([[2, 3, 4]]), requires_grad=False)
    
    output = model(src, tgt_in)
    
    # Calculate loss
    output_flat = output.data.reshape(-1, output.data.shape[-1])
    target_flat = tgt_out.data.reshape(-1)
    
    output_tensor = Tensor(output_flat, requires_grad=True)
    target_tensor = Tensor(target_flat, requires_grad=False)
    
    loss = cross_entropy_loss(output_tensor, target_tensor)
    print(f"Loss: {loss.data}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check if output tensor has gradient
    if output_tensor.grad is not None:
        print(f"✅ Output tensor has gradient! Shape: {output_tensor.grad.shape}")
    else:
        print("❌ Output tensor has no gradient!")
    
    # Step
    optimizer.step()
    
    # Check if parameters changed
    params_after = list(model.parameters())
    changes = []
    for i, (param, initial) in enumerate(zip(params_after, initial_values)):
        if hasattr(param, 'data'):
            change = np.linalg.norm(param.data - initial)
            changes.append(change)
            if change > 1e-8:
                print(f"✅ Param {i} changed by {change:.6f}")
            else:
                print(f"❌ Param {i} did NOT change")
    
    if sum(changes) > 1e-8:
        print(f"\n✅ Total parameter change: {sum(changes):.6f}")
    else:
        print("\n❌ NO parameters were updated!")

def diagnose_weight_initialization():
    """Check weight initialization patterns."""
    print("\n🔍 Checking weight initialization...")
    
    # Test embedding initialization
    from neural_arch.nn import Embedding
    emb = Embedding(100, 32)
    
    print("\nEmbedding weights:")
    print(f"  Mean: {np.mean(emb.weight.data):.6f}")
    print(f"  Std: {np.std(emb.weight.data):.6f}")
    print(f"  Min: {np.min(emb.weight.data):.6f}")
    print(f"  Max: {np.max(emb.weight.data):.6f}")
    
    # Test linear layer initialization
    from neural_arch.nn import Linear
    linear = Linear(32, 64)
    
    print("\nLinear weights:")
    print(f"  Mean: {np.mean(linear.weight.data):.6f}")
    print(f"  Std: {np.std(linear.weight.data):.6f}")
    print(f"  Min: {np.min(linear.weight.data):.6f}")
    print(f"  Max: {np.max(linear.weight.data):.6f}")

def main():
    """Main debug function."""
    print("🐛 Translation Model Debug Script V2")
    print("=" * 50)
    
    # Run tests
    simple_gradient_test()
    test_model_forward()
    test_parameter_access()
    diagnose_weight_initialization()
    test_optimizer_step()
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main()