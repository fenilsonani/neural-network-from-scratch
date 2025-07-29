"""Debug script to diagnose training issues."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
from neural_arch.core import Tensor
from neural_arch.optim import Adam
from neural_arch.functional import cross_entropy_loss
from vocabulary import Vocabulary
from model import TranslationTransformer

def check_parameter_updates(model, optimizer, src_data, tgt_data, pad_idx):
    """Check if parameters are actually being updated."""
    print("\n🔍 Checking parameter updates...")
    
    # Get initial parameter values
    initial_params = {}
    params = model.parameters()
    for i, param in enumerate(params):
        if param.requires_grad:
            initial_params[i] = param.data.copy()
    
    # Do one forward/backward pass
    src_tensor = Tensor(np.array([src_data[0]]), requires_grad=False)
    tgt_input = Tensor(np.array([tgt_data[0][:-1]]), requires_grad=False)
    tgt_output = Tensor(np.array([tgt_data[0][1:]]), requires_grad=False)
    
    # Forward pass
    output = model(src_tensor, tgt_input)
    
    # Calculate loss
    output_reshaped = output.data.reshape(-1, output.data.shape[-1])
    target_reshaped = tgt_output.data.reshape(-1)
    
    # Create mask for non-padding tokens
    mask = (target_reshaped != pad_idx)
    if np.sum(mask) > 0:
        output_tensor = Tensor(output_reshaped[mask], requires_grad=True)
        target_tensor = Tensor(target_reshaped[mask], requires_grad=False)
        
        loss = cross_entropy_loss(output_tensor, target_tensor)
        print(f"Loss value: {loss.data}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        print("\n📊 Gradient magnitudes:")
        params = list(model.parameters())
        for i, param in enumerate(params):
            if param.requires_grad and param.grad is not None:
                grad_magnitude = np.linalg.norm(param.grad)
                print(f"  Param {i}: {grad_magnitude:.6f}")
        
        # Update parameters
        optimizer.step()
        
        # Check if parameters changed
        print("\n🔄 Parameter changes:")
        any_changed = False
        params = list(model.parameters())
        for i, param in enumerate(params):
            if param.requires_grad and i in initial_params:
                change = np.linalg.norm(param.data - initial_params[i])
                if change > 1e-8:
                    any_changed = True
                    print(f"  Param {i}: changed by {change:.6f}")
                else:
                    print(f"  Param {i}: NO CHANGE ❌")
        
        if not any_changed:
            print("\n⚠️  WARNING: No parameters were updated!")
    else:
        print("❌ No valid tokens after masking!")

def check_model_initialization(model):
    """Check if model is initialized properly."""
    print("\n🔍 Checking model initialization...")
    
    params = list(model.parameters())
    for i, param in enumerate(params):
        if param.requires_grad:
            mean = np.mean(param.data)
            std = np.std(param.data)
            min_val = np.min(param.data)
            max_val = np.max(param.data)
            print(f"\nParam {i}:")
            print(f"  Shape: {param.data.shape}")
            print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
            print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
            
            # Check for extreme values
            if std < 1e-6:
                print(f"  ⚠️  WARNING: Very low variance!")
            if abs(mean) > 1.0:
                print(f"  ⚠️  WARNING: Large mean value!")
            if max_val > 5.0 or min_val < -5.0:
                print(f"  ⚠️  WARNING: Extreme values detected!")

def check_loss_calculation():
    """Test loss calculation with known values."""
    print("\n🔍 Testing loss calculation...")
    
    # Create simple test case
    vocab_size = 10
    batch_size = 2
    seq_len = 3
    
    # Create logits with one clear winner per position
    logits = np.random.randn(batch_size * seq_len, vocab_size) * 0.1
    for i in range(batch_size * seq_len):
        logits[i, i % vocab_size] = 5.0  # Make one class very likely
    
    # Create targets
    targets = np.arange(batch_size * seq_len) % vocab_size
    
    # Calculate loss
    logits_tensor = Tensor(logits, requires_grad=True)
    targets_tensor = Tensor(targets, requires_grad=False)
    
    loss = cross_entropy_loss(logits_tensor, targets_tensor)
    print(f"Test loss: {loss.data:.4f}")
    print(f"Expected loss (approx): {-np.log(np.exp(5.0) / (np.exp(5.0) + 9 * np.exp(0.1))):.4f}")
    
    # Check gradient
    loss.backward()
    print(f"Gradient shape: {logits_tensor.grad.shape}")
    print(f"Gradient magnitude: {np.linalg.norm(logits_tensor.grad):.4f}")

def check_model_output_variation(model, src_vocab, tgt_vocab):
    """Check if model produces different outputs for different inputs."""
    print("\n🔍 Checking output variation...")
    
    test_sentences = ["hello", "goodbye", "how are you"]
    outputs = []
    
    for sentence in test_sentences:
        src_indices = src_vocab.encode(sentence, max_length=10)
        src_tensor = Tensor(np.array([src_indices]), requires_grad=False)
        
        # Get model output for first position
        tgt_start = np.array([[tgt_vocab.word2idx[tgt_vocab.sos_token]]])
        tgt_tensor = Tensor(tgt_start, requires_grad=False)
        
        output = model(src_tensor, tgt_tensor)
        output_probs = np.exp(output.data[0, 0]) / np.sum(np.exp(output.data[0, 0]))
        outputs.append(output_probs)
        
        # Show top 5 predictions
        top_5_idx = np.argsort(output_probs)[-5:][::-1]
        print(f"\n'{sentence}' top predictions:")
        for idx in top_5_idx:
            word = tgt_vocab.idx2word.get(idx, '<UNK>')
            print(f"  {word}: {output_probs[idx]:.4f}")
    
    # Check if outputs are different
    all_same = True
    for i in range(1, len(outputs)):
        if not np.allclose(outputs[0], outputs[i], atol=1e-4):
            all_same = False
            break
    
    if all_same:
        print("\n⚠️  WARNING: Model produces same output for different inputs!")
    else:
        print("\n✅ Model produces different outputs for different inputs")

def main():
    """Main debug function."""
    print("🐛 Translation Model Debug Script")
    print("=" * 50)
    
    # Create small vocabularies
    src_vocab = Vocabulary("english")
    tgt_vocab = Vocabulary("spanish")
    
    # Add some test data
    test_pairs = [
        ("hello", "hola"),
        ("goodbye", "adiós"),
        ("thank you", "gracias"),
        ("yes", "sí"),
        ("no", "no")
    ]
    
    # Build vocabularies
    for en, es in test_pairs:
        src_vocab.add_sentence(en)
        tgt_vocab.add_sentence(es)
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    # Create small model
    model = TranslationTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=64,
        n_heads=2,
        n_layers=1,
        d_ff=128,
        dropout=0.0
    )
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Prepare data
    src_data = [src_vocab.encode(en, max_length=10) for en, _ in test_pairs]
    tgt_data = [tgt_vocab.encode(es, max_length=10) for _, es in test_pairs]
    
    # Run checks
    check_loss_calculation()
    check_model_initialization(model)
    check_model_output_variation(model, src_vocab, tgt_vocab)
    check_parameter_updates(model, optimizer, src_data, tgt_data, 
                          src_vocab.word2idx[src_vocab.pad_token])
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main()