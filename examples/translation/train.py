"""Training script for English-Spanish translation model."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import time
import json
from typing import List, Tuple, Dict
from neural_arch.core import Tensor
from neural_arch.optim import Adam
from neural_arch.functional import cross_entropy_loss
from vocabulary import Vocabulary, create_dataset
from model import TranslationTransformer


def load_dataset(filename: str) -> List[Tuple[str, str]]:
    """Load dataset from JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['pairs']


def create_batch(data: List[List[int]], batch_size: int, pad_idx: int = 0) -> np.ndarray:
    """Create a padded batch from sequences."""
    # Get max length in batch
    max_len = max(len(seq) for seq in data[:batch_size])
    
    # Create padded batch
    batch = np.full((batch_size, max_len), pad_idx, dtype=np.int32)
    for i, seq in enumerate(data[:batch_size]):
        batch[i, :len(seq)] = seq
    
    return batch


def train_epoch(model: TranslationTransformer,
                optimizer: Adam,
                src_data: List[List[int]],
                tgt_data: List[List[int]],
                batch_size: int = 32,
                pad_idx: int = 0) -> float:
    """Train one epoch."""
    total_loss = 0.0
    n_batches = 0
    
    # Shuffle data
    indices = np.random.permutation(len(src_data))
    
    for i in range(0, len(src_data), batch_size):
        # Get batch indices
        batch_indices = indices[i:i + batch_size]
        
        # Create batches
        src_batch = create_batch([src_data[j] for j in batch_indices], len(batch_indices), pad_idx)
        tgt_batch = create_batch([tgt_data[j] for j in batch_indices], len(batch_indices), pad_idx)
        
        # Create tensors
        src_tensor = Tensor(src_batch, requires_grad=False)
        tgt_input = Tensor(tgt_batch[:, :-1], requires_grad=False)  # All but last
        tgt_output = Tensor(tgt_batch[:, 1:], requires_grad=False)  # All but first
        
        # Create masks
        src_mask = model.create_padding_mask(src_batch, pad_idx)
        tgt_mask = model.create_look_ahead_mask(tgt_input.data.shape[1])
        
        # Forward pass
        output = model(src_tensor, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Reshape for loss calculation
        output_reshaped = output.data.reshape(-1, output.data.shape[-1])
        target_reshaped = tgt_output.data.reshape(-1)
        
        # Calculate loss (ignore padding)
        mask = (target_reshaped != pad_idx)
        if np.sum(mask) > 0:
            output_tensor = Tensor(output_reshaped[mask], requires_grad=True)
            target_tensor = Tensor(target_reshaped[mask], requires_grad=False)
            
            loss = cross_entropy_loss(output_tensor, target_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(model: TranslationTransformer,
             src_data: List[List[int]],
             tgt_data: List[List[int]],
             src_vocab: Vocabulary,
             tgt_vocab: Vocabulary,
             n_examples: int = 5):
    """Evaluate model with examples."""
    print("\nTranslation Examples:")
    print("-" * 50)
    
    # Random indices
    indices = np.random.choice(len(src_data), min(n_examples, len(src_data)), replace=False)
    
    for idx in indices:
        # Get source
        src = Tensor(np.array([src_data[idx]]), requires_grad=False)
        
        # Generate translation
        output_indices = model.generate(
            src, 
            max_length=50,
            sos_idx=tgt_vocab.word2idx[tgt_vocab.sos_token],
            eos_idx=tgt_vocab.word2idx[tgt_vocab.eos_token]
        )
        
        # Decode
        src_text = src_vocab.decode(src_data[idx], remove_special=True)
        tgt_text = tgt_vocab.decode(tgt_data[idx], remove_special=True)
        pred_text = tgt_vocab.decode(output_indices, remove_special=True)
        
        print(f"Source (EN): {src_text}")
        print(f"Target (ES): {tgt_text}")
        print(f"Predicted:   {pred_text}")
        print("-" * 50)


def main():
    """Main training function."""
    print("🌐 Training English-Spanish Translation Model")
    print("=" * 50)
    
    # Create vocabularies
    src_vocab = Vocabulary("english")
    tgt_vocab = Vocabulary("spanish")
    
    # Load datasets
    print("\n📚 Loading datasets...")
    try:
        train_pairs = load_dataset("data/train.json")
        val_pairs = load_dataset("data/val.json")
        print(f"✅ Loaded {len(train_pairs)} training pairs")
        print(f"✅ Loaded {len(val_pairs)} validation pairs")
    except FileNotFoundError:
        print("❌ Dataset files not found. Please run create_dataset.py first!")
        return
    
    # Create dataset
    print(f"\n📚 Processing dataset...")
    src_data, tgt_data = create_dataset(train_pairs, src_vocab, tgt_vocab, max_length=30)
    val_src_data, val_tgt_data = create_dataset(val_pairs, src_vocab, tgt_vocab, max_length=30)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Model hyperparameters
    d_model = 128  # Smaller for demo
    n_heads = 4
    n_layers = 2
    d_ff = 256
    
    # Create model
    print(f"\n🤖 Creating Transformer model...")
    print(f"  - d_model: {d_model}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - d_ff: {d_ff}")
    
    model = TranslationTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training
    n_epochs = 50
    batch_size = 16
    
    print(f"\n🏃 Starting training for {n_epochs} epochs...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {optimizer.lr}")
    print(f"  - Training samples: {len(src_data)}")
    print(f"  - Validation samples: {len(val_src_data)}")
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, optimizer, src_data, tgt_data, 
            batch_size=batch_size,
            pad_idx=src_vocab.word2idx[src_vocab.pad_token]
        )
        
        # Validate
        val_loss = train_epoch(
            model, optimizer, val_src_data, val_tgt_data, 
            batch_size=batch_size,
            pad_idx=src_vocab.word2idx[src_vocab.pad_token]
        )
        
        elapsed = time.time() - start_time
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  📈 New best validation loss: {val_loss:.4f}")
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:3d}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s")
            
            # Evaluate
            if (epoch + 1) % 10 == 0:
                print("\n🔍 Validation Examples:")
                evaluate(model, val_src_data, val_tgt_data, src_vocab, tgt_vocab, n_examples=3)
    
    print("\n✅ Training complete!")
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    evaluate(model, src_data, tgt_data, src_vocab, tgt_vocab, n_examples=5)
    
    # Save vocabularies
    src_vocab.save("vocab_en.json")
    tgt_vocab.save("vocab_es.json")
    print("\n💾 Vocabularies saved!")


if __name__ == "__main__":
    main()