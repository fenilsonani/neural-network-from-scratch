"""Optimized training script for English-Spanish translation model."""

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
                batch_size: int = 8,
                pad_idx: int = 0,
                show_progress: bool = True) -> float:
    """Train one epoch with progress tracking."""
    total_loss = 0.0
    n_batches = 0
    
    # Shuffle data
    indices = np.random.permutation(len(src_data))
    
    # Calculate total batches
    total_batches = len(src_data) // batch_size
    
    for batch_idx, i in enumerate(range(0, len(src_data), batch_size)):
        # Show progress
        if show_progress and batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.1f}%)", end='\r')
        
        # Get batch indices
        batch_indices = indices[i:i + batch_size]
        if len(batch_indices) < batch_size:
            continue  # Skip incomplete batches
        
        # Create batches
        src_batch = create_batch([src_data[j] for j in batch_indices], batch_size, pad_idx)
        tgt_batch = create_batch([tgt_data[j] for j in batch_indices], batch_size, pad_idx)
        
        # Create tensors
        src_tensor = Tensor(src_batch, requires_grad=False)
        tgt_input = Tensor(tgt_batch[:, :-1], requires_grad=False)
        tgt_output = Tensor(tgt_batch[:, 1:], requires_grad=False)
        
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
    
    if show_progress:
        print()  # New line after progress
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(model: TranslationTransformer,
             src_data: List[List[int]],
             tgt_data: List[List[int]],
             src_vocab: Vocabulary,
             tgt_vocab: Vocabulary,
             n_examples: int = 5):
    """Evaluate model with examples."""
    print("\nTranslation Examples:")
    print("-" * 60)
    
    # Use first n_examples instead of random for consistency
    for idx in range(min(n_examples, len(src_data))):
        # Get source
        src = Tensor(np.array([src_data[idx]]), requires_grad=False)
        
        # Generate translation
        output_indices = model.generate(
            src, 
            max_length=30,
            sos_idx=tgt_vocab.word2idx[tgt_vocab.sos_token],
            eos_idx=tgt_vocab.word2idx[tgt_vocab.eos_token]
        )
        
        # Decode
        src_text = src_vocab.decode(src_data[idx], remove_special=True)
        tgt_text = tgt_vocab.decode(tgt_data[idx], remove_special=True)
        pred_text = tgt_vocab.decode(output_indices, remove_special=True)
        
        print(f"EN: {src_text[:50]}")
        print(f"ES: {tgt_text[:50]}")
        print(f"PR: {pred_text[:50]}")
        print("-" * 60)


def test_common_phrases(model, src_vocab, tgt_vocab):
    """Test on common phrases."""
    print("\n🌟 Common Phrase Translations:")
    print("-" * 60)
    
    test_phrases = [
        "hello",
        "how are you",
        "thank you",
        "good morning",
        "where is the bathroom",
        "i love you",
        "goodbye",
        "yes",
        "no",
        "please"
    ]
    
    for phrase in test_phrases:
        # Encode
        src_indices = src_vocab.encode(phrase, max_length=20)
        src_tensor = Tensor(np.array([src_indices]), requires_grad=False)
        
        # Generate
        output_indices = model.generate(
            src_tensor,
            max_length=20,
            sos_idx=tgt_vocab.word2idx[tgt_vocab.sos_token],
            eos_idx=tgt_vocab.word2idx[tgt_vocab.eos_token]
        )
        
        # Decode
        translation = tgt_vocab.decode(output_indices, remove_special=True)
        print(f"{phrase:25} → {translation}")


def main():
    """Main training function."""
    print("🌐 Optimized Translation Model Training")
    print("=" * 60)
    
    # Configuration
    DATASET_SIZE = 20000  # Medium size
    BATCH_SIZE = 8        # Smaller batches for faster processing
    N_EPOCHS = 50         # Reasonable number of epochs
    MAX_SEQ_LEN = 25      # Shorter sequences for efficiency
    
    # Create vocabularies
    src_vocab = Vocabulary("english")
    tgt_vocab = Vocabulary("spanish")
    
    # Load datasets
    print(f"\n📚 Loading dataset (using first {DATASET_SIZE} pairs)...")
    try:
        all_train_pairs = load_dataset("data/train_large.json")
        all_val_pairs = load_dataset("data/val_large.json")
        
        # Use subset for medium-sized training
        train_pairs = all_train_pairs[:DATASET_SIZE]
        val_pairs = all_val_pairs[:2000]  # 2000 validation pairs
        
        print(f"✅ Using {len(train_pairs)} training pairs")
        print(f"✅ Using {len(val_pairs)} validation pairs")
    except FileNotFoundError:
        print("❌ Dataset files not found. Please run download_europarl.py first!")
        return
    
    # Create dataset
    print(f"\n📚 Processing dataset (max length: {MAX_SEQ_LEN})...")
    src_data, tgt_data = create_dataset(train_pairs, src_vocab, tgt_vocab, max_length=MAX_SEQ_LEN)
    val_src_data, val_tgt_data = create_dataset(val_pairs, src_vocab, tgt_vocab, max_length=MAX_SEQ_LEN)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Model configuration - Balanced for performance
    d_model = 128   # Moderate size
    n_heads = 4     # Moderate attention heads
    n_layers = 3    # 3 layers for balance
    d_ff = 256      # Feed-forward dimension
    
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
    
    # Training info
    print(f"\n🏃 Training Configuration:")
    print(f"  - Epochs: {N_EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {optimizer.lr}")
    print(f"  - Training batches per epoch: {len(src_data) // BATCH_SIZE}")
    print(f"  - Estimated time per epoch: ~{(len(src_data) // BATCH_SIZE) * 0.1:.0f} seconds")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print(f"\n🚀 Starting training...\n")
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        # Training
        print(f"Epoch {epoch + 1}/{N_EPOCHS}")
        train_loss = train_epoch(
            model, optimizer, src_data, tgt_data, 
            batch_size=BATCH_SIZE,
            pad_idx=src_vocab.word2idx[src_vocab.pad_token],
            show_progress=True
        )
        
        # Validation (no progress bar)
        val_loss = train_epoch(
            model, optimizer, val_src_data, val_tgt_data, 
            batch_size=BATCH_SIZE,
            pad_idx=src_vocab.word2idx[src_vocab.pad_token],
            show_progress=False
        )
        
        epoch_time = time.time() - epoch_start
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s | 📈 Best!")
        else:
            patience_counter += 1
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            evaluate(model, val_src_data, val_tgt_data, src_vocab, tgt_vocab, n_examples=3)
            test_common_phrases(model, src_vocab, tgt_vocab)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch + 1}")
            break
    
    print("\n✅ Training complete!")
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    evaluate(model, val_src_data, val_tgt_data, src_vocab, tgt_vocab, n_examples=5)
    test_common_phrases(model, src_vocab, tgt_vocab)
    
    # Save everything
    src_vocab.save("vocab_en_final.json")
    tgt_vocab.save("vocab_es_final.json")
    
    model_info = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "best_val_loss": float(best_val_loss),
        "dataset_size": DATASET_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_trained": epoch + 1
    }
    
    with open("model_config.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n💾 Model configuration and vocabularies saved!")
    print("\n🎉 You can now use translate.py to test the model!")


if __name__ == "__main__":
    main()