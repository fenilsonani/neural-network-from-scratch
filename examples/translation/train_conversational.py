"""Training script optimized for conversational translation."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import time
import json
from typing import List, Tuple, Dict
from neural_arch.core import Tensor, Parameter
from neural_arch.optim import Adam
from neural_arch.functional import cross_entropy_loss
from vocabulary import Vocabulary, create_dataset
from model_v2 import TranslationTransformer


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


def clip_gradients(parameters, max_norm: float = 1.0):
    """Clip gradients by global norm."""
    total_norm = 0.0
    for param in parameters:
        if hasattr(param, 'grad') and param.grad is not None:
            total_norm += np.sum(param.grad ** 2)
    
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for param in parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad *= scale
    
    return total_norm


def train_epoch(model: TranslationTransformer,
                optimizer: Adam,
                src_data: List[List[int]],
                tgt_data: List[List[int]],
                batch_size: int = 16,
                pad_idx: int = 0,
                max_grad_norm: float = 1.0,
                show_progress: bool = True) -> float:
    """Train one epoch with gradient clipping."""
    total_loss = 0.0
    n_batches = 0
    
    # Shuffle data
    indices = np.random.permutation(len(src_data))
    
    # Calculate total batches
    total_batches = len(src_data) // batch_size
    
    for batch_idx, i in enumerate(range(0, len(src_data), batch_size)):
        if show_progress and batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.1f}%)", end='\r')
        
        # Get batch indices
        batch_indices = indices[i:i + batch_size]
        if len(batch_indices) < batch_size:
            continue
        
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
            
            # Connect gradients to model output
            if hasattr(output, 'backward'):
                full_grad = np.zeros_like(output_reshaped)
                full_grad[mask] = output_tensor.grad
                grad_reshaped = full_grad.reshape(output.data.shape)
                output.backward(grad_reshaped)
            
            # Clip gradients
            params = list(model.parameters())
            clip_gradients(params, max_grad_norm)
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.data
            n_batches += 1
    
    if show_progress:
        print()
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_conversational(model: TranslationTransformer,
                           src_vocab: Vocabulary,
                           tgt_vocab: Vocabulary):
    """Evaluate on conversational test phrases."""
    print("\n🌟 Conversational Translation Test:")
    print("-" * 60)
    
    test_phrases = [
        "hello",
        "how are you",
        "thank you",
        "good morning", 
        "i love you",
        "goodbye",
        "yes",
        "no",
        "please",
        "where is the bathroom",
        "what's your name",
        "nice to meet you",
        "see you later",
        "i'm hungry",
        "help me"
    ]
    
    correct = 0
    expected = {
        "hello": ["hola"],
        "how are you": ["cómo estás", "cómo está", "qué tal"],
        "thank you": ["gracias"],
        "good morning": ["buenos días"],
        "i love you": ["te amo", "te quiero"],
        "goodbye": ["adiós"],
        "yes": ["sí"],
        "no": ["no"],
        "please": ["por favor"],
        "where is the bathroom": ["dónde está el baño"],
    }
    
    for phrase in test_phrases:
        # Encode
        src_indices = src_vocab.encode(phrase, max_length=20)
        src_tensor = Tensor(np.array([src_indices]), requires_grad=False)
        
        # Generate with low temperature for more deterministic output
        output_indices = model.generate(
            src_tensor,
            max_length=15,  # Shorter max length
            sos_idx=tgt_vocab.word2idx[tgt_vocab.sos_token],
            eos_idx=tgt_vocab.word2idx[tgt_vocab.eos_token],
            temperature=0.5  # Lower temperature
        )
        
        # Decode
        translation = tgt_vocab.decode(output_indices, remove_special=True)
        
        # Check if correct
        is_correct = False
        if phrase in expected:
            for exp in expected[phrase]:
                if translation.lower().strip('.').strip() == exp.lower():
                    is_correct = True
                    correct += 1
                    break
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {phrase:25} → {translation}")
    
    accuracy = correct / len(expected) * 100 if expected else 0
    print(f"\nAccuracy on key phrases: {accuracy:.1f}%")


def main():
    """Main training function."""
    print("🌐 Conversational Translation Model Training")
    print("=" * 60)
    
    # First create conversational dataset
    if not os.path.exists("data/train_conversational.json"):
        print("📥 Creating conversational dataset...")
        os.system("python download_conversational.py")
    
    # Configuration
    BATCH_SIZE = 32       # Larger batch size since we have more data
    N_EPOCHS = 100        # More epochs for better learning
    MAX_SEQ_LEN = 20      # Slightly longer for Tatoeba sentences
    LEARNING_RATE = 0.001 # Good learning rate
    MAX_GRAD_NORM = 1.0   
    
    # Create vocabularies
    src_vocab = Vocabulary("english")
    tgt_vocab = Vocabulary("spanish")
    
    # Load datasets
    print(f"\n📚 Loading conversational dataset...")
    try:
        # Try Tatoeba dataset first (it's real conversational data)
        train_pairs = load_dataset("data/train_tatoeba.json")
        val_pairs = load_dataset("data/val_tatoeba.json")
        
        print(f"✅ Loaded {len(train_pairs)} training pairs from Tatoeba")
        print(f"✅ Loaded {len(val_pairs)} validation pairs from Tatoeba")
    except FileNotFoundError:
        try:
            # Fallback to other conversational dataset
            train_pairs = load_dataset("data/train_conversational.json")
            val_pairs = load_dataset("data/val_conversational.json")
            
            print(f"✅ Loaded {len(train_pairs)} training pairs")
            print(f"✅ Loaded {len(val_pairs)} validation pairs")
        except FileNotFoundError:
            print("❌ No dataset found! Please run:")
            print("   python process_spa_file.py")
            return
    
    # Create dataset
    print(f"\n📚 Processing dataset (max length: {MAX_SEQ_LEN})...")
    src_data, tgt_data = create_dataset(train_pairs, src_vocab, tgt_vocab, max_length=MAX_SEQ_LEN)
    val_src_data, val_tgt_data = create_dataset(val_pairs, src_vocab, tgt_vocab, max_length=MAX_SEQ_LEN)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Moderate model size for Tatoeba data
    d_model = 128   # Good size for vocabulary
    n_heads = 4     
    n_layers = 3    # 3 layers for better learning
    d_ff = 256      # Reasonable FFN size
    
    # Create model
    print(f"\n🤖 Creating Conversational Transformer model...")
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
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training info
    print(f"\n🏃 Training Configuration:")
    print(f"  - Epochs: {N_EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {optimizer.lr}")
    print(f"  - Training batches per epoch: {len(src_data) // BATCH_SIZE}")
    
    best_val_loss = float('inf')
    patience = 10
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
            max_grad_norm=MAX_GRAD_NORM,
            show_progress=True
        )
        
        # Validation
        val_loss = train_epoch(
            model, optimizer, val_src_data, val_tgt_data, 
            batch_size=BATCH_SIZE,
            pad_idx=src_vocab.word2idx[src_vocab.pad_token],
            max_grad_norm=MAX_GRAD_NORM,
            show_progress=False
        )
        
        epoch_time = time.time() - epoch_start
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s | 📈 Best!")
            
            # Save best model vocabularies
            src_vocab.save("vocab_en_conversational.json")
            tgt_vocab.save("vocab_es_conversational.json")
        else:
            patience_counter += 1
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            evaluate_conversational(model, src_vocab, tgt_vocab)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch + 1}")
            break
    
    print("\n✅ Training complete!")
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    evaluate_conversational(model, src_vocab, tgt_vocab)
    
    # Save model config
    model_info = {
        "model_type": "conversational",
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "best_val_loss": float(best_val_loss),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs_trained": epoch + 1
    }
    
    with open("model_config_conversational.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n💾 Model configuration and vocabularies saved!")
    print("\n🎉 Conversational model ready! Test with translate.py")


if __name__ == "__main__":
    main()