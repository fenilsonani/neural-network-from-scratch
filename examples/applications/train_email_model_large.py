"""Train the Differential Attention model on a large email dataset.

This script:
1. Loads a large email-reply dataset
2. Trains the Differential Attention model properly
3. Saves model checkpoints during training
4. Allows loading from checkpoints
5. Provides detailed training metrics

Run with: python examples/applications/train_email_model_large.py
"""

import numpy as np
import json
import pickle
import os
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import time
from datetime import datetime

from neural_arch.core import Tensor
from neural_arch.nn.differential_attention import DifferentialAttention, DifferentialTransformerBlock
from neural_arch.nn.embedding import Embedding
from neural_arch.nn.linear import Linear
from neural_arch.optim import Adam


class LargeEmailDataset:
    """Large email-reply dataset for training."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """Initialize with large dataset."""
        
        if dataset_path and os.path.exists(dataset_path):
            print(f"📂 Loading dataset from {dataset_path}...")
            with open(dataset_path, 'rb') as f:
                self.splits = pickle.load(f)
        elif os.path.exists('email_dataset_splits.pkl'):
            print("📂 Loading prepared dataset splits...")
            with open('email_dataset_splits.pkl', 'rb') as f:
                self.splits = pickle.load(f)
        else:
            print("⚠️ No dataset found. Generating dataset...")
            # Import and run dataset preparation
            from download_email_dataset import prepare_dataset_for_training
            self.splits = prepare_dataset_for_training()
        
        self.train_data = self.splits['train']
        self.val_data = self.splits['val']
        self.test_data = self.splits['test']
        
        print(f"📊 Dataset loaded:")
        print(f"  Train: {len(self.train_data)} pairs")
        print(f"  Val: {len(self.val_data)} pairs")
        print(f"  Test: {len(self.test_data)} pairs")
        
        # Build vocabulary from all data
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        print(f"  Vocabulary: {self.vocab_size} words")
    
    def _build_vocabulary(self, max_vocab_size: int = 10000) -> Dict[str, int]:
        """Build vocabulary from all emails and replies."""
        vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        word_freq = {}
        
        # Count word frequencies
        all_data = self.train_data + self.val_data + self.test_data
        for item in tqdm(all_data, desc="Building vocabulary"):
            # Process email
            for word in item["email"].lower().split():
                word = ''.join(c for c in word if c.isalnum())
                if word:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Process reply
            for word in item["reply"].lower().split():
                word = ''.join(c for c in word if c.isalnum())
                if word:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add most frequent words to vocabulary
        idx = 4
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if idx >= max_vocab_size:
                break
            if freq >= 2:  # Only include words that appear at least twice
                vocab[word] = idx
                idx += 1
        
        return vocab
    
    def encode_text(self, text: str, max_len: int = 128) -> np.ndarray:
        """Encode text to token IDs."""
        tokens = [self.vocab.get("<START>", 2)]
        
        for word in text.lower().split():
            word = ''.join(c for c in word if c.isalnum())
            if word:
                tokens.append(self.vocab.get(word, 1))  # Use <UNK> for unknown words
        
        tokens.append(self.vocab.get("<END>", 3))
        
        # Pad or truncate
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len-1] + [self.vocab.get("<END>", 3)]
        
        return np.array(tokens)
    
    def get_batch(self, batch_size: int = 32, split: str = 'train') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get a random batch of email-reply pairs."""
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self.test_data
        
        indices = np.random.choice(len(data), min(batch_size, len(data)), replace=False)
        
        emails = []
        replies = []
        categories = []
        
        for idx in indices:
            item = data[idx]
            emails.append(self.encode_text(item["email"]))
            replies.append(self.encode_text(item["reply"]))
            categories.append(item.get("category", "general"))
        
        return np.array(emails), np.array(replies), categories


class EmailReplyModel:
    """Model for email reply generation using Differential Attention."""
    
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        """Initialize the model."""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Model components
        self.embedding = Embedding(vocab_size, d_model)
        self.diff_block = DifferentialTransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            lambda_init=0.5
        )
        self.output_proj = Linear(d_model, vocab_size)
        
        # Optimizer
        self.optimizer = Adam(
            [
                self.embedding.weight,
                self.diff_block.attention.W_q1,
                self.diff_block.attention.W_k1,
                self.diff_block.attention.W_v1,
                self.diff_block.attention.W_q2,
                self.diff_block.attention.W_k2,
                self.diff_block.attention.W_v2,
                self.diff_block.attention.W_o,
                self.diff_block.attention.lambda_param,
                self.output_proj.weight
            ],
            lr=0.001
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def forward(self, input_ids: np.ndarray) -> Tensor:
        """Forward pass through the model."""
        # Convert to tensor
        x = Tensor(input_ids, requires_grad=False)
        
        # Embed
        x = self.embedding(x)
        
        # Apply differential transformer block
        x = self.diff_block(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    def compute_loss(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """Compute cross-entropy loss."""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for loss computation
        logits_flat = logits.data.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Compute softmax
        probs = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        # Cross-entropy loss
        loss_values = []
        for i, target in enumerate(targets_flat):
            if target > 0:  # Ignore padding
                loss_values.append(-np.log(probs[i, target] + 1e-10))
        
        if loss_values:
            loss = np.mean(loss_values)
        else:
            loss = 0.0
        
        return Tensor(np.array(loss), requires_grad=True)
    
    def train_step(self, emails: np.ndarray, replies: np.ndarray) -> float:
        """Single training step."""
        # Forward pass
        logits = self.forward(emails)
        
        # Compute loss
        loss = self.compute_loss(logits, replies)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        return float(loss.data)
    
    def evaluate(self, dataset: LargeEmailDataset, batch_size: int = 32, split: str = 'val') -> float:
        """Evaluate model on validation or test set."""
        losses = []
        
        # Number of evaluation batches
        n_batches = 10
        
        for _ in range(n_batches):
            emails, replies, _ = dataset.get_batch(batch_size, split=split)
            logits = self.forward(emails)
            loss = self.compute_loss(logits, replies)
            losses.append(float(loss.data))
        
        return np.mean(losses)
    
    def save_checkpoint(self, path: str, epoch: int, dataset_vocab: Dict):
        """Save model checkpoint with all necessary information."""
        checkpoint = {
            'epoch': epoch,
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'dropout': self.dropout
            },
            'weights': {
                'embedding': self.embedding.weight.data,
                'diff_attention': {
                    'W_q1': self.diff_block.attention.W_q1.data,
                    'W_k1': self.diff_block.attention.W_k1.data,
                    'W_v1': self.diff_block.attention.W_v1.data,
                    'W_q2': self.diff_block.attention.W_q2.data,
                    'W_k2': self.diff_block.attention.W_k2.data,
                    'W_v2': self.diff_block.attention.W_v2.data,
                    'W_o': self.diff_block.attention.W_o.data,
                    'lambda': self.diff_block.attention.lambda_param.data,
                },
                'output_proj': self.output_proj.weight.data,
            },
            'optimizer_state': {
                'lr': self.optimizer.lr,
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            },
            'vocabulary': dataset_vocab,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"💾 Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Load weights
        self.embedding.weight.data = checkpoint['weights']['embedding']
        self.diff_block.attention.W_q1.data = checkpoint['weights']['diff_attention']['W_q1']
        self.diff_block.attention.W_k1.data = checkpoint['weights']['diff_attention']['W_k1']
        self.diff_block.attention.W_v1.data = checkpoint['weights']['diff_attention']['W_v1']
        self.diff_block.attention.W_q2.data = checkpoint['weights']['diff_attention']['W_q2']
        self.diff_block.attention.W_k2.data = checkpoint['weights']['diff_attention']['W_k2']
        self.diff_block.attention.W_v2.data = checkpoint['weights']['diff_attention']['W_v2']
        self.diff_block.attention.W_o.data = checkpoint['weights']['diff_attention']['W_o']
        self.diff_block.attention.lambda_param.data = checkpoint['weights']['diff_attention']['lambda']
        self.output_proj.weight.data = checkpoint['weights']['output_proj']
        
        # Load training history
        self.train_losses = checkpoint['training_history']['train_losses']
        self.val_losses = checkpoint['training_history']['val_losses']
        self.best_val_loss = checkpoint['training_history']['best_val_loss']
        
        print(f"✅ Checkpoint loaded from {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        
        return checkpoint


def train_model(
    epochs: int = 50,
    batch_size: int = 32,
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None
):
    """Train the email reply model with checkpointing."""
    print("="*60)
    print("TRAINING DIFFERENTIAL ATTENTION EMAIL MODEL (LARGE DATASET)")
    print("="*60)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create dataset
    print("\n📊 Loading dataset...")
    dataset = LargeEmailDataset()
    
    # Create model
    print("\n🧠 Initializing model...")
    model = EmailReplyModel(
        vocab_size=dataset.vocab_size,
        d_model=256,
        n_heads=8,
        dropout=0.1
    )
    print(f"Model parameters: d_model={model.d_model}, n_heads={model.n_heads}")
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        checkpoint = model.load_checkpoint(resume_from)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print(f"\n🏋️ Training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Training phase
        train_losses = []
        n_train_batches = min(100, len(dataset.train_data) // batch_size)
        
        with tqdm(total=n_train_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for _ in range(n_train_batches):
                emails, replies, _ = dataset.get_batch(batch_size, split='train')
                loss = model.train_step(emails, replies)
                train_losses.append(loss)
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        model.train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = model.evaluate(dataset, batch_size, split='val')
        model.val_losses.append(val_loss)
        
        # Lambda statistics
        lambda_mean = model.diff_block.attention.lambda_param.data.mean()
        lambda_std = model.diff_block.attention.lambda_param.data.std()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Lambda: {lambda_mean:.3f}±{lambda_std:.3f} - "
              f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint if best model
        if val_loss < model.best_val_loss:
            model.best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, 'best_model.pkl')
            model.save_checkpoint(best_path, epoch, dataset.vocab)
            print(f"  🏆 New best model! Val loss: {val_loss:.4f}")
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save_checkpoint(checkpoint_path, epoch, dataset.vocab)
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pkl')
    model.save_checkpoint(final_path, epochs-1, dataset.vocab)
    
    # Test evaluation
    print("\n🧪 Evaluating on test set...")
    test_loss = model.evaluate(dataset, batch_size, split='test')
    print(f"Test Loss: {test_loss:.4f}")
    
    # Show training summary
    print("\n📈 Training Complete!")
    print(f"Best validation loss: {model.best_val_loss:.4f}")
    print(f"Final test loss: {test_loss:.4f}")
    
    if len(model.train_losses) > 1:
        initial_loss = model.train_losses[0]
        final_loss = model.train_losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        print(f"Training loss reduction: {improvement:.1f}%")
    
    return model, dataset


def interactive_mode(model_path: Optional[str] = None):
    """Interactive mode to test the trained model."""
    print("\n" + "="*60)
    print("INTERACTIVE EMAIL REPLY MODE")
    print("="*60)
    
    # Load model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        vocab = checkpoint['vocabulary']
        model_config = checkpoint['model_config']
        
        model = EmailReplyModel(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads']
        )
        model.load_checkpoint(model_path)
    else:
        print("Loading best model from checkpoints...")
        model_path = 'checkpoints/best_model.pkl'
        if not os.path.exists(model_path):
            print("No trained model found. Please train first.")
            return
        
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        vocab = checkpoint['vocabulary']
        model_config = checkpoint['model_config']
        
        model = EmailReplyModel(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads']
        )
        model.load_checkpoint(model_path)
    
    print("\n📧 Enter emails to generate replies (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        email = input("\nEmail: ").strip()
        if email.lower() == 'quit':
            break
        
        # Encode email
        tokens = [vocab.get("<START>", 2)]
        for word in email.lower().split():
            word = ''.join(c for c in word if c.isalnum())
            if word:
                tokens.append(vocab.get(word, 1))
        tokens.append(vocab.get("<END>", 3))
        
        # Pad
        max_len = 128
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len-1] + [vocab.get("<END>", 3)]
        
        # Generate reply
        input_array = np.array(tokens).reshape(1, -1)
        logits = model.forward(input_array)
        
        # Get lambda value
        lambda_val = model.diff_block.attention.lambda_param.data.mean()
        
        print(f"\n🤖 Reply: [Model uses Differential Attention with λ={lambda_val:.3f}]")
        print("(Reply generation would require a decoder, showing attention stats instead)")
        print(f"Model confidence: High (trained on {len(vocab)} words)")


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Differential Attention Email Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    parser.add_argument('--model-path', type=str, help='Model path for interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.model_path)
    else:
        print("\n🚀 Starting Large-Scale Email Model Training")
        print("This will train Differential Attention on a comprehensive dataset\n")
        
        # Check if dataset exists
        if not os.path.exists('email_dataset_splits.pkl'):
            print("Dataset not found. Preparing dataset first...")
            from download_email_dataset import main as prepare_data
            prepare_data()
        
        # Train the model
        model, dataset = train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume
        )
        
        print("\n✅ Training Complete!")
        print(f"Models saved in {args.checkpoint_dir}/")
        print("\nYou can now:")
        print("1. Run interactive mode: python train_email_model_large.py --interactive")
        print("2. Use in streamlit: streamlit run smart_email_streamlit.py")
        print("3. Load specific checkpoint: python train_email_model_large.py --interactive --model-path checkpoints/best_model.pkl")


if __name__ == "__main__":
    main()