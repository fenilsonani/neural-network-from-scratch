#!/usr/bin/env python3
"""Complete Distributed Training Example.

This example demonstrates end-to-end distributed training with Neural Forge,
including data parallel training, gradient synchronization, and multi-node setup.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_arch.core import Tensor
from neural_arch.nn import (
    Module, Sequential, Linear, ReLU, Dropout, LayerNorm,
    CrossEntropyLoss
)
from neural_arch.optim import Adam, SGD
from neural_arch.distributed import (
    init_process_group, destroy_process_group, is_initialized,
    get_world_size, get_rank, barrier,
    DistributedDataParallel, DistributedSampler,
    get_distributed_info
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Rank %(rank)s] %(levelname)s: %(message)s'
)

def get_logger():
    """Get logger with rank information."""
    logger = logging.getLogger(__name__)
    rank = get_rank() if is_initialized() else 0
    
    # Add rank to all log records
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    return logger

logger = get_logger()


class SimpleTransformer(Module):
    """Simple transformer model for distributed training demonstration."""
    
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_classes: int):
        """Initialize transformer model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_classes: Number of output classes
        """
        super().__init__()
        self.d_model = d_model
        
        # Simple embedding (without learned positional encoding for simplicity)
        self.embedding = Linear(vocab_size, d_model)
        
        # Transformer layers (simplified)
        layers = []
        for _ in range(num_layers):
            layers.extend([
                Linear(d_model, d_model * 4),
                ReLU(),
                Dropout(0.1),
                Linear(d_model * 4, d_model),
                LayerNorm(d_model),
                Dropout(0.1)
            ])
        
        self.layers = Sequential(*layers)
        
        # Output projection
        self.output_proj = Linear(d_model, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, vocab_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # Apply layers
        hidden = self.layers(embedded)
        
        # Global average pooling over sequence dimension
        pooled = hidden.mean(axis=1)  # (batch_size, d_model)
        
        # Output projection
        output = self.output_proj(pooled)  # (batch_size, num_classes)
        
        return output


class SyntheticDataset:
    """Synthetic dataset for distributed training demonstration."""
    
    def __init__(self, size: int, seq_len: int, vocab_size: int, num_classes: int):
        """Initialize synthetic dataset.
        
        Args:
            size: Dataset size
            seq_len: Sequence length
            vocab_size: Vocabulary size
            num_classes: Number of classes
        """
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        # Generate synthetic data
        np.random.seed(42)  # For reproducibility
        self.data = np.random.randn(size, seq_len, vocab_size).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size).astype(np.int64)
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get item at index.
        
        Args:
            idx: Dataset index
            
        Returns:
            Tuple of (data, label) tensors
        """
        return (
            Tensor(self.data[idx], dtype=np.float32),
            Tensor(self.labels[idx], dtype=np.int64)
        )


class DistributedTrainer:
    """Distributed trainer for Neural Forge models."""
    
    def __init__(self, model: Module, optimizer, criterion, dataset, args):
        """Initialize distributed trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss criterion
            dataset: Training dataset
            args: Training arguments
        """
        self.args = args
        self.dataset = dataset
        self.criterion = criterion
        
        # Wrap model with DDP
        self.model = DistributedDataParallel(model)
        self.optimizer = optimizer
        
        # Create distributed sampler
        self.sampler = DistributedSampler(
            dataset_size=len(dataset),
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True
        )
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
        logger.info(f"Initialized distributed trainer")
        logger.info(f"  Model parameters: {sum(p.size for p in model.parameters()):,}")
        logger.info(f"  Dataset size: {len(dataset):,}")
        logger.info(f"  Samples per rank: {len(self.sampler):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.sampler.set_epoch(self.epoch)
        
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        # Training loop
        batch_count = 0
        for idx in self.sampler:
            if batch_count >= self.args.steps_per_epoch:
                break
            
            # Get batch data
            batch_indices = []
            for _ in range(self.args.batch_size):
                if idx < len(self.dataset):
                    batch_indices.append(idx)
                    idx += 1
                else:
                    break
            
            if not batch_indices:
                continue
            
            # Load batch
            batch_data = []
            batch_labels = []
            for bidx in batch_indices:
                data, label = self.dataset[bidx]
                batch_data.append(data.data)
                batch_labels.append(label.data)
            
            if not batch_data:
                continue
            
            # Stack into batches
            batch_x = Tensor(np.stack(batch_data), dtype=np.float32)
            batch_y = Tensor(np.array(batch_labels), dtype=np.int64)
            
            # Forward pass
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient synchronization
            self.model.sync_gradients()
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += float(loss.data)
            total_samples += len(batch_indices)
            batch_count += 1
            self.step += 1
            
            # Log progress
            if batch_count % self.args.log_interval == 0:
                avg_loss = total_loss / batch_count
                logger.info(
                    f"Epoch {self.epoch}, Step {self.step}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Samples: {total_samples}"
                )
        
        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(batch_count, 1)
        
        # Synchronize metrics across ranks
        if is_initialized():
            loss_tensor = Tensor(np.array([avg_loss]), dtype=np.float32)
            from neural_arch.distributed import all_reduce, ReduceOp
            avg_loss_tensor = all_reduce(loss_tensor, ReduceOp.AVERAGE)
            avg_loss = float(avg_loss_tensor.data[0])
        
        metrics = {
            'loss': avg_loss,
            'samples': total_samples,
            'time': epoch_time,
            'steps': batch_count
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation (simplified for demo)."""
        self.model.eval()
        
        # Simple validation with a few samples
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad() if hasattr(torch, 'no_grad') else self:  # Simplified
            for i in range(min(10, len(self.dataset))):
                data, label = self.dataset[i]
                data = data.unsqueeze(0)  # Add batch dimension
                label = label.unsqueeze(0)
                
                output = self.model(data)
                loss = self.criterion(output, label)
                
                val_loss += float(loss.data)
                val_samples += 1
        
        avg_val_loss = val_loss / max(val_samples, 1)
        
        return {'val_loss': avg_val_loss, 'val_samples': val_samples}
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting distributed training for {self.args.epochs} epochs")
        
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation (only on rank 0 to avoid redundancy)
            val_metrics = {}
            if get_rank() == 0:
                val_metrics = self.validate()
            
            # Log epoch results
            if get_rank() == 0:
                logger.info(
                    f"Epoch {epoch} completed: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Time: {train_metrics['time']:.2f}s, "
                    f"Samples: {train_metrics['samples']}"
                )
                
                if val_metrics:
                    logger.info(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint (simplified)
            if train_metrics['loss'] < self.best_loss:
                self.best_loss = train_metrics['loss']
                if get_rank() == 0:
                    logger.info(f"New best loss: {self.best_loss:.4f}")
            
            # Synchronize between epochs
            barrier()
        
        logger.info("Training completed!")


def setup_distributed_training(args):
    """Setup distributed training environment."""
    
    # Check if we're in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Launched with distributed launcher
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        logger.info(f"Initializing distributed training: rank {rank}/{world_size}")
        
        # Initialize process group
        init_process_group(
            backend=args.backend,
            world_size=world_size,
            rank=rank
        )
        
        logger.info("Distributed process group initialized")
        
    else:
        # Single process training
        logger.info("Running in single-process mode")
        init_process_group(backend=args.backend, world_size=1, rank=0)
    
    # Log distributed info
    dist_info = get_distributed_info()
    logger.info(f"Distributed training info: {dist_info}")


def create_model_and_optimizer(args):
    """Create model and optimizer."""
    
    # Create model
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_classes=args.num_classes
    )
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate)
    
    # Create loss criterion
    criterion = CrossEntropyLoss()
    
    logger.info(f"Created model with {sum(p.size for p in model.parameters()):,} parameters")
    
    return model, optimizer, criterion


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Distributed Training Example")
    
    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size per rank')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    
    # Dataset arguments
    parser.add_argument('--dataset-size', type=int, default=10000, help='Dataset size')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence length')
    
    # Distributed arguments
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'])
    
    args = parser.parse_args()
    
    try:
        # Setup distributed training
        setup_distributed_training(args)
        
        # Create dataset
        dataset = SyntheticDataset(
            size=args.dataset_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes
        )
        
        # Create model and optimizer
        model, optimizer, criterion = create_model_and_optimizer(args)
        
        # Create trainer
        trainer = DistributedTrainer(model, optimizer, criterion, dataset, args)
        
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if is_initialized():
            destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())