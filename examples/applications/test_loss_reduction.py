"""Quick test to verify loss reduction with optimized model."""

import numpy as np
import pickle
from train_email_model_large import EmailReplyModel, LargeEmailDataset

# Load dataset
print("Loading dataset...")
dataset = LargeEmailDataset()

# Create optimized model
print("\nCreating optimized model...")
model = EmailReplyModel(
    vocab_size=dataset.vocab_size,
    d_model=512,
    n_heads=8,
    n_layers=2,
    dropout=0.1
)

print(f"Model config: d_model={model.d_model}, n_heads={model.n_heads}, n_layers={model.n_layers}")
print(f"Learning rate: {model.current_lr}")

# Train for a few steps and check loss reduction
print("\nTraining for 20 steps to verify loss reduction...")
losses = []

for i in range(20):
    emails, replies, _ = dataset.get_batch(32, split='train')
    loss = model.train_step(emails, replies)
    losses.append(loss)
    
    if i % 5 == 0:
        print(f"Step {i:3d}: Loss = {loss:.4f}")

# Check improvement
initial_loss = np.mean(losses[:5])
final_loss = np.mean(losses[-5:])
improvement = (initial_loss - final_loss) / initial_loss * 100

print(f"\n📊 Results:")
print(f"Initial loss (avg first 5): {initial_loss:.4f}")
print(f"Final loss (avg last 5): {final_loss:.4f}")
print(f"Improvement: {improvement:.1f}%")

if final_loss < initial_loss:
    print("✅ Loss is decreasing! The optimizations are working.")
else:
    print("⚠️ Loss is not decreasing. Further optimization needed.")

print(f"\nExpected: Loss should drop from ~8.7 to ~2-4 with continued training")