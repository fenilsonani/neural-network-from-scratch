#!/usr/bin/env python3
"""Example usage of the Lion optimizer implementation.

This example demonstrates:
1. Basic Lion optimizer usage
2. Comparison with Adam optimizer
3. Hyperparameter recommendations
4. Training a simple neural network
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_arch.core.base import Parameter
from neural_arch.optim.lion import Lion
from neural_arch.optim.adam import Adam


def simple_linear_model(x, weights, bias):
    """Simple linear model: y = W*x + b"""
    return np.dot(x, weights.data) + bias.data[0]  # bias is now an array


def mse_loss(predictions, targets):
    """Mean squared error loss"""
    return np.mean((predictions - targets) ** 2)


def compute_gradients(x, y, predictions, weights, bias):
    """Manually compute gradients for linear model"""
    n = len(x)
    error = predictions - y
    
    # dL/dW = (2/n) * X^T * (predictions - y)
    weights.grad = (2.0 / n) * np.dot(x.T, error)
    
    # dL/db = (2/n) * sum(predictions - y) - convert to array for tensor compatibility
    bias.grad = np.array([(2.0 / n) * np.sum(error)])  # Match bias shape


def train_model(optimizer, x_train, y_train, epochs=100):
    """Train a simple linear model"""
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = simple_linear_model(x_train, optimizer.parameters["weights"], 
                                        optimizer.parameters["bias"])
        loss = mse_loss(predictions, y_train)
        losses.append(loss)
        
        # Compute gradients
        compute_gradients(x_train, y_train, predictions, 
                         optimizer.parameters["weights"], 
                         optimizer.parameters["bias"])
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss:.6f}")
    
    return losses


def main():
    """Demonstrate Lion optimizer usage and comparison with Adam."""
    print("Lion Optimizer Example")
    print("=" * 50)
    
    # Generate synthetic data: y = 2*x1 + 3*x2 + 1 + noise
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([2.0, 3.0])
    true_bias = 1.0
    noise = 0.1 * np.random.randn(n_samples)
    y = np.dot(X, true_weights) + true_bias + noise
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    print()
    
    # Initialize model parameters for Lion (use arrays for all parameters)
    weights_lion = Parameter(np.random.randn(n_features) * 0.1, name="weights")
    bias_lion = Parameter(np.array([np.random.randn() * 0.1]), name="bias")  # Make bias an array
    params_lion = {"weights": weights_lion, "bias": bias_lion}
    
    # Initialize model parameters for Adam (same initial values)
    weights_adam = Parameter(weights_lion.data.copy(), name="weights")
    bias_adam = Parameter(bias_lion.data.copy(), name="bias")
    params_adam = {"weights": weights_adam, "bias": bias_adam}
    
    print("Initial parameters:")
    print(f"Weights: {weights_lion.data}")
    print(f"Bias: {bias_lion.data}")
    print()
    
    # Create optimizers with recommended hyperparameters
    print("Optimizer configurations:")
    
    # Lion: smaller learning rate, often works well with higher weight decay
    lion_optimizer = Lion(
        params_lion,
        lr=1e-3,      # Lion typically needs smaller LR than Adam
        beta1=0.9,    # Standard momentum for update direction
        beta2=0.99,   # Standard momentum for buffer update
        weight_decay=0.01  # Lion often works well with higher weight decay
    )
    print(f"Lion: {lion_optimizer}")
    
    # Adam: standard hyperparameters
    adam_optimizer = Adam(
        params_adam,
        lr=1e-2,      # Adam can use higher learning rates
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01
    )
    print(f"Adam: {adam_optimizer}")
    print()
    
    # Train both models
    print("Training with Lion optimizer:")
    print("-" * 30)
    lion_losses = train_model(lion_optimizer, X, y, epochs=100)
    
    print("\nTraining with Adam optimizer:")
    print("-" * 30)
    adam_losses = train_model(adam_optimizer, X, y, epochs=100)
    
    # Compare final results
    print("\nFinal Results:")
    print("=" * 50)
    
    print("Lion Optimizer:")
    print(f"  Final weights: {weights_lion.data}")
    print(f"  Final bias: {bias_lion.data[0]:.6f}")
    print(f"  Final loss: {lion_losses[-1]:.6f}")
    print(f"  Weight error: {np.linalg.norm(weights_lion.data - true_weights):.6f}")
    print(f"  Bias error: {abs(bias_lion.data[0] - true_bias):.6f}")
    
    print("\nAdam Optimizer:")
    print(f"  Final weights: {weights_adam.data}")
    print(f"  Final bias: {bias_adam.data[0]:.6f}")
    print(f"  Final loss: {adam_losses[-1]:.6f}")
    print(f"  Weight error: {np.linalg.norm(weights_adam.data - true_weights):.6f}")
    print(f"  Bias error: {abs(bias_adam.data[0] - true_bias):.6f}")
    
    # Demonstrate Lion optimizer features
    print("\nLion Optimizer Features:")
    print("=" * 50)
    
    # Statistics
    stats = lion_optimizer.get_statistics()
    print("Optimizer statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # State dict operations
    print("\nState dict operations:")
    state_dict = lion_optimizer.get_state_dict()
    print(f"State dict keys: {list(state_dict.keys())}")
    print(f"Number of parameter states: {len(state_dict['state'])}")
    
    # Learning rate adjustment
    print(f"\nOriginal learning rate: {lion_optimizer.get_lr()}")
    lion_optimizer.set_lr(5e-4)
    print(f"Updated learning rate: {lion_optimizer.get_lr()}")
    
    print("\nKey differences of Lion vs Adam:")
    print("=" * 50)
    print("1. Sign-based updates: Lion uses sign(interpolation) instead of adaptive scaling")
    print("2. Memory efficient: Only stores momentum buffer (not second moment)")
    print("3. Simpler computation: No bias correction or second moment estimation")
    print("4. Different learning rates: Lion typically needs 3-10x smaller LR than Adam")
    print("5. Robust to hyperparameters: Less sensitive to beta values")
    print("6. Better generalization: Often achieves better test performance")


if __name__ == "__main__":
    main()