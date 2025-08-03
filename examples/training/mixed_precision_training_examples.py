"""Mixed precision training examples for Neural Forge.

This module provides comprehensive examples of how to use mixed precision training
with Neural Forge, including:
- Basic mixed precision setup
- Advanced configuration options
- Integration with different model types
- Best practices and troubleshooting
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import Neural Forge components
from src.neural_arch.core.base import Module, Parameter
from src.neural_arch.core.tensor import Tensor
from src.neural_arch.nn.linear import Linear
from src.neural_arch.nn.activation import ReLU
from src.neural_arch.optim.adam import Adam
from src.neural_arch.optim.adamw import AdamW

# Import mixed precision components
from src.neural_arch.optimization.mixed_precision import (
    AutocastConfig,
    AutocastPolicy,
    PrecisionConfig,
    MixedPrecisionManager,
    autocast,
    create_precision_config,
    get_recommended_precision_config,
    create_training_context,
)

# Import advanced mixed precision components
try:
    from src.neural_arch.optimization.grad_scaler import (
        AdvancedGradScaler,
        ScalerConfig,
        ScalingStrategy,
        create_scaler,
    )
    from src.neural_arch.optimization.amp_optimizer import (
        create_amp_adam,
        create_amp_adamw,
        AMPOptimizer,
        get_recommended_scaler_config,
    )
    ADVANCED_AMP_AVAILABLE = True
except ImportError:
    ADVANCED_AMP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMLPModel(Module):
    """Simple MLP model for demonstration."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, output_size: int = 10):
        super().__init__()
        self.layer1 = Linear(input_size, hidden_size)
        self.activation1 = ReLU()
        self.layer2 = Linear(hidden_size, hidden_size)
        self.activation2 = ReLU()
        self.layer3 = Linear(hidden_size, output_size)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        return x


class TransformerLikeModel(Module):
    """Transformer-like model for demonstration."""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 512, num_layers: int = 6):
        super().__init__()
        self.embedding = Linear(vocab_size, d_model)
        
        # Create multiple transformer-like layers
        self.layers = []
        for i in range(num_layers):
            # Attention simulation (simplified)
            attention = Linear(d_model, d_model)
            feedforward = Linear(d_model, d_model * 4)
            output_proj = Linear(d_model * 4, d_model)
            
            setattr(self, f"attention_{i}", attention)
            setattr(self, f"feedforward_{i}", feedforward)
            setattr(self, f"output_proj_{i}", output_proj)
            
            self.layers.append((attention, feedforward, output_proj))
        
        self.output_layer = Linear(d_model, vocab_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Embedding
        x = self.embedding(x)
        
        # Transformer layers
        for attention, feedforward, output_proj in self.layers:
            # Simplified attention
            attn_out = attention(x)
            
            # Feedforward
            ff_out = feedforward(attn_out)
            ff_out = Tensor(np.maximum(0, ff_out.data), requires_grad=True)  # ReLU
            x = output_proj(ff_out)
        
        # Output
        output = self.output_layer(x)
        return output


def generate_classification_data(batch_size: int, input_size: int, num_classes: int) -> Tuple[Tensor, Tensor]:
    """Generate synthetic classification data."""
    # Input features
    x_data = np.random.randn(batch_size, input_size).astype(np.float32)
    x = Tensor(x_data, requires_grad=False)
    
    # Target labels (one-hot encoded)
    y_indices = np.random.randint(0, num_classes, batch_size)
    y_data = np.eye(num_classes)[y_indices].astype(np.float32)
    y = Tensor(y_data, requires_grad=False)
    
    return x, y


def compute_cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute cross-entropy loss."""
    # Softmax
    exp_preds = np.exp(predictions.data - np.max(predictions.data, axis=1, keepdims=True))
    softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    # Cross-entropy
    epsilon = 1e-7
    log_probs = np.log(softmax_preds + epsilon)
    loss = -np.sum(targets.data * log_probs) / targets.data.shape[0]
    
    return Tensor(np.array([loss]), requires_grad=True)


def example_1_basic_mixed_precision():
    """Example 1: Basic mixed precision training setup."""
    print("\n" + "="*60)
    print("🔥 Example 1: Basic Mixed Precision Training")
    print("="*60)
    
    # Create model and data
    model = SimpleMLPModel(input_size=784, hidden_size=256, output_size=10)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Create mixed precision manager
    mp_manager = MixedPrecisionManager()
    
    print(f"📊 Model parameters: {sum(p.data.size for p in model.parameters()):,}")
    print(f"🔧 Mixed precision enabled: {mp_manager.config.enabled}")
    
    # Training loop
    num_steps = 100
    start_time = time.time()
    
    for step in range(num_steps):
        # Generate data
        x, y = generate_classification_data(32, 784, 10)
        
        # Forward pass with autocast
        with mp_manager.autocast():
            predictions = model(x)
            loss = compute_cross_entropy_loss(predictions, y)
        
        # Backward pass with mixed precision
        success = mp_manager.backward_and_step(loss, optimizer, model)
        
        if step % 20 == 0:
            scale = mp_manager.scaler.get_scale()
            print(f"Step {step:3d}: Loss = {loss.data[0]:.4f}, "
                  f"Scale = {scale:8.0f}, Success = {success}")
    
    training_time = time.time() - start_time
    
    # Get statistics
    stats = mp_manager.get_statistics()
    print(f"\n📈 Training completed in {training_time:.2f}s")
    print(f"🎯 Success rate: {stats['success_rate']:.1%}")
    print(f"📏 Current scale: {stats['current_scale']:.0f}")


def example_2_advanced_scaler_configuration():
    """Example 2: Advanced gradient scaler configuration."""
    print("\n" + "="*60)
    print("🚀 Example 2: Advanced Gradient Scaler Configuration")
    print("="*60)
    
    if not ADVANCED_AMP_AVAILABLE:
        print("❌ Advanced AMP components not available")
        return
    
    # Create model
    model = TransformerLikeModel(vocab_size=1000, d_model=512, num_layers=4)
    
    # Create advanced scaler with conservative strategy
    scaler_config = ScalerConfig(
        init_scale=16384.0,  # More conservative initial scale
        strategy=ScalingStrategy.CONSERVATIVE,
        growth_interval=3000,  # Slower growth
        gradient_clip_threshold=1.0,  # Enable gradient clipping
        stability_check_interval=50,  # Frequent stability checks
    )
    
    scaler = AdvancedGradScaler(scaler_config)
    
    # Create AMP-aware optimizer
    optimizer = create_amp_adam(
        model.parameters(),
        lr=0.0001,  # Smaller learning rate for stability
        scaler_config=scaler_config
    )
    
    print(f"📊 Model parameters: {sum(p.data.size for p in model.parameters()):,}")
    print(f"⚙️  Scaling strategy: {scaler_config.strategy.value}")
    print(f"📏 Initial scale: {scaler_config.init_scale}")
    
    # Training loop
    num_steps = 200
    start_time = time.time()
    
    for step in range(num_steps):
        # Generate sequence data
        batch_size, seq_len = 16, 128
        x_data = np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.float32)
        x = Tensor(x_data, requires_grad=False)
        
        y_data = np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.float32)
        y = Tensor(y_data, requires_grad=False)
        
        # Forward pass with autocast
        with autocast():
            predictions = model(x)
            loss = compute_cross_entropy_loss(predictions.data.reshape(-1, 1000), 
                                            np.eye(1000)[y.data.astype(int).flatten()])
        
        # Backward pass with AMP optimizer
        optimizer.zero_grad()
        optimizer.backward(loss)
        success = optimizer.step()
        
        if step % 40 == 0:
            stats = optimizer.get_statistics()
            print(f"Step {step:3d}: Loss = {loss.data[0]:.4f}, "
                  f"Scale = {stats.get('current_scale', 0):8.0f}, "
                  f"Success Rate = {stats.get('success_rate', 0):.1%}")
    
    training_time = time.time() - start_time
    
    # Final statistics
    final_stats = optimizer.get_statistics()
    print(f"\n📈 Training completed in {training_time:.2f}s")
    print(f"🎯 Final success rate: {final_stats['success_rate']:.1%}")
    print(f"⚠️  Overflow rate: {final_stats.get('skip_rate', 0):.1%}")
    print(f"✂️  Gradient clipping rate: {final_stats.get('clip_rate', 0):.1%}")


def example_3_different_autocast_policies():
    """Example 3: Comparing different autocast policies."""
    print("\n" + "="*60)
    print("🎨 Example 3: Different Autocast Policies")
    print("="*60)
    
    # Test different policies
    policies = [
        (AutocastPolicy.CONSERVATIVE, "🛡️  Conservative"),
        (AutocastPolicy.SELECTIVE, "⚖️  Selective"),
        (AutocastPolicy.AGGRESSIVE, "⚡ Aggressive"),
    ]
    
    results = {}
    
    for policy, policy_name in policies:
        print(f"\n{policy_name} Policy:")
        print("-" * 30)
        
        # Create model and optimizer
        model = SimpleMLPModel(input_size=512, hidden_size=128, output_size=20)
        optimizer = AdamW(model.parameters(), lr=0.001)
        
        # Create precision config with specific policy
        precision_config = create_precision_config(
            enabled=True,
            policy=policy.value,
            loss_scale=32768.0
        )
        
        mp_manager = MixedPrecisionManager(precision_config)
        
        # Training loop
        num_steps = 50
        start_time = time.time()
        losses = []
        
        for step in range(num_steps):
            # Generate data
            x, y = generate_classification_data(16, 512, 20)
            
            # Forward pass with specific policy
            with mp_manager.autocast():
                predictions = model(x)
                loss = compute_cross_entropy_loss(predictions, y)
            
            # Backward pass
            success = mp_manager.backward_and_step(loss, optimizer, model)
            
            if success:
                losses.append(loss.data[0])
        
        training_time = time.time() - start_time
        
        # Store results
        stats = mp_manager.get_statistics()
        results[policy.value] = {
            "training_time": training_time,
            "final_loss": losses[-1] if losses else float('inf'),
            "success_rate": stats['success_rate'],
            "avg_loss": np.mean(losses) if losses else float('inf'),
        }
        
        print(f"  ⏱️  Time: {training_time:.2f}s")
        print(f"  📉 Final loss: {results[policy.value]['final_loss']:.4f}")
        print(f"  🎯 Success rate: {results[policy.value]['success_rate']:.1%}")
    
    # Compare results
    print(f"\n📊 Policy Comparison:")
    print("-" * 40)
    
    fastest_policy = min(results.keys(), key=lambda k: results[k]['training_time'])
    most_stable = max(results.keys(), key=lambda k: results[k]['success_rate'])
    best_convergence = min(results.keys(), key=lambda k: results[k]['final_loss'])
    
    print(f"⚡ Fastest: {fastest_policy}")
    print(f"🛡️  Most stable: {most_stable}")
    print(f"🎯 Best convergence: {best_convergence}")


def example_4_recommended_configurations():
    """Example 4: Using recommended configurations for different scenarios."""
    print("\n" + "="*60)
    print("💡 Example 4: Recommended Configurations")
    print("="*60)
    
    scenarios = [
        ("small transformer", "transformer", "small", "stable"),
        ("large CNN", "cnn", "large", "normal"),
        ("unstable RNN", "rnn", "medium", "unstable"),
    ]
    
    for scenario_name, model_type, model_size, stability in scenarios:
        print(f"\n🎯 Scenario: {scenario_name}")
        print("-" * 30)
        
        # Get recommended configuration
        config = get_recommended_precision_config(
            model_type=model_type,
            model_size=model_size,
            training_stability=stability
        )
        
        print(f"  📦 Model type: {model_type}")
        print(f"  📏 Model size: {model_size}")
        print(f"  🌊 Stability: {stability}")
        print(f"  ⚙️  Policy: {config.autocast_config.policy.value}")
        print(f"  📊 Loss scale: {config.loss_scale}")
        print(f"  📈 Growth interval: {config.growth_interval}")
        
        # Create model and test configuration
        if model_type == "transformer":
            model = TransformerLikeModel(vocab_size=500, d_model=256, num_layers=2)
        else:
            model = SimpleMLPModel(input_size=256, hidden_size=128, output_size=10)
        
        # Create training context
        optimizer = Adam(model.parameters(), lr=0.001)
        manager, amp_optimizer, autocast_context = create_training_context(
            model, optimizer, config
        )
        
        # Quick validation training
        num_steps = 20
        successes = 0
        
        for step in range(num_steps):
            x, y = generate_classification_data(8, 256, 10)
            
            with autocast_context:
                predictions = model(x)
                loss = compute_cross_entropy_loss(predictions, y)
            
            if hasattr(amp_optimizer, 'backward'):
                amp_optimizer.zero_grad()
                amp_optimizer.backward(loss)
                success = amp_optimizer.step()
            else:
                success = manager.backward_and_step(loss, amp_optimizer, model)
            
            if success:
                successes += 1
        
        success_rate = successes / num_steps
        print(f"  ✅ Validation success rate: {success_rate:.1%}")
        
        if success_rate < 0.8:
            print(f"  ⚠️  Low success rate - consider more conservative settings")
        else:
            print(f"  🎉 Good stability with recommended settings")


def example_5_troubleshooting_and_monitoring():
    """Example 5: Troubleshooting and monitoring mixed precision training."""
    print("\n" + "="*60)
    print("🔧 Example 5: Troubleshooting and Monitoring")
    print("="*60)
    
    if not ADVANCED_AMP_AVAILABLE:
        print("❌ Advanced AMP components not available for detailed monitoring")
        return
    
    # Create a model that might have stability issues
    model = TransformerLikeModel(vocab_size=2000, d_model=1024, num_layers=8)
    
    # Create scaler with detailed monitoring
    scaler_config = ScalerConfig(
        init_scale=65536.0,
        strategy=ScalingStrategy.DYNAMIC,
        stability_check_interval=10,  # Frequent checks
        gradient_clip_threshold=5.0,  # Allow larger gradients initially
    )
    
    optimizer = create_amp_adam(model.parameters(), lr=0.001, scaler_config=scaler_config)
    
    print(f"📊 Large model parameters: {sum(p.data.size for p in model.parameters()):,}")
    print(f"🔍 Monitoring configuration:")
    print(f"  - Stability checks every {scaler_config.stability_check_interval} steps")
    print(f"  - Gradient clipping at {scaler_config.gradient_clip_threshold}")
    print(f"  - Dynamic scaling strategy")
    
    # Training with monitoring
    num_steps = 100
    monitoring_interval = 20
    
    for step in range(num_steps):
        # Generate challenging data (larger sequences)
        batch_size, seq_len = 4, 256
        x_data = np.random.randint(0, 2000, (batch_size, seq_len)).astype(np.float32)
        x = Tensor(x_data, requires_grad=False)
        
        y_data = np.random.randint(0, 2000, (batch_size, seq_len)).astype(np.float32)
        y = Tensor(y_data, requires_grad=False)
        
        # Forward pass
        with autocast():
            predictions = model(x)
            loss = compute_cross_entropy_loss(
                predictions.data.reshape(-1, 2000),
                np.eye(2000)[y.data.astype(int).flatten()]
            )
        
        # Backward pass with monitoring
        optimizer.zero_grad()
        optimizer.backward(loss)
        success = optimizer.step()
        
        # Detailed monitoring
        if step % monitoring_interval == 0:
            stats = optimizer.get_statistics()
            
            print(f"\n📈 Step {step} Monitoring Report:")
            print(f"  📉 Loss: {loss.data[0]:.4f}")
            print(f"  📏 Current scale: {stats.get('current_scale', 0):,.0f}")
            print(f"  🎯 Success rate: {stats.get('success_rate', 0):.1%}")
            print(f"  ⚠️  Skip rate: {stats.get('skip_rate', 0):.1%}")
            print(f"  ✂️  Clip rate: {stats.get('clip_rate', 0):.1%}")
            
            # Gradient analysis
            if 'avg_gradient_norm' in stats:
                print(f"  📊 Avg grad norm: {stats['avg_gradient_norm']:.4f}")
                print(f"  📊 Max grad norm: {stats.get('max_gradient_norm', 0):.4f}")
            
            # Warning checks
            if stats.get('skip_rate', 0) > 0.1:
                print(f"  🚨 HIGH OVERFLOW RATE! Consider:")
                print(f"     - Reducing learning rate")
                print(f"     - Using more conservative scaling")
                print(f"     - Reducing model complexity")
            
            if stats.get('clip_rate', 0) > 0.5:
                print(f"  ⚠️  High gradient clipping rate")
                print(f"     - Gradients may be too large")
                print(f"     - Consider gradient norm monitoring")
    
    # Final diagnosis
    final_stats = optimizer.get_statistics()
    print(f"\n🏁 Final Training Diagnosis:")
    print(f"  🎯 Overall success rate: {final_stats['success_rate']:.1%}")
    print(f"  ⚠️  Total overflows: {final_stats.get('skipped_steps', 0)}")
    print(f"  ✂️  Total clips: {final_stats.get('clipped_steps', 0)}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if final_stats['success_rate'] > 0.95:
        print(f"  ✅ Excellent stability - training is working well")
        print(f"  💡 Consider slightly more aggressive settings for speed")
    elif final_stats['success_rate'] > 0.85:
        print(f"  ✅ Good stability - minor adjustments may help")
        print(f"  💡 Monitor for consistent performance")
    else:
        print(f"  🚨 Poor stability - requires attention")
        print(f"  💡 Reduce learning rate by 2-5x")
        print(f"  💡 Use more conservative scaling strategy")
        print(f"  💡 Consider reducing model size or batch size")


def main():
    """Run all mixed precision training examples."""
    print("🚀 Neural Forge Mixed Precision Training Examples")
    print("=" * 60)
    print("This script demonstrates various mixed precision training scenarios")
    print("and best practices for Neural Forge.")
    
    try:
        # Run examples
        example_1_basic_mixed_precision()
        example_2_advanced_scaler_configuration()
        example_3_different_autocast_policies()
        example_4_recommended_configurations()
        example_5_troubleshooting_and_monitoring()
        
        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("="*60)
        
        print("\n💡 Key Takeaways:")
        print("  1. Start with recommended configurations for your model type")
        print("  2. Monitor success rates and adjust scaling strategy as needed")
        print("  3. Use conservative policies for unstable training")
        print("  4. Enable gradient clipping for large models")
        print("  5. Watch for overflow patterns and adjust accordingly")
        
        if not ADVANCED_AMP_AVAILABLE:
            print("\n⚠️  Note: Some advanced features were not available.")
            print("   Install the complete optimization package for full functionality.")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()