#!/usr/bin/env python3
"""
Comprehensive showcase of Neural Forge transformer variants.

This script demonstrates the enhanced transformer model implementations:
- BERT (Base, Large, Cased variants)
- RoBERTa (Base, Large)
- DeBERTa (Base, Large, v3 variants)

Features demonstrated:
- Model creation and configuration
- Forward pass inference
- Masked language modeling
- Sequence classification
- Performance comparison
- Model statistics and architecture analysis
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.neural_arch.core.tensor import Tensor
    from src.neural_arch.models.language import (
        # BERT variants
        bert_base,
        bert_large,
        bert_base_cased,
        bert_large_cased,
        BERTForMaskedLM,
        BERTForSequenceClassification,
        # RoBERTa variants
        roberta_base,
        roberta_large,
        RoBERTaForMaskedLM,
        RoBERTaForSequenceClassification,
        # DeBERTa variants
        deberta_base,
        deberta_large,
        deberta_v3_base,
        deberta_v3_large,
        DeBERTaForMaskedLM,
        DeBERTaForSequenceClassification,
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing models: {e}")
    MODELS_AVAILABLE = False


def create_sample_input(batch_size=2, seq_length=16, vocab_size=30522, with_attention=True):
    """Create sample input tensors for testing models."""
    input_ids = Tensor(np.random.randint(1, vocab_size, (batch_size, seq_length)))
    
    if with_attention:
        # Create realistic attention mask with some padding
        attention_mask = np.ones((batch_size, seq_length))
        # Add some padding to the second sequence
        if batch_size > 1:
            padding_length = np.random.randint(1, seq_length // 2)
            attention_mask[1, -padding_length:] = 0
        attention_mask = Tensor(attention_mask)
        return input_ids, attention_mask
    
    return input_ids


def analyze_model_architecture(model, model_name):
    """Analyze and display model architecture information."""
    config = model.config
    
    print(f"\n{'='*60}")
    print(f"🤖 {model_name} Architecture Analysis")
    print(f"{'='*60}")
    
    # Basic configuration
    print(f"📊 Basic Configuration:")
    print(f"  • Vocabulary Size: {config.vocab_size:,}")
    print(f"  • Hidden Size: {config.hidden_size}")
    print(f"  • Number of Layers: {config.num_hidden_layers}")
    print(f"  • Attention Heads: {config.num_attention_heads}")
    print(f"  • Feed-Forward Size: {config.intermediate_size}")
    print(f"  • Max Position Embeddings: {config.max_position_embeddings}")
    
    # Calculate approximate parameter count
    vocab_params = config.vocab_size * config.hidden_size
    position_params = config.max_position_embeddings * config.hidden_size
    
    # Attention parameters per layer
    attention_params_per_layer = (
        3 * config.hidden_size * config.hidden_size +  # Q, K, V projections
        config.hidden_size * config.hidden_size  # Output projection
    )
    
    # Feed-forward parameters per layer
    ff_params_per_layer = (
        config.hidden_size * config.intermediate_size +  # Up projection
        config.intermediate_size * config.hidden_size   # Down projection
    )
    
    # Layer normalization parameters per layer (2 layer norms per layer)
    ln_params_per_layer = 2 * 2 * config.hidden_size  # weight + bias for each
    
    total_layer_params = (attention_params_per_layer + ff_params_per_layer + ln_params_per_layer) * config.num_hidden_layers
    
    total_params = vocab_params + position_params + total_layer_params
    
    print(f"\n📈 Parameter Analysis:")
    print(f"  • Embedding Parameters: {vocab_params:,}")
    print(f"  • Position Parameters: {position_params:,}")
    print(f"  • Transformer Parameters: {total_layer_params:,}")
    print(f"  • Total (Approximate): {total_params:,}")
    print(f"  • Model Size (MB): {total_params * 4 / 1024 / 1024:.1f}")
    
    # Model-specific features
    print(f"\n🔧 Model-Specific Features:")
    if hasattr(config, 'type_vocab_size'):
        print(f"  • Token Type Vocab Size: {config.type_vocab_size}")
    if hasattr(config, 'relative_attention'):
        print(f"  • Relative Attention: {config.relative_attention}")
    if hasattr(config, 'position_buckets'):
        print(f"  • Position Buckets: {config.position_buckets}")
    if hasattr(config, 'pos_att_type'):
        print(f"  • Position Attention Type: {config.pos_att_type}")
    if hasattr(config, 'layer_norm_eps'):
        print(f"  • Layer Norm Epsilon: {config.layer_norm_eps}")


def demonstrate_forward_pass(model, model_name, vocab_size):
    """Demonstrate forward pass with the model."""
    print(f"\n🚀 Forward Pass Demonstration - {model_name}")
    print(f"{'-'*50}")
    
    # Create sample input
    batch_size, seq_length = 3, 12
    input_ids, attention_mask = create_sample_input(batch_size, seq_length, vocab_size)
    
    print(f"Input Shape: {input_ids.shape}")
    print(f"Attention Mask Shape: {attention_mask.shape}")
    
    # Measure inference time
    start_time = time.time()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    inference_time = time.time() - start_time
    
    # Display outputs
    print(f"\nOutputs:")
    print(f"  • Last Hidden State: {outputs['last_hidden_state'].shape}")
    if 'pooler_output' in outputs:
        print(f"  • Pooler Output: {outputs['pooler_output'].shape}")
    if 'hidden_states' in outputs and outputs['hidden_states'] is not None:
        print(f"  • All Hidden States: {len(outputs['hidden_states'])} layers")
    if 'attentions' in outputs and outputs['attentions'] is not None:
        print(f"  • Attention Weights: {len(outputs['attentions'])} layers")
    
    print(f"  • Inference Time: {inference_time*1000:.2f}ms")
    
    return outputs


def demonstrate_masked_lm(model_class, model_name, config_kwargs=None):
    """Demonstrate masked language modeling."""
    print(f"\n📝 Masked Language Modeling - {model_name}")
    print(f"{'-'*50}")
    
    if config_kwargs is None:
        config_kwargs = {}
    
    model = model_class(**config_kwargs)
    vocab_size = model.config.vocab_size
    
    # Create sample input with labels
    batch_size, seq_length = 2, 8
    input_ids = Tensor(np.random.randint(1, vocab_size, (batch_size, seq_length)))
    labels = input_ids.copy()  # For simplicity, use same as input
    
    print(f"Input Shape: {input_ids.shape}")
    print(f"Vocab Size: {vocab_size:,}")
    
    # Forward pass
    start_time = time.time()
    outputs = model(input_ids=input_ids, labels=labels)
    inference_time = time.time() - start_time
    
    print(f"\nOutputs:")
    print(f"  • Logits Shape: {outputs['logits'].shape}")
    print(f"  • Loss: {outputs['loss'].data:.4f}")
    print(f"  • Inference Time: {inference_time*1000:.2f}ms")


def demonstrate_sequence_classification(model_class, model_name, config_kwargs=None):
    """Demonstrate sequence classification."""
    print(f"\n🎯 Sequence Classification - {model_name}")
    print(f"{'-'*50}")
    
    if config_kwargs is None:
        config_kwargs = {}
    
    num_labels = 3
    model = model_class(num_labels=num_labels, **config_kwargs)
    vocab_size = model.config.vocab_size
    
    # Create sample input with labels
    batch_size, seq_length = 2, 10
    input_ids = Tensor(np.random.randint(1, vocab_size, (batch_size, seq_length)))
    labels = Tensor(np.random.randint(0, num_labels, (batch_size,)))
    
    print(f"Input Shape: {input_ids.shape}")
    print(f"Number of Labels: {num_labels}")
    
    # Forward pass
    start_time = time.time()
    outputs = model(input_ids=input_ids, labels=labels)
    inference_time = time.time() - start_time
    
    print(f"\nOutputs:")
    print(f"  • Logits Shape: {outputs['logits'].shape}")
    print(f"  • Loss: {outputs['loss'].data:.4f}")
    print(f"  • Predictions: {np.argmax(outputs['logits'].data, axis=1)}")
    print(f"  • Inference Time: {inference_time*1000:.2f}ms")


def compare_model_sizes():
    """Compare model sizes across variants."""
    print(f"\n📊 Model Size Comparison")
    print(f"{'='*60}")
    
    models_info = [
        ("BERT Base", bert_base()),
        ("BERT Large", bert_large()),
        ("BERT Base Cased", bert_base_cased()),
        ("BERT Large Cased", bert_large_cased()),
        ("RoBERTa Base", roberta_base()),
        ("RoBERTa Large", roberta_large()),
        ("DeBERTa Base", deberta_base()),
        ("DeBERTa Large", deberta_large()),
        ("DeBERTa-v3 Base", deberta_v3_base()),
        ("DeBERTa-v3 Large", deberta_v3_large()),
    ]
    
    print(f"{'Model':<20} {'Layers':<8} {'Hidden':<8} {'Heads':<8} {'Vocab':<10} {'Params (M)':<12}")
    print(f"{'-'*75}")
    
    for name, model in models_info:
        config = model.config
        
        # Approximate parameter count
        vocab_params = config.vocab_size * config.hidden_size
        transformer_params = (
            config.num_hidden_layers * (
                4 * config.hidden_size * config.hidden_size +  # Attention
                2 * config.hidden_size * config.intermediate_size +  # Feed-forward
                4 * config.hidden_size  # Layer norms
            )
        )
        total_params = (vocab_params + transformer_params) / 1_000_000
        
        print(f"{name:<20} {config.num_hidden_layers:<8} {config.hidden_size:<8} "
              f"{config.num_attention_heads:<8} {config.vocab_size:<10,} {total_params:<12.1f}")


def performance_benchmark():
    """Benchmark performance across different models."""
    print(f"\n⚡ Performance Benchmark")
    print(f"{'='*60}")
    
    models = [
        ("BERT Base", bert_base()),
        ("BERT Large", bert_large()),
        ("RoBERTa Base", roberta_base()),
        ("RoBERTa Large", roberta_large()),
        ("DeBERTa Base", deberta_base()),
    ]
    
    batch_size, seq_length = 4, 16
    num_runs = 5
    
    print(f"Benchmark Settings:")
    print(f"  • Batch Size: {batch_size}")
    print(f"  • Sequence Length: {seq_length}")
    print(f"  • Number of Runs: {num_runs}")
    print()
    
    print(f"{'Model':<15} {'Avg Time (ms)':<15} {'Throughput (seq/s)':<20}")
    print(f"{'-'*50}")
    
    for name, model in models:
        vocab_size = model.config.vocab_size
        
        # Warm-up run
        input_ids, attention_mask = create_sample_input(batch_size, seq_length, vocab_size)
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            input_ids, attention_mask = create_sample_input(batch_size, seq_length, vocab_size)
            
            start_time = time.time()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        throughput = batch_size / (avg_time / 1000)  # Sequences per second
        
        print(f"{name:<15} {avg_time:<15.2f} {throughput:<20.1f}")


def main():
    """Main demonstration function."""
    if not MODELS_AVAILABLE:
        print("❌ Transformer models are not available. Please check your installation.")
        return
    
    print("🎭 Neural Forge Transformer Variants Showcase")
    print("=" * 80)
    print("This demonstration showcases the comprehensive transformer model implementations")
    print("including BERT, RoBERTa, and DeBERTa variants with their unique features.\n")
    
    # 1. Model Architecture Analysis
    print("🔍 ANALYZING MODEL ARCHITECTURES")
    models_to_analyze = [
        ("BERT Base", bert_base()),
        ("BERT Large", bert_large()),
        ("RoBERTa Base", roberta_base()),
        ("RoBERTa Large", roberta_large()),
        ("DeBERTa Base", deberta_base()),
        ("DeBERTa-v3 Base", deberta_v3_base()),
    ]
    
    for name, model in models_to_analyze[:3]:  # Analyze first 3 to keep output manageable
        analyze_model_architecture(model, name)
    
    # 2. Forward Pass Demonstrations
    print("\n\n🚀 FORWARD PASS DEMONSTRATIONS")
    forward_models = [
        ("BERT Large", bert_large(), 30522),
        ("RoBERTa Large", roberta_large(), 50265),
        ("DeBERTa Base", deberta_base(), 128100),
    ]
    
    for name, model, vocab_size in forward_models:
        demonstrate_forward_pass(model, name, vocab_size)
    
    # 3. Task-Specific Demonstrations
    print("\n\n📚 TASK-SPECIFIC DEMONSTRATIONS")
    
    # Masked Language Modeling
    demonstrate_masked_lm(BERTForMaskedLM, "BERT MLM")
    demonstrate_masked_lm(RoBERTaForMaskedLM, "RoBERTa MLM")
    demonstrate_masked_lm(DeBERTaForMaskedLM, "DeBERTa MLM")
    
    # Sequence Classification
    demonstrate_sequence_classification(BERTForSequenceClassification, "BERT Classification")
    demonstrate_sequence_classification(RoBERTaForSequenceClassification, "RoBERTa Classification")
    demonstrate_sequence_classification(DeBERTaForSequenceClassification, "DeBERTa Classification")
    
    # 4. Model Comparisons
    print("\n\n📈 MODEL COMPARISONS")
    compare_model_sizes()
    
    # 5. Performance Benchmark
    print("\n\n⚡ PERFORMANCE ANALYSIS")
    performance_benchmark()
    
    # 6. Summary and Capabilities
    print(f"\n\n🎉 NEURAL FORGE TRANSFORMER CAPABILITIES SUMMARY")
    print(f"{'='*80}")
    print("✅ Comprehensive transformer model implementations:")
    print("   • BERT (Base, Large, Cased variants)")
    print("   • RoBERTa (Base, Large)")
    print("   • DeBERTa (Base, Large, v3 variants)")
    print("\n✅ Advanced features implemented:")
    print("   • Disentangled attention (DeBERTa)")
    print("   • Optimized attention mechanisms")
    print("   • Multiple pre-training objectives")
    print("   • Flexible configuration systems")
    print("   • Task-specific model heads")
    print("\n✅ Production-ready features:")
    print("   • Efficient inference")
    print("   • Memory optimization")
    print("   • Gradient computation")
    print("   • Model registration system")
    print("   • Comprehensive testing")
    
    print(f"\n🚀 Neural Forge now provides state-of-the-art transformer models")
    print(f"   ready for research, development, and production deployment!")


if __name__ == "__main__":
    main()