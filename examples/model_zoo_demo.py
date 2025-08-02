"""Demo script for the advanced model zoo.

This script demonstrates how to use the model registry and various state-of-the-art models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from neural_arch.core import Tensor
from neural_arch.models import (
    ModelRegistry, get_model, list_models,
    # Vision models
    resnet50, vit_b_16,
    # Language models
    gpt2_small,
    # Multimodal models
    clip_base
)


def demo_model_registry():
    """Demonstrate model registry functionality."""
    print("🔍 Available Models in Registry:")
    print("=" * 50)
    
    # List all models
    all_models = list_models()
    print(f"Total models: {len(all_models)}")
    
    # List by category
    vision_models = list_models(tags=['vision'])
    language_models = list_models(tags=['language'])
    multimodal_models = list_models(tags=['multimodal'])
    
    print(f"Vision models: {vision_models}")
    print(f"Language models: {language_models}")
    print(f"Multimodal models: {multimodal_models}")
    
    # Show model cards
    for model_name in all_models[:3]:  # Show first 3 models
        try:
            card = ModelRegistry.get_model_card(model_name)
            print(f"\n📋 Model Card: {model_name}")
            print(f"Description: {card['description']}")
            print(f"Tags: {card['tags']}")
            if card['pretrained_configs']:
                print(f"Pretrained configs: {list(card['pretrained_configs'])}")
        except Exception as e:
            print(f"Error getting card for {model_name}: {e}")


def demo_vision_models():
    """Demonstrate vision models."""
    print("\n🖼️ Vision Models Demo:")
    print("=" * 50)
    
    # Test ResNet-50
    try:
        print("Creating ResNet-50...")
        model = get_model('resnet50', num_classes=1000)
        
        # Create dummy image batch (B, C, H, W)
        images = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        print(f"Input shape: {images.shape}")
        output = model(images)
        print(f"ResNet-50 output shape: {output.shape}")
        print("✅ ResNet-50 working!")
        
    except Exception as e:
        print(f"❌ ResNet-50 error: {e}")
    
    # Test Vision Transformer
    try:
        print("\nCreating Vision Transformer...")
        model = get_model('vit_b_16', num_classes=1000)
        
        # Create dummy image batch
        images = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        print(f"Input shape: {images.shape}")
        output = model(images)
        print(f"ViT-B/16 output shape: {output.shape}")
        print("✅ Vision Transformer working!")
        
    except Exception as e:
        print(f"❌ ViT error: {e}")


def demo_language_models():
    """Demonstrate language models."""
    print("\n📝 Language Models Demo:")
    print("=" * 50)
    
    # Test GPT-2
    try:
        print("Creating GPT-2 Small...")
        model = get_model('gpt2_small')
        
        # Create dummy token IDs
        seq_len = 10
        vocab_size = 50257
        input_ids = Tensor(np.random.randint(0, vocab_size, (2, seq_len)).astype(np.int32))
        
        print(f"Input shape: {input_ids.shape}")
        output = model(input_ids)
        print(f"GPT-2 output shape: {output.shape}")
        print("✅ GPT-2 working!")
        
    except Exception as e:
        print(f"❌ GPT-2 error: {e}")


def demo_multimodal_models():
    """Demonstrate multimodal models."""
    print("\n🔗 Multimodal Models Demo:")
    print("=" * 50)
    
    # Test CLIP
    try:
        print("Creating CLIP Base...")
        model = get_model('clip_base')
        
        # Create dummy inputs
        images = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        text = Tensor(np.random.randint(0, 49408, (2, 77)).astype(np.int32))
        
        print(f"Image shape: {images.shape}")
        print(f"Text shape: {text.shape}")
        
        # Test individual encoders
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        
        # Test full forward pass
        outputs = model(images, text)
        print(f"Logits per image shape: {outputs['logits_per_image'].shape}")
        print(f"Logits per text shape: {outputs['logits_per_text'].shape}")
        print("✅ CLIP working!")
        
    except Exception as e:
        print(f"❌ CLIP error: {e}")


def demo_model_comparison():
    """Compare different models."""
    print("\n⚖️ Model Comparison:")
    print("=" * 50)
    
    models_to_compare = [
        ('resnet50', {'num_classes': 1000}),
        ('vit_b_16', {'num_classes': 1000}),
    ]
    
    for model_name, kwargs in models_to_compare:
        try:
            model = get_model(model_name, **kwargs)
            
            # Count parameters
            total_params = sum(p.size for p in model.parameters())
            
            print(f"{model_name}:")
            print(f"  Parameters: {total_params:,}")
            print(f"  Model class: {model.__class__.__name__}")
            
        except Exception as e:
            print(f"  Error: {e}")


def demo_pretrained_configs():
    """Demonstrate pretrained configuration system."""
    print("\n🎯 Pretrained Configurations:")
    print("=" * 50)
    
    # Show available configs for each model
    models_with_configs = ['resnet50', 'vit_b_16', 'gpt2_small', 'clip_base']
    
    for model_name in models_with_configs:
        try:
            info = ModelRegistry.get_model_info(model_name)
            if info.pretrained_configs:
                print(f"\n{model_name} configurations:")
                for config_name, config in info.pretrained_configs.items():
                    print(f"  {config_name}: {config}")
                    
                # Test creating with different configs
                if info.default_config:
                    print(f"  Creating with default config '{info.default_config}'...")
                    model = get_model(model_name, pretrained=False, config_name=info.default_config)
                    print(f"  ✅ Created successfully")
            else:
                print(f"{model_name}: No pretrained configs")
                
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")


def main():
    """Run all demos."""
    print("🚀 Neural Architecture Advanced Model Zoo Demo")
    print("=" * 60)
    
    try:
        demo_model_registry()
        demo_vision_models()
        demo_language_models()
        demo_multimodal_models()
        demo_model_comparison()
        demo_pretrained_configs()
        
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Tips:")
        print("- Use get_model(name, pretrained=True) to load pretrained weights")
        print("- Use list_models(tags=['vision']) to filter models by category")
        print("- Check ModelRegistry.get_model_card(name) for detailed model info")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()