#!/usr/bin/env python3
"""Quick test of the model zoo functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from neural_arch.core import Tensor
from neural_arch.models import (
    ModelRegistry, get_model, list_models,
    resnet50, vit_b_16, gpt2_small, clip_base
)

def main():
    print("🚀 Testing Model Zoo...")
    
    # Test model registry
    print(f"📊 Total registered models: {len(list_models())}")
    
    # Test ResNet
    print("\n🔍 Testing ResNet-50...")
    try:
        model = resnet50(num_classes=10)
        print("✅ ResNet-50 created successfully")
    except Exception as e:
        print(f"❌ ResNet-50 failed: {e}")
    
    # Test ViT
    print("\n🔍 Testing ViT-Base...")
    try:
        model = vit_b_16(num_classes=10)
        print("✅ ViT-Base created successfully")
    except Exception as e:
        print(f"❌ ViT-Base failed: {e}")
    
    # Test GPT-2
    print("\n🔍 Testing GPT-2 Small...")
    try:
        model = gpt2_small()
        print("✅ GPT-2 Small created successfully")
    except Exception as e:
        print(f"❌ GPT-2 Small failed: {e}")
    
    # Test CLIP
    print("\n🔍 Testing CLIP Base...")
    try:
        model = clip_base()
        print("✅ CLIP Base created successfully")
    except Exception as e:
        print(f"❌ CLIP Base failed: {e}")
    
    # Test get_model function
    print("\n🔍 Testing get_model function...")
    try:
        model = get_model('resnet50', num_classes=100)
        print("✅ get_model works successfully")
    except Exception as e:
        print(f"❌ get_model failed: {e}")
        
    print("\n🎉 Model Zoo test completed!")

if __name__ == "__main__":
    main()