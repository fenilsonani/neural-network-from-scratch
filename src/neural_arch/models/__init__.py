"""Advanced Model Zoo for Neural Architecture.

State-of-the-art implementations of vision, language, and multimodal models.
"""

from .registry import ModelRegistry, register_model, get_model, list_models
from .utils import load_pretrained_weights, download_weights, ModelCard

# Vision Models
from .vision import (
    ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
    resnet18, resnet34, resnet50, resnet101, resnet152,
    EfficientNet, EfficientNetB0, EfficientNetB1, EfficientNetB2,
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    VisionTransformer, ViT_B_16, ViT_L_16, ViT_H_14,
    vit_b_16, vit_l_16, vit_h_14,
    ConvNeXt, ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase,
    convnext_tiny, convnext_small, convnext_base
)

# Language Models  
from .language import (
    GPT2, GPT2Small, GPT2Medium, GPT2Large,
    gpt2_small, gpt2_medium, gpt2_large, gpt2_xl,
    BERT, BERTBase, BERTLarge,
    bert_base, bert_large,
    T5, T5Small, T5Base, T5Large,
    t5_small, t5_base, t5_large,
    RoBERTa, RoBERTaBase, RoBERTaLarge,
    roberta_base, roberta_large
)

# Multimodal Models
from .multimodal import (
    CLIP, CLIPBase, CLIPLarge,
    clip_base, clip_large,
    ALIGN, ALIGNBase,
    align_base,
    Flamingo, FlamingoBase,
    flamingo_base
)

__all__ = [
    # Registry
    'ModelRegistry', 'register_model', 'get_model', 'list_models',
    'load_pretrained_weights', 'download_weights', 'ModelCard',
    
    # Vision
    'ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'EfficientNet', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
    'VisionTransformer', 'ViT_B_16', 'ViT_L_16', 'ViT_H_14',
    'vit_b_16', 'vit_l_16', 'vit_h_14',
    'ConvNeXt', 'ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtBase',
    'convnext_tiny', 'convnext_small', 'convnext_base',
    
    # Language
    'GPT2', 'GPT2Small', 'GPT2Medium', 'GPT2Large',
    'gpt2_small', 'gpt2_medium', 'gpt2_large', 'gpt2_xl',
    'BERT', 'BERTBase', 'BERTLarge',
    'bert_base', 'bert_large',
    'T5', 'T5Small', 'T5Base', 'T5Large',
    't5_small', 't5_base', 't5_large',
    'RoBERTa', 'RoBERTaBase', 'RoBERTaLarge',
    'roberta_base', 'roberta_large',
    
    # Multimodal
    'CLIP', 'CLIPBase', 'CLIPLarge',
    'clip_base', 'clip_large',
    'ALIGN', 'ALIGNBase',
    'align_base',
    'Flamingo', 'FlamingoBase',
    'flamingo_base'
]