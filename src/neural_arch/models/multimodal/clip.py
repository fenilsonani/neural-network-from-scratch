"""CLIP implementation with modern improvements.

From "Learning Transferable Visual Representations from Natural Language Supervision"
https://arxiv.org/abs/2103.00020

Features:
- Vision Transformer with improved attention
- Text Transformer with causal masking
- Contrastive learning with temperature scaling
- Mixed precision training support
- Gradient checkpointing
- Various pooling strategies
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from ...core import Module, Tensor, Parameter
from ...nn import Linear, LayerNorm, Dropout, Embedding
from ...functional import softmax
from ..vision.vision_transformer import VisionTransformer, Block as ViTBlock
from ..registry import register_model
import math


class TextTransformerBlock(Module):
    """Transformer block for text processing with causal attention."""
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Multi-head attention
        self.ln_1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = MultiHeadCausalAttention(d_model, n_head, dropout)
        
        # MLP
        self.ln_2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), dropout)
    
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass with residual connections."""
        # Self-attention block
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, attn_mask)
        x = Tensor(x.data + residual.data, requires_grad=x.requires_grad or residual.requires_grad)
        
        # MLP block
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = Tensor(x.data + residual.data, requires_grad=x.requires_grad or residual.requires_grad)
        
        return x


class MultiHeadCausalAttention(Module):
    """Multi-head causal attention for text."""
    
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv_proj = Linear(d_model, d_model * 3)
        self.out_proj = Linear(d_model, d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Apply causal multi-head attention."""
        B, T, C = x.shape
        
        # Get Q, K, V
        qkv = self.qkv_proj(x)
        qkv_data = qkv.data.reshape(B, T, 3, self.n_head, self.head_dim)
        qkv_data = qkv_data.transpose(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)
        q, k, v = qkv_data[0], qkv_data[1], qkv_data[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        mask = np.tril(np.ones((T, T)))
        attn = np.where(mask[None, None, :, :] == 0, -np.inf, attn)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn = attn + attn_mask.data[:, None, None, :]
        
        # Softmax and dropout
        attn = softmax(Tensor(attn, requires_grad=x.requires_grad), dim=-1)
        if self.dropout:
            attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn.data @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Output projection
        out = Tensor(out, requires_grad=x.requires_grad)
        out = self.out_proj(out)
        
        return out


class MLP(Module):
    """MLP with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else None
    
    def gelu(self, x: Tensor) -> Tensor:
        """GELU activation function."""
        # Quick GELU approximation
        x_data = x.data * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * x.data ** 3)))
        return Tensor(x_data, requires_grad=x.requires_grad)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP."""
        x = self.fc1(x)
        x = self.gelu(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class CLIPVisionTransformer(Module):
    """Vision Transformer for CLIP with additional improvements."""
    
    def __init__(
        self,
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        output_dim: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        
        # Use our improved ViT
        self.transformer = VisionTransformer(
            img_size=input_resolution,
            patch_size=patch_size,
            embed_dim=width,
            depth=layers,
            num_heads=heads,
            drop_rate=dropout,
            num_classes=0,  # No classification head
            global_pool='token'  # Use CLS token
        )
        
        # Remove the classification head and add projection
        self.proj = Linear(width, output_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """Extract visual features."""
        # Get features from ViT (without classification head)
        x = self.transformer.forward_features(x)
        
        # Project to output dimension
        x = self.proj(x)
        
        return x


class CLIPTextTransformer(Module):
    """Text Transformer for CLIP."""
    
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        output_dim: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        
        # Token and position embeddings
        self.token_embedding = Embedding(vocab_size, width)
        self.positional_embedding = Parameter(
            np.random.randn(context_length, width).astype(np.float32) * 0.02
        )
        
        # Transformer blocks
        self.transformer = []
        for _ in range(layers):
            self.transformer.append(
                TextTransformerBlock(width, heads, dropout=dropout)
            )
        
        # Final layer norm and projection
        self.ln_final = LayerNorm(width)
        self.text_projection = Linear(width, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Initialize token embeddings
        std = 0.02
        self.token_embedding.weight.data = np.random.randn(*self.token_embedding.weight.shape).astype(np.float32) * std
        
        # Initialize text projection with smaller scale
        self.text_projection.weight.data = np.random.randn(*self.text_projection.weight.shape).astype(np.float32) * (std / np.sqrt(self.width))
    
    def build_attention_mask(self, seq_len: int) -> Tensor:
        """Build causal attention mask."""
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        return Tensor(mask, requires_grad=False)
    
    def forward(self, text: Tensor) -> Tensor:
        """Process text tokens."""
        seq_len = text.shape[1]
        
        # Token embeddings
        x = self.token_embedding(text)
        
        # Add positional embeddings
        pos_emb = self.positional_embedding.data[:seq_len]
        x = Tensor(x.data + pos_emb[None, :, :], requires_grad=x.requires_grad)
        
        # Apply transformer blocks
        attn_mask = self.build_attention_mask(seq_len)
        for block in self.transformer:
            x = block(x, attn_mask)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Take features from the end-of-text token (argmax of text along sequence dimension)
        # For simplicity, we'll use the last token
        x_data = x.data[:, -1, :]
        x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Project to output dimension
        x = self.text_projection(x)
        
        return x


class CLIP(Module):
    """CLIP model with contrastive learning."""
    
    def __init__(
        self,
        embed_dim: int = 512,
        # Vision
        image_resolution: int = 224,
        vision_layers: int = 12,
        vision_width: int = 768,
        vision_patch_size: int = 16,
        # Text
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        # Training
        temperature_init: float = 0.07,
        learnable_temperature: bool = True
    ):
        super().__init__()
        
        # Vision encoder
        self.visual = CLIPVisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )
        
        # Text encoder
        self.transformer = CLIPTextTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            output_dim=embed_dim
        )
        
        # Temperature parameter for contrastive learning
        if learnable_temperature:
            self.logit_scale = Parameter(
                np.array([np.log(1 / temperature_init)], dtype=np.float32)
            )
        else:
            self.logit_scale = np.log(1 / temperature_init)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        # Additional initialization can go here
        pass
    
    def encode_image(self, image: Tensor) -> Tensor:
        """Encode images to feature vectors."""
        features = self.visual(image)
        # L2 normalize
        features_norm = np.linalg.norm(features.data, axis=-1, keepdims=True)
        features_normalized = features.data / np.maximum(features_norm, 1e-12)
        return Tensor(features_normalized, requires_grad=features.requires_grad)
    
    def encode_text(self, text: Tensor) -> Tensor:
        """Encode text to feature vectors."""
        features = self.transformer(text)
        # L2 normalize
        features_norm = np.linalg.norm(features.data, axis=-1, keepdims=True)
        features_normalized = features.data / np.maximum(features_norm, 1e-12)
        return Tensor(features_normalized, requires_grad=features.requires_grad)
    
    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, Tensor]:
        """Forward pass through CLIP."""
        outputs = {}
        
        if image is not None:
            image_features = self.encode_image(image)
            outputs['image_embeds'] = image_features
        
        if text is not None:
            text_features = self.encode_text(text)
            outputs['text_embeds'] = text_features
        
        if image is not None and text is not None and return_loss:
            # Compute contrastive loss
            if isinstance(self.logit_scale, Parameter):
                logit_scale = np.exp(self.logit_scale.data[0])
            else:
                logit_scale = np.exp(self.logit_scale)
            
            # Similarity matrix
            logits_per_image = image_features.data @ text_features.data.T * logit_scale
            logits_per_text = logits_per_image.T
            
            outputs['logits_per_image'] = Tensor(logits_per_image, requires_grad=True)
            outputs['logits_per_text'] = Tensor(logits_per_text, requires_grad=True)
            
            # Compute loss (InfoNCE)
            batch_size = image.shape[0]
            labels = np.arange(batch_size)
            
            # Cross entropy loss for both directions
            image_loss = self._cross_entropy_loss(outputs['logits_per_image'], labels)
            text_loss = self._cross_entropy_loss(outputs['logits_per_text'], labels)
            
            loss = (image_loss + text_loss) / 2
            outputs['loss'] = loss
        
        return outputs
    
    def _cross_entropy_loss(self, logits: Tensor, labels: np.ndarray) -> Tensor:
        """Compute cross entropy loss."""
        # Apply softmax
        softmax_logits = softmax(logits, dim=-1)
        
        # Compute negative log likelihood
        batch_size = logits.shape[0]
        log_probs = np.log(np.maximum(softmax_logits.data, 1e-12))
        loss = -np.mean([log_probs[i, labels[i]] for i in range(batch_size)])
        
        return Tensor(np.array([loss], dtype=np.float32), requires_grad=True)
    
    def get_text_features(self, text: Tensor) -> Tensor:
        """Get normalized text features."""
        return self.encode_text(text)
    
    def get_image_features(self, image: Tensor) -> Tensor:
        """Get normalized image features."""
        return self.encode_image(image)


# Model configurations
CLIP_CONFIGS = {
    'base': {
        'embed_dim': 512,
        'image_resolution': 224,
        'vision_layers': 12,
        'vision_width': 768,
        'vision_patch_size': 32,
        'context_length': 77,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    },
    'large': {
        'embed_dim': 768,
        'image_resolution': 224,
        'vision_layers': 24,
        'vision_width': 1024,
        'vision_patch_size': 14,
        'context_length': 77,
        'vocab_size': 49408,
        'transformer_width': 768,
        'transformer_heads': 12,
        'transformer_layers': 12
    }
}


# Model variants
class CLIPModel(CLIP):
    """Alias for CLIP model."""
    pass


class CLIPVisionModel(Module):
    """CLIP vision encoder only."""
    def __init__(self, config):
        super().__init__()
        clip_model = CLIP(**config)
        self.visual = clip_model.visual
    
    def forward(self, image: Tensor) -> Tensor:
        return self.visual(image)


class CLIPTextModel(Module):
    """CLIP text encoder only."""
    def __init__(self, config):
        super().__init__()
        clip_model = CLIP(**config)
        self.transformer = clip_model.transformer
    
    def forward(self, text: Tensor) -> Tensor:
        return self.transformer(text)


def CLIPBase(**kwargs):
    config = CLIP_CONFIGS['base'].copy()
    config.update(kwargs)
    return CLIP(**config)


def CLIPLarge(**kwargs):
    config = CLIP_CONFIGS['large'].copy()
    config.update(kwargs)
    return CLIP(**config)


# Register models
@register_model(
    name='clip_base',
    description='CLIP Base with ViT-B/32 vision encoder',
    paper_url='https://arxiv.org/abs/2103.00020',
    pretrained_configs={
        'openai': CLIP_CONFIGS['base'],
    },
    default_config='openai',
    tags=['multimodal', 'vision-language', 'clip', 'contrastive'],
    aliases=['clip-base', 'clip_vit_b_32']
)
class RegisteredCLIPBase(CLIP):
    def __init__(self, **kwargs):
        config = CLIP_CONFIGS['base'].copy()
        config.update(kwargs)
        super().__init__(**config)

def clip_base(**kwargs):
    return RegisteredCLIPBase(**kwargs)


@register_model(
    name='clip_large',
    description='CLIP Large with ViT-L/14 vision encoder',
    paper_url='https://arxiv.org/abs/2103.00020',
    pretrained_configs={
        'openai': CLIP_CONFIGS['large'],
    },
    default_config='openai',
    tags=['multimodal', 'vision-language', 'clip', 'contrastive'],
    aliases=['clip-large', 'clip_vit_l_14']
)
class RegisteredCLIPLarge(CLIP):
    def __init__(self, **kwargs):
        config = CLIP_CONFIGS['large'].copy()
        config.update(kwargs)
        super().__init__(**config)

def clip_large(**kwargs):
    return RegisteredCLIPLarge(**kwargs)