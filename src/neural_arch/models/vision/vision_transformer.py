"""Vision Transformer (ViT) implementation.

From "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
https://arxiv.org/abs/2010.11929

Features:
- Patch embedding with positional encoding
- Multi-head self-attention
- MLP blocks with GELU activation
- Stochastic depth
- Layer scale
- Attention dropout
"""

import numpy as np
from typing import Optional, Tuple, List
from ...core import Module, Tensor, Parameter
from ...nn import Linear, LayerNorm, Dropout, MultiHeadAttention
from ...functional import softmax
from ..registry import register_model


class PatchEmbed(Module):
    """Image to Patch Embedding."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Module] = None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Linear projection of flattened patches
        self.proj = Linear(patch_size * patch_size * in_channels, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else None
    
    def forward(self, x: Tensor) -> Tensor:
        """Convert image to patches and embed.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input size {H}x{W} != {self.img_size}x{self.img_size}"
        
        # Extract patches
        patch_h = patch_w = self.patch_size
        num_patches_h = H // patch_h
        num_patches_w = W // patch_w
        
        # Reshape to extract patches
        x_data = x.data.reshape(B, C, num_patches_h, patch_h, num_patches_w, patch_w)
        x_data = x_data.transpose(0, 2, 4, 1, 3, 5)  # B, nh, nw, C, ph, pw
        x_data = x_data.reshape(B, num_patches_h * num_patches_w, -1)  # B, num_patches, patch_dim
        
        # Apply linear projection
        x = Tensor(x_data, requires_grad=x.requires_grad)
        x = self.proj(x)
        
        if self.norm:
            x = self.norm(x)
        
        return x


class Attention(Module):
    """Multi-head self-attention with improvements."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Single matrix for Q, K, V projections
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop) if attn_drop > 0 else None
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop) if proj_drop > 0 else None
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head self-attention.
        
        Args:
            x: Input tensor (B, N, C)
            
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)
        qkv_data = qkv.data.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_data = qkv_data.transpose(2, 0, 3, 1, 4)  # 3, B, num_heads, N, head_dim
        q, k, v = qkv_data[0], qkv_data[1], qkv_data[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = softmax(Tensor(attn, requires_grad=x.requires_grad), dim=-1)
        
        if self.attn_drop:
            attn = self.attn_drop(attn)
        
        # Apply attention to values
        x_data = (attn.data @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Final projection
        x = self.proj(x)
        if self.proj_drop:
            x = self.proj_drop(x)
        
        return x


class MLP(Module):
    """MLP block with GELU activation."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = Linear(in_features, hidden_features)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop) if drop > 0 else None
    
    def gelu(self, x: Tensor) -> Tensor:
        """GELU activation function."""
        # Approximation of GELU
        x_data = x.data * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * x.data ** 3)))
        return Tensor(x_data, requires_grad=x.requires_grad)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP."""
        x = self.fc1(x)
        x = self.gelu(x)
        if self.drop:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop:
            x = self.drop(x)
        return x


class Block(Module):
    """Transformer block with LayerNorm and residual connections."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        layer_scale_init: Optional[float] = None
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path_rate = drop_path
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        
        # Layer scale parameters
        if layer_scale_init is not None:
            self.layer_scale_1 = Parameter(
                np.ones(dim, dtype=np.float32) * layer_scale_init
            )
            self.layer_scale_2 = Parameter(
                np.ones(dim, dtype=np.float32) * layer_scale_init
            )
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None
    
    def drop_path(self, x: Tensor, drop_prob: float = 0.) -> Tensor:
        """Apply stochastic depth."""
        if drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - drop_prob
        # Create binary mask
        shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + np.random.rand(*shape)
        binary_mask = np.floor(random_tensor).astype(x.data.dtype)
        
        output = x.data / keep_prob * binary_mask
        return Tensor(output, requires_grad=x.requires_grad)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transformer block."""
        # Self-attention block
        y = self.attn(self.norm1(x))
        if self.layer_scale_1 is not None:
            y = Tensor(y.data * self.layer_scale_1.data, requires_grad=y.requires_grad)
        x = Tensor(x.data + self.drop_path(y, self.drop_path_rate).data, requires_grad=x.requires_grad)
        
        # MLP block
        y = self.mlp(self.norm2(x))
        if self.layer_scale_2 is not None:
            y = Tensor(y.data * self.layer_scale_2.data, requires_grad=y.requires_grad)
        x = Tensor(x.data + self.drop_path(y, self.drop_path_rate).data, requires_grad=x.requires_grad)
        
        return x


class VisionTransformer(Module):
    """Vision Transformer with modern improvements.
    
    Features:
    - Patch embedding with learnable position embeddings
    - Class token for classification
    - Stochastic depth
    - Layer scale
    - Various pooling strategies
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        layer_scale_init: Optional[float] = None,
        global_pool: str = 'token',  # 'token', 'avg'
        class_token: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1 if class_token else 0
        self.global_pool = global_pool
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        if class_token:
            self.cls_token = Parameter(np.zeros((1, 1, embed_dim), dtype=np.float32))
        
        # Position embeddings
        self.pos_embed = Parameter(
            np.zeros((1, num_patches + self.num_tokens, embed_dim), dtype=np.float32)
        )
        self.pos_drop = Dropout(drop_rate) if drop_rate > 0 else None
        
        # Stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, depth).tolist()
        
        # Transformer blocks
        self.blocks = []
        for i in range(depth):
            block = Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias,
                drop_rate, attn_drop_rate, dpr[i], layer_scale_init
            )
            self.blocks.append(block)
        
        # Final norm
        self.norm = LayerNorm(embed_dim)
        
        # Classification head
        self.head = Linear(embed_dim, num_classes) if num_classes > 0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal distribution."""
        # Initialize patch embedding
        if hasattr(self.patch_embed.proj, 'weight'):
            # Truncated normal initialization
            std = 0.02
            self.patch_embed.proj.weight.data = np.random.randn(*self.patch_embed.proj.weight.shape).astype(np.float32) * std
        
        # Initialize position embeddings
        std = 0.02
        self.pos_embed.data = np.random.randn(*self.pos_embed.shape).astype(np.float32) * std
        
        if hasattr(self, 'cls_token'):
            self.cls_token.data = np.random.randn(*self.cls_token.shape).astype(np.float32) * std
    
    def forward_features(self, x: Tensor) -> Tensor:
        """Extract features from input images."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        if hasattr(self, 'cls_token'):
            cls_tokens = np.broadcast_to(self.cls_token.data, (B, 1, self.embed_dim))
            x_data = np.concatenate([cls_tokens, x.data], axis=1)
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Add position embeddings
        x = Tensor(x.data + self.pos_embed.data[:, :x.shape[1]], requires_grad=x.requires_grad)
        
        if self.pos_drop:
            x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Global pooling
        if self.global_pool == 'avg':
            x_data = np.mean(x.data[:, self.num_tokens:], axis=1)
            x = Tensor(x_data, requires_grad=x.requires_grad)
        else:  # 'token'
            x_data = x.data[:, 0]
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ViT."""
        x = self.forward_features(x)
        if self.head is not None:
            x = self.head(x)
        return x


# Model configurations
def ViT_B_16(**kwargs):
    """ViT-Base with 16x16 patches."""
    return VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )

def ViT_L_16(**kwargs):
    """ViT-Large with 16x16 patches."""
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )

def ViT_H_14(**kwargs):
    """ViT-Huge with 14x14 patches."""
    return VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs
    )


# Register models
@register_model(
    name='vit_b_16',
    description='Vision Transformer Base with 16x16 patches',
    paper_url='https://arxiv.org/abs/2010.11929',
    pretrained_configs={
        'imagenet': {'num_classes': 1000, 'drop_path_rate': 0.1},
        'imagenet21k': {'num_classes': 21843, 'drop_path_rate': 0.1}
    },
    default_config='imagenet',
    tags=['vision', 'classification', 'transformer', 'vit'],
    aliases=['vit-b-16', 'vit_base_16']
)
class RegisteredViTB16(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)

def vit_b_16(**kwargs):
    return RegisteredViTB16(**kwargs)


@register_model(
    name='vit_l_16',
    description='Vision Transformer Large with 16x16 patches',
    paper_url='https://arxiv.org/abs/2010.11929',
    pretrained_configs={
        'imagenet': {'num_classes': 1000, 'drop_path_rate': 0.2},
        'imagenet21k': {'num_classes': 21843, 'drop_path_rate': 0.2}
    },
    default_config='imagenet',
    tags=['vision', 'classification', 'transformer', 'vit'],
    aliases=['vit-l-16', 'vit_large_16']
)
class RegisteredViTL16(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)

def vit_l_16(**kwargs):
    return RegisteredViTL16(**kwargs)


@register_model(
    name='vit_h_14',
    description='Vision Transformer Huge with 14x14 patches',
    paper_url='https://arxiv.org/abs/2010.11929',
    pretrained_configs={
        'imagenet': {'num_classes': 1000, 'drop_path_rate': 0.3},
        'imagenet21k': {'num_classes': 21843, 'drop_path_rate': 0.3}
    },
    default_config='imagenet21k',
    tags=['vision', 'classification', 'transformer', 'vit'],
    aliases=['vit-h-14', 'vit_huge_14']
)
class RegisteredViTH14(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)

def vit_h_14(**kwargs):
    return RegisteredViTH14(**kwargs)


# Additional variants
def vit_b_32(**kwargs):
    """ViT-Base with 32x32 patches."""
    return VisionTransformer(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs
    )

def vit_l_32(**kwargs):
    """ViT-Large with 32x32 patches."""
    return VisionTransformer(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )