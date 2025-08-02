"""ConvNeXt implementation.

From "A ConvNet for the 2020s"
https://arxiv.org/abs/2201.03545

Modern CNN design inspired by Vision Transformers.
Key improvements:
- Depthwise convolutions
- Larger kernel sizes (7x7)
- GELU activation
- Layer normalization
- Inverted bottleneck design
"""

from typing import List, Optional

import numpy as np

from ...core import Module, Parameter, Tensor
from ...functional import gelu
from ...nn import Conv2d, LayerNorm, Linear
from ..registry import register_model


class DropPath(Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + np.random.rand(*x.shape[:1] + (1,) * (len(x.shape) - 1))
        random_tensor = np.floor(random_tensor)
        
        output_data = x.data * random_tensor / keep_prob
        return Tensor(output_data, requires_grad=x.requires_grad)


class ConvNeXtBlock(Module):
    """ConvNeXt Block with depthwise convolution and inverted bottleneck."""
    
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        
        # Depthwise convolution (7x7)
        self.dwconv = Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm (applied channelwise)
        self.norm = LayerNorm(dim)
        
        # Pointwise convolutions for inverted bottleneck  
        self.pwconv1 = Linear(dim, 4 * dim)  # expand
        self.pwconv2 = Linear(4 * dim, dim)  # project back
        
        # Layer scale parameter
        if layer_scale_init_value > 0:
            self.gamma = Parameter(
                layer_scale_init_value * np.ones(dim, dtype=np.float32),
                name="layer_scale"
            )
        else:
            self.gamma = None
            
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
    
    def forward(self, x: Tensor) -> Tensor:
        input_x = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        
        # Permute for LayerNorm: (N, C, H, W) -> (N, H, W, C)
        x_data = np.transpose(x.data, (0, 2, 3, 1))
        x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # LayerNorm
        x = self.norm(x)
        
        # Pointwise convolution 1 (expand)
        x = self.pwconv1(x)
        x = gelu(x)
        
        # Pointwise convolution 2 (project)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x_data = x.data * self.gamma.data
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Permute back: (N, H, W, C) -> (N, C, H, W)
        x_data = np.transpose(x.data, (0, 3, 1, 2))
        x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Drop path and residual connection
        if self.drop_path is not None:
            x = self.drop_path(x)
        
        x_data = x.data + input_x.data
        return Tensor(x_data, requires_grad=x.requires_grad or input_x.requires_grad)


class ConvNeXt(Module):
    """ConvNeXt model.
    
    Args:
        in_chans: Number of input image channels
        num_classes: Number of classes for classification head
        depths: Number of blocks at each stage
        dims: Feature dimension at each stage
        drop_path_rate: Stochastic depth rate
        layer_scale_init_value: Init value for Layer Scale
        head_init_scale: Init scaling value for classifier weights and biases
    """

    def __init__(
        self, 
        in_chans: int = 3,
        num_classes: int = 1000, 
        depths: List[int] = [3, 3, 9, 3], 
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()
        
        # Stem: 4x4 conv with stride 4 (patchify)
        self.downsample_layers = []
        stem = [
            Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        ]
        self.downsample_layers.append(stem)
        
        # Downsampling layers between stages
        for i in range(3):
            downsample_layer = [
                LayerNorm(dims[i], eps=1e-6),
                Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            ]
            self.downsample_layers.append(downsample_layer)
        
        # Register downsampling layers as submodules
        for i, layers in enumerate(self.downsample_layers):
            for j, layer in enumerate(layers):
                self.register_module(f"downsample_{i}_{j}", layer)
        
        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = []
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                block = ConvNeXtBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                )
                stage_blocks.append(block)
            self.stages.append(stage_blocks)
            cur += depths[i]
        
        # Register stage blocks as submodules
        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage):
                self.register_module(f"stage_{i}_block_{j}", block)
        
        # Final layer norm
        self.norm = LayerNorm(dims[-1], eps=1e-6)
        
        # Classifier head
        self.head = Linear(dims[-1], num_classes)
        
        # Initialize head with smaller scale
        if head_init_scale != 1.0:
            self.head.weight.data *= head_init_scale
            if self.head.bias is not None:
                self.head.bias.data *= head_init_scale

    def _downsample_forward(self, x: Tensor, layers: List[Module]) -> Tensor:
        """Apply downsampling layers."""
        for layer in layers:
            if isinstance(layer, LayerNorm):
                # Permute for LayerNorm: (N, C, H, W) -> (N, H, W, C)
                x_data = np.transpose(x.data, (0, 2, 3, 1))
                x = Tensor(x_data, requires_grad=x.requires_grad)
                x = layer(x)
                # Permute back: (N, H, W, C) -> (N, C, H, W)
                x_data = np.transpose(x.data, (0, 3, 1, 2))
                x = Tensor(x_data, requires_grad=x.requires_grad)
            else:
                x = layer(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        """Extract features from input."""
        for i in range(4):
            # Downsampling
            x = self._downsample_forward(x, self.downsample_layers[i])
            
            # Stage blocks
            for block in self.stages[i]:
                x = block(x)
        
        # Global average pooling
        x_data = np.mean(x.data, axis=(2, 3))  # (N, C)
        x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Final norm
        x = self.norm(x)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model(
    name="convnext_tiny",
    description="ConvNeXt Tiny model",
    paper_url="https://arxiv.org/abs/2201.03545",
    pretrained_configs={"imagenet": {"num_classes": 1000}},
    default_config="imagenet",
    tags=["vision", "classification", "convnext"],
    aliases=["convnext-tiny"],
)
class RegisteredConvNeXtTiny(ConvNeXt):
    def __init__(self, **kwargs):
        super().__init__([3, 3, 9, 3], [96, 192, 384, 768], **kwargs)


def convnext_tiny(**kwargs):
    return RegisteredConvNeXtTiny(**kwargs)


ConvNeXtTiny = convnext_tiny
ConvNeXtSmall = lambda **kwargs: ConvNeXt([3, 3, 27, 3], [96, 192, 384, 768], **kwargs)
ConvNeXtBase = lambda **kwargs: ConvNeXt([3, 3, 27, 3], [128, 256, 512, 1024], **kwargs)
ConvNeXtLarge = lambda **kwargs: ConvNeXt([3, 3, 27, 3], [192, 384, 768, 1536], **kwargs)


# Function variants for consistency
def convnext_small(**kwargs):
    return ConvNeXt([3, 3, 27, 3], [96, 192, 384, 768], **kwargs)


def convnext_base(**kwargs):
    return ConvNeXt([3, 3, 27, 3], [128, 256, 512, 1024], **kwargs)


def convnext_large(**kwargs):
    return ConvNeXt([3, 3, 27, 3], [192, 384, 768, 1536], **kwargs)
