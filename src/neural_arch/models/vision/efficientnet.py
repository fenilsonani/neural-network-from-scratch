"""EfficientNet implementation with compound scaling.

From "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
https://arxiv.org/abs/1905.11946

Features:
- Compound scaling of depth, width, and resolution
- Mobile Inverted Bottleneck Convolution (MBConv)
- Squeeze-and-Excitation blocks
- Stochastic depth
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from ...core import Module, Parameter, Tensor
from ...functional import relu, sigmoid
from ...nn import BatchNorm2d, Conv2d, Dropout, Linear
from ..registry import register_model


class SqueezeExcitation(Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()
        self.fc1 = Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = Conv2d(reduced_channels, in_channels, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        # Global average pooling
        y_data = np.mean(x.data, axis=(2, 3), keepdims=True)
        y = Tensor(y_data, requires_grad=x.requires_grad)
        
        # Squeeze
        y = self.fc1(y)
        y = relu(y)
        
        # Excitation
        y = self.fc2(y)
        y = sigmoid(y)
        
        # Scale
        return Tensor(x.data * y.data, requires_grad=x.requires_grad or y.requires_grad)


class DropPath(Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + np.random.rand(x.shape[0], 1, 1, 1)
        random_tensor = np.floor(random_tensor)
        
        output_data = x.data * random_tensor / keep_prob
        return Tensor(output_data, requires_grad=x.requires_grad)


class MBConvBlock(Module):
    """Mobile Inverted Bottleneck Convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
        id_skip: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.id_skip = id_skip and stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            self.expand_bn = BatchNorm2d(expanded_channels)
        else:
            self.expand_conv = None
            self.expand_bn = None
        
        # Depthwise convolution
        self.depthwise_conv = Conv2d(
            expanded_channels, 
            expanded_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expanded_channels,
            bias=False
        )
        self.depthwise_bn = BatchNorm2d(expanded_channels)
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(expanded_channels, se_channels)
        else:
            self.se = None
        
        # Output projection
        self.project_conv = Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = BatchNorm2d(out_channels)
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
    
    def forward(self, x: Tensor) -> Tensor:
        input_tensor = x
        
        # Expansion phase
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = relu(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = relu(x)
        
        # Squeeze-and-Excitation
        if self.se is not None:
            x = self.se(x)
        
        # Output projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Skip connection and drop path
        if self.id_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x_data = x.data + input_tensor.data
            x = Tensor(x_data, requires_grad=x.requires_grad or input_tensor.requires_grad)
        
        return x


class EfficientNet(Module):
    """EfficientNet with compound scaling.
    
    Args:
        width_coefficient: Width scaling coefficient
        depth_coefficient: Depth scaling coefficient  
        dropout_rate: Dropout rate for classifier
        num_classes: Number of classes for classification
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        width_coefficient: float,
        depth_coefficient: float,
        dropout_rate: float = 0.2,
        num_classes: int = 1000,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        # Base configuration for EfficientNet-B0
        base_config = [
            # (kernel_size, stride, expand_ratio, channels, num_blocks, se_ratio)
            (3, 1, 1, 16, 1, 0.25),   # MBConv1_3x3, SE, 16, 1
            (3, 2, 6, 24, 2, 0.25),   # MBConv6_3x3, SE, 24, 2
            (5, 2, 6, 40, 2, 0.25),   # MBConv6_5x5, SE, 40, 2
            (3, 2, 6, 80, 3, 0.25),   # MBConv6_3x3, SE, 80, 3
            (5, 1, 6, 112, 3, 0.25),  # MBConv6_5x5, SE, 112, 3
            (5, 2, 6, 192, 4, 0.25),  # MBConv6_5x5, SE, 192, 4
            (3, 1, 6, 320, 1, 0.25),  # MBConv6_3x3, SE, 320, 1
        ]
        
        def round_filters(filters: int, width_coefficient: float) -> int:
            """Round number of filters based on width multiplier."""
            if not width_coefficient:
                return filters
            filters *= width_coefficient
            new_filters = max(8, int(filters + 4) // 8 * 8)
            # Make sure that round down does not go down by more than 10%
            if new_filters < 0.9 * filters:
                new_filters += 8
            return int(new_filters)
        
        def round_repeats(repeats: int, depth_coefficient: float) -> int:
            """Round number of repeats based on depth multiplier."""
            if not depth_coefficient:
                return repeats
            return int(math.ceil(depth_coefficient * repeats))
        
        # Stem
        stem_channels = round_filters(32, width_coefficient)
        self.stem_conv = Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = BatchNorm2d(stem_channels)
        
        # Build blocks
        blocks = []
        in_channels = stem_channels
        
        # Calculate total number of blocks for drop path
        total_blocks = sum([round_repeats(repeats, depth_coefficient) for _, _, _, _, repeats, _ in base_config])
        block_idx = 0
        
        for kernel_size, stride, expand_ratio, channels, repeats, se_ratio in base_config:
            out_channels = round_filters(channels, width_coefficient)
            num_repeats = round_repeats(repeats, depth_coefficient)
            
            for i in range(num_repeats):
                drop_path = drop_path_rate * block_idx / total_blocks
                
                blocks.append(MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,  # Only first block in stage has stride > 1
                    expand_ratio=expand_ratio,
                    se_ratio=se_ratio,
                    drop_path=drop_path,
                ))
                
                in_channels = out_channels
                block_idx += 1
        
        self.blocks = blocks
        
        # Register blocks as submodules
        for i, block in enumerate(self.blocks):
            self.register_module(f"block_{i}", block)
        
        # Head
        head_channels = round_filters(1280, width_coefficient)
        self.head_conv = Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.head_bn = BatchNorm2d(head_channels)
        
        # Classifier
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(head_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling."""
        for module in [self.stem_conv, self.head_conv]:
            if hasattr(module, 'weight'):
                fan_out = module.weight.data.shape[0] * module.weight.data.shape[2] * module.weight.data.shape[3]
                module.weight.data = np.random.normal(0, math.sqrt(2.0 / fan_out), module.weight.data.shape).astype(np.float32)
    
    def forward_features(self, x: Tensor) -> Tensor:
        """Extract features."""
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = relu(x)
        
        # Blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = relu(x)
        
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.forward_features(x)
        
        # Global average pooling
        x_data = np.mean(x.data, axis=(2, 3))
        x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Classifier
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


# Model variants
@register_model(
    name="efficientnet_b0",
    description="EfficientNet-B0 baseline model",
    paper_url="https://arxiv.org/abs/1905.11946",
    pretrained_configs={"imagenet": {"num_classes": 1000}},
    default_config="imagenet",
    tags=["vision", "classification", "efficientnet"],
    aliases=["efficientnet-b0"],
)
class RegisteredEfficientNetB0(EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(1.0, 1.0, 0.2, **kwargs)


def efficientnet_b0(**kwargs):
    return RegisteredEfficientNetB0(**kwargs)


# Additional variants
EfficientNetB0 = efficientnet_b0
EfficientNetB1 = lambda **kwargs: EfficientNet(1.0, 1.1, 0.2, **kwargs)
EfficientNetB2 = lambda **kwargs: EfficientNet(1.1, 1.2, 0.3, **kwargs)
EfficientNetB3 = lambda **kwargs: EfficientNet(1.2, 1.4, 0.3, **kwargs)
EfficientNetB4 = lambda **kwargs: EfficientNet(1.4, 1.8, 0.4, **kwargs)
EfficientNetB5 = lambda **kwargs: EfficientNet(1.6, 2.2, 0.4, **kwargs)
EfficientNetB6 = lambda **kwargs: EfficientNet(1.8, 2.6, 0.5, **kwargs)
EfficientNetB7 = lambda **kwargs: EfficientNet(2.0, 3.1, 0.5, **kwargs)


# Function variants for consistency
def efficientnet_b1(**kwargs):
    return EfficientNet(1.0, 1.1, 0.2, **kwargs)


def efficientnet_b2(**kwargs):
    return EfficientNet(1.1, 1.2, 0.3, **kwargs)


def efficientnet_b3(**kwargs):
    return EfficientNet(1.2, 1.4, 0.3, **kwargs)


def efficientnet_b4(**kwargs):
    return EfficientNet(1.4, 1.8, 0.4, **kwargs)


def efficientnet_b5(**kwargs):
    return EfficientNet(1.6, 2.2, 0.4, **kwargs)


def efficientnet_b6(**kwargs):
    return EfficientNet(1.8, 2.6, 0.5, **kwargs)


def efficientnet_b7(**kwargs):
    return EfficientNet(2.0, 3.1, 0.5, **kwargs)
