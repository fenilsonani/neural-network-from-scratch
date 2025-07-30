"""ConvNeXt implementation.

From "A ConvNet for the 2020s"
https://arxiv.org/abs/2201.03545

Modern CNN design inspired by Vision Transformers.
"""

from ...core import Module
from ..registry import register_model


class ConvNeXt(Module):
    """ConvNeXt model."""
    
    def __init__(self, depths: list, dims: list, num_classes: int = 1000):
        super().__init__()
        # Implementation will be added
        pass
    
    def forward(self, x):
        return x


@register_model(
    name='convnext_tiny',
    description='ConvNeXt Tiny model',
    paper_url='https://arxiv.org/abs/2201.03545',
    pretrained_configs={'imagenet': {'num_classes': 1000}},
    default_config='imagenet',
    tags=['vision', 'classification', 'convnext'],
    aliases=['convnext-tiny']
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