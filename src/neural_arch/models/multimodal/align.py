"""ALIGN implementation.

From "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision"
https://arxiv.org/abs/2102.05918
"""

from ...core import Module
from ..registry import register_model


class ALIGN(Module):
    """ALIGN model."""
    
    def __init__(self, vision_model: str = "efficientnet_l2", text_model: str = "bert_large"):
        super().__init__()
        # Implementation will be added
        pass
    
    def forward(self, image=None, text=None):
        return {}


@register_model(
    name='align_base',
    description='ALIGN model with EfficientNet and BERT',
    paper_url='https://arxiv.org/abs/2102.05918',
    pretrained_configs={'google': {}},
    default_config='google',
    tags=['multimodal', 'vision-language', 'align'],
    aliases=['align-base']
)
class RegisteredALIGNBase(ALIGN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def align_base(**kwargs):
    return RegisteredALIGNBase(**kwargs)

ALIGNBase = align_base
ALIGNModel = ALIGN
ALIGNVisionModel = ALIGN
ALIGNTextModel = ALIGN