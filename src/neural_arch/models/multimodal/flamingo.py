"""Flamingo implementation.

From "Flamingo: a Visual Language Model for Few-Shot Learning"
https://arxiv.org/abs/2204.14198
"""

from ...core import Module
from ..registry import register_model


class Flamingo(Module):
    """Flamingo model."""

    def __init__(self, vision_encoder: str = "vit_l", language_model: str = "chinchilla"):
        super().__init__()
        # Implementation will be added
        pass

    def forward(self, image=None, text=None):
        return {}


@register_model(
    name="flamingo_base",
    description="Flamingo model for few-shot learning",
    paper_url="https://arxiv.org/abs/2204.14198",
    pretrained_configs={"deepmind": {}},
    default_config="deepmind",
    tags=["multimodal", "few-shot", "flamingo"],
    aliases=["flamingo-base"],
)
class RegisteredFlamingoBase(Flamingo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def flamingo_base(**kwargs):
    return RegisteredFlamingoBase(**kwargs)


FlamingoBase = flamingo_base
FlamingoModel = Flamingo
