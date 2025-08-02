"""T5 implementation.

From "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
https://arxiv.org/abs/1910.10683
"""

from ...core import Module
from ..registry import register_model


class T5(Module):
    """T5 model."""

    def __init__(
        self, d_model: int = 512, num_layers: int = 6, num_heads: int = 8, vocab_size: int = 32128
    ):
        super().__init__()
        # Implementation will be added
        pass

    def forward(self, x):
        return x


@register_model(
    name="t5_small",
    description="T5 Small model for text-to-text generation",
    paper_url="https://arxiv.org/abs/1910.10683",
    pretrained_configs={"google": {"d_model": 512, "num_layers": 6}},
    default_config="google",
    tags=["language", "t5", "text2text"],
    aliases=["t5-small"],
)
class RegisteredT5Small(T5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def t5_small(**kwargs):
    return RegisteredT5Small(**kwargs)


T5Small = t5_small
T5Base = lambda **kwargs: T5(d_model=768, num_layers=12, **kwargs)
T5Large = lambda **kwargs: T5(d_model=1024, num_layers=24, **kwargs)
T53B = lambda **kwargs: T5(d_model=1024, num_layers=24, **kwargs)
T511B = lambda **kwargs: T5(d_model=1024, num_layers=24, **kwargs)
T5Model = T5
T5ForConditionalGeneration = T5


# Function variants for consistency
def t5_base(**kwargs):
    return T5(d_model=768, num_layers=12, **kwargs)


def t5_large(**kwargs):
    return T5(d_model=1024, num_layers=24, **kwargs)


def t5_3b(**kwargs):
    return T5(d_model=1024, num_layers=24, **kwargs)


def t5_11b(**kwargs):
    return T5(d_model=1024, num_layers=24, **kwargs)
