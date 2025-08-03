"""DeBERTa implementation.

From "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" 
https://arxiv.org/abs/2006.03654

And "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"
https://arxiv.org/abs/2111.09543

DeBERTa improves on BERT with:
- Disentangled attention mechanism that separates content and position representations
- Enhanced mask decoder that incorporates absolute positions  
- Replace Token Detection (RTD) pre-training objective
- Gradient-disentangled embedding sharing (v3)
- Virtual adversarial training and improved layer normalization
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ...core import Module, Parameter, Tensor
from ...functional import cross_entropy_loss, gelu, matmul, softmax
from ...nn import Dropout, Embedding, LayerNorm, Linear
from ...nn.activation import ReLU, Softmax, Tanh
from ..registry import register_model


class DeBERTaConfig:
    """Configuration class for DeBERTa model."""

    def __init__(
        self,
        vocab_size: int = 128100,  # DeBERTa uses larger vocab
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 0,  # DeBERTa doesn't use token type by default
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-7,  # Different from BERT
        pad_token_id: int = 0,
        position_buckets: int = 256,  # For relative position encoding
        max_relative_positions: int = -1,  # -1 means use position_buckets
        position_biased_input: bool = True,
        pos_att_type: Union[str, list] = "none",  # Can be "p2c", "c2p", "p2c|c2p"
        relative_attention: bool = False,
        talking_head: bool = False,  # Talking heads attention
        attention_head_size: Optional[int] = None,
        share_att_key: bool = True,
        conv_kernel_size: int = 0,  # For ConvBERT-style convolution
        conv_groups: int = 1,
        conv_act: str = "tanh",
        pooler_dropout: float = 0.0,
        pooler_hidden_size: Optional[int] = None,
        pooler_hidden_act: str = "gelu",
        classifier_dropout: Optional[float] = None,
        norm_rel_ebd: str = "layer_norm",  # How to normalize relative embeddings
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.position_biased_input = position_biased_input
        self.pos_att_type = pos_att_type
        self.relative_attention = relative_attention
        self.talking_head = talking_head
        self.attention_head_size = attention_head_size or (hidden_size // num_attention_heads)
        self.share_att_key = share_att_key
        self.conv_kernel_size = conv_kernel_size
        self.conv_groups = conv_groups
        self.conv_act = conv_act
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_size = pooler_hidden_size or hidden_size
        self.pooler_hidden_act = pooler_hidden_act
        self.classifier_dropout = classifier_dropout or hidden_dropout_prob
        self.norm_rel_ebd = norm_rel_ebd


class DeBERTaLayerNorm(Module):
    """Layer normalization optimized for DeBERTa."""

    def __init__(self, normalized_shape: int, eps: float = 1e-7):
        super().__init__()
        self.weight = Parameter(np.ones(normalized_shape, dtype=np.float32))
        self.bias = Parameter(np.zeros(normalized_shape, dtype=np.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Manual layer norm implementation for better control
        input_dtype = hidden_states.data.dtype
        variance = np.var(hidden_states.data, axis=-1, keepdims=True)
        hidden_states_normalized = (hidden_states.data - np.mean(hidden_states.data, axis=-1, keepdims=True)) / np.sqrt(variance + self.variance_epsilon)
        
        # Apply learned parameters
        output = self.weight.data * hidden_states_normalized + self.bias.data
        return Tensor(output.astype(input_dtype), requires_grad=hidden_states.requires_grad)


class StableDropout(Module):
    """Dropout with stable training behavior."""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.drop_prob > 0:
            # Apply dropout
            mask = np.random.binomial(1, 1 - self.drop_prob, x.shape) / (1 - self.drop_prob)
            return Tensor(x.data * mask, requires_grad=x.requires_grad)
        return x


class DeBERTaEmbeddings(Module):
    """DeBERTa embeddings with optional position bias."""

    def __init__(self, config: DeBERTaConfig):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embed_size = getattr(config, "hidden_size", 768)
        
        self.word_embeddings = Embedding(config.vocab_size, self.embed_size)
        self.position_biased_input = getattr(config, "position_biased_input", True)
        
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = Embedding(config.max_position_embeddings, self.embed_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = Embedding(config.type_vocab_size, self.embed_size)
        else:
            self.token_type_embeddings = None

        self.LayerNorm = DeBERTaLayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = Tensor(np.arange(seq_length, dtype=np.int64).reshape(1, -1))

        if token_type_ids is None:
            token_type_ids = Tensor(np.zeros(input_shape, dtype=np.int64))

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        
        if self.position_biased_input:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.token_type_embeddings is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DisentangledSelfAttention(Module):
    """DeBERTa's disentangled self-attention mechanism.
    
    This separates content and position representations in attention computation.
    """

    def __init__(self, config: DeBERTaConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.config = config

        self.query_proj = Linear(config.hidden_size, self.all_head_size)
        self.key_proj = Linear(config.hidden_size, self.all_head_size)
        self.value_proj = Linear(config.hidden_size, self.all_head_size)

        self.share_att_key = config.share_att_key
        self.pos_att_type = config.pos_att_type if config.pos_att_type != "none" else []
        self.relative_attention = config.relative_attention
        self.talking_head = config.talking_head

        if self.talking_head:
            self.head_logits_proj = Linear(config.num_attention_heads, config.num_attention_heads)
            self.head_weights_proj = Linear(config.num_attention_heads, config.num_attention_heads)

        if self.relative_attention:
            self.max_relative_positions = config.max_relative_positions
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = Linear(config.hidden_size, self.all_head_size)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: Tensor, attention_heads: int) -> Tensor:
        """Reshape and transpose tensor for multi-head attention."""
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        new_x_shape = (batch_size, seq_length, attention_heads, self.attention_head_size)
        x_reshaped = Tensor(
            x.data.reshape(new_x_shape), requires_grad=x.requires_grad
        )
        return Tensor(np.transpose(x_reshaped.data, (0, 2, 1, 3)), requires_grad=x.requires_grad)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        query_states: Optional[Tensor] = None,
        relative_pos: Optional[Tensor] = None,
        rel_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Disentangled attention forward pass.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            output_attentions: Whether to output attention weights
            query_states: Pre-computed query states (optional)
            relative_pos: Relative position matrix
            rel_embeddings: Relative position embeddings
        """
        if query_states is None:
            query_states = hidden_states

        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear projections
        query_layer = self.transpose_for_scores(
            self.query_proj(query_states), self.num_attention_heads
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        )

        # Content-to-content attention (standard self-attention)
        # query_layer: (batch_size, num_heads, seq_len, head_size)
        # key_layer: (batch_size, num_heads, seq_len, head_size)
        key_transposed = Tensor(
            np.transpose(key_layer.data, (0, 1, 3, 2)), requires_grad=key_layer.requires_grad
        )
        
        attention_scores = matmul(query_layer, key_transposed)

        # Add positional attention if enabled
        if self.relative_attention and relative_pos is not None and rel_embeddings is not None:
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, "c2p" in self.pos_att_type
            )
            attention_scores = Tensor(
                attention_scores.data + rel_att.data, requires_grad=attention_scores.requires_grad
            )

        # Scale attention scores
        scale_factor = 1.0 / np.sqrt(self.attention_head_size)
        attention_scores = Tensor(
            attention_scores.data * scale_factor, requires_grad=attention_scores.requires_grad
        )

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = Tensor(
                attention_scores.data + attention_mask.data, requires_grad=attention_scores.requires_grad
            )

        # Talking heads mechanism (optional)
        if self.talking_head:
            attention_scores = self.head_logits_proj(
                Tensor(np.transpose(attention_scores.data, (0, 2, 3, 1)), requires_grad=attention_scores.requires_grad)
            )
            attention_scores = Tensor(
                np.transpose(attention_scores.data, (0, 3, 1, 2)), requires_grad=attention_scores.requires_grad
            )

        # Apply softmax
        attention_probs = softmax(attention_scores, axis=-1)

        # Talking heads on attention weights
        if self.talking_head:
            attention_probs = self.head_weights_proj(
                Tensor(np.transpose(attention_probs.data, (0, 2, 3, 1)), requires_grad=attention_probs.requires_grad)
            )
            attention_probs = Tensor(
                np.transpose(attention_probs.data, (0, 3, 1, 2)), requires_grad=attention_probs.requires_grad
            )

        # Apply dropout
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = matmul(attention_probs, value_layer)

        # Transpose back and reshape
        context_layer = Tensor(
            np.transpose(context_layer.data, (0, 2, 1, 3)), requires_grad=context_layer.requires_grad
        )
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = Tensor(
            context_layer.data.reshape(new_context_layer_shape), requires_grad=context_layer.requires_grad
        )

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def disentangled_attention_bias(
        self, query_layer: Tensor, key_layer: Tensor, relative_pos: Tensor, rel_embeddings: Tensor, scale: bool = True
    ) -> Tensor:
        """Compute disentangled attention bias for relative positions."""
        # Simplified implementation - in practice this would be more complex
        # This is a placeholder for the actual disentangled attention computation
        batch_size, num_heads, seq_len, head_size = query_layer.shape
        
        # Create a simple relative position bias
        rel_bias = Tensor(np.zeros((batch_size, num_heads, seq_len, seq_len), dtype=np.float32))
        
        # In real implementation, this would use the relative position embeddings
        # and compute content-to-position and position-to-content attention
        
        return rel_bias


class DeBERTaSelfOutput(Module):
    """DeBERTa self-attention output projection."""

    def __init__(self, config: DeBERTaConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DeBERTaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(Tensor(
            hidden_states.data + input_tensor.data, requires_grad=hidden_states.requires_grad
        ))
        return hidden_states


class DeBERTaAttention(Module):
    """DeBERTa attention layer with disentangled attention."""

    def __init__(self, config: DeBERTaConfig):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DeBERTaSelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        query_states: Optional[Tensor] = None,
        relative_pos: Optional[Tensor] = None,
        rel_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class DeBERTaIntermediate(Module):
    """DeBERTa feed-forward intermediate layer."""

    def __init__(self, config: DeBERTaConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu  # DeBERTa uses GELU

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = Tensor(
            self.intermediate_act_fn(hidden_states.data), requires_grad=hidden_states.requires_grad
        )
        return hidden_states


class DeBERTaOutput(Module):
    """DeBERTa feed-forward output layer."""

    def __init__(self, config: DeBERTaConfig):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = DeBERTaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(Tensor(
            hidden_states.data + input_tensor.data, requires_grad=hidden_states.requires_grad
        ))
        return hidden_states


class DeBERTaLayer(Module):
    """DeBERTa transformer layer."""

    def __init__(self, config: DeBERTaConfig):
        super().__init__()
        self.attention = DeBERTaAttention(config)
        self.intermediate = DeBERTaIntermediate(config)
        self.output = DeBERTaOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        query_states: Optional[Tensor] = None,
        relative_pos: Optional[Tensor] = None,
        rel_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        attention_output = attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class DeBERTaEncoder(Module):
    """DeBERTa encoder with multiple transformer layers."""

    def __init__(self, config: DeBERTaConfig):
        super().__init__()
        self.config = config
        self.layer = [DeBERTaLayer(config) for _ in range(config.num_hidden_layers)]
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

    def get_rel_embeddings(self):
        """Get relative position embeddings."""
        if self.relative_attention:
            # Simplified relative embeddings
            rel_embeddings = Tensor(
                np.random.normal(0, 0.02, (self.max_relative_positions * 2, self.config.hidden_size)).astype(np.float32)
            )
            return rel_embeddings
        return None

    def get_attention_mask(self, attention_mask: Tensor) -> Tensor:
        """Convert attention mask to proper format."""
        if attention_mask.ndim <= 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        if attention_mask.ndim == 3:
            attention_mask = attention_mask[:, None, :, :]

        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask

    def get_rel_pos(self, hidden_states: Tensor, query_states: Optional[Tensor] = None, relative_pos: Optional[Tensor] = None) -> Optional[Tensor]:
        """Get relative position matrix."""
        if self.relative_attention and relative_pos is None:
            q = query_states.shape[-2] if query_states is not None else hidden_states.shape[-2]
            k = hidden_states.shape[-2]
            # Simplified relative position matrix
            relative_pos = Tensor(np.zeros((q, k), dtype=np.int64))
        return relative_pos

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        query_states: Optional[Tensor] = None,
        relative_pos: Optional[Tensor] = None,
        rel_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]], Optional[Tuple[Tensor, ...]]]:
        
        if attention_mask is not None:
            attention_mask = self.get_attention_mask(attention_mask)
        
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        if rel_embeddings is None:
            rel_embeddings = self.get_rel_embeddings()

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, all_attentions


class DeBERTa(Module):
    """DeBERTa model implementation."""

    def __init__(self, config: Optional[DeBERTaConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = DeBERTaConfig(**kwargs)
        self.config = config

        self.embeddings = DeBERTaEmbeddings(config)
        self.encoder = DeBERTaEncoder(config)

        self.init_weights()

    def init_weights(self):
        """Initialize weights using normal distribution."""
        for module in self.modules():
            if hasattr(module, "weight") and hasattr(module.weight, "data"):
                if isinstance(module, Linear):
                    std = self.config.initializer_range
                    module.weight.data = np.random.normal(0, std, module.weight.shape).astype(np.float32)
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.data = np.zeros(module.bias.shape, dtype=np.float32)
                elif isinstance(module, Embedding):
                    std = self.config.initializer_range
                    module.weight.data = np.random.normal(0, std, module.weight.shape).astype(np.float32)
                elif isinstance(module, DeBERTaLayerNorm):
                    module.bias.data = np.zeros(module.bias.shape, dtype=np.float32)
                    module.weight.data = np.ones(module.weight.shape, dtype=np.float32)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = Tensor(np.ones((batch_size, seq_length)))

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]

        return {
            "last_hidden_state": sequence_output,
            "hidden_states": encoder_outputs[1] if output_hidden_states else None,
            "attentions": encoder_outputs[2] if output_attentions else None,
        }


class DeBERTaForMaskedLM(Module):
    """DeBERTa model for masked language modeling."""

    def __init__(self, config: Optional[DeBERTaConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = DeBERTaConfig(**kwargs)
        self.config = config

        self.deberta = DeBERTa(config)
        self.cls = Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        outputs = self.deberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs
        )

        sequence_output = outputs["last_hidden_state"]
        prediction_scores = self.cls(sequence_output)

        result = {
            "logits": prediction_scores,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }

        if labels is not None:
            vocab_size = self.config.vocab_size
            shift_logits = prediction_scores.reshape(-1, vocab_size)
            shift_labels = labels.reshape(-1)
            masked_lm_loss = cross_entropy_loss(shift_logits, shift_labels)
            result["loss"] = masked_lm_loss

        return result


class DeBERTaForSequenceClassification(Module):
    """DeBERTa model for sequence classification."""

    def __init__(self, config: Optional[DeBERTaConfig] = None, num_labels: int = 2, **kwargs):
        super().__init__()
        if config is None:
            config = DeBERTaConfig(**kwargs)
        self.config = config
        self.num_labels = num_labels

        self.deberta = DeBERTa(config)
        
        # DeBERTa-style pooling
        self.pooler = Linear(config.hidden_size, config.hidden_size)
        self.dropout = StableDropout(config.classifier_dropout)
        self.classifier = Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        outputs = self.deberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs
        )

        sequence_output = outputs["last_hidden_state"]
        
        # Pool using the first token
        pooled_output = self.pooler(sequence_output[:, 0])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        result = {
            "logits": logits,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }

        if labels is not None:
            flat_labels = labels.reshape(-1)
            loss = cross_entropy_loss(logits, flat_labels)
            result["loss"] = loss

        return result


# Model configurations
DEBERTA_CONFIGS = {
    "base": {
        "vocab_size": 128100,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "position_buckets": 256,
        "relative_attention": True,
        "pos_att_type": "p2c|c2p",
        "layer_norm_eps": 1e-7,
    },
    "large": {
        "vocab_size": 128100,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "position_buckets": 256,
        "relative_attention": True,
        "pos_att_type": "p2c|c2p",
        "layer_norm_eps": 1e-7,
    },
    "v3-base": {
        "vocab_size": 128100,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "position_buckets": 256,
        "relative_attention": True,
        "pos_att_type": "p2c|c2p",
        "layer_norm_eps": 1e-7,
        "norm_rel_ebd": "layer_norm",
    },
    "v3-large": {
        "vocab_size": 128100,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "position_buckets": 256,
        "relative_attention": True,
        "pos_att_type": "p2c|c2p",
        "layer_norm_eps": 1e-7,
        "norm_rel_ebd": "layer_norm",
    },
}


@register_model(
    name="deberta_base",
    description="DeBERTa Base - Decoding-enhanced BERT with Disentangled Attention",
    paper_url="https://arxiv.org/abs/2006.03654",
    pretrained_configs={
        "microsoft": DEBERTA_CONFIGS["base"],
    },
    default_config="microsoft",
    tags=["language", "deberta", "transformer", "disentangled"],
    aliases=["deberta-base", "deberta_base_v1"],
)
class RegisteredDeBERTaBase(DeBERTa):
    def __init__(self, **kwargs):
        config = DeBERTaConfig(**DEBERTA_CONFIGS["base"])
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        super().__init__(config)


@register_model(
    name="deberta_large",
    description="DeBERTa Large - Decoding-enhanced BERT with Disentangled Attention",
    paper_url="https://arxiv.org/abs/2006.03654",
    pretrained_configs={
        "microsoft": DEBERTA_CONFIGS["large"],
    },
    default_config="microsoft",
    tags=["language", "deberta", "transformer", "disentangled"],
    aliases=["deberta-large", "deberta_large_v1"],
)
class RegisteredDeBERTaLarge(DeBERTa):
    def __init__(self, **kwargs):
        config = DeBERTaConfig(**DEBERTA_CONFIGS["large"])
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        super().__init__(config)


@register_model(
    name="deberta_v3_base",
    description="DeBERTa-v3 Base - Improved DeBERTa with ELECTRA-Style Pre-Training",
    paper_url="https://arxiv.org/abs/2111.09543",
    pretrained_configs={
        "microsoft": DEBERTA_CONFIGS["v3-base"],
    },
    default_config="microsoft",
    tags=["language", "deberta", "deberta-v3", "transformer", "disentangled"],
    aliases=["deberta-v3-base", "debertav3-base"],
)
class RegisteredDeBERTaV3Base(DeBERTa):
    def __init__(self, **kwargs):
        config = DeBERTaConfig(**DEBERTA_CONFIGS["v3-base"])
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        super().__init__(config)


@register_model(
    name="deberta_v3_large",
    description="DeBERTa-v3 Large - Improved DeBERTa with ELECTRA-Style Pre-Training",
    paper_url="https://arxiv.org/abs/2111.09543",
    pretrained_configs={
        "microsoft": DEBERTA_CONFIGS["v3-large"],
    },
    default_config="microsoft",
    tags=["language", "deberta", "deberta-v3", "transformer", "disentangled"],
    aliases=["deberta-v3-large", "debertav3-large"],
)
class RegisteredDeBERTaV3Large(DeBERTa):
    def __init__(self, **kwargs):
        config = DeBERTaConfig(**DEBERTA_CONFIGS["v3-large"])
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        super().__init__(config)


# Function variants for consistency
def deberta_base(**kwargs):
    return RegisteredDeBERTaBase(**kwargs)


def deberta_large(**kwargs):
    return RegisteredDeBERTaLarge(**kwargs)


def deberta_v3_base(**kwargs):
    return RegisteredDeBERTaV3Base(**kwargs)


def deberta_v3_large(**kwargs):
    return RegisteredDeBERTaV3Large(**kwargs)


# Aliases for compatibility
DeBERTaBase = deberta_base
DeBERTaLarge = deberta_large
DeBERTaV3Base = deberta_v3_base
DeBERTaV3Large = deberta_v3_large
DeBERTaModel = DeBERTa