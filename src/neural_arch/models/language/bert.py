"""BERT implementation with modern improvements.

From "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
https://arxiv.org/abs/1810.04805
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple

from ...core import Module, Tensor, Parameter
from ...nn import Linear, LayerNorm, Dropout, Embedding
from ...nn.activation import ReLU, Softmax, Tanh
from ...functional import gelu, cross_entropy_loss, softmax, matmul
from ...nn.transformer import MultiHeadAttention
from ..registry import register_model


class BERTConfig:
    """Configuration class for BERT model."""
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
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
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout or hidden_dropout_prob


class BERTEmbeddings(Module):
    """BERT embeddings combining word, position and token_type embeddings."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.pad_token_id = config.pad_token_id
        
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)
        
        self.position_embedding_type = config.position_embedding_type
        
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
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
            
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + token_type_embeddings
        
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(Module):
    """BERT self-attention mechanism."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """Efficiently reshape and transpose tensor for multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, all_head_size)
            
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_size)
        """
        batch_size, seq_len, _ = x.shape
        # Reshape to separate heads: (batch_size, seq_len, num_heads, head_size)
        new_x_shape = (batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        x_reshaped = Tensor(
            x.data.reshape(new_x_shape), 
            requires_grad=x.requires_grad,
            name=f"reshaped({x.name or 'tensor'})"
        )
        
        # Transpose to (batch_size, num_heads, seq_len, head_size) for efficient attention computation
        transposed_data = np.transpose(x_reshaped.data, (0, 2, 1, 3))
        return Tensor(
            transposed_data, 
            requires_grad=x.requires_grad,
            name=f"transposed({x.name or 'tensor'})"
        )
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Optimized forward pass with vectorized attention computation."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Apply linear transformations efficiently
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape and transpose for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Efficient attention computation using proper tensor operations
        # query_layer: (batch_size, num_heads, seq_len, head_size)
        # key_layer: (batch_size, num_heads, seq_len, head_size)
        # Transpose key for dot product: (batch_size, num_heads, head_size, seq_len)
        key_transposed = Tensor(
            np.transpose(key_layer.data, (0, 1, 3, 2)),
            requires_grad=key_layer.requires_grad,
            name=f"key_transposed({key_layer.name or 'tensor'})"
        )
        
        # Compute attention scores using tensor matmul (maintains gradients)
        # Result: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = matmul(query_layer, key_transposed)
        
        # Scale by sqrt(head_size) for numerical stability
        scale_factor = 1.0 / np.sqrt(self.attention_head_size)
        attention_scores = Tensor(
            attention_scores.data * scale_factor,
            requires_grad=attention_scores.requires_grad,
            name=f"scaled_attention({attention_scores.name or 'tensor'})"
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Broadcast mask to match attention scores shape
            from ...functional.arithmetic import add
            attention_scores = add(attention_scores, attention_mask)
            
        # Apply softmax to get attention probabilities
        attention_probs = softmax(attention_scores, axis=-1)
        
        # Apply attention dropout
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values using proper tensor matmul (maintains gradients)
        # attention_probs: (batch_size, num_heads, seq_len, seq_len)
        # value_layer: (batch_size, num_heads, seq_len, head_size)
        # Result: (batch_size, num_heads, seq_len, head_size)
        context_layer = matmul(attention_probs, value_layer)
        
        # Transpose back to (batch_size, seq_len, num_heads, head_size)
        context_transposed = Tensor(
            np.transpose(context_layer.data, (0, 2, 1, 3)),
            requires_grad=context_layer.requires_grad,
            name=f"context_transposed({context_layer.name or 'tensor'})"
        )
        
        # Reshape to (batch_size, seq_len, all_head_size)
        context_final = Tensor(
            context_transposed.data.reshape(batch_size, seq_len, self.all_head_size),
            requires_grad=context_transposed.requires_grad,
            name=f"context_final({context_transposed.name or 'tensor'})"
        )
        
        outputs = (context_final, attention_probs) if output_attentions else (context_final,)
        return outputs


class BERTSelfOutput(Module):
    """BERT self-attention output projection and normalization."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(Module):
    """BERT attention layer combining self-attention and output projection."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BERTIntermediate(Module):
    """BERT intermediate (feed-forward) layer."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size)
        # Use proper GELU activation for better performance
        self.intermediate_act_fn = None  # Will use gelu function directly
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)  # Use proper GELU activation
        return hidden_states


class BERTOutput(Module):
    """BERT output layer for feed-forward network."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(Module):
    """BERT transformer layer."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BERTEncoder(Module):
    """BERT encoder with multiple transformer layers."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        # Use ModuleList for proper parameter registration and gradient flow
        from ...nn.container import ModuleList
        self.layer = ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]], Optional[Tuple[Tensor, ...]]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return hidden_states, all_hidden_states, all_attentions


class BERTPooler(Module):
    """BERT pooler for extracting sentence-level representation."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.activation = Tanh()
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        # Pool the model by taking the hidden state corresponding to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERT(Module):
    """BERT model for various downstream tasks."""
    
    def __init__(self, config: Optional[BERTConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BERTConfig(**kwargs)
        self.config = config
        
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using normal distribution."""
        # This is a simplified weight initialization
        for module in self.modules():
            if isinstance(module, Linear):
                # Initialize linear layers
                std = self.config.initializer_range
                module.weight.data = np.random.normal(0, std, module.weight.shape).astype(np.float32)
                if module.bias is not None:
                    module.bias.data = np.zeros(module.bias.shape, dtype=np.float32)
            elif isinstance(module, Embedding):
                # Initialize embedding layers
                std = self.config.initializer_range
                module.weight.data = np.random.normal(0, std, module.weight.shape).astype(np.float32)
                
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
            
        # Convert attention mask to proper format for scaled dot-product attention
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        
        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs[1] if output_hidden_states else None,
            "attentions": encoder_outputs[2] if output_attentions else None,
        }


class BERTForMaskedLM(Module):
    """BERT model for masked language modeling."""
    
    def __init__(self, config: Optional[BERTConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BERTConfig(**kwargs)
        self.config = config
        
        self.bert = BERT(config)
        self.cls = Linear(config.hidden_size, config.vocab_size)
        
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        sequence_output = outputs["last_hidden_state"]
        prediction_scores = self.cls(sequence_output)
        
        result = {
            "logits": prediction_scores,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }
        
        if labels is not None:
            # Calculate proper masked language modeling loss using cross-entropy
            # Reshape predictions and labels for loss computation
            vocab_size = self.config.vocab_size
            
            # Flatten predictions and labels
            shift_logits = prediction_scores.reshape(-1, vocab_size)
            shift_labels = labels.reshape(-1)
            
            # Only compute loss on masked tokens (labels != -100)
            # For simplicity, we'll compute loss on all tokens for now
            masked_lm_loss = cross_entropy_loss(shift_logits, shift_labels)
            result["loss"] = masked_lm_loss
            
        return result


class BERTForSequenceClassification(Module):
    """BERT model for sequence classification."""
    
    def __init__(self, config: Optional[BERTConfig] = None, num_labels: int = 2, **kwargs):
        super().__init__()
        if config is None:
            config = BERTConfig(**kwargs)
        self.config = config
        self.num_labels = num_labels
        
        self.bert = BERT(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Linear(config.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        pooled_output = outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {
            "logits": logits,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }
        
        if labels is not None:
            # Calculate proper classification loss using cross-entropy
            # Flatten labels for loss computation
            flat_labels = labels.reshape(-1)
            
            # Compute cross-entropy loss
            loss = cross_entropy_loss(logits, flat_labels)
            result["loss"] = loss
            
        return result


@register_model(
    name='bert_base',
    description='BERT Base model with bidirectional attention',
    paper_url='https://arxiv.org/abs/1810.04805',
    pretrained_configs={'uncased': {'vocab_size': 30522, 'hidden_size': 768}},
    default_config='uncased',
    tags=['language', 'bert', 'transformer'],
    aliases=['bert-base-uncased']
)
class RegisteredBERTBase(BERT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def bert_base(**kwargs):
    return RegisteredBERTBase(**kwargs)

BERTBase = bert_base
BERTLarge = lambda **kwargs: BERT(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, **kwargs)
BERTModel = BERT

# Function variants for consistency
def bert_large(**kwargs):
    return BERT(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, **kwargs)