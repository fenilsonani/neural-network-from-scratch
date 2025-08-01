"""RoBERTa implementation.

From "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
https://arxiv.org/abs/1907.11692

RoBERTa improves on BERT with:
- Dynamic masking instead of static masking
- Larger batch sizes and learning rates
- Training on more data
- Removing Next Sentence Prediction (NSP) task
- Different tokenization (using BPE)
"""

import numpy as np
from typing import Optional, Dict, Any
from ...core import Module, Tensor
from ...functional import gelu, cross_entropy_loss
from .bert import (
    BERTConfig, BERTEmbeddings, BERTEncoder, BERTPooler,
    BERTSelfAttention, BERTSelfOutput, BERTAttention,
    BERTIntermediate, BERTOutput, BERTLayer
)
from ..registry import register_model


class RoBERTaConfig(BERTConfig):
    """Configuration class for RoBERTa model.
    
    Inherits from BERTConfig but with RoBERTa-specific defaults.
    """
    
    def __init__(
        self,
        vocab_size: int = 50265,  # Different vocab size for RoBERTa
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 514,  # Slightly different for RoBERTa
        type_vocab_size: int = 1,  # RoBERTa doesn't use token type embeddings
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,  # Different epsilon
        pad_token_id: int = 1,  # Different pad token
        bos_token_id: int = 0,  # Beginning of sequence token
        eos_token_id: int = 2,  # End of sequence token
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
        )
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class RoBERTaEmbeddings(BERTEmbeddings):
    """RoBERTa embeddings - similar to BERT but without token type embeddings."""
    
    def __init__(self, config: RoBERTaConfig):
        super().__init__(config)
        # RoBERTa doesn't use token type embeddings in practice
        # but we keep the structure for compatibility
        self.padding_idx = config.pad_token_id
        
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
            # RoBERTa uses different position encoding starting from padding_idx + 1
            position_ids = Tensor(
                np.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=np.int64).reshape(1, -1)
            )
            
        if token_type_ids is None:
            token_type_ids = Tensor(np.zeros(input_shape, dtype=np.int64))
            
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RoBERTa(Module):
    """RoBERTa model for various downstream tasks.
    
    This implementation reuses most of BERT's architecture but with
    RoBERTa-specific improvements and configurations.
    """
    
    def __init__(self, config: Optional[RoBERTaConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = RoBERTaConfig(**kwargs)
        self.config = config
        
        self.embeddings = RoBERTaEmbeddings(config)
        self.encoder = BERTEncoder(config)  # Reuse BERT encoder
        self.pooler = BERTPooler(config)  # Reuse BERT pooler
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using RoBERTa's initialization scheme."""
        # Similar to BERT but with some differences
        for module in self.modules():
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                if hasattr(module, 'weight'):
                    # Initialize linear layers
                    std = self.config.initializer_range
                    if hasattr(module.weight, 'data'):
                        module.weight.data = np.random.normal(0, std, module.weight.shape).astype(np.float32)
                if hasattr(module, 'bias') and module.bias is not None:
                    if hasattr(module.bias, 'data'):
                        module.bias.data = np.zeros(module.bias.shape, dtype=np.float32)
                        
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
        """Forward pass through RoBERTa model."""
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


class RoBERTaForMaskedLM(Module):
    """RoBERTa model for masked language modeling."""
    
    def __init__(self, config: Optional[RoBERTaConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = RoBERTaConfig(**kwargs)
        self.config = config
        
        self.roberta = RoBERTa(config)
        # Reuse BERT's Linear layer for the language modeling head
        from ...nn import Linear
        self.lm_head = Linear(config.hidden_size, config.vocab_size)
        
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        sequence_output = outputs["last_hidden_state"]
        prediction_scores = self.lm_head(sequence_output)
        
        result = {
            "logits": prediction_scores,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }
        
        if labels is not None:
            # Calculate proper masked language modeling loss using cross-entropy
            vocab_size = self.config.vocab_size
            
            # Flatten predictions and labels
            shift_logits = prediction_scores.reshape(-1, vocab_size)
            shift_labels = labels.reshape(-1)
            
            # Compute cross-entropy loss
            masked_lm_loss = cross_entropy_loss(shift_logits, shift_labels)
            result["loss"] = masked_lm_loss
            
        return result


class RoBERTaForSequenceClassification(Module):
    """RoBERTa model for sequence classification."""
    
    def __init__(self, config: Optional[RoBERTaConfig] = None, num_labels: int = 2, **kwargs):
        super().__init__()
        if config is None:
            config = RoBERTaConfig(**kwargs)
        self.config = config
        self.num_labels = num_labels
        
        self.roberta = RoBERTa(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        from ...nn import Dropout, Linear
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
        outputs = self.roberta(
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
            flat_labels = labels.reshape(-1)
            
            # Compute cross-entropy loss
            loss = cross_entropy_loss(logits, flat_labels)
            result["loss"] = loss
            
        return result


# Model configurations
ROBERTA_CONFIGS = {
    'base': {
        'vocab_size': 50265,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'max_position_embeddings': 514,
        'type_vocab_size': 1,
        'layer_norm_eps': 1e-5,
        'pad_token_id': 1,
        'bos_token_id': 0,
        'eos_token_id': 2,
    },
    'large': {
        'vocab_size': 50265,
        'hidden_size': 1024,
        'num_hidden_layers': 24,
        'num_attention_heads': 16,
        'intermediate_size': 4096,
        'max_position_embeddings': 514,
        'type_vocab_size': 1,
        'layer_norm_eps': 1e-5,
        'pad_token_id': 1,
        'bos_token_id': 0,
        'eos_token_id': 2,
    }
}


@register_model(
    name='roberta_base',
    description='RoBERTa Base - A Robustly Optimized BERT Pretraining Approach',
    paper_url='https://arxiv.org/abs/1907.11692',
    pretrained_configs={
        'fairseq': ROBERTA_CONFIGS['base'],
    },
    default_config='fairseq',
    tags=['language', 'roberta', 'bert', 'transformer'],
    aliases=['roberta-base', 'roberta_base_v1']
)
class RegisteredRoBERTaBase(RoBERTa):
    def __init__(self, **kwargs):
        config = RoBERTaConfig(**ROBERTA_CONFIGS['base'])
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        super().__init__(config)


@register_model(
    name='roberta_large',
    description='RoBERTa Large - A Robustly Optimized BERT Pretraining Approach',
    paper_url='https://arxiv.org/abs/1907.11692',
    pretrained_configs={
        'fairseq': ROBERTA_CONFIGS['large'],
    },
    default_config='fairseq',
    tags=['language', 'roberta', 'bert', 'transformer'],
    aliases=['roberta-large', 'roberta_large_v1']
)
class RegisteredRoBERTaLarge(RoBERTa):
    def __init__(self, **kwargs):
        config = RoBERTaConfig(**ROBERTA_CONFIGS['large'])
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        super().__init__(config)


# Function variants for consistency
def roberta_base(**kwargs):
    return RegisteredRoBERTaBase(**kwargs)

def roberta_large(**kwargs):
    return RegisteredRoBERTaLarge(**kwargs)

# Aliases for compatibility
RoBERTaBase = roberta_base
RoBERTaLarge = roberta_large
RoBERTaModel = RoBERTa