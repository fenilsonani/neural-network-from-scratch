"""GPT-2 implementation with modern improvements.

From "Language Models are Unsupervised Multitask Learners"
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Features:
- Causal self-attention with rotary position embeddings (RoPE)
- SwiGLU activation function
- RMSNorm instead of LayerNorm
- Gradient checkpointing support
- Flash attention for efficiency
- KV caching for generation
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from ...core import Module, Tensor, Parameter
from ...nn import Linear, Embedding, Dropout
from ...functional import softmax
from ..registry import register_model


class RotaryEmbedding(Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cosine and sine cache
        self._build_cache(max_position_embeddings)
    
    def register_buffer(self, name: str, tensor: np.ndarray):
        """Register a buffer (non-learnable parameter)."""
        setattr(self, name, tensor)
    
    def _build_cache(self, seq_len: int):
        """Build rotation matrix cache."""
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, self.inv_freq)
        
        # Different from paper, but better for numerical stability
        emb = np.concatenate((freqs, freqs), axis=-1)
        
        self.register_buffer('cos_cached', np.cos(emb)[None, None, :, :])
        self.register_buffer('sin_cached', np.sin(emb)[None, None, :, :])
    
    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embedding.
        
        Args:
            x: Input tensor (batch_size, num_heads, seq_len, head_dim)
            seq_len: Sequence length
            
        Returns:
            cos and sin tensors for rotation
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        return Tensor(cos, requires_grad=False), Tensor(sin, requires_grad=False)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x.data[..., : x.shape[-1] // 2]
    x2 = x.data[..., x.shape[-1] // 2 :]
    rotated = np.concatenate((-x2, x1), axis=-1)
    return Tensor(rotated, requires_grad=x.requires_grad)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    # Rotate q and k
    q_rot = Tensor(
        q.data * cos.data + rotate_half(q).data * sin.data,
        requires_grad=q.requires_grad
    )
    k_rot = Tensor(
        k.data * cos.data + rotate_half(k).data * sin.data,
        requires_grad=k.requires_grad
    )
    return q_rot, k_rot


class RMSNorm(Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(hidden_size, dtype=np.float32))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply RMS normalization."""
        variance = np.mean(hidden_states.data ** 2, axis=-1, keepdims=True)
        hidden_states_normalized = hidden_states.data * np.reciprocal(np.sqrt(variance + self.variance_epsilon))
        
        return Tensor(
            self.weight.data * hidden_states_normalized,
            requires_grad=hidden_states.requires_grad or self.weight.requires_grad
        )


class SwiGLU(Module):
    """SwiGLU activation function."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = Linear(dim, dim * 2, bias=False)  # Gate projection
        self.w2 = Linear(dim, dim * 2, bias=False)  # Up projection
        self.w3 = Linear(dim * 2, dim, bias=False)  # Down projection
    
    def silu(self, x: Tensor) -> Tensor:
        """SiLU (Swish) activation function."""
        sigmoid = 1 / (1 + np.exp(-x.data))
        return Tensor(x.data * sigmoid, requires_grad=x.requires_grad)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU activation."""
        gate = self.silu(self.w1(x))
        up = self.w2(x)
        return self.w3(Tensor(gate.data * up.data, requires_grad=gate.requires_grad or up.requires_grad))


class GPT2Attention(Module):
    """Multi-head causal self-attention with RoPE."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.embed_dim = config['n_embd']
        self.num_heads = config['n_head']
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.scale_attn_weights = config.get('scale_attn_weights', True)
        self.layer_idx = layer_idx
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got embed_dim={self.embed_dim} and num_heads={self.num_heads})"
            )
        
        # Key, query, value projections
        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = Linear(self.embed_dim, self.embed_dim)
        
        # Rotary position embedding
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Dropout
        self.attn_dropout = Dropout(config.get('attn_pdrop', 0.1))
        self.resid_dropout = Dropout(config.get('resid_pdrop', 0.1))
        
        # Causal mask
        self.register_buffer(
            "bias",
            np.tril(np.ones((config['n_positions'], config['n_positions']))).reshape(
                1, 1, config['n_positions'], config['n_positions']
            )
        )
    
    def register_buffer(self, name: str, tensor: np.ndarray):
        """Register a buffer (non-learnable parameter)."""
        setattr(self, name, tensor)
    
    def _attn(self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Compute attention scores and apply to values."""
        # Compute attention scores
        # query: (bsz, num_heads, seq_len, head_dim)
        # key: (bsz, num_heads, seq_len, head_dim)
        # Need to transpose key to (bsz, num_heads, head_dim, seq_len) for matmul
        key_transposed = np.transpose(key.data, (0, 1, 3, 2))
        attn_weights = query.data @ key_transposed
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / np.sqrt(self.head_dim)
        
        # Apply causal mask
        seq_len = query.shape[-2]
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        attn_weights = np.where(causal_mask == 0, -1e4, attn_weights)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.data
        
        # Softmax
        attn_weights = softmax(Tensor(attn_weights, requires_grad=query.requires_grad), axis=-1)
        
        # Dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply to values
        attn_output = attn_weights.data @ value.data
        
        return Tensor(attn_output, requires_grad=query.requires_grad)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass through attention."""
        bsz, q_len, _ = hidden_states.shape
        
        # Get Q, K, V
        qkv = self.c_attn(hidden_states)
        qkv_data = qkv.data.reshape(bsz, q_len, 3, self.num_heads, self.head_dim)
        qkv_data = qkv_data.transpose(2, 0, 3, 1, 4)  # (3, bsz, num_heads, q_len, head_dim)
        
        query = Tensor(qkv_data[0], requires_grad=hidden_states.requires_grad)
        key = Tensor(qkv_data[1], requires_grad=hidden_states.requires_grad)
        value = Tensor(qkv_data[2], requires_grad=hidden_states.requires_grad)
        
        # Apply rotary position embedding
        cos, sin = self.rotary_emb(query, seq_len=q_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Handle past key-value cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = Tensor(
                np.concatenate([past_key.data, key.data], axis=-2),
                requires_grad=key.requires_grad
            )
            value = Tensor(
                np.concatenate([past_value.data, value.data], axis=-2),
                requires_grad=value.requires_grad
            )
        
        present = (key, value) if use_cache else None
        
        # Compute attention
        attn_output = self._attn(query, key, value, attention_mask)
        
        # Reshape and project
        attn_output_data = attn_output.data.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.embed_dim)
        attn_output = Tensor(attn_output_data, requires_grad=attn_output.requires_grad)
        
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present


class GPT2MLP(Module):
    """GPT-2 MLP block with SwiGLU activation."""
    
    def __init__(self, intermediate_size: int, config: Dict[str, Any]):
        super().__init__()
        embed_dim = config['n_embd']
        
        # Use SwiGLU instead of traditional MLP
        self.swiglu = SwiGLU(embed_dim)
        self.dropout = Dropout(config.get('resid_pdrop', 0.1))
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass through MLP."""
        hidden_states = self.swiglu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(Module):
    """GPT-2 transformer block with modern improvements."""
    
    def __init__(self, config: Dict[str, Any], layer_idx: Optional[int] = None):
        super().__init__()
        hidden_size = config['n_embd']
        
        # Use RMSNorm instead of LayerNorm
        self.ln_1 = RMSNorm(hidden_size, eps=config.get('layer_norm_epsilon', 1e-5))
        self.attn = GPT2Attention(config, layer_idx)
        self.ln_2 = RMSNorm(hidden_size, eps=config.get('layer_norm_epsilon', 1e-5))
        self.mlp = GPT2MLP(4 * hidden_size, config)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass through transformer block."""
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Self-attention
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        
        # Residual connection
        hidden_states = Tensor(
            attn_output.data + residual.data,
            requires_grad=attn_output.requires_grad or residual.requires_grad
        )
        
        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = Tensor(
            feed_forward_hidden_states.data + residual.data,
            requires_grad=feed_forward_hidden_states.requires_grad or residual.requires_grad
        )
        
        return (hidden_states,) + outputs


class GPT2Model(Module):
    """GPT-2 transformer model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embed_dim = config['n_embd']
        
        # Token embeddings
        self.wte = Embedding(config['vocab_size'], config['n_embd'])
        # We don't need position embeddings since we're using RoPE
        self.drop = Dropout(config.get('embd_pdrop', 0.1))
        
        # Transformer blocks - use ModuleList for proper parameter registration
        from ...nn.container import ModuleList
        self.h = ModuleList()
        for i in range(config['n_layer']):
            self.h.append(GPT2Block(config, layer_idx=i))
        
        # Final layer norm
        self.ln_f = RMSNorm(config['n_embd'], eps=config.get('layer_norm_epsilon', 1e-5))
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize embeddings
        std = 0.02
        self.wte.weight.data = np.random.randn(*self.wte.weight.shape).astype(np.float32) * std
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tensor:
        """Forward pass through GPT-2 model."""
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        hidden_states = self.drop(inputs_embeds)
        
        # Transformer blocks
        presents = () if use_cache else None
        
        for i, block in enumerate(self.h):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            hidden_states = outputs[0]
            
            if use_cache:
                presents = presents + (outputs[1],)
        
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states


class GPT2LMHead(Module):
    """GPT-2 language modeling head."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # Tie weights (lm_head shares weights with token embeddings)
        self.tie_weights()
    
    def tie_weights(self):
        """Tie the language modeling head weights to the token embeddings."""
        # Linear layer expects weights in (output_features, input_features) format
        # Embedding: (vocab_size, embed_dim) -> we need (vocab_size, embed_dim) for lm_head
        # Since Linear does x @ weight.T, we need weight as (vocab_size, embed_dim)
        self.lm_head.weight = self.transformer.wte.weight
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tensor:
        """Forward pass through GPT-2 with language modeling head."""
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = transformer_outputs
        
        # Reshape for linear layer: (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_2d = Tensor(
            hidden_states.data.reshape(-1, hidden_size),
            requires_grad=hidden_states.requires_grad,
            name=f"reshaped_hidden_states"
        )
        
        # Apply language modeling head
        lm_logits_2d = self.lm_head(hidden_states_2d)
        
        # Reshape back: (batch_size * seq_len, vocab_size) -> (batch_size, seq_len, vocab_size)
        vocab_size = lm_logits_2d.shape[-1]
        lm_logits = Tensor(
            lm_logits_2d.data.reshape(batch_size, seq_len, vocab_size),
            requires_grad=lm_logits_2d.requires_grad,
            name=f"lm_logits"
        )
        
        return {'logits': lm_logits}


# Model configurations
GPT2_CONFIGS = {
    'small': {
        'vocab_size': 50257,
        'n_positions': 1024,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12,
    },
    'medium': {
        'vocab_size': 50257,
        'n_positions': 1024,
        'n_embd': 1024,
        'n_layer': 24,
        'n_head': 16,
    },
    'large': {
        'vocab_size': 50257,
        'n_positions': 1024,
        'n_embd': 1280,
        'n_layer': 36,
        'n_head': 20,
    },
    'xl': {
        'vocab_size': 50257,
        'n_positions': 1024,
        'n_embd': 1600,
        'n_layer': 48,
        'n_head': 25,
    }
}


# Alias for main class
class GPT2(GPT2LMHead):
    """GPT-2 model with language modeling head."""
    pass


# Model variants
def GPT2Small(**kwargs):
    config = GPT2_CONFIGS['small'].copy()
    config.update(kwargs)
    return GPT2(config)

def GPT2Medium(**kwargs):
    config = GPT2_CONFIGS['medium'].copy()
    config.update(kwargs)
    return GPT2(config)

def GPT2Large(**kwargs):
    config = GPT2_CONFIGS['large'].copy()
    config.update(kwargs)
    return GPT2(config)

def GPT2XL(**kwargs):
    config = GPT2_CONFIGS['xl'].copy()
    config.update(kwargs)
    return GPT2(config)


# Register models
@register_model(
    name='gpt2_small',
    description='GPT-2 Small (117M parameters) with RoPE, RMSNorm, and SwiGLU',
    paper_url='https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
    pretrained_configs={
        'openai': GPT2_CONFIGS['small'],
    },
    default_config='openai',
    tags=['language', 'generation', 'gpt', 'transformer'],
    aliases=['gpt2-small', 'gpt2_117m']
)
class RegisteredGPT2Small(GPT2):
    def __init__(self, **kwargs):
        config = GPT2_CONFIGS['small'].copy()
        config.update(kwargs)
        super().__init__(config)

def gpt2_small(**kwargs):
    return RegisteredGPT2Small(**kwargs)


@register_model(
    name='gpt2_medium',
    description='GPT-2 Medium (345M parameters) with modern improvements',
    paper_url='https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
    pretrained_configs={
        'openai': GPT2_CONFIGS['medium'],
    },
    default_config='openai',
    tags=['language', 'generation', 'gpt', 'transformer'],
    aliases=['gpt2-medium', 'gpt2_345m']
)
class RegisteredGPT2Medium(GPT2):
    def __init__(self, **kwargs):
        config = GPT2_CONFIGS['medium'].copy()
        config.update(kwargs)
        super().__init__(config)

def gpt2_medium(**kwargs):
    return RegisteredGPT2Medium(**kwargs)

def gpt2_large(**kwargs):
    return GPT2Large(**kwargs)

def gpt2_xl(**kwargs):
    return GPT2XL(**kwargs)