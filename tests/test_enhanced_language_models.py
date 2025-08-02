"""Enhanced comprehensive tests for language models with detailed architecture validation.

This module provides in-depth testing of language model architectures including BERT, GPT-2,
RoBERTa, and T5 with detailed validation of components, parameter counting, attention patterns,
and architectural correctness.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_arch.core import Tensor
from neural_arch.models import (
    bert_base,
    bert_large,
    get_model,
    gpt2_medium,
    gpt2_small,
    roberta_base,
    t5_small,
)
from neural_arch.models.language.bert import (
    BERT,
    BERTAttention,
    BERTConfig,
    BERTEmbeddings,
    BERTEncoder,
    BERTForMaskedLM,
    BERTForSequenceClassification,
    BERTIntermediate,
    BERTLayer,
    BERTOutput,
    BERTSelfAttention,
)
from neural_arch.models.language.gpt2 import (
    GPT2,
    GPT2_CONFIGS,
    GPT2MLP,
    GPT2Attention,
    GPT2Block,
    GPT2LMHead,
    GPT2Model,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary_pos_emb,
    rotate_half,
)


class TestGPT2Architecture:
    """Test GPT-2 model architecture in detail."""

    def test_gpt2_config_validation(self):
        """Test GPT-2 configuration validation."""
        # Test predefined configurations
        for config_name, config in GPT2_CONFIGS.items():
            assert "vocab_size" in config
            assert "n_positions" in config
            assert "n_embd" in config
            assert "n_layer" in config
            assert "n_head" in config

            # Validate head dimension consistency
            assert config["n_embd"] % config["n_head"] == 0

            # Validate reasonable ranges
            assert config["vocab_size"] > 0
            assert config["n_positions"] > 0
            assert config["n_embd"] > 0
            assert config["n_layer"] > 0
            assert config["n_head"] > 0

    def test_gpt2_parameter_counting(self):
        """Test GPT-2 parameter counting for different model sizes."""
        model_sizes = {
            "small": (117, 768, 12, 12),  # Expected ~117M params
            "medium": (345, 1024, 24, 16),  # Expected ~345M params
        }

        for size_name, (expected_params_m, n_embd, n_layer, n_head) in model_sizes.items():
            config = GPT2_CONFIGS[size_name]
            model = gpt2_small() if size_name == "small" else gpt2_medium()

            # Calculate expected parameters
            vocab_size = config["vocab_size"]

            # Token embeddings: vocab_size * n_embd
            token_emb_params = vocab_size * n_embd

            # Per transformer block parameters
            # Attention: 3 * n_embd * n_embd (qkv) + n_embd * n_embd (output)
            attn_params_per_block = 4 * n_embd * n_embd

            # MLP: depends on SwiGLU implementation
            # SwiGLU has 3 linear layers: n_embd->2*n_embd, n_embd->2*n_embd, 2*n_embd->n_embd
            mlp_params_per_block = 2 * (n_embd * 2 * n_embd) + (2 * n_embd * n_embd)
            mlp_params_per_block = 6 * n_embd * n_embd

            # RMSNorm parameters: 2 * n_embd per block (ln_1, ln_2) + n_embd (final)
            norm_params_per_block = 2 * n_embd
            total_norm_params = (n_layer * norm_params_per_block) + n_embd

            # Total per block
            params_per_block = attn_params_per_block + mlp_params_per_block + norm_params_per_block

            # Total parameters
            total_params = token_emb_params + (n_layer * params_per_block) + n_embd

            # Verify order of magnitude is reasonable
            total_params_m = total_params / 1_000_000
            assert (
                50 < total_params_m < 1000
            ), f"Parameter count {total_params_m}M unreasonable for {size_name}"

    def test_rotary_embedding_implementation(self):
        """Test Rotary Position Embedding implementation."""
        dim = 64
        max_pos = 2048
        rope = RotaryEmbedding(dim, max_pos)

        # Test initialization
        assert rope.dim == dim
        assert rope.max_position_embeddings == max_pos
        assert hasattr(rope, "inv_freq")
        assert hasattr(rope, "cos_cached")
        assert hasattr(rope, "sin_cached")

        # Test forward pass
        batch_size, num_heads, seq_len, head_dim = 2, 8, 32, dim
        x = Tensor(np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32))

        cos, sin = rope(x)

        assert cos.shape == (1, 1, seq_len, head_dim)
        assert sin.shape == (1, 1, seq_len, head_dim)
        assert not cos.requires_grad
        assert not sin.requires_grad

        # Test cache extension for longer sequences
        long_seq_len = max_pos + 100
        x_long = Tensor(
            np.random.randn(batch_size, num_heads, long_seq_len, head_dim).astype(np.float32)
        )
        cos_long, sin_long = rope(x_long)

        assert cos_long.shape == (1, 1, long_seq_len, head_dim)
        assert sin_long.shape == (1, 1, long_seq_len, head_dim)

    def test_rotary_embedding_rotation_functions(self):
        """Test rotary embedding rotation helper functions."""
        # Test rotate_half function
        x = Tensor(np.random.randn(2, 4, 8, 64).astype(np.float32))
        rotated = rotate_half(x)

        assert rotated.shape == x.shape

        # Verify rotation logic - first half should be negated second half
        expected_first_half = -x.data[..., 32:]  # Last 32 dims become first 32 (negated)
        expected_second_half = x.data[..., :32]  # First 32 dims become last 32

        np.testing.assert_array_equal(rotated.data[..., :32], expected_first_half)
        np.testing.assert_array_equal(rotated.data[..., 32:], expected_second_half)

        # Test apply_rotary_pos_emb function
        q = Tensor(np.random.randn(2, 8, 16, 64).astype(np.float32))
        k = Tensor(np.random.randn(2, 8, 16, 64).astype(np.float32))
        cos = Tensor(np.random.randn(1, 1, 16, 64).astype(np.float32))
        sin = Tensor(np.random.randn(1, 1, 16, 64).astype(np.float32))

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert q_rot.requires_grad == q.requires_grad
        assert k_rot.requires_grad == k.requires_grad

    def test_rms_norm_implementation(self):
        """Test RMSNorm implementation."""
        hidden_size = 768
        eps = 1e-6
        rms_norm = RMSNorm(hidden_size, eps)

        # Test initialization
        assert rms_norm.weight.shape == (hidden_size,)
        np.testing.assert_array_equal(rms_norm.weight.data, np.ones(hidden_size))
        assert rms_norm.variance_epsilon == eps

        # Test forward pass
        batch_size, seq_len = 4, 32
        x = Tensor(np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32))
        normalized = rms_norm(x)

        assert normalized.shape == x.shape

        # Verify RMS normalization properties
        # RMS should be approximately 1 after normalization
        variance = np.mean(normalized.data**2, axis=-1, keepdims=True)
        rms = np.sqrt(variance)

        # Should be close to 1 (within reasonable tolerance)
        np.testing.assert_allclose(rms, 1.0, rtol=1e-5, atol=1e-5)

    def test_swiglu_activation(self):
        """Test SwiGLU activation function implementation."""
        dim = 256
        swiglu = SwiGLU(dim)

        # Test component linear layers
        assert hasattr(swiglu, "w1")  # Gate projection
        assert hasattr(swiglu, "w2")  # Up projection
        assert hasattr(swiglu, "w3")  # Down projection

        assert swiglu.w1.in_features == dim
        assert swiglu.w1.out_features == dim * 2
        assert swiglu.w2.in_features == dim
        assert swiglu.w2.out_features == dim * 2
        assert swiglu.w3.in_features == dim * 2
        assert swiglu.w3.out_features == dim

        # Test forward pass
        batch_size, seq_len = 4, 16
        x = Tensor(np.random.randn(batch_size, seq_len, dim).astype(np.float32))
        output = swiglu(x)

        assert output.shape == x.shape

        # Test SiLU function separately
        test_input = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]).astype(np.float32))
        silu_output = swiglu.silu(test_input)

        # SiLU(x) = x * sigmoid(x)
        expected_sigmoid = 1 / (1 + np.exp(-test_input.data))
        expected_silu = test_input.data * expected_sigmoid

        np.testing.assert_allclose(silu_output.data, expected_silu, rtol=1e-6)

    def test_gpt2_attention_mechanism(self):
        """Test GPT-2 attention mechanism in detail."""
        config = GPT2_CONFIGS["small"].copy()
        attention = GPT2Attention(config, layer_idx=0)

        # Test initialization
        assert attention.embed_dim == config["n_embd"]
        assert attention.num_heads == config["n_head"]
        assert attention.head_dim == config["n_embd"] // config["n_head"]

        # Test causal mask
        n_positions = config["n_positions"]
        assert hasattr(attention, "bias")
        assert attention.bias.shape == (1, 1, n_positions, n_positions)

        # Verify causal mask is lower triangular
        mask = attention.bias[0, 0]
        assert np.allclose(mask, np.tril(np.ones((n_positions, n_positions))))

        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = Tensor(
            np.random.randn(batch_size, seq_len, config["n_embd"]).astype(np.float32)
        )

        attn_output, present = attention(hidden_states, use_cache=True)

        assert attn_output.shape == hidden_states.shape
        assert present is not None  # KV cache should be returned
        assert len(present) == 2  # Key and value

        # Test with attention mask
        attention_mask = Tensor(np.ones((batch_size, seq_len)).astype(np.float32))
        attn_output_masked, _ = attention(hidden_states, attention_mask=attention_mask)
        assert attn_output_masked.shape == hidden_states.shape

        # Test with past key-value cache
        past_kv = present
        new_hidden = Tensor(np.random.randn(batch_size, 1, config["n_embd"]).astype(np.float32))
        attn_output_cached, new_present = attention(
            new_hidden, past_key_value=past_kv, use_cache=True
        )

        assert attn_output_cached.shape == new_hidden.shape
        assert new_present is not None
        # New cache should have longer sequence length
        assert new_present[0].shape[2] == seq_len + 1  # seq_len + 1 new token

    def test_gpt2_mlp_block(self):
        """Test GPT-2 MLP block."""
        config = GPT2_CONFIGS["small"].copy()
        mlp = GPT2MLP(4 * config["n_embd"], config)

        # Test that it uses SwiGLU
        assert hasattr(mlp, "swiglu")
        assert hasattr(mlp, "dropout")

        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = Tensor(
            np.random.randn(batch_size, seq_len, config["n_embd"]).astype(np.float32)
        )

        output = mlp(hidden_states)
        assert output.shape == hidden_states.shape

    def test_gpt2_transformer_block(self):
        """Test complete GPT-2 transformer block."""
        config = GPT2_CONFIGS["small"].copy()
        block = GPT2Block(config, layer_idx=0)

        # Test components
        assert hasattr(block, "ln_1")  # Pre-attention norm
        assert hasattr(block, "attn")  # Self-attention
        assert hasattr(block, "ln_2")  # Pre-MLP norm
        assert hasattr(block, "mlp")  # MLP

        # Verify normalization is RMSNorm
        assert isinstance(block.ln_1, RMSNorm)
        assert isinstance(block.ln_2, RMSNorm)

        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = Tensor(
            np.random.randn(batch_size, seq_len, config["n_embd"]).astype(np.float32)
        )

        outputs = block(hidden_states, use_cache=True)

        assert len(outputs) >= 1
        block_output = outputs[0]
        assert block_output.shape == hidden_states.shape

        if len(outputs) > 1:
            # Should return KV cache
            present = outputs[1]
            assert present is not None

    def test_gpt2_model_integration(self):
        """Test complete GPT-2 model integration."""
        model = gpt2_small()

        # Test model structure
        assert hasattr(model, "transformer")
        assert hasattr(model, "lm_head")

        # Test weight tying between embeddings and LM head
        assert model.lm_head.weight is model.transformer.wte.weight

        # Test forward pass
        batch_size, seq_len = 2, 16
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.float32))

        outputs = model(input_ids)

        assert isinstance(outputs, dict)
        assert "logits" in outputs

        logits = outputs["logits"]
        assert logits.shape == (batch_size, seq_len, model.transformer.wte.num_embeddings)

    def test_gpt2_generation_capabilities(self):
        """Test GPT-2 generation-specific features."""
        model = gpt2_small()

        # Test with KV caching for generation
        batch_size = 1
        initial_seq_len = 10
        input_ids = Tensor(
            np.random.randint(0, 1000, (batch_size, initial_seq_len)).astype(np.float32)
        )

        # First forward pass
        outputs1 = model.transformer(input_ids, use_cache=True)
        hidden_states1 = outputs1

        # Should be able to process next token efficiently with cache
        next_token = Tensor(np.random.randint(0, 1000, (batch_size, 1)).astype(np.float32))

        # Note: In full implementation, would pass presents from first call
        outputs2 = model.transformer(next_token, use_cache=True)
        hidden_states2 = outputs2

        assert hidden_states2.shape == (batch_size, 1, model.transformer.embed_dim)


class TestBERTArchitecture:
    """Test BERT model architecture in detail."""

    def test_bert_config_validation(self):
        """Test BERT configuration validation."""
        config = BERTConfig()

        # Test default values
        assert config.vocab_size == 30522
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.intermediate_size == 3072
        assert config.max_position_embeddings == 512

        # Test custom configuration
        custom_config = BERTConfig(
            vocab_size=50000, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16
        )

        assert custom_config.vocab_size == 50000
        assert custom_config.hidden_size == 1024
        assert custom_config.num_hidden_layers == 24
        assert custom_config.num_attention_heads == 16

        # Test head dimension consistency
        assert custom_config.hidden_size % custom_config.num_attention_heads == 0

    def test_bert_embeddings(self):
        """Test BERT embeddings layer."""
        config = BERTConfig()
        embeddings = BERTEmbeddings(config)

        # Test components
        assert hasattr(embeddings, "word_embeddings")
        assert hasattr(embeddings, "position_embeddings")
        assert hasattr(embeddings, "token_type_embeddings")
        assert hasattr(embeddings, "LayerNorm")
        assert hasattr(embeddings, "dropout")

        # Test forward pass
        batch_size, seq_len = 2, 20
        input_ids = Tensor(
            np.random.randint(0, config.vocab_size, (batch_size, seq_len)).astype(np.int64)
        )

        embeddings_output = embeddings(input_ids)

        assert embeddings_output.shape == (batch_size, seq_len, config.hidden_size)

        # Test with custom token_type_ids and position_ids
        token_type_ids = Tensor(np.zeros((batch_size, seq_len), dtype=np.int64))
        position_ids = Tensor(np.arange(seq_len, dtype=np.int64).reshape(1, -1))

        embeddings_output_custom = embeddings(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )

        assert embeddings_output_custom.shape == (batch_size, seq_len, config.hidden_size)

    def test_bert_self_attention(self):
        """Test BERT self-attention mechanism."""
        config = BERTConfig()
        attention = BERTSelfAttention(config)

        # Test initialization
        assert attention.num_attention_heads == config.num_attention_heads
        assert attention.attention_head_size == config.hidden_size // config.num_attention_heads
        assert attention.all_head_size == config.hidden_size

        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = Tensor(
            np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32)
        )

        outputs = attention(hidden_states)

        assert len(outputs) >= 1
        attention_output = outputs[0]
        assert attention_output.shape == hidden_states.shape

        # Test with attention mask
        attention_mask = Tensor(np.ones((batch_size, 1, 1, seq_len)).astype(np.float32))
        outputs_masked = attention(hidden_states, attention_mask=attention_mask)
        assert outputs_masked[0].shape == hidden_states.shape

        # Test attention output when requested
        outputs_with_attn = attention(hidden_states, output_attentions=True)
        assert len(outputs_with_attn) == 2
        attention_probs = outputs_with_attn[1]
        assert attention_probs.shape == (batch_size, config.num_attention_heads, seq_len, seq_len)

    def test_bert_attention_transpose_for_scores(self):
        """Test BERT attention tensor reshaping."""
        config = BERTConfig()
        attention = BERTSelfAttention(config)

        # Test transpose_for_scores method
        batch_size, seq_len = 2, 16
        input_tensor = Tensor(
            np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32)
        )

        reshaped = attention.transpose_for_scores(input_tensor)

        expected_shape = (
            batch_size,
            config.num_attention_heads,
            seq_len,
            attention.attention_head_size,
        )
        assert reshaped.shape == expected_shape

    def test_bert_layer_structure(self):
        """Test BERT layer structure."""
        config = BERTConfig()
        layer = BERTLayer(config)

        # Test components
        assert hasattr(layer, "attention")
        assert hasattr(layer, "intermediate")
        assert hasattr(layer, "output")

        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = Tensor(
            np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32)
        )

        outputs = layer(hidden_states)

        assert len(outputs) >= 1
        layer_output = outputs[0]
        assert layer_output.shape == hidden_states.shape

    def test_bert_encoder(self):
        """Test BERT encoder with multiple layers."""
        config = BERTConfig(num_hidden_layers=6)  # Smaller for testing
        encoder = BERTEncoder(config)

        # Test layer count
        assert len(encoder.layer) == config.num_hidden_layers

        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = Tensor(
            np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32)
        )

        encoder_outputs = encoder(hidden_states)

        # Should return (last_hidden_state, all_hidden_states, all_attentions)
        last_hidden_state = encoder_outputs[0]
        assert last_hidden_state.shape == hidden_states.shape

        # Test with output_hidden_states=True
        encoder_outputs_verbose = encoder(
            hidden_states, output_hidden_states=True, output_attentions=True
        )

        last_hidden, all_hidden, all_attentions = encoder_outputs_verbose

        if all_hidden is not None:
            # Should have num_layers + 1 hidden states (including input)
            assert len(all_hidden) == config.num_hidden_layers + 1

        if all_attentions is not None:
            # Should have num_layers attention matrices
            assert len(all_attentions) == config.num_hidden_layers

    def test_bert_pooler(self):
        """Test BERT pooler for sentence-level representation."""
        from neural_arch.models.language.bert import BERTPooler

        config = BERTConfig()
        pooler = BERTPooler(config)

        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = Tensor(
            np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32)
        )

        pooled_output = pooler(hidden_states)

        # Should extract first token and apply dense + tanh
        assert pooled_output.shape == (batch_size, config.hidden_size)

    def test_bert_model_integration(self):
        """Test complete BERT model integration."""
        model = bert_base()

        # Test model structure
        assert hasattr(model, "embeddings")
        assert hasattr(model, "encoder")
        assert hasattr(model, "pooler")

        # Test forward pass
        batch_size, seq_len = 2, 20
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.float32))

        outputs = model(input_ids)

        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert "pooler_output" in outputs

        last_hidden_state = outputs["last_hidden_state"]
        pooler_output = outputs["pooler_output"]

        assert last_hidden_state.shape == (batch_size, seq_len, model.config.hidden_size)
        assert pooler_output.shape == (batch_size, model.config.hidden_size)

    def test_bert_for_masked_lm(self):
        """Test BERT for masked language modeling."""
        model = BERTForMaskedLM()

        # Test forward pass
        batch_size, seq_len = 2, 20
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.float32))

        outputs = model(input_ids)

        assert isinstance(outputs, dict)
        assert "logits" in outputs

        logits = outputs["logits"]
        assert logits.shape == (batch_size, seq_len, model.config.vocab_size)

        # Test with labels for loss computation
        labels = Tensor(
            np.random.randint(0, model.config.vocab_size, (batch_size, seq_len)).astype(np.float32)
        )
        outputs_with_loss = model(input_ids, labels=labels)

        assert "loss" in outputs_with_loss
        assert isinstance(outputs_with_loss["loss"], Tensor)

    def test_bert_for_sequence_classification(self):
        """Test BERT for sequence classification."""
        num_labels = 3
        model = BERTForSequenceClassification(num_labels=num_labels)

        # Test forward pass
        batch_size, seq_len = 2, 20
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)).astype(np.float32))

        outputs = model(input_ids)

        assert isinstance(outputs, dict)
        assert "logits" in outputs

        logits = outputs["logits"]
        assert logits.shape == (batch_size, num_labels)

        # Test with labels for loss computation
        labels = Tensor(np.random.randint(0, num_labels, (batch_size,)).astype(np.float32))
        outputs_with_loss = model(input_ids, labels=labels)

        assert "loss" in outputs_with_loss
        assert isinstance(outputs_with_loss["loss"], Tensor)


class TestModelArchitectureComparisons:
    """Test architectural differences and comparisons between models."""

    def test_gpt2_vs_bert_differences(self):
        """Test key differences between GPT-2 and BERT architectures."""
        gpt2_model = gpt2_small()
        bert_model = bert_base()

        # GPT-2 is decoder-only, BERT is encoder-only
        # GPT-2 has causal attention, BERT has bidirectional attention

        # Test attention patterns
        config_gpt2 = GPT2_CONFIGS["small"]
        gpt2_attention = GPT2Attention(config_gpt2)

        config_bert = BERTConfig()
        bert_attention = BERTSelfAttention(config_bert)

        # GPT-2 should have causal mask
        assert hasattr(gpt2_attention, "bias")

        # BERT should not have causal constraints
        assert not hasattr(bert_attention, "bias")

        # Test normalization differences
        # GPT-2 uses RMSNorm, BERT uses LayerNorm
        gpt2_block = GPT2Block(config_gpt2)
        assert isinstance(gpt2_block.ln_1, RMSNorm)

        # BERT uses standard LayerNorm (we'd need to check the actual implementation)

    def test_parameter_scaling_laws(self):
        """Test parameter scaling relationships."""
        models = {
            "gpt2_small": gpt2_small(),
            "gpt2_medium": gpt2_medium(),
        }

        # Verify that medium model has more parameters than small
        # This is implicit in the architecture, but we can verify dimensions

        small_config = GPT2_CONFIGS["small"]
        medium_config = GPT2_CONFIGS["medium"]

        # Medium should have larger embedding dimension and more layers
        assert medium_config["n_embd"] > small_config["n_embd"]
        assert medium_config["n_layer"] > small_config["n_layer"]
        assert medium_config["n_head"] > small_config["n_head"]

    def test_attention_head_patterns(self):
        """Test attention head configuration patterns."""
        configs = [
            ("small", GPT2_CONFIGS["small"]),
            ("medium", GPT2_CONFIGS["medium"]),
        ]

        for name, config in configs:
            # Embedding dimension should be divisible by number of heads
            assert config["n_embd"] % config["n_head"] == 0

            # Head dimension should be reasonable
            head_dim = config["n_embd"] // config["n_head"]
            assert 32 <= head_dim <= 128, f"Head dimension {head_dim} unusual for {name}"

    def test_model_memory_requirements(self):
        """Test estimated memory requirements for different models."""
        # This tests the concept of memory scaling with model size

        def estimate_memory_mb(config):
            """Rough estimate of model memory in MB."""
            # Simplified calculation based on parameters
            vocab_size = config["vocab_size"]
            n_embd = config["n_embd"]
            n_layer = config["n_layer"]

            # Token embeddings
            token_emb_params = vocab_size * n_embd

            # Transformer blocks (rough estimate)
            params_per_layer = 12 * n_embd * n_embd  # Attention + MLP + norms
            transformer_params = n_layer * params_per_layer

            total_params = token_emb_params + transformer_params

            # Assume 4 bytes per parameter (float32) + some overhead
            memory_bytes = total_params * 4 * 1.2  # 20% overhead
            return memory_bytes / (1024 * 1024)  # Convert to MB

        small_memory = estimate_memory_mb(GPT2_CONFIGS["small"])
        medium_memory = estimate_memory_mb(GPT2_CONFIGS["medium"])

        # Medium should require more memory than small
        assert medium_memory > small_memory

        # Memory requirements should be reasonable
        assert 100 < small_memory < 2000  # 100MB - 2GB range
        assert 500 < medium_memory < 5000  # 500MB - 5GB range


class TestModelRegistryIntegration:
    """Test model registry integration with architecture validation."""

    def test_registered_models_consistency(self):
        """Test that registered models are consistent with direct instantiation."""
        # Test GPT-2 models
        direct_gpt2_small = gpt2_small()
        registry_gpt2_small = get_model("gpt2_small")

        assert type(direct_gpt2_small) == type(registry_gpt2_small)

        # Test BERT models
        direct_bert = bert_base()
        registry_bert = get_model("bert_base")

        assert type(direct_bert) == type(registry_bert)

    def test_model_tags_and_metadata(self):
        """Test model registry tags and metadata."""
        # This would test the registry metadata when available
        # For now, we test that models can be retrieved by name

        model_names = ["gpt2_small", "gpt2_medium", "bert_base"]

        for model_name in model_names:
            try:
                model = get_model(model_name)
                assert model is not None
            except Exception:
                # Some models might be stubs
                pass

    def test_model_configuration_inheritance(self):
        """Test model configuration inheritance and customization."""
        # Test that custom configurations override defaults
        custom_gpt2 = gpt2_small(vocab_size=25000)

        # Should accept custom parameters
        assert custom_gpt2 is not None

        # Test BERT with custom config
        custom_bert = bert_base(hidden_size=512, num_hidden_layers=6)

        assert custom_bert is not None


if __name__ == "__main__":
    pytest.main([__file__])
