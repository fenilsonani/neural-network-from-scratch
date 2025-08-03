"""Specific tests for DeBERTa model implementations.

This test suite focuses on DeBERTa-specific features:
- Disentangled attention mechanism
- Relative position embeddings
- Enhanced layer normalization
- Stable dropout
- v3 specific improvements
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.neural_arch.core.tensor import Tensor
    from src.neural_arch.models.language.deberta import (
        DeBERTa,
        DeBERTaConfig,
        DeBERTaLayerNorm,
        StableDropout,
        DisentangledSelfAttention,
        DeBERTaEncoder,
        DeBERTaEmbeddings,
        deberta_base,
        deberta_large,
        deberta_v3_base,
        deberta_v3_large,
        DeBERTaForMaskedLM,
        DeBERTaForSequenceClassification,
        DEBERTA_CONFIGS,
    )
    DEBERTA_AVAILABLE = True
except ImportError:
    DEBERTA_AVAILABLE = False


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestDeBERTaConfig:
    """Test DeBERTa configuration system."""

    def test_default_config(self):
        """Test default DeBERTa configuration."""
        config = DeBERTaConfig()
        
        assert config.vocab_size == 128100
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.intermediate_size == 3072
        assert config.relative_attention is False  # Default
        assert config.position_buckets == 256
        assert config.layer_norm_eps == 1e-7

    def test_base_config(self):
        """Test base model configuration."""
        config = DeBERTaConfig(**DEBERTA_CONFIGS["base"])
        
        assert config.relative_attention is True
        assert config.pos_att_type == "p2c|c2p"
        assert config.position_buckets == 256

    def test_v3_config(self):
        """Test v3 model configuration."""
        config = DeBERTaConfig(**DEBERTA_CONFIGS["v3-base"])
        
        assert hasattr(config, 'norm_rel_ebd')
        assert config.norm_rel_ebd == "layer_norm"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid attention heads
        with pytest.raises(ValueError):
            config = DeBERTaConfig(hidden_size=768, num_attention_heads=7)  # Not divisible
            model = DeBERTa(config)

    def test_custom_config(self):
        """Test custom configuration parameters."""
        config = DeBERTaConfig(
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            relative_attention=True,
            talking_head=True,
            position_buckets=128
        )
        
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 8
        assert config.attention_head_size == 64  # 512 / 8
        assert config.talking_head is True
        assert config.position_buckets == 128


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestDeBERTaLayerNorm:
    """Test DeBERTa-specific layer normalization."""

    def test_layer_norm_initialization(self):
        """Test layer norm initialization."""
        ln = DeBERTaLayerNorm(768)
        
        assert ln.weight.shape == (768,)
        assert ln.bias.shape == (768,)
        assert np.allclose(ln.weight.data, 1.0)
        assert np.allclose(ln.bias.data, 0.0)

    def test_layer_norm_forward(self):
        """Test layer norm forward pass."""
        ln = DeBERTaLayerNorm(4, eps=1e-5)
        
        # Test input
        x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        output = ln(x)
        
        assert output.shape == x.shape
        
        # Check normalization (mean ~0, variance ~1)
        mean = np.mean(output.data, axis=-1)
        var = np.var(output.data, axis=-1)
        
        np.testing.assert_allclose(mean, 0.0, atol=1e-5)
        np.testing.assert_allclose(var, 1.0, rtol=1e-4)

    def test_layer_norm_epsilon(self):
        """Test layer norm epsilon parameter."""
        ln = DeBERTaLayerNorm(4, eps=1e-7)
        assert ln.variance_epsilon == 1e-7


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestStableDropout:
    """Test stable dropout implementation."""

    def test_stable_dropout_training(self):
        """Test stable dropout in training mode."""
        dropout = StableDropout(0.5)
        dropout.training = True
        
        x = Tensor(np.ones((4, 8)))
        output = dropout(x)
        
        assert output.shape == x.shape
        # In training mode with 0.5 dropout, roughly half should be zeroed
        # and the rest scaled by 1/(1-0.5) = 2
        non_zero_mask = output.data != 0
        zero_mask = output.data == 0
        
        # Check scaling of non-zero elements
        if np.any(non_zero_mask):
            non_zero_values = output.data[non_zero_mask]
            expected_values = x.data[non_zero_mask] / (1 - 0.5)
            np.testing.assert_allclose(non_zero_values, expected_values)

    def test_stable_dropout_eval(self):
        """Test stable dropout in eval mode."""
        dropout = StableDropout(0.5)
        dropout.training = False
        
        x = Tensor(np.ones((4, 8)))
        output = dropout(x)
        
        # In eval mode, should be identity
        np.testing.assert_array_equal(output.data, x.data)

    def test_stable_dropout_zero_prob(self):
        """Test stable dropout with zero probability."""
        dropout = StableDropout(0.0)
        dropout.training = True
        
        x = Tensor(np.ones((4, 8)))
        output = dropout(x)
        
        # With 0.0 dropout, should be identity
        np.testing.assert_array_equal(output.data, x.data)


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestDisentangledSelfAttention:
    """Test disentangled self-attention mechanism."""

    def test_attention_initialization(self):
        """Test attention layer initialization."""
        config = DeBERTaConfig(
            hidden_size=768,
            num_attention_heads=12,
            relative_attention=True,
            pos_att_type="p2c|c2p"
        )
        
        attention = DisentangledSelfAttention(config)
        
        assert attention.num_attention_heads == 12
        assert attention.attention_head_size == 64  # 768 / 12
        assert attention.all_head_size == 768
        assert attention.relative_attention is True
        assert "p2c" in attention.pos_att_type
        assert "c2p" in attention.pos_att_type

    def test_transpose_for_scores(self):
        """Test tensor reshaping for multi-head attention."""
        config = DeBERTaConfig(hidden_size=768, num_attention_heads=12)
        attention = DisentangledSelfAttention(config)
        
        batch_size, seq_len = 2, 8
        x = Tensor(np.random.randn(batch_size, seq_len, 768))
        
        reshaped = attention.transpose_for_scores(x, 12)
        expected_shape = (batch_size, 12, seq_len, 64)
        
        assert reshaped.shape == expected_shape

    def test_attention_forward_basic(self):
        """Test basic attention forward pass."""
        config = DeBERTaConfig(
            hidden_size=64,  # Smaller for testing
            num_attention_heads=4,
            relative_attention=False
        )
        
        attention = DisentangledSelfAttention(config)
        
        batch_size, seq_len = 2, 6
        hidden_states = Tensor(np.random.randn(batch_size, seq_len, 64))
        
        outputs = attention(hidden_states)
        context_layer = outputs[0]
        
        assert context_layer.shape == (batch_size, seq_len, 64)

    def test_attention_with_mask(self):
        """Test attention with attention mask."""
        config = DeBERTaConfig(hidden_size=64, num_attention_heads=4)
        attention = DisentangledSelfAttention(config)
        
        batch_size, seq_len = 2, 6
        hidden_states = Tensor(np.random.randn(batch_size, seq_len, 64))
        attention_mask = Tensor(np.ones((batch_size, 1, 1, seq_len)) * -10000.0)
        attention_mask.data[0, 0, 0, -2:] = 0  # Mask last 2 positions for first batch
        
        outputs = attention(hidden_states, attention_mask=attention_mask)
        context_layer = outputs[0]
        
        assert context_layer.shape == (batch_size, seq_len, 64)

    def test_attention_output_attentions(self):
        """Test attention with output_attentions=True."""
        config = DeBERTaConfig(hidden_size=64, num_attention_heads=4)
        attention = DisentangledSelfAttention(config)
        
        batch_size, seq_len = 2, 4
        hidden_states = Tensor(np.random.randn(batch_size, seq_len, 64))
        
        outputs = attention(hidden_states, output_attentions=True)
        context_layer, attention_probs = outputs
        
        assert context_layer.shape == (batch_size, seq_len, 64)
        assert attention_probs.shape == (batch_size, 4, seq_len, seq_len)
        
        # Attention probabilities should sum to 1
        attn_sums = np.sum(attention_probs.data, axis=-1)
        np.testing.assert_allclose(attn_sums, 1.0, rtol=1e-4)


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestDeBERTaEncoder:
    """Test DeBERTa encoder."""

    def test_encoder_initialization(self):
        """Test encoder initialization."""
        config = DeBERTaConfig(num_hidden_layers=6, relative_attention=True)
        encoder = DeBERTaEncoder(config)
        
        assert len(encoder.layer) == 6
        assert encoder.relative_attention is True

    def test_encoder_attention_mask_conversion(self):
        """Test attention mask conversion."""
        config = DeBERTaConfig()
        encoder = DeBERTaEncoder(config)
        
        batch_size, seq_len = 2, 8
        attention_mask = Tensor(np.ones((batch_size, seq_len)))
        attention_mask.data[0, -2:] = 0  # Mask last 2 positions
        
        converted_mask = encoder.get_attention_mask(attention_mask)
        
        # Should be 4D: (batch, 1, seq, seq)
        assert converted_mask.ndim == 4
        assert converted_mask.shape == (batch_size, 1, seq_len, seq_len)

    def test_encoder_relative_embeddings(self):
        """Test relative embeddings generation."""
        config = DeBERTaConfig(relative_attention=True, max_relative_positions=32)
        encoder = DeBERTaEncoder(config)
        
        rel_embeddings = encoder.get_rel_embeddings()
        
        if rel_embeddings is not None:
            # Should have embeddings for both directions
            expected_size = config.max_relative_positions * 2
            assert rel_embeddings.shape[0] == expected_size
            assert rel_embeddings.shape[1] == config.hidden_size

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        config = DeBERTaConfig(num_hidden_layers=3, hidden_size=64)
        encoder = DeBERTaEncoder(config)
        
        batch_size, seq_len = 2, 6
        hidden_states = Tensor(np.random.randn(batch_size, seq_len, 64))
        
        output_hidden_states, all_hidden_states, all_attentions = encoder(
            hidden_states, output_hidden_states=True, output_attentions=True
        )
        
        assert output_hidden_states.shape == (batch_size, seq_len, 64)
        assert len(all_hidden_states) == 4  # 3 layers + input
        assert len(all_attentions) == 3


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestDeBERTaEmbeddings:
    """Test DeBERTa embeddings."""

    def test_embeddings_initialization(self):
        """Test embeddings initialization."""
        config = DeBERTaConfig(vocab_size=1000, hidden_size=64, max_position_embeddings=128)
        embeddings = DeBERTaEmbeddings(config)
        
        assert embeddings.word_embeddings.weight.shape == (1000, 64)
        if embeddings.position_embeddings is not None:
            assert embeddings.position_embeddings.weight.shape == (128, 64)

    def test_embeddings_forward(self):
        """Test embeddings forward pass."""
        config = DeBERTaConfig(vocab_size=1000, hidden_size=64)
        embeddings = DeBERTaEmbeddings(config)
        
        batch_size, seq_len = 2, 8
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_len)))
        
        output = embeddings(input_ids=input_ids)
        
        assert output.shape == (batch_size, seq_len, 64)

    def test_embeddings_position_biased(self):
        """Test position biased embeddings."""
        config = DeBERTaConfig(position_biased_input=True)
        embeddings = DeBERTaEmbeddings(config)
        
        assert embeddings.position_biased_input is True
        assert embeddings.position_embeddings is not None

    def test_embeddings_no_position_bias(self):
        """Test embeddings without position bias."""
        config = DeBERTaConfig(position_biased_input=False)
        embeddings = DeBERTaEmbeddings(config)
        
        assert embeddings.position_biased_input is False
        assert embeddings.position_embeddings is None


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestDeBERTaModelVariants:
    """Test DeBERTa model variants."""

    @pytest.mark.parametrize("model_func,expected_layers", [
        (deberta_base, 12),
        (deberta_large, 24),
        (deberta_v3_base, 12),
        (deberta_v3_large, 24),
    ])
    def test_model_variant_layers(self, model_func, expected_layers):
        """Test model variant layer counts."""
        model = model_func()
        assert model.config.num_hidden_layers == expected_layers

    def test_deberta_forward_pass(self):
        """Test DeBERTa forward pass."""
        model = deberta_base()
        
        batch_size, seq_len = 2, 6
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_len)))
        
        outputs = model(input_ids=input_ids)
        
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, model.config.hidden_size)

    def test_deberta_with_all_outputs(self):
        """Test DeBERTa with all output types."""
        model = deberta_base()
        
        batch_size, seq_len = 2, 4
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_len)))
        
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=True
        )
        
        assert "last_hidden_state" in outputs
        assert "hidden_states" in outputs
        assert "attentions" in outputs
        
        if outputs["hidden_states"] is not None:
            assert len(outputs["hidden_states"]) == model.config.num_hidden_layers + 1
        
        if outputs["attentions"] is not None:
            assert len(outputs["attentions"]) == model.config.num_hidden_layers


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestDeBERTaTaskHeads:
    """Test DeBERTa task-specific heads."""

    def test_masked_lm_head(self):
        """Test DeBERTa masked LM head."""
        model = DeBERTaForMaskedLM()
        
        batch_size, seq_len = 2, 6
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_len)))
        labels = Tensor(np.random.randint(1, 1000, (batch_size, seq_len)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, model.config.vocab_size)

    def test_classification_head(self):
        """Test DeBERTa classification head."""
        num_labels = 3
        model = DeBERTaForSequenceClassification(num_labels=num_labels)
        
        batch_size, seq_len = 2, 6
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_len)))
        labels = Tensor(np.random.randint(0, num_labels, (batch_size,)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, num_labels)

    def test_classification_pooling(self):
        """Test DeBERTa classification pooling strategy."""
        model = DeBERTaForSequenceClassification(num_labels=2)
        
        batch_size, seq_len = 2, 6
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_len)))
        
        outputs = model(input_ids=input_ids)
        
        # Should use first token for pooling
        assert outputs["logits"].shape == (batch_size, 2)


@pytest.mark.skipif(not DEBERTA_AVAILABLE, reason="DeBERTa not available")
class TestDeBERTaFeatureComparison:
    """Test DeBERTa features compared to other models."""

    def test_relative_vs_absolute_position(self):
        """Test relative vs absolute position embeddings."""
        # Test with relative attention
        config_rel = DeBERTaConfig(relative_attention=True, position_biased_input=False)
        model_rel = DeBERTa(config_rel)
        
        # Test with absolute position
        config_abs = DeBERTaConfig(relative_attention=False, position_biased_input=True)
        model_abs = DeBERTa(config_abs)
        
        batch_size, seq_len = 2, 6
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_len)))
        
        outputs_rel = model_rel(input_ids=input_ids)
        outputs_abs = model_abs(input_ids=input_ids)
        
        # Both should work but produce different outputs
        assert outputs_rel["last_hidden_state"].shape == outputs_abs["last_hidden_state"].shape
        
        # Outputs should be different due to different position encoding
        assert not np.allclose(
            outputs_rel["last_hidden_state"].data,
            outputs_abs["last_hidden_state"].data,
            rtol=1e-3
        )

    def test_layer_norm_epsilon_difference(self):
        """Test DeBERTa's different layer norm epsilon."""
        model = deberta_base()
        
        # DeBERTa uses 1e-7 vs BERT's 1e-12
        assert model.config.layer_norm_eps == 1e-7

    def test_vocabulary_size_difference(self):
        """Test DeBERTa's larger vocabulary."""
        model = deberta_base()
        
        # DeBERTa uses much larger vocabulary
        assert model.config.vocab_size == 128100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])