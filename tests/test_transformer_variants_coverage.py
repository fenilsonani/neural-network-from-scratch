"""Comprehensive test coverage for additional transformer variants.

This test suite provides thorough coverage for:
- BERT variants (Base, Large, Cased)
- RoBERTa variants (Base, Large)
- DeBERTa variants (Base, Large, v3)
- Task-specific heads (MLM, Classification)
- Configuration systems
- Error handling and edge cases
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
    from src.neural_arch.backends import set_backend, available_backends
    from src.neural_arch.models.language import (
        # BERT variants
        bert_base,
        bert_large,
        bert_base_cased,
        bert_large_cased,
        BERT,
        BERTForMaskedLM,
        BERTForSequenceClassification,
        # RoBERTa variants
        roberta_base,
        roberta_large,
        RoBERTa,
        RoBERTaForMaskedLM,
        RoBERTaForSequenceClassification,
        # DeBERTa variants
        deberta_base,
        deberta_large,
        deberta_v3_base,
        deberta_v3_large,
        DeBERTa,
        DeBERTaForMaskedLM,
        DeBERTaForSequenceClassification,
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MODELS_AVAILABLE = False


@pytest.fixture(scope="session", autouse=True)
def setup_backend():
    """Set up the best available backend for testing."""
    if not MODELS_AVAILABLE:
        return
    
    try:
        # Try to use MPS if available, otherwise fall back to numpy
        backends = available_backends()
        if "mps" in backends:
            set_backend("mps")
            print("Using MPS backend for tests")
        else:
            set_backend("numpy")
            print("Using numpy backend for tests")
    except Exception as e:
        print(f"Backend setup failed: {e}")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestBERTVariantsComprehensive:
    """Comprehensive tests for BERT model variants."""

    @pytest.mark.parametrize("model_func,expected_config", [
        (bert_base, {"hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 30522}),
        (bert_large, {"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 30522}),
        (bert_base_cased, {"hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 28996}),
        (bert_large_cased, {"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 28996}),
    ])
    def test_bert_variant_configs(self, model_func, expected_config):
        """Test BERT variant configurations."""
        model = model_func()
        config = model.config
        
        for key, expected_value in expected_config.items():
            actual_value = getattr(config, key)
            assert actual_value == expected_value, f"{key}: expected {expected_value}, got {actual_value}"

    def test_bert_parameter_initialization(self):
        """Test BERT parameter initialization."""
        model = bert_base()
        params = model.parameters()
        
        # Check that we have parameters
        assert len(params) > 0
        
        # Check parameter shapes and initialization
        for name, param in params.items():
            if hasattr(param, 'data'):
                # Parameters should be finite
                assert np.all(np.isfinite(param.data)), f"Non-finite values in {name}"
                
                # Parameters should not be all zeros (except some biases)
                if "bias" not in name.lower():
                    assert not np.allclose(param.data, 0.0), f"All-zero parameter {name}"

    def test_bert_attention_mechanism(self):
        """Test BERT attention mechanism."""
        model = bert_base()
        
        # Test with attention outputs
        batch_size, seq_length = 2, 8
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
        
        outputs = model(input_ids=input_ids, output_attentions=True)
        
        assert "attentions" in outputs
        assert outputs["attentions"] is not None
        
        # Should have attention weights for each layer
        attentions = outputs["attentions"]
        assert len(attentions) == model.config.num_hidden_layers

    def test_bert_hidden_states_output(self):
        """Test BERT hidden states output."""
        model = bert_base()
        
        batch_size, seq_length = 2, 6
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
        
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        
        assert "hidden_states" in outputs
        assert outputs["hidden_states"] is not None
        
        # Should have hidden states for each layer + embedding layer
        hidden_states = outputs["hidden_states"]
        expected_layers = model.config.num_hidden_layers + 1
        assert len(hidden_states) == expected_layers

    def test_bert_gradient_flow(self):
        """Test gradient flow through BERT."""
        model = bert_base()
        
        batch_size, seq_length = 2, 4
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)), requires_grad=True)
        
        outputs = model(input_ids=input_ids)
        loss = outputs["last_hidden_state"].sum()
        
        # Should be able to compute loss
        assert loss.requires_grad
        
        # Test backward pass
        try:
            loss.backward()
            # If we get here, gradient computation succeeded
            assert True
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestRoBERTaVariantsComprehensive:
    """Comprehensive tests for RoBERTa model variants."""

    @pytest.mark.parametrize("model_func,expected_config", [
        (roberta_base, {"hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 50265}),
        (roberta_large, {"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 50265}),
    ])
    def test_roberta_variant_configs(self, model_func, expected_config):
        """Test RoBERTa variant configurations."""
        model = model_func()
        config = model.config
        
        for key, expected_value in expected_config.items():
            actual_value = getattr(config, key)
            assert actual_value == expected_value, f"{key}: expected {expected_value}, got {actual_value}"

    def test_roberta_position_embeddings(self):
        """Test RoBERTa position embedding handling."""
        model = roberta_base()
        
        # RoBERTa handles position embeddings differently
        assert hasattr(model.embeddings, 'padding_idx')
        assert model.embeddings.padding_idx == model.config.pad_token_id

    def test_roberta_token_type_embeddings(self):
        """Test RoBERTa token type embedding configuration."""
        model = roberta_base()
        
        # RoBERTa typically uses minimal token type vocab
        assert model.config.type_vocab_size == 1

    def test_roberta_forward_consistency(self):
        """Test RoBERTa forward pass consistency."""
        model = roberta_base()
        
        batch_size, seq_length = 2, 6
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
        
        # Multiple forward passes should be deterministic
        outputs1 = model(input_ids=input_ids)
        outputs2 = model(input_ids=input_ids)
        
        np.testing.assert_allclose(
            outputs1["last_hidden_state"].data,
            outputs2["last_hidden_state"].data,
            rtol=1e-6,
            atol=1e-7
        )


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestDeBERTaVariantsComprehensive:
    """Comprehensive tests for DeBERTa model variants."""

    @pytest.mark.parametrize("model_func,expected_config", [
        (deberta_base, {"hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 128100}),
        (deberta_large, {"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 128100}),
        (deberta_v3_base, {"hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 128100}),
        (deberta_v3_large, {"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 128100}),
    ])
    def test_deberta_variant_configs(self, model_func, expected_config):
        """Test DeBERTa variant configurations."""
        model = model_func()
        config = model.config
        
        for key, expected_value in expected_config.items():
            actual_value = getattr(config, key)
            assert actual_value == expected_value, f"{key}: expected {expected_value}, got {actual_value}"

    def test_deberta_disentangled_attention_config(self):
        """Test DeBERTa disentangled attention configuration."""
        model = deberta_base()
        config = model.config
        
        # DeBERTa-specific features
        assert hasattr(config, 'relative_attention')
        assert hasattr(config, 'position_buckets')
        assert hasattr(config, 'pos_att_type')
        
        # Should be configured for disentangled attention
        assert config.relative_attention is True
        assert config.pos_att_type == "p2c|c2p"
        assert config.position_buckets == 256

    def test_deberta_v3_specific_features(self):
        """Test DeBERTa-v3 specific features."""
        model = deberta_v3_base()
        config = model.config
        
        # v3-specific configuration
        assert hasattr(config, 'norm_rel_ebd')
        assert config.norm_rel_ebd == "layer_norm"

    def test_deberta_layer_norm_epsilon(self):
        """Test DeBERTa layer normalization epsilon."""
        model = deberta_base()
        config = model.config
        
        # DeBERTa uses different layer norm epsilon
        assert config.layer_norm_eps == 1e-7

    def test_deberta_encoder_relative_embeddings(self):
        """Test DeBERTa encoder relative embeddings."""
        model = deberta_base()
        encoder = model.encoder
        
        # Should have relative attention enabled
        assert encoder.relative_attention is True
        
        # Should be able to get relative embeddings
        rel_embeddings = encoder.get_rel_embeddings()
        if rel_embeddings is not None:
            assert rel_embeddings.shape[1] == model.config.hidden_size


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestTaskSpecificHeads:
    """Test task-specific model heads for all variants."""

    @pytest.mark.parametrize("model_class,vocab_size", [
        (BERTForMaskedLM, 30522),
        (RoBERTaForMaskedLM, 50265),
        (DeBERTaForMaskedLM, 128100),
    ])
    def test_masked_lm_heads(self, model_class, vocab_size):
        """Test masked language modeling heads."""
        model = model_class()
        
        batch_size, seq_length = 2, 6
        input_ids = Tensor(np.random.randint(1, vocab_size, (batch_size, seq_length)))
        labels = Tensor(np.random.randint(1, vocab_size, (batch_size, seq_length)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        # Check outputs
        assert "logits" in outputs
        assert "loss" in outputs
        
        # Check shapes
        assert outputs["logits"].shape == (batch_size, seq_length, vocab_size)
        assert outputs["loss"].ndim == 0 or outputs["loss"].size == 1

    @pytest.mark.parametrize("model_class,num_labels", [
        (BERTForSequenceClassification, 2),
        (RoBERTaForSequenceClassification, 3),
        (DeBERTaForSequenceClassification, 5),
    ])
    def test_sequence_classification_heads(self, model_class, num_labels):
        """Test sequence classification heads."""
        model = model_class(num_labels=num_labels)
        
        batch_size, seq_length = 2, 8
        vocab_size = model.config.vocab_size
        input_ids = Tensor(np.random.randint(1, vocab_size, (batch_size, seq_length)))
        labels = Tensor(np.random.randint(0, num_labels, (batch_size,)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        # Check outputs
        assert "logits" in outputs
        assert "loss" in outputs
        
        # Check shapes
        assert outputs["logits"].shape == (batch_size, num_labels)
        assert outputs["loss"].ndim == 0 or outputs["loss"].size == 1


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_input_handling(self):
        """Test handling of edge case inputs."""
        model = bert_base()
        
        # Test minimum sequence length
        batch_size, seq_length = 1, 1
        input_ids = Tensor(np.array([[1]]))  # Single token
        
        outputs = model(input_ids=input_ids)
        assert outputs["last_hidden_state"].shape == (batch_size, seq_length, model.config.hidden_size)

    def test_attention_mask_edge_cases(self):
        """Test attention mask edge cases."""
        model = bert_base()
        
        batch_size, seq_length = 2, 6
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
        
        # Test with all attention masked
        attention_mask = Tensor(np.zeros((batch_size, seq_length)))
        
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Should not crash
            assert outputs["last_hidden_state"].shape == (batch_size, seq_length, model.config.hidden_size)
        except Exception as e:
            # If it fails, should be graceful
            assert "attention" in str(e).lower() or "mask" in str(e).lower()

    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes."""
        model = bert_base()
        
        # Test mismatched input and attention mask shapes
        input_ids = Tensor(np.random.randint(1, 1000, (2, 6)))
        attention_mask = Tensor(np.ones((2, 8)))  # Wrong sequence length
        
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Should either work (with broadcasting) or fail gracefully
        except Exception as e:
            # Should have meaningful error message
            assert len(str(e)) > 0

    def test_large_sequence_lengths(self):
        """Test with sequence lengths at model limits."""
        model = bert_base()
        max_length = model.config.max_position_embeddings
        
        # Test at maximum position embeddings
        batch_size = 1
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, max_length)))
        
        try:
            outputs = model(input_ids=input_ids)
            assert outputs["last_hidden_state"].shape == (batch_size, max_length, model.config.hidden_size)
        except Exception as e:
            # May fail due to memory or implementation limits
            assert "position" in str(e).lower() or "length" in str(e).lower() or "memory" in str(e).lower()


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestModelComparisons:
    """Test comparisons between different model variants."""

    def test_output_shape_consistency(self):
        """Test that all models produce consistent output shapes."""
        models = [
            (bert_base(), 30522),
            (roberta_base(), 50265),
            (deberta_base(), 128100),
        ]
        
        batch_size, seq_length = 2, 8
        
        for model, vocab_size in models:
            input_ids = Tensor(np.random.randint(1, min(vocab_size, 1000), (batch_size, seq_length)))
            outputs = model(input_ids=input_ids)
            
            # All models should produce same shape for last hidden state
            expected_shape = (batch_size, seq_length, model.config.hidden_size)
            assert outputs["last_hidden_state"].shape == expected_shape

    def test_attention_pattern_consistency(self):
        """Test attention pattern consistency across models."""
        models = [bert_base(), roberta_base(), deberta_base()]
        
        batch_size, seq_length = 2, 6
        
        for model in models:
            vocab_size = model.config.vocab_size
            input_ids = Tensor(np.random.randint(1, min(vocab_size, 1000), (batch_size, seq_length)))
            
            outputs = model(input_ids=input_ids, output_attentions=True)
            
            if outputs.get("attentions") is not None:
                attentions = outputs["attentions"]
                
                # Check attention weights are valid probabilities
                for layer_attention in attentions:
                    # Should be non-negative
                    assert np.all(layer_attention.data >= 0)
                    
                    # Should sum to approximately 1 along last dimension
                    attention_sums = np.sum(layer_attention.data, axis=-1)
                    np.testing.assert_allclose(attention_sums, 1.0, rtol=1e-4, atol=1e-5)

    def test_parameter_count_scaling(self):
        """Test that Large models have more parameters than Base models."""
        base_models = [bert_base(), roberta_base(), deberta_base()]
        large_models = [bert_large(), roberta_large(), deberta_large()]
        
        for base_model, large_model in zip(base_models, large_models):
            base_params = len(base_model.parameters())
            large_params = len(large_model.parameters())
            
            # Large models should have more parameters
            assert large_params >= base_params
            
            # Large models should have larger hidden dimensions
            assert large_model.config.hidden_size > base_model.config.hidden_size
            assert large_model.config.num_hidden_layers >= base_model.config.num_hidden_layers


if __name__ == "__main__":
    # Run tests with coverage reporting
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src.neural_arch.models.language",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])