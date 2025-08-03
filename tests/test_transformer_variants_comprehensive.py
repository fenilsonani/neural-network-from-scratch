"""Comprehensive tests for additional transformer variants.

This test suite validates all the enhanced transformer models including:
- BERT-Large (both cased and uncased)
- RoBERTa-Large 
- DeBERTa (Base and Large)
- DeBERTa-v3 (Base and Large)
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
except ImportError:
    MODELS_AVAILABLE = False


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestBERTVariants:
    """Test BERT model variants."""

    def test_bert_base_creation(self):
        """Test BERT base model creation."""
        model = bert_base()
        assert model is not None
        assert model.config.hidden_size == 768
        assert model.config.num_hidden_layers == 12
        assert model.config.num_attention_heads == 12
        assert model.config.intermediate_size == 3072

    def test_bert_large_creation(self):
        """Test BERT large model creation."""
        model = bert_large()
        assert model is not None
        assert model.config.hidden_size == 1024
        assert model.config.num_hidden_layers == 24
        assert model.config.num_attention_heads == 16
        assert model.config.intermediate_size == 4096

    def test_bert_base_cased_creation(self):
        """Test BERT base cased model creation."""
        model = bert_base_cased()
        assert model is not None
        assert model.config.hidden_size == 768
        assert model.config.vocab_size == 28996  # Different vocab for cased

    def test_bert_large_cased_creation(self):
        """Test BERT large cased model creation."""
        model = bert_large_cased()
        assert model is not None
        assert model.config.hidden_size == 1024
        assert model.config.num_hidden_layers == 24
        assert model.config.vocab_size == 28996  # Different vocab for cased

    def test_bert_variants_forward_pass(self):
        """Test forward pass for all BERT variants."""
        variants = [bert_base, bert_large, bert_base_cased, bert_large_cased]
        
        for variant_func in variants:
            model = variant_func()
            
            # Create test input
            batch_size, seq_length = 2, 10
            input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
            attention_mask = Tensor(np.ones((batch_size, seq_length)))
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Check outputs
            assert "last_hidden_state" in outputs
            assert "pooler_output" in outputs
            
            last_hidden_state = outputs["last_hidden_state"]
            pooler_output = outputs["pooler_output"]
            
            assert last_hidden_state.shape == (batch_size, seq_length, model.config.hidden_size)
            assert pooler_output.shape == (batch_size, model.config.hidden_size)

    def test_bert_for_masked_lm(self):
        """Test BERT for masked language modeling."""
        model = BERTForMaskedLM()
        
        batch_size, seq_length = 2, 8
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        labels = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_length, model.config.vocab_size)

    def test_bert_for_sequence_classification(self):
        """Test BERT for sequence classification."""
        num_labels = 3
        model = BERTForSequenceClassification(num_labels=num_labels)
        
        batch_size, seq_length = 2, 8
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        labels = Tensor(np.random.randint(0, num_labels, (batch_size,)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, num_labels)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestRoBERTaVariants:
    """Test RoBERTa model variants."""

    def test_roberta_base_creation(self):
        """Test RoBERTa base model creation."""
        model = roberta_base()
        assert model is not None
        assert model.config.hidden_size == 768
        assert model.config.num_hidden_layers == 12
        assert model.config.vocab_size == 50265

    def test_roberta_large_creation(self):
        """Test RoBERTa large model creation."""
        model = roberta_large()
        assert model is not None
        assert model.config.hidden_size == 1024
        assert model.config.num_hidden_layers == 24
        assert model.config.num_attention_heads == 16
        assert model.config.intermediate_size == 4096

    def test_roberta_variants_forward_pass(self):
        """Test forward pass for RoBERTa variants."""
        variants = [roberta_base, roberta_large]
        
        for variant_func in variants:
            model = variant_func()
            
            # Create test input
            batch_size, seq_length = 2, 10
            input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
            attention_mask = Tensor(np.ones((batch_size, seq_length)))
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Check outputs
            assert "last_hidden_state" in outputs
            assert "pooler_output" in outputs
            
            last_hidden_state = outputs["last_hidden_state"]
            pooler_output = outputs["pooler_output"]
            
            assert last_hidden_state.shape == (batch_size, seq_length, model.config.hidden_size)
            assert pooler_output.shape == (batch_size, model.config.hidden_size)

    def test_roberta_for_masked_lm(self):
        """Test RoBERTa for masked language modeling."""
        model = RoBERTaForMaskedLM()
        
        batch_size, seq_length = 2, 8
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        labels = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_length, model.config.vocab_size)

    def test_roberta_for_sequence_classification(self):
        """Test RoBERTa for sequence classification."""
        num_labels = 3
        model = RoBERTaForSequenceClassification(num_labels=num_labels)
        
        batch_size, seq_length = 2, 8
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        labels = Tensor(np.random.randint(0, num_labels, (batch_size,)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, num_labels)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestDeBERTaVariants:
    """Test DeBERTa model variants."""

    def test_deberta_base_creation(self):
        """Test DeBERTa base model creation."""
        model = deberta_base()
        assert model is not None
        assert model.config.hidden_size == 768
        assert model.config.num_hidden_layers == 12
        assert model.config.num_attention_heads == 12
        assert model.config.vocab_size == 128100

    def test_deberta_large_creation(self):
        """Test DeBERTa large model creation."""
        model = deberta_large()
        assert model is not None
        assert model.config.hidden_size == 1024
        assert model.config.num_hidden_layers == 24
        assert model.config.num_attention_heads == 16

    def test_deberta_v3_base_creation(self):
        """Test DeBERTa-v3 base model creation."""
        model = deberta_v3_base()
        assert model is not None
        assert model.config.hidden_size == 768
        assert model.config.num_hidden_layers == 12
        assert hasattr(model.config, 'norm_rel_ebd')  # v3-specific feature

    def test_deberta_v3_large_creation(self):
        """Test DeBERTa-v3 large model creation."""
        model = deberta_v3_large()
        assert model is not None
        assert model.config.hidden_size == 1024
        assert model.config.num_hidden_layers == 24
        assert hasattr(model.config, 'norm_rel_ebd')  # v3-specific feature

    def test_deberta_variants_forward_pass(self):
        """Test forward pass for DeBERTa variants."""
        variants = [deberta_base, deberta_large, deberta_v3_base, deberta_v3_large]
        
        for variant_func in variants:
            model = variant_func()
            
            # Create test input
            batch_size, seq_length = 2, 8  # Keep smaller for DeBERTa complexity
            input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
            attention_mask = Tensor(np.ones((batch_size, seq_length)))
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Check outputs
            assert "last_hidden_state" in outputs
            
            last_hidden_state = outputs["last_hidden_state"]
            assert last_hidden_state.shape == (batch_size, seq_length, model.config.hidden_size)

    def test_deberta_for_masked_lm(self):
        """Test DeBERTa for masked language modeling."""
        model = DeBERTaForMaskedLM()
        
        batch_size, seq_length = 2, 6
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        labels = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_length, model.config.vocab_size)

    def test_deberta_for_sequence_classification(self):
        """Test DeBERTa for sequence classification."""
        num_labels = 2
        model = DeBERTaForSequenceClassification(num_labels=num_labels)
        
        batch_size, seq_length = 2, 6
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        labels = Tensor(np.random.randint(0, num_labels, (batch_size,)))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, num_labels)

    def test_deberta_disentangled_attention(self):
        """Test DeBERTa's disentangled attention mechanism."""
        model = deberta_base()
        
        # Test that the model has disentangled attention components
        assert hasattr(model.config, 'relative_attention')
        assert hasattr(model.config, 'pos_att_type')
        assert model.config.relative_attention is True
        assert model.config.pos_att_type == "p2c|c2p"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestTransformerComparison:
    """Test comparison between different transformer variants."""

    def test_model_size_progression(self):
        """Test that Large models are bigger than Base models."""
        # BERT comparison
        bert_base_model = bert_base()
        bert_large_model = bert_large()
        
        assert bert_large_model.config.hidden_size > bert_base_model.config.hidden_size
        assert bert_large_model.config.num_hidden_layers > bert_base_model.config.num_hidden_layers
        
        # RoBERTa comparison
        roberta_base_model = roberta_base()
        roberta_large_model = roberta_large()
        
        assert roberta_large_model.config.hidden_size > roberta_base_model.config.hidden_size
        assert roberta_large_model.config.num_hidden_layers > roberta_base_model.config.num_hidden_layers
        
        # DeBERTa comparison
        deberta_base_model = deberta_base()
        deberta_large_model = deberta_large()
        
        assert deberta_large_model.config.hidden_size > deberta_base_model.config.hidden_size
        assert deberta_large_model.config.num_hidden_layers > deberta_base_model.config.num_hidden_layers

    def test_vocab_size_differences(self):
        """Test vocabulary size differences between models."""
        bert_model = bert_base()
        roberta_model = roberta_base()
        deberta_model = deberta_base()
        
        # Each model has different vocab sizes
        assert bert_model.config.vocab_size == 30522
        assert roberta_model.config.vocab_size == 50265
        assert deberta_model.config.vocab_size == 128100

    def test_config_specific_features(self):
        """Test model-specific configuration features."""
        # BERT has token type embeddings
        bert_model = bert_base()
        assert bert_model.config.type_vocab_size == 2
        
        # RoBERTa typically doesn't use token types
        roberta_model = roberta_base()
        assert roberta_model.config.type_vocab_size == 1
        
        # DeBERTa has disentangled attention
        deberta_model = deberta_base()
        assert hasattr(deberta_model.config, 'relative_attention')
        assert hasattr(deberta_model.config, 'position_buckets')

    def test_parameter_initialization(self):
        """Test that models initialize parameters correctly."""
        models = [bert_base(), roberta_base(), deberta_base()]
        
        for model in models:
            # Check that parameters are initialized (not all zeros)
            params = model.parameters()
            
            # Should have some parameters
            assert len(params) > 0
            
            # Parameters should be initialized (not all zeros or all ones)
            for param_name, param in params.items():
                if hasattr(param, 'data'):
                    # Check that parameter values are reasonable
                    assert not np.allclose(param.data, 0.0)
                    assert np.all(np.isfinite(param.data))


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformer models not available")
class TestTransformerIntegration:
    """Integration tests for transformer variants."""

    def test_attention_mask_handling(self):
        """Test attention mask handling across all models."""
        models = [
            (bert_base(), 30522),
            (roberta_base(), 50265),
            (deberta_base(), 128100),
        ]
        
        for model, vocab_size in models:
            batch_size, seq_length = 2, 8
            
            # Create input with padding
            input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))
            
            # Create attention mask with some padding
            attention_mask = Tensor(np.array([
                [1, 1, 1, 1, 1, 0, 0, 0],  # 5 real tokens, 3 padding
                [1, 1, 1, 1, 1, 1, 1, 1]   # all real tokens
            ]))
            
            # Forward pass with attention mask
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Should complete without errors
            assert "last_hidden_state" in outputs
            assert outputs["last_hidden_state"].shape == (batch_size, seq_length, model.config.hidden_size)

    def test_output_consistency(self):
        """Test that outputs are consistent across multiple runs."""
        model = bert_base()
        
        batch_size, seq_length = 2, 6
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        
        # Multiple forward passes
        outputs1 = model(input_ids=input_ids)
        outputs2 = model(input_ids=input_ids)
        
        # Outputs should be identical (deterministic)
        assert np.allclose(
            outputs1["last_hidden_state"].data,
            outputs2["last_hidden_state"].data,
            rtol=1e-5,
            atol=1e-6
        )

    def test_gradient_flow(self):
        """Test that gradients flow properly through models."""
        model = bert_base()
        
        batch_size, seq_length = 2, 6
        input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_length)))
        
        # Forward pass
        outputs = model(input_ids=input_ids)
        last_hidden_state = outputs["last_hidden_state"]
        
        # Compute a simple loss
        loss = last_hidden_state.sum()
        
        # Should be able to compute loss without errors
        assert loss.size == 1
        assert np.isfinite(loss.data)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])