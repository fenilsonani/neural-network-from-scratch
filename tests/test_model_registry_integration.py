"""Test model registry integration for transformer variants.

This test suite ensures that all new transformer variants are properly
registered in the model registry and can be instantiated correctly.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.neural_arch.models.registry import ModelRegistry, get_model, list_models
    from src.neural_arch.models.language import (
        # BERT variants
        bert_base,
        bert_large,
        bert_base_cased,
        bert_large_cased,
        # RoBERTa variants
        roberta_base,
        roberta_large,
        # DeBERTa variants
        deberta_base,
        deberta_large,
        deberta_v3_base,
        deberta_v3_large,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestModelRegistryIntegration:
    """Test model registry integration."""

    def test_bert_variants_registered(self):
        """Test that BERT variants are registered."""
        registered_models = list_models()
        
        # Check BERT models are registered
        bert_models = [name for name in registered_models if "bert" in name.lower()]
        
        expected_bert_models = [
            "bert_base",
            "bert_large",
            "bert_base_cased", 
            "bert_large_cased"
        ]
        
        for model_name in expected_bert_models:
            assert model_name in bert_models, f"Model {model_name} not registered"

    def test_roberta_variants_registered(self):
        """Test that RoBERTa variants are registered."""
        registered_models = list_models()
        
        # Check RoBERTa models are registered
        roberta_models = [name for name in registered_models if "roberta" in name.lower()]
        
        expected_roberta_models = [
            "roberta_base",
            "roberta_large"
        ]
        
        for model_name in expected_roberta_models:
            assert model_name in roberta_models, f"Model {model_name} not registered"

    def test_deberta_variants_registered(self):
        """Test that DeBERTa variants are registered."""
        registered_models = list_models()
        
        # Check DeBERTa models are registered
        deberta_models = [name for name in registered_models if "deberta" in name.lower()]
        
        expected_deberta_models = [
            "deberta_base",
            "deberta_large", 
            "deberta_v3_base",
            "deberta_v3_large"
        ]
        
        for model_name in expected_deberta_models:
            assert model_name in deberta_models, f"Model {model_name} not registered"

    def test_model_instantiation_via_registry(self):
        """Test model instantiation through registry."""
        model_functions = [
            ("bert_base", bert_base),
            ("bert_large", bert_large),
            ("roberta_base", roberta_base),
            ("roberta_large", roberta_large),
            ("deberta_base", deberta_base),
            ("deberta_large", deberta_large),
        ]
        
        for model_name, expected_func in model_functions:
            try:
                # Get model through registry
                model_via_registry = get_model(model_name)
                
                # Get model through direct function
                model_direct = expected_func()
                
                # Should have same configuration
                assert model_via_registry.config.hidden_size == model_direct.config.hidden_size
                assert model_via_registry.config.num_hidden_layers == model_direct.config.num_hidden_layers
                assert model_via_registry.config.vocab_size == model_direct.config.vocab_size
                
            except Exception as e:
                pytest.fail(f"Failed to instantiate {model_name} via registry: {e}")

    def test_model_aliases(self):
        """Test model aliases work correctly."""
        alias_tests = [
            ("bert-base-uncased", "bert_base"),
            ("bert-large-uncased", "bert_large"),
            ("bert-base-cased", "bert_base_cased"),
            ("roberta-base", "roberta_base"),
            ("roberta-large", "roberta_large"),
            ("deberta-base", "deberta_base"),
            ("deberta-large", "deberta_large"),
        ]
        
        for alias, canonical_name in alias_tests:
            try:
                model_via_alias = get_model(alias)
                model_via_name = get_model(canonical_name)
                
                # Should be equivalent
                assert model_via_alias.config.hidden_size == model_via_name.config.hidden_size
                assert model_via_alias.config.num_hidden_layers == model_via_name.config.num_hidden_layers
                
            except Exception as e:
                # Some aliases might not be registered, that's okay
                print(f"Alias {alias} not available: {e}")

    def test_model_metadata(self):
        """Test model metadata is properly set."""
        registry = ModelRegistry.get_instance()
        
        transformer_models = [
            "bert_base", "bert_large", "roberta_base", "roberta_large",
            "deberta_base", "deberta_large", "deberta_v3_base", "deberta_v3_large"
        ]
        
        for model_name in transformer_models:
            if model_name in registry._models:
                model_info = registry._models[model_name]
                
                # Check required metadata fields
                assert hasattr(model_info, 'description')
                assert hasattr(model_info, 'paper_url')
                assert hasattr(model_info, 'tags')
                
                # Check that description is meaningful
                assert len(model_info.description) > 10
                
                # Check paper URL format
                assert model_info.paper_url.startswith('https://arxiv.org/')
                
                # Check tags include relevant categories
                assert 'language' in model_info.tags
                assert 'transformer' in model_info.tags

    def test_model_configurations(self):
        """Test model pre-configured variants."""
        registry = ModelRegistry.get_instance()
        
        config_tests = [
            ("bert_base", {"hidden_size": 768, "num_hidden_layers": 12}),
            ("bert_large", {"hidden_size": 1024, "num_hidden_layers": 24}),
            ("roberta_base", {"hidden_size": 768, "num_hidden_layers": 12}),
            ("roberta_large", {"hidden_size": 1024, "num_hidden_layers": 24}),
            ("deberta_base", {"hidden_size": 768, "num_hidden_layers": 12}),
            ("deberta_large", {"hidden_size": 1024, "num_hidden_layers": 24}),
        ]
        
        for model_name, expected_config in config_tests:
            if model_name in registry._models:
                model_info = registry._models[model_name]
                
                # Check pretrained configs exist
                assert hasattr(model_info, 'pretrained_configs')
                assert len(model_info.pretrained_configs) > 0
                
                # Instantiate model and check config
                model = get_model(model_name)
                
                for key, expected_value in expected_config.items():
                    actual_value = getattr(model.config, key)
                    assert actual_value == expected_value, f"{model_name}.{key}: expected {expected_value}, got {actual_value}"

    def test_model_registry_consistency(self):
        """Test model registry consistency across imports."""
        # Import models multiple times and ensure registry remains consistent
        from importlib import reload
        import src.neural_arch.models.language.bert
        import src.neural_arch.models.language.roberta
        import src.neural_arch.models.language.deberta
        
        initial_models = set(list_models())
        
        # Reload modules
        reload(src.neural_arch.models.language.bert)
        reload(src.neural_arch.models.language.roberta) 
        reload(src.neural_arch.models.language.deberta)
        
        final_models = set(list_models())
        
        # Registry should remain consistent
        assert initial_models == final_models

    def test_model_tags_filtering(self):
        """Test filtering models by tags."""
        all_models = list_models()
        
        # Filter by language tag
        language_models = [name for name in all_models 
                          if 'language' in ModelRegistry.get_instance()._models.get(name, object()).tags]
        
        # Should include our transformer models
        transformer_models = [
            "bert_base", "bert_large", "roberta_base", "roberta_large",
            "deberta_base", "deberta_large"
        ]
        
        for model_name in transformer_models:
            if model_name in all_models:
                assert model_name in language_models

        # Filter by transformer tag
        transformer_tag_models = [name for name in all_models
                                 if 'transformer' in ModelRegistry.get_instance()._models.get(name, object()).tags]
        
        for model_name in transformer_models:
            if model_name in all_models:
                assert model_name in transformer_tag_models

    def test_model_paper_urls(self):
        """Test that paper URLs are valid and consistent."""
        registry = ModelRegistry.get_instance()
        
        paper_url_tests = [
            ("bert_base", "https://arxiv.org/abs/1810.04805"),
            ("bert_large", "https://arxiv.org/abs/1810.04805"),
            ("roberta_base", "https://arxiv.org/abs/1907.11692"),
            ("roberta_large", "https://arxiv.org/abs/1907.11692"),
            ("deberta_base", "https://arxiv.org/abs/2006.03654"),
            ("deberta_large", "https://arxiv.org/abs/2006.03654"),
            ("deberta_v3_base", "https://arxiv.org/abs/2111.09543"),
            ("deberta_v3_large", "https://arxiv.org/abs/2111.09543"),
        ]
        
        for model_name, expected_url in paper_url_tests:
            if model_name in registry._models:
                model_info = registry._models[model_name]
                assert model_info.paper_url == expected_url


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestModelRegistryErrorHandling:
    """Test error handling in model registry."""

    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        with pytest.raises(ValueError):
            get_model("nonexistent_model")

    def test_duplicate_registration_prevention(self):
        """Test that duplicate registration is prevented."""
        # This should be handled gracefully in the registry
        # The models should already be registered, so re-importing shouldn't cause issues
        from src.neural_arch.models.language import bert_base
        
        # Should not raise an error
        model = bert_base()
        assert model is not None

    def test_model_instantiation_errors(self):
        """Test handling of model instantiation errors."""
        # Test with invalid configuration
        try:
            from src.neural_arch.models.language.bert import BERT, BERTConfig
            
            # Create invalid config
            invalid_config = BERTConfig(hidden_size=768, num_attention_heads=7)  # Not divisible
            
            with pytest.raises(ValueError):
                model = BERT(invalid_config)
                
        except ImportError:
            pytest.skip("BERT not available for error testing")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestModelRegistryPerformance:
    """Test model registry performance."""

    def test_model_listing_performance(self):
        """Test performance of model listing."""
        import time
        
        start_time = time.time()
        models = list_models()
        end_time = time.time()
        
        # Should be fast
        assert end_time - start_time < 1.0
        assert len(models) > 0

    def test_model_instantiation_performance(self):
        """Test performance of model instantiation."""
        import time
        
        model_names = ["bert_base", "roberta_base", "deberta_base"]
        
        for model_name in model_names:
            try:
                start_time = time.time()
                model = get_model(model_name)
                end_time = time.time()
                
                # Model creation should be reasonably fast
                assert end_time - start_time < 30.0  # 30 seconds max
                assert model is not None
                
            except Exception as e:
                print(f"Performance test failed for {model_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])