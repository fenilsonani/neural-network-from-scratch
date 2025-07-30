"""Comprehensive tests for config defaults to improve coverage from 61.11% to 95%+.

This file targets all preset configurations, preset access functions, and edge cases
in the config defaults module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.config.defaults import (
    DEFAULT_CONFIG,
    DEVELOPMENT_CONFIG,
    PRODUCTION_CONFIG,
    HIGH_PERFORMANCE_CONFIG,
    EDUCATIONAL_CONFIG,
    RESEARCH_CONFIG,
    TESTING_CONFIG,
    CONFIG_PRESETS,
    get_preset_config,
    list_preset_configs
)
from neural_arch.config.config import Config


class TestConfigDefaults:
    """Comprehensive tests for config defaults."""
    
    def test_default_config_properties(self):
        """Test all properties of DEFAULT_CONFIG."""
        assert DEFAULT_CONFIG.debug is False
        assert DEFAULT_CONFIG.log_level == "INFO"
        assert DEFAULT_CONFIG.device == "cpu"
        assert DEFAULT_CONFIG.dtype == "float32"
        assert DEFAULT_CONFIG.batch_size == 32
        assert DEFAULT_CONFIG.learning_rate == 0.001
        assert DEFAULT_CONFIG.max_epochs == 100
        assert DEFAULT_CONFIG.model_name == "neural_arch_model"
        assert DEFAULT_CONFIG.checkpoint_dir == "./checkpoints"
        assert DEFAULT_CONFIG.data_dir == "./data"
        assert DEFAULT_CONFIG.gradient_clipping is True
        assert DEFAULT_CONFIG.gradient_clip_value == 10.0
        assert DEFAULT_CONFIG.optimizer == "adam"
        
        # Test optimizer params
        assert DEFAULT_CONFIG.optimizer_params["beta1"] == 0.9
        assert DEFAULT_CONFIG.optimizer_params["beta2"] == 0.999
        assert DEFAULT_CONFIG.optimizer_params["eps"] == 1e-8
        
        # Test performance thresholds
        assert DEFAULT_CONFIG.performance_thresholds["tensor_creation_ms"] == 10.0
        assert DEFAULT_CONFIG.performance_thresholds["matmul_ms"] == 100.0
        assert DEFAULT_CONFIG.performance_thresholds["training_step_ms"] == 1000.0
        
        assert DEFAULT_CONFIG.memory_monitoring is True
        assert DEFAULT_CONFIG.enable_profiling is False
        assert DEFAULT_CONFIG.benchmark_mode is False
    
    def test_development_config_properties(self):
        """Test all properties of DEVELOPMENT_CONFIG."""
        assert DEVELOPMENT_CONFIG.debug is True
        assert DEVELOPMENT_CONFIG.log_level == "DEBUG"
        assert DEVELOPMENT_CONFIG.batch_size == 16
        assert DEVELOPMENT_CONFIG.max_epochs == 10
        assert DEVELOPMENT_CONFIG.save_frequency == 1
        assert DEVELOPMENT_CONFIG.enable_profiling is True
        assert DEVELOPMENT_CONFIG.memory_monitoring is True
        assert DEVELOPMENT_CONFIG.log_gradients is True
        assert DEVELOPMENT_CONFIG.log_weights is True
        assert DEVELOPMENT_CONFIG.metrics_frequency == 10
        assert DEVELOPMENT_CONFIG.tensorboard_dir == "./logs/dev"
        
        # Should inherit other values from DEFAULT_CONFIG
        assert DEVELOPMENT_CONFIG.device == "cpu"
        assert DEVELOPMENT_CONFIG.dtype == "float32"
        assert DEVELOPMENT_CONFIG.learning_rate == 0.001
    
    def test_production_config_properties(self):
        """Test all properties of PRODUCTION_CONFIG."""
        assert PRODUCTION_CONFIG.debug is False
        assert PRODUCTION_CONFIG.log_level == "WARNING"
        assert PRODUCTION_CONFIG.batch_size == 64
        assert PRODUCTION_CONFIG.max_epochs == 1000
        assert PRODUCTION_CONFIG.save_frequency == 50
        assert PRODUCTION_CONFIG.enable_profiling is False
        assert PRODUCTION_CONFIG.memory_monitoring is False
        assert PRODUCTION_CONFIG.log_gradients is False
        assert PRODUCTION_CONFIG.log_weights is False
        assert PRODUCTION_CONFIG.metrics_frequency == 1000
        assert PRODUCTION_CONFIG.benchmark_mode is True
        
        # Test stricter performance thresholds
        assert PRODUCTION_CONFIG.performance_thresholds["tensor_creation_ms"] == 5.0
        assert PRODUCTION_CONFIG.performance_thresholds["matmul_ms"] == 50.0
        assert PRODUCTION_CONFIG.performance_thresholds["training_step_ms"] == 500.0
    
    def test_high_performance_config_properties(self):
        """Test all properties of HIGH_PERFORMANCE_CONFIG."""
        assert HIGH_PERFORMANCE_CONFIG.batch_size == 128
        assert HIGH_PERFORMANCE_CONFIG.gradient_clipping is False
        assert HIGH_PERFORMANCE_CONFIG.memory_monitoring is False
        assert HIGH_PERFORMANCE_CONFIG.enable_profiling is False
        assert HIGH_PERFORMANCE_CONFIG.benchmark_mode is True
        
        # Test very strict performance thresholds
        assert HIGH_PERFORMANCE_CONFIG.performance_thresholds["tensor_creation_ms"] == 2.0
        assert HIGH_PERFORMANCE_CONFIG.performance_thresholds["matmul_ms"] == 25.0
        assert HIGH_PERFORMANCE_CONFIG.performance_thresholds["training_step_ms"] == 250.0
        
        assert HIGH_PERFORMANCE_CONFIG.num_threads is None
        
        # Should inherit production values
        assert HIGH_PERFORMANCE_CONFIG.debug is False
        assert HIGH_PERFORMANCE_CONFIG.log_level == "WARNING"
    
    def test_educational_config_properties(self):
        """Test all properties of EDUCATIONAL_CONFIG."""
        assert EDUCATIONAL_CONFIG.debug is True
        assert EDUCATIONAL_CONFIG.log_level == "DEBUG"
        assert EDUCATIONAL_CONFIG.batch_size == 8
        assert EDUCATIONAL_CONFIG.learning_rate == 0.01
        assert EDUCATIONAL_CONFIG.max_epochs == 50
        assert EDUCATIONAL_CONFIG.save_frequency == 5
        assert EDUCATIONAL_CONFIG.enable_profiling is True
        assert EDUCATIONAL_CONFIG.memory_monitoring is True
        assert EDUCATIONAL_CONFIG.log_gradients is True
        assert EDUCATIONAL_CONFIG.log_weights is True
        assert EDUCATIONAL_CONFIG.metrics_frequency == 1
        assert EDUCATIONAL_CONFIG.tensorboard_dir == "./logs/educational"
        assert EDUCATIONAL_CONFIG.experiment_name == "educational_experiment"
        assert EDUCATIONAL_CONFIG.tags == ["education", "learning", "tutorial"]
    
    def test_research_config_properties(self):
        """Test all properties of RESEARCH_CONFIG."""
        assert RESEARCH_CONFIG.debug is True
        assert RESEARCH_CONFIG.log_level == "INFO"
        assert RESEARCH_CONFIG.save_frequency == 10
        assert RESEARCH_CONFIG.enable_profiling is True
        assert RESEARCH_CONFIG.memory_monitoring is True
        assert RESEARCH_CONFIG.log_gradients is True
        assert RESEARCH_CONFIG.log_weights is False
        assert RESEARCH_CONFIG.metrics_frequency == 100
        assert RESEARCH_CONFIG.tensorboard_dir == "./logs/research"
        assert RESEARCH_CONFIG.tags == ["research", "experiment"]
        
        # Test metadata
        assert RESEARCH_CONFIG.metadata["research_purpose"] == "neural_architecture_experiment"
        assert RESEARCH_CONFIG.metadata["baseline_comparison"] is True
    
    def test_testing_config_properties(self):
        """Test all properties of TESTING_CONFIG."""
        assert TESTING_CONFIG.debug is True
        assert TESTING_CONFIG.log_level == "ERROR"
        assert TESTING_CONFIG.device == "cpu"
        assert TESTING_CONFIG.dtype == "float32"
        assert TESTING_CONFIG.batch_size == 4
        assert TESTING_CONFIG.learning_rate == 0.1
        assert TESTING_CONFIG.max_epochs == 2
        assert TESTING_CONFIG.patience == 1
        assert TESTING_CONFIG.model_name == "test_model"
        assert TESTING_CONFIG.checkpoint_dir == "./test_checkpoints"
        assert TESTING_CONFIG.data_dir == "./test_data"
        assert TESTING_CONFIG.gradient_clipping is True
        assert TESTING_CONFIG.gradient_clip_value == 1.0
        assert TESTING_CONFIG.random_seed == 42
        assert TESTING_CONFIG.enable_profiling is False
        assert TESTING_CONFIG.memory_monitoring is False
        assert TESTING_CONFIG.benchmark_mode is False
        assert TESTING_CONFIG.save_frequency == 1
        assert TESTING_CONFIG.metrics_frequency == 1
        
        # Test relaxed performance thresholds
        assert TESTING_CONFIG.performance_thresholds["tensor_creation_ms"] == 100.0
        assert TESTING_CONFIG.performance_thresholds["matmul_ms"] == 1000.0
        assert TESTING_CONFIG.performance_thresholds["training_step_ms"] == 10000.0
    
    def test_config_presets_dictionary(self):
        """Test CONFIG_PRESETS dictionary contains all expected presets."""
        expected_presets = {
            "default": DEFAULT_CONFIG,
            "development": DEVELOPMENT_CONFIG,
            "dev": DEVELOPMENT_CONFIG,
            "production": PRODUCTION_CONFIG,
            "prod": PRODUCTION_CONFIG,
            "high_performance": HIGH_PERFORMANCE_CONFIG,
            "performance": HIGH_PERFORMANCE_CONFIG,
            "educational": EDUCATIONAL_CONFIG,
            "education": EDUCATIONAL_CONFIG,
            "research": RESEARCH_CONFIG,
            "testing": TESTING_CONFIG,
            "test": TESTING_CONFIG,
        }
        
        assert len(CONFIG_PRESETS) == len(expected_presets)
        
        for name, config in expected_presets.items():
            assert name in CONFIG_PRESETS
            assert CONFIG_PRESETS[name] is config
    
    def test_get_preset_config_valid_names(self):
        """Test get_preset_config with all valid preset names."""
        # Test all primary names
        assert get_preset_config("default") is DEFAULT_CONFIG
        assert get_preset_config("development") is DEVELOPMENT_CONFIG
        assert get_preset_config("production") is PRODUCTION_CONFIG
        assert get_preset_config("high_performance") is HIGH_PERFORMANCE_CONFIG
        assert get_preset_config("educational") is EDUCATIONAL_CONFIG
        assert get_preset_config("research") is RESEARCH_CONFIG
        assert get_preset_config("testing") is TESTING_CONFIG
        
        # Test aliases
        assert get_preset_config("dev") is DEVELOPMENT_CONFIG
        assert get_preset_config("prod") is PRODUCTION_CONFIG
        assert get_preset_config("performance") is HIGH_PERFORMANCE_CONFIG
        assert get_preset_config("education") is EDUCATIONAL_CONFIG
        assert get_preset_config("test") is TESTING_CONFIG
    
    def test_get_preset_config_invalid_names(self):
        """Test get_preset_config with invalid preset names."""
        invalid_names = [
            "invalid",
            "unknown_preset",
            "not_a_preset",
            "",
            "DEFAULT",  # Case sensitive
            "PRODUCTION",
            "dev_production",
            "custom_preset"
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises(ValueError) as exc_info:
                get_preset_config(invalid_name)
            
            assert f"Unknown preset '{invalid_name}'" in str(exc_info.value)
            assert "Available presets:" in str(exc_info.value)
            
            # Check that all available presets are listed
            error_message = str(exc_info.value)
            for preset_name in CONFIG_PRESETS.keys():
                assert preset_name in error_message
    
    def test_list_preset_configs(self):
        """Test list_preset_configs function."""
        preset_list = list_preset_configs()
        
        # Should return a list
        assert isinstance(preset_list, list)
        
        # Should contain all preset names
        expected_names = set(CONFIG_PRESETS.keys())
        actual_names = set(preset_list)
        assert actual_names == expected_names
        
        # Should be the same length
        assert len(preset_list) == len(CONFIG_PRESETS)
        
        # Should not contain duplicates
        assert len(preset_list) == len(set(preset_list))
    
    def test_config_inheritance_consistency(self):
        """Test that derived configs properly inherit from base configs."""
        # DEVELOPMENT_CONFIG should inherit from DEFAULT_CONFIG
        # Check some inherited properties
        assert DEVELOPMENT_CONFIG.device == DEFAULT_CONFIG.device
        assert DEVELOPMENT_CONFIG.dtype == DEFAULT_CONFIG.dtype
        assert DEVELOPMENT_CONFIG.model_name == DEFAULT_CONFIG.model_name
        assert DEVELOPMENT_CONFIG.checkpoint_dir == DEFAULT_CONFIG.checkpoint_dir
        assert DEVELOPMENT_CONFIG.data_dir == DEFAULT_CONFIG.data_dir
        assert DEVELOPMENT_CONFIG.gradient_clipping == DEFAULT_CONFIG.gradient_clipping
        assert DEVELOPMENT_CONFIG.gradient_clip_value == DEFAULT_CONFIG.gradient_clip_value
        assert DEVELOPMENT_CONFIG.optimizer == DEFAULT_CONFIG.optimizer
        assert DEVELOPMENT_CONFIG.optimizer_params == DEFAULT_CONFIG.optimizer_params
        
        # HIGH_PERFORMANCE_CONFIG should inherit from PRODUCTION_CONFIG
        assert HIGH_PERFORMANCE_CONFIG.debug == PRODUCTION_CONFIG.debug
        assert HIGH_PERFORMANCE_CONFIG.log_level == PRODUCTION_CONFIG.log_level
        assert HIGH_PERFORMANCE_CONFIG.max_epochs == PRODUCTION_CONFIG.max_epochs
        assert HIGH_PERFORMANCE_CONFIG.save_frequency == PRODUCTION_CONFIG.save_frequency
        assert HIGH_PERFORMANCE_CONFIG.log_gradients == PRODUCTION_CONFIG.log_gradients
        assert HIGH_PERFORMANCE_CONFIG.log_weights == PRODUCTION_CONFIG.log_weights
        assert HIGH_PERFORMANCE_CONFIG.metrics_frequency == PRODUCTION_CONFIG.metrics_frequency
    
    def test_config_instances_are_config_type(self):
        """Test that all config instances are proper Config objects."""
        configs_to_test = [
            DEFAULT_CONFIG,
            DEVELOPMENT_CONFIG,
            PRODUCTION_CONFIG,
            HIGH_PERFORMANCE_CONFIG,
            EDUCATIONAL_CONFIG,
            RESEARCH_CONFIG,
            TESTING_CONFIG
        ]
        
        for config in configs_to_test:
            assert isinstance(config, Config)
            
            # Should have essential Config methods/properties
            assert hasattr(config, 'update')
            assert hasattr(config, 'to_dict')
            
            # Should be callable methods
            assert callable(config.update)
            assert callable(config.to_dict)
            
            # Note: save method might not be available in all Config implementations
            # This is testing the actual behavior
    
    def test_config_preset_immutability(self):
        """Test that preset configs maintain their identity after access."""
        # Get configs multiple times
        config1 = get_preset_config("default")
        config2 = get_preset_config("default")
        
        # Should be the same object
        assert config1 is config2
        assert config1 is DEFAULT_CONFIG
        
        # Test with different preset
        dev_config1 = get_preset_config("development")
        dev_config2 = get_preset_config("dev")  # Alias
        
        assert dev_config1 is dev_config2
        assert dev_config1 is DEVELOPMENT_CONFIG
    
    def test_all_configs_have_required_properties(self):
        """Test that all preset configs have essential properties."""
        required_properties = [
            'debug', 'log_level', 'device', 'dtype', 'batch_size',
            'learning_rate', 'gradient_clipping'
        ]
        
        configs_to_test = [
            DEFAULT_CONFIG,
            DEVELOPMENT_CONFIG,
            PRODUCTION_CONFIG,
            HIGH_PERFORMANCE_CONFIG,
            EDUCATIONAL_CONFIG,
            RESEARCH_CONFIG,
            TESTING_CONFIG
        ]
        
        for config in configs_to_test:
            for prop in required_properties:
                assert hasattr(config, prop), f"Config missing required property: {prop}"
                # Property should not be None (can be False/0 but not None)
                assert getattr(config, prop) is not None, f"Property {prop} is None"
    
    def test_performance_configs_hierarchy(self):
        """Test performance-related configs have appropriate values."""
        # Performance should be stricter as we go up the hierarchy
        
        # Batch sizes should generally increase for performance
        assert DEFAULT_CONFIG.batch_size <= PRODUCTION_CONFIG.batch_size
        assert PRODUCTION_CONFIG.batch_size <= HIGH_PERFORMANCE_CONFIG.batch_size
        
        # Educational should have smallest batch size for learning
        assert EDUCATIONAL_CONFIG.batch_size <= DEFAULT_CONFIG.batch_size
        assert TESTING_CONFIG.batch_size <= EDUCATIONAL_CONFIG.batch_size
        
        # Performance thresholds should be stricter (lower) for high-performance configs
        default_tensor_ms = DEFAULT_CONFIG.performance_thresholds["tensor_creation_ms"]
        prod_tensor_ms = PRODUCTION_CONFIG.performance_thresholds["tensor_creation_ms"]
        hp_tensor_ms = HIGH_PERFORMANCE_CONFIG.performance_thresholds["tensor_creation_ms"]
        
        assert hp_tensor_ms <= prod_tensor_ms <= default_tensor_ms
        
        # Testing should have most relaxed thresholds
        test_tensor_ms = TESTING_CONFIG.performance_thresholds["tensor_creation_ms"]
        assert test_tensor_ms >= default_tensor_ms
    
    def test_debug_and_logging_hierarchy(self):
        """Test debug and logging configurations are consistent."""
        # Production configs should have less verbose logging
        debug_configs = [DEVELOPMENT_CONFIG, EDUCATIONAL_CONFIG, RESEARCH_CONFIG, TESTING_CONFIG]
        production_configs = [PRODUCTION_CONFIG, HIGH_PERFORMANCE_CONFIG]
        
        for config in debug_configs:
            assert config.debug is True, f"Debug config should have debug=True"
        
        for config in production_configs:
            assert config.debug is False, f"Production config should have debug=False"
        
        # Log levels should be appropriate
        assert TESTING_CONFIG.log_level == "ERROR"  # Suppress logs during testing
        assert DEVELOPMENT_CONFIG.log_level == "DEBUG"  # Verbose for development
        assert EDUCATIONAL_CONFIG.log_level == "DEBUG"  # Verbose for learning
        assert RESEARCH_CONFIG.log_level == "INFO"  # Moderate for research
        assert PRODUCTION_CONFIG.log_level == "WARNING"  # Minimal for production
        assert DEFAULT_CONFIG.log_level == "INFO"  # Balanced default
    
    def test_special_config_features(self):
        """Test special features in specific configs."""
        # Educational config should have very frequent metrics
        assert EDUCATIONAL_CONFIG.metrics_frequency == 1
        
        # Testing config should have fixed random seed for reproducibility
        assert TESTING_CONFIG.random_seed == 42
        
        # High performance config should disable gradient clipping for speed
        assert HIGH_PERFORMANCE_CONFIG.gradient_clipping is False
        
        # Research config should have metadata
        assert hasattr(RESEARCH_CONFIG, 'metadata')
        assert isinstance(RESEARCH_CONFIG.metadata, dict)
        assert len(RESEARCH_CONFIG.metadata) > 0
        
        # Educational config should have tags
        assert hasattr(EDUCATIONAL_CONFIG, 'tags')
        assert isinstance(EDUCATIONAL_CONFIG.tags, list)
        assert len(EDUCATIONAL_CONFIG.tags) > 0
        assert "education" in EDUCATIONAL_CONFIG.tags
    
    def test_directory_configurations(self):
        """Test directory configurations in different presets."""
        # Default directories
        assert DEFAULT_CONFIG.checkpoint_dir == "./checkpoints"
        assert DEFAULT_CONFIG.data_dir == "./data"
        
        # Testing should have separate directories
        assert TESTING_CONFIG.checkpoint_dir == "./test_checkpoints"
        assert TESTING_CONFIG.data_dir == "./test_data"
        
        # Development and educational should have tensorboard directories
        assert DEVELOPMENT_CONFIG.tensorboard_dir == "./logs/dev"
        assert EDUCATIONAL_CONFIG.tensorboard_dir == "./logs/educational"
        assert RESEARCH_CONFIG.tensorboard_dir == "./logs/research"
    
    def test_error_handling_edge_cases(self):
        """Test edge cases in error handling."""
        # Test with None - the function converts None to string "None"
        with pytest.raises(ValueError) as exc_info:
            get_preset_config(None)
        assert "Unknown preset 'None'" in str(exc_info.value)
        
        # Test with numeric types - gets converted to string
        with pytest.raises(ValueError) as exc_info:
            get_preset_config(123)
        assert "Unknown preset '123'" in str(exc_info.value)
        
        # Test with boolean - gets converted to string
        with pytest.raises(ValueError) as exc_info:
            get_preset_config(True)
        assert "Unknown preset 'True'" in str(exc_info.value)
        
        # Test case sensitivity
        with pytest.raises(ValueError):
            get_preset_config("Default")  # Capital D
        
        with pytest.raises(ValueError):
            get_preset_config("DEVELOPMENT")  # All caps
    
    def test_preset_list_ordering(self):
        """Test that preset list is consistent and ordered."""
        list1 = list_preset_configs()
        list2 = list_preset_configs()
        
        # Should return the same list each time
        assert list1 == list2
        
        # Should contain no None values
        assert None not in list1
        
        # Should contain only strings
        for preset_name in list1:
            assert isinstance(preset_name, str)
            assert len(preset_name) > 0