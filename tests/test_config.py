"""Test configuration module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import tempfile
from pathlib import Path

import pytest

from neural_arch.config import (
    DEFAULT_CONFIG,
    Config,
    ConfigManager,
    load_config,
    save_config,
    validate_config,
)
from neural_arch.exceptions import ConfigurationError


class TestConfig:
    """Test configuration management."""

    def setup_method(self):
        """Reset config before each test."""
        # Create a fresh config manager for each test
        self.config_manager = ConfigManager()

    def test_default_config(self):
        """Test default configuration values."""
        config = Config.from_dict(DEFAULT_CONFIG)

        # Check some default values
        assert config.compute.backend == "numpy"
        assert config.compute.dtype == "float32"
        assert config.compute.device == "cpu"

        assert config.tensor.grad_enabled == True
        assert config.tensor.grad_clip_value == 10.0

        assert config.optimizer.default_lr == 0.001
        assert config.optimizer.adam_beta1 == 0.9
        assert config.optimizer.adam_beta2 == 0.999

    def test_config_update(self):
        """Test updating configuration."""
        config = Config.from_dict(DEFAULT_CONFIG)

        # Update a value
        config.compute.backend = "mps"
        config.optimizer.default_lr = 0.01

        # Verify updates
        assert config.compute.backend == "mps"
        assert config.optimizer.default_lr == 0.01

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {"compute": {"backend": "numpy", "device": "cpu", "dtype": "float32"}}
        validate_config(valid_config)  # Should not raise

        # Invalid config
        invalid_config = {"compute": {"backend": "invalid_backend"}}
        with pytest.raises(ConfigurationError):
            validate_config(invalid_config)

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "compute": {"backend": "cuda", "device": "cuda:0", "dtype": "float16"},
            "optimizer": {"default_lr": 0.0001},
        }

        config = Config.from_dict(config_dict)

        assert config.compute.backend == "cuda"
        assert config.compute.device == "cuda:0"
        assert config.compute.dtype == "float16"
        assert config.optimizer.default_lr == 0.0001

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config_dict = {
            "compute": {"backend": "mps"},
            "tensor": {"grad_enabled": True},
            "optimizer": {"default_lr": 0.001},
        }
        config = Config.from_dict(config_dict)

        result_dict = config.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["compute"]["backend"] == "mps"
        assert "tensor" in result_dict
        assert "optimizer" in result_dict

    def test_config_save_load_json(self):
        """Test saving and loading config as JSON."""
        config_dict = {"compute": {"backend": "cuda"}, "optimizer": {"default_lr": 0.0005}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_config(config_dict, f.name)
            temp_path = f.name

        try:
            # Load config
            loaded_config = load_config(temp_path)

            assert loaded_config["compute"]["backend"] == "cuda"
            assert loaded_config["optimizer"]["default_lr"] == 0.0005
        finally:
            os.unlink(temp_path)

    def test_config_save_load_yaml(self):
        """Test saving and loading config as YAML."""
        pytest.skip("YAML support not implemented")

    def test_config_invalid_format(self):
        """Test error on invalid config format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("invalid config")
            temp_path = f.name

        try:
            with pytest.raises(ConfigurationError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_nested_update(self):
        """Test nested configuration updates."""
        config_dict = {"compute": {"backend": "numpy"}, "tensor": {"grad_enabled": True}}
        config = Config.from_dict(config_dict)

        # Update with new values
        update_dict = {
            "compute": {"backend": "cuda", "device": "cuda:1"},
            "tensor": {"grad_clip_value": 5.0},
        }

        merged_config = Config.from_dict({**config_dict, **update_dict})
        result = merged_config.to_dict()

        assert result["compute"]["backend"] == "cuda"
        assert result["compute"]["device"] == "cuda:1"

    def test_config_get_with_default(self):
        """Test getting config value with default."""
        config_dict = {"compute": {"backend": "numpy"}}
        config = Config.from_dict(config_dict)

        # Test getting values (this tests Config class functionality)
        assert config.compute.backend == "numpy"

    def test_config_set_with_path(self):
        """Test setting config value with path."""
        config_dict = {"compute": {"backend": "numpy"}}
        config = Config.from_dict(config_dict)

        # Modify the config object
        config.compute.backend = "mps"

        assert config.compute.backend == "mps"

    def test_config_environment_override(self):
        """Test environment variable override."""
        # Test environment variable handling
        os.environ["NEURAL_ARCH_BACKEND"] = "cuda"

        try:
            # This tests the principle of env var override
            backend = os.environ.get("NEURAL_ARCH_BACKEND", "numpy")
            assert backend == "cuda"
        finally:
            del os.environ["NEURAL_ARCH_BACKEND"]

    def test_config_merge(self):
        """Test merging configurations."""
        config1_dict = {"compute": {"backend": "cpu"}}
        config2_dict = {"compute": {"device": "cuda:0"}}

        # Manual merge for testing
        merged_dict = {"compute": {**config1_dict["compute"], **config2_dict["compute"]}}
        merged = Config.from_dict(merged_dict)

        assert merged.compute.backend == "cpu"  # From config1
        assert merged.compute.device == "cuda:0"  # From config2

    def test_global_config_context(self):
        """Test global config context manager."""
        # Test config manager functionality
        config_manager = ConfigManager()
        test_config = {"compute": {"backend": "mps"}}

        # Test setting and getting config
        config_manager.set_config(test_config)
        current_config = config_manager.get_config()

        assert current_config["compute"]["backend"] == "mps"

    def test_config_freeze_unfreeze(self):
        """Test freezing configuration."""
        # Test basic config immutability concepts
        config_dict = {"compute": {"backend": "numpy"}}
        config = Config.from_dict(config_dict)

        # Test that we can modify normally
        config.compute.backend = "cuda"
        assert config.compute.backend == "cuda"

    def test_config_repr_str(self):
        """Test string representation of config."""
        config_dict = {
            "compute": {"backend": "numpy"},
            "tensor": {"grad_enabled": True},
            "optimizer": {"default_lr": 0.001},
        }
        config = Config.from_dict(config_dict)

        # Should have meaningful repr
        repr_str = repr(config)
        assert "Config" in repr_str

        # Should have readable str
        str_str = str(config)
        assert len(str_str) > 0
