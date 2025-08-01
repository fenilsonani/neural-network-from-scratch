"""Configuration validation utilities (placeholder)."""

from .config import Config
from typing import Union, Dict, Any


class ConfigValidator:
    """Configuration validator (placeholder)."""
    
    @staticmethod
    def validate(config: Union[Config, Dict[str, Any]]) -> bool:
        """Validate configuration."""
        if isinstance(config, dict):
            # Convert dict to Config object for validation
            config_obj = Config(**config)
            config_obj._validate()
        else:
            config._validate()
        return True


def validate_config(config: Union[Config, Dict[str, Any]]) -> bool:
    """Validate configuration function."""
    return ConfigValidator.validate(config)