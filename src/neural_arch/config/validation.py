"""Configuration validation utilities (placeholder)."""

from .config import Config


class ConfigValidator:
    """Configuration validator (placeholder)."""
    
    @staticmethod
    def validate(config: Config) -> bool:
        """Validate configuration."""
        config._validate()
        return True


def validate_config(config: Config) -> bool:
    """Validate configuration function."""
    return ConfigValidator.validate(config)