"""Enterprise-grade configuration management system."""

import json
import os
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Type-safe configuration container for neural architecture.

    This provides enterprise-grade configuration management with:
    - Type safety and validation
    - Environment variable override support
    - Hierarchical configuration merging
    - Schema validation
    - Default value management
    """

    # Core system settings
    debug: bool = False
    log_level: str = "INFO"
    random_seed: Optional[int] = None

    # Device and compute settings
    device: str = "cpu"
    dtype: str = "float32"
    num_threads: Optional[int] = None
    memory_limit_gb: Optional[float] = None

    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    patience: int = 10
    gradient_clipping: bool = True
    gradient_clip_value: float = 10.0

    # Model settings
    model_name: str = "neural_arch_model"
    model_version: str = "1.0.0"
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 10

    # Data settings
    data_dir: str = "./data"
    vocab_size: int = 10000
    sequence_length: int = 128
    preprocessing: Dict[str, Any] = field(default_factory=dict)

    # Optimization settings
    optimizer: str = "adam"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)

    # Performance settings
    enable_profiling: bool = False
    benchmark_mode: bool = False
    memory_monitoring: bool = True
    performance_thresholds: Dict[str, float] = field(default_factory=dict)

    # Monitoring and logging
    tensorboard_dir: Optional[str] = None
    metrics_frequency: int = 100
    log_gradients: bool = False
    log_weights: bool = False

    # Enterprise features
    experiment_name: Optional[str] = None
    experiment_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive", config_key="batch_size")

        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive", config_key="learning_rate")

        if self.max_epochs <= 0:
            raise ConfigurationError("max_epochs must be positive", config_key="max_epochs")

        if self.gradient_clip_value <= 0:
            raise ConfigurationError(
                "gradient_clip_value must be positive", config_key="gradient_clip_value")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(
                f"Invalid log_level: {self.log_level}",
                config_key="log_level",
                config_value=self.log_level
            )

        if self.device not in ["cpu", "cuda"]:
            logger.warning(f"Unusual device specified: {self.device}")

        if self.dtype not in ["float32", "float64", "int32", "int64"]:
            raise ConfigurationError(
                f"Unsupported dtype: {self.dtype}",
                config_key="dtype",
                config_value=self.dtype
            )

    def update(self, **kwargs) -> 'Config':
        """Update configuration with new values.

        Args:
            **kwargs: Configuration values to update

        Returns:
            New Config instance with updated values
        """
        config_dict = asdict(self)
        config_dict.update(kwargs)
        return Config(**config_dict)

    def merge(self, other: 'Config') -> 'Config':
        """Merge with another configuration.

        Args:
            other: Configuration to merge

        Returns:
            New Config instance with merged values
        """
        self_dict = asdict(self)
        other_dict = asdict(other)

        # Deep merge dictionaries
        merged = self._deep_merge(self_dict, other_dict)
        return Config(**merged)

    def _deep_merge(self, dict1: dict, dict2: dict) -> dict:
        """Deep merge two dictionaries."""
        result = deepcopy(dict1)

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'Config':
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Config':
        """Create configuration from YAML string."""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)


class ConfigManager:
    """Enterprise configuration manager with environment variable support.

    Features:
    - Environment variable override
    - Configuration file loading (JSON/YAML)
    - Hierarchical configuration merging
    - Configuration validation
    - Hot-reload support
    """

    def __init__(self, base_config: Optional[Config] = None):
        """Initialize configuration manager.

        Args:
            base_config: Base configuration to start with
        """
        self.base_config = base_config or Config()
        self._watchers: List[callable] = []

    def load_from_file(self, file_path: Union[str, Path]) -> Config:
        """Load configuration from file.

        Args:
            file_path: Path to configuration file (.json or .yaml)

        Returns:
            Loaded configuration

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                config_key="file_path",
                config_value=str(file_path)
            )

        try:
            content = file_path.read_text()

            if file_path.suffix.lower() in ['.json']:
                config = Config.from_json(content)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ConfigurationError("PyYAML not installed for YAML config files")
                config = Config.from_yaml(content)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {file_path.suffix}",
                    config_key="file_format",
                    config_value=file_path.suffix
                )

            logger.info(f"Loaded configuration from {file_path}")
            return config

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {file_path}: {e}",
                original_exception=e
            ) from e

    def save_to_file(self, config: Config, file_path: Union[str, Path]) -> None:
        """Save configuration to file.

        Args:
            config: Configuration to save
            file_path: Path to save configuration file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if file_path.suffix.lower() in ['.json']:
                content = config.to_json()
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                content = config.to_yaml()
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {file_path.suffix}",
                    config_key="file_format",
                    config_value=file_path.suffix
                )

            file_path.write_text(content)
            logger.info(f"Saved configuration to {file_path}")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration to {file_path}: {e}",
                original_exception=e
            ) from e

    def load_from_env(self, prefix: str = "NEURAL_ARCH_") -> Config:
        """Load configuration from environment variables.

        Args:
            prefix: Prefix for environment variables

        Returns:
            Configuration with environment variable overrides
        """
        env_config = {}

        # Map environment variable names to config keys
        env_mapping = {
            f"{prefix}DEBUG": "debug",
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}DEVICE": "device",
            f"{prefix}BATCH_SIZE": "batch_size",
            f"{prefix}LEARNING_RATE": "learning_rate",
            f"{prefix}MAX_EPOCHS": "max_epochs",
            f"{prefix}RANDOM_SEED": "random_seed",
            f"{prefix}CHECKPOINT_DIR": "checkpoint_dir",
            f"{prefix}DATA_DIR": "data_dir",
        }

        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion based on config key
                if config_key in ["debug"]:
                    env_config[config_key] = value.lower() in ("true", "1", "yes", "on")
                elif config_key in ["batch_size", "max_epochs", "random_seed"]:
                    env_config[config_key] = int(value)
                elif config_key in ["learning_rate"]:
                    env_config[config_key] = float(value)
                else:
                    env_config[config_key] = value

        # Merge with base configuration
        if env_config:
            logger.info(f"Loaded {len(env_config)} configuration values from environment")
            return self.base_config.update(**env_config)

        return self.base_config

    def get_config(
        self,
        config_file: Optional[Union[str, Path]] = None,
        env_prefix: str = "NEURAL_ARCH_",
        validate: bool = True
    ) -> Config:
        """Get complete configuration with all overrides applied.

        Priority order:
        1. Base configuration
        2. Configuration file
        3. Environment variables

        Args:
            config_file: Optional configuration file path
            env_prefix: Environment variable prefix
            validate: Whether to validate final configuration

        Returns:
            Final configuration with all overrides
        """
        config = self.base_config

        # Load from file if specified
        if config_file:
            file_config = self.load_from_file(config_file)
            config = config.merge(file_config)

        # Apply environment variable overrides
        config = self.load_from_env(env_prefix)

        # Validate final configuration
        if validate:
            config._validate()

        return config

    def watch_file(self, file_path: Union[str, Path], callback: callable) -> None:
        """Watch configuration file for changes (placeholder for future implementation).

        Args:
            file_path: Configuration file to watch
            callback: Function to call when file changes
        """
        # This would implement file system watching in a production system
        self._watchers.append(callback)
        logger.info(f"Added watcher for {file_path}")


# Convenience functions
def load_config(
    config_file: Optional[Union[str, Path]] = None,
    env_prefix: str = "NEURAL_ARCH_"
) -> Config:
    """Load configuration with file and environment overrides.

    Args:
        config_file: Optional configuration file path
        env_prefix: Environment variable prefix

    Returns:
        Complete configuration
    """
    manager = ConfigManager()
    return manager.get_config(config_file, env_prefix)


def save_config(config: Config, file_path: Union[str, Path]) -> None:
    """Save configuration to file.

    Args:
        config: Configuration to save
        file_path: File path to save to
    """
    manager = ConfigManager()
    manager.save_to_file(config, file_path)
