"""Global configuration system for neural architecture optimizations."""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization features."""
    
    # Backend selection
    auto_backend_selection: bool = True
    prefer_gpu: bool = True
    jit_threshold_elements: int = 10_000
    cuda_kernel_threshold_elements: int = 100_000
    
    # Operator fusion
    enable_fusion: bool = True
    fusion_patterns: Dict[str, bool] = None
    
    # Mixed precision
    enable_mixed_precision: bool = False
    mixed_precision_loss_scale: float = 65536.0
    mixed_precision_overflow_check: bool = True
    
    # Memory optimization
    enable_memory_pooling: bool = True
    enable_gradient_checkpointing: bool = False
    
    # Distributed training
    enable_distributed: bool = False
    distributed_backend: str = "nccl"
    
    # JIT compilation
    enable_jit: bool = True
    jit_cache_size: int = 1000
    
    def __post_init__(self):
        if self.fusion_patterns is None:
            self.fusion_patterns = {
                "linear_gelu": True,
                "linear_relu": True,
                "conv_bn_relu": True,
                "layernorm_linear": True,
            }


class GlobalConfig:
    """Global configuration manager for the neural architecture framework."""
    
    _instance: Optional['GlobalConfig'] = None
    _config: OptimizationConfig = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            cls._instance._config = OptimizationConfig()
        return cls._instance
    
    @property
    def optimization(self) -> OptimizationConfig:
        """Get optimization configuration."""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration
            for key, value in config_dict.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to JSON file."""
        try:
            config_dict = asdict(self._config)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Backend selection
        if os.getenv("NEURAL_ARCH_AUTO_BACKEND", "").lower() in ["false", "0"]:
            self._config.auto_backend_selection = False
        
        if os.getenv("NEURAL_ARCH_PREFER_CPU", "").lower() in ["true", "1"]:
            self._config.prefer_gpu = False
        
        # JIT compilation
        if os.getenv("NEURAL_ARCH_DISABLE_JIT", "").lower() in ["true", "1"]:
            self._config.enable_jit = False
        
        jit_threshold = os.getenv("NEURAL_ARCH_JIT_THRESHOLD")
        if jit_threshold:
            try:
                self._config.jit_threshold_elements = int(jit_threshold)
            except ValueError:
                logger.warning(f"Invalid JIT threshold: {jit_threshold}")
        
        # Operator fusion
        if os.getenv("NEURAL_ARCH_DISABLE_FUSION", "").lower() in ["true", "1"]:
            self._config.enable_fusion = False
        
        # Mixed precision
        if os.getenv("NEURAL_ARCH_MIXED_PRECISION", "").lower() in ["true", "1"]:
            self._config.enable_mixed_precision = True
        
        # Memory optimization
        if os.getenv("NEURAL_ARCH_DISABLE_MEMORY_POOLING", "").lower() in ["true", "1"]:
            self._config.enable_memory_pooling = False
        
        if os.getenv("NEURAL_ARCH_GRADIENT_CHECKPOINTING", "").lower() in ["true", "1"]:
            self._config.enable_gradient_checkpointing = True
        
        logger.debug("Loaded configuration from environment variables")
    
    def get_effective_backend_for_size(self, tensor_size: int) -> str:
        """Get effective backend recommendation for tensor size."""
        if not self._config.auto_backend_selection:
            return "numpy"
        
        if tensor_size > self._config.cuda_kernel_threshold_elements:
            return "cuda"  # Prefer CUDA for very large tensors
        elif tensor_size > self._config.jit_threshold_elements and self._config.enable_jit:
            return "jit"   # Use JIT for medium-large tensors
        else:
            return "numpy" # Use NumPy for small tensors
    
    def should_use_fusion(self, pattern: str) -> bool:
        """Check if a fusion pattern should be used."""
        if not self._config.enable_fusion:
            return False
        return self._config.fusion_patterns.get(pattern, False)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = OptimizationConfig()
        logger.info("Reset configuration to defaults")


# Global instance
config = GlobalConfig()

# Load from environment on import
config.load_from_env()


def configure(**kwargs) -> None:
    """Convenient function to update global configuration."""
    config.update_config(**kwargs)


def get_config() -> GlobalConfig:
    """Get the global configuration instance."""
    return config


def reset_config() -> None:
    """Reset configuration to defaults."""
    config.reset_to_defaults()