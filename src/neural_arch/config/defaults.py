"""Default configuration presets for different environments."""

from .config import Config

# Default configuration for general use
DEFAULT_CONFIG = Config(
    debug=False,
    log_level="INFO",
    device="cpu",
    dtype="float32",
    batch_size=32,
    learning_rate=0.001,
    max_epochs=100,
    model_name="neural_arch_model",
    checkpoint_dir="./checkpoints",
    data_dir="./data",
    gradient_clipping=True,
    gradient_clip_value=10.0,
    optimizer="adam",
    optimizer_params={"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    performance_thresholds={
        "tensor_creation_ms": 10.0,
        "matmul_ms": 100.0,
        "training_step_ms": 1000.0,
    },
    memory_monitoring=True,
    enable_profiling=False,
    benchmark_mode=False,
)


# Development configuration with debugging enabled
DEVELOPMENT_CONFIG = DEFAULT_CONFIG.update(
    debug=True,
    log_level="DEBUG",
    batch_size=16,  # Smaller batches for faster iteration
    max_epochs=10,  # Fewer epochs for quick testing
    save_frequency=1,  # Save more frequently
    enable_profiling=True,
    memory_monitoring=True,
    log_gradients=True,
    log_weights=True,
    metrics_frequency=10,  # More frequent metrics
    tensorboard_dir="./logs/dev",
)


# Production configuration optimized for performance
PRODUCTION_CONFIG = DEFAULT_CONFIG.update(
    debug=False,
    log_level="WARNING",  # Less verbose logging
    batch_size=64,  # Larger batches for efficiency
    max_epochs=1000,
    save_frequency=50,  # Save less frequently
    enable_profiling=False,
    memory_monitoring=False,  # Disable for performance
    log_gradients=False,
    log_weights=False,
    metrics_frequency=1000,  # Less frequent metrics
    benchmark_mode=True,  # Enable performance optimizations
    performance_thresholds={
        "tensor_creation_ms": 5.0,  # Stricter requirements
        "matmul_ms": 50.0,
        "training_step_ms": 500.0,
    },
)


# High-performance configuration for large-scale training
HIGH_PERFORMANCE_CONFIG = PRODUCTION_CONFIG.update(
    batch_size=128,
    gradient_clipping=False,  # May be disabled for speed
    memory_monitoring=False,
    enable_profiling=False,
    benchmark_mode=True,
    performance_thresholds={
        "tensor_creation_ms": 2.0,  # Very strict requirements
        "matmul_ms": 25.0,
        "training_step_ms": 250.0,
    },
    num_threads=None,  # Use all available cores
)


# Educational configuration for learning and experimentation
EDUCATIONAL_CONFIG = DEFAULT_CONFIG.update(
    debug=True,
    log_level="DEBUG",
    batch_size=8,  # Very small batches for easy understanding
    learning_rate=0.01,  # Higher learning rate for visible progress
    max_epochs=50,
    save_frequency=5,
    enable_profiling=True,
    memory_monitoring=True,
    log_gradients=True,
    log_weights=True,
    metrics_frequency=1,  # Log every step for detailed monitoring
    tensorboard_dir="./logs/educational",
    experiment_name="educational_experiment",
    tags=["education", "learning", "tutorial"],
)


# Research configuration for experiments
RESEARCH_CONFIG = DEFAULT_CONFIG.update(
    debug=True,
    log_level="INFO",
    save_frequency=10,
    enable_profiling=True,
    memory_monitoring=True,
    log_gradients=True,
    log_weights=False,
    metrics_frequency=100,
    tensorboard_dir="./logs/research",
    tags=["research", "experiment"],
    metadata={
        "research_purpose": "neural_architecture_experiment",
        "baseline_comparison": True,
    },
)


# Testing configuration for unit and integration tests
TESTING_CONFIG = Config(
    debug=True,
    log_level="ERROR",  # Suppress logs during testing
    device="cpu",  # Always use CPU for consistent testing
    dtype="float32",
    batch_size=4,  # Very small for fast tests
    learning_rate=0.1,  # High learning rate for quick convergence
    max_epochs=2,  # Minimal epochs
    patience=1,
    model_name="test_model",
    checkpoint_dir="./test_checkpoints",
    data_dir="./test_data",
    gradient_clipping=True,
    gradient_clip_value=1.0,  # Aggressive clipping for stability
    random_seed=42,  # Fixed seed for reproducible tests
    enable_profiling=False,
    memory_monitoring=False,
    benchmark_mode=False,
    save_frequency=1,
    metrics_frequency=1,
    performance_thresholds={
        "tensor_creation_ms": 100.0,  # Relaxed for testing
        "matmul_ms": 1000.0,
        "training_step_ms": 10000.0,
    },
)


# Configuration presets mapping
CONFIG_PRESETS = {
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


def get_preset_config(preset_name: str) -> Config:
    """Get a preset configuration by name.

    Args:
        preset_name: Name of the preset configuration

    Returns:
        Configuration preset

    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in CONFIG_PRESETS:
        available = ", ".join(CONFIG_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

    return CONFIG_PRESETS[preset_name]


def list_preset_configs() -> list:
    """List all available preset configuration names.

    Returns:
        List of preset names
    """
    return list(CONFIG_PRESETS.keys())
