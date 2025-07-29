"""Main CLI interface for Neural Architecture."""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from ..__version__ import __version__
from ..config import load_config, get_preset_config, list_preset_configs
from ..exceptions import NeuralArchError
from .commands import (
    train_command, test_command, benchmark_command, 
    info_command, config_command, export_command
)

logger = logging.getLogger(__name__)


def setup_logging(level: str, format_style: str = "standard") -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Logging format style (standard, detailed, json)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    if format_style == "detailed":
        log_format = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
    elif format_style == "json":
        # In a real implementation, this would use structured JSON logging
        log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    else:  # standard
        log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="neural-arch",
        description="Enterprise-grade neural network implementation from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neural-arch train --config config.yaml --epochs 100
  neural-arch test --model checkpoint.pt --data test_data/
  neural-arch benchmark --operations matmul,softmax
  neural-arch info --system --performance
  neural-arch config create --preset production --output prod_config.yaml

For more information, visit: https://github.com/your-org/neural-arch
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--version",
        action="version",
        version=f"Neural Architecture {__version__}"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file path (.json or .yaml)"
    )
    
    parser.add_argument(
        "--preset",
        choices=list_preset_configs(),
        help="Use a preset configuration"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-format",
        choices=["standard", "detailed", "json"],
        default="standard",
        help="Set logging format (default: standard)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (equivalent to --log-level DEBUG)"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Available commands",
        help="Command to run"
    )
    
    # Training command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a neural network model",
        description="Train a neural network with specified configuration"
    )
    train_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data directory or file"
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default="./models",
        help="Output directory for trained model (default: ./models)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (overrides config)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="Resume training from checkpoint"
    )
    train_parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation during training"
    )
    train_parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging"
    )
    
    # Testing command
    test_parser = subparsers.add_parser(
        "test",
        help="Test a trained model",
        description="Evaluate model performance on test data"
    )
    test_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model file"
    )
    test_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to test data directory or file"
    )
    test_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for test results"
    )
    test_parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "loss"],
        help="Metrics to compute (default: accuracy loss)"
    )
    test_parser.add_argument(
        "--batch-size",
        type=int,
        help="Test batch size (overrides config)"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmarks",
        description="Benchmark tensor operations and model performance"
    )
    benchmark_parser.add_argument(
        "--operations",
        nargs="+",
        default=["all"],
        help="Operations to benchmark (default: all)"
    )
    benchmark_parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 500, 1000],
        help="Tensor sizes to test (default: 100 500 1000)"
    )
    benchmark_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)"
    )
    benchmark_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for benchmark results"
    )
    benchmark_parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system and library information",
        description="Display system information, capabilities, and library details"
    )
    info_parser.add_argument(
        "--system",
        action="store_true",
        help="Show system information"
    )
    info_parser.add_argument(
        "--performance",
        action="store_true",
        help="Show performance capabilities"
    )
    info_parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration"
    )
    info_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all information"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration files",
        description="Create, validate, and manage configuration files"
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_action",
        title="Config Actions",
        help="Configuration action to perform"
    )
    
    # Config create
    create_config_parser = config_subparsers.add_parser(
        "create",
        help="Create a new configuration file"
    )
    create_config_parser.add_argument(
        "--preset",
        choices=list_preset_configs(),
        default="default",
        help="Base preset to use (default: default)"
    )
    create_config_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output configuration file path"
    )
    create_config_parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="yaml",
        help="Configuration file format (default: yaml)"
    )
    
    # Config validate
    validate_config_parser = config_subparsers.add_parser(
        "validate",
        help="Validate a configuration file"
    )
    validate_config_parser.add_argument(
        "config_file",
        type=Path,
        help="Configuration file to validate"
    )
    
    # Config show
    show_config_parser = config_subparsers.add_parser(
        "show",
        help="Show current configuration"
    )
    show_config_parser.add_argument(
        "--format",
        choices=["json", "yaml", "table"],
        default="yaml",
        help="Output format (default: yaml)"
    )
    
    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export models to different formats",
        description="Export trained models to various deployment formats"
    )
    export_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model file"
    )
    export_parser.add_argument(
        "--format",
        choices=["onnx", "tensorrt", "coreml", "tflite"],
        required=True,
        help="Export format"
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path"
    )
    export_parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimizations during export"
    )
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Handle logging setup
    log_level = parsed_args.log_level
    if parsed_args.quiet:
        log_level = "ERROR"
    elif parsed_args.verbose:
        log_level = "DEBUG"
    
    setup_logging(log_level, parsed_args.log_format)
    
    # Load configuration
    try:
        if parsed_args.preset:
            config = get_preset_config(parsed_args.preset)
        elif parsed_args.config:
            config = load_config(parsed_args.config)
        else:
            config = load_config()  # Use defaults with env overrides
        
        logger.info(f"Loaded configuration: preset={parsed_args.preset}, file={parsed_args.config}")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Route to appropriate command handler
    try:
        if parsed_args.command == "train":
            return train_command(parsed_args, config)
        elif parsed_args.command == "test":
            return test_command(parsed_args, config)
        elif parsed_args.command == "benchmark":
            return benchmark_command(parsed_args, config)
        elif parsed_args.command == "info":
            return info_command(parsed_args, config)
        elif parsed_args.command == "config":
            return config_command(parsed_args, config)
        elif parsed_args.command == "export":
            return export_command(parsed_args, config)
        else:
            parser.print_help()
            return 1
            
    except NeuralArchError as e:
        logger.error(f"Neural Architecture Error: {e}")
        if log_level == "DEBUG":
            logger.exception("Full traceback:")
        return 1
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if log_level == "DEBUG":
            logger.exception("Full traceback:")
        return 1


def create_cli() -> argparse.ArgumentParser:
    """Create CLI parser for external use.
    
    Returns:
        Configured argument parser
    """
    return create_parser()


if __name__ == "__main__":
    sys.exit(main())