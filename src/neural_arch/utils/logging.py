"""
Structured logging configuration for Neural Forge

Provides consistent, structured logging across the entire framework with:
- JSON formatting for production environments
- Console formatting for development
- Performance tracking integration
- Security-aware logging (no sensitive data)
- Distributed training log aggregation
"""

import json
import logging
import logging.config
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union
import warnings

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")


class NeuralForgeFormatter(logging.Formatter):
    """Custom formatter for Neural Forge with structured output"""
    
    def __init__(self, use_json: bool = False, include_trace: bool = False):
        super().__init__()
        self.use_json = use_json
        self.include_trace = include_trace
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with Neural Forge structure"""
        
        # Create base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": os.getpid(),
            "thread_id": record.thread,
        }
        
        # Add Neural Forge specific context
        if hasattr(record, 'component'):
            log_data["component"] = record.component
            
        if hasattr(record, 'operation'):
            log_data["operation"] = record.operation
            
        if hasattr(record, 'device'):
            log_data["device"] = record.device
            
        if hasattr(record, 'model_name'):
            log_data["model_name"] = record.model_name
            
        if hasattr(record, 'batch_size'):
            log_data["batch_size"] = record.batch_size
            
        if hasattr(record, 'epoch'):
            log_data["epoch"] = record.epoch
            
        if hasattr(record, 'loss'):
            log_data["loss"] = record.loss
            
        if hasattr(record, 'performance_metrics'):
            log_data["performance"] = record.performance_metrics
        
        # Add exception information if present
        if record.exc_info and self.include_trace:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        if self.use_json:
            return json.dumps(log_data, default=str)
        else:
            # Human-readable format for console
            timestamp = log_data["timestamp"][:19]  # Remove microseconds
            level_color = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green  
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[41m', # Red background
            }.get(record.levelname, '')
            reset_color = '\033[0m'
            
            # Base format
            formatted = f"{timestamp} {level_color}{record.levelname:8}{reset_color} {record.name:20} {record.getMessage()}"
            
            # Add context if available
            context_parts = []
            if hasattr(record, 'component'):
                context_parts.append(f"component={record.component}")
            if hasattr(record, 'operation'): 
                context_parts.append(f"op={record.operation}")
            if hasattr(record, 'device'):
                context_parts.append(f"device={record.device}")
            if hasattr(record, 'epoch'):
                context_parts.append(f"epoch={record.epoch}")
            if hasattr(record, 'loss'):
                context_parts.append(f"loss={record.loss:.4f}")
                
            if context_parts:
                formatted += f" [{', '.join(context_parts)}]"
            
            # Add location for debug/error
            if record.levelno >= logging.ERROR or record.levelno <= logging.DEBUG:
                formatted += f" ({record.module}.{record.funcName}:{record.lineno})"
                
            return formatted


class PerformanceLogger:
    """Logger with built-in performance tracking"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation with context"""
        self.logger.info(
            f"Starting {operation}",
            extra={
                'operation': operation,
                'operation_status': 'started',
                **kwargs
            }
        )
        return time.time()
    
    def log_operation_end(self, operation: str, start_time: float, **kwargs):
        """Log the completion of an operation with timing"""
        duration = time.time() - start_time
        
        self.logger.info(
            f"Completed {operation} in {duration:.4f}s",
            extra={
                'operation': operation,
                'operation_status': 'completed', 
                'duration': duration,
                'performance_metrics': {
                    'duration_seconds': duration,
                    'throughput_ops_per_second': kwargs.get('ops_count', 1) / duration if duration > 0 else 0,
                },
                **kwargs
            }
        )
        return duration
    
    def log_training_step(self, epoch: int, batch_idx: int, loss: float, 
                         batch_size: int, learning_rate: float, **kwargs):
        """Log training step with comprehensive metrics"""
        self.logger.info(
            f"Training step - Epoch {epoch}, Batch {batch_idx}, Loss {loss:.6f}",
            extra={
                'component': 'training',
                'operation': 'training_step',
                'epoch': epoch,
                'batch_idx': batch_idx,
                'loss': loss,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                **kwargs
            }
        )
    
    def log_validation_step(self, epoch: int, val_loss: float, val_accuracy: Optional[float] = None, **kwargs):
        """Log validation step"""
        message = f"Validation - Epoch {epoch}, Loss {val_loss:.6f}"
        if val_accuracy is not None:
            message += f", Accuracy {val_accuracy:.4f}"
            
        self.logger.info(
            message,
            extra={
                'component': 'validation',
                'operation': 'validation_step',
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                **kwargs
            }
        )


def setup_logging(
    level: Union[str, int] = logging.INFO,
    use_json: bool = False,
    log_file: Optional[str] = None,
    include_trace: bool = False,
    disable_existing: bool = True,
) -> logging.Logger:
    """
    Set up structured logging for Neural Forge
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Whether to use JSON formatting (for production)
        log_file: Optional file to write logs to
        include_trace: Whether to include stack traces in logs
        disable_existing: Whether to disable existing loggers
        
    Returns:
        Configured logger instance
    """
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = NeuralForgeFormatter(use_json=use_json, include_trace=include_trace)
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers if requested
    if disable_existing:
        root_logger.handlers.clear()
    
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        # Always use JSON format for files
        file_formatter = NeuralForgeFormatter(use_json=True, include_trace=include_trace)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Create Neural Forge logger
    logger = logging.getLogger('neural_forge')
    logger.setLevel(level)
    
    logger.info(
        "Neural Forge logging initialized",
        extra={
            'component': 'logging',
            'operation': 'initialization',
            'level': logging.getLevelName(level),
            'json_format': use_json,
            'log_file': log_file,
            'include_trace': include_trace,
        }
    )
    
    return logger


def get_logger(name: str = 'neural_forge') -> logging.Logger:
    """Get a logger instance with Neural Forge configuration"""
    return logging.getLogger(name)


def get_performance_logger(name: str = 'neural_forge') -> PerformanceLogger:
    """Get a performance logger instance"""
    logger = get_logger(name)
    return PerformanceLogger(logger)


# Context managers for logging operations
@contextmanager
def log_operation(operation_name: str, logger: Optional[logging.Logger] = None, **kwargs):
    """Context manager for logging operations with timing"""
    if logger is None:
        logger = get_logger()
    
    perf_logger = PerformanceLogger(logger)
    start_time = perf_logger.log_operation_start(operation_name, **kwargs)
    
    try:
        yield
        perf_logger.log_operation_end(operation_name, start_time, **kwargs)
    except Exception as e:
        perf_logger.log_operation_end(operation_name, start_time, error=str(e), **kwargs)
        logger.error(
            f"Operation {operation_name} failed: {e}",
            extra={'operation': operation_name, 'error': str(e)},
            exc_info=True
        )
        raise


# Environment-based configuration
def configure_logging_from_env():
    """Configure logging based on environment variables"""
    
    # Get configuration from environment
    log_level = os.getenv('NEURAL_FORGE_LOG_LEVEL', 'INFO').upper()
    use_json = os.getenv('NEURAL_FORGE_LOG_JSON', '').lower() in ('true', '1', 'yes')
    log_file = os.getenv('NEURAL_FORGE_LOG_FILE')
    include_trace = os.getenv('NEURAL_FORGE_LOG_TRACE', '').lower() in ('true', '1', 'yes')
    
    # Set up logging
    logger = setup_logging(
        level=log_level,
        use_json=use_json,
        log_file=log_file,
        include_trace=include_trace
    )
    
    logger.info(
        "Logging configured from environment variables",
        extra={
            'component': 'logging',
            'log_level': log_level,
            'use_json': use_json,
            'log_file': log_file,
            'include_trace': include_trace,
        }
    )
    
    return logger


# Initialize logging on import if not already configured
if not logging.getLogger().handlers:
    configure_logging_from_env()


# Export main interface
__all__ = [
    'setup_logging',
    'get_logger', 
    'get_performance_logger',
    'PerformanceLogger',
    'log_operation',
    'configure_logging_from_env',
    'NeuralForgeFormatter',
]