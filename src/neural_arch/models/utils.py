"""Utilities for model zoo including weight loading and model cards."""

import json
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError
import numpy as np

from ..core import Module, Parameter
from ..exceptions import ModelError

logger = logging.getLogger(__name__)

# Default paths
NEURAL_ARCH_HOME = Path(os.getenv('NEURAL_ARCH_HOME', '~/.neural_arch')).expanduser()
WEIGHTS_DIR = NEURAL_ARCH_HOME / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Weight hosting configuration
WEIGHT_URL_BASE = "https://github.com/neural-arch/model-weights/releases/download"


class ModelCard:
    """Model card containing metadata and documentation."""
    
    def __init__(
        self,
        name: str,
        description: str,
        architecture: str,
        paper_title: Optional[str] = None,
        paper_url: Optional[str] = None,
        original_implementation: Optional[str] = None,
        training_data: Optional[str] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        limitations: Optional[str] = None,
        biases: Optional[str] = None,
        carbon_footprint: Optional[str] = None,
        license: str = "MIT",
        citation: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.architecture = architecture
        self.paper_title = paper_title
        self.paper_url = paper_url
        self.original_implementation = original_implementation
        self.training_data = training_data
        self.preprocessing = preprocessing or {}
        self.metrics = metrics or {}
        self.limitations = limitations
        self.biases = biases
        self.carbon_footprint = carbon_footprint
        self.license = license
        self.citation = citation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'architecture': self.architecture,
            'paper_title': self.paper_title,
            'paper_url': self.paper_url,
            'original_implementation': self.original_implementation,
            'training_data': self.training_data,
            'preprocessing': self.preprocessing,
            'metrics': self.metrics,
            'limitations': self.limitations,
            'biases': self.biases,
            'carbon_footprint': self.carbon_footprint,
            'license': self.license,
            'citation': self.citation
        }
    
    def save(self, path: Path) -> None:
        """Save model card to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ModelCard':
        """Load model card from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def __str__(self) -> str:
        """Generate markdown representation."""
        lines = [
            f"# {self.name}",
            f"\n{self.description}\n",
            f"## Architecture\n{self.architecture}\n"
        ]
        
        if self.paper_title and self.paper_url:
            lines.append(f"## Paper\n[{self.paper_title}]({self.paper_url})\n")
        
        if self.original_implementation:
            lines.append(f"## Original Implementation\n{self.original_implementation}\n")
        
        if self.training_data:
            lines.append(f"## Training Data\n{self.training_data}\n")
        
        if self.preprocessing:
            lines.append("## Preprocessing")
            for key, value in self.preprocessing.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        if self.metrics:
            lines.append("## Performance Metrics")
            for metric, value in self.metrics.items():
                lines.append(f"- **{metric}**: {value}")
            lines.append("")
        
        if self.limitations:
            lines.append(f"## Limitations\n{self.limitations}\n")
        
        if self.biases:
            lines.append(f"## Biases\n{self.biases}\n")
        
        if self.carbon_footprint:
            lines.append(f"## Environmental Impact\n{self.carbon_footprint}\n")
        
        lines.append(f"## License\n{self.license}\n")
        
        if self.citation:
            lines.append(f"## Citation\n```\n{self.citation}\n```\n")
        
        return '\n'.join(lines)


def get_weight_url(model_name: str, config_name: str) -> str:
    """Get URL for model weights.
    
    Args:
        model_name: Model name
        config_name: Configuration name
        
    Returns:
        Weight file URL
    """
    filename = f"{model_name}_{config_name}.npz"
    return f"{WEIGHT_URL_BASE}/v1.0/{filename}"


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 8192,
    timeout: int = 300
) -> None:
    """Download file from URL with progress bar.
    
    Args:
        url: Source URL
        destination: Destination file path
        chunk_size: Download chunk size
        timeout: Request timeout in seconds
    """
    logger.info(f"Downloading from {url}")
    
    try:
        req = Request(url, headers={'User-Agent': 'neural-arch/1.0'})
        with urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                downloaded = 0
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Progress: {progress:.1f}%")
                
                tmp_path = Path(tmp_file.name)
            
            # Move to destination
            shutil.move(str(tmp_path), str(destination))
            logger.info(f"Downloaded to {destination}")
            
    except URLError as e:
        raise ModelError(f"Failed to download from {url}: {e}")
    except Exception as e:
        raise ModelError(f"Download error: {e}")


def compute_file_hash(path: Path, algorithm: str = 'sha256') -> str:
    """Compute hash of file.
    
    Args:
        path: File path
        algorithm: Hash algorithm
        
    Returns:
        Hex digest
    """
    hasher = hashlib.new(algorithm)
    
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    
    return hasher.hexdigest()


def download_weights(
    model_name: str,
    config_name: str,
    cache_dir: Optional[Path] = None,
    force_download: bool = False
) -> Path:
    """Download pretrained weights.
    
    Args:
        model_name: Model name
        config_name: Configuration name
        cache_dir: Cache directory (default: ~/.neural_arch/weights)
        force_download: Force re-download even if cached
        
    Returns:
        Path to downloaded weights
    """
    cache_dir = cache_dir or WEIGHTS_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_name}_{config_name}.npz"
    cache_path = cache_dir / filename
    
    # Check if already cached
    if cache_path.exists() and not force_download:
        logger.info(f"Using cached weights: {cache_path}")
        return cache_path
    
    # Download weights
    url = get_weight_url(model_name, config_name)
    download_file(url, cache_path)
    
    # Verify download (in production, check against known hashes)
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        raise ModelError(f"Downloaded file is empty or missing: {cache_path}")
    
    return cache_path


def load_pretrained_weights(
    model: Module,
    model_name: str,
    config_name: str,
    strict: bool = True,
    map_location: Optional[str] = None
) -> None:
    """Load pretrained weights into model.
    
    Args:
        model: Model instance
        model_name: Model name
        config_name: Configuration name
        strict: Require all keys to match
        map_location: Device to map weights to
    """
    # Download weights if needed
    weight_path = download_weights(model_name, config_name)
    
    # Load weights
    logger.info(f"Loading weights from {weight_path}")
    checkpoint = np.load(weight_path, allow_pickle=True)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict'].item()
    else:
        state_dict = dict(checkpoint)
    
    # Convert numpy arrays to parameters
    converted_state = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            converted_state[key] = value
        else:
            logger.warning(f"Skipping non-array value for key: {key}")
    
    # Load into model
    missing_keys = []
    unexpected_keys = []
    model_params = dict(model.named_parameters())
    
    for key, value in converted_state.items():
        if key in model_params:
            param = model_params[key]
            if param.shape != value.shape:
                raise ModelError(
                    f"Shape mismatch for {key}: "
                    f"model {param.shape} vs weights {value.shape}"
                )
            param.data = value
        else:
            unexpected_keys.append(key)
    
    for key in model_params:
        if key not in converted_state:
            missing_keys.append(key)
    
    if strict and (missing_keys or unexpected_keys):
        raise ModelError(
            f"Strict loading failed. "
            f"Missing: {missing_keys}, Unexpected: {unexpected_keys}"
        )
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    logger.info("Weights loaded successfully")


def save_weights(
    model: Module,
    path: Path,
    include_optimizer: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save model weights to file.
    
    Args:
        model: Model instance
        path: Output file path
        include_optimizer: Include optimizer state
        metadata: Additional metadata to save
    """
    # Get state dict
    state_dict = model.state_dict()
    
    # Prepare checkpoint
    checkpoint = {
        'state_dict': state_dict,
        'model_name': model.__class__.__name__,
        'neural_arch_version': '1.0.0'
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Save as numpy archive
    np.savez_compressed(path, **checkpoint)
    logger.info(f"Saved weights to {path}")


def create_model_card_from_model(
    model: Module,
    name: str,
    description: str,
    **kwargs
) -> ModelCard:
    """Create model card from model instance.
    
    Args:
        model: Model instance
        name: Model name
        description: Model description
        **kwargs: Additional model card fields
        
    Returns:
        ModelCard instance
    """
    # Analyze model architecture
    total_params = sum(p.size for p in model.parameters())
    architecture = f"{model.__class__.__name__} with {total_params:,} parameters"
    
    return ModelCard(
        name=name,
        description=description,
        architecture=architecture,
        **kwargs
    )