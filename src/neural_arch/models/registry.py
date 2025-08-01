"""Model Registry for managing and accessing models in the zoo."""

import json
import logging
from typing import Dict, Any, Optional, List, Callable, Type
from pathlib import Path
from dataclasses import dataclass, asdict

from ..core import Module
from ..exceptions import ModelError

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    model_class: Type[Module]
    description: str
    paper_url: Optional[str] = None
    pretrained_configs: Dict[str, Dict[str, Any]] = None
    default_config: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.pretrained_configs is None:
            self.pretrained_configs = {}
        if self.tags is None:
            self.tags = []


class ModelRegistry:
    """Central registry for all models in the zoo."""
    
    _models: Dict[str, ModelInfo] = {}
    _aliases: Dict[str, str] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[Module],
        description: str,
        paper_url: Optional[str] = None,
        pretrained_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        default_config: Optional[str] = None,
        tags: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None
    ) -> None:
        """Register a model in the zoo.
        
        Args:
            name: Unique model name
            model_class: Model class (must inherit from Module)
            description: Model description
            paper_url: URL to the paper
            pretrained_configs: Available pretrained configurations
            default_config: Default configuration name
            tags: Model tags for categorization
            aliases: Alternative names for the model
        """
        if name in cls._models:
            raise ValueError(f"Model '{name}' is already registered")
        
        if not issubclass(model_class, Module):
            raise TypeError(f"Model class must inherit from Module, got {model_class}")
        
        # Validate pretrained configs
        if pretrained_configs and default_config:
            if default_config not in pretrained_configs:
                raise ValueError(f"Default config '{default_config}' not in pretrained_configs")
        
        model_info = ModelInfo(
            name=name,
            model_class=model_class,
            description=description,
            paper_url=paper_url,
            pretrained_configs=pretrained_configs or {},
            default_config=default_config,
            tags=tags or []
        )
        
        cls._models[name] = model_info
        
        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in cls._aliases:
                    raise ValueError(f"Alias '{alias}' is already registered")
                cls._aliases[alias] = name
        
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def get_model_info(cls, name: str) -> ModelInfo:
        """Get model information by name or alias.
        
        Args:
            name: Model name or alias
            
        Returns:
            ModelInfo object
            
        Raises:
            ModelError: If model not found
        """
        # Check if it's an alias
        if name in cls._aliases:
            name = cls._aliases[name]
        
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ModelError(
                f"Model '{name}' not found. Available models: {available}"
            )
        
        return cls._models[name]
    
    @classmethod
    def create_model(
        cls,
        name: str,
        pretrained: bool = False,
        config_name: Optional[str] = None,
        **kwargs
    ) -> Module:
        """Create a model instance.
        
        Args:
            name: Model name or alias
            pretrained: Whether to load pretrained weights
            config_name: Pretrained configuration name
            **kwargs: Additional arguments for model construction
            
        Returns:
            Model instance
        """
        model_info = cls.get_model_info(name)
        
        # Get configuration
        if pretrained:
            if not model_info.pretrained_configs:
                raise ModelError(f"No pretrained configs available for '{name}'")
            
            config_name = config_name or model_info.default_config
            if not config_name:
                configs = list(model_info.pretrained_configs.keys())
                raise ModelError(
                    f"No config specified. Available: {configs}"
                )
            
            if config_name not in model_info.pretrained_configs:
                configs = list(model_info.pretrained_configs.keys())
                raise ModelError(
                    f"Config '{config_name}' not found. Available: {configs}"
                )
            
            # Merge pretrained config with user kwargs
            config = model_info.pretrained_configs[config_name].copy()
            config.update(kwargs)
            kwargs = config
        
        # Create model instance
        try:
            model = model_info.model_class(**kwargs)
        except Exception as e:
            raise ModelError(f"Failed to create model '{name}': {e}")
        
        # Load pretrained weights if requested
        if pretrained:
            from .utils import load_pretrained_weights
            load_pretrained_weights(model, name, config_name)
        
        return model
    
    @classmethod
    def list_models(
        cls,
        tags: Optional[List[str]] = None,
        with_pretrained: Optional[bool] = None
    ) -> List[str]:
        """List available models.
        
        Args:
            tags: Filter by tags
            with_pretrained: Filter by pretrained availability
            
        Returns:
            List of model names
        """
        models = []
        
        for name, info in cls._models.items():
            # Filter by tags
            if tags:
                if not any(tag in info.tags for tag in tags):
                    continue
            
            # Filter by pretrained availability
            if with_pretrained is not None:
                has_pretrained = bool(info.pretrained_configs)
                if has_pretrained != with_pretrained:
                    continue
            
            models.append(name)
        
        return sorted(models)
    
    @classmethod
    def get_model_card(cls, name: str) -> Dict[str, Any]:
        """Get model card with full information.
        
        Args:
            name: Model name or alias
            
        Returns:
            Model card dictionary
        """
        info = cls.get_model_info(name)
        
        card = {
            'name': info.name,
            'description': info.description,
            'paper_url': info.paper_url,
            'tags': info.tags,
            'pretrained_configs': list(info.pretrained_configs.keys()),
            'default_config': info.default_config,
            'aliases': [k for k, v in cls._aliases.items() if v == info.name]
        }
        
        # Add configuration details
        if info.pretrained_configs:
            card['configurations'] = {}
            for config_name, config in info.pretrained_configs.items():
                card['configurations'][config_name] = config
        
        return card
    
    @classmethod
    def save_registry(cls, path: Path) -> None:
        """Save registry to JSON file.
        
        Args:
            path: Output file path
        """
        registry_data = {
            'models': {},
            'aliases': cls._aliases
        }
        
        for name, info in cls._models.items():
            registry_data['models'][name] = {
                'description': info.description,
                'paper_url': info.paper_url,
                'pretrained_configs': info.pretrained_configs,
                'default_config': info.default_config,
                'tags': info.tags
            }
        
        with open(path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._models.clear()
        cls._aliases.clear()


def register_model(
    name: str,
    description: str,
    paper_url: Optional[str] = None,
    pretrained_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    default_config: Optional[str] = None,
    tags: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None
) -> Callable:
    """Decorator to register a model class.
    
    Example:
        @register_model(
            name='resnet50',
            description='ResNet-50 from Deep Residual Learning for Image Recognition',
            paper_url='https://arxiv.org/abs/1512.03385',
            pretrained_configs={
                'imagenet': {'num_classes': 1000, 'input_size': 224}
            },
            default_config='imagenet',
            tags=['vision', 'classification', 'resnet']
        )
        class ResNet50(Module):
            ...
    """
    def decorator(model_class: Type[Module]) -> Type[Module]:
        ModelRegistry.register(
            name=name,
            model_class=model_class,
            description=description,
            paper_url=paper_url,
            pretrained_configs=pretrained_configs,
            default_config=default_config,
            tags=tags,
            aliases=aliases
        )
        return model_class
    
    return decorator


def get_model(
    name: str,
    pretrained: bool = False,
    config_name: Optional[str] = None,
    **kwargs
) -> Module:
    """Get a model from the registry.
    
    Args:
        name: Model name or alias
        pretrained: Whether to load pretrained weights
        config_name: Pretrained configuration name
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    return ModelRegistry.create_model(
        name=name,
        pretrained=pretrained,
        config_name=config_name,
        **kwargs
    )


def list_models(
    tags: Optional[List[str]] = None,
    with_pretrained: Optional[bool] = None
) -> List[str]:
    """List available models.
    
    Args:
        tags: Filter by tags
        with_pretrained: Filter by pretrained availability
        
    Returns:
        List of model names
    """
    return ModelRegistry.list_models(tags=tags, with_pretrained=with_pretrained)