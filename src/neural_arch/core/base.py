"""Base classes and abstractions for neural network components."""

import abc
from typing import Dict, Any, Optional, Iterator, Tuple, Protocol
from collections import OrderedDict

from .tensor import Tensor


class ParameterDict:
    """A dictionary-like object that also supports iteration for parameters."""
    
    def __init__(self, module: 'Module', recurse: bool = True):
        self.module = module
        self.recurse = recurse
        self._params = None
    
    def _get_params(self):
        """Lazy computation of parameter dictionary."""
        if self._params is None:
            self._params = {}
            for name, param in self.module._parameters.items():
                self._params[name] = param
            
            if self.recurse:
                for name, submodule in self.module._modules.items():
                    sub_params = submodule.parameters(recurse=True)
                    if hasattr(sub_params, 'items'):
                        for sub_name, sub_param in sub_params.items():
                            self._params[f"{name}.{sub_name}"] = sub_param
                    else:
                        # Handle iterator case
                        for i, sub_param in enumerate(sub_params):
                            self._params[f"{name}.param_{i}"] = sub_param
        return self._params
    
    def keys(self):
        return self._get_params().keys()
    
    def values(self):
        return self._get_params().values()
    
    def items(self):
        return self._get_params().items()
    
    def __getitem__(self, key):
        return self._get_params()[key]
    
    def __contains__(self, key):
        return key in self._get_params()
    
    def __iter__(self):
        """Allow iteration over parameter values (for optimizer compatibility)."""
        return iter(self._get_params().values())
    
    def __len__(self):
        return len(self._get_params())


class Parameter(Tensor):
    """A tensor that is automatically registered as a module parameter.
    
    Parameters are special tensors that:
    - Always require gradients (requires_grad=True)
    - Are automatically registered when assigned to modules
    - Are included in parameter() iterations
    - Are saved/loaded with model state
    """
    
    def __init__(self, data, name: Optional[str] = None) -> None:
        """Initialize parameter tensor.
        
        Args:
            data: Initial tensor data
            name: Optional parameter name for debugging
        """
        super().__init__(data, requires_grad=True, name=name)
        # The name is already set by the parent Tensor class
    
    def __repr__(self) -> str:
        name_str = f", name={self.name!r}" if self.name else ""
        return f"Parameter(shape={self.shape}{name_str})"


class ModuleMeta(type):
    """Metaclass for Module to enable automatic parameter registration."""
    
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        # Auto-register parameters after initialization
        instance._register_parameters()
        return instance


class Module(metaclass=ModuleMeta):
    """Base class for all neural network modules.
    
    This is the enterprise-grade base class that provides:
    - Automatic parameter registration and management
    - Hierarchical module structure
    - Training/evaluation mode switching
    - State dict save/load functionality
    - Device placement management
    - Forward hook system for debugging/monitoring
    """
    
    def __init__(self) -> None:
        """Initialize the module."""
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._modules: OrderedDict[str, 'Module'] = OrderedDict()
        self._training: bool = True
        self._hooks: Dict[str, list] = {"forward": [], "backward": []}
    
    def _register_parameters(self) -> None:
        """Automatically register Parameter instances as module parameters."""
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                self._parameters[name] = value
                # Update the private _name attribute instead of the property
                value._name = f"{self.__class__.__name__}.{name}"
            elif isinstance(value, Module):
                self._modules[name] = value
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to automatically register parameters and modules."""
        super().__setattr__(name, value)
        
        # Only register if the module is already initialized
        if hasattr(self, '_parameters'):
            if isinstance(value, Parameter):
                self._parameters[name] = value
                # Update the private _name attribute instead of the property
                value._name = f"{self.__class__.__name__}.{name}"
            elif isinstance(value, Module):
                self._modules[name] = value
    
    def parameters(self, recurse: bool = True):
        """Return module parameters.
        
        This returns a special object that can be used both as a dictionary
        (for tests) and as an iterator (for training loops).
        
        Args:
            recurse: If True, includes parameters of submodules recursively
            
        Returns:
            ParameterDict object that supports both dict and iterator interfaces
        """
        return ParameterDict(self, recurse)
    
    def _parameters_iterator(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters."""
        for param in self._parameters.values():
            yield param
        
        if recurse:
            for module in self._modules.values():
                if hasattr(module, '_parameters_iterator'):
                    yield from module._parameters_iterator(recurse=True)
                else:
                    # Fallback for compatibility
                    params = module.parameters(recurse=True)
                    if hasattr(params, '__iter__') and not isinstance(params, dict):
                        yield from params
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Return an iterator over module parameters with their names.
        
        Args:
            prefix: Prefix to prepend to parameter names
            recurse: If True, yields parameters of submodules recursively
            
        Yields:
            Tuple[str, Parameter]: (name, parameter) pairs
        """
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        
        if recurse:
            for name, module in self._modules.items():
                module_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_parameters(module_prefix, recurse=True)
    
    def modules(self) -> Iterator['Module']:
        """Return an iterator over all modules in the network.
        
        Yields:
            Module: All modules including self and submodules
        """
        yield self
        for module in self._modules.values():
            yield from module.modules()
    
    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """Return an iterator over all modules with their names.
        
        Args:
            prefix: Prefix to prepend to module names
            
        Yields:
            Tuple[str, Module]: (name, module) pairs
        """
        yield prefix, self
        for name, module in self._modules.items():
            module_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(module_prefix)
    
    def train(self, mode: bool = True) -> 'Module':
        """Set the module in training mode.
        
        Args:
            mode: If True, sets training mode; if False, sets evaluation mode
            
        Returns:
            Self for method chaining
        """
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set the module in evaluation mode.
        
        Returns:
            Self for method chaining
        """
        return self.train(False)
    
    @property
    def training(self) -> bool:
        """Whether the module is in training mode."""
        return self._training
    
    def state_dict(self) -> Dict[str, Any]:
        """Return a dictionary containing module state.
        
        Returns:
            Dictionary with parameter values and submodule states
        """
        state = {}
        
        # Add parameters
        for name, param in self._parameters.items():
            state[name] = param.data.copy()
        
        # Add submodule states
        for name, module in self._modules.items():
            for sub_name, sub_state in module.state_dict().items():
                state[f"{name}.{sub_name}"] = sub_state
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load module state from a state dictionary.
        
        Args:
            state_dict: Dictionary containing parameter values
            strict: If True, requires exact key matching
            
        Raises:
            KeyError: If strict=True and keys don't match
        """
        current_state = self.state_dict()
        
        if strict:
            missing_keys = set(current_state.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(current_state.keys())
            
            if missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")
        
        # Load parameter values
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].copy()
    
    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for param in self.parameters():
            param.zero_grad()
    
    def register_forward_hook(self, hook: callable) -> None:
        """Register a forward hook.
        
        Args:
            hook: Function called with (module, input, output) after forward pass
        """
        self._hooks["forward"].append(hook)
    
    def register_backward_hook(self, hook: callable) -> None:
        """Register a backward hook.
        
        Args:
            hook: Function called with (module, grad_input, grad_output) during backward pass
        """
        self._hooks["backward"].append(hook)
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Define the forward computation.
        
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Call the forward method with hook support.
        
        This allows modules to be called like functions while providing
        enterprise features like hooks for monitoring and debugging.
        """
        # Pre-forward hooks could go here
        
        result = self.forward(*args, **kwargs)
        
        # Post-forward hooks
        for hook in self._hooks["forward"]:
            hook(self, args, result)
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the module."""
        params = sum(p.data.size for p in self.parameters())
        return f"{self.__class__.__name__}(parameters={params:,})"


class Optimizer(abc.ABC):
    """Abstract base class for all optimizers.
    
    Enterprise-grade optimizer interface providing:
    - Consistent API across all optimizers
    - State management and serialization
    - Parameter group support
    - Learning rate scheduling integration
    """
    
    def __init__(self, parameters, **kwargs) -> None:
        """Initialize optimizer.
        
        Args:
            parameters: Parameters to optimize (dict or iterator)
            **kwargs: Optimizer-specific arguments
        """
        # Handle both dict and iterator inputs
        if hasattr(parameters, 'items'):
            self.parameters = parameters
        else:
            self.parameters = {f"param_{i}": param for i, param in enumerate(parameters)}
        
        self.state: Dict[str, Any] = {}
        self.defaults = kwargs
    
    @abc.abstractmethod
    def step(self) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError("Subclasses must implement step()")
    
    @abc.abstractmethod
    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        raise NotImplementedError("Subclasses must implement zero_grad()")
    
    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dictionary."""
        return {
            'state': self.state,
            'defaults': self.defaults,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from dictionary."""
        self.state = state_dict['state']
        self.defaults = state_dict['defaults']