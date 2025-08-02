"""Base module class for neural network components."""

from typing import Any, Dict, Iterator, List, Optional

from ..core.tensor import Tensor


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, "Module"] = {}
        self._training = True

    def __call__(self, *args, **kwargs):
        """Make the module callable."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Define the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def parameters(self) -> List[Tensor]:
        """Return an iterator over module parameters."""
        params = []

        # Add own parameters
        for param in self._parameters.values():
            params.append(param)

        # Add parameters from submodules
        for module in self._modules.values():
            params.extend(module.parameters())

        return params

    def named_parameters(self) -> Iterator[tuple]:
        """Return an iterator over module parameters, yielding parameter name and parameter."""
        for name, param in self._parameters.items():
            yield name, param

        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                yield f"{module_name}.{param_name}", param

    def train(self, mode: bool = True):
        """Set the module in training mode."""
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)

    def training(self) -> bool:
        """Return whether the module is in training mode."""
        return self._training

    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for param in self.parameters():
            if hasattr(param, "zero_grad"):
                param.zero_grad()

    def register_parameter(self, name: str, param: Optional[Tensor]):
        """Register a parameter to the module."""
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Tensor):
            raise TypeError(f"Expected Parameter or None, got {type(param)}")
        else:
            self._parameters[name] = param

    def register_module(self, name: str, module: Optional["Module"]):
        """Register a child module to the current module."""
        if module is None:
            self._modules[name] = None
        elif not isinstance(module, Module):
            raise TypeError(f"Expected Module or None, got {type(module)}")
        else:
            self._modules[name] = module

    def add_module(self, name: str, module: Optional["Module"]):
        """Add a child module to the current module."""
        self.register_module(name, module)

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Tensor):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        if "_parameters" in self.__dict__:
            parameters = self.__dict__["_parameters"]
            if name in parameters:
                return parameters[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def state_dict(self) -> Dict[str, Any]:
        """Return a dictionary containing a whole state of the module."""
        state = {}
        for name, param in self.named_parameters():
            state[name] = param.data
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Copy parameters from state_dict into this module and its descendants."""
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name]

    def __repr__(self):
        return f"{self.__class__.__name__}()"
