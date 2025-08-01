"""Container modules for organizing multiple modules."""

from typing import Any, Iterator, List, Union
from ..core import Module


class ModuleList(Module):
    """A list container for modules.
    
    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.
    
    This is essential for proper parameter registration and gradient flow.
    """
    
    def __init__(self, modules=None) -> None:
        """Initialize ModuleList.
        
        Args:
            modules: An iterable of modules to add to the list
        """
        super().__init__()
        self._modules_list: List[Module] = []
        
        if modules is not None:
            self += modules
    
    def _get_abs_string_index(self, idx):
        """Get the absolute value of the string index."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f'index {idx} is out of range')
        if idx < 0:
            idx += len(self)
        return str(idx)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, 'ModuleList']:
        """Get module(s) by index."""
        if isinstance(idx, slice):
            return self.__class__(list(self._modules_list)[idx])
        else:
            if not isinstance(idx, int):
                raise TypeError(f'ModuleList indices must be integers, not {type(idx).__name__}')
            if not (-len(self._modules_list) <= idx < len(self._modules_list)):
                raise IndexError(f'ModuleList index out of range')
            if idx < 0:
                idx += len(self._modules_list)
            return self._modules_list[idx]
    
    def __setitem__(self, idx: int, module: Module) -> None:
        """Set module at index."""
        if not isinstance(module, Module):
            raise TypeError(f'ModuleList can only contain Module instances, got {type(module)}')
        
        if not isinstance(idx, int):
            raise TypeError(f'ModuleList indices must be integers, not {type(idx).__name__}')
        
        if not (-len(self._modules_list) <= idx < len(self._modules_list)):
            raise IndexError(f'ModuleList index out of range')
        
        if idx < 0:
            idx += len(self._modules_list)
        
        # Remove old module from registry
        old_name = str(idx)
        if old_name in self._modules:
            del self._modules[old_name]
        
        # Add new module
        self._modules_list[idx] = module
        self._modules[str(idx)] = module
    
    def __delitem__(self, idx: Union[int, slice]) -> None:
        """Delete module(s) at index."""
        if isinstance(idx, slice):
            for k in sorted(range(len(self._modules_list))[idx], reverse=True):
                delattr(self, str(k))
        else:
            if not isinstance(idx, int):
                raise TypeError(f'ModuleList indices must be integers, not {type(idx).__name__}')
            if not (-len(self._modules_list) <= idx < len(self._modules_list)):
                raise IndexError(f'ModuleList index out of range')
            if idx < 0:
                idx += len(self._modules_list)
            del self._modules[str(idx)]
            del self._modules_list[idx]
            # Shift all subsequent modules
            for i in range(idx, len(self._modules_list)):
                self._modules[str(i)] = self._modules[str(i + 1)]
                del self._modules[str(i + 1)]
    
    def __len__(self) -> int:
        """Return number of modules in the list."""
        return len(self._modules_list)
    
    def __iter__(self) -> Iterator[Module]:
        """Iterate over modules."""
        return iter(self._modules_list)
    
    def __iadd__(self, modules) -> 'ModuleList':
        """Add modules to the list in-place."""
        return self.extend(modules)
    
    def __add__(self, other) -> 'ModuleList':
        """Add modules to create a new ModuleList."""
        combined = ModuleList()
        combined.extend(self)
        combined.extend(other)
        return combined
    
    def append(self, module: Module) -> 'ModuleList':
        """Append a module to the end of the list.
        
        Args:
            module: Module to append
            
        Returns:
            Self for method chaining
        """
        if not isinstance(module, Module):
            raise TypeError(f'ModuleList can only contain Module instances, got {type(module)}')
        
        self._modules_list.append(module)
        self._modules[str(len(self._modules_list) - 1)] = module
        return self
    
    def extend(self, modules) -> 'ModuleList':
        """Extend the list with modules from an iterable.
        
        Args:
            modules: Iterable of modules to extend with
            
        Returns:
            Self for method chaining
        """
        if not isinstance(modules, (list, tuple)):
            modules = list(modules)
        
        for module in modules:
            self.append(module)
        return self
    
    def insert(self, index: int, module: Module) -> None:
        """Insert a module at a given index.
        
        Args:
            index: Index to insert at
            module: Module to insert
        """
        if not isinstance(module, Module):
            raise TypeError(f'ModuleList can only contain Module instances, got {type(module)}')
        
        # Insert in the list
        self._modules_list.insert(index, module)
        
        # Rebuild the _modules registry to maintain correct indices
        old_modules = list(self._modules_list)
        self._modules.clear()
        for i, mod in enumerate(old_modules):
            self._modules[str(i)] = mod
    
    def forward(self, x):
        """Forward pass - not typically used for ModuleList.
        
        This is provided for completeness but ModuleList is typically
        used as a container where individual modules are called directly.
        """
        for module in self._modules_list:
            x = module(x)
        return x


class Sequential(ModuleList):
    """A sequential container.
    
    Modules will be added to it in the order they are passed in the constructor.
    The forward() method accepts any input and forwards it to the first module it contains.
    It then "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.
    """
    
    def __init__(self, *args):
        """Initialize Sequential container.
        
        Args:
            *args: Modules to add sequentially
        """
        super().__init__()
        for idx, module in enumerate(args):
            self.append(module)
    
    def forward(self, x):
        """Forward pass through all modules sequentially."""
        for module in self._modules_list:
            x = module(x)
        return x


# Add operator import for _get_abs_string_index
import operator