"""Comprehensive tests for core/base.py to improve coverage from 73.52% to 95%+.

This file tests Module, Parameter, and Optimizer base classes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.core.base import Module, Parameter, Optimizer
from neural_arch.core.tensor import Tensor


class SimpleModule(Module):
    """Simple module for testing."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(np.random.randn(in_features, out_features).astype(np.float32))
        self.bias = Parameter(np.zeros(out_features).astype(np.float32))
    
    def forward(self, x):
        return x @ self.weight.data + self.bias.data


class NestedModule(Module):
    """Module containing other modules."""
    def __init__(self):
        super().__init__()
        self.layer1 = SimpleModule(10, 5)
        self.layer2 = SimpleModule(5, 2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ModuleWithoutParams(Module):
    """Module without parameters."""
    def forward(self, x):
        return x * 2


class TestModuleBase:
    """Test Module base class functionality."""
    
    def test_module_basic_functionality(self):
        """Test basic module functionality."""
        module = SimpleModule(3, 2)
        
        # Check attributes exist
        assert hasattr(module, '_parameters')
        assert hasattr(module, '_modules')
        assert hasattr(module, 'training')
        
        # Check parameters registered
        assert 'weight' in module._parameters
        assert 'bias' in module._parameters
        assert isinstance(module.weight, Parameter)
        assert isinstance(module.bias, Parameter)
    
    def test_module_parameters_method(self):
        """Test parameters() method."""
        module = SimpleModule(3, 2)
        
        # Get all parameters
        params = list(module.parameters())
        assert len(params) == 2
        assert module.weight in params
        assert module.bias in params
        
        # Test with recurse=False
        params_no_recurse = list(module.parameters(recurse=False))
        assert len(params_no_recurse) == 2
    
    def test_module_parameters_iterator(self):
        """Test _parameters_iterator method."""
        module = NestedModule()
        
        # Test iterator directly
        params = list(module._parameters_iterator(recurse=True))
        assert len(params) == 4  # 2 params from each layer
        
        # Test without recursion
        params_no_recurse = list(module._parameters_iterator(recurse=False))
        assert len(params_no_recurse) == 0  # NestedModule has no direct params
    
    def test_module_parameters_iterator_fallback(self):
        """Test _parameters_iterator fallback for compatibility."""
        # Create a mock module without _parameters_iterator
        class LegacyModule:
            def parameters(self, recurse=True):
                return [Parameter(np.array([1, 2, 3]))]
        
        module = NestedModule()
        legacy = LegacyModule()
        module._modules['legacy'] = legacy
        
        # Should use fallback
        params = list(module._parameters_iterator(recurse=True))
        assert len(params) == 5  # 4 from nested + 1 from legacy
    
    def test_module_named_parameters(self):
        """Test named_parameters method."""
        module = SimpleModule(3, 2)
        
        # Get named parameters
        named_params = list(module.named_parameters())
        assert len(named_params) == 2
        
        names = [name for name, _ in named_params]
        assert 'weight' in names
        assert 'bias' in names
        
        # Test with prefix
        named_params_prefix = list(module.named_parameters(prefix='layer'))
        names_prefix = [name for name, _ in named_params_prefix]
        assert 'layer.weight' in names_prefix
        assert 'layer.bias' in names_prefix
    
    def test_module_named_parameters_nested(self):
        """Test named_parameters with nested modules."""
        module = NestedModule()
        
        # Get all named parameters
        named_params = list(module.named_parameters())
        assert len(named_params) == 4
        
        names = [name for name, _ in named_params]
        assert 'layer1.weight' in names
        assert 'layer1.bias' in names
        assert 'layer2.weight' in names
        assert 'layer2.bias' in names
        
        # Test without recursion
        named_params_no_recurse = list(module.named_parameters(recurse=False))
        assert len(named_params_no_recurse) == 0
    
    def test_module_modules_method(self):
        """Test modules() method."""
        module = NestedModule()
        
        # Get all modules
        modules = list(module.modules())
        assert len(modules) == 3  # self + layer1 + layer2
        assert module in modules
        assert module.layer1 in modules
        assert module.layer2 in modules
    
    def test_module_named_modules(self):
        """Test named_modules method."""
        module = NestedModule()
        
        # Get named modules
        named_modules = list(module.named_modules())
        assert len(named_modules) == 3
        
        names = [name for name, _ in named_modules]
        assert '' in names  # Root module
        assert 'layer1' in names
        assert 'layer2' in names
        
        # Test with prefix
        named_modules_prefix = list(module.named_modules(prefix='model'))
        names_prefix = [name for name, _ in named_modules_prefix]
        assert 'model' in names_prefix
        assert 'model.layer1' in names_prefix
        assert 'model.layer2' in names_prefix
    
    def test_module_train_eval_modes(self):
        """Test train() and eval() methods."""
        module = SimpleModule(3, 2)
        
        # Default should be training
        assert module.training is True
        
        # Switch to eval
        module.eval()
        assert module.training is False
        
        # Switch back to train
        module.train()
        assert module.training is True
        
        # Test with argument
        module.train(False)
        assert module.training is False
        
        module.train(True)
        assert module.training is True
    
    def test_module_train_eval_recursive(self):
        """Test train/eval propagates to submodules."""
        module = NestedModule()
        
        # All should be in training mode
        assert module.training is True
        assert module.layer1.training is True
        assert module.layer2.training is True
        
        # Switch to eval
        module.eval()
        assert module.training is False
        assert module.layer1.training is False
        assert module.layer2.training is False
        
        # Switch back
        module.train()
        assert module.training is True
        assert module.layer1.training is True
        assert module.layer2.training is True
    
    def test_module_setattr_parameter(self):
        """Test __setattr__ with Parameter."""
        module = Module()
        
        # Add parameter via setattr
        param = Parameter(np.array([1, 2, 3]))
        module.param = param
        
        assert 'param' in module._parameters
        assert module._parameters['param'] is param
    
    def test_module_setattr_module(self):
        """Test __setattr__ with Module."""
        module = Module()
        
        # Add submodule via setattr
        submodule = SimpleModule(3, 2)
        module.submodule = submodule
        
        assert 'submodule' in module._modules
        assert module._modules['submodule'] is submodule
    
    def test_module_setattr_regular_attribute(self):
        """Test __setattr__ with regular attributes."""
        module = Module()
        
        # Add regular attribute
        module.some_value = 42
        assert module.some_value == 42
        assert 'some_value' not in module._parameters
        assert 'some_value' not in module._modules
    
    def test_module_zero_grad(self):
        """Test zero_grad method."""
        module = SimpleModule(3, 2)
        
        # Set some gradients
        module.weight.grad = np.ones_like(module.weight.data)
        module.bias.grad = np.ones_like(module.bias.data)
        
        # Zero gradients
        module.zero_grad()
        
        assert module.weight.grad is None
        assert module.bias.grad is None
    
    def test_module_zero_grad_nested(self):
        """Test zero_grad on nested modules."""
        module = NestedModule()
        
        # Set gradients on all parameters
        for param in module.parameters():
            param.grad = np.ones_like(param.data)
        
        # Zero all gradients
        module.zero_grad()
        
        # Check all gradients are None
        for param in module.parameters():
            assert param.grad is None
    
    def test_module_state_dict(self):
        """Test state_dict method."""
        module = SimpleModule(3, 2)
        
        # Get state dict
        state = module.state_dict()
        
        assert 'weight' in state
        assert 'bias' in state
        np.testing.assert_array_equal(state['weight'], module.weight.data)
        np.testing.assert_array_equal(state['bias'], module.bias.data)
    
    def test_module_load_state_dict(self):
        """Test load_state_dict method."""
        module = SimpleModule(3, 2)
        
        # Create new state
        new_state = {
            'weight': np.ones((3, 2)),
            'bias': np.zeros(2)
        }
        
        # Load state
        module.load_state_dict(new_state)
        
        np.testing.assert_array_equal(module.weight.data, new_state['weight'])
        np.testing.assert_array_equal(module.bias.data, new_state['bias'])
    
    def test_module_without_parameters(self):
        """Test module without parameters."""
        module = ModuleWithoutParams()
        
        # Should have empty parameter list
        params = list(module.parameters())
        assert len(params) == 0
        
        # Named parameters should be empty
        named_params = list(module.named_parameters())
        assert len(named_params) == 0
        
        # State dict should be empty
        state = module.state_dict()
        assert len(state) == 0


class TestParameterBase:
    """Test Parameter base class functionality."""
    
    def test_parameter_creation(self):
        """Test Parameter creation."""
        data = np.array([1, 2, 3])
        param = Parameter(data)
        
        np.testing.assert_array_equal(param.data, data)
        assert param.requires_grad is True
        assert param.grad is None
    
    def test_parameter_with_name(self):
        """Test Parameter with name."""
        data = np.array([1, 2, 3])
        param = Parameter(data, name="test_param")
        
        assert param.name == "test_param"
    
    def test_parameter_zero_grad(self):
        """Test Parameter zero_grad."""
        param = Parameter(np.array([1, 2, 3]))
        param.grad = np.array([0.1, 0.2, 0.3])
        
        param.zero_grad()
        assert param.grad is None
    
    def test_parameter_shape_property(self):
        """Test Parameter shape property."""
        param = Parameter(np.array([[1, 2], [3, 4]]))
        assert param.shape == (2, 2)
    
    def test_parameter_dtype_property(self):
        """Test Parameter dtype property."""
        param = Parameter(np.array([1, 2, 3], dtype=np.float32))
        # Parameter.dtype returns DType enum, not numpy dtype
        from neural_arch.core.dtype import DType
        assert param.dtype == DType.FLOAT32


class SimpleOptimizer(Optimizer):
    """Simple optimizer for testing base class."""
    def step(self):
        """Dummy step implementation."""
        pass
    
    def zero_grad(self):
        """Zero gradients."""
        for param in self.parameters.values():
            param.grad = None


class TestOptimizerBase:
    """Test Optimizer base class functionality."""
    
    def test_optimizer_creation(self):
        """Test Optimizer creation."""
        param1 = Parameter(np.array([1, 2, 3]))
        param2 = Parameter(np.array([4, 5, 6]))
        params = {'p1': param1, 'p2': param2}
        
        optimizer = SimpleOptimizer(params, lr=0.01)
        
        assert optimizer.parameters is params
        assert optimizer.defaults['lr'] == 0.01
    
    def test_optimizer_zero_grad(self):
        """Test Optimizer zero_grad."""
        param1 = Parameter(np.array([1, 2, 3]))
        param2 = Parameter(np.array([4, 5, 6]))
        param1.grad = np.ones(3)
        param2.grad = np.ones(3)
        
        params = {'p1': param1, 'p2': param2}
        optimizer = SimpleOptimizer(params)
        
        optimizer.zero_grad()
        
        assert param1.grad is None
        assert param2.grad is None
    
    def test_optimizer_step_not_implemented(self):
        """Test Optimizer base class step is abstract."""
        # Test that the base class cannot be instantiated
        params = {'p': Parameter(np.array([1, 2, 3]))}
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Optimizer(params)