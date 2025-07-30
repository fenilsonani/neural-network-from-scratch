"""Test module initialization and imports."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import neural_arch
from neural_arch import __version__


class TestModuleInit:
    """Test neural_arch module initialization."""
    
    def test_version_exists(self):
        """Test that version is defined."""
        assert hasattr(neural_arch, '__version__')
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_all_imports(self):
        """Test that all expected modules are importable."""
        # Core modules
        from neural_arch import core
        from neural_arch.core import Tensor, Module, Parameter, Device, DeviceType, DType
        
        # NN modules
        from neural_arch import nn
        from neural_arch.nn import Linear, Embedding, LayerNorm, Dropout
        from neural_arch.nn import MultiHeadAttention, TransformerBlock
        
        # Functional modules
        from neural_arch import functional
        from neural_arch.functional import relu, softmax, cross_entropy_loss
        
        # Optimizer modules
        from neural_arch import optim
        from neural_arch.optim import Adam, SGD
        
        # Backend modules
        from neural_arch import backends
        from neural_arch.backends import get_backend, set_backend, available_backends
        
        # Ensure all imports worked
        assert core is not None
        assert nn is not None
        assert functional is not None
        assert optim is not None
        assert backends is not None
    
    def test_lazy_import_performance(self):
        """Test that imports don't take too long."""
        import time
        start = time.time()
        
        # Re-import to test import time
        import importlib
        importlib.reload(neural_arch)
        
        end = time.time()
        # Import should be fast (less than 1 second)
        assert end - start < 1.0
    
    def test_module_attributes(self):
        """Test module has expected attributes."""
        expected_attrs = [
            '__version__',
            '__author__',
            '__email__',
            '__license__',
            'core',
            'nn', 
            'functional',
            'optim',
            'backends'
        ]
        
        for attr in expected_attrs:
            assert hasattr(neural_arch, attr), f"Missing attribute: {attr}"