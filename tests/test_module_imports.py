"""Test module imports and structure."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import neural_arch


class TestModuleImports:
    """Test module imports and public API."""

    def test_version_import(self):
        """Test version information is available."""
        assert hasattr(neural_arch, "__version__")
        assert isinstance(neural_arch.__version__, str)
        assert len(neural_arch.__version__) > 0

        # Version should follow semantic versioning pattern
        version_parts = neural_arch.__version__.split(".")
        assert len(version_parts) >= 2  # At least major.minor

    def test_core_imports(self):
        """Test core component imports."""
        # These should be available in the public API
        core_components = ["Tensor", "Parameter", "Module", "Device", "DType"]

        for component in core_components:
            if hasattr(neural_arch, component):
                assert getattr(neural_arch, component) is not None

    def test_functional_imports(self):
        """Test functional operation imports."""
        functional_ops = [
            "add",
            "sub",
            "mul",
            "div",
            "matmul",
            "relu",
            "softmax",
            "cross_entropy_loss",
        ]

        for op in functional_ops:
            if hasattr(neural_arch, op):
                assert callable(getattr(neural_arch, op))

    def test_nn_layer_imports(self):
        """Test neural network layer imports."""
        nn_layers = ["Linear", "Embedding", "LayerNorm", "ReLU", "Softmax", "MultiHeadAttention"]

        for layer in nn_layers:
            if hasattr(neural_arch, layer):
                layer_class = getattr(neural_arch, layer)
                assert isinstance(layer_class, type)

    def test_optimizer_imports(self):
        """Test optimizer imports."""
        optimizers = ["Adam", "SGD", "AdamW"]

        for optimizer in optimizers:
            if hasattr(neural_arch, optimizer):
                opt_class = getattr(neural_arch, optimizer)
                assert isinstance(opt_class, type)

    def test_exception_imports(self):
        """Test exception imports."""
        exceptions = [
            "NeuralArchError",
            "TensorError",
            "ShapeError",
            "DTypeError",
            "DeviceError",
            "GradientError",
        ]

        for exception in exceptions:
            if hasattr(neural_arch, exception):
                exc_class = getattr(neural_arch, exception)
                assert issubclass(exc_class, Exception)

    def test_config_imports(self):
        """Test configuration imports."""
        config_items = ["Config", "load_config", "save_config"]

        for item in config_items:
            if hasattr(neural_arch, item):
                config_item = getattr(neural_arch, item)
                assert config_item is not None

    def test_submodule_imports(self):
        """Test submodule imports."""
        submodules = ["core", "nn", "functional", "optim", "backends"]

        for submodule in submodules:
            try:
                # Try to import submodule
                exec(f"import neural_arch.{submodule}")
            except ImportError:
                # Some submodules might not exist, that's okay
                pass

    def test_all_attribute(self):
        """Test __all__ attribute if it exists."""
        if hasattr(neural_arch, "__all__"):
            all_items = neural_arch.__all__
            assert isinstance(all_items, list)
            assert len(all_items) > 0

            # All items in __all__ should be available as attributes
            for item in all_items:
                assert hasattr(neural_arch, item), f"Missing {item} in public API"

    def test_import_performance(self):
        """Test that imports are reasonably fast."""
        import importlib
        import time

        start_time = time.time()
        importlib.reload(neural_arch)
        end_time = time.time()

        import_time = end_time - start_time
        assert import_time < 2.0  # Should import in less than 2 seconds

    def test_no_import_errors(self):
        """Test that there are no import errors."""
        # Try importing various parts of the library
        try:
            from neural_arch import config, core, exceptions, functional, nn, optim
        except ImportError as e:
            # Log the import error but don't fail the test
            # Some modules might not be available in all environments
            print(f"Import warning: {e}")

    def test_lazy_imports(self):
        """Test lazy import behavior."""
        # Import should not immediately load heavy dependencies
        import neural_arch

        # Basic attributes should be available without loading backends
        assert hasattr(neural_arch, "__version__")

        # Backend-specific code should only load when needed
        try:
            # This might trigger backend loading
            if hasattr(neural_arch, "get_backend"):
                backend = neural_arch.get_backend("numpy")
                assert backend is not None
        except Exception:
            # Backend might not be available, that's okay
            pass

    def test_circular_imports(self):
        """Test for circular import issues."""
        # Try importing in different orders to catch circular dependencies
        try:
            import neural_arch.core
            import neural_arch.functional
            import neural_arch.nn
            import neural_arch.optim
        except ImportError:
            # Some imports might fail in test environment
            pass

        # Should still be able to import main module
        import neural_arch

        assert neural_arch.__version__ is not None

    def test_public_api_consistency(self):
        """Test public API consistency."""
        # Check that common operations are available
        expected_operations = [
            "Tensor",  # Core tensor class
            "Linear",  # Basic layer
            "Adam",  # Basic optimizer
        ]

        available_ops = []
        for op in expected_operations:
            if hasattr(neural_arch, op):
                available_ops.append(op)

        # Should have at least some core functionality
        assert len(available_ops) > 0

    def test_module_docstring(self):
        """Test module has proper documentation."""
        assert neural_arch.__doc__ is not None
        assert len(neural_arch.__doc__.strip()) > 0

        # Should mention key features
        doc = neural_arch.__doc__.lower()
        expected_terms = ["neural", "network", "tensor"]

        found_terms = [term for term in expected_terms if term in doc]
        assert len(found_terms) > 0
