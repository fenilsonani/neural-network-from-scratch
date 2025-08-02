"""Comprehensive tests for backends/__init__.py to improve coverage from 73.68% to 100%.

This file targets the backend module imports and exception handling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import importlib
import sys as sys_module
from unittest.mock import MagicMock, patch

import pytest


class TestBackendsInit:
    """Comprehensive tests for backends module initialization."""

    def test_basic_imports(self):
        """Test that basic imports work correctly."""
        from neural_arch import backends

        # Basic backend functionality should always be available
        assert hasattr(backends, "Backend")
        assert hasattr(backends, "get_backend")
        assert hasattr(backends, "set_backend")
        assert hasattr(backends, "available_backends")
        assert hasattr(backends, "current_backend")
        assert hasattr(backends, "register_backend")
        assert hasattr(backends, "NumpyBackend")

        # Utility functions
        assert hasattr(backends, "auto_select_backend")
        assert hasattr(backends, "get_device_for_backend")
        assert hasattr(backends, "get_backend_for_device")
        assert hasattr(backends, "print_available_devices")

    def test_optional_backend_imports_mps_missing(self):
        """Test MPS backend import failure handling."""
        # Save original modules
        original_modules = {}
        mps_related = ["neural_arch.backends.mps_backend", "metal", "pyobjc"]

        for mod in mps_related:
            if mod in sys_module.modules:
                original_modules[mod] = sys_module.modules[mod]
                del sys_module.modules[mod]

        # Mock import to raise ImportError
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "mps_backend" in name:
                raise ImportError("MPS backend not available")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                # Force reload of backends module
                if "neural_arch.backends" in sys_module.modules:
                    del sys_module.modules["neural_arch.backends"]

                import neural_arch.backends as backends_test

                # MPSBackend should not be in __all__
                assert "MPSBackend" not in backends_test.__all__
                assert not hasattr(backends_test, "MPSBackend")

        finally:
            # Restore original modules
            for mod, original in original_modules.items():
                sys_module.modules[mod] = original

    def test_optional_backend_imports_cuda_missing(self):
        """Test CUDA backend import failure handling."""
        # Save original modules
        original_modules = {}
        cuda_related = ["neural_arch.backends.cuda_backend", "torch", "cuda"]

        for mod in cuda_related:
            if mod in sys_module.modules:
                original_modules[mod] = sys_module.modules[mod]
                del sys_module.modules[mod]

        # Mock import to raise ImportError
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "cuda_backend" in name:
                raise ImportError("CUDA backend not available")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                # Force reload of backends module
                if "neural_arch.backends" in sys_module.modules:
                    del sys_module.modules["neural_arch.backends"]

                import neural_arch.backends as backends_test

                # CudaBackend should not be in __all__
                assert "CudaBackend" not in backends_test.__all__
                assert not hasattr(backends_test, "CudaBackend")

        finally:
            # Restore original modules
            for mod, original in original_modules.items():
                sys_module.modules[mod] = original

    def test_optional_backend_imports_jax_missing(self):
        """Test JAX backend import failure handling."""
        # Save original modules
        original_modules = {}
        jax_related = ["neural_arch.backends.jax_backend", "jax"]

        for mod in jax_related:
            if mod in sys_module.modules:
                original_modules[mod] = sys_module.modules[mod]
                del sys_module.modules[mod]

        # Mock import to raise ImportError
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "jax_backend" in name:
                raise ImportError("JAX backend not available")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                # Force reload of backends module
                if "neural_arch.backends" in sys_module.modules:
                    del sys_module.modules["neural_arch.backends"]

                import neural_arch.backends as backends_test

                # JAXBackend should not be in __all__
                assert "JAXBackend" not in backends_test.__all__
                assert not hasattr(backends_test, "JAXBackend")

        finally:
            # Restore original modules
            for mod, original in original_modules.items():
                sys_module.modules[mod] = original

    def test_all_optional_backends_missing(self):
        """Test when all optional backends fail to import."""
        # Save original modules
        original_modules = {}
        all_optional = [
            "neural_arch.backends.mps_backend",
            "neural_arch.backends.cuda_backend",
            "neural_arch.backends.jax_backend",
            "metal",
            "pyobjc",
            "torch",
            "cuda",
            "jax",
        ]

        for mod in all_optional:
            if mod in sys_module.modules:
                original_modules[mod] = sys_module.modules[mod]
                del sys_module.modules[mod]

        # Mock import to raise ImportError for all optional backends
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if any(backend in name for backend in ["mps_backend", "cuda_backend", "jax_backend"]):
                raise ImportError(f"{name} not available")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                # Force reload of backends module
                if "neural_arch.backends" in sys_module.modules:
                    del sys_module.modules["neural_arch.backends"]

                import neural_arch.backends as backends_test

                # Only basic backends should be available
                assert "Backend" in backends_test.__all__
                assert "NumpyBackend" in backends_test.__all__

                # No optional backends
                assert "MPSBackend" not in backends_test.__all__
                assert "CudaBackend" not in backends_test.__all__
                assert "JAXBackend" not in backends_test.__all__

                # Should still have all utility functions
                assert "auto_select_backend" in backends_test.__all__
                assert "get_device_for_backend" in backends_test.__all__

        finally:
            # Restore original modules
            for mod, original in original_modules.items():
                sys_module.modules[mod] = original

    def test_partial_backend_availability(self):
        """Test when only some optional backends are available."""
        # This test simulates having MPS available but not CUDA/JAX
        original_modules = {}
        cuda_jax_related = [
            "neural_arch.backends.cuda_backend",
            "neural_arch.backends.jax_backend",
            "torch",
            "cuda",
            "jax",
        ]

        for mod in cuda_jax_related:
            if mod in sys_module.modules:
                original_modules[mod] = sys_module.modules[mod]
                del sys_module.modules[mod]

        # Create a mock MPS backend module
        mock_mps_module = MagicMock()
        mock_mps_module.MPSBackend = MagicMock()

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "mps_backend" in name:
                # Return mock MPS module
                return mock_mps_module
            elif "cuda_backend" in name or "jax_backend" in name:
                raise ImportError(f"{name} not available")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                # Force reload of backends module
                if "neural_arch.backends" in sys_module.modules:
                    del sys_module.modules["neural_arch.backends"]

                import neural_arch.backends as backends_test

                # MPS should be available
                assert "MPSBackend" in backends_test.__all__

                # CUDA and JAX should not be available
                assert "CudaBackend" not in backends_test.__all__
                assert "JAXBackend" not in backends_test.__all__

        finally:
            # Restore original modules
            for mod, original in original_modules.items():
                sys_module.modules[mod] = original

    def test_all_exports_accessible(self):
        """Test that all items in __all__ are actually accessible."""
        from neural_arch import backends

        for item in backends.__all__:
            assert hasattr(backends, item), f"{item} in __all__ but not accessible"

            # Check it's not None
            attr = getattr(backends, item)
            assert attr is not None, f"{item} is None"

    def test_no_unexpected_exports(self):
        """Test that there are no unexpected public exports."""
        from neural_arch import backends

        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(backends) if not attr.startswith("_")]

        # Remove Python built-ins and module attributes
        module_attrs = {
            "__builtins__",
            "__cached__",
            "__doc__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__spec__",
        }
        public_attrs = [attr for attr in public_attrs if attr not in module_attrs]

        # All public attributes should be in __all__ (with some exceptions)
        # Exception: module imports like 'sys', 'importlib' might exist
        expected_extras = {"sys", "importlib", "np", "os"}  # Common module imports

        for attr in public_attrs:
            if attr not in expected_extras:
                assert attr in backends.__all__, f"{attr} is public but not in __all__"
