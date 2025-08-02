"""Comprehensive tests for backends/utils to improve coverage from 77.53% to 95%+.

This file tests auto_select_backend and device-related utilities.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import platform
from unittest.mock import MagicMock, patch

import pytest

from neural_arch.backends.backend import current_backend, get_backend, set_backend
from neural_arch.backends.utils import (
    auto_select_backend,
    get_backend_for_device,
    get_device_for_backend,
    print_available_devices,
)


class TestBackendsUtilsComprehensive:
    """Comprehensive tests for backend utilities."""

    def test_auto_select_backend_prefer_cpu(self):
        """Test auto_select_backend with prefer_gpu=False."""
        # Should always select numpy when prefer_gpu=False
        backend = auto_select_backend(prefer_gpu=False)
        assert backend.name == "numpy"

        # Current backend should be set to numpy
        assert get_backend().name == "numpy"

    def test_auto_select_backend_on_apple_silicon(self):
        """Test auto_select_backend on Apple Silicon Mac."""
        # Mock platform to simulate Apple Silicon
        with patch("sys.platform", "darwin"):
            with patch("platform.machine", return_value="arm64"):
                # Test when MPS is available
                with patch(
                    "neural_arch.backends.utils.available_backends", return_value=["numpy", "mps"]
                ):
                    backend = auto_select_backend(prefer_gpu=True)
                    assert backend.name == "mps"

                # Test when MPS is not available
                with patch("neural_arch.backends.utils.available_backends", return_value=["numpy"]):
                    backend = auto_select_backend(prefer_gpu=True)
                    assert backend.name == "numpy"

    def test_auto_select_backend_on_intel_mac(self):
        """Test auto_select_backend on Intel Mac."""
        # Mock platform to simulate Intel Mac
        with patch("sys.platform", "darwin"):
            with patch("platform.machine", return_value="x86_64"):
                # Should check for CUDA even on Mac
                with patch(
                    "neural_arch.backends.utils.available_backends", return_value=["numpy", "cuda"]
                ):
                    backend = auto_select_backend(prefer_gpu=True)
                    assert backend.name == "cuda"

                # Fallback to numpy
                with patch("neural_arch.backends.utils.available_backends", return_value=["numpy"]):
                    backend = auto_select_backend(prefer_gpu=True)
                    assert backend.name == "numpy"

    def test_auto_select_backend_on_linux_with_cuda(self):
        """Test auto_select_backend on Linux with CUDA."""
        # Mock platform to simulate Linux
        with patch("sys.platform", "linux"):
            with patch("platform.machine", return_value="x86_64"):
                # Test when CUDA is available
                with patch(
                    "neural_arch.backends.utils.available_backends", return_value=["numpy", "cuda"]
                ):
                    backend = auto_select_backend(prefer_gpu=True)
                    assert backend.name == "cuda"

    def test_auto_select_backend_on_windows(self):
        """Test auto_select_backend on Windows."""
        # Mock platform to simulate Windows
        with patch("sys.platform", "win32"):
            with patch("platform.machine", return_value="AMD64"):
                # Test when CUDA is available
                with patch(
                    "neural_arch.backends.utils.available_backends", return_value=["numpy", "cuda"]
                ):
                    backend = auto_select_backend(prefer_gpu=True)
                    assert backend.name == "cuda"

                # Test fallback
                with patch("neural_arch.backends.utils.available_backends", return_value=["numpy"]):
                    backend = auto_select_backend(prefer_gpu=True)
                    assert backend.name == "numpy"

    def test_get_device_for_backend_with_current(self):
        """Test get_device_for_backend with current backend."""
        # Set numpy backend
        set_backend("numpy")
        device = get_device_for_backend()  # No backend_name provided
        assert device == "cpu"

        # Mock different backends
        mock_backend = MagicMock()

        # Test with cuda backend
        mock_backend.name = "cuda"
        with patch("neural_arch.backends.utils.get_backend", return_value=mock_backend):
            device = get_device_for_backend()
            assert device == "cuda"

        # Test with mps backend
        mock_backend.name = "mps"
        with patch("neural_arch.backends.utils.get_backend", return_value=mock_backend):
            device = get_device_for_backend()
            assert device == "mps"

    def test_get_device_for_backend_with_name(self):
        """Test get_device_for_backend with specific backend names."""
        assert get_device_for_backend("numpy") == "cpu"
        assert get_device_for_backend("cuda") == "cuda"
        assert get_device_for_backend("mps") == "mps"

        # Unknown backend should return cpu
        assert get_device_for_backend("unknown") == "cpu"
        assert get_device_for_backend("jax") == "cpu"

    def test_get_backend_for_device(self):
        """Test get_backend_for_device with various device strings."""
        # Basic device names
        assert get_backend_for_device("cpu") == "numpy"
        assert get_backend_for_device("cuda") == "cuda"
        assert get_backend_for_device("mps") == "mps"

        # Device names with indices
        assert get_backend_for_device("cuda:0") == "cuda"
        assert get_backend_for_device("cuda:1") == "cuda"
        assert get_backend_for_device("cuda:7") == "cuda"

        # Case insensitive
        assert get_backend_for_device("CPU") == "numpy"
        assert get_backend_for_device("CUDA") == "cuda"
        assert get_backend_for_device("MPS") == "mps"

        # Mixed case with index
        assert get_backend_for_device("CUDA:0") == "cuda"
        assert get_backend_for_device("Cuda:2") == "cuda"

        # Unknown devices default to numpy
        assert get_backend_for_device("gpu") == "numpy"
        assert get_backend_for_device("tpu") == "numpy"
        assert get_backend_for_device("unknown") == "numpy"

    def test_print_available_devices_cuda_available(self, capsys):
        """Test print_available_devices when CUDA is available."""
        # Mock device capabilities
        mock_caps = {
            "cpu": {"available": True, "architecture": "x86_64"},
            "cuda": {
                "available": True,
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA GeForce RTX 3090",
                        "memory": 24e9,  # 24 GB
                        "compute_capability": "8.6",
                    },
                    {
                        "index": 1,
                        "name": "NVIDIA GeForce RTX 3080",
                        "memory": 10e9,  # 10 GB
                        "compute_capability": "8.6",
                    },
                ],
            },
            "mps": {"available": False},
        }

        with patch("neural_arch.core.device.get_device_capabilities", return_value=mock_caps):
            with patch(
                "neural_arch.backends.utils.available_backends", return_value=["numpy", "cuda"]
            ):
                print_available_devices()

        captured = capsys.readouterr()
        output = captured.out

        # Check output contains expected information
        assert "Available Compute Devices:" in output
        assert "CPU:" in output
        assert "Architecture: x86_64" in output
        assert "CUDA:" in output
        assert "Available: True" in output
        assert "Device 0: NVIDIA GeForce RTX 3090" in output
        assert "Memory: 24.0 GB" in output
        assert "Device 1: NVIDIA GeForce RTX 3080" in output
        assert "Memory: 10.0 GB" in output
        assert "Compute Capability: 8.6" in output
        assert "MPS: Not available" in output
        assert "Available Backends: numpy, cuda" in output

    def test_print_available_devices_mps_available(self, capsys):
        """Test print_available_devices when MPS is available."""
        # Mock device capabilities
        mock_caps = {
            "cpu": {"available": True, "architecture": "arm64"},
            "cuda": {"available": False, "devices": []},
            "mps": {"available": True, "unified_memory": True},
        }

        with patch("neural_arch.core.device.get_device_capabilities", return_value=mock_caps):
            with patch(
                "neural_arch.backends.utils.available_backends", return_value=["numpy", "mps"]
            ):
                print_available_devices()

        captured = capsys.readouterr()
        output = captured.out

        # Check output contains expected information
        assert "Available Compute Devices:" in output
        assert "CPU:" in output
        assert "Architecture: arm64" in output
        assert "CUDA: Not available" in output
        assert "MPS (Metal Performance Shaders):" in output
        assert "Available: True" in output
        assert "Unified Memory: True" in output
        assert "Available Backends: numpy, mps" in output

    def test_print_available_devices_cpu_only(self, capsys):
        """Test print_available_devices with only CPU available."""
        # Mock device capabilities
        mock_caps = {
            "cpu": {"available": True, "architecture": "x86_64"},
            "cuda": {"available": False, "devices": []},
            "mps": {"available": False},
        }

        with patch("neural_arch.core.device.get_device_capabilities", return_value=mock_caps):
            with patch("neural_arch.backends.utils.available_backends", return_value=["numpy"]):
                print_available_devices()

        captured = capsys.readouterr()
        output = captured.out

        # Check output contains expected information
        assert "Available Compute Devices:" in output
        assert "CPU:" in output
        assert "Available: True" in output
        assert "CUDA: Not available" in output
        assert "MPS: Not available" in output
        assert "Available Backends: numpy" in output
        assert "-" * 50 in output  # Check separator lines

    def test_backend_selection_edge_cases(self):
        """Test edge cases in backend selection."""
        # Empty available backends (should not happen but test robustness)
        with patch("neural_arch.backends.utils.available_backends", return_value=[]):
            # Should still try to set numpy
            backend = auto_select_backend(prefer_gpu=True)
            # Will return whatever get_backend() returns
            assert backend is not None

    def test_device_string_edge_cases(self):
        """Test edge cases in device string handling."""
        # Empty device string
        assert get_backend_for_device("") == "numpy"

        # Device with multiple colons
        assert get_backend_for_device("cuda:0:1") == "cuda"

        # Whitespace
        assert get_backend_for_device(" cuda ") == "cuda"
        assert get_backend_for_device("cuda ") == "cuda"
        assert get_backend_for_device(" cuda") == "cuda"

        # Mixed separators
        assert get_backend_for_device("cuda-0") == "numpy"  # Not recognized
        assert get_backend_for_device("cuda_0") == "numpy"  # Not recognized

    def test_platform_detection_combinations(self):
        """Test various platform detection combinations."""
        platforms = [
            ("darwin", "arm64", ["numpy", "mps"], "mps"),  # Apple Silicon
            ("darwin", "x86_64", ["numpy", "cuda"], "cuda"),  # Intel Mac with eGPU
            ("linux", "x86_64", ["numpy", "cuda"], "cuda"),  # Linux with NVIDIA
            ("linux", "aarch64", ["numpy"], "numpy"),  # ARM Linux
            ("win32", "AMD64", ["numpy", "cuda"], "cuda"),  # Windows with NVIDIA
            ("freebsd", "amd64", ["numpy"], "numpy"),  # FreeBSD
        ]

        for sys_platform, machine, available, expected in platforms:
            with patch("sys.platform", sys_platform):
                with patch("platform.machine", return_value=machine):
                    with patch(
                        "neural_arch.backends.utils.available_backends", return_value=available
                    ):
                        backend = auto_select_backend(prefer_gpu=True)
                        assert backend.name == expected, f"Failed for {sys_platform}/{machine}"
