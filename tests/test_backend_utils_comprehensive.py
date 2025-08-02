"""Comprehensive tests for backends/utils module to maximize coverage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import platform
from unittest.mock import MagicMock, patch

import pytest

from neural_arch.backends.utils import (
    auto_select_backend,
    get_backend_for_device,
    get_device_for_backend,
    print_available_devices,
)


class TestAutoSelectBackend:
    """Test auto_select_backend function comprehensively."""

    def test_prefer_cpu_fallback(self):
        """Test that prefer_gpu=False selects numpy backend."""
        with patch("neural_arch.backends.utils.available_backends") as mock_available:
            with patch("neural_arch.backends.utils.set_backend") as mock_set:
                with patch("neural_arch.backends.utils.get_backend") as mock_get:
                    mock_available.return_value = ["numpy", "cuda", "mps"]
                    mock_backend = MagicMock()
                    mock_get.return_value = mock_backend

                    result = auto_select_backend(prefer_gpu=False)

                    mock_set.assert_called_once_with("numpy")
                    mock_get.assert_called_once()
                    assert result == mock_backend

    @patch("sys.platform", "darwin")
    @patch("platform.machine", return_value="arm64")
    def test_apple_silicon_mps_preference(self, mock_machine):
        """Test MPS preference on Apple Silicon Macs."""
        with patch("neural_arch.backends.utils.available_backends") as mock_available:
            with patch("neural_arch.backends.utils.set_backend") as mock_set:
                with patch("neural_arch.backends.utils.get_backend") as mock_get:
                    mock_available.return_value = ["numpy", "cuda", "mps"]
                    mock_backend = MagicMock()
                    mock_get.return_value = mock_backend

                    result = auto_select_backend(prefer_gpu=True)

                    mock_set.assert_called_with("mps")
                    assert result == mock_backend

    @patch("sys.platform", "darwin")
    @patch("platform.machine", return_value="arm64")
    def test_apple_silicon_mps_fallback_on_error(self, mock_machine):
        """Test fallback when MPS backend fails to initialize."""
        with patch("neural_arch.backends.utils.available_backends") as mock_available:
            with patch("neural_arch.backends.utils.set_backend") as mock_set:
                with patch("neural_arch.backends.utils.get_backend") as mock_get:
                    mock_available.return_value = ["numpy", "cuda", "mps"]

                    # First call (MPS) raises error, second call (CUDA) raises error, third call (numpy) succeeds
                    mock_set.side_effect = [
                        ValueError("MPS not available"),
                        ValueError("CUDA not available"),
                        None,
                    ]
                    mock_backend = MagicMock()
                    mock_get.return_value = mock_backend

                    result = auto_select_backend(prefer_gpu=True)

                    # Should try MPS, then CUDA, then fall back to numpy
                    assert mock_set.call_count == 3
                    mock_set.assert_any_call("mps")
                    mock_set.assert_any_call("cuda")
                    mock_set.assert_any_call("numpy")
                    assert result == mock_backend

    @patch("sys.platform", "linux")
    def test_cuda_preference_on_linux(self):
        """Test CUDA preference on non-Apple systems."""
        with patch("neural_arch.backends.utils.available_backends") as mock_available:
            with patch("neural_arch.backends.utils.set_backend") as mock_set:
                with patch("neural_arch.backends.utils.get_backend") as mock_get:
                    mock_available.return_value = ["numpy", "cuda"]
                    mock_backend = MagicMock()
                    mock_get.return_value = mock_backend

                    result = auto_select_backend(prefer_gpu=True)

                    mock_set.assert_called_with("cuda")
                    assert result == mock_backend

    def test_cuda_fallback_on_error(self):
        """Test fallback to CPU when CUDA fails."""
        with patch("neural_arch.backends.utils.available_backends") as mock_available:
            with patch("neural_arch.backends.utils.set_backend") as mock_set:
                with patch("neural_arch.backends.utils.get_backend") as mock_get:
                    mock_available.return_value = ["numpy", "cuda"]

                    # First call (CUDA) raises error, second call (numpy) succeeds
                    mock_set.side_effect = [ImportError("CUDA not available"), None]
                    mock_backend = MagicMock()
                    mock_get.return_value = mock_backend

                    result = auto_select_backend(prefer_gpu=True)

                    assert mock_set.call_count == 2
                    mock_set.assert_any_call("cuda")
                    mock_set.assert_any_call("numpy")
                    assert result == mock_backend


class TestGetDeviceForBackend:
    """Test get_device_for_backend function comprehensively."""

    def test_get_device_for_numpy_backend(self):
        """Test device string for numpy backend."""
        result = get_device_for_backend("numpy")
        assert result == "cpu"

    def test_get_device_for_cuda_backend(self):
        """Test device string for CUDA backend."""
        result = get_device_for_backend("cuda")
        assert result == "cuda"

    def test_get_device_for_mps_backend(self):
        """Test device string for MPS backend."""
        result = get_device_for_backend("mps")
        assert result == "mps"

    def test_get_device_for_unknown_backend(self):
        """Test device string for unknown backend defaults to CPU."""
        result = get_device_for_backend("unknown_backend")
        assert result == "cpu"

    def test_get_device_for_current_backend(self):
        """Test getting device for current backend (None parameter)."""
        with patch("neural_arch.backends.utils.get_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.name = "cuda"
            mock_get.return_value = mock_backend

            result = get_device_for_backend(None)

            mock_get.assert_called_once()
            assert result == "cuda"


class TestGetBackendForDevice:
    """Test get_backend_for_device function comprehensively."""

    def test_get_backend_for_cpu_device(self):
        """Test backend for CPU device."""
        result = get_backend_for_device("cpu")
        assert result == "numpy"

    def test_get_backend_for_cuda_device(self):
        """Test backend for CUDA device."""
        result = get_backend_for_device("cuda")
        assert result == "cuda"

    def test_get_backend_for_cuda_device_with_index(self):
        """Test backend for CUDA device with index."""
        result = get_backend_for_device("cuda:0")
        assert result == "cuda"

        result = get_backend_for_device("CUDA:1")
        assert result == "cuda"

    def test_get_backend_for_mps_device(self):
        """Test backend for MPS device."""
        result = get_backend_for_device("mps")
        assert result == "mps"

        result = get_backend_for_device("MPS")
        assert result == "mps"

    def test_get_backend_for_unknown_device(self):
        """Test backend for unknown device defaults to numpy."""
        result = get_backend_for_device("unknown_device")
        assert result == "numpy"


class TestPrintAvailableDevices:
    """Test print_available_devices function comprehensively."""

    def test_print_available_devices_output(self, capsys):
        """Test that print_available_devices produces expected output."""
        # Mock the device capabilities
        mock_caps = {
            "cpu": {"architecture": "arm64", "available": True},
            "cuda": {"available": False, "devices": []},
            "mps": {"available": True, "unified_memory": True},
        }

        with patch("neural_arch.core.device.get_device_capabilities") as mock_get_caps:
            with patch("neural_arch.backends.utils.available_backends") as mock_available:
                mock_get_caps.return_value = mock_caps
                mock_available.return_value = ["numpy", "mps"]

                print_available_devices()

                captured = capsys.readouterr()
                output = captured.out

                # Check that key sections are present
                assert "Available Compute Devices:" in output
                assert "CPU:" in output
                assert "Architecture: arm64" in output
                assert "CUDA: Not available" in output
                assert "MPS (Metal Performance Shaders):" in output
                assert "Available: True" in output
                assert "Available Backends: numpy, mps" in output

    def test_print_available_devices_with_cuda(self, capsys):
        """Test output when CUDA devices are available."""
        mock_caps = {
            "cpu": {"architecture": "x86_64", "available": True},
            "cuda": {
                "available": True,
                "devices": [
                    {
                        "index": 0,
                        "name": "GeForce RTX 3080",
                        "memory": 10737418240,  # 10GB
                        "compute_capability": "8.6",
                    }
                ],
            },
            "mps": {"available": False, "unified_memory": False},
        }

        with patch("neural_arch.core.device.get_device_capabilities") as mock_get_caps:
            with patch("neural_arch.backends.utils.available_backends") as mock_available:
                mock_get_caps.return_value = mock_caps
                mock_available.return_value = ["numpy", "cuda"]

                print_available_devices()

                captured = capsys.readouterr()
                output = captured.out

                # Check CUDA device details
                assert "CUDA:" in output
                assert "Available: True" in output
                assert "Device 0: GeForce RTX 3080" in output
                assert "Memory: 10.7 GB" in output
                assert "Compute Capability: 8.6" in output
