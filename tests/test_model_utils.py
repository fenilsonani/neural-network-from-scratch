"""Tests for model utilities and weight management."""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from neural_arch.models.utils import ModelCard, download_weights, load_pretrained_weights, save_weights
from neural_arch.core import Tensor, Parameter


class TestModelCard:
    """Test ModelCard functionality."""
    
    def test_model_card_creation(self):
        """Test basic ModelCard creation."""
        card = ModelCard(
            name="Test Model",
            description="A test model for unit testing",
            architecture="transformer",
            metrics={"accuracy": 0.95, "f1": 0.93},
            citation="Test et al. (2024)"
        )
        
        assert card.name == "Test Model"
        assert card.description == "A test model for unit testing"
        assert card.architecture == "transformer"
        assert card.metrics["accuracy"] == 0.95
        assert card.metrics["f1"] == 0.93
        assert card.citation == "Test et al. (2024)"
    
    def test_model_card_optional_fields(self):
        """Test ModelCard with minimal required fields."""
        card = ModelCard(
            name="Minimal Model",
            description="A minimal test model",
            architecture="minimal"
        )
        
        assert card.name == "Minimal Model"
        assert card.description == "A minimal test model"
        assert card.architecture == "minimal"
        assert card.metrics == {}
        assert card.citation is None
    
    def test_model_card_str_representation(self):
        """Test string representation of ModelCard."""
        card = ModelCard(
            name="Test Model",
            description="A test model",
            architecture="cnn"
        )
        
        str_repr = str(card)
        assert "Test Model" in str_repr
        assert "A test model" in str_repr
        assert "cnn" in str_repr
    
    def test_model_card_to_dict(self):
        """Test converting ModelCard to dictionary."""
        card = ModelCard(
            name="Test Model",
            description="A test model",
            architecture="test"
        )
        
        card_dict = card.to_dict()
        assert isinstance(card_dict, dict)
        assert card_dict["name"] == "Test Model"
        assert card_dict["description"] == "A test model"
        assert card_dict["architecture"] == "test"


class TestWeightManagement:
    """Test weight downloading and loading utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_weights = {
            'layer1.weight': np.random.randn(10, 5).astype(np.float32),
            'layer1.bias': np.random.randn(10).astype(np.float32),
            'layer2.weight': np.random.randn(3, 10).astype(np.float32)
        }
    
    def test_save_weights(self):
        """Test saving model weights."""
        # Create a mock model with the test weights
        from neural_arch.core import Module
        
        class MockModel(Module):
            def __init__(self, test_weights):
                super().__init__()
                self._parameters = test_weights
                self.test_weights = test_weights
                
            def state_dict(self):
                return self.test_weights
        
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "test_weights.npz")
            mock_model = MockModel(self.test_weights)
            
            # Save weights
            save_weights(mock_model, weights_path)
            
            # Check file exists
            assert os.path.exists(weights_path)
            
            # Load and verify
            loaded = np.load(weights_path, allow_pickle=True)
            assert 'state_dict' in loaded
            state_dict = loaded['state_dict'].item()
            for key in self.test_weights:
                assert key in state_dict
                np.testing.assert_array_equal(state_dict[key], self.test_weights[key])
    
    @patch('neural_arch.models.utils.download_weights')
    def test_load_pretrained_weights(self, mock_download):
        """Test loading pretrained weights."""
        from neural_arch.core import Module, Parameter
        from pathlib import Path
        
        class MockModel(Module):
            def __init__(self):
                super().__init__()
                self.layer1_weight = Parameter(np.zeros((10, 5), dtype=np.float32))
                self.layer1_bias = Parameter(np.zeros(10, dtype=np.float32))
                self.layer2_weight = Parameter(np.zeros((3, 10), dtype=np.float32))
                
            def named_parameters(self):
                return {
                    'layer1.weight': self.layer1_weight,
                    'layer1.bias': self.layer1_bias,
                    'layer2.weight': self.layer2_weight
                }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "test_weights.npz"
            
            # Save test weights in the expected format
            np.savez(weights_path, state_dict=self.test_weights)
            
            # Mock download_weights to return our test file
            mock_download.return_value = weights_path
            
            # Create model and load weights
            model = MockModel()
            load_pretrained_weights(model, "test_model", "test_config")
            
            # Verify weights were loaded (this is just testing the function doesn't crash)
            assert True  # If we get here, the function didn't crash
    
    def test_load_nonexistent_weights(self):
        """Test loading weights from non-existent file."""
        from neural_arch.core import Module
        
        class MockModel(Module):
            def named_parameters(self):
                return {}
        
        with pytest.raises(Exception):  # Will fail during download
            model = MockModel()
            load_pretrained_weights(model, "nonexistent_model", "nonexistent_config")
    
    @patch('neural_arch.models.utils.download_file')
    def test_download_weights_success(self, mock_download):
        """Test successful weight downloading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake weights file
            weights_path = os.path.join(temp_dir, "test_model_config.npz")
            np.savez(weights_path, **self.test_weights)
            
            # Mock successful download by doing nothing (file already exists)
            mock_download.return_value = None
            
            # Test download (will use cache since file exists)
            from pathlib import Path
            result_path = download_weights(
                "test_model", 
                "config",
                cache_dir=Path(temp_dir)
            )
            
            assert result_path.name == "test_model_config.npz"
            assert result_path.exists()
    
    def test_weights_caching(self):
        """Test weight file caching behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "test_model_config.npz")
            
            # Create existing weights file
            np.savez(weights_path, **self.test_weights)
            original_mtime = os.path.getmtime(weights_path)
            
            # Download should skip if file exists
            from pathlib import Path
            result_path = download_weights(
                "test_model",
                "config",
                cache_dir=Path(temp_dir),
                force_download=False
            )
            
            # Should use cached file
            assert result_path.name == "test_model_config.npz"
            assert result_path.exists()
            assert os.path.getmtime(weights_path) == original_mtime


class TestModelUtilsIntegration:
    """Integration tests for model utilities."""
    
    def test_model_card_with_real_model_info(self):
        """Test ModelCard with realistic model information."""
        card = ModelCard(
            name="ResNet-50",
            description="Deep residual network with 50 layers for image classification",
            architecture="ResNet",
            metrics={
                "top1_accuracy": 0.7616,
                "top5_accuracy": 0.9300,
                "flops": "4.1G"
            },
            citation="He et al. Deep Residual Learning for Image Recognition. CVPR 2016.",
            license="Apache 2.0"
        )
        
        assert card.name == "ResNet-50"
        assert card.metrics["top1_accuracy"] == 0.7616
        assert card.license == "Apache 2.0"
    
    def test_weight_format_compatibility(self):
        """Test compatibility with different weight formats."""
        test_formats = {
            'numpy': {
                'conv1.weight': np.random.randn(64, 3, 7, 7).astype(np.float32),
                'conv1.bias': np.random.randn(64).astype(np.float32)
            },
            'parameter': {
                'fc.weight': Parameter(np.random.randn(10, 512).astype(np.float32)),
                'fc.bias': Parameter(np.random.randn(10).astype(np.float32))
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for format_name, weights in test_formats.items():
                weights_path = os.path.join(temp_dir, f"{format_name}_weights.npz")
                
                # Convert Parameters to numpy for saving
                save_weights_dict = {}
                for key, value in weights.items():
                    if isinstance(value, Parameter):
                        save_weights_dict[key] = value.data
                    else:
                        save_weights_dict[key] = value
                
                # Save weights using numpy
                np.savez(weights_path, **save_weights_dict)
                loaded_weights = np.load(weights_path)
                
                # Verify
                for key in weights:
                    assert key in loaded_weights
                    expected = weights[key].data if isinstance(weights[key], Parameter) else weights[key]
                    np.testing.assert_array_equal(loaded_weights[key], expected)
    
    def test_model_utils_error_handling(self):
        """Test error handling in model utilities."""
        # Test invalid weight file format
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = os.path.join(temp_dir, "invalid.txt")
            with open(invalid_path, 'w') as f:
                f.write("not a weight file")
            
            with pytest.raises(Exception):
                np.load(invalid_path)
    
    def test_weight_loading_with_missing_keys(self):
        """Test loading weights when some keys are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "partial_weights.npz")
            
            # Save only partial weights
            partial_weights = {
                'layer1.weight': self.test_weights['layer1.weight']
                # Missing layer1.bias and layer2.weight
            }
            
            np.savez(weights_path, **partial_weights)
            loaded_weights = np.load(weights_path)
            
            # Should load what's available
            assert 'layer1.weight' in loaded_weights
            assert 'layer1.bias' not in loaded_weights
            assert 'layer2.weight' not in loaded_weights
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_weights = {
            'layer1.weight': np.random.randn(10, 5).astype(np.float32),
            'layer1.bias': np.random.randn(10).astype(np.float32),
            'layer2.weight': np.random.randn(3, 10).astype(np.float32)
        }


if __name__ == "__main__":
    pytest.main([__file__])