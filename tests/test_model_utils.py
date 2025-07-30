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
            parameters={"layers": 12, "hidden_size": 768},
            performance={"accuracy": 0.95, "f1": 0.93},
            citation="Test et al. (2024)"
        )
        
        assert card.name == "Test Model"
        assert card.description == "A test model for unit testing"
        assert card.architecture == "transformer"
        assert card.parameters["layers"] == 12
        assert card.parameters["hidden_size"] == 768
        assert card.performance["accuracy"] == 0.95
        assert card.performance["f1"] == 0.93
        assert card.citation == "Test et al. (2024)"
    
    def test_model_card_optional_fields(self):
        """Test ModelCard with minimal required fields."""
        card = ModelCard(
            name="Minimal Model",
            description="A minimal test model"
        )
        
        assert card.name == "Minimal Model"
        assert card.description == "A minimal test model"
        assert card.architecture is None
        assert card.parameters == {}
        assert card.performance == {}
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
            parameters={"layers": 6}
        )
        
        card_dict = card.to_dict()
        assert isinstance(card_dict, dict)
        assert card_dict["name"] == "Test Model"
        assert card_dict["description"] == "A test model"
        assert card_dict["parameters"]["layers"] == 6


class TestWeightManagement:
    """Test weight downloading and loading utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_weights = {
            'layer1.weight': np.random.randn(10, 5).astype(np.float32),
            'layer1.bias': np.random.randn(10).astype(np.float32),
            'layer2.weight': np.random.randn(3, 10).astype(np.float32)
        }
    
    def test_save_weights(self):
        """Test saving model weights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "test_weights.npz")
            
            # Save weights
            save_weights(self.test_weights, weights_path)
            
            # Check file exists
            assert os.path.exists(weights_path)
            
            # Load and verify
            loaded = np.load(weights_path)
            for key in self.test_weights:
                assert key in loaded
                np.testing.assert_array_equal(loaded[key], self.test_weights[key])
    
    def test_load_pretrained_weights(self):
        """Test loading pretrained weights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "test_weights.npz")
            
            # Save test weights
            np.savez(weights_path, **self.test_weights)
            
            # Load weights
            loaded_weights = load_pretrained_weights(weights_path)
            
            assert isinstance(loaded_weights, dict)
            for key in self.test_weights:
                assert key in loaded_weights
                np.testing.assert_array_equal(loaded_weights[key], self.test_weights[key])
    
    def test_load_nonexistent_weights(self):
        """Test loading weights from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_pretrained_weights("nonexistent_weights.npz")
    
    @patch('urllib.request.urlretrieve')
    def test_download_weights_success(self, mock_urlretrieve):
        """Test successful weight downloading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "downloaded_weights.npz")
            
            # Mock successful download
            mock_urlretrieve.return_value = (weights_path, None)
            
            # Create fake weights file for the mock
            np.savez(weights_path, **self.test_weights)
            
            # Test download
            result_path = download_weights(
                "https://example.com/weights.npz",
                weights_path
            )
            
            assert result_path == weights_path
            assert os.path.exists(weights_path)
            mock_urlretrieve.assert_called_once()
    
    @patch('urllib.request.urlretrieve')
    def test_download_weights_failure(self, mock_urlretrieve):
        """Test weight downloading failure."""
        # Mock download failure
        mock_urlretrieve.side_effect = Exception("Download failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "failed_weights.npz")
            
            with pytest.raises(Exception, match="Download failed"):
                download_weights("https://example.com/weights.npz", weights_path)
    
    def test_download_weights_creates_directory(self):
        """Test that download_weights creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "weights.npz")
            
            with patch('urllib.request.urlretrieve') as mock_urlretrieve:
                # Mock successful download
                def mock_download(url, path):
                    # Create the file
                    np.savez(path, test=np.array([1, 2, 3]))
                    return path, None
                
                mock_urlretrieve.side_effect = mock_download
                
                result_path = download_weights("https://example.com/weights.npz", nested_path)
                
                assert os.path.exists(result_path)
                assert os.path.exists(os.path.dirname(nested_path))
    
    def test_weights_caching(self):
        """Test weight file caching behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "cached_weights.npz")
            
            # Create existing weights file
            np.savez(weights_path, **self.test_weights)
            original_mtime = os.path.getmtime(weights_path)
            
            with patch('urllib.request.urlretrieve') as mock_urlretrieve:
                # Download should skip if file exists
                result_path = download_weights(
                    "https://example.com/weights.npz",
                    weights_path,
                    force_download=False
                )
                
                # Should not download if file exists
                mock_urlretrieve.assert_not_called()
                assert result_path == weights_path
                assert os.path.getmtime(weights_path) == original_mtime


class TestModelUtilsIntegration:
    """Integration tests for model utilities."""
    
    def test_model_card_with_real_model_info(self):
        """Test ModelCard with realistic model information."""
        card = ModelCard(
            name="ResNet-50",
            description="Deep residual network with 50 layers for image classification",
            architecture="ResNet",
            parameters={
                "layers": 50,
                "parameters": "25.6M",
                "input_size": [224, 224, 3],
                "classes": 1000
            },
            performance={
                "top1_accuracy": 0.7616,
                "top5_accuracy": 0.9300,
                "flops": "4.1G"
            },
            citation="He et al. Deep Residual Learning for Image Recognition. CVPR 2016.",
            tags=["vision", "classification", "resnet"],
            license="Apache 2.0"
        )
        
        assert card.name == "ResNet-50"
        assert card.parameters["parameters"] == "25.6M"
        assert card.performance["top1_accuracy"] == 0.7616
        assert "vision" in card.tags
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
                
                # Save and load
                save_weights(save_weights_dict, weights_path)
                loaded_weights = load_pretrained_weights(weights_path)
                
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
                load_pretrained_weights(invalid_path)
        
        # Test saving to invalid path
        with pytest.raises(Exception):
            save_weights(self.test_weights, "/invalid/path/weights.npz")
    
    def test_weight_loading_with_missing_keys(self):
        """Test loading weights when some keys are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "partial_weights.npz")
            
            # Save only partial weights
            partial_weights = {
                'layer1.weight': self.test_weights['layer1.weight']
                # Missing layer1.bias and layer2.weight
            }
            
            save_weights(partial_weights, weights_path)
            loaded_weights = load_pretrained_weights(weights_path)
            
            # Should load what's available
            assert 'layer1.weight' in loaded_weights
            assert 'layer1.bias' not in loaded_weights
            assert 'layer2.weight' not in loaded_weights
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_weights = {
            'layer1.weight': np.random.randn(10, 5).astype(np.float32),
            'layer1.bias': np.random.randn(10).astype(np.float32),
            'layer2.weight': np.random.randn(3, 10).astype(np.float32)
        }


if __name__ == "__main__":
    pytest.main([__file__])