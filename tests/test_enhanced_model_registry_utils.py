"""Enhanced comprehensive tests for model registry and utilities.

This module provides in-depth testing of the model registry system, model cards,
weight loading/saving, pretrained model management, and all utility functions
with detailed validation of functionality and error handling.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from neural_arch.core import Module, Parameter, Tensor
from neural_arch.exceptions import ModelError
from neural_arch.models.registry import (
    ModelInfo,
    ModelRegistry,
    get_model,
    list_models,
    register_model,
)
from neural_arch.models.utils import (
    WEIGHTS_DIR,
    ModelCard,
    compute_file_hash,
    create_model_card_from_model,
    download_file,
    download_weights,
    get_weight_url,
    load_pretrained_weights,
    save_weights,
)


class MockModule(Module):
    """Mock module for testing registry functionality."""

    def __init__(self, num_classes: int = 10, hidden_size: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.linear = MockLinear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def parameters(self):
        return [self.linear.weight, self.linear.bias]

    def named_parameters(self):
        return {"linear.weight": self.linear.weight, "linear.bias": self.linear.bias}

    def state_dict(self):
        return {"linear.weight": self.linear.weight.data, "linear.bias": self.linear.bias.data}


class MockLinear:
    """Mock linear layer for testing."""

    def __init__(self, in_features: int, out_features: int):
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32))
        self.bias = Parameter(np.random.randn(out_features).astype(np.float32))
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor(x.data @ self.weight.data.T + self.bias.data)


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test ModelInfo creation with defaults."""
        info = ModelInfo(
            name="test_model", model_class=MockModule, description="Test model description"
        )

        assert info.name == "test_model"
        assert info.model_class == MockModule
        assert info.description == "Test model description"
        assert info.paper_url is None
        assert info.pretrained_configs == {}
        assert info.default_config is None
        assert info.tags == []

    def test_model_info_with_full_config(self):
        """Test ModelInfo with all fields."""
        pretrained_configs = {
            "default": {"num_classes": 1000, "hidden_size": 128},
            "small": {"num_classes": 100, "hidden_size": 64},
        }

        info = ModelInfo(
            name="full_model",
            model_class=MockModule,
            description="Full model with all options",
            paper_url="https://arxiv.org/abs/1234.5678",
            pretrained_configs=pretrained_configs,
            default_config="default",
            tags=["vision", "classification", "test"],
        )

        assert info.name == "full_model"
        assert info.paper_url == "https://arxiv.org/abs/1234.5678"
        assert info.pretrained_configs == pretrained_configs
        assert info.default_config == "default"
        assert info.tags == ["vision", "classification", "test"]


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        ModelRegistry.clear()

    def test_registry_registration(self):
        """Test basic model registration."""
        ModelRegistry.register(
            name="test_model", model_class=MockModule, description="Test model", tags=["test"]
        )

        assert "test_model" in ModelRegistry._models
        info = ModelRegistry._models["test_model"]
        assert info.name == "test_model"
        assert info.model_class == MockModule
        assert info.description == "Test model"
        assert info.tags == ["test"]

    def test_registry_duplicate_registration(self):
        """Test error on duplicate model registration."""
        ModelRegistry.register(
            name="duplicate_model", model_class=MockModule, description="First registration"
        )

        with pytest.raises(ValueError, match="already registered"):
            ModelRegistry.register(
                name="duplicate_model", model_class=MockModule, description="Second registration"
            )

    def test_registry_invalid_model_class(self):
        """Test error on invalid model class."""

        class NotAModule:
            pass

        with pytest.raises(TypeError, match="must inherit from Module"):
            ModelRegistry.register(
                name="invalid_model", model_class=NotAModule, description="Invalid model class"
            )

    def test_registry_with_pretrained_configs(self):
        """Test registration with pretrained configurations."""
        pretrained_configs = {
            "imagenet": {"num_classes": 1000, "hidden_size": 512},
            "cifar10": {"num_classes": 10, "hidden_size": 256},
        }

        ModelRegistry.register(
            name="pretrained_model",
            model_class=MockModule,
            description="Model with pretrained configs",
            pretrained_configs=pretrained_configs,
            default_config="imagenet",
        )

        info = ModelRegistry.get_model_info("pretrained_model")
        assert info.pretrained_configs == pretrained_configs
        assert info.default_config == "imagenet"

    def test_registry_invalid_default_config(self):
        """Test error on invalid default config."""
        with pytest.raises(ValueError, match="Default config.*not in pretrained_configs"):
            ModelRegistry.register(
                name="invalid_default",
                model_class=MockModule,
                description="Invalid default config",
                pretrained_configs={"config1": {}},
                default_config="nonexistent_config",
            )

    def test_registry_aliases(self):
        """Test model aliases."""
        ModelRegistry.register(
            name="original_name",
            model_class=MockModule,
            description="Original model",
            aliases=["alias1", "alias2"],
        )

        # Test accessing by original name
        info_original = ModelRegistry.get_model_info("original_name")
        assert info_original.name == "original_name"

        # Test accessing by aliases
        info_alias1 = ModelRegistry.get_model_info("alias1")
        info_alias2 = ModelRegistry.get_model_info("alias2")

        assert info_alias1.name == "original_name"
        assert info_alias2.name == "original_name"

        # Test duplicate alias error
        with pytest.raises(ValueError, match="Alias.*already registered"):
            ModelRegistry.register(
                name="another_model",
                model_class=MockModule,
                description="Another model",
                aliases=["alias1"],  # Duplicate alias
            )

    def test_registry_get_nonexistent_model(self):
        """Test error when getting nonexistent model."""
        with pytest.raises(ModelError, match="Model.*not found"):
            ModelRegistry.get_model_info("nonexistent_model")

    def test_registry_create_model_basic(self):
        """Test basic model creation."""
        ModelRegistry.register(
            name="basic_model", model_class=MockModule, description="Basic model"
        )

        model = ModelRegistry.create_model("basic_model", num_classes=5)

        assert isinstance(model, MockModule)
        assert model.num_classes == 5  # Custom parameter

    def test_registry_create_model_with_pretrained(self):
        """Test model creation with pretrained config."""
        pretrained_configs = {"default": {"num_classes": 1000, "hidden_size": 256}}

        ModelRegistry.register(
            name="pretrained_model",
            model_class=MockModule,
            description="Model with pretrained",
            pretrained_configs=pretrained_configs,
            default_config="default",
        )

        with patch("neural_arch.models.utils.load_pretrained_weights") as mock_load:
            model = ModelRegistry.create_model("pretrained_model", pretrained=True)

            assert isinstance(model, MockModule)
            assert model.num_classes == 1000  # From pretrained config
            assert model.hidden_size == 256  # From pretrained config
            mock_load.assert_called_once()

    def test_registry_create_model_pretrained_errors(self):
        """Test errors in pretrained model creation."""
        # Model without pretrained configs
        ModelRegistry.register(
            name="no_pretrained", model_class=MockModule, description="No pretrained configs"
        )

        with pytest.raises(ModelError, match="No pretrained configs available"):
            ModelRegistry.create_model("no_pretrained", pretrained=True)

        # Model with configs but no default
        ModelRegistry.register(
            name="no_default",
            model_class=MockModule,
            description="No default config",
            pretrained_configs={"config1": {}},
        )

        with pytest.raises(ModelError, match="No config specified"):
            ModelRegistry.create_model("no_default", pretrained=True)

        # Invalid config name
        ModelRegistry.register(
            name="with_configs",
            model_class=MockModule,
            description="Has configs",
            pretrained_configs={"valid_config": {}},
        )

        with pytest.raises(ModelError, match="Config.*not found"):
            ModelRegistry.create_model("with_configs", pretrained=True, config_name="invalid")

    def test_registry_create_model_error_handling(self):
        """Test error handling in model creation."""

        class FailingModule(Module):
            def __init__(self, **kwargs):
                if "fail" in kwargs:
                    raise ValueError("Intentional failure")
                super().__init__()

        ModelRegistry.register(
            name="failing_model",
            model_class=FailingModule,
            description="Model that fails to create",
        )

        with pytest.raises(ModelError, match="Failed to create model"):
            ModelRegistry.create_model("failing_model", fail=True)

    def test_registry_list_models(self):
        """Test listing models with filters."""
        # Register multiple models
        ModelRegistry.register("model1", MockModule, "Model 1", tags=["vision", "classification"])
        ModelRegistry.register("model2", MockModule, "Model 2", tags=["nlp", "language"])
        ModelRegistry.register("model3", MockModule, "Model 3", tags=["vision", "detection"])
        ModelRegistry.register(
            "model4", MockModule, "Model 4", pretrained_configs={"default": {}}, tags=["audio"]
        )

        # Test listing all models
        all_models = ModelRegistry.list_models()
        assert set(all_models) == {"model1", "model2", "model3", "model4"}

        # Test filtering by tags
        vision_models = ModelRegistry.list_models(tags=["vision"])
        assert set(vision_models) == {"model1", "model3"}

        nlp_models = ModelRegistry.list_models(tags=["nlp"])
        assert nlp_models == ["model2"]

        # Test filtering by pretrained availability
        pretrained_models = ModelRegistry.list_models(with_pretrained=True)
        assert pretrained_models == ["model4"]

        non_pretrained_models = ModelRegistry.list_models(with_pretrained=False)
        assert set(non_pretrained_models) == {"model1", "model2", "model3"}

    def test_registry_get_model_card(self):
        """Test getting model card."""
        pretrained_configs = {"imagenet": {"num_classes": 1000}, "cifar10": {"num_classes": 10}}

        ModelRegistry.register(
            name="card_model",
            model_class=MockModule,
            description="Model for card testing",
            paper_url="https://arxiv.org/abs/1234.5678",
            pretrained_configs=pretrained_configs,
            default_config="imagenet",
            tags=["vision", "classification"],
            aliases=["card_alias"],
        )

        card = ModelRegistry.get_model_card("card_model")

        assert card["name"] == "card_model"
        assert card["description"] == "Model for card testing"
        assert card["paper_url"] == "https://arxiv.org/abs/1234.5678"
        assert card["tags"] == ["vision", "classification"]
        assert card["pretrained_configs"] == ["imagenet", "cifar10"]
        assert card["default_config"] == "imagenet"
        assert card["aliases"] == ["card_alias"]
        assert "configurations" in card
        assert card["configurations"]["imagenet"] == {"num_classes": 1000}

    def test_registry_save_and_clear(self):
        """Test saving registry and clearing."""
        # Register some models
        ModelRegistry.register("model1", MockModule, "Model 1", tags=["test"])
        ModelRegistry.register("model2", MockModule, "Model 2", aliases=["alias1"])

        # Test save registry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_path = Path(f.name)

        try:
            ModelRegistry.save_registry(save_path)

            # Check saved content
            with open(save_path, "r") as f:
                data = json.load(f)

            assert "models" in data
            assert "aliases" in data
            assert "model1" in data["models"]
            assert "model2" in data["models"]
            assert data["aliases"]["alias1"] == "model2"

        finally:
            save_path.unlink()

        # Test clear
        assert len(ModelRegistry._models) > 0
        assert len(ModelRegistry._aliases) > 0

        ModelRegistry.clear()

        assert len(ModelRegistry._models) == 0
        assert len(ModelRegistry._aliases) == 0


class TestRegisterModelDecorator:
    """Test register_model decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        ModelRegistry.clear()

    def test_register_model_decorator(self):
        """Test register_model decorator functionality."""

        @register_model(
            name="decorated_model",
            description="Model registered with decorator",
            paper_url="https://example.com/paper",
            pretrained_configs={"default": {"num_classes": 100}},
            default_config="default",
            tags=["test", "decorator"],
            aliases=["decorated_alias"],
        )
        class DecoratedModel(MockModule):
            pass

        # Test that model was registered
        info = ModelRegistry.get_model_info("decorated_model")
        assert info.name == "decorated_model"
        assert info.model_class == DecoratedModel
        assert info.description == "Model registered with decorator"
        assert info.paper_url == "https://example.com/paper"
        assert info.tags == ["test", "decorator"]

        # Test alias works
        info_alias = ModelRegistry.get_model_info("decorated_alias")
        assert info_alias.name == "decorated_model"

        # Test that decorator returns the class unchanged
        assert DecoratedModel == DecoratedModel  # Class should be returned as-is


class TestGlobalRegistryFunctions:
    """Test global registry functions."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        ModelRegistry.clear()

    def test_get_model_function(self):
        """Test global get_model function."""
        ModelRegistry.register(
            name="global_test", model_class=MockModule, description="Test global function"
        )

        model = get_model("global_test", num_classes=42)

        assert isinstance(model, MockModule)
        assert model.num_classes == 42

    def test_list_models_function(self):
        """Test global list_models function."""
        ModelRegistry.register("model1", MockModule, "Model 1", tags=["vision"])
        ModelRegistry.register("model2", MockModule, "Model 2", tags=["nlp"])

        # Test global function
        all_models = list_models()
        assert set(all_models) == {"model1", "model2"}

        vision_models = list_models(tags=["vision"])
        assert vision_models == ["model1"]


class TestModelCard:
    """Test ModelCard functionality."""

    def test_model_card_creation(self):
        """Test ModelCard creation with various fields."""
        card = ModelCard(
            name="test_model",
            description="A test model for unit testing",
            architecture="Simple neural network",
            paper_title="Test Paper",
            paper_url="https://arxiv.org/abs/1234.5678",
            training_data="Synthetic test data",
            metrics={"accuracy": 95.2, "f1": 0.94},
            license="Apache-2.0",
            citation="@article{test2024, title={Test}, author={Author}}",
        )

        assert card.name == "test_model"
        assert card.description == "A test model for unit testing"
        assert card.architecture == "Simple neural network"
        assert card.paper_title == "Test Paper"
        assert card.paper_url == "https://arxiv.org/abs/1234.5678"
        assert card.metrics["accuracy"] == 95.2
        assert card.license == "Apache-2.0"

    def test_model_card_to_dict(self):
        """Test converting ModelCard to dictionary."""
        card = ModelCard(
            name="dict_test",
            description="Test dict conversion",
            architecture="Test arch",
            metrics={"acc": 0.9},
        )

        card_dict = card.to_dict()

        assert isinstance(card_dict, dict)
        assert card_dict["name"] == "dict_test"
        assert card_dict["description"] == "Test dict conversion"
        assert card_dict["architecture"] == "Test arch"
        assert card_dict["metrics"] == {"acc": 0.9}
        assert card_dict["license"] == "MIT"  # Default value

    def test_model_card_save_load(self):
        """Test saving and loading ModelCard."""
        original_card = ModelCard(
            name="save_load_test",
            description="Test save/load functionality",
            architecture="Test architecture",
            metrics={"loss": 0.1, "accuracy": 99.0},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_path = Path(f.name)

        try:
            # Test save
            original_card.save(save_path)
            assert save_path.exists()

            # Test load
            loaded_card = ModelCard.load(save_path)

            assert loaded_card.name == original_card.name
            assert loaded_card.description == original_card.description
            assert loaded_card.architecture == original_card.architecture
            assert loaded_card.metrics == original_card.metrics

        finally:
            save_path.unlink()

    def test_model_card_markdown_generation(self):
        """Test ModelCard markdown string generation."""
        card = ModelCard(
            name="Markdown Test Model",
            description="A model for testing markdown generation",
            architecture="Test architecture with 1M parameters",
            paper_title="Test Paper Title",
            paper_url="https://example.com/paper",
            training_data="ImageNet",
            preprocessing={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            metrics={"top1_accuracy": 76.1, "top5_accuracy": 92.9},
            limitations="Works best on natural images",
            license="MIT",
            citation="@article{test2024}",
        )

        markdown = str(card)

        assert "# Markdown Test Model" in markdown
        assert "A model for testing markdown generation" in markdown
        assert "## Architecture" in markdown
        assert "Test architecture with 1M parameters" in markdown
        assert "## Paper" in markdown
        assert "[Test Paper Title](https://example.com/paper)" in markdown
        assert "## Training Data" in markdown
        assert "ImageNet" in markdown
        assert "## Preprocessing" in markdown
        assert "**mean**" in markdown
        assert "## Performance Metrics" in markdown
        assert "**top1_accuracy**: 76.1" in markdown
        assert "## Limitations" in markdown
        assert "Works best on natural images" in markdown
        assert "## License" in markdown
        assert "MIT" in markdown
        assert "## Citation" in markdown
        assert "@article{test2024}" in markdown


class TestModelUtils:
    """Test model utility functions."""

    def test_get_weight_url(self):
        """Test weight URL generation."""
        url = get_weight_url("resnet50", "imagenet")

        expected_url = "https://github.com/neural-arch/model-weights/releases/download/v1.0/resnet50_imagenet.npz"
        assert url == expected_url

    def test_compute_file_hash(self):
        """Test file hash computation."""
        # Create temporary file with known content
        test_content = b"Hello, world!"

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_content)
            temp_path = Path(f.name)

        try:
            # Compute hash
            hash_value = compute_file_hash(temp_path)

            # Verify hash (known SHA256 of "Hello, world!")
            import hashlib

            expected_hash = hashlib.sha256(test_content).hexdigest()
            assert hash_value == expected_hash

            # Test different algorithm
            md5_hash = compute_file_hash(temp_path, algorithm="md5")
            expected_md5 = hashlib.md5(test_content).hexdigest()
            assert md5_hash == expected_md5

        finally:
            temp_path.unlink()

    @patch("neural_arch.models.utils.urlopen")
    def test_download_file(self, mock_urlopen):
        """Test file download functionality."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.read.side_effect = [b"chunk1", b"chunk2", b""]  # End with empty chunk
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            temp_path.unlink()  # Remove file so download can create it

            download_file("https://example.com/file.txt", temp_path)

            assert temp_path.exists()
            content = temp_path.read_bytes()
            assert content == b"chunk1chunk2"

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_download_weights_caching(self):
        """Test weight download with caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create cached file
            cached_file = cache_dir / "test_model_config1.npz"
            np.savez(cached_file, weight1=np.array([1, 2, 3]))

            with patch("neural_arch.models.utils.download_file") as mock_download:
                # Test cached file is used
                result_path = download_weights("test_model", "config1", cache_dir)

                assert result_path == cached_file
                mock_download.assert_not_called()  # Should not download

                # Test force download
                result_path = download_weights(
                    "test_model", "config1", cache_dir, force_download=True
                )

                mock_download.assert_called_once()  # Should download despite cache

    def test_save_and_load_weights(self):
        """Test saving and loading model weights."""
        # Create mock model
        model = MockModule(num_classes=10, hidden_size=64)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            weight_path = Path(f.name)

        try:
            # Test save weights
            metadata = {"epoch": 42, "accuracy": 95.5}
            save_weights(model, weight_path, metadata=metadata)

            assert weight_path.exists()

            # Load and verify
            checkpoint = np.load(weight_path, allow_pickle=True)

            assert "state_dict" in checkpoint
            assert "model_name" in checkpoint
            assert "metadata" in checkpoint

            state_dict = checkpoint["state_dict"].item()
            assert "linear.weight" in state_dict
            assert "linear.bias" in state_dict

            metadata_loaded = checkpoint["metadata"].item()
            assert metadata_loaded["epoch"] == 42
            assert metadata_loaded["accuracy"] == 95.5

        finally:
            weight_path.unlink()

    def test_load_pretrained_weights(self):
        """Test loading pretrained weights into model."""
        # Create model and save weights
        original_model = MockModule(num_classes=5, hidden_size=32)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            weight_path = Path(f.name)

        try:
            # Save original weights
            save_weights(original_model, weight_path)

            # Create new model with different initialization
            new_model = MockModule(num_classes=5, hidden_size=32)

            # Ensure weights are different initially
            assert not np.array_equal(
                new_model.linear.weight.data, original_model.linear.weight.data
            )

            # Mock download_weights to return our saved file
            with patch("neural_arch.models.utils.download_weights", return_value=weight_path):
                load_pretrained_weights(new_model, "test_model", "test_config")

            # Check weights are now the same
            np.testing.assert_array_equal(
                new_model.linear.weight.data, original_model.linear.weight.data
            )
            np.testing.assert_array_equal(
                new_model.linear.bias.data, original_model.linear.bias.data
            )

        finally:
            weight_path.unlink()

    def test_load_pretrained_weights_shape_mismatch(self):
        """Test error handling for shape mismatches in weight loading."""
        # Create model with different shape
        model = MockModule(num_classes=10, hidden_size=64)

        # Create weights file with different shape
        wrong_weights = {
            "linear.weight": np.random.randn(5, 32).astype(np.float32),  # Wrong shape
            "linear.bias": np.random.randn(5).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            weight_path = Path(f.name)

        try:
            np.savez(weight_path, state_dict=wrong_weights)

            with patch("neural_arch.models.utils.download_weights", return_value=weight_path):
                with pytest.raises(ModelError, match="Shape mismatch"):
                    load_pretrained_weights(model, "test_model", "test_config")

        finally:
            weight_path.unlink()

    def test_load_pretrained_weights_missing_keys(self):
        """Test handling of missing and unexpected keys in weight loading."""
        model = MockModule(num_classes=5, hidden_size=32)

        # Create weights with missing and unexpected keys
        partial_weights = {
            "linear.weight": model.linear.weight.data,  # Present
            # 'linear.bias' missing
            "unexpected_key": np.array([1, 2, 3]),  # Unexpected
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            weight_path = Path(f.name)

        try:
            np.savez(weight_path, state_dict=partial_weights)

            with patch("neural_arch.models.utils.download_weights", return_value=weight_path):
                # Test strict=False (should warn but not error)
                load_pretrained_weights(model, "test_model", "test_config", strict=False)

                # Test strict=True (should error)
                with pytest.raises(ModelError, match="Strict loading failed"):
                    load_pretrained_weights(model, "test_model", "test_config", strict=True)

        finally:
            weight_path.unlink()

    def test_create_model_card_from_model(self):
        """Test creating model card from model instance."""
        model = MockModule(num_classes=10, hidden_size=64)

        card = create_model_card_from_model(
            model=model,
            name="Test Model",
            description="A test model",
            paper_title="Test Paper",
            paper_url="https://example.com",
        )

        assert card.name == "Test Model"
        assert card.description == "A test model"
        assert "MockModule" in card.architecture
        assert "parameters" in card.architecture
        assert card.paper_title == "Test Paper"
        assert card.paper_url == "https://example.com"


class TestModelRegistryIntegration:
    """Test integration between registry and utils."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        ModelRegistry.clear()

    def test_end_to_end_model_workflow(self):
        """Test complete workflow from registration to usage."""

        # Register model with pretrained config
        @register_model(
            name="workflow_test",
            description="End-to-end workflow test",
            pretrained_configs={"default": {"num_classes": 100, "hidden_size": 128}},
            default_config="default",
            tags=["test", "integration"],
        )
        class WorkflowTestModel(MockModule):
            pass

        # Test model listing
        models = list_models(tags=["test"])
        assert "workflow_test" in models

        # Test model card generation
        card = ModelRegistry.get_model_card("workflow_test")
        assert card["name"] == "workflow_test"
        assert "default" in card["pretrained_configs"]

        # Test model creation without pretrained
        model1 = get_model("workflow_test", num_classes=50)
        assert isinstance(model1, WorkflowTestModel)
        assert model1.num_classes == 50

        # Test model creation with pretrained (mock weight loading)
        with patch("neural_arch.models.utils.load_pretrained_weights") as mock_load:
            model2 = get_model("workflow_test", pretrained=True)
            assert isinstance(model2, WorkflowTestModel)
            assert model2.num_classes == 100  # From pretrained config
            mock_load.assert_called_once_with(model2, "workflow_test", "default")

    def test_registry_persistence(self):
        """Test registry persistence through save/load."""
        # Register multiple models
        ModelRegistry.register("persist1", MockModule, "Persistent model 1", tags=["test"])
        ModelRegistry.register("persist2", MockModule, "Persistent model 2", aliases=["p2"])

        # Save registry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_path = Path(f.name)

        try:
            ModelRegistry.save_registry(save_path)

            # Clear registry
            ModelRegistry.clear()
            assert len(ModelRegistry._models) == 0

            # Verify data was saved correctly
            with open(save_path, "r") as f:
                data = json.load(f)

            assert "persist1" in data["models"]
            assert "persist2" in data["models"]
            assert data["aliases"]["p2"] == "persist2"

            # Note: In a full implementation, there would be a load_registry method

        finally:
            save_path.unlink()


class TestErrorHandling:
    """Test error handling across registry and utils."""

    def test_model_error_inheritance(self):
        """Test that ModelError is properly used."""
        # Test that ModelError can be raised and caught
        with pytest.raises(ModelError):
            raise ModelError("Test error message")

        # Test that ModelError inherits from Exception
        assert issubclass(ModelError, Exception)

    def test_network_error_handling(self):
        """Test handling of network errors in downloads."""
        with patch("neural_arch.models.utils.urlopen", side_effect=Exception("Network error")):
            with pytest.raises(ModelError, match="Download error"):
                download_file("https://example.com/file.txt", Path("test.txt"))

    def test_file_system_error_handling(self):
        """Test handling of file system errors."""
        # Test with invalid path
        invalid_path = Path("/nonexistent/directory/file.txt")

        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            compute_file_hash(invalid_path)


if __name__ == "__main__":
    pytest.main([__file__])
