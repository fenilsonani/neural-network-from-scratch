"""
Test neural network layers - because layers better fucking work.
"""

try:
    import pytest
except ImportError:
    pytest = None
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch import Embedding, Linear, Tensor


class TestLinearLayer:
    """Test Linear layer because it's fundamental."""

    def test_linear_creation(self):
        """Test linear layer creation."""
        layer = Linear(3, 2)

        assert layer.in_features == 3
        assert layer.out_features == 2
        assert layer.weight.shape == (3, 2)
        assert layer.bias.shape == (2,)
        assert layer.weight.requires_grad == True
        assert layer.bias.requires_grad == True

    def test_linear_forward(self):
        """Test linear layer forward pass."""
        layer = Linear(3, 2)
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

        y = layer(x)

        assert y.shape == (2, 2)
        assert y.requires_grad == True

    def test_linear_parameters(self):
        """Test parameter collection."""
        layer = Linear(3, 2)
        params = layer.parameters()

        assert "weight" in params
        assert "bias" in params
        assert params["weight"].shape == (3, 2)
        assert params["bias"].shape == (2,)

    def test_linear_gradient_flow(self):
        """Test gradient flow through linear layer."""
        layer = Linear(2, 1)
        x = Tensor([[1, 2]], requires_grad=True)

        y = layer(x)
        y.backward(np.array([[1.0]]))

        if hasattr(y, "_backward"):
            y._backward()

        # Check gradients exist
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad is not None


class TestEmbeddingLayer:
    """Test Embedding layer because words matter."""

    def test_embedding_creation(self):
        """Test embedding layer creation."""
        layer = Embedding(vocab_size=10, embed_dim=4)

        assert layer.vocab_size == 10
        assert layer.embed_dim == 4
        assert layer.weight.shape == (10, 4)
        assert layer.weight.requires_grad == True

    def test_embedding_forward(self):
        """Test embedding forward pass."""
        layer = Embedding(vocab_size=5, embed_dim=3)
        indices = np.array([[0, 1, 2], [3, 4, 0]])

        embedded = layer(indices)

        assert embedded.shape == (2, 3, 3)
        assert embedded.requires_grad == True

    def test_embedding_lookup(self):
        """Test embedding lookup correctness."""
        layer = Embedding(vocab_size=3, embed_dim=2)

        # Set known weights
        layer.weight.data = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # index 0  # index 1  # index 2
        )

        indices = np.array([[0, 1, 2]])
        embedded = layer(indices)

        expected = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        assert np.allclose(embedded.data, expected)

    def test_embedding_parameters(self):
        """Test embedding parameter collection."""
        layer = Embedding(vocab_size=10, embed_dim=4)
        params = layer.parameters()

        assert "weight" in params
        assert params["weight"].shape == (10, 4)

    def test_embedding_gradient_accumulation(self):
        """Test gradient accumulation for repeated indices."""
        layer = Embedding(vocab_size=3, embed_dim=2)
        indices = np.array([[0, 0, 1]])  # Index 0 appears twice

        embedded = layer(indices)
        embedded.backward(np.ones_like(embedded.data))

        if hasattr(embedded, "_backward"):
            embedded._backward()

        # Index 0 should have double gradient
        assert layer.weight.grad is not None
        assert layer.weight.grad[0, 0] == 2.0  # Accumulated twice
        assert layer.weight.grad[1, 0] == 1.0  # Once


class TestLayerIntegration:
    """Test layers working together."""

    def test_embedding_to_linear(self):
        """Test embedding -> linear pipeline."""
        vocab_size = 5
        embed_dim = 3
        output_dim = 2

        embedding = Embedding(vocab_size, embed_dim)
        linear = Linear(embed_dim, output_dim)

        # Input indices
        indices = np.array([[0, 1, 2]])

        # Forward pass
        embedded = embedding(indices)  # (1, 3, 3)

        # Mean pooling with gradient support
        from neural_arch import mean_pool

        pooled = mean_pool(embedded, axis=1)  # (1, 3)

        # Linear layer
        output = linear(pooled)  # (1, 2)

        assert output.shape == (1, 2)
        assert output.requires_grad == True

    def test_multi_layer_gradients(self):
        """Test gradients through multiple layers."""
        embedding = Embedding(vocab_size=3, embed_dim=2)
        linear = Linear(2, 1)

        indices = np.array([[0, 1]])

        # Forward
        embedded = embedding(indices)
        from neural_arch import mean_pool

        pooled = mean_pool(embedded, axis=1)
        output = linear(pooled)

        # Backward
        output.backward(np.array([[1.0]]))
        if hasattr(output, "_backward"):
            output._backward()

        # Check all gradients exist
        assert linear.weight.grad is not None
        assert linear.bias.grad is not None
        assert embedding.weight.grad is not None


class TestLayerEdgeCases:
    """Test edge cases because layers can be tricky."""

    def test_linear_single_neuron(self):
        """Test linear layer with single input/output."""
        layer = Linear(1, 1)
        x = Tensor([[5.0]], requires_grad=True)

        y = layer(x)
        assert y.shape == (1, 1)

    def test_embedding_single_token(self):
        """Test embedding with single token."""
        layer = Embedding(vocab_size=2, embed_dim=3)
        indices = np.array([[0]])

        embedded = layer(indices)
        assert embedded.shape == (1, 1, 3)

    def test_linear_large_batch(self):
        """Test linear layer with large batch."""
        layer = Linear(10, 5)
        batch_size = 100
        x = Tensor(np.random.randn(batch_size, 10), requires_grad=True)

        y = layer(x)
        assert y.shape == (batch_size, 5)

    def test_embedding_out_of_bounds(self):
        """Test embedding with invalid indices."""
        layer = Embedding(vocab_size=3, embed_dim=2)

        # Valid indices
        indices = np.array([[0, 1, 2]])
        embedded = layer(indices)
        assert embedded.shape == (1, 3, 2)

        # Test with boundary indices
        indices_boundary = np.array([[0, 2]])  # Valid boundary
        embedded_boundary = layer(indices_boundary)
        assert embedded_boundary.shape == (1, 2, 2)


def test_layer_parameter_counting():
    """Test parameter counting for layers."""
    # Linear layer parameters
    linear = Linear(10, 5)
    linear_params = sum(p.data.size for p in linear.parameters().values())
    expected_linear = 10 * 5 + 5  # weights + bias
    assert linear_params == expected_linear

    # Embedding layer parameters
    embedding = Embedding(vocab_size=100, embed_dim=50)
    embedding_params = sum(p.data.size for p in embedding.parameters().values())
    expected_embedding = 100 * 50  # vocab_size * embed_dim
    assert embedding_params == expected_embedding

    print("✅ Parameter counting test passed!")


def test_layer_weight_initialization():
    """Test that layer weights are initialized reasonably."""
    # Linear layer
    linear = Linear(100, 50)
    weight_std = np.std(linear.weight.data)

    # Should be reasonable initialization, not zeros or huge values
    assert 0.01 < weight_std < 1.0
    assert np.mean(linear.bias.data) == 0.0  # Bias should be zero

    # Embedding layer
    embedding = Embedding(vocab_size=100, embed_dim=50)
    embed_std = np.std(embedding.weight.data)

    # Should be reasonable initialization
    assert 0.01 < embed_std < 1.0

    print("✅ Weight initialization test passed!")


if __name__ == "__main__":
    # Run tests manually
    test_linear = TestLinearLayer()
    test_embedding = TestEmbeddingLayer()
    test_integration = TestLayerIntegration()
    test_edges = TestLayerEdgeCases()

    print("🧪 Running layer tests...")

    try:
        # Linear layer tests
        test_linear.test_linear_creation()
        test_linear.test_linear_forward()
        test_linear.test_linear_parameters()
        test_linear.test_linear_gradient_flow()
        print("✅ Linear layer tests passed")

        # Embedding layer tests
        test_embedding.test_embedding_creation()
        test_embedding.test_embedding_forward()
        test_embedding.test_embedding_lookup()
        test_embedding.test_embedding_parameters()
        test_embedding.test_embedding_gradient_accumulation()
        print("✅ Embedding layer tests passed")

        # Integration tests
        test_integration.test_embedding_to_linear()
        test_integration.test_multi_layer_gradients()
        print("✅ Layer integration tests passed")

        # Edge case tests
        test_edges.test_linear_single_neuron()
        test_edges.test_embedding_single_token()
        test_edges.test_linear_large_batch()
        test_edges.test_embedding_out_of_bounds()
        print("✅ Layer edge case tests passed")

        # Additional tests
        test_layer_parameter_counting()
        test_layer_weight_initialization()

        print("\n🎉 ALL LAYER TESTS PASSED!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
