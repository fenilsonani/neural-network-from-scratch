"""Ultra-comprehensive tests for Embedding layer to achieve 95%+ test coverage.

This test suite covers the Embedding layer implementation to ensure robust 95%+ test coverage,
including initialization, forward pass, gradient computation, and edge cases.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core.base import Parameter
from neural_arch.core.tensor import Tensor
from neural_arch.nn.embedding import Embedding


class TestEmbedding95Coverage:
    """Comprehensive Embedding tests targeting 95%+ coverage."""

    def test_embedding_initialization_comprehensive(self):
        """Test all Embedding initialization parameters."""
        # Test basic initialization
        vocab_size, embed_dim = 1000, 256
        embedding = Embedding(vocab_size, embed_dim)

        assert embedding.vocab_size == vocab_size
        assert embedding.embed_dim == embed_dim
        assert embedding.weight.shape == (vocab_size, embed_dim)
        assert isinstance(embedding.weight, Parameter)
        assert embedding.weight.requires_grad == True

        # Test with custom name
        custom_name = "custom_embedding"
        embedding_named = Embedding(vocab_size, embed_dim, name=custom_name)
        assert embedding_named.name == custom_name
        assert custom_name in embedding_named.weight.name

    def test_embedding_initialization_default_name(self):
        """Test Embedding initialization with default name."""
        vocab_size, embed_dim = 500, 128
        embedding = Embedding(vocab_size, embed_dim)

        expected_name = f"Embedding({vocab_size}, {embed_dim})"
        assert embedding.name == expected_name
        assert expected_name in embedding.weight.name

    def test_embedding_weight_initialization_properties(self):
        """Test properties of weight initialization."""
        vocab_size, embed_dim = 100, 64
        embedding = Embedding(vocab_size, embed_dim)

        # Test initialization scale
        expected_scale = 1.0 / np.sqrt(embed_dim)
        weight_data = embedding.weight.data

        # Weights should be within the expected range
        assert np.all(weight_data >= -expected_scale)
        assert np.all(weight_data <= expected_scale)

        # Weights should not be all zeros or ones
        assert not np.allclose(weight_data, 0.0)
        assert not np.allclose(weight_data, 1.0)

        # Test data type
        assert weight_data.dtype == np.float32

    def test_embedding_forward_pass_comprehensive(self):
        """Test Embedding forward pass with various input configurations."""
        vocab_size, embed_dim = 50, 32
        embedding = Embedding(vocab_size, embed_dim)

        # Test 1D input (single sequence)
        indices_1d = Tensor(np.array([0, 5, 10, 15]).astype(np.int32))
        output_1d = embedding.forward(indices_1d)
        assert output_1d.shape == (4, embed_dim)

        # Test 2D input (batch of sequences)
        indices_2d = Tensor(np.array([[0, 5, 10], [15, 20, 25]]).astype(np.int32))
        output_2d = embedding.forward(indices_2d)
        assert output_2d.shape == (2, 3, embed_dim)

        # Test 3D input (batch of sequences with additional dimension)
        indices_3d = Tensor(np.array([[[0, 5], [10, 15]], [[20, 25], [30, 35]]]).astype(np.int32))
        output_3d = embedding.forward(indices_3d)
        assert output_3d.shape == (2, 2, 2, embed_dim)

        # Test single token
        indices_single = Tensor(np.array([7]).astype(np.int32))
        output_single = embedding.forward(indices_single)
        assert output_single.shape == (1, embed_dim)

    def test_embedding_forward_with_numpy_array(self):
        """Test Embedding forward pass with numpy array input."""
        vocab_size, embed_dim = 30, 16
        embedding = Embedding(vocab_size, embed_dim)

        # Test with numpy array directly
        indices_np = np.array([0, 5, 10, 15], dtype=np.int32)
        output_np = embedding.forward(indices_np)
        assert output_np.shape == (4, embed_dim)

        # Test with Tensor input for comparison
        indices_tensor = Tensor(indices_np)
        output_tensor = embedding.forward(indices_tensor)

        # Results should be identical
        assert np.allclose(output_np.data, output_tensor.data)

    def test_embedding_index_validation_and_handling(self):
        """Test index validation and edge cases."""
        vocab_size, embed_dim = 20, 8
        embedding = Embedding(vocab_size, embed_dim)

        # Test valid indices
        valid_indices = [0, 5, 10, 19]  # All within vocab_size
        indices_valid = Tensor(np.array(valid_indices).astype(np.int32))
        output_valid = embedding.forward(indices_valid)
        assert output_valid.shape == (4, embed_dim)

        # Test with maximum valid index
        max_index = Tensor(np.array([vocab_size - 1]).astype(np.int32))
        output_max = embedding.forward(max_index)
        assert output_max.shape == (1, embed_dim)

        # Test with minimum valid index
        min_index = Tensor(np.array([0]).astype(np.int32))
        output_min = embedding.forward(min_index)
        assert output_min.shape == (1, embed_dim)

    def test_embedding_lookup_correctness(self):
        """Test correctness of embedding lookup."""
        vocab_size, embed_dim = 10, 4
        embedding = Embedding(vocab_size, embed_dim)

        # Test that lookup returns correct embeddings
        indices = Tensor(np.array([0, 2, 5]).astype(np.int32))
        output = embedding.forward(indices)

        # Manual verification
        expected_0 = embedding.weight.data[0]
        expected_2 = embedding.weight.data[2]
        expected_5 = embedding.weight.data[5]

        assert np.allclose(output.data[0], expected_0)
        assert np.allclose(output.data[1], expected_2)
        assert np.allclose(output.data[2], expected_5)

    def test_embedding_batch_lookup_correctness(self):
        """Test correctness of batch embedding lookup."""
        vocab_size, embed_dim = 15, 6
        embedding = Embedding(vocab_size, embed_dim)

        # Test batch lookup
        indices = Tensor(np.array([[0, 3], [7, 12]]).astype(np.int32))
        output = embedding.forward(indices)

        # Verify each element
        assert np.allclose(output.data[0, 0], embedding.weight.data[0])
        assert np.allclose(output.data[0, 1], embedding.weight.data[3])
        assert np.allclose(output.data[1, 0], embedding.weight.data[7])
        assert np.allclose(output.data[1, 1], embedding.weight.data[12])

    def test_embedding_repeated_indices(self):
        """Test embedding with repeated indices."""
        vocab_size, embed_dim = 10, 4
        embedding = Embedding(vocab_size, embed_dim)

        # Test with repeated indices
        indices = Tensor(np.array([0, 0, 5, 5, 0]).astype(np.int32))
        output = embedding.forward(indices)

        # All instances of same index should have same embedding
        assert np.allclose(output.data[0], output.data[1])  # Both index 0
        assert np.allclose(output.data[0], output.data[4])  # Both index 0
        assert np.allclose(output.data[2], output.data[3])  # Both index 5

        # Different indices should have different embeddings
        assert not np.allclose(output.data[0], output.data[2])  # Index 0 vs 5

    def test_embedding_gradient_computation_setup(self):
        """Test gradient computation setup."""
        vocab_size, embed_dim = 20, 8
        embedding = Embedding(vocab_size, embed_dim)

        # Test with requires_grad=True (weight parameter)
        indices = Tensor(np.array([0, 5, 10]).astype(np.int32))
        output = embedding.forward(indices)

        # Output should require gradients because weight does
        assert output.requires_grad == True
        assert hasattr(output, "_backward")

        # Test gradient setup
        assert embedding.weight.requires_grad == True

    def test_embedding_gradient_computation_disabled(self):
        """Test embedding when gradients are disabled."""
        vocab_size, embed_dim = 15, 6
        embedding = Embedding(vocab_size, embed_dim)

        # Disable gradients
        embedding.weight.requires_grad = False

        indices = Tensor(np.array([0, 5, 10]).astype(np.int32))
        output = embedding.forward(indices)

        # Output should not require gradients
        assert output.requires_grad == False
        assert not hasattr(output, "_backward")

    def test_embedding_call_interface(self):
        """Test the __call__ interface."""
        vocab_size, embed_dim = 25, 12
        embedding = Embedding(vocab_size, embed_dim)

        # Test calling with numpy array
        indices_np = np.array([0, 5, 10, 15], dtype=np.int32)
        output_call = embedding(indices_np)
        output_forward = embedding.forward(indices_np)

        # Results should be identical
        assert np.allclose(output_call.data, output_forward.data)
        assert output_call.shape == output_forward.shape

    def test_embedding_different_vocab_sizes(self):
        """Test embedding with different vocabulary sizes."""
        embed_dim = 16
        vocab_sizes = [1, 2, 10, 100, 1000, 10000]

        for vocab_size in vocab_sizes:
            embedding = Embedding(vocab_size, embed_dim)
            assert embedding.vocab_size == vocab_size
            assert embedding.weight.shape == (vocab_size, embed_dim)

            # Test with valid indices
            max_index = min(vocab_size - 1, 5)  # Use smaller index for small vocabs
            indices = Tensor(np.array([0, max_index]).astype(np.int32))
            output = embedding.forward(indices)
            assert output.shape == (2, embed_dim)

    def test_embedding_different_embed_dims(self):
        """Test embedding with different embedding dimensions."""
        vocab_size = 100
        embed_dims = [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024]

        for embed_dim in embed_dims:
            embedding = Embedding(vocab_size, embed_dim)
            assert embedding.embed_dim == embed_dim
            assert embedding.weight.shape == (vocab_size, embed_dim)

            indices = Tensor(np.array([0, 10, 50]).astype(np.int32))
            output = embedding.forward(indices)
            assert output.shape == (3, embed_dim)

    def test_embedding_edge_case_shapes(self):
        """Test embedding with edge case shapes."""
        # Test minimum viable embedding
        embedding_min = Embedding(1, 1)
        indices_min = Tensor(np.array([0]).astype(np.int32))
        output_min = embedding_min.forward(indices_min)
        assert output_min.shape == (1, 1)

        # Test single vocab item, multiple dimensions
        embedding_single_vocab = Embedding(1, 64)
        indices_batch = Tensor(np.array([0, 0, 0]).astype(np.int32))
        output_batch = embedding_single_vocab.forward(indices_batch)
        assert output_batch.shape == (3, 64)

        # All outputs should be identical (same index)
        assert np.allclose(output_batch.data[0], output_batch.data[1])
        assert np.allclose(output_batch.data[1], output_batch.data[2])

    def test_embedding_numerical_stability(self):
        """Test numerical stability and precision."""
        vocab_size, embed_dim = 50, 32
        embedding = Embedding(vocab_size, embed_dim)

        # Test with all valid indices
        all_indices = Tensor(np.arange(vocab_size).astype(np.int32))
        output_all = embedding.forward(all_indices)
        assert output_all.shape == (vocab_size, embed_dim)
        assert np.all(np.isfinite(output_all.data))

        # Test multiple lookups
        for _ in range(10):
            random_indices = Tensor(np.random.randint(0, vocab_size, size=20).astype(np.int32))
            output_random = embedding.forward(random_indices)
            assert np.all(np.isfinite(output_random.data))

    def test_embedding_dtype_handling(self):
        """Test data type handling."""
        vocab_size, embed_dim = 30, 16
        embedding = Embedding(vocab_size, embed_dim)

        # Test with different input integer types
        indices_int32 = Tensor(np.array([0, 5, 10]).astype(np.int32))
        output_int32 = embedding.forward(indices_int32)

        indices_int64 = Tensor(np.array([0, 5, 10]).astype(np.int64))
        output_int64 = embedding.forward(indices_int64)

        # Results should be identical regardless of input int type
        assert np.allclose(output_int32.data, output_int64.data)

        # Output should always be float32
        assert output_int32.data.dtype == np.float32
        assert output_int64.data.dtype == np.float32

    def test_embedding_memory_efficiency(self):
        """Test memory efficiency with large embeddings."""
        # Test with moderately large embedding
        vocab_size, embed_dim = 1000, 128
        embedding = Embedding(vocab_size, embed_dim)

        # Test multiple batches
        for batch_size in [1, 16, 64, 256]:
            indices = Tensor(np.random.randint(0, vocab_size, size=batch_size).astype(np.int32))
            output = embedding.forward(indices)
            assert output.shape == (batch_size, embed_dim)
            assert np.all(np.isfinite(output.data))

    def test_embedding_parameter_access(self):
        """Test parameter access and properties."""
        vocab_size, embed_dim = 40, 20
        embedding = Embedding(vocab_size, embed_dim)

        # Test parameter properties
        assert hasattr(embedding, "weight")
        assert isinstance(embedding.weight, Parameter)
        assert embedding.weight.data.shape == (vocab_size, embed_dim)
        assert embedding.weight.requires_grad == True

        # Test parameter name
        assert ".weight" in embedding.weight.name
        assert embedding.name in embedding.weight.name

    def test_embedding_state_consistency(self):
        """Test that embedding state remains consistent."""
        vocab_size, embed_dim = 25, 12
        embedding = Embedding(vocab_size, embed_dim)

        # Store initial weight
        initial_weight = embedding.weight.data.copy()

        # Perform multiple forward passes
        for _ in range(5):
            indices = Tensor(np.random.randint(0, vocab_size, size=10).astype(np.int32))
            output = embedding.forward(indices)
            assert output.shape == (10, embed_dim)

        # Weight should remain unchanged (no training)
        assert np.allclose(embedding.weight.data, initial_weight)

    def test_embedding_output_properties(self):
        """Test properties of embedding output."""
        vocab_size, embed_dim = 35, 18
        embedding = Embedding(vocab_size, embed_dim)

        indices = Tensor(np.array([0, 10, 20, 30]).astype(np.int32))
        output = embedding.forward(indices)

        # Test output properties
        assert isinstance(output, Tensor)
        assert output.data.dtype == np.float32
        assert output.shape == (4, embed_dim)
        assert np.all(np.isfinite(output.data))

        # Test that output contains actual weight values
        for i, idx in enumerate([0, 10, 20, 30]):
            expected = embedding.weight.data[idx]
            actual = output.data[i]
            assert np.allclose(actual, expected)

    def test_embedding_zero_based_indexing(self):
        """Test that embedding uses zero-based indexing correctly."""
        vocab_size, embed_dim = 5, 4
        embedding = Embedding(vocab_size, embed_dim)

        # Test each valid index
        for i in range(vocab_size):
            indices = Tensor(np.array([i]).astype(np.int32))
            output = embedding.forward(indices)
            expected = embedding.weight.data[i]
            assert np.allclose(output.data[0], expected)

    def test_embedding_reproducibility(self):
        """Test embedding reproducibility with same initialization."""
        vocab_size, embed_dim = 20, 8

        # Set seed for reproducibility
        np.random.seed(42)
        embedding1 = Embedding(vocab_size, embed_dim)

        # Reset seed
        np.random.seed(42)
        embedding2 = Embedding(vocab_size, embed_dim)

        # Weights should be identical
        assert np.allclose(embedding1.weight.data, embedding2.weight.data)

        # Outputs should be identical
        indices = Tensor(np.array([0, 5, 10]).astype(np.int32))
        output1 = embedding1.forward(indices)
        output2 = embedding2.forward(indices)
        assert np.allclose(output1.data, output2.data)

    def test_embedding_large_batch_processing(self):
        """Test embedding with large batch sizes."""
        vocab_size, embed_dim = 100, 64
        embedding = Embedding(vocab_size, embed_dim)

        # Test with large batch
        large_batch_size = 1000
        indices = Tensor(np.random.randint(0, vocab_size, size=large_batch_size).astype(np.int32))
        output = embedding.forward(indices)

        assert output.shape == (large_batch_size, embed_dim)
        assert np.all(np.isfinite(output.data))

    def test_embedding_complex_indexing_patterns(self):
        """Test embedding with complex indexing patterns."""
        vocab_size, embed_dim = 15, 8
        embedding = Embedding(vocab_size, embed_dim)

        # Test with complex 3D indexing
        indices_3d = Tensor(np.random.randint(0, vocab_size, size=(2, 3, 4)).astype(np.int32))
        output_3d = embedding.forward(indices_3d)
        assert output_3d.shape == (2, 3, 4, embed_dim)

        # Test with complex 4D indexing
        indices_4d = Tensor(np.random.randint(0, vocab_size, size=(2, 2, 2, 2)).astype(np.int32))
        output_4d = embedding.forward(indices_4d)
        assert output_4d.shape == (2, 2, 2, 2, embed_dim)

        # Verify correctness for a few elements
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        idx = indices_4d.data[i, j, k, l]
                        expected = embedding.weight.data[idx]
                        actual = output_4d.data[i, j, k, l]
                        assert np.allclose(actual, expected)
