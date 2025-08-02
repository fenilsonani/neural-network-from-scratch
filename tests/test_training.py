"""
Integration tests for full training pipeline - the real deal.
"""

try:
    import pytest
except ImportError:
    pytest = None
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_arch import (
    Adam,
    Embedding,
    Linear,
    Tensor,
    create_text_vocab,
    relu,
    softmax,
    text_to_sequences,
)


class SimpleTestModel:
    """Simple model for testing."""

    def __init__(self, vocab_size: int, embed_dim: int = 8, hidden_dim: int = 16):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, vocab_size)

    def forward(self, x: np.ndarray) -> Tensor:
        embedded = self.embedding(x)
        pooled_data = np.mean(embedded.data, axis=1)
        pooled = Tensor(pooled_data, embedded.requires_grad)

        h1 = self.linear1(pooled)
        h1_relu = relu(h1)
        output = self.linear2(h1_relu)

        return output

    def parameters(self):
        params = {}
        params.update(self.embedding.parameters())
        params.update(self.linear1.parameters())
        params.update(self.linear2.parameters())
        return params


class TestTrainingPipeline:
    """Test complete training pipeline."""

    def test_data_preprocessing(self):
        """Test text data preprocessing."""
        text = "hello world"
        char_to_idx, idx_to_char = create_text_vocab(text)

        # Check vocabulary
        expected_chars = [" ", "d", "e", "h", "l", "o", "r", "w"]
        assert sorted(char_to_idx.keys()) == expected_chars
        assert len(idx_to_char) == len(char_to_idx)

        # Check sequences
        sequences = text_to_sequences(text, seq_len=3, char_to_idx=char_to_idx)
        assert sequences.shape[1] == 4  # seq_len + 1
        assert len(sequences) == len(text) - 3

    def test_single_training_step(self):
        """Test single training step doesn't crash."""
        # Create simple data
        text = "abc def"
        char_to_idx, idx_to_char = create_text_vocab(text)
        sequences = text_to_sequences(text, seq_len=2, char_to_idx=char_to_idx)

        # Create model
        model = SimpleTestModel(vocab_size=len(char_to_idx))
        optimizer = Adam(model.parameters(), lr=0.1)

        # Get batch
        batch = sequences[:2]  # Small batch
        inputs = batch[:, :-1]
        targets = batch[:, -1]

        # Forward pass
        outputs = model.forward(inputs)

        # Compute loss
        batch_size = outputs.shape[0]
        vocab_size = outputs.shape[1]
        target_one_hot = np.zeros((batch_size, vocab_size))
        target_one_hot[np.arange(batch_size), targets] = 1.0

        probs = softmax(outputs)
        eps = 1e-8
        loss_val = -np.mean(np.sum(target_one_hot * np.log(probs.data + eps), axis=1))

        # Backward pass
        grad = probs.data - target_one_hot
        grad /= batch_size

        probs.backward(grad)
        if hasattr(probs, "_backward"):
            probs._backward()
        if hasattr(outputs, "_backward"):
            outputs._backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()

        # Should not crash
        assert loss_val > 0
        assert np.isfinite(loss_val)

    def test_multi_step_training(self):
        """Test training for multiple steps."""
        # Create data
        text = "hello world test"
        char_to_idx, idx_to_char = create_text_vocab(text)
        sequences = text_to_sequences(text, seq_len=3, char_to_idx=char_to_idx)

        # Create model
        model = SimpleTestModel(vocab_size=len(char_to_idx), embed_dim=4, hidden_dim=8)
        optimizer = Adam(model.parameters(), lr=0.1)

        losses = []

        # Train for 10 steps
        for step in range(10):
            # Get random batch
            batch_indices = np.random.choice(len(sequences), size=2, replace=True)
            batch = sequences[batch_indices]

            inputs = batch[:, :-1]
            targets = batch[:, -1]

            # Forward
            outputs = model.forward(inputs)

            # Loss
            batch_size = outputs.shape[0]
            vocab_size = outputs.shape[1]
            target_one_hot = np.zeros((batch_size, vocab_size))
            target_one_hot[np.arange(batch_size), targets] = 1.0

            probs = softmax(outputs)
            loss_val = -np.mean(np.sum(target_one_hot * np.log(probs.data + 1e-8), axis=1))
            losses.append(loss_val)

            # Backward
            grad = probs.data - target_one_hot
            grad /= batch_size

            probs.backward(grad)
            if hasattr(probs, "_backward"):
                probs._backward()
            if hasattr(outputs, "_backward"):
                outputs._backward()

            # Update
            optimizer.step()
            optimizer.zero_grad()

        # Training should complete without crashes
        assert len(losses) == 10
        assert all(np.isfinite(loss) for loss in losses)

        # Loss should generally decrease or at least not explode
        final_loss = np.mean(losses[-3:])
        initial_loss = np.mean(losses[:3])
        assert final_loss < initial_loss * 2  # Allow some variance

    def test_overfitting_on_small_data(self):
        """Test that model can overfit on very small dataset."""
        # Tiny dataset - should be able to memorize
        text = "ab"
        char_to_idx, idx_to_char = create_text_vocab(text)
        sequences = text_to_sequences(text, seq_len=1, char_to_idx=char_to_idx)

        # Only one sequence: [a] -> b
        assert len(sequences) == 1

        model = SimpleTestModel(vocab_size=len(char_to_idx), embed_dim=4, hidden_dim=8)
        optimizer = Adam(model.parameters(), lr=0.1)

        losses = []

        # Train on same sequence many times
        for step in range(50):
            inputs = sequences[:, :-1]  # [a]
            targets = sequences[:, -1]  # b

            outputs = model.forward(inputs)

            # Loss
            batch_size = 1
            vocab_size = len(char_to_idx)
            target_one_hot = np.zeros((batch_size, vocab_size))
            target_one_hot[0, targets[0]] = 1.0

            probs = softmax(outputs)
            loss_val = -np.log(probs.data[0, targets[0]] + 1e-8)
            losses.append(loss_val)

            # Backward
            grad = probs.data - target_one_hot

            probs.backward(grad)
            if hasattr(probs, "_backward"):
                probs._backward()
            if hasattr(outputs, "_backward"):
                outputs._backward()

            optimizer.step()
            optimizer.zero_grad()

        # Should overfit - loss should decrease significantly
        assert losses[-1] < losses[0] * 0.5  # At least 50% reduction

    def test_model_prediction_changes(self):
        """Test that model predictions change during training."""
        text = "hello test"
        char_to_idx, idx_to_char = create_text_vocab(text)
        sequences = text_to_sequences(text, seq_len=2, char_to_idx=char_to_idx)

        model = SimpleTestModel(vocab_size=len(char_to_idx))
        optimizer = Adam(model.parameters(), lr=0.1)

        # Get initial prediction
        test_input = sequences[0:1, :-1]
        initial_output = model.forward(test_input)
        initial_probs = softmax(initial_output).data.copy()

        # Train for several steps
        for step in range(20):
            batch_idx = step % len(sequences)
            inputs = sequences[batch_idx : batch_idx + 1, :-1]
            targets = sequences[batch_idx : batch_idx + 1, -1]

            outputs = model.forward(inputs)
            probs = softmax(outputs)

            # Simple loss
            target_prob = probs.data[0, targets[0]]
            loss_val = -np.log(target_prob + 1e-8)

            # Backward
            grad = np.zeros_like(probs.data)
            grad[0, targets[0]] = -1.0 / (target_prob + 1e-8)

            probs.backward(grad)
            if hasattr(probs, "_backward"):
                probs._backward()
            if hasattr(outputs, "_backward"):
                outputs._backward()

            optimizer.step()
            optimizer.zero_grad()

        # Get final prediction
        final_output = model.forward(test_input)
        final_probs = softmax(final_output).data

        # Predictions should have changed
        prob_diff = np.sum(np.abs(final_probs - initial_probs))
        assert prob_diff > 0.01  # Should be noticeable change


class TestTrainingStability:
    """Test training stability and error handling."""

    def test_gradient_explosion_handling(self):
        """Test that gradient clipping prevents explosion."""
        model = SimpleTestModel(vocab_size=5, embed_dim=2, hidden_dim=4)
        optimizer = Adam(model.parameters(), lr=1.0)  # Large LR

        # Create artificial large gradients
        for param in model.parameters().values():
            param.grad = np.ones_like(param.data) * 100.0  # Huge gradients

        # Should not crash
        optimizer.step()

        # Parameters should change but not explode
        for param in model.parameters().values():
            assert np.all(np.isfinite(param.data))
            assert np.all(np.abs(param.data) < 1000)  # Reasonable bounds

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        model = SimpleTestModel(vocab_size=3)

        # Test with extreme inputs
        extreme_input = np.array([[0, 1, 2] * 10])  # Long sequence

        try:
            output = model.forward(extreme_input[:, :5])  # Truncate
            probs = softmax(output)

            # Should not produce NaN or Inf
            assert np.all(np.isfinite(output.data))
            assert np.all(np.isfinite(probs.data))
            assert np.allclose(np.sum(probs.data, axis=1), 1.0)

        except Exception as e:
            if pytest:
                pytest.fail(f"Numerical stability test failed: {e}")
            else:
                raise AssertionError(f"Numerical stability test failed: {e}")

    def test_empty_gradients(self):
        """Test handling of parameters without gradients."""
        model = SimpleTestModel(vocab_size=3)
        optimizer = Adam(model.parameters(), lr=0.1)

        # Don't set any gradients
        original_params = {name: param.data.copy() for name, param in model.parameters().items()}

        # Should not crash
        optimizer.step()
        optimizer.zero_grad()

        # Parameters should not change
        for name, param in model.parameters().items():
            assert np.array_equal(param.data, original_params[name])


def test_end_to_end_training():
    """Test complete end-to-end training pipeline."""
    print("🚀 Running end-to-end training test...")

    # Create data
    text = "neural networks are awesome"
    char_to_idx, idx_to_char = create_text_vocab(text)
    sequences = text_to_sequences(text, seq_len=4, char_to_idx=char_to_idx)

    print(f"Text: '{text}'")
    print(f"Vocabulary: {len(char_to_idx)} characters")
    print(f"Sequences: {len(sequences)}")

    # Create model
    model = SimpleTestModel(vocab_size=len(char_to_idx), embed_dim=8, hidden_dim=16)

    optimizer = Adam(model.parameters(), lr=0.05)

    # Track metrics
    losses = []
    accuracies = []

    print("Training...")

    # Training loop
    for epoch in range(15):
        epoch_losses = []
        correct = 0
        total = 0

        # Shuffle data
        np.random.shuffle(sequences)

        # Mini-batches - ensure consistent batch size
        batch_size = 2
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            if len(batch) < batch_size:
                # Pad the last batch if needed
                padding_needed = batch_size - len(batch)
                padding = np.repeat(batch[-1:], padding_needed, axis=0)
                batch = np.concatenate([batch, padding], axis=0)

            inputs = batch[:, :-1]
            targets = batch[:, -1]

            # Forward
            outputs = model.forward(inputs)

            # Loss
            batch_size = outputs.shape[0]
            vocab_size = outputs.shape[1]

            target_one_hot = np.zeros((batch_size, vocab_size))
            target_one_hot[np.arange(batch_size), targets] = 1.0

            probs = softmax(outputs)
            loss_val = -np.mean(np.sum(target_one_hot * np.log(probs.data + 1e-8), axis=1))
            epoch_losses.append(loss_val)

            # Accuracy
            predicted = np.argmax(outputs.data, axis=1)
            correct += np.sum(predicted == targets)
            total += len(targets)

            # Backward
            grad = probs.data - target_one_hot
            grad /= batch_size

            probs.backward(grad)
            if hasattr(probs, "_backward"):
                probs._backward()
            if hasattr(outputs, "_backward"):
                outputs._backward()

            # Update
            optimizer.step()
            optimizer.zero_grad()

        # Record metrics
        avg_loss = np.mean(epoch_losses)
        accuracy = correct / total if total > 0 else 0

        losses.append(avg_loss)
        accuracies.append(accuracy)

        if epoch % 3 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    # Verify training worked
    assert len(losses) == 15
    assert len(accuracies) == 15

    # Loss should generally decrease
    early_loss = np.mean(losses[:3])
    late_loss = np.mean(losses[-3:])
    assert late_loss <= early_loss  # Should improve or stay stable

    # Accuracy should be reasonable
    final_accuracy = accuracies[-1]
    assert final_accuracy >= 0.1  # At least better than random

    print(f"✅ Training completed successfully!")
    print(f"Initial loss: {losses[0]:.4f} -> Final loss: {losses[-1]:.4f}")
    print(f"Final accuracy: {final_accuracy:.4f}")


if __name__ == "__main__":
    # Run tests manually
    test_pipeline = TestTrainingPipeline()
    test_stability = TestTrainingStability()

    print("🧪 Running training tests...")

    try:
        # Pipeline tests
        test_pipeline.test_data_preprocessing()
        test_pipeline.test_single_training_step()
        test_pipeline.test_multi_step_training()
        test_pipeline.test_overfitting_on_small_data()
        test_pipeline.test_model_prediction_changes()
        print("✅ Training pipeline tests passed")

        # Stability tests
        test_stability.test_gradient_explosion_handling()
        test_stability.test_numerical_stability()
        test_stability.test_empty_gradients()
        print("✅ Training stability tests passed")

        # End-to-end test
        test_end_to_end_training()
        print("✅ End-to-end training test passed")

        print("\n🎉 ALL TRAINING TESTS PASSED!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
