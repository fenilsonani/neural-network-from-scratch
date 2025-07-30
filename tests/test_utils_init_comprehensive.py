"""Comprehensive tests for utils/__init__.py to improve coverage from 81.82% to 100%.

This file targets utility functions including propagate_gradients and text utilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.utils import propagate_gradients, create_text_vocab, text_to_sequences
from neural_arch.core.tensor import Tensor


class TestUtilsInit:
    """Comprehensive tests for utils module."""
    
    def test_propagate_gradients_with_backward(self):
        """Test propagate_gradients with tensor that has _backward method."""
        # Create a tensor with requires_grad
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = Tensor([[2, 2, 2]], requires_grad=True)
        
        # Perform operation to create computation graph
        from neural_arch.functional.arithmetic import mul
        z = mul(x, y)
        
        # Mock backward was called
        backward_called = False
        original_backward = z._backward if hasattr(z, '_backward') else None
        
        def mock_backward():
            nonlocal backward_called
            backward_called = True
            if original_backward:
                original_backward()
        
        z._backward = mock_backward
        
        # Call propagate_gradients
        propagate_gradients(z)
        
        # Verify _backward was called
        assert backward_called
    
    def test_propagate_gradients_without_backward(self):
        """Test propagate_gradients with tensor without _backward method."""
        # Create a tensor without requires_grad
        x = Tensor([[1, 2, 3]], requires_grad=False)
        
        # Should not have _backward
        assert not hasattr(x, '_backward') or x._backward is None
        
        # Call propagate_gradients - should not raise error
        propagate_gradients(x)  # Should be no-op
        
        # Create a mock object without _backward
        class MockTensor:
            def __init__(self):
                self.data = np.array([1, 2, 3])
        
        mock = MockTensor()
        propagate_gradients(mock)  # Should handle gracefully
    
    def test_create_text_vocab_basic(self):
        """Test create_text_vocab with basic text."""
        text = "hello world"
        char_to_idx, idx_to_char = create_text_vocab(text)
        
        # Check unique characters are mapped
        unique_chars = sorted(set(text))
        assert len(char_to_idx) == len(unique_chars)
        assert len(idx_to_char) == len(unique_chars)
        
        # Check all characters are in vocab
        for char in text:
            assert char in char_to_idx
            idx = char_to_idx[char]
            assert idx_to_char[idx] == char
        
        # Check sorted order
        chars_list = [idx_to_char[i] for i in range(len(idx_to_char))]
        assert chars_list == sorted(chars_list)
    
    def test_create_text_vocab_empty(self):
        """Test create_text_vocab with empty text."""
        text = ""
        char_to_idx, idx_to_char = create_text_vocab(text)
        
        # Should return empty mappings
        assert len(char_to_idx) == 0
        assert len(idx_to_char) == 0
    
    def test_create_text_vocab_single_char(self):
        """Test create_text_vocab with single character."""
        text = "aaaa"
        char_to_idx, idx_to_char = create_text_vocab(text)
        
        # Should have one mapping
        assert len(char_to_idx) == 1
        assert len(idx_to_char) == 1
        assert char_to_idx['a'] == 0
        assert idx_to_char[0] == 'a'
    
    def test_create_text_vocab_special_chars(self):
        """Test create_text_vocab with special characters."""
        text = "Hello, World!\n\t123"
        char_to_idx, idx_to_char = create_text_vocab(text)
        
        # Check all special characters are handled
        special_chars = [',', ' ', '!', '\n', '\t']
        for char in special_chars:
            assert char in char_to_idx
            idx = char_to_idx[char]
            assert idx_to_char[idx] == char
        
        # Numbers should be included
        for digit in "123":
            assert digit in char_to_idx
    
    def test_create_text_vocab_unicode(self):
        """Test create_text_vocab with unicode characters."""
        text = "Hello 世界 🌍"
        char_to_idx, idx_to_char = create_text_vocab(text)
        
        # All unique characters including unicode
        unique_chars = set(text)
        assert len(char_to_idx) == len(unique_chars)
        
        # Check unicode characters are handled
        assert '世' in char_to_idx
        assert '界' in char_to_idx
        assert '🌍' in char_to_idx
    
    def test_text_to_sequences_basic(self):
        """Test text_to_sequences with basic input."""
        text = "hello"
        seq_len = 2
        char_to_idx = {'h': 0, 'e': 1, 'l': 2, 'o': 3}
        
        sequences = text_to_sequences(text, seq_len, char_to_idx)
        
        # Expected sequences: "hel", "ell", "llo"
        expected = np.array([
            [0, 1, 2],  # "hel"
            [1, 2, 2],  # "ell"
            [2, 2, 3],  # "llo"
        ])
        
        np.testing.assert_array_equal(sequences, expected)
    
    def test_text_to_sequences_full_workflow(self):
        """Test text_to_sequences with create_text_vocab workflow."""
        text = "abcdefg"
        seq_len = 3
        
        # Create vocab
        char_to_idx, idx_to_char = create_text_vocab(text)
        
        # Convert to sequences
        sequences = text_to_sequences(text, seq_len, char_to_idx)
        
        # Should have len(text) - seq_len sequences
        assert len(sequences) == len(text) - seq_len
        
        # Each sequence should have seq_len + 1 elements
        assert sequences.shape == (len(text) - seq_len, seq_len + 1)
        
        # Verify sequences
        for i, seq in enumerate(sequences):
            # Convert back to text
            text_seq = ''.join(idx_to_char[idx] for idx in seq)
            # Should match substring
            assert text_seq == text[i:i + seq_len + 1]
    
    def test_text_to_sequences_missing_chars(self):
        """Test text_to_sequences with characters not in vocabulary."""
        text = "hello world"
        seq_len = 2
        # Incomplete vocabulary (missing some chars)
        char_to_idx = {'h': 1, 'e': 2, 'l': 3}  # Missing 'o', ' ', 'w', 'r', 'd'
        
        sequences = text_to_sequences(text, seq_len, char_to_idx)
        
        # Missing characters should map to 0 (default)
        # Check that it doesn't crash and uses get() default
        assert sequences.shape[0] == len(text) - seq_len
        
        # First sequence "hel" -> [1, 2, 3]
        expected_first = [1, 2, 3]
        np.testing.assert_array_equal(sequences[0], expected_first)
        
        # Fourth sequence "lo " -> [3, 0, 0] (o and space missing)
        expected_fourth = [3, 0, 0]
        np.testing.assert_array_equal(sequences[3], expected_fourth)
    
    def test_text_to_sequences_edge_cases(self):
        """Test text_to_sequences with edge cases."""
        # Text shorter than seq_len
        text = "hi"
        seq_len = 5
        char_to_idx = {'h': 0, 'i': 1}
        
        sequences = text_to_sequences(text, seq_len, char_to_idx)
        # Should return empty array
        assert len(sequences) == 0
        assert sequences.shape == (0,)
        
        # Text exactly seq_len + 1
        text = "abc"
        seq_len = 2
        char_to_idx = {'a': 0, 'b': 1, 'c': 2}
        
        sequences = text_to_sequences(text, seq_len, char_to_idx)
        # Should have exactly one sequence
        assert len(sequences) == 1
        np.testing.assert_array_equal(sequences[0], [0, 1, 2])
        
        # Empty text
        text = ""
        seq_len = 1
        char_to_idx = {}
        
        sequences = text_to_sequences(text, seq_len, char_to_idx)
        assert len(sequences) == 0
    
    def test_text_to_sequences_large_text(self):
        """Test text_to_sequences with larger text."""
        # Generate large text
        text = "abcdefghijklmnopqrstuvwxyz" * 10
        seq_len = 10
        
        char_to_idx, _ = create_text_vocab(text)
        sequences = text_to_sequences(text, seq_len, char_to_idx)
        
        # Verify shape
        expected_num_seqs = len(text) - seq_len
        assert sequences.shape == (expected_num_seqs, seq_len + 1)
        
        # Verify all values are valid indices
        assert np.all(sequences >= 0)
        assert np.all(sequences < len(char_to_idx))
    
    def test_integrated_text_processing(self):
        """Test integrated workflow of text processing utilities."""
        # Sample text
        text = "The quick brown fox jumps"
        seq_len = 5
        
        # Create vocabulary
        char_to_idx, idx_to_char = create_text_vocab(text)
        
        # Convert to sequences
        sequences = text_to_sequences(text, seq_len, char_to_idx)
        
        # Use with tensor for mock training
        for seq in sequences[:3]:  # Just test first 3
            # Input and target
            input_seq = seq[:-1]
            target = seq[-1]
            
            # Could be used as: Tensor(input_seq) -> model -> predict target
            assert len(input_seq) == seq_len
            assert target in range(len(char_to_idx))
            
            # Verify we can reconstruct text
            reconstructed = ''.join(idx_to_char[idx] for idx in seq)
            start_idx = np.where(sequences[0] == seq[0])[0][0] if len(sequences) > 0 else 0
            # Basic check that it's valid text
            assert len(reconstructed) == seq_len + 1