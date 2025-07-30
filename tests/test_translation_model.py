"""Tests for translation model and vocabulary."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'translation'))

import numpy as np
import pytest
import json
import tempfile
from neural_arch.core import Tensor
from vocabulary import Vocabulary, create_dataset
from model_v2 import TranslationTransformer, PositionalEncoding


class TestVocabulary:
    """Test Vocabulary class."""
    
    def test_init(self):
        """Test vocabulary initialization."""
        vocab = Vocabulary("english")
        
        # Check special tokens
        assert vocab.pad_token == "<PAD>"
        assert vocab.sos_token == "<SOS>"
        assert vocab.eos_token == "<EOS>"
        assert vocab.unk_token == "<UNK>"
        
        # Check initial vocab
        assert len(vocab) == 4  # Only special tokens
        assert vocab.word2idx[vocab.pad_token] == 0
        assert vocab.word2idx[vocab.sos_token] == 1
        
    def test_add_sentence(self):
        """Test adding sentences to vocabulary."""
        vocab = Vocabulary("test")
        
        # Add a sentence
        vocab.add_sentence("hello world")
        
        assert "hello" in vocab.word2idx
        assert "world" in vocab.word2idx
        assert len(vocab) == 6  # 4 special + 2 words
        
    def test_encode_decode(self):
        """Test encoding and decoding."""
        vocab = Vocabulary("test")
        vocab.add_sentence("hello world how are you")
        
        # Test encoding
        text = "hello world"
        indices = vocab.encode(text, max_length=10)
        
        # Note: encode() doesn't add SOS, just EOS
        assert indices[0] == vocab.word2idx["hello"]
        assert indices[1] == vocab.word2idx["world"]
        assert indices[2] == vocab.word2idx[vocab.eos_token]
        assert indices[3] == vocab.word2idx[vocab.pad_token]  # Padding
        
        # Test decoding
        decoded = vocab.decode(indices, remove_special=True)
        assert decoded == "hello world"
        
    def test_unknown_words(self):
        """Test handling of unknown words."""
        vocab = Vocabulary("test")
        vocab.add_sentence("hello world")
        
        # Encode with unknown word
        indices = vocab.encode("hello unknown world", max_length=10)
        
        # "hello" is index 0, "unknown" is index 1 (UNK), "world" is index 2
        assert indices[1] == vocab.word2idx[vocab.unk_token]
        
    def test_save_load(self):
        """Test saving and loading vocabulary."""
        vocab = Vocabulary("test")
        vocab.add_sentence("hello world test")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vocab.save(f.name)
            temp_path = f.name
        
        # Load from file
        loaded_vocab = Vocabulary.load(temp_path)
        
        assert loaded_vocab.language == vocab.language
        assert loaded_vocab.word2idx == vocab.word2idx
        assert loaded_vocab.idx2word == vocab.idx2word
        
        # Clean up
        os.unlink(temp_path)


class TestCreateDataset:
    """Test dataset creation function."""
    
    def test_create_dataset(self):
        """Test creating dataset from pairs."""
        # Create vocabularies
        src_vocab = Vocabulary("english")
        tgt_vocab = Vocabulary("spanish")
        
        # Sample pairs
        pairs = [
            ("hello world", "hola mundo"),
            ("how are you", "cómo estás"),
            ("thank you", "gracias")
        ]
        
        # Create dataset
        src_data, tgt_data = create_dataset(pairs, src_vocab, tgt_vocab, max_length=10)
        
        assert len(src_data) == 3
        assert len(tgt_data) == 3
        
        # Check first example - create_dataset uses encode which doesn't add SOS
        # First word should be "hello" / "hola"
        assert src_data[0][0] == src_vocab.word2idx["hello"]
        assert tgt_data[0][0] == tgt_vocab.word2idx["hola"]
        
        # Check vocabularies were built
        assert "hello" in src_vocab.word2idx
        assert "hola" in tgt_vocab.word2idx


class TestPositionalEncoding:
    """Test positional encoding."""
    
    def test_encoding_shape(self):
        """Test positional encoding shape."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        
        assert pe.encoding.shape == (100, 64)
        
    def test_encoding_values(self):
        """Test that positional encoding has expected properties."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        
        # Check that values are bounded
        assert np.abs(pe.encoding).max() <= 1.0
        
        # Check that different positions have different encodings
        assert not np.allclose(pe.encoding[0], pe.encoding[1])
        
    def test_apply_encoding(self):
        """Test applying positional encoding to tensor."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        
        # Create input tensor
        x = Tensor(np.zeros((2, 10, 64)))
        
        # Apply encoding
        output = pe(x)
        
        # Check shape preserved
        assert output.data.shape == x.data.shape
        
        # Check that encoding was added
        assert not np.allclose(output.data, x.data)


class TestTranslationTransformer:
    """Test TranslationTransformer model."""
    
    def test_init(self):
        """Test model initialization."""
        model = TranslationTransformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128
        )
        
        assert model.d_model == 64
        assert model.n_heads == 4
        assert model.n_layers == 2
        assert len(model.encoder_layers) == 2
        assert len(model.decoder_layers) == 2
        
    def test_parameters(self):
        """Test that model returns parameters correctly."""
        model = TranslationTransformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64
        )
        
        # Get parameters
        params = list(model.parameters())
        
        # Should have many parameters
        assert len(params) > 10
        
        # All should be Parameter objects
        from neural_arch.core import Parameter
        for param in params:
            assert isinstance(param, Parameter)
            assert param.requires_grad
            
    def test_forward(self):
        """Test forward pass."""
        model = TranslationTransformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128
        )
        
        # Create inputs
        src = Tensor(np.array([[1, 2, 3, 0, 0]]))  # Source with padding
        tgt = Tensor(np.array([[1, 4, 5, 6]]))     # Target
        
        # Forward pass
        output = model(src, tgt)
        
        # Check output shape
        assert output.data.shape == (1, 4, 100)  # (batch, tgt_len, vocab_size)
        
    def test_masks(self):
        """Test mask creation."""
        model = TranslationTransformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1
        )
        
        # Test padding mask
        seq = np.array([1, 2, 3, 0, 0])
        mask = model.create_padding_mask(seq, pad_idx=0)
        expected = np.array([0, 0, 0, 1, 1], dtype=np.float32)
        assert np.array_equal(mask, expected)
        
        # Test look-ahead mask
        size = 4
        mask = model.create_look_ahead_mask(size)
        assert mask.shape == (size, size)
        assert np.array_equal(mask, np.triu(np.ones((size, size)), k=1))
        
    def test_generate(self):
        """Test generation/inference."""
        model = TranslationTransformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64
        )
        
        # Create source input
        src = Tensor(np.array([[1, 2, 3, 4, 2]]))  # SOS, words, EOS
        
        # Generate
        output = model.generate(
            src,
            max_length=10,
            sos_idx=1,
            eos_idx=2,
            temperature=1.0
        )
        
        # Check output
        assert isinstance(output, list)
        assert len(output) <= 10
        assert all(isinstance(idx, int) for idx in output)


class TestIntegration:
    """Integration tests for the translation system."""
    
    def test_small_training_step(self):
        """Test a single training step."""
        # Create small model
        model = TranslationTransformer(
            src_vocab_size=20,
            tgt_vocab_size=20,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64
        )
        
        # Create simple data
        src = Tensor(np.array([[1, 2, 3, 4, 0]]))
        tgt_in = Tensor(np.array([[1, 5, 6, 7]]))
        tgt_out = Tensor(np.array([[5, 6, 7, 2]]))
        
        # Forward pass
        output = model(src, tgt_in)
        
        # Compute loss
        from neural_arch.functional import cross_entropy_loss
        output_flat = output.data.reshape(-1, output.data.shape[-1])
        target_flat = tgt_out.data.reshape(-1)
        
        loss = cross_entropy_loss(
            Tensor(output_flat, requires_grad=True),
            Tensor(target_flat, requires_grad=False)
        )
        
        # Loss should be reasonable
        assert 0 < loss.data < 10  # Log of vocab size
        
    def test_translation_pipeline(self):
        """Test full translation pipeline."""
        # Create vocabularies
        src_vocab = Vocabulary("english")
        tgt_vocab = Vocabulary("spanish")
        
        # Add some words
        for sentence in ["hello world", "how are you", "i am fine"]:
            src_vocab.add_sentence(sentence)
        for sentence in ["hola mundo", "cómo estás", "estoy bien"]:
            tgt_vocab.add_sentence(sentence)
            
        # Create model
        model = TranslationTransformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64
        )
        
        # Test translation
        test_sentence = "hello world"
        src_indices = src_vocab.encode(test_sentence, max_length=10)
        src_tensor = Tensor(np.array([src_indices]))
        
        # Generate translation
        output_indices = model.generate(
            src_tensor,
            max_length=10,
            sos_idx=tgt_vocab.word2idx[tgt_vocab.sos_token],
            eos_idx=tgt_vocab.word2idx[tgt_vocab.eos_token],
            temperature=0.1  # Low temperature for more deterministic output
        )
        
        # Decode
        translation = tgt_vocab.decode(output_indices, remove_special=True)
        
        # Check that we got some output
        assert isinstance(translation, str)
        assert len(translation) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])