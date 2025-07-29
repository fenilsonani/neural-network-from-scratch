"""Vocabulary management for English-Spanish translation."""

from typing import Dict, List, Tuple, Optional
import json
import re
from collections import Counter


class Vocabulary:
    """Handles vocabulary for translation tasks."""
    
    def __init__(self, language: str):
        self.language = language
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_count: Dict[str, int] = Counter()
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        # Initialize special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            self.add_word(token)
    
    def add_word(self, word: str) -> int:
        """Add a word to vocabulary."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
        return self.word2idx[word]
    
    def add_sentence(self, sentence: str):
        """Add all words in a sentence to vocabulary."""
        words = self.tokenize(sentence)
        for word in words:
            self.add_word(word)
    
    def tokenize(self, sentence: str) -> List[str]:
        """Simple tokenization by splitting on spaces and punctuation."""
        # Convert to lowercase
        sentence = sentence.lower().strip()
        # Add spaces around punctuation
        sentence = re.sub(r"([.!?¿¡,])", r" \1 ", sentence)
        # Remove extra spaces
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.split()
    
    def encode(self, sentence: str, max_length: Optional[int] = None) -> List[int]:
        """Convert sentence to indices."""
        words = self.tokenize(sentence)
        indices = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        
        # Add EOS token
        indices.append(self.word2idx[self.eos_token])
        
        # Pad or truncate to max_length if specified
        if max_length is not None:
            if len(indices) < max_length:
                indices.extend([self.word2idx[self.pad_token]] * (max_length - len(indices)))
            else:
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Convert indices back to sentence."""
        words = []
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if remove_special and word in [self.pad_token, self.sos_token, self.eos_token]:
                    continue
                words.append(word)
        return " ".join(words)
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """Save vocabulary to file."""
        data = {
            "language": self.language,
            "word2idx": self.word2idx,
            "word_count": dict(self.word_count)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(data["language"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        vocab.word_count = Counter(data["word_count"])
        return vocab


def create_dataset(pairs: List[Tuple[str, str]], 
                  src_vocab: Vocabulary, 
                  tgt_vocab: Vocabulary,
                  max_length: int = 50) -> Tuple[List[List[int]], List[List[int]]]:
    """Create encoded dataset from sentence pairs."""
    src_data = []
    tgt_data = []
    
    for src_sentence, tgt_sentence in pairs:
        # Add sentences to vocabulary
        src_vocab.add_sentence(src_sentence)
        tgt_vocab.add_sentence(tgt_sentence)
        
        # Encode sentences
        src_encoded = src_vocab.encode(src_sentence, max_length)
        tgt_encoded = tgt_vocab.encode(tgt_sentence, max_length)
        
        src_data.append(src_encoded)
        tgt_data.append(tgt_encoded)
    
    return src_data, tgt_data