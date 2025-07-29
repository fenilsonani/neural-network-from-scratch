"""Utility functions for backward compatibility."""

import numpy as np
from typing import Dict, Tuple, Optional

from ..core import Tensor


def propagate_gradients(tensor: Tensor) -> None:
    """Propagate gradients through computation graph (backward compatibility)."""
    if hasattr(tensor, '_backward'):
        tensor._backward()


def create_text_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create character-level vocabulary from text."""
    chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char


def text_to_sequences(text: str, seq_len: int, char_to_idx: Dict[str, int]) -> np.ndarray:
    """Convert text to training sequences."""
    sequences = []
    for i in range(len(text) - seq_len):
        seq = text[i:i + seq_len + 1]
        indices = [char_to_idx.get(c, 0) for c in seq]
        sequences.append(indices)
    return np.array(sequences)