"""Translation model using transformer architecture."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
from typing import Dict, Optional, Tuple, List
from neural_arch.core import Tensor
from neural_arch.nn import Linear, Embedding, LayerNorm
from neural_arch.nn import MultiHeadAttention, TransformerBlock, TransformerDecoderBlock
from neural_arch.functional import softmax, cross_entropy_loss
from neural_arch.functional.arithmetic import matmul


class PositionalEncoding:
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        self.encoding = self._create_encoding()
    
    def _create_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        encoding = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)
        
        return encoding
    
    def __call__(self, x: Tensor) -> Tensor:
        """Add positional encoding to input tensor."""
        seq_len = x.data.shape[1]
        pos_encoding = self.encoding[:seq_len, :]
        return Tensor(x.data + pos_encoding, requires_grad=x.requires_grad)


class TranslationTransformer:
    """Transformer model for translation."""
    
    def __init__(self, 
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 100,
                 dropout: float = 0.1):
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Embeddings
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder layers
        self.encoder_layers = []
        for _ in range(n_layers):
            self.encoder_layers.append(
                TransformerBlock(d_model, n_heads, d_ff, dropout)
            )
        
        # Decoder layers
        self.decoder_layers = []
        for _ in range(n_layers):
            self.decoder_layers.append(
                TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            )
        
        # Output projection
        self.output_projection = Linear(d_model, tgt_vocab_size)
        
        # Scale factor for embeddings
        self.scale = np.sqrt(d_model)
    
    def create_padding_mask(self, x: np.ndarray, pad_idx: int = 0) -> np.ndarray:
        """Create padding mask (1 for padding, 0 for real tokens)."""
        return (x == pad_idx).astype(np.float32)
    
    def create_look_ahead_mask(self, size: int) -> np.ndarray:
        """Create look-ahead mask for decoder self-attention."""
        mask = np.triu(np.ones((size, size)), k=1)
        return mask
    
    def encode(self, src: Tensor, src_mask: Optional[np.ndarray] = None) -> Tensor:
        """Encode source sequence."""
        # Embedding and positional encoding
        x = self.src_embedding(src)
        x = Tensor(x.data * self.scale, x.requires_grad)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        
        return x
    
    def decode(self, tgt: Tensor, memory: Tensor, 
               tgt_mask: Optional[np.ndarray] = None,
               memory_mask: Optional[np.ndarray] = None) -> Tensor:
        """Decode target sequence given encoder output."""
        # Embedding and positional encoding
        x = self.tgt_embedding(tgt)
        x = Tensor(x.data * self.scale, x.requires_grad)
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        return x
    
    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None,
                memory_mask: Optional[np.ndarray] = None) -> Tensor:
        """Forward pass through the model."""
        # Encode source
        memory = self.encode(src, src_mask)
        
        # Decode target
        output = self.decode(tgt, memory, tgt_mask, memory_mask)
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output
    
    def __call__(self, src: Tensor, tgt: Tensor, **kwargs) -> Tensor:
        return self.forward(src, tgt, **kwargs)
    
    def parameters(self) -> Dict[str, Tensor]:
        """Get all model parameters."""
        params = {}
        
        # Embedding parameters
        params.update({f'src_emb_{k}': v for k, v in self.src_embedding.parameters().items()})
        params.update({f'tgt_emb_{k}': v for k, v in self.tgt_embedding.parameters().items()})
        
        # Encoder parameters
        for i, layer in enumerate(self.encoder_layers):
            params.update({f'enc_{i}_{k}': v for k, v in layer.parameters().items()})
        
        # Decoder parameters
        for i, layer in enumerate(self.decoder_layers):
            params.update({f'dec_{i}_{k}': v for k, v in layer.parameters().items()})
        
        # Output projection parameters
        params.update({f'output_{k}': v for k, v in self.output_projection.parameters().items()})
        
        return params
    
    def generate(self, src: Tensor, max_length: int = 50, 
                 sos_idx: int = 1, eos_idx: int = 2, 
                 temperature: float = 1.0) -> List[int]:
        """Generate translation using greedy decoding."""
        # Encode source
        src_mask = self.create_padding_mask(src.data[0])
        memory = self.encode(src, src_mask)
        
        # Start with SOS token
        output = [sos_idx]
        
        for _ in range(max_length):
            # Create target tensor
            tgt = Tensor(np.array([output]), requires_grad=False)
            
            # Create masks
            tgt_mask = self.create_look_ahead_mask(len(output))
            
            # Decode
            dec_output = self.decode(tgt, memory, tgt_mask, src_mask)
            
            # Get last token probabilities
            logits = self.output_projection(dec_output)
            last_logits = logits.data[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = softmax(Tensor(last_logits[np.newaxis, :], requires_grad=False))
            
            # Get next token (greedy)
            next_token = np.argmax(probs.data)
            output.append(int(next_token))
            
            # Stop if EOS token
            if next_token == eos_idx:
                break
        
        return output