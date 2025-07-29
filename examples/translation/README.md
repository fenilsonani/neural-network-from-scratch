# English to Spanish Translation Model

A transformer-based neural machine translation model built from scratch using our neural architecture framework.

## Features

- **Transformer Architecture**: Full encoder-decoder transformer model
- **Attention Mechanism**: Multi-head self-attention and cross-attention
- **Positional Encoding**: Sinusoidal positional embeddings
- **Vocabulary Management**: Dynamic vocabulary building with special tokens
- **Interactive Translation**: Real-time translation interface

## Architecture Details

```
Model Components:
├── Encoder (2 layers)
│   ├── Multi-Head Self-Attention (4 heads)
│   ├── Feed-Forward Network (256 dims)
│   └── Layer Normalization
├── Decoder (2 layers)
│   ├── Masked Multi-Head Self-Attention
│   ├── Multi-Head Cross-Attention
│   ├── Feed-Forward Network
│   └── Layer Normalization
└── Output Projection Layer
```

## Quick Start

### 1. Train the Model

```bash
python train.py
```

This will:
- Build vocabularies from the sample dataset
- Train a small transformer model (128 dims, 4 heads, 2 layers)
- Show translation examples during training
- Save vocabularies for later use

### 2. Interactive Translation

```bash
python translate.py
```

This provides an interactive console where you can type English sentences and see Spanish translations.

## Usage Example

```python
from vocabulary import Vocabulary
from model import TranslationTransformer
from translate import Translator

# Load vocabularies
src_vocab = Vocabulary.load("vocab_en.json")
tgt_vocab = Vocabulary.load("vocab_es.json")

# Create model
model = TranslationTransformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=128,
    n_heads=4,
    n_layers=2,
    d_ff=256
)

# Create translator
translator = Translator(model, src_vocab, tgt_vocab)

# Translate
translation = translator.translate("hello world")
print(translation)  # "hola mundo"
```

## Model Configuration

The default model uses these hyperparameters:

- `d_model`: 128 (embedding dimension)
- `n_heads`: 4 (attention heads)
- `n_layers`: 2 (transformer layers)
- `d_ff`: 256 (feed-forward dimension)
- `max_seq_len`: 50 (maximum sequence length)
- `dropout`: 0.1 (dropout rate during training)

## Training Data

The demo includes 50 common English-Spanish phrase pairs for training. In practice, you would use a larger parallel corpus like:
- UN Parallel Corpus
- OpenSubtitles
- Multi30k
- WMT datasets

## Extending the Model

### Adding More Training Data

```python
# Add more training pairs
NEW_PAIRS = [
    ("I am happy", "Estoy feliz"),
    ("The book is red", "El libro es rojo"),
    # ... more pairs
]

# Extend the SAMPLE_PAIRS in train.py
SAMPLE_PAIRS.extend(NEW_PAIRS)
```

### Improving Model Performance

1. **Increase Model Size**: 
   ```python
   model = TranslationTransformer(
       d_model=512,    # Larger embeddings
       n_heads=8,      # More attention heads
       n_layers=6,     # Deeper network
       d_ff=2048       # Wider feed-forward
   )
   ```

2. **Better Training**:
   - Use learning rate scheduling
   - Implement beam search for decoding
   - Add label smoothing
   - Use larger batch sizes

3. **Data Augmentation**:
   - Back-translation
   - Paraphrasing
   - Noise injection

## Limitations

This is a demonstration model with limitations:
- Small vocabulary (built from sample data)
- Limited training data (50 sentence pairs)
- Simple tokenization (space-based)
- No subword tokenization (BPE/SentencePiece)
- Greedy decoding (no beam search)

## Future Improvements

- [ ] Implement beam search decoding
- [ ] Add subword tokenization
- [ ] Support for multiple languages
- [ ] Model checkpointing and saving
- [ ] BLEU score evaluation
- [ ] Attention visualization
- [ ] Larger pre-trained models

## Technical Details

The implementation showcases:
- Custom transformer architecture from scratch
- Gradient flow through attention mechanisms
- Positional encoding implementation
- Teacher forcing during training
- Autoregressive generation
- Padding and masking strategies

This demonstrates that our neural architecture framework can handle complex sequence-to-sequence tasks!