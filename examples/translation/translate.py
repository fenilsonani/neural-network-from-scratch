"""Interactive translation script for English to Spanish."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import json
from typing import Optional
from neural_arch.core import Tensor
from vocabulary import Vocabulary
from model import TranslationTransformer


class Translator:
    """Interactive translator class."""
    
    def __init__(self, model: TranslationTransformer, 
                 src_vocab: Vocabulary, 
                 tgt_vocab: Vocabulary):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def translate(self, text: str, temperature: float = 1.0) -> str:
        """Translate English text to Spanish."""
        # Encode source
        src_indices = self.src_vocab.encode(text, max_length=50)
        src_tensor = Tensor(np.array([src_indices]), requires_grad=False)
        
        # Generate translation
        output_indices = self.model.generate(
            src_tensor,
            max_length=50,
            sos_idx=self.tgt_vocab.word2idx[self.tgt_vocab.sos_token],
            eos_idx=self.tgt_vocab.word2idx[self.tgt_vocab.eos_token],
            temperature=temperature
        )
        
        # Decode
        translation = self.tgt_vocab.decode(output_indices, remove_special=True)
        return translation
    
    def interactive_mode(self):
        """Run interactive translation mode."""
        print("\n🌐 English to Spanish Translator")
        print("=" * 50)
        print("Type English text to translate (or 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                # Get input
                text = input("\n🇬🇧 English: ").strip()
                
                # Check for quit
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not text:
                    continue
                
                # Translate
                translation = self.translate(text)
                print(f"🇪🇸 Spanish: {translation}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def create_pretrained_model():
    """Create a model with some pretrained weights (for demo)."""
    # In a real scenario, you would load saved weights
    # For now, we'll create a model and train it briefly
    
    from train import load_dataset, create_dataset, train_epoch
    from neural_arch.optim import Adam
    
    print("🔧 Preparing model...")
    
    # Create vocabularies
    src_vocab = Vocabulary("english")
    tgt_vocab = Vocabulary("spanish")
    
    # Load training data
    try:
        train_pairs = load_dataset("data/train.json")
        print(f"✅ Loaded {len(train_pairs)} training pairs")
    except FileNotFoundError:
        print("❌ Dataset not found. Please run create_dataset.py first!")
        return None, None, None
    
    # Create dataset
    src_data, tgt_data = create_dataset(train_pairs[:200], src_vocab, tgt_vocab, max_length=20)
    
    # Create model
    model = TranslationTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.0  # No dropout for inference
    )
    
    # Quick training
    print("🏃 Quick training (this may take a moment)...")
    optimizer = Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):  # Quick training
        loss = train_epoch(
            model, optimizer, src_data, tgt_data,
            batch_size=8,
            pad_idx=src_vocab.word2idx[src_vocab.pad_token]
        )
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/50 - Loss: {loss:.4f}")
    
    return model, src_vocab, tgt_vocab


def main():
    """Main function."""
    # Check if vocabularies exist
    if os.path.exists("vocab_en_final.json") and os.path.exists("vocab_es_final.json"):
        print("📚 Loading vocabularies...")
        src_vocab = Vocabulary.load("vocab_en_final.json")
        tgt_vocab = Vocabulary.load("vocab_es_final.json")
        
        # Create model
        model = TranslationTransformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            dropout=0.0
        )
        print("⚠️  Note: Using untrained model. Run train.py first for better results.")
    else:
        print("📚 Creating and training model...")
        model, src_vocab, tgt_vocab = create_pretrained_model()
    
    # Create translator
    translator = Translator(model, src_vocab, tgt_vocab)
    
    # Show some examples
    print("\n📝 Example translations:")
    examples = ["hello", "how are you", "thank you", "good morning", "i love you"]
    for text in examples:
        translation = translator.translate(text)
        print(f"  {text} → {translation}")
    
    # Run interactive mode
    translator.interactive_mode()


if __name__ == "__main__":
    main()