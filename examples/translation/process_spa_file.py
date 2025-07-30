"""Process spa.txt file from Tatoeba dataset."""

import os
import json
import random
from typing import List, Tuple

def process_spa_file(filename="spa.txt"):
    """Process the spa.txt file from Tatoeba."""
    print(f"📖 Processing {filename}...")
    print("=" * 60)
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return []
    
    pairs = []
    problematic_lines = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i} lines... Found {len(pairs)} good pairs")
            
            line = line.strip()
            if not line:
                continue
            
            # Tatoeba format: English[TAB]Spanish[TAB]Attribution
            parts = line.split('\t')
            
            if len(parts) >= 2:
                eng = parts[0].strip()
                spa = parts[1].strip()
                
                # Clean and validate
                if eng and spa:
                    # Basic quality checks
                    eng_words = len(eng.split())
                    spa_words = len(spa.split())
                    
                    if (2 <= eng_words <= 15 and 
                        2 <= spa_words <= 20 and
                        len(eng) >= 3 and len(spa) >= 3 and
                        eng[0].isalpha() and spa[0].isalpha()):
                        
                        # Convert to lowercase for consistency
                        pairs.append((eng.lower(), spa.lower()))
                    else:
                        problematic_lines += 1
            else:
                problematic_lines += 1
                if i < 10:  # Show first few problematic lines
                    print(f"  Problem with line {i+1}: {repr(line[:100])}")
    
    print(f"\n✅ Successfully processed {len(pairs)} sentence pairs")
    print(f"⚠️  Skipped {problematic_lines} problematic lines")
    
    return pairs

def analyze_dataset(pairs: List[Tuple[str, str]]):
    """Analyze the dataset for common patterns."""
    print("\n📊 Dataset Analysis:")
    print("-" * 40)
    
    # Length statistics
    eng_lengths = [len(eng.split()) for eng, _ in pairs]
    spa_lengths = [len(spa.split()) for _, spa in pairs]
    
    print(f"English sentences:")
    print(f"  Average length: {sum(eng_lengths)/len(eng_lengths):.1f} words")
    print(f"  Min/Max: {min(eng_lengths)}/{max(eng_lengths)} words")
    
    print(f"\nSpanish sentences:")
    print(f"  Average length: {sum(spa_lengths)/len(spa_lengths):.1f} words")
    print(f"  Min/Max: {min(spa_lengths)}/{max(spa_lengths)} words")
    
    # Common starters
    print("\n🗣️ Most common conversation starters:")
    starters = {}
    for eng, spa in pairs[:5000]:
        first_word = eng.split()[0] if eng.split() else ''
        if first_word:
            starters[first_word] = starters.get(first_word, 0) + 1
    
    for word, count in sorted(starters.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  '{word}': {count} times")
    
    # Check for key conversational phrases
    print("\n✅ Key conversational phrases found:")
    key_phrases = {
        "hello": "hola",
        "thank you": "gracias",
        "how are you": "cómo estás",
        "goodbye": "adiós",
        "yes": "sí",
        "no": "no",
        "please": "por favor",
        "i love you": "te amo",
    }
    
    for eng_phrase, expected_spa in key_phrases.items():
        found = False
        for eng, spa in pairs:
            if eng_phrase in eng:
                print(f"  '{eng_phrase}' → '{spa}' {'✅' if expected_spa in spa else '❓'}")
                found = True
                break
        if not found:
            print(f"  '{eng_phrase}' → NOT FOUND ❌")

def save_datasets(pairs: List[Tuple[str, str]]):
    """Save train/val/test splits."""
    print("\n💾 Saving datasets...")
    
    # Shuffle for randomness
    random.shuffle(pairs)
    
    # Remove exact duplicates
    pairs = list(set(pairs))
    print(f"Unique pairs after deduplication: {len(pairs)}")
    
    # Split
    n_total = len(pairs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save
    datasets = {
        "train_tatoeba.json": {
            "pairs": train_pairs,
            "source": "Tatoeba",
            "type": "conversational",
            "language_pair": "en-es"
        },
        "val_tatoeba.json": {
            "pairs": val_pairs,
            "source": "Tatoeba",
            "type": "conversational",
            "language_pair": "en-es"
        },
        "test_tatoeba.json": {
            "pairs": test_pairs,
            "source": "Tatoeba",
            "type": "conversational",
            "language_pair": "en-es"
        }
    }
    
    for filename, data in datasets.items():
        filepath = os.path.join("data", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {filename} with {len(data['pairs'])} pairs")

def show_samples(pairs: List[Tuple[str, str]], n: int = 30):
    """Show sample pairs."""
    print(f"\n📝 Sample sentence pairs:")
    print("-" * 70)
    
    # Show some short ones first
    short_pairs = [(e, s) for e, s in pairs if len(e.split()) <= 5][:10]
    print("Short conversational examples:")
    for eng, spa in short_pairs:
        print(f"  {eng:30} → {spa}")
    
    print("\nRandom examples:")
    random_sample = random.sample(pairs, min(20, len(pairs)))
    for eng, spa in random_sample:
        if len(eng) <= 50:  # Don't show very long ones
            print(f"  {eng:30} → {spa}")

def main():
    """Main function."""
    print("🌐 Processing Tatoeba Spanish-English Dataset")
    print("=" * 60)
    
    # Process the spa.txt file
    pairs = process_spa_file("spa.txt")
    
    if not pairs:
        print("\n❌ No valid pairs found!")
        print("Please make sure spa.txt is in the correct format:")
        print("  English[TAB]Spanish[TAB]Attribution")
        return
    
    # Analyze the dataset
    analyze_dataset(pairs)
    
    # Show samples
    show_samples(pairs)
    
    # Save datasets
    save_datasets(pairs)
    
    print("\n✅ Tatoeba dataset successfully processed!")
    print(f"📊 Total usable pairs: {len(pairs)}")
    print("\n💡 Next steps:")
    print("1. Update train_conversational.py to use 'train_tatoeba.json'")
    print("2. Run: python train_conversational.py")
    print("\nThis dataset should give you proper conversational translations!")

if __name__ == "__main__":
    main()