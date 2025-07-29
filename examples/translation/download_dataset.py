"""Download and prepare English-Spanish translation dataset."""

import os
import urllib.request
import zipfile
import tarfile
import json
from typing import List, Tuple
import re


def download_file(url: str, filename: str):
    """Download a file with progress reporting."""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"Downloading: {percent:.1f}%", end='\r')
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename, reporthook=download_progress)
    print(f"\n✅ Downloaded {filename}")


def download_tatoeba_dataset():
    """Download Tatoeba English-Spanish sentence pairs."""
    url = "http://www.manythings.org/anki/spa-eng.zip"
    filename = "spa-eng.zip"
    
    if not os.path.exists(filename):
        download_file(url, filename)
    
    # Extract
    print("Extracting dataset...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Read sentences
    pairs = []
    with open("spa.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                english = parts[0]
                spanish = parts[1]
                pairs.append((english, spanish))
    
    print(f"✅ Loaded {len(pairs)} sentence pairs")
    return pairs


def download_opus_dataset():
    """Download OPUS OpenSubtitles dataset (smaller sample)."""
    # Using a preprocessed sample for easier handling
    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-es.txt.zip"
    filename = "opus_en_es.zip"
    
    if not os.path.exists(filename):
        print("Downloading OPUS OpenSubtitles dataset (this may take a while)...")
        download_file(url, filename)
    
    # Extract
    print("Extracting dataset...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("opus_data")
    
    # Read parallel sentences
    pairs = []
    en_file = "opus_data/OpenSubtitles.en-es.en"
    es_file = "opus_data/OpenSubtitles.en-es.es"
    
    if os.path.exists(en_file) and os.path.exists(es_file):
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(es_file, 'r', encoding='utf-8') as f_es:
            
            for en_line, es_line in zip(f_en, f_es):
                en_line = en_line.strip()
                es_line = es_line.strip()
                if en_line and es_line:
                    pairs.append((en_line, es_line))
                    if len(pairs) >= 100000:  # Limit to 100k for demo
                        break
    
    print(f"✅ Loaded {len(pairs)} sentence pairs from OPUS")
    return pairs


def clean_sentence(sentence: str) -> str:
    """Clean and normalize a sentence."""
    # Convert to lowercase
    sentence = sentence.lower().strip()
    
    # Remove HTML tags
    sentence = re.sub(r'<[^>]+>', '', sentence)
    
    # Fix spacing around punctuation
    sentence = re.sub(r'\s+([.!?,;:])', r'\1', sentence)
    sentence = re.sub(r'([¿¡])\s+', r'\1', sentence)
    
    # Remove extra spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Remove very long sentences (likely errors)
    if len(sentence.split()) > 50:
        return ""
    
    return sentence


def prepare_dataset(pairs: List[Tuple[str, str]], 
                   min_length: int = 3,
                   max_length: int = 30) -> List[Tuple[str, str]]:
    """Clean and filter sentence pairs."""
    cleaned_pairs = []
    
    for en, es in pairs:
        # Clean sentences
        en_clean = clean_sentence(en)
        es_clean = clean_sentence(es)
        
        # Check length constraints
        en_words = en_clean.split()
        es_words = es_clean.split()
        
        if (min_length <= len(en_words) <= max_length and
            min_length <= len(es_words) <= max_length and
            en_clean and es_clean):
            cleaned_pairs.append((en_clean, es_clean))
    
    return cleaned_pairs


def save_dataset(pairs: List[Tuple[str, str]], filename: str):
    """Save dataset to JSON file."""
    data = {
        "pairs": pairs,
        "metadata": {
            "total_pairs": len(pairs),
            "languages": ["english", "spanish"]
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(pairs)} pairs to {filename}")


def create_train_val_test_split(pairs: List[Tuple[str, str]], 
                               train_ratio: float = 0.8,
                               val_ratio: float = 0.1) -> Tuple[List, List, List]:
    """Split dataset into train/validation/test sets."""
    import random
    random.seed(42)
    
    # Shuffle pairs
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    # Calculate split sizes
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train:n_train + n_val]
    test_pairs = shuffled[n_train + n_val:]
    
    return train_pairs, val_pairs, test_pairs


def main():
    """Download and prepare translation datasets."""
    print("🌐 Translation Dataset Downloader")
    print("=" * 50)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    
    # Download Tatoeba dataset (smaller, cleaner)
    print("\n1️⃣ Downloading Tatoeba dataset...")
    tatoeba_pairs = download_tatoeba_dataset()
    
    # Clean and filter
    print("\n🧹 Cleaning dataset...")
    cleaned_pairs = prepare_dataset(tatoeba_pairs, min_length=3, max_length=25)
    print(f"✅ Cleaned dataset: {len(cleaned_pairs)} pairs")
    
    # Create splits
    print("\n📊 Creating train/val/test splits...")
    train_pairs, val_pairs, test_pairs = create_train_val_test_split(cleaned_pairs)
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val: {len(val_pairs)} pairs")
    print(f"  Test: {len(test_pairs)} pairs")
    
    # Save datasets
    save_dataset(train_pairs, "train.json")
    save_dataset(val_pairs, "val.json")
    save_dataset(test_pairs, "test.json")
    save_dataset(cleaned_pairs, "all_pairs.json")
    
    # Show some examples
    print("\n📝 Example pairs:")
    for i, (en, es) in enumerate(train_pairs[:5]):
        print(f"{i+1}. EN: {en}")
        print(f"   ES: {es}")
    
    # Try to download larger dataset (optional)
    print("\n2️⃣ Optionally download larger OPUS dataset? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        try:
            opus_pairs = download_opus_dataset()
            opus_cleaned = prepare_dataset(opus_pairs[:50000])  # Use first 50k
            save_dataset(opus_cleaned, "opus_subset.json")
        except Exception as e:
            print(f"⚠️  Could not download OPUS dataset: {e}")
    
    print("\n✅ Dataset preparation complete!")
    print("\n📁 Created files:")
    print("  - data/train.json (training set)")
    print("  - data/val.json (validation set)")
    print("  - data/test.json (test set)")
    print("  - data/all_pairs.json (all cleaned pairs)")
    
    # Go back to translation directory
    os.chdir("..")


if __name__ == "__main__":
    main()