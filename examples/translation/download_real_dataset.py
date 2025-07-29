"""Download real English-Spanish parallel corpus."""

import os
import requests
import gzip
import json
from typing import List, Tuple
import re
from tqdm import tqdm


def download_file_with_progress(url: str, filename: str):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                pbar.update(len(data))


def download_un_corpus():
    """Download UN Parallel Corpus (smaller subset)."""
    print("📥 Downloading UN Parallel Corpus subset...")
    
    # Using a preprocessed subset from OPUS
    base_url = "https://opus.nlpl.eu/download.php?f=UN/v20090831/moses/"
    
    files = {
        "en-es.txt.zip": "UN.en-es.zip"
    }
    
    for remote, local in files.items():
        if not os.path.exists(local):
            try:
                url = base_url + remote
                download_file_with_progress(url, local)
            except Exception as e:
                print(f"⚠️  Could not download {remote}: {e}")
    
    return []  # Will extract later if successful


def download_tatoeba_full():
    """Download full Tatoeba dataset."""
    print("\n📥 Downloading Tatoeba dataset...")
    
    # Direct link to sentences file
    url = "https://downloads.tatoeba.org/exports/sentences.csv"
    links_url = "https://downloads.tatoeba.org/exports/links.csv"
    
    sentences_file = "sentences_tatoeba.csv"
    links_file = "links_tatoeba.csv"
    
    # Check if we need to download
    if not os.path.exists(sentences_file):
        print("This is a large file (>100MB), downloading...")
        try:
            download_file_with_progress(url, sentences_file)
        except:
            print("Failed to download sentences")
            return []
    
    if not os.path.exists(links_file):
        try:
            download_file_with_progress(links_url, links_file)
        except:
            print("Failed to download links")
            return []
    
    # Process to extract English-Spanish pairs
    print("Processing Tatoeba data...")
    
    # Load sentences
    sentences = {}
    with open(sentences_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading sentences"):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                sent_id, lang, text = parts[0], parts[1], parts[2]
                if lang in ['eng', 'spa']:
                    sentences[sent_id] = (lang, text)
    
    # Load links and create pairs
    pairs = []
    with open(links_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Creating pairs"):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                id1, id2 = parts[0], parts[1]
                if id1 in sentences and id2 in sentences:
                    lang1, text1 = sentences[id1]
                    lang2, text2 = sentences[id2]
                    
                    if lang1 == 'eng' and lang2 == 'spa':
                        pairs.append((text1, text2))
                    elif lang1 == 'spa' and lang2 == 'eng':
                        pairs.append((text2, text1))
    
    print(f"✅ Extracted {len(pairs)} English-Spanish pairs from Tatoeba")
    return pairs


def download_opensubtitles_sample():
    """Download OpenSubtitles sample data."""
    print("\n📥 Downloading OpenSubtitles sample...")
    
    # Using a smaller, preprocessed sample
    url = "https://raw.githubusercontent.com/facebookresearch/MIXER/main/data/test/opensubs.test.en"
    es_url = "https://raw.githubusercontent.com/facebookresearch/MIXER/main/data/test/opensubs.test.es"
    
    en_file = "opensubs_sample.en"
    es_file = "opensubs_sample.es"
    
    pairs = []
    
    try:
        # Download English
        if not os.path.exists(en_file):
            response = requests.get(url)
            with open(en_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        # Download Spanish
        if not os.path.exists(es_file):
            response = requests.get(es_url)
            with open(es_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        # Read pairs
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(es_file, 'r', encoding='utf-8') as f_es:
            
            for en_line, es_line in zip(f_en, f_es):
                en_text = en_line.strip()
                es_text = es_line.strip()
                if en_text and es_text:
                    pairs.append((en_text, es_text))
        
        print(f"✅ Loaded {len(pairs)} pairs from OpenSubtitles sample")
    except Exception as e:
        print(f"⚠️  Could not download OpenSubtitles: {e}")
    
    return pairs


def download_news_commentary():
    """Download News Commentary dataset."""
    print("\n📥 Downloading News Commentary dataset...")
    
    url = "http://data.statmt.org/news-commentary/v16/training/news-commentary-v16.en-es.tsv.gz"
    gz_file = "news-commentary-v16.en-es.tsv.gz"
    
    pairs = []
    
    try:
        if not os.path.exists(gz_file):
            download_file_with_progress(url, gz_file)
        
        # Extract and read
        with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing News Commentary"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    en_text = parts[0].strip()
                    es_text = parts[1].strip()
                    if en_text and es_text and len(en_text.split()) <= 50 and len(es_text.split()) <= 50:
                        pairs.append((en_text, es_text))
        
        print(f"✅ Loaded {len(pairs)} pairs from News Commentary")
    except Exception as e:
        print(f"⚠️  Could not process News Commentary: {e}")
    
    return pairs


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Fix quotes and punctuation
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very long sentences
    if len(text.split()) > 50:
        return ""
    
    return text.strip()


def combine_and_clean_datasets(all_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Combine all datasets and clean."""
    print("\n🧹 Cleaning and filtering dataset...")
    
    cleaned_pairs = []
    seen = set()
    
    for en, es in tqdm(all_pairs, desc="Cleaning"):
        # Clean texts
        en_clean = clean_text(en.lower())
        es_clean = clean_text(es.lower())
        
        # Filter by length
        en_words = en_clean.split()
        es_words = es_clean.split()
        
        if (3 <= len(en_words) <= 30 and 
            3 <= len(es_words) <= 30 and
            en_clean and es_clean):
            
            # Avoid duplicates
            pair_key = (en_clean, es_clean)
            if pair_key not in seen:
                seen.add(pair_key)
                cleaned_pairs.append((en_clean, es_clean))
    
    return cleaned_pairs


def save_datasets(pairs: List[Tuple[str, str]]):
    """Save dataset splits."""
    import random
    random.seed(42)
    
    # Shuffle
    random.shuffle(pairs)
    
    # Split
    n_total = len(pairs)
    n_train = int(n_total * 0.9)
    n_val = int(n_total * 0.05)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    # Save
    os.makedirs("data", exist_ok=True)
    
    datasets = {
        "data/train_large.json": train_pairs,
        "data/val_large.json": val_pairs,
        "data/test_large.json": test_pairs,
        "data/all_large.json": pairs
    }
    
    for filename, data in datasets.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "pairs": data,
                "metadata": {
                    "total_pairs": len(data),
                    "languages": ["english", "spanish"]
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(data)} pairs to {filename}")


def main():
    """Download and prepare real translation datasets."""
    print("🌐 Real Translation Dataset Downloader")
    print("=" * 50)
    
    all_pairs = []
    
    # Try to download from multiple sources
    
    # 1. OpenSubtitles sample (quick)
    pairs = download_opensubtitles_sample()
    all_pairs.extend(pairs)
    
    # 2. News Commentary (medium size)
    pairs = download_news_commentary()
    all_pairs.extend(pairs)
    
    # 3. Tatoeba (large but may be slow)
    if input("\nDownload full Tatoeba dataset? (large, ~200MB) [y/N]: ").lower() == 'y':
        pairs = download_tatoeba_full()
        all_pairs.extend(pairs)
    
    # Clean and combine
    cleaned_pairs = combine_and_clean_datasets(all_pairs)
    
    print(f"\n📊 Total cleaned pairs: {len(cleaned_pairs)}")
    
    # Save datasets
    save_datasets(cleaned_pairs)
    
    # Show examples
    print("\n📝 Example pairs:")
    for i, (en, es) in enumerate(cleaned_pairs[:10]):
        print(f"{i+1}. EN: {en}")
        print(f"   ES: {es}")
    
    print("\n✅ Dataset preparation complete!")
    print("\nNext steps:")
    print("1. Update train.py to use 'data/train_large.json'")
    print("2. Run training with more epochs (100-200)")
    print("3. Test the improved model")


if __name__ == "__main__":
    # Install required package
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm...")
        os.system("pip install tqdm")
        from tqdm import tqdm
    
    main()