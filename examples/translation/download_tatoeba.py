"""Download Tatoeba dataset - high quality sentence pairs."""

import os
import requests
import csv
import json
import random
from typing import List, Tuple

def download_tatoeba_dataset():
    """Download Tatoeba Spanish-English dataset."""
    print("📥 Downloading Tatoeba dataset...")
    
    # Tatoeba direct download link for Spanish-English
    url = "https://www.manythings.org/anki/spa-eng.zip"
    
    os.makedirs("data", exist_ok=True)
    
    # Download
    print(f"Downloading from {url}...")
    response = requests.get(url)
    
    with open("data/spa-eng.zip", 'wb') as f:
        f.write(response.content)
    print("✅ Downloaded!")
    
    # Extract
    import zipfile
    with zipfile.ZipFile("data/spa-eng.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    print("✅ Extracted!")
    
    # Process the file
    pairs = []
    with open("data/spa.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng = parts[0].strip()
                spa = parts[1].strip()
                
                # Clean up
                if eng and spa:
                    # Remove any attribution info
                    if '\t' in spa:
                        spa = spa.split('\t')[0]
                    
                    # Quality checks
                    if (3 <= len(eng.split()) <= 15 and 
                        3 <= len(spa.split()) <= 20 and
                        eng[0].isalpha() and spa[0].isalpha()):
                        pairs.append((eng.lower(), spa.lower()))
    
    print(f"✅ Processed {len(pairs)} sentence pairs")
    return pairs

def download_opus_tatoeba():
    """Alternative: Download from OPUS."""
    print("\n📥 Trying OPUS Tatoeba dataset...")
    
    url = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-es.txt.zip"
    
    try:
        response = requests.get(url, timeout=30)
        with open("data/tatoeba-opus.zip", 'wb') as f:
            f.write(response.content)
        
        import zipfile
        with zipfile.ZipFile("data/tatoeba-opus.zip", 'r') as zip_ref:
            zip_ref.extractall("data/tatoeba-opus")
        
        # Find and process files
        pairs = []
        for root, dirs, files in os.walk("data/tatoeba-opus"):
            for file in files:
                if file.endswith('.en-es'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if ' ||| ' in line:
                                en, es = line.strip().split(' ||| ', 1)
                                if 3 <= len(en.split()) <= 15:
                                    pairs.append((en.lower(), es.lower()))
        
        return pairs
    except Exception as e:
        print(f"❌ OPUS download failed: {e}")
        return []

def create_conversational_supplements():
    """Create additional conversational examples."""
    print("\n🔨 Adding conversational supplements...")
    
    # Essential conversational patterns
    supplements = [
        # Greetings
        ("hello", "hola"),
        ("hi", "hola"),
        ("hey", "hey"),
        ("good morning", "buenos días"),
        ("good afternoon", "buenas tardes"),
        ("good evening", "buenas noches"),
        ("good night", "buenas noches"),
        ("goodbye", "adiós"),
        ("bye", "adiós"),
        ("see you later", "hasta luego"),
        ("see you tomorrow", "hasta mañana"),
        ("see you soon", "hasta pronto"),
        
        # Common phrases
        ("how are you", "cómo estás"),
        ("how are you doing", "cómo te va"),
        ("i'm fine", "estoy bien"),
        ("i'm good", "estoy bien"),
        ("and you", "y tú"),
        ("thank you", "gracias"),
        ("thanks", "gracias"),
        ("you're welcome", "de nada"),
        ("please", "por favor"),
        ("excuse me", "disculpa"),
        ("sorry", "lo siento"),
        ("no problem", "no hay problema"),
        
        # Questions
        ("what's your name", "cómo te llamas"),
        ("my name is", "me llamo"),
        ("where are you from", "de dónde eres"),
        ("i'm from", "soy de"),
        ("where is", "dónde está"),
        ("how much", "cuánto cuesta"),
        ("what time is it", "qué hora es"),
        
        # Common responses
        ("yes", "sí"),
        ("no", "no"),
        ("maybe", "quizás"),
        ("i don't know", "no sé"),
        ("i understand", "entiendo"),
        ("i don't understand", "no entiendo"),
        
        # Daily conversation
        ("i want", "quiero"),
        ("i need", "necesito"),
        ("i like", "me gusta"),
        ("i love", "me encanta"),
        ("can you help me", "puedes ayudarme"),
        ("let's go", "vamos"),
        ("come here", "ven aquí"),
        ("wait", "espera"),
    ]
    
    # Expand with variations
    expanded = []
    for en, es in supplements:
        # Original
        expanded.append((en, es))
        # With punctuation
        expanded.append((en + ".", es + "."))
        expanded.append((en + "?", es + "?"))
        expanded.append((en + "!", es + "!"))
        # Capitalized
        expanded.append((en.capitalize(), es.capitalize()))
        # Some variations
        if "you" in en:
            expanded.append((en.replace("you", "you guys"), es))
    
    return expanded

def save_training_data(pairs: List[Tuple[str, str]]):
    """Save the training data."""
    print("\n💾 Saving training data...")
    
    # Shuffle
    random.shuffle(pairs)
    
    # Remove duplicates
    pairs = list(set(pairs))
    
    # Split
    n_total = len(pairs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    # Save
    datasets = {
        "train_conversational.json": {
            "pairs": train_pairs,
            "source": "Tatoeba + Conversational",
            "type": "conversational"
        },
        "val_conversational.json": {
            "pairs": val_pairs,
            "source": "Tatoeba + Conversational",
            "type": "conversational"
        },
        "test_conversational.json": {
            "pairs": test_pairs,
            "source": "Tatoeba + Conversational",
            "type": "conversational"
        }
    }
    
    for filename, data in datasets.items():
        with open(f"data/{filename}", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {filename} with {len(data['pairs'])} pairs")

def main():
    """Main function."""
    print("🌐 Downloading Conversational Translation Dataset")
    print("=" * 60)
    
    all_pairs = []
    
    # Try multiple sources
    try:
        # 1. Try Tatoeba from manythings.org
        tatoeba_pairs = download_tatoeba_dataset()
        all_pairs.extend(tatoeba_pairs)
    except Exception as e:
        print(f"❌ Tatoeba download failed: {e}")
    
    # 2. Add conversational supplements
    supplements = create_conversational_supplements()
    all_pairs.extend(supplements * 10)  # Repeat to increase weight
    
    if len(all_pairs) < 1000:
        print("\n❌ Not enough data collected!")
        print("Creating synthetic conversational data...")
        
        # Create more synthetic examples
        templates = [
            ("i want {}", "quiero {}"),
            ("i need {}", "necesito {}"),
            ("where is the {}", "dónde está el {}"),
            ("can i have {}", "puedo tener {}"),
            ("do you have {}", "tienes {}"),
            ("i like {}", "me gusta {}"),
        ]
        
        nouns = [
            ("water", "agua"),
            ("food", "comida"),
            ("coffee", "café"),
            ("beer", "cerveza"),
            ("bathroom", "baño"),
            ("hotel", "hotel"),
            ("restaurant", "restaurante"),
            ("help", "ayuda"),
        ]
        
        for template_en, template_es in templates:
            for noun_en, noun_es in nouns:
                all_pairs.append((
                    template_en.format(noun_en),
                    template_es.format(noun_es)
                ))
    
    # Save the data
    save_training_data(all_pairs)
    
    # Show statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"Total pairs: {len(set(all_pairs))}")
    
    print("\n📝 Sample pairs:")
    for i in range(min(20, len(all_pairs))):
        en, es = all_pairs[i]
        print(f"  {en:30} → {es}")
    
    print("\n✅ Dataset ready!")
    print("💡 Now you can train with: python train_conversational.py")

if __name__ == "__main__":
    main()