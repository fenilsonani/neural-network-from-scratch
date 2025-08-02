"""Download conversational datasets for better translation quality."""

import requests
import json
import gzip
import os
from typing import List, Tuple

def download_tatoeba_sentences():
    """Download Tatoeba sentences dataset (conversational)."""
    print("📥 Downloading Tatoeba conversational dataset...")
    
    # Tatoeba provides sentence pairs in many languages
    url = "https://downloads.tatoeba.org/exports/sentences_links.tar.bz2"
    
    # For now, let's create a simple conversational dataset manually
    conversational_pairs = [
        # Greetings
        ("hello", "hola"),
        ("hi", "hola"),
        ("good morning", "buenos días"),
        ("good afternoon", "buenas tardes"),
        ("good evening", "buenas noches"),
        ("good night", "buenas noches"),
        ("goodbye", "adiós"),
        ("bye", "adiós"),
        ("see you later", "hasta luego"),
        ("see you tomorrow", "hasta mañana"),
        ("see you soon", "hasta pronto"),
        
        # Basic phrases
        ("yes", "sí"),
        ("no", "no"),
        ("please", "por favor"),
        ("thank you", "gracias"),
        ("thanks", "gracias"),
        ("you're welcome", "de nada"),
        ("excuse me", "disculpe"),
        ("sorry", "lo siento"),
        ("i'm sorry", "lo siento"),
        
        # Questions
        ("how are you", "cómo estás"),
        ("how are you doing", "cómo te va"),
        ("what's your name", "cómo te llamas"),
        ("my name is john", "me llamo juan"),
        ("where are you from", "de dónde eres"),
        ("i'm from america", "soy de américa"),
        ("do you speak english", "hablas inglés"),
        ("i don't understand", "no entiendo"),
        ("can you repeat", "puedes repetir"),
        
        # Common expressions
        ("i love you", "te amo"),
        ("i like you", "me gustas"),
        ("i miss you", "te extraño"),
        ("i need help", "necesito ayuda"),
        ("where is the bathroom", "dónde está el baño"),
        ("how much does it cost", "cuánto cuesta"),
        ("what time is it", "qué hora es"),
        ("i'm hungry", "tengo hambre"),
        ("i'm thirsty", "tengo sed"),
        ("i'm tired", "estoy cansado"),
        
        # Numbers and basics
        ("one", "uno"),
        ("two", "dos"),
        ("three", "tres"),
        ("water", "agua"),
        ("food", "comida"),
        ("house", "casa"),
        ("car", "coche"),
        ("friend", "amigo"),
        ("family", "familia"),
        
        # Simple sentences
        ("i want water", "quiero agua"),
        ("i need food", "necesito comida"),
        ("let's go", "vamos"),
        ("come here", "ven aquí"),
        ("wait please", "espera por favor"),
        ("help me", "ayúdame"),
        ("i don't know", "no sé"),
        ("i understand", "entiendo"),
        ("do you understand", "entiendes"),
        ("speak slowly please", "habla despacio por favor"),
    ]
    
    # Expand with variations
    expanded_pairs = []
    for en, es in conversational_pairs:
        expanded_pairs.append((en, es))
        expanded_pairs.append((en + ".", es + "."))
        expanded_pairs.append((en + "?", es + "?"))
        expanded_pairs.append((en.capitalize(), es.capitalize()))
        expanded_pairs.append((en.upper(), es.upper()))
    
    # Add more natural variations
    additional_pairs = [
        ("Hello, how are you?", "Hola, ¿cómo estás?"),
        ("I'm fine, thank you.", "Estoy bien, gracias."),
        ("Nice to meet you.", "Mucho gusto."),
        ("What's up?", "¿Qué tal?"),
        ("Nothing much.", "No mucho."),
        ("See you!", "¡Nos vemos!"),
        ("Take care.", "Cuídate."),
        ("Have a good day.", "Que tengas un buen día."),
        ("Good luck!", "¡Buena suerte!"),
        ("Congratulations!", "¡Felicidades!"),
        ("Happy birthday!", "¡Feliz cumpleaños!"),
        ("Merry Christmas!", "¡Feliz Navidad!"),
        ("Happy New Year!", "¡Feliz Año Nuevo!"),
        ("I love Spanish.", "Me encanta el español."),
        ("Spain is beautiful.", "España es hermosa."),
        ("The weather is nice.", "El clima está agradable."),
        ("It's raining.", "Está lloviendo."),
        ("It's hot today.", "Hace calor hoy."),
        ("It's cold.", "Hace frío."),
        ("I'm learning Spanish.", "Estoy aprendiendo español."),
        ("Do you like coffee?", "¿Te gusta el café?"),
        ("Yes, I love coffee.", "Sí, me encanta el café."),
        ("No, I prefer tea.", "No, prefiero el té."),
        ("What do you want to eat?", "¿Qué quieres comer?"),
        ("I want pizza.", "Quiero pizza."),
        ("Let's eat!", "¡Vamos a comer!"),
        ("The food is delicious.", "La comida está deliciosa."),
        ("Can I have the bill?", "¿Me trae la cuenta?"),
        ("Keep the change.", "Quédese con el cambio."),
    ]
    
    expanded_pairs.extend(additional_pairs)
    
    return expanded_pairs


def create_mixed_dataset():
    """Create a mixed dataset with conversational and some formal pairs."""
    print("🔨 Creating mixed conversational dataset...")
    
    # Get conversational pairs
    conversational = download_tatoeba_sentences()
    
    # Try to load some existing data if available
    formal_pairs = []
    if os.path.exists("data/train.json"):
        with open("data/train.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Take only short, simple sentences from formal data
            for en, es in data['pairs'][:1000]:
                if len(en.split()) <= 8 and len(es.split()) <= 8:
                    formal_pairs.append((en, es))
    
    # Combine datasets - 80% conversational, 20% formal
    all_pairs = conversational * 3  # Repeat conversational data
    all_pairs.extend(formal_pairs)
    
    # Shuffle
    import random
    random.shuffle(all_pairs)
    
    # Split into train/val/test
    n_total = len(all_pairs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train+n_val]
    test_pairs = all_pairs[n_train+n_val:]
    
    # Save datasets
    os.makedirs("data", exist_ok=True)
    
    datasets = {
        "train_conversational.json": {"pairs": train_pairs, "type": "conversational"},
        "val_conversational.json": {"pairs": val_pairs, "type": "conversational"},
        "test_conversational.json": {"pairs": test_pairs, "type": "conversational"}
    }
    
    for filename, data in datasets.items():
        with open(f"data/{filename}", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {filename} with {len(data['pairs'])} pairs")
    
    return train_pairs, val_pairs, test_pairs


def main():
    """Main function."""
    print("🌐 Creating Conversational Translation Dataset")
    print("=" * 50)
    
    train, val, test = create_mixed_dataset()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Training pairs: {len(train)}")
    print(f"Validation pairs: {len(val)}")
    print(f"Test pairs: {len(test)}")
    
    print("\n📝 Sample pairs:")
    for i in range(5):
        en, es = train[i]
        print(f"  {en} → {es}")
    
    print("\n✅ Conversational dataset ready!")
    print("💡 Now train with: python train_conversational.py")


if __name__ == "__main__":
    main()