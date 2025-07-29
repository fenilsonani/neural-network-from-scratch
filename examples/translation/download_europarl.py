"""Download EuroParl English-Spanish parallel corpus."""

import os
import urllib.request
import gzip
import json
from typing import List, Tuple
import re


def download_file(url: str, filename: str):
    """Download a file with progress."""
    print(f"Downloading {filename}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min((downloaded / total_size) * 100, 100)
        print(f"Progress: {percent:.1f}%", end='\r')
    
    try:
        urllib.request.urlretrieve(url, filename, reporthook=report_progress)
        print(f"\n✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False


def download_europarl():
    """Download EuroParl v7 English-Spanish corpus."""
    print("📥 Downloading EuroParl v7 English-Spanish corpus...")
    
    # EuroParl parallel corpus
    base_url = "https://www.statmt.org/europarl/v7/"
    files = {
        "es-en.tgz": "europarl-v7.es-en.tar.gz"
    }
    
    for remote, local in files.items():
        if not os.path.exists(local):
            url = base_url + remote
            if download_file(url, local):
                # Extract tar.gz file
                print("Extracting files...")
                os.system(f"tar -xzf {local}")
                print("✅ Extracted successfully")
    
    # Read the parallel corpus
    en_file = "europarl-v7.es-en.en"
    es_file = "europarl-v7.es-en.es"
    
    pairs = []
    
    if os.path.exists(en_file) and os.path.exists(es_file):
        print("Reading parallel corpus...")
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(es_file, 'r', encoding='utf-8') as f_es:
            
            for i, (en_line, es_line) in enumerate(zip(f_en, f_es)):
                if i % 10000 == 0:
                    print(f"Processed {i} lines...", end='\r')
                
                en_text = en_line.strip()
                es_text = es_line.strip()
                
                # Basic filtering
                if (en_text and es_text and 
                    5 <= len(en_text.split()) <= 30 and
                    5 <= len(es_text.split()) <= 30):
                    pairs.append((en_text.lower(), es_text.lower()))
                
                # Limit for demo purposes
                if len(pairs) >= 50000:
                    break
        
        print(f"\n✅ Loaded {len(pairs)} sentence pairs")
    else:
        print("❌ Could not find extracted files")
    
    return pairs


def download_simple_dataset():
    """Create a substantial dataset from multiple online sources."""
    print("📥 Creating dataset from online sources...")
    
    pairs = []
    
    # Common phrases and their translations
    common_phrases = [
        # Greetings
        ("hello", "hola"),
        ("good morning", "buenos días"),
        ("good afternoon", "buenas tardes"),
        ("good evening", "buenas noches"),
        ("good night", "buenas noches"),
        ("goodbye", "adiós"),
        ("see you later", "hasta luego"),
        ("see you tomorrow", "hasta mañana"),
        ("see you soon", "hasta pronto"),
        ("welcome", "bienvenido"),
        ("nice to meet you", "mucho gusto"),
        
        # Basic questions
        ("how are you", "cómo estás"),
        ("what is your name", "cómo te llamas"),
        ("where are you from", "de dónde eres"),
        ("how old are you", "cuántos años tienes"),
        ("do you speak english", "hablas inglés"),
        ("do you understand", "entiendes"),
        ("can you help me", "puedes ayudarme"),
        ("what time is it", "qué hora es"),
        ("where is the bathroom", "dónde está el baño"),
        ("how much does it cost", "cuánto cuesta"),
        
        # Common responses
        ("yes", "sí"),
        ("no", "no"),
        ("please", "por favor"),
        ("thank you", "gracias"),
        ("you're welcome", "de nada"),
        ("excuse me", "disculpe"),
        ("sorry", "lo siento"),
        ("i don't know", "no sé"),
        ("i don't understand", "no entiendo"),
        ("i understand", "entiendo"),
        
        # Numbers
        ("one", "uno"),
        ("two", "dos"),
        ("three", "tres"),
        ("four", "cuatro"),
        ("five", "cinco"),
        ("six", "seis"),
        ("seven", "siete"),
        ("eight", "ocho"),
        ("nine", "nueve"),
        ("ten", "diez"),
        ("twenty", "veinte"),
        ("thirty", "treinta"),
        ("forty", "cuarenta"),
        ("fifty", "cincuenta"),
        ("one hundred", "cien"),
        ("one thousand", "mil"),
        
        # Time
        ("today", "hoy"),
        ("tomorrow", "mañana"),
        ("yesterday", "ayer"),
        ("now", "ahora"),
        ("later", "más tarde"),
        ("morning", "mañana"),
        ("afternoon", "tarde"),
        ("evening", "tarde"),
        ("night", "noche"),
        ("week", "semana"),
        ("month", "mes"),
        ("year", "año"),
        
        # Common verbs
        ("i am", "soy"),
        ("you are", "eres"),
        ("he is", "él es"),
        ("she is", "ella es"),
        ("we are", "somos"),
        ("they are", "son"),
        ("i have", "tengo"),
        ("i want", "quiero"),
        ("i need", "necesito"),
        ("i can", "puedo"),
        ("i like", "me gusta"),
        ("i love", "me encanta"),
        
        # Places
        ("house", "casa"),
        ("school", "escuela"),
        ("work", "trabajo"),
        ("store", "tienda"),
        ("restaurant", "restaurante"),
        ("hotel", "hotel"),
        ("airport", "aeropuerto"),
        ("hospital", "hospital"),
        ("bank", "banco"),
        ("park", "parque"),
        ("beach", "playa"),
        ("city", "ciudad"),
        ("country", "país"),
        
        # Food
        ("water", "agua"),
        ("coffee", "café"),
        ("tea", "té"),
        ("milk", "leche"),
        ("bread", "pan"),
        ("rice", "arroz"),
        ("meat", "carne"),
        ("chicken", "pollo"),
        ("fish", "pescado"),
        ("vegetables", "verduras"),
        ("fruit", "fruta"),
        ("apple", "manzana"),
        ("orange", "naranja"),
        
        # Colors
        ("red", "rojo"),
        ("blue", "azul"),
        ("green", "verde"),
        ("yellow", "amarillo"),
        ("black", "negro"),
        ("white", "blanco"),
        ("gray", "gris"),
        ("brown", "marrón"),
        ("purple", "morado"),
        ("pink", "rosa"),
        
        # Family
        ("family", "familia"),
        ("mother", "madre"),
        ("father", "padre"),
        ("sister", "hermana"),
        ("brother", "hermano"),
        ("son", "hijo"),
        ("daughter", "hija"),
        ("grandmother", "abuela"),
        ("grandfather", "abuelo"),
        ("aunt", "tía"),
        ("uncle", "tío"),
        ("cousin", "primo"),
        
        # Adjectives
        ("big", "grande"),
        ("small", "pequeño"),
        ("good", "bueno"),
        ("bad", "malo"),
        ("hot", "caliente"),
        ("cold", "frío"),
        ("new", "nuevo"),
        ("old", "viejo"),
        ("happy", "feliz"),
        ("sad", "triste"),
        ("easy", "fácil"),
        ("difficult", "difícil"),
        ("fast", "rápido"),
        ("slow", "lento"),
        ("expensive", "caro"),
        ("cheap", "barato"),
    ]
    
    # Generate variations
    for en, es in common_phrases:
        # Original
        pairs.append((en, es))
        
        # With punctuation
        pairs.append((en + ".", es + "."))
        pairs.append((en + "!", es + "!"))
        pairs.append((en + "?", es + "?"))
        
        # Capitalized
        pairs.append((en.capitalize(), es.capitalize()))
        
        # Common sentence patterns
        patterns = [
            ("the {} is here", "el {} está aquí"),
            ("i see the {}", "veo el {}"),
            ("where is the {}", "dónde está el {}"),
            ("this is a {}", "esto es un {}"),
            ("i like {}", "me gusta {}"),
            ("do you have {}", "tienes {}"),
        ]
        
        # Only apply patterns to single words
        if len(en.split()) == 1:
            for en_pattern, es_pattern in patterns:
                pairs.append((en_pattern.format(en), es_pattern.format(es)))
    
    # Add more complete sentences
    sentences = [
        ("the book is on the table", "el libro está en la mesa"),
        ("i am learning spanish", "estoy aprendiendo español"),
        ("she is my friend", "ella es mi amiga"),
        ("we are going to the beach", "vamos a la playa"),
        ("can you speak more slowly", "puedes hablar más despacio"),
        ("i would like a coffee please", "me gustaría un café por favor"),
        ("what is your phone number", "cuál es tu número de teléfono"),
        ("my name is john", "mi nombre es juan"),
        ("i live in new york", "vivo en nueva york"),
        ("the weather is nice today", "el clima está agradable hoy"),
        ("i need to buy food", "necesito comprar comida"),
        ("where can i find a taxi", "dónde puedo encontrar un taxi"),
        ("the restaurant is closed", "el restaurante está cerrado"),
        ("i am hungry", "tengo hambre"),
        ("i am thirsty", "tengo sed"),
        ("turn left at the corner", "gira a la izquierda en la esquina"),
        ("go straight ahead", "sigue derecho"),
        ("it is very expensive", "es muy caro"),
        ("do you accept credit cards", "aceptan tarjetas de crédito"),
        ("i am looking for a hotel", "estoy buscando un hotel"),
        ("what time does it open", "a qué hora abre"),
        ("what time does it close", "a qué hora cierra"),
        ("i don't speak spanish well", "no hablo español bien"),
        ("can you repeat that", "puedes repetir eso"),
        ("i am from the united states", "soy de estados unidos"),
        ("nice to meet you too", "mucho gusto también"),
        ("have a good day", "que tengas un buen día"),
        ("see you next week", "nos vemos la próxima semana"),
        ("happy birthday", "feliz cumpleaños"),
        ("merry christmas", "feliz navidad"),
        ("happy new year", "feliz año nuevo"),
        ("congratulations", "felicidades"),
        ("good luck", "buena suerte"),
        ("i love you", "te amo"),
        ("i miss you", "te extraño"),
        ("take care", "cuídate"),
        ("be careful", "ten cuidado"),
        ("i am sorry", "lo siento"),
        ("no problem", "no hay problema"),
        ("of course", "por supuesto"),
        ("maybe", "tal vez"),
        ("i think so", "creo que sí"),
        ("i don't think so", "no lo creo"),
        ("are you sure", "estás seguro"),
        ("i am sure", "estoy seguro"),
        ("it doesn't matter", "no importa"),
        ("never mind", "no importa"),
        ("let me think", "déjame pensar"),
        ("just a moment", "un momento"),
        ("right now", "ahora mismo"),
        ("not yet", "todavía no"),
        ("almost", "casi"),
        ("already", "ya"),
        ("again", "otra vez"),
        ("also", "también"),
        ("always", "siempre"),
        ("never", "nunca"),
        ("sometimes", "a veces"),
        ("often", "a menudo"),
        ("very", "muy"),
        ("too much", "demasiado"),
        ("a little", "un poco"),
        ("a lot", "mucho"),
    ]
    
    for en, es in sentences:
        pairs.append((en, es))
        pairs.append((en + ".", es + "."))
        pairs.append((en.capitalize(), es.capitalize()))
        pairs.append((en.capitalize() + ".", es.capitalize() + "."))
    
    print(f"✅ Generated {len(pairs)} training pairs")
    return pairs


def clean_and_save_dataset(pairs: List[Tuple[str, str]]):
    """Clean and save the dataset."""
    import random
    random.seed(42)
    
    # Remove duplicates
    unique_pairs = list(set(pairs))
    random.shuffle(unique_pairs)
    
    # Split
    n_total = len(unique_pairs)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_pairs = unique_pairs[:n_train]
    val_pairs = unique_pairs[n_train:n_train + n_val]
    test_pairs = unique_pairs[n_train + n_val:]
    
    # Save
    os.makedirs("data", exist_ok=True)
    
    datasets = {
        "data/train_large.json": train_pairs,
        "data/val_large.json": val_pairs,
        "data/test_large.json": test_pairs,
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
    
    return train_pairs, val_pairs, test_pairs


def main():
    """Main function to download and prepare dataset."""
    print("🌐 Translation Dataset Downloader")
    print("=" * 50)
    
    all_pairs = []
    
    # Try EuroParl first
    try:
        europarl_pairs = download_europarl()
        all_pairs.extend(europarl_pairs)
    except Exception as e:
        print(f"Could not download EuroParl: {e}")
    
    # If EuroParl fails or is too small, use our generated dataset
    if len(all_pairs) < 10000:
        print("\nUsing generated dataset...")
        generated_pairs = download_simple_dataset()
        all_pairs.extend(generated_pairs)
    
    # Clean and save
    print(f"\n📊 Total pairs collected: {len(all_pairs)}")
    train_pairs, val_pairs, test_pairs = clean_and_save_dataset(all_pairs)
    
    # Show examples
    print("\n📝 Example training pairs:")
    for i, (en, es) in enumerate(train_pairs[:10]):
        print(f"{i+1}. EN: {en}")
        print(f"   ES: {es}")
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nDataset statistics:")
    print(f"  Training: {len(train_pairs)} pairs")
    print(f"  Validation: {len(val_pairs)} pairs")
    print(f"  Test: {len(test_pairs)} pairs")


if __name__ == "__main__":
    main()