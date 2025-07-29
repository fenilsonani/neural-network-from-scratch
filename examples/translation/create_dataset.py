"""Create a comprehensive English-Spanish dataset."""

import json
import os
from typing import List, Tuple
import random


# Comprehensive dataset of common English-Spanish translations
TRANSLATION_DATA = [
    # Greetings and Basic Phrases
    ("hello", "hola"),
    ("good morning", "buenos días"),
    ("good afternoon", "buenas tardes"),
    ("good evening", "buenas noches"),
    ("good night", "buenas noches"),
    ("goodbye", "adiós"),
    ("see you later", "hasta luego"),
    ("see you tomorrow", "hasta mañana"),
    ("see you soon", "hasta pronto"),
    ("nice to meet you", "mucho gusto"),
    ("how are you", "cómo estás"),
    ("i am fine", "estoy bien"),
    ("thank you", "gracias"),
    ("thank you very much", "muchas gracias"),
    ("you're welcome", "de nada"),
    ("please", "por favor"),
    ("excuse me", "disculpe"),
    ("i'm sorry", "lo siento"),
    ("yes", "sí"),
    ("no", "no"),
    
    # Questions
    ("what is your name", "cuál es tu nombre"),
    ("my name is john", "mi nombre es juan"),
    ("where are you from", "de dónde eres"),
    ("i am from spain", "soy de españa"),
    ("i am from mexico", "soy de méxico"),
    ("how old are you", "cuántos años tienes"),
    ("i am twenty years old", "tengo veinte años"),
    ("do you speak english", "hablas inglés"),
    ("do you speak spanish", "hablas español"),
    ("i speak a little spanish", "hablo un poco de español"),
    ("i don't understand", "no entiendo"),
    ("can you repeat", "puedes repetir"),
    ("can you speak slowly", "puedes hablar despacio"),
    ("what does that mean", "qué significa eso"),
    ("how do you say", "cómo se dice"),
    
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
    
    # Days of the week
    ("monday", "lunes"),
    ("tuesday", "martes"),
    ("wednesday", "miércoles"),
    ("thursday", "jueves"),
    ("friday", "viernes"),
    ("saturday", "sábado"),
    ("sunday", "domingo"),
    
    # Time expressions
    ("today", "hoy"),
    ("tomorrow", "mañana"),
    ("yesterday", "ayer"),
    ("now", "ahora"),
    ("later", "más tarde"),
    ("early", "temprano"),
    ("late", "tarde"),
    ("what time is it", "qué hora es"),
    ("it's one o'clock", "es la una"),
    ("it's two o'clock", "son las dos"),
    
    # Common verbs and phrases
    ("i want", "quiero"),
    ("i need", "necesito"),
    ("i have", "tengo"),
    ("i don't have", "no tengo"),
    ("i like", "me gusta"),
    ("i don't like", "no me gusta"),
    ("i love", "me encanta"),
    ("i love you", "te amo"),
    ("i can", "puedo"),
    ("i can't", "no puedo"),
    ("i know", "sé"),
    ("i don't know", "no sé"),
    ("i understand", "entiendo"),
    ("i think", "creo"),
    ("i believe", "creo"),
    
    # Food and drink
    ("water", "agua"),
    ("coffee", "café"),
    ("tea", "té"),
    ("beer", "cerveza"),
    ("wine", "vino"),
    ("bread", "pan"),
    ("rice", "arroz"),
    ("meat", "carne"),
    ("chicken", "pollo"),
    ("fish", "pescado"),
    ("vegetables", "verduras"),
    ("fruit", "fruta"),
    ("apple", "manzana"),
    ("orange", "naranja"),
    ("banana", "plátano"),
    
    # Places
    ("house", "casa"),
    ("home", "hogar"),
    ("school", "escuela"),
    ("work", "trabajo"),
    ("office", "oficina"),
    ("store", "tienda"),
    ("restaurant", "restaurante"),
    ("hotel", "hotel"),
    ("airport", "aeropuerto"),
    ("hospital", "hospital"),
    ("bank", "banco"),
    ("church", "iglesia"),
    ("park", "parque"),
    ("beach", "playa"),
    ("city", "ciudad"),
    
    # Directions
    ("where is", "dónde está"),
    ("here", "aquí"),
    ("there", "allí"),
    ("left", "izquierda"),
    ("right", "derecha"),
    ("straight", "derecho"),
    ("north", "norte"),
    ("south", "sur"),
    ("east", "este"),
    ("west", "oeste"),
    
    # Family
    ("family", "familia"),
    ("mother", "madre"),
    ("father", "padre"),
    ("sister", "hermana"),
    ("brother", "hermano"),
    ("daughter", "hija"),
    ("son", "hijo"),
    ("grandmother", "abuela"),
    ("grandfather", "abuelo"),
    ("aunt", "tía"),
    ("uncle", "tío"),
    
    # Descriptions
    ("big", "grande"),
    ("small", "pequeño"),
    ("good", "bueno"),
    ("bad", "malo"),
    ("hot", "caliente"),
    ("cold", "frío"),
    ("new", "nuevo"),
    ("old", "viejo"),
    ("young", "joven"),
    ("beautiful", "hermoso"),
    ("ugly", "feo"),
    ("happy", "feliz"),
    ("sad", "triste"),
    ("easy", "fácil"),
    ("difficult", "difícil"),
    
    # Common sentences
    ("the book is on the table", "el libro está en la mesa"),
    ("i am learning spanish", "estoy aprendiendo español"),
    ("she is my friend", "ella es mi amiga"),
    ("he is my brother", "él es mi hermano"),
    ("we are students", "somos estudiantes"),
    ("they are teachers", "ellos son profesores"),
    ("the weather is nice", "el clima está agradable"),
    ("it is raining", "está lloviendo"),
    ("the sun is shining", "el sol está brillando"),
    ("i am hungry", "tengo hambre"),
    ("i am thirsty", "tengo sed"),
    ("i am tired", "estoy cansado"),
    ("i am happy", "estoy feliz"),
    ("can you help me", "puedes ayudarme"),
    ("of course", "por supuesto"),
    ("maybe", "tal vez"),
    ("i don't think so", "no lo creo"),
    ("that's right", "eso es correcto"),
    ("that's wrong", "eso está mal"),
    ("it doesn't matter", "no importa"),
    ("what happened", "qué pasó"),
    ("nothing happened", "no pasó nada"),
    ("everything is fine", "todo está bien"),
    ("be careful", "ten cuidado"),
    ("good luck", "buena suerte"),
    ("congratulations", "felicidades"),
    ("happy birthday", "feliz cumpleaños"),
    ("merry christmas", "feliz navidad"),
    ("happy new year", "feliz año nuevo"),
    
    # Shopping
    ("how much does it cost", "cuánto cuesta"),
    ("it's expensive", "es caro"),
    ("it's cheap", "es barato"),
    ("i want to buy", "quiero comprar"),
    ("do you have", "tienes"),
    ("i'm just looking", "solo estoy mirando"),
    ("can i pay with card", "puedo pagar con tarjeta"),
    ("where is the bathroom", "dónde está el baño"),
    ("i need help", "necesito ayuda"),
    ("thank you for your help", "gracias por tu ayuda"),
    
    # Extended phrases
    ("could you please repeat that", "podrías repetir eso por favor"),
    ("i would like a coffee", "me gustaría un café"),
    ("what do you recommend", "qué recomiendas"),
    ("i don't speak spanish very well", "no hablo español muy bien"),
    ("i'm learning spanish", "estoy aprendiendo español"),
    ("how long have you been here", "cuánto tiempo has estado aquí"),
    ("i've been here for two days", "he estado aquí por dos días"),
    ("where can i find a restaurant", "dónde puedo encontrar un restaurante"),
    ("is there a pharmacy nearby", "hay una farmacia cerca"),
    ("i need to go to the hospital", "necesito ir al hospital"),
    ("can you call a taxi", "puedes llamar un taxi"),
    ("how far is it", "qué tan lejos está"),
    ("it's very close", "está muy cerca"),
    ("it's far away", "está lejos"),
    ("turn left at the corner", "gira a la izquierda en la esquina"),
    ("go straight ahead", "sigue derecho"),
    ("it's on your right", "está a tu derecha"),
    ("i lost my passport", "perdí mi pasaporte"),
    ("can you help me find it", "puedes ayudarme a encontrarlo"),
    ("i need to call the police", "necesito llamar a la policía"),
    ("is it safe here", "es seguro aquí"),
    ("what's the wifi password", "cuál es la contraseña del wifi"),
    ("can i have the menu please", "puedo tener el menú por favor"),
    ("i'm allergic to nuts", "soy alérgico a las nueces"),
    ("the food was delicious", "la comida estaba deliciosa"),
    ("can i have the bill", "puedo tener la cuenta"),
    ("keep the change", "quédate con el cambio"),
    ("where can i exchange money", "dónde puedo cambiar dinero"),
    ("what's the exchange rate", "cuál es el tipo de cambio"),
    ("i'd like to make a reservation", "me gustaría hacer una reservación"),
    ("for how many people", "para cuántas personas"),
    ("what time does it open", "a qué hora abre"),
    ("what time does it close", "a qué hora cierra"),
    ("is it open on sundays", "está abierto los domingos"),
    ("how do i get there", "cómo llego allí"),
    ("can you show me on the map", "puedes mostrarme en el mapa"),
    ("i'm looking for this address", "estoy buscando esta dirección"),
    ("is this the right way", "es este el camino correcto"),
    ("i think i'm lost", "creo que estoy perdido"),
    ("can you take a photo of us", "puedes tomarnos una foto"),
    ("smile", "sonríe"),
    ("it's a beautiful day", "es un día hermoso"),
    ("i had a great time", "la pasé muy bien"),
    ("see you next time", "nos vemos la próxima vez"),
    ("take care", "cuídate"),
    ("have a nice day", "que tengas un buen día"),
    ("sweet dreams", "dulces sueños"),
    ("i miss you", "te extraño"),
    ("i'll be right back", "ya regreso"),
    ("wait for me", "espérame"),
    ("let's go", "vamos"),
    ("hurry up", "apúrate"),
    ("slow down", "más despacio"),
    ("be quiet", "silencio"),
    ("speak louder", "habla más fuerte"),
    ("i can't hear you", "no te escucho"),
    ("can you write it down", "puedes escribirlo"),
    ("how do you spell that", "cómo se deletrea eso"),
    ("what's your phone number", "cuál es tu número de teléfono"),
    ("what's your email", "cuál es tu correo electrónico"),
    ("add me on social media", "agrégame en redes sociales"),
    ("do you have facebook", "tienes facebook"),
    ("let's keep in touch", "mantengámonos en contacto"),
    ("it was nice talking to you", "fue agradable hablar contigo"),
    ("i hope to see you again", "espero verte de nuevo"),
    ("have a safe trip", "que tengas un buen viaje"),
    ("welcome to spain", "bienvenido a españa"),
    ("enjoy your stay", "disfruta tu estancia"),
    ("the party was fun", "la fiesta estuvo divertida"),
    ("i'm bored", "estoy aburrido"),
    ("let's do something", "hagamos algo"),
    ("what do you want to do", "qué quieres hacer"),
    ("i don't mind", "no me importa"),
    ("it's up to you", "depende de ti"),
    ("whatever you want", "lo que tú quieras"),
    ("i agree", "estoy de acuerdo"),
    ("i disagree", "no estoy de acuerdo"),
    ("you're right", "tienes razón"),
    ("you're wrong", "estás equivocado"),
    ("that's interesting", "eso es interesante"),
    ("tell me more", "cuéntame más"),
    ("really", "en serio"),
    ("are you serious", "hablas en serio"),
    ("i'm just kidding", "solo estoy bromeando"),
    ("don't worry", "no te preocupes"),
    ("everything will be fine", "todo estará bien"),
    ("trust me", "confía en mí"),
    ("i promise", "lo prometo"),
    ("i swear", "lo juro"),
    ("cross my heart", "te lo juro"),
    ("fingers crossed", "crucemos los dedos"),
    ("knock on wood", "toquemos madera"),
    ("bless you", "salud"),
    ("get well soon", "que te mejores pronto"),
    ("take your time", "tómate tu tiempo"),
    ("no rush", "sin prisa"),
    ("make yourself at home", "siéntete como en casa"),
    ("help yourself", "sírvete"),
    ("after you", "después de ti"),
    ("ladies first", "las damas primero"),
    ("age before beauty", "la edad antes que la belleza"),
    ("the early bird catches the worm", "al que madruga dios le ayuda"),
    ("better late than never", "más vale tarde que nunca"),
    ("practice makes perfect", "la práctica hace al maestro"),
    ("time flies", "el tiempo vuela"),
    ("time is money", "el tiempo es oro"),
    ("break a leg", "mucha suerte"),
    ("piece of cake", "pan comido"),
    ("it's raining cats and dogs", "está lloviendo a cántaros"),
    ("i'm all ears", "soy todo oídos"),
    ("it's a small world", "el mundo es un pañuelo"),
    ("actions speak louder than words", "las acciones hablan más que las palabras"),
]


def augment_dataset(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Augment dataset with variations."""
    augmented = list(pairs)
    
    # Add capitalized versions
    for en, es in pairs:
        # Capitalize first letter
        augmented.append((en.capitalize(), es.capitalize()))
        
        # Add punctuation variations
        if not en.endswith(('.', '!', '?')):
            augmented.append((en + ".", es + "."))
            augmented.append((en + "!", es + "!"))
            augmented.append((en + "?", es + "?"))
    
    # Add "I" contractions
    contractions = [
        ("i am", "i'm"),
        ("i have", "i've"),
        ("i will", "i'll"),
        ("i would", "i'd"),
        ("i had", "i'd"),
    ]
    
    for long_form, short_form in contractions:
        for en, es in pairs:
            if long_form in en:
                new_en = en.replace(long_form, short_form)
                augmented.append((new_en, es))
    
    return augmented


def create_dataset_files():
    """Create train, validation, and test datasets."""
    # Get all pairs
    all_pairs = augment_dataset(TRANSLATION_DATA)
    
    # Remove duplicates
    unique_pairs = list(set(all_pairs))
    
    # Shuffle
    random.seed(42)
    random.shuffle(unique_pairs)
    
    # Split ratios
    n_total = len(unique_pairs)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    # Create splits
    train_pairs = unique_pairs[:n_train]
    val_pairs = unique_pairs[n_train:n_train + n_val]
    test_pairs = unique_pairs[n_train + n_val:]
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save datasets
    datasets = {
        "data/train.json": train_pairs,
        "data/val.json": val_pairs,
        "data/test.json": test_pairs,
        "data/all_pairs.json": unique_pairs
    }
    
    for filename, pairs in datasets.items():
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
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total unique pairs: {n_total}")
    print(f"  Training set: {len(train_pairs)} pairs")
    print(f"  Validation set: {len(val_pairs)} pairs")
    print(f"  Test set: {len(test_pairs)} pairs")
    
    # Show examples
    print(f"\n📝 Example pairs from training set:")
    for i, (en, es) in enumerate(train_pairs[:10]):
        print(f"{i+1:2d}. EN: {en}")
        print(f"    ES: {es}")


def main():
    """Create the dataset files."""
    print("🌐 Creating English-Spanish Translation Dataset")
    print("=" * 50)
    
    create_dataset_files()
    
    print("\n✅ Dataset creation complete!")
    print("\n📁 Created files:")
    print("  - data/train.json")
    print("  - data/val.json") 
    print("  - data/test.json")
    print("  - data/all_pairs.json")
    print("\nYou can now run train.py to train the translation model!")


if __name__ == "__main__":
    main()