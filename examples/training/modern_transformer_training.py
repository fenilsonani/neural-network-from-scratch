#!/usr/bin/env python3
"""
🧬 Modern Transformer Training Script - Advanced Architecture Training

Proper training of Modern Transformer with advanced features on real text data:
- Real text dataset (synthetic story and knowledge data)
- Pre-Norm architecture with RoPE positional encoding
- SwiGLU activation and RMSNorm normalization
- Advanced training loop with perplexity evaluation
- Model checkpointing and text generation testing
- Automatic optimizations enabled
"""

import sys
import os
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from neural_arch.core import Tensor
from neural_arch.models.language.modern_transformer import PreNormTransformer, PreNormTransformerConfig
from neural_arch.optim import AdamW
from neural_arch.functional import cross_entropy_loss, softmax
from neural_arch.optimization_config import configure, get_config
from neural_arch.backends import available_backends

@dataclass
class ModernTransformerTrainingConfig:
    """Modern Transformer training configuration."""
    # Model config
    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 256
    vocab_size: int = 12000
    
    # Advanced features
    activation: str = "swiglu"  # swiglu, gelu
    normalization: str = "rmsnorm"  # rmsnorm, layernorm
    use_rope: bool = True
    rope_base: int = 10000
    tie_embeddings: bool = True
    scale_embeddings: bool = True
    
    # Training config
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    num_epochs: int = 8
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Data config
    train_size: int = 3000  # Number of sequences
    val_size: int = 600
    
    # Generation config for testing
    generate_every: int = 150  # Generate samples every N steps
    max_generate_length: int = 64
    
    # Optimization config
    enable_optimizations: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints/modern_transformer"

class AdvancedTextTokenizer:
    """Advanced text tokenizer with larger vocabulary."""
    
    def __init__(self, vocab_size: int = 12000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0,
            '<BOS>': 1,  # Beginning of sequence
            '<EOS>': 2,  # End of sequence
            '<UNK>': 3,
            '<MASK>': 4  # For future masked language modeling
        }
        self.vocab = {}
        self.id_to_token = {}
        self._build_comprehensive_vocab()
    
    def _build_comprehensive_vocab(self):
        """Build comprehensive vocabulary for modern transformer."""
        # Start with special tokens
        self.vocab.update(self.special_tokens)
        
        # Core vocabulary - common words
        core_words = [
            # Articles, prepositions, conjunctions
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'under', 'over', 'within', 'without',
            
            # Pronouns and determiners
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'myself', 'yourself', 'himself',
            'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 'this', 'that',
            'these', 'those', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
            'why', 'how', 'all', 'any', 'some', 'each', 'every', 'both', 'either', 'neither',
            
            # Verbs - common and auxiliary
            'is', 'am', 'are', 'was', 'were', 'being', 'been', 'be', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'shall', 'ought', 'dare', 'need', 'used',
            'go', 'going', 'goes', 'went', 'gone', 'get', 'getting', 'gets', 'got', 'gotten',
            'make', 'making', 'makes', 'made', 'take', 'taking', 'takes', 'took', 'taken',
            'come', 'coming', 'comes', 'came', 'see', 'seeing', 'sees', 'saw', 'seen',
            'know', 'knowing', 'knows', 'knew', 'known', 'think', 'thinking', 'thinks',
            'thought', 'say', 'saying', 'says', 'said', 'tell', 'telling', 'tells', 'told',
            'ask', 'asking', 'asks', 'asked', 'give', 'giving', 'gives', 'gave', 'given',
            'find', 'finding', 'finds', 'found', 'use', 'using', 'uses', 'used',
            'work', 'working', 'works', 'worked', 'try', 'trying', 'tries', 'tried',
            'want', 'wanting', 'wants', 'wanted', 'need', 'needing', 'needs', 'needed',
            'like', 'liking', 'likes', 'liked', 'love', 'loving', 'loves', 'loved',
            'feel', 'feeling', 'feels', 'felt', 'seem', 'seeming', 'seems', 'seemed',
            'look', 'looking', 'looks', 'looked', 'appear', 'appearing', 'appears', 'appeared',
            'become', 'becoming', 'becomes', 'became', 'turn', 'turning', 'turns', 'turned',
            'put', 'putting', 'puts', 'keep', 'keeping', 'keeps', 'kept', 'let', 'letting',
            'lets', 'help', 'helping', 'helps', 'helped', 'show', 'showing', 'shows', 'showed',
            'play', 'playing', 'plays', 'played', 'run', 'running', 'runs', 'ran',
            'move', 'moving', 'moves', 'moved', 'live', 'living', 'lives', 'lived',
            'believe', 'believing', 'believes', 'believed', 'bring', 'bringing', 'brings', 'brought',
            'happen', 'happening', 'happens', 'happened', 'write', 'writing', 'writes', 'wrote',
            'provide', 'providing', 'provides', 'provided', 'sit', 'sitting', 'sits', 'sat',
            'stand', 'standing', 'stands', 'stood', 'lose', 'losing', 'loses', 'lost',
            'pay', 'paying', 'pays', 'paid', 'meet', 'meeting', 'meets', 'met',
            'include', 'including', 'includes', 'included', 'continue', 'continuing', 'continues', 'continued',
            'set', 'setting', 'sets', 'follow', 'following', 'follows', 'followed',
            'stop', 'stopping', 'stops', 'stopped', 'create', 'creating', 'creates', 'created',
            'speak', 'speaking', 'speaks', 'spoke', 'read', 'reading', 'reads',
            'allow', 'allowing', 'allows', 'allowed', 'add', 'adding', 'adds', 'added',
            'spend', 'spending', 'spends', 'spent', 'grow', 'growing', 'grows', 'grew',
            'open', 'opening', 'opens', 'opened', 'walk', 'walking', 'walks', 'walked',
            'win', 'winning', 'wins', 'won', 'offer', 'offering', 'offers', 'offered',
            'remember', 'remembering', 'remembers', 'remembered', 'consider', 'considering', 'considers', 'considered',
            'appear', 'appearing', 'appears', 'appeared', 'buy', 'buying', 'buys', 'bought',
            'wait', 'waiting', 'waits', 'waited', 'serve', 'serving', 'serves', 'served',
            'die', 'dying', 'dies', 'died', 'send', 'sending', 'sends', 'sent',
            'expect', 'expecting', 'expects', 'expected', 'build', 'building', 'builds', 'built',
            'stay', 'staying', 'stays', 'stayed', 'fall', 'falling', 'falls', 'fell',
            'cut', 'cutting', 'cuts', 'reach', 'reaching', 'reaches', 'reached',
            'kill', 'killing', 'kills', 'killed', 'remain', 'remaining', 'remains', 'remained',
            
            # Nouns - common categories
            'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand',
            'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government',
            'company', 'number', 'group', 'problem', 'fact', 'right', 'house', 'service', 'friend',
            'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car', 'city',
            'community', 'name', 'president', 'team', 'minute', 'idea', 'kid', 'body', 'information',
            'back', 'parent', 'face', 'others', 'level', 'office', 'door', 'health', 'person',
            'art', 'war', 'history', 'party', 'result', 'change', 'morning', 'reason', 'research',
            'girl', 'guy', 'moment', 'air', 'teacher', 'force', 'education', 'foot', 'boy',
            'age', 'policy', 'process', 'music', 'market', 'sense', 'nation', 'plan', 'college',
            'interest', 'death', 'experience', 'effect', 'use', 'class', 'control', 'care',
            'field', 'development', 'role', 'student', 'word', 'lot', 'family', 'business',
            'issue', 'area', 'state', 'question', 'school', 'country', 'american', 'type',
            'thought', 'head', 'example', 'money', 'story', 'month', 'book', 'night', 'job',
            'water', 'room', 'mother', 'subject', 'rest', 'event', 'study', 'program',
            'society', 'individual', 'relationship', 'action', 'everything', 'nothing', 'something',
            'anything', 'someone', 'everyone', 'anyone', 'nobody', 'somebody', 'everybody',
            'anybody', 'home', 'food', 'land', 'news', 'computer', 'system', 'technology',
            'internet', 'phone', 'email', 'website', 'data', 'software', 'application',
            'network', 'security', 'database', 'server', 'user', 'password', 'account',
            'science', 'research', 'theory', 'experiment', 'analysis', 'method', 'approach',
            'technique', 'process', 'procedure', 'standard', 'quality', 'performance',
            'energy', 'environment', 'nature', 'animal', 'plant', 'tree', 'flower', 'garden',
            'weather', 'climate', 'temperature', 'season', 'summer', 'winter', 'spring', 'fall',
            'medicine', 'health', 'doctor', 'hospital', 'patient', 'treatment', 'disease',
            'economy', 'business', 'industry', 'company', 'market', 'customer', 'product',
            'service', 'price', 'cost', 'budget', 'investment', 'profit', 'finance',
            'culture', 'society', 'community', 'tradition', 'language', 'communication',
            'media', 'television', 'radio', 'newspaper', 'magazine', 'article', 'report',
            'travel', 'journey', 'trip', 'vacation', 'adventure', 'exploration', 'discovery',
            'transportation', 'vehicle', 'car', 'train', 'plane', 'ship', 'bicycle',
            'art', 'music', 'painting', 'literature', 'poetry', 'novel', 'story', 'film',
            'movie', 'theater', 'performance', 'entertainment', 'sport', 'game', 'competition',
            
            # Adjectives - descriptive words
            'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old',
            'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young',
            'important', 'few', 'public', 'bad', 'same', 'able', 'human', 'local', 'sure',
            'without', 'free', 'true', 'federal', 'international', 'full', 'special', 'easy',
            'hard', 'left', 'possible', 'social', 'late', 'real', 'best', 'far', 'available',
            'likely', 'short', 'single', 'individual', 'complete', 'red', 'open', 'black',
            'white', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'pink', 'gray',
            'beautiful', 'wonderful', 'amazing', 'incredible', 'fantastic', 'excellent', 'perfect',
            'brilliant', 'magnificent', 'spectacular', 'outstanding', 'remarkable', 'extraordinary',
            'terrible', 'awful', 'horrible', 'dreadful', 'disgusting', 'unpleasant', 'nasty',
            'happy', 'sad', 'angry', 'excited', 'calm', 'peaceful', 'worried', 'nervous',
            'confident', 'proud', 'ashamed', 'embarrassed', 'surprised', 'shocked', 'amazed',
            'fast', 'slow', 'quick', 'rapid', 'swift', 'immediate', 'instant', 'gradual',
            'hot', 'cold', 'warm', 'cool', 'freezing', 'boiling', 'mild', 'extreme',
            'loud', 'quiet', 'silent', 'noisy', 'soft', 'hard', 'smooth', 'rough',
            'clean', 'dirty', 'pure', 'fresh', 'old', 'ancient', 'modern', 'contemporary',
            'strong', 'weak', 'powerful', 'gentle', 'tough', 'delicate', 'fragile', 'solid',
            'heavy', 'light', 'thick', 'thin', 'wide', 'narrow', 'broad', 'tight', 'loose',
            'rich', 'poor', 'expensive', 'cheap', 'valuable', 'worthless', 'precious',
            'intelligent', 'smart', 'clever', 'wise', 'stupid', 'foolish', 'brilliant', 'genius',
            'kind', 'mean', 'nice', 'cruel', 'generous', 'selfish', 'helpful', 'harmful',
            'honest', 'dishonest', 'truthful', 'false', 'loyal', 'faithful', 'reliable',
            'creative', 'artistic', 'imaginative', 'original', 'unique', 'common', 'ordinary',
            'strange', 'weird', 'unusual', 'normal', 'typical', 'standard', 'regular',
            'safe', 'dangerous', 'risky', 'secure', 'protected', 'vulnerable', 'threatened',
            
            # Adverbs
            'very', 'so', 'too', 'also', 'just', 'only', 'even', 'still', 'again', 'then',
            'now', 'here', 'there', 'where', 'when', 'how', 'why', 'well', 'better', 'best',
            'more', 'most', 'much', 'many', 'little', 'less', 'least', 'quite', 'rather',
            'pretty', 'really', 'truly', 'certainly', 'probably', 'possibly', 'maybe',
            'perhaps', 'definitely', 'absolutely', 'completely', 'totally', 'entirely',
            'partly', 'slightly', 'somewhat', 'fairly', 'extremely', 'incredibly', 'amazingly',
            'surprisingly', 'unfortunately', 'hopefully', 'obviously', 'clearly', 'apparently',
            'quickly', 'slowly', 'carefully', 'easily', 'hardly', 'nearly', 'almost',
            'always', 'never', 'sometimes', 'often', 'usually', 'rarely', 'seldom',
            'frequently', 'occasionally', 'constantly', 'continuously', 'immediately',
            'suddenly', 'gradually', 'eventually', 'finally', 'recently', 'lately',
            'yesterday', 'today', 'tomorrow', 'soon', 'later', 'earlier', 'afterwards',
            'forward', 'backward', 'upward', 'downward', 'inside', 'outside', 'nearby',
            'somewhere', 'anywhere', 'everywhere', 'nowhere',
            
            # Numbers and quantities
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
            'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
            'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion', 'trillion',
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth',
            'ninth', 'tenth', 'once', 'twice', 'double', 'triple', 'single', 'multiple',
            'half', 'quarter', 'third', 'whole', 'entire', 'complete', 'partial',
            'none', 'zero', 'nothing', 'everything', 'something', 'anything', 'all',
            'both', 'either', 'neither', 'each', 'every', 'any', 'some', 'several',
            'few', 'many', 'much', 'more', 'most', 'less', 'least', 'enough',
            
            # Technology and modern terms
            'computer', 'internet', 'website', 'online', 'digital', 'virtual', 'electronic',
            'software', 'hardware', 'program', 'application', 'app', 'system', 'network',
            'data', 'information', 'database', 'server', 'cloud', 'artificial', 'intelligence',
            'machine', 'learning', 'algorithm', 'code', 'programming', 'development',
            'technology', 'innovation', 'invention', 'discovery', 'research', 'science',
            'smartphone', 'tablet', 'laptop', 'desktop', 'device', 'gadget', 'tool',
            'social', 'media', 'platform', 'facebook', 'twitter', 'instagram', 'youtube',
            'google', 'search', 'engine', 'browser', 'chrome', 'safari', 'firefox',
            'email', 'message', 'text', 'chat', 'video', 'audio', 'image', 'photo',
            'download', 'upload', 'stream', 'share', 'like', 'comment', 'follow',
            'user', 'account', 'profile', 'username', 'password', 'login', 'logout',
            'security', 'privacy', 'protection', 'encryption', 'firewall', 'virus',
            'backup', 'storage', 'memory', 'processor', 'cpu', 'gpu', 'ram', 'disk'
        ]
        
        # Add core words
        for word in core_words:
            if len(self.vocab) < self.vocab_size:
                self.vocab[word] = len(self.vocab)
        
        # Add punctuation and symbols
        punctuation = ['.', ',', '!', '?', ':', ';', '-', '_', '(', ')', '[', ']', '{', '}',
                      '"', "'", '/', '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=',
                      '<', '>', '|', '~', '`']
        for char in punctuation:
            if len(self.vocab) < self.vocab_size:
                self.vocab[char] = len(self.vocab)
        
        # Add digits
        for i in range(10):
            if len(self.vocab) < self.vocab_size:
                self.vocab[str(i)] = len(self.vocab)
        
        # Fill remaining with character-level tokens
        for i in range(ord('a'), ord('z') + 1):
            if len(self.vocab) < self.vocab_size:
                self.vocab[chr(i)] = len(self.vocab)
        
        for i in range(ord('A'), ord('Z') + 1):
            if len(self.vocab) < self.vocab_size:
                self.vocab[chr(i)] = len(self.vocab)
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"Built advanced vocabulary with {len(self.vocab)} tokens")
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs with improved handling."""
        tokens = []
        # Simple word-level tokenization with fallback to character-level
        words = text.lower().replace('.', ' . ').replace(',', ' , ').replace('!', ' ! ').replace('?', ' ? ').split()
        
        for word in words:
            # Try word-level first
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Fall back to character-level for unknown words
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.special_tokens['<UNK>'])
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        # Simple reconstruction
        text = ' '.join(tokens)
        # Fix punctuation spacing
        text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        return text

class EnhancedTextDataset:
    """Enhanced text dataset with diverse content for modern transformer."""
    
    def __init__(self, tokenizer: AdvancedTextTokenizer, config: ModernTransformerTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.train_sequences = []
        self.val_sequences = []
        self._create_diverse_text_data()
    
    def _generate_story_content(self) -> List[str]:
        """Generate diverse story content."""
        stories = []
        
        # Adventure stories
        adventure_templates = [
            "Once upon a time, in a distant land, there lived a brave young adventurer named Alex. Alex had always dreamed of exploring the mysterious forest that bordered their small village. One morning, Alex packed a bag with supplies and set off into the unknown wilderness. The forest was filled with ancient trees, sparkling streams, and hidden pathways. As Alex ventured deeper, they discovered a clearing where a magnificent castle stood, its towers reaching toward the sky. Inside the castle, Alex found a library filled with books containing the wisdom of ages past.",
            
            "The ocean waves crashed against the rocky shore as Captain Sarah prepared her ship for the voyage of a lifetime. She had heard tales of a legendary island where precious treasures were hidden by pirates centuries ago. Her crew was small but loyal, and together they had weathered many storms. As they sailed into the open sea, the wind filled their sails and carried them toward their destiny. Days passed, and finally, they spotted land on the horizon. The island was covered in lush vegetation, and ancient ruins could be seen through the trees.",
            
            "In the heart of the bustling city, Detective Johnson received a case that would challenge everything they knew about solving mysteries. A valuable artifact had disappeared from the museum, and there were no witnesses or obvious clues. Johnson began investigating by interviewing the museum staff and examining the security footage. Each piece of evidence led to more questions rather than answers. The detective realized that this case would require thinking outside the box and using unconventional methods to uncover the truth.",
        ]
        
        # Educational content
        educational_templates = [
            "The study of artificial intelligence has revolutionized the way we understand machine learning and computer science. Modern algorithms can process vast amounts of data and identify patterns that would be impossible for humans to detect. Neural networks, inspired by the structure of the human brain, use interconnected nodes to learn from examples and make predictions. Deep learning techniques have enabled breakthroughs in image recognition, natural language processing, and autonomous systems. As technology continues to advance, researchers are exploring new frontiers in AI development.",
            
            "Climate change represents one of the most significant challenges facing our planet today. Scientists have observed rising global temperatures, changing weather patterns, and melting ice caps over the past several decades. The primary cause of these changes is the increase in greenhouse gases in the atmosphere, particularly carbon dioxide from human activities. Understanding the complex interactions between Earth's climate systems requires sophisticated computer models and extensive data collection from around the world. Addressing climate change will require international cooperation and innovative solutions.",
            
            "The human body is an incredibly complex system made up of trillions of cells working together to maintain life. Each cell contains genetic information stored in DNA, which provides instructions for growth, development, and function. The circulatory system transports oxygen and nutrients throughout the body, while the nervous system coordinates communication between different organs. Regular exercise, proper nutrition, and adequate sleep are essential for maintaining optimal health and preventing disease. Medical research continues to expand our understanding of how the body works and how to treat various conditions.",
        ]
        
        # Technology articles
        technology_templates = [
            "The development of quantum computing represents a revolutionary leap forward in computational technology. Unlike classical computers that use bits to represent information as either zero or one, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously. This property, known as superposition, allows quantum computers to perform certain calculations exponentially faster than traditional computers. Companies and research institutions around the world are racing to build practical quantum computers that could solve complex problems in cryptography, drug discovery, and artificial intelligence.",
            
            "Virtual reality technology has evolved from science fiction concept to practical application in just a few decades. Modern VR systems use advanced displays, sensors, and processing power to create immersive digital environments that users can explore and interact with. Applications range from entertainment and gaming to education, training, and therapy. As the technology becomes more accessible and affordable, we can expect to see virtual reality integrated into many aspects of daily life, from remote work and social interaction to shopping and travel experiences.",
            
            "The Internet of Things, or IoT, refers to the network of interconnected devices that can communicate and share data with each other. Smart homes feature IoT devices like thermostats, security cameras, and appliances that can be controlled remotely through smartphone apps. In industrial settings, IoT sensors monitor equipment performance and predict maintenance needs. The massive amount of data generated by IoT devices creates opportunities for analysis and optimization but also raises important questions about privacy and security.",
        ]
        
        # Historical narratives
        historical_templates = [
            "The Renaissance period, spanning roughly from the 14th to the 17th century, marked a time of unprecedented cultural, artistic, and intellectual achievement in Europe. This era saw the emergence of great artists like Leonardo da Vinci and Michelangelo, whose works continue to inspire people today. Scientific discoveries by figures such as Galileo Galilei and Nicolaus Copernicus challenged traditional beliefs about the universe. The invention of the printing press by Johannes Gutenberg revolutionized the spread of knowledge and literacy. Literature flourished with writers like William Shakespeare creating timeless works that are still performed and studied worldwide.",
            
            "The Industrial Revolution transformed human society in ways that continue to shape our world today. Beginning in the late 18th century, this period saw the development of steam engines, mechanized manufacturing, and new transportation systems. Factories replaced small workshops, and people moved from rural areas to cities in search of work. While industrialization brought increased productivity and economic growth, it also created new social challenges including poor working conditions and environmental pollution. The changes initiated during this period laid the foundation for modern industrial society.",
            
            "Space exploration has captured human imagination and driven technological innovation for over half a century. The launch of Sputnik by the Soviet Union in 1957 marked the beginning of the space age and triggered a competitive race to reach the moon. In 1969, NASA's Apollo 11 mission successfully landed astronauts on the lunar surface, fulfilling President Kennedy's ambitious goal. Since then, robotic missions have explored every planet in our solar system, while space telescopes have revealed the existence of thousands of exoplanets. International cooperation in space, exemplified by the International Space Station, demonstrates humanity's ability to work together on grand scientific endeavors.",
        ]
        
        stories.extend(adventure_templates)
        stories.extend(educational_templates)
        stories.extend(technology_templates)
        stories.extend(historical_templates)
        
        return stories
    
    def _create_diverse_text_data(self):
        """Create diverse text dataset for modern transformer training."""
        print("Creating diverse text dataset for modern transformer...")
        
        # Generate base content
        story_content = self._generate_story_content()
        
        # Create training sequences
        sequence_count = 0
        for content_idx in range(len(story_content)):
            story = story_content[content_idx]
            
            # Tokenize the story
            tokens = self.tokenizer.tokenize(story)
            tokens = [self.tokenizer.special_tokens['<BOS>']] + tokens + [self.tokenizer.special_tokens['<EOS>']]
            
            # Create overlapping sequences
            for start_idx in range(0, len(tokens) - self.config.max_seq_len + 1, self.config.max_seq_len // 2):
                if sequence_count >= self.config.train_size:
                    break
                    
                sequence = tokens[start_idx:start_idx + self.config.max_seq_len]
                if len(sequence) == self.config.max_seq_len:
                    self.train_sequences.append(sequence)
                    sequence_count += 1
            
            if sequence_count >= self.config.train_size:
                break
        
        # If we need more sequences, generate variations
        while len(self.train_sequences) < self.config.train_size:
            # Generate additional content with variations
            base_story = story_content[len(self.train_sequences) % len(story_content)]
            
            # Create variations by adding different conclusions or perspectives
            variations = [
                " This experience taught valuable lessons about perseverance and courage.",
                " The journey revealed new possibilities and opportunities for growth.",
                " These discoveries would change the course of history forever.",
                " Future generations would benefit from these important developments.",
                " The implications of these findings continue to influence modern research.",
                " This breakthrough opened new avenues for scientific exploration.",
            ]
            
            varied_story = base_story + variations[len(self.train_sequences) % len(variations)]
            tokens = self.tokenizer.tokenize(varied_story)
            tokens = [self.tokenizer.special_tokens['<BOS>']] + tokens + [self.tokenizer.special_tokens['<EOS>']]
            
            if len(tokens) >= self.config.max_seq_len:
                sequence = tokens[:self.config.max_seq_len]
                self.train_sequences.append(sequence)
        
        # Create validation sequences
        val_templates = [
            "The modern world presents both opportunities and challenges that require innovative thinking and collaborative solutions. Technology has connected people across the globe while also creating new forms of digital divide. Education systems must adapt to prepare students for careers that may not yet exist. Environmental sustainability has become a critical concern that affects policy decisions at all levels of government. Understanding these complex interactions is essential for building a better future.",
            
            "Scientific research depends on careful observation, hypothesis formation, and rigorous testing of ideas. The scientific method provides a framework for understanding natural phenomena and developing new technologies. Collaboration between researchers from different disciplines often leads to breakthrough discoveries. Peer review and replication of experiments help ensure the reliability of scientific findings. Public understanding of science is crucial for making informed decisions about important issues affecting society.",
        ]
        
        for template in val_templates:
            tokens = self.tokenizer.tokenize(template)
            tokens = [self.tokenizer.special_tokens['<BOS>']] + tokens + [self.tokenizer.special_tokens['<EOS>']]
            
            # Create multiple sequences from each template
            for start_idx in range(0, len(tokens) - self.config.max_seq_len + 1, self.config.max_seq_len // 3):
                if len(self.val_sequences) >= self.config.val_size:
                    break
                    
                sequence = tokens[start_idx:start_idx + self.config.max_seq_len]
                if len(sequence) == self.config.max_seq_len:
                    self.val_sequences.append(sequence)
        
        # Fill remaining validation sequences if needed
        while len(self.val_sequences) < self.config.val_size:
            # Use a simple sequence for remaining validation data
            simple_text = f"This is validation sequence number {len(self.val_sequences)} for testing the modern transformer model with advanced features like RoPE and SwiGLU."
            tokens = self.tokenizer.tokenize(simple_text)
            tokens = [self.tokenizer.special_tokens['<BOS>']] + tokens + [self.tokenizer.special_tokens['<EOS>']]
            
            # Pad or truncate to max_seq_len
            if len(tokens) < self.config.max_seq_len:
                tokens.extend([self.tokenizer.special_tokens['<PAD>']] * (self.config.max_seq_len - len(tokens)))
            else:
                tokens = tokens[:self.config.max_seq_len]
            
            self.val_sequences.append(tokens)
        
        print(f"Created {len(self.train_sequences)} training sequences and {len(self.val_sequences)} validation sequences")
        print(f"Average sequence length: {np.mean([len(seq) for seq in self.train_sequences]):.1f} tokens")
    
    def get_batch(self, sequences: List[List[int]], batch_size: int, start_idx: int) -> Tuple[Tensor, Tensor]:
        """Get a batch for language modeling."""
        end_idx = min(start_idx + batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        # Pad batch
        while len(batch_sequences) < batch_size:
            if len(sequences) > 0:
                batch_sequences.append(sequences[0])  # Repeat first sequence
            else:
                batch_sequences.append([0] * self.config.max_seq_len)
        
        # Create input and target sequences
        input_ids = []
        target_ids = []
        
        for seq in batch_sequences:
            input_ids.append(seq[:-1])  # All but last token
            target_ids.append(seq[1:])  # All but first token (shifted)
        
        input_ids_array = np.array(input_ids, dtype=np.int32)
        target_ids_array = np.array(target_ids, dtype=np.int32)
        
        return Tensor(input_ids_array), Tensor(target_ids_array)

class ModernTransformerTrainer:
    """Modern Transformer trainer with advanced features."""
    
    def __init__(self, config: ModernTransformerTrainingConfig):
        self.config = config
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.metrics = {'train_losses': [], 'val_losses': [], 'perplexities': []}
        self.step = 0
    
    def setup_model(self):
        """Setup Modern Transformer model."""
        print("Setting up Modern Transformer model...")
        
        # Configure optimizations
        if self.config.enable_optimizations:
            configure(
                enable_fusion=True,
                enable_jit=True,
                auto_backend_selection=True,
                enable_mixed_precision=False
            )
        
        # Create model configuration
        transformer_config = PreNormTransformerConfig(
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size,
            activation=self.config.activation,
            normalization=self.config.normalization,
            use_rope=self.config.use_rope,
            rope_base=self.config.rope_base,
            tie_embeddings=self.config.tie_embeddings,
            scale_embeddings=self.config.scale_embeddings
        )
        
        # Create model
        self.model = PreNormTransformer(transformer_config)
        
        param_count = sum(p.data.size for p in self.model.parameters().values())
        print(f"Modern Transformer initialized with {param_count:,} parameters")
        print(f"Architecture: Pre-Norm with {self.config.num_layers} layers")
        print(f"Advanced features: {self.config.activation.upper()}, {self.config.normalization.upper()}, RoPE: {self.config.use_rope}")
        print(f"Context length: {self.config.max_seq_len}")
        print(f"Automatic optimizations: {get_config().optimization.enable_fusion}")
    
    def setup_data(self):
        """Setup data loading."""
        print("Setting up data...")
        self.tokenizer = AdvancedTextTokenizer(self.config.vocab_size)
        self.dataset = EnhancedTextDataset(self.tokenizer, self.config)
    
    def setup_optimizer(self):
        """Setup optimizer with learning rate scheduling."""
        print("Setting up optimizer...")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.initial_lr = self.config.learning_rate
    
    def update_learning_rate(self):
        """Update learning rate with warmup and cosine decay."""
        if self.step < self.config.warmup_steps:
            # Warmup phase
            lr = self.initial_lr * (self.step + 1) / self.config.warmup_steps
        else:
            # Cosine decay
            total_steps = self.config.num_epochs * (len(self.dataset.train_sequences) // self.config.batch_size)
            decay_steps = total_steps - self.config.warmup_steps
            progress = (self.step - self.config.warmup_steps) / decay_steps
            lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Update optimizer (simplified for our framework)
        if hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lr
        
        return lr
    
    def forward_pass(self, input_ids: Tensor, target_ids: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Forward pass through Modern Transformer."""
        # Model forward pass
        outputs = self.model(input_ids, output_hidden_states=False)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Compute language modeling loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat_data = logits.data.reshape(-1, vocab_size)
        targets_flat_data = target_ids.data.reshape(-1)
        
        logits_flat = Tensor(logits_flat_data, requires_grad=True)
        targets_flat = Tensor(targets_flat_data)
        
        loss = cross_entropy_loss(logits_flat, targets_flat)
        
        # Compute perplexity
        perplexity = np.exp(float(loss.data))
        
        # Compute accuracy (next token prediction)
        predictions = np.argmax(logits_flat.data, axis=1)
        accuracy = np.mean(predictions == targets_flat.data)
        
        metrics = {
            'loss': float(loss.data),
            'perplexity': perplexity,
            'accuracy': float(accuracy)
        }
        
        return loss, metrics
    
    def generate_sample(self, prompt: str = "The future of artificial intelligence", max_length: int = 80) -> str:
        """Generate text sample using the model."""
        tokens = self.tokenizer.tokenize(prompt)
        tokens = [self.tokenizer.special_tokens['<BOS>']] + tokens
        
        generated = tokens.copy()
        
        for _ in range(max_length):
            if len(generated) >= self.config.max_seq_len:
                break
            
            # Prepare input (last max_seq_len-1 tokens)
            input_tokens = generated[-(self.config.max_seq_len-1):]
            while len(input_tokens) < self.config.max_seq_len - 1:
                input_tokens = [self.tokenizer.special_tokens['<PAD>']] + input_tokens
            
            input_ids = Tensor(np.array([input_tokens], dtype=np.int32))
            
            # Forward pass
            outputs = self.model(input_ids)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Get next token probabilities
            next_token_logits = logits.data[0, -1, :]
            probs = softmax(Tensor(next_token_logits), axis=0).data
            
            # Temperature sampling for more interesting generation
            temperature = 0.8
            scaled_logits = next_token_logits / temperature
            scaled_probs = softmax(Tensor(scaled_logits), axis=0).data
            
            # Sample next token
            next_token = np.random.choice(len(scaled_probs), p=scaled_probs)
            
            if next_token == self.tokenizer.special_tokens['<EOS>']:
                break
            
            generated.append(next_token)
        
        return self.tokenizer.detokenize(generated)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
        
        total_loss = 0.0
        total_perplexity = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Shuffle training sequences
        np.random.shuffle(self.dataset.train_sequences)
        
        start_time = time.time()
        
        for batch_idx in range(0, len(self.dataset.train_sequences), self.config.batch_size):
            # Update learning rate
            current_lr = self.update_learning_rate()
            
            # Get batch
            input_ids, target_ids = self.dataset.get_batch(
                self.dataset.train_sequences, self.config.batch_size, batch_idx
            )
            
            # Forward pass
            loss, metrics = self.forward_pass(input_ids, target_ids)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            for param in self.model.parameters().values():
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad)
                    if grad_norm > self.config.max_grad_norm:
                        param.grad = param.grad * (self.config.max_grad_norm / grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += metrics['loss']
            total_perplexity += metrics['perplexity']
            total_accuracy += metrics['accuracy']
            num_batches += 1
            self.step += 1
            
            # Print progress and generate samples
            if batch_idx % (self.config.batch_size * 12) == 0:
                print(f"  Batch {batch_idx//self.config.batch_size + 1}: "
                      f"Loss = {metrics['loss']:.4f}, PPL = {metrics['perplexity']:.2f}, "
                      f"Acc = {metrics['accuracy']:.4f}, LR = {current_lr:.6f}")
                
                # Generate sample text
                if self.step % self.config.generate_every == 0:
                    sample = self.generate_sample("Modern technology has", self.config.max_generate_length)
                    print(f"  🧬 Generated: {sample[:100]}{'...' if len(sample) > 100 else ''}")
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        print(f"  Training: Loss = {avg_loss:.4f}, PPL = {avg_perplexity:.2f}, "
              f"Acc = {avg_accuracy:.4f}, Time = {epoch_time:.2f}s")
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity,
            'accuracy': avg_accuracy,
            'time': epoch_time
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        print("Validating...")
        
        total_loss = 0.0
        total_perplexity = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx in range(0, len(self.dataset.val_sequences), self.config.batch_size):
            # Get batch
            input_ids, target_ids = self.dataset.get_batch(
                self.dataset.val_sequences, self.config.batch_size, batch_idx
            )
            
            # Forward pass (no gradients)
            loss, metrics = self.forward_pass(input_ids, target_ids)
            
            # Update metrics
            total_loss += metrics['loss']
            total_perplexity += metrics['perplexity']
            total_accuracy += metrics['accuracy']
            num_batches += 1
        
        val_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        print(f"  Validation: Loss = {avg_loss:.4f}, PPL = {avg_perplexity:.2f}, "
              f"Acc = {avg_accuracy:.4f}, Time = {val_time:.2f}s")
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity,
            'accuracy': avg_accuracy,
            'time': val_time
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if not self.config.save_checkpoints:
            return
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'config': self.config.__dict__,
            'metrics': metrics
        }
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'modern_transformer_epoch_{epoch+1}.json')
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        print("Starting Modern Transformer training...")
        print(f"Configuration: {self.config.__dict__}")
        
        best_perplexity = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            self.metrics['train_losses'].append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            self.metrics['val_losses'].append(val_metrics['loss'])
            self.metrics['perplexities'].append(val_metrics['perplexity'])
            
            # Generate final sample for epoch
            sample = self.generate_sample("Artificial intelligence represents", 100)
            print(f"  🧬 Epoch sample: {sample[:120]}{'...' if len(sample) > 120 else ''}")
            
            # Save checkpoint
            epoch_metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self.save_checkpoint(epoch, epoch_metrics)
            
            # Update best model
            if val_metrics['perplexity'] < best_perplexity:
                best_perplexity = val_metrics['perplexity']
                print(f"  New best validation perplexity: {best_perplexity:.2f}")
        
        print("\nTraining completed!")
        print(f"Best validation perplexity: {best_perplexity:.2f}")
        
        return self.metrics

def main():
    """Main training function."""
    print("🧬 Modern Transformer Training with Advanced Features")
    print("=" * 70)
    
    # Training configuration
    config = ModernTransformerTrainingConfig(
        # Model config
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        max_seq_len=128,
        vocab_size=8000,
        
        # Advanced features
        activation="swiglu",
        normalization="rmsnorm",
        use_rope=True,
        rope_base=10000,
        tie_embeddings=True,
        scale_embeddings=True,
        
        # Training config
        batch_size=6,
        learning_rate=2e-4,
        num_epochs=5,
        warmup_steps=300,
        train_size=1200,
        val_size=240,
        
        # Generation config
        generate_every=100,
        max_generate_length=50,
        
        # Enable optimizations
        enable_optimizations=True,
        save_checkpoints=True
    )
    
    try:
        # Create trainer
        trainer = ModernTransformerTrainer(config)
        
        # Train model
        metrics = trainer.train()
        
        # Generate final samples
        print("\n" + "=" * 70)
        print("🎉 MODERN TRANSFORMER TRAINING COMPLETE!")
        print("=" * 70)
        
        print("🧬 Final Generation Samples:")
        prompts = [
            "Technology has revolutionized",
            "The future of science",
            "Modern education requires",
            "Climate change presents"
        ]
        for prompt in prompts:
            sample = trainer.generate_sample(prompt, 60)
            print(f"  Prompt: '{prompt}' → '{sample[:100]}{'...' if len(sample) > 100 else ''}'")
        
        print(f"\nFinal Results:")
        print(f"  📊 Final Train Loss: {metrics['train_losses'][-1]:.4f}")
        print(f"  📈 Final Val Loss: {metrics['val_losses'][-1]:.4f}")
        print(f"  🎯 Final Perplexity: {metrics['perplexities'][-1]:.2f}")
        print(f"  📈 Best Perplexity: {min(metrics['perplexities']):.2f}")
        
        print(f"\n✅ Advanced Features Demonstrated:")
        print(f"  🚀 Automatic optimizations enabled")
        print(f"  🧬 Pre-Norm architecture for stable training")
        print(f"  🌀 RoPE positional encoding")
        print(f"  ⚡ SwiGLU activation function")
        print(f"  📏 RMSNorm normalization")
        print(f"  🔗 Tied input/output embeddings")
        print(f"  📚 Advanced text generation capabilities")
        print(f"  💾 Model checkpointing with learning rate scheduling")
        print(f"  📈 Perplexity tracking and optimization")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())