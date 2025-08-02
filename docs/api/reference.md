# API Reference - Neural Architecture Framework

Complete API documentation for the neural network implementation.

## Core Components

### Tensor Class

The foundation of the neural network - a tensor with automatic differentiation.

```python
class Tensor:
    """Tensor with automatic differentiation support."""
```

#### Constructor
```python
Tensor(data: Union[np.ndarray, list, float], requires_grad: bool = False)
```

**Parameters:**
- `data`: Input data as NumPy array, list, or scalar
- `requires_grad`: Whether to track gradients for this tensor

**Example:**
```python
# Create tensors
x = Tensor([[1, 2, 3]], requires_grad=True)
y = Tensor([4.5], requires_grad=False)
```

#### Properties
```python
@property
def shape(self) -> Tuple[int, ...]:
    """Get tensor shape."""
    return self.data.shape

@property  
def data(self) -> np.ndarray:
    """Get underlying NumPy array."""

@property
def grad(self) -> Optional[np.ndarray]:
    """Get accumulated gradients."""

@property
def requires_grad(self) -> bool:
    """Check if gradients are tracked."""
```

#### Methods
```python
def zero_grad(self) -> None:
    """Reset gradients to None."""

def backward(self, gradient: Optional[np.ndarray] = None) -> None:
    """Accumulate gradients with optional input gradient."""
```

**Example:**
```python
x = Tensor([1, 2, 3], requires_grad=True)
x.backward(np.array([0.1, 0.2, 0.3]))
print(x.grad)  # [0.1 0.2 0.3]
x.zero_grad()
print(x.grad)  # None
```

---

## Tensor Operations

### Mathematical Operations

#### **Addition**
```python
def add(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise addition with broadcasting support."""
```

**Example:**
```python
a = Tensor([[1, 2]], requires_grad=True)
b = Tensor([[3, 4]], requires_grad=True)
c = add(a, b)  # [[4, 6]]
```

#### **Multiplication**
```python
def mul(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise multiplication with broadcasting support."""
```

**Example:**
```python
a = Tensor([2, 3], requires_grad=True)
b = Tensor([4, 5], requires_grad=True)
c = mul(a, b)  # [8, 15]
```

#### **Matrix Multiplication**
```python
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication with gradient support."""
```

**Example:**
```python
a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([[5, 6], [7, 8]], requires_grad=True)
c = matmul(a, b)  # [[19, 22], [43, 50]]
```

#### **Mean Pooling**
```python
def mean_pool(x: Tensor, axis: int = 1) -> Tensor:
    """Mean pooling with gradient support."""
```

**Example:**
```python
x = Tensor([[[1, 2, 3], [4, 5, 6]]], requires_grad=True)  # (1, 2, 3)
pooled = mean_pool(x, axis=1)  # (1, 3) - average across axis 1
```

### **Activation Functions**

#### **ReLU**
```python
def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit activation function."""
```

**Mathematical Definition:** `f(x) = max(0, x)`

**Example:**
```python
x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
y = relu(x)  # [0, 0, 0, 1, 2]
```

#### **Softmax**
```python
def softmax(x: Tensor) -> Tensor:
    """Softmax activation function with numerical stability."""
```

**Mathematical Definition:** `softmax(x_i) = exp(x_i) / sum(exp(x_j))`

**Example:**
```python
x = Tensor([[1, 2, 3]], requires_grad=True)
y = softmax(x)  # [[0.0900, 0.2447, 0.6652]] (approximately)
```

---

## 🧱 **Neural Network Layers**

### **Linear Layer**

Fully connected layer with learnable weights and biases.

```python
class Linear:
    """Fully connected linear layer."""
    
    def __init__(self, in_features: int, out_features: int):
        """Initialize linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
```

#### **Mathematical Definition**
`y = xW + b` where:
- `x`: Input tensor `(batch_size, in_features)`
- `W`: Weight matrix `(in_features, out_features)`
- `b`: Bias vector `(out_features,)`
- `y`: Output tensor `(batch_size, out_features)`

#### Methods
```python
def __call__(self, x: Tensor) -> Tensor:
    """Forward pass through the layer."""

def parameters(self) -> Dict[str, Tensor]:
    """Get layer parameters (weight and bias)."""
```

**Example:**
```python
layer = Linear(3, 2)  # 3 inputs -> 2 outputs
x = Tensor([[1, 2, 3]], requires_grad=True)  # batch_size=1
y = layer(x)  # Shape: (1, 2)

# Get parameters
params = layer.parameters()
# params = {'weight': Tensor(...), 'bias': Tensor(...)}
```

### **Embedding Layer**

Token embedding layer for discrete inputs.

```python
class Embedding:
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        """Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
        """
```

#### **Mathematical Definition**
For input indices `idx`, returns `embedding_matrix[idx]`

#### Methods
```python
def __call__(self, indices: np.ndarray) -> Tensor:
    """Embed input indices."""

def parameters(self) -> Dict[str, Tensor]:
    """Get embedding parameters (weight matrix)."""
```

**Example:**
```python
embedding = Embedding(vocab_size=1000, embed_dim=128)
indices = np.array([[0, 1, 2]])  # Shape: (1, 3)
embedded = embedding(indices)    # Shape: (1, 3, 128)

# Get parameters
params = embedding.parameters()
# params = {'weight': Tensor(shape=(1000, 128))}
```

---

## ⚡ **Optimizers**

### **Adam Optimizer**

Adaptive Moment Estimation optimizer with momentum and bias correction.

```python
class Adam:
    """Adam optimizer with gradient clipping."""
    
    def __init__(self, parameters: Dict[str, Tensor], lr: float = 0.01):
        """Initialize Adam optimizer.
        
        Args:
            parameters: Dictionary of parameters to optimize
            lr: Learning rate
        """
```

#### **Mathematical Definition**
Adam update rule:
1. `m_t = β₁ * m_{t-1} + (1 - β₁) * g_t`
2. `v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²`
3. `m̂_t = m_t / (1 - β₁ᵗ)`
4. `v̂_t = v_t / (1 - β₂ᵗ)`  
5. `θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)`

Where:
- `β₁ = 0.9` (momentum decay)
- `β₂ = 0.999` (RMSprop decay)
- `ε = 1e-8` (numerical stability)

#### Methods
```python
def step(self) -> None:
    """Update parameters using computed gradients."""

def zero_grad(self) -> None:
    """Zero all parameter gradients."""
```

**Example:**
```python
# Collect model parameters
model_params = {}
model_params.update(layer1.parameters())
model_params.update(layer2.parameters())

# Create optimizer
optimizer = Adam(model_params, lr=0.001)

# Training step
loss = compute_loss(...)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

## 📝 **Text Processing Utilities**

### **Vocabulary Creation**
```python
def create_text_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create character-level vocabulary from text.
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (char_to_idx, idx_to_char) mappings
    """
```

**Example:**
```python
text = "hello world"
char_to_idx, idx_to_char = create_text_vocab(text)
# char_to_idx = {' ': 0, 'd': 1, 'e': 2, 'h': 3, 'l': 4, 'o': 5, 'r': 6, 'w': 7}
# idx_to_char = {0: ' ', 1: 'd', 2: 'e', 3: 'h', 4: 'l', 5: 'o', 6: 'r', 7: 'w'}
```

### **Sequence Generation**
```python
def text_to_sequences(text: str, seq_len: int, char_to_idx: Dict[str, int]) -> np.ndarray:
    """Convert text to training sequences.
    
    Args:
        text: Input text string
        seq_len: Length of each sequence
        char_to_idx: Character to index mapping
        
    Returns:
        Array of sequences with shape (num_sequences, seq_len + 1)
    """
```

**Example:**
```python
text = "hello"
char_to_idx = {'h': 0, 'e': 1, 'l': 2, 'o': 3}
sequences = text_to_sequences(text, seq_len=2, char_to_idx=char_to_idx)
# sequences = [[0, 1, 2], [1, 2, 2], [2, 2, 3]]  # "he" -> "l", "el" -> "l", "ll" -> "o"
```

---

## 🔄 **Gradient Utilities**

### **Gradient Propagation**
```python
def propagate_gradients(tensor: Tensor) -> None:
    """Propagate gradients through computation graph.
    
    Args:
        tensor: Output tensor to propagate gradients from
    """
```

**Example:**
```python
# Forward pass
x = Tensor([[1, 2]], requires_grad=True)
y = relu(x)
z = mul(y, Tensor([2, 3]))

# Backward pass
z.backward(np.array([[1, 1]]))
propagate_gradients(z)  # Propagates gradients to x
```

---

## 🧪 **Complete Usage Examples**

### **Simple Neural Network**

```python
from neural_arch import *

class SimpleNN:
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, vocab_size)
    
    def forward(self, x: np.ndarray) -> Tensor:
        # Embed tokens
        embedded = self.embedding(x)  # (batch, seq, embed)
        
        # Pool sequence dimension
        pooled = mean_pool(embedded, axis=1)  # (batch, embed)
        
        # Feed-forward layers
        hidden = relu(self.linear1(pooled))  # (batch, hidden)
        output = self.linear2(hidden)        # (batch, vocab)
        
        return softmax(output)
    
    def parameters(self) -> Dict[str, Tensor]:
        params = {}
        params.update(self.embedding.parameters())
        params.update(self.linear1.parameters())
        params.update(self.linear2.parameters())
        return params

# Training
vocab_size = 1000
model = SimpleNN(vocab_size)
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model.forward(input_sequences)
    
    # Compute loss (cross-entropy)
    targets = target_sequences
    loss = compute_cross_entropy_loss(outputs, targets)
    
    # Backward pass
    loss.backward()
    if hasattr(loss, '_backward'):
        loss._backward()
    
    # Update parameters
    optimizer.step()
    optimizer.zero_grad()
```

### **Text Generation Pipeline**

```python
def train_text_model(text: str, epochs: int = 100):
    """Train a text generation model."""
    
    # Prepare data
    char_to_idx, idx_to_char = create_text_vocab(text)
    sequences = text_to_sequences(text, seq_len=10, char_to_idx=char_to_idx)
    
    # Create model
    vocab_size = len(char_to_idx)
    model = SimpleNN(vocab_size, embed_dim=64, hidden_dim=128)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data
        np.random.shuffle(sequences)
        
        total_loss = 0
        for batch in create_batches(sequences, batch_size=32):
            inputs = batch[:, :-1]  # All but last
            targets = batch[:, -1]  # Last character
            
            # Forward pass
            outputs = model.forward(inputs)
            
            # Compute loss
            loss = cross_entropy_loss(outputs, targets)
            total_loss += loss
            
            # Backward pass
            loss.backward()
            if hasattr(loss, '_backward'):
                loss._backward()
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    return model, char_to_idx, idx_to_char

def generate_text(model, prompt: str, length: int, char_to_idx, idx_to_char):
    """Generate text using trained model."""
    
    context = [char_to_idx.get(c, 0) for c in prompt]
    generated = prompt
    
    for _ in range(length):
        # Predict next character
        inputs = np.array([context[-10:]])  # Last 10 characters
        probs = model.forward(inputs)
        
        # Sample from distribution
        next_idx = np.random.choice(len(probs.data[0]), p=probs.data[0])
        next_char = idx_to_char[next_idx]
        
        # Update context and output
        context.append(next_idx)
        generated += next_char
    
    return generated

# Usage
text = "Your training text here..."
model, char_to_idx, idx_to_char = train_text_model(text)
generated = generate_text(model, "Hello", 100, char_to_idx, idx_to_char)
print(generated)
```

---

## 🎯 **Performance Considerations**

### **Memory Management**
- Always call `optimizer.zero_grad()` after each training step
- Use `tensor.zero_grad()` to free gradient memory when needed
- Be aware of tensor shapes to avoid unnecessary memory allocation

### **Numerical Stability**
- Gradients are automatically clipped to prevent explosion
- Softmax uses numerically stable implementation
- Adam optimizer includes epsilon for numerical stability

### **Performance Tips**
- Use vectorized operations when possible
- Prefer matrix multiplication over element-wise operations for large tensors  
- Monitor memory usage with large models or long sequences

---

## 🔍 **Debugging and Introspection**

### **Checking Gradients**
```python
# Verify gradients are computed
x = Tensor([1, 2, 3], requires_grad=True)
y = relu(x)
y.backward(np.array([1, 1, 1]))

print(f"x.grad: {x.grad}")  # Should show gradients
print(f"y.grad: {y.grad}")  # Should show gradients
```

### **Parameter Inspection**
```python
# Check parameter shapes and values
model = SimpleNN(vocab_size=100)
for name, param in model.parameters().items():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
```

### **Gradient Flow Verification**
```python
# Test gradient flow through model
model = SimpleNN(vocab_size=50)
x = np.array([[0, 1, 2]])  # Sample input
y = model.forward(x)

# Backward pass
y.backward(np.ones_like(y.data))
if hasattr(y, '_backward'):
    y._backward()

# Check all parameters have gradients
for name, param in model.parameters().items():
    if param.grad is None:
        print(f"WARNING: {name} has no gradient!")
    else:
        print(f"{name}: gradient norm = {np.linalg.norm(param.grad):.6f}")
```

---

## 📊 **Type Information**

All functions and classes include complete type hints for better development experience:

```python
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Type aliases for clarity
TensorLike = Union[np.ndarray, list, float]
ParameterDict = Dict[str, Tensor]
VocabMapping = Tuple[Dict[str, int], Dict[int, str]]
```

---

## 🚀 **Quick Reference**

### **Common Patterns**

```python
# Create tensor with gradients
x = Tensor(data, requires_grad=True)

# Forward pass
y = relu(matmul(x, weight) + bias)

# Backward pass  
y.backward(gradient)
if hasattr(y, '_backward'):
    y._backward()

# Parameter update
optimizer.step()
optimizer.zero_grad()
```

### **Import Everything**
```python
from neural_arch import (
    # Core components
    Tensor, Parameter,
    
    # Neural network layers
    Linear, Embedding, LayerNorm,
    MultiHeadAttention, TransformerBlock, TransformerDecoderBlock,
    
    # Optimizers
    Adam,
    
    # Operations
    add, mul, matmul, relu, softmax, mean_pool,
    
    # Utilities
    create_text_vocab, text_to_sequences, propagate_gradients
)

# Translation components (from examples)
from examples.translation.vocabulary import Vocabulary
from examples.translation.model_v2 import TranslationTransformer, PositionalEncoding
```

---

## 🤖 **Transformer Components**

### **Multi-Head Attention**

The core attention mechanism for transformer architectures.

```python
class MultiHeadAttention:
    """Multi-head scaled dot-product attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
```

#### **Mathematical Definition**
Multi-head attention computes:
1. `Q = X @ W_Q`, `K = X @ W_K`, `V = X @ W_V`
2. Split Q, K, V into `num_heads` pieces
3. For each head: `Attention(Q, K, V) = softmax(QK^T / √d_k) @ V`
4. Concatenate heads and project: `output = Concat(head_1, ..., head_h) @ W_O`

#### **Forward Method**
```python
def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
    """Apply multi-head attention.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        mask: Optional attention mask (1 = mask, 0 = keep)
        
    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
```

**Example:**
```python
# Create multi-head attention
attn = MultiHeadAttention(d_model=512, num_heads=8)

# Input sequence
x = Tensor(np.random.randn(2, 10, 512))

# Apply attention
output = attn(x)  # Shape: (2, 10, 512)

# With masking (e.g., padding mask)
mask = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])  # Mask positions 4-9
output = attn(x, mask=mask)
```

### **Layer Normalization**

Normalization across features for stable training.

```python
class LayerNorm:
    """Layer normalization with learnable parameters."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """Initialize layer normalization.
        
        Args:
            normalized_shape: Size of the feature dimension
            eps: Small constant for numerical stability
        """
```

**Example:**
```python
# Create layer normalization
norm = LayerNorm(512)

# Normalize features
x = Tensor(np.random.randn(2, 10, 512) * 5 + 2)  # High variance input
normalized = norm(x)
# Output has mean ≈ 0, std ≈ 1 along feature dimension
```

### **Transformer Block**

Complete transformer encoder block with attention and feed-forward.

```python
class TransformerBlock:
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
```

**Architecture:**
1. Multi-head self-attention with residual connection
2. Layer normalization
3. Position-wise feed-forward network with residual connection
4. Layer normalization

**Example:**
```python
# Create transformer block
block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)

# Process sequence
x = Tensor(np.random.randn(2, 10, 512))
output = block(x)  # Shape: (2, 10, 512)

# Stack multiple blocks
blocks = [TransformerBlock(512, 8, 2048) for _ in range(6)]
for block in blocks:
    x = block(x)
```

### **Transformer Decoder Block**

Transformer decoder with self-attention and cross-attention.

```python
class TransformerDecoderBlock:
    """Transformer decoder block with cross-attention."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        """Initialize decoder block."""
```

**Architecture:**
1. Masked self-attention (causal)
2. Cross-attention to encoder output
3. Position-wise feed-forward network
4. Residual connections and layer normalization

**Example:**
```python
# Create decoder block
decoder = TransformerDecoderBlock(d_model=512, num_heads=8, d_ff=2048)

# Decoder input and encoder output
tgt = Tensor(np.random.randn(2, 8, 512))   # Target sequence
memory = Tensor(np.random.randn(2, 10, 512))  # Encoder output

# Apply decoder
output = decoder(tgt, memory)  # Shape: (2, 8, 512)

# With causal mask for autoregressive generation
tgt_mask = np.triu(np.ones((8, 8)), k=1)  # Upper triangular mask
output = decoder(tgt, memory, tgt_mask=tgt_mask)
```

### **Positional Encoding**

Sinusoidal position embeddings for sequence order information.

```python
class PositionalEncoding:
    """Fixed sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
```

**Example:**
```python
# Create positional encoding
pos_enc = PositionalEncoding(d_model=512, max_len=1000)

# Add to embeddings
embeddings = Tensor(np.random.randn(2, 100, 512))
with_positions = pos_enc(embeddings)
```

---

## 🌐 **Translation Components**

### **Vocabulary**

Token vocabulary management for translation tasks.

```python
class Vocabulary:
    """Vocabulary for text tokenization."""
    
    def __init__(self, language: str):
        """Initialize vocabulary with special tokens."""
```

**Special Tokens:**
- `<PAD>`: Padding token (index 0)
- `<SOS>`: Start of sequence (index 1)
- `<EOS>`: End of sequence (index 2)
- `<UNK>`: Unknown token (index 3)

**Methods:**
```python
def add_sentence(self, sentence: str) -> None:
    """Add all words in sentence to vocabulary."""

def encode(self, sentence: str, max_length: Optional[int] = None) -> List[int]:
    """Convert sentence to token indices."""

def decode(self, indices: List[int], remove_special: bool = True) -> str:
    """Convert indices back to sentence."""

def save(self, filepath: str) -> None:
    """Save vocabulary to JSON file."""

@classmethod
def load(cls, filepath: str) -> 'Vocabulary':
    """Load vocabulary from JSON file."""
```

**Example:**
```python
# Create vocabularies
src_vocab = Vocabulary("english")
tgt_vocab = Vocabulary("spanish")

# Build vocabulary
src_vocab.add_sentence("hello world")
tgt_vocab.add_sentence("hola mundo")

# Encode/decode
indices = src_vocab.encode("hello world", max_length=10)
# [4, 5, 2, 0, 0, 0, 0, 0, 0, 0]  # words + EOS + padding

text = src_vocab.decode(indices, remove_special=True)
# "hello world"
```

### **Translation Transformer**

Complete encoder-decoder transformer for translation.

```python
class TranslationTransformer:
    """Transformer model for sequence-to-sequence translation."""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 3,
                 d_ff: int = 1024, dropout: float = 0.1, max_len: int = 5000):
        """Initialize translation model."""
```

**Methods:**
```python
def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
    """Forward pass for training.
    
    Args:
        src: Source sequence indices (batch_size, src_len)
        tgt: Target sequence indices (batch_size, tgt_len)
        
    Returns:
        Output logits (batch_size, tgt_len, tgt_vocab_size)
    """

def generate(self, src: Tensor, max_length: int = 50,
             sos_idx: int = 1, eos_idx: int = 2,
             temperature: float = 1.0) -> List[int]:
    """Generate translation using greedy/sampling decoding."""
```

**Example:**
```python
# Create model
model = TranslationTransformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=256,
    n_heads=8,
    n_layers=3
)

# Training
src = Tensor(np.array([[1, 4, 5, 2, 0]]))  # Source with padding
tgt_in = Tensor(np.array([[1, 6, 7, 8]]))  # Target input
output = model(src, tgt_in)  # (1, 4, 10000)

# Generation
translation = model.generate(
    src,
    max_length=20,
    temperature=0.8
)
```

This API reference covers all public interfaces of the neural architecture implementation. For more examples and advanced usage, see the comprehensive test suite in the `tests/` directory.

---

**The API is designed to be simple, consistent, and educational while maintaining production-ready performance and reliability.** 🧠✨