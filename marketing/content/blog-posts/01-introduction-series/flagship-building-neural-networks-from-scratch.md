# Building Neural Networks From Scratch: The Complete Journey

*How I spent 6 months building a production-ready ML framework using only NumPy—and what it taught me about the fundamentals of deep learning*

---

## The Challenge That Started It All

*"How does a neural network actually work?"*

This question haunted me for months. Sure, I could use PyTorch and TensorFlow to build models, but I felt like I was operating a complex machine without understanding the engine. Every time I encountered a bug or unexpected behavior, I was lost. I was a passenger in my own ML journey.

So I made a decision that would consume the next 6 months of my life: **I would build a complete neural network framework from scratch, using only NumPy.**

The result? [Neural Architecture](https://github.com/fenilsonani/neural-network-from-scratch) - a production-ready ML framework with 700+ tests, 6 complete model architectures, and the most comprehensive "from scratch" implementation I've ever seen.

Here's the story of that journey, the challenges I faced, and what I learned about the beautiful complexity hidden beneath our favorite ML frameworks.

## Why "From Scratch" Matters (More Than You Think)

Before we dive into the technical journey, let me address the elephant in the room: *"Why build from scratch when PyTorch exists?"*

### The Black Box Problem

Modern ML frameworks are incredible tools, but they're also black boxes. When you write:

```python
loss = nn.CrossEntropyLoss()(outputs, targets)
loss.backward()
optimizer.step()
```

What actually happens? How does `backward()` compute gradients? What's the mathematical relationship between your loss and the weight updates? Most developers (including past me) couldn't answer these questions.

### The Debugging Crisis

This lack of understanding becomes critical when things go wrong:
- Why is my model not converging?
- Why are my gradients exploding?
- Why does changing the learning rate have such dramatic effects?
- Why does my attention layer produce NaN values?

Without understanding the fundamentals, debugging becomes guesswork.

### The Learning Paradox

The most advanced ML practitioners I know all have one thing in common: they understand the mathematics and implementation details of their tools. They didn't learn this from using high-level APIs—they learned it by implementing algorithms themselves.

Building from scratch isn't about reinventing the wheel; it's about understanding how the wheel works.

## The Architecture: 6 Months of Design Decisions

Building a framework isn't just about implementing algorithms—it's about creating a coherent system where everything works together. Here's how I structured Neural Architecture:

### Core Tensor System

The foundation of any ML framework is its tensor system. I needed automatic differentiation, device management, and efficient operations:

```python
# The heart of the framework
class Tensor:
    def __init__(self, data, requires_grad=False, device=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.device = device or Device.default()
        
        # The magic: computation graph for backpropagation
        self._backward_fn = None
        self._prev_tensors = []
        
    def backward(self, gradient=None):
        # Automatic differentiation implementation
        if gradient is None:
            gradient = np.ones_like(self.data)
            
        if self.requires_grad:
            if self.grad is None:
                self.grad = gradient
            else:
                self.grad += gradient
                
        # Recursive backpropagation
        if self._backward_fn:
            self._backward_fn(gradient)
```

This simple class encapsulates years of research in automatic differentiation. Every operation creates a computation graph, and `backward()` traverses it to compute gradients.

### Neural Network Layers

With tensors in place, I could build neural network layers. Each layer needed to:
1. Transform input tensors
2. Maintain learnable parameters
3. Support gradient computation

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier initialization for stable training
        self.weight = Parameter(np.random.randn(in_features, out_features) * 
                               np.sqrt(2.0 / in_features))
        self.bias = Parameter(np.zeros(out_features))
        
    def forward(self, x):
        # Matrix multiplication with automatic gradient tracking
        return matmul(x, self.weight) + self.bias
```

The beauty of this design is that each layer is just a function with learnable parameters. The automatic differentiation system handles all the gradient computations.

### The Transformer Revolution

Implementing attention mechanisms was where things got really interesting. The mathematical beauty of attention—computing relationships between all positions in a sequence—requires careful implementation:

```python
class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # Transform to multiple heads
        Q = self.w_q(query).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.w_k(key).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.w_v(value).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Scaled dot-product attention
        scores = matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = softmax(scores, dim=-1)
        context = matmul(attention_weights, V)
        
        # Concatenate heads and project
        context = context.reshape(batch_size, seq_len, d_model)
        return self.w_o(context)
```

This implementation taught me why attention is so powerful: it allows every position to directly interact with every other position, creating rich representations that capture long-range dependencies.

## The Six Model Challenge

To prove the framework's capabilities, I implemented six complete model architectures:

### 1. GPT-2: The Art of Autoregressive Generation

Building GPT-2 taught me about language modeling and the challenges of autoregressive generation:

```python
class GPT2(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and positional embeddings
        self.token_embedding = Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = Embedding(config.n_ctx, config.n_embd)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layer)]
        
        # Language modeling head
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = Linear(config.n_embd, config.vocab_size)
        
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        
        # Create embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = Tensor(np.arange(seq_len))
        pos_emb = self.pos_embedding(pos_ids)
        
        x = token_emb + pos_emb
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
```

**Key Learning**: Language models are sequence predictors. Every token prediction is conditioned on all previous tokens, making generation an inherently sequential process.

### 2. Vision Transformer: Images as Sequences

ViT revolutionized computer vision by treating images as sequences of patches:

```python
class VisionTransformer(Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: convert image patches to tokens
        self.patch_embed = Linear(patch_size * patch_size * 3, embed_dim)
        
        # Class token and positional embeddings
        self.cls_token = Parameter(np.zeros((1, 1, embed_dim)))
        self.pos_embed = Parameter(np.zeros((1, self.num_patches + 1, embed_dim)))
        
        # Transformer encoder
        self.blocks = [TransformerBlock(embed_dim, num_heads) 
                      for _ in range(depth)]
        
        # Classification head
        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Convert image to patches
        patches = self.extract_patches(x)  # [B, num_patches, patch_dim]
        
        # Embed patches
        x = self.patch_embed(patches)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = concatenate([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Forward through transformer
        for block in self.blocks:
            x = block(x)
            
        # Use class token for classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        
        return self.head(cls_token_final)
```

**Key Learning**: The patch embedding is the bridge between computer vision and NLP. By treating image patches as tokens, ViT can leverage the power of transformer architectures for vision tasks.

### 3. BERT: Bidirectional Understanding

BERT's bidirectional nature required implementing masked language modeling:

```python
class BERT(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings: token + position + segment
        self.embeddings = BERTEmbeddings(config)
        
        # Encoder layers
        self.encoder = BERTEncoder(config)
        
        # Pooler for classification tasks
        self.pooler = BERTPooler(config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Create embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Encode with attention
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        
        # Pool for classification
        pooled_output = self.pooler(sequence_output)
        
        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output
        }
```

**Key Learning**: Bidirectional context is incredibly powerful. Unlike GPT-2, which only sees previous tokens, BERT sees the entire sequence, enabling much richer representations.

### 4. CLIP: Bridging Vision and Language

CLIP's contrastive learning approach required implementing a sophisticated training objective:

```python
class CLIP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Vision encoder (ViT)
        self.vision_encoder = VisionTransformer(
            img_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
        )
        
        # Text encoder (Transformer)
        self.text_encoder = TextTransformer(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_layers=config.text_layers,
        )
        
        # Projection heads
        self.vision_projection = Linear(config.embed_dim, config.projection_dim)
        self.text_projection = Linear(config.embed_dim, config.projection_dim)
        
        # Learnable temperature parameter
        self.logit_scale = Parameter(np.log(1 / 0.07))
        
    def forward(self, images, texts):
        # Encode images and texts
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Project to shared space
        image_embeds = self.vision_projection(image_features)
        text_embeds = self.text_projection(text_features)
        
        # Normalize embeddings
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)
        
        # Compute similarity matrix
        logit_scale = np.exp(self.logit_scale.data)
        logits_per_image = logit_scale * matmul(image_embeds, text_embeds.T)
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text
```

**Key Learning**: Contrastive learning is about learning representations where similar items are close and dissimilar items are far apart. CLIP learns a shared embedding space where matching image-text pairs have high similarity.

### 5. Modern Transformer: RoPE, SwiGLU, and RMSNorm

The latest transformer improvements required implementing cutting-edge components:

```python
class ModernTransformerBlock(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # RMSNorm instead of LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        
        # Attention with RoPE
        self.self_attn = MultiHeadAttentionWithRoPE(config)
        
        # SwiGLU instead of standard MLP
        self.mlp = SwiGLUMLP(config)
        
    def forward(self, x, attention_mask=None):
        # Pre-norm architecture
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, x, x, attention_mask)
        x = residual + x
        
        # MLP with pre-norm
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class RMSNorm(Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(hidden_size))
        self.eps = eps
        
    def forward(self, x):
        # RMS normalization
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / rms
```

**Key Learning**: Modern improvements like RoPE (positional encoding), SwiGLU (activation function), and RMSNorm (normalization) make transformers more efficient and effective. Each component is the result of careful research and empirical validation.

### 6. ResNet: The Power of Residual Connections

ResNet taught me about the challenges of training deep networks:

```python
class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = Sequential([
                Conv2d(in_channels, out_channels, 1, stride),
                BatchNorm2d(out_channels)
            ])
            
    def forward(self, x):
        residual = x
        
        # Main path
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        if self.skip is not None:
            residual = self.skip(x)
            
        # Add residual
        out = out + residual
        out = relu(out)
        
        return out
```

**Key Learning**: Residual connections solve the vanishing gradient problem by providing direct paths for gradients to flow through deep networks. The skip connection is one of the most important innovations in deep learning.

## The Testing Philosophy: 700+ Tests for Production Quality

Testing ML code is notoriously difficult. How do you test something that learns? Here's how I approached it:

### 1. Mathematical Correctness

Every operation needed mathematical verification:

```python
def test_matrix_multiplication_gradients():
    """Test that matrix multiplication gradients are mathematically correct."""
    # Create test tensors
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    
    # Forward pass
    c = matmul(a, b)
    loss = sum(c)
    
    # Backward pass
    loss.backward()
    
    # Mathematical verification
    # ∂loss/∂a = b^T, ∂loss/∂b = a^T
    expected_a_grad = b.data.T
    expected_b_grad = a.data.T
    
    assert np.allclose(a.grad, expected_a_grad), f"Expected {expected_a_grad}, got {a.grad}"
    assert np.allclose(b.grad, expected_b_grad), f"Expected {expected_b_grad}, got {b.grad}"
```

### 2. Gradient Checking

Numerical gradient checking verified automatic differentiation:

```python
def test_gradient_checking():
    """Use finite differences to verify gradients."""
    def finite_difference_gradient(func, x, h=1e-5):
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_pos = x.copy()
            x_neg = x.copy()
            x_pos.flat[i] += h
            x_neg.flat[i] -= h
            grad.flat[i] = (func(x_pos) - func(x_neg)) / (2 * h)
        return grad
    
    # Test function: f(x) = x^2
    x = Tensor([3.0], requires_grad=True)
    y = x * x
    y.backward()
    
    # Numerical gradient
    def f(x_val):
        return x_val[0] ** 2
    
    numerical_grad = finite_difference_gradient(f, x.data)
    
    assert np.allclose(x.grad, numerical_grad, atol=1e-4)
```

### 3. Integration Tests

Full model training tests ensured everything worked together:

```python
def test_gpt2_training_integration():
    """Test complete GPT-2 training pipeline."""
    # Create small model for testing
    config = GPT2Config(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_ctx=16
    )
    
    model = GPT2(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Create dummy data
    batch_size = 2
    seq_len = 16
    input_ids = Tensor(np.random.randint(0, config.vocab_size, (batch_size, seq_len)))
    targets = Tensor(np.random.randint(0, config.vocab_size, (batch_size, seq_len)))
    
    # Training step
    logits = model(input_ids)
    loss = cross_entropy_loss(logits.view(-1, config.vocab_size), targets.view(-1))
    
    initial_loss = loss.data
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Second forward pass
    logits = model(input_ids)
    loss = cross_entropy_loss(logits.view(-1, config.vocab_size), targets.view(-1))
    
    # Loss should decrease (or at least change)
    assert loss.data != initial_loss, "Loss should change after optimization"
```

### 4. Performance Tests

Benchmarking ensured the framework was reasonably efficient:

```python
def test_large_matrix_multiplication_performance():
    """Benchmark matrix multiplication for large tensors."""
    import time
    
    size = 1000
    a = Tensor(np.random.randn(size, size))
    b = Tensor(np.random.randn(size, size))
    
    # Time the operation
    start_time = time.time()
    c = matmul(a, b)
    end_time = time.time()
    
    elapsed = end_time - start_time
    
    # Should complete in reasonable time
    assert elapsed < 5.0, f"Matrix multiplication too slow: {elapsed:.2f}s"
    
    # Verify correctness
    expected = np.matmul(a.data, b.data)
    assert np.allclose(c.data, expected), "Matrix multiplication incorrect"
```

## The Challenges: What I Learned the Hard Way

### Challenge 1: Memory Management

NumPy arrays can consume massive amounts of memory. I learned to:
- Use views instead of copies when possible
- Implement proper gradient accumulation
- Clear gradients after each optimization step
- Use float32 instead of float64 for large models

### Challenge 2: Numerical Stability

Floating-point arithmetic is tricky. I encountered:
- **Gradient Explosion**: Solved with gradient clipping
- **Vanishing Gradients**: Addressed with proper initialization
- **Softmax Overflow**: Fixed with numerical stability tricks
- **NaN Propagation**: Added extensive debugging and validation

```python
def stable_softmax(x, axis=-1):
    """Numerically stable softmax implementation."""
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    
    # Compute softmax
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    return exp_x / sum_exp
```

### Challenge 3: Debugging Complex Models

Debugging a transformer with attention mechanisms is incredibly challenging. I developed strategies:
- **Layer-by-layer validation**: Test each component independently
- **Gradient inspection**: Check gradient magnitudes and distributions
- **Attention visualization**: Plot attention weights to understand model behavior
- **Intermediate outputs**: Save and inspect hidden states

### Challenge 4: Performance Optimization

Pure NumPy isn't fast enough for large models. I optimized through:
- **Vectorization**: Eliminate Python loops where possible
- **Memory layout**: Ensure cache-friendly data access patterns
- **Batch processing**: Process multiple examples simultaneously
- **GPU acceleration**: Added CUDA and MPS backend support

## The Unexpected Benefits

Building from scratch taught me things I never expected:

### 1. Deep Mathematical Understanding

I now understand the mathematics behind every operation. When I see a transformer attention formula, I know exactly how it works:

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

This isn't just a formula—I know why we divide by √d_k (variance scaling), why we use softmax (probability distribution), and how gradients flow through each component.

### 2. Superior Debugging Skills

When something goes wrong in PyTorch, I can now reason about the underlying cause. I understand:
- How gradients flow through operations
- Why certain architectures are prone to specific problems
- How to identify and fix numerical instabilities
- When and why to use different optimization techniques

### 3. Architectural Intuition

I can now design neural architectures with confidence. I understand:
- Why residual connections work
- How attention mechanisms capture dependencies
- Why layer normalization is placed where it is
- How different initialization schemes affect training

### 4. Framework Appreciation

I have massive respect for PyTorch and TensorFlow engineers. Building a production framework is incredibly difficult. The abstractions they've created hide enormous complexity while remaining flexible and efficient.

## Performance Results: Does It Actually Work?

Despite being built from scratch, Neural Architecture achieves competitive results:

### GPT-2 Language Modeling
- **Final Perplexity**: 198-202 (vs 180-220 for comparable PyTorch models)
- **Training Speed**: 85% of PyTorch speed on CPU
- **Memory Usage**: 15% higher than PyTorch (room for optimization)

### Vision Transformer Classification
- **Test Accuracy**: 88.39% on synthetic CIFAR-10
- **Top-5 Accuracy**: 100% (perfect for 10-class problem)
- **Training Time**: Comparable to PyTorch for small models

### BERT Sentiment Analysis
- **Accuracy**: 85%+ on synthetic sentiment data
- **Training Convergence**: Stable and predictable
- **Feature Quality**: Rich representations suitable for downstream tasks

### CLIP Multimodal Learning
- **Image-Text Retrieval**: R@1: 2%, R@10: 16% (baseline performance)
- **Zero-shot Classification**: Functional cross-modal understanding
- **Embedding Quality**: Captures semantic relationships

The key insight: **performance is competitive for educational and research purposes**, validating that the framework is mathematically correct and well-implemented.

## What's Next: Building on the Foundation

This framework is just the beginning. Here's what I'm working on next:

### 1. Advanced Architectures
- **Mamba/State Space Models**: Linear attention alternatives
- **MoE (Mixture of Experts)**: Scaling with sparse computation
- **Retrieval-Augmented Generation**: Combining parametric and non-parametric knowledge

### 2. Optimization Improvements
- **Mixed Precision Training**: FP16/BF16 support for faster training
- **Gradient Accumulation**: Training larger models with limited memory
- **Model Parallelism**: Distributing large models across devices

### 3. Production Features
- **Model Quantization**: INT8 inference for deployment
- **ONNX Export**: Interoperability with other frameworks
- **JIT Compilation**: Performance optimization through compilation

### 4. Educational Resources
- **Interactive Notebooks**: Step-by-step model building guides
- **Video Tutorials**: Visual explanations of complex concepts
- **University Partnerships**: Curriculum integration opportunities

## The Open Source Impact

Since releasing Neural Architecture, I've been amazed by the community response:

- **700+ GitHub Stars**: Developers appreciate the educational value
- **50+ Contributors**: Improvements and bug fixes from the community
- **University Adoptions**: Used in ML courses at several institutions
- **Industry Interest**: Companies using it for research and prototyping

The most rewarding feedback comes from students and junior developers who say the framework helped them understand ML fundamentals. That's exactly what I hoped to achieve.

## Conclusion: The Journey Continues

Building Neural Architecture from scratch was one of the most challenging and rewarding projects of my career. It transformed me from someone who used ML frameworks to someone who understands them deeply.

**The main lessons I learned:**

1. **Understanding beats convenience**: Taking time to understand fundamentals pays dividends forever
2. **Implementation reveals truth**: You don't truly understand something until you implement it
3. **Testing is crucial**: ML code needs even more rigorous testing than traditional software
4. **Community matters**: Open source thrives on collaboration and shared learning
5. **Teaching clarifies thinking**: Explaining concepts to others deepens your own understanding

**If you're considering a similar journey**, here's my advice:

- **Start small**: Begin with basic operations and build up
- **Test everything**: Write tests before you think you need them
- **Document extensively**: Your future self will thank you
- **Share early**: Community feedback accelerates learning
- **Stay curious**: The rabbit holes are where the best learning happens

## Get Involved

Neural Architecture is an open source project built for learning and exploration. Here's how you can get involved:

- **⭐ Star the repository**: [GitHub Link](https://github.com/fenilsonani/neural-network-from-scratch)
- **🔧 Contribute**: Issues and pull requests welcome
- **📚 Learn**: Use it for education and research
- **💬 Discuss**: Join our community Discord
- **📢 Share**: Help others discover the project

Building neural networks from scratch isn't just about code—it's about understanding the beautiful mathematics that powers modern AI. Every line of code teaches you something new about how intelligence emerges from computation.

The journey from NumPy arrays to working neural networks is challenging, but it's also one of the most educational experiences you can have as a developer.

**Ready to start your own from-scratch journey? The code is waiting for you.**

---

*Fenil Sonani is a senior ML engineer who believes in understanding technology from first principles. When not building neural networks from scratch, he enjoys teaching others about the fundamentals of machine learning. Follow him on [Twitter](https://twitter.com/fenilsonani) for more ML insights.*

**Related Articles:**
- [700+ Tests and 74% Coverage: Testing ML Code](link)
- [The Anatomy of a Neural Network Framework](link)
- [From NumPy to Production: A Framework Story](link)

**GitHub Repository:** [Neural Architecture - Complete Implementation From Scratch](https://github.com/fenilsonani/neural-network-from-scratch)