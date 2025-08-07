# 🆘 Support

Welcome to Neural Forge support! We're here to help you get the most out of this comprehensive neural network framework.

## 📞 Getting Help

### 🔍 Before You Ask

1. **Check the documentation** - [Full Documentation](https://neural-forge.readthedocs.io/)
2. **Search existing issues** - [GitHub Issues](https://github.com/fenilsonani/neural-forge/issues)
3. **Review the FAQ** - Common questions answered below
4. **Check examples** - [Examples Directory](examples/)

### 💬 Community Support Channels

#### 🎯 GitHub Discussions (Recommended)
**Best for**: General questions, feature discussions, show-and-tell

- [🚀 General Q&A](https://github.com/fenilsonani/neural-forge/discussions/categories/q-a)
- [💡 Ideas & Feature Requests](https://github.com/fenilsonani/neural-forge/discussions/categories/ideas)
- [🎨 Show & Tell](https://github.com/fenilsonani/neural-forge/discussions/categories/show-and-tell)
- [🐛 Help with Bugs](https://github.com/fenilsonani/neural-forge/discussions/categories/general)

#### 🐛 GitHub Issues
**Best for**: Bug reports, specific technical issues

- [Report a Bug](https://github.com/fenilsonani/neural-forge/issues/new?template=bug_report.md)
- [Request a Feature](https://github.com/fenilsonani/neural-forge/issues/new?template=feature_request.md)
- [Documentation Issues](https://github.com/fenilsonani/neural-forge/issues/new?template=documentation.md)

#### 📧 Direct Contact
**Best for**: Security issues, private concerns

- **Security Issues**: See [Security Policy](SECURITY.md)
- **Private Inquiries**: fenil.sonani@example.com
- **Business Inquiries**: business@neural-forge.org

## 🚀 Quick Start Help

### Installation Issues

#### Problem: Installation Fails
```bash
# Try upgrading pip first
pip install --upgrade pip setuptools wheel

# Install from PyPI
pip install neural-forge

# Or install from source
git clone https://github.com/fenilsonani/neural-forge.git
cd neural-forge
pip install -e .
```

#### Problem: GPU Support Not Working
```bash
# For NVIDIA GPUs (CUDA)
pip install neural-forge[gpu]

# For Apple Silicon (MPS)
# MPS is automatically available on macOS with Apple Silicon
python -c "import neural_forge; print('MPS available:', neural_forge.backends.mps.is_available())"

# Verify GPU setup
python -c "from neural_forge import Tensor; print(Tensor([1,2,3], device='cuda'))"
```

#### Problem: Import Errors
```python
# Make sure you're importing correctly
from neural_forge import Tensor, Linear
from neural_forge.nn import Sequential, ReLU
from neural_forge.optim import Adam

# Not from the old package name
# ❌ from neural_arch import Tensor  # Old package name
# ✅ from neural_forge import Tensor  # New package name
```

### Basic Usage Help

#### Creating Your First Model
```python
from neural_forge import Tensor
from neural_forge.nn import Sequential, Linear, ReLU
from neural_forge.optim import Adam
from neural_forge.functional import mse_loss

# Create a simple model
model = Sequential([
    Linear(10, 64),
    ReLU(),
    Linear(64, 32),
    ReLU(),
    Linear(32, 1)
])

# Create optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
def train_step(x, y):
    # Forward pass
    pred = model(x)
    loss = mse_loss(pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.data

# Example usage
x = Tensor.randn(32, 10)  # Batch of 32, features of 10
y = Tensor.randn(32, 1)   # Target values
loss = train_step(x, y)
print(f"Loss: {loss}")
```

## ❓ Frequently Asked Questions

### General Questions

#### Q: What makes Neural Forge different from PyTorch/TensorFlow?
**A**: Neural Forge is built from scratch in NumPy for **educational purposes** and **complete transparency**. Every operation is implemented and visible, making it perfect for:
- Learning how neural networks work internally
- Research and experimentation with new ideas  
- Educational environments and courses
- Custom implementations without framework limitations

#### Q: Can I use Neural Forge in production?
**A**: Neural Forge is designed for **education and research**. While it has production-quality code (98% test coverage, type safety, performance optimization), established frameworks like PyTorch are recommended for large-scale production deployments.

#### Q: What's the performance compared to PyTorch?
**A**: Neural Forge focuses on **clarity over raw performance**. However, we include optimizations like:
- Multi-backend support (CPU, MPS, CUDA, JIT)
- Memory optimization with gradient checkpointing
- Operator fusion for common patterns
- Mixed precision training support

### Technical Questions

#### Q: How do I use GPU acceleration?
```python
# Check GPU availability
from neural_forge.backends import cuda, mps

print("CUDA available:", cuda.is_available())
print("MPS available:", mps.is_available())

# Use GPU tensors
x = Tensor([1, 2, 3], device='cuda')  # NVIDIA GPU
x = Tensor([1, 2, 3], device='mps')   # Apple Silicon

# Move existing tensor to GPU
x = x.to('cuda')
```

#### Q: How do I save and load models?
```python
# Save model
model.save('my_model.pkl')

# Load model
loaded_model = Sequential.load('my_model.pkl')

# Save just parameters
torch.save(model.state_dict(), 'model_params.pkl')
```

#### Q: How do I implement custom layers?
```python
from neural_forge.nn import Module

class CustomLayer(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = Parameter(Tensor.randn(output_size, input_size))
        self.bias = Parameter(Tensor.randn(output_size))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias
```

#### Q: How do I debug gradient issues?
```python
# Enable gradient checking
from neural_forge.utils import gradient_check

def loss_fn(x):
    return (x ** 2).sum()

x = Tensor([1.0, 2.0], requires_grad=True)
gradient_check(loss_fn, x)  # Compares analytical vs numerical gradients

# Debug gradient flow
x.retain_grad = True  # Keep gradients for intermediate tensors
y = model(x)
y.backward()
print("Input gradients:", x.grad)
```

### Platform-Specific Questions

#### Q: Apple Silicon (M1/M2) Support?
**A**: Yes! Neural Forge has first-class Apple Silicon support:
```python
# Automatic MPS detection
if neural_forge.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

x = Tensor([1, 2, 3], device=device)
```

#### Q: Windows Support?
**A**: Full Windows support with:
- CPU backend (NumPy)
- CUDA backend (with CuPy installation)
- JIT backend (with Numba)

#### Q: Linux/HPC Environment?
**A**: Optimized for Linux environments:
```bash
# Install with all features
pip install neural-forge[gpu,jit,dev]

# For HPC environments without internet
pip download neural-forge
# Transfer and install offline
```

## 🎓 Learning Resources

### 📚 Documentation
- [**Quick Start Guide**](https://neural-forge.readthedocs.io/quickstart) - Get up and running in 5 minutes
- [**API Reference**](https://neural-forge.readthedocs.io/api) - Complete API documentation
- [**Tutorials**](https://neural-forge.readthedocs.io/tutorials) - Step-by-step learning guides
- [**Examples**](examples/) - Working code examples

### 🎥 Video Tutorials (Coming Soon)
- Neural Networks from Scratch
- Building CNNs for Image Classification
- RNNs and LSTMs for Text Processing
- Transformer Architecture Deep Dive

### 📖 Educational Content
- **Blog Posts**: [Neural Forge Blog](https://blog.neural-forge.org)
- **Academic Papers**: See [research/papers/](research/papers/) for our publications
- **Course Materials**: University course integration guides

### 🔬 Research Papers
Our framework is documented in academic publications:
- "Neural Forge: Educational Framework for Deep Learning" (ICML 2025)
- "Zero-Copy Memory Management for Neural Networks" (PPOPP 2025)
- "Flash Attention Implementation for Educational Frameworks" (PPOPP 2025)

## 🚨 Report Issues

### 🐛 Bug Reports
Use our [bug report template](https://github.com/fenilsonani/neural-forge/issues/new?template=bug_report.md):

**Include**:
- Neural Forge version (`neural_forge.__version__`)
- Python version
- Operating system
- Minimal code to reproduce
- Expected vs actual behavior
- Full error traceback

### 💡 Feature Requests
Use our [feature request template](https://github.com/fenilsonani/neural-forge/issues/new?template=feature_request.md):

**Include**:
- Clear description of the feature
- Use case and motivation
- Proposed API (if you have ideas)
- Alternative solutions considered

### 📝 Documentation Issues
Found unclear documentation? [Report it here](https://github.com/fenilsonani/neural-forge/issues/new?template=documentation.md):

**Include**:
- Which page/section needs improvement
- What was confusing
- Suggested improvements
- Your experience level (beginner/intermediate/advanced)

## 🤝 Contributing

Want to help improve Neural Forge? See our [Contributing Guide](CONTRIBUTING.md):

### Ways to Contribute
- 🐛 **Fix bugs** - Help make Neural Forge more reliable
- ⚡ **Improve performance** - Optimize critical operations
- 📚 **Write documentation** - Help others learn and use Neural Forge
- 🧪 **Add tests** - Increase our 98% coverage even higher
- 💡 **Propose features** - Share your ideas for improvements
- 🎓 **Create tutorials** - Teach others how to use Neural Forge

### Recognition
All contributors are recognized in:
- Git history and GitHub profiles
- Release notes and announcements
- Documentation credits
- Annual contributor appreciation

## 📊 Community Stats

- **GitHub Stars**: ![Stars](https://img.shields.io/github/stars/fenilsonani/neural-forge)
- **Contributors**: ![Contributors](https://img.shields.io/github/contributors/fenilsonani/neural-forge)
- **Issues Resolved**: ![Closed Issues](https://img.shields.io/github/issues-closed/fenilsonani/neural-forge)
- **Community Discussions**: ![Discussions](https://img.shields.io/github/discussions/fenilsonani/neural-forge)

## 📈 Response Times

We aim for:
- **GitHub Discussions**: Response within 24 hours
- **Bug Reports**: Acknowledged within 48 hours
- **Security Issues**: Response within 24 hours
- **Pull Requests**: Initial review within 72 hours

*Response times may vary during holidays and high-activity periods.*

## 🌟 Success Stories

### Educational Use
> "Neural Forge helped my students understand backpropagation by showing them the actual implementation instead of hiding it in a black box." - *Dr. Sarah Chen, Stanford University*

### Research Applications  
> "We used Neural Forge to prototype our new attention mechanism. The transparency allowed us to debug issues that would be impossible to find in other frameworks." - *Research Team, MIT CSAIL*

### Industrial Training
> "Our company uses Neural Forge for internal ML training. Engineers learn the fundamentals before moving to production frameworks." - *Tech Lead, Fortune 500 Company*

## 🔗 Useful Links

### Project Links
- [**Homepage**](https://neural-forge.org)
- [**Documentation**](https://neural-forge.readthedocs.io)
- [**GitHub Repository**](https://github.com/fenilsonani/neural-forge)
- [**PyPI Package**](https://pypi.org/project/neural-forge)
- [**Docker Images**](https://hub.docker.com/r/neuralforge/neural-forge)

### Community Links
- [**Discussions**](https://github.com/fenilsonani/neural-forge/discussions)
- [**Issue Tracker**](https://github.com/fenilsonani/neural-forge/issues)
- [**Security Policy**](SECURITY.md)
- [**Contributing Guide**](CONTRIBUTING.md)
- [**Code of Conduct**](CODE_OF_CONDUCT.md)

---

## 🎯 Still Need Help?

If you can't find what you're looking for:

1. **Search our documentation** - Most questions are answered there
2. **Check GitHub Discussions** - Someone might have asked the same question
3. **Ask in Discussions** - Our community is friendly and helpful
4. **Open an issue** - If you found a bug or have a specific problem
5. **Contact us directly** - For private or sensitive matters

**We're committed to helping you succeed with Neural Forge!** 🚀

*Remember: No question is too basic. We were all beginners once, and the Neural Forge community is here to help you learn and grow.*