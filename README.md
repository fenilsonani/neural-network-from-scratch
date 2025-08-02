# Neural Architecture Framework

[![Tests](https://github.com/fenilsonani/neural-network-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/fenilsonani/neural-network-from-scratch/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A neural network implementation built from scratch using NumPy, designed for educational purposes and research experimentation.

## Project Status

**Current Status**: Active Development  
**Stage**: Alpha - Core functionality implemented, testing and optimization ongoing

### What Works
- ✅ Basic tensor operations with automatic differentiation
- ✅ Core neural network layers (Linear, Embedding, LayerNorm, Attention)
- ✅ Basic optimizers (Adam, SGD)
- ✅ Transformer architecture components
- ✅ Multiple backend support (CPU/NumPy with MPS and CUDA backends in development)
- ✅ Translation example application

### In Development
- 🔄 Comprehensive test suite cleanup and standardization
- 🔄 Performance optimizations and benchmarking
- 🔄 GPU acceleration backends (MPS, CUDA)
- 🔄 Documentation improvements

### Planned
- 📋 Additional model architectures (CNN, RNN)
- 📋 Advanced optimization techniques
- 📋 Distributed training support
- 📋 Production deployment tools

## Features

### Core Components
- **Tensor System**: Automatic differentiation with gradient tracking
- **Neural Layers**: Linear, embedding, normalization, attention, and transformer blocks
- **Optimizers**: Adam and SGD with configurable parameters
- **Backend System**: Pluggable backend architecture supporting different compute engines
- **Model Zoo**: Pre-built architectures for common tasks

### Educational Focus
This framework is designed to be:
- **Readable**: Clear, well-documented code that's easy to understand
- **Modular**: Components can be used independently or combined
- **Extensible**: Easy to add new layers, optimizers, or backends
- **Educational**: Extensive comments explaining the mathematics and implementation

## Installation

### Prerequisites
- Python 3.8+
- NumPy

### Basic Installation
```bash
git clone https://github.com/fenilsonani/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/fenilsonani/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install -e ".[dev]"
```

### Optional Dependencies
```bash
# For GPU acceleration (experimental)
pip install mlx-lm  # Apple Silicon
# or
pip install cupy    # NVIDIA CUDA

# For visualization
pip install matplotlib seaborn

# For notebooks
pip install jupyter
```

## Quick Start

### Basic Tensor Operations
```python
from neural_arch.core import Tensor
from neural_arch.functional import matmul

# Create tensors with automatic differentiation
a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([[5, 6], [7, 8]], requires_grad=True)

# Perform operations
c = matmul(a, b)
loss = c.sum()

# Compute gradients
loss.backward()
print(f"Gradient of a: {a.grad}")
```

### Simple Neural Network
```python
from neural_arch.nn import Linear, ReLU
from neural_arch.optim import Adam

# Create a simple network
layer1 = Linear(2, 4)
activation = ReLU()
layer2 = Linear(4, 1)

# Forward pass
x = Tensor([[1, 2]])
h = activation(layer1(x))
output = layer2(h)

# Training setup
optimizer = Adam([layer1.weight, layer1.bias, layer2.weight, layer2.bias])
```

### Transformer Example
```python
from neural_arch.nn import MultiHeadAttention, TransformerBlock

# Create transformer components
attention = MultiHeadAttention(d_model=256, num_heads=8)
transformer = TransformerBlock(d_model=256, num_heads=8, d_ff=1024)

# Process sequences
seq_len, batch_size, d_model = 10, 2, 256
x = Tensor.randn(seq_len, batch_size, d_model)
output = transformer(x)
```

## Examples

### Translation Application
A working English-Spanish translation example is included:

```bash
cd examples/translation

# Prepare data (requires downloading Tatoeba dataset)
python process_spa_file.py

# Train the model
python train_conversational.py

# Test translation
python translate.py
```

### Additional Examples
- **Basic Training Loop**: `examples/basic_training.py`
- **Custom Layer Creation**: `examples/custom_layers.py`
- **Backend Comparison**: `examples/backend_demo.py`

## Project Structure

```
neural-arch/
├── src/neural_arch/          # Main package
│   ├── core/                 # Core tensor and device management
│   ├── nn/                   # Neural network layers
│   ├── functional/           # Functional operations
│   ├── optim/               # Optimizers
│   ├── backends/            # Compute backends
│   └── models/              # Pre-built model architectures
├── tests/                   # Test suite
├── examples/                # Example applications
├── docs/                    # Documentation
└── benchmarks/              # Performance benchmarks
```

## Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neural_arch

# Run specific test categories
pytest tests/test_core/
pytest tests/test_nn/
```

**Note**: Test suite is currently being refactored for better organization and reliability.

## Performance

Current performance characteristics:
- **CPU (NumPy)**: Good performance for small to medium models
- **Memory Usage**: Reasonable for educational/research workloads
- **GPU Support**: Experimental - MPS and CUDA backends in development

Performance will improve as optimization work continues.

## Documentation

- **Documentation Index**: [docs/README.md](docs/README.md)
- **API Reference**: [docs/api/reference.md](docs/api/reference.md)
- **User Guide**: [docs/user-guide/demo.md](docs/user-guide/demo.md)
- **Contributing**: [docs/development/contributing.md](docs/development/contributing.md)
- **Performance Guide**: [docs/advanced/performance.md](docs/advanced/performance.md)

## Contributing

We welcome contributions! This project is particularly suitable for:
- Educational improvements and examples
- Performance optimizations
- Additional neural network architectures
- Better documentation and tutorials
- Bug fixes and code quality improvements

See [CONTRIBUTING.md](docs/development/contributing.md) for guidelines.

## Educational Use

This framework is ideal for:
- **Learning**: Understanding neural networks from first principles
- **Research**: Experimenting with novel architectures
- **Teaching**: Clear code that demonstrates concepts
- **Prototyping**: Quick implementation of ideas

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with NumPy for numerical computations
- Inspired by PyTorch's design philosophy
- Thanks to the open-source ML community

---

**Note**: This is an educational/research framework. For production workloads, consider using established frameworks like PyTorch or TensorFlow.