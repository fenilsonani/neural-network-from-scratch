# 🤝 Contributing to Neural Forge

Thank you for your interest in contributing to Neural Forge! This comprehensive neural network framework is built from scratch with educational clarity and production-quality performance. This guide will help you get started.

## 🎯 Project Philosophy

Neural Forge maintains several core principles:

### 🔬 Educational Excellence
- **From-scratch implementation** - Build understanding with clear, readable code
- **Mathematical transparency** - Show the math behind neural networks
- **Comprehensive testing** - Every component thoroughly validated
- **Type-safe codebase** - Complete type hints throughout

### ⚡ Production Quality
- **Performance focused** - Optimized operations with benchmarking
- **Multi-backend support** - CPU, MPS (Apple Silicon), CUDA, JIT
- **Memory efficient** - Advanced memory management and optimization
- **Numerical stability** - Robust handling of edge cases

### 🚀 Modern Development
- **CI/CD pipeline** - Automated testing and quality assurance
- **98% test coverage** - Comprehensive test suite with 3,800+ tests
- **Professional documentation** - Complete API docs and guides
- **Security-first** - Vulnerability scanning and secure practices

## 🚀 Getting Started

### 1. Development Setup

```bash
# Clone the repository
git clone https://github.com/fenilsonani/neural-forge.git
cd neural-forge

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e .
pip install -r requirements.txt

# Install pre-commit hooks (recommended)
pre-commit install
```

### 2. Verify Your Setup

```bash
# Run all tests to ensure everything works
pytest -v

# Run specific test categories
pytest tests/test_core/ -v
pytest tests/test_nn/ -v
pytest tests/test_optimization/ -v

# Check code quality
black --check src/
flake8 src/
mypy src/
```

### 3. Explore the Codebase

```bash
# Core implementation
src/neural_arch/

# Test suite (3,800+ tests)
tests/

# Examples and demos  
examples/
```

## 📝 Contributing Guidelines

### 🎯 Types of Contributions Welcome

#### 🐛 Bug Fixes
- Numerical instability issues
- Memory leaks or inefficient operations
- Incorrect gradient computations
- Edge case handling improvements

#### ⚡ Performance Improvements
- Faster tensor operations
- Memory optimization
- Algorithm efficiency improvements
- Backend-specific optimizations

#### 🧪 Testing Enhancements
- Additional edge cases
- Performance regression tests
- Stress testing scenarios
- Mathematical verification tests

#### 🤖 New Features
- Additional neural network layers
- Advanced optimizers and schedulers
- New activation functions
- Model architectures

#### 📚 Documentation
- Code examples and tutorials
- Mathematical explanations
- Performance optimization guides
- API documentation improvements

### 🔧 Development Workflow

#### 1. Before Making Changes

```bash
# Create a new branch for your contribution
git checkout -b feature/your-feature-name

# Run tests to establish baseline
pytest -v
```

#### 2. Development Standards

##### Code Quality Requirements
- ✅ **Type hints required** - All functions must have complete type annotations
- ✅ **Zero `Any` types** - Follow strict typing guidelines
- ✅ **Comprehensive tests** - Every new function needs corresponding tests
- ✅ **Performance benchmarks** - Performance-critical code needs benchmarking
- ✅ **Documentation** - All public functions need docstrings
- ✅ **Security checks** - No hardcoded secrets, proper error handling

##### Code Style
```python
def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication with automatic differentiation.
    
    Args:
        a: Left tensor for multiplication (shape: [M, K])
        b: Right tensor for multiplication (shape: [K, N])
        
    Returns:
        Result tensor with shape [M, N] and gradient tracking
        
    Raises:
        ValueError: If tensor shapes are incompatible
    """
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Incompatible shapes: {a.shape} and {b.shape}")
    
    result_data = np.matmul(a.data, b.data)
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    if result.requires_grad:
        def backward() -> None:
            if result.grad is not None:
                if a.requires_grad:
                    a.backward(np.matmul(result.grad, b.data.swapaxes(-2, -1)))
                if b.requires_grad:
                    b.backward(np.matmul(a.data.swapaxes(-2, -1), result.grad))
        
        result._backward = backward
    
    return result
```

#### 3. Testing Your Changes

```bash
# Run all tests
pytest -v

# Run specific test file for your changes
pytest tests/test_your_feature.py -v

# Run with coverage
pytest --cov=src/neural_arch --cov-report=html

# Performance benchmarks
python benchmarks/your_benchmark.py
```

#### 4. Performance Requirements

All contributions must meet performance standards:

```python
def test_operation_performance(benchmark):
    """Test that operation meets speed requirements."""
    large_tensor = Tensor.randn(1000, 1000)
    
    # Benchmark the operation
    result = benchmark(your_operation, large_tensor)
    
    # Performance assertions
    assert benchmark.stats.stats.mean < 0.001  # < 1ms average
```

### 📋 Pull Request Process

#### 1. Pre-Submission Checklist

- [ ] **All tests pass** - `pytest -v` shows 100% success
- [ ] **New tests added** - Your changes are comprehensively tested
- [ ] **Performance verified** - No regression in benchmarks
- [ ] **Type hints complete** - All functions properly typed
- [ ] **Documentation updated** - Relevant docs reflect changes
- [ ] **Security checks pass** - No vulnerabilities introduced
- [ ] **Code formatted** - `black` and `flake8` pass
- [ ] **Pre-commit hooks pass** - All automated checks succeed

#### 2. Pull Request Template

When creating a PR, use this template:

```markdown
## 🎯 Change Summary
Brief description of what this PR does and why.

## 🧪 Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality  
- [ ] Performance benchmarks updated if applicable
- [ ] Edge cases thoroughly tested

## ⚡ Performance Impact
- [ ] No performance regression measured
- [ ] Benchmarks meet requirements
- [ ] Memory usage optimized

## 🔒 Security
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented
- [ ] Error handling secure

## 📚 Documentation
- [ ] Code properly documented
- [ ] API docs updated if needed
- [ ] Examples added if applicable

## 🔬 Technical Details
Detailed explanation of implementation approach, trade-offs, and considerations.
```

#### 3. Review Process

1. **Automated checks** - CI pipeline runs tests, linting, security scans
2. **Code review** - Focus on correctness, performance, maintainability
3. **Mathematical verification** - Gradient computation validation if applicable
4. **Integration testing** - Ensure no regressions
5. **Performance review** - Benchmark results evaluation

## 🧪 Testing Guidelines

### Test Categories

#### 1. Unit Tests
```python
def test_tensor_creation():
    """Test tensor creation with various inputs."""
    # Test different dtypes, shapes, requires_grad settings
    tensor = Tensor([[1, 2], [3, 4]], requires_grad=True)
    assert tensor.shape == (2, 2)
    assert tensor.requires_grad is True
```

#### 2. Integration Tests  
```python
def test_training_pipeline():
    """Test complete training pipeline."""
    model = Sequential([Linear(10, 5), ReLU(), Linear(5, 1)])
    optimizer = Adam(model.parameters())
    # Test forward pass, loss computation, backward pass, optimization
```

#### 3. Performance Tests
```python
def test_cuda_performance(benchmark):
    """Benchmark CUDA operations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    large_tensor = Tensor.randn(5000, 5000, device='cuda')
    result = benchmark(matrix_multiply, large_tensor, large_tensor)
```

#### 4. Mathematical Verification
```python
def test_gradient_correctness():
    """Verify gradients using finite differences."""
    def numerical_gradient(f, x, h=1e-5):
        grad = np.zeros_like(x.data)
        for i in range(x.data.size):
            x_pos = x.data.copy()
            x_neg = x.data.copy()
            x_pos.flat[i] += h
            x_neg.flat[i] -= h
            grad.flat[i] = (f(Tensor(x_pos)).data - f(Tensor(x_neg)).data) / (2 * h)
        return grad
    
    x = Tensor.randn(3, 3, requires_grad=True)
    y = your_function(x)
    y.backward()
    
    numerical_grad = numerical_gradient(your_function, x)
    assert np.allclose(x.grad, numerical_grad, rtol=1e-5)
```

## 🏗️ Architecture Guidelines

### Design Principles

1. **Modularity** - Clean separation between components
2. **Extensibility** - Easy to add new layers, optimizers, backends
3. **Performance** - Optimize critical paths, profile regularly
4. **Safety** - Strong typing, input validation, error handling
5. **Clarity** - Educational value, readable implementations

### Adding New Components

#### New Neural Network Layer
```python
class NewLayer(Module):
    """New neural network layer implementation."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = Parameter(Tensor.randn(output_size, input_size))
        self.bias = Parameter(Tensor.randn(output_size))
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass implementation."""
        return linear(x, self.weight, self.bias)
```

#### New Optimizer
```python
class NewOptimizer(Optimizer):
    """New optimization algorithm."""
    
    def __init__(self, params: List[Parameter], lr: float = 0.01):
        super().__init__(params)
        self.lr = lr
        self.state = {}
    
    def step(self) -> None:
        """Perform optimization step."""
        for param in self.params:
            if param.grad is None:
                continue
            
            # Your optimization logic here
            param.data -= self.lr * param.grad
```

## 🔒 Security Guidelines

- **Never commit secrets** - Use environment variables for sensitive data
- **Validate inputs** - Check tensor shapes, dtypes, value ranges  
- **Handle errors securely** - Don't leak sensitive information in error messages
- **Use secure dependencies** - Regularly update packages, scan for vulnerabilities
- **Follow OWASP guidelines** - Implement secure coding practices

## 🌟 Recognition

All contributors are recognized through:
- Git commit history and authorship
- Pull request acknowledgments
- Documentation credits  
- Release notes mentions
- Contributor listing in README

## 📚 Learning Resources

- **Neural Network Concepts** - [Deep Learning Book](http://www.deeplearningbook.org/)
- **Automatic Differentiation** - Understanding backpropagation mechanics
- **Performance Optimization** - NumPy optimization techniques
- **Testing Best Practices** - pytest and testing methodologies

## 💬 Community

- **GitHub Discussions** - Ask questions, share ideas
- **Issue Tracker** - Report bugs, request features
- **Code Reviews** - Learn from feedback, help others

---

**Thank you for contributing to Neural Forge!** Every contribution helps make this the best educational neural network framework available. 🚀