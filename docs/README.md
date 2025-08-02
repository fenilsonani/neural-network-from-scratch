# Documentation

Welcome to the Neural Architecture Framework documentation!

## Quick Start

### New Users
1. **[Main README](../README.md)** - Project overview and quick start guide
2. **[API Reference](api/reference.md)** - Complete API documentation
3. **[Demo Guide](user-guide/demo.md)** - Interactive demonstration

### Essential Documentation

- **[API Reference](api/reference.md)** - Complete API documentation with examples
- **[Testing Guide](user-guide/testing.md)** - Running and writing tests
- **[Contributing Guide](development/contributing.md)** - How to contribute to the project

## Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── user-guide/                 # User-facing documentation
│   ├── demo.md                  # Interactive demo guide
│   └── testing.md               # Testing documentation
├── api/                         # API documentation
│   └── reference.md             # Complete API reference
├── development/                 # Development documentation
│   ├── contributing.md          # Contribution guidelines
│   └── changelog.md             # Version history
├── advanced/                    # Advanced topics
│   ├── performance.md           # Performance optimization
│   ├── cuda-acceleration.md     # GPU acceleration
│   ├── memory-optimization.md   # Memory management
│   ├── distributed-training.md  # Distributed training
│   ├── ci-cd-architecture.md    # CI/CD setup
│   └── ci-cd-troubleshooting.md # CI/CD troubleshooting
├── reports/                     # Generated reports
│   ├── coverage-update.md       # Coverage reports
│   ├── functional-coverage.md   # Functional test coverage
│   ├── multi-agent-coverage.md  # Multi-agent test coverage
│   └── test-execution-summary.md# Test execution results
└── sphinx/                      # Sphinx documentation build
```

## Documentation Categories

### For Users
- **[Demo Guide](user-guide/demo.md)** - Getting started with examples
- **[API Reference](api/reference.md)** - Complete API documentation
- **[Testing Guide](user-guide/testing.md)** - Running tests and validation

### For Developers
- **[Contributing Guide](development/contributing.md)** - Development workflow
- **[Changelog](development/changelog.md)** - Project history
- **[Performance Guide](advanced/performance.md)** - Optimization techniques

### For Advanced Users
- **[CUDA Acceleration](advanced/cuda-acceleration.md)** - GPU setup and usage
- **[Memory Optimization](advanced/memory-optimization.md)** - Memory management
- **[Distributed Training](advanced/distributed-training.md)** - Multi-GPU training
- **[CI/CD Architecture](advanced/ci-cd-architecture.md)** - Development infrastructure

### Project Reports
- **[Coverage Reports](reports/)** - Test coverage analysis
- **[Execution Summaries](reports/)** - Test run results

## Building Documentation

### Sphinx Documentation
```bash
cd docs/sphinx
make html
```

The generated documentation will be available in `docs/sphinx/_build/html/`.

### Local Development
- All markdown files can be viewed directly in any markdown viewer
- Cross-references use relative paths for portability
- Documentation is organized by user journey and technical depth

## Contributing to Documentation

We welcome documentation improvements! See the [Contributing Guide](development/contributing.md) for details.

### High Priority
- Tutorial and getting started guides
- Example walkthroughs  
- API documentation completeness
- Performance benchmarking results

### Medium Priority
- Advanced usage patterns
- Integration guides
- Troubleshooting sections
- FAQ compilation

## Documentation Standards

- **Clarity**: Write for users at different skill levels
- **Accuracy**: Ensure code examples work and are tested
- **Organization**: Use the directory structure appropriately
- **Maintenance**: Keep documentation in sync with code changes
- **Naming**: Use kebab-case for file names (e.g., `memory-optimization.md`)

## Getting Help

- **Issues**: Report documentation issues on [GitHub](https://github.com/fenilsonani/neural-network-from-scratch/issues)
- **Examples**: Check the [examples/](../examples/) directory for working code
- **API Questions**: See the [API Reference](api/reference.md)

---

**Note**: This documentation follows a structured approach for better discoverability and maintenance.