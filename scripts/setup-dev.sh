#!/bin/bash
# Neural Forge Development Environment Setup Script
# This script sets up a complete development environment for Neural Forge

set -euo pipefail  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print header
print_header() {
    echo -e "${BLUE}"
    echo "=================================="
    echo "   Neural Forge Dev Setup"
    echo "=================================="
    echo -e "${NC}"
    echo "Setting up your Neural Forge development environment..."
    echo ""
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        REQUIRED_VERSION="3.8"
        
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Python $PYTHON_VERSION detected (>= 3.8 required)"
        else
            log_error "Python >= 3.8 is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Git
    if command_exists git; then
        log_success "Git is available"
    else
        log_error "Git is not installed"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/neural_arch" ]]; then
        log_error "Please run this script from the Neural Forge root directory"
        exit 1
    fi
    
    log_success "All system requirements met"
    echo ""
}

# Set up virtual environment
setup_virtual_env() {
    log_info "Setting up virtual environment..."
    
    VENV_DIR="venv"
    
    if [[ -d "$VENV_DIR" ]]; then
        log_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    log_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment set up successfully"
    echo ""
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Make sure we're in the virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_info "Activating virtual environment..."
        source venv/bin/activate
    fi
    
    # Install the package in editable mode with all development dependencies
    log_info "Installing Neural Forge in development mode..."
    pip install -e ".[dev,gpu,docs,test,benchmark]"
    
    # Install additional development tools
    log_info "Installing additional development tools..."
    pip install \
        black \
        isort \
        flake8 \
        mypy \
        bandit \
        safety \
        pytest \
        pytest-cov \
        pytest-xdist \
        pre-commit \
        sphinx \
        sphinx-rtd-theme \
        myst-parser
    
    log_success "All dependencies installed"
    echo ""
}

# Set up pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source venv/bin/activate
    fi
    
    if [[ -f ".pre-commit-config.yaml" ]]; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
    fi
    
    echo ""
}

# Run initial tests
run_tests() {
    log_info "Running initial test suite to verify setup..."
    
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source venv/bin/activate
    fi
    
    # Test basic import
    log_info "Testing basic import..."
    if python -c "import neural_arch; print('✅ Basic import successful')"; then
        log_success "Basic import test passed"
    else
        log_error "Basic import test failed"
        return 1
    fi
    
    # Run a quick subset of tests
    log_info "Running core tests..."
    if pytest tests/test_tensor.py -v --tb=short --disable-warnings -q; then
        log_success "Core tests passed"
    else
        log_warning "Some core tests failed - this might be normal during development"
    fi
    
    echo ""
}

# Set up IDE configurations
setup_ide() {
    log_info "Setting up IDE configurations..."
    
    # VS Code settings
    mkdir -p .vscode
    
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.banditEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/build": true,
        "**/dist": true,
        "**/*.egg-info": true
    }
}
EOF
    
    # VS Code extensions recommendations
    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "ms-python.isort",
        "ms-toolsai.jupyter",
        "charliermarsh.ruff",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml",
        "yzhang.markdown-all-in-one",
        "ms-vscode.test-adapter-converter"
    ]
}
EOF
    
    # EditorConfig
    cat > .editorconfig << 'EOF'
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.{yml,yaml}]
indent_size = 2

[*.{json,md}]
indent_size = 2

[Makefile]
indent_style = tab
EOF
    
    log_success "IDE configurations created"
    echo ""
}

# Create development scripts
create_dev_scripts() {
    log_info "Creating development scripts..."
    
    mkdir -p scripts
    
    # Test script
    cat > scripts/test.sh << 'EOF'
#!/bin/bash
# Run comprehensive test suite
set -e

echo "Running Neural Forge test suite..."

# Activate virtual environment if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source venv/bin/activate
fi

# Run tests with coverage
pytest tests/ \
    --cov=src/neural_arch \
    --cov-report=html \
    --cov-report=term \
    --cov-fail-under=95 \
    -v

echo "✅ Test suite completed successfully!"
EOF

    # Format script
    cat > scripts/format.sh << 'EOF'
#!/bin/bash
# Format code using black and isort
set -e

echo "Formatting Neural Forge codebase..."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source venv/bin/activate
fi

echo "Running black..."
black src/ tests/ --line-length 100

echo "Running isort..."
isort src/ tests/ --profile black

echo "✅ Code formatting completed!"
EOF

    # Lint script
    cat > scripts/lint.sh << 'EOF'
#!/bin/bash
# Run linting checks
set -e

echo "Running linting checks..."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source venv/bin/activate
fi

echo "Running flake8..."
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

echo "Running mypy..."
mypy src/neural_arch --strict --ignore-missing-imports

echo "Running bandit..."
bandit -r src/ -f json || true

echo "✅ Linting checks completed!"
EOF

    # Documentation build script
    cat > scripts/build-docs.sh << 'EOF'
#!/bin/bash
# Build documentation
set -e

echo "Building Neural Forge documentation..."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source venv/bin/activate
fi

cd docs/sphinx
make html

echo "✅ Documentation built successfully!"
echo "Open docs/sphinx/_build/html/index.html to view"
EOF

    # Make scripts executable
    chmod +x scripts/*.sh
    
    log_success "Development scripts created"
    echo ""
}

# Display completion message
show_completion() {
    echo -e "${GREEN}"
    echo "=================================="
    echo "     Setup Complete! 🎉"
    echo "=================================="
    echo -e "${NC}"
    echo ""
    echo "Your Neural Forge development environment is ready!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo -e "   ${YELLOW}source venv/bin/activate${NC}"
    echo ""
    echo "2. Run tests to verify everything works:"
    echo -e "   ${YELLOW}./scripts/test.sh${NC}"
    echo ""
    echo "3. Start developing! Key commands:"
    echo -e "   ${YELLOW}./scripts/format.sh${NC}   - Format code"
    echo -e "   ${YELLOW}./scripts/lint.sh${NC}     - Run linting"
    echo -e "   ${YELLOW}./scripts/build-docs.sh${NC} - Build docs"
    echo ""
    echo "4. Before committing:"
    echo -e "   ${YELLOW}pre-commit run --all-files${NC}"
    echo ""
    echo "Happy coding! 🚀"
}

# Main execution
main() {
    print_header
    check_requirements
    setup_virtual_env
    install_dependencies
    setup_precommit
    setup_ide
    create_dev_scripts
    run_tests
    show_completion
}

# Run main function
main "$@"