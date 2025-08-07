#!/usr/bin/env python3
"""
Neural Forge Development Environment Validator

This script validates that the development environment is set up correctly
and all components are working as expected.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.results: List[Tuple[str, bool, str]] = []
    
    def add_result(self, test_name: str, success: bool, message: str = ""):
        self.results.append((test_name, success, message))
        if success:
            self.passed += 1
        else:
            self.failed += 1
    
    def add_warning(self, test_name: str, message: str):
        self.results.append((test_name, None, message))
        self.warnings += 1

def print_header():
    """Print validation header"""
    print(f"{Colors.BLUE}{Colors.BOLD}")
    print("=" * 50)
    print("   Neural Forge Environment Validator")
    print("=" * 50)
    print(f"{Colors.END}")

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements"""
    if sys.version_info >= (3, 8):
        return True, f"Python {sys.version.split()[0]}"
    return False, f"Python {sys.version.split()[0]} (requires >= 3.8)"

def check_module_import(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, f"Module '{module_name}' not found"
        
        # Try to actually import it
        importlib.import_module(module_name)
        return True, f"Module '{module_name}' imported successfully"
    except Exception as e:
        return False, f"Failed to import '{module_name}': {str(e)}"

def check_neural_arch_components() -> List[Tuple[bool, str]]:
    """Check Neural Arch core components"""
    components = [
        "neural_arch",
        "neural_arch.core",
        "neural_arch.nn",
        "neural_arch.optim",
        "neural_arch.backends",
        "neural_arch.functional",
    ]
    
    results = []
    for component in components:
        success, message = check_module_import(component)
        results.append((success, message))
    
    return results

def check_development_tools() -> List[Tuple[bool, str]]:
    """Check development tools availability"""
    tools = [
        ("pytest", "Testing framework"),
        ("black", "Code formatter"),
        ("isort", "Import sorter"),
        ("flake8", "Linter"),
        ("mypy", "Type checker"),
        ("pre_commit", "Pre-commit hooks"),
        ("sphinx", "Documentation generator"),
    ]
    
    results = []
    for tool, description in tools:
        success, message = check_module_import(tool)
        if success:
            results.append((True, f"{description} available"))
        else:
            results.append((False, f"{description} not available"))
    
    return results

def check_file_structure() -> List[Tuple[bool, str]]:
    """Check project file structure"""
    required_files = [
        ("pyproject.toml", "Project configuration"),
        ("README.md", "Project documentation"),
        ("LICENSE", "License file"),
        ("CONTRIBUTING.md", "Contributing guidelines"),
        ("SECURITY.md", "Security policy"),
        ("CHANGELOG.md", "Change log"),
        ("SUPPORT.md", "Support information"),
        (".pre-commit-config.yaml", "Pre-commit configuration"),
        ("src/neural_arch/__init__.py", "Neural Arch package"),
        ("tests/", "Test directory"),
        ("docs/", "Documentation directory"),
    ]
    
    results = []
    for file_path, description in required_files:
        path = Path(file_path)
        if path.exists():
            results.append((True, f"{description} exists"))
        else:
            results.append((False, f"{description} missing: {file_path}"))
    
    return results

def check_git_hooks() -> Tuple[bool, str]:
    """Check if git pre-commit hooks are installed"""
    try:
        git_hooks_path = Path(".git/hooks/pre-commit")
        if git_hooks_path.exists():
            return True, "Pre-commit hooks installed"
        return False, "Pre-commit hooks not installed (run: pre-commit install)"
    except Exception as e:
        return False, f"Error checking git hooks: {str(e)}"

def run_basic_functionality_test() -> Tuple[bool, str]:
    """Run basic functionality test"""
    try:
        # Test basic tensor creation and operations
        test_code = '''
import neural_arch
from neural_arch.core import Tensor
import numpy as np

# Test tensor creation
x = Tensor([1, 2, 3], requires_grad=True)
y = Tensor([4, 5, 6], requires_grad=True)

# Test basic operations
z = x + y
loss = z.sum()

# Test backward pass
loss.backward()

# Verify gradients exist
assert x.grad is not None
assert y.grad is not None

print("✅ Basic functionality test passed")
'''
        
        exec(test_code)
        return True, "Basic functionality test passed"
        
    except Exception as e:
        return False, f"Basic functionality test failed: {str(e)}"

def check_performance_backends() -> List[Tuple[bool, str]]:
    """Check available performance backends"""
    backends = []
    
    try:
        from neural_arch.backends import cuda
        if cuda.is_available():
            backends.append((True, "CUDA backend available"))
        else:
            backends.append((None, "CUDA backend not available (optional)"))
    except:
        backends.append((None, "CUDA backend not available (optional)"))
    
    try:
        from neural_arch.backends import mps
        if mps.is_available():
            backends.append((True, "MPS backend available"))
        else:
            backends.append((None, "MPS backend not available (optional)"))
    except:
        backends.append((None, "MPS backend not available (optional)"))
    
    try:
        from neural_arch.backends import jit
        if jit.is_available():
            backends.append((True, "JIT backend available"))
        else:
            backends.append((None, "JIT backend not available (optional)"))
    except:
        backends.append((None, "JIT backend not available (optional)"))
    
    return backends

def run_validation():
    """Run all validation checks"""
    print_header()
    
    result = ValidationResult()
    
    # Python version check
    success, message = check_python_version()
    result.add_result("Python Version", success, message)
    
    # Core module checks
    for success, message in check_neural_arch_components():
        result.add_result("Neural Arch Component", success, message)
    
    # Development tools
    for success, message in check_development_tools():
        if success:
            result.add_result("Development Tool", success, message)
        else:
            result.add_warning("Development Tool", message)
    
    # File structure
    for success, message in check_file_structure():
        result.add_result("File Structure", success, message)
    
    # Git hooks
    success, message = check_git_hooks()
    if success:
        result.add_result("Git Hooks", success, message)
    else:
        result.add_warning("Git Hooks", message)
    
    # Basic functionality
    success, message = run_basic_functionality_test()
    result.add_result("Basic Functionality", success, message)
    
    # Performance backends (optional)
    for success, message in check_performance_backends():
        if success is None:
            result.add_warning("Performance Backend", message)
        else:
            result.add_result("Performance Backend", success, message)
    
    # Print results
    print("\nValidation Results:")
    print("-" * 50)
    
    for test_name, success, message in result.results:
        if success is True:
            print(f"{Colors.GREEN}✅ PASS{Colors.END} {test_name}: {message}")
        elif success is False:
            print(f"{Colors.RED}❌ FAIL{Colors.END} {test_name}: {message}")
        else:  # Warning
            print(f"{Colors.YELLOW}⚠️  WARN{Colors.END} {test_name}: {message}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Summary: {Colors.GREEN}{result.passed} passed{Colors.END}, "
          f"{Colors.RED}{result.failed} failed{Colors.END}, "
          f"{Colors.YELLOW}{result.warnings} warnings{Colors.END}")
    
    if result.failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Environment validation successful!{Colors.END}")
        print("Your development environment is ready for Neural Forge development.")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Environment validation failed{Colors.END}")
        print("Please fix the failed checks before proceeding with development.")
        return False

def main():
    """Main validation function"""
    success = run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()