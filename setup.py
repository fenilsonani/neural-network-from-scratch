"""Setup configuration for Neural Architecture package."""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we can import the package to get version info
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def get_version():
    """Get version from __version__.py file."""
    version_file = Path(__file__).parent / "src" / "neural_arch" / "__version__.py"
    namespace = {}
    exec(version_file.read_text(), namespace)
    return namespace["__version__"]

def get_long_description():
    """Get long description from README file."""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return "Educational neural network implementation from scratch using NumPy"

def get_requirements():
    """Get requirements from requirements.txt file."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        requirements = []
        for line in requirements_file.read_text().splitlines():
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Extract package name (ignore comments after package)
                package = line.split('#')[0].strip()
                if package:
                    requirements.append(package)
        return requirements
    return ["numpy>=1.21.0"]

setup(
    # Package metadata
    name="neural-forge",
    version=get_version(),
    description="Neural Forge - Comprehensive neural network framework built from scratch in NumPy",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Fenil Sonani",
    author_email="fenil@fenilsonani.com",
    url="https://github.com/fenilsonani/neural-forge",
    project_urls={
        "Bug Reports": "https://github.com/fenilsonani/neural-forge/issues",
        "Source": "https://github.com/fenilsonani/neural-forge",
        "Documentation": "https://github.com/fenilsonani/neural-forge/tree/main/docs",
    },
    
    # Package configuration
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-xdist>=2.5.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-xdist>=2.5.0", 
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "neural-forge=neural_arch.cli.main:main",
        ],
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Typing :: Typed",
    ],
    
    # Additional metadata
    keywords=[
        "neural-forge", "neural-networks", "deep-learning", "machine-learning",
        "artificial-intelligence", "numpy", "from-scratch", "cnn", "rnn", "lstm",
        "gru", "transformer", "attention", "educational", "research-tool"
    ],
    license="MIT",
    platforms=["any"],
    zip_safe=False,
    
    # Include additional files
    include_package_data=True,
    package_data={
        "neural_arch": [
            "py.typed",  # PEP 561 marker for type information
        ],
    },
    
    # Minimum requirements validation
    setup_requires=[
        "setuptools>=45",
        "wheel",
    ],
)