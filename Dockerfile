# Neural Forge - Production Docker Image
# Multi-stage build for optimized production deployment

# Stage 1: Build stage with development dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash neural-forge
USER neural-forge
WORKDIR /home/neural-forge

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt \
    && pip install --user --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY --chown=neural-forge:neural-forge . ./neural-forge/

# Install the package
WORKDIR /home/neural-forge/neural-forge
RUN pip install --user -e .

# Stage 2: Production runtime
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/neural-forge/.local/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash neural-forge
USER neural-forge
WORKDIR /home/neural-forge

# Copy installed packages from builder
COPY --from=builder /home/neural-forge/.local /home/neural-forge/.local
COPY --from=builder /home/neural-forge/neural-forge /home/neural-forge/neural-forge

# Set working directory
WORKDIR /home/neural-forge/neural-forge

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from neural_arch.core import Tensor; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "from neural_arch.backends import available_backends, print_available_devices; print('Available backends:', available_backends()); print_available_devices()"]

# Stage 3: Development image
FROM production as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER neural-forge

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install --user --no-cache-dir -r requirements-dev.txt

# Install package in development mode
RUN pip install --user -e .

# Development command
CMD ["/bin/bash"]

# Stage 4: Jupyter notebook server
FROM development as jupyter

# Expose Jupyter port
EXPOSE 8888

# Install Jupyter extensions
RUN pip install --user jupyter-lab jupyterlab-git

# Default Jupyter command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/home/neural-forge/neural-forge"]

# Stage 5: GPU-enabled image (requires NVIDIA Docker runtime)
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN useradd --create-home --shell /bin/bash neural-forge
USER neural-forge
WORKDIR /home/neural-forge

# Copy source and install
COPY --chown=neural-forge:neural-forge . ./neural-forge/
WORKDIR /home/neural-forge/neural-forge

# Install requirements
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user cupy-cuda12x  # Install CuPy for CUDA support
RUN pip install --user -e .

# Add health check for GPU
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from neural_arch.backends import get_backend; print('CUDA available:', get_backend('cuda').is_available)" || exit 1

# Default GPU command
CMD ["python", "-c", "from neural_arch.backends import available_backends, print_available_devices; print('Available backends:', available_backends()); print_available_devices()"]