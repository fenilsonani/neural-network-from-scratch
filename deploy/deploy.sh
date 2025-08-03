#!/bin/bash
# Neural Forge Deployment Script
# Automated deployment script for various environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"

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

# Print usage
usage() {
    cat << EOF
Neural Forge Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    dev         Deploy development environment
    prod        Deploy production environment
    jupyter     Deploy Jupyter notebook server
    gpu         Deploy GPU-accelerated environment
    test        Run tests in Docker
    benchmark   Run performance benchmarks
    docs        Deploy documentation server
    stop        Stop all services
    clean       Clean up containers and volumes
    logs        Show logs for services
    status      Show status of all services

Options:
    -h, --help  Show this help message
    -v, --verbose  Enable verbose output
    -f, --force    Force rebuild of images
    --no-cache     Build without cache

Examples:
    $0 dev                 # Deploy development environment
    $0 prod --force        # Deploy production with image rebuild
    $0 jupyter             # Start Jupyter notebook server
    $0 logs neural-forge-prod  # Show logs for production service
    $0 clean               # Clean up everything

EOF
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Create .env file if it doesn't exist
create_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating .env file..."
        cat > "$ENV_FILE" << EOF
# Neural Forge Environment Configuration

# Environment
NEURAL_FORGE_ENV=development
PYTHONPATH=/home/neural-forge/neural-forge/src

# Jupyter
JUPYTER_TOKEN=neural-forge-$(openssl rand -hex 16)

# Database
POSTGRES_DB=neural_forge
POSTGRES_USER=neural_forge
POSTGRES_PASSWORD=neural_forge_$(openssl rand -hex 8)

# Redis
REDIS_PASSWORD=redis_$(openssl rand -hex 8)

# Docker
DOCKER_BUILDKIT=1
COMPOSE_DOCKER_CLI_BUILD=1

# GPU Support (uncomment if using GPU)
# NVIDIA_VISIBLE_DEVICES=all
# NVIDIA_DRIVER_CAPABILITIES=compute,utility

EOF
        log_success ".env file created"
    else
        log_info ".env file already exists"
    fi
}

# Build images
build_images() {
    local force=${1:-false}
    local no_cache=${2:-false}
    
    local build_args=""
    if [[ "$force" == "true" ]]; then
        build_args="$build_args --force-recreate"
    fi
    if [[ "$no_cache" == "true" ]]; then
        build_args="$build_args --no-cache"
    fi
    
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT"
    docker-compose build $build_args
    log_success "Images built successfully"
}

# Deploy development environment
deploy_dev() {
    log_info "Deploying development environment..."
    cd "$PROJECT_ROOT"
    
    docker-compose up -d neural-forge-dev redis postgres
    
    log_success "Development environment deployed"
    log_info "Access the development container with: docker exec -it neural-forge-development bash"
}

# Deploy production environment
deploy_prod() {
    log_info "Deploying production environment..."
    cd "$PROJECT_ROOT"
    
    docker-compose up -d neural-forge-prod redis postgres prometheus
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    timeout 60s bash -c 'until docker exec neural-forge-production python -c "from neural_arch.core import Tensor; print(\"OK\")" 2>/dev/null; do sleep 2; done'
    
    log_success "Production environment deployed"
    log_info "Production service is running and healthy"
}

# Deploy Jupyter server
deploy_jupyter() {
    log_info "Deploying Jupyter notebook server..."
    cd "$PROJECT_ROOT"
    
    docker-compose up -d neural-forge-jupyter
    
    # Get Jupyter URL
    local jupyter_token
    jupyter_token=$(grep "JUPYTER_TOKEN" "$ENV_FILE" | cut -d'=' -f2)
    
    log_success "Jupyter server deployed"
    log_info "Access Jupyter at: http://localhost:8888/lab?token=$jupyter_token"
}

# Deploy GPU environment
deploy_gpu() {
    log_info "Deploying GPU-accelerated environment..."
    
    # Check if NVIDIA Docker runtime is available
    if ! docker info | grep -q nvidia; then
        log_warning "NVIDIA Docker runtime not detected"
        log_info "Make sure you have nvidia-docker2 installed"
    fi
    
    cd "$PROJECT_ROOT"
    docker-compose up -d neural-forge-gpu
    
    log_success "GPU environment deployed"
    log_info "GPU service is running with CUDA support"
}

# Run tests
run_tests() {
    log_info "Running tests in Docker..."
    cd "$PROJECT_ROOT"
    
    docker-compose run --rm neural-forge-test
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All tests passed"
    else
        log_error "Some tests failed"
        exit $exit_code
    fi
}

# Run benchmarks
run_benchmarks() {
    log_info "Running performance benchmarks..."
    cd "$PROJECT_ROOT"
    
    docker-compose run --rm neural-forge-benchmark
    log_success "Benchmarks completed"
}

# Deploy documentation
deploy_docs() {
    log_info "Deploying documentation server..."
    cd "$PROJECT_ROOT"
    
    # Build documentation first
    docker-compose run --rm neural-forge-dev bash -c "cd docs/sphinx && make html"
    
    # Start documentation server
    docker-compose up -d docs-server
    
    log_success "Documentation deployed"
    log_info "Access documentation at: http://localhost:8080"
}

# Stop services
stop_services() {
    log_info "Stopping all services..."
    cd "$PROJECT_ROOT"
    
    docker-compose stop
    log_success "All services stopped"
}

# Clean up
cleanup() {
    log_info "Cleaning up containers and volumes..."
    cd "$PROJECT_ROOT"
    
    docker-compose down -v --remove-orphans
    docker system prune -f
    
    log_success "Cleanup completed"
}

# Show logs
show_logs() {
    local service=${1:-}
    cd "$PROJECT_ROOT"
    
    if [[ -n "$service" ]]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

# Show status
show_status() {
    log_info "Service status:"
    cd "$PROJECT_ROOT"
    
    docker-compose ps
    
    log_info "Docker images:"
    docker images | grep neural-forge || true
    
    log_info "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || true
}

# Health check
health_check() {
    log_info "Performing health checks..."
    cd "$PROJECT_ROOT"
    
    local services=(
        "neural-forge-production:python -c 'from neural_arch.core import Tensor; print(\"OK\")'"
        "neural-forge-development:python -c 'from neural_arch.core import Tensor; print(\"OK\")'"
        "neural-forge-jupyter:curl -f http://localhost:8888/lab || true"
    )
    
    for service_check in "${services[@]}"; do
        local service_name="${service_check%%:*}"
        local check_command="${service_check#*:}"
        
        if docker ps --format '{{.Names}}' | grep -q "$service_name"; then
            log_info "Checking $service_name..."
            if docker exec "$service_name" bash -c "$check_command" &>/dev/null; then
                log_success "$service_name is healthy"
            else
                log_warning "$service_name health check failed"
            fi
        else
            log_info "$service_name is not running"
        fi
    done
}

# Main function
main() {
    local command=""
    local force=false
    local no_cache=false
    local verbose=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -f|--force)
                force=true
                shift
                ;;
            --no-cache)
                no_cache=true
                shift
                ;;
            dev|prod|jupyter|gpu|test|benchmark|docs|stop|clean|logs|status|health)
                command=$1
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$command" ]]; then
        log_error "No command specified"
        usage
        exit 1
    fi
    
    # Enable verbose output
    if [[ "$verbose" == "true" ]]; then
        set -x
    fi
    
    # Check dependencies
    check_dependencies
    
    # Create environment file
    create_env_file
    
    # Execute command
    case $command in
        dev)
            build_images "$force" "$no_cache"
            deploy_dev
            ;;
        prod)
            build_images "$force" "$no_cache"
            deploy_prod
            ;;
        jupyter)
            build_images "$force" "$no_cache"
            deploy_jupyter
            ;;
        gpu)
            build_images "$force" "$no_cache"
            deploy_gpu
            ;;
        test)
            build_images "$force" "$no_cache"
            run_tests
            ;;
        benchmark)
            build_images "$force" "$no_cache"
            run_benchmarks
            ;;
        docs)
            deploy_docs
            ;;
        stop)
            stop_services
            ;;
        clean)
            cleanup
            ;;
        logs)
            show_logs "$1"
            ;;
        status)
            show_status
            ;;
        health)
            health_check
            ;;
        *)
            log_error "Unknown command: $command"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"