#!/bin/bash
"""Comprehensive Distributed Training Test Runner.

This script runs a complete suite of distributed training tests and validation
for Neural Forge, covering single-node, multi-node, and performance scenarios.
"""

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/test_results/distributed"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Neural Forge Distributed Training Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Test Results Directory: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Test configuration
PYTHON_CMD="python3"
BACKENDS=("gloo")  # Start with CPU backend
WORLD_SIZES=(1 2 4)
TEST_TIMEOUT=300  # 5 minutes

# Function to run a test and capture results
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local log_file="$RESULTS_DIR/${test_name}_${TIMESTAMP}.log"
    local result_file="$RESULTS_DIR/${test_name}_${TIMESTAMP}.json"
    
    echo -e "${BLUE}Running: $test_name${NC}"
    echo "Command: $test_cmd"
    echo "Log: $log_file"
    echo ""
    
    # Run test with timeout
    if timeout $TEST_TIMEOUT bash -c "$test_cmd" > "$log_file" 2>&1; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        echo "PASSED" > "${log_file}.status"
        return 0
    else
        local exit_code=$?
        echo -e "${RED}❌ FAILED: $test_name (exit code: $exit_code)${NC}"
        echo "FAILED" > "${log_file}.status"
        echo "Exit Code: $exit_code" >> "${log_file}.status"
        
        # Show last few lines of log for immediate feedback
        echo -e "${YELLOW}Last 10 lines of output:${NC}"
        tail -n 10 "$log_file" || echo "No output available"
        echo ""
        return 1
    fi
}

# Function to check Python environment
check_environment() {
    echo -e "${BLUE}=== Environment Check ===${NC}"
    
    # Check Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo -e "${RED}Python not found: $PYTHON_CMD${NC}"
        exit 1
    fi
    
    echo "Python: $(which $PYTHON_CMD)"
    echo "Version: $($PYTHON_CMD --version)"
    
    # Check Neural Forge installation
    if ! $PYTHON_CMD -c "import sys; sys.path.insert(0, '$PROJECT_ROOT/src'); import neural_arch" 2>/dev/null; then
        echo -e "${RED}Neural Forge not importable${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Environment check passed${NC}"
    echo ""
}

# Function to run basic functionality tests
run_basic_tests() {
    echo -e "${BLUE}=== Basic Functionality Tests ===${NC}"
    
    # Test 1: Basic distributed imports and initialization
    run_test "basic_imports" "cd '$PROJECT_ROOT' && $PYTHON_CMD -c \"
import sys
sys.path.insert(0, 'src')
from neural_arch.distributed import init_process_group, get_distributed_info
print('✅ Imports successful')
info = get_distributed_info()
print(f'✅ Distributed info: {info}')
\""
    
    # Test 2: Single process distributed functionality
    run_test "single_process_distributed" "cd '$PROJECT_ROOT' && $PYTHON_CMD -c \"
import sys
sys.path.insert(0, 'src')
from neural_arch.distributed import init_process_group, destroy_process_group, is_initialized, get_world_size, get_rank
init_process_group(backend='gloo', world_size=1, rank=0)
print(f'✅ Initialized: {is_initialized()}')
print(f'✅ World size: {get_world_size()}')
print(f'✅ Rank: {get_rank()}')
destroy_process_group()
print('✅ Single process test completed')
\""
    
    # Test 3: Communication primitives
    run_test "communication_primitives" "cd '$PROJECT_ROOT' && $PYTHON_CMD -c \"
import sys, numpy as np
sys.path.insert(0, 'src')
from neural_arch.core import Tensor
from neural_arch.distributed import init_process_group, destroy_process_group, all_reduce, all_gather, broadcast, ReduceOp
init_process_group(backend='gloo', world_size=1, rank=0)
tensor = Tensor(np.array([1.0, 2.0, 3.0]), dtype=np.float32)
result = all_reduce(tensor, ReduceOp.SUM)
print(f'✅ All-reduce: {result.data}')
gathered = all_gather(tensor)
print(f'✅ All-gather: {len(gathered)} tensors')
broadcast_result = broadcast(tensor, src=0)
print(f'✅ Broadcast: {broadcast_result.data}')
destroy_process_group()
print('✅ Communication primitives test completed')
\""
    
    echo ""
}

# Function to run comprehensive validation
run_validation_suite() {
    echo -e "${BLUE}=== Comprehensive Validation Suite ===${NC}"
    
    for backend in "${BACKENDS[@]}"; do
        for world_size in "${WORLD_SIZES[@]}"; do
            test_name="validation_${backend}_ws${world_size}"
            test_cmd="cd '$PROJECT_ROOT' && $PYTHON_CMD scripts/validate_distributed_training.py --backend $backend --world-size $world_size --output '$RESULTS_DIR/${test_name}_${TIMESTAMP}_report.json'"
            
            run_test "$test_name" "$test_cmd"
        done
    done
    
    echo ""
}

# Function to run distributed sampler tests
run_sampler_tests() {
    echo -e "${BLUE}=== Distributed Sampler Tests ===${NC}"
    
    run_test "sampler_functionality" "cd '$PROJECT_ROOT' && $PYTHON_CMD -c \"
import sys
sys.path.insert(0, 'src')
from neural_arch.distributed import DistributedSampler
import numpy as np

# Test basic functionality
dataset_size = 100
world_size = 4

all_indices = set()
for rank in range(world_size):
    sampler = DistributedSampler(dataset_size, world_size, rank, shuffle=False)
    indices = list(sampler)
    all_indices.update(indices)
    print(f'Rank {rank}: {len(indices)} samples')

print(f'✅ Total unique indices: {len(all_indices)}')
print(f'✅ Coverage complete: {len(all_indices) == dataset_size}')

# Test shuffling
sampler = DistributedSampler(50, 1, 0, shuffle=True)
sampler.set_epoch(0)
indices_0 = list(sampler)
sampler.set_epoch(1)
indices_1 = list(sampler)
print(f'✅ Shuffling works: {indices_0 != indices_1}')
print('✅ Distributed sampler tests completed')
\""
    
    echo ""
}

# Function to run performance benchmarks
run_performance_tests() {
    echo -e "${BLUE}=== Performance Benchmarks ===${NC}"
    
    run_test "communication_performance" "cd '$PROJECT_ROOT' && $PYTHON_CMD -c \"
import sys, time, numpy as np
sys.path.insert(0, 'src')
from neural_arch.core import Tensor
from neural_arch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

init_process_group(backend='gloo', world_size=1, rank=0)

sizes = [100, 500, 1000]
for size in sizes:
    tensor = Tensor(np.random.randn(size, size), dtype=np.float32)
    
    # Warmup
    for _ in range(3):
        all_reduce(tensor, ReduceOp.SUM)
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        all_reduce(tensor, ReduceOp.SUM)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10 * 1000  # ms
    print(f'✅ {size}x{size} all-reduce: {avg_time:.2f} ms')

destroy_process_group()
print('✅ Performance benchmarks completed')
\""
    
    echo ""
}

# Function to run example training
run_training_example() {
    echo -e "${BLUE}=== Training Example ===${NC}"
    
    run_test "training_example" "cd '$PROJECT_ROOT' && $PYTHON_CMD examples/distributed_training_example.py --epochs 2 --steps-per-epoch 5 --batch-size 4 --dataset-size 100"
    
    echo ""
}

# Function to run comprehensive tests
run_comprehensive_tests() {
    echo -e "${BLUE}=== Comprehensive Test Suite ===${NC}"
    
    run_test "comprehensive_tests" "cd '$PROJECT_ROOT' && $PYTHON_CMD tests/test_distributed_comprehensive.py"
    
    echo ""
}

# Function to generate summary report
generate_summary() {
    echo -e "${BLUE}=== Test Summary Report ===${NC}"
    
    local summary_file="$RESULTS_DIR/summary_${TIMESTAMP}.txt"
    
    echo "Neural Forge Distributed Training Test Summary" > "$summary_file"
    echo "=============================================" >> "$summary_file"
    echo "Timestamp: $TIMESTAMP" >> "$summary_file"
    echo "Test Results Directory: $RESULTS_DIR" >> "$summary_file"
    echo "" >> "$summary_file"
    
    local total_tests=0
    local passed_tests=0
    
    for status_file in "$RESULTS_DIR"/*_"$TIMESTAMP".log.status; do
        if [[ -f "$status_file" ]]; then
            total_tests=$((total_tests + 1))
            local test_name=$(basename "$status_file" .log.status | sed "s/_${TIMESTAMP}//")
            local status=$(cat "$status_file" | head -n1)
            
            if [[ "$status" == "PASSED" ]]; then
                passed_tests=$((passed_tests + 1))
                echo "✅ $test_name: PASSED" >> "$summary_file"
            else
                echo "❌ $test_name: FAILED" >> "$summary_file"
            fi
        fi
    done
    
    echo "" >> "$summary_file"
    echo "Overall Results:" >> "$summary_file"
    echo "  Total Tests: $total_tests" >> "$summary_file"
    echo "  Passed: $passed_tests" >> "$summary_file"
    echo "  Failed: $((total_tests - passed_tests))" >> "$summary_file"
    
    if [[ $passed_tests -eq $total_tests ]]; then
        echo "  Status: ✅ ALL TESTS PASSED" >> "$summary_file"
        echo -e "${GREEN}🎉 ALL TESTS PASSED ($passed_tests/$total_tests)${NC}"
    else
        echo "  Status: ❌ SOME TESTS FAILED" >> "$summary_file"
        echo -e "${RED}❌ SOME TESTS FAILED ($passed_tests/$total_tests)${NC}"
    fi
    
    echo "" >> "$summary_file"
    echo "Detailed logs available in: $RESULTS_DIR" >> "$summary_file"
    
    # Display summary
    echo ""
    cat "$summary_file"
    echo ""
    echo "Full summary saved to: $summary_file"
}

# Function to cleanup
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    # Kill any remaining processes
    pkill -f "distributed_training" 2>/dev/null || true
    pkill -f "validate_distributed" 2>/dev/null || true
}

# Main execution
main() {
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Parse command line arguments
    SKIP_BASIC=false
    SKIP_VALIDATION=false
    SKIP_PERFORMANCE=false
    SKIP_TRAINING=false
    SKIP_COMPREHENSIVE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-basic)
                SKIP_BASIC=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --skip-performance)
                SKIP_PERFORMANCE=true
                shift
                ;;
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --skip-comprehensive)
                SKIP_COMPREHENSIVE=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-basic         Skip basic functionality tests"
                echo "  --skip-validation    Skip validation suite"
                echo "  --skip-performance   Skip performance tests"
                echo "  --skip-training      Skip training example"
                echo "  --skip-comprehensive Skip comprehensive tests"
                echo "  --help, -h          Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run test suite
    check_environment
    
    if [[ "$SKIP_BASIC" != true ]]; then
        run_basic_tests
    fi
    
    if [[ "$SKIP_VALIDATION" != true ]]; then
        run_validation_suite
    fi
    
    run_sampler_tests
    
    if [[ "$SKIP_PERFORMANCE" != true ]]; then
        run_performance_tests
    fi
    
    if [[ "$SKIP_TRAINING" != true ]]; then
        run_training_example
    fi
    
    if [[ "$SKIP_COMPREHENSIVE" != true ]]; then
        run_comprehensive_tests
    fi
    
    # Generate summary
    generate_summary
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Neural Forge Distributed Tests Complete${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Run main function
main "$@"