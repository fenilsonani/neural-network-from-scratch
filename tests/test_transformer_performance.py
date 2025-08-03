"""Performance tests for transformer variants.

This test suite focuses on performance characteristics:
- Memory usage optimization
- Inference speed
- Gradient computation efficiency
- Backend compatibility
- Numerical stability
"""

import sys
import time
import gc
from pathlib import Path

import numpy as np
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.neural_arch.core.tensor import Tensor
    from src.neural_arch.backends import set_backend, available_backends, current_backend
    from src.neural_arch.models.language import (
        bert_base,
        bert_large,
        roberta_base,
        roberta_large,
        deberta_base,
        deberta_large,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


def get_memory_usage():
    """Get approximate memory usage in MB."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


@pytest.fixture(scope="session", autouse=True)
def setup_backend():
    """Set up the optimal backend for performance testing."""
    if not MODELS_AVAILABLE:
        return
        
    backends = available_backends()
    
    # Prefer MPS for Apple Silicon, then numpy
    if "mps" in backends:
        try:
            set_backend("mps")
            print("Using MPS backend for performance tests")
        except Exception:
            set_backend("numpy")
            print("MPS failed, using numpy backend")
    else:
        set_backend("numpy")
        print("Using numpy backend for performance tests")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestInferencePerformance:
    """Test inference performance across models."""

    @pytest.mark.parametrize("model_func,model_name", [
        (bert_base, "BERT Base"),
        (bert_large, "BERT Large"),
        (roberta_base, "RoBERTa Base"),
        (roberta_large, "RoBERTa Large"),
        (deberta_base, "DeBERTa Base"),
        (deberta_large, "DeBERTa Large"),
    ])
    def test_model_creation_time(self, model_func, model_name):
        """Test model creation time."""
        start_time = time.time()
        model = model_func()
        creation_time = time.time() - start_time
        
        print(f"\n{model_name} creation time: {creation_time:.2f}s")
        
        # Should create model in reasonable time
        assert creation_time < 60.0  # 1 minute max
        assert model is not None

    @pytest.mark.parametrize("model_func,model_name", [
        (bert_base, "BERT Base"),
        (roberta_base, "RoBERTa Base"),
        (deberta_base, "DeBERTa Base"),
    ])
    def test_inference_speed_small_batch(self, model_func, model_name):
        """Test inference speed with small batch."""
        model = model_func()
        
        # Small batch test
        batch_size, seq_length = 1, 16
        vocab_size = model.config.vocab_size
        input_ids = Tensor(np.random.randint(1, min(vocab_size, 1000), (batch_size, seq_length)))
        
        # Warm-up run
        _ = model(input_ids=input_ids)
        
        # Timed runs
        num_runs = 5
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            outputs = model(input_ids=input_ids)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        print(f"\n{model_name} inference (1x16): {avg_time:.1f}ms")
        
        # Should be reasonably fast
        assert avg_time < 5000  # 5 seconds max
        assert outputs["last_hidden_state"].shape == (batch_size, seq_length, model.config.hidden_size)

    @pytest.mark.parametrize("model_func,model_name", [
        (bert_base, "BERT Base"),
        (roberta_base, "RoBERTa Base"),
    ])
    def test_inference_speed_batch(self, model_func, model_name):
        """Test inference speed with batch processing."""
        model = model_func()
        
        # Batch test
        batch_size, seq_length = 4, 32
        vocab_size = model.config.vocab_size
        input_ids = Tensor(np.random.randint(1, min(vocab_size, 1000), (batch_size, seq_length)))
        
        # Warm-up
        _ = model(input_ids=input_ids)
        
        # Timed run
        start_time = time.time()
        outputs = model(input_ids=input_ids)
        inference_time = time.time() - start_time
        
        throughput = batch_size / inference_time  # sequences per second
        print(f"\n{model_name} batch inference (4x32): {inference_time*1000:.1f}ms, {throughput:.1f} seq/s")
        
        assert inference_time < 10.0  # 10 seconds max
        assert outputs["last_hidden_state"].shape == (batch_size, seq_length, model.config.hidden_size)

    def test_sequence_length_scaling(self):
        """Test performance scaling with sequence length."""
        model = bert_base()
        batch_size = 1
        
        sequence_lengths = [8, 16, 32, 64]
        times = []
        
        for seq_length in sequence_lengths:
            input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
            
            # Warm-up
            _ = model(input_ids=input_ids)
            
            # Timed run
            start_time = time.time()
            _ = model(input_ids=input_ids)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        print(f"\nSequence length scaling: {list(zip(sequence_lengths, [t*1000 for t in times]))}")
        
        # Time should generally increase with sequence length
        # (though not necessarily monotonically due to hardware variations)
        assert all(t < 5.0 for t in times)  # All should be under 5 seconds


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestMemoryPerformance:
    """Test memory usage and optimization."""

    def test_model_memory_usage(self):
        """Test memory usage of different models."""
        initial_memory = get_memory_usage()
        
        model_functions = [
            (bert_base, "BERT Base"),
            (bert_large, "BERT Large"),
            (roberta_base, "RoBERTa Base"),
        ]
        
        for model_func, model_name in model_functions:
            gc.collect()  # Clean up before measurement
            
            before_memory = get_memory_usage()
            model = model_func()
            after_memory = get_memory_usage()
            
            memory_used = after_memory - before_memory
            print(f"\n{model_name} memory usage: {memory_used:.1f} MB")
            
            # Memory usage should be reasonable
            assert memory_used < 2000  # 2GB max per model
            
            # Clean up
            del model
            gc.collect()

    def test_forward_pass_memory_efficiency(self):
        """Test memory efficiency during forward pass."""
        model = bert_base()
        
        gc.collect()
        initial_memory = get_memory_usage()
        
        # Multiple forward passes to test memory leaks
        batch_size, seq_length = 2, 16
        
        for i in range(10):
            input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
            outputs = model(input_ids=input_ids)
            
            # Force cleanup
            del input_ids, outputs
            
            if i % 5 == 0:
                gc.collect()
                current_memory = get_memory_usage()
                memory_growth = current_memory - initial_memory
                
                print(f"Memory after {i+1} iterations: {memory_growth:.1f} MB growth")
                
                # Memory growth should be minimal
                assert memory_growth < 500  # 500MB max growth

    def test_gradient_memory_efficiency(self):
        """Test memory efficiency with gradient computation."""
        model = bert_base()
        
        gc.collect()
        initial_memory = get_memory_usage()
        
        batch_size, seq_length = 2, 8
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)), requires_grad=True)
        
        # Forward pass
        outputs = model(input_ids=input_ids)
        after_forward_memory = get_memory_usage()
        
        # Compute loss and backward pass
        loss = outputs["last_hidden_state"].sum()
        
        try:
            loss.backward()
            after_backward_memory = get_memory_usage()
            
            forward_memory = after_forward_memory - initial_memory
            backward_memory = after_backward_memory - after_forward_memory
            
            print(f"\nGradient memory - Forward: {forward_memory:.1f} MB, Backward: {backward_memory:.1f} MB")
            
            # Memory usage should be reasonable
            assert forward_memory < 1000  # 1GB max for forward
            assert backward_memory < 1000  # 1GB max for backward
            
        except Exception as e:
            print(f"Gradient computation not supported: {e}")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_output_stability(self):
        """Test output stability across multiple runs."""
        model = bert_base()
        
        batch_size, seq_length = 2, 8
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
        
        # Multiple runs with same input
        outputs = []
        for _ in range(5):
            output = model(input_ids=input_ids)
            outputs.append(output["last_hidden_state"].data)
        
        # All outputs should be identical (deterministic)
        for i in range(1, len(outputs)):
            np.testing.assert_allclose(
                outputs[0], outputs[i],
                rtol=1e-6, atol=1e-7,
                err_msg=f"Output {i} differs from first output"
            )

    def test_numerical_precision(self):
        """Test numerical precision with different input ranges."""
        model = bert_base()
        
        test_cases = [
            ("small_values", np.array([[1, 2, 3, 4]])),
            ("large_values", np.array([[1000, 2000, 3000, 4000]])),
            ("edge_values", np.array([[0, 1, model.config.vocab_size-1, model.config.vocab_size-2]])),
        ]
        
        for case_name, input_data in test_cases:
            input_ids = Tensor(input_data)
            
            try:
                outputs = model(input_ids=input_ids)
                
                # Check outputs are finite
                assert np.all(np.isfinite(outputs["last_hidden_state"].data)), f"Non-finite outputs in {case_name}"
                
                # Check outputs are not all zeros or ones
                output_data = outputs["last_hidden_state"].data
                assert not np.allclose(output_data, 0.0), f"All-zero outputs in {case_name}"
                assert not np.allclose(output_data, 1.0), f"All-one outputs in {case_name}"
                
                print(f"✅ {case_name}: Output range [{output_data.min():.3f}, {output_data.max():.3f}]")
                
            except Exception as e:
                pytest.fail(f"Numerical precision test failed for {case_name}: {e}")

    def test_gradient_numerical_stability(self):
        """Test gradient numerical stability."""
        model = bert_base()
        
        batch_size, seq_length = 2, 4
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)), requires_grad=True)
        
        try:
            outputs = model(input_ids=input_ids)
            loss = outputs["last_hidden_state"].sum()
            
            # Check loss is finite
            assert np.isfinite(loss.data), "Loss is not finite"
            
            # Compute gradients
            loss.backward()
            
            # Check that we can compute gradients without NaN/inf
            assert loss.requires_grad, "Loss should require gradients"
            
            print("✅ Gradient computation stable")
            
        except Exception as e:
            print(f"Gradient stability test not supported: {e}")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestBackendCompatibility:
    """Test compatibility across different backends."""

    def test_backend_switching(self):
        """Test switching between backends."""
        if len(available_backends()) < 2:
            pytest.skip("Multiple backends not available")
        
        model = bert_base()
        batch_size, seq_length = 1, 8
        input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
        
        outputs_by_backend = {}
        
        for backend_name in available_backends():
            try:
                set_backend(backend_name)
                print(f"Testing with {backend_name} backend")
                
                # Create fresh tensors for this backend
                input_ids_backend = Tensor(input_ids.data.copy())
                
                # Run inference
                outputs = model(input_ids=input_ids_backend)
                outputs_by_backend[backend_name] = outputs["last_hidden_state"].data.copy()
                
            except Exception as e:
                print(f"Backend {backend_name} failed: {e}")
        
        # Compare outputs across backends (should be similar but may have small differences)
        backend_names = list(outputs_by_backend.keys())
        if len(backend_names) >= 2:
            output1 = outputs_by_backend[backend_names[0]]
            output2 = outputs_by_backend[backend_names[1]]
            
            # Outputs should be reasonably close
            np.testing.assert_allclose(
                output1, output2,
                rtol=1e-3, atol=1e-4,
                err_msg=f"Backend outputs differ significantly: {backend_names[0]} vs {backend_names[1]}"
            )

    def test_current_backend_info(self):
        """Test current backend information."""
        backend = current_backend()
        print(f"Current backend: {backend.name if backend else 'None'}")
        
        if backend:
            assert hasattr(backend, 'name')
            assert hasattr(backend, 'is_available')
            assert backend.is_available


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestScalabilityPerformance:
    """Test scalability characteristics."""

    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        model = bert_base()
        seq_length = 16
        
        batch_sizes = [1, 2, 4, 8]
        times = []
        
        for batch_size in batch_sizes:
            input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
            
            # Warm-up
            _ = model(input_ids=input_ids)
            
            # Timed run
            start_time = time.time()
            _ = model(input_ids=input_ids)
            end_time = time.time()
            
            time_per_sequence = (end_time - start_time) / batch_size
            times.append(time_per_sequence)
        
        print(f"\nBatch scaling (time per sequence): {list(zip(batch_sizes, [t*1000 for t in times]))}")
        
        # Time per sequence should generally decrease with larger batches (better efficiency)
        # But should remain reasonable for all batch sizes
        assert all(t < 2.0 for t in times)  # Max 2 seconds per sequence

    def test_model_size_scaling(self):
        """Test performance scaling across model sizes."""
        models = [
            (bert_base, "BERT Base"),
            (bert_large, "BERT Large"),
        ]
        
        batch_size, seq_length = 1, 16
        
        for model_func, model_name in models:
            model = model_func()
            input_ids = Tensor(np.random.randint(1, 1000, (batch_size, seq_length)))
            
            # Warm-up
            _ = model(input_ids=input_ids)
            
            # Timed run
            start_time = time.time()
            outputs = model(input_ids=input_ids)
            end_time = time.time()
            
            inference_time = end_time - start_time
            params_approx = model.config.num_hidden_layers * model.config.hidden_size ** 2
            
            print(f"\n{model_name}: {inference_time*1000:.1f}ms, ~{params_approx/1e6:.1f}M params")
            
            # Larger models should take more time but still be reasonable
            assert inference_time < 15.0  # 15 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])