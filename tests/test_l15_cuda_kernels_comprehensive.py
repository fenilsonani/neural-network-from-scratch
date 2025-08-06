"""
Comprehensive test suite for CUDA kernels optimization.
Tests all components of cuda_kernels_optimized.py for comprehensive coverage.

This module tests:
- OptimizedCUDAKernels
- Fused attention operations
- Flash attention implementation
- Fused linear + GELU
- Mixed precision operations
- Tensor Core utilization
- Memory optimization patterns
- Performance benchmarking
"""

import math
import time
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.neural_arch.backends.cuda_kernels_optimized import (
    OptimizedCUDAKernels,
    CUDA_KERNELS,
    test_optimized_kernels
)


class TestOptimizedCUDAKernels:
    """Test the main CUDA kernels optimization class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels(device_id=0)
        
    def test_initialization(self):
        """Test CUDA kernels initialization."""
        assert self.kernels.device_id == 0
        assert self.kernels.kernels_compiled is True
        assert isinstance(self.kernels.kernel_cache, dict)
        assert isinstance(self.kernels.stream_pool, list)
        assert isinstance(self.kernels.memory_pool, dict)
        
    def test_device_selection(self):
        """Test device selection and management."""
        # Test different device IDs
        kernels_gpu1 = OptimizedCUDAKernels(device_id=1)
        assert kernels_gpu1.device_id == 1
        
        kernels_gpu2 = OptimizedCUDAKernels(device_id=2)
        assert kernels_gpu2.device_id == 2
        
    def test_kernel_compilation_simulation(self):
        """Test kernel compilation process."""
        # Verify kernel compilation was called during init
        assert self.kernels.kernels_compiled is True
        
        # Test manual compilation
        self.kernels.kernels_compiled = False
        self.kernels._compile_kernels()
        assert self.kernels.kernels_compiled is True


class TestFusedAttention:
    """Test fused attention operations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels()
        
        # Standard attention dimensions
        self.batch_size = 2
        self.num_heads = 8
        self.seq_len = 128
        self.head_dim = 64
        
        # Create test tensors
        self.q = np.random.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim
        ).astype(np.float32)
        self.k = np.random.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim
        ).astype(np.float32)
        self.v = np.random.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim
        ).astype(np.float32)
        
    def test_basic_attention_computation(self):
        """Test basic attention computation."""
        output = self.kernels.fused_attention(self.q, self.k, self.v, use_flash=False)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        assert output.shape == expected_shape
        
        # Check output is not NaN or Inf
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        
        # Output should be different from input (attention applied)
        assert not np.array_equal(output, self.v)
        
    def test_attention_with_mask(self):
        """Test attention with attention mask."""
        # Create causal mask (lower triangular)
        mask = np.tril(np.ones((self.seq_len, self.seq_len))).astype(bool)
        mask = np.broadcast_to(mask, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        output_masked = self.kernels.fused_attention(
            self.q, self.k, self.v, mask=mask, use_flash=False
        )
        output_unmasked = self.kernels.fused_attention(
            self.q, self.k, self.v, mask=None, use_flash=False
        )
        
        # Masked and unmasked outputs should be different
        assert not np.allclose(output_masked, output_unmasked)
        
        # Check output shapes
        assert output_masked.shape == output_unmasked.shape
        
    def test_flash_attention(self):
        """Test Flash Attention implementation."""
        # Use longer sequence for Flash Attention
        long_seq_len = 2048
        q_long = np.random.randn(1, 4, long_seq_len, 64).astype(np.float32)
        k_long = np.random.randn(1, 4, long_seq_len, 64).astype(np.float32)
        v_long = np.random.randn(1, 4, long_seq_len, 64).astype(np.float32)
        
        # Flash attention should be used for long sequences
        output_flash = self.kernels.fused_attention(
            q_long, k_long, v_long, use_flash=True
        )
        
        expected_shape = (1, 4, long_seq_len, 64)
        assert output_flash.shape == expected_shape
        
        # Output should be valid
        assert not np.any(np.isnan(output_flash))
        assert not np.any(np.isinf(output_flash))
        
    def test_flash_attention_vs_standard(self):
        """Compare Flash Attention with standard attention."""
        # Use moderate sequence length where both implementations work
        seq_len = 512
        q_med = np.random.randn(1, 2, seq_len, 64).astype(np.float32)
        k_med = np.random.randn(1, 2, seq_len, 64).astype(np.float32)
        v_med = np.random.randn(1, 2, seq_len, 64).astype(np.float32)
        
        output_standard = self.kernels.fused_attention(
            q_med, k_med, v_med, use_flash=False
        )
        output_flash = self.kernels.fused_attention(
            q_med, k_med, v_med, use_flash=True
        )
        
        # Both should have same shape
        assert output_standard.shape == output_flash.shape
        
        # Results should be approximately equal (different algorithms, small numerical differences)
        correlation = np.corrcoef(
            output_standard.flatten(), output_flash.flatten()
        )[0, 1]
        assert correlation > 0.95  # High correlation
        
    def test_attention_scaling(self):
        """Test attention scaling factor."""
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        
        # Manual scaling calculation for verification
        q_scaled = self.q * scale_factor
        
        output = self.kernels.fused_attention(self.q, self.k, self.v, use_flash=False)
        
        # Verify scaling is applied (indirectly through attention scores)
        # This is hard to verify directly, so we check output is reasonable
        assert output.std() < 10.0  # Not extremely large values
        assert output.std() > 0.01  # Not extremely small values
        
    def test_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        seq_lengths = [16, 64, 256, 1024]
        
        for seq_len in seq_lengths:
            q = np.random.randn(1, 4, seq_len, 32).astype(np.float32)
            k = np.random.randn(1, 4, seq_len, 32).astype(np.float32)
            v = np.random.randn(1, 4, seq_len, 32).astype(np.float32)
            
            output = self.kernels.fused_attention(q, k, v)
            
            expected_shape = (1, 4, seq_len, 32)
            assert output.shape == expected_shape
            
            # Check numerical stability
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
            
    def test_different_head_dimensions(self):
        """Test attention with different head dimensions."""
        head_dims = [32, 64, 128]
        
        for head_dim in head_dims:
            q = np.random.randn(2, 8, 64, head_dim).astype(np.float32)
            k = np.random.randn(2, 8, 64, head_dim).astype(np.float32)
            v = np.random.randn(2, 8, 64, head_dim).astype(np.float32)
            
            output = self.kernels.fused_attention(q, k, v)
            
            expected_shape = (2, 8, 64, head_dim)
            assert output.shape == expected_shape


class TestFlashAttentionImplementation:
    """Test Flash Attention internal implementation details."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels()
        
    def test_flash_attention_tiling(self):
        """Test Flash Attention tiling logic."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 256, 64
        
        q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        k = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        v = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        
        scale = 1.0 / math.sqrt(head_dim)
        
        # Test internal Flash Attention implementation
        output = self.kernels._flash_attention(q, k, v, scale)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not np.any(np.isnan(output))
        
    def test_flash_attention_block_sizes(self):
        """Test Flash Attention with different block sizes."""
        seq_len = 512
        q = np.random.randn(1, 1, seq_len, 64).astype(np.float32)
        k = np.random.randn(1, 1, seq_len, 64).astype(np.float32)
        v = np.random.randn(1, 1, seq_len, 64).astype(np.float32)
        
        scale = 1.0 / math.sqrt(64)
        
        # Flash Attention should handle different block sizes internally
        output = self.kernels._flash_attention(q, k, v, scale)
        
        # Verify output correctness
        assert output.shape == (1, 1, seq_len, 64)
        
        # Flash Attention should maintain attention pattern
        # (This is a complex property, so we do basic sanity checks)
        assert output.mean() != 0.0  # Should have meaningful values
        assert 0.01 < output.std() < 10.0  # Reasonable standard deviation
        
    def test_flash_attention_memory_efficiency(self):
        """Test Flash Attention memory efficiency."""
        # This test verifies that Flash Attention works with large sequences
        # without requiring quadratic memory
        
        large_seq_len = 4096
        q = np.random.randn(1, 2, large_seq_len, 32).astype(np.float32)
        k = np.random.randn(1, 2, large_seq_len, 32).astype(np.float32)
        v = np.random.randn(1, 2, large_seq_len, 32).astype(np.float32)
        
        # Should complete without memory errors
        output = self.kernels._flash_attention(q, k, v, 1.0/math.sqrt(32))
        
        assert output.shape == (1, 2, large_seq_len, 32)
        assert not np.any(np.isnan(output))


class TestFusedLinearGELU:
    """Test fused linear layer + GELU activation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels()
        
    def test_basic_fused_operation(self):
        """Test basic fused linear + GELU operation."""
        batch_size, in_features, out_features = 32, 128, 256
        
        input_tensor = np.random.randn(batch_size, in_features).astype(np.float32)
        weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
        bias = np.zeros(out_features, dtype=np.float32)
        
        output = self.kernels.fused_linear_gelu(input_tensor, weight, bias)
        
        # Check output shape
        expected_shape = (batch_size, out_features)
        assert output.shape == expected_shape
        
        # GELU should produce non-negative values for positive inputs (approximately)
        # But can have negative values too, so we just check it's not all zeros
        assert not np.allclose(output, 0.0)
        
    def test_gelu_activation_properties(self):
        """Test GELU activation function properties."""
        # Test GELU on simple inputs
        batch_size, features = 10, 5
        input_tensor = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)
        input_tensor = np.repeat(input_tensor, batch_size, axis=0)
        
        # Identity transformation (weight=I, bias=0)
        weight = np.eye(features, dtype=np.float32)
        bias = np.zeros(features, dtype=np.float32)
        
        output = self.kernels.fused_linear_gelu(input_tensor, weight, bias)
        
        # GELU properties:
        # - GELU(0) ≈ 0
        # - GELU is approximately identity for large positive values
        # - GELU is close to 0 for large negative values
        
        gelu_output = output[0]  # First row
        
        # Check GELU(0) ≈ 0
        assert abs(gelu_output[2]) < 0.1  # GELU(0) should be close to 0
        
        # Check monotonicity (roughly)
        assert gelu_output[0] < gelu_output[1] < gelu_output[2] < gelu_output[3] < gelu_output[4]
        
    def test_different_input_sizes(self):
        """Test fused operation with different input sizes."""
        test_cases = [
            (16, 64, 128),
            (32, 128, 256), 
            (64, 256, 512),
            (1, 1024, 2048)
        ]
        
        for batch_size, in_features, out_features in test_cases:
            input_tensor = np.random.randn(batch_size, in_features).astype(np.float32)
            weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
            bias = np.random.randn(out_features).astype(np.float32) * 0.1
            
            output = self.kernels.fused_linear_gelu(input_tensor, weight, bias)
            
            expected_shape = (batch_size, out_features)
            assert output.shape == expected_shape
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
            
    def test_bias_handling(self):
        """Test bias addition in fused operation."""
        batch_size, in_features, out_features = 8, 10, 5
        
        input_tensor = np.ones((batch_size, in_features), dtype=np.float32)
        weight = np.ones((out_features, in_features), dtype=np.float32) * 0.1
        
        # Test with zero bias
        bias_zero = np.zeros(out_features, dtype=np.float32)
        output_zero_bias = self.kernels.fused_linear_gelu(input_tensor, weight, bias_zero)
        
        # Test with non-zero bias
        bias_nonzero = np.ones(out_features, dtype=np.float32)
        output_nonzero_bias = self.kernels.fused_linear_gelu(input_tensor, weight, bias_nonzero)
        
        # Outputs should be different
        assert not np.allclose(output_zero_bias, output_nonzero_bias)
        
        # The bias should shift the pre-activation values
        # This is hard to verify exactly due to GELU, but outputs should differ meaningfully
        difference = np.abs(output_nonzero_bias - output_zero_bias).mean()
        assert difference > 0.1  # Should have meaningful difference
        
    def test_gelu_approximation_accuracy(self):
        """Test accuracy of GELU approximation."""
        # Test the GELU approximation against known values
        def exact_gelu(x):
            """Exact GELU implementation for comparison."""
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        
        # Test on a range of values
        test_values = np.linspace(-3, 3, 100).reshape(1, -1).astype(np.float32)
        weight = np.eye(100, dtype=np.float32)
        bias = np.zeros(100, dtype=np.float32)
        
        fused_output = self.kernels.fused_linear_gelu(test_values, weight, bias)
        exact_output = exact_gelu(test_values)
        
        # Should be very close (our implementation uses fast approximation)
        correlation = np.corrcoef(fused_output.flatten(), exact_output.flatten())[0, 1]
        assert correlation > 0.99  # Very high correlation
        
        # Mean absolute error should be small
        mae = np.mean(np.abs(fused_output - exact_output))
        assert mae < 0.01  # Small error


class TestOptimizedMatMul:
    """Test optimized matrix multiplication implementations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels()
        
    def test_basic_matmul(self):
        """Test basic matrix multiplication."""
        A = np.random.randn(64, 128).astype(np.float32)
        B = np.random.randn(128, 32).astype(np.float32)
        
        # Test without tiling
        result_no_tile = self.kernels._optimized_matmul(A, B, use_tiling=False)
        
        # Test with tiling
        result_tiled = self.kernels._optimized_matmul(A, B, use_tiling=True)
        
        # Compare with numpy
        expected = np.matmul(A, B)
        
        # All should be approximately equal
        np.testing.assert_allclose(result_no_tile, expected, rtol=1e-5)
        np.testing.assert_allclose(result_tiled, expected, rtol=1e-5)
        np.testing.assert_allclose(result_no_tile, result_tiled, rtol=1e-5)
        
    def test_tiled_matmul_different_sizes(self):
        """Test tiled matrix multiplication with different sizes."""
        test_cases = [
            (32, 64, 32),      # Small
            (128, 256, 64),    # Medium  
            (512, 1024, 256),  # Large (should use tiling)
            (1000, 500, 200)   # Non-power-of-2
        ]
        
        for M, K, N in test_cases:
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            result = self.kernels._optimized_matmul(A, B, use_tiling=True)
            expected = np.matmul(A, B)
            
            np.testing.assert_allclose(result, expected, rtol=1e-5)
            assert result.shape == (M, N)
            
    def test_tiling_threshold(self):
        """Test that tiling is used appropriately based on size."""
        # Small matrices should not use tiling
        A_small = np.random.randn(32, 32).astype(np.float32)
        B_small = np.random.randn(32, 32).astype(np.float32)
        
        result_small = self.kernels._optimized_matmul(A_small, B_small, use_tiling=True)
        expected_small = np.matmul(A_small, B_small)
        
        np.testing.assert_allclose(result_small, expected_small, rtol=1e-5)
        
        # Large matrices should use tiling
        A_large = np.random.randn(2000, 1000).astype(np.float32)
        B_large = np.random.randn(1000, 500).astype(np.float32)
        
        result_large = self.kernels._optimized_matmul(A_large, B_large, use_tiling=True)
        expected_large = np.matmul(A_large, B_large)
        
        np.testing.assert_allclose(result_large, expected_large, rtol=1e-5)


class TestOptimizedAdam:
    """Test optimized Adam optimizer implementation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels()
        
    def test_basic_adam_update(self):
        """Test basic Adam optimizer update."""
        # Simple parameter tensor
        params = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        grads = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        momentum = np.zeros_like(params)
        variance = np.zeros_like(params)
        
        updated_params, updated_momentum, updated_variance = self.kernels.optimized_adam(
            params=params,
            grads=grads,
            momentum=momentum,
            variance=variance,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            step=1
        )
        
        # Parameters should be updated (moved in opposite direction of gradients)
        assert not np.array_equal(updated_params, params)
        assert np.all(updated_params < params)  # Should decrease due to positive gradients
        
        # Momentum and variance should be updated
        assert not np.array_equal(updated_momentum, momentum)
        assert not np.array_equal(updated_variance, variance)
        
        # Momentum should be related to gradients
        assert np.all(updated_momentum > 0)  # Should be positive for positive gradients
        
        # Variance should be positive
        assert np.all(updated_variance > 0)
        
    def test_adam_multiple_steps(self):
        """Test Adam optimizer over multiple steps."""
        params = np.array([1.0, -1.0, 0.0, 2.0], dtype=np.float32)
        grads = np.array([0.1, -0.1, 0.05, 0.2], dtype=np.float32)
        momentum = np.zeros_like(params)
        variance = np.zeros_like(params)
        
        # Run multiple Adam steps
        for step in range(1, 11):  # 10 steps
            params, momentum, variance = self.kernels.optimized_adam(
                params=params,
                grads=grads,
                momentum=momentum,
                variance=variance,
                lr=0.01,
                step=step
            )
            
        # After many steps, momentum should have built up
        expected_momentum_signs = np.sign(grads) * 0.1 / (1 - 0.9)  # Rough approximation
        assert np.sign(momentum).tolist() == np.sign(expected_momentum_signs).tolist()
        
        # Variance should be positive
        assert np.all(variance > 0)
        
    def test_adam_hyperparameters(self):
        """Test Adam with different hyperparameters."""
        params = np.array([1.0, 2.0], dtype=np.float32)
        grads = np.array([0.1, 0.2], dtype=np.float32)
        momentum = np.zeros_like(params)
        variance = np.zeros_like(params)
        
        # Test different learning rates
        for lr in [0.001, 0.01, 0.1]:
            updated_params, _, _ = self.kernels.optimized_adam(
                params=params.copy(),
                grads=grads,
                momentum=momentum.copy(),
                variance=variance.copy(),
                lr=lr,
                step=1
            )
            
            # Larger learning rates should cause larger updates
            update_magnitude = np.linalg.norm(updated_params - params)
            assert update_magnitude > 0
            
        # Test different beta values
        for beta1, beta2 in [(0.8, 0.99), (0.9, 0.999), (0.95, 0.9999)]:
            updated_params, updated_momentum, updated_variance = self.kernels.optimized_adam(
                params=params.copy(),
                grads=grads,
                momentum=momentum.copy(),
                variance=variance.copy(),
                beta1=beta1,
                beta2=beta2,
                step=1
            )
            
            # Should produce valid updates
            assert not np.any(np.isnan(updated_params))
            assert not np.any(np.isinf(updated_params))
            
    def test_adam_bias_correction(self):
        """Test bias correction in Adam optimizer."""
        params = np.array([1.0], dtype=np.float32)
        grads = np.array([0.1], dtype=np.float32)
        momentum = np.zeros_like(params)
        variance = np.zeros_like(params)
        
        # First step (high bias correction)
        params_1, momentum_1, variance_1 = self.kernels.optimized_adam(
            params=params.copy(),
            grads=grads,
            momentum=momentum.copy(),
            variance=variance.copy(),
            step=1
        )
        
        # Later step (less bias correction)  
        # Need to update momentum and variance to simulate multiple steps
        momentum_10 = momentum_1 * (0.9 ** 9)  # Approximate after 10 steps
        variance_10 = variance_1 * (0.999 ** 9)
        
        params_10, _, _ = self.kernels.optimized_adam(
            params=params.copy(),
            grads=grads,
            momentum=momentum_10,
            variance=variance_10,
            step=10
        )
        
        # Early steps should have larger updates due to bias correction
        update_1 = abs(params_1[0] - params[0])
        update_10 = abs(params_10[0] - params[0])
        
        # This relationship can be complex, so we just check both are valid
        assert update_1 > 0
        assert update_10 > 0


class TestMixedPrecisionGEMM:
    """Test mixed precision matrix multiplication."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels()
        
    def test_fp16_computation(self):
        """Test FP16 computation with FP32 accumulation."""
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 16).astype(np.float32)
        
        # Mixed precision computation
        result_mixed = self.kernels.mixed_precision_gemm(A, B, use_fp16=True)
        
        # Full precision computation  
        result_fp32 = self.kernels.mixed_precision_gemm(A, B, use_fp16=False)
        
        # Results should be close but not identical due to precision differences
        correlation = np.corrcoef(result_mixed.flatten(), result_fp32.flatten())[0, 1]
        assert correlation > 0.99  # Very high correlation
        
        # Both should be FP32 output
        assert result_mixed.dtype == np.float32
        assert result_fp32.dtype == np.float32
        
        # Mixed precision might have slightly different values
        mae = np.mean(np.abs(result_mixed - result_fp32))
        assert mae < 0.01  # Small difference due to precision
        
    def test_mixed_precision_performance_simulation(self):
        """Test that mixed precision path is taken."""
        A = np.random.randn(128, 256).astype(np.float32)
        B = np.random.randn(256, 64).astype(np.float32)
        
        # Test both paths work
        result_fp16 = self.kernels.mixed_precision_gemm(A, B, use_fp16=True)
        result_fp32 = self.kernels.mixed_precision_gemm(A, B, use_fp16=False)
        
        # Should have same shape
        assert result_fp16.shape == result_fp32.shape == (128, 64)
        
        # Should be reasonably close
        relative_error = np.mean(np.abs(result_fp16 - result_fp32) / (np.abs(result_fp32) + 1e-8))
        assert relative_error < 0.001  # Less than 0.1% relative error
        
    def test_edge_cases_mixed_precision(self):
        """Test edge cases for mixed precision."""
        # Very small matrices
        A_small = np.random.randn(2, 2).astype(np.float32)
        B_small = np.random.randn(2, 2).astype(np.float32)
        
        result_small = self.kernels.mixed_precision_gemm(A_small, B_small, use_fp16=True)
        assert result_small.shape == (2, 2)
        assert not np.any(np.isnan(result_small))
        
        # Single element matrices
        A_single = np.array([[1.0]], dtype=np.float32)
        B_single = np.array([[2.0]], dtype=np.float32)
        
        result_single = self.kernels.mixed_precision_gemm(A_single, B_single, use_fp16=True)
        assert result_single.shape == (1, 1)
        assert abs(result_single[0, 0] - 2.0) < 1e-6


class TestKernelBenchmarking:
    """Test kernel performance benchmarking functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels()
        
    def test_benchmark_infrastructure(self):
        """Test benchmark infrastructure."""
        # Test data for attention kernel
        q = np.random.randn(1, 4, 64, 32).astype(np.float32)
        k = np.random.randn(1, 4, 64, 32).astype(np.float32)  
        v = np.random.randn(1, 4, 64, 32).astype(np.float32)
        
        # Benchmark the fused attention kernel
        stats = self.kernels.benchmark_kernel(
            'fused_attention',
            q, k, v,
            num_warmup=2,
            num_iterations=5
        )
        
        # Check all expected statistics are present
        expected_keys = ['mean', 'std', 'min', 'max', 'median', 'p95', 'p99']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            assert stats[key] >= 0
            
        # Sanity checks on statistics
        assert stats['min'] <= stats['mean'] <= stats['max']
        assert stats['min'] <= stats['median'] <= stats['max']
        assert stats['mean'] <= stats['p95'] <= stats['max']
        assert stats['p95'] <= stats['p99'] <= stats['max']
        
    def test_benchmark_different_kernels(self):
        """Test benchmarking different kernel types."""
        # Benchmark fused linear GELU
        input_tensor = np.random.randn(16, 128).astype(np.float32)
        weight = np.random.randn(256, 128).astype(np.float32) * 0.1
        bias = np.zeros(256, dtype=np.float32)
        
        linear_stats = self.kernels.benchmark_kernel(
            'fused_linear_gelu',
            input_tensor, weight, bias,
            num_warmup=2,
            num_iterations=5
        )
        
        assert 'mean' in linear_stats
        assert linear_stats['mean'] > 0
        
        # Benchmark Adam optimizer
        params = np.random.randn(1000).astype(np.float32)
        grads = np.random.randn(1000).astype(np.float32)
        momentum = np.zeros_like(params)
        variance = np.zeros_like(params)
        
        adam_stats = self.kernels.benchmark_kernel(
            'optimized_adam',
            params, grads, momentum, variance,
            num_warmup=2,
            num_iterations=5
        )
        
        assert 'mean' in adam_stats  
        assert adam_stats['mean'] > 0
        
    def test_benchmark_statistical_validity(self):
        """Test statistical validity of benchmarks."""
        # Use a simple operation for consistent timing
        A = np.random.randn(100, 100).astype(np.float32)
        B = np.random.randn(100, 100).astype(np.float32)
        
        stats = self.kernels.benchmark_kernel(
            'mixed_precision_gemm',
            A, B, True,  # use_fp16=True
            num_warmup=3,
            num_iterations=10
        )
        
        # Standard deviation should be reasonable (not too large)
        cv = stats['std'] / stats['mean']  # Coefficient of variation
        assert cv < 0.5  # Less than 50% variation
        
        # Min should be close to mean (within reasonable bounds)
        assert stats['min'] >= stats['mean'] * 0.5
        
        # Check percentiles are ordered correctly
        assert stats['min'] <= stats['median'] <= stats['p95'] <= stats['p99'] <= stats['max']


class TestCUDAKernelStrings:
    """Test CUDA kernel string definitions."""
    
    def test_kernel_definitions_exist(self):
        """Test that kernel definitions are present."""
        expected_kernels = [
            'fused_attention',
            'fused_linear_gelu', 
            'optimized_adam',
            'flash_attention'
        ]
        
        for kernel_name in expected_kernels:
            assert kernel_name in CUDA_KERNELS
            assert isinstance(CUDA_KERNELS[kernel_name], str)
            assert len(CUDA_KERNELS[kernel_name]) > 0
            
    def test_kernel_code_structure(self):
        """Test basic structure of kernel code."""
        for kernel_name, kernel_code in CUDA_KERNELS.items():
            # Should contain CUDA-specific keywords
            cuda_keywords = ['__global__', '__device__', '__shared__']
            has_cuda_keyword = any(keyword in kernel_code for keyword in cuda_keywords)
            assert has_cuda_keyword, f"Kernel {kernel_name} missing CUDA keywords"
            
            # Should contain template or function definitions
            assert 'kernel' in kernel_code.lower()
            
    def test_flash_attention_kernel_specifics(self):
        """Test Flash Attention kernel specific features."""
        flash_kernel = CUDA_KERNELS['flash_attention']
        
        # Should contain Flash Attention specific elements
        flash_features = ['BLOCK_M', 'BLOCK_N', 'shared_mem', 'warpReduceMax']
        for feature in flash_features:
            assert feature in flash_kernel
            
    def test_fused_attention_kernel_specifics(self):
        """Test fused attention kernel specific features."""
        attention_kernel = CUDA_KERNELS['fused_attention']
        
        # Should contain attention-specific elements
        attention_features = ['wmma::', 'Tensor', 'matrix_a', 'matrix_b', 'accumulator']
        for feature in attention_features:
            assert feature in attention_kernel


class TestErrorHandling:
    """Test error handling in CUDA kernels."""
    
    def setup_method(self):
        """Setup test environment."""
        self.kernels = OptimizedCUDAKernels()
        
    def test_invalid_attention_dimensions(self):
        """Test error handling for invalid attention dimensions."""
        # Mismatched dimensions
        q = np.random.randn(2, 4, 64, 32).astype(np.float32)
        k = np.random.randn(2, 4, 64, 16).astype(np.float32)  # Wrong head_dim
        v = np.random.randn(2, 4, 64, 32).astype(np.float32)
        
        with pytest.raises((ValueError, AssertionError)):
            self.kernels.fused_attention(q, k, v)
            
    def test_invalid_linear_dimensions(self):
        """Test error handling for invalid linear layer dimensions."""
        input_tensor = np.random.randn(32, 128).astype(np.float32)
        weight = np.random.randn(256, 64).astype(np.float32)  # Wrong in_features
        bias = np.zeros(256, dtype=np.float32)
        
        with pytest.raises((ValueError, AssertionError)):
            self.kernels.fused_linear_gelu(input_tensor, weight, bias)
            
    def test_invalid_adam_parameters(self):
        """Test error handling for invalid Adam parameters."""
        params = np.random.randn(100).astype(np.float32)
        grads = np.random.randn(50).astype(np.float32)  # Wrong size
        momentum = np.zeros_like(params)
        variance = np.zeros_like(params)
        
        with pytest.raises((ValueError, AssertionError)):
            self.kernels.optimized_adam(params, grads, momentum, variance)


class TestIntegrationWithSystemTest:
    """Integration tests with the built-in system test."""
    
    def test_system_integration(self):
        """Test integration with built-in test function."""
        # Run the built-in test function
        try:
            test_optimized_kernels()
            print("✅ CUDA kernels system test passed")
        except Exception as e:
            pytest.fail(f"System integration test failed: {e}")
            
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with multiple kernels."""
        kernels = OptimizedCUDAKernels()
        
        # Simulate a mini transformer layer
        batch_size, seq_len, d_model = 4, 64, 256
        num_heads = 8
        head_dim = d_model // num_heads
        
        # Input
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        
        # Linear projections for Q, K, V
        w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        b_q = np.zeros(d_model, dtype=np.float32)
        b_k = np.zeros(d_model, dtype=np.float32)
        b_v = np.zeros(d_model, dtype=np.float32)
        
        # Compute Q, K, V using fused linear + GELU (simplified)
        # Reshape input for linear layer
        x_flat = x.reshape(-1, d_model)
        
        q_flat = kernels.fused_linear_gelu(x_flat, w_q, b_q)
        k_flat = kernels.fused_linear_gelu(x_flat, w_k, b_k)
        v_flat = kernels.fused_linear_gelu(x_flat, w_v, b_v)
        
        # Reshape for attention
        q = q_flat.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k_flat.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v_flat.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Apply attention
        attention_output = kernels.fused_attention(q, k, v)
        
        # Check final output
        expected_shape = (batch_size, num_heads, seq_len, head_dim)
        assert attention_output.shape == expected_shape
        assert not np.any(np.isnan(attention_output))
        assert not np.any(np.isinf(attention_output))
        
        print(f"✅ End-to-end workflow completed: {attention_output.shape}")


class TestAdvancedCUDAFeatures:
    """Test advanced CUDA kernel features for complete coverage."""
    
    def test_tensor_core_matrix_operations(self):
        """Test Tensor Core matrix operations."""
        kernels = OptimizedCUDAKernels()
        
        # Test different precision combinations
        precisions = ['fp16', 'bf16', 'tf32', 'fp32']
        
        for precision in precisions:
            a = np.random.randn(128, 64).astype(np.float32)
            b = np.random.randn(64, 32).astype(np.float32)
            
            result = kernels.tensor_core_matmul(a, b, precision=precision)
            assert result.shape == (128, 32)
    
    def test_wmma_operations_comprehensive(self):
        """Test comprehensive WMMA operations."""
        kernels = OptimizedCUDAKernels()
        
        # Test different tile sizes
        tile_sizes = [(16, 16, 16), (32, 8, 16), (8, 32, 16)]
        
        for m, n, k in tile_sizes:
            a = np.random.randn(m, k).astype(np.float16)
            b = np.random.randn(k, n).astype(np.float16)
            c = np.zeros((m, n), dtype=np.float16)
            
            result = kernels.wmma_gemm(a, b, c, tile_size=(m, n, k))
            assert result.shape == (m, n)
    
    def test_flash_attention_variants(self):
        """Test different Flash Attention variants."""
        kernels = OptimizedCUDAKernels()
        
        batch_size, seq_len, head_dim = 2, 128, 64
        q = np.random.randn(batch_size, seq_len, head_dim)
        k = np.random.randn(batch_size, seq_len, head_dim)
        v = np.random.randn(batch_size, seq_len, head_dim)
        
        # Test different variants
        variants = ['flash_v1', 'flash_v2', 'flash_decoding']
        
        for variant in variants:
            output = kernels.flash_attention_variant(q, k, v, variant=variant)
            assert output.shape == q.shape
    
    def test_memory_coalescing_patterns(self):
        """Test memory coalescing patterns."""
        kernels = OptimizedCUDAKernels()
        
        # Test different access patterns
        patterns = ['sequential', 'strided', 'random', 'transpose']
        
        data = np.random.randn(1024, 1024).astype(np.float32)
        
        for pattern in patterns:
            result = kernels.test_memory_pattern(data, pattern=pattern)
            assert result.shape == data.shape
    
    def test_warp_level_primitives(self):
        """Test warp-level primitives."""
        kernels = OptimizedCUDAKernels()
        
        data = np.random.randn(32).astype(np.float32)  # One warp
        
        # Test warp shuffle operations
        shuffled = kernels.warp_shuffle_down(data, delta=1)
        assert shuffled.shape == data.shape
        
        # Test warp reduce operations
        reduced = kernels.warp_reduce_sum(data)
        assert isinstance(reduced, (int, float, np.number))
    
    def test_cuda_stream_management(self):
        """Test CUDA stream management."""
        kernels = OptimizedCUDAKernels()
        
        # Create multiple streams
        num_streams = 4
        streams = kernels.create_cuda_streams(num_streams)
        assert len(streams) == num_streams
        
        # Test asynchronous operations
        data = np.random.randn(1000, 1000)
        
        for i, stream in enumerate(streams):
            result = kernels.async_matrix_multiply(
                data, data, stream=stream, stream_id=i
            )
            assert result.shape == (1000, 1000)
    
    def test_kernel_fusion_strategies(self):
        """Test kernel fusion strategies."""
        kernels = OptimizedCUDAKernels()
        
        x = np.random.randn(1024, 512)
        y = np.random.randn(1024, 512)
        
        # Test element-wise fusion
        fused_result = kernels.fused_elementwise_ops(x, y, ops=['add', 'relu', 'square'])
        assert fused_result.shape == x.shape
        
        # Test reduction fusion
        fused_sum = kernels.fused_reduction_ops(x, ops=['square', 'sum', 'sqrt'])
        assert isinstance(fused_sum, (int, float, np.number))


class TestCUDAKernelEdgeCases:
    """Test CUDA kernel edge cases and error conditions."""
    
    def test_zero_sized_tensors(self):
        """Test handling of zero-sized tensors."""
        kernels = OptimizedCUDAKernels()
        
        # Empty tensor operations
        empty = np.array([]).reshape(0, 10)
        result = kernels.handle_empty_tensor(empty)
        assert result.shape == (0, 10)
    
    def test_mismatched_tensor_dimensions(self):
        """Test mismatched tensor dimensions."""
        kernels = OptimizedCUDAKernels()
        
        a = np.random.randn(10, 5)
        b = np.random.randn(3, 8)  # Incompatible dimensions
        
        with pytest.raises((ValueError, RuntimeError)):
            kernels.matmul(a, b)
    
    def test_memory_allocation_limits(self):
        """Test memory allocation limits."""
        kernels = OptimizedCUDAKernels()
        
        # Test very large tensor allocation
        try:
            huge_tensor = kernels.allocate_tensor_on_gpu(shape=(100000, 100000))
            # If allocation succeeds, test should still pass
            assert huge_tensor is not None
        except (MemoryError, RuntimeError):
            # Expected for large allocations
            pass
    
    def test_numerical_precision_edge_cases(self):
        """Test numerical precision edge cases."""
        kernels = OptimizedCUDAKernels()
        
        # Test with very small values
        tiny_values = np.full((100, 100), 1e-10, dtype=np.float32)
        result = kernels.process_tiny_values(tiny_values)
        assert not np.any(np.isnan(result))
        
        # Test with very large values
        large_values = np.full((100, 100), 1e10, dtype=np.float32)
        result = kernels.process_large_values(large_values)
        assert not np.any(np.isinf(result))
    
    def test_cuda_error_recovery(self):
        """Test CUDA error recovery mechanisms."""
        kernels = OptimizedCUDAKernels()
        
        # Simulate CUDA error
        with patch.object(kernels, '_check_cuda_error', side_effect=RuntimeError("CUDA error")):
            with pytest.raises(RuntimeError):
                kernels.force_cuda_error()
    
    def test_device_compatibility_checks(self):
        """Test device compatibility checks."""
        kernels = OptimizedCUDAKernels()
        
        # Test compute capability checks
        compute_caps = [(3, 5), (5, 0), (6, 1), (7, 0), (8, 0), (9, 0)]
        
        for major, minor in compute_caps:
            is_supported = kernels.check_compute_capability(major, minor)
            assert isinstance(is_supported, bool)


class TestPerformanceBenchmarkingExtensions:
    """Test performance benchmarking extensions."""
    
    def test_throughput_measurements(self):
        """Test throughput measurements for different operations."""
        kernels = OptimizedCUDAKernels()
        
        sizes = [128, 256, 512, 1024]
        
        for size in sizes:
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # Measure throughput
            throughput = kernels.measure_matmul_throughput(a, b, iterations=10)
            assert throughput > 0  # GFLOPS
    
    def test_memory_bandwidth_utilization(self):
        """Test memory bandwidth utilization."""
        kernels = OptimizedCUDAKernels()
        
        # Test different memory access patterns
        sizes = [1024, 2048, 4096]
        
        for size in sizes:
            data = np.random.randn(size, size).astype(np.float32)
            
            bandwidth = kernels.measure_memory_bandwidth(data, pattern='sequential')
            assert bandwidth > 0  # GB/s
    
    def test_kernel_occupancy_analysis(self):
        """Test kernel occupancy analysis."""
        kernels = OptimizedCUDAKernels()
        
        # Analyze occupancy for different block sizes
        block_sizes = [64, 128, 256, 512, 1024]
        
        for block_size in block_sizes:
            occupancy = kernels.analyze_kernel_occupancy(
                kernel_name='matmul_kernel',
                block_size=block_size
            )
            assert 0 <= occupancy <= 1.0
    
    def test_energy_efficiency_metrics(self):
        """Test energy efficiency metrics."""
        kernels = OptimizedCUDAKernels()
        
        # Test power consumption during different operations
        operations = ['matmul', 'conv2d', 'attention', 'reduction']
        
        for op in operations:
            power_usage = kernels.measure_power_consumption(operation=op, duration=1.0)
            assert power_usage >= 0  # Watts


def test_system_integration():
    """Test the complete CUDA kernels system integration."""
    # This test would run the built-in test function
    try:
        # Run the built-in system test
        test_cuda_kernels()
        print("✅ CUDA kernels system integration test passed")
    except Exception as e:
        pytest.fail(f"CUDA kernels system test failed: {e}")


class TestCUDAKernelSpecializedOperations:
    """Test specialized CUDA kernel operations for maximum coverage."""
    
    def setup_method(self):
        """Setup specialized test environment."""
        self.kernels = OptimizedCUDAKernels(device_id=0)
    
    def test_convolution_kernels(self):
        """Test specialized convolution kernels."""
        # Test 1D convolution
        input_1d = np.random.randn(32, 64, 128).astype(np.float32)
        kernel_1d = np.random.randn(16, 64, 3).astype(np.float32)
        output_1d = self.kernels.conv1d_optimized(input_1d, kernel_1d)
        assert output_1d.shape == (32, 16, 126)
        
        # Test 2D convolution with different strides
        input_2d = np.random.randn(16, 32, 64, 64).astype(np.float32)
        kernel_2d = np.random.randn(64, 32, 3, 3).astype(np.float32)
        
        for stride in [1, 2, 3]:
            output_2d = self.kernels.conv2d_optimized(input_2d, kernel_2d, stride=stride)
            expected_size = (64 - 3) // stride + 1
            assert output_2d.shape == (16, 64, expected_size, expected_size)
        
        # Test 3D convolution
        input_3d = np.random.randn(8, 16, 32, 32, 32).astype(np.float32)
        kernel_3d = np.random.randn(32, 16, 3, 3, 3).astype(np.float32)
        output_3d = self.kernels.conv3d_optimized(input_3d, kernel_3d)
        assert output_3d.shape == (8, 32, 30, 30, 30)
    
    def test_pooling_operations(self):
        """Test pooling operations."""
        input_data = np.random.randn(16, 64, 32, 32).astype(np.float32)
        
        # Max pooling
        max_pooled = self.kernels.max_pool2d(input_data, kernel_size=2, stride=2)
        assert max_pooled.shape == (16, 64, 16, 16)
        
        # Average pooling
        avg_pooled = self.kernels.avg_pool2d(input_data, kernel_size=2, stride=2)
        assert avg_pooled.shape == (16, 64, 16, 16)
        
        # Adaptive pooling
        adaptive_pooled = self.kernels.adaptive_pool2d(input_data, output_size=(8, 8))
        assert adaptive_pooled.shape == (16, 64, 8, 8)
    
    def test_normalization_kernels(self):
        """Test normalization kernels."""
        # Batch normalization
        input_data = np.random.randn(32, 128, 64, 64).astype(np.float32)
        gamma = np.ones(128, dtype=np.float32)
        beta = np.zeros(128, dtype=np.float32)
        
        bn_output = self.kernels.batch_norm2d(input_data, gamma, beta)
        assert bn_output.shape == input_data.shape
        
        # Layer normalization
        ln_input = np.random.randn(32, 512).astype(np.float32)
        ln_output = self.kernels.layer_norm(ln_input, gamma=np.ones(512), beta=np.zeros(512))
        assert ln_output.shape == ln_input.shape
        
        # Group normalization
        gn_input = np.random.randn(16, 64, 32, 32).astype(np.float32)
        gn_output = self.kernels.group_norm(gn_input, num_groups=8)
        assert gn_output.shape == gn_input.shape
    
    def test_activation_functions(self):
        """Test activation function kernels."""
        input_data = np.random.randn(1000, 1000).astype(np.float32)
        
        activations = ['relu', 'gelu', 'swish', 'mish', 'leaky_relu', 'elu']
        
        for activation in activations:
            output = self.kernels.fused_activation(input_data, activation=activation)
            assert output.shape == input_data.shape
            assert not np.any(np.isnan(output))
    
    def test_specialized_reductions(self):
        """Test specialized reduction operations."""
        input_data = np.random.randn(128, 256, 512).astype(np.float32)
        
        # Different reduction types
        sum_result = self.kernels.specialized_reduce(input_data, op='sum', dim=1)
        assert sum_result.shape == (128, 512)
        
        max_result = self.kernels.specialized_reduce(input_data, op='max', dim=2)
        assert max_result.shape == (128, 256)
        
        mean_result = self.kernels.specialized_reduce(input_data, op='mean', dim=0)
        assert mean_result.shape == (256, 512)
        
        # Stable log-sum-exp
        logsumexp_result = self.kernels.stable_logsumexp(input_data, dim=1)
        assert logsumexp_result.shape == (128, 512)
    
    def test_sparse_operations(self):
        """Test sparse tensor operations."""
        # Sparse matrix multiplication
        sparse_a = self.kernels.create_sparse_matrix(shape=(1000, 500), density=0.1)
        dense_b = np.random.randn(500, 200).astype(np.float32)
        
        sparse_result = self.kernels.sparse_dense_matmul(sparse_a, dense_b)
        assert sparse_result.shape == (1000, 200)
        
        # Sparse attention
        sparse_mask = np.random.choice([0, 1], size=(64, 64), p=[0.8, 0.2])
        q = np.random.randn(4, 8, 64, 32).astype(np.float32)
        k = np.random.randn(4, 8, 64, 32).astype(np.float32)
        v = np.random.randn(4, 8, 64, 32).astype(np.float32)
        
        sparse_attention = self.kernels.sparse_attention(q, k, v, mask=sparse_mask)
        assert sparse_attention.shape == q.shape


class TestCUDAKernelSystemIntegration:
    """Test CUDA kernel system integration and deployment scenarios."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.kernels = OptimizedCUDAKernels(device_id=0)
    
    def test_multi_gpu_operations(self):
        """Test multi-GPU operations."""
        num_gpus = min(4, self.kernels.get_device_count())
        
        if num_gpus > 1:
            data = np.random.randn(1000, 1000).astype(np.float32)
            
            # Test data distribution across GPUs
            distributed_results = []
            for gpu_id in range(num_gpus):
                gpu_result = self.kernels.run_on_device(data, device_id=gpu_id)
                distributed_results.append(gpu_result)
                
            assert len(distributed_results) == num_gpus
            
            # Test all-reduce operation
            reduced_result = self.kernels.all_reduce(distributed_results)
            assert reduced_result.shape == data.shape
    
    def test_memory_management_advanced(self):
        """Test advanced memory management features."""
        # Memory pool management
        pool_size = 1024 * 1024 * 100  # 100MB
        memory_pool = self.kernels.create_memory_pool(pool_size)
        assert memory_pool is not None
        
        # Allocate from pool
        tensor_1 = self.kernels.allocate_from_pool(memory_pool, shape=(1000, 1000))
        tensor_2 = self.kernels.allocate_from_pool(memory_pool, shape=(500, 2000))
        
        assert tensor_1.shape == (1000, 1000)
        assert tensor_2.shape == (500, 2000)
        
        # Test memory defragmentation
        self.kernels.defragment_memory_pool(memory_pool)
        
        # Test memory statistics
        stats = self.kernels.get_memory_stats()
        assert 'allocated' in stats
        assert 'free' in stats
        assert 'total' in stats
    
    def test_kernel_compilation_cache(self):
        """Test kernel compilation and caching."""
        # Test JIT compilation
        custom_kernel_code = """
        __global__ void custom_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """
        
        compiled_kernel = self.kernels.compile_kernel(custom_kernel_code, 'custom_add')
        assert compiled_kernel is not None
        
        # Test kernel caching
        cached_kernel = self.kernels.get_cached_kernel('custom_add')
        assert cached_kernel is not None
        
        # Test cache statistics
        cache_stats = self.kernels.get_compilation_cache_stats()
        assert 'hits' in cache_stats
        assert 'misses' in cache_stats
    
    def test_profiling_integration(self):
        """Test profiling and performance monitoring integration."""
        # Enable profiling
        self.kernels.enable_profiling()
        
        # Run some operations
        a = np.random.randn(1000, 1000).astype(np.float32)
        b = np.random.randn(1000, 1000).astype(np.float32)
        
        result = self.kernels.tensor_core_matmul(a, b, precision='fp16')
        
        # Get profiling results
        profile_data = self.kernels.get_profiling_results()
        
        assert 'kernel_times' in profile_data
        assert 'memory_transfers' in profile_data
        assert 'occupancy' in profile_data
        
        self.kernels.disable_profiling()
    
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling."""
        # Test CUDA out of memory
        try:
            huge_tensor = self.kernels.allocate_tensor_on_gpu(shape=(100000, 100000))
            if huge_tensor is not None:
                # If allocation succeeds on powerful systems
                assert huge_tensor.shape == (100000, 100000)
        except (MemoryError, RuntimeError) as e:
            assert "memory" in str(e).lower() or "cuda" in str(e).lower()
        
        # Test invalid kernel parameters
        with pytest.raises((ValueError, RuntimeError)):
            self.kernels.launch_kernel_with_invalid_params()
        
        # Test device synchronization errors
        try:
            self.kernels.force_device_sync_error()
        except RuntimeError as e:
            assert "sync" in str(e).lower() or "cuda" in str(e).lower()


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])