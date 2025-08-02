"""Gradient checkpointing for memory-efficient training.

This module implements gradient checkpointing (also known as activation checkpointing),
a technique that trades computation for memory by recomputing intermediate activations
during the backward pass instead of storing them during the forward pass.

Key benefits:
- Reduces memory usage by 50-90% depending on model architecture
- Enables training of larger models on the same hardware
- Minimal impact on training speed (10-30% slowdown for significant memory savings)
"""

import logging
import weakref
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.device import Device
from ..core.tensor import GradientFunction, Tensor, no_grad

logger = logging.getLogger(__name__)


class CheckpointFunction:
    """Represents a checkpointed function in the computational graph."""

    def __init__(
        self,
        function: Callable,
        inputs: List[Tensor],
        outputs: List[Tensor],
        save_inputs: bool = True,
    ):
        """Initialize checkpointed function.

        Args:
            function: The function to checkpoint
            inputs: Input tensors to the function
            outputs: Output tensors from the function
            save_inputs: Whether to save input tensors for recomputation
        """
        self.function = function
        self.inputs = inputs if save_inputs else []
        self.outputs = outputs
        self.save_inputs = save_inputs

        # Store input data for recomputation (using weak references to avoid cycles)
        self.input_data = []
        if save_inputs:
            for inp in inputs:
                if isinstance(inp, Tensor):
                    # Store the actual data, not the tensor object
                    self.input_data.append(inp.data.copy())
                else:
                    self.input_data.append(inp)

    def recompute_forward(self) -> List[Tensor]:
        """Recompute the forward pass to get intermediate activations."""
        if not self.save_inputs:
            raise RuntimeError("Cannot recompute forward pass without saved inputs")

        # Recreate input tensors from saved data
        reconstructed_inputs = []
        for i, (original_input, saved_data) in enumerate(zip(self.inputs, self.input_data)):
            if isinstance(original_input, Tensor):
                # Create new tensor with same properties but requires_grad=True for recomputation
                new_tensor = Tensor(
                    saved_data,
                    requires_grad=True,
                    dtype=original_input.dtype,
                    device=original_input.device,
                    name=f"{original_input.name}_recomputed" if original_input.name else None,
                )
                reconstructed_inputs.append(new_tensor)
            else:
                reconstructed_inputs.append(saved_data)

        # Recompute forward pass
        logger.debug(f"Recomputing forward pass for checkpointed function")
        return self.function(*reconstructed_inputs)


class GradientCheckpointManager:
    """Manages gradient checkpointing state and configuration."""

    def __init__(self):
        self.enabled = False
        self.checkpoint_functions: List[CheckpointFunction] = []
        self.memory_savings = 0
        self.recompute_count = 0

    def enable(self):
        """Enable gradient checkpointing."""
        self.enabled = True
        logger.info("Gradient checkpointing enabled")

    def disable(self):
        """Disable gradient checkpointing."""
        self.enabled = False
        logger.info("Gradient checkpointing disabled")

    def clear(self):
        """Clear all checkpointed functions."""
        self.checkpoint_functions.clear()
        self.memory_savings = 0
        self.recompute_count = 0

    def add_checkpoint(self, checkpoint_fn: CheckpointFunction):
        """Add a checkpointed function."""
        self.checkpoint_functions.append(checkpoint_fn)

        # Estimate memory savings (rough approximation)
        memory_saved = 0
        for output in checkpoint_fn.outputs:
            if isinstance(output, Tensor):
                memory_saved += output.memory_usage()

        self.memory_savings += memory_saved
        logger.debug(f"Added checkpoint, estimated memory saved: {memory_saved / (1024**2):.1f} MB")

    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpointing statistics."""
        return {
            "enabled": self.enabled,
            "num_checkpoints": len(self.checkpoint_functions),
            "memory_savings_mb": self.memory_savings / (1024**2),
            "recompute_count": self.recompute_count,
        }


# Global checkpoint manager
_checkpoint_manager = GradientCheckpointManager()


def get_checkpoint_manager() -> GradientCheckpointManager:
    """Get the global checkpoint manager."""
    return _checkpoint_manager


@contextmanager
def checkpoint_scope():
    """Context manager for gradient checkpointing operations."""
    manager = get_checkpoint_manager()
    old_state = manager.enabled
    manager.enable()
    try:
        yield manager
    finally:
        manager.enabled = old_state


def checkpoint(function: Callable) -> Callable:
    """Decorator to enable gradient checkpointing for a function.

    This decorator wraps a function to use gradient checkpointing, trading
    computation for memory by recomputing activations during backward pass.

    Args:
        function: Function to checkpoint

    Returns:
        Checkpointed version of the function

    Example:
        @checkpoint
        def expensive_layer(x):
            x = linear1(x)
            x = gelu(x)
            x = linear2(x)
            return x
    """

    @wraps(function)
    def checkpointed_function(*args, **kwargs):
        manager = get_checkpoint_manager()

        if not manager.enabled:
            # Checkpointing disabled, run normally
            return function(*args, **kwargs)

        # Extract tensor inputs
        tensor_inputs = []
        for arg in args:
            if isinstance(arg, Tensor):
                tensor_inputs.append(arg)

        # Run forward pass normally
        outputs = function(*args, **kwargs)

        # Ensure outputs is a list
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        # Create checkpoint function
        checkpoint_fn = CheckpointFunction(function, tensor_inputs, outputs, save_inputs=True)
        manager.add_checkpoint(checkpoint_fn)

        # Set up gradient functions for outputs
        for i, output in enumerate(outputs):
            if isinstance(output, Tensor) and output.requires_grad:

                def make_backward_fn(checkpoint_fn, output_idx):
                    def backward_fn(grad_output: np.ndarray) -> None:
                        manager.recompute_count += 1

                        # Recompute forward pass to get intermediate activations
                        with no_grad():
                            recomputed_outputs = checkpoint_fn.recompute_forward()

                        # Get the specific output we need gradients for
                        if isinstance(recomputed_outputs, (list, tuple)):
                            target_output = recomputed_outputs[output_idx]
                        else:
                            target_output = recomputed_outputs

                        # Propagate gradients through recomputed graph
                        if hasattr(target_output, "backward"):
                            target_output.backward(grad_output)

                    return backward_fn

                output._grad_fn = GradientFunction(
                    make_backward_fn(checkpoint_fn, i),
                    tensor_inputs,
                    f"checkpoint_{function.__name__}",
                )

        # Return original format
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    return checkpointed_function


class SequentialCheckpoint:
    """Sequential layer with automatic gradient checkpointing.

    This class wraps a sequence of layers and applies gradient checkpointing
    to reduce memory usage during training.
    """

    def __init__(self, *layers, checkpoint_segments: Optional[int] = None):
        """Initialize sequential checkpoint.

        Args:
            *layers: Sequence of layers/functions
            checkpoint_segments: Number of segments to checkpoint (default: auto)
        """
        self.layers = list(layers)
        self.checkpoint_segments = checkpoint_segments or max(1, len(layers) // 2)

        # Create checkpointed segments
        self.segments = self._create_segments()

    def _create_segments(self) -> List[Callable]:
        """Create checkpointed segments from layers."""
        if len(self.layers) <= 1:
            return self.layers

        segment_size = max(1, len(self.layers) // self.checkpoint_segments)
        segments = []

        for i in range(0, len(self.layers), segment_size):
            segment_layers = self.layers[i : i + segment_size]

            @checkpoint
            def create_segment(layers):
                def segment_fn(x):
                    for layer in layers:
                        x = layer(x)
                    return x

                return segment_fn

            segments.append(create_segment(segment_layers))

        return segments

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass through checkpointed segments."""
        for segment in self.segments:
            x = segment(x)
        return x

    def parameters(self) -> List[Tensor]:
        """Get all parameters from layers."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params


def memory_efficient_attention(
    query: Tensor, key: Tensor, value: Tensor, scale: Optional[float] = None, chunk_size: int = 1024
) -> Tensor:
    """Memory-efficient attention using gradient checkpointing and chunking.

    This function implements attention with gradient checkpointing and chunking
    to reduce memory usage for long sequences.

    Args:
        query: Query tensor [batch, heads, seq_len, head_dim]
        key: Key tensor [batch, heads, seq_len, head_dim]
        value: Value tensor [batch, heads, seq_len, head_dim]
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        chunk_size: Size of chunks for memory efficiency

    Returns:
        Attention output tensor
    """
    batch_size, num_heads, seq_len, head_dim = query.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    # For short sequences, use regular attention
    if seq_len <= chunk_size:

        @checkpoint
        def attention_chunk(q, k, v):
            # Compute attention scores
            scores = q @ k.T * scale

            # Apply softmax
            from ..functional import softmax

            attn_weights = softmax(scores, axis=-1)

            # Apply attention to values
            return attn_weights @ v

        return attention_chunk(query, key, value)

    # For long sequences, use chunked attention with checkpointing
    output_chunks = []

    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)

        @checkpoint
        def attention_chunk_fn(q_chunk, k_full, v_full):
            # Compute scores for this chunk
            scores = q_chunk @ k_full.T * scale

            # Apply softmax
            from ..functional import softmax

            attn_weights = softmax(scores, axis=-1)

            # Apply attention to values
            return attn_weights @ v_full

        q_chunk = query[:, :, i:end_i, :]
        chunk_output = attention_chunk_fn(q_chunk, key, value)
        output_chunks.append(chunk_output)

    # Concatenate chunks
    from ..functional import concatenate

    return concatenate(output_chunks, axis=2)


class CheckpointedTransformerLayer:
    """Transformer layer with gradient checkpointing for memory efficiency."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize checkpointed transformer layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # Initialize layers (this would use actual layer implementations)
        from ..nn.attention import MultiHeadAttention
        from ..nn.linear import Linear
        from ..nn.normalization import LayerNorm

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = SequentialCheckpoint(
            Linear(d_model, d_ff), lambda x: gelu(x), Linear(d_ff, d_model)  # GELU activation
        )
        self.norm2 = LayerNorm(d_model)

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass with checkpointing."""

        @checkpoint
        def attention_block(x, mask):
            # Self-attention with residual connection
            attn_output = self.self_attention(x, x, x, mask)
            return self.norm1(x + attn_output)

        @checkpoint
        def feedforward_block(x):
            # Feed-forward with residual connection
            ff_output = self.feed_forward(x)
            return self.norm2(x + ff_output)

        # Apply checkpointed blocks
        x = attention_block(x, mask)
        x = feedforward_block(x)

        return x


def estimate_memory_savings(
    model_layers: int, batch_size: int, seq_len: int, d_model: int
) -> Dict[str, float]:
    """Estimate memory savings from gradient checkpointing.

    Args:
        model_layers: Number of transformer layers
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension

    Returns:
        Dictionary with memory estimates
    """
    # Rough estimates based on typical transformer memory usage

    # Without checkpointing: store all intermediate activations
    activation_memory_per_layer = batch_size * seq_len * d_model * 4  # 4 bytes per float32
    total_activation_memory = activation_memory_per_layer * model_layers

    # With checkpointing: only store inputs and recompute
    checkpointed_memory = activation_memory_per_layer * 2  # Input + output only

    memory_saved = total_activation_memory - checkpointed_memory
    savings_percentage = (memory_saved / total_activation_memory) * 100

    return {
        "original_memory_mb": total_activation_memory / (1024**2),
        "checkpointed_memory_mb": checkpointed_memory / (1024**2),
        "memory_saved_mb": memory_saved / (1024**2),
        "savings_percentage": savings_percentage,
        "compute_overhead_estimate": 25.0,  # Typical 15-35% overhead
    }


# Utility functions for common checkpointing patterns
def gelu(x: Tensor) -> Tensor:
    """GELU activation (placeholder - would use actual implementation)."""
    from ..functional import gelu as gelu_impl

    return gelu_impl(x)


def checkpoint_sequential(*layers) -> SequentialCheckpoint:
    """Create a sequential model with gradient checkpointing.

    Args:
        *layers: Sequence of layers

    Returns:
        SequentialCheckpoint instance
    """
    return SequentialCheckpoint(*layers)


# Context managers for fine-grained control
@contextmanager
def no_checkpoint():
    """Temporarily disable gradient checkpointing."""
    manager = get_checkpoint_manager()
    old_state = manager.enabled
    manager.disable()
    try:
        yield
    finally:
        manager.enabled = old_state


@contextmanager
def force_checkpoint():
    """Force enable gradient checkpointing."""
    manager = get_checkpoint_manager()
    old_state = manager.enabled
    manager.enable()
    try:
        yield
    finally:
        manager.enabled = old_state
