#!/usr/bin/env python3
"""
🧬 Modern Transformer Streamlit Demo - Advanced Architecture Features

Interactive web interface for Modern Transformer with:
- RoPE (Rotary Position Embedding) for superior positional encoding
- Pre-Norm architecture for stable training gradients
- SwiGLU and advanced activation functions
- RMSNorm for improved normalization
- Performance comparisons and feature analysis
"""

import streamlit as st
import sys
import os
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from neural_arch.core import Tensor
from neural_arch.models.language.modern_transformer import PreNormTransformer, PreNormTransformerConfig
from neural_arch.optim import AdamW
from neural_arch.functional import softmax
from neural_arch.optimization_config import configure, get_config
from neural_arch.backends import available_backends

# Page configuration
st.set_page_config(
    page_title="🧬 Modern Transformer Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #16a085;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .modern-card {
        background: linear-gradient(135deg, #16a085 0%, #1abc9c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .feature-highlight {
        background: #e8f8f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #16a085;
        margin: 0.5rem 0;
    }
    .comparison-table {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .rope-visualization {
        background: linear-gradient(45deg, #e8f8f5, #d5f4e6);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #16a085;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🧬 Modern Transformer Demo</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Pre-Norm Architecture with RoPE, SwiGLU, and Advanced Features</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🎛️ Modern Transformer Config")

# Architecture settings
st.sidebar.subheader("Architecture Parameters")
d_model = st.sidebar.selectbox("Model Dimension:", [256, 384, 512, 768], index=1)
num_layers = st.sidebar.slider("Number of Layers:", 2, 12, 6)
num_heads = st.sidebar.slider("Attention Heads:", 4, 16, 8)
d_ff = st.sidebar.selectbox("Feed-Forward Dimension:", [1024, 1536, 2048, 3072], index=1)

# Advanced features
st.sidebar.subheader("🚀 Advanced Features")
activation = st.sidebar.selectbox("Activation Function:", ["gelu", "swiglu"], index=1)
normalization = st.sidebar.selectbox("Normalization:", ["layernorm", "rmsnorm"], index=1)
use_rope = st.sidebar.checkbox("RoPE (Rotary Position Embedding)", value=True)
rope_base = st.sidebar.slider("RoPE Base Frequency:", 1000, 20000, 10000, step=1000)
tie_embeddings = st.sidebar.checkbox("Tie Input/Output Embeddings", value=True)
scale_embeddings = st.sidebar.checkbox("Scale Embeddings by √d_model", value=True)

# Model settings
st.sidebar.subheader("Model Settings")
vocab_size = st.sidebar.slider("Vocabulary Size:", 1000, 30000, 10000, step=1000)
max_seq_len = st.sidebar.slider("Max Sequence Length:", 128, 2048, 512, step=128)
batch_size = st.sidebar.slider("Batch Size:", 1, 8, 2)
seq_length = st.sidebar.slider("Input Sequence Length:", 16, 128, 32, step=16)

# Optimization settings
st.sidebar.subheader("⚙️ Optimizations")
enable_optimizations = st.sidebar.checkbox("Enable Automatic Optimizations", value=True)
enable_fusion = st.sidebar.checkbox("Operator Fusion", value=True)
enable_jit = st.sidebar.checkbox("JIT Compilation", value=True)
auto_backend = st.sidebar.checkbox("Auto Backend Selection", value=True)

if enable_optimizations:
    configure(
        enable_fusion=enable_fusion,
        enable_jit=enable_jit,
        auto_backend_selection=auto_backend,
        enable_mixed_precision=False
    )

# Calculate derived values
head_dim = d_model // num_heads

# Helper functions
def create_rope_visualization() -> go.Figure:
    """Create a visualization of RoPE frequency patterns."""
    dim = min(64, head_dim)  # Limit for visualization
    seq_len = 32
    
    # Generate RoPE frequencies
    freqs = 1.0 / (rope_base ** (np.arange(0, dim, 2) / dim))
    
    # Create position encodings
    positions = np.arange(seq_len)
    angles = np.outer(positions, freqs)
    
    # Create visualization data
    fig = go.Figure()
    
    for i in range(min(8, len(freqs))):  # Show first 8 frequencies
        fig.add_trace(go.Scatter(
            x=positions,
            y=np.sin(angles[:, i]),
            name=f'Freq {i+1} (base^{2*i}/{dim})',
            mode='lines'
        ))
    
    fig.update_layout(
        title="RoPE Frequency Patterns",
        xaxis_title="Position",
        yaxis_title="Sin(Position × Frequency)",
        height=400,
        template="plotly_white"
    )
    
    return fig

def generate_sample_text(vocab_size: int, seq_length: int) -> List[str]:
    """Generate sample text patterns for analysis."""
    patterns = [
        "Arithmetic sequence with mathematical patterns and numerical relationships",
        "Natural language processing with transformers and attention mechanisms for understanding",  
        "Artificial intelligence research focuses on machine learning and neural networks",
        "Modern transformer architectures utilize positional encodings and multi-head attention"
    ]
    
    # Simple tokenization for demo
    tokens = []
    for pattern in patterns[:min(4, seq_length//8)]:
        words = pattern.split()[:seq_length//4]
        for word in words:
            token_id = hash(word) % (vocab_size - 100) + 100
            tokens.append(token_id)
    
    # Pad or truncate to desired length
    if len(tokens) < seq_length:
        tokens.extend([1] * (seq_length - len(tokens)))  # Padding
    else:
        tokens = tokens[:seq_length]
    
    return tokens

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🚀 Advanced Architecture Analysis")
    
    # Feature showcase
    st.markdown("### Modern Features Overview")
    
    features_enabled = []
    if use_rope:
        features_enabled.append("🌀 RoPE Positional Encoding")
    if activation == "swiglu":
        features_enabled.append("⚡ SwiGLU Activation")
    if normalization == "rmsnorm":
        features_enabled.append("📏 RMSNorm Normalization")
    if tie_embeddings:
        features_enabled.append("🔗 Tied Embeddings")
    
    features_enabled.append("🏗️ Pre-Norm Architecture")
    
    for feature in features_enabled:
        st.markdown(f"✅ {feature}")

with col2:
    st.markdown("## ⚙️ Configuration")
    
    config = get_config()
    
    st.markdown(f"""
    **Architecture:**
    - Model Dim: {d_model}
    - Layers: {num_layers}
    - Heads: {num_heads} (dim {head_dim})
    - FFN Dim: {d_ff}
    - Vocab Size: {vocab_size:,}
    - Max Length: {max_seq_len}
    
    **Advanced Features:**
    - Activation: {activation.upper()}
    - Normalization: {normalization.upper()}
    - RoPE: {'✅' if use_rope else '❌'}
    - Tied Embeddings: {'✅' if tie_embeddings else '❌'}
    
    **Optimizations:**
    - Fusion: {'✅' if config.optimization.enable_fusion else '❌'}
    - JIT: {'✅' if config.optimization.enable_jit else '❌'}
    - Auto Backend: {'✅' if config.optimization.auto_backend_selection else '❌'}
    """)

# RoPE Visualization
if use_rope:
    st.markdown("### 🌀 RoPE Visualization")
    st.markdown('<div class="rope-visualization">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **RoPE Configuration:**
        - Base Frequency: {rope_base:,}
        - Head Dimension: {head_dim}
        - Frequency Count: {head_dim // 2}
        - Max Sequence: {max_seq_len}
        """)
    
    with col2:
        rope_fig = create_rope_visualization()
        st.plotly_chart(rope_fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Demo execution
if st.button("🚀 Run Modern Transformer", type="primary", use_container_width=True):
    with st.spinner("Initializing Modern Transformer with advanced features..."):
        # Create configuration
        transformer_config = PreNormTransformerConfig(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            activation=activation,
            normalization=normalization,
            use_rope=use_rope,
            rope_base=rope_base,
            tie_embeddings=tie_embeddings,
            scale_embeddings=scale_embeddings
        )
        
        # Create model
        model = PreNormTransformer(transformer_config)
        param_count = sum(p.data.size for p in model.parameters().values())
        
        # Generate sample sequences
        sequences = []
        for i in range(batch_size):
            tokens = generate_sample_text(vocab_size, seq_length)
            sequences.append(tokens)
        
        input_ids = np.array(sequences, dtype=np.int32)
        input_tensor = Tensor(input_ids)
        
        # Run inference
        start_time = time.time()
        outputs = model(input_tensor, output_hidden_states=True)
        inference_time = time.time() - start_time
        
        # Extract outputs
        logits = outputs['logits']
        last_hidden_state = outputs['last_hidden_state']
        if 'hidden_states' in outputs:
            all_hidden_states = outputs['hidden_states']
        else:
            all_hidden_states = None
    
    # Display results
    st.markdown("## 📊 Results")
    
    # Model card
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### Modern Transformer - Advanced Architecture")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Parameters:** {param_count:,}")
        st.markdown(f"**Inference Time:** {inference_time:.4f}s")
    with col2:
        st.markdown(f"**Backend:** {input_tensor.backend.name}")
        st.markdown(f"**Architecture:** Pre-Norm")
    with col3:
        st.markdown(f"**Activation:** {activation.upper()}")
        st.markdown(f"**Normalization:** {normalization.upper()}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Output analysis
    st.markdown("### 🎯 Model Outputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Output Shapes and Statistics")
        st.metric("Logits Shape", str(logits.shape))
        st.metric("Hidden State Shape", str(last_hidden_state.shape))
        st.metric("Sequence Length", seq_length)
        st.metric("Vocabulary Coverage", f"{vocab_size:,} tokens")
        
        # Token predictions
        probabilities = softmax(logits, axis=-1)
        predicted_tokens = np.argmax(probabilities.data, axis=-1)
        
        st.markdown("#### Next Token Predictions")
        for i in range(min(batch_size, 2)):
            next_token = predicted_tokens[i, -1]  # Last position prediction
            confidence = np.max(probabilities.data[i, -1, :])
            st.markdown(f"**Sequence {i+1}:** Token {next_token} (confidence: {confidence:.3f})")
    
    with col2:
        st.markdown("#### Performance Metrics")
        
        # Performance chart
        metrics = {
            'Throughput': batch_size / inference_time,
            'Latency (ms)': inference_time * 1000 / batch_size,
            'Tokens/sec': (batch_size * seq_length) / inference_time,
            'Params/ms': param_count / (inference_time * 1000)
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color='#16a085'
            )
        ])
        
        fig.update_layout(
            title="Performance Metrics",
            yaxis_title="Value",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.markdown("### 🧬 Advanced Features Analysis")
    
    # Pre-Norm vs Post-Norm comparison
    st.markdown("#### 🏗️ Pre-Norm Architecture Benefits")
    st.markdown('<div class="feature-highlight">', unsafe_allow_html=True)
    st.markdown("""
    **Pre-Norm Design Pattern:** `LayerNorm → Attention → Residual → LayerNorm → FFN → Residual`
    
    **Advantages over Post-Norm:**
    - **Stable Training**: Better gradient flow through residual connections
    - **Easier Optimization**: Reduced need for learning rate warmup
    - **Deep Networks**: Enables training of very deep transformers
    - **Consistent Scaling**: More predictable behavior with model depth
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # RoPE analysis
    if use_rope:
        st.markdown("#### 🌀 RoPE (Rotary Position Embedding) Analysis")
        st.markdown('<div class="feature-highlight">', unsafe_allow_html=True)
        st.markdown(f"""
        **RoPE Configuration Active:**
        - Base Frequency: {rope_base:,} Hz
        - Frequency Bands: {head_dim // 2}
        - Rotation Applied: To query and key vectors in attention
        
        **RoPE Advantages:**
        - **Relative Positions**: Naturally encodes relative distances
        - **Extrapolation**: Better handling of longer sequences than seen in training
        - **Efficiency**: No additional parameters needed
        - **Mathematical Elegance**: Rotation in complex plane preserves distances
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Activation function analysis
    if activation == "swiglu":
        st.markdown("#### ⚡ SwiGLU Activation Analysis")
        st.markdown('<div class="feature-highlight">', unsafe_allow_html=True)
        st.markdown("""
        **SwiGLU Formula:** `SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)`
        
        **Benefits over GELU/ReLU:**
        - **Gating Mechanism**: Multiplicative gating for selective information flow
        - **Smooth Gradients**: Better gradient properties than ReLU
        - **Performance**: Consistently outperforms GELU in large models
        - **Parameter Efficiency**: Effective use of feed-forward capacity
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Normalization analysis
    if normalization == "rmsnorm":
        st.markdown("#### 📏 RMSNorm Analysis")
        st.markdown('<div class="feature-highlight">', unsafe_allow_html=True)
        st.markdown("""
        **RMSNorm Formula:** `RMSNorm(x) = x / RMS(x) * γ` where `RMS(x) = √(mean(x²))`
        
        **Advantages over LayerNorm:**
        - **Simpler Computation**: No mean subtraction, only RMS scaling
        - **Faster**: Reduced computational overhead
        - **Stable**: Similar training stability to LayerNorm
        - **Memory Efficient**: Less intermediate computation required
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Architecture comparison
st.markdown("---")
st.markdown("## 🔄 Modern vs Traditional Transformer")

st.markdown('<div class="comparison-table">', unsafe_allow_html=True)

comparison_data = {
    "Feature": [
        "Normalization Order",
        "Position Encoding", 
        "Activation Function",
        "Normalization Type",
        "Training Stability",
        "Gradient Flow",
        "Parameter Efficiency",
        "Sequence Extrapolation"
    ],
    "Traditional Transformer": [
        "Post-Norm (after attention/FFN)",
        "Sinusoidal/Learned Absolute",
        "ReLU/GELU",
        "LayerNorm",
        "Requires careful tuning",
        "Can be unstable in deep networks",
        "Standard",
        "Limited to training lengths"
    ],
    "Modern Transformer": [
        "Pre-Norm (before attention/FFN)",
        "RoPE (Rotary Relative)" if use_rope else "Sinusoidal/Learned",
        activation.upper(),
        normalization.upper(),
        "More stable, less sensitive",
        "Improved through pre-norm",
        "Tied embeddings" if tie_embeddings else "Standard",
        "Better extrapolation" if use_rope else "Standard"
    ]
}

import pandas as pd
df = pd.DataFrame(comparison_data)
st.dataframe(df, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Performance implications
st.markdown("### ⚡ Performance Implications")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Computational Benefits:**
    - Pre-Norm: Stable training, fewer iterations needed
    - RoPE: No extra parameters, O(d) complexity
    - SwiGLU: Better parameter utilization
    - RMSNorm: Faster normalization computation
    """)

with col2:
    st.markdown("""  
    **Training Benefits:**
    - Reduced sensitivity to hyperparameters
    - Better convergence properties
    - Improved gradient flow in deep networks
    - Enhanced performance on downstream tasks
    """)

# Information section
st.markdown("---")  
st.markdown("## 📚 About Modern Transformers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Key Innovations
    - **Pre-Norm Architecture**: Stable training for deep networks
    - **RoPE**: Superior positional encoding with relative positions  
    - **SwiGLU**: Advanced activation with gating mechanism
    - **RMSNorm**: Efficient normalization with fewer operations
    - **Architectural Improvements**: Evidence-based design choices
    """)

with col2:
    st.markdown("""
    ### 🚀 Automatic Optimizations
    - **Attention Fusion**: Optimized multi-head attention operations
    - **FFN Fusion**: Fused feed-forward network computations
    - **JIT Compilation**: Runtime optimization for modern patterns
    - **Backend Selection**: Optimal hardware utilization
    - **Zero Configuration**: Advanced features work out of the box
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🧬 Modern Transformer Demo - Neural Architecture Framework</p>', 
    unsafe_allow_html=True
)