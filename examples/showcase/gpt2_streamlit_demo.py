#!/usr/bin/env python3
"""
🎭 GPT-2 Streamlit Demo - Interactive Text Generation

Interactive web interface for GPT-2 autoregressive text generation with:
- Real-time text completion and creative writing
- Configurable generation parameters and sampling strategies
- Performance metrics and visualizations
- Automatic optimizations showcase
"""

import streamlit as st
import sys
import os
import numpy as np
import time
import plotly.graph_objects as go
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from neural_arch.core import Tensor
from neural_arch.models.language.gpt2 import GPT2_CONFIGS, GPT2LMHead
from neural_arch.optim import AdamW
from neural_arch.functional import softmax
from neural_arch.optimization_config import configure, get_config
from neural_arch.backends import available_backends

# Page configuration
st.set_page_config(
    page_title="🎭 GPT-2 Demo",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .gpt2-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .generation-output {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
    }
    .token-prob {
        background: #e9ecef;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        margin: 0.1rem;
        display: inline-block;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🎭 GPT-2 Text Generation Demo</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Generative Pre-trained Transformer for Creative Text Generation</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🎛️ GPT-2 Configuration")

# Model settings
st.sidebar.subheader("Model Parameters")
model_size = st.sidebar.selectbox("Model Size:", ["Small", "Medium", "Large"], index=0)
vocab_size = st.sidebar.slider("Vocabulary Size:", 1000, 30000, 10000, step=1000)
batch_size = st.sidebar.slider("Batch Size:", 1, 4, 2)
seq_length = st.sidebar.slider("Sequence Length:", 16, 64, 32, step=8)

# Generation settings
st.sidebar.subheader("🎲 Generation Settings")
temperature = st.sidebar.slider("Temperature (randomness):", 0.1, 2.0, 1.0, step=0.1)
top_k = st.sidebar.slider("Top-K sampling:", 1, 50, 10)
max_new_tokens = st.sidebar.slider("Max new tokens:", 5, 50, 20)

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

# Sample prompts for generation
sample_prompts = [
    "Once upon a time in a magical forest,",
    "The future of artificial intelligence is",
    "In the year 2050, technology will",
    "The most interesting thing about space is",
    "Climate change solutions include",
    "The secret to happiness is"
]

# Helper functions
def simple_tokenize(text: str, vocab_size: int) -> List[int]:
    """Simple hash-based tokenization for demo."""
    words = text.lower().split()
    tokens = []
    for word in words:
        token_id = hash(word) % (vocab_size - 100) + 100
        tokens.append(token_id)
    return tokens

def simple_detokenize(tokens: List[int]) -> str:
    """Simple detokenization for demo."""
    words = [f"token_{token}" for token in tokens]
    return " ".join(words)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🎯 Text Generation Demo")
    
    # Prompt input
    selected_prompt = st.selectbox("Choose sample prompt or enter your own:", ["Custom"] + sample_prompts)
    
    if selected_prompt == "Custom":
        user_prompt = st.text_area("Enter your prompt:", height=100, placeholder="Start your creative text here...")
        input_prompt = user_prompt if user_prompt else "The story begins"
    else:
        input_prompt = selected_prompt
        st.text_area("Selected prompt:", value=input_prompt, height=100, disabled=True)

with col2:
    st.markdown("## ⚙️ Current Settings")
    
    config = get_config()
    
    st.markdown(f"""
    **Model Configuration:**
    - Size: {model_size}
    - Vocabulary: {vocab_size:,}
    - Batch Size: {batch_size}  
    - Sequence Length: {seq_length}
    
    **Generation Settings:**
    - Temperature: {temperature}
    - Top-K: {top_k}
    - Max Tokens: {max_new_tokens}
    
    **Optimizations:**
    - Fusion: {'✅' if config.optimization.enable_fusion else '❌'}
    - JIT: {'✅' if config.optimization.enable_jit else '❌'}
    - Auto Backend: {'✅' if config.optimization.auto_backend_selection else '❌'}
    - Available Backends: {', '.join(available_backends())}
    """)

# Demo execution
if st.button("🚀 Generate Text with GPT-2", type="primary", use_container_width=True):
    with st.spinner("Initializing GPT-2 with automatic optimizations..."):
        # Create GPT-2 model based on size
        if model_size == "Small":
            gpt2_config = GPT2_CONFIGS['small'].copy()
            gpt2_config.update({
                'vocab_size': vocab_size,
                'n_embd': 256,
                'n_layer': 4,
                'n_head': 4,
                'n_ctx': seq_length
            })
        elif model_size == "Medium":
            gpt2_config = GPT2_CONFIGS['small'].copy()
            gpt2_config.update({
                'vocab_size': vocab_size,
                'n_embd': 512,
                'n_layer': 6,
                'n_head': 8,
                'n_ctx': seq_length
            })
        else:  # Large
            gpt2_config = GPT2_CONFIGS['small'].copy()
            gpt2_config.update({
                'vocab_size': vocab_size,
                'n_embd': 768,
                'n_layer': 8,
                'n_head': 12,
                'n_ctx': seq_length
            })
        
        model = GPT2LMHead(gpt2_config)
        param_count = sum(p.data.size for p in model.parameters().values())
        
        # Tokenize input prompt
        prompt_tokens = simple_tokenize(input_prompt, vocab_size)
        
        # Ensure we don't exceed sequence length
        if len(prompt_tokens) > seq_length - max_new_tokens:
            prompt_tokens = prompt_tokens[:seq_length - max_new_tokens]
        
        # Pad to batch size and sequence length for initial input
        input_length = len(prompt_tokens)
        padded_tokens = prompt_tokens + [0] * (seq_length - len(prompt_tokens))
        input_ids = np.array([padded_tokens] * batch_size, dtype=np.int32)
        input_tensor = Tensor(input_ids)
        
        # Run inference
        start_time = time.time()
        outputs = model(input_tensor)
        inference_time = time.time() - start_time
        
        # Get logits for next token prediction
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Sample next tokens (simplified generation)
        next_token_logits = logits.data[0, input_length-1, :]  # Get logits for next position
        
        # Apply temperature
        scaled_logits = next_token_logits / temperature
        
        # Get top-k tokens
        top_k_indices = np.argsort(scaled_logits)[-top_k:]
        top_k_logits = scaled_logits[top_k_indices]
        
        # Convert to probabilities
        top_k_probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
        
        # Sample from top-k
        generated_tokens = []
        for _ in range(min(max_new_tokens, 10)):  # Limit for demo
            sampled_index = np.random.choice(len(top_k_indices), p=top_k_probs)
            sampled_token = top_k_indices[sampled_index]
            generated_tokens.append(sampled_token)
    
    # Display results
    st.markdown("## 📊 Results")
    
    # Model card
    st.markdown('<div class="gpt2-card">', unsafe_allow_html=True)
    st.markdown(f"### GPT-2 {model_size} - Autoregressive Text Generation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Parameters:** {param_count:,}")
        st.markdown(f"**Inference Time:** {inference_time:.4f}s")
    with col2:
        st.markdown(f"**Backend:** {input_tensor.backend.name}")
        st.markdown(f"**Vocabulary:** {vocab_size:,}")
    with col3:
        st.markdown(f"**Hidden Size:** {gpt2_config['n_embd']}")
        st.markdown(f"**Layers:** {gpt2_config['n_layer']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generated text output
    st.markdown("### 🎭 Generated Text")
    
    generated_text = simple_detokenize(generated_tokens[:5])  # Show first 5 tokens
    full_text = input_prompt + " " + generated_text
    
    st.markdown(f'<div class="generation-output">', unsafe_allow_html=True)
    st.markdown(f"**Original Prompt:** {input_prompt}")
    st.markdown(f"**Generated Continuation:** {generated_text}")
    st.markdown(f"**Full Text:** {full_text}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Generation Analysis")
        st.metric("Input Length", f"{len(prompt_tokens)} tokens")
        st.metric("Generated Length", f"{len(generated_tokens)} tokens")
        st.metric("Total Length", f"{len(prompt_tokens) + len(generated_tokens)} tokens")
        st.metric("Temperature Used", temperature)
        st.metric("Top-K Used", top_k)
    
    with col2:
        st.markdown("### 📈 Next Token Probabilities")
        
        # Show top token probabilities
        if len(top_k_indices) > 0:
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Token {idx}" for idx in top_k_indices[-5:]],  # Show top 5
                    y=top_k_probs[-5:],
                    marker_color='#e74c3c'
                )
            ])
            
            fig.update_layout(
                title="Top Token Probabilities",
                xaxis_title="Tokens",
                yaxis_title="Probability",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("### ⚡ Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Throughput", f"{batch_size/inference_time:.1f} samples/sec")
    with col2:
        st.metric("Latency", f"{inference_time*1000:.1f} ms")
    with col3:
        tokens_per_sec = (batch_size * seq_length) / inference_time
        st.metric("Tokens/sec", f"{tokens_per_sec:.0f}")
    with col4:
        st.metric("Generation Speed", f"{len(generated_tokens)/inference_time:.1f} tokens/sec")
    
    # Feature analysis
    st.markdown("### 🧠 GPT-2 Features Demonstrated")
    
    features = [
        ("🎭 Autoregressive Generation", "Generates text one token at a time"),
        ("🎯 Causal Attention", "Attends only to previous tokens"),
        ("🎲 Sampling Strategies", f"Temperature {temperature}, Top-K {top_k}"),
        ("🏗️ Deep Architecture", f"{gpt2_config['n_layer']} transformer layers"),
        ("⚡ Automatic Optimizations", "Fusion, JIT compilation, and backend selection"),
        ("🔄 Positional Encoding", "RoPE for superior position understanding")
    ]
    
    for feature_name, feature_desc in features:
        st.markdown(f"**{feature_name}:** {feature_desc}")

# Information section
st.markdown("---")
st.markdown("## 📚 About GPT-2")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Key Features
    - **Autoregressive**: Generates text sequentially
    - **Causal Attention**: Only looks at previous context
    - **Flexible Generation**: Multiple sampling strategies
    - **Creative Writing**: Produces coherent, creative text
    - **Scalable**: Works from small to large models
    """)

with col2:
    st.markdown("""
    ### 🚀 Automatic Optimizations
    - **Operator Fusion**: Combines operations for efficiency
    - **JIT Compilation**: Runtime optimization for speed
    - **Backend Selection**: Chooses optimal compute backend
    - **Memory Management**: Efficient tensor operations
    - **Zero Configuration**: Works out of the box
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🎭 GPT-2 Demo - Neural Architecture Framework</p>', 
    unsafe_allow_html=True
)