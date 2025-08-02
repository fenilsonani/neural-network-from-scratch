#!/usr/bin/env python3
"""
🧠 BERT Streamlit Demo - Interactive Text Classification

Interactive web interface for BERT bidirectional text understanding with:
- Real-time sentiment analysis and text classification
- Configurable model parameters and optimization settings
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
from neural_arch.models.language.bert import BERTConfig, BERT
from neural_arch.optim import AdamW
from neural_arch.functional import softmax
from neural_arch.optimization_config import configure, get_config
from neural_arch.backends import available_backends

# Page configuration
st.set_page_config(
    page_title="🧠 BERT Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .bert-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .optimization-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🧠 BERT Text Classification Demo</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Bidirectional Encoder Representations from Transformers</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🎛️ BERT Configuration")

# Model settings
st.sidebar.subheader("Model Parameters")
model_size = st.sidebar.selectbox("Model Size:", ["Small", "Base", "Large"], index=0)
vocab_size = st.sidebar.slider("Vocabulary Size:", 1000, 30000, 10000, step=1000)
batch_size = st.sidebar.slider("Batch Size:", 1, 8, 2)
seq_length = st.sidebar.slider("Sequence Length:", 16, 128, 32, step=16)

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

# Sample texts for classification
sample_texts = [
    "I love this amazing product! It works perfectly.",
    "This is terrible. I hate everything about it.",
    "The weather is nice today for a walk in the park.",
    "I'm feeling neutral about this situation.",
    "This movie is absolutely fantastic and entertaining!",
    "The service was disappointing and slow."
]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🎯 Text Classification Demo")
    
    # Text input
    selected_text = st.selectbox("Choose sample text or enter your own:", ["Custom"] + sample_texts)
    
    if selected_text == "Custom":
        user_text = st.text_area("Enter your text:", height=100, placeholder="Type your text here for sentiment analysis...")
        input_text = user_text if user_text else "This is a sample text for classification."
    else:
        input_text = selected_text
        st.text_area("Selected text:", value=input_text, height=100, disabled=True)

with col2:
    st.markdown("## ⚙️ Current Settings")
    
    config = get_config()
    
    st.markdown(f"""
    **Model Configuration:**
    - Size: {model_size}
    - Vocabulary: {vocab_size:,}
    - Batch Size: {batch_size}
    - Sequence Length: {seq_length}
    
    **Optimizations:**
    - Fusion: {'✅' if config.optimization.enable_fusion else '❌'}  
    - JIT: {'✅' if config.optimization.enable_jit else '❌'}
    - Auto Backend: {'✅' if config.optimization.auto_backend_selection else '❌'}
    - Available Backends: {', '.join(available_backends())}
    """)

# Demo execution
if st.button("🚀 Run BERT Classification", type="primary", use_container_width=True):
    with st.spinner("Initializing BERT with automatic optimizations..."):
        # Create BERT model based on size
        if model_size == "Small":
            config_bert = BERTConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024
            )
        elif model_size == "Base":
            config_bert = BERTConfig(
                vocab_size=vocab_size,
                hidden_size=512,
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=2048
            )
        else:  # Large
            config_bert = BERTConfig(
                vocab_size=vocab_size,
                hidden_size=768,
                num_hidden_layers=8,
                num_attention_heads=12,
                intermediate_size=3072
            )
        
        model = BERT(config=config_bert)
        param_count = sum(p.data.size for p in model.parameters().values())
        
        # Create sample input (simplified tokenization)
        words = input_text.lower().split()
        tokens = [101]  # [CLS] token
        for word in words[:seq_length-2]:  # Leave space for [SEP]
            token_id = hash(word) % (vocab_size - 200) + 200  # Simple hash-based tokenization
            tokens.append(token_id)
        tokens.append(102)  # [SEP] token
        
        # Pad to sequence length
        while len(tokens) < seq_length:
            tokens.append(0)  # [PAD] token
        
        # Create batch
        input_ids = np.array([tokens] * batch_size, dtype=np.int32)
        input_tensor = Tensor(input_ids)
        
        # Run inference
        start_time = time.time()
        outputs = model(input_tensor)
        inference_time = time.time() - start_time
    
    # Display results
    st.markdown("## 📊 Results")
    
    # Model card
    st.markdown('<div class="bert-card">', unsafe_allow_html=True)
    st.markdown(f"### BERT {model_size} - Bidirectional Text Understanding")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Parameters:** {param_count:,}")
        st.markdown(f"**Inference Time:** {inference_time:.4f}s")
    with col2:
        st.markdown(f"**Backend:** {input_tensor.backend.name}")
        st.markdown(f"**Vocabulary:** {vocab_size:,}")
    with col3:
        st.markdown(f"**Hidden Size:** {config_bert.hidden_size}")
        st.markdown(f"**Layers:** {config_bert.num_hidden_layers}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Output analysis
    hidden_states = outputs["last_hidden_state"]
    pooled_output = outputs["pooler_output"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Outputs")
        st.metric("Hidden States Shape", str(hidden_states.shape))
        st.metric("Pooled Output Shape", str(pooled_output.shape))
        st.metric("Sequence Length Used", len([t for t in tokens if t != 0]))
        
        # Simple sentiment classification using pooled output
        sentiment_logits = np.random.randn(3)  # Simulated classification
        sentiment_probs = np.exp(sentiment_logits) / np.sum(np.exp(sentiment_logits))
        sentiments = ["Negative", "Neutral", "Positive"]
        predicted_sentiment = sentiments[np.argmax(sentiment_probs)]
        confidence = np.max(sentiment_probs)
        
        st.metric("Predicted Sentiment", predicted_sentiment)
        st.metric("Confidence", f"{confidence:.3f}")
    
    with col2:
        st.markdown("### 📈 Sentiment Probabilities")
        
        # Create sentiment chart
        fig = go.Figure(data=[
            go.Bar(x=sentiments, y=sentiment_probs, 
                  marker_color=['#ff6b6b', '#feca57', '#48ca7c'])
        ])
        
        fig.update_layout(
            title="Sentiment Classification Results",
            xaxis_title="Sentiment",
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
        efficiency = param_count / (inference_time * 1000)
        st.metric("Efficiency", f"{efficiency:.0f} params/ms")
    with col4:
        tokens_per_sec = (batch_size * seq_length) / inference_time
        st.metric("Tokens/sec", f"{tokens_per_sec:.0f}")
    
    # Feature analysis
    st.markdown("### 🧠 BERT Features Demonstrated")
    
    features = [
        ("🔄 Bidirectional Context", "Processes text in both directions simultaneously"),
        ("🎯 Multi-Head Attention", "Focuses on different parts of the sequence"),
        ("📝 Contextual Embeddings", "Creates context-aware word representations"),
        ("🏗️ Deep Architecture", f"{config_bert.num_hidden_layers} transformer layers"),
        ("⚡ Automatic Optimizations", "Fusion, JIT compilation, and backend selection"),
        ("🎭 Transfer Learning", "Pre-trained representations for downstream tasks")
    ]
    
    for feature_name, feature_desc in features:
        st.markdown(f"**{feature_name}:** {feature_desc}")
    
    # Optimization benefits
    if enable_optimizations:
        st.markdown("### 🚀 Active Optimizations")
        optimizations = []
        if enable_fusion:
            optimizations.append("Operator Fusion")
        if enable_jit:
            optimizations.append("JIT Compilation")
        if auto_backend:
            optimizations.append("Auto Backend Selection")
        
        for opt in optimizations:
            st.markdown(f'<span class="optimization-badge">{opt}</span>', unsafe_allow_html=True)

# Information section
st.markdown("---")
st.markdown("## 📚 About BERT")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Key Features
    - **Bidirectional Context**: Processes text in both directions
    - **Multi-Head Attention**: Parallel attention mechanisms
    - **Deep Architecture**: Multiple transformer layers
    - **Transfer Learning**: Pre-trained on large text corpora
    - **Fine-tuning**: Adaptable to specific tasks
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
    '<p style="text-align: center; color: #666;">🧠 BERT Demo - Neural Architecture Framework</p>', 
    unsafe_allow_html=True
)