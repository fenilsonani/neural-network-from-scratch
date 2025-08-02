#!/usr/bin/env python3
"""
🌟 CLIP Streamlit Demo - Multimodal Vision-Language Understanding

Interactive web interface for CLIP multimodal AI with:
- Real-time image-text similarity computation
- Cross-modal retrieval and zero-shot classification
- Contrastive learning visualization
- Joint embedding space analysis
- Performance metrics and optimization showcase
"""

import streamlit as st
import sys
import os
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from neural_arch.core import Tensor
from neural_arch.models.multimodal.clip import CLIP, CLIP_CONFIGS
from neural_arch.optim import AdamW
from neural_arch.functional import softmax
from neural_arch.optimization_config import configure, get_config
from neural_arch.backends import available_backends

# Page configuration
st.set_page_config(
    page_title="🌟 CLIP Demo",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e67e22;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .clip-card {
        background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .similarity-matrix {
        background: #fdf6e3;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e67e22;
        margin: 1rem 0;
    }
    .multimodal-info {
        background: #fef5e7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #e67e22;
        margin: 0.5rem 0;
    }
    .embedding-space {
        background: linear-gradient(45deg, #fdf6e3, #fef5e7);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #f39c12;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌟 CLIP Multimodal Demo</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Contrastive Language-Image Pre-training for Vision-Language Understanding</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🎛️ CLIP Configuration")

# Model settings
st.sidebar.subheader("Model Architecture")
model_size = st.sidebar.selectbox("Model Size:", ["Base", "Large"], index=0)
embed_dim = st.sidebar.selectbox("Embedding Dimension:", [256, 512, 768], index=0)
image_resolution = st.sidebar.selectbox("Image Resolution:", [64, 128, 224], index=0)
batch_size = st.sidebar.slider("Batch Size:", 1, 4, 2)

# Vision encoder settings
st.sidebar.subheader("🖼️ Vision Encoder")
vision_layers = st.sidebar.slider("Vision Layers:", 4, 12, 6)
vision_width = st.sidebar.selectbox("Vision Width:", [256, 384, 512, 768], index=2)

# Text encoder settings  
st.sidebar.subheader("📝 Text Encoder")
text_layers = st.sidebar.slider("Text Layers:", 4, 12, 6)
text_width = st.sidebar.selectbox("Text Width:", [256, 384, 512], index=1)
context_length = st.sidebar.slider("Context Length:", 32, 128, 77)
vocab_size = st.sidebar.slider("Vocabulary Size:", 5000, 30000, 10000, step=1000)

# Advanced settings
st.sidebar.subheader("🚀 Advanced Features")
temperature_init = st.sidebar.slider("Temperature Init:", 0.01, 0.2, 0.07, step=0.01)
learnable_temperature = st.sidebar.checkbox("Learnable Temperature", value=True)

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

# Helper functions
def create_synthetic_images(num_images: int, resolution: int) -> Tuple[np.ndarray, List[str]]:
    """Create synthetic images with different patterns."""
    images = []
    descriptions = []
    
    patterns = [
        ("gradient", "beautiful sky with gradient colors"),
        ("checkerboard", "geometric black and white pattern"),
        ("radial", "colorful flower with radial petals"),
        ("texture", "abstract textured surface")
    ]
    
    for i in range(num_images):
        pattern_type, desc = patterns[i % len(patterns)]
        img = np.zeros((3, resolution, resolution), dtype=np.float32)
        
        if pattern_type == "gradient":
            # Sky gradient
            for y in range(resolution):
                gradient = y / resolution
                img[2, y, :] = 0.8 - 0.4 * gradient  # Blue
                img[0, y, :] = 0.2 + 0.3 * gradient  # Red
                img[1, y, :] = 0.4 + 0.2 * gradient  # Green
        
        elif pattern_type == "checkerboard":
            # Checkerboard
            square_size = max(8, resolution // 8)
            for y in range(0, resolution, square_size):
                for x in range(0, resolution, square_size):
                    if ((y // square_size) + (x // square_size)) % 2:
                        img[:, y:y+square_size, x:x+square_size] = 0.9
                    else:
                        img[:, y:y+square_size, x:x+square_size] = 0.1
        
        elif pattern_type == "radial":
            # Radial flower
            center = resolution // 2
            for y in range(resolution):
                for x in range(resolution):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    angle = np.arctan2(y - center, x - center)
                    petals = np.sin(6 * angle) * np.exp(-dist / (resolution/4))
                    img[0, y, x] = 0.5 + 0.3 * petals  # Red
                    img[1, y, x] = 0.6 - dist / (resolution*2)  # Green
                    img[2, y, x] = 0.2  # Blue
        
        else:  # texture
            # Random texture
            img = np.random.uniform(0.3, 0.7, (3, resolution, resolution)).astype(np.float32)
            for _ in range(3):
                y, x = np.random.randint(resolution//4, 3*resolution//4, 2)
                size = resolution // 8
                intensity = np.random.uniform(0.7, 1.3)
                img[:, y:y+size, x:x+size] *= intensity
        
        # Add noise
        img += np.random.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)
        
        images.append(img)
        descriptions.append(desc)
    
    return np.stack(images, axis=0), descriptions

def create_text_tokens(descriptions: List[str], vocab_size: int, context_length: int) -> np.ndarray:
    """Create text tokens from descriptions."""
    tokens = []
    
    for desc in descriptions:
        words = desc.split()
        token_sequence = [1]  # [CLS] token
        
        for word in words[:context_length-2]:
            token_id = hash(word) % (vocab_size - 100) + 100
            token_sequence.append(token_id)
        
        token_sequence.append(2)  # [SEP] token
        
        # Pad to context length
        while len(token_sequence) < context_length:
            token_sequence.append(0)  # [PAD] token
            
        tokens.append(token_sequence[:context_length])
    
    return np.array(tokens, dtype=np.int32)

def visualize_similarity_matrix(similarity_matrix: np.ndarray, descriptions: List[str]) -> go.Figure:
    """Create similarity matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f"Text {i+1}" for i in range(len(descriptions))],
        y=[f"Image {i+1}" for i in range(similarity_matrix.shape[0])],
        colorscale='RdYlBu_r',
        text=[[f"{val:.3f}" for val in row] for row in similarity_matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Image %{y}<br>Text %{x}<br>Similarity: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Image-Text Similarity Matrix",
        xaxis_title="Text Descriptions",
        yaxis_title="Images",
        height=400,
        width=500
    )
    
    return fig

def create_embedding_space_viz(image_embeds: np.ndarray, text_embeds: np.ndarray, descriptions: List[str]) -> go.Figure:
    """Visualize joint embedding space (2D projection)."""
    # Simple 2D projection using first two dimensions
    fig = go.Figure()
    
    # Plot image embeddings
    fig.add_trace(go.Scatter(
        x=image_embeds[:, 0],
        y=image_embeds[:, 1],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='circle'),
        name='Images',
        text=[f"Image {i+1}" for i in range(len(image_embeds))],
        hovertemplate="<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
    ))
    
    # Plot text embeddings
    fig.add_trace(go.Scatter(
        x=text_embeds[:, 0],
        y=text_embeds[:, 1],
        mode='markers',
        marker=dict(size=12, color='red', symbol='square'),
        name='Texts',
        text=descriptions,
        hovertemplate="<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
    ))
    
    # Draw connections between matching pairs
    for i in range(min(len(image_embeds), len(text_embeds))):
        fig.add_trace(go.Scatter(
            x=[image_embeds[i, 0], text_embeds[i, 0]],
            y=[image_embeds[i, 1], text_embeds[i, 1]],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="Joint Embedding Space (2D Projection)",
        xaxis_title="Embedding Dimension 1",
        yaxis_title="Embedding Dimension 2",
        height=400,
        template="plotly_white"
    )
    
    return fig

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🎭 Multimodal Understanding Demo")
    
    # Sample descriptions
    st.markdown("### Text Descriptions")
    
    default_descriptions = [
        "a beautiful blue sky with gradient colors",
        "geometric black and white checkerboard pattern",
        "colorful flower with red petals and green center",
        "abstract animal texture with mixed colors"
    ]
    
    descriptions = []
    for i in range(batch_size):
        if i < len(default_descriptions):
            desc = st.text_input(f"Description {i+1}:", value=default_descriptions[i], key=f"desc_{i}")
        else:
            desc = st.text_input(f"Description {i+1}:", key=f"desc_{i}")
        descriptions.append(desc if desc else f"sample description {i+1}")

with col2:
    st.markdown("## ⚙️ Current Settings")
    
    config = get_config()
    
    st.markdown(f"""
    **Model Configuration:**
    - Size: {model_size}
    - Embed Dim: {embed_dim}
    - Image Resolution: {image_resolution}×{image_resolution}
    - Batch Size: {batch_size}
    
    **Vision Encoder:**
    - Layers: {vision_layers}
    - Width: {vision_width}
    
    **Text Encoder:**
    - Layers: {text_layers}
    - Width: {text_width}
    - Context Length: {context_length}
    - Vocab Size: {vocab_size:,}
    
    **Contrastive Learning:**
    - Temperature: {temperature_init}
    - Learnable: {'✅' if learnable_temperature else '❌'}
    
    **Optimizations:**
    - Fusion: {'✅' if config.optimization.enable_fusion else '❌'}
    - JIT: {'✅' if config.optimization.enable_jit else '❌'}
    - Auto Backend: {'✅' if config.optimization.auto_backend_selection else '❌'}
    """)

# Demo execution
if st.button("🚀 Run CLIP Multimodal Analysis", type="primary", use_container_width=True):
    with st.spinner("Initializing CLIP with multimodal capabilities..."):
        # Create CLIP configuration
        clip_config = {
            'embed_dim': embed_dim,
            'image_resolution': image_resolution,
            'vision_layers': vision_layers,
            'vision_width': vision_width,
            'vision_patch_size': 16,  # Fixed for demo
            'context_length': context_length,
            'vocab_size': vocab_size,
            'transformer_width': text_width,
            'transformer_heads': text_width // 64,
            'transformer_layers': text_layers,
            'temperature_init': temperature_init,
            'learnable_temperature': learnable_temperature
        }
        
        # Create CLIP model
        model = CLIP(**clip_config)
        param_count = sum(p.data.size for p in model.parameters().values())
        
        # Create sample data
        images_array, generated_descriptions = create_synthetic_images(batch_size, image_resolution)
        text_tokens = create_text_tokens(descriptions, vocab_size, context_length)
        
        images_tensor = Tensor(images_array)
        text_tensor = Tensor(text_tokens)
        
        # Run inference
        start_time = time.time()
        outputs = model(images_tensor, text_tensor, return_loss=True)
        inference_time = time.time() - start_time
        
        # Extract results
        image_embeds = outputs['image_embeds']
        text_embeds = outputs['text_embeds']
        logits_per_image = outputs['logits_per_image']
        logits_per_text = outputs['logits_per_text']
        contrastive_loss = outputs['loss']
        
        # Compute similarities
        similarity_matrix = image_embeds.data @ text_embeds.data.T
        
        # Apply temperature scaling
        if isinstance(model.logit_scale, type(model.token_embedding)):  # Parameter
            temperature = np.exp(model.logit_scale.data[0])
        else:
            temperature = np.exp(model.logit_scale)
        
        scaled_similarities = similarity_matrix * temperature
    
    # Display results
    st.markdown("## 📊 Results")
    
    # Model card
    st.markdown('<div class="clip-card">', unsafe_allow_html=True)
    st.markdown("### CLIP - Multimodal Vision-Language Understanding")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Parameters:** {param_count:,}")
        st.markdown(f"**Inference Time:** {inference_time:.4f}s")
    with col2:
        st.markdown(f"**Backend:** {images_tensor.backend.name}")
        st.markdown(f"**Temperature:** {temperature:.4f}")
    with col3:
        st.markdown(f"**Contrastive Loss:** {contrastive_loss.data[0]:.4f}")
        st.markdown(f"**Embedding Dim:** {embed_dim}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Similarity analysis
    st.markdown("### 🔗 Cross-Modal Similarity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Similarity Matrix")
        st.markdown('<div class="similarity-matrix">', unsafe_allow_html=True)
        
        # Display similarity matrix
        sim_fig = visualize_similarity_matrix(scaled_similarities, descriptions)
        st.plotly_chart(sim_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Best Matches")
        
        # Find best matches
        for i in range(batch_size):
            best_text_idx = np.argmax(scaled_similarities[i])
            best_similarity = scaled_similarities[i, best_text_idx]
            
            st.markdown(f"**Image {i+1}** matches best with:")
            st.markdown(f"- **Text {best_text_idx+1}**: \"{descriptions[best_text_idx]}\"")
            st.markdown(f"- **Similarity**: {best_similarity:.3f}")
            st.progress(float(np.clip(best_similarity, 0, 1)))
            st.markdown("---")
    
    # Joint embedding space
    st.markdown("### 🌐 Joint Embedding Space")
    st.markdown('<div class="embedding-space">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Embedding space visualization
        embed_fig = create_embedding_space_viz(image_embeds.data, text_embeds.data, descriptions)
        st.plotly_chart(embed_fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Embedding Statistics")
        
        # Compute embedding statistics
        image_norms = np.linalg.norm(image_embeds.data, axis=1)
        text_norms = np.linalg.norm(text_embeds.data, axis=1)
        
        st.metric("Avg Image Embed Norm", f"{np.mean(image_norms):.3f}")
        st.metric("Avg Text Embed Norm", f"{np.mean(text_norms):.3f}")
        st.metric("Embedding Similarity", f"{np.mean(np.diag(similarity_matrix)):.3f}")
        
        # Cross-modal alignment
        cross_modal_sim = np.mean([similarity_matrix[i, i] for i in range(min(similarity_matrix.shape))])
        st.metric("Cross-Modal Alignment", f"{cross_modal_sim:.3f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("### ⚡ Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        throughput = batch_size / inference_time
        st.metric("Throughput", f"{throughput:.1f} pairs/sec")
    with col2:
        latency = inference_time * 1000 / batch_size
        st.metric("Latency per Pair", f"{latency:.1f} ms")
    with col3:
        vision_ops = batch_size * vision_layers * (image_resolution // 16)**2
        st.metric("Vision Ops/sec", f"{vision_ops/inference_time/1e6:.2f}M")
    with col4:
        text_ops = batch_size * text_layers * context_length
        st.metric("Text Ops/sec", f"{text_ops/inference_time/1e3:.2f}K")
    
    # Feature analysis
    st.markdown("### 🧠 CLIP Multimodal Features")
    
    features = [
        ("🌐 Joint Embedding Space", "Images and text mapped to shared representation space"),
        ("🔗 Contrastive Learning", f"InfoNCE loss optimizes cross-modal similarity"),
        ("🖼️ Vision Transformer", f"Patch-based image processing with {vision_layers} layers"),
        ("📝 Text Transformer", f"Causal language modeling with {text_layers} layers"),
        ("🌡️ Temperature Scaling", f"Learnable temperature {temperature:.4f} for calibration"),
        ("⚡ Automatic Optimizations", "Fusion, JIT compilation across both modalities")
    ]
    
    for feature_name, feature_desc in features:
        st.markdown(f"**{feature_name}:** {feature_desc}")
    
    # Contrastive learning explanation
    st.markdown("### 🎯 Contrastive Learning Analysis")
    st.markdown('<div class="multimodal-info">', unsafe_allow_html=True)
    st.markdown(f"""
    **CLIP Training Objective:**
    
    **InfoNCE Loss**: Maximizes similarity between correct image-text pairs while minimizing similarity with incorrect pairs.
    
    **Current Results:**
    - **Contrastive Loss**: {contrastive_loss.data[0]:.4f}
    - **Temperature**: {temperature:.4f} (controls sharpness of similarity distribution)
    - **Positive Pairs**: Diagonal elements of similarity matrix
    - **Negative Pairs**: Off-diagonal elements
    
    **What This Means:**
    - Lower loss = Better alignment between matching image-text pairs
    - Higher temperature = Softer probability distributions
    - Cross-modal alignment score shows how well images and texts are aligned
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Architecture details
st.markdown("---")
st.markdown("## 🏗️ CLIP Architecture Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🖼️ Vision Encoder")
    st.markdown(f"""
    **Vision Transformer Configuration:**
    - **Input Resolution**: {image_resolution}×{image_resolution}
    - **Patch Size**: 16×16 (standard)
    - **Layers**: {vision_layers} transformer blocks
    - **Width**: {vision_width} dimensions
    - **Output**: {embed_dim}D visual features
    
    **Processing Pipeline:**
    1. Image → Patches → Linear projection
    2. Position embeddings added
    3. Transformer encoder layers
    4. Global representation extraction
    5. Projection to joint embedding space
    """)

with col2:
    st.markdown("### 📝 Text Encoder")
    st.markdown(f"""
    **Text Transformer Configuration:**
    - **Context Length**: {context_length} tokens
    - **Vocabulary**: {vocab_size:,} tokens
    - **Layers**: {text_layers} transformer blocks
    - **Width**: {text_width} dimensions
    - **Output**: {embed_dim}D text features
    
    **Processing Pipeline:**
    1. Text → Tokens → Embeddings
    2. Position embeddings added
    3. Causal transformer layers
    4. [EOS] token representation
    5. Projection to joint embedding space
    """)

# Applications
st.markdown("---")
st.markdown("## 🎯 CLIP Applications")

applications = [
    ("🔍 Zero-Shot Classification", "Classify images using natural language descriptions"),
    ("🔄 Cross-Modal Retrieval", "Find images using text queries or vice versa"),
    ("🎨 Creative Applications", "Generate art based on text descriptions"),
    ("📊 Content Moderation", "Understand image content through text analysis"),
    ("🌐 Multimodal Search", "Search across images and text simultaneously"),
    ("🤖 Visual Question Answering", "Answer questions about image content")
]

col1, col2 = st.columns(2)

for i, (app_name, app_desc) in enumerate(applications):
    col = col1 if i % 2 == 0 else col2
    with col:
        st.markdown(f"**{app_name}**")
        st.markdown(f"{app_desc}")
        st.markdown("---")

# Information section
st.markdown("## 📚 About CLIP")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Key Innovations
    - **Contrastive Learning**: Learn from image-text pairs
    - **Large Scale Training**: 400M image-text pairs
    - **Zero-Shot Transfer**: No fine-tuning needed
    - **Natural Language Supervision**: Train with descriptive text
    - **Joint Embedding**: Unified representation space
    """)

with col2:
    st.markdown("""
    ### 🚀 Automatic Optimizations
    - **Vision-Text Fusion**: Optimized cross-modal operations
    - **Attention Optimization**: Efficient multi-head attention
    - **JIT Compilation**: Runtime optimization for both encoders
    - **Backend Selection**: Optimal hardware utilization
    - **Zero Configuration**: Multimodal features work seamlessly
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🌟 CLIP Demo - Neural Architecture Framework</p>', 
    unsafe_allow_html=True
)