#!/usr/bin/env python3
"""
🖼️ Vision Transformer Streamlit Demo - Transformer for Computer Vision

Interactive web interface for Vision Transformer with:
- Real-time image classification with patch embeddings
- Configurable transformer parameters and attention visualization
- Performance metrics and patch-based analysis
- Automatic optimizations showcase
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
from neural_arch.models.vision.vision_transformer import VisionTransformer
from neural_arch.optim import AdamW
from neural_arch.functional import softmax
from neural_arch.optimization_config import configure, get_config
from neural_arch.backends import available_backends

# Page configuration
st.set_page_config(
    page_title="🖼️ Vision Transformer Demo",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #9b59b6;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .vit-card {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .patch-visualization {
        border: 2px solid #9b59b6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .attention-info {
        background: #f4e8f7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #9b59b6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🖼️ Vision Transformer Demo</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Transformer Architecture Applied to Computer Vision</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🎛️ ViT Configuration")

# Model settings
st.sidebar.subheader("Architecture Settings")
img_size = st.sidebar.selectbox("Image Size:", [64, 128, 224], index=0)
patch_size = st.sidebar.selectbox("Patch Size:", [8, 16, 32], index=1)
embed_dim = st.sidebar.selectbox("Embedding Dimension:", [256, 384, 512, 768], index=1)
depth = st.sidebar.slider("Transformer Depth:", 2, 12, 6)
num_heads = st.sidebar.slider("Attention Heads:", 4, 16, 8)
num_classes = st.sidebar.slider("Number of Classes:", 10, 1000, 100, step=10)

# Advanced settings
st.sidebar.subheader("Advanced Features")
drop_rate = st.sidebar.slider("Dropout Rate:", 0.0, 0.3, 0.1, step=0.05)
attn_drop_rate = st.sidebar.slider("Attention Dropout:", 0.0, 0.3, 0.0, step=0.05)
global_pool = st.sidebar.selectbox("Global Pooling:", ["token", "avg"], index=0)

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
def create_synthetic_image(pattern_type: str, size: int) -> np.ndarray:
    """Create synthetic images with different patterns for ViT analysis."""
    img = np.zeros((3, size, size), dtype=np.float32)
    
    if pattern_type == "geometric":
        # Geometric shapes pattern
        center = size // 2
        for y in range(size):
            for x in range(size):
                dist_center = np.sqrt((x - center)**2 + (y - center)**2)
                # Create concentric circles
                if dist_center < size//6:
                    img[0, y, x] = 0.8  # Red center
                elif dist_center < size//4:
                    img[1, y, x] = 0.8  # Green ring
                elif dist_center < size//3:
                    img[2, y, x] = 0.8  # Blue outer ring
    
    elif pattern_type == "stripes":
        # Vertical and horizontal stripes
        stripe_width = size // 8
        for y in range(size):
            for x in range(size):
                # Vertical stripes
                if (x // stripe_width) % 2:
                    img[0, y, x] = 0.7
                # Horizontal stripes
                if (y // stripe_width) % 2:
                    img[1, y, x] = 0.7
                # Diagonal pattern
                if ((x + y) // stripe_width) % 2:
                    img[2, y, x] = 0.5
    
    elif pattern_type == "grid":
        # Grid pattern with patches
        grid_size = size // 4
        for i in range(4):
            for j in range(4):
                y_start, y_end = i * grid_size, (i + 1) * grid_size
                x_start, x_end = j * grid_size, (j + 1) * grid_size
                
                # Different colors for different grid cells
                color_intensity = 0.3 + 0.5 * ((i + j) % 2)
                channel = (i + j) % 3
                img[channel, y_start:y_end, x_start:x_end] = color_intensity
    
    else:  # "complex"
        # Complex pattern combining multiple elements
        center = size // 2
        for y in range(size):
            for x in range(size):
                # Radial pattern
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                angle = np.arctan2(y - center, x - center)
                
                # Multi-frequency pattern
                r_pattern = 0.5 + 0.3 * np.sin(4 * angle) * np.exp(-dist / (size/3))
                g_pattern = 0.5 + 0.3 * np.cos(6 * angle) * np.exp(-dist / (size/4))
                b_pattern = 0.5 + 0.2 * np.sin(8 * angle) * np.exp(-dist / (size/5))
                
                img[0, y, x] = r_pattern
                img[1, y, x] = g_pattern
                img[2, y, x] = b_pattern
    
    # Add slight noise
    img += np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    return img

def visualize_patches(image: np.ndarray, patch_size: int) -> go.Figure:
    """Visualize how the image is divided into patches."""
    img_size = image.shape[1]  # Assuming square image
    num_patches_per_side = img_size // patch_size
    
    # Convert to displayable format
    img_display = np.transpose(image, (1, 2, 0))
    img_display = (img_display * 255).astype(np.uint8)
    
    fig = go.Figure()
    
    # Add the image
    fig.add_trace(go.Image(z=img_display, name="Original Image"))
    
    # Add patch boundaries
    for i in range(num_patches_per_side + 1):
        # Vertical lines
        fig.add_shape(
            type="line",
            x0=i * patch_size, y0=0,
            x1=i * patch_size, y1=img_size,
            line=dict(color="red", width=2)
        )
        # Horizontal lines
        fig.add_shape(
            type="line",
            x0=0, y0=i * patch_size,
            x1=img_size, y1=i * patch_size,
            line=dict(color="red", width=2)
        )
    
    fig.update_layout(
        title=f"Image Divided into {patch_size}×{patch_size} Patches",
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        height=400,
        width=400
    )
    
    return fig

# Calculate derived values
num_patches = (img_size // patch_size) ** 2
head_dim = embed_dim // num_heads

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🎨 Vision Transformer Analysis")
    
    # Pattern selection for analysis
    st.markdown("### Select Image Pattern for Analysis")
    
    pattern_type = st.selectbox(
        "Image Pattern:",
        ["geometric", "stripes", "grid", "complex"],
        format_func=lambda x: {
            "geometric": "🔵 Geometric Shapes",
            "stripes": "📏 Stripes Pattern", 
            "grid": "⬜ Grid Pattern",
            "complex": "🌟 Complex Pattern"
        }[x]
    )
    
    # Batch size
    batch_size = st.slider("Batch Size:", 1, 4, 2)

with col2:
    st.markdown("## ⚙️ Current Settings")
    
    config = get_config()
    
    st.markdown(f"""
    **Architecture:**
    - Image Size: {img_size}×{img_size}
    - Patch Size: {patch_size}×{patch_size}
    - Patches: {num_patches} ({img_size//patch_size}×{img_size//patch_size})
    - Embed Dim: {embed_dim}
    - Depth: {depth} layers
    - Attention Heads: {num_heads}
    - Head Dimension: {head_dim}
    
    **Advanced:**
    - Dropout: {drop_rate:.2f}
    - Attention Dropout: {attn_drop_rate:.2f}
    - Global Pool: {global_pool}
    - Classes: {num_classes}
    
    **Optimizations:**
    - Fusion: {'✅' if config.optimization.enable_fusion else '❌'}
    - JIT: {'✅' if config.optimization.enable_jit else '❌'}
    - Auto Backend: {'✅' if config.optimization.auto_backend_selection else '❌'}
    """)

# Demo execution
if st.button("🚀 Run Vision Transformer", type="primary", use_container_width=True):
    with st.spinner("Initializing Vision Transformer with attention mechanisms..."):
        # Create ViT model
        model = VisionTransformer(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            global_pool=global_pool
        )
        
        param_count = sum(p.data.size for p in model.parameters().values())
        
        # Create sample images
        images = []
        for i in range(batch_size):
            img = create_synthetic_image(pattern_type, img_size)
            images.append(img)
        
        images_array = np.stack(images, axis=0)
        images_tensor = Tensor(images_array)
        
        # Run inference
        start_time = time.time()
        outputs = model(images_tensor)
        inference_time = time.time() - start_time
        
        # Get predictions
        probabilities = softmax(outputs, axis=-1)
        predicted_classes = np.argmax(probabilities.data, axis=-1)
        confidence_scores = np.max(probabilities.data, axis=-1)
    
    # Display results
    st.markdown("## 📊 Results")
    
    # Model card
    st.markdown('<div class="vit-card">', unsafe_allow_html=True)
    st.markdown("### Vision Transformer - Attention for Computer Vision")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Parameters:** {param_count:,}")
        st.markdown(f"**Inference Time:** {inference_time:.4f}s")
    with col2:
        st.markdown(f"**Backend:** {images_tensor.backend.name}")
        st.markdown(f"**Patches per Image:** {num_patches}")
    with col3:
        st.markdown(f"**Attention Heads:** {num_heads}")
        st.markdown(f"**Transformer Layers:** {depth}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Patch visualization
    st.markdown("### 🧩 Patch Embedding Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image with Patch Boundaries")
        if len(images) > 0:
            patch_fig = visualize_patches(images[0], patch_size)
            st.plotly_chart(patch_fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Patch Statistics")
        st.metric("Total Patches", num_patches)
        st.metric("Patch Dimension", f"{patch_size}×{patch_size}")
        st.metric("Sequence Length", num_patches + 1)  # +1 for CLS token
        st.metric("Embedding Dimension", embed_dim)
        
        # Patch processing info
        st.markdown('<div class="attention-info">', unsafe_allow_html=True)
        st.markdown(f"""
        **Patch Processing:**
        1. Image divided into {num_patches} patches
        2. Each patch flattened to {patch_size * patch_size * 3} pixels
        3. Linear projection to {embed_dim} dimensions
        4. Position embeddings added
        5. CLS token prepended for classification
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Classification results
    st.markdown("### 🎯 Classification Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Predictions")
        for i in range(batch_size):
            pred_class = predicted_classes[i]
            confidence = confidence_scores[i]
            
            st.markdown(f"**Image {i+1}:**")
            st.markdown(f"- Predicted Class: {pred_class}")
            st.markdown(f"- Confidence: {confidence:.3f}")
            st.progress(float(confidence))
            st.markdown("---")
    
    with col2:
        st.markdown("#### Confidence Distribution")
        
        # Confidence chart
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Image {i+1}" for i in range(batch_size)],
                y=confidence_scores,
                marker_color='#9b59b6',
                text=[f"{conf:.3f}" for conf in confidence_scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Classification Confidence",
            xaxis_title="Images",
            yaxis_title="Confidence Score",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("### ⚡ Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        throughput = batch_size / inference_time
        st.metric("Throughput", f"{throughput:.1f} images/sec")
    with col2:
        latency = inference_time * 1000 / batch_size
        st.metric("Latency per Image", f"{latency:.1f} ms")
    with col3:
        patches_per_sec = (batch_size * num_patches) / inference_time
        st.metric("Patches/sec", f"{patches_per_sec:.0f}")
    with col4:
        attention_ops = batch_size * depth * num_heads * num_patches**2
        ops_per_sec = attention_ops / inference_time
        st.metric("Attention Ops/sec", f"{ops_per_sec/1e6:.2f}M")
    
    # Architecture analysis
    st.markdown("### 🏗️ Vision Transformer Features")
    
    features = [
        ("🧩 Patch Embeddings", f"Images split into {num_patches} patches of size {patch_size}×{patch_size}"),
        ("🎯 Multi-Head Attention", f"{num_heads} attention heads with {head_dim} dimensions each"),
        ("📍 Position Embeddings", "Learnable position information for spatial understanding"),
        ("🏷️ Classification Token", "[CLS] token for global image representation"),
        ("🔄 Transformer Layers", f"{depth} encoder layers with self-attention and MLP"),
        ("⚡ Automatic Optimizations", "Fusion, JIT compilation, and backend selection")
    ]
    
    for feature_name, feature_desc in features:
        st.markdown(f"**{feature_name}:** {feature_desc}")
    
    # Attention mechanism explanation
    st.markdown("### 🧠 Self-Attention in Vision")
    st.markdown('<div class="attention-info">', unsafe_allow_html=True)
    st.markdown(f"""
    **How Vision Transformer Processes Images:**
    
    1. **Patch Extraction**: Image → {num_patches} patches of {patch_size}×{patch_size} pixels
    2. **Linear Projection**: Each patch → {embed_dim}D embedding vector
    3. **Position Encoding**: Add learnable position embeddings
    4. **Self-Attention**: Patches attend to all other patches globally
    5. **MLP Processing**: Feed-forward networks process attended features
    6. **Classification**: [CLS] token aggregates global information
    
    **Attention Benefits:**
    - **Global Context**: Each patch can attend to any other patch
    - **Spatial Relationships**: Learns spatial dependencies automatically
    - **Scalability**: Attention complexity scales with number of patches
    - **Flexibility**: No convolution bias, learns spatial structure from data
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Architecture comparison
st.markdown("---")
st.markdown("## 🔄 ViT vs CNN Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🖼️ Vision Transformer")
    st.markdown("""
    **Advantages:**
    - Global attention from the start
    - No inductive bias about locality
    - Excellent with large datasets
    - Scalable architecture
    - Unified processing for all patches
    
    **Characteristics:**
    - Patch-based processing
    - Self-attention mechanisms
    - Position embeddings
    - Transformer encoder layers
    """)

with col2:
    st.markdown("### 🔲 Convolutional Networks")
    st.markdown("""
    **Advantages:**
    - Strong inductive bias for images
    - Translation invariance
    - Local feature extraction
    - Parameter sharing
    - Works well with smaller datasets
    
    **Characteristics:**
    - Convolution operations
    - Pooling layers
    - Hierarchical features
    - Spatial locality bias
    """)

# Technical details
st.markdown("---")
st.markdown("## 📚 About Vision Transformers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Key Innovations
    - **Patch Embeddings**: Treat image patches as tokens
    - **Self-Attention**: Global interactions between patches
    - **Position Embeddings**: Spatial awareness without convolution
    - **Classification Token**: Global image representation
    - **Scalability**: Performance improvements with model/data size
    """)

with col2:
    st.markdown("""
    ### 🚀 Automatic Optimizations
    - **Attention Fusion**: Optimized multi-head attention
    - **JIT Compilation**: Runtime optimization for transformers
    - **Backend Selection**: Optimal hardware utilization
    - **Memory Management**: Efficient attention computations
    - **Zero Configuration**: Works out of the box
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🖼️ Vision Transformer Demo - Neural Architecture Framework</p>', 
    unsafe_allow_html=True
)