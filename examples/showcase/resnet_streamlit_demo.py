#!/usr/bin/env python3
"""
🏗️ ResNet Streamlit Demo - Interactive Computer Vision

Interactive web interface for ResNet deep residual networks with:
- Real-time image classification and computer vision
- Configurable architecture parameters and residual blocks
- Performance metrics and visualizations
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
from neural_arch.models.vision.resnet import ResNet18, ResNet34, ResNet50
from neural_arch.optim import AdamW
from neural_arch.functional import softmax
from neural_arch.optimization_config import configure, get_config
from neural_arch.backends import available_backends

# Page configuration
st.set_page_config(
    page_title="🏗️ ResNet Demo",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .resnet-card {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .image-preview {
        border: 2px solid #2ecc71;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .architecture-info {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🏗️ ResNet Computer Vision Demo</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Deep Residual Learning for Image Recognition</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🎛️ ResNet Configuration")

# Model settings
st.sidebar.subheader("Architecture Selection")
architecture = st.sidebar.selectbox("ResNet Architecture:", ["ResNet-18", "ResNet-34", "ResNet-50"], index=0)
num_classes = st.sidebar.slider("Number of Classes:", 10, 1000, 100, step=10)
batch_size = st.sidebar.slider("Batch Size:", 1, 4, 1)
image_size = st.sidebar.selectbox("Image Size:", [32, 64, 128], index=0)

# Advanced settings
st.sidebar.subheader("Advanced Features")
use_se = st.sidebar.checkbox("Squeeze-and-Excitation blocks", value=False, help="Channel attention mechanism")
drop_path_rate = st.sidebar.slider("Stochastic Depth Rate:", 0.0, 0.3, 0.0, step=0.05, help="Regularization technique")

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
    """Create synthetic images with different patterns."""
    img = np.zeros((3, size, size), dtype=np.float32)
    
    if pattern_type == "gradient":
        # Sky-like gradient
        for y in range(size):
            gradient = y / size
            img[2, y, :] = 0.8 - 0.4 * gradient  # Blue
            img[0, y, :] = 0.2 + 0.3 * gradient  # Red
            img[1, y, :] = 0.4 + 0.2 * gradient  # Green
    
    elif pattern_type == "checkerboard":
        # Checkerboard pattern
        square_size = max(4, size // 8)
        for y in range(0, size, square_size):
            for x in range(0, size, square_size):
                if ((y // square_size) + (x // square_size)) % 2:
                    img[:, y:y+square_size, x:x+square_size] = 0.9
                else:
                    img[:, y:y+square_size, x:x+square_size] = 0.1
    
    elif pattern_type == "radial":
        # Radial flower pattern
        center = size // 2
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                angle = np.arctan2(y - center, x - center)
                petals = np.sin(6 * angle) * np.exp(-dist / (size/4))
                img[0, y, x] = 0.5 + 0.3 * petals  # Red petals
                img[1, y, x] = 0.6 - dist / (size*2)  # Green center
                img[2, y, x] = 0.2  # Blue background
    
    else:  # texture
        # Random texture
        img = np.random.uniform(0.3, 0.7, (3, size, size)).astype(np.float32)
        for _ in range(3):
            y, x = np.random.randint(size//4, 3*size//4, 2)
            patch_size = size // 8
            intensity = np.random.uniform(0.7, 1.3)
            img[:, y:y+patch_size, x:x+patch_size] *= intensity
    
    # Add slight noise
    img += np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    return img

def display_image_grid(images: np.ndarray, titles: List[str]) -> go.Figure:
    """Display a grid of images using plotly."""
    fig = go.Figure()
    
    # Convert to displayable format (RGB, 0-255)
    display_images = []
    for i, img in enumerate(images):
        # Convert from (C, H, W) to (H, W, C) and scale to 0-255
        img_display = np.transpose(img, (1, 2, 0))
        img_display = (img_display * 255).astype(np.uint8)
        display_images.append(img_display)
    
    # Create subplot grid
    from plotly.subplots import make_subplots
    rows = 2
    cols = 2
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        specs=[[{"type": "image"}, {"type": "image"}],
               [{"type": "image"}, {"type": "image"}]]
    )
    
    for i, (img, title) in enumerate(zip(display_images, titles)):
        row = i // cols + 1
        col = i % cols + 1
        fig.add_trace(
            go.Image(z=img, name=title),
            row=row, col=col
        )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🖼️ Computer Vision Demo")
    
    # Image pattern selection
    st.markdown("### Select Image Patterns for Classification")
    
    pattern_types = ["gradient", "checkerboard", "radial", "texture"]
    pattern_names = ["Sky Gradient", "Checkerboard", "Radial Flower", "Random Texture"]
    
    selected_patterns = []
    for i, (pattern, name) in enumerate(zip(pattern_types, pattern_names)):
        if st.checkbox(f"{name}", value=i < batch_size, key=f"pattern_{i}"):
            selected_patterns.append((pattern, name))
    
    # Limit to batch size
    selected_patterns = selected_patterns[:batch_size]
    while len(selected_patterns) < batch_size:
        selected_patterns.append(("gradient", "Sky Gradient"))

with col2:
    st.markdown("## ⚙️ Current Settings")
    
    config = get_config()
    
    # Architecture info
    if architecture == "ResNet-18":
        layers_info = "2-2-2-2 layers, BasicBlock"
        theoretical_params = "11.7M (full size)"
    elif architecture == "ResNet-34":
        layers_info = "3-4-6-3 layers, BasicBlock"  
        theoretical_params = "21.8M (full size)"
    else:  # ResNet-50
        layers_info = "3-4-6-3 layers, Bottleneck"
        theoretical_params = "25.6M (full size)"
    
    st.markdown(f"""
    **Architecture:**
    - Model: {architecture}
    - Configuration: {layers_info}
    - Classes: {num_classes}
    - Image Size: {image_size}×{image_size}
    - Batch Size: {batch_size}
    
    **Advanced Features:**
    - SE Blocks: {'✅' if use_se else '❌'}
    - Stochastic Depth: {drop_path_rate:.2f}
    
    **Optimizations:**
    - Fusion: {'✅' if config.optimization.enable_fusion else '❌'}
    - JIT: {'✅' if config.optimization.enable_jit else '❌'}
    - Auto Backend: {'✅' if config.optimization.auto_backend_selection else '❌'}
    """)

# Demo execution
if st.button("🚀 Run ResNet Classification", type="primary", use_container_width=True):
    with st.spinner(f"Initializing {architecture} with residual learning..."):
        # Create ResNet model
        if architecture == "ResNet-18":
            model = ResNet18(
                num_classes=10,  # Simplified for demo
                use_se=use_se,
                drop_path_rate=drop_path_rate
            )
        elif architecture == "ResNet-34":
            model = ResNet34(
                num_classes=10,
                use_se=use_se,
                drop_path_rate=drop_path_rate
            )
        else:  # ResNet-50
            model = ResNet50(
                num_classes=10,
                use_se=use_se,
                drop_path_rate=drop_path_rate
            )
        
        param_count = sum(p.data.size for p in model.parameters().values())
        
        # Create sample images based on selected patterns
        images = []
        pattern_labels = []
        for i in range(batch_size):
            pattern_type, pattern_name = selected_patterns[i] if i < len(selected_patterns) else selected_patterns[0]
            img = create_synthetic_image(pattern_type, image_size)
            images.append(img)
            pattern_labels.append(pattern_name)
        
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
    st.markdown('<div class="resnet-card">', unsafe_allow_html=True)
    st.markdown(f"### {architecture} - Deep Residual Learning")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Parameters:** {param_count:,}")
        st.markdown(f"**Inference Time:** {inference_time:.4f}s")
    with col2:
        st.markdown(f"**Backend:** {images_tensor.backend.name}")
        st.markdown(f"**Image Size:** {image_size}×{image_size}")
    with col3:
        st.markdown(f"**SE Blocks:** {'✅' if use_se else '❌'}")
        st.markdown(f"**Stochastic Depth:** {drop_path_rate:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Image visualization and predictions
    st.markdown("### 🖼️ Input Images and Predictions")
    
    # Display images
    if len(images) > 0:
        fig = display_image_grid(images, pattern_labels)
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictions table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Classification Results")
        for i in range(batch_size):
            pattern_name = pattern_labels[i] if i < len(pattern_labels) else "Unknown"
            pred_class = predicted_classes[i]
            confidence = confidence_scores[i]
            
            st.markdown(f"**Image {i+1} ({pattern_name}):**")
            st.markdown(f"- Predicted Class: {pred_class}")
            st.markdown(f"- Confidence: {confidence:.3f}")
            st.markdown("---")
    
    with col2:
        st.markdown("### 📈 Confidence Distribution")
        
        # Confidence chart
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Image {i+1}" for i in range(batch_size)],
                y=confidence_scores,
                marker_color='#2ecc71',
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
        efficiency = param_count / (inference_time * 1000)
        st.metric("Efficiency", f"{efficiency:.0f} params/ms")
    with col4:
        pixels_per_sec = (batch_size * image_size * image_size * 3) / inference_time
        st.metric("Pixels/sec", f"{pixels_per_sec/1e6:.2f}M")
    
    # Architecture analysis
    st.markdown("### 🏗️ ResNet Architecture Features")
    
    features = [
        ("🔗 Residual Connections", "Skip connections enabling very deep networks"),
        ("🧱 Building Blocks", f"{'Bottleneck' if architecture == 'ResNet-50' else 'BasicBlock'} architecture"),
        ("📊 Batch Normalization", "Stable training and improved convergence"),
        ("🎯 Global Average Pooling", "Spatial dimension reduction for classification"),
        ("⚡ Automatic Optimizations", "Fusion, JIT compilation, and backend selection")
    ]
    
    if use_se:
        features.append(("🔍 Squeeze-Excitation", "Channel attention mechanism"))
    
    if drop_path_rate > 0:
        features.append(("🎲 Stochastic Depth", f"Regularization with {drop_path_rate:.2f} drop rate"))
    
    for feature_name, feature_desc in features:
        st.markdown(f"**{feature_name}:** {feature_desc}")
    
    # Residual learning explanation
    st.markdown("### 🔬 Residual Learning Benefits")
    st.markdown('<div class="architecture-info">', unsafe_allow_html=True)
    st.markdown("""
    **Why Residual Connections Work:**
    - **Vanishing Gradients**: Skip connections provide gradient highways
    - **Identity Mapping**: Easier to learn residual functions F(x) = H(x) - x  
    - **Deep Networks**: Enables training of 50+ layer networks
    - **Optimization**: Smoother loss landscape for better convergence
    - **Feature Reuse**: Lower layers can contribute directly to output
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Architecture comparison
st.markdown("---")
st.markdown("## 🏗️ ResNet Architecture Comparison")

comparison_data = {
    "Architecture": ["ResNet-18", "ResNet-34", "ResNet-50"],
    "Layers": ["2-2-2-2", "3-4-6-3", "3-4-6-3"],
    "Block Type": ["BasicBlock", "BasicBlock", "Bottleneck"],
    "Theoretical Params": ["11.7M", "21.8M", "25.6M"],
    "Best Use Case": ["Fast inference", "Balanced", "High accuracy"]
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Architecture Details")
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

with col2:
    st.markdown("### 🎯 Key Differences")
    st.markdown("""
    **ResNet-18**: Fastest, lightweight, good for mobile
    **ResNet-34**: More capacity, still efficient  
    **ResNet-50**: Bottleneck blocks, highest accuracy
    
    **Block Types:**
    - **BasicBlock**: 3×3 → 3×3 convolutions
    - **Bottleneck**: 1×1 → 3×3 → 1×1 (parameter efficient)
    """)

# Information section
st.markdown("---")
st.markdown("## 📚 About ResNet")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Key Features
    - **Deep Residual Learning**: Skip connections for very deep networks
    - **Residual Blocks**: Identity mappings + learned residuals
    - **Batch Normalization**: Stable training and faster convergence
    - **Global Average Pooling**: Spatial pooling for classification
    - **Scalable Architecture**: From ResNet-18 to ResNet-152
    """)

with col2:
    st.markdown("""
    ### 🚀 Automatic Optimizations
    - **Convolution Fusion**: Optimized conv-bn-relu patterns
    - **JIT Compilation**: Runtime optimization for convolutions
    - **Backend Selection**: Optimal hardware utilization
    - **Memory Management**: Efficient residual computations
    - **Zero Configuration**: Works out of the box
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🏗️ ResNet Demo - Neural Architecture Framework</p>', 
    unsafe_allow_html=True
)