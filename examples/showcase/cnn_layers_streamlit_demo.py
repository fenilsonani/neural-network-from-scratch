#!/usr/bin/env python3
"""
CNN Layers Interactive Streamlit Demo

Interactive demonstration of CNN layer capabilities:
- Real-time convolution visualization
- Layer-by-layer feature map analysis
- Interactive parameter exploration
- Multiple architecture comparisons
- Performance benchmarking

Run with: streamlit run examples/showcase/cnn_layers_streamlit_demo.py
"""

import sys
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from neural_arch.core import Tensor
from neural_arch.nn import (
    AdaptiveAvgPool2d, BatchNorm2d, Conv1d, Conv2d, ConvTranspose2d,
    GlobalAvgPool2d, SpatialDropout2d, Sequential
)
from neural_arch.functional import relu
from neural_arch.optimization_config import configure

# Configure page
st.set_page_config(
    page_title="CNN Layers Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable optimizations
configure(
    enable_fusion=True,
    enable_jit=True,
    auto_backend_selection=True
)


@st.cache_resource
def create_sample_models():
    """Create sample CNN models for demonstration."""
    models = {
        "Simple CNN": Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1),
            BatchNorm2d(16),
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(32),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(64),
            AdaptiveAvgPool2d((4, 4))
        ),
        "Deep CNN": Sequential(
            Conv2d(3, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            Conv2d(32, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(64),
            Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(128),
            GlobalAvgPool2d()
        ),
        "CNN with Dropout": Sequential(
            Conv2d(3, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            SpatialDropout2d(0.1),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(64),
            SpatialDropout2d(0.2),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(128),
            AdaptiveAvgPool2d((2, 2))
        )
    }
    return models


@st.cache_data
def generate_sample_images(num_images=5, size=64):
    """Generate sample images for visualization."""
    images = []
    names = []
    
    for i in range(num_images):
        img = np.zeros((3, size, size), dtype=np.float32)
        
        if i == 0:  # Gradient
            for c in range(3):
                img[c] = np.linspace(0, 1, size).reshape(1, -1)
            names.append("Horizontal Gradient")
            
        elif i == 1:  # Checkerboard
            for x in range(0, size, 8):
                for y in range(0, size, 8):
                    if (x//8 + y//8) % 2 == 0:
                        img[:, y:y+8, x:x+8] = 1.0
            names.append("Checkerboard")
            
        elif i == 2:  # Circle
            center = size // 2
            y, x = np.ogrid[:size, :size]
            mask = (x - center)**2 + (y - center)**2 <= (size//4)**2
            img[0, mask] = 1.0
            img[1, mask] = 0.5
            names.append("Circle")
            
        elif i == 3:  # Stripes
            for y in range(0, size, 6):
                img[:, y:y+3, :] = 1.0
            names.append("Horizontal Stripes")
            
        else:  # Random
            img = np.random.rand(3, size, size).astype(np.float32)
            names.append("Random Noise")
        
        images.append(img)
    
    return images, names


def visualize_feature_maps(feature_maps, title="Feature Maps"):
    """Create visualization of feature maps."""
    if len(feature_maps.shape) != 4:  # (batch, channels, height, width)
        st.error("Expected 4D feature maps")
        return
    
    batch_size, num_channels, height, width = feature_maps.shape
    
    # Select first batch and limit number of channels to display
    max_channels = min(16, num_channels)
    maps_to_show = feature_maps[0, :max_channels]
    
    # Create subplot grid
    cols = 4
    rows = (max_channels + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Channel {i}" for i in range(max_channels)],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    for i in range(max_channels):
        row = i // cols + 1
        col = i % cols + 1
        
        feature_map = maps_to_show[i]
        
        fig.add_trace(
            go.Heatmap(
                z=feature_map,
                colorscale='Viridis',
                showscale=False,
                hovertemplate=f'Channel {i}<br>x: %{{x}}<br>y: %{{y}}<br>Value: %{{z:.3f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=title,
        height=150 * rows,
        showlegend=False
    )
    
    # Remove axis labels for cleaner look
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig


def run_layer_analysis(input_tensor, layer, layer_name):
    """Run forward pass through a single layer and analyze output."""
    start_time = time.time()
    
    if isinstance(layer, Conv2d):
        output = layer(input_tensor)
        output = relu(output)  # Apply ReLU after conv
    else:
        output = layer(input_tensor)
    
    inference_time = time.time() - start_time
    
    return output, inference_time


def main():
    """Main Streamlit application."""
    st.title("🔬 CNN Layers Interactive Demo")
    st.markdown("**Explore Convolutional Neural Network layers with real-time visualization**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        models = create_sample_models()
        selected_model = st.selectbox(
            "Choose CNN Architecture:",
            list(models.keys()),
            help="Select different CNN architectures to compare"
        )
        
        # Image selection
        st.subheader("📸 Input Image")
        sample_images, image_names = generate_sample_images()
        
        selected_image_idx = st.selectbox(
            "Sample Image:",
            range(len(image_names)),
            format_func=lambda x: image_names[x]
        )
        
        # Custom image size
        image_size = st.slider(
            "Image Size:",
            min_value=32, max_value=128, value=64, step=16,
            help="Size of the input image (affects processing time)"
        )
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        show_feature_maps = st.checkbox("Show Feature Maps", value=True)
        show_performance = st.checkbox("Show Performance Metrics", value=True)
        layer_by_layer = st.checkbox("Layer-by-Layer Analysis", value=False)
        
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Model Architecture")
        
        # Display model architecture
        model = models[selected_model]
        
        st.write("**Layers:**")
        for i, layer in enumerate(model._modules_list):
            layer_type = layer.__class__.__name__
            if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                st.write(f"{i+1}. {layer_type} ({layer.in_channels}→{layer.out_channels})")
            else:
                st.write(f"{i+1}. {layer_type}")
        
        # Input image visualization
        st.subheader("🖼️ Input Image")
        
        # Regenerate image with current size if needed
        if image_size != 64:
            current_images, _ = generate_sample_images(len(image_names), image_size)
            input_image = current_images[selected_image_idx]
        else:
            input_image = sample_images[selected_image_idx]
        
        # Display input image
        img_display = np.transpose(input_image, (1, 2, 0))
        st.image(img_display, caption=image_names[selected_image_idx], width=200)
        
        # Image statistics
        st.write(f"**Shape:** {input_image.shape}")
        st.write(f"**Range:** [{input_image.min():.3f}, {input_image.max():.3f}]")
        st.write(f"**Mean:** {input_image.mean():.3f}")
        st.write(f"**Std:** {input_image.std():.3f}")
    
    with col2:
        st.subheader("🔬 CNN Analysis Results")
        
        # Run inference
        input_tensor = Tensor(input_image.reshape(1, *input_image.shape), requires_grad=True)
        
        if layer_by_layer:
            # Layer-by-layer analysis
            st.write("**Layer-by-Layer Feature Maps:**")
            
            current_tensor = input_tensor
            total_time = 0
            
            for i, layer in enumerate(model._modules_list):
                layer_name = f"Layer {i+1}: {layer.__class__.__name__}"
                
                output_tensor, layer_time = run_layer_analysis(current_tensor, layer, layer_name)
                total_time += layer_time
                
                # Show layer info
                with st.expander(f"📊 {layer_name}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Input Shape:** {current_tensor.shape}")
                        st.write(f"**Output Shape:** {output_tensor.shape}")
                        st.write(f"**Time:** {layer_time*1000:.2f}ms")
                    
                    with col_b:
                        if hasattr(layer, 'weight'):
                            num_params = np.prod(layer.weight.shape)
                            st.write(f"**Parameters:** {num_params:,}")
                        
                        if len(output_tensor.shape) == 4:  # Feature maps
                            st.write(f"**Channels:** {output_tensor.shape[1]}")
                            st.write(f"**Spatial:** {output_tensor.shape[2]}×{output_tensor.shape[3]}")
                    
                    # Show feature maps for conv layers
                    if show_feature_maps and len(output_tensor.shape) == 4 and output_tensor.shape[1] > 1:
                        fig = visualize_feature_maps(output_tensor.data, f"{layer_name} Output")
                        st.plotly_chart(fig, use_container_width=True)
                
                current_tensor = output_tensor
            
            # Final results
            st.success(f"**Total Processing Time:** {total_time*1000:.2f}ms")
            st.write(f"**Final Output Shape:** {current_tensor.shape}")
        
        else:
            # Full model analysis
            start_time = time.time()
            
            current_tensor = input_tensor
            intermediate_outputs = []
            
            for layer in model._modules_list:
                if isinstance(layer, Conv2d):
                    current_tensor = layer(current_tensor)
                    current_tensor = relu(current_tensor)
                else:
                    current_tensor = layer(current_tensor)
                
                intermediate_outputs.append(current_tensor.data.copy())
            
            total_time = time.time() - start_time
            
            # Performance metrics
            if show_performance:
                col_perf1, col_perf2, col_perf3 = st.columns(3)
                
                with col_perf1:
                    st.metric("Processing Time", f"{total_time*1000:.2f}ms")
                
                with col_perf2:
                    total_params = sum(
                        np.prod(layer.weight.shape) if hasattr(layer, 'weight') else 0
                        for layer in model._modules_list
                    )
                    st.metric("Total Parameters", f"{total_params:,}")
                
                with col_perf3:
                    st.metric("Output Size", f"{current_tensor.shape[1:]}")
            
            # Feature map visualization
            if show_feature_maps:
                st.write("**Feature Maps from Key Layers:**")
                
                # Show feature maps from conv layers
                conv_outputs = []
                conv_names = []
                
                layer_idx = 0
                for i, layer in enumerate(model._modules_list):
                    if isinstance(layer, Conv2d):
                        conv_outputs.append(intermediate_outputs[i])
                        conv_names.append(f"Conv2d Layer {layer_idx + 1}")
                        layer_idx += 1
                
                # Display feature maps in tabs
                if conv_outputs:
                    tabs = st.tabs(conv_names[:4])  # Limit to first 4 conv layers
                    
                    for tab, output, name in zip(tabs, conv_outputs[:4], conv_names[:4]):
                        with tab:
                            fig = visualize_feature_maps(output, name)
                            st.plotly_chart(fig, use_container_width=True)
    
    # Architecture comparison section
    st.subheader("📈 Architecture Comparison")
    
    if st.button("🔄 Compare All Architectures"):
        comparison_data = []
        
        progress_bar = st.progress(0)
        
        for i, (name, model) in enumerate(models.items()):
            # Run inference
            start_time = time.time()
            
            test_tensor = input_tensor
            for layer in model._modules_list:
                if isinstance(layer, Conv2d):
                    test_tensor = layer(test_tensor)
                    test_tensor = relu(test_tensor)
                else:
                    test_tensor = layer(test_tensor)
            
            inference_time = time.time() - start_time
            
            # Calculate parameters
            total_params = sum(
                np.prod(layer.weight.shape) if hasattr(layer, 'weight') else 0
                for layer in model.modules_list
            )
            
            comparison_data.append({
                'Architecture': name,
                'Inference Time (ms)': inference_time * 1000,
                'Parameters': total_params,
                'Output Shape': str(test_tensor.shape[1:]),
                'Memory Usage (MB)': test_tensor.data.nbytes / (1024 * 1024)
            })
            
            progress_bar.progress((i + 1) / len(models))
        
        # Display comparison table
        st.write("**Performance Comparison:**")
        st.dataframe(comparison_data, use_container_width=True)
        
        # Visualization
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_time = px.bar(
                comparison_data,
                x='Architecture',
                y='Inference Time (ms)',
                title="Inference Time Comparison"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col_chart2:
            fig_params = px.bar(
                comparison_data,
                x='Architecture',
                y='Parameters',
                title="Parameter Count Comparison"
            )
            st.plotly_chart(fig_params, use_container_width=True)
    
    # Educational content
    with st.expander("📚 Learn About CNN Layers"):
        st.markdown("""
        **Convolutional Neural Networks (CNNs) are the backbone of computer vision:**
        
        **🔍 Conv2d Layers:**
        - Extract local features using learnable filters
        - Preserve spatial relationships in images
        - Parameters: kernel_size, stride, padding, channels
        
        **📊 BatchNorm Layers:**
        - Normalize activations for stable training
        - Reduce internal covariate shift
        - Include learnable scale and shift parameters
        
        **💧 SpatialDropout:**
        - Regularization technique for CNNs
        - Drops entire feature maps instead of individual neurons
        - Prevents overfitting in convolutional layers
        
        **🎯 Pooling Layers:**
        - Reduce spatial dimensions
        - AdaptiveAvgPool: Fixed output size regardless of input
        - GlobalAvgPool: Reduce to single value per channel
        
        **🔄 ConvTranspose:**
        - Upsampling operation (learned interpolation)
        - Used in autoencoders, GANs, segmentation
        - "Reverse" of convolution operation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🔬 **CNN Layers Demo** - Powered by Neural Architecture Framework")


if __name__ == "__main__":
    main()