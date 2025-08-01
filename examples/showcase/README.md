# 🎭 Interactive Model Showcase

Interactive Streamlit demonstrations showcasing the full capabilities of our neural architecture framework. Each demo provides real-time inference, parameter exploration, and visualization of state-of-the-art deep learning models.

## 🚀 Quick Start

### Installation
```bash
# Install Streamlit and visualization dependencies
pip install streamlit plotly matplotlib seaborn pillow

# Optional: Install framework in development mode
pip install -e ../../

# Verify installation
streamlit --version
```

### Launch Demos
```bash
# From the showcase directory
cd examples/showcase

# Individual model demos
streamlit run gpt2_streamlit_demo.py          # Text generation
streamlit run vit_streamlit_demo.py           # Image classification  
streamlit run clip_streamlit_demo.py          # Multimodal similarity
streamlit run bert_streamlit_demo.py          # Sentiment analysis

# Comprehensive demo (all models)
streamlit run comprehensive_demo.py           # Multi-model interface
```

## 🎯 Demo Catalog

### 📝 Language Models

#### 🤖 GPT-2 Text Generation (`gpt2_streamlit_demo.py`)
**Real-time autoregressive text generation with interactive controls**

```bash
streamlit run gpt2_streamlit_demo.py
```

**Features:**
- **Interactive Prompts**: Custom text input with real-time generation
- **Temperature Control**: Creativity vs consistency slider (0.1 - 2.0)
- **Length Control**: Generated text length adjustment (10-200 tokens)
- **Multiple Samples**: Generate multiple completions simultaneously
- **Model Variants**: Switch between different GPT-2 configurations
- **Export Options**: Download generated text as files

**Demo Highlights:**
- Creative writing assistance with prompt engineering
- Temperature effects visualization (low = coherent, high = creative)
- Real-time token-by-token generation display
- Probability distribution visualization for next tokens

**Use Cases:**
- Story completion and creative writing
- Code generation and documentation
- Email and content drafting
- Language style transfer experiments

#### 🧠 BERT Sentiment Analysis (`bert_streamlit_demo.py`)
**Real-time text classification with attention visualization**

```bash
streamlit run bert_streamlit_demo.py
```

**Features:**
- **Text Input**: Multi-line text analysis with instant classification
- **Sentiment Scoring**: Positive/negative probability distribution
- **Attention Visualization**: Token-level attention weight heatmaps  
- **Batch Processing**: Analyze multiple texts simultaneously
- **Confidence Metrics**: Model certainty indicators
- **Export Results**: Download analysis results as CSV/JSON

**Demo Highlights:**
- Real-time sentiment classification with confidence scores
- Interactive attention visualization showing model focus
- Comparison mode for analyzing multiple texts
- Performance metrics on sample datasets

**Use Cases:**
- Social media sentiment monitoring
- Customer feedback analysis
- Content moderation and filtering
- Market research and opinion mining

#### 🔬 Modern Transformer Demo (`modern_transformer_streamlit_demo.py`)
**Next-generation architecture with advanced features**

```bash
streamlit run modern_transformer_streamlit_demo.py
```

**Features:**
- **Advanced Architecture**: Pre-Norm + RoPE + SwiGLU + RMSNorm
- **Position Encoding**: Rotary position encoding visualization
- **Activation Functions**: SwiGLU vs traditional activation comparison
- **Normalization**: RMSNorm vs LayerNorm performance analysis
- **Architecture Comparison**: Side-by-side with standard transformers

**Demo Highlights:**
- Cutting-edge transformer innovations in action
- Visual comparison of architectural improvements
- Performance benchmarking against standard models
- Educational visualization of advanced components

### 👁️ Computer Vision Models

#### 🖼️ Vision Transformer Classification (`vit_streamlit_demo.py`)
**Image classification with patch-based attention visualization**

```bash
streamlit run vit_streamlit_demo.py
```

**Features:**
- **Image Upload**: Drag-and-drop image classification interface
- **Patch Visualization**: Visual breakdown of image patches
- **Attention Maps**: Multi-head attention pattern visualization
- **Classification Results**: Top-K predictions with confidence scores
- **Model Comparison**: Different ViT architectures side-by-side
- **Synthetic Data**: Built-in test images for quick experimentation

**Demo Highlights:**
- Real-time image classification with attention overlay
- Interactive patch grid showing transformer input processing
- Multi-head attention visualization across different layers
- Class activation mapping for interpretability

**Use Cases:**
- Medical image analysis and diagnosis
- Quality control in manufacturing
- Wildlife and species identification
- Art and style classification

#### 🏗️ ResNet Computer Vision (`resnet_streamlit_demo.py`)
**Deep residual learning with feature visualization**

```bash
streamlit run resnet_streamlit_demo.py
```

**Features:**
- **Deep Architecture**: Residual block visualization and analysis
- **Feature Maps**: Layer-by-layer feature extraction display
- **Skip Connections**: Gradient flow visualization through residuals
- **Architecture Variants**: ResNet-18, ResNet-34, ResNet-50 comparison
- **Performance Metrics**: Speed vs accuracy trade-offs
- **Gradient Analysis**: Backpropagation flow through deep networks

**Demo Highlights:**
- Visual explanation of residual learning benefits
- Deep network feature hierarchy exploration
- Gradient flow analysis preventing vanishing gradients
- Architecture scaling effects on performance

### 🌟 Multimodal Models

#### 🔗 CLIP Cross-Modal Demo (`clip_streamlit_demo.py`)
**Vision-language understanding with similarity search**

```bash
streamlit run clip_streamlit_demo.py
```

**Features:**
- **Image-Text Matching**: Upload images and find matching descriptions
- **Text-to-Image Search**: Query images using natural language
- **Similarity Visualization**: Cross-modal embedding space exploration
- **Zero-Shot Classification**: Classify images with custom text labels
- **Contrastive Learning**: Positive/negative pair analysis
- **Embedding Explorer**: 2D/3D visualization of learned representations

**Demo Highlights:**
- Real-time cross-modal similarity computation
- Interactive embedding space visualization with t-SNE/UMAP
- Zero-shot image classification with custom categories
- Contrastive learning dynamics visualization

**Use Cases:**
- Content-based image retrieval systems
- Automatic image captioning and tagging
- Visual question answering applications
- Cross-modal recommendation systems

### 🎛️ Comprehensive Multi-Model Interface

#### 🌐 Unified Demo Platform (`comprehensive_demo.py`)
**All models in one integrated interface**

```bash
streamlit run comprehensive_demo.py
```

**Features:**
- **Model Selection**: Switch between all available architectures
- **Unified Interface**: Consistent UX across different model types
- **Performance Comparison**: Side-by-side model benchmarking
- **Architecture Explorer**: Interactive model structure visualization
- **Batch Processing**: Process multiple inputs across different models
- **Export Dashboard**: Comprehensive results export functionality

**Demo Highlights:**
- One-stop interface for exploring all model capabilities
- Performance benchmarking across different architectures
- Educational comparison of model strengths and limitations
- Production-ready interface suitable for deployment

## 🛠️ Technical Implementation

### Streamlit Architecture Pattern
All demos follow a consistent architectural pattern for maintainability:

```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from neural_arch.core import Tensor
from neural_arch.models import get_model

def main():
    st.set_page_config(
        page_title="Model Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    # Sidebar configuration
    with st.sidebar:
        model_config = configure_model()
        demo_settings = configure_demo()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        user_input = get_user_input()
        
    with col2:
        if user_input:
            results = run_inference(user_input, model_config)
            display_results(results)
            
    # Visualization section
    st.subheader("Model Analysis")
    create_visualizations(results)

if __name__ == "__main__":
    main()
```

### Interactive Components

#### Real-time Inference
```python
@st.cache_resource
def load_model(model_type, config):
    """Cached model loading for performance."""
    model = get_model(model_type, config)
    return model

def run_inference(input_data, model):
    """Real-time model inference with progress tracking."""
    with st.spinner("Processing..."):
        start_time = time.time()
        
        # Convert input to tensor
        input_tensor = Tensor(np.array(input_data))
        
        # Model forward pass
        output = model(input_tensor)
        
        # Post-processing
        results = process_output(output)
        
        # Performance metrics
        inference_time = time.time() - start_time
        st.sidebar.metric("Inference Time", f"{inference_time:.3f}s")
        
        return results
```

#### Visualization Framework
```python
def create_attention_heatmap(attention_weights, tokens):
    """Interactive attention visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Attention Patterns",
        xaxis_title="Keys",
        yaxis_title="Queries"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_embedding_scatter(embeddings, labels):
    """2D/3D embedding space visualization."""
    fig = go.Figure(data=go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers+text',
        text=labels,
        marker=dict(size=10, opacity=0.7),
        textposition="top center"
    ))
    
    fig.update_layout(
        title="Embedding Space Visualization",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

## 📊 Performance Optimization

### Caching Strategy
```python
# Model caching for fast reload
@st.cache_resource
def load_model(model_name):
    return get_model(model_name)

# Data processing caching
@st.cache_data
def preprocess_data(raw_data):
    return processed_data

# Computation caching
@st.cache_data
def compute_embeddings(text_list):
    return embedding_matrix
```

### Memory Management
```python
def optimize_memory():
    """Memory optimization for long-running demos."""
    import gc
    
    # Clear unused variables
    gc.collect()
    
    # Monitor memory usage
    import psutil
    memory_usage = psutil.virtual_memory().percent
    st.sidebar.metric("Memory Usage", f"{memory_usage:.1f}%")
```

## 🎨 UI/UX Design Principles

### Consistent Layout
- **Sidebar**: Model configuration and settings
- **Main Area**: Input interface and primary results
- **Visualization**: Dedicated section for charts and analysis
- **Metrics**: Performance indicators in sidebar/footer

### Interactive Elements
- **Sliders**: Continuous parameters (temperature, confidence threshold)
- **Selectboxes**: Discrete choices (model variants, output format)
- **File Uploaders**: Image/text input with drag-and-drop
- **Text Areas**: Multi-line text input with syntax highlighting

### Responsive Design
```python
# Adaptive column layout
if st.sidebar.checkbox("Wide Layout"):
    col1, col2, col3 = st.columns([2, 1, 2])
else:
    col1, col2 = st.columns([1, 1])

# Mobile-friendly components
if st.session_state.get('mobile_mode'):
    st.components.v1.html(mobile_optimized_html)
```

## 🔧 Customization Guide

### Adding New Demos
1. **Create Demo File**: Follow naming pattern `[model]_streamlit_demo.py`
2. **Implement Standard Interface**: Use template pattern above
3. **Add Visualization**: Include model-specific visualizations
4. **Update Comprehensive Demo**: Add to unified interface
5. **Documentation**: Update this README with demo description

### Custom Visualizations
```python
def create_custom_plot(data, plot_type):
    """Template for custom visualization."""
    if plot_type == "heatmap":
        fig = go.Figure(data=go.Heatmap(z=data))
    elif plot_type == "3d_scatter":
        fig = go.Figure(data=go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode='markers'
        ))
    
    st.plotly_chart(fig, use_container_width=True)
```

### Theme Customization
```python
# Custom CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)
```

## 🚀 Deployment Options

### Local Development
```bash
# Development mode with auto-reload
streamlit run demo.py --server.runOnSave true

# Custom port and host
streamlit run demo.py --server.port 8080 --server.address 0.0.0.0
```

### Production Deployment
```bash
# Docker deployment
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "comprehensive_demo.py"]

# Cloud deployment (Streamlit Cloud, Heroku, etc.)
# Add streamlit config files and deployment scripts
```

### Performance Monitoring
```python
# Add performance tracking
import time
import logging

def track_usage(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        logging.info(f"Function {func.__name__} took {duration:.3f}s")
        return result
    return wrapper
```

## 📚 Educational Features

### Model Architecture Visualization
```python
def visualize_architecture(model_type):
    """Interactive model architecture diagram."""
    if model_type == "transformer":
        # Create interactive transformer diagram
        create_transformer_diagram()
    elif model_type == "cnn":
        # Create CNN layer visualization
        create_cnn_diagram()
```

### Learning Resources
Each demo includes:
- **Architecture Explanations**: In-context model descriptions
- **Parameter Impact**: Interactive parameter exploration
- **Performance Analysis**: Speed vs accuracy trade-offs
- **Use Case Examples**: Real-world application scenarios
- **Further Reading**: Links to papers and resources

## 🐛 Troubleshooting

### Common Issues

**Streamlit Import Errors**
```bash
pip install --upgrade streamlit
pip install --no-cache-dir streamlit
```

**Model Loading Issues**
```python
# Add error handling
try:
    model = load_model(model_type)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Please check model configuration")
```

**Memory Issues**
```python
# Reduce model size for demos
if st.sidebar.checkbox("Lite Mode"):
    config.hidden_size = config.hidden_size // 2
    config.num_layers = config.num_layers // 2
```

**Performance Issues**
```bash
# Enable caching
export STREAMLIT_ENABLE_CACHING=true

# Increase memory limit
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
```

## 🤝 Contributing

### Demo Development Guidelines
1. **Consistent UX**: Follow established UI patterns
2. **Performance**: Implement proper caching strategies
3. **Error Handling**: Graceful error messages and recovery
4. **Documentation**: Clear instructions and explanations
5. **Accessibility**: Mobile-friendly and keyboard navigation

### Code Quality Standards
- **Type Hints**: Full type annotation coverage
- **Docstrings**: Comprehensive function documentation
- **Error Handling**: Try-catch blocks with user-friendly messages
- **Performance**: Caching and memory optimization
- **Testing**: Unit tests for core functionality

---

## 🎯 Quick Demo Commands

```bash
# Fast start - text generation
streamlit run gpt2_streamlit_demo.py

# Computer vision - image classification  
streamlit run vit_streamlit_demo.py

# Multimodal - image-text matching
streamlit run clip_streamlit_demo.py

# All models - comprehensive interface
streamlit run comprehensive_demo.py
```

**🎭 Ready to explore?** Each demo provides hands-on experience with state-of-the-art deep learning models through intuitive web interfaces!