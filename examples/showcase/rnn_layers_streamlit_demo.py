#!/usr/bin/env python3
"""
RNN Layers Interactive Streamlit Demo

Interactive demonstration of RNN layer capabilities:
- Real-time sequence processing visualization
- Hidden state evolution analysis
- Architecture comparison (RNN vs LSTM vs GRU)
- Bidirectional processing exploration
- Performance benchmarking

Run with: streamlit run examples/showcase/rnn_layers_streamlit_demo.py
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
from neural_arch.nn import Dropout, Embedding, GRU, Linear, LSTM, RNN, Sequential
from neural_arch.functional import softmax
from neural_arch.optimization_config import configure

# Configure page
st.set_page_config(
    page_title="RNN Layers Demo",
    page_icon="🔄",
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
    """Create sample RNN models for demonstration."""
    vocab_size = 100
    embed_dim = 32
    hidden_dim = 64
    
    models = {
        "Basic RNN": {
            "embedding": Embedding(vocab_size, embed_dim),
            "rnn": RNN(embed_dim, hidden_dim, num_layers=1, batch_first=True),
            "classifier": Linear(hidden_dim, vocab_size)
        },
        "Bidirectional RNN": {
            "embedding": Embedding(vocab_size, embed_dim),
            "rnn": RNN(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True),
            "classifier": Linear(hidden_dim * 2, vocab_size)
        },
        "LSTM": {
            "embedding": Embedding(vocab_size, embed_dim),
            "rnn": LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True),
            "classifier": Linear(hidden_dim, vocab_size)
        },
        "Bidirectional LSTM": {
            "embedding": Embedding(vocab_size, embed_dim),
            "rnn": LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True),
            "classifier": Linear(hidden_dim * 2, vocab_size)
        },
        "GRU": {
            "embedding": Embedding(vocab_size, embed_dim),
            "rnn": GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True),
            "classifier": Linear(hidden_dim, vocab_size)
        },
        "Bidirectional GRU": {
            "embedding": Embedding(vocab_size, embed_dim),
            "rnn": GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True),
            "classifier": Linear(hidden_dim * 2, vocab_size)
        }
    }
    return models


@st.cache_data
def generate_sample_sequences():
    """Generate sample sequences for demonstration."""
    sequences = {
        "Arithmetic": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Fibonacci": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        "Powers of 2": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "Random": np.random.randint(1, 100, 10).tolist(),
        "Repeating": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        "Palindrome": [1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
        "Alternating": [1, 10, 2, 20, 3, 30, 4, 40, 5, 50],
        "Sine Wave": [int(50 + 30 * np.sin(i * 0.5)) for i in range(10)]
    }
    return sequences


def run_rnn_forward(model_components, input_sequence, return_states=False):
    """Run forward pass through RNN model and optionally return hidden states."""
    embedding = model_components["embedding"]
    rnn = model_components["rnn"]
    classifier = model_components["classifier"]
    
    # Convert sequence to tensor
    input_tensor = Tensor(np.array(input_sequence).reshape(1, -1), requires_grad=True)
    
    # Embedding
    embedded = embedding(input_tensor)
    
    # RNN forward
    start_time = time.time()
    
    if isinstance(rnn, LSTM):
        rnn_output, (h_n, c_n) = rnn(embedded)
        final_hidden = h_n
        final_cell = c_n if return_states else None
    else:
        rnn_output, h_n = rnn(embedded)
        final_hidden = h_n
        final_cell = None
    
    inference_time = time.time() - start_time
    
    # Classification (predict next token for each position)
    predictions = classifier(rnn_output)
    
    # Apply softmax for probabilities
    pred_probs = softmax(predictions).data
    
    results = {
        'predictions': pred_probs,
        'hidden_states': rnn_output.data,
        'final_hidden': final_hidden.data,
        'inference_time': inference_time
    }
    
    if final_cell is not None:
        results['final_cell'] = final_cell.data
    
    return results


def visualize_hidden_states(hidden_states, title="Hidden State Evolution"):
    """Visualize the evolution of hidden states over time."""
    # hidden_states: (batch, seq_len, hidden_dim) or (batch, seq_len, hidden_dim * 2) for bidirectional
    states = hidden_states[0]  # Take first batch
    seq_len, hidden_dim = states.shape
    
    fig = go.Figure(data=go.Heatmap(
        z=states.T,  # Transpose to show hidden dims on y-axis
        x=list(range(seq_len)),
        y=list(range(hidden_dim)),
        colorscale='RdBu',
        zmid=0,
        hoverongaps=False,
        hovertemplate='Time: %{x}<br>Hidden Unit: %{y}<br>Activation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Hidden Unit",
        height=400
    )
    
    return fig


def visualize_prediction_probabilities(predictions, input_sequence, vocab_size=100):
    """Visualize prediction probabilities over time."""
    seq_len = predictions.shape[1]
    
    # Get top-k predictions for each time step
    k = 5
    top_k_indices = np.argsort(predictions[0], axis=1)[:, -k:]
    top_k_probs = np.sort(predictions[0], axis=1)[:, -k:]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Top-5 Predictions", "Prediction Confidence"],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Top predictions heatmap
    time_steps = list(range(seq_len))
    
    for k_idx in range(k):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=top_k_indices[:, k_idx],
                mode='markers+lines',
                name=f'Top-{k-k_idx}',
                marker=dict(size=top_k_probs[:, k_idx] * 20)  # Size proportional to probability
            ),
            row=1, col=1
        )
    
    # Plot 2: Confidence over time (entropy)
    entropies = []
    for t in range(seq_len):
        prob_dist = predictions[0, t]
        # Avoid log(0) by adding small epsilon
        prob_dist = prob_dist + 1e-10
        entropy = -np.sum(prob_dist * np.log(prob_dist))
        entropies.append(entropy)
    
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=entropies,
            mode='lines+markers',
            name='Prediction Entropy',
            line=dict(color='red', width=3)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Token", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_yaxes(title_text="Entropy (Uncertainty)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    
    return fig


def compare_architectures(models, input_sequence):
    """Compare different RNN architectures on the same sequence."""
    comparison_results = []
    
    for name, model in models.items():
        try:
            results = run_rnn_forward(model, input_sequence)
            
            # Calculate some metrics
            hidden_states = results['hidden_states']
            predictions = results['predictions']
            
            # Hidden state statistics
            hidden_mean = np.mean(np.abs(hidden_states))
            hidden_std = np.std(hidden_states)
            
            # Prediction confidence (1 - entropy)
            avg_entropy = np.mean([
                -np.sum(predictions[0, t] * np.log(predictions[0, t] + 1e-10))
                for t in range(predictions.shape[1])
            ])
            confidence = 1 / (1 + avg_entropy)  # Normalize
            
            comparison_results.append({
                'Architecture': name,
                'Inference Time (ms)': results['inference_time'] * 1000,
                'Hidden State Mean': hidden_mean,
                'Hidden State Std': hidden_std,
                'Prediction Confidence': confidence,
                'Bidirectional': 'Bidirectional' in name,
                'RNN Type': name.split()[-1] if 'Bidirectional' not in name else name.split()[-2]
            })
            
        except Exception as e:
            st.error(f"Error with {name}: {e}")
    
    return comparison_results


def main():
    """Main Streamlit application."""
    st.title("🔄 RNN Layers Interactive Demo")
    st.markdown("**Explore Recurrent Neural Network layers with real-time sequence analysis**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        models = create_sample_models()
        selected_model = st.selectbox(
            "Choose RNN Architecture:",
            list(models.keys()),
            help="Select different RNN architectures to compare"
        )
        
        # Sequence selection
        st.subheader("📊 Input Sequence")
        sample_sequences = generate_sample_sequences()
        
        sequence_type = st.selectbox(
            "Sequence Type:",
            list(sample_sequences.keys()),
            help="Choose different sequence patterns"
        )
        
        if sequence_type == "Custom":
            custom_sequence = st.text_input(
                "Custom Sequence (comma-separated):",
                "1,2,3,4,5"
            )
            try:
                input_sequence = [int(x.strip()) for x in custom_sequence.split(',')]
            except:
                input_sequence = [1, 2, 3, 4, 5]
        else:
            input_sequence = sample_sequences[sequence_type]
        
        # Sequence length adjustment
        max_length = st.slider(
            "Sequence Length:",
            min_value=5, max_value=20, value=min(10, len(input_sequence)),
            help="Adjust the length of the sequence to process"
        )
        
        input_sequence = input_sequence[:max_length]
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        show_hidden_states = st.checkbox("Show Hidden States", value=True)
        show_predictions = st.checkbox("Show Predictions", value=True)
        show_architecture_details = st.checkbox("Architecture Details", value=False)
        
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Sequence Information")
        
        # Display input sequence
        st.write("**Input Sequence:**")
        st.write(input_sequence)
        
        # Sequence statistics
        st.write("**Statistics:**")
        st.write(f"Length: {len(input_sequence)}")
        st.write(f"Range: [{min(input_sequence)}, {max(input_sequence)}]")
        st.write(f"Mean: {np.mean(input_sequence):.2f}")
        st.write(f"Std: {np.std(input_sequence):.2f}")
        
        # Architecture details
        if show_architecture_details:
            st.subheader("🏗️ Architecture Details")
            model = models[selected_model]
            
            for component_name, component in model.items():
                with st.expander(f"{component_name.title()}"):
                    st.write(f"**Type:** {component.__class__.__name__}")
                    if hasattr(component, 'input_size'):
                        st.write(f"**Input Size:** {component.input_size}")
                    if hasattr(component, 'hidden_size'):
                        st.write(f"**Hidden Size:** {component.hidden_size}")
                    if hasattr(component, 'num_layers'):
                        st.write(f"**Layers:** {component.num_layers}")
                    if hasattr(component, 'bidirectional'):
                        st.write(f"**Bidirectional:** {component.bidirectional}")
        
        # Pattern analysis
        st.subheader("📈 Pattern Analysis")
        
        # Simple pattern detection
        diffs = np.diff(input_sequence)
        if np.all(diffs == diffs[0]):
            st.success(f"🔢 Arithmetic sequence (diff: {diffs[0]})")
        elif len(set(input_sequence)) == 1:
            st.info("🔄 Constant sequence")
        elif input_sequence == input_sequence[::-1]:
            st.info("🪞 Palindrome sequence")
        else:
            st.write("🎲 Complex/Random pattern")
    
    with col2:
        st.subheader("🧠 RNN Analysis Results")
        
        # Run inference
        model = models[selected_model]
        results = run_rnn_forward(model, input_sequence, return_states=True)
        
        # Performance metrics
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        
        with col_perf1:
            st.metric("Processing Time", f"{results['inference_time']*1000:.2f}ms")
        
        with col_perf2:
            hidden_dim = results['hidden_states'].shape[-1]
            st.metric("Hidden Dimension", f"{hidden_dim}")
        
        with col_perf3:
            seq_len = results['hidden_states'].shape[1]
            st.metric("Sequence Length", f"{seq_len}")
        
        # Hidden states visualization
        if show_hidden_states:
            st.write("**Hidden State Evolution:**")
            
            hidden_fig = visualize_hidden_states(
                results['hidden_states'],
                f"{selected_model} Hidden States"
            )
            st.plotly_chart(hidden_fig, use_container_width=True)
            
            # Hidden state statistics
            hidden_stats = results['hidden_states'][0]  # First batch
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Mean Activation", f"{np.mean(hidden_stats):.3f}")
            with col_stat2:
                st.metric("Std Activation", f"{np.std(hidden_stats):.3f}")
            with col_stat3:
                st.metric("Max Activation", f"{np.max(np.abs(hidden_stats)):.3f}")
        
        # Predictions visualization
        if show_predictions:
            st.write("**Next Token Predictions:**")
            
            pred_fig = visualize_prediction_probabilities(
                results['predictions'],
                input_sequence
            )
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Show actual next token predictions
            st.write("**Top Predictions for Each Position:**")
            predictions = results['predictions'][0]  # First batch
            
            for t in range(min(5, len(input_sequence))):  # Show first 5 positions
                top_3_indices = np.argsort(predictions[t])[-3:][::-1]
                top_3_probs = predictions[t][top_3_indices]
                
                with st.expander(f"Position {t+1} (after token {input_sequence[t]})"):
                    for i, (token_id, prob) in enumerate(zip(top_3_indices, top_3_probs)):
                        st.write(f"{i+1}. Token {token_id}: {prob:.3f}")
    
    # Architecture comparison section
    st.subheader("📊 Architecture Comparison")
    
    if st.button("🔄 Compare All RNN Architectures"):
        with st.spinner("Comparing architectures..."):
            comparison_data = compare_architectures(models, input_sequence)
        
        # Display comparison table
        st.write("**Performance Comparison:**")
        st.dataframe(comparison_data, use_container_width=True)
        
        # Visualizations
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_time = px.bar(
                comparison_data,
                x='Architecture',
                y='Inference Time (ms)',
                color='RNN Type',
                title="Inference Time by Architecture",
                text='Inference Time (ms)'
            )
            fig_time.update_traces(texttemplate='%{text:.1f}ms', textposition='outside')
            fig_time.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col_chart2:
            fig_confidence = px.scatter(
                comparison_data,
                x='Hidden State Mean',
                y='Prediction Confidence',
                color='RNN Type',
                size='Inference Time (ms)',
                hover_data=['Architecture'],
                title="Hidden State Activity vs Prediction Confidence"
            )
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Summary insights
        st.write("**🔍 Insights:**")
        fastest = min(comparison_data, key=lambda x: x['Inference Time (ms)'])
        most_confident = max(comparison_data, key=lambda x: x['Prediction Confidence'])
        
        col_insight1, col_insight2 = st.columns(2)
        with col_insight1:
            st.success(f"⚡ **Fastest:** {fastest['Architecture']} ({fastest['Inference Time (ms)']:.1f}ms)")
        with col_insight2:
            st.success(f"🎯 **Most Confident:** {most_confident['Architecture']} ({most_confident['Prediction Confidence']:.3f})")
    
    # Educational content
    with st.expander("📚 Learn About RNN Architectures"):
        st.markdown("""
        **Recurrent Neural Networks (RNNs) are designed for sequential data:**
        
        **🔄 Basic RNN:**
        - Simple recurrent connection
        - Good for short sequences
        - Suffers from vanishing gradient problem
        
        **🧠 LSTM (Long Short-Term Memory):**
        - Uses gates (forget, input, output) to control information flow
        - Maintains separate cell state and hidden state
        - Better at learning long-term dependencies
        
        **⚡ GRU (Gated Recurrent Unit):**
        - Simplified version of LSTM
        - Uses reset and update gates
        - Fewer parameters, often faster training
        
        **🔄 Bidirectional Processing:**
        - Processes sequence in both directions
        - Captures future context for each position
        - Doubles the hidden dimension
        - Cannot be used for real-time generation
        
        **📊 Hidden States:**
        - Internal memory of the network
        - Encodes information about the sequence so far
        - Visualization shows which neurons activate for different patterns
        """)
    
    # Sequence generation demo
    with st.expander("🎯 Interactive Sequence Generation"):
        st.write("**Generate Next Tokens:**")
        
        generation_model = st.selectbox(
            "Generator Model:",
            list(models.keys()),
            key="generation_model"
        )
        
        seed_sequence = st.text_input(
            "Seed Sequence (comma-separated):",
            value=",".join(map(str, input_sequence[:3]))
        )
        
        num_generate = st.slider("Tokens to Generate:", 1, 10, 3)
        
        if st.button("🎲 Generate Sequence"):
            try:
                seed = [int(x.strip()) for x in seed_sequence.split(',')]
                current_sequence = seed.copy()
                
                gen_model = models[generation_model]
                
                for _ in range(num_generate):
                    # Get predictions for current sequence
                    results = run_rnn_forward(gen_model, current_sequence)
                    predictions = results['predictions'][0, -1]  # Last position predictions
                    
                    # Sample next token (greedy)
                    next_token = np.argmax(predictions)
                    current_sequence.append(int(next_token))
                
                st.success(f"**Generated Sequence:** {current_sequence}")
                st.write(f"**Original:** {seed}")
                st.write(f"**Generated:** {current_sequence[len(seed):]}")
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("🔄 **RNN Layers Demo** - Powered by Neural Architecture Framework")


if __name__ == "__main__":
    main()