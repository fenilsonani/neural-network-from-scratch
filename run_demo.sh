#!/bin/bash

# Neural Architecture Framework - Streamlit Demo Launcher
echo "🚀 Launching Neural Architecture Framework Demo..."
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements_demo.txt"
    exit 1
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Install demo requirements if needed
echo "📦 Installing demo requirements..."
pip install -r requirements_demo.txt

# Launch Streamlit
echo "🌟 Starting Streamlit demo..."
echo "   Navigate to: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run streamlit_demo.py