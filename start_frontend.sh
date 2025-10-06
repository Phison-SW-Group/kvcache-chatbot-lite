#!/bin/bash

# Start Gradio frontend

echo "Starting KVCache Chatbot Frontend..."
echo "===================================="

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

# Navigate to web directory
cd "$(dirname "$0")/src/web"

# Check if dependencies are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
fi

# Start the frontend
echo ""
echo "Starting Gradio interface on http://localhost:7860"
echo "Make sure the backend API is running at http://localhost:8000"
echo ""
python app.py

