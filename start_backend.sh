#!/bin/bash

# Start backend API server

echo "Starting KVCache Chatbot Backend..."
echo "==================================="

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

# Navigate to API directory
cd "$(dirname "$0")/src/api"

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
fi

# Create uploads directory
mkdir -p uploads

# Start the server
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
echo ""
python main.py

