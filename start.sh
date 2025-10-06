#!/bin/bash

# KVCache Chatbot Unified Startup Script
# This script starts both backend and frontend

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "   KVCache Chatbot Startup"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Activating .venv..."
    source .venv/bin/activate
fi

echo "✓ Using Python: $(which python)"
echo ""

# ========================================
# Backend Setup
# ========================================
echo "📦 Setting up backend..."
cd src/api

# Check and install backend dependencies
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing backend dependencies..."
    uv pip install -r requirements.txt
else
    echo "✓ Backend dependencies already installed"
fi

# Create uploads directory
mkdir -p uploads

echo ""
echo "🚀 Starting backend server..."
# Start backend in background
python main.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"
echo "  Log: logs/backend.log"

# Wait for backend to be ready
echo "  Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Backend failed to start. Check logs/backend.log"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

cd "$PROJECT_ROOT"

# ========================================
# Frontend Setup
# ========================================
echo ""
echo "📦 Setting up frontend..."
cd src/web

# Check and install frontend dependencies
if ! python -c "import gradio" 2>/dev/null; then
    echo "Installing frontend dependencies..."
    uv pip install -r requirements.txt
else
    echo "✓ Frontend dependencies already installed"
fi

echo ""
echo "🚀 Starting frontend server..."
echo ""
echo "========================================="
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:7860"
echo "  API Docs: http://localhost:8000/docs"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Trap to cleanup on exit
trap 'echo ""; echo "🛑 Stopping servers..."; kill $BACKEND_PID 2>/dev/null || true; echo "✓ Servers stopped"; exit 0' INT TERM

# Start frontend in foreground
python app.py

# Cleanup (if we reach here)
kill $BACKEND_PID 2>/dev/null || true

