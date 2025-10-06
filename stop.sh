#!/bin/bash

# Stop all running chatbot servers

echo "🛑 Stopping KVCache Chatbot servers..."

# Stop backend (port 8000)
BACKEND_PID=$(lsof -ti:8000 2>/dev/null)
if [ ! -z "$BACKEND_PID" ]; then
    kill -9 $BACKEND_PID 2>/dev/null
    echo "✓ Backend stopped (port 8000)"
else
    echo "  Backend not running"
fi

# Stop frontend (port 7860)
FRONTEND_PID=$(lsof -ti:7860 2>/dev/null)
if [ ! -z "$FRONTEND_PID" ]; then
    kill -9 $FRONTEND_PID 2>/dev/null
    echo "✓ Frontend stopped (port 7860)"
else
    echo "  Frontend not running"
fi

echo ""
echo "✅ All servers stopped"

