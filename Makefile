# KVCache Chatbot - Cross-platform Makefile
# Usage: make [target]

.PHONY: help install backend frontend all start stop clean logs test

# Default target (following Unix conventions)
all: start

# Help target
help:
	@echo "KVCache Chatbot - Available Commands:"
	@echo "======================================"
	@echo "  make install     - Install all dependencies"
	@echo "  make backend     - Start backend server only"
	@echo "  make frontend    - Start frontend server only"
	@echo "  make all         - Start both backend and frontend (default)"
	@echo "  make start       - Start both backend and frontend (alias for all)"
	@echo "  make stop        - Stop all servers"
	@echo "  make clean       - Clean up logs and temporary files"
	@echo "  make logs        - Show backend logs"
	@echo "  make test        - Run basic health checks"
	@echo ""

# Install dependencies
install:
	@echo "Installing backend dependencies..."
	@cd src/api && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	@cd src/web && pip install -r requirements.txt
	@echo "Creating necessary directories..."
	@mkdir -p logs
	@mkdir -p src/api/uploads
	@echo "✅ Installation complete!"

# Start backend only
backend:
	@echo "Starting backend server..."
	@cd src/api && python main.py

# Start frontend only
frontend:
	@echo "Starting frontend server..."
	@echo "Make sure backend is running at http://localhost:8000"
	@cd src/web && python app.py

# Start both services (with proper sequencing)
# Both 'all' and 'start' do the same thing, following Unix conventions
all start:
	@echo "Starting KVCache Chatbot..."
	@echo "========================================="
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:7860"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "========================================="
	@echo ""
	@echo "Press Ctrl+C to stop servers"
	@echo ""
	@echo "Creating necessary directories..."
	@mkdir -p logs
	@mkdir -p src/api/uploads
	@echo "Starting backend in background..."
	@cd src/api && python main.py > ../../logs/backend.log 2>&1 &
	@echo "Waiting for backend to be ready..."
	@timeout 30 bash -c 'until curl -s http://localhost:8000/health > /dev/null; do sleep 1; done' || echo "Backend startup timeout"
	@echo "Starting frontend..."
	@cd src/web && python app.py

# Stop all servers (kill processes on specified ports)
stop:
	@echo "Stopping servers..."
	@echo "Stopping backend (port 8000)..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "No backend process found"
	@echo "Stopping frontend (port 7860)..."
	@lsof -ti:7860 | xargs kill -9 2>/dev/null || echo "No frontend process found"
	@echo "✅ Servers stopped"

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -f logs/*.log
	@rm -rf src/api/uploads/*
	@echo "✅ Cleanup complete"

# Show logs
logs:
	@echo "Backend logs:"
	@echo "============="
	@tail -f logs/backend.log

# Health check
test:
	@echo "Running health checks..."
	@echo "Checking backend health..."
	@curl -s http://localhost:8000/health && echo "✅ Backend is healthy" || echo "❌ Backend is not responding"
	@echo "Checking frontend accessibility..."
	@curl -s http://localhost:7860 > /dev/null && echo "✅ Frontend is accessible" || echo "❌ Frontend is not accessible"
