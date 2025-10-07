# KVCache Chatbot - Cross-platform Makefile
# Usage: make [target]

.PHONY: help install backend frontend all start stop clean logs test

# Default target (following Unix conventions)
all: start

# Help target
help:
	@echo "Available targets: install backend frontend all start stop clean logs test"

# Install dependencies
install:
	@uv sync && mkdir -p logs src/api/uploads

# Start backend only
backend:
	@cd src/api && python main.py

# Start frontend only
frontend:
	@cd src/web && python app.py --backend-port 3023

# Start both services (with proper sequencing)
# Both 'all' and 'start' do the same thing, following Unix conventions
all start:
	@cd src/api && python main.py &
	@cd src/web && python app.py --backend-port 3023

# Stop all servers (kill processes on specified ports)
stop:
	@lsof -ti:3023 | xargs kill -9 2>/dev/null; lsof -ti:7860 | xargs kill -9 2>/dev/null

# Clean up
clean:
	@rm -f logs/*.log && rm -rf src/api/uploads/*

# Show logs
logs:
	@tail -f logs/backend.log

# Health check
test:
	@curl -s http://0.0.0.0:3023/health && curl -s http://0.0.0.0:7860 > /dev/null
