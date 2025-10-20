# Getting Started

## Prerequisites

- **Python 3.9+** (recommended: Python 3.10+)
- **uv** package manager (recommended) or pip
- **Git** (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd chatbot-for-kvcache-demo
```

### 2. Install Dependencies

**Option A: Using uv (Recommended)**
```bash
uv sync
```

**Option B: Using pip**
```bash
# Install backend dependencies
cd src/api
pip install -r requirements.txt

# Install frontend dependencies
cd ../web
pip install -r requirements.txt
```

**Option C: Using uv with pip**
```bash
# Install backend dependencies
cd src/api
uv pip install -r requirements.txt

# Install frontend dependencies
cd ../web
uv pip install -r requirements.txt
```

## Quick Start

### Start the Application

**Option A: Using Make (Cross-platform)**
```bash
make start
```

**Option B: Using npm scripts**
```bash
npm start
```

**Option C: Manual start**
```bash
# Terminal 1 - Backend
cd src/api && python main.py

# Terminal 2 - Frontend
cd src/web && python app.py --backend-port 8000
```

### Access the Application

- **Frontend**: http://localhost:7860
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Configuration (Optional)

### LLM Configuration

Create a `.env` file in the `src/api/` directory:

```env
# LLM Settings - OpenAI Compatible API
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

**Note**: Without API configuration, the system uses mock responses for testing.

### Model Server Configuration

For local model deployment, configure these settings in `.env`:

```env
# Model Server Settings
LLM_SERVER_EXE=path/to/your/model/server
LLM_SERVER_MODEL_NAME_OR_PATH=path/to/model
LLM_SERVER_CACHE=path/to/cache/directory
LLM_SERVER_LOG=path/to/log/directory
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Stop existing processes
   make stop
   # or
   npm run stop
   ```

2. **Dependencies not installed**
   ```bash
   make install
   # or
   npm run install-deps
   ```

3. **Permission errors on Windows**
   - Run terminal as Administrator
   - Or use WSL (Windows Subsystem for Linux)

### Health Check

```bash
make test
# or
npm test
```

## Next Steps

- See [QUICK_START.md](QUICK_START.md) for usage instructions
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system overview
- Check the API documentation at http://localhost:8000/docs
