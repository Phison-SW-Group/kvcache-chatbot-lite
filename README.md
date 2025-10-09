# KVCache Chatbot

A modern multi-turn conversation chatbot with document upload support and KV cache optimization, featuring a clean separation between frontend (Gradio) and backend (FastAPI).

## ✨ Key Features

- 💬 **Multi-turn Conversations**: Maintains conversation history for context-aware responses
- 📄 **Document Management**: Upload and select documents independently from chat sessions
- ⚡ **Streaming Responses**: Real-time streaming for better user experience
- 🚀 **KV Cache Integration**: Optimized document processing with pre-caching
- 🔧 **Model Management**: Start/stop local model servers with cache control
- 🔌 **Clean Architecture**: RESTful API with frontend-backend separation

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (recommended: Python 3.10+)
- **uv** package manager (recommended) or pip
- **Git** (for cloning the repository)

### Installation & Setup

**Option A: Using Make (Recommended)**
```bash
git clone <repository-url>
cd chatbot-for-kvcache-demo
make start
```

**Option B: Using npm**
```bash
git clone <repository-url>
cd chatbot-for-kvcache-demo
npm start
```

**Option C: Manual setup**
```bash
git clone <repository-url>
cd chatbot-for-kvcache-demo
uv sync  # or pip install dependencies
# Terminal 1: cd src/api && python main.py
# Terminal 2: cd src/web && python app.py --backend-port 8000
```

### Access the Application
- **Frontend**: http://localhost:7860
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📚 Documentation

### 🛠️ Installation Guide
For detailed installation instructions, dependency management, and configuration:
**[📖 GETTING_STARTED.md](docs/GETTING_STARTED.md)**

### 🎯 Usage Guide
For comprehensive usage instructions, workflows, and troubleshooting:
**[📖 QUICK_START.md](docs/QUICK_START.md)**

### 🏗️ Architecture Overview
For system architecture, data flow, and technical details:
**[📖 ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## 🎮 Basic Usage

1. **Upload Documents**: Use the left sidebar to upload `.txt` files
2. **Select Document**: Choose from dropdown to use document context
3. **Chat**: Type messages and get responses with document context
4. **Cache Optimization**: Use "Cache" button for faster document processing
5. **Model Management**: Start/stop model servers from the interface

## 🔧 Configuration

### LLM Configuration (Optional)
Create a `.env` file in `src/api/`:
```env
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

### Model Server Configuration
```env
LLM_SERVER_EXE=path/to/your/model/server
LLM_SERVER_MODEL_NAME_OR_PATH=path/to/model
LLM_SERVER_CACHE=path/to/cache/directory
LLM_SERVER_LOG=path/to/log/directory
```

**Note**: Without API configuration, the system uses mock responses for testing.

## 🛠️ Development

### Project Structure
```
src/
├── api/              # FastAPI backend
│   ├── main.py       # Main application
│   ├── config.py     # Configuration
│   ├── models.py     # Data models
│   ├── routers/      # API routes
│   │   ├── document.py    # Document management
│   │   ├── session.py     # Chat sessions
│   │   ├── model.py       # Model management
│   │   └── logs.py        # Log management
│   └── services/     # Business logic
│       ├── document_manager.py  # Document storage
│       ├── session_service.py   # Session management
│       ├── llm_service.py       # LLM integration
│       └── model.py             # Model server management
└── web/              # Gradio frontend
    └── app.py        # Gradio interface
```

### Available Commands
```bash
make start          # Start both frontend and backend
make stop           # Stop all services
make install        # Install dependencies
make test           # Health check
make clean          # Clean logs and uploads
```

## 🔗 API Endpoints

### Document Management
- `POST /api/v1/documents/upload` - Upload document
- `POST /api/v1/documents/upload_and_cache` - Upload and cache document
- `GET /api/v1/documents/list` - List all documents
- `DELETE /api/v1/documents/{doc_id}` - Delete document
- `POST /api/v1/documents/cache/{doc_id}` - Cache existing document

### Chat Sessions
- `GET /api/v1/session/{session_id}` - Get session info
- `POST /api/v1/session/{session_id}/messages/stream` - Send message (streaming)
- `GET /api/v1/session/{session_id}/messages` - Get chat history
- `DELETE /api/v1/session/{session_id}` - Delete session

### Model Management
- `POST /api/v1/model/up/reset` - Start model with reset
- `POST /api/v1/model/up/without_reset` - Start model without reset
- `POST /api/v1/model/down` - Stop model
- `GET /api/v1/model/status` - Get model status

## 🚨 Troubleshooting

### Common Issues
1. **Port already in use**: `make stop` or `npm run stop`
2. **Dependencies not installed**: `make install` or `npm run install-deps`
3. **Permission errors on Windows**: Run as Administrator or use WSL

### Health Check
```bash
make test
# or
npm test
```

## 🏗️ Technology Stack

**Backend:**
- FastAPI - Modern Python web framework
- Pydantic - Data validation and serialization
- Uvicorn - ASGI server
- httpx - Async HTTP client

**Frontend:**
- Gradio - ML/AI web interface framework
- httpx - HTTP client for API communication

**Development:**
- uv - Fast Python package manager
- Make - Cross-platform build automation
- npm - Node.js package management

## 📄 License

MIT

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Need help?** Check out our detailed documentation:
- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Installation and setup
- **[QUICK_START.md](docs/QUICK_START.md)** - Usage guide and workflows
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture overview
