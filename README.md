# KVCache Chatbot

A multi-turn conversation chatbot with document upload support, featuring a clean separation between frontend (Gradio) and backend (FastAPI).

## Features

- 💬 **Multi-turn Conversations**: Maintains conversation history for context-aware responses
- 📄 **Document Upload**: Upload documents and ask questions about their content
- ⚡ **Streaming Responses**: Real-time streaming for better user experience
- 🔌 **Frontend-Backend Separation**: Clean REST API architecture
- 🚀 **Extensible Design**: Easy to add new document formats and LLM providers

## Architecture

```
kvcache_chatbot/
├── src/
│   ├── api/              # FastAPI backend
│   │   ├── main.py       # Main application
│   │   ├── config.py     # Configuration
│   │   ├── models.py     # Data models
│   │   ├── routers/      # API routes
│   │   │   ├── chat.py   # Chat endpoints
│   │   │   └── upload.py # File upload endpoints
│   │   └── services/     # Business logic
│   │       ├── session_service.py   # Session management
│   │       ├── document_service.py  # Document processing
│   │       └── llm_service.py       # LLM integration
│   └── web/              # Gradio frontend
│       └── app.py        # Gradio interface
└── README.md
```

## Prerequisites

- Python 3.10+
- Virtual environment activated: `source .venv/bin/activate`

## Installation

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Install Backend Dependencies

```bash
cd src/api
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd ../web
pip install -r requirements.txt
```

## Running the Application

### Quick Start (Recommended)

**One-command startup for both frontend and backend:**

```bash
source .venv/bin/activate
./start.sh
```

This script will:
- ✅ Automatically check and install dependencies
- ✅ Start backend in background (port 8000)
- ✅ Start frontend in foreground (port 7860)
- ✅ Press `Ctrl+C` to stop both services

**To stop services:**

```bash
./stop.sh
```

Or press `Ctrl+C` in the running terminal

### Manual Start (Alternative)

If you want to start frontend and backend separately:

**Terminal 1 - Backend:**
```bash
source .venv/bin/activate
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
source .venv/bin/activate
./start_frontend.sh
```

### Access Points

- **Frontend**: http://localhost:7860
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Backend Logs**: `logs/backend.log` (when using unified startup)

## Usage

1. **Open the Gradio Interface**: Navigate to http://localhost:7860
2. **Upload a Document** (optional):
   - Click "Upload Document" in the right panel
   - Select a `.txt` file
   - Click "Upload"
3. **Start Chatting**:
   - Type your message in the text box
   - Enable "Use uploaded document context" if you want answers based on the document
   - Click "Send" or press Enter
4. **Multi-turn Conversation**: The chatbot remembers previous messages in the conversation
5. **Clear Chat**: Click "Clear Chat" to start a new session

## API Endpoints

### Chat Endpoints

- `POST /api/v1/chat/message` - Send message (non-streaming)
- `POST /api/v1/chat/stream` - Send message (streaming with SSE)
- `GET /api/v1/chat/session/{session_id}` - Get session info
- `GET /api/v1/chat/history/{session_id}` - Get chat history
- `DELETE /api/v1/chat/session/{session_id}` - Delete session

### Upload Endpoints

- `POST /api/v1/upload` - Upload document
- `DELETE /api/v1/upload/{session_id}` - Delete uploaded document

## Configuration

### LLM API Configuration (Optional)

Create a `.env` file in the `src/api/` directory:

```env
# LLM Settings - OpenAI Compatible API
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-openai-api-key-here
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

**Note:** If you don't configure an API key, the system will use mock responses for testing.

For detailed configuration instructions, see: [LLM_SETUP.md](LLM_SETUP.md)

## Extending the System

### Adding New Document Formats

1. Create a processor in `src/api/services/document_service.py`:

```python
class PdfProcessor:
    async def process(self, file_path: Path) -> str:
        # PDF processing logic
        pass

# Register the processor
document_service.register_processor('.pdf', PdfProcessor())
```

2. Update `ALLOWED_EXTENSIONS` in `config.py`
3. Update file types in `src/web/app.py` (line with `file_types=[".txt"]`)

### Integrating Real LLM (OpenAI Compatible API)

The system has built-in OpenAI compatible API support!

**Quick Setup:**

1. Create a `.env` file in `src/api/`
2. Add the following configuration:
   ```env
   LLM_MODEL=gpt-3.5-turbo
   LLM_API_KEY=your-openai-api-key-here
   ```
3. Restart the backend

**Get API Key:**
- OpenAI: https://platform.openai.com/api-keys
- Detailed instructions: See [LLM_SETUP.md](LLM_SETUP.md)

**No API key?** The system will use mock responses, suitable for testing.

### Adding Database Storage

Currently uses in-memory session storage. To add database:

1. Install database driver: `pip install sqlalchemy asyncpg`
2. Create models in `src/api/database.py`
3. Update `SessionManager` in `session_service.py` to use database

## Technology Stack

**Backend:**
- FastAPI - Modern Python web framework
- Pydantic - Data validation
- Uvicorn - ASGI server

**Frontend:**
- Gradio - ML/AI web interface framework
- httpx - HTTP client

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# chatbot
# chatbot
