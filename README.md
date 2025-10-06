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

**一鍵啟動前後端：**

```bash
source .venv/bin/activate
./start.sh
```

這個腳本會：
- ✅ 自動檢查並安裝依賴
- ✅ 後台啟動後端 (port 8000)
- ✅ 前台啟動前端 (port 7860)
- ✅ 按 `Ctrl+C` 會同時停止兩個服務

**停止服務：**

```bash
./stop.sh
```

或直接在運行的終端按 `Ctrl+C`

### Manual Start (Alternative)

如果你想分別啟動前後端：

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

- **前端界面**: http://localhost:7860
- **後端 API**: http://localhost:8000
- **API 文檔**: http://localhost:8000/docs
- **後端日誌**: `logs/backend.log` (統一啟動時)

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

### LLM API 配置（可選）

在 `src/api/` 目錄下創建 `.env` 文件：

```env
# LLM Settings - OpenAI Compatible API
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-openai-api-key-here
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

**注意：** 如果不配置 API key，系統會使用 mock 響應進行測試。

詳細配置說明請參考：[LLM_SETUP.md](LLM_SETUP.md)

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

系統已內建 OpenAI compatible API 支持！

**快速設定：**

1. 在 `src/api/` 創建 `.env` 文件
2. 添加以下配置：
   ```env
   LLM_MODEL=gpt-3.5-turbo
   LLM_API_KEY=your-openai-api-key-here
   ```
3. 重啟後端即可

**獲取 API Key：**
- OpenAI: https://platform.openai.com/api-keys
- 詳細說明：查看 [LLM_SETUP.md](LLM_SETUP.md)

**不配置 API key？** 系統會使用 mock 響應，適合測試。

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
