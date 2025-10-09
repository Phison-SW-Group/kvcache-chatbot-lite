# KVCache Chatbot Architecture

## System Overview

The KVCache Chatbot is a modern web application with a clean separation between frontend and backend, featuring advanced document management and KV cache optimization.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Gradio)                        │
│                         Port: 7860                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐         ┌─────────────────────────────┐ │
│  │ Document Upload   │         │     Chat Interface          │ │
│  │ - File upload     │         │  ┌─────────────────────┐   │ │
│  │ - Upload button   │         │  │   Chat messages     │   │ │
│  │ - Cache button    │         │  └─────────────────────┘   │ │
│  └───────────────────┘         │  ┌─────────────────────┐   │ │
│                                 │  │   Message input     │   │ │
│  ┌───────────────────┐         │  └─────────────────────┘   │ │
│  │ Document Selector │         └─────────────────────────────┘ │
│  │ - Dropdown list   │                                          │
│  │ - Refresh button  │         ┌─────────────────────────────┐ │
│  └───────────────────┘         │     Model Management         │ │
│                                 │  ┌─────────────────────┐   │ │
│  ┌───────────────────┐         │  │   Model Status      │   │ │
│  │ Model Controls    │         │  └─────────────────────┘   │ │
│  │ - Restart button  │         │  ┌─────────────────────┐   │ │
│  │ - Stop button     │         │  │   Deployment Logs   │   │ │
│  └───────────────────┘         │  └─────────────────────┘   │ │
│                                 └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Backend (FastAPI)                       │
│                         Port: 8000                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐         ┌─────────────────────────────┐ │
│  │ Document Manager   │         │     Session Manager        │ │
│  │ - Independent docs │         │  ┌─────────────────────┐   │ │
│  │ - KV Cache ops     │         │  │   Chat sessions     │   │ │
│  │ - File processing  │         │  └─────────────────────┘   │ │
│  └───────────────────┘         │  ┌─────────────────────┐   │ │
│                                 │  │   Message history   │   │ │
│  ┌───────────────────┐         │  └─────────────────────┘   │ │
│  │ Model Server       │         └─────────────────────────────┘ │
│  │ - Start/stop       │                                          │
│  │ - Status monitoring│         ┌─────────────────────────────┐ │
│  │ - Log management   │         │     LLM Service            │ │
│  └───────────────────┘         │  ┌─────────────────────┐   │ │
│                                 │  │   OpenAI API        │   │ │
│  ┌───────────────────┐         │  └─────────────────────┘   │ │
│  │ Document Service   │         │  ┌─────────────────────┐   │ │
│  │ - Text extraction  │         │  │   Local Model       │   │ │
│  │ - Format support   │         │  └─────────────────────┘   │ │
│  └───────────────────┘         └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         External Services                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐         ┌─────────────────────────────┐ │
│  │ File System       │         │     Model Server           │ │
│  │ - Document storage│         │  ┌─────────────────────┐   │ │
│  │ - Upload directory│         │  │   KV Cache           │   │ │
│  └───────────────────┘         │  └─────────────────────┘   │ │
│                                 │  ┌─────────────────────┐   │ │
│  ┌───────────────────┐         │  │   Model Inference    │   │ │
│  │ OpenAI API        │         │  └─────────────────────┘   │ │
│  │ - GPT models      │         └─────────────────────────────┘ │
│  │ - Streaming       │                                          │
│  └───────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```
│                                 │  Model Controls (Placeholder)│ │
│                                 └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │ HTTP/SSE
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend API (FastAPI)                       │
│                         Port: 8000                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Routers                            │  │
│  │  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │  │ Session Router │  │Document Router│ │Upload Router│  │  │
│  │  │                │  │               │  │  (Legacy)   │  │  │
│  │  │ - GET session  │  │ - POST upload │  └─────────────┘  │  │
│  │  │ - POST message │  │ - GET list    │                   │  │
│  │  │ - POST stream  │  │ - GET info    │                   │  │
│  │  │ - DELETE       │  │ - DELETE doc  │                   │  │
│  │  └────────────────┘  └──────────────┘                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Services Layer                         │  │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────┐ │  │
│  │  │ Session Manager │  │Document Manager   │  │LLM      │ │  │
│  │  │                 │  │                   │  │Service  │ │  │
│  │  │ - Store sessions│  │- Store documents  │  │         │ │  │
│  │  │ - Manage history│  │- Unique IDs       │  │- OpenAI │ │  │
│  │  │ - Session expiry│  │- Metadata tracking│  │- Gemini │ │  │
│  │  └─────────────────┘  └──────────────────┘  └─────────┘ │  │
│  │                                                           │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │           Document Service                        │   │  │
│  │  │  - Process documents (txt, pdf, docx, md)        │   │  │
│  │  │  - Extract text content                           │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  File Storage  │
                  │  (uploads/)    │
                  └────────────────┘
```

## Data Flow

### 1. Document Upload Flow
```
User → Upload File → Frontend
                     ↓
            POST /documents/upload
                     ↓
         Document Router → Document Service (extract text)
                     ↓
         Document Manager (store with UUID)
                     ↓
         Return doc_id to Frontend
                     ↓
         Update dropdown list
```

### 2. Document Caching Flow
```
User → Click "Cache" button
     → POST /api/v1/documents/cache/{doc_id}
     → Backend retrieves document content
     → Prepare system message with document
     → Run inference with max_tokens=2
     → KV Cache populated in model server
     → Cache ready for prefix matching
```

### 3. Chat with Document Flow
```
User → Select Document from Dropdown
     → Type message
     → Send
                     ↓
         Frontend gets doc_id from dropdown
                     ↓
         POST /session/{id}/messages/stream
         { message: "...", document_id: "..." }
                     ↓
         Session Router:
           1. Get document content by doc_id
           2. Insert as system message:
              messages = [
                {role: "system", content: "document-content"},
                ...chat history...
              ]
           3. Send to LLM Service
                     ↓
         LLM Service:
           1. Check if model server running
           2. Use KV Cache for faster inference
           3. Stream response back
                     ↓
         Stream response back to Frontend
```

### 4. Model Management Flow
```
User → Click "Restart Model"
     → POST /api/v1/model/up/reset
     → Model Server:
           1. Stop existing process
           2. Clear KV Cache
           3. Start new model server
           4. Update status
     → Return status to frontend
```

### 5. Session Management Flow
```
New Chat Session → Generate UUID
                 → Store in Session Manager
                 → Maintain message history
                 → Auto-expire after timeout
```

## Key Components

### Backend

#### 1. Document Manager (`services/document_manager.py`)
- Manages independent documents with unique IDs
- Stores document metadata and content in memory
- Provides CRUD operations for documents

#### 2. Session Manager (`services/session_service.py`)
- Manages chat sessions with unique IDs
- Maintains conversation history per session
- Auto-cleanup of expired sessions

#### 3. LLM Service (`services/llm_service.py`)
- Interfaces with LLM APIs (OpenAI, Gemini, etc.)
- Handles streaming responses
- Manages API configuration

#### 4. Model Server (`services/model.py`)
- Manages local model server lifecycle
- Handles KV cache operations
- Provides start/stop/restart functionality
- Monitors model server status and logs

#### 5. Document Service (`services/document_service.py`)
- Processes different document formats
- Extracts text content
- Extensible for multiple file types

#### 6. Model Log Service (`services/model_log.py`)
- Manages model server logs
- Provides log filtering and retrieval
- Tracks model operations and errors

### Frontend

#### ChatbotClient Class
- Handles all API communication
- Manages session lifecycle
- Provides document upload/list/delete operations
- Streams chat responses

#### UI Components
- Document upload area with file validation
- Document dropdown selector with refresh
- Chat interface with streaming responses
- Model management controls (start/stop/restart)
- Model status monitoring
- Deployment logs viewer

## API Endpoints

### Documents
- `POST /api/v1/documents/upload` - Upload new document
- `POST /api/v1/documents/upload_and_cache` - Upload and cache document
- `GET /api/v1/documents/list` - List all documents
- `GET /api/v1/documents/{doc_id}` - Get document info
- `DELETE /api/v1/documents/{doc_id}` - Delete document
- `POST /api/v1/documents/cache/{doc_id}` - Cache existing document

### Sessions
- `GET /api/v1/session/{session_id}` - Get session info
- `GET /api/v1/session/{session_id}/messages` - Get chat history
- `POST /api/v1/session/{session_id}/messages` - Send message (non-streaming)
- `POST /api/v1/session/{session_id}/messages/stream` - Send message (streaming)
- `DELETE /api/v1/session/{session_id}` - Delete session

### Model Management
- `POST /api/v1/model/up/without_reset` - Start model without reset
- `POST /api/v1/model/up/reset` - Start model with reset
- `POST /api/v1/model/down` - Stop model
- `GET /api/v1/model/status` - Get model status

### Logs
- `GET /api/v1/logs/current` - Get current model logs
- `GET /api/v1/logs/recent` - Get recent logs
- `GET /api/v1/logs/sessions` - List log sessions

## KV Cache Optimization

### How KV Cache Works
The KV Cache is a memory optimization technique used by transformer-based language models to store computed key-value pairs from previous tokens. This allows for faster inference when processing sequences with common prefixes.

### Implementation in KVCache Chatbot

1. **Document Pre-caching**:
   - When a document is uploaded with "Cache" button
   - Document content is sent as a system message
   - Model runs inference with `max_tokens=2`
   - KV Cache is populated with document prefix
   - Subsequent queries benefit from cached computation

2. **Cache Persistence**:
   - KV Cache remains in memory during model server lifetime
   - Cache survives across multiple chat sessions
   - Cache is cleared when model server restarts with reset

3. **Performance Benefits**:
   - Faster response times for document-based queries
   - Reduced computational overhead
   - Better user experience with streaming responses

### Cache Management
- **Cache Population**: Automatic when using "Cache" button
- **Cache Clearing**: Manual via "Restart Model" with reset
- **Cache Monitoring**: Available through model logs
- **Cache Persistence**: Maintained until model server restart

## Storage

### In-Memory Storage
- **Sessions**: Stored in `SessionManager._sessions` dict
- **Documents**: Stored in `DocumentManager._documents` dict
- **Files**: Saved to `uploads/` directory

### Future Enhancements
- Redis for session storage (distributed systems)
- Database for document metadata (PostgreSQL/MongoDB)
- S3/Object storage for files (scalability)
- Persistent KV Cache storage
- Multi-model support
- Document versioning
- Advanced caching strategies

## Security Considerations

### Current Implementation
- CORS enabled for all origins (development only)
- File size limits enforced
- File type validation
- No authentication (local use)

### Production Recommendations
- Add user authentication (JWT tokens)
- Implement rate limiting
- Restrict CORS to specific origins
- Add input sanitization
- Implement document access control
- Add encryption for sensitive documents
- Use HTTPS only

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework for APIs
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server for production deployment
- **httpx**: Async HTTP client for external API calls
- **aiofiles**: Async file operations

### Frontend
- **Gradio**: ML/AI web interface framework
- **httpx**: HTTP client for API communication
- **Streaming**: Server-Sent Events (SSE) for real-time responses

### Model Integration
- **OpenAI API**: Cloud-based LLM services
- **Local Model Server**: Custom model deployment
- **KV Cache**: Memory optimization for transformer models

### Development Tools
- **uv**: Fast Python package manager
- **Make**: Cross-platform build automation
- **npm**: Node.js package management
- **Git**: Version control

## Performance Considerations

### Current Optimizations
- KV Cache for document prefix matching
- Streaming responses for better UX
- In-memory session/document storage
- Async/await throughout the stack

### Scalability Limitations
- Single-process model server
- In-memory storage (not distributed)
- No load balancing
- Limited concurrent users

### Recommended Improvements
- Horizontal scaling with multiple model servers
- Redis for distributed session storage
- Database for persistent document metadata
- CDN for static file serving
- Load balancer for high availability

