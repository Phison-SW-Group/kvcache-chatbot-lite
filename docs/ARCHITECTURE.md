# KVCache Chatbot Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Gradio)                        │
│                         Port: 7860                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐         ┌─────────────────────────────┐ │
│  │ Document Upload   │         │     Chat Interface          │ │
│  │ - Click to upload │         │  ┌─────────────────────┐   │ │
│  │ - Upload button   │         │  │   Chat messages     │   │ │
│  └───────────────────┘         │  └─────────────────────┘   │ │
│                                 │  ┌─────────────────────┐   │ │
│  ┌───────────────────┐         │  │   Message input     │   │ │
│  │ Select Document   │         │  └─────────────────────┘   │ │
│  │ - Dropdown list   │         └─────────────────────────────┘ │
│  │ - Refresh button  │                                          │
│  └───────────────────┘         ┌─────────────────────────────┐ │
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

### 2. Chat with Document Flow
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
           3. Send to LLM
                     ↓
         Stream response back to Frontend
```

### 3. Session Management Flow
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

#### 4. Document Service (`services/document_service.py`)
- Processes different document formats
- Extracts text content
- Extensible for multiple file types

### Frontend

#### ChatbotClient Class
- Handles all API communication
- Manages session lifecycle
- Provides document upload/list/delete operations
- Streams chat responses

#### UI Components
- Document upload area
- Document dropdown selector
- Chat interface with streaming
- Model controls (placeholder)

## API Endpoints

### Documents
- `POST /api/v1/documents/upload` - Upload new document
- `GET /api/v1/documents/list` - List all documents
- `GET /api/v1/documents/{doc_id}` - Get document info
- `DELETE /api/v1/documents/{doc_id}` - Delete document

### Sessions
- `GET /api/v1/session/{session_id}` - Get session info
- `GET /api/v1/session/{session_id}/messages` - Get chat history
- `POST /api/v1/session/{session_id}/messages` - Send message (non-streaming)
- `POST /api/v1/session/{session_id}/messages/stream` - Send message (streaming)
- `DELETE /api/v1/session/{session_id}` - Delete session

## Storage

### In-Memory Storage
- **Sessions**: Stored in `SessionManager._sessions` dict
- **Documents**: Stored in `DocumentManager._documents` dict
- **Files**: Saved to `uploads/` directory

### Future Enhancements
- Redis for session storage (distributed systems)
- Database for document metadata (PostgreSQL/MongoDB)
- S3/Object storage for files (scalability)

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

