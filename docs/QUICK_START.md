# Quick Start Guide - New Document Upload Feature

## What Changed?

Documents are now **independent** from chat sessions! You can:
1. Upload documents once
2. Select any document from a dropdown menu
3. Use the same document across multiple chat sessions

## How to Use

### Step 1: Start the Application

```bash
# Start backend
./start_backend.sh

# In another terminal, start frontend
./start_frontend.sh
```

### Step 2: Upload a Document

1. Look at the left sidebar "Document Management" section
2. Click on the file upload area under "Upload Document"
3. Select a `.txt` file
4. Click the "Upload" button
5. You'll see a success message with the filename and size

### Step 3: Select a Document

1. In the "Select Document" section below, you'll see a dropdown menu
2. Click the dropdown to see all uploaded documents
3. Select the document you want to use
4. If you upload more documents, click "Refresh List" to update the dropdown

### Step 4: Chat with Document Context

1. With a document selected in the dropdown
2. Type your message in the chat input box
3. Press Enter or click Send
4. The LLM will receive your message with the document content as context

**The document content is inserted at the beginning of the system prompt, so the LLM sees:**
```
System: [document-plain-text-content]
User: [your-query]
```

### Step 5: Switch Documents

1. Simply select a different document from the dropdown
2. Your next message will use the newly selected document
3. Previous chat history remains in the session

### Step 6: Chat Without Documents

1. Select "No document selected" from the dropdown
2. Your messages will be sent without document context

## Example Workflow

```
1. Upload "product_manual.txt"
   ✅ Document 'product_manual.txt' uploaded successfully

2. Select "product_manual.txt" from dropdown

3. Ask: "What is the return policy?"
   → LLM receives: [product_manual.txt content] + "What is the return policy?"
   → LLM answers based on the document

4. Upload "faq.txt"
   ✅ Document 'faq.txt' uploaded successfully

5. Click "Refresh List" button

6. Select "faq.txt" from dropdown

7. Ask: "How do I contact support?"
   → LLM receives: [faq.txt content] + "How do I contact support?"
   → LLM answers based on the new document

8. Switch back to "product_manual.txt" to ask more questions about the product
```

## Tips

- **Multiple documents**: You can upload as many documents as you want
- **Reusable**: Documents persist across sessions (until backend restarts)
- **Easy switching**: Change documents anytime via the dropdown
- **No document mode**: Select "No document selected" for regular chat
- **Refresh**: If the dropdown doesn't show new uploads, click "Refresh List"

## Supported File Types

Currently supported:
- `.txt` (Plain text files)

Coming soon:
- `.pdf` (PDF documents)
- `.docx` (Word documents)
- `.md` (Markdown files)

## Troubleshooting

### Document not showing in dropdown
- Click the "Refresh List" button
- Check the upload status message for errors

### Upload fails
- Check file size (limit: see backend config)
- Ensure file type is `.txt`
- Make sure backend is running

### Chat not using document
- Verify a document is selected (not "No document selected")
- Check backend logs for errors

## API Changes (For Developers)

### Old API
```python
# Upload bound to session
POST /api/v1/session/{session_id}/document

# Message with boolean flag
POST /api/v1/session/{session_id}/messages
{
  "message": "...",
  "use_document": true
}
```

### New API
```python
# Upload independent
POST /api/v1/documents/upload
→ Returns: { "doc_id": "...", "filename": "...", ... }

# List documents
GET /api/v1/documents/list
→ Returns: [{ "doc_id": "...", "filename": "...", ... }]

# Message with document ID
POST /api/v1/session/{session_id}/messages
{
  "message": "...",
  "document_id": "abc-123-xyz"  # or null
}
```

## File Structure

```
src/api/
├── routers/
│   ├── document.py       # NEW: Independent document endpoints
│   ├── session.py        # UPDATED: Uses document_id
│   └── upload.py         # LEGACY: Old session-bound upload
├── services/
│   ├── document_manager.py  # NEW: Manages documents independently
│   ├── document_service.py  # Processes documents
│   ├── session_service.py   # Manages chat sessions
│   └── llm_service.py       # LLM integration
└── models.py             # UPDATED: New document models

src/web/
└── app.py                # UPDATED: Document dropdown UI
```

## Next Steps

See `REFACTORING_SUMMARY.md` for detailed technical changes.
See `ARCHITECTURE.md` for system architecture overview.

