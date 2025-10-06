"""
Gradio frontend for KVCache Chatbot
"""
import gradio as gr
import httpx
import uuid
from typing import List, Tuple, Optional, Generator
import json


# Backend API configuration
API_BASE_URL = "http://localhost:8000/api/v1"


class ChatbotClient:
    """Client for interacting with the backend API"""
    
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self.session_id = str(uuid.uuid4())
        self.client = httpx.Client(timeout=30.0)
    
    def upload_document(self, file_path: str) -> dict:
        """Upload a document to the backend"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'session_id': self.session_id}
            response = self.client.post(
                f"{self.api_base_url}/upload",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    def send_message(self, message: str, use_document: bool = False) -> str:
        """Send a message and get response (non-streaming)"""
        payload = {
            "session_id": self.session_id,
            "message": message,
            "use_document": use_document
        }
        response = self.client.post(
            f"{self.api_base_url}/chat/message",
            json=payload
        )
        response.raise_for_status()
        return response.json()["message"]
    
    def stream_message(self, message: str, use_document: bool = False):
        """Send a message and get streaming response"""
        payload = {
            "session_id": self.session_id,
            "message": message,
            "use_document": use_document
        }
        
        with self.client.stream(
            "POST",
            f"{self.api_base_url}/chat/stream",
            json=payload
        ) as response:
            response.raise_for_status()
            
            # Read SSE stream line by line
            for line in response.iter_lines():
                # Decode bytes to string if necessary
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse SSE data lines
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        # Check for errors
                        if data.get("error"):
                            yield f"Error: {data['error']}"
                            break
                        
                        # Yield chunks until done
                        if not data.get("done"):
                            chunk = data.get("chunk", "")
                            if chunk:  # Only yield non-empty chunks
                                yield chunk
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse SSE data: {line}, error: {e}")
                        continue
    
    def get_chat_history(self) -> list:
        """Get full chat history"""
        response = self.client.get(
            f"{self.api_base_url}/chat/history/{self.session_id}"
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json().get("messages", [])
    
    def reset_session(self):
        """Reset the session (create new session ID)"""
        try:
            self.client.delete(f"{self.api_base_url}/chat/session/{self.session_id}")
        except:
            pass
        self.session_id = str(uuid.uuid4())


# Global client instance
client = ChatbotClient(API_BASE_URL)


def chat_with_bot(
    message: str,
    history: List[Tuple[str, str]],
    use_document: bool
) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
    """
    Handle chat interaction
    
    Args:
        message: User message
        history: Chat history
        use_document: Whether to use uploaded document
        
    Yields:
        Updated history and empty message box
    """
    if not message.strip():
        yield history, ""
        return
    
    # Add user message to history
    history.append((message, ""))
    yield history, ""
    
    # Stream response from backend
    full_response = ""
    try:
        for chunk in client.stream_message(message, use_document):
            full_response += chunk
            history[-1] = (message, full_response)
            yield history, ""
    except Exception as e:
        error_msg = f"Error: {str(e)}. Please make sure the backend server is running."
        history[-1] = (message, error_msg)
        yield history, ""


def upload_file(file) -> str:
    """
    Handle file upload
    
    Args:
        file: Uploaded file object
        
    Returns:
        Status message
    """
    if file is None:
        return "Please select a file to upload"
    
    try:
        result = client.upload_document(file.name)
        return f"✅ {result['message']}\nFile: {result['filename']}\nSize: {result['file_size']} bytes"
    except httpx.HTTPStatusError as e:
        return f"❌ Upload failed: {e.response.json().get('detail', str(e))}"
    except Exception as e:
        return f"❌ Upload failed: {str(e)}. Please make sure the backend server is running."


def clear_chat() -> Tuple[List, str]:
    """Clear chat history and reset session"""
    client.reset_session()
    return [], "✓ Chat cleared. New session started."


# Create Gradio interface with simplified layout
with gr.Blocks(title="KVCache Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 KVCache Chatbot")
    
    with gr.Row():
        # Left sidebar - Document upload
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Upload Document")
            file_upload = gr.File(
                label="",
                file_types=[".txt"],
                type="filepath",
                height=200
            )
            upload_btn = gr.Button("Upload", variant="primary", size="lg")
            upload_status = gr.Textbox(
                label="",
                placeholder="Upload status will appear here...",
                interactive=False,
                lines=2,
                show_label=False
            )
        
        # Main chat area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=450,
                show_copy_button=True
            )
            
            # Message input with checkbox
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Type your message... (Press Enter to send)",
                    scale=9,
                    lines=1,
                    show_label=False
                )
                use_doc_checkbox = gr.Checkbox(
                    label="Use Doc",
                    value=False,
                    scale=1
                )
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
    
    # Bottom section - Model controls and logging
    with gr.Row():
        # Left - Model controls
        with gr.Column(scale=1):
            model_name = gr.Textbox(
                label="Model Name",
                placeholder="Enter model name (e.g., gemini-2.5-flash)",
                lines=1,
                interactive=True
            )
            restart_btn = gr.Button("Restart Model", variant="primary", size="lg")
        
        # Right - Deploy logging
        with gr.Column(scale=3):
            deploy_log = gr.Textbox(
                label="Deploy Model Logging",
                placeholder="Model logs will appear here...",
                lines=4,
                interactive=False,
                max_lines=10
            )
    
    # Event handlers
    # Chat events
    msg.submit(
        fn=chat_with_bot,
        inputs=[msg, chatbot, use_doc_checkbox],
        outputs=[chatbot, msg]
    )
    
    # Upload events
    upload_btn.click(
        fn=upload_file,
        inputs=[file_upload],
        outputs=[upload_status]
    )
    
    # Clear chat
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, upload_status]
    )
    
    # Model control events (placeholder functions)
    restart_btn.click(
        fn=lambda: "Model restart initiated... (Not implemented yet)",
        inputs=[],
        outputs=[deploy_log]
    )


if __name__ == "__main__":
    print("Starting Gradio frontend...")
    print("Make sure the backend API is running at http://localhost:8000")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

