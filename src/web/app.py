"""
Gradio frontend for KVCache Chatbot
"""
import argparse
import gradio as gr
import httpx
import uuid
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator


@dataclass
class WebArgs:
    ip         : str = "localhost"
    port       : int = 7860
    share      : bool = False
    backend_ip : str = "localhost"
    backend_port : int = 8000

    def __post_init__(self):
        self.backend_base_url = f"http://{self.backend_ip}:{self.backend_port}/api/v1"

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--ip', default=cls.ip, help="Frontend IP address")
        parser.add_argument('-p', '--port', type=int, default=cls.port, help="Frontend port")
        parser.add_argument('--share', action="store_true", help="Enable Gradio sharing")
        parser.add_argument('--backend-ip', default=cls.backend_ip, help="Backend API IP address")
        parser.add_argument('--backend-port', type=int, default=cls.backend_port, help="Backend API port")

        args = parser.parse_args()
        return cls(
            ip=args.ip,
            port=args.port,
            share=args.share,
            backend_ip=args.backend_ip,
            backend_port=args.backend_port
        )

class ChatbotClient:
    """Client for interacting with the backend API"""

    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self.session_id = str(uuid.uuid4())
        self.client = httpx.Client(timeout=30.0)

    def upload_document(self, file_path: str) -> dict:
        """Upload a document (independent of session)"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.client.post(
                f"{self.api_base_url}/documents/upload",
                files=files
            )
            response.raise_for_status()
            return response.json()
    
    def upload_document_and_cache(self, file_path: str) -> dict:
        """
        Upload a document and pre-cache it in KV Cache
        
        This function uploads a document and triggers KV Cache pre-warming:
        1. Upload document
        2. Run inference with document as system message (max_tokens=2)
        3. Restart model server with reset=False to preserve KV Cache
        
        This enables prefix matching acceleration for subsequent queries.
        """
        with open(file_path, 'rb') as f:
            files = {'file': f}
            # Use longer timeout as this includes model restart
            response = self.client.post(
                f"{self.api_base_url}/documents/upload_and_cache",
                files=files,
                timeout=180.0  # 3 minutes to allow for model restart
            )
            response.raise_for_status()
            return response.json()

    def list_documents(self) -> list:
        """Get list of all uploaded documents"""
        response = self.client.get(f"{self.api_base_url}/documents/list")
        response.raise_for_status()
        return response.json()

    def delete_document(self, doc_id: str) -> dict:
        """Delete a document"""
        response = self.client.delete(f"{self.api_base_url}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    
    def send_message(self, message: str, document_id: Optional[str] = None) -> str:
        """Send a message and get response (non-streaming)"""
        payload = {
            "message": message,
            "document_id": document_id
        }
        response = self.client.post(
            f"{self.api_base_url}/session/{self.session_id}/messages",
            json=payload
        )
        response.raise_for_status()
        return response.json()["message"]

    def stream_message(self, message: str, document_id: Optional[str] = None):
        """Send a message and get streaming response"""
        payload = {
            "message": message,
            "document_id": document_id
        }

        with self.client.stream(
            "POST",
            f"{self.api_base_url}/session/{self.session_id}/messages/stream",
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
            f"{self.api_base_url}/session/{self.session_id}/messages"
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json().get("messages", [])
    
    def reset_session(self):
        """Reset the session (create new session ID)"""
        try:
            self.client.delete(f"{self.api_base_url}/session/{self.session_id}")
        except:
            pass
        self.session_id = str(uuid.uuid4())

    def start_model_without_reset(self, model_name: str = None) -> dict:
        """Start model without resetting configuration"""
        payload = {"model_name": model_name} if model_name else {}
        # Use longer timeout for model startup (can take 1-2 minutes to load)
        response = self.client.post(
            f"{self.api_base_url}/model/up/without_reset",
            json=payload,
            timeout=180.0  # 3 minutes
        )
        response.raise_for_status()
        return response.json()

    def start_model_with_reset(self, model_name: str = None) -> dict:
        """Start model with reset (restart with new configuration)"""
        payload = {"model_name": model_name} if model_name else {}
        # Use longer timeout for model startup (can take 1-2 minutes to load)
        response = self.client.post(
            f"{self.api_base_url}/model/up/reset",
            json=payload,
            timeout=180.0  # 3 minutes
        )
        response.raise_for_status()
        return response.json()

    def stop_model(self) -> dict:
        """Stop the currently running model"""
        response = self.client.post(f"{self.api_base_url}/model/down")
        response.raise_for_status()
        return response.json()

    def get_model_status(self) -> dict:
        """Get current model status"""
        response = self.client.get(f"{self.api_base_url}/model/status")
        response.raise_for_status()
        return response.json()


class ChatbotWeb:

    def __init__(self, backend_url):
        # Global client instance
        self.client: Optional[ChatbotClient] = ChatbotClient(backend_url)

    def chat_with_bot(
        self,
        message: str,
        history: List[Tuple[str, str]],
        selected_doc: Optional[str]
    ) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
        """
        Handle chat interaction
        
        Args:
            message: User message
            history: Chat history
            selected_doc: Selected document ID (None if no document selected)

        Yields:
            Updated history and empty message box
        """
        if not message.strip():
            yield history, ""
            return

        # Add user message to history
        history.append((message, ""))
        yield history, ""

        # Parse document ID from dropdown value
        doc_id = None
        if selected_doc and selected_doc != "None":
            doc_id = selected_doc

        # Stream response from backend
        full_response = ""
        try:
            for chunk in self.client.stream_message(message, doc_id):
                full_response += chunk
                history[-1] = (message, full_response)
                yield history, ""
        except Exception as e:
            error_msg = f"Error: {str(e)}. Please make sure the backend server is running."
            history[-1] = (message, error_msg)
            yield history, ""


    def get_document_choices(self) -> List[Tuple[str, str]]:
        """Get list of documents for dropdown"""
        try:
            docs = self.client.list_documents()
            if not docs:
                return [("No documents uploaded", "None")]
            choices = [("No document selected", "None")]
            for doc in docs:
                choices.append((doc['filename'], doc['doc_id']))
            return choices
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return [("Error loading documents", "None")]


    def upload_file(self, file) -> Tuple[str, gr.Dropdown]:
        """
        Handle file upload

        Args:
            file: Uploaded file object

        Returns:
            Status message and updated dropdown
        """
        if file is None:
            return "Please select a file to upload", gr.Dropdown(choices=self.get_document_choices())

        try:
            result = self.client.upload_document(file.name)
            msg = f"✅ {result['message']}\nFile: {result['filename']}\nSize: {result['file_size']} bytes"
            # Refresh dropdown choices
            return msg, gr.Dropdown(choices=self.get_document_choices())
        except httpx.HTTPStatusError as e:
            return f"❌ Upload failed: {e.response.json().get('detail', str(e))}", gr.Dropdown(choices=self.get_document_choices())
        except Exception as e:
            return f"❌ Upload failed: {str(e)}. Please make sure the backend server is running.", gr.Dropdown(choices=self.get_document_choices())

    def upload_file_and_cache(self, file) -> Tuple[str, gr.Dropdown]:
        """
        Handle file upload with KV Cache pre-warming
        
        This function uploads a document and pre-caches it in the model's KV Cache
        by running inference with the document content as system message.
        This enables prefix matching acceleration for subsequent queries.

        Args:
            file: Uploaded file object

        Returns:
            Status message and updated dropdown
        """
        if file is None:
            return "Please select a file to upload", gr.Dropdown(choices=self.get_document_choices())

        try:
            result = self.client.upload_document_and_cache(file.name)
            msg = f"✅ {result['message']}\nFile: {result['filename']}\nSize: {result['file_size']} bytes\n\n🚀 KV Cache is ready for prefix matching!"
            # Refresh dropdown choices
            return msg, gr.Dropdown(choices=self.get_document_choices())
        except httpx.HTTPStatusError as e:
            return f"❌ Upload and cache failed: {e.response.json().get('detail', str(e))}", gr.Dropdown(choices=self.get_document_choices())
        except Exception as e:
            return f"❌ Upload and cache failed: {str(e)}. Please make sure the backend server is running.", gr.Dropdown(choices=self.get_document_choices())


    def clear_chat(self) -> Tuple[List, str, gr.Dropdown]:
        """Clear chat history and reset session"""
        self.client.reset_session()
        # Reset dropdown to "None" to prevent using old document context
        return [], "✓ Chat cleared. New session started.", gr.Dropdown(value="None")


    def refresh_documents(self) -> gr.Dropdown:
        """Refresh document list"""
        return gr.Dropdown(choices=self.get_document_choices())

    def on_page_load(self) -> Tuple[List, str]:
        """Handle page load/refresh - create new session"""
        self.client.reset_session()
        return [], ""


    def restart_model(self) -> Generator[tuple, None, None]:
        """Restart model with new configuration (with loading status updates)"""
        # Initial loading message
        loading_status = "🔄 Starting model server..."
        loading_log = "🔄 Restarting model server...\n⏳ This may take 30-90 seconds while the model loads...\n"
        yield loading_status, loading_log

        try:
            result = self.client.start_model_with_reset()

            # Format status message for model_status display
            if result.get('status') == 'success':
                status_msg = f"✅ {result['message']}"
            else:
                status_msg = f"❌ {result['message']}"

            # Format detailed logging for deploy_log display
            if result.get('status') == 'success':
                log_msg = f"✅ {result['message']}\n"
                log_msg += f"PID: {result.get('pid', 'N/A')}\n"
                log_msg += f"Port: {result.get('port', 'N/A')}\n"
                log_msg += f"Time: {result['timestamp']}\n"
                if result.get('command'):
                    log_msg += f"Command: {result['command']}\n"
                yield status_msg, log_msg
            else:
                # Handle error cases with detailed information
                log_msg = f"❌ {result['message']}\n"

                # Add details if available
                if result.get('details'):
                    details = result['details']
                    log_msg += "\n📋 Details:\n"

                    # Show specific error types
                    if details.get('error_type'):
                        log_msg += f"Error Type: {details['error_type']}\n"

                    if details.get('exe_path'):
                        log_msg += f"Executable: {details['exe_path']}\n"

                    if details.get('model_path'):
                        log_msg += f"Model: {details['model_path']}\n"

                    if details.get('error_output'):
                        log_msg += f"Error Output:\n{details['error_output']}\n"

                    if details.get('command'):
                        log_msg += f"Command: {details['command']}\n"

                    if details.get('working_dir'):
                        log_msg += f"Working Directory: {details['working_dir']}\n"

                    if details.get('log_file'):
                        log_msg += f"Log File: {details['log_file']}\n"

                    if details.get('hint'):
                        log_msg += f"💡 Hint: {details['hint']}\n"

                yield status_msg, log_msg

        except Exception as e:
            error_msg = f"❌ Failed to restart model: {str(e)}\n\nThis is a network or API error. Please check if the backend server is running."
            yield error_msg, error_msg


    def create_web(self):
        # Create Gradio interface with simplified layout
        with gr.Blocks(title="KVCache Chatbot", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🤖 KVCache Chatbot")

            with gr.Row():
                # Left sidebar - Document management
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Document Management")

                    # Upload section
                    gr.Markdown("**Upload Document**")
                    file_upload = gr.File(
                        label="",
                        file_types=[".txt"],
                        type="filepath",
                        height=150
                    )
                    with gr.Row():
                        upload_btn = gr.Button("Upload", variant="primary", size="sm")
                        upload_cache_btn = gr.Button("Upload & Cache", variant="secondary", size="sm")
                    upload_status = gr.Textbox(
                        label="",
                        placeholder="Upload status will appear here...",
                        interactive=False,
                        lines=3,
                        show_label=False
                    )

                    gr.Markdown("---")

                    # Document selector section
                    gr.Markdown("**Select Document**")
                    doc_dropdown = gr.Dropdown(
                        choices=[("No document selected", "None")],
                        value="None",
                        label="",
                        show_label=False,
                        interactive=True
                    )
                    refresh_btn = gr.Button("Refresh List", size="sm")

                # Main chat area
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=450,
                        show_copy_button=True
                    )

                    # Message input
                    with gr.Row():
                        msg = gr.Textbox(
                            label="",
                            placeholder="Type your message... (Press Enter to send)",
                            scale=9,
                            lines=1,
                            show_label=False
                        )
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            # Bottom section - Model controls and logging
            with gr.Row():
                # Left - Model controls
                with gr.Column(scale=1):
                    restart_btn = gr.Button("Restart Model", variant="primary", size="lg")
                    model_status = gr.Textbox(
                        label="Model Status",
                        placeholder="Model status will appear here...",
                        lines=2,
                        interactive=False,
                        show_label=True
                    )

                # Right - Deploy logging
                with gr.Column(scale=3):
                    deploy_log = gr.Textbox(
                        label="Deploy Model Logging",
                        placeholder="Model logs will appear here...",
                        lines=8,
                        interactive=False,
                        max_lines=20,
                        show_copy_button=True
                    )

            # Event handlers
            # Chat events
            msg.submit(
                fn=self.chat_with_bot,
                inputs=[msg, chatbot, doc_dropdown],
                outputs=[chatbot, msg]
            )

            # Upload events
            upload_btn.click(
                fn=self.upload_file,
                inputs=[file_upload],
                outputs=[upload_status, doc_dropdown]
            )
            
            upload_cache_btn.click(
                fn=self.upload_file_and_cache,
                inputs=[file_upload],
                outputs=[upload_status, doc_dropdown]
            )

            # Document management events
            refresh_btn.click(
                fn=self.refresh_documents,
                inputs=[],
                outputs=[doc_dropdown]
            )

            # Clear chat - handle both Clear button and Chatbot's built-in clear button
            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, upload_status, doc_dropdown]
            )

            # Handle Gradio Chatbot's built-in clear button (trash icon)
            chatbot.clear(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, upload_status, doc_dropdown]
            )

            # Page load event - reset session on page load/refresh
            demo.load(
                fn=self.on_page_load,
                inputs=[],
                outputs=[chatbot, upload_status]
            )

            # Model control events
            restart_btn.click(
                fn=self.restart_model,
                inputs=[],
                outputs=[model_status, deploy_log]
            )

        return demo


if __name__ == "__main__":
    args = WebArgs.from_args()

    print("Starting Gradio frontend...")
    print(f"Backend API: {args.backend_base_url}")
    print(f"Frontend:    http://{args.ip}:{args.port}")

    # Initialize ChatbotWeb with backend configuration
    chatbot_web = ChatbotWeb(args.backend_base_url)
    demo = chatbot_web.create_web()

    demo.launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
    )
