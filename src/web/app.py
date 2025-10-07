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
    ip         : str = "0.0.0.0"
    port       : int = 7860
    share      : bool = False
    backend_ip : str = "0.0.0.0"
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


    def clear_chat(self) -> Tuple[List, str]:
        """Clear chat history and reset session"""
        self.client.reset_session()
        return [], "✓ Chat cleared. New session started."


    def refresh_documents(self) -> gr.Dropdown:
        """Refresh document list"""
        return gr.Dropdown(choices=self.get_document_choices())


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
                    upload_btn = gr.Button("Upload", variant="primary", size="lg")
                    upload_status = gr.Textbox(
                        label="",
                        placeholder="Upload status will appear here...",
                        interactive=False,
                        lines=2,
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

            # Document management events
            refresh_btn.click(
                fn=self.refresh_documents,
                inputs=[],
                outputs=[doc_dropdown]
            )

            # Clear chat
            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, upload_status]
            )

            # Model control events (placeholder functions)
            restart_btn.click(
                fn=lambda: "Model restart initiated... (Not implemented yet)",
                inputs=[],
                outputs=[deploy_log]
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
