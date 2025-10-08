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
    
    def cache_document(self, doc_id: str) -> dict:
        """
        Cache an already uploaded document in KV Cache
        
        This function triggers KV Cache pre-warming for an existing document:
        1. Retrieve document content from document manager
        2. Run inference with document as system message (max_tokens=2)
        
        This enables prefix matching acceleration for subsequent queries.
        """
        response = self.client.post(
            f"{self.api_base_url}/documents/cache/{doc_id}",
            timeout=60.0  # 1 minute for caching operation
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
    
    def get_document_info(self, doc_id: str) -> dict:
        """Get document information including metadata"""
        response = self.client.get(f"{self.api_base_url}/documents/{doc_id}")
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
    
    def get_model_logs(self, lines: int = 100, pattern: Optional[str] = None) -> dict:
        """
        Get model server logs with optional pattern filtering
        
        Args:
            lines: Number of recent lines to retrieve
            pattern: Regex pattern to filter logs
            
        Returns:
            Dict with log information
        """
        params = {"lines": lines}
        if pattern:
            params["pattern"] = pattern
        
        response = self.client.get(
            f"{self.api_base_url}/logs/server",
            params=params
        )
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
    ) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
        """
        Handle chat interaction
        
        Args:
            message: User message
            history: Chat history
            selected_doc: Selected document ID (None if no document selected)

        Yields:
            Updated history, empty message box, and updated logs
        """
        if not message.strip():
            yield history, "", ""
            return

        # Add user message to history
        history.append((message, ""))
        yield history, "", ""

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
                yield history, "", ""
            
            # After response is complete, fetch and display new logs
            import time
            time.sleep(0.5)  # Wait for logs to be written
            filtered_logs = self.fetch_model_logs()
            yield history, "", filtered_logs
            
        except Exception as e:
            error_msg = f"Error: {str(e)}. Please make sure the backend server is running."
            history[-1] = (message, error_msg)
            yield history, "", ""


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


    def upload_file(self, file) -> Tuple[str, gr.Dropdown, str]:
        """
        Handle file upload

        Args:
            file: Uploaded file object

        Returns:
            Status message, updated dropdown, and uploaded document ID
        """
        if file is None:
            return "Please select a file to upload", gr.Dropdown(choices=self.get_document_choices()), None

        try:
            result = self.client.upload_document(file.name)
            msg = f"✅ {result['message']}\nFile: {result['filename']}\nSize: {result['file_size']} bytes"
            # Refresh dropdown choices and return the doc_id
            return msg, gr.Dropdown(choices=self.get_document_choices()), result['doc_id']
        except httpx.HTTPStatusError as e:
            return f"❌ Upload failed: {e.response.json().get('detail', str(e))}", gr.Dropdown(choices=self.get_document_choices()), None
        except Exception as e:
            return f"❌ Upload failed: {str(e)}. Please make sure the backend server is running.", gr.Dropdown(choices=self.get_document_choices()), None

    def cache_selected_document(self, last_uploaded_doc_id: Optional[str], selected_doc: Optional[str]) -> Tuple[str, str]:
        """
        Cache the selected document in KV Cache
        Priority: last uploaded document > dropdown selected document
        
        Args:
            last_uploaded_doc_id: ID of the most recently uploaded document
            selected_doc: Selected document ID from dropdown
            
        Returns:
            Status message and model logs
        """
        # Priority: use last uploaded document if available, otherwise use dropdown selection
        doc_id = last_uploaded_doc_id if last_uploaded_doc_id else selected_doc
        
        if not doc_id or doc_id == "None":
            return "Please upload a document first or select one from the dropdown", ""
        
        try:
            result = self.client.cache_document(doc_id)
            msg = f"✅ {result['message']}\nFile: {result['filename']}\nSize: {result['file_size']} bytes\n\n🚀 KV Cache is ready for prefix matching!"
            
            # Fetch and display new logs after cache operation
            import time
            time.sleep(0.5)
            filtered_logs = self.fetch_model_logs()
            
            return msg, filtered_logs
        except httpx.HTTPStatusError as e:
            return f"❌ Cache failed: {e.response.json().get('detail', str(e))}", ""
        except Exception as e:
            return f"❌ Cache failed: {str(e)}. Please make sure the backend server is running.", ""

    def upload_file_and_cache(self, file) -> Tuple[str, gr.Dropdown, str]:
        """
        Handle file upload with KV Cache pre-warming
        
        This function uploads a document and pre-caches it in the model's KV Cache
        by running inference with the document content as system message.
        This enables prefix matching acceleration for subsequent queries.

        Args:
            file: Uploaded file object

        Returns:
            Status message, updated dropdown, and model logs
        """
        if file is None:
            return "Please select a file to upload", gr.Dropdown(choices=self.get_document_choices()), ""

        try:
            result = self.client.upload_document_and_cache(file.name)
            msg = f"✅ {result['message']}\nFile: {result['filename']}\nSize: {result['file_size']} bytes\n\n🚀 KV Cache is ready for prefix matching!"
            
            # Fetch and display new logs after cache operation
            import time
            time.sleep(0.5)
            filtered_logs = self.fetch_model_logs()
            
            # Refresh dropdown choices
            return msg, gr.Dropdown(choices=self.get_document_choices()), filtered_logs
        except httpx.HTTPStatusError as e:
            return f"❌ Upload and cache failed: {e.response.json().get('detail', str(e))}", gr.Dropdown(choices=self.get_document_choices()), ""
        except Exception as e:
            return f"❌ Upload and cache failed: {str(e)}. Please make sure the backend server is running.", gr.Dropdown(choices=self.get_document_choices()), ""


    def clear_chat(self) -> Tuple[List, str, gr.Dropdown]:
        """Clear chat history and reset session"""
        self.client.reset_session()
        # Reset dropdown to "None" to prevent using old document context
        return [], "✓ Chat cleared. New session started.", gr.Dropdown(value="None")


    def refresh_documents(self) -> gr.Dropdown:
        """Refresh document list"""
        return gr.Dropdown(choices=self.get_document_choices())
    
    def on_document_select(self, selected_doc: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Handle document selection from dropdown
        
        Args:
            selected_doc: Selected document ID
            history: Current chat history
        
        Returns:
            Updated history with system message (or cleared history)
        """
        # Remove previous document selection message if exists
        # Check if the last message is a document selection system message
        if history and len(history) > 0:
            last_user_msg, last_bot_msg = history[-1]
            # System messages have empty user message and contain "Document Selected"
            if last_user_msg == "" and "📄 **Document Selected:**" in last_bot_msg:
                history = history[:-1]  # Remove the last system message
        
        # If no document selected, just return history without adding any message
        if not selected_doc or selected_doc == "None":
            return history
        
        try:
            # Get document info with preview
            doc_info = self.client.get_document_info(selected_doc)
            
            # Add system message to chatbot
            system_msg = f"📄 **Document Selected:** {doc_info['filename']}\n"
            system_msg += f"📊 Size: {doc_info['file_size']} bytes"
            if doc_info.get('total_lines'):
                system_msg += f" | Lines: {doc_info['total_lines']}"
            
            if doc_info.get('content_preview'):
                system_msg += f"\n\n**Preview:**\n```\n{doc_info['content_preview']}\n```"
                if doc_info.get('total_lines', 0) > 10:
                    system_msg += "\n_(showing first 10 lines)_"
            
            # Insert system message at the end of history
            new_history = history + [("", system_msg)]
            
            return new_history
            
        except Exception as e:
            error_msg = f"❌ Error loading document: {str(e)}"
            return history + [("", error_msg)]

    def on_page_load(self) -> Tuple[List, str, gr.Dropdown]:
        """Handle page load/refresh - create new session and refresh document list"""
        self.client.reset_session()
        return [], "", gr.Dropdown(choices=self.get_document_choices(), value="None")


    def fetch_model_logs(self) -> str:
        """
        Fetch filtered model logs using multiple patterns:
        - [MDW][Info][Runtime] (model runtime info)
        - prompt eval time = (prompt processing performance)
        - eval time = (token generation performance) 
        - total time = (total processing time)
        - slot update_slots: ... Hit token cnt (GPU): (complete slot update line with GPU cache stats)
        
        Only displays the most recent logs to show new activity
        
        Returns:
            Formatted log string
        """
        try:
            # Define multiple filter patterns for comprehensive log capture
            patterns = [
                r"\[MDW\]\[Info\]\[Runtime\]",  # Model runtime info
                r"prompt eval time =",          # Prompt processing time
                r"eval time =",                 # Token generation time
                r"total time =",                # Total processing time
                r"slot update_slots:.*Hit token cnt \(GPU\):"  # Complete slot update line with GPU hit count
            ]
            
            # Combine patterns with OR operator
            combined_pattern = "|".join(f"({pattern})" for pattern in patterns)
            
            # Request more lines to filter, but only show the last 20
            result = self.client.get_model_logs(lines=1000, pattern=combined_pattern)
            
            if not result.get('logs'):
                return "📭 No relevant logs available yet (checking for runtime, performance, slot, and cache hit logs)"
            
            logs = result['logs']
            
            # Only show the most recent 20 lines to focus on new activity
            recent_logs = logs[-20:] if len(logs) > 20 else logs
            
            header = f""

            return header + "".join(recent_logs)
            
        except Exception as e:
            return f"❌ Failed to fetch logs: {str(e)}"
    
    def restart_model(self) -> Generator[tuple, None, None]:
        """Restart model with new configuration (with loading status updates)"""
        # Initial loading message
        loading_status = "🔄 Starting model server..."
        loading_log = "🔄 Restarting model server...\n⏳ This may take 30-90 seconds while the model loads...\n"
        yield loading_status, loading_log, gr.Dropdown(choices=self.get_document_choices(), value="None")

        try:
            result = self.client.start_model_with_reset()

            # Format status message for model_status display
            if result.get('status') == 'success':
                status_msg = f"✅ {result['message']}"
            else:
                status_msg = f"❌ {result['message']}"

            # Format detailed logging for deploy_log display
            if result.get('status') == 'success':
                # Show basic success info first
                log_msg = f"✅ {result['message']}\n"
                log_msg += f"PID: {result.get('pid', 'N/A')}\n"
                log_msg += f"Port: {result.get('port', 'N/A')}\n"
                log_msg += f"Time: {result['timestamp']}\n"
                log_msg += "\n🔄 Fetching model server logs...\n"
                yield status_msg, log_msg, gr.Dropdown(choices=self.get_document_choices(), value="None")
                
                # Wait a moment for logs to be written, then fetch filtered logs
                import time
                time.sleep(1)
                
                # Fetch and display filtered logs
                filtered_logs = self.fetch_model_logs()
                yield status_msg, filtered_logs, gr.Dropdown(choices=self.get_document_choices(), value="None")
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

                yield status_msg, log_msg, gr.Dropdown(choices=self.get_document_choices(), value="None")

        except Exception as e:
            error_msg = f"❌ Failed to restart model: {str(e)}\n\nThis is a network or API error. Please check if the backend server is running."
            yield error_msg, error_msg, gr.Dropdown(choices=self.get_document_choices(), value="None")


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
                        upload_cache_btn = gr.Button("Cache", variant="secondary", size="sm")
                    upload_status = gr.Textbox(
                        label="",
                        placeholder="Upload status will appear here...",
                        interactive=False,
                        lines=3,
                        show_label=False
                    )
                    # Hidden state to track the most recently uploaded document ID
                    last_uploaded_doc_id = gr.State(value=None)

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

                    gr.Markdown("---")

                    # Model controls section (moved back to left sidebar)
                    gr.Markdown("**Model Controls**")
                    restart_btn = gr.Button("Restart Model", variant="primary", size="lg")
                    model_status = gr.Textbox(
                        label="Model Status",
                        placeholder="Model status...",
                        lines=1,
                        interactive=False,
                        show_label=False
                    )

                # Main chat area
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=400,
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
                    
                    # Model logging directly under message input (tighter layout)
                    with gr.Row():
                        gr.Markdown("**📊 Model Logging**", elem_classes="compact-header")
                        refresh_log_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm", scale=0, min_width=100)
                    
                    deploy_log = gr.Textbox(
                        label="",
                        placeholder="Model server logs will appear here...\nClick 'Refresh' to fetch latest logs.",
                        lines=5,
                        interactive=False,
                        max_lines=15,
                        show_copy_button=True,
                        show_label=False
                    )

            # Event handlers
            # Chat events
            msg.submit(
                fn=self.chat_with_bot,
                inputs=[msg, chatbot, doc_dropdown],
                outputs=[chatbot, msg, deploy_log]
            )

            # Upload events
            upload_btn.click(
                fn=self.upload_file,
                inputs=[file_upload],
                outputs=[upload_status, doc_dropdown, last_uploaded_doc_id]
            )
            
            upload_cache_btn.click(
                fn=self.cache_selected_document,
                inputs=[last_uploaded_doc_id, doc_dropdown],
                outputs=[upload_status, deploy_log]
            )

            # Document management events
            refresh_btn.click(
                fn=self.refresh_documents,
                inputs=[],
                outputs=[doc_dropdown]
            )
            
            # Document selection event - show system message in chatbot
            doc_dropdown.change(
                fn=self.on_document_select,
                inputs=[doc_dropdown, chatbot],
                outputs=[chatbot]
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
                outputs=[chatbot, upload_status, doc_dropdown]
            )

            # Model control events
            restart_btn.click(
                fn=self.restart_model,
                inputs=[],
                outputs=[model_status, deploy_log, doc_dropdown]
            )
            
            # Log refresh event
            refresh_log_btn.click(
                fn=self.fetch_model_logs,
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
