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
            timeout=300.0  # 5 minutes for caching operation (large documents may need more time)
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
        """Send a message and get streaming response, yields tuples of (chunk, rag_info)"""
        payload = {
            "message": message,
            "document_id": document_id
        }

        rag_info = None

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

                        # Check for RAG info
                        if data.get("rag_info") and not rag_info:
                            rag_info = data["rag_info"]
                            # Yield RAG info immediately
                            yield ("", rag_info)
                            continue

                        # Check for errors
                        if data.get("error"):
                            yield (f"Error: {data['error']}", rag_info)
                            break

                        # Yield chunks until done
                        if not data.get("done"):
                            chunk = data.get("chunk", "")
                            if chunk:  # Only yield non-empty chunks
                                yield (chunk, rag_info)
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

    def list_models(self) -> list:
        """Get list of all configured models"""
        response = self.client.get(f"{self.api_base_url}/model/list")
        response.raise_for_status()
        return response.json()

    def start_model_without_reset(self, serving_name: str = None) -> dict:
        """Start model without resetting configuration"""
        payload = {"serving_name": serving_name} if serving_name else {}
        # Use longer timeout for model startup (can take 1-2 minutes to load)
        response = self.client.post(
            f"{self.api_base_url}/model/up/without_reset",
            json=payload,
            timeout=180.0  # 3 minutes
        )
        response.raise_for_status()
        return response.json()

    def start_model_with_reset(self, serving_name: str = None) -> dict:
        """Start model with reset (restart with new configuration)"""
        payload = {"serving_name": serving_name} if serving_name else {}
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
        rag_info_displayed = False
        try:
            for chunk, rag_info in self.client.stream_message(message, doc_id):
                # Update document selection message with RAG info
                if rag_info and not rag_info_displayed:
                    rag_preview = self._format_rag_preview(rag_info)
                    if rag_preview:
                        # Find and update the document selection system message
                        for i in range(len(history) - 2, -1, -1):  # Search backwards, skip last (current message)
                            user_msg, bot_msg = history[i]
                            if user_msg == "" and "📄 **Document Selected:**" in bot_msg:
                                # Replace the hint text with RAG preview
                                lines = bot_msg.split("\n")
                                # Keep the first 3 lines (filename, size/groups, empty line)
                                # Replace the hint line
                                if len(lines) >= 3:
                                    updated_msg = "\n".join(lines[:3]) + "\n" + rag_preview
                                    history[i] = ("", updated_msg)
                                    yield history, "", ""
                                break
                        rag_info_displayed = True

                # Append chunks to response
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


    def _format_rag_info(self, rag_info: dict) -> str:
        """Format RAG retrieval information for display"""
        if not rag_info:
            return ""

        method = rag_info.get("method", "unknown")

        if method == "rag":
            group_id = rag_info.get("group_id", "unknown")
            similarity = rag_info.get("similarity_score", 0)
            filename = rag_info.get("filename", "document")
            return f"🔍 **RAG Retrieved:** `{group_id}` (similarity: {similarity:.4f}) from `{filename}`"
        elif method == "full_document":
            filename = rag_info.get("filename", "document")
            return f"📄 **Using full document:** `{filename}`"

        return ""

    def _format_rag_preview(self, rag_info: dict) -> str:
        """Format RAG retrieval preview with content snippet"""
        if not rag_info:
            return ""

        method = rag_info.get("method", "unknown")

        if method == "rag":
            content = rag_info.get("content_preview", "")

            if content:
                # Show first 300 characters of the retrieved content
                preview_content = content[:300].strip()
                if len(content) > 300:
                    preview_content += "..."
                return f"**📝 Retrieved Context:**\n```\n{preview_content}\n```"
            else:
                return "🔍 **Content retrieved successfully**"
        elif method == "full_document":
            return "📄 **Using full document content**"

        return ""

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

    def get_model_choices(self) -> List[Tuple[str, str]]:
        """Get list of models for dropdown"""
        try:
            models = self.client.list_models()
            if not models:
                return [("No models configured", "None")]
            choices = [("Select a model", "None")]
            for model in models:
                # Display serving_name, use serving_name as value
                choices.append((model['serving_name'], model['serving_name']))
            return choices
        except Exception as e:
            print(f"Error fetching models: {e}")
            return [("Error loading models", "None")]


    def upload_file(self, files) -> Tuple[str, gr.Dropdown, List[str]]:
        """
        Handle multiple file uploads

        Args:
            files: List of uploaded file objects

        Returns:
            Status message, updated dropdown, and list of uploaded document IDs
        """
        if files is None or len(files) == 0:
            return "Please select at least one file to upload", gr.Dropdown(choices=self.get_document_choices()), []

        success_results = []
        failed_results = []
        uploaded_doc_ids = []

        # Process each file
        for file in files:
            try:
                result = self.client.upload_document(file.name)
                success_results.append({
                    'filename': result['filename'],
                    'size': result['file_size'],
                    'doc_id': result['doc_id']
                })
                uploaded_doc_ids.append(result['doc_id'])
            except httpx.HTTPStatusError as e:
                # Try to parse JSON error response, fallback to text if not JSON
                try:
                    error_detail = e.response.json().get('detail', str(e))
                except (ValueError, AttributeError):
                    # Response is not JSON or has no json() method
                    error_detail = e.response.text if hasattr(e.response, 'text') else str(e)

                failed_results.append({
                    'filename': file.name.split('/')[-1],
                    'error': error_detail
                })
            except Exception as e:
                failed_results.append({
                    'filename': file.name.split('/')[-1],
                    'error': str(e)
                })

        # Build status message
        msg = ""

        if success_results:
            msg += f"✅ Successfully uploaded {len(success_results)} file(s):\n"
            for res in success_results:
                msg += f"  • {res['filename']} ({res['size']} bytes)\n"

        if failed_results:
            msg += f"\n❌ Failed to upload {len(failed_results)} file(s):\n"
            for res in failed_results:
                msg += f"  • {res['filename']}: {res['error']}\n"

        if not success_results and not failed_results:
            msg = "No files were processed"

        # Refresh dropdown choices and return all uploaded doc_ids
        return msg, gr.Dropdown(choices=self.get_document_choices()), uploaded_doc_ids

    def cache_selected_document(self, last_uploaded_doc_ids: Optional[List[str]], selected_doc: Optional[str]) -> Tuple[str, str]:
        """
        Cache documents in KV Cache (supports multiple documents)
        Priority: last uploaded documents > dropdown selected document

        Args:
            last_uploaded_doc_ids: List of IDs from the most recently uploaded documents
            selected_doc: Selected document ID from dropdown

        Returns:
            Status message and model logs
        """
        # Determine which documents to cache
        doc_ids_to_cache = []

        # Priority: use last uploaded documents if available, otherwise use dropdown selection
        if last_uploaded_doc_ids and len(last_uploaded_doc_ids) > 0:
            doc_ids_to_cache = last_uploaded_doc_ids
        elif selected_doc and selected_doc != "None":
            doc_ids_to_cache = [selected_doc]

        if not doc_ids_to_cache:
            return "Please upload documents first or select one from the dropdown", ""

        # Process each document
        success_results = []
        failed_results = []

        for doc_id in doc_ids_to_cache:
            try:
                # Get initial logs before starting cache
                initial_logs = self.fetch_model_logs()

                # Start cache operation with progress tracking
                result = self.client.cache_document(doc_id)
                success_results.append({
                    'filename': result['filename'],
                    'size': result['file_size'],
                    'doc_id': doc_id,
                    'cached_groups': result.get('cached_groups', 0),
                    'total_groups': result.get('total_groups', 0),
                    'failed_groups': result.get('failed_groups', [])
                })
            except httpx.HTTPStatusError as e:
                # Try to parse JSON error response, fallback to text if not JSON
                try:
                    error_detail = e.response.json().get('detail', str(e))
                except (ValueError, AttributeError):
                    # Response is not JSON or has no json() method
                    error_detail = e.response.text if hasattr(e.response, 'text') else str(e)

                failed_results.append({
                    'doc_id': doc_id,
                    'error': error_detail
                })
            except Exception as e:
                failed_results.append({
                    'doc_id': doc_id,
                    'error': str(e)
                })

        # Build status message
        msg = ""

        if success_results:
            msg += f"✅ Successfully cached {len(success_results)} document(s):\n"
            for res in success_results:
                cached_groups = res.get('cached_groups', 0)
                total_groups = res.get('total_groups', 0)
                failed_groups = res.get('failed_groups', [])

                msg += f"  • {res['filename']} ({res['size']} bytes)\n"
                msg += f"    📊 Groups: {cached_groups}/{total_groups} cached"

                if failed_groups:
                    msg += f" (Failed: {', '.join(failed_groups)})"
                msg += "\n"
            msg += "\n🚀 KV Cache is ready for prefix matching!"

        if failed_results:
            msg += f"\n❌ Failed to cache {len(failed_results)} document(s):\n"
            for res in failed_results:
                msg += f"  • {res['doc_id']}: {res['error']}\n"

        if not success_results and not failed_results:
            msg = "No documents were processed"

        # Fetch and display new logs after cache operation
        import time
        time.sleep(0.5)
        filtered_logs = self.fetch_model_logs()

        # Add real-time cache progress to logs
        if success_results:
            for res in success_results:
                cached_groups = res.get('cached_groups', 0)
                total_groups = res.get('total_groups', 0)
                failed_groups = res.get('failed_groups', [])

                if cached_groups > 0:
                    filtered_logs += f"\n📊 Real-time Cache Progress for {res['filename']}:\n"
                    filtered_logs += f"  ✅ Successfully cached: {cached_groups}/{total_groups} groups\n"
                    if failed_groups:
                        filtered_logs += f"  ❌ Failed groups: {', '.join(failed_groups)}\n"
                    filtered_logs += f"  🚀 KV Cache ready for prefix matching!\n"

        return msg, filtered_logs



    def clear_chat(self) -> Tuple[List, str, gr.Dropdown, List]:
        """Clear chat history and reset session"""
        self.client.reset_session()
        # Reset dropdown to "None" to prevent using old document context
        # Also clear the last uploaded doc IDs
        return [], "✓ Chat cleared. New session started.", gr.Dropdown(value="None"), []


    def refresh_documents(self) -> gr.Dropdown:
        """Refresh document list"""
        return gr.Dropdown(choices=self.get_document_choices())

    def refresh_models(self) -> gr.Dropdown:
        """Refresh model list"""
        return gr.Dropdown(choices=self.get_model_choices())

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

            # Show groups info if available (RAG mode)
            if doc_info.get('total_groups'):
                system_msg += f" | Found: {doc_info['total_groups']} groups"

            system_msg += "\n\n💡 _Enter your query to retrieve relevant content from this document._"

            # Insert system message at the end of history
            new_history = history + [("", system_msg)]

            return new_history

        except Exception as e:
            error_msg = f"❌ Error loading document: {str(e)}"
            return history + [("", error_msg)]

    def on_page_load(self) -> Tuple[List, str, gr.Dropdown, List, gr.Dropdown]:
        """Handle page load/refresh - create new session and refresh document and model lists"""
        self.client.reset_session()
        return [], "", gr.Dropdown(choices=self.get_document_choices(), value="None"), [], gr.Dropdown(choices=self.get_model_choices(), value="None")


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

    def start_model_with_reset(self, selected_model: Optional[str]) -> Generator[tuple, None, None]:
        """Start model with new configuration (with loading status updates)"""
        # Check if a model is selected
        if not selected_model or selected_model == "None":
            error_msg = "❌ Please select a model first"
            yield error_msg, error_msg, gr.Dropdown(choices=self.get_document_choices(), value="None")
            return

        # Initial loading message
        loading_status = f"🔄 Starting model: {selected_model}..."
        loading_log = f"🔄 Starting model server with: {selected_model}\n⏳ This may take 30-90 seconds while the model loads...\n"
        yield loading_status, loading_log, gr.Dropdown(choices=self.get_document_choices(), value="None")

        try:
            result = self.client.start_model_with_reset(serving_name=selected_model)

            # Format status message for model_status display
            status_msg = result['message']

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
                log_msg = f"{result['message']}\n"

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

    def check_prefix_tree_exists(self) -> bool:
        """
        Check if prefix_tree.bin exists in cache directory
        Returns True if exists, False otherwise
        """
        try:
            # Call a dedicated endpoint to check prefix_tree.bin existence
            url = f"{self.client.api_base_url}/model/check_cache_existence"
            print(f"DEBUG: Checking prefix_tree at: {url}")
            response = self.client.client.get(url)
            response.raise_for_status()
            result = response.json()
            print(f"DEBUG: API response: {result}")
            exists = result.get("prefix_tree_exists", False)
            print(f"DEBUG: prefix_tree_exists = {exists}")
            return exists
        except Exception as e:
            # If check endpoint doesn't exist or fails, assume it doesn't exist
            print(f"DEBUG: Exception in check_prefix_tree_exists: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_model_without_reset(self, selected_model: Optional[str]) -> Tuple[str, str]:
        """
        Start model without resetting configuration
        Checks for prefix_tree.bin before making API call
        """
        # Check if a model is selected
        if not selected_model or selected_model == "None":
            return "❌ Please select a model first", ""

        try:
            # First check if prefix_tree.bin exists
            url = f"{self.client.api_base_url}/model/check_cache_existence"
            try:
                response = self.client.client.get(url)
                response.raise_for_status()
                result = response.json()

                debug_msg = f"🔍 DEBUG INFO:\n"
                debug_msg += f"API URL: {url}\n"
                debug_msg += f"Response: {json.dumps(result, indent=2)}\n\n"

                if not result.get("prefix_tree_exists", False):
                    error_msg = debug_msg + "❌ prefix_tree.bin not found in cache directory.\n\n💡 Please use 'Start Model with Reset' first to create the cache file."
                    return error_msg, ""
            except Exception as e:
                error_msg = f"🔍 DEBUG: API call failed\n"
                error_msg += f"URL: {url}\n"
                error_msg += f"Error: {str(e)}\n\n"
                error_msg += "❌ Could not check prefix_tree.bin.\n\n💡 Please check backend connection."
                return error_msg, ""

            # If prefix_tree.bin exists, proceed with starting model
            result = self.client.start_model_without_reset(serving_name=selected_model)

            # Format status message
            status_msg = f"✅ {result['message']}"
            if result.get('pid'):
                status_msg += f" (PID: {result['pid']})"
            if result.get('port'):
                status_msg += f" (Port: {result['port']})"

            # Fetch updated logs
            logs = self.fetch_model_logs()

            return status_msg, logs

        except Exception as e:
            error_msg = f"❌ Failed to start model without reset: {str(e)}"
            return error_msg, ""

    def stop_model(self) -> Tuple[str, str]:
        """
        Stop the currently running model

        Returns:
            Status message and model logs
        """
        try:
            result = self.client.stop_model()

            # Format status message
            if result.get('status') == 'success':
                status_msg = result['message']
            else:
                status_msg = result['message']

            # Fetch and display logs after stop operation
            import time
            time.sleep(0.5)
            filtered_logs = self.fetch_model_logs()

            return status_msg, filtered_logs

        except Exception as e:
            error_msg = f"❌ Failed to stop model: {str(e)}\n\nThis is a network or API error. Please check if the backend server is running."
            return error_msg, ""


    def create_web(self):
        # Create Gradio interface with simplified layout
        with gr.Blocks(title="KVCache Chatbot", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🤖 KVCache Chatbot")

            # Top row - Document (left) and Chat (right)
            with gr.Row():
                # Left top - Document management
                with gr.Column(scale=1):
                    # gr.Markdown("### 📄 Document Management")

                    # Upload section
                    gr.Markdown("**📤 Upload** Document(s)")
                    file_upload = gr.File(
                        label="",
                        file_types=[".pdf"],
                        type="filepath",
                        file_count="multiple",
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
                    # Hidden state to track the most recently uploaded document IDs
                    last_uploaded_doc_ids = gr.State(value=[])

                    # gr.Markdown("---")

                    # Document selector section
                    gr.Markdown("**📋 Select** Document(s)")
                    doc_dropdown = gr.Dropdown(
                        choices=[("No document selected", "None")],
                        value="None",
                        label="",
                        show_label=False,
                        interactive=True
                    )
                    refresh_btn = gr.Button("Refresh List", size="sm")

                # Right top - Chat area
                with gr.Column(scale=3):
                    gr.Markdown("**💬 Chatbot**")

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
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1, size="md", min_width=80)

            # Bottom row - Model controls (left) and Model logging (right)
            with gr.Row():
                # Left bottom - Model controls
                with gr.Column(scale=1):
                    gr.Markdown("**🤖 Model**")

                    # Model selector
                    model_dropdown = gr.Dropdown(
                        choices=[("Select a model", "None")],
                        value="None",
                        label="",
                        show_label=False,
                        interactive=True
                    )

                    # Model control buttons (vertical stack)
                    start_btn = gr.Button("Start with Reset", variant="primary", size="sm")
                    start_no_reset_btn = gr.Button("Start without Reset", variant="primary", size="sm")
                    down_btn = gr.Button("Stop Model", variant="secondary", size="sm")

                    # Model status display
                    model_status = gr.Textbox(
                        label="",
                        placeholder="Model status...",
                        lines=3,
                        interactive=False,
                        show_label=False
                    )

                # Right bottom - Model logging
                with gr.Column(scale=3):
                    gr.Markdown("**📊 Logging**")
                    deploy_log = gr.Textbox(
                        label="",
                        placeholder="Model server logs will appear here...",
                        lines=13.2,
                        interactive=False,
                        max_lines=20,
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
                outputs=[upload_status, doc_dropdown, last_uploaded_doc_ids]
            )

            upload_cache_btn.click(
                fn=self.cache_selected_document,
                inputs=[last_uploaded_doc_ids, doc_dropdown],
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
                outputs=[chatbot, upload_status, doc_dropdown, last_uploaded_doc_ids]
            )

            # Handle Gradio Chatbot's built-in clear button (trash icon)
            chatbot.clear(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, upload_status, doc_dropdown, last_uploaded_doc_ids]
            )

            # Page load event - reset session on page load/refresh
            demo.load(
                fn=self.on_page_load,
                inputs=[],
                outputs=[chatbot, upload_status, doc_dropdown, last_uploaded_doc_ids, model_dropdown]
            )

            # Model control events
            start_btn.click(
                fn=self.start_model_with_reset,
                inputs=[model_dropdown],
                outputs=[model_status, deploy_log, doc_dropdown]
            )

            start_no_reset_btn.click(
                fn=self.start_model_without_reset,
                inputs=[model_dropdown],
                outputs=[model_status, deploy_log]
            )

            down_btn.click(
                fn=self.stop_model,
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
