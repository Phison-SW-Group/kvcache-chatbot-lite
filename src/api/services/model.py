"""
Model server management service
Cross-platform implementation for starting/stopping Llama server
"""
import os
import sys
import subprocess
import signal
import time
import psutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import urllib.request
import urllib.error

from config import settings


@dataclass(frozen=True)
class ModelServerStatus:
    """Model server status constants"""
    SUCCESS: str = "success"
    ERROR: str = "error"
    STOPPED: str = "stopped"
    RUNNING: str = "running"
    UNKNOWN: str = "unknown"


@dataclass
class ModelServerConfig:
    """Configuration for model server"""
    exe: str
    model_path: str
    cache_path: str
    log_path: str
    port: int = 13141
    host: str = "0.0.0.0"
    context_size: int = 16384
    offload_gb: int = 100
    dram_offload_gb: int = 0
    ngl: int = 100
    log_level: int = 9

    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            exe=settings.LLM_SERVER_EXE or "",
            model_path=settings.LLM_SERVER_MODEL_NAME_OR_PATH or "",
            cache_path=settings.LLM_SERVER_CACHE or "",
            log_path=settings.LLM_SERVER_LOG or ""
        )


class ModelServer:
    """Cross-platform model server manager"""
    
    def __init__(self):
        self.config = ModelServerConfig.from_env()
        self.process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger(__name__)
        self.status = ModelServerStatus()

    # # TODO: Deprecated method - use _validate_paths_detailed() instead
    # def _validate_paths(self) -> bool:
    #     """Validate that required paths exist"""
    #     if not self.config.exe:
    #         self.logger.error("LLM_SERVER_EXE not configured")
    #         return False
            
    #     if not self.config.model_path:
    #         self.logger.error("LLM_SERVER_MODEL_NAME_OR_PATH not configured")
    #         return False
            
    #     exe_path = Path(self.config.exe)
    #     if not exe_path.exists():
    #         self.logger.error(f"Server executable not found: {exe_path}")
    #         return False
            
    #     model_path = Path(self.config.model_path)
    #     if not model_path.exists():
    #         self.logger.error(f"Model file not found: {model_path}")
    #         return False
            
    #     return True
    
    def _validate_paths_detailed(self) -> Dict[str, Any]:
        """Validate paths with detailed error information"""
        errors = []
        details = {}
        
        # Check executable path
        if not self.config.exe:
            errors.append("LLM_SERVER_EXE not configured")
            details["exe_configured"] = False
        else:
            exe_path = Path(self.config.exe)
            details["exe_path"] = str(exe_path)
            details["exe_configured"] = True
            
            if not exe_path.exists():
                errors.append(f"Server executable not found: {exe_path}")
                details["exe_exists"] = False
            else:
                details["exe_exists"] = True
                # Check if executable is actually executable
                if not os.access(exe_path, os.X_OK):
                    errors.append(f"Server executable is not executable: {exe_path}")
                    details["exe_executable"] = False
                else:
                    details["exe_executable"] = True
        
        # Check model path
        if not self.config.model_path:
            errors.append("LLM_SERVER_MODEL_NAME_OR_PATH not configured")
            details["model_configured"] = False
        else:
            model_path = Path(self.config.model_path)
            details["model_path"] = str(model_path)
            details["model_configured"] = True
            
            if not model_path.exists():
                errors.append(f"Model file not found: {model_path}")
                details["model_exists"] = False
            else:
                details["model_exists"] = True
                # Check if model file is readable
                if not os.access(model_path, os.R_OK):
                    errors.append(f"Model file is not readable: {model_path}")
                    details["model_readable"] = False
                else:
                    details["model_readable"] = True
        
        # Check cache path
        if self.config.cache_path:
            cache_path = Path(self.config.cache_path)
            details["cache_path"] = str(cache_path)
            if not cache_path.exists():
                try:
                    cache_path.mkdir(parents=True, exist_ok=True)
                    details["cache_created"] = True
                except Exception as e:
                    errors.append(f"Cannot create cache directory: {cache_path} - {str(e)}")
                    details["cache_created"] = False
            else:
                details["cache_created"] = True

        # Check log path
        if self.config.log_path:
            log_path = Path(self.config.log_path)
            details["log_path"] = str(log_path)
            log_dir = log_path.parent
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    details["log_dir_created"] = True
                except Exception as e:
                    errors.append(f"Cannot create log directory: {log_dir} - {str(e)}")
                    details["log_dir_created"] = False
            else:
                details["log_dir_created"] = True

        if errors:
            return {
                "valid": False,
                "message": "; ".join(errors),
                "details": details
            }
        else:
            return {
                "valid": True,
                "message": "All paths validated successfully",
                "details": details
            }
    
    def _get_server_args(self, reset: bool = True) -> list:
        """Build server command arguments"""
        args = [
            str(self.config.exe),
            "-m", str(self.config.model_path),
            "-e", "-s", "0",
            "--host", self.config.host,
            "-c", str(self.config.context_size),
            "-mg", "0",
            "--offload-path", str(self.config.cache_path),
            "--ssd-kv-offload-gb", str(self.config.offload_gb),
            "--log-file", str(self.config.log_path),
            "--parallel", "1",
            "--no-context-shift",
            "--kv-cache-resume-policy", "0" if reset else "1",
            "--port", str(self.config.port),
            "--dram-kv-offload-gb", str(self.config.dram_offload_gb),
            "-ngl", str(self.config.ngl),
            "-lv", str(self.config.log_level)
        ]
        return args
    
    def _is_running(self) -> bool:
        """Check if server process is running"""
        # First check if we have a process reference
        if self.process is not None and self.process.poll() is None:
            return True

        # Fallback: check if port is occupied (handles reload scenarios)
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', self.config.port))
            sock.close()
            return False  # Port is free, server not running
        except OSError:
            # Port is occupied, assume server is running
            return True

    def _is_http_ready(self) -> bool:
        """Check if HTTP endpoint is responding (OpenAI-compatible)."""
        url = f"http://localhost:{self.config.port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as e:
            # 503 with "Loading model" means server is up but model still loading
            if e.code == 503:
                self.logger.debug("Server is up, model still loading...")
                return False
            return False
        except urllib.error.URLError:
            return False
        except Exception:
            return False

    def _is_completion_ready(self) -> bool:
        """Check if chat/completions endpoint is ready (not returning 503)."""
        import json

        url = f"http://localhost:{self.config.port}/v1/chat/completions"

        # Prepare a simple test request
        test_payload = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(test_payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=5.0) as resp:
                # Any 2xx response means the model is ready
                return 200 <= resp.status < 300

        except urllib.error.HTTPError as e:
            # 503 means model is still loading
            if e.code == 503:
                try:
                    error_data = json.loads(e.read().decode('utf-8'))
                    if error_data.get('error', {}).get('message') == 'Loading model':
                        self.logger.debug("Model still loading (503)...")
                        return False
                except:
                    pass
            # Other HTTP errors might indicate the model is ready but request is invalid
            # (which is fine for our test purposes)
            return False
        except urllib.error.URLError:
            return False
        except Exception as e:
            self.logger.debug(f"Completion check error: {e}")
            return False

    def _wait_until_ready(self, timeout_seconds: int = 90) -> bool:
        """Poll the chat/completions endpoint until it is ready or timeout."""
        start = time.time()
        # Poll every 0.5 seconds as requested
        interval = 0.5
        try:
            while time.time() - start < timeout_seconds:
                if not self._is_running():
                    self.logger.warning("Process stopped while waiting for model to load")
                    return False

                # Check if completion endpoint is ready
                if self._is_completion_ready():
                    elapsed = time.time() - start
                    self.logger.info(f"Model ready after {elapsed:.1f} seconds")
                    return True

                time.sleep(interval)

            self.logger.warning(f"Model loading timeout after {timeout_seconds} seconds")
            return False
        except KeyboardInterrupt:
            self.logger.info("Startup interrupted by user")
            return False

    def up(self, reset: bool = True) -> Dict[str, Any]:
        """
        Start the model server
        
        Args:
            reset: Whether to reset KV cache (True = reset, False = resume)
            
        Returns:
            Dict with status and message
        """
        try:
            # Validate configuration with detailed error messages
            validation_result = self._validate_paths_detailed()
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "message": validation_result["message"],
                    "details": validation_result["details"]
                }
            
            # Stop existing server if running
            if self._is_running():
                self.logger.info("Stopping existing server process")
                self.down()
                time.sleep(2)  # Wait for process to terminate

            # Prepare command - using Path normalization like start_llama_non_reuse.py
            exe_path = Path(self.config.exe)
            model_path = Path(self.config.model_path)
            cache_path = Path(self.config.cache_path)
            log_path = Path(self.config.log_path)

            # Ensure cache and log directories exist
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            args = self._get_server_args(reset)
            working_dir = exe_path.parent

            self.logger.info(f"Starting server with args: {' '.join(args)}")
            self.logger.info(f"Working directory: {working_dir}")

            # Start server process - exactly like start_llama_non_reuse.py
            popen_kwargs = dict(cwd=str(working_dir))
            if sys.platform == "win32":
                # Create new process group to isolate from parent console
                # This prevents accidental signal propagation but requires special handling for shutdown
                popen_kwargs.update(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

            # Don't capture stdout/stderr to avoid pipe buffer blocking
            # llama-server logs are written to --log-file anyway
            self.process = subprocess.Popen(
                args,
                **popen_kwargs
            )

            self.logger.info(f"llama-server started. PID={self.process.pid}")

            # Wait a brief moment to check if process started successfully
            time.sleep(0.5)

            # Check if process is still running (didn't immediately crash)
            if self.process.poll() is not None:
                # Process terminated immediately
                return {
                    "status": self.status.ERROR,
                    "message": "Server process terminated immediately after start",
                    "details": {
                        "command": " ".join(args),
                        "working_dir": str(working_dir),
                        "log_file": str(log_path),
                        "hint": "Check log file for error details"
                    }
                }

            # Process is running, now wait for model to fully load
            self.logger.info("Server process started, waiting for model to load...")

            if self._wait_until_ready(timeout_seconds=90):
                return {
                    "status": self.status.SUCCESS,
                    "message": f"Model server started and ready (reset={reset})",
                    "pid": self.process.pid,
                    "port": self.config.port,
                    "command": " ".join(args)
                }
            else:
                # Timeout or process died while loading
                if self.process.poll() is not None:
                    return {
                        "status": self.status.ERROR,
                        "message": "Server process terminated while loading model",
                        "details": {
                            "command": " ".join(args),
                            "log_file": str(log_path),
                            "hint": "Check log file for error details"
                        }
                    }
                else:
                    return {
                        "status": self.status.ERROR,
                        "message": "Model loading timeout - server running but model not ready",
                        "details": {
                            "pid": self.process.pid,
                            "port": self.config.port,
                            "log_file": str(log_path),
                            "hint": "Model may still be loading. Check log file or try again later."
                        }
                    }
                
        except FileNotFoundError as e:
            return {
                "status": self.status.ERROR,
                "message": f"Executable not found: {str(e)}",
                "details": {
                    "exe_path": self.config.exe,
                    "error_type": "FileNotFoundError"
                }
            }
        except PermissionError as e:
            return {
                "status": self.status.ERROR,
                "message": f"Permission denied: {str(e)}",
                "details": {
                    "exe_path": self.config.exe,
                    "error_type": "PermissionError"
                }
            }
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            return {
                "status": self.status.ERROR,
                "message": f"Failed to start server: {str(e)}",
                "details": {
                    "error_type": type(e).__name__,
                    "command": " ".join(args)
                }
            }
    
    def down(self) -> Dict[str, Any]:
        """
        Stop the model server
        
        Returns:
            Dict with status and message
        """
        try:
            if not self._is_running():
                return {
                    "status": self.status.SUCCESS,
                    "message": "Server was not running"
                }
            
            # If we have process reference, use it
            if self.process is not None and self.process.poll() is None:
                self.logger.info(f"Stopping server process (PID: {self.process.pid})")

                if sys.platform == "win32":
                    # Windows: Send CTRL_C_EVENT for graceful shutdown
                    # This should trigger llama-server to save prefix_tree.bin
                    try:
                        self.logger.info("Sending CTRL_C_EVENT for graceful shutdown...")
                        os.kill(self.process.pid, signal.CTRL_C_EVENT)
                    except Exception as e:
                        # If CTRL_C fails, try CTRL_BREAK as fallback
                        self.logger.warning(f"CTRL_C_EVENT failed: {e}, trying CTRL_BREAK_EVENT")
                        self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    # Unix/Linux: Send SIGTERM to process group
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                # Wait for process to terminate (allow time for cache saving)
                try:
                    self.logger.info("Waiting for server to gracefully shutdown (saving KV cache)...")
                    # Flushing KV cache to SSD can take time, especially for large caches
                    self.process.wait(timeout=60)  # Give enough time for "Flushing KV cache to SSD"
                    self.logger.info("Server shut down gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    self.logger.warning("Graceful shutdown timeout (60s), force killing process")
                    self.logger.warning("This may prevent prefix_tree.bin from being saved")
                    if sys.platform == "win32":
                        self.process.kill()
                    else:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()

                self.process = None
            else:
                # Fallback: find and kill process by port (for reload scenarios)
                self.logger.info(f"Process reference lost, finding process by port {self.config.port}")
                killed = False

                for proc in psutil.process_iter(['pid', 'name', 'connections']):
                    try:
                        if proc.info['name'] and 'llama-server' in proc.info['name'].lower():
                            for conn in proc.connections():
                                if conn.laddr.port == self.config.port:
                                    self.logger.info(f"Found server process: PID {proc.pid}")
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=10)
                                    except psutil.TimeoutExpired:
                                        proc.kill()
                                    killed = True
                                    break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                if not killed:
                    return {
                        "status": self.status.ERROR,
                        "message": f"Could not find server process on port {self.config.port}"
                    }
            
            return {
                "status": self.status.SUCCESS,
                "message": "Server stopped successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")
            return {
                "status": self.status.ERROR,
                "message": f"Failed to stop server: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get server status
        
        Returns:
            Dict with server status information
        """
        if not self._is_running():
            return {
                "status": self.status.STOPPED,
                "message": "Server is not running"
            }
        
        try:
            # Get process info
            process = psutil.Process(self.process.pid)
            return {
                "status": self.status.RUNNING,
                "message": "Server is running",
                "pid": self.process.pid,
                "port": self.config.port,
                "memory_usage": process.memory_info().rss,
                "cpu_percent": process.cpu_percent(),
                "create_time": process.create_time()
            }
        except Exception as e:
            return {
                "status": self.status.UNKNOWN,
                "message": f"Could not get status: {str(e)}"
            }


# Global model server instance
model_server = ModelServer()


if __name__ == "__main__":
    import json
    print("=" * 60)
    print("Testing ModelServer.up() with reset=True")
    print("=" * 60)
    
    result = model_server.up(reset=True)
    print(json.dumps(result, indent=2, default=str))
    
    if result["status"] == "success":
        print("\n✅ Model server started successfully!")
        print(f"PID: {result.get('pid')}")
        print(f"Port: {result.get('port')}")
        print("\nWaiting 5 seconds before checking status...")
        import time
        time.sleep(5)
        
        status = model_server.get_status()
        print("\nStatus check:")
        print(json.dumps(status, indent=2, default=str))
        
        print("\nTest complete. Remember to stop the server manually if needed.")
    else:
        print("\n❌ Failed to start model server")
        print("Check the error details above")