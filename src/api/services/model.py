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
import threading
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import urllib.request
import urllib.error

from config import settings


# llama-server creates this subdirectory under cache_path
MAESTRO_CACHE_SUBDIR = "maestro_phison"


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
    alias: Optional[str] = None
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
            cache_path=settings.LLM_SERVER_CACHE or "",
            log_path=settings.LLM_SERVER_LOG or "",
            model_path=settings.MODEL_NAME_OR_PATH or "",
            alias=settings.MODEL_SERVING_NAME or "",
        )


class ModelServer:
    """Cross-platform model server manager"""

    def __init__(self):
        self.config = ModelServerConfig.from_env()
        self.process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger(__name__)
        self.status = ModelServerStatus()
        self.log_threads: list = []  # Keep track of log streaming threads
        self.last_reset_mode: Optional[bool] = None  # Track last startup mode (True=reset, False=resume)

    def _stream_output_to_log(self, stream, stream_name: str, log_file_path: str):
        """
        Stream subprocess output to both console and log file

        Args:
            stream: stdout or stderr stream
            stream_name: Name of the stream for logging
            log_file_path: Path to log file
        """
        try:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                for line in iter(stream.readline, b''):
                    if line:
                        decoded_line = line.decode('utf-8', errors='replace')
                        # Write to console (terminal)
                        sys.stdout.write(decoded_line)
                        sys.stdout.flush()
                        # Write to log file
                        log_file.write(decoded_line)
                        log_file.flush()
        except Exception as e:
            self.logger.error(f"Error streaming {stream_name}: {e}")

    # # TODO: Deprecated method - use _validate_paths_detailed() instead
    # def _validate_paths(self) -> bool:
    #     """Validate that required paths exist"""
    #     if not self.config.exe:
    #         self.logger.error("LLM_SERVER_EXE not configured")
    #         return False

    #     if not self.config.model_path:
    #         self.logger.error("MODEL_NAME_OR_PATH not configured")
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
            errors.append("MODEL_NAME_OR_PATH not configured")
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
            "-a", str(self.config.alias),
            "-e", "-s", "0",
            "--host", self.config.host,
            "-c", str(self.config.context_size),
            "-mg", "0",
            "--offload-path", str(self.config.cache_path),
            "--ssd-kv-offload-gb", str(self.config.offload_gb),
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

    def _send_ctrl_c_via_console_attach(self, pid: int) -> bool:
        """
        Send CTRL_C event to process using console attachment (Windows only).
        This is the most reliable way to trigger graceful shutdown with prefix_tree.bin saving.

        Based on the working implementation from process_manager.py
        """
        if sys.platform != "win32":
            return False

        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            ATTACH_PARENT_PROCESS = -1

            self.logger.info(f"Attempting console attach to PID {pid}...")

            # Step 1: Free current console attachment
            try:
                kernel32.FreeConsole()
            except Exception:
                pass

            # Step 2: Attach to target process console
            if not kernel32.AttachConsole(pid):
                err = ctypes.get_last_error()
                try:
                    msg = str(ctypes.WinError(err))
                except Exception:
                    msg = ""
                self.logger.warning(f"AttachConsole({pid}) failed, last_error={err} {msg}")
                return False

            try:
                # Step 3: Ignore CTRL_C in current process to avoid self-interrupt
                kernel32.SetConsoleCtrlHandler(None, True)

                # Step 4: Send CTRL_C_EVENT (0) to process group
                # This is equivalent to pressing Ctrl+C in the console
                sent = kernel32.GenerateConsoleCtrlEvent(0, pid)  # 0 = CTRL_C_EVENT
                if sent:
                    self.logger.info("✅ Successfully sent CTRL_C via console attach")
                    return True
                else:
                    self.logger.warning("GenerateConsoleCtrlEvent returned False")
                    return False
            finally:
                # Step 5: Clean up console attachments
                time.sleep(0.2)  # Small delay for event propagation
                kernel32.FreeConsole()
                kernel32.SetConsoleCtrlHandler(None, False)
                try:
                    kernel32.AttachConsole(ATTACH_PARENT_PROCESS)
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(f"Failed to send CTRL_C via console attach: {e}")
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
            from services.model_log import model_log_service
            model_log_service.append_log(f"Starting model server (reset={reset})...")

            # Validate configuration with detailed error messages
            validation_result = self._validate_paths_detailed()
            if not validation_result["valid"]:
                model_log_service.append_log(f"Validation failed: {validation_result['message']}")
                return {
                    "status": "error",
                    "message": validation_result["message"],
                    "details": validation_result["details"]
                }


            # Store the reset mode for later reference
            self.last_reset_mode = reset

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

            model_log_service.append_log(f"Server command: {' '.join(args)}")
            model_log_service.append_log(f"Working directory: {working_dir}")

            # Start server process with stdout/stderr capture for logging
            # KEY: Do NOT use CREATE_NEW_PROCESS_GROUP to allow Ctrl+C signal for prefix_tree.bin generation
            popen_kwargs = dict(
                cwd=str(working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1  # Line buffered
            )

            # Note: Intentionally NOT using CREATE_NEW_PROCESS_GROUP on Windows
            # to allow the process to receive console control events (Ctrl+C)

            # Start process with stdout/stderr capture
            self.process = subprocess.Popen(
                args,
                **popen_kwargs
            )

            self.logger.info(f"llama-server started. PID={self.process.pid}")
            model_log_service.append_log(f"llama-server process started. PID={self.process.pid}")

            # Start threads to stream stdout and stderr to log file
            # Use model_log_service's current log path
            current_log_path = model_log_service.get_current_log_path() or str(log_path)

            stdout_thread = threading.Thread(
                target=self._stream_output_to_log,
                args=(self.process.stdout, "stdout", current_log_path),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._stream_output_to_log,
                args=(self.process.stderr, "stderr", current_log_path),
                daemon=True
            )

            stdout_thread.start()
            stderr_thread.start()

            self.log_threads = [stdout_thread, stderr_thread]
            model_log_service.append_log(f"Logging to: {current_log_path}")

            # Wait a brief moment to check if process started successfully
            time.sleep(0.5)

            # Check if process is still running (didn't immediately crash)
            if self.process.poll() is not None:
                # Process terminated immediately
                error_msg = "Server process terminated immediately after start"
                model_log_service.append_log(f"ERROR: {error_msg}")
                return {
                    "status": self.status.ERROR,
                    "message": error_msg,
                    "details": {
                        "command": " ".join(args),
                        "working_dir": str(working_dir),
                        "log_file": str(log_path),
                        "hint": "Check llama-server log file for error details"
                    }
                }

            # Process is running, now wait for model to fully load
            self.logger.info("Server process started, waiting for model to load...")
            model_log_service.append_log("Waiting for model to load...")

            if self._wait_until_ready(timeout_seconds=90):
                model_log_service.append_log(f"✅ Model server started and ready (PID={self.process.pid}, Port={self.config.port})")
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
                    error_msg = "Server process terminated while loading model"
                    model_log_service.append_log(f"ERROR: {error_msg}")
                    return {
                        "status": self.status.ERROR,
                        "message": error_msg,
                        "details": {
                            "command": " ".join(args),
                            "log_file": str(log_path),
                            "hint": "Check llama-server log file for error details"
                        }
                    }
                else:
                    error_msg = "Model loading timeout - server running but model not ready"
                    model_log_service.append_log(f"ERROR: {error_msg}")
                    return {
                        "status": self.status.ERROR,
                        "message": error_msg,
                        "details": {
                            "pid": self.process.pid,
                            "port": self.config.port,
                            "log_file": str(log_path),
                            "hint": "Model may still be loading. Check llama-server log file or try again later."
                        }
                    }

        except FileNotFoundError as e:
            error_msg = f"Executable not found: {str(e)}"
            model_log_service.append_log(f"ERROR: {error_msg}")
            return {
                "status": self.status.ERROR,
                "message": error_msg,
                "details": {
                    "exe_path": self.config.exe,
                    "error_type": "FileNotFoundError"
                }
            }
        except PermissionError as e:
            error_msg = f"Permission denied: {str(e)}"
            model_log_service.append_log(f"ERROR: {error_msg}")
            return {
                "status": self.status.ERROR,
                "message": error_msg,
                "details": {
                    "exe_path": self.config.exe,
                    "error_type": "PermissionError"
                }
            }
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            error_msg = f"Failed to start server: {str(e)}"
            model_log_service.append_log(f"ERROR: {error_msg}")
            return {
                "status": self.status.ERROR,
                "message": error_msg,
                "details": {
                    "error_type": type(e).__name__,
                    "command": " ".join(args)
                }
            }

    async def down(self) -> Dict[str, Any]:
        """
        Stop the model server

        Returns:
            Dict with status and message
        """
        try:
            from services.model_log import model_log_service
            model_log_service.append_log("Stopping model server...")

            if not self._is_running():
                model_log_service.append_log("Server was not running")
                return {
                    "status": self.status.SUCCESS,
                    "message": "Server was not running"
                }

            # If we have process reference, use it
            if self.process is not None and self.process.poll() is None:
                pid = self.process.pid
                self.logger.info(f"Stopping server process (PID: {pid})")
                model_log_service.append_log(f"Stopping server process (PID: {pid})")

                signal_sent = False

                if sys.platform == "win32":
                    # Windows: Use console attach method to send CTRL_C (most reliable)
                    # This is equivalent to pressing Ctrl+C in the console window
                    self.logger.info("Using console attach method to send CTRL_C...")
                    model_log_service.append_log("Sending CTRL_C via console attachment (equivalent to Ctrl+C)...")

                    if self._send_ctrl_c_via_console_attach(pid):
                        signal_sent = True
                        model_log_service.append_log("✅ CTRL_C sent successfully via console attach")
                    else:
                        self.logger.warning("Console attach method failed, trying fallback...")
                        model_log_service.append_log("⚠️ Console attach failed, trying fallback signal...")
                        # Fallback: try SIGTERM
                        try:
                            os.kill(pid, signal.SIGTERM)
                            signal_sent = True
                            model_log_service.append_log("✅ Sent SIGTERM as fallback")
                        except Exception as e:
                            self.logger.error(f"Fallback signal also failed: {e}")
                            model_log_service.append_log(f"❌ Fallback signal failed: {e}")
                else:
                    # Unix/Linux: Send SIGTERM to process group
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        signal_sent = True
                    except Exception as e:
                        self.logger.error(f"Failed to send SIGTERM: {e}")

                if not signal_sent:
                    error_msg = "Failed to send shutdown signal to process"
                    model_log_service.append_log(f"❌ {error_msg}")
                    return {
                        "status": self.status.ERROR,
                        "message": error_msg
                    }

                # Wait for process to terminate (allow time for cache saving)
                # llama-server may need significant time to save prefix_tree.bin for large KV caches
                _timeout = 30  # Increased timeout to allow proper KV cache saving
                try:
                    self.logger.info("Waiting for server to gracefully shutdown (saving KV cache)...")
                    model_log_service.append_log("Waiting for server to gracefully shutdown (saving KV cache)...")

                    # Check if process already terminated
                    if self.process.poll() is not None:
                        self.logger.info("Server already terminated")
                        model_log_service.append_log("✅ Server already terminated")
                    else:
                        # llama-server should respond to SIGTERM/SIGINT quickly (usually < 5s)
                        # Use async polling instead of blocking wait
                        for i in range(_timeout):
                            await asyncio.sleep(1)
                            if self.process.poll() is not None:
                                self.logger.info(f"Server shut down gracefully after {i+1} seconds")
                                model_log_service.append_log(f"✅ Server shut down gracefully after {i+1} seconds")
                                break
                        else:
                            # Timeout reached
                            raise subprocess.TimeoutExpired("", _timeout)

                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    self.logger.warning(f"Graceful shutdown timeout after {_timeout}s, force killing process")
                    self.logger.warning("⚠️ This will prevent prefix_tree.bin from being saved!")
                    model_log_service.append_log(f"⚠️ Graceful shutdown timeout after {_timeout}s, force killing process")
                    model_log_service.append_log("⚠️ This will prevent prefix_tree.bin from being saved!")

                    # Force kill the process
                    try:
                        if sys.platform == "win32":
                            self.process.kill()
                        else:
                            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        # Async wait for force kill
                        for i in range(2):
                            await asyncio.sleep(1)
                            if self.process.poll() is not None:
                                self.logger.info(f"Process force-killed after {i+1} seconds")
                                model_log_service.append_log(f"✅ Process force-killed after {i+1} seconds")
                                break
                        else:
                            self.logger.error("Process still running after force kill")
                            model_log_service.append_log("❌ Process still running after force kill")
                    except Exception as e:
                        self.logger.error(f"Failed to force kill process: {e}")
                        model_log_service.append_log(f"❌ Failed to force kill process: {e}")

                self.process = None

                # Check if prefix_tree.bin was created
                try:
                    cache_path = Path(self.config.cache_path)
                    resume_policy_mode = "reset (0)" if self.last_reset_mode else "resume (1)"
                    model_log_service.append_log(f"Checking for prefix_tree.bin (startup mode was: {resume_policy_mode})...")

                    # llama-server creates maestro_phison subdirectory
                    prefix_tree_file = cache_path / MAESTRO_CACHE_SUBDIR / "prefix_tree.bin"

                    # Fallback: also check directly in cache_path
                    if not prefix_tree_file.exists():
                        prefix_tree_file_alt = cache_path / "prefix_tree.bin"
                        if prefix_tree_file_alt.exists():
                            prefix_tree_file = prefix_tree_file_alt

                    if prefix_tree_file.exists():
                        file_size = prefix_tree_file.stat().st_size
                        self.logger.info(f"✅ prefix_tree.bin found: {prefix_tree_file} ({file_size} bytes)")
                        model_log_service.append_log(f"✅ prefix_tree.bin found at: {prefix_tree_file}")
                        model_log_service.append_log(f"   File size: {file_size:,} bytes")
                    else:
                        self.logger.warning(f"⚠️ prefix_tree.bin NOT found at: {prefix_tree_file}")
                        model_log_service.append_log(f"⚠️ prefix_tree.bin NOT found at: {prefix_tree_file}")
                        model_log_service.append_log(f"   Startup mode: {resume_policy_mode}")

                        if self.last_reset_mode:
                            model_log_service.append_log("   ⚠️ Model was started in RESET mode (--kv-cache-resume-policy 0)")
                            model_log_service.append_log("   ⚠️ In reset mode, llama-server may not save prefix_tree.bin")
                            model_log_service.append_log("   ℹ️ To enable prefix_tree.bin saving, use RESUME mode (policy 1)")
                        else:
                            model_log_service.append_log("   Model was in RESUME mode - prefix_tree.bin should have been saved")
                            model_log_service.append_log("   Possible reasons for missing file:")
                            model_log_service.append_log("   1. No inference was performed (no KV cache to save)")
                            model_log_service.append_log("   2. CTRL_C signal not properly handled")
                            model_log_service.append_log("   3. Insufficient time before forced shutdown")
                except Exception as e:
                    self.logger.error(f"Error checking prefix_tree.bin: {e}")
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

            model_log_service.append_log("✅ Server stopped successfully")
            return {
                "status": self.status.SUCCESS,
                "message": "Server stopped successfully"
            }

        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")
            error_msg = f"Failed to stop server: {str(e)}"
            model_log_service.append_log(f"ERROR: {error_msg}")
            return {
                "status": self.status.ERROR,
                "message": error_msg
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
