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

from config import settings


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
        
    def _validate_paths(self) -> bool:
        """Validate that required paths exist"""
        if not self.config.exe:
            self.logger.error("LLM_SERVER_EXE not configured")
            return False
            
        if not self.config.model_path:
            self.logger.error("LLM_SERVER_MODEL_NAME_OR_PATH not configured")
            return False
            
        exe_path = Path(self.config.exe)
        if not exe_path.exists():
            self.logger.error(f"Server executable not found: {exe_path}")
            return False
            
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return False
            
        return True
    
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
        if self.process is None:
            return False
        return self.process.poll() is None
    
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
            
            # Prepare command
            args = self._get_server_args(reset)
            working_dir = Path(self.config.exe).parent
            
            self.logger.info(f"Starting server with args: {' '.join(args)}")
            self.logger.info(f"Working directory: {working_dir}")
            
            # Start server process with output capture
            if sys.platform == "win32":
                # Windows: Use CREATE_NEW_PROCESS_GROUP to allow proper signal handling
                self.process = subprocess.Popen(
                    args,
                    cwd=working_dir,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                # Unix/Linux: Use preexec_fn to create new process group
                self.process = subprocess.Popen(
                    args,
                    cwd=working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setsid
                )
            
            # Wait a moment to check if process started successfully
            time.sleep(2)
            
            if self._is_running():
                return {
                    "status": "success",
                    "message": f"Model server started successfully (reset={reset})",
                    "pid": self.process.pid,
                    "port": self.config.port,
                    "command": " ".join(args)
                }
            else:
                # Try to get error output from the process
                try:
                    stdout, stderr = self.process.communicate(timeout=1)
                    error_output = stderr.strip() if stderr else "No error output available"
                except:
                    error_output = "Process failed to start (timeout)"
                
                return {
                    "status": "error",
                    "message": "Failed to start server process",
                    "details": {
                        "error_output": error_output,
                        "command": " ".join(args),
                        "working_dir": str(working_dir)
                    }
                }
                
        except FileNotFoundError as e:
            return {
                "status": "error",
                "message": f"Executable not found: {str(e)}",
                "details": {
                    "exe_path": self.config.exe,
                    "error_type": "FileNotFoundError"
                }
            }
        except PermissionError as e:
            return {
                "status": "error",
                "message": f"Permission denied: {str(e)}",
                "details": {
                    "exe_path": self.config.exe,
                    "error_type": "PermissionError"
                }
            }
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            return {
                "status": "error",
                "message": f"Failed to start server: {str(e)}",
                "details": {
                    "error_type": type(e).__name__,
                    "command": " ".join(self._get_server_args(reset)) if hasattr(self, 'config') else "Unknown"
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
                    "status": "success",
                    "message": "Server was not running"
                }
            
            self.logger.info(f"Stopping server process (PID: {self.process.pid})")
            
            if sys.platform == "win32":
                # Windows: Send CTRL_BREAK_EVENT to process group
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Unix/Linux: Send SIGTERM to process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait for process to terminate
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                self.logger.warning("Graceful shutdown failed, force killing process")
                if sys.platform == "win32":
                    self.process.kill()
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
            
            self.process = None
            
            return {
                "status": "success",
                "message": "Server stopped successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")
            return {
                "status": "error",
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
                "status": "stopped",
                "message": "Server is not running"
            }
        
        try:
            # Get process info
            process = psutil.Process(self.process.pid)
            return {
                "status": "running",
                "message": "Server is running",
                "pid": self.process.pid,
                "port": self.config.port,
                "memory_usage": process.memory_info().rss,
                "cpu_percent": process.cpu_percent(),
                "create_time": process.create_time()
            }
        except Exception as e:
            return {
                "status": "unknown",
                "message": f"Could not get status: {str(e)}"
            }


# Global model server instance
model_server = ModelServer()