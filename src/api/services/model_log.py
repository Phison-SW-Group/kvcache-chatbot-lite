"""
Model log service for managing model server logs with unique session IDs
"""
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class LogSession:
    """Log session information"""
    session_id: str
    log_file_path: str
    start_time: datetime


class ModelLogService:
    """
    Service for managing model server logs with unique session identifiers
    Each backend startup creates a new log session with unique ID
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize model log service
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[LogSession] = None
        self.logger = logging.getLogger(__name__)
    
    def create_session(self) -> LogSession:
        """
        Create a new log session with unique ID
        
        Returns:
            LogSession object with session info
        """
        # Generate unique session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # Short unique identifier
        session_id = f"{timestamp}_{unique_id}"
        
        # Create log file path
        log_filename = f"model_{session_id}.log"
        log_file_path = self.log_dir / log_filename
        
        # Create session
        self.current_session = LogSession(
            session_id=session_id,
            log_file_path=str(log_file_path),
            start_time=datetime.now()
        )
        
        # Write session header to log file
        self._write_session_header()
        
        self.logger.info(f"Created new log session: {session_id}")
        self.logger.info(f"Log file: {log_file_path}")
        
        return self.current_session
    
    def _write_session_header(self):
        """Write session information header to log file"""
        if not self.current_session:
            return
        
        header = [
            "=" * 80,
            f"Model Server Log Session",
            f"Session ID: {self.current_session.session_id}",
            f"Start Time: {self.current_session.start_time.isoformat()}",
            "=" * 80,
            ""
        ]
        
        with open(self.current_session.log_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header))
    
    def get_current_log_path(self) -> Optional[str]:
        """
        Get current session log file path
        
        Returns:
            Path to current log file or None if no session
        """
        return self.current_session.log_file_path if self.current_session else None
    
    def get_session_id(self) -> Optional[str]:
        """
        Get current session ID
        
        Returns:
            Current session ID or None if no session
        """
        return self.current_session.session_id if self.current_session else None
    
    def append_log(self, message: str):
        """
        Append a log message to the current session log file
        
        Args:
            message: Log message to append
        """
        if not self.current_session:
            self.logger.warning("No active log session, skipping log append")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.current_session.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            self.logger.error(f"Failed to write log: {e}")
    
    def get_recent_logs(self, lines: int = 100, pattern: Optional[str] = None) -> list[str]:
        """
        Get recent log lines from current session
        
        Args:
            lines: Number of recent lines to retrieve
            pattern: Optional regex pattern to filter logs
            
        Returns:
            List of recent log lines (filtered if pattern provided)
        """
        if not self.current_session:
            return []
        
        try:
            with open(self.current_session.log_file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
                # Apply pattern filter if provided
                if pattern:
                    import re
                    try:
                        regex = re.compile(pattern)
                        all_lines = [line for line in all_lines if regex.search(line)]
                    except re.error as e:
                        self.logger.error(f"Invalid regex pattern: {e}")
                        return [f"ERROR: Invalid regex pattern: {e}\n"]
                
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except Exception as e:
            self.logger.error(f"Failed to read logs: {e}")
            return []
    
    def get_server_logs(self, lines: int = 100, pattern: Optional[str] = None) -> list[str]:
        """
        Get logs from the llama-server log file (the actual model server logs)
        
        Args:
            lines: Number of recent lines to retrieve
            pattern: Optional regex pattern to filter logs
            
        Returns:
            List of recent log lines from llama-server (filtered if pattern provided)
        """
        if not self.current_session:
            return []
        
        # The llama-server writes to the same log file
        log_path = self.current_session.log_file_path
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
                # Apply pattern filter if provided
                if pattern:
                    import re
                    try:
                        regex = re.compile(pattern)
                        all_lines = [line for line in all_lines if regex.search(line)]
                    except re.error as e:
                        self.logger.error(f"Invalid regex pattern: {e}")
                        return [f"ERROR: Invalid regex pattern: {e}\n"]
                
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except Exception as e:
            self.logger.error(f"Failed to read server logs: {e}")
            return [f"ERROR: Failed to read logs: {e}\n"]
    
    def list_log_sessions(self) -> list[dict]:
        """
        List all log session files
        
        Returns:
            List of log file information
        """
        log_files = []
        
        for log_file in self.log_dir.glob("model_*.log"):
            try:
                stat = log_file.stat()
                log_files.append({
                    "filename": log_file.name,
                    "path": str(log_file),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error reading log file {log_file}: {e}")
        
        # Sort by creation time, newest first
        log_files.sort(key=lambda x: x["created"], reverse=True)
        
        return log_files


# Global model log service instance
model_log_service = ModelLogService()

