"""
Logs router for viewing model server logs
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from services.model_log import model_log_service


router = APIRouter(tags=["logs"])


class LogSessionInfo(BaseModel):
    """Log session information"""
    session_id: str
    log_file_path: str
    start_time: str


class LogFileInfo(BaseModel):
    """Log file information"""
    filename: str
    path: str
    size: int
    created: str
    modified: str


@router.get("/logs/current", response_model=LogSessionInfo)
async def get_current_session():
    """
    Get current log session information

    Returns:
        Current log session details
    """
    if not model_log_service.current_session:
        raise HTTPException(status_code=404, detail="No active log session")

    return LogSessionInfo(
        session_id=model_log_service.current_session.session_id,
        log_file_path=model_log_service.current_session.log_file_path,
        start_time=model_log_service.current_session.start_time.isoformat()
    )


@router.get("/logs/recent")
async def get_recent_logs(
    lines: int = Query(100, ge=1, le=1000),
    pattern: Optional[str] = Query(None, description="Regex pattern to filter logs")
):
    """
    Get recent log lines from current session

    Args:
        lines: Number of recent lines to retrieve (default: 100, max: 1000)
        pattern: Optional regex pattern to filter logs (e.g., "\\[MDW\\]\\[Info\\]\\[Runtime\\]")

    Returns:
        List of recent log lines (filtered if pattern provided)
    """
    if not model_log_service.current_session:
        raise HTTPException(status_code=404, detail="No active log session")

    log_lines = model_log_service.get_recent_logs(lines, pattern)

    return {
        "session_id": model_log_service.current_session.session_id,
        "lines_count": len(log_lines),
        "pattern": pattern,
        "logs": log_lines
    }


@router.get("/logs/server")
async def get_server_logs(
    lines: int = Query(100, ge=1, le=1000),
    pattern: Optional[str] = Query(None, description="Regex pattern to filter logs")
):
    """
    Get llama-server logs (actual model server output)

    Args:
        lines: Number of recent lines to retrieve (default: 100, max: 1000)
        pattern: Optional regex pattern to filter logs (e.g., "\\[MDW\\]\\[Info\\]\\[Runtime\\]")

    Returns:
        List of recent log lines from llama-server (filtered if pattern provided)
    """
    if not model_log_service.current_session:
        raise HTTPException(status_code=404, detail="No active log session")

    log_lines = model_log_service.get_server_logs(lines, pattern)

    return {
        "session_id": model_log_service.current_session.session_id,
        "lines_count": len(log_lines),
        "pattern": pattern,
        "logs": log_lines
    }


@router.get("/logs/sessions", response_model=List[LogFileInfo])
async def list_log_sessions():
    """
    List all log session files

    Returns:
        List of all log files with metadata
    """
    return model_log_service.list_log_sessions()

