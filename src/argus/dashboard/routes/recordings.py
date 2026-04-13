"""Continuous recording management API routes."""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from argus.dashboard.api_response import api_not_found, api_success

logger = structlog.get_logger()
router = APIRouter()


@router.get("/{camera_id}/segments")
async def list_segments(
    request: Request,
    camera_id: str,
    start: str | None = Query(None, description="ISO date start filter"),
    end: str | None = Query(None, description="ISO date end filter"),
) -> JSONResponse:
    """List recording segments for a camera."""
    manager = getattr(request.app.state, "recording_manager", None)
    if manager is None:
        return api_not_found("Continuous recording not enabled")

    from datetime import datetime, timezone

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None

    segments = manager.get_segments(camera_id, start=start_dt, end=end_dt)
    return api_success([
        {
            "camera_id": s.camera_id,
            "path": str(s.segment_path),
            "filename": s.segment_path.name,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "end_time": s.end_time.isoformat() if s.end_time else None,
            "frame_count": s.frame_count,
            "file_size_mb": round(s.file_size_bytes / 1_048_576, 1) if s.file_size_bytes else 0,
        }
        for s in segments
    ])


@router.get("/{camera_id}/segment/{filename}/video")
async def serve_segment_video(
    request: Request,
    camera_id: str,
    filename: str,
) -> FileResponse | JSONResponse:
    """Serve a recording segment MP4 with HTTP Range support."""
    manager = getattr(request.app.state, "recording_manager", None)
    if manager is None:
        return api_not_found("Continuous recording not enabled")

    # Security: reject path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return api_not_found("Invalid filename")

    # Find the file and verify it resolves within the output directory
    output_dir = manager._output_dir
    matches = list(output_dir.rglob(f"{camera_id}/{filename}"))
    if not matches:
        matches = list(output_dir.rglob(filename))

    if not matches or not matches[0].exists():
        return api_not_found(f"Segment {filename} not found")

    # Verify resolved path is within output directory (symlink protection)
    resolved = matches[0].resolve()
    if not str(resolved).startswith(str(output_dir.resolve())):
        return api_not_found("Invalid path")

    return FileResponse(
        path=str(matches[0]),
        media_type="video/mp4",
        filename=filename,
    )


@router.get("/storage-stats")
async def storage_stats(request: Request) -> JSONResponse:
    """Return storage usage statistics."""
    manager = getattr(request.app.state, "recording_manager", None)
    retention = getattr(request.app.state, "retention_manager", None)

    stats = {}
    if manager is not None:
        stats["recording"] = manager.get_storage_stats()
    if retention is not None:
        stats["retention"] = retention.get_retention_stats()

    return api_success(stats)


@router.post("/cleanup")
async def trigger_cleanup(request: Request) -> JSONResponse:
    """Trigger manual retention cleanup."""
    retention = getattr(request.app.state, "retention_manager", None)
    if retention is None:
        return api_not_found("Retention manager not enabled")

    result = await asyncio.to_thread(retention.run_cleanup)
    return api_success(result)
