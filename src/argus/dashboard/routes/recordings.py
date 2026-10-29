"""Continuous recording management API routes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import formatdate
import hashlib
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import unicodedata
from urllib.parse import quote

import anyio
import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from argus.dashboard.api_response import (
    api_conflict, api_forbidden, api_not_found, api_success, api_unavailable,
    api_validation_error,
)
from argus.dashboard.auth import require_permission
from argus.dashboard.routes.replay import _parse_single_byte_range, _range_not_satisfiable

logger = structlog.get_logger()
router = APIRouter()


class _RecordingVideoResponse(StreamingResponse):
    """Own the opened file even when ASGI send fails before iteration starts."""

    def __init__(self, content, *, video_file, **kwargs):
        self._video_file = video_file
        super().__init__(content, **kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # A lazy generator's finally and a BackgroundTask are insufficient
            # if response.start raises or is cancelled before the first next().
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self._video_file.close)


def _safe_component(value: str, *, camera: bool = False) -> bool:
    # Existing camera IDs and archive names are not an ASCII-only contract.
    # Permit safe single path components, but never path/glob/ADS aliases or
    # names that Windows silently normalizes into a different component.
    return (isinstance(value, str) and 0 < len(value) <= 255
            and value not in {".", ".."} and ".." not in value
            and value.rstrip(" .") == value
            and not any(character in '/\\:*?[]<>"|' or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
                        for character in value)
            and not PureWindowsPath(value).is_reserved()
            and (camera or Path(value).suffix.lower() == ".mp4"))


def _valid_date(value: str) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        return False
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except ValueError:
        return False


def _is_link_or_reparse(path: Path) -> bool:
    """Reject symlinks and Windows junction/reparse aliases, including in-root ones."""
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _verified_segment(root: Path, path: Path, camera_id: str) -> Path | None:
    """Accept only real root/date/exact-camera/filename files, never glob aliases."""
    try:
        parts = path.relative_to(root).parts
        if (len(parts) != 3 or not _valid_date(parts[0]) or parts[1] != camera_id
                or not _safe_component(parts[2])):
            return None
        current = root
        for part in parts:
            current = current / part
            if _is_link_or_reparse(current):
                return None
        resolved = path.resolve(strict=True)
        info = path.lstat()
        if (not resolved.is_relative_to(root) or resolved.relative_to(root).parts != parts
                or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _segment_paths(manager, camera_id: str, *, filename: str | None = None,
                   date: str | None = None) -> list[Path]:
    """Scan the recorder's bounded layout without following descendant links."""
    if (not _safe_component(camera_id, camera=True)
            or filename is not None and not _safe_component(filename)
            or date is not None and not _valid_date(date)):
        return []
    root = Path(manager._output_dir).resolve(strict=True)
    if not root.is_dir():
        return []
    dates = [root / date] if date is not None else sorted(root.iterdir())
    result = []
    for date_dir in dates:
        if not _valid_date(date_dir.name):
            continue
        camera_dir = date_dir / camera_id
        try:
            if (_is_link_or_reparse(date_dir) or not date_dir.is_dir()
                    or _is_link_or_reparse(camera_dir) or not camera_dir.is_dir()):
                continue
            candidates = [camera_dir / filename] if filename is not None else sorted(camera_dir.iterdir())
            for candidate in candidates:
                verified = _verified_segment(root, candidate, camera_id)
                if verified is not None:
                    result.append(verified)
        except (OSError, RuntimeError):
            continue
    return result


def _filter_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


@router.get("/{camera_id}/segments")
def list_segments(
    request: Request,
    camera_id: str,
    start: str | None = Query(None, description="ISO date start filter"),
    end: str | None = Query(None, description="ISO date end filter"),
) -> JSONResponse:
    """List recording segments for a camera."""
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")
    if not _safe_component(camera_id, camera=True):
        return api_not_found("Invalid camera identifier")
    manager = getattr(request.app.state, "recording_manager", None)
    if manager is None:
        return api_not_found("Continuous recording not enabled")

    try:
        start_dt, end_dt = _filter_timestamp(start), _filter_timestamp(end)
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("reversed timestamp range")
    except ValueError:
        return api_validation_error("录像时间范围无效")
    try:
        segments = []
        root = Path(manager._output_dir).resolve(strict=True)
        for path in _segment_paths(manager, camera_id):
            try:
                info = path.lstat()
                if _verified_segment(root, path, camera_id) is None:
                    continue
            except FileNotFoundError:
                # Retention may remove one segment while listing its peers.
                continue
            mtime = datetime.fromtimestamp(info.st_mtime, tz=timezone.utc)
            if (start_dt is not None and mtime < start_dt) or (end_dt is not None and mtime > end_dt):
                continue
            date = path.parent.parent.name
            segments.append({
                "camera_id": camera_id, "filename": path.name, "date": date,
                "video_url": f"/api/recordings/{quote(camera_id, safe='')}/segment/{quote(path.name, safe='')}/video?date={date}",
                "start_time": mtime.isoformat(), "end_time": None, "frame_count": 0,
                "file_size_mb": round(info.st_size / 1_048_576, 1),
            })
        return api_success(segments)
    except FileNotFoundError:
        return api_success([])
    except (OSError, RuntimeError, TypeError, ValueError):
        return api_unavailable("录像目录暂不可用")


@router.get("/{camera_id}/segment/{filename}/video", response_model=None)
def serve_segment_video(
    request: Request,
    camera_id: str,
    filename: str,
    date: str | None = Query(None, description="Exact recording date (YYYY-MM-DD); required for duplicate names"),
):
    """Serve a recording segment MP4 with HTTP Range support."""
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")
    if not _safe_component(camera_id, camera=True) or not _safe_component(filename):
        return api_not_found("Invalid recording identifier")
    if date is not None and not _valid_date(date):
        return api_validation_error("录像日期无效")
    manager = getattr(request.app.state, "recording_manager", None)
    if manager is None:
        return api_not_found("Continuous recording not enabled")

    video_file = None
    try:
        matches = _segment_paths(manager, camera_id, filename=filename, date=date)
        if not matches:
            return api_not_found("录像不存在")
        if len(matches) != 1:
            return api_conflict("录像名称跨日期重复，请指定 date")
        root = Path(manager._output_dir).resolve(strict=True)
        path = matches[0]
        video_file = path.open("rb")
        info = os.fstat(video_file.fileno())
        # Bind the stream to the already-open file. A late path/link swap must
        # not redirect Starlette's deferred file open outside this camera.
        if (_verified_segment(root, path, camera_id) != path
                or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not os.path.samestat(info, path.stat())):
            video_file.close()
            return api_not_found("录像身份已变化")
    except (OSError, RuntimeError, TypeError, ValueError):
        if video_file is not None:
            video_file.close()
        return api_not_found("录像不存在或暂不可用")

    try:
        size = info.st_size
        # Metadata is useful for revalidation, but it is not a content digest
        # or proof that the segment is sealed. The recorder writes this path.
        etag = 'W/"' + hashlib.sha256(
            f"{info.st_dev}:{info.st_ino}:{info.st_mtime_ns}:{size}".encode("ascii")
        ).hexdigest() + '"'
        modified = formatdate(info.st_mtime, usegmt=True)
        encoded_filename = quote(filename, safe="")
        disposition = (f'inline; filename="{filename}"' if encoded_filename == filename
                       else f"inline; filename*=utf-8''{encoded_filename}")
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, no-store",
                   "X-Content-Type-Options": "nosniff", "Last-Modified": modified,
                   "ETag": etag, "Content-Disposition": disposition}
        range_header = request.headers.get("range")
        if_range = request.headers.get("if-range")
        # Neither the weak metadata ETag nor second-precision Last-Modified
        # proves an immutable segment. Without a strong validator, If-Range
        # must not splice bytes from a changed file into an older download.
        if if_range is not None:
            range_header = None
        start, length, status_code = 0, size, 200
        if range_header:
            byte_range = _parse_single_byte_range(range_header, size)
            if byte_range is None:
                video_file.close()
                return _range_not_satisfiable(size)
            start, end = byte_range
            length, status_code = end - start + 1, 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(length)

        def stream_file():
            try:
                video_file.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = video_file.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
            finally:
                video_file.close()

        return _RecordingVideoResponse(stream_file(), video_file=video_file,
            status_code=status_code, media_type="video/mp4", headers=headers)
    except BaseException:
        # Header/response construction can fail before the response owns it.
        video_file.close()
        raise


@router.get("/storage-stats")
def storage_stats(request: Request) -> JSONResponse:
    """Return storage usage statistics."""
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")
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
    """Trigger manual retention cleanup. Requires admin role."""
    from argus.dashboard.auth import require_role
    if not require_role(request, "admin"):
        return api_forbidden("需要管理员权限")

    retention = getattr(request.app.state, "retention_manager", None)
    if retention is None:
        return api_not_found("Retention manager not enabled")

    result = await asyncio.to_thread(retention.run_cleanup)
    return api_success(result)
