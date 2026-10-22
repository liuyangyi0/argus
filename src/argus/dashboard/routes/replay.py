"""Alert replay API for multi-track timeline playback (UX v2 §1.3).

Provides endpoints to retrieve recording metadata, signal timeseries,
individual frames, MP4 video, and historical reference frames for alert replay.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response, StreamingResponse

from argus.dashboard.api_contracts import (
    ReplayClipListResponse,
    ReplayClipMutationResponse,
    ReplayClipRequest,
    ReplayMetadataResponse,
    ReplayPinFrameRequest,
    ReplayPinFrameResponse,
    ReplaySignalsResponse,
)
from argus.dashboard.api_response import (
    ApiEnvelope,
    api_error,
    api_forbidden,
    api_not_found,
    api_success,
    api_typed_success,
    api_unavailable,
    api_validation_error,
    error_code_for_status,
)
from argus.dashboard.auth import require_permission
from argus.storage.models import AlertRecord

router = APIRouter()

# Maximum cameras to return in a storyboard; keeps the grid usable.
_MAX_STORYBOARD_CAMERAS = 4
# Window used to match a correlation-partner alert to the primary alert
# when no event_group_id linkage exists.
_CORRELATION_WINDOW_SECONDS = 5.0
_PRIVATE_MEDIA_CACHE = "private, max-age=3600"
_MUTABLE_MEDIA_CACHE = "private, no-store"
_RANGE_CHUNK_SIZE = 64 * 1024


def _video_cache_control(store, alert_id: str) -> str:
    """Never cache the mutable pre-trigger artifact as a final MP4."""
    try:
        metadata = store.load_metadata(alert_id)
    except Exception:
        return _MUTABLE_MEDIA_CACHE
    if isinstance(metadata, dict) and metadata.get("status") == "complete":
        return _PRIVATE_MEDIA_CACHE
    return _MUTABLE_MEDIA_CACHE


def _parse_single_byte_range(
    range_header: str,
    file_size: int,
) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range, returning an inclusive ``(start, end)``.

    Multi-range responses are intentionally unsupported.  Malformed,
    unsatisfiable, and empty-file ranges return ``None`` so the caller can
    produce a standards-compliant 416 response instead of silently serving a
    different part of the file.
    """
    if file_size <= 0:
        return None

    unit, separator, raw_spec = range_header.strip().partition("=")
    if separator != "=" or unit.strip().lower() != "bytes":
        return None

    spec = raw_spec.strip()
    if not spec or "," in spec or spec.count("-") != 1:
        return None

    raw_start, raw_end = (part.strip() for part in spec.split("-", 1))
    if not raw_start and not raw_end:
        return None

    try:
        if not raw_start:
            # Suffix range: the final N bytes.  A suffix longer than the file
            # selects the complete representation; zero is unsatisfiable.
            if not raw_end.isdigit():
                return None
            suffix_length = int(raw_end)
            if suffix_length <= 0:
                return None
            return max(file_size - suffix_length, 0), file_size - 1

        if not raw_start.isdigit():
            return None
        start = int(raw_start)
        if start < 0 or start >= file_size:
            return None

        if not raw_end:
            return start, file_size - 1

        if not raw_end.isdigit():
            return None
        end = int(raw_end)
        if end < start:
            return None
        return start, min(end, file_size - 1)
    except ValueError:
        return None


def _range_not_satisfiable(file_size: int):
    """Return a unified JSON error with RFC 7233 range metadata."""
    return api_error(
        error_code_for_status(416),
        "请求的字节范围无效",
        status_code=416,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes */{file_size}",
            "Cache-Control": "private, no-store",
        },
    )


def _get_recording_store(request: Request):
    """Get the AlertRecordingStore from app state."""
    return getattr(request.app.state, "recording_store", None)


def _get_db(request: Request):
    """Get the Database from app state."""
    return getattr(request.app.state, "db", None)


@router.get(
    "/{alert_id}/metadata",
    response_model=ApiEnvelope[ReplayMetadataResponse],
)
def replay_metadata(request: Request, alert_id: str):
    """Return recording metadata for an alert.

    Response includes video_url for direct MP4 playback.
    """
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    metadata = store.load_metadata(alert_id)
    if metadata is None:
        return api_not_found("录像不存在")

    # Normalize optional fields written by newer recorders so archived replay
    # artifacts keep the same public DTO after an upgrade.
    metadata.setdefault("linked_alert_id", None)
    metadata.setdefault("frame_identity", None)
    metadata.setdefault("pinned_frames", [])

    # Enrich with video URL for frontend.
    metadata["video_url"] = f"/api/replay/{alert_id}/video"

    return api_typed_success(metadata, ReplayMetadataResponse)


@router.get(
    "/{alert_id}/signals",
    response_model=ApiEnvelope[ReplaySignalsResponse],
)
def replay_signals(request: Request, alert_id: str):
    """Return signal timeseries for the 5 replay tracks.

    Response: {timestamps[], anomaly_scores[], simplex_scores[],
               cusum_evidence: {zone: float[]}, yolo_persons[],
               operator_actions[], key_frames[]}
    """
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    signals = store.load_signals(alert_id)
    if signals is None:
        return api_not_found("录像不存在")

    # Backfill fields which did not exist in early replay artifacts.  Keeping
    # list lengths aligned lets clients render each track independently
    # without interpreting an absent key as a successful empty result.
    frame_count = len(signals.get("timestamps", []))
    signals.setdefault("frame_numbers", list(range(frame_count)))
    signals.setdefault("frame_identities", [None] * frame_count)
    signals.setdefault("yolo_boxes", [[] for _ in range(frame_count)])
    signals.setdefault("candidate_records", [[] for _ in range(frame_count)])
    signals.setdefault("candidate_events", [])
    signals.setdefault("has_heatmaps", False)
    signals.setdefault("trajectory_points", {})

    # Enrich with operator actions from alert workflow history
    db = _get_db(request)
    operator_actions = []
    if db is not None:
        alert_record = db.get_alert(alert_id)
        if alert_record is not None:
            # Include workflow transitions as operator actions
            if alert_record.acknowledged and alert_record.acknowledged_by:
                operator_actions.append({
                    "timestamp": alert_record.resolved_at.timestamp() if alert_record.resolved_at else 0,
                    "user": alert_record.acknowledged_by,
                    "action": alert_record.workflow_status,
                })
    signals["operator_actions"] = operator_actions

    # Auto-generate key frames
    key_frames = _compute_key_frames(signals)
    signals["key_frames"] = key_frames

    # Ensure persisted clip ranges are exposed to the frontend even for older
    # recordings saved before the clips field was introduced.
    clips = store.list_clips(alert_id)
    signals["clips"] = clips if clips is not None else []

    return api_typed_success(signals, ReplaySignalsResponse)


@router.get("/{alert_id}/video")
def replay_video(request: Request, alert_id: str):
    """Return the MP4 video file with HTTP Range support for seeking.

    The browser <video> element uses Range requests for efficient playback.
    Proper 206 Partial Content responses are required for seeking and
    progressive loading (especially when moov atom is at end of file).
    """
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    # Resolve mutable/complete state before opening the artifact. Store
    # metadata access is locked with finalization, which avoids lock↔open
    # inversion on platforms that prohibit renaming an open file.
    cache_control = _video_cache_control(store, alert_id)
    video_file = None
    for _ in range(2):
        video_path = store.get_video_path(alert_id)
        if video_path is None:
            break
        try:
            # Open before returning the response so a recording→complete
            # rename cannot invalidate a lazily opened path.
            video_file = video_path.open("rb")
            break
        except FileNotFoundError:
            # Finalization may have renamed pre.mp4 after path resolution.
            continue
        except (IsADirectoryError, PermissionError):
            break
    if video_file is None:
        return api_not_found("视频文件不存在")

    try:
        file_size = os.fstat(video_file.fileno()).st_size
    except OSError:
        video_file.close()
        return api_not_found("视频文件不存在")
    range_header = request.headers.get("range")

    def stream_open_file(start: int, content_length: int):
        try:
            video_file.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = video_file.read(min(_RANGE_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        finally:
            video_file.close()

    if range_header:
        byte_range = _parse_single_byte_range(range_header, file_size)
        if byte_range is None:
            video_file.close()
            return _range_not_satisfiable(file_size)
        start, end = byte_range
        content_length = end - start + 1

        return StreamingResponse(
            stream_open_file(start, content_length),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": cache_control,
            },
        )

    # No Range header — stream the already-open file descriptor.
    return StreamingResponse(
        stream_open_file(0, file_size),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": cache_control,
        },
    )


@router.get("/{alert_id}/frame/{index}")
def replay_frame(request: Request, alert_id: str, index: int):
    """Return a single JPEG frame by index (extracted from MP4).

    Used for heatmap compositing, pin-frame thumbnails, and reference comparison.
    """
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    if index < 0:
        return api_validation_error("帧索引不能为负")

    frame_bytes = store.load_frame(alert_id, index)
    if frame_bytes is None:
        return api_not_found("录像帧不存在")

    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": _PRIVATE_MEDIA_CACHE},
    )


@router.get("/{alert_id}/heatmap/{index}")
def replay_heatmap_frame(request: Request, alert_id: str, index: int):
    """Return a single heatmap overlay JPEG by index."""
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    if index < 0:
        return api_validation_error("帧索引不能为负")

    heatmap_bytes = store.load_heatmap_frame(alert_id, index)
    if heatmap_bytes is None:
        return api_not_found("热力图帧不存在")

    return Response(
        content=heatmap_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": _PRIVATE_MEDIA_CACHE},
    )


@router.get("/{alert_id}/reference")
def replay_reference(
    request: Request,
    alert_id: str,
    date: str = Query(default="", description="Reference date YYYY-MM-DD"),
    time_str: str = Query(default="", alias="time", description="Reference time HH:MM:SS"),
    frame_offset_seconds: float = Query(
        default=0.0,
        description="Seconds to shift the reference frame relative to trigger time",
    ),
):
    """Return a historical reference frame from the same camera at the same time of day.

    If no specific date is given, defaults to yesterday. The optional
    ``frame_offset_seconds`` shifts the target timestamp so operators can scrub
    through the historical recording (e.g. ``+2.5`` reads the reference frame
    2.5 seconds after trigger time).
    Returns {available: bool, frame_base64: str|null, source_date: str}.
    """
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_success({"available": False, "frame_base64": None, "source_date": ""})

    # Get the alert's metadata to find camera_id and trigger time
    metadata = store.load_metadata(alert_id)
    if metadata is None:
        return api_success({"available": False, "frame_base64": None, "source_date": ""})

    camera_id = metadata["camera_id"]
    trigger_ts = metadata["trigger_timestamp"]
    trigger_dt = datetime.fromtimestamp(trigger_ts, tz=timezone.utc)

    # Determine reference date
    if date:
        try:
            ref_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return api_validation_error("日期格式无效，请使用 YYYY-MM-DD")
    else:
        ref_date = trigger_dt - timedelta(days=1)

    # Build the reference timestamp. An explicit ``time`` query selects a
    # wall-clock point; otherwise preserve the alert's trigger time of day.
    if time_str:
        try:
            requested_time = datetime.strptime(time_str, "%H:%M:%S")
        except ValueError:
            return api_validation_error("时间格式无效，请使用 HH:MM:SS")
        ref_hour = requested_time.hour
        ref_minute = requested_time.minute
        ref_second = requested_time.second
        ref_microsecond = 0
    else:
        ref_hour = trigger_dt.hour
        ref_minute = trigger_dt.minute
        ref_second = trigger_dt.second
        ref_microsecond = trigger_dt.microsecond

    # Shift by the requested offset so operators can scrub around that point.
    ref_dt = ref_date.replace(
        hour=ref_hour,
        minute=ref_minute,
        second=ref_second,
        microsecond=ref_microsecond,
    )
    if frame_offset_seconds:
        ref_dt = ref_dt + timedelta(seconds=float(frame_offset_seconds))
    ref_date_str = ref_dt.strftime("%Y-%m-%d")

    # Search for recordings from the same camera on the reference date
    archive_dir = store.archive_dir / ref_date_str / camera_id
    if not archive_dir.exists():
        return api_success({
            "available": False,
            "frame_base64": None,
            "source_date": ref_date_str,
        })

    # Find the recording whose span covers (or is closest to) ref_ts, then
    # extract the frame inside that recording aligned to ref_ts. This lets the
    # frontend scrub the historical recording via frame_offset_seconds.
    best_rec_dir: Path | None = None
    best_meta: dict | None = None
    best_distance = float("inf")
    ref_ts = ref_dt.timestamp()

    for rec_dir in archive_dir.iterdir():
        if not rec_dir.is_dir():
            continue
        meta_path = rec_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            continue

        start_ts = meta.get("start_timestamp", meta.get("trigger_timestamp", 0))
        end_ts = meta.get("end_timestamp", meta.get("trigger_timestamp", 0))
        if start_ts <= ref_ts <= end_ts:
            dist = 0.0
        else:
            dist = min(abs(start_ts - ref_ts), abs(end_ts - ref_ts))
        if dist < best_distance:
            best_distance = dist
            best_rec_dir = rec_dir
            best_meta = meta

    best_frame: bytes | None = None
    if best_rec_dir is not None and best_meta is not None:
        start_ts = best_meta.get(
            "start_timestamp", best_meta.get("trigger_timestamp", 0)
        )
        end_ts = best_meta.get(
            "end_timestamp", best_meta.get("trigger_timestamp", 0)
        )
        frame_count = int(best_meta.get("frame_count", 0) or 0)
        fps = float(best_meta.get("fps", 15) or 15)
        trigger_idx = int(best_meta.get("trigger_frame_index", 0) or 0)

        # Pick the in-recording frame index that best aligns with ref_ts.
        if frame_count > 0:
            if end_ts > start_ts:
                rel = (ref_ts - start_ts) / (end_ts - start_ts)
                target_idx = int(round(rel * (frame_count - 1)))
            else:
                target_idx = int(round((ref_ts - start_ts) * fps))
            target_idx = max(0, min(frame_count - 1, target_idx))
        else:
            target_idx = trigger_idx

        # If the archived downsampled form is all we have, fall back to the
        # preserved trigger frame regardless of offset.
        trigger_frame_path = best_rec_dir / "trigger_frame.jpg"
        from argus.storage.alert_recording import _find_video_file
        video_path = _find_video_file(best_rec_dir)
        if video_path is not None:
            from argus.core.video_encoder import extract_frame_jpeg
            best_frame = extract_frame_jpeg(video_path, target_idx)
        if best_frame is None and trigger_frame_path.exists():
            best_frame = trigger_frame_path.read_bytes()

    if best_frame is not None:
        return api_success({
            "available": True,
            "frame_base64": base64.b64encode(best_frame).decode("ascii"),
            "source_date": ref_date_str,
        })

    return api_success({
        "available": False,
        "frame_base64": None,
        "source_date": ref_date_str,
    })


@router.post(
    "/{alert_id}/pin-frame",
    response_model=ApiEnvelope[ReplayPinFrameResponse],
)
async def pin_frame(
    request: Request,
    alert_id: str,
    body: ReplayPinFrameRequest,
):
    """Pin a frame as a key frame bookmark (operator action).

    Request body: {"index": int, "label": str}
    """
    if not require_permission(request, "pin_frame"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    if not store.pin_frame(alert_id, body.index, body.label):
        return api_not_found("录像不存在")

    return api_typed_success({"success": True}, ReplayPinFrameResponse)


@router.post(
    "/{alert_id}/clips",
    response_model=ApiEnvelope[ReplayClipMutationResponse],
)
async def add_clip(
    request: Request,
    alert_id: str,
    body: ReplayClipRequest,
):
    """Persist an operator-marked clip range for this recording.

    Request body: ``{"start_index": int, "end_index": int, "label"?: str}``.
    The clip is appended to ``signals.json`` so it survives page reload.
    """
    if not require_permission(request, "pin_frame"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    start_index = body.start_index
    end_index = body.end_index
    label = body.label

    metadata = store.load_metadata(alert_id)
    if metadata is None:
        return api_not_found("录像不存在")
    frame_count = int(metadata.get("frame_count", 0) or 0)
    if frame_count and (start_index >= frame_count or end_index >= frame_count):
        return api_validation_error(
            f"帧索引越界，录像共 {frame_count} 帧"
        )

    clip = store.add_clip(alert_id, start_index, end_index, label)
    if clip is None:
        return api_not_found("录像不存在")

    clips = store.list_clips(alert_id) or []
    return api_typed_success(
        {"clip": clip, "clips": clips},
        ReplayClipMutationResponse,
    )


@router.delete(
    "/{alert_id}/clips/{index}",
    response_model=ApiEnvelope[ReplayClipListResponse],
)
def delete_clip(request: Request, alert_id: str, index: int):
    """Remove a persisted clip by its array index."""
    if not require_permission(request, "pin_frame"):
        return api_forbidden("权限不足")

    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    if index < 0:
        return api_validation_error("clip 索引不能为负")

    if not store.delete_clip(alert_id, index):
        return api_not_found("clip 不存在")

    clips = store.list_clips(alert_id) or []
    return api_typed_success({"clips": clips}, ReplayClipListResponse)


def _alert_trigger_timestamp(alert: AlertRecord) -> float:
    """Epoch-seconds trigger timestamp for an alert record.

    AlertRecord stores ``timestamp`` as a naive ``datetime`` (written via
    :meth:`datetime.utcnow`-style helpers across the codebase), so we read it
    as UTC to produce a stable epoch value. This is what downstream clients
    use to align videos across cameras.
    """
    ts = alert.timestamp
    if ts is None:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


def _storyboard_item(primary_ts: float, alert: AlertRecord) -> dict:
    """Shape a single AlertRecord as a storyboard camera entry."""
    ts = _alert_trigger_timestamp(alert)
    return {
        "alert_id": alert.alert_id,
        "camera_id": alert.camera_id,
        "trigger_timestamp": ts,
        "metadata_url": f"/api/replay/{alert.alert_id}/metadata",
        "video_url": f"/api/replay/{alert.alert_id}/video",
        "signals_url": f"/api/replay/{alert.alert_id}/signals",
        "trigger_offset_s": ts - primary_ts,
    }


@router.get("/storyboard/{alert_id}")
def replay_storyboard(request: Request, alert_id: str):
    """Return a synchronised multi-camera storyboard for an alert.

    The storyboard bundles the primary alert with any correlated or
    event-grouped alerts from *other* cameras so the frontend can play them
    on a shared master timeline. Each entry includes the per-camera
    ``trigger_offset_s`` (seconds from the primary alert) used for alignment.

    Resolution rules (deduped, capped at :data:`_MAX_STORYBOARD_CAMERAS`):

    1. The primary alert itself (always first, offset 0).
    2. If ``correlation_partner`` is set, the closest alert from that camera
       whose trigger falls within ±5 s of the primary alert.
    3. If ``event_group_id`` is set, all alerts in the same event group from
       cameras different from the primary's camera.
    """
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")

    db = _get_db(request)
    if db is None:
        return api_unavailable("数据库未配置")

    primary = db.get_alert(alert_id)
    if primary is None:
        return api_not_found("告警不存在")

    primary_ts = _alert_trigger_timestamp(primary)
    items: list[dict] = [_storyboard_item(primary_ts, primary)]
    seen_alert_ids: set[str] = {primary.alert_id}
    seen_camera_ids: set[str] = {primary.camera_id}

    # Rule 2: correlation_partner — find the closest matching alert from the
    # partner camera within the ±5 s window. We search the DB directly to
    # avoid loading the whole alerts table.
    if primary.correlation_partner:
        with db.get_session() as session:
            from sqlalchemy import select
            candidates = session.scalars(
                select(AlertRecord)
                .where(AlertRecord.camera_id == primary.correlation_partner)
                .where(AlertRecord.alert_id != primary.alert_id)
            ).all()
        best: AlertRecord | None = None
        best_delta = _CORRELATION_WINDOW_SECONDS
        for cand in candidates:
            delta = abs(_alert_trigger_timestamp(cand) - primary_ts)
            if delta <= best_delta:
                best = cand
                best_delta = delta
        if best is not None and best.alert_id not in seen_alert_ids:
            items.append(_storyboard_item(primary_ts, best))
            seen_alert_ids.add(best.alert_id)
            seen_camera_ids.add(best.camera_id)

    # Rule 3: event_group_id — include every alert in the same group from a
    # camera we have not added yet.
    if primary.event_group_id and len(items) < _MAX_STORYBOARD_CAMERAS:
        with db.get_session() as session:
            from sqlalchemy import select
            group_alerts = session.scalars(
                select(AlertRecord)
                .where(AlertRecord.event_group_id == primary.event_group_id)
                .where(AlertRecord.alert_id != primary.alert_id)
                .order_by(AlertRecord.timestamp.asc())
            ).all()
        for cand in group_alerts:
            if len(items) >= _MAX_STORYBOARD_CAMERAS:
                break
            if cand.alert_id in seen_alert_ids:
                continue
            if cand.camera_id in seen_camera_ids:
                # Spec: only include alerts from *different* cameras.
                continue
            items.append(_storyboard_item(primary_ts, cand))
            seen_alert_ids.add(cand.alert_id)
            seen_camera_ids.add(cand.camera_id)

    return api_success({
        "primary_alert_id": primary.alert_id,
        "cameras": items,
        "count": len(items),
    })


def _compute_key_frames(signals: dict) -> list[dict]:
    """Auto-compute key frame bookmarks from signal data.

    Three system key frames:
    1. First appearance: anomaly_score first rises above background noise
    2. Evidence threshold: CUSUM evidence crosses trigger threshold
    3. Trigger: the actual alert trigger frame (highest score)
    """
    key_frames = []
    scores = signals.get("anomaly_scores", [])
    if not scores:
        return key_frames

    # 1. First appearance: first frame where score > 2x median of first quarter
    quarter = max(1, len(scores) // 4)
    baseline_median = sorted(scores[:quarter])[quarter // 2] if quarter > 0 else 0
    noise_threshold = max(baseline_median * 2, 0.1)
    for i, s in enumerate(scores):
        if s > noise_threshold:
            key_frames.append({
                "index": i,
                "type": "first_appearance",
                "label": "首次出现",
            })
            break

    # 2. Evidence threshold: find where CUSUM evidence is highest
    cusum = signals.get("cusum_evidence", {})
    if cusum:
        first_zone = next(iter(cusum.values()), [])
        if first_zone:
            max_evidence_idx = 0
            max_evidence = 0
            for i, e in enumerate(first_zone):
                if e > max_evidence:
                    max_evidence = e
                    max_evidence_idx = i
            if max_evidence > 0:
                key_frames.append({
                    "index": max_evidence_idx,
                    "type": "evidence_threshold",
                    "label": "证据超阈",
                })

    # 3. Trigger: frame with highest anomaly score
    max_score_idx = scores.index(max(scores))
    key_frames.append({
        "index": max_score_idx,
        "type": "trigger",
        "label": "触发",
    })

    return key_frames
