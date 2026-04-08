"""Alert replay API for multi-track timeline playback (UX v2 §1.3).

Provides endpoints to retrieve recording metadata, signal timeseries,
individual frames, and historical reference frames for alert replay.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response, StreamingResponse

from argus.dashboard.api_response import (
    api_success,
    api_not_found,
    api_unavailable,
    api_validation_error,
)

router = APIRouter()


def _get_recording_store(request: Request):
    """Get the AlertRecordingStore from app state."""
    return getattr(request.app.state, "recording_store", None)


def _get_db(request: Request):
    """Get the Database from app state."""
    return getattr(request.app.state, "db", None)


@router.get("/{alert_id}/metadata")
def replay_metadata(request: Request, alert_id: str):
    """Return recording metadata for an alert.

    Response: {alert_id, camera_id, severity, start_ts, end_ts, trigger_ts,
               fps, frame_count, status, linked_alert_id}
    """
    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    metadata = store.load_metadata(alert_id)
    if metadata is None:
        return api_not_found("录像不存在")

    return api_success(metadata)


@router.get("/{alert_id}/signals")
def replay_signals(request: Request, alert_id: str):
    """Return signal timeseries for the 5 replay tracks.

    Response: {timestamps[], anomaly_scores[], simplex_scores[],
               cusum_evidence: {zone: float[]}, yolo_persons[],
               operator_actions[], key_frames[]}
    """
    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    signals = store.load_signals(alert_id)
    if signals is None:
        return api_not_found("录像不存在")

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

    return api_success(signals)


@router.get("/{alert_id}/frame/{index}")
def replay_frame(request: Request, alert_id: str, index: int):
    """Return a single JPEG frame by index."""
    store = _get_recording_store(request)
    if store is None:
        return Response(status_code=503)

    if index < 0:
        return Response(status_code=400)

    frame_bytes = store.load_frame(alert_id, index)
    if frame_bytes is None:
        return Response(status_code=404)

    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{alert_id}/heatmap/{index}")
def replay_heatmap_frame(request: Request, alert_id: str, index: int):
    """Return a single heatmap overlay JPEG by index (§1.3 overlay toggle)."""
    store = _get_recording_store(request)
    if store is None:
        return Response(status_code=503)

    if index < 0:
        return Response(status_code=400)

    heatmap_bytes = store.load_heatmap_frame(alert_id, index)
    if heatmap_bytes is None:
        return Response(status_code=404)

    return Response(
        content=heatmap_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{alert_id}/frames")
async def replay_frames_stream(
    request: Request,
    alert_id: str,
    start: int = Query(default=0, ge=0),
    end: int = Query(default=50, ge=0),
):
    """Stream a range of frames as multipart MJPEG.

    Used for batch playback. Max 200 frames per request.
    """
    store = _get_recording_store(request)
    if store is None:
        return Response(status_code=503)

    # Clamp range
    end = min(end, start + 200)
    frame_list = store.list_frames(alert_id)
    if not frame_list:
        return Response(status_code=404)

    end = min(end, len(frame_list))

    def generate():
        for i in range(start, end):
            frame_bytes = store.load_frame(alert_id, i)
            if frame_bytes is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
                b"\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{alert_id}/reference")
def replay_reference(
    request: Request,
    alert_id: str,
    date: str = Query(default="", description="Reference date YYYY-MM-DD"),
    time_str: str = Query(default="", alias="time", description="Reference time HH:MM:SS"),
):
    """Return a historical reference frame from the same camera at the same time of day.

    If no specific date is given, defaults to yesterday.
    Returns {available: bool, frame_base64: str|null, source_date: str}.
    """
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

    # Build the reference timestamp (same time of day on reference date)
    ref_dt = ref_date.replace(
        hour=trigger_dt.hour,
        minute=trigger_dt.minute,
        second=trigger_dt.second,
    )
    ref_date_str = ref_dt.strftime("%Y-%m-%d")

    # Search for recordings from the same camera on the reference date
    archive_dir = store.archive_dir / ref_date_str / camera_id
    if not archive_dir.exists():
        return api_success({
            "available": False,
            "frame_base64": None,
            "source_date": ref_date_str,
        })

    # Find the recording closest to the reference time
    best_frame: bytes | None = None
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
            # Find the trigger frame from this recording
            dist = abs(meta["trigger_timestamp"] - ref_ts)
            if dist < best_distance:
                trigger_idx = meta.get("trigger_frame_index", 0)
                frame_path = rec_dir / "frames" / f"{trigger_idx:06d}.jpg"
                if frame_path.exists():
                    best_frame = frame_path.read_bytes()
                    best_distance = dist
        except (json.JSONDecodeError, KeyError):
            continue

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


@router.post("/{alert_id}/pin-frame")
async def pin_frame(request: Request, alert_id: str):
    """Pin a frame as a key frame bookmark (operator action).

    Request body: {"index": int, "label": str}
    """
    store = _get_recording_store(request)
    if store is None:
        return api_unavailable("录像存储未配置")

    body = await request.json()
    index = body.get("index", 0)
    label = body.get("label", "")

    if not store.pin_frame(alert_id, index, label):
        return api_not_found("录像不存在")

    return api_success({"success": True})


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
        # Use the first zone's evidence
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
