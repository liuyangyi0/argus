"""Alert management API routes (Chinese UI with bulk actions and export)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from argus.dashboard.api_response import (
    ErrorCode,
    api_error,
    api_forbidden,
    api_internal_error,
    api_not_found,
    api_success,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.forms import parse_request_form

if TYPE_CHECKING:
    from argus.storage.alert_recording import AlertRecordingStore

logger = structlog.get_logger()

router = APIRouter()

# Database/CSV-friendly timestamp format used in exports and audit fields.
_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def _cleanup_alert_images(
    alert_id: str,
    image_paths: list[str],
    recording_store: AlertRecordingStore | None,
) -> None:
    """Remove on-disk image files and recording directory tied to a deleted alert."""
    for path_str in image_paths:
        try:
            Path(path_str).unlink(missing_ok=True)
        except OSError:
            logger.debug("alerts.image_delete_failed", path=path_str, exc_info=True)

    if recording_store is not None:
        recording_store.delete_recording(alert_id)


def _generate_composite(
    snapshot_path: str, heatmap_path: str, alpha: float = 0.4
) -> bytes | None:
    """Blend heatmap onto snapshot and return JPEG bytes."""
    snapshot = cv2.imread(snapshot_path)
    heatmap = cv2.imread(heatmap_path)
    if snapshot is None or heatmap is None:
        return None
    if heatmap.shape[:2] != snapshot.shape[:2]:
        heatmap = cv2.resize(heatmap, (snapshot.shape[1], snapshot.shape[0]))
    composite = cv2.addWeighted(snapshot, 1.0, heatmap, alpha, 0)
    _, buffer = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def _is_safe_path(file_path: str, alerts_dir: Path) -> bool:
    """Verify that a file path is under the alerts directory."""
    try:
        resolved = Path(file_path).resolve()
        safe_root = alerts_dir.resolve()
        return str(resolved).startswith(str(safe_root))
    except (ValueError, OSError):
        return False


@router.get("/{alert_id}/detail")
async def alert_detail(request: Request, alert_id: str):
    """Get single alert detail as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    alert = db.get_alert(alert_id)
    if not alert:
        return api_not_found("告警不存在")

    recording_store: AlertRecordingStore | None = getattr(
        request.app.state, "recording_store", None,
    )
    has_recording = False
    if recording_store:
        has_recording = recording_store.has_recording(alert_id)
    rec_status = "complete" if has_recording else None

    data = {
        "alert_id": alert.alert_id,
        "timestamp": alert.timestamp.strftime(_TIMESTAMP_FMT) if alert.timestamp else None,
        "camera_id": alert.camera_id,
        "zone_id": getattr(alert, "zone_id", ""),
        "severity": alert.severity,
        "anomaly_score": alert.anomaly_score,
        "acknowledged": alert.acknowledged,
        "false_positive": alert.false_positive,
        "has_recording": has_recording,
        "recording_status": rec_status,
        "workflow_status": getattr(alert, "workflow_status", "new"),
        "notes": getattr(alert, "notes", ""),
        "classification_label": getattr(alert, "classification_label", None),
        "classification_confidence": getattr(alert, "classification_confidence", None),
        "corroborated": getattr(alert, "corroborated", None),
        "correlation_partner": getattr(alert, "correlation_partner", None),
        "segmentation_count": getattr(alert, "segmentation_count", None),
        "segmentation_total_area_px": getattr(alert, "segmentation_total_area_px", None),
        "event_group_id": getattr(alert, "event_group_id", None),
        "event_group_count": getattr(alert, "event_group_count", None),
        "snapshot_path": getattr(alert, "snapshot_path", None),
        "heatmap_path": getattr(alert, "heatmap_path", None),
        "speed_ms": getattr(alert, "speed_ms", None),
        "assigned_to": getattr(alert, "assigned_to", None),
        "resolved_at": getattr(alert, "resolved_at", None),
    }

    return api_success(data)


# ── Image serving (must be before /{alert_id} catch-all) ──

@router.get("/{alert_id}/image/{image_type}")
def alert_image(request: Request, alert_id: str, image_type: str):
    """Serve alert snapshot, heatmap, or composite overlay image."""
    if image_type not in ("snapshot", "heatmap", "composite"):
        return Response(status_code=400)

    db = request.app.state.db
    if not db:
        return Response(status_code=503)

    alert = db.get_alert(alert_id)
    if alert is None:
        return Response(status_code=404)

    alerts_dir = getattr(request.app.state, "alerts_dir", Path("data/alerts"))

    if image_type == "composite":
        if not alert.snapshot_path or not alert.heatmap_path:
            if alert.snapshot_path and _is_safe_path(alert.snapshot_path, alerts_dir):
                path = Path(alert.snapshot_path)
                if path.exists():
                    return Response(content=path.read_bytes(), media_type="image/jpeg")
            return Response(status_code=404)
        if not _is_safe_path(alert.snapshot_path, alerts_dir) or not _is_safe_path(alert.heatmap_path, alerts_dir):
            return Response(status_code=403)
        data = _generate_composite(alert.snapshot_path, alert.heatmap_path)
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/jpeg")

    path_str = alert.snapshot_path if image_type == "snapshot" else alert.heatmap_path
    if not path_str:
        return Response(status_code=404)
    if not _is_safe_path(path_str, alerts_dir):
        return Response(status_code=403)
    path = Path(path_str)
    if not path.exists():
        return Response(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/jpeg")


_WF_LABELS = {
    "new": ("待处理", "var(--status-warn)"),
    "acknowledged": ("已确认", "var(--status-info)"),
    "investigating": ("调查中", "var(--status-info)"),
    "resolved": ("已解决", "var(--status-ok)"),
    "closed": ("已关闭", "var(--text-tertiary)"),
    "false_positive": ("误报", "var(--status-alert)"),
    "uncertain": ("不确定", "var(--status-warn)"),
}

_FEEDBACK_CATEGORIES = [
    ("lens_glare", "镜头反光"),
    ("insect", "昆虫"),
    ("shadow", "光影变化"),
    ("vibration", "相机振动"),
    ("insulation", "保温棉脱落"),
    ("condensation", "冷凝水/雾气"),
    ("other", "其他"),
]


@router.post("/{alert_id}/workflow")
async def alert_workflow_transition(request: Request, alert_id: str):
    """Transition alert workflow status with severity-based handling (UX v2 §5.1).

    Handling policy by severity:
    - LOW/INFO: Quick actions allowed (acknowledge, false_positive) without confirmation
    - MEDIUM: Quick actions allowed but require `confirmed: true` in request body
    - HIGH: Only "investigate" action accepted from quick panel; must enter detail page
            for full workflow transitions

    JSON request body: {status, notes?, category?, assigned_to?, confirmed?}
    JSON response: {success, severity, handling_policy, require_confirmation,
                    require_detail_view, next_actions}
    """
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    # Accept both JSON and form data
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        data = await request.json()
    else:
        form = await parse_request_form(request)
        data = dict(form)

    new_status = data.get("status", "")
    notes = data.get("notes", "")
    category = data.get("category", "")
    assigned_to = data.get("assigned_to", "")
    confirmed = data.get("confirmed", False)
    from_detail_view = data.get("from_detail_view", False)

    if category and notes:
        notes = f"[{category}] {notes}"
    elif category:
        notes = f"[{category}]"

    valid_statuses = {
        "acknowledged", "investigating", "resolved", "closed",
        "false_positive", "uncertain",
    }
    if new_status not in valid_statuses:
        return api_validation_error(f"无效状态: {new_status}")

    # Fetch alert to check severity
    alert = db.get_alert(alert_id)
    if alert is None:
        return api_not_found("告警不存在")

    severity = alert.severity
    quick_actions = {"acknowledged", "false_positive"}

    # UX v2 §5.1: Severity-based handling enforcement
    if severity == "high" and new_status in quick_actions and not from_detail_view:
        return api_forbidden("HIGH 级告警必须进入详情页处理")

    if severity == "medium" and new_status in quick_actions and not confirmed:
        return api_error(
            ErrorCode.VALIDATION_ERROR,
            "MEDIUM 级告警需要确认后处理",
            status_code=400,
            data={
                "severity": severity,
                "handling_policy": "confirm",
                "require_confirmation": True,
                "next_actions": ["acknowledged", "false_positive", "investigating"],
            },
        )

    # Execute the transition
    success = db.update_alert_workflow(
        alert_id, new_status, notes=notes or None, assigned_to=assigned_to or None,
    )
    if not success:
        return api_internal_error("状态转换失败")

    # Submit to feedback queue if applicable
    _submit_workflow_feedback(request, alert_id, new_status, category, notes)

    # Determine handling metadata from canonical mapping
    from argus.alerts.grader import HANDLING_POLICIES
    from argus.config.schema import AlertSeverity as _Sev
    handling_policy = HANDLING_POLICIES.get(_Sev(severity), "quick")

    _next_actions_map = {
        "new": ["acknowledged", "investigating", "false_positive"],
        "acknowledged": ["investigating", "resolved", "false_positive"],
        "investigating": ["resolved", "false_positive", "uncertain"],
        "resolved": ["closed"],
    }
    next_actions = _next_actions_map.get(new_status, [])

    # Return JSON response for API consumers
    if "json" in content_type:
        return api_success({
            "severity": severity,
            "handling_policy": handling_policy,
            "require_confirmation": severity == "medium",
            "require_detail_view": severity == "high",
            "next_actions": next_actions,
        })

    # Re-render the detail view for HTMX consumers
    return await alert_detail(request, alert_id)


# ── Bulk operations ──

@router.post("/bulk-acknowledge")
async def bulk_acknowledge(request: Request):
    """Bulk acknowledge alerts."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    data = await request.json()
    alert_ids = data.get("alert_ids", [])

    count = 0
    for aid in alert_ids:
        if db.acknowledge_alert(aid, "operator"):
            count += 1

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="bulk_acknowledge",
            target_type="alert",
            target_id=",".join(alert_ids[:10]),
            detail=f"批量确认 {count} 条告警",
            ip_address=client_ip,
        )

    return api_success({"count": count, "message": f"已确认 {count} 条告警"})


@router.post("/bulk-false-positive")
async def bulk_false_positive(request: Request):
    """Bulk mark alerts as false positive."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    data = await request.json()
    alert_ids = data.get("alert_ids", [])

    count = 0
    for aid in alert_ids:
        if db.mark_false_positive(aid):
            count += 1

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="bulk_false_positive",
            target_type="alert",
            target_id=",".join(alert_ids[:10]),
            detail=f"批量标记 {count} 条为误报",
            ip_address=client_ip,
        )

    return api_success({"count": count, "message": f"已标记 {count} 条为误报"})


@router.post("/bulk-delete")
async def bulk_delete(request: Request):
    """Bulk delete alerts."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    data = await request.json()
    alert_ids = data.get("alert_ids", [])

    recording_store = getattr(request.app.state, "recording_store", None)
    count = 0
    for aid in alert_ids:
        success, image_paths = db.delete_alert(aid)
        if success:
            count += 1
            _cleanup_alert_images(aid, image_paths, recording_store)

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="bulk_delete",
            target_type="alert",
            target_id=",".join(alert_ids[:10]),
            detail=f"批量删除 {count} 条告警",
            ip_address=client_ip,
        )

    return api_success({"count": count, "message": f"已删除 {count} 条告警"})


# ── Event Group ──

@router.get("/group/{event_group_id}")
def get_alert_group(
    request: Request,
    event_group_id: str,
    limit: int = Query(50, ge=1, le=200),
):
    """Get all alerts belonging to an event group."""
    from sqlalchemy import select
    from argus.storage.models import AlertRecord

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    with db.get_session() as session:
        stmt = (
            select(AlertRecord)
            .where(AlertRecord.event_group_id == event_group_id)
            .order_by(AlertRecord.timestamp.desc())
            .limit(limit)
        )
        records = list(session.scalars(stmt).all())

    return api_success(data={
        "event_group_id": event_group_id,
        "count": len(records),
        "alerts": [r.to_dict() for r in records],
    })


# ── CSV Export ──

@router.get("/export-csv")
def export_csv(
    request: Request,
    camera_id: str | None = Query(None),
    severity: str | None = Query(None),
):
    """Export alerts as CSV file."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    alerts = db.get_alerts(camera_id=camera_id, severity=severity, limit=10000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["告警ID", "时间", "摄像头", "区域", "严重度", "异常分数", "已确认", "误报", "备注"])

    for a in alerts:
        ts = a.timestamp.strftime(_TIMESTAMP_FMT) if a.timestamp else ""
        writer.writerow([
            a.alert_id, ts, a.camera_id, a.zone_id,
            a.severity, f"{a.anomaly_score:.4f}",
            "是" if a.acknowledged else "否",
            "是" if a.false_positive else "否",
            a.notes or "",
        ])

    content = output.getvalue()
    return Response(
        content=content.encode("utf-8-sig"),  # BOM for Excel compatibility
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=argus_alerts.csv"},
    )


@router.get("/export-pdf", response_model=None)
def export_pdf_report(
    request: Request,
    camera_id: str | None = Query(None),
    severity: str | None = Query(None),
):
    """Export alerts as a printable HTML report (use browser Ctrl+P to save as PDF)."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    alerts = db.get_alerts(camera_id=camera_id, severity=severity, limit=10000)

    severity_cn = {"high": "高", "medium": "中", "low": "低", "info": "提示"}
    from datetime import datetime as dt

    rows_html = ""
    for i, a in enumerate(alerts, 1):
        ts = a.timestamp.strftime(_TIMESTAMP_FMT) if a.timestamp else ""
        sev = severity_cn.get(a.severity, a.severity)
        status = "已确认" if a.acknowledged else ("误报" if a.false_positive else "待处理")
        rows_html += f"""<tr>
            <td>{i}</td><td>{ts}</td><td>{a.camera_id}</td>
            <td>{a.zone_id}</td><td>{sev}</td>
            <td>{a.anomaly_score:.4f}</td><td>{status}</td>
            <td>{a.notes or ''}</td>
        </tr>"""

    now = dt.now().strftime("%Y-%m-%d %H:%M")
    filter_desc = ""
    if camera_id:
        filter_desc += f" | 摄像头: {camera_id}"
    if severity:
        filter_desc += f" | 严重度: {severity_cn.get(severity, severity)}"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Argus 告警报告</title>
<style>
  body {{ font-family: 'Microsoft YaHei', sans-serif; padding: 20px; color: #333; }}
  h1 {{ font-size: 18px; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  .meta {{ font-size: 12px; color: #666; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: bold; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  @media print {{ body {{ padding: 0; }} }}
</style></head><body>
<h1>Argus 核电站异物检测 — 告警报告</h1>
<div class="meta">生成时间: {now} | 共 {len(alerts)} 条{filter_desc}</div>
<table>
<tr><th>#</th><th>时间</th><th>摄像头</th><th>区域</th><th>严重度</th><th>异常分数</th><th>状态</th><th>备注</th></tr>
{rows_html}
</table>
<script>window.print()</script>
</body></html>"""

    return HTMLResponse(content=html)


# ── Single alert operations ──

@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(request: Request, alert_id: str):
    """Acknowledge an alert."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    if db.acknowledge_alert(alert_id, "operator"):
        audit = getattr(request.app.state, "audit_logger", None)
        client_ip = request.client.host if request.client else ""
        if audit:
            audit.log(
                user="operator",
                action="acknowledge_alert",
                target_type="alert",
                target_id=alert_id,
                ip_address=client_ip,
            )
        return api_success({"alert_id": alert_id, "message": "已确认"})
    return api_not_found("告警不存在或操作失败")


@router.post("/{alert_id}/false-positive")
def mark_false_positive(request: Request, alert_id: str):
    """Mark an alert as a false positive and submit to feedback queue.

    False positive feedback loop (A4-3 / Section 6): when a normal scene
    triggers a false alarm, the feedback is queued for retraining and the
    snapshot is copied to the baseline directory via FeedbackManager.
    """
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    if db.mark_false_positive(alert_id):
        audit = getattr(request.app.state, "audit_logger", None)
        client_ip = request.client.host if request.client else ""
        if audit:
            audit.log(
                user="operator",
                action="mark_false_positive",
                target_type="alert",
                target_id=alert_id,
                ip_address=client_ip,
            )

        # Submit to feedback queue (handles baseline copy + queue entry)
        _submit_workflow_feedback(request, alert_id, "false_positive")

        return api_success({"alert_id": alert_id, "message": "已标记误报"})
    return api_not_found("告警不存在或操作失败")


@router.delete("/{alert_id}")
def delete_alert(request: Request, alert_id: str):
    """Delete a single alert by ID."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    success, image_paths = db.delete_alert(alert_id)
    if not success:
        return api_not_found("告警不存在")

    recording_store = getattr(request.app.state, "recording_store", None)
    _cleanup_alert_images(alert_id, image_paths, recording_store)

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="delete_alert",
            target_type="alert",
            target_id=alert_id,
            ip_address=client_ip,
        )

    return api_success({"alert_id": alert_id, "message": "已删除"})


def _submit_workflow_feedback(
    request: Request,
    alert_id: str,
    workflow_status: str,
    category: str | None = None,
    notes: str | None = None,
) -> None:
    """Submit feedback to the queue when an alert workflow transition happens.

    Maps workflow statuses to feedback types:
    - acknowledged → confirmed
    - false_positive → false_positive
    - uncertain → uncertain
    Other statuses (investigating, resolved, closed) don't generate feedback.
    """
    _STATUS_TO_FEEDBACK = {
        "acknowledged": "confirmed",
        "false_positive": "false_positive",
        "uncertain": "uncertain",
    }
    feedback_type = _STATUS_TO_FEEDBACK.get(workflow_status)
    if feedback_type is None:
        return

    feedback_mgr = getattr(request.app.state, "feedback_manager", None)
    if feedback_mgr is None:
        # Fallback to legacy _add_fp_snapshot_to_baseline for FP
        if workflow_status == "false_positive":
            _add_fp_snapshot_to_baseline(request, alert_id)
        return

    db = request.app.state.db
    if not db:
        return
    alert = db.get_alert(alert_id)
    if alert is None:
        return

    try:
        feedback_mgr.submit_feedback(
            alert_id=alert_id,
            feedback_type=feedback_type,
            camera_id=alert.camera_id,
            zone_id=getattr(alert, "zone_id", "default") or "default",
            category=category,
            notes=notes,
            submitted_by="operator",
            anomaly_score=alert.anomaly_score,
            snapshot_path=alert.snapshot_path,
        )
    except Exception:
        import structlog
        structlog.get_logger().warning(
            "feedback.submit_failed",
            alert_id=alert_id,
            feedback_type=feedback_type,
            exc_info=True,
        )


def _add_fp_snapshot_to_baseline(request: Request, alert_id: str) -> None:
    """Copy the false-positive alert's snapshot into the current baseline version.

    This creates a feedback loop: FP -> add to baseline -> retrain -> fewer FPs.
    """
    import shutil
    import structlog

    log = structlog.get_logger()

    db = request.app.state.db
    if not db:
        return

    alert = db.get_alert(alert_id)
    if alert is None or not alert.snapshot_path:
        return

    snapshot = Path(alert.snapshot_path)
    if not snapshot.exists():
        log.warning("fp_feedback.snapshot_missing", alert_id=alert_id, path=str(snapshot))
        return

    # Get baseline manager
    baseline_mgr = getattr(request.app.state, "baseline_manager", None)
    if baseline_mgr is None:
        config = getattr(request.app.state, "config", None)
        if config is None:
            return
        from argus.anomaly.baseline import BaselineManager
        baseline_mgr = BaselineManager(baselines_dir=str(config.storage.baselines_dir))

    camera_id = alert.camera_id
    zone_id = getattr(alert, "zone_id", "default") or "default"

    baseline_dir = baseline_mgr.get_baseline_dir(camera_id, zone_id)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename based on alert id to avoid collisions
    ext = snapshot.suffix or ".png"
    dest = baseline_dir / f"fp_{alert_id[:16]}{ext}"
    try:
        shutil.copy2(str(snapshot), str(dest))
    except OSError as exc:
        log.error("fp_feedback.copy_failed", alert_id=alert_id, error=str(exc))
        return

    # Audit trail
    audit = getattr(request.app.state, "audit_logger", None)
    if audit:
        client_ip = request.client.host if request.client else ""
        audit.log(
            user="operator",
            action="fp_add_to_baseline",
            target_type="baseline",
            target_id=f"{camera_id}/{zone_id}",
            detail=f"Added FP snapshot {alert_id} as {dest.name}",
            ip_address=client_ip,
        )

    log.info(
        "fp_feedback.added_to_baseline",
        alert_id=alert_id,
        camera_id=camera_id,
        zone_id=zone_id,
        dest=str(dest),
    )


@router.post("/{alert_id}/annotations")
async def save_annotations(request: Request, alert_id: str):
    """Save operator-drawn annotations (bounding boxes) for an alert frame.

    Used by the Canvas annotation overlay for marking anomalies/false positives.
    Body: {"annotations": [{"x":int,"y":int,"width":int,"height":int,"label":str}]}
    """
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    alert = db.get_alert(alert_id)
    if not alert:
        return api_not_found("告警不存在")

    body = await request.json()
    annotations = body.get("annotations", [])

    # Store annotations as JSON in alert metadata directory
    alerts_dir = Path(getattr(request.app.state, "alerts_dir", "data/alerts"))
    anno_path = alerts_dir / alert_id / "annotations.json"
    anno_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(anno_path, "w") as f:
        json.dump({"alert_id": alert_id, "annotations": annotations}, f)

    return api_success({"alert_id": alert_id, "count": len(annotations)})


@router.get("/json")
def alerts_json(
    request: Request,
    camera_id: str | None = Query(None),
    severity: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """JSON API for alerts (for external integrations)."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    alerts = db.get_alerts(camera_id=camera_id, severity=severity, limit=limit)

    # Batch-fetch recording status to avoid N+1 queries
    recording_map: dict = {}
    if hasattr(db, "get_alert_recordings_batch"):
        alert_ids = [a.alert_id for a in alerts]
        recording_map = db.get_alert_recordings_batch(alert_ids)

    result = []
    for a in alerts:
        d = a.to_dict()
        rec = recording_map.get(a.alert_id)
        d["has_recording"] = rec is not None
        d["recording_status"] = rec.status if rec else None
        result.append(d)

    total = db.get_alert_count(camera_id=camera_id, severity=severity)
    return api_success({"alerts": result, "total": total})
