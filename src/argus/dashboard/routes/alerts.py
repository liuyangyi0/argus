"""Alert management API routes (Chinese UI with bulk actions and export)."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from argus.dashboard.components import empty_state, page_header, status_badge

router = APIRouter()


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


# ── Image serving (must be before /{alert_id} catch-all) ──

@router.get("/{alert_id}/image/{image_type}")
async def alert_image(request: Request, alert_id: str, image_type: str):
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


@router.get("/{alert_id}/detail", response_class=HTMLResponse)
async def alert_detail(request: Request, alert_id: str):
    """Alert detail view with large image and metadata."""
    db = request.app.state.db
    if not db:
        return HTMLResponse('<p style="color:#f44336;">数据库不可用</p>')

    alert = db.get_alert(alert_id)
    if alert is None:
        return HTMLResponse('<p style="color:#f44336;">告警不存在</p>')

    ts = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S") if alert.timestamp else "N/A"

    has_snapshot = bool(alert.snapshot_path)
    has_heatmap = bool(alert.heatmap_path)
    has_composite = has_snapshot and has_heatmap
    default_type = "composite" if has_composite else ("snapshot" if has_snapshot else "")
    img_url = f"/api/alerts/{alert_id}/image/{default_type}" if default_type else ""

    toggle_buttons = ""
    if has_snapshot:
        toggles = []
        if has_snapshot:
            toggles.append(
                f'<span class="img-toggle" '
                f'onclick="document.getElementById(\'detail-img\').src='
                f"'/api/alerts/{alert_id}/image/snapshot'\">原图</span>"
            )
        if has_heatmap:
            toggles.append(
                f'<span class="img-toggle" '
                f'onclick="document.getElementById(\'detail-img\').src='
                f"'/api/alerts/{alert_id}/image/heatmap'\">热力图</span>"
            )
        if has_composite:
            toggles.append(
                f'<span class="img-toggle active" '
                f'onclick="document.getElementById(\'detail-img\').src='
                f"'/api/alerts/{alert_id}/image/composite'\">叠加图</span>"
            )
        toggle_buttons = '<div style="margin-bottom:12px;">' + "".join(toggles) + "</div>"

    img_html = (
        f'<img id="detail-img" src="{img_url}" '
        f'style="max-width:100%;border-radius:6px;border:1px solid #2a2d37;" />'
        if img_url
        else '<div class="empty-state"><div class="message">暂无图片</div></div>'
    )

    ack_btn = ""
    if not alert.acknowledged:
        ack_btn = (
            f'<button class="btn btn-primary" '
            f'hx-post="/api/alerts/{alert_id}/acknowledge" '
            f'hx-swap="outerHTML">确认告警</button>'
        )
    else:
        ack_btn = f'<span style="color:#4caf50;">已确认 ({alert.acknowledged_by or "operator"})</span>'

    fp_btn = ""
    if not alert.false_positive:
        fp_btn = (
            f'<button class="btn btn-ghost" style="margin-left:8px;" '
            f'hx-post="/api/alerts/{alert_id}/false-positive" '
            f'hx-swap="outerHTML">标记误报</button>'
        )
    else:
        fp_btn = '<span style="color:#ff9800;margin-left:8px;">已标记误报</span>'

    close_btn = (
        '<button class="modal-close" '
        "onclick=\"document.getElementById('alert-modal').classList.remove('active')\">"
        "&times;</button>"
    )

    return HTMLResponse(f"""
    {close_btn}
    <div style="display:flex;gap:24px;flex-wrap:wrap;">
        <div style="flex:1;min-width:400px;">
            {toggle_buttons}
            {img_html}
        </div>
        <div style="flex:0 0 280px;">
            <h3 style="color:#4fc3f7;margin-bottom:16px;">告警详情</h3>
            <table style="font-size:13px;">
                <tr><td style="color:#8890a0;padding:6px 12px 6px 0;">告警ID</td>
                    <td style="padding:6px 0;font-size:12px;">{alert.alert_id}</td></tr>
                <tr><td style="color:#8890a0;padding:6px 12px 6px 0;">时间</td>
                    <td style="padding:6px 0;">{ts}</td></tr>
                <tr><td style="color:#8890a0;padding:6px 12px 6px 0;">摄像头</td>
                    <td style="padding:6px 0;">{alert.camera_id}</td></tr>
                <tr><td style="color:#8890a0;padding:6px 12px 6px 0;">区域</td>
                    <td style="padding:6px 0;">{alert.zone_id}</td></tr>
                <tr><td style="color:#8890a0;padding:6px 12px 6px 0;">严重度</td>
                    <td style="padding:6px 0;">{status_badge(alert.severity)}</td></tr>
                <tr><td style="color:#8890a0;padding:6px 12px 6px 0;">异常分数</td>
                    <td style="padding:6px 0;">{alert.anomaly_score:.4f}</td></tr>
            </table>
            <div style="margin-top:20px;">{ack_btn}{fp_btn}</div>
        </div>
    </div>""")


# ── Main alerts list ──

@router.get("", response_class=HTMLResponse)
async def alerts_list(
    request: Request,
    camera_id: str | None = Query(None),
    severity: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Alert list page with filtering, pagination, bulk actions, and thumbnails."""
    db = request.app.state.db
    if not db:
        return HTMLResponse(empty_state("数据库不可用"))

    offset = (page - 1) * page_size
    alerts = db.get_alerts(camera_id=camera_id, severity=severity, limit=page_size, offset=offset)
    total = db.get_alert_count(camera_id=camera_id, severity=severity)

    # Statistics
    stats_html = ""
    try:
        total_all = db.get_alert_count()
        cnt_high = db.get_alert_count(severity="high")
        cnt_medium = db.get_alert_count(severity="medium")
        cnt_low = db.get_alert_count(severity="low")
        cnt_info = db.get_alert_count(severity="info")
        stats_html = f"""
        <div class="flex gap-16 mb-16" style="font-size:13px;color:#8890a0;">
            <span>总计: <strong style="color:#e0e0e0;">{total_all}</strong></span>
            <span>{status_badge("high")} {cnt_high}</span>
            <span>{status_badge("medium")} {cnt_medium}</span>
            <span>{status_badge("low")} {cnt_low}</span>
            <span>{status_badge("info")} {cnt_info}</span>
        </div>"""
    except Exception:
        pass

    # Filter controls
    severity_options = ""
    for sev, label in [("", "全部"), ("info", "提示"), ("low", "低"), ("medium", "中"), ("high", "高")]:
        selected = "selected" if sev == (severity or "") else ""
        severity_options += f'<option value="{sev}" {selected}>{label}</option>'

    camera_manager = request.app.state.camera_manager
    cam_options = '<option value="">全部摄像头</option>'
    if camera_manager:
        for s in camera_manager.get_status():
            sel = "selected" if s.camera_id == camera_id else ""
            cam_options += f'<option value="{s.camera_id}" {sel}>{s.camera_id}</option>'

    sev_param = f"&severity={severity}" if severity else ""
    cam_param = f"&camera_id={camera_id}" if camera_id else ""
    base_url = "/alerts?"

    filters_html = f"""
    <div class="flex gap-12 mb-16" style="flex-wrap:wrap;align-items:center;">
        <select class="form-select" style="width:auto;"
                onchange="window.location.href='{base_url}camera_id='+this.value+'{sev_param}'">
            {cam_options}
        </select>
        <select class="form-select" style="width:auto;"
                onchange="window.location.href='{base_url}severity='+this.value+'{cam_param}'">
            {severity_options}
        </select>
        <span style="color:#8890a0;font-size:13px;">共 {total} 条告警</span>
        <div style="margin-left:auto;">
            <a href="/api/alerts/export-csv?{cam_param.lstrip('&')}{sev_param}" class="btn btn-ghost btn-sm" download>导出CSV</a>
        </div>
    </div>"""

    # Alert table with checkboxes
    rows = ""
    for a in alerts:
        ts = a.timestamp.strftime("%Y-%m-%d %H:%M:%S") if a.timestamp else ""

        thumb = '<span style="color:#616161;font-size:11px;">—</span>'
        if a.snapshot_path:
            img_type = "composite" if a.heatmap_path else "snapshot"
            thumb = (
                f'<img src="/api/alerts/{a.alert_id}/image/{img_type}" '
                f'class="alert-thumb" alt="" loading="lazy" '
                f'hx-get="/api/alerts/{a.alert_id}/detail" '
                f'hx-target="#alert-modal-content" hx-swap="innerHTML" '
                f"onclick=\"document.getElementById('alert-modal').classList.add('active')\" />"
            )

        ack_html = (
            '<span style="color:#4caf50;font-size:12px;">已确认</span>'
            if a.acknowledged else
            f'<button class="btn btn-sm btn-primary" '
            f'hx-post="/api/alerts/{a.alert_id}/acknowledge" hx-swap="outerHTML">确认</button>'
        )
        fp_html = (
            '<span style="color:#ff9800;font-size:12px;">误报</span>'
            if a.false_positive else
            f'<button class="btn btn-sm btn-ghost" '
            f'hx-post="/api/alerts/{a.alert_id}/false-positive" hx-swap="outerHTML">标记</button>'
        )

        rows += f"""
        <tr>
            <td class="checkbox-cell"><input type="checkbox" name="alert_ids" value="{a.alert_id}" onchange="updateBulkBar()"></td>
            <td>{thumb}</td>
            <td style="font-size:12px;">{a.alert_id[:24]}</td>
            <td>{ts}</td>
            <td>{a.camera_id}</td>
            <td>{a.zone_id}</td>
            <td>{status_badge(a.severity)}</td>
            <td>{a.anomaly_score:.3f}</td>
            <td>{ack_html}</td>
            <td>{fp_html}</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="10" style="color:#616161;text-align:center;padding:24px;">暂无告警记录</td></tr>'

    # Pagination
    total_pages = max(1, (total + page_size - 1) // page_size)
    pagination = ""
    if total_pages > 1:
        prev_cls = ' style="opacity:0.3;pointer-events:none;"' if page <= 1 else ""
        next_cls = ' style="opacity:0.3;pointer-events:none;"' if page >= total_pages else ""
        params = sev_param + cam_param
        pagination = f"""
        <div style="display:flex;justify-content:center;gap:12px;margin-top:16px;align-items:center;">
            <a href="/alerts?page={page-1}{params}" class="btn btn-sm btn-ghost"{prev_cls}>上一页</a>
            <span style="color:#8890a0;font-size:13px;">第 {page}/{total_pages} 页</span>
            <a href="/alerts?page={page+1}{params}" class="btn btn-sm btn-ghost"{next_cls}>下一页</a>
        </div>"""

    header = page_header("告警中心", f"监控与管理异常检测告警")

    # Bulk action bar JS
    bulk_js = """
    <script>
    function updateBulkBar() {
        var checked = document.querySelectorAll('input[name="alert_ids"]:checked');
        var bar = document.getElementById('bulk-bar');
        var count = document.getElementById('bulk-count');
        if (checked.length > 0) {
            bar.classList.add('active');
            count.textContent = checked.length + ' 条已选';
        } else {
            bar.classList.remove('active');
        }
    }
    function selectAll(cb) {
        document.querySelectorAll('input[name="alert_ids"]').forEach(function(c) { c.checked = cb.checked; });
        updateBulkBar();
    }
    function bulkAction(action) {
        var ids = Array.from(document.querySelectorAll('input[name="alert_ids"]:checked')).map(function(c) { return c.value; });
        if (ids.length === 0) return;
        fetch('/api/alerts/bulk-' + action, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({alert_ids: ids})
        }).then(function(r) { return r.json(); }).then(function(d) {
            window.showToast(d.message || '操作完成', 'success');
            htmx.ajax('GET', '/api/alerts', {target: '#content', swap: 'innerHTML'});
        });
    }
    </script>"""

    return HTMLResponse(f"""
    {header}
    {stats_html}
    {filters_html}
    <div class="card">
        <form id="alerts-form">
            <table>
                <thead>
                    <tr>
                        <th class="checkbox-cell"><input type="checkbox" onchange="selectAll(this)"></th>
                        <th>快照</th><th>告警ID</th><th>时间</th><th>摄像头</th>
                        <th>区域</th><th>严重度</th><th>分数</th><th>确认</th><th>误报</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </form>
        {pagination}
    </div>
    <div id="bulk-bar" class="bulk-bar">
        <span id="bulk-count" class="count">0 条已选</span>
        <div class="flex gap-8">
            <button class="btn btn-sm btn-primary" onclick="bulkAction('acknowledge')">批量确认</button>
            <button class="btn btn-sm btn-warning" onclick="bulkAction('false-positive')">批量标记误报</button>
        </div>
    </div>
    {bulk_js}""")


# ── Bulk operations ──

@router.post("/bulk-acknowledge")
async def bulk_acknowledge(request: Request):
    """Bulk acknowledge alerts."""
    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

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

    return JSONResponse({"status": "ok", "count": count, "message": f"已确认 {count} 条告警"})


@router.post("/bulk-false-positive")
async def bulk_false_positive(request: Request):
    """Bulk mark alerts as false positive."""
    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

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

    return JSONResponse({"status": "ok", "count": count, "message": f"已标记 {count} 条为误报"})


# ── CSV Export ──

@router.get("/export-csv")
async def export_csv(
    request: Request,
    camera_id: str | None = Query(None),
    severity: str | None = Query(None),
):
    """Export alerts as CSV file."""
    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    alerts = db.get_alerts(camera_id=camera_id, severity=severity, limit=10000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["告警ID", "时间", "摄像头", "区域", "严重度", "异常分数", "已确认", "误报", "备注"])

    for a in alerts:
        ts = a.timestamp.strftime("%Y-%m-%d %H:%M:%S") if a.timestamp else ""
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


# ── Single alert operations ──

@router.post("/{alert_id}/acknowledge", response_class=HTMLResponse)
async def acknowledge_alert(request: Request, alert_id: str):
    """Acknowledge an alert."""
    db = request.app.state.db
    if db and db.acknowledge_alert(alert_id, "operator"):
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
        return HTMLResponse('<span style="color:#4caf50;font-size:12px;">已确认</span>')
    return HTMLResponse('<span style="color:#f44336;font-size:12px;">操作失败</span>')


@router.post("/{alert_id}/false-positive", response_class=HTMLResponse)
async def mark_false_positive(request: Request, alert_id: str):
    """Mark an alert as a false positive."""
    db = request.app.state.db
    if db and db.mark_false_positive(alert_id):
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
        return HTMLResponse('<span style="color:#ff9800;font-size:12px;">已标记误报</span>')
    return HTMLResponse('<span style="color:#f44336;font-size:12px;">操作失败</span>')


@router.get("/json")
async def alerts_json(
    request: Request,
    camera_id: str | None = Query(None),
    severity: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """JSON API for alerts (for external integrations)."""
    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    alerts = db.get_alerts(camera_id=camera_id, severity=severity, limit=limit)
    return [a.to_dict() for a in alerts]
