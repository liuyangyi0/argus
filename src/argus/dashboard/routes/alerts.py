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


@router.get("/{alert_id}/detail", response_class=HTMLResponse)
async def alert_detail(request: Request, alert_id: str):
    """Alert detail view with evidence, baseline comparison, and feedback workflow."""
    db = request.app.state.db
    if not db:
        return HTMLResponse('<p style="color:var(--status-critical);">数据库不可用</p>')

    alert = db.get_alert(alert_id)
    if alert is None:
        return HTMLResponse('<p style="color:var(--status-critical);">告警不存在</p>')

    ts = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S") if alert.timestamp else "N/A"

    # Image toggle buttons
    has_snapshot = bool(alert.snapshot_path)
    has_heatmap = bool(alert.heatmap_path)
    has_composite = has_snapshot and has_heatmap
    default_type = "composite" if has_composite else ("snapshot" if has_snapshot else "")
    img_url = f"/api/alerts/{alert_id}/image/{default_type}" if default_type else ""

    toggles = []
    if has_snapshot:
        toggles.append(
            f'<span class="img-toggle" '
            f'onclick="document.getElementById(\'detail-img\').src='
            f"'/api/alerts/{alert_id}/image/snapshot';"
            f'document.querySelectorAll(\'.img-toggle\').forEach(t=>t.classList.remove(\'active\'));this.classList.add(\'active\')">原图</span>'
        )
    if has_heatmap:
        toggles.append(
            f'<span class="img-toggle" '
            f'onclick="document.getElementById(\'detail-img\').src='
            f"'/api/alerts/{alert_id}/image/heatmap';"
            f'document.querySelectorAll(\'.img-toggle\').forEach(t=>t.classList.remove(\'active\'));this.classList.add(\'active\')">热力图</span>'
        )
    if has_composite:
        toggles.append(
            f'<span class="img-toggle active" '
            f'onclick="document.getElementById(\'detail-img\').src='
            f"'/api/alerts/{alert_id}/image/composite';"
            f'document.querySelectorAll(\'.img-toggle\').forEach(t=>t.classList.remove(\'active\'));this.classList.add(\'active\')">叠加图</span>'
        )
    toggle_html = f'<div style="margin-bottom:var(--space-3);">{"".join(toggles)}</div>' if toggles else ""

    img_html = (
        f'<img id="detail-img" src="{img_url}" '
        f'style="max-width:100%;border-radius:var(--radius-md);border:1px solid var(--border-subtle);" />'
        if img_url
        else '<div class="empty-state"><div class="message">暂无图片</div></div>'
    )

    # Workflow status
    wf_status = getattr(alert, "workflow_status", "new") or "new"
    wf_label, wf_color = _WF_LABELS.get(wf_status, ("未知", "var(--text-tertiary)"))

    # Evidence section
    score_pct = f"{alert.anomaly_score * 100:.1f}%" if alert.anomaly_score else "—"

    # Action buttons based on current workflow state
    actions_html = ""
    if wf_status == "new":
        actions_html = f"""
        <div class="flex gap-8" style="flex-wrap:wrap;">
            <button class="btn btn-primary"
                    hx-post="/api/alerts/{alert_id}/workflow" hx-vals='{{"status":"acknowledged"}}'
                    hx-target="#alert-modal-content" hx-swap="innerHTML">确认真实</button>
            <button class="btn btn-ghost"
                    onclick="document.getElementById('fp-form-{alert_id}').style.display='block'">标记误报</button>
            <button class="btn btn-ghost"
                    hx-post="/api/alerts/{alert_id}/workflow" hx-vals='{{"status":"investigating","assigned_to":"班组长"}}'
                    hx-target="#alert-modal-content" hx-swap="innerHTML">升级给班组长</button>
        </div>"""
    elif wf_status == "acknowledged":
        actions_html = f"""
        <div class="flex gap-8">
            <button class="btn btn-success btn-sm"
                    hx-post="/api/alerts/{alert_id}/workflow" hx-vals='{{"status":"resolved"}}'
                    hx-target="#alert-modal-content" hx-swap="innerHTML">标记已解决</button>
        </div>"""
    elif wf_status == "investigating":
        actions_html = f"""
        <div class="flex gap-8">
            <button class="btn btn-success btn-sm"
                    hx-post="/api/alerts/{alert_id}/workflow" hx-vals='{{"status":"resolved"}}'
                    hx-target="#alert-modal-content" hx-swap="innerHTML">标记已解决</button>
        </div>"""

    # False positive feedback form (hidden by default)
    fp_options = "".join(
        f'<option value="{val}">{label}</option>' for val, label in _FEEDBACK_CATEGORIES
    )
    fp_form = f"""
    <div id="fp-form-{alert_id}" class="card mt-16" style="display:none;border:1px solid var(--status-warn);">
        <h3 style="color:var(--status-warn-text);">标记为误报</h3>
        <form hx-post="/api/alerts/{alert_id}/workflow" hx-target="#alert-modal-content" hx-swap="innerHTML">
            <input type="hidden" name="status" value="false_positive">
            <div class="form-group">
                <label class="form-label">误报分类</label>
                <select name="category" class="form-select">{fp_options}</select>
            </div>
            <div class="form-group">
                <label class="form-label">备注（可选）</label>
                <textarea name="notes" class="form-textarea" rows="2" placeholder="简要描述误报原因..."></textarea>
            </div>
            <div class="flex gap-8">
                <button type="submit" class="btn btn-warning">确认标记误报</button>
                <button type="button" class="btn btn-ghost"
                        onclick="this.closest('[id^=fp-form]').style.display='none'">取消</button>
            </div>
        </form>
    </div>"""

    # Notes display
    notes_html = ""
    if alert.notes:
        notes_html = f"""
        <div class="card mt-16" style="border-left:3px solid var(--status-info);">
            <h3>备注</h3>
            <p style="font-size:var(--text-sm);color:var(--text-secondary);">{alert.notes}</p>
        </div>"""

    close_btn = (
        '<button class="modal-close" '
        "onclick=\"document.getElementById('alert-modal').classList.remove('active')\">"
        "&times;</button>"
    )

    return HTMLResponse(f"""
    {close_btn}
    <div style="display:flex;gap:var(--space-5);flex-wrap:wrap;">
        <div style="flex:1;min-width:400px;">
            {toggle_html}
            {img_html}
        </div>
        <div style="flex:0 0 320px;">
            <div class="flex-between mb-16">
                <h3 style="color:var(--status-info-text);">告警详情</h3>
                <span style="color:{wf_color};font-size:var(--text-sm);font-weight:var(--font-semibold);">{wf_label}</span>
            </div>
            <table>
                <tr><td style="color:var(--text-secondary);">告警ID</td>
                    <td class="mono" style="font-size:var(--text-xs);">{alert.alert_id}</td></tr>
                <tr><td style="color:var(--text-secondary);">时间</td><td>{ts}</td></tr>
                <tr><td style="color:var(--text-secondary);">摄像头</td><td>{alert.camera_id}</td></tr>
                <tr><td style="color:var(--text-secondary);">区域</td><td>{alert.zone_id}</td></tr>
                <tr><td style="color:var(--text-secondary);">严重度</td><td>{status_badge(alert.severity)}</td></tr>
                <tr><td style="color:var(--text-secondary);">异常分数</td><td>{alert.anomaly_score:.4f} ({score_pct})</td></tr>
            </table>

            <div style="margin-top:var(--space-5);">{actions_html}</div>
        </div>
    </div>
    {fp_form}
    {notes_html}""")


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
        <div class="flex gap-16 mb-16" style="font-size:var(--text-sm);color:var(--text-secondary);">
            <span>总计: <strong style="color:var(--text-primary);">{total_all}</strong></span>
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
        <span style="color:var(--text-secondary);font-size:var(--text-sm);">共 {total} 条告警</span>
        <div style="margin-left:auto;">
            <a href="/api/alerts/export-csv?{cam_param.lstrip('&')}{sev_param}" class="btn btn-ghost btn-sm" download>导出CSV</a>
        </div>
    </div>"""

    # Alert table with checkboxes
    rows = ""
    for a in alerts:
        ts = a.timestamp.strftime("%Y-%m-%d %H:%M:%S") if a.timestamp else ""

        thumb = '<span style="color:var(--text-tertiary);font-size:var(--text-xs);">—</span>'
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
            '<span style="color:var(--status-ok-text);font-size:var(--text-xs);">已确认</span>'
            if a.acknowledged else
            f'<button class="btn btn-sm btn-primary" '
            f'hx-post="/api/alerts/{a.alert_id}/acknowledge" hx-swap="outerHTML">确认</button>'
        )
        fp_html = (
            '<span style="color:var(--status-warn-text);font-size:var(--text-xs);">误报</span>'
            if a.false_positive else
            f'<button class="btn btn-sm btn-ghost" '
            f'hx-post="/api/alerts/{a.alert_id}/false-positive" hx-swap="outerHTML">标记</button>'
        )

        rows += f"""
        <tr>
            <td class="checkbox-cell"><input type="checkbox" name="alert_ids" value="{a.alert_id}" onchange="updateBulkBar()"></td>
            <td>{thumb}</td>
            <td class="mono" style="font-size:var(--text-xs);">{a.alert_id[:24]}</td>
            <td>{ts}</td>
            <td>{a.camera_id}</td>
            <td>{a.zone_id}</td>
            <td>{status_badge(a.severity)}</td>
            <td>{a.anomaly_score:.3f}</td>
            <td>{ack_html}</td>
            <td>{fp_html}</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="10" style="color:var(--text-tertiary);text-align:center;padding:var(--space-5);">暂无告警记录</td></tr>'

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
            <span style="color:var(--text-secondary);font-size:var(--text-sm);">第 {page}/{total_pages} 页</span>
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


# ── Workflow transition ──

@router.post("/{alert_id}/workflow")
async def alert_workflow_transition(request: Request, alert_id: str):
    """Transition alert workflow status (confirm/false_positive/resolve/escalate)."""
    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    # Accept both JSON and form data
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        data = await request.json()
    else:
        form = await request.form()
        data = dict(form)

    new_status = data.get("status", "")
    notes = data.get("notes", "")
    category = data.get("category", "")
    assigned_to = data.get("assigned_to", "")

    if category and notes:
        notes = f"[{category}] {notes}"
    elif category:
        notes = f"[{category}]"

    valid_statuses = {
        "acknowledged", "investigating", "resolved", "closed",
        "false_positive", "uncertain",
    }
    if new_status not in valid_statuses:
        return JSONResponse({"error": f"无效状态: {new_status}"}, status_code=400)

    success = db.update_alert_workflow(
        alert_id, new_status, notes=notes or None, assigned_to=assigned_to or None,
    )
    if not success:
        return JSONResponse({"error": "告警不存在"}, status_code=404)

    # Submit to feedback queue if applicable
    _submit_workflow_feedback(request, alert_id, new_status, category, notes)

    # Re-render the detail view
    return await alert_detail(request, alert_id)


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
    """Mark an alert as a false positive and submit to feedback queue.

    False positive feedback loop (A4-3 / Section 6): when a normal scene
    triggers a false alarm, the feedback is queued for retraining and the
    snapshot is copied to the baseline directory via FeedbackManager.
    """
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

        # Submit to feedback queue (handles baseline copy + queue entry)
        _submit_workflow_feedback(request, alert_id, "false_positive")

        return HTMLResponse('<span style="color:#ff9800;font-size:12px;">已标记误报</span>')
    return HTMLResponse('<span style="color:#f44336;font-size:12px;">操作失败</span>')


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
