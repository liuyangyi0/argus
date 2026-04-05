"""Audit log dashboard routes (Chinese UI)."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from argus.dashboard.components import empty_state, page_header

router = APIRouter()

# Human-readable labels for action codes
_ACTION_LABELS: dict[str, str] = {
    "acknowledge_alert": "确认告警",
    "mark_false_positive": "标记误报",
    "update_zone": "更新区域",
    "update_config": "修改配置",
    "deploy_model": "部署模型",
    "start_camera": "启动摄像头",
    "stop_camera": "停止摄像头",
    "capture_baseline": "采集基线",
    "train_model": "训练模型",
    "clear_lock": "清除锁定",
    "bulk_acknowledge": "批量确认",
    "bulk_false_positive": "批量标记误报",
}


def _audit_entry_to_dict(entry) -> dict:
    """Convert an AuditLog ORM object to a JSON-serializable dict."""
    return {
        "id": entry.id,
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        "user": entry.user,
        "action": entry.action,
        "target_type": entry.target_type,
        "target_id": entry.target_id or "",
        "details": entry.detail or "",
        "ip_address": entry.ip_address or "",
    }


# ── JSON API endpoint (for Vue frontend) ──


@router.get("/json")
async def audit_json(
    request: Request,
    user: str | None = Query(None),
    action: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """JSON API: paginated audit log entries with optional filters."""
    audit = getattr(request.app.state, "audit_logger", None)
    if not audit:
        return JSONResponse({"error": "审计日志不可用"}, status_code=503)

    offset = (page - 1) * page_size
    logs = audit.get_logs(
        user=user or None, action=action or None, limit=page_size, offset=offset
    )
    total = audit.count_logs(user=user or None, action=action or None)

    return JSONResponse({
        "entries": [_audit_entry_to_dict(e) for e in logs],
        "total": total,
        "page": page,
        "page_size": page_size,
    })


# ── HTML endpoint (legacy HTMX) ──


@router.get("", response_class=HTMLResponse)
async def audit_logs(
    request: Request,
    user: str | None = Query(None),
    action: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Audit log list page with filtering and pagination."""
    audit = getattr(request.app.state, "audit_logger", None)
    if not audit:
        return HTMLResponse(empty_state("审计日志不可用"))

    offset = (page - 1) * page_size
    logs = audit.get_logs(user=user or None, action=action or None, limit=page_size, offset=offset)
    total = audit.count_logs(user=user or None, action=action or None)

    # Collect distinct users for dropdown
    all_logs_sample = audit.get_logs(limit=1000)
    users = sorted({e.user for e in all_logs_sample if e.user})

    # Filter controls
    user_options = '<option value="">全部用户</option>'
    for u in users:
        sel = "selected" if u == (user or "") else ""
        user_options += f'<option value="{u}" {sel}>{u}</option>'

    action_options = '<option value="">全部操作</option>'
    for action_code, action_label in sorted(_ACTION_LABELS.items(), key=lambda x: x[1]):
        sel = "selected" if action_code == (action or "") else ""
        action_options += f'<option value="{action_code}" {sel}>{action_label}</option>'

    user_param = f"&user={user}" if user else ""
    action_param = f"&action={action}" if action else ""
    base_url = "/audit?"

    filters_html = f"""
    <div class="flex gap-12 mb-16" style="flex-wrap:wrap;align-items:center;">
        <select class="form-select" style="width:auto;"
                onchange="window.location.href='{base_url}user='+this.value+'{action_param}'">
            {user_options}
        </select>
        <select class="form-select" style="width:auto;"
                onchange="window.location.href='{base_url}action='+this.value+'{user_param}'">
            {action_options}
        </select>
        <span style="color:#8890a0;font-size:13px;">共 {total} 条记录</span>
    </div>"""

    # Table rows
    rows = ""
    for entry in logs:
        ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S") if entry.timestamp else ""
        action_label = _ACTION_LABELS.get(entry.action, entry.action)
        target_id = entry.target_id or "—"
        detail = entry.detail or "—"
        ip = entry.ip_address or "—"

        rows += f"""
        <tr>
            <td style="font-size:12px;">{ts}</td>
            <td>{entry.user}</td>
            <td>{action_label}</td>
            <td>{entry.target_type}</td>
            <td style="font-size:12px;color:#8890a0;">{target_id}</td>
            <td style="font-size:12px;color:#8890a0;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{detail}</td>
            <td style="font-size:12px;color:#8890a0;">{ip}</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="7" style="color:#616161;text-align:center;padding:24px;">暂无审计记录</td></tr>'

    # Pagination
    total_pages = max(1, (total + page_size - 1) // page_size)
    pagination = ""
    if total_pages > 1:
        prev_cls = ' style="opacity:0.3;pointer-events:none;"' if page <= 1 else ""
        next_cls = ' style="opacity:0.3;pointer-events:none;"' if page >= total_pages else ""
        params = user_param + action_param
        pagination = f"""
        <div style="display:flex;justify-content:center;gap:12px;margin-top:16px;align-items:center;">
            <a href="/audit?page={page-1}{params}" class="btn btn-sm btn-ghost"{prev_cls}>上一页</a>
            <span style="color:#8890a0;font-size:13px;">第 {page}/{total_pages} 页</span>
            <a href="/audit?page={page+1}{params}" class="btn btn-sm btn-ghost"{next_cls}>下一页</a>
        </div>"""

    is_htmx = request.headers.get("HX-Request") == "true"
    header = "" if is_htmx else page_header("审计日志", "系统操作记录与追踪")

    return HTMLResponse(f"""
    {header}
    {filters_html}
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>时间</th><th>用户</th><th>操作</th><th>目标类型</th>
                    <th>目标ID</th><th>详情</th><th>IP地址</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        {pagination}
    </div>""")
