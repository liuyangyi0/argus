"""Audit log dashboard routes (Chinese UI)."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from argus.dashboard.api_response import api_paginated, api_unavailable

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
        return api_unavailable("审计日志不可用")

    offset = (page - 1) * page_size
    logs = audit.get_logs(
        user=user or None, action=action or None, limit=page_size, offset=offset
    )
    total = audit.count_logs(user=user or None, action=action or None)

    return api_paginated("entries", [_audit_entry_to_dict(e) for e in logs], total, page, page_size)
