"""Region management routes for the system settings panel."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from argus.dashboard.api_response import (
    api_conflict,
    api_forbidden,
    api_not_found,
    api_success,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.auth import require_role

router = APIRouter()
_NOTIFICATION_TEMPLATE_METHODS = {"email", "sms", "webhook"}


def _parse_notification_methods(value: object) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []

    methods: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        method = str(item).strip()
        if not method or method in seen:
            continue
        seen.add(method)
        methods.append(method)
    return methods


def _parse_notification_template_ids(value: object) -> list[int]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []

    template_ids: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        try:
            template_id = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if template_id <= 0 or template_id in seen:
            continue
        seen.add(template_id)
        template_ids.append(template_id)
    return template_ids


def _notification_template_to_summary(template) -> dict:
    return {
        "id": template.id,
        "name": template.name,
        "method": template.method,
        "enabled": bool(template.enabled),
    }


def _region_to_dict(region, db) -> dict:
    methods = _parse_notification_methods(region.notification_methods)
    template_ids = _parse_notification_template_ids(region.notification_template_ids)
    templates = [
        _notification_template_to_summary(item)
        for item in db.get_notification_templates_by_ids(template_ids)
    ]
    return {
        "id": region.id,
        "name": region.name,
        "owner": region.owner,
        "email": region.email or "",
        "phone": region.phone or "",
        "notification_methods": methods,
        "notification_template_ids": template_ids,
        "notification_templates": templates,
        "notification_methods_text": "、".join(methods),
        "created_at": region.created_at.isoformat() if region.created_at else None,
        "updated_at": region.updated_at.isoformat() if region.updated_at else None,
    }


def _normalize_region_payload(body: dict, db) -> tuple[dict | None, str | None]:
    if not isinstance(body, dict):
        return None, "请求体必须是 JSON 对象"

    name = str(body.get("name", "")).strip()
    owner = str(body.get("owner", "")).strip()
    email = str(body.get("email", "")).strip() or None
    phone = str(body.get("phone", "")).strip() or None
    methods = _parse_notification_methods(body.get("notification_methods", []))
    template_ids = _parse_notification_template_ids(body.get("notification_template_ids", []))

    if not name:
        return None, "区域名称不能为空"
    if not owner:
        return None, "负责人不能为空"
    if email and "@" not in email:
        return None, "邮箱格式不正确"

    if template_ids:
        templates = db.get_notification_templates_by_ids(template_ids)
        if len(templates) != len(template_ids):
            return None, "存在无效的通知内容配置"
        if any(template.method not in _NOTIFICATION_TEMPLATE_METHODS or template.method not in methods for template in templates):
            return None, "所选通知内容配置与通知方式不匹配"

    return {
        "name": name,
        "owner": owner,
        "email": email,
        "phone": phone,
        "notification_methods": ",".join(methods),
        "notification_template_ids": ",".join(str(template_id) for template_id in template_ids),
    }, None


def _audit(request: Request, action: str, target_id: str, detail: str) -> None:
    audit = getattr(request.app.state, "audit_logger", None)
    if not audit:
        return
    current_user = getattr(request.state, "user", {}) or {}
    audit.log(
        user=current_user.get("username", "unknown"),
        action=action,
        target_type="region",
        target_id=target_id,
        detail=detail,
        ip_address=getattr(request.client, "host", "") or "",
    )


@router.get("/json")
async def regions_json(
    request: Request,
    name: str | None = Query(None),
    owner: str | None = Query(None),
    phone: str | None = Query(None),
    email: str | None = Query(None),
):
    """JSON API: list regions with optional filters."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    regions = db.get_regions(
        name=(name or "").strip() or None,
        owner=(owner or "").strip() or None,
        phone=(phone or "").strip() or None,
        email=(email or "").strip() or None,
    )
    return api_success({"regions": [_region_to_dict(region, db) for region in regions]})


@router.post("/json")
async def create_region_json(request: Request):
    """JSON API: create a region entry."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    body = await request.json()
    payload, error = _normalize_region_payload(body, db)
    if error:
        return api_validation_error(error)

    assert payload is not None
    exists = db.get_region_by_name(payload["name"])
    if exists is not None:
        return api_conflict(f"区域 {payload['name']} 已存在")

    region = db.create_region(**payload)
    _audit(request, "create_region", str(region.id), f"新增区域 {region.name}")
    return api_success({"region": _region_to_dict(region, db)})


@router.put("/{region_id}/json")
async def update_region_json(request: Request, region_id: int):
    """JSON API: update a region entry."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    region = db.get_region(region_id)
    if region is None:
        return api_not_found("区域不存在")

    body = await request.json()
    payload, error = _normalize_region_payload(body, db)
    if error:
        return api_validation_error(error)

    assert payload is not None
    exists = db.get_region_by_name(payload["name"])
    if exists is not None and exists.id != region_id:
        return api_conflict(f"区域 {payload['name']} 已存在")

    db.update_region(region_id, **payload)
    updated = db.get_region(region_id)
    assert updated is not None
    _audit(request, "update_region", str(region_id), f"编辑区域 {updated.name}")
    return api_success({"region": _region_to_dict(updated, db)})


@router.delete("/{region_id}/json")
async def delete_region_json(request: Request, region_id: int):
    """JSON API: delete a region entry."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    region = db.get_region(region_id)
    if region is None:
        return api_not_found("区域不存在")

    db.delete_region(region_id)
    _audit(request, "delete_region", str(region_id), f"删除区域 {region.name}")
    return api_success({"id": region_id})
