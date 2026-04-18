"""User management routes for the system settings panel."""

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
from argus.dashboard.auth import hash_password, require_role

router = APIRouter()


def _user_to_dict(user) -> dict:
    """Convert a User ORM object to a JSON-serializable dict."""
    return {
        "username": user.username,
        "role": user.role,
        "display_name": user.display_name or "",
        "active": user.active,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _audit(request: Request, action: str, target_id: str, detail: str) -> None:
    audit = getattr(request.app.state, "audit_logger", None)
    if not audit:
        return

    current_user = getattr(request.state, "user", {}) or {}
    audit.log(
        user=current_user.get("username", "unknown"),
        action=action,
        target_type="user",
        target_id=target_id,
        detail=detail,
        ip_address=getattr(request.client, "host", "") or "",
    )


@router.get("/json")
async def users_json(
    request: Request,
    username: str | None = Query(None),
    display_name: str | None = Query(None),
    role: str | None = Query(None),
):
    """JSON API: list users with optional filters."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    users = db.get_all_users()
    username_filter = (username or "").strip().lower()
    display_name_filter = (display_name or "").strip().lower()
    role_filter = (role or "").strip()

    if username_filter:
        users = [user for user in users if username_filter in user.username.lower()]
    if display_name_filter:
        users = [
            user for user in users
            if display_name_filter in (user.display_name or "").lower()
        ]
    if role_filter:
        users = [user for user in users if user.role == role_filter]

    return api_success({"users": [_user_to_dict(user) for user in users]})


@router.post("/json")
async def create_user_json(request: Request):
    """JSON API: create a new user."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    body = await request.json()
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    display_name = str(body.get("display_name", "")).strip() or None
    role = str(body.get("role", "viewer")).strip()
    active = bool(body.get("active", True))

    if not username or not password:
        return api_validation_error("用户名和密码不能为空")
    if role not in ("admin", "operator", "viewer"):
        return api_validation_error("角色无效")
    if len(password) < 6:
        return api_validation_error("密码至少 6 位")
    if db.get_user(username) is not None:
        return api_conflict(f"用户 {username} 已存在")

    user = db.create_user(username, hash_password(password), role, display_name)
    if not active:
        db.update_user(username, active=False)
        refreshed = db.get_user(username)
        if refreshed is not None:
            user = refreshed

    _audit(request, "create_user", username, f"新增用户 {username}")
    return api_success({"user": _user_to_dict(user)})


@router.put("/{username}/json")
async def update_user_json(request: Request, username: str):
    """JSON API: update a user."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    user = db.get_user(username)
    if user is None:
        return api_not_found("用户不存在")

    body = await request.json()
    display_name = str(body.get("display_name", "")).strip() or None
    role = str(body.get("role", user.role)).strip()
    password = str(body.get("password", ""))
    active = bool(body.get("active", user.active))
    current_user = getattr(request.state, "user", {}) or {}

    if role not in ("admin", "operator", "viewer"):
        return api_validation_error("角色无效")
    if password and len(password) < 6:
        return api_validation_error("密码至少 6 位")
    if username == current_user.get("username") and not active:
        return api_validation_error("不能禁用当前登录用户")

    update_fields: dict[str, object] = {
        "display_name": display_name,
        "role": role,
        "active": active,
    }
    if password:
        update_fields["password_hash"] = hash_password(password)

    db.update_user(username, **update_fields)
    updated = db.get_user(username)
    assert updated is not None

    _audit(request, "update_user", username, f"编辑用户 {username}")
    return api_success({"user": _user_to_dict(updated)})


@router.delete("/{username}/json")
async def delete_user_json(request: Request, username: str):
    """JSON API: delete a user."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    current_user = getattr(request.state, "user", {}) or {}
    if username == current_user.get("username"):
        return api_validation_error("不能删除当前登录用户")
    if db.get_user(username) is None:
        return api_not_found("用户不存在")

    db.delete_user(username)
    _audit(request, "delete_user", username, f"删除用户 {username}")
    return api_success({"username": username})


@router.post("/{username}/toggle-active/json")
async def toggle_active_json(request: Request, username: str):
    """JSON API: toggle user active status."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    user = db.get_user(username)
    if user is None:
        return api_not_found("用户不存在")

    current_user = getattr(request.state, "user", {}) or {}
    if username == current_user.get("username") and user.active:
        return api_validation_error("不能禁用当前登录用户")

    db.update_user(username, active=not user.active)
    _audit(request, "toggle_user_active", username, f"切换用户状态 {username}")
    return api_success({
        "username": username,
        "active": not user.active,
    })
