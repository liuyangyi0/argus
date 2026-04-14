"""User management routes — admin-only RBAC panel (Chinese UI)."""

from __future__ import annotations

from fastapi import APIRouter, Request

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

_ROLE_LABELS = {
    "admin": "管理员",
    "operator": "操作员",
    "viewer": "观察者",
}


def _user_to_dict(u) -> dict:
    """Convert a User ORM object to a JSON-serializable dict."""
    return {
        "username": u.username,
        "role": u.role,
        "display_name": u.display_name or "",
        "active": u.active,
        "last_login": u.last_login.isoformat() if u.last_login else None,
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }


# ── JSON API endpoints (for Vue frontend) ──


@router.get("/json")
async def users_json(request: Request):
    """JSON API: list all users (admin only)."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    users = db.get_all_users()
    return api_success({"users": [_user_to_dict(u) for u in users]})


@router.post("/json")
async def create_user_json(request: Request):
    """JSON API: create a new user (admin only)."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    body = await request.json()
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    display_name = str(body.get("display_name", "")).strip() or None
    role = str(body.get("role", "viewer"))

    if not username or not password:
        return api_validation_error("用户名和密码不能为空")
    if role not in ("admin", "operator", "viewer"):
        return api_validation_error("角色无效")
    if len(password) < 6:
        return api_validation_error("密码至少6位")

    if db.get_user(username) is not None:
        return api_conflict(f"用户 {username} 已存在")

    db.create_user(username, hash_password(password), role, display_name)
    return api_success({"username": username})


@router.delete("/{username}/json")
async def delete_user_json(request: Request, username: str):
    """JSON API: delete a user (admin only)."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    current_user = getattr(request.state, "user", {})
    if username == current_user.get("username"):
        return api_validation_error("不能删除当前登录用户")

    if db.get_user(username) is None:
        return api_not_found("用户不存在")

    db.delete_user(username)
    return api_success({"username": username})


@router.post("/{username}/toggle-active/json")
async def toggle_active_json(request: Request, username: str):
    """JSON API: toggle user active status (admin only)."""
    if not require_role(request, "admin"):
        return api_forbidden("权限不足")

    db = request.app.state.db
    user = db.get_user(username)
    if user is None:
        return api_not_found("用户不存在")

    db.update_user(username, active=not user.active)
    return api_success({
        "username": username,
        "active": not user.active,
    })


