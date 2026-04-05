"""User management routes — admin-only RBAC panel (Chinese UI)."""

from __future__ import annotations

import html
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from argus.dashboard.auth import hash_password, require_role
from argus.dashboard.components import empty_state, page_header

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
        return JSONResponse({"error": "权限不足"}, status_code=403)

    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    users = db.get_all_users()
    return JSONResponse({"users": [_user_to_dict(u) for u in users]})


@router.post("/json")
async def create_user_json(request: Request):
    """JSON API: create a new user (admin only)."""
    if not require_role(request, "admin"):
        return JSONResponse({"error": "权限不足"}, status_code=403)

    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    body = await request.json()
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    display_name = str(body.get("display_name", "")).strip() or None
    role = str(body.get("role", "viewer"))

    if not username or not password:
        return JSONResponse({"error": "用户名和密码不能为空"}, status_code=400)
    if role not in ("admin", "operator", "viewer"):
        return JSONResponse({"error": "角色无效"}, status_code=400)
    if len(password) < 6:
        return JSONResponse({"error": "密码至少6位"}, status_code=400)

    if db.get_user(username) is not None:
        return JSONResponse({"error": f"用户 {username} 已存在"}, status_code=400)

    db.create_user(username, hash_password(password), role, display_name)
    return JSONResponse({"success": True, "username": username, "message": "User created"})


@router.delete("/{username}/json")
async def delete_user_json(request: Request, username: str):
    """JSON API: delete a user (admin only)."""
    if not require_role(request, "admin"):
        return JSONResponse({"error": "权限不足"}, status_code=403)

    db = request.app.state.db
    current_user = getattr(request.state, "user", {})
    if username == current_user.get("username"):
        return JSONResponse({"error": "不能删除当前登录用户"}, status_code=400)

    if db.get_user(username) is None:
        return JSONResponse({"error": "用户不存在"}, status_code=404)

    db.delete_user(username)
    return JSONResponse({"success": True, "message": "User deleted"})


@router.post("/{username}/toggle-active/json")
async def toggle_active_json(request: Request, username: str):
    """JSON API: toggle user active status (admin only)."""
    if not require_role(request, "admin"):
        return JSONResponse({"error": "权限不足"}, status_code=403)

    db = request.app.state.db
    user = db.get_user(username)
    if user is None:
        return JSONResponse({"error": "用户不存在"}, status_code=404)

    db.update_user(username, active=not user.active)
    return JSONResponse({
        "success": True,
        "username": username,
        "active": not user.active,
        "message": f"User {'deactivated' if user.active else 'activated'}",
    })


# ── HTML endpoints (legacy HTMX) ──


@router.get("", response_class=HTMLResponse)
async def users_page(request: Request):
    """User management page (admin only)."""
    if not require_role(request, "admin"):
        return HTMLResponse(
            '<div class="card"><p style="color:#f44336;">权限不足：仅管理员可访问用户管理页面。</p></div>',
            status_code=403,
        )

    db = request.app.state.db
    if not db:
        return HTMLResponse(empty_state("数据库不可用"))

    users = db.get_all_users()
    current_user = getattr(request.state, "user", {})
    current_username = current_user.get("username", "")

    rows = ""
    for u in users:
        last_login = (
            u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else "从未"
        )
        role_label = _ROLE_LABELS.get(u.role, u.role)
        active_label = (
            '<span style="color:#4caf50;">启用</span>'
            if u.active
            else '<span style="color:#f44336;">禁用</span>'
        )

        safe_username = html.escape(u.username)
        safe_display = html.escape(u.display_name) if u.display_name else "—"

        # Don't allow deleting yourself
        delete_btn = ""
        if u.username != current_username:
            delete_btn = (
                f'<button class="btn btn-danger btn-sm" style="margin-left:4px;" '
                f'hx-delete="/api/users/{safe_username}" '
                f'hx-target="#users-table-body" hx-swap="innerHTML" '
                f'hx-confirm="确定删除用户 {safe_username}？此操作不可撤销。">删除</button>'
            )

        toggle_active_label = "禁用" if u.active else "启用"
        toggle_btn = (
            f'<button class="btn btn-ghost btn-sm" '
            f'hx-post="/api/users/{safe_username}/toggle-active" '
            f'hx-target="#users-table-body" hx-swap="innerHTML">{toggle_active_label}</button>'
        )

        rows += f"""
        <tr>
            <td>{safe_username}</td>
            <td>{safe_display}</td>
            <td>{role_label}</td>
            <td>{active_label}</td>
            <td>{last_login}</td>
            <td>
                {toggle_btn}
                {delete_btn}
            </td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="6" style="color:#616161;text-align:center;padding:24px;">暂无用户</td></tr>'

    role_options = "".join(
        f'<option value="{k}">{v}</option>' for k, v in _ROLE_LABELS.items()
    )

    is_htmx = request.headers.get("HX-Request") == "true"
    header = "" if is_htmx else page_header("用户管理", "管理系统用户和角色权限")

    return HTMLResponse(f"""
    {header}
    <div class="card" style="margin-bottom:16px;">
        <h3>添加用户</h3>
        <form hx-post="/api/users" hx-target="#users-table-body" hx-swap="innerHTML"
              hx-on::after-request="if(event.detail.successful)this.reset()">
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">用户名</label>
                    <input type="text" name="username" class="form-input"
                           placeholder="admin" required maxlength="50">
                </div>
                <div class="form-group">
                    <label class="form-label">显示名</label>
                    <input type="text" name="display_name" class="form-input"
                           placeholder="管理员" maxlength="100">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">密码</label>
                    <input type="password" name="password" class="form-input"
                           placeholder="初始密码" required minlength="6">
                </div>
                <div class="form-group">
                    <label class="form-label">角色</label>
                    <select name="role" class="form-select">
                        {role_options}
                    </select>
                </div>
            </div>
            <div class="form-actions">
                <button type="submit" class="btn btn-primary">添加用户</button>
            </div>
        </form>
    </div>
    <div class="card">
        <h3>用户列表</h3>
        <table>
            <thead>
                <tr>
                    <th>用户名</th><th>显示名</th><th>角色</th>
                    <th>状态</th><th>最后登录</th><th>操作</th>
                </tr>
            </thead>
            <tbody id="users-table-body">{rows}</tbody>
        </table>
    </div>""")


@router.post("")
async def create_user(request: Request):
    """Create a new user (admin only). Returns updated table body HTML."""
    if not require_role(request, "admin"):
        return JSONResponse({"error": "权限不足"}, status_code=403)

    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    display_name = str(form.get("display_name", "")).strip() or None
    role = str(form.get("role", "viewer"))

    if not username or not password:
        return JSONResponse({"error": "用户名和密码不能为空"}, status_code=400)
    if role not in ("admin", "operator", "viewer"):
        return JSONResponse({"error": "角色无效"}, status_code=400)
    if len(password) < 6:
        return JSONResponse({"error": "密码至少6位"}, status_code=400)

    if db.get_user(username) is not None:
        return JSONResponse({"error": f"用户 {username} 已存在"}, status_code=400)

    db.create_user(username, hash_password(password), role, display_name)
    return _users_table_rows_response(db)


@router.delete("/{username}")
async def delete_user(request: Request, username: str):
    """Delete a user (admin only). Returns updated table body HTML."""
    if not require_role(request, "admin"):
        return JSONResponse({"error": "权限不足"}, status_code=403)

    db = request.app.state.db
    current_user = getattr(request.state, "user", {})
    if username == current_user.get("username"):
        return JSONResponse({"error": "不能删除当前登录用户"}, status_code=400)

    db.delete_user(username)
    return _users_table_rows_response(db)


@router.post("/{username}/toggle-active")
async def toggle_active(request: Request, username: str):
    """Toggle user active status (admin only). Returns updated table body HTML."""
    if not require_role(request, "admin"):
        return JSONResponse({"error": "权限不足"}, status_code=403)

    db = request.app.state.db
    user = db.get_user(username)
    if user is None:
        return JSONResponse({"error": "用户不存在"}, status_code=404)

    db.update_user(username, active=not user.active)
    return _users_table_rows_response(db)


def _users_table_rows_response(db) -> HTMLResponse:
    """Render the <tbody> rows for the users table."""
    users = db.get_all_users()
    rows = ""
    for u in users:
        last_login = u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else "从未"
        role_label = _ROLE_LABELS.get(u.role, u.role)
        safe_username = html.escape(u.username)
        safe_display = html.escape(u.display_name) if u.display_name else "—"
        active_label = (
            '<span style="color:#4caf50;">启用</span>'
            if u.active
            else '<span style="color:#f44336;">禁用</span>'
        )
        toggle_active_label = "禁用" if u.active else "启用"
        toggle_btn = (
            f'<button class="btn btn-ghost btn-sm" '
            f'hx-post="/api/users/{safe_username}/toggle-active" '
            f'hx-target="#users-table-body" hx-swap="innerHTML">{toggle_active_label}</button>'
        )
        delete_btn = (
            f'<button class="btn btn-danger btn-sm" style="margin-left:4px;" '
            f'hx-delete="/api/users/{safe_username}" '
            f'hx-target="#users-table-body" hx-swap="innerHTML" '
            f'hx-confirm="确定删除用户 {safe_username}？此操作不可撤销。">删除</button>'
        )
        rows += f"""
        <tr>
            <td>{safe_username}</td>
            <td>{safe_display}</td>
            <td>{role_label}</td>
            <td>{active_label}</td>
            <td>{last_login}</td>
            <td>{toggle_btn}{delete_btn}</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="6" style="color:#616161;text-align:center;padding:24px;">暂无用户</td></tr>'

    return HTMLResponse(rows)
