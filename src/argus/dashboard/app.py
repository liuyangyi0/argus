"""FastAPI dashboard application.

Provides a web UI for monitoring cameras, viewing alerts, and managing
the Argus system. Uses HTMX for dynamic server-driven updates with
minimal JavaScript. Full Chinese language interface.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from argus.dashboard.routes.alerts import router as alerts_router
from argus.dashboard.routes.audit import router as audit_router
from argus.dashboard.routes.backup import router as backup_router
from argus.dashboard.routes.cameras import router as cameras_router
from argus.dashboard.routes.config import router as config_router
from argus.dashboard.routes.detection import router as detection_router
from argus.dashboard.routes.system import router as system_router
from argus.dashboard.routes.tasks import router as tasks_router
from argus.dashboard.routes.reports import router as reports_router
from argus.dashboard.routes.users import router as users_router
from argus.dashboard.routes.zones import router as zones_router
from argus.dashboard.websocket import ConnectionManager, verify_ws_token

if TYPE_CHECKING:
    from argus.capture.manager import CameraManager
    from argus.core.health import HealthMonitor
    from argus.dashboard.tasks import TaskManager
    from argus.storage.database import Database


def create_app(
    database: Database | None = None,
    camera_manager: CameraManager | None = None,
    health_monitor: HealthMonitor | None = None,
    alerts_dir: str | None = None,
    config: object | None = None,
    config_path: str | None = None,
    task_manager: object | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Dependencies are injected via app.state so route handlers can access them.
    """
    from argus.config.schema import AuthConfig, DashboardConfig

    dashboard_config = getattr(config, "dashboard", None) or DashboardConfig()
    ws_manager = ConnectionManager(
        heartbeat_seconds=dashboard_config.websocket_heartbeat_seconds,
        max_connections=dashboard_config.websocket_max_connections,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await ws_manager.start()
        yield
        await ws_manager.stop()

    app = FastAPI(
        title="Argus - 核电站异物检测系统",
        description="Nuclear power plant foreign object visual detection system",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store dependencies in app state
    app.state.db = database
    app.state.camera_manager = camera_manager
    app.state.health_monitor = health_monitor
    app.state.alerts_dir = Path(alerts_dir) if alerts_dir else Path("data/alerts")
    app.state.config = config
    app.state.config_path = config_path
    app.state.task_manager = task_manager
    app.state.ws_manager = ws_manager

    # Security middleware (order matters: outermost first)
    from argus.dashboard.auth import (
        AuthMiddleware,
        RateLimitMiddleware,
        SecurityHeadersMiddleware,
        auth_router,
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests_per_minute=60)

    auth_config = getattr(config, "auth", None) or AuthConfig()
    app.add_middleware(
        AuthMiddleware,
        config=auth_config,
        session_secret=session_secret,
        database=database,
    )

    # Login/logout routes
    app.include_router(auth_router)

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routers
    app.include_router(cameras_router, prefix="/api/cameras", tags=["cameras"])
    app.include_router(alerts_router, prefix="/api/alerts", tags=["alerts"])
    app.include_router(zones_router, prefix="/api/zones", tags=["zones"])
    app.include_router(config_router, prefix="/api/config", tags=["config"])
    app.include_router(detection_router, prefix="/api/detection", tags=["detection"])
    app.include_router(system_router, prefix="/api/system", tags=["system"])
    app.include_router(tasks_router, prefix="/api/tasks", tags=["tasks"])
    app.include_router(audit_router, prefix="/api/audit", tags=["audit"])
    app.include_router(backup_router, prefix="/api/backup", tags=["backup"])
    app.include_router(users_router, prefix="/api/users", tags=["users"])
    app.include_router(reports_router, prefix="/api/reports", tags=["reports"])

    # Try to register baseline router (may not exist yet)
    try:
        from argus.dashboard.routes.baseline import router as baseline_router
        app.include_router(baseline_router, prefix="/api/baseline", tags=["baseline"])
    except ImportError:
        pass

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        # Authenticate via query param
        token = websocket.query_params.get("token", "")
        auth_cfg = getattr(config, "auth", None) or AuthConfig()
        if not verify_ws_token(token, auth_cfg):
            await websocket.close(code=4401, reason="Authentication required")
            return

        client_id = f"ws-{uuid.uuid4().hex[:8]}"
        try:
            client = await ws_manager.accept(websocket, client_id)
            await ws_manager.handle_client(client)
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(client_id)

    # Page routes
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return _render_page(request, "overview")

    @app.get("/cameras", response_class=HTMLResponse)
    async def cameras_page(request: Request):
        return _render_page(request, "cameras")

    @app.get("/alerts", response_class=HTMLResponse)
    async def alerts_page(request: Request):
        return _render_page(request, "alerts")

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        return _render_page(request, "models")

    @app.get("/system", response_class=HTMLResponse)
    async def system_page(request: Request):
        return _render_page(request, "system")

    # Legacy routes — redirect to merged locations
    _LEGACY_REDIRECTS = {
        "/baseline": "/models",
        "/zones": "/cameras",
        "/detection": "/cameras",
        "/config": "/system",
        "/backup": "/system",
        "/audit": "/system",
        "/reports": "/system",
        "/users": "/system",
    }
    for old_path, new_path in _LEGACY_REDIRECTS.items():
        app.add_api_route(
            old_path,
            lambda target=new_path: RedirectResponse(target, status_code=301),
            response_class=RedirectResponse,
        )

    return app


def _render_user_info(request: Request) -> str:
    """Render current user info and logout link for the nav bar."""
    user = getattr(request.state, "user", None)
    if not user:
        return ""
    role_labels = {"admin": "管理员", "operator": "操作员", "viewer": "观察者"}
    role = role_labels.get(user.get("role", ""), user.get("role", ""))
    username = user.get("username", "")
    return (
        f'<span style="color:var(--text-secondary);font-size:var(--text-xs);">'
        f'{username} ({role})</span>'
        f'<a href="/logout" style="color:var(--text-secondary);font-size:var(--text-xs);'
        f'margin-left:var(--space-2);text-decoration:none;">退出</a>'
    )


_NAV_ICONS: dict[str, str] = {
    "overview": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "cameras": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    "alerts": '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>',
    "models": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "system": '<rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>',
}

_NAV_ITEMS = [
    ("overview", "/", "总览"),
    ("cameras", "/cameras", "摄像头"),
    ("alerts", "/alerts", "告警"),
    ("models", "/models", "模型"),
    ("system", "/system", "系统"),
]

_CONTENT_URL_MAP = {
    "overview": "/api/system/overview",
    "cameras": "/api/cameras",
    "alerts": "/api/alerts",
    "models": "/api/baseline",
    "system": "/api/system",
}


def _render_page(request: Request, active_page: str) -> HTMLResponse:
    """Render the main HTML shell with sidebar layout and 5-page navigation."""
    nav_html = ""
    for page_id, href, label in _NAV_ITEMS:
        active_cls = ' class="active"' if page_id == active_page else ""
        icon_path = _NAV_ICONS.get(page_id, "")
        icon_svg = f'<svg viewBox="0 0 24 24">{icon_path}</svg>' if icon_path else ""
        nav_html += f'<a href="{href}"{active_cls}>{icon_svg}{label}</a>\n'

    content_url = _CONTENT_URL_MAP.get(active_page, f"/api/{active_page}")

    # Task indicator in sidebar footer
    task_manager = getattr(request.app.state, "task_manager", None)
    task_count = 0
    if task_manager:
        task_count = sum(1 for t in task_manager.get_active_tasks()
                         if t.status.value in ("pending", "running"))
    task_indicator = ""
    if task_count > 0:
        task_indicator = (
            f'<div class="task-indicator">{task_count} 个任务运行中</div>'
        )

    user_info = _render_user_info(request)

    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Argus - 核电站异物检测系统</title>
    <link rel="stylesheet" href="/static/css/tokens.css">
    <link rel="stylesheet" href="/static/css/argus.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
</head>
<body>
    <div class="app-layout">
        <aside class="sidebar">
            <div class="sidebar-logo">ARGUS</div>
            <nav class="sidebar-nav">
                {nav_html}
            </nav>
            <div class="sidebar-footer">
                {task_indicator}
                {user_info}
                <div style="margin-top:var(--space-2);">v0.2.0</div>
            </div>
        </aside>
        <div class="main-content">
            <div class="container">
                <div id="content"
                     hx-get="{content_url}"
                     hx-trigger="load"
                     hx-swap="innerHTML">
                    <div class="empty-state"><div class="spinner"></div><div class="message mt-8">加载中...</div></div>
                </div>
            </div>
        </div>
    </div>
    <div id="alert-modal" class="modal-overlay"
         onclick="if(event.target===this)this.classList.remove('active')">
        <div id="alert-modal-content" class="modal-content">
            <p style="color:var(--text-secondary);">加载中...</p>
        </div>
    </div>
    <div id="toast-container" class="toast-container"></div>
    <script src="/static/js/toast.js"></script>
    <script src="/static/js/ws-client.js"></script>
    <script src="/static/js/zone_editor.js"></script>
    <script src="/static/js/notifications.js"></script>
    <script src="/static/js/keyboard.js"></script>
</body>
</html>""")
