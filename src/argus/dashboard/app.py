"""FastAPI dashboard application.

Provides a web UI for monitoring cameras, viewing alerts, and managing
the Argus system. Uses HTMX for dynamic server-driven updates with
minimal JavaScript. Full Chinese language interface.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from argus.dashboard.routes.alerts import router as alerts_router
from argus.dashboard.routes.cameras import router as cameras_router
from argus.dashboard.routes.config import router as config_router
from argus.dashboard.routes.detection import router as detection_router
from argus.dashboard.routes.system import router as system_router
from argus.dashboard.routes.tasks import router as tasks_router
from argus.dashboard.routes.zones import router as zones_router

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
    from argus.config.schema import AuthConfig

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

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

    # Security middleware (order matters: outermost first)
    from argus.dashboard.auth import (
        AuthMiddleware,
        RateLimitMiddleware,
        SecurityHeadersMiddleware,
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests_per_minute=60)

    auth_config = getattr(config, "auth", None) or AuthConfig()
    app.add_middleware(AuthMiddleware, config=auth_config)

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

    # Try to register baseline router (may not exist yet)
    try:
        from argus.dashboard.routes.baseline import router as baseline_router
        app.include_router(baseline_router, prefix="/api/baseline", tags=["baseline"])
    except ImportError:
        pass

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

    @app.get("/baseline", response_class=HTMLResponse)
    async def baseline_page(request: Request):
        return _render_page(request, "baseline")

    @app.get("/zones", response_class=HTMLResponse)
    async def zones_page(request: Request):
        return _render_page(request, "zones")

    @app.get("/detection", response_class=HTMLResponse)
    async def detection_page(request: Request):
        return _render_page(request, "detection")

    @app.get("/config", response_class=HTMLResponse)
    async def config_page(request: Request):
        return _render_page(request, "config")

    @app.get("/system", response_class=HTMLResponse)
    async def system_page(request: Request):
        return _render_page(request, "system")

    return app


def _render_page(request: Request, active_page: str) -> HTMLResponse:
    """Render the main HTML shell with the active page content."""
    nav_items = [
        ("overview", "/", "总览"),
        ("cameras", "/cameras", "摄像头"),
        ("baseline", "/baseline", "基线与模型"),
        ("zones", "/zones", "检测区域"),
        ("alerts", "/alerts", "告警中心"),
        ("detection", "/detection", "检测调试"),
        ("config", "/config", "系统设置"),
    ]

    nav_html = ""
    for page_id, href, label in nav_items:
        active_cls = ' class="active"' if page_id == active_page else ""
        nav_html += f'<a href="{href}"{active_cls}>{label}</a>\n'

    # Map page to initial content URL
    content_url_map = {
        "overview": "/api/system/overview",
        "cameras": "/api/cameras",
        "baseline": "/api/baseline",
        "zones": "/api/zones",
        "alerts": "/api/alerts",
        "detection": "/api/detection",
        "config": "/api/config",
        "system": "/api/system",
    }
    content_url = content_url_map.get(active_page, f"/api/{active_page}")

    # Task indicator: show active task count in nav
    task_manager = getattr(request.app.state, "task_manager", None)
    task_count = 0
    if task_manager:
        task_count = sum(1 for t in task_manager.get_active_tasks()
                         if t.status.value in ("pending", "running"))
    task_indicator = ""
    if task_count > 0:
        task_indicator = (
            f'<span style="background:#e65100;color:#fff;padding:2px 8px;'
            f'border-radius:10px;font-size:11px;font-weight:600;">'
            f'{task_count} 个任务运行中</span>'
        )

    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Argus - 核电站异物检测系统</title>
    <link rel="stylesheet" href="/static/css/argus.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
</head>
<body>
    <nav>
        <span class="logo">ARGUS</span>
        {nav_html}
        <div class="nav-right">
            {task_indicator}
        </div>
    </nav>
    <div class="container">
        <div id="content"
             hx-get="{content_url}"
             hx-trigger="load"
             hx-swap="innerHTML">
            <div class="empty-state"><div class="spinner"></div><div class="message mt-8">加载中...</div></div>
        </div>
    </div>
    <div id="alert-modal" class="modal-overlay"
         onclick="if(event.target===this)this.classList.remove('active')">
        <div id="alert-modal-content" class="modal-content">
            <p style="color:#8890a0;">加载中...</p>
        </div>
    </div>
    <div id="toast-container" class="toast-container"></div>
    <script src="/static/js/toast.js"></script>
    <script src="/static/js/zone_editor.js"></script>
    <script src="/static/js/alert_audio.js"></script>
</body>
</html>""")
