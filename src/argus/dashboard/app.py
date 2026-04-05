"""FastAPI application — JSON API + Vue SPA static serving.

Backend provides /api/* JSON endpoints and /ws WebSocket.
Frontend is a Vue 3 + Ant Design SPA served from web/dist/.
"""

from __future__ import annotations

import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
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
    """Create and configure the FastAPI application."""
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
        version="0.2.0",
        lifespan=lifespan,
    )

    # ── App state (dependency injection) ──
    app.state.db = database
    app.state.camera_manager = camera_manager
    app.state.health_monitor = health_monitor
    app.state.alerts_dir = Path(alerts_dir) if alerts_dir else Path("data/alerts")
    app.state.config = config
    app.state.config_path = config_path
    app.state.task_manager = task_manager
    app.state.ws_manager = ws_manager

    # ── Security middleware ──
    from argus.dashboard.auth import (
        AuthMiddleware,
        RateLimitMiddleware,
        SecurityHeadersMiddleware,
        auth_router,
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests_per_minute=60)

    auth_config = getattr(config, "auth", None) or AuthConfig()
    session_secret = secrets.token_hex(32)
    app.add_middleware(
        AuthMiddleware,
        config=auth_config,
        session_secret=session_secret,
        database=database,
    )
    app.state.session_secret = session_secret

    app.include_router(auth_router)

    # ── API routes ──
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

    try:
        from argus.dashboard.routes.baseline import router as baseline_router
        app.include_router(baseline_router, prefix="/api/baseline", tags=["baseline"])
    except ImportError:
        pass

    # ── WebSocket ──
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
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

    # ── Vue SPA static files ──
    # Serve built Vue app from web/dist/ (production)
    vue_dist = Path(__file__).parent.parent.parent.parent / "web" / "dist"
    if vue_dist.is_dir():
        # Serve static assets (js, css, images)
        app.mount("/assets", StaticFiles(directory=str(vue_dist / "assets")), name="vue-assets")

        # SPA fallback: all non-API routes serve index.html
        @app.get("/{full_path:path}")
        async def serve_spa(request: Request, full_path: str):
            # Don't intercept API or WebSocket routes
            if full_path.startswith("api/") or full_path.startswith("ws"):
                return
            # Serve actual files if they exist (favicon, etc.)
            file_path = vue_dist / full_path
            if file_path.is_file():
                return FileResponse(str(file_path))
            # SPA fallback — Vue Router handles client-side routing
            return FileResponse(str(vue_dist / "index.html"))

    return app
