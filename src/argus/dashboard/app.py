"""FastAPI application — JSON API + Vue SPA static serving.

Backend provides /api/* JSON endpoints and /ws WebSocket.
Frontend is a Vue 3 + Ant Design SPA served from web/dist/.
go2rtc handles camera video streaming (WebRTC/MSE/HLS).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from argus.dashboard.routes.alerts import router as alerts_router
from argus.dashboard.routes.audit import router as audit_router
from argus.dashboard.routes.backup import router as backup_router
from argus.dashboard.routes.cameras import router as cameras_router
from argus.dashboard.routes.config import router as config_router
from argus.dashboard.routes.sensors import router as sensors_router
from argus.dashboard.routes.system import router as system_router
from argus.dashboard.routes.tasks import router as tasks_router
from argus.dashboard.routes.reports import router as reports_router
from argus.dashboard.routes.users import router as users_router
from argus.dashboard.routes.zones import router as zones_router
from argus.dashboard.websocket import ConnectionManager, verify_ws_token
from argus.sensors.fusion import SensorFusion
from argus.streaming.go2rtc_manager import Go2RTCManager

if TYPE_CHECKING:
    from argus.capture.manager import CameraManager
    from argus.core.health import HealthMonitor
    from argus.dashboard.tasks import TaskManager
    from argus.storage.database import Database

logger = structlog.get_logger()


def create_app(
    database: Database | None = None,
    camera_manager: CameraManager | None = None,
    health_monitor: HealthMonitor | None = None,
    alerts_dir: str | None = None,
    config: object | None = None,
    config_path: str | None = None,
    task_manager: object | None = None,
    go2rtc_instance: Go2RTCManager | None = None,
    sensor_fusion: SensorFusion | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    from argus.anomaly.baseline_lifecycle import BaselineLifecycle
    from argus.config.schema import AuthConfig, DashboardConfig

    dashboard_config = getattr(config, "dashboard", None) or DashboardConfig()
    ws_manager = ConnectionManager(
        heartbeat_seconds=dashboard_config.websocket_heartbeat_seconds,
        max_connections=dashboard_config.websocket_max_connections,
    )

    # go2rtc streaming proxy — reuse instance from __main__ if provided
    go2rtc: Go2RTCManager | None = go2rtc_instance
    if go2rtc is None and dashboard_config.go2rtc_enabled:
        go2rtc = Go2RTCManager(
            api_port=dashboard_config.go2rtc_api_port,
            rtsp_port=dashboard_config.go2rtc_rtsp_port,
            binary_path=dashboard_config.go2rtc_binary,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Expand the default asyncio thread pool for non-streaming API requests
        _default_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="api-default",
        )
        asyncio.get_running_loop().set_default_executor(_default_executor)

        await ws_manager.start()

        # Start go2rtc if not already running. When launched via __main__,
        # go2rtc is started earlier so USB cameras can be redirected before
        # the pipeline opens them — the guard below makes this a no-op in
        # that path. Dashboard-only mode (tests, standalone) still needs to
        # run the full start-and-register sequence.
        if go2rtc is not None and not go2rtc.running:
            from argus.streaming.go2rtc_manager import start_and_register_cameras
            cameras_cfg = list(getattr(config, "cameras", []) or [])
            try:
                await asyncio.to_thread(start_and_register_cameras, go2rtc, cameras_cfg)
            except Exception as exc:
                logger.warning(
                    "go2rtc.start_failed",
                    error=str(exc),
                    hint="Video streaming will fall back to MJPEG",
                )
                app.state.go2rtc = None

        yield

        if go2rtc is not None:
            go2rtc.close()
        await ws_manager.stop()

        # Shut down thread pools AFTER uvicorn stops accepting requests.
        # Use wait=True so in-flight tasks finish before the executor is
        # torn down.  cancel_futures=True (Python 3.9+) cancels queued-but-
        # not-started futures to avoid blocking on stale work.
        from argus.dashboard.routes.cameras import _STREAM_EXECUTOR
        _STREAM_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        _default_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="Argus - 核电站异物检测系统",
        version="0.2.0",
        lifespan=lifespan,
    )

    # ── App state (dependency injection) ──
    app.state.db = database
    app.state.database = database
    app.state.camera_manager = camera_manager
    app.state.health_monitor = health_monitor
    app.state.alerts_dir = Path(alerts_dir) if alerts_dir else Path("data/alerts")
    app.state.config = config
    app.state.config_path = config_path
    app.state.task_manager = task_manager
    app.state.ws_manager = ws_manager
    app.state.go2rtc = go2rtc
    app.state.baseline_lifecycle = BaselineLifecycle(database) if database else None

    # External sensor fusion — generic (camera_id, zone_id) -> multiplier store.
    # When __main__ provides a shared instance, reuse it so the HTTP API and
    # the camera pipelines point at the same store. Otherwise build a local
    # one from config (the path used by tests and standalone dashboard mode).
    if sensor_fusion is not None:
        app.state.sensor_fusion = sensor_fusion
    else:
        from argus.config.schema import SensorFusionConfig

        fusion_cfg = getattr(config, "sensor_fusion", None) or SensorFusionConfig()
        app.state.sensor_fusion = SensorFusion(
            enabled=fusion_cfg.enabled,
            default_valid_for_s=fusion_cfg.default_valid_for_s,
        )

    if task_manager is not None and getattr(task_manager, "_on_change", None) is None:
        task_manager._on_change = ws_manager.broadcast

    if health_monitor is not None and getattr(health_monitor, "_on_change", None) is None:
        health_monitor._on_change = ws_manager.broadcast

    # Feedback manager for the feedback queue (Section 6)
    if database:
        from argus.alerts.feedback import FeedbackManager
        from argus.config.schema import FeedbackConfig

        feedback_cfg = getattr(config, "feedback", None) or FeedbackConfig()
        baselines_dir = getattr(config, "storage", None)
        baselines_path = baselines_dir.baselines_dir if baselines_dir else "data/baselines"
        app.state.feedback_manager = FeedbackManager(
            database=database,
            baselines_dir=str(baselines_path),
            alerts_dir=str(app.state.alerts_dir),
            config=feedback_cfg,
        )
    else:
        app.state.feedback_manager = None

    # UX v2 §5: Global degradation manager
    from argus.core.degradation import GlobalDegradationManager

    app.state.degradation_manager = GlobalDegradationManager(ws_manager=ws_manager)

    # FR-033: Alert recording store for replay
    # The store is created in __main__.py and set on app.state after create_app().
    # Initialize to None here so replay routes degrade gracefully if not wired.
    if not hasattr(app.state, "recording_store"):
        app.state.recording_store = None

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
    app.include_router(system_router, prefix="/api/system", tags=["system"])
    app.include_router(tasks_router, prefix="/api/tasks", tags=["tasks"])
    app.include_router(audit_router, prefix="/api/audit", tags=["audit"])
    app.include_router(backup_router, prefix="/api/backup", tags=["backup"])
    app.include_router(users_router, prefix="/api/users", tags=["users"])
    app.include_router(reports_router, prefix="/api/reports", tags=["reports"])
    app.include_router(sensors_router, prefix="/api/sensors", tags=["sensors"])

    from argus.dashboard.routes.models import router as models_router
    app.include_router(models_router, prefix="/api/models", tags=["models"])

    from argus.dashboard.routes.replay import router as replay_router
    app.include_router(replay_router, prefix="/api/replay", tags=["replay"])

    from argus.dashboard.routes.degradation import router as degradation_router
    app.include_router(degradation_router, prefix="/api/degradation", tags=["degradation"])

    from argus.dashboard.routes.training_jobs import router as training_jobs_router
    app.include_router(training_jobs_router, prefix="/api/training-jobs", tags=["training-jobs"])

    from argus.dashboard.routes.streaming import router as streaming_router
    app.include_router(streaming_router, prefix="/api/streaming", tags=["streaming"])

    from argus.dashboard.routes.labeling import router as labeling_router
    app.include_router(labeling_router, prefix="/api/labeling", tags=["labeling"])

    from argus.dashboard.routes.recordings import router as recordings_router
    app.include_router(recordings_router, prefix="/api/recordings", tags=["recordings"])

    from argus.dashboard.routes.calibration import router as calibration_router
    app.include_router(calibration_router, prefix="/api/calibration", tags=["calibration"])

    from argus.dashboard.routes.physics import router as physics_router
    app.include_router(physics_router, prefix="/api/physics", tags=["physics"])

    try:
        from argus.dashboard.routes.baseline import router as baseline_router
        app.include_router(baseline_router, prefix="/api/baseline", tags=["baseline"])
    except ImportError:
        logger.debug("app.baseline_router_import_failed", exc_info=True)

    # ── WebSocket ──
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        token = websocket.query_params.get("token", "")
        auth_cfg = getattr(config, "auth", None) or AuthConfig()
        secret = getattr(app.state, "session_secret", "")
        if not verify_ws_token(token, auth_cfg, session_secret=secret):
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
