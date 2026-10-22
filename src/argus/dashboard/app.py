"""FastAPI application — JSON API + Vue SPA static serving.

Backend provides /api/* JSON endpoints and /ws WebSocket.
Frontend is a Vue 3 + Ant Design SPA served from web/dist/.
go2rtc handles camera video streaming (WebRTC/MSE/HLS).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from argus.dashboard.api_response import (
    ErrorCode,
    JsonApiEnvelopeMiddleware,
    api_error,
    error_code_for_status,
    install_api_envelope_openapi,
)
from argus.dashboard.routes.alerts import router as alerts_router
from argus.dashboard.routes.audit import router as audit_router
from argus.dashboard.routes.backup import router as backup_router
from argus.dashboard.routes.cameras import router as cameras_router
from argus.dashboard.routes.config import router as config_router
from argus.dashboard.routes.me import router as me_router
from argus.dashboard.routes.sensors import router as sensors_router
from argus.dashboard.routes.system import router as system_router
from argus.dashboard.routes.tasks import router as tasks_router
from argus.dashboard.routes.reports import router as reports_router
from argus.dashboard.routes.regions import router as regions_router
from argus.dashboard.routes.users import router as users_router
from argus.dashboard.routes.zones import router as zones_router
from argus.dashboard.websocket import ConnectionManager, resolve_ws_identity
from argus.sensors.fusion import SensorFusion
from argus.streaming.go2rtc_manager import Go2RTCManager

if TYPE_CHECKING:
    from argus.camera.orchestrator import CameraOrchestrator
    from argus.capture.manager import CameraManager
    from argus.core.health import HealthMonitor
    from argus.streaming.stream_registry import StreamRegistry
    from argus.storage.database import Database

logger = structlog.get_logger()


def _resolve_session_secret(auth_config: object) -> str:
    """Return a restart-stable, purpose-separated session signing secret."""
    configured = os.environ.get("ARGUS_SESSION_SECRET", "").strip()
    api_token = str(getattr(auth_config, "api_token", "") or "")
    material = configured or api_token
    if not material:
        # Auth-disabled development instances do not need cross-restart
        # sessions. Keep an unpredictable per-process value in that case.
        return secrets.token_hex(32)
    return hashlib.sha256(
        b"argus.dashboard.session.v1\0" + material.encode("utf-8")
    ).hexdigest()


_HTTP_ERROR_MESSAGES: dict[int, str] = {
    400: "请求无效",
    401: "Authentication required",
    403: "权限不足",
    404: "资源不存在",
    405: "请求方法不支持",
    409: "资源状态冲突",
    422: "请求参数校验失败",
    429: "请求过于频繁",
    500: "内部错误",
    503: "服务不可用",
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_headers(
    request: Request,
    headers: dict[str, str] | None = None,
) -> dict[str, str]:
    merged = dict(headers or {})
    request_id = _request_id(request)
    if request_id:
        merged["X-Request-ID"] = request_id
    return merged


def _http_error_payload(
    request: Request,
    exc: StarletteHTTPException,
) -> tuple[int, str, object]:
    """Normalize FastAPI/Starlette HTTP errors without losing route details."""
    detail = exc.detail
    request_id = _request_id(request)

    if isinstance(detail, dict) and {"code", "msg", "data"}.issubset(detail):
        try:
            code = int(detail["code"])
        except (TypeError, ValueError):
            code = error_code_for_status(exc.status_code)
        msg = str(detail["msg"])
        original_data = detail["data"]
        if isinstance(original_data, dict):
            data = {**original_data, "request_id": request_id}
        else:
            data = {"request_id": request_id, "details": original_data}
        return code, msg, data

    msg = detail if isinstance(detail, str) and detail else _HTTP_ERROR_MESSAGES.get(
        exc.status_code,
        "请求失败",
    )
    data: dict[str, object] = {"request_id": request_id}
    if detail is not None and not isinstance(detail, str):
        data["details"] = jsonable_encoder(detail)
    return error_code_for_status(exc.status_code), str(msg), data


def _mount_vue_spa(app: FastAPI, vue_dist: Path) -> None:
    """Mount the built Vue SPA when a usable build is present.

    ``web/dist`` is an ignored build artifact and may exist in a partially
    generated state during local testing. Avoid letting a stale directory break
    backend startup.
    """
    dist_root = vue_dist.resolve()
    index_file = dist_root / "index.html"
    if not index_file.is_file():
        if vue_dist.exists():
            logger.warning("app.vue_dist_incomplete", path=str(vue_dist))
        return

    assets_dir = dist_root / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="vue-assets")
    else:
        logger.warning("app.vue_assets_missing", path=str(assets_dir))

    # SPA fallback: all non-API routes serve index.html
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        # Don't intercept API or WebSocket routes
        if full_path.startswith("api/") or full_path.startswith("ws"):
            raise HTTPException(status_code=404)
        # Serve actual files if they exist (favicon, etc.)
        file_path = (dist_root / full_path).resolve()
        try:
            file_path.relative_to(dist_root)
        except ValueError as exc:
            # Reject decoded traversal segments and symlinks that resolve
            # outside the build root instead of serving the sibling file or
            # disguising the attempt as a normal SPA route.
            raise HTTPException(status_code=404) from exc
        if file_path.is_file():
            return FileResponse(str(file_path))
        # SPA fallback — Vue Router handles client-side routing
        return FileResponse(str(index_file))


def create_app(
    database: Database | None = None,
    camera_manager: CameraManager | None = None,
    health_monitor: HealthMonitor | None = None,
    alerts_dir: str | None = None,
    config: object | None = None,
    config_path: str | None = None,
    task_manager: object | None = None,
    go2rtc_instance: Go2RTCManager | None = None,
    stream_registry: StreamRegistry | None = None,
    camera_orchestrator: CameraOrchestrator | None = None,
    sensor_fusion: SensorFusion | None = None,
    baseline_manager: object | None = None,
    model_trainer: object | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    from argus.anomaly.baseline_lifecycle import BaselineLifecycle
    from argus.config.schema import AuthConfig, DashboardConfig

    dashboard_config = getattr(config, "dashboard", None) or DashboardConfig()
    ws_manager = ConnectionManager(
        heartbeat_seconds=dashboard_config.websocket_heartbeat_seconds,
        max_connections=dashboard_config.websocket_max_connections,
    )
    stream_proxy_connections: set[str] = set()

    # go2rtc streaming proxy — reuse instance from __main__ if provided
    go2rtc: Go2RTCManager | None = go2rtc_instance
    if go2rtc is None and dashboard_config.go2rtc_enabled:
        go2rtc = Go2RTCManager(
            api_port=dashboard_config.go2rtc_api_port,
            rtsp_port=dashboard_config.go2rtc_rtsp_port,
            webrtc_port=dashboard_config.go2rtc_webrtc_port,
            webrtc_listen_host=dashboard_config.go2rtc_webrtc_listen_host,
            binary_path=dashboard_config.go2rtc_binary,
            config_dir=dashboard_config.go2rtc_config_dir,
            replace_stale_process=dashboard_config.go2rtc_replace_stale_process,
        )

    if stream_registry is None and go2rtc is not None:
        from argus.streaming.stream_registry import StreamRegistry

        stream_registry = StreamRegistry(go2rtc)
    if go2rtc is not None and stream_registry is not None:
        setattr(go2rtc, "_stream_registry", stream_registry)

    if (
        camera_orchestrator is None
        and camera_manager is not None
        and stream_registry is not None
    ):
        from argus.camera.orchestrator import CameraOrchestrator

        camera_orchestrator = CameraOrchestrator(camera_manager, stream_registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Expand the default asyncio thread pool for non-streaming API requests
        _default_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="api-default",
        )
        asyncio.get_running_loop().set_default_executor(_default_executor)
        ws_started = False
        release_worker = None
        try:
            await ws_manager.start()
            ws_started = True

            from argus.dashboard.routes.models import release_worker_for_app

            release_worker = release_worker_for_app(app.state)
            if release_worker is not None:
                # A separate thread owns durable release queue execution. A
                # second process using this database fails startup instead of
                # independently swapping the same camera's GPU runtime.
                release_worker.start()

            # Start go2rtc if not already running. When launched via __main__,
            # go2rtc is started earlier so USB cameras can be redirected before
            # the pipeline opens them — the guard below makes this a no-op in
            # that path. Dashboard-only mode (tests, standalone) still needs to
            # run the full start-and-register sequence.
            if go2rtc is not None and not go2rtc.running:
                cameras_cfg = list(getattr(config, "cameras", []) or [])
                try:
                    if stream_registry is not None:
                        await asyncio.to_thread(stream_registry.reconcile, cameras_cfg)
                    else:
                        from argus.streaming.go2rtc_manager import (
                            register_go2rtc_streams,
                        )

                        await asyncio.to_thread(
                            register_go2rtc_streams,
                            go2rtc,
                            cameras_cfg,
                        )
                except Exception as exc:
                    logger.warning(
                        "go2rtc.start_failed",
                        error=str(exc),
                        hint="Video streaming will fall back to MJPEG",
                    )
                    app.state.go2rtc = None

            yield
        finally:
            # Async context managers receive shutdown/cancellation at the
            # yield expression. Cleanup therefore has to live in finally.
            if release_worker is not None:
                await asyncio.get_running_loop().run_in_executor(
                    _default_executor,
                    release_worker.stop,
                )
            if go2rtc is not None:
                try:
                    go2rtc.close()
                except Exception:
                    logger.warning("go2rtc.close_failed", exc_info=True)
            if ws_started:
                try:
                    await ws_manager.stop()
                except Exception:
                    logger.warning("ws.manager_stop_failed", exc_info=True)

            # The MJPEG executor belongs to the process, not this app. Closing
            # one app must not break another app or a subsequent lifespan in
            # the same interpreter. concurrent.futures joins it at process
            # exit; only this lifespan's private API executor is closed here.
            _default_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="Argus - 视觉异常检测系统",
        version="0.2.0",
        lifespan=lifespan,
    )

    # ── Unified failure contract ──
    # Streaming/file successes are intentionally not wrapped. Their failures
    # still pass through these handlers and use the canonical JSON envelope.
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        request_id = _request_id(request)
        return api_error(
            ErrorCode.UNPROCESSABLE_ENTITY,
            _HTTP_ERROR_MESSAGES[422],
            status_code=422,
            data={
                "request_id": request_id,
                "errors": jsonable_encoder(exc.errors()),
            },
            headers=_error_headers(request),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code, msg, data = _http_error_payload(request, exc)
        return api_error(
            code,
            msg,
            status_code=exc.status_code,
            data=data,
            headers=_error_headers(request, exc.headers),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = _request_id(request)
        logger.error(
            "dashboard.unhandled_exception",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return api_error(
            ErrorCode.INTERNAL_ERROR,
            _HTTP_ERROR_MESSAGES[500],
            status_code=500,
            data={"request_id": request_id},
            headers=_error_headers(request),
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
    app.state.stream_registry = stream_registry
    app.state.camera_orchestrator = camera_orchestrator
    app.state.baseline_lifecycle = BaselineLifecycle(database) if database else None
    app.state.baseline_manager = baseline_manager
    app.state.model_trainer = model_trainer

    if (
        camera_manager is not None
        and go2rtc is not None
        and getattr(camera_manager, "_source_resolver", None) is None
        and hasattr(camera_manager, "set_source_resolver")
    ):
        def _dashboard_source_resolver(cam_config):
            current_go2rtc = getattr(app.state, "go2rtc", None)
            if current_go2rtc is None or not current_go2rtc.running:
                return cam_config

            orchestrator = getattr(app.state, "camera_orchestrator", None)
            if orchestrator is not None:
                return orchestrator.runtime_camera_config(cam_config)

            registry = getattr(app.state, "stream_registry", None)
            if registry is None:
                return cam_config
            resolution = registry.ensure_registered(cam_config, start_if_needed=False)
            if resolution is None:
                return registry.runtime_camera_config(cam_config)
            return registry.runtime_camera_config(cam_config, resolution)

        camera_manager.set_source_resolver(_dashboard_source_resolver)

    # External sensor fusion — generic (camera_id, zone_id) -> multiplier store.
    # When __main__ provides a shared instance, reuse it so the HTTP API and
    # the camera pipelines point at the same store. Otherwise build a local
    # one from config (the path used by tests and standalone dashboard mode).
    if sensor_fusion is not None:
        app.state.sensor_fusion = sensor_fusion
    else:
        from argus.config.schema import SensorFusionConfig

        fusion_cfg = getattr(config, "sensor_fusion", None) or SensorFusionConfig()
        from argus.sensors.fusion import SensorFusionHead

        app.state.sensor_fusion = SensorFusion(
            enabled=fusion_cfg.enabled,
            default_valid_for_s=fusion_cfg.default_valid_for_s,
            head=SensorFusionHead.from_config(fusion_cfg),
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
    if health_monitor is not None:
        if getattr(health_monitor, "_degradation_manager", None) is None:
            health_monitor._degradation_manager = app.state.degradation_manager
        if (
            getattr(health_monitor, "_feedback_manager", None) is None
            and app.state.feedback_manager is not None
        ):
            health_monitor._feedback_manager = app.state.feedback_manager

    # FR-033: Alert recording store for replay
    # The store is created in __main__.py and set on app.state after create_app().
    # Initialize to None here so replay routes degrade gracefully if not wired.
    if not hasattr(app.state, "recording_store"):
        app.state.recording_store = None

    # ── Security middleware ──
    from argus.dashboard.auth import (
        AuthMiddleware,
        RateLimitMiddleware,
        RequestIdMiddleware,
        SecurityHeadersMiddleware,
        auth_router,
    )

    app.add_middleware(RateLimitMiddleware, max_requests_per_minute=60)

    auth_config = getattr(config, "auth", None) or AuthConfig()
    session_secret = _resolve_session_secret(auth_config)
    app.add_middleware(
        AuthMiddleware,
        config=auth_config,
        session_secret=session_secret,
        database=database,
    )
    # Contract boundary for legacy JSON handlers.  This pure-ASGI middleware
    # buffers only application/json payloads; file/video/MJPEG/SSE responses
    # are passed through immediately.
    app.add_middleware(JsonApiEnvelopeMiddleware)
    # Security headers must wrap auth/rate-limit short-circuit responses too.
    # RequestId remains outermost so those responses can still include the
    # caller-visible correlation ID.
    app.add_middleware(SecurityHeadersMiddleware)
    # Added last so it is the outermost user middleware and covers auth/rate
    # limit responses as well as normal API and streaming responses.
    app.add_middleware(RequestIdMiddleware)
    app.state.session_secret = session_secret
    app.state.auth_enabled = auth_config.enabled
    app.state.session_max_age = auth_config.session_timeout_minutes * 60

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
    app.include_router(me_router, prefix="/api/me", tags=["me"])
    app.include_router(regions_router, prefix="/api/regions", tags=["regions"])
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

    from argus.dashboard.routes.scene_workflow import router as scene_workflow_router
    app.include_router(scene_workflow_router, prefix="/api/scene-workflow", tags=["scene-workflow"])

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

    # ── Authenticated same-origin video WebSocket proxy ──
    @app.websocket("/ws/stream/{camera_id}")
    async def stream_websocket_endpoint(
        websocket: WebSocket,
        camera_id: str,
        mode: str = "webrtc",
    ):
        from argus.dashboard.websocket import allowed_topics_for_role
        from argus.streaming.preview_gateway import PreviewGateway
        from argus.streaming.websocket_proxy import (
            is_same_origin_websocket,
            proxy_go2rtc_websocket,
        )

        if mode not in {"webrtc", "mse"}:
            await websocket.close(code=4400, reason="Unsupported stream mode")
            return

        token = websocket.query_params.get("token", "") or websocket.cookies.get(
            "argus_session",
            "",
        )
        auth_cfg = getattr(config, "auth", None) or AuthConfig()
        secret = getattr(app.state, "session_secret", "")
        identity = resolve_ws_identity(
            token,
            auth_cfg,
            session_secret=secret,
            database=database,
        )
        if identity is None:
            await websocket.close(code=4401, reason="Authentication required")
            return
        if (
            identity.get("auth_kind") == "session"
            and not is_same_origin_websocket(websocket)
        ):
            await websocket.close(code=4403, reason="Same-origin stream required")
            return

        role = str(identity.get("role", "viewer"))
        if "cameras" not in allowed_topics_for_role(role):
            await websocket.close(code=4403, reason="Camera read permission required")
            return

        gateway = PreviewGateway(app.state)
        camera = gateway.find_camera_config(camera_id)
        if camera is None:
            await websocket.close(code=4404, reason="Camera not found")
            return
        target = gateway.resolve_proxy_target(camera)
        if target is None:
            await websocket.close(code=1013, reason="Video stream unavailable")
            return

        stream_client_id = f"stream-{uuid.uuid4().hex}"
        if len(stream_proxy_connections) >= dashboard_config.websocket_max_connections:
            await websocket.close(code=1013, reason="Max stream connections reached")
            return
        # This check-and-reserve block contains no await and is therefore
        # atomic on the application's asyncio event loop.
        stream_proxy_connections.add(stream_client_id)
        close_code = 1000
        close_reason = ""
        proxy_task: asyncio.Task | None = None
        revalidation_task: asyncio.Task | None = None
        try:
            await websocket.accept()
            proxy_task = asyncio.create_task(
                proxy_go2rtc_websocket(websocket, target, mode)
            )
            if identity.get("auth_kind") == "session":

                async def wait_until_session_invalid() -> None:
                    while True:
                        await asyncio.sleep(30)
                        current = resolve_ws_identity(
                            token,
                            auth_cfg,
                            session_secret=secret,
                            database=database,
                        )
                        if current is None or "cameras" not in allowed_topics_for_role(
                            str(current.get("role", "viewer"))
                        ):
                            return

                revalidation_task = asyncio.create_task(
                    wait_until_session_invalid()
                )
                done, _ = await asyncio.wait(
                    {proxy_task, revalidation_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if proxy_task in done:
                    await proxy_task
                else:
                    close_code = 4401
                    close_reason = "Session expired or revoked"
            else:
                await proxy_task
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - proxy failures must close fail-safe
            close_code = 1013
            close_reason = "Video proxy unavailable"
            logger.warning(
                "stream_proxy.failed",
                camera_id=camera_id,
                mode=mode,
                error=str(exc),
            )
        finally:
            stream_proxy_connections.discard(stream_client_id)
            proxy_tasks = [
                task
                for task in (proxy_task, revalidation_task)
                if task is not None
            ]
            for task in proxy_tasks:
                if not task.done():
                    task.cancel()
            if proxy_tasks:
                await asyncio.gather(*proxy_tasks, return_exceptions=True)
            try:
                await websocket.close(code=close_code, reason=close_reason)
            except RuntimeError:
                pass

    # ── Dashboard event WebSocket ──
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        from argus.streaming.websocket_proxy import is_same_origin_websocket

        token = websocket.query_params.get("token", "") or websocket.cookies.get(
            "argus_session",
            "",
        )
        auth_cfg = getattr(config, "auth", None) or AuthConfig()
        secret = getattr(app.state, "session_secret", "")
        identity = resolve_ws_identity(
            token,
            auth_cfg,
            session_secret=secret,
            database=database,
        )
        if identity is None:
            await websocket.close(code=4401, reason="Authentication required")
            return
        if (
            identity.get("auth_kind") == "session"
            and not is_same_origin_websocket(websocket)
        ):
            await websocket.close(code=4403, reason="Same-origin socket required")
            return

        client_id = f"ws-{uuid.uuid4().hex[:8]}"
        try:
            client = await ws_manager.accept(
                websocket,
                client_id,
                role=str(identity.get("role", "viewer")),
                username=str(identity.get("username", "")),
                identity_resolver=(
                    lambda: resolve_ws_identity(
                        token,
                        auth_cfg,
                        session_secret=secret,
                        database=database,
                    )
                ) if identity.get("auth_kind") == "session" else None,
            )
            await ws_manager.handle_client(client)
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(client_id)

    # ── Vue SPA static files ──
    # Serve built Vue app from web/dist/ (production)
    vue_dist = Path(__file__).parent.parent.parent.parent / "web" / "dist"
    if vue_dist.is_dir():
        _mount_vue_spa(app, vue_dist)

    # Installed last so every API router participates in the generated
    # contract and media/file exceptions stay explicit.
    install_api_envelope_openapi(app)

    return app
