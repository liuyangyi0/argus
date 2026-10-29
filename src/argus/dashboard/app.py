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
    """Use independent server material for personal-account session signatures.

    Deriving this key from the shared API credential would let automation
    impersonate any account. Without a separately configured secret, sessions
    intentionally expire when this application instance restarts.
    """
    configured = os.environ.get("ARGUS_SESSION_SECRET", "").strip()
    api_token = str(getattr(auth_config, "api_token", "") or "")
    if not configured:
        return secrets.token_hex(32)
    if api_token.strip() and secrets.compare_digest(
        configured.encode("utf-8"), api_token.strip().encode("utf-8"),
    ):
        raise ValueError("ARGUS_SESSION_SECRET must be independent from the shared API token")
    return hashlib.sha256(
        b"argus.dashboard.session.v1\0" + configured.encode("utf-8")
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
    from argus.capture.manager import CameraManager
    compute_manager = camera_manager if isinstance(camera_manager, CameraManager) else None
    if compute_manager is not None:
        compute_manager.enable_compute_ownership(retain_until_exit=task_manager is not None or model_trainer is not None)

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

            compute_available = compute_manager is None or compute_manager.claim_compute_ownership()
            release_worker = (None if getattr(app.state, "task_training_maintenance_requested", False) or not compute_available
                              else release_worker_for_app(app.state))
            if release_worker is not None:
                # A separate thread owns durable release queue execution. A
                # second process using this database fails startup instead of
                # independently swapping the same camera's GPU runtime.
                if compute_manager is None or compute_manager.authorize_legacy_compute():
                    release_worker.start()
                else:
                    release_worker = None

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

            task_events = getattr(app.state, "task_events", None)
            if task_events is not None:
                try:
                    if not await asyncio.to_thread(task_events.start):
                        logger.warning("app.task_event_worker_unavailable")
                except Exception:
                    # A validation-only evidence worker does not own the
                    # camera, preview, legacy pipeline or Dashboard lifetime.
                    logger.warning("app.task_event_worker_start_failed")
            task_coordinator = getattr(app.state, "task_deployment_coordinator", None)
            if task_coordinator is not None:
                available = await asyncio.to_thread(task_coordinator.start)
                if not available:
                    logger.warning("app.task_deployment_owner_unavailable")

            training_runtime = getattr(app.state, "task_training_runtime", None)
            if training_runtime is not None:
                try:
                    if training_runtime.enabled and not await asyncio.to_thread(training_runtime.start):
                        logger.warning("app.task_training_owner_unavailable")
                except Exception:
                    logger.error("app.task_training_start_failed")

            yield
        finally:
            # Async context managers receive shutdown/cancellation at the
            # yield expression. Cleanup therefore has to live in finally.
            training_runtime = getattr(app.state, "task_training_runtime", None)
            training_cleanup_complete = training_runtime is None
            if training_runtime is not None:
                try:
                    training_cleanup_complete = await asyncio.to_thread(training_runtime.close, timeout=5) is True
                    if not training_cleanup_complete:
                        logger.error("app.task_training_cleanup_pending")
                except Exception:
                    logger.error("app.task_training_cleanup_failed")
            task_coordinator = getattr(app.state, "task_deployment_coordinator", None)
            if task_coordinator is not None:
                try:
                    await asyncio.to_thread(task_coordinator.close, wait=False)
                except Exception:
                    logger.error("app.task_deployment_close_failed")
            for name in ("task_execution",):
                service = getattr(app.state, name, None)
                close = getattr(service, "close", None)
                if callable(close):
                    try:
                        cleaned = await asyncio.get_running_loop().run_in_executor(_default_executor, close)
                        if cleaned is False:
                            logger.error("app.task_service_cleanup_pending", service=name)
                    except Exception:
                        logger.warning("app.task_service_cleanup_failed", service=name)
            if task_coordinator is not None:
                try:
                    if not await asyncio.to_thread(task_coordinator.join, timeout=5.0):
                        logger.error("app.task_deployment_owner_cleanup_pending")
                except Exception:
                    logger.error("app.task_deployment_owner_release_failed")
            workflows = getattr(app.state, "task_workflows", None)
            task_events = getattr(app.state, "task_events", None)
            if task_events is not None:
                try:
                    if not await asyncio.to_thread(task_events.close):
                        logger.error("app.task_service_cleanup_pending", service="task_events")
                except Exception:
                    logger.warning("app.task_service_cleanup_failed", service="task_events")
            if workflows is not None:
                try:
                    if await asyncio.to_thread(workflows.close) is False:
                        logger.error("app.task_service_cleanup_pending", service="task_workflows")
                except Exception:
                    logger.warning("app.task_service_cleanup_failed", service="task_workflows")
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
            if (compute_manager is not None and training_cleanup_complete
                    and not training_compute_exclusion_required(None, database)):
                if not compute_manager.close_compute_ownership():
                    logger.info("app.compute_ownership_retained")

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

    # Additive P0-P2 task identity bootstrap. Existing release services remain
    # authoritative; this never promotes an artifact or claims runtime applied.
    from argus.storage.analysis_tasks import AnalysisTaskService
    from argus.storage.database import Database
    from argus.config.schema import CameraConfig, StorageConfig

    app.state.analysis_task_service = None
    app.state.analysis_task_bootstrap_errors = set()
    task_storage_config = getattr(config, "storage", None)
    if not isinstance(task_storage_config, StorageConfig):
        task_storage_config = StorageConfig()
    if isinstance(database, Database):
        analysis_tasks = AnalysisTaskService(database, models_root=task_storage_config.models_dir)
        for camera in getattr(config, "cameras", []):
            if isinstance(camera, CameraConfig):
                try:
                    analysis_tasks.bootstrap_legacy(
                        camera.camera_id,
                        camera.model_dump(mode="json", exclude={"source"}),
                    )
                except Exception as exc:
                    # A damaged compatibility identity may block its task, not
                    # the dashboard needed to inspect/repair the installation.
                    # Do not log exception text: it can include source secrets.
                    app.state.analysis_task_bootstrap_errors.add(camera.camera_id)
                    logger.warning(
                        "app.analysis_task_bootstrap_failed",
                        camera_id=camera.camera_id, error_type=type(exc).__name__,
                    )
                    continue
                try:
                    analysis_tasks.sync_legacy_model_projection(camera.camera_id)
                except Exception as exc:
                    # This is an observation, never the release authority. A
                    # registry CAS conflict must not stop unrelated cameras or
                    # make FastAPI unavailable. Reads retain not_synced/stale.
                    logger.warning(
                        "app.analysis_task_projection_failed",
                        camera_id=camera.camera_id, error_type=type(exc).__name__,
                    )
        app.state.analysis_task_service = analysis_tasks

    app.state.task_execution = None
    app.state.task_workflows = None
    app.state.task_automation = None
    app.state.task_deployments = None
    app.state.task_deployment_coordinator = None
    app.state.task_events = None
    app.state.task_releases = None
    app.state.task_training = None
    app.state.task_training_runtime = None
    app.state.task_training_admission = None
    from argus.config.schema import TaskTrainingConfig
    training_config = getattr(config, "task_training", None)
    if not isinstance(training_config, TaskTrainingConfig):
        training_config = TaskTrainingConfig()
    training_config = training_config.model_copy(deep=True)
    app.state.task_training_config = training_config
    from argus.runtime.task_training_resources import training_compute_exclusion_required
    app.state.task_training_maintenance_requested = training_compute_exclusion_required(config, database)
    if compute_manager is not None and compute_manager.is_training_maintenance():
        app.state.task_training_maintenance_requested = True
    if app.state.task_training_maintenance_requested:
        enter = getattr(camera_manager, "enter_training_maintenance", None)
        if callable(enter) and enter() is not True:
            logger.warning("app.task_training_maintenance_waiting_for_cleanup")
    if app.state.analysis_task_service is not None:
        from argus.capture.task_execution_manager import TaskExecutionManager
        from argus.storage.task_workflows import TaskWorkflowBlocked, TaskWorkflowService
        from argus.storage.task_deployments import TaskDeploymentService

        app.state.task_deployments = TaskDeploymentService(database, app.state.analysis_task_service)

        def _task_candidate_metadata(*, task_id, candidate, training_manifest):
            from argus.core.analysis_task_contracts import TaskModelDescriptor, template_spec

            service = app.state.analysis_task_service
            task = service.get_task(task_id)
            binding = training_manifest["binding"]
            if (task is None or task["authority"] != "managed" or task.get("archived")
                    or binding["task_id"] != task_id
                    or binding["camera_id"] != task["camera_id"]
                    or binding["task_revision"] != task["current_revision"]
                    or binding["task_config_sha256"] != task["config_sha256"]):
                raise ValueError("candidate task binding is stale")
            if not isinstance(candidate, dict) or set(candidate) not in (
                {"relative_path", "descriptor"}, {"relative_path", "descriptor", "artifact_sha256"},
            ):
                raise ValueError("candidate requires a model path, descriptor and optional artifact SHA-256")
            descriptor = TaskModelDescriptor.model_validate(candidate["descriptor"]).model_dump(mode="json")
            expected = set(template_spec(task["template_id"])["model_roles"].values())
            if descriptor["adapter_id"] not in expected:
                raise ValueError("candidate adapter does not match the training task")
            if descriptor["labels"] != training_manifest["dataset"]["label_schema"]["labels"]:
                raise ValueError("candidate label schema does not match the training dataset")
            return descriptor

        def _validate_task_candidate(*, task_id, candidate, training_manifest):
            service = app.state.analysis_task_service
            descriptor = _task_candidate_metadata(task_id=task_id, candidate=candidate,
                                                   training_manifest=training_manifest)
            _path, actual_sha = service._trusted_artifact(candidate["relative_path"])
            expected_sha = candidate.get("artifact_sha256")
            if expected_sha is not None:
                if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                        or any(char not in "0123456789abcdef" for char in expected_sha)):
                    raise ValueError("candidate needs an exact artifact SHA-256")
                if actual_sha != expected_sha:
                    raise ValueError("candidate content does not match the training result")
            return {"relative_path": candidate["relative_path"], "descriptor": descriptor,
                    "artifact_sha256": actual_sha}

        def _register_task_candidate(*, task_id, candidate, training_manifest, actor, idempotency_key):
            service = app.state.analysis_task_service
            # Metadata/approval identity is checked even for reconciliation,
            # but a committed registry receipt must not depend on its source
            # file remaining present. Runtime loading rechecks artifact bytes.
            descriptor = _task_candidate_metadata(task_id=task_id, candidate=candidate,
                                                   training_manifest=training_manifest)
            arguments = ({"expected_artifact_sha256": candidate["artifact_sha256"]}
                         if "artifact_sha256" in candidate else {})
            model = service.get_task_model_registration_receipt(
                candidate["relative_path"], descriptor,
                scope_camera_id=training_manifest["binding"]["camera_id"],
                requested_by=actor, idempotency_key=idempotency_key, **arguments,
            )
            if model is None:
                if "artifact_sha256" not in candidate:
                    # An old in-flight intent without a frozen artifact SHA
                    # cannot acquire identity from whichever file is here now.
                    raise TaskWorkflowBlocked("LEGACY_CANDIDATE_IDENTITY_UNRECORDED")
                normalized = _validate_task_candidate(task_id=task_id, candidate=candidate,
                                                       training_manifest=training_manifest)
                model = service.register_task_model(
                    normalized["relative_path"], normalized["descriptor"],
                    scope_camera_id=training_manifest["binding"]["camera_id"],
                    requested_by=actor, idempotency_key=idempotency_key,
                    expected_artifact_sha256=normalized["artifact_sha256"],
                )
            if "artifact_sha256" in candidate and model["artifact_sha256"] != candidate["artifact_sha256"]:
                raise ValueError("candidate registration identity does not match the training result")
            _task_candidate_metadata(task_id=task_id, candidate=candidate, training_manifest=training_manifest)
            return {"artifact_id": model["model_id"]}

        app.state.task_workflows = TaskWorkflowService(
            database, task_storage_config.baselines_dir.parent / "task_datasets",
            candidate_registrar=_register_task_candidate,
            candidate_validator=_validate_task_candidate,
        )
        from argus.storage.task_release import TaskReleaseService
        from argus.storage.task_events import TaskEventService
        from argus.core.task_training import TaskTrainingService

        app.state.task_releases = TaskReleaseService(
            database, app.state.analysis_task_service,
            evidence_root=task_storage_config.baselines_dir.parent / "task_release_evidence",
        )
        app.state.task_training = TaskTrainingService(
            app.state.task_workflows, task_storage_config.models_dir / "task_training",
            task_storage_config.models_dir, configuration=training_config,
        )
        from argus.runtime.task_training_resources import TaskTrainingResourceAdmission
        from argus.runtime.task_training_runtime import TaskTrainingRuntime
        from argus.runtime.task_training import SpawnTrainingWorker
        from functools import partial
        app.state.task_training_admission = TaskTrainingResourceAdmission(
            training_config, camera_manager, database=database,
        )
        app.state.task_training_runtime = TaskTrainingRuntime(
            app.state.task_training, resource_admission=app.state.task_training_admission,
            interval_seconds=training_config.poll_interval_seconds, enabled=training_config.enabled,
            worker_factory=partial(SpawnTrainingWorker,
                worker_ownership_path=compute_manager.get_training_worker_lock_path()) if compute_manager is not None else SpawnTrainingWorker,
        )
        # Construction and GET are inert. Only explicit enabled lifespan
        # startup schedules approved jobs behind exclusive resource admission.
        app.state.task_events = TaskEventService(
            database, app.state.analysis_task_service,
            evidence_root=task_storage_config.baselines_dir.parent / "task_events",
            release_service=app.state.task_releases,
            current_runtime=lambda task_id: (
                app.state.task_execution.event_context(task_id)
                if app.state.task_execution is not None else None
            ),
            broadcaster=ws_manager.broadcast,
        )
        app.state.task_workflows.set_event_provider(app.state.task_events)
        if camera_manager is not None:
            app.state.task_execution = TaskExecutionManager(
                app.state.analysis_task_service, camera_manager,
                models_root=task_storage_config.models_dir,
            )
            app.state.task_execution.set_event_sink(app.state.task_events)
            source_canceller = getattr(camera_manager, "set_source_task_canceller", None)
            if callable(source_canceller):
                source_canceller(app.state.task_execution.stop_source)
            from argus.runtime.task_automation import TaskAutomationService

            app.state.task_automation = TaskAutomationService(
                app.state.analysis_task_service, app.state.task_execution,
                app.state.task_workflows, camera_manager,
            )
            from argus.runtime.task_deployments import TaskDeploymentCoordinator

            app.state.task_deployment_coordinator = TaskDeploymentCoordinator(
                app.state.task_deployments, app.state.task_execution,
                on_observed=app.state.task_automation.reconcile_deployment,
            )

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

    from argus.dashboard.training_maintenance import TrainingMaintenanceMiddleware
    app.add_middleware(TrainingMaintenanceMiddleware, enabled=app.state.task_training_maintenance_requested,
                       compute_admission=compute_manager.authorize_legacy_compute if compute_manager is not None else None,
                       maintenance_check=compute_manager.is_training_maintenance if compute_manager is not None else None,
                       require_compute_authority=True)
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
    from argus.dashboard.routes.analysis_tasks import (
        router as analysis_tasks_router,
        templates_router as analysis_task_templates_router,
    )

    app.include_router(analysis_tasks_router, prefix="/api/analysis-tasks", tags=["analysis-tasks"])
    app.include_router(analysis_task_templates_router, prefix="/api/analysis-task-templates", tags=["analysis-tasks"])
    from argus.dashboard.routes.task_platform import (
        models_router as task_models_router, router as task_platform_router,
    )

    app.include_router(task_platform_router, prefix="/api/analysis-tasks", tags=["analysis-tasks"])
    app.include_router(task_models_router, prefix="/api/analysis-task-models", tags=["analysis-tasks"])
    from argus.dashboard.routes.task_workflows import router as task_workflows_router

    app.include_router(task_workflows_router, prefix="/api/analysis-tasks", tags=["analysis-tasks"])
    from argus.dashboard.routes.task_deployments import router as task_deployments_router

    app.include_router(task_deployments_router, prefix="/api/analysis-tasks", tags=["analysis-tasks"])
    from argus.dashboard.routes.task_lifecycle import router as task_lifecycle_router

    app.include_router(task_lifecycle_router, prefix="/api/analysis-tasks", tags=["analysis-tasks"])
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
    configured_dist = getattr(dashboard_config, "frontend_dist_dir", None)
    vue_dist = (
        Path(configured_dist)
        if configured_dist is not None
        else Path(__file__).parent.parent.parent.parent / "web" / "dist"
    )
    if configured_dist is not None and not (vue_dist / "index.html").is_file():
        raise ValueError("Configured dashboard.frontend_dist_dir has no index.html")
    if vue_dist.is_dir():
        _mount_vue_spa(app, vue_dist)

    # Installed last so every API router participates in the generated
    # contract and media/file exceptions stay explicit.
    install_api_envelope_openapi(app)

    return app
