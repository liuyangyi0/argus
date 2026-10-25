"""Model management API routes (C4)."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from argus.dashboard.api_contracts import (
    BackboneUpgradeRequest,
    ModelAbLiveCompareRequest,
    ModelBatchInferenceRequest,
    ModelPromoteRequest,
    ModelReexportRequest,
    ModelRetireRequest,
    ModelRollbackRequest,
    ReleaseReadinessResponse,
    RollbackReadinessResponse,
)
from argus.dashboard.api_response import (
    ApiEnvelope,
    ErrorCode,
    api_error,
    api_conflict,
    api_forbidden,
    api_internal_error,
    api_not_found,
    api_success,
    api_typed_success,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.auth import current_username, require_human_permission, require_permission
from argus.dashboard.model_runtime import get_registry
from argus.preprocessing.scene_profile import runtime_artifact_sha256
from argus.storage.model_registry import (
    ModelDeletionConflictError,
    ModelRegistry,
    validate_persisted_release_evidence,
)
from argus.storage.model_release_service import (
    ModelReleaseService,
    RuntimeReleasePreparation,
)
from argus.storage.models import (
    ModelRecord,
    ModelStage,
    ReleaseOperationStatus,
    ShadowInferenceLog,
)
from argus.storage.release_pipeline import (
    ReleasePipeline,
    StageTransitionError,
)
from argus.storage.release_execution import ModelReleaseWorker

logger = structlog.get_logger()

router = APIRouter()

MAX_BATCH_SIZE = 100

# Dedicated thread pool for CPU-intensive model inference (not asyncio default pool)
_AB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ab-compare")


def _deny_without_permission(request: Request, permission: str) -> JSONResponse | None:
    """Return a 403 response when the current user lacks a permission."""
    if require_permission(request, permission):
        return None
    return api_forbidden("权限不足")


def _get_registry(request: Request) -> ModelRegistry | None:
    """Get ModelRegistry from app state, or None if unavailable."""
    return get_registry(request)


def _get_session_factory(request: Request):
    """Get SQLAlchemy session factory from app state, or None."""
    db = getattr(request.app.state, "database", None) or getattr(request.app.state, "db", None)
    if db is None:
        return None
    return getattr(db, "get_session", None)


def _get_release_anomaly_config(config: object | None) -> object | None:
    if config is None:
        return None
    anomaly_cfg = getattr(config, "anomaly", None)
    if anomaly_cfg is not None:
        return anomaly_cfg
    cameras = getattr(config, "cameras", None) or []
    if not cameras:
        return None
    return getattr(cameras[0], "anomaly", None)


def _get_release_pipeline(request: Request) -> ReleasePipeline | None:
    """Get or create ReleasePipeline singleton from app state."""
    cached = getattr(request.app.state, "_release_pipeline", None)
    if cached is not None:
        return cached

    db = getattr(request.app.state, "database", None) or getattr(request.app.state, "db", None)
    if db is None:
        return None
    session_factory = getattr(db, "get_session", None)
    if session_factory is None:
        return None
    config = getattr(request.app.state, "config", None)
    kwargs = {}
    anomaly_cfg = _get_release_anomaly_config(config)
    if anomaly_cfg:
        kwargs["min_shadow_days"] = getattr(anomaly_cfg, "min_shadow_days", 3)
        kwargs["min_canary_days"] = getattr(anomaly_cfg, "min_canary_days", 7)
        kwargs["min_shadow_samples"] = getattr(
            anomaly_cfg,
            "min_shadow_samples",
            0,
        )
        kwargs["max_shadow_alert_rate_delta"] = getattr(
            anomaly_cfg,
            "max_shadow_alert_rate_delta",
            1.0,
        )
        kwargs["max_shadow_score_divergence"] = getattr(
            anomaly_cfg,
            "max_shadow_score_divergence",
            1.0,
        )
        kwargs["max_shadow_latency_ms"] = getattr(
            anomaly_cfg,
            "max_shadow_latency_ms",
            0.0,
        )
    # Publishing is owned by ModelReleaseService and happens only after the
    # runtime identity has been confirmed.  Broadcasting here would expose an
    # intermediate database state that may still be compensated.
    pipeline = ReleasePipeline(session_factory=session_factory, **kwargs)
    request.app.state._release_pipeline = pipeline
    return pipeline


def _get_release_service(request: Request) -> ModelReleaseService | None:
    """Return the application's release service, independent of request lifetime."""

    return release_service_for_app(request.app.state)


def release_service_for_app(state) -> ModelReleaseService | None:
    """Create the one service shared by HTTP submission and the owned worker."""

    cached = getattr(state, "model_release_service", None)
    if cached is not None:
        return cached
    # The pipeline factory needs only app.state. Do not retain the HTTP
    # Request in a worker callback after its response has been returned.
    from types import SimpleNamespace

    context = SimpleNamespace(app=SimpleNamespace(state=state))
    pipeline = _get_release_pipeline(context)
    session_factory = _get_session_factory(context)
    if pipeline is None or session_factory is None:
        return None

    def _runtime_prepare(plan) -> RuntimeReleasePreparation:
        camera_manager = getattr(state, "camera_manager", None)
        if getattr(plan, "rollback_context", None) is not None:
            runtime = _safe_runtime_state(camera_manager, plan.camera_id)
            context = plan.rollback_context
            if not runtime or (
                runtime.get("primary_model_version") != context["source_model_version_id"]
                or runtime.get("release_revision") != context["source_release_revision"]
            ):
                return RuntimeReleasePreparation(
                    success=False, error="Rollback source does not match the running model revision",
                )
        if (
            camera_manager is None
            or inspect.getattr_static(
                camera_manager,
                "prepare_model_release",
                None,
            ) is None
        ):
            return RuntimeReleasePreparation(
                success=False,
                error="camera runtime does not support two-phase model release",
            )
        prepare = getattr(camera_manager, "prepare_model_release", None)
        if not callable(prepare):
            return RuntimeReleasePreparation(
                success=False,
                error="camera runtime does not support two-phase model release",
            )
        return prepare(plan)

    ws_manager = getattr(state, "ws_manager", None)
    publisher = (
        ws_manager.broadcast
        if ws_manager is not None and hasattr(ws_manager, "broadcast")
        else None
    )
    service = ModelReleaseService(
        session_factory,
        pipeline,
        runtime_prepare=_runtime_prepare,
        event_publisher=publisher,
    )
    state.model_release_service = service
    return service


def release_worker_for_app(state) -> ModelReleaseWorker | None:
    worker = getattr(state, "model_release_worker", None)
    if worker is not None:
        return worker
    service = release_service_for_app(state)
    if service is None:
        return None
    worker = ModelReleaseWorker(service)
    state.model_release_worker = worker
    return worker


def _expected_runtime_record(request: Request, model: ModelRecord) -> ModelRecord | None:
    """Return the release-stage record this model should occupy at runtime."""
    if model.stage == ModelStage.SHADOW.value:
        getter_name = "get_shadow_model_for_camera"
    elif model.stage in (ModelStage.CANARY.value, ModelStage.PRODUCTION.value):
        getter_name = "get_model_for_camera"
    else:
        return None

    session_factory = _get_session_factory(request)
    if session_factory is None:
        return None
    try:
        from argus.anomaly.model_router import ModelRouter

        router = ModelRouter(session_factory=session_factory)
        getter = getattr(router, getter_name)
        return getter(model.camera_id)
    except Exception as exc:  # pragma: no cover - defensive UI metadata
        logger.warning(
            "models.runtime_expected_lookup_failed",
            camera_id=model.camera_id,
            model_version_id=model.model_version_id,
            error=str(exc),
        )
        return None


def _model_runtime_state(request: Request, model: ModelRecord) -> str | None:
    """Map registry state to the table's applied/waiting/failed UI contract."""
    expected = _expected_runtime_record(request, model)
    if expected is None or expected.model_version_id != model.model_version_id:
        return None

    manager = getattr(request.app.state, "camera_manager", None)
    get_runtime_state = (
        getattr(manager, "get_runtime_model_state", None)
        if manager is not None
        else None
    )
    if not callable(get_runtime_state):
        return "waiting"
    try:
        runtime_state = get_runtime_state(model.camera_id)
    except Exception:
        logger.warning(
            "models.runtime_state_snapshot_failed",
            camera_id=model.camera_id,
            exc_info=True,
        )
        return "failed"
    if not isinstance(runtime_state, dict):
        return "waiting"

    if model.stage == ModelStage.SHADOW.value:
        current_shadow = runtime_state.get("shadow_model_version")
        return "applied" if current_shadow == model.model_version_id else "failed"

    current_primary = runtime_state.get("primary_model_version")
    return "applied" if current_primary == model.model_version_id else "failed"


@router.get("/status")
def models_status(request: Request) -> JSONResponse:
    """Aggregate health status of every detector across all cameras.

    Returns a flat list of :class:`~argus.core.model_status.ModelStatus`
    records (anomaly + YOLO per camera). The frontend polls this every few
    seconds so operators can see "cam1 anomaly is failing" without having
    to grep server logs.
    """
    manager = getattr(request.app.state, "camera_manager", None)
    if manager is None:
        return api_success({"models": []})

    get_statuses = getattr(manager, "get_model_statuses_snapshot", None)
    if not callable(get_statuses):
        logger.error("models.camera_runtime_contract_missing")
        return api_success({"models": []})
    try:
        snapshot = get_statuses()
    except Exception:
        logger.warning("models.status_snapshot_failed", exc_info=True)
        snapshot = []
    out = snapshot if isinstance(snapshot, list) else []
    return api_success({"models": out})


@router.get("/json")
def list_models(request: Request, camera_id: str | None = None):
    """List all registered models, optionally filtered by camera_id."""
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    models = registry.list_models(camera_id=camera_id)
    latest_operations = {}
    # Match the operation-detail endpoint's authorization. A REST snapshot
    # must restore accepted commands after refresh or a lost WS notification,
    # without exposing command/evidence details to read-only model viewers.
    if models and require_permission(request, "manage_models"):
        service = _get_release_service(request)
        if service is None:
            return api_unavailable("发布状态服务不可用")
        try:
            latest_operations = service.get_latest_operations(
                [model.model_version_id for model in models],
            )
        except Exception:
            logger.error("models.latest_operations_lookup_failed", exc_info=True)
            return api_internal_error("发布状态读取失败")
    return api_success({
        "models": [
            {
                **m.to_dict(),
                "runtime_state": _model_runtime_state(request, m),
                "latest_release_operation": (
                    latest_operations[m.model_version_id].to_dict()
                    if m.model_version_id in latest_operations else None
                ),
            }
            for m in models
        ],
        "total": len(models),
    })


@router.post("/{version_id}/activate")
async def activate_model(request: Request, version_id: str):
    """Reject the legacy direct-activation command.

    Activation used to mutate ``is_active`` before a best-effort hot reload.
    That ordering cannot prove that database and runtime identities agree.  A
    canary must instead be promoted to production through the two-phase release
    command, which also enforces the v3 regression and trusted-soak evidence.
    """
    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied

    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")
    if registry.get_by_version_id(version_id) is None:
        return api_not_found(f"模型版本不存在: {version_id}")
    return api_conflict(
        "直接激活入口已关闭，未修改 registry 或运行时。请通过 "
        f"POST /api/models/{version_id}/promote 提交受控发布；production "
        "仍必须通过 baseline v3 与可信 72 小时 soak 门禁。"
    )


@router.post("/{version_id}/rollback")
def rollback_model(request: Request, version_id: str, body: ModelRollbackRequest | None = None):
    """Queue an explicit retired target; never infer or directly mutate a fallback."""
    if not require_human_permission(request, "rollback_model"):
        return api_forbidden("回滚需要具有权限的独立用户登录会话")
    if body is None:
        return api_conflict("旧回滚命令已禁用；请先检查目标版本并提交受控两阶段回滚")
    service = _get_release_service(request)
    if service is None:
        return api_unavailable("数据库不可用")
    worker = release_worker_for_app(request.app.state)
    if worker is None or not worker.running:
        return api_unavailable("模型发布执行器未运行")
    try:
        operation, record = service.submit_rollback(
            model_version_id=version_id,
            source_model_version_id=body.source_model_version_id,
            source_release_revision=body.source_release_revision,
            expected_revision=body.expected_revision,
            reason=body.reason, requested_by=current_username(request),
            idempotency_key=body.idempotency_key,
        )
        worker.notify()
        payload = {
            "operation_id": operation.operation_id, "operation": operation.to_dict(),
            "model": record.to_dict(), "runtime_synced": operation.status == "applied",
            "runtime_state": operation.to_dict().get("runtime_state"),
            "scene_reconfirmation_required": True,
        }
        if operation.status == "failed":
            return api_error(ErrorCode.CONFLICT, operation.error or "回滚失败", status_code=409, data=payload)
        return api_success(payload, status_code=202)
    except (ValueError, StageTransitionError) as exc:
        return api_conflict(str(exc))


@router.get("/{version_id}/rollback-readiness", response_model=ApiEnvelope[RollbackReadinessResponse])
def get_rollback_readiness(request: Request, version_id: str):
    if not require_human_permission(request, "rollback_model"):
        return api_forbidden("回滚检查需要具有权限的独立用户登录会话")
    service = _get_release_service(request)
    registry = _get_registry(request)
    if service is None or registry is None:
        return api_unavailable("数据库不可用")
    model = registry.get_by_version_id(version_id)
    if model is None:
        return api_not_found(f"模型版本不存在: {version_id}")
    context: dict = {}
    reasons: list[str] = []
    try:
        context = service.rollback_readiness(version_id)
    except Exception as exc:
        # Paths and detailed evidence exceptions remain in the server log.
        logger.warning("models.rollback_readiness_rejected", model_version_id=version_id, error=str(exc))
        reasons.append("历史生产记录或原始发布证据未通过复核；请核对目标场景绑定及证据文件")
    worker = release_worker_for_app(request.app.state)
    if worker is None or not worker.running:
        reasons.append("模型发布执行器未运行")
    manager = getattr(request.app.state, "camera_manager", None)
    runtime = _safe_runtime_state(manager, model.camera_id)
    if _public_runtime_method(manager, "prepare_model_release") is None or not runtime or (
        runtime.get("primary_model_version") != context.get("source_model_version_id")
        or runtime.get("release_revision") != context.get("source_release_revision")
    ):
        reasons.append("当前运行模型与待替换的生产版本或修订号不一致")
    if any(op.camera_id == model.camera_id for op in service.list_pending_operations()):
        reasons.append("该摄像头已有待完成的发布操作")
    return api_typed_success({
        "model_version_id": version_id, "camera_id": model.camera_id,
        "expected_revision": context.get("expected_revision", int(model.release_revision or 0)),
        "source_model_version_id": context.get("source_model_version_id"),
        "source_release_revision": context.get("source_release_revision"),
        "rollback_allowed": not reasons, "reasons": reasons,
        "scene_reconfirmation_required": True, "checked_at": datetime.now(timezone.utc),
    }, RollbackReadinessResponse)


def _run_batch_inference(
    image_paths: list[str], detector: object,
) -> tuple[list[dict], int]:
    """Run inference on a batch of images (blocking, runs in thread pool)."""
    import cv2

    results: list[dict] = []
    scored = 0
    threshold = getattr(detector, "threshold", 0.5)

    for img_path in image_paths:
        entry: dict = {"path": img_path}
        try:
            p = Path(img_path)
            if not p.exists():
                entry["error"] = "File not found"
                results.append(entry)
                continue

            frame = cv2.imread(str(p))
            if frame is None:
                entry["error"] = "Could not read image"
                results.append(entry)
                continue

            prediction = detector.predict(frame)
            score = float(prediction.anomaly_score) if hasattr(prediction, "anomaly_score") else (
                float(prediction.get("score", 0.0)) if isinstance(prediction, dict) else float(prediction)
            )
            entry["score"] = round(score, 4)
            entry["is_anomalous"] = score >= threshold
            scored += 1
        except Exception as e:
            entry["error"] = str(e)

        results.append(entry)

    return results, scored


@router.post("/batch-inference")
async def batch_inference(request: Request, body: ModelBatchInferenceRequest):
    """Score multiple images against a camera's active anomaly model.

    Request body: { camera_id: str, image_paths: list[str] }
    Returns: { results: [{path, score, is_anomalous, error?}], total, scored }
    """
    camera_id = body.camera_id
    image_paths = body.image_paths

    # Get the camera's anomaly detector via public API
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None:
        return api_unavailable("相机管理器不可用")

    det_status = camera_manager.get_detector_status(camera_id)
    if det_status is None:
        return api_not_found(f"相机 '{camera_id}' 未运行或不存在")

    get_detector = getattr(camera_manager, "get_anomaly_inference_handle", None)
    detector = get_detector(camera_id) if callable(get_detector) else None
    if detector is None:
        return api_unavailable(f"相机 '{camera_id}' 的异常检测器不可用")

    results, scored = await asyncio.to_thread(
        _run_batch_inference, image_paths, detector,
    )

    logger.info(
        "batch_inference.complete",
        camera_id=camera_id,
        total=len(image_paths),
        scored=scored,
    )

    return api_success({
        "results": results,
        "total": len(image_paths),
        "scored": scored,
    })


# ── Release pipeline endpoints ──


@router.get("/release-operations/{operation_id}")
def get_release_operation(request: Request, operation_id: str):
    """Return durable desired/applied state for a release command."""

    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied
    service = _get_release_service(request)
    if service is None:
        return api_unavailable("数据库不可用")
    operation = service.get_operation(operation_id)
    if operation is None:
        return api_not_found(f"Release operation not found: {operation_id}")
    return api_success({"operation": operation.to_dict()})


def _public_runtime_method(manager: object | None, name: str):
    """Return one explicitly implemented runtime method, not a mock/proxy ghost."""

    if manager is None:
        return None
    try:
        if inspect.getattr_static(manager, name, None) is None:
            return None
    except (AttributeError, TypeError):
        return None
    try:
        method = getattr(manager, name, None)
    except Exception:
        return None
    return method if callable(method) else None


def _append_readiness_gate(
    gates: list[dict],
    *,
    key: str,
    label: str,
    passed: bool,
    reason: str,
) -> None:
    gates.append({
        "key": key,
        "label": label,
        "passed": passed is True,
        "blocking": True,
        "reason": None if passed is True else reason,
    })


def _safe_runtime_state(manager: object | None, camera_id: str) -> dict | None:
    """Read and whitelist runtime identity fields so paths cannot reach the API."""

    getter = _public_runtime_method(manager, "get_runtime_model_state")
    if getter is None:
        return None
    try:
        state = getter(camera_id)
    except Exception:
        logger.warning(
            "models.release_readiness_runtime_state_failed",
            camera_id=camera_id,
            exc_info=True,
        )
        return None
    if not isinstance(state, dict):
        return None
    public_string_keys = (
        "primary_model_version",
        "shadow_model_version",
        "runtime_artifact_sha256",
        "scene_profile_id",
    )
    public = {
        key: value
        for key in public_string_keys
        if (value := state.get(key)) is None
        or isinstance(value, str)
    }
    revision = state.get("release_revision")
    if revision is None or type(revision) is int:
        public["release_revision"] = revision
    return public


@router.get(
    "/{version_id}/release-readiness",
    response_model=ApiEnvelope[ReleaseReadinessResponse],
)
def get_release_readiness(request: Request, version_id: str):
    """Return the server-authoritative, fail-closed production gate report.

    This read endpoint never accepts evidence paths and never reconstructs an
    authorization bundle.  A canary therefore remains NO-GO until its actual
    promotion request supplies server-side evidence sources.  Once production
    has been applied, persisted evidence is revalidated against the current
    artifact before the runtime can be reported safe.
    """

    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")
    model = registry.get_by_version_id(version_id)
    if model is None:
        return api_not_found(f"Model not found: {version_id}")

    service = _get_release_service(request)
    if service is None:
        return api_unavailable("发布状态服务不可用")
    try:
        operation = service.get_latest_operation(
            version_id,
            target_stage=ModelStage.PRODUCTION.value,
        )
    except Exception:
        logger.error(
            "models.release_readiness_operation_lookup_failed",
            model_version_id=version_id,
            exc_info=True,
        )
        return api_internal_error("发布状态读取失败")

    gates: list[dict] = []
    stage_eligible = model.stage in {
        ModelStage.CANARY.value,
        ModelStage.PRODUCTION.value,
    }
    _append_readiness_gate(
        gates,
        key="canary_stage",
        label="Canary stage completed",
        passed=stage_eligible,
        reason=(
            f"Applied stage {model.stage!r} is not eligible for production; "
            "the model must complete canary first."
        ),
    )

    artifact_sha256: str | None = None
    if model.model_path:
        try:
            artifact_sha256 = runtime_artifact_sha256(model.model_path)
        except Exception:
            logger.warning(
                "models.release_readiness_artifact_unavailable",
                model_version_id=version_id,
                exc_info=True,
            )
    _append_readiness_gate(
        gates,
        key="artifact_sha256",
        label="Registry-authorized artifact fingerprint",
        passed=artifact_sha256 is not None,
        reason=(
            "The exact runtime artifact SHA-256 cannot be calculated from the "
            "registry-authorized artifact."
        ),
    )

    manager = getattr(request.app.state, "camera_manager", None)
    prepare_release = _public_runtime_method(manager, "prepare_model_release")
    _append_readiness_gate(
        gates,
        key="two_phase_runtime",
        label="Two-phase runtime release",
        passed=prepare_release is not None,
        reason="Camera runtime does not expose the two-phase release contract.",
    )

    runtime_state = _safe_runtime_state(manager, model.camera_id)
    runtime_version_matches = bool(
        runtime_state
        and runtime_state.get("primary_model_version") == model.model_version_id
    )
    _append_readiness_gate(
        gates,
        key="runtime_model_identity",
        label="Applied runtime model identity",
        passed=runtime_version_matches,
        reason="The camera runtime is not serving this exact model version.",
    )
    runtime_artifact_matches = bool(
        artifact_sha256
        and runtime_state
        and runtime_state.get("runtime_artifact_sha256") == artifact_sha256
    )
    _append_readiness_gate(
        gates,
        key="runtime_artifact_identity",
        label="Applied runtime artifact identity",
        passed=runtime_artifact_matches,
        reason="The camera runtime did not prove the current artifact SHA-256.",
    )
    runtime_revision_matches = bool(
        runtime_state
        and type(runtime_state.get("release_revision")) is int
        and runtime_state.get("release_revision") == int(model.release_revision or 0)
    )
    _append_readiness_gate(
        gates,
        key="runtime_release_revision",
        label="Applied runtime release revision",
        passed=runtime_revision_matches,
        reason="The camera runtime release revision does not match the registry.",
    )

    scene_status: dict | None = None
    get_scene_status = _public_runtime_method(manager, "get_scene_guard_status")
    if get_scene_status is not None:
        try:
            observed = get_scene_status(model.camera_id)
            scene_status = observed if isinstance(observed, dict) else None
        except Exception:
            logger.warning(
                "models.release_readiness_scene_guard_failed",
                camera_id=model.camera_id,
                exc_info=True,
            )
    _append_readiness_gate(
        gates,
        key="scene_guard_enabled",
        label="SceneGuard enabled",
        passed=bool(scene_status and scene_status.get("enabled") is True),
        reason="SceneGuard is unavailable or disabled for this camera.",
    )
    _append_readiness_gate(
        gates,
        key="scene_guard_usable",
        label="SceneGuard scene match",
        passed=bool(scene_status and scene_status.get("usable") is True),
        reason="SceneGuard has not confirmed that the current scene is usable.",
    )
    profile_id = scene_status.get("profile_id") if scene_status else None
    _append_readiness_gate(
        gates,
        key="scene_profile",
        label="Approved scene profile binding",
        passed=isinstance(profile_id, str) and bool(profile_id.strip()),
        reason="SceneGuard has no bound scene profile.",
    )
    _append_readiness_gate(
        gates,
        key="scene_model_binding",
        label="Scene profile model binding",
        passed=bool(
            scene_status
            and scene_status.get("bound_model_version") == model.model_version_id
        ),
        reason="The scene profile is not bound to this exact model version.",
    )
    bound_artifact_sha256 = (
        scene_status.get("bound_model_artifact_sha256")
        if scene_status
        else None
    )
    scene_artifact_matches = bool(
        artifact_sha256
        and (
            bound_artifact_sha256 in (None, "")
            or bound_artifact_sha256 == artifact_sha256
        )
    )
    _append_readiness_gate(
        gates,
        key="scene_artifact_binding",
        label="Scene profile artifact binding",
        passed=scene_artifact_matches,
        reason="The scene profile artifact binding conflicts with the current artifact.",
    )

    if model.stage == ModelStage.PRODUCTION.value:
        evidence_validation = None
        try:
            evidence_validation = validate_persisted_release_evidence(model)
        except Exception:
            evidence_valid = False
            logger.error(
                "models.release_readiness_evidence_validation_failed",
                model_version_id=version_id,
                exc_info=True,
            )
        else:
            evidence_valid = evidence_validation.valid is True
        if not evidence_valid and evidence_validation is not None:
            logger.warning(
                "models.release_readiness_evidence_rejected",
                model_version_id=version_id,
                reasons=evidence_validation.reasons,
            )
        evidence_reason = (
            "Persisted production evidence failed server revalidation against "
            "the current registry row and artifact."
        )
    else:
        evidence_valid = False
        evidence_reason = (
            "Production evidence is not reconstructed by this GET request; "
            "submit server-side evidence sources with the production promotion."
        )
    for key, label in (
        ("baseline_v3", "Baseline v3 and strict decoder evidence"),
        ("regression_metrics", "Release regression metrics"),
        ("trusted_soak", "Trusted signed 72-hour soak"),
    ):
        _append_readiness_gate(
            gates,
            key=key,
            label=label,
            passed=evidence_valid,
            reason=evidence_reason,
        )

    failed_reasons = [
        gate["reason"]
        for gate in gates
        if gate["blocking"] and not gate["passed"] and gate["reason"]
    ]
    desired_status = (
        operation.status
        if operation is not None
        and operation.status in {
            ReleaseOperationStatus.PENDING.value,
            ReleaseOperationStatus.APPLIED.value,
            ReleaseOperationStatus.FAILED.value,
        }
        else None
    )
    if desired_status == ReleaseOperationStatus.PENDING.value:
        failed_reasons.append("A production release operation is still pending.")
    elif desired_status == ReleaseOperationStatus.FAILED.value:
        failed_reasons.append("The latest production release operation failed.")
    reasons = list(dict.fromkeys(failed_reasons))

    all_blocking_gates_passed = all(
        gate["passed"] for gate in gates if gate["blocking"]
    )
    production_ready = bool(
        model.stage == ModelStage.PRODUCTION.value
        and all_blocking_gates_passed
    )
    promotion_allowed = bool(
        model.stage == ModelStage.CANARY.value
        and all_blocking_gates_passed
        and desired_status not in {
            ReleaseOperationStatus.PENDING.value,
            ReleaseOperationStatus.FAILED.value,
        }
    )
    if model.stage == ModelStage.PRODUCTION.value:
        status = "applied"
    elif desired_status == ReleaseOperationStatus.PENDING.value:
        status = "pending"
    elif desired_status == ReleaseOperationStatus.FAILED.value:
        status = "failed"
    elif promotion_allowed:
        status = "ready"
    else:
        status = "no_go"

    return api_typed_success(
        {
            "model_version_id": model.model_version_id,
            "camera_id": model.camera_id,
            "target_stage": ModelStage.PRODUCTION.value,
            "status": status,
            "production_ready": production_ready,
            "promotion_allowed": promotion_allowed,
            "operation_id": operation.operation_id if operation else None,
            "desired_stage": operation.target_stage if operation else None,
            "desired_status": desired_status,
            "applied_stage": model.stage,
            "applied_revision": int(model.release_revision or 0),
            "runtime_state": runtime_state,
            "gates": gates,
            "reasons": reasons,
            "checked_at": datetime.now(timezone.utc),
        },
        ReleaseReadinessResponse,
    )


@router.post("/{version_id}/promote")
def promote_model(
    request: Request,
    version_id: str,
    body: ModelPromoteRequest,
):
    """Promote a model to the next release stage.

    Body: { target_stage, reason?, canary_camera_id?,
            release_evidence_sources? }

    For a production promotion the API accepts only paths to server-side
    regression/config/profile/baseline/corpus/soak artifacts.  It rebuilds the
    canonical evidence itself; client-supplied evidence booleans or digests are
    rejected.  The authenticated identity is always used for audit/approval.
    """
    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied

    service = _get_release_service(request)
    if service is None:
        return api_unavailable("数据库不可用")

    target_stage = body.target_stage
    if target_stage == ModelStage.PRODUCTION.value and not require_human_permission(
        request, "manage_models",
    ):
        return api_forbidden("生产发布审批需要具有权限的独立用户登录会话")
    triggered_by = current_username(request)

    try:
        if target_stage == ModelStage.PRODUCTION.value:
            if "release_evidence" in body.model_fields_set:
                return api_validation_error(
                    "Client-supplied release_evidence is not accepted; use "
                    "release_evidence_sources so the server can rebuild it"
                )
            release_registry = _get_registry(request)
            if release_registry is None:
                return api_unavailable("数据库不可用")
            production_candidate = release_registry.get_by_version_id(version_id)
            if production_candidate is None:
                return api_not_found(f"Model not found: {version_id}")
            if not production_candidate.model_path:
                return api_validation_error("Registry model_path is missing")
            if body.release_evidence_sources is None:
                return api_validation_error("release_evidence_sources must be an object")

        if target_stage == ModelStage.RETIRED.value:
            registry = _get_registry(request)
            if registry is None:
                return api_unavailable("数据库不可用")
            existing = registry.get_by_version_id(version_id)
            if existing is None:
                return api_not_found(f"Model not found: {version_id}")
            if existing.stage == ModelStage.PRODUCTION.value and existing.is_active:
                return api_validation_error(
                    "Active production models cannot be retired directly; "
                    "promote a replacement to production instead."
                )

        worker = release_worker_for_app(request.app.state)
        if worker is None or not worker.running:
            return api_unavailable("模型发布执行器未运行")
        operation, record = service.submit_transition(
            model_version_id=version_id,
            target_stage=target_stage,
            requested_by=triggered_by,
            reason=body.reason,
            canary_camera_id=body.canary_camera_id,
            release_evidence_sources=(
                body.release_evidence_sources.model_dump()
                if body.release_evidence_sources is not None
                else None
            ),
            expected_revision=body.expected_revision,
            idempotency_key=(
                request.headers.get("idempotency-key")
                or body.idempotency_key
            ),
        )
        worker.notify()
        payload = {
            "operation_id": operation.operation_id,
            "operation": operation.to_dict(),
            "model": record.to_dict(),
            "runtime_synced": operation.status == ReleaseOperationStatus.APPLIED.value,
            "runtime_state": operation.to_dict().get("runtime_state"),
        }
        if operation.status == ReleaseOperationStatus.FAILED.value:
            return api_error(
                ErrorCode.CONFLICT,
                operation.error or "Runtime release state was not applied",
                status_code=409,
                data=payload,
            )
        return api_success(payload, status_code=202)
    except ValueError as e:
        return api_validation_error(str(e))
    except OSError as e:
        return api_validation_error(
            f"Release evidence source is unavailable: {e}"
        )
    except StageTransitionError as e:
        return api_conflict(str(e))


@router.post("/{version_id}/retire")
def retire_model(
    request: Request,
    version_id: str,
    body: ModelRetireRequest | None = None,
):
    """Retire a model (any stage → retired)."""
    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied

    service = _get_release_service(request)
    if service is None:
        return api_unavailable("数据库不可用")

    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    body = body or ModelRetireRequest()

    triggered_by = current_username(request)

    try:
        existing = registry.get_by_version_id(version_id)
        if existing is None:
            return api_not_found(f"Model not found: {version_id}")
        if existing.stage == ModelStage.PRODUCTION.value and existing.is_active:
            return api_validation_error(
                "Active production models cannot be retired directly; "
                "promote a replacement to production instead."
            )

        worker = release_worker_for_app(request.app.state)
        if worker is None or not worker.running:
            return api_unavailable("模型发布执行器未运行")
        operation, record = service.submit_transition(
            model_version_id=version_id,
            target_stage=ModelStage.RETIRED.value,
            requested_by=triggered_by,
            reason=body.reason,
            expected_revision=body.expected_revision,
            idempotency_key=(
                request.headers.get("idempotency-key")
                or body.idempotency_key
            ),
        )
        worker.notify()
        payload = {
            "operation_id": operation.operation_id,
            "operation": operation.to_dict(),
            "model": record.to_dict(),
            "runtime_synced": operation.status == ReleaseOperationStatus.APPLIED.value,
            "runtime_state": operation.to_dict().get("runtime_state"),
        }
        if operation.status == ReleaseOperationStatus.FAILED.value:
            return api_error(
                ErrorCode.CONFLICT,
                operation.error or "Runtime retirement was not applied",
                status_code=409,
                data=payload,
            )
        return api_success(payload, status_code=202)
    except ValueError as e:
        return api_not_found(str(e))
    except StageTransitionError as e:
        return api_conflict(str(e))


@router.delete("/{version_id}")
def delete_model(request: Request, version_id: str):
    """Delete a model version (only non-active, non-production models)."""
    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied

    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    record = registry.get_by_version_id(version_id)
    if record is None:
        return api_not_found(f"模型不存在: {version_id}")
    if record.is_active:
        return api_validation_error("无法删除当前激活的模型")

    if record.stage in (ModelStage.SHADOW.value, ModelStage.CANARY.value, ModelStage.PRODUCTION.value):
        return api_validation_error(
            f"无法删除处于 {record.stage} 阶段的模型，请先退役"
        )

    try:
        result = registry.delete_model(
            version_id,
            requested_by=current_username(request),
            expected_revision=int(record.release_revision or 0),
        )
    except ModelDeletionConflictError as exc:
        return api_conflict(str(exc))
    except ValueError as exc:
        return api_not_found(str(exc))
    # The registry returns only after its deletion/audit transaction commits.
    # Let ConnectionManager assign the transport resource revision rather than
    # reusing a release revision from a row that no longer exists.
    ws_manager = getattr(request.app.state, "ws_manager", None)
    if ws_manager is not None:
        try:
            accepted = ws_manager.broadcast(
                "models",
                {
                    "type": "model.deleted",
                    "event": "model.deleted",
                    "model_version_id": version_id,
                    "camera_id": record.camera_id,
                    "deleted": True,
                    "previous_stage": record.stage,
                    "previous_release_revision": int(record.release_revision or 0),
                    "artifacts_retained": result.get("artifacts_retained", True),
                    "garbage_collection": result.get("garbage_collection", "deferred"),
                },
                event_type="model.deleted",
                resource_id=version_id,
            )
            if accepted is False:
                logger.warning("model.deleted_event_not_queued", model_version_id=version_id)
        except Exception:
            # A transport failure cannot undo a committed deletion or turn it
            # into a retryable delete. Periodic REST reconciliation remains.
            logger.warning("model.deleted_event_failed", model_version_id=version_id, exc_info=True)
    logger.info("model.deleted", model_version_id=version_id)
    return api_success(result)


def _get_trainer(request: Request):
    """Construct a ModelTrainer from app state config."""
    trainer = getattr(request.app.state, "model_trainer", None)
    if trainer is not None:
        bind_registry = _public_runtime_method(trainer, "bind_model_registry")
        if bind_registry is not None:
            registry = _get_registry(request)
            if registry is None:
                return None
            bind_registry(registry)
        return trainer

    from argus.anomaly.baseline import BaselineManager
    from argus.anomaly.trainer import ModelTrainer

    config = getattr(request.app.state, "config", None)
    if config is None:
        return None
    baselines_dir = str(config.storage.baselines_dir)
    models_dir = str(config.storage.models_dir)
    exports_dir = str(config.storage.exports_dir)
    bm = BaselineManager(baselines_dir=baselines_dir)
    return ModelTrainer(
        baseline_manager=bm,
        models_dir=models_dir,
        exports_dir=exports_dir,
        model_registry=_get_registry(request),
    )


def _get_baseline_manager(request: Request):
    """Get or create a BaselineManager from app state or config."""
    baseline_mgr = getattr(request.app.state, "baseline_manager", None)
    if baseline_mgr is not None:
        return baseline_mgr
    config = getattr(request.app.state, "config", None)
    if config:
        from argus.anomaly.baseline import BaselineManager
        return BaselineManager(baselines_dir=str(config.storage.baselines_dir))
    return None


def _artifact_operation_context(record: ModelRecord) -> tuple[str, int]:
    """Resolve explicit zone/image identity before deriving a new candidate."""
    from argus.anomaly.artifacts import artifact_metadata

    source = Path(record.model_path)
    metadata = artifact_metadata(source)
    artifact_camera = metadata.get("camera_id")
    if artifact_camera is not None and artifact_camera != record.camera_id:
        raise ValueError("模型产物与 registry 相机身份不匹配")
    try:
        params = json.loads(record.training_params or "{}")
    except (TypeError, ValueError):
        params = {}
    if not isinstance(params, dict):
        params = {}
    zone_id = metadata.get("zone_id") or params.get("zone_id")
    if not isinstance(zone_id, str) or not zone_id.strip():
        camera_parts = [
            index for index, part in enumerate(source.parts)
            if part == record.camera_id
        ]
        if camera_parts and camera_parts[-1] + 1 < len(source.parts):
            zone_id = source.parts[camera_parts[-1] + 1]
        else:
            zone_id = source.name if source.is_dir() else source.parent.name
    image_size = metadata.get("image_size", params.get("image_size", 256))
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("模型训练参数 image_size 无效")
    return zone_id, image_size


@router.post("/{version_id}/reexport")
async def reexport_model(
    request: Request,
    version_id: str,
    body: ModelReexportRequest | None = None,
):
    """Re-export model from checkpoint, fitting PostProcessor MinMax.

    Body: { export_format?: str, quantization?: str }
    """
    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied

    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    record = registry.get_by_version_id(version_id)
    if record is None:
        return api_not_found(f"模型不存在: {version_id}")
    if not record.model_path:
        return api_validation_error("模型没有关联的文件路径")

    model_dir = Path(record.model_path)
    if not model_dir.exists():
        return api_not_found(f"模型目录不存在: {record.model_path}")

    body = body or ModelReexportRequest()
    export_format = body.export_format
    quantization = body.quantization
    try:
        zone_id, image_size = _artifact_operation_context(record)
    except (OSError, ValueError) as exc:
        return api_validation_error(str(exc))

    trainer = _get_trainer(request)
    if trainer is None:
        return api_unavailable("配置不可用")

    result = await asyncio.to_thread(
        trainer.reexport_model,
        model_dir=model_dir,
        export_format=export_format,
        quantization=quantization,
        model_type=record.model_type,
        camera_id=record.camera_id,
        zone_id=zone_id,
        image_size=image_size,
    )

    if result.get("status") == "error":
        return api_internal_error(result["error"])

    return api_success(result)


@router.post("/{version_id}/recalibrate")
async def recalibrate_model(request: Request, version_id: str):
    """Re-calibrate score normalization using baseline images.

    Runs conformal calibration + sigmoid recalibration, persists to calibration.json.
    """
    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied

    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    record = registry.get_by_version_id(version_id)
    if record is None:
        return api_not_found(f"模型不存在: {version_id}")
    if not record.model_path:
        return api_validation_error("模型没有关联的文件路径")

    model_dir = Path(record.model_path)

    try:
        zone_id, _image_size = _artifact_operation_context(record)
    except (OSError, ValueError) as exc:
        return api_validation_error(str(exc))

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return api_unavailable("基线管理器不可用")

    baseline_dir = baseline_mgr.get_baseline_dir(record.camera_id, zone_id)
    if not baseline_dir.is_dir():
        return api_not_found(f"基线目录不存在: {baseline_dir}")

    trainer = _get_trainer(request)
    if trainer is None:
        return api_unavailable("配置不可用")

    result = await asyncio.to_thread(
        trainer.recalibrate_model,
        model_dir=model_dir,
        baseline_dir=baseline_dir,
        camera_id=record.camera_id,
        zone_id=zone_id,
        model_type=record.model_type,
    )

    if result.get("status") == "error":
        return api_internal_error(result["error"])

    return api_success(result)


@router.get("/{version_id}/stage-history")
async def stage_history(request: Request, version_id: str):
    """Get version event history for a specific model."""
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    events = registry.get_version_events(model_version_id=version_id)
    return api_success({
        "events": [e.to_dict() for e in events],
        "total": len(events),
    })


@router.get("/events/list")
async def list_version_events(
    request: Request,
    camera_id: str | None = None,
    limit: int = 50,
):
    """Query global version transition events."""
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    events = registry.get_version_events(camera_id=camera_id, limit=limit)
    return api_success({
        "events": [e.to_dict() for e in events],
        "total": len(events),
    })


@router.get("/{version_id}/shadow-report")
async def shadow_report(
    request: Request,
    version_id: str,
    camera_id: str | None = None,
    days: int = 7,
):
    """Get shadow inference comparison report."""
    pipeline = _get_release_pipeline(request)
    if pipeline is None:
        return api_unavailable("数据库不可用")

    stats = pipeline.get_shadow_stats(
        shadow_version_id=version_id,
        camera_id=camera_id,
        days=days,
    )
    stats["gate_report"] = pipeline.get_shadow_gate_report(
        shadow_version_id=version_id,
        camera_id=camera_id,
        days=days,
    )
    return api_success(stats)


# ── A/B Comparison endpoints ──


def _query_shadow_logs(session, version_id: str, camera_id: str | None, days: int):
    """Build a filtered ShadowInferenceLog query (shared by ab-scores/ab-distribution)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query = (
        session.query(ShadowInferenceLog)
        .filter(ShadowInferenceLog.shadow_version_id == version_id)
        .filter(ShadowInferenceLog.timestamp >= cutoff)
    )
    if camera_id:
        query = query.filter(ShadowInferenceLog.camera_id == camera_id)
    return query


@router.get("/{version_id}/ab-scores")
async def ab_compare_scores(
    request: Request,
    version_id: str,
    camera_id: str | None = None,
    days: int = 7,
    limit: int = 500,
):
    """Get time-series shadow vs production scores for A/B comparison chart."""
    session_factory = _get_session_factory(request)
    if session_factory is None:
        return api_unavailable("数据库不可用")

    with session_factory() as session:
        logs = list(
            _query_shadow_logs(session, version_id, camera_id, days)
            .order_by(ShadowInferenceLog.timestamp.desc())
            .limit(limit)
            .all()
        )

    logs.reverse()

    scores = [
        {
            "t": log.timestamp.isoformat() if log.timestamp else None,
            "shadow": round(log.shadow_score, 4),
            "production": round(log.production_score, 4) if log.production_score is not None else None,
            "shadow_alert": log.shadow_would_alert,
            "prod_alert": log.production_alerted,
        }
        for log in logs
    ]

    return api_success({
        "scores": scores,
        "total": len(scores),
        "shadow_version_id": version_id,
    })


@router.get("/{version_id}/ab-distribution")
async def ab_compare_distribution(
    request: Request,
    version_id: str,
    camera_id: str | None = None,
    days: int = 7,
    bins: int = 20,
):
    """Get score distribution histograms for shadow vs production."""
    session_factory = _get_session_factory(request)
    if session_factory is None:
        return api_unavailable("数据库不可用")

    with session_factory() as session:
        # Only fetch the two score columns, not full ORM objects
        rows = (
            _query_shadow_logs(session, version_id, camera_id, days)
            .with_entities(ShadowInferenceLog.shadow_score, ShadowInferenceLog.production_score)
            .limit(50000)
            .all()
        )

    if not rows:
        return api_success({
            "bin_edges": [],
            "shadow_counts": [],
            "production_counts": [],
            "total_samples": 0,
        })

    bin_edges = [round(i / bins, 4) for i in range(bins + 1)]
    shadow_counts = [0] * bins
    prod_counts = [0] * bins

    for shadow_score, prod_score in rows:
        shadow_counts[min(int(shadow_score * bins), bins - 1)] += 1
        if prod_score is not None:
            prod_counts[min(int(prod_score * bins), bins - 1)] += 1

    return api_success({
        "bin_edges": bin_edges,
        "shadow_counts": shadow_counts,
        "production_counts": prod_counts,
        "total_samples": len(rows),
    })


# ── Shadow detector cache (LRU, max 3 entries) ──
_shadow_detector_cache: OrderedDict[str, object] = OrderedDict()
_shadow_cache_lock = threading.Lock()


def _get_or_load_shadow_detector(shadow_model_path: Path, shadow_version_id: str, threshold: float):
    """Get cached shadow detector or load it. Thread-safe, LRU eviction (max 3)."""
    with _shadow_cache_lock:
        cached = _shadow_detector_cache.get(shadow_version_id)
        if cached is not None:
            # Move to end for LRU ordering
            _shadow_detector_cache.move_to_end(shadow_version_id)  # type: ignore[attr-defined]
            return cached

    from argus.anomaly.detector import AnomalibDetector

    det = AnomalibDetector(model_path=shadow_model_path, threshold=threshold)

    with _shadow_cache_lock:
        if len(_shadow_detector_cache) >= 3:
            # Evict least-recently-used (first item)
            evicted_key, _ = _shadow_detector_cache.popitem(last=False)  # type: ignore[call-arg]
            logger.debug("ab_compare.cache_evict", evicted=evicted_key)
        _shadow_detector_cache[shadow_version_id] = det

    logger.info("ab_compare.shadow_loaded", version_id=shadow_version_id)
    return det


def _run_ab_heatmap(
    frame: np.ndarray,
    production_detector: object,
    shadow_model_path: Path,
    shadow_version_id: str,
    shadow_threshold: float,
) -> dict:
    """Run both models on a frame and produce heatmap images (blocking)."""
    result: dict = {}

    # --- Production model ---
    t0 = time.perf_counter()
    try:
        prod_pred = production_detector.predict(frame)
        result["production_score"] = round(float(prod_pred.anomaly_score), 4)
        result["production_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if prod_pred.anomaly_map is not None:
            heatmap = _anomaly_map_to_jpeg(prod_pred.anomaly_map, frame.shape[:2])
            result["production_heatmap"] = base64.b64encode(heatmap).decode()
        else:
            result["production_heatmap"] = None
    except Exception as e:
        result["production_error"] = str(e)

    # --- Shadow model (cached) ---
    t1 = time.perf_counter()
    try:
        shadow_det = _get_or_load_shadow_detector(
            shadow_model_path, shadow_version_id, shadow_threshold,
        )
        shadow_pred = shadow_det.predict(frame)
        result["shadow_score"] = round(float(shadow_pred.anomaly_score), 4)
        result["shadow_latency_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        if shadow_pred.anomaly_map is not None:
            heatmap = _anomaly_map_to_jpeg(shadow_pred.anomaly_map, frame.shape[:2])
            result["shadow_heatmap"] = base64.b64encode(heatmap).decode()
        else:
            result["shadow_heatmap"] = None
    except Exception as e:
        result["shadow_error"] = str(e)

    # --- Original frame thumbnail ---
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result["original_frame"] = base64.b64encode(buf.tobytes()).decode()

    return result


def _anomaly_map_to_jpeg(anomaly_map: np.ndarray, target_hw: tuple[int, int]) -> bytes:
    """Convert an anomaly map (float or uint8) to a JET-colored JPEG thumbnail."""
    amap = anomaly_map
    if amap.ndim == 3:
        amap = amap[:, :, 0]

    # Normalize to 0-255
    if amap.dtype != np.uint8:
        amap = np.clip(amap, 0, 1)
        amap = (amap * 255).astype(np.uint8)

    # Resize to match frame dimensions
    h, w = target_hw
    if amap.shape[:2] != (h, w):
        amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)

    colored = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


@router.post("/ab-live-compare")
async def ab_live_compare(request: Request, body: ModelAbLiveCompareRequest):
    """Run both production and shadow model on a live camera frame.

    Body: { camera_id: str, shadow_version_id: str }
    Returns: base64 encoded heatmaps for production and shadow, plus scores.
    """
    camera_id = body.camera_id
    shadow_version_id = body.shadow_version_id

    # Get camera frame
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None:
        return api_unavailable("相机管理器不可用")

    frame = camera_manager.get_latest_frame(camera_id)
    if frame is None:
        return api_not_found(f"相机 '{camera_id}' 无可用帧")

    # Get production detector
    get_detector = getattr(camera_manager, "get_anomaly_inference_handle", None)
    production_detector = get_detector(camera_id) if callable(get_detector) else None
    if production_detector is None:
        return api_unavailable(f"相机 '{camera_id}' 的异常检测器不可用")

    # Get shadow model path from registry
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    shadow_record = registry.get_by_version_id(shadow_version_id)
    if shadow_record is None:
        return api_not_found(f"影子模型不存在: {shadow_version_id}")
    if not shadow_record.model_path:
        return api_validation_error("影子模型没有关联的文件路径")

    shadow_model_path = Path(shadow_record.model_path)
    if not shadow_model_path.exists():
        return api_not_found(f"影子模型文件不存在: {shadow_record.model_path}")

    shadow_threshold = 0.5
    if shadow_record.calibration_thresholds:
        import json
        try:
            cal = json.loads(shadow_record.calibration_thresholds)
            shadow_threshold = cal.get("threshold", 0.5)
        except Exception:
            logger.warning("models.calibration_thresholds_parse_failed", exc_info=True)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _AB_EXECUTOR,
        _run_ab_heatmap, frame, production_detector,
        shadow_model_path, shadow_version_id, shadow_threshold,
    )

    return api_success(result)


# ── Backbone management endpoints ──


@router.get("/backbone/status")
async def backbone_status(request: Request):
    """Get current backbone loading status."""
    from argus.anomaly.backbone_manager import BackboneManager

    manager = BackboneManager.get_instance()
    return api_success({
        "version": manager.version,
        "loaded": manager.is_loaded,
    })


@router.post("/backbone/upgrade")
async def backbone_upgrade(request: Request, body: BackboneUpgradeRequest):
    """Upgrade the shared backbone.

    Body: { backbone_path: str, version: str, triggered_by: str }
    """
    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied

    from pathlib import Path

    from argus.anomaly.backbone_manager import BackboneManager

    backbone_path = body.backbone_path
    version = body.version
    triggered_by = current_username(request)

    manager = BackboneManager.get_instance()
    success = await asyncio.to_thread(manager.upgrade, Path(backbone_path), version)

    if success:
        logger.info(
            "backbone.upgraded",
            version=version,
            triggered_by=triggered_by,
        )
        return api_success({
            "version": version,
        })
    else:
        return api_internal_error("Backbone升级失败")


@router.get("/threshold-preview")
def threshold_preview(
    request: Request,
    camera_id: str | None = None,
    threshold: float = 0.5,
    days: int = 7,
) -> JSONResponse:
    """Preview detection metrics at a given anomaly threshold.

    Uses historical inference records to compute:
    - Total frames
    - Frames above threshold (would trigger alert)
    - Alert rate (fraction of frames above threshold)
    - Score percentiles (p50, p90, p95, p99)

    This allows operators to tune the threshold with real data feedback.
    """
    from sqlalchemy import select
    from argus.storage.models import InferenceRecord

    db = request.app.state.db
    if db is None:
        return api_unavailable("数据库未初始化")

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()

    with db.get_session() as session:
        base = select(InferenceRecord).where(InferenceRecord.timestamp >= cutoff_ts)
        if camera_id:
            base = base.where(InferenceRecord.camera_id == camera_id)

        # Get all scores for this camera in the time window
        score_stmt = select(InferenceRecord.anomaly_score).where(
            InferenceRecord.timestamp >= cutoff_ts
        )
        if camera_id:
            score_stmt = score_stmt.where(InferenceRecord.camera_id == camera_id)

        scores = [row[0] for row in session.execute(score_stmt).all()]

    if not scores:
        return api_success(data={
            "total_frames": 0,
            "above_threshold": 0,
            "alert_rate": 0.0,
            "threshold": threshold,
            "percentiles": {},
            "message": "暂无推理记录",
        })

    scores_arr = np.array(scores)
    total = len(scores_arr)
    above = int(np.sum(scores_arr >= threshold))
    alert_rate = above / total if total > 0 else 0.0

    percentiles = {
        "p50": float(np.percentile(scores_arr, 50)),
        "p75": float(np.percentile(scores_arr, 75)),
        "p90": float(np.percentile(scores_arr, 90)),
        "p95": float(np.percentile(scores_arr, 95)),
        "p99": float(np.percentile(scores_arr, 99)),
        "max": float(np.max(scores_arr)),
        "min": float(np.min(scores_arr)),
        "mean": float(np.mean(scores_arr)),
    }

    # Generate histogram for score distribution
    hist_counts, hist_edges = np.histogram(scores_arr, bins=50, range=(0.0, 1.0))
    histogram = [
        {"bin_start": float(hist_edges[i]), "bin_end": float(hist_edges[i + 1]), "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    return api_success(data={
        "total_frames": total,
        "above_threshold": above,
        "alert_rate": round(alert_rate, 6),
        "threshold": threshold,
        "percentiles": percentiles,
        "histogram": histogram,
        "days": days,
        "camera_id": camera_id,
    })
