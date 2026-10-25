"""Managed shadow task commands. Never a second production publisher."""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Annotated

import structlog
from fastapi import APIRouter, Header, Request
from pydantic import ValidationError

from argus.core.task_execution import TaskExecutionError
from argus.dashboard.analysis_task_contracts import AnalysisTaskResponse
from argus.dashboard.api_response import (
    ApiEnvelope, api_conflict, api_forbidden, api_internal_error, api_not_found,
    api_typed_success, api_unavailable, api_validation_error,
)
from argus.dashboard.auth import current_username, require_human_permission, require_permission
from argus.dashboard.routes.analysis_tasks import _public_task, _service
from argus.dashboard.task_platform_contracts import (
    TaskBundleCreateRequest, TaskCreateRequest, TaskDocumentResponse,
    TaskDocumentsResponse, TaskModelRegisterRequest, TaskReleaseReadinessResponse,
    TaskResultsResponse, TaskRevisionRequest, TaskStartRequest, TaskUpdateRequest,
)

logger = structlog.get_logger()
router = APIRouter()
models_router = APIRouter()
CommandKey = Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=128)]


def _actor(request: Request, permission: str, *, human: bool = False) -> str | None:
    allowed = require_human_permission(request, permission) if human else require_permission(request, permission)
    actor = current_username(request)
    return actor if allowed and actor and actor.strip() and actor.strip().lower() != "unknown" else None


def safe_document(value: object) -> object:
    """Do not disclose resolved model locations or camera-source credentials."""
    if isinstance(value, dict):
        return {str(key): safe_document(item) for key, item in value.items()
                if key not in {"artifact_path", "model_path", "source_declared", "source_runtime",
                               "reason", "error", "exception", "traceback"}}
    if isinstance(value, (tuple, list)):
        return [safe_document(item) for item in value]
    return value


def command(call: Callable, *, model=TaskDocumentResponse, transform=None, status_code=200):
    try:
        result = call()
    except LookupError:
        return api_not_found()
    except (ValidationError, ValueError):
        return api_validation_error("任务参数、模型格式或作用域不符合模板要求")
    except Exception as exc:
        # Domain exceptions are mapped without leaking filesystem or SDK text.
        name = type(exc).__name__
        code = getattr(exc, "code", None)
        if code is None and isinstance(exc, RuntimeError) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,100}", str(exc)):
            code = str(exc)
        logger.warning("task_platform.command_failed", error_type=name, reason_code=code)
        if code in {"UNAVAILABLE", "LOAD_TIMEOUT", "RESOURCE_EXHAUSTED"}:
            return api_unavailable(f"任务执行不可用：{code}")
        if (name.endswith(("Conflict", "IntegrityError")) or isinstance(exc, RuntimeError)):
            return api_conflict(f"任务条件未满足：{code}" if code else "任务状态或版本冲突，请刷新后检查",
                                {"reason_code": code or name})
        return api_internal_error("任务操作失败；请检查诊断后使用同一幂等键重试")
    try:
        payload = transform(result) if transform else {"document": safe_document(result)}
        return api_typed_success(payload, model, status_code=status_code)
    except Exception as exc:
        # A write may already be committed. Response validation failures are
        # server failures, never invalid client input or a reason to mint a new
        # command key. The persisted receipt makes the original command retryable.
        logger.error("task_platform.response_failed", error_type=type(exc).__name__)
        return api_internal_error("操作结果未能返回；请使用同一幂等键重试对账")


def _configured_camera(request: Request, camera_id: str) -> bool:
    return any(getattr(camera, "camera_id", None) == camera_id
               for camera in getattr(getattr(request.app.state, "config", None), "cameras", []))


def _mutate_stopped_task(request: Request, service, action: str, task_id: str,
                         body, actor: str, idempotency_key: str):
    """Reconcile committed commands, otherwise serialize writes with start/stop."""
    arguments = {**body.model_dump(), "requested_by": actor, "idempotency_key": idempotency_key}

    def receipt():
        return service.get_task_command_receipt(f"{action}_task", task_id, **arguments)

    committed = receipt()
    if committed is not None:
        return committed

    def write():
        return getattr(service, f"{action}_task")(task_id, **arguments)

    executor = getattr(request.app.state, "task_execution", None)
    if executor is None:
        return write()
    gate = getattr(executor, "mutate_if_stopped", None)
    if not callable(gate):
        raise TaskExecutionError("UNAVAILABLE", "task mutation coordinator unavailable")
    try:
        return gate(task_id, write)
    except TaskExecutionError as exc:
        if exc.code == "BUSY":
            # A parallel identical command can commit between our read and
            # control admission. Reconcile it without bypassing any new write.
            committed = receipt()
            if committed is not None:
                return committed
        raise


@router.post("", response_model=ApiEnvelope[AnalysisTaskResponse])
def create_task(request: Request, body: TaskCreateRequest, idempotency_key: CommandKey):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    if not _configured_camera(request, body.camera_id):
        return api_not_found("摄像头不存在")
    return command(lambda: service.create_task(**body.model_dump(), requested_by=actor,
                                               idempotency_key=idempotency_key),
                   model=AnalysisTaskResponse, transform=lambda task: _public_task(request, task), status_code=201)


@router.put("/{task_id}", response_model=ApiEnvelope[AnalysisTaskResponse])
def update_task(request: Request, task_id: str, body: TaskUpdateRequest, idempotency_key: CommandKey):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    return command(lambda: _mutate_stopped_task(request, service, "update", task_id,
                                               body, actor, idempotency_key),
                   model=AnalysisTaskResponse, transform=lambda task: _public_task(request, task))


@router.post("/{task_id}/archive", response_model=ApiEnvelope[AnalysisTaskResponse])
def archive_task(request: Request, task_id: str, body: TaskRevisionRequest, idempotency_key: CommandKey):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    return command(lambda: _mutate_stopped_task(request, service, "archive", task_id,
                                               body, actor, idempotency_key),
                   model=AnalysisTaskResponse, transform=lambda task: _public_task(request, task))


@models_router.get("", response_model=ApiEnvelope[TaskDocumentsResponse])
def list_task_models(request: Request):
    if not require_permission(request, "read_models"):
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    return command(service.list_models, model=TaskDocumentsResponse,
                   transform=lambda result: {"documents": safe_document(result)})


@models_router.post("", response_model=ApiEnvelope[TaskDocumentResponse])
def register_task_model(request: Request, body: TaskModelRegisterRequest, idempotency_key: CommandKey):
    actor = _actor(request, "manage_models", human=True)
    if actor is None:
        return api_forbidden("模型登记需要已登录的人工管理身份")
    service = _service(request)
    if service is None:
        return api_unavailable()
    if body.scope_camera_id is not None and not _configured_camera(request, body.scope_camera_id):
        return api_not_found("摄像头不存在")
    return command(lambda: service.register_task_model(**body.model_dump(), requested_by=actor,
                                                       idempotency_key=idempotency_key), status_code=201)


@router.get("/{task_id}/compatible-models", response_model=ApiEnvelope[TaskDocumentsResponse])
def compatible_models(request: Request, task_id: str):
    if not require_permission(request, "read_models"):
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    return command(lambda: service.list_compatible_models(task_id), model=TaskDocumentsResponse,
                   transform=lambda result: {"documents": safe_document(result)})


@router.get("/{task_id}/bundles", response_model=ApiEnvelope[TaskDocumentsResponse])
def bundles(request: Request, task_id: str):
    if not require_permission(request, "read_models"):
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    return command(lambda: service.list_bundles(task_id), model=TaskDocumentsResponse,
                   transform=lambda result: {"documents": safe_document(result)})


@router.get("/{task_id}/model-compatibility", response_model=ApiEnvelope[TaskDocumentsResponse])
def model_compatibility(request: Request, task_id: str):
    if not require_permission(request, "read_models"):
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    return command(lambda: service.list_model_compatibility(task_id), model=TaskDocumentsResponse,
                   transform=lambda result: {"documents": safe_document(result)})


@router.post("/{task_id}/bundles", response_model=ApiEnvelope[TaskDocumentResponse])
def freeze_bundle(request: Request, task_id: str, body: TaskBundleCreateRequest, idempotency_key: CommandKey):
    actor = _actor(request, "manage_models")
    if actor is None:
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    return command(lambda: service.freeze_bundle(task_id, **body.model_dump(), requested_by=actor,
                                                 idempotency_key=idempotency_key), status_code=201)


@router.post("/{task_id}/runtime/start", response_model=ApiEnvelope[TaskDocumentResponse])
def start_task(request: Request, task_id: str, body: TaskStartRequest):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    # An unpersisted start would bypass desired state and be lost on restart.
    # Keep the old URL explicit rather than silently guessing a CAS revision.
    return api_conflict("请使用任务部署接口，提交部署版本与幂等键", {"reason_code": "USE_DURABLE_TASK_DEPLOYMENT"})


@router.post("/{task_id}/runtime/stop", response_model=ApiEnvelope[TaskDocumentResponse])
def stop_task(request: Request, task_id: str):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    # A runtime-only stop leaves desired=running and would restart unexpectedly.
    return api_conflict("请使用任务部署接口持久化停止意图", {"reason_code": "USE_DURABLE_TASK_DEPLOYMENT"})


@router.get("/{task_id}/results", response_model=ApiEnvelope[TaskResultsResponse])
def task_results(request: Request, task_id: str):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    if service.get_task(task_id) is None:
        return api_not_found()
    executor = getattr(request.app.state, "task_execution", None)
    if executor is None:
        return api_unavailable("独立任务执行器不可用")
    return command(lambda: executor.results(task_id, limit=20), model=TaskResultsResponse,
                   transform=lambda result: {"task_id": task_id, "mode": "shadow", "results": safe_document(result)})


@router.get("/{task_id}/release-readiness", response_model=ApiEnvelope[TaskReleaseReadinessResponse])
def release_readiness(request: Request, task_id: str):
    if not require_permission(request, "read_models"):
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable()
    task = service.get_task(task_id)
    if task is None:
        return api_not_found()
    # Legacy must continue through its existing full readiness API. Never
    # reinterpret a managed bundle or successful shadow run as release proof.
    return api_typed_success({
        "task_id": task_id, "authority": task["authority"],
        "production_supported": False, "production_ready": False,
        "reasons": (["USE_EXISTING_LEGACY_MODEL_RELEASE_SERVICE"] if task["authority"] == "legacy"
                    else ["TASK_SPECIFIC_EVALUATION_POLICY_REQUIRED", "TRUSTED_RELEASE_EVIDENCE_REQUIRED",
                          "SHADOW_EXECUTION_IS_NOT_PRODUCTION_APPROVAL"]),
    }, TaskReleaseReadinessResponse)
