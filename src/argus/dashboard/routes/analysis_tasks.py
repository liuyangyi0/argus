"""Read-only task identities; legacy release authority is deliberately unchanged."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from argus.core.analysis_task_contracts import template_specs
from argus.core.task_guidance import task_guidance
from argus.dashboard.analysis_task_contracts import (
    AnalysisTaskResponse,
    AnalysisTasksResponse,
    AnalysisTaskTemplatesResponse,
)
from argus.dashboard.api_response import (
    ApiEnvelope,
    api_forbidden,
    api_internal_error,
    api_not_found,
    api_typed_success,
    api_unavailable,
)
from argus.dashboard.auth import require_permission
from argus.storage.analysis_tasks import AnalysisTaskService

logger = structlog.get_logger()
router = APIRouter()
templates_router = APIRouter()


def _service(request: Request) -> AnalysisTaskService | None:
    service = getattr(request.app.state, "analysis_task_service", None)
    if service is not None:
        return service
    database = getattr(request.app.state, "database", None)
    return AnalysisTaskService(database) if database is not None else None


def _runtime_view(request: Request, task: dict) -> tuple[dict, dict]:
    """Use public task/source snapshots, never camera.running as task state."""
    manager = getattr(request.app.state, "camera_manager", None)
    task_getter = getattr(manager, "get_analysis_task_runtime", None)
    source_getter = getattr(manager, "get_source_runtime_status", None)

    def snapshot(getter, identity: str) -> dict:
        try:
            candidate = getter(identity) if callable(getter) else None
            return candidate if isinstance(candidate, dict) else {}
        except Exception:
            # One failing worker must not hide an independent source snapshot.
            # Do not publish SDK exceptions: they may contain URLs/local paths.
            logger.warning("analysis_tasks.runtime_snapshot_unavailable")
            return {}

    executor = getattr(request.app.state, "task_execution", None)
    authoritative_executor = task.get("authority") == "managed" and executor is not None
    if authoritative_executor:
        def task_getter(identity):
            candidate = executor.status(identity)
            # None from the authoritative owner means no execution exists.
            # Exceptions/invalid snapshots still remain unknown, not stopped.
            if candidate is None:
                return {"state": "stopped", "running": False}
            if not isinstance(candidate, dict) or candidate.get("task_id") != identity:
                return {}
            if candidate.get("state") == "stopped" and "running" not in candidate:
                # A retained, cleaned execution has this real status shape.
                # Never overwrite an explicit contradictory/unknown flag.
                return {**candidate, "running": False}
            return candidate
    task_status = snapshot(task_getter, task["task_id"])
    source_status = snapshot(source_getter, task["camera_id"])
    allowed_states = {
        "unknown", "stopped", "starting", "running", "stopping", "failed", "blocked",
        "waiting_source", "sampling_insufficient", "cleanup_pending", "runtime_dead", "cleanup_failed", "draining",
    }
    state_reasons = {
        "waiting_source": "WAITING_SOURCE",
        "sampling_insufficient": "SAMPLING_INSUFFICIENT",
        "cleanup_pending": "CLEANUP_PENDING",
        "runtime_dead": "RUNTIME_DEAD",
        "cleanup_failed": "CLEANUP_FAILED",
        "failed": "RUNTIME_FAILED",
        "draining": "DRAINING",
    }
    allowed_reasons = {
        *state_reasons.values(), "SCENE_BLOCKED", "MANUAL_STOP", "DEPLOYMENT_UNAVAILABLE",
        "STALE_TASK", "SOURCE_CHANGED", "CANCELLED", "RESOURCE_EXHAUSTED", "LOAD_TIMEOUT", "INFER_TIMEOUT", "UNAVAILABLE", "INVALID_OUTPUT",
        "WORKER_FAILED", "INVALID_BUNDLE", "FRAME_TOO_LARGE", "SAMPLING_INSUFFICIENT",
        "LEGACY_NOT_STARTED", "LEGACY_INITIALIZATION_FAILED",
    }

    def public_status(status: dict, *, source: bool = False) -> dict:
        state = status.get("state")
        result: dict = {"state": state if isinstance(state, str) and state in allowed_states else "unknown"}
        for key in (("running", "connected") if source else ("running",)):
            if type(status.get(key)) is bool:
                result[key] = status[key]
        generation = status.get("generation")
        if type(generation) is int and generation >= 0:
            result["generation"] = generation
        reason = status.get("reason_code") or status.get("error_code")
        result["reason_code"] = (
            reason if isinstance(reason, str) and reason in allowed_reasons
            else state_reasons.get(result["state"])
        )
        if not source:
            for key in ("model_version", "artifact_sha256"):
                if isinstance(status.get(key), str):
                    result[key] = status[key]
            if type(status.get("model_identity_verified")) is bool:
                result["model_identity_verified"] = status["model_identity_verified"]
            if type(status.get("applied_task_revision")) is int and status["applied_task_revision"] >= 1:
                result["applied_task_revision"] = status["applied_task_revision"]
            if isinstance(status.get("bundle_id"), str):
                result["bundle_id"] = status["bundle_id"]
            if type(status.get("deployment_revision")) is int and status["deployment_revision"] >= 0:
                result["deployment_revision"] = status["deployment_revision"]
            if isinstance(status.get("deployment_operation_id"), str):
                result["deployment_operation_id"] = status["deployment_operation_id"]
            # Real isolated execution snapshots omit mode. Only that owner,
            # with an explicit shadow-only flag, may fill the absent field.
            # An unknown/contradictory mode or production flag stays unknown.
            no_production = status.get("production_allowed", False) is False
            no_alerts = status.get("alerts_emitted", False) is False
            if no_production and no_alerts and (status.get("mode") == "shadow" or (
                authoritative_executor and "mode" not in status
                and status.get("production_allowed") is False
                and isinstance(status.get("bundle_id"), str) and bool(status["bundle_id"])
            )):
                result["mode"] = "shadow"
            event_error = status.get("event_error")
            if isinstance(event_error, str) and event_error in {"EVENT_QUEUE_UNAVAILABLE", "EVENT_QUEUE_FAILED"}:
                result["event_error"] = event_error
        return result

    return public_status(task_status), public_status(source_status, source=True)


def _public_task(request: Request, task: dict) -> dict:
    """Whitelist nested fields; task config may contain credentials/paths."""
    runtime, source = _runtime_view(request, task)
    payload = {
        key: task.get(key)
        for key in (
            "task_id", "camera_id", "template_id", "authority", "current_revision",
            "config_sha256", "created_at", "name",
        )
    }
    payload["lifecycle"] = "archived" if task.get("archived") else task.get("lifecycle", "active")
    payload["config"] = task.get("config") if task.get("authority") == "managed" else None
    deployment = task.get("deployment")
    payload["deployment"] = None
    if isinstance(deployment, dict):
        payload["deployment"] = {
            key: deployment[key]
            for key in (
                "authority", "desired_artifact_id", "applied_artifact_id",
                "registry_artifact_id", "registry_model_version_id",
                "registry_release_revision", "registry_stage", "projection_status",
                "runtime_status", "observed_at",
                "desired_bundle_id", "applied_bundle_id", "previous_bundle_id", "desired_state",
                "deployment_revision", "applied_deployment_revision", "current_operation_id", "failure_code",
            )
            if key in deployment
        }
    payload.update(runtime=runtime, source=source)
    payload["guidance"] = task_guidance(payload)
    return payload


@templates_router.get("", response_model=ApiEnvelope[AnalysisTaskTemplatesResponse])
def analysis_task_templates(request: Request):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    keys = {"template_id", "name", "availability", "supports_creation", "deployment_authority",
            "safety_contract", "default_config", "model_roles", "model_role_requirements"}
    return api_typed_success({"templates": [
        {key: value for key, value in item.items() if key in keys}
        for item in template_specs()
    ]}, AnalysisTaskTemplatesResponse)


@router.get("", response_model=ApiEnvelope[AnalysisTasksResponse])
def analysis_tasks_list(request: Request, camera_id: str | None = None):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable("分析任务存储不可用")
    try:
        tasks = service.list_tasks(camera_id=camera_id)
        bootstrap_errors = getattr(request.app.state, "analysis_task_bootstrap_errors", set())
        failed_cameras = set(bootstrap_errors) if isinstance(bootstrap_errors, (set, dict)) else set()
        if camera_id is not None:
            failed_cameras.intersection_update({camera_id})
        present_cameras = {task.get("camera_id") for task in tasks}
        if failed_cameras - present_cameras:
            # Do not turn a failed identity migration into a successful empty
            # list. A later manager bootstrap repairs this naturally on read;
            # this endpoint never retries or writes a migration itself.
            return api_unavailable("部分摄像头分析任务登记未完成，请检查启动诊断")
        return api_typed_success({
            "tasks": [_public_task(request, task) for task in tasks],
        }, AnalysisTasksResponse)
    except Exception:
        logger.warning("analysis_tasks.list_failed")
        return api_internal_error("无法读取分析任务")


@router.get("/{task_id}", response_model=ApiEnvelope[AnalysisTaskResponse])
def analysis_task_detail(request: Request, task_id: str):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    service = _service(request)
    if service is None:
        return api_unavailable("分析任务存储不可用")
    try:
        task = service.get_task(task_id)
        if task is None:
            return api_not_found("分析任务不存在")
        return api_typed_success(_public_task(request, task), AnalysisTaskResponse)
    except Exception:
        logger.warning("analysis_tasks.detail_failed")
        return api_internal_error("无法读取分析任务")
