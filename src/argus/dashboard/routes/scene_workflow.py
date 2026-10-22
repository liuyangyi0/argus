"""Typed scene/baseline/training bridge; never a second model publisher."""

from __future__ import annotations

import threading
import json
from pathlib import Path
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from argus.dashboard.api_response import (
    ApiEnvelope,
    api_conflict,
    api_forbidden,
    api_not_found,
    api_success,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.auth import current_username, require_permission
from argus.dashboard.model_runtime import get_registry
from argus.storage.scene_workflow import SceneWorkflowConflict, SceneWorkflowService

router = APIRouter()
logger = structlog.get_logger()
_SERVICE_LOCK = threading.Lock()
Identifier = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$")]
Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class WorkflowContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator(
        "normal_reference_confirmed", "operator_approved", "acknowledge_runtime_unchanged",
        mode="before", check_fields=False,
    )
    @classmethod
    def literal_confirmation(cls, value: object) -> bool:
        if value is not True:
            raise ValueError("confirmation must be the literal JSON boolean true")
        return True


class SceneCreateRequest(WorkflowContract):
    workflow_id: Identifier
    reference_path: StrictStr = Field(min_length=1, max_length=512)
    reference_sha256: Sha256
    normal_reference_confirmed: Literal[True]


class NormalSource(WorkflowContract):
    video: StrictStr = Field(min_length=1, max_length=512)
    annotations: StrictStr = Field(min_length=1, max_length=512)


class StrictBaselineRequest(WorkflowContract):
    sources: list[NormalSource] = Field(min_length=2, max_length=32)
    sample_every: StrictInt = Field(default=30, ge=1, le=10000)
    max_images: StrictInt = Field(default=2000, ge=150, le=20000)


class SceneTrainingRequest(WorkflowContract):
    model_type: Literal["patchcore", "efficient_ad", "dinomaly2"] = "patchcore"
    image_size: StrictInt = Field(default=256, ge=64, le=1024)


class ProfileApprovalRequest(WorkflowContract):
    operator_approved: Literal[True]


class ProfileSelectionRequest(WorkflowContract):
    acknowledge_runtime_unchanged: Literal[True]
    expected_active_profile_id: Annotated[
        StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"),
    ] | None
    expected_active_profile_revision: Sha256 | None


class WorkflowSummary(WorkflowContract):
    workflow_id: str
    profile_id: str
    state: str
    created_by: str
    created_at: str
    reference_sha256: str
    baseline_schema: str | None
    training_ready: bool
    train_count: int
    calibration_count: int
    baseline_task_id: str | None
    training_job_id: str | None
    training_status: str | None
    model_version_id: str | None
    approved: bool
    active: bool
    error: str | None
    blockers: list[str]


class ProfileSummary(WorkflowContract):
    profile_id: str
    approved: bool
    active: bool
    model_version_id: str | None


class ReferenceSummary(WorkflowContract):
    path: str
    sha256: str


class WorkflowIssue(WorkflowContract):
    workflow_id: str
    message: str


class WorkflowRequirements(WorkflowContract):
    baseline_schema: Literal["argus.scene-profile-baseline.v3"]
    min_sources: Literal[2]
    min_train_images: Literal[100]
    min_calibration_images: Literal[50]
    production_gate_unchanged: Literal[True]


class WorkflowSnapshot(WorkflowContract):
    camera_id: str
    active_profile_revision: str | None
    workflows: list[WorkflowSummary]
    issues: list[WorkflowIssue]
    profiles: list[ProfileSummary]
    references: list[ReferenceSummary]
    source_root_label: str
    requirements: WorkflowRequirements


class TaskCreated(WorkflowContract):
    task_id: str


class TrainingCreated(WorkflowContract):
    job_id: str
    status: Literal["pending_confirmation"]


class ProfileApproved(WorkflowContract):
    profile_id: str


def _service(request: Request, camera_id: str, permission: str):
    if not require_permission(request, permission):
        return None, api_forbidden("权限不足")
    actor = current_username(request)
    if permission != "read_models" and (not actor.strip() or actor.strip().lower() == "unknown"):
        return None, api_forbidden("需要可审计的认证操作人身份")
    config = getattr(request.app.state, "config", None)
    camera = next((item for item in getattr(config, "cameras", []) if item.camera_id == camera_id), None)
    if camera is None:
        return None, api_not_found("摄像头不存在")
    database = getattr(request.app.state, "database", None) or getattr(request.app.state, "db", None)
    registry = get_registry(request)
    if database is None or registry is None:
        return None, api_unavailable("数据库或模型注册表不可用")
    key = (
        id(database), str(config.storage.baselines_dir), camera.scene_guard.profiles_dir,
        json.dumps(camera.scene_guard.model_dump(mode="json"), sort_keys=True, allow_nan=False),
    )
    with _SERVICE_LOCK:
        cache = getattr(request.app.state, "_scene_workflow_services", None)
        if cache is None:
            cache = {}
            request.app.state._scene_workflow_services = cache
        if key not in cache:
            cache[key] = SceneWorkflowService(
                baselines_dir=Path(config.storage.baselines_dir),
                profiles_dir=Path(camera.scene_guard.profiles_dir), database=database, registry=registry,
                scene_guard_policy=camera.scene_guard,
            )
        return cache[key], None


def _error(exc: Exception):
    if isinstance(exc, (SceneWorkflowConflict, FileExistsError)):
        return api_conflict(str(exc))
    if isinstance(exc, FileNotFoundError):
        return api_not_found("场景工作流或受控来源不存在")
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return api_validation_error(str(exc))
    raise exc


def _notify(request: Request, camera_id: str, workflow_id: str) -> None:
    manager = getattr(request.app.state, "ws_manager", None)
    if manager is None:
        return
    try:
        manager.broadcast(
            "models", {"type": "scene_workflow.changed", "camera_id": camera_id, "workflow_id": workflow_id},
            event_type="scene_workflow.changed", resource_id=f"{camera_id}:{workflow_id}",
        )
    except Exception:
        # The mutation is already durable; REST reconciliation remains the
        # fallback, not a misleading failed request that invites resubmission.
        logger.warning("scene_workflow.event_failed", camera_id=camera_id, workflow_id=workflow_id)


@router.get("/{camera_id}", response_model=ApiEnvelope[WorkflowSnapshot])
def scene_workflow_snapshot(request: Request, camera_id: Identifier):
    try:
        service, error = _service(request, camera_id, "read_models")
        if error is not None:
            return error
        return api_success(service.snapshot(camera_id))
    except Exception as exc:
        return _error(exc)


@router.post("/{camera_id}/workflows", response_model=ApiEnvelope[WorkflowSummary], status_code=201)
def create_scene_workflow(request: Request, camera_id: Identifier, body: SceneCreateRequest):
    try:
        service, error = _service(request, camera_id, "manage_baselines")
        if error is not None:
            return error
        created = service.create(camera_id, actor=current_username(request), **body.model_dump())
        _notify(request, camera_id, body.workflow_id)
        return api_success(created, status_code=201)
    except Exception as exc:
        return _error(exc)


@router.post("/{camera_id}/workflows/{workflow_id}/baseline-jobs", response_model=ApiEnvelope[TaskCreated], status_code=202)
def create_scene_baseline(request: Request, camera_id: Identifier, workflow_id: Identifier, body: StrictBaselineRequest):
    try:
        service, error = _service(request, camera_id, "manage_baselines")
        if error is not None:
            return error
        tasks = getattr(request.app.state, "task_manager", None)
        if tasks is None:
            return api_unavailable("后台任务服务不可用")
        task_id = service.start_baseline(
            camera_id, workflow_id, actor=current_username(request), task_manager=tasks, **body.model_dump(),
        )
        _notify(request, camera_id, workflow_id)
        return api_success({"task_id": task_id}, status_code=202)
    except Exception as exc:
        return _error(exc)


@router.post("/{camera_id}/workflows/{workflow_id}/training-jobs", response_model=ApiEnvelope[TrainingCreated], status_code=201)
def create_scene_training(request: Request, camera_id: Identifier, workflow_id: Identifier, body: SceneTrainingRequest):
    try:
        service, error = _service(request, camera_id, "manage_training")
        if error is not None:
            return error
        job_id = service.submit_training(camera_id, workflow_id, actor=current_username(request), **body.model_dump())
        _notify(request, camera_id, workflow_id)
        return api_success({"job_id": job_id, "status": "pending_confirmation"}, status_code=201)
    except Exception as exc:
        return _error(exc)


@router.post("/{camera_id}/workflows/{workflow_id}/approve-profile", response_model=ApiEnvelope[ProfileApproved])
def approve_scene_profile(request: Request, camera_id: Identifier, workflow_id: Identifier, body: ProfileApprovalRequest):
    try:
        service, error = _service(request, camera_id, "manage_models")
        if error is not None:
            return error
        profile_id = service.approve_profile(camera_id, workflow_id, actor=current_username(request))
        _notify(request, camera_id, workflow_id)
        return api_success({"profile_id": profile_id})
    except Exception as exc:
        return _error(exc)


@router.post("/{camera_id}/workflows/{workflow_id}/activate-profile", response_model=ApiEnvelope[ProfileApproved])
def select_scene_profile(request: Request, camera_id: Identifier, workflow_id: Identifier, body: ProfileSelectionRequest):
    try:
        service, error = _service(request, camera_id, "manage_models")
        if error is not None:
            return error
        profile_id = service.activate_profile(
            camera_id, workflow_id, actor=current_username(request),
            expected_active_profile_id=body.expected_active_profile_id,
            expected_active_profile_revision=body.expected_active_profile_revision,
        )
        _notify(request, camera_id, workflow_id)
        return api_success({"profile_id": profile_id})
    except Exception as exc:
        return _error(exc)
