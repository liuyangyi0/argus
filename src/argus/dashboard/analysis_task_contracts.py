"""Public, credential-free contracts for the P0–P2 analysis-task surface."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from argus.dashboard.api_contracts import ApiContractModel


RuntimeState = Literal[
    "unknown", "stopped", "starting", "running", "stopping", "failed", "blocked",
    "waiting_source", "sampling_insufficient", "cleanup_pending", "runtime_dead", "cleanup_failed", "draining",
]


class AnalysisTaskTemplateResponse(ApiContractModel):
    template_id: str
    name: str
    availability: Literal["legacy_compatibility", "shadow"]
    supports_creation: bool = False
    deployment_authority: Literal["legacy", "managed"] = "legacy"
    safety_contract: str
    default_config: dict[str, object] | None = None
    model_roles: dict[str, str] = Field(default_factory=dict)
    model_role_requirements: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


class AnalysisTaskTemplatesResponse(ApiContractModel):
    templates: list[AnalysisTaskTemplateResponse]


class AnalysisTaskRuntimeResponse(ApiContractModel):
    """Unknown is not stopped, and running is not proof of model readiness."""

    state: RuntimeState = "unknown"
    running: bool | None = None
    generation: int | None = Field(default=None, ge=0)
    reason_code: str | None = None
    model_version: str | None = None
    artifact_sha256: str | None = None
    model_identity_verified: bool | None = None
    applied_task_revision: int | None = Field(default=None, ge=1)
    bundle_id: str | None = None
    deployment_revision: int | None = Field(default=None, ge=0)
    deployment_operation_id: str | None = None
    mode: Literal["shadow"] | None = None
    event_error: Literal["EVENT_QUEUE_UNAVAILABLE", "EVENT_QUEUE_FAILED"] | None = None


class AnalysisTaskSourceResponse(ApiContractModel):
    state: RuntimeState = "unknown"
    running: bool | None = None
    connected: bool | None = None
    generation: int | None = Field(default=None, ge=0)
    reason_code: str | None = None


class AnalysisTaskDeploymentResponse(ApiContractModel):
    """Registry intent and verified runtime application remain separate facts."""

    authority: str
    desired_artifact_id: str | None = None
    applied_artifact_id: str | None = None
    registry_artifact_id: str | None = None
    registry_model_version_id: str | None = None
    registry_release_revision: int | None = Field(default=None, ge=0)
    registry_stage: str | None = None
    projection_status: Literal["not_synced", "unbound", "blocked", "authorized", "stale", "shadow_unconfigured"] = "not_synced"
    runtime_status: str = "unknown"
    observed_at: str | None = None
    desired_bundle_id: str | None = None
    applied_bundle_id: str | None = None
    previous_bundle_id: str | None = None
    desired_state: Literal["running", "stopped"] | None = None
    deployment_revision: int = Field(default=0, ge=0)
    applied_deployment_revision: int | None = Field(default=None, ge=0)
    current_operation_id: str | None = None
    failure_code: str | None = None


class TaskGuidanceResponse(ApiContractModel):
    """Navigation advice only; it cannot approve a model or start a worker."""

    schema_version: Literal["argus.task-guidance.v1"] = "argus.task-guidance.v1"
    workflow_kind: Literal["legacy_anomaly", "model_task", "algorithm"]
    training_requirement: Literal["legacy_workflow", "optional", "not_applicable"]
    is_authorization: Literal[False] = False
    production_scope: Literal["existing_legacy_gates", "unsupported"]
    state: Literal["unknown", "setup", "attention", "transitioning", "stopped", "observed_running", "archived"]
    reason_code: str
    title: str
    detail: str
    target: Literal["none", "source", "configuration", "models", "deployment", "events", "results",
                    "legacy_baseline", "legacy_runtime"]
    action_label: str


class AnalysisTaskResponse(ApiContractModel):
    task_id: str
    camera_id: str
    template_id: str
    authority: str
    name: str | None = None
    lifecycle: str = "active"
    config: dict[str, object] | None = None
    current_revision: int = Field(ge=1, description="Imported configuration snapshot revision; not proof of runtime application")
    config_sha256: str = Field(description="Imported configuration snapshot SHA-256; legacy configuration authority is unchanged")
    created_at: str | None = None
    deployment: AnalysisTaskDeploymentResponse | None = None
    runtime: AnalysisTaskRuntimeResponse = Field(default_factory=AnalysisTaskRuntimeResponse)
    source: AnalysisTaskSourceResponse = Field(default_factory=AnalysisTaskSourceResponse)
    guidance: TaskGuidanceResponse | None = None


class AnalysisTasksResponse(ApiContractModel):
    tasks: list[AnalysisTaskResponse]
