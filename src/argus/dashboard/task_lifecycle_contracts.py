"""HTTP-only contracts for task events, dossiers and training execution."""

from typing import Literal

from pydantic import Field, StrictBool, StrictStr

from argus.dashboard.api_contracts import ApiContractModel
from argus.dashboard.task_release_contracts import (
    TaskReleaseRequest as TaskReleaseDossierRequest,
    TaskReleaseEvidenceUpdateRequest as TaskReleaseEvidenceRequest,
    TaskReleaseReviewRequest,
    TaskReleaseRevokeRequest,
)

__all__ = [
    "TaskReleaseDossierRequest", "TaskReleaseEvidenceRequest", "TaskReleaseReviewRequest",
    "TaskReleaseRevokeRequest", "TaskEventReviewRequest", "TaskDetectionTrainingParameters",
    "TaskTrainingExecutionRequest", "TaskTrainingCancelRequest",
    "TaskTrainingCapabilitiesResponse", "TaskTrainingTrustedBaseOption",
    "TaskTrainingRuntimeCapabilities", "TaskTrainingResourceCapabilities",
]


class TaskEventReviewRequest(ApiContractModel):
    expected_revision: int = Field(ge=1, strict=True)
    status: Literal["acknowledged", "false_positive", "resolved"]
    notes: str | None = Field(default=None, max_length=4000)


class TaskDetectionTrainingParameters(ApiContractModel):
    epochs: int = Field(default=10, ge=1, le=1000, strict=True)
    imgsz: int = Field(default=640, ge=32, le=2048, multiple_of=32, strict=True)
    batch: int = Field(default=4, ge=1, le=64, strict=True)
    seed: int = Field(default=0, ge=0, le=2147483647, strict=True)


class TaskTrainingExecutionRequest(ApiContractModel):
    expected_revision: int = Field(ge=1, strict=True)
    expected_package_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    base_model_path: str = Field(min_length=1, max_length=500)
    base_model_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    params: TaskDetectionTrainingParameters = Field(default_factory=TaskDetectionTrainingParameters)


class TaskTrainingCancelRequest(ApiContractModel):
    expected_revision: int = Field(ge=1, strict=True)


class TaskTrainingTrustedBaseOption(ApiContractModel):
    model_id: StrictStr = Field(min_length=1, max_length=128)
    name: StrictStr = Field(min_length=1, max_length=200)
    relative_path: StrictStr = Field(min_length=1, max_length=500)
    artifact_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    configured_trust: Literal[True] = True


class TaskTrainingRuntimeCapabilities(ApiContractModel):
    state: Literal["disabled", "not_started", "running", "waiting_resources", "reconciling",
                   "owner_unavailable", "stopping", "stopped", "cleanup_unconfirmed", "error"]
    enabled: StrictBool
    error_code: StrictStr | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,100}$")


class TaskTrainingResourceCapabilities(ApiContractModel):
    state: Literal["disabled", "waiting_maintenance", "blocked", "ready", "leased"]
    reason_code: StrictStr | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,100}$")
    device: Literal["cuda:0"] = "cuda:0"
    free_memory_mb: int | None = Field(default=None, ge=0, le=2147483647, strict=True)
    memory_limit_enforced: Literal[False] = False


class TaskTrainingCapabilitiesResponse(ApiContractModel):
    schema: Literal["argus.task-training-capabilities.v1"]
    recipe_id: Literal["yolov8_detection.v1"]
    configured_enabled: StrictBool
    maintenance_approved: StrictBool
    resource_mode: Literal["exclusive_maintenance"]
    device: Literal["cuda:0"]
    minimum_free_memory_mb: int = Field(ge=256, le=1048576, strict=True)
    poll_interval_seconds: float = Field(ge=0.1, le=60.0, strict=True, allow_inf_nan=False)
    trusted_base_models: list[TaskTrainingTrustedBaseOption]
    runtime: TaskTrainingRuntimeCapabilities
    resource: TaskTrainingResourceCapabilities
    validation_only: Literal[True] = True
    production_allowed: Literal[False] = False
