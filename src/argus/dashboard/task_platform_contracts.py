"""Typed management envelopes; JSON documents are validated by domain services."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from argus.dashboard.api_contracts import ApiContractModel


class TaskCreateRequest(ApiContractModel):
    camera_id: str = Field(min_length=1, max_length=50)
    template_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    config: dict[str, object] = Field(default_factory=dict)


class TaskUpdateRequest(ApiContractModel):
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    config: dict[str, object] | None = None


class TaskRevisionRequest(ApiContractModel):
    expected_revision: int = Field(ge=1)


class TaskModelRegisterRequest(ApiContractModel):
    relative_path: str = Field(min_length=1, max_length=500)
    descriptor: dict[str, object]
    scope_camera_id: str | None = Field(default=None, max_length=50)


class TaskBundleCreateRequest(TaskRevisionRequest):
    model_ids: dict[str, str] = Field(default_factory=dict)


class TaskStartRequest(ApiContractModel):
    bundle_id: str = Field(pattern="^bundle-[0-9a-f]{32}$")


class TaskDeploymentRequest(ApiContractModel):
    action: Literal["deploy", "stop", "rollback"]
    bundle_id: str | None = Field(default=None, pattern="^bundle-[0-9a-f]{32}$")
    expected_deployment_revision: int = Field(ge=0)


class TaskConfigurationRestoreRequest(TaskRevisionRequest):
    bundle_id: str = Field(pattern="^bundle-[0-9a-f]{32}$")


class TaskDocumentResponse(ApiContractModel):
    document: dict[str, object]


class TaskDocumentsResponse(ApiContractModel):
    documents: list[dict[str, object]]


class TaskResultsResponse(ApiContractModel):
    task_id: str
    mode: Literal["shadow"] = "shadow"
    results: list[dict[str, object]]


class TaskReleaseReadinessResponse(ApiContractModel):
    task_id: str
    production_supported: bool = False
    production_ready: bool = False
    reasons: list[str]
    authority: Literal["legacy", "managed"]


class TaskWorkflowCreateRequest(ApiContractModel):
    pass


class TaskCollectionRequest(TaskRevisionRequest):
    params: dict[str, object] = Field(default_factory=dict)


class TaskReviewRequest(TaskRevisionRequest):
    sample_id: str = Field(min_length=1, max_length=128)
    expected_image_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    decision: Literal["approved", "rejected", "needs_review"]
    notes: str = Field(default="", max_length=4000)
    annotation: dict[str, object] = Field(default_factory=dict)


class TaskReviewImportRequest(TaskRevisionRequest):
    payload: dict[str, object]


class TaskTrainingProposalRequest(TaskRevisionRequest):
    recipe_id: str = Field(min_length=1, max_length=128)


class TaskCandidateImportRequest(TaskRevisionRequest):
    candidate: dict[str, object]
    expected_package_sha256: str = Field(pattern="^[0-9a-f]{64}$")
