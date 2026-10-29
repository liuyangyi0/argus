"""Closed DTOs for release dossiers; review never confers production access."""

from typing import Literal

from pydantic import Field

from argus.dashboard.api_contracts import ApiContractModel


class TaskReleaseEvidenceSource(ApiContractModel):
    role: Literal["evaluation", "calibration", "baseline", "corpus", "soak", "soak_telemetry"]
    relative_path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern="^[0-9a-f]{64}$")


class TaskReleaseRequest(ApiContractModel):
    bundle_id: str = Field(pattern="^bundle-[0-9a-f]{32}$")
    expected_task_revision: int = Field(ge=1, strict=True)
    expected_deployment_revision: int = Field(ge=0, strict=True)
    evidence_sources: list[TaskReleaseEvidenceSource] = Field(default_factory=list, max_length=6)


class TaskReleaseEvidenceUpdateRequest(ApiContractModel):
    expected_revision: int = Field(ge=1, strict=True)
    evidence_sources: list[TaskReleaseEvidenceSource] = Field(max_length=6)


class TaskReleaseReviewRequest(ApiContractModel):
    expected_revision: int = Field(ge=1, strict=True)
    expected_subject_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    decision: Literal["approve", "reject"]
    notes: str = Field(default="", max_length=4000)


class TaskReleaseRevokeRequest(ApiContractModel):
    expected_revision: int = Field(ge=1, strict=True)
    notes: str = Field(min_length=1, max_length=4000)
