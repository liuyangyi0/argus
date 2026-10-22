"""Typed request and response contracts for the public dashboard API.

The ASGI envelope middleware guarantees the wire-level ``{code,msg,data}``
shape.  These models describe the payload inside ``data`` and give FastAPI a
stable OpenAPI contract for the operational camera, alert and replay APIs.

Media responses (JPEG, MP4, MJPEG and downloads) intentionally do not use
these models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)


PipelineModeValue = Literal[
    "active",
    "maintenance",
    "learning",
    "collection",
    "training",
]

AlertWorkflowValue = Literal[
    "acknowledged",
    "confirmed_anomaly",
    "investigating",
    "resolved",
    "closed",
    "false_positive",
    "uncertain",
]

UserRoleValue = Literal["viewer", "operator", "engineer", "admin"]


class ApiContractModel(BaseModel):
    """Closed payload model used by public API contracts."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# User management contracts


class UserCreateRequest(ApiContractModel):
    username: StrictStr = Field(min_length=1, max_length=50)
    password: StrictStr = Field(min_length=6, max_length=1024)
    role: UserRoleValue = "viewer"
    display_name: StrictStr | None = Field(default=None, max_length=100)
    active: StrictBool = True


class UserUpdateRequest(ApiContractModel):
    display_name: StrictStr | None = Field(default=None, max_length=100)
    role: UserRoleValue | None = None
    password: StrictStr | None = Field(default=None, max_length=1024)
    active: StrictBool | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_explicit_nulls(cls, value):
        """Omission means no change; null must not silently become a value."""
        if isinstance(value, dict):
            for field_name in ("role", "password", "active"):
                if field_name in value and value[field_name] is None:
                    raise ValueError(f"{field_name} cannot be null")
        return value


# ---------------------------------------------------------------------------
# Model mutation contracts


class ModelBatchInferenceRequest(ApiContractModel):
    camera_id: StrictStr = Field(min_length=1, max_length=256)
    image_paths: list[StrictStr] = Field(min_length=1, max_length=100)


class ReleaseEvidenceSourcesRequest(ApiContractModel):
    regression_report_path: StrictStr = Field(min_length=1, max_length=4096)
    config_path: StrictStr = Field(min_length=1, max_length=4096)
    scene_profile_path: StrictStr = Field(min_length=1, max_length=4096)
    baseline_path: StrictStr = Field(min_length=1, max_length=4096)
    corpus_path: StrictStr = Field(min_length=1, max_length=4096)
    soak_report_path: StrictStr = Field(min_length=1, max_length=4096)
    soak_telemetry_path: StrictStr = Field(min_length=1, max_length=4096)


class ModelPromoteRequest(ApiContractModel):
    target_stage: Literal["shadow", "canary", "production", "retired"]
    reason: StrictStr | None = Field(default=None, max_length=4000)
    canary_camera_id: StrictStr | None = Field(default=None, max_length=256)
    release_evidence_sources: ReleaseEvidenceSourcesRequest | None = None
    expected_revision: StrictInt | None = Field(default=None, ge=0)
    idempotency_key: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    # Accepted only so the route can return the existing explicit safety error
    # instead of silently ignoring a client-authored evidence bundle.
    release_evidence: dict[str, JsonValue] | None = None
    # Legacy audit/bypass inputs are parsed but never trusted; authenticated
    # request identity and server policy remain authoritative.
    triggered_by: StrictStr | None = Field(default=None, max_length=256)
    allow_bypass: StrictBool | None = None


class ModelRetireRequest(ApiContractModel):
    reason: StrictStr = Field(default="manual retirement", max_length=4000)
    expected_revision: StrictInt | None = Field(default=None, ge=0)
    idempotency_key: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    triggered_by: StrictStr | None = Field(default=None, max_length=256)


class ModelReexportRequest(ApiContractModel):
    export_format: Literal["openvino", "onnx", "torch"] = "openvino"
    quantization: Literal["fp16", "fp32", "int8"] = "fp16"


class ModelAbLiveCompareRequest(ApiContractModel):
    camera_id: StrictStr = Field(min_length=1, max_length=256)
    shadow_version_id: StrictStr = Field(min_length=1, max_length=512)


class BackboneUpgradeRequest(ApiContractModel):
    backbone_path: StrictStr = Field(min_length=1, max_length=4096)
    version: StrictStr = Field(min_length=1, max_length=256)
    triggered_by: StrictStr | None = Field(default=None, max_length=256)


class ReleaseReadinessGate(ApiContractModel):
    """One independently visible, server-evaluated production safety gate."""

    key: str
    label: str
    passed: bool
    blocking: bool
    reason: str | None = None


class ReleaseReadinessResponse(ApiContractModel):
    """Authoritative fail-closed view of production release readiness."""

    model_version_id: str
    camera_id: str
    target_stage: Literal["production"]
    status: Literal["ready", "no_go", "pending", "applied", "failed"]
    production_ready: bool
    promotion_allowed: bool
    operation_id: str | None
    desired_stage: str | None
    desired_status: Literal["pending", "applied", "failed"] | None
    applied_stage: str
    applied_revision: int = Field(ge=0)
    runtime_state: dict[str, JsonValue] | None
    gates: list[ReleaseReadinessGate]
    reasons: list[str]
    checked_at: datetime


# ---------------------------------------------------------------------------
# Camera contracts


class CameraModeRequest(ApiContractModel):
    mode: PipelineModeValue


class CameraModeResponse(ApiContractModel):
    camera_id: str
    previous_mode: PipelineModeValue | None
    pipeline_mode: PipelineModeValue


class CameraStatsResponse(ApiContractModel):
    frames_captured: int
    frames_analyzed: int
    anomalies_detected: int
    alerts_emitted: int
    avg_latency_ms: float
    frames_dropped: int | None = None


class CameraLifecycleResponse(ApiContractModel):
    """Canonical payload returned after camera start/stop.

    A manager which has not materialised a runtime status yet returns only
    ``camera_id``; every other field is therefore explicitly optional.
    """

    camera_id: str
    name: str | None = None
    connected: bool | None = None
    running: bool | None = None
    pipeline_mode: PipelineModeValue | None = None
    health: dict[str, JsonValue] | None = None
    capture_status: dict[str, JsonValue] | None = None
    stats: CameraStatsResponse | None = None


class CameraSummaryResponse(ApiContractModel):
    camera_id: str
    name: str
    region_id: int | None
    region_name: str | None
    region_owner: str | None
    region_phone: str | None
    connected: bool
    running: bool
    pipeline_mode: PipelineModeValue | None
    health: dict[str, JsonValue] | None
    capture_status: dict[str, JsonValue] | None
    stats: CameraStatsResponse | None


class CameraListResponse(ApiContractModel):
    cameras: list[CameraSummaryResponse]


class SceneGuardRecoveryProgress(ApiContractModel):
    current: int = Field(ge=0)
    required: int = Field(ge=0)
    complete: bool | None = None


class SceneGuardMetrics(ApiContractModel):
    reference_available: bool
    reference_keypoints: int | None = None
    frame_keypoints: int | None = None
    raw_matches: int | None = None
    good_matches: int | None = None
    inlier_count: int | None = None
    inlier_ratio: float | None = None
    reference_coverage_ratio: float | None = None
    frame_coverage_ratio: float | None = None
    coverage_ratio: float | None = None
    reprojection_error_px: float | None = None
    overlap_ratio: float | None = None
    shift_px: float | None = None
    max_corner_displacement_px: float | None = None
    rotation_deg: float | None = None
    scale: float | None = None
    scale_delta: float | None = None
    anisotropy: float | None = None
    perspective: float | None = None
    brightness: float | None = None
    contrast_std: float | None = None
    blur_variance: float | None = None
    transform_reasonable: bool | None = None
    estimator: str | None = None


class SceneGuardStatusResponse(ApiContractModel):
    """Public fail-closed camera/scene compatibility state."""

    enabled: bool
    usable: bool
    state: str
    raw_state: str | None = None
    reason: str | None
    profile_id: str | None
    bound_model_version: str | None
    bound_model_artifact_sha256: str | None
    runtime_model_artifact_sha256: str | None
    reference_source: str | None
    reference_sha256: str | None
    reference_error: str | None
    runtime_error: str | None
    check_interval_frames: int | None = None
    check_interval_seconds: float | None = None
    last_checked_frame: int | None
    quick_match_score: float | None = None
    quick_shift_px: float | None = None
    reference_available: bool
    policy_recovery_count: int = Field(ge=0)
    required_recovery_count: int = Field(ge=0)
    recovery_progress: SceneGuardRecoveryProgress
    metrics: SceneGuardMetrics


class CameraLifecycleStageResponse(ApiContractModel):
    id: str
    name: str
    status: str
    info: str


class CameraRuntimeResponse(ApiContractModel):
    pipeline_mode: PipelineModeValue | None
    anomaly_locked: bool
    learning_progress: dict[str, JsonValue] | None
    capture_status: dict[str, JsonValue] | None
    scene_guard: SceneGuardStatusResponse | None


class CameraRunnerResponse(ApiContractModel):
    model_ref: str | None
    health_status: str
    cusum_state: dict[str, JsonValue] | str
    lock_state: str
    last_heartbeat: float
    version_tag: str
    degradation_state: str
    consecutive_failures: int
    latest_frame_identity: dict[str, JsonValue] | None = None


class CameraSourceDiagnosticsResponse(ApiContractModel):
    camera_id: str
    declared_source: str
    runtime_source: str
    go2rtc_stream: str | None
    last_frame_index: int | None
    last_capture_ts: float | None
    last_pipeline_ts: float | None
    last_snapshot_frame_index: int | None
    last_snapshot_ts: float | None
    last_snapshot_identity: dict[str, JsonValue] | None
    last_preview_stream_frame_index: int | None
    last_preview_stream_ts: float | None
    last_preview_stream_identity: dict[str, JsonValue] | None
    fps_observed: float | None
    dropped_frames: int
    source_consistency: str
    latest_frame_identity: dict[str, JsonValue] | None


class CameraDetailResponse(ApiContractModel):
    camera_id: str
    declared_source: str
    runtime_source: str
    go2rtc_stream: str | None
    last_frame_index: int | None
    last_capture_ts: float | None
    last_pipeline_ts: float | None
    last_snapshot_frame_index: int | None
    last_snapshot_ts: float | None
    last_preview_stream_frame_index: int | None
    last_preview_stream_ts: float | None
    fps_observed: float | None
    dropped_frames: int
    source_consistency: str
    source_diagnostics: CameraSourceDiagnosticsResponse
    name: str
    region_id: int | None
    region_name: str | None
    region_owner: str | None
    region_phone: str | None
    connected: bool
    running: bool
    model_backend: str | None
    expected_model_backend: str
    scene_guard: SceneGuardStatusResponse | None
    stats: CameraStatsResponse | None
    stages: list[CameraLifecycleStageResponse]
    config: dict[str, JsonValue]
    runtime: CameraRuntimeResponse
    runner: CameraRunnerResponse | None
    detector: dict[str, JsonValue] | None
    candidate_diagnostics: list[dict[str, JsonValue]]
    health: dict[str, JsonValue] | None


# ---------------------------------------------------------------------------
# Alert workflow contracts


class AlertWorkflowTransitionRequest(ApiContractModel):
    status: AlertWorkflowValue
    notes: StrictStr = Field(default="", max_length=4000)
    category: StrictStr = Field(default="", max_length=128)
    assigned_to: StrictStr = Field(default="", max_length=128)
    confirmed: StrictBool = False
    from_detail_view: StrictBool = False


class AlertWorkflowTransitionResponse(ApiContractModel):
    severity: str
    handling_policy: str
    require_confirmation: bool
    require_detail_view: bool
    next_actions: list[AlertWorkflowValue]


class AlertIdListRequest(ApiContractModel):
    alert_ids: list[StrictStr] = Field(default_factory=list, max_length=500)


class AlertMutationResponse(ApiContractModel):
    alert_id: str
    message: str


class AlertBulkMutationResponse(ApiContractModel):
    count: int = Field(ge=0)
    message: str


# ---------------------------------------------------------------------------
# Replay contracts


class ReplayPinnedFrameResponse(ApiContractModel):
    index: int = Field(ge=0)
    label: str
    type: str


class ReplayMetadataResponse(ApiContractModel):
    alert_id: str
    camera_id: str
    severity: str
    trigger_timestamp: float
    trigger_frame_index: int = Field(ge=0)
    start_timestamp: float
    end_timestamp: float
    frame_count: int = Field(ge=0)
    fps: int = Field(ge=0)
    linked_alert_id: str | None
    status: str
    video_file: str | None
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    codec: str
    created_at: str
    frame_identity: dict[str, JsonValue] | None = None
    pinned_frames: list[ReplayPinnedFrameResponse] = Field(default_factory=list)
    video_url: str


class ReplayOperatorActionResponse(ApiContractModel):
    timestamp: float
    user: str
    action: str


class ReplayKeyFrameResponse(ApiContractModel):
    index: int = Field(ge=0)
    type: str
    label: str


class ReplayTrajectoryPointResponse(ApiContractModel):
    frame_index: int = Field(ge=0)
    t: float
    x: float
    y: float


class ReplayPersonSignalResponse(ApiContractModel):
    timestamp: float
    count: int = Field(ge=0)
    bboxes: list[JsonValue]


class ReplayClipResponse(ApiContractModel):
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    label: str
    created_at: str


class ReplaySignalsResponse(ApiContractModel):
    timestamps: list[float]
    frame_numbers: list[int] = Field(default_factory=list)
    frame_identities: list[dict[str, JsonValue] | None] = Field(default_factory=list)
    anomaly_scores: list[float]
    simplex_scores: list[float | None]
    cusum_evidence: dict[str, list[float]]
    yolo_persons: list[ReplayPersonSignalResponse]
    yolo_boxes: list[list[JsonValue]] = Field(default_factory=list)
    candidate_records: list[list[dict[str, JsonValue]]] = Field(default_factory=list)
    candidate_events: list[dict[str, JsonValue]] = Field(default_factory=list)
    has_heatmaps: bool
    trajectory_points: dict[str, list[ReplayTrajectoryPointResponse]]
    operator_actions: list[ReplayOperatorActionResponse]
    key_frames: list[ReplayKeyFrameResponse]
    clips: list[ReplayClipResponse]


class ReplayPinFrameRequest(ApiContractModel):
    index: StrictInt = Field(ge=0)
    label: StrictStr = Field(default="", max_length=200)


class ReplayPinFrameResponse(ApiContractModel):
    success: bool


class ReplayClipRequest(ApiContractModel):
    start_index: StrictInt = Field(ge=0)
    end_index: StrictInt = Field(ge=0)
    label: StrictStr = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_range(self) -> "ReplayClipRequest":
        if self.end_index < self.start_index:
            raise ValueError("end_index cannot be less than start_index")
        return self


class ReplayClipMutationResponse(ApiContractModel):
    clip: ReplayClipResponse
    clips: list[ReplayClipResponse]


class ReplayClipListResponse(ApiContractModel):
    clips: list[ReplayClipResponse]


# ---------------------------------------------------------------------------
# Physics/trajectory replay contract


class TrajectoryMmPointResponse(ApiContractModel):
    x_mm: float
    y_mm: float
    z_mm: float


class TrajectoryFitPointResponse(TrajectoryMmPointResponse):
    x_px: float | None
    y_px: float | None


class TrajectoryFitResponse(ApiContractModel):
    track_id: int
    is_primary: bool
    model_type: str
    r_squared: float
    speed_ms: float | None
    speed_px_per_sec: float | None
    origin: TrajectoryFitPointResponse
    landing: TrajectoryFitPointResponse


class TrajectoryPrimaryResponse(ApiContractModel):
    speed_ms: float | None
    speed_px_per_sec: float | None
    trajectory_model: str
    origin: TrajectoryMmPointResponse | None
    landing: TrajectoryMmPointResponse | None


class TrajectoryClassificationResponse(ApiContractModel):
    label: str
    confidence: float | None


class AlertTrajectoryResponse(ApiContractModel):
    alert_id: str
    speed_ms: float | None
    speed_px_per_sec: float | None
    trajectory_model: str | None
    points: JsonValue | None
    origin: TrajectoryMmPointResponse | None
    landing: TrajectoryMmPointResponse | None
    primary: TrajectoryPrimaryResponse | None
    trajectories: list[TrajectoryFitResponse]
    classification: TrajectoryClassificationResponse | None
