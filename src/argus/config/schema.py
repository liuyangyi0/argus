"""Pydantic configuration models for Argus with industrial-grade validation."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ZonePriority(str, Enum):
    CRITICAL = "critical"
    STANDARD = "standard"
    LOW_PRIORITY = "low_priority"


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ZoneConfig(BaseModel):
    """A region of interest within a camera's view."""

    zone_id: str
    name: str
    polygon: list[tuple[int, int]] = Field(
        description="ROI polygon vertices in pixel coordinates"
    )
    zone_type: Literal["include", "exclude"] = "include"
    priority: ZonePriority = ZonePriority.STANDARD
    anomaly_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class MOG2Config(BaseModel):
    """MOG2 background subtractor parameters."""

    history: int = Field(default=500, ge=10, le=5000)
    var_threshold: float = Field(default=25.0, ge=1.0, le=500.0)
    detect_shadows: bool = True
    change_pct_threshold: float = Field(
        default=0.005, ge=0.0001, le=0.5,
        description="Fraction of pixels that must change (0.0001-0.5)",
    )
    denoise: bool = True
    enable_stabilization: bool = Field(
        default=True,
        description="Phase correlation image alignment to compensate camera micro-vibration",
    )
    heartbeat_frames: int = Field(
        default=150, ge=10, le=3000,
        description="Force full detection every N frames (10-3000)",
    )
    lock_score_threshold: float = Field(
        default=0.85, ge=0.5, le=0.99,
        description="Anomaly score threshold to engage region lock",
    )
    lock_clear_frames: int = Field(
        default=10, ge=1, le=100,
        description="Consecutive normal frames needed to auto-clear lock",
    )


class CameraHealthConfig(BaseModel):
    """Camera hardware health monitoring."""

    enabled: bool = Field(default=True)
    freeze_detection: bool = Field(default=True)
    lens_contamination_detection: bool = Field(default=True)
    displacement_detection: bool = Field(default=True)
    flash_suppression: bool = Field(default=True)
    gain_drift_detection: bool = Field(default=True)
    freeze_window_frames: int = Field(default=10, ge=5, le=30)
    sharpness_drop_pct: float = Field(default=0.3, ge=0.1, le=0.8)
    displacement_threshold_px: float = Field(default=20.0, ge=5.0, le=100.0)
    flash_sigma: float = Field(default=3.0, ge=2.0, le=5.0)
    gain_drift_threshold_pct: float = Field(default=20.0, ge=5.0, le=50.0)


class SimplexConfig(BaseModel):
    """Simplex safety channel: formally verifiable frame-difference detector."""

    enabled: bool = Field(default=True, description="Enable simplex parallel detection channel")
    diff_threshold: int = Field(default=30, ge=10, le=100)
    min_area_px: int = Field(default=500, ge=100, le=50000)
    min_static_seconds: float = Field(default=30.0, ge=5.0, le=600.0)
    morph_kernel_size: int = Field(default=5, ge=3, le=15)
    match_radius_px: int = Field(default=50, ge=10, le=200)


class PersonFilterConfig(BaseModel):
    """YOLO object detection parameters (YOLO-001/002/003).

    Despite the name (kept for backward compatibility), this configures
    multi-class COCO detection with optional tracking.
    """

    confidence: float = Field(default=0.4, ge=0.1, le=0.95)
    skip_frame_on_person: bool = False
    model_name: str = "yolo11n.pt"
    classes_to_detect: list[int] = Field(
        default=[0],
        description="COCO class IDs to detect. [0]=person only.",
    )
    enable_tracking: bool = Field(
        default=False,
        description="Enable BoT-SORT tracking for persistent object IDs across frames.",
    )


class CaptureQualityConfig(BaseModel):
    """Frame quality filters for baseline capture."""

    enabled: bool = True
    blur_threshold: float = Field(
        default=100.0, ge=10.0, le=1000.0,
        description="Laplacian variance below this is considered blurry",
    )
    blur_adaptive_pct: float = Field(
        default=0.3, ge=0.1, le=0.9,
        description="Adaptive blur floor = median(accepted scores) * this",
    )
    brightness_min: float = Field(default=30.0, ge=0.0, le=128.0)
    brightness_max: float = Field(default=225.0, ge=128.0, le=255.0)
    brightness_std_min: float = Field(
        default=10.0, ge=0.0, le=50.0,
        description="Minimum grayscale standard deviation (rejects flat frames)",
    )
    saturated_pixel_max_pct: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Max fraction of pixels < 5 or > 250",
    )
    ssim_dedup_threshold: float = Field(
        default=0.98, ge=0.90, le=1.0,
        description="SSIM >= this vs previous accepted frame → duplicate",
    )
    ssim_resize: int = Field(
        default=256, ge=64, le=512,
        description="Resize frames to NxN before SSIM computation",
    )
    person_confidence: float = Field(
        default=0.3, ge=0.1, le=0.9,
        description="YOLO person detection confidence for baseline filtering",
    )
    entropy_min: float = Field(
        default=3.0, ge=0.0, le=8.0,
        description="Shannon entropy below this = encoder error frame",
    )


class AnomalyConfig(BaseModel):
    """Anomalib anomaly detection parameters."""

    model_type: Literal["patchcore", "efficient_ad", "fastflow", "padim", "dinomaly2"] = "patchcore"
    threshold: float = Field(default=0.7, ge=0.1, le=0.99)
    image_size: tuple[int, int] = (256, 256)
    enable_multiscale: bool = Field(
        default=False,
        description="Enable sliding window multi-scale detection for small objects",
    )
    tile_size: int = Field(
        default=512, ge=256, le=1920,
        description="Tile size in pixels for sliding window (256-1920)",
    )
    tile_overlap: float = Field(
        default=0.25, ge=0.0, le=0.5,
        description="Overlap ratio between adjacent tiles (0.0-0.5)",
    )
    # Dinomaly2 parameters (only used when model_type="dinomaly2")
    dinomaly_backbone: str = Field(
        default="dinov2_vitb14",
        description="DINOv2 backbone variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14). "
        "Mapped to anomalib encoder names automatically (e.g. dinov2_vitb14 → dinov2reg_vit_base_14).",
    )
    dinomaly_encoder_layers: list[int] = Field(
        default=[2, 5, 8, 11],
        description="Intermediate ViT layers to extract features from",
    )
    dinomaly_few_shot_images: int = Field(
        default=8, ge=1, le=100,
        description="Minimum baseline images for few-shot mode (Dinomaly2 supports 8-shot)",
    )
    dinomaly_multi_class: bool = Field(
        default=False,
        description="Use unified multi-class model for all cameras (shared backbone)",
    )
    # INT8 quantization (B2)
    quantization: Literal["fp32", "fp16", "int8"] = Field(
        default="fp16",
        description="Model precision for inference (fp32=full, fp16=half, int8=quantized)",
    )
    quantization_calibration_images: int = Field(
        default=100, ge=50, le=1000,
        description="Number of baseline images used for INT8 calibration",
    )
    # Conformal prediction calibration
    enable_calibration: bool = Field(
        default=True,
        description="Apply conformal calibration to anomaly scores when calibration.json exists",
    )
    # SSIM fallback parameters (used when no trained model is available)
    ssim_baseline_frames: int = Field(
        default=15, ge=5, le=100,
        description="Number of frames to collect for SSIM baseline calibration",
    )
    ssim_sensitivity: float = Field(
        default=50.0, ge=1.0, le=200.0,
        description="Sigmoid sensitivity for SSIM score normalization",
    )
    ssim_midpoint: float = Field(
        default=0.015, ge=0.001, le=0.5,
        description="Sigmoid midpoint for SSIM score normalization",
    )

    # Release pipeline parameters
    shadow_sample_rate: int = Field(
        default=5, ge=1, le=100,
        description="Run shadow inference on every Nth frame (1 = every frame)",
    )
    warmup_frames: int = Field(
        default=10, ge=1, le=50,
        description="Number of baseline frames to run during model warmup verification",
    )
    warmup_max_latency_ms: float = Field(
        default=500.0, ge=50.0, le=5000.0,
        description="Maximum average inference latency (ms) allowed during warmup",
    )
    verify_model_signature: bool = Field(
        default=False,
        description="Require cryptographic signature verification on model files",
    )
    min_shadow_days: int = Field(
        default=3, ge=0, le=90,
        description="Minimum days in shadow stage before promotion to canary",
    )
    min_canary_days: int = Field(
        default=7, ge=0, le=90,
        description="Minimum days in canary stage before promotion to production",
    )


class ClassifierConfig(BaseModel):
    """Open vocabulary detection classifier (D1)."""

    enabled: bool = Field(default=False)
    model_name: str = Field(default="yolov8s-worldv2.pt")
    vocabulary: list[str] = Field(
        default_factory=lambda: [
            "wrench", "bolt", "nut", "screwdriver", "hammer",
            "rag", "glove", "plastic bag", "tape", "wire",
            "insulation", "debris", "paint chip",
            "insect", "bird", "shadow", "reflection",
        ]
    )
    min_anomaly_score_to_classify: float = Field(default=0.5, ge=0.0, le=1.0)
    high_risk_labels: list[str] = Field(
        default_factory=lambda: ["wrench", "bolt", "nut", "screwdriver", "hammer"]
    )
    low_risk_labels: list[str] = Field(
        default_factory=lambda: ["insect", "shadow", "reflection"]
    )


class SegmenterConfig(BaseModel):
    """SAM 2 instance segmentation (D2)."""

    enabled: bool = Field(default=False)
    model_size: str = Field(
        default="small",
        pattern="^(tiny|small|base_plus|large)$",
        description="SAM 2 model size: tiny, small, base_plus, large",
    )
    max_points: int = Field(
        default=5, ge=1, le=20,
        description="Max anomaly peaks to segment per frame",
    )
    min_anomaly_score: float = Field(
        default=0.7, ge=0.1, le=0.99,
        description="Minimum anomaly score to trigger segmentation",
    )
    min_mask_area_px: int = Field(
        default=100, ge=10, le=100000,
        description="Minimum mask area in pixels — smaller objects are discarded",
    )


class DriftConfig(BaseModel):
    """Score distribution drift monitoring via Kolmogorov-Smirnov test."""

    enabled: bool = Field(default=True)
    reference_window: int = Field(
        default=500, ge=100, le=5000,
        description="Number of initial scores to collect as reference distribution",
    )
    test_window: int = Field(
        default=100, ge=50, le=1000,
        description="Sliding window size for comparison against reference",
    )
    p_value_threshold: float = Field(
        default=0.01, ge=0.001, le=0.1,
        description="KS test significance level",
    )
    ks_threshold: float = Field(
        default=0.1, ge=0.01, le=0.5,
        description="KS statistic threshold for drift detection",
    )
    cooldown_minutes: int = Field(
        default=30, ge=5, le=1440,
        description="Minutes before re-alerting drift for same camera",
    )


class CameraConfig(BaseModel):
    """Configuration for a single camera."""

    camera_id: str
    name: str
    source: str  # RTSP URL, USB device index, or file path
    protocol: Literal["rtsp", "usb", "file"] = "rtsp"
    fps_target: int = Field(default=5, ge=1, le=30)
    resolution: tuple[int, int] = (1920, 1080)
    zones: list[ZoneConfig] = Field(default_factory=list)
    reconnect_delay: float = Field(default=5.0, ge=1.0, le=300.0)
    max_reconnect_attempts: int = Field(default=-1, ge=-1)
    watchdog_timeout: float = Field(
        default=30.0, ge=5.0, le=300.0,
        description="Seconds without frames before forced reconnect (5-300)",
    )
    mog2: MOG2Config = Field(default_factory=MOG2Config)
    person_filter: PersonFilterConfig = Field(default_factory=PersonFilterConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    simplex: SimplexConfig = Field(default_factory=SimplexConfig)
    health: CameraHealthConfig = Field(default_factory=CameraHealthConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)


class CameraGroupConfig(BaseModel):
    """A group of cameras sharing a combined baseline and model.

    Camera groups allow multiple cameras with similar views (e.g. corridor cameras)
    to share a single baseline set and trained model, reducing storage and update cost.
    """

    group_id: str = Field(description="Unique group identifier (e.g. CORRIDOR-A)")
    name: str = Field(description="Human-readable group name")
    camera_ids: list[str] = Field(min_length=1, description="Member camera IDs")
    zone_id: str = Field(default="default", description="Zone within each camera to use")

    @field_validator("group_id")
    @classmethod
    def validate_group_id(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z0-9_-]+$", v):
            raise ValueError(f"group_id must be alphanumeric with hyphens/underscores, got {v!r}")
        return v


class SeverityThresholds(BaseModel):
    """Anomaly score thresholds for each severity level.

    Must be ordered: info < low < medium < high.
    """

    info: float = Field(default=0.50, ge=0.1, le=0.99)
    low: float = Field(default=0.70, ge=0.1, le=0.99)
    medium: float = Field(default=0.85, ge=0.1, le=0.99)
    high: float = Field(default=0.95, ge=0.1, le=0.99)

    @model_validator(mode="after")
    def check_ordering(self) -> SeverityThresholds:
        if not (self.info < self.low < self.medium < self.high):
            raise ValueError(
                f"Severity thresholds must be ordered: "
                f"info({self.info}) < low({self.low}) < medium({self.medium}) < high({self.high})"
            )
        return self


class TemporalConfirmation(BaseModel):
    """Require anomaly persistence before alerting."""

    min_consecutive_frames: int = Field(default=3, ge=1, le=30)
    max_gap_seconds: float = Field(default=10.0, ge=1.0, le=120.0)
    min_spatial_overlap: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Min IoU between consecutive anomaly heatmaps (0=disabled)",
    )
    # CUSUM evidence accumulation
    evidence_lambda: float = Field(
        default=0.95, ge=0.80, le=0.99,
        description="Exponential decay factor for evidence accumulation (0.8-0.99)",
    )
    evidence_threshold: float = Field(
        default=3.0, ge=0.5, le=20.0,
        description="Accumulated evidence threshold to trigger alert (0.5-20.0)",
    )


class SuppressionConfig(BaseModel):
    """Suppress duplicate alerts within time windows."""

    same_zone_window_seconds: float = Field(default=300.0, ge=10.0, le=3600.0)
    same_camera_window_seconds: float = Field(default=60.0, ge=5.0, le=3600.0)


class WebhookConfig(BaseModel):
    enabled: bool = False
    url: str = ""
    timeout: float = Field(default=5.0, ge=1.0, le=30.0)


class EmailConfig(BaseModel):
    enabled: bool = False
    min_severity: AlertSeverity = AlertSeverity.HIGH
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str = ""
    smtp_password: str = ""
    from_address: str = "argus@plant.local"
    use_tls: bool = True
    recipients: list[str] = Field(default_factory=list)


class CalibrationConfig(BaseModel):
    """Conformal prediction score calibration."""

    enabled: bool = Field(
        default=False,
        description="Use calibrated thresholds instead of manual severity_thresholds",
    )
    target_fpr_info: float = Field(default=0.10, ge=0.001, le=0.5)
    target_fpr_low: float = Field(default=0.01, ge=0.0001, le=0.1)
    target_fpr_medium: float = Field(default=0.001, ge=0.00001, le=0.01)
    target_fpr_high: float = Field(default=0.0001, ge=0.000001, le=0.001)
    min_calibration_samples: int = Field(default=50, ge=20, le=1000)


class AlertConfig(BaseModel):
    """Alert system configuration."""

    severity_thresholds: SeverityThresholds = Field(default_factory=SeverityThresholds)
    temporal: TemporalConfirmation = Field(default_factory=TemporalConfirmation)
    suppression: SuppressionConfig = Field(default_factory=SuppressionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    zone_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"critical": 1.2, "standard": 1.0, "low_priority": 0.8}
    )
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    circuit_breaker_threshold: int = Field(default=5, ge=1, le=50)
    circuit_breaker_timeout: float = Field(default=60.0, ge=10.0, le=600.0)


class AuthConfig(BaseModel):
    """Dashboard authentication settings."""

    enabled: bool = False
    api_token: str = Field(
        default="",
        description="Shared API token for HTTP Basic Auth (must be set when auth is enabled)",
    )
    session_timeout_minutes: int = Field(
        default=480, ge=10, le=1440,
        description="Session timeout in minutes (default: 480 = 8-hour shift)",
    )


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1024, le=65535)
    debug: bool = False
    mjpeg_fps: float = Field(
        default=5.0, ge=1.0, le=30.0,
        description="MJPEG stream target FPS",
    )
    websocket_heartbeat_seconds: int = Field(
        default=30, ge=5, le=300,
        description="WebSocket ping interval in seconds",
    )
    websocket_max_connections: int = Field(
        default=100, ge=1, le=1000,
        description="Maximum concurrent WebSocket connections",
    )


class LoggingConfig(BaseModel):
    """Log rotation and output settings."""

    log_dir: Path = Path("data/logs")
    max_file_size_mb: int = Field(default=50, ge=1, le=500)
    backup_count: int = Field(default=10, ge=1, le=50)
    log_format: Literal["json", "console"] = "json"


class StorageConfig(BaseModel):
    database_url: str = "sqlite:///data/db/argus.db"
    baselines_dir: Path = Path("data/baselines")
    models_dir: Path = Path("data/models")
    exports_dir: Path = Path("data/exports")
    backbones_dir: Path = Path("data/backbones")
    foe_objects_dir: Path = Path("data/foe_objects")
    model_packages_dir: Path = Path("data/model_packages")
    alerts_dir: Path = Path("data/alerts")
    alert_retention_days: int = Field(
        default=90, ge=7, le=3650,
        description="Days to retain alert records and images (7-3650)",
    )


class ModelsConfig(BaseModel):
    """Paths to model files."""

    yolo_path: str = "yolo11n.pt"
    anomalib_model_dir: Path = Path("data/models")
    anomalib_export_dir: Path = Path("data/exports")


class CameraOverlapConfig(BaseModel):
    """Overlap between two cameras for cross-correlation."""

    camera_a: str
    camera_b: str
    homography: list[list[float]] = Field(
        description="3x3 homography matrix projecting from camera_a to camera_b",
    )


class RetrainingConfig(BaseModel):
    """Scheduled automatic retraining configuration (C4 + A4)."""

    enabled: bool = Field(default=False, description="Enable scheduled retraining")
    interval_hours: int = Field(
        default=24, ge=1, le=168,
        description="Hours between retraining checks",
    )
    min_new_baselines: int = Field(
        default=20, ge=5, le=500,
        description="Minimum new baseline images to trigger retraining",
    )
    auto_deploy: bool = Field(
        default=False, description="Auto-deploy if quality grade meets threshold",
    )
    auto_deploy_min_grade: str = Field(
        default="B", description="Minimum quality grade for auto-deploy (A/B/C/F)",
    )

    # Two-level training architecture
    backbone_retrain_interval_days: int = Field(
        default=30, ge=7, le=180,
        description="Days between backbone SSL fine-tuning checks",
    )
    backbone_type: str = Field(
        default="dinov2_vitb14",
        description="DINOv2 backbone variant for SSL pretraining",
    )

    # Training validation thresholds
    validation_auroc_threshold: float = Field(
        default=0.99, ge=0.90, le=1.0,
        description="Minimum AUROC on holdout set for validation pass",
    )
    validation_recall_threshold: float = Field(
        default=0.95, ge=0.80, le=1.0,
        description="Minimum recall on synthetic anomalies for validation pass",
    )
    historical_replay_days: int = Field(
        default=30, ge=7, le=90,
        description="Days of historical alerts to replay for regression testing",
    )

    # Human confirmation (nuclear environment)
    require_human_confirmation: bool = Field(
        default=True,
        description="Require human confirmation before training starts (nuclear safety)",
    )
    confirmation_timeout_hours: int = Field(
        default=72, ge=1, le=168,
        description="Auto-reject pending jobs after this many hours",
    )

    @field_validator("auto_deploy_min_grade")
    @classmethod
    def validate_grade(cls, v: str) -> str:
        if v not in ("A", "B", "C", "F"):
            raise ValueError(f"Grade must be A, B, C, or F, got {v!r}")
        return v


class CrossCameraConfig(BaseModel):
    """Cross-camera anomaly correlation."""

    enabled: bool = Field(default=False)
    overlap_pairs: list[CameraOverlapConfig] = Field(default_factory=list)
    corroboration_threshold: float = Field(
        default=0.3, ge=0.1, le=0.9,
        description="Partner anomaly score threshold to consider corroborated",
    )
    max_age_seconds: float = Field(
        default=5.0, ge=1.0, le=30.0,
        description="Max age of partner data to use for correlation",
    )
    uncorroborated_severity_downgrade: int = Field(
        default=1, ge=0, le=2,
        description="Downgrade severity by N levels when uncorroborated (0=disabled)",
    )


class BaselineCaptureConfig(BaseModel):
    """Advanced baseline capture job configuration."""

    default_strategy: Literal["uniform", "active", "scheduled"] = "active"
    diversity_threshold: float = Field(
        default=0.3, ge=0.1, le=0.9,
        description="Min cosine distance for active sampling (higher = more diverse)",
    )
    dino_backbone: str = Field(
        default="dinov2_vits14",
        description="DINOv2 backbone for active sampling feature extraction",
    )
    dino_image_size: int = Field(
        default=224, ge=112, le=518,
        description="Input image size for DINOv2 feature extraction",
    )
    schedule_periods: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {
            "dawn": (5, 8),
            "noon": (11, 13),
            "dusk": (16, 19),
            "night": (22, 2),
        },
        description="Time windows for scheduled sampling {name: (start_hour, end_hour)}",
    )
    frames_per_period: int = Field(
        default=50, ge=5, le=200,
        description="Target frames per time period in scheduled mode",
    )
    pause_on_anomaly_lock: bool = Field(
        default=True,
        description="Auto-pause capture when camera has anomaly lock active",
    )
    post_capture_review: bool = Field(
        default=True,
        description="Run existing model on captured frames to flag outliers",
    )
    review_flag_percentile: float = Field(
        default=0.99, ge=0.9, le=1.0,
        description="Flag frames above this anomaly score percentile for review",
    )


class ArgusConfig(BaseModel):
    """Top-level configuration for the Argus system."""

    node_id: str = "argus-edge-01"
    cameras: list[CameraConfig] = Field(default_factory=list)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    capture_quality: CaptureQualityConfig = Field(default_factory=CaptureQualityConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    segmenter: SegmenterConfig = Field(default_factory=SegmenterConfig)
    cross_camera: CrossCameraConfig = Field(default_factory=CrossCameraConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)
    baseline_capture: BaselineCaptureConfig = Field(default_factory=BaselineCaptureConfig)
    camera_groups: list[CameraGroupConfig] = Field(
        default_factory=list,
        description="Camera groups for shared baselines and models",
    )
    log_level: str = "INFO"
