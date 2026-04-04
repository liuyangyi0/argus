"""Pydantic configuration models for Argus with industrial-grade validation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class AnomalyConfig(BaseModel):
    """Anomalib anomaly detection parameters."""

    model_type: Literal["patchcore", "efficient_ad", "fastflow", "padim"] = "patchcore"
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
    mog2: MOG2Config = Field(default_factory=MOG2Config)
    person_filter: PersonFilterConfig = Field(default_factory=PersonFilterConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)


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


class AlertConfig(BaseModel):
    """Alert system configuration."""

    severity_thresholds: SeverityThresholds = Field(default_factory=SeverityThresholds)
    temporal: TemporalConfirmation = Field(default_factory=TemporalConfirmation)
    suppression: SuppressionConfig = Field(default_factory=SuppressionConfig)
    zone_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"critical": 1.2, "standard": 1.0, "low_priority": 0.8}
    )
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)


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
    log_level: str = "INFO"
