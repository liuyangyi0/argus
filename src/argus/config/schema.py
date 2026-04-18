"""Pydantic configuration models for Argus with industrial-grade validation."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _default_foe_vocab() -> list[str]:
    """Canonical FOE vocabulary — single source of truth.

    Also exported as FOE_VOCAB from argus.anomaly.classifier.
    """
    from argus.anomaly.classifier import FOE_VOCAB

    return list(FOE_VOCAB)


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
    require_corroboration: bool = Field(
        default=False,
        description="F4: Drop uncorroborated alerts entirely for this zone instead of "
        "severity downgrade. Strict mode for zones that must have cross-camera confirmation.",
    )


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
    sahi_enabled: bool = Field(
        default=False,
        description="Enable SAHI sliced inference for small/distant object detection",
    )
    sahi_slice_size: int = Field(
        default=640, ge=320, le=1920,
        description="SAHI slice width/height in pixels",
    )
    sahi_overlap_ratio: float = Field(
        default=0.25, ge=0.0, le=0.5,
        description="Overlap ratio between adjacent slices for SAHI",
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


class EnsembleDetectionConfig(BaseModel):
    """Multi-model ensemble for reduced false positives."""

    enabled: bool = Field(default=False, description="Enable multi-model ensemble detection")
    method: Literal["mean", "max", "weighted", "vote", "bayesian"] = Field(
        default="mean", description="Score fusion method",
    )
    weights: list[float] | None = Field(
        default=None, description="Model weights for 'weighted' method",
    )
    bayesian_prior: float = Field(
        default=0.01, ge=0.001, le=0.5,
        description="Prior probability of anomaly for Bayesian fusion",
    )
    dynamic_fpr_weighting: bool = Field(
        default=False, description="Dynamically adjust weights based on per-model FPR",
    )


class AnomalyConfig(BaseModel):
    """Anomalib anomaly detection parameters."""

    model_type: Literal["patchcore", "efficient_ad", "fastflow", "padim", "dinomaly2"] = "patchcore"
    threshold: float = Field(default=0.7, ge=0.1, le=0.99)
    image_size: tuple[int, int] = Field(
        default=(256, 256),
        description="Anomaly model input resolution. Must match training resolution. "
        "512x512 recommended for new models targeting ≥8mm² detection at 5m.",
    )
    min_contour_area: int = Field(
        default=50, ge=1, le=10000,
        description="Minimum anomaly contour area in heatmap pixels. Lower for small objects.",
    )
    enable_multiscale: bool = Field(
        default=False,
        description="Enable sliding window multi-scale detection for small objects",
    )
    tile_size: int = Field(
        default=512, ge=256, le=1920,
        description="Tile size in pixels for sliding window (256-1920). Ignored when pyramid_mode is enabled.",
    )
    tile_overlap: float = Field(
        default=0.25, ge=0.0, le=0.5,
        description="Overlap ratio between adjacent tiles (0.0-0.5)",
    )
    pyramid_mode: bool = Field(
        default=True,
        description="Use 3-level pyramid (512/768/1024) instead of single tile_size. "
        "Captures anomalies at different scales — small, medium, and large.",
    )
    pyramid_sizes: list[int] = Field(
        default=[512, 768, 1024],
        description="Tile sizes for pyramid levels (smallest to largest)",
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
        default=True,
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
    # Backbone training resource limits
    backbone_max_per_camera: int = Field(
        default=5000, ge=100, le=20000,
        description="Maximum baseline images per camera for backbone SSL training",
    )
    backbone_max_total_images: int = Field(
        default=50000, ge=1000, le=200000,
        description="Maximum total images across all cameras for backbone training",
    )
    # Multi-model ensemble
    ensemble: EnsembleDetectionConfig = Field(
        default_factory=EnsembleDetectionConfig,
        description="Multi-model ensemble detection for reduced false positives",
    )


class ClassifierConfig(BaseModel):
    """Open vocabulary detection classifier (D1)."""

    enabled: bool = Field(default=False)
    model_name: str = Field(default="yolov8s-worldv2.pt")
    vocabulary: list[str] = Field(
        default_factory=lambda: list(_default_foe_vocab()),
    )
    min_anomaly_score_to_classify: float = Field(default=0.5, ge=0.0, le=1.0)
    high_risk_labels: list[str] = Field(
        default_factory=lambda: [
            "wrench", "bolt", "nut", "screwdriver", "hammer",
            "gasket", "o_ring", "cotter_pin", "safety_wire", "washer",
            "hose_clamp", "concrete_chip", "metal_shaving",
            "graphite_gasket", "rubber_seal", "electrode_fragment",
            "bearing_ball", "lens_piece",
        ]
    )
    low_risk_labels: list[str] = Field(
        default_factory=lambda: ["insect", "shadow", "reflection"]
    )
    suppress_labels: list[str] = Field(
        default_factory=lambda: ["crane", "overhead_bridge", "scaffold"],
        description="Objects that suppress anomaly detection in their region",
    )
    custom_vocabulary_path: str | None = Field(
        default=None,
        description="Path to custom vocabulary JSON file (overrides default vocabulary)",
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
    timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0,
        description="Maximum seconds to wait for SAM2 inference before timeout",
    )


class DegradationConfig(BaseModel):
    """Degradation and resilience configuration for inference runners."""

    max_consecutive_failures: int = Field(
        default=5, ge=1, le=50,
        description="Consecutive detection failures before attempting model restart",
    )
    refuse_start_on_backbone_failure: bool = Field(
        default=False,
        description="Refuse to start camera if backbone model fails to load (True=strict, False=SSIM fallback)",
    )
    watchdog_check_interval_seconds: float = Field(
        default=15.0, ge=5.0, le=120.0,
        description="Interval for process-level watchdog to check thread liveness",
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


class RingBufferConfig(BaseModel):
    """Alert ring buffer configuration for replay recordings (FR-033)."""

    enabled: bool = Field(default=True, description="Enable alert replay recording")
    pre_trigger_seconds: int = Field(
        default=60, ge=10, le=120,
        description="Seconds of footage to keep before alert trigger",
    )
    post_trigger_seconds: int = Field(
        default=30, ge=10, le=60,
        description="Seconds of footage to capture after alert trigger",
    )
    jpeg_quality: int = Field(
        default=85, ge=60, le=95,
        description="JPEG compression quality for buffered frames",
    )
    video_crf: int = Field(
        default=23, ge=18, le=28,
        description="H.264 CRF quality (lower = better quality, larger files)",
    )
    video_preset: str = Field(
        default="veryfast",
        pattern=r"^(ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow)$",
        description="libx264 encoding preset (ultrafast/veryfast/fast/medium/slow)",
    )
    max_recording_age_days: int = Field(
        default=30, ge=7, le=365,
        description="Days to keep full recordings before archiving to trigger-frame-only",
    )
    archive_dir: str = Field(
        default="data/recordings",
        description="Directory for solidified alert recordings",
    )


class LowLightConfig(BaseModel):
    """Low-light detection: bypass MOG2 when scene is dark.

    When mean frame brightness drops below ``brightness_threshold``, the
    pipeline skips the MOG2 pre-filter (which would otherwise discard the
    frame as "no change") and shortens the heartbeat interval so that
    temporal evidence can accumulate and trigger alerts.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances
    low-light frames before anomaly detection, improving feature visibility
    at <1ms per frame cost.
    """

    enabled: bool = Field(default=True, description="Enable automatic low-light MOG2 bypass")
    brightness_threshold: float = Field(
        default=40.0, ge=5.0, le=128.0,
        description="Mean grayscale brightness below which low-light mode activates",
    )
    clahe_enabled: bool = Field(
        default=True,
        description="Apply CLAHE preprocessing on low-light frames before anomaly detection",
    )
    clahe_clip_limit: float = Field(
        default=3.0, ge=1.0, le=10.0,
        description="CLAHE contrast clip limit (higher = more contrast, more noise)",
    )
    clahe_grid_size: int = Field(
        default=8, ge=2, le=16,
        description="CLAHE tile grid size (NxN tiles for local histogram equalization)",
    )
    brightness_jump_threshold: float = Field(
        default=30.0, ge=5.0, le=100.0,
        description="Brightness change between frames to freeze MOG2 learning rate",
    )


class EventCameraConfig(BaseModel):
    """Event camera (neuromorphic sensor) configuration.

    Reserved interface for future Prophesee/iniVation event camera integration.
    Not implemented — SDK stubs only.
    """

    sdk: Literal["metavision", "dv", "none"] = Field(
        default="none",
        description="Event camera SDK: metavision (Prophesee), dv (iniVation), none (disabled)",
    )
    accumulation_window_ms: float = Field(
        default=33.0, ge=1.0, le=500.0,
        description="Time window (ms) for accumulating events into a frame-equivalent",
    )
    event_rate_threshold: int = Field(
        default=1000, ge=100, le=1000000,
        description="Events/second above which a trigger is generated",
    )
    trigger_frame_camera: bool = Field(
        default=True,
        description="When event burst detected, trigger frame camera to boost FPS",
    )
    trigger_fps_boost: int = Field(
        default=120, ge=30, le=240,
        description="Target FPS for frame camera when triggered by event camera",
    )


class GigEConfig(BaseModel):
    """GigE Vision camera-specific parameters."""

    exposure: float = Field(
        default=0,
        ge=0,
        description="Exposure time in microseconds (0 = auto)",
    )
    gain: float = Field(
        default=0,
        ge=0,
        description="Gain in dB (0 = auto)",
    )
    pixel_format: Literal["Mono8", "BayerBG8", "BayerGB8", "BayerGR8", "BayerRG8"] = Field(
        default="Mono8",
        description="SDK pixel format for GigE camera sensor output",
    )
    capture_script: str | None = Field(
        default=None,
        description="Path to GStreamer capture script for browser preview "
        "(go2rtc exec source).  If None, preview is disabled for this camera.",
    )


class AlignmentConfig(BaseModel):
    """Sub-pixel phase-correlation alignment for camera micro-vibration.

    Runs before anomaly detection to cancel 1-2px frame-to-frame shifts
    caused by pumps, fans, and turbines — the #1 source of false positives
    in fixed-camera nuclear plant deployments.
    """

    enabled: bool = Field(
        default=False,
        description="Enable phase-correlation alignment preprocessing stage",
    )
    max_shift_px: float = Field(
        default=5.0, ge=0.5, le=50.0,
        description="Shifts above this (on either axis) are treated as real motion and skipped",
    )
    downsample: int = Field(
        default=4, ge=1, le=16,
        description="Run phase correlation at 1/N resolution to keep latency low",
    )
    ref_update_interval_s: float = Field(
        default=60.0, ge=1.0, le=3600.0,
        description="Seconds between reference frame refreshes to track slow scene drift",
    )


class CameraConfig(BaseModel):
    """Configuration for a single camera."""

    camera_id: str
    name: str
    region_id: int | None = Field(default=None, description="Linked business region id")

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_region_field(cls, data):
        """Gracefully handle older configs that still store region_name."""
        if isinstance(data, dict) and "region_id" not in data and "region_name" in data:
            legacy = data.get("region_name")
            if isinstance(legacy, int):
                data["region_id"] = legacy
            elif isinstance(legacy, str) and legacy.strip().isdigit():
                data["region_id"] = int(legacy.strip())
            else:
                data["region_id"] = None
        return data

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                f"camera_id must contain only letters, digits, hyphens and underscores, got {v!r}"
            )
        return v
    source: str  # RTSP URL, USB device index, file path, or GigE camera IP
    protocol: Literal["rtsp", "usb", "file", "event", "gige"] = "rtsp"
    fps_target: int = Field(default=5, ge=1, le=120)
    resolution: tuple[int, int] = (1920, 1080)
    zones: list[ZoneConfig] = Field(default_factory=list)
    reconnect_delay: float = Field(default=5.0, ge=1.0, le=300.0)
    max_reconnect_attempts: int = Field(default=-1, ge=-1)
    watchdog_timeout: float = Field(
        default=30.0, ge=5.0, le=300.0,
        description="Seconds without frames before forced reconnect (5-300)",
    )
    mog2: MOG2Config = Field(default_factory=MOG2Config)
    alignment: AlignmentConfig = Field(
        default_factory=AlignmentConfig,
        description="Phase-correlation alignment for camera micro-vibration",
    )
    person_filter: PersonFilterConfig = Field(default_factory=PersonFilterConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    simplex: SimplexConfig = Field(default_factory=SimplexConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    ring_buffer: RingBufferConfig = Field(default_factory=RingBufferConfig)
    low_light: LowLightConfig = Field(default_factory=LowLightConfig)
    gige: GigEConfig = Field(
        default_factory=GigEConfig,
        description="GigE Vision camera parameters (exposure, gain, pixel format, capture script)",
    )
    event: EventCameraConfig = Field(
        default_factory=EventCameraConfig,
        description="Event camera (neuromorphic) config — reserved interface",
    )
    calibration_file: str | None = Field(
        default=None,
        description="Path to camera calibration JSON (intrinsic/extrinsic parameters)",
    )
    tracker_match_distance: float = Field(
        default=50.0, ge=10.0, le=1000.0,
        description="Max pixel distance for centroid matching between frames "
        "(increase for high-speed objects: 500+ at 25m/s)",
    )
    tracker_max_gap_frames: int = Field(
        default=5, ge=1, le=30,
        description="Max consecutive frames without match before track is lost",
    )
    tracker_stationary_threshold: float = Field(
        default=10.0, ge=1.0, le=100.0,
        description="Velocity magnitude below which an object is considered stationary (px/frame)",
    )


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
    # F3: Strict spatial continuity — reject anomalies that drift across frames
    # (e.g. peeling paint, loose insulation whose bright pixels shift each frame).
    spatial_continuity_enabled: bool = Field(
        default=True,
        description="Gate CUSUM accumulation on IoU between consecutive anomaly masks",
    )
    spatial_continuity_iou_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum IoU between consecutive anomaly masks for evidence to continue",
    )
    spatial_continuity_mode: Literal["decay", "reset"] = Field(
        default="reset",
        description="Behavior on IoU below threshold: 'reset' zeroes evidence (stricter), "
        "'decay' halves evidence (legacy behavior)",
    )
    # F5: Stationary-object suppression — once a track sits still long enough,
    # stop re-triggering alerts for it every frame (e.g. settled FOE at pool bottom).
    stationary_suppress_enabled: bool = Field(
        default=True,
        description="Suppress alerts for tracks that have been stationary for a long time",
    )
    stationary_suppress_after_s: float = Field(
        default=10.0, ge=1.0, le=600.0,
        description="Seconds a track must remain stationary before it is marked suppressed",
    )


class SuppressionConfig(BaseModel):
    """Suppress duplicate alerts within time windows."""

    same_zone_window_seconds: float = Field(default=300.0, ge=10.0, le=3600.0)
    same_camera_window_seconds: float = Field(default=60.0, ge=5.0, le=3600.0)


class EarlyWarningConfig(BaseModel):
    """F1: Single-frame fast-path bypass of CUSUM evidence accumulation.

    When a single frame has an extremely high anomaly score AND corroborating
    evidence from YOLO detection or open-vocabulary classifier, emit the alert
    immediately rather than waiting for CUSUM evidence to accumulate. Suppression
    windows still apply to prevent storms.
    """

    enabled: bool = Field(
        default=True,
        description="Enable single-frame early-warning fast path",
    )
    score_threshold: float = Field(
        default=0.95, ge=0.5, le=0.99,
        description="Adjusted anomaly score (post zone multiplier) required to fire the fast path",
    )
    require_detection_or_classifier: bool = Field(
        default=True,
        description="Require corroborating YOLO detection or classifier label before firing. "
        "Set False to allow pure-anomaly-only early warning (more false positives).",
    )
    classifier_min_confidence: float = Field(
        default=0.9, ge=0.1, le=0.99,
        description="If the classifier fired, its confidence must be >= this value to qualify "
        "as early-warning corroboration.",
    )


class WebhookConfig(BaseModel):
    enabled: bool = False
    url: str = ""
    timeout: float = Field(default=5.0, ge=1.0, le=30.0)


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
    early_warning: EarlyWarningConfig = Field(default_factory=EarlyWarningConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    zone_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"critical": 1.2, "standard": 1.0, "low_priority": 0.8}
    )
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    circuit_breaker_threshold: int = Field(default=5, ge=1, le=50)
    circuit_breaker_timeout: float = Field(default=60.0, ge=10.0, le=600.0)
    category_enabled: bool = Field(default=True, description="启用告警自动分类")
    enabled_categories: list[str] = Field(
        default=[
            "projectile", "static_foreign", "scene_change",
            "environmental", "person_intrusion", "equipment_displacement",
        ],
        description="启用的告警分类列表",
    )


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


class AudioAlertSeverityConfig(BaseModel):
    """Audio alert settings for a single severity level."""

    enabled: bool = False
    sound: str = Field(
        default="beep_single",
        description="Sound identifier: beep_single, beep_double, beep_double_voice",
    )
    voice_template: str = Field(
        default="",
        description="TTS voice template, e.g. '{camera} 高级别告警'",
    )


class AudioAlertConfig(BaseModel):
    """Audio alert configuration per severity (UX v2 §2.5)."""

    low: AudioAlertSeverityConfig = Field(
        default_factory=lambda: AudioAlertSeverityConfig(enabled=False),
    )
    medium: AudioAlertSeverityConfig = Field(
        default_factory=lambda: AudioAlertSeverityConfig(
            enabled=True, sound="beep_single",
        ),
    )
    high: AudioAlertSeverityConfig = Field(
        default_factory=lambda: AudioAlertSeverityConfig(
            enabled=True, sound="beep_double_voice",
            voice_template="{camera} 高级别告警",
        ),
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
    audio_alerts: AudioAlertConfig = Field(default_factory=AudioAlertConfig)

    # go2rtc streaming proxy
    go2rtc_enabled: bool = Field(
        default=True,
        description="Enable go2rtc for WebRTC/MSE camera streaming",
    )
    go2rtc_api_port: int = Field(
        default=1984, ge=1024, le=65535,
        description="go2rtc HTTP API / WebRTC signalling port",
    )
    go2rtc_rtsp_port: int = Field(
        default=8554, ge=1024, le=65535,
        description="go2rtc RTSP listener port",
    )
    go2rtc_binary: str | None = Field(
        default=None,
        description="Path to go2rtc binary (auto-detected if None)",
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
    inference_records_dir: Path = Path("data/inference_records")
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


class FeedbackConfig(BaseModel):
    """Feedback loop configuration (Section 6)."""

    auto_baseline_on_fp: bool = Field(
        default=True,
        description="Automatically add FP snapshots to baseline directory",
    )
    auto_validation_on_confirmed: bool = Field(
        default=True,
        description="Automatically add confirmed anomaly frames to validation set",
    )
    uncertain_escalation_role: str = Field(
        default="supervisor",
        description="Role to notify when operator selects 'uncertain'",
    )
    passive_feedback_enabled: bool = Field(
        default=True,
        description="Generate feedback entries from drift/health events",
    )
    validation_dir: Path = Field(
        default=Path("data/validation"),
        description="Directory for confirmed anomaly frames used in recall testing",
    )


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
    active_sleep_min_seconds: float = Field(
        default=1.0, ge=0.1, le=60.0,
        description="Minimum seconds between active-sampling frame grabs to avoid CPU starvation",
    )
    active_cpu_threads: int = Field(
        default=1, ge=1, le=8,
        description="Maximum CPU threads used by active-sampling DINOv2/FAISS inference",
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


class ImagingConfig(BaseModel):
    """Multi-modal imaging: DoFP polarization + NIR strobe (M1).

    Enables polarization-based water surface reflection removal and
    NIR strobed illumination for high-speed capture without motion blur.
    """

    enabled: bool = Field(
        default=False,
        description="Enable multi-modal imaging pipeline",
    )
    mode: Literal["visible_only", "polarization", "polarization_nir"] = Field(
        default="visible_only",
        description="Imaging modality: visible_only (standard), polarization (DoFP deglare), "
        "polarization_nir (DoFP + NIR strobe)",
    )
    camera_sdk: Literal["opencv", "arena", "spinnaker"] = Field(
        default="opencv",
        description="Camera acquisition SDK: opencv (standard), arena (LUCID), spinnaker (FLIR)",
    )
    nir_strobe_enabled: bool = Field(
        default=False,
        description="Enable NIR 850nm strobe synchronized via ExposureActive GPIO",
    )
    polarization_processing: bool = Field(
        default=False,
        description="Enable DoFP polarization demosaicing and deglare processing",
    )
    fusion_channels: int = Field(
        default=3, ge=1, le=5,
        description="Number of fused input channels: 1=gray, 3=RGB, 4=RGB+DoLP, 5=RGB+DoLP+NIR",
    )
    deglare_method: Literal["stokes", "min_intensity"] = Field(
        default="stokes",
        description="Reflection removal method: stokes (classical Stokes-based), "
        "min_intensity (fast I_min approximation)",
    )
    dolp_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="DoLP threshold above which a pixel is considered specular reflection",
    )


class PhysicsConfig(BaseModel):
    """Physics-based speed monitoring and trajectory analysis (M5/M8-M10).

    Phase 1: pixel-level speed estimation (no calibration required).
    Phase 2: calibrated speed, trajectory fitting, origin/landing estimation.
    """

    speed_enabled: bool = Field(
        default=False,
        description="Enable speed monitoring (phase 1: pixel speed, phase 2: calibrated m/s)",
    )
    trajectory_enabled: bool = Field(
        default=False,
        description="Enable trajectory fitting and physics model analysis (phase 2)",
    )
    localization_enabled: bool = Field(
        default=False,
        description="Enable origin/landing point estimation (phase 2)",
    )
    triangulation_enabled: bool = Field(
        default=False,
        description="Enable multi-camera 3D triangulation (phase 2)",
    )
    calibration_dir: Path = Field(
        default=Path("data/calibration"),
        description="Directory for per-camera calibration files",
    )
    pool_surface_z_mm: float = Field(
        default=0.0,
        description="Z coordinate of water surface in world frame (mm)",
    )
    gravity_ms2: float = Field(
        default=9.81, ge=9.0, le=10.0,
        description="Local gravitational acceleration (m/s^2)",
    )
    trajectory_history_length: int = Field(
        default=300, ge=30, le=1000,
        description="Maximum trajectory points per track (30fps * 10s = 300)",
    )
    speed_smoothing_window: int = Field(
        default=5, ge=1, le=30,
        description="Frames for moving-average speed smoothing",
    )
    min_trajectory_points: int = Field(
        default=5, ge=3, le=50,
        description="Minimum tracked points before trajectory fitting",
    )
    origin_accuracy_target_mm: float = Field(
        default=200.0, ge=50.0, le=1000.0,
        description="Target accuracy for origin point estimation (mm)",
    )
    landing_accuracy_target_mm: float = Field(
        default=500.0, ge=100.0, le=2000.0,
        description="Target accuracy for water surface landing estimation (mm)",
    )
    pixel_scale_mm_per_px: float | None = Field(
        default=None,
        description="Simple pixel-to-mm scale factor (phase 1 quick calibration). "
        "Set to None to use full calibration file.",
    )
    use_drag_model: bool = Field(
        default=True,
        description="Use air-drag corrected trajectory model instead of pure free-fall",
    )


class ContinuousRecordingConfig(BaseModel):
    """Continuous 24/7 video recording with segmented storage (M6).

    Records all camera feeds to disk in fixed-duration segments
    for compliance with 180-day local + 360-day archive retention.
    """

    enabled: bool = Field(
        default=False,
        description="Enable continuous recording for all cameras",
    )
    segment_duration_hours: float = Field(
        default=4.0, ge=0.5, le=8.0,
        description="Maximum segment duration in hours before rotation",
    )
    local_retention_days: int = Field(
        default=180, ge=30, le=365,
        description="Days to retain recordings on local storage",
    )
    archive_enabled: bool = Field(
        default=False,
        description="Enable archival to external storage after local retention",
    )
    archive_path: Path | None = Field(
        default=None,
        description="External storage mount point for archived recordings",
    )
    archive_retention_days: int = Field(
        default=360, ge=30, le=730,
        description="Days to retain recordings on archive storage",
    )
    encoding_crf: int = Field(
        default=26, ge=18, le=35,
        description="H.264 CRF for continuous recording (higher = smaller, lower quality)",
    )
    encoding_preset: str = Field(
        default="veryfast",
        pattern=r"^(ultrafast|superfast|veryfast|faster|fast|medium)$",
        description="libx264 encoding preset for continuous recording",
    )
    encoding_fps: int = Field(
        default=10, ge=1, le=30,
        description="Target FPS for continuous recording (lower than detection FPS to save space)",
    )
    output_dir: Path = Field(
        default=Path("data/continuous_recordings"),
        description="Directory for continuous recording segments",
    )
    cleanup_interval_hours: float = Field(
        default=6.0, ge=1.0, le=24.0,
        description="Hours between retention cleanup scans",
    )

    @model_validator(mode="after")
    def _validate_archive(self) -> ContinuousRecordingConfig:
        if self.archive_enabled and self.archive_path is None:
            raise ValueError("archive_path must be set when archive_enabled=True")
        return self


class SensorFusionConfig(BaseModel):
    """External sensor fusion hook — generic multi-modal signal input.

    Third-party sensors (temperature, vibration, radiation, lidar, etc.)
    can push short-lived multipliers keyed on ``(camera_id, zone_id)`` that
    bias the alert grader's severity calculation. The config here only
    governs activation and default TTL — the transport is HTTP
    ``/api/sensors/signal``.
    """

    enabled: bool = Field(
        default=False,
        description="Consult external sensor fusion multipliers during grading",
    )
    default_valid_for_s: float = Field(
        default=60.0, ge=0.0, le=3600.0,
        description="Default time-to-live (seconds) for a pushed signal when "
        "the client does not specify one",
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
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    baseline_capture: BaselineCaptureConfig = Field(default_factory=BaselineCaptureConfig)
    imaging: ImagingConfig = Field(default_factory=ImagingConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    continuous_recording: ContinuousRecordingConfig = Field(
        default_factory=ContinuousRecordingConfig,
    )
    sensor_fusion: SensorFusionConfig = Field(
        default_factory=SensorFusionConfig,
        description="External sensor fusion hook (generic multi-modal signal input)",
    )
    camera_groups: list[CameraGroupConfig] = Field(
        default_factory=list,
        description="Camera groups for shared baselines and models",
    )
    log_level: str = "INFO"
