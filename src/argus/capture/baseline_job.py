"""Baseline capture job — stateful, long-running frame collection task.

Supports three sampling strategies (uniform, active, scheduled),
three purity safeguards (anomaly lock, quality filter, post-capture review),
and produces immutable versioned datasets.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import structlog

from argus.anomaly.baseline import BaselineManager
from argus.capture.quality import FrameQualityFilter, CaptureStats
from argus.capture.samplers import (
    BaseSampler,
    UniformSampler,
    ActiveSampler,
    ScheduledSampler,
)
from argus.config.schema import CaptureQualityConfig, BaselineCaptureConfig

logger = structlog.get_logger()


@dataclass
class BaselineCaptureJobConfig:
    """Configuration for a baseline capture job."""

    camera_id: str
    target_frames: int
    duration_hours: float
    sampling_strategy: str  # "uniform" / "active" / "scheduled"
    storage_path: str  # baselines root dir
    quality_config: CaptureQualityConfig = field(default_factory=CaptureQualityConfig)
    pause_on_anomaly_lock: bool = True
    diversity_threshold: float = 0.3
    dino_backbone: str = "dinov2_vits14"
    dino_image_size: int = 224
    schedule_periods: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "dawn": (5, 8),
            "noon": (11, 13),
            "dusk": (16, 19),
            "night": (22, 2),
        }
    )
    frames_per_period: int = 50
    post_capture_review: bool = True
    review_flag_percentile: float = 0.99
    models_dir: str = "data/models"
    exports_dir: str = "data/exports"


def run_baseline_capture_job(
    progress_callback: Callable[[int, str], None],
    *,
    job_config: BaselineCaptureJobConfig,
    camera_manager: Any,  # CameraManager (avoid circular import)
    pause_event: Any,  # threading.Event
    abort_event: Any,  # threading.Event
) -> dict:
    """Run a baseline capture job to completion.

    This is a long-running, blocking function designed to be executed in a
    background thread or task runner.  It captures frames from a camera,
    applies quality filtering and sampling strategies, and produces an
    immutable versioned dataset with SHA-256 manifest.

    Args:
        progress_callback: Called with (percent, message) to report progress.
        job_config: Full capture job configuration.
        camera_manager: Camera manager instance providing ``get_raw_frame``
            and ``is_anomaly_locked`` methods.
        pause_event: A ``threading.Event`` that is *set* when the job should
            run and *cleared* when the job should pause.
        abort_event: A ``threading.Event`` that is *set* to abort the job.

    Returns:
        Summary dict with capture results including collected frame count,
        duration, stats, review results, and output directory.

    Raises:
        TaskAbortedError: If the job is aborted via *abort_event*.
    """
    from argus.dashboard.tasks import TaskAbortedError

    # ── Setup ────────────────────────────────────────────────────────────
    bm = BaselineManager(job_config.storage_path)
    version_dir = bm.create_new_version(job_config.camera_id)

    sampler = _create_sampler(job_config)
    quality_filter = FrameQualityFilter(job_config.quality_config)
    stats = CaptureStats()

    collected = 0
    last_saved_frame = None
    start_time = time.monotonic()
    deadline = start_time + job_config.duration_hours * 3600

    progress_callback(
        0,
        f"开始采集: {job_config.sampling_strategy} 策略, "
        f"目标 {job_config.target_frames} 帧",
    )

    # ── Main capture loop ────────────────────────────────────────────────
    try:
        while collected < job_config.target_frames and time.monotonic() < deadline:
            # Pause checkpoint — blocks with zero CPU while paused
            pause_event.wait()

            # Abort check
            if abort_event.is_set():
                raise TaskAbortedError("采集任务已中止")

            # Layer 1: Anomaly lock check
            if (
                job_config.pause_on_anomaly_lock
                and camera_manager.is_anomaly_locked(job_config.camera_id)
            ):
                progress_pct = int(100 * collected / job_config.target_frames)
                progress_callback(progress_pct, "等待异常锁定解除...")
                while not abort_event.is_set():
                    time.sleep(2.0)
                    if not camera_manager.is_anomaly_locked(job_config.camera_id):
                        break
                progress_callback(progress_pct, "异常锁定已解除, 继续采集")
                continue

            # Grab raw frame
            frame = camera_manager.get_raw_frame(job_config.camera_id)
            if frame is None:
                stats.null_frames += 1
                time.sleep(1.0)
                continue

            stats.total_grabbed += 1

            # Layer 2: Quality filter (includes person detection)
            qr = quality_filter.check(frame, last_saved_frame)
            if not qr.accepted:
                stats.record_rejection(qr.rejection_reason)
                time.sleep(sampler.get_sleep_interval())
                continue

            # Sampler strategy filter
            accepted, sample_meta = sampler.should_accept(frame)
            if not accepted:
                time.sleep(sampler.get_sleep_interval())
                continue

            # Save frame
            frame_path = version_dir / f"baseline_{collected:05d}.png"
            cv2.imwrite(str(frame_path), frame)

            # Write per-frame sidecar metadata
            frame_meta = {
                "frame_index": collected,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "blur_score": float(qr.blur_score),
                "brightness_mean": float(qr.brightness_mean),
                "entropy": float(qr.entropy),
                **sample_meta,
            }
            meta_path = version_dir / f"baseline_{collected:05d}.json"
            meta_path.write_text(json.dumps(frame_meta, indent=2))

            # Notify sampler of saved frame
            sampler.on_frame_saved(frame, collected)

            # Track state
            last_saved_frame = frame
            collected += 1
            stats.accepted += 1
            if qr.brightness_mean is not None:
                stats.brightness_values.append(float(qr.brightness_mean))

            # Progress
            progress_pct = int(100 * collected / job_config.target_frames)
            progress_callback(
                progress_pct,
                f"已采集 {collected}/{job_config.target_frames} 帧",
            )

            # Sleep before next grab
            time.sleep(sampler.get_sleep_interval())

    finally:
        # ── Finalization ─────────────────────────────────────────────────
        # Always write outputs even on abort/error so partial data is usable.
        bm.set_current_version(
            job_config.camera_id, "default", version_dir.name
        )

        _write_manifest(version_dir)
        _write_stats(version_dir, stats, collected, start_time)
        _write_job_config(version_dir, job_config)

    # ── Layer 3: Post-capture review ─────────────────────────────────────
    review_result = None
    if job_config.post_capture_review and collected > 0:
        try:
            from argus.capture.post_review import post_capture_review

            review_result = post_capture_review(
                version_dir=version_dir,
                camera_id=job_config.camera_id,
                models_dir=Path(job_config.models_dir),
                exports_dir=Path(job_config.exports_dir),
                flag_percentile=job_config.review_flag_percentile,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.warning("post_review.failed", error=str(e))
            review_result = {"skipped": True, "reason": str(e)}

    progress_callback(100, f"采集完成: {collected} 帧")

    return {
        "camera_id": job_config.camera_id,
        "version": version_dir.name,
        "collected_frames": collected,
        "target_frames": job_config.target_frames,
        "strategy": job_config.sampling_strategy,
        "duration_seconds": round(time.monotonic() - start_time, 1),
        "stats": stats.to_dict(),
        "review": review_result,
        "output_dir": str(version_dir),
    }


# ── Internal helpers ─────────────────────────────────────────────────────


def _create_sampler(job_config: BaselineCaptureJobConfig) -> BaseSampler:
    """Instantiate the sampling strategy from job config."""
    strategy = job_config.sampling_strategy

    if strategy == "uniform":
        return UniformSampler(job_config.duration_hours, job_config.target_frames)

    if strategy == "active":
        return ActiveSampler(
            diversity_threshold=job_config.diversity_threshold,
            backbone=job_config.dino_backbone,
            image_size=job_config.dino_image_size,
        )

    if strategy == "scheduled":
        inner = ActiveSampler(
            diversity_threshold=job_config.diversity_threshold,
            backbone=job_config.dino_backbone,
            image_size=job_config.dino_image_size,
        )
        return ScheduledSampler(
            schedule_periods=job_config.schedule_periods,
            frames_per_period=job_config.frames_per_period,
            inner_sampler=inner,
            duration_hours=job_config.duration_hours,
        )

    raise ValueError(f"Unknown sampling strategy: {strategy!r}")


def _write_manifest(version_dir: Path) -> Path:
    """Write SHA-256 manifest for all PNG frames."""
    manifest = version_dir / "manifest.sha256"
    lines: list[str] = []
    for png in sorted(version_dir.glob("*.png")):
        sha = hashlib.sha256(png.read_bytes()).hexdigest()
        lines.append(f"{sha}  {png.name}")
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def _write_stats(
    version_dir: Path,
    stats: CaptureStats,
    collected: int,
    start_time: float,
) -> Path:
    """Write aggregated capture statistics."""
    blur_scores: list[float] = []
    brightness_values: list[float] = []

    for meta_file in sorted(version_dir.glob("baseline_*.json")):
        try:
            meta = json.loads(meta_file.read_text())
            if "blur_score" in meta:
                blur_scores.append(meta["blur_score"])
            if "brightness_mean" in meta:
                brightness_values.append(meta["brightness_mean"])
        except (json.JSONDecodeError, KeyError):
            continue

    stats_dict = {
        "total_captured": collected,
        "total_grabbed": stats.total_grabbed,
        "total_rejected": stats.total_rejected,
        "rejection_breakdown": {
            "blur": stats.rejected_blur,
            "exposure": stats.rejected_exposure,
            "duplicate": stats.rejected_duplicate,
            "person": stats.rejected_person,
            "encoder": stats.rejected_encoder,
        },
        "brightness_distribution": _distribution(brightness_values),
        "laplacian_distribution": _distribution(blur_scores),
        "capture_duration_seconds": round(time.monotonic() - start_time, 1),
    }
    path = version_dir / "stats.json"
    path.write_text(json.dumps(stats_dict, indent=2))
    return path


def _distribution(values: list[float]) -> dict:
    """Compute basic distribution stats."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "std": 0}
    arr = np.array(values)
    return {
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
    }


def _write_job_config(
    version_dir: Path,
    job_config: BaselineCaptureJobConfig,
) -> Path:
    """Write complete job configuration snapshot."""
    config_dict = {}
    for f in job_config.__dataclass_fields__:
        val = getattr(job_config, f)
        if isinstance(val, Path):
            config_dict[f] = str(val)
        elif hasattr(val, "model_dump"):  # Pydantic BaseModel
            config_dict[f] = val.model_dump()
        else:
            config_dict[f] = val
    path = version_dir / "capture_job.json"
    path.write_text(json.dumps(config_dict, indent=2, default=str))
    return path
