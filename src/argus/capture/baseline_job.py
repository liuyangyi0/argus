# baseline_job.py

"""Baseline capture job — stateful, long-running frame collection task.

Supports three sampling strategies (uniform, active, scheduled),
three purity safeguards (anomaly lock, quality filter, post-capture review),
and produces immutable versioned datasets.
"""

from __future__ import annotations

import hashlib
import json
import shutil
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
    get_active_sampler_unavailable_reason,
)
from argus.config.schema import CaptureQualityConfig, BaselineCaptureConfig
from argus.core.pipeline import PipelineMode
from argus.core.pipeline_mode_guard import PipelineModeGuard

logger = structlog.get_logger()


@dataclass
class BaselineCaptureJobConfig:
    """Configuration for a baseline capture job."""

    camera_id: str
    target_frames: int
    duration_hours: float
    sampling_strategy: str  # "uniform" / "active" / "scheduled"
    storage_path: str  # baselines root dir
    session_label: str = "daytime"
    quality_config: CaptureQualityConfig = field(default_factory=CaptureQualityConfig)
    pause_on_anomaly_lock: bool = True
    diversity_threshold: float = 0.3
    dino_backbone: str = "dinov2_vits14"
    dino_image_size: int = 224
    active_sleep_min_seconds: float = 1.0
    active_cpu_threads: int = 1
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


@dataclass
class SamplerSelection:
    sampler: BaseSampler
    effective_strategy: str
    warning: str | None = None


def run_baseline_capture_job(
    progress_callback: Callable[[int, str], None],
    *,
    job_config: BaselineCaptureJobConfig,
    camera_manager: Any,  # CameraManager (avoid circular import)
    pause_event: Any,  # threading.Event
    abort_event: Any,  # threading.Event
    lifecycle: Any | None = None,
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
    min_required_frames = min(job_config.target_frames, max(10, job_config.target_frames // 10))

    sampler_selection = _create_sampler(job_config)
    sampler = sampler_selection.sampler
    effective_strategy = sampler_selection.effective_strategy
    enable_duplicate_filter = effective_strategy != "uniform"
    # Baseline capture runs alongside a live pipeline that is already using YOLO.
    # Keep capture filtering lightweight to avoid concurrent detector contention
    # freezing the dashboard during manual capture.
    quality_filter = FrameQualityFilter(
        job_config.quality_config,
        enable_person_detection=False,
        enable_duplicate_filter=enable_duplicate_filter,
    )
    stats = CaptureStats()

    collected = 0
    last_saved_frame = None
    start_time = time.monotonic()
    deadline = start_time + job_config.duration_hours * 3600
    last_logged_progress = -1
    last_progress_log_time = start_time
    last_null_log_time = start_time
    last_rejection_log_time = start_time
    waiting_on_lock = False
    loop_error: Exception | None = None

    def emit_progress(progress: int, message: str, *, force: bool = False) -> None:
        nonlocal last_logged_progress, last_progress_log_time

        now = time.monotonic()
        progress_callback(progress, message)
        should_log = force
        if not should_log and progress >= last_logged_progress + 5:
            should_log = True
        if not should_log and now - last_progress_log_time >= 5.0:
            should_log = True
        if should_log:
            logger.info(
                "baseline.capture_progress",
                camera_id=job_config.camera_id,
                session_label=job_config.session_label,
                requested_strategy=job_config.sampling_strategy,
                strategy=effective_strategy,
                progress=progress,
                message=message,
                collected=collected,
                target_frames=job_config.target_frames,
                total_grabbed=stats.total_grabbed,
                null_frames=stats.null_frames,
                total_rejected=stats.total_rejected,
            )
            last_logged_progress = progress
            last_progress_log_time = now

    logger.info(
        "baseline.capture_started",
        camera_id=job_config.camera_id,
        session_label=job_config.session_label,
        requested_strategy=job_config.sampling_strategy,
        strategy=effective_strategy,
        duplicate_filter_enabled=enable_duplicate_filter,
        target_frames=job_config.target_frames,
        duration_hours=job_config.duration_hours,
        min_required_frames=min_required_frames,
        output_dir=str(version_dir),
    )

    emit_progress(
        0,
        f"开始采集: {effective_strategy} 策略, "
        f"目标 {job_config.target_frames} 帧",
        force=True,
    )
    if sampler_selection.warning:
        emit_progress(0, sampler_selection.warning, force=True)
        logger.warning(
            "baseline.capture_sampler_fallback",
            camera_id=job_config.camera_id,
            requested_strategy=job_config.sampling_strategy,
            effective_strategy=effective_strategy,
            reason=sampler_selection.warning,
        )

    # ── Main capture loop ────────────────────────────────────────────────
    # PipelineModeGuard switches the camera into COLLECTION for the loop
    # duration (痛点 1) and guarantees restoration on exit. We use explicit
    # enter/exit instead of `with` to avoid re-indenting the long loop body.
    failed_dir_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_guard = PipelineModeGuard(
        camera_manager,
        job_config.camera_id,
        PipelineMode.COLLECTION,
        reason=f"baseline_capture:{job_config.session_label}",
    )
    mode_guard.__enter__()
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
                if not waiting_on_lock:
                    logger.info(
                        "baseline.capture_waiting_on_lock",
                        camera_id=job_config.camera_id,
                        collected=collected,
                        target_frames=job_config.target_frames,
                    )
                    waiting_on_lock = True
                emit_progress(progress_pct, "等待异常锁定解除...", force=True)
                while not abort_event.is_set() and time.monotonic() < deadline:
                    time.sleep(2.0)
                    if not camera_manager.is_anomaly_locked(job_config.camera_id):
                        break
                logger.info(
                    "baseline.capture_lock_released",
                    camera_id=job_config.camera_id,
                    collected=collected,
                    target_frames=job_config.target_frames,
                )
                waiting_on_lock = False
                emit_progress(progress_pct, "异常锁定已解除, 继续采集", force=True)
                continue

            # Grab raw frame
            frame = camera_manager.get_raw_frame(job_config.camera_id)
            if frame is None:
                stats.null_frames += 1
                now = time.monotonic()
                if stats.null_frames == 1 or now - last_null_log_time >= 5.0:
                    logger.warning(
                        "baseline.capture_no_frame",
                        camera_id=job_config.camera_id,
                        null_frames=stats.null_frames,
                        collected=collected,
                        target_frames=job_config.target_frames,
                    )
                    last_null_log_time = now
                time.sleep(1.0)
                continue

            stats.total_grabbed += 1

            # Layer 2: Quality filter (includes person detection)
            qr = quality_filter.check(frame, last_saved_frame)
            if not qr.accepted:
                stats.record_rejection(qr.rejection_reason)
                now = time.monotonic()
                if stats.total_rejected == 1 or now - last_rejection_log_time >= 5.0:
                    logger.info(
                        "baseline.capture_filtering",
                        camera_id=job_config.camera_id,
                        rejection_reason=qr.rejection_reason,
                        total_rejected=stats.total_rejected,
                        collected=collected,
                        total_grabbed=stats.total_grabbed,
                    )
                    last_rejection_log_time = now
                time.sleep(sampler.get_sleep_interval())
                continue

            # Sampler strategy filter
            accepted, sample_meta = sampler.should_accept(frame)
            if not accepted:
                time.sleep(sampler.get_sleep_interval())
                continue

            # Save frame
            frame_path = version_dir / f"baseline_{collected:05d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                logger.warning(
                    "baseline.capture_write_failed",
                    camera_id=job_config.camera_id,
                    path=str(frame_path),
                )
                time.sleep(sampler.get_sleep_interval())
                continue

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
            emit_progress(
                progress_pct,
                f"已采集 {collected}/{job_config.target_frames} 帧",
            )

            # Sleep before next grab
            time.sleep(sampler.get_sleep_interval())

    except Exception as exc:
        loop_error = exc
        raise

    finally:
        # Restore pipeline mode FIRST so the camera resumes detection while
        # we do disk I/O for the metadata files.
        mode_guard.__exit__(
            type(loop_error) if loop_error is not None else None,
            loop_error,
            loop_error.__traceback__ if loop_error is not None else None,
        )

        # ── Finalization ─────────────────────────────────────────────────
        # Always write outputs (痛点 7: keep failed runs visible to the user
        # instead of rmtree-ing them silently).
        if loop_error is not None:
            status = "partial" if collected >= min_required_frames else "failed"
            error_message: str | None = str(loop_error)
        elif collected < min_required_frames:
            status = "failed"
            error_message = "insufficient_frames"
        else:
            status = "ok"
            error_message = None

        _write_manifest(version_dir)
        _write_stats(version_dir, stats, collected, start_time)
        _write_job_config(version_dir, job_config)
        _write_capture_meta(
            version_dir, job_config, stats,
            error=error_message, status=status,
        )

        if status == "ok":
            bm.set_current_version(
                job_config.camera_id, "default", version_dir.name
            )
            if lifecycle is not None:
                lifecycle.register_version(
                    job_config.camera_id,
                    "default",
                    version_dir.name,
                    image_count=collected,
                )
        elif status == "failed":
            # Rename failed runs so list_baselines can surface them with
            # a distinct status without polluting the active version pool.
            failed_dir = version_dir.with_name(
                f"failed_{failed_dir_timestamp}_{version_dir.name}"
            )
            try:
                version_dir.rename(failed_dir)
                version_dir = failed_dir
            except OSError as rename_err:
                logger.warning(
                    "baseline.capture_rename_failed",
                    camera_id=job_config.camera_id,
                    src=str(version_dir),
                    error=str(rename_err),
                )

        logger.info(
            "baseline.capture_finalized",
            camera_id=job_config.camera_id,
            version=version_dir.name,
            status=status,
            error=error_message,
            collected=collected,
            target_frames=job_config.target_frames,
            total_grabbed=stats.total_grabbed,
            null_frames=stats.null_frames,
            total_rejected=stats.total_rejected,
            output_dir=str(version_dir),
        )

    if collected < min_required_frames:
        logger.error(
            "baseline.capture_failed",
            camera_id=job_config.camera_id,
            collected=collected,
            min_required_frames=min_required_frames,
            total_grabbed=stats.total_grabbed,
            null_frames=stats.null_frames,
            total_rejected=stats.total_rejected,
            preserved_dir=str(version_dir),
        )
        raise RuntimeError(
            f"采集失败: 仅采集到 {collected} 帧，至少需要 {min_required_frames} 帧。"
            "请检查摄像头实时画面、连接状态或采集间隔设置。"
            f" (失败现场已保留至 {version_dir.name})"
        )

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

    emit_progress(100, f"采集完成: {collected} 帧", force=True)
    logger.info(
        "baseline.capture_completed",
        camera_id=job_config.camera_id,
        version=version_dir.name,
        session_label=job_config.session_label,
        requested_strategy=job_config.sampling_strategy,
        strategy=effective_strategy,
        collected=collected,
        target_frames=job_config.target_frames,
        total_grabbed=stats.total_grabbed,
        null_frames=stats.null_frames,
        total_rejected=stats.total_rejected,
        duration_seconds=round(time.monotonic() - start_time, 1),
        output_dir=str(version_dir),
    )

    return {
        "camera_id": job_config.camera_id,
        "session_label": job_config.session_label,
        "version": version_dir.name,
        "collected_frames": collected,
        "target_frames": job_config.target_frames,
        "requested_strategy": job_config.sampling_strategy,
        "strategy": effective_strategy,
        "strategy_warning": sampler_selection.warning,
        "duration_seconds": round(time.monotonic() - start_time, 1),
        "stats": stats.to_dict(),
        "review": review_result,
        "output_dir": str(version_dir),
    }


# ── Internal helpers ─────────────────────────────────────────────────────


def _create_sampler(job_config: BaselineCaptureJobConfig) -> SamplerSelection:
    """Instantiate the sampling strategy from job config."""
    strategy = job_config.sampling_strategy
    duration_based_interval = max(
        job_config.active_sleep_min_seconds,
        (job_config.duration_hours * 3600) / max(job_config.target_frames, 1),
    )

    if strategy == "uniform":
        return SamplerSelection(
            sampler=UniformSampler(job_config.duration_hours, job_config.target_frames),
            effective_strategy="uniform",
        )

    unavailable_reason = get_active_sampler_unavailable_reason()

    if strategy == "active":
        if unavailable_reason is not None:
            return SamplerSelection(
                sampler=UniformSampler(job_config.duration_hours, job_config.target_frames),
                effective_strategy="uniform",
                warning=(
                    "高级采集依赖不可用"
                    f"（{unavailable_reason}），已自动降级为均匀采集"
                ),
            )

        return SamplerSelection(
            sampler=ActiveSampler(
                diversity_threshold=job_config.diversity_threshold,
                backbone=job_config.dino_backbone,
                image_size=job_config.dino_image_size,
                sleep_interval_seconds=duration_based_interval,
                cpu_threads=job_config.active_cpu_threads,
            ),
            effective_strategy="active",
        )

    if strategy == "scheduled":
        inner = None
        warning = None
        if unavailable_reason is None:
            inner = ActiveSampler(
                diversity_threshold=job_config.diversity_threshold,
                backbone=job_config.dino_backbone,
                image_size=job_config.dino_image_size,
                sleep_interval_seconds=duration_based_interval,
                cpu_threads=job_config.active_cpu_threads,
            )
        else:
            warning = (
                "高级采集依赖不可用"
                f"（{unavailable_reason}），已切换为定时采集（不含多样性筛选）"
            )

        return SamplerSelection(
            sampler=ScheduledSampler(
                schedule_periods=job_config.schedule_periods,
                frames_per_period=job_config.frames_per_period,
                inner_sampler=inner,
                duration_hours=job_config.duration_hours,
            ),
            effective_strategy="scheduled",
            warning=warning,
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


def _write_capture_meta(
    version_dir: Path,
    job_config: BaselineCaptureJobConfig,
    stats: CaptureStats,
    *,
    error: str | None = None,
    status: str = "ok",
) -> Path:
    """Write capture metadata compatible with dashboard reporting.

    The optional ``error`` field carries the failure reason when capture
    aborted abnormally (truncated to 500 chars to keep the meta file small).
    The ``status`` field is one of "ok" | "failed" | "partial".
    """
    meta: dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "camera_id": job_config.camera_id,
        "session_label": job_config.session_label,
        "quality_filter_enabled": job_config.quality_config.enabled,
        "stats": stats.to_dict(),
        "status": status,
    }
    if error is not None:
        meta["error"] = error[:500]
    path = version_dir / "capture_meta.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path
