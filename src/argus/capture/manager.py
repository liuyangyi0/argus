"""Multi-camera thread pool manager.

Orchestrates multiple camera pipelines, each running in its own thread.
Provides a centralized interface for starting, stopping, and monitoring
all cameras, and aggregates alerts from all pipelines.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import structlog

from argus.alerts.grader import Alert, AlertSeverity
from argus.capture.source_runtime import CameraSourceRuntime, SourceCaptureView, create_source_capture
from argus.config.schema import AlertConfig, AnomalyConfig, CameraConfig, ClassifierConfig, CrossCameraConfig, SegmenterConfig
from argus.core.analysis_task_contracts import legacy_task_id
from argus.core.analysis_task_runtime import TaskRuntime, TaskSupervisor
from argus.core.correlation import CameraOverlapPair, CrossCameraCorrelator
from argus.core.error_channel import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    get_error_channel,
)
from argus.core.frame_identity import FrameIdentity
from argus.core.pipeline import DetectionPipeline, PipelineMode, PipelineStats
from argus.core.runner import CameraInferenceRunner, RunnerSnapshot
from argus.core.runtime import (
    CameraRuntime,
    CameraRuntimeAdapter,
    FrameOutcomeKind,
    ModelIdentity,
)

if TYPE_CHECKING:
    from argus.core.health import HealthMonitor
    from argus.storage.audit import AuditLogger
    from argus.storage.database import Database
    from argus.storage.inference_buffer import InferenceBuffer
    from argus.storage.inference_store import InferenceRecordStore
    from argus.storage.models import ModelRecord

logger = structlog.get_logger()


@dataclass
class CameraStatus:
    """Status of a single camera pipeline."""

    camera_id: str
    name: str
    connected: bool = False
    running: bool = False
    stats: PipelineStats | None = None
    capture_status: dict | None = None
    error: str | None = None
    lifecycle: dict | None = None


class _TaskComputeReservation:
    def __init__(self, manager, token):
        self.manager, self.token = manager, token
        self.released = False

    def release(self):
        with self.manager._lock:
            if not self.released:
                self.manager._task_compute_reservations.remove(self.token)
                self.released = True


@dataclass
class _PreparedCameraRelease:
    """Manager-owned runtime transaction kept alive through the DB CAS."""

    manager: "CameraManager"
    camera_id: str
    pipeline: DetectionPipeline
    runner: CameraInferenceRunner | None
    plan: object
    primary_swap: object | None
    shadow_change: bool
    previous_shadow: object | None
    target_shadow: object | None
    previous_identity: ModelIdentity | None
    target_primary_version: str | None
    target_primary_path: str | None
    target_primary_artifact_sha256: str | None
    target_scene_profile_id: str | None
    target_release_revision: int | None
    adopt_primary_identity: bool = False
    committed: bool = False
    identity_adopted: bool = False
    lock_held: bool = False
    shadow_lock_held: bool = False

    def commit(self):
        from argus.storage.model_release_service import RuntimeReleaseCommit

        if self.committed:
            return RuntimeReleaseCommit(False, error="runtime transaction already committed")
        self.pipeline._scene_processing_lock.acquire()
        self.lock_held = True
        self.pipeline._shadow_runner_lock.acquire()
        self.shadow_lock_held = True
        if self.pipeline.get_shadow_runner() is not self.previous_shadow:
            return RuntimeReleaseCommit(False, error="shadow runtime changed after prepare")

        primary_committed = False
        if self.primary_swap is not None:
            if self.pipeline.commit_anomaly_model_release(self.primary_swap) is not True:
                return RuntimeReleaseCommit(False, error="primary standby swap failed")
            primary_committed = True

        try:
            if self.runner is not None and self.adopt_primary_identity:
                self.runner.adopt_prepared_model_identity(
                    version=self.target_primary_version,
                    path=self.target_primary_path,
                    artifact_sha256=self.target_primary_artifact_sha256,
                    scene_profile_id=self.target_scene_profile_id,
                    release_revision=self.target_release_revision,
                )
                self.identity_adopted = True
            shadow_state = None
            if self.shadow_change:
                shadow_state = self.pipeline.set_shadow_runner(self.target_shadow)
            self.committed = True
            return RuntimeReleaseCommit(
                True,
                state={
                    "camera_id": self.camera_id,
                    "primary_swapped": primary_committed,
                    "primary_model_version": self.target_primary_version,
                    "primary_model_path": self.target_primary_path,
                    "shadow": shadow_state,
                },
            )
        except Exception as exc:
            restored_primary = True
            if primary_committed:
                restored_primary = (
                    self.pipeline.abort_anomaly_model_release(self.primary_swap) is True
                )
            if (
                self.identity_adopted
                and self._primary_identity_restored(restored_primary)
                and self.runner is not None
                and self.previous_identity is not None
            ):
                self.runner.adopt_prepared_model_identity(
                    version=self.previous_identity.version,
                    path=self.previous_identity.path,
                    artifact_sha256=self.previous_identity.artifact_sha256,
                    scene_profile_id=self.previous_identity.scene_profile_id,
                    release_revision=self.previous_identity.release_revision,
                )
                self.identity_adopted = False
            return RuntimeReleaseCommit(
                False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def abort(self):
        from argus.storage.model_release_service import RuntimeReleaseCommit

        restored_shadow = True
        restored_primary = True
        try:
            if self.committed and self.shadow_change:
                try:
                    self.pipeline.set_shadow_runner(self.previous_shadow)
                except Exception:
                    restored_shadow = False
            if self.primary_swap is not None:
                restored_primary = (
                    self.pipeline.abort_anomaly_model_release(self.primary_swap) is True
                )
            if (
                self.identity_adopted
                and self._primary_identity_restored(restored_primary)
                and self.runner is not None
                and self.previous_identity is not None
            ):
                self.runner.adopt_prepared_model_identity(
                    version=self.previous_identity.version,
                    path=self.previous_identity.path,
                    artifact_sha256=self.previous_identity.artifact_sha256,
                    scene_profile_id=self.previous_identity.scene_profile_id,
                    release_revision=self.previous_identity.release_revision,
                )
                self.identity_adopted = False
            self.committed = False
        finally:
            self._release_lock()
        success = restored_shadow and restored_primary
        return RuntimeReleaseCommit(
            success,
            state={
                "camera_id": self.camera_id,
                "primary_restored": restored_primary,
                "shadow_restored": restored_shadow,
            },
            error=None if success else "runtime rollback was incomplete",
        )

    def finalize(self):
        from argus.storage.model_release_service import RuntimeReleaseCommit

        self._release_lock()
        return RuntimeReleaseCommit(True, state={"released": True})

    def _primary_identity_restored(self, restored_primary: bool) -> bool:
        if restored_primary:
            return True
        # Scene-reference compensation may fail closed after the detector has
        # already been restored. Keep runner identity aligned with that old
        # artifact even while the scene gate correctly remains blocked.
        checker = getattr(self.pipeline, "is_anomaly_model_release_restored", None)
        return bool(
            self.primary_swap is not None
            and callable(checker)
            and checker(self.primary_swap) is True
        )

    def _release_lock(self) -> None:
        if self.shadow_lock_held:
            self.shadow_lock_held = False
            self.pipeline._shadow_runner_lock.release()
        if self.lock_held:
            self.lock_held = False
            self.pipeline._scene_processing_lock.release()


class CameraManager:
    """Manages multiple camera detection pipelines in a thread pool.

    Each camera gets its own DetectionPipeline running in a dedicated thread.
    Alerts from all cameras are routed through a single callback.

    Usage:
        manager = CameraManager(cameras, alert_config, on_alert=handle_alert)
        manager.start_all()
        # ... system runs ...
        manager.stop_all()
    """

    def __init__(
        self,
        cameras: list[CameraConfig],
        alert_config: AlertConfig,
        on_alert: Callable[[Alert], None] | None = None,
        on_status_change: Callable[[str, dict], None] | None = None,
        cross_camera_config: CrossCameraConfig | None = None,
        segmenter_config: SegmenterConfig | None = None,
        classifier_config: ClassifierConfig | None = None,
        physics_config: object | None = None,
        imaging_config: object | None = None,
        health_monitor: HealthMonitor | None = None,
        audit_logger: AuditLogger | None = None,
        record_store: InferenceRecordStore | None = None,
        database: Database | None = None,
        alert_recording_store: object | None = None,
        event_bus: object | None = None,
        sensor_fusion: object | None = None,
        inference_buffer: InferenceBuffer | None = None,
        source_resolver: Callable[[CameraConfig], CameraConfig] | None = None,
        baselines_dir: str | Path = "data/baselines",
    ):
        self._cameras = cameras
        self._alert_config = alert_config
        self._on_alert = on_alert
        self._on_status_change = on_status_change
        self._source_resolver = source_resolver
        self._baselines_dir = Path(baselines_dir).expanduser()
        self._segmenter_config = segmenter_config
        self._classifier_config = classifier_config
        self._physics_config = physics_config
        self._imaging_config = imaging_config
        self._health_monitor = health_monitor
        self._audit_logger = audit_logger
        self._record_store = record_store
        self._db = database
        self._alert_recording_store = alert_recording_store
        self._event_bus = event_bus
        self._sensor_fusion = sensor_fusion
        self._inference_buffer = inference_buffer
        self._runners: dict[str, CameraInferenceRunner] = {}
        self._pipelines: dict[str, DetectionPipeline] = {}  # backward compat
        self._threads: dict[str, threading.Thread] = {}
        self._sources: dict[str, CameraSourceRuntime] = {}
        self._retiring_sources: dict[str, CameraSourceRuntime] = {}
        self._starting_sources: dict[str, CameraSourceRuntime] = {}
        self._source_runtime_configs: dict[str, CameraConfig] = {}
        self._source_declared_inputs: dict[str, tuple[str, str]] = {}
        self._legacy_task_errors: dict[str, str] = {}
        self._legacy_candidate_sources: dict[str, CameraSourceRuntime] = {}
        self._source_snapshot_identities: dict[str, FrameIdentity] = {}
        self._source_preview_identities: dict[str, FrameIdentity] = {}
        self._source_task_canceller: Callable[[str], bool] | None = None
        self._task_supervisor = TaskSupervisor()
        self._legacy_task_enabled: dict[str, bool] = {}
        self._legacy_task_locks: dict[str, threading.RLock] = {}
        self._starting_cameras: set[str] = set()
        self._starting_generations: dict[str, int] = {}
        self._camera_generations: dict[str, int] = {}
        self._stopping_cameras: set[str] = set()
        self._camera_cleanup_locks: dict[str, threading.RLock] = {}
        self._retiring_cameras: dict[str, tuple[object | None, object | None]] = {}
        self._camera_lifecycle_errors: dict[str, str] = {}
        self._module_desired: dict[str, bool] = {}
        self._module_revision = 0
        self._module_apply_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._alert_count = 0
        self._last_frame_counts: dict[str, int] = {}  # camera_id -> last known frame count

        # Backpressure tracking
        self._pending_frames: dict[str, int] = {}
        self._frames_dropped: dict[str, int] = {}
        self._max_pending: int = 30
        self._bp_lock = threading.Lock()
        self._last_bp_warn: dict[str, float] = {}  # rate-limit warnings

        # M6: Continuous recording manager (injected from __main__.py)
        self._continuous_recording_mgr = None

        # B1-5: Shared anomaly detector for dinomaly_multi_class mode
        self._shared_anomaly_detector = None
        self._shared_detector_lock = threading.Lock()

        # C3: Cross-camera anomaly correlation
        self._correlator: CrossCameraCorrelator | None = None
        self._cross_camera_config = cross_camera_config
        if cross_camera_config and cross_camera_config.enabled and cross_camera_config.overlap_pairs:
            pairs = [
                CameraOverlapPair(
                    camera_a=p.camera_a,
                    camera_b=p.camera_b,
                    homography=p.homography,
                )
                for p in cross_camera_config.overlap_pairs
            ]
            self._correlator = CrossCameraCorrelator(
                pairs=pairs,
                corroboration_threshold=cross_camera_config.corroboration_threshold,
            )
            logger.info(
                "manager.cross_camera_enabled",
                pairs=len(pairs),
                threshold=cross_camera_config.corroboration_threshold,
            )

        # Shared inference executor for all pipelines — sized to camera count
        n_workers = min(max(len(cameras), 2) * 2, os.cpu_count() or 4)
        self._inference_executor = ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="inference"
        )
        # Separate pool for ring-buffer JPEG encoding so cv2.imencode (5-15ms)
        # never competes with YOLO/Anomaly futures for inference worker slots.
        self._encode_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="rb-encode"
        )

        # 5.3: Process-level watchdog
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

        # Source-only operation must never preload any recognition engine.
        # Legacy startup retains the shared model behavior through lazy loading.
        self._shared_yolo = None
        self._shared_yolo_attempted = False
        self._shared_yolo_lock = threading.Lock()

    def _get_legacy_shared_yolo(self, cam_config: CameraConfig):
        with self._shared_yolo_lock:
            if self._shared_yolo_attempted:
                return self._shared_yolo
            self._shared_yolo_attempted = True
            try:
                from argus.person.detector import get_shared_yolo

                cameras = getattr(self, "_cameras", ())
                model_name = (cameras[0] if cameras else cam_config).person_filter.model_name
                self._shared_yolo = get_shared_yolo(model_name)
            except Exception as e:
                logger.warning("manager.shared_yolo_failed", error=str(e))
            return self._shared_yolo

    def _get_shared_detector(self, anomaly_config: AnomalyConfig) -> object:
        """Get or create the shared anomaly detector for multi-class Dinomaly mode.

        Thread-safe: uses a lock to ensure the detector is created only once.
        Returns an AnomalibDetector instance shared across all cameras.
        """
        if self._shared_anomaly_detector is not None:
            return self._shared_anomaly_detector

        with self._shared_detector_lock:
            # Double-check after acquiring lock
            if self._shared_anomaly_detector is not None:
                return self._shared_anomaly_detector

            from argus.anomaly.detector import AnomalibDetector

            # For shared multi-class mode, select an explicitly authorized
            # registry model whenever a database is configured.  Filesystem
            # discovery remains only for isolated legacy/offline managers that
            # have no registry at all.
            model_path = None
            for cam in self._cameras:
                candidate = self._resolve_model_path(cam)
                if candidate is not None:
                    model_path = candidate
                    break

            self._shared_anomaly_detector = AnomalibDetector(
                model_path=model_path,
                threshold=anomaly_config.threshold,
                image_size=anomaly_config.image_size,
                inference_backend=anomaly_config.inference_backend,
                ssim_baseline_frames=anomaly_config.ssim_baseline_frames,
                ssim_sensitivity=anomaly_config.ssim_sensitivity,
                ssim_midpoint=anomaly_config.ssim_midpoint,
                ssim_global_change_suppress_fraction=(
                    anomaly_config.ssim_global_change_suppress_fraction
                ),
                enable_calibration=anomaly_config.enable_calibration,
            )
            logger.info(
                "manager.shared_detector_created",
                model_path=str(model_path),
                model_type=anomaly_config.model_type,
            )
            return self._shared_anomaly_detector

    def _create_shadow_runner(
        self,
        cam_config: CameraConfig,
        shadow_record: ModelRecord | None = None,
        primary_record: ModelRecord | None = None,
    ) -> object | None:
        """Create a ShadowRunner if a shadow-stage model exists for this camera."""
        try:
            db = getattr(self, "_db", None)
            if db is None:
                return None

            if shadow_record is None:
                shadow_record = self._get_shadow_model_record(cam_config.camera_id)
            if shadow_record is None:
                return None

            from argus.anomaly.shadow_runner import ShadowRunner

            if primary_record is None:
                primary_record = self._get_primary_model_record(cam_config.camera_id)
            primary_version = (
                primary_record.model_version_id if primary_record is not None else None
            )

            sample_rate = getattr(cam_config.anomaly, "shadow_sample_rate", 5)

            shadow_model_path = self._resolve_model_record_path(
                shadow_record,
                cam_config.camera_id,
                backend=cam_config.anomaly.inference_backend,
            )
            if shadow_model_path is None or not shadow_model_path.exists():
                logger.warning(
                    "manager.shadow_model_missing",
                    camera_id=cam_config.camera_id,
                    model_path=str(getattr(shadow_record, "model_path", None)),
                    shadow_version=shadow_record.model_version_id,
                )
                return None

            runner = ShadowRunner(
                shadow_model_path=shadow_model_path,
                shadow_version_id=shadow_record.model_version_id,
                production_version_id=primary_version,
                camera_id=cam_config.camera_id,
                session_factory=db.get_session,
                sample_rate=sample_rate,
                threshold=cam_config.anomaly.threshold,
                inference_backend=cam_config.anomaly.inference_backend,
            )
            logger.info(
                "manager.shadow_runner_created",
                camera_id=cam_config.camera_id,
                shadow_version=shadow_record.model_version_id,
            )
            return runner
        except Exception as e:
            logger.warning(
                "manager.shadow_runner_failed",
                camera_id=cam_config.camera_id,
                error=str(e),
            )
            return None

    def start_all(self) -> list[str]:
        """Start all camera pipelines. Returns list of successfully started camera IDs."""
        self._stop_event.clear()
        started = []

        for cam_config in self._cameras:
            if self._start_camera(cam_config):
                started.append(cam_config.camera_id)

        # Keep camera-list and video-wall subscribers in the same lifecycle
        # refresh cycle.  Startup normally happens before the dashboard
        # broadcaster is wired, but this also covers embedded/test runtimes
        # that inject ``on_status_change`` at construction time.
        for camera_id in started:
            self._notify_camera_status(camera_id)
        if started:
            self._broadcast_wall_update()

        # 5.3: Start process-level watchdog thread
        self._start_watchdog()

        logger.info(
            "manager.started",
            total=len(self._cameras),
            started=len(started),
            camera_ids=started,
        )
        return started

    def stop_all(self) -> None:
        """Stop all camera pipelines and wait for threads to finish."""
        logger.info("manager.stopping", cameras=len(self._threads))
        self._stop_watchdog()
        self._stop_event.set()

        with self._lock:
            self._ensure_lifecycle_state()
            camera_ids = (set(self._pipelines) | set(self._threads) | self._starting_cameras
                          | set(self._retiring_cameras) | set(self._sources)
                          | set(self._retiring_sources) | set(self._starting_sources))
        failures = []
        for camera_id in camera_ids:
            try:
                self.stop_camera(camera_id)
            except Exception as exc:
                failures.append(f"{camera_id}: {exc}")
                logger.error("manager.camera_cleanup_failed", camera_id=camera_id, error=str(exc))
        for executor in (self._inference_executor, self._encode_executor):
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                failures.append(f"executor: {exc}")
        with self._lock:
            if self._starting_cameras:
                failures.append("camera initialization is still draining: " + ", ".join(sorted(self._starting_cameras)))
        logger.info("manager.stopped")
        if failures:
            raise RuntimeError("camera cleanup incomplete: " + "; ".join(failures))

    def _ensure_lifecycle_state(self) -> None:
        """Initialize ownership fields for legacy adapters constructed via __new__."""
        for name, factory in (
            ("_starting_cameras", set), ("_starting_generations", dict),
            ("_camera_generations", dict), ("_stopping_cameras", set),
            ("_camera_cleanup_locks", dict),
            ("_retiring_cameras", dict), ("_camera_lifecycle_errors", dict),
            ("_module_desired", dict),
            ("_sources", dict), ("_retiring_sources", dict), ("_starting_sources", dict),
            ("_source_runtime_configs", dict), ("_source_declared_inputs", dict),
            ("_legacy_task_errors", dict),
            ("_legacy_candidate_sources", dict),
            ("_source_snapshot_identities", dict), ("_source_preview_identities", dict),
            ("_legacy_task_enabled", dict), ("_legacy_task_locks", dict),
        ):
            if not hasattr(self, name):
                setattr(self, name, factory())
        if not hasattr(self, "_module_revision"):
            self._module_revision = 0
        if not hasattr(self, "_module_apply_lock"):
            self._module_apply_lock = threading.RLock()
        if not hasattr(self, "_task_supervisor"):
            self._task_supervisor = TaskSupervisor()
        if not hasattr(self, "_shared_yolo_lock"):
            self._shared_yolo_lock = threading.Lock()
            self._shared_yolo_attempted = getattr(self, "_shared_yolo", None) is not None

    def _desired_auxiliary_states(self) -> dict[str, bool]:
        result = {}
        for key, config, field in (
            ("classifier.enabled", self._classifier_config, "enabled"),
            ("segmenter.enabled", self._segmenter_config, "enabled"),
            ("physics.speed_enabled", self._physics_config, "speed_enabled"),
        ):
            result[key] = self._module_desired.get(key, bool(getattr(config, field, False)))
        return result

    def _camera_lifecycle_status_locked(self, camera_id: str) -> dict:
        generation = self._camera_generations.get(camera_id, 0)
        reason = self._camera_lifecycle_errors.get(camera_id)
        starting = camera_id in self._starting_cameras or camera_id in self._starting_sources
        stopping = camera_id in self._stopping_cameras
        retiring = camera_id in self._retiring_cameras or camera_id in self._retiring_sources
        cancelled_start = (camera_id in self._starting_cameras
                           and self._starting_generations.get(camera_id) != generation)
        if reason:
            state = "cleanup_failed"
        elif stopping:
            state = "stopping"
        elif retiring or cancelled_start:
            state = "draining"
        elif starting:
            state = "starting"
        elif camera_id in self._pipelines:
            thread = getattr(self, "_threads", {}).get(camera_id)
            state = "running" if thread is not None and thread.is_alive() else "runtime_dead"
        elif camera_id in self._sources:
            state = "source_only"
        else:
            state = "stopped"
        if reason is None and cancelled_start:
            reason = "cancelled camera initialization is still draining"
        return {
            "state": state, "generation": generation, "reason": reason,
            "starting": starting, "stopping": stopping,
            "cleanup_pending": retiring or stopping or cancelled_start,
            "legacy_error": self._legacy_task_errors.get(camera_id),
        }

    def add_camera_config(self, cam_config: CameraConfig) -> None:
        """Thread-safe addition of a new camera config to the managed list."""
        self._bootstrap_task_identity(cam_config)
        with self._lock:
            self._cameras.append(cam_config)

    def _bootstrap_task_identity(self, cam_config: CameraConfig) -> None:
        """Import identity once, without overwriting approved task snapshots."""
        from argus.storage.analysis_tasks import AnalysisTaskService
        from argus.storage.database import Database

        database = getattr(self, "_db", None)
        if isinstance(database, Database):
            service = AnalysisTaskService(database)
            service.bootstrap_legacy(
                cam_config.camera_id, cam_config.model_dump(mode="json", exclude={"source"}),
            )
            try:
                service.sync_legacy_model_projection(cam_config.camera_id)
            except Exception:
                # This is a non-authoritative read-model projection, not the
                # model release gate. Concurrent publishing may invalidate its
                # snapshot. Keep the existing registry/runner checks in charge
                # and retry projection on the next explicit lifecycle action.
                logger.warning(
                    "manager.analysis_task_projection_unavailable",
                    camera_id=cam_config.camera_id,
                )

    def remove_camera_config(self, camera_id: str) -> None:
        """Thread-safe removal of a camera config from the managed list."""
        with self._lock:
            self._cameras[:] = [c for c in self._cameras if c.camera_id != camera_id]

    def start_camera(self, camera_id: str) -> bool:
        """Start a single camera by ID."""
        cam_config = next((c for c in self._cameras if c.camera_id == camera_id), None)
        if cam_config is None:
            logger.error("manager.camera_not_found", camera_id=camera_id)
            return False
        result = self._start_camera(cam_config)
        if result:
            self._broadcast_camera_and_wall(camera_id)
        return result

    def enable_compute_ownership(self, ownership=None, *, retain_until_exit=False) -> None:
        """Assembly-time, inert machine-local exclusion for managed instances."""
        from argus.runtime.compute_ownership import ComputeOwnership
        with self._lock:
            if retain_until_exit:
                # Legacy background executors do not expose positive cleanup
                # leases yet. Never infer their exit from an empty camera map.
                self._compute_ownership_process_lifetime = True
            if getattr(self, "_compute_ownership", None) is None:
                self._compute_ownership = ownership if ownership is not None else ComputeOwnership()

    def claim_compute_ownership(self) -> bool:
        with self._lock:
            return self._claim_compute_ownership_locked()

    def _claim_compute_ownership_locked(self) -> bool:
        ownership = getattr(self, "_compute_ownership", None)
        return ownership is None or ownership.acquire(self) is True

    def get_training_worker_lock_path(self):
        """Assembly-owned child guard path, never selected by a training request."""
        with self._lock:
            ownership = getattr(self, "_compute_ownership", None)
            return ownership.worker_path if ownership is not None else None

    def is_training_maintenance(self) -> bool:
        with self._lock:
            return getattr(self, "_training_maintenance_mode", False) is True

    def authorize_legacy_compute(self) -> bool:
        """Fence ad-hoc/background legacy compute before it can load a model.

        Its HTTP response does not prove that caches/background work exited, so
        the exclusion is deliberately sticky until a fresh process/manager.
        """
        with self._lock:
            if getattr(self, "_training_maintenance_mode", False):
                return False
            if not self._claim_compute_ownership_locked():
                return False
            self._legacy_compute_authority_used = True
            return True

    def close_compute_ownership(self) -> bool:
        """Release only when no model or pending cleanup still belongs to us."""
        with self._lock:
            self._ensure_lifecycle_state()
            if (getattr(self, "_compute_ownership_process_lifetime", False)
                    or getattr(self, "_legacy_compute_authority_used", False)):
                return False
            if (self._pipelines or self._runners or self._threads or self._starting_cameras
                    or self._retiring_cameras or getattr(self, "_task_compute_reservations", set())
                    or self._task_supervisor.list() or getattr(self, "_shared_anomaly_detector", None) is not None
                    or getattr(self, "_shared_yolo", None) is not None):
                return False
            from argus.runtime.task_training_resources import training_compute_exclusion_required, training_resource_lease_held
            if training_resource_lease_held(self) or training_compute_exclusion_required(None, getattr(self, "_db", None)):
                return False
            ownership = getattr(self, "_compute_ownership", None)
            return ownership is None or ownership.release(self) is True

    def enter_training_maintenance(self) -> bool:
        """Process-sticky exclusion, never an implicit stop or scene approval.

        Source/preview ownership is unchanged. A running, starting, prepared or
        uncleared model prevents entry. Restart with normal config to leave.
        """
        with self._lock:
            self._ensure_lifecycle_state()
            if getattr(self, "_training_maintenance_mode", False):
                return self._claim_compute_ownership_locked()
            if getattr(self, "_legacy_compute_authority_used", False):
                return False
            if (self._pipelines or self._runners or self._threads or self._starting_cameras
                    or self._retiring_cameras or getattr(self, "_task_compute_reservations", set())
                    or self._task_supervisor.list() or getattr(self, "_shared_anomaly_detector", None) is not None
                    or getattr(self, "_shared_yolo", None) is not None):
                return False
            if not self._claim_compute_ownership_locked():
                return False
            self._training_maintenance_mode = True
            return True

    def acquire_task_compute_reservation(self):
        """Exclude maintenance before even a managed model starts loading."""
        with self._lock:
            if getattr(self, "_training_maintenance_mode", False):
                raise RuntimeError("training maintenance excludes model execution")
            if not self._claim_compute_ownership_locked():
                raise RuntimeError("another Argus instance owns compute")
            if not hasattr(self, "_task_compute_reservations"):
                self._task_compute_reservations = set()
            token = object()
            self._task_compute_reservations.add(token)
            return _TaskComputeReservation(self, token)

    def start_source(self, camera_id: str) -> bool:
        """Start acquisition/preview/recording without constructing legacy models."""
        cam_config = self.get_camera_config(camera_id)
        if cam_config is None:
            return False
        result = self._start_source(cam_config)
        if result:
            with self._lock:
                self._legacy_task_enabled.setdefault(camera_id, False)
            self._broadcast_camera_and_wall(camera_id)
        return result

    def stop_source(self, camera_id: str) -> None:
        """Stop this source and its subscribers; retain failed cleanup ownership."""
        self.stop_camera(camera_id)

    def set_source_task_canceller(self, callback: Callable[[str], bool] | None) -> None:
        """Wire prepared/queued managed workers without transferring source ownership."""
        if callback is not None and not callable(callback):
            raise TypeError("source task canceller must be callable")
        with self._lock:
            self._source_task_canceller = callback

    def _resolve_source_config(self, cam_config: CameraConfig) -> CameraConfig:
        if self._source_resolver is not None:
            try:
                resolved = self._source_resolver(cam_config)
                if resolved is not None:
                    return resolved
            except Exception as exc:
                logger.warning("manager.source_resolve_failed", camera_id=cam_config.camera_id,
                               error=str(exc))
        return cam_config

    def _create_source_capture(self, runtime_config, declared_config, stream_id):
        return create_source_capture(runtime_config, source_declared=declared_config.source,
                                     stream_id=stream_id)

    def _start_source(self, cam_config, *, runtime_config=None, expected_generation=None) -> bool:
        camera_id = cam_config.camera_id
        runtime_config = runtime_config or self._resolve_source_config(cam_config)
        # Reconnect must use this source's frozen capture settings, never an
        # in-place edit of the manager/YAML object made after source startup.
        cam_config = cam_config.model_copy(deep=True)
        runtime_config = runtime_config.model_copy(deep=True)
        stream_id = self._frame_identity_stream_id(cam_config, runtime_config)
        with self._lock:
            self._ensure_lifecycle_state()
            previous = self._sources.get(camera_id)
            declared = self._source_declared_inputs.get(camera_id)
        if previous is not None:
            state = previous.state
            with self._lock:
                return (self._sources.get(camera_id) is previous
                        and declared == (cam_config.source, cam_config.protocol)
                        and camera_id not in self._stopping_cameras
                        and camera_id not in self._retiring_sources
                        and (camera_id not in self._retiring_cameras
                             or self._legacy_candidate_sources.get(camera_id) is previous)
                        and not self._stop_event.is_set()
                        and (expected_generation is None
                             or self._camera_generations.get(camera_id, 0) == expected_generation)
                        and state.running and not state.cleanup_pending)
        with self._lock:
            self._ensure_lifecycle_state()
            generation = self._camera_generations.get(camera_id, 0)
            if (self._stop_event.is_set()
                    or expected_generation is not None and expected_generation != generation
                    or camera_id in self._stopping_cameras or camera_id in self._retiring_sources
                    or camera_id in self._retiring_cameras):
                return False
            if camera_id in self._sources or camera_id in self._starting_sources:
                return False

            def on_frame(frame_data):
                # Ownership is independent of legacy generations. A failed or
                # restarted legacy task cannot stop recording for peer tasks.
                with self._lock:
                    current = (self._sources.get(camera_id) is source
                               and camera_id not in self._stopping_cameras)
                if current and self._continuous_recording_mgr is not None:
                    wall_ts = (frame_data.identity.capture_wall_ts
                               if frame_data.identity is not None else time.time())
                    self._continuous_recording_mgr.push_frame(camera_id, frame_data.frame, wall_ts)

            source = CameraSourceRuntime(
                camera_id, lambda: self._create_source_capture(runtime_config, cam_config, stream_id),
                source=runtime_config.source, protocol=runtime_config.protocol, on_frame=on_frame,
            )
            source.configure_frame_identity(source_declared=cam_config.source,
                                            source_runtime=runtime_config.source, stream_id=stream_id)
            self._starting_sources[camera_id] = source
        published = False
        try:
            connected = source.start()
            with self._lock:
                if (connected and self._starting_sources.get(camera_id) is source
                        and self._camera_generations.get(camera_id, 0) == generation
                        and not self._stop_event.is_set()
                        and camera_id not in self._stopping_cameras
                        and camera_id not in self._retiring_sources):
                    self._sources[camera_id] = source
                    self._source_runtime_configs[camera_id] = runtime_config
                    self._source_declared_inputs[camera_id] = (cam_config.source, cam_config.protocol)
                    published = True
            return published
        finally:
            if not published:
                clean = source.stop()
                with self._lock:
                    if not clean:
                        self._retiring_sources[camera_id] = source
                        self._camera_lifecycle_errors[camera_id] = "source cleanup is still pending"
                    elif self._retiring_sources.get(camera_id) is source:
                        self._retiring_sources.pop(camera_id, None)
            with self._lock:
                if self._starting_sources.get(camera_id) is source:
                    self._starting_sources.pop(camera_id, None)

    def stop_camera(self, camera_id: str) -> None:
        """Invalidate pending starts and retain resources until cleanup is confirmed."""
        with self._lock:
            self._ensure_lifecycle_state()
            self._camera_generations[camera_id] = self._camera_generations.get(camera_id, 0) + 1
            if camera_id in self._stopping_cameras:
                raise RuntimeError(f"camera {camera_id} stop is already in progress")
            self._stopping_cameras.add(camera_id)
            cleanup_lock = self._camera_cleanup_locks.setdefault(camera_id, threading.RLock())
        failures = []
        try:
            with cleanup_lock:
                with self._lock:
                    # Refresh after taking cleanup ownership: a cancelled
                    # candidate may have retired itself while we waited.
                    pipeline = self._pipelines.pop(camera_id, None)
                    self._runners.pop(camera_id, None)
                    thread = self._threads.pop(camera_id, None)
                    previous = self._retiring_cameras.get(camera_id)
                    if pipeline is None and previous is not None:
                        pipeline, thread = previous
                    if pipeline is not None or thread is not None:
                        self._retiring_cameras[camera_id] = (pipeline, thread)
                    self._legacy_candidate_sources.pop(camera_id, None)
                    source = self._sources.pop(camera_id, None)
                    if source is None:
                        source = self._starting_sources.get(camera_id)
                    if source is None:
                        source = self._retiring_sources.get(camera_id)
                    if source is not None:
                        self._retiring_sources[camera_id] = source
                    cancel_tasks = getattr(self, "_source_task_canceller", None)
                if cancel_tasks is not None:
                    try:
                        if cancel_tasks(camera_id) is not True:
                            failures.append("managed task preparation cleanup is still pending")
                    except Exception:
                        failures.append("managed task preparation cleanup failed")
                try:
                    if not self._task_supervisor.stop_source(camera_id):
                        failures.append("analysis task cleanup is still pending")
                except Exception as exc:
                    failures.append(f"analysis task cleanup: {exc}")
                if pipeline is not None:
                    try:
                        pipeline.shutdown()
                    except Exception as exc:
                        failures.append(f"shutdown: {exc}")
                if thread is not None:
                    try:
                        if thread is not threading.current_thread() and thread.is_alive():
                            thread.join(timeout=30.0)
                        if thread.is_alive():
                            failures.append("previous camera thread is still running")
                    except Exception as exc:
                        failures.append(f"thread cleanup: {exc}")
                if source is not None:
                    try:
                        if not source.stop():
                            failures.append("source cleanup is still pending")
                    except Exception as exc:
                        failures.append(f"source cleanup: {exc}")
                if not failures:
                    if not self._task_supervisor.unregister_clean_source(camera_id, source=source):
                        failures.append("analysis task ownership cleanup is still pending")
                with self._lock:
                    if failures:
                        self._camera_lifecycle_errors[camera_id] = "; ".join(failures)
                    elif self._retiring_cameras.get(camera_id, (None,))[0] is pipeline:
                        self._retiring_cameras.pop(camera_id, None)
                        self._retiring_sources.pop(camera_id, None)
                        self._source_runtime_configs.pop(camera_id, None)
                        self._source_declared_inputs.pop(camera_id, None)
                        self._source_snapshot_identities.pop(camera_id, None)
                        self._source_preview_identities.pop(camera_id, None)
                        self._camera_lifecycle_errors.pop(camera_id, None)
                        self._legacy_task_errors.pop(camera_id, None)
        finally:
            with self._lock:
                self._stopping_cameras.discard(camera_id)
            self._broadcast_camera_and_wall(camera_id)
        if failures:
            raise RuntimeError(f"camera {camera_id} cleanup incomplete: " + "; ".join(failures))

    def restart_camera(self, camera_id: str) -> bool:
        """Restart only if no later stop superseded this restart's generation."""
        with self._lock:
            self._ensure_lifecycle_state()
            generation = self._camera_generations.get(camera_id, 0)
        self.stop_camera(camera_id)
        cam_config = next((c for c in self._cameras if c.camera_id == camera_id), None)
        if cam_config is None:
            return False
        result = self._start_camera(cam_config, expected_generation=generation + 1)
        if result:
            self._broadcast_camera_and_wall(camera_id)
        return result

    def get_status(self) -> list[CameraStatus]:
        """Get status of all configured cameras."""
        with self._lock:
            self._ensure_lifecycle_state()
            cameras = list(self._cameras)
            pipelines = dict(self._pipelines)
            threads = dict(self._threads)
            sources = dict(self._sources)
            lifecycle = {c.camera_id: self._camera_lifecycle_status_locked(c.camera_id) for c in cameras}
        statuses = []
        for cam_config in cameras:
            pipeline = pipelines.get(cam_config.camera_id)
            if pipeline:
                statuses.append(CameraStatus(
                    camera_id=cam_config.camera_id,
                    name=cam_config.name,
                    connected=pipeline._camera.state.connected,
                    running=bool(
                        threads.get(cam_config.camera_id)
                        and threads[cam_config.camera_id].is_alive()
                    ),
                    lifecycle=lifecycle[cam_config.camera_id],
                    stats=pipeline.stats,
                    capture_status=(
                        pipeline.get_capture_status()
                        if hasattr(pipeline, "get_capture_status")
                        else None
                    ),
                ))
            else:
                source = sources.get(cam_config.camera_id)
                statuses.append(CameraStatus(
                    camera_id=cam_config.camera_id,
                    name=cam_config.name,
                    connected=source.state.connected if source is not None else False,
                    running=source.state.running if source is not None else False,
                    capture_status=source.runtime_status() if source is not None else None,
                    error=(self._camera_lifecycle_errors.get(cam_config.camera_id)
                           or self._legacy_task_errors.get(cam_config.camera_id)),
                    lifecycle=lifecycle[cam_config.camera_id],
                ))
        return statuses

    def serialize_camera_status(self, status: CameraStatus) -> dict:
        """Serialize one camera status for both REST and WebSocket consumers.

        Keeping this shape in the manager prevents the camera list endpoint
        and lifecycle broadcasts from drifting after start/stop/reconnect.
        Route-specific fields such as region labels can still be layered on
        top by the dashboard.
        """
        with self._bp_lock:
            dropped = self._frames_dropped.get(status.camera_id, 0)

        pipeline_mode = self.get_pipeline_mode(status.camera_id)
        if not isinstance(pipeline_mode, str):
            pipeline_mode = None

        health = None
        if self._health_monitor is not None:
            try:
                health = self._health_monitor.get_camera_health(status.camera_id)
            except Exception:
                logger.debug(
                    "manager.camera_health_snapshot_failed",
                    camera_id=status.camera_id,
                    exc_info=True,
                )

        stats = status.stats
        return {
            "camera_id": status.camera_id,
            "name": status.name,
            "connected": status.connected,
            "running": status.running,
            "error": status.error,
            "lifecycle": status.lifecycle,
            "pipeline_mode": pipeline_mode,
            "health": health,
            "capture_status": status.capture_status,
            "stats": {
                "frames_captured": stats.frames_captured,
                "frames_analyzed": stats.frames_analyzed,
                "anomalies_detected": stats.anomalies_detected,
                "alerts_emitted": stats.alerts_emitted,
                "avg_latency_ms": round(stats.avg_latency_ms, 1),
                "frames_dropped": dropped,
            } if stats else None,
        }

    def get_camera_status_payload(self, camera_id: str) -> dict | None:
        """Return the canonical public status payload for one camera."""
        status = next(
            (item for item in self.get_status() if item.camera_id == camera_id),
            None,
        )
        if status is None:
            return None
        return self.serialize_camera_status(status)

    def get_runtime(self, camera_id: str) -> CameraRuntime | None:
        """Return the stable public runtime adapter for one configured camera."""
        configured = self.get_camera_config(camera_id) is not None
        if not configured and camera_id not in self._pipelines and camera_id not in self._runners:
            return None
        return CameraRuntimeAdapter(
            camera_id=camera_id,
            latest_frame=lambda: self.get_latest_frame(camera_id),
            snapshot=lambda: self.get_runner_snapshot(camera_id),
            status=lambda: self.get_camera_status_payload(camera_id),
            scene_guard=lambda: self.get_scene_guard_status(camera_id),
            reconnect=lambda: self.reconnect_camera(camera_id),
        )

    def get_source_runtime_status(self, camera_id: str) -> dict | None:
        """Observe the shared source, independently of any analysis task."""
        with self._lock:
            self._ensure_lifecycle_state()
            source = self._sources.get(camera_id) or self._retiring_sources.get(camera_id)
            configured = any(c.camera_id == camera_id for c in self._cameras)
        if source is not None:
            return source.snapshot()
        if configured:
            return {"state": "stopped", "running": False, "connected": False}
        return None

    def get_camera_source(self, camera_id: str) -> CameraSourceRuntime | None:
        """Return the current shared source for trusted task/collection services.

        This never starts a device. Callers must retain the exact source identity
        and refuse stale results when the camera is stopped or replaced.
        """
        with self._lock:
            self._ensure_lifecycle_state()
            if (camera_id in self._stopping_cameras
                    or camera_id in self._retiring_cameras
                    and self._legacy_candidate_sources.get(camera_id) is not self._sources.get(camera_id)
                    or camera_id in self._retiring_sources):
                return None
            return self._sources.get(camera_id)

    def get_analysis_task_runtime(self, task_id: str) -> dict | None:
        """Return observed task state; registry stage is never runtime evidence."""
        runtime = None
        with self._lock:
            self._ensure_lifecycle_state()
            camera_id = next((c.camera_id for c in self._cameras
                              if legacy_task_id(c.camera_id) == task_id), None)
            if camera_id is None:
                try:
                    runtime = self._task_supervisor.get(task_id)
                except KeyError:
                    return None
            else:
                lifecycle = self._camera_lifecycle_status_locked(camera_id)
                runner = self._runners.get(camera_id)
                enabled = self._legacy_task_enabled.get(camera_id, True)
        # Task callbacks may inspect the manager. Never acquire a task lock
        # while holding the manager lock (task -> source -> manager is valid).
        if camera_id is None:
            return runtime.snapshot() if runtime is not None else None
        running = lifecycle["state"] == "running" and enabled
        state = lifecycle["state"] if enabled else "stopped"
        reason = None
        if runner is None and lifecycle.get("legacy_error"):
            state, reason = "failed", lifecycle["legacy_error"]
        elif state == "source_only":
            state, reason = "stopped", "LEGACY_NOT_STARTED"
        source = self.get_source_runtime_status(camera_id)
        guard = self.get_scene_guard_status(camera_id)
        if running and source and not source.get("connected"):
            state, reason = "blocked", "WAITING_SOURCE"
        elif running and guard and guard.get("enabled") and not guard.get("usable"):
            state, reason = "blocked", "SCENE_BLOCKED"
        identity = runner.model_identity if runner is not None else None
        verified = isinstance(identity, ModelIdentity) and bool(
            identity.version and identity.artifact_sha256
        )
        return {
            "state": state, "running": running,
            "generation": lifecycle["generation"], "reason_code": reason,
            "desired_state": "running" if enabled else "stopped",
            "model_version": identity.version if verified else None,
            "artifact_sha256": identity.artifact_sha256 if verified else None,
            "model_identity_verified": verified,
        }

    def set_analysis_task_enabled(self, task_id: str, enabled: bool) -> bool:
        """Stop/resume the legacy consumer without closing its shared source.

        P2 retains the loaded legacy models while stopped. Draining is serialized
        with frame processing; resuming reconfirms the original scene gate.
        This is not a model release or a way to change pipeline safety policy.
        """
        with self._lock:
            self._ensure_lifecycle_state()
            camera_id = next((c.camera_id for c in self._cameras
                              if legacy_task_id(c.camera_id) == task_id), None)
            if camera_id is None:
                return False
            lock = self._legacy_task_locks.setdefault(camera_id, threading.RLock())
            pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            if not enabled:
                with self._lock:
                    self._legacy_task_enabled[camera_id] = False
                    self._legacy_task_errors.pop(camera_id, None)
                return True
            if not self.start_camera(camera_id):
                return False
            with self._lock:
                pipeline = self._pipelines.get(camera_id)
            if pipeline is None:
                return False
        with lock:
            with self._lock:
                if self._pipelines.get(camera_id) is not pipeline:
                    return False
                previous = self._legacy_task_enabled.get(camera_id, True)
            if enabled and not previous:
                pipeline.reconfirm_task_session()
            with self._lock:
                if self._pipelines.get(camera_id) is not pipeline:
                    return False
                self._legacy_task_enabled[camera_id] = enabled
        self._broadcast_camera_and_wall(camera_id)
        return True

    def register_analysis_task_runtime(self, runtime: TaskRuntime) -> TaskRuntime:
        """Internal adapter seam for bounded consumers, not a public plugin API.

        No deployment authorization or production alert path is created here.
        Only the manager's exact current source can be subscribed to.
        """
        with self._lock:
            self._ensure_lifecycle_state()
            if getattr(self, "_training_maintenance_mode", False):
                raise RuntimeError("training maintenance excludes analysis tasks")
            source = self._sources.get(runtime.source.camera_id)
            if (source is not runtime.source
                    or runtime.source.camera_id in self._stopping_cameras
                    or runtime.source.camera_id in self._retiring_cameras
                    and self._legacy_candidate_sources.get(runtime.source.camera_id) is not source
                    or runtime.source.camera_id in self._retiring_sources):
                raise ValueError("task must use the manager-owned current source")
            if runtime.task_id == legacy_task_id(source.camera_id):
                raise ValueError("legacy task is owned by its compatibility runner")
            return self._task_supervisor.register(runtime)

    def unregister_analysis_task_runtime(self, task_id: str, *, runtime: TaskRuntime) -> bool:
        """Detach only the expected, fully cleaned application task instance."""
        self._ensure_lifecycle_state()
        return self._task_supervisor.unregister_clean_task(task_id, runtime=runtime)

    def set_degradation_publisher(
        self,
        publisher: Callable[[str, dict], None] | None,
    ) -> dict[str, str]:
        """Wire degradation events without exposing pipeline instances.

        The detached failure map lets application bootstrap report a partial
        wiring problem while ``CameraManager`` retains sole ownership of its
        pipeline collection.
        """
        with self._lock:
            pipelines = list(self._pipelines.items())
        failures: dict[str, str] = {}
        for camera_id, pipeline in pipelines:
            try:
                pipeline.set_degradation_publisher(publisher)
            except Exception as exc:
                failures[camera_id] = f"{type(exc).__name__}: {exc}"
        return failures

    def get_pipeline(self, camera_id: str):
        """Get the DetectionPipeline for a camera, or None."""
        return self._pipelines.get(camera_id)

    def update_segmenter_params(
        self,
        *,
        max_points: int | None = None,
        min_anomaly_score: float | None = None,
        min_mask_area_px: int | None = None,
        timeout_seconds: float | None = None,
    ) -> int:
        """Hot-patch SAM2 runtime params across every live pipeline.

        Returns the number of pipelines that had a segmenter attached and
        accepted the update. ``max_points`` and ``min_anomaly_score`` are
        stored on the pipeline object itself (they gate the peak-extraction
        step, which lives in pipeline.py, not in the segmenter class).
        ``min_mask_area_px`` and ``timeout_seconds`` are delegated to
        ``InstanceSegmenter.update_runtime_params`` so the values take
        effect on the very next call. ``model_size`` is intentionally NOT
        tunable at runtime — changing it would require re-downloading the
        checkpoint and can only happen via a pipeline restart.
        """
        updated = 0
        with getattr(self, "_lock", threading.RLock()):
            pipelines = list(self._pipelines.values())
        for pipeline in pipelines:
            segmenter = getattr(pipeline, "_segmenter", None)
            if segmenter is None:
                continue
            try:
                updater = getattr(pipeline, "update_segmenter_params", None)
                if callable(updater):
                    if updater(
                        max_points=max_points, min_anomaly_score=min_anomaly_score,
                        min_mask_area_px=min_mask_area_px, timeout_seconds=timeout_seconds,
                    ) is not True:
                        continue
                else:
                    segmenter.update_runtime_params(
                        min_mask_area_px=min_mask_area_px,
                        timeout_seconds=timeout_seconds,
                    )
                    if max_points is not None:
                        pipeline._segmenter_max_points = int(max_points)
                    if min_anomaly_score is not None:
                        pipeline._segmenter_min_score = float(min_anomaly_score)
                updated += 1
            except Exception:
                logger.warning(
                    "camera_manager.segmenter_params_push_failed",
                    exc_info=True,
                )
        return updated

    def update_cross_camera_pairs(
        self,
        pairs: list[CameraOverlapPair],
        corroboration_threshold: float | None = None,
    ) -> bool:
        """Hot-update cross-camera correlation pairs at runtime.

        If the correlator already exists, delegates to its ``update_pairs``
        method. If no correlator is present but pairs are supplied, creates
        a new one. Returns True when the correlator was updated or created.
        """
        try:
            if self._correlator is not None:
                self._correlator.update_pairs(pairs, corroboration_threshold)
                return True
            if pairs:
                self._correlator = CrossCameraCorrelator(
                    pairs=pairs,
                    corroboration_threshold=corroboration_threshold or 0.3,
                )
                logger.info(
                    "manager.cross_camera_created_runtime",
                    pairs=len(pairs),
                )
                return True
            return False
        except Exception:
            logger.warning(
                "manager.cross_camera_pairs_push_failed", exc_info=True,
            )
            return False

    def update_classifier_vocabulary(self, vocabulary: list[str]) -> int:
        """Hot-swap the classifier vocabulary on every live pipeline.

        Returns the number of pipelines that had a classifier attached and
        accepted the update. Pipelines without a classifier are skipped
        silently. Used by the dashboard "分类器" panel so operators can edit
        the FOE list without restarting the service.
        """
        updated = 0
        with getattr(self, "_lock", threading.RLock()):
            pipelines = list(self._pipelines.values())
        for pipeline in pipelines:
            classifier = getattr(pipeline, "_classifier", None)
            if classifier is None:
                continue
            try:
                updater = getattr(pipeline, "update_classifier_vocabulary", None)
                if callable(updater):
                    if updater(vocabulary) is not True:
                        continue
                else:
                    classifier.update_vocabulary(vocabulary)
                updated += 1
            except Exception:
                logger.warning(
                    "camera_manager.classifier_vocabulary_push_failed",
                    exc_info=True,
                )
        return updated

    def get_latest_frame(self, camera_id: str) -> np.ndarray | None:
        """Get the latest processed frame from a camera for live preview."""
        pipeline = self._pipelines.get(camera_id)
        source = getattr(self, "_sources", {}).get(camera_id)
        if source is not None and (
            pipeline is None or not self._legacy_task_enabled.get(camera_id, True)
        ):
            latest = source.latest_frame
            return latest.frame if latest is not None else None
        if pipeline is None:
            return None
        return pipeline.get_latest_frame()

    def get_latest_frame_with_identity(
        self,
        camera_id: str,
    ) -> tuple[np.ndarray | None, FrameIdentity | None]:
        """Get the latest processed frame and its identity."""
        pipeline = self._pipelines.get(camera_id)
        source = getattr(self, "_sources", {}).get(camera_id)
        if source is not None and (
            pipeline is None or not self._legacy_task_enabled.get(camera_id, True)
        ):
            latest = source.latest_frame
            return (latest.frame, latest.identity) if latest is not None else (None, None)
        if pipeline is None:
            return None, None
        getter = getattr(pipeline, "get_latest_frame_with_identity", None)
        if callable(getter):
            return getter()
        return pipeline.get_latest_frame(), None

    def get_latest_frame_identity(self, camera_id: str) -> FrameIdentity | None:
        """Get identity for the latest processed frame."""
        source = getattr(self, "_sources", {}).get(camera_id)
        pipeline = self._pipelines.get(camera_id)
        if source is not None and (pipeline is None or not self._legacy_task_enabled.get(camera_id, True)):
            latest = source.latest_frame
            return latest.identity if latest is not None else None
        if pipeline is None:
            return None
        get_identity = getattr(pipeline, "get_latest_frame_identity", None)
        if not callable(get_identity):
            return None
        return get_identity()

    def record_snapshot_access(
        self,
        camera_id: str,
        snapshot_id: str,
    ) -> FrameIdentity | None:
        """Mark that a frame was served by the snapshot API."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None or not self._legacy_task_enabled.get(camera_id, True):
            identity = self.get_latest_frame_identity(camera_id)
            if identity is not None:
                identity = identity.with_updates(snapshot_id=snapshot_id)
                with self._lock:
                    self._source_snapshot_identities[camera_id] = identity
            return identity
        mark = getattr(pipeline, "mark_snapshot_identity", None)
        if not callable(mark):
            return None
        return mark(snapshot_id)

    def record_preview_stream_access(self, camera_id: str) -> FrameIdentity | None:
        """Mark that a frame was served by the backend preview stream."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None or not self._legacy_task_enabled.get(camera_id, True):
            identity = self.get_latest_frame_identity(camera_id)
            if identity is not None:
                identity = identity.with_updates(preview_stream_ts=time.time())
                with self._lock:
                    self._source_preview_identities[camera_id] = identity
            return identity
        mark = getattr(pipeline, "mark_preview_stream_identity", None)
        if not callable(mark):
            return None
        return mark()

    def get_frame_identity_status(self, camera_id: str) -> dict | None:
        """Return source and frame identity diagnostics for one camera."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None or not self._legacy_task_enabled.get(camera_id, True):
            source = self.get_camera_source(camera_id)
            if source is None:
                return None
            identity = self.get_latest_frame_identity(camera_id)
            with self._lock:
                snapshot = self._source_snapshot_identities.get(camera_id)
                preview = self._source_preview_identities.get(camera_id)
                declared = self._source_declared_inputs.get(camera_id)
                runtime_config = self._source_runtime_configs.get(camera_id)
            token = source.source_token
            def current(value):
                return value if value is not None and (
                    value.source_session_id, value.source_epoch,
                ) == token else None
            snapshot, preview = current(snapshot), current(preview)
            declared_source = declared[0] if declared else source.source
            return {
                "declared_source": declared_source, "runtime_source": source.source,
                "go2rtc_stream": identity.stream_id if identity is not None else (
                    camera_id if runtime_config is not None and declared
                    and (runtime_config.source, runtime_config.protocol) != declared else None
                ),
                "latest": identity.to_dict() if identity is not None else None,
                "last_snapshot": snapshot.to_dict() if snapshot is not None else None,
                "last_preview_stream": preview.to_dict() if preview is not None else None,
                "last_frame_index": identity.frame_index if identity is not None else None,
                "last_capture_ts": identity.capture_wall_ts if identity is not None else None,
                "last_pipeline_ts": None,
                "last_snapshot_ts": None,
                "last_snapshot_id": snapshot.snapshot_id if snapshot is not None else None,
                "last_snapshot_frame_index": snapshot.frame_index if snapshot is not None else None,
                "last_preview_stream_ts": preview.preview_stream_ts if preview is not None else None,
                "last_preview_stream_frame_index": preview.frame_index if preview is not None else None,
            }
        get_status = getattr(pipeline, "get_frame_identity_status", None)
        if not callable(get_status):
            return None
        return get_status()

    def get_scene_guard_status(self, camera_id: str) -> dict | None:
        """Return the read-only scene guard status for one camera."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        get_status = getattr(pipeline, "get_scene_guard_status", None)
        if not callable(get_status):
            return None
        status = get_status()
        return status if isinstance(status, dict) else None

    def get_managed_scene_guard_status(self, camera_id: str, *, source=None) -> dict:
        """Nonblocking, source-bound safety evidence for managed shadow tasks."""
        blocked = {"enabled": True, "usable": False, "reference_available": False,
                   "profile_id": None, "consumer_enabled": False,
                   "reason_code": "SCENE_EVIDENCE_UNAVAILABLE"}
        # A final task commit may already hold the source lock. Never wait for
        # manager lifecycle or legacy geometry calculations in that callback.
        if not self._lock.acquire(blocking=False):
            return {**blocked, "reason_code": "SCENE_EVIDENCE_BUSY"}
        try:
            current = getattr(self, "_sources", {}).get(camera_id)
            pipeline = self._pipelines.get(camera_id)
            enabled = getattr(self, "_legacy_task_enabled", {}).get(camera_id, True)
            thread = getattr(self, "_threads", {}).get(camera_id)
            consumer = bool(
                enabled and pipeline is not None and thread is not None and thread.is_alive()
                and camera_id not in getattr(self, "_stopping_cameras", ())
                and camera_id not in getattr(self, "_retiring_cameras", ())
                and camera_id not in getattr(self, "_retiring_sources", ())
            )
        finally:
            self._lock.release()
        if not consumer or current is None or (source is not None and source is not current):
            return blocked
        token = current.source_token
        getter = getattr(pipeline, "get_managed_scene_guard_observation", None)
        if token is None or not callable(getter):
            return blocked
        observation = getter()
        if (not isinstance(observation, dict) or observation.get("source_token") != token
                or observation.get("usable") is not True):
            reason = observation.get("reason_code") if isinstance(observation, dict) else None
            if reason in {"SCENE_EVIDENCE_BUSY", "SCENE_EVIDENCE_EXPIRED", "SCENE_PROFILE_UNAPPROVED"}:
                return {**blocked, "reason_code": reason}
            return blocked
        # Copy under the lifecycle lock again; replacement/disable that won
        # during the observation invalidates it. There is still no SDK call.
        if not self._lock.acquire(blocking=False):
            return {**blocked, "reason_code": "SCENE_EVIDENCE_BUSY"}
        try:
            if (getattr(self, "_sources", {}).get(camera_id) is not current
                    or self._pipelines.get(camera_id) is not pipeline
                    or not getattr(self, "_legacy_task_enabled", {}).get(camera_id, True)
                    or getattr(self, "_threads", {}).get(camera_id) is not thread or not thread.is_alive()
                    or camera_id in getattr(self, "_stopping_cameras", ())
                    or camera_id in getattr(self, "_retiring_cameras", ())
                    or camera_id in getattr(self, "_retiring_sources", ())):
                return blocked
        finally:
            self._lock.release()
        return {**observation, "consumer_enabled": True}

    def commit_managed_scene_if_current(self, camera_id: str, *, source, profile_id: str, callback) -> bool:
        """Fence consumer/source ownership and guard evidence through a short memory commit."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            pipeline = self._pipelines.get(camera_id)
            thread = getattr(self, "_threads", {}).get(camera_id)
            if (getattr(self, "_sources", {}).get(camera_id) is not source
                    or pipeline is None or thread is None or not thread.is_alive()
                    or not getattr(self, "_legacy_task_enabled", {}).get(camera_id, True)
                    or camera_id in getattr(self, "_stopping_cameras", ())
                    or camera_id in getattr(self, "_retiring_cameras", ())
                    or camera_id in getattr(self, "_retiring_sources", ())):
                return False
            commit = getattr(pipeline, "commit_managed_scene_guard_observation", None)
            return bool(callable(commit) and commit(source.source_token, profile_id, callback))
        finally:
            self._lock.release()

    def get_active_tracks_snapshot(self, camera_id: str) -> list[dict] | None:
        """Return active physics tracks, or ``None`` when runtime is absent."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        getter = getattr(pipeline, "get_active_tracks_snapshot", None)
        if not callable(getter):
            return []
        tracks = getter()
        return tracks if isinstance(tracks, list) else []

    def get_model_statuses_snapshot(self, camera_id: str | None = None) -> list[dict]:
        """Return detached detector-health records for running cameras."""
        with self._lock:
            pipelines = list(self._pipelines.items())

        records: list[dict] = []
        for current_camera_id, pipeline in pipelines:
            if camera_id is not None and current_camera_id != camera_id:
                continue
            getter = getattr(pipeline, "get_model_statuses_snapshot", None)
            if not callable(getter):
                continue
            try:
                snapshot = getter()
            except Exception:
                logger.warning(
                    "manager.model_status_snapshot_failed",
                    camera_id=current_camera_id,
                    exc_info=True,
                )
                continue
            if isinstance(snapshot, list):
                records.extend(item for item in snapshot if isinstance(item, dict))
        return records

    def get_system_status_snapshots(self) -> list[dict]:
        """Return per-camera System-page diagnostics without exposing pipelines."""
        with self._lock:
            self._ensure_lifecycle_state()
            pipelines = dict(self._pipelines)
            for camera_id, (pipeline, _) in self._retiring_cameras.items():
                pipelines.setdefault(camera_id, pipeline)
            for camera_id in self._starting_cameras:
                pipelines.setdefault(camera_id, None)
            lifecycle = {camera_id: self._camera_lifecycle_status_locked(camera_id) for camera_id in pipelines}

        result: list[dict] = []
        for camera_id, pipeline in pipelines.items():
            getter = getattr(pipeline, "get_system_status_snapshot", None)
            snapshot = None
            try:
                if callable(getter):
                    snapshot = getter()
            except Exception:
                logger.warning(
                    "manager.system_status_snapshot_failed",
                    camera_id=camera_id,
                    exc_info=True,
                )
            if not isinstance(snapshot, dict):
                snapshot = {}
            snapshot = {"camera_id": camera_id, **snapshot, "lifecycle": lifecycle[camera_id]}
            state = lifecycle[camera_id]["state"]
            if state != "running" or not isinstance(snapshot.get("components"), dict):
                components = dict(snapshot.get("components") or {})
                reason = lifecycle[camera_id]["reason"] or f"camera lifecycle is {state}"
                for name in ("classifier", "segmenter"):
                    detail = dict(components.get(name) or {})
                    detail.update(
                        state="failed" if state in {"cleanup_failed", "runtime_dead"} else "loading" if state == "starting" else "draining",
                        ready=False, degraded=True, reason=reason,
                    )
                    detail.setdefault("loaded", False)
                    detail.setdefault("backend", None)
                    detail["busy"] = bool(detail.get("busy")) or state in {"starting", "stopping", "draining"}
                    components[name] = detail
                snapshot["components"] = components
            result.append(snapshot)
        return result

    def get_runtime_component_summary(self) -> dict:
        """Aggregate component attachment/readiness for config status panels."""
        snapshots = self.get_system_status_snapshots()
        with getattr(self, "_lock", threading.RLock()):
            camera_ids = set(self._pipelines)
        reported = {item.get("camera_id") for item in snapshots}
        for camera_id in camera_ids - reported:
            unavailable = {
                "state": "failed", "ready": False, "loaded": False,
                "degraded": True, "busy": False,
                "reason": "runtime status unavailable", "backend": None,
            }
            snapshots.append({
                "camera_id": camera_id,
                "components": {"classifier": unavailable, "segmenter": unavailable},
            })
        components = [
            item.get("components", {})
            for item in snapshots
            if isinstance(item.get("components"), dict)
        ]
        summary = {
            "total_pipelines": len(snapshots),
            "classifier_attached": sum(
                1 for item in components if item.get("classifier_attached") is True
            ),
            "classifier_loaded": sum(
                1 for item in components if item.get("classifier_loaded") is True
            ),
            "segmenter_attached": sum(
                1 for item in components if item.get("segmenter_attached") is True
            ),
            "segmenter_loaded": sum(
                1 for item in components if item.get("segmenter_loaded") is True
            ),
            "imaging_attached": sum(
                1 for item in components if item.get("imaging_attached") is True
            ),
            "correlator_present": self._correlator is not None,
        }
        for name in ("classifier", "segmenter"):
            details = [
                {"camera_id": snapshot["camera_id"], **snapshot["components"][name]}
                for snapshot in snapshots
                if isinstance(snapshot.get("components"), dict)
                and isinstance(snapshot["components"].get(name), dict)
            ]
            summary[f"{name}_details"] = details
            summary[f"{name}_ready"] = sum(item.get("ready") is True for item in details)
            summary[f"{name}_degraded"] = sum(item.get("degraded") is True for item in details)
            summary[f"{name}_failed"] = sum(item.get("state") == "failed" for item in details)
        return summary

    def update_pipeline_thresholds(
        self,
        *,
        camera_ids: list[str] | None = None,
        anomaly_threshold: float | None = None,
        severity_changed: bool = False,
        temporal_changed: bool = False,
        suppression_changed: bool = False,
    ) -> tuple[int, dict[str, bool]]:
        """Apply threshold/cache changes to a stable snapshot of live pipelines."""
        selected = set(camera_ids) if camera_ids is not None else None
        with self._lock:
            pipelines = list(self._pipelines.items())

        updated = 0
        section_hot: dict[str, bool] = {}
        for camera_id, pipeline in pipelines:
            if selected is not None and camera_id not in selected:
                continue
            applied = pipeline.update_thresholds(
                anomaly_threshold=anomaly_threshold,
                severity_changed=severity_changed,
                temporal_changed=temporal_changed,
                suppression_changed=suppression_changed,
            ) or {}
            for key, value in applied.items():
                section_hot[key] = section_hot.get(key, True) and bool(value)
            updated += 1
        return updated, section_hot

    def clear_camera_anomaly_lock(self, camera_id: str) -> bool:
        """Clear one camera's anomaly lock through the manager boundary."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        pipeline.clear_anomaly_lock()
        return True

    def get_camera_zones_snapshot(self, camera_id: str) -> list | None:
        """Return a detached list of the live camera's zone configuration."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return list(pipeline.camera_config.zones)

    def update_camera_zones(self, camera_id: str, zones: list) -> bool:
        """Replace live zones for one camera without exposing its pipeline."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        pipeline.update_zones(list(zones))
        return True

    def apply_runtime_camera_config(self, camera_config: CameraConfig) -> bool:
        """Apply the hot-reloadable subset of one persisted camera config."""
        pipeline = self._pipelines.get(camera_config.camera_id)
        if pipeline is None:
            return False
        pipeline.update_thresholds(
            anomaly_threshold=camera_config.anomaly.threshold,
        )
        if camera_config.zones:
            pipeline.update_zones(camera_config.zones)
        return True

    def reload_module_for_cameras(
        self,
        camera_ids: list[str],
        key: str,
        value: bool,
    ) -> dict[str, int]:
        """Hot-reload one optional module across selected live cameras."""
        with self._lock:
            self._ensure_lifecycle_state()
        with self._module_apply_lock:
            selected = set(camera_ids)
            with self._lock:
                self._module_desired[key] = value
                self._module_revision += 1
                config_section, field = key.split(".", 1)
                config = getattr(self, f"_{config_section}_config", None)
                if config is not None and hasattr(config, field):
                    setattr(config, field, value)
                pipelines = [
                    (camera_id, pipeline, self._camera_generations.get(camera_id, 0))
                    for camera_id, pipeline in self._pipelines.items()
                    if camera_id in selected
                ]
                # A pending start must reconcile before it can publish, but it
                # has not yet proved applied readiness at this request boundary.
                active_ids = {camera_id for camera_id, _, _ in pipelines}
                pending = len(((self._starting_cameras | self._stopping_cameras | set(self._retiring_cameras)) & selected) - active_ids)
            seen = pending
            succeeded = 0
            failed = pending
            for camera_id, pipeline, generation in pipelines:
                seen += 1
                try:
                    applied = pipeline.reload_module(key, value) is True
                    with self._lock:
                        still_current = (
                            self._pipelines.get(camera_id) is pipeline
                            and self._camera_generations.get(camera_id, 0) == generation
                        )
                    if applied and still_current:
                        succeeded += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                    logger.warning(
                        "manager.module_hot_reload_failed",
                        camera_id=camera_id,
                        key=key,
                        exc_info=True,
                    )
            return {"seen": seen, "succeeded": succeeded, "failed": failed, "pending": pending}

    def get_runtime_model_state(self, camera_id: str) -> dict | None:
        """Return the primary/shadow model versions applied by a live runtime."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        getter = getattr(pipeline, "get_runtime_model_state", None)
        if not callable(getter):
            return None
        state = getter()
        if not isinstance(state, dict):
            return None
        result = dict(state)
        identity = self.get_model_slot_identity(camera_id)
        if identity is not None:
            result.update({
                "runtime_artifact_sha256": identity.artifact_sha256,
                "scene_profile_id": identity.scene_profile_id,
                "release_revision": identity.release_revision,
            })
        return result

    def get_anomaly_backend_status(self, camera_id: str) -> dict | None:
        """Return configured and applied anomaly backend names."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        getter = getattr(pipeline, "get_anomaly_backend_status", None)
        if not callable(getter):
            return None
        status = getter()
        return dict(status) if isinstance(status, dict) else None

    def get_anomaly_inference_handle(self, camera_id: str) -> object | None:
        """Return the runtime's opaque anomaly detector for explicit jobs."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        getter = getattr(pipeline, "get_anomaly_inference_handle", None)
        return getter() if callable(getter) else None

    def get_wall_status(self, camera_id: str) -> dict | None:
        """Return the read-only video-wall inference snapshot for one camera."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        getter = getattr(pipeline, "get_wall_status", None)
        if not callable(getter):
            return None
        status = getter()
        return dict(status) if isinstance(status, dict) else None

    def get_raw_frame(self, camera_id: str) -> np.ndarray | None:
        """Get unmasked raw frame for baseline capture."""
        source = self.get_camera_source(camera_id)
        if source is not None:
            latest = source.latest_frame
            return latest.frame if latest is not None else None
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_raw_frame()

    def reconnect_camera(self, camera_id: str) -> bool:
        """Request capture reconnect through a public manager operation."""
        source = getattr(self, "_sources", {}).get(camera_id)
        if source is not None:
            source.request_reconnect()
            return True
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        camera = getattr(pipeline, "_camera", None)
        if camera is None:
            return False
        try:
            request_reconnect = getattr(camera, "request_reconnect", None)
            if callable(request_reconnect):
                request_reconnect()
            else:
                reconnect = getattr(camera, "reconnect", None)
                if not callable(reconnect) or reconnect() is False:
                    return False
            self._broadcast_camera_and_wall(camera_id)
            return True
        except Exception:
            logger.warning(
                "manager.camera_reconnect_failed",
                camera_id=camera_id,
                exc_info=True,
            )
            return False

    def set_source_resolver(
        self,
        source_resolver: Callable[[CameraConfig], CameraConfig] | None,
    ) -> None:
        """Set the runtime source resolver used when starting pipelines."""
        self._source_resolver = source_resolver

    def get_camera_config(self, camera_id: str) -> CameraConfig | None:
        """Return the persisted/original camera config, not the runtime copy."""
        return next((c for c in self._cameras if c.camera_id == camera_id), None)

    def is_anomaly_locked(self, camera_id: str) -> bool:
        """Check if a camera's pipeline has an active anomaly lock."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        with pipeline._lock_state_lock:
            return pipeline._locked

    def set_pipeline_mode(self, camera_id: str, mode: PipelineMode) -> bool:
        """Set pipeline operating mode for a camera (DET-006)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        pipeline.set_mode(mode)
        return True

    def get_pipeline_mode(self, camera_id: str) -> str | None:
        """Get pipeline operating mode for a camera (DET-006)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.mode.value

    def list_camera_ids(self) -> list[str]:
        """List all camera IDs that currently have a pipeline registered."""
        return list(self._pipelines.keys())

    def get_all_pipeline_modes(self) -> dict[str, str]:
        """Return a snapshot of every camera's current pipeline mode."""
        return {cid: p.mode.value for cid, p in self._pipelines.items()}

    def set_all_pipeline_modes(self, mode: PipelineMode) -> dict[str, str]:
        """Switch every registered pipeline into the given mode.

        Returns a {camera_id: previous_mode} mapping useful for restoration
        by GlobalPipelineModeGuard.
        """
        previous: dict[str, str] = {}
        for cid, pipeline in self._pipelines.items():
            previous[cid] = pipeline.mode.value
            pipeline.set_mode(mode)
        return previous

    def get_learning_progress(self, camera_id: str) -> dict | None:
        """Get learning mode progress for a camera (DET-010)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_learning_progress()

    def get_diagnostics(self, camera_id: str, n: int = 50) -> list | None:
        """Get recent frame diagnostics for a camera (DET-008)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_diagnostics_buffer().get_recent(n)

    def evaluate_threshold(self, camera_id: str, threshold: float) -> dict | None:
        """Evaluate how a new threshold would affect alert counts (DET-005)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_diagnostics_buffer().evaluate_threshold(threshold)

    def get_detector_status(self, camera_id: str) -> dict | None:
        """Get anomaly detector status for a camera (DET-004)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        status = pipeline.get_detector_status()
        result = {
            "mode": status.mode,
            "model_path": status.model_path,
            "model_loaded": status.model_loaded,
            "threshold": status.threshold,
            "ssim_calibration_progress": status.ssim_calibration_progress,
            "ssim_calibrated": status.ssim_calibrated,
            "ssim_noise_floor": status.ssim_noise_floor,
        }
        quality_getter = getattr(pipeline, "get_input_quality_status", None)
        if callable(quality_getter):
            result.update(quality_getter())
        return result

    def reload_model(
        self,
        camera_id: str,
        model_path: str,
        *,
        version_tag: str | None = None,
    ) -> bool:
        """Fail closed for the retired path-based model mutation API.

        A caller-controlled path is not a release identity. Runtime mutation is
        only permitted through :meth:`prepare_model_release`, which is invoked
        by ``ModelReleaseService`` after registry, artifact, revision and scene
        profile validation.
        """
        current_identity = self.get_model_slot_identity(camera_id)
        current_version = (
            current_identity.version if current_identity is not None else None
        )
        logger.error(
            "manager.legacy_model_reload_blocked",
            camera_id=camera_id,
            attempted_version=version_tag,
            current_version=current_version,
            requested_artifact_name=Path(model_path).name,
        )
        self._broadcast_activation_failed(
            camera_id=camera_id,
            attempted_version=version_tag,
            current_version=current_version,
            reason="release_service_required",
        )
        return False

    def prepare_model_release(self, plan):
        """Prepare an explicit registry release without reading desired stage from DB.

        The plan is already policy-validated and carries the artifact hash.  No
        directory discovery is permitted here: a registry row either names the
        exact artifact or a legacy directory with the deterministic root-level
        ``model.pt``/``model.xml`` for the configured backend.
        """

        from argus.anomaly.shadow_runner import ShadowRunner
        from argus.preprocessing.scene_profile import runtime_artifact_sha256
        from argus.storage.model_registry import validate_persisted_release_evidence
        from argus.storage.model_release_service import (
            RuntimeReleaseIdentity,
            RuntimeReleasePreparation,
        )
        from argus.storage.models import ModelStage

        def failed(error: str, *, state: object = None):
            return RuntimeReleasePreparation(
                success=False,
                state=state,
                error=error,
            )

        camera_id = getattr(plan, "camera_id", None)
        model_version_id = getattr(plan, "model_version_id", None)
        target_stage = getattr(plan, "target_stage", None)
        from_stage = getattr(plan, "from_stage", None)
        artifact_sha256 = getattr(plan, "runtime_artifact_sha256", None)
        if not all(isinstance(value, str) and value for value in (
            camera_id,
            model_version_id,
            target_stage,
            from_stage,
        )):
            return failed("release plan identity is incomplete")

        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return failed("camera runtime is not running")
        cam_config = self.get_camera_config(camera_id)
        if cam_config is None:
            return failed("camera configuration is unavailable")
        if self._db is None:
            return failed("registry database is unavailable")

        def exact_path(raw_path: object, expected_hash: str | None) -> Path:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("registry model_path is missing")
            registry_path = Path(raw_path).expanduser()
            backend = cam_config.anomaly.inference_backend
            path = registry_path
            if registry_path.is_dir():
                # Backward-compatible registry directories use deterministic
                # artifact names. This is not discovery: no glob/rglob/mtime
                # selection is allowed, and ambiguity fails closed below.
                if backend in {"torch-cuda", "torch-cpu"}:
                    path = registry_path / "model.pt"
                elif backend == "openvino":
                    path = registry_path / "model.xml"
                else:
                    candidates = [
                        candidate
                        for candidate in (
                            registry_path / "model.pt",
                            registry_path / "model.xml",
                        )
                        if candidate.is_file()
                    ]
                    if len(candidates) != 1:
                        raise ValueError(
                            "auto backend registry directory must contain exactly one "
                            "deterministic model.pt/model.xml"
                        )
                    path = candidates[0]
            if not path.is_file():
                raise ValueError("registry artifact file does not exist")
            suffix = path.suffix.lower()
            if backend in {"torch-cuda", "torch-cpu"} and suffix != ".pt":
                raise ValueError(f"{backend} requires an exact model.pt path")
            if backend == "openvino" and suffix != ".xml":
                raise ValueError("openvino requires an exact model.xml path")
            if suffix not in {".pt", ".xml"}:
                raise ValueError("registry artifact is not deployable")
            registry_hash = runtime_artifact_sha256(registry_path)
            artifact_hash = runtime_artifact_sha256(path)
            if (
                expected_hash is None
                or registry_hash != expected_hash
                or artifact_hash != expected_hash
            ):
                raise ValueError("registry artifact hash changed after policy validation")
            return path.resolve()

        previous_shadow = pipeline.get_shadow_runner()
        runner = self._runners.get(camera_id)
        typed_runner = runner if isinstance(runner, CameraInferenceRunner) else None
        current_scene_status = pipeline.get_scene_guard_status()
        previous_identity = (
            typed_runner.model_identity
            if typed_runner is not None
            else ModelIdentity(
                version=getattr(pipeline, "_model_version_id", None),
                path=(
                    str(getattr(pipeline, "_anomaly_model_path"))
                    if getattr(pipeline, "_anomaly_model_path", None) is not None
                    else None
                ),
                artifact_sha256=getattr(
                    pipeline,
                    "_model_artifact_sha256",
                    None,
                ),
                scene_profile_id=(
                    current_scene_status.get("profile_id")
                    if isinstance(current_scene_status, dict)
                    else None
                ),
            )
        )

        primary_swap = None
        target_primary_version = previous_identity.version
        target_primary_path = previous_identity.path
        target_primary_artifact_sha256 = previous_identity.artifact_sha256
        target_scene_profile_id = previous_identity.scene_profile_id
        target_release_revision = previous_identity.release_revision
        adopt_primary_identity = False
        primary_target_record = None
        if target_stage in {ModelStage.CANARY.value, ModelStage.PRODUCTION.value}:
            try:
                target_path = exact_path(getattr(plan, "model_path", None), artifact_sha256)
            except (OSError, ValueError) as exc:
                return failed(str(exc))
            target_primary_version = model_version_id
            target_primary_path = str(target_path)
            target_primary_artifact_sha256 = artifact_sha256
            target_release_revision = int(getattr(plan, "expected_revision", 0)) + 1
            adopt_primary_identity = True
            current_hash = getattr(pipeline, "_model_artifact_sha256", None)
            already_applied = (
                previous_identity.version == model_version_id
                and previous_identity.path is not None
                and Path(previous_identity.path).resolve() == target_path
                and current_hash == artifact_sha256
            )
            if not already_applied:
                try:
                    primary_swap = pipeline.prepare_anomaly_model_release(
                        target_path,
                        version_tag=model_version_id,
                        expected_artifact_sha256=artifact_sha256,
                    )
                except Exception as exc:
                    return failed(
                        f"primary runtime prepare failed: {type(exc).__name__}: {exc}"
                    )
        elif (
            from_stage == ModelStage.CANARY.value
            and previous_identity.version == model_version_id
        ):
            # Demoting/retiring a canary must prepare the still-authorized
            # production model before the canary disappears from routing.
            primary_target_record = self._get_active_model_record(camera_id)
            if primary_target_record is None:
                return failed("no active production model is available for canary rollback")
            validation = validate_persisted_release_evidence(primary_target_record)
            if not validation.valid:
                return failed(
                    "fallback production evidence rejected: "
                    + "; ".join(validation.reasons)
                )
            try:
                fallback_hash = runtime_artifact_sha256(primary_target_record.model_path)
                target_path = exact_path(primary_target_record.model_path, fallback_hash)
                primary_swap = pipeline.prepare_anomaly_model_release(
                    target_path,
                    version_tag=primary_target_record.model_version_id,
                    expected_artifact_sha256=fallback_hash,
                )
            except Exception as exc:
                return failed(
                    f"production rollback prepare failed: {type(exc).__name__}: {exc}"
                )
            target_primary_version = primary_target_record.model_version_id
            target_primary_path = str(target_path)
            target_primary_artifact_sha256 = fallback_hash
            target_release_revision = int(
                getattr(primary_target_record, "release_revision", 0) or 0
            )
            adopt_primary_identity = True
        elif (
            from_stage == ModelStage.PRODUCTION.value
            and target_stage == ModelStage.RETIRED.value
        ):
            return failed(
                "active production cannot be retired without a prepared replacement"
            )

        shadow_change = False
        target_shadow = previous_shadow
        if target_stage == ModelStage.SHADOW.value:
            try:
                shadow_path = exact_path(getattr(plan, "model_path", None), artifact_sha256)
                target_shadow = ShadowRunner(
                    shadow_model_path=shadow_path,
                    shadow_version_id=model_version_id,
                    production_version_id=target_primary_version,
                    camera_id=camera_id,
                    session_factory=self._db.get_session,
                    sample_rate=getattr(cam_config.anomaly, "shadow_sample_rate", 5),
                    threshold=cam_config.anomaly.threshold,
                    inference_backend=cam_config.anomaly.inference_backend,
                )
                if target_shadow.prepare() is not True:
                    raise RuntimeError("shadow standby did not load")
                if runtime_artifact_sha256(shadow_path) != artifact_sha256:
                    raise RuntimeError("shadow artifact changed during prepare")
            except Exception as exc:
                if primary_swap is not None:
                    pipeline.abort_anomaly_model_release(primary_swap)
                return failed(
                    f"shadow runtime prepare failed: {type(exc).__name__}: {exc}"
                )
            shadow_change = (
                getattr(previous_shadow, "shadow_version_id", None)
                != model_version_id
            )
        elif (
            from_stage == ModelStage.SHADOW.value
            and getattr(previous_shadow, "shadow_version_id", None) == model_version_id
        ):
            target_shadow = None
            shadow_change = previous_shadow is not None

        if adopt_primary_identity:
            prepared_profile_id = getattr(
                primary_swap,
                "target_scene_profile_id",
                None,
            )
            if prepared_profile_id is not None:
                target_scene_profile_id = str(prepared_profile_id)
            else:
                scene_status = pipeline.get_scene_guard_status()
                target_scene_profile_id = (
                    scene_status.get("profile_id")
                    if isinstance(scene_status, dict)
                    else None
                )

        transaction = _PreparedCameraRelease(
            manager=self,
            camera_id=camera_id,
            pipeline=pipeline,
            runner=typed_runner,
            plan=plan,
            primary_swap=primary_swap,
            shadow_change=shadow_change,
            previous_shadow=previous_shadow,
            target_shadow=target_shadow,
            previous_identity=previous_identity,
            target_primary_version=target_primary_version,
            target_primary_path=target_primary_path,
            target_primary_artifact_sha256=target_primary_artifact_sha256,
            target_scene_profile_id=target_scene_profile_id,
            target_release_revision=target_release_revision,
            adopt_primary_identity=adopt_primary_identity,
        )
        return RuntimeReleasePreparation(
            success=True,
            identity=RuntimeReleaseIdentity(
                camera_id=camera_id,
                model_version_id=model_version_id,
                target_stage=target_stage,
                runtime_artifact_sha256=artifact_sha256,
            ),
            transaction=transaction,
            state={
                "camera_id": camera_id,
                "primary_prepared": primary_swap is not None,
                "primary_model_version": target_primary_version,
                "primary_model_path": target_primary_path,
                "shadow_prepared": target_shadow is not None and shadow_change,
                "shadow_change": shadow_change,
            },
        )

    def apply_model_release_state(
        self,
        camera_id: str,
        reason: str = "release_transition",
    ) -> dict:
        """Fail closed for the retired registry-to-runtime synchronization API.

        Reading a mutable registry row and then reloading runtime in a separate
        step cannot preserve desired/applied atomicity. ``ModelReleaseService``
        owns the only supported prepare/commit/rollback path.
        """
        logger.error(
            "manager.legacy_release_sync_blocked",
            camera_id=camera_id,
            reason=reason,
        )
        return {
            "camera_id": camera_id,
            "reason": reason,
            "running": camera_id in self._pipelines,
            "applied": False,
            "primary_model_version": None,
            "primary_model_stage": None,
            "primary_model_path": None,
            "primary_reloaded": False,
            "shadow_previous_version": None,
            "shadow_model_version": None,
            "shadow_attached": False,
            "shadow_detached": False,
            "errors": ["release_service_required"],
        }

    def _broadcast_activation_failed(
        self,
        *,
        camera_id: str,
        attempted_version: str | None,
        current_version: str | None,
        reason: str,
        model_path: str | None = None,
    ) -> None:
        """Broadcast a ``model.activation_failed`` event to WebSocket clients."""
        if not self._on_status_change:
            return
        payload: dict = {
            "event": "model.activation_failed",
            "camera_id": camera_id,
            "attempted_version": attempted_version,
            "current_version": current_version,
            "reason": reason,
        }
        if model_path is not None:
            payload["model_path"] = model_path
        try:
            self._on_status_change("models", payload)
        except Exception:
            logger.debug("manager.activation_failed_broadcast_failed", exc_info=True)

    def get_latest_anomaly_map(self, camera_id: str) -> np.ndarray | None:
        """Get the latest anomaly heatmap from a camera for overlay stream."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_latest_anomaly_map()

    def get_drift_status(self, camera_id: str) -> dict | None:
        """Get drift detection status for a camera."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        status = pipeline.get_drift_status()
        if status is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "is_drifted": status.is_drifted,
            "ks_statistic": round(status.ks_statistic, 4),
            "p_value": round(status.p_value, 6),
            "reference_mean": round(status.reference_mean, 4),
            "current_mean": round(status.current_mean, 4),
            "samples_collected": status.samples_collected,
        }

    def get_backpressure_stats(self) -> dict[str, dict]:
        """Get per-camera backpressure statistics."""
        with self._bp_lock:
            return {
                cam_id: {
                    "pending": self._pending_frames.get(cam_id, 0),
                    "dropped": self._frames_dropped.get(cam_id, 0),
                    "backpressured": self._pending_frames.get(cam_id, 0) >= self._max_pending,
                }
                for cam_id in [c.camera_id for c in self._cameras]
            }

    def _broadcast_wall_update(self) -> None:
        """Broadcast full wall status to all WebSocket subscribers.

        Called after camera start/stop so the dashboard updates immediately
        instead of waiting for the next poll cycle.
        """
        if not self._on_status_change:
            return
        try:
            cameras = []
            for cam_config in self._cameras:
                pipeline = self._pipelines.get(cam_config.camera_id)
                is_online = pipeline is not None and cam_config.camera_id in self._threads
                tile: dict = {
                    "camera_id": cam_config.camera_id,
                    "name": cam_config.name,
                    "status": "online" if is_online else "offline",
                    "fps": None,
                    "current_score": 0.0,
                    "score_sparkline": [],
                }
                if pipeline is not None and hasattr(pipeline, "get_wall_status"):
                    wall_data = pipeline.get_wall_status()
                    tile["current_score"] = wall_data.get("current_score", 0.0)
                    tile["score_sparkline"] = wall_data.get("score_sparkline", [])
                stats = pipeline.stats if pipeline else None
                if stats is not None:
                    tile["fps"] = getattr(stats, "current_fps", None)
                cameras.append(tile)
            self._on_status_change("wall", {"cameras": cameras})
        except Exception:
            logger.debug("manager.wall_broadcast_failed", exc_info=True)

    def _broadcast_camera_and_wall(self, camera_id: str) -> None:
        """Publish one lifecycle refresh to both camera consumers."""
        self._notify_camera_status(camera_id)
        self._broadcast_wall_update()

    def _notify_camera_status(self, camera_id: str) -> None:
        """Notify WebSocket subscribers of camera status change."""
        if not self._on_status_change:
            return
        try:
            payload = self.get_camera_status_payload(camera_id)
            if payload is None:
                return
            self._on_status_change("cameras", payload)
        except Exception as e:
            logger.debug("manager.notify_failed", error=str(e))

    def _get_active_model_record(self, camera_id: str):
        db = getattr(self, "_db", None)
        if db is None:
            return None
        try:
            from argus.storage.model_registry import ModelRegistry

            registry = ModelRegistry(session_factory=db.get_session)
            return registry.get_active(camera_id)
        except Exception as e:
            logger.warning("manager.active_model_lookup_failed", camera_id=camera_id, error=str(e))
            return None

    def _get_primary_model_record(self, camera_id: str):
        db = getattr(self, "_db", None)
        if db is None:
            return None
        try:
            from argus.anomaly.model_router import ModelRouter

            return ModelRouter(session_factory=db.get_session).get_model_for_camera(camera_id)
        except Exception as e:
            logger.warning("manager.primary_model_lookup_failed", camera_id=camera_id, error=str(e))
            return self._get_active_model_record(camera_id)

    def _get_shadow_model_record(self, camera_id: str):
        db = getattr(self, "_db", None)
        if db is None:
            return None
        try:
            from argus.anomaly.model_router import ModelRouter

            return ModelRouter(session_factory=db.get_session).get_shadow_model_for_camera(camera_id)
        except Exception as e:
            logger.warning("manager.shadow_model_lookup_failed", camera_id=camera_id, error=str(e))
            return None

    def _has_registered_model_records(self, camera_id: str) -> bool:
        db = getattr(self, "_db", None)
        if db is None:
            return False
        try:
            from argus.anomaly.model_router import ModelRouter

            return ModelRouter(session_factory=db.get_session).has_model_records(camera_id)
        except Exception as e:
            logger.warning("manager.model_records_lookup_failed", camera_id=camera_id, error=str(e))
            return False

    def _resolve_model_record_path(
        self,
        record,
        camera_id: str,
        backend: str = "auto",
    ) -> Path | None:
        """Resolve exactly the runtime artifact authorized by one registry row.

        Registry directories are compatibility containers, not discovery roots.
        Only a deterministic root-level ``model.pt``/``model.xml`` may be
        selected, and both the container and selected artifact must retain the
        SHA-256 persisted by the applied release operation (or, for legacy
        production rows, the validated production release evidence).
        """
        model_path_raw = getattr(record, "model_path", None)
        record_camera_id = getattr(record, "camera_id", None)
        if not model_path_raw or record_camera_id != camera_id:
            logger.error(
                "manager.registry_model_identity_invalid",
                camera_id=camera_id,
                record_camera_id=record_camera_id,
                model_version=getattr(record, "model_version_id", None),
            )
            return None

        registry_path = Path(model_path_raw).expanduser()
        artifact_path: Path | None = None
        if registry_path.is_file():
            artifact_path = registry_path
        elif registry_path.is_dir():
            if backend in {"torch-cuda", "torch-cpu"}:
                candidate = registry_path / "model.pt"
                artifact_path = candidate if candidate.is_file() else None
            elif backend == "openvino":
                candidate = registry_path / "model.xml"
                artifact_path = candidate if candidate.is_file() else None
            elif backend == "auto":
                candidates = [
                    candidate
                    for candidate in (
                        registry_path / "model.pt",
                        registry_path / "model.xml",
                    )
                    if candidate.is_file()
                ]
                artifact_path = candidates[0] if len(candidates) == 1 else None
            else:
                logger.error(
                    "manager.registry_model_backend_invalid",
                    camera_id=camera_id,
                    backend=backend,
                )
                return None

        if artifact_path is None:
            logger.error(
                "manager.registry_artifact_not_exact",
                camera_id=camera_id,
                model_version=getattr(record, "model_version_id", None),
                registry_path=str(registry_path),
                backend=backend,
            )
            return None

        suffix = artifact_path.suffix.lower()
        if (
            (backend in {"torch-cuda", "torch-cpu"} and suffix != ".pt")
            or (backend == "openvino" and suffix != ".xml")
            or suffix not in {".pt", ".xml"}
        ):
            logger.error(
                "manager.registry_artifact_backend_mismatch",
                camera_id=camera_id,
                model_version=getattr(record, "model_version_id", None),
                artifact_path=str(artifact_path),
                backend=backend,
            )
            return None

        expected_sha256 = self._persisted_runtime_artifact_sha256(record)
        if expected_sha256 is None:
            logger.error(
                "manager.registry_artifact_authority_missing",
                camera_id=camera_id,
                model_version=getattr(record, "model_version_id", None),
                stage=getattr(record, "stage", None),
            )
            return None

        from argus.preprocessing.scene_profile import runtime_artifact_sha256

        try:
            registry_sha256 = runtime_artifact_sha256(registry_path)
            artifact_sha256 = runtime_artifact_sha256(artifact_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error(
                "manager.registry_artifact_fingerprint_failed",
                camera_id=camera_id,
                model_version=getattr(record, "model_version_id", None),
                error=str(exc),
            )
            return None
        if registry_sha256 != expected_sha256 or artifact_sha256 != expected_sha256:
            logger.error(
                "manager.registry_artifact_fingerprint_mismatch",
                camera_id=camera_id,
                model_version=getattr(record, "model_version_id", None),
                expected_sha256=expected_sha256,
                registry_sha256=registry_sha256,
                artifact_sha256=artifact_sha256,
            )
            return None
        return artifact_path.resolve()

    def _persisted_runtime_artifact_sha256(self, record) -> str | None:
        """Return the immutable runtime digest persisted with applied state."""
        from argus.preprocessing.scene_profile import normalize_runtime_artifact_sha256
        from argus.storage.model_registry import validate_persisted_release_evidence
        from argus.storage.models import (
            ModelReleaseOperation,
            ModelStage,
            ReleaseOperationStatus,
        )

        evidence_sha256: str | None = None
        if getattr(record, "stage", None) == ModelStage.PRODUCTION.value:
            validation = validate_persisted_release_evidence(record)
            if not validation.valid:
                logger.error(
                    "manager.production_release_evidence_rejected",
                    camera_id=getattr(record, "camera_id", None),
                    model_version=getattr(record, "model_version_id", None),
                    reasons=list(validation.reasons),
                )
                return None
            try:
                evidence = json.loads(record.release_evidence_json)
                evidence_sha256 = normalize_runtime_artifact_sha256(
                    evidence["artifacts"]["runtime_artifact_sha256"]
                )
            except (KeyError, TypeError, ValueError):
                logger.error(
                    "manager.production_runtime_digest_missing",
                    camera_id=getattr(record, "camera_id", None),
                    model_version=getattr(record, "model_version_id", None),
                )
                return None

        db = getattr(self, "_db", None)
        if db is None:
            return evidence_sha256

        operation = None
        runtime_state_json: str | None = None
        try:
            with db.get_session() as session:
                operation = (
                    session.query(ModelReleaseOperation)
                    .filter_by(
                        model_version_id=getattr(record, "model_version_id", None),
                        camera_id=getattr(record, "camera_id", None),
                        target_stage=getattr(record, "stage", None),
                        applied_revision=int(getattr(record, "release_revision", 0) or 0),
                        status=ReleaseOperationStatus.APPLIED.value,
                    )
                    .order_by(ModelReleaseOperation.id.desc())
                    .first()
                )
                if operation is not None:
                    runtime_state_json = operation.runtime_state_json
        except Exception as exc:
            logger.error(
                "manager.release_operation_lookup_failed",
                camera_id=getattr(record, "camera_id", None),
                model_version=getattr(record, "model_version_id", None),
                error=str(exc),
            )
            return None

        if operation is None:
            # Production evidence predates ModelReleaseService operations and
            # remains a complete signed authority source. Other stages have no
            # safe legacy digest and therefore fail closed.
            return evidence_sha256

        try:
            runtime_state = json.loads(runtime_state_json)
            identity = runtime_state["identity"]
            if (
                identity.get("camera_id") != getattr(record, "camera_id", None)
                or identity.get("model_version_id")
                != getattr(record, "model_version_id", None)
                or identity.get("target_stage") != getattr(record, "stage", None)
            ):
                raise ValueError("applied operation identity does not match registry row")
            operation_sha256 = normalize_runtime_artifact_sha256(
                identity.get("runtime_artifact_sha256")
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(
                "manager.release_operation_identity_invalid",
                camera_id=getattr(record, "camera_id", None),
                model_version=getattr(record, "model_version_id", None),
                operation_id=getattr(operation, "operation_id", None),
                error=str(exc),
            )
            return None

        if evidence_sha256 is not None and evidence_sha256 != operation_sha256:
            logger.error(
                "manager.release_authority_digest_conflict",
                camera_id=getattr(record, "camera_id", None),
                model_version=getattr(record, "model_version_id", None),
                evidence_sha256=evidence_sha256,
                operation_sha256=operation_sha256,
            )
            return None
        return operation_sha256

    def _resolve_model_path(self, cam_config: CameraConfig) -> Path | None:
        record = self._get_primary_model_record(cam_config.camera_id)
        if record is not None:
            resolved = self._resolve_model_record_path(
                record,
                cam_config.camera_id,
                backend=cam_config.anomaly.inference_backend,
            )
            if resolved is not None:
                return resolved
            logger.warning(
                "manager.primary_model_missing",
                camera_id=cam_config.camera_id,
                model_version=getattr(record, "model_version_id", None),
                model_path=getattr(record, "model_path", None),
            )
            return None
        if self._has_registered_model_records(cam_config.camera_id):
            return None
        if self._db is not None:
            logger.warning(
                "manager.model_not_registry_authorized",
                camera_id=cam_config.camera_id,
            )
            return None
        return DetectionPipeline._find_model(
            cam_config.camera_id,
            backend=cam_config.anomaly.inference_backend,
        )

    def _resolve_authorized_model_identity(self, camera_id: str) -> ModelIdentity | None:
        """Resolve the registry-selected model used for runner recovery."""
        cam_config = self.get_camera_config(camera_id)
        if cam_config is None:
            return None
        record = self._get_primary_model_record(camera_id)
        if record is None:
            return None
        path = self._resolve_model_record_path(
            record,
            camera_id,
            backend=cam_config.anomaly.inference_backend,
        )
        if path is None:
            return None
        from argus.preprocessing.scene_profile import runtime_artifact_sha256

        scene_status = self.get_scene_guard_status(camera_id)
        return ModelIdentity(
            version=getattr(record, "model_version_id", None),
            path=str(path),
            artifact_sha256=runtime_artifact_sha256(path),
            scene_profile_id=(
                scene_status.get("profile_id")
                if isinstance(scene_status, dict)
                else None
            ),
            release_revision=int(getattr(record, "release_revision", 0) or 0),
        )

    def _model_version_id(self, cam_config: CameraConfig, model_path: Path | None = None) -> str:
        """Derive a version tag for the camera's model."""
        record = self._get_primary_model_record(cam_config.camera_id)
        if record is not None:
            return (
                record.model_version_id
                if model_path is not None
                else f"{cam_config.anomaly.model_type}:ssim_fallback"
            )
        if self._has_registered_model_records(cam_config.camera_id):
            return f"{cam_config.anomaly.model_type}:ssim_fallback"
        resolved_model_path = model_path
        if resolved_model_path is None and self._db is None:
            resolved_model_path = DetectionPipeline._find_model(
                cam_config.camera_id,
                backend=cam_config.anomaly.inference_backend,
            )
        if resolved_model_path:
            return f"{cam_config.anomaly.model_type}:{resolved_model_path.parent.name}"
        return f"{cam_config.anomaly.model_type}:ssim_fallback"

    @staticmethod
    def _frame_identity_stream_id(
        declared_config: CameraConfig,
        runtime_config: CameraConfig,
    ) -> str | None:
        """Infer the go2rtc stream name when runtime capture is redirected."""
        if (
            declared_config.source != runtime_config.source
            or declared_config.protocol != runtime_config.protocol
        ):
            return declared_config.camera_id
        return None

    def get_runner_snapshot(self, camera_id: str) -> RunnerSnapshot | None:
        """Get point-in-time snapshot of a camera's inference runner state."""
        runner = self._runners.get(camera_id)
        if runner is None:
            return None
        return runner.get_snapshot()

    def get_model_slot_identity(self, camera_id: str) -> ModelIdentity | None:
        """Return the model identity actually applied by a live runner."""
        runner = self._runners.get(camera_id)
        if not isinstance(runner, CameraInferenceRunner):
            return None
        return runner.model_identity

    @property
    def alert_count(self) -> int:
        return self._alert_count

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and bool(self._threads or getattr(self, "_sources", {}))

    def _start_camera(
        self, cam_config: CameraConfig, *, expected_generation: int | None = None,
    ) -> bool:
        """Initialize and start a single camera pipeline in a new thread."""
        camera_id = cam_config.camera_id

        with self._lock:
            self._ensure_lifecycle_state()
            if getattr(self, "_training_maintenance_mode", False):
                self._legacy_task_errors[camera_id] = "TRAINING_MAINTENANCE_ACTIVE"
                return False
            if not self._claim_compute_ownership_locked():
                self._legacy_task_errors[camera_id] = "COMPUTE_OWNERSHIP_UNAVAILABLE"
                return False
            current_generation = self._camera_generations.get(camera_id, 0)
            if self._stop_event.is_set() or (
                expected_generation is not None and expected_generation != current_generation
            ):
                return False
            if (camera_id in self._stopping_cameras or camera_id in self._retiring_cameras
                    or camera_id in self._retiring_sources):
                logger.warning("manager.camera_cleanup_pending", camera_id=camera_id)
                return False
            if camera_id in self._threads and self._threads[camera_id].is_alive():
                logger.warning("manager.already_running", camera_id=camera_id)
                return True
            if camera_id in self._pipelines:
                logger.warning("manager.previous_runtime_needs_cleanup", camera_id=camera_id)
                return False
            if camera_id in self._starting_cameras:
                logger.warning("manager.already_starting", camera_id=camera_id)
                return self._starting_generations.get(camera_id) == current_generation
            self._starting_cameras.add(camera_id)
            generation = current_generation + 1
            self._camera_generations[camera_id] = generation
            self._starting_generations[camera_id] = generation

        pipeline = None
        source_runtime = None
        thread = None
        published = False
        try:
            self._bootstrap_task_identity(cam_config)
            runtime_cam_config = self._resolve_source_config(cam_config)
            if not self._start_source(cam_config, runtime_config=runtime_cam_config,
                                      expected_generation=generation):
                return False
            with self._lock:
                source_runtime = self._sources.get(camera_id)
                runtime_cam_config = self._source_runtime_configs.get(camera_id, runtime_cam_config)

            frame_identity_stream_id = self._frame_identity_stream_id(
                cam_config,
                runtime_cam_config,
            )

            def _alert_handler(alert: Alert):
                source_token = None
                with self._lock:
                    if (
                        not published
                        or pipeline is None
                        or self._stop_event.is_set()
                        or self._camera_generations.get(camera_id) != generation
                        or self._pipelines.get(camera_id) is not pipeline
                        or not self._legacy_task_enabled.get(camera_id, True)
                    ):
                        return
                if source_runtime is not None:
                    identity = pipeline.get_latest_frame_identity()
                    source_token = ((identity.source_session_id, identity.source_epoch)
                                    if identity is not None else None)
                    if source_token is None or source_runtime.source_token != source_token:
                        return
                # C3: Cross-camera correlation check before emitting
                if self._correlator is not None:
                    anomaly_map = pipeline.get_latest_anomaly_map()
                    location = self._anomaly_peak_location(anomaly_map)
                    max_age = (
                        self._cross_camera_config.max_age_seconds
                        if self._cross_camera_config
                        else 5.0
                    )
                    result = self._correlator.check(
                        camera_id=alert.camera_id,
                        anomaly_location=location,
                        timestamp=time.time(),
                        max_age_seconds=max_age,
                    )
                    alert.corroborated = result.corroborated
                    alert.correlation_partner = result.partner_camera
                    if not result.corroborated:
                        # F4: Force-corroboration mode — drop the alert entirely
                        # instead of downgrading, when the zone demands it.
                        if self._zone_requires_corroboration(alert.camera_id, alert.zone_id):
                            logger.info(
                                "alerts.suppressed_uncorroborated",
                                alert_id=alert.alert_id,
                                camera_id=alert.camera_id,
                                zone_id=alert.zone_id,
                                partner=result.partner_camera,
                            )
                            return

                        downgrade = (
                            self._cross_camera_config.uncorroborated_severity_downgrade
                            if self._cross_camera_config
                            else 1
                        )
                        if downgrade > 0:
                            alert.severity = self._downgrade_severity(
                                alert.severity, downgrade
                            )
                        logger.info(
                            "manager.alert_uncorroborated",
                            camera_id=alert.camera_id,
                            partner=result.partner_camera,
                            severity=alert.severity.value,
                        )

                accepted_alerts = []

                def _commit_alert():
                    with self._lock:
                        if (
                            self._stop_event.is_set()
                            or self._camera_generations.get(camera_id) != generation
                            or self._pipelines.get(camera_id) is not pipeline
                            or not self._legacy_task_enabled.get(camera_id, True)
                        ):
                            return
                        self._alert_count += 1
                        # Acceptance is the linearization point.  Delivery of
                        # this accepted result may finish after a reconnect,
                        # but slow disk/DB effects must never hold the source
                        # lock and stall every other task on the camera.
                        accepted_alerts.append(alert)

                # Correlation can take time: a reconnect during that work
                # invalidates the original frame before any external dispatch.
                if source_runtime is not None:
                    source_runtime.commit_if_current(source_token, _commit_alert)
                else:
                    _commit_alert()
                if accepted_alerts and self._on_alert:
                    self._on_alert(accepted_alerts[0])

            # B1-5: Use shared detector for dinomaly multi-class mode
            shared_detector = None
            anomaly_cfg = runtime_cam_config.anomaly
            if (
                anomaly_cfg.model_type == "dinomaly2"
                and anomaly_cfg.dinomaly_multi_class
            ):
                shared_detector = self._get_shared_detector(anomaly_cfg)

            # Shadow runner: check for shadow-stage model to evaluate
            shadow_runner = self._create_shadow_runner(runtime_cam_config)
            if shared_detector is not None:
                shared_model_path = getattr(shared_detector, "_model_path", None)
                model_path = (
                    Path(shared_model_path)
                    if isinstance(shared_model_path, str | Path)
                    else None
                )
            else:
                model_path = self._resolve_model_path(runtime_cam_config)
            model_version_id = self._model_version_id(runtime_cam_config, model_path)

            pipeline = DetectionPipeline(
                camera_config=runtime_cam_config,
                alert_config=self._alert_config,
                on_alert=_alert_handler,
                shared_yolo_model=self._get_legacy_shared_yolo(runtime_cam_config),
                on_drift=self._on_status_change,
                segmenter_config=self._segmenter_config,
                classifier_config=self._classifier_config,
                physics_config=self._physics_config,
                imaging_config=self._imaging_config,
                model_version_id=model_version_id,
                model_path=model_path,
                # CameraManager already resolved the registry/shared-model
                # decision.  A second filesystem discovery inside the pipeline
                # could silently load an unrelated stale artifact.
                auto_discover_model=False,
                shared_anomaly_detector=shared_detector,
                shadow_runner=shadow_runner,
                recording_store=self._alert_recording_store,
                database=self._db,
                event_bus=self._event_bus,
                inference_executor=self._inference_executor,
                encode_executor=self._encode_executor,
                sensor_fusion=self._sensor_fusion,
                frame_identity_declared_source=cam_config.source,
                frame_identity_runtime_source=runtime_cam_config.source,
                frame_identity_stream_id=frame_identity_stream_id,
                baselines_dir=self._baselines_dir,
            )

            def _share_capture(capture):
                # The constructor's unopened adapter never becomes a reader.
                # Its task view owns only a subscription to the existing source.
                if source_runtime is None or self.get_camera_source(camera_id) is not source_runtime:
                    raise RuntimeError("camera source changed during legacy initialization")
                capture.stop()
                return SourceCaptureView(source_runtime, legacy_task_id(camera_id))

            # A public ownership handoff keeps standalone pipelines compatible
            # while the managed path has exactly one capture reader.
            bind_capture = getattr(pipeline, "bind_capture_runtime", None)
            if callable(bind_capture):
                bind_capture(_share_capture)

            # 5.1: Wrap pipeline in CameraInferenceRunner
            deg_config = runtime_cam_config.degradation
            runner = CameraInferenceRunner(
                pipeline=pipeline,
                health_monitor=self._health_monitor,
                version_tag=model_version_id,
                audit_logger=self._audit_logger,
                max_consecutive_failures=deg_config.max_consecutive_failures,
                refuse_start_on_backbone_failure=deg_config.refuse_start_on_backbone_failure,
                inference_buffer=self._inference_buffer,
                authorized_model_resolver=(
                    self._resolve_authorized_model_identity
                    if self._db is not None
                    else None
                ),
                initial_model_path=model_path,
            )
            if self._record_store is not None:
                runner.set_record_store(self._record_store)

            if not runner.initialize():
                with self._lock:
                    self._legacy_task_errors[camera_id] = "LEGACY_INITIALIZATION_FAILED"
                logger.error("manager.init_failed", camera_id=camera_id)
                get_error_channel().emit(
                    severity=SEVERITY_WARNING,
                    source="capture_manager",
                    code="init_failed",
                    message=f"摄像头 {camera_id} 初始化失败",
                    context={"camera_id": camera_id},
                )
                return False

            thread = threading.Thread(
                target=self._camera_loop,
                args=(camera_id, runner),
                name=f"argus-{camera_id}",
                daemon=True,
            )

            # SDK setup is outside the manager lock. Reconcile all changes
            # that happened during it, then publish with a revision CAS.
            for _ in range(32):
                with self._lock:
                    if (
                        self._stop_event.is_set()
                        or self._camera_generations.get(camera_id) != generation
                        or camera_id in self._stopping_cameras
                    ):
                        return False
                    module_revision = self._module_revision
                    desired = self._desired_auxiliary_states()
                reconcile = getattr(pipeline, "reconcile_auxiliary_desired", None)
                if not callable(reconcile) or reconcile(desired) is not True:
                    logger.warning("manager.auxiliary_reconcile_failed", camera_id=camera_id)
                    return False
                with self._lock:
                    if (
                        self._stop_event.is_set()
                        or self._camera_generations.get(camera_id) != generation
                        or camera_id in self._stopping_cameras
                    ):
                        return False
                    if module_revision != self._module_revision:
                        continue
                    self._runners[camera_id] = runner
                    self._pipelines[camera_id] = pipeline
                    if self._sources.get(camera_id) is not source_runtime:
                        return False
                    self._legacy_task_locks.setdefault(camera_id, threading.RLock())
                    self._threads[camera_id] = thread
                    # Starting under the publication lock prevents stop from
                    # attempting to join a not-yet-started published thread.
                    thread.start()
                    published = True
                    self._camera_lifecycle_errors.pop(camera_id, None)
                    self._legacy_task_errors.pop(camera_id, None)
                    break
            if not published:
                logger.warning("manager.module_revision_unstable", camera_id=camera_id)
                return False
            logger.info(
                "manager.camera_started",
                camera_id=camera_id,
                source=cam_config.source,
                protocol=cam_config.protocol,
                runtime_source=runtime_cam_config.source,
                runtime_protocol=runtime_cam_config.protocol,
            )
            return True
        finally:
            with self._lock:
                if (not published and self._sources.get(camera_id) is source_runtime
                        and source_runtime is not None
                        and self._camera_generations.get(camera_id) == generation):
                    self._legacy_task_errors.setdefault(camera_id, "LEGACY_INITIALIZATION_FAILED")
            if not published and pipeline is not None:
                failures = []
                with self._lock:
                    cleanup_lock = self._camera_cleanup_locks.setdefault(camera_id, threading.RLock())
                with cleanup_lock:
                    with self._lock:
                        if self._pipelines.get(camera_id) is pipeline:
                            self._pipelines.pop(camera_id, None)
                            self._runners.pop(camera_id, None)
                            self._threads.pop(camera_id, None)
                        self._retiring_cameras[camera_id] = (pipeline, thread)
                        if (source_runtime is not None and self._sources.get(camera_id) is source_runtime
                                and camera_id not in self._stopping_cameras):
                            self._legacy_candidate_sources[camera_id] = source_runtime
                    try:
                        pipeline.shutdown()
                    except Exception as exc:
                        failures.append(f"candidate shutdown: {exc}")
                    if thread is not None:
                        try:
                            if thread is not threading.current_thread() and thread.is_alive():
                                thread.join(timeout=30.0)
                            if thread.is_alive():
                                failures.append("candidate camera thread is still running")
                        except Exception as exc:
                            failures.append(f"candidate thread cleanup: {exc}")
                    with self._lock:
                        if failures:
                            self._camera_lifecycle_errors[camera_id] = "; ".join(failures)
                            logger.error("manager.candidate_cleanup_failed", camera_id=camera_id, error="; ".join(failures))
                        elif self._retiring_cameras.get(camera_id, (None,))[0] is pipeline:
                            self._retiring_cameras.pop(camera_id, None)
                            self._legacy_candidate_sources.pop(camera_id, None)
                            self._camera_lifecycle_errors.pop(camera_id, None)
            with self._lock:
                if self._starting_generations.get(camera_id) == generation:
                    self._starting_cameras.discard(camera_id)
                    self._starting_generations.pop(camera_id, None)

    def _camera_loop(self, camera_id: str, runner: CameraInferenceRunner) -> None:
        """Main loop for a single camera thread with frame-rate watchdog."""
        with self._lock:
            self._ensure_lifecycle_state()
        pipeline = runner.pipeline
        logger.info("camera_loop.started", camera_id=camera_id)
        last_frame_time = time.monotonic()
        # MED-03: Configurable watchdog timeout from camera config
        cam_config = next((c for c in self._cameras if c.camera_id == camera_id), None)
        watchdog_timeout = cam_config.watchdog_timeout if cam_config else 30.0
        last_connected = bool(
            getattr(getattr(pipeline._camera, "state", None), "connected", False)
        )

        # Initialize backpressure counters
        with self._bp_lock:
            self._pending_frames[camera_id] = 0
            self._frames_dropped[camera_id] = 0

        try:
            # A manager-wide stop event is not sufficient for the REST
            # stop/restart path.  The runner identity is the per-camera
            # generation token: removing or replacing it stops this loop and
            # prevents an old generation from racing a restarted pipeline.
            while (
                not self._stop_event.is_set()
                and self._runners.get(camera_id) is runner
            ):
                try:
                    if not self._legacy_task_enabled.get(camera_id, True):
                        # Preview, recording and other task subscriptions keep
                        # consuming the independently owned source.
                        last_frame_time = time.monotonic()
                        self._stop_event.wait(0.05)
                        continue
                    # Backpressure check: skip non-heartbeat frames when overloaded
                    with self._bp_lock:
                        pending = self._pending_frames.get(camera_id, 0)
                    is_heartbeat = (
                        pipeline.stats.frames_captured > 0
                        and cam_config is not None
                        and getattr(cam_config.mog2, "heartbeat_frames", 0) > 0
                        and pipeline.stats.frames_captured % cam_config.mog2.heartbeat_frames == 0
                    )

                    if pending >= self._max_pending and not is_heartbeat:
                        with self._bp_lock:
                            self._frames_dropped[camera_id] = self._frames_dropped.get(camera_id, 0) + 1
                        pipeline.stats.frames_dropped_backpressure += 1
                        now = time.monotonic()
                        last_warn = self._last_bp_warn.get(camera_id, 0)
                        if now - last_warn > 60.0:
                            logger.warning(
                                "backpressure.dropping_frame",
                                camera_id=camera_id,
                                pending=pending,
                                total_dropped=self._frames_dropped.get(camera_id, 0),
                            )
                            self._last_bp_warn[camera_id] = now
                        self._stop_event.wait(0.01)  # Brief yield
                        continue

                    with self._bp_lock:
                        self._pending_frames[camera_id] = self._pending_frames.get(camera_id, 0) + 1

                    try:
                        task_lock = self._legacy_task_locks.setdefault(camera_id, threading.RLock())
                        with task_lock:
                            if not self._legacy_task_enabled.get(camera_id, True):
                                continue
                            run_once_outcome = getattr(runner, "run_once_outcome", None)
                            if callable(run_once_outcome):
                                outcome = run_once_outcome()
                                alert = outcome.alert
                            else:
                                outcome = None
                                alert = runner.run_once()
                    finally:
                        with self._bp_lock:
                            self._pending_frames[camera_id] = max(0, self._pending_frames.get(camera_id, 0) - 1)

                    if self._runners.get(camera_id) is not runner:
                        break

                    connected = bool(
                        getattr(
                            getattr(pipeline._camera, "state", None),
                            "connected",
                            False,
                        )
                    )
                    if connected != last_connected:
                        self._broadcast_camera_and_wall(camera_id)
                        last_connected = connected

                    # Yield CPU when no frame was produced (camera disconnected,
                    # reconnecting, or stream exhausted) to prevent busy-spin.
                    idle_or_failed = (
                        outcome is not None
                        and outcome.kind in {
                            FrameOutcomeKind.NO_FRAME,
                            FrameOutcomeKind.FAILED,
                        }
                    )
                    if idle_or_failed or (
                        outcome is None
                        and alert is None
                        and pipeline.stats.frames_captured
                        == self._last_frame_counts.get(camera_id, 0)
                    ):
                        self._stop_event.wait(0.05)

                    # M6: Feed frame to continuous recorder (if active)
                    if (self._continuous_recording_mgr is not None
                            and not getattr(pipeline, "_capture_runtime_bound", False)):
                        raw = pipeline.get_raw_frame()
                        if raw is not None:
                            self._continuous_recording_mgr.push_frame(
                                camera_id, raw, time.time(),
                            )

                    # C3: Feed anomaly map to cross-camera correlator
                    if self._correlator is not None:
                        anomaly_map = pipeline.get_latest_anomaly_map()
                        if anomaly_map is not None:
                            self._correlator.update(
                                camera_id, anomaly_map, time.time()
                            )

                    # Update watchdog timer on successful frame processing
                    if pipeline.stats.frames_captured > 0:
                        current = pipeline.stats.frames_captured
                        prev = self._last_frame_counts.get(camera_id, 0)
                        if current > prev:
                            last_frame_time = time.monotonic()
                            self._last_frame_counts[camera_id] = current
                            # Notify every 25 frames (~5s at 5fps) to avoid flooding
                            if current % 25 == 0:
                                self._notify_camera_status(camera_id)

                    # Watchdog: force reconnect on stale stream
                    if time.monotonic() - last_frame_time > watchdog_timeout:
                        logger.warning(
                            "watchdog.stale_stream",
                            camera_id=camera_id,
                            stale_seconds=watchdog_timeout,
                        )
                        pipeline._camera.reconnect()
                        last_frame_time = time.monotonic()
                        connected = bool(
                            getattr(
                                getattr(pipeline._camera, "state", None),
                                "connected",
                                False,
                            )
                        )
                        if connected != last_connected:
                            self._broadcast_camera_and_wall(camera_id)
                            last_connected = connected

                except Exception as e:
                    logger.error(
                        "camera_loop.error",
                        camera_id=camera_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    connected = bool(
                        getattr(
                            getattr(pipeline._camera, "state", None),
                            "connected",
                            False,
                        )
                    )
                    if (
                        self._runners.get(camera_id) is runner
                        and connected != last_connected
                    ):
                        self._broadcast_camera_and_wall(camera_id)
                        last_connected = connected
                    self._stop_event.wait(1.0)
        except Exception as e:
            logger.error("camera_loop.fatal", camera_id=camera_id, error=str(e))
            get_error_channel().emit(
                severity=SEVERITY_ERROR,
                source="capture_manager",
                code="camera_loop_fatal",
                message=f"摄像头 {camera_id} 处理循环致命错误: {e}",
                context={
                    "camera_id": camera_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
        finally:
            logger.info("camera_loop.stopped", camera_id=camera_id)
            # Keep the completed thread handle until stop/watchdog confirms
            # pipeline cleanup; dropping it here hides a dead live runtime.

    @staticmethod
    def _anomaly_peak_location(anomaly_map: np.ndarray | None) -> tuple[int, int]:
        """Find the peak anomaly location in a heatmap. Returns (x, y)."""
        if anomaly_map is None or anomaly_map.size == 0:
            return (0, 0)
        idx = np.argmax(anomaly_map)
        h, w = anomaly_map.shape[:2]
        y, x = divmod(int(idx), w)
        return (x, y)

    _SEVERITY_ORDER = [
        AlertSeverity.INFO,
        AlertSeverity.LOW,
        AlertSeverity.MEDIUM,
        AlertSeverity.HIGH,
    ]

    @classmethod
    def _downgrade_severity(cls, severity: AlertSeverity, levels: int) -> AlertSeverity:
        """Downgrade alert severity by N levels (minimum INFO)."""
        try:
            idx = cls._SEVERITY_ORDER.index(severity)
        except ValueError:
            return severity
        new_idx = max(0, idx - levels)
        return cls._SEVERITY_ORDER[new_idx]

    def _zone_requires_corroboration(self, camera_id: str, zone_id: str) -> bool:
        """Return True when the zone's F4 ``require_corroboration`` flag is set.

        Missing camera or zone (e.g. synthetic "default" zone_id) safely
        returns False so we fall back to the existing downgrade behaviour.
        """
        cam_config = next(
            (c for c in self._cameras if c.camera_id == camera_id), None,
        )
        if cam_config is None:
            return False
        for zone in cam_config.zones:
            if zone.zone_id == zone_id:
                return bool(getattr(zone, "require_corroboration", False))
        return False

    # --- 5.3: Process-level watchdog ---

    def _start_watchdog(self) -> None:
        """Start the process-level watchdog thread."""
        if self._watchdog_thread is not None:
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="argus-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        """Stop the process-level watchdog thread."""
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5.0)
            self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        """Periodically check thread liveness and restart dead camera threads."""
        # Get check interval from first camera config or use default
        interval = 15.0
        if self._cameras:
            interval = self._cameras[0].degradation.watchdog_check_interval_seconds

        while not self._watchdog_stop.wait(interval):
            if self._stop_event.is_set():
                break

            with self._lock:
                dead_cameras = [
                    cam_id for cam_id, thread in self._threads.items()
                    if not thread.is_alive()
                ]

            for camera_id in dead_cameras:
                logger.warning(
                    "watchdog.thread_dead",
                    camera_id=camera_id,
                    msg="Camera thread died, attempting restart",
                )
                # Dead threads still own capture/model cleanup. Retire them
                # through the same retryable path used by explicit stops.
                try:
                    with self._lock:
                        self._ensure_lifecycle_state()
                        generation = self._camera_generations.get(camera_id, 0)
                    self.stop_camera(camera_id)
                except Exception as exc:
                    logger.error("watchdog.cleanup_failed", camera_id=camera_id, error=str(exc))
                    continue

                # Attempt restart
                cam_config = next(
                    (c for c in self._cameras if c.camera_id == camera_id), None
                )
                if cam_config:
                    try:
                        if self._start_camera(cam_config, expected_generation=generation + 1):
                            logger.info("watchdog.restarted", camera_id=camera_id)
                    except Exception as e:
                        logger.error(
                            "watchdog.restart_failed",
                            camera_id=camera_id,
                            error=str(e),
                        )

            # Prune stale alert grader trackers (hourly equivalent)
            # Watchdog runs every ~15s, prune every ~240 iterations ≈ hourly
            if not hasattr(self, "_watchdog_ticks"):
                self._watchdog_ticks = 0
            self._watchdog_ticks += 1
            if self._watchdog_ticks % max(1, int(3600 / interval)) == 0:
                for pipeline in list(self._pipelines.values()):
                    grader = getattr(pipeline, "_alert_grader", None)
                    if grader and hasattr(grader, "prune_stale_trackers"):
                        grader.prune_stale_trackers()

            # Write heartbeat file for external process monitors
            try:
                from pathlib import Path
                heartbeat_path = Path("data/.watchdog")
                heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = heartbeat_path.with_suffix(".tmp")
                tmp.write_text(str(time.time()))
                tmp.replace(heartbeat_path)
            except Exception:
                logger.debug("camera.watchdog_heartbeat_failed", exc_info=True)
