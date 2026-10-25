"""Training job executor — orchestrates the full job lifecycle.

Processes confirmed (queued) training jobs:
1. Resolves active backbone checkpoint (for anomaly_head jobs)
2. Runs training via ModelTrainer or BackboneTrainer
3. Runs three-step validation via TrainingValidator
4. Assembles model package via ModelPackager
5. Registers model version in ModelRegistry
6. Updates TrainingJob status in database
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from argus.anomaly.artifacts import artifact_metadata, exported_artifact, training_directory
from argus.anomaly.scene_training import (
    revalidate_scene_training_input,
    validate_scene_training_calibration,
    validate_scene_training_descriptor,
    validate_scene_training_input,
)
from argus.anomaly.trainer import ModelTrainer, TrainingStatus
from argus.storage.models import (
    BackboneRecord,
    ModelStage,
    ReleaseOperationStatus,
    TrainingJobStatus,
    TrainingJobType,
)

if TYPE_CHECKING:
    from argus.anomaly.backbone_trainer import BackboneTrainer
    from argus.anomaly.model_package import ModelPackager
    from argus.anomaly.training_validator import TrainingValidator
    from argus.capture.manager import CameraManager
    from argus.storage.database import Database
    from argus.storage.model_release_service import ModelReleaseService
    from argus.storage.model_registry import ModelRegistry

from argus.core.error_channel import (
    SEVERITY_ERROR,
    get_error_channel,
)
from argus.core.pipeline import PipelineMode
from argus.core.pipeline_mode_guard import GlobalPipelineModeGuard

logger = structlog.get_logger()


def _baseline_quality_evidence(value: object) -> dict:
    """Snapshot training-input diagnostics without inventing a passing result."""
    unavailable = {
        "report_status": "not_evaluated", "passed": None, "integrity_passed": None,
        "warnings": [], "errors": [], "production_readiness": "not_evaluated",
    }
    if value is None:
        return unavailable
    try:
        if not isinstance(value, dict):
            raise ValueError("baseline quality report must be an object")
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > 512 * 1024:
            raise ValueError("baseline quality report exceeds its evidence budget")
        snapshot = json.loads(encoded)
        if type(snapshot.get("passed")) is not bool:
            raise ValueError("baseline quality passed must be a literal boolean")
        for field in ("warnings", "errors"):
            values = snapshot.setdefault(field, [])
            if not isinstance(values, list) or any(type(item) is not str for item in values):
                raise ValueError(f"baseline quality {field} must contain strings")
        if snapshot["passed"] and (snapshot["errors"] or snapshot.get("integrity_passed") is False):
            raise ValueError("a passing quality report cannot contain failed integrity or errors")
        if (
            snapshot.get("scope") == "observed_fixed_scene_only" and snapshot["passed"]
            and snapshot.get("integrity_passed") is not True
        ):
            raise ValueError("fixed-scene suitability requires explicit integrity evidence")
    except (TypeError, ValueError, OverflowError):
        return {**unavailable, "report_status": "invalid", "errors": ["训练输入质量报告缺失必要字段或格式无效"]}
    snapshot["report_status"] = "available"
    # Input suitability is not production readiness or approval.
    snapshot["production_readiness"] = "not_evaluated"
    return snapshot


# ── Hyperparameter validation ──

_HYPERPARAMETER_LIMITS: dict[str, dict] = {
    "epochs": {"type": int, "min": 1, "max": 200},
    "lr": {"type": float, "min": 1e-8, "max": 1.0},
    "batch_size": {"type": int, "min": 1, "max": 256},
    "image_size": {"type": int, "min": 64, "max": 1024},
    "backbone_type": {"type": str, "allowed": {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"}},
    "export_format": {"type": str, "allowed": {"openvino", "onnx", "torch"}},
    "quantization": {"type": str, "allowed": {"fp16", "fp32", "int8"}},
    "model_type": {"type": str, "allowed": {"patchcore", "efficient_ad", "fastflow", "padim", "dinomaly2"}},
    "calibration_images": {"type": int, "min": 1, "max": 500},
}


def validate_hyperparameters(params: dict) -> list[str]:
    """Validate hyperparameters against known limits. Returns list of errors."""
    errors: list[str] = []
    if not isinstance(params, dict):
        return ["hyperparameters must be an object"]
    if "scene_workflow" in params:
        try:
            validate_scene_training_descriptor(params["scene_workflow"])
        except ValueError as exc:
            errors.append(str(exc))
        if params.get("skip_baseline_validation", False) is not False:
            errors.append("scene workflow training cannot skip baseline validation")
        if params.get("labeling_entry_ids"):
            errors.append("scene workflow training cannot consume unrelated labeling entries")
    for key, value in params.items():
        spec = _HYPERPARAMETER_LIMITS.get(key)
        if spec is None:
            continue  # unknown params are passed through
        expected_type = spec["type"]
        if not isinstance(value, expected_type):
            # Allow int where float is expected
            if expected_type is float and isinstance(value, int):
                pass
            else:
                errors.append(f"{key}: expected {expected_type.__name__}, got {type(value).__name__}")
                continue
        if "min" in spec and value < spec["min"]:
            errors.append(f"{key}={value} below minimum {spec['min']}")
        if "max" in spec and value > spec["max"]:
            errors.append(f"{key}={value} above maximum {spec['max']}")
        if "allowed" in spec and value not in spec["allowed"]:
            errors.append(f"{key}={value!r} not in {spec['allowed']}")

    # Cross-field validation
    quantization = params.get("quantization")
    export_format = params.get("export_format")
    if quantization and quantization != "fp32" and export_format and export_format != "openvino":
        errors.append(f"量化 ({quantization}) 仅支持 openvino 导出格式，当前为 {export_format}")

    return errors


class TrainingJobExecutor:
    """Executes queued training jobs end-to-end."""

    def __init__(
        self,
        database: Database,
        trainer: ModelTrainer,
        backbone_trainer: BackboneTrainer | None = None,
        training_validator: TrainingValidator | None = None,
        model_packager: ModelPackager | None = None,
        model_registry: ModelRegistry | None = None,
        baselines_dir: Path = Path("data/baselines"),
        model_packages_dir: Path = Path("data/model_packages"),
        camera_manager: "CameraManager | None" = None,
        model_release_service: "ModelReleaseService | None" = None,
    ):
        if model_packager is not None and model_release_service is None:
            raise ValueError(
                "model_release_service is required when packaging models"
            )
        self._db = database
        self._trainer = trainer
        self._backbone_trainer = backbone_trainer
        self._validator = training_validator
        self._packager = model_packager
        self._registry = model_registry
        if issubclass(type(trainer), ModelTrainer) and model_registry is not None:
            # Register while a temporary merged dataset still exists inside
            # trainer.train(), so its data hash describes what was consumed.
            trainer.bind_model_registry(model_registry)
        self._baselines_dir = baselines_dir
        self._model_packages_dir = model_packages_dir
        self._model_packages_dir.mkdir(parents=True, exist_ok=True)
        self._camera_manager = camera_manager
        self._release_service = model_release_service

    def _training_mode_guard(self, reason: str):
        """Return a GlobalPipelineModeGuard or a no-op when no manager is wired.

        Switching every camera to TRAINING during a job (痛点 3) frees the
        shared GPU; the guard restores each camera's previous mode on exit
        even if training raises.
        """
        if self._camera_manager is None:
            from contextlib import nullcontext
            return nullcontext()
        return GlobalPipelineModeGuard(
            self._camera_manager,
            PipelineMode.TRAINING,
            reason=reason,
        )

    def recover_stale_jobs(self, max_running_hours: float = 6.0) -> int:
        """Flag old RUNNING jobs without assuming that their worker stopped.

        A wall-clock timeout cannot distinguish a crashed worker from a slow
        live worker in another process. Keep the reservation until an operator
        verifies termination; otherwise retries can race a late model commit.
        """
        running_jobs = self._db.list_training_jobs(status=TrainingJobStatus.RUNNING.value)
        recovered = 0
        now = datetime.now(timezone.utc)
        for job in running_jobs:
            if job.started_at is None:
                continue
            started_at = job.started_at
            if started_at.tzinfo is None:
                # SQLite DateTime preserves UTC values without tzinfo.
                started_at = started_at.replace(tzinfo=timezone.utc)
            elapsed_hours = (now - started_at).total_seconds() / 3600
            if elapsed_hours > max_running_hours:
                error = "Running job exceeded its time limit; verify worker termination before administrator reconciliation"
                if job.error == error:
                    continue
                updated = self._db.update_training_job(
                    job.job_id,
                    expected_status=TrainingJobStatus.RUNNING.value,
                    error=error,
                )
                if not updated:
                    continue
                logger.warning(
                    "job_executor.stale_job_requires_reconciliation",
                    job_id=job.job_id,
                    elapsed_hours=round(elapsed_hours, 1),
                )
                recovered += 1
        return recovered

    def process_queued_jobs(self) -> int:
        """Process all queued training jobs. Returns number processed."""
        # Flag potentially stale workers, but do not free their reservation.
        self.recover_stale_jobs()

        jobs = self._db.list_training_jobs(status=TrainingJobStatus.QUEUED.value)
        processed = 0
        for job in jobs:
            try:
                if self._execute_job(job):
                    processed += 1
            except Exception as e:
                logger.error(
                    "job_executor.failed",
                    job_id=job.job_id,
                    error=str(e),
                )
        return processed

    def execute(self, job_id: str) -> None:
        """Execute a single training job by job_id."""
        job = self._db.get_training_job(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        if job.status != TrainingJobStatus.QUEUED.value:
            raise ValueError(f"Job {job_id} is not queued (status={job.status})")
        self._execute_job(job)

    def _execute_job(self, job) -> bool:
        """Atomically claim a queued job; stale readers must not train it."""
        job_id = job.job_id
        now = datetime.now(timezone.utc)
        claimed = self._db.update_training_job(
            job_id, expected_status=TrainingJobStatus.QUEUED.value,
            status=TrainingJobStatus.RUNNING.value, started_at=now, error=None,
        )
        if not claimed:
            logger.info("job_executor.claim_lost", job_id=job_id)
            return False

        logger.info(
            "job_executor.started",
            job_id=job_id,
            job_type=job.job_type,
            camera_id=job.camera_id,
        )

        start = time.monotonic()

        try:
            if job.job_type == TrainingJobType.SSL_BACKBONE.value:
                self._execute_backbone(job)
            elif job.job_type == TrainingJobType.ANOMALY_HEAD.value:
                self._execute_head(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
        except Exception as e:
            duration = time.monotonic() - start
            # Per-invocation evidence prevents a previous/concurrent job's
            # passing report from being reused for this failure.
            carried_metrics = getattr(e, "_argus_training_metrics", None)
            failure_metrics = dict(carried_metrics) if isinstance(carried_metrics, dict) else {}
            if job.job_type == TrainingJobType.ANOMALY_HEAD.value:
                quality = getattr(e, "_argus_baseline_quality", None)
                failure_metrics["baseline_quality"] = (
                    quality if isinstance(quality, dict) else _baseline_quality_evidence(None)
                )
            failed = self._db.update_training_job(
                job_id,
                expected_status=TrainingJobStatus.RUNNING.value,
                status=TrainingJobStatus.FAILED.value,
                error=str(e),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=duration,
                **({"metrics": json.dumps(failure_metrics, ensure_ascii=False, allow_nan=False)} if failure_metrics else {}),
            )
            if not failed:
                logger.warning("job_executor.late_failure_ignored", job_id=job_id, error=str(e))
                return True
            logger.error("job_executor.failed", job_id=job_id, error=str(e))
            get_error_channel().emit(
                severity=SEVERITY_ERROR,
                source="job_executor",
                code="job_failed",
                message=f"训练任务失败 (job_id={job_id}): {e}",
                context={
                    "job_id": job_id,
                    "job_type": getattr(job, "job_type", None),
                    "camera_id": getattr(job, "camera_id", None),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "duration_seconds": duration,
                },
            )
        return True

    def _execute_backbone(self, job) -> None:
        """Execute SSL backbone fine-tuning job."""
        if self._backbone_trainer is None:
            raise RuntimeError("BackboneTrainer not configured")

        # Parse and validate hyperparameters
        params = json.loads(job.hyperparameters) if job.hyperparameters else {}
        param_errors = validate_hyperparameters(params)
        if param_errors:
            raise ValueError(f"Invalid hyperparameters: {'; '.join(param_errors)}")
        if "scene_workflow" in params or (
            isinstance(getattr(job, "dataset_version", None), str)
            and job.dataset_version.startswith("scene-")
        ):
            raise ValueError("scene workflow input is only supported for anomaly_head training")

        # Get all camera IDs from config (or use all available)
        camera_ids = []
        for cam_dir in self._baselines_dir.iterdir():
            if cam_dir.is_dir() and not cam_dir.name.startswith("."):
                camera_ids.append(cam_dir.name)

        with self._training_mode_guard(reason=f"backbone_train:{job.job_id}"):
            result = self._backbone_trainer.train(
                camera_ids=camera_ids,
                baselines_dir=self._baselines_dir,
                backbone_type=params.get("backbone_type", "dinov2_vitb14"),
                epochs=params.get("epochs", 5),
                lr=params.get("lr", 1e-5),
            )

        if not result.success:
            raise RuntimeError(result.error or "Backbone training failed")

        backbone_version_id = f"ssl_{Path(result.checkpoint_path).stem}"
        backbone_record = BackboneRecord(
            backbone_version_id=backbone_version_id,
            backbone_type=params.get("backbone_type", "dinov2_vitb14"),
            checkpoint_path=result.checkpoint_path,
            checkpoint_hash=result.checkpoint_hash,
            dataset_hash=result.dataset_hash,
            camera_ids_used=json.dumps(camera_ids),
            training_job_id=job.job_id,
            is_active=False,
        )
        # Registration, active-backbone selection and completion must share
        # the job's conditional transaction. A late worker must not change the
        # backbone and only afterwards discover that it lost job ownership.
        completed = self._db.complete_backbone_training_job(
            backbone_record, job_id=job.job_id,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=result.duration_seconds,
            metrics=json.dumps({
                "final_loss": result.final_loss,
                "total_images": result.total_images,
                "epochs": result.epochs_completed,
            }),
            artifacts_path=result.checkpoint_path,
        )
        if not completed:
            raise RuntimeError("Training job state changed before backbone completion; reconciliation required")

        logger.info(
            "job_executor.backbone_complete",
            job_id=job.job_id,
            backbone=backbone_version_id,
        )

    def _execute_head(self, job) -> None:
        """Execute anomaly detection head training job."""
        evidence: dict = {}
        try:
            self._execute_head_with_evidence(job, evidence)
        except Exception as exc:
            # Keep the original exception class/message, while carrying the
            # immutable report to the outer conditional failure write.
            exc._argus_baseline_quality = evidence.get("baseline_quality")
            exc._argus_training_metrics = evidence.get("failure_metrics")
            raise

    def _execute_head_with_evidence(self, job, evidence: dict) -> None:
        camera_id = job.camera_id
        if not camera_id:
            raise ValueError("camera_id is required for anomaly_head jobs")
        zone_id = job.zone_id or "default"

        # Resolve backbone checkpoint
        # Parse and validate hyperparameters
        params = json.loads(job.hyperparameters) if job.hyperparameters else {}
        param_errors = validate_hyperparameters(params)
        if param_errors:
            raise ValueError(f"Invalid hyperparameters: {'; '.join(param_errors)}")

        scene_input = None
        dataset_version = getattr(job, "dataset_version", None)
        if (
            isinstance(dataset_version, str) and dataset_version.startswith("scene-")
            and "scene_workflow" not in params
        ):
            raise ValueError("scene baseline training requires its scene_workflow descriptor; legacy fallback is forbidden")
        if "scene_workflow" in params:
            if (
                job.job_type != TrainingJobType.ANOMALY_HEAD.value
                or job.zone_id != "default"
                or dataset_version != f"scene-{params['scene_workflow']['workflow_id']}"
            ):
                raise ValueError("scene workflow job type, default zone and dataset_version must match its descriptor")
            if job.dataset_selection:
                raise ValueError("scene workflow training cannot merge dataset_selection")
            scene_input = validate_scene_training_input(
                baselines_dir=self._baselines_dir,
                camera_id=camera_id,
                zone_id=zone_id,
                scene_workflow=params["scene_workflow"],
            )

        model_type = job.model_type or params.get("model_type", "patchcore")
        # Only Dinomaly currently consumes the custom DINO encoder. Other
        # algorithms must not claim the active SSL backbone in their package.
        backbone = self._db.get_active_backbone() if model_type == "dinomaly2" else None
        backbone_checkpoint = backbone.checkpoint_path if backbone else None

        camera_config = (
            self._camera_manager.get_camera_config(camera_id)
            if self._camera_manager is not None else None
        )
        anomaly_config = getattr(camera_config, "anomaly", None)
        backend = getattr(anomaly_config, "inference_backend", "auto")
        torch_backend = backend in {"torch-cuda", "torch-cpu"}
        export_format = params.get("export_format", "torch" if torch_backend else "openvino")
        quantization = params.get("quantization", "fp16" if export_format == "openvino" else "fp32")
        if torch_backend and export_format != "torch":
            raise ValueError(f"{backend} requires torch export; {export_format} is not deployable")
        if torch_backend and quantization != "fp32":
            raise ValueError(f"{backend} training currently requires fp32 Torch artifacts")

        # 痛点 2: optional multi-version dataset selection. When provided,
        # we merge several baseline versions into a tmp dir before training.
        selection = None
        if job.dataset_selection:
            from argus.anomaly.dataset_selection import DatasetSelection

            try:
                selection = DatasetSelection.from_json(job.dataset_selection)
            except ValueError as e:
                raise ValueError(f"dataset_selection 无效: {e}")
            if selection.camera_id != camera_id:
                raise ValueError(
                    f"dataset_selection 摄像头 ({selection.camera_id}) "
                    f"与 job.camera_id ({camera_id}) 不一致"
                )

        train_kwargs = dict(
            camera_id=camera_id,
            zone_id=zone_id,
            model_type=model_type,
            image_size=params.get("image_size", 256),
            export_format=export_format,
            quantization=quantization,
            anomaly_config=anomaly_config,
            backbone_checkpoint=backbone_checkpoint,
            skip_baseline_validation=params.get("skip_baseline_validation", False),
        )
        if scene_input is not None:
            train_kwargs.update(
                baseline_dir_override=scene_input.images_dir,
                image_count_override=scene_input.image_count,
                baseline_versions_label=f"scene:{scene_input.lineage['workflow_id']}",
                require_calibration=True,
                scene_training_input=scene_input,
            )

        def train_with_evidence(**kwargs):
            result = self._trainer.train(**kwargs)
            # Capture before context-manager cleanup: a restore/temporary-data
            # cleanup failure must not erase an already produced report.
            evidence["baseline_quality"] = _baseline_quality_evidence(getattr(result, "pre_validation", None))
            if scene_input is not None:
                evidence["failure_metrics"] = {
                    "scene_workflow": dict(scene_input.lineage),
                    "calibration_status": result.calibration_status,
                    "train_count": scene_input.training_count,
                    "calibration_count": scene_input.calibration_count,
                }
            return result

        with self._training_mode_guard(reason=f"head_train:{job.job_id}"):
            if selection is not None:
                from argus.anomaly.trainer import _DatasetMerger, _build_merger_items

                bm = self._trainer._baseline_manager
                resolved_dirs = bm.resolve_dataset_dirs(selection)
                image_count = bm.count_images_multi(resolved_dirs)
                merger_items = _build_merger_items(selection, resolved_dirs)
                with _DatasetMerger(merger_items) as merged_root:
                    result = train_with_evidence(
                        **train_kwargs,
                        baseline_dir_override=merged_root,
                        image_count_override=image_count,
                        baseline_versions_label=selection.version_summary(),
                    )
            else:
                result = train_with_evidence(**train_kwargs)

        baseline_quality = evidence["baseline_quality"]
        if result.status.value != TrainingStatus.COMPLETE.value:
            raise RuntimeError(result.error or f"Training failed with status {result.status}")

        if not result.model_path:
            raise RuntimeError("Training produced no deployable artifact; export is required")
        if result.calibration_status == "failed":
            raise RuntimeError("Training calibration failed; partial artifact cannot be registered")
        if scene_input is not None:
            if result.calibration_status != "ok":
                raise RuntimeError("Scene training requires successful calibration before registration")
            scene_input = revalidate_scene_training_input(scene_input)
        runtime_artifact = Path(result.model_path)
        if runtime_artifact.is_dir():
            # Compatibility for older trainer implementations; still register
            # the exact file rather than the whole mutable training directory.
            runtime_artifact = exported_artifact(
                runtime_artifact, params.get("export_format", "openvino"),
            )
        formats = {".pt": "torch", ".xml": "openvino", ".onnx": "onnx"}
        if not runtime_artifact.is_file() or runtime_artifact.suffix not in formats:
            raise RuntimeError("Training returned a missing or unsupported deployment artifact")
        if torch_backend and runtime_artifact.suffix != ".pt":
            raise RuntimeError(f"{backend} training returned a non-Torch deployment artifact")
        runtime_artifact = exported_artifact(
            runtime_artifact.parent, formats[runtime_artifact.suffix], runtime_artifact,
        )
        metadata = artifact_metadata(runtime_artifact)
        if scene_input is not None:
            calibration = runtime_artifact.parent / "calibration.json"
            if (
                metadata.get("calibration_status") != "ok"
                or metadata.get("camera_id") != camera_id
                or metadata.get("zone_id") != zone_id
                or calibration.is_symlink() or not calibration.is_file()
                or calibration.stat().st_size == 0
            ):
                raise RuntimeError("Scene training requires a finalized calibrated runtime artifact")
            validate_scene_training_calibration(
                model_path=runtime_artifact, scene_training_input=scene_input,
            )

        model_version_id = result.model_version_id
        if selection is not None and model_version_id is None and self._registry is not None:
            raise RuntimeError(
                "Merged-dataset training must register before its temporary dataset is removed"
            )
        if model_version_id is None and self._registry is not None and result.model_path:
            baseline_dir = (
                scene_input.root if scene_input is not None
                else self._baselines_dir / camera_id / zone_id
            )
            model_version_id = self._registry.register(
                model_path=runtime_artifact,
                baseline_dir=baseline_dir,
                camera_id=camera_id,
                model_type=model_type,
                training_params={
                    "zone_id": zone_id,
                    "image_size": params.get("image_size", 256),
                    "export_format": result.actual_export_format or formats[runtime_artifact.suffix],
                    "quantization": result.actual_quantization or (
                        params.get("quantization", "fp16")
                        if runtime_artifact.suffix == ".xml" else "fp32"
                    ),
                    "job_id": job.job_id,
                    "image_count": result.image_count,
                    "train_count": result.train_count,
                    "val_count": result.val_count,
                    "run_id": result.run_id,
                    "training_dir": result.training_dir,
                    "export_dir": result.export_dir,
                    "artifact_manifest": metadata,
                    "baseline_quality": baseline_quality,
                    **({"scene_workflow": dict(scene_input.lineage)}
                       if scene_input is not None else {}),
                    "labeling_entry_ids": _coerce_int_list(
                        params.get("labeling_entry_ids")
                    ),
                },
                registered_by="training_job_executor",
            )

        # Build validation report dict
        val_report_dict = None
        if result.validation_report is not None:
            val_report_dict = result.validation_report.to_dict()

        # Package model (if packager configured).
        # If packaging fails after a CANDIDATE has been registered, retire it
        # only through ModelReleaseService.  Candidate retirement is an
        # audited, CAS-protected applied-state transition and its service-side
        # runtime transaction is deliberately a no-op; a failed package must
        # never touch shadow, primary, or production runtime state.
        artifacts_path = str(runtime_artifact)
        if self._packager and result.model_path:
            try:
                model_dir = (
                    Path(result.training_dir) if result.training_dir
                    else training_directory(runtime_artifact)
                )
                export_dir = (
                    Path(result.export_dir) if result.export_dir
                    else Path(metadata["export_dir"]) if metadata.get("export_dir")
                    else runtime_artifact.parent
                )
                calibration_path = model_dir / "calibration.json"

                pkg_path = self._packager.package(
                    output_dir=self._model_packages_dir,
                    model_dir=model_dir,
                    export_dir=export_dir if export_dir.exists() else None,
                    training_job=job.to_dict(),
                    validation_report=val_report_dict,
                    calibration_path=calibration_path if calibration_path.exists() else None,
                    runtime_artifact=runtime_artifact,
                    backbone_ref=backbone.backbone_version_id if backbone else "pretrained",
                    dataset_ref=job.dataset_version or "",
                    camera_id=camera_id,
                    model_type=job.model_type,
                    training_params=params,
                )
                artifacts_path = str(pkg_path)
                _packaging_error = None
            except Exception as e:
                err_str = str(e)
                logger.error(
                    "job_executor.packaging_failed",
                    job_id=job.job_id,
                    camera_id=camera_id,
                    model_version_id=model_version_id,
                    error=err_str,
                )

                retirement: dict[str, object] | None = None
                retirement_error: str | None = None
                if model_version_id:
                    try:
                        if self._release_service is None:
                            raise RuntimeError(
                                "candidate cleanup release service is unavailable"
                            )
                        operation, retired = self._release_service.request_transition(
                            model_version_id=model_version_id,
                            target_stage=ModelStage.RETIRED.value,
                            requested_by="training_job_executor",
                            reason=f"packaging_failed: {err_str}",
                            idempotency_key=(
                                f"packaging-failed:{job.job_id}:"
                                f"{model_version_id[:32]}"
                            ),
                        )
                        retirement = {
                            "operation_id": operation.operation_id,
                            "status": operation.status,
                            "stage": retired.stage,
                            "runtime_state": operation.to_dict().get("runtime_state"),
                        }
                        if (
                            operation.status != ReleaseOperationStatus.APPLIED.value
                            or retired.stage != ModelStage.RETIRED.value
                            or retired.is_active is not False
                        ):
                            raise RuntimeError(
                                operation.error
                                or "candidate retirement was not atomically applied"
                            )
                    except Exception as retire_err:  # pragma: no cover — defensive
                        retirement_error = str(retire_err)
                        logger.error(
                            "job_executor.candidate_retirement_failed",
                            model_version_id=model_version_id,
                            error=retirement_error,
                        )

                # Keep compensation receipts alongside data-quality evidence.
                # Carry the same per-job metrics to the outer failure CAS so
                # it cannot overwrite them with only a generic quality state.
                packaging_metrics = evidence.setdefault("failure_metrics", {})
                packaging_metrics.update({
                    "packaging_error": err_str,
                    "candidate_retirement": retirement,
                    "candidate_retirement_error": retirement_error,
                    "baseline_quality": baseline_quality,
                })
                try:
                    self._db.update_training_job(
                        job.job_id,
                        expected_status=TrainingJobStatus.RUNNING.value,
                        metrics=json.dumps(packaging_metrics, ensure_ascii=False, allow_nan=False),
                    )
                except Exception as db_err:  # pragma: no cover — defensive
                    logger.error(
                        "job_executor.metrics_update_failed",
                        job_id=job.job_id,
                        error=str(db_err),
                    )

                if retirement_error is not None:
                    raise RuntimeError(
                        f"{err_str}; candidate retirement failed: {retirement_error}"
                    ) from e

                # Re-raise so ``_execute_job`` marks the job FAILED.
                raise
        else:
            _packaging_error = None

        # Build metrics dict
        metrics = {"baseline_quality": baseline_quality}
        if scene_input is not None:
            metrics["scene_workflow"] = dict(scene_input.lineage)
            metrics["calibration_status"] = result.calibration_status
            metrics["train_count"] = scene_input.training_count
            metrics["calibration_count"] = scene_input.calibration_count
        if _packaging_error:
            metrics["packaging_error"] = _packaging_error
        if result.quality_report:
            metrics["quality_grade"] = result.quality_report.grade
        if result.val_stats:
            metrics["val_score_mean"] = result.val_stats.get("mean")
            metrics["val_score_p95"] = result.val_stats.get("p95")
        if result.threshold_recommended:
            metrics["threshold"] = result.threshold_recommended

        completed = self._db.update_training_job(
            job.job_id,
            expected_status=TrainingJobStatus.RUNNING.value,
            status=TrainingJobStatus.COMPLETE.value,
            error=None,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=result.duration_seconds,
            metrics=json.dumps(metrics),
            validation_report=json.dumps(val_report_dict) if val_report_dict else None,
            artifacts_path=artifacts_path,
            model_version_id=model_version_id,
        )
        if not completed:
            raise RuntimeError("Training job state changed before model completion; reconciliation required")

        self._mark_labeling_entries_trained(job, params, model_version_id)

        logger.info(
            "job_executor.head_complete",
            job_id=job.job_id,
            camera_id=camera_id,
            zone_id=zone_id,
            grade=result.quality_report.grade if result.quality_report else "?",
            version=model_version_id,
        )

    def _mark_labeling_entries_trained(
        self,
        job,
        params: dict,
        model_version_id: str | None,
    ) -> None:
        """Consume active-learning labels only after a completed model exists."""
        if not model_version_id:
            return
        entry_ids = _coerce_int_list(params.get("labeling_entry_ids"))
        if not entry_ids:
            return
        mark_fn = getattr(self._db, "mark_labeling_trained", None)
        if not callable(mark_fn):
            logger.warning(
                "job_executor.labeling_mark_unavailable",
                job_id=job.job_id,
                model_version_id=model_version_id,
            )
            return
        try:
            count = mark_fn(entry_ids, trained_into=model_version_id)
            logger.info(
                "job_executor.labeling_entries_trained",
                job_id=job.job_id,
                model_version_id=model_version_id,
                labeling_count=count,
            )
        except Exception as e:
            logger.warning(
                "job_executor.labeling_mark_failed",
                job_id=job.job_id,
                model_version_id=model_version_id,
                error=str(e),
            )


def _coerce_int_list(value: object) -> list[int]:
    """Return a de-duplicated int list from untrusted JSON metadata."""
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    seen: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            entry_id = int(item)
        except (TypeError, ValueError):
            continue
        if entry_id <= 0 or entry_id in seen:
            continue
        seen.add(entry_id)
        ids.append(entry_id)
    return ids
