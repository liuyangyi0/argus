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

from argus.anomaly.trainer import TrainingStatus
from argus.storage.models import (
    BackboneRecord,
    TrainingJobStatus,
    TrainingJobType,
)

if TYPE_CHECKING:
    from argus.anomaly.backbone_trainer import BackboneTrainer
    from argus.anomaly.model_package import ModelPackager
    from argus.anomaly.trainer import ModelTrainer
    from argus.anomaly.training_validator import TrainingValidator
    from argus.storage.database import Database
    from argus.storage.model_registry import ModelRegistry

logger = structlog.get_logger()

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
    ):
        self._db = database
        self._trainer = trainer
        self._backbone_trainer = backbone_trainer
        self._validator = training_validator
        self._packager = model_packager
        self._registry = model_registry
        self._baselines_dir = baselines_dir
        self._model_packages_dir = model_packages_dir
        self._model_packages_dir.mkdir(parents=True, exist_ok=True)

    def recover_stale_jobs(self, max_running_hours: float = 6.0) -> int:
        """Recover jobs stuck in RUNNING state (e.g. after process crash).

        Jobs that have been RUNNING for longer than max_running_hours are
        marked as FAILED with a timeout error.
        """
        running_jobs = self._db.list_training_jobs(status=TrainingJobStatus.RUNNING.value)
        recovered = 0
        now = datetime.now(timezone.utc)
        for job in running_jobs:
            if job.started_at is None:
                continue
            elapsed_hours = (now - job.started_at).total_seconds() / 3600
            if elapsed_hours > max_running_hours:
                self._db.update_training_job(
                    job.job_id,
                    status=TrainingJobStatus.FAILED.value,
                    error=f"Job timed out after {elapsed_hours:.1f}h (limit: {max_running_hours}h)",
                    completed_at=now,
                    duration_seconds=elapsed_hours * 3600,
                )
                logger.warning(
                    "job_executor.stale_job_recovered",
                    job_id=job.job_id,
                    elapsed_hours=round(elapsed_hours, 1),
                )
                recovered += 1
        return recovered

    def process_queued_jobs(self) -> int:
        """Process all queued training jobs. Returns number processed."""
        # First, recover any jobs stuck in RUNNING from prior crashes
        self.recover_stale_jobs()

        jobs = self._db.list_training_jobs(status=TrainingJobStatus.QUEUED.value)
        processed = 0
        for job in jobs:
            try:
                self._execute_job(job)
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

    def _execute_job(self, job) -> None:
        """Internal: execute a job record directly (avoids re-fetching)."""
        job_id = job.job_id
        now = datetime.now(timezone.utc)
        self._db.update_training_job(
            job_id, status=TrainingJobStatus.RUNNING.value, started_at=now,
        )

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
            self._db.update_training_job(
                job_id,
                status=TrainingJobStatus.FAILED.value,
                error=str(e),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=duration,
            )
            logger.error("job_executor.failed", job_id=job_id, error=str(e))

    def _execute_backbone(self, job) -> None:
        """Execute SSL backbone fine-tuning job."""
        if self._backbone_trainer is None:
            raise RuntimeError("BackboneTrainer not configured")

        # Parse and validate hyperparameters
        params = json.loads(job.hyperparameters) if job.hyperparameters else {}
        param_errors = validate_hyperparameters(params)
        if param_errors:
            raise ValueError(f"Invalid hyperparameters: {'; '.join(param_errors)}")

        # Get all camera IDs from config (or use all available)
        camera_ids = []
        for cam_dir in self._baselines_dir.iterdir():
            if cam_dir.is_dir() and not cam_dir.name.startswith("."):
                camera_ids.append(cam_dir.name)

        result = self._backbone_trainer.train(
            camera_ids=camera_ids,
            baselines_dir=self._baselines_dir,
            backbone_type=params.get("backbone_type", "dinov2_vitb14"),
            epochs=params.get("epochs", 5),
            lr=params.get("lr", 1e-5),
        )

        if not result.success:
            raise RuntimeError(result.error or "Backbone training failed")

        backbone_record = BackboneRecord(
            backbone_version_id=f"ssl_{Path(result.checkpoint_path).stem}",
            backbone_type=params.get("backbone_type", "dinov2_vitb14"),
            checkpoint_path=result.checkpoint_path,
            checkpoint_hash=result.checkpoint_hash,
            dataset_hash=result.dataset_hash,
            camera_ids_used=json.dumps(camera_ids),
            training_job_id=job.job_id,
            is_active=False,
        )
        self._db.save_backbone(backbone_record)
        self._db.activate_backbone(backbone_record.backbone_version_id)

        # Update job
        self._db.update_training_job(
            job.job_id,
            status=TrainingJobStatus.COMPLETE.value,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=result.duration_seconds,
            metrics=json.dumps({
                "final_loss": result.final_loss,
                "total_images": result.total_images,
                "epochs": result.epochs_completed,
            }),
            artifacts_path=result.checkpoint_path,
        )

        logger.info(
            "job_executor.backbone_complete",
            job_id=job.job_id,
            backbone=backbone_record.backbone_version_id,
        )

    def _execute_head(self, job) -> None:
        """Execute anomaly detection head training job."""
        camera_id = job.camera_id
        if not camera_id:
            raise ValueError("camera_id is required for anomaly_head jobs")

        # Resolve backbone checkpoint
        backbone_checkpoint = None
        backbone = self._db.get_active_backbone()
        if backbone:
            backbone_checkpoint = backbone.checkpoint_path

        # Parse and validate hyperparameters
        params = json.loads(job.hyperparameters) if job.hyperparameters else {}
        param_errors = validate_hyperparameters(params)
        if param_errors:
            raise ValueError(f"Invalid hyperparameters: {'; '.join(param_errors)}")

        # Run training (trainer already has validation wired in)
        result = self._trainer.train(
            camera_id=camera_id,
            zone_id=job.zone_id or "default",
            model_type=job.model_type or params.get("model_type", "patchcore"),
            image_size=params.get("image_size", 256),
            export_format=params.get("export_format", "openvino"),
            quantization=params.get("quantization", "fp16"),
            backbone_checkpoint=backbone_checkpoint,
            skip_baseline_validation=params.get("skip_baseline_validation", False),
        )

        if result.status.value != TrainingStatus.COMPLETE.value:
            raise RuntimeError(result.error or f"Training failed with status {result.status}")

        model_version_id = result.model_version_id
        if model_version_id is None and self._registry is not None and result.model_path:
            baseline_dir = self._baselines_dir / camera_id / (job.zone_id or "default")
            model_version_id = self._registry.register(
                model_path=result.model_path,
                baseline_dir=baseline_dir,
                camera_id=camera_id,
                model_type=job.model_type or params.get("model_type", "patchcore"),
                training_params={
                    "image_size": params.get("image_size", 256),
                    "export_format": params.get("export_format", "openvino"),
                    "quantization": params.get("quantization", "fp16"),
                    "job_id": job.job_id,
                },
            )

        # Build validation report dict
        val_report_dict = None
        if result.validation_report is not None:
            val_report_dict = result.validation_report.to_dict()

        # Package model (if packager configured)
        artifacts_path = result.model_path
        if self._packager and result.model_path:
            try:
                model_dir = Path(result.model_path)
                export_dir = self._trainer.exports_dir / camera_id / (job.zone_id or "default")
                calibration_path = model_dir / "calibration.json"

                pkg_path = self._packager.package(
                    output_dir=self._model_packages_dir,
                    model_dir=model_dir,
                    export_dir=export_dir if export_dir.exists() else None,
                    training_job=job.to_dict(),
                    validation_report=val_report_dict,
                    calibration_path=calibration_path if calibration_path.exists() else None,
                    backbone_ref=backbone.backbone_version_id if backbone else "pretrained",
                    dataset_ref=job.dataset_version or "",
                    camera_id=camera_id,
                    model_type=job.model_type,
                    training_params=params,
                )
                artifacts_path = str(pkg_path)
            except Exception as e:
                logger.error("job_executor.packaging_failed", error=str(e))
                _packaging_error = str(e)
            else:
                _packaging_error = None
        else:
            _packaging_error = None

        # Build metrics dict
        metrics = {}
        if _packaging_error:
            metrics["packaging_error"] = _packaging_error
        if result.quality_report:
            metrics["quality_grade"] = result.quality_report.grade
        if result.val_stats:
            metrics["val_score_mean"] = result.val_stats.get("mean")
            metrics["val_score_p95"] = result.val_stats.get("p95")
        if result.threshold_recommended:
            metrics["threshold"] = result.threshold_recommended

        self._db.update_training_job(
            job.job_id,
            status=TrainingJobStatus.COMPLETE.value,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=result.duration_seconds,
            metrics=json.dumps(metrics),
            validation_report=json.dumps(val_report_dict) if val_report_dict else None,
            artifacts_path=artifacts_path,
            model_version_id=model_version_id,
        )

        logger.info(
            "job_executor.head_complete",
            job_id=job.job_id,
            camera_id=camera_id,
            grade=result.quality_report.grade if result.quality_report else "?",
            version=model_version_id,
        )
