"""Model training orchestration.

Coordinates the full training pipeline: validate baselines, train model,
export to optimized format, and swap the live model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

from argus.anomaly.baseline import BaselineManager

logger = structlog.get_logger()


class TrainingStatus(str, Enum):
    IDLE = "idle"
    TRAINING = "training"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TrainingResult:
    status: TrainingStatus
    model_path: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    image_count: int = 0


class ModelTrainer:
    """Orchestrates anomaly model training and export.

    Wraps Anomalib's training API into a simple interface that can be
    called from the scheduler or the dashboard.
    """

    def __init__(
        self,
        baseline_manager: BaselineManager,
        models_dir: str | Path = "data/models",
        exports_dir: str | Path = "data/exports",
    ):
        self._baseline_manager = baseline_manager
        self._models_dir = Path(models_dir)
        self._exports_dir = Path(exports_dir)
        self._status = TrainingStatus.IDLE
        self._last_result: TrainingResult | None = None

    @property
    def status(self) -> TrainingStatus:
        return self._status

    @property
    def last_result(self) -> TrainingResult | None:
        return self._last_result

    def train(
        self,
        camera_id: str,
        zone_id: str = "default",
        model_type: str = "patchcore",
        image_size: int = 256,
        export_format: str | None = "openvino",
    ) -> TrainingResult:
        """Train an anomaly detection model for a specific camera/zone.

        This is a blocking call that can take several minutes depending
        on the number of baseline images and the model type.
        """
        start = time.monotonic()
        self._status = TrainingStatus.TRAINING

        # Validate baselines
        baseline_dir = self._baseline_manager.get_baseline_dir(camera_id, zone_id)
        image_count = self._baseline_manager.count_images(camera_id, zone_id)

        if image_count < 10:
            result = TrainingResult(
                status=TrainingStatus.FAILED,
                error=f"Insufficient baselines: {image_count} (need >= 10)",
                image_count=image_count,
            )
            self._status = TrainingStatus.FAILED
            self._last_result = result
            return result

        logger.info(
            "training.started",
            camera_id=camera_id,
            zone_id=zone_id,
            model_type=model_type,
            images=image_count,
        )

        output_dir = self._models_dir / camera_id / zone_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            engine, model = self._train_anomalib(
                data_dir=baseline_dir,
                output_dir=output_dir,
                model_type=model_type,
                image_size=image_size,
            )
        except ImportError:
            result = TrainingResult(
                status=TrainingStatus.FAILED,
                error="anomalib not installed",
                image_count=image_count,
                duration_seconds=time.monotonic() - start,
            )
            self._status = TrainingStatus.FAILED
            self._last_result = result
            return result
        except Exception as e:
            logger.error("training.failed", error=str(e))
            result = TrainingResult(
                status=TrainingStatus.FAILED,
                error=str(e),
                image_count=image_count,
                duration_seconds=time.monotonic() - start,
            )
            self._status = TrainingStatus.FAILED
            self._last_result = result
            return result

        # Export
        export_path = None
        if export_format:
            self._status = TrainingStatus.EXPORTING
            try:
                export_path = str(self._exports_dir / camera_id / zone_id)
                logger.info("training.exporting", format=export_format, path=export_path)
                self._export_model(
                    engine=engine,
                    model=model,
                    export_format=export_format,
                    export_path=export_path,
                )
            except Exception as e:
                logger.error("training.export_failed", error=str(e))

        # Validate output: at least one model file must exist
        model_files = (
            list(output_dir.rglob("*.ckpt"))
            + list(output_dir.rglob("*.pt"))
            + list(output_dir.rglob("*.xml"))
        )
        if not model_files:
            duration = time.monotonic() - start
            result = TrainingResult(
                status=TrainingStatus.FAILED,
                error="Training produced no model files",
                image_count=image_count,
                duration_seconds=duration,
            )
            self._status = TrainingStatus.FAILED
            self._last_result = result
            return result

        if export_format and export_path:
            export_dir = Path(export_path)
            export_files = (
                list(export_dir.rglob("*.xml"))
                + list(export_dir.rglob("*.onnx"))
            ) if export_dir.exists() else []
            if not export_files:
                logger.warning(
                    "training.export_files_missing",
                    export_path=export_path,
                    format=export_format,
                )

        duration = time.monotonic() - start
        result = TrainingResult(
            status=TrainingStatus.COMPLETE,
            model_path=str(output_dir),
            duration_seconds=duration,
            image_count=image_count,
        )
        self._status = TrainingStatus.COMPLETE
        self._last_result = result

        logger.info(
            "training.complete",
            camera_id=camera_id,
            duration=f"{duration:.1f}s",
            model_path=str(output_dir),
        )
        return result

    def _train_anomalib(
        self,
        data_dir: Path,
        output_dir: Path,
        model_type: str,
        image_size: int,
    ) -> tuple:
        """Execute Anomalib training. Raises ImportError if anomalib is not installed.

        Returns (engine, model) so the caller can use them for export.
        """
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd, Patchcore

        datamodule = Folder(
            name="baseline",
            root=str(data_dir.parent),
            normal_dir=data_dir.name,
            image_size=(image_size, image_size),
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
        )

        if model_type == "efficient_ad":
            model = EfficientAd()
        else:
            model = Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.1,
            )

        max_epochs = 70 if model_type == "efficient_ad" else 1

        engine = Engine(
            default_root_dir=str(output_dir),
            max_epochs=max_epochs,
        )

        engine.fit(model=model, datamodule=datamodule)

        return engine, model

    @staticmethod
    def _export_model(
        engine,
        model,
        export_format: str,
        export_path: str,
    ) -> None:
        """Export trained model to an optimized inference format.

        CRIT-02/HIGH-12: Use correct Anomalib 2.x export API (export_mode parameter).
        """
        # Anomalib 2.x uses export_mode (str), not export_type (enum)
        engine.export(
            model=model,
            export_mode=export_format,  # "openvino" or "onnx"
        )
        logger.info("training.exported", format=export_format, path=export_path)
