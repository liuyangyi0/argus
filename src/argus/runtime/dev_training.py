"""Fast local trainer used only for development smoke tests."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from argus.anomaly.trainer import QualityReport, TrainingResult, TrainingStatus


def write_dev_openvino_ir(
    target_dir: Path,
    *,
    camera_id: str,
    zone_id: str,
    model_type: str,
    image_size: int,
    quantization: str,
) -> None:
    """Write a tiny valid OpenVINO IR with anomalib-compatible outputs."""
    try:
        import openvino as ov
        from openvino import opset8 as ops
    except Exception as exc:  # pragma: no cover - dependency is required by project
        raise RuntimeError("OpenVINO is required for --dev-fast-training") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    input_tensor = ops.parameter(
        [1, 3, int(image_size), int(image_size)],
        dtype=np.float32,
        name="input",
    )
    score_axes = ops.constant(np.array([1, 2, 3], dtype=np.int64))
    pred_score = ops.reduce_mean(input_tensor, score_axes, keep_dims=False)
    pred_score.set_friendly_name("pred_score")
    pred_score.output(0).get_tensor().set_names({"pred_score"})

    map_axes = ops.constant(np.array([1], dtype=np.int64))
    anomaly_map = ops.reduce_mean(input_tensor, map_axes, keep_dims=False)
    anomaly_map.set_friendly_name("anomaly_map")
    anomaly_map.output(0).get_tensor().set_names({"anomaly_map"})

    model = ov.Model([pred_score, anomaly_map], [input_tensor], "argus_dev_fast")
    ov.save_model(model, target_dir / "model.xml")

    (target_dir / "calibration.json").write_text(
        json.dumps({"threshold": 0.66}, ensure_ascii=False),
        encoding="utf-8",
    )
    (target_dir / "metadata.json").write_text(
        json.dumps(
            {
                "camera_id": camera_id,
                "zone_id": zone_id,
                "model_type": model_type,
                "image_size": image_size,
                "quantization": quantization,
                "dev_fast_training": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class DevFastModelTrainer:
    """Create deterministic model artifacts without running anomalib training."""

    def __init__(
        self,
        *,
        baseline_manager,
        models_dir: str | Path,
        exports_dir: str | Path,
    ) -> None:
        self._baseline_manager = baseline_manager
        self._models_dir = Path(models_dir)
        self._exports_dir = Path(exports_dir)
        self._status = TrainingStatus.IDLE
        self._last_result: TrainingResult | None = None

    @property
    def exports_dir(self) -> Path:
        return self._exports_dir

    @property
    def status(self) -> TrainingStatus:
        return self._status

    @property
    def last_result(self) -> TrainingResult | None:
        return self._last_result

    def train(
        self,
        *,
        camera_id: str,
        zone_id: str = "default",
        model_type: str = "patchcore",
        image_size: int = 256,
        export_format: str | None = "openvino",
        quantization: str = "fp16",
        **_: object,
    ) -> TrainingResult:
        start = time.monotonic()
        self._status = TrainingStatus.TRAINING

        baseline_dir = self._baseline_manager.get_baseline_dir(camera_id, zone_id)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        if not any(baseline_dir.iterdir()):
            (baseline_dir / "dev_baseline.png").write_bytes(b"dev-baseline")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        model_dir = self._models_dir / camera_id / zone_id / f"dev-fast-{stamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._write_openvino_stub(
            model_dir,
            camera_id=camera_id,
            zone_id=zone_id,
            model_type=model_type,
            image_size=image_size,
            quantization=quantization,
        )

        if export_format:
            export_dir = self._exports_dir / camera_id / zone_id / f"dev-fast-{stamp}"
            export_dir.mkdir(parents=True, exist_ok=True)
            self._write_openvino_stub(
                export_dir,
                camera_id=camera_id,
                zone_id=zone_id,
                model_type=model_type,
                image_size=image_size,
                quantization=quantization,
            )

        result = TrainingResult(
            status=TrainingStatus.COMPLETE,
            model_path=str(model_dir),
            duration_seconds=time.monotonic() - start,
            image_count=30,
            train_count=24,
            val_count=6,
            val_stats={"mean": 0.12, "p95": 0.41},
            quality_report=QualityReport(
                grade="B",
                score_stats={"mean": 0.12, "p95": 0.41},
                threshold_recommended=0.66,
                suggestions=["dev-fast-training"],
            ),
            threshold_recommended=0.66,
            baseline_validation_skipped=True,
        )
        self._status = TrainingStatus.COMPLETE
        self._last_result = result
        return result

    def reexport_model(
        self,
        *,
        model_dir: Path,
        export_format: str = "openvino",
        quantization: str = "fp16",
        model_type: str = "patchcore",
    ) -> dict:
        camera_id = model_dir.parts[-3] if len(model_dir.parts) >= 3 else "unknown"
        zone_id = model_dir.parts[-2] if len(model_dir.parts) >= 2 else "default"
        export_dir = self._exports_dir / camera_id / zone_id / f"reexport-{export_format}-{quantization}"
        export_dir.mkdir(parents=True, exist_ok=True)
        self._write_openvino_stub(
            export_dir,
            camera_id=camera_id,
            zone_id=zone_id,
            model_type=model_type,
            image_size=256,
            quantization=quantization,
        )
        return {
            "status": "ok",
            "export_path": str(export_dir),
            "format": export_format,
            "quantization": quantization,
            "dev_fast_training": True,
        }

    @staticmethod
    def _write_openvino_stub(
        target_dir: Path,
        *,
        camera_id: str,
        zone_id: str,
        model_type: str,
        image_size: int,
        quantization: str,
    ) -> None:
        write_dev_openvino_ir(
            target_dir,
            camera_id=camera_id,
            zone_id=zone_id,
            model_type=model_type,
            image_size=image_size,
            quantization=quantization,
        )
