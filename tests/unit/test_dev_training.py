from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from argus.anomaly.detector import AnomalibDetector
from argus.anomaly.baseline import BaselineManager
from argus.anomaly.trainer import TrainingStatus
from argus.runtime.dev_training import DevFastModelTrainer


def test_dev_fast_model_trainer_creates_model_and_reexport_artifacts(tmp_path):
    trainer = DevFastModelTrainer(
        baseline_manager=BaselineManager(tmp_path / "baselines"),
        models_dir=tmp_path / "models",
        exports_dir=tmp_path / "exports",
    )

    result = trainer.train(camera_id="cam_01", zone_id="default")

    assert result.status == TrainingStatus.COMPLETE
    assert result.model_path
    model_dir = tmp_path / "models" / "cam_01" / "default"
    created_model_dir = Path(result.model_path)
    assert created_model_dir.parent == model_dir
    assert (created_model_dir / "model.xml").is_file()
    assert (created_model_dir / "metadata.json").is_file()

    reexport = trainer.reexport_model(
        model_dir=created_model_dir,
        export_format="openvino",
        quantization="fp16",
    )

    assert reexport["status"] == "ok"
    assert reexport["dev_fast_training"] is True
    assert (tmp_path / "exports" / "cam_01" / "default" / "reexport-openvino-fp16" / "model.xml").is_file()


def test_dev_fast_model_loads_in_anomalib_detector(tmp_path):
    trainer = DevFastModelTrainer(
        baseline_manager=BaselineManager(tmp_path / "baselines"),
        models_dir=tmp_path / "models",
        exports_dir=tmp_path / "exports",
    )
    result = trainer.train(camera_id="cam_01", zone_id="default")

    detector = AnomalibDetector(
        model_path=Path(result.model_path) / "model.xml",
        camera_id="cam_01",
    )
    try:
        assert detector.load() is True
        prediction = detector.predict(np.full((480, 640, 3), 128, dtype=np.uint8))

        assert prediction.detection_failed is False
        assert prediction.anomaly_map is not None
        assert prediction.anomaly_map.shape == (256, 256)
        assert detector.status.loaded is True
        assert detector.status.backend == "openvino"
    finally:
        detector._engine = None
        gc.collect()


def test_detector_syncs_dev_fast_model_input_size(tmp_path):
    trainer = DevFastModelTrainer(
        baseline_manager=BaselineManager(tmp_path / "baselines"),
        models_dir=tmp_path / "models",
        exports_dir=tmp_path / "exports",
    )
    result = trainer.train(camera_id="cam_01", zone_id="default", image_size=64)

    detector = AnomalibDetector(
        model_path=Path(result.model_path) / "model.xml",
        camera_id="cam_01",
    )
    try:
        assert detector.load() is True
        assert detector.image_size == (64, 64)
        prediction = detector.predict(np.full((480, 640, 3), 128, dtype=np.uint8))

        assert prediction.detection_failed is False
        assert prediction.anomaly_map is not None
        assert prediction.anomaly_map.shape == (64, 64)
        assert detector.status.image_size == (64, 64)
        assert detector.status.consecutive_failures == 0
    finally:
        detector._engine = None
        gc.collect()
