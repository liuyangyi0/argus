"""Manual training records keep the exact versioned export contract."""

import json
from pathlib import Path

import pytest

from argus.anomaly.trainer import QualityReport, TrainingResult, TrainingStatus
from argus.dashboard.routes.baseline import _train_model_task
from argus.storage.database import Database
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import TrainingRecord


def prepare_training_result(tmp_path, monkeypatch):
    baselines = tmp_path / "baselines"
    current = baselines / "cam_01" / "default" / "v001"
    current.mkdir(parents=True)
    (current.parent / "current.txt").write_text("v001", encoding="utf-8")
    (current / "normal.png").write_bytes(b"baseline")
    export_dir = tmp_path / "exports" / "cam_01" / "default" / "run-unique"
    export_dir.mkdir(parents=True)
    artifact = export_dir / "model.pt"
    artifact.write_bytes(b"exact-runtime-artifact")
    training_dir = tmp_path / "models" / "cam_01" / "default" / "run-unique"
    training_dir.mkdir(parents=True)
    result = TrainingResult(
        status=TrainingStatus.COMPLETE,
        model_path=str(artifact),
        export_dir=str(export_dir),
        training_dir=str(training_dir),
        run_id="run-unique",
        image_count=30,
        train_count=24,
        val_count=6,
        quality_report=QualityReport(grade="A"),
        actual_export_format="torch",
        actual_quantization="fp32",
    )
    monkeypatch.setattr(
        "argus.anomaly.trainer.ModelTrainer.train",
        lambda self, **kwargs: result,
    )
    database_url = f"sqlite:///{tmp_path / 'training.db'}"
    kwargs = dict(
        baselines_dir=str(baselines),
        models_dir=str(tmp_path / "models"),
        camera_id="cam_01",
        model_type="patchcore",
        export_format="openvino",
        quantization="fp16",
        database_url=database_url,
    )
    return result, kwargs


def test_manual_training_records_actual_format_and_exact_export(tmp_path, monkeypatch):
    expected, kwargs = prepare_training_result(tmp_path, monkeypatch)
    result = _train_model_task(lambda *_: None, **kwargs)
    database = Database(database_url=kwargs["database_url"])
    database.initialize()
    try:
        with database.get_session() as session:
            record = session.query(TrainingRecord).one()
            assert record.export_path == expected.export_dir
            assert record.model_path == expected.model_path
            assert record.export_format == "torch"
        registered = ModelRegistry(database.get_session).get_by_version_id(
            result["model_version_id"],
        )
        params = json.loads(registered.training_params)
        assert params["quantization"] == "fp32"
        assert params["run_id"] == expected.run_id
        assert params["training_dir"] == expected.training_dir
        assert result["export_dir"] == expected.export_dir
    finally:
        database.close()


def test_candidate_registration_failure_does_not_report_training_complete(tmp_path, monkeypatch):
    expected, kwargs = prepare_training_result(tmp_path, monkeypatch)

    def reject_candidate(*args, **kwargs):
        raise RuntimeError("injected registry failure")

    monkeypatch.setattr(ModelRegistry, "register", reject_candidate)
    with pytest.raises(RuntimeError, match="训练产物登记失败"):
        _train_model_task(lambda *_: None, **kwargs)

    assert Path(expected.model_path).is_file()
    database = Database(database_url=kwargs["database_url"])
    database.initialize()
    try:
        with database.get_session() as session:
            assert session.query(TrainingRecord).count() == 0
    finally:
        database.close()
