"""Device-policy protocol tests only: no real SDK, model, or image processing."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from argus.anomaly.trainer import ModelTrainer, _training_device_kwargs


@pytest.mark.parametrize("backend,available,expected", [
    ("torch-cuda", True, {"accelerator": "gpu", "devices": 1}),
    ("torch-cpu", False, {"accelerator": "cpu", "devices": 1}),
    ("torch-cpu", True, {"accelerator": "cpu", "devices": 1}),
    ("auto", False, {}),
    ("openvino", True, {}),
])
def test_training_device_policy(backend, available, expected):
    assert _training_device_kwargs(backend, cuda_available=available) == expected


def test_required_cuda_cannot_fall_back_to_cpu():
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        _training_device_kwargs("torch-cuda", cuda_available=False)


def _fake_training_sdks(monkeypatch, available):
    folder, engine, model = MagicMock(), MagicMock(), MagicMock()
    values = {
        "torch": {"float32": object(), "cuda": SimpleNamespace(
            is_available=lambda: available, mem_get_info=lambda: (16 * 1024**3, 16 * 1024**3),
        )},
        "anomalib.data": {"Folder": folder},
        "anomalib.engine": {"Engine": engine},
        "anomalib.models": {"Patchcore": model},
        "torchvision.transforms.v2": {key: MagicMock() for key in ("Compose", "Resize", "ToDtype", "ToImage")},
        "lightning.pytorch.callbacks": {"Timer": MagicMock()},
    }
    for name in ("anomalib", "torchvision", "torchvision.transforms", "lightning", "lightning.pytorch"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name, attrs in values.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    return folder, engine, model


@pytest.mark.parametrize("available", [True, False])
def test_cuda_policy_reaches_real_training_entrypoint_before_model_creation(monkeypatch, tmp_path, available):
    folder, engine, model = _fake_training_sdks(monkeypatch, available)
    trainer = ModelTrainer(SimpleNamespace())
    monkeypatch.setattr(trainer, "_free_gpu_for_training", lambda: None)
    kwargs = dict(data_dir=tmp_path, output_dir=tmp_path / "output", model_type="patchcore", image_size=256,
        anomaly_config=SimpleNamespace(inference_backend="torch-cuda"))
    if available:
        trainer._train_anomalib(**kwargs)
        assert engine.call_args.kwargs["accelerator"] == "gpu"
        assert engine.call_args.kwargs["devices"] == 1
        engine.return_value.fit.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
            trainer._train_anomalib(**kwargs)
        folder.assert_not_called()
        model.assert_not_called()
        engine.assert_not_called()


def test_output_smoke_and_calibration_keep_explicit_backend(monkeypatch, tmp_path):
    detector = MagicMock()
    detector.predict.return_value = SimpleNamespace(anomaly_score=0.2, detection_failed=False)
    constructor = MagicMock(return_value=detector)
    module = ModuleType("argus.anomaly.detector")
    module.AnomalibDetector = constructor
    monkeypatch.setitem(sys.modules, "argus.anomaly.detector", module)
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"protocol-only-not-a-model")
    monkeypatch.setattr("argus.anomaly.trainer._list_images", lambda _: [tmp_path / "mock.png"])
    monkeypatch.setattr("argus.anomaly.trainer.cv2.imread", lambda _: object())
    validation, loaded = ModelTrainer(SimpleNamespace())._validate_output(
        tmp_path, None, tmp_path, artifact_path=artifact, inference_backend="torch-cuda",
    )
    assert validation["smoke_test_passed"] is True and loaded is detector
    assert constructor.call_args.kwargs["inference_backend"] == "torch-cuda"
    assert constructor.call_args.kwargs["enable_calibration"] is False
