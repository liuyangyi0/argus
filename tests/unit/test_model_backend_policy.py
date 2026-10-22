from __future__ import annotations

import pytest

from argus.anomaly.detector import AnomalibDetector
from argus.core.model_discovery import find_runtime_model, resolve_runtime_model_path


def test_torch_cuda_policy_rejects_openvino_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_xml = tmp_path / "data" / "exports" / "cam_01" / "default" / "weights" / "openvino" / "model.xml"
    model_xml.parent.mkdir(parents=True)
    model_xml.write_text("<xml />", encoding="utf-8")

    detector = AnomalibDetector(model_path=model_xml, inference_backend="torch-cuda")

    with pytest.raises(RuntimeError, match="torch-cuda requires model.pt"):
        detector.load()

    assert detector.status.loaded is False
    assert detector.status.last_error == "torch-cuda requires model.pt, got .xml"


def test_torch_cuda_policy_rejects_missing_cuda(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"placeholder")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    detector = AnomalibDetector(model_path=model_pt, inference_backend="torch-cuda")

    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        detector.load()

    assert detector.status.loaded is False
    assert "torch.cuda.is_available" in (detector.status.last_error or "")


def test_explicit_backend_model_discovery_filters_artifact_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    openvino = tmp_path / "data" / "exports" / "cam_01" / "default" / "weights" / "openvino" / "model.xml"
    torch_model = tmp_path / "data" / "exports" / "cam_01" / "default" / "weights" / "torch" / "model.pt"
    openvino.parent.mkdir(parents=True)
    torch_model.parent.mkdir(parents=True)
    openvino.write_text("<xml />", encoding="utf-8")
    torch_model.write_bytes(b"pt")

    assert find_runtime_model("cam_01", backend="openvino") == openvino.resolve()
    assert find_runtime_model("cam_01", backend="torch-cuda") == torch_model.resolve()
    assert resolve_runtime_model_path(openvino, "cam_01", backend="torch-cuda") is None
    assert resolve_runtime_model_path(torch_model, "cam_01", backend="openvino") is None
