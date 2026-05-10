"""Tests for anomaly trainer export — static input shape for OpenVINO.

Regression: Dinomaly2 exported without explicit ``input_size`` produced an
IR with dynamic ``?,3,?,?`` input that crashed inference with
``Broadcast Check failed``. The fix locks the exported input shape to the
training image_size; these tests verify the export call carries that arg.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from argus.anomaly.trainer import ModelTrainer


class TestExportStaticInputShape:
    def test_openvino_export_passes_static_input_size(self, tmp_path):
        """_export_model must pass input_size=(H,W) for openvino exports."""
        mock_engine = MagicMock()
        mock_model = MagicMock()
        export_path = str(tmp_path / "export")
        Path(export_path).mkdir(parents=True, exist_ok=True)

        ModelTrainer._export_model(
            engine=mock_engine,
            model=mock_model,
            export_format="openvino",
            export_path=export_path,
            image_size=256,
        )

        mock_engine.export.assert_called_once()
        kwargs = mock_engine.export.call_args.kwargs
        assert kwargs["input_size"] == (256, 256)
        # dynamic_axes={} required for static IR — see d1884c6 (PR #22 fix)
        assert kwargs["onnx_kwargs"] == {"dynamo": False, "dynamic_axes": {}}

    def test_onnx_export_also_locks_shape(self, tmp_path):
        mock_engine = MagicMock()
        export_path = str(tmp_path / "export")
        Path(export_path).mkdir(parents=True, exist_ok=True)

        ModelTrainer._export_model(
            engine=mock_engine,
            model=MagicMock(),
            export_format="onnx",
            export_path=export_path,
            image_size=384,
        )

        kwargs = mock_engine.export.call_args.kwargs
        assert kwargs["input_size"] == (384, 384)

    def test_torch_export_does_not_set_input_size(self, tmp_path):
        """Torch .pt export does not need the static-shape kwargs at all."""
        mock_engine = MagicMock()
        export_path = str(tmp_path / "export")
        Path(export_path).mkdir(parents=True, exist_ok=True)

        ModelTrainer._export_model(
            engine=mock_engine,
            model=MagicMock(),
            export_format="torch",
            export_path=export_path,
            image_size=256,
        )

        kwargs = mock_engine.export.call_args.kwargs
        assert "input_size" not in kwargs
        assert "onnx_kwargs" not in kwargs


class TestAssertOpenvinoStatic:
    """Guard: reject an exported IR whose input still has a dynamic dim.

    Symptom when it slips through: CPU inference raises
    'Broadcast Check failed, Value -1 not in range' at first predict().
    """

    def _make_model_mock(self, *, static: bool):
        """Build a mock ov.Model with one input whose shape is static/dynamic."""
        input_tensor = MagicMock()
        input_tensor.any_name = "input"
        input_tensor.partial_shape.is_static = static
        model = MagicMock()
        model.inputs = [input_tensor]
        return model

    def test_raises_on_dynamic_input(self, tmp_path):
        xml = tmp_path / "model.xml"
        xml.write_text("<net/>")
        fake_model = self._make_model_mock(static=False)
        with patch("openvino.Core") as core_cls:
            core_cls.return_value.read_model.return_value = fake_model
            with pytest.raises(RuntimeError, match="non-static input shape"):
                ModelTrainer._assert_openvino_static(tmp_path)

    def test_passes_on_static_input(self, tmp_path):
        xml = tmp_path / "model.xml"
        xml.write_text("<net/>")
        fake_model = self._make_model_mock(static=True)
        with patch("openvino.Core") as core_cls:
            core_cls.return_value.read_model.return_value = fake_model
            ModelTrainer._assert_openvino_static(tmp_path)  # no raise

    def test_noop_when_no_xml_present(self, tmp_path):
        """No IR on disk yet → nothing to check; other code reports missing."""
        with patch("openvino.Core") as core_cls:
            ModelTrainer._assert_openvino_static(tmp_path)
            core_cls.assert_not_called()
