"""Tests for anomaly trainer export — static input shape for OpenVINO.

Regression: Dinomaly2 exported without explicit ``input_size`` produced an
IR with dynamic ``?,3,?,?`` input that crashed inference with
``Broadcast Check failed``. The fix locks the exported input shape to the
training image_size; these tests verify the export call carries that arg.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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
