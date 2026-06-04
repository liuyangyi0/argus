"""Tests for hot_reload_with_warmup model loading."""

import hashlib
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.anomaly.detector import AnomalibDetector


def _mock_anomalib_deploy(mock_engine):
    """Install a fake anomalib.deploy module that returns mock_engine."""
    mod = types.ModuleType("anomalib.deploy")
    mod.OpenVINOInferencer = lambda path: mock_engine
    mod.TorchInferencer = lambda path: mock_engine
    sys.modules["anomalib.deploy"] = mod
    return mod


@pytest.fixture()
def detector():
    """Create an AnomalibDetector without loading a real model."""
    det = AnomalibDetector.__new__(AnomalibDetector)
    det._engine = MagicMock()
    det._model_path = Path("/fake/model")
    det._loaded = True
    det._reload_lock = threading.Lock()
    det._calibration_scores = None
    det._calibration_n = 0
    det._load_calibration = MagicMock()
    det.threshold = 0.5
    return det


@pytest.fixture()
def model_dir(tmp_path):
    """Create a fake model directory with a file."""
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"fake model data")
    return tmp_path


@pytest.fixture(autouse=True)
def mock_deploy():
    """Auto-mock anomalib.deploy for all tests in this module."""
    mock_engine = MagicMock()
    mock_engine.predict.return_value = MagicMock()
    _mock_anomalib_deploy(mock_engine)
    yield mock_engine
    sys.modules.pop("anomalib.deploy", None)


def _compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
    elif path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                h.update(f.read_bytes())
    return h.hexdigest()[:16]


class TestSHA256Verification:

    def test_sha256_match_succeeds(self, detector, model_dir):
        expected_hash = _compute_hash(model_dir)
        result = detector.hot_reload_with_warmup(
            model_dir, expected_hash=expected_hash,
        )
        assert result["success"] is True
        assert result["sha256_verified"] is True

    def test_sha256_mismatch_rejects(self, detector, model_dir):
        result = detector.hot_reload_with_warmup(
            model_dir, expected_hash="wrong_hash_value",
        )
        assert result["success"] is False
        assert "SHA256 mismatch" in result["error"]

    def test_sha256_mismatch_preserves_old_model(self, detector, model_dir):
        old_engine = detector._engine
        result = detector.hot_reload_with_warmup(
            model_dir, expected_hash="wrong_hash_value",
        )
        assert result["success"] is False
        assert detector._engine is old_engine


class TestWarmupLatency:

    def test_latency_within_limit_succeeds(self, detector, model_dir):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        result = detector.hot_reload_with_warmup(
            model_dir,
            warmup_frames=frames,
            max_latency_ms=5000.0,
        )
        assert result["success"] is True
        assert result["warmup_latency_ms"] is not None
        assert result["warmup_latency_ms"] >= 0

    def test_latency_exceeding_limit_rejects(self, detector, model_dir, mock_deploy):
        import time

        def slow_predict(frame):
            time.sleep(0.1)
            return MagicMock()

        mock_deploy.predict.side_effect = slow_predict

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        result = detector.hot_reload_with_warmup(
            model_dir,
            warmup_frames=frames,
            max_latency_ms=10.0,
        )
        assert result["success"] is False
        assert "latency" in result["error"].lower()


class TestAtomicSwap:

    def test_successful_warmup_swaps_model(self, detector, model_dir):
        old_engine = detector._engine
        result = detector.hot_reload_with_warmup(model_dir)
        assert result["success"] is True
        assert detector._engine is not old_engine

    def test_failed_load_preserves_old_model(self, detector, model_dir):
        old_engine = detector._engine
        # Make both inferencers fail
        mod = types.ModuleType("anomalib.deploy")
        mod.OpenVINOInferencer = MagicMock(side_effect=RuntimeError("corrupt"))
        mod.TorchInferencer = MagicMock(side_effect=RuntimeError("corrupt"))
        sys.modules["anomalib.deploy"] = mod

        result = detector.hot_reload_with_warmup(model_dir)
        assert result["success"] is False
        assert detector._engine is old_engine


class TestInferenceSerialization:

    def test_predict_serializes_engine_access(self):
        detector = AnomalibDetector(model_path=None, threshold=0.5, image_size=(16, 16))
        active = 0
        max_active = 0
        active_lock = threading.Lock()

        class _Engine:
            def predict(self, _rgb):
                nonlocal active, max_active
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with active_lock:
                    active -= 1
                return types.SimpleNamespace(
                    pred_score=np.array([0.2], dtype=np.float32),
                    anomaly_map=None,
                )

        detector._engine = _Engine()
        detector._loaded = True
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        threads = [
            threading.Thread(target=detector.predict, args=(frame,))
            for _ in range(2)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        assert all(not thread.is_alive() for thread in threads)
        assert max_active == 1


class TestCallback:

    def test_callback_set_on_completion(self, detector, model_dir):
        done_event = threading.Event()
        result = detector.hot_reload_with_warmup(model_dir, callback=done_event)
        assert result["success"] is True
        assert done_event.is_set()

    def test_callback_set_on_failure(self, detector, model_dir):
        done_event = threading.Event()
        result = detector.hot_reload_with_warmup(
            model_dir, expected_hash="bad", callback=done_event,
        )
        assert result["success"] is False
        assert done_event.is_set()
