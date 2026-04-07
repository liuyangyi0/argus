"""Tests for baseline capture sampling strategies (Phase 3)."""

import importlib
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from argus.capture.samplers import (
    UniformSampler,
    ActiveSampler,
    ScheduledSampler,
    get_active_sampler_unavailable_reason,
    _resolve_model_image_size,
)


class FakeFaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, vectors: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            scores = np.full((vectors.shape[0], k), -1.0, dtype=np.float32)
            indices = np.full((vectors.shape[0], k), -1, dtype=np.int64)
            return scores, indices

        similarities = vectors @ self._vectors.T
        best_idx = np.argmax(similarities, axis=1)
        best_scores = similarities[np.arange(vectors.shape[0]), best_idx]
        return best_scores.reshape(-1, 1).astype(np.float32), best_idx.reshape(-1, 1)


class TestUniformSampler:
    def test_always_accepts(self):
        sampler = UniformSampler(duration_hours=1.0, target_frames=100)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        accepted, meta = sampler.should_accept(frame)
        assert accepted is True
        assert meta["strategy"] == "uniform"

    def test_sleep_interval(self):
        sampler = UniformSampler(duration_hours=2.0, target_frames=360)
        interval = sampler.get_sleep_interval()
        assert abs(interval - 20.0) < 0.01  # 7200/360 = 20s


class TestActiveSampler:
    def test_model_image_size_uses_pretrained_cfg(self):
        model = MagicMock()
        model.pretrained_cfg = {"input_size": (3, 518, 518)}

        assert _resolve_model_image_size(model, 224) == 518

    def test_configured_sleep_interval(self):
        sampler = ActiveSampler(diversity_threshold=0.3, sleep_interval_seconds=2.5)

        assert sampler.get_sleep_interval() == 2.5

    def test_dependency_probe_reports_missing_module(self, monkeypatch):
        real_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name == "faiss":
                raise ImportError("No module named 'faiss'")
            return real_import_module(name, package)

        monkeypatch.setattr("argus.capture.samplers.importlib.import_module", fake_import_module)

        reason = get_active_sampler_unavailable_reason()

        assert reason == "缺少依赖: faiss"

    def _make_sampler(self):
        """Create an ActiveSampler with mocked model and an in-memory index."""

        sampler = ActiveSampler(diversity_threshold=0.3)
        sampler._model = MagicMock()  # prevent real model loading
        sampler._torch = MagicMock()  # prevent real torch usage
        sampler._faiss_index = FakeFaissIndex(384)
        return sampler

    def test_first_frame_always_accepted(self):
        """First frame should always be accepted (empty index)."""
        sampler = self._make_sampler()

        fake_feature = np.random.randn(1, 384).astype(np.float32)
        fake_feature /= np.linalg.norm(fake_feature, axis=1, keepdims=True)

        with patch.object(sampler, "_extract_features", return_value=fake_feature):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            accepted, meta = sampler.should_accept(frame)

        assert accepted is True
        assert meta["strategy"] == "active"
        assert meta["index_size"] == 0

    def test_duplicate_frame_rejected(self):
        """An identical frame should be rejected (cosine distance ~0)."""
        sampler = self._make_sampler()

        fixed_feature = np.random.randn(1, 384).astype(np.float32)
        fixed_feature /= np.linalg.norm(fixed_feature, axis=1, keepdims=True)

        with patch.object(sampler, "_extract_features", return_value=fixed_feature.copy()):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            accepted1, _ = sampler.should_accept(frame)
            assert accepted1 is True
            sampler.on_frame_saved(frame, 0)

            accepted2, meta2 = sampler.should_accept(frame)
            assert accepted2 is False
            assert meta2["diversity_score"] < 0.3

    def test_diverse_frame_accepted(self):
        """A very different frame should be accepted."""
        sampler = self._make_sampler()

        feature1 = np.zeros((1, 384), dtype=np.float32)
        feature1[0, 0] = 1.0
        feature2 = np.zeros((1, 384), dtype=np.float32)
        feature2[0, 1] = 1.0  # orthogonal → cosine distance = 1.0

        call_count = [0]
        def mock_extract(frame):
            call_count[0] += 1
            return feature1.copy() if call_count[0] <= 1 else feature2.copy()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(sampler, "_extract_features", side_effect=mock_extract):
            accepted1, _ = sampler.should_accept(frame)
            sampler.on_frame_saved(frame, 0)

            accepted2, meta2 = sampler.should_accept(frame)
            assert accepted2 is True
            assert meta2["diversity_score"] > 0.3


class TestScheduledSampler:
    def test_outside_window_rejected(self):
        """Frame outside all time windows should be rejected."""
        # Use a window that's definitely not now (3am-4am if test runs during day)
        sampler = ScheduledSampler(
            schedule_periods={"test": (3, 4)},
            frames_per_period=10,
            duration_hours=24.0,
        )
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        from datetime import datetime
        now = datetime.now()
        if 3 <= now.hour < 4:
            pytest.skip("Test runs during the 3-4am window")

        accepted, meta = sampler.should_accept(frame)
        assert accepted is False
        assert "outside_window" in str(meta.get("reason", ""))

    def test_inside_window_accepted(self):
        """Frame inside current time window should be accepted."""
        from datetime import datetime
        now = datetime.now()
        # Create a window that includes the current hour
        start = now.hour
        end = (now.hour + 2) % 24

        sampler = ScheduledSampler(
            schedule_periods={"current": (start, end)},
            frames_per_period=100,
            duration_hours=24.0,
        )
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        accepted, meta = sampler.should_accept(frame)
        assert accepted is True

    def test_quota_enforcement(self):
        """Should reject once period quota is full."""
        from datetime import datetime
        now = datetime.now()
        start = now.hour
        end = (now.hour + 2) % 24

        sampler = ScheduledSampler(
            schedule_periods={"current": (start, end)},
            frames_per_period=2,
            duration_hours=24.0,
        )
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Accept 2 frames
        for i in range(2):
            accepted, _ = sampler.should_accept(frame)
            assert accepted is True
            sampler.on_frame_saved(frame, i)

        # Third should be rejected (quota full)
        accepted, meta = sampler.should_accept(frame)
        assert accepted is False
        assert "quota_full" in str(meta.get("reason", ""))
