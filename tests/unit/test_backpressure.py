"""Tests for frame queue backpressure (Phase 4)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from argus.core.pipeline import PipelineStats


class TestPipelineStatsBackpressure:
    """Test backpressure field in PipelineStats."""

    def test_default_zero(self):
        stats = PipelineStats()
        assert stats.frames_dropped_backpressure == 0

    def test_increment(self):
        stats = PipelineStats()
        stats.frames_dropped_backpressure += 5
        assert stats.frames_dropped_backpressure == 5


class TestCameraManagerBackpressure:
    """Test backpressure tracking in CameraManager."""

    def _make_manager(self):
        """Create a minimal CameraManager for testing."""
        from argus.capture.manager import CameraManager
        from argus.config.schema import AlertConfig

        cam = MagicMock()
        cam.camera_id = "test_cam"
        cam.name = "Test"
        cam.source = "test.mp4"
        cam.protocol = "file"
        cam.anomaly = MagicMock()
        cam.anomaly.model_type = "patchcore"
        cam.person_filter = MagicMock()
        cam.person_filter.model_name = "yolo11n.pt"

        with patch("argus.capture.manager.CameraManager.__init__", return_value=None):
            mgr = CameraManager.__new__(CameraManager)
            mgr._cameras = [cam]
            mgr._pipelines = {}
            mgr._threads = {}
            mgr._stop_event = threading.Event()
            mgr._lock = threading.Lock()
            mgr._alert_count = 0
            mgr._last_frame_counts = {}
            mgr._pending_frames = {}
            mgr._frames_dropped = {}
            mgr._max_pending = 30
            mgr._bp_lock = threading.Lock()
            mgr._last_bp_warn = {}
            mgr._shared_yolo = None
            mgr._shared_anomaly_detector = None
            mgr._shared_detector_lock = threading.Lock()
            mgr._correlator = None
            mgr._cross_camera_config = None
            mgr._on_alert = None
            mgr._on_status_change = None
            mgr._alert_config = AlertConfig()
            mgr._segmenter_config = None
            mgr._classifier_config = None
            return mgr

    def test_get_backpressure_stats_empty(self):
        mgr = self._make_manager()
        stats = mgr.get_backpressure_stats()
        assert "test_cam" in stats
        assert stats["test_cam"]["pending"] == 0
        assert stats["test_cam"]["dropped"] == 0
        assert stats["test_cam"]["backpressured"] is False

    def test_backpressure_detected(self):
        mgr = self._make_manager()
        mgr._pending_frames["test_cam"] = 30
        stats = mgr.get_backpressure_stats()
        assert stats["test_cam"]["backpressured"] is True

    def test_not_backpressured_below_threshold(self):
        mgr = self._make_manager()
        mgr._pending_frames["test_cam"] = 29
        stats = mgr.get_backpressure_stats()
        assert stats["test_cam"]["backpressured"] is False

    def test_dropped_counter(self):
        mgr = self._make_manager()
        mgr._frames_dropped["test_cam"] = 42
        stats = mgr.get_backpressure_stats()
        assert stats["test_cam"]["dropped"] == 42

    def test_thread_safety(self):
        """Concurrent access to backpressure counters should be safe."""
        mgr = self._make_manager()
        errors = []

        def increment():
            try:
                for _ in range(100):
                    with mgr._bp_lock:
                        mgr._pending_frames["test_cam"] = mgr._pending_frames.get("test_cam", 0) + 1
                    with mgr._bp_lock:
                        mgr._pending_frames["test_cam"] = max(0, mgr._pending_frames.get("test_cam", 0) - 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # After balanced increments/decrements, should be 0
        assert mgr._pending_frames["test_cam"] == 0
