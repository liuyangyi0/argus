"""Tests for baseline capture job (Phase 4)."""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.capture.baseline_job import (
    BaselineCaptureJobConfig,
    run_baseline_capture_job,
    _write_manifest,
    _write_stats,
    _distribution,
)
from argus.capture.quality import CaptureStats
from argus.config.schema import CaptureQualityConfig


@pytest.fixture
def tmp_baselines(tmp_path):
    return tmp_path / "baselines"


@pytest.fixture
def mock_camera_manager():
    """Mock CameraManager that returns random frames."""
    mgr = MagicMock()
    mgr.get_raw_frame.return_value = np.random.randint(
        0, 255, (480, 640, 3), dtype=np.uint8
    )
    mgr.is_anomaly_locked.return_value = False
    return mgr


@pytest.fixture
def basic_job_config(tmp_baselines):
    return BaselineCaptureJobConfig(
        camera_id="test_cam",
        target_frames=5,
        duration_hours=1.0,
        sampling_strategy="uniform",
        storage_path=str(tmp_baselines),
        quality_config=CaptureQualityConfig(enabled=False),  # disable filtering for basic tests
        pause_on_anomaly_lock=False,
        post_capture_review=False,
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Patch time.sleep in baseline_job to avoid real waits in tests."""
    monkeypatch.setattr("argus.capture.baseline_job.time.sleep", lambda _: None)


class TestBaselineCaptureJob:
    def test_uniform_capture_completes(self, basic_job_config, mock_camera_manager):
        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()
        progress_calls = []

        def progress_cb(pct, msg):
            progress_calls.append((pct, msg))

        result = run_baseline_capture_job(
            progress_cb,
            job_config=basic_job_config,
            camera_manager=mock_camera_manager,
            pause_event=pause_ev,
            abort_event=abort_ev,
        )

        assert result["collected_frames"] == 5
        assert result["camera_id"] == "test_cam"
        assert result["strategy"] == "uniform"

        # Check output files exist
        output_dir = Path(result["output_dir"])
        assert output_dir.exists()
        assert (output_dir / "manifest.sha256").exists()
        assert (output_dir / "stats.json").exists()
        assert (output_dir / "capture_job.json").exists()

        # Check PNG files
        pngs = list(output_dir.glob("*.png"))
        assert len(pngs) == 5

        # Check sidecar JSONs
        sidecars = list(output_dir.glob("baseline_*.json"))
        assert len(sidecars) == 5

    def test_abort_stops_capture(self, basic_job_config, mock_camera_manager):
        basic_job_config.target_frames = 1000  # large target
        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        # Set abort immediately — the loop will detect it on first iteration
        abort_ev.set()

        from argus.dashboard.tasks import TaskAbortedError
        with pytest.raises(TaskAbortedError):
            run_baseline_capture_job(
                lambda p, m: None,
                job_config=basic_job_config,
                camera_manager=mock_camera_manager,
                pause_event=pause_ev,
                abort_event=abort_ev,
            )

    def test_null_frames_handled(self, basic_job_config, mock_camera_manager):
        """Camera returning None should not crash the job."""
        # Return None for first 3 calls, then valid frames
        call_count = [0]
        def get_frame(camera_id):
            call_count[0] += 1
            if call_count[0] <= 3:
                return None
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        mock_camera_manager.get_raw_frame.side_effect = get_frame

        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        result = run_baseline_capture_job(
            lambda p, m: None,
            job_config=basic_job_config,
            camera_manager=mock_camera_manager,
            pause_event=pause_ev,
            abort_event=abort_ev,
        )

        assert result["collected_frames"] == 5


class TestHelpers:
    def test_write_manifest(self, tmp_path):
        # Create some test PNGs
        for i in range(3):
            (tmp_path / f"baseline_{i:05d}.png").write_bytes(b"fake png data " + str(i).encode())

        path = _write_manifest(tmp_path)
        assert path.exists()
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            assert "  " in line  # sha256  filename format

    def test_distribution_empty(self):
        result = _distribution([])
        assert result["min"] == 0
        assert result["max"] == 0

    def test_distribution_values(self):
        result = _distribution([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert abs(result["mean"] - 3.0) < 0.01
