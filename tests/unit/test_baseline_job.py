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
    _create_sampler,
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
        session_label="night",
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
    def test_capture_runtime_error_cleans_empty_partial_version(
        self,
        basic_job_config,
        mock_camera_manager,
        monkeypatch,
    ):
        class BrokenSampler:
            def should_accept(self, frame):
                raise RuntimeError("boom")

            def get_sleep_interval(self):
                return 0.0

            def on_frame_saved(self, frame, index):
                return None

        monkeypatch.setattr(
            "argus.capture.baseline_job._create_sampler",
            lambda config: type("Selection", (), {
                "sampler": BrokenSampler(),
                "effective_strategy": "active",
                "warning": None,
            })(),
        )

        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        with pytest.raises(RuntimeError, match="boom"):
            run_baseline_capture_job(
                lambda p, m: None,
                job_config=basic_job_config,
                camera_manager=mock_camera_manager,
                pause_event=pause_ev,
                abort_event=abort_ev,
            )

        base_dir = Path(basic_job_config.storage_path) / basic_job_config.camera_id / "default"
        assert list(base_dir.glob("v*")) == []

    def test_active_sampler_respects_duration_based_interval(self, basic_job_config):
        basic_job_config.sampling_strategy = "active"
        basic_job_config.target_frames = 10
        basic_job_config.duration_hours = 1.0
        basic_job_config.active_sleep_min_seconds = 0.5

        selection = _create_sampler(basic_job_config)

        assert selection.effective_strategy == "active"
        assert selection.sampler.get_sleep_interval() == pytest.approx(360.0)

    def test_active_capture_falls_back_when_faiss_missing(
        self,
        basic_job_config,
        mock_camera_manager,
        monkeypatch,
    ):
        basic_job_config.sampling_strategy = "active"
        progress_calls = []
        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        monkeypatch.setattr(
            "argus.capture.baseline_job.get_active_sampler_unavailable_reason",
            lambda: "缺少依赖: faiss",
        )

        result = run_baseline_capture_job(
            lambda pct, msg: progress_calls.append((pct, msg)),
            job_config=basic_job_config,
            camera_manager=mock_camera_manager,
            pause_event=pause_ev,
            abort_event=abort_ev,
        )

        assert result["collected_frames"] == 5
        assert result["requested_strategy"] == "active"
        assert result["strategy"] == "uniform"
        assert "faiss" in result["strategy_warning"]
        assert any("已自动降级为均匀采集" in msg for _, msg in progress_calls)

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
        assert (output_dir / "capture_meta.json").exists()

        meta = json.loads((output_dir / "capture_meta.json").read_text())
        assert meta["session_label"] == "night"
        assert meta["camera_id"] == "test_cam"

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

    def test_capture_fails_when_too_few_frames_collected(self, basic_job_config, mock_camera_manager):
        """A capture with too few saved frames should fail instead of creating an empty current version."""
        basic_job_config.target_frames = 10
        basic_job_config.duration_hours = 0.00001
        mock_camera_manager.get_raw_frame.return_value = None

        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        with pytest.raises(RuntimeError, match="仅采集到 0 帧"):
            run_baseline_capture_job(
                lambda p, m: None,
                job_config=basic_job_config,
                camera_manager=mock_camera_manager,
                pause_event=pause_ev,
                abort_event=abort_ev,
            )

        base_dir = Path(basic_job_config.storage_path) / basic_job_config.camera_id / "default"
        assert not (base_dir / "current.txt").exists()
        assert list(base_dir.glob("v*")) == []

    def test_capture_skips_person_detector_in_live_capture_mode(self, basic_job_config, mock_camera_manager, monkeypatch):
        """Baseline capture should not invoke the expensive person detector path."""
        checks = {"person": 0}

        def fail_if_called(self, frame):
            checks["person"] += 1
            raise AssertionError("person detector should be disabled for baseline capture")

        monkeypatch.setattr("argus.capture.quality.FrameQualityFilter._check_person", fail_if_called)

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
        assert checks["person"] == 0

    def test_uniform_capture_disables_duplicate_filter(self, basic_job_config, mock_camera_manager, monkeypatch):
        """Uniform quick capture should not reject a static scene as duplicates forever."""
        checks = {"duplicate": 0}

        def fail_if_called(self, gray, prev_frame):
            checks["duplicate"] += 1
            raise AssertionError("duplicate filter should be disabled for uniform capture")

        monkeypatch.setattr("argus.capture.quality.FrameQualityFilter._compute_ssim", fail_if_called)

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
        assert checks["duplicate"] == 0

    def test_capture_logs_start_and_completion(self, basic_job_config, mock_camera_manager, monkeypatch):
        """Baseline capture should emit operator-visible lifecycle logs."""
        events = []

        def fake_info(event, **kwargs):
            events.append((event, kwargs))

        monkeypatch.setattr("argus.capture.baseline_job.logger.info", fake_info)

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
        names = [event for event, _ in events]
        assert "baseline.capture_started" in names
        assert "baseline.capture_progress" in names
        assert "baseline.capture_completed" in names

    def test_capture_logs_no_frame_stalls(self, basic_job_config, mock_camera_manager, monkeypatch):
        """Repeated missing frames should emit warning logs for live debugging."""
        warnings = []

        def fake_warning(event, **kwargs):
            warnings.append((event, kwargs))

        call_count = [0]

        def get_frame(camera_id):
            call_count[0] += 1
            if call_count[0] <= 2:
                return None
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        mock_camera_manager.get_raw_frame.side_effect = get_frame
        monkeypatch.setattr("argus.capture.baseline_job.logger.warning", fake_warning)

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
        assert any(event == "baseline.capture_no_frame" for event, _ in warnings)


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
