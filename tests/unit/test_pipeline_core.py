"""Comprehensive unit tests for DetectionPipeline.process_frame() branches.

Covers: frame validation, MOG2 prefilter, person filter, anomaly detection,
learning mode suppression, mode management, and pipeline stats.
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from argus.anomaly.detector import AnomalyResult
from argus.capture.camera import FrameData
from argus.config.schema import (
    AlertConfig,
    AnomalyConfig,
    CameraConfig,
    MOG2Config,
    PersonFilterConfig,
    SeverityThresholds,
    SuppressionConfig,
    TemporalConfirmation,
)
from argus.core.pipeline import DetectionPipeline, PipelineMode
from argus.prefilter.mog2 import PreFilterResult
from argus.person.detector import ObjectDetectionResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_configs():
    """Create minimal camera + alert configs with simplex/drift/ring_buffer disabled."""
    from argus.config.schema import (
        DriftConfig,
        LowLightConfig,
        RingBufferConfig,
        SimplexConfig,
    )

    cam = CameraConfig(
        camera_id="test_cam",
        name="Test Camera",
        source="test.mp4",
        protocol="file",
        fps_target=5,
        resolution=(640, 480),
        mog2=MOG2Config(history=500, heartbeat_frames=150),
        person_filter=PersonFilterConfig(
            model_name="yolo11n.pt",
            skip_frame_on_person=True,
        ),
        anomaly=AnomalyConfig(threshold=0.7),
        simplex=SimplexConfig(enabled=False),
        drift=DriftConfig(enabled=False),
        ring_buffer=RingBufferConfig(enabled=False),
        low_light=LowLightConfig(enabled=False),
    )
    alert = AlertConfig(
        severity_thresholds=SeverityThresholds(info=0.3, low=0.5, medium=0.7, high=0.9),
        temporal_confirmation=TemporalConfirmation(),
        suppression=SuppressionConfig(),
    )
    return cam, alert


def _make_frame(height=480, width=640, channels=3):
    """Return a synthetic BGR frame filled with random pixels."""
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)


def _make_frame_data(frame=None, camera_id="test_cam", frame_number=1):
    """Build a FrameData with sensible defaults."""
    if frame is None:
        frame = _make_frame()
    return FrameData(
        frame=frame,
        camera_id=camera_id,
        timestamp=time.monotonic(),
        frame_number=frame_number,
        resolution=(640, 480),
    )


def _build_pipeline(cam_config=None, alert_config=None):
    """Construct a DetectionPipeline with all heavy components mocked out.

    Returns the pipeline instance. Internal sub-components (_prefilter,
    _object_detector, _anomaly_detector, _alert_grader, _camera, _zone_mask)
    are replaced with MagicMocks so no real models are loaded.
    """
    if cam_config is None or alert_config is None:
        cam_config, alert_config = _make_configs()

    with (
        patch("argus.core.pipeline.MOG2PreFilter"),
        patch("argus.core.pipeline.YOLOObjectDetector"),
        patch("argus.core.pipeline.AnomalibDetector"),
        patch("argus.core.pipeline.AlertGrader"),
        patch("argus.core.pipeline.CameraCapture"),
        patch("argus.core.pipeline.ZoneMaskEngine") as mock_zone,
        patch("argus.core.pipeline.METRICS"),
    ):
        # ZoneMaskEngine.apply should return the frame unchanged by default
        mock_zone_inst = MagicMock()
        mock_zone_inst.apply.side_effect = lambda f: f
        mock_zone_inst.get_include_zones.return_value = []
        mock_zone.return_value = mock_zone_inst

        pipeline = DetectionPipeline(
            camera_config=cam_config,
            alert_config=alert_config,
        )

    # Replace sub-components with fresh mocks for per-test control
    pipeline._prefilter = MagicMock()
    pipeline._object_detector = MagicMock()
    pipeline._anomaly_detector = MagicMock()
    pipeline._anomaly_detector.is_calibrated = False
    pipeline._alert_grader = MagicMock()
    pipeline._camera = MagicMock()
    pipeline._zone_mask = MagicMock()
    pipeline._zone_mask.apply.side_effect = lambda f: f
    pipeline._zone_mask.get_include_zones.return_value = []
    pipeline._simplex = None
    pipeline._event_bus = None
    pipeline._shadow_runner = None
    pipeline._classifier = None
    pipeline._segmenter = None

    # Ensure mode starts as ACTIVE (constructor sets ACTIVE before initialize)
    pipeline._mode = PipelineMode.ACTIVE
    pipeline._learning_start_time = None

    return pipeline


# ---------------------------------------------------------------------------
# TestFrameValidation
# ---------------------------------------------------------------------------

class TestFrameValidation:
    """process_frame should reject invalid frames early."""

    def test_none_frame_returns_none(self):
        pipeline = _build_pipeline()
        fd = _make_frame_data(frame=None)
        # FrameData stores None; pipeline should return None immediately
        result = pipeline.process_frame(fd)
        assert result is None

    def test_empty_frame_returns_none(self):
        pipeline = _build_pipeline()
        empty = np.array([], dtype=np.uint8)
        fd = _make_frame_data(frame=empty)
        result = pipeline.process_frame(fd)
        assert result is None

    def test_2d_frame_returns_none(self):
        """A 2D grayscale frame (missing channel dim) should be rejected."""
        pipeline = _build_pipeline()
        gray = np.zeros((480, 640), dtype=np.uint8)
        fd = _make_frame_data(frame=gray)
        result = pipeline.process_frame(fd)
        assert result is None

    def test_valid_frame_does_not_return_none_for_validation(self):
        """A valid 3-channel frame passes the validation gate."""
        pipeline = _build_pipeline()
        # Set up sub-components to allow the frame through but return no alert
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=False, change_ratio=0.0,
        )
        fd = _make_frame_data()
        # MOG2 no-change -> returns None, but that's the prefilter branch,
        # not the validation branch
        result = pipeline.process_frame(fd)
        assert result is None
        # Confirm prefilter was actually called (i.e. validation passed)
        assert pipeline._prefilter.process.called


# ---------------------------------------------------------------------------
# TestMOG2PreFilter
# ---------------------------------------------------------------------------

class TestMOG2PreFilter:
    """Tests for Stage 1 — MOG2 background subtraction gating."""

    def test_no_change_skips_processing(self):
        """When MOG2 reports no change the pipeline returns None early."""
        pipeline = _build_pipeline()
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=False, change_ratio=0.001,
        )
        # Ensure we don't hit the heartbeat bypass
        pipeline._last_heartbeat_time = time.monotonic()

        fd = _make_frame_data()
        result = pipeline.process_frame(fd)

        assert result is None
        assert pipeline.stats.frames_skipped_no_change == 1
        # Anomaly detector should NOT have been called
        pipeline._anomaly_detector.predict.assert_not_called()

    def test_change_detected_continues(self):
        """When MOG2 reports change, anomaly detection should run."""
        pipeline = _build_pipeline()
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.05,
        )
        pipeline._last_heartbeat_time = time.monotonic()

        anomaly_result = AnomalyResult(
            anomaly_score=0.3, anomaly_map=None,
            is_anomalous=False, threshold=0.7,
        )
        pipeline._anomaly_detector.predict.return_value = anomaly_result
        pipeline._object_detector.detect.return_value = ObjectDetectionResult()

        fd = _make_frame_data()
        result = pipeline.process_frame(fd)

        # Low score + not anomalous -> None, but anomaly detector was called
        assert result is None
        pipeline._anomaly_detector.predict.assert_called_once()

    def test_heartbeat_bypasses_prefilter(self):
        """When heartbeat interval has elapsed, MOG2 result is ignored."""
        pipeline = _build_pipeline()
        # Set last heartbeat far in the past
        pipeline._last_heartbeat_time = 0.0

        # MOG2 would say no change, but heartbeat should bypass it
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=False, change_ratio=0.0,
        )
        anomaly_result = AnomalyResult(
            anomaly_score=0.2, anomaly_map=None,
            is_anomalous=False, threshold=0.7,
        )
        pipeline._anomaly_detector.predict.return_value = anomaly_result
        pipeline._object_detector.detect.return_value = ObjectDetectionResult()

        fd = _make_frame_data()
        pipeline.process_frame(fd)

        # Anomaly detector should run despite no-change MOG2
        pipeline._anomaly_detector.predict.assert_called_once()
        assert pipeline.stats.frames_heartbeat == 1


# ---------------------------------------------------------------------------
# TestPersonFilter
# ---------------------------------------------------------------------------

class TestPersonFilter:
    """Tests for Stage 2 — YOLO person detection."""

    def test_person_detected_skips_frame(self):
        """With skip_frame_on_person=True and person detected, return None."""
        pipeline = _build_pipeline()
        # Bypass MOG2
        pipeline._last_heartbeat_time = 0.0
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.1,
        )

        # YOLO detects a person
        detection = ObjectDetectionResult(has_persons=True)
        pipeline._object_detector.detect.return_value = detection

        # Anomaly detector still called (parallel mode) but person short-circuits
        anomaly_result = AnomalyResult(
            anomaly_score=0.9, anomaly_map=None,
            is_anomalous=True, threshold=0.7,
        )
        pipeline._anomaly_detector.predict.return_value = anomaly_result

        fd = _make_frame_data()
        # Use the thread pool submit path — mock _INFERENCE_EXECUTOR
        with patch.object(pipeline, "_inference_executor") as mock_exec:
            from concurrent.futures import Future
            yolo_fut = Future()
            yolo_fut.set_result(detection)
            anomaly_fut = Future()
            anomaly_fut.set_result(anomaly_result)
            mock_exec.submit.side_effect = [yolo_fut, anomaly_fut]

            result = pipeline.process_frame(fd)

        assert result is None
        assert pipeline.stats.frames_skipped_person == 1

    def test_no_person_continues(self):
        """When no person is detected, anomaly evaluation proceeds."""
        pipeline = _build_pipeline()
        pipeline._last_heartbeat_time = 0.0
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.1,
        )

        detection = ObjectDetectionResult(has_persons=False)
        anomaly_result = AnomalyResult(
            anomaly_score=0.3, anomaly_map=None,
            is_anomalous=False, threshold=0.7,
        )

        with patch.object(pipeline, "_inference_executor") as mock_exec:
            from concurrent.futures import Future
            yolo_fut = Future()
            yolo_fut.set_result(detection)
            anomaly_fut = Future()
            anomaly_fut.set_result(anomaly_result)
            mock_exec.submit.side_effect = [yolo_fut, anomaly_fut]

            result = pipeline.process_frame(_make_frame_data())

        # Low score -> no alert, but pipeline didn't skip
        assert result is None
        assert pipeline.stats.frames_skipped_person == 0
        assert pipeline.stats.frames_analyzed == 1


# ---------------------------------------------------------------------------
# TestAnomalyDetection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    """Tests for Stage 3 — Anomalib anomaly scoring and alert generation."""

    def _run_with_anomaly(self, pipeline, anomaly_score, is_anomalous, alert_to_return=None):
        """Helper: run process_frame through the anomaly path.

        Patches the thread executor so YOLO and anomaly run inline.
        """
        pipeline._last_heartbeat_time = 0.0
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.1,
        )

        detection = ObjectDetectionResult(has_persons=False)
        anomaly_result = AnomalyResult(
            anomaly_score=anomaly_score,
            anomaly_map=None,
            is_anomalous=is_anomalous,
            threshold=0.7,
        )

        pipeline._alert_grader.evaluate.return_value = alert_to_return

        with (
            patch.object(pipeline, "_inference_executor") as mock_exec,
            patch("argus.core.pipeline.METRICS"),
        ):
            from concurrent.futures import Future
            yolo_fut = Future()
            yolo_fut.set_result(detection)
            anomaly_fut = Future()
            anomaly_fut.set_result(anomaly_result)
            mock_exec.submit.side_effect = [yolo_fut, anomaly_fut]

            return pipeline.process_frame(_make_frame_data())

    def test_low_score_no_alert(self):
        """Anomaly score below threshold -> is_anomalous=False -> None."""
        pipeline = _build_pipeline()
        result = self._run_with_anomaly(pipeline, anomaly_score=0.3, is_anomalous=False)
        assert result is None
        # AlertGrader.evaluate should NOT be called when not anomalous
        pipeline._alert_grader.evaluate.assert_not_called()

    def test_high_score_generates_alert(self):
        """Anomaly score above threshold -> grader returns Alert -> returns Alert."""
        from argus.alerts.grader import Alert, AlertSeverity

        mock_alert = MagicMock(spec=Alert)
        mock_alert.severity = AlertSeverity.HIGH
        mock_alert.anomaly_score = 0.95

        pipeline = _build_pipeline()
        result = self._run_with_anomaly(
            pipeline,
            anomaly_score=0.95,
            is_anomalous=True,
            alert_to_return=mock_alert,
        )

        assert result is mock_alert
        assert pipeline.stats.anomalies_detected == 1

    def test_anomalous_but_grader_suppresses(self):
        """Anomaly detected but grader suppresses (temporal / cool-down) -> None."""
        pipeline = _build_pipeline()
        result = self._run_with_anomaly(
            pipeline,
            anomaly_score=0.85,
            is_anomalous=True,
            alert_to_return=None,  # grader returns None (suppressed)
        )
        assert result is None
        assert pipeline.stats.anomalies_detected == 1


# ---------------------------------------------------------------------------
# TestLearningMode
# ---------------------------------------------------------------------------

class TestLearningMode:
    """Tests for LEARNING mode alert suppression (DET-006/DET-010)."""

    def test_learning_mode_suppresses_alerts(self):
        """In LEARNING mode, even high anomaly scores should not produce alerts."""
        pipeline = _build_pipeline()
        pipeline._mode = PipelineMode.LEARNING
        pipeline._learning_start_time = time.monotonic()
        pipeline._learning_duration = 9999  # won't expire during test

        pipeline._last_heartbeat_time = 0.0
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.1,
        )
        detection = ObjectDetectionResult(has_persons=False)
        anomaly_result = AnomalyResult(
            anomaly_score=0.99, anomaly_map=None,
            is_anomalous=True, threshold=0.7,
        )

        with patch.object(pipeline, "_inference_executor") as mock_exec:
            from concurrent.futures import Future
            yolo_fut = Future()
            yolo_fut.set_result(detection)
            anomaly_fut = Future()
            anomaly_fut.set_result(anomaly_result)
            mock_exec.submit.side_effect = [yolo_fut, anomaly_fut]

            result = pipeline.process_frame(_make_frame_data())

        assert result is None
        # AlertGrader.evaluate should NOT be called in learning mode
        pipeline._alert_grader.evaluate.assert_not_called()

    def test_learning_mode_auto_expires(self):
        """After learning duration expires, mode switches to ACTIVE."""
        pipeline = _build_pipeline()
        pipeline._mode = PipelineMode.LEARNING
        pipeline._learning_start_time = time.monotonic() - 1000
        pipeline._learning_duration = 1  # already expired

        pipeline._last_heartbeat_time = 0.0
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.1,
        )
        detection = ObjectDetectionResult(has_persons=False)
        anomaly_result = AnomalyResult(
            anomaly_score=0.2, anomaly_map=None,
            is_anomalous=False, threshold=0.7,
        )

        with patch.object(pipeline, "_inference_executor") as mock_exec:
            from concurrent.futures import Future
            yolo_fut = Future()
            yolo_fut.set_result(detection)
            anomaly_fut = Future()
            anomaly_fut.set_result(anomaly_result)
            mock_exec.submit.side_effect = [yolo_fut, anomaly_fut]

            pipeline.process_frame(_make_frame_data())

        assert pipeline.mode == PipelineMode.ACTIVE
        assert pipeline._auto_learning_complete is True


# ---------------------------------------------------------------------------
# TestModeManagement
# ---------------------------------------------------------------------------

class TestModeManagement:
    """Tests for set_mode / mode property."""

    def test_set_mode_active(self):
        pipeline = _build_pipeline()
        pipeline.set_mode(PipelineMode.ACTIVE)
        assert pipeline.mode == PipelineMode.ACTIVE

    def test_set_mode_maintenance(self):
        pipeline = _build_pipeline()
        pipeline.set_mode(PipelineMode.MAINTENANCE)
        assert pipeline.mode == PipelineMode.MAINTENANCE

    def test_set_mode_learning(self):
        pipeline = _build_pipeline()
        pipeline.set_mode(PipelineMode.LEARNING)
        assert pipeline.mode == PipelineMode.LEARNING

    def test_mode_round_trips(self):
        """Cycling through all modes works correctly."""
        pipeline = _build_pipeline()
        for mode in PipelineMode:
            pipeline.set_mode(mode)
            assert pipeline.mode == mode


# ---------------------------------------------------------------------------
# TestPipelineStats
# ---------------------------------------------------------------------------

class TestPipelineStats:
    """Tests for PipelineStats bookkeeping."""

    def test_frames_captured_incremented(self):
        """stats.frames_captured increases by 1 after each process_frame call."""
        pipeline = _build_pipeline()
        pipeline._last_heartbeat_time = time.monotonic()
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=False, change_ratio=0.0,
        )

        assert pipeline.stats.frames_captured == 0
        pipeline.process_frame(_make_frame_data())
        assert pipeline.stats.frames_captured == 1
        pipeline.process_frame(_make_frame_data(frame_number=2))
        assert pipeline.stats.frames_captured == 2

    def test_frames_analyzed_incremented_on_full_run(self):
        """frames_analyzed increases when frame goes through all stages."""
        pipeline = _build_pipeline()
        pipeline._last_heartbeat_time = 0.0
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.1,
        )
        detection = ObjectDetectionResult(has_persons=False)
        anomaly_result = AnomalyResult(
            anomaly_score=0.2, anomaly_map=None,
            is_anomalous=False, threshold=0.7,
        )

        with patch.object(pipeline, "_inference_executor") as mock_exec:
            from concurrent.futures import Future
            yolo_fut = Future()
            yolo_fut.set_result(detection)
            anomaly_fut = Future()
            anomaly_fut.set_result(anomaly_result)
            mock_exec.submit.side_effect = [yolo_fut, anomaly_fut]

            pipeline.process_frame(_make_frame_data())

        assert pipeline.stats.frames_analyzed == 1

    def test_snapshot_returns_all_counters(self):
        """PipelineStats.snapshot() includes all expected keys."""
        pipeline = _build_pipeline()
        snap = pipeline.stats.snapshot()
        expected_keys = {
            "captured", "skipped_no_change", "skipped_person",
            "analyzed", "heartbeats", "anomalies", "alerts", "avg_latency_ms",
            "current_fps",
        }
        assert set(snap.keys()) == expected_keys

    def test_stats_initial_values(self):
        """All counters start at zero."""
        pipeline = _build_pipeline()
        snap = pipeline.stats.snapshot()
        for key, val in snap.items():
            assert val == 0 or val == 0.0, f"{key} should be 0, got {val}"


# ---------------------------------------------------------------------------
# TestExceptionSafety
# ---------------------------------------------------------------------------

class TestStationarySuppressionIntegrationF5:
    """F5: when every active track is suppressed, grader is not invoked."""

    def _run_anomaly_frame(
        self,
        pipeline,
        *,
        anomaly_score: float,
        is_anomalous: bool,
        anomaly_map=None,
        alert_to_return=None,
    ):
        pipeline._last_heartbeat_time = 0.0
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.1,
        )

        detection = ObjectDetectionResult(has_persons=False)
        anomaly_result = AnomalyResult(
            anomaly_score=anomaly_score,
            anomaly_map=anomaly_map,
            is_anomalous=is_anomalous,
            threshold=0.7,
        )

        pipeline._alert_grader.evaluate.return_value = alert_to_return

        with (
            patch.object(pipeline, "_inference_executor") as mock_exec,
            patch("argus.core.pipeline.METRICS"),
        ):
            from concurrent.futures import Future
            yolo_fut = Future()
            yolo_fut.set_result(detection)
            anomaly_fut = Future()
            anomaly_fut.set_result(anomaly_result)
            mock_exec.submit.side_effect = [yolo_fut, anomaly_fut]

            return pipeline.process_frame(_make_frame_data())

    @staticmethod
    def _make_tracked(track_id: int, suppressed: bool):
        from argus.core.temporal_tracker import TrackedAnomaly
        return TrackedAnomaly(
            track_id=track_id,
            first_seen_frame=1,
            last_seen_frame=1,
            consecutive_frames=20,
            centroid_x=100.0,
            centroid_y=100.0,
            max_score=0.9,
            avg_score=0.85,
            velocity=(0.0, 0.0),
            is_stationary=True,
            persistence_seconds=4.0,
            area_px=500,
            suppressed=suppressed,
        )

    def test_all_suppressed_frame_skips_alert(self):
        """If every active track is suppressed, grader.evaluate must not be called."""
        from argus.core.temporal_tracker import TemporalAnalysis

        pipeline = _build_pipeline()

        # Wire fake physics plumbing so the pre-pass runs.
        pipeline._physics_postproc = MagicMock()
        pipeline._physics_postproc.extract_regions.return_value = [MagicMock()]

        suppressed_only = TemporalAnalysis(
            active_tracks=[self._make_tracked(1, suppressed=True)],
            new_tracks=0,
            lost_tracks=0,
            severity_boost=0.0,
        )
        pipeline._physics_tracker = MagicMock()
        pipeline._physics_tracker.update.return_value = suppressed_only

        result = self._run_anomaly_frame(
            pipeline,
            anomaly_score=0.95,
            is_anomalous=True,
            anomaly_map=np.zeros((32, 32), dtype=np.float32),
        )

        assert result is None
        pipeline._alert_grader.evaluate.assert_not_called()

    @staticmethod
    def _real_alert(severity: "AlertSeverity", score: float) -> "Alert":
        from argus.alerts.grader import Alert
        return Alert(
            alert_id="ALT-test",
            camera_id="test_cam",
            zone_id="default",
            severity=severity,
            anomaly_score=score,
            timestamp=time.time(),
            frame_number=1,
        )

    def test_new_track_in_suppressed_frame_still_fires(self):
        """Mixed suppressed + non-suppressed tracks still go through the grader."""
        from argus.alerts.grader import AlertSeverity
        from argus.core.temporal_tracker import TemporalAnalysis

        real_alert = self._real_alert(AlertSeverity.HIGH, 0.95)

        pipeline = _build_pipeline()
        pipeline._physics_postproc = MagicMock()
        pipeline._physics_postproc.extract_regions.return_value = [MagicMock(), MagicMock()]

        mixed = TemporalAnalysis(
            active_tracks=[
                self._make_tracked(1, suppressed=True),
                self._make_tracked(2, suppressed=False),
            ],
            new_tracks=1,
            lost_tracks=0,
            severity_boost=0.0,
        )
        pipeline._physics_tracker = MagicMock()
        pipeline._physics_tracker.update.return_value = mixed

        result = self._run_anomaly_frame(
            pipeline,
            anomaly_score=0.95,
            is_anomalous=True,
            anomaly_map=np.zeros((32, 32), dtype=np.float32),
            alert_to_return=real_alert,
        )

        # Grader ran because not all tracks were suppressed.
        pipeline._alert_grader.evaluate.assert_called()
        assert result is real_alert

    def test_no_active_tracks_does_not_skip(self):
        """An empty active_tracks list must not be mistaken for 'all suppressed'."""
        from argus.alerts.grader import AlertSeverity
        from argus.core.temporal_tracker import TemporalAnalysis

        real_alert = self._real_alert(AlertSeverity.LOW, 0.72)

        pipeline = _build_pipeline()
        pipeline._physics_postproc = MagicMock()
        pipeline._physics_postproc.extract_regions.return_value = []

        empty = TemporalAnalysis(
            active_tracks=[],
            new_tracks=0,
            lost_tracks=0,
            severity_boost=0.0,
        )
        pipeline._physics_tracker = MagicMock()
        pipeline._physics_tracker.update.return_value = empty

        result = self._run_anomaly_frame(
            pipeline,
            anomaly_score=0.72,
            is_anomalous=True,
            anomaly_map=np.zeros((32, 32), dtype=np.float32),
            alert_to_return=real_alert,
        )

        pipeline._alert_grader.evaluate.assert_called()
        assert result is real_alert


class TestExceptionSafety:
    """process_frame should catch exceptions and return None (CRIT-03)."""

    def test_anomaly_detector_exception_returns_none(self):
        """If anomaly detector throws, process_frame returns None."""
        pipeline = _build_pipeline()
        pipeline._last_heartbeat_time = 0.0
        pipeline._prefilter.process.return_value = PreFilterResult(
            has_change=True, change_ratio=0.1,
        )

        with patch.object(pipeline, "_inference_executor") as mock_exec:
            from concurrent.futures import Future
            yolo_fut = Future()
            yolo_fut.set_result(ObjectDetectionResult(has_persons=False))
            anomaly_fut = Future()
            anomaly_fut.set_exception(RuntimeError("model crashed"))
            mock_exec.submit.side_effect = [yolo_fut, anomaly_fut]

            result = pipeline.process_frame(_make_frame_data())

        assert result is None

    def test_prefilter_exception_returns_none(self):
        """If MOG2 prefilter throws, process_frame returns None."""
        pipeline = _build_pipeline()
        pipeline._last_heartbeat_time = time.monotonic()  # skip heartbeat
        pipeline._prefilter.process.side_effect = RuntimeError("MOG2 broken")

        result = pipeline.process_frame(_make_frame_data())
        assert result is None
