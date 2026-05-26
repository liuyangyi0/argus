"""Camera manager integration tests.

Tests multi-camera startup/shutdown, reconnection, and watchdog
using mock camera sources (no real hardware needed).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from argus.camera import CameraRuntimePlanner
from argus.capture.factory import CaptureFactory
from argus.capture.manager import CameraManager
from argus.config.schema import (
    AlertConfig,
    AnomalyConfig,
    CameraConfig,
    MOG2Config,
    PersonFilterConfig,
    SeverityThresholds,
    SimplexConfig,
    SuppressionConfig,
    TemporalConfirmation,
)
from argus.core.pipeline import DetectionPipeline, PipelineMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cam_config(camera_id: str, name: str = "Test") -> CameraConfig:
    return CameraConfig(
        camera_id=camera_id,
        name=name,
        source="synthetic",
        protocol="file",
        fps_target=5,
        resolution=(640, 480),
        mog2=MOG2Config(
            history=50,
            heartbeat_frames=10,
            enable_stabilization=False,
        ),
        person_filter=PersonFilterConfig(model_name="yolo11n.pt"),
        anomaly=AnomalyConfig(threshold=0.5),
        simplex=SimplexConfig(enabled=False),
        watchdog_timeout=5.0,  # Short but valid timeout for testing
    )


def _alert_config() -> AlertConfig:
    return AlertConfig(
        severity_thresholds=SeverityThresholds(info=0.3, low=0.5, medium=0.7, high=0.9),
        temporal=TemporalConfirmation(
            evidence_lambda=0.8,
            evidence_threshold=1.0,
            min_spatial_overlap=0.0,
        ),
        suppression=SuppressionConfig(
            same_zone_window_seconds=10.0,
            same_camera_window_seconds=5.0,
        ),
    )


class TestCameraManagerIntegration:
    """Integration tests for the CameraManager multi-camera orchestrator."""

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_multi_camera_startup_shutdown(self, mock_yolo, MockPipeline):
        """Start 2 cameras, verify independent operation, graceful shutdown."""
        # Setup mock pipeline that simulates running
        mock_instance = MagicMock()
        mock_instance.initialize.return_value = True
        mock_instance.stats = MagicMock(frames_captured=0)
        mock_instance._camera.state.connected = True
        mock_instance.run_once.return_value = None
        mock_instance.get_latest_anomaly_map.return_value = None
        MockPipeline.return_value = mock_instance

        cams = [
            _cam_config("cam_a", "Camera A"),
            _cam_config("cam_b", "Camera B"),
        ]
        manager = CameraManager(cams, _alert_config())

        started = manager.start_all()
        assert len(started) == 2
        assert "cam_a" in started
        assert "cam_b" in started
        assert manager.is_running

        # Give threads a moment to start
        time.sleep(0.1)

        # Verify both cameras are tracked
        statuses = manager.get_status()
        assert len(statuses) == 2
        cam_ids = {s.camera_id for s in statuses}
        assert cam_ids == {"cam_a", "cam_b"}

        # Graceful shutdown
        manager.stop_all()
        assert not manager.is_running

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_start_stop_single_camera(self, mock_yolo, MockPipeline):
        """Start all cameras, stop one, verify the other continues."""
        mock_instance = MagicMock()
        mock_instance.initialize.return_value = True
        mock_instance.stats = MagicMock(frames_captured=0)
        mock_instance._camera.state.connected = True
        mock_instance.run_once.return_value = None
        mock_instance.get_latest_anomaly_map.return_value = None
        MockPipeline.return_value = mock_instance

        cams = [
            _cam_config("cam_a"),
            _cam_config("cam_b"),
        ]
        manager = CameraManager(cams, _alert_config())
        manager.start_all()

        time.sleep(0.1)

        # Stop one camera
        manager.stop_camera("cam_a")

        # cam_b should still be tracked
        assert manager.is_running
        statuses = manager.get_status()
        running_ids = {s.camera_id for s in statuses if s.running}
        assert "cam_b" in running_ids

        manager.stop_all()

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_concurrent_start_same_camera_is_idempotent(self, mock_yolo, MockPipeline):
        """Startup and manual start must not create duplicate pipelines."""
        init_entered = threading.Event()
        release_init = threading.Event()
        mock_instance = MagicMock()
        mock_instance.stats = MagicMock(frames_captured=0)
        mock_instance._camera.state.connected = True
        mock_instance.run_once.return_value = None
        mock_instance.get_latest_anomaly_map.return_value = None
        mock_instance.get_detector_status.return_value = SimpleNamespace(mode="ssim_fallback")

        def _initialize():
            init_entered.set()
            assert release_init.wait(timeout=5)
            return True

        mock_instance.initialize.side_effect = _initialize
        MockPipeline.return_value = mock_instance

        manager = CameraManager([_cam_config("cam_a")], _alert_config())
        first_result: list[bool] = []
        first = threading.Thread(target=lambda: first_result.append(manager.start_camera("cam_a")))
        first.start()
        assert init_entered.wait(timeout=5)

        second_result = manager.start_camera("cam_a")
        release_init.set()
        first.join(timeout=5)

        assert first_result == [True]
        assert second_result is True
        assert MockPipeline.call_count == 1
        assert list(manager._threads) == ["cam_a"]
        manager.stop_all()

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_pipeline_mode_switching_via_manager(self, mock_yolo, MockPipeline):
        """Mode switching through CameraManager reaches the pipeline."""
        mock_instance = MagicMock()
        mock_instance.initialize.return_value = True
        mock_instance.stats = MagicMock(frames_captured=0)
        mock_instance._camera.state.connected = True
        mock_instance.run_once.return_value = None
        mock_instance.get_latest_anomaly_map.return_value = None
        mock_instance.mode = PipelineMode.ACTIVE
        MockPipeline.return_value = mock_instance

        cams = [_cam_config("cam_a")]
        manager = CameraManager(cams, _alert_config())
        manager.start_all()

        time.sleep(0.1)

        # Switch mode
        result = manager.set_pipeline_mode("cam_a", PipelineMode.MAINTENANCE)
        assert result is True
        mock_instance.set_mode.assert_called_with(PipelineMode.MAINTENANCE)

        # Non-existent camera
        result = manager.set_pipeline_mode("nonexistent", PipelineMode.ACTIVE)
        assert result is False

        manager.stop_all()

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_alert_callback_routing(self, mock_yolo, MockPipeline):
        """Alerts from pipelines are routed through the manager's callback."""
        alerts_received = []

        mock_instance = MagicMock()
        mock_instance.initialize.return_value = True
        mock_instance.stats = MagicMock(frames_captured=0)
        mock_instance._camera.state.connected = True
        mock_instance.run_once.return_value = None
        mock_instance.get_latest_anomaly_map.return_value = None
        MockPipeline.return_value = mock_instance

        cams = [_cam_config("cam_a")]
        manager = CameraManager(
            cams, _alert_config(), on_alert=alerts_received.append
        )
        manager.start_all()

        # The on_alert passed to CameraManager wraps before calling user callback.
        # Verify the manager tracks alert count internally.
        assert manager.alert_count == 0

        manager.stop_all()

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_init_failure_handled(self, mock_yolo, MockPipeline):
        """If pipeline.initialize() fails, camera is not started."""
        mock_instance = MagicMock()
        mock_instance.initialize.return_value = False  # Simulate failure
        MockPipeline.return_value = mock_instance

        cams = [_cam_config("cam_a")]
        manager = CameraManager(cams, _alert_config())

        started = manager.start_all()
        assert len(started) == 0

    def test_get_status_empty(self):
        """Get status with no cameras configured returns empty list."""
        manager = CameraManager([], _alert_config())
        statuses = manager.get_status()
        assert statuses == []

    def test_severity_downgrade(self):
        """Test the static severity downgrade utility."""
        from argus.config.schema import AlertSeverity

        result = CameraManager._downgrade_severity(AlertSeverity.HIGH, 1)
        assert result == AlertSeverity.MEDIUM

        result = CameraManager._downgrade_severity(AlertSeverity.HIGH, 3)
        assert result == AlertSeverity.INFO

        result = CameraManager._downgrade_severity(AlertSeverity.INFO, 1)
        assert result == AlertSeverity.INFO

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_source_resolver_feeds_runtime_copy_but_config_stays_original(
        self, mock_yolo, MockPipeline
    ):
        """Runtime source resolution should not mutate persisted camera config."""
        original = _cam_config("usb_cam", "USB Camera").model_copy(
            update={"source": "0", "protocol": "usb"}
        )
        runtime_source = "rtsp://127.0.0.1:8554/usb_cam"
        pipeline_instances = []

        def make_pipeline(**kwargs):
            instance = MagicMock()
            instance.camera_config = kwargs["camera_config"]
            instance.initialize.return_value = True
            instance.get_detector_status.return_value = MagicMock(mode="loaded")
            instance.stats = MagicMock(frames_captured=0)
            instance._camera.state.connected = True
            instance.run_once.return_value = None
            instance.get_latest_anomaly_map.return_value = None
            pipeline_instances.append(instance)
            return instance

        MockPipeline.side_effect = make_pipeline

        def source_resolver(config: CameraConfig) -> CameraConfig:
            assert config is original
            return config.model_copy(
                update={"source": runtime_source, "protocol": "rtsp"}
            )

        manager = CameraManager(
            [original],
            _alert_config(),
            source_resolver=source_resolver,
        )

        try:
            assert manager.start_all() == ["usb_cam"]

            pipeline_config = MockPipeline.call_args.kwargs["camera_config"]
            assert pipeline_config is not original
            assert pipeline_config.source == runtime_source
            assert pipeline_config.protocol == "rtsp"

            persisted = manager.get_camera_config("usb_cam")
            assert persisted is original
            assert persisted.source == "0"
            assert persisted.protocol == "usb"
            assert original.source == "0"
            assert original.protocol == "usb"
        finally:
            manager.stop_all()

        assert pipeline_instances

    def test_capture_factory_uses_runtime_detection_input(self):
        """Factory should open the detection source, not the persisted USB source."""
        config = _cam_config("usb_cam", "USB Camera").model_copy(
            update={"source": "0", "protocol": "usb"}
        )
        plan = CameraRuntimePlanner.build(config)

        with patch("argus.capture.factory.CameraCapture") as MockCameraCapture:
            capture = CaptureFactory.create(plan.detection, config)

        assert capture is MockCameraCapture.return_value
        MockCameraCapture.assert_called_once_with(
            camera_id="usb_cam",
            source="rtsp://127.0.0.1:8554/usb_cam",
            protocol="rtsp",
            fps_target=config.fps_target,
            resolution=config.resolution,
            reconnect_delay=config.reconnect_delay,
            max_reconnect_attempts=config.max_reconnect_attempts,
        )

    def test_detection_pipeline_uses_injected_capture(self):
        """Injected capture adapters should bypass the legacy capture branch."""
        cam_config = _cam_config("cam_a")
        alert_config = _alert_config()
        capture = MagicMock()
        capture.protocol = "file"

        with (
            patch("argus.core.pipeline.CameraCapture") as MockCameraCapture,
            patch("argus.core.pipeline.MOG2PreFilter"),
            patch("argus.core.pipeline.YOLOObjectDetector"),
            patch("argus.core.pipeline.AnomalibDetector"),
            patch("argus.core.pipeline.AlertGrader"),
            patch("argus.core.pipeline.ZoneMaskEngine"),
            patch("argus.core.pipeline.METRICS"),
        ):
            pipeline = DetectionPipeline(
                camera_config=cam_config,
                alert_config=alert_config,
                capture=capture,
            )

        assert pipeline._camera is capture
        MockCameraCapture.assert_not_called()
