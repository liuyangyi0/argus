"""Tests for F4: force-corroboration zone mode in CameraManager.

When a zone has ``require_corroboration=True`` and the cross-camera correlator
reports the alert as uncorroborated, the manager MUST drop the alert entirely
(no dispatch) rather than downgrading severity.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.alerts.grader import Alert
from argus.capture.manager import CameraManager
from argus.config.schema import (
    AlertConfig,
    AlertSeverity,
    AnomalyConfig,
    CameraConfig,
    CameraOverlapConfig,
    CrossCameraConfig,
    MOG2Config,
    PersonFilterConfig,
    SeverityThresholds,
    SimplexConfig,
    SuppressionConfig,
    TemporalConfirmation,
    ZoneConfig,
    ZonePriority,
)


def _cam_config(
    camera_id: str,
    zones: list[ZoneConfig],
) -> CameraConfig:
    """Build a minimal camera config with the given zones."""
    return CameraConfig(
        camera_id=camera_id,
        name=camera_id,
        source="synthetic",
        protocol="file",
        fps_target=5,
        resolution=(640, 480),
        zones=zones,
        mog2=MOG2Config(
            history=50,
            heartbeat_frames=10,
            enable_stabilization=False,
        ),
        person_filter=PersonFilterConfig(model_name="yolo11n.pt"),
        anomaly=AnomalyConfig(threshold=0.5),
        simplex=SimplexConfig(enabled=False),
        watchdog_timeout=5.0,
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


def _cross_config() -> CrossCameraConfig:
    return CrossCameraConfig(
        enabled=True,
        overlap_pairs=[
            CameraOverlapConfig(
                camera_a="cam_a",
                camera_b="cam_b",
                homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ),
        ],
        corroboration_threshold=0.3,
        max_age_seconds=5.0,
        uncorroborated_severity_downgrade=1,
    )


def _make_alert(zone_id: str, severity: AlertSeverity = AlertSeverity.HIGH) -> Alert:
    return Alert(
        alert_id=f"ALT-test-{zone_id}",
        camera_id="cam_a",
        zone_id=zone_id,
        severity=severity,
        anomaly_score=0.95,
        timestamp=0.0,
        frame_number=1,
    )


def _captured_alert_handler(mock_pipeline_cls):
    """Extract the ``on_alert`` callback captured by the patched DetectionPipeline."""
    assert mock_pipeline_cls.called
    _args, kwargs = mock_pipeline_cls.call_args
    handler = kwargs.get("on_alert")
    assert handler is not None, "DetectionPipeline was not constructed with on_alert"
    return handler


class TestForceCorroboration:
    """F4: uncorroborated alerts from zones with require_corroboration=True are dropped."""

    def _patched_manager(
        self,
        MockPipeline,
        zones: list[ZoneConfig],
        correlator_result_corroborated: bool,
    ) -> tuple[CameraManager, list[Alert]]:
        """Build a manager whose correlator deterministically returns the given result."""
        mock_instance = MagicMock()
        mock_instance.initialize.return_value = True
        mock_instance.stats = MagicMock(frames_captured=0)
        mock_instance._camera.state.connected = True
        mock_instance.run_once.return_value = None
        mock_instance.get_latest_anomaly_map.return_value = None
        MockPipeline.return_value = mock_instance

        alerts_received: list[Alert] = []
        manager = CameraManager(
            [_cam_config("cam_a", zones), _cam_config("cam_b", [])],
            _alert_config(),
            on_alert=alerts_received.append,
            cross_camera_config=_cross_config(),
        )

        # Replace the correlator with a deterministic mock.
        correlator_mock = MagicMock()
        correlator_mock.check.return_value = MagicMock(
            corroborated=correlator_result_corroborated,
            partner_camera="cam_b",
            partner_score_at_location=0.2 if not correlator_result_corroborated else 0.8,
        )
        manager._correlator = correlator_mock

        return manager, alerts_received

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_require_corroboration_suppresses_uncorroborated_alert(
        self, _yolo, MockPipeline,
    ):
        """Zone with require_corroboration=True → uncorroborated alert dropped entirely."""
        strict_zone = ZoneConfig(
            zone_id="critical_zone",
            name="Critical",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            priority=ZonePriority.CRITICAL,
            require_corroboration=True,
        )
        manager, alerts_received = self._patched_manager(
            MockPipeline, [strict_zone], correlator_result_corroborated=False,
        )

        # Start the camera so _alert_handler is constructed and passed to pipeline.
        manager.start_all()
        try:
            handler = _captured_alert_handler(MockPipeline)
            alert = _make_alert("critical_zone", severity=AlertSeverity.HIGH)

            handler(alert)

            # Alert should have been dropped — dispatcher never called.
            assert alerts_received == []
            # Internal counter also should not advance for suppressed alerts.
            assert manager.alert_count == 0
        finally:
            manager.stop_all()

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_require_corroboration_off_falls_back_to_severity_downgrade(
        self, _yolo, MockPipeline,
    ):
        """Zone with require_corroboration=False → existing downgrade behavior preserved."""
        relaxed_zone = ZoneConfig(
            zone_id="relaxed_zone",
            name="Relaxed",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            priority=ZonePriority.STANDARD,
            require_corroboration=False,
        )
        manager, alerts_received = self._patched_manager(
            MockPipeline, [relaxed_zone], correlator_result_corroborated=False,
        )
        manager.start_all()
        try:
            handler = _captured_alert_handler(MockPipeline)
            alert = _make_alert("relaxed_zone", severity=AlertSeverity.HIGH)

            handler(alert)

            # Alert still dispatched — with severity downgraded by 1 level.
            assert len(alerts_received) == 1
            dispatched = alerts_received[0]
            assert dispatched.corroborated is False
            # HIGH (3) -> MEDIUM (2) after downgrade of 1 level.
            assert dispatched.severity == AlertSeverity.MEDIUM
        finally:
            manager.stop_all()

    @patch("argus.capture.manager.DetectionPipeline")
    @patch("argus.person.detector.get_shared_yolo", return_value=MagicMock())
    def test_require_corroboration_true_but_corroborated_passes_through(
        self, _yolo, MockPipeline,
    ):
        """Zone with require_corroboration=True but alert IS corroborated → dispatch normally."""
        strict_zone = ZoneConfig(
            zone_id="critical_zone",
            name="Critical",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            priority=ZonePriority.CRITICAL,
            require_corroboration=True,
        )
        manager, alerts_received = self._patched_manager(
            MockPipeline, [strict_zone], correlator_result_corroborated=True,
        )
        manager.start_all()
        try:
            handler = _captured_alert_handler(MockPipeline)
            alert = _make_alert("critical_zone", severity=AlertSeverity.HIGH)

            handler(alert)

            assert len(alerts_received) == 1
            assert alerts_received[0].corroborated is True
            # No downgrade because corroborated.
            assert alerts_received[0].severity == AlertSeverity.HIGH
        finally:
            manager.stop_all()

    def test_zone_requires_corroboration_helper_handles_missing_zone(self):
        """The helper safely returns False for unknown zone IDs (default zone, etc)."""
        relaxed_zone = ZoneConfig(
            zone_id="z1",
            name="Z1",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            priority=ZonePriority.STANDARD,
            require_corroboration=False,
        )
        manager = CameraManager(
            [_cam_config("cam_a", [relaxed_zone])],
            _alert_config(),
        )
        # Unknown camera
        assert manager._zone_requires_corroboration("cam_unknown", "z1") is False
        # Unknown zone
        assert manager._zone_requires_corroboration("cam_a", "nonexistent") is False
        # Existing relaxed zone
        assert manager._zone_requires_corroboration("cam_a", "z1") is False

    def test_zone_requires_corroboration_helper_detects_strict_zone(self):
        """The helper returns True for zones where require_corroboration is set."""
        strict_zone = ZoneConfig(
            zone_id="critical",
            name="Critical",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            priority=ZonePriority.CRITICAL,
            require_corroboration=True,
        )
        manager = CameraManager(
            [_cam_config("cam_a", [strict_zone])],
            _alert_config(),
        )
        assert manager._zone_requires_corroboration("cam_a", "critical") is True
