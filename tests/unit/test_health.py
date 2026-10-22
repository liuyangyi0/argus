"""Tests for the health monitoring module."""

import pytest

from argus.core.health import HealthMonitor, HealthStatus, runtime_detection_health


class TestHealthMonitor:
    def test_initial_status_is_unhealthy(self):
        """No cameras = unhealthy."""
        monitor = HealthMonitor()
        health = monitor.get_health()
        assert health.status == HealthStatus.UNHEALTHY
        assert len(health.cameras) == 0

    def test_all_connected_is_healthy(self):
        """All cameras connected = healthy."""
        monitor = HealthMonitor()
        monitor.update_camera("cam_01", connected=True, frames_captured=100)
        monitor.update_camera("cam_02", connected=True, frames_captured=50)

        health = monitor.get_health()
        assert health.status == HealthStatus.HEALTHY
        assert len(health.cameras) == 2

    def test_partial_connection_is_degraded(self):
        """Some cameras disconnected = degraded."""
        monitor = HealthMonitor()
        monitor.update_camera("cam_01", connected=True)
        monitor.update_camera("cam_02", connected=False, error="Connection lost")

        health = monitor.get_health()
        assert health.status == HealthStatus.DEGRADED

    def test_all_disconnected_is_unhealthy(self):
        """All cameras disconnected = unhealthy."""
        monitor = HealthMonitor()
        monitor.update_camera("cam_01", connected=False)
        monitor.update_camera("cam_02", connected=False)

        health = monitor.get_health()
        assert health.status == HealthStatus.UNHEALTHY

    def test_alert_counter(self):
        """Should track total alerts."""
        monitor = HealthMonitor()
        monitor.record_alert()
        monitor.record_alert()
        monitor.record_alert()

        health = monitor.get_health()
        assert health.total_alerts == 3

    def test_uptime_increases(self):
        """Uptime should be positive."""
        monitor = HealthMonitor()
        health = monitor.get_health()
        assert health.uptime_seconds >= 0

    def test_system_info(self):
        """Should include platform and Python version."""
        monitor = HealthMonitor()
        health = monitor.get_health()
        assert health.platform != ""
        assert health.python_version != ""

    def test_get_camera_health_returns_single_camera_snapshot(self):
        """Should expose a single camera snapshot for dashboard consumers."""
        monitor = HealthMonitor()
        monitor.update_camera("cam_01", connected=False, frames_captured=12, error="offline")

        camera_health = monitor.get_camera_health("cam_01")

        assert camera_health is not None
        assert camera_health["camera_id"] == "cam_01"
        assert camera_health["connected"] is False
        assert camera_health["frames_captured"] == 12
        assert camera_health["error"] == "offline"

    def test_repeated_poll_does_not_refresh_last_frame_or_hide_stall(self, monkeypatch):
        clock = [10.0]
        monkeypatch.setattr("argus.core.health.time.monotonic", lambda: clock[0])
        monitor = HealthMonitor()
        monitor.update_camera("cam", connected=True, frames_captured=12)

        clock[0] = 41.0
        monitor.update_camera("cam", connected=True, frames_captured=12)

        snapshot = monitor.get_camera_health("cam")
        assert snapshot["last_frame_time"] == 10.0
        assert snapshot["capture_state"] == "stale"
        assert snapshot["detection_reason"] == "capture_stale"
        assert monitor.check_stale_cameras() == ["cam"]
        assert monitor.get_health().status == HealthStatus.UNHEALTHY

    def test_connected_camera_without_first_frame_becomes_stale(self, monkeypatch):
        clock = [10.0]
        monkeypatch.setattr("argus.core.health.time.monotonic", lambda: clock[0])
        monitor = HealthMonitor()
        monitor.update_camera("cam", connected=True, frames_captured=0)
        assert monitor.get_camera_health("cam")["capture_state"] == "waiting"
        clock[0] = 41.0
        monitor.update_camera("cam", connected=True, frames_captured=0)
        assert monitor.check_stale_cameras() == ["cam"]

    def test_counter_reset_requires_a_new_frame_before_freshness_is_renewed(self, monkeypatch):
        clock = [10.0]
        monkeypatch.setattr("argus.core.health.time.monotonic", lambda: clock[0])
        monitor = HealthMonitor()
        monitor.update_camera("cam", connected=True, frames_captured=100)
        clock[0] = 41.0
        monitor.update_camera("cam", connected=True, frames_captured=0)
        assert monitor.get_camera_health("cam")["last_frame_time"] == 10.0
        assert monitor.check_stale_cameras() == ["cam"]
        clock[0] = 42.0
        monitor.update_camera("cam", connected=True, frames_captured=1)
        assert monitor.get_camera_health("cam")["capture_state"] == "receiving"

    def test_scene_blocking_and_model_failure_do_not_imply_process_failure(self):
        monitor = HealthMonitor()
        monitor.update_camera(
            "cam", connected=True, frames_captured=10,
            scene_guard_usable=False, scene_guard_state="moved",
            scene_guard_reason="camera_moved", model_ready=False, model_error="CUDA OOM",
        )
        assert monitor.get_health().status == HealthStatus.HEALTHY
        camera = monitor.get_detection_readiness()["cameras"][0]
        assert camera["capture_state"] == "receiving"
        assert camera["detection_state"] == "blocked"
        assert camera["scene_guard_state"] == "moved"
        assert camera["model_ready"] is False
        assert camera["model_error"] == "CUDA OOM"
        assert monitor.get_detection_readiness()["ready"] is False

    def test_reconnect_waits_for_a_new_frame_instead_of_reusing_old_counter(self, monkeypatch):
        clock = [10.0]
        monkeypatch.setattr("argus.core.health.time.monotonic", lambda: clock[0])
        monitor = HealthMonitor()
        monitor.update_camera("cam", connected=True, frames_captured=100)
        clock[0] = 11.0
        monitor.update_camera("cam", connected=False, frames_captured=100)
        clock[0] = 12.0
        monitor.update_camera("cam", connected=True, frames_captured=100)
        assert monitor.get_camera_health("cam")["capture_state"] == "waiting"
        clock[0] = 43.0
        monitor.update_camera("cam", connected=True, frames_captured=100)
        assert monitor.get_camera_health("cam")["capture_state"] == "stale"

    def test_readiness_change_broadcasts_without_connection_change(self):
        changes = []
        monitor = HealthMonitor(on_change=lambda topic, payload: changes.append(payload))
        monitor.update_camera(
            "cam", connected=True, frames_captured=10,
            scene_guard_usable=False, model_ready=True,
        )
        monitor.update_camera(
            "cam", connected=True, frames_captured=11,
            scene_guard_usable=True, model_ready=True,
        )
        assert len(changes) == 2
        assert changes[-1]["cameras"][0]["detection_state"] == "ready"

    @pytest.mark.parametrize("mode", ["collection", "training", "learning"])
    def test_non_operational_modes_never_report_ready(self, mode):
        monitor = HealthMonitor()
        monitor.update_camera(
            "cam", connected=True, frames_captured=10,
            scene_guard_usable=True, model_ready=True, pipeline_mode=mode,
        )
        readiness = monitor.get_detection_readiness()
        assert readiness["ready"] is False
        camera = readiness["cameras"][0]
        assert camera["capture_state"] == "receiving"
        assert camera["detection_state"] == "blocked"
        assert camera["pipeline_mode"] == mode
        assert camera["detection_reason"] == (
            "learning_alerts_suppressed" if mode == "learning" else f"pipeline_mode_{mode}"
        )

    @pytest.mark.parametrize("mode", ["active", "maintenance"])
    def test_operational_modes_can_report_ready(self, mode):
        monitor = HealthMonitor()
        monitor.update_camera(
            "cam", connected=True, frames_captured=10,
            scene_guard_usable=True, model_ready=True, pipeline_mode=mode,
        )
        assert monitor.get_detection_readiness()["ready"] is True

    def test_unknown_mode_is_not_assumed_active(self):
        monitor = HealthMonitor()
        monitor.update_camera(
            "cam", connected=True, frames_captured=10,
            **runtime_detection_health({"usable": True}, [
                {"name": "anomaly", "loaded": True, "backend": "torch-cuda"},
            ]),
        )
        camera = monitor.get_camera_health("cam")
        assert camera["detection_state"] == "waiting"
        assert camera["detection_reason"] == "runtime_mode_unavailable"

    def test_removed_camera_is_forgotten_but_configured_offline_camera_is_kept(self):
        changes = []
        monitor = HealthMonitor(on_change=lambda topic, data: changes.append(data))
        monitor.update_camera("removed", connected=False)
        monitor.update_camera("offline", connected=False)
        monitor.update_camera(
            "online", connected=True, frames_captured=10,
            scene_guard_usable=True, model_ready=True,
        )
        monitor.retain_cameras(["offline", "online"])
        assert monitor.get_camera_health("removed") is None
        assert monitor.get_camera_health("offline") is not None
        assert monitor.get_detection_readiness()["ready"] is False
        assert {c["camera_id"] for c in changes[-1]["cameras"]} == {"offline", "online"}
        monitor.retain_cameras(["online"])
        assert monitor.get_detection_readiness()["ready"] is True
        monitor.retain_cameras([])
        assert monitor.get_detection_readiness()["reasons"] == ["no_cameras"]


def test_runtime_detection_health_rejects_failed_or_fallback_models():
    summary = runtime_detection_health(
        {"usable": True, "state": "stable"},
        [{"name": "anomaly", "loaded": True, "backend": "ssim-fallback"}],
    )
    assert summary["scene_guard_usable"] is True
    assert summary["model_ready"] is False
    assert runtime_detection_health(None, [])["model_ready"] is None


def test_runtime_detection_health_clears_historical_error_after_inference_recovers():
    summary = runtime_detection_health(
        {"usable": True, "state": "stable"},
        [{
            "name": "anomaly", "loaded": True, "backend": "torch-cuda",
            "last_error": "old failure", "consecutive_failures": 0,
        }],
    )
    assert summary["model_ready"] is True
    assert summary["model_error"] is None
