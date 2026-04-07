"""Tests for the health monitoring module."""

from argus.core.health import HealthMonitor, HealthStatus


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
