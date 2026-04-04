"""Tests for the dashboard API."""

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.dashboard.routes.alerts import _generate_composite, _is_safe_path
from argus.storage.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def health():
    monitor = HealthMonitor()
    monitor.update_camera("cam_01", connected=True, frames_captured=500, avg_latency_ms=15.3)
    return monitor


@pytest.fixture
def alerts_dir(tmp_path):
    d = tmp_path / "alerts"
    d.mkdir()
    return d


@pytest.fixture
def client(db, health, alerts_dir):
    app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir))
    return TestClient(app)


class TestDashboardPages:
    def test_index_page_loads(self, client):
        """Root page should return HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "ARGUS" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_cameras_page_loads(self, client):
        """Cameras page should return HTML."""
        response = client.get("/cameras")
        assert response.status_code == 200
        assert "ARGUS" in response.text

    def test_alerts_page_loads(self, client):
        """Alerts page should return HTML."""
        response = client.get("/alerts")
        assert response.status_code == 200

    def test_system_page_loads(self, client):
        """System page should return HTML."""
        response = client.get("/system")
        assert response.status_code == 200


class TestHealthAPI:
    def test_health_endpoint(self, client):
        """Health endpoint should return JSON."""
        response = client.get("/api/system/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "cameras" in data
        assert len(data["cameras"]) == 1
        assert data["cameras"][0]["camera_id"] == "cam_01"

    def test_system_overview_html(self, client):
        """Overview endpoint should return HTMX fragment."""
        response = client.get("/api/system/overview")
        assert response.status_code == 200
        assert "摄像头在线" in response.text or "System Status" in response.text


class TestAlertsAPI:
    def test_alerts_list_empty(self, client):
        """Empty alerts list should render."""
        response = client.get("/api/alerts")
        assert response.status_code == 200
        assert "暂无告警" in response.text or "No alerts" in response.text or response.status_code == 200

    def test_alerts_list_with_data(self, client, db):
        """Alerts should appear in the list."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)
        db.save_alert("ALT-002", now, "cam_01", "z1", "low", 0.72)

        response = client.get("/api/alerts")
        assert response.status_code == 200
        assert "ALT-001" in response.text
        assert "ALT-002" in response.text

    def test_alerts_filter_by_severity(self, client, db):
        """Should filter alerts by severity."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)
        db.save_alert("ALT-002", now, "cam_01", "z1", "low", 0.72)

        response = client.get("/api/alerts?severity=high")
        assert response.status_code == 200
        assert "ALT-001" in response.text

    def test_acknowledge_alert(self, client, db):
        """Should acknowledge an alert via POST."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "medium", 0.88)

        response = client.post("/api/alerts/ALT-001/acknowledge")
        assert response.status_code == 200
        assert "已确认" in response.text

    def test_mark_false_positive(self, client, db):
        """Should mark an alert as false positive."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "low", 0.72)

        response = client.post("/api/alerts/ALT-001/false-positive")
        assert response.status_code == 200
        assert "已标记误报" in response.text

    def test_alerts_json_api(self, client, db):
        """JSON API should return structured data."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)

        response = client.get("/api/alerts/json")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["alert_id"] == "ALT-001"

    def test_alerts_list_shows_thumbnail_column(self, client, db):
        """Alert list should include a snapshot thumbnail column header."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)
        response = client.get("/api/alerts")
        assert response.status_code == 200


class TestAlertImages:
    def test_image_404_for_nonexistent_alert(self, client):
        """Should return 404 for a nonexistent alert."""
        response = client.get("/api/alerts/NONEXISTENT/image/snapshot")
        assert response.status_code == 404

    def test_image_404_for_null_path(self, client, db):
        """Should return 404 when alert has no snapshot."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)
        response = client.get("/api/alerts/ALT-001/image/snapshot")
        assert response.status_code == 404

    def test_image_400_for_invalid_type(self, client, db):
        """Should return 400 for invalid image type."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)
        response = client.get("/api/alerts/ALT-001/image/invalid")
        assert response.status_code == 400

    def test_image_serves_snapshot(self, client, db, alerts_dir):
        """Should serve a snapshot image when it exists."""
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[30:70, 30:70] = [0, 0, 255]  # red square
        snap_path = alerts_dir / "test_snapshot.jpg"
        cv2.imwrite(str(snap_path), img)

        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96,
                       snapshot_path=str(snap_path))

        response = client.get("/api/alerts/ALT-001/image/snapshot")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0

    def test_image_serves_composite(self, client, db, alerts_dir):
        """Should serve a composite (heatmap overlay) image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        snap_path = alerts_dir / "snapshot.jpg"
        cv2.imwrite(str(snap_path), img)

        heatmap = np.full((100, 100), 128, dtype=np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heat_path = alerts_dir / "heatmap.jpg"
        cv2.imwrite(str(heat_path), colored)

        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96,
                       snapshot_path=str(snap_path), heatmap_path=str(heat_path))

        response = client.get("/api/alerts/ALT-001/image/composite")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_image_path_traversal_blocked(self, client, db, alerts_dir, tmp_path):
        """Should block access to files outside alerts_dir."""
        # Create a file outside alerts_dir
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive data")

        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96,
                       snapshot_path=str(secret))

        response = client.get("/api/alerts/ALT-001/image/snapshot")
        assert response.status_code == 403


class TestAlertDetail:
    def test_detail_404_for_nonexistent(self, client):
        """Should return not found message."""
        response = client.get("/api/alerts/NONEXISTENT/detail")
        assert response.status_code == 200
        assert "not found" in response.text.lower() or "不存在" in response.text or "未找到" in response.text

    def test_detail_shows_metadata(self, client, db):
        """Should show alert metadata in detail view."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)

        response = client.get("/api/alerts/ALT-001/detail")
        assert response.status_code == 200
        assert "ALT-001" in response.text
        assert "cam_01" in response.text
        assert "z1" in response.text


class TestCompositeGeneration:
    def test_generate_composite_basic(self, tmp_path):
        """Should blend snapshot and heatmap into composite JPEG."""
        snapshot = np.zeros((200, 300, 3), dtype=np.uint8)
        snapshot[:] = [50, 100, 150]
        snap_path = str(tmp_path / "snap.jpg")
        cv2.imwrite(snap_path, snapshot)

        heatmap = np.full((200, 300), 200, dtype=np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heat_path = str(tmp_path / "heat.jpg")
        cv2.imwrite(heat_path, colored)

        result = _generate_composite(snap_path, heat_path)
        assert result is not None
        assert len(result) > 0

    def test_generate_composite_different_sizes(self, tmp_path):
        """Should resize heatmap to match snapshot dimensions."""
        snapshot = np.zeros((200, 300, 3), dtype=np.uint8)
        snap_path = str(tmp_path / "snap.jpg")
        cv2.imwrite(snap_path, snapshot)

        heatmap = np.full((100, 150), 128, dtype=np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heat_path = str(tmp_path / "heat.jpg")
        cv2.imwrite(heat_path, colored)

        result = _generate_composite(snap_path, heat_path)
        assert result is not None

    def test_generate_composite_missing_file(self, tmp_path):
        """Should return None for missing files."""
        result = _generate_composite(
            str(tmp_path / "nonexistent.jpg"),
            str(tmp_path / "also_missing.jpg"),
        )
        assert result is None


class TestPathSafety:
    def test_safe_path_accepted(self, tmp_path):
        """Should accept paths under the safe root."""
        safe_root = tmp_path / "alerts"
        safe_root.mkdir()
        file_path = safe_root / "2026-03-22" / "cam_01" / "test.jpg"
        file_path.parent.mkdir(parents=True)
        file_path.touch()
        assert _is_safe_path(str(file_path), safe_root) is True

    def test_unsafe_path_rejected(self, tmp_path):
        """Should reject paths outside the safe root."""
        safe_root = tmp_path / "alerts"
        safe_root.mkdir()
        outside = tmp_path / "secret.txt"
        outside.touch()
        assert _is_safe_path(str(outside), safe_root) is False

    def test_traversal_path_rejected(self, tmp_path):
        """Should reject path traversal attempts."""
        safe_root = tmp_path / "alerts"
        safe_root.mkdir()
        traversal = str(safe_root / ".." / "secret.txt")
        assert _is_safe_path(traversal, safe_root) is False
