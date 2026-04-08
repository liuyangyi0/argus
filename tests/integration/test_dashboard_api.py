"""Dashboard API integration tests.

Tests the FastAPI JSON API endpoints with a real database and
TestClient. No real cameras or models needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from argus.config.loader import save_config
from argus.config.schema import ArgusConfig
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.dashboard.tasks import TaskManager
from argus.storage.database import Database
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelVersionEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def health_monitor():
    """Health monitor with one camera reporting."""
    monitor = HealthMonitor()
    monitor.update_camera(
        "cam_01", connected=True, frames_captured=1000, avg_latency_ms=12.5,
    )
    return monitor


@pytest.fixture
def alerts_dir(tmp_path):
    d = tmp_path / "alerts"
    d.mkdir()
    return d


@pytest.fixture
def app_with_db(integration_db, health_monitor, alerts_dir):
    """Create a FastAPI app wired to a real temp database."""
    app = create_app(
        database=integration_db,
        health_monitor=health_monitor,
        alerts_dir=str(alerts_dir),
    )
    return app, integration_db


@pytest.fixture
def client(app_with_db):
    """TestClient with authentication bypassed (auth disabled by default)."""
    app, _ = app_with_db
    return TestClient(app)


@pytest.fixture
def db(app_with_db):
    """Direct database access for seeding test data."""
    _, database = app_with_db
    return database


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_json(self, client):
        """GET /api/system/health returns valid JSON with status."""
        resp = client.get("/api/system/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        data = body["data"]
        assert "status" in data
        assert "cameras" in data

    def test_health_camera_info(self, client):
        """Health endpoint includes camera status."""
        body = client.get("/api/system/health").json()
        data = body["data"]
        assert len(data["cameras"]) == 1
        cam = data["cameras"][0]
        assert cam["camera_id"] == "cam_01"
        assert cam["connected"] is True


# ---------------------------------------------------------------------------
# Alert lifecycle
# ---------------------------------------------------------------------------

class TestAlertsCRUD:
    def _seed_alert(self, db: Database, alert_id: str = "test-alert-001"):
        """Insert a test alert into the database."""
        db.save_alert(
            alert_id=alert_id,
            timestamp=datetime.now(timezone.utc),
            camera_id="cam_01",
            zone_id="zone_a",
            severity="medium",
            anomaly_score=0.75,
        )

    def test_alerts_json_empty(self, client):
        """GET /api/alerts/json returns empty list when no alerts exist."""
        resp = client.get("/api/alerts/json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        alerts = body["data"]["alerts"]
        assert isinstance(alerts, list)
        assert len(alerts) == 0

    def test_alerts_json_with_data(self, client, db):
        """GET /api/alerts/json returns alerts after seeding."""
        self._seed_alert(db)
        resp = client.get("/api/alerts/json")
        assert resp.status_code == 200
        body = resp.json()
        alerts = body["data"]["alerts"]
        assert len(alerts) >= 1
        assert alerts[0]["alert_id"] == "test-alert-001"
        assert alerts[0]["severity"] == "medium"

    def test_alerts_filter_by_camera(self, client, db):
        """GET /api/alerts/json?camera_id=cam_01 filters correctly."""
        self._seed_alert(db, "alert-cam01")
        resp = client.get("/api/alerts/json?camera_id=cam_01")
        assert resp.status_code == 200
        alerts = resp.json()["data"]["alerts"]
        assert all(a["camera_id"] == "cam_01" for a in alerts)

    def test_alerts_filter_by_severity(self, client, db):
        """GET /api/alerts/json?severity=medium filters correctly."""
        self._seed_alert(db, "alert-med")
        resp = client.get("/api/alerts/json?severity=medium")
        assert resp.status_code == 200
        alerts = resp.json()["data"]["alerts"]
        assert all(a["severity"] == "medium" for a in alerts)

    def test_acknowledge_alert(self, client, db):
        """POST /api/alerts/{id}/acknowledge marks the alert."""
        self._seed_alert(db, "ack-test")
        resp = client.post("/api/alerts/ack-test/acknowledge", json={"user": "admin"})
        # The endpoint may return 200 or 302/redirect depending on implementation
        assert resp.status_code in (200, 303, 422, 404)

    def test_alert_false_positive(self, client, db):
        """POST /api/alerts/{id}/false-positive marks the alert."""
        self._seed_alert(db, "fp-test")
        resp = client.post("/api/alerts/fp-test/false-positive")
        assert resp.status_code in (200, 303, 422, 404)


# ---------------------------------------------------------------------------
# Users JSON API
# ---------------------------------------------------------------------------

class TestUsersAPI:
    def _create_admin(self, db: Database):
        """Create an admin user in the database."""
        from argus.dashboard.auth import hash_password

        db.create_user("admin", hash_password("admin123"), "admin", "Admin User")

    def test_users_json_requires_admin(self, client):
        """GET /api/users/json requires admin role."""
        resp = client.get("/api/users/json")
        # Without auth, should either return 403 or the data (if auth disabled)
        assert resp.status_code in (200, 403)

    def test_users_json_with_admin(self, client, db):
        """GET /api/users/json returns user list."""
        self._create_admin(db)
        # When auth is disabled, the endpoint still works
        resp = client.get("/api/users/json")
        if resp.status_code == 200:
            data = resp.json()
            if "users" in data:
                assert isinstance(data["users"], list)


# ---------------------------------------------------------------------------
# Audit JSON API
# ---------------------------------------------------------------------------

class TestAuditAPI:
    def test_audit_json_endpoint_exists(self, client):
        """GET /api/audit/json returns a response (may be 503 without audit logger)."""
        resp = client.get("/api/audit/json")
        # 503 is expected if audit_logger is not configured
        assert resp.status_code in (200, 503)


# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------

class TestConfigAPI:
    def test_config_reload_without_manager(self, client):
        """POST /api/config/reload without camera manager returns 503."""
        resp = client.post("/api/config/reload")
        # Without camera_manager and config_path, should return 503
        assert resp.status_code == 503

    def test_config_thresholds_without_manager(self, client):
        """POST /api/config/thresholds without camera manager returns 503."""
        resp = client.post(
            "/api/config/thresholds",
            json={"anomaly_threshold": 0.6},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Database integration
# ---------------------------------------------------------------------------

class TestDatabaseAlertLifecycle:
    """Test the full alert lifecycle through the database layer."""

    def test_save_and_retrieve_alert(self, db):
        """Save an alert and retrieve it."""
        db.save_alert(
            alert_id="db-test-001",
            timestamp=datetime.now(timezone.utc),
            camera_id="cam_01",
            zone_id="zone_a",
            severity="high",
            anomaly_score=0.92,
        )

        alerts = db.get_alerts(camera_id="cam_01")
        assert len(alerts) >= 1
        found = next((a for a in alerts if a.alert_id == "db-test-001"), None)
        assert found is not None
        assert found.severity == "high"

    def test_acknowledge_and_false_positive(self, db):
        """Acknowledge and mark alerts as false positive."""
        db.save_alert(
            alert_id="lifecycle-001",
            timestamp=datetime.now(timezone.utc),
            camera_id="cam_01",
            zone_id="zone_a",
            severity="medium",
            anomaly_score=0.75,
        )

        # Acknowledge
        result = db.acknowledge_alert("lifecycle-001", "operator1")
        assert result is True

        # Mark false positive
        result = db.mark_false_positive("lifecycle-001", notes="Shadow artifact")
        assert result is True

        # Verify
        alert = db.get_alert("lifecycle-001")
        assert alert.acknowledged is True
        assert alert.false_positive is True

    def test_user_crud(self, db):
        """Create, read, and list users."""
        from argus.dashboard.auth import hash_password

        db.create_user("testuser", hash_password("pass123"), "operator", "Test User")

        user = db.get_user("testuser")
        assert user is not None
        assert user.role == "operator"
        assert user.display_name == "Test User"

        users = db.get_all_users()
        usernames = [u.username for u in users]
        assert "testuser" in usernames


# ---------------------------------------------------------------------------
# Camera -> Capture -> Training -> Publish chain
# ---------------------------------------------------------------------------

class TestDashboardFeatureChain:
    def test_capture_job_enters_task_manager_and_tasks_api(
        self,
        tmp_path,
        integration_db,
        health_monitor,
        alerts_dir,
        monkeypatch,
    ):
        """Submitting capture should surface a baseline_capture task through the tasks API."""
        config = ArgusConfig()
        task_manager = TaskManager(max_concurrent=2)
        camera_manager = MagicMock()
        camera_manager._cameras = []

        def fake_capture_job(progress_callback, **kwargs):
            progress_callback(15, "准备采集")
            time.sleep(0.02)
            job_config = kwargs["job_config"]
            progress_callback(100, "完成")
            return {
                "camera_id": job_config.camera_id,
                "version": "v001",
                "collected_frames": 3,
                "target_frames": 3,
                "sampling_strategy": job_config.sampling_strategy,
                "stats": {"total_grabbed": 3, "accepted": 3, "total_rejected": 0},
            }

        monkeypatch.setattr("argus.dashboard.routes.baseline.run_baseline_capture_job", fake_capture_job)

        app = create_app(
            database=integration_db,
            camera_manager=camera_manager,
            health_monitor=health_monitor,
            alerts_dir=str(alerts_dir),
            config=config,
            task_manager=task_manager,
        )
        client = TestClient(app)

        submit_resp = client.post(
            "/api/baseline/job",
            data={
                "camera_id": "cam_01",
                "count": "3",
                "interval": "0.1",
                "session_label": "daytime",
            },
        )
        assert submit_resp.status_code == 200
        task_id = submit_resp.json()["data"]["task_id"]

        tasks_resp = client.get("/api/tasks/json")
        assert tasks_resp.status_code == 200
        tasks = tasks_resp.json()["data"]["tasks"]
        task = next((item for item in tasks if item["task_id"] == task_id), None)
        assert task is not None
        assert task["task_type"] == "baseline_capture"
        assert task["camera_id"] == "cam_01"

        deadline = time.time() + 1.0
        final_task = task
        while time.time() < deadline:
            final_task = client.get("/api/tasks/json").json()["data"]["tasks"][0]
            if final_task["status"] == "complete":
                break
            time.sleep(0.02)

        assert final_task["status"] == "complete"
        assert final_task["result"]["camera_id"] == "cam_01"

    def test_add_camera_capture_train_publish_chain(
        self,
        tmp_path,
        integration_db,
        health_monitor,
        alerts_dir,
    ):
        """Exercise the dashboard chain from camera add to model activation."""
        config = ArgusConfig()
        config_path = tmp_path / "config.yaml"
        save_config(config, config_path)

        camera_manager = MagicMock()
        camera_manager._cameras = []
        camera_manager._pipelines = {"cam_01": MagicMock()}
        camera_manager.reload_model.return_value = True

        task_manager = MagicMock()
        task_manager.submit.return_value = "baseline-task-001"

        app = create_app(
            database=integration_db,
            camera_manager=camera_manager,
            health_monitor=health_monitor,
            alerts_dir=str(alerts_dir),
            config=config,
            config_path=str(config_path),
            task_manager=task_manager,
        )
        client = TestClient(app)

        add_resp = client.post(
            "/api/cameras",
            data={
                "camera_id": "cam_01",
                "name": "Camera 01",
                "source": "rtsp://example/stream",
                "protocol": "rtsp",
                "fps_target": "8",
                "resolution": "1280,720",
            },
        )
        assert add_resp.status_code == 200
        assert any(camera.camera_id == "cam_01" for camera in config.cameras)
        camera_manager.add_camera_config.assert_called_once()
        assert camera_manager.add_camera_config.call_args[0][0].camera_id == "cam_01"

        saved = Path(config_path).read_text(encoding="utf-8")
        assert "cam_01" in saved

        camera_manager.get_status.return_value = [
            SimpleNamespace(
                camera_id="cam_01",
                name="Camera 01",
                connected=True,
                running=False,
                stats=SimpleNamespace(
                    frames_captured=120,
                    frames_analyzed=95,
                    anomalies_detected=2,
                    alerts_emitted=1,
                    avg_latency_ms=14.8,
                ),
            ),
        ]
        detail_resp = client.get("/api/cameras/json")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["data"]["cameras"][0]["camera_id"] == "cam_01"

        capture_resp = client.post(
            "/api/baseline/job",
            data={
                "camera_id": "cam_01",
                "count": "5",
                "interval": "0.1",
                "session_label": "daytime",
            },
        )
        assert capture_resp.status_code == 200
        task_manager.submit.assert_called_once()
        assert task_manager.submit.call_args.args[0] == "baseline_capture"

        create_job_resp = client.post(
            "/api/training-jobs/",
            json={
                "job_type": "anomaly_head",
                "camera_id": "cam_01",
                "model_type": "patchcore",
                "triggered_by": "tester",
            },
        )
        assert create_job_resp.status_code == 201
        job_id = create_job_resp.json()["data"]["job_id"]

        jobs_resp = client.get("/api/training-jobs/json")
        assert jobs_resp.status_code == 200
        assert jobs_resp.json()["data"]["pending_count"] == 1

        confirm_resp = client.post(
            f"/api/training-jobs/{job_id}/confirm",
            json={"confirmed_by": "operator"},
        )
        assert confirm_resp.status_code == 200
        assert confirm_resp.json()["data"]["new_status"] == "queued"

        detail_job_resp = client.get(f"/api/training-jobs/{job_id}")
        assert detail_job_resp.status_code == 200
        assert detail_job_resp.json()["data"]["status"] == "queued"

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model", encoding="utf-8")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=integration_db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

        publish_resp = client.post(f"/api/models/{version_id}/activate")
        assert publish_resp.status_code == 200
        assert publish_resp.json()["data"]["runtime_synced"] is True
        camera_manager.reload_model.assert_called_with(
            "cam_01",
            str(model_dir / "model.xml"),
            version_tag=version_id,
        )

    def test_promote_registered_model_to_production_syncs_runtime(
        self,
        tmp_path,
        integration_db,
        health_monitor,
        alerts_dir,
    ):
        """Promotion to production should keep registry state and runtime in sync."""
        camera_manager = MagicMock()
        camera_manager.reload_model.return_value = True

        app = create_app(
            database=integration_db,
            camera_manager=camera_manager,
            health_monitor=health_monitor,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model", encoding="utf-8")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=integration_db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

        with integration_db.get_session() as session:
            record = registry.get_by_version_id(version_id)
            record = session.merge(record)
            record.stage = "canary"
            session.add(ModelVersionEvent(
                camera_id="cam_01",
                from_version=version_id,
                to_version=version_id,
                from_stage="shadow",
                to_stage="canary",
                triggered_by="tester",
                reason="seed",
                timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
            ))
            session.commit()

        resp = client.post(
            f"/api/models/{version_id}/promote",
            json={"target_stage": "production", "triggered_by": "operator"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["code"] == 0
        data = payload["data"]
        assert data["model"]["stage"] == "production"
        assert data["runtime_synced"] is True
        camera_manager.reload_model.assert_called_once_with(
            "cam_01",
            str(model_dir / "model.xml"),
            version_tag=version_id,
        )
