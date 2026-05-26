"""Tests for the alert JSON API routes (/api/alerts/*)."""

import io
import json
import zipfile
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, AuthConfig
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.dashboard.auth import create_session_token
from argus.storage.alert_recording import AlertRecordingStore
from argus.storage.database import Database
from argus.storage.models import AlertRecord, AlertRecordingRecord


# ── Fixtures (mirrors test_dashboard.py pattern) ──


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


@pytest.fixture
def auth_client(db, health, alerts_dir):
    config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="alerts-test-token"))
    app = create_app(
        database=db,
        health_monitor=health,
        alerts_dir=str(alerts_dir),
        config=config,
    )
    return TestClient(app), app


def _insert_alert(db: Database, alert_id: str = "test-alert-001", **overrides):
    """Helper to insert a test alert record."""
    defaults = {
        "alert_id": alert_id,
        "camera_id": "cam_01",
        "timestamp": datetime(2026, 4, 10, 12, 0, 0),
        "zone_id": "zone_a",
        "severity": "high",
        "anomaly_score": 0.95,
        "snapshot_path": "test.jpg",
        "heatmap_path": "test_hm.jpg",
        "workflow_status": "new",
    }
    defaults.update(overrides)
    with db.get_session() as session:
        alert = AlertRecord(**defaults)
        session.add(alert)
        session.commit()


def _set_role_cookie(client: TestClient, app, role: str) -> None:
    token = create_session_token(f"{role}_user", role, app.state.session_secret)
    client.cookies.set("argus_session", token)


def _save_recording_record(db: Database, alert_id: str, *, status: str = "complete"):
    db.save_alert_recording(
        AlertRecordingRecord(
            alert_id=alert_id,
            camera_id="cam_01",
            severity="high",
            recording_path=f"/tmp/{alert_id}",
            start_timestamp=0,
            end_timestamp=1,
            trigger_timestamp=0.5,
            frame_count=1,
            fps=10,
            file_size_bytes=128,
            status=status,
        )
    )


# ── GET /api/alerts/json ──


class TestAlertsJsonEndpoint:
    def test_get_alerts_json_empty(self, client):
        """No alerts in DB should return empty list."""
        resp = client.get("/api/alerts/json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["alerts"] == []
        assert body["data"]["total"] == 0

    def test_get_alerts_json_with_data(self, client, db):
        """Inserted alert should appear in JSON response."""
        _insert_alert(db, "alert-abc")
        resp = client.get("/api/alerts/json")
        assert resp.status_code == 200
        body = resp.json()
        alerts = body["data"]["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "alert-abc"
        assert alerts[0]["severity"] == "high"
        assert body["data"]["total"] == 1

    def test_get_alerts_filter_by_camera(self, client, db):
        """Filter by camera_id should return only matching alerts."""
        _insert_alert(db, "a1", camera_id="cam_01")
        _insert_alert(db, "a2", camera_id="cam_02")
        resp = client.get("/api/alerts/json?camera_id=cam_01")
        body = resp.json()
        alerts = body["data"]["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["camera_id"] == "cam_01"

    def test_get_alerts_filter_by_severity(self, client, db):
        """Filter by severity should return only matching alerts."""
        _insert_alert(db, "s1", severity="high")
        _insert_alert(db, "s2", severity="low")
        _insert_alert(db, "s3", severity="high")
        resp = client.get("/api/alerts/json?severity=low")
        body = resp.json()
        alerts = body["data"]["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "low"

    def test_get_alerts_json_limit(self, client, db):
        """Limit parameter should cap the number of returned alerts."""
        for i in range(5):
            _insert_alert(db, f"lim-{i}")
        resp = client.get("/api/alerts/json?limit=2")
        body = resp.json()
        assert len(body["data"]["alerts"]) == 2
        # total should still reflect all alerts
        assert body["data"]["total"] == 5

    def test_get_alerts_json_offset(self, client, db):
        """Offset parameter should skip earlier rows while preserving total."""
        for i in range(5):
            _insert_alert(
                db,
                f"off-{i}",
                timestamp=datetime(2026, 4, 10, 12, i, 0),
            )

        resp = client.get("/api/alerts/json?limit=2&offset=2")
        body = resp.json()

        assert resp.status_code == 200
        assert [a["alert_id"] for a in body["data"]["alerts"]] == ["off-2", "off-1"]
        assert body["data"]["total"] == 5


class TestAlertDetailRecordingState:
    def test_detail_uses_db_recording_status_when_store_is_absent(self, client, db):
        _insert_alert(db, "detail-db-rec")
        _save_recording_record(db, "detail-db-rec", status="recording")

        resp = client.get("/api/alerts/detail-db-rec/detail")

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["has_recording"] is True
        assert data["recording_status"] == "recording"

    def test_detail_prefers_replay_metadata_status_over_db_status(
        self, db, health, alerts_dir, tmp_path,
    ):
        _insert_alert(db, "detail-store-rec")
        _save_recording_record(db, "detail-store-rec", status="recording")

        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))
        app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir))
        app.state.recording_store = store
        client = TestClient(app)

        rec_dir = store.archive_dir / "2026-04-10" / "cam_01" / "detail-store-rec"
        rec_dir.mkdir(parents=True)
        (rec_dir / "recording.mp4").write_bytes(b"mp4 bytes")
        (rec_dir / "metadata.json").write_text(
            json.dumps({
                "alert_id": "detail-store-rec",
                "camera_id": "cam_01",
                "status": "complete",
            }),
            encoding="utf-8",
        )

        resp = client.get("/api/alerts/detail-store-rec/detail")

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["has_recording"] is True
        assert data["recording_status"] == "complete"


# ── Image path safety ──


class TestAlertImageSafety:
    def test_rejects_sibling_path_with_shared_prefix(self, client, db, alerts_dir):
        """Resolved path containment must not accept startswith-style siblings."""
        evil_dir = alerts_dir.parent / f"{alerts_dir.name}_evil"
        evil_dir.mkdir()
        evil_file = evil_dir / "snapshot.jpg"
        evil_file.write_bytes(b"not really a jpeg")
        _insert_alert(db, "evil-prefix", snapshot_path=str(evil_file))

        resp = client.get("/api/alerts/evil-prefix/image/snapshot")

        assert resp.status_code == 403

    def test_allows_file_under_alerts_dir(self, client, db, alerts_dir):
        safe_file = alerts_dir / "snapshot.jpg"
        safe_file.write_bytes(b"jpeg bytes")
        _insert_alert(db, "safe-image", snapshot_path=str(safe_file))

        resp = client.get("/api/alerts/safe-image/image/snapshot")

        assert resp.status_code == 200
        assert resp.content == b"jpeg bytes"


class TestEvidencePackageExport:
    def test_export_evidence_package_contains_images_and_replay(
        self, db, health, alerts_dir, tmp_path,
    ):
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))
        app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir))
        app.state.recording_store = store
        client = TestClient(app)

        snapshot = alerts_dir / "snapshot.jpg"
        heatmap = alerts_dir / "heatmap.jpg"
        snapshot.write_bytes(b"snapshot bytes")
        heatmap.write_bytes(b"heatmap bytes")
        _insert_alert(
            db,
            "alert-zip",
            snapshot_path=str(snapshot),
            heatmap_path=str(heatmap),
        )

        rec_dir = store.archive_dir / "2026-04-10" / "cam_01" / "alert-zip"
        rec_dir.mkdir(parents=True)
        (rec_dir / "recording.mp4").write_bytes(b"mp4 bytes")
        (rec_dir / "metadata.json").write_text(
            json.dumps({
                "alert_id": "alert-zip",
                "camera_id": "cam_01",
                "status": "complete",
                "frame_count": 1,
            }),
            encoding="utf-8",
        )
        (rec_dir / "signals.json").write_text(
            json.dumps({"timestamps": [0], "anomaly_scores": [0.95]}),
            encoding="utf-8",
        )

        resp = client.get("/api/alerts/alert-zip/evidence.zip")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = set(zf.namelist())
            assert "manifest.json" in names
            assert "images/snapshot.jpg" in names
            assert "images/heatmap.jpg" in names
            assert "replay/metadata.json" in names
            assert "replay/signals.json" in names
            assert "replay/recording.mp4" in names

            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["alert_id"] == "alert-zip"
            assert manifest["has_recording"] is True
            assert "replay/recording.mp4" in manifest["files"]

    def test_export_evidence_package_unknown_alert_returns_404(self, client):
        resp = client.get("/api/alerts/missing/evidence.zip")

        assert resp.status_code == 404


# ── POST /api/alerts/{alert_id}/acknowledge ──


class TestAcknowledgeAlert:
    def test_acknowledge_alert(self, client, db):
        """Acknowledging an existing alert should succeed."""
        _insert_alert(db, "ack-001")
        resp = client.post("/api/alerts/ack-001/acknowledge")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["alert_id"] == "ack-001"

        # Verify in DB
        alerts = db.get_alerts()
        assert alerts[0].acknowledged is True

    def test_acknowledge_nonexistent_alert(self, client):
        """Acknowledging a non-existent alert should return 404."""
        resp = client.post("/api/alerts/no-such-alert/acknowledge")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == 40400


# ── handle_alerts permission checks ──


class TestAlertMutationPermissions:
    @pytest.mark.parametrize(
        ("method", "path", "json_body"),
        [
            ("post", "/api/alerts/perm-001/acknowledge", None),
            ("post", "/api/alerts/perm-001/false-positive", None),
            ("delete", "/api/alerts/perm-001", None),
            ("post", "/api/alerts/perm-001/workflow", {"status": "investigating"}),
            ("post", "/api/alerts/bulk-acknowledge", {"alert_ids": ["perm-001"]}),
            ("post", "/api/alerts/bulk-false-positive", {"alert_ids": ["perm-001"]}),
            ("post", "/api/alerts/bulk-delete", {"alert_ids": ["perm-001"]}),
        ],
    )
    def test_viewer_cannot_mutate_alerts(self, auth_client, db, method, path, json_body):
        """Viewer role lacks handle_alerts and must be blocked from mutations."""
        client, app = auth_client
        _set_role_cookie(client, app, "viewer")
        _insert_alert(db, "perm-001")

        resp = client.request(method.upper(), path, json=json_body)

        assert resp.status_code == 403
        assert resp.json()["code"] == 40300

    def test_operator_can_acknowledge_alert(self, auth_client, db):
        """Operator role has handle_alerts and can use mutation endpoints."""
        client, app = auth_client
        _set_role_cookie(client, app, "operator")
        _insert_alert(db, "perm-op")

        resp = client.post("/api/alerts/perm-op/acknowledge")

        assert resp.status_code == 200
        assert resp.json()["code"] == 0


# ── POST /api/alerts/bulk-acknowledge ──


class TestBulkAcknowledge:
    def test_bulk_acknowledge(self, client, db):
        """Bulk acknowledge should mark multiple alerts."""
        _insert_alert(db, "bulk-1")
        _insert_alert(db, "bulk-2")
        _insert_alert(db, "bulk-3")
        resp = client.post(
            "/api/alerts/bulk-acknowledge",
            json={"alert_ids": ["bulk-1", "bulk-2"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["count"] == 2

        # Verify third alert is still unacknowledged
        alerts = db.get_alerts()
        ack_map = {a.alert_id: a.acknowledged for a in alerts}
        assert ack_map["bulk-1"] is True
        assert ack_map["bulk-2"] is True
        assert ack_map["bulk-3"] is False

    def test_bulk_acknowledge_with_invalid_ids(self, client, db):
        """Bulk acknowledge should skip non-existent IDs gracefully."""
        _insert_alert(db, "exists-1")
        resp = client.post(
            "/api/alerts/bulk-acknowledge",
            json={"alert_ids": ["exists-1", "does-not-exist"]},
        )
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["count"] == 1


# ── DELETE /api/alerts/{alert_id} ──


class TestDeleteAlert:
    def test_delete_alert(self, client, db):
        """Deleting an existing alert should remove it."""
        _insert_alert(db, "del-001")
        resp = client.delete("/api/alerts/del-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["alert_id"] == "del-001"

        # Verify it's gone
        assert db.get_alert_count() == 0

    def test_delete_nonexistent_alert(self, client):
        """Deleting a non-existent alert should return 404."""
        resp = client.delete("/api/alerts/no-such-alert")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == 40400


# ── POST /api/alerts/{alert_id}/false-positive ──


class TestFalsePositive:
    def test_mark_false_positive(self, client, db):
        """Marking an alert as false positive should succeed."""
        _insert_alert(db, "fp-001")
        resp = client.post("/api/alerts/fp-001/false-positive")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0

        # Verify in DB
        alerts = db.get_alerts()
        assert alerts[0].false_positive is True

    def test_mark_false_positive_nonexistent(self, client):
        """Marking a non-existent alert as false positive should return 404."""
        resp = client.post("/api/alerts/nonexistent/false-positive")
        assert resp.status_code == 404


# ── POST /api/alerts/bulk-delete ──


class TestBulkDelete:
    def test_bulk_delete(self, client, db):
        """Bulk delete should remove multiple alerts."""
        _insert_alert(db, "bd-1")
        _insert_alert(db, "bd-2")
        _insert_alert(db, "bd-3")
        resp = client.post(
            "/api/alerts/bulk-delete",
            json={"alert_ids": ["bd-1", "bd-3"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["count"] == 2
        assert db.get_alert_count() == 1
