"""Tests for the dashboard API."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette import requests as starlette_requests

from argus.__main__ import _register_training_job_processing
from argus.capture.manager import CameraManager
from argus.config.loader import load_config, save_config
from argus.config.schema import AlertConfig, ArgusConfig, CameraConfig
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.dashboard.routes.baseline import _train_model_task
from argus.dashboard.routes.alerts import _generate_composite, _is_safe_path
from argus.storage.model_registry import ModelRegistry
from argus.storage.database import Database
from argus.storage.models import ModelVersionEvent


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
        """Root page should return Vue SPA HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert '<div id="app">' in response.text or "<div id=\"app\">" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_cameras_page_loads(self, client):
        """Cameras page should return Vue SPA HTML (client-side routing)."""
        response = client.get("/cameras")
        assert response.status_code == 200
        assert '<div id="app">' in response.text or "<div id=\"app\">" in response.text

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
        """Health endpoint should return JSON envelope."""
        response = client.get("/api/system/health")
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 0
        data = body["data"]
        assert "status" in data
        assert "cameras" in data
        assert len(data["cameras"]) == 1
        assert data["cameras"][0]["camera_id"] == "cam_01"

class TestCameraForms:
    def test_add_camera_accepts_urlencoded_form_without_multipart(self, db, health, alerts_dir, monkeypatch, tmp_path):
        """Add-camera form should persist config and sync runtime state for urlencoded bodies."""
        config = ArgusConfig()
        config_path = tmp_path / "config.yaml"
        save_config(config, config_path)

        camera_manager = MagicMock()
        camera_manager._cameras = []

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
            config_path=str(config_path),
        )
        client = TestClient(app)

        monkeypatch.setattr(starlette_requests, "parse_options_header", None)

        response = client.post(
            "/api/cameras",
            data={
                "camera_id": "usb_cam_01",
                "name": "USB Camera 01",
                "source": "0",
                "protocol": "usb",
                "fps_target": "10",
                "resolution": "1280,720",
            },
        )

        assert response.status_code == 200
        assert any(camera.camera_id == "usb_cam_01" for camera in config.cameras)
        # The manager's add_camera_config() is called to thread-safely sync the list
        camera_manager.add_camera_config.assert_called_once()
        added = camera_manager.add_camera_config.call_args[0][0]
        assert added.camera_id == "usb_cam_01"
        assert config.cameras[-1].protocol == "usb"
        assert config.cameras[-1].resolution == (1280, 720)

        saved_config = load_config(config_path)
        assert any(camera.camera_id == "usb_cam_01" for camera in saved_config.cameras)

    def test_wall_status_uses_health_monitor_snapshot(self, health, alerts_dir):
        """Video wall status should not crash when reading per-camera health."""
        health.update_camera("cam_02", connected=False, error="stream lost")

        camera_manager = MagicMock()
        camera_manager.get_status.return_value = [
            SimpleNamespace(
                camera_id="cam_02",
                name="Camera 02",
                connected=False,
                running=False,
                model_version_id=None,
                stats=None,
            ),
        ]
        camera_manager.get_pipeline.return_value = None
        camera_manager.get_backpressure_stats.return_value = {
            "cam_02": {"pending": 0, "dropped": 0, "backpressured": False},
        }

        app = create_app(
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.get("/api/cameras/wall/status")

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["cameras"][0]["camera_id"] == "cam_02"
        assert data["cameras"][0]["degradation"] == "rtsp_broken"

    def test_wall_status_uses_batched_db_query(self, db, health, alerts_dir):
        """wall/status must collapse to O(1) SQL queries regardless of camera count.

        Regression target: the previous loop called ``get_alert_count`` and
        ``get_alerts(limit=1)`` per camera, producing 2N+1 queries per poll.
        With 4 cameras * 4 clients * 5 Hz this saturated SQLite and contended
        with AlertDispatcher writes. The route now uses
        ``get_wall_status_batch`` for a single round-trip.
        """
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import event

        # Seed alerts: one today (active, high severity) + one yesterday
        # (must NOT count toward alert_count_today, verifying the ``since``
        # cutoff in the batch query).
        now = datetime.now(tz=timezone.utc)
        for i in range(5):
            cam_id = f"cam_{i:02d}"
            db.save_alert(f"A-{i}-1", now - timedelta(minutes=i), cam_id, "z1", "high", 0.9)
            db.save_alert(f"A-{i}-2", now - timedelta(days=1), cam_id, "z1", "low", 0.6)

        camera_manager = MagicMock()
        camera_manager.get_status.return_value = [
            SimpleNamespace(
                camera_id=f"cam_{i:02d}",
                name=f"Camera {i:02d}",
                connected=True,
                running=True,
                model_version_id=None,
                stats=None,
            )
            for i in range(5)
        ]
        camera_manager.get_pipeline.return_value = None
        camera_manager.get_backpressure_stats.return_value = {
            f"cam_{i:02d}": {"pending": 0, "dropped": 0, "backpressured": False}
            for i in range(5)
        }

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        counter = {"n": 0}

        def _count(conn, cursor, stmt, params, context, executemany):  # noqa: ARG001
            counter["n"] += 1

        event.listen(db._engine, "before_cursor_execute", _count)
        try:
            response = client.get("/api/cameras/wall/status")
        finally:
            event.remove(db._engine, "before_cursor_execute", _count)

        assert response.status_code == 200
        body = response.json()
        data = body["data"]
        assert len(data["cameras"]) == 5

        # JSON shape must stay backward-compatible (frontend is untouched).
        sample = data["cameras"][0]
        required_keys = {
            "camera_id", "name", "status", "model_version", "fps",
            "current_score", "score_sparkline", "alert_count_today",
            "active_alert", "degradation", "frames_dropped", "backpressured",
        }
        assert required_keys.issubset(sample.keys())

        # Today's alerts get counted and the latest-if-active shows up.
        for tile in data["cameras"]:
            assert tile["alert_count_today"] == 1
            assert tile["active_alert"] is not None
            assert tile["active_alert"]["severity"] == "high"

        # Core invariant: query count is bounded (not proportional to N).
        # batch implementation issues ~2 statements (count + latest-row).
        # Allow a small ceiling for incidental framework queries, but assert
        # it is nowhere near 2*N+1 = 11 we had before.
        assert counter["n"] <= 4, f"expected <=4 SQL queries, got {counter['n']}"


class TestBaselineForms:
    def test_create_app_initializes_baseline_lifecycle(self, db, health, alerts_dir):
        """Dashboard app should expose a baseline lifecycle manager when a database is available."""
        app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir))

        assert app.state.baseline_lifecycle is not None

    def test_start_capture_submits_task(self, db, health, alerts_dir):
        """Baseline capture form should submit through the unified job endpoint."""
        task_manager = MagicMock()
        task_manager.submit.return_value = "baseline_capture-test1234"
        task_manager.get_task.return_value = None

        camera_manager = MagicMock()
        config = ArgusConfig()
        config.cameras = []

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
            task_manager=task_manager,
        )
        client = TestClient(app)

        response = client.post(
            "/api/baseline/job",
            data={
                "camera_id": "cam_01",
                "count": "5",
                "interval": "0.1",
                "session_label": "daytime",
            },
        )

        assert response.status_code == 200
        task_manager.submit.assert_called_once()
        assert task_manager.submit.call_args.args[0] == "baseline_capture"
        submit_kwargs = task_manager.submit.call_args.kwargs
        assert submit_kwargs["camera_id"] == "cam_01"
        job_config = submit_kwargs["job_config"]
        assert job_config.camera_id == "cam_01"
        assert job_config.session_label == "daytime"
        assert job_config.target_frames == 5
        assert job_config.sampling_strategy == "uniform"
        assert job_config.quality_config == config.capture_quality
        assert submit_kwargs["lifecycle"] is app.state.baseline_lifecycle

    def test_legacy_capture_endpoint_alias_submits_task(self, db, health, alerts_dir):
        """Quick capture clients posting to /capture should still start the same job flow."""
        task_manager = MagicMock()
        task_manager.submit.return_value = "baseline_capture-legacy"
        task_manager.get_task.return_value = None

        camera_manager = MagicMock()
        config = ArgusConfig()
        config.cameras = []

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
            task_manager=task_manager,
        )
        client = TestClient(app)

        response = client.post(
            "/api/baseline/capture",
            data={
                "camera_id": "cam_01",
                "count": "3",
                "interval": "0.2",
                "session_label": "daytime",
            },
        )

        assert response.status_code == 200
        task_manager.submit.assert_called_once()
        assert task_manager.submit.call_args.args[0] == "baseline_capture"

    def test_baseline_list_json_reads_default_zone_versions(self, db, health, alerts_dir, tmp_path):
        """Baseline list JSON should surface default-zone captures after a reload."""
        baselines_dir = tmp_path / "baselines"
        version_dir = baselines_dir / "cam_01" / "default" / "v001"
        version_dir.mkdir(parents=True)
        (version_dir / "baseline_00000.png").write_bytes(b"img")
        (version_dir / "capture_meta.json").write_text('{"session_label": "daytime"}')

        config = ArgusConfig()
        config.storage.baselines_dir = baselines_dir

        app = create_app(
            database=db,
            camera_manager=MagicMock(),
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        app.state.baseline_lifecycle.register_version("cam_01", "default", "v001", image_count=1)
        client = TestClient(app)

        response = client.get("/api/baseline/list/json")

        assert response.status_code == 200
        payload = response.json()
        assert payload["code"] == 0
        assert payload["data"]["baselines"] == [{
            "camera_id": "cam_01",
            "version": "v001",
            "image_count": 1,
            "session_label": "daytime",
            "status": "ready",
            "state": "draft",
        }]

    def test_delete_baseline_version_removes_directory_and_record(self, db, health, alerts_dir, tmp_path):
        """Deleting a baseline version should remove both disk data and lifecycle record."""
        baselines_dir = tmp_path / "baselines"
        version_dir = baselines_dir / "cam_01" / "default" / "v001"
        version_dir.mkdir(parents=True)
        (version_dir / "baseline_00000.png").write_bytes(b"img")
        (version_dir.parent / "current.txt").write_text("v001")

        config = ArgusConfig()
        config.storage.baselines_dir = baselines_dir

        app = create_app(
            database=db,
            camera_manager=MagicMock(),
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        app.state.baseline_lifecycle.register_version("cam_01", "default", "v001", image_count=1)
        client = TestClient(app)

        response = client.request(
            "DELETE",
            "/api/baseline/version",
            json={"camera_id": "cam_01", "version": "v001"},
        )

        assert response.status_code == 200
        assert not version_dir.exists()
        assert not (version_dir.parent / "current.txt").exists()
        assert app.state.baseline_lifecycle.get_version("cam_01", "default", "v001") is None

    def test_delete_baseline_version_rejects_active_version(self, db, health, alerts_dir, tmp_path):
        """Active baseline versions must be retired before deletion."""
        baselines_dir = tmp_path / "baselines"
        version_dir = baselines_dir / "cam_01" / "default" / "v001"
        version_dir.mkdir(parents=True)
        (version_dir / "baseline_00000.png").write_bytes(b"img")

        config = ArgusConfig()
        config.storage.baselines_dir = baselines_dir

        app = create_app(
            database=db,
            camera_manager=MagicMock(),
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        lifecycle = app.state.baseline_lifecycle
        lifecycle.register_version("cam_01", "default", "v001", image_count=1)
        lifecycle.verify("cam_01", "default", "v001", verified_by="tester")
        lifecycle.activate("cam_01", "default", "v001", user="tester")
        client = TestClient(app)

        response = client.request(
            "DELETE",
            "/api/baseline/version",
            json={"camera_id": "cam_01", "version": "v001"},
        )

        assert response.status_code == 400
        assert version_dir.exists()

    def test_delete_baseline_version_post_alias(self, db, health, alerts_dir, tmp_path):
        """Frontend-friendly POST alias should delete the same baseline version."""
        baselines_dir = tmp_path / "baselines"
        version_dir = baselines_dir / "cam_01" / "default" / "v001"
        version_dir.mkdir(parents=True)
        (version_dir / "baseline_00000.png").write_bytes(b"img")

        config = ArgusConfig()
        config.storage.baselines_dir = baselines_dir

        app = create_app(
            database=db,
            camera_manager=MagicMock(),
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        client = TestClient(app)

        response = client.post(
            "/api/baseline/version/delete",
            json={"camera_id": "cam_01", "version": "v001"},
        )

        assert response.status_code == 200
        assert not version_dir.exists()

class TestCameraJsonAPI:
    def test_cameras_json_returns_wrapped_list(self, db, health, alerts_dir):
        """Camera JSON API should return a stable object with a cameras field."""
        camera_manager = MagicMock()
        camera_manager.get_status.return_value = [
            SimpleNamespace(
                camera_id="cam_01",
                name="Camera 01",
                connected=True,
                running=True,
                stats=SimpleNamespace(
                    frames_captured=12,
                    frames_analyzed=8,
                    anomalies_detected=1,
                    alerts_emitted=1,
                    avg_latency_ms=15.2,
                ),
            ),
        ]

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.get("/api/cameras/json")

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 0
        data = body["data"]
        assert "cameras" in data
        assert data["cameras"][0]["camera_id"] == "cam_01"
        assert data["cameras"][0]["stats"]["frames_captured"] == 12

    def test_camera_detail_returns_config_and_runtime_sections(self, db, health, alerts_dir):
        """Camera detail API should aggregate config, runtime, detector and health data."""
        camera_config = CameraConfig(
            camera_id="cam_01",
            name="Camera 01",
            source="rtsp://example/stream",
            protocol="rtsp",
            fps_target=8,
            resolution=(1280, 720),
        )

        camera_manager = MagicMock()
        camera_manager._cameras = [camera_config]
        camera_manager.get_status.return_value = [
            SimpleNamespace(
                camera_id="cam_01",
                name="Camera 01",
                connected=True,
                running=True,
                stats=SimpleNamespace(
                    frames_captured=12,
                    frames_analyzed=8,
                    anomalies_detected=1,
                    alerts_emitted=1,
                    avg_latency_ms=15.2,
                ),
            ),
        ]
        camera_manager.get_runner_snapshot.return_value = SimpleNamespace(
            model_ref="patchcore",
            health_status="healthy",
            cusum_state="stable",
            lock_state=SimpleNamespace(value="unlocked"),
            last_heartbeat=123.4,
            version_tag="v1",
            degradation_state=SimpleNamespace(value="normal"),
            consecutive_failures=0,
        )
        camera_manager.get_detector_status.return_value = {
            "mode": "ssim",
            "model_path": None,
            "model_loaded": False,
            "threshold": 0.15,
            "ssim_calibration_progress": 0.5,
            "ssim_calibrated": False,
            "ssim_noise_floor": 0.01,
        }
        camera_manager.get_learning_progress.return_value = {"progress_pct": 10}
        camera_manager.get_pipeline_mode.return_value = "learning"
        camera_manager.is_anomaly_locked.return_value = False

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.get("/api/cameras/cam_01/detail/json")

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["camera_id"] == "cam_01"
        assert data["config"]["source"] == "rtsp://example/stream"
        assert data["runtime"]["pipeline_mode"] == "learning"
        assert data["runner"]["version_tag"] == "v1"
        assert data["detector"]["threshold"] == 0.15
        assert data["health"]["camera_id"] == "cam_01"


class TestCameraControlRoutes:
    def test_start_camera_offloads_to_thread(self, db, health, alerts_dir, monkeypatch):
        """Camera start route should offload blocking initialization from the event loop."""
        camera_manager = MagicMock()
        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            return True

        monkeypatch.setattr("argus.dashboard.routes.cameras.asyncio.to_thread", fake_to_thread)

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post("/api/cameras/cam_01/start")

        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == camera_manager.start_camera
        assert calls[0][1] == ("cam_01",)

    def test_stop_camera_offloads_to_thread(self, db, health, alerts_dir, monkeypatch):
        """Camera stop route should offload thread joins from the event loop."""
        camera_manager = MagicMock()
        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            return None

        monkeypatch.setattr("argus.dashboard.routes.cameras.asyncio.to_thread", fake_to_thread)

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post("/api/cameras/cam_01/stop")

        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == camera_manager.stop_camera
        assert calls[0][1] == ("cam_01",)

    def test_restart_camera_offloads_stop_and_start(self, db, health, alerts_dir, monkeypatch):
        """Camera restart route should offload stop/start instead of blocking the server."""
        camera_manager = MagicMock()
        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            if func == camera_manager.start_camera:
                return True
            return None

        monkeypatch.setattr("argus.dashboard.routes.config.asyncio.to_thread", fake_to_thread)

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post("/api/config/camera/cam_01/restart")

        assert response.status_code == 200
        assert len(calls) == 2
        assert calls[0][0] == camera_manager.stop_camera
        assert calls[1][0] == camera_manager.start_camera
        assert calls[0][1] == ("cam_01",)
        assert calls[1][1] == ("cam_01",)


class TestModelPublishRoutes:
    def test_activate_model_syncs_runtime(self, db, health, alerts_dir, tmp_path):
        """Activating a registered model should also hot-reload the running camera."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

        camera_manager = MagicMock()
        camera_manager.reload_model.return_value = True

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post(f"/api/models/{version_id}/activate")

        assert response.status_code == 200
        payload = response.json()
        assert payload["code"] == 0
        data = payload["data"]
        assert data["activated"] == version_id
        assert data["runtime_synced"] is True
        # _resolve_model_file resolves the directory to the actual model file
        camera_manager.reload_model.assert_called_once_with(
            "cam_01",
            str(model_dir / "model.xml"),
            version_tag=version_id,
        )

    def test_config_reload_model_uses_manager_reload(self, db, health, alerts_dir, tmp_path, monkeypatch):
        """Config reload-model route should use the shared runtime reload path."""
        monkeypatch.chdir(tmp_path)

        model_dir = tmp_path / "data" / "models" / "cam_01" / "default"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "model.xml"
        model_path.write_text("model")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

        camera_manager = MagicMock()
        camera_manager._pipelines = {"cam_01": MagicMock()}
        camera_manager.reload_model.return_value = True

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post(
            "/api/config/reload-model",
            json={"camera_id": "cam_01", "model_path": str(model_path)},
        )

        assert response.status_code == 200
        assert response.json()["code"] == 0
        assert response.json()["data"]["result"] == "ok"
        camera_manager.reload_model.assert_called_once_with(
            "cam_01",
            str(model_path.resolve()),
            version_tag=version_id,
        )

    def test_config_reload_model_resolves_checkpoint_to_exported_torch(
        self,
        db,
        health,
        alerts_dir,
        tmp_path,
        monkeypatch,
    ):
        """Config reload-model should resolve nested ckpt paths to deployable torch exports."""
        monkeypatch.chdir(tmp_path)

        model_dir = tmp_path / "data" / "models" / "cam_01" / "default"
        ckpt_dir = model_dir / "Patchcore" / "baseline" / "v0" / "weights" / "lightning"
        ckpt_dir.mkdir(parents=True)
        ckpt_path = ckpt_dir / "model.ckpt"
        ckpt_path.write_text("checkpoint")

        export_dir = tmp_path / "data" / "exports" / "cam_01" / "default" / "weights" / "torch"
        export_dir.mkdir(parents=True)
        exported_model = export_dir / "model.pt"
        exported_model.write_text("torch model")

        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

        camera_manager = MagicMock()
        camera_manager._pipelines = {"cam_01": MagicMock()}
        camera_manager.reload_model.return_value = True

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post(
            "/api/config/reload-model",
            json={"camera_id": "cam_01", "model_path": str(ckpt_path)},
        )

        assert response.status_code == 200
        assert response.json()["code"] == 0
        assert response.json()["data"]["result"] == "ok"
        camera_manager.reload_model.assert_called_once_with(
            "cam_01",
            str(exported_model),
            version_tag=version_id,
        )

    def test_config_reload_model_rejects_checkpoint_without_runtime_artifact(
        self,
        db,
        health,
        alerts_dir,
        tmp_path,
        monkeypatch,
    ):
        """Config reload-model should fail clearly when only a training checkpoint exists."""
        monkeypatch.chdir(tmp_path)

        model_dir = tmp_path / "data" / "models" / "cam_01" / "default"
        ckpt_dir = model_dir / "Patchcore" / "baseline" / "v0" / "weights" / "lightning"
        ckpt_dir.mkdir(parents=True)
        ckpt_path = ckpt_dir / "model.ckpt"
        ckpt_path.write_text("checkpoint")

        camera_manager = MagicMock()
        camera_manager._pipelines = {"cam_01": MagicMock()}

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post(
            "/api/config/reload-model",
            json={"camera_id": "cam_01", "model_path": str(ckpt_path)},
        )

        assert response.status_code == 404
        assert response.json()["msg"] == "未找到可部署的模型文件 (.xml/.pt)"
        camera_manager.reload_model.assert_not_called()

    def test_baseline_deploy_resolves_checkpoint_to_exported_torch(
        self,
        db,
        health,
        alerts_dir,
        tmp_path,
        monkeypatch,
    ):
        """Baseline deploy should prefer exported torch artifacts over nested lightning checkpoints."""
        monkeypatch.chdir(tmp_path)

        model_dir = tmp_path / "data" / "models" / "cam_01" / "default"
        ckpt_dir = model_dir / "Patchcore" / "baseline" / "v0" / "weights" / "lightning"
        ckpt_dir.mkdir(parents=True)
        ckpt_path = ckpt_dir / "model.ckpt"
        ckpt_path.write_text("checkpoint")

        export_dir = tmp_path / "data" / "exports" / "cam_01" / "default" / "weights" / "torch"
        export_dir.mkdir(parents=True)
        exported_model = export_dir / "model.pt"
        exported_model.write_text("torch model")

        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

        camera_manager = MagicMock()
        camera_manager._pipelines = {"cam_01": MagicMock()}
        camera_manager.reload_model.return_value = True

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post(
            "/api/baseline/deploy",
            json={"camera_id": "cam_01", "model_path": str(ckpt_path)},
        )

        assert response.status_code == 200
        assert response.json()["code"] == 0
        assert response.json()["data"]["model_version_id"] == version_id
        camera_manager.reload_model.assert_called_once_with(
            "cam_01",
            str(exported_model),
            version_tag=version_id,
        )

    def test_promote_production_syncs_runtime(self, db, health, alerts_dir, tmp_path):
        """Promoting a canary model to production should sync the runtime model."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

        with db.get_session() as session:
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

        camera_manager = MagicMock()
        camera_manager.reload_model.return_value = True

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post(
            f"/api/models/{version_id}/promote",
            json={"target_stage": "production", "triggered_by": "operator"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["code"] == 0
        data = payload["data"]
        assert data["model"]["stage"] == "production"
        assert data["runtime_synced"] is True
        camera_manager.reload_model.assert_called_once_with(
            "cam_01",
            str(model_dir / "model.xml"),
            version_tag=version_id,
        )


class TestTrainingRegistration:
    def test_train_model_task_registers_candidate_model(self, monkeypatch, tmp_path):
        """Manual training task should register its output into the model registry."""
        baselines_dir = tmp_path / "baselines"
        current_dir = baselines_dir / "cam_01" / "default" / "v001"
        current_dir.mkdir(parents=True)
        (current_dir.parent / "current.txt").write_text("v001")
        (current_dir / "img.png").write_bytes(b"img")

        models_dir = tmp_path / "models"
        output_dir = models_dir / "cam_01" / "default"
        output_dir.mkdir(parents=True)
        (output_dir / "model.xml").write_text("model")

        db = Database(database_url=f"sqlite:///{tmp_path / 'train.db'}")
        db.initialize()

        from argus.anomaly.trainer import QualityReport, TrainingResult, TrainingStatus

        def fake_train(self, **kwargs):
            return TrainingResult(
                status=TrainingStatus.COMPLETE,
                model_path=str(output_dir),
                duration_seconds=1.0,
                image_count=30,
                train_count=24,
                val_count=6,
                pre_validation={"passed": True},
                val_stats={"mean": 0.1, "std": 0.01, "max": 0.2, "p95": 0.18},
                quality_report=QualityReport(grade="A"),
                threshold_recommended=0.42,
                output_validation={"checkpoint_valid": True, "export_valid": True, "smoke_test_passed": True},
            )

        monkeypatch.setattr("argus.anomaly.trainer.ModelTrainer.train", fake_train)

        messages = []

        result = _train_model_task(
            lambda progress, message: messages.append((progress, message)),
            baselines_dir=str(baselines_dir),
            models_dir=str(models_dir),
            camera_id="cam_01",
            model_type="patchcore",
            export_format="openvino",
            quantization="fp16",
            database_url=f"sqlite:///{tmp_path / 'train.db'}",
        )

        registry = ModelRegistry(session_factory=db.get_session)
        models = registry.list_models(camera_id="cam_01")

        assert result["model_version_id"] is not None
        assert len(models) == 1
        assert models[0].stage == "candidate"
        assert messages[-1][0] == 100
        db.close()


class TestCameraManagerModelRouting:
    def test_prefers_active_registry_model_on_startup(self, tmp_path):
        """CameraManager should restore the active registry model for restarted cameras."""
        db = Database(database_url=f"sqlite:///{tmp_path / 'camera.db'}")
        db.initialize()

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
        registry.activate(version_id, allow_bypass=True)

        cam_config = CameraConfig(
            camera_id="cam_01",
            name="Camera 01",
            source="0",
            protocol="file",
        )

        manager = CameraManager(
            cameras=[cam_config],
            alert_config=AlertConfig(),
            database=db,
        )

        resolved = manager._resolve_model_path(cam_config)
        assert resolved == model_dir / "model.xml"
        assert manager._model_version_id(cam_config, resolved) == version_id
        db.close()

    def test_reload_model_updates_pipeline_version_id(self):
        """Hot reload should update both runner snapshot tag and pipeline record tag."""
        manager = CameraManager(cameras=[], alert_config=AlertConfig())
        pipeline = MagicMock()
        pipeline.reload_anomaly_model.return_value = True
        runner = MagicMock()
        manager._pipelines["cam_01"] = pipeline
        manager._runners["cam_01"] = runner
        manager._notify_status_change = MagicMock()

        ok = manager.reload_model("cam_01", "data/models/cam_01/model.xml", version_tag="v2")

        assert ok is True
        runner.set_version_tag.assert_called_once_with("v2")
        pipeline.set_model_version_id.assert_called_once_with("v2")

    def test_reload_model_failure_broadcasts_activation_failed(self):
        """Reload failure should emit a models-topic event; detector keeps old engine."""
        events: list[tuple[str, dict]] = []
        manager = CameraManager(
            cameras=[],
            alert_config=AlertConfig(),
            on_status_change=lambda topic, data: events.append((topic, data)),
        )
        pipeline = MagicMock()
        pipeline.reload_anomaly_model.return_value = False
        pipeline._model_version_id = "v1-old"
        manager._pipelines["cam_01"] = pipeline

        ok = manager.reload_model("cam_01", "data/models/cam_01/model.xml", version_tag="v2")

        assert ok is False
        assert len(events) == 1
        topic, data = events[0]
        assert topic == "models"
        assert data["event"] == "model.activation_failed"
        assert data["camera_id"] == "cam_01"
        assert data["attempted_version"] == "v2"
        assert data["current_version"] == "v1-old"
        assert data["reason"] == "hot_reload_failed"


class TestSchedulerWiring:
    def test_register_training_job_processing_wires_executor(self, monkeypatch):
        """Main scheduler setup should register queued training job processing."""
        scheduler = MagicMock()
        database = MagicMock()
        database.get_session = MagicMock()
        config = ArgusConfig()

        baseline_manager = MagicMock()
        model_trainer = MagicMock()
        backbone_trainer_cls = MagicMock(return_value=MagicMock())
        registry_cls = MagicMock(return_value=MagicMock())
        executor_cls = MagicMock(return_value=MagicMock())
        register_task = MagicMock()

        monkeypatch.setattr("argus.anomaly.backbone_trainer.BackboneTrainer", backbone_trainer_cls)
        monkeypatch.setattr("argus.storage.model_registry.ModelRegistry", registry_cls)
        monkeypatch.setattr("argus.anomaly.job_executor.TrainingJobExecutor", executor_cls)
        monkeypatch.setattr("argus.__main__.create_job_processing_task", register_task)

        _register_training_job_processing(
            scheduler,
            config=config,
            database=database,
            baseline_manager=baseline_manager,
            model_trainer=model_trainer,
        )

        backbone_trainer_cls.assert_called_once()
        executor_cls.assert_called_once()
        # Ensure the shared model_trainer instance was forwarded to the executor
        _, kwargs = executor_cls.call_args
        assert kwargs["trainer"] is model_trainer
        register_task.assert_called_once()


class TestAlertsAPI:
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
        """JSON API should return structured envelope with alerts list."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)

        response = client.get("/api/alerts/json")
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 0
        alerts = body["data"]["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "ALT-001"


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
        """Should return None for missing files.

        Note: ultralytics monkey-patches cv2.imread to raise FileNotFoundError
        instead of returning None, so we mock it to restore standard behavior.
        """
        from unittest.mock import patch

        with patch("cv2.imread", return_value=None):
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


class TestClassifierConfigRoutes:
    """GET/PUT routes behind the System → 分类器 panel (stage 2.2 + 2.3)."""

    @pytest.fixture
    def classifier_client(self, db, health, alerts_dir):
        from argus.config.schema import ClassifierConfig
        config = ArgusConfig()
        config.classifier = ClassifierConfig(
            enabled=False,
            model_name="yolov8s-worldv2.pt",
            vocabulary=["wrench", "bolt", "bird"],
            high_risk_labels=["wrench", "bolt"],
            low_risk_labels=["bird"],
            suppress_labels=[],
            min_anomaly_score_to_classify=0.5,
        )
        camera_manager = MagicMock()
        camera_manager.get_status.return_value = []
        camera_manager.update_classifier_vocabulary.return_value = 0
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        # Stub auth.require_role so the PUT route doesn't 403 in tests where
        # we aren't carrying a session cookie.
        from argus.dashboard import auth as dashboard_auth
        _orig_require_role = dashboard_auth.require_role
        dashboard_auth.require_role = lambda request, *roles: True
        try:
            yield TestClient(app), config, camera_manager
        finally:
            dashboard_auth.require_role = _orig_require_role

    def test_get_classifier_config_returns_full_payload(self, classifier_client):
        client, _cfg, _cm = classifier_client
        res = client.get("/api/config/classifier")
        assert res.status_code == 200
        body = res.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["enabled"] is False
        assert data["model_name"] == "yolov8s-worldv2.pt"
        assert data["vocabulary"] == ["wrench", "bolt", "bird"]
        assert data["high_risk_labels"] == ["wrench", "bolt"]
        assert data["low_risk_labels"] == ["bird"]
        # Runtime counters default to zero when no pipelines are alive
        assert data["runtime"] == {
            "total_pipelines": 0,
            "pipelines_attached": 0,
            "pipelines_loaded": 0,
        }

    def test_put_vocabulary_rejects_empty_list(self, classifier_client):
        client, _cfg, _cm = classifier_client
        res = client.put(
            "/api/config/classifier/vocabulary",
            json={"vocabulary": ["  ", ""]},
        )
        assert res.status_code == 400
        assert "词表" in res.json()["msg"]

    def test_put_vocabulary_rejects_high_risk_not_in_vocab(self, classifier_client):
        client, _cfg, _cm = classifier_client
        res = client.put(
            "/api/config/classifier/vocabulary",
            json={
                "vocabulary": ["wrench"],
                "high_risk_labels": ["hammer"],
            },
        )
        assert res.status_code == 400
        assert "hammer" in res.json()["msg"]

    def test_put_vocabulary_dedupes_and_trims(self, classifier_client):
        client, cfg, _cm = classifier_client
        res = client.put(
            "/api/config/classifier/vocabulary",
            json={
                "vocabulary": [" wrench ", "wrench", "bolt", "", "bolt"],
                "high_risk_labels": ["wrench"],
            },
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["vocabulary"] == ["wrench", "bolt"]
        # Config mutation happens in-place on the injected ClassifierConfig
        assert cfg.classifier.vocabulary == ["wrench", "bolt"]
        assert cfg.classifier.high_risk_labels == ["wrench"]

    def test_put_vocabulary_pushes_to_pipelines(self, classifier_client):
        client, cfg, cm = classifier_client
        cm.update_classifier_vocabulary.return_value = 3
        res = client.put(
            "/api/config/classifier/vocabulary",
            json={
                "vocabulary": ["wrench", "hammer"],
                "high_risk_labels": ["hammer"],
                "low_risk_labels": ["wrench"],
            },
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["pipelines_updated"] == 3
        cm.update_classifier_vocabulary.assert_called_once_with(["wrench", "hammer"])
        assert cfg.classifier.vocabulary == ["wrench", "hammer"]
        assert cfg.classifier.high_risk_labels == ["hammer"]
        assert cfg.classifier.low_risk_labels == ["wrench"]


class TestSegmenterConfigRoutes:
    """GET /api/config/segmenter behind the System → 分割器 panel (stage 2.5)."""

    @pytest.fixture
    def segmenter_client(self, db, health, alerts_dir):
        from argus.config.schema import SegmenterConfig
        config = ArgusConfig()
        config.segmenter = SegmenterConfig(
            enabled=False,
            model_size="small",
            max_points=5,
            min_anomaly_score=0.7,
            min_mask_area_px=100,
            timeout_seconds=10.0,
        )
        camera_manager = MagicMock()
        camera_manager.get_status.return_value = []
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        return TestClient(app), config, camera_manager

    def test_get_segmenter_config_returns_full_payload(self, segmenter_client):
        client, _cfg, _cm = segmenter_client
        res = client.get("/api/config/segmenter")
        assert res.status_code == 200
        body = res.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["enabled"] is False
        assert data["model_size"] == "small"
        assert data["max_points"] == 5
        assert data["min_anomaly_score"] == 0.7
        assert data["min_mask_area_px"] == 100
        assert data["timeout_seconds"] == 10.0
        assert data["runtime"] == {
            "total_pipelines": 0,
            "pipelines_attached": 0,
            "pipelines_loaded": 0,
        }

    def test_modules_endpoint_exposes_segmenter_enabled(self, segmenter_client):
        """ModuleTogglePanel reads /api/config/modules — segmenter.enabled must
        be in the dict so the toggle on the 功能模块 tab can reflect state."""
        client, _cfg, _cm = segmenter_client
        res = client.get("/api/config/modules")
        assert res.status_code == 200
        data = res.json()["data"]
        assert "segmenter.enabled" in data
        assert data["segmenter.enabled"] is False

    def test_get_segmenter_config_counts_attached_pipelines(self, segmenter_client):
        """When a pipeline has _segmenter set and _loaded=True, counters reflect it."""
        client, cfg, cm = segmenter_client
        cfg.segmenter.enabled = True

        cam_status = SimpleNamespace(camera_id="cam_a")
        fake_segmenter = MagicMock()
        fake_segmenter._loaded = True
        fake_pipeline = SimpleNamespace(_segmenter=fake_segmenter)

        cm.get_status.return_value = [cam_status]
        cm.get_pipeline.return_value = fake_pipeline

        res = client.get("/api/config/segmenter")
        data = res.json()["data"]
        assert data["enabled"] is True
        assert data["runtime"]["total_pipelines"] == 1
        assert data["runtime"]["pipelines_attached"] == 1
        assert data["runtime"]["pipelines_loaded"] == 1


class TestCrossCameraConfigRoutes:
    """GET /api/config/cross-camera behind the System → 跨相机 panel (stage 2.8)."""

    @pytest.fixture
    def cross_camera_client(self, db, health, alerts_dir):
        import numpy as np
        from argus.config.schema import CrossCameraConfig, CameraOverlapConfig
        config = ArgusConfig()
        config.cross_camera = CrossCameraConfig(
            enabled=False,
            corroboration_threshold=0.3,
            max_age_seconds=5.0,
            uncorroborated_severity_downgrade=1,
            overlap_pairs=[
                CameraOverlapConfig(
                    camera_a="cam_01",
                    camera_b="cam_02",
                    homography=np.eye(3),
                ),
            ],
        )
        camera_manager = MagicMock()
        camera_manager.get_status.return_value = []
        # The correlator attribute on CameraManager is `_correlator`
        camera_manager._correlator = None
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        return TestClient(app), config, camera_manager

    def test_get_cross_camera_config_returns_full_payload(self, cross_camera_client):
        client, _cfg, _cm = cross_camera_client
        res = client.get("/api/config/cross-camera")
        assert res.status_code == 200
        body = res.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["enabled"] is False
        assert data["corroboration_threshold"] == 0.3
        assert data["max_age_seconds"] == 5.0
        assert data["uncorroborated_severity_downgrade"] == 1
        assert len(data["overlap_pairs"]) == 1
        pair = data["overlap_pairs"][0]
        assert pair["camera_a"] == "cam_01"
        assert pair["camera_b"] == "cam_02"
        # Identity 3x3 matrix
        assert pair["homography"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert data["runtime"]["total_pipelines"] == 0
        assert data["runtime"]["correlator_present"] is False

    def test_modules_endpoint_exposes_cross_camera_enabled(self, cross_camera_client):
        client, _cfg, _cm = cross_camera_client
        res = client.get("/api/config/modules")
        assert res.status_code == 200
        data = res.json()["data"]
        assert "cross_camera.enabled" in data
        assert data["cross_camera.enabled"] is False

    def test_get_cross_camera_config_detects_correlator_presence(self, cross_camera_client):
        """When a correlator is actually attached to the manager, runtime reports it."""
        client, cfg, cm = cross_camera_client
        cfg.cross_camera.enabled = True
        cm._correlator = MagicMock()
        cam_status = SimpleNamespace(camera_id="cam_a")
        cm.get_status.return_value = [cam_status]
        cm.get_pipeline.return_value = SimpleNamespace()
        res = client.get("/api/config/cross-camera")
        data = res.json()["data"]
        assert data["enabled"] is True
        assert data["runtime"]["total_pipelines"] == 1
        assert data["runtime"]["correlator_present"] is True


class TestSegmenterParamsRoute:
    """PUT /api/config/segmenter/params (stage 2.6)."""

    @pytest.fixture
    def client_with_cfg(self, db, health, alerts_dir):
        from argus.config.schema import SegmenterConfig
        config = ArgusConfig()
        config.segmenter = SegmenterConfig(
            enabled=True,
            model_size="small",
            max_points=5,
            min_anomaly_score=0.7,
            min_mask_area_px=100,
            timeout_seconds=10.0,
        )
        camera_manager = MagicMock()
        camera_manager.get_status.return_value = []
        camera_manager.update_segmenter_params.return_value = 0
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        from argus.dashboard import auth as dashboard_auth
        _orig_require_role = dashboard_auth.require_role
        dashboard_auth.require_role = lambda request, *roles: True
        try:
            yield TestClient(app), config, camera_manager
        finally:
            dashboard_auth.require_role = _orig_require_role

    def test_put_updates_config_in_place(self, client_with_cfg):
        client, cfg, _cm = client_with_cfg
        res = client.put(
            "/api/config/segmenter/params",
            json={"max_points": 8, "min_anomaly_score": 0.6},
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["max_points"] == 8
        assert data["min_anomaly_score"] == 0.6
        # untouched fields keep their defaults
        assert data["min_mask_area_px"] == 100
        assert cfg.segmenter.max_points == 8
        assert cfg.segmenter.min_anomaly_score == 0.6

    def test_put_pushes_to_camera_manager(self, client_with_cfg):
        client, _cfg, cm = client_with_cfg
        cm.update_segmenter_params.return_value = 2
        res = client.put(
            "/api/config/segmenter/params",
            json={
                "max_points": 6,
                "min_anomaly_score": 0.65,
                "min_mask_area_px": 200,
                "timeout_seconds": 15.0,
            },
        )
        assert res.status_code == 200
        assert res.json()["data"]["pipelines_updated"] == 2
        cm.update_segmenter_params.assert_called_once_with(
            max_points=6,
            min_anomaly_score=0.65,
            min_mask_area_px=200,
            timeout_seconds=15.0,
        )

    @pytest.mark.parametrize("payload,expected_msg_frag", [
        ({"max_points": 0}, "max_points"),
        ({"max_points": 99}, "max_points"),
        ({"min_anomaly_score": -0.1}, "min_anomaly_score"),
        ({"min_anomaly_score": 1.5}, "min_anomaly_score"),
        ({"min_mask_area_px": -5}, "min_mask_area_px"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": 500}, "timeout_seconds"),
    ])
    def test_put_rejects_out_of_range(self, client_with_cfg, payload, expected_msg_frag):
        client, _cfg, _cm = client_with_cfg
        res = client.put("/api/config/segmenter/params", json=payload)
        assert res.status_code == 400
        assert expected_msg_frag in res.json()["msg"]

    def test_put_empty_body_is_noop(self, client_with_cfg):
        """All fields optional — empty body should succeed and change nothing."""
        client, cfg, _cm = client_with_cfg
        before = cfg.segmenter.max_points
        res = client.put("/api/config/segmenter/params", json={})
        assert res.status_code == 200
        assert cfg.segmenter.max_points == before


class TestCameraManagerSegmenterBroadcast:
    """CameraManager.update_segmenter_params hot-swap fanout (stage 2.6)."""

    def test_updates_attached_pipelines_only(self):
        manager = CameraManager.__new__(CameraManager)
        manager._pipelines = {}

        seg_a = MagicMock()
        seg_b = MagicMock()
        pipe_a = SimpleNamespace(_segmenter=seg_a, _segmenter_max_points=5, _segmenter_min_score=0.7)
        pipe_b = SimpleNamespace(_segmenter=seg_b, _segmenter_max_points=5, _segmenter_min_score=0.7)
        pipe_c = SimpleNamespace(_segmenter=None)  # no segmenter attached

        manager._pipelines = {"a": pipe_a, "b": pipe_b, "c": pipe_c}

        updated = manager.update_segmenter_params(
            max_points=8,
            min_anomaly_score=0.5,
            min_mask_area_px=150,
            timeout_seconds=12.0,
        )

        assert updated == 2
        assert pipe_a._segmenter_max_points == 8
        assert pipe_a._segmenter_min_score == 0.5
        assert pipe_b._segmenter_max_points == 8
        assert pipe_b._segmenter_min_score == 0.5
        seg_a.update_runtime_params.assert_called_once_with(
            min_mask_area_px=150,
            timeout_seconds=12.0,
        )
        seg_b.update_runtime_params.assert_called_once_with(
            min_mask_area_px=150,
            timeout_seconds=12.0,
        )

    def test_swallows_per_pipeline_errors(self):
        manager = CameraManager.__new__(CameraManager)
        manager._pipelines = {}

        ok = MagicMock()
        broken = MagicMock()
        broken.update_runtime_params.side_effect = RuntimeError("boom")
        manager._pipelines = {
            "ok": SimpleNamespace(_segmenter=ok, _segmenter_max_points=5, _segmenter_min_score=0.7),
            "broken": SimpleNamespace(_segmenter=broken, _segmenter_max_points=5, _segmenter_min_score=0.7),
        }
        updated = manager.update_segmenter_params(timeout_seconds=20.0)
        assert updated == 1
        ok.update_runtime_params.assert_called_once()


class TestCameraManagerClassifierBroadcast:
    """CameraManager.update_classifier_vocabulary hot-swap fanout (stage 2.3)."""

    def test_update_hits_every_pipeline_with_classifier(self):
        manager = CameraManager.__new__(CameraManager)
        manager._pipelines = {}

        classifier_a = MagicMock()
        classifier_b = MagicMock()
        pipeline_a = SimpleNamespace(_classifier=classifier_a)
        pipeline_b = SimpleNamespace(_classifier=classifier_b)
        pipeline_c = SimpleNamespace(_classifier=None)  # no classifier attached

        manager._pipelines = {"a": pipeline_a, "b": pipeline_b, "c": pipeline_c}

        updated = manager.update_classifier_vocabulary(["x", "y"])

        assert updated == 2
        classifier_a.update_vocabulary.assert_called_once_with(["x", "y"])
        classifier_b.update_vocabulary.assert_called_once_with(["x", "y"])

    def test_update_swallows_per_pipeline_errors(self):
        manager = CameraManager.__new__(CameraManager)
        manager._pipelines = {}

        ok = MagicMock()
        broken = MagicMock()
        broken.update_vocabulary.side_effect = RuntimeError("boom")

        manager._pipelines = {
            "ok": SimpleNamespace(_classifier=ok),
            "broken": SimpleNamespace(_classifier=broken),
        }

        # Broken pipeline must not take down the whole operation.
        updated = manager.update_classifier_vocabulary(["x"])
        assert updated == 1  # only the healthy one counted
        ok.update_vocabulary.assert_called_once_with(["x"])


class TestCrossCameraConfigPut:
    """PUT /api/config/cross-camera/pairs (stage 2.9)."""

    @pytest.fixture
    def cc_client(self, db, health, alerts_dir):
        from argus.config.schema import CrossCameraConfig, CameraOverlapConfig
        config = ArgusConfig()
        config.cross_camera = CrossCameraConfig(
            enabled=True,
            corroboration_threshold=0.3,
            max_age_seconds=5.0,
            uncorroborated_severity_downgrade=1,
            overlap_pairs=[
                CameraOverlapConfig(
                    camera_a="cam_01",
                    camera_b="cam_02",
                    homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                ),
            ],
        )
        camera_manager = MagicMock()
        camera_manager.get_status.return_value = []
        camera_manager._correlator = MagicMock()
        camera_manager.update_cross_camera_pairs.return_value = True
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        from argus.dashboard import auth as dashboard_auth
        _orig_require_role = dashboard_auth.require_role
        dashboard_auth.require_role = lambda request, *roles: True
        try:
            yield TestClient(app), config, camera_manager
        finally:
            dashboard_auth.require_role = _orig_require_role

    def test_put_updates_scalar_params(self, cc_client):
        client, cfg, _cm = cc_client
        res = client.put(
            "/api/config/cross-camera/pairs",
            json={"corroboration_threshold": 0.5, "max_age_seconds": 10.0},
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["corroboration_threshold"] == 0.5
        assert data["max_age_seconds"] == 10.0
        assert cfg.cross_camera.corroboration_threshold == 0.5
        assert cfg.cross_camera.max_age_seconds == 10.0

    def test_put_replaces_overlap_pairs(self, cc_client):
        client, cfg, _cm = cc_client
        new_pairs = [
            {
                "camera_a": "cam_03",
                "camera_b": "cam_04",
                "homography": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            },
        ]
        res = client.put(
            "/api/config/cross-camera/pairs",
            json={"overlap_pairs": new_pairs},
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert len(data["overlap_pairs"]) == 1
        assert data["overlap_pairs"][0]["camera_a"] == "cam_03"
        assert len(cfg.cross_camera.overlap_pairs) == 1

    def test_put_pushes_to_camera_manager(self, cc_client):
        client, _cfg, cm = cc_client
        client.put(
            "/api/config/cross-camera/pairs",
            json={"overlap_pairs": [
                {"camera_a": "a", "camera_b": "b",
                 "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            ]},
        )
        cm.update_cross_camera_pairs.assert_called_once()

    def test_put_rejects_non_3x3_matrix(self, cc_client):
        client, _cfg, _cm = cc_client
        res = client.put(
            "/api/config/cross-camera/pairs",
            json={"overlap_pairs": [
                {"camera_a": "a", "camera_b": "b",
                 "homography": [[1, 0], [0, 1]]},
            ]},
        )
        assert res.status_code == 400
        assert "3" in res.json()["msg"]

    def test_put_rejects_same_camera(self, cc_client):
        client, _cfg, _cm = cc_client
        res = client.put(
            "/api/config/cross-camera/pairs",
            json={"overlap_pairs": [
                {"camera_a": "cam_01", "camera_b": "cam_01",
                 "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            ]},
        )
        assert res.status_code == 400
        assert "相同" in res.json()["msg"]

    def test_put_rejects_duplicate_pair(self, cc_client):
        client, _cfg, _cm = cc_client
        res = client.put(
            "/api/config/cross-camera/pairs",
            json={"overlap_pairs": [
                {"camera_a": "a", "camera_b": "b",
                 "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                {"camera_a": "b", "camera_b": "a",
                 "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            ]},
        )
        assert res.status_code == 400
        assert "重复" in res.json()["msg"]

    def test_put_empty_body_is_noop(self, cc_client):
        client, cfg, cm = cc_client
        before_threshold = cfg.cross_camera.corroboration_threshold
        before_pairs = len(cfg.cross_camera.overlap_pairs)
        res = client.put("/api/config/cross-camera/pairs", json={})
        assert res.status_code == 200
        assert cfg.cross_camera.corroboration_threshold == before_threshold
        assert len(cfg.cross_camera.overlap_pairs) == before_pairs
        cm.update_cross_camera_pairs.assert_not_called()
