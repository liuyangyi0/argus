"""Tests for the dashboard API."""

from datetime import datetime, timezone
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette import requests as starlette_requests

from argus.runtime.training_job_wiring import (
    register_training_job_processing as _register_training_job_processing,
)
from argus.capture.manager import CameraManager
from argus.config.loader import load_config, save_config
from argus.config.schema import AlertConfig, ArgusConfig, AuthConfig, CameraConfig
from argus.core.health import HealthMonitor
from argus.dashboard.auth import create_session_token, hash_password
from argus.dashboard.app import _mount_vue_spa, create_app
from argus.dashboard.routes.baseline import _train_model_task
from argus.dashboard.routes.alerts import _generate_composite, _is_safe_path
from argus.streaming.go2rtc_manager import usb_to_go2rtc_source
from argus.storage.model_registry import ModelRegistry
from argus.storage.model_release_service import (
    RuntimeReleaseCommit,
    RuntimeReleaseIdentity,
    RuntimeReleasePreparation,
)
from argus.storage.database import Database
from argus.storage.models import ModelVersionEvent
from argus.storage.release_pipeline import ReleasePipeline


def _release_model_fixture(
    db: Database,
    registry: ModelRegistry,
    version_id: str,
    model_path: Path,
    release_evidence_factory,
    *,
    camera_id: str = "cam_01",
) -> None:
    pipeline = ReleasePipeline(db.get_session, min_shadow_days=0, min_canary_days=0)
    pipeline.transition(
        model_version_id=version_id,
        target_stage="shadow",
        triggered_by="test-fixture",
    )
    pipeline.transition(
        model_version_id=version_id,
        target_stage="canary",
        triggered_by="test-fixture",
        canary_camera_id=camera_id,
    )
    pipeline.transition(
        model_version_id=version_id,
        target_stage="production",
        triggered_by="test-fixture",
        release_evidence=release_evidence_factory(
            model_path=model_path,
            model_version_id=version_id,
            registry_model_hash=registry.get_by_version_id(version_id).model_hash,
        ),
    )


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
def capture_compute_manager(tmp_path):
    from argus.runtime.compute_ownership import ComputeOwnership
    manager = CameraManager([], AlertConfig())
    ownership = ComputeOwnership(tmp_path / "capture-control.lock")
    manager.enable_compute_ownership(ownership)
    yield manager
    # The task manager in these control-plane tests only records submit calls;
    # no camera/model/background task was started.
    assert ownership.release(manager)
    manager._inference_executor.shutdown(wait=False)
    manager._encode_executor.shutdown(wait=False)


def _two_phase_release_runtime() -> SimpleNamespace:
    """Minimal runtime double that proves the release identity it prepared."""

    def prepare_model_release(plan) -> RuntimeReleasePreparation:
        class Transaction:
            def commit(self) -> RuntimeReleaseCommit:
                return RuntimeReleaseCommit(
                    True,
                    state={
                        "camera_id": plan.camera_id,
                        "primary_model_version": (
                            plan.model_version_id
                            if plan.target_stage in {"canary", "production"}
                            else None
                        ),
                        "shadow_model_version": (
                            plan.model_version_id
                            if plan.target_stage == "shadow"
                            else None
                        ),
                        "errors": [],
                    },
                )

            def abort(self) -> RuntimeReleaseCommit:
                return RuntimeReleaseCommit(True, state={"restored": True})

            def finalize(self) -> RuntimeReleaseCommit:
                return RuntimeReleaseCommit(True, state={"released": True})

        return RuntimeReleasePreparation(
            True,
            identity=RuntimeReleaseIdentity(
                camera_id=plan.camera_id,
                model_version_id=plan.model_version_id,
                target_stage=plan.target_stage,
                runtime_artifact_sha256=plan.runtime_artifact_sha256,
            ),
            transaction=Transaction(),
        )

    return SimpleNamespace(prepare_model_release=prepare_model_release)


class TestDashboardPages:
    def test_installed_app_serves_explicit_frontend_directory(self, tmp_path, db):
        dist = tmp_path / "installed-frontend"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text('<div id="packaged-ui"></div>', encoding="utf-8")
        (dist / "assets/app.js").write_text("/* packaged asset */", encoding="utf-8")
        config = ArgusConfig()
        config.dashboard.go2rtc_enabled = False
        config.dashboard.frontend_dist_dir = dist
        client = TestClient(create_app(database=db, config=config))
        assert 'id="packaged-ui"' in client.get("/models/registry").text
        assert client.get("/assets/app.js").text == "/* packaged asset */"
        assert client.get("/api/not-a-route").status_code == 404

    def test_explicit_missing_frontend_is_a_configuration_error(self, tmp_path):
        config = ArgusConfig()
        config.dashboard.go2rtc_enabled = False
        config.dashboard.frontend_dist_dir = tmp_path / "missing-dist"
        with pytest.raises(ValueError, match="frontend_dist_dir"):
            create_app(config=config)

    def test_core_business_routes_fallback_to_vue_spa(self, tmp_path):
        """Every dashboard route in the operator loop must serve the SPA shell."""
        vue_dist = tmp_path / "dist"
        assets = vue_dist / "assets"
        assets.mkdir(parents=True)
        (vue_dist / "index.html").write_text(
            '<!doctype html><div id="app"></div><script src="/assets/app.js"></script>',
            encoding="utf-8",
        )
        (assets / "app.js").write_text("console.log('argus')", encoding="utf-8")

        app = FastAPI()
        _mount_vue_spa(app, vue_dist)
        client = TestClient(app)

        for path in [
            "/cameras",
            "/alerts",
            "/replay/ALT-smoke",
            "/replay/ALT-smoke/storyboard",
            "/models/baseline",
            "/models/training",
            "/models/registry",
            "/system/overview",
            "/system/config",
            "/system/degradation",
            "/reports",
        ]:
            response = client.get(path)
            assert response.status_code == 200, path
            assert '<div id="app">' in response.text, path
            assert "text/html" in response.headers["content-type"], path

    def test_spa_fallback_does_not_swallow_api_or_ws_misses(self, tmp_path):
        """Unknown API/WS paths should 404 instead of returning index.html."""
        vue_dist = tmp_path / "dist"
        vue_dist.mkdir()
        (vue_dist / "index.html").write_text('<div id="app"></div>', encoding="utf-8")

        app = FastAPI()
        _mount_vue_spa(app, vue_dist)
        client = TestClient(app)

        for path in ["/api/not-a-route", "/ws/not-a-route"]:
            response = client.get(path)
            assert response.status_code == 404, path
            assert "<div id=\"app\">" not in response.text, path

    @pytest.mark.parametrize(
        "traversal_path",
        [
            "/%2e%2e/sibling-secret.txt",
            "/..%2fsibling-secret.txt",
        ],
    )
    def test_spa_fallback_rejects_paths_outside_dist(
        self,
        tmp_path,
        traversal_path,
    ):
        vue_dist = tmp_path / "dist"
        vue_dist.mkdir()
        (vue_dist / "index.html").write_text('<div id="app"></div>', encoding="utf-8")
        secret = "must-not-be-served-by-the-spa"
        (tmp_path / "sibling-secret.txt").write_text(secret, encoding="utf-8")

        app = FastAPI()
        _mount_vue_spa(app, vue_dist)
        client = TestClient(app)

        response = client.get(traversal_path)

        assert secret not in response.text
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_spa_fallback_rejects_plain_parent_segment(self, tmp_path):
        vue_dist = tmp_path / "dist"
        vue_dist.mkdir()
        (vue_dist / "index.html").write_text('<div id="app"></div>', encoding="utf-8")
        secret = "must-not-be-served-by-the-spa"
        (tmp_path / "sibling-secret.txt").write_text(secret, encoding="utf-8")

        app = FastAPI()
        _mount_vue_spa(app, vue_dist)
        spa_route = next(
            route
            for route in app.routes
            if getattr(route, "path", None) == "/{full_path:path}"
        )

        with pytest.raises(HTTPException) as exc_info:
            await spa_route.endpoint(MagicMock(), "../sibling-secret.txt")

        assert exc_info.value.status_code == 404

    def test_vue_spa_mount_tolerates_missing_assets_dir(self, tmp_path):
        """A partial web/dist should not break backend startup."""
        vue_dist = tmp_path / "dist"
        vue_dist.mkdir()
        (vue_dist / "index.html").write_text('<div id="app"></div>')

        app = FastAPI()
        _mount_vue_spa(app, vue_dist)
        response = TestClient(app).get("/system")

        assert response.status_code == 200
        assert '<div id="app">' in response.text

    def test_index_page_loads(self, client):
        """Root serves the built SPA, or an explicit JSON 404 before build."""
        response = client.get("/")
        if response.status_code == 404:
            assert response.json()["code"] == 40400
        else:
            assert response.status_code == 200
            assert '<div id="app">' in response.text or "<div id=\"app\">" in response.text
            assert "text/html" in response.headers["content-type"]

    def test_cameras_page_loads(self, client):
        """Cameras serves the built SPA, or an explicit JSON 404 before build."""
        response = client.get("/cameras")
        if response.status_code == 404:
            assert response.json()["code"] == 40400
        else:
            assert response.status_code == 200
            assert '<div id="app">' in response.text or "<div id=\"app\">" in response.text

    def test_alerts_page_loads(self, client):
        """Alerts route does not masquerade as HTML without a built SPA."""
        response = client.get("/alerts")
        if response.status_code == 404:
            assert response.json()["code"] == 40400
        else:
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_system_page_loads(self, client):
        """System route does not masquerade as HTML without a built SPA."""
        response = client.get("/system")
        if response.status_code == 404:
            assert response.json()["code"] == 40400
        else:
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]


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
        camera_manager.get_camera_config.return_value = None

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

    def test_wall_status_surfaces_model_backend_mismatch(self, health, alerts_dir):
        """Video wall must expose backend policy failures, not hide them in logs."""
        camera_manager = MagicMock()
        camera_manager.get_status.return_value = [
            SimpleNamespace(
                camera_id="cam_gpu",
                name="GPU Camera",
                connected=True,
                running=True,
                model_version_id=None,
                stats=None,
            ),
        ]
        camera_manager.get_backpressure_stats.return_value = {
            "cam_gpu": {"pending": 0, "dropped": 0, "backpressured": False},
        }
        camera_manager.get_wall_status.return_value = {
            "current_score": 0.12,
            "score_sparkline": [0.1, 0.12],
        }
        camera_manager.get_anomaly_backend_status.return_value = {
            "actual": "openvino",
            "expected": "torch-cuda",
        }

        app = create_app(
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.get("/api/cameras/wall/status")

        assert response.status_code == 200
        tile = response.json()["data"]["cameras"][0]
        assert tile["degradation"] == "model_backend_mismatch"
        assert tile["model_backend"] == "openvino"
        assert tile["expected_model_backend"] == "torch-cuda"

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

    def test_start_capture_submits_task(self, db, health, alerts_dir, capture_compute_manager):
        """Baseline capture form should submit through the unified job endpoint."""
        task_manager = MagicMock()
        task_manager.submit.return_value = "baseline_capture-test1234"
        task_manager.get_task.return_value = None

        camera_manager = capture_compute_manager
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

    def test_legacy_capture_endpoint_alias_submits_task(self, db, health, alerts_dir, capture_compute_manager):
        """Quick capture clients posting to /capture should still start the same job flow."""
        task_manager = MagicMock()
        task_manager.submit.return_value = "baseline_capture-legacy"
        task_manager.get_task.return_value = None

        camera_manager = capture_compute_manager
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
        camera_manager.get_anomaly_backend_status.return_value = {
            "actual": "torch-cuda",
            "expected": "torch-cuda",
        }
        camera_manager.get_scene_guard_status.return_value = {
            "enabled": True,
            "usable": False,
            "state": "recovering",
            "reason": "awaiting consecutive approved-scene matches",
            "profile_id": "scene-main-v3",
            "recovery_progress": {"current": 2, "required": 3},
        }

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
        assert data["scene_guard"] == data["runtime"]["scene_guard"]
        assert data["scene_guard"]["usable"] is False
        assert data["scene_guard"]["profile_id"] == "scene-main-v3"
        assert data["scene_guard"]["bound_model_version"] is None
        assert data["scene_guard"]["required_recovery_count"] == 3
        assert data["scene_guard"]["recovery_progress"] == {
            "current": 2,
            "required": 3,
            "complete": False,
        }
        assert data["runner"]["version_tag"] == "v1"
        assert data["detector"]["threshold"] == 0.15
        assert data["model_backend"] == "torch-cuda"
        assert data["expected_model_backend"] == "torch-cuda"
        assert data["health"]["camera_id"] == "cam_01"


class TestCameraControlRoutes:
    def test_start_camera_offloads_to_thread(self, db, health, alerts_dir, monkeypatch):
        """Camera start route should offload blocking initialization from the event loop."""
        camera_manager = MagicMock()
        camera_manager.get_camera_config.return_value = None
        camera_manager.get_camera_status_payload.return_value = {
            "camera_id": "cam_01",
            "connected": True,
            "running": True,
        }
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
        data = response.json()["data"]
        assert {
            key: data[key] for key in ("camera_id", "connected", "running")
        } == {
            "camera_id": "cam_01",
            "connected": True,
            "running": True,
        }

    def test_stop_camera_offloads_to_thread(self, db, health, alerts_dir, monkeypatch):
        """Camera stop route should offload thread joins from the event loop."""
        camera_manager = MagicMock()
        camera_manager.get_camera_status_payload.return_value = {
            "camera_id": "cam_01",
            "connected": False,
            "running": False,
        }
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
        data = response.json()["data"]
        assert {
            key: data[key] for key in ("camera_id", "connected", "running")
        } == {
            "camera_id": "cam_01",
            "connected": False,
            "running": False,
        }

    def test_set_camera_mode_switches_running_pipeline(self, db, health, alerts_dir, monkeypatch):
        """Camera mode route should expose ACTIVE/LEARNING/MAINTENANCE control."""
        from argus.core.pipeline import PipelineMode

        camera_manager = MagicMock()
        camera_manager.get_pipeline_mode.return_value = "learning"
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

        response = client.post("/api/cameras/cam_01/mode", json={"mode": "active"})

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 0
        assert body["data"] == {
            "camera_id": "cam_01",
            "previous_mode": "learning",
            "pipeline_mode": "active",
        }
        assert calls[0][0] == camera_manager.set_pipeline_mode
        assert calls[0][1] == ("cam_01", PipelineMode.ACTIVE)

    def test_set_camera_mode_rejects_invalid_mode(self, db, health, alerts_dir):
        camera_manager = MagicMock()
        camera_manager.get_pipeline_mode.return_value = "learning"
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post("/api/cameras/cam_01/mode", json={"mode": "bad"})

        assert response.status_code == 422
        assert response.json()["code"] == 42200
        assert response.json()["data"]["errors"]
        camera_manager.set_pipeline_mode.assert_not_called()

    def test_set_camera_mode_requires_running_pipeline(self, db, health, alerts_dir):
        camera_manager = MagicMock()
        camera_manager.get_pipeline_mode.return_value = None
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post("/api/cameras/cam_01/mode", json={"mode": "active"})

        assert response.status_code == 404
        camera_manager.set_pipeline_mode.assert_not_called()

    def test_config_endpoint_keeps_original_usb_source_after_stream_registration(
        self, db, health, alerts_dir,
    ):
        """go2rtc registration must not leak runtime RTSP URLs into config APIs."""
        camera_config = CameraConfig(
            camera_id="usb_cam",
            name="USB Camera",
            source="0",
            protocol="usb",
        )
        config = ArgusConfig(cameras=[camera_config])
        camera_manager = MagicMock()
        camera_manager._cameras = [camera_config]
        go2rtc = MagicMock()
        type(go2rtc).running = property(lambda self: True)
        go2rtc.register_camera.return_value = "rtsp://127.0.0.1:8554/usb_cam"

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
            go2rtc_instance=go2rtc,
        )
        client = TestClient(app)

        stream_resp = client.get("/api/streaming/usb_cam")
        config_resp = client.get("/api/cameras/usb_cam/config")

        assert stream_resp.status_code == 200
        assert config_resp.status_code == 200
        payload = config_resp.json()["data"]
        assert payload["source"] == "0"
        assert payload["protocol"] == "usb"
        go2rtc.register_camera.assert_called_with(
            "usb_cam",
            usb_to_go2rtc_source(
                "0",
                resolution=camera_config.resolution,
                fps=camera_config.fps_target,
                pixel_format=camera_config.usb.pixel_format,
            ),
            "rtsp",
        )

    def test_delete_camera_removes_go2rtc_stream_from_app_state(
        self, db, health, alerts_dir,
    ):
        camera_config = CameraConfig(
            camera_id="cam_01",
            name="Camera 01",
            source="rtsp://example/stream",
            protocol="rtsp",
        )
        config = ArgusConfig(cameras=[camera_config])
        camera_manager = MagicMock()
        camera_manager._cameras = [camera_config]
        go2rtc = MagicMock()
        type(go2rtc).running = property(lambda self: True)

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
            go2rtc_instance=go2rtc,
        )
        client = TestClient(app)

        response = client.delete("/api/cameras/cam_01")

        assert response.status_code == 200
        go2rtc.remove_stream.assert_called_once_with("cam_01")

    def test_streaming_register_uses_camera_manager_config_lookup(
        self, db, health, alerts_dir,
    ):
        camera_config = CameraConfig(
            camera_id="usb_cam",
            name="USB Camera",
            source="0",
            protocol="usb",
        )
        config = ArgusConfig(cameras=[camera_config])
        camera_manager = MagicMock()
        camera_manager._cameras = [camera_config]
        camera_manager.get_camera_config.return_value = camera_config
        go2rtc = MagicMock()
        type(go2rtc).running = property(lambda self: True)
        go2rtc.register_camera.return_value = "rtsp://127.0.0.1:8554/usb_cam"

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
            go2rtc_instance=go2rtc,
        )
        client = TestClient(app)

        response = client.post("/api/streaming/usb_cam/register")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["runtime_source"] == "rtsp://127.0.0.1:8554/usb_cam"
        assert payload["runtime_protocol"] == "rtsp"

    def test_running_camera_connection_probe_uses_latest_frame(
        self, db, health, alerts_dir, monkeypatch,
    ):
        """Running cameras should not be probed by reopening the physical source."""
        camera_config = CameraConfig(
            camera_id="usb_cam",
            name="USB Camera",
            source="0",
            protocol="usb",
        )
        config = ArgusConfig(cameras=[camera_config])
        camera_manager = MagicMock()
        camera_manager._cameras = [camera_config]
        camera_manager.get_status.return_value = [
            SimpleNamespace(
                camera_id="usb_cam",
                name="USB Camera",
                connected=True,
                running=True,
                stats=None,
            ),
        ]
        camera_manager.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        def fail_probe(*_args, **_kwargs):
            raise AssertionError("physical source should not be reopened")

        monkeypatch.setattr("argus.dashboard.routes.cameras._probe_source_blocking", fail_probe)

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        client = TestClient(app)

        response = client.post("/api/cameras/usb_cam/test-connection")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["ok"] is True
        assert payload["resolution"] == [640, 480]
        assert payload["source"] == "running_pipeline"

    def test_restart_camera_offloads_atomic_restart(self, db, health, alerts_dir, monkeypatch):
        """One owned restart operation must not abandon half a stop/start pair."""
        camera_manager = MagicMock()
        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            if func == camera_manager.restart_camera:
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
        assert len(calls) == 1
        assert calls[0][0] == camera_manager.restart_camera
        assert calls[0][1] == ("cam_01",)
        camera_manager.stop_camera.assert_not_called()
        camera_manager.start_camera.assert_not_called()


class TestModelPublishRoutes:
    @staticmethod
    def _set_role_cookie(client: TestClient, app, role: str) -> None:
        username = f"{role}_user"
        db = app.state.db
        if db.get_user(username) is None:
            db.create_user(
                username,
                hash_password("role-test-password"),
                role,
                username,
            )
        token = create_session_token(username, role, app.state.session_secret)
        client.cookies.set("argus_session", token)

    @staticmethod
    def _register_model(registry: ModelRegistry, root: Path, camera_id: str = "cam_01") -> str:
        model_dir = root / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.xml").write_text("model")
        (model_dir / "model.bin").write_bytes(b"weights")
        baseline_dir = root / "baseline"
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "img.png").write_bytes(b"img")
        return registry.register(model_dir, baseline_dir, camera_id, "patchcore")

    def test_legacy_activate_fails_closed_even_for_production(
        self,
        db,
        health,
        alerts_dir,
        tmp_path,
        release_evidence_factory,
    ):
        """A production row cannot be re-applied outside the release service."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model")
        (model_dir / "model.bin").write_bytes(b"weights")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
        _release_model_fixture(
            db,
            registry,
            version_id,
            model_dir,
            release_evidence_factory,
        )

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

        assert response.status_code == 409
        assert "直接激活入口已关闭" in response.json()["msg"]
        unchanged = registry.get_by_version_id(version_id)
        assert unchanged.stage == "production"
        assert unchanged.is_active is True
        camera_manager.reload_model.assert_not_called()

    def test_activate_candidate_model_rejected_by_closed_legacy_entrypoint(
        self, db, health, alerts_dir, tmp_path,
    ):
        """Candidates are directed to the sole two-phase promotion API."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
        # Stage stays at default 'candidate'.

        camera_manager = MagicMock()
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post(f"/api/models/{version_id}/activate")

        assert response.status_code == 409
        body = response.json()
        # Must point at the controlled release command without mutating state.
        msg = (body.get("msg") or body.get("message") or "").lower()
        assert "/promote" in msg
        unchanged = registry.get_by_version_id(version_id)
        assert unchanged.stage == "candidate"
        assert unchanged.is_active is False
        camera_manager.reload_model.assert_not_called()

    @pytest.mark.parametrize("role", ["viewer", "operator"])
    def test_model_mutation_rejects_roles_without_manage_models(
        self, role, db, health, alerts_dir, tmp_path,
    ):
        """Model release writes are guarded by backend RBAC, not only the UI."""
        config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="test-token"))
        app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir), config=config)
        client = TestClient(app)
        self._set_role_cookie(client, app, role)

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = self._register_model(registry, tmp_path / role)

        response = client.post(
            f"/api/models/{version_id}/promote",
            json={"target_stage": "shadow", "triggered_by": role},
            headers={"accept": "application/json"},
        )

        assert response.status_code == 403

    @pytest.mark.parametrize("role", ["engineer", "admin"])
    def test_model_mutation_allows_manage_models_roles(
        self, role, db, health, alerts_dir, tmp_path,
        running_release_client, await_release_operation,
    ):
        config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="test-token"))
        config.dashboard.go2rtc_enabled = False
        app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir), config=config)
        client = running_release_client(app)
        self._set_role_cookie(client, app, role)

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = self._register_model(registry, tmp_path / role)
        app.state.camera_manager = _two_phase_release_runtime()

        response = client.post(
            f"/api/models/{version_id}/promote",
            json={"target_stage": "shadow", "triggered_by": role},
        )

        assert response.status_code == 202, response.text
        assert await_release_operation(client, response)["model"]["stage"] == "shadow"


class TestUserManagementRoutes:
    @staticmethod
    def _human_admin_client(db, health, alerts_dir):
        db.create_user("user-admin", hash_password("admin-secret"), "admin", None)
        app = create_app(
            database=db,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=ArgusConfig(auth=AuthConfig(enabled=True, api_token="user-test-token")),
        )
        client = TestClient(app)
        client.cookies.set("argus_session", create_session_token(
            "user-admin", "admin", app.state.session_secret,
        ))
        return client

    def test_create_and_update_engineer_role(self, db, health, alerts_dir):
        """User management accepts the canonical four-role RBAC set."""
        client = self._human_admin_client(db, health, alerts_dir)

        create_resp = client.post(
            "/api/users/json",
            json={
                "username": "eng1",
                "password": "secret1",
                "role": "engineer",
                "display_name": "Engineer One",
            },
        )
        assert create_resp.status_code == 200
        assert create_resp.json()["data"]["user"]["role"] == "engineer"

        update_resp = client.put(
            "/api/users/eng1/json",
            json={
                "role": "engineer",
                "display_name": "Engineer Updated",
                "active": True,
            },
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["data"]["user"]["role"] == "engineer"

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {
                "username": "strict-create",
                "password": "secret1",
                "role": "viewer",
                "active": "false",
            },
        ],
    )
    def test_create_user_rejects_non_object_and_string_boolean(
        self,
        db,
        health,
        alerts_dir,
        payload,
    ):
        app = create_app(
            database=db,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.post("/api/users/json", json=payload)

        assert response.status_code == 422
        assert response.json()["code"] == 42200
        assert db.get_user("strict-create") is None

    def test_update_user_rejects_string_boolean_without_mutating_user(
        self,
        db,
        health,
        alerts_dir,
    ):
        db.create_user(
            "strict-update",
            hash_password("secret1"),
            "viewer",
            "Strict Update",
        )
        app = create_app(
            database=db,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
        )
        client = TestClient(app)

        response = client.put(
            "/api/users/strict-update/json",
            json={"active": "false"},
        )

        assert response.status_code == 422
        assert response.json()["code"] == 42200
        unchanged = db.get_user("strict-update")
        assert unchanged is not None
        assert unchanged.active is True

    def test_update_user_preserves_omitted_fields(
        self,
        db,
        health,
        alerts_dir,
    ):
        db.create_user(
            "partial-update",
            hash_password("secret1"),
            "viewer",
            "Keep This Name",
        )
        client = self._human_admin_client(db, health, alerts_dir)

        response = client.put(
            "/api/users/partial-update/json",
            json={"role": "operator"},
        )

        assert response.status_code == 200
        user = response.json()["data"]["user"]
        assert user["role"] == "operator"
        assert user["display_name"] == "Keep This Name"
        assert user["active"] is True

    def test_config_reload_model_is_fail_closed(self, db, health, alerts_dir, tmp_path, monkeypatch):
        """A path-based reload cannot bypass the two-phase release service."""
        monkeypatch.chdir(tmp_path)

        model_dir = tmp_path / "data" / "models" / "cam_01" / "default"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "model.xml"
        model_path.write_text("model")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

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

        assert response.status_code == 409
        assert response.json()["data"]["runtime_changed"] is False
        camera_manager.reload_model.assert_not_called()

    def test_config_reload_model_resolves_checkpoint_to_exported_torch(
        self,
        db,
        health,
        alerts_dir,
        tmp_path,
        monkeypatch,
    ):
        """Even a resolvable export cannot bypass the release service."""
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
        registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

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

        assert response.status_code == 409
        assert response.json()["data"]["runtime_changed"] is False
        camera_manager.reload_model.assert_not_called()

    def test_config_reload_model_rejects_checkpoint_without_runtime_artifact(
        self,
        db,
        health,
        alerts_dir,
        tmp_path,
        monkeypatch,
    ):
        """Legacy reload remains closed regardless of artifact layout."""
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

        assert response.status_code == 409
        assert response.json()["data"]["runtime_changed"] is False
        camera_manager.reload_model.assert_not_called()

    def test_baseline_deploy_is_fail_closed_for_registered_production(
        self,
        db,
        health,
        alerts_dir,
        tmp_path,
        monkeypatch,
        release_evidence_factory,
    ):
        """The legacy path deploy cannot reload even an approved production."""
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
        version_id = registry.register(export_dir, baseline_dir, "cam_01", "patchcore")
        _release_model_fixture(
            db,
            registry,
            version_id,
            export_dir,
            release_evidence_factory,
        )

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
            json={"camera_id": "cam_01", "model_path": str(export_dir)},
        )

        assert response.status_code == 409
        assert "路径部署入口已关闭" in response.json()["msg"]
        unchanged = registry.get_by_version_id(version_id)
        assert unchanged.stage == "production"
        assert unchanged.is_active is True
        camera_manager.reload_model.assert_not_called()

    def test_v2_baseline_keeps_production_promotion_no_go(
        self,
        db,
        health,
        alerts_dir,
        tmp_path,
        release_baseline_factory,
        release_evidence_sources_factory,
        running_release_client,
        await_release_operation,
    ):
        """Offline v2 baseline decoder evidence cannot authorize production."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model")
        (model_dir / "model.bin").write_bytes(b"weights")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")
        baseline_dir = release_baseline_factory(root=baseline_dir)

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

        config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="release-test-token"))
        config.dashboard.go2rtc_enabled = False
        db.create_user("release-admin", hash_password("release-secret"), "admin", None)
        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        client = running_release_client(app)
        client.cookies.set("argus_session", create_session_token(
            "release-admin", "admin", app.state.session_secret,
        ))
        registered = registry.get_by_version_id(version_id)
        release_evidence_sources = release_evidence_sources_factory(
            root=tmp_path / "release-sources",
            model_path=model_dir,
            model_version_id=version_id,
            registry_model_hash=registered.model_hash,
            baseline_path=baseline_dir,
        )

        response = client.post(
            f"/api/models/{version_id}/promote",
            json={
                "target_stage": "production",
                "triggered_by": "operator",
                "release_evidence_sources": release_evidence_sources,
            },
        )

        failed = await_release_operation(client, response, status="failed")
        assert "decoder evidence is insufficient" in failed["operation"]["error"]
        assert registry.get_by_version_id(version_id).stage == "canary"
        camera_manager.reload_model.assert_not_called()


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
    def test_scene_guard_status_delegates_to_pipeline(self):
        manager = CameraManager(cameras=[], alert_config=AlertConfig())
        pipeline = MagicMock()
        pipeline.get_scene_guard_status.return_value = {
            "enabled": True,
            "state": "stable",
        }
        manager._pipelines["cam_01"] = pipeline

        assert manager.get_scene_guard_status("cam_01") == {
            "enabled": True,
            "state": "stable",
        }
        assert manager.get_scene_guard_status("missing") is None

    def test_prefers_active_registry_model_on_startup(
        self,
        tmp_path,
        release_evidence_factory,
    ):
        """CameraManager should restore the active registry model for restarted cameras."""
        db = Database(database_url=f"sqlite:///{tmp_path / 'camera.db'}")
        db.initialize()

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("model")
        (model_dir / "model.bin").write_bytes(b"weights")
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "img.png").write_bytes(b"img")

        registry = ModelRegistry(session_factory=db.get_session)
        version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
        _release_model_fixture(
            db,
            registry,
            version_id,
            model_dir,
            release_evidence_factory,
        )

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

    def test_reload_model_entrypoint_is_fail_closed(self):
        """Path-based reload must not bypass ModelReleaseService."""
        manager = CameraManager(cameras=[], alert_config=AlertConfig())
        pipeline = MagicMock()
        pipeline.reload_anomaly_model.return_value = True
        runner = MagicMock()
        manager._pipelines["cam_01"] = pipeline
        manager._runners["cam_01"] = runner
        manager._notify_status_change = MagicMock()

        ok = manager.reload_model("cam_01", "data/models/cam_01/model.xml", version_tag="v2")

        assert ok is False
        runner.set_version_tag.assert_not_called()
        pipeline.reload_anomaly_model.assert_not_called()
        pipeline.set_model_version_id.assert_not_called()

    def test_detector_status_includes_input_quality(self):
        manager = CameraManager(cameras=[], alert_config=AlertConfig())
        pipeline = MagicMock()
        pipeline.get_detector_status.return_value = SimpleNamespace(
            mode="ssim_fallback",
            model_path=None,
            model_loaded=False,
            threshold=0.7,
            ssim_calibration_progress=0.0,
            ssim_calibrated=False,
            ssim_noise_floor=None,
        )
        pipeline.get_input_quality_status.return_value = {
            "low_light": True,
            "detection_limited": True,
            "ssim_calibration_blocked": True,
        }
        manager._pipelines["cam_01"] = pipeline

        status = manager.get_detector_status("cam_01")

        assert status["mode"] == "ssim_fallback"
        assert status["low_light"] is True
        assert status["ssim_calibration_blocked"] is True

    def test_blocked_reload_broadcasts_release_service_requirement(self):
        """Blocked legacy reload should be visible without touching the detector."""
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
        assert data["current_version"] is None
        assert data["reason"] == "release_service_required"
        pipeline.reload_anomaly_model.assert_not_called()


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

        monkeypatch.setitem(sys.modules, "argus.anomaly.backbone_trainer", SimpleNamespace(
            BackboneTrainer=backbone_trainer_cls,
        ))
        monkeypatch.setattr("argus.storage.model_registry.ModelRegistry", registry_cls)
        monkeypatch.setattr("argus.anomaly.job_executor.TrainingJobExecutor", executor_cls)
        monkeypatch.setattr(
            "argus.runtime.training_job_wiring.create_job_processing_task",
            register_task,
        )

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


class TestModuleToggleRoutes:
    def test_toggle_reports_partial_hot_reload_and_persistence(
        self, db, health, alerts_dir,
    ):
        config = ArgusConfig()
        config.cameras = [
            SimpleNamespace(camera_id="cam_a"),
            SimpleNamespace(camera_id="cam_b"),
        ]
        config.classifier.enabled = False

        camera_manager = MagicMock()
        camera_manager.reload_module_for_cameras.return_value = {
            "seen": 2,
            "succeeded": 1,
            "failed": 1,
        }

        app = create_app(
            database=db,
            camera_manager=camera_manager,
            health_monitor=health,
            alerts_dir=str(alerts_dir),
            config=config,
        )
        client = TestClient(app)

        response = client.post(
            "/api/config/modules",
            json={"key": "classifier.enabled", "value": True},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["value"] is True
        assert data["hot_reloadable"] is True
        assert data["pipelines_seen"] == 2
        assert data["hot_reloaded"] == 1
        assert data["hot_reload_failed"] == 1
        assert data["restart_required"] is True
        assert data["persisted"] is False
        assert config.classifier.enabled is True
        camera_manager.reload_module_for_cameras.assert_called_once_with(
            ["cam_a", "cam_b"],
            "classifier.enabled",
            True,
        )

    def test_toggle_without_config_returns_unavailable(self, db, health, alerts_dir):
        app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir))
        app.state.config = None
        client = TestClient(app)

        response = client.post(
            "/api/config/modules",
            json={"key": "classifier.enabled", "value": True},
        )

        assert response.status_code == 503
        assert "配置" in response.json()["msg"]


class TestGlobalDegradationWiring:
    def test_create_app_wires_health_monitor_to_degradation_manager(
        self, db, health, alerts_dir,
    ):
        assert getattr(health, "_degradation_manager", None) is None

        app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir))
        client = TestClient(app)

        assert health._degradation_manager is app.state.degradation_manager

        health.update_camera("cam_01", connected=True)
        health.update_camera(
            "cam_01",
            connected=False,
            reconnect_count=2,
            error="rtsp timeout",
        )

        summary = client.get("/api/degradation/summary")
        assert summary.status_code == 200
        data = summary.json()["data"]
        assert data["active_count"] == 1
        assert data["max_level"] == "warning"
        assert data["events"][0]["category"] == "rtsp_broken"
        assert data["events"][0]["camera_id"] == "cam_01"

        health.update_camera("cam_01", connected=True)
        resolved = client.get("/api/degradation/summary").json()["data"]
        assert resolved["active_count"] == 0


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
            "pipelines_ready": 0,
            "pipelines_degraded": 0,
            "pipelines_failed": 0,
            "details": [],
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
            "pipelines_ready": 0,
            "pipelines_degraded": 0,
            "pipelines_failed": 0,
            "details": [],
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

        cm.get_runtime_component_summary.return_value = {
            "total_pipelines": 1,
            "segmenter_attached": 1,
            "segmenter_loaded": 1,
        }

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
        cm.get_runtime_component_summary.return_value = {
            "total_pipelines": 1,
            "correlator_present": True,
        }
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
