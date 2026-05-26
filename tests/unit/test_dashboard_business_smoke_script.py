from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from argus.config.loader import load_config
from scripts.smoke_dashboard_business_flow import (
    DashboardBusinessSmokeFailure,
    _business_browser_pages,
    _check_business_browser_dom,
    _clean_env,
    _cleanup_go2rtc_api_port,
    _infer_camera_protocol,
    _models_system_browser_pages,
    _model_operation_timeout,
    _objective_checklist,
    _parse_resolution,
    _prepare_runtime_config,
    _seed_model_registry,
    _seed_training_baselines,
    _verify_business_apis,
    _verify_dev_video_alert_semantics,
    _verify_system_degradation_apis,
    _wait_for,
    _wait_for_completed_alert,
    _websocket_url,
    parse_args,
    run_camera_preflight,
)


def test_prepare_runtime_config_redirects_writable_paths(tmp_path):
    runtime_config, camera_id = _prepare_runtime_config(
        config_path=Path("configs/default.yaml"),
        work_dir=tmp_path,
        port=18181,
        use_yolo=False,
    )

    config = load_config(runtime_config)

    assert camera_id == config.cameras[0].camera_id
    assert config.dashboard.host == "127.0.0.1"
    assert config.dashboard.port == 18181
    assert config.dashboard.go2rtc_enabled is False
    assert config.cameras[0].anomaly.min_shadow_days == 0
    assert config.cameras[0].anomaly.min_canary_days == 0
    assert config.storage.database_url == f"sqlite:///{(tmp_path / 'argus.db').as_posix()}"
    assert config.storage.alerts_dir == tmp_path / "alerts"
    assert config.storage.models_dir == tmp_path / "models"
    assert config.storage.baselines_dir == tmp_path / "baselines"
    assert config.logging.log_dir == tmp_path / "logs"
    assert "missing-yolo.pt" in config.cameras[0].person_filter.model_name
    assert config.alerts.severity_thresholds.low == pytest.approx(0.55)


def test_prepare_runtime_config_can_use_usb_source_with_go2rtc(monkeypatch, tmp_path):
    ports = iter([21984, 28554, 28555])
    monkeypatch.setattr(
        "scripts.smoke_dashboard_business_flow._free_port",
        lambda: next(ports),
    )

    runtime_config, camera_id = _prepare_runtime_config(
        config_path=Path("configs/default.yaml"),
        work_dir=tmp_path,
        port=18181,
        use_yolo=False,
        camera_source="0",
        camera_protocol="usb",
        camera_id="usb_cam",
        camera_name="USB Cam",
        camera_resolution=(1280, 720),
        usb_device_name="OBSBOT Meet 2 StreamCamera",
        usb_device_id="@device_pnp_usb_vid_3564_pid_3022",
    )

    config = load_config(runtime_config)
    camera = config.cameras[0]
    assert camera_id == "usb_cam"
    assert camera.camera_id == "usb_cam"
    assert camera.name == "USB Cam"
    assert camera.protocol == "usb"
    assert camera.source == "0"
    assert camera.resolution == (1280, 720)
    assert camera.usb.device_name == "OBSBOT Meet 2 StreamCamera"
    assert camera.usb.device_id == "@device_pnp_usb_vid_3564_pid_3022"
    assert config.dashboard.go2rtc_enabled is True
    assert config.dashboard.go2rtc_api_port == 21984
    assert config.dashboard.go2rtc_rtsp_port == 28554
    assert config.dashboard.go2rtc_webrtc_port == 28555


def test_infer_camera_protocol_from_source_shape():
    assert _infer_camera_protocol("rtsp://example.test/stream") == "rtsp"
    assert _infer_camera_protocol("0") == "usb"
    assert _infer_camera_protocol("/dev/video2") == "usb"
    assert _infer_camera_protocol("data/dev/demo_camera.avi") == "file"


def test_parse_resolution_accepts_comma_or_x_separator():
    assert _parse_resolution("640,480") == (640, 480)
    assert _parse_resolution("1280x720") == (1280, 720)
    with pytest.raises(Exception, match="camera-resolution"):
        _parse_resolution("bad")


def test_business_browser_pages_include_dynamic_alert_and_camera_ids():
    pages = _business_browser_pages(alert_id="ALERT-12345678", camera_id="cam_a")

    assert "/cameras/cam_a" in pages
    assert "/alerts?id=ALERT-12345678" in pages
    assert "/replay/ALERT-12345678" in pages
    assert "cam_a" in pages["/cameras/cam_a"]
    assert "12345678" in pages["/alerts?id=ALERT-12345678"]
    assert "ALERT-12345678" in pages["/replay/ALERT-12345678"]


def test_websocket_url_derives_ws_endpoint_from_http_base_url():
    assert _websocket_url("http://127.0.0.1:8080") == "ws://127.0.0.1:8080/ws"
    assert _websocket_url("https://example.test") == "wss://example.test/ws"


def test_parse_args_accepts_normal_training_mode():
    args = parse_args([
        "--training-mode", "normal",
        "--training-baseline-count", "32",
        "--training-image-size", "96",
        "--browser", "off",
    ])

    assert args.training_mode == "normal"
    assert args.training_baseline_count == 32
    assert args.training_image_size == 96


def test_model_operation_timeout_scales_for_normal_training_reexport():
    assert _model_operation_timeout(10) == 30.0
    assert _model_operation_timeout(90) == 90.0
    assert _model_operation_timeout(420) == 180.0


def test_parse_args_accepts_hardware_camera_source():
    args = parse_args([
        "--camera-source", "0",
        "--camera-protocol", "usb",
        "--camera-id", "usb_cam",
        "--camera-resolution", "1280x720",
        "--require-go2rtc",
        "--activation-delay", "5",
        "--browser", "off",
    ])

    assert args.camera_source == "0"
    assert args.camera_protocol == "usb"
    assert args.camera_id == "usb_cam"
    assert args.require_go2rtc is True
    assert args.activation_delay == 5.0


def test_parse_args_accepts_preflight_mode():
    args = parse_args([
        "--preflight",
        "--camera-source", "0",
        "--camera-protocol", "usb",
        "--preflight-timeout", "1.5",
        "--preflight-measure-seconds", "4.5",
        "--browser", "off",
    ])

    assert args.preflight is True
    assert args.preflight_timeout == 1.5
    assert args.preflight_measure_seconds == 4.5


def test_parse_args_accepts_local_rtsp_fixture():
    args = parse_args([
        "--rtsp-fixture",
        "--rtsp-fixture-seconds", "60",
        "--dev-video-motion", "book",
        "--require-go2rtc",
        "--browser", "off",
    ])

    assert args.rtsp_fixture is True
    assert args.rtsp_fixture_seconds == 60
    assert args.dev_video_motion == "book"
    assert args.require_go2rtc is True
    assert args.recording_timeout == 90.0


def test_wait_for_completed_alert_timeout_reports_last_evidence_state():
    class RunningProcess:
        returncode = None

        def poll(self):
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/alerts/json":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "alerts": [
                            {
                                "alert_id": "ALT-1",
                                "camera_id": "cam_a",
                                "severity": "medium",
                            }
                        ]
                    },
                },
            )
        if request.url.path == "/api/alerts/ALT-1/detail":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "alert_id": "ALT-1",
                        "has_recording": True,
                        "recording_status": "recording",
                        "snapshot_path": "snapshot.jpg",
                        "heatmap_path": None,
                    },
                },
            )
        return httpx.Response(404, json={"code": 404, "message": "not found"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://argus.test",
    )

    with pytest.raises(DashboardBusinessSmokeFailure) as exc:
        _wait_for_completed_alert(
            client,
            camera_id="cam_a",
            alert_timeout_s=0.05,
            recording_timeout_s=0.05,
            process=RunningProcess(),
        )

    msg = str(exc.value)
    assert "Last alert evidence state" in msg
    assert '"recording_status": "recording"' in msg
    assert '"heatmap_path": null' in msg


def test_wait_for_checks_predicate_once_at_deadline():
    calls = {"count": 0}

    def predicate():
        calls["count"] += 1
        return "ok"

    assert _wait_for("edge condition", predicate, timeout_s=0) == "ok"
    assert calls["count"] == 1


def test_book_dev_video_semantics_rejects_projectile_category():
    args = parse_args(["--dev-video-motion", "book", "--browser", "off"])
    alert = {
        "alert_id": "ALT-1",
        "realtime": {
            "detection_type": "projectile",
            "category": "projectile",
        },
    }

    with pytest.raises(DashboardBusinessSmokeFailure, match="reported as projectile"):
        _verify_dev_video_alert_semantics(args, alert)


def test_book_dev_video_semantics_accepts_scene_change():
    args = parse_args(["--dev-video-motion", "book", "--browser", "off"])
    alert = {
        "alert_id": "ALT-1",
        "realtime": {
            "detection_type": "anomaly",
            "category": "scene_change",
        },
    }

    result = _verify_dev_video_alert_semantics(args, alert)

    assert result == {
        "motion": "book",
        "detection_type": "anomaly",
        "category": "scene_change",
    }


def test_run_camera_preflight_delegates_to_core_preflight(monkeypatch, tmp_path):
    captured = {}

    def fake_run_preflight(args):
        captured["args"] = args
        return {
            "ok": True,
            "mode": "preflight",
            "camera_input": {"camera_id": args.camera_id},
            "errors": [],
        }

    monkeypatch.setattr("scripts.smoke_core_loop.run_preflight", fake_run_preflight)
    args = parse_args([
        "--preflight",
        "--work-dir", str(tmp_path),
        "--camera-source", "0",
        "--camera-protocol", "usb",
        "--camera-id", "usb_cam",
        "--camera-resolution", "1280x720",
        "--usb-device-name", "OBSBOT Meet 2 StreamCamera",
        "--usb-device-id", "@device_pnp_usb_vid_3564_pid_3022",
        "--require-go2rtc",
        "--preflight-measure-seconds", "6.5",
        "--browser", "off",
    ])

    result = run_camera_preflight(args)

    delegated = captured["args"]
    assert delegated.camera_source == "0"
    assert delegated.camera_protocol == "usb"
    assert delegated.camera_id == "usb_cam"
    assert delegated.camera_resolution == "1280x720"
    assert delegated.usb_device_name == "OBSBOT Meet 2 StreamCamera"
    assert delegated.usb_device_id == "@device_pnp_usb_vid_3564_pid_3022"
    assert delegated.require_go2rtc is True
    assert delegated.preflight_measure_seconds == 6.5
    assert delegated.dev_video_motion == "settle"
    assert delegated.work_dir == tmp_path
    assert result["ok"] is True
    assert "business_smoke_command" in result


def test_run_camera_preflight_can_delegate_local_rtsp_fixture(monkeypatch, tmp_path):
    captured = {}

    class FakeRtspFixture:
        def __init__(self, *, work_dir, resolution, seconds, motion):
            captured["fixture_init"] = {
                "work_dir": work_dir,
                "resolution": resolution,
                "seconds": seconds,
                "motion": motion,
            }

        def start(self):
            captured["fixture_started"] = True
            return {
                "source_url": "rtsp://127.0.0.1:15554/argus_rtsp_fixture",
                "stream_name": "argus_rtsp_fixture",
            }

        def close(self):
            captured["fixture_closed"] = True

    def fake_run_preflight(args):
        captured["args"] = args
        return {
            "ok": True,
            "mode": "preflight",
            "camera_input": {"camera_id": args.camera_id},
            "errors": [],
        }

    monkeypatch.setattr(
        "scripts.smoke_dashboard_business_flow._RtspFixture",
        FakeRtspFixture,
    )
    monkeypatch.setattr("scripts.smoke_core_loop.run_preflight", fake_run_preflight)

    args = parse_args([
        "--preflight",
        "--work-dir", str(tmp_path),
        "--rtsp-fixture",
        "--rtsp-fixture-seconds", "60",
        "--dev-video-motion", "book",
        "--require-go2rtc",
        "--browser", "off",
    ])

    result = run_camera_preflight(args)

    delegated = captured["args"]
    assert delegated.camera_source == "rtsp://127.0.0.1:15554/argus_rtsp_fixture"
    assert delegated.camera_protocol == "rtsp"
    assert delegated.require_go2rtc is True
    assert captured["fixture_init"]["work_dir"] == tmp_path
    assert captured["fixture_init"]["resolution"] == (640, 480)
    assert captured["fixture_init"]["seconds"] == 60
    assert captured["fixture_init"]["motion"] == "book"
    assert captured["fixture_started"] is True
    assert captured["fixture_closed"] is True
    assert delegated.dev_video_motion == "book"
    assert result["rtsp_fixture"]["stream_name"] == "argus_rtsp_fixture"


def test_models_system_browser_pages_include_seeded_model_and_system_markers():
    model_ids = {
        "candidate": "cam_a-patchcore-candidate",
        "shadow": "cam_a-patchcore-shadow",
        "canary": "cam_a-patchcore-canary",
        "production": "cam_a-patchcore-production",
    }

    pages = _models_system_browser_pages(
        camera_id="cam_a",
        model_ids=model_ids,
        trained_version_id="cam_a-patchcore-trained",
    )

    assert "/models/registry" in pages
    assert "/system/overview" in pages
    assert "/system/config" in pages
    assert "cam_a-patchcore-production" in pages["/models/registry"]
    assert "cam_a-patchcore-trained" in pages["/models/registry"]
    assert "fallback 模式" in pages["/system/overview"]
    assert "保存当前配置" in pages["/system/config"]


def test_business_browser_dom_retries_until_markers_are_rendered(monkeypatch, tmp_path):
    args = parse_args(["--browser", "required", "--browser-timeout", "3"])
    calls: list[int] = []

    monkeypatch.setattr(
        "scripts.smoke_dashboard_business_flow._find_headless_browser",
        lambda explicit_path=None: "chrome.exe",
    )

    def fake_dump(**kwargs):
        calls.append(kwargs["virtual_time_ms"])
        if len(calls) == 1:
            return "<html><head><title>ARGUS</title></head><body><div id=\"app\"></div></body></html>"
        return "系统配置 检测参数 保存当前配置 告警音频配置"

    monkeypatch.setattr("scripts.smoke_dashboard_business_flow._dump_dom_with_browser", fake_dump)
    monkeypatch.setattr("scripts.smoke_dashboard_business_flow._business_browser_pages", lambda **_: {})

    result = _check_business_browser_dom(
        args,
        base_url="http://127.0.0.1:1",
        work_dir=tmp_path,
        alert_id="ALT-1",
        camera_id="c",
        additional_pages={
            "/system/config": ["系统配置", "检测参数", "保存当前配置", "告警音频配置"]
        },
    )

    assert result["status"] == "checked"
    assert result["routes_checked"][0]["route"] == "/system/config"
    assert len(calls) == 2
    assert calls[1] > calls[0]


def test_cleanup_go2rtc_api_port_uses_exit_endpoint(monkeypatch):
    calls: list[str] = []
    running = {"value": True}

    def fake_post(url, timeout):
        calls.append(url)
        assert timeout == 1.0
        running["value"] = False
        return httpx.Response(200)

    def fake_get(url, timeout):
        assert timeout == 1.0
        if running["value"]:
            return httpx.Response(200)
        raise httpx.ConnectError("closed")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("taskkill fallback should not run after API exit")

    monkeypatch.setattr("scripts.smoke_dashboard_business_flow.httpx.post", fake_post)
    monkeypatch.setattr("scripts.smoke_dashboard_business_flow.httpx.get", fake_get)
    monkeypatch.setattr("scripts.smoke_dashboard_business_flow.subprocess.run", fail_run)

    _cleanup_go2rtc_api_port(21984)

    assert calls == ["http://127.0.0.1:21984/api/exit"]


def test_objective_checklist_maps_business_evidence_to_completion_items():
    alert_id = "ALERT-12345678"

    checklist = _objective_checklist(
        camera_id="cam_a",
        camera_row={
            "connected": True,
            "running": True,
            "stats": {"frames_captured": 42},
        },
        camera_media={
            "snapshot": {"bytes": 2048},
            "streaming": {"fallback": "/api/cameras/cam_a/stream", "go2rtc": None},
        },
        mode_result={"pipeline_mode": "active"},
        alert={
            "alert_id": alert_id,
            "severity": "medium",
            "_realtime_payload": {"alert_id": alert_id},
        },
        alert_detail={
            "has_recording": True,
            "recording_status": "complete",
            "snapshot_path": "snapshot.jpg",
            "heatmap_path": "heatmap.jpg",
        },
        api_result={
            "evidence_zip_bytes": 2048,
            "replay": {"frame_count": 10, "signal_points": 10},
            "reports": {
                "total_alerts": 1,
                "evidence_complete_rate": 100.0,
                "recording_rate": 100.0,
                "camera_distribution": {"camera_id": "cam_a", "count": 1},
            },
        },
        models_system_result={
            "config_update": {"pipelines_updated": 1},
            "fallback_models": [
                {
                    "name": "anomaly",
                    "camera_id": "cam_a",
                    "loaded": True,
                    "backend": "ssim-fallback",
                }
            ],
            "degradation": {
                "anomaly": {"degraded": False},
                "global": {"active_count": 0},
            },
            "training_export": {
                "training_mode": "dev-fast",
                "job_id": "job-1",
                "model_version_id": "model-1",
                "status": "complete",
                "reexport": {"status": "ok"},
            },
            "release_api": {
                "transitions": [
                    {"stage": "shadow", "runtime_synced": True},
                    {"stage": "canary", "runtime_synced": True},
                    {"stage": "production", "runtime_synced": True},
                ],
                "rollback": {
                    "activated": "model-0",
                    "runtime_synced": True,
                },
            },
        },
        browser_result={
            "status": "checked",
            "routes_checked": [
                {"route": "/cameras/cam_a"},
                {"route": f"/alerts?id={alert_id}"},
                {"route": f"/replay/{alert_id}"},
                {"route": "/models/registry"},
                {"route": "/system/overview"},
                {"route": "/system/config"},
                {"route": "/reports"},
            ]
        },
    )

    assert [item["passed"] for item in checklist] == [True] * 6
    assert checklist[0]["requirement"].startswith("1. Cameras")
    assert checklist[1]["evidence"]["websocket_alert_id"] == alert_id
    assert checklist[3]["evidence"]["training_mode"] == "dev-fast"
    assert checklist[3]["evidence"]["release_stages"] == ["shadow", "canary", "production"]
    assert checklist[4]["evidence"]["fallback_models"][0]["backend"] == "ssim-fallback"
    assert checklist[5]["evidence"]["total_alerts"] == 1


def test_business_api_verification_rejects_single_frame_replay():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/alerts/ALT-1/evidence.zip":
            return httpx.Response(200, content=b"x" * 2048)
        if request.url.path == "/api/replay/ALT-1/metadata":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "status": "complete",
                        "frame_count": 1,
                    },
                },
            )
        return httpx.Response(404, json={"code": 404, "message": "not found"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://argus.test",
    )

    with pytest.raises(DashboardBusinessSmokeFailure, match="replay metadata incomplete"):
        _verify_business_apis(client, alert_id="ALT-1", camera_id="cam_a")


def test_objective_checklist_does_not_pass_full_ui_objective_when_browser_is_off():
    alert_id = "ALERT-12345678"

    checklist = _objective_checklist(
        camera_id="cam_a",
        camera_row={
            "connected": True,
            "running": True,
            "stats": {"frames_captured": 42},
        },
        camera_media={
            "snapshot": {"bytes": 2048},
            "streaming": {"fallback": "/api/cameras/cam_a/stream", "go2rtc": None},
        },
        mode_result={"pipeline_mode": "active"},
        alert={
            "alert_id": alert_id,
            "severity": "medium",
            "_realtime_payload": {"alert_id": alert_id},
        },
        alert_detail={
            "has_recording": True,
            "recording_status": "complete",
            "snapshot_path": "snapshot.jpg",
            "heatmap_path": "heatmap.jpg",
        },
        api_result={
            "evidence_zip_bytes": 2048,
            "replay": {"frame_count": 10, "signal_points": 10},
            "reports": {
                "total_alerts": 1,
                "evidence_complete_rate": 100.0,
                "recording_rate": 100.0,
                "camera_distribution": {"camera_id": "cam_a", "count": 1},
            },
        },
        models_system_result={
            "config_update": {"pipelines_updated": 1},
            "fallback_models": [
                {
                    "name": "anomaly",
                    "camera_id": "cam_a",
                    "loaded": True,
                    "backend": "ssim-fallback",
                }
            ],
            "degradation": {
                "anomaly": {"degraded": False},
                "global": {"active_count": 0},
            },
            "training_export": {
                "training_mode": "normal",
                "job_id": "job-1",
                "model_version_id": "model-1",
                "status": "complete",
                "reexport": {"status": "ok"},
            },
            "release_api": {
                "transitions": [
                    {"stage": "shadow", "runtime_synced": True},
                    {"stage": "canary", "runtime_synced": True},
                    {"stage": "production", "runtime_synced": True},
                ],
                "rollback": {
                    "activated": "model-0",
                    "runtime_synced": True,
                },
            },
        },
        browser_result={"status": "off", "routes_checked": []},
    )

    assert [item["passed"] for item in checklist] == [False] * 6
    assert all(item["evidence"]["browser_checked"] is False for item in checklist)


def test_seed_training_baselines_creates_current_version(tmp_path):
    result = _seed_training_baselines(
        work_dir=tmp_path,
        camera_id="cam_a",
        count=30,
        image_size=64,
    )

    baseline_dir = Path(result["path"])
    assert result["count"] == 30
    assert result["image_size"] == 64
    assert baseline_dir.name == "v001"
    assert len(list(baseline_dir.glob("*.png"))) == 30
    assert (baseline_dir.parent / "current.txt").read_text(encoding="utf-8") == "v001"


def test_seed_model_registry_creates_all_release_stages(tmp_path):
    from argus.storage.database import Database
    from argus.storage.models import ModelRecord

    database_url = f"sqlite:///{tmp_path / 'argus.db'}"

    model_ids = _seed_model_registry(
        work_dir=tmp_path,
        database_url=database_url,
        camera_id="cam_a",
    )

    assert set(model_ids) == {"candidate", "shadow", "canary", "production"}

    db = Database(database_url=database_url)
    db.initialize()
    try:
        with db.get_session() as session:
            rows = {
                row.model_version_id: row
                for row in session.query(ModelRecord).all()
            }
        assert rows[model_ids["candidate"]].stage == "candidate"
        assert rows[model_ids["shadow"]].stage == "shadow"
        assert rows[model_ids["canary"]].stage == "canary"
        assert rows[model_ids["production"]].stage == "production"
        assert rows[model_ids["production"]].is_active is True
    finally:
        db.close()


def test_verify_system_degradation_apis_checks_anomaly_and_global_contracts():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/system/anomaly-degradation":
            data = {
                "anomaly": {
                    "degraded": True,
                    "reason": "load_failed",
                    "since": 123.0,
                    "cameras": [
                        {
                            "camera_id": "cam_a",
                            "degraded": True,
                            "reason": "load_failed",
                            "since": 123.0,
                        }
                    ],
                }
            }
        elif path == "/api/degradation/active":
            data = {
                "items": [
                    {
                        "event_id": "evt-1",
                        "level": "warning",
                        "category": "model_fallback",
                        "camera_id": "cam_a",
                    }
                ]
            }
        elif path == "/api/degradation/summary":
            data = {
                "active_count": 1,
                "max_level": "warning",
                "events": [{"event_id": "evt-1"}],
            }
        elif path == "/api/degradation/history":
            data = {"items": [{"event_id": "evt-1"}]}
        else:
            return httpx.Response(404, json={"code": 404, "message": "not found"})
        return httpx.Response(200, json={"code": 0, "data": data})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://argus.test",
    )

    result = _verify_system_degradation_apis(client, camera_id="cam_a")

    assert result["anomaly"]["degraded"] is True
    assert result["anomaly"]["cameras"][0]["camera_id"] == "cam_a"
    assert result["global"] == {
        "active_count": 1,
        "max_level": "warning",
        "history_count": 1,
    }


def test_clean_env_removes_argus_overrides(monkeypatch):
    monkeypatch.setenv("ARGUS__DASHBOARD__PORT", "1")
    monkeypatch.setenv("OTHER_ENV", "ok")

    env = _clean_env()

    assert "ARGUS__DASHBOARD__PORT" not in env
    assert env["OTHER_ENV"] == "ok"
    assert env["PYTHONUNBUFFERED"] == "1"


@pytest.mark.parametrize(
    "argv",
    [
        ["--port", "-1"],
        ["--port", "65536"],
        ["--timeout", "0"],
        ["--recording-timeout", "0"],
        ["--training-timeout", "0"],
        ["--training-mode", "unknown"],
        ["--training-baseline-count", "29"],
        ["--training-image-size", "63"],
        ["--camera-protocol", "usb"],
        ["--camera-source", "0", "--disable-go2rtc", "--require-go2rtc"],
        ["--rtsp-fixture", "--camera-source", "0"],
        ["--rtsp-fixture", "--camera-protocol", "usb"],
        ["--rtsp-fixture-seconds", "0"],
        ["--camera-resolution", "bad"],
        ["--dev-video-motion", "unknown"],
        ["--activation-delay", "-1"],
        ["--preflight-timeout", "0"],
        ["--preflight-measure-seconds", "0"],
        ["--min-frames", "0"],
        ["--browser-timeout", "0"],
        ["--browser-virtual-time-ms", "0"],
    ],
)
def test_parse_args_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
