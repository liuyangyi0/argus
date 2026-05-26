from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

from scripts.smoke_core_loop import (
    _infer_camera_protocol,
    _inspect_usb_video_devices,
    _inspect_windows_camera_privacy,
    _inspect_windows_dshow_devices,
    _opencv_runtime_info,
    _parse_resolution,
    _prepare_config,
    parse_args,
    run_preflight,
)


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


def test_prepare_config_default_uses_generated_file_camera(tmp_path):
    video_path = tmp_path / "dev_camera.avi"
    config = _prepare_config(
        Path("configs/default.yaml"),
        tmp_path,
        video_path,
        use_yolo=False,
    )

    camera = config.cameras[0]
    assert camera.camera_id == "dev_cam"
    assert camera.protocol == "file"
    assert camera.source == str(video_path)
    assert camera.fps_target == 10
    assert config.dashboard.go2rtc_enabled is False
    assert config.storage.database_url == f"sqlite:///{tmp_path / 'argus.db'}"
    assert camera.person_filter.model_name == str(tmp_path / "missing-yolo.pt")


def test_prepare_config_can_use_usb_hardware_with_go2rtc(monkeypatch, tmp_path):
    ports = iter([21984, 28554, 28555])
    monkeypatch.setattr("scripts.smoke_core_loop._free_port", lambda: next(ports))

    config = _prepare_config(
        Path("configs/default.yaml"),
        tmp_path,
        tmp_path / "unused.avi",
        use_yolo=False,
        camera_source="0",
        camera_protocol="usb",
        camera_id="usb_smoke",
        camera_name="USB smoke",
        camera_resolution=(1280, 720),
        go2rtc_enabled=True,
    )

    camera = config.cameras[0]
    assert camera.camera_id == "usb_smoke"
    assert camera.name == "USB smoke"
    assert camera.protocol == "usb"
    assert camera.source == "0"
    assert camera.resolution == (1280, 720)
    assert camera.fps_target == 60
    assert config.dashboard.go2rtc_enabled is True
    assert config.dashboard.go2rtc_api_port == 21984
    assert config.dashboard.go2rtc_rtsp_port == 28554
    assert config.dashboard.go2rtc_webrtc_port == 28555


def test_prepare_config_accepts_stable_usb_device_selector(tmp_path):
    config = _prepare_config(
        Path("configs/default.yaml"),
        tmp_path,
        tmp_path / "unused.avi",
        use_yolo=False,
        camera_source="0",
        camera_protocol="usb",
        camera_resolution=(1920, 1080),
        usb_device_name="OBSBOT Meet 2 StreamCamera",
        usb_device_id="@device_pnp_usb_vid_3564_pid_3022",
        go2rtc_enabled=False,
    )

    camera = config.cameras[0]
    assert camera.usb.device_name == "OBSBOT Meet 2 StreamCamera"
    assert camera.usb.device_id == "@device_pnp_usb_vid_3564_pid_3022"


def test_parse_args_rejects_conflicting_go2rtc_flags():
    with pytest.raises(SystemExit):
        parse_args(["--camera-source", "0", "--disable-go2rtc", "--require-go2rtc"])


def test_parse_args_requires_source_when_protocol_is_set():
    with pytest.raises(SystemExit):
        parse_args(["--camera-protocol", "usb"])


def test_parse_args_accepts_hardware_activation_delay():
    args = parse_args([
        "--camera-source",
        "0",
        "--camera-protocol",
        "usb",
        "--activation-delay",
        "10",
    ])

    assert args.activation_delay == 10.0


def test_parse_args_rejects_negative_activation_delay():
    with pytest.raises(SystemExit):
        parse_args(["--activation-delay", "-1"])


def test_parse_args_accepts_preflight_mode():
    args = parse_args([
        "--preflight",
        "--camera-source",
        "0",
        "--camera-protocol",
        "usb",
        "--preflight-timeout",
        "1.5",
        "--preflight-measure-seconds",
        "4.5",
    ])

    assert args.preflight is True
    assert args.preflight_timeout == 1.5
    assert args.preflight_measure_seconds == 4.5


def test_parse_args_accepts_book_dev_video_motion():
    args = parse_args(["--dev-video-motion", "book"])

    assert args.dev_video_motion == "book"


def test_parse_args_accepts_usb_device_selector():
    args = parse_args([
        "--camera-source",
        "0",
        "--camera-protocol",
        "usb",
        "--usb-device-name",
        "OBSBOT Meet 2 StreamCamera",
        "--usb-device-id",
        "@device_pnp_usb_vid_3564_pid_3022",
    ])

    assert args.usb_device_name == "OBSBOT Meet 2 StreamCamera"
    assert args.usb_device_id == "@device_pnp_usb_vid_3564_pid_3022"


def test_parse_args_rejects_invalid_preflight_timeout():
    with pytest.raises(SystemExit):
        parse_args(["--preflight", "--preflight-timeout", "0"])


def test_parse_args_rejects_invalid_preflight_measure_seconds():
    with pytest.raises(SystemExit):
        parse_args(["--preflight", "--preflight-measure-seconds", "0"])


def test_inspect_usb_video_devices_reports_windows_inventory(monkeypatch):
    calls = []

    class FakeResult:
        returncode = 0
        stdout = (
            '[{"Name":"USB Camera","PNPClass":"Camera","Status":"OK",'
            '"Manufacturer":"Acme","DeviceID":"USB\\\\VID_123"}]'
        )
        stderr = ""

    def fake_run(command, *, capture_output, text, timeout):
        calls.append(
            {
                "command": command,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return FakeResult()

    monkeypatch.setattr("scripts.smoke_core_loop.platform.system", lambda: "Windows")
    monkeypatch.setattr("scripts.smoke_core_loop.subprocess.run", fake_run)

    result = _inspect_usb_video_devices(timeout_s=1.25)

    assert result["platform"] == "Windows"
    assert result["supported"] is True
    assert result["source"] == "PnpDevice"
    assert result["devices"] == [
        {
            "name": "USB Camera",
            "pnp_class": "Camera",
            "status": "OK",
            "manufacturer": "Acme",
            "device_id": "USB\\VID_123",
        }
    ]
    assert result["device_count"] == 1
    assert calls[0]["command"][:3] == ["powershell.exe", "-NoProfile", "-Command"]
    assert calls[0]["timeout"] == 1.25


def test_inspect_usb_video_devices_is_best_effort_off_windows(monkeypatch):
    monkeypatch.setattr("scripts.smoke_core_loop.platform.system", lambda: "Linux")

    result = _inspect_usb_video_devices()

    assert result["platform"] == "Linux"
    assert result["supported"] is False
    assert result["devices"] == []


def test_opencv_runtime_info_warns_on_gui_and_headless_wheel_conflict(monkeypatch):
    installed = {
        "opencv-python": "4.13.0.92",
        "opencv-python-headless": "4.13.0.92",
    }

    def fake_version(package_name):
        if package_name in installed:
            return installed[package_name]
        raise importlib_metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr("scripts.smoke_core_loop.importlib_metadata.version", fake_version)

    result = _opencv_runtime_info()

    assert result["installed_packages"] == installed
    assert any("headless OpenCV" in warning for warning in result["warnings"])


def test_inspect_windows_camera_privacy_reports_registry_entries(monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = '[{"Scope":"HKCU","Value":"Allow","LastUsedTimeStart":null,"LastUsedTimeStop":null}]'
        stderr = ""

    monkeypatch.setattr("scripts.smoke_core_loop.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "scripts.smoke_core_loop.subprocess.run",
        lambda *args, **kwargs: FakeResult(),
    )

    result = _inspect_windows_camera_privacy()

    assert result["entries"] == [
        {
            "scope": "HKCU",
            "value": "Allow",
            "last_used_start": None,
            "last_used_stop": None,
        }
    ]


def test_inspect_windows_dshow_devices_reports_enumeration_failure(monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = (
            '[dshow @ 000] Could not enumerate video devices (or none found).\n'
            '[dshow @ 000] "Microphone" (audio)\n'
            '[dshow @ 000]   Alternative name "@device_cm_audio"'
        )

    monkeypatch.setattr("scripts.smoke_core_loop.platform.system", lambda: "Windows")
    monkeypatch.setattr("scripts.smoke_core_loop.shutil.which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(
        "scripts.smoke_core_loop.subprocess.run",
        lambda *args, **kwargs: FakeResult(),
    )

    result = _inspect_windows_dshow_devices()

    assert result["video_enumeration_failed"] is True
    assert result["video_device_count"] == 0
    assert result["devices"] == [
        {
            "name": "Microphone",
            "kind": "audio",
            "alternative_name": "@device_cm_audio",
        }
    ]


def test_inspect_windows_dshow_devices_skips_conda_ffmpeg(monkeypatch):
    monkeypatch.setattr("scripts.smoke_core_loop.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "scripts.smoke_core_loop.shutil.which",
        lambda name: r"C:\Users\here\miniconda3\Library\bin\ffmpeg.exe",
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("conda ffmpeg should not be invoked for DShow diagnostics")

    monkeypatch.setattr("scripts.smoke_core_loop.subprocess.run", fail_run)

    result = _inspect_windows_dshow_devices()

    assert result["supported"] is False
    assert result["binary"] is None
    assert result["candidate_binary"].endswith("ffmpeg.exe")
    assert result["binary_source"] == "path_conda_skipped"
    assert result["diagnostic_only"] is True
    assert any("conda/miniconda" in warning for warning in result["warnings"])


def test_run_preflight_reports_capture_success(monkeypatch, tmp_path):
    captured_probe: dict[str, float | int | str] = {}

    def fake_probe(source, protocol, *, timeout_ms, measure_seconds):
        captured_probe.update({
            "source": source,
            "protocol": protocol,
            "timeout_ms": timeout_ms,
            "measure_seconds": measure_seconds,
        })
        return {
            "ok": True,
            "source": source,
            "protocol": protocol,
            "backend": "fake",
            "shape": [480, 640, 3],
            "attempts": [],
        }

    monkeypatch.setattr("scripts.smoke_core_loop._probe_capture_source", fake_probe)
    args = parse_args([
        "--preflight",
        "--work-dir",
        str(tmp_path),
        "--camera-source",
        "data/dev/fake.avi",
        "--camera-protocol",
        "file",
        "--disable-go2rtc",
        "--preflight-measure-seconds",
        "7.5",
    ])

    result = run_preflight(args)

    assert result["ok"] is True
    assert result["mode"] == "preflight"
    assert result["camera_input"]["probe_protocol"] == "file"
    assert result["usb_devices"] is None
    assert result["windows_camera_privacy"] is None
    assert result["dshow_devices"] is None
    assert result["capture_probe"]["backend"] == "fake"
    assert captured_probe["measure_seconds"] == 7.5
    assert {item["component"] for item in result["expected_degradations"]} == {
        "person_filter",
        "anomaly_detector",
    }
    assert result["hints"] == []


def test_run_preflight_generates_book_dev_video(monkeypatch, tmp_path):
    captured_video: dict[str, object] = {}

    def fake_create_dev_video(output, **kwargs):
        captured_video.update({"output": output, **kwargs})
        return {
            "output": str(output),
            "width": kwargs["width"],
            "height": kwargs["height"],
            "fps": kwargs["fps"],
            "frames": kwargs["fps"] * kwargs["seconds"],
            "anomaly_start_frame": int(kwargs["fps"] * kwargs["anomaly_start_s"]),
            "motion": kwargs["motion"],
        }

    def fake_probe(source, protocol, *, timeout_ms, measure_seconds):
        return {
            "ok": True,
            "source": source,
            "protocol": protocol,
            "backend": "fake",
            "shape": [480, 640, 3],
            "attempts": [],
        }

    monkeypatch.setattr("scripts.smoke_core_loop.create_dev_video", fake_create_dev_video)
    monkeypatch.setattr("scripts.smoke_core_loop._probe_capture_source", fake_probe)
    args = parse_args([
        "--preflight",
        "--work-dir",
        str(tmp_path),
        "--disable-go2rtc",
        "--dev-video-motion",
        "book",
    ])

    result = run_preflight(args)

    assert result["ok"] is True
    assert captured_video["motion"] == "book"
    assert captured_video["output"] == tmp_path / "dev_camera.avi"


def test_run_preflight_reports_numeric_usb_index_mapping(monkeypatch, tmp_path):
    def fake_probe(source, protocol, *, timeout_ms, measure_seconds):
        return {
            "ok": True,
            "source": source,
            "protocol": protocol,
            "backend": "fake",
            "shape": [1080, 1920, 3],
            "measured_fps": 60.0,
            "attempts": [],
        }

    monkeypatch.setattr("scripts.smoke_core_loop._probe_capture_source", fake_probe)
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_usb_video_devices",
        lambda timeout_s: {
            "platform": "Windows",
            "supported": True,
            "source": "PnpDevice",
            "devices": [
                {
                    "name": "OBSBOT Meet 2 StreamCamera",
                    "status": "OK",
                    "device_id": "USB\\VID_3564&PID_3022",
                }
            ],
            "device_count": 1,
        },
    )
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_windows_camera_privacy",
        lambda timeout_s: {"platform": "Windows", "supported": True, "entries": []},
    )
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_windows_dshow_devices",
        lambda timeout_s: {
            "platform": "Windows",
            "supported": True,
            "devices": [
                {
                    "name": "OBSBOT Meet 2 StreamCamera",
                    "kind": "video",
                    "alternative_name": "@device_pnp_usb_vid_3564_pid_3022",
                }
            ],
            "video_device_count": 1,
        },
    )
    args = parse_args([
        "--preflight",
        "--work-dir",
        str(tmp_path),
        "--camera-source",
        "0",
        "--camera-protocol",
        "usb",
        "--disable-go2rtc",
    ])

    result = run_preflight(args)

    selection = result["camera_input"]["usb_selection"]
    assert result["ok"] is True
    assert selection["selection_mode"] == "numeric_index"
    assert selection["selected_pnp_device"]["name"] == "OBSBOT Meet 2 StreamCamera"
    assert selection["selected_dshow_device"]["alternative_name"] == (
        "@device_pnp_usb_vid_3564_pid_3022"
    )
    assert any("usb.device_name" in hint for hint in result["hints"])


def test_run_preflight_reports_explicit_usb_device_name_match(monkeypatch, tmp_path):
    def fake_probe(source, protocol, *, timeout_ms, measure_seconds):
        return {
            "ok": True,
            "source": source,
            "protocol": protocol,
            "backend": "fake",
            "shape": [1080, 1920, 3],
            "measured_fps": 60.0,
            "attempts": [],
        }

    monkeypatch.setattr("scripts.smoke_core_loop._probe_capture_source", fake_probe)
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_usb_video_devices",
        lambda timeout_s: {
            "platform": "Windows",
            "supported": True,
            "source": "PnpDevice",
            "devices": [
                {
                    "name": "OBSBOT Meet 2 StreamCamera",
                    "status": "Unknown",
                    "device_id": "USB\\VID_3564&PID_FEFB\\OLD",
                },
                {
                    "name": "OBSBOT Meet 2 StreamCamera",
                    "status": "OK",
                    "device_id": "USB\\VID_3564&PID_FEFB\\CURRENT",
                },
            ],
            "device_count": 2,
        },
    )
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_windows_camera_privacy",
        lambda timeout_s: {"platform": "Windows", "supported": True, "entries": []},
    )
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_windows_dshow_devices",
        lambda timeout_s: {
            "platform": "Windows",
            "supported": True,
            "devices": [
                {
                    "name": "OBSBOT Meet 2 StreamCamera",
                    "kind": "video",
                    "alternative_name": "@device_pnp_usb_vid_3564_pid_fefb",
                }
            ],
            "video_device_count": 1,
        },
    )
    args = parse_args([
        "--preflight",
        "--work-dir",
        str(tmp_path),
        "--camera-source",
        "0",
        "--camera-protocol",
        "usb",
        "--usb-device-name",
        "OBSBOT Meet 2 StreamCamera",
        "--disable-go2rtc",
    ])

    result = run_preflight(args)

    selection = result["camera_input"]["usb_selection"]
    assert result["ok"] is True
    assert selection["selection_mode"] == "explicit_device_id_or_name"
    assert selection["selected_pnp_device"]["device_id"].endswith("CURRENT")
    assert selection["selected_dshow_device"]["alternative_name"] == (
        "@device_pnp_usb_vid_3564_pid_fefb"
    )
    assert selection["warnings"] == []


def test_run_preflight_reports_capture_failure(monkeypatch, tmp_path):
    def fake_probe(source, protocol, *, timeout_ms, measure_seconds):
        return {
            "ok": False,
            "source": source,
            "protocol": protocol,
            "attempts": [{"backend": "fake", "opened": False, "frame_ok": False}],
        }

    monkeypatch.setattr("scripts.smoke_core_loop._probe_capture_source", fake_probe)
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_usb_video_devices",
        lambda timeout_s: {
            "platform": "Windows",
            "supported": True,
            "source": "PnpDevice",
            "devices": [{"name": "USB Camera", "status": "OK"}],
        },
    )
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_windows_camera_privacy",
        lambda timeout_s: {
            "platform": "Windows",
            "supported": True,
            "entries": [{"scope": "HKCU", "value": "Allow"}],
        },
    )
    monkeypatch.setattr(
        "scripts.smoke_core_loop._inspect_windows_dshow_devices",
        lambda timeout_s: {
            "platform": "Windows",
            "supported": True,
            "video_enumeration_failed": True,
            "video_device_count": 0,
            "devices": [],
        },
    )
    args = parse_args([
        "--preflight",
        "--work-dir",
        str(tmp_path),
        "--camera-source",
        "0",
        "--camera-protocol",
        "usb",
        "--disable-go2rtc",
    ])

    result = run_preflight(args)

    assert result["ok"] is False
    assert result["usb_devices"]["devices"][0]["name"] == "USB Camera"
    assert "Windows enumerated USB camera devices" in result["hints"][0]
    assert any("DirectShow/FFmpeg" in hint for hint in result["hints"])
    assert "did not produce a readable frame" in result["errors"][0]


def test_run_preflight_fails_when_fast_motion_fps_is_below_requirement(
    monkeypatch, tmp_path,
):
    def fake_probe(source, protocol, *, timeout_ms, measure_seconds):
        return {
            "ok": True,
            "source": source,
            "protocol": protocol,
            "backend": "fake",
            "shape": [1080, 1920, 3],
            "measured_fps": 35.0,
            "attempts": [],
        }

    monkeypatch.setattr("scripts.smoke_core_loop._probe_capture_source", fake_probe)
    args = parse_args([
        "--preflight",
        "--work-dir",
        str(tmp_path),
        "--camera-source",
        "0",
        "--camera-protocol",
        "usb",
        "--disable-go2rtc",
    ])

    result = run_preflight(args)

    assert result["ok"] is False
    assert "measured 35.0fps below required 50.0fps" in result["errors"][0]
    assert any("Increase lighting" in hint for hint in result["hints"])
