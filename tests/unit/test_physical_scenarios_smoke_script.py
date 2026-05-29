from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from scripts.smoke_physical_scenarios import (
    build_business_command,
    build_preflight_command,
    _child_summary,
    _extract_last_json_object,
    _write_summary,
    parse_args,
    run_physical_smoke,
    run_preflight,
    run_scenario,
)


def test_parse_args_defaults_to_obsbot_book_scenario() -> None:
    args = parse_args(["--dry-run"])

    assert args.scenario == "book"
    assert args.camera_source == "0"
    assert args.camera_protocol == "usb"
    assert args.camera_resolution == "1920,1080"
    assert args.usb_device_name == "OBSBOT Meet 2 StreamCamera"
    assert args.work_dir is None
    assert args.keep_work_dir is False
    assert args.require_go2rtc is True
    assert args.browser == "required"
    assert args.preflight is True
    assert args.preflight_timeout == 15.0
    assert args.preflight_measure_seconds == 15.0
    assert args.no_alert_observe_seconds == 30.0
    assert args.stream_output is True


def test_parse_args_can_disable_streaming_for_captured_logs() -> None:
    args = parse_args(["--no-stream-output", "--dry-run"])

    assert args.stream_output is False


def test_parse_args_can_disable_preflight_for_debugging() -> None:
    args = parse_args(["--no-preflight", "--dry-run"])

    result = run_physical_smoke(args)

    assert args.preflight is False
    assert "preflight" not in result


def test_preflight_only_runs_no_physical_scenarios() -> None:
    args = parse_args(["--preflight-only", "--dry-run"])

    result = run_physical_smoke(args)

    assert result["ok"] is True
    assert result["preflight_only"] is True
    assert result["preflight"]["dry_run"] is True
    assert result["scenarios"] == []


def test_build_book_command_includes_static_scene_expectations() -> None:
    args = parse_args(["--scenario", "book", "--dry-run"])

    command = build_business_command(args, "book")

    assert "--expect-alert-category" in command
    assert "scene_change" in command
    assert "static_foreign" in command
    assert "--expect-detection-type" in command
    assert "anomaly" in command
    assert "--forbid-alert-category" in command
    assert "--forbid-detection-type" in command
    assert command.count("projectile") == 2


def test_build_projectile_command_includes_projectile_expectations() -> None:
    args = parse_args(["--scenario", "projectile", "--dry-run"])

    command = build_business_command(args, "projectile")

    assert "--expect-alert-category" in command
    assert "--expect-detection-type" in command
    assert command.count("projectile") == 2
    assert "--forbid-alert-category" not in command


def test_build_stable_command_includes_no_alert_expectation() -> None:
    args = parse_args([
        "--scenario",
        "stable",
        "--no-alert-observe-seconds",
        "12.5",
        "--dry-run",
    ])

    command = build_business_command(args, "stable")

    assert "--expect-no-alert" in command
    assert "--no-alert-observe-seconds" in command
    assert "12.5" in command
    assert "--expect-alert-category" not in command
    assert "--expect-detection-type" not in command


def test_build_preflight_command_includes_probe_options_without_semantics() -> None:
    args = parse_args([
        "--scenario",
        "book",
        "--preflight-timeout",
        "12",
        "--preflight-measure-seconds",
        "18",
        "--dry-run",
    ])

    command = build_preflight_command(args)

    assert "--preflight" in command
    assert "--preflight-timeout" in command
    assert "12.0" in command
    assert "--preflight-measure-seconds" in command
    assert "18.0" in command
    assert "--expect-alert-category" not in command
    assert "--expect-detection-type" not in command
    assert "--forbid-alert-category" not in command


def test_work_dir_is_split_by_child_smoke(tmp_path) -> None:
    args = parse_args([
        "--scenario",
        "all",
        "--work-dir",
        str(tmp_path),
        "--keep-work-dir",
        "--dry-run",
    ])

    preflight = build_preflight_command(args)
    stable = build_business_command(args, "stable")
    book = build_business_command(args, "book")
    projectile = build_business_command(args, "projectile")

    assert "--work-dir" in preflight
    assert str(tmp_path / "preflight") in preflight
    assert str(tmp_path / "stable") in stable
    assert str(tmp_path / "book") in book
    assert str(tmp_path / "projectile") in projectile
    assert "--keep-work-dir" in preflight
    assert "--keep-work-dir" in stable
    assert "--keep-work-dir" in book
    assert "--keep-work-dir" in projectile


def test_summary_defaults_to_work_dir(tmp_path) -> None:
    args = parse_args([
        "--work-dir",
        str(tmp_path),
        "--dry-run",
    ])
    result = run_physical_smoke(args)

    path = _write_summary(args, result)

    assert path == tmp_path / "physical_smoke_summary.json"
    assert result["summary_path"] == str(path)
    assert path is not None
    data = path.read_text(encoding="utf-8")
    assert '"summary_path"' in data
    assert '"scenarios"' in data


def test_summary_path_can_be_explicit(tmp_path) -> None:
    summary_path = tmp_path / "nested" / "summary.json"
    args = parse_args([
        "--summary-path",
        str(summary_path),
        "--dry-run",
    ])
    result = run_physical_smoke(args)

    path = _write_summary(args, result)

    assert path == summary_path
    assert summary_path.exists()


def test_extract_last_json_object_ignores_logs_and_trailing_noise() -> None:
    text = 'log before\n{"ok": false, "old": true}\n{"ok": true, "value": 3}\n[rtsp] warning\n'

    result = _extract_last_json_object(text)

    assert result == {"ok": True, "value": 3}


def test_extract_last_json_object_prefers_outer_object_with_nested_ok() -> None:
    text = (
        "camera log\n"
        '{"ok": true, "mode": "preflight", '
        '"capture_probe": {"ok": true, "measured_fps": 61.1}, '
        '"go2rtc": {"running": true}}\n'
        "[rtsp] warning after json\n"
    )

    result = _extract_last_json_object(text)

    assert result == {
        "ok": True,
        "mode": "preflight",
        "capture_probe": {"ok": True, "measured_fps": 61.1},
        "go2rtc": {"running": True},
    }


def test_streaming_command_keeps_bounded_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_args(["--scenario", "book", "--process-timeout", "5", "--log-tail-chars", "32"])
    monkeypatch.setattr("scripts.smoke_physical_scenarios.CAPTURE_TAIL_CHARS", 256)
    large_stdout = "x" * 384 + '{"ok": true, "value": 9}\n'
    captured = {}

    class FakePipe:
        def __iter__(self):
            yield large_stdout

        def close(self):
            pass

    class FakeProcess:
        stdout = FakePipe()
        stderr = FakePipe()

        def wait(self, timeout):
            captured["timeout"] = timeout
            return 0

    monkeypatch.setattr(
        "scripts.smoke_physical_scenarios.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )

    result = run_scenario(args, "book")

    assert result["ok"] is True
    assert len(result["stdout_tail"]) == 32
    assert result["evidence"]["ok"] is True
    assert result["evidence"]["alert"]["category"] is None
    assert captured["timeout"] == 5


def test_child_summary_extracts_preflight_fps_and_go2rtc() -> None:
    summary = _child_summary({
        "ok": True,
        "mode": "preflight",
        "work_dir": "C:/tmp/argus-physical/preflight",
        "config": "configs/default.yaml",
        "camera_input": {
            "camera_id": "c",
            "protocol": "usb",
            "probe_protocol": "rtsp",
            "resolution": [1920, 1080],
            "preflight_measure_seconds": 15.0,
            "requested_preflight_measure_seconds": 8.0,
            "effective_preflight_measure_seconds": 15.0,
            "usb_selection": {"selection_mode": "explicit_device_id_or_name"},
        },
        "capture_probe": {
            "ok": True,
            "backend": "ffmpeg",
            "shape": [1080, 1920, 3],
            "measured_fps": 61.1,
            "measured_frames": 917,
            "measured_elapsed_seconds": 15.0,
        },
        "go2rtc": {
            "enabled": True,
            "required": True,
            "running": True,
            "resolutions": {"c": {"go2rtc_managed": True}},
        },
        "hints": [],
        "errors": [],
    })

    assert summary is not None
    assert summary["mode"] == "preflight"
    assert summary["work_dir"] == "C:/tmp/argus-physical/preflight"
    assert summary["config"] == "configs/default.yaml"
    assert summary["camera_input"]["resolution"] == [1920, 1080]
    assert summary["camera_input"]["requested_preflight_measure_seconds"] == 8.0
    assert summary["camera_input"]["effective_preflight_measure_seconds"] == 15.0
    assert summary["capture_probe"]["measured_fps"] == 61.1
    assert summary["go2rtc"]["running"] is True


def test_child_summary_extracts_scenario_semantics_and_browser_routes() -> None:
    summary = _child_summary({
        "ok": True,
        "base_url": "http://127.0.0.1:18080",
        "work_dir": "C:/tmp/argus-physical/projectile",
        "runtime_config": "C:/tmp/argus-physical/projectile/dashboard_business_config.yaml",
        "camera": {"camera_id": "c", "pipeline_mode": "active"},
        "camera_media": {"streaming": {"go2rtc": True}},
        "alert": {
            "alert_id": "ALT-1",
            "severity": "high",
            "recording_status": "complete",
            "snapshot_path": "C:/tmp/argus-physical/projectile/alerts/ALT-1_snapshot.jpg",
            "heatmap_path": "C:/tmp/argus-physical/projectile/alerts/ALT-1_heatmap.jpg",
            "realtime": {
                "detection_type": "projectile",
                "category": "projectile",
                "speed_px_per_sec": 1020.0,
                "trajectory_model": "projectile",
            },
        },
        "api": {
            "evidence_zip_bytes": 4096,
            "replay": {"frame_count": 32, "signal_points": 32},
        },
        "alert_semantics": {"detection_type": "projectile", "category": "projectile"},
        "browser": {"status": "checked", "routes_checked": [{"route": "/alerts?id=ALT-1"}]},
        "objective_checklist": [{"requirement": "Alerts realtime display", "passed": True}],
    })

    assert summary is not None
    assert summary["work_dir"] == "C:/tmp/argus-physical/projectile"
    assert summary["runtime_config"].endswith("dashboard_business_config.yaml")
    assert summary["alert"]["detection_type"] == "projectile"
    assert summary["alert"]["speed_px_per_sec"] == 1020.0
    assert summary["alert"]["snapshot_path"].endswith("ALT-1_snapshot.jpg")
    assert summary["alert"]["heatmap_path"].endswith("ALT-1_heatmap.jpg")
    assert summary["api"]["evidence_zip_bytes"] == 4096
    assert summary["api"]["replay"]["frame_count"] == 32
    assert summary["browser"]["routes_checked"] == ["/alerts?id=ALT-1"]
    assert summary["objective_checklist"] == [
        {"requirement": "Alerts realtime display", "passed": True}
    ]


def test_child_summary_extracts_no_alert_scenario() -> None:
    summary = _child_summary({
        "ok": True,
        "mode": "no_alert",
        "base_url": "http://127.0.0.1:18080",
        "work_dir": "C:/tmp/argus-physical/stable",
        "runtime_config": "C:/tmp/argus-physical/stable/dashboard_business_config.yaml",
        "camera": {"camera_id": "c", "pipeline_mode": "active", "frames_captured": 125},
        "camera_media": {"streaming": {"go2rtc": True}},
        "no_alert": {"observed_seconds": 30.0, "polls": 31, "alerts_seen": 0},
        "browser": {"status": "not_applicable"},
    })

    assert summary is not None
    assert summary["mode"] == "no_alert"
    assert summary["work_dir"] == "C:/tmp/argus-physical/stable"
    assert summary["no_alert"]["alerts_seen"] == 0
    assert summary["camera_streaming"]["go2rtc"] is True


def test_parse_args_can_disable_go2rtc_for_debugging() -> None:
    args = parse_args(["--no-require-go2rtc", "--dry-run"])

    command = build_business_command(args, "book")

    assert args.require_go2rtc is False
    assert "--require-go2rtc" not in command


@pytest.mark.parametrize(
    "argv",
    [
        ["--activation-delay", "-1"],
        ["--timeout", "0"],
        ["--no-alert-observe-seconds", "0"],
        ["--process-timeout", "0"],
        ["--no-preflight", "--preflight-only"],
    ],
)
def test_parse_args_rejects_invalid_values(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_dry_run_scenario_returns_command_without_subprocess() -> None:
    args = parse_args(["--scenario", "book", "--dry-run"])

    result = run_scenario(args, "book")

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["scenario"] == "book"
    assert "place a book" in result["instruction"]
    assert "static_foreign" in result["instruction"]
    assert "smoke_dashboard_business_flow.py" in " ".join(result["command"])


def test_run_all_dry_run_returns_both_scenarios() -> None:
    args = parse_args(["--scenario", "all", "--dry-run"])

    result = run_physical_smoke(args)

    assert result["ok"] is True
    assert result["preflight"]["ok"] is True
    assert result["preflight"]["dry_run"] is True
    assert [item["scenario"] for item in result["scenarios"]] == ["stable", "book", "projectile"]


def test_preflight_failure_skips_physical_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_args(["--scenario", "book", "--process-timeout", "5"])

    monkeypatch.setattr(
        "scripts.smoke_physical_scenarios.run_preflight",
        lambda _args: {"ok": False, "returncode": 7},
    )
    monkeypatch.setattr(
        "scripts.smoke_physical_scenarios.run_scenario",
        lambda *_args: pytest.fail("scenario should not run after preflight failure"),
    )

    result = run_physical_smoke(args)

    assert result["ok"] is False
    assert result["preflight"]["returncode"] == 7
    assert result["scenarios"] == []


def test_run_preflight_extracts_outer_evidence_with_nested_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_args(["--scenario", "book", "--process-timeout", "5"])

    def fake_stream(command, *, timeout_s, log_tail_chars):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "camera log\n"
                '{"ok": true, "mode": "preflight", '
                '"camera_input": {"resolution": [1920, 1080]}, '
                '"capture_probe": {"ok": true, "measured_fps": 61.1}, '
                '"go2rtc": {"running": true}}\n'
                "[rtsp] warning after json\n"
            ),
            stderr="",
            timed_out=False,
        )

    monkeypatch.setattr("scripts.smoke_physical_scenarios._run_streaming_command", fake_stream)

    result = run_preflight(args)

    assert result["ok"] is True
    assert result["evidence"]["mode"] == "preflight"
    assert result["evidence"]["camera_input"]["resolution"] == [1920, 1080]
    assert result["evidence"]["capture_probe"]["measured_fps"] == 61.1
    assert result["evidence"]["go2rtc"]["running"] is True


def test_run_scenario_streams_subprocess_output_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_args(["--scenario", "book", "--process-timeout", "5"])
    captured = {}

    def fake_stream(command, *, timeout_s, log_tail_chars):
        captured["command"] = command
        captured["timeout_s"] = timeout_s
        captured["log_tail_chars"] = log_tail_chars
        return SimpleNamespace(
            returncode=0,
            stdout='live output\n{"ok": true, "alert_semantics": {"category": "scene_change"}}',
            stderr="",
            timed_out=False,
        )

    monkeypatch.setattr("scripts.smoke_physical_scenarios._run_streaming_command", fake_stream)

    result = run_scenario(args, "book")

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert "live output" in result["stdout_tail"]
    assert result["evidence"]["alert_semantics"]["category"] == "scene_change"
    assert captured["timeout_s"] == 5
    assert "smoke_dashboard_business_flow.py" in " ".join(captured["command"])


def test_run_scenario_records_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_args(["--scenario", "book", "--process-timeout", "5", "--no-stream-output"])

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=7, stdout="x" * 20, stderr="boom")

    monkeypatch.setattr("scripts.smoke_physical_scenarios.subprocess.run", fake_run)

    result = run_scenario(args, "book")

    assert result["ok"] is False
    assert result["returncode"] == 7
    assert result["timed_out"] is False
    assert result["stderr_tail"] == "boom"


def test_run_scenario_records_non_streaming_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_args(["--scenario", "book", "--process-timeout", "5", "--no-stream-output"])

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["business"],
            timeout=5,
            output="partial stdout",
            stderr=b"late stderr",
        )

    monkeypatch.setattr("scripts.smoke_physical_scenarios.subprocess.run", fake_run)

    result = run_scenario(args, "book")

    assert result["ok"] is False
    assert result["returncode"] is None
    assert result["timed_out"] is True
    assert result["stdout_tail"] == "partial stdout"
    assert result["stderr_tail"] == "late stderr"
