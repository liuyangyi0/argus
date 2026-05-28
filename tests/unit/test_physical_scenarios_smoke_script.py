from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.smoke_physical_scenarios import (
    build_business_command,
    parse_args,
    run_physical_smoke,
    run_scenario,
)


def test_parse_args_defaults_to_obsbot_book_scenario() -> None:
    args = parse_args(["--dry-run"])

    assert args.scenario == "book"
    assert args.camera_source == "0"
    assert args.camera_protocol == "usb"
    assert args.camera_resolution == "1920,1080"
    assert args.usb_device_name == "OBSBOT Meet 2 StreamCamera"
    assert args.require_go2rtc is True
    assert args.browser == "required"
    assert args.stream_output is True


def test_parse_args_can_disable_streaming_for_captured_logs() -> None:
    args = parse_args(["--no-stream-output", "--dry-run"])

    assert args.stream_output is False


def test_build_book_command_includes_static_scene_expectations() -> None:
    args = parse_args(["--scenario", "book", "--dry-run"])

    command = build_business_command(args, "book")

    assert "--expect-alert-category" in command
    assert "scene_change" in command
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


def test_parse_args_can_disable_go2rtc_for_debugging() -> None:
    args = parse_args(["--no-require-go2rtc", "--dry-run"])

    command = build_business_command(args, "book")

    assert args.require_go2rtc is False
    assert "--require-go2rtc" not in command


@pytest.mark.parametrize("argv", [["--activation-delay", "-1"], ["--timeout", "0"], ["--process-timeout", "0"]])
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
    assert "smoke_dashboard_business_flow.py" in " ".join(result["command"])


def test_run_all_dry_run_returns_both_scenarios() -> None:
    args = parse_args(["--scenario", "all", "--dry-run"])

    result = run_physical_smoke(args)

    assert result["ok"] is True
    assert [item["scenario"] for item in result["scenarios"]] == ["book", "projectile"]


def test_run_scenario_streams_subprocess_output_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_args(["--scenario", "book", "--process-timeout", "5"])
    captured = {}

    def fake_stream(command, *, timeout_s, log_tail_chars):
        captured["command"] = command
        captured["timeout_s"] = timeout_s
        captured["log_tail_chars"] = log_tail_chars
        return SimpleNamespace(returncode=0, stdout="live output", stderr="", timed_out=False)

    monkeypatch.setattr("scripts.smoke_physical_scenarios._run_streaming_command", fake_stream)

    result = run_scenario(args, "book")

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["stdout_tail"] == "live output"
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
