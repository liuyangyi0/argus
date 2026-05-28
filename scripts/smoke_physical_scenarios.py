"""Guided physical-scene smoke wrapper for OBSBOT/USB hardware validation.

This script wraps ``smoke_dashboard_business_flow.py`` with the semantic
assertions needed for real-world scenarios:

* ``book``: an operator places a book on an initially empty table.  The alert
  must be a static scene-change anomaly and must not be a projectile.
* ``projectile``: an operator passes a small object quickly through the frame.
  The alert must be reported as a projectile.

The underlying business smoke still verifies camera stream, alert delivery,
Replay evidence, model/release APIs, reports, and optionally browser DOM pages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = ("book", "projectile")
CAPTURE_TAIL_CHARS = 2_000_000


class PhysicalScenarioSmokeFailure(RuntimeError):
    """Raised when a physical scenario smoke fails."""


def _child_work_dir(args: argparse.Namespace, child_name: str | None) -> Path | None:
    if args.work_dir is None:
        return None
    name = child_name or "run"
    return args.work_dir / name


@dataclass(frozen=True)
class _CommandRunResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def _base_business_command(
    args: argparse.Namespace,
    *,
    preflight: bool = False,
    child_name: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "smoke_dashboard_business_flow.py"),
        "--camera-source",
        args.camera_source,
        "--camera-protocol",
        args.camera_protocol,
        "--camera-resolution",
        args.camera_resolution,
        "--activation-delay",
        str(args.activation_delay),
        "--timeout",
        str(args.timeout),
        "--recording-timeout",
        str(args.recording_timeout),
        "--browser",
        args.browser,
    ]
    work_dir = _child_work_dir(args, child_name)
    if work_dir is not None:
        cmd.extend(["--work-dir", str(work_dir)])
    if args.keep_work_dir:
        cmd.append("--keep-work-dir")
    if args.require_go2rtc:
        cmd.append("--require-go2rtc")
    if args.usb_device_name:
        cmd.extend(["--usb-device-name", args.usb_device_name])
    if args.usb_device_id:
        cmd.extend(["--usb-device-id", args.usb_device_id])
    if preflight:
        cmd.extend([
            "--preflight",
            "--preflight-timeout",
            str(args.preflight_timeout),
            "--preflight-measure-seconds",
            str(args.preflight_measure_seconds),
        ])
    return cmd


def _scenario_expectation_args(scenario: str) -> list[str]:
    if scenario == "book":
        return [
            "--expect-alert-category",
            "scene_change",
            "--expect-detection-type",
            "anomaly",
            "--forbid-alert-category",
            "projectile",
            "--forbid-detection-type",
            "projectile",
        ]
    if scenario == "projectile":
        return [
            "--expect-alert-category",
            "projectile",
            "--expect-detection-type",
            "projectile",
        ]
    raise PhysicalScenarioSmokeFailure(f"unknown scenario: {scenario}")


def build_business_command(args: argparse.Namespace, scenario: str) -> list[str]:
    return _base_business_command(args, child_name=scenario) + _scenario_expectation_args(scenario)


def build_preflight_command(args: argparse.Namespace) -> list[str]:
    return _base_business_command(args, preflight=True, child_name="preflight")


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    best_end = -1
    best_len = -1
    idx = text.find("{")
    while idx >= 0:
        try:
            value, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        absolute_end = idx + end
        if (
            isinstance(value, dict)
            and "ok" in value
            and (absolute_end > best_end or (absolute_end == best_end and end > best_len))
        ):
            best = value
            best_end = absolute_end
            best_len = end
        idx = text.find("{", idx + 1)
    return best


def _child_summary(child: dict[str, Any] | None) -> dict[str, Any] | None:
    if not child:
        return None

    if child.get("mode") == "preflight":
        probe = child.get("capture_probe") or {}
        camera_input = child.get("camera_input") or {}
        go2rtc = child.get("go2rtc") or {}
        return {
            "mode": child.get("mode"),
            "ok": child.get("ok"),
            "camera_input": {
                "camera_id": camera_input.get("camera_id"),
                "protocol": camera_input.get("protocol"),
                "probe_protocol": camera_input.get("probe_protocol"),
                "resolution": camera_input.get("resolution"),
                "preflight_measure_seconds": camera_input.get("preflight_measure_seconds"),
                "usb_selection": camera_input.get("usb_selection"),
            },
            "capture_probe": {
                "ok": probe.get("ok"),
                "backend": probe.get("backend"),
                "shape": probe.get("shape"),
                "measured_fps": probe.get("measured_fps"),
                "measured_frames": probe.get("measured_frames"),
                "measured_elapsed_seconds": probe.get("measured_elapsed_seconds"),
            },
            "go2rtc": {
                "enabled": go2rtc.get("enabled"),
                "required": go2rtc.get("required"),
                "running": go2rtc.get("running"),
                "resolutions": go2rtc.get("resolutions"),
            },
            "hints": child.get("hints") or [],
            "errors": child.get("errors") or [],
        }

    alert = child.get("alert") or {}
    realtime = alert.get("realtime") or {}
    browser = child.get("browser") or {}
    checklist = child.get("objective_checklist") or []
    return {
        "ok": child.get("ok"),
        "base_url": child.get("base_url"),
        "camera": child.get("camera"),
        "camera_streaming": (child.get("camera_media") or {}).get("streaming"),
        "alert": {
            "alert_id": alert.get("alert_id"),
            "severity": alert.get("severity"),
            "recording_status": alert.get("recording_status"),
            "detection_type": realtime.get("detection_type"),
            "category": realtime.get("category"),
            "speed_px_per_sec": realtime.get("speed_px_per_sec"),
            "trajectory_model": realtime.get("trajectory_model"),
        },
        "alert_semantics": child.get("alert_semantics"),
        "browser": {
            "status": browser.get("status"),
            "routes_checked": [
                item.get("route")
                for item in (browser.get("routes_checked") or [])
                if item.get("route")
            ],
        },
        "objective_checklist": [
            {
                "requirement": item.get("requirement"),
                "passed": item.get("passed"),
            }
            for item in checklist
        ],
    }


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _append_tail(current: str, chunk: str, limit: int) -> str:
    return (current + chunk)[-limit:]


def _run_streaming_command(
    command: list[str],
    *,
    timeout_s: float,
    log_tail_chars: int,
) -> _CommandRunResult:
    proc = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    capture_limit = max(log_tail_chars, CAPTURE_TAIL_CHARS)
    captured = {"stdout": "", "stderr": ""}
    lock = threading.Lock()

    def reader(pipe, sink, key: str) -> None:
        if pipe is None:
            return
        try:
            for chunk in pipe:
                sink.write(chunk)
                sink.flush()
                with lock:
                    captured[key] = _append_tail(captured[key], chunk, capture_limit)
        finally:
            pipe.close()

    threads = [
        threading.Thread(target=reader, args=(proc.stdout, sys.stdout, "stdout"), daemon=True),
        threading.Thread(target=reader, args=(proc.stderr, sys.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        returncode = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        returncode = proc.wait()
    for thread in threads:
        thread.join(timeout=1.0)

    with lock:
        stdout = captured["stdout"]
        stderr = captured["stderr"]
    return _CommandRunResult(
        returncode=None if timed_out else returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


def _run_business_command(command: list[str], args: argparse.Namespace) -> _CommandRunResult:
    if args.stream_output:
        return _run_streaming_command(
            command,
            timeout_s=args.process_timeout,
            log_tail_chars=args.log_tail_chars,
        )
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=args.process_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return _CommandRunResult(
            returncode=None,
            stdout=_as_text(exc.stdout or exc.output),
            stderr=_as_text(exc.stderr),
            timed_out=True,
        )
    return _CommandRunResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _operator_instruction(scenario: str, args: argparse.Namespace) -> str:
    if scenario == "book":
        return (
            f"After activation, place a book on the empty table within "
            f"{args.activation_delay:.1f}s. Expected: scene_change/anomaly."
        )
    if scenario == "projectile":
        return (
            f"After activation, pass a small object quickly through the frame "
            f"within {args.activation_delay:.1f}s. Expected: projectile/projectile."
        )
    return f"Run scenario {scenario!r}."


def run_scenario(args: argparse.Namespace, scenario: str) -> dict[str, Any]:
    command = build_business_command(args, scenario)
    result: dict[str, Any] = {
        "scenario": scenario,
        "instruction": _operator_instruction(scenario, args),
        "command": command,
    }
    if args.dry_run:
        result["ok"] = True
        result["dry_run"] = True
        return result

    print(f"[argus] scenario={scenario}: {result['instruction']}", flush=True)
    completed = _run_business_command(command, args)
    result.update({
        "ok": completed.returncode == 0 and not completed.timed_out,
        "returncode": completed.returncode,
        "timed_out": completed.timed_out,
        "stdout_tail": completed.stdout[-args.log_tail_chars:],
        "stderr_tail": completed.stderr[-args.log_tail_chars:],
    })
    child = _extract_last_json_object(completed.stdout)
    summary = _child_summary(child)
    if summary is not None:
        result["evidence"] = summary
    return result


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    command = build_preflight_command(args)
    result: dict[str, Any] = {
        "instruction": "Verify camera capture, go2rtc registration, resolution, and FPS before physical actions.",
        "command": command,
    }
    if args.dry_run:
        result["ok"] = True
        result["dry_run"] = True
        return result

    print(f"[argus] preflight: {result['instruction']}", flush=True)
    completed = _run_business_command(command, args)
    result.update({
        "ok": completed.returncode == 0 and not completed.timed_out,
        "returncode": completed.returncode,
        "timed_out": completed.timed_out,
        "stdout_tail": completed.stdout[-args.log_tail_chars:],
        "stderr_tail": completed.stderr[-args.log_tail_chars:],
    })
    child = _extract_last_json_object(completed.stdout)
    summary = _child_summary(child)
    if summary is not None:
        result["evidence"] = summary
    return result


def run_physical_smoke(args: argparse.Namespace) -> dict[str, Any]:
    preflight_result = run_preflight(args) if args.preflight else None
    if preflight_result is not None and not preflight_result.get("ok"):
        return {
            "ok": False,
            "preflight": preflight_result,
            "scenarios": [],
        }

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = [run_scenario(args, scenario) for scenario in scenarios]
    output = {"ok": all(item.get("ok") for item in results)}
    if preflight_result is not None:
        output["preflight"] = preflight_result
    output["scenarios"] = results
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guided Argus physical scenario smoke")
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        default="book",
        help="Physical validation scenario to run.",
    )
    parser.add_argument("--camera-source", default="0")
    parser.add_argument("--camera-protocol", choices=["usb", "rtsp", "file"], default="usb")
    parser.add_argument("--camera-resolution", default="1920,1080")
    parser.add_argument("--usb-device-name", default="OBSBOT Meet 2 StreamCamera")
    parser.add_argument("--usb-device-id", default=None)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional root directory for child smoke evidence; uses preflight/book/projectile subdirectories.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Pass --keep-work-dir to child business smokes so temporary evidence directories are retained.",
    )
    parser.add_argument("--require-go2rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--activation-delay",
        type=float,
        default=10.0,
        help="Seconds after activation for the operator to introduce the physical target.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--recording-timeout", type=float, default=90.0)
    parser.add_argument("--browser", choices=["auto", "required", "off"], default="required")
    parser.add_argument("--process-timeout", type=float, default=300.0)
    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a camera/go2rtc/FPS preflight before prompting for physical actions.",
    )
    parser.add_argument("--preflight-timeout", type=float, default=8.0)
    parser.add_argument("--preflight-measure-seconds", type=float, default=15.0)
    parser.add_argument("--log-tail-chars", type=int, default=12000)
    parser.add_argument(
        "--stream-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the underlying business smoke output live; use --no-stream-output for captured logs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned command without running it.")
    args = parser.parse_args(argv)
    if args.activation_delay < 0:
        parser.error("--activation-delay must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.recording_timeout <= 0:
        parser.error("--recording-timeout must be positive")
    if args.process_timeout <= 0:
        parser.error("--process-timeout must be positive")
    if args.preflight_timeout <= 0:
        parser.error("--preflight-timeout must be positive")
    if args.preflight_measure_seconds <= 0:
        parser.error("--preflight-measure-seconds must be positive")
    if args.log_tail_chars <= 0:
        parser.error("--log-tail-chars must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_physical_smoke(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
