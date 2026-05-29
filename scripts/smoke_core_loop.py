r"""Local development smoke for the Argus core business loop.

This smoke derives a temporary file-camera setup from ``configs/default.yaml``
so a workstation without USB/RTSP hardware can still exercise the main
operator path:

    Cameras -> Alerts -> Replay -> Models -> System -> Reports

The camera/alert/replay/report portion uses a real ``CameraManager`` and a
deterministic dev video. The model training/export portion uses a tiny fake
trainer so the release UI/API contract can be checked without running an
Anomalib training job.

Example:

    .\.venv\Scripts\python.exe scripts\smoke_core_loop.py

Hardware validation examples:

    .\.venv\Scripts\python.exe scripts\smoke_core_loop.py --preflight --camera-source 0 --camera-protocol usb --require-go2rtc
    .\.venv\Scripts\python.exe scripts\smoke_core_loop.py --camera-source 0 --camera-protocol usb --require-go2rtc --activation-delay 10
    .\.venv\Scripts\python.exe scripts\smoke_core_loop.py --camera-source rtsp://user:pass@host/stream --camera-protocol rtsp --require-go2rtc --activation-delay 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from argus.alerts.dispatcher import AlertDispatcher
from argus.anomaly.job_executor import TrainingJobExecutor
from argus.anomaly.trainer import TrainingResult, TrainingStatus
from argus.capture.manager import CameraManager
from argus.config.loader import load_config
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.runtime.dev_video import DEV_VIDEO_MOTIONS, create_dev_video
from argus.streaming.preview_gateway import PreviewGateway
from argus.storage.alert_recording import AlertRecordingStore
from argus.storage.database import Database
from argus.storage.model_registry import ModelRegistry
from argus.storage.release_pipeline import ReleasePipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
_MIN_FAST_USB_MEASURE_SECONDS = 15.0


class SmokeFailure(RuntimeError):
    """Raised when the smoke cannot prove a required business step."""


class _FakeTrainer:
    def __init__(self, model_dir: Path, exports_dir: Path) -> None:
        self.exports_dir = exports_dir
        self._model_dir = model_dir

    def train(self, **_kwargs: Any) -> TrainingResult:
        return TrainingResult(
            status=TrainingStatus.COMPLETE,
            model_path=str(self._model_dir),
            duration_seconds=0.25,
            model_version_id=None,
        )


class _FakeReexportTrainer:
    def __init__(self, export_path: Path) -> None:
        self._export_path = export_path
        self.calls: list[dict[str, Any]] = []

    def reexport_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        self._export_path.mkdir(parents=True, exist_ok=True)
        (self._export_path / "model.xml").write_text("<xml/>", encoding="utf-8")
        return {
            "status": "ok",
            "export_path": str(self._export_path),
            "format": kwargs.get("export_format"),
            "quantization": kwargs.get("quantization"),
        }


class _FakePipeline:
    def update_thresholds(
        self,
        *,
        anomaly_threshold: float | None,
        severity_changed: bool,
        temporal_changed: bool,
        suppression_changed: bool,
    ) -> dict[str, bool]:
        return {
            "anomaly_threshold": anomaly_threshold is not None,
            "severity": severity_changed,
            "temporal": temporal_changed,
            "suppression": suppression_changed,
        }

    def reload_module(self, _key: str, _value: bool) -> bool:
        return True

    def is_anomaly_degraded(self) -> bool:
        return False

    def get_anomaly_degradation_reason(self) -> str | None:
        return None

    def get_anomaly_degradation_started_at(self) -> float | None:
        return None


class _FakeRuntimeManager:
    """Small runtime double for release-pipeline API smoke.

    The real camera smoke has already proven camera processing. The fake runtime
    lets the model release endpoints run without loading a fake OpenVINO file
    into the live detector.
    """

    def __init__(self, camera_id: str) -> None:
        self._pipelines = {camera_id: _FakePipeline()}
        self._camera_id = camera_id
        self.release_state_calls: list[str] = []
        self.reload_calls: list[tuple[str, str, str | None]] = []

    def get_pipeline(self, camera_id: str):
        return self._pipelines.get(camera_id)

    def apply_model_release_state(self, camera_id: str) -> dict[str, Any]:
        self.release_state_calls.append(camera_id)
        return {
            "camera_id": camera_id,
            "running": camera_id == self._camera_id,
            "primary_reloaded": True,
            "shadow_attached": False,
            "shadow_detached": False,
            "errors": [],
        }

    def reload_model(
        self,
        camera_id: str,
        model_path: str,
        *,
        version_tag: str | None = None,
    ) -> bool:
        self.reload_calls.append((camera_id, model_path, version_tag))
        return True


def _api_data(response, *, label: str) -> Any:
    if response.status_code >= 400:
        raise SmokeFailure(
            f"{label} failed: HTTP {response.status_code} {response.text[:500]}"
        )
    payload = response.json()
    if payload.get("code") != 0:
        raise SmokeFailure(f"{label} failed: {payload}")
    return payload.get("data")


def _wait_for(
    label: str,
    predicate: Callable[[], Any],
    *,
    timeout_s: float,
    interval_s: float = 0.5,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception as exc:  # Keep waiting; include final error if any.
            last_error = exc
        time.sleep(interval_s)
    suffix = f" Last error: {last_error}" if last_error else ""
    raise SmokeFailure(f"Timed out waiting for {label}.{suffix}")


async def _read_mjpeg_first_chunk(app, camera_id: str) -> bytes:
    class _StreamRequest:
        def __init__(self) -> None:
            self.calls = 0

        async def is_disconnected(self) -> bool:
            self.calls += 1
            return self.calls > 1

    stream_executor = ThreadPoolExecutor(max_workers=1)
    try:
        gateway = PreviewGateway(
            SimpleNamespace(camera_manager=app.state.camera_manager),
            stream_executor=stream_executor,
            stream_semaphore=asyncio.Semaphore(1),
            max_stream_duration=1.0,
        )
        response = gateway.latest_frame_stream_response(_StreamRequest(), camera_id)
        content_type = response.headers.get("content-type", "")
        if (
            response.status_code != 200
            or "multipart/x-mixed-replace" not in content_type
        ):
            raise SmokeFailure(
                f"mjpeg stream failed: HTTP {response.status_code} {content_type}"
            )
        try:
            return await anext(response.body_iterator)
        except StopAsyncIteration as exc:
            raise SmokeFailure("mjpeg stream ended before first frame") from exc
    finally:
        stream_executor.shutdown(wait=False, cancel_futures=True)


def _infer_camera_protocol(source: str) -> str:
    lowered = source.lower()
    if lowered.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if source.strip().isdigit() or lowered.startswith("/dev/video"):
        return "usb"
    return "file"


def _parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_raw, height_raw = value.lower().replace("x", ",").split(",", 1)
        width = int(width_raw.strip())
        height = int(height_raw.strip())
    except (AttributeError, ValueError) as exc:
        raise SmokeFailure(
            f"Invalid --camera-resolution {value!r}; expected WIDTH,HEIGHT"
        ) from exc
    if width <= 0 or height <= 0:
        raise SmokeFailure("--camera-resolution width and height must be positive")
    return width, height


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _opencv_runtime_info() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package_name in (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ):
        try:
            packages[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue

    info: dict[str, Any] = {
        "version": None,
        "module_path": None,
        "installed_packages": packages,
        "warnings": [],
    }
    try:
        import cv2
    except Exception as exc:
        info["warnings"].append(f"OpenCV import failed: {exc}")
        return info

    info["version"] = getattr(cv2, "__version__", None)
    info["module_path"] = getattr(cv2, "__file__", None)
    full_packages = {"opencv-python", "opencv-contrib-python"} & packages.keys()
    headless_packages = {
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    } & packages.keys()
    if full_packages and headless_packages:
        info["warnings"].append(
            "Both GUI and headless OpenCV wheels are installed; keep only one "
            "OpenCV package in the environment when debugging USB capture."
        )
    return info


def _probe_capture_source(
    source: str,
    protocol: str,
    *,
    timeout_ms: int = 3000,
    measure_seconds: float = 2.0,
) -> dict[str, Any]:
    """Open a camera-like source and read one frame for fast hardware checks."""

    try:
        import cv2
    except Exception as exc:
        return {
            "ok": False,
            "source": source,
            "protocol": protocol,
            "error": f"OpenCV import failed: {exc}",
            "attempts": [],
        }

    if protocol == "usb" and str(source).strip().isdigit():
        capture_source: int | str = int(str(source).strip())
    else:
        capture_source = source

    backend_candidates: list[tuple[int, str]] = []
    if protocol == "usb" and platform.system() == "Windows":
        backend_candidates.extend(
            [
                (cv2.CAP_DSHOW, "dshow"),
                (cv2.CAP_MSMF, "msmf"),
            ]
        )
    if protocol in {"file", "rtsp"} and hasattr(cv2, "CAP_FFMPEG"):
        backend_candidates.append((cv2.CAP_FFMPEG, "ffmpeg"))
    backend_candidates.append((cv2.CAP_ANY, "any"))

    attempts: list[dict[str, Any]] = []
    for backend, backend_name in backend_candidates:
        cap = cv2.VideoCapture()
        try:
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)

            opened = bool(cap.open(capture_source, backend))
            frame_ok = False
            shape = None
            measured_fps = None
            measured_frames = 0
            measured_elapsed = 0.0
            if opened:
                ok, frame = cap.read()
                frame_ok = bool(ok and frame is not None)
                shape = list(frame.shape) if frame_ok else None
                if frame_ok and protocol in {"rtsp", "usb"} and measure_seconds > 0:
                    measured_frames = 0
                    start = time.perf_counter()
                    deadline = start + measure_seconds
                    while time.perf_counter() < deadline:
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            measured_frames += 1
                            shape = list(frame.shape)
                    measured_elapsed = time.perf_counter() - start
                    if measured_elapsed > 0 and measured_frames > 0:
                        measured_fps = measured_frames / measured_elapsed
            attempt = {
                "backend": backend_name,
                "opened": opened,
                "frame_ok": frame_ok,
                "shape": shape,
                "measured_fps": measured_fps,
                "measured_frames": measured_frames,
                "measured_elapsed_seconds": measured_elapsed,
            }
        except Exception as exc:
            attempt = {
                "backend": backend_name,
                "opened": False,
                "frame_ok": False,
                "shape": None,
                "error": str(exc),
            }
        finally:
            cap.release()
        attempts.append(attempt)
        if attempt["frame_ok"]:
            return {
                "ok": True,
                "source": source,
                "protocol": protocol,
                "backend": backend_name,
                "shape": attempt["shape"],
                "measured_fps": attempt.get("measured_fps"),
                "measured_frames": attempt.get("measured_frames", 0),
                "measured_elapsed_seconds": attempt.get("measured_elapsed_seconds", 0.0),
                "attempts": attempts,
            }

    return {
        "ok": False,
        "source": source,
        "protocol": protocol,
        "attempts": attempts,
    }


def _inspect_usb_video_devices(timeout_s: float = 8.0) -> dict[str, Any]:
    """Best-effort OS inventory for USB/camera devices.

    This is diagnostic only. A missing/failed inventory should never make
    preflight fail by itself because OpenCV/go2rtc are the authoritative checks.
    """

    system = platform.system()
    info: dict[str, Any] = {
        "platform": system,
        "supported": system == "Windows",
        "source": None,
        "devices": [],
    }
    if system != "Windows":
        info["note"] = "USB device inventory is currently implemented for Windows."
        return info

    command = r"""
$devices = @(Get-PnpDevice -Class Camera -ErrorAction SilentlyContinue)
if ($devices.Count -eq 0) {
  $devices = @(Get-PnpDevice -Class Image -ErrorAction SilentlyContinue)
}
$devices = $devices |
  Where-Object { $_ -ne $null } |
  Sort-Object InstanceId -Unique |
  Select-Object FriendlyName,Class,Status,InstanceId
$devices | ConvertTo-Json -Depth 4 -Compress
""".strip()
    info["source"] = "PnpDevice"
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        info.update({"error": str(exc)})
        return info

    stdout = proc.stdout.strip()
    info.update(
        {
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
        }
    )
    if proc.returncode != 0:
        return info
    if not stdout:
        return info

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        info.update({"error": f"Failed to parse PowerShell JSON: {exc}"})
        return info

    if isinstance(parsed, dict):
        devices = [parsed]
    elif isinstance(parsed, list):
        devices = [device for device in parsed if isinstance(device, dict)]
    else:
        devices = []

    info["devices"] = [
        {
            "name": device.get("Name") or device.get("FriendlyName"),
            "pnp_class": device.get("PNPClass") or device.get("Class"),
            "status": device.get("Status"),
            "manufacturer": device.get("Manufacturer"),
            "device_id": device.get("DeviceID") or device.get("InstanceId"),
        }
        for device in devices
    ]
    info["device_count"] = len(info["devices"])
    return info


def _inspect_windows_camera_privacy(timeout_s: float = 3.0) -> dict[str, Any]:
    system = platform.system()
    info: dict[str, Any] = {
        "platform": system,
        "supported": system == "Windows",
        "entries": [],
    }
    if system != "Windows":
        return info

    command = r"""
$paths = @(
  @{ Scope = 'HKCU'; Path = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam' },
  @{ Scope = 'HKLM'; Path = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam' }
)
$entries = foreach ($entry in $paths) {
  $item = Get-ItemProperty -Path $entry.Path -ErrorAction SilentlyContinue
  if ($null -ne $item) {
    [pscustomobject]@{
      Scope = $entry.Scope
      Value = $item.Value
      LastUsedTimeStart = $item.LastUsedTimeStart
      LastUsedTimeStop = $item.LastUsedTimeStop
    }
  }
}
$entries | ConvertTo-Json -Depth 4 -Compress
""".strip()
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        info.update({"error": str(exc)})
        return info

    stdout = proc.stdout.strip()
    info.update({"returncode": proc.returncode, "stderr": proc.stderr.strip()})
    if proc.returncode != 0 or not stdout:
        return info

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        info.update({"error": f"Failed to parse PowerShell JSON: {exc}"})
        return info

    entries = [parsed] if isinstance(parsed, dict) else parsed
    if isinstance(entries, list):
        info["entries"] = [
            {
                "scope": entry.get("Scope"),
                "value": entry.get("Value"),
                "last_used_start": entry.get("LastUsedTimeStart"),
                "last_used_stop": entry.get("LastUsedTimeStop"),
            }
            for entry in entries
            if isinstance(entry, dict)
        ]
    return info


def _looks_like_conda_binary(path: str | None) -> bool:
    if not path:
        return False
    lowered = path.replace("\\", "/").lower()
    return "/conda/" in lowered or "/miniconda" in lowered or "/anaconda" in lowered


def _resolve_dshow_ffmpeg_binary() -> dict[str, Any]:
    env_binary = os.getenv("ARGUS_FFMPEG_BINARY")
    warnings: list[str] = []
    if env_binary:
        return {
            "binary": env_binary,
            "source": "ARGUS_FFMPEG_BINARY",
            "warnings": warnings,
        }

    suffix = ".exe" if platform.system() == "Windows" else ""
    local_binary = REPO_ROOT / "bin" / f"ffmpeg{suffix}"
    if local_binary.is_file():
        return {
            "binary": str(local_binary),
            "source": "project_bin",
            "warnings": warnings,
        }

    path_binary = shutil.which("ffmpeg")
    if _looks_like_conda_binary(path_binary):
        warnings.append(
            "ffmpeg resolved from a conda/miniconda path; skipping DirectShow "
            "enumeration to avoid treating a non-runtime diagnostic dependency as authoritative."
        )
        return {
            "binary": None,
            "source": "path_conda_skipped",
            "candidate": path_binary,
            "warnings": warnings,
        }

    return {
        "binary": path_binary,
        "source": "PATH" if path_binary else None,
        "warnings": warnings,
    }


def _inspect_windows_dshow_devices(timeout_s: float = 5.0) -> dict[str, Any]:
    system = platform.system()
    ffmpeg = _resolve_dshow_ffmpeg_binary()
    binary = ffmpeg.get("binary")
    info: dict[str, Any] = {
        "platform": system,
        "supported": system == "Windows" and binary is not None,
        "binary": binary,
        "binary_source": ffmpeg.get("source"),
        "candidate_binary": ffmpeg.get("candidate"),
        "warnings": ffmpeg.get("warnings", []),
        "diagnostic_only": True,
        "devices": [],
    }
    if system != "Windows":
        return info
    if binary is None:
        info["error"] = (
            "ffmpeg was not found in ARGUS_FFMPEG_BINARY, bin/, or PATH"
            if not ffmpeg.get("candidate")
            else "ffmpeg was skipped because only a conda/miniconda PATH binary was found"
        )
        return info

    try:
        proc = subprocess.run(
            [str(binary), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        info.update({"error": str(exc)})
        return info

    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    info.update(
        {
            "returncode": proc.returncode,
            "video_enumeration_failed": "Could not enumerate video devices" in output,
            "output_excerpt": output[:4000],
        }
    )

    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in output.splitlines():
        device_match = re.search(r'\]\s+"(?P<name>.+)"\s+\((?P<kind>video|audio)\)', line)
        if device_match:
            current = {
                "name": device_match.group("name"),
                "kind": device_match.group("kind"),
                "alternative_name": None,
            }
            devices.append(current)
            continue
        alternative_match = re.search(r'\]\s+Alternative name "(?P<name>.+)"', line)
        if current is not None and alternative_match:
            current["alternative_name"] = alternative_match.group("name")

    info["devices"] = devices
    info["video_device_count"] = sum(1 for device in devices if device.get("kind") == "video")
    return info


def _usb_device_selection_report(
    *,
    source: str,
    usb_cfg: Any,
    usb_devices: dict[str, Any] | None,
    dshow_devices: dict[str, Any] | None,
) -> dict[str, Any]:
    def _matches(
        device: dict[str, Any],
        *,
        device_id: str | None,
        device_name: str | None,
        id_fields: tuple[str, ...],
        name_fields: tuple[str, ...],
    ) -> bool:
        if device_id:
            token = str(device_id).lower()
            for field in id_fields:
                value = str(device.get(field) or "").lower()
                if value and (token in value or value in token):
                    return True
        if device_name:
            token = str(device_name).lower()
            for field in name_fields:
                value = str(device.get(field) or "").lower()
                if value and (token == value or token in value):
                    return True
        return False

    def _select_match(
        devices: list[dict[str, Any]],
        *,
        device_id: str | None,
        device_name: str | None,
        id_fields: tuple[str, ...],
        name_fields: tuple[str, ...],
    ) -> dict[str, Any] | None:
        matches = [
            device for device in devices
            if _matches(
                device,
                device_id=device_id,
                device_name=device_name,
                id_fields=id_fields,
                name_fields=name_fields,
            )
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: str(item.get("status") or "").lower() != "ok")
        return matches[0]

    report: dict[str, Any] = {
        "configured_source": source,
        "configured_device_name": getattr(usb_cfg, "device_name", None),
        "configured_device_id": getattr(usb_cfg, "device_id", None),
        "source_is_numeric_index": str(source).strip().isdigit(),
        "index": int(source) if str(source).strip().isdigit() else None,
        "selected_pnp_device": None,
        "selected_dshow_device": None,
        "warnings": [],
    }
    pnp_devices = (usb_devices or {}).get("devices") or []
    dshow_video_devices = [
        device
        for device in (dshow_devices or {}).get("devices") or []
        if device.get("kind") == "video"
    ]
    if report["configured_device_id"] or report["configured_device_name"]:
        report["selection_mode"] = "explicit_device_id_or_name"
        report["selected_pnp_device"] = _select_match(
            pnp_devices,
            device_id=report["configured_device_id"],
            device_name=report["configured_device_name"],
            id_fields=("device_id",),
            name_fields=("name",),
        )
        report["selected_dshow_device"] = _select_match(
            dshow_video_devices,
            device_id=report["configured_device_id"],
            device_name=report["configured_device_name"],
            id_fields=("alternative_name",),
            name_fields=("name",),
        )
        if report["selected_pnp_device"] is None and report["selected_dshow_device"] is None:
            report["warnings"].append(
                "usb.device_name or usb.device_id was configured, but no inspected "
                "USB camera inventory entry matched it."
            )
        return report

    report["selection_mode"] = "numeric_index" if report["source_is_numeric_index"] else "source_string"
    if not report["source_is_numeric_index"]:
        return report

    index = int(report["index"])
    if 0 <= index < len(pnp_devices):
        report["selected_pnp_device"] = pnp_devices[index]

    if 0 <= index < len(dshow_video_devices):
        report["selected_dshow_device"] = dshow_video_devices[index]

    report["warnings"].append(
        "USB source is a numeric index; set usb.device_name or usb.device_id "
        "for stable camera selection when multiple USB cameras are present."
    )
    return report


def _expected_degradations(*, use_yolo: bool, protocol: str) -> list[dict[str, Any]]:
    degradations: list[dict[str, Any]] = []
    if not use_yolo:
        degradations.append(
            {
                "component": "person_filter",
                "expected": True,
                "reason": "smoke uses missing-yolo.pt to verify graceful offline mode",
            }
        )
    if protocol in {"file", "usb", "rtsp"}:
        degradations.append(
            {
                "component": "anomaly_detector",
                "expected": True,
                "reason": "smoke starts from SSIM fallback unless a trained model is explicitly loaded",
            }
        )
    return degradations


def _wait_before_activation(*, seconds: float, camera_id: str) -> None:
    if seconds <= 0:
        return
    print(
        (
            f"[smoke] Camera {camera_id} baseline is calibrated. "
            f"Introduce the test object now; switching to active mode in {seconds:.1f}s."
        ),
        file=sys.stderr,
        flush=True,
    )
    time.sleep(seconds)


def _effective_preflight_measure_seconds(camera, requested_seconds: float) -> float:
    if str(camera.protocol) != "usb":
        return requested_seconds
    fast_cfg = getattr(camera, "fast_motion", None)
    min_fps = float(getattr(fast_cfg, "min_runtime_fps", 0.0) or 0.0)
    if min_fps <= 0:
        return requested_seconds
    return max(requested_seconds, _MIN_FAST_USB_MEASURE_SECONDS)


def _prepare_config(
    config_path: Path,
    work_dir: Path,
    video_path: Path,
    *,
    use_yolo: bool,
    camera_source: str | None = None,
    camera_protocol: str | None = None,
    camera_id: str = "dev_cam",
    camera_name: str | None = None,
    camera_resolution: tuple[int, int] = (640, 480),
    usb_device_name: str | None = None,
    usb_device_id: str | None = None,
    go2rtc_enabled: bool | None = None,
):
    config = load_config(config_path).model_copy(deep=True)
    config.node_id = "argus-smoke"
    source = camera_source or str(video_path)
    protocol = camera_protocol or _infer_camera_protocol(source)
    if protocol not in {"file", "usb", "rtsp"}:
        raise SmokeFailure(
            f"Unsupported smoke camera protocol {protocol!r}; use file, usb, or rtsp"
        )
    if go2rtc_enabled is None:
        go2rtc_enabled = protocol in {"usb", "rtsp"} and camera_source is not None
    config.dashboard.go2rtc_enabled = bool(go2rtc_enabled)
    if config.dashboard.go2rtc_enabled:
        config.dashboard.go2rtc_api_port = _free_port()
        config.dashboard.go2rtc_rtsp_port = _free_port()
        config.dashboard.go2rtc_webrtc_port = _free_port()
    config.auth.enabled = False

    config.storage.database_url = f"sqlite:///{work_dir / 'argus.db'}"
    config.storage.baselines_dir = work_dir / "baselines"
    config.storage.models_dir = work_dir / "models"
    config.storage.exports_dir = work_dir / "exports"
    config.storage.backbones_dir = work_dir / "backbones"
    config.storage.foe_objects_dir = work_dir / "foe_objects"
    config.storage.model_packages_dir = work_dir / "model_packages"
    config.storage.alerts_dir = work_dir / "alerts"
    config.storage.inference_records_dir = work_dir / "inference_records"

    if config.cameras:
        camera = config.cameras[0].model_copy(deep=True)
    else:
        from argus.config.schema import CameraConfig

        camera = CameraConfig(camera_id="dev_cam", name="Dev Camera")
    camera.camera_id = camera_id
    camera.name = camera_name or (
        "Dev smoke camera" if camera_source is None else "Hardware smoke camera"
    )
    camera.protocol = protocol
    camera.source = source
    if camera_source is None:
        camera.fps_target = 10
    camera.resolution = camera_resolution
    if protocol == "usb":
        if usb_device_name is not None:
            camera.usb.device_name = usb_device_name
        if usb_device_id is not None:
            camera.usb.device_id = usb_device_id
    camera.zones = []
    camera.anomaly.ssim_baseline_frames = 15
    camera.ring_buffer.enabled = True
    if not use_yolo:
        # Avoid a network/download dependency in local smoke. The detector path
        # should degrade gracefully and still analyze every frame.
        camera.person_filter.model_name = str(work_dir / "missing-yolo.pt")
    config.cameras = [camera]
    return config


def _run_camera_alert_replay_reports(
    *,
    client: TestClient,
    manager: CameraManager,
    dispatcher: AlertDispatcher,
    events: list[tuple[str, dict[str, Any]]],
    camera_id: str,
    timeout_s: float,
    recording_timeout_s: float,
    require_go2rtc: bool = False,
    activation_delay_s: float = 0.0,
) -> dict[str, Any]:
    started = manager.start_all()
    if camera_id not in started:
        raise SmokeFailure(f"Camera {camera_id} did not start; started={started}")

    def camera_online():
        data = _api_data(client.get("/api/cameras/json"), label="cameras json")
        rows = data.get("cameras") or []
        row = next((item for item in rows if item["camera_id"] == camera_id), None)
        if row and row.get("connected") and row.get("running"):
            stats = row.get("stats") or {}
            if stats.get("frames_captured", 0) >= 5:
                return row
        return None

    camera_row = _wait_for(
        "camera to connect and capture frames",
        camera_online,
        timeout_s=min(timeout_s, 30),
    )

    snapshot = client.get(f"/api/cameras/{camera_id}/snapshot")
    if snapshot.status_code != 200 or "image/jpeg" not in snapshot.headers.get("content-type", ""):
        raise SmokeFailure(
            f"snapshot failed: HTTP {snapshot.status_code} {snapshot.headers.get('content-type')}"
        )

    streaming = _api_data(
        client.get(f"/api/streaming/{camera_id}"),
        label="streaming info",
    )
    expected_fallback = f"/api/cameras/{camera_id}/stream"
    if streaming.get("fallback") != expected_fallback:
        raise SmokeFailure(f"streaming fallback mismatch: {streaming}")
    if require_go2rtc and streaming.get("go2rtc") is not True:
        raise SmokeFailure(f"go2rtc streaming was required but unavailable: {streaming}")

    first_stream_chunk = asyncio.run(_read_mjpeg_first_chunk(client.app, camera_id))
    if b"--frame" not in first_stream_chunk or b"\xff\xd8" not in first_stream_chunk:
        raise SmokeFailure(
            f"mjpeg first frame invalid: bytes={len(first_stream_chunk)}"
        )

    def ssim_ready():
        status = manager.get_detector_status(camera_id) or {}
        return status if status.get("ssim_calibrated") else None

    detector_status = _wait_for(
        "SSIM fallback calibration",
        ssim_ready,
        timeout_s=min(timeout_s, 40),
    )

    _wait_before_activation(seconds=activation_delay_s, camera_id=camera_id)

    mode_data = _api_data(
        client.post(f"/api/cameras/{camera_id}/mode", json={"mode": "active"}),
        label="set camera active mode",
    )
    if mode_data.get("pipeline_mode") != "active":
        raise SmokeFailure(f"camera mode did not become active: {mode_data}")

    def latest_alert():
        dispatcher.flush_db_queue()
        data = _api_data(
            client.get("/api/alerts/json", params={"camera_id": camera_id, "limit": 10}),
            label="alerts json",
        )
        alerts = data.get("alerts") or []
        return alerts[0] if alerts else None

    alert = _wait_for(
        "alert generated from camera source",
        latest_alert,
        timeout_s=timeout_s,
        interval_s=1.0,
    )
    alert_id = alert["alert_id"]

    def realtime_alert_payload():
        for topic, payload in events:
            if topic == "alerts" and payload.get("alert_id") == alert_id:
                return payload
        return None

    realtime_payload = _wait_for(
        "alert websocket payload",
        realtime_alert_payload,
        timeout_s=5,
        interval_s=0.1,
    )
    if realtime_payload.get("camera_id") != camera_id:
        raise SmokeFailure(f"alert websocket camera mismatch: {realtime_payload}")
    if not realtime_payload.get("snapshot_path") or not realtime_payload.get("heatmap_path"):
        raise SmokeFailure(f"alert websocket evidence missing: {realtime_payload}")
    if realtime_payload.get("has_recording") is not True:
        raise SmokeFailure(f"alert websocket recording flag missing: {realtime_payload}")
    if not realtime_payload.get("recording_status"):
        raise SmokeFailure(f"alert websocket recording status missing: {realtime_payload}")
    if not Path(realtime_payload["snapshot_path"]).exists():
        raise SmokeFailure(
            f"alert websocket snapshot file missing: {realtime_payload['snapshot_path']}"
        )
    if not Path(realtime_payload["heatmap_path"]).exists():
        raise SmokeFailure(
            f"alert websocket heatmap file missing: {realtime_payload['heatmap_path']}"
        )

    detail = _api_data(client.get(f"/api/alerts/{alert_id}/detail"), label="alert detail")
    if not detail.get("snapshot_path") or not detail.get("heatmap_path"):
        raise SmokeFailure(f"alert evidence images missing: {detail}")

    def complete_recording():
        dispatcher.flush_db_queue()
        current = _api_data(
            client.get(f"/api/alerts/{alert_id}/detail"),
            label="alert detail recording",
        )
        if current.get("recording_status") == "complete":
            return current
        return None

    complete_detail = _wait_for(
        "recording to complete",
        complete_recording,
        timeout_s=recording_timeout_s,
        interval_s=1.0,
    )

    refreshed_alerts = _api_data(
        client.get("/api/alerts/json", params={"camera_id": camera_id, "limit": 10}),
        label="alerts json after recording",
    ).get("alerts") or []
    refreshed_alert = next(
        (item for item in refreshed_alerts if item.get("alert_id") == alert_id),
        None,
    )
    if not refreshed_alert:
        raise SmokeFailure("completed alert missing from refreshed alerts list")
    if refreshed_alert.get("has_recording") is not True:
        raise SmokeFailure(f"refreshed alert lost recording flag: {refreshed_alert}")
    if refreshed_alert.get("recording_status") != "complete":
        raise SmokeFailure(
            f"refreshed alert recording status is not complete: {refreshed_alert}"
        )

    replay_meta = _api_data(
        client.get(f"/api/replay/{alert_id}/metadata"),
        label="replay metadata",
    )
    if replay_meta.get("status") != "complete":
        raise SmokeFailure(f"replay metadata not complete: {replay_meta}")

    replay_signals = _api_data(
        client.get(f"/api/replay/{alert_id}/signals"),
        label="replay signals",
    )
    if not replay_signals.get("timestamps"):
        raise SmokeFailure("replay signals missing timestamps")

    evidence_zip = client.get(f"/api/alerts/{alert_id}/evidence.zip")
    if evidence_zip.status_code != 200 or len(evidence_zip.content) < 1024:
        raise SmokeFailure(
            f"evidence zip invalid: HTTP {evidence_zip.status_code}, bytes={len(evidence_zip.content)}"
        )

    reports = _api_data(client.get("/api/reports/json"), label="reports json")
    evidence = reports.get("evidence") or {}
    if evidence.get("total_alerts", 0) < 1 or evidence.get("evidence_complete_rate") != 100.0:
        raise SmokeFailure(f"reports evidence stats incomplete: {evidence}")

    report_days = 30
    daily_trend = _api_data(
        client.get("/api/reports/daily-trend/json", params={"days": report_days}),
        label="reports daily trend",
    )
    if len(daily_trend.get("labels") or []) != report_days:
        raise SmokeFailure(
            f"reports daily trend did not return {report_days} days: {daily_trend}"
        )
    trend_total = sum(
        sum(daily_trend.get(key) or [])
        for key in ("high", "medium", "low", "info")
    )
    if trend_total < 1:
        raise SmokeFailure(
            f"reports daily trend did not include the smoke alert: {daily_trend}"
        )

    severity_dist = _api_data(
        client.get("/api/reports/severity-dist/json", params={"days": report_days}),
        label="reports severity distribution",
    )
    severity_total = sum(
        severity_dist.get(key, 0) for key in ("high", "medium", "low", "info")
    )
    if severity_total < 1:
        raise SmokeFailure(
            f"reports severity distribution missed the smoke alert: {severity_dist}"
        )

    camera_dist = _api_data(
        client.get("/api/reports/camera-dist/json", params={"days": report_days}),
        label="reports camera distribution",
    )
    camera_dist_rows = camera_dist.get("cameras") or []
    camera_dist_row = next(
        (item for item in camera_dist_rows if item.get("camera_id") == camera_id),
        None,
    )
    if not camera_dist_row or camera_dist_row.get("count", 0) < 1:
        raise SmokeFailure(
            f"reports camera distribution missed {camera_id}: {camera_dist}"
        )

    fp_trend = _api_data(
        client.get("/api/reports/fp-trend/json", params={"days": report_days}),
        label="reports false-positive trend",
    )
    if len(fp_trend.get("labels") or []) != report_days:
        raise SmokeFailure(
            f"reports FP trend did not return {report_days} days: {fp_trend}"
        )
    if len(fp_trend.get("rates") or []) != report_days:
        raise SmokeFailure(f"reports FP trend rates length mismatch: {fp_trend}")

    compliance_csv = client.get(
        "/api/reports/compliance",
        params={"days": report_days, "format": "csv"},
    )
    if compliance_csv.status_code != 200:
        raise SmokeFailure(
            f"reports compliance csv failed: HTTP {compliance_csv.status_code} {compliance_csv.text[:500]}"
        )
    if "text/csv" not in compliance_csv.headers.get("content-type", ""):
        raise SmokeFailure(
            f"reports compliance csv content-type mismatch: {compliance_csv.headers.get('content-type')}"
        )
    compliance_text = compliance_csv.content.decode("utf-8-sig", errors="replace")
    for required in ("## 告警统计", "## 证据完整性", "Replay录像", "完整证据"):
        if required not in compliance_text:
            raise SmokeFailure(f"reports compliance csv missing {required!r}")

    model_status_rows = _api_data(
        client.get("/api/models/status"),
        label="system model status",
    ).get("models") or []
    anomaly_status = next(
        (
            item
            for item in model_status_rows
            if item.get("camera_id") == camera_id and item.get("name") == "anomaly"
        ),
        None,
    )
    if anomaly_status is None:
        raise SmokeFailure(
            f"system model status missing anomaly row for {camera_id}: {model_status_rows}"
        )
    if detector_status.get("mode") == "ssim_fallback":
        if anomaly_status.get("backend") != "ssim-fallback":
            raise SmokeFailure(
                f"system model status did not surface SSIM fallback: {anomaly_status}"
            )
    elif anomaly_status.get("backend") in (None, "", "none"):
        raise SmokeFailure(
            f"system model status missing active backend: {anomaly_status}"
        )

    return {
        "camera": {
            "camera_id": camera_id,
            "frames_captured": (camera_row.get("stats") or {}).get("frames_captured"),
            "detector_mode": detector_status.get("mode"),
            "ssim_noise_floor": detector_status.get("ssim_noise_floor"),
            "streaming_go2rtc": streaming.get("go2rtc"),
            "mjpeg_first_chunk_bytes": len(first_stream_chunk),
            "activation_delay_seconds": activation_delay_s,
        },
        "alert": {
            "alert_id": alert_id,
            "severity": alert.get("severity"),
            "recording_status": complete_detail.get("recording_status"),
            "api_recording_status": refreshed_alert.get("recording_status"),
            "replay_frames": replay_meta.get("frame_count"),
            "evidence_zip_bytes": len(evidence_zip.content),
            "realtime_recording_status": realtime_payload.get("recording_status"),
            "realtime_has_recording": realtime_payload.get("has_recording"),
        },
        "reports": {
            **evidence,
            "daily_trend_points": len(daily_trend.get("labels") or []),
            "severity_total": severity_total,
            "camera_distribution": camera_dist_row,
            "fp_trend_points": len(fp_trend.get("rates") or []),
            "compliance_csv_bytes": len(compliance_csv.content),
        },
        "system_model_status": {
            "backend": anomaly_status.get("backend"),
            "loaded": anomaly_status.get("loaded"),
            "consecutive_failures": anomaly_status.get("consecutive_failures"),
            "total_inferences": anomaly_status.get("total_inferences"),
            "last_error": anomaly_status.get("last_error"),
        },
    }


def _run_models_and_system_smoke(
    *,
    client: TestClient,
    app,
    database: Database,
    work_dir: Path,
    camera_id: str,
) -> dict[str, Any]:
    fake_runtime = _FakeRuntimeManager(camera_id)
    app.state.camera_manager = fake_runtime
    app.state._release_pipeline = ReleasePipeline(
        database.get_session,
        min_shadow_days=0,
        min_canary_days=0,
    )

    model_dir = work_dir / "models" / camera_id / "default" / "smoke-trained"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.xml").write_text("<xml/>", encoding="utf-8")
    baseline_dir = work_dir / "baselines" / camera_id / "default"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "baseline.png").write_bytes(b"png")

    registry = ModelRegistry(session_factory=database.get_session)
    pipeline = app.state._release_pipeline

    previous_model_dir = work_dir / "models" / camera_id / "default" / "smoke-previous"
    previous_model_dir.mkdir(parents=True, exist_ok=True)
    (previous_model_dir / "model.xml").write_text("<xml/>", encoding="utf-8")
    previous_version_id = registry.register(
        previous_model_dir,
        baseline_dir,
        camera_id,
        "patchcore",
        training_params={"seed": "previous-production"},
    )
    for target_stage in ("shadow", "canary", "production"):
        pipeline.transition(
            model_version_id=previous_version_id,
            target_stage=target_stage,
            triggered_by="smoke",
            canary_camera_id=camera_id if target_stage == "canary" else None,
        )

    job = _api_data(
        client.post(
            "/api/training-jobs/",
            json={
                "job_type": "anomaly_head",
                "camera_id": camera_id,
                "model_type": "patchcore",
                "hyperparameters": {"skip_baseline_validation": True},
            },
        ),
        label="create training job",
    )
    job_id = job["job_id"]
    _api_data(
        client.post(f"/api/training-jobs/{job_id}/confirm", json={"confirmed_by": "smoke"}),
        label="confirm training job",
    )

    executor = TrainingJobExecutor(
        database=database,
        trainer=_FakeTrainer(model_dir, work_dir / "exports"),
        model_registry=registry,
        baselines_dir=work_dir / "baselines",
        model_packages_dir=work_dir / "model_packages",
    )
    executor.execute(job_id)

    job_detail = _api_data(client.get(f"/api/training-jobs/{job_id}"), label="training job")
    version_id = job_detail.get("model_version_id")
    if job_detail.get("status") != "complete" or not version_id:
        raise SmokeFailure(f"training job did not complete with model version: {job_detail}")

    export_trainer = _FakeReexportTrainer(work_dir / "exports" / camera_id / "default")
    import argus.dashboard.routes.models as models_route

    original_get_trainer = models_route._get_trainer
    models_route._get_trainer = lambda _request: export_trainer
    try:
        reexport = _api_data(
            client.post(
                f"/api/models/{version_id}/reexport",
                json={"export_format": "openvino", "quantization": "fp16"},
            ),
            label="reexport model",
        )
    finally:
        models_route._get_trainer = original_get_trainer

    stages = [
        {"target_stage": "shadow", "triggered_by": "smoke"},
        {
            "target_stage": "canary",
            "triggered_by": "smoke",
            "canary_camera_id": camera_id,
        },
        {"target_stage": "production", "triggered_by": "smoke"},
    ]
    release_states = []
    for body in stages:
        result = _api_data(
            client.post(f"/api/models/{version_id}/promote", json=body),
            label=f"promote model to {body['target_stage']}",
        )
        release_states.append(result.get("model", {}).get("stage"))
        if result.get("runtime_synced") is not True:
            raise SmokeFailure(f"runtime did not sync on promotion: {result}")

    rollback = _api_data(
        client.post(f"/api/models/{version_id}/rollback"),
        label="rollback model",
    )
    if rollback.get("runtime_synced") is not True:
        raise SmokeFailure(f"rollback runtime did not sync: {rollback}")

    config_patch = _api_data(
        client.post(
            "/api/config/detection-params",
            json={"anomaly_threshold": 0.66, "sev_low": 0.6},
        ),
        label="system detection config patch",
    )
    if not config_patch.get("anomaly_threshold", {}).get("changed"):
        raise SmokeFailure(f"detection config patch did not report change: {config_patch}")

    module_toggle = _api_data(
        client.post(
            "/api/config/modules",
            json={"key": "classifier.enabled", "value": False},
        ),
        label="system module toggle",
    )

    degraded_pipeline = MagicMock()
    degraded_pipeline.is_anomaly_degraded.return_value = True
    degraded_pipeline.get_anomaly_degradation_reason.return_value = "smoke_forced"
    degraded_pipeline.get_anomaly_degradation_started_at.return_value = 123.0
    fake_runtime._pipelines = {camera_id: degraded_pipeline}
    degradation = _api_data(
        client.get("/api/system/anomaly-degradation"),
        label="system anomaly degradation",
    )
    if degradation.get("anomaly", {}).get("degraded") is not True:
        raise SmokeFailure(f"system degradation did not surface fallback: {degradation}")

    return {
        "training_job_id": job_id,
        "model_version_id": version_id,
        "reexport": reexport,
        "release_stages": release_states,
        "rollback_activated": rollback.get("activated"),
        "config_patch": config_patch,
        "module_toggle": module_toggle,
        "degradation": degradation.get("anomaly"),
    }


def _start_go2rtc_for_smoke(config, *, require_go2rtc: bool):
    if not getattr(config.dashboard, "go2rtc_enabled", False):
        return None, None, {}

    from argus.streaming.go2rtc_manager import Go2RTCManager
    from argus.streaming.stream_registry import StreamRegistry

    go2rtc = Go2RTCManager(
        api_port=config.dashboard.go2rtc_api_port,
        rtsp_port=config.dashboard.go2rtc_rtsp_port,
        webrtc_port=config.dashboard.go2rtc_webrtc_port,
        binary_path=config.dashboard.go2rtc_binary,
    )
    registry = StreamRegistry(go2rtc)
    setattr(go2rtc, "_stream_registry", registry)
    try:
        resolutions = registry.reconcile(config.cameras)
    except Exception as exc:
        go2rtc.close()
        if require_go2rtc:
            raise SmokeFailure(f"go2rtc startup/registration failed: {exc}") from exc
        return None, None, {"error": str(exc)}

    if require_go2rtc:
        missing = [
            camera.camera_id
            for camera in config.cameras
            if getattr(camera, "protocol", None) in {"usb", "rtsp"}
            and not getattr(resolutions.get(camera.camera_id), "go2rtc_managed", False)
        ]
        if missing:
            go2rtc.close()
            raise SmokeFailure(f"go2rtc did not manage required streams: {missing}")

    return go2rtc, registry, {
        camera_id: {
            "original_source": resolution.original_source,
            "original_protocol": resolution.original_protocol,
            "runtime_source": resolution.runtime_source,
            "runtime_protocol": resolution.runtime_protocol,
            "go2rtc_managed": resolution.go2rtc_managed,
        }
        for camera_id, resolution in resolutions.items()
    }


def _go2rtc_binary_preflight(binary_path: str | None) -> dict[str, Any]:
    from argus.streaming.go2rtc_manager import _find_go2rtc_binary

    resolved = Path(binary_path) if binary_path else _find_go2rtc_binary()
    info: dict[str, Any] = {
        "configured_binary": binary_path,
        "resolved_binary": str(resolved) if resolved else None,
        "exists": bool(resolved and resolved.is_file()),
    }
    if not resolved or not resolved.is_file():
        return info

    try:
        proc = subprocess.run(
            [str(resolved), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        info.update({"version_ok": False, "version_error": str(exc)})
        return info

    version_output = (proc.stdout or proc.stderr).strip()
    info.update(
        {
            "version_ok": proc.returncode == 0,
            "version_output": version_output,
        }
    )
    return info


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    base_work_dir = args.work_dir.resolve() if args.work_dir else None

    temp_ctx = None
    if base_work_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="argus-core-preflight-")
        work_dir = Path(temp_ctx.name)
    else:
        work_dir = base_work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    go2rtc = None
    try:
        video_path = work_dir / "dev_camera.avi"
        camera_resolution = _parse_resolution(args.camera_resolution)
        if args.camera_source is None:
            create_dev_video(
                video_path,
                width=camera_resolution[0],
                height=camera_resolution[1],
                fps=10,
                seconds=max(args.video_seconds, 3),
                anomaly_start_s=1.0,
                motion=args.dev_video_motion,
            )

        config = _prepare_config(
            config_path,
            work_dir,
            video_path,
            use_yolo=args.use_yolo,
            camera_source=args.camera_source,
            camera_protocol=args.camera_protocol,
            camera_id=args.camera_id,
            camera_name=args.camera_name,
            camera_resolution=camera_resolution,
            usb_device_name=args.usb_device_name,
            usb_device_id=args.usb_device_id,
            go2rtc_enabled=False if args.disable_go2rtc else None,
        )
        camera = config.cameras[0]
        camera_id = camera.camera_id
        opencv_info = _opencv_runtime_info()
        usb_devices = (
            _inspect_usb_video_devices(timeout_s=max(8.0, args.preflight_timeout))
            if camera.protocol == "usb"
            else None
        )
        camera_privacy = (
            _inspect_windows_camera_privacy(timeout_s=3.0)
            if camera.protocol == "usb"
            else None
        )
        dshow_devices = (
            _inspect_windows_dshow_devices(timeout_s=max(5.0, args.preflight_timeout))
            if camera.protocol == "usb"
            else None
        )
        usb_selection = (
            _usb_device_selection_report(
                source=str(camera.source),
                usb_cfg=camera.usb,
                usb_devices=usb_devices,
                dshow_devices=dshow_devices,
            )
            if camera.protocol == "usb"
            else None
        )
        go2rtc_info = {
            "enabled": bool(config.dashboard.go2rtc_enabled),
            "required": bool(args.require_go2rtc),
            "api_port": config.dashboard.go2rtc_api_port,
            "rtsp_port": config.dashboard.go2rtc_rtsp_port,
            "webrtc_port": config.dashboard.go2rtc_webrtc_port,
            "binary": _go2rtc_binary_preflight(config.dashboard.go2rtc_binary),
            "running": False,
            "resolutions": {},
        }

        go2rtc_resolution: dict[str, Any] = {}
        if config.dashboard.go2rtc_enabled:
            go2rtc, _stream_registry, go2rtc_resolution = _start_go2rtc_for_smoke(
                config,
                require_go2rtc=args.require_go2rtc,
            )
            go2rtc_info["running"] = bool(getattr(go2rtc, "running", False))
            go2rtc_info["resolutions"] = go2rtc_resolution

        resolution = go2rtc_resolution.get(camera_id, {})
        probe_source = resolution.get("runtime_source") or camera.source
        probe_protocol = resolution.get("runtime_protocol") or camera.protocol
        measure_seconds = _effective_preflight_measure_seconds(
            camera,
            args.preflight_measure_seconds,
        )
        capture_probe = _probe_capture_source(
            str(probe_source),
            str(probe_protocol),
            timeout_ms=int(args.preflight_timeout * 1000),
            measure_seconds=measure_seconds,
        )

        errors: list[str] = []
        hints: list[str] = []
        if args.require_go2rtc:
            if not go2rtc_info["running"]:
                errors.append("go2rtc is required but not running")
            if not resolution.get("go2rtc_managed"):
                errors.append(f"go2rtc did not manage camera {camera_id}")
        if not capture_probe.get("ok"):
            errors.append(
                f"camera {camera_id} did not produce a readable frame "
                f"from {probe_protocol}:{probe_source}"
            )
            if camera.protocol == "usb" and usb_devices is not None:
                raw_device_count = usb_devices.get("device_count")
                device_count = (
                    int(raw_device_count)
                    if raw_device_count is not None
                    else len(usb_devices.get("devices") or [])
                )
                if device_count:
                    hints.append(
                        "Windows enumerated USB camera devices, but the configured "
                        f"source {camera.source!r} did not yield a frame. Try another "
                        "camera index and close other applications using the camera."
                    )
                elif usb_devices.get("error"):
                    hints.append(
                        "USB device inventory failed; rerun preflight or check Windows "
                        "camera permissions before diagnosing OpenCV/go2rtc."
                    )
                else:
                    hints.append(
                        "Windows did not enumerate any Camera devices; check device "
                        "connection, driver state, and privacy permissions first."
                    )
                privacy_values = [
                    str(entry.get("value")).lower()
                    for entry in (camera_privacy or {}).get("entries", [])
                    if entry.get("value") is not None
                ]
                if any(value == "deny" for value in privacy_values):
                    hints.append(
                        "Windows camera privacy settings include Deny; enable camera "
                        "access for desktop apps before rerunning USB preflight."
                    )
                if (
                    (dshow_devices or {}).get("video_enumeration_failed")
                    and device_count
                ):
                    hints.append(
                        "DirectShow/FFmpeg could not enumerate video devices even "
                        "though Windows PnP lists cameras; check camera privacy, "
                        "driver state, and whether another app owns the camera."
                    )
            hints.extend(opencv_info.get("warnings") or [])
        else:
            fast_cfg = getattr(camera, "fast_motion", None)
            min_fps = float(getattr(fast_cfg, "min_runtime_fps", 0.0) or 0.0)
            measured_fps = capture_probe.get("measured_fps")
            if min_fps > 0 and measured_fps is not None and measured_fps < min_fps:
                errors.append(
                    f"camera {camera_id} measured {measured_fps:.1f}fps below "
                    f"required {min_fps:.1f}fps for fast-motion detection"
                )
                hints.append(
                    "The camera accepted the requested mode but delivered fewer "
                    "decoded frames than required. Increase lighting, reduce "
                    "exposure, close competing camera apps, or lower the fast-motion "
                    "min_runtime_fps before relying on small-projectile alerts."
                )
        if usb_selection is not None:
            hints.extend(usb_selection.get("warnings") or [])

        return {
            "ok": not errors,
            "mode": "preflight",
            "work_dir": str(work_dir),
            "config": str(config_path),
            "opencv": opencv_info,
            "camera_input": {
                "camera_id": camera_id,
                "source": camera.source,
                "protocol": camera.protocol,
                "probe_source": probe_source,
                "probe_protocol": probe_protocol,
                "resolution": camera.resolution,
                "preflight_measure_seconds": measure_seconds,
                "requested_preflight_measure_seconds": args.preflight_measure_seconds,
                "effective_preflight_measure_seconds": measure_seconds,
                "usb_selection": usb_selection,
            },
            "go2rtc": go2rtc_info,
            "usb_devices": usb_devices,
            "windows_camera_privacy": camera_privacy,
            "dshow_devices": dshow_devices,
            "capture_probe": capture_probe,
            "expected_degradations": _expected_degradations(
                use_yolo=args.use_yolo,
                protocol=str(camera.protocol),
            ),
            "hints": hints,
            "errors": errors,
        }
    finally:
        if go2rtc is not None:
            go2rtc.close()
        if temp_ctx is not None:
            temp_ctx.cleanup()


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    base_work_dir = args.work_dir.resolve() if args.work_dir else None

    temp_ctx = None
    if base_work_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="argus-core-smoke-")
        work_dir = Path(temp_ctx.name)
    else:
        work_dir = base_work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    database: Database | None = None
    manager: CameraManager | None = None
    dispatcher: AlertDispatcher | None = None
    go2rtc = None
    try:
        video_path = work_dir / "dev_camera.avi"
        camera_resolution = _parse_resolution(args.camera_resolution)
        if args.camera_source is None:
            create_dev_video(
                video_path,
                width=camera_resolution[0],
                height=camera_resolution[1],
                fps=10,
                seconds=max(args.video_seconds, 20),
                anomaly_start_s=6.0,
                motion=args.dev_video_motion,
            )
        config = _prepare_config(
            config_path,
            work_dir,
            video_path,
            use_yolo=args.use_yolo,
            camera_source=args.camera_source,
            camera_protocol=args.camera_protocol,
            camera_id=args.camera_id,
            camera_name=args.camera_name,
            camera_resolution=camera_resolution,
            usb_device_name=args.usb_device_name,
            usb_device_id=args.usb_device_id,
            go2rtc_enabled=False if args.disable_go2rtc else None,
        )
        camera_id = config.cameras[0].camera_id
        go2rtc, stream_registry, go2rtc_resolution = _start_go2rtc_for_smoke(
            config,
            require_go2rtc=args.require_go2rtc,
        )

        database = Database(database_url=config.storage.database_url)
        database.initialize()
        recording_store = AlertRecordingStore(
            archive_dir=str(work_dir / "recordings")
        )
        health = HealthMonitor()
        events: list[tuple[str, dict[str, Any]]] = []
        dispatcher = AlertDispatcher(
            config.alerts,
            database,
            alerts_dir=Path(config.storage.alerts_dir),
            on_alert=lambda topic, payload: events.append((topic, payload)),
            audio_config=getattr(config.dashboard, "audio_alerts", None),
        )
        manager = CameraManager(
            [],
            config.alerts,
            on_alert=dispatcher.dispatch,
            on_status_change=lambda topic, payload: events.append((topic, payload)),
            health_monitor=health,
            database=database,
            alert_recording_store=recording_store,
        )
        manager.add_camera_config(config.cameras[0])

        app = create_app(
            database=database,
            camera_manager=manager,
            health_monitor=health,
            alerts_dir=str(config.storage.alerts_dir),
            config=config,
            config_path=None,
            go2rtc_instance=go2rtc,
            stream_registry=stream_registry,
        )
        app.state.recording_store = recording_store
        client = TestClient(app)

        camera_flow = _run_camera_alert_replay_reports(
            client=client,
            manager=manager,
            dispatcher=dispatcher,
            events=events,
            camera_id=camera_id,
            timeout_s=args.timeout,
            recording_timeout_s=args.recording_timeout,
            require_go2rtc=args.require_go2rtc,
            activation_delay_s=args.activation_delay,
        )

        manager.stop_all()
        manager = None
        dispatcher.flush_db_queue()

        model_system_flow = _run_models_and_system_smoke(
            client=client,
            app=app,
            database=database,
            work_dir=work_dir,
            camera_id=camera_id,
        )

        return {
            "ok": True,
            "work_dir": str(work_dir),
            "config": str(config_path),
            "video": str(video_path) if args.camera_source is None else None,
            "camera_input": {
                "camera_id": camera_id,
                "source": config.cameras[0].source,
                "protocol": config.cameras[0].protocol,
                "go2rtc_enabled": config.dashboard.go2rtc_enabled,
                "go2rtc_running": bool(getattr(go2rtc, "running", False)),
                "go2rtc_resolutions": go2rtc_resolution,
            },
            "expected_degradations": _expected_degradations(
                use_yolo=args.use_yolo,
                protocol=str(config.cameras[0].protocol),
            ),
            "events_seen": len(events),
            "camera_alert_replay_reports": camera_flow,
            "models_system": model_system_flow,
        }
    finally:
        if manager is not None:
            manager.stop_all()
        if dispatcher is not None:
            dispatcher.close()
        if go2rtc is not None:
            go2rtc.close()
        if database is not None:
            database.close()
        if temp_ctx is not None and args.keep_work_dir:
            # TemporaryDirectory cannot be preserved directly; copy path is
            # printed before cleanup if callers want to reproduce with --work-dir.
            print(
                "NOTE: --keep-work-dir only preserves explicit --work-dir paths; "
                f"temporary path {work_dir} will be removed.",
                file=sys.stderr,
            )
        if temp_ctx is not None:
            temp_ctx.cleanup()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Argus core loop local smoke")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Only check dependencies, go2rtc registration, and first frame capture.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base config to derive from (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for smoke data. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep an explicit --work-dir for inspection after the run.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=75.0,
        help="Seconds to wait for camera alert generation (default: 75)",
    )
    parser.add_argument(
        "--recording-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for post-trigger recording completion (default: 45)",
    )
    parser.add_argument(
        "--video-seconds",
        type=int,
        default=20,
        help="Length of generated looping dev video (default: 20)",
    )
    parser.add_argument(
        "--dev-video-motion",
        choices=DEV_VIDEO_MOTIONS,
        default="settle",
        help=(
            "Generated dev video pattern. Use stable for no-alert checks, book "
            "to simulate a book being placed on an empty table, or projectile "
            "to simulate a small fast fly-through object."
        ),
    )
    parser.add_argument(
        "--camera-source",
        default=None,
        help=(
            "Use a real camera/video source instead of the generated dev video. "
            "Examples: 0 for USB, rtsp://host/stream for RTSP, or a video file path."
        ),
    )
    parser.add_argument(
        "--camera-protocol",
        choices=("file", "usb", "rtsp"),
        default=None,
        help="Protocol for --camera-source. Inferred from the source when omitted.",
    )
    parser.add_argument(
        "--camera-id",
        default="dev_cam",
        help="Camera ID used inside the temporary smoke config (default: dev_cam).",
    )
    parser.add_argument(
        "--camera-name",
        default=None,
        help="Camera display name used inside the temporary smoke config.",
    )
    parser.add_argument(
        "--camera-resolution",
        default="640,480",
        help="Expected capture resolution as WIDTH,HEIGHT or WIDTHxHEIGHT (default: 640,480).",
    )
    parser.add_argument(
        "--usb-device-name",
        default=None,
        help="Stable USB camera display name for go2rtc/DirectShow selection.",
    )
    parser.add_argument(
        "--usb-device-id",
        default=None,
        help="Stable USB camera DirectShow alternative name or PNP identifier for go2rtc selection.",
    )
    parser.add_argument(
        "--disable-go2rtc",
        action="store_true",
        help="Force MJPEG fallback even when --camera-source is USB/RTSP.",
    )
    parser.add_argument(
        "--require-go2rtc",
        action="store_true",
        help="Fail if USB/RTSP hardware is not managed by go2rtc and exposed to the browser.",
    )
    parser.add_argument(
        "--activation-delay",
        type=float,
        default=0.0,
        help=(
            "Seconds to wait after SSIM baseline calibration before switching "
            "the camera to active mode. Useful for hardware smoke runs where "
            "an operator needs time to introduce a test object."
        ),
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait per capture backend during --preflight (default: 3).",
    )
    parser.add_argument(
        "--preflight-measure-seconds",
        type=float,
        default=2.0,
        help=(
            "Seconds to sample decoded frames for preflight FPS measurement "
            "(default: 2; USB fast-motion checks use at least 15)."
        ),
    )
    parser.add_argument(
        "--use-yolo",
        action="store_true",
        help="Use the config's YOLO person detector instead of forcing graceful offline mode.",
    )
    args = parser.parse_args(argv)
    if args.disable_go2rtc and args.require_go2rtc:
        parser.error("--disable-go2rtc and --require-go2rtc cannot be used together")
    if args.camera_protocol is not None and args.camera_source is None:
        parser.error("--camera-protocol requires --camera-source")
    if args.activation_delay < 0:
        parser.error("--activation-delay must be non-negative")
    if args.preflight_timeout <= 0:
        parser.error("--preflight-timeout must be positive")
    if args.preflight_measure_seconds <= 0:
        parser.error("--preflight-measure-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        result = run_preflight(args) if args.preflight else run_smoke(args)
    except SmokeFailure as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
