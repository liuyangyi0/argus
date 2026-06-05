"""Browser smoke for the real Dashboard business data path.

This starts ``python -m argus`` with a temporary config derived from
``configs/default.yaml``. By default it uses ``--dev-video`` for a deterministic
local file camera, but it can also point the same business path at a USB/RTSP
or existing file source via ``--camera-source``. It waits for the camera to
produce a real alert with completed replay evidence, checks the
authoritative JSON APIs, runs a deterministic dev-fast training/export job,
seeds model release records in the temporary registry, then opens the resulting
business pages with a headless browser:

    Camera detail -> Alerts deep link -> Replay -> Models -> System -> Reports

The temporary config points all writable storage paths at the smoke work dir so
the run does not mutate ``data/`` or the checked-in default config.

Use ``--rtsp-fixture`` to publish the generated development video through a
local go2rtc RTSP server before Argus starts. That exercises the RTSP capture
path and Argus' own go2rtc browser playback metadata without requiring camera
hardware.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from argus.config.loader import load_config, save_config
from argus.runtime.dev_video import DEV_VIDEO_MOTIONS, create_dev_video
from argus.runtime.dev_training import write_dev_openvino_ir
from scripts.smoke_dashboard_routes import (
    DashboardSmokeFailure,
    _dump_dom_with_browser,
    _find_headless_browser,
    _free_port,
    _tail,
    _wait_for_camera,
)


class DashboardBusinessSmokeFailure(RuntimeError):
    """Raised when the real business browser path is not proven."""


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
        raise DashboardBusinessSmokeFailure(
            f"Invalid --camera-resolution {value!r}; expected WIDTH,HEIGHT"
        ) from exc
    if width <= 0 or height <= 0:
        raise DashboardBusinessSmokeFailure("--camera-resolution width and height must be positive")
    return width, height


def _expected_degradations(*, use_yolo: bool, protocol: str | None) -> list[dict[str, Any]]:
    degradations: list[dict[str, Any]] = []
    if not use_yolo:
        degradations.append(
            {
                "component": "person_filter",
                "expected": True,
                "reason": "business smoke uses missing-yolo.pt to verify graceful offline mode",
            }
        )
    if protocol in {"file", "usb", "rtsp", None}:
        degradations.append(
            {
                "component": "anomaly_detector",
                "expected": True,
                "reason": "business smoke expects SSIM fallback/unloaded status before model promotion",
            }
        )
    return degradations


def _rtsp_fixture_anomaly_start_s(seconds: int) -> float:
    """Delay RTSP fixture anomalies past service startup and baseline calibration."""
    return max(1.0, min(18.0, float(seconds) - 10.0))


class _RtspFixture:
    """Local go2rtc RTSP source backed by a generated development video."""

    def __init__(
        self,
        *,
        work_dir: Path,
        resolution: tuple[int, int],
        seconds: int,
        motion: str = "settle",
        stream_name: str = "argus_rtsp_fixture",
    ) -> None:
        self.work_dir = work_dir
        self.resolution = resolution
        self.seconds = seconds
        self.motion = motion
        self.stream_name = stream_name
        self.video_path = work_dir / "rtsp_fixture.avi"
        self.api_port = _free_port()
        self.rtsp_port = _free_port()
        self.webrtc_port = _free_port()
        self._manager = None
        self.info: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        from argus.streaming.go2rtc_manager import Go2RTCManager

        width, height = self.resolution
        anomaly_start_s = _rtsp_fixture_anomaly_start_s(self.seconds)
        meta = create_dev_video(
            self.video_path,
            width=width,
            height=height,
            fps=10,
            seconds=self.seconds,
            anomaly_start_s=anomaly_start_s,
            motion=self.motion,
        )
        manager = Go2RTCManager(
            api_port=self.api_port,
            rtsp_port=self.rtsp_port,
            webrtc_port=self.webrtc_port,
        )
        source = f"ffmpeg:{self.video_path}#video=h264"
        self._manager = manager
        manager.start(initial_streams={self.stream_name: source})
        self.info = {
            "source_url": f"rtsp://127.0.0.1:{self.rtsp_port}/{self.stream_name}",
            "stream_name": self.stream_name,
            "video_path": str(self.video_path),
            "go2rtc_source": source,
            "api_port": self.api_port,
            "rtsp_port": self.rtsp_port,
            "webrtc_port": self.webrtc_port,
            "seconds": self.seconds,
            "anomaly_start_s": anomaly_start_s,
            "video": meta,
        }
        return self.info

    def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None


def _api_data(response: httpx.Response, *, label: str) -> Any:
    if response.status_code >= 400:
        raise DashboardBusinessSmokeFailure(
            f"{label} failed: HTTP {response.status_code} {response.text[:500]}"
        )
    payload = response.json()
    if payload.get("code") != 0:
        raise DashboardBusinessSmokeFailure(f"{label} failed: {payload}")
    return payload.get("data")


def _wait_for(
    label: str,
    predicate,
    *,
    timeout_s: float,
    interval_s: float = 0.5,
    process: subprocess.Popen | None = None,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise DashboardBusinessSmokeFailure(
                f"argus exited early with code {process.returncode}"
            )
        try:
            value = predicate()
            if value:
                return value
        except Exception as exc:
            last_error = exc
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval_s, remaining))
    if process is not None and process.poll() is not None:
        raise DashboardBusinessSmokeFailure(
            f"argus exited early with code {process.returncode}"
        )
    try:
        value = predicate()
        if value:
            return value
    except Exception as exc:
        last_error = exc
    suffix = f" Last error: {last_error}" if last_error else ""
    raise DashboardBusinessSmokeFailure(f"Timed out waiting for {label}.{suffix}")


def _websocket_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws"


def _go2rtc_api_running(api_port: int) -> bool:
    try:
        resp = httpx.get(f"http://127.0.0.1:{api_port}/api", timeout=1.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _cleanup_go2rtc_api_port(api_port: int | None) -> None:
    """Stop a go2rtc process bound to a smoke-owned API port."""
    if not api_port:
        return

    try:
        httpx.post(f"http://127.0.0.1:{api_port}/api/exit", timeout=1.0)
    except httpx.HTTPError:
        pass

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _go2rtc_api_running(api_port):
            return
        time.sleep(0.2)

    if os.name != "nt":
        return

    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return

    for line in result.stdout.splitlines():
        if f":{api_port}" not in line or "LISTENING" not in line:
            continue
        parts = line.strip().split()
        if not parts:
            continue
        subprocess.run(
            ["taskkill", "/F", "/PID", parts[-1]],
            capture_output=True,
            text=True,
            timeout=5,
        )
        break


class _AlertWebSocketListener:
    def __init__(self, *, base_url: str, timeout_s: float = 10.0) -> None:
        self._url = _websocket_url(base_url)
        self._timeout_s = timeout_s
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._errors: queue.Queue[str] = queue.Queue()
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection = None

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._run,
            name="argus-business-smoke-ws",
            daemon=True,
        )
        self._thread.start()
        if not self._connected.wait(self._timeout_s):
            error = self._errors.get_nowait() if not self._errors.empty() else "connection timeout"
            raise DashboardBusinessSmokeFailure(f"alert websocket did not connect: {error}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        connection = self._connection
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        try:
            from websockets.sync.client import connect

            with connect(self._url, open_timeout=self._timeout_s, close_timeout=1.0) as websocket:
                self._connection = websocket
                websocket.send(json.dumps({"action": "subscribe", "topics": ["alerts"]}))
                self._connected.set()
                while not self._stop.is_set():
                    try:
                        raw = websocket.recv(timeout=0.5)
                    except TimeoutError:
                        continue
                    self._handle_raw_message(raw)
        except Exception as exc:
            if self._stop.is_set() and type(exc).__name__ == "ConnectionClosedOK":
                self._connected.set()
                return
            self._errors.put(f"{type(exc).__name__}: {exc}")
            self._connected.set()

    def _handle_raw_message(self, raw: str) -> None:
        payload = json.loads(raw)
        if payload.get("topic") == "ping":
            connection = self._connection
            if connection is not None:
                connection.send(json.dumps({"action": "pong"}))
            return
        if payload.get("topic") == "alerts" and isinstance(payload.get("data"), dict):
            self._messages.put(payload["data"])

    def wait_for_alert(self, alert_id: str, *, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last_seen: list[str] = []
        while time.monotonic() < deadline:
            remaining = max(0.1, min(0.5, deadline - time.monotonic()))
            try:
                payload = self._messages.get(timeout=remaining)
            except queue.Empty:
                if not self._errors.empty():
                    raise DashboardBusinessSmokeFailure(
                        f"alert websocket listener failed: {self._errors.get_nowait()}"
                    )
                continue
            seen_id = str(payload.get("alert_id") or "")
            if seen_id:
                last_seen.append(seen_id)
            if seen_id == alert_id:
                return payload
        raise DashboardBusinessSmokeFailure(
            f"Timed out waiting for alert websocket payload {alert_id}; last_seen={last_seen[-5:]}"
        )


def _prepare_runtime_config(
    *,
    config_path: Path,
    work_dir: Path,
    port: int,
    use_yolo: bool,
    camera_source: str | None = None,
    camera_protocol: str | None = None,
    camera_id: str | None = None,
    camera_name: str | None = None,
    camera_resolution: tuple[int, int] | None = None,
    usb_device_name: str | None = None,
    usb_device_id: str | None = None,
    go2rtc_enabled: bool | None = None,
) -> tuple[Path, str]:
    config = load_config(config_path).model_copy(deep=True)
    if not config.cameras:
        raise DashboardBusinessSmokeFailure("config has no cameras")

    camera = config.cameras[0].model_copy(deep=True)
    if camera_id:
        camera.camera_id = camera_id
    if camera_name:
        camera.name = camera_name
    if camera_resolution:
        camera.resolution = camera_resolution
    if camera_source is not None:
        camera.source = camera_source
        camera.protocol = camera_protocol or _infer_camera_protocol(camera_source)
        if go2rtc_enabled is None:
            go2rtc_enabled = camera.protocol in {"usb", "rtsp"}
    elif go2rtc_enabled is None:
        go2rtc_enabled = False

    camera.ring_buffer.enabled = True
    if camera.protocol == "usb":
        if usb_device_name is not None:
            camera.usb.device_name = usb_device_name
        if usb_device_id is not None:
            camera.usb.device_id = usb_device_id
    camera.anomaly.ssim_baseline_frames = min(camera.anomaly.ssim_baseline_frames, 15)
    if not use_yolo:
        camera.person_filter.model_name = str(work_dir / "missing-yolo.pt")
    config.cameras = [camera]

    camera.anomaly.min_shadow_days = 0
    camera.anomaly.min_canary_days = 0
    # The business smoke must exercise real Replay clips, not INFO-only
    # trigger-frame snapshots. RTSP fixture compression can lower the fallback
    # anomaly score to ~0.6, so make that a LOW alert in the isolated config.
    thresholds = config.alerts.severity_thresholds
    thresholds.low = max(thresholds.info + 0.01, min(thresholds.low, 0.55))

    config.dashboard.host = "127.0.0.1"
    config.dashboard.port = port
    config.dashboard.go2rtc_enabled = bool(go2rtc_enabled)
    if config.dashboard.go2rtc_enabled:
        config.dashboard.go2rtc_api_port = _free_port()
        config.dashboard.go2rtc_rtsp_port = _free_port()
        config.dashboard.go2rtc_webrtc_port = _free_port()

    config.storage.database_url = f"sqlite:///{work_dir / 'argus.db'}"
    config.storage.baselines_dir = work_dir / "baselines"
    config.storage.models_dir = work_dir / "models"
    config.storage.exports_dir = work_dir / "exports"
    config.storage.backbones_dir = work_dir / "backbones"
    config.storage.foe_objects_dir = work_dir / "foe_objects"
    config.storage.model_packages_dir = work_dir / "model_packages"
    config.storage.alerts_dir = work_dir / "alerts"
    config.storage.inference_records_dir = work_dir / "inference_records"
    config.models.yolo_path = camera.person_filter.model_name
    config.models.anomalib_model_dir = work_dir / "models"
    config.models.anomalib_export_dir = work_dir / "exports"
    config.logging.log_dir = work_dir / "logs"

    runtime_config = work_dir / "dashboard_business_config.yaml"
    save_config(config, runtime_config)
    return runtime_config, camera.camera_id


def _wait_for_detector_ready(
    client: httpx.Client,
    *,
    camera_id: str,
    timeout_s: float,
    process: subprocess.Popen,
) -> dict[str, Any]:
    def detector_ready():
        data = _api_data(
            client.get(f"/api/cameras/{camera_id}/detail/json", timeout=5),
            label="camera detail",
        )
        detector = data.get("detector") or {}
        if detector.get("ssim_calibrated") or detector.get("mode") not in {None, "", "ssim_fallback"}:
            return data
        return None

    return _wait_for(
        "detector fallback calibration",
        detector_ready,
        timeout_s=timeout_s,
        interval_s=0.5,
        process=process,
    )


def _should_wait_for_detector_before_no_alert(args: argparse.Namespace) -> bool:
    return (
        args.expect_no_alert
        and args.observe_mode in {"active", "maintenance"}
        and not args.allow_detection_limited_no_alert
    )


def _set_camera_mode(client: httpx.Client, *, camera_id: str, mode: str) -> dict[str, Any]:
    data = _api_data(
        client.post(f"/api/cameras/{camera_id}/mode", json={"mode": mode}, timeout=10),
        label=f"set camera {mode} mode",
    )
    if data.get("pipeline_mode") != mode:
        raise DashboardBusinessSmokeFailure(f"camera mode did not become {mode}: {data}")
    return data


def _activate_camera(client: httpx.Client, *, camera_id: str) -> dict[str, Any]:
    return _set_camera_mode(client, camera_id=camera_id, mode="active")


def _verify_camera_media_apis(
    client: httpx.Client,
    *,
    camera_id: str,
    require_go2rtc: bool,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    snapshot = client.get(f"/api/cameras/{camera_id}/snapshot", timeout=10)
    if snapshot.status_code != 200:
        raise DashboardBusinessSmokeFailure(
            f"camera snapshot failed: HTTP {snapshot.status_code} {snapshot.text[:300]}"
        )
    content_type = snapshot.headers.get("content-type", "")
    if "image/jpeg" not in content_type:
        raise DashboardBusinessSmokeFailure(f"camera snapshot content-type mismatch: {content_type}")
    if len(snapshot.content) < 1024:
        raise DashboardBusinessSmokeFailure(f"camera snapshot is too small: {len(snapshot.content)} bytes")

    snapshot_path: Path | None = None
    if work_dir is not None:
        snapshot_path = work_dir / f"{camera_id}_camera_snapshot.jpg"
        snapshot_path.write_bytes(snapshot.content)

    snapshot_brightness_mean: float | None = None
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(snapshot.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            snapshot_brightness_mean = float(image.mean())
    except Exception:
        snapshot_brightness_mean = None

    streaming = _api_data(
        client.get(f"/api/streaming/{camera_id}", timeout=10),
        label="camera streaming info",
    )
    expected_fallback = f"/api/cameras/{camera_id}/stream"
    if streaming.get("fallback") != expected_fallback:
        raise DashboardBusinessSmokeFailure(f"streaming fallback mismatch: {streaming}")
    if require_go2rtc and not streaming.get("go2rtc"):
        raise DashboardBusinessSmokeFailure(f"go2rtc was required but not exposed: {streaming}")

    return {
        "snapshot": {
            "status": snapshot.status_code,
            "content_type": content_type,
            "bytes": len(snapshot.content),
            "path": str(snapshot_path) if snapshot_path is not None else None,
            "brightness_mean": snapshot_brightness_mean,
        },
        "streaming": {
            "go2rtc": streaming.get("go2rtc"),
            "fallback": streaming.get("fallback"),
        },
    }


def _recent_alerts(client: httpx.Client, *, camera_id: str, limit: int = 10) -> list[dict[str, Any]]:
    data = _api_data(
        client.get("/api/alerts/json", params={"camera_id": camera_id, "limit": limit}, timeout=5),
        label="alerts json",
    )
    return list(data.get("alerts") or [])


def _wait_for_completed_alert(
    client: httpx.Client,
    *,
    camera_id: str,
    alert_timeout_s: float,
    recording_timeout_s: float,
    process: subprocess.Popen,
    realtime_listener: _AlertWebSocketListener | None = None,
    known_alert_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    known_alert_ids = known_alert_ids or set()

    def latest_alert():
        alerts = [
            alert
            for alert in _recent_alerts(client, camera_id=camera_id)
            if str(alert.get("alert_id") or "") not in known_alert_ids
        ]
        return alerts[0] if alerts else None

    alert = _wait_for(
        "alert generated from dev video",
        latest_alert,
        timeout_s=alert_timeout_s,
        interval_s=1.0,
        process=process,
    )
    alert_id = alert["alert_id"]
    if realtime_listener is not None:
        realtime_payload = realtime_listener.wait_for_alert(
            alert_id,
            timeout_s=min(20.0, recording_timeout_s),
        )
        if realtime_payload.get("camera_id") != camera_id:
            raise DashboardBusinessSmokeFailure(
                f"alert websocket camera mismatch: {realtime_payload}"
            )
        if realtime_payload.get("severity") != alert.get("severity"):
            raise DashboardBusinessSmokeFailure(
                f"alert websocket severity mismatch: {realtime_payload}"
            )
        if not realtime_payload.get("snapshot_path") or not realtime_payload.get("heatmap_path"):
            raise DashboardBusinessSmokeFailure(
                f"alert websocket evidence fields missing: {realtime_payload}"
            )
        alert["_realtime_payload"] = realtime_payload

    last_detail: dict[str, Any] | None = None

    def recording_complete():
        nonlocal last_detail
        detail = _api_data(
            client.get(f"/api/alerts/{alert_id}/detail", timeout=5),
            label="alert detail",
        )
        last_detail = detail
        if (
            detail.get("has_recording")
            and detail.get("recording_status") == "complete"
            and detail.get("snapshot_path")
            and detail.get("heatmap_path")
        ):
            return detail
        return None

    try:
        detail = _wait_for(
            "alert recording completion",
            recording_complete,
            timeout_s=recording_timeout_s,
            interval_s=1.0,
            process=process,
        )
    except DashboardBusinessSmokeFailure as exc:
        evidence_state = {
            key: last_detail.get(key) if last_detail else None
            for key in (
                "alert_id",
                "has_recording",
                "recording_status",
                "snapshot_path",
                "heatmap_path",
            )
        }
        raise DashboardBusinessSmokeFailure(
            f"{exc} Last alert evidence state: "
            f"{json.dumps(evidence_state, ensure_ascii=False, default=str)}"
        ) from exc
    return alert, detail


def _verify_no_alert_window(
    client: httpx.Client,
    *,
    camera_id: str,
    observe_seconds: float,
    process: subprocess.Popen,
    known_alert_ids: set[str] | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + observe_seconds
    polls = 0
    known_alert_ids = known_alert_ids or set()
    while True:
        if process.poll() is not None:
            raise DashboardBusinessSmokeFailure(
                f"argus exited early with code {process.returncode}"
            )
        alerts = [
            alert
            for alert in _recent_alerts(client, camera_id=camera_id, limit=20)
            if str(alert.get("alert_id") or "") not in known_alert_ids
        ]
        polls += 1
        if alerts:
            alert = alerts[0]
            realtime = alert.get("_realtime_payload") or alert.get("realtime") or {}
            raise DashboardBusinessSmokeFailure(
                "expected no alerts but observed "
                f"{alert.get('alert_id')} severity={alert.get('severity')} "
                f"category={realtime.get('category') or alert.get('category')} "
                f"detection_type={realtime.get('detection_type') or alert.get('detection_type')}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))
    return {
        "observed_seconds": observe_seconds,
        "polls": polls,
        "alerts_seen": 0,
    }


def _verify_no_alert_detector_ready(
    *,
    detector: dict[str, Any],
    observe_mode: str,
    allow_detection_limited: bool = False,
) -> dict[str, Any]:
    """Ensure active/maintenance no-alert observations did not pass with detection disabled."""
    if observe_mode in {"collection", "training"}:
        return {
            "checked": False,
            "reason": f"{observe_mode} mode intentionally disables alert detection",
        }
    if allow_detection_limited:
        return {
            "checked": False,
            "reason": "--allow-detection-limited-no-alert was set",
        }

    failures: list[str] = []
    if detector.get("low_light"):
        failures.append(
            f"low_light=true brightness={detector.get('last_brightness')}"
        )
    if detector.get("detection_limited"):
        failures.append(
            "detection_limited=true "
            f"reason={detector.get('detection_limited_reason')}"
        )
    if detector.get("ssim_calibration_blocked"):
        failures.append(
            "ssim_calibration_blocked=true "
            f"reason={detector.get('ssim_calibration_blocked_reason')}"
        )

    detector_mode = detector.get("mode")
    if detector_mode in {None, "", "ssim_fallback"} and not detector.get("ssim_calibrated"):
        failures.append("ssim fallback is not calibrated")

    if failures:
        raise DashboardBusinessSmokeFailure(
            "no-alert observation is not valid because detection was limited: "
            + "; ".join(failures)
        )

    return {
        "checked": True,
        "mode": detector_mode,
        "ssim_calibrated": detector.get("ssim_calibrated"),
        "detection_limited": detector.get("detection_limited"),
        "low_light": detector.get("low_light"),
    }


def _extract_alert_semantics(alert: dict[str, Any]) -> dict[str, Any]:
    realtime = alert.get("_realtime_payload") or alert.get("realtime") or {}
    detected_objects = realtime.get("detected_objects") or alert.get("detected_objects") or []
    detected_object_classes: list[str] = []
    if isinstance(detected_objects, list):
        for item in detected_objects:
            if not isinstance(item, dict):
                continue
            class_name = item.get("class_name") or item.get("class")
            if class_name:
                detected_object_classes.append(str(class_name))
    classification_confidence = (
        realtime.get("classification_confidence")
        if "classification_confidence" in realtime
        else alert.get("classification_confidence")
    )
    return {
        "detection_type": str(realtime.get("detection_type") or alert.get("detection_type") or ""),
        "category": str(realtime.get("category") or alert.get("category") or ""),
        "detected_object_classes": sorted(set(detected_object_classes)),
        "classification_label": (
            realtime.get("classification_label")
            if realtime.get("classification_label") is not None
            else alert.get("classification_label")
        ),
        "classification_confidence": classification_confidence,
    }


def _verify_projectile_evidence(alert: dict[str, Any]) -> dict[str, Any]:
    realtime = alert.get("_realtime_payload") or alert.get("realtime") or {}
    detected_objects = realtime.get("detected_objects") or alert.get("detected_objects") or []
    fast_objects = [
        item
        for item in detected_objects
        if str(item.get("class_name") or item.get("class") or "").lower() == "fast_projectile"
    ]
    if not fast_objects:
        raise DashboardBusinessSmokeFailure(
            "projectile alert missing fast_projectile detected_object evidence"
        )

    primary = fast_objects[0]
    bbox = primary.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        raise DashboardBusinessSmokeFailure("projectile alert missing bbox evidence")

    speed = (
        realtime.get("speed_px_per_sec")
        or primary.get("speed_px_per_sec")
        or alert.get("speed_px_per_sec")
    )
    if not isinstance(speed, int | float) or speed <= 0:
        raise DashboardBusinessSmokeFailure("projectile alert missing positive speed evidence")

    trajectory = (
        realtime.get("trajectory_points")
        or primary.get("trajectory_points")
        or alert.get("trajectory_points")
    )
    if not isinstance(trajectory, list) or not trajectory:
        raise DashboardBusinessSmokeFailure("projectile alert missing trajectory evidence")

    trajectory_model = realtime.get("trajectory_model") or alert.get("trajectory_model")
    if trajectory_model != "projectile":
        raise DashboardBusinessSmokeFailure(
            f"projectile alert trajectory_model={trajectory_model!r}; expected 'projectile'"
        )

    return {
        "detected_object_class": "fast_projectile",
        "bbox": [int(value) for value in bbox],
        "speed_px_per_sec": float(speed),
        "trajectory_model": trajectory_model,
        "trajectory_points": len(trajectory),
    }


def _verify_alert_semantic_expectations(
    args: argparse.Namespace,
    alert: dict[str, Any],
) -> dict[str, Any]:
    """Verify operator-declared semantic expectations for physical scenario smokes."""
    semantics = _extract_alert_semantics(alert)
    checks = {
        "detection_type": {
            "actual": semantics["detection_type"],
            "expected": set(args.expect_detection_type or []),
            "forbidden": set(args.forbid_detection_type or []),
        },
        "category": {
            "actual": semantics["category"],
            "expected": set(args.expect_alert_category or []),
            "forbidden": set(args.forbid_alert_category or []),
        },
    }
    detected_class_set = set(semantics["detected_object_classes"])
    expected_detected_classes = set(args.expect_detected_object_class or [])
    forbidden_detected_classes = set(args.forbid_detected_object_class or [])
    for field, item in checks.items():
        actual = item["actual"]
        expected = item["expected"]
        forbidden = item["forbidden"]
        if expected and actual not in expected:
            raise DashboardBusinessSmokeFailure(
                f"alert {field}={actual!r} did not match expected values "
                f"{sorted(expected)!r}"
            )
        if forbidden and actual in forbidden:
            raise DashboardBusinessSmokeFailure(
                f"alert {field}={actual!r} matched forbidden values "
                f"{sorted(forbidden)!r}"
            )
    missing_detected_classes = sorted(expected_detected_classes - detected_class_set)
    if missing_detected_classes:
        raise DashboardBusinessSmokeFailure(
            "alert detected_object_classes missing expected values "
            f"{missing_detected_classes!r}; actual={semantics['detected_object_classes']!r}"
        )
    forbidden_detected_matches = sorted(forbidden_detected_classes & detected_class_set)
    if forbidden_detected_matches:
        raise DashboardBusinessSmokeFailure(
            "alert detected_object_classes matched forbidden values "
            f"{forbidden_detected_matches!r}"
        )
    result = {
        **semantics,
        "expected_detection_type": sorted(checks["detection_type"]["expected"]),
        "expected_category": sorted(checks["category"]["expected"]),
        "expected_detected_object_classes": sorted(expected_detected_classes),
        "forbidden_detection_type": sorted(checks["detection_type"]["forbidden"]),
        "forbidden_category": sorted(checks["category"]["forbidden"]),
        "forbidden_detected_object_classes": sorted(forbidden_detected_classes),
    }
    expects_projectile = (
        "projectile" in checks["detection_type"]["expected"]
        or "projectile" in checks["category"]["expected"]
        or "fast_projectile" in expected_detected_classes
    )
    if expects_projectile:
        result["projectile_evidence"] = _verify_projectile_evidence(alert)
    return result


def _verify_dev_video_alert_semantics(
    args: argparse.Namespace,
    alert: dict[str, Any],
) -> dict[str, Any] | None:
    if args.camera_source is not None or args.dev_video_motion != "book":
        return None

    semantics = _extract_alert_semantics(alert)
    if semantics["detection_type"] == "projectile" or semantics["category"] == "projectile":
        raise DashboardBusinessSmokeFailure(
            "book dev-video scenario was reported as projectile; expected a "
            "static scene-change/foreign-object anomaly"
        )
    return {
        "motion": args.dev_video_motion,
        **semantics,
    }


def _physical_action_window_message(args: argparse.Namespace) -> str:
    """Build the operator prompt printed exactly when active detection starts."""
    expected_parts = []
    forbidden_parts = []
    if args.expect_alert_category:
        expected_parts.append(f"category in {sorted(args.expect_alert_category)!r}")
    if args.expect_detection_type:
        expected_parts.append(f"detection_type in {sorted(args.expect_detection_type)!r}")
    if args.expect_detected_object_class:
        expected_parts.append(
            f"detected object classes include {sorted(args.expect_detected_object_class)!r}"
        )
    if args.forbid_alert_category:
        forbidden_parts.append(f"category not in {sorted(args.forbid_alert_category)!r}")
    if args.forbid_detection_type:
        forbidden_parts.append(f"detection_type not in {sorted(args.forbid_detection_type)!r}")
    if args.forbid_detected_object_class:
        forbidden_parts.append(
            f"detected object classes exclude {sorted(args.forbid_detected_object_class)!r}"
        )

    message = (
        f"[argus] camera {args.observe_mode}; introduce the physical test target within "
        f"{args.activation_delay:.1f}s"
    )
    if expected_parts:
        message += f". Expected: {', '.join(expected_parts)}"
    if forbidden_parts:
        message += f". Forbidden: {', '.join(forbidden_parts)}"
    return f"{message}."


def _verify_business_apis(
    client: httpx.Client,
    *,
    alert_id: str,
    camera_id: str,
) -> dict[str, Any]:
    evidence_zip = client.get(f"/api/alerts/{alert_id}/evidence.zip", timeout=15)
    if evidence_zip.status_code != 200 or len(evidence_zip.content) < 1024:
        raise DashboardBusinessSmokeFailure(
            f"evidence zip invalid: HTTP {evidence_zip.status_code}, bytes={len(evidence_zip.content)}"
        )

    replay_meta = _api_data(
        client.get(f"/api/replay/{alert_id}/metadata", timeout=10),
        label="replay metadata",
    )
    if replay_meta.get("status") != "complete" or replay_meta.get("frame_count", 0) < 2:
        raise DashboardBusinessSmokeFailure(f"replay metadata incomplete: {replay_meta}")

    replay_signals = _api_data(
        client.get(f"/api/replay/{alert_id}/signals", timeout=10),
        label="replay signals",
    )
    if not replay_signals.get("timestamps"):
        raise DashboardBusinessSmokeFailure("replay signals missing timestamps")

    reports = _api_data(client.get("/api/reports/json", timeout=10), label="reports json")
    evidence = reports.get("evidence") or {}
    if evidence.get("total_alerts", 0) < 1 or evidence.get("evidence_complete_rate") != 100.0:
        raise DashboardBusinessSmokeFailure(f"reports evidence stats incomplete: {evidence}")

    camera_dist = _api_data(
        client.get("/api/reports/camera-dist/json", params={"days": 30}, timeout=10),
        label="reports camera distribution",
    )
    camera_rows = camera_dist.get("cameras") or []
    camera_row = next((item for item in camera_rows if item.get("camera_id") == camera_id), None)
    if not camera_row or camera_row.get("count", 0) < 1:
        raise DashboardBusinessSmokeFailure(
            f"reports camera distribution missed {camera_id}: {camera_dist}"
        )

    return {
        "evidence_zip_bytes": len(evidence_zip.content),
        "replay": {
            "status": replay_meta.get("status"),
            "frame_count": replay_meta.get("frame_count"),
            "signal_points": len(replay_signals.get("timestamps") or []),
        },
        "reports": {
            "total_alerts": evidence.get("total_alerts"),
            "evidence_complete_rate": evidence.get("evidence_complete_rate"),
            "recording_rate": evidence.get("recording_rate"),
            "camera_distribution": camera_row,
        },
    }


def _model_artifact_dir(root: Path, camera_id: str, name: str) -> Path:
    model_dir = root / "models" / camera_id / "default" / name
    write_dev_openvino_ir(
        model_dir,
        camera_id=camera_id,
        zone_id="default",
        model_type="patchcore",
        image_size=256,
        quantization="fp16",
    )
    return model_dir


def _baseline_artifact_dir(root: Path, camera_id: str) -> Path:
    baseline_dir = root / "baselines" / camera_id / "default" / "smoke-baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "baseline.png").write_bytes(b"baseline")
    return baseline_dir


def _seed_model_registry(*, work_dir: Path, database_url: str, camera_id: str) -> dict[str, str]:
    """Create model records for all release stages in the temporary database."""
    from argus.storage.database import Database
    from argus.storage.model_registry import ModelRegistry
    from argus.storage.models import ModelStage

    db = Database(database_url=database_url)
    db.initialize()
    try:
        registry = ModelRegistry(session_factory=db.get_session)
        baseline_dir = _baseline_artifact_dir(work_dir, camera_id)
        seed_tag = f"seed-{time.time_ns()}"

        stage_ids: dict[str, str] = {}

        def register(name: str) -> str:
            return registry.register(
                _model_artifact_dir(work_dir, camera_id, f"{seed_tag}-{name}"),
                baseline_dir,
                camera_id,
                "patchcore",
                training_params={"smoke": True, "stage": name, "seed_tag": seed_tag},
            )

        stage_ids["candidate"] = register("candidate")

        shadow_id = register("shadow")
        registry.promote(shadow_id, ModelStage.SHADOW.value, triggered_by="smoke")
        stage_ids["shadow"] = shadow_id

        canary_id = register("canary")
        registry.promote(canary_id, ModelStage.SHADOW.value, triggered_by="smoke")
        registry.promote(
            canary_id,
            ModelStage.CANARY.value,
            triggered_by="smoke",
            canary_camera_id=camera_id,
        )
        stage_ids["canary"] = canary_id

        production_id = register("production")
        registry.promote(production_id, ModelStage.SHADOW.value, triggered_by="smoke")
        registry.promote(
            production_id,
            ModelStage.CANARY.value,
            triggered_by="smoke",
            canary_camera_id=camera_id,
        )
        registry.promote(production_id, ModelStage.PRODUCTION.value, triggered_by="smoke")
        stage_ids["production"] = production_id

        return stage_ids
    finally:
        db.close()


def _seed_training_baselines(
    *,
    work_dir: Path,
    camera_id: str,
    count: int,
    image_size: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    baseline_dir = work_dir / "baselines" / camera_id / "default" / "v001"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for index in range(count):
        brightness = 50 + int(150 * (index / max(count, 1)))
        frame = np.full((image_size, image_size, 3), brightness, dtype=np.uint8)
        noise = rng.integers(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(baseline_dir / f"baseline_{index:05d}.png"), frame)

    current = baseline_dir.parent / "current.txt"
    current.write_text(baseline_dir.name, encoding="utf-8")
    return {
        "path": str(baseline_dir),
        "count": count,
        "image_size": image_size,
    }


def _verify_system_degradation_apis(
    client: httpx.Client,
    *,
    camera_id: str,
) -> dict[str, Any]:
    anomaly_data = _api_data(
        client.get("/api/system/anomaly-degradation", timeout=10),
        label="system anomaly degradation",
    )
    anomaly = anomaly_data.get("anomaly") or {}
    cameras = anomaly.get("cameras") or []
    if not isinstance(anomaly.get("degraded"), bool):
        raise DashboardBusinessSmokeFailure(f"system anomaly degradation has invalid aggregate: {anomaly}")
    if cameras and not any(item.get("camera_id") == camera_id for item in cameras):
        raise DashboardBusinessSmokeFailure(
            f"system anomaly degradation missed {camera_id}: {anomaly}"
        )
    if cameras and anomaly.get("degraded") is not any(bool(item.get("degraded")) for item in cameras):
        raise DashboardBusinessSmokeFailure(f"system anomaly degradation aggregate mismatch: {anomaly}")

    active = _api_data(
        client.get("/api/degradation/active", timeout=10),
        label="degradation active",
    )
    summary = _api_data(
        client.get("/api/degradation/summary", timeout=10),
        label="degradation summary",
    )
    history = _api_data(
        client.get("/api/degradation/history", params={"days": 7}, timeout=10),
        label="degradation history",
    )
    active_items = active.get("items")
    history_items = history.get("items")
    if not isinstance(active_items, list) or not isinstance(history_items, list):
        raise DashboardBusinessSmokeFailure(
            f"global degradation endpoints returned invalid items: active={active}, history={history}"
        )
    if not isinstance(summary.get("active_count"), int) or not isinstance(summary.get("events"), list):
        raise DashboardBusinessSmokeFailure(f"global degradation summary is invalid: {summary}")
    if summary.get("active_count") != len(active_items):
        raise DashboardBusinessSmokeFailure(
            f"global degradation active count mismatch: active={active}, summary={summary}"
        )

    return {
        "anomaly": anomaly,
        "global": {
            "active_count": summary.get("active_count"),
            "max_level": summary.get("max_level"),
            "history_count": len(history_items),
        },
    }


def _verify_models_system_apis(
    client: httpx.Client,
    *,
    camera_id: str,
    model_ids: dict[str, str],
) -> dict[str, Any]:
    models_data = _api_data(
        client.get("/api/models/json", params={"camera_id": camera_id}, timeout=10),
        label="models json",
    )
    models = models_data.get("models") or []
    by_id = {item.get("model_version_id"): item for item in models}
    missing = [version_id for version_id in model_ids.values() if version_id not in by_id]
    if missing:
        raise DashboardBusinessSmokeFailure(f"models registry missed seeded versions: {missing}")

    expected_stages = {
        model_ids["candidate"]: "candidate",
        model_ids["shadow"]: "shadow",
        model_ids["canary"]: "canary",
        model_ids["production"]: "production",
    }
    for version_id, expected_stage in expected_stages.items():
        row = by_id[version_id]
        if row.get("stage") != expected_stage:
            raise DashboardBusinessSmokeFailure(
                f"model {version_id} stage mismatch: {row.get('stage')} != {expected_stage}"
            )
    if by_id[model_ids["production"]].get("is_active") is not True:
        raise DashboardBusinessSmokeFailure("seeded production model is not active")

    config_result = _api_data(
        client.post(
            "/api/config/detection-params",
            json={"anomaly_threshold": 0.66, "sev_low": 0.55},
            timeout=10,
        ),
        label="config detection params",
    )
    if config_result.get("pipelines_updated", 0) < 1:
        raise DashboardBusinessSmokeFailure(f"system config update did not reach pipelines: {config_result}")

    status_data = _api_data(client.get("/api/models/status", timeout=10), label="models status")
    model_statuses = status_data.get("models") or []
    fallback_rows = [
        item for item in model_statuses
        if item.get("camera_id") == camera_id
        and (
            not item.get("loaded")
            or item.get("backend") in {"ssim-fallback", "none"}
        )
    ]
    if not fallback_rows:
        raise DashboardBusinessSmokeFailure(
            f"system model status did not expose fallback/unloaded state for {camera_id}: {model_statuses}"
        )

    degradation = _verify_system_degradation_apis(client, camera_id=camera_id)

    return {
        "registry_total": models_data.get("total"),
        "seeded_versions": model_ids,
        "production_active": model_ids["production"],
        "config_update": config_result,
        "fallback_models": fallback_rows,
        "degradation": degradation,
    }


def _model_operation_timeout(timeout_s: float) -> float:
    """Bound long model-management HTTP calls for normal-training smoke runs."""
    return max(30.0, min(float(timeout_s), 180.0))


def _exercise_training_export_apis(
    client: httpx.Client,
    *,
    camera_id: str,
    training_mode: str,
    image_size: int,
    timeout_s: float,
    process: subprocess.Popen,
) -> dict[str, Any]:
    hyperparameters = {
        "skip_baseline_validation": training_mode == "dev-fast",
        "export_format": "openvino",
        "quantization": "fp16",
        "image_size": image_size,
    }
    job = _api_data(
        client.post(
            "/api/training-jobs/",
            json={
                "job_type": "anomaly_head",
                "camera_id": camera_id,
                "model_type": "patchcore",
                "hyperparameters": hyperparameters,
            },
            timeout=10,
        ),
        label="create training job",
    )
    job_id = job["job_id"]
    _api_data(
        client.post(
            f"/api/training-jobs/{job_id}/confirm",
            json={"confirmed_by": "smoke"},
            timeout=10,
        ),
        label="confirm training job",
    )

    def training_complete():
        detail = _api_data(
            client.get(f"/api/training-jobs/{job_id}", timeout=10),
            label="training job",
        )
        if detail.get("status") == "complete" and detail.get("model_version_id"):
            return detail
        if detail.get("status") == "failed":
            raise DashboardBusinessSmokeFailure(f"training job failed: {detail}")
        return None

    detail = _wait_for(
        f"{training_mode} training job completion",
        training_complete,
        timeout_s=timeout_s,
        interval_s=1.0,
        process=process,
    )
    version_id = detail["model_version_id"]

    reexport = _api_data(
        client.post(
            f"/api/models/{version_id}/reexport",
            json={"export_format": "openvino", "quantization": "fp16"},
            timeout=_model_operation_timeout(timeout_s),
        ),
        label="reexport trained model",
    )
    if reexport.get("status") != "ok":
        raise DashboardBusinessSmokeFailure(f"model reexport did not succeed: {reexport}")

    return {
        "training_mode": training_mode,
        "job_id": job_id,
        "model_version_id": version_id,
        "status": detail.get("status"),
        "artifacts_path": detail.get("artifacts_path"),
        "metrics": detail.get("metrics"),
        "reexport": reexport,
    }


def _exercise_model_release_apis(
    client: httpx.Client,
    *,
    camera_id: str,
    model_ids: dict[str, str],
    version_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    candidate_id = version_id
    operation_timeout = _model_operation_timeout(timeout_s)
    retired_canary = _api_data(
        client.post(
            f"/api/models/{model_ids['canary']}/retire",
            json={"triggered_by": "smoke", "reason": "make trained model the only canary"},
            timeout=operation_timeout,
        ),
        label="retire seeded canary",
    )
    if (retired_canary.get("model") or {}).get("stage") != "retired":
        raise DashboardBusinessSmokeFailure(f"seeded canary did not retire: {retired_canary}")

    transitions = [
        (
            "shadow",
            {
                "target_stage": "shadow",
                "triggered_by": "smoke",
                "reason": "business smoke shadow promotion",
            },
        ),
        (
            "canary",
            {
                "target_stage": "canary",
                "triggered_by": "smoke",
                "reason": "business smoke canary promotion",
                "canary_camera_id": camera_id,
            },
        ),
        (
            "production",
            {
                "target_stage": "production",
                "triggered_by": "smoke",
                "reason": "business smoke production promotion",
            },
        ),
    ]

    transition_results: list[dict[str, Any]] = []
    for expected_stage, payload in transitions:
        data = _api_data(
            client.post(
                f"/api/models/{candidate_id}/promote",
                json=payload,
                timeout=operation_timeout,
            ),
            label=f"model promote {expected_stage}",
        )
        model = data.get("model") or {}
        if model.get("stage") != expected_stage:
            raise DashboardBusinessSmokeFailure(
                f"model promote {expected_stage} returned wrong stage: {model}"
            )
        runtime_state = data.get("runtime_state")
        if data.get("runtime_synced") is not True:
            raise DashboardBusinessSmokeFailure(
                f"model promote {expected_stage} did not sync runtime: {data}"
            )
        if isinstance(runtime_state, dict) and runtime_state.get("errors"):
            raise DashboardBusinessSmokeFailure(
                f"model promote {expected_stage} had runtime errors: {runtime_state}"
            )
        if expected_stage in {"canary", "production"} and isinstance(runtime_state, dict):
            if runtime_state.get("primary_model_version") != candidate_id:
                raise DashboardBusinessSmokeFailure(
                    f"model promote {expected_stage} did not route trained model: {runtime_state}"
                )
            if runtime_state.get("primary_model_stage") != expected_stage:
                raise DashboardBusinessSmokeFailure(
                    f"model promote {expected_stage} runtime stage mismatch: {runtime_state}"
                )
        transition_results.append({
            "stage": expected_stage,
            "runtime_synced": data.get("runtime_synced"),
            "runtime_state": data.get("runtime_state"),
        })

    rollback_data = _api_data(
        client.post(f"/api/models/{candidate_id}/rollback", timeout=operation_timeout),
        label="model rollback",
    )
    if rollback_data.get("activated") != model_ids["production"]:
        raise DashboardBusinessSmokeFailure(
            f"model rollback activated wrong version: {rollback_data}"
        )
    if rollback_data.get("runtime_synced") is not True:
        raise DashboardBusinessSmokeFailure(f"model rollback did not sync runtime: {rollback_data}")

    models_data = _api_data(
        client.get("/api/models/json", params={"camera_id": camera_id}, timeout=10),
        label="models json after rollback",
    )
    rows = {
        item.get("model_version_id"): item
        for item in models_data.get("models") or []
    }
    production = rows.get(model_ids["production"])
    api_candidate = rows.get(candidate_id)
    if not production or production.get("stage") != "production" or production.get("is_active") is not True:
        raise DashboardBusinessSmokeFailure(
            f"rollback did not restore previous production model: {production}"
        )
    if not api_candidate or api_candidate.get("stage") != "production" or api_candidate.get("is_active") is True:
        raise DashboardBusinessSmokeFailure(
            f"rolled-back candidate state is unexpected: {api_candidate}"
        )

    return {
        "promoted_version": candidate_id,
        "retired_seeded_canary": model_ids["canary"],
        "transitions": transition_results,
        "rollback": {
            "requested_from": candidate_id,
            "activated": rollback_data.get("activated"),
            "runtime_synced": rollback_data.get("runtime_synced"),
            "runtime_state": rollback_data.get("runtime_state"),
        },
        "active_after_rollback": model_ids["production"],
    }


def _business_browser_pages(
    *,
    alert_id: str,
    camera_id: str,
    alert_semantics: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    pages = {
        f"/cameras/{camera_id}": [camera_id, "实时画面", "输入质量", "已采集帧", "告警触发"],
        f"/alerts?id={alert_id}": ["告警中心", alert_id[-8:], "告警信息", "录像"],
        f"/replay/{alert_id}": ["录像回放", alert_id, "FRAME", "热力"],
        "/reports": ["报表统计", "告警总数", "Replay录像覆盖率", "完整证据率"],
    }
    semantics = alert_semantics or {}
    if (
        semantics.get("detection_type") == "projectile"
        or semantics.get("category") == "projectile"
        or semantics.get("projectile_evidence")
    ):
        pages[f"/alerts?id={alert_id}"].extend(["抛射物", "物理数据", "px/s", "projectile"])
    return pages


def _models_system_browser_pages(
    *,
    camera_id: str,
    model_ids: dict[str, str],
    trained_version_id: str,
) -> dict[str, list[str]]:
    return {
        "/models/registry": [
            "模型总数",
            "模型版本",
            "候选",
            "影子",
            "金丝雀",
            "生产",
            "已激活",
            model_ids["production"],
            trained_version_id,
        ],
        "/system/overview": [
            "模型运行状态",
            camera_id,
            "Backend",
            "输入质量",
            "稳定",
            "降级监控",
        ],
        "/system/config": [
            "系统配置",
            "检测参数",
            "保存当前配置",
            "告警音频配置",
        ],
    }


def _objective_checklist(
    *,
    camera_id: str,
    camera_row: dict[str, Any],
    camera_media: dict[str, Any],
    mode_result: dict[str, Any],
    alert: dict[str, Any],
    alert_detail: dict[str, Any],
    api_result: dict[str, Any],
    models_system_result: dict[str, Any],
    browser_result: dict[str, Any],
) -> list[dict[str, Any]]:
    camera_stats = camera_row.get("stats") or {}
    snapshot = camera_media.get("snapshot") or {}
    streaming = camera_media.get("streaming") or {}
    replay = api_result.get("replay") or {}
    reports = api_result.get("reports") or {}
    training = models_system_result.get("training_export") or {}
    release = models_system_result.get("release_api") or {}
    config_update = models_system_result.get("config_update") or {}
    fallback_models = models_system_result.get("fallback_models") or []
    degradation = models_system_result.get("degradation") or {}
    anomaly_degradation = degradation.get("anomaly") or {}
    global_degradation = degradation.get("global") or {}
    browser_routes = [
        item.get("route")
        for item in (browser_result.get("routes_checked") or [])
        if item.get("route")
    ]
    browser_checked = browser_result.get("status") == "checked"
    camera_route = f"/cameras/{camera_id}"
    alerts_route = f"/alerts?id={alert.get('alert_id')}"
    replay_route = f"/replay/{alert.get('alert_id')}"
    release_transitions = release.get("transitions") or []
    fallback_or_degraded = bool(
        fallback_models
        or anomaly_degradation.get("degraded")
        or global_degradation.get("active_count")
    )

    return [
        {
            "requirement": "1. Cameras stable video stream",
            "passed": bool(
                camera_row.get("connected")
                and camera_row.get("running")
                and camera_stats.get("frames_captured", 0) > 0
                and snapshot.get("bytes", 0) > 1024
                and streaming.get("fallback") == f"/api/cameras/{camera_id}/stream"
                and browser_checked
                and camera_route in browser_routes
            ),
            "evidence": {
                "browser_checked": browser_checked,
                "camera_id": camera_id,
                "connected": camera_row.get("connected"),
                "running": camera_row.get("running"),
                "pipeline_mode": mode_result.get("pipeline_mode"),
                "frames_captured": camera_stats.get("frames_captured"),
                "snapshot_bytes": snapshot.get("bytes"),
                "streaming": streaming,
                "browser_route": camera_route if camera_route in browser_routes else None,
            },
        },
        {
            "requirement": "2. Alerts realtime display",
            "passed": bool(
                alert.get("alert_id")
                and alert.get("_realtime_payload")
                and browser_checked
                and alerts_route in browser_routes
            ),
            "evidence": {
                "browser_checked": browser_checked,
                "alert_id": alert.get("alert_id"),
                "severity": alert.get("severity"),
                "websocket_alert_id": (alert.get("_realtime_payload") or {}).get("alert_id"),
                "browser_route": alerts_route if alerts_route in browser_routes else None,
            },
        },
        {
            "requirement": "3. Replay evidence",
            "passed": bool(
                alert_detail.get("has_recording")
                and alert_detail.get("recording_status") == "complete"
                and alert_detail.get("snapshot_path")
                and alert_detail.get("heatmap_path")
                and replay.get("frame_count", 0) >= 2
                and browser_checked
                and replay_route in browser_routes
            ),
            "evidence": {
                "browser_checked": browser_checked,
                "recording_status": alert_detail.get("recording_status"),
                "snapshot_path": alert_detail.get("snapshot_path"),
                "heatmap_path": alert_detail.get("heatmap_path"),
                "evidence_zip_bytes": api_result.get("evidence_zip_bytes"),
                "replay_frame_count": replay.get("frame_count"),
                "signal_points": replay.get("signal_points"),
                "browser_route": replay_route if replay_route in browser_routes else None,
            },
        },
        {
            "requirement": "4. Models train/export/release/rollback",
            "passed": bool(
                training.get("status") == "complete"
                and (training.get("reexport") or {}).get("status") == "ok"
                and release_transitions
                and all(item.get("runtime_synced") is True for item in release_transitions)
                and (release.get("rollback") or {}).get("runtime_synced") is True
                and browser_checked
                and "/models/registry" in browser_routes
            ),
            "evidence": {
                "browser_checked": browser_checked,
                "training_mode": training.get("training_mode"),
                "job_id": training.get("job_id"),
                "trained_model_version": training.get("model_version_id"),
                "reexport_status": (training.get("reexport") or {}).get("status"),
                "release_stages": [item.get("stage") for item in release_transitions],
                "rollback_activated": (release.get("rollback") or {}).get("activated"),
                "browser_route": "/models/registry" if "/models/registry" in browser_routes else None,
            },
        },
        {
            "requirement": "5. System config and fallback/degradation",
            "passed": bool(
                config_update.get("pipelines_updated", 0) > 0
                and fallback_or_degraded
                and browser_checked
                and "/system/overview" in browser_routes
                and "/system/config" in browser_routes
            ),
            "evidence": {
                "browser_checked": browser_checked,
                "pipelines_updated": config_update.get("pipelines_updated"),
                "fallback_models": [
                    {
                        "name": item.get("name"),
                        "camera_id": item.get("camera_id"),
                        "loaded": item.get("loaded"),
                        "backend": item.get("backend"),
                    }
                    for item in fallback_models
                ],
                "anomaly_degraded": anomaly_degradation.get("degraded"),
                "global_active_degradations": global_degradation.get("active_count"),
                "browser_routes": [
                    route for route in ("/system/overview", "/system/config") if route in browser_routes
                ],
            },
        },
        {
            "requirement": "6. Reports stats from alerts and evidence",
            "passed": bool(
                reports.get("total_alerts", 0) >= 1
                and reports.get("evidence_complete_rate") == 100.0
                and (reports.get("camera_distribution") or {}).get("count", 0) >= 1
                and browser_checked
                and "/reports" in browser_routes
            ),
            "evidence": {
                "browser_checked": browser_checked,
                "total_alerts": reports.get("total_alerts"),
                "evidence_complete_rate": reports.get("evidence_complete_rate"),
                "recording_rate": reports.get("recording_rate"),
                "camera_distribution": reports.get("camera_distribution"),
                "browser_route": "/reports" if "/reports" in browser_routes else None,
            },
        },
    ]


def _check_business_browser_dom(
    args: argparse.Namespace,
    *,
    base_url: str,
    work_dir: Path,
    alert_id: str,
    camera_id: str,
    alert_semantics: dict[str, Any] | None = None,
    additional_pages: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if args.browser == "off":
        return {"status": "off", "routes_checked": []}

    browser_path = _find_headless_browser(args.browser_path)
    if not browser_path:
        if args.browser == "required":
            raise DashboardBusinessSmokeFailure(
                "headless browser required but Chrome/Edge/Chromium was not found"
            )
        return {"status": "skipped", "reason": "Chrome/Edge/Chromium not found", "routes_checked": []}

    checked: list[dict[str, Any]] = []
    pages = _business_browser_pages(
        alert_id=alert_id,
        camera_id=camera_id,
        alert_semantics=alert_semantics,
    )
    if additional_pages:
        pages.update(additional_pages)

    for route, markers in pages.items():
        safe_route = route.strip("/").replace("/", "_").replace("?", "_").replace("=", "_")
        profile_dir = work_dir / "browser-profiles" / safe_route
        profile_dir.mkdir(parents=True, exist_ok=True)
        dom = ""
        missing: list[str] = list(markers)
        last_error: DashboardSmokeFailure | None = None
        for attempt in range(3):
            try:
                dom = _dump_dom_with_browser(
                    browser_path=browser_path,
                    url=f"{base_url}{route}",
                    user_data_dir=profile_dir,
                    timeout_s=args.browser_timeout,
                    virtual_time_ms=args.browser_virtual_time_ms * (attempt + 1),
                )
            except DashboardSmokeFailure as exc:
                last_error = exc
                if attempt == 2:
                    raise DashboardBusinessSmokeFailure(str(exc)) from exc
                time.sleep(0.5)
                continue

            missing = [marker for marker in markers if marker not in dom]
            if not missing:
                break
            time.sleep(0.5)

        if missing:
            snippet = " ".join(dom.split())[:500]
            suffix = f"; last_error={last_error}" if last_error else ""
            raise DashboardBusinessSmokeFailure(
                f"browser DOM missing markers for {route}: {missing}; snippet={snippet!r}{suffix}"
            )
        if "/login" in dom and "登录" in dom:
            raise DashboardBusinessSmokeFailure(f"browser DOM reached login page for {route}")
        checked.append({
            "route": route,
            "markers": markers,
            "dom_bytes": len(dom.encode("utf-8", errors="replace")),
        })

    return {
        "status": "checked",
        "browser": browser_path,
        "routes_checked": checked,
    }


def _clean_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("ARGUS__")}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _default_camera_id(config_path: Path) -> str:
    config = load_config(config_path)
    if not config.cameras:
        raise DashboardBusinessSmokeFailure("config has no cameras")
    return config.cameras[0].camera_id


def run_camera_preflight(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.smoke_core_loop import run_preflight as _run_core_preflight

    config_path = args.config.resolve()
    fixture_tmp = None
    fixture: _RtspFixture | None = None
    fixture_info: dict[str, Any] | None = None
    camera_source = args.camera_source
    camera_protocol = args.camera_protocol
    try:
        if args.rtsp_fixture:
            fixture_work_dir = args.work_dir.resolve() if args.work_dir else None
            if fixture_work_dir is None:
                fixture_tmp = tempfile.TemporaryDirectory(prefix="argus-rtsp-fixture-")
                fixture_work_dir = Path(fixture_tmp.name)
            fixture_work_dir.mkdir(parents=True, exist_ok=True)
            fixture = _RtspFixture(
                work_dir=fixture_work_dir,
                resolution=_parse_resolution(args.camera_resolution),
                seconds=args.rtsp_fixture_seconds,
                motion=args.dev_video_motion,
            )
            fixture_info = fixture.start()
            camera_source = fixture_info["source_url"]
            camera_protocol = "rtsp"

        core_args = argparse.Namespace(
            config=config_path,
            work_dir=args.work_dir,
            camera_source=camera_source,
            camera_protocol=camera_protocol,
            camera_id=args.camera_id or _default_camera_id(config_path),
            camera_name=args.camera_name,
            camera_resolution=args.camera_resolution,
            usb_device_name=args.usb_device_name,
            usb_device_id=args.usb_device_id,
            disable_go2rtc=args.disable_go2rtc,
            require_go2rtc=args.require_go2rtc,
            preflight_timeout=args.preflight_timeout,
            preflight_measure_seconds=args.preflight_measure_seconds,
            video_seconds=20,
            dev_video_motion=args.dev_video_motion,
            use_yolo=args.use_yolo,
        )
        result = _run_core_preflight(core_args)
        if fixture_info is not None:
            result["rtsp_fixture"] = fixture_info
        if fixture_info is not None:
            result["business_smoke_command"] = (
                "scripts/smoke_dashboard_business_flow.py "
                "--rtsp-fixture --require-go2rtc --browser required"
            )
        else:
            result["business_smoke_command"] = (
                "scripts/smoke_dashboard_business_flow.py "
                "--camera-source <source> --camera-protocol <file|usb|rtsp> --browser required"
            )
        return result
    finally:
        if fixture is not None:
            fixture.close()
        if fixture_tmp is not None:
            fixture_tmp.cleanup()


def run_dashboard_business_smoke(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="argus-dashboard-business-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = work_dir / "argus.stdout.log"
    stderr_path = work_dir / "argus.stderr.log"
    port = args.port or _free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc: subprocess.Popen | None = None
    fixture: _RtspFixture | None = None
    fixture_info: dict[str, Any] | None = None
    go2rtc_api_port: int | None = None
    runtime_config: Path | None = None
    camera_id: str | None = None
    camera_row: dict[str, Any] | None = None
    camera_media_result: dict[str, Any] | None = None
    detector_detail: dict[str, Any] | None = None
    mode_result: dict[str, Any] | None = None
    try:
        camera_resolution = _parse_resolution(args.camera_resolution)
        camera_source = args.camera_source
        camera_protocol = args.camera_protocol
        if args.rtsp_fixture:
            fixture = _RtspFixture(
                work_dir=work_dir,
                resolution=camera_resolution,
                seconds=args.rtsp_fixture_seconds,
                motion=args.dev_video_motion,
            )
            fixture_info = fixture.start()
            camera_source = fixture_info["source_url"]
            camera_protocol = "rtsp"

        runtime_config, camera_id = _prepare_runtime_config(
            config_path=args.config.resolve(),
            work_dir=work_dir,
            port=port,
            use_yolo=args.use_yolo,
            camera_source=camera_source,
            camera_protocol=camera_protocol,
            camera_id=args.camera_id,
            camera_name=args.camera_name,
            camera_resolution=camera_resolution,
            usb_device_name=args.usb_device_name,
            usb_device_id=args.usb_device_id,
            go2rtc_enabled=None if not args.disable_go2rtc else False,
        )
        runtime_loaded = load_config(runtime_config)
        if runtime_loaded.dashboard.go2rtc_enabled:
            go2rtc_api_port = runtime_loaded.dashboard.go2rtc_api_port
        video_path = work_dir / "dev_camera.avi"
        cmd = [
            sys.executable,
            "-m",
            "argus",
            "--config",
            str(runtime_config),
        ]
        if args.camera_source is None and fixture_info is None:
            cmd.extend([
                "--dev-video",
                "--dev-video-path",
                str(video_path),
                "--dev-video-motion",
                args.dev_video_motion,
            ])
        if args.training_mode == "dev-fast":
            cmd.append("--dev-fast-training")

        training_baselines: dict[str, Any] | None = None
        if args.training_mode == "normal":
            training_baselines = _seed_training_baselines(
                work_dir=work_dir,
                camera_id=camera_id,
                count=args.training_baseline_count,
                image_size=args.training_image_size,
            )

        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=_clean_env(),
                stdout=stdout,
                stderr=stderr,
                text=True,
            )

            with httpx.Client(base_url=base_url, follow_redirects=False) as client:
                camera_row = _wait_for_camera(
                    client,
                    camera_id=camera_id,
                    timeout_s=min(args.timeout, 45.0),
                    min_frames=args.min_frames,
                    process=proc,
                )
                camera_media_result = _verify_camera_media_apis(
                    client,
                    camera_id=camera_id,
                    require_go2rtc=args.require_go2rtc,
                    work_dir=work_dir,
                )
                if (
                    not args.expect_no_alert
                    or _should_wait_for_detector_before_no_alert(args)
                ):
                    detector_detail = _wait_for_detector_ready(
                        client,
                        camera_id=camera_id,
                        timeout_s=min(args.timeout, 45.0),
                        process=proc,
                    )
                with _AlertWebSocketListener(base_url=base_url) as realtime_listener:
                    preexisting_alert_ids = {
                        str(alert.get("alert_id"))
                        for alert in _recent_alerts(client, camera_id=camera_id, limit=50)
                        if alert.get("alert_id")
                    }
                    mode_result = _set_camera_mode(
                        client,
                        camera_id=camera_id,
                        mode=args.observe_mode,
                    )
                    if args.activation_delay > 0:
                        print(_physical_action_window_message(args), flush=True)
                        time.sleep(args.activation_delay)
                    if args.expect_no_alert:
                        no_alert_result = _verify_no_alert_window(
                            client,
                            camera_id=camera_id,
                            observe_seconds=args.no_alert_observe_seconds,
                            process=proc,
                            known_alert_ids=preexisting_alert_ids,
                        )
                        camera_row = _wait_for_camera(
                            client,
                            camera_id=camera_id,
                            timeout_s=min(args.timeout, 10.0),
                            min_frames=args.min_frames,
                            process=proc,
                        )
                        detector_detail = _api_data(
                            client.get(f"/api/cameras/{camera_id}/detail/json", timeout=5),
                            label="camera detail",
                        )
                        detector = detector_detail.get("detector") or {}
                        detector_no_alert_check = _verify_no_alert_detector_ready(
                            detector=detector,
                            observe_mode=args.observe_mode,
                            allow_detection_limited=args.allow_detection_limited_no_alert,
                        )
                        return {
                            "ok": True,
                            "mode": "no_alert",
                            "base_url": base_url,
                            "work_dir": str(work_dir),
                            "runtime_config": str(runtime_config),
                            "camera": {
                                "camera_id": camera_id,
                                "connected": camera_row.get("connected"),
                                "running": camera_row.get("running"),
                                "pipeline_mode": mode_result.get("pipeline_mode"),
                                "frames_captured": (camera_row.get("stats") or {}).get("frames_captured"),
                                "detector": detector,
                            },
                            "camera_media": camera_media_result,
                            "no_alert": {
                                **no_alert_result,
                                "detector_ready": detector_no_alert_check,
                            },
                            "rtsp_fixture": fixture_info,
                            "browser": {
                                "status": "not_applicable",
                                "reason": "--expect-no-alert does not open alert or replay detail pages",
                                "routes_checked": [],
                            },
                            "expected_degradations": _expected_degradations(
                                use_yolo=args.use_yolo,
                                protocol=camera_protocol or _infer_camera_protocol(camera_source or ""),
                            ),
                        }
                    alert, alert_detail = _wait_for_completed_alert(
                        client,
                        camera_id=camera_id,
                        alert_timeout_s=args.timeout,
                        recording_timeout_s=args.recording_timeout,
                        process=proc,
                        realtime_listener=realtime_listener,
                        known_alert_ids=preexisting_alert_ids,
                    )
                alert_id = alert["alert_id"]
                alert_semantics = _verify_alert_semantic_expectations(args, alert)
                dev_video_semantics = _verify_dev_video_alert_semantics(args, alert)
                api_result = _verify_business_apis(
                    client,
                    alert_id=alert_id,
                    camera_id=camera_id,
                )
                model_ids = _seed_model_registry(
                    work_dir=work_dir,
                    database_url=f"sqlite:///{work_dir / 'argus.db'}",
                    camera_id=camera_id,
                )
                models_system_result = _verify_models_system_apis(
                    client,
                    camera_id=camera_id,
                    model_ids=model_ids,
                )
                training_export_result = _exercise_training_export_apis(
                    client,
                    camera_id=camera_id,
                    training_mode=args.training_mode,
                    image_size=args.training_image_size,
                    timeout_s=args.training_timeout,
                    process=proc,
                )
                release_api_result = _exercise_model_release_apis(
                    client,
                    camera_id=camera_id,
                    model_ids=model_ids,
                    version_id=training_export_result["model_version_id"],
                    timeout_s=args.training_timeout,
                )
                models_system_result["training_export"] = training_export_result
                models_system_result["release_api"] = release_api_result
                browser_result = _check_business_browser_dom(
                    args,
                    base_url=base_url,
                    work_dir=work_dir,
                    alert_id=alert_id,
                    camera_id=camera_id,
                    alert_semantics=alert_semantics,
                    additional_pages=_models_system_browser_pages(
                        camera_id=camera_id,
                        model_ids=model_ids,
                        trained_version_id=training_export_result["model_version_id"],
                    ),
                )
                objective_checklist = _objective_checklist(
                    camera_id=camera_id,
                    camera_row=camera_row,
                    camera_media=camera_media_result,
                    mode_result=mode_result,
                    alert=alert,
                    alert_detail=alert_detail,
                    api_result=api_result,
                    models_system_result=models_system_result,
                    browser_result=browser_result,
                )

        return {
            "ok": True,
            "base_url": base_url,
            "work_dir": str(work_dir),
            "runtime_config": str(runtime_config),
            "camera": {
                "camera_id": camera_id,
                "connected": camera_row.get("connected"),
                "running": camera_row.get("running"),
                "pipeline_mode": mode_result.get("pipeline_mode"),
                "frames_captured": (camera_row.get("stats") or {}).get("frames_captured"),
                "detector": (detector_detail.get("detector") or {}),
            },
            "camera_media": camera_media_result,
            "alert": {
                "alert_id": alert_id,
                "severity": alert.get("severity"),
                "recording_status": alert_detail.get("recording_status"),
                "has_recording": alert_detail.get("has_recording"),
                "snapshot_path": alert_detail.get("snapshot_path"),
                "heatmap_path": alert_detail.get("heatmap_path"),
                "realtime": alert.get("_realtime_payload"),
            },
            "api": api_result,
            "models_system": models_system_result,
            "training_baselines": training_baselines,
            "rtsp_fixture": fixture_info,
            "alert_semantics": alert_semantics,
            "dev_video_semantics": dev_video_semantics,
            "browser": browser_result,
            "objective_checklist": objective_checklist,
            "expected_degradations": _expected_degradations(
                use_yolo=args.use_yolo,
                protocol=camera_protocol or _infer_camera_protocol(camera_source or ""),
            ),
        }
    except Exception as exc:
        error = str(exc) if isinstance(
            exc,
            (DashboardBusinessSmokeFailure, DashboardSmokeFailure),
        ) else f"{type(exc).__name__}: {exc}"
        partial: dict[str, Any] = {}
        if runtime_config is not None:
            partial["runtime_config"] = str(runtime_config)
        if camera_id is not None:
            partial["camera_id"] = camera_id
        if camera_row is not None:
            partial["camera"] = {
                "camera_id": camera_id,
                "connected": camera_row.get("connected"),
                "running": camera_row.get("running"),
                "frames_captured": (camera_row.get("stats") or {}).get("frames_captured"),
            }
        if mode_result is not None:
            partial["mode"] = mode_result
        if camera_media_result is not None:
            partial["camera_media"] = camera_media_result
        if detector_detail is not None:
            partial["detector"] = detector_detail.get("detector") or {}

        result = {
            "ok": False,
            "error": error,
            "base_url": base_url,
            "work_dir": str(work_dir),
            "stdout_tail": _tail(stdout_path),
            "stderr_tail": _tail(stderr_path),
        }
        if partial:
            result["partial"] = partial
        return result
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        _cleanup_go2rtc_api_port(go2rtc_api_port)
        if fixture is not None:
            fixture.close()
        if not args.keep_work_dir and args.work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke real Dashboard business data in a browser")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base Argus config to derive from (default: configs/default.yaml)",
    )
    parser.add_argument("--work-dir", type=Path, default=None, help="Optional smoke work directory")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep temporary work directory")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Probe the configured camera source and go2rtc path, then exit before running the full business flow.",
    )
    parser.add_argument("--port", type=int, default=0, help="Dashboard port; 0 picks a free port")
    parser.add_argument("--timeout", type=float, default=90.0, help="Seconds to wait for alert generation")
    parser.add_argument(
        "--observe-mode",
        choices=["active", "maintenance", "collection", "training"],
        default="active",
        help=(
            "Pipeline mode to set before alert/no-alert observation. "
            "Use collection or training with --expect-no-alert to verify detection is stopped."
        ),
    )
    parser.add_argument(
        "--expect-no-alert",
        action="store_true",
        help="After setting --observe-mode, observe the camera and fail if any alert is generated.",
    )
    parser.add_argument(
        "--no-alert-observe-seconds",
        type=float,
        default=30.0,
        help="Seconds to observe with --expect-no-alert (default: 30).",
    )
    parser.add_argument(
        "--allow-detection-limited-no-alert",
        action="store_true",
        help=(
            "Allow active/maintenance --expect-no-alert to pass even when low light "
            "or another input-quality gate has limited detection."
        ),
    )
    parser.add_argument(
        "--recording-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for replay recording completion",
    )
    parser.add_argument(
        "--training-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for the training job to complete",
    )
    parser.add_argument(
        "--training-mode",
        choices=["dev-fast", "normal"],
        default="dev-fast",
        help=(
            "Training backend for the model step. dev-fast uses deterministic "
            "local OpenVINO artifacts; normal runs the real ModelTrainer."
        ),
    )
    parser.add_argument(
        "--training-baseline-count",
        type=int,
        default=36,
        help="Number of synthetic baseline images to seed for --training-mode normal",
    )
    parser.add_argument(
        "--training-image-size",
        type=int,
        default=64,
        help="Image size used for the training job and synthetic normal baselines",
    )
    parser.add_argument("--min-frames", type=int, default=8, help="Minimum camera frames before activation")
    parser.add_argument("--use-yolo", action="store_true", help="Use configured YOLO model instead of offline fallback")
    parser.add_argument(
        "--camera-source",
        default=None,
        help=(
            "Use a real camera/video source instead of generated dev video. "
            "Examples: 0 for USB, rtsp://host/stream for RTSP, or a video file path."
        ),
    )
    parser.add_argument(
        "--camera-protocol",
        choices=["file", "usb", "rtsp"],
        default=None,
        help="Protocol for --camera-source. Inferred from the source when omitted.",
    )
    parser.add_argument(
        "--rtsp-fixture",
        action="store_true",
        help=(
            "Start a local go2rtc fixture that publishes generated dev video as RTSP. "
            "Use this to exercise RTSP input and Argus go2rtc playback without hardware."
        ),
    )
    parser.add_argument(
        "--rtsp-fixture-seconds",
        type=int,
        default=180,
        help="Length of generated RTSP fixture video in seconds (default: 180).",
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
    parser.add_argument("--camera-id", default=None, help="Override the first config camera ID")
    parser.add_argument("--camera-name", default=None, help="Override the first config camera display name")
    parser.add_argument(
        "--camera-resolution",
        default="640,480",
        help="Expected capture resolution as WIDTH,HEIGHT or WIDTHxHEIGHT",
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
        help="Fail unless /api/streaming/{camera_id} exposes go2rtc playback metadata.",
    )
    parser.add_argument(
        "--activation-delay",
        type=float,
        default=0.0,
        help=(
            "Seconds to wait after switching the camera to active mode. "
            "Useful for hardware runs where an operator needs time to introduce a test object."
        ),
    )
    parser.add_argument(
        "--expect-alert-category",
        action="append",
        default=[],
        help=(
            "Require the alert category to match this value. Repeat to allow "
            "multiple values, e.g. scene_change or projectile."
        ),
    )
    parser.add_argument(
        "--expect-detection-type",
        action="append",
        default=[],
        help=(
            "Require the realtime detection_type to match this value. Repeat "
            "to allow multiple values, e.g. anomaly or projectile."
        ),
    )
    parser.add_argument(
        "--expect-detected-object-class",
        action="append",
        default=[],
        help=(
            "Require realtime detected_objects to include this class_name/class. "
            "Repeat to require multiple classes, e.g. fast_projectile."
        ),
    )
    parser.add_argument(
        "--forbid-alert-category",
        action="append",
        default=[],
        help="Fail if the alert category matches this value. May be repeated.",
    )
    parser.add_argument(
        "--forbid-detection-type",
        action="append",
        default=[],
        help="Fail if the realtime detection_type matches this value. May be repeated.",
    )
    parser.add_argument(
        "--forbid-detected-object-class",
        action="append",
        default=[],
        help="Fail if realtime detected_objects include this class_name/class. May be repeated.",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait per capture backend during --preflight.",
    )
    parser.add_argument(
        "--preflight-measure-seconds",
        type=float,
        default=2.0,
        help=(
            "Seconds to sample decoded frames for preflight FPS measurement. "
            "USB fast-motion checks use at least 15 seconds."
        ),
    )
    parser.add_argument(
        "--browser",
        choices=["auto", "required", "off"],
        default="auto",
        help="Headless browser DOM smoke mode (default: auto)",
    )
    parser.add_argument("--browser-path", default=None, help="Explicit Chrome/Edge/Chromium executable")
    parser.add_argument("--browser-timeout", type=float, default=30.0, help="Seconds per browser DOM dump")
    parser.add_argument(
        "--browser-virtual-time-ms",
        type=int,
        default=9000,
        help="Chrome virtual time budget per route, in milliseconds",
    )
    args = parser.parse_args(argv)
    if args.disable_go2rtc and args.require_go2rtc:
        parser.error("--disable-go2rtc and --require-go2rtc cannot be used together")
    if args.rtsp_fixture and args.camera_source is not None:
        parser.error("--rtsp-fixture cannot be combined with --camera-source")
    if args.rtsp_fixture and args.camera_protocol not in {None, "rtsp"}:
        parser.error("--rtsp-fixture uses RTSP; omit --camera-protocol or set rtsp")
    if (
        args.camera_protocol is not None
        and args.camera_source is None
        and not args.rtsp_fixture
    ):
        parser.error("--camera-protocol requires --camera-source")
    if args.rtsp_fixture_seconds <= 0:
        parser.error("--rtsp-fixture-seconds must be positive")
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.no_alert_observe_seconds <= 0:
        parser.error("--no-alert-observe-seconds must be positive")
    if args.expect_no_alert and (
        args.expect_alert_category
        or args.expect_detection_type
        or args.expect_detected_object_class
    ):
        parser.error("--expect-no-alert cannot be combined with expected alert semantics")
    if (
        args.expect_no_alert
        and args.rtsp_fixture
        and args.observe_mode in {"active", "maintenance"}
        and args.dev_video_motion != "stable"
    ):
        parser.error(
            "--expect-no-alert with active/maintenance RTSP fixture requires "
            "--dev-video-motion stable; use collection/training to verify "
            "suppression on anomalous inputs"
        )
    if not args.expect_no_alert and args.observe_mode in {"collection", "training"}:
        parser.error("--observe-mode collection/training requires --expect-no-alert")
    if args.recording_timeout <= 0:
        parser.error("--recording-timeout must be positive")
    if args.training_timeout <= 0:
        parser.error("--training-timeout must be positive")
    if args.training_baseline_count < 30:
        parser.error("--training-baseline-count must be at least 30")
    if args.training_image_size < 64:
        parser.error("--training-image-size must be at least 64")
    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    if args.activation_delay < 0:
        parser.error("--activation-delay must be non-negative")
    if args.preflight_timeout <= 0:
        parser.error("--preflight-timeout must be positive")
    if args.preflight_measure_seconds <= 0:
        parser.error("--preflight-measure-seconds must be positive")
    try:
        _parse_resolution(args.camera_resolution)
    except DashboardBusinessSmokeFailure as exc:
        parser.error(str(exc))
    if args.browser_timeout <= 0:
        parser.error("--browser-timeout must be positive")
    if args.browser_virtual_time_ms <= 0:
        parser.error("--browser-virtual-time-ms must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    result = run_camera_preflight(args) if args.preflight else run_dashboard_business_smoke(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
