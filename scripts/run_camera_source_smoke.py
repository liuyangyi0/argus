"""Explicit, time-bounded Windows USB source-only smoke, with no saved pixels.

The parent owns one spawned capture process and enforces a 45-second deadline.
Only CameraManager.start_source() is used: no detection task is started.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.abc
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time


BLOCKED_SDKS = {
    "torch", "torchvision", "anomalib", "ultralytics", "onnxruntime", "onnx",
    "openvino", "timm", "lightglue", "transformers", "sam2", "segment_anything",
}
PLANNED_CHECKS = ["manager_source_only_without_model_sdks", "same_identity_twelve_frames",
    "shared_immutable_pixels_independent_metadata", "closing_a_keeps_b_and_source_running",
    "source_stop_confirmed", "manager_shutdown_confirmed"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class SourceOnlyImportBlocker(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in BLOCKED_SDKS:
            self.attempts.append(fullname)
            raise ImportError(f"source-only smoke forbids loading model SDK: {fullname}")
        return None


def frame_metadata(frame) -> dict:
    identity = frame.identity
    require(identity is not None, "captured frame has no source identity")
    # Explicitly retain unknown media values as null; capture sequence is not
    # a decoder index or PTS. Do not persist source URLs, paths or pixel data.
    fields = ("camera_id", "source_session_id", "source_epoch", "frame_index", "source_frame_index",
        "pts", "time_base_num", "time_base_den", "frame_index_provenance", "timestamp_provenance",
        "capture_wall_ts", "capture_monotonic_ts")
    return {**{key: getattr(identity, key) for key in fields},
        "actual_resolution": list(frame.resolution), "pixel_shape": list(frame.frame.shape),
        "pixel_dtype": str(frame.frame.dtype)}


def next_frame(subscription, deadline: float):
    while time.monotonic() < deadline:
        frame = subscription.get(timeout=min(0.25, max(0.0, deadline - time.monotonic())))
        if frame is not None:
            return frame
    raise RuntimeError("USB capture exceeded the 20-second acquisition deadline")


def initial_aligned_frames(subscription_a, subscription_b, token, deadline: float):
    """Discard only the legitimate prefix captured between registrations.

    A/B subscriptions cannot be registered atomically through the public API.
    Once their first common identity is found, later pairs must match exactly;
    do not keep aligning away missing frames or subscriber queue problems.
    """
    a, b = next_frame(subscription_a, deadline), next_frame(subscription_b, deadline)
    discarded = {"a": 0, "b": 0}
    while True:
        for frame in (a, b):
            require(frame.identity is not None, "alignment frame has no source identity")
            require((frame.identity.source_session_id, frame.identity.source_epoch) == token,
                "source changed generation before subscriber alignment")
        if a.identity.frame_index == b.identity.frame_index:
            return a, b, discarded
        if a.identity.frame_index < b.identity.frame_index:
            discarded["a"] += 1
            a = next_frame(subscription_a, deadline)
        else:
            discarded["b"] += 1
            b = next_frame(subscription_b, deadline)


def capture_worker(root_text: str, device_index: int) -> None:
    root = Path(root_text)
    os.chdir(root)  # Any unexpected relative runtime paths remain isolated.
    report = {"status": "failed", "entrypoint": "CameraManager.start_source", "manager_exercised": False,
        "model_training_or_inference_performed": False, "frames_or_videos_saved": False,
        "camera_device_index": device_index, "requested_fps": 5, "requested_resolution": [640, 480],
        "requested_backend": "auto", "checks": [], "matched_frames": [], "peer_after_close": []}
    blocker = SourceOnlyImportBlocker()
    manager = source = subscription_a = subscription_b = None
    try:
        require(not any(name.split(".", 1)[0] in BLOCKED_SDKS for name in sys.modules),
            "capture child already has model SDKs loaded before source initialization")
        sys.meta_path.insert(0, blocker)
        import numpy as np

        from argus.capture.manager import CameraManager
        from argus.config.schema import AlertConfig, CameraConfig

        camera_id = "camera-source-smoke"
        config = CameraConfig(camera_id=camera_id, name="Source-only bounded USB smoke",
            source=str(device_index), protocol="usb", fps_target=5, resolution=(640, 480),
            reconnect_delay=1.0, max_reconnect_attempts=0,
            usb={"preferred_backend": "auto", "pixel_format": "auto"})
        alerts = []
        manager = CameraManager([config], AlertConfig(), on_alert=alerts.append, baselines_dir=root / "unused-baselines")
        capture_started = time.monotonic()
        deadline = capture_started + 20.0
        require(manager.start_source(camera_id), "CameraManager could not start the explicitly selected USB source")
        report["manager_exercised"] = True
        source = manager.get_camera_source(camera_id)
        require(source is not None and source.state.running, "source-only start did not publish a running source")
        require(not blocker.attempts, "source-only startup attempted a forbidden model SDK import")
        require(not manager._pipelines and not manager._runners,
            "source-only start unexpectedly created a legacy detection pipeline or runner")
        report["checks"].append({"name": "manager_source_only_without_model_sdks", "status": "passed"})
        token = source.source_token
        require(token is not None, "source token is unavailable after start")
        subscription_a = source.subscribe("smoke-a", mode="ordered_window", capacity=64)
        subscription_b = source.subscribe("smoke-b", mode="ordered_window", capacity=64)
        first_a, first_b, alignment = initial_aligned_frames(subscription_a, subscription_b, token, deadline)
        report["initial_alignment_discarded_prefix"] = alignment
        previous_sequence = -1
        for number in range(12):
            a = first_a if number == 0 else next_frame(subscription_a, deadline)
            b = first_b if number == 0 else next_frame(subscription_b, deadline)
            meta_a, meta_b = frame_metadata(a), frame_metadata(b)
            require(meta_a == meta_b, "two independent subscribers received different original frame identities")
            require(a.frame_number == a.identity.frame_index and a.timestamp == a.identity.capture_monotonic_ts,
                "capture metadata does not align with frame identity")
            require(a.identity.frame_index > previous_sequence, "source capture sequence is not strictly increasing")
            previous_sequence = a.identity.frame_index
            require((a.identity.source_session_id, a.identity.source_epoch) == token,
                "source reconnected or changed generation during same-frame validation")
            require(a is not b and a.identity is not b.identity, "task-owned frame metadata was unexpectedly shared")
            require(np.shares_memory(a.frame, b.frame), "subscribers did not share immutable source pixels")
            require(not a.frame.flags.writeable and not b.frame.flags.writeable, "source pixels are mutable")
            try:
                a.frame.setflags(write=True)
            except ValueError:
                pass
            else:
                raise RuntimeError("subscriber could re-enable source pixel writes")
            report["matched_frames"].append(meta_a)
        require(subscription_a.status.dropped_frames == 0 and subscription_b.status.dropped_frames == 0,
            "bounded subscriber queues overflowed during the twelve-frame identity comparison")
        report["checks"].extend([
            {"name": "same_identity_twelve_frames", "status": "passed"},
            {"name": "shared_immutable_pixels_independent_metadata", "status": "passed"},
        ])
        report["subscriber_a_before_close"] = asdict(subscription_a.status)
        report["subscriber_b_before_close"] = asdict(subscription_b.status)
        source.hub.unsubscribe("smoke-a", subscription=subscription_a)
        require(subscription_a.status.closed, "subscription A did not close")
        for _ in range(4):
            frame = next_frame(subscription_b, deadline)
            metadata = frame_metadata(frame)
            require(frame.identity.frame_index > previous_sequence, "peer did not continue to a newer source frame")
            previous_sequence = frame.identity.frame_index
            require(source.source_token == token and source.state.running
                and manager.get_camera_source(camera_id) is source, "closing A stopped or replaced the shared source")
            report["peer_after_close"].append(metadata)
        require(not subscription_b.status.closed and subscription_b.status.dropped_frames == 0,
            "closing A damaged or overflowed B's subscription")
        report["checks"].append({"name": "closing_a_keeps_b_and_source_running", "status": "passed"})
        report["source_before_stop"] = source.snapshot()
        report["capture_negotiation"] = source.runtime_status()
        report["acquisition_duration_seconds"] = time.monotonic() - capture_started
        require(time.monotonic() <= deadline, "source acquisition ran beyond its 20-second budget")
        manager.stop_source(camera_id)
        require(manager.get_camera_source(camera_id) is None, "stopped source remains exposed as current")
        report["source_after_stop"] = source.snapshot()
        require(not source.state.running and not source.state.cleanup_pending and source.source_token is None,
            "source stop did not confirm thread/device cleanup")
        require(subscription_b.get(timeout=0.05) is None, "stopped source left an old queued frame visible")
        # Raw subscribers own their registrations. Source stop invalidates
        # queues/tokens; task shutdown normally detaches their subscriptions.
        source.hub.unsubscribe("smoke-b", subscription=subscription_b)
        require(subscription_b.status.closed, "explicit cleanup could not detach raw subscriber B")
        require(not alerts, "source-only capture unexpectedly emitted a detection alert")
        report["checks"].append({"name": "source_stop_confirmed", "status": "passed"})
        require(not blocker.attempts and not any(name.split(".", 1)[0] in BLOCKED_SDKS for name in sys.modules),
            "source-only run imported a forbidden model SDK")
        manager.stop_all()
        manager = None
        report["checks"].append({"name": "manager_shutdown_confirmed", "status": "passed"})
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if manager is not None:
            try:
                manager.stop_all()
            except Exception as exc:
                report["status"] = "failed"
                report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["blocked_sdk_import_attempts"] = list(blocker.attempts)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "capture-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="explicit authorization to open the selected USB camera")
    parser.add_argument("--output-root", type=Path, required=True, help="new absolute isolated output directory; must not exist")
    parser.add_argument("--device-index", type=int, required=True, help="explicit local USB camera index; no device scanning")
    args = parser.parse_args()
    if not args.execute:
        parser.error("real camera capture requires explicit --execute")
    if os.name != "nt" or sys.platform != "win32":
        parser.error("this smoke is restricted to the authorized Windows test host")
    if args.device_index < 0:
        parser.error("--device-index must be a nonnegative explicit USB index")
    root = args.output_root
    if not root.is_absolute() or root.exists() or not root.parent.is_dir():
        parser.error("--output-root must be a new absolute directory beneath an existing parent")
    root = root.resolve()
    root.mkdir(exist_ok=False)
    report = {"schema_version": "argus.camera-source-smoke.v1", "status": "failed",
        "started_at": datetime.now(timezone.utc).isoformat(), "hardware_plumbing_only": True,
        "model_quality_evaluated": False, "production_allowed": False, "source_videos_read_or_modified": False,
        "frames_or_videos_saved": False, "camera_device_index": args.device_index,
        "parent_deadline_seconds": 45, "acquisition_deadline_seconds": 20,
        "planned_checks": PLANNED_CHECKS,
        "limitations": ["Only source lifecycle and in-memory fan-out are checked; no model or SceneGuard quality is evaluated.",
            "No frames or videos are saved. Missing source frame index, PTS and time base stay null.",
            "Requested 640x480 at 5 FPS is not a claim about actual driver negotiation; measured metadata is separate."]}
    process = multiprocessing.get_context("spawn").Process(target=capture_worker,
        args=(str(root), args.device_index), name="argus-owned-camera-source-smoke")
    try:
        process.start()
        report["owned_child_pid"] = process.pid
        process.join(timeout=45)
        if process.is_alive():
            report["error"] = {"type": "CaptureDeadlineExceeded", "message": "owned capture child exceeded 45 seconds"}
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            report["cleanup_mode"] = "owned_child_forced_termination"
            report["owned_child_termination_confirmed"] = not process.is_alive()
        else:
            report["cleanup_mode"] = "child_exited"
            report["owned_child_exit_code"] = process.exitcode
            child_path = root / "capture-result.json"
            require(child_path.is_file(), "capture child exited without a result report")
            child = json.loads(child_path.read_text(encoding="utf-8"))
            report["capture"] = child
            if process.exitcode == 0 and child.get("status") == "passed":
                report["status"] = "passed"
            else:
                report["error"] = child.get("error", {"type": "CaptureFailed", "message": "capture child did not pass"})
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            report["owned_child_termination_confirmed"] = not process.is_alive()
        if process.pid is not None and not process.is_alive():
            process.close()
        completed = {item["name"] for item in report.get("capture", {}).get("checks", []) if item["status"] == "passed"}
        report["unverified_checks"] = [name for name in PLANNED_CHECKS if name not in completed]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (root / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(root / "result.json"),
        "frames_or_videos_saved": False, "production_allowed": False}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
