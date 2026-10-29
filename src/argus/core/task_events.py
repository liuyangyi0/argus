"""Pure, bounded object-detection incident aggregation; never release authority.

The manager admits an observation only after its source/task/deployment fences.
This module preserves that exact identity and applies the frozen task's rules.
It produces validation events; neither a worker flag nor this aggregator can
authorize production or publish a legacy alert.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math

from argus.core.task_execution import TaskExecutionError, frame_provenance, validate_execution_manifest


class TaskEventValidationError(ValueError):
    pass


def _canonical(value) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def same_json_identity(left, right) -> bool:
    """JSON identity is typed: True/1/1.0 are not interchangeable."""
    try:
        return _canonical(left) == _canonical(right)
    except (TypeError, ValueError):
        return False


def validate_event_source_identity(source) -> None:
    """Validate the immutable, credential-free source/PTS envelope on reads too."""
    fields = {"camera_id", "source_session_id", "source_epoch", "source_runtime_hash", "frame_index",
              "source_frame_index", "pts", "time_base_num", "time_base_den", "frame_index_provenance",
              "timestamp_provenance", "capture_wall_ts", "capture_monotonic_ts"}
    if (not isinstance(source, dict) or set(source) != fields
            or any(not isinstance(source[key], str) or not source[key]
                   for key in ("camera_id", "source_session_id", "frame_index_provenance", "timestamp_provenance"))
            or not isinstance(source["source_runtime_hash"], str)
            or len(source["source_runtime_hash"]) != 10
            or any(char not in "0123456789abcdef" for char in source["source_runtime_hash"])):
        raise TaskEventValidationError("source/frame identity envelope is invalid")
    for key in ("source_epoch", "frame_index", "source_frame_index", "pts", "time_base_num", "time_base_den"):
        value = source[key]
        if value is not None and type(value) is not int:
            raise TaskEventValidationError("source sequence/PTS identity must use exact integer types")
    if (type(source["source_epoch"]) is not int or source["source_epoch"] < 1
            or type(source["frame_index"]) is not int or source["frame_index"] < 0
            or source["source_frame_index"] is not None and source["source_frame_index"] < 0):
        raise TaskEventValidationError("source/frame sequence identity is invalid")
    numerator, denominator = source["time_base_num"], source["time_base_den"]
    if ((numerator is None) != (denominator is None)
            or numerator is not None and (numerator <= 0 or denominator <= 0)
            or source["pts"] is not None and numerator is None):
        raise TaskEventValidationError("known source PTS requires a complete positive time base")
    if any(type(source[key]) not in (int, float) or not math.isfinite(source[key]) or source[key] < 0
           for key in ("capture_monotonic_ts", "capture_wall_ts")):
        raise TaskEventValidationError("capture clock identity must use finite non-negative non-boolean numbers")


def validate_event_sample_origin(origin, identity, image_size, binding) -> None:
    """Pure validation shared by the pre-write collector and frozen datasets."""
    try:
        if not isinstance(origin, dict) or set(origin) != {
            "schema", "event_id", "immutable_sha256", "evidence_sha256", "derivation", "payload",
        }:
            raise ValueError("invalid event origin fields")
        payload = origin["payload"]
        validate_event_source_identity(identity)
        first = payload["first_source"]
        validate_event_source_identity(first)
        event_binding, size = payload["binding"], payload["frame_size"]
        valid = (origin["schema"] == "argus.task-event-sample.v1"
                 and origin["derivation"] == "verified_jpeg_to_png.v1"
                 and isinstance(origin["event_id"], str) and bool(origin["event_id"])
                 and origin["event_id"] == payload["event_id"]
                 and payload["schema_version"] == "argus.task-event.v1"
                 and payload["event_type"] == "object_detected"
                 and origin["immutable_sha256"] == _sha({"payload": payload,
                                                        "evidence_sha256": origin["evidence_sha256"]})
                 and payload["replay"]["sha256"] == origin["evidence_sha256"]
                 and same_json_identity(payload["source"], identity)
                 and identity["camera_id"] == binding["camera_id"]
                 and all(same_json_identity(event_binding[key], value) for key, value in binding.items())
                 and isinstance(size, dict) and set(size) == {"width", "height"}
                 and all(type(size[key]) is int and 0 < size[key] <= 16384 for key in size)
                 and isinstance(image_size, list) and len(image_size) == 2
                 and all(type(value) is int for value in image_size)
                 and same_json_identity([size["width"], size["height"]], image_size)
                 and all(same_json_identity(first[key], identity[key]) for key in (
                     "camera_id", "source_session_id", "source_epoch", "source_runtime_hash"))
                 and first["frame_index"] <= identity["frame_index"]
                 and first["capture_monotonic_ts"] <= identity["capture_monotonic_ts"]
                 and isinstance(origin["evidence_sha256"], str)
                 and len(origin["evidence_sha256"]) == 64
                 and all(char in "0123456789abcdef" for char in origin["evidence_sha256"]))
        if not valid:
            raise ValueError("event sample origin mismatch")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise TaskEventValidationError("event sample provenance changed") from exc


def _iou(left, right):
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1]))
    total = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - overlap)
    return overlap / total if total > 0 else 0.0


def _in_region(box, polygon, width, height):
    if not polygon:
        return True
    x, y = (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)
    inside = False
    previous = polygon[-1]
    for point in polygon:
        ax, ay = previous
        bx, by = point
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
        previous = point
    return inside


def validate_observation(manifest: dict, result: dict, frame, *, deployment_revision: int,
                         operation_id: str) -> dict:
    """Validate the exact, same-frame envelope before deriving any event."""
    try:
        manifest = validate_execution_manifest(manifest)
    except TaskExecutionError as exc:
        raise TaskEventValidationError("frozen task execution bundle is invalid") from exc
    if manifest.get("template_id") != "object_detection.v1" or manifest.get("stage") != "shadow":
        raise TaskEventValidationError("only the fixed object-detection shadow template is supported")
    if type(deployment_revision) is not int or deployment_revision < 1 or not isinstance(operation_id, str) or not operation_id:
        raise TaskEventValidationError("a durable applied deployment identity is required")
    if (not isinstance(result, dict) or result.get("schema_version") != "argus.task-shadow-result.v1"
            or result.get("production_allowed") is not False or result.get("alerts_emitted") is not False):
        raise TaskEventValidationError("worker result is not a shadow observation")
    for key in ("task_id", "task_revision", "task_config_sha256", "manifest_sha256", "bundle_id", "template_id"):
        if not same_json_identity(result.get(key), manifest.get(key)):
            raise TaskEventValidationError("observation does not match the frozen execution bundle")
    if (not isinstance(result.get("execution_id"), str) or not result["execution_id"]
            or len(result["execution_id"]) > 64
            or type(result.get("request_id")) is not int or result["request_id"] < 1):
        raise TaskEventValidationError("missing worker execution/request identity")
    expected = {role: {"model_id": model["model_id"], "artifact_sha256": model["artifact_sha256"]}
                for role, model in manifest["models"].items()}
    if not same_json_identity(result.get("models"), expected):
        raise TaskEventValidationError("observation model identity mismatch")
    source = frame_provenance(frame)
    validate_event_source_identity(source)
    if (not same_json_identity(source, result.get("source")) or source["camera_id"] != manifest["camera_id"]
            or type(frame.frame_number) is not int
            or not same_json_identity(frame.timestamp, source["capture_monotonic_ts"])):
        raise TaskEventValidationError("observation source/frame identity mismatch")
    pixels = frame.frame
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype.name != "uint8":
        raise TaskEventValidationError("observation requires the original BGR uint8 frame")
    height, width = pixels.shape[:2]
    if (type(frame.resolution) is not tuple or len(frame.resolution) != 2
            or any(type(value) is not int for value in frame.resolution)
            or frame.resolution != (width, height)):
        raise TaskEventValidationError("frame dimensions must exactly match its integer source resolution")
    rules = manifest["config"]["rules"]
    labels = set(manifest["models"]["detector"]["labels"])
    requested = set(rules["labels"])
    data = result.get("data")
    if not isinstance(data, dict) or data.get("state") != "observed" or not isinstance(data.get("detections"), list):
        raise TaskEventValidationError("missing complete detection observation")
    if len(data["detections"]) > 100:
        raise TaskEventValidationError("detection observation exceeds its bounded contract")
    detections = []
    for item in data["detections"]:
        if not isinstance(item, dict) or set(item) != {"label", "confidence", "bbox_xyxy"}:
            raise TaskEventValidationError("invalid object-detection schema")
        label, score, box = item["label"], item["confidence"], item["bbox_xyxy"]
        if (not isinstance(label, str) or label not in labels
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise TaskEventValidationError("invalid detection label, score or source-pixel coordinates")
        if (score >= rules["confidence_threshold"] and (not requested or label in requested)
                and _in_region(box, rules["region_polygon"], width, height)):
            detections.append(deepcopy(item))
    binding = {key: deepcopy(manifest[key]) for key in (
        "task_id", "camera_id", "task_revision", "task_config_sha256", "bundle_id", "manifest_sha256", "template_id")}
    binding.update(deployment_revision=deployment_revision, deployment_operation_id=operation_id,
                   execution_id=result["execution_id"], models=expected)
    return {"binding": binding, "source": source, "request_id": result["request_id"],
            "detections": detections, "width": width, "height": height}


@dataclass
class _Track:
    track_id: str
    label: str
    box: list
    first_source: dict
    first_seen: float
    last_emitted: float | None = None


@dataclass
class _Context:
    identity: str
    deployment_revision: int
    last_frame: int = -1
    last_request: int = -1
    last_seen: float | None = None
    tracks: list[_Track] = field(default_factory=list)


class TaskEventEngine:
    """One-owner temporal state, bounded by declared task and detection counts."""

    def __init__(self, *, max_tasks: int = 128):
        if type(max_tasks) is not int or not 1 <= max_tasks <= 4096:
            raise ValueError("max_tasks must be between 1 and 4096")
        self._max_tasks = max_tasks
        self._contexts: dict[str, _Context] = {}

    def forget(self, task_id: str) -> None:
        self._contexts.pop(task_id, None)

    def forget_inactive(self, is_active) -> None:
        """Bounded lifecycle cleanup; no active task loses its cooldown."""
        for task_id in tuple(self._contexts):
            if not is_active(task_id):
                self.forget(task_id)

    def observe(self, manifest: dict, result: dict, frame, *, deployment_revision: int,
                operation_id: str) -> list[dict]:
        observation = validate_observation(manifest, result, frame,
            deployment_revision=deployment_revision, operation_id=operation_id)
        source, binding = observation["source"], observation["binding"]
        now, index, request = source["capture_monotonic_ts"], source["frame_index"], observation["request_id"]
        task_id = binding["task_id"]
        identity = _sha({"binding": binding, "source_session_id": source["source_session_id"],
                         "source_epoch": source["source_epoch"]})
        context = self._contexts.get(task_id)
        if context is not None and deployment_revision < context.deployment_revision:
            raise TaskEventValidationError("a newer task deployment already superseded this observation")
        if context is None or context.identity != identity:
            if context is None and len(self._contexts) >= self._max_tasks:
                raise TaskEventValidationError("event aggregation task capacity exhausted")
            context = _Context(identity, deployment_revision)
            self._contexts[task_id] = context
        if index <= context.last_frame or request <= context.last_request:
            return []
        max_gap = manifest["config"]["sampling"]["max_gap_seconds"]
        if context.last_seen is not None:
            if now <= context.last_seen:
                raise TaskEventValidationError("capture time did not advance within the same source session")
            if now - context.last_seen > max_gap:
                context.tracks.clear()
        rules = manifest["config"]["rules"]
        available = list(context.tracks)
        tracks, events = [], []
        for number, detection in enumerate(observation["detections"]):
            matches = [track for track in available if track.label == detection["label"]
                       and _iou(track.box, detection["bbox_xyxy"]) >= 0.5]
            # Ambiguous associations start a new duration; they never inherit
            # another object's already-confirmed presence or cooldown state.
            if len(matches) == 1 and sum(
                item["label"] == detection["label"] and _iou(matches[0].box, item["bbox_xyxy"]) >= 0.5
                for item in observation["detections"]
            ) == 1:
                track = matches[0]
                available.remove(track)
            else:
                track = _Track(_sha({"context": identity, "first_frame": index, "object": number,
                                     "label": detection["label"], "bbox": detection["bbox_xyxy"]}),
                               detection["label"], detection["bbox_xyxy"], deepcopy(source), now)
            track.box = detection["bbox_xyxy"]
            tracks.append(track)
            duration = max(0.0, now - track.first_seen)
            cooled = track.last_emitted is None or now - track.last_emitted >= rules["cooldown_seconds"]
            if duration < rules["min_duration_seconds"] or not cooled:
                continue
            event_id = "TEV-" + _sha({"track_id": track.track_id, "frame_index": index,
                                      "request_id": request})[:48]
            event = {"schema_version": "argus.task-event.v1", "event_id": event_id,
                     "event_type": "object_detected", "mode": "validation", "status": "new",
                     "binding": deepcopy(binding), "source": deepcopy(source),
                     "first_source": track.first_source, "request_id": request,
                     "occurred_at": datetime.fromtimestamp(source["capture_wall_ts"], timezone.utc).isoformat(),
                     "object": deepcopy(detection), "duration_seconds": duration,
                     "frame_size": {"width": observation["width"], "height": observation["height"]},
                     "production_allowed": False, "alerts_emitted": False,
                     "replay": {"kind": "keyframe_only", "ready": False}}
            events.append(event)
            track.last_emitted = now
        context.tracks, context.last_seen = tracks, now
        context.last_frame, context.last_request = index, request
        return events
