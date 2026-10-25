"""Fixed, shadow-only task adapters. Model SDKs are imported only in workers.

Only declared YOLOv8 raw ONNX outputs are accepted; this is not an arbitrary
model/plugin loader. Missing packages, unsupported exports, and unavailable
execution providers fail explicitly instead of downloading or falling back.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from argus.core.analysis_task_contracts import model_role_incompatibilities


class TaskAdapterError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _artifact_bytes(descriptor: dict, models_root: str | Path, limit_bytes: int) -> bytes:
    root = Path(models_root).resolve(strict=True)
    path = Path(descriptor["artifact_path"]).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix.lower() != ".onnx" or not path.is_file():
        raise TaskAdapterError("INVALID_BUNDLE", "model must be an ONNX file beneath models_root")
    with path.open("rb") as stream:
        payload = stream.read(limit_bytes + 1)
    if len(payload) > limit_bytes:
        raise TaskAdapterError("RESOURCE_EXHAUSTED", "registered model exceeds its load budget")
    digest = descriptor.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or hashlib.sha256(payload).hexdigest() != digest:
        raise TaskAdapterError("INVALID_BUNDLE", "registered model SHA-256 does not match")
    return payload


def _reject_external_onnx_data(payload: bytes) -> None:
    try:
        onnx = importlib.import_module("onnx")
    except ImportError as exc:
        raise TaskAdapterError("UNAVAILABLE", "onnx is required for safe ONNX validation") from exc
    model = onnx.load_model_from_string(payload)

    def inspect_graph(graph) -> None:
        tensors = list(graph.initializer)
        for sparse in getattr(graph, "sparse_initializer", []):
            tensors.extend([sparse.values, sparse.indices])
        for tensor in tensors:
            if getattr(tensor, "external_data", None) or getattr(tensor, "data_location", 0) == 1:
                raise TaskAdapterError("INVALID_BUNDLE", "external ONNX tensor files are not supported")
        for node in graph.node:
            if node.domain not in ("", "ai.onnx"):
                raise TaskAdapterError("INVALID_BUNDLE", "custom ONNX operator domains are not supported")
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    inspect_graph(attribute.g)
                for nested in attribute.graphs:
                    inspect_graph(nested)
                tensors = [attribute.t] if attribute.HasField("t") else []
                tensors.extend(attribute.tensors)
                sparse_tensors = list(getattr(attribute, "sparse_tensors", []))
                if attribute.HasField("sparse_tensor"):
                    sparse_tensors.append(attribute.sparse_tensor)
                for sparse in sparse_tensors:
                    tensors.extend([sparse.values, sparse.indices])
                if any(getattr(tensor, "external_data", None)
                       or getattr(tensor, "data_location", 0) == 1 for tensor in tensors):
                    raise TaskAdapterError("INVALID_BUNDLE", "external ONNX constants are not supported")

    if getattr(model, "functions", None):
        raise TaskAdapterError("INVALID_BUNDLE", "ONNX local functions are not supported by this adapter")
    inspect_graph(model.graph)


def _iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def _in_region(box: list[float], polygon: list[list[float]], width: int, height: int) -> bool:
    if not polygon:
        return True
    x, y = (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > y) != (previous[1] > y):
            boundary = ((previous[0] - current[0]) * (y - current[1])
                        / (previous[1] - current[1]) + current[0])
            if x < boundary:
                inside = not inside
        previous = current
    return inside


class OnnxYoloAdapter:
    """YOLOv8 raw detection or COCO-17 pose, one image and one raw output."""

    def __init__(
        self, descriptor: dict, config: dict, models_root: str | Path,
        *, ort_module: Any | None = None, validator=None,
    ) -> None:
        self.descriptor = descriptor
        self.rules = config["rules"]
        self.device = config["resources"]["device"]
        self.pose = descriptor["adapter_id"] == "onnx_pose.v1"
        expected = "yolov8_pose.v1" if self.pose else "yolov8_detection.v1"
        if descriptor.get("adapter_id") not in ("onnx_detection.v1", "onnx_pose.v1"):
            raise TaskAdapterError("INVALID_BUNDLE", "unsupported model adapter")
        if descriptor.get("output_schema") != expected:
            raise TaskAdapterError("INVALID_BUNDLE", "adapter/output schema mismatch")
        self.labels = descriptor["labels"]
        if self.pose and self.labels != ["person"]:
            raise TaskAdapterError("INVALID_BUNDLE", "COCO-17 pose requires labels=['person']")
        self.height, self.width = descriptor["input_size"]
        memory_budget = config["resources"]["memory_limit_mb"] * 1024 * 1024
        # Reservation checks bound declared buffers before ORT allocates them.
        # They are not an OS-enforced process working-set limit.
        if self.height * self.width * 3 * 5 > memory_budget // 4:
            raise TaskAdapterError("RESOURCE_EXHAUSTED", "static model input exceeds its declared buffer budget")
        if (descriptor.get("input_layout"), descriptor.get("input_dtype"),
                descriptor.get("color_order")) != ("NCHW", "float32", "RGB"):
            raise TaskAdapterError("INVALID_BUNDLE", "unsupported model preprocessing descriptor")
        payload = _artifact_bytes(
            descriptor, models_root, config["resources"]["memory_limit_mb"] * 1024 * 1024 // 2,
        )
        (validator or _reject_external_onnx_data)(payload)
        if ort_module is None:
            try:
                ort_module = importlib.import_module("onnxruntime")
            except ImportError as exc:
                raise TaskAdapterError("UNAVAILABLE", "onnxruntime is not installed") from exc
        provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}.get(self.device)
        if provider is None or provider not in ort_module.get_available_providers():
            raise TaskAdapterError("UNAVAILABLE", f"requested {self.device} execution provider unavailable")
        version = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(getattr(ort_module, "__version__", "")))
        if version is None or tuple(map(int, version.groups())) < (1, 23, 2):
            # Older wrappers accept arbitrary kwargs but ignore enable_fallback,
            # making a post-construction disable_fallback() call too late.
            raise TaskAdapterError("UNAVAILABLE", "onnxruntime >= 1.23.2 is required to disable constructor fallback")
        if self.device == "cuda" and sys.platform == "win32":
            # Spawn workers do not inherit DLLs loaded by a parent importing
            # torch. ORT's supported loader searches installed torch/NVIDIA
            # packages without changing PATH or reading model-local DLLs.
            preload = getattr(ort_module, "preload_dlls", None)
            if not callable(preload):
                raise TaskAdapterError("UNAVAILABLE", "ORT CUDA DLL preload API is unavailable")
            try:
                preload(cuda=True, cudnn=True, msvc=True, directory=None)
            except Exception as exc:
                raise TaskAdapterError("UNAVAILABLE", "ORT CUDA dependency preload failed") from exc
        options = ort_module.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        if self.device == "cuda":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort_module.InferenceSession(
            payload, sess_options=options, providers=[provider],
            enable_fallback=0, read_config_from_model=False,
        )
        # ORT's Python wrapper can retry EP failures on CPU independently of
        # graph-placement fallback. Disable that path before any run call.
        self.session.disable_fallback()
        active = self.session.get_providers()
        # ORT registers a default CPU EP even when CPU node assignment is
        # forbidden. The session config above rejects such nodes during
        # Initialize; registration alone is not evidence of CPU execution.
        allowed = [[provider]] if self.device == "cpu" else [
            [provider], [provider, "CPUExecutionProvider"],
        ]
        if active not in allowed:
            raise TaskAdapterError("UNAVAILABLE", "model provider does not match the requested device")
        self.provider = provider
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or list(inputs[0].shape) != [
            1, 3, self.height, self.width,
        ]:
            raise TaskAdapterError("INVALID_BUNDLE", "ONNX input must exactly match static NCHW descriptor")
        outputs = self.session.get_outputs()
        channels = 56 if self.pose else 4 + len(self.labels)
        if (len(outputs) != 1 or outputs[0].type != "tensor(float)"
                or len(outputs[0].shape) != 3 or list(outputs[0].shape[:2]) != [1, channels]
                or type(outputs[0].shape[2]) is not int or not 1 <= outputs[0].shape[2] <= 100000):
            raise TaskAdapterError("INVALID_BUNDLE", "only single raw YOLOv8 output is supported")
        self.output_shape = list(outputs[0].shape)
        if channels * self.output_shape[2] * 4 > min(32 * 1024 * 1024, memory_budget // 8):
            raise TaskAdapterError("RESOURCE_EXHAUSTED", "static model output exceeds its declared buffer budget")
        self.input_name = inputs[0].name

    def reset(self, reason: str) -> None:
        """The raw model has no temporal state."""

    def infer(self, frame: np.ndarray, timestamp: float) -> dict:
        import cv2

        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise TaskAdapterError("INVALID_FRAME", "adapter requires BGR uint8 HWC frames")
        height, width = frame.shape[:2]
        scale = min(self.width / width, self.height / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        pad_x, pad_y = (self.width - resized_w) // 2, (self.height - resized_h) // 2
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = cv2.resize(
            frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR,
        )
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = np.asarray(outputs[0])
        if raw.dtype != np.float32 or list(raw.shape) != self.output_shape:
            raise TaskAdapterError("INVALID_OUTPUT", "output does not match declared raw YOLOv8 schema")
        if not np.isfinite(raw).all():
            raise TaskAdapterError("INVALID_OUTPUT", "model output contains non-finite values")
        predictions = raw[0].T
        confidence = predictions[:, 4] if self.pose else predictions[:, 4:].max(axis=1)
        classes = np.zeros(len(predictions), dtype=int) if self.pose else predictions[:, 4:].argmax(axis=1)
        scores = predictions[:, 4:5] if self.pose else predictions[:, 4:]
        if np.any((scores < 0) | (scores > 1)):
            raise TaskAdapterError("INVALID_OUTPUT", "confidence values must be probabilities")
        indexes = np.flatnonzero(confidence >= self.rules["confidence_threshold"])
        indexes = indexes[np.argsort(-confidence[indexes], kind="stable")][:1000]
        detections = []
        for index in indexes:
            row = predictions[index]
            label = self.labels[int(classes[index])]
            if self.rules["labels"] and label not in self.rules["labels"]:
                continue
            cx, cy, box_w, box_h = map(float, row[:4])
            if box_w <= 0 or box_h <= 0:
                continue
            box = [max(0.0, (cx - box_w / 2 - pad_x) / scale),
                   max(0.0, (cy - box_h / 2 - pad_y) / scale),
                   min(float(width), (cx + box_w / 2 - pad_x) / scale),
                   min(float(height), (cy + box_h / 2 - pad_y) / scale)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if not _in_region(box, self.rules["region_polygon"], width, height):
                continue
            if any(item["label"] == label and _iou(box, item["bbox_xyxy"]) > self.rules["iou_threshold"]
                   for item in detections):
                continue
            detection = {"label": label, "confidence": float(confidence[index]), "bbox_xyxy": box}
            if self.pose:
                points = row[5:].reshape(17, 3)
                if np.any((points[:, 2] < 0) | (points[:, 2] > 1)):
                    raise TaskAdapterError("INVALID_OUTPUT", "invalid keypoint confidence")
                detection["keypoints"] = [
                    {"index": point_index, "x": float((x - pad_x) / scale),
                     "y": float((y - pad_y) / scale), "confidence": float(score)}
                    for point_index, (x, y, score) in enumerate(points)
                ]
                detection["keypoint_schema"] = "coco17.v1"
            detections.append(detection)
            if len(detections) == 100:
                break
        return {"state": "observed", "detections": detections}


class PpeAdapter:
    """Explicit bare-head evidence, never absence-of-helmet inference."""

    def __init__(self, detector, config: dict) -> None:
        self.detector = detector
        self.provider = detector.provider
        self.config = config
        self._tracks: list[dict] = []
        self._last_timestamp: float | None = None

    def reset(self, reason: str) -> None:
        self.detector.reset(reason)
        self._tracks.clear()
        self._last_timestamp = None

    @staticmethod
    def _belongs(part: dict, person: dict) -> bool:
        box, body = part["bbox_xyxy"], person["bbox_xyxy"]
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return body[0] <= x <= body[2] and body[1] <= y <= body[1] + 0.4 * (body[3] - body[1])

    def infer(self, frame: np.ndarray, timestamp: float) -> dict:
        if not math.isfinite(timestamp):
            raise TaskAdapterError("INVALID_FRAME", "missing finite capture timestamp")
        if (self._last_timestamp is not None and (
            timestamp <= self._last_timestamp
            or timestamp - self._last_timestamp > self.config["sampling"]["max_gap_seconds"]
        )):
            self.reset("sampling_gap")
        detected = self.detector.infer(frame, timestamp)["detections"]
        people = [item for item in detected if item["label"] == "person"]
        helmets = [item for item in detected if item["label"] == "helmet"]
        bare_heads = [item for item in detected if item["label"] == "bare_head"]
        outcomes, tracks, used = [], [], set()
        for person in people:
            positive = [item for item in helmets if self._belongs(item, person)
                        and sum(self._belongs(item, body) for body in people) == 1]
            negative = [item for item in bare_heads if self._belongs(item, person)
                        and sum(self._belongs(item, body) for body in people) == 1]
            state, reason, duration = "unknown", "no_explicit_head_evidence", 0.0
            if positive and not negative:
                state, reason = "compliant", "associated_helmet"
            elif positive and negative:
                reason = "conflicting_head_evidence"
            elif negative:
                candidates = [(index, track) for index, track in enumerate(self._tracks)
                              if index not in used and _iou(person["bbox_xyxy"], track["bbox"]) >= 0.5]
                if (len(candidates) == 1 and sum(
                    _iou(body["bbox_xyxy"], candidates[0][1]["bbox"]) >= 0.5 for body in people
                ) == 1):
                    index, previous = candidates[0]
                    used.add(index)
                    started = previous["started"]
                else:
                    started = timestamp
                duration = max(0.0, timestamp - started)
                tracks.append({"bbox": person["bbox_xyxy"], "started": started})
                state = "violation" if duration >= self.config["rules"]["min_duration_seconds"] else "pending"
                reason = "explicit_bare_head_associated_with_person"
            outcomes.append({"person": person, "state": state, "reason": reason,
                             "duration_seconds": duration,
                             "helmet_evidence": positive, "bare_head_evidence": negative})
        self._tracks = tracks
        self._last_timestamp = timestamp
        return {"state": "observed", "people": outcomes, "detections": detected,
                "has_candidate_event": any(item["state"] == "violation" for item in outcomes)}


class PpeRoleDetector:
    """Fixed same-frame person/head composition, never a partial observation."""

    def __init__(self, person_detector, head_detector) -> None:
        if person_detector.provider != head_detector.provider:
            raise TaskAdapterError("INVALID_BUNDLE", "PPE role providers do not match")
        self.person_detector = person_detector
        self.head_detector = head_detector
        self.provider = person_detector.provider

    def reset(self, reason: str) -> None:
        self.person_detector.reset(reason)
        self.head_detector.reset(reason)

    def infer(self, frame: np.ndarray, timestamp: float) -> dict:
        # Both reads consume the exact same immutable source image/time. If
        # either fails there is no aggregate result to publish or label.
        people = self.person_detector.infer(frame, timestamp)
        heads = self.head_detector.infer(frame, timestamp)
        if people.get("state") != "observed" or heads.get("state") != "observed":
            raise TaskAdapterError("INVALID_OUTPUT", "all PPE roles must finish the same observation")
        return {"state": "observed", "detections": (
            [item for item in people["detections"] if item["label"] == "person"]
            + [item for item in heads["detections"] if item["label"] in {"helmet", "bare_head"}]
        )}


class FastMotionAdapter:
    def __init__(self, config: dict, *, detector_factory=None) -> None:
        if detector_factory is None:
            from argus.prefilter.fast_motion import FastMotionDetector

            detector_factory = FastMotionDetector
        if config["sampling"]["mode"] != "ordered_window":
            raise TaskAdapterError("INVALID_BUNDLE", "fast motion requires ordered_window")
        if config["resources"]["device"] != "cpu":
            raise TaskAdapterError("UNAVAILABLE", "classical fast motion is CPU-only; CUDA was requested")
        self.provider = "classical-cpu"
        self.config = config
        self.detector = detector_factory(min_confidence=config["rules"]["confidence_threshold"])
        self._previous_timestamp: float | None = None
        self._sampling_blocked = False

    def reset(self, reason: str) -> None:
        self._previous_timestamp = None
        self._sampling_blocked = False
        self.detector.reset()

    def infer(self, frame: np.ndarray, timestamp: float) -> dict:
        if not math.isfinite(timestamp):
            raise TaskAdapterError("INVALID_FRAME", "missing finite capture timestamp")
        if self._sampling_blocked:
            return {"state": "sampling_insufficient", "reason": "sampling_gap_requires_reset",
                    "detections": [], "has_candidate_event": False}
        previous = self._previous_timestamp
        self._previous_timestamp = timestamp
        if previous is None:
            self.detector.process(frame, timestamp=timestamp)
            return {"state": "warming_up", "detections": [], "has_candidate_event": False}
        delta = timestamp - previous
        # A small timing tolerance accommodates scheduler clock noise, not
        # skipped frames. Required FPS remains part of this task's contract.
        maximum_gap = min(self.config["sampling"]["max_gap_seconds"],
                          1.05 / self.config["sampling"]["fps_target"])
        if delta <= 0 or delta > maximum_gap:
            self.detector.reset()
            self._sampling_blocked = True
            return {"state": "sampling_insufficient", "reason": "effective_frame_rate_below_requirement",
                    "effective_fps": 1 / delta if delta > 0 else None,
                    "detections": [], "has_candidate_event": False}
        result = self.detector.process(frame, timestamp=timestamp)
        detections = [candidate.to_detected_object() for candidate in result.candidates
                      if _in_region(list(candidate.bbox), self.config["rules"]["region_polygon"],
                                    frame.shape[1], frame.shape[0])]
        return {"state": "observed", "detections": detections,
                "has_candidate_event": bool(detections),
                "suppression_reason": result.suppression_reason, "effective_fps": 1 / delta}


def build_task_adapter(manifest: dict, models_root: str | Path):
    template, models, config = manifest["template_id"], manifest["models"], manifest["config"]
    if template == "fast_motion.v1" and not models:
        return FastMotionAdapter(config)
    if template == "ppe_compliance_dual.v1":
        roles = ("person_detector", "head_detector")
        if set(models) != set(roles) or any(
            model_role_incompatibilities(template, role, models[role], config["rules"]["labels"])
            for role in roles
        ):
            raise TaskAdapterError("INVALID_BUNDLE", "dual PPE model role/label mismatch")
        # Verify *all* artifact bytes before loading the first SDK session.
        # Each role receives a share of the task's declared load/buffer budget.
        role_config = deepcopy(config)
        role_config["resources"]["memory_limit_mb"] = config["resources"]["memory_limit_mb"] // len(roles)
        role_config["rules"]["labels"] = []
        for role in roles:
            _artifact_bytes(models[role], models_root,
                            role_config["resources"]["memory_limit_mb"] * 1024 * 1024 // 2)
        person = OnnxYoloAdapter(models[roles[0]], deepcopy(role_config), models_root)
        head = OnnxYoloAdapter(models[roles[1]], deepcopy(role_config), models_root)
        return PpeAdapter(PpeRoleDetector(person, head), config)
    role = "pose" if template == "pose_estimation.v1" else "detector"
    expected = "onnx_pose.v1" if role == "pose" else "onnx_detection.v1"
    if template not in ("object_detection.v1", "pose_estimation.v1", "ppe_compliance.v1"):
        raise TaskAdapterError("INVALID_BUNDLE", "task template is not executable")
    if set(models) != {role} or models[role].get("adapter_id") != expected:
        raise TaskAdapterError("INVALID_BUNDLE", "template/model role mismatch")
    detector_config = deepcopy(config)
    if template == "ppe_compliance.v1":
        if not {"person", "helmet", "bare_head"} <= set(models[role]["labels"]):
            raise TaskAdapterError("INVALID_BUNDLE", "PPE requires explicit person, helmet and bare_head labels")
        # Task label selection must not remove people/helmets needed to
        # disambiguate the explicit bare-head evidence association.
        detector_config["rules"]["labels"] = []
    adapter = OnnxYoloAdapter(models[role], detector_config, models_root)
    return PpeAdapter(adapter, config) if template == "ppe_compliance.v1" else adapter
