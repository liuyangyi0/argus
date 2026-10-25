"""Convert exact-frame shadow observations into explicitly unreviewed proposals."""

from __future__ import annotations

import math
from copy import deepcopy

_IDENTITY_FIELDS = ("camera_id", "source_session_id", "source_epoch", "frame_index",
                    "source_frame_index", "pts", "time_base_num", "time_base_den",
                    "capture_wall_ts", "capture_monotonic_ts", "frame_index_provenance", "timestamp_provenance")


def propose_review_annotations(envelope: dict, results: list[dict]) -> dict:
    """Never align by approximate timestamps or carry forward human approval."""
    payload = deepcopy(envelope)
    binding = envelope["binding"]
    if envelope.get("schema") != "argus.task-review-envelope.v1":
        raise ValueError("invalid task review envelope")
    proposals = []
    for sample in envelope["samples"]:
        identity = sample["identity"]
        matches = [result for result in results
                   if result.get("schema_version") == "argus.task-shadow-result.v1"
                   and result.get("task_id") == binding["task_id"]
                   and result.get("task_revision") == binding["task_revision"]
                   and result.get("task_config_sha256") == binding["task_config_sha256"]
                   and result.get("template_id") == binding["template_id"]
                   and result.get("production_allowed") is False
                   and all(result.get("source", {}).get(key) == identity.get(key) for key in _IDENTITY_FIELDS)
                   and result.get("data", {}).get("state") == "observed"]
        if not matches:
            continue
        result = matches[-1]
        width, height = sample["image_size"]
        if width <= 0 or height <= 0:
            raise ValueError("invalid annotation image size")
        objects = []
        for detection in result["data"].get("detections", []):
            # Fast-motion legacy candidates use a different, temporal schema.
            # They remain review evidence until a dedicated annotation adapter
            # exists; never guess a class or box convention.
            if "bbox_xyxy" not in detection:
                continue
            box = detection["bbox_xyxy"]
            if len(box) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
                raise ValueError("invalid model annotation geometry")
            normalized = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
            if not (0 <= normalized[0] < normalized[2] <= 1 and 0 <= normalized[1] < normalized[3] <= 1):
                raise ValueError("model annotation is outside the reviewed image")
            item = {"label": detection["label"], "bbox_xyxy": normalized}
            if binding["template_id"] == "pose_estimation.v1":
                points = []
                for point in detection.get("keypoints", []):
                    x, y = point["x"] / width, point["y"] / height
                    visible = point["confidence"] >= 0.5 and 0 <= x <= 1 and 0 <= y <= 1
                    points.append({"index": point["index"], "x": x if visible else 0.0,
                                   "y": y if visible else 0.0, "visibility": 2 if visible else 0})
                item["keypoints"] = points
                item["keypoint_schema"] = "coco17.v1"
            objects.append(item)
        if binding["template_id"] == "fast_motion.v1":
            continue
        proposals.append({
            "sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"], "origin": "auto",
            "annotation": {"schema": "argus.task-annotation.v1", "coordinate_space": "normalized",
                           "kind": "pose" if binding["template_id"] == "pose_estimation.v1" else "detection",
                           "objects": objects},
            "source": {"bundle_id": result["bundle_id"], "manifest_sha256": result["manifest_sha256"],
                       "models": result["models"], "request_id": result["request_id"],
                       "alignment": "exact_source_frame_identity.v1"},
        })
    payload["results"] = proposals
    payload["target"] = "cvat"
    return payload
