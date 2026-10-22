"""Offline video evaluation for fixed-scene anomaly monitoring.

The evaluator intentionally accepts any detector-like object with
``predict(frame).anomaly_score`` so it can be used with AnomalibDetector,
PersistentChangeScorer adapters, or deterministic test doubles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog

from argus.anomaly.metrics import evaluate_at_threshold, find_optimal_threshold

logger = structlog.get_logger()


@dataclass(frozen=True)
class VideoLabelRange:
    """Inclusive frame range from an offline video annotation file."""

    start_frame: int
    end_frame: int
    label: int
    name: str = ""

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "label": self.label,
            "name": self.name,
        }


def _range_frame_value(item: dict, frame_key: str, sec_key: str, fps: float) -> int | None:
    if frame_key in item:
        return int(item[frame_key])
    if sec_key in item:
        return int(round(float(item[sec_key]) * fps))
    return None


def _parse_ranges(
    items: list[dict],
    *,
    fps: float,
    label: int,
) -> list[VideoLabelRange]:
    ranges: list[VideoLabelRange] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = _range_frame_value(item, "start_frame", "start_sec", fps)
        end = _range_frame_value(item, "end_frame", "end_sec", fps)
        if start is None:
            start = _range_frame_value(item, "frame", "time_sec", fps)
        if end is None:
            end = start
        if start is None or end is None:
            continue
        start_i = max(0, int(start))
        end_i = max(start_i, int(end))
        ranges.append(VideoLabelRange(
            start_frame=start_i,
            end_frame=end_i,
            label=label,
            name=str(item.get("label") or item.get("name") or ""),
        ))
    return ranges


def load_video_annotations(annotation_path: str | Path, *, fps: float) -> dict:
    """Load JSON video annotations.

    Supported schema:
        {
          "positive_ranges": [{"start_frame": 10, "end_frame": 20}],
          "ignore_ranges": [{"start_sec": 4.0, "end_sec": 5.0}]
        }

    ``anomaly_ranges`` and ``events`` are accepted aliases for positive ranges.
    Frame ranges are inclusive. When annotations are provided, frames outside
    positive/ignore ranges are treated as normal.
    """
    path = Path(annotation_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("video annotation JSON must be an object")

    positive_items: list[dict] = []
    for key in ("positive_ranges", "anomaly_ranges", "events"):
        value = raw.get(key)
        if isinstance(value, list):
            positive_items.extend(value)

    ignore_items = raw.get("ignore_ranges")
    if not isinstance(ignore_items, list):
        ignore_items = []

    return {
        "path": str(path),
        "positive_ranges": [
            item.to_dict()
            for item in _parse_ranges(positive_items, fps=fps, label=1)
        ],
        "ignore_ranges": [
            item.to_dict()
            for item in _parse_ranges(ignore_items, fps=fps, label=-1)
        ],
    }


def _ranges_from_dicts(items: list[dict]) -> list[VideoLabelRange]:
    ranges: list[VideoLabelRange] = []
    for item in items:
        ranges.append(VideoLabelRange(
            start_frame=int(item["start_frame"]),
            end_frame=int(item["end_frame"]),
            label=int(item["label"]),
            name=str(item.get("name") or ""),
        ))
    return ranges


def _label_for_frame(
    frame_index: int,
    *,
    positive_ranges: list[VideoLabelRange],
    ignore_ranges: list[VideoLabelRange],
    has_annotations: bool,
) -> int | None:
    if any(item.contains(frame_index) for item in ignore_ranges):
        return None
    if any(item.contains(frame_index) for item in positive_ranges):
        return 1
    return 0 if has_annotations else None


def _extract_score(result: Any) -> float:
    if isinstance(result, dict):
        value = result.get("anomaly_score", result.get("score"))
    else:
        value = getattr(result, "anomaly_score", None)
    if value is None:
        raise ValueError("detector result does not contain anomaly_score")
    score = float(value)
    if not np.isfinite(score):
        raise ValueError("detector returned a non-finite anomaly_score")
    return score


def _decision(label: int | None, predicted: int) -> str:
    if label is None:
        return "unlabeled"
    if label == 1 and predicted == 1:
        return "tp"
    if label == 1 and predicted == 0:
        return "fn"
    if label == 0 and predicted == 1:
        return "fp"
    return "tn"


def _export_misclassified_frame(
    export_dir: Path,
    frame: np.ndarray,
    record: dict,
) -> str | None:
    if record["decision"] == "fp":
        subdir = export_dir / "false_positives"
        prefix = "fp"
    elif record["decision"] == "fn":
        subdir = export_dir / "false_negatives"
        prefix = "fn"
    else:
        return None

    subdir.mkdir(parents=True, exist_ok=True)
    score = f"{float(record['score']):.3f}".replace(".", "p")
    path = subdir / f"{prefix}_{int(record['frame_index']):06d}_score_{score}.jpg"
    if cv2.imwrite(str(path), frame):
        return str(path)
    return None


def evaluate_video(
    detector,
    video_path: str | Path,
    *,
    annotation_path: str | Path | None = None,
    threshold: float = 0.7,
    sample_every: int = 1,
    max_frames: int | None = None,
    export_dir: str | Path | None = None,
    sweep: bool = False,
) -> dict:
    """Score an offline video and compute frame-level metrics when labeled.

    Args:
        detector: object with ``predict(frame)`` returning ``anomaly_score``.
        video_path: local video file.
        annotation_path: optional JSON ranges. Outside positives is normal.
        threshold: anomaly decision threshold.
        sample_every: score every Nth frame.
        max_frames: optional cap on raw frame index reads.
        export_dir: optional output root for FP/FN review frames.
        sweep: include best-F1 threshold when labels are available.
    """
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")

    video = Path(video_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    annotations = (
        load_video_annotations(annotation_path, fps=fps)
        if annotation_path is not None
        else None
    )
    positive_ranges = _ranges_from_dicts(annotations["positive_ranges"]) if annotations else []
    ignore_ranges = _ranges_from_dicts(annotations["ignore_ranges"]) if annotations else []
    export_root = Path(export_dir) if export_dir is not None else None

    frame_records: list[dict] = []
    failed_frames: list[dict] = []
    ignored_frames = 0
    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            current_index = frame_index
            frame_index += 1
            if current_index % sample_every != 0:
                continue

            label = _label_for_frame(
                current_index,
                positive_ranges=positive_ranges,
                ignore_ranges=ignore_ranges,
                has_annotations=annotations is not None,
            )
            if label is None and annotations is not None:
                ignored_frames += 1
                continue

            try:
                score = _extract_score(detector.predict(frame))
            except Exception as exc:
                failed_frames.append({
                    "frame_index": current_index,
                    "error": str(exc),
                })
                continue

            predicted = 1 if score >= threshold else 0
            timestamp_sec = current_index / fps
            record = {
                "frame_index": current_index,
                "timestamp_sec": float(timestamp_sec),
                "score": float(score),
                "label": label,
                "predicted": predicted,
                "decision": _decision(label, predicted),
            }
            if export_root is not None:
                export_path = _export_misclassified_frame(export_root, frame, record)
                if export_path:
                    record["export_path"] = export_path
            frame_records.append(record)
    finally:
        cap.release()

    labeled = [record for record in frame_records if record["label"] is not None]
    metrics = None
    optimal_threshold = None
    if labeled:
        y_true = np.array([record["label"] for record in labeled], dtype=np.int64)
        y_scores = np.array([record["score"] for record in labeled], dtype=np.float64)
        metrics = evaluate_at_threshold(y_true, y_scores, threshold)
        if sweep:
            optimal_threshold = find_optimal_threshold(y_true, y_scores, target="f1")

    false_positive_frames = [
        int(record["frame_index"]) for record in frame_records if record["decision"] == "fp"
    ]
    false_negative_frames = [
        int(record["frame_index"]) for record in frame_records if record["decision"] == "fn"
    ]
    result = {
        "video_path": str(video),
        "annotation_path": str(annotation_path) if annotation_path is not None else None,
        "threshold": float(threshold),
        "sample_every": int(sample_every),
        "fps": float(fps),
        "total_frames": total_frames,
        "scored_frames": len(frame_records),
        "ignored_frames": ignored_frames,
        "failed_frames": failed_frames,
        "metrics": metrics,
        "optimal_threshold": optimal_threshold,
        "false_positive_frames": false_positive_frames,
        "false_negative_frames": false_negative_frames,
        "frames": frame_records,
        "annotations": annotations,
    }

    logger.info(
        "video_eval.complete",
        video=str(video),
        scored_frames=len(frame_records),
        false_positives=len(false_positive_frames),
        false_negatives=len(false_negative_frames),
    )
    return result


def write_video_evaluation_report(result: dict, output_path: str | Path) -> None:
    """Write an evaluation result as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "VideoLabelRange",
    "evaluate_video",
    "load_video_annotations",
    "write_video_evaluation_report",
]
