"""Tests for offline video evaluation tooling."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from argus.validation.video_eval import evaluate_video, load_video_annotations


class MeanBrightnessDetector:
    def predict(self, frame: np.ndarray):
        return SimpleNamespace(anomaly_score=float(frame.mean() / 255.0))


def _write_test_video(path, frames: list[np.ndarray], *, fps: int = 10) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG VideoWriter unavailable")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def test_load_video_annotations_accepts_seconds(tmp_path):
    annotation_path = tmp_path / "labels.json"
    annotation_path.write_text(
        json.dumps({
            "positive_ranges": [{"start_sec": 0.2, "end_sec": 0.4, "label": "object"}],
            "ignore_ranges": [{"start_frame": 8, "end_frame": 9}],
        }),
        encoding="utf-8",
    )

    annotations = load_video_annotations(annotation_path, fps=10.0)

    assert annotations["positive_ranges"] == [{
        "start_frame": 2,
        "end_frame": 4,
        "label": 1,
        "name": "object",
    }]
    assert annotations["ignore_ranges"][0]["start_frame"] == 8


def test_evaluate_video_computes_metrics_and_exports_misclassified_frames(tmp_path):
    dark = np.full((32, 48, 3), 20, dtype=np.uint8)
    bright = np.full((32, 48, 3), 240, dtype=np.uint8)
    video_path = tmp_path / "eval.avi"
    _write_test_video(
        video_path,
        [
            dark,    # normal -> TN
            dark,    # normal -> TN
            bright,  # positive -> TP
            dark,    # positive -> FN
            bright,  # normal -> FP
            bright,  # ignored
        ],
    )
    annotation_path = tmp_path / "labels.json"
    annotation_path.write_text(
        json.dumps({
            "positive_ranges": [{"start_frame": 2, "end_frame": 3}],
            "ignore_ranges": [{"start_frame": 5, "end_frame": 5}],
        }),
        encoding="utf-8",
    )
    export_dir = tmp_path / "review"

    result = evaluate_video(
        MeanBrightnessDetector(),
        video_path,
        annotation_path=annotation_path,
        threshold=0.5,
        export_dir=export_dir,
        sweep=True,
    )

    assert result["scored_frames"] == 5
    assert result["ignored_frames"] == 1
    assert result["metrics"]["confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
    assert result["metrics"]["precision"] == pytest.approx(0.5)
    assert result["metrics"]["recall"] == pytest.approx(0.5)
    assert result["false_positive_frames"] == [4]
    assert result["false_negative_frames"] == [3]
    assert result["optimal_threshold"] is not None
    assert (export_dir / "false_positives").is_dir()
    assert (export_dir / "false_negatives").is_dir()
    assert len(list((export_dir / "false_positives").glob("*.jpg"))) == 1
    assert len(list((export_dir / "false_negatives").glob("*.jpg"))) == 1


def test_evaluate_video_without_annotations_keeps_scores_unlabeled(tmp_path):
    frame = np.full((16, 16, 3), 240, dtype=np.uint8)
    video_path = tmp_path / "unlabeled.avi"
    _write_test_video(video_path, [frame, frame], fps=5)

    result = evaluate_video(MeanBrightnessDetector(), video_path, threshold=0.5)

    assert result["metrics"] is None
    assert result["scored_frames"] == 2
    assert result["frames"][0]["label"] is None
    assert result["frames"][0]["decision"] == "unlabeled"
