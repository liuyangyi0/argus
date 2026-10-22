from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.inventory_regression_corpus import (
    build_inventory,
    build_markdown,
    validate_annotations,
)


def _write_video(path: Path, *, frames: int = 6, fps: float = 3.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (16, 12),
    )
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((12, 16, 3), index * 20, dtype=np.uint8))
    writer.release()


def test_build_inventory_validates_video_and_annotations(tmp_path: Path) -> None:
    video = tmp_path / "clips" / "positive" / "case.avi"
    annotation = tmp_path / "annotations" / "case.labels.json"
    _write_video(video)
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps({
        "positive_ranges": [{"start_frame": 1, "end_frame": 3, "label": "static"}],
        "ignore_ranges": [],
    }))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """suite_id: test-suite
default_camera_id: cam
clips:
  - id: case
    kind: static_foreign
    video: clips/positive/case.avi
    annotations: annotations/case.labels.json
    expected_video:
      resolution: 16x12
      fps: 3
      frames: 6
""",
        encoding="utf-8",
    )

    result = build_inventory(manifest, include_sha256=True)

    assert result["status"] == "ok"
    assert result["counts"] == {
        "total": 1, "ok": 1, "warning": 0, "error": 0, "unregistered": 0,
    }
    assert result["cases"][0]["annotations"]["positive_range_count"] == 1
    assert len(result["cases"][0]["sha256"]) == 64
    assert "| `case` | static_foreign | 16x12 |" in build_markdown(result)


def test_build_inventory_reports_unregistered_and_out_of_bounds(tmp_path: Path) -> None:
    registered = tmp_path / "clips" / "positive" / "registered.avi"
    unregistered = tmp_path / "clips" / "negative" / "extra.avi"
    annotation = tmp_path / "annotations" / "registered.labels.json"
    _write_video(registered)
    _write_video(unregistered)
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps({
        "positive_ranges": [{"start_frame": 2, "end_frame": 99, "label": "static"}],
    }))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """clips:
  - id: registered
    kind: static_foreign
    camera_id: cam
    video: clips/positive/registered.avi
    annotations: annotations/registered.labels.json
""",
        encoding="utf-8",
    )

    result = build_inventory(manifest)

    assert result["status"] == "error"
    assert result["counts"]["unregistered"] == 1
    assert result["unregistered_videos"] == [str(unregistered.resolve())]
    assert "outside video" in result["cases"][0]["errors"][0]


def test_validate_annotations_rejects_invalid_schema(tmp_path: Path) -> None:
    annotation = tmp_path / "bad.json"
    annotation.write_text(json.dumps({"positive_ranges": "not-a-list"}))

    summary, errors, warnings = validate_annotations(annotation, fps=30.0, total_frames=100)

    assert summary["valid"] is False
    assert errors == ["annotation.positive_ranges: must be a list"]
    assert warnings == []
