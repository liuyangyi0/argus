"""Inventory and validate the real-video regression corpus.

The inventory is intentionally model-free. It verifies that every manifest
entry points to a readable video, that the recorded metadata and annotations
fit the video, and that no video under the corpus ``clips`` directory is left
unregistered.

Example:
    python scripts/inventory_regression_corpus.py \
        --manifest data/regression/manifest.yaml \
        --output-dir data/regression/reports/inventory \
        --sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

try:
    from scripts.run_regression_suite import RegressionClip, load_manifest
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_regression_suite import RegressionClip, load_manifest


VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    """Return OpenCV metadata for one video without running inference."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {"readable": False, "error": "could_not_open"}
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        duration = frames / fps if frames > 0 and fps > 0 else None
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            return {
                "readable": False,
                "error": "could_not_decode_first_frame",
                "width": width,
                "height": height,
                "fps": fps,
                "frames": frames,
                "duration_seconds": duration,
            }
        return {
            "readable": True,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frames,
            "duration_seconds": duration,
        }
    finally:
        capture.release()


def _read_annotation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotation root must be an object")
    return payload


def _range_frame(item: dict[str, Any], frame_key: str, sec_key: str, fps: float) -> int | None:
    if frame_key in item:
        return int(item[frame_key])
    if sec_key in item:
        return int(round(float(item[sec_key]) * fps))
    return None


def validate_annotations(
    annotation_path: Path,
    *,
    fps: float,
    total_frames: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate annotation schema and range bounds."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload = _read_annotation(annotation_path)
    except Exception as exc:
        return {"path": str(annotation_path), "valid": False}, [f"annotation: {exc}"], []

    summary: dict[str, Any] = {
        "path": str(annotation_path),
        "valid": True,
        "positive_range_count": 0,
        "ignore_range_count": 0,
        "labels": [],
    }
    labels: set[str] = set()
    for key in ("positive_ranges", "ignore_ranges"):
        values = payload.get(key, [])
        if not isinstance(values, list):
            errors.append(f"annotation.{key}: must be a list")
            continue
        summary[f"{key[:-1]}_count"] = len(values)
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                errors.append(f"annotation.{key}[{index}]: must be an object")
                continue
            start = _range_frame(item, "start_frame", "start_sec", fps)
            end = _range_frame(item, "end_frame", "end_sec", fps)
            if start is None:
                errors.append(f"annotation.{key}[{index}]: missing start")
                continue
            if end is None:
                end = start
            if start < 0 or end < start:
                errors.append(f"annotation.{key}[{index}]: invalid range {start}-{end}")
            if total_frames > 0 and end >= total_frames:
                errors.append(
                    f"annotation.{key}[{index}]: end frame {end} outside video "
                    f"(last frame {total_frames - 1})"
                )
            label = str(item.get("label") or item.get("name") or "").strip()
            if label:
                labels.add(label)
            else:
                warnings.append(f"annotation.{key}[{index}]: missing label")

    summary["labels"] = sorted(labels)
    summary["valid"] = not errors
    return summary, errors, warnings


def _expected_resolution(raw_clip: dict[str, Any]) -> tuple[int, int] | None:
    expected = raw_clip.get("expected_video") or {}
    value = expected.get("resolution") if isinstance(expected, dict) else None
    if isinstance(value, str) and "x" in value.lower():
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def inventory_clip(
    clip: RegressionClip,
    raw_clip: dict[str, Any],
    *,
    include_sha256: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {
        "case_id": clip.clip_id,
        "kind": clip.kind,
        "camera_id": clip.camera_id,
        "video_path": str(clip.video_path) if clip.video_path else None,
        "annotation_path": str(clip.annotation_path) if clip.annotation_path else None,
        "expected_detection_types": list(clip.expected_detection_types),
        "tags": list(clip.tags),
        "description": clip.description,
    }
    if clip.video_path is None:
        errors.append("video path missing from manifest")
        result.update(status="error", errors=errors, warnings=warnings)
        return result
    if not clip.video_path.is_file():
        errors.append("video file missing")
        result.update(status="error", errors=errors, warnings=warnings)
        return result

    result["size_bytes"] = clip.video_path.stat().st_size
    if include_sha256:
        result["sha256"] = _sha256(clip.video_path)
    metadata = probe_video(clip.video_path)
    result["video"] = metadata
    if not metadata.get("readable"):
        errors.append(f"video: {metadata.get('error', 'unreadable')}")
    else:
        expected = raw_clip.get("expected_video") or {}
        expected_resolution = _expected_resolution(raw_clip)
        if expected_resolution and expected_resolution != (
            metadata["width"], metadata["height"]
        ):
            errors.append(
                f"video.resolution: expected {expected_resolution[0]}x{expected_resolution[1]}, "
                f"got {metadata['width']}x{metadata['height']}"
            )
        expected_fps = expected.get("fps") if isinstance(expected, dict) else None
        if expected_fps is not None and not math.isclose(
            float(expected_fps), float(metadata["fps"]), rel_tol=0.01, abs_tol=0.1
        ):
            errors.append(f"video.fps: expected {expected_fps}, got {metadata['fps']:.4f}")
        expected_frames = expected.get("frames") if isinstance(expected, dict) else None
        if expected_frames is not None and abs(int(expected_frames) - metadata["frames"]) > 1:
            errors.append(
                f"video.frames: expected {expected_frames}, got {metadata['frames']}"
            )

    if clip.annotation_path is not None:
        if not clip.annotation_path.is_file():
            errors.append("annotation file missing")
        elif metadata.get("readable"):
            annotation, annotation_errors, annotation_warnings = validate_annotations(
                clip.annotation_path,
                fps=float(metadata["fps"]),
                total_frames=int(metadata["frames"]),
            )
            result["annotations"] = annotation
            errors.extend(annotation_errors)
            warnings.extend(annotation_warnings)
    elif not clip.is_negative:
        warnings.append("positive clip has no annotation file")

    result["status"] = "error" if errors else "warning" if warnings else "ok"
    result["errors"] = errors
    result["warnings"] = warnings
    return result


def discover_unregistered_videos(root: Path, registered: set[Path]) -> list[str]:
    clips_dir = root / "clips"
    if not clips_dir.is_dir():
        return []
    discovered = {
        path.resolve()
        for path in clips_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }
    return [str(path) for path in sorted(discovered - registered)]


def build_inventory(manifest_path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    raw, clips = load_manifest(manifest_path)
    raw_by_id = {
        str(item.get("id")): item
        for item in raw.get("clips") or []
        if isinstance(item, dict) and item.get("id")
    }
    cases = [
        inventory_clip(clip, raw_by_id.get(clip.clip_id, {}), include_sha256=include_sha256)
        for clip in clips
    ]
    registered = {
        clip.video_path.resolve()
        for clip in clips
        if clip.video_path is not None and clip.video_path.exists()
    }
    unregistered = discover_unregistered_videos(manifest_path.parent, registered)
    counts = {
        "total": len(cases),
        "ok": sum(case["status"] == "ok" for case in cases),
        "warning": sum(case["status"] == "warning" for case in cases),
        "error": sum(case["status"] == "error" for case in cases),
        "unregistered": len(unregistered),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "suite_id": raw.get("suite_id"),
        "source_host": raw.get("source_host"),
        "source_dir": raw.get("source_dir"),
        "sha256_enabled": include_sha256,
        "status": "error" if counts["error"] else "warning" if counts["warning"] or unregistered else "ok",
        "counts": counts,
        "unregistered_videos": unregistered,
        "cases": cases,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    lines = [
        "# Regression Corpus Inventory",
        "",
        f"- Status: **{payload['status']}**",
        f"- Manifest: `{payload['manifest']}`",
        f"- Cases: {counts['total']} (ok={counts['ok']}, warning={counts['warning']}, error={counts['error']})",
        f"- Unregistered videos: {counts['unregistered']}",
        "",
        "## Cases",
        "",
        "| Case | Kind | Video | FPS | Frames | Positive ranges | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        video = case.get("video") or {}
        annotations = case.get("annotations") or {}
        lines.append(
            f"| `{case['case_id']}` | {case['kind']} | "
            f"{video.get('width', '-')}x{video.get('height', '-')} | "
            f"{video.get('fps', '-')} | {video.get('frames', '-')} | "
            f"{annotations.get('positive_range_count', 0)} | {case['status']} |"
        )
        for message in case.get("errors") or []:
            lines.append(f"\n- **{case['case_id']} error:** {message}")
        for message in case.get("warnings") or []:
            lines.append(f"\n- **{case['case_id']} warning:** {message}")
    if payload["unregistered_videos"]:
        lines.extend(["", "## Unregistered Videos", ""])
        lines.extend(f"- `{path}`" for path in payload["unregistered_videos"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/regression/manifest.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/regression/reports/inventory"),
    )
    parser.add_argument("--sha256", action="store_true", help="Hash complete video files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_inventory(args.manifest, include_sha256=args.sha256)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "corpus_inventory.json"
    md_path = args.output_dir / "corpus_inventory.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    print(f"JSON inventory written to {json_path}")
    print(f"Markdown inventory written to {md_path}")
    print(json.dumps(payload["counts"], ensure_ascii=False))
    return 1 if payload["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
