"""Prepare immutable source identities and export reviewed regression labels.

Preparation decodes videos only when explicitly invoked. It neither starts an
Argus runtime nor loads any model. PyAV timestamps here are review metadata;
production release still requires the existing strict decoder/soak gates.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from argus.integrations.contracts import (
    FrameTimestamp,
    ReviewAnnotations,
    ReviewBundle,
    ReviewClip,
    ReviewRange,
    file_sha256,
    require_verified_media,
    review_source_paths,
    write_json,
)


_SAFE_CLIP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_SPLITS = {"unassigned", "train", "validation", "test"}
_REVIEW_LABEL_SCHEMA = "argus.review-regression-labels.v1"


def _relative_timestamps(timestamps: list[FrameTimestamp]) -> list[float]:
    origin = timestamps[0].pts * Fraction(
        timestamps[0].time_base_num, timestamps[0].time_base_den,
    )
    return [
        float(item.pts * Fraction(item.time_base_num, item.time_base_den) - origin)
        for item in timestamps
    ]


def _safe_id(value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_CLIP_ID.fullmatch(value):
        raise ValueError("clip id must be a safe filename (letters, digits, dot, dash, underscore)")
    if value.split(".", 1)[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    } or value.endswith("."):
        raise ValueError("clip id cannot be a reserved Windows filename")
    return value


def _resolve(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a nonempty local file path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def _inspect_video(path: Path) -> dict[str, Any]:
    """Inspect every decoded frame without manufacturing FPS-based timestamps."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("video preparation requires PyAV: pip install 'av>=12'") from exc

    timestamps: list[FrameTimestamp] = []
    width = height = 0
    with av.open(str(path), options={"err_detect": "explode"}) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"review requires exactly one video stream: {path}")
        stream = container.streams.video[0]
        rotation = stream.metadata.get("rotate", "0")
        if float(rotation) % 360:
            raise ValueError("normalize rotated video before review to preserve box coordinates")
        rate = stream.average_rate
        fps = float(rate) if rate is not None else 0.0
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("video has no valid nominal frame rate")
        declared_frames = stream.frames
        for index, frame in enumerate(container.decode(stream)):
            if getattr(frame, "is_corrupt", False):
                raise ValueError(f"corrupt decoded frame {index}: {path}")
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"frame {index} has no exact presentation timestamp")
            time_base = Fraction(frame.time_base)
            if not index:
                width, height = frame.width, frame.height
            if (frame.width, frame.height) != (width, height):
                raise ValueError("video resolution changes during the clip")
            # Browser/CVAT display transforms must not rotate pixel coordinates
            # silently. Reject display matrices rather than guessing their effect.
            if any(
                getattr(item.type, "name", "") == "DISPLAYMATRIX"
                for item in getattr(frame, "side_data", [])
            ):
                raise ValueError("normalize video display-matrix transforms before review")
            frame_rotation = getattr(frame, "rotation", 0)
            if frame_rotation and float(frame_rotation) % 360:
                raise ValueError("normalize rotated video before review")
            timestamps.append(FrameTimestamp(
                frame_index=index,
                pts=frame.pts,
                time_base_num=time_base.numerator,
                time_base_den=time_base.denominator,
                timestamp_sec=float(frame.pts * time_base),
            ))
    if not timestamps:
        raise ValueError(f"video decoded no frames: {path}")
    if declared_frames and declared_frames != len(timestamps):
        raise ValueError("decoded frame count differs from the declared source frame count")
    return {
        "fps": fps, "frame_count": len(timestamps), "width": width, "height": height,
        "timestamps": timestamps,
    }


def _range_from_source(
    raw: Any, timestamps: list[FrameTimestamp], *, default_label: str = "anomaly",
    relative_seconds: list[float] | None = None,
) -> ReviewRange:
    if not isinstance(raw, dict):
        raise ValueError("annotation ranges must be objects")
    count = len(timestamps)
    if relative_seconds is None:
        relative_seconds = _relative_timestamps(timestamps)

    # The regression evaluator also accepts single-frame points and a missing
    # end endpoint. Preserve that contract without accepting contradictory forms.
    raw = dict(raw)
    single_point = not any(key in raw for key in ("end_frame", "end_sec"))
    point_keys = {key for key in ("frame", "time_sec") if key in raw}
    if point_keys:
        if len(point_keys) != 1 or any(
            key in raw for key in ("start_frame", "start_sec", "end_frame", "end_sec")
        ):
            raise ValueError("single-frame annotations cannot mix point and range coordinates")
        key = next(iter(point_keys))
        raw["start_frame" if key == "frame" else "start_sec"] = raw[key]

    def coordinate(prefix: str) -> int:
        frame_key, sec_key = f"{prefix}_frame", f"{prefix}_sec"
        if (frame_key in raw) == (sec_key in raw):
            raise ValueError("each range endpoint must specify exactly one frame or seconds value")
        if frame_key in raw:
            frame = raw[frame_key]
            if type(frame) is not int:
                raise ValueError("source frame indices must be integers")
            return frame
        seconds = raw[sec_key]
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("annotation seconds must be finite and nonnegative")
        if seconds > relative_seconds[-1] + 1e-8:
            raise ValueError("annotation seconds exceed the last source frame timestamp")
        if single_point:
            right = bisect.bisect_left(relative_seconds, seconds)
            candidates = {max(0, right - 1), min(count - 1, right)}
            return min(candidates, key=lambda index: (abs(relative_seconds[index] - seconds), index))
        if prefix == "start":
            return bisect.bisect_left(relative_seconds, seconds - 1e-8)
        return bisect.bisect_right(relative_seconds, seconds + 1e-8) - 1

    start = coordinate("start")
    end = start if single_point else coordinate("end")
    if not 0 <= start <= end < count:
        raise ValueError("source annotation range is empty, reversed, or outside the video")
    label = raw.get("label", raw.get("name", default_label))
    if not isinstance(label, str) or not label.strip():
        raise ValueError("source range labels must be nonempty strings")
    return ReviewRange(
        start_frame=start, end_frame=end, label=label,
        event_id=raw.get("event_id"),
    )


def _read_annotations(
    path: Path | None, timestamps: list[FrameTimestamp], *,
    video_sha256: str | None = None, clip_id: str | None = None,
) -> ReviewAnnotations:
    if path is None:
        return ReviewAnnotations()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotation JSON must be an object")
    if "source_video_sha256" in payload and payload["source_video_sha256"] != video_sha256:
        raise ValueError("annotation source video SHA256 does not match the manifest clip")
    source_keys = ("positive_ranges", "anomaly_ranges", "events")
    if "review_label_schema_version" in payload:
        if (
            payload["review_label_schema_version"] != _REVIEW_LABEL_SCHEMA
            or payload.get("review_event_semantics") != "source-object-events"
            or payload.get("evaluation_event_semantics") != "anomaly-episodes"
            or not isinstance(payload.get("review_positive_ranges"), list)
            or payload.get("source_video_sha256") != video_sha256
            or payload.get("source_clip_id") != clip_id
        ):
            raise ValueError("review label provenance or event semantics are invalid")
        # Evaluator episodes are deliberately coalesced. Re-review must recover
        # the original object events, without adding those episodes as tags.
        source_keys = ("review_positive_ranges",)
    relative_seconds = _relative_timestamps(timestamps)
    positives: list[ReviewRange] = []
    for key in source_keys:
        values = payload.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list")
        positives.extend(
            _range_from_source(item, timestamps, relative_seconds=relative_seconds)
            for item in values
        )
    ignores = payload.get("ignore_ranges", [])
    if not isinstance(ignores, list):
        raise ValueError("ignore_ranges must be a list")
    return ReviewAnnotations(
        positive_ranges=positives,
        ignore_ranges=[
            _range_from_source(
                item, timestamps, default_label="ignore", relative_seconds=relative_seconds,
            ) for item in ignores
        ],
        boxes=payload.get("boxes", []),
        # Legacy labels remain proposals until an explicit human review.
        reviewed=payload.get("reviewed", False),
        reviewed_by=payload.get("reviewed_by"),
    )


def prepare_review_bundle(
    manifest_path: str | Path, *, evaluation_dir: str | Path | None = None,
    split_file: str | Path | None = None,
) -> ReviewBundle:
    import yaml

    manifest = Path(manifest_path).resolve(strict=True)
    manifest_hash = file_sha256(manifest)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict) or type(payload.get("version")) is not int
        or payload["version"] != 1
    ):
        raise ValueError("review requires a version: 1 regression manifest")
    rows = payload.get("clips")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest clips must be a nonempty list")
    splits: dict[str, str] = {}
    if split_file is not None:
        splits = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if not isinstance(splits, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or value not in _SPLITS
            for key, value in splits.items()
        ):
            raise ValueError("split file must map clip ids to train/validation/test/unassigned")
    clips: list[ReviewClip] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every manifest clip must be an object")
        clip_id = _safe_id(row.get("id"))
        camera_id = row.get("camera_id") or payload.get("default_camera_id")
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError(f"manifest clip {clip_id} requires an explicit camera identity")
        video = _resolve(manifest.parent, row.get("video"))
        video_hash = file_sha256(video)
        metadata = _inspect_video(video)
        if file_sha256(video) != video_hash:
            raise ValueError(f"video changed during preparation: {clip_id}")
        annotations_path = (
            _resolve(manifest.parent, row["annotations"]) if row.get("annotations") else None
        )
        annotation_hash = file_sha256(annotations_path) if annotations_path else None
        annotations = _read_annotations(
            annotations_path, metadata["timestamps"], video_sha256=video_hash, clip_id=clip_id,
        )
        if annotations_path and file_sha256(annotations_path) != annotation_hash:
            raise ValueError(f"annotation changed during preparation: {clip_id}")
        evaluation = None
        evaluation_path = None
        if evaluation_dir is not None:
            directory = Path(evaluation_dir)
            choices = [directory / f"{clip_id}.json", directory / "video_eval" / f"{clip_id}.json"]
            existing = [path for path in choices if path.is_file()]
            if len(existing) != 1:
                raise ValueError(f"expected one pipeline evaluation JSON for {clip_id}")
            evaluation_path = existing[0]
        elif row.get("evaluation_json"):
            evaluation_path = _resolve(manifest.parent, row["evaluation_json"])
        if evaluation_path is not None:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if not isinstance(evaluation, dict):
                raise ValueError("pipeline evaluation JSON must be an object")
        reference = evaluation.get("scene_reference", {}) if evaluation else {}
        if not isinstance(reference, dict):
            reference = {}
        clips.append(ReviewClip(
            clip_id=clip_id,
            camera_id=camera_id,
            source_session_id=row.get("source_session_id") or payload.get("default_source_session_id"),
            video_path=str(video), video_sha256=video_hash,
            split=splits.get(clip_id, row.get("split", "unassigned")),
            annotations=annotations, annotation_sha256=annotation_hash,
            annotation_path=str(annotations_path) if annotations_path else None,
            evaluation=evaluation,
            evaluation_path=str(evaluation_path.resolve()) if evaluation_path else None,
            scene_profile_id=row.get("scene_profile_id") or reference.get("profile_id"),
            model_version_id=evaluation.get("model_version_id") if evaluation else None,
            kind=row.get("kind", "unknown"), tags=row.get("tags", []),
            description=row.get("description", ""),
            expected_detection_types=row.get("expected_detection_types", []),
            scene_reference_path=(
                str(_resolve(manifest.parent, row["scene_reference"]))
                if row.get("scene_reference") else None
            ),
            scene_reference_frame_index=row.get("scene_reference_frame"),
            **metadata,
        ))
    unknown = set(splits) - {clip.clip_id for clip in clips}
    if unknown:
        raise ValueError(f"split file contains unknown clip ids: {sorted(unknown)}")
    if file_sha256(manifest) != manifest_hash:
        raise ValueError("manifest changed during preparation")
    identity = {
        "manifest_sha256": manifest_hash,
        "clips": [{
            "id": clip.clip_id, "video_sha256": clip.video_sha256,
            "annotation_sha256": clip.annotation_sha256,
            "source_session_id": clip.source_session_id, "split": clip.split,
        } for clip in clips],
    }
    bundle_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    bundle = ReviewBundle(
        bundle_id=bundle_id, suite_id=payload.get("suite_id") or manifest.stem,
        manifest_sha256=manifest_hash, manifest_path=str(manifest), clips=clips,
    )
    require_verified_media(bundle)
    return bundle


def export_review_labels(
    bundle: ReviewBundle, output_dir: str | Path, *, overwrite: bool = False,
    split: str | None = None, protected_paths: Iterable[str | Path] = (),
) -> Path:
    """Write new labels and a manifest accepted by run_regression_suite.py.

    The source corpus is never edited. Callers select a split explicitly once
    assignments exist, preventing train/validation clips leaking into a test run.
    """
    require_verified_media(bundle)
    if split is not None and split not in _SPLITS:
        raise ValueError("unknown review dataset split")
    if split is None and any(clip.split != "unassigned" for clip in bundle.clips):
        raise ValueError("select an explicit split when exporting an assigned dataset")
    clips = [clip for clip in bundle.clips if split is None or clip.split == split]
    if not clips:
        raise ValueError("selected split contains no clips")
    if any(not clip.annotations.reviewed for clip in clips):
        raise ValueError("every exported clip requires completed human review")
    root = Path(output_dir).resolve()
    label_payloads: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    filenames: set[str] = set()
    for clip in clips:
        clip_id = _safe_id(clip.clip_id)
        if clip_id.casefold() in filenames:
            raise ValueError("clip ids collide on case-insensitive filesystems")
        filenames.add(clip_id.casefold())
        annotations = clip.annotations.model_dump(mode="json")
        annotations["review_positive_ranges"] = annotations["positive_ranges"]
        annotations["positive_ranges"] = _regression_episodes(clip)
        if clip.kind.strip().lower() in {"normal", "negative", "background"} and (
            annotations["positive_ranges"]
        ):
            raise ValueError(
                f"reviewed positive clip {clip.clip_id} cannot retain normal/negative kind"
            )
        annotations.update({
            "review_label_schema_version": _REVIEW_LABEL_SCHEMA,
            "review_event_semantics": "source-object-events",
            "evaluation_event_semantics": "anomaly-episodes",
            "source_video_sha256": clip.video_sha256,
            "source_clip_id": clip.clip_id,
            "review_bundle_id": bundle.bundle_id,
            "source_session_id": clip.source_session_id,
            "split": clip.split,
        })
        label_payloads[f"{clip_id}.labels.json"] = annotations
        row: dict[str, Any] = {
            "id": clip.clip_id, "kind": clip.kind, "camera_id": clip.camera_id,
            "source_session_id": clip.source_session_id,
            "video": str(Path(clip.video_path).resolve()),
            "split": clip.split, "tags": clip.tags, "description": clip.description,
            "expected_detection_types": clip.expected_detection_types,
            "scene_profile_id": clip.scene_profile_id,
        }
        if clip.scene_reference_path is not None:
            row["scene_reference"] = clip.scene_reference_path
        if clip.scene_reference_frame_index is not None:
            row["scene_reference_frame"] = clip.scene_reference_frame_index
        rows.append(row)
    generation = hashlib.sha256(json.dumps(
        label_payloads, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    generation_dir = root / "annotations" / generation
    for row in rows:
        row["annotations"] = f"annotations/{generation}/{row['id']}.labels.json"
    manifest_path = root / "manifest.json"
    manifest_payload = {
        "version": 1, "suite_id": bundle.suite_id,
        "review_bundle_id": bundle.bundle_id,
        "source_manifest_sha256": bundle.manifest_sha256,
        "description": "Human-reviewed labels; rerun the production pipeline for acceptance.",
        "clips": rows,
    }
    sources = review_source_paths(bundle)
    sources.update(Path(path).resolve() for path in protected_paths)
    destinations = [manifest_path, generation_dir, *(
        generation_dir / name for name in label_payloads
    )]
    for path in destinations:
        if (
            path.resolve() in sources or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("review export cannot replace sources or follow symlink outputs")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"review output already exists: {manifest_path}")

    # Published generations are immutable. Only the manifest is replaced, and
    # only after a complete generation is present. A failed overwrite therefore
    # cannot leave an old manifest pointing to partly replaced labels.
    generation_dir.parent.mkdir(parents=True, exist_ok=True)
    if not generation_dir.exists():
        staging = Path(tempfile.mkdtemp(prefix=".argus-review-", dir=generation_dir.parent))
        try:
            for name, payload in label_payloads.items():
                write_json(staging / name, payload)
            try:
                os.rename(staging, generation_dir)
            except FileExistsError:
                # A concurrent exporter may have published these exact bytes.
                pass
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    _verify_generation(generation_dir, label_payloads)
    write_json(manifest_path, manifest_payload, overwrite=overwrite)
    return manifest_path


def _verify_generation(directory: Path, payloads: dict[str, dict[str, Any]]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("review label generation is not a regular directory")
    if {path.name for path in directory.iterdir()} != set(payloads):
        raise ValueError("existing review label generation is incomplete or changed")
    for name, expected in payloads.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("review label generation contains an invalid output")
        actual = json.loads(path.read_text(encoding="utf-8"))
        if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
            expected, sort_keys=True, allow_nan=False,
        ):
            raise ValueError("existing review label generation is incomplete or changed")


def _regression_positive_ranges(clip: ReviewClip) -> list[ReviewRange]:
    """Include box-only truth in the range-only regression evaluator.

    Tags/ranges already covering the same label/frame remain authoritative, as
    in the CVAT importer. Geometry adds coverage only where no such tag exists.
    """
    result = list(clip.annotations.positive_ranges)
    ranges_by_label: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in result:
        ranges_by_label[item.label].append((item.start_frame, item.end_frame))
    for label, intervals in ranges_by_label.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        ranges_by_label[label] = merged
    grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
    for box in clip.annotations.boxes:
        intervals = ranges_by_label.get(box.label, [])
        position = bisect.bisect_right(intervals, (box.frame_index, clip.frame_count)) - 1
        if position >= 0 and intervals[position][1] >= box.frame_index:
            continue
        if box.track_id is not None:
            event_id = f"review-box-track-{box.track_id}"
        else:
            identity = json.dumps(box.model_dump(mode="json"), sort_keys=True)
            event_id = "review-box-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        grouped[box.label, event_id].add(box.frame_index)
    for (label, event_id), indices in sorted(grouped.items()):
        ordered = sorted(indices)
        start = previous = ordered[0]
        for index in ordered[1:]:
            if index == previous + 1:
                previous = index
                continue
            result.append(ReviewRange(
                start_frame=start, end_frame=previous, label=label, event_id=event_id,
            ))
            start = previous = index
        result.append(ReviewRange(
            start_frame=start, end_frame=previous, label=label, event_id=event_id,
        ))
    return result


def _regression_episodes(clip: ReviewClip) -> list[dict[str, Any]]:
    """Map object annotations to the runner's contiguous anomaly episodes.

    The runner cannot score two simultaneous objects as independent detection
    events. Union overlapping/adjacent coverage while retaining object truth in
    review_positive_ranges and boxes for a lossless future annotation round.
    """
    groups: list[list[ReviewRange]] = []
    end = -2
    for item in sorted(
        _regression_positive_ranges(clip), key=lambda item: (item.start_frame, item.end_frame),
    ):
        if not groups or item.start_frame > end + 1:
            groups.append([item])
            end = item.end_frame
        else:
            groups[-1].append(item)
            end = max(end, item.end_frame)
    episodes: list[dict[str, Any]] = []
    for group in groups:
        source_event_ids: set[str] = set()
        for item in group:
            identity = json.dumps(item.model_dump(mode="json"), sort_keys=True)
            source_event_ids.add(item.event_id or (
                "review-range-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
            ))
        start, end = group[0].start_frame, max(item.end_frame for item in group)
        identity = json.dumps({
            "source_video_sha256": clip.video_sha256, "start_frame": start,
            "end_frame": end, "source_event_ids": sorted(source_event_ids),
        }, sort_keys=True)
        episodes.append({
            "start_frame": start, "end_frame": end, "label": "anomaly",
            "event_id": "review-episode-" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "source_event_ids": sorted(source_event_ids),
            "source_labels": sorted({item.label for item in group}),
        })
    return episodes
