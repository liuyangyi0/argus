"""Lossless identity and frame-number contract for offline video review.

Argus and CVAT use zero-based inclusive frame indices; FiftyOne uses one-based
frame numbers. Exact presentation timestamps accompany the source frame index.
These review artifacts are not signed release evidence or model authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FrameTimestamp(ReviewModel):
    frame_index: int = Field(ge=0, strict=True)
    pts: int = Field(strict=True)
    time_base_num: int = Field(gt=0, strict=True)
    time_base_den: int = Field(gt=0, strict=True)
    timestamp_sec: float

    @model_validator(mode="after")
    def exact_timestamp(self) -> Self:
        seconds = float(self.pts * Fraction(self.time_base_num, self.time_base_den))
        if not math.isclose(seconds, self.timestamp_sec, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("timestamp_sec does not match exact PTS/time base")
        return self


class ReviewRange(ReviewModel):
    start_frame: int = Field(ge=0, strict=True)
    end_frame: int = Field(ge=0, strict=True)
    label: str = Field(min_length=1)
    event_id: str | None = None

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end_frame < self.start_frame:
            raise ValueError("frame ranges must be inclusive and ordered")
        if self.label != self.label.strip():
            raise ValueError("labels cannot contain surrounding whitespace")
        return self


class ReviewBox(ReviewModel):
    frame_index: int = Field(ge=0, strict=True)
    label: str = Field(min_length=1)
    xtl: float = Field(ge=0)
    ytl: float = Field(ge=0)
    xbr: float = Field(gt=0)
    ybr: float = Field(gt=0)
    track_id: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def rectangle(self) -> Self:
        if self.xbr <= self.xtl or self.ybr <= self.ytl:
            raise ValueError("boxes must have positive width and height")
        return self


class ReviewAnnotations(ReviewModel):
    positive_ranges: list[ReviewRange] = Field(default_factory=list)
    ignore_ranges: list[ReviewRange] = Field(default_factory=list)
    boxes: list[ReviewBox] = Field(default_factory=list)
    reviewed: bool = Field(default=False, strict=True)
    reviewed_by: str | None = None

    @model_validator(mode="after")
    def explicit_review(self) -> Self:
        if self.reviewed and not (self.reviewed_by and self.reviewed_by.strip()):
            raise ValueError("reviewed annotations require an explicit reviewed_by")
        return self


class ReviewClip(ReviewModel):
    clip_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    source_session_id: str | None = None
    video_path: str = Field(min_length=1)
    video_sha256: str
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0, strict=True)
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    split: Literal["unassigned", "train", "validation", "test"] = "unassigned"
    timestamps: list[FrameTimestamp]
    annotations: ReviewAnnotations = Field(default_factory=ReviewAnnotations)
    evaluation: dict[str, Any] | None = None
    evaluation_path: str | None = None
    scene_profile_id: str | None = None
    model_version_id: str | None = None
    annotation_sha256: str | None = None
    annotation_path: str | None = None
    kind: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    expected_detection_types: list[str] = Field(default_factory=list)
    scene_reference_path: str | None = None
    scene_reference_frame_index: int | None = Field(default=None, ge=0, strict=True)

    @field_validator("video_sha256", "annotation_sha256")
    @classmethod
    def valid_hash(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("expected lowercase SHA256")
        return value

    @field_validator("clip_id", "camera_id", "source_session_id")
    @classmethod
    def clean_identity(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError("identity must be nonempty without surrounding whitespace")
        return value

    @model_validator(mode="after")
    def source_bounds(self) -> Self:
        if len(self.timestamps) != self.frame_count:
            raise ValueError("a timestamp is required for every original decoded frame")
        previous: Fraction | None = None
        for index, stamp in enumerate(self.timestamps):
            if stamp.frame_index != index:
                raise ValueError("timestamps must enumerate consecutive zero-based frames")
            exact = stamp.pts * Fraction(stamp.time_base_num, stamp.time_base_den)
            if previous is not None and exact <= previous:
                raise ValueError("source presentation timestamps must be strictly increasing")
            previous = exact
        for item in self.annotations.positive_ranges + self.annotations.ignore_ranges:
            if item.end_frame >= self.frame_count:
                raise ValueError("annotation range is outside the original video")
        for box in self.annotations.boxes:
            if box.frame_index >= self.frame_count:
                raise ValueError("box frame is outside the original video")
            if box.xbr > self.width or box.ybr > self.height:
                raise ValueError("box coordinates are outside the original video")
        if (
            self.scene_reference_frame_index is not None
            and self.scene_reference_frame_index >= self.frame_count
        ):
            raise ValueError("scene reference frame is outside the original video")
        if self.evaluation is not None:
            self._validate_evaluation()
        return self

    def _validate_evaluation(self) -> None:
        evaluation = self.evaluation
        assert evaluation is not None
        if evaluation.get("schema_version") != "argus.pipeline-video-eval.v1":
            raise ValueError("review accepts only production pipeline video-eval results")
        if evaluation.get("evaluation_mode") != "pipeline":
            raise ValueError("detector-only scores cannot be attached as pipeline predictions")
        if evaluation.get("source_video_sha256") != self.video_sha256:
            raise ValueError("evaluation video SHA256 does not match review source")
        if evaluation.get("camera_id") != self.camera_id:
            raise ValueError("evaluation camera does not match review source")
        frames = evaluation.get("frames")
        if not isinstance(frames, list):
            raise ValueError("evaluation frames must be a list")
        seen: set[int] = set()
        for row in frames:
            if not isinstance(row, dict):
                raise ValueError("evaluation frames must be objects")
            index = row.get("frame_index")
            if type(index) is not int or not 0 <= index < self.frame_count or index in seen:
                raise ValueError("evaluation frame indices must be unique and in bounds")
            seen.add(index)
        # Reject NaN/Infinity even inside the otherwise extensible report.
        json.dumps(evaluation, allow_nan=False)


class ReviewBundle(ReviewModel):
    schema_version: Literal["argus.video-review.v1"] = "argus.video-review.v1"
    bundle_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    manifest_sha256: str
    manifest_path: str | None = None
    clips: list[ReviewClip] = Field(min_length=1)

    @field_validator("manifest_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("manifest_sha256 must be lowercase SHA256")
        return value

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        ids: set[str] = set()
        assignments: dict[tuple[str, ...], str] = {}
        for clip in self.clips:
            if clip.clip_id in ids:
                raise ValueError(f"duplicate clip id: {clip.clip_id}")
            ids.add(clip.clip_id)
            if clip.split != "unassigned" and not clip.source_session_id:
                raise ValueError("assigned splits require an explicit source_session_id")
            keys = [("video", clip.video_sha256)]
            if clip.source_session_id:
                keys.append(("session", clip.source_session_id))
            for key in keys:
                existing = assignments.setdefault(key, clip.split)
                if existing != clip.split:
                    raise ValueError("the same video or capture session crosses dataset splits")
        return self


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_verified_media(bundle: ReviewBundle) -> None:
    """Recheck every file before any external task/dataset mutation."""
    validated = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    for clip in validated.clips:
        video = Path(clip.video_path)
        if not video.is_file() or file_sha256(video) != clip.video_sha256:
            raise ValueError(f"review source is missing or changed: {clip.clip_id}")


def load_bundle(path: str | Path) -> ReviewBundle:
    return ReviewBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> None:
    """Write atomically; exclusive mode never replaces an existing output."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".argus-review-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # link is an atomic no-clobber publish on both supported filesystems.
            os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def review_source_paths(bundle: ReviewBundle) -> set[Path]:
    """Return source files that no review artifact may replace.

    A pipeline report is source evidence too, even when supplied by a manifest
    rather than the CLI's evaluation directory option.
    """
    sources = {
        Path(source).resolve()
        for clip in bundle.clips
        for source in (
            clip.video_path, clip.annotation_path, clip.scene_reference_path,
            clip.evaluation_path,
        )
        if source is not None
    }
    if bundle.manifest_path is not None:
        sources.add(Path(bundle.manifest_path).resolve())
    return sources


def save_bundle(bundle: ReviewBundle, path: str | Path, *, overwrite: bool = False) -> None:
    validated = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    if Path(path).resolve() in review_source_paths(validated):
        raise ValueError("a review bundle cannot replace its original source files")
    write_json(path, validated.model_dump(mode="json"), overwrite=overwrite)
