"""Source-level regression coverage for offline review identity and frame bounds."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from argus.integrations.contracts import (
    FrameTimestamp,
    ReviewAnnotations,
    ReviewBundle,
    ReviewClip,
    ReviewRange,
    file_sha256,
    load_bundle,
    require_verified_media,
    save_bundle,
)


def make_clip(tmp_path: Path, **changes) -> ReviewClip:
    video = tmp_path / "source.mp4"
    if not video.exists():
        video.write_bytes(b"identity-only fixture; no decoder is invoked")
    values = {
        "clip_id": "clip_01", "camera_id": "cam_01", "video_path": str(video),
        "video_sha256": file_sha256(video), "fps": 30.0, "frame_count": 3,
        "width": 640, "height": 480,
        "timestamps": [FrameTimestamp(
            frame_index=index, pts=100 + index, time_base_num=1, time_base_den=30,
            timestamp_sec=(100 + index) / 30,
        ) for index in range(3)],
    }
    values.update(changes)
    return ReviewClip(**values)


def bundle(*clips: ReviewClip) -> ReviewBundle:
    return ReviewBundle(
        bundle_id="review-01", suite_id="suite", manifest_sha256="a" * 64,
        clips=list(clips),
    )


def test_bundle_round_trip_keeps_exact_source_pts_and_zero_based_ranges(tmp_path):
    clip = make_clip(tmp_path, annotations=ReviewAnnotations(
        positive_ranges=[ReviewRange(start_frame=0, end_frame=2, label="object")],
    ))
    output = tmp_path / "review.json"
    save_bundle(bundle(clip), output)
    loaded = load_bundle(output)
    assert loaded.clips[0].timestamps[0].pts == 100
    assert loaded.clips[0].annotations.positive_ranges[0].start_frame == 0
    assert loaded.clips[0].annotations.reviewed is False
    assert json.loads(output.read_text())["schema_version"] == "argus.video-review.v1"


def test_save_refuses_to_clobber_and_media_changes_are_rejected(tmp_path):
    review = bundle(make_clip(tmp_path))
    output = tmp_path / "review.json"
    save_bundle(review, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_bundle(review, output)
    assert output.read_bytes() == original
    Path(review.clips[0].video_path).write_bytes(b"replaced source")
    with pytest.raises(ValueError, match="missing or changed"):
        require_verified_media(review)


def test_rejects_manufactured_timestamps_and_missing_source_frames(tmp_path):
    with pytest.raises(ValidationError, match="PTS/time base"):
        FrameTimestamp(frame_index=0, pts=1, time_base_num=1, time_base_den=30,
                       timestamp_sec=1.0)
    with pytest.raises(ValidationError, match="every original"):
        make_clip(tmp_path, timestamps=[])


def test_rejects_non_increasing_timestamps(tmp_path):
    stamps = make_clip(tmp_path).timestamps
    stamps[1] = stamps[0].model_copy(update={"frame_index": 1})
    with pytest.raises(ValidationError, match="strictly increasing"):
        make_clip(tmp_path, timestamps=stamps)


def test_frame_range_cannot_be_clipped_silently(tmp_path):
    with pytest.raises(ValidationError, match="outside the original video"):
        make_clip(tmp_path, annotations=ReviewAnnotations(positive_ranges=[
            ReviewRange(start_frame=0, end_frame=3, label="anomaly"),
        ]))


def test_reviewed_requires_real_reviewer():
    with pytest.raises(ValidationError, match="reviewed_by"):
        ReviewAnnotations(reviewed=True, reviewed_by=" ")


@pytest.mark.parametrize("second_split", ["test", "unassigned"])
def test_capture_session_cannot_cross_splits(tmp_path, second_split):
    first = make_clip(tmp_path, split="train", source_session_id="capture-day-01")
    second = make_clip(tmp_path, clip_id="clip_02", split=second_split,
                       source_session_id="capture-day-01")
    with pytest.raises(ValidationError, match="crosses dataset splits"):
        bundle(first, second)


def test_assigned_split_without_capture_session_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="source_session_id"):
        bundle(make_clip(tmp_path, split="test"))


def test_predictions_are_bound_to_actual_source_and_camera(tmp_path):
    clip = make_clip(tmp_path)
    evaluation = {
        "schema_version": "argus.pipeline-video-eval.v1", "evaluation_mode": "pipeline",
        "source_video_sha256": "b" * 64, "camera_id": clip.camera_id, "frames": [],
    }
    with pytest.raises(ValidationError, match="SHA256"):
        make_clip(tmp_path, evaluation=evaluation)
    evaluation["source_video_sha256"] = clip.video_sha256
    evaluation["camera_id"] = "different-camera"
    with pytest.raises(ValidationError, match="camera"):
        make_clip(tmp_path, evaluation=evaluation)


def test_predictions_cannot_duplicate_frames_or_include_nan(tmp_path):
    clip = make_clip(tmp_path)
    evaluation = {
        "schema_version": "argus.pipeline-video-eval.v1", "evaluation_mode": "pipeline",
        "source_video_sha256": clip.video_sha256, "camera_id": clip.camera_id,
        "frames": [{"frame_index": 0}, {"frame_index": 0}],
    }
    with pytest.raises(ValidationError, match="unique"):
        make_clip(tmp_path, evaluation=evaluation)
    evaluation["frames"] = [{"frame_index": 0, "score": float("nan")}]
    with pytest.raises(ValidationError):
        make_clip(tmp_path, evaluation=evaluation)


def test_validation_rechecks_shallow_mutations_before_external_writes(tmp_path):
    review = bundle(make_clip(tmp_path))
    review.clips[0].timestamps.pop()
    with pytest.raises(ValidationError):
        require_verified_media(review)


def test_save_bundle_cannot_overwrite_manifest_supplied_pipeline_report(tmp_path):
    report = tmp_path / "pipeline.json"
    report.write_text('{"source": "pipeline evidence"}', encoding="utf-8")
    original = report.read_bytes()
    review = bundle(make_clip(tmp_path, evaluation_path=str(report)))
    with pytest.raises(ValueError, match="original source"):
        save_bundle(review, report, overwrite=True)
    assert report.read_bytes() == original
