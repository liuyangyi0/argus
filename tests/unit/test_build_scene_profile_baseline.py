from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.preprocessing.scene_guard import (
    SceneGuardResult,
    SceneMatchMetrics,
    SceneState,
)
from argus.validation.strict_video_decoder import (
    StrictDecodedFrame,
    StrictFrameTimestamp,
)
from scripts import build_scene_profile_baseline as builder


_CAMERA_ID = "camera-test-01"
_SCENE_PROFILE_ID = "scene-profile-test-01"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(path: Path, value: tuple[int, int, int] = (11, 22, 33)) -> np.ndarray:
    frame = np.full((4, 5, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    return frame


def _source(path: Path) -> None:
    path.write_bytes(b"recorded-video-source")


def _annotation(
    path: Path,
    video: Path,
    *,
    session_id: str = "session-01",
    camera_id: str = _CAMERA_ID,
    positive_ranges: list[dict[str, object]] | None = None,
    ignore_ranges: list[dict[str, object]] | None = None,
) -> None:
    path.write_text(
        json.dumps({
            "schema_version": "argus.normal-video-review.v1",
            "source_video_sha256": _sha256(video),
            "source_session_id": session_id,
            "camera_id": camera_id,
            "review_complete": True,
            "reviewed_by": "test-reviewer",
            "reviewed_at": "2026-07-14T09:30:00+08:00",
            "positive_ranges": positive_ranges or [],
            "ignore_ranges": ignore_ranges or [],
        }),
        encoding="utf-8",
    )


class _FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        fps: float = 10.0,
        reported_frame_count: float = -1.0,
    ) -> None:
        self._frames = [frame.copy() for frame in frames]
        self._fps = fps
        self._reported_frame_count = (
            float(len(frames))
            if reported_frame_count < 0.0
            else reported_frame_count
        )
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API compatibility
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._reported_frame_count
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index].copy()
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _patch_video(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    frames: list[np.ndarray],
    *,
    reported_frame_count: float = -1.0,
) -> None:
    expected = str(path.resolve())

    def make_capture(value: str) -> _FakeCapture:
        assert value == expected
        return _FakeCapture(
            frames,
            reported_frame_count=reported_frame_count,
        )

    monkeypatch.setattr(builder.cv2, "VideoCapture", make_capture)


def _patch_videos(
    monkeypatch: pytest.MonkeyPatch,
    sources: dict[Path, list[np.ndarray]],
) -> None:
    frames_by_path = {
        str(path.resolve()): frames for path, frames in sources.items()
    }

    def make_capture(value: str) -> _FakeCapture:
        assert value in frames_by_path
        return _FakeCapture(frames_by_path[value])

    monkeypatch.setattr(builder.cv2, "VideoCapture", make_capture)


def _result(
    state: SceneState,
    *,
    homography: np.ndarray | None = None,
) -> SceneGuardResult:
    return SceneGuardResult(
        state=state,
        raw_state=state,
        reason=f"test-{state.value}",
        metrics=SceneMatchMetrics(
            reference_available=True,
            transform_reasonable=state in {SceneState.STABLE, SceneState.ALIGNABLE},
        ),
        homography=homography,
    )


def _patch_guard(
    monkeypatch: pytest.MonkeyPatch,
    results: list[SceneGuardResult],
) -> list[np.ndarray]:
    remaining = list(results)
    evaluated: list[np.ndarray] = []

    class FakeGuard:
        def __init__(self, reference_frame, config) -> None:
            assert reference_frame.shape == (4, 5, 3)
            assert config is not None

        def evaluate(self, frame, *, return_aligned):
            assert return_aligned is False
            evaluated.append(frame.copy())
            return remaining.pop(0)

    monkeypatch.setattr(builder, "SceneGuard", FakeGuard)
    return evaluated


def test_positive_and_ignore_frames_are_excluded_before_scene_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    frames = [
        np.full_like(reference, 40),
        np.full_like(reference, 80),
        np.full_like(reference, 120),
    ]
    _patch_video(monkeypatch, video, frames)
    evaluated = _patch_guard(monkeypatch, [_result(SceneState.STABLE)])
    annotations = tmp_path / "clip.labels.json"
    _annotation(
        annotations,
        video,
        positive_ranges=[{"start_frame": 0, "end_frame": 0}],
        ignore_ranges=[{"start_frame": 1, "end_frame": 1}],
    )
    output = tmp_path / "output"

    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=output,
        sample_every=1,
        max_images=10,
        annotations=[annotations],
    )

    assert len(evaluated) == 1
    assert np.array_equal(evaluated[0], frames[2])
    assert manifest["selection"]["accepted_images"] == 1
    assert manifest["selection"]["accepted_state_counts"] == {"stable": 1}
    assert manifest["selection"]["rejection_counts"]["annotation_positive"] == 1
    assert manifest["selection"]["rejection_counts"]["annotation_ignore"] == 1
    assert manifest["validation_only"] is True
    assert manifest["operator_approved"] is False
    assert manifest["active_profile_modified"] is False
    assert manifest["corpus_evidence_complete"] is True
    assert manifest["training_ready"] is False
    assert manifest["release_eligible"] is False
    assert manifest["samples"][0]["split_role"] == "diagnostic_only"


def test_alignable_frame_is_really_warped_and_reference_fills_footprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    frame = np.full_like(reference, 200)
    _patch_video(monkeypatch, video, [frame])
    homography = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    _patch_guard(
        monkeypatch,
        [_result(SceneState.ALIGNABLE, homography=homography)],
    )

    output = tmp_path / "output"
    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=output,
        sample_every=1,
        max_images=1,
        allow_alignable=True,
    )

    saved = cv2.imread(str(output / manifest["samples"][0]["path"]))
    assert saved is not None
    assert np.array_equal(saved[:, 0], reference[:, 0])
    assert np.all(saved[:, 1:] == 200)
    assert manifest["samples"][0]["normalized"] is True
    assert manifest["training_ready"] is False
    assert manifest["diagnostic_samples_available"] is True
    assert manifest["dataset_status"] == "diagnostic_only_not_training_ready"
    assert (
        manifest["samples"][0]["normalization"]
        == "homography_to_reference_with_reference_fill"
    )


def test_alignable_normalization_matches_production_fractional_warp_contract(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path, value=(7, 17, 27))
    frame = np.arange(reference.size, dtype=np.uint8).reshape(reference.shape)
    homography = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    expected = cv2.warpPerspective(
        frame,
        homography,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    validity = cv2.warpPerspective(
        np.full(frame.shape[:2], 255, dtype=np.uint8),
        homography,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    expected[validity == 0] = reference[validity == 0]

    normalized = builder._normalize_alignable_frame(
        frame,
        reference,
        homography,
    )

    assert np.array_equal(normalized, expected)


def test_alignable_frame_is_rejected_without_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    _patch_video(monkeypatch, video, [reference])
    _patch_guard(
        monkeypatch,
        [_result(SceneState.ALIGNABLE, homography=np.eye(3))],
    )

    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=tmp_path / "output",
        sample_every=1,
        max_images=1,
    )

    assert manifest["selection"]["accepted_images"] == 0
    assert manifest["selection"]["rejection_counts"] == {
        "alignable_not_allowed": 1,
    }


def test_moved_frame_is_excluded_and_dataset_is_not_training_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    _patch_video(monkeypatch, video, [reference])
    _patch_guard(monkeypatch, [_result(SceneState.MOVED)])

    output = tmp_path / "output"
    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=output,
        sample_every=1,
        max_images=1,
    )

    assert manifest["selection"]["accepted_images"] == 0
    assert manifest["selection"]["confirmed_state_counts"] == {"moved": 1}
    assert manifest["selection"]["rejected_state_counts"] == {"moved": 1}
    assert manifest["selection"]["rejection_counts"]["scene_guard_moved"] == 1
    assert manifest["dataset_status"] == "rejected_no_samples"
    assert manifest["training_ready"] is False
    assert list((output / "images").iterdir()) == []


def test_manifest_binds_source_reference_and_output_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    annotations = tmp_path / "clip.labels.json"
    _annotation(annotations, video)
    _patch_video(monkeypatch, video, [reference])
    _patch_guard(monkeypatch, [_result(SceneState.STABLE)])

    output = tmp_path / "output"
    returned = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=output,
        sample_every=1,
        max_images=2,
        annotations=[annotations],
    )
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert persisted == returned
    assert persisted["schema_version"] == "argus.scene-profile-baseline.v2"
    assert persisted["camera_id"] == _CAMERA_ID
    assert persisted["scene_profile_id"] == _SCENE_PROFILE_ID
    assert persisted["reference"]["sha256"] == _sha256(reference_path)
    assert persisted["sources"][0]["sha256"] == _sha256(video)
    assert persisted["sources"][0]["annotation_sha256"] == _sha256(annotations)
    sample_path = output / persisted["samples"][0]["path"]
    assert persisted["samples"][0]["sha256"] == _sha256(sample_path)
    assert persisted["samples"][0]["source_sha256"] == _sha256(video)
    assert persisted["training_ready"] is False
    assert persisted["release_eligible"] is False
    assert persisted["training_readiness_scope"] == "offline_training_input_only"
    assert persisted["split_contract"]["single_session_policy"] == "diagnostic_only"
    assert persisted["decoder_evidence"]["sufficient_for_model_release"] is False
    assert (
        persisted["decoder_evidence"]["decode_error_distinguishable_from_clean_eof"]
        is False
    )
    assert persisted["decoder_evidence"]["variable_frame_rate_timestamps_preserved"] is False
    assert (
        persisted["normalization_contract"]["contract_id"]
        == "argus.detection-pipeline.scene-guard-warp.v1"
    )
    assert (
        persisted["normalization_contract"]["source_footprint_mask"][
            "invalid_pixel_policy"
        ]
        == "copy_exact_reference_pixel"
    )
    assert (
        persisted["parameters"]["deterministic_sample_rule"]
        == "zero_based_frame_index_mod_sample_every_equals_zero"
    )


def test_max_images_truncation_is_explicit_and_invalidates_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    annotations = tmp_path / "clip.labels.json"
    _annotation(annotations, video)
    _patch_video(monkeypatch, video, [reference, reference])
    _patch_guard(
        monkeypatch,
        [_result(SceneState.STABLE), _result(SceneState.STABLE)],
    )

    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=tmp_path / "output",
        sample_every=1,
        max_images=1,
        annotations=[annotations],
    )

    assert manifest["selection"]["geometrically_accepted"] == 2
    assert manifest["selection"]["accepted_images"] == 1
    assert manifest["selection"]["truncated_by_max_images"] is True
    assert manifest["selection"]["rejection_counts"]["max_images_limit"] == 1
    assert manifest["selection_evidence_complete"] is False
    assert manifest["corpus_evidence_complete"] is False
    assert manifest["training_ready"] is False
    assert manifest["diagnostic_samples_available"] is True
    assert manifest["samples"][0]["frame_index"] == 0


def test_two_reviewed_source_sessions_are_grouped_without_frame_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    train_video = tmp_path / "train-session.mp4"
    calibration_video = tmp_path / "calibration-session.mp4"
    train_video.write_bytes(b"recorded-train-session")
    calibration_video.write_bytes(b"recorded-calibration-session")
    _patch_videos(
        monkeypatch,
        {
            train_video: [reference, reference],
            calibration_video: [reference, reference],
        },
    )
    _patch_guard(
        monkeypatch,
        [_result(SceneState.STABLE) for _ in range(4)],
    )
    train_annotation = tmp_path / "train.labels.json"
    calibration_annotation = tmp_path / "calibration.labels.json"
    _annotation(train_annotation, train_video, session_id="session-a")
    _annotation(
        calibration_annotation,
        calibration_video,
        session_id="session-b",
    )

    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[train_video, calibration_video],
        reference=reference_path,
        output_dir=tmp_path / "output",
        sample_every=1,
        max_images=10,
        min_training_images=2,
        min_calibration_images=2,
        min_accepted_time_span_seconds=0.19,
        annotations=[train_annotation, calibration_annotation],
    )

    assert manifest["training_ready"] is True
    assert manifest["dataset_status"] == "offline_training_input_ready"
    assert manifest["release_eligible"] is False
    assert manifest["split_contract"]["declared_independent_session_count"] == 2
    assert manifest["split_contract"]["source_counts"] == {
        "train": 1,
        "calibration": 1,
        "diagnostic_only": 0,
    }
    assert manifest["split_contract"]["sample_counts"] == {
        "train": 2,
        "calibration": 2,
        "diagnostic_only": 0,
    }
    assert [item["split_role"] for item in manifest["samples"]] == [
        "train",
        "train",
        "calibration",
        "calibration",
    ]
    assert manifest["sources"][0]["split_role"] == "train"
    assert manifest["sources"][1]["split_role"] == "calibration"


def test_two_files_declared_as_one_session_remain_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    first = tmp_path / "part-1.mp4"
    second = tmp_path / "part-2.mp4"
    first.write_bytes(b"session-part-1")
    second.write_bytes(b"session-part-2")
    _patch_videos(monkeypatch, {first: [reference], second: [reference]})
    _patch_guard(
        monkeypatch,
        [_result(SceneState.STABLE), _result(SceneState.STABLE)],
    )
    first_annotation = tmp_path / "part-1.labels.json"
    second_annotation = tmp_path / "part-2.labels.json"
    _annotation(first_annotation, first, session_id="same-session")
    _annotation(second_annotation, second, session_id="same-session")

    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[first, second],
        reference=reference_path,
        output_dir=tmp_path / "output",
        sample_every=1,
        max_images=10,
        annotations=[first_annotation, second_annotation],
    )

    assert manifest["training_ready"] is False
    assert manifest["split_contract"]["declared_independent_session_count"] == 1
    assert manifest["split_contract"]["source_counts"] == {
        "train": 0,
        "calibration": 0,
        "diagnostic_only": 2,
    }
    assert {item["split_role"] for item in manifest["samples"]} == {
        "diagnostic_only"
    }


def test_empty_annotation_can_only_create_diagnostic_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    annotation = tmp_path / "clip.labels.json"
    annotation.write_text("{}\n", encoding="utf-8")
    _patch_video(monkeypatch, video, [reference, reference])
    _patch_guard(
        monkeypatch,
        [_result(SceneState.STABLE), _result(SceneState.STABLE)],
    )

    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=tmp_path / "output",
        sample_every=1,
        max_images=10,
        min_training_images=2,
        min_calibration_images=2,
        min_accepted_time_span_seconds=0.1,
        annotations=[annotation],
    )

    review = manifest["sources"][0]["annotation_review_evidence"]
    assert manifest["annotation_exclusion_evidence_complete"] is False
    assert manifest["training_ready"] is False
    assert manifest["dataset_status"] == "diagnostic_only_not_training_ready"
    assert review["complete"] is False
    assert "source_video_sha256_missing_or_invalid" in review["reasons"]
    assert "review_complete_not_literal_true" in review["reasons"]


def test_annotation_identity_and_review_mismatch_are_not_trusted(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "clip.labels.json"
    annotation.write_text(
        json.dumps({
            "schema_version": "argus.normal-video-review.v1",
            "source_video_sha256": "a" * 64,
            "source_session_id": "session-01",
            "camera_id": "different-camera",
            "review_complete": True,
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-07-14T09:30:00",
            "positive_ranges": [],
            "ignore_ranges": [],
        }),
        encoding="utf-8",
    )

    loaded = builder._load_annotations_strict(
        annotation,
        source_video_sha256="b" * 64,
        camera_id=_CAMERA_ID,
    )

    evidence = loaded["review_evidence"]
    assert evidence["complete"] is False
    assert evidence["source_video_sha256_matches"] is False
    assert evidence["camera_id_matches"] is False
    assert evidence["reviewed_at_has_timezone"] is False
    assert evidence["reasons"] == [
        "source_video_sha256_mismatch",
        "camera_id_mismatch",
        "reviewed_at_missing_invalid_or_without_timezone",
    ]


@pytest.mark.parametrize(
    "range_payload, error",
    [
        ({"start_frame": -1, "end_frame": 1}, "cannot be negative"),
        ({"start_frame": 2, "end_frame": 1}, "precedes"),
        ({"start_frame": float("inf"), "end_frame": 1}, "must be finite"),
        ({"start_frame": 0}, "must include start_frame and end_frame"),
        (
            {"start_frame": 0, "end_frame": 1, "start_sec": 0.0},
            "ambiguous seconds/legacy coordinates",
        ),
    ],
)
def test_malformed_or_ambiguous_annotation_ranges_are_rejected(
    tmp_path: Path,
    range_payload: dict[str, object],
    error: str,
) -> None:
    annotation = tmp_path / "clip.labels.json"
    annotation.write_text(
        json.dumps({
            "schema_version": "argus.normal-video-review.v1",
            "source_video_sha256": "a" * 64,
            "source_session_id": "session-01",
            "camera_id": _CAMERA_ID,
            "review_complete": True,
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-07-14T09:30:00+08:00",
            "positive_ranges": [range_payload],
            "ignore_ranges": [],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        builder._load_annotations_strict(
            annotation,
            source_video_sha256="a" * 64,
            camera_id=_CAMERA_ID,
        )


@pytest.mark.parametrize("reported_frame_count", [0.0, 1.0])
def test_unknown_or_mismatched_inventory_can_only_produce_diagnostic_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported_frame_count: float,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    annotations = tmp_path / "clip.labels.json"
    _annotation(annotations, video)
    _patch_video(
        monkeypatch,
        video,
        [reference, reference],
        reported_frame_count=reported_frame_count,
    )
    _patch_guard(
        monkeypatch,
        [_result(SceneState.STABLE), _result(SceneState.STABLE)],
    )

    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=tmp_path / "output",
        sample_every=1,
        max_images=10,
        annotations=[annotations],
    )

    assert manifest["selection"]["accepted_images"] == 2
    assert manifest["source_inventory_evidence_complete"] is False
    assert manifest["corpus_evidence_complete"] is False
    assert manifest["training_ready"] is False
    assert manifest["diagnostic_samples_available"] is True


def test_different_paths_with_identical_video_content_are_rejected(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    _reference(reference_path)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    _source(first)
    _source(second)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="duplicate video content"):
        builder.build_scene_profile_baseline(
            camera_id=_CAMERA_ID,
            scene_profile_id=_SCENE_PROFILE_ID,
            videos=[first, second],
            reference=reference_path,
            output_dir=output,
            sample_every=1,
            max_images=10,
        )

    assert not output.exists()


def test_non_empty_output_directory_is_rejected_without_modification(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("owned by user", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        builder.build_scene_profile_baseline(
            camera_id=_CAMERA_ID,
            scene_profile_id=_SCENE_PROFILE_ID,
            videos=[video],
            reference=reference_path,
            output_dir=output,
            sample_every=1,
            max_images=1,
        )

    assert sentinel.read_text(encoding="utf-8") == "owned by user"


def test_cli_requires_explicit_camera_and_scene_profile_id(tmp_path: Path) -> None:
    common = [
        "--video",
        str(tmp_path / "clip.mp4"),
        "--reference",
        str(tmp_path / "reference.png"),
        "--output-dir",
        str(tmp_path / "output"),
        "--sample-every",
        "1",
        "--max-images",
        "100",
    ]

    with pytest.raises(SystemExit):
        builder.parse_args(common)

    args = builder.parse_args([
        "--camera-id",
        _CAMERA_ID,
        "--scene-profile-id",
        _SCENE_PROFILE_ID,
        *common,
    ])
    assert args.camera_id == _CAMERA_ID
    assert args.scene_profile_id == _SCENE_PROFILE_ID
    assert args.min_training_images == 100
    assert args.min_calibration_images == 20
    assert args.min_accepted_time_span_seconds == 300.0
    assert args.strict_release_decoder is False

    strict_args = builder.parse_args([
        "--camera-id",
        _CAMERA_ID,
        "--scene-profile-id",
        _SCENE_PROFILE_ID,
        *common,
        "--strict-release-decoder",
    ])
    assert strict_args.strict_release_decoder is True


@pytest.mark.parametrize(
    ("camera_id", "scene_profile_id", "error"),
    [
        ("", _SCENE_PROFILE_ID, "camera_id"),
        (_CAMERA_ID, " ", "scene_profile_id"),
        (" camera", _SCENE_PROFILE_ID, "camera_id"),
    ],
)
def test_builder_rejects_ambiguous_scene_identity_before_writing(
    tmp_path: Path,
    camera_id: str,
    scene_profile_id: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        builder.build_scene_profile_baseline(
            camera_id=camera_id,
            scene_profile_id=scene_profile_id,
            videos=[tmp_path / "missing.mp4"],
            reference=tmp_path / "missing.png",
            output_dir=tmp_path / "output",
            sample_every=1,
            max_images=100,
        )
    assert not (tmp_path / "output").exists()


def test_failed_build_removes_staging_and_never_publishes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    _patch_video(monkeypatch, video, [reference])
    _patch_guard(monkeypatch, [_result(SceneState.STABLE)])
    monkeypatch.setattr(builder.cv2, "imwrite", lambda *_args, **_kwargs: False)
    output = tmp_path / "output"

    with pytest.raises(OSError, match="could not write sample"):
        builder.build_scene_profile_baseline(
            camera_id=_CAMERA_ID,
            scene_profile_id=_SCENE_PROFILE_ID,
            videos=[video],
            reference=reference_path,
            output_dir=output,
            sample_every=1,
            max_images=1,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output.*.staging")) == []


def test_strict_release_decoder_emits_v3_hash_bound_pts_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference = _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)
    frames = [reference.copy(), np.full_like(reference, 77)]
    timestamps = (
        StrictFrameTimestamp(
            frame_index=0,
            pts=100,
            pts_time_seconds=0.1,
            time_base="1/1000",
            duration_pts=100,
            duration_seconds=0.1,
        ),
        StrictFrameTimestamp(
            frame_index=1,
            pts=250,
            pts_time_seconds=0.25,
            time_base="1/1000",
            duration_pts=150,
            duration_seconds=0.15,
        ),
    )

    class FakeStrictDecoder:
        def __init__(self, source: Path, *, expected_sha256: str) -> None:
            assert Path(source).resolve() == video.resolve()
            assert expected_sha256 == _sha256(video)
            self.timestamps = timestamps
            self.nominal_frame_rate = 8.0
            self.probed_frame_count = 2
            self.declared_frame_count = 2
            self._index = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __next__(self) -> StrictDecodedFrame:
            if self._index >= len(frames):
                raise StopIteration
            index = self._index
            self._index += 1
            return StrictDecodedFrame(
                frame_index=index,
                frame=frames[index].copy(),
                timestamp=timestamps[index],
            )

        def evidence(self) -> dict[str, object]:
            assert self._index == len(frames)
            return {
                "schema_version": "argus.strict-video-decoder-evidence.v1",
                "backend": "ffprobe+ffmpeg-rawvideo",
                "source_sha256_matches": True,
                "decode_error_distinguishable_from_clean_eof": True,
                "variable_frame_rate_timestamps_preserved": True,
                "complete_decode_verified": True,
                "source_exhausted": True,
                "truncated": False,
                "sufficient_for_model_release": True,
            }

    monkeypatch.setattr(builder, "StrictVideoDecoder", FakeStrictDecoder)
    _patch_guard(
        monkeypatch,
        [_result(SceneState.STABLE), _result(SceneState.STABLE)],
    )

    output = tmp_path / "output"
    manifest = builder.build_scene_profile_baseline(
        camera_id=_CAMERA_ID,
        scene_profile_id=_SCENE_PROFILE_ID,
        videos=[video],
        reference=reference_path,
        output_dir=output,
        sample_every=1,
        max_images=10,
        strict_release_decoder=True,
    )

    assert manifest["schema_version"] == "argus.scene-profile-baseline.v3"
    assert manifest["decoder_evidence"] == {
        "backend": "ffprobe+ffmpeg-rawvideo",
        "strict_release_decoder_requested": True,
        "all_sources_strictly_audited": True,
        "source_count": 1,
        "read_termination_signal": (
            "ffmpeg_clean_zero_exit_after_complete_raw_frame_stream"
        ),
        "decoded_frame_count_cross_check": (
            "ffmpeg_raw_frames_equal_ffprobe_show_frames_inventory"
        ),
        "source_sha256_checked_before_and_after_decode": True,
        "decode_error_distinguishable_from_clean_eof": True,
        "variable_frame_rate_timestamps_preserved": True,
        "time_basis": "per_frame_pts_times_stream_time_base",
        "timestamp_evidence_format": (
            "sha256_bound_jsonl_one_record_per_decoded_frame"
        ),
        "sufficient_for_model_release": True,
    }
    assert manifest["release_eligible"] is False
    assert manifest["operator_approved"] is False
    assert manifest["active_profile_modified"] is False
    assert manifest["samples"][0]["pts"] == 100
    assert manifest["samples"][0]["time_base"] == "1/1000"
    assert manifest["samples"][1]["time_seconds"] == 0.25

    timestamp_evidence = manifest["sources"][0]["frame_timestamp_evidence"]
    timestamp_path = output / timestamp_evidence["path"]
    assert timestamp_evidence["frame_count"] == 2
    assert timestamp_evidence["sha256"] == _sha256(timestamp_path)
    records = [
        json.loads(line)
        for line in timestamp_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["pts"] for item in records] == [100, 250]
    assert {item["time_base"] for item in records} == {"1/1000"}


def test_strict_release_decoder_failure_never_publishes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    _reference(reference_path)
    video = tmp_path / "clip.mp4"
    _source(video)

    class FailingStrictDecoder:
        def __init__(self, source: Path, *, expected_sha256: str) -> None:
            self.nominal_frame_rate = 10.0
            self.probed_frame_count = 1
            self.declared_frame_count = 1

        def __enter__(self):
            raise OSError("ffmpeg decoder error")

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(builder, "StrictVideoDecoder", FailingStrictDecoder)
    _patch_guard(monkeypatch, [])
    output = tmp_path / "output"

    with pytest.raises(OSError, match="ffmpeg decoder error"):
        builder.build_scene_profile_baseline(
            camera_id=_CAMERA_ID,
            scene_profile_id=_SCENE_PROFILE_ID,
            videos=[video],
            reference=reference_path,
            output_dir=output,
            sample_every=1,
            max_images=10,
            strict_release_decoder=True,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output.*.staging")) == []
