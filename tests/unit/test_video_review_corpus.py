"""Regression contracts for review preparation and independent label export."""

import json
from pathlib import Path

import pytest

from argus.integrations import corpus
from argus.integrations.contracts import FrameTimestamp, ReviewAnnotations, ReviewRange


def fixture_manifest(tmp_path: Path, monkeypatch, *, annotations=None, **row_changes):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video fixture")
    row = {"id": "clip_01", "camera_id": "cam_01", "kind": "static_foreign",
           "video": video.name, "source_session_id": "session-01"}
    if annotations is not None:
        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps(annotations), encoding="utf-8")
        row["annotations"] = labels.name
    row.update(row_changes)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(json.dumps({"version": 1, "suite_id": "review-suite",
                                    "clips": [row]}), encoding="utf-8")
    # VFR fixture: original PTS are authoritative; nominal 30 FPS is not.
    metadata = {
        "fps": 30.0, "width": 640, "height": 480, "frame_count": 3,
        "timestamps": [FrameTimestamp(
            frame_index=i, pts=pts, time_base_num=1, time_base_den=1000,
            timestamp_sec=pts / 1000,
        ) for i, pts in enumerate([5000, 5040, 5150])],
    }
    monkeypatch.setattr(corpus, "_inspect_video", lambda _: metadata)
    return manifest


def test_prepare_preserves_source_identity_and_does_not_approve_legacy_labels(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch, annotations={
        "positive_ranges": [{"start_frame": 0, "end_frame": 2, "label": "book"}],
    })
    review = corpus.prepare_review_bundle(manifest)
    clip = review.clips[0]
    assert clip.annotations.reviewed is False
    assert clip.annotation_path == str(tmp_path / "labels.json")
    assert clip.annotation_sha256 is not None
    assert clip.timestamps[0].pts == 5000
    assert clip.source_session_id == "session-01"
    assert clip.split == "unassigned"


def test_seconds_annotations_map_through_exact_relative_pts(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch, annotations={
        "positive_ranges": [{"start_sec": 0.03, "end_sec": 0.15, "label": "object"}],
    })
    event = corpus.prepare_review_bundle(manifest).clips[0].annotations.positive_ranges[0]
    assert (event.start_frame, event.end_frame) == (1, 2)


def test_unlabeled_ignore_range_keeps_ignore_semantics(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch, annotations={
        "ignore_ranges": [{"start_frame": 0, "end_frame": 1}],
    })
    annotations = corpus.prepare_review_bundle(manifest).clips[0].annotations
    assert annotations.ignore_ranges[0].label == "ignore"
    assert annotations.positive_ranges == []


def test_missing_annotations_never_imply_reviewed_normal(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch)
    review = corpus.prepare_review_bundle(manifest)
    with pytest.raises(ValueError, match="human review"):
        corpus.export_review_labels(review, tmp_path / "reviewed")


def test_export_creates_compatible_manifest_and_preserves_original_labels(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch, annotations={"positive_ranges": []})
    source_labels = (tmp_path / "labels.json").read_bytes()
    review = corpus.prepare_review_bundle(manifest)
    clip = review.clips[0].model_copy(update={"annotations": ReviewAnnotations(
        reviewed=True, reviewed_by="engineer-01", positive_ranges=[
            ReviewRange(start_frame=1, end_frame=2, label="static_foreign", event_id="event-01"),
        ],
    )})
    review = review.model_copy(update={"clips": [clip]})
    generated = corpus.export_review_labels(review, tmp_path / "reviewed")
    payload = json.loads(generated.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    row = payload["clips"][0]
    labels = json.loads((generated.parent / row["annotations"]).read_text(encoding="utf-8"))
    assert labels["positive_ranges"][0]["source_event_ids"] == ["event-01"]
    assert labels["review_positive_ranges"][0]["event_id"] == "event-01"
    assert labels["positive_ranges"][0]["start_frame"] == 1
    assert row["video"] == str(tmp_path / "video.mp4")
    assert row["source_session_id"] == "session-01"
    assert (tmp_path / "labels.json").read_bytes() == source_labels
    with pytest.raises(FileExistsError):
        corpus.export_review_labels(review, tmp_path / "reviewed")


def test_export_of_assigned_dataset_requires_explicit_split(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch, split="test", annotations={
        "reviewed": True, "reviewed_by": "reviewer", "positive_ranges": [],
    })
    review = corpus.prepare_review_bundle(manifest)
    with pytest.raises(ValueError, match="explicit split"):
        corpus.export_review_labels(review, tmp_path / "out")
    assert corpus.export_review_labels(review, tmp_path / "out", split="test").is_file()


def test_prepare_rejects_partial_or_wrong_source_evaluations(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch)
    directory = tmp_path / "eval"
    directory.mkdir()
    with pytest.raises(ValueError, match="expected one"):
        corpus.prepare_review_bundle(manifest, evaluation_dir=directory)
    (directory / "clip_01.json").write_text(json.dumps({
        "schema_version": "argus.pipeline-video-eval.v1", "evaluation_mode": "pipeline",
        "source_video_sha256": "0" * 64, "camera_id": "cam_01", "frames": [],
    }))
    with pytest.raises(ValueError, match="SHA256"):
        corpus.prepare_review_bundle(manifest, evaluation_dir=directory)


def test_save_and_export_cannot_replace_source_reference(tmp_path, monkeypatch):
    reference = tmp_path / "manifest.json"
    reference.write_bytes(b"immutable reference")
    manifest = fixture_manifest(tmp_path, monkeypatch, scene_reference=str(reference), annotations={
        "reviewed": True, "reviewed_by": "reviewer", "positive_ranges": [],
    })
    review = corpus.prepare_review_bundle(manifest)
    with pytest.raises(ValueError, match="cannot replace"):
        corpus.export_review_labels(review, tmp_path, overwrite=True)
    assert reference.read_bytes() == b"immutable reference"


@pytest.mark.parametrize("clip_id", ["../escape", "CON", "valid.", "a/b", "a\\b"])
def test_manifest_ids_cannot_escape_export_directory(tmp_path, monkeypatch, clip_id):
    manifest = fixture_manifest(tmp_path, monkeypatch, id=clip_id)
    with pytest.raises(ValueError, match="clip id"):
        corpus.prepare_review_bundle(manifest)


@pytest.mark.parametrize("point, expected", [
    ({"frame": 1}, 1), ({"time_sec": 0.03}, 1),
    ({"start_frame": 2}, 2), ({"start_sec": 0.14}, 2),
])
def test_prepare_accepts_legacy_single_frame_annotations(tmp_path, monkeypatch, point, expected):
    manifest = fixture_manifest(tmp_path, monkeypatch, annotations={"events": [point]})
    event = corpus.prepare_review_bundle(manifest).clips[0].annotations.positive_ranges[0]
    assert (event.start_frame, event.end_frame) == (expected, expected)


def test_prepare_rejects_ambiguous_legacy_point_and_range(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch, annotations={
        "events": [{"frame": 1, "start_frame": 0, "end_frame": 2}],
    })
    with pytest.raises(ValueError, match="cannot mix"):
        corpus.prepare_review_bundle(manifest)


def test_prepare_tracks_manifest_evaluation_source_for_output_protection(tmp_path, monkeypatch):
    evaluation = tmp_path / "report.json"
    manifest = fixture_manifest(tmp_path, monkeypatch, evaluation_json=evaluation.name)
    from argus.integrations.contracts import file_sha256, save_bundle

    evaluation.write_text(json.dumps({
        "schema_version": "argus.pipeline-video-eval.v1", "evaluation_mode": "pipeline",
        "source_video_sha256": file_sha256(tmp_path / "video.mp4"),
        "camera_id": "cam_01", "frames": [],
    }), encoding="utf-8")
    original = evaluation.read_bytes()
    review = corpus.prepare_review_bundle(manifest)
    assert review.clips[0].evaluation_path == str(evaluation.resolve())
    with pytest.raises(ValueError, match="original source"):
        save_bundle(review, evaluation, overwrite=True)
    assert evaluation.read_bytes() == original


def _reviewed_bundle(tmp_path, monkeypatch, **annotations):
    manifest = fixture_manifest(tmp_path, monkeypatch, annotations={
        "reviewed": True, "reviewed_by": "reviewer", **annotations,
    })
    return corpus.prepare_review_bundle(manifest)


def test_box_only_review_exports_truth_for_range_only_regression_reader(tmp_path, monkeypatch):
    review = _reviewed_bundle(tmp_path, monkeypatch, boxes=[{
        "frame_index": 1, "label": "foreign", "xtl": 10, "ytl": 20,
        "xbr": 30, "ybr": 40, "track_id": 7,
    }])
    output = corpus.export_review_labels(review, tmp_path / "out")
    row = json.loads(output.read_text())["clips"][0]
    labels = json.loads((output.parent / row["annotations"]).read_text())
    assert len(labels["positive_ranges"]) == 1
    event = labels["positive_ranges"][0]
    assert (event["start_frame"], event["end_frame"]) == (1, 1)
    assert event["source_event_ids"] == ["review-box-track-7"]
    assert event["source_labels"] == ["foreign"]
    assert labels["boxes"][0]["track_id"] == 7


def test_box_geometry_does_not_duplicate_existing_same_label_event(tmp_path, monkeypatch):
    review = _reviewed_bundle(tmp_path, monkeypatch,
        positive_ranges=[{"start_frame": 0, "end_frame": 2, "label": "foreign"}],
        boxes=[{"frame_index": 1, "label": "foreign", "xtl": 10, "ytl": 20,
                "xbr": 30, "ybr": 40, "track_id": 7}],
    )
    output = corpus.export_review_labels(review, tmp_path / "out")
    row = json.loads(output.read_text())["clips"][0]
    labels = json.loads((output.parent / row["annotations"]).read_text())
    assert len(labels["positive_ranges"]) == 1


def test_positive_review_cannot_be_exported_as_normal_kind(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path, monkeypatch, kind="normal", annotations={
        "reviewed": True, "reviewed_by": "reviewer",
        "positive_ranges": [{"start_frame": 1, "end_frame": 1}],
    })
    review = corpus.prepare_review_bundle(manifest)
    with pytest.raises(ValueError, match="cannot retain"):
        corpus.export_review_labels(review, tmp_path / "out")


def test_failed_overwrite_keeps_previous_manifest_and_all_referenced_labels(tmp_path, monkeypatch):
    review = _reviewed_bundle(tmp_path, monkeypatch)
    review = review.model_copy(update={"clips": [
        review.clips[0], review.clips[0].model_copy(update={"clip_id": "clip_02"}),
    ]})
    output = corpus.export_review_labels(review, tmp_path / "out")
    original_manifest = output.read_bytes()
    previous_labels = {
        output.parent / row["annotations"]:
        (output.parent / row["annotations"]).read_bytes()
        for row in json.loads(original_manifest)["clips"]
    }
    updated = review.model_copy(update={"clips": [
        clip.model_copy(update={"annotations": ReviewAnnotations(
            reviewed=True, reviewed_by="second-reviewer", positive_ranges=[
                ReviewRange(start_frame=1, end_frame=2, label="foreign"),
            ],
        )}) for clip in review.clips
    ]})
    real_write = corpus.write_json

    def fail_second_label(path, payload, **kwargs):
        if Path(path).name == "clip_02.labels.json":
            raise OSError("injected disk failure")
        return real_write(path, payload, **kwargs)

    monkeypatch.setattr(corpus, "write_json", fail_second_label)
    with pytest.raises(OSError, match="disk failure"):
        corpus.export_review_labels(updated, output.parent, overwrite=True)
    assert output.read_bytes() == original_manifest
    assert all(path.read_bytes() == content for path, content in previous_labels.items())


def test_label_generation_is_immutable_even_with_overwrite(tmp_path, monkeypatch):
    review = _reviewed_bundle(tmp_path, monkeypatch)
    output = corpus.export_review_labels(review, tmp_path / "out")
    row = json.loads(output.read_text())["clips"][0]
    labels = output.parent / row["annotations"]
    labels.write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or changed"):
        corpus.export_review_labels(review, output.parent, overwrite=True)
    assert labels.read_text() == '{}'


def test_export_protects_input_bundle_even_when_overwrite_is_enabled(tmp_path, monkeypatch):
    review = _reviewed_bundle(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    input_bundle = output_dir / "manifest.json"
    input_bundle.write_text('{"bundle": "input"}', encoding="utf-8")
    original = input_bundle.read_bytes()
    with pytest.raises(ValueError, match="cannot replace"):
        corpus.export_review_labels(
            review, output_dir, overwrite=True, protected_paths=[input_bundle],
        )
    assert input_bundle.read_bytes() == original


def test_overlapping_object_events_export_as_episodes_and_round_trip_losslessly(tmp_path, monkeypatch):
    review = _reviewed_bundle(tmp_path, monkeypatch, positive_ranges=[
        {"start_frame": 0, "end_frame": 1, "label": "book", "event_id": "object-A"},
        {"start_frame": 1, "end_frame": 2, "label": "tool", "event_id": "object-B"},
    ])
    output = corpus.export_review_labels(review, tmp_path / "out")
    row = json.loads(output.read_text())["clips"][0]
    labels = json.loads((output.parent / row["annotations"]).read_text())
    assert labels["evaluation_event_semantics"] == "anomaly-episodes"
    assert labels["review_event_semantics"] == "source-object-events"
    assert len(labels["positive_ranges"]) == 1
    event = labels["positive_ranges"][0]
    assert (event["start_frame"], event["end_frame"]) == (0, 2)
    assert event["source_event_ids"] == ["object-A", "object-B"]
    assert labels["review_positive_ranges"] == [
        event.model_dump(mode="json") for event in review.clips[0].annotations.positive_ranges
    ]
    restored = corpus.prepare_review_bundle(output)
    assert restored.clips[0].annotations == review.clips[0].annotations


def test_new_review_labels_cannot_be_attached_to_a_different_video(tmp_path, monkeypatch):
    review = _reviewed_bundle(tmp_path, monkeypatch)
    output = corpus.export_review_labels(review, tmp_path / "out")
    Path(review.clips[0].video_path).write_bytes(b"a different video with the same filename")
    with pytest.raises(ValueError, match="source video SHA256"):
        corpus.prepare_review_bundle(output)


def test_relative_seconds_subtract_exact_pts_before_float_conversion():
    origin = 2**62
    timestamps = [FrameTimestamp(
        frame_index=index, pts=origin + delta, time_base_num=1, time_base_den=1000,
        timestamp_sec=(origin + delta) / 1000,
    ) for index, delta in enumerate([0, 40, 150])]
    result = corpus._range_from_source({"start_sec": 0.03, "end_sec": 0.15}, timestamps)
    assert (result.start_frame, result.end_frame) == (1, 2)
