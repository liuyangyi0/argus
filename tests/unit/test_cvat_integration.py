"""CVAT contract regressions; fake SDK avoids network, decode and optional imports."""

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus.integrations import cvat
from argus.integrations.contracts import (
    FrameTimestamp,
    ReviewAnnotations,
    ReviewBox,
    ReviewBundle,
    ReviewClip,
    ReviewRange,
    file_sha256,
)


@pytest.fixture
def bundle(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"verified source bytes, not decoded in adapter unit tests")
    clip = ReviewClip(
        clip_id="clip-1",
        camera_id="camera-1",
        source_session_id="capture-session-1",
        video_path=str(video),
        video_sha256=file_sha256(video),
        fps=25,
        frame_count=8,
        width=100,
        height=80,
        split="test",
        timestamps=[
            FrameTimestamp(
                frame_index=index,
                pts=index,
                time_base_num=1,
                time_base_den=25,
                timestamp_sec=index / 25,
            )
            for index in range(8)
        ],
    )
    return ReviewBundle(
        bundle_id="review-1",
        suite_id="suite-1",
        manifest_sha256="a" * 64,
        clips=[clip],
    )


def label_schema(names=("anomaly", "ignore")):
    return [
        {
            "id": index + 1,
            "name": name,
            "type": "tag" if name == "ignore" else "any",
            "attributes": [
                {
                    "id": offset + index, "name": attribute,
                    "mutable": False, "input_type": "text",
                    "default_value": "", "values": [],
                }
                for offset, attribute in (
                    [(100, "argus_event_id"), (200, "argus_ignore_reason")]
                    if name == "ignore"
                    else [(100, "argus_event_id"), (300, "argus_source_geometry")]
                )
            ],
        }
        for index, name in enumerate(names)
    ]


def rectangle(frame, points=(0, 0, 10, 10), *, outside=False):
    return {
        "frame": frame,
        "type": "rectangle",
        "rotation": 0,
        "points": list(points),
        "outside": outside,
        "attributes": [],
    }


class FakeTask:
    def __init__(self, task_id, spec, clip):
        self.id = task_id
        self.name = spec.name
        self.mode = "interpolation"
        self.dimension = "2d"
        self.size = clip.frame_count
        self.status = "annotation"
        self.created_date = "2026-10-21T00:00:00Z"
        self.updated_date = self.created_date
        self.labels = label_schema(tuple(item["name"] for item in spec.labels))
        self.meta = {
            "size": clip.frame_count,
            "start_frame": 0,
            "stop_frame": clip.frame_count - 1,
            "frame_filter": "step=1",
            "deleted_frames": [],
            "included_frames": None,
            "frames": [{"name": "source.mp4", "width": clip.width, "height": clip.height}],
        }
        self.annotations = {"tags": [], "shapes": [], "tracks": []}
        self.upload_calls = []
        self.set_calls = []
        self.fail_upload = False

    def fetch(self):
        return self

    def get_meta(self):
        return deepcopy(self.meta)

    def get_labels(self):
        return deepcopy(self.labels)

    def get_annotations(self):
        return deepcopy(self.annotations)

    def upload_data(self, **kwargs):
        self.upload_calls.append(kwargs)
        if self.fail_upload:
            raise OSError("remote response may contain a secret; must not surface")

    def set_annotations(self, data):
        self.set_calls.append(data)
        self.annotations = deepcopy(data)


@pytest.fixture
def sdk(monkeypatch, bundle):
    tasks = {}
    calls = []

    def create(*, spec):
        calls.append(spec)
        task = FakeTask(len(tasks) + 1, spec, bundle.clips[0])
        tasks[task.id] = task
        return task

    client = SimpleNamespace(tasks=SimpleNamespace(create=create, retrieve=tasks.__getitem__))
    models = SimpleNamespace(
        TaskWriteRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        LabeledDataRequest=lambda **kwargs: kwargs,
    )

    @contextmanager
    def session(*args, **kwargs):
        yield client, models, SimpleNamespace(LOCAL="local")

    monkeypatch.setattr(cvat, "_client_session", session)
    return SimpleNamespace(client=client, tasks=tasks, calls=calls)


def test_tags_are_zero_based_and_ignore_is_explicit(bundle):
    result = cvat.annotations_from_cvat(
        bundle.clips[0],
        {"tags": [
            {"frame": 0, "label_id": 1, "attributes": [{"spec_id": 100, "value": "event-a"}]},
            {"frame": 1, "label_id": 1, "attributes": [{"spec_id": 100, "value": "event-a"}]},
            {"frame": 7, "label_id": 2},
        ]},
        label_schema(),
        reviewed_by="reviewer",
        completed=True,
    )
    assert result.positive_ranges == [ReviewRange(start_frame=0, end_frame=1, label="anomaly", event_id="event-a")]
    assert result.ignore_ranges == [ReviewRange(start_frame=7, end_frame=7, label="ignore")]
    assert result.reviewed and result.reviewed_by == "reviewer"


def test_tracks_interpolate_keyframes_and_respect_outside_intervals(bundle):
    result = cvat.annotations_from_cvat(
        bundle.clips[0],
        {"tracks": [{
            "id": 42,
            "frame": 0,
            "label_id": 1,
            "shapes": [
                rectangle(0),
                rectangle(3, (30, 0, 40, 10), outside=True),
                rectangle(5, (50, 0, 60, 10)),
            ],
        }]},
        label_schema(),
        reviewed_by="reviewer",
        completed=True,
    )
    assert [box.frame_index for box in result.boxes] == [0, 1, 2, 5, 6, 7]
    assert [box.xtl for box in result.boxes] == pytest.approx([0, 10, 20, 50, 50, 50])
    assert {box.track_id for box in result.boxes} == {42}
    assert [(item.start_frame, item.end_frame) for item in result.positive_ranges] == [(0, 2), (5, 7)]


@pytest.mark.parametrize("change", [
    {"type": "polygon", "points": [0, 0, 10, 0, 10, 10]},
    {"rotation": 15},
    {"points": [0, 0, 101, 10]},
    {"points": [0, 0, float("nan"), 10]},
    {"outside": "false"},
    {"frame": 8},
])
def test_unsupported_geometry_and_invalid_coordinates_are_not_dropped(bundle, change):
    shape = {**rectangle(0), "label_id": 1, **change}
    with pytest.raises(cvat.CVATIntegrationError):
        cvat.annotations_from_cvat(bundle.clips[0], {"shapes": [shape]}, label_schema(), reviewed_by="reviewer", completed=True)


def test_unknown_label_id_is_not_treated_as_normal(bundle):
    with pytest.raises(cvat.CVATIntegrationError, match="unknown label"):
        cvat.annotations_from_cvat(bundle.clips[0], {"tags": [{"frame": 0, "label_id": 999}]}, label_schema(), reviewed_by="reviewer", completed=True, confirm_normal=True)


@pytest.mark.parametrize("reviewer,completed", [("", True), ("   ", True), ("reviewer", False)])
def test_final_import_requires_completed_task_and_named_reviewer(bundle, reviewer, completed):
    with pytest.raises(cvat.CVATIntegrationError):
        cvat.annotations_from_cvat(bundle.clips[0], {}, label_schema(), reviewed_by=reviewer, completed=completed, confirm_normal=True)


def test_empty_annotations_need_separate_explicit_normal_confirmation(bundle):
    with pytest.raises(cvat.CVATIntegrationError, match="unknown"):
        cvat.annotations_from_cvat(bundle.clips[0], {}, label_schema(), reviewed_by="reviewer", completed=True)
    result = cvat.annotations_from_cvat(bundle.clips[0], {}, label_schema(), reviewed_by="reviewer", completed=True, confirm_normal=True)
    assert result.reviewed and not result.positive_ranges
    with pytest.raises(cvat.CVATIntegrationError, match="confirmed-normal"):
        cvat.annotations_from_cvat(bundle.clips[0], {"tags": [{"frame": 0, "label_id": 1}]}, label_schema(), reviewed_by="reviewer", completed=True, confirm_normal=True)


def test_duplicate_track_keyframes_fail_instead_of_double_counting(bundle):
    with pytest.raises(cvat.CVATIntegrationError, match="duplicate keyframes"):
        cvat.annotations_from_cvat(bundle.clips[0], {"tracks": [{"id": 1, "frame": 0, "label_id": 1, "shapes": [rectangle(0), rectangle(0)]}]}, label_schema(), reviewed_by="reviewer", completed=True)


def test_export_receipt_is_durable_and_retry_does_not_overwrite_human_work(bundle, sdk, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    receipt = cvat.export_bundle_to_cvat(bundle, url="http://localhost:8080/", receipt_path=receipt_path)
    task = sdk.tasks[1]
    assert task.upload_calls[0]["params"] == {"start_frame": 0, "stop_frame": 7, "frame_step": 1, "image_quality": 100}
    assert receipt["clips"]["clip-1"]["status"] == "ready"
    assert json.loads(receipt_path.read_text())["clips"]["clip-1"]["task_id"] == 1
    task.annotations = {"tags": [{"frame": 2, "label_id": 1}]}
    cvat.export_bundle_to_cvat(bundle, url="http://localhost:8080", receipt_path=receipt_path)
    assert len(sdk.calls) == 1
    assert len(task.upload_calls) == 1
    assert task.annotations == {"tags": [{"frame": 2, "label_id": 1}]}


def test_failed_upload_keeps_remote_identity_and_refuses_duplicate_creation(bundle, sdk, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    original_create = sdk.client.tasks.create

    def failing_create(**kwargs):
        task = original_create(**kwargs)
        task.fail_upload = True
        return task

    sdk.client.tasks.create = failing_create
    with pytest.raises(cvat.CVATIntegrationError, match="export failed") as error:
        cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    assert "secret" not in str(error.value)
    entry = json.loads(receipt_path.read_text())["clips"]["clip-1"]
    assert entry["status"] == "uploading" and entry["task_id"] == 1
    with pytest.raises(cvat.CVATIntegrationError, match="uncertain outcome"):
        cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    assert len(sdk.calls) == 1


@pytest.mark.parametrize("change", [
    {"frame_filter": "step=2"},
    {"deleted_frames": [1]},
    {"included_frames": [0, 2, 4, 6]},
    {"size": 7},
    {"start_frame": 1},
    {"stop_frame": 6},
    {"frames": [{"name": "other.mp4", "width": 100, "height": 80}]},
    {"frames": [{"name": "source.mp4", "width": 50, "height": 40}]},
])
def test_import_rejects_changed_source_selection(bundle, sdk, tmp_path, change):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    task = sdk.tasks[1]
    task.status = "completed"
    task.meta.update(change)
    with pytest.raises(cvat.CVATIntegrationError):
        cvat.import_bundle_from_cvat(bundle, url="http://localhost", receipt_path=receipt_path, reviewed_by="reviewer", confirmed_normal_clip_ids=["clip-1"])


@pytest.mark.parametrize("field,value", [("name", "renamed task"), ("created_date", "2027-01-01"), ("id", 999)])
def test_import_rejects_rebound_task_identity(bundle, sdk, tmp_path, field, value):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    task = sdk.tasks[1]
    task.status = "completed"
    setattr(task, field, value)
    with pytest.raises(cvat.CVATIntegrationError):
        cvat.import_bundle_from_cvat(bundle, url="http://localhost", receipt_path=receipt_path, reviewed_by="reviewer", confirmed_normal_clip_ids=["clip-1"])


def test_import_detects_local_video_tampering_before_contacting_cvat(bundle, sdk, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    Path(bundle.clips[0].video_path).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="source is missing or changed"):
        cvat.import_bundle_from_cvat(bundle, url="http://localhost", receipt_path=receipt_path, reviewed_by="reviewer")


def test_receipt_cannot_replace_verified_source_bytes(bundle, sdk):
    with pytest.raises(cvat.CVATIntegrationError, match="cannot replace"):
        cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=bundle.clips[0].video_path)
    assert not sdk.calls


def test_receipt_binding_includes_session_timestamp_and_split(bundle, sdk, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    changed = bundle.model_copy(update={"clips": [bundle.clips[0].model_copy(update={"source_session_id": "other-session"})]})
    with pytest.raises(cvat.CVATIntegrationError, match="identity mismatch"):
        cvat.import_bundle_from_cvat(changed, url="http://localhost", receipt_path=receipt_path, reviewed_by="reviewer")


def test_review_import_preserves_source_identity_and_leaves_original_unreviewed(bundle, sdk, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    task = sdk.tasks[1]
    task.status = "completed"
    task.annotations = {"shapes": [{**rectangle(7), "label_id": 1}]}
    reviewed = cvat.import_bundle_from_cvat(bundle, url="http://localhost", receipt_path=receipt_path, reviewed_by="reviewer")
    assert reviewed.clips[0].annotations.boxes[0].frame_index == 7
    assert reviewed.clips[0].timestamps == bundle.clips[0].timestamps
    assert reviewed.clips[0].split == "test"
    assert reviewed.clips[0].video_sha256 == bundle.clips[0].video_sha256
    assert not bundle.clips[0].annotations.reviewed


def test_seeded_tracks_have_outside_keyframes_at_annotation_gaps(bundle):
    clip = bundle.clips[0].model_copy(update={"annotations": ReviewAnnotations(boxes=[
        ReviewBox(frame_index=0, label="anomaly", xtl=0, ytl=0, xbr=10, ybr=10, track_id=5),
        ReviewBox(frame_index=3, label="anomaly", xtl=30, ytl=0, xbr=40, ybr=10, track_id=5),
    ])})
    seed = cvat._seed_annotations(clip, label_schema())
    assert not seed["shapes"]
    track = seed["tracks"][0]
    assert [(shape["frame"], shape["outside"]) for shape in track["shapes"]] == [(0, False), (1, True), (3, False), (4, True)]
    track["id"] = 50
    result = cvat.annotations_from_cvat(clip, seed, label_schema(), reviewed_by="reviewer", completed=True)
    assert [box.frame_index for box in result.boxes] == [0, 3]


def test_custom_ignore_reason_survives_cvat_tag_roundtrip(bundle):
    clip = bundle.clips[0].model_copy(update={"annotations": ReviewAnnotations(ignore_ranges=[
        ReviewRange(start_frame=1, end_frame=3, label="camera_adjustment"),
        ReviewRange(start_frame=4, end_frame=5, label="blocked_lens"),
    ])})
    seed = cvat._seed_annotations(clip, label_schema())
    result = cvat.annotations_from_cvat(clip, seed, label_schema(), reviewed_by="reviewer", completed=True)
    assert [(item.start_frame, item.end_frame, item.label) for item in result.ignore_ranges] == [(4, 5, "blocked_lens"), (1, 3, "camera_adjustment")]
    assert not result.positive_ranges


def test_positive_tags_keep_adjacent_and_overlapping_events_distinct_with_boxes(bundle):
    clip = bundle.clips[0].model_copy(update={"annotations": ReviewAnnotations(
        positive_ranges=[
            ReviewRange(start_frame=0, end_frame=2, label="anomaly"),
            ReviewRange(start_frame=3, end_frame=5, label="anomaly"),
            ReviewRange(start_frame=2, end_frame=4, label="anomaly", event_id="overlap"),
        ],
        boxes=[
            ReviewBox(frame_index=index, label="anomaly", xtl=0, ytl=0, xbr=10, ybr=10, track_id=7)
            for index in range(6)
        ] + [ReviewBox(frame_index=2, label="anomaly", xtl=20, ytl=20, xbr=30, ybr=30)],
    )})
    seed = cvat._seed_annotations(clip, label_schema())
    for index, track in enumerate(seed["tracks"]):
        track["id"] = 1000 + index
    result = cvat.annotations_from_cvat(clip, seed, label_schema(), reviewed_by="reviewer", completed=True)
    assert len(result.positive_ranges) == 3
    assert {(item.start_frame, item.end_frame) for item in result.positive_ranges} == {(0, 2), (3, 5), (2, 4)}
    assert len({item.event_id for item in result.positive_ranges}) == 3
    assert len(result.boxes) == 7


def test_ignore_boxes_fail_before_creating_any_remote_task(bundle, sdk, tmp_path):
    clip = bundle.clips[0].model_copy(update={"annotations": ReviewAnnotations(boxes=[
        ReviewBox(frame_index=0, label="ignore", xtl=0, ytl=0, xbr=10, ybr=10),
    ])})
    changed = bundle.model_copy(update={"clips": [clip]})
    with pytest.raises(cvat.CVATIntegrationError, match="Ignore must be"):
        cvat.export_bundle_to_cvat(changed, url="http://localhost", receipt_path=tmp_path / "receipt.json")
    assert not sdk.calls


def test_uploaded_bytes_mutating_mid_upload_never_become_ready(bundle, sdk, tmp_path):
    original_create = sdk.client.tasks.create

    def mutating_create(**kwargs):
        task = original_create(**kwargs)
        original_upload = task.upload_data

        def mutate(**upload_kwargs):
            original_upload(**upload_kwargs)
            Path(bundle.clips[0].video_path).write_bytes(b"changed during upload")

        task.upload_data = mutate
        return task

    sdk.client.tasks.create = mutating_create
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(cvat.CVATIntegrationError, match="changed during CVAT upload"):
        cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    assert json.loads(receipt_path.read_text())["clips"]["clip-1"]["status"] == "uploading"


def test_receipt_cannot_overwrite_original_manifest(bundle, sdk, tmp_path):
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_text("original")
    changed = bundle.model_copy(update={"manifest_path": str(manifest_path)})
    with pytest.raises(cvat.CVATIntegrationError, match="cannot replace"):
        cvat.export_bundle_to_cvat(changed, url="http://localhost", receipt_path=manifest_path)
    assert manifest_path.read_text() == "original"


def test_receipt_cannot_overwrite_manifest_attached_evaluation(bundle, sdk, tmp_path):
    evaluation_path = tmp_path / "pipeline-report.json"
    evaluation_path.write_text("original source evidence")
    clip = bundle.clips[0].model_copy(update={"evaluation_path": str(evaluation_path)})
    changed = bundle.model_copy(update={"clips": [clip]})
    with pytest.raises(cvat.CVATIntegrationError, match="cannot replace"):
        cvat.export_bundle_to_cvat(
            changed, url="http://localhost", receipt_path=evaluation_path,
        )
    assert evaluation_path.read_text() == "original source evidence"
    assert not sdk.calls


@pytest.mark.parametrize("url", ["localhost:8080", "http://user:password@localhost", "https://host?token=secret", "https://host/#secret"])
def test_url_never_embeds_secrets(url):
    with pytest.raises(cvat.CVATIntegrationError):
        cvat._server_url(url)


@pytest.mark.parametrize("collection", ["intervals", "new_annotation_kind"])
def test_unsupported_annotation_collections_cannot_become_confirmed_normal(bundle, collection):
    with pytest.raises(cvat.CVATIntegrationError, match="Unsupported CVAT"):
        cvat.annotations_from_cvat(
            bundle.clips[0],
            {collection: [{"label_id": 1, "start": 0, "stop": 3}]},
            label_schema(), reviewed_by="reviewer", completed=True, confirm_normal=True,
        )


def test_independent_track_is_not_swallowed_by_existing_same_label_tags(bundle):
    clip = bundle.clips[0].model_copy(update={"annotations": ReviewAnnotations(
        positive_ranges=[ReviewRange(
            start_frame=0, end_frame=5, label="anomaly", event_id="source-event",
        )],
        boxes=[ReviewBox(
            frame_index=2, label="anomaly", xtl=0, ytl=0, xbr=10, ybr=10,
        )],
    )})
    annotations = cvat._seed_annotations(clip, label_schema())
    annotations["tracks"].append({
        "id": 42, "frame": 2, "label_id": 1, "attributes": [],
        "shapes": [rectangle(2), rectangle(4, outside=True)],
    })
    result = cvat.annotations_from_cvat(
        clip, annotations, label_schema(), reviewed_by="reviewer", completed=True,
    )
    assert {(item.start_frame, item.end_frame, item.event_id) for item in result.positive_ranges} == {
        (0, 5, "source-event"), (2, 3, "cvat-track-42"),
    }
    assert len(result.boxes) == 3


def test_independent_same_frame_shapes_are_separate_events(bundle):
    result = cvat.annotations_from_cvat(
        bundle.clips[0], {"shapes": [
            {**rectangle(2), "id": 10, "label_id": 1},
            {**rectangle(2, (20, 0, 30, 10)), "id": 11, "label_id": 1},
        ]}, label_schema(), reviewed_by="reviewer", completed=True,
    )
    assert {item.event_id for item in result.positive_ranges} == {
        "cvat-shape-10", "cvat-shape-11",
    }


def test_generated_track_identity_cannot_merge_with_explicit_tag_event(bundle):
    result = cvat.annotations_from_cvat(
        bundle.clips[0], {
            "tags": [{
                "frame": 2, "label_id": 1,
                "attributes": [{"spec_id": 100, "value": "cvat-track-42"}],
            }],
            "tracks": [{
                "id": 42, "frame": 2, "label_id": 1,
                "shapes": [rectangle(2), rectangle(3, outside=True)],
            }],
        }, label_schema(), reviewed_by="reviewer", completed=True,
    )
    assert {item.event_id for item in result.positive_ranges} == {
        "cvat-track-42", "cvat-track-42-generated",
    }


@pytest.mark.parametrize("track_id", [None, 5])
@pytest.mark.parametrize("clear_cloned_marker", [False, True])
def test_cloned_source_geometry_cannot_reuse_source_event_identity(
    bundle, track_id, clear_cloned_marker,
):
    clip = bundle.clips[0].model_copy(update={"annotations": ReviewAnnotations(
        positive_ranges=[ReviewRange(
            start_frame=0, end_frame=5, label="anomaly", event_id="source-event",
        )],
        boxes=[ReviewBox(
            frame_index=2, label="anomaly", xtl=0, ytl=0, xbr=10, ybr=10,
            track_id=track_id,
        )],
    )})
    annotations = cvat._seed_annotations(clip, label_schema())
    collection = "shapes" if track_id is None else "tracks"
    original = annotations[collection][0]
    original["id"] = 50
    cloned = deepcopy(original)
    cloned["id"] = 51
    if clear_cloned_marker:
        cloned["attributes"] = [
            attribute for attribute in cloned["attributes"]
            if attribute["spec_id"] != 300
        ]
    annotations[collection].append(cloned)
    with pytest.raises(cvat.CVATIntegrationError, match="Cloned source geometry"):
        cvat.annotations_from_cvat(
            clip, annotations, label_schema(), reviewed_by="reviewer", completed=True,
        )


def test_unmarked_geometry_may_explicitly_share_one_event_identity(bundle):
    result = cvat.annotations_from_cvat(
        bundle.clips[0], {"shapes": [
            {
                **rectangle(index), "id": 50 + index, "label_id": 1,
                "attributes": [{"spec_id": 100, "value": "one-manual-event"}],
            }
            for index in (1, 2)
        ]}, label_schema(), reviewed_by="reviewer", completed=True,
    )
    assert result.positive_ranges == [ReviewRange(
        start_frame=1, end_frame=2, label="anomaly", event_id="one-manual-event",
    )]


@pytest.mark.parametrize("change", [
    {"mutable": True}, {"input_type": "select"},
    {"default_value": "another-event"}, {"values": ["another-event"]},
])
def test_attribute_semantic_changes_invalidate_task_binding(bundle, sdk, tmp_path, change):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    task = sdk.tasks[1]
    task.status = "completed"
    task.labels[0]["attributes"][0].update(change)
    with pytest.raises(cvat.CVATIntegrationError, match="attributes differ"):
        cvat.import_bundle_from_cvat(
            bundle, url="http://localhost", receipt_path=receipt_path,
            reviewed_by="reviewer", confirmed_normal_clip_ids=["clip-1"],
        )


def test_label_type_change_invalidates_task_binding(bundle, sdk, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    sdk.tasks[1].labels[0]["type"] = "polygon"
    with pytest.raises(cvat.CVATIntegrationError, match="attributes differ"):
        cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)


def test_changed_seed_annotations_cannot_reuse_old_upload_receipt(bundle, sdk, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    clip = bundle.clips[0].model_copy(update={"annotations": ReviewAnnotations(
        positive_ranges=[ReviewRange(start_frame=0, end_frame=3, label="anomaly")],
    )})
    changed = bundle.model_copy(update={"clips": [clip]})
    with pytest.raises(cvat.CVATIntegrationError, match="identity mismatch"):
        cvat.export_bundle_to_cvat(changed, url="http://localhost", receipt_path=receipt_path)
    assert len(sdk.calls) == 1


def test_task_edit_during_annotation_read_is_not_marked_reviewed(bundle, sdk, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    cvat.export_bundle_to_cvat(bundle, url="http://localhost", receipt_path=receipt_path)
    task = sdk.tasks[1]
    task.status = "completed"
    original = task.get_annotations

    def edit_during_read():
        result = original()
        task.updated_date = "2026-10-21T00:00:01Z"
        return result

    task.get_annotations = edit_during_read
    with pytest.raises(cvat.CVATIntegrationError, match="changed during review"):
        cvat.import_bundle_from_cvat(
            bundle, url="http://localhost", receipt_path=receipt_path,
            reviewed_by="reviewer", confirmed_normal_clip_ids=["clip-1"],
        )
