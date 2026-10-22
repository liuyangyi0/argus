"""Offline adapter contracts; fake FiftyOne never starts a DB or opens media."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus.integrations import fiftyone as adapter
from argus.integrations.contracts import ReviewBundle, file_sha256
from argus.integrations.scene_comparison import SceneComparisonReview


class _Document(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self._dirty_fields = set()

    def __setitem__(self, name, value):
        self._dirty_fields.add(name)
        super().__setitem__(name, value)

    def get_field(self, name):
        return self.get(name)


class _Frames:
    """Model public save/reload cache semantics, not an unbounded plain dict."""

    def __init__(self, sample):
        self.sample = sample
        self.persisted = {}
        self.pending = {}
        self.peak_cache = 0
        self.batch_sizes = []

    def __getitem__(self, number):
        if number not in self.pending:
            self.pending[number] = deepcopy(
                self.persisted.get(number, _Document(frame_number=number))
            )
            self.pending[number]._dirty_fields.clear()
            self.peak_cache = max(self.peak_cache, len(self.pending))
        return self.pending[number]

    def __iter__(self):
        return iter(set(self.persisted) | set(self.pending))

    def save(self):
        assert self.sample.dataset is not None, "new frames require an attached sample"
        self.batch_sizes.append(len(self.pending))
        # Existing FiftyOne Frame documents save dirty field updates, not a
        # full cached-document replacement. This matters when a reviewer saves
        # different fields in another App/Python client during an import.
        for number, frame in self.pending.items():
            stored = deepcopy(self.persisted.get(number, _Document(frame_number=number)))
            for field in frame._dirty_fields:
                stored[field] = deepcopy(frame[field])
            stored._dirty_fields.clear()
            self.persisted[number] = stored
            frame._dirty_fields.clear()
        self.pending.clear()

    def reload(self):
        self.pending.clear()


class _Sample(_Document):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.frames = _Frames(self)
        self.metadata = None
        self.dataset = None
        self.saves = 0

    def save(self):
        self.saves += 1
        if (
            self.dataset is not None and self.dataset.fail_frame_save
            and self.get("argus_bundle_id") == "review-2"
        ):
            self.dataset.fail_frame_save = False
            raise RuntimeError("injected frame write failure")
        if self.dataset is not None:
            self.frames.save()


class _View:
    def __init__(self, sample_filter, frame_filter=None):
        self.sample_filter = sample_filter
        self.frame_filter = frame_filter

    def match_frames(self, frame_filter):
        assert isinstance(frame_filter, _Expression), "frame filtering must use scoped ViewField"
        return _View(self.sample_filter, frame_filter)


class _Expression:
    def __init__(self, operator, *values):
        self.operator = operator
        self.values = values

    def __and__(self, other):
        return _Expression("$and", self, other)

    def __eq__(self, other):
        return _Expression("$eq", self, other)

    def __invert__(self):
        return _Expression("$not", self)

    def if_else(self, true_expr, false_expr):
        return _Expression("$cond", self, true_expr, false_expr)

    def is_in(self, values):
        return _Expression("$in", self, values)

    def to_mongo(self, prefix=None):
        values = [
            value.to_mongo(prefix=prefix) if isinstance(value, (_Expression, _ViewField)) else value
            for value in self.values
        ]
        if self.operator == "$cond":
            return {"$cond": dict(zip(("if", "then", "else"), values))}
        return {self.operator: values}

    def evaluate(self, frame):
        values = [
            value.evaluate(frame) if isinstance(value, (_Expression, _ViewField)) else value
            for value in self.values
        ]
        if self.operator == "$eq":
            return values[0] == values[1]
        if self.operator == "$and":
            return all(values)
        if self.operator == "$not":
            return not values[0]
        if self.operator == "$in":
            return values[0] in values[1]
        if self.operator == "$gt" and values[1] is None:
            return values[0] is not None
        if self.operator == "$cond":
            return values[1] if values[0] else values[2]
        raise AssertionError(self.operator)


class _ViewField:
    def __init__(self, field):
        self.field = field

    def __eq__(self, value):
        return _Expression("$eq", self, value)

    def is_in(self, values):
        return _Expression("$in", self, values)

    def exists(self):
        return _Expression("$gt", self, None)

    def to_mongo(self, prefix=None):
        return f"{prefix}.{self.field}" if prefix else f"${self.field}"

    def evaluate(self, frame):
        return frame.get(self.field)


class _Dataset:
    def __init__(self, name, persistent=True):
        self.name = name
        self.persistent = persistent
        self.info = {}
        self.samples = []
        self.views = {}
        self.schema = {}
        self.frame_schema = {}
        self.fail_frame_save = False
        self.fail_saved_view = None
        self.media_type = None

    def iter_samples(self):
        return iter(self.samples)

    def add_sample_field(self, field, kind, **kwargs):
        self.schema[field] = kind

    def add_frame_field(self, field, kind, **kwargs):
        self.frame_schema[field] = kind

    def create_index(self, field, unique=False):
        assert field == "argus_clip_id" and unique

    def add_sample(self, sample):
        sample.dataset = self
        self.samples.append(sample)

    def save(self):
        pass

    def match(self, sample_filter):
        return _View(sample_filter)

    def save_view(self, name, view, overwrite=False):
        assert overwrite
        if self.fail_saved_view == name:
            self.fail_saved_view = None
            raise RuntimeError("injected saved-view failure")
        self.views[name] = view


@pytest.fixture
def fake_fo(monkeypatch):
    datasets = {}

    def create(**kwargs):
        dataset = _Dataset(**kwargs)
        datasets[dataset.name] = dataset
        return dataset

    fake = SimpleNamespace(
        Dataset=create, dataset_exists=lambda name: name in datasets,
        load_dataset=lambda name: datasets[name], Sample=_Sample,
        VideoMetadata=lambda **kwargs: SimpleNamespace(**kwargs),
        Detection=_Document, Detections=_Document, Classification=_Document,
        ViewField=_ViewField,
        StringField=str, BooleanField=bool, IntField=int, FloatField=float,
        ListField=list, EmbeddedDocumentField=dict,
        launch_app=Mock(side_effect=AssertionError("browser launch is explicit CLI only")),
    )
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: fake if name == "fiftyone" else None)
    monkeypatch.setattr(adapter, "_dataset_import_lock", lambda name: nullcontext())
    return fake, datasets


def _record(index, predicted=0, **changes):
    return {
        "frame_index": index, "timestamp_sec": index / 25,
        "predicted": predicted, "score": 0.25,
        "score_eligible": True, "decision": "tn", "label": 0,
        "pipeline_mode": "active",
        "scene_guard": {"usable": True, "state": "stable", "profile_id": "profile-v1"},
        **changes,
    }


@pytest.fixture
def bundle(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"verified media bytes; no decoder is invoked")
    video_hash = file_sha256(video)
    return ReviewBundle.model_validate({
        "bundle_id": "review-1", "suite_id": "suite-1", "manifest_sha256": "a" * 64,
        "clips": [{
            "clip_id": "clip-1", "camera_id": "cam-1", "source_session_id": "capture-1",
            "video_path": str(video), "video_sha256": video_hash,
            "fps": 25.0, "width": 200, "height": 100, "frame_count": 8, "split": "test",
            "timestamps": [{
                "frame_index": index, "pts": 250 + index, "time_base_num": 1,
                "time_base_den": 25, "timestamp_sec": (250 + index) / 25,
            } for index in range(8)],
            "scene_profile_id": "profile-v1", "model_version_id": "model-v1",
            "annotations": {
                "reviewed": True, "reviewed_by": "engineer",
                "positive_ranges": [{"start_frame": 2, "end_frame": 3, "label": "foreign_object", "event_id": "event-1"}],
                "ignore_ranges": [{"start_frame": 4, "end_frame": 4, "label": "occluded"}],
                "boxes": [{"frame_index": 2, "label": "foreign_object", "xtl": 20, "ytl": 10, "xbr": 60, "ybr": 30, "track_id": 3}],
            },
            "evaluation": {
                "schema_version": "argus.pipeline-video-eval.v1", "evaluation_mode": "pipeline",
                "source_video_sha256": video_hash, "camera_id": "cam-1",
                "source_artifacts_stable": True,
                "model_version_id": "model-v1",
                "model_identity": {"actual": {"artifact_sha256": "b" * 64}},
                "frames": [
                    _record(0), _record(2, decision="tp", label=1),
                    _record(3, scene_guard={"usable": False, "state": "moved"}, decision="fn", label=1),
                    _record(4, predicted=1, decision="fp"),
                    _record(6, score_eligible=False, decision="excluded", exclusion_reason="warmup"),
                    _record(7, scene_guard={"state": "unknown"}),
                ],
                "failed_frames": [{"frame_index": 5, "error": "decoder stage failed"}],
                "metrics": {"event_recall": 0.5},
            },
        }],
    })


def _updated(bundle, update):
    payload = bundle.model_dump(mode="json")
    update(payload)
    return ReviewBundle.model_validate(payload)


@pytest.fixture
def scene_comparison(bundle, tmp_path, monkeypatch):
    """Importer tests isolate the separately tested report-validation boundary."""
    def decision(state, raw_state=None, *, error=False):
        return {
            "status": "error" if error else "evaluated", "state": state,
            "raw_state": raw_state or state, "reason": "injected failure" if error else state,
            "metrics": None if error else {"inliers": 30, "corner_shift_px": 0.0},
            "elapsed_ms": None if error else 2.5,
        }

    rows = []
    for index, baseline, lightglue, category in (
        (0, decision("stable"), decision("moved"), "confirmed_disagreement"),
        (2, decision("calibrating", "stable"), decision("calibrating", "moved"), "raw_disagreement"),
        (5, decision("stable"), decision("unknown", error=True), "error"),
    ):
        stamp = bundle.clips[0].timestamps[index]
        error = category == "error"
        rows.append({
            "frame_index": index, "pts": stamp.pts, "time_base": "1/25",
            "timestamp_sec": stamp.timestamp_sec, "relative_timestamp_sec": index / 25,
            "frame_sha256": str(index) * 64,
            "baseline": baseline, "lightglue": lightglue,
            "comparable": not error,
            "state_disagreement": None if error else baseline["state"] != lightglue["state"],
            "raw_state_disagreement": None if error else baseline["raw_state"] != lightglue["raw_state"],
            "comparison_error": error, "error_algorithms": ["lightglue"] if error else [],
            "review_category": category,
        })
    report_path = tmp_path / "scene-comparison.json"
    report_path.write_text(json.dumps({"frames": rows}), encoding="utf-8")
    review = SceneComparisonReview(
        clip_id="clip-1", report_path=report_path, report_sha256=file_sha256(report_path),
        report_status="partial", profile_id="profile-v1", frame_rows=tuple(rows),
        summary={"compared_frames": 3, "release_authority": False},
        _clip_snapshot_json=bundle.clips[0].model_dump_json(),
    )

    def verify(comparisons, *, bundle=None):
        for comparison in comparisons.values():
            if file_sha256(comparison.report_path) != comparison.report_sha256:
                raise ValueError("scene comparison report missing or changed")

    monkeypatch.setattr(adapter, "require_verified_scene_comparisons", verify)
    return review


def test_video_metadata_pts_frame_numbers_and_real_boxes_are_lossless(fake_fo, bundle):
    fo, _ = fake_fo
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    sample = dataset.samples[0]
    assert sample.metadata.total_frame_count == 8
    assert sample.metadata.frame_width == 200
    assert sample.metadata.frame_rate == 25
    assert set(sample.frames) == set(range(1, 9))
    first = sample.frames[1]
    assert first["argus_frame_index"] == 0
    assert first["argus_pts"] == 250
    assert first["argus_timestamp_sec"] == 10.0
    assert first["argus_relative_timestamp_sec"] == 0.0
    assert first["argus_eval_timestamp_sec"] == 0.0
    frame = sample.frames[3]
    assert frame["argus_frame_index"] == 2
    assert frame["argus_relative_timestamp_sec"] == pytest.approx(0.08)
    assert frame["ground_truth_boxes"]["detections"][0]["bounding_box"] == [0.1, 0.1, 0.2, 0.2]
    assert frame["ground_truth_boxes"]["detections"][0]["index"] == 3
    assert sample.frames[4]["ground_truth_boxes"]["detections"] == []
    assert sample["argus_model_sha256"] == "b" * 64
    fo.launch_app.assert_not_called()


def test_decisions_distinguish_unsampled_blocked_errors_ignore_and_old_evaluation(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    sample = dataset.samples[0]
    assert [sample.frames[index]["argus_decision"] for index in range(1, 9)] == [
        "tn", "unsampled", "fn", "blocked", "ignored", "error", "excluded", "unavailable",
    ]
    assert sample.frames[3]["argus_eval_decision"] == "tp"
    assert sample.frames[4]["argus_scene_guard_usable"] is False
    assert sample.frames[5]["ground_truth"] is None
    assert sample.frames[6]["argus_error"] == "decoder stage failed"
    assert sample.frames[2]["argus_predicted"] is None
    assert sample["argus_event_metrics_status"] == "source_report_only_rerun_required"
    assert dataset.views["argus-fn"].frame_filter.to_mongo(prefix="$$this") == {
        "$eq": ["$$this.argus_decision", "fn"],
    }
    assert dataset.views["argus-blocked"].frame_filter.to_mongo(prefix="$$this") == {
        "$eq": ["$$this.argus_blocked", True],
    }


def test_unreviewed_frames_cannot_become_normal_or_false_positives(fake_fo, bundle):
    changed = _updated(bundle, lambda payload: payload["clips"][0]["annotations"].update(reviewed=False, reviewed_by=None))
    sample = adapter.import_bundle_to_fiftyone(changed, dataset_name="review").samples[0]
    assert sample.frames[1]["argus_decision"] == "unreviewed"
    assert sample.frames[1]["ground_truth"] is None
    assert sample.frames[3]["ground_truth"] is None
    assert sample.frames[3]["argus_proposed_boxes"]["detections"]
    assert sample.frames[4]["argus_decision"] == "blocked"


@pytest.mark.parametrize("mode", ["collection", "training", "learning", "unknown"])
def test_non_detection_pipeline_mode_cannot_be_relabelled_normal(fake_fo, bundle, mode):
    changed = _updated(bundle, lambda payload: payload["clips"][0]["evaluation"]["frames"][0].update(pipeline_mode=mode))
    sample = adapter.import_bundle_to_fiftyone(changed, dataset_name="review").samples[0]
    assert sample.frames[1]["argus_decision"] == "excluded"
    assert sample.frames[1]["argus_score_eligible"] is False
    assert sample.frames[1]["argus_exclusion_reason"] == f"pipeline_mode_{mode}"


def test_explicit_report_failure_and_unstable_sources_cannot_become_normal(fake_fo, bundle):
    failed = _updated(bundle, lambda payload: payload["clips"][0]["evaluation"]["frames"][0].update(decision="error"))
    sample = adapter.import_bundle_to_fiftyone(failed, dataset_name="review").samples[0]
    assert sample.frames[1]["argus_decision"] == "error"
    assert sample.frames[1]["argus_error"]
    unstable = _updated(bundle, lambda payload: payload["clips"][0]["evaluation"].update(source_artifacts_stable=False))
    sample = adapter.import_bundle_to_fiftyone(unstable, dataset_name="review").samples[0]
    assert sample.frames[1]["argus_decision"] == "unavailable"


@pytest.mark.parametrize("missing, null", [(True, False), (False, True)])
def test_missing_or_null_pipeline_mode_keeps_diagnostics_without_scoring(fake_fo, bundle, missing, null):
    def change(payload):
        row = payload["clips"][0]["evaluation"]["frames"][0]
        if missing:
            row.pop("pipeline_mode")
        elif null:
            row["pipeline_mode"] = None

    sample = adapter.import_bundle_to_fiftyone(_updated(bundle, change), dataset_name="review").samples[0]
    assert sample.frames[1]["argus_decision"] == "unavailable"
    assert sample.frames[1]["argus_score_eligible"] is False
    assert sample.frames[1]["argus_score"] == 0.25
    assert sample.frames[1]["argus_eval_decision"] == "tn"


@pytest.mark.parametrize("stable", ["missing", None, False])
def test_missing_or_null_source_stability_cannot_be_assumed_true(fake_fo, bundle, stable):
    def change(payload):
        report = payload["clips"][0]["evaluation"]
        if stable == "missing":
            report.pop("source_artifacts_stable")
        else:
            report["source_artifacts_stable"] = stable

    sample = adapter.import_bundle_to_fiftyone(_updated(bundle, change), dataset_name="review").samples[0]
    assert sample.frames[1]["argus_decision"] == "unavailable"
    assert sample.frames[1]["argus_score_eligible"] is False
    assert sample.frames[1]["argus_predicted"] is False


def test_relative_time_uses_exact_pts_before_converting_large_absolute_values(fake_fo, bundle):
    def change(payload):
        for index, stamp in enumerate(payload["clips"][0]["timestamps"]):
            stamp["pts"] = 10**18 + index
            stamp["timestamp_sec"] = (10**18 + index) / 25

    sample = adapter.import_bundle_to_fiftyone(_updated(bundle, change), dataset_name="review").samples[0]
    assert sample.frames[2]["argus_relative_timestamp_sec"] == pytest.approx(0.04)
    assert sample.frames[3]["argus_relative_timestamp_sec"] == pytest.approx(0.08)


def test_scene_frames_expose_large_pts_as_lossless_strings_for_browser_review(
    fake_fo, bundle, scene_comparison,
):
    def change(payload):
        for index, stamp in enumerate(payload["clips"][0]["timestamps"]):
            stamp["pts"] = 10**18 + index
            stamp["timestamp_sec"] = (10**18 + index) / 25

    changed = _updated(bundle, change)
    rows = deepcopy(scene_comparison.frame_rows)
    for row in rows:
        stamp = changed.clips[0].timestamps[row["frame_index"]]
        row.update(pts=stamp.pts, timestamp_sec=stamp.timestamp_sec)
    scene_comparison.report_path.write_text(json.dumps({"frames": rows}), encoding="utf-8")
    comparison = replace(
        scene_comparison, report_sha256=file_sha256(scene_comparison.report_path),
        frame_rows=rows, _clip_snapshot_json=changed.clips[0].model_dump_json(),
    )
    dataset = adapter.import_bundle_to_fiftyone(
        changed, dataset_name="review", scene_comparisons={"clip-1": comparison},
    )
    sample = dataset.samples[0]
    assert sample.frames[2]["argus_pts"] == 10**18 + 1
    assert sample.frames[2]["argus_pts_exact"] == "1000000000000000001"
    assert sample.frames[2]["argus_time_base_exact"] == "1/25"
    assert sample.frames[2]["argus_relative_timestamp_sec"] == pytest.approx(0.04)
    assert dataset.frame_schema["argus_pts_exact"] is str
    assert dataset.frame_schema["argus_time_base_exact"] is str


def test_display_seconds_are_recomputed_from_pts_not_tolerated_redundant_float(fake_fo, bundle):
    def change(payload):
        for index, stamp in enumerate(payload["clips"][0]["timestamps"]):
            stamp["pts"] = 10**18 + index
            # The shared contract's relative float tolerance accepts this at a
            # very large absolute origin. It is not a display-time authority.
            stamp["timestamp_sec"] = (10**18 + index) / 25 + 1024

    changed = _updated(bundle, change)
    assert changed.clips[0].timestamps[0].timestamp_sec != float(Fraction(10**18, 25))
    sample = adapter.import_bundle_to_fiftyone(changed, dataset_name="review").samples[0]
    assert sample.frames[1]["argus_timestamp_sec"] == float(Fraction(10**18, 25))
    assert sample.frames[2]["argus_relative_timestamp_sec"] == pytest.approx(0.04)
    assert sample.frames[2]["argus_pts_exact"] == "1000000000000000001"


def test_frame_schema_revision_refreshes_identical_generation_without_erasing_review(
    fake_fo, bundle, scene_comparison, monkeypatch,
):
    monkeypatch.setattr(adapter, "_FRAME_SCHEMA_REVISION", 1)
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    sample = dataset.samples[0]
    generation = sample["argus_import_generation"]
    fingerprint = sample["argus_clip_fingerprint"]
    frame = sample.frames[1]
    frame["argus_scene_human_state"] = "moved"
    frame["argus_scene_human_note"] = "preserve manual review across schema upgrade"
    frame["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    sample.frames.save()
    for frame in sample.frames.persisted.values():
        frame.pop("argus_pts_exact", None)
        frame.pop("argus_time_base_exact", None)
    dataset.frame_schema.pop("argus_pts_exact")
    dataset.frame_schema.pop("argus_time_base_exact")
    saves = sample.saves
    monkeypatch.setattr(adapter, "_FRAME_SCHEMA_REVISION", 2)
    adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    assert sample["argus_import_generation"] == generation
    assert sample["argus_clip_fingerprint"] != fingerprint
    assert sample.saves > saves
    frame = sample.frames[1]
    assert frame["argus_pts_exact"] == "250" and frame["argus_time_base_exact"] == "1/25"
    assert frame["argus_scene_human_note"] == "preserve manual review across schema upgrade"
    assert frame["argus_scene_reviewed_report_sha256"] == scene_comparison.report_sha256
    assert dataset.views["argus-scene-reviewed"].frame_filter.evaluate(frame) is True
    saves = sample.saves
    adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    assert sample.saves == saves


def test_repeat_import_is_idempotent_and_new_review_rebuilds_managed_fields(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    sample = dataset.samples[0]
    saves = sample.saves
    sample["user_note"] = "keep"
    sample.frames[3]["user_frame_note"] = "keep frame note"
    adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    assert len(dataset.samples) == 1 and sample.saves == saves

    def change(payload):
        payload["bundle_id"] = "review-2"
        payload["clips"][0]["annotations"]["positive_ranges"] = []
        payload["clips"][0]["annotations"]["boxes"] = []
        payload["clips"][0]["evaluation"]["frames"][1]["predicted"] = 1

    adapter.import_bundle_to_fiftyone(_updated(bundle, change), dataset_name="review")
    assert len(dataset.samples) == 1
    assert sample["user_note"] == "keep"
    assert sample.frames[3]["user_frame_note"] == "keep frame note"
    assert sample.frames[3]["ground_truth_boxes"]["detections"] == []
    assert sample.frames[3]["ground_truth"]["label"] == "normal"
    assert sample.frames[3]["argus_decision"] == "fp"
    assert sample.frames[3]["argus_eval_decision"] == "tp"


def test_partial_refresh_is_marked_failed_and_retry_rebuilds_without_duplicates(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    sample = dataset.samples[0]
    changed = _updated(bundle, lambda payload: payload.update(bundle_id="review-2"))
    dataset.fail_frame_save = True
    with pytest.raises(RuntimeError, match="injected"):
        adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert dataset.info["argus_review"]["state"] == "failed"
    assert sample["argus_import_state"] == "pending"
    adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert len(dataset.samples) == 1
    assert dataset.info["argus_review"]["state"] == "complete"
    assert sample["argus_import_state"] == "complete"


def test_user_dataset_is_never_adopted_or_overwritten(fake_fo, bundle):
    fo, _ = fake_fo
    user_dataset = fo.Dataset(name="review")
    user_dataset.info = {"user": "original"}
    with pytest.raises(ValueError, match="not owned"):
        adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    assert user_dataset.info == {"user": "original"}
    assert not user_dataset.schema


def test_all_source_hashes_are_verified_before_fiftyone_import(monkeypatch, bundle):
    Path(bundle.clips[0].video_path).write_bytes(b"changed")
    import_module = Mock(side_effect=AssertionError("must not import FiftyOne"))
    monkeypatch.setattr(adapter.importlib, "import_module", import_module)
    with pytest.raises(ValueError, match="missing or changed"):
        adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    import_module.assert_not_called()


def test_source_changed_during_frame_writes_cannot_be_marked_complete(fake_fo, bundle, monkeypatch):
    _, datasets = fake_fo
    original_write = adapter._write_clip

    def mutate(*args, **kwargs):
        sample = original_write(*args, **kwargs)
        Path(bundle.clips[0].video_path).write_bytes(b"changed while importing")
        return sample

    monkeypatch.setattr(adapter, "_write_clip", mutate)
    with pytest.raises(ValueError, match="missing or changed"):
        adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    dataset = datasets["review"]
    assert dataset.info["argus_review"]["state"] == "failed"
    assert dataset.samples[0]["argus_import_state"] == "pending"
    assert not dataset.views


def test_new_dataset_schema_failure_keeps_owned_retryable_dataset(fake_fo, bundle, monkeypatch):
    _, datasets = fake_fo
    original = adapter._ensure_schema
    monkeypatch.setattr(adapter, "_ensure_schema", Mock(side_effect=RuntimeError("schema write failed")))
    with pytest.raises(RuntimeError, match="schema write failed"):
        adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    assert datasets["review"].info["argus_review"]["owner"] == adapter._OWNER
    assert datasets["review"].info["argus_review"]["state"] == "failed"
    monkeypatch.setattr(adapter, "_ensure_schema", original)
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    assert len(dataset.samples) == 1
    assert dataset.info["argus_review"]["state"] == "complete"


def test_media_identity_change_requires_a_new_dataset(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    previous_info = deepcopy(dataset.info)
    changed = _updated(bundle, lambda payload: payload["clips"][0].update(width=201))
    with pytest.raises(ValueError, match="media changed"):
        adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert dataset.info == previous_info


def test_actual_evaluated_model_and_profile_do_not_take_manifest_requested_identity(fake_fo, bundle):
    def change(payload):
        report = payload["clips"][0]["evaluation"]
        report["model_identity"]["actual"]["version_id"] = "actual-model-v2"
        report["scene_reference"] = {"profile_id": "actual-profile-v2"}

    changed = _updated(bundle, change)
    sample = adapter.import_bundle_to_fiftyone(changed, dataset_name="review").samples[0]
    assert sample["argus_model_version_id"] == "actual-model-v2"
    assert sample["argus_scene_profile_id"] == "actual-profile-v2"
    assert sample["argus_expected_model_version_id"] == "model-v1"
    assert sample["argus_expected_scene_profile_id"] == "profile-v1"
    assert sample["argus_model_identity_status"] == "reported"


def test_requested_or_report_declared_model_is_not_misrepresented_as_actual(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    sample = dataset.samples[0]
    assert sample["argus_model_version_id"] is None
    assert sample["argus_expected_model_version_id"] == "model-v1"
    assert sample["argus_report_model_version_id"] == "model-v1"
    assert sample["argus_model_identity_status"] == "unavailable"

    changed = _updated(bundle, lambda payload: payload["clips"][0].update(evaluation=None))
    sample = adapter.import_bundle_to_fiftyone(changed, dataset_name="review").samples[0]
    assert sample["argus_model_version_id"] is None
    assert sample["argus_model_sha256"] is None
    assert sample["argus_report_model_version_id"] is None
    assert sample["argus_model_identity_status"] == "not_evaluated"
    assert sample["argus_scene_profile_id"] is None
    assert sample["argus_expected_scene_profile_id"] == "profile-v1"


def test_long_video_frames_are_attached_and_flushed_in_bounded_batches(fake_fo, bundle, monkeypatch):
    monkeypatch.setattr(adapter, "_FRAME_WRITE_BATCH_SIZE", 3)
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    frames = dataset.samples[0].frames
    assert frames.peak_cache <= 3
    assert [size for size in frames.batch_sizes if size] == [3, 3, 2]
    assert set(frames.persisted) == set(range(1, 9))
    assert frames.persisted[3]["argus_frame_index"] == 2


def test_saved_view_failure_quarantines_already_completed_generation(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    sample = dataset.samples[0]
    changed = _updated(bundle, lambda payload: payload.update(bundle_id="review-2"))
    dataset.fail_saved_view = "argus-fn"
    with pytest.raises(RuntimeError, match="saved-view failure"):
        adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert dataset.info["argus_review"]["state"] == "failed"
    assert sample["argus_import_state"] == "pending"
    assert dataset.views["argus-current"].sample_filter["argus_import_state"] == "complete"
    # Retrying the exact generation must rebuild the quarantined sample.
    saves = sample.saves
    adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert sample.saves > saves
    assert sample["argus_import_state"] == "complete"
    assert dataset.info["argus_review"]["state"] == "complete"


def test_malformed_prediction_and_conflicting_failed_frame_do_not_mutate_dataset(fake_fo, bundle):
    _, datasets = fake_fo
    changed = _updated(bundle, lambda payload: payload["clips"][0]["evaluation"]["frames"][0].update(predicted="false"))
    with pytest.raises(ValueError, match="predictions"):
        adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert not datasets
    changed = _updated(bundle, lambda payload: payload["clips"][0]["evaluation"]["failed_frames"].append({"frame_index": 0, "error": "failed"}))
    with pytest.raises(ValueError, match="contradictory"):
        adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert not datasets


@pytest.mark.parametrize("changes, message", [
    ({"fps": 30.0}, "fps differs"),
    ({"source_total_frames_known": True, "source_total_frames": 9}, "source frame count"),
    ({"total_frames": 9}, "total_frames"),
    ({"total_frames": 7, "source_exhausted": True}, "exhausted evaluation"),
    ({"processed_frames": 5}, "processed_frames"),
    ({"sample_every": 2}, "sampled-frame grid"),
])
def test_contradictory_evaluation_frame_map_is_rejected_before_dataset_mutation(
    fake_fo, bundle, changes, message,
):
    _, datasets = fake_fo
    changed = _updated(bundle, lambda payload: payload["clips"][0]["evaluation"].update(changes))
    with pytest.raises(ValueError, match=message):
        adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert not datasets


def test_shifted_evaluation_timestamps_cannot_become_false_normal_frames(fake_fo, bundle):
    _, datasets = fake_fo
    changed = _updated(bundle, lambda payload: payload["clips"][0]["evaluation"]["frames"][0].update(timestamp_sec=0.04))
    with pytest.raises(ValueError, match="original frame index"):
        adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert not datasets


def test_foreign_host_in_progress_import_is_not_silently_stolen(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    dataset.info["argus_review"].update(state="importing", importing_host="another-analysis-host")
    with pytest.raises(RuntimeError, match="another host"):
        adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")


def test_completed_dataset_remains_bound_to_one_importing_host(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    dataset.info["argus_review"]["importing_host"] = "another-analysis-host"
    old_info = deepcopy(dataset.info)
    saves = dataset.samples[0].saves
    with pytest.raises(RuntimeError, match="another host"):
        adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    assert dataset.info == old_info
    assert dataset.samples[0].saves == saves


def test_incremental_bundles_cannot_put_one_capture_session_in_train_and_test(fake_fo, bundle, tmp_path):
    train = _updated(bundle, lambda payload: payload["clips"][0].update(split="train"))
    dataset = adapter.import_bundle_to_fiftyone(train, dataset_name="review")
    old_info = deepcopy(dataset.info)
    second_video = tmp_path / "another.mp4"
    second_video.write_bytes(b"another video from the same capture session")

    def change(payload):
        payload["bundle_id"] = "review-2"
        payload["clips"][0].update(
            clip_id="clip-2", video_path=str(second_video),
            video_sha256=file_sha256(second_video), evaluation=None,
        )

    with pytest.raises(ValueError, match="crosses dataset splits"):
        adapter.import_bundle_to_fiftyone(_updated(bundle, change), dataset_name="review")
    assert len(dataset.samples) == 1 and dataset.info == old_info


def test_previously_assigned_clip_cannot_be_moved_back_to_unassigned(fake_fo, bundle):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    changed = _updated(bundle, lambda payload: payload["clips"][0].update(split="unassigned"))
    with pytest.raises(ValueError, match="split is immutable"):
        adapter.import_bundle_to_fiftyone(changed, dataset_name="review")
    assert dataset.samples[0]["argus_split"] == "test"


def test_unassigned_history_requires_whole_capture_session_to_be_assigned(fake_fo, bundle):
    def make_unassigned(payload):
        first = payload["clips"][0]
        first["split"] = "unassigned"
        second = deepcopy(first)
        second["clip_id"] = "same-session-2"
        payload["clips"].append(second)

    pending = _updated(bundle, make_unassigned)
    dataset = adapter.import_bundle_to_fiftyone(pending, dataset_name="review")

    def assign_only_first(payload):
        payload["clips"] = [payload["clips"][0]]
        payload["clips"][0]["split"] = "test"

    with pytest.raises(ValueError, match="update all unassigned members"):
        adapter.import_bundle_to_fiftyone(_updated(pending, assign_only_first), dataset_name="review")

    def assign_both(payload):
        for clip in payload["clips"]:
            clip["split"] = "test"

    adapter.import_bundle_to_fiftyone(_updated(pending, assign_both), dataset_name="review")
    assert [sample["argus_split"] for sample in dataset.samples] == ["test", "test"]


def test_missing_optional_dependency_preserves_actionable_module_name(monkeypatch, bundle):
    def missing(name):
        raise ModuleNotFoundError("install the optional FiftyOne analysis dependency", name=name)

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError) as caught:
        adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    assert caught.value.name == "fiftyone"


def test_scene_comparison_attaches_diagnostic_fields_without_changing_pipeline_truth(
    fake_fo, bundle, scene_comparison,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    sample = dataset.samples[0]
    frame = sample.frames[1]
    assert frame["argus_scene_disagreement"] is True
    assert frame["argus_scene_baseline_state"] == "stable"
    assert frame["argus_scene_lightglue_state"] == "moved"
    assert frame["argus_scene_baseline_elapsed_ms"] == 2.5
    assert json.loads(frame["argus_scene_baseline_metrics_json"])["inliers"] == 30
    assert frame["argus_scene_report_sha256"] == scene_comparison.report_sha256
    assert frame["argus_frame_index"] == 0 and frame["argus_pts"] == 250
    assert frame["argus_scene_guard_usable"] is True
    assert frame["argus_decision"] == "tn" and frame["argus_truth"] is False
    assert frame["ground_truth"]["label"] == "normal"
    assert "argus_scene_usable" not in frame
    assert sample["argus_scene_comparison_status"] == "partial"
    assert sample["argus_scene_comparison_profile_id"] == "profile-v1"
    assert json.loads(sample["argus_scene_comparison_summary_json"]) == scene_comparison.summary
    assert "frames" not in json.loads(sample["argus_scene_comparison_summary_json"])


def test_scene_views_distinguish_confirmed_raw_error_and_unsampled_frames(
    fake_fo, bundle, scene_comparison,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    sample = dataset.samples[0]
    confirmed = dataset.views["argus-scene-disagreements"].frame_filter
    raw = dataset.views["argus-scene-raw-disagreements"].frame_filter
    error = dataset.views["argus-scene-errors"].frame_filter
    assert confirmed.evaluate(sample.frames[1]) is True
    assert confirmed.evaluate(sample.frames[3]) is False
    assert raw.evaluate(sample.frames[3]) is True
    assert error.evaluate(sample.frames[6]) is True
    assert sample.frames[6]["argus_scene_disagreement"] is None
    assert sample.frames[6]["argus_scene_raw_disagreement"] is None
    assert sample.frames[2]["argus_scene_has_comparison"] is False
    assert sample.frames[2]["argus_scene_lightglue_state"] is None
    assert confirmed.to_mongo(prefix="$$this") == {
        "$eq": ["$$this.argus_scene_disagreement", True],
    }
    assert dataset.views["argus-scene-disagreements"].sample_filter == {
        "argus_import_state": "complete", "argus_import_generation": sample["argus_import_generation"],
    }


def test_scene_unreviewed_view_dynamically_requires_current_report_hash_and_legal_human_state(
    fake_fo, bundle, scene_comparison,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    frame = dataset.samples[0].frames[1]
    expression = dataset.views["argus-scene-unreviewed"].frame_filter
    assert expression.evaluate(frame) is True
    frame["argus_scene_human_state"] = "moved"
    frame["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    assert expression.evaluate(frame) is False
    frame["argus_scene_reviewed_report_sha256"] = "f" * 64
    assert expression.evaluate(frame) is True
    frame["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    frame["argus_scene_human_state"] = "approved"
    assert expression.evaluate(frame) is True
    serialized = json.dumps(expression.to_mongo(prefix="$$this"))
    assert "$$this.argus_scene_reviewed_report_sha256" in serialized
    assert "$$this.argus_scene_report_sha256" in serialized
    assert "$$this.argus_scene_human_state" in serialized
    assert "$$this.argus_scene_has_comparison" in serialized


def test_scene_reimport_preserves_human_review_but_new_report_hash_invalidates_it(
    fake_fo, bundle, scene_comparison,
):
    mapping = {"clip-1": scene_comparison}
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review", scene_comparisons=mapping)
    sample = dataset.samples[0]
    frame = sample.frames[1]
    frame["argus_scene_human_state"] = "moved"
    frame["argus_scene_human_note"] = "camera shifted after recording"
    frame["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    sample.frames.save()
    saves = sample.saves
    adapter.import_bundle_to_fiftyone(bundle, dataset_name="review", scene_comparisons=mapping)
    assert sample.saves == saves
    assert dataset.views["argus-scene-unreviewed"].frame_filter.evaluate(sample.frames[1]) is False
    old_generation = sample["argus_import_generation"]
    scene_comparison.report_path.write_text('{"revision":2}', encoding="utf-8")
    new_review = replace(scene_comparison, report_sha256=file_sha256(scene_comparison.report_path))
    adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": new_review},
    )
    assert len(dataset.samples) == 1
    assert sample["argus_import_generation"] != old_generation
    assert sample.frames[1]["argus_scene_human_state"] == "moved"
    assert sample.frames[1]["argus_scene_human_note"] == "camera shifted after recording"
    assert sample.frames[1]["argus_scene_reviewed_report_sha256"] == scene_comparison.report_sha256
    assert dataset.views["argus-scene-unreviewed"].frame_filter.evaluate(sample.frames[1]) is True


def test_import_without_scene_report_clears_old_managed_fields_but_keeps_human_notes(
    fake_fo, bundle, scene_comparison,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    sample = dataset.samples[0]
    sample.frames[1]["argus_scene_human_note"] = "keep this historical observation"
    sample.frames[1]["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    sample.frames.save()
    adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    assert sample["argus_has_scene_comparison"] is False
    assert sample["argus_scene_comparison_sha256"] is None
    assert sample["argus_scene_comparison_summary_json"] is None
    assert sample.frames[1]["argus_scene_has_comparison"] is False
    assert sample.frames[1]["argus_scene_disagreement"] is False
    assert sample.frames[1]["argus_scene_lightglue_metrics_json"] is None
    assert sample.frames[1]["argus_scene_report_sha256"] is None
    assert sample.frames[1]["argus_scene_human_note"] == "keep this historical observation"
    assert sample.frames[1]["argus_scene_reviewed_report_sha256"] == scene_comparison.report_sha256
    assert dataset.views["argus-scene-unreviewed"].frame_filter.evaluate(sample.frames[1]) is False


def test_scene_report_relocation_refreshes_path_and_preserves_hash_bound_review(
    fake_fo, bundle, scene_comparison, tmp_path,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    sample = dataset.samples[0]
    sample.frames[1]["argus_scene_human_state"] = "moved"
    sample.frames[1]["argus_scene_human_note"] = "same report, relocated archive"
    sample.frames[1]["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    sample.frames.save()
    previous_fingerprint = sample["argus_clip_fingerprint"]
    relocated_path = tmp_path / "relocated-scene-report.json"
    relocated_path.write_bytes(scene_comparison.report_path.read_bytes())
    relocated = replace(scene_comparison, report_path=relocated_path)
    adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": relocated},
    )
    assert sample["argus_scene_comparison_path"] == str(relocated_path)
    assert sample["argus_clip_fingerprint"] != previous_fingerprint
    assert sample.frames[1]["argus_scene_human_note"] == "same report, relocated archive"
    assert sample.frames[1]["argus_scene_reviewed_report_sha256"] == scene_comparison.report_sha256
    assert dataset.views["argus-scene-unreviewed"].frame_filter.evaluate(sample.frames[1]) is False


def test_scene_review_views_separate_current_and_stale_without_cached_review_flags(
    fake_fo, bundle, scene_comparison,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    frame = dataset.samples[0].frames[1]
    reviewed = dataset.views["argus-scene-reviewed"].frame_filter
    stale = dataset.views["argus-scene-stale-review"].frame_filter
    pending = dataset.views["argus-scene-unreviewed"].frame_filter
    assert reviewed.evaluate(frame) is False
    assert stale.evaluate(frame) is False
    assert pending.evaluate(frame) is True
    for value in (None, ""):
        frame["argus_scene_reviewed_report_sha256"] = value
        frame["argus_scene_human_state"] = "moved"
        assert reviewed.evaluate(frame) is False
        assert stale.evaluate(frame) is False
        assert pending.evaluate(frame) is True
    frame["argus_scene_reviewed_report_sha256"] = "e" * 64
    assert reviewed.evaluate(frame) is False
    assert stale.evaluate(frame) is True
    assert pending.evaluate(frame) is True
    frame["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    assert reviewed.evaluate(frame) is True
    assert stale.evaluate(frame) is False
    assert pending.evaluate(frame) is False
    frame["argus_scene_human_state"] = "approved"
    assert reviewed.evaluate(frame) is False
    assert pending.evaluate(frame) is True
    serialized = json.dumps(pending.to_mongo(prefix="$$this"))
    assert '"$cond"' in serialized and '"$gt"' in serialized


def test_scene_report_remove_and_exact_restore_preserve_only_hash_bound_review(
    fake_fo, bundle, scene_comparison,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    sample = dataset.samples[0]
    frame = sample.frames[3]
    frame["argus_scene_human_state"] = "uncertain"
    frame["argus_scene_human_note"] = "historical note bound to this exact report"
    frame["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    sample.frames.save()
    adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    assert sample.frames[3]["argus_scene_has_comparison"] is False
    assert dataset.views["argus-scene-reviewed"].frame_filter.evaluate(sample.frames[3]) is False
    assert dataset.views["argus-scene-stale-review"].frame_filter.evaluate(sample.frames[3]) is False
    adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    frame = sample.frames[3]
    assert frame["argus_scene_human_note"] == "historical note bound to this exact report"
    assert frame["argus_scene_reviewed_report_sha256"] == scene_comparison.report_sha256
    assert dataset.views["argus-scene-reviewed"].frame_filter.evaluate(frame) is True
    assert dataset.views["argus-scene-unreviewed"].frame_filter.evaluate(frame) is False


def test_scene_report_sampling_change_clears_observations_not_historical_review(
    fake_fo, bundle, scene_comparison,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    sample = dataset.samples[0]
    frame = sample.frames[3]
    frame["argus_scene_human_state"] = "moved"
    frame["argus_scene_human_note"] = "frame no longer sampled in next report"
    frame["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
    sample.frames.save()
    scene_comparison.report_path.write_text('{"sampling_revision":2}', encoding="utf-8")
    changed = replace(
        scene_comparison, report_sha256=file_sha256(scene_comparison.report_path),
        frame_rows=(deepcopy(scene_comparison.frame_rows[0]),),
    )
    adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": changed},
    )
    frame = sample.frames[3]
    assert frame["argus_scene_has_comparison"] is False
    assert frame["argus_scene_report_sha256"] is None
    assert frame["argus_scene_lightglue_state"] is None
    assert frame["argus_scene_human_note"] == "frame no longer sampled in next report"
    assert frame["argus_scene_reviewed_report_sha256"] == scene_comparison.report_sha256
    for name in ("argus-scene-reviewed", "argus-scene-stale-review", "argus-scene-unreviewed"):
        assert dataset.views[name].frame_filter.evaluate(frame) is False


def test_refresh_does_not_erase_review_saved_after_frame_was_cached(
    fake_fo, bundle, scene_comparison, monkeypatch,
):
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    sample = dataset.samples[0]
    changed = _updated(bundle, lambda payload: payload.update(bundle_id="review-2"))
    original_get = _Frames.__getitem__
    injected = False

    def reviewer_saves_after_cache_read(frames, number):
        nonlocal injected
        frame = original_get(frames, number)
        if frames is sample.frames and number == 1 and not injected:
            injected = True
            assert frame.get_field("argus_scene_human_note") is None
            stored = frames.persisted[number]
            stored["argus_scene_human_state"] = "moved"
            stored["argus_scene_human_note"] = "saved by another client during import"
            stored["argus_scene_reviewed_report_sha256"] = scene_comparison.report_sha256
        return frame

    monkeypatch.setattr(_Frames, "__getitem__", reviewer_saves_after_cache_read)
    adapter.import_bundle_to_fiftyone(
        changed, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    frame = sample.frames[1]
    assert injected
    assert frame["argus_scene_human_state"] == "moved"
    assert frame["argus_scene_human_note"] == "saved by another client during import"
    assert frame["argus_scene_reviewed_report_sha256"] == scene_comparison.report_sha256
    assert dataset.views["argus-scene-reviewed"].frame_filter.evaluate(frame) is True


def test_scene_video_vfr_rejected_before_fiftyone_import_even_at_unsampled_frame(
    monkeypatch, bundle, scene_comparison,
):
    def change(payload):
        clip = payload["clips"][0]
        clip["evaluation"] = None
        for index, stamp in enumerate(clip["timestamps"]):
            stamp.update(
                pts=250000 + 1000 * index + (100 if index == 1 else 0),
                time_base_num=1, time_base_den=25000,
            )
            stamp["timestamp_sec"] = stamp["pts"] / 25000

    changed = _updated(bundle, change)
    optional_import = Mock(side_effect=AssertionError("must not start FiftyOne"))
    monkeypatch.setattr(adapter.importlib, "import_module", optional_import)
    with pytest.raises(ValueError, match="nominal-FPS seek grid.*frame 1"):
        adapter.import_bundle_to_fiftyone(
            changed, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
        )
    optional_import.assert_not_called()


def test_scene_video_fine_tick_quantized_cfr_is_not_rejected_as_vfr(
    fake_fo, bundle, scene_comparison,
):
    def change(payload):
        clip = payload["clips"][0]
        clip.update(fps=30000 / 1001, evaluation=None)
        for index, stamp in enumerate(clip["timestamps"]):
            pts = -10**12 + round(Fraction(index * 1001 * 1_000_000, 30000))
            stamp.update(
                pts=pts, time_base_num=1, time_base_den=1_000_000,
                timestamp_sec=pts / 1_000_000,
            )

    changed = _updated(bundle, change)
    dataset = adapter.import_bundle_to_fiftyone(
        changed, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    assert dataset.samples[0].frames[1]["argus_relative_timestamp_sec"] == 0.0
    assert dataset.samples[0].frames[2]["argus_relative_timestamp_sec"] == pytest.approx(0.033367)


@pytest.mark.parametrize("origin", [0, 250, -250])
def test_scene_video_nominal_grid_uses_exact_relative_pts_not_absolute_origin(
    fake_fo, bundle, scene_comparison, origin,
):
    def change(payload):
        clip = payload["clips"][0]
        clip["evaluation"] = None
        for index, stamp in enumerate(clip["timestamps"]):
            stamp.update(pts=origin + index, timestamp_sec=(origin + index) / 25)

    changed = _updated(bundle, change)
    dataset = adapter.import_bundle_to_fiftyone(
        changed, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    assert dataset.samples[0].frames[1]["argus_relative_timestamp_sec"] == 0.0
    assert dataset.samples[0].frames[3]["argus_relative_timestamp_sec"] == pytest.approx(0.08)


def test_video_without_scene_comparison_keeps_existing_vfr_import_behavior(fake_fo, bundle):
    def change(payload):
        clip = payload["clips"][0]
        clip["evaluation"] = None
        for index, stamp in enumerate(clip["timestamps"]):
            pts = 250000 + 1000 * index + (100 if index == 1 else 0)
            stamp.update(pts=pts, time_base_num=1, time_base_den=25000, timestamp_sec=pts / 25000)

    dataset = adapter.import_bundle_to_fiftyone(_updated(bundle, change), dataset_name="review")
    assert dataset.samples[0].frames[2]["argus_relative_timestamp_sec"] == pytest.approx(0.044)


def test_scene_report_is_verified_before_optional_fiftyone_import(monkeypatch, bundle):
    verify = Mock(side_effect=ValueError("changed scene report"))
    optional_import = Mock(side_effect=AssertionError("must not import FiftyOne"))
    monkeypatch.setattr(adapter, "require_verified_scene_comparisons", verify)
    monkeypatch.setattr(adapter.importlib, "import_module", optional_import)
    with pytest.raises(ValueError, match="changed scene report"):
        adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    verify.assert_called_once()
    assert verify.call_args.kwargs["bundle"].bundle_id == bundle.bundle_id
    optional_import.assert_not_called()


@pytest.mark.parametrize("invalid", [[], "report.json", [("clip-1", "report.json")]])
def test_scene_comparison_argument_requires_an_explicit_mapping(monkeypatch, bundle, invalid):
    optional_import = Mock(side_effect=AssertionError("must not import FiftyOne"))
    monkeypatch.setattr(adapter.importlib, "import_module", optional_import)
    with pytest.raises(ValueError, match="clip-ID mapping"):
        adapter.import_bundle_to_fiftyone(
            bundle, dataset_name="review", scene_comparisons=invalid,
        )
    optional_import.assert_not_called()


def test_scene_report_changed_during_import_quarantines_frames_and_keeps_retryable_dataset(
    fake_fo, bundle, scene_comparison, monkeypatch,
):
    _, datasets = fake_fo
    original_write = adapter._write_clip

    def mutate(*args, **kwargs):
        sample = original_write(*args, **kwargs)
        scene_comparison.report_path.write_bytes(b"changed while importing")
        return sample

    monkeypatch.setattr(adapter, "_write_clip", mutate)
    with pytest.raises(ValueError, match="scene comparison report missing or changed"):
        adapter.import_bundle_to_fiftyone(
            bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
        )
    dataset = datasets["review"]
    assert dataset.info["argus_review"]["state"] == "failed"
    assert dataset.samples[0]["argus_import_state"] == "pending"
    assert not dataset.views


def test_scene_saved_view_failure_does_not_publish_a_partial_generation(
    fake_fo, bundle, scene_comparison,
):
    dataset = adapter.import_bundle_to_fiftyone(bundle, dataset_name="review")
    dataset.fail_saved_view = "argus-scene-unreviewed"
    with pytest.raises(RuntimeError, match="saved-view failure"):
        adapter.import_bundle_to_fiftyone(
            bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
        )
    assert dataset.info["argus_review"]["state"] == "failed"
    assert dataset.samples[0]["argus_import_state"] == "pending"
    dataset = adapter.import_bundle_to_fiftyone(
        bundle, dataset_name="review", scene_comparisons={"clip-1": scene_comparison},
    )
    assert dataset.info["argus_review"]["state"] == "complete"
    assert dataset.samples[0]["argus_import_state"] == "complete"
