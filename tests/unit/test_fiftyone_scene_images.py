"""Image adapter contracts; fake FiftyOne never opens media or a database."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus.integrations import fiftyone_scene_images as adapter
from argus.integrations.contracts import ReviewBundle, file_sha256


class NetworkTimeout(RuntimeError):
    """Driver-shaped unknown outcome, without importing the optional driver."""

    __module__ = "pymongo.errors"


class AutoReconnect(RuntimeError):
    __module__ = "pymongo.errors"


class OperationError(RuntimeError):
    __module__ = "mongoengine.errors"


class CollectionInvalid(RuntimeError):
    __module__ = "pymongo.errors"


class OperationFailure(RuntimeError):
    __module__ = "pymongo.errors"

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class _MetadataCursor:
    def __init__(self, documents):
        self._documents = iter(documents)
        self.close = Mock()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._documents)


class _Expression:
    def __init__(self, operator, *arguments):
        self.operator = operator
        self.arguments = arguments

    def __eq__(self, other):
        return _Expression("eq", self, other)

    def __and__(self, other):
        return _Expression("and", self, other)

    def __invert__(self):
        return _Expression("not", self)

    def exists(self):
        return _Expression("exists", self)

    def if_else(self, true_value, false_value):
        return _Expression("if", self, true_value, false_value)

    def is_in(self, values):
        return _Expression("in", self, values)

    def evaluate(self, sample):
        values = [
            item.evaluate(sample) if isinstance(item, _Expression) else item
            for item in self.arguments
        ]
        if self.operator == "field":
            return sample.get(values[0])
        if self.operator == "eq":
            return values[0] == values[1]
        if self.operator == "and":
            return all(values)
        if self.operator == "not":
            return not values[0]
        if self.operator == "exists":
            return values[0] is not None
        if self.operator == "if":
            return values[1] if values[0] else values[2]
        if self.operator == "in":
            return values[0] in values[1]
        raise AssertionError(self.operator)


class _View:
    def __init__(self, filters=()):
        self.filters = filters

    def match(self, value):
        return _View((*self.filters, value))

    def contains(self, sample):
        return all(
            all(sample.get(key) == value for key, value in expression.items())
            if isinstance(expression, dict) else expression.evaluate(sample)
            for expression in self.filters
        )


class _Sample(dict):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.metadata = None
        self.dataset = None
        self.dirty = set()
        self.stored = {}
        self.writes = []
        self.record = None
        self.reload_count = 0

    def __setitem__(self, name, value):
        self.dirty.add(name)
        super().__setitem__(name, value)

    def get_field(self, name):
        return self.get(name)

    def save(self):
        assert self.dataset is not None, "sample must be attached before save"
        if self.dataset.before_sample_save is not None:
            self.dataset.before_sample_save(self)
        unknown = self.dataset.sample_unknown_fault
        if unknown is not None:
            self.dataset.sample_unknown_fault = None
            fields = frozenset(self.dirty)
            snapshot = {name: deepcopy(self[name]) for name in fields}
            record = self.record
            filepath, metadata = self.filepath, deepcopy(self.metadata)

            def late_sample_write():
                record["fields"].update(snapshot)
                record["writes"].append(fields)
                record["filepath"], record["metadata"] = filepath, metadata

            self.dataset.pending_writes.append(late_sample_write)
            raise unknown("sample update may still be running on the server")
        if self.dataset.fail_next_sample_save:
            self.dataset.fail_next_sample_save = False
            raise RuntimeError("injected sample save failure")
        self.writes.append(frozenset(self.dirty))
        for name in self.dirty:
            self.stored[name] = deepcopy(self[name])
        self.dirty.clear()
        self.record["filepath"] = self.filepath
        self.record["metadata"] = deepcopy(self.metadata)

    def reload(self):
        # Like a cached ODM document, saved dirty fields merge with concurrent
        # updates. This fake never replaces the whole persisted document.
        dict.clear(self)
        dict.update(self, deepcopy(self.stored))
        self.dirty.clear()
        self.reload_count += 1
        if self.record is not None:
            self.filepath = self.record["filepath"]
            self.metadata = deepcopy(self.record["metadata"])

    @classmethod
    def from_record(cls, record, dataset):
        sample = cls(record["filepath"])
        sample.dataset = dataset
        sample.record = record
        sample.stored = record["fields"]
        sample.writes = record["writes"]
        sample.reload()
        return sample


class _Dataset:
    _DURABLE_CONTROLS = frozenset({
        "persisted_info", "persisted_media_type", "schema", "views",
        "before_sample_save", "fail_next_sample_save", "fail_saved_view",
        "fail_complete_save", "info_save_fault", "info_save_attempts", "info_writes",
        "hard_refresh_writes", "cached_load_writes",
        "info_unknown_fault", "sample_unknown_fault", "pending_writes",
    })

    def __init__(self, name, persistent=True, *, store=None):
        self.name = name
        self.persistent = persistent
        self.version = "1.fake-review"
        self._store = store if store is not None else SimpleNamespace(
            persisted_info={}, persisted_media_type=None, records=[], schema={}, views={},
            before_sample_save=None, fail_next_sample_save=False, fail_saved_view=None,
            fail_complete_save=False, info_save_fault=None, info_save_attempts=0,
            info_writes=[], hard_refresh_writes=0, cached_load_writes=0,
            info_unknown_fault=None, sample_unknown_fault=None, pending_writes=[],
            identity="fake-dataset-id:" + name, version=self.version,
        )
        self.info = deepcopy(self.persisted_info)
        self.reload_count = 0
        self.media_type = self.persisted_media_type
        self.samples = [_Sample.from_record(record, self) for record in self._store.records]

    def __getattr__(self, name):
        if name in self._DURABLE_CONTROLS:
            return getattr(self._store, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._DURABLE_CONTROLS and "_store" in self.__dict__:
            setattr(self._store, name, value)
        else:
            object.__setattr__(self, name, value)

    def iter_samples(self):
        return iter(self.samples)

    def add_sample_field(self, name, field, **kwargs):
        self.schema[name] = field

    def create_index(self, name, unique=False):
        assert name == "argus_scene_image_key" and unique

    def add_sample(self, sample):
        self.samples.append(sample)
        sample.dataset = self
        sample.record = {
            "fields": sample.stored, "writes": sample.writes,
            "filepath": sample.filepath, "metadata": deepcopy(sample.metadata),
        }
        self._store.records.append(sample.record)
        sample.save()

    def save(self):
        self.info_save_attempts += 1
        state = self.info[adapter._INFO_KEY]["state"] if adapter._INFO_KEY in self.info else None
        unknown = self.info_unknown_fault
        if unknown is not None and unknown[0] == state:
            self.info_unknown_fault = None
            snapshot, media_type, store = deepcopy(self.info), self.media_type, self._store

            def late_metadata_write():
                store.persisted_info = deepcopy(snapshot)
                store.persisted_media_type = media_type
                store.info_writes.append(deepcopy(snapshot))

            self.pending_writes.append(late_metadata_write)
            raise unknown[1]("metadata update may still be running on the server")
        fault = self.info_save_fault
        if fault is not None and fault[0] == state:
            self.info_save_fault = None
            if fault[1] == "before":
                raise RuntimeError("injected metadata failure before commit")
        if self.fail_complete_save and self.info[adapter._INFO_KEY]["state"] == "complete":
            self.fail_complete_save = False
            raise RuntimeError("injected complete metadata failure")
        self.persisted_info = deepcopy(self.info)
        self.persisted_media_type = self.media_type
        self.info_writes.append(deepcopy(self.persisted_info))
        if fault is not None and fault[0] == state and fault[1] == "after":
            raise RuntimeError("injected metadata failure after commit")

    def reload(self):
        # The real public hard refresh can save schema/frame changes before
        # reloading samples. Any use is a safety regression, not a harmless read.
        self.reload_count += 1
        self.hard_refresh_writes += 1
        for sample in self.samples:
            sample.reload()
        raise AssertionError("unsafe public dataset hard refresh was invoked")

    def match(self, expression):
        return _View().match(expression)

    def save_view(self, name, view, overwrite=False):
        assert overwrite
        if self.fail_saved_view == name:
            self.fail_saved_view = None
            raise RuntimeError("injected saved view failure")
        self.views[name] = view


class _ImportLocks:
    """Persistent unique-key lock store, not an in-process mutex shortcut."""

    def __init__(self):
        self.documents = {}
        self.insert_fault = None
        self.delete_fault = None
        self.insert_acknowledged = True
        self.delete_acknowledged = True
        self.before_delete = None
        self.read_preference = SimpleNamespace(mode=0)
        self.read_concern = SimpleNamespace(level="local")
        self.write_concern = SimpleNamespace(acknowledged=True)
        self.indexes = {"_id_": {"key": [("_id", 1)]}}
        self.index_information = Mock(side_effect=lambda: deepcopy(self.indexes))
        self.insert_one = Mock(side_effect=self._insert_one)
        self.delete_one = Mock(side_effect=self._delete_one)

    def _insert_one(self, document):
        if self.insert_fault == "before":
            raise RuntimeError("injected lock acquisition failure before commit")
        key = document["_id"]
        if key in self.documents:
            raise RuntimeError("duplicate persistent import lock")
        self.documents[key] = deepcopy(document)
        if self.insert_fault == "after":
            raise RuntimeError("injected lock acquisition failure after commit")
        return SimpleNamespace(acknowledged=self.insert_acknowledged)

    def _delete_one(self, query):
        if self.before_delete is not None:
            self.before_delete(query)
        if self.delete_fault == "before":
            raise RuntimeError("injected lock release failure before commit")
        key = query["_id"]
        document = self.documents.get(key)
        count = int(document is not None and document.get("token") == query["token"])
        if count:
            del self.documents[key]
        if self.delete_fault == "after":
            raise RuntimeError("injected lock release failure after commit")
        return SimpleNamespace(
            acknowledged=self.delete_acknowledged,
            deleted_count=count,
        )


@pytest.fixture
def fake_fo(monkeypatch, tmp_path):
    datasets = {}
    stores = {}

    def create(**kwargs):
        result = _Dataset(**kwargs)
        result.info_save_fault = fake.next_create_info_save_fault
        fake.next_create_info_save_fault = None
        datasets[result.name] = result
        stores[result.name] = result._store
        return result

    def load(name):
        if fake.config.singleton_cache is not False:
            # A bad path through a previously loaded singleton may flush dirty
            # metadata/notes. Safe code must never request that object.
            cached = datasets[name]
            cached.cached_load_writes += 1
            cached.save()
            for sample in cached.samples:
                sample.save()
            return cached
        result = _Dataset(name, store=stores[name])
        datasets[name] = result
        return result

    def find_one(query, projection):
        assert projection == {"_id": 1, "name": 1, "info": 1, "media_type": 1, "version": 1}
        store = stores.get(query["name"])
        if store is None:
            return None
        return {
            "_id": store.identity, "name": query["name"],
            "info": deepcopy(store.persisted_info), "media_type": store.persisted_media_type,
            "version": store.version,
        }

    fake = SimpleNamespace(
        Dataset=create, dataset_exists=lambda name: name in datasets,
        load_dataset=Mock(side_effect=load), Sample=_Sample,
        ImageMetadata=lambda **kwargs: SimpleNamespace(**kwargs),
        ViewField=lambda name: _Expression("field", name),
        StringField=str, BooleanField=bool, IntField=int, FloatField=float,
        next_create_info_save_fault=None, __version__="1.fake-review",
        config=SimpleNamespace(singleton_cache=False, database_admin=False),
    )
    locks = _ImportLocks()
    database = SimpleNamespace(
        read_preference=SimpleNamespace(mode=0),
        read_concern=SimpleNamespace(level="local"),
        write_concern=SimpleNamespace(acknowledged=True),
        client=SimpleNamespace(is_mongos=False),
        lock_metadata=[{
            "name": "argus_scene_image_import_locks", "type": "collection", "options": {},
        }],
        lock_cursors=[],
    )

    def list_collections(*, filter):
        assert filter == {"name": "argus_scene_image_import_locks"}
        cursor = _MetadataCursor(deepcopy(database.lock_metadata))
        database.lock_cursors.append(cursor)
        return cursor

    def create_lock_collection(name):
        assert name == "argus_scene_image_import_locks"
        database.lock_metadata = [{"name": name, "type": "collection", "options": {}}]
        return locks

    def get_collection(name):
        assert name == "argus_scene_image_import_locks"
        return locks

    database.get_collection = Mock(side_effect=get_collection)
    database.list_collections = Mock(side_effect=list_collections)
    database.create_collection = Mock(side_effect=create_lock_collection)
    collection = SimpleNamespace(
        find_one=Mock(side_effect=find_one), database=database,
        read_preference=SimpleNamespace(mode=0),
        read_concern=SimpleNamespace(level="local"),
        write_concern=SimpleNamespace(acknowledged=True),
    )
    database.datasets = collection
    # Never inspect the workstation's real SDK config in these control-flow
    # tests. Config preflight tests below explicitly replace this fixture file.
    config_path = tmp_path / "fiftyone-test-config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FIFTYONE_CONFIG_PATH", str(config_path))
    # Each mocked boundary call represents a new independent CLI process. The
    # dedicated boundary tests below exercise the real one-attempt guard.
    sdk_import = Mock(return_value=(fake, collection))
    real_load_review_sdk = adapter._load_review_sdk
    monkeypatch.setattr(adapter, "_load_review_sdk", sdk_import)
    monkeypatch.setattr(adapter, "_dataset_import_lock", lambda name: nullcontext())
    return SimpleNamespace(
        fo=fake, datasets=datasets, stores=stores, collection=collection,
        sdk_import=sdk_import, real_load_review_sdk=real_load_review_sdk,
        database=database, locks=locks, config_path=config_path,
    )


def _decision(state, raw=None, *, error=False):
    return {
        "status": "error" if error else "evaluated", "state": state,
        "raw_state": raw or state, "reason": "decode comparison error" if error else state,
        "metrics": None if error else {"inliers": 20},
        "elapsed_ms": None if error else 3.25,
    }


@pytest.fixture
def review(tmp_path, monkeypatch):
    video = tmp_path / "original-vfr.mp4"
    video.write_bytes(b"immutable source bytes; decoding is never invoked by these fakes")
    origin = 2**53 + 1
    offsets = [0, 2, 5, 9]
    bundle = ReviewBundle.model_validate({
        "bundle_id": "bundle-1", "suite_id": "suite-1", "manifest_sha256": "a" * 64,
        "clips": [{
            "clip_id": "clip-1", "camera_id": "camera-1", "source_session_id": "session-1",
            "video_path": str(video), "video_sha256": file_sha256(video),
            "fps": 25.0, "frame_count": 4, "width": 120, "height": 80,
            "split": "test", "timestamps": [{
                "frame_index": index, "pts": origin + offset,
                "time_base_num": 1, "time_base_den": 1000,
                "timestamp_sec": float(Fraction(origin + offset, 1000)),
            } for index, offset in enumerate(offsets)],
        }],
    })
    output = tmp_path / "images"
    output.mkdir()
    rows = []
    for index, baseline, lightglue, category in (
        (0, _decision("stable"), _decision("moved"), "confirmed_disagreement"),
        (2, _decision("calibrating", "stable"), _decision("calibrating", "moved"), "raw_disagreement"),
        (3, _decision("stable"), _decision("unknown", error=True), "error"),
    ):
        error = category == "error"
        stamp = bundle.clips[0].timestamps[index]
        row = {
            "frame_index": index, "pts": stamp.pts, "time_base": "1/1000",
            "timestamp_sec": float(Fraction(stamp.pts, 1000)),
            "relative_timestamp_sec": offsets[index] / 1000,
            "frame_sha256": str(index + 1) * 64,
            "baseline": baseline, "lightglue": lightglue,
            "comparable": not error, "comparison_error": error,
            "state_disagreement": None if error else baseline["state"] != lightglue["state"],
            "raw_state_disagreement": None if error else baseline["raw_state"] != lightglue["raw_state"],
            "error_algorithms": ["lightglue"] if error else [], "review_category": category,
        }
        image = output / f"frame-{index}.png"
        image.write_bytes(f"mock PNG for source frame {index}".encode())
        rows.append({
            "clip_id": "clip-1", "camera_id": "camera-1",
            "source_video_sha256": bundle.clips[0].video_sha256,
            "frame_index": index, "frame_number": index + 1,
            "pts_exact": str(stamp.pts), "time_base_exact": "1/1000",
            "timestamp_sec": row["timestamp_sec"],
            "relative_timestamp_sec": row["relative_timestamp_sec"],
            "frame_sha256": row["frame_sha256"], "image_path": image.name,
            "image_sha256": file_sha256(image), "report_sha256": "b" * 64,
            "comparison": row,
        })
    manifest = {
        "schema_version": "argus.scene-review-images.v1", "artifact_id": "c" * 64,
        "diagnostic_only": True, "gating_effect": "none", "release_authority": False,
        "ground_truth_authority": False, "bundle_id": bundle.bundle_id,
        "suite_id": bundle.suite_id, "bundle_sha256": "d" * 64,
        "selection": "disagreements", "reports": {"clip-1": {
            "report_path": str(tmp_path / "scene-report.json"), "report_sha256": "b" * 64,
            "report_status": "partial", "profile_id": "profile-1",
            "summary": {"sampled_frames": 3, "comparison_error_count": 1, "release_authority": False},
        }},
        "decoder_evidence": {}, "images": rows,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    state = SimpleNamespace(
        bundle=bundle, manifest=manifest, path=path, comparisons={"clip-1": object()},
    )
    loader = Mock(side_effect=lambda *args, **kwargs: deepcopy(state.manifest))
    monkeypatch.setattr(adapter, "load_scene_review_images", loader)
    state.loader = loader
    return state


def _import(review, **kwargs):
    return adapter.import_scene_review_images(
        review.bundle, manifest_path=review.path, scene_comparisons=review.comparisons,
        dataset_name="vfr-images", **kwargs,
    )


def _update_manifest(review, callback):
    callback(review.manifest)
    review.manifest["artifact_id"] = hashlib.sha256(
        json.dumps(review.manifest, sort_keys=True).encode()
    ).hexdigest()
    review.path.write_text(json.dumps(review.manifest), encoding="utf-8")


def _set_review_assignment(review, *, split, session):
    payload = review.bundle.model_dump(mode="json")
    payload["clips"][0].update(split=split, source_session_id=session)
    review.bundle = ReviewBundle.model_validate(payload)


def _switch_review_clip(review, *, split="test", session="session-2", same_video=False):
    """Prepare a separate current bundle while retaining the previous dataset."""
    payload = review.bundle.model_dump(mode="json")
    clip = payload["clips"][0]
    old_id = clip["clip_id"]
    clip.update(
        clip_id="clip-2", camera_id="camera-2", split=split, source_session_id=session,
    )
    if not same_video:
        video = review.path.parent.parent / "another-original-vfr.mp4"
        video.write_bytes(b"different immutable source bytes")
        clip.update(video_path=str(video), video_sha256=file_sha256(video))
    review.bundle = ReviewBundle.model_validate(payload)
    review.comparisons = {"clip-2": object()}

    def change(manifest):
        manifest["reports"] = {"clip-2": manifest["reports"][old_id]}
        for image in manifest["images"]:
            image.update(
                clip_id="clip-2", camera_id="camera-2",
                source_video_sha256=clip["video_sha256"],
            )

    _update_manifest(review, change)


def _add_zero_selection_clip(review, *, with_report=True, split="test", session="session-2"):
    payload = review.bundle.model_dump(mode="json")
    clip = deepcopy(payload["clips"][0])
    video = review.path.parent.parent / "zero-selection-vfr.mp4"
    video.write_bytes(b"immutable source for a clip with no selected PNG")
    clip.update(
        clip_id="clip-2", camera_id="camera-2", video_path=str(video),
        video_sha256=file_sha256(video), split=split, source_session_id=session,
    )
    payload["clips"].append(clip)
    review.bundle = ReviewBundle.model_validate(payload)
    if with_report:
        review.comparisons["clip-2"] = object()

        def change(manifest):
            manifest["reports"]["clip-2"] = deepcopy(manifest["reports"]["clip-1"])
            manifest["decoder_evidence"]["clip-2"] = {"complete_decode_verified": True}

        _update_manifest(review, change)


def test_vfr_is_image_only_with_exact_original_frame_and_report_identity(fake_fo, review):
    before_video = file_sha256(review.bundle.clips[0].video_path)
    dataset = _import(review)
    assert dataset.media_type == "image"
    assert len(dataset.samples) == 3
    first, raw, error = dataset.samples
    assert first.filepath == str(review.path.parent / "frame-0.png")
    assert first["argus_frame_index"] == 0
    assert first["argus_frame_number"] == 1
    assert raw["argus_frame_index"] == 2
    assert raw["argus_frame_number"] == 3  # Not sample ordinal + 1.
    assert raw["argus_pts_exact"] == str(2**53 + 6)
    assert raw["argus_time_base_exact"] == "1/1000"
    assert raw["argus_relative_timestamp_sec"] == 0.005
    assert first.metadata.width == 120 and first.metadata.height == 80
    assert first.metadata.num_channels == 3
    assert first["argus_scene_report_sha256"] == "b" * 64
    assert first["argus_image_sha256"] == file_sha256(first.filepath)
    assert json.loads(first["argus_scene_comparison_summary_json"])["sampled_frames"] == 3
    assert json.loads(raw["argus_scene_comparison_json"])["frame_index"] == 2
    assert first["argus_scene_release_authority"] is False
    assert first["argus_scene_ground_truth_authority"] is False
    assert not {"ground_truth", "argus_score", "argus_scene_guard_usable"} & first.keys()
    assert not set(adapter._HUMAN_FIELDS) & first.keys()
    assert dataset.views["argus-scene-disagreements"].contains(first)
    assert not dataset.views["argus-scene-disagreements"].contains(raw)
    assert dataset.views["argus-scene-raw-disagreements"].contains(raw)
    assert dataset.views["argus-scene-errors"].contains(error)
    assert error["argus_scene_disagreement"] is None
    assert review.loader.call_count == 2
    assert file_sha256(review.bundle.clips[0].video_path) == before_video


def test_review_views_are_dynamic_and_normalize_missing_or_null(fake_fo, review):
    dataset = _import(review)
    sample = dataset.samples[0]
    unreviewed = dataset.views["argus-scene-unreviewed"]
    reviewed = dataset.views["argus-scene-reviewed"]
    stale = dataset.views["argus-scene-stale-review"]
    assert unreviewed.contains(sample) and not reviewed.contains(sample)
    for value in (None, ""):
        sample["argus_scene_reviewed_report_sha256"] = value
        sample["argus_scene_human_state"] = "moved"
        assert unreviewed.contains(sample) and not stale.contains(sample)
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    assert reviewed.contains(sample) and not unreviewed.contains(sample)
    sample["argus_scene_human_state"] = "approved"
    assert not reviewed.contains(sample) and unreviewed.contains(sample)
    sample["argus_scene_human_state"] = "moved"
    sample["argus_scene_reviewed_report_sha256"] = "f" * 64
    assert stale.contains(sample) and unreviewed.contains(sample)


def test_report_refresh_keeps_human_notes_but_new_hash_is_stale(fake_fo, review):
    dataset = _import(review)
    sample = dataset.samples[0]
    original_key = sample["argus_scene_image_key"]
    sample["argus_scene_human_state"] = "moved"
    sample["argus_scene_human_note"] = "operator saw camera movement"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    write_offset = len(sample.writes)

    def change(manifest):
        manifest["reports"]["clip-1"]["report_sha256"] = "e" * 64
        manifest["reports"]["clip-1"]["summary"]["revision"] = 2
        for image in manifest["images"]:
            image["report_sha256"] = "e" * 64

    _update_manifest(review, change)
    previous = dataset
    dataset = _import(review)
    assert dataset is not previous and dataset.samples[0] is not sample
    sample.reload()
    assert len(dataset.samples) == 3
    assert sample["argus_scene_image_key"] == original_key
    assert sample["argus_scene_human_note"] == "operator saw camera movement"
    assert sample["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert dataset.views["argus-scene-stale-review"].contains(sample)
    assert not dataset.views["argus-scene-reviewed"].contains(sample)
    for fields in sample.writes[write_offset:]:
        assert not set(adapter._HUMAN_FIELDS) & fields


def test_review_is_bound_to_report_content_hash_across_a_b_identical_a(fake_fo, review):
    dataset = _import(review)
    sample = dataset.samples[0]
    original = deepcopy(review.manifest)
    sample["argus_scene_human_state"] = "moved"
    sample["argus_scene_human_note"] = "decision applies to the exact original report A"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    offset = len(sample.writes)

    def report_b(manifest):
        manifest["reports"]["clip-1"]["report_sha256"] = "e" * 64
        for image in manifest["images"]:
            image["report_sha256"] = "e" * 64

    _update_manifest(review, report_b)
    dataset = _import(review)
    sample.reload()
    assert dataset.views["argus-scene-stale-review"].contains(sample)
    assert not dataset.views["argus-scene-reviewed"].contains(sample)
    review.manifest = original
    review.path.write_text(json.dumps(original), encoding="utf-8")
    dataset = _import(review)
    sample.reload()
    assert sample["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert sample["argus_scene_human_note"] == "decision applies to the exact original report A"
    assert dataset.views["argus-scene-reviewed"].contains(sample)
    assert not dataset.views["argus-scene-stale-review"].contains(sample)
    assert all(not set(adapter._HUMAN_FIELDS) & fields for fields in sample.writes[offset:])


def test_absent_clip_returns_with_new_report_and_keeps_old_confirmation_stale(fake_fo, review):
    dataset = _import(review)
    historical = dataset.samples[0]
    historical["argus_scene_human_state"] = "moved"
    historical["argus_scene_human_note"] = "review A survives the clip being absent"
    historical["argus_scene_reviewed_report_sha256"] = "b" * 64
    historical.save()
    offset = len(historical.writes)
    original_bundle = review.bundle
    original_manifest = deepcopy(review.manifest)
    original_comparisons = dict(review.comparisons)
    _switch_review_clip(review)
    dataset = _import(review)
    assert not dataset.views["argus-current"].contains(historical)
    assert not dataset.views["argus-scene-reviewed"].contains(historical)
    assert len(historical.writes) == offset
    review.bundle = original_bundle
    review.manifest = original_manifest
    review.comparisons = original_comparisons

    def report_b(manifest):
        manifest["reports"]["clip-1"]["report_sha256"] = "e" * 64
        for image in manifest["images"]:
            image["report_sha256"] = "e" * 64

    _update_manifest(review, report_b)
    dataset = _import(review)
    historical.reload()
    assert historical["argus_scene_human_note"] == "review A survives the clip being absent"
    assert historical["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert dataset.views["argus-current"].contains(historical)
    assert dataset.views["argus-scene-stale-review"].contains(historical)
    assert not dataset.views["argus-scene-reviewed"].contains(historical)
    assert all(not set(adapter._HUMAN_FIELDS) & fields for fields in historical.writes[offset:])


@pytest.mark.parametrize("failure", ["sample", "view", "metadata"])
def test_new_report_failed_refresh_and_retry_preserve_saved_review_notes(fake_fo, review, failure):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_state"] = "moved"
    sample["argus_scene_human_note"] = "saved before interrupted report B import"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    offset = len(sample.writes)

    def report_b(manifest):
        manifest["reports"]["clip-1"]["report_sha256"] = "e" * 64
        for image in manifest["images"]:
            image["report_sha256"] = "e" * 64

    _update_manifest(review, report_b)
    if failure == "sample":
        dataset.fail_next_sample_save = True
    elif failure == "view":
        dataset.fail_saved_view = "argus-scene-unreviewed"
    else:
        dataset.fail_complete_save = True
    with pytest.raises(RuntimeError, match="injected"):
        _import(review)
    sample.reload()
    assert sample["argus_scene_human_note"] == "saved before interrupted report B import"
    assert sample["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert not dataset.views["argus-current"].contains(sample)
    dataset = _import(review)
    sample.reload()
    assert sample["argus_scene_human_note"] == "saved before interrupted report B import"
    assert sample["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert dataset.views["argus-scene-stale-review"].contains(sample)
    assert not dataset.views["argus-scene-reviewed"].contains(sample)
    assert all(not set(adapter._HUMAN_FIELDS) & fields for fields in sample.writes[offset:])


def test_removed_selection_is_historical_and_notes_survive_restore(fake_fo, review):
    dataset = _import(review)
    removed = dataset.samples[1]
    removed["argus_scene_human_note"] = "keep this original-frame note"
    removed.save()
    original = deepcopy(review.manifest)
    _update_manifest(review, lambda manifest: manifest.update(images=manifest["images"][:1]))
    dataset = _import(review)
    assert len(dataset.samples) == 3
    assert not dataset.views["argus-current"].contains(removed)
    assert not dataset.views["argus-scene-unreviewed"].contains(removed)
    review.manifest = original
    review.path.write_text(json.dumps(original), encoding="utf-8")
    dataset = _import(review)
    removed.reload()
    assert len(dataset.samples) == 3
    assert dataset.views["argus-current"].contains(removed)
    assert removed["argus_scene_human_note"] == "keep this original-frame note"


def test_partial_selection_can_assign_split_and_session_then_repeat_with_stale_review(
    fake_fo, review,
):
    _set_review_assignment(review, split="unassigned", session=None)
    dataset = _import(review)
    selected, historical = dataset.samples[:2]
    all_images = deepcopy(review.manifest["images"])
    for sample in (selected, historical):
        sample["argus_scene_human_note"] = f"preserve frame {sample['argus_frame_index']}"
        sample["argus_scene_human_state"] = "moved"
        sample["argus_scene_reviewed_report_sha256"] = "b" * 64
        sample.save()
    selected_offset, historical_offset = len(selected.writes), len(historical.writes)
    _set_review_assignment(review, split="test", session="session-1")

    def refresh(manifest):
        manifest["images"] = manifest["images"][:1]
        manifest["reports"]["clip-1"]["report_sha256"] = "e" * 64
        manifest["images"][0]["report_sha256"] = "e" * 64

    _update_manifest(review, refresh)
    dataset = _import(review)
    previous = dataset
    dataset = _import(review)
    assert dataset is not previous
    selected.reload()
    historical.reload()
    assert selected["argus_split"] == "test"
    assert selected["argus_source_session_id"] == "session-1"
    assert historical["argus_split"] == "unassigned"
    assert historical["argus_source_session_id"] is None
    assert historical["argus_scene_human_note"] == "preserve frame 2"
    assert len(historical.writes) == historical_offset
    assert dataset.views["argus-scene-stale-review"].contains(selected)
    assert not dataset.views["argus-scene-reviewed"].contains(selected)
    assert not dataset.views["argus-current"].contains(historical)
    assert selected["argus_scene_human_note"] == "preserve frame 0"
    assert all(
        not set(adapter._HUMAN_FIELDS) & fields for fields in selected.writes[selected_offset:]
    )

    for image in all_images:
        image["report_sha256"] = "e" * 64
    _update_manifest(review, lambda manifest: manifest.update(images=all_images))
    dataset = _import(review)
    historical.reload()
    assert historical["argus_source_session_id"] == "session-1"
    assert historical["argus_scene_human_note"] == "preserve frame 2"
    assert dataset.views["argus-scene-stale-review"].contains(historical)


@pytest.mark.parametrize("failure", ["sample", "view", "metadata"])
def test_partial_assignment_import_failure_can_retry_without_rewriting_history(
    fake_fo, review, failure,
):
    _set_review_assignment(review, split="unassigned", session=None)
    dataset = _import(review)
    first, second, historical = dataset.samples
    first["argus_scene_human_note"] = "manual note survives interrupted assignment"
    first.save()
    _set_review_assignment(review, split="test", session="session-1")
    _update_manifest(review, lambda manifest: manifest.update(images=manifest["images"][:2]))
    if failure == "sample":
        def fail_second_save(sample):
            if sample.get_field("argus_scene_image_key") == second["argus_scene_image_key"]:
                dataset.before_sample_save = None
                raise RuntimeError("injected failure after the first assignment")

        dataset.before_sample_save = fail_second_save
    elif failure == "view":
        dataset.fail_saved_view = "argus-scene-unreviewed"
    else:
        dataset.fail_complete_save = True
    with pytest.raises(RuntimeError, match="injected"):
        _import(review)
    first.reload()
    assert first["argus_source_session_id"] == "session-1"
    assert historical["argus_source_session_id"] is None
    dataset = _import(review)
    first.reload()
    assert dataset.info[adapter._INFO_KEY]["state"] == "complete"
    assert first["argus_scene_human_note"] == "manual note survives interrupted assignment"
    assert not dataset.views["argus-current"].contains(historical)


@pytest.mark.parametrize("conflict", ["session", "split"])
def test_concrete_historical_source_assignments_cannot_be_normalized_away(
    fake_fo, review, conflict,
):
    dataset = _import(review)
    historical = dataset.samples[1]
    if conflict == "session":
        historical["argus_source_session_id"] = "another-session"
    else:
        historical["argus_split"] = "train"
    historical.save()
    generation = dataset.info[adapter._INFO_KEY]["generation"]
    _update_manifest(review, lambda manifest: manifest.update(images=manifest["images"][:1]))
    with pytest.raises(
        ValueError, match="conflicting (clip capture-session identities|assigned clip splits)",
    ):
        _import(review)
    assert dataset.info[adapter._INFO_KEY]["generation"] == generation


@pytest.mark.parametrize("mutation", ["split", "session", "remove-session"])
def test_current_bundle_cannot_remove_or_change_a_partial_historical_assignment(
    fake_fo, review, mutation,
):
    _set_review_assignment(review, split="unassigned", session=None)
    _import(review)
    _set_review_assignment(review, split="test", session="session-1")
    _update_manifest(review, lambda manifest: manifest.update(images=manifest["images"][:1]))
    _import(review)
    if mutation == "split":
        _set_review_assignment(review, split="unassigned", session="session-1")
    elif mutation == "session":
        _set_review_assignment(review, split="test", session="another-session")
    else:
        _set_review_assignment(review, split="unassigned", session=None)
    with pytest.raises(ValueError, match="immutable"):
        _import(review)


def test_omitted_clip_with_legal_ancestor_history_does_not_block_another_clip(fake_fo, review):
    _set_review_assignment(review, split="unassigned", session=None)
    dataset = _import(review)
    _set_review_assignment(review, split="test", session="session-1")
    _update_manifest(review, lambda manifest: manifest.update(images=manifest["images"][:1]))
    dataset = _import(review)
    previous = dataset.samples[1]
    previous_offset = len(previous.writes)
    _switch_review_clip(review)
    dataset = _import(review)
    assert len(dataset.samples) == 4
    assert len(previous.writes) == previous_offset
    assert previous["argus_split"] == "unassigned"
    assert not dataset.views["argus-current"].contains(previous)
    assert dataset.views["argus-current"].contains(dataset.samples[-1])


@pytest.mark.parametrize("shared_identity", ["session", "video"])
def test_normalized_history_still_blocks_cross_clip_split_leakage(fake_fo, review, shared_identity):
    _set_review_assignment(review, split="unassigned", session=None)
    dataset = _import(review)
    _set_review_assignment(review, split="test", session="session-1")
    _update_manifest(review, lambda manifest: manifest.update(images=manifest["images"][:1]))
    dataset = _import(review)
    _switch_review_clip(
        review, split="train", same_video=shared_identity == "video",
        session="session-1" if shared_identity == "session" else "another-session",
    )
    generation = dataset.info[adapter._INFO_KEY]["generation"]
    with pytest.raises(ValueError, match="historical video/capture session crosses"):
        _import(review)
    assert len(dataset.samples) == 3
    assert dataset.info[adapter._INFO_KEY]["generation"] == generation


@pytest.mark.parametrize("with_report", [True, False])
@pytest.mark.parametrize("mutation", ["split", "session", "camera", "video", "geometry", "pts"])
def test_zero_selection_source_is_persistently_bound_even_without_a_report(
    fake_fo, review, with_report, mutation,
):
    _add_zero_selection_clip(review, with_report=with_report)
    dataset = _import(review)
    registry = deepcopy(dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY])
    assert registry["clips"]["clip-2"]["split"] == "test"
    assert registry["clips"]["clip-2"]["source_session_id"] == "session-2"
    assert all(sample["argus_clip_id"] == "clip-1" for sample in dataset.samples)
    payload = review.bundle.model_dump(mode="json")
    clip = payload["clips"][1]
    if mutation == "split":
        clip["split"] = "train"
    elif mutation == "session":
        clip["source_session_id"] = "reassigned-session"
    elif mutation == "camera":
        clip["camera_id"] = "reassigned-camera"
    elif mutation == "video":
        clip["video_sha256"] = "f" * 64
    elif mutation == "geometry":
        clip["width"] += 1
    else:
        stamp = clip["timestamps"][1]
        stamp["pts"] += 1
        stamp["timestamp_sec"] = float(Fraction(stamp["pts"], 1000))
    review.bundle = ReviewBundle.model_validate(payload)
    writes = [len(sample.writes) for sample in dataset.samples]
    with pytest.raises(ValueError, match="immutable|source/frame mapping changed"):
        _import(review)
    assert dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == registry
    assert [len(sample.writes) for sample in dataset.samples] == writes


@pytest.mark.parametrize("shared_identity", ["video", "session"])
def test_omitted_zero_selection_source_retains_cross_clip_split_binding(
    fake_fo, review, shared_identity,
):
    _add_zero_selection_clip(review)
    dataset = _import(review)
    old_source = review.bundle.clips[1]
    payload = review.bundle.model_dump(mode="json")
    payload["clips"] = payload["clips"][:1]
    review.bundle = ReviewBundle.model_validate(payload)
    review.comparisons.pop("clip-2")
    _update_manifest(review, lambda manifest: manifest["reports"].pop("clip-2"))
    _import(review)
    assert "clip-2" in dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]["clips"]
    payload = review.bundle.model_dump(mode="json")
    clip = old_source.model_dump(mode="json")
    clip.update(clip_id="clip-3", camera_id="camera-3", split="train")
    if shared_identity == "video":
        clip["source_session_id"] = "session-3"
    else:
        clip["video_sha256"] = "f" * 64
    payload["clips"].append(clip)
    review.bundle = ReviewBundle.model_validate(payload)
    writes = [len(sample.writes) for sample in dataset.samples]
    with pytest.raises(ValueError, match="historical video/capture session crosses"):
        _import(review)
    assert [len(sample.writes) for sample in dataset.samples] == writes


def test_failed_empty_dataset_import_keeps_source_registry_before_any_sample_writes(
    fake_fo, review, monkeypatch,
):
    _add_zero_selection_clip(review)
    ensure_schema = adapter._ensure_schema

    def fail_schema(dataset, fo):
        persisted = dataset.persisted_info[adapter._INFO_KEY]
        assert persisted["state"] == "importing"
        assert persisted[adapter._SOURCE_REGISTRY_KEY]["clips"]["clip-2"]["split"] == "test"
        raise RuntimeError("injected schema failure before samples")

    monkeypatch.setattr(adapter, "_ensure_schema", fail_schema)
    with pytest.raises(RuntimeError, match="before samples"):
        _import(review)
    dataset = fake_fo.datasets["vfr-images"]
    assert dataset.samples == []
    assert dataset.persisted_info[adapter._INFO_KEY]["state"] == "failed"
    registry = dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]
    assert registry["clips"]["clip-2"]["source_session_id"] == "session-2"
    monkeypatch.setattr(adapter, "_ensure_schema", ensure_schema)
    payload = review.bundle.model_dump(mode="json")
    payload["clips"][1].update(split="train", source_session_id="another-session")
    review.bundle = ReviewBundle.model_validate(payload)
    with pytest.raises(ValueError, match="immutable"):
        _import(review)
    assert dataset.samples == []
    assert dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == registry


@pytest.mark.parametrize("phase", ["before", "after"])
def test_first_owner_commit_also_binds_full_source_registry_or_leaves_unadoptable_orphan(
    fake_fo, review, phase,
):
    _add_zero_selection_clip(review, with_report=False)
    fake_fo.fo.next_create_info_save_fault = ("failed", phase)
    with pytest.raises(RuntimeError, match=f"metadata failure {phase} commit"):
        _import(review)
    dataset = fake_fo.datasets["vfr-images"]
    assert dataset.samples == []
    assert dataset.info_save_attempts == 1
    if phase == "before":
        assert dataset.persisted_info == {}
        assert dataset.info_writes == []
        # A dirty local owner marker is not authority to adopt this orphan.
        # Raw metadata refuses it without loading or rewriting the old cache.
        cached_info = deepcopy(dataset.info)
        loads = fake_fo.fo.load_dataset.call_count
        with pytest.raises(ValueError, match="refusing unrelated/native-video dataset"):
            _import(review)
        assert dataset.info == cached_info
        assert dataset.persisted_info == {}
        assert dataset.info_save_attempts == 1
        assert fake_fo.fo.load_dataset.call_count == loads
        assert dataset.reload_count == dataset.hard_refresh_writes == 0
        return
    info = dataset.persisted_info[adapter._INFO_KEY]
    assert info["owner"] == adapter._OWNER and info["state"] == "failed"
    assert info[adapter._SOURCE_REGISTRY_REQUIRED_KEY] is True
    registry = deepcopy(info[adapter._SOURCE_REGISTRY_KEY])
    assert set(registry["clips"]) == {"clip-1", "clip-2"}
    assert registry["clips"]["clip-2"]["source_session_id"] == "session-2"
    original = review.bundle
    payload = original.model_dump(mode="json")
    payload["clips"][1]["source_session_id"] = "another-session"
    review.bundle = ReviewBundle.model_validate(payload)
    # Losing the caller's acknowledgement cannot release a committed binding.
    with pytest.raises(ValueError, match="capture-session identity is immutable"):
        _import(review)
    assert dataset.samples == []
    assert dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == registry
    review.bundle = original
    dataset = _import(review)
    assert dataset.info[adapter._INFO_KEY]["state"] == "complete"


@pytest.mark.parametrize("phase", ["before", "after"])
def test_existing_importing_registry_commit_failure_new_process_reads_only_persisted_assignments(
    fake_fo, review, phase,
):
    dataset = _import(review)
    historical = dataset.samples[0]
    historical["argus_scene_human_note"] = "keep notes across uncertain metadata commits"
    historical.save()
    old_registry = deepcopy(dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY])
    offsets = [len(sample.writes) for sample in dataset.samples]
    _add_zero_selection_clip(review, with_report=False)
    dataset.info_save_fault = ("importing", phase)
    with pytest.raises(RuntimeError, match=f"metadata failure {phase} commit"):
        _import(review)
    assert [len(sample.writes) for sample in dataset.samples] == offsets
    persisted = deepcopy(dataset.persisted_info[adapter._INFO_KEY])
    if phase == "before":
        assert persisted["state"] == "complete"
        assert persisted[adapter._SOURCE_REGISTRY_KEY] == old_registry
    else:
        assert persisted["state"] == "importing"
        assert persisted[adapter._SOURCE_REGISTRY_KEY]["clips"]["clip-2"]["split"] == "test"
        original = review.bundle
        payload = original.model_dump(mode="json")
        payload["clips"][1]["source_session_id"] = "another-session"
        review.bundle = ReviewBundle.model_validate(payload)
        with pytest.raises(ValueError, match="capture-session identity is immutable"):
            _import(review)
        assert [len(sample.writes) for sample in dataset.samples] == offsets
        assert dataset.persisted_info[adapter._INFO_KEY] == persisted
        review.bundle = original
    # In-memory remnants of either failure must not erase durable history or
    # invent a capture-session binding that never reached the database.
    dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]["clips"]["clip-2"] = {
        "video_sha256": "f" * 64, "camera_id": "dirty-camera",
        "media_identity": "e" * 64, "source_session_id": "dirty-session", "split": "train",
    }
    dataset = _import(review)
    historical.reload()
    assert historical["argus_scene_human_note"] == "keep notes across uncertain metadata commits"
    registry = dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]
    assert registry["clips"]["clip-1"] == old_registry["clips"]["clip-1"]
    assert registry["clips"]["clip-2"]["source_session_id"] == "session-2"
    assert dataset.info[adapter._INFO_KEY]["state"] == "complete"
    assert all(not set(adapter._HUMAN_FIELDS) & fields for fields in historical.writes[offsets[0]:])


def test_registry_still_binds_sources_after_all_image_samples_are_removed(fake_fo, review):
    _add_zero_selection_clip(review)
    dataset = _import(review)
    dataset._store.records.clear()
    dataset.samples.clear()
    payload = review.bundle.model_dump(mode="json")
    payload["clips"][1]["source_session_id"] = "another-session"
    review.bundle = ReviewBundle.model_validate(payload)
    with pytest.raises(ValueError, match="capture-session identity is immutable"):
        _import(review)
    assert dataset.samples == []


def test_legacy_dataset_cannot_promote_partial_sample_history_to_complete_registry(fake_fo, review):
    _set_review_assignment(review, split="unassigned", session=None)
    dataset = _import(review)
    _set_review_assignment(review, split="test", session="session-1")
    _update_manifest(review, lambda manifest: manifest.update(images=manifest["images"][:1]))
    _import(review)
    dataset.info[adapter._INFO_KEY].pop(adapter._SOURCE_REGISTRY_KEY)
    dataset.info[adapter._INFO_KEY].pop(adapter._SOURCE_REGISTRY_REQUIRED_KEY)
    dataset.save()
    historical = dataset.samples[1]
    historical["argus_scene_human_note"] = "never rewrite legacy operator notes"
    historical.save()
    persisted_info = deepcopy(dataset.persisted_info)
    attempts = dataset.info_save_attempts
    offsets = [len(sample.writes) for sample in dataset.samples]
    with pytest.raises(ValueError, match="legacy/incomplete source registry.*new image dataset"):
        _import(review)
    assert dataset.info == persisted_info
    assert dataset.persisted_info == persisted_info
    assert dataset.info_save_attempts == attempts
    assert historical["argus_split"] == "unassigned"
    assert historical["argus_scene_human_note"] == "never rewrite legacy operator notes"
    assert [len(sample.writes) for sample in dataset.samples] == offsets


@pytest.mark.parametrize("history", ["empty", "zero-with-report", "zero-without-report", "absent-zero"])
def test_legacy_sources_are_unprovable_even_when_surviving_samples_are_valid(
    fake_fo, review, history,
):
    if history == "empty":
        _update_manifest(review, lambda manifest: manifest.update(images=[]))
    else:
        _add_zero_selection_clip(review, with_report=history != "zero-without-report")
    dataset = _import(review)
    if history == "absent-zero":
        payload = review.bundle.model_dump(mode="json")
        payload["clips"] = payload["clips"][:1]
        review.bundle = ReviewBundle.model_validate(payload)
        review.comparisons.pop("clip-2")
        _update_manifest(review, lambda manifest: manifest["reports"].pop("clip-2"))
        _import(review)
    dataset.info[adapter._INFO_KEY].pop(adapter._SOURCE_REGISTRY_KEY)
    dataset.info[adapter._INFO_KEY].pop(adapter._SOURCE_REGISTRY_REQUIRED_KEY)
    dataset.save()
    persisted_info = deepcopy(dataset.persisted_info)
    attempts = dataset.info_save_attempts
    offsets = [len(sample.writes) for sample in dataset.samples]
    with pytest.raises(ValueError, match="legacy/incomplete source registry"):
        _import(review)
    assert dataset.info == persisted_info
    assert dataset.persisted_info == persisted_info
    assert dataset.info_save_attempts == attempts
    assert [len(sample.writes) for sample in dataset.samples] == offsets
    if history == "empty":
        assert dataset.samples == []


def test_shaped_v1_registry_cannot_claim_complete_history_after_old_partial_migration(
    fake_fo, review,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "notes from a legacy dataset remain untouched"
    sample.save()
    info = dataset.info[adapter._INFO_KEY]
    info[adapter._SOURCE_REGISTRY_KEY]["schema_version"] = "argus.scene-review-image-sources.v1"
    assert info[adapter._SOURCE_REGISTRY_REQUIRED_KEY] is True
    dataset.save()
    persisted = deepcopy(dataset.persisted_info)
    attempts = dataset.info_save_attempts
    offsets = [len(item.writes) for item in dataset.samples]
    with pytest.raises(ValueError, match="legacy/incomplete v1 source registry.*new image dataset"):
        _import(review)
    assert dataset.persisted_info == persisted
    assert dataset.info == persisted
    assert dataset.info_save_attempts == attempts
    assert [len(item.writes) for item in dataset.samples] == offsets
    assert sample["argus_scene_human_note"] == "notes from a legacy dataset remain untouched"


@pytest.mark.parametrize("legacy", ["missing", "v1"])
def test_known_legacy_registry_is_rejected_before_reload_preserving_dirty_notes(
    fake_fo, review, legacy,
):
    dataset = _import(review)
    info = dataset.info[adapter._INFO_KEY]
    if legacy == "missing":
        info.pop(adapter._SOURCE_REGISTRY_KEY)
        info.pop(adapter._SOURCE_REGISTRY_REQUIRED_KEY)
    else:
        info[adapter._SOURCE_REGISTRY_KEY]["schema_version"] = "argus.scene-review-image-sources.v1"
    dataset.save()
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "unsaved legacy notes must not be hard-refreshed"
    cached = deepcopy(dict(sample))
    dirty = set(sample.dirty)
    persisted = deepcopy(dataset.persisted_info)
    reloads = dataset.reload_count
    attempts = dataset.info_save_attempts
    offsets = [len(item.writes) for item in dataset.samples]
    loads = fake_fo.fo.load_dataset.call_count
    with pytest.raises(ValueError, match="legacy/incomplete"):
        _import(review)
    assert dataset.reload_count == reloads
    assert dataset.info_save_attempts == attempts
    assert dataset.persisted_info == persisted
    assert dict(sample) == cached and sample.dirty == dirty
    assert [len(item.writes) for item in dataset.samples] == offsets
    assert fake_fo.fo.load_dataset.call_count == loads
    assert dataset.hard_refresh_writes == dataset.cached_load_writes == 0


@pytest.mark.parametrize("damage", [
    "null", "empty", "empty-clips", "schema", "bool-sha", "bool-media", "bool-split",
    "bool-session", "extra-fields", "invalid-clip", "missing-registry", "false-required",
    "missing-required",
])
def test_malformed_durable_registry_fails_closed_before_sample_writes(fake_fo, review, damage):
    dataset = _import(review)
    info = dataset.info[adapter._INFO_KEY]
    registry = info[adapter._SOURCE_REGISTRY_KEY]
    if damage == "null":
        info[adapter._SOURCE_REGISTRY_KEY] = None
    elif damage == "empty":
        info[adapter._SOURCE_REGISTRY_KEY] = {}
    elif damage == "empty-clips":
        registry["clips"] = {}
    elif damage == "schema":
        registry["schema_version"] = "unrecognized-schema"
    elif damage == "missing-registry":
        info.pop(adapter._SOURCE_REGISTRY_KEY)
    elif damage == "false-required":
        info[adapter._SOURCE_REGISTRY_REQUIRED_KEY] = False
    elif damage == "missing-required":
        info.pop(adapter._SOURCE_REGISTRY_REQUIRED_KEY)
    elif damage == "invalid-clip":
        registry["clips"][" bad identity "] = registry["clips"].pop("clip-1")
    else:
        source = registry["clips"]["clip-1"]
        key = {
            "bool-sha": "video_sha256", "bool-media": "media_identity",
            "bool-split": "split", "bool-session": "source_session_id",
            "extra-fields": "unrecognized-field",
        }[damage]
        source[key] = True
    dataset.save()
    writes = [len(sample.writes) for sample in dataset.samples]
    with pytest.raises(ValueError, match="source registry"):
        _import(review)
    assert [len(sample.writes) for sample in dataset.samples] == writes


def test_current_source_registry_budget_is_checked_before_optional_sdk(
    fake_fo, review, monkeypatch,
):
    monkeypatch.setattr(adapter, "_MAX_SOURCE_REGISTRY_BYTES", 1)
    with pytest.raises(ValueError, match="metadata budget"):
        _import(review)
    fake_fo.sdk_import.assert_not_called()
    assert fake_fo.datasets == {}


def test_merged_source_registry_budget_is_checked_before_sample_writes(
    fake_fo, review, monkeypatch,
):
    dataset = _import(review)
    old_registry = deepcopy(dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY])
    writes = [len(sample.writes) for sample in dataset.samples]
    _switch_review_clip(review)
    clip = review.bundle.clips[0]
    current_registry = adapter._source_registry({clip.clip_id: (
        clip.video_sha256, clip.camera_id, clip.source_session_id,
        clip.split, adapter._media_identity(clip),
    )})
    budget = max(len(adapter._json(value).encode("utf-8")) for value in (
        old_registry, current_registry,
    ))
    monkeypatch.setattr(adapter, "_MAX_SOURCE_REGISTRY_BYTES", budget)
    with pytest.raises(ValueError, match="metadata budget"):
        _import(review)
    assert [len(sample.writes) for sample in dataset.samples] == writes
    assert dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == old_registry


def test_raw_ledger_and_fresh_objects_ignore_dirty_cached_zero_selection_bindings(fake_fo, review):
    _add_zero_selection_clip(review)
    dataset = _import(review)
    registry = deepcopy(dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY])
    # A stale or dirty object must neither erase the durable zero-image binding
    # nor be hard-refreshed as a side effect of a rejected new CLI attempt.
    dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]["clips"].pop("clip-2")
    payload = review.bundle.model_dump(mode="json")
    payload["clips"][1].update(split="train", source_session_id="another-session")
    review.bundle = ReviewBundle.model_validate(payload)
    reloads = dataset.reload_count
    cached_info = deepcopy(dataset.info)
    loads = fake_fo.fo.load_dataset.call_count
    with pytest.raises(ValueError, match="immutable"):
        _import(review)
    assert dataset.reload_count == reloads
    assert dataset.info == cached_info
    assert dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == registry
    assert fake_fo.fo.load_dataset.call_count == loads + 1
    assert fake_fo.datasets["vfr-images"] is not dataset
    assert dataset.hard_refresh_writes == dataset.cached_load_writes == 0


def test_refresh_does_not_erase_manual_review_saved_after_sample_was_cached(fake_fo, review):
    dataset = _import(review)
    sample = dataset.samples[0]
    injected = False
    write_offset = len(sample.writes)

    def reviewer_saves(cached):
        nonlocal injected
        if cached.get_field("argus_scene_image_key") == sample["argus_scene_image_key"] and not injected:
            assert cached is not sample
            injected = True
            assert cached.get_field("argus_scene_human_note") is None
            cached.stored.update({
                "argus_scene_human_note": "saved concurrently by another App client",
                "argus_scene_human_state": "moved",
                "argus_scene_reviewed_report_sha256": "b" * 64,
            })

    dataset.before_sample_save = reviewer_saves
    _import(review)
    sample.reload()
    assert injected
    assert sample["argus_scene_human_note"] == "saved concurrently by another App client"
    assert dataset.views["argus-scene-reviewed"].contains(sample)
    assert all(not set(adapter._HUMAN_FIELDS) & fields for fields in sample.writes[write_offset:])


def test_complete_clip_mapping_is_bound_including_unselected_timestamps(fake_fo, review):
    dataset = _import(review)
    old_generation = dataset.info[adapter._INFO_KEY]["generation"]
    payload = review.bundle.model_dump(mode="json")
    # Original index 1 has no image sample. It still belongs to the immutable
    # source timeline and cannot be silently altered by a sampling refresh.
    stamp = payload["clips"][0]["timestamps"][1]
    stamp["pts"] -= 1
    stamp["timestamp_sec"] = float(Fraction(stamp["pts"], 1000))
    review.bundle = ReviewBundle.model_validate(payload)
    with pytest.raises(ValueError, match="source/frame mapping changed"):
        _import(review)
    assert dataset.info[adapter._INFO_KEY]["generation"] == old_generation


@pytest.mark.parametrize("frame_number", [True, 1.0, "1"])
def test_original_frame_number_rejects_equal_noninteger_history(fake_fo, review, frame_number):
    dataset = _import(review)
    first = dataset.samples[0]
    first["argus_frame_number"] = frame_number
    first.save()
    with pytest.raises(ValueError, match="original-frame number was changed"):
        _import(review)


def test_media_identity_is_computed_once_per_clip_not_per_selected_image(fake_fo, review, monkeypatch):
    original = adapter._media_identity
    identity = Mock(wraps=original)
    monkeypatch.setattr(adapter, "_media_identity", identity)
    _import(review)
    assert identity.call_count == len(review.bundle.clips)
    identity.reset_mock()
    _import(review)
    assert identity.call_count == len(review.bundle.clips)


@pytest.mark.parametrize("kind", ["unrelated", "native-video", "other-suite", "other-host"])
def test_existing_datasets_require_separate_image_ownership(fake_fo, review, kind):
    dataset = _import(review)
    if kind == "unrelated":
        dataset.info.clear()
    elif kind == "native-video":
        dataset.media_type = "video"
    elif kind == "other-suite":
        dataset.info[adapter._INFO_KEY]["suite_id"] = "another-suite"
    else:
        dataset.info[adapter._INFO_KEY]["importing_host"] = "another-host"
    dataset.save()
    with pytest.raises((ValueError, RuntimeError)):
        _import(review)


@pytest.mark.parametrize("kind", ["unrelated", "native-video", "other-suite", "other-host"])
def test_unrelated_dirty_cache_is_not_loaded_flushed_or_refreshed_by_fresh_process(
    fake_fo, review, kind,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "unsaved edits in an unrelated cached dataset"
    if kind == "unrelated":
        dataset.info.clear()
    elif kind == "native-video":
        dataset.media_type = "video"
    elif kind == "other-suite":
        dataset.info[adapter._INFO_KEY]["suite_id"] = "another-suite"
    else:
        dataset.info[adapter._INFO_KEY]["importing_host"] = "another-host"
    cached = deepcopy(dict(sample))
    dirty = set(sample.dirty)
    info = deepcopy(dataset.info)
    persisted = deepcopy(dataset.persisted_info)
    reloads = sample.reload_count
    fresh = _import(review)
    assert fresh is not dataset and fresh.samples[0] is not sample
    assert dataset.reload_count == 0 and sample.reload_count == reloads
    assert dict(sample) == cached and sample.dirty == dirty
    assert dataset.info == info
    assert fresh.info[adapter._INFO_KEY]["state"] == "complete"
    assert dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == (
        persisted[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]
    )
    assert "argus_scene_human_note" not in sample.stored
    assert dataset.hard_refresh_writes == dataset.cached_load_writes == 0


@pytest.mark.parametrize("kind", ["unrelated", "native-video", "other-suite", "other-host"])
def test_raw_ownership_is_checked_before_sdk_load_or_hard_refresh(fake_fo, review, kind):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "unsaved draft survives persisted ownership refusal"
    draft = deepcopy(dict(sample))
    dirty = set(sample.dirty)
    sample_reloads = sample.reload_count
    if kind == "unrelated":
        dataset.persisted_info.clear()
    elif kind == "native-video":
        dataset.persisted_media_type = "video"
    elif kind == "other-suite":
        dataset.persisted_info[adapter._INFO_KEY]["suite_id"] = "another-suite"
    else:
        dataset.persisted_info[adapter._INFO_KEY]["importing_host"] = "another-host"
    # Cache still appears owned. Raw persisted metadata must reject the target
    # before even requesting an SDK object, whose loader could save or migrate.
    assert dataset.info[adapter._INFO_KEY]["owner"] == adapter._OWNER
    assert dataset.media_type == "image"
    reloads = dataset.reload_count
    attempts = dataset.info_save_attempts
    offsets = [len(sample.writes) for sample in dataset.samples]
    persisted = deepcopy(dataset.persisted_info)
    loads = fake_fo.fo.load_dataset.call_count
    with pytest.raises((ValueError, RuntimeError)):
        _import(review)
    assert dataset.reload_count == reloads
    assert fake_fo.fo.load_dataset.call_count == loads
    assert dataset.info_save_attempts == attempts
    assert [len(sample.writes) for sample in dataset.samples] == offsets
    assert dataset.persisted_info == persisted
    assert dataset.hard_refresh_writes == dataset.cached_load_writes == 0
    assert dict(sample) == draft and sample.dirty == dirty
    assert sample.reload_count == sample_reloads


def test_original_pixel_identity_cannot_change_under_preserved_manual_notes(fake_fo, review):
    dataset = _import(review)
    sample = dataset.samples[0]
    _update_manifest(review, lambda manifest: manifest["images"][0].update(frame_sha256="f" * 64))
    with pytest.raises(ValueError, match="original-frame pixels changed"):
        _import(review)
    assert sample["argus_import_state"] == "complete"


@pytest.mark.parametrize("failure", ["sample", "view", "metadata", "revalidation"])
def test_failures_quarantine_touched_samples_and_mark_generation_failed(fake_fo, review, failure):
    dataset = _import(review)
    if failure == "sample":
        dataset.fail_next_sample_save = True
    elif failure == "view":
        dataset.fail_saved_view = "argus-scene-unreviewed"
    elif failure == "metadata":
        dataset.fail_complete_save = True
    else:
        review.loader.side_effect = [
            deepcopy(review.manifest), ValueError("pixel/report identity changed during import"),
        ]
    with pytest.raises((RuntimeError, ValueError)):
        _import(review)
    dataset = fake_fo.datasets["vfr-images"]
    assert dataset.info[adapter._INFO_KEY]["state"] == "failed"
    assert dataset.samples[0]["argus_import_state"] == "pending"
    assert not dataset.views["argus-current"].contains(dataset.samples[0])
    review.loader.side_effect = lambda *args, **kwargs: deepcopy(review.manifest)
    dataset = _import(review)
    assert dataset.info[adapter._INFO_KEY]["state"] == "complete"
    assert all(dataset.views["argus-current"].contains(sample) for sample in dataset.samples)


def test_artifact_is_verified_before_importing_optional_sdk(fake_fo, review):
    review.loader.side_effect = ValueError("invalid PNG/report/PTS binding")
    with pytest.raises(ValueError, match="invalid PNG/report/PTS binding"):
        _import(review)
    fake_fo.sdk_import.assert_not_called()
    assert fake_fo.datasets == {}


def test_loader_receives_original_unresolved_path_before_optional_sdk(fake_fo, review, monkeypatch):
    path = review.path.parent / "symlink-parent" / "manifest.json"
    received = []

    def reject_alias(manifest_path, **kwargs):
        received.append(manifest_path)
        raise ValueError("symlink/reparse component rejected")

    monkeypatch.setattr(adapter, "load_scene_review_images", reject_alias)
    monkeypatch.setattr(Path, "resolve", lambda *args, **kwargs: pytest.fail("must not hide alias components"))
    with pytest.raises(ValueError, match="symlink/reparse"):
        adapter.import_scene_review_images(
            review.bundle, manifest_path=path, scene_comparisons=review.comparisons,
            dataset_name="vfr-images",
        )
    assert received == [Path(os.path.abspath(path))]
    fake_fo.sdk_import.assert_not_called()


def test_manifest_mutation_during_import_is_not_published(fake_fo, review):
    first = deepcopy(review.manifest)
    mutated = deepcopy(first)
    # Python dict equality would treat True and 1 as equal. Canonical JSON hash
    # comparisons remain type-sensitive even across an independently mocked
    # artifact loader, and so cannot promote this changed report to complete.
    mutated["reports"]["clip-1"]["summary"]["release_authority"] = 0
    review.loader.side_effect = [first, mutated]
    with pytest.raises(ValueError, match="artifact changed while importing"):
        _import(review)
    dataset = fake_fo.datasets["vfr-images"]
    assert dataset.info[adapter._INFO_KEY]["state"] == "failed"
    assert all(sample["argus_import_state"] == "pending" for sample in dataset.samples)


def _real_sdk_boundary(fake_fo, monkeypatch, *, modules=None):
    """Fake imports/connections only, retaining the actual process safety gate."""
    database = SimpleNamespace(get_db_conn=Mock(return_value=fake_fo.database))
    imports = Mock(side_effect=lambda name: {
        "fiftyone": fake_fo.fo, "fiftyone.core.odm.database": database,
    }[name])
    monkeypatch.setattr(adapter, "_load_review_sdk", fake_fo.real_load_review_sdk)
    monkeypatch.setattr(adapter, "_SDK_ATTEMPTED", False)
    monkeypatch.setattr(adapter, "sys", SimpleNamespace(modules={} if modules is None else modules))
    monkeypatch.setattr(adapter, "importlib", SimpleNamespace(import_module=imports))
    monkeypatch.setenv("FIFTYONE_SINGLETON_CACHE", "true")
    monkeypatch.setenv("FIFTYONE_DATABASE_ADMIN", "true")
    monkeypatch.setenv("FIFTYONE_MODULE_PATH", "unsafe.startup.hook")
    return imports, database


@pytest.mark.parametrize("location", ["custom", "default"])
@pytest.mark.parametrize("hooks", [
    ["unsafe.startup.hook"], "unsafe.startup.hook", "", False, 0, {},
])
def test_file_module_hooks_are_refused_before_sdk_import_despite_empty_environment_override(
    fake_fo, monkeypatch, tmp_path, location, hooks,
):
    imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    if location == "default":
        config_path = tmp_path / ".fiftyone" / "config.json"
        config_path.parent.mkdir()
        monkeypatch.delenv("FIFTYONE_CONFIG_PATH")
        monkeypatch.setattr(adapter, "os", SimpleNamespace(
            environ=os.environ, path=SimpleNamespace(
                expanduser=lambda value: str(tmp_path), join=os.path.join, isfile=os.path.isfile,
            ),
        ))
    else:
        config_path = fake_fo.config_path
    config_path.write_text(json.dumps({"module_path": hooks}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="config module hooks before SDK initialization"):
        adapter._load_review_sdk()
    assert os.environ["FIFTYONE_MODULE_PATH"] == ""
    imports.assert_not_called()
    database.get_db_conn.assert_not_called()
    fake_fo.collection.find_one.assert_not_called()
    fake_fo.locks.insert_one.assert_not_called()
    # Fixing a file must not permit reuse of a possibly attempted SDK process.
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fresh independent scene-images-import CLI process"):
        adapter._load_review_sdk()
    imports.assert_not_called()


@pytest.mark.parametrize("contents", [{}, {"module_path": None}, {"module_path": []}])
def test_hook_free_file_preflight_preserves_database_configuration(fake_fo, monkeypatch, contents):
    imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    contents = {**contents, "database_uri": "mongodb://review.invalid/dedicated-review"}
    original = json.dumps(contents).encode("utf-8")
    fake_fo.config_path.write_bytes(original)
    original_path = os.environ["FIFTYONE_CONFIG_PATH"]
    assert adapter._load_review_sdk() == (fake_fo.fo, fake_fo.collection)
    assert fake_fo.config_path.read_bytes() == original
    assert os.environ["FIFTYONE_CONFIG_PATH"] == original_path
    assert imports.call_count == 2
    database.get_db_conn.assert_called_once_with()


@pytest.mark.parametrize("damage", [
    "malformed-json", "non-object", "invalid-encoding", "oversized",
])
def test_uninspectable_sdk_config_fails_before_optional_sdk_or_target_access(
    fake_fo, monkeypatch, damage,
):
    imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    if damage == "malformed-json":
        raw = b'{"module_path": ['
    elif damage == "non-object":
        raw = b'["unsafe.startup.hook"]'
    elif damage == "invalid-encoding":
        raw = b'\xff\xff'
    else:
        monkeypatch.setattr(adapter, "_MAX_SDK_CONFIG_BYTES", 64)
        raw = json.dumps({"padding": "x" * 65}).encode("utf-8")
    fake_fo.config_path.write_bytes(raw)
    with pytest.raises(
        RuntimeError, match="cannot safely inspect FiftyOne config|JSON config object",
    ):
        adapter._load_review_sdk()
    imports.assert_not_called()
    database.get_db_conn.assert_not_called()
    fake_fo.collection.find_one.assert_not_called()
    fake_fo.locks.insert_one.assert_not_called()


def test_unreadable_sdk_config_fails_before_optional_sdk(fake_fo, monkeypatch):
    imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    monkeypatch.setattr(
        adapter, "open", Mock(side_effect=PermissionError("config unavailable")), raising=False,
    )
    with pytest.raises(RuntimeError, match="cannot safely inspect FiftyOne config"):
        adapter._load_review_sdk()
    imports.assert_not_called()
    database.get_db_conn.assert_not_called()


@pytest.mark.parametrize("path_value", ["", "missing-config.json"])
def test_absent_explicit_sdk_config_uses_defaults_without_reading_default_file(
    fake_fo, monkeypatch, tmp_path, path_value,
):
    imports, _database = _real_sdk_boundary(fake_fo, monkeypatch)
    default = tmp_path / ".fiftyone" / "config.json"
    default.parent.mkdir()
    default.write_text(json.dumps({"module_path": ["must.not.run"]}), encoding="utf-8")
    monkeypatch.setenv(
        "FIFTYONE_CONFIG_PATH", "" if not path_value else str(tmp_path / path_value),
    )
    monkeypatch.setattr(adapter, "os", SimpleNamespace(
        environ=os.environ, path=SimpleNamespace(
            expanduser=lambda value: str(tmp_path), join=os.path.join, isfile=os.path.isfile,
        ),
    ))
    assert adapter._load_review_sdk() == (fake_fo.fo, fake_fo.collection)
    assert imports.call_count == 2


def test_sdk_config_hook_refusal_preserves_saved_review_and_dirty_draft(
    fake_fo, review, monkeypatch,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "persisted human review"
    sample.save()
    sample["argus_scene_human_note"] = "unsaved human draft"
    draft, dirty = deepcopy(dict(sample)), set(sample.dirty)
    persisted, writes = deepcopy(dataset.persisted_info), len(sample.writes)
    imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    fake_fo.config_path.write_text(json.dumps({"module_path": ["unsafe.hook"]}), encoding="utf-8")
    lock_attempts = fake_fo.locks.insert_one.call_count
    with pytest.raises(RuntimeError, match="config module hooks before SDK initialization"):
        _import(review)
    imports.assert_not_called()
    database.get_db_conn.assert_not_called()
    assert fake_fo.locks.insert_one.call_count == lock_attempts
    assert dict(sample) == draft and sample.dirty == dirty
    assert sample.stored["argus_scene_human_note"] == "persisted human review"
    assert len(sample.writes) == writes and dataset.persisted_info == persisted


@pytest.mark.parametrize("handle_name", ["database", "collection"])
@pytest.mark.parametrize("damage", [
    "secondary-preferred", "secondary", "boolean-primary", "missing-preference",
    "missing-mode", "unacknowledged", "numeric-ack", "missing-concern", "missing-ack",
])
def test_sdk_boundary_requires_primary_acknowledged_database_and_target_collection(
    fake_fo, monkeypatch, handle_name, damage,
):
    _imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    handle = fake_fo.database if handle_name == "database" else fake_fo.collection
    if damage == "secondary-preferred":
        handle.read_preference.mode = 3
    elif damage == "secondary":
        handle.read_preference.mode = 2
    elif damage == "boolean-primary":
        handle.read_preference.mode = False
    elif damage == "missing-preference":
        del handle.read_preference
    elif damage == "missing-mode":
        del handle.read_preference.mode
    elif damage == "unacknowledged":
        handle.write_concern.acknowledged = False
    elif damage == "numeric-ack":
        handle.write_concern.acknowledged = 1
    elif damage == "missing-concern":
        del handle.write_concern
    else:
        del handle.write_concern.acknowledged
    with pytest.raises(RuntimeError, match="PRIMARY reads and acknowledged writes"):
        adapter._load_review_sdk()
    database.get_db_conn.assert_called_once_with()
    fake_fo.collection.find_one.assert_not_called()
    fake_fo.fo.load_dataset.assert_not_called()
    fake_fo.locks.insert_one.assert_not_called()


@pytest.mark.parametrize("handle_name", ["database", "collection"])
@pytest.mark.parametrize("level", [
    None, "majority", "snapshot", "available", "", 0, False, ["local"],
    "missing-concern", "missing-level",
])
def test_sdk_boundary_requires_explicit_local_reads_before_target_access(
    fake_fo, monkeypatch, handle_name, level,
):
    _imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    handle = getattr(fake_fo, handle_name)
    if level == "missing-concern":
        del handle.read_concern
    elif level == "missing-level":
        del handle.read_concern.level
    else:
        handle.read_concern.level = level
    with pytest.raises(RuntimeError, match="explicit local"):
        adapter._load_review_sdk()
    database.get_db_conn.assert_called_once_with()
    fake_fo.collection.find_one.assert_not_called()
    fake_fo.fo.load_dataset.assert_not_called()
    fake_fo.locks.insert_one.assert_not_called()
    assert fake_fo.datasets == {}


@pytest.mark.parametrize("damage", [
    "mongos", "missing-client", "unknown-topology", "duplicate-metadata", "bad-metadata",
    "wrong-name", "view", "timeseries", "capped", "auto-id-disabled", "clustered",
    "collection-ttl", "view-option", "pipeline-option", "missing-options",
    "missing-id-index", "wrong-id-key", "boolean-id-order", "nonunique-id", "sparse-id",
    "partial-id", "hidden-id", "ttl-id", "other-ttl", "uninspectable-indexes",
    "bad-index-spec",
])
def test_unsafe_lock_collection_contract_never_grants_target_access(fake_fo, review, damage):
    metadata = fake_fo.database.lock_metadata[0]
    identity = fake_fo.locks.indexes["_id_"]
    if damage == "mongos":
        fake_fo.database.client.is_mongos = True
    elif damage == "missing-client":
        del fake_fo.database.client
    elif damage == "unknown-topology":
        fake_fo.database.client.is_mongos = None
    elif damage == "duplicate-metadata":
        fake_fo.database.lock_metadata.append(deepcopy(metadata))
    elif damage == "bad-metadata":
        fake_fo.database.lock_metadata = ["not collection metadata"]
    elif damage == "wrong-name":
        metadata["name"] = "unrelated-collection"
    elif damage == "view":
        metadata["type"] = "view"
    elif damage == "missing-options":
        metadata.pop("options")
    elif damage in {
        "timeseries", "capped", "auto-id-disabled", "clustered", "collection-ttl",
        "view-option", "pipeline-option",
    }:
        name, value = {
            "timeseries": ("timeseries", {"timeField": "created_at"}),
            "capped": ("capped", True), "auto-id-disabled": ("autoIndexId", False),
            "clustered": ("clusteredIndex", {"key": {"_id": 1}, "unique": True}),
            "collection-ttl": ("expireAfterSeconds", 60),
            "view-option": ("viewOn", "another-collection"), "pipeline-option": ("pipeline", []),
        }[damage]
        metadata["options"][name] = value
    elif damage == "missing-id-index":
        fake_fo.locks.indexes = {}
    elif damage == "wrong-id-key":
        identity["key"] = [("token", 1)]
    elif damage == "boolean-id-order":
        identity["key"] = [("_id", True)]
    elif damage in {"nonunique-id", "sparse-id", "hidden-id", "partial-id", "ttl-id"}:
        name, value = {
            "nonunique-id": ("unique", False), "sparse-id": ("sparse", True),
            "hidden-id": ("hidden", True), "partial-id": ("partialFilterExpression", {}),
            "ttl-id": ("expireAfterSeconds", 60),
        }[damage]
        identity[name] = value
    elif damage == "other-ttl":
        fake_fo.locks.indexes["created_at_1"] = {
            "key": [("created_at", 1)], "expireAfterSeconds": 1,
        }
    elif damage == "uninspectable-indexes":
        fake_fo.locks.index_information.side_effect = lambda: None
    else:
        fake_fo.locks.indexes["unexpected"] = "uninspectable index specification"
    with pytest.raises(RuntimeError, match="safe database import lock collection contract"):
        adapter._validated_lock_collection(fake_fo.collection)
    with pytest.raises(RuntimeError, match="database import lock"):
        _import(review)
    fake_fo.locks.insert_one.assert_not_called()
    fake_fo.locks.delete_one.assert_not_called()
    fake_fo.collection.find_one.assert_not_called()
    fake_fo.fo.load_dataset.assert_not_called()
    fake_fo.database.create_collection.assert_not_called()
    assert fake_fo.datasets == {}
    assert all(cursor.close.call_count == 1 for cursor in fake_fo.database.lock_cursors)


@pytest.mark.parametrize("location", ["collection", "id-index", "other-index"])
@pytest.mark.parametrize("collation", [
    None, "simple", {}, {"locale": None}, {"locale": "en"}, {"locale": ["simple"]},
    {"locale": "en", "numericOrdering": True},
    {"locale": "simple", "strength": 1}, {"locale": "simple", "numericOrdering": True},
])
def test_nonbinary_or_malformed_lock_collation_never_grants_target_access(
    fake_fo, review, location, collation,
):
    if location == "collection":
        specification = fake_fo.database.lock_metadata[0]["options"]
    elif location == "id-index":
        specification = fake_fo.locks.indexes["_id_"]
    else:
        specification = {"key": [("created_at", 1)]}
        fake_fo.locks.indexes["created_at_1"] = specification
    specification["collation"] = deepcopy(collation)
    with pytest.raises(RuntimeError, match="safe database import lock collection contract"):
        adapter._validated_lock_collection(fake_fo.collection)
    with pytest.raises(RuntimeError, match="database import lock"):
        _import(review)
    fake_fo.locks.insert_one.assert_not_called()
    fake_fo.locks.delete_one.assert_not_called()
    fake_fo.collection.find_one.assert_not_called()
    fake_fo.fo.load_dataset.assert_not_called()
    fake_fo.database.create_collection.assert_not_called()
    assert fake_fo.datasets == {}


@pytest.mark.parametrize("explicit_simple", [False, True])
def test_default_or_explicit_binary_lock_collation_allows_import(fake_fo, review, explicit_simple):
    fake_fo.locks.indexes["created_at_1"] = {"key": [("created_at", 1)]}
    if explicit_simple:
        for specification in (
            fake_fo.database.lock_metadata[0]["options"],
            *fake_fo.locks.indexes.values(),
        ):
            specification["collation"] = {"locale": "simple"}
    dataset = _import(review)
    assert dataset.persisted_info[adapter._INFO_KEY]["state"] == "complete"
    assert fake_fo.locks.documents == {}


@pytest.mark.parametrize("race", [None, "namespace-exists", "client-exists"])
def test_absent_lock_collection_is_created_and_rechecked_without_bypassing_contract(
    fake_fo, review, race,
):
    fake_fo.database.lock_metadata = []
    create = fake_fo.database.create_collection.side_effect

    def concurrent_create(name):
        result = create(name)
        if race == "namespace-exists":
            raise OperationFailure("another importer already created this collection", 48)
        if race == "client-exists":
            raise CollectionInvalid("another importer already created this collection")
        return result

    fake_fo.database.create_collection.side_effect = concurrent_create
    dataset = _import(review)
    fake_fo.database.create_collection.assert_called_once_with("argus_scene_image_import_locks")
    assert fake_fo.database.list_collections.call_count == 2
    assert fake_fo.locks.index_information.call_count == 1
    assert all(cursor.close.call_count == 1 for cursor in fake_fo.database.lock_cursors)
    assert dataset.persisted_info[adapter._INFO_KEY]["state"] == "complete"
    assert fake_fo.locks.documents == {}


@pytest.mark.parametrize("fault", ["permission", "unknown-outcome", "wrong-code", "unsafe-race"])
def test_lock_collection_creation_failure_or_unsafe_race_cannot_create_target(
    fake_fo, review, fault,
):
    fake_fo.database.lock_metadata = []
    create = fake_fo.database.create_collection.side_effect

    def fail_create(name):
        if fault == "permission":
            raise PermissionError("cannot create import lock collection")
        if fault == "unknown-outcome":
            create(name)
            raise NetworkTimeout("collection creation may still be running")
        if fault == "wrong-code":
            raise OperationFailure("not NamespaceExists", True)
        create(name)
        fake_fo.database.lock_metadata[0]["options"]["capped"] = True
        raise OperationFailure("concurrent unsafe creator", 48)

    fake_fo.database.create_collection.side_effect = fail_create
    with pytest.raises(RuntimeError, match="database import lock"):
        _import(review)
    fake_fo.locks.insert_one.assert_not_called()
    fake_fo.collection.find_one.assert_not_called()
    assert fake_fo.datasets == {}
    assert all(cursor.close.call_count == 1 for cursor in fake_fo.database.lock_cursors)


def test_database_lock_serializes_importers_with_unshared_local_locks(fake_fo, review, monkeypatch):
    local_entries = []

    @contextmanager
    def independent_local_lock(name):
        assert name == "vfr-images"
        local_entries.append(object())
        yield

    monkeypatch.setattr(adapter, "_dataset_import_lock", independent_local_lock)
    read = fake_fo.collection.find_one.side_effect
    interleaved = False

    def interleave_second_importer(query, projection):
        nonlocal interleaved
        if not interleaved:
            interleaved = True
            owner = deepcopy(fake_fo.locks.documents["vfr-images"])
            with pytest.raises(RuntimeError, match="database import lock could not be acquired"):
                _import(review)
            assert fake_fo.locks.documents["vfr-images"] == owner
            assert fake_fo.datasets == {}  # The contender never creates the target.
        return read(query, projection)

    fake_fo.collection.find_one.side_effect = interleave_second_importer
    dataset = _import(review)
    assert interleaved and len(local_entries) == 2
    assert local_entries[0] is not local_entries[1]
    assert fake_fo.collection.find_one.call_count == 1  # No contender preflight.
    assert dataset.persisted_info[adapter._INFO_KEY]["state"] == "complete"
    assert fake_fo.locks.insert_one.call_count == 2
    assert fake_fo.locks.delete_one.call_count == 1
    acquired = fake_fo.locks.insert_one.call_args_list[0].args[0]
    released = fake_fo.locks.delete_one.call_args.args[0]
    assert released == {"_id": "vfr-images", "token": acquired["token"]}
    assert fake_fo.locks.documents == {}


@pytest.mark.parametrize("fault", [
    "before", "after", "unacknowledged", "numeric-ack", "missing-ack",
])
def test_uncertain_database_lock_acquisition_never_grants_target_access(fake_fo, review, fault):
    if fault in {"before", "after"}:
        fake_fo.locks.insert_fault = fault
    elif fault == "unacknowledged":
        fake_fo.locks.insert_acknowledged = False
    elif fault == "numeric-ack":
        fake_fo.locks.insert_acknowledged = 1
    else:
        insert = fake_fo.locks.insert_one.side_effect

        def no_ack(document):
            insert(document)
            return SimpleNamespace()

        fake_fo.locks.insert_one.side_effect = no_ack
    with pytest.raises(
        RuntimeError, match="database import lock could not be acquired or its outcome is uncertain",
    ):
        _import(review)
    fake_fo.collection.find_one.assert_not_called()
    fake_fo.fo.load_dataset.assert_not_called()
    fake_fo.locks.delete_one.assert_not_called()
    assert fake_fo.datasets == {}
    if fault == "before":
        assert fake_fo.locks.documents == {}
    else:
        retained = deepcopy(fake_fo.locks.documents)
        assert set(retained) == {"vfr-images"}
        # A new CLI still cannot steal a possibly committed lock.
        fake_fo.locks.insert_fault = None
        fake_fo.locks.insert_acknowledged = True
        fake_fo.locks.insert_one.side_effect = fake_fo.locks._insert_one
        with pytest.raises(RuntimeError, match="database import lock could not be acquired"):
            _import(review)
        assert fake_fo.locks.documents == retained
        fake_fo.collection.find_one.assert_not_called()
        fake_fo.locks.delete_one.assert_not_called()


def test_old_foreign_database_lock_is_not_expired_or_taken_over(fake_fo, review):
    retained = {
        "_id": "vfr-images", "token": "another-importer", "host": "another-host",
        "pid": -1, "created_at": "1900-01-01T00:00:00Z",
    }
    fake_fo.locks.documents["vfr-images"] = deepcopy(retained)
    with pytest.raises(RuntimeError, match="administrator recover the lock"):
        _import(review)
    assert fake_fo.locks.documents["vfr-images"] == retained
    fake_fo.locks.delete_one.assert_not_called()
    fake_fo.collection.find_one.assert_not_called()


@pytest.mark.parametrize("fault", ["before", "after", "unacknowledged", "foreign-owner"])
def test_existing_target_lock_refusal_preserves_ledger_saved_note_and_unsaved_draft(
    fake_fo, review, fault,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "saved review before lock refusal"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    sample["argus_scene_human_note"] = "unsaved draft before lock refusal"
    saved_info, draft = deepcopy(dataset.persisted_info), deepcopy(dict(sample))
    dirty, writes = set(sample.dirty), len(sample.writes)
    reads, loads = fake_fo.collection.find_one.call_count, fake_fo.fo.load_dataset.call_count
    saves, releases = dataset.info_save_attempts, fake_fo.locks.delete_one.call_count
    if fault in {"before", "after"}:
        fake_fo.locks.insert_fault = fault
    elif fault == "unacknowledged":
        fake_fo.locks.insert_acknowledged = False
    else:
        fake_fo.locks.documents["vfr-images"] = {
            "_id": "vfr-images", "token": "another-owner",
        }
    with pytest.raises(RuntimeError, match="database import lock could not be acquired"):
        _import(review)
    assert fake_fo.collection.find_one.call_count == reads
    assert fake_fo.fo.load_dataset.call_count == loads
    assert fake_fo.locks.delete_one.call_count == releases
    assert dataset.persisted_info == saved_info and dataset.info_save_attempts == saves
    assert dict(sample) == draft and sample.dirty == dirty and len(sample.writes) == writes
    assert sample.stored["argus_scene_human_note"] == "saved review before lock refusal"
    assert sample.stored["argus_scene_reviewed_report_sha256"] == "b" * 64


@pytest.mark.parametrize("damage", ["secondary", "boolean-primary", "w0", "missing-concern"])
def test_lock_collection_itself_requires_primary_and_acknowledged_writes(fake_fo, review, damage):
    if damage == "secondary":
        fake_fo.locks.read_preference.mode = 2
    elif damage == "boolean-primary":
        fake_fo.locks.read_preference.mode = False
    elif damage == "w0":
        fake_fo.locks.write_concern.acknowledged = False
    else:
        del fake_fo.locks.write_concern
    with pytest.raises(RuntimeError, match="safe database import lock collection contract"):
        _import(review)
    fake_fo.locks.insert_one.assert_not_called()
    fake_fo.collection.find_one.assert_not_called()
    assert fake_fo.datasets == {}


@pytest.mark.parametrize("handle_name", ["database", "collection", "locks"])
@pytest.mark.parametrize("level", [
    None, "majority", "snapshot", "available", "", 0, False, ["local"],
    "missing-concern", "missing-level",
])
def test_local_read_concern_is_rechecked_before_any_target_read_or_review_write(
    fake_fo, review, handle_name, level,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "saved note survives unsafe read concern"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    sample["argus_scene_human_note"] = "unsaved draft survives unsafe read concern"
    persisted, draft = deepcopy(dataset.persisted_info), deepcopy(dict(sample))
    dirty, writes = set(sample.dirty), len(sample.writes)
    reads, loads = fake_fo.collection.find_one.call_count, fake_fo.fo.load_dataset.call_count
    attempts = dataset.info_save_attempts
    acquisitions = fake_fo.locks.insert_one.call_count
    releases = fake_fo.locks.delete_one.call_count
    handle = getattr(fake_fo, handle_name)
    if level == "missing-concern":
        del handle.read_concern
    elif level == "missing-level":
        del handle.read_concern.level
    else:
        handle.read_concern.level = level
    with pytest.raises(RuntimeError, match="safe database import lock collection contract"):
        _import(review)
    assert fake_fo.collection.find_one.call_count == reads
    assert fake_fo.fo.load_dataset.call_count == loads
    assert fake_fo.locks.insert_one.call_count == acquisitions
    assert fake_fo.locks.delete_one.call_count == releases
    assert dataset.persisted_info == persisted and dataset.info_save_attempts == attempts
    assert dict(sample) == draft and sample.dirty == dirty and len(sample.writes) == writes
    assert sample.stored["argus_scene_human_note"] == "saved note survives unsafe read concern"
    assert sample.stored["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert fake_fo.locks.documents == {}


@pytest.mark.parametrize("phase", ["before", "after"])
def test_first_owner_failure_releases_database_lock_before_fresh_cli_retry(fake_fo, review, phase):
    _add_zero_selection_clip(review, with_report=False)
    fake_fo.fo.next_create_info_save_fault = ("failed", phase)
    with pytest.raises(RuntimeError, match=f"metadata failure {phase} commit"):
        _import(review)
    dataset = fake_fo.datasets["vfr-images"]
    persisted = deepcopy(dataset.persisted_info)
    assert fake_fo.locks.documents == {}
    assert fake_fo.locks.delete_one.call_count == 1
    if phase == "before":
        with pytest.raises(ValueError, match="refusing unrelated/native-video dataset"):
            _import(review)
        assert dataset.persisted_info == persisted == {}
        assert dataset.samples == []
    else:
        registry = persisted[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]
        assert set(registry["clips"]) == {"clip-1", "clip-2"}
        fresh = _import(review)
        assert fresh.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == registry
        assert fresh.persisted_info[adapter._INFO_KEY]["state"] == "complete"
    assert fake_fo.locks.documents == {}
    assert fake_fo.locks.delete_one.call_count == 2


@pytest.mark.parametrize("phase", ["before", "after"])
def test_existing_registry_failure_releases_lock_and_preserves_notes_on_retry(
    fake_fo, review, phase,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "saved before source registry ACK failure"
    sample["argus_scene_human_state"] = "moved"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    offset = len(sample.writes)
    _add_zero_selection_clip(review, with_report=False)
    dataset.info_save_fault = ("importing", phase)
    with pytest.raises(RuntimeError, match=f"metadata failure {phase} commit"):
        _import(review)
    assert fake_fo.locks.documents == {}
    assert len(sample.writes) == offset
    fresh = _import(review)
    sample.reload()
    assert fake_fo.locks.documents == {}
    assert fresh.persisted_info[adapter._INFO_KEY]["state"] == "complete"
    assert sample["argus_scene_human_note"] == "saved before source registry ACK failure"
    assert sample["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert all(not set(adapter._HUMAN_FIELDS) & fields for fields in sample.writes[offset:])


def _finish_server_writes_and_recover_exact_lock(fake_fo, dataset, token):
    """Simulate an administrator confirming server completion, not a timeout takeover."""
    while dataset.pending_writes:
        dataset.pending_writes.pop(0)()
    assert all(not store.pending_writes for store in fake_fo.stores.values())
    released = fake_fo.locks.delete_one({"_id": dataset.name, "token": token})
    assert released.acknowledged is True and released.deleted_count == 1


@pytest.mark.parametrize("stage", ["registry", "sample"])
@pytest.mark.parametrize("error_type", [NetworkTimeout, AutoReconnect, TimeoutError, ConnectionError])
def test_unknown_write_retains_mutex_until_server_work_and_manual_recovery_are_confirmed(
    fake_fo, review, stage, error_type,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "human note predates an unknown database outcome"
    sample["argus_scene_human_state"] = "moved"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    offset = len(sample.writes)
    _add_zero_selection_clip(review, with_report=False)

    def report_b(manifest):
        manifest["reports"]["clip-1"]["report_sha256"] = "e" * 64
        for image in manifest["images"]:
            image["report_sha256"] = "e" * 64

    _update_manifest(review, report_b)
    if stage == "registry":
        dataset.info_unknown_fault = ("importing", error_type)
    else:
        dataset.sample_unknown_fault = error_type
    release_calls = fake_fo.locks.delete_one.call_count
    with pytest.raises(adapter.ImportLockFailure, match="database import lock retained") as caught:
        _import(review)
    assert isinstance(caught.value.uncertain_error, error_type)
    assert len(dataset.pending_writes) == 1  # A client error did not cancel this write.
    assert fake_fo.locks.delete_one.call_count == release_calls
    retained = deepcopy(fake_fo.locks.documents[dataset.name])
    reads = fake_fo.collection.find_one.call_count
    with pytest.raises(RuntimeError, match="database import lock could not be acquired"):
        _import(review)
    assert fake_fo.collection.find_one.call_count == reads
    assert fake_fo.locks.documents[dataset.name] == retained
    assert len(dataset.pending_writes) == 1
    assert sample.stored["argus_scene_human_note"] == (
        "human note predates an unknown database outcome"
    )
    assert sample.stored["argus_scene_reviewed_report_sha256"] == "b" * 64
    # The server can commit after the caller already received an exception.
    # A PID exit alone must never be treated as permission to enter this target.
    while dataset.pending_writes:
        dataset.pending_writes.pop(0)()
    if stage == "registry":
        registry = dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]
        assert registry["clips"]["clip-2"]["source_session_id"] == "session-2"
    with pytest.raises(RuntimeError, match="database import lock could not be acquired"):
        _import(review)
    assert fake_fo.locks.documents[dataset.name] == retained
    _finish_server_writes_and_recover_exact_lock(fake_fo, dataset, retained["token"])
    fresh = _import(review)
    sample.reload()
    assert fake_fo.locks.documents == {} and not dataset.pending_writes
    assert fresh.persisted_info[adapter._INFO_KEY]["state"] == "complete"
    assert sample["argus_scene_human_note"] == "human note predates an unknown database outcome"
    assert sample["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert fresh.views["argus-scene-stale-review"].contains(sample)
    assert all(not set(adapter._HUMAN_FIELDS) & fields for fields in sample.writes[offset:])


def test_first_owner_unknown_write_cannot_publish_late_owner_after_another_importer_enters(
    fake_fo, review,
):
    _add_zero_selection_clip(review, with_report=False)
    create = fake_fo.fo.Dataset

    def create_with_unknown_owner_write(**kwargs):
        dataset = create(**kwargs)
        dataset.info_unknown_fault = ("failed", NetworkTimeout)
        return dataset

    fake_fo.fo.Dataset = create_with_unknown_owner_write
    with pytest.raises(adapter.ImportLockFailure, match="database import lock retained"):
        _import(review)
    dataset = fake_fo.datasets["vfr-images"]
    assert dataset.persisted_info == {} and dataset.samples == []
    retained = deepcopy(fake_fo.locks.documents[dataset.name])
    reads = fake_fo.collection.find_one.call_count
    with pytest.raises(RuntimeError, match="database import lock could not be acquired"):
        _import(review)
    assert fake_fo.collection.find_one.call_count == reads
    _finish_server_writes_and_recover_exact_lock(fake_fo, dataset, retained["token"])
    registry = dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]
    assert set(registry["clips"]) == {"clip-1", "clip-2"}
    fake_fo.fo.Dataset = create
    fresh = _import(review)
    assert fresh.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == registry
    assert fresh.persisted_info[adapter._INFO_KEY]["state"] == "complete"


@pytest.mark.parametrize("stage", ["sample", "metadata"])
@pytest.mark.parametrize("interruption", [None, KeyboardInterrupt, SystemExit])
def test_unknown_compensation_write_keeps_primary_failure_and_stops_further_compensation(
    fake_fo, review, stage, interruption,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "retain operator notes when rollback also times out"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    attempts, release_calls = dataset.info_save_attempts, fake_fo.locks.delete_one.call_count
    initial = deepcopy(review.manifest)
    loader_calls = 0
    expected_type = interruption or (AutoReconnect if stage == "sample" else NetworkTimeout)

    def fail_revalidation(*args, **kwargs):
        nonlocal loader_calls
        loader_calls += 1
        if loader_calls == 2:
            if stage == "sample":
                dataset.sample_unknown_fault = expected_type
            else:
                dataset.info_unknown_fault = ("failed", expected_type)
            raise ValueError("deterministic artifact rejection before compensation")
        return deepcopy(initial)

    review.loader.side_effect = fail_revalidation
    # Catch the raw interruption too, so an incorrect implementation fails this
    # assertion instead of aborting the future test-machine pytest process.
    with pytest.raises(BaseException) as caught:
        _import(review)
    assert isinstance(caught.value, adapter.ImportLockFailure)
    assert "database import lock retained" in str(caught.value)
    assert isinstance(caught.value.import_error, ValueError)
    assert isinstance(caught.value.uncertain_error, expected_type)
    assert "import error=ValueError" in str(caught.value)
    assert f"uncertain error={expected_type.__name__}" in str(caught.value)
    # Sample uncertainty stops the failed-info save entirely; metadata
    # uncertainty stops immediately after that single unconfirmed attempt.
    assert dataset.info_save_attempts == attempts + (1 if stage == "sample" else 2)
    assert fake_fo.locks.delete_one.call_count == release_calls
    assert len(dataset.pending_writes) == 1
    retained = deepcopy(fake_fo.locks.documents[dataset.name])
    assert sample.stored["argus_scene_human_note"] == (
        "retain operator notes when rollback also times out"
    )
    review.loader.side_effect = lambda *args, **kwargs: deepcopy(initial)
    _finish_server_writes_and_recover_exact_lock(fake_fo, dataset, retained["token"])
    fresh = _import(review)
    assert fresh.persisted_info[adapter._INFO_KEY]["state"] == "complete"
    assert sample.stored["argus_scene_human_note"] == (
        "retain operator notes when rollback also times out"
    )


@pytest.mark.parametrize("chain", ["cause", "context", "group", "mongoengine", "cycle"])
def test_wrapped_unknown_database_failure_is_not_lost_at_the_import_boundary(
    fake_fo, review, chain,
):
    dataset = _import(review)
    unknown = NetworkTimeout("mongodb://review-user:secret-password@private-host/review")
    if chain == "mongoengine":
        error = OperationError("private server detail must not be printed")
        error.__cause__ = unknown
    elif chain == "group":
        error = ExceptionGroup("SDK wrapper failures", [ValueError("known"), unknown])
    else:
        error = RuntimeError("opaque SDK wrapper")
        if chain == "context":
            error.__context__ = unknown
        else:
            error.__cause__ = unknown
            if chain == "cycle":
                unknown.__context__ = error

    def fail_save(_sample):
        raise error

    dataset.before_sample_save = fail_save
    release_calls = fake_fo.locks.delete_one.call_count
    with pytest.raises(adapter.ImportLockFailure, match="database import lock retained") as caught:
        _import(review)
    assert caught.value.import_error is error
    assert caught.value.uncertain_error is not None
    assert fake_fo.locks.delete_one.call_count == release_calls
    assert "vfr-images" in fake_fo.locks.documents
    assert "secret-password" not in str(caught.value)
    assert "private-host" not in str(caught.value)
    assert "private server detail" not in str(caught.value)


def test_primary_import_error_and_failed_lock_cleanup_are_both_preserved_without_db_secrets(
    fake_fo, review,
):
    dataset = _import(review)
    primary = ValueError("image artifact changed while importing")
    review.loader.side_effect = [deepcopy(review.manifest), primary]
    cleanup = RuntimeError("mongodb://review-user:secret-password@private-host/review")
    fake_fo.locks.delete_one.side_effect = cleanup
    with pytest.raises(adapter.ImportLockFailure, match="release is unconfirmed") as caught:
        _import(review)
    assert caught.value.import_error is primary and caught.value.cleanup_error is cleanup
    assert "import error=ValueError: image artifact changed while importing" in str(caught.value)
    assert "cleanup error=RuntimeError" in str(caught.value)
    assert "secret-password" not in str(caught.value) and "private-host" not in str(caught.value)
    assert dataset.persisted_info[adapter._INFO_KEY]["state"] == "failed"
    assert "vfr-images" in fake_fo.locks.documents


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_interrupted_lock_acquisition_does_not_grant_access_to_the_target(
    fake_fo, review, interruption, phase,
):
    insert = fake_fo.locks.insert_one.side_effect
    interrupted = interruption("operator interrupted lock acquisition")

    def interrupted_insert(document):
        if phase == "after":
            insert(document)
        raise interrupted

    fake_fo.locks.insert_one.side_effect = interrupted_insert
    with pytest.raises(BaseException) as caught:
        _import(review)
    assert isinstance(caught.value, RuntimeError)
    assert "database import lock could not be acquired or its outcome is uncertain" in str(
        caught.value
    )
    assert caught.value.__cause__ is interrupted
    fake_fo.collection.find_one.assert_not_called()
    fake_fo.fo.load_dataset.assert_not_called()
    fake_fo.locks.delete_one.assert_not_called()
    assert fake_fo.datasets == {}
    assert bool(fake_fo.locks.documents) is (phase == "after")


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_cleanup_interruption_does_not_hide_an_existing_import_failure(
    fake_fo, review, interruption,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "keep notes when lock cleanup is interrupted"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    primary = ValueError("image artifact changed while importing")
    review.loader.side_effect = [deepcopy(review.manifest), primary]
    cleanup = interruption("operator interrupted lock cleanup")
    fake_fo.locks.delete_one.side_effect = cleanup
    with pytest.raises(BaseException) as caught:
        _import(review)
    assert isinstance(caught.value, adapter.ImportLockFailure)
    assert "database import lock release is unconfirmed" in str(caught.value)
    assert caught.value.import_error is primary and caught.value.cleanup_error is cleanup
    assert f"cleanup error={interruption.__name__}" in str(caught.value)
    assert "import error=ValueError: image artifact changed while importing" in str(caught.value)
    assert "vfr-images" in fake_fo.locks.documents
    assert sample.stored["argus_scene_human_note"] == "keep notes when lock cleanup is interrupted"
    assert sample.stored["argus_scene_reviewed_report_sha256"] == "b" * 64


def test_v1_rejection_and_failed_cleanup_both_remain_visible_without_touching_saved_reviews(
    fake_fo, review,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "retain the historical v1 review note"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]["schema_version"] = (
        "argus.scene-review-image-sources.v1"
    )
    dataset.save()
    saved = deepcopy(dataset.persisted_info)
    attempts, writes = dataset.info_save_attempts, len(sample.writes)
    loads = fake_fo.fo.load_dataset.call_count
    cleanup = RuntimeError("mongodb://review-user:secret-password@private-host/review")
    fake_fo.locks.delete_one.side_effect = cleanup
    with pytest.raises(adapter.ImportLockFailure, match="release is unconfirmed") as caught:
        _import(review)
    assert isinstance(caught.value.import_error, ValueError)
    assert caught.value.cleanup_error is cleanup
    assert "legacy/incomplete v1 source registry" in str(caught.value)
    assert "cleanup error=RuntimeError" in str(caught.value)
    assert "secret-password" not in str(caught.value) and "private-host" not in str(caught.value)
    assert fake_fo.fo.load_dataset.call_count == loads
    assert dataset.persisted_info == saved and dataset.info_save_attempts == attempts
    assert len(sample.writes) == writes
    assert sample.stored["argus_scene_human_note"] == "retain the historical v1 review note"
    assert sample.stored["argus_scene_reviewed_report_sha256"] == "b" * 64


def test_sdk_failure_stringification_is_not_required_to_keep_an_unknown_write_locked(
    fake_fo, review,
):
    class UnprintableDriverError(RuntimeError):
        __module__ = "pymongo.errors"

        def __str__(self):
            raise AssertionError("an opaque SDK exception must not be stringified")

    dataset = _import(review)
    opaque = UnprintableDriverError("secret driver details")

    def fail_save(_sample):
        raise opaque

    dataset.before_sample_save = fail_save
    release_calls = fake_fo.locks.delete_one.call_count
    with pytest.raises(adapter.ImportLockFailure, match="database import lock retained") as caught:
        _import(review)
    assert caught.value.import_error is opaque and caught.value.uncertain_error is opaque
    assert "UnprintableDriverError" in str(caught.value)
    assert "secret driver details" not in str(caught.value)
    assert fake_fo.locks.delete_one.call_count == release_calls


def test_import_lease_retains_the_first_unknown_failure_across_later_known_errors():
    lease = adapter._ImportLease()
    known = ValueError("known client rejection")
    unknown = AutoReconnect("unknown server result")
    assert lease.record_failure(known) is False
    assert lease.record_failure(unknown) is True
    assert lease.record_failure(known) is True
    assert lease.uncertain_failure is unknown and lease.uncertain is True


@pytest.mark.parametrize("diagnostic_type", [MemoryError, KeyboardInterrupt])
def test_classifier_failure_keeps_the_original_lease_failure_sticky(monkeypatch, diagnostic_type):
    original = NetworkTimeout("original unknown server result")
    classifier = Mock(side_effect=diagnostic_type("diagnostic could not finish"))
    monkeypatch.setattr(adapter, "_uncertain_database_failure", classifier)
    lease = adapter._ImportLease()
    assert lease.record_failure(original) is True
    assert lease.uncertain_failure is original
    assert lease.record_failure(ValueError("later deterministic rejection")) is True
    assert lease.record_failure(AutoReconnect("later unknown result")) is True
    assert lease.uncertain_failure is original and lease.uncertain is True
    classifier.assert_called_once_with(original)


@pytest.mark.parametrize("diagnostic_type", [MemoryError, KeyboardInterrupt])
@pytest.mark.parametrize("stage", ["primary-write", "sample-compensation"])
def test_classifier_failure_retains_lock_primary_error_and_notes_without_further_compensation(
    fake_fo, review, monkeypatch, diagnostic_type, stage,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "keep review when failure inspection cannot complete"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    attempts, releases = dataset.info_save_attempts, fake_fo.locks.delete_one.call_count
    original = ValueError("deterministic artifact rejection before compensation")
    compensation = RuntimeError("compensation result could not be classified")
    before_sample_save = Mock(side_effect=original if stage == "primary-write" else compensation)
    diagnostic = diagnostic_type("diagnostic cannot replace the original failure")
    classifier = Mock(side_effect=diagnostic if stage == "primary-write" else [None, diagnostic])
    monkeypatch.setattr(adapter, "_uncertain_database_failure", classifier)
    if stage == "primary-write":
        dataset.before_sample_save = before_sample_save
        uncertain = original
    else:
        initial = deepcopy(review.manifest)
        loader_calls = 0

        def fail_revalidation(*args, **kwargs):
            nonlocal loader_calls
            loader_calls += 1
            if loader_calls == 2:
                dataset.before_sample_save = before_sample_save
                raise original
            return deepcopy(initial)

        review.loader.side_effect = fail_revalidation
        uncertain = compensation
    # Catch BaseException so an unsafe diagnostic escape fails an assertion
    # instead of interrupting the future test-machine run.
    with pytest.raises(BaseException) as caught:
        _import(review)
    assert isinstance(caught.value, adapter.ImportLockFailure)
    assert caught.value.import_error is original and caught.value.uncertain_error is uncertain
    assert caught.value.cleanup_error is None
    assert fake_fo.locks.delete_one.call_count == releases
    assert dataset.info_save_attempts == attempts + 1  # No failed-state compensation save.
    assert dataset.persisted_info[adapter._INFO_KEY]["state"] == "importing"
    before_sample_save.assert_called_once()
    assert classifier.call_count == (1 if stage == "primary-write" else 2)
    assert classifier.call_args.args == (uncertain,)
    assert sample.stored["argus_scene_human_note"] == (
        "keep review when failure inspection cannot complete"
    )
    assert sample.stored["argus_scene_reviewed_report_sha256"] == "b" * 64
    retained = deepcopy(fake_fo.locks.documents[dataset.name])
    reads = fake_fo.collection.find_one.call_count
    with pytest.raises(RuntimeError, match="database import lock could not be acquired"):
        _import(review)
    assert fake_fo.collection.find_one.call_count == reads
    assert fake_fo.locks.documents[dataset.name] == retained
    assert fake_fo.locks.delete_one.call_count == releases


@pytest.mark.parametrize("stage", ["schema", "view"])
@pytest.mark.parametrize("failure", ["private-details", "unprintable-wrapper"])
def test_compensation_persists_only_a_safe_error_summary_without_stringifying_sdk_wrappers(
    fake_fo, review, monkeypatch, stage, failure,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "keep human notes when diagnostic error text is unsafe"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    str_calls = Mock(side_effect=AssertionError("opaque wrapper must not be stringified"))

    class OpaqueWrapperError(RuntimeError):
        def __str__(self):
            return str_calls()

    if failure == "private-details":
        error = RuntimeError("mongodb://review-user:secret-password@private-host/review")
        expected = "RuntimeError"
    else:
        error = OpaqueWrapperError("private driver details")
        expected = "OpaqueWrapperError"
    helper = "_ensure_schema" if stage == "schema" else "_save_views"
    monkeypatch.setattr(adapter, helper, Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        _import(review)
    assert caught.value is error
    info = dataset.persisted_info[adapter._INFO_KEY]
    assert info["state"] == "failed" and info["error"] == expected
    assert "secret-password" not in info["error"] and "private-host" not in info["error"]
    str_calls.assert_not_called()
    assert sample.stored["argus_scene_human_note"] == (
        "keep human notes when diagnostic error text is unsafe"
    )
    assert sample.stored["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert fake_fo.locks.documents == {}


@pytest.mark.parametrize("failure", [
    "before", "after", "unacknowledged", "numeric-ack", "missing-ack",
    "zero-count", "boolean-count", "float-count", "missing-count", "replaced-owner",
])
def test_unconfirmed_database_lock_release_reports_failure_without_rewriting_human_notes(
    fake_fo, review, failure,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "manual record survives uncertain lock cleanup"
    sample["argus_scene_reviewed_report_sha256"] = "b" * 64
    sample.save()
    registry = deepcopy(dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY])
    offset = len(sample.writes)
    if failure in {"before", "after"}:
        fake_fo.locks.delete_fault = failure
    elif failure in {"unacknowledged", "numeric-ack"}:
        fake_fo.locks.delete_acknowledged = False if failure == "unacknowledged" else 1
    elif failure == "replaced-owner":
        def replace_owner(query):
            fake_fo.locks.documents[query["_id"]]["token"] = "replacement-owner"

        fake_fo.locks.before_delete = replace_owner
    else:
        delete = fake_fo.locks.delete_one.side_effect

        def unconfirmed_result(query):
            delete(query)
            if failure == "missing-ack":
                return SimpleNamespace(deleted_count=1)
            if failure == "missing-count":
                return SimpleNamespace(acknowledged=True)
            return SimpleNamespace(acknowledged=True, deleted_count={
                "zero-count": 0, "boolean-count": True, "float-count": 1.0,
            }[failure])

        fake_fo.locks.delete_one.side_effect = unconfirmed_result
    with pytest.raises(RuntimeError, match="database import lock release is unconfirmed"):
        _import(review)
    assert dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY] == registry
    assert dataset.persisted_info[adapter._INFO_KEY]["state"] == "complete"
    assert sample.stored["argus_scene_human_note"] == (
        "manual record survives uncertain lock cleanup"
    )
    assert sample.stored["argus_scene_reviewed_report_sha256"] == "b" * 64
    assert all(not set(adapter._HUMAN_FIELDS) & fields for fields in sample.writes[offset:])
    if failure == "replaced-owner":
        assert fake_fo.locks.documents["vfr-images"]["token"] == "replacement-owner"
    elif failure == "before":
        retained = deepcopy(fake_fo.locks.documents)
        reads = fake_fo.collection.find_one.call_count
        with pytest.raises(RuntimeError, match="database import lock could not be acquired"):
            _import(review)
        assert fake_fo.locks.documents == retained
        assert fake_fo.collection.find_one.call_count == reads


@pytest.mark.parametrize("module_name", ["fiftyone", "fiftyone.core.sample"])
def test_sdk_boundary_rejects_preloaded_sdk_without_loading_saving_or_losing_drafts(
    fake_fo, review, monkeypatch, module_name,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "unsaved cache must not be adopted or flushed"
    draft = deepcopy(dict(sample))
    dirty = set(sample.dirty)
    saved = deepcopy(dataset.persisted_info)
    sample_writes = len(sample.writes)
    reloads = sample.reload_count
    loads = fake_fo.fo.load_dataset.call_count
    imports, database = _real_sdk_boundary(
        fake_fo, monkeypatch, modules={module_name: fake_fo.fo},
    )
    # Even switching an already-used SDK's option off is not proof that its
    # Sample singleton cache is empty. The process is rejected before use.
    assert fake_fo.fo.config.singleton_cache is False
    with pytest.raises(RuntimeError, match="fresh independent scene-images-import CLI process"):
        _import(review)
    imports.assert_not_called()
    database.get_db_conn.assert_not_called()
    assert fake_fo.fo.load_dataset.call_count == loads
    assert sample.reload_count == reloads and len(sample.writes) == sample_writes
    assert dict(sample) == draft and sample.dirty == dirty
    assert dataset.persisted_info == saved
    assert dataset.hard_refresh_writes == dataset.cached_load_writes == 0


@pytest.mark.parametrize("module_path", [None, [], ()])
def test_sdk_boundary_sets_safe_options_before_first_import_and_allows_one_attempt(
    fake_fo, monkeypatch, module_path,
):
    imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    fake_fo.fo.config.module_path = module_path
    original_import = imports.side_effect

    def guarded_import(name):
        assert os.environ["FIFTYONE_SINGLETON_CACHE"] == "false"
        assert os.environ["FIFTYONE_DATABASE_ADMIN"] == "false"
        assert os.environ["FIFTYONE_MODULE_PATH"] == ""
        return original_import(name)

    imports.side_effect = guarded_import
    assert adapter._load_review_sdk() == (fake_fo.fo, fake_fo.collection)
    assert [call.args[0] for call in imports.call_args_list] == [
        "fiftyone", "fiftyone.core.odm.database",
    ]
    database.get_db_conn.assert_called_once_with()
    with pytest.raises(RuntimeError, match="fresh independent scene-images-import CLI process"):
        adapter._load_review_sdk()
    assert imports.call_count == 2


@pytest.mark.parametrize("damage", [
    "missing-config", "missing-cache-option", "cache-true", "cache-zero",
    "missing-admin-option", "admin-true", "admin-zero", "hook-string",
    "hook-list", "hook-tuple", "hook-false",
])
def test_sdk_boundary_refuses_missing_or_unenforced_capabilities_before_db_access(
    fake_fo, monkeypatch, damage,
):
    imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    config = fake_fo.fo.config
    if damage == "missing-config":
        del fake_fo.fo.config
    elif damage == "missing-cache-option":
        del config.singleton_cache
    elif damage == "missing-admin-option":
        del config.database_admin
    elif damage in {"cache-true", "cache-zero"}:
        config.singleton_cache = True if damage == "cache-true" else 0
    elif damage in {"admin-true", "admin-zero"}:
        config.database_admin = True if damage == "admin-true" else 0
    else:
        config.module_path = {
            "hook-string": "", "hook-list": ["startup.hook"],
            "hook-tuple": ("startup.hook",), "hook-false": False,
        }[damage]
    with pytest.raises(RuntimeError, match="singleton_cache=False.*database_admin=False"):
        adapter._load_review_sdk()
    assert [call.args[0] for call in imports.call_args_list] == ["fiftyone"]
    database.get_db_conn.assert_not_called()
    with pytest.raises(RuntimeError, match="fresh independent scene-images-import CLI process"):
        adapter._load_review_sdk()
    assert imports.call_count == 1


def test_partial_sdk_initialization_failure_cannot_be_retried_in_same_process(
    fake_fo, monkeypatch,
):
    imports, database = _real_sdk_boundary(fake_fo, monkeypatch)
    imports.side_effect = RuntimeError("injected partial SDK initialization")
    with pytest.raises(RuntimeError, match="partial SDK initialization"):
        adapter._load_review_sdk()
    with pytest.raises(RuntimeError, match="fresh independent scene-images-import CLI process"):
        adapter._load_review_sdk()
    imports.assert_called_once_with("fiftyone")
    database.get_db_conn.assert_not_called()


def test_committed_owner_failure_requires_another_process_before_retry(
    fake_fo, review, monkeypatch,
):
    imports, _database = _real_sdk_boundary(fake_fo, monkeypatch)
    fake_fo.fo.next_create_info_save_fault = ("failed", "after")
    with pytest.raises(RuntimeError, match="metadata failure after commit"):
        _import(review)
    dataset = fake_fo.datasets["vfr-images"]
    persisted = deepcopy(dataset.persisted_info)
    reads = fake_fo.collection.find_one.call_count
    with pytest.raises(RuntimeError, match="fresh independent scene-images-import CLI process"):
        _import(review)
    assert imports.call_count == 2
    assert fake_fo.collection.find_one.call_count == reads
    assert dataset.persisted_info == persisted and dataset.samples == []


@pytest.mark.parametrize("version", ["older-sdk", "newer-sdk", None, True, ""])
def test_sdk_version_mismatch_is_rejected_before_loading_or_automatic_migration(
    fake_fo, review, version,
):
    dataset = _import(review)
    dataset._store.version = version
    loads = fake_fo.fo.load_dataset.call_count
    attempts = dataset.info_save_attempts
    persisted = deepcopy(dataset.persisted_info)
    with pytest.raises(ValueError, match="automatic migration is disabled|invalid persisted identity"):
        _import(review)
    assert fake_fo.fo.load_dataset.call_count == loads
    assert dataset.info_save_attempts == attempts and dataset.persisted_info == persisted
    assert dataset.hard_refresh_writes == dataset.cached_load_writes == 0


@pytest.mark.parametrize("change", ["id", "name", "version", "media", "info-type", "removed"])
def test_raw_identity_changes_during_sdk_load_are_rejected_before_import_writes(
    fake_fo, review, change,
):
    dataset = _import(review)
    before = fake_fo.collection.find_one({"name": dataset.name}, adapter._DATASET_PROJECTION)
    after = deepcopy(before)
    if change == "removed":
        after = None
    elif change == "id":
        after["_id"] = "another-dataset-id"
    elif change == "name":
        after["name"] = "another-name"
    elif change == "version":
        after["version"] = "another-sdk"
    elif change == "media":
        after["media_type"] = "video"
    else:
        # Canonical comparison must not accept True == 1 in Python dicts.
        before["info"]["typed_marker"] = True
        after["info"]["typed_marker"] = 1
        dataset.persisted_info["typed_marker"] = True
    fake_fo.collection.find_one.side_effect = [before, after]
    attempts = dataset.info_save_attempts
    offsets = [len(sample.writes) for sample in dataset.samples]
    with pytest.raises(RuntimeError, match="persisted identity changed while loading"):
        _import(review)
    assert dataset.info_save_attempts == attempts
    assert [len(sample.writes) for sample in dataset.samples] == offsets
    assert dataset.hard_refresh_writes == dataset.cached_load_writes == 0


@pytest.mark.parametrize("damage", ["session-and-split", "session-only", "split-only"])
def test_historical_samples_cannot_repair_a_degraded_v2_source_registry(
    fake_fo, review, damage,
):
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "keep review while refusing degraded source ownership"
    sample.save()
    source = dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]["clips"]["clip-1"]
    if damage in {"session-and-split", "session-only"}:
        source["source_session_id"] = None
    if damage in {"session-and-split", "split-only"}:
        source["split"] = "unassigned"
    dataset.save()
    persisted = deepcopy(dataset.persisted_info)
    offsets = [len(item.writes) for item in dataset.samples]
    attempts = dataset.info_save_attempts
    with pytest.raises(ValueError, match="source registry"):
        _import(review)
    assert dataset.persisted_info == persisted and dataset.info_save_attempts == attempts
    assert [len(item.writes) for item in dataset.samples] == offsets
    assert sample.stored["argus_scene_human_note"] == (
        "keep review while refusing degraded source ownership"
    )


def test_deleted_v2_clip_entry_is_not_reconstructed_from_surviving_samples(fake_fo, review):
    _add_zero_selection_clip(review)
    dataset = _import(review)
    sample = dataset.samples[0]
    sample["argus_scene_human_note"] = "never infer lost ledger history from my note"
    sample.save()
    dataset.info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]["clips"].pop("clip-1")
    dataset.save()
    persisted = deepcopy(dataset.persisted_info)
    attempts = dataset.info_save_attempts
    offset = len(sample.writes)
    with pytest.raises(ValueError, match="absent from the durable source registry"):
        _import(review)
    assert dataset.persisted_info == persisted and dataset.info_save_attempts == attempts
    assert len(sample.writes) == offset
    assert sample.stored["argus_scene_human_note"] == "never infer lost ledger history from my note"


def test_unregistered_historical_sample_cannot_extend_complete_v2_ownership(fake_fo, review):
    dataset = _import(review)
    existing = dataset.samples[0]
    injected = _Sample(existing.filepath)
    for field, value in existing.stored.items():
        injected[field] = deepcopy(value)
    injected["argus_clip_id"] = "unregistered-clip"
    injected["argus_scene_image_key"] = adapter._image_key(
        injected["argus_camera_id"], "unregistered-clip", injected["argus_video_sha256"],
        injected["argus_frame_index"],
    )
    injected["argus_scene_human_note"] = "unregistered sample note must remain untouched"
    dataset.add_sample(injected)
    persisted = deepcopy(dataset.persisted_info)
    attempts = dataset.info_save_attempts
    offset = len(injected.writes)
    with pytest.raises(ValueError, match="absent from the durable source registry"):
        _import(review)
    assert dataset.persisted_info == persisted and dataset.info_save_attempts == attempts
    assert len(injected.writes) == offset
    assert injected.stored["argus_scene_human_note"] == (
        "unregistered sample note must remain untouched"
    )


def test_current_bundle_can_introduce_a_new_clip_without_historical_sample_adoption(fake_fo, review):
    dataset = _import(review)
    original_registry = deepcopy(dataset.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY])
    _switch_review_clip(review)
    fresh = _import(review)
    registry = fresh.persisted_info[adapter._INFO_KEY][adapter._SOURCE_REGISTRY_KEY]
    assert registry["clips"]["clip-1"] == original_registry["clips"]["clip-1"]
    assert registry["clips"]["clip-2"]["source_session_id"] == "session-2"
    assert len(fresh.samples) == 6 and fresh.info[adapter._INFO_KEY]["state"] == "complete"
