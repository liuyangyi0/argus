"""Event collection uses synthetic images and never opens a capture/model."""

from copy import deepcopy
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from argus.capture.task_event_collector import TaskEventCollector
from argus.core.frame_identity import build_frame_identity


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def binding():
    return {"task_id": "task-one", "camera_id": "cam", "template_id": "object_detection.v1",
            "task_revision": 1, "task_config_sha256": "a" * 64}


def context():
    return {"task_id": "task-one", "bundle_id": "bundle-one", "manifest_sha256": "b" * 64,
            "execution_id": "worker-one", "deployment_revision": 1,
            "deployment_operation_id": "operation-one", "source_session_id": "session", "source_epoch": 1}


class Events:
    def __init__(self):
        self.current = True
        self.rows = []
        self.evidence = {}

    def collection_context_current(self, task_binding, frozen):
        assert task_binding == binding() and frozen == context()
        return self.current

    def collection_events(self, task_binding, after_cursor, *, limit, context):
        return deepcopy([row for row in self.rows if row["cursor"] > after_cursor][:limit])

    def collection_evidence(self, task_binding, event_id, expected_sha, *, context):
        value = self.evidence[event_id]
        assert expected_sha == digest({"payload": value["payload"], "evidence_sha256": value["evidence_sha256"]})
        return deepcopy(value)

    def add(self, cursor=6):
        ok, encoded = cv2.imencode(".jpg", np.zeros((2, 3, 3), dtype=np.uint8))
        assert ok
        content = encoded.tobytes()
        identity = build_frame_identity(camera_id="cam", source_declared="mock", source_runtime="mock",
            stream_id=None, frame_index=cursor, source_session_id="session", source_epoch=1,
            source_frame_index=99, pts=123, time_base_num=1, time_base_den=90000).to_dict()
        identity = {key: identity[key] for key in (
            "camera_id", "source_session_id", "source_epoch", "frame_index", "source_frame_index", "pts",
            "time_base_num", "time_base_den", "frame_index_provenance", "timestamp_provenance",
            "capture_wall_ts", "capture_monotonic_ts")}
        identity["source_runtime_hash"] = hashlib.sha1(b"mock").hexdigest()[:10]
        identity["capture_monotonic_ts"] = float(cursor)
        event_id = f"event-{cursor}"
        payload = {"schema_version": "argus.task-event.v1", "event_type": "object_detected",
                   "event_id": event_id, "binding": {**binding(), **context()}, "source": identity,
                   "first_source": deepcopy(identity), "frame_size": {"width": 3, "height": 2},
                   "replay": {"sha256": hashlib.sha256(content).hexdigest()}}
        evidence = {"payload": payload, "source": identity, "bytes": content, "mime": "image/jpeg",
                    "image_size": [3, 2], "evidence_sha256": hashlib.sha256(content).hexdigest()}
        self.evidence[event_id] = evidence
        self.rows.append({"cursor": cursor, "event_id": event_id,
                          "immutable_sha256": digest({"payload": payload, "evidence_sha256": evidence["evidence_sha256"]})})


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("synthetic event collector did not reach expected state")
        time.sleep(0.005)


def worker(tmp_path, events, **kwargs):
    manager = SimpleNamespace(get_camera_source=lambda _: pytest.fail("event collector must not read camera"))
    options = {"interval": 0.1, "max_duration": 0.5, **kwargs}
    return TaskEventCollector(manager, binding(), tmp_path, "collection", event_provider=events,
                              event_context=context(), start_cursor=5, **options)


def test_committed_keyframe_keeps_exact_identity_and_derivation(tmp_path):
    events = Events()
    events.add(4)  # Before the frozen cursor: never imported as a new event.
    collector = worker(tmp_path, events, max_samples=1)
    collector.start()
    events.add()
    wait_for(lambda: collector.snapshot()["state"] == "completed")
    assert collector.stop()
    snapshot = collector.snapshot()
    sample = snapshot["samples"][0]
    assert sample["identity"]["source_frame_index"] == 99
    assert sample["identity"]["pts"] == 123
    assert sample["event_origin"]["event_id"] == "event-6"
    assert sample["event_origin"]["derivation"] == "verified_jpeg_to_png.v1"
    assert sample["auto_annotation"] is None
    assert sample["binding"] == binding()
    assert (tmp_path / sample["relative_path"]).read_bytes().startswith(b"\x89PNG")
    assert sample["image_sha256"] != sample["event_origin"]["evidence_sha256"]
    assert snapshot["last_cursor"] == 6 and snapshot["start_cursor"] == 5


def test_late_encoded_event_after_restart_is_not_written(tmp_path):
    events = Events()
    entered, release = threading.Event(), threading.Event()

    def encode(pixels):
        entered.set()
        assert release.wait(3)
        return cv2.imencode(".png", pixels)[1].tobytes()

    collector = worker(tmp_path, events, encoder=encode)
    collector.start()
    events.add()
    assert entered.wait(3)
    events.current = False
    release.set()
    wait_for(lambda: collector.snapshot()["state"] == "failed")
    assert collector.stop()
    assert collector.snapshot()["error_code"] == "EVENT_EXECUTION_CHANGED"
    assert not list(tmp_path.rglob("*.png"))


def test_commit_retry_keeps_the_same_receipt_and_images(tmp_path):
    events, attempts = Events(), []

    def finish(receipt):
        attempts.append(receipt)
        if len(attempts) == 1:
            raise RuntimeError("unknown database acknowledgement")

    collector = worker(tmp_path, events, max_samples=1, on_finish=finish)
    collector.start()
    events.add()
    wait_for(lambda: collector.snapshot()["error_code"] == "MANIFEST_COMMIT_FAILED")
    assert collector.stop() and collector.retry_finalize()
    assert attempts[0] == attempts[1]
    assert len(list(tmp_path.rglob("*.png"))) == 1
    assert collector.snapshot()["persistence_pending"] is False


@pytest.mark.parametrize("change", ["digest", "size", "mime"])
def test_replaced_or_malformed_evidence_never_writes_a_sample(tmp_path, change):
    events = Events()
    events.add()
    evidence = events.evidence["event-6"]
    if change == "digest":
        evidence["bytes"] += b"tampered"
    elif change == "size":
        evidence["image_size"] = [20, 30]
    else:
        evidence["mime"] = "image/png"
    collector = worker(tmp_path, events)
    collector.start()
    wait_for(lambda: collector.snapshot()["state"] == "failed")
    assert collector.stop()
    assert collector.snapshot()["samples"] == []
    assert not list(tmp_path.rglob("*.png"))


def test_distinct_event_frames_inside_the_minimum_interval_are_throttled(tmp_path):
    events = Events()
    events.add(6)
    events.add(7)
    events.evidence["event-7"]["source"]["capture_monotonic_ts"] = 6.05
    events.evidence["event-7"]["payload"]["first_source"]["capture_monotonic_ts"] = 6.05
    events.rows[1]["immutable_sha256"] = digest({"payload": events.evidence["event-7"]["payload"],
                                                "evidence_sha256": events.evidence["event-7"]["evidence_sha256"]})
    collector = worker(tmp_path, events)
    collector.start()
    wait_for(lambda: collector.snapshot()["last_cursor"] == 7)
    assert collector.stop()
    assert len(collector.snapshot()["samples"]) == 1
    assert collector.snapshot()["throttled_events"] == 1


@pytest.mark.parametrize("interval", [0.0, 0.1])
def test_same_exact_frame_is_deduplicated_even_without_an_interval(tmp_path, interval):
    events = Events()
    events.add(6)
    events.add(7)
    duplicate_source = deepcopy(events.evidence["event-6"]["source"])
    events.evidence["event-7"]["source"] = duplicate_source
    events.evidence["event-7"]["payload"]["source"] = duplicate_source
    events.evidence["event-7"]["payload"]["first_source"] = deepcopy(duplicate_source)
    events.rows[1]["immutable_sha256"] = digest({"payload": events.evidence["event-7"]["payload"],
                                                "evidence_sha256": events.evidence["event-7"]["evidence_sha256"]})
    collector = worker(tmp_path, events, interval=interval)
    collector.start()
    wait_for(lambda: collector.snapshot()["last_cursor"] == 7)
    assert collector.stop()
    assert len(collector.snapshot()["samples"]) == 1
    assert collector.snapshot()["duplicate_frame_events"] == 1
    assert collector.snapshot()["throttled_events"] == 0


def test_same_frame_key_cannot_acquire_different_original_pts(tmp_path):
    events = Events()
    events.add(6)
    events.add(7)
    changed_source = {**events.evidence["event-6"]["source"], "pts": 124}
    events.evidence["event-7"]["source"] = changed_source
    events.evidence["event-7"]["payload"]["source"] = changed_source
    events.evidence["event-7"]["payload"]["first_source"] = deepcopy(changed_source)
    events.rows[1]["immutable_sha256"] = digest({"payload": events.evidence["event-7"]["payload"],
                                                "evidence_sha256": events.evidence["event-7"]["evidence_sha256"]})
    collector = worker(tmp_path, events)
    collector.start()
    wait_for(lambda: collector.snapshot()["state"] == "failed")
    assert collector.stop()
    assert collector.snapshot()["error_code"] == "EVENT_FRAME_IDENTITY_CHANGED"
    assert len(collector.snapshot()["samples"]) == 1
    assert len(list(tmp_path.rglob("*.png"))) == 1


def test_previously_committed_workflow_frame_is_skipped_without_new_image(tmp_path):
    events = Events()
    events.add()
    original_source = deepcopy(events.evidence["event-6"]["source"])
    collector = worker(tmp_path, events, existing_sources=[original_source])
    original_source["pts"] = 999  # Caller mutation cannot change the frozen ledger.
    collector.start()
    wait_for(lambda: collector.snapshot()["last_cursor"] == 6)
    assert collector.stop()
    assert collector.snapshot()["existing_frame_count"] == 1
    assert collector.snapshot()["duplicate_frame_events"] == 1
    assert collector.snapshot()["samples"] == []
    assert list(tmp_path.rglob("*.png")) == []


def test_previously_committed_frame_cannot_be_rebound_to_another_pts(tmp_path):
    events = Events()
    events.add()
    historical = {**events.evidence["event-6"]["source"], "pts": 122}
    collector = worker(tmp_path, events, existing_sources=[historical])
    collector.start()
    wait_for(lambda: collector.snapshot()["state"] == "failed")
    assert collector.stop()
    assert collector.snapshot()["error_code"] == "EVENT_FRAME_IDENTITY_CHANGED"
    assert collector.snapshot()["samples"] == []
    assert list(tmp_path.rglob("*.png")) == []


def test_existing_frame_ledger_never_silently_evicts_old_deduplication_evidence(tmp_path):
    events = Events()
    events.add()
    with pytest.raises(RuntimeError, match="EVENT_COLLECTION_HISTORY_LIMIT"):
        worker(tmp_path, events, existing_sources=[events.evidence["event-6"]["source"]]
               * (TaskEventCollector.MAX_EXISTING_FRAMES + 1))


def test_stop_timeout_retains_work_until_encoder_exits(tmp_path):
    events = Events()
    entered, release = threading.Event(), threading.Event()

    def encode(pixels):
        entered.set()
        assert release.wait(3)
        return cv2.imencode(".png", pixels)[1].tobytes()

    collector = worker(tmp_path, events, encoder=encode)
    collector.start()
    events.add()
    assert entered.wait(3)
    assert collector.stop(timeout=0) is False
    release.set()
    assert collector.stop()
    assert collector.snapshot()["samples"] == []


def test_start_exception_after_thread_creation_still_joins_before_finalizing(tmp_path, monkeypatch):
    collector = worker(tmp_path, Events())
    original = threading.Thread.start

    def start_then_fail(thread):
        original(thread)
        raise RuntimeError("late start acknowledgement failure")

    monkeypatch.setattr(threading.Thread, "start", start_then_fail)
    with pytest.raises(RuntimeError, match="late start"):
        collector.start()
    assert collector.finish_start_failure()
    assert not collector._thread.is_alive()
    assert collector.snapshot()["cleanup_confirmed"] is True


def test_unknown_start_state_does_not_certify_cleanup(tmp_path):
    collector = worker(tmp_path, Events())
    collector._thread = SimpleNamespace(is_alive=lambda: None, _started=threading.Event())
    assert collector.finish_start_failure() is False
    assert collector.snapshot()["cleanup_confirmed"] is False


def test_unstarted_thread_failure_remains_stoppable_and_retryable(tmp_path, monkeypatch):
    collector = worker(tmp_path, Events())

    def never_started(thread):
        raise RuntimeError("OS thread creation failed")

    monkeypatch.setattr(threading.Thread, "start", never_started)
    with pytest.raises(RuntimeError, match="creation failed"):
        collector.start()
    assert collector.finish_start_failure()
    assert collector._thread is None
    assert collector.stop()
    assert collector.retry_finalize()


@pytest.mark.parametrize("alive,started,ident", [
    (None, True, 123), (0, True, 123), (False, False, 123), (False, True, None),
])
def test_unknown_thread_state_cannot_stop_or_replay_a_frozen_receipt(tmp_path, alive, started, ident):
    finalized = []
    collector = worker(tmp_path, Events(), on_finish=finalized.append)
    marker = threading.Event()
    if started:
        marker.set()
    collector._thread = SimpleNamespace(is_alive=lambda: alive, _started=marker, ident=ident,
                                         join=lambda **kwargs: None)
    collector._finish_snapshot = {"state": "failed", "error_code": "frozen-error"}
    collector._cleanup_confirmed = True  # A prior cached observation is not fresh proof.
    assert collector.stop(timeout=0) is False
    assert collector.retry_finalize() is False
    assert collector.snapshot()["cleanup_confirmed"] is False
    assert finalized == []


def test_unknown_start_can_recover_after_os_confirms_it_never_started(tmp_path):
    finalized = []
    collector = worker(tmp_path, Events(), on_finish=finalized.append)
    marker = threading.Event()
    collector._thread = SimpleNamespace(is_alive=lambda: None, _started=marker, ident=None)
    assert collector.finish_start_failure() is False
    assert not finalized
    collector._thread.is_alive = lambda: False
    assert collector.stop(timeout=0)
    assert collector._thread is None
    assert collector.retry_finalize()
    assert finalized[0]["state"] == "failed"
    assert finalized[0]["error_code"] == "COLLECTION_START_FAILED"


@pytest.mark.parametrize("change", ["source", "event", "digest", "binding", "context", "size_type"])
def test_provider_envelope_mismatch_is_rejected_before_any_image_write(tmp_path, change):
    events = Events()
    events.add()
    original = events.collection_evidence

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "source":
            result["source"] = {**result["source"], "pts": 124}
        elif change == "event":
            result["payload"]["event_id"] = "another-event"
        elif change == "digest":
            result["payload"]["uncommitted"] = True
        elif change == "binding":
            result["payload"]["binding"]["camera_id"] = "other-camera"
        elif change == "context":
            result["payload"]["binding"]["execution_id"] = "other-worker"
        else:
            result["image_size"] = [3.0, 2]
        return result

    events.collection_evidence = changed
    collector = worker(tmp_path, events)
    collector.start()
    wait_for(lambda: collector.snapshot()["state"] == "failed")
    assert collector.stop()
    assert collector.snapshot()["samples"] == []
    assert list(tmp_path.rglob("*.png")) == []


def test_internally_consistent_provider_digest_cannot_change_the_camera(tmp_path):
    events = Events()
    events.add()
    events.evidence["event-6"]["source"]["camera_id"] = "another-camera"
    events.rows[0]["immutable_sha256"] = digest({"payload": events.evidence["event-6"]["payload"],
                                                "evidence_sha256": events.evidence["event-6"]["evidence_sha256"]})
    collector = worker(tmp_path, events)
    collector.start()
    wait_for(lambda: collector.snapshot()["state"] == "failed")
    assert collector.stop()
    assert collector.snapshot()["samples"] == []
    assert list(tmp_path.rglob("*.png")) == []


@pytest.mark.parametrize("pixels", [
    np.zeros((2, 3, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.uint8),
    np.zeros((2, 4, 3), dtype=np.uint8), None,
])
def test_decoder_contract_failure_does_not_reach_encoder_or_image_write(tmp_path, pixels):
    events = Events()
    events.add()
    collector = worker(tmp_path, events, decoder=lambda *_: pixels,
                       encoder=lambda *_: pytest.fail("invalid decoded pixels must not be encoded"))
    collector.start()
    wait_for(lambda: collector.snapshot()["state"] == "failed")
    assert collector.stop()
    assert collector.snapshot()["samples"] == []
    assert list(tmp_path.rglob("*.png")) == []


def test_failed_cleanup_is_not_committed_but_can_gain_its_first_positive_proof(tmp_path, monkeypatch):
    events, receipts = Events(), []
    events.add()
    collector = worker(tmp_path, events, max_samples=1, on_finish=receipts.append)
    original = collector._cleanup
    calls = []

    def cleanup():
        calls.append(True)
        if len(calls) == 1:
            return False
        return original()

    monkeypatch.setattr(collector, "_cleanup", cleanup)
    collector.start()
    wait_for(lambda: collector._finish_snapshot is not None)
    assert receipts == []
    snapshot = collector.snapshot()
    assert snapshot["cleanup_confirmed"] is False and snapshot["persistence_pending"] is True
    assert collector.stop() and collector.retry_finalize()
    assert len(receipts) == 1 and receipts[0]["cleanup_confirmed"] is True
    assert receipts[0]["samples"] == snapshot["samples"]
    assert len(list(tmp_path.rglob("*.png"))) == 1
