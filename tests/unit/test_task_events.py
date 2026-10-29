"""Object-task event workflow with synthetic pixels/results, never inference."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from argus.core.analysis_task_contracts import get_default_config
from argus.core.task_events import TaskEventEngine, TaskEventValidationError, validate_observation
from argus.core.task_execution import frame_provenance
from argus.capture.task_execution_manager import TaskExecutionManager
from argus.storage.database import Database
from argus.storage.models import (
    AlertRecord, AnalysisTaskRecord, AnalysisTaskRevisionRecord, OutboxEvent,
    TaskDeploymentOperationRecord, TaskDeploymentRecord, TaskEventRecord,
    TaskExecutionBundleRecord, TaskModelRecord,
)
from argus.storage.task_events import TaskEventConflict, TaskEventEvidenceUnavailable, TaskEventService
from test_task_execution import synthetic_frame
from test_task_execution_manager import FakeTaskService, FakeCameraManager, FakeWorker, wait_for


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def manifest_for(tmp_path, *, duration=0.0, cooldown=30.0, collection=None):
    config = get_default_config("object_detection.v1")
    config["rules"].update(labels=["tool"], min_duration_seconds=duration, cooldown_seconds=cooldown)
    if collection is not None:
        config["collection"].update(deepcopy(collection))
    descriptor = {"name": "Synthetic only", "adapter_id": "onnx_detection.v1",
                  "output_schema": "yolov8_detection.v1", "labels": ["tool", "person"],
                  "input_size": [32, 32], "input_layout": "NCHW", "input_dtype": "float32", "color_order": "RGB"}
    stored = {"schema_version": "argus.task-execution-bundle.v1", "task_id": "task-1", "camera_id": "cam",
              "bundle_id": "bundle-1", "task_revision": 1, "task_config_sha256": sha(config),
              "template_id": "object_detection.v1", "authority": "managed", "stage": "shadow", "config": config,
              "models": {"detector": {"model_id": "model-1", "artifact_sha256": "a" * 64, **descriptor}}}
    manifest = deepcopy(stored)
    manifest["manifest_sha256"] = sha(stored)
    manifest["models"]["detector"]["artifact_path"] = str(tmp_path / "not-executed.onnx")
    return stored, manifest, descriptor


def observation(manifest, *, number=1, timestamp=10.0, epoch=1, box=None, label="tool", confidence=0.9):
    frame = synthetic_frame(number, timestamp, epoch)
    result = {"schema_version": "argus.task-shadow-result.v1", "execution_id": "execution-1", "request_id": number,
              **{key: manifest[key] for key in ("task_id", "task_revision", "task_config_sha256", "manifest_sha256", "bundle_id", "template_id")},
              "source": frame_provenance(frame), "production_allowed": False, "alerts_emitted": False,
              "models": {"detector": {"model_id": "model-1", "artifact_sha256": "a" * 64}},
              "data": {"state": "observed", "detections": [{"label": label, "confidence": confidence,
                        "bbox_xyxy": box or [0.0, 0.0, 3.0, 3.0]}]}}
    return result, frame


@pytest.fixture
def events(tmp_path, request):
    db = Database(f"sqlite:///{tmp_path / 'events.db'}")
    db.initialize()
    options = getattr(request, "param", 0.0)
    if isinstance(options, dict):
        stored, manifest, descriptor = manifest_for(
            tmp_path, duration=options.get("duration", 0.0), collection=options.get("collection"))
    else:
        stored, manifest, descriptor = manifest_for(tmp_path, duration=options)
    now = datetime.now(timezone.utc)
    with db.get_session() as session:
        session.add(AnalysisTaskRecord(task_id="task-1", camera_id="cam", template_id="object_detection.v1",
                    authority="managed", name="Object", current_revision=1, archived=False, created_by="human"))
        session.add(AnalysisTaskRevisionRecord(task_id="task-1", revision=1,
                    config_json=canonical(stored["config"]), config_sha256=stored["task_config_sha256"]))
        session.add(TaskModelRecord(model_id="model-1", authority="managed", relative_path="not-executed.onnx",
                    artifact_sha256="a" * 64, descriptor_json=canonical(descriptor), descriptor_sha256=sha(descriptor),
                    created_by="human"))
        session.add(TaskExecutionBundleRecord(bundle_id="bundle-1", task_id="task-1", task_revision=1,
                    template_id="object_detection.v1", manifest_json=canonical(stored), manifest_sha256=sha(stored),
                    created_by="human"))
        session.add(TaskDeploymentRecord(task_id="task-1", authority="managed", projection_status="managed",
                    runtime_status="running", desired_state="running", deployment_revision=1,
                    desired_bundle_id="bundle-1", applied_bundle_id="bundle-1", applied_deployment_revision=1,
                    current_operation_id="operation-1"))
        session.add(TaskDeploymentOperationRecord(operation_id="operation-1", task_id="task-1", idempotency_key="deploy-1",
                    request_sha256="b" * 64, action="deploy", bundle_id="bundle-1", deployment_revision=1,
                    status="applied", requested_by="human", created_at=now))
        session.commit()
    current = {"task_id": "task-1", "bundle_id": "bundle-1", "manifest_sha256": manifest["manifest_sha256"],
               "execution_id": "execution-1", "deployment_revision": 1, "deployment_operation_id": "operation-1",
               "source_session_id": "session", "source_epoch": 1}
    release = SimpleNamespace(authorize_event=lambda *args, **kwargs: {
        "authorized": False, "release_id": None, "reasons": ["TEMPLATE_POLICY_NOT_APPROVED"]})
    service = TaskEventService(db, None, evidence_root=tmp_path / "evidence", release_service=release,
                               current_runtime=lambda task_id: current)
    yield SimpleNamespace(db=db, manifest=manifest, current=current, release=release, service=service)
    service.close()
    db.close()


def process(platform, **kwargs):
    result, frame = observation(platform.manifest, **kwargs)
    return platform.service.process_observation(platform.manifest, result, frame,
                                                deployment_revision=1, operation_id="operation-1")


def count(db, model):
    with db.get_session() as session:
        return session.scalar(select(func.count()).select_from(model))


def test_validation_event_outbox_keyframe_and_original_identity(events):
    records = process(events)
    assert len(records) == 1
    event = records[0]
    assert event["mode"] == "validation" and event["alert_id"] is None
    assert event["payload"]["production_allowed"] is False
    assert event["payload"]["source"]["source_frame_index"] is None
    assert event["payload"]["source"]["pts"] is None
    assert count(events.db, TaskEventRecord) == count(events.db, OutboxEvent) == 1
    assert count(events.db, AlertRecord) == 0
    data, mime = events.service.load_evidence("task-1", event["event_id"])
    assert mime == "image/jpeg" and data.startswith(b"\xff\xd8")
    assert hashlib.sha256(data).hexdigest() == event["evidence_sha256"]
    replay = events.service.replay_metadata("task-1", event["event_id"])
    assert replay["kind"] == "keyframe_only" and replay["video_available"] is False and replay["ready"]
    assert replay["source"]["frame_index"] == 1
    assert events.service.get("other-task", event["event_id"]) is None
    with pytest.raises(LookupError):
        events.service.load_evidence("other-task", event["event_id"])


@pytest.mark.parametrize("field", ["execution_id", "bundle_id", "deployment_revision", "deployment_operation_id",
                                  "source_session_id", "source_epoch"])
def test_late_result_does_not_create_event_or_alert(events, field):
    events.current[field] = "other" if field not in {"deployment_revision", "source_epoch"} else 2
    assert process(events) == []
    assert count(events.db, TaskEventRecord) == count(events.db, AlertRecord) == count(events.db, OutboxEvent) == 0


@pytest.mark.parametrize("mutate", [
    lambda result: result.update(production_allowed=True),
    lambda result: result["models"]["detector"].update(artifact_sha256="f" * 64),
    lambda result: result["source"].update(pts=10),
    lambda result: result["data"]["detections"][0].update(label="not-registered"),
    lambda result: result["data"]["detections"][0].update(confidence=float("nan")),
    lambda result: result["data"]["detections"][0].update(bbox_xyxy=[0, 0, 99, 99]),
])
def test_invalid_model_frame_or_worker_flags_rejected(events, mutate):
    result, frame = observation(events.manifest)
    mutate(result)
    with pytest.raises((TaskEventValidationError, ValueError)):
        events.service.process_observation(events.manifest, result, frame, deployment_revision=1, operation_id="operation-1")
    assert count(events.db, TaskEventRecord) == 0


def test_duration_cooldown_and_same_frame_deduplication(tmp_path):
    _, manifest, _ = manifest_for(tmp_path, duration=0.5, cooldown=30.0)
    engine = TaskEventEngine()
    def apply(number, timestamp):
        result, frame = observation(manifest, number=number, timestamp=timestamp)
        return engine.observe(manifest, result, frame, deployment_revision=1, operation_id="operation-1")
    assert apply(1, 0.0) == []
    assert apply(2, 0.5)[0]["duration_seconds"] == 0.5
    assert apply(2, 0.5) == []
    assert apply(3, 0.8) == []
    # A sampling gap does not inherit the previous object's confirmed duration.
    assert apply(4, 2.0) == []
    assert apply(5, 2.5)[0]["first_source"]["frame_index"] == 4


def test_deployment_change_cannot_inherit_temporal_state(tmp_path):
    _, manifest, _ = manifest_for(tmp_path, duration=0.5)
    engine = TaskEventEngine()
    result, frame = observation(manifest)
    assert engine.observe(manifest, result, frame, deployment_revision=1, operation_id="operation-1") == []
    result, frame = observation(manifest, number=2, timestamp=10.5)
    assert engine.observe(manifest, result, frame, deployment_revision=2, operation_id="operation-2") == []
    with pytest.raises(TaskEventValidationError, match="superseded"):
        engine.observe(manifest, result, frame, deployment_revision=1, operation_id="operation-1")


def test_database_stopped_or_changed_revision_blocks_pending_event(events):
    with events.db.get_session() as session:
        session.get(TaskDeploymentRecord, "task-1").desired_state = "stopped"
        session.commit()
    assert process(events) == []
    assert count(events.db, TaskEventRecord) == 0


def test_keyframe_failure_is_explicit_and_never_a_production_alert(events, monkeypatch):
    monkeypatch.setattr(events.service, "_encoder", lambda frame: (_ for _ in ()).throw(OSError("fake encode")))
    event = process(events)[0]
    assert event["evidence_status"] == "failed" and event["mode"] == "validation"
    assert count(events.db, AlertRecord) == 0
    assert not events.service.replay_metadata("task-1", event["event_id"])["ready"]
    with pytest.raises(TaskEventEvidenceUnavailable):
        events.service.load_evidence("task-1", event["event_id"])


def test_review_cas_notes_and_retry_preserve_receipts(events):
    event = process(events)[0]
    reviewed = events.service.review("task-1", event["event_id"], expected_revision=1,
                                     status="acknowledged", notes="keep this note", actor="reviewer")
    assert reviewed["revision"] == 2 and reviewed["notes"] == "keep this note"
    assert events.service.review("task-1", event["event_id"], expected_revision=1,
        status="acknowledged", notes="keep this note", actor="reviewer") == reviewed
    assert count(events.db, OutboxEvent) == 2
    with pytest.raises(TaskEventConflict):
        events.service.review("task-1", event["event_id"], expected_revision=1,
                              status="resolved", notes="stale overwrite", actor="reviewer")
    latest = events.service.review("task-1", event["event_id"], expected_revision=2,
                                   status="resolved", notes=None, actor="other-reviewer")
    assert latest["notes"] == "keep this note"
    assert len(latest["payload"]["review_history"]) == 2


def test_keyframe_tamper_does_not_look_like_available_replay(events):
    event = process(events)[0]
    with events.db.get_session() as session:
        record = session.get(TaskEventRecord, event["event_id"])
        path = events.service._root / record.evidence_path
    path.write_bytes(b"tampered synthetic evidence")
    assert not events.service.replay_metadata("task-1", event["event_id"])["ready"]


def test_database_commit_ack_loss_recovers_same_event_without_duplicate(events, monkeypatch):
    original = events.service._persist
    calls = []
    def lose_ack(event, encoded, error):
        record = original(event, encoded, error)
        calls.append(event["event_id"])
        if len(calls) == 1:
            raise OSError("simulated lost commit acknowledgement")
        return record
    monkeypatch.setattr(events.service, "_persist", lose_ack)
    event = process(events)[0]
    assert calls == [event["event_id"], event["event_id"]]
    assert count(events.db, TaskEventRecord) == count(events.db, OutboxEvent) == 1


@pytest.mark.parametrize("after_commit", [False, True])
def test_real_transaction_failure_keeps_event_outbox_pair_atomic(events, monkeypatch, after_commit):
    commit = Session.commit
    calls = []
    def fail_once(session):
        calls.append(True)
        if len(calls) == 1:
            if after_commit:
                commit(session)
            raise OSError("simulated transaction acknowledgement failure")
        return commit(session)
    monkeypatch.setattr(Session, "commit", fail_once)
    event = process(events)[0]
    assert event["mode"] == "validation"
    assert count(events.db, TaskEventRecord) == count(events.db, OutboxEvent) == 1
    assert count(events.db, AlertRecord) == 0
    assert len(calls) == (1 if after_commit else 2)


def test_unknown_write_retains_exact_event_for_later_retry(events, monkeypatch):
    original = events.service._persist
    seen = []
    def unavailable(event, *args):
        seen.append(event["event_id"])
        raise OSError("simulated database unavailable")
    monkeypatch.setattr(events.service, "_persist", unavailable)
    with pytest.raises(RuntimeError, match="retry exhausted"):
        process(events)
    assert len(set(seen)) == 1 and events.service.snapshot()["pending"] == 1
    monkeypatch.setattr(events.service, "_persist", original)
    records = process(events, number=2, timestamp=10.2)
    assert len(records) == 1 and records[0]["event_id"] == seen[0]
    assert count(events.db, TaskEventRecord) == 1


def test_evidence_encode_completion_cannot_publish_after_source_replacement(events, monkeypatch):
    encode = events.service._encoder
    def replace_source(frame):
        encoded = encode(frame)
        events.current["source_epoch"] = 2
        return encoded
    monkeypatch.setattr(events.service, "_encoder", replace_source)
    assert process(events) == []
    assert count(events.db, TaskEventRecord) == count(events.db, OutboxEvent) == 0


def test_rejected_source_replacement_does_not_leave_uncommitted_keyframe(events, monkeypatch):
    encode = events.service._encoder

    def replace_source(frame):
        encoded = encode(frame)
        events.current["source_epoch"] = 2
        return encoded

    monkeypatch.setattr(events.service, "_encoder", replace_source)
    assert process(events) == []
    assert not events.service._root.exists() or not list(events.service._root.iterdir())


def test_release_read_completion_cannot_publish_after_source_generation_changes(events):
    def replace_source(*args, **kwargs):
        events.current["source_epoch"] = 2
        return {"authorized": False, "reasons": ["NO_APPROVED_TASK_RELEASE"]}

    events.release.authorize_event = replace_source
    assert process(events) == []
    assert count(events.db, TaskEventRecord) == count(events.db, OutboxEvent) == 0
    assert not events.service._root.exists() or not list(events.service._root.iterdir())


def test_replaced_source_object_cannot_keep_queued_task_observations_current(events, tmp_path):
    tasks, camera = FakeTaskService(), FakeCameraManager()
    tasks.manifests = {"task-1": events.manifest}
    tasks.tasks["task-1"].update(template_id="object_detection.v1",
                                config_sha256=events.manifest["task_config_sha256"])

    class Worker(FakeWorker):
        execution_id = "execution-1"

    executor = TaskExecutionManager(tasks, camera, models_root=tmp_path, worker_factory=Worker)
    try:
        executor.deploy("task-1", "bundle-1", operation_id="operation-1", deployment_revision=1)
        events.service._current = executor.event_context
        assert executor.event_context("task-1") is not None
        result, frame = observation(events.manifest)
        # Replacement can have an equal-looking session/epoch token. Source
        # object ownership, not token coincidence, fences the retained packet.
        camera.source = type(camera.source)()
        assert events.service.process_observation(events.manifest, result, frame,
            deployment_revision=1, operation_id="operation-1") == []
        assert executor.event_context("task-1") is None
        assert count(events.db, TaskEventRecord) == count(events.db, OutboxEvent) == 0
    finally:
        assert executor.close()


def test_invalid_frozen_config_cannot_reduce_confirmation_rule(events):
    manifest = deepcopy(events.manifest)
    manifest["config"]["rules"]["confidence_threshold"] = 0.0
    result, frame = observation(manifest)
    with pytest.raises(ValueError):
        events.service.process_observation(manifest, result, frame, deployment_revision=1, operation_id="operation-1")
    assert count(events.db, TaskEventRecord) == 0


def test_missing_outbox_never_counts_as_recovered_success(events, monkeypatch):
    event = process(events)[0]
    with events.db.get_session() as session:
        outbox = session.scalar(select(OutboxEvent))
        session.delete(outbox)
        session.commit()
    payload = event["payload"]
    with pytest.raises(RuntimeError, match="atomicity"):
        events.service._persist(payload, None, "not-encoded-again")
    assert count(events.db, AlertRecord) == 0


def test_storage_quota_is_visible_without_claiming_video_or_alert(events, monkeypatch):
    monkeypatch.setattr(events.service, "_max_evidence_bytes", 1)
    event = process(events)[0]
    assert event["evidence_status"] == "failed"
    assert event["payload"]["replay"]["reason"] == "KEYFRAME_STORAGE_UNAVAILABLE"
    assert event["alert_id"] is None


def test_review_does_not_approve_a_release_or_relabel_validation_as_production(events):
    event = process(events)[0]
    reviewed = events.service.review("task-1", event["event_id"], expected_revision=1,
                                     status="resolved", notes="reviewed only", actor="operator")
    assert reviewed["mode"] == "validation" and reviewed["payload"]["production_allowed"] is False
    assert count(events.db, AlertRecord) == 0


def test_backwards_capture_clock_never_creates_a_new_event(tmp_path):
    _, manifest, _ = manifest_for(tmp_path)
    engine = TaskEventEngine()
    result, frame = observation(manifest, number=1, timestamp=10.0)
    assert engine.observe(manifest, result, frame, deployment_revision=1, operation_id="operation-1")
    result, frame = observation(manifest, number=2, timestamp=9.0)
    with pytest.raises(TaskEventValidationError, match="capture time"):
        engine.observe(manifest, result, frame, deployment_revision=1, operation_id="operation-1")


@pytest.mark.parametrize("failure", [False, RuntimeError("fake event sink failure")])
def test_manager_event_sink_failure_cannot_stop_peer_source_or_task(tmp_path, failure):
    _, manifest, _ = manifest_for(tmp_path)
    tasks, camera = FakeTaskService(), FakeCameraManager()
    tasks.manifests = {"task-1": manifest}
    tasks.tasks["task-1"].update(template_id="object_detection.v1", config_sha256=manifest["task_config_sha256"])
    sink = SimpleNamespace(offer=Mock(return_value=False))
    if failure:
        sink.offer.side_effect = failure
    class Worker(FakeWorker):
        execution_id = "execution-1"
        def infer(self, frame):
            result, _ = observation(self.manifest, number=frame.frame_number, timestamp=frame.timestamp)
            result["source"] = frame_provenance(frame)
            return result
    executor = TaskExecutionManager(tasks, camera, models_root=tmp_path, worker_factory=Worker, event_sink=sink)
    try:
        executor.deploy("task-1", "bundle-1", operation_id="operation-1", deployment_revision=1)
        source_token = camera.source.source_token
        camera.source.hub.publish(synthetic_frame())
        wait_for(lambda: len(executor.results("task-1")) == 1)
        assert executor.status("task-1")["event_error"] == ("EVENT_QUEUE_FAILED" if failure else "EVENT_QUEUE_UNAVAILABLE")
        assert executor.status("task-1")["state"] == "running"
        assert camera.source.source_token == source_token
        assert executor.event_context("task-1")["execution_id"] == "execution-1"
    finally:
        assert executor.close()


def test_bounded_queue_rejects_without_running_encoder_under_source_callback(events, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    encode = events.service._encoder
    def blocked_encoder(frame):
        entered.set()
        assert release.wait(3)
        return encode(frame)
    monkeypatch.setattr(events.service, "_encoder", blocked_encoder)
    events.service._queue = __import__("queue").Queue(maxsize=1)
    events.service.start()
    result, frame = observation(events.manifest)
    try:
        assert events.service.offer(events.manifest, result, frame, deployment_revision=1, operation_id="operation-1")
        assert entered.wait(2)
        assert events.service.offer(events.manifest, result, frame, deployment_revision=1, operation_id="operation-1")
        assert events.service.offer(events.manifest, result, frame, deployment_revision=1, operation_id="operation-1") is False
    finally:
        release.set()
        assert events.service.close()


@pytest.mark.parametrize("missing", ["schema_version", "applied", "applied_stage", "production_allowed", "binding"])
def test_partial_production_boolean_never_authorizes_an_alert(events, missing):
    result, frame = observation(events.manifest)
    binding = {key: events.manifest[key] for key in (
        "task_id", "camera_id", "task_revision", "task_config_sha256", "bundle_id", "manifest_sha256", "template_id")}
    binding.update(deployment_revision=1, deployment_operation_id="operation-1", execution_id="execution-1",
                   models=result["models"], source_session_id="session", source_epoch=1)
    grant = {"schema_version": "argus.task-production-event-authorization.v1", "authorized": True,
             "production_allowed": True, "applied": True, "applied_stage": "production", "binding": binding,
             "release_id": "not-an-actual-approved-release"}
    grant.pop(missing)
    events.release.authorize_event = lambda *args, **kwargs: grant
    event = process(events)[0]
    assert event["mode"] == "validation" and event["alert_id"] is None
    assert count(events.db, AlertRecord) == 0
    with events.db.get_session() as session:
        assert session.scalar(select(func.count()).select_from(OutboxEvent).where(OutboxEvent.topic == "alerts")) == 0


@pytest.mark.parametrize("field", ["artifact_sha256", "source_epoch", "execution_id"])
def test_wrong_production_binding_cannot_authorize_an_alert(events, field):
    result, _ = observation(events.manifest)
    binding = {key: events.manifest[key] for key in (
        "task_id", "camera_id", "task_revision", "task_config_sha256", "bundle_id", "manifest_sha256", "template_id")}
    binding.update(deployment_revision=1, deployment_operation_id="operation-1", execution_id="execution-1",
                   models=deepcopy(result["models"]), source_session_id="session", source_epoch=1)
    if field == "artifact_sha256":
        binding["models"]["detector"][field] = "f" * 64
    else:
        binding[field] = 2 if field == "source_epoch" else "other-execution"
    events.release.authorize_event = lambda *args, **kwargs: {
        "schema_version": "argus.task-production-event-authorization.v1", "authorized": True,
        "production_allowed": True, "applied": True, "applied_stage": "production", "binding": binding,
        "release_id": "not-an-actual-approved-release"}
    assert process(events)[0]["mode"] == "validation"
    assert count(events.db, AlertRecord) == 0


def test_authorization_failure_is_retained_not_assumed_success(events):
    def unavailable(*args, **kwargs):
        raise OSError("simulated release read unavailable")
    events.release.authorize_event = unavailable
    with pytest.raises(RuntimeError, match="retry exhausted"):
        process(events)
    assert events.service.snapshot()["pending"] == 1
    assert count(events.db, TaskEventRecord) == count(events.db, AlertRecord) == 0


def test_background_outbox_delivery_does_not_need_any_legacy_alert(events, monkeypatch):
    sent = []
    monkeypatch.setattr(events.service, "_broadcaster", lambda topic, payload, **metadata: sent.append((topic, payload, metadata)))
    event = process(events)[0]
    assert count(events.db, AlertRecord) == 0
    events.service.start()
    wait_for(lambda: len(sent) == 1)
    assert sent[0][0] == "task_events" and sent[0][1]["event_id"] == event["event_id"]
    assert sent[0][2]["event_id"] == event["event_id"] + ":1"
    events.service.review("task-1", event["event_id"], expected_revision=1,
                           status="acknowledged", notes="reviewed", actor="human")
    wait_for(lambda: len(sent) == 2)
    assert sent[1][2]["event_type"] == "task_event.updated"
    assert sent[1][1]["revision"] == 2
    assert events.service.close()
    with events.db.get_session() as session:
        assert {row.status for row in session.scalars(select(OutboxEvent))} == {"published"}


def test_rejected_outbox_head_is_retained_and_retried_before_next_revision(events, monkeypatch):
    event = process(events)[0]
    events.service.review("task-1", event["event_id"], expected_revision=1,
                           status="acknowledged", notes=None, actor="human")
    sent = []
    def reject(topic, payload, **metadata):
        sent.append(metadata["event_id"])
        return False
    monkeypatch.setattr(events.service, "_broadcaster", reject)
    assert events.service.drain_outbox() == 0
    assert sent == [event["event_id"] + ":1"]
    with events.db.get_session() as session:
        rows = list(session.scalars(select(OutboxEvent).order_by(OutboxEvent.id)))
        assert [row.status for row in rows] == ["pending", "pending"]
        assert [row.attempts for row in rows] == [1, 0]
    monkeypatch.setattr(events.service, "_broadcaster", lambda topic, payload, **metadata: sent.append(metadata["event_id"]))
    assert events.service.drain_outbox() == 2
    assert sent == [event["event_id"] + ":1", event["event_id"] + ":1", event["event_id"] + ":2"]


@pytest.mark.parametrize("value", [True, 1.0])
def test_equal_looking_task_revision_cannot_impersonate_integer_identity(tmp_path, value):
    _, manifest, _ = manifest_for(tmp_path)
    result, frame = observation(manifest)
    result["task_revision"] = value
    with pytest.raises(TaskEventValidationError, match="frozen execution bundle"):
        validate_observation(manifest, result, frame, deployment_revision=1, operation_id="operation-1")


@pytest.mark.parametrize("field", ["source_epoch", "frame_index", "source_frame_index", "pts", "time_base_num", "time_base_den"])
@pytest.mark.parametrize("value", [True, 1.0])
def test_equal_looking_source_integers_are_not_the_same_original_frame(tmp_path, field, value):
    _, manifest, _ = manifest_for(tmp_path)
    result, frame = observation(manifest)
    frame.identity = frame.identity.with_updates(source_frame_index=1, pts=1, time_base_num=1, time_base_den=1)
    result["source"] = frame_provenance(frame)
    result["source"][field] = value
    with pytest.raises(TaskEventValidationError, match="source/frame identity"):
        validate_observation(manifest, result, frame, deployment_revision=1, operation_id="operation-1")


@pytest.mark.parametrize("field", ["source_frame_index", "pts", "time_base_num", "time_base_den"])
@pytest.mark.parametrize("value", [True, 1.0])
def test_even_matching_source_metadata_requires_real_integer_types(tmp_path, field, value):
    _, manifest, _ = manifest_for(tmp_path)
    result, frame = observation(manifest)
    frame.identity = frame.identity.with_updates(**{field: value})
    result["source"] = frame_provenance(frame)
    with pytest.raises(TaskEventValidationError, match="exact integer"):
        validate_observation(manifest, result, frame, deployment_revision=1, operation_id="operation-1")


@pytest.mark.parametrize("resolution", [(4.0, 4), (4, 4.0), [4, 4]])
def test_equal_looking_frame_dimensions_require_the_exact_resolution_contract(tmp_path, resolution):
    _, manifest, _ = manifest_for(tmp_path)
    result, frame = observation(manifest)
    frame.resolution = resolution
    with pytest.raises(TaskEventValidationError, match="frame dimensions"):
        validate_observation(manifest, result, frame, deployment_revision=1, operation_id="operation-1")


def test_unknown_pts_and_source_frame_number_remain_unknown(tmp_path):
    _, manifest, _ = manifest_for(tmp_path)
    result, frame = observation(manifest)
    observation_value = validate_observation(manifest, result, frame, deployment_revision=1, operation_id="operation-1")
    assert {observation_value["source"][key] for key in ("source_frame_index", "pts", "time_base_num", "time_base_den")} == {None}


@pytest.mark.parametrize("field,value", [("source_epoch", True), ("deployment_revision", 1.0), ("task_revision", True)])
def test_equal_looking_production_binding_is_not_production_authority(events, field, value):
    result, _ = observation(events.manifest)
    binding = {key: events.manifest[key] for key in (
        "task_id", "camera_id", "task_revision", "task_config_sha256", "bundle_id", "manifest_sha256", "template_id")}
    binding.update(deployment_revision=1, deployment_operation_id="operation-1", execution_id="execution-1",
                   models=result["models"], source_session_id="session", source_epoch=1)
    binding[field] = value
    events.release.authorize_event = lambda *args, **kwargs: {
        "schema_version": "argus.task-production-event-authorization.v1", "authorized": True,
        "production_allowed": True, "applied": True, "applied_stage": "production", "binding": binding,
        "release_id": "not-an-actual-approved-release"}
    assert process(events)[0]["mode"] == "validation"
    assert count(events.db, AlertRecord) == 0


@pytest.mark.parametrize("field,value", [("source_epoch", True), ("deployment_revision", 1.0)])
def test_runtime_identity_comparison_is_also_type_exact(events, field, value):
    events.current[field] = value
    assert process(events) == []
    assert count(events.db, TaskEventRecord) == count(events.db, OutboxEvent) == 0


def test_first_runtime_frame_before_applied_receipt_does_not_consume_cooldown(events):
    with events.db.get_session() as session:
        deployment = session.get(TaskDeploymentRecord, "task-1")
        deployment.applied_bundle_id = None
        deployment.applied_deployment_revision = None
        session.get(TaskDeploymentOperationRecord, "operation-1").status = "running"
        session.commit()
    assert process(events, number=1, timestamp=10.0) == []
    assert count(events.db, TaskEventRecord) == 0
    with events.db.get_session() as session:
        deployment = session.get(TaskDeploymentRecord, "task-1")
        deployment.applied_bundle_id = "bundle-1"
        deployment.applied_deployment_revision = 1
        session.get(TaskDeploymentOperationRecord, "operation-1").status = "applied"
        session.commit()
    created = process(events, number=2, timestamp=10.2)
    assert len(created) == 1
    assert created[0]["payload"]["source"]["frame_index"] == 2
    assert created[0]["payload"]["first_source"]["frame_index"] == 2


def test_receipt_lag_never_replays_an_event_after_a_new_source_generation(events):
    with events.db.get_session() as session:
        session.get(TaskDeploymentOperationRecord, "operation-1").status = "running"
        session.commit()
    assert process(events) == []
    with events.db.get_session() as session:
        session.get(TaskDeploymentOperationRecord, "operation-1").status = "applied"
        session.commit()
    events.current["source_epoch"] = 2
    assert process(events, number=2, timestamp=10.2) == []
    created = process(events, number=3, timestamp=10.4, epoch=2)
    assert len(created) == 1 and created[0]["source_epoch"] == 2
    assert created[0]["payload"]["first_source"]["frame_index"] == 3


@pytest.mark.parametrize("events", [0.5], indirect=True)
def test_queue_overflow_cannot_bridge_confirmation_duration_even_below_max_gap(events):
    import queue
    assert process(events, number=1, timestamp=10.0) == []
    events.service._queue = queue.Queue(maxsize=1)
    events.service.start()
    try:
        with events.service._process_lock:
            result, frame = observation(events.manifest, number=2, timestamp=10.2)
            assert events.service.offer(events.manifest, result, frame, deployment_revision=1, operation_id="operation-1")
            wait_for(lambda: events.service._queue.empty())
            result, frame = observation(events.manifest, number=3, timestamp=10.4)
            assert events.service.offer(events.manifest, result, frame, deployment_revision=1, operation_id="operation-1")
            result, frame = observation(events.manifest, number=4, timestamp=10.45)
            assert events.service.offer(events.manifest, result, frame, deployment_revision=1, operation_id="operation-1") is False
        wait_for(lambda: events.service._queue.unfinished_tasks == 0)
        assert process(events, number=5, timestamp=10.6) == []
        assert process(events, number=6, timestamp=10.9) == []
        event = process(events, number=7, timestamp=11.2)[0]
        assert event["payload"]["first_source"]["frame_index"] == 5
        assert event["payload"]["duration_seconds"] == pytest.approx(0.6)
    finally:
        assert events.service.close()


@pytest.mark.parametrize("events", [0.5], indirect=True)
def test_concurrent_gap_markers_never_reuse_a_packet_continuity(events, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    assert process(events, number=1, timestamp=10.0) == []
    original_token = events.service._continuity["task-1"]
    tokens = []
    renew = events.service._renew_continuity_locked

    def record(task_id):
        token = renew(task_id)
        tokens.append(token)
        return token

    monkeypatch.setattr(events.service, "_renew_continuity_locked", record)
    barrier = threading.Barrier(8)

    def invalidate():
        barrier.wait(timeout=3)
        for _ in range(20):
            events.service._mark_gap("task-1")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(invalidate) for _ in range(8)]
        for future in futures:
            future.result(timeout=5)
    assert len(tokens) == len({id(token) for token in tokens}) == 160
    assert events.service._continuity["task-1"] is tokens[-1]
    # A packet accepted before either a worker failure or an offer refusal
    # cannot regain admission after concurrent invalidations.
    result, frame = observation(events.manifest, number=2, timestamp=10.6)
    for stale_token in (original_token, tokens[0], tokens[-2]):
        assert events.service.process_observation(events.manifest, result, frame,
            deployment_revision=1, operation_id="operation-1", _continuity=stale_token) == []
    assert count(events.db, TaskEventRecord) == 0
    assert process(events, number=3, timestamp=10.8) == []
    assert process(events, number=4, timestamp=11.4)[0]["payload"]["first_source"]["frame_index"] == 3


@pytest.mark.parametrize("events", [0.5], indirect=True)
def test_inactive_cleanup_rejects_still_queued_tokens_after_same_task_returns(events):
    assert process(events, number=1, timestamp=10.0) == []
    events.service.start()
    try:
        with events.service._process_lock:
            result, frame = observation(events.manifest, number=2, timestamp=10.6)
            assert events.service.offer(events.manifest, result, frame,
                deployment_revision=1, operation_id="operation-1")
            wait_for(lambda: events.service._queue.empty())
            events.service._current = lambda task_id: None
            events.service._prune_continuity()
            assert "task-1" not in events.service._continuity
            assert "task-1" not in events.service._processed_continuity
            events.service._current = lambda task_id: events.current
        wait_for(lambda: events.service._queue.unfinished_tasks == 0)
        assert count(events.db, TaskEventRecord) == 0
        assert process(events, number=3, timestamp=10.8) == []
        assert process(events, number=4, timestamp=11.4)[0]["payload"]["first_source"]["frame_index"] == 3
    finally:
        assert events.service.close()


def test_failed_offers_keep_continuity_maps_bounded_without_a_worker(events):
    for index in range(300):
        manifest = {**events.manifest, "task_id": f"inactive-{index}"}
        result, frame = observation(manifest)
        assert not events.service.offer(manifest, result, frame,
            deployment_revision=1, operation_id="operation-1")
    assert len(events.service._continuity) == 128
    assert len(events.service._processed_continuity) <= 128
    assert "inactive-0" not in events.service._continuity
    events.service._prune_continuity()
    # The fixture's fake runtime treats every id as active; now let the runtime
    # report stopped/archived tasks exactly as the real manager does.
    events.service._current = lambda task_id: None
    events.service._prune_continuity()
    assert not events.service._continuity and not events.service._processed_continuity


class _EventThreadDouble:
    """Lifecycle-only owner; never starts a Python/OS thread or handles media."""

    def __init__(self, mode="running"):
        self.mode, self.ident, self.alive = mode, None, False
        self._started = threading.Event()
        self.starts = self.joins = 0
        self.unknown = False

    def start(self):
        self.starts += 1
        if self.mode == "unknown":
            self.unknown = True
            raise RuntimeError("simulated unconfirmed start")
        self.ident, self.alive = 12345, True
        self._started.set()
        if self.mode == "live_then_raise":
            raise RuntimeError("simulated lost start reply")

    def is_alive(self):
        if self.unknown:
            raise RuntimeError("simulated unavailable thread state")
        return self.alive

    def join(self, timeout=None):
        self.joins += 1
        self.alive = False


@pytest.mark.parametrize("phase", ["construction", "start"])
def test_event_worker_start_refusal_keeps_offer_closed_and_close_safe(events, monkeypatch, phase):
    unstarted = threading.Thread(target=lambda: None)
    unstarted.start = Mock(side_effect=RuntimeError("simulated thread resource exhaustion"))

    def factory(**kwargs):
        if phase == "construction":
            raise RuntimeError("simulated thread construction failure")
        return unstarted

    monkeypatch.setattr("argus.storage.task_events.threading.Thread", factory)
    assert events.service.start() is False
    assert events.service._thread is None
    assert events.service.snapshot()["last_error"] == "EVENT_WORKER_START_FAILED"
    result, frame = observation(events.manifest)
    assert not events.service.offer(events.manifest, result, frame,
        deployment_revision=1, operation_id="operation-1")
    assert events.service.close() is True
    assert events.service.snapshot()["state"] == "stopped"
    assert count(events.db, TaskEventRecord) == count(events.db, AlertRecord) == 0


def test_event_worker_start_refusal_can_retry_without_creating_two_owners(events, monkeypatch):
    unstarted = threading.Thread(target=lambda: None)
    unstarted.start = Mock(side_effect=RuntimeError("simulated thread resource exhaustion"))
    running = _EventThreadDouble()
    factory = Mock(side_effect=[unstarted, running])
    monkeypatch.setattr("argus.storage.task_events.threading.Thread", factory)
    assert events.service.start() is False
    assert events.service.start() is True
    assert events.service.start() is True
    assert factory.call_count == 2 and running.starts == 1
    assert events.service.close() is True
    assert running.joins == 1


@pytest.mark.parametrize("mode", ["live_then_raise", "unknown"])
def test_start_error_retains_live_or_unknown_thread_handle_and_never_fakes_cleanup(events, monkeypatch, mode):
    owner = _EventThreadDouble(mode)
    factory = Mock(return_value=owner)
    monkeypatch.setattr("argus.storage.task_events.threading.Thread", factory)
    assert events.service.start() is False
    assert events.service._thread is owner
    assert events.service.start() is (mode == "live_then_raise")
    assert events.service._thread is owner and factory.call_count == owner.starts == 1
    if mode == "unknown":
        result, frame = observation(events.manifest)
        assert not events.service.offer(events.manifest, result, frame,
            deployment_revision=1, operation_id="operation-1")
        assert events.service.close() is False
        assert events.service._thread is owner and owner.joins == 0
        assert events.service.snapshot()["last_error"] == "EVENT_WORKER_STATE_UNKNOWN"
        assert events.service.snapshot()["state"] == "cleanup_pending"
    else:
        assert events.service.close() is True
        assert owner.joins == 1
        assert events.service.snapshot()["state"] == "stopped"


def test_close_timeout_remains_cleanup_pending_until_thread_exit_is_confirmed(events, monkeypatch):
    owner = _EventThreadDouble()
    monkeypatch.setattr("argus.storage.task_events.threading.Thread", Mock(return_value=owner))
    monkeypatch.setattr(owner, "join", Mock())
    assert events.service.start() is True
    assert events.service.close(timeout=0.0) is False
    assert events.service.snapshot()["state"] == "cleanup_pending"
    assert events.service._thread is owner
    owner.alive = False
    assert events.service.close(timeout=0.0) is True
    assert events.service.snapshot()["state"] == "stopped"


@pytest.mark.parametrize("unresolved", ["queue", "pending_write"])
def test_closed_without_worker_does_not_hide_unresolved_event_work(events, unresolved):
    if unresolved == "queue":
        events.service._queue.put_nowait(object())
    else:
        events.service._pending_write = ([{"event_id": "synthetic-unresolved"}], None, None)
    assert events.service.close() is False
    assert events.service.snapshot()["state"] == "cleanup_pending"
    if unresolved == "queue":
        events.service._queue.get_nowait()
        events.service._queue.task_done()
    else:
        events.service._pending_write = None
    assert events.service.close() is True
    assert events.service.snapshot()["state"] == "stopped"


def test_offer_waiting_on_admission_cannot_enqueue_after_confirmed_close(events, monkeypatch):
    thread_class = threading.Thread
    owner = _EventThreadDouble()
    monkeypatch.setattr("argus.storage.task_events.threading.Thread", Mock(return_value=owner))
    assert events.service.start()
    checked = threading.Event()
    original_state = events.service._thread_state
    outcome = []
    result, frame = observation(events.manifest)
    # Let the controlling test thread take the close-side admission barrier
    # reentrantly while a real offer is waiting outside it.
    events.service._continuity_lock = threading.RLock()

    def state(thread):
        value = original_state(thread)
        if threading.current_thread() is admission:
            checked.set()
        return value

    monkeypatch.setattr(events.service, "_thread_state", state)
    admission = thread_class(target=lambda: outcome.append(events.service.offer(
        events.manifest, result, frame, deployment_revision=1, operation_id="operation-1")))
    with events.service._continuity_lock:
        admission.start()
        assert checked.wait(2)
        assert events.service.close()
    admission.join(2)
    assert not admission.is_alive()
    assert outcome == [False]
    assert events.service.snapshot()["state"] == "stopped"
    assert events.service._queue.empty()


def collection_binding(events):
    return {key: events.manifest[key] for key in (
        "task_id", "camera_id", "template_id", "task_revision", "task_config_sha256")}


def test_collection_provider_exact_authority_and_caller_transaction_cursor(events, tmp_path):
    assert events.service.authority_for(events.db)
    assert not events.service.authority_for(object())
    assert events.service.collection_cursor() == 0
    with events.db.get_session() as session:
        session.execute(text("BEGIN IMMEDIATE"))
        context = events.service.collection_context(collection_binding(events), session=session)
        assert context == events.current
        assert events.service.collection_cursor(session=session) == 0
        session.rollback()
    other = Database(f"sqlite:///{tmp_path / 'other-collection.db'}")
    other.initialize()
    try:
        with other.get_session() as session, pytest.raises(TaskEventConflict, match="another database"):
            events.service.collection_cursor(session=session)
    finally:
        other.close()


def test_collection_cursor_uses_commit_order_and_does_not_consume_outbox(events):
    binding = collection_binding(events)
    start = events.service.collection_cursor()
    first = process(events)[0]
    first_cursor = events.service.collection_cursor()
    second = process(events, number=2, timestamp=10.5, box=[3.0, 3.0, 4.0, 4.0])[0]
    with events.db.get_session() as session:
        # Neither timestamps nor event hash lexical order determines the cursor.
        fixed = datetime(2026, 1, 1)
        for row in session.scalars(select(TaskEventRecord)):
            row.created_at = fixed
        before = [(row.id, row.status, row.attempts) for row in session.scalars(select(OutboxEvent))]
        session.commit()
    page = events.service.collection_events(binding, start, limit=1)
    assert page[0]["event_id"] == first["event_id"]
    assert page[0]["cursor"] == first_cursor
    next_page = events.service.collection_events(binding, page[0]["cursor"])
    assert [row["event_id"] for row in next_page] == [second["event_id"]]
    assert next_page[0]["cursor"] > page[0]["cursor"]
    with events.db.get_session() as session:
        assert [(row.id, row.status, row.attempts) for row in session.scalars(select(OutboxEvent))] == before


def test_collection_review_updates_do_not_change_evidence_identity_or_notes(events):
    binding = collection_binding(events)
    event = process(events)[0]
    receipt = events.service.collection_events(binding, 0)[0]
    first = events.service.collection_evidence(binding, event["event_id"], receipt["immutable_sha256"])
    reviewed = events.service.review("task-1", event["event_id"], expected_revision=1,
        status="acknowledged", notes="retained independent review", actor="reviewer")
    assert events.service.collection_events(binding, receipt["cursor"]) == []
    assert events.service.collection_events(binding, 0) == [receipt]
    again = events.service.collection_evidence(binding, event["event_id"], receipt["immutable_sha256"])
    assert again == first
    assert again["mime"] == "image/jpeg" and again["image_size"] == [4, 4]
    assert hashlib.sha256(again["bytes"]).hexdigest() == again["evidence_sha256"]
    assert sha({"payload": again["payload"], "evidence_sha256": again["evidence_sha256"]}) == receipt["immutable_sha256"]
    assert not {"status", "notes", "revision", "review_history"}.intersection(again["payload"])
    assert events.service.get("task-1", event["event_id"])["notes"] == reviewed["notes"]
    assert again["source"]["source_frame_index"] is None and again["source"]["pts"] is None


def test_collection_evidence_reads_exact_verified_bytes_once(events, monkeypatch):
    binding = collection_binding(events)
    event = process(events)[0]
    receipt = events.service.collection_events(binding, 0)[0]
    opened = []
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        opened.append((path, args, kwargs))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    result = events.service.collection_evidence(binding, event["event_id"], receipt["immutable_sha256"])
    assert len(opened) == 1 and opened[0][1] == ("rb",)
    assert hashlib.sha256(result["bytes"]).hexdigest() == result["evidence_sha256"]


@pytest.mark.parametrize("field,value", [
    ("task_id", "other-task"), ("camera_id", "other-camera"), ("template_id", "pose_estimation.v1"),
    ("task_revision", True), ("task_revision", 2), ("task_config_sha256", "b" * 64),
])
def test_collection_provider_rejects_other_or_malformed_task_binding(events, field, value):
    event = process(events)[0]
    binding = collection_binding(events)
    receipt = events.service.collection_events(binding, 0)[0]
    binding[field] = value
    with pytest.raises((TaskEventConflict, LookupError)):
        events.service.collection_evidence(binding, event["event_id"], receipt["immutable_sha256"])


@pytest.mark.parametrize("field,value", [
    ("execution_id", "another-execution"), ("bundle_id", "another-bundle"),
    ("deployment_revision", 2), ("deployment_operation_id", "another-operation"),
    ("source_session_id", "another-session"), ("source_epoch", 2),
])
def test_collection_context_stops_late_execution_and_source_reuse(events, field, value):
    binding = collection_binding(events)
    context = events.service.collection_context(binding)
    event = process(events)[0]
    receipt = events.service.collection_events(binding, 0, context=context)[0]
    events.current[field] = value
    assert not events.service.collection_context_current(binding, context)
    # Historical receipt reads do not fabricate current execution authority.
    assert events.service.collection_evidence(binding, event["event_id"], receipt["immutable_sha256"])["source"]["frame_index"] == 1
    wrong = {**context, field: value}
    with pytest.raises(TaskEventConflict):
        events.service.collection_evidence(binding, event["event_id"], receipt["immutable_sha256"], context=wrong)


@pytest.mark.parametrize("change", ["task_revision", "archived", "desired_state", "applied_revision"])
def test_collection_context_current_rechecks_durable_task_and_deployment(events, change):
    binding = collection_binding(events)
    context = events.service.collection_context(binding)
    with events.db.get_session() as session:
        if change == "task_revision":
            session.get(AnalysisTaskRecord, "task-1").current_revision = 2
        elif change == "archived":
            session.get(AnalysisTaskRecord, "task-1").archived = True
        elif change == "desired_state":
            session.get(TaskDeploymentRecord, "task-1").desired_state = "stopped"
        else:
            session.get(TaskDeploymentRecord, "task-1").applied_deployment_revision = 2
        session.commit()
    assert not events.service.collection_context_current(binding, context)


@pytest.mark.parametrize("change", ["missing", "replaced", "digest", "payload", "outbox"])
def test_collection_evidence_revalidates_exact_file_and_immutable_receipt(events, change):
    binding = collection_binding(events)
    event = process(events)[0]
    receipt = events.service.collection_events(binding, 0)[0]
    with events.db.get_session() as session:
        row = session.get(TaskEventRecord, event["event_id"])
        path = events.service._root / row.evidence_path
        if change == "digest":
            row.evidence_sha256 = "b" * 64
        elif change == "payload":
            payload = json.loads(row.payload_json)
            payload["object"]["confidence"] = 0.6
            row.payload_json = canonical(payload)
        elif change == "outbox":
            session.delete(session.scalar(select(OutboxEvent)))
        session.commit()
    if change == "missing":
        path.unlink()
    elif change == "replaced":
        path.write_bytes(b"changed synthetic keyframe")
    with pytest.raises((TaskEventConflict, TaskEventEvidenceUnavailable)):
        events.service.collection_evidence(binding, event["event_id"], receipt["immutable_sha256"])


@pytest.mark.parametrize("cursor,limit", [
    (True, 1), (-1, 1), (0.0, 1), (2**63, 1), (0, True), (0, 0), (0, 17),
])
def test_collection_provider_bounds_polling(events, cursor, limit):
    with pytest.raises(ValueError):
        events.service.collection_events(collection_binding(events), cursor, limit)


@pytest.mark.parametrize("changes", [
    {"pts": 123}, {"time_base_num": 1}, {"time_base_den": 90000},
    {"pts": 123, "time_base_num": 1}, {"pts": 123, "time_base_den": 90000},
])
def test_source_pts_requires_a_complete_time_base_before_event_persistence(events, changes):
    result, frame = observation(events.manifest)
    frame.identity = frame.identity.with_updates(**changes)
    result["source"] = frame_provenance(frame)
    with pytest.raises(TaskEventValidationError, match="complete positive time base"):
        events.service.process_observation(events.manifest, result, frame,
            deployment_revision=1, operation_id="operation-1")
    assert count(events.db, TaskEventRecord) == 0


@pytest.mark.parametrize("changes", [
    {"pts": None, "time_base_num": 1, "time_base_den": 90000},
    {"pts": -12, "time_base_num": 1, "time_base_den": 90000},
])
def test_complete_time_base_preserves_unknown_or_negative_original_pts(events, changes):
    result, frame = observation(events.manifest)
    frame.identity = frame.identity.with_updates(**changes)
    result["source"] = frame_provenance(frame)
    event = events.service.process_observation(events.manifest, result, frame,
        deployment_revision=1, operation_id="operation-1")[0]
    receipt = events.service.collection_events(collection_binding(events), 0)[0]
    evidence = events.service.collection_evidence(collection_binding(events), event["event_id"], receipt["immutable_sha256"])
    assert evidence["source"]["pts"] == changes["pts"]
    assert evidence["source"]["time_base_num"] == 1 and evidence["source"]["time_base_den"] == 90000


@pytest.mark.parametrize("source_name,field,value", [
    ("source", "source_frame_index", True), ("source", "source_frame_index", 1.0),
    ("source", "pts", True), ("source", "pts", 1.0), ("source", "pts", 123),
    ("source", "time_base_num", 1), ("source", "time_base_den", 90000),
    ("source", "capture_monotonic_ts", -1.0), ("source", "capture_wall_ts", True),
    ("source", "source_runtime", "not-exportable"), ("first_source", "source_epoch", 2),
    ("first_source", "frame_index", 2), ("first_source", "capture_monotonic_ts", 1000.0),
])
def test_collection_rejects_malformed_historical_source_before_evidence_io(events, monkeypatch, source_name, field, value):
    process(events)
    with events.db.get_session() as session:
        row = session.scalar(select(TaskEventRecord))
        payload = json.loads(row.payload_json)
        payload[source_name][field] = value
        row.payload_json = canonical(payload)
        session.commit()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("must reject metadata before image IO"))
    with pytest.raises(TaskEventConflict, match="identity is invalid"):
        events.service.collection_events(collection_binding(events), 0)


@pytest.mark.parametrize("change", ["model_id", "artifact_sha256", "model_role", "manifest_sha256"])
def test_collection_rejects_model_identity_drift_from_the_frozen_bundle(events, change):
    process(events)
    with events.db.get_session() as session:
        row = session.scalar(select(TaskEventRecord))
        payload = json.loads(row.payload_json)
        binding = payload["binding"]
        if change == "model_role":
            binding["models"] = {"unexpected": binding["models"]["detector"]}
        elif change == "manifest_sha256":
            binding["manifest_sha256"] = "b" * 64
        else:
            binding["models"]["detector"][change] = "another-model" if change == "model_id" else "b" * 64
        row.payload_json = canonical(payload)
        session.commit()
    with pytest.raises(TaskEventConflict, match="identity is invalid"):
        events.service.collection_events(collection_binding(events), 0)


def test_keyframe_path_cannot_alias_another_event_even_when_image_bytes_match(events, monkeypatch):
    binding = collection_binding(events)
    event = process(events)[0]
    receipt = events.service.collection_events(binding, 0)[0]
    data, _ = events.service.load_evidence("task-1", event["event_id"])
    (events.service._root / "another-event.jpg").write_bytes(data)
    with events.db.get_session() as session:
        row = session.get(TaskEventRecord, event["event_id"])
        row.evidence_path = "another-event.jpg"
        session.commit()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("cross-event path must fail before opening"))
    with pytest.raises(TaskEventEvidenceUnavailable, match="another event"):
        events.service.collection_evidence(binding, event["event_id"], receipt["immutable_sha256"])
    with pytest.raises(TaskEventEvidenceUnavailable, match="another event"):
        events.service.load_evidence("task-1", event["event_id"])


@pytest.mark.parametrize("events", [{"collection": {
    "enabled": True, "policy": "events", "interval_seconds": 0.1, "max_samples": 1,
}}], indirect=True)
def test_event_collection_workflow_preserves_exact_frame_origin_and_independent_human_review(events, tmp_path):
    from argus.capture.task_collector import png_image_size
    from argus.storage.task_workflows import TaskWorkflowService

    workflows = TaskWorkflowService(events.db, tmp_path / "event-dataset")
    workflows.set_event_provider(events.service)
    manager = Mock()
    manager.get_camera_source.side_effect = AssertionError("event collection must not read a camera")
    workflow = workflows.create("task-1", actor="engineer", idempotency_key="synthetic-event-flow")
    try:
        collecting = workflows.start_collection(workflow["workflow_id"], expected_revision=workflow["revision"],
            manager=manager, actor="engineer", collection_id="exact-event-collection",
            params={"max_samples": 1, "max_duration": 5.0})
        assert collecting["operation"]["start_cursor"] == 0
        assert collecting["operation"]["event_context"] == events.current
        result, frame = observation(events.manifest, number=7, timestamp=10.5)
        frame.identity = frame.identity.with_updates(
            source_frame_index=39, pts=12012, time_base_num=1, time_base_den=90000,
            frame_index_provenance="decoder_index", timestamp_provenance="decoder_pts",
        )
        result["source"] = frame_provenance(frame)
        event = events.service.process_observation(events.manifest, result, frame,
                                                   deployment_revision=1, operation_id="operation-1")[0]

        def finalized():
            current = workflows.get(workflow["workflow_id"])
            return (current["state"] == "review"
                    and current["collection_runtime"]["persistence_pending"] is False
                    and current["collection_runtime"]["cleanup_confirmed"] is True)

        wait_for(finalized)
        current = workflows.get(workflow["workflow_id"])
        assert len(current["manifest"]["samples"]) == 1
        sample = current["manifest"]["samples"][0]
        assert sample["identity"] == result["source"]
        assert sample["identity"]["frame_index"] == 7
        assert sample["identity"]["source_frame_index"] == 39
        assert sample["identity"]["pts"] == 12012
        assert (sample["identity"]["time_base_num"], sample["identity"]["time_base_den"]) == (1, 90000)
        assert sample["auto_annotation"] is None and current["manifest"]["reviews"] == {}
        image = workflows.read_sample_image(workflow["workflow_id"], sample["sample_id"])
        assert png_image_size(image) == (4, 4)
        assert hashlib.sha256(image).hexdigest() == sample["image_sha256"]
        origin = sample["event_origin"]
        assert origin["event_id"] == event["event_id"]
        assert origin["evidence_sha256"] == event["evidence_sha256"]
        assert origin["derivation"] == "verified_jpeg_to_png.v1"
        assert origin["payload"]["source"] == sample["identity"]
        assert sha({"payload": origin["payload"], "evidence_sha256": origin["evidence_sha256"]}) == origin["immutable_sha256"]

        reviewed = workflows.review(workflow["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
            expected_image_sha256=sample["image_sha256"], decision="approved", actor="human-reviewer",
            notes="keep independently approved sample", annotation={"schema": "argus.task-annotation.v1",
                "kind": "detection", "coordinate_space": "normalized",
                "objects": [{"label": "tool", "bbox_xyxy": [0.0, 0.0, 0.75, 0.75]}]})
        events.service.review("task-1", event["event_id"], expected_revision=1,
            status="false_positive", notes="separate event note", actor="event-reviewer")
        collector = workflows._collectors[workflow["workflow_id"]]
        workflows._finish_collection(workflow["workflow_id"], collector.collection_id,
                                     deepcopy(collector._finish_snapshot))
        final = workflows.get(workflow["workflow_id"])
        assert final["manifest"]["reviews"] == reviewed["manifest"]["reviews"]
        assert final["manifest"]["reviews"][sample["sample_id"]]["decision"] == "approved"
        assert final["manifest"]["reviews"][sample["sample_id"]]["notes"] == "keep independently approved sample"
        assert final["manifest"]["samples"] == current["manifest"]["samples"]
        manager.get_camera_source.assert_not_called()
    finally:
        assert workflows.close()


@pytest.mark.parametrize("events", [{"collection": {
    "enabled": True, "policy": "events", "interval_seconds": 0.1, "max_samples": 1,
}}], indirect=True)
def test_event_collection_workflow_two_rounds_deduplicate_frames_and_isolate_cameras(events, tmp_path):
    from argus.storage.task_workflows import TaskWorkflowService

    workflows = TaskWorkflowService(events.db, tmp_path / "two-round-dataset", event_provider=events.service)
    manager = Mock()
    manager.get_camera_source.side_effect = AssertionError("event collection must not read a camera")
    workflow = workflows.create("task-1", actor="engineer", idempotency_key="two-round-event-flow")
    workflow_id = workflow["workflow_id"]
    # Two real incidents can share one original frame. Commit the second one
    # later so the second collection must use its persisted source ledger.
    result, frame = observation(events.manifest, number=7, timestamp=10.5)
    frame.identity = frame.identity.with_updates(
        source_frame_index=39, pts=12012, time_base_num=1, time_base_den=90000,
        frame_index_provenance="decoder_index", timestamp_provenance="decoder_pts",
    )
    result["source"] = frame_provenance(frame)
    result["data"]["detections"].append({
        "label": "tool", "confidence": 0.8, "bbox_xyxy": [3.0, 3.0, 4.0, 4.0],
    })
    shared_frame_events = TaskEventEngine().observe(
        events.manifest, result, frame, deployment_revision=1, operation_id="operation-1")
    assert len(shared_frame_events) == 2
    encoded = events.service._encoder(frame.frame)

    def finalized(collection_id):
        current = workflows.get(workflow_id)
        runtime = current["collection_runtime"]
        return (current["state"] == "review" and runtime["collection_id"] == collection_id
                and runtime["persistence_pending"] is False and runtime["cleanup_confirmed"] is True)

    other_service = None
    try:
        workflows.start_collection(workflow_id, expected_revision=workflow["revision"], manager=manager,
            actor="engineer", collection_id="round-one", params={"max_samples": 1, "max_duration": 5.0})
        first_event = events.service._persist(shared_frame_events[0], encoded, None)
        wait_for(lambda: finalized("round-one"))
        first = workflows.get(workflow_id)
        sample = first["manifest"]["samples"][0]
        reviewed = workflows.review(workflow_id, sample["sample_id"], expected_revision=first["revision"],
            expected_image_sha256=sample["image_sha256"], decision="approved", actor="human-reviewer",
            notes="preserve approval and note across collection rounds", annotation={
                "schema": "argus.task-annotation.v1", "kind": "detection", "coordinate_space": "normalized",
                "objects": [{"label": "tool", "bbox_xyxy": [0.0, 0.0, 0.75, 0.75]}],
            })
        old_reviews = deepcopy(reviewed["manifest"]["reviews"])
        first_image = workflows.read_sample_image(workflow_id, sample["sample_id"])
        collecting = workflows.start_collection(workflow_id, expected_revision=reviewed["revision"], manager=manager,
            actor="engineer", collection_id="round-two", params={"max_samples": 1, "max_duration": 5.0})
        start_cursor = collecting["operation"]["start_cursor"]
        duplicate_event = events.service._persist(shared_frame_events[1], encoded, None)
        assert duplicate_event["event_id"] != first_event["event_id"]
        assert duplicate_event["payload"]["source"] == sample["identity"]

        # A complete second task/camera uses the same source counters and PTS;
        # its durable creation must not advance or contaminate this collector.
        other_stored = {key: deepcopy(value) for key, value in events.manifest.items()
                        if key != "manifest_sha256"}
        other_stored["models"]["detector"].pop("artifact_path")
        other_stored.update(task_id="task-other", camera_id="cam-other", bundle_id="bundle-other")
        other_manifest = {**deepcopy(other_stored), "manifest_sha256": sha(other_stored)}
        other_manifest["models"]["detector"]["artifact_path"] = str(tmp_path / "not-executed.onnx")
        with events.db.get_session() as session:
            session.add_all([
                AnalysisTaskRecord(task_id="task-other", camera_id="cam-other", template_id="object_detection.v1",
                    authority="managed", name="Other camera", current_revision=1, archived=False, created_by="human"),
                AnalysisTaskRevisionRecord(task_id="task-other", revision=1,
                    config_json=canonical(other_stored["config"]), config_sha256=other_stored["task_config_sha256"]),
                TaskExecutionBundleRecord(bundle_id="bundle-other", task_id="task-other", task_revision=1,
                    template_id="object_detection.v1", manifest_json=canonical(other_stored),
                    manifest_sha256=sha(other_stored), created_by="human"),
                TaskDeploymentRecord(task_id="task-other", authority="managed", projection_status="managed",
                    runtime_status="running", desired_state="running", deployment_revision=1,
                    desired_bundle_id="bundle-other", applied_bundle_id="bundle-other", applied_deployment_revision=1,
                    current_operation_id="operation-other"),
                TaskDeploymentOperationRecord(operation_id="operation-other", task_id="task-other", idempotency_key="other",
                    request_sha256="c" * 64, action="deploy", bundle_id="bundle-other", deployment_revision=1,
                    status="applied", requested_by="human", created_at=datetime.now(timezone.utc)),
            ])
            session.commit()
        other_context = {**events.current, "task_id": "task-other", "bundle_id": "bundle-other",
                         "manifest_sha256": other_manifest["manifest_sha256"],
                         "deployment_operation_id": "operation-other"}
        other_service = TaskEventService(events.db, None, evidence_root=tmp_path / "other-evidence",
            release_service=events.release, current_runtime=lambda task_id: other_context)
        other_result, other_frame = observation(other_manifest, number=7, timestamp=10.5)
        other_frame.camera_id = "cam-other"
        other_frame.identity = frame.identity.with_updates(camera_id="cam-other")
        other_result["source"] = frame_provenance(other_frame)
        foreign_event = other_service.process_observation(other_manifest, other_result, other_frame,
            deployment_revision=1, operation_id="operation-other")[0]
        new_event = process(events, number=8, timestamp=11.0)[0]
        page = events.service.collection_events(collection_binding(events), start_cursor, context=events.current)
        assert [entry["event_id"] for entry in page] == [duplicate_event["event_id"], new_event["event_id"]]
        assert foreign_event["camera_id"] == "cam-other"
        wait_for(lambda: finalized("round-two"))
        final = workflows.get(workflow_id)
        assert len(final["manifest"]["samples"]) == 2
        assert final["manifest"]["samples"][0] == sample
        assert final["manifest"]["reviews"] == old_reviews
        second = final["manifest"]["samples"][1]
        assert second["event_origin"]["event_id"] == new_event["event_id"]
        assert second["identity"] == new_event["payload"]["source"]
        assert second["identity"]["frame_index"] == 8 and second["identity"]["camera_id"] == "cam"
        assert final["collection_runtime"]["duplicate_frame_events"] == 1
        assert final["collection_runtime"]["existing_frame_count"] == 1
        assert workflows.read_sample_image(workflow_id, sample["sample_id"]) == first_image
        assert len(list((tmp_path / "two-round-dataset" / "task-1").rglob("*.png"))) == 2
        assert count(events.db, AlertRecord) == 0
        manager.get_camera_source.assert_not_called()
    finally:
        assert workflows.close()
        if other_service is not None:
            assert other_service.close()
