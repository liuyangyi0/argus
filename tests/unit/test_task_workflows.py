"""Transactional task data workflows; fake image bytes, no training or SDKs."""

import copy
import hashlib
import json
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.core.frame_identity import build_frame_identity
from argus.storage.analysis_tasks import AnalysisTaskService
from argus.storage.database import Database
from argus.storage.models import TaskDatasetRecord, TaskWorkflowRecord
from argus.storage.task_workflows import (
    TaskWorkflowBlocked, TaskWorkflowConflict, TaskWorkflowService, _canonical, _sha,
)


def annotation(label="person"):
    return {"schema": "argus.task-annotation.v1", "kind": "detection", "coordinate_space": "normalized",
            "objects": [{"label": label, "bbox_xyxy": [0.1, 0.2, 0.5, 0.8]}]}


def png_image(index=0):
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + bytes([index]) * 9) * 2)) + chunk(b"IEND", b""))


@pytest.fixture
def environment(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'task.db'}")
    db.initialize()
    tasks = AnalysisTaskService(db)
    task = tasks.create_task("cam", "object_detection.v1", "people", {
        "rules": {"labels": ["person"]}, "collection": {"interval_seconds": 0.1},
    }, requested_by="operator", idempotency_key="create-test")
    service = TaskWorkflowService(db, tmp_path / "images", encoder=lambda _: png_image())
    workflow = service.create(task["task_id"], actor="operator")
    yield SimpleNamespace(db=db, tasks=tasks, task=task, service=service, workflow=workflow)
    service.close()
    db.close()


def seed_samples(env, *, count=2):
    workflow_id = env.workflow["workflow_id"]
    with env.db.get_session() as session:
        record, manifest = env.service._record(session, workflow_id)
        for index in range(count):
            data = png_image(index)
            relative = f"{env.task['task_id']}/collection-{index}/sample-{index}.png"
            path = env.service.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            identity = build_frame_identity(camera_id="cam", source_declared="mock", source_runtime="mock",
                stream_id=None, frame_index=index, source_session_id=f"session-{index}", source_epoch=1,
                source_frame_index=200 + index, pts=10000 + index, time_base_num=1, time_base_den=90000).to_dict()
            manifest["samples"].append({"sample_id": f"sample-{index}", "collection_id": f"collection-{index}",
                "relative_path": relative, "image_sha256": hashlib.sha256(data).hexdigest(),
                "byte_size": len(data), "identity": identity, "binding": manifest["binding"],
                "auto_annotation": None, "image_size": [3, 2]})
        env.service._save(session, record, manifest, "review")
        session.commit()
    return env.service.get(workflow_id)


def reviewed(env):
    current = seed_samples(env)
    for sample in current["manifest"]["samples"]:
        current = env.service.review(current["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
            expected_image_sha256=sample["image_sha256"], decision="approved", actor="reviewer",
            notes="keep this human note", annotation=annotation())
    return current


def export_ready(env):
    current = reviewed(env)
    env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")
    current = env.service.get(current["workflow_id"])
    current = env.service.propose_training(current["workflow_id"], expected_revision=current["revision"],
        recipe_id="external_detection.v1", actor="operator")
    return env.service.approve_training(current["workflow_id"], expected_revision=current["revision"], actor="approver")


def test_image_endpoint_returns_the_single_verified_read(environment, monkeypatch):
    env = environment
    current = seed_samples(env)
    original = Path.open
    reads = []

    def opened(path, *args, **kwargs):
        if path.suffix == ".png":
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    assert env.service.read_sample_image(current["workflow_id"], "sample-0") == png_image()
    assert len(reads) == 1
    (env.service.root / current["manifest"]["samples"][0]["relative_path"]).write_bytes(b"tampered")
    with pytest.raises(TaskWorkflowConflict, match="content changed"):
        env.service.read_sample_image(current["workflow_id"], "sample-0")


@pytest.mark.parametrize("change", ["unknown", "negative", "reverse", "bool", "nan", "label", "confidence", "space"])
def test_invalid_annotations_cannot_be_human_approved(environment, change):
    env = environment
    current = seed_samples(env)
    value = annotation()
    if change == "unknown":
        value["executable"] = "payload"
    elif change == "negative":
        value["objects"][0]["bbox_xyxy"][0] = -0.1
    elif change == "reverse":
        value["objects"][0]["bbox_xyxy"] = [0.8, 0.2, 0.1, 0.9]
    elif change == "bool":
        value["objects"][0]["bbox_xyxy"][0] = True
    elif change == "nan":
        value["objects"][0]["bbox_xyxy"][0] = float("nan")
    elif change == "label":
        value["objects"][0]["label"] = "not-a-task-class"
    elif change == "confidence":
        value["objects"][0]["confidence"] = 0.9
    else:
        value["coordinate_space"] = "pixels"
    sample = current["manifest"]["samples"][0]
    with pytest.raises(ValueError):
        env.service.review(current["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
            expected_image_sha256=sample["image_sha256"], decision="approved", actor="reviewer", annotation=value)
    assert env.service.get(current["workflow_id"])["revision"] == current["revision"]


def test_pose_requires_complete_unique_coco17_keypoints():
    value = {"schema": "argus.task-annotation.v1", "kind": "pose", "coordinate_space": "normalized",
             "objects": [{"label": "person", "bbox_xyxy": [0, 0, 1, 1], "keypoint_schema": "coco17.v1",
                          "keypoints": [{"index": index, "x": 0.5, "y": 0.6, "visibility": 2} for index in range(17)]}]}
    assert TaskWorkflowService._annotation(value, "pose_estimation.v1") == value
    value["objects"][0]["keypoints"][-1]["index"] = 0
    with pytest.raises(ValueError, match="unique"):
        TaskWorkflowService._annotation(value, "pose_estimation.v1")


def test_report_update_preserves_notes_but_invalidates_auto_review(environment):
    env = environment
    current = reviewed(env)
    exported = env.service.export_review(current["workflow_id"], target="fiftyone")
    sample = current["manifest"]["samples"][0]
    payload = {**exported, "results": [{"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
        "origin": "auto", "annotation": annotation(), "source": "offline-review"}]}
    changed = env.service.import_review(current["workflow_id"], expected_revision=current["revision"],
                                         payload=payload, actor="reviewer")
    review = changed["manifest"]["reviews"][sample["sample_id"]]
    assert review["decision"] == "needs_review" and review["annotation"] is None
    assert review["notes"] == "keep this human note"
    assert changed["manifest"]["review_history"][-1]["decision"] == "approved"
    with pytest.raises(TaskWorkflowConflict, match="report"):
        env.service.import_review(current["workflow_id"], expected_revision=changed["revision"], payload=payload, actor="reviewer")
    with pytest.raises(TaskWorkflowBlocked, match="HUMAN_REVIEW_INCOMPLETE"):
        env.service.freeze(current["workflow_id"], expected_revision=changed["revision"], actor="reviewer")


def test_frozen_dataset_is_hashed_session_disjoint_and_immutable(environment):
    env = environment
    current = reviewed(env)
    dataset = env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")
    assert env.service.get_dataset(dataset["dataset_id"]) == dataset
    assert {row["split"] for row in dataset["samples"]} == {"train", "validation"}
    with env.db.get_session() as session:
        stored = session.get(TaskDatasetRecord, dataset["dataset_id"])
        assert _sha(stored.manifest_json) == stored.manifest_sha256
    current = env.service.get(current["workflow_id"])
    sample = current["manifest"]["samples"][0]
    env.service.review(current["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
        expected_image_sha256=sample["image_sha256"], decision="rejected", actor="reviewer")
    assert env.service.get_dataset(dataset["dataset_id"]) == dataset
    assert env.service.get(current["workflow_id"])["manifest"]["dataset_id"] is None


@pytest.mark.parametrize("corruption", ["missing_sha", "wrong_sha", "schema", "binding", "dataset_id"])
def test_frozen_manifest_corruption_is_rejected(environment, corruption):
    env = environment
    current = reviewed(env)
    dataset = env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")
    # Explicitly disable the immutable trigger to model an old/corrupted DB,
    # not a supported application mutation.
    with env.db._engine.begin() as conn:
        triggers = conn.execute(text("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='task_datasets'")).scalars().all()
        for trigger in triggers:
            conn.execute(text(f'DROP TRIGGER "{trigger}"'))
    with env.db.get_session() as session:
        row = session.get(TaskDatasetRecord, dataset["dataset_id"])
        if corruption == "missing_sha":
            row.manifest_sha256 = None
        elif corruption == "wrong_sha":
            row.manifest_sha256 = "a" * 64
        else:
            changed = json.loads(row.manifest_json)
            if corruption == "binding":
                changed["binding"]["task_id"] = "different-task"
            else:
                changed[corruption] = "wrong"
            row.manifest_json = _canonical(changed)
            row.manifest_sha256 = _sha(row.manifest_json)
        session.commit()
    with pytest.raises(TaskWorkflowBlocked, match="DATASET_MANIFEST"):
        env.service.get_dataset(dataset["dataset_id"])


def test_zero_images_and_one_session_cannot_be_frozen(environment):
    env = environment
    current = seed_samples(env, count=0)
    with pytest.raises(TaskWorkflowBlocked):
        env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")


def test_task_revision_change_blocks_old_workflow_mutation(environment):
    env = environment
    current = reviewed(env)
    env.tasks.update_task(env.task["task_id"], expected_revision=1, name="new revision",
                          requested_by="operator", idempotency_key="change-revision")
    with pytest.raises(TaskWorkflowConflict, match="configuration changed"):
        env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")
    assert env.service.read_sample_image(current["workflow_id"], "sample-0") == png_image()


def test_collection_policy_override_is_explicitly_blocked(environment):
    env = environment
    with pytest.raises(TaskWorkflowBlocked, match="POLICY_MISMATCH"):
        env.service.start_collection(env.workflow["workflow_id"], expected_revision=1,
            actor="operator", manager=object(), params={"policy": "events"})
    assert env.service.get(env.workflow["workflow_id"])["state"] == "collectable"


@pytest.mark.parametrize("policy", ["disagreement", "low_confidence"])
def test_unimplemented_collection_policies_are_never_replaced_by_periodic(environment, policy):
    env = environment
    task = env.tasks.create_task("cam", "object_detection.v1", "unsupported collection", {
        "collection": {"policy": policy},
    }, requested_by="operator", idempotency_key="unsupported-policy")
    workflow = env.service.create(task["task_id"], actor="operator")
    with pytest.raises(TaskWorkflowBlocked, match="POLICY_NOT_IMPLEMENTED"):
        env.service.start_collection(workflow["workflow_id"], expected_revision=1,
                                     actor="operator", manager=object())
    assert env.service.get(workflow["workflow_id"])["state"] == "collectable"


def test_event_collection_requires_server_wired_provider_and_rejects_http_cursor(environment):
    env = environment
    task = env.tasks.create_task("cam", "object_detection.v1", "event collection", {
        "collection": {"policy": "events"},
    }, requested_by="operator", idempotency_key="event-policy")
    workflow = env.service.create(task["task_id"], actor="operator")
    with pytest.raises(ValueError, match="unknown collection parameters"):
        env.service.start_collection(workflow["workflow_id"], expected_revision=1,
            actor="operator", manager=object(), params={"start_cursor": 0})
    with pytest.raises(TaskWorkflowBlocked, match="EVENT_COLLECTION_UNAVAILABLE"):
        env.service.start_collection(workflow["workflow_id"], expected_revision=1,
            actor="operator", manager=object())
    with pytest.raises(ValueError, match="same database"):
        env.service.set_event_provider(SimpleNamespace(authority_for=lambda db: False))
    assert env.service.get(workflow["workflow_id"])["revision"] == 1


def test_candidate_receipt_projects_the_exact_original_request_and_preserves_historical_manifest(environment):
    env = environment
    current = export_ready(env)
    package = env.service.export_training_package(current["workflow_id"])
    calls = []

    def register(**kwargs):
        calls.append(kwargs)
        return {"artifact_id": "registered-model"}

    env.service._candidate_registrar = register
    candidate = {"relative_path": "candidate.onnx", "descriptor": {"name": "original request"}}
    args = {"expected_revision": current["revision"], "candidate": candidate,
            "expected_package_sha256": package["package_sha256"], "actor": "engineer"}
    first = env.service.import_candidate(current["workflow_id"], **args)
    expected = {"workflow_id": current["workflow_id"], "package_sha256": package["package_sha256"],
                "candidate": candidate}
    assert first["candidate_import_receipt"]["request_identity"] == expected
    assert first["manifest"]["training"]["candidate"]["request_identity"] == expected
    assert _sha(_canonical(first["manifest"])) == first["manifest_sha256"]
    # Model a valid historical receipt that predates the optional projection.
    with env.db.get_session() as session:
        row, manifest = env.service._record(session, current["workflow_id"])
        manifest["training"]["candidate"].pop("request_identity")
        row.manifest_json = _canonical(manifest)
        row.manifest_sha256 = _sha(row.manifest_json)
        session.commit()
    historical = env.service.get(current["workflow_id"])
    retried = env.service.import_candidate(current["workflow_id"], **args)
    assert retried["candidate_import_receipt"]["request_identity"] == expected
    assert retried["manifest"] == historical["manifest"]
    assert retried["manifest_sha256"] == historical["manifest_sha256"]
    assert len(calls) == 1
    with pytest.raises(TaskWorkflowConflict):
        env.service.import_candidate(current["workflow_id"], **{**args,
            "candidate": {**candidate, "relative_path": "another-candidate.onnx"}})


def test_collection_start_failure_is_retryable_and_keeps_zero_images_failed(environment):
    env = environment
    manager = SimpleNamespace(get_camera_source=lambda _: None)
    failed = env.service.start_collection(env.workflow["workflow_id"], expected_revision=1,
                                          actor="operator", manager=manager)
    assert failed["state"] == "collection_failed"
    assert not failed["manifest"]["samples"]
    repeated = env.service.start_collection(env.workflow["workflow_id"], expected_revision=failed["revision"],
                                            actor="operator", manager=manager)
    assert repeated["state"] == "collection_failed"
    assert len(repeated["manifest"]["collections"]) == 2


def test_orphan_recovery_requires_proof_owner_exited(environment, monkeypatch):
    env = environment
    with env.db.get_session() as session:
        record, manifest = env.service._record(session, env.workflow["workflow_id"])
        env.service._save(session, record, manifest, "collecting", operation={
            "kind": "collection", "collection_id": "old", "owner_id": "unknown",
            "process_id": os.getpid(), "process_identity": "unknown-process"})
        session.commit()
    current = env.service.get(env.workflow["workflow_id"])
    with pytest.raises(TaskWorkflowBlocked, match="OWNER_NOT_CONFIRMED"):
        env.service.stop_collection(current["workflow_id"], expected_revision=current["revision"], actor="operator")
    monkeypatch.setattr("argus.storage.task_workflows._process_alive", lambda _: False)
    recovered = env.service.stop_collection(current["workflow_id"], expected_revision=current["revision"], actor="operator")
    assert recovered["state"] == "collection_failed"
    assert recovered["operation"] is None


def test_external_training_unknown_candidate_commit_reuses_idempotency_and_preserves_notes(environment):
    env = environment
    current = export_ready(env)
    package = env.service.export_training_package(current["workflow_id"])
    assert package["local_training_state"] == "blocked" and package["production_readiness"] == "NO-GO"
    calls = []

    def registrar(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            raise RuntimeError("registration committed, acknowledgement lost")
        return {"artifact_id": "candidate-only"}

    env.service._candidate_registrar = registrar
    candidate = {"path": "managed.onnx"}
    with pytest.raises(RuntimeError, match="acknowledgement"):
        env.service.import_candidate(current["workflow_id"], expected_revision=current["revision"], candidate=candidate,
            expected_package_sha256=package["package_sha256"], actor="operator")
    interrupted = env.service.get(current["workflow_id"])
    assert interrupted["state"] == "candidate_importing"
    done = env.service.import_candidate(current["workflow_id"], expected_revision=interrupted["revision"], candidate=candidate,
        expected_package_sha256=package["package_sha256"], actor="second-operator")
    assert done["state"] == "candidate_registered"
    assert calls[0]["idempotency_key"] == calls[1]["idempotency_key"]
    assert calls[0]["actor"] == calls[1]["actor"] == "operator"
    assert done["manifest"]["reviews"]["sample-0"]["notes"] == "keep this human note"
    assert done["manifest"]["training"]["candidate"]["stage"] == "candidate"


def test_close_refuses_to_claim_cleanup_while_worker_or_persistence_pending(environment):
    env = environment
    worker = SimpleNamespace(stop=lambda: True, retry_finalize=lambda: False)
    env.service._collectors["mock"] = worker
    assert not env.service.close()
    worker.retry_finalize = lambda: True
    assert env.service.close()


def test_completed_receipt_allows_owned_collector_cleanup_retry_without_rewriting_reviews(environment, monkeypatch):
    from argus.capture.task_collector import TaskFrameCollector

    env = environment
    monkeypatch.setattr(TaskFrameCollector, "_cleanup", lambda self: False)
    failed = env.service.start_collection(env.workflow["workflow_id"], expected_revision=1,
        actor="operator", manager=SimpleNamespace(get_camera_source=lambda _: None))
    assert failed["state"] == "collection_failed"
    assert failed["collection_runtime"]["cleanup_confirmed"] is False
    collector = env.service._collectors[failed["workflow_id"]]

    def cleaned():
        collector._cleanup_confirmed = True
        return True

    monkeypatch.setattr(collector, "_cleanup", cleaned)
    recovered = env.service.stop_collection(failed["workflow_id"], expected_revision=failed["revision"], actor="operator")
    assert recovered["collection_runtime"]["cleanup_confirmed"] is True
    assert recovered["revision"] == failed["revision"]
    assert recovered["manifest"] == failed["manifest"]


def test_new_collection_claim_ack_loss_is_recoverable_when_terminal_prior_worker_remains(environment, monkeypatch):
    env = environment
    manager = SimpleNamespace(get_camera_source=lambda _: None)
    first = env.service.start_collection(env.workflow["workflow_id"], expected_revision=1,
        actor="operator", manager=manager, collection_id="first-collection")
    assert first["state"] == "collection_failed"
    commit = Session.commit
    lost = []

    def commit_then_lose_ack(session):
        new_claim = any(isinstance(row, TaskWorkflowRecord) and row.state == "collecting"
                        and json.loads(row.operation_json or "{}").get("collection_id") == "second-collection"
                        for row in session.identity_map.values())
        commit(session)
        if new_claim and not lost:
            lost.append(True)
            raise RuntimeError("new claim committed, acknowledgement lost")

    monkeypatch.setattr(Session, "commit", commit_then_lose_ack)
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        env.service.start_collection(first["workflow_id"], expected_revision=first["revision"],
            actor="operator", manager=manager, collection_id="second-collection")
    interrupted = env.service.get(first["workflow_id"])
    assert interrupted["operation"]["collection_id"] == "second-collection"
    assert interrupted["collection_runtime"]["collection_id"] == "first-collection"
    with pytest.raises(TaskWorkflowConflict, match="ownership changed"):
        env.service.stop_collection(first["workflow_id"], expected_revision=interrupted["revision"],
            actor="operator", expected_collection_id="first-collection")
    recovered = env.service.stop_collection(first["workflow_id"], expected_revision=interrupted["revision"],
        actor="operator", expected_collection_id="second-collection")
    assert recovered["state"] == "collection_failed"
    assert recovered["manifest"]["collections"][-1]["collection_id"] == "second-collection"
    assert recovered["manifest"]["collections"][-1]["error_code"] == "COLLECTION_OWNER_LOST"
    assert len(recovered["manifest"]["collections"]) == 2


def test_auto_annotation_after_training_approval_preserves_notes_but_revokes_workflow_approval(environment):
    env = environment
    current = export_ready(env)
    dataset_id = current["manifest"]["dataset_id"]
    original_dataset = env.service.get_dataset(dataset_id)
    sample = current["manifest"]["samples"][0]
    payload = env.service.export_review(current["workflow_id"], target="fiftyone")
    payload["results"] = [{"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
                           "origin": "auto", "annotation": annotation(), "source": {"report": "updated"}}]
    changed = env.service.import_review(current["workflow_id"], expected_revision=current["revision"],
                                       payload=payload, actor="reviewer")
    assert changed["state"] == "review"
    assert changed["manifest"]["dataset_id"] is None and changed["manifest"]["training"] is None
    assert changed["manifest"]["reviews"][sample["sample_id"]]["decision"] == "needs_review"
    assert changed["manifest"]["reviews"][sample["sample_id"]]["notes"] == "keep this human note"
    assert env.service.get_dataset(dataset_id) == original_dataset
    with pytest.raises(TaskWorkflowBlocked, match="APPROVED_EXTERNAL_TRAINING_REQUIRED"):
        env.service.export_training_package(current["workflow_id"])


def test_fixed_fast_motion_has_no_external_training_recipe_and_old_proposal_cannot_be_approved(environment):
    env = environment
    assert all("fast_motion.v1" not in recipe["templates"] for recipe in env.service.recipes.list())
    env.task = env.tasks.create_task("cam", "fast_motion.v1", "fast object", {},
        requested_by="operator", idempotency_key="fast-motion-task")
    env.workflow = env.service.create(env.task["task_id"], actor="operator")
    current = seed_samples(env)
    for sample in current["manifest"]["samples"]:
        current = env.service.review(current["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
            expected_image_sha256=sample["image_sha256"], decision="approved", actor="reviewer",
            annotation=annotation("fast_projectile"))
    env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")
    current = env.service.get(current["workflow_id"])
    proposal = env.service.propose_training(current["workflow_id"], expected_revision=current["revision"],
        recipe_id="external_detection.v1", actor="operator")
    assert proposal["state"] == "training_blocked"
    assert proposal["manifest"]["training"]["blocked_reason"] == "RECIPE_TEMPLATE_MISMATCH"
    # Model a proposal persisted by the previous recipe declaration. Reading
    # history stays possible, but the removed capability cannot be approved.
    with env.db.get_session() as session:
        record, manifest = env.service._record(session, current["workflow_id"])
        manifest["training"]["blocked_reason"] = None
        env.service._save(session, record, manifest, "training_proposed")
        session.commit()
    historical = env.service.get(current["workflow_id"])
    with pytest.raises(TaskWorkflowBlocked, match="RECIPE_TEMPLATE_MISMATCH"):
        env.service.approve_training(current["workflow_id"], expected_revision=historical["revision"], actor="reviewer")
    assert env.service.get(current["workflow_id"])["manifest"]["training"]["approved_by"] is None


def test_workflow_record_binding_mismatch_rejected_even_with_matching_digest(environment):
    env = environment
    with env.db.get_session() as session:
        record = session.get(TaskWorkflowRecord, env.workflow["workflow_id"])
        manifest = json.loads(record.manifest_json)
        manifest["binding"]["task_id"] = "another-task"
        record.manifest_json = _canonical(manifest)
        record.manifest_sha256 = _sha(record.manifest_json)
        session.commit()
    with pytest.raises(TaskWorkflowBlocked, match="SCHEMA_INVALID"):
        env.service.get(env.workflow["workflow_id"])


def test_sample_declared_dimensions_must_match_png_header(environment):
    env = environment
    current = seed_samples(env)
    sample = current["manifest"]["samples"][0]
    sample["image_size"] = [1920, 1080]
    with pytest.raises(TaskWorkflowConflict, match="dimensions"):
        env.service._verify_sample(sample, current["manifest"]["binding"])


def test_collection_unknown_commit_can_replay_after_review_suggestion_changed(environment):
    env = environment
    current = seed_samples(env)
    sample = current["manifest"]["samples"][0]
    receipt = {"collection_id": sample["collection_id"], "state": "completed", "error_code": None,
               "binding": current["manifest"]["binding"], "samples": [copy.deepcopy(sample)],
               "discarded_frames": 0, "dropped_frames": 0}
    with env.db.get_session() as session:
        record, manifest = env.service._record(session, current["workflow_id"])
        manifest["samples"] = []
        env.service._save(session, record, manifest, "collecting", operation={
            "kind": "collection", "collection_id": sample["collection_id"], "owner_id": env.service._owner_id})
        session.commit()
    env.service._finish_collection(current["workflow_id"], sample["collection_id"], receipt)
    with env.db.get_session() as session:
        record, manifest = env.service._record(session, current["workflow_id"])
        manifest["samples"][0]["auto_annotation"] = {"annotation": annotation()}
        env.service._save(session, record, manifest, "review")
        session.commit()
    before = env.service.get(current["workflow_id"])
    env.service._finish_collection(current["workflow_id"], sample["collection_id"], receipt)
    assert env.service.get(current["workflow_id"])["revision"] == before["revision"]


def test_undeclared_labels_are_closed_deterministically_when_dataset_is_frozen(environment):
    env = environment
    env.task = env.tasks.create_task("cam", "object_detection.v1", "open labels", {},
        requested_by="operator", idempotency_key="open-labels")
    env.workflow = env.service.create(env.task["task_id"], actor="operator")
    current = reviewed(env)
    dataset = env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")
    assert dataset["label_schema"] == {"schema": "argus.task-label-schema.v1", "labels": ["person"],
                                       "origin": "reviewed_annotations", "kind": "detection"}
    assert env.service.get_dataset(dataset["dataset_id"]) == dataset


def test_negative_only_undeclared_dataset_cannot_export_guessed_training_classes(environment):
    env = environment
    env.task = env.tasks.create_task("cam", "object_detection.v1", "negative only", {},
        requested_by="operator", idempotency_key="negative-only")
    env.workflow = env.service.create(env.task["task_id"], actor="operator")
    current = seed_samples(env)
    for sample in current["manifest"]["samples"]:
        negative = annotation()
        negative["objects"] = []
        current = env.service.review(current["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
            expected_image_sha256=sample["image_sha256"], decision="approved", actor="reviewer", annotation=negative)
    env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")
    current = env.service.get(current["workflow_id"])
    current = env.service.propose_training(current["workflow_id"], expected_revision=current["revision"],
                                          recipe_id="external_detection.v1", actor="operator")
    with pytest.raises(TaskWorkflowBlocked, match="LABEL_SCHEMA_EMPTY"):
        env.service.approve_training(current["workflow_id"], expected_revision=current["revision"], actor="reviewer")
    assert env.service.get(current["workflow_id"])["manifest"]["training"]["approved_by"] is None


def test_final_candidate_commit_ack_loss_returns_same_receipt_on_exact_retry(environment, monkeypatch):
    env = environment
    current = export_ready(env)
    package = env.service.export_training_package(current["workflow_id"])
    calls = []
    env.service._candidate_registrar = lambda **kwargs: calls.append(kwargs) or {"artifact_id": "candidate"}
    commit = Session.commit
    lost = []

    def commit_then_lose_ack(session):
        final = any(isinstance(row, TaskWorkflowRecord) and row.state == "candidate_registered"
                    for row in session.identity_map.values())
        commit(session)
        if final and not lost:
            lost.append(True)
            raise RuntimeError("final acknowledgement lost")

    monkeypatch.setattr(Session, "commit", commit_then_lose_ack)
    arguments = {"expected_revision": current["revision"], "candidate": {"path": "candidate.onnx"},
                 "expected_package_sha256": package["package_sha256"], "actor": "operator"}
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        env.service.import_candidate(current["workflow_id"], **arguments)
    done = env.service.import_candidate(current["workflow_id"], **arguments)
    assert done["state"] == "candidate_registered"
    assert len(calls) == 1


@pytest.mark.parametrize("denied_check", [1, 2, 3, 4])
def test_candidate_rechecks_live_personal_authority_and_preserves_pending_identity(environment, denied_check):
    env = environment
    current = export_ready(env)
    package = env.service.export_training_package(current["workflow_id"])
    calls, authorizations = [], []
    env.service._candidate_registrar = lambda **kwargs: calls.append(copy.deepcopy(kwargs)) or {"artifact_id": "candidate"}

    def authorize():
        authorizations.append(True)
        return len(authorizations) != denied_check

    arguments = {"expected_revision": current["revision"], "candidate": {"path": "candidate.onnx"},
                 "expected_package_sha256": package["package_sha256"], "actor": "operator"}
    with pytest.raises(TaskWorkflowBlocked, match="ACTOR_AUTHORITY_CHANGED"):
        env.service.import_candidate(current["workflow_id"], **arguments, authorize=authorize)
    pending = env.service.get(current["workflow_id"])
    assert pending["manifest"]["reviews"] == current["manifest"]["reviews"]
    assert pending["state"] == ("training_export_ready" if denied_check <= 2 else "candidate_importing")
    assert len(calls) == (1 if denied_check == 4 else 0)
    done = env.service.import_candidate(current["workflow_id"], **arguments, authorize=lambda: True)
    assert done["state"] == "candidate_registered"
    assert done["manifest"]["reviews"] == current["manifest"]["reviews"]
    assert len({call["idempotency_key"] for call in calls}) == 1


def test_pending_candidate_keeps_original_request_separate_from_normalized_descriptor(environment):
    env = environment
    current = export_ready(env)
    package = env.service.export_training_package(current["workflow_id"])
    candidate = {"relative_path": "candidate.onnx", "descriptor": {"name": "original"}}
    normalized = {"relative_path": "candidate.onnx", "descriptor": {"name": "original", "input_layout": "NCHW"},
                  "artifact_sha256": "a" * 64}
    env.service._candidate_validator = lambda **kwargs: copy.deepcopy(normalized)
    env.service._candidate_registrar = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unknown registry write"))
    args = {"expected_revision": current["revision"], "candidate": candidate,
            "expected_package_sha256": package["package_sha256"], "actor": "engineer"}
    with pytest.raises(RuntimeError, match="unknown registry"):
        env.service.import_candidate(current["workflow_id"], **args)
    pending = env.service.get(current["workflow_id"])
    assert pending["operation"]["candidate"] == normalized
    assert pending["operation"]["request_identity"] == {
        "workflow_id": current["workflow_id"], "package_sha256": package["package_sha256"], "candidate": candidate,
    }
    with env.db.get_session() as session:
        row = session.get(TaskWorkflowRecord, current["workflow_id"])
        operation = json.loads(row.operation_json)
        operation["request_identity"]["candidate"]["relative_path"] = "wrong.onnx"
        row.operation_json = _canonical(operation)
        session.commit()
    with pytest.raises(TaskWorkflowConflict, match="request identity changed"):
        env.service.import_candidate(current["workflow_id"], **args)


def test_same_candidate_concurrent_registrations_have_one_receipt_and_keep_human_notes(environment):
    env = environment
    current = export_ready(env)
    package = env.service.export_training_package(current["workflow_id"])
    barrier, calls = threading.Barrier(2), []

    def registrar(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        barrier.wait(timeout=5)
        return {"artifact_id": "same-registered-artifact"}

    env.service._candidate_registrar = registrar
    args = {"expected_revision": current["revision"], "candidate": {"path": "candidate.onnx"},
            "expected_package_sha256": package["package_sha256"], "actor": "engineer"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(env.service.import_candidate, current["workflow_id"], **args) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0] == results[1]
    assert results[0]["revision"] == current["revision"] + 2
    assert results[0]["manifest"]["reviews"] == current["manifest"]["reviews"]
    assert calls[0]["idempotency_key"] == calls[1]["idempotency_key"]


def test_collection_admission_keeps_unknown_claims_and_local_cleanup_ownership(environment):
    env = environment
    old = env.workflow
    newer = env.service.create(env.task["task_id"], actor="operator")
    with env.db.get_session() as session:
        row, manifest = env.service._record(session, old["workflow_id"])
        env.service._save(session, row, manifest, "collecting", operation={
            "collection_id": "unknown-start", "owner_id": env.service._owner_id,
        })
        session.commit()
    with pytest.raises(TaskWorkflowBlocked, match="TASK_COLLECTION_RESOURCE_BUSY"):
        env.service.start_collection(newer["workflow_id"], expected_revision=1, manager=object(), actor="operator")
    claimed = env.service.get(old["workflow_id"])
    env.service.stop_collection(old["workflow_id"], expected_revision=claimed["revision"], actor="operator")
    local_state = {"state": "failed", "cleanup_confirmed": False, "persistence_pending": False}
    env.service._collectors[old["workflow_id"]] = SimpleNamespace(snapshot=lambda: dict(local_state))
    try:
        with pytest.raises(TaskWorkflowBlocked, match="TASK_COLLECTION_RESOURCE_BUSY"):
            env.service.start_collection(newer["workflow_id"], expected_revision=1, manager=object(), actor="operator")
    finally:
        env.service._collectors.pop(old["workflow_id"])
    allowed = env.service.start_collection(newer["workflow_id"], expected_revision=1,
        manager=SimpleNamespace(get_camera_source=lambda _: None), actor="operator")
    assert allowed["state"] == "collection_failed"  # Admitted; synthetic source is absent.


def test_global_collection_admission_limit_is_checked_before_allocating_a_worker(environment, monkeypatch):
    env = environment
    monkeypatch.setattr("argus.storage.task_workflows._MAX_ACTIVE_COLLECTIONS", 1)
    with env.db.get_session() as session:
        row, manifest = env.service._record(session, env.workflow["workflow_id"])
        env.service._save(session, row, manifest, "collecting", operation={"collection_id": "held"})
        session.commit()
    task = env.tasks.create_task("cam", "object_detection.v1", "other task", {},
        requested_by="operator", idempotency_key="another-admission-task")
    workflow = env.service.create(task["task_id"], actor="operator")
    with pytest.raises(TaskWorkflowBlocked, match="COLLECTION_RESOURCE_LIMIT"):
        env.service.start_collection(workflow["workflow_id"], expected_revision=1, manager=object(), actor="operator")
    assert workflow["workflow_id"] not in env.service._collectors
    assert env.service.get(workflow["workflow_id"])["revision"] == 1


def _event_sample(sample):
    sample = copy.deepcopy(sample)
    source = {key: sample["identity"][key] for key in (
        "camera_id", "source_session_id", "source_epoch", "frame_index", "source_frame_index", "pts",
        "time_base_num", "time_base_den", "frame_index_provenance", "timestamp_provenance",
        "capture_wall_ts", "capture_monotonic_ts")}
    source["source_runtime_hash"] = "a" * 10
    sample["identity"] = source
    payload = {"schema_version": "argus.task-event.v1", "event_type": "object_detected", "event_id": "event-one",
               "binding": {**sample["binding"], "bundle_id": "bundle-one", "manifest_sha256": "b" * 64,
                           "execution_id": "execution-one", "deployment_revision": 1,
                           "deployment_operation_id": "operation-one"}, "source": source,
               "first_source": copy.deepcopy(source), "frame_size": {"width": 3, "height": 2},
               "replay": {"sha256": "e" * 64}}
    sample["event_origin"] = {"schema": "argus.task-event-sample.v1", "event_id": "event-one",
        "evidence_sha256": "e" * 64, "derivation": "verified_jpeg_to_png.v1", "payload": payload,
        "immutable_sha256": _sha(_canonical({"payload": payload, "evidence_sha256": "e" * 64}))}
    return sample


@pytest.mark.parametrize("change", ["pts_type", "half_time_base", "size_type", "replay_sha", "payload_schema"])
def test_frozen_event_sample_checks_full_typed_provenance_not_only_a_self_digest(environment, change):
    env = environment
    current = seed_samples(env, count=1)
    sample = _event_sample(current["manifest"]["samples"][0])
    assert env.service._verify_sample(sample, current["manifest"]["binding"])
    payload = sample["event_origin"]["payload"]
    if change == "pts_type":
        sample["identity"]["pts"] = float(sample["identity"]["pts"])
    elif change == "half_time_base":
        sample["identity"]["time_base_num"] = None
    elif change == "size_type":
        payload["frame_size"]["width"] = 3.0
    elif change == "replay_sha":
        payload["replay"]["sha256"] = "f" * 64
    else:
        payload["schema_version"] = "wrong-event-schema"
    sample["event_origin"]["immutable_sha256"] = _sha(_canonical({"payload": payload, "evidence_sha256": "e" * 64}))
    with pytest.raises((TaskWorkflowConflict, ValueError)):
        env.service._verify_sample(sample, current["manifest"]["binding"])


def test_collection_receipt_retry_rejects_equal_looking_float_pts_without_changing_notes(environment):
    env = environment
    current = seed_samples(env, count=1)
    sample = current["manifest"]["samples"][0]
    receipt = {"collection_id": sample["collection_id"], "state": "completed", "error_code": None,
               "binding": current["manifest"]["binding"], "samples": [copy.deepcopy(sample)]}
    with env.db.get_session() as session:
        record, manifest = env.service._record(session, current["workflow_id"])
        manifest["samples"] = []
        env.service._save(session, record, manifest, "collecting", operation={
            "kind": "collection", "collection_id": sample["collection_id"], "owner_id": env.service._owner_id})
        session.commit()
    env.service._finish_collection(current["workflow_id"], sample["collection_id"], receipt)
    current = env.service.get(current["workflow_id"])
    reviewed = env.service.review(current["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
        expected_image_sha256=sample["image_sha256"], decision="needs_review", notes="keep independent human note",
        actor="reviewer")
    replay = copy.deepcopy(receipt)
    replay["samples"][0]["identity"]["pts"] = float(sample["identity"]["pts"])
    with pytest.raises(TaskWorkflowConflict, match="receipt changed"):
        env.service._finish_collection(current["workflow_id"], sample["collection_id"], replay)
    assert env.service.get(current["workflow_id"])["manifest"]["reviews"] == reviewed["manifest"]["reviews"]
