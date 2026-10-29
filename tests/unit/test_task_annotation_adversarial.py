"""Adversarial annotation exchange: mock PNG evidence, no SDK or inference."""

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy.orm import Session

from argus.storage.task_workflows import TaskWorkflowBlocked, TaskWorkflowConflict
from test_task_workflows import annotation, export_ready, reviewed, seed_samples
from test_task_workflows import environment as _workflow_environment

# Reuse the existing mock-only fixture without redeclaring its setup/teardown.
environment = _workflow_environment


def returned(env, *, current=None, target="cvat", all_samples=False):
    current = current or seed_samples(env)
    payload = env.service.export_annotation_exchange(current["workflow_id"], target=target)
    samples = payload["samples"] if all_samples else payload["samples"][:1]
    payload["results"] = [{**copy.deepcopy(sample), "annotation": annotation(),
                           "notes": "external note, not a personal approval"} for sample in samples]
    return current, payload


def ingest(env, current, payload, *, actor="tool-importer"):
    return env.service.import_annotation_exchange(current["workflow_id"],
        expected_revision=current["revision"], payload=payload, actor=actor)


@pytest.mark.parametrize("target", ["cvat", "fiftyone"])
def test_task_annotation_adversarial_export_is_image_only_and_exact(environment, target):
    env = environment
    current, payload = returned(env, target=target)
    assert payload["external_side_effects"] is False
    assert payload["report_sha256"] == current["manifest_sha256"]
    assert payload["label_schema"]["labels"] == ["person"]
    assert payload["approval_policy"] == "external_annotations_require_argus_human_review"
    for sample in payload["samples"]:
        assert set(sample) == {"sample_id", "image_sha256", "image_size", "identity"}
        assert sample["identity"]["source_frame_index"] >= 200
        assert sample["identity"]["pts"] >= 10000
        assert sample["identity"]["time_base_den"] == 90000
        assert not {"relative_path", "source_runtime", "source_declared"} & sample.keys()


@pytest.mark.parametrize("field", [
    "schema", "workflow_id", "workflow_revision", "report_sha256", "binding", "label_schema",
    "label_schema_sha256", "media_type", "external_side_effects", "approval_policy", "samples",
])
def test_task_annotation_adversarial_changed_envelope_is_atomic(environment, field):
    env = environment
    current, payload = returned(env)
    if field == "schema":
        payload[field] = "argus.task-annotation-exchange.v1"
    elif field == "workflow_revision":
        payload[field] = float(payload[field])
    elif field == "external_side_effects":
        payload[field] = 0
    elif field in {"binding", "label_schema", "samples"}:
        payload[field] = {} if field != "samples" else []
    else:
        payload[field] = "forged"
    with pytest.raises((TaskWorkflowConflict, ValueError)):
        ingest(env, current, payload)
    assert env.service.get(current["workflow_id"])["manifest_sha256"] == current["manifest_sha256"]


@pytest.mark.parametrize("field", [
    "image_sha256", "image_size", "camera_id", "source_session_id", "source_epoch", "frame_index",
    "source_frame_index", "pts", "time_base_num", "time_base_den", "capture_wall_ts",
    "capture_monotonic_ts", "frame_index_provenance", "timestamp_provenance",
])
def test_task_annotation_adversarial_changed_returned_source_is_atomic(environment, field):
    env = environment
    current, payload = returned(env)
    row = payload["results"][0]
    if field in {"image_sha256", "image_size"}:
        row[field] = "a" * 64 if field == "image_sha256" else [2, 3]
    else:
        row["identity"][field] = "forged"
    with pytest.raises(TaskWorkflowConflict, match="source identity"):
        ingest(env, current, payload)
    assert env.service.get(current["workflow_id"])["revision"] == current["revision"]


@pytest.mark.parametrize("change", ["duplicate", "unknown", "missing_identity", "approval", "label", "bad_last_row"])
def test_task_annotation_adversarial_rows_are_validated_before_any_write(environment, change):
    env = environment
    current, payload = returned(env)
    if change == "duplicate":
        payload["results"].append(copy.deepcopy(payload["results"][0]))
    elif change == "unknown":
        payload["results"][0]["sample_id"] = "unknown-sample"
    elif change == "missing_identity":
        del payload["results"][0]["identity"]
    elif change == "approval":
        payload["results"][0]["decision"] = "approved"
    elif change == "label":
        payload["results"][0]["annotation"] = annotation("not-a-task-label")
    else:
        payload["results"].append({**copy.deepcopy(payload["samples"][1]), "annotation": annotation("forged")})
    with pytest.raises((ValueError, TaskWorkflowConflict)):
        ingest(env, current, payload)
    stored = env.service.get(current["workflow_id"])
    assert stored["manifest_sha256"] == current["manifest_sha256"]
    assert stored["manifest"]["reviews"] == {}
    assert all(sample["auto_annotation"] is None for sample in stored["manifest"]["samples"])


def test_task_annotation_adversarial_return_revokes_review_dataset_and_training(environment):
    env = environment
    approved = export_ready(env)
    old_dataset = env.service.get_dataset(approved["manifest"]["dataset_id"])
    current, payload = returned(env, current=approved)
    changed = ingest(env, current, payload)
    sample_id = payload["results"][0]["sample_id"]
    review = changed["manifest"]["reviews"][sample_id]
    sample = next(row for row in changed["manifest"]["samples"] if row["sample_id"] == sample_id)
    assert changed["state"] == "review"
    assert changed["manifest"]["dataset_id"] is None and changed["manifest"]["training"] is None
    assert review["decision"] == "needs_review" and review["annotation"] is None
    assert review["notes"] == "keep this human note"
    assert sample["auto_annotation"]["external_notes"] == payload["results"][0]["notes"]
    assert changed["manifest"]["review_history"][-1]["decision"] == "approved"
    assert env.service.get_dataset(old_dataset["dataset_id"]) == old_dataset
    with pytest.raises(TaskWorkflowBlocked, match="HUMAN_REVIEW_INCOMPLETE"):
        env.service.freeze(current["workflow_id"], expected_revision=changed["revision"], actor="reviewer")
    with pytest.raises((TaskWorkflowConflict, TaskWorkflowBlocked)):
        env.service.export_training_package(current["workflow_id"])
    assert changed["production_readiness"] == "NO-GO"


def test_task_annotation_adversarial_identical_concurrent_returns_have_one_mutation(environment):
    env = environment
    current, payload = returned(env)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ingest(env, current, payload), range(8)))
    assert all(result == results[0] for result in results)
    assert results[0]["revision"] == current["revision"] + 1
    with pytest.raises(TaskWorkflowConflict):
        ingest(env, current, payload, actor="different-person")


@pytest.mark.parametrize("committed", [False, True])
def test_task_annotation_adversarial_unknown_commit_is_retryable(environment, monkeypatch, committed):
    env = environment
    current, payload = returned(env)
    commit = Session.commit

    def lose_ack(session):
        if committed:
            commit(session)
        raise RuntimeError("simulated annotation acknowledgement loss")

    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", lose_ack)
        with pytest.raises(RuntimeError):
            ingest(env, current, payload)
    changed = ingest(env, current, payload)
    assert changed["revision"] == current["revision"] + 1
    assert changed["manifest"]["reviews"]["sample-0"]["review_revision"] == 1


def test_task_annotation_adversarial_old_envelope_cannot_apply_to_new_report(environment):
    env = environment
    current, payload = returned(env)
    sample = current["manifest"]["samples"][0]
    changed = env.service.review(current["workflow_id"], sample["sample_id"],
        expected_revision=current["revision"], expected_image_sha256=sample["image_sha256"],
        decision="approved", actor="reviewer", annotation=annotation())
    with pytest.raises(TaskWorkflowConflict):
        ingest(env, changed, payload)
    assert env.service.get(current["workflow_id"])["manifest_sha256"] == changed["manifest_sha256"]


def test_task_annotation_adversarial_v1_external_human_cannot_restore_approval(environment):
    env = environment
    current = reviewed(env)
    payload = env.service.export_review(current["workflow_id"], target="cvat")
    sample = current["manifest"]["samples"][0]
    payload["results"] = [{"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
                           "origin": "human", "decision": "approved", "annotation": annotation()}]
    with pytest.raises(TaskWorkflowBlocked, match="LEGACY_EXTERNAL_HUMAN"):
        env.service.import_review(current["workflow_id"], expected_revision=current["revision"],
                                  payload=payload, actor="importing-person")
    assert env.service.get(current["workflow_id"])["manifest_sha256"] == current["manifest_sha256"]


def test_task_annotation_adversarial_v1_auto_remains_draft_and_revokes_training(environment):
    env = environment
    current = export_ready(env)
    payload = env.service.export_review(current["workflow_id"], target="fiftyone")
    sample = current["manifest"]["samples"][0]
    payload["results"] = [{"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
                           "origin": "auto", "decision": "approved", "notes": "overwrite attempt",
                           "annotation": annotation(), "source": "mock-shadow-evidence"}]
    changed = env.service.import_review(current["workflow_id"], expected_revision=current["revision"],
                                        payload=payload, actor="personal-importer")
    review = changed["manifest"]["reviews"][sample["sample_id"]]
    assert review["decision"] == "needs_review" and review["annotation"] is None
    assert review["notes"] == "keep this human note"
    assert changed["manifest"]["dataset_id"] is None and changed["manifest"]["training"] is None
    with pytest.raises(TaskWorkflowBlocked, match="HUMAN_REVIEW_INCOMPLETE"):
        env.service.freeze(current["workflow_id"], expected_revision=changed["revision"], actor="reviewer")


def test_task_annotation_adversarial_receipt_retry_does_not_revoke_later_human_review(environment):
    env = environment
    current, payload = returned(env)
    changed = ingest(env, current, payload)
    sample = changed["manifest"]["samples"][0]
    approved = env.service.review(current["workflow_id"], sample["sample_id"],
        expected_revision=changed["revision"], expected_image_sha256=sample["image_sha256"],
        decision="approved", actor="reviewer", annotation=annotation(), notes="later human note")
    retry = ingest(env, current, payload)
    assert retry["manifest_sha256"] == approved["manifest_sha256"]
    assert retry["revision"] == approved["revision"]
    assert retry["manifest"]["reviews"][sample["sample_id"]]["decision"] == "approved"
    assert retry["manifest"]["reviews"][sample["sample_id"]]["notes"] == "later human note"


def test_task_annotation_adversarial_changed_image_blocks_return_without_mutation(environment):
    env = environment
    current, payload = returned(env)
    sample = current["manifest"]["samples"][0]
    (env.service.root / sample["relative_path"]).write_bytes(b"tampered mock image")
    with pytest.raises(TaskWorkflowConflict, match="image content changed"):
        ingest(env, current, payload)
    assert env.service.get(current["workflow_id"])["manifest_sha256"] == current["manifest_sha256"]
