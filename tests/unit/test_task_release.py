"""Release dossiers with opaque mock files only; never execute an ML engine."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from argus.dashboard.task_release_contracts import TaskReleaseRequest, TaskReleaseReviewRequest
from argus.storage.analysis_tasks import (
    AnalysisTaskConflictError, AnalysisTaskIntegrityError, AnalysisTaskService, AnalysisTaskValidationError,
)
from argus.storage.database import Database
from argus.storage.models import TaskDeploymentRecord, TaskReleaseRecord
from argus.storage.task_release import REQUIRED_EVIDENCE_ROLES, TaskReleaseBlocked, TaskReleaseService


@pytest.fixture
def release_platform(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'release.db'}")
    db.initialize()
    root = tmp_path / "models"
    root.mkdir()
    (root / "opaque.onnx").write_bytes(b"not executable; model identity fixture only")
    tasks = AnalysisTaskService(db, models_root=root)
    task = tasks.create_task("camera-a", "object_detection.v1", "Objects", {},
                             requested_by="requester", idempotency_key="create-task")
    model = tasks.register_task_model("opaque.onnx", {
        "name": "Opaque test model", "adapter_id": "onnx_detection.v1",
        "output_schema": "yolov8_detection.v1", "labels": ["person", "package"],
        "input_size": [640, 640],
    }, requested_by="requester", idempotency_key="register-model")
    bundle = tasks.freeze_bundle(task["task_id"], expected_revision=1,
        model_ids={"detector": model["model_id"]}, requested_by="requester", idempotency_key="freeze")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    sources = []
    for role in sorted(REQUIRED_EVIDENCE_ROLES):
        # A client claiming "passed" is deliberately not a trusted verdict.
        payload = json.dumps({"role": role, "passed": True}).encode()
        (evidence / f"{role}.json").write_bytes(payload)
        sources.append({"role": role, "relative_path": f"{role}.json",
                        "sha256": hashlib.sha256(payload).hexdigest()})
    service = TaskReleaseService(db, tasks, evidence_root=evidence)
    yield db, tasks, task, bundle, service, evidence, sources
    db.close()


def request(platform, *, key="request-release", sources=None, actor="requester"):
    return platform[4].request(platform[2]["task_id"], bundle_id=platform[3]["bundle_id"],
        expected_task_revision=1, expected_deployment_revision=0,
        evidence_sources=platform[6] if sources is None else sources,
        actor=actor, idempotency_key=key)


def review(platform, document, *, key="review-release", actor="reviewer", decision="approve", notes="checked"):
    return platform[4].review(document["release_id"], expected_revision=document["revision"],
        expected_subject_sha256=document["subject_sha256"], decision=decision, notes=notes,
        actor=actor, idempotency_key=key)


def test_dossier_freezes_exact_task_bundle_model_scene_and_sampling(release_platform):
    document = request(release_platform)
    binding = document["binding"]
    assert document["state"] == "requested"
    assert document["task_id"] == release_platform[2]["task_id"]
    assert document["bundle_id"] == release_platform[3]["bundle_id"]
    assert binding["task_revision"] == 1
    assert binding["deployment_revision"] == 0
    assert binding["task_config_sha256"] == release_platform[2]["config_sha256"]
    assert binding["bundle_manifest_sha256"] == release_platform[3]["manifest_sha256"]
    assert binding["config"]["sampling"] == release_platform[2]["config"]["sampling"]
    assert binding["config"]["scene"] == release_platform[2]["config"]["scene"]
    assert binding["models"] == release_platform[3]["manifest"]["models"]
    assert all(set(item) == {"role", "sha256"} for item in document["evidence_sources"])
    assert "relative_path" not in json.dumps(release_platform[4].get(document["release_id"]))
    assert document["production_allowed"] is False


def test_independent_review_and_client_pass_flags_never_authorize_production(release_platform):
    approved = review(release_platform, request(release_platform))
    assert approved["state"] == "review_approved"
    service, task = release_platform[4], release_platform[2]
    current = service.get(approved["release_id"])
    assert current["review_effective"] is True
    readiness = service.readiness(task["task_id"], approved["release_id"])
    assert readiness["production_ready"] is False
    assert readiness["production_supported"] is False
    assert "TEMPLATE_POLICY_NOT_APPROVED" in readiness["reasons"]
    assert "TRUSTED_TASK_EVIDENCE_VERIFIER_UNAVAILABLE" in readiness["reasons"]
    assert "PRODUCTION_APPLY_UNSUPPORTED" in readiness["reasons"]
    identity = dict(bundle_id=release_platform[3]["bundle_id"], deployment_revision=0,
                    execution_id="untrusted-production", source_session_id="source", source_epoch=1)
    assert service.authorize_event(task["task_id"], **identity)["authorized"] is False
    with release_platform[0].get_session() as session:
        assert service.authorize_event(task["task_id"], session=session, **identity)["authorized"] is False
        deployment = session.get(TaskDeploymentRecord, task["task_id"])
        assert deployment.applied_bundle_id is None and deployment.deployment_revision == 0


def test_self_review_is_refused_and_incomplete_evidence_is_not_approved(release_platform):
    document = request(release_platform)
    with pytest.raises(TaskReleaseBlocked, match="INDEPENDENT"):
        review(release_platform, document, actor="requester")
    incomplete = request(release_platform, key="empty-dossier", sources=[])
    with pytest.raises(TaskReleaseBlocked, match="INCOMPLETE"):
        review(release_platform, incomplete, key="approve-incomplete")
    rejected = review(release_platform, incomplete, key="reject-incomplete", decision="reject", notes="missing proof")
    assert rejected["state"] == "rejected"
    assert rejected["review"]["notes"] == "missing proof"


@pytest.mark.parametrize("username", ["system", "automation", "api-token"])
def test_personal_account_names_are_not_used_as_authentication_kind(release_platform, username):
    document = request(release_platform, actor=username)
    approved = review(release_platform, document, actor="independent-reviewer")
    assert approved["created_by"] == username
    assert approved["state"] == "review_approved"
    assert approved["production_allowed"] is False


def test_report_change_invalidates_review_preserves_notes_and_requires_new_subject(release_platform):
    service, evidence, sources = release_platform[4:]
    document = review(release_platform, request(release_platform), notes="retain my review notes")
    original_subject = document["subject_sha256"]
    path = evidence / "evaluation.json"
    path.write_bytes(b'{"changed": true}')
    stale = service.get(document["release_id"])
    assert stale["state"] == "stale" and stale["review_effective"] is False
    assert stale["review"]["notes"] == "retain my review notes"
    assert "EVIDENCE_SOURCE_IDENTITY_MISMATCH" in stale["invalidation_reasons"]
    changed = [dict(item) for item in sources]
    next(item for item in changed if item["role"] == "evaluation")["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    revised = service.revise_evidence(document["release_id"], expected_revision=2,
        evidence_sources=changed, actor="requester", idempotency_key="revise-evidence")
    assert revised["state"] == "requested" and revised["review"] is None
    assert revised["review_history"][0]["notes"] == "retain my review notes"
    assert revised["subject_sha256"] != original_subject
    with pytest.raises(AnalysisTaskConflictError, match="subject changed"):
        service.review(revised["release_id"], expected_revision=3, expected_subject_sha256=original_subject,
            decision="approve", notes="old form", actor="reviewer", idempotency_key="stale-review-form")
    approved = review(release_platform, revised, key="review-new-evidence")
    assert service.get(approved["release_id"])["review_effective"] is True
    assert service.readiness(release_platform[2]["task_id"], approved["release_id"])["production_ready"] is False


def test_evidence_reviser_cannot_approve_their_own_subject(release_platform):
    service, evidence, sources = release_platform[4:]
    document = review(release_platform, request(release_platform), notes="original review")
    changed = [dict(item) for item in sources]
    path = evidence / "evaluation.json"
    path.write_bytes(b'{"revision": 2}')
    next(item for item in changed if item["role"] == "evaluation")["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    revised = service.revise_evidence(document["release_id"], expected_revision=2,
        evidence_sources=changed, actor="evidence-editor", idempotency_key="edit-evidence")
    assert revised["evidence_authors"] == ["evidence-editor", "requester"]
    with pytest.raises(TaskReleaseBlocked, match="INDEPENDENT"):
        review(release_platform, revised, key="self-approve-edit", actor="evidence-editor")
    approved = review(release_platform, revised, key="independent-after-edit", actor="another-reviewer")
    assert approved["review_history"][0]["notes"] == "original review"
    assert service.get(approved["release_id"])["review_effective"] is True
    assert approved["production_allowed"] is False


def test_old_dossier_without_evidence_authors_is_readable_but_not_approved(release_platform):
    service = release_platform[4]
    document = review(release_platform, request(release_platform), notes="old review notes")
    with release_platform[0].get_session() as session:
        row = session.get(TaskReleaseRecord, document["release_id"])
        historical = json.loads(row.document_json)
        historical.pop("evidence_authors", None)
        service._save(row, historical)
        session.commit()
    current = service.get(document["release_id"])
    assert current["state"] == "stale" and current["review_effective"] is False
    assert "EVIDENCE_AUTHOR_IDENTITY_UNAVAILABLE" in current["invalidation_reasons"]
    assert current["review"]["notes"] == "old review notes"
    revised = service.revise_evidence(document["release_id"], expected_revision=2,
        evidence_sources=release_platform[6], actor="new-submitter", idempotency_key="restate-old-evidence")
    with pytest.raises(TaskReleaseBlocked, match="INDEPENDENT"):
        review(release_platform, revised, actor="new-submitter", key="self-review-old-evidence")
    approved = review(release_platform, revised, actor="independent", key="review-restated-evidence")
    assert service.get(approved["release_id"])["review_effective"] is True


@pytest.mark.parametrize("change", ["task_revision", "deployment_revision", "model_bytes"])
def test_binding_changes_invalidate_but_never_erase_historical_review(release_platform, change):
    db, tasks, task, _bundle, service, _root, _sources = release_platform
    document = review(release_platform, request(release_platform), notes="historical notes")
    if change == "task_revision":
        tasks.update_task(task["task_id"], expected_revision=1, config={"rules": {"confidence_threshold": 0.8}},
                          requested_by="requester", idempotency_key="edit-task")
    elif change == "deployment_revision":
        with db.get_session() as session:
            session.get(TaskDeploymentRecord, task["task_id"]).deployment_revision = 1
            session.commit()
    else:
        (tasks._models_root / "opaque.onnx").write_bytes(b"different opaque bytes")
    current = service.get(document["release_id"])
    assert current["state"] == "stale" and current["review_effective"] is False
    assert current["review_history"][0]["notes"] == "historical notes"
    assert "RELEASE_BINDING_CHANGED" in current["invalidation_reasons"]
    # A lost ACK reconciles the original receipt instead of creating another dossier.
    assert request(release_platform)["release_id"] == document["release_id"]


def test_same_idempotency_key_concurrency_has_one_dossier_and_rejects_changed_actor(release_platform):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _index: request(release_platform), range(8)))
    assert all(document == results[0] for document in results)
    with release_platform[0].get_session() as session:
        assert session.scalar(select(func.count()).select_from(TaskReleaseRecord)) == 1
    with pytest.raises(AnalysisTaskConflictError, match="different request"):
        request(release_platform, actor="other-person")


@pytest.mark.parametrize("committed", [False, True])
def test_request_commit_failure_and_lost_ack_retry(release_platform, monkeypatch, committed):
    commit = Session.commit
    def fail(session):
        if committed:
            commit(session)
        raise RuntimeError("simulated commit acknowledgement loss")
    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", fail)
        with pytest.raises(RuntimeError):
            request(release_platform)
    document = request(release_platform)
    assert document["revision"] == 1
    with release_platform[0].get_session() as session:
        assert session.scalar(select(func.count()).select_from(TaskReleaseRecord)) == 1


def test_review_lost_ack_reconciles_even_after_evidence_changes(release_platform, monkeypatch):
    document = request(release_platform)
    commit = Session.commit
    def lose_ack(session):
        commit(session)
        raise RuntimeError("simulated review commit acknowledgement loss")
    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", lose_ack)
        with pytest.raises(RuntimeError):
            review(release_platform, document)
    (release_platform[5] / "evaluation.json").write_bytes(b"later change")
    retried = review(release_platform, document)
    assert retried["revision"] == 2 and len(retried["review_history"]) == 1
    assert release_platform[4].get(document["release_id"])["review_effective"] is False


def test_review_receipt_cannot_be_replayed_as_another_person_or_review(release_platform):
    document = request(release_platform)
    approved = review(release_platform, document)
    assert review(release_platform, document) == approved
    with pytest.raises(AnalysisTaskConflictError, match="different request"):
        review(release_platform, document, actor="other-person")
    with pytest.raises(AnalysisTaskConflictError, match="different request"):
        review(release_platform, document, notes="changed review notes")
    with pytest.raises(AnalysisTaskConflictError, match="revision changed"):
        review(release_platform, document, key="new-key-old-review")
    assert len(release_platform[4].get(document["release_id"])["review_history"]) == 1


def test_review_rechecks_evidence_after_preflight(release_platform, monkeypatch):
    service = release_platform[4]
    document = request(release_platform)
    freshness = service._freshness
    def race(value):
        result = freshness(value)
        (release_platform[5] / "evaluation.json").write_bytes(b"changed after preflight")
        return result
    monkeypatch.setattr(service, "_freshness", race)
    with pytest.raises(TaskReleaseBlocked, match="IDENTITY_MISMATCH"):
        review(release_platform, document)
    assert service.get(document["release_id"])["review"] is None


def test_revoke_retains_review_and_is_idempotent(release_platform):
    service = release_platform[4]
    document = review(release_platform, request(release_platform))
    args = dict(expected_revision=2, notes="withdraw pending review", actor="requester", idempotency_key="revoke-release")
    revoked = service.revoke(document["release_id"], **args)
    assert revoked == service.revoke(document["release_id"], **args)
    current = service.get(document["release_id"])
    assert current["state"] == "revoked" and current["review_effective"] is False
    assert current["review_history"][0]["notes"] == "checked"
    with pytest.raises(TaskReleaseBlocked, match="IMMUTABLE"):
        service.revise_evidence(document["release_id"], expected_revision=3,
            evidence_sources=release_platform[6], actor="requester", idempotency_key="edit-revoked")


@pytest.mark.parametrize("path", ["../outside.json", "/outside.json", "C:/outside.json", "a\\b.json", "a:stream", "a//b.json"])
def test_path_escape_and_ambiguous_paths_are_rejected(release_platform, path):
    item = dict(release_platform[6][0], relative_path=path)
    with pytest.raises(AnalysisTaskValidationError, match="relative"):
        request(release_platform, sources=[item])


def test_unconfigured_root_and_missing_files_fail_closed(release_platform):
    service = release_platform[4]
    service._evidence_root = None
    with pytest.raises(TaskReleaseBlocked, match="ROOT_UNAVAILABLE"):
        request(release_platform)
    service._evidence_root = release_platform[5]
    (release_platform[5] / "evaluation.json").unlink()
    with pytest.raises(TaskReleaseBlocked, match="IDENTITY_MISMATCH"):
        request(release_platform)


def test_corrupt_persisted_dossier_is_not_treated_as_approved(release_platform):
    document = request(release_platform)
    with release_platform[0].get_session() as session:
        row = session.get(TaskReleaseRecord, document["release_id"])
        row.document_json = row.document_json.replace('"requested"', '"review_approved"')
        session.commit()
    with pytest.raises(AnalysisTaskIntegrityError):
        release_platform[4].get(document["release_id"])


def test_other_templates_and_wrong_task_dossier_are_not_release_shortcuts(release_platform):
    _db, tasks, _task, _bundle, service, _root, _sources = release_platform
    other = tasks.create_task("camera-a", "fast_motion.v1", "Motion", {},
                             requested_by="requester", idempotency_key="create-motion")
    bundle = tasks.freeze_bundle(other["task_id"], expected_revision=1, model_ids={},
                                  requested_by="requester", idempotency_key="freeze-motion")
    with pytest.raises(TaskReleaseBlocked, match="ONLY_ACTIVE_OBJECT"):
        service.request(other["task_id"], bundle_id=bundle["bundle_id"], expected_task_revision=1,
            expected_deployment_revision=0, evidence_sources=[], actor="requester", idempotency_key="motion-release")
    document = request(release_platform)
    with pytest.raises(AnalysisTaskValidationError, match="does not belong"):
        service.readiness(other["task_id"], document["release_id"])


def test_release_dtos_reject_worker_or_client_production_authorization_fields():
    with pytest.raises(ValueError):
        TaskReleaseRequest(bundle_id="bundle-" + "a" * 32, expected_task_revision=1,
                           expected_deployment_revision=0, production_allowed=True)
    with pytest.raises(ValueError):
        TaskReleaseReviewRequest(expected_revision=1, expected_subject_sha256="a" * 64,
                                 decision="approve", actor="forged-reviewer")


@pytest.mark.parametrize("revision", [True, 1.0, "1"])
def test_release_http_revisions_do_not_coerce_ambiguous_types(revision):
    with pytest.raises(ValueError):
        TaskReleaseRequest(bundle_id="bundle-" + "a" * 32, expected_task_revision=revision,
                           expected_deployment_revision=0)
    with pytest.raises(ValueError):
        TaskReleaseReviewRequest(expected_revision=revision, expected_subject_sha256="a" * 64,
                                 decision="approve")
