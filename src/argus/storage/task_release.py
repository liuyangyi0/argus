"""Task release dossiers, independent review and an explicit production NO-GO.

A dossier review is not an approved template policy, an evidence attestation,
or a runtime application receipt. Managed execution remains shadow-only. This
module deliberately has no method that turns a user-supplied report, Boolean
or reviewer into production authorization.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from uuid import uuid4

from sqlalchemy import select, text

from argus.storage.analysis_tasks import (
    AnalysisTaskConflictError, AnalysisTaskIntegrityError, AnalysisTaskService,
    AnalysisTaskValidationError,
)
from argus.storage.models import (
    AnalysisTaskRecord, AnalysisTaskRevisionRecord, TaskDeploymentRecord,
    TaskExecutionBundleRecord, TaskReleaseRecord,
)

SCHEMA = "argus.object-detection-release-dossier.v1"
POLICY_ID = "argus.object-detection-production.v1"
REQUIRED_EVIDENCE_ROLES = frozenset({
    "evaluation", "calibration", "baseline", "corpus", "soak", "soak_telemetry",
})
_SHA = re.compile(r"[0-9a-f]{64}")
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc)


def _timestamp():
    return _now().isoformat()


class TaskReleaseBlocked(AnalysisTaskConflictError):
    """The request cannot be interpreted as permission to publish."""


class TaskReleaseService:
    """Durable object-detection release requests without a new publish bypass."""

    def __init__(self, db, task_service, *, evidence_root: str | Path | None = None):
        self._db, self._tasks = db, task_service
        self._evidence_root = Path(evidence_root).absolute() if evidence_root is not None else None

    @staticmethod
    def _actor(actor):
        actor = AnalysisTaskService._actor(actor)
        # Human authorization is established by the authenticated route and
        # active DB account, not by guessing from the account's name.
        if actor.lower() == "unknown":
            raise AnalysisTaskValidationError("an authenticated personal actor is required")
        return actor

    @staticmethod
    def _begin(session):
        if session.get_bind().dialect.name != "sqlite":
            raise RuntimeError("task release dossiers require serialized SQLite writes")
        session.execute(text("BEGIN IMMEDIATE"))

    @staticmethod
    def _sources(value):
        if not isinstance(value, list) or len(value) > len(REQUIRED_EVIDENCE_ROLES):
            raise AnalysisTaskValidationError("evidence_sources must be a bounded list")
        result, roles = [], set()
        for item in value:
            if not isinstance(item, dict) or set(item) != {"role", "relative_path", "sha256"}:
                raise AnalysisTaskValidationError("evidence sources require role, relative_path and sha256")
            role, name, digest = item["role"], item["relative_path"], item["sha256"]
            if not isinstance(role, str) or role not in REQUIRED_EVIDENCE_ROLES or role in roles:
                raise AnalysisTaskValidationError("evidence roles must be declared and unique")
            if (not isinstance(name, str) or not 1 <= len(name) <= 500
                    or any(ord(char) < 32 for char in name) or "\\" in name or ":" in name
                    or PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute()
                    or any(part in {"", ".", ".."} for part in name.split("/"))):
                raise AnalysisTaskValidationError("evidence must use a controlled relative file path")
            if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
                raise AnalysisTaskValidationError("evidence source sha256 is invalid")
            result.append({"role": role, "relative_path": name, "sha256": digest})
            roles.add(role)
        return sorted(result, key=lambda item: item["role"])

    def _check_sources(self, sources):
        """Verify file identity only; this never validates the report's claims."""
        if not sources:
            return
        if self._evidence_root is None:
            raise TaskReleaseBlocked("EVIDENCE_ROOT_UNAVAILABLE")
        try:
            root = self._evidence_root.resolve(strict=True)
            if self._evidence_root.is_symlink():
                raise ValueError("linked evidence root")
            for source in sources:
                path = root / source["relative_path"]
                for part in (path, *path.parents):
                    if part == root:
                        break
                    info = part.lstat()
                    if part.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                        raise ValueError("linked evidence")
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    raise ValueError("invalid evidence path")
                digest, size = hashlib.sha256(), 0
                with resolved.open("rb") as handle:
                    while chunk := handle.read(65536):
                        size += len(chunk)
                        if size > _MAX_EVIDENCE_BYTES:
                            raise ValueError("evidence size limit")
                        digest.update(chunk)
                if size == 0 or digest.hexdigest() != source["sha256"]:
                    raise ValueError("evidence identity changed")
        except (OSError, ValueError) as exc:
            raise TaskReleaseBlocked("EVIDENCE_SOURCE_IDENTITY_MISMATCH") from exc

    def _binding(self, task_id, bundle_id, task_revision, deployment_revision):
        if (type(task_revision) is not int or task_revision < 1
                or type(deployment_revision) is not int or deployment_revision < 0):
            raise AnalysisTaskValidationError("release revisions must be exact integers")
        task = self._tasks.get_task(task_id)
        if task is None:
            raise LookupError("task not found")
        if (task["authority"] != "managed" or task["template_id"] != "object_detection.v1"
                or task["archived"]):
            raise TaskReleaseBlocked("ONLY_ACTIVE_OBJECT_DETECTION_SUPPORTED")
        if task["current_revision"] != task_revision:
            raise AnalysisTaskConflictError("task revision changed")
        bundle = self._tasks.get_bundle(bundle_id)
        if bundle is None or bundle["task_id"] != task_id:
            raise AnalysisTaskValidationError("bundle does not belong to this task")
        manifest = self._tasks.resolve_bundle(task_id, bundle_id)
        if manifest["task_revision"] != task_revision:
            raise AnalysisTaskConflictError("bundle task revision changed")
        with self._db.get_session() as session:
            deployment = session.get(TaskDeploymentRecord, task_id)
            if deployment is None or deployment.deployment_revision != deployment_revision:
                raise AnalysisTaskConflictError("deployment revision changed")
        return {
            "task_id": task_id, "camera_id": task["camera_id"], "template_id": task["template_id"],
            "task_revision": task_revision, "task_config_sha256": task["config_sha256"],
            "bundle_id": bundle_id, "bundle_manifest_sha256": bundle["manifest_sha256"],
            "deployment_revision": deployment_revision,
            "config": task["config"], "models": bundle["manifest"]["models"],
        }

    @staticmethod
    def _assert_binding(session, binding):
        task = session.get(AnalysisTaskRecord, binding["task_id"])
        revision = session.scalar(select(AnalysisTaskRevisionRecord).where(
            AnalysisTaskRevisionRecord.task_id == binding["task_id"],
            AnalysisTaskRevisionRecord.revision == binding["task_revision"],
        ))
        deployment = session.get(TaskDeploymentRecord, binding["task_id"])
        bundle = session.get(TaskExecutionBundleRecord, binding["bundle_id"])
        if (task is None or task.archived or task.authority != "managed"
                or task.template_id != "object_detection.v1" or task.camera_id != binding["camera_id"]
                or task.current_revision != binding["task_revision"] or revision is None
                or revision.config_sha256 != binding["task_config_sha256"]
                or deployment is None or deployment.authority != "managed"
                or deployment.deployment_revision != binding["deployment_revision"]
                or bundle is None or bundle.task_id != binding["task_id"]
                or bundle.manifest_sha256 != binding["bundle_manifest_sha256"]):
            raise AnalysisTaskConflictError("release task, bundle or deployment binding changed")

    @staticmethod
    def _decode(row):
        try:
            document = json.loads(row.document_json)
            if (_canonical(document) != row.document_json or _sha(document) != row.document_sha256
                    or set(document) - {"revocation", "evidence_authors"} != {
                        "schema_version", "release_id", "revision", "binding", "evidence_sources",
                        "policy_id", "subject_sha256", "state", "review", "review_history",
                        "created_by", "created_at", "updated_at",
                    }
                    or document["schema_version"] != SCHEMA or document["release_id"] != row.release_id
                    or document["binding"]["task_id"] != row.task_id
                    or document["revision"] != row.revision or document["created_by"] != row.created_by
                    or document["policy_id"] != POLICY_ID
                    or document["state"] not in {"requested", "review_approved", "rejected", "revoked"}
                    or TaskReleaseService._sources(document["evidence_sources"]) != document["evidence_sources"]
                    or not isinstance(document["review_history"], list)
                    or document["subject_sha256"] != _sha({
                        "binding": document["binding"], "evidence_sources": document["evidence_sources"],
                        "policy_id": POLICY_ID,
                    })):
                raise ValueError("inconsistent release dossier")
            authors = document.get("evidence_authors")
            if authors is not None and (
                not isinstance(authors, list) or not authors
                or any(TaskReleaseService._actor(author) != author for author in authors)
                or len(set(authors)) != len(authors) or document["created_by"] not in authors
            ):
                raise ValueError("inconsistent release evidence authors")
            review = document["review"]
            if review is not None and (
                not isinstance(review, dict) or review not in document["review_history"]
                or review.get("decision") not in {"approve", "reject"}
                or review.get("actor") == document["created_by"]
                or authors is not None and review.get("actor") in authors
                or review.get("subject_sha256") != document["subject_sha256"]
                or review.get("scope") != "dossier_review_only_not_production_authorization"
            ):
                raise ValueError("inconsistent release review")
            if document["state"] == "review_approved" and (review is None or review["decision"] != "approve"):
                raise ValueError("release review is missing")
            return document
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisTaskIntegrityError("task release dossier integrity check failed") from exc

    @staticmethod
    def _save(row, document):
        row.revision = document["revision"]
        row.document_json, row.document_sha256 = _canonical(document), _sha(document)
        row.updated_at = _now()

    @staticmethod
    def _public(document):
        result = json.loads(_canonical(document))
        result["task_id"] = document["binding"]["task_id"]
        result["bundle_id"] = document["binding"]["bundle_id"]
        result["evidence_sources"] = [{"role": source["role"], "sha256": source["sha256"]}
                                      for source in document["evidence_sources"]]
        result["production_allowed"] = False
        result["approval_scope"] = "dossier_review_only_not_production_authorization"
        return result

    def request(self, task_id, *, bundle_id, expected_task_revision, expected_deployment_revision,
                evidence_sources, actor, idempotency_key):
        actor, sources = self._actor(actor), self._sources(evidence_sources)
        request = {"action": "request_task_release", "task_id": task_id, "bundle_id": bundle_id,
                   "expected_task_revision": expected_task_revision,
                   "expected_deployment_revision": expected_deployment_revision,
                   "evidence_sources": sources, "actor": actor}
        with self._db.get_session() as session:
            receipt = self._tasks._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
        binding = self._binding(task_id, bundle_id, expected_task_revision, expected_deployment_revision)
        self._check_sources(sources)
        with self._db.get_session() as session:
            self._begin(session)
            receipt = self._tasks._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            self._assert_binding(session, binding)
            release_id = f"release-{uuid4().hex}"
            subject = {"binding": binding, "evidence_sources": sources, "policy_id": POLICY_ID}
            document = {"schema_version": SCHEMA, "release_id": release_id, "revision": 1,
                        **subject, "subject_sha256": _sha(subject), "state": "requested",
                        "review": None, "review_history": [], "created_by": actor,
                        "evidence_authors": [actor],
                        "created_at": _timestamp(), "updated_at": _timestamp()}
            row = TaskReleaseRecord(release_id=release_id, task_id=task_id, created_by=actor)
            self._save(row, document)
            session.add(row)
            result = self._public(document)
            self._tasks._save_receipt(session, idempotency_key, request, result,
                action=request["action"], subject_id=release_id, actor=actor)
            session.commit()
            return result

    def _load(self, session, release_id, expected_revision=None):
        row = session.get(TaskReleaseRecord, release_id)
        if row is None:
            raise LookupError("release dossier not found")
        document = self._decode(row)
        if expected_revision is not None and (type(expected_revision) is not int
                                              or expected_revision != row.revision):
            raise AnalysisTaskConflictError("release dossier revision changed")
        return row, document

    def _freshness(self, document):
        reasons = []
        if not document.get("evidence_authors"):
            # Historical dossiers do not prove who supplied revised evidence.
            # Keep their notes readable, but never infer an independent review.
            reasons.append("EVIDENCE_AUTHOR_IDENTITY_UNAVAILABLE")
        binding = document["binding"]
        try:
            current = self._binding(binding["task_id"], binding["bundle_id"],
                                    binding["task_revision"], binding["deployment_revision"])
            if current != binding:
                reasons.append("RELEASE_BINDING_CHANGED")
        except (LookupError, AnalysisTaskConflictError, AnalysisTaskIntegrityError, ValueError, OSError):
            reasons.append("RELEASE_BINDING_CHANGED")
        try:
            self._check_sources(document["evidence_sources"])
        except TaskReleaseBlocked:
            reasons.append("EVIDENCE_SOURCE_IDENTITY_MISMATCH")
        return reasons

    def get(self, release_id):
        with self._db.get_session() as session:
            _row, document = self._load(session, release_id)
        result = self._public(document)
        invalidation = self._freshness(document)
        result["stored_state"] = result["state"]
        if invalidation and result["state"] not in {"revoked", "rejected"}:
            result["state"] = "stale"
        result["review_effective"] = bool(document["review"] and document["review"]["decision"] == "approve"
                                          and document["state"] == "review_approved" and not invalidation)
        result["invalidation_reasons"] = invalidation
        return result

    def list(self, task_id):
        if self._tasks.get_task(task_id) is None:
            raise LookupError("task not found")
        with self._db.get_session() as session:
            ids = list(session.scalars(select(TaskReleaseRecord.release_id).where(
                TaskReleaseRecord.task_id == task_id,
            ).order_by(TaskReleaseRecord.created_at.desc(), TaskReleaseRecord.release_id).limit(100)))
        return [self.get(release_id) for release_id in ids]

    def revise_evidence(self, release_id, *, expected_revision, evidence_sources, actor, idempotency_key):
        actor, sources = self._actor(actor), self._sources(evidence_sources)
        request = {"action": "revise_task_release", "release_id": release_id,
                   "expected_revision": expected_revision, "evidence_sources": sources, "actor": actor}
        with self._db.get_session() as session:
            receipt = self._tasks._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
        self._check_sources(sources)
        with self._db.get_session() as session:
            self._begin(session)
            receipt = self._tasks._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            row, document = self._load(session, release_id, expected_revision)
            if document["state"] == "revoked":
                raise TaskReleaseBlocked("REVOKED_RELEASE_IS_IMMUTABLE")
            self._assert_binding(session, document["binding"])
            self._check_sources(sources)
            document["evidence_sources"] = sources
            document["evidence_authors"] = sorted(set(
                document.get("evidence_authors") or [document["created_by"]]
            ) | {actor})
            document["subject_sha256"] = _sha({"binding": document["binding"],
                "evidence_sources": sources, "policy_id": POLICY_ID})
            document["review"] = None
            document["state"], document["revision"] = "requested", document["revision"] + 1
            document["updated_at"] = _timestamp()
            self._save(row, document)
            result = self._public(document)
            self._tasks._save_receipt(session, idempotency_key, request, result,
                action=request["action"], subject_id=release_id, actor=actor)
            session.commit()
            return result

    def review(self, release_id, *, expected_revision, expected_subject_sha256, decision, notes,
               actor, idempotency_key):
        actor = self._actor(actor)
        if decision not in {"approve", "reject"} or not isinstance(notes, str) or len(notes) > 4000:
            raise AnalysisTaskValidationError("invalid dossier review")
        request = {"action": "review_task_release", "release_id": release_id,
                   "expected_revision": expected_revision, "expected_subject_sha256": expected_subject_sha256,
                   "decision": decision, "notes": notes, "actor": actor}
        with self._db.get_session() as session:
            receipt = self._tasks._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            _row, before = self._load(session, release_id, expected_revision)
        if decision == "approve":
            if {item["role"] for item in before["evidence_sources"]} != REQUIRED_EVIDENCE_ROLES:
                raise TaskReleaseBlocked("RELEASE_EVIDENCE_INCOMPLETE")
            if self._freshness(before):
                raise TaskReleaseBlocked("RELEASE_REVIEW_SUBJECT_STALE")
        with self._db.get_session() as session:
            self._begin(session)
            receipt = self._tasks._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            row, document = self._load(session, release_id, expected_revision)
            if document["state"] not in {"requested", "rejected"}:
                raise TaskReleaseBlocked("RELEASE_REVIEW_STATE_INVALID")
            if document["subject_sha256"] != expected_subject_sha256:
                raise AnalysisTaskConflictError("reviewed release subject changed")
            if not document.get("evidence_authors"):
                raise TaskReleaseBlocked("EVIDENCE_AUTHOR_IDENTITY_UNAVAILABLE")
            if actor in document["evidence_authors"]:
                raise TaskReleaseBlocked("INDEPENDENT_RELEASE_REVIEW_REQUIRED")
            if decision == "approve":
                self._assert_binding(session, document["binding"])
                self._check_sources(document["evidence_sources"])
            review = {"decision": decision, "notes": notes, "actor": actor,
                      "subject_sha256": document["subject_sha256"], "reviewed_at": _timestamp(),
                      "scope": "dossier_review_only_not_production_authorization"}
            document["review"] = review
            document["review_history"].append(review)
            document["state"] = "review_approved" if decision == "approve" else "rejected"
            document["revision"] += 1
            document["updated_at"] = _timestamp()
            self._save(row, document)
            result = self._public(document)
            self._tasks._save_receipt(session, idempotency_key, request, result,
                action=request["action"], subject_id=release_id, actor=actor)
            session.commit()
            return result

    def revoke(self, release_id, *, expected_revision, notes, actor, idempotency_key):
        actor = self._actor(actor)
        if not isinstance(notes, str) or not 1 <= len(notes.strip()) <= 4000:
            raise AnalysisTaskValidationError("revocation requires a reason")
        request = {"action": "revoke_task_release", "release_id": release_id,
                   "expected_revision": expected_revision, "notes": notes, "actor": actor}
        with self._db.get_session() as session:
            self._begin(session)
            receipt = self._tasks._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            row, document = self._load(session, release_id, expected_revision)
            if document["state"] == "revoked":
                raise TaskReleaseBlocked("RELEASE_ALREADY_REVOKED")
            document["state"] = "revoked"
            document["revocation"] = {"actor": actor, "notes": notes, "revoked_at": _timestamp()}
            document["revision"] += 1
            document["updated_at"] = _timestamp()
            self._save(row, document)
            result = self._public(document)
            self._tasks._save_receipt(session, idempotency_key, request, result,
                action=request["action"], subject_id=release_id, actor=actor)
            session.commit()
            return result

    def readiness(self, task_id, release_id=None):
        task = self._tasks.get_task(task_id)
        if task is None:
            raise LookupError("task not found")
        document = self.get(release_id) if release_id else None
        if document is not None and document["binding"]["task_id"] != task_id:
            raise AnalysisTaskValidationError("release does not belong to this task")
        reasons = ["TEMPLATE_POLICY_NOT_APPROVED", "PRODUCTION_APPLY_UNSUPPORTED",
                   "TRUSTED_TASK_EVIDENCE_VERIFIER_UNAVAILABLE"]
        if task["authority"] != "managed" or task["template_id"] != "object_detection.v1":
            reasons.append("TEMPLATE_NOT_SUPPORTED")
        if document is None:
            reasons.append("RELEASE_DOSSIER_REQUIRED")
        else:
            if not document["review_effective"]:
                reasons.append("INDEPENDENT_RELEASE_REVIEW_REQUIRED")
            reasons.extend(document["invalidation_reasons"])
            if document["state"] == "revoked":
                reasons.append("RELEASE_REVOKED")
            if {item["role"] for item in document["evidence_sources"]} != REQUIRED_EVIDENCE_ROLES:
                reasons.append("RELEASE_EVIDENCE_INCOMPLETE")
        return {"task_id": task_id, "release_id": release_id, "policy_id": POLICY_ID,
                "production_supported": False, "production_ready": False, "production_allowed": False,
                "decision": "NO-GO", "authority": task["authority"], "reasons": list(dict.fromkeys(reasons)),
                "review_scope": "dossier_review_only_not_production_authorization"}

    def authorize_event(self, task_id, *, bundle_id, deployment_revision, execution_id,
                        source_session_id, source_epoch, session=None):
        """Fast fail-closed boundary, safe to call inside the event transaction.

        No approved template contract or production runtime exists yet. Do not
        read a worker's ``production_allowed`` flag, perform slow file hashing
        on the source thread, or turn a dossier review into an applied release.
        """
        return {"authorized": False, "production_allowed": False, "release_id": None,
                "reasons": ["TEMPLATE_POLICY_NOT_APPROVED", "PRODUCTION_APPLY_UNSUPPORTED"]}
