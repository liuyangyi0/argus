"""Scene-owned, immutable inputs for the Dashboard training/release workflow.

This service creates no model release authority. A reviewed reference, strict
baseline, successful calibration, profile approval and production release are
separate steps. Only ModelReleaseService may change applied model stages.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterator

from argus.config.schema import SceneGuardConfig as SceneGuardPolicy
from argus.dashboard.tasks import TaskSubmissionRejected
from argus.preprocessing.scene_guard import SceneGuardConfig
from argus.preprocessing.scene_guard_policy import runtime_scene_guard_config
from argus.preprocessing.scene_profile import (
    RUNTIME_ARTIFACT_HASH_SCHEME,
    SceneProfileStore,
    runtime_artifact_sha256,
)
from argus.storage.models import AuditLog, TrainingJobRecord, TrainingJobStatus
from argus.storage.release_evidence import (
    evidence_artifact_sha256,
    registry_training_data_hash,
)

WORKFLOW_SCHEMA = "argus.scene-workflow.v1"
BASELINE_SCHEMA = "argus.scene-profile-baseline.v3"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 16 * 1024 * 1024


class SceneWorkflowConflict(RuntimeError):
    """An existing immutable input or operation requires reconciliation."""


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("invalid camera/workflow identifier")
    return value


def _reject_links(path: Path) -> Path:
    """Reject lexical symlink/reparse components before resolving a path."""
    absolute = Path(os.path.abspath(path))
    for item in (*reversed(absolute.parents), absolute):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError("scene workflow paths must not contain links/reparse points")
    return absolute.resolve()


def _relative_file(root: Path, value: str, suffixes: set[str]) -> Path:
    if (
        not isinstance(value, str) or not value or len(value) > 512
        or "\\" in value or ":" in value or "\x00" in value
    ):
        raise ValueError("input must be a controlled relative file path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {".", ".."} for part in value.split("/")):
        raise ValueError("input path must stay inside its controlled root")
    target = _reject_links(root / relative)
    if not target.is_relative_to(root) or target.suffix.lower() not in suffixes:
        raise ValueError("input path is outside its controlled root or has an invalid type")
    if not target.is_file():
        raise FileNotFoundError("controlled source file does not exist")
    return target


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _reject_links(path)
    with path.open("rb") as handle:
        raw = handle.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("workflow metadata exceeds the controlled budget")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("workflow metadata must be an object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _reject_links(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SceneWorkflowService:
    """One controlled bridge between recorded sessions and model candidates."""

    def __init__(self, *, baselines_dir: Path, profiles_dir: Path, database, registry,
                 scene_guard_policy: SceneGuardPolicy | None = None):
        self.baselines_dir = _reject_links(baselines_dir)
        self.profiles_dir = _reject_links(profiles_dir)
        self.root = _reject_links(self.baselines_dir.parent / "scene_workflows")
        self.sources_root = _reject_links(self.baselines_dir.parent / "regression")
        self.database = database
        self.registry = registry
        self.policy = SceneGuardPolicy.model_validate(
            (scene_guard_policy or SceneGuardPolicy()).model_dump(mode="json"),
        )
        self.scene_guard = asdict(runtime_scene_guard_config(self.policy))

    def _assert_policy(self, record: dict[str, Any] | None = None) -> None:
        if (
            self.policy.enabled is not True or self.policy.allow_alignable is not False
            or self.policy.require_profile_model_binding is not True
            or self.policy.require_operator_approved_profile is not True
            or self.policy.fallback_to_baseline is not False
        ):
            raise ValueError("scene workflow requires the existing strict SceneGuard deployment policy")
        if record is not None and record.get("scene_guard") != self.scene_guard:
            raise SceneWorkflowConflict("camera scene policy changed; use a new workflow, do not reuse old thresholds")

    def _directory(self, camera_id: str, workflow_id: str) -> Path:
        return _reject_links(self.root / _identifier(camera_id) / _identifier(workflow_id))

    def _baseline(self, camera_id: str, workflow_id: str) -> Path:
        return _reject_links(
            self.baselines_dir / _identifier(camera_id) / "default" / f"scene-{_identifier(workflow_id)}"
        )

    def _profile_store(self, camera_id: str, profile_id: str | None = None) -> SceneProfileStore:
        _reject_links(self.profiles_dir)
        _reject_links(self.profiles_dir / _identifier(camera_id))
        _reject_links(self.profiles_dir / camera_id / "active.json")
        if profile_id is not None:
            _reject_links(self.profiles_dir / camera_id / _identifier(profile_id))
        return SceneProfileStore(self.profiles_dir)

    def _active_selection(self, camera_id: str, store: SceneProfileStore) -> tuple[str | None, str | None]:
        """Bind an observed profile ID to the exact pointer, including ABA edits."""
        marker = _reject_links(self.profiles_dir / _identifier(camera_id) / "active.json")
        before = _file_sha256(marker) if marker.exists() else None
        profile_id = store.get_active_profile_id(camera_id)
        after = _file_sha256(marker) if marker.exists() else None
        if before != after:
            raise SceneWorkflowConflict("active scene changed while reading; refresh before confirming")
        return profile_id, after

    def _load(self, camera_id: str, workflow_id: str) -> dict[str, Any]:
        record = _read_json(self._directory(camera_id, workflow_id) / "workflow.json")
        if (
            record.get("schema_version") != WORKFLOW_SCHEMA
            or record.get("camera_id") != camera_id
            or record.get("workflow_id") != workflow_id
            or record.get("scene_profile_id") != workflow_id
        ):
            raise ValueError("scene workflow identity is invalid")
        return record

    def _save(self, record: dict[str, Any]) -> None:
        _write_json(self._directory(record["camera_id"], record["workflow_id"]) / "workflow.json", record)

    @contextmanager
    def _lock(self, camera_id: str, workflow_id: str, *, wait: bool = False) -> Iterator[None]:
        path = self._directory(camera_id, workflow_id) / ".operation.lock"
        deadline = time.monotonic() + (5.0 if wait else 0.0)
        while True:
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError as exc:
                if time.monotonic() >= deadline:
                    raise SceneWorkflowConflict("workflow is busy; a stale lock requires administrator review") from exc
                # Only wait for a normal concurrent metadata write. Never
                # expire/delete/take ownership of someone else's lock.
                time.sleep(0.01)
        try:
            os.close(descriptor)
            yield
        finally:
            path.unlink()

    @contextmanager
    def _activation_lock(self, camera_id: str) -> Iterator[None]:
        path = _reject_links(self.root / _identifier(camera_id)) / ".activation.lock"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise SceneWorkflowConflict("camera scene selection is busy or requires administrator reconciliation") from exc
        try:
            os.close(descriptor)
            yield
        finally:
            path.unlink()

    def _audit(self, camera_id: str, workflow_id: str, actor: str, action: str) -> None:
        if (
            not isinstance(actor, str) or not actor.strip()
            or actor.strip().lower() == "unknown" or self.database is None
        ):
            raise ValueError("an authenticated operator and audit database are required")
        with self.database.get_session() as session:
            session.add(AuditLog(
                user=actor, action=action, target_type="scene_workflow",
                target_id=f"{camera_id}/{workflow_id}", detail="Explicit scene workflow operation requested",
            ))
            session.commit()

    def create(
        self, camera_id: str, workflow_id: str, *, reference_path: str,
        reference_sha256: str, normal_reference_confirmed: bool, actor: str,
    ) -> dict[str, Any]:
        _identifier(camera_id)
        _identifier(workflow_id)
        self._assert_policy()
        if normal_reference_confirmed is not True or not _SHA256.fullmatch(reference_sha256):
            raise ValueError("explicit normal-reference confirmation and SHA256 are required")
        reference = _relative_file(
            _reject_links(self.baselines_dir / camera_id), reference_path, {".jpg", ".jpeg", ".png"},
        )
        if _file_sha256(reference) != reference_sha256:
            raise SceneWorkflowConflict("selected reference changed; refresh before confirming")
        if _reject_links(self.profiles_dir / camera_id / workflow_id).exists():
            raise SceneWorkflowConflict("runtime profile ID already exists; use a new workflow ID")
        self._audit(camera_id, workflow_id, actor, "scene_workflow.create.requested")
        now = datetime.now(timezone.utc).isoformat()
        draft = SceneProfileStore(self.root).create_from_baseline(
            camera_id, workflow_id, reference,
            metadata={
                "operator_approved": False, "validation_only": True,
                "created_by": actor, "normal_reference_confirmed_by": actor,
                "normal_reference_confirmed_at": now,
            },
            activate=False,
        )
        # Check the bytes actually consumed, not only the earlier path read.
        if draft.source_baseline_sha256 != reference_sha256:
            raise SceneWorkflowConflict("selected reference changed during draft creation; do not reuse this ID")
        record = {
            "schema_version": WORKFLOW_SCHEMA, "camera_id": camera_id,
            "workflow_id": workflow_id, "scene_profile_id": workflow_id,
            "created_by": actor, "created_at": now, "state": "draft",
            "reference_sha256": draft.reference_sha256,
            "scene_guard": dict(self.scene_guard),
            "baseline_sha256": None, "manifest_sha256": None,
            "baseline_task_id": None, "training_job_id": None,
            "approved": False, "error": None,
        }
        self._save(record)
        return self.describe(record)

    def start_baseline(self, camera_id: str, workflow_id: str, *, sources: list[dict[str, str]],
                       sample_every: int, max_images: int, actor: str, task_manager) -> str:
        videos, annotations = [], []
        for source in sources:
            videos.append(_relative_file(self.sources_root, source["video"], {".mp4", ".avi", ".mkv", ".mov"}))
            annotations.append(_relative_file(self.sources_root, source["annotations"], {".json"}))
        if len(videos) < 2 or len(set(videos)) != len(videos):
            raise ValueError("at least two distinct reviewed normal recording sessions are required")
        with self._lock(camera_id, workflow_id):
            record = self._load(camera_id, workflow_id)
            self._assert_policy(record)
            if record.get("state") not in {"draft", "baseline_failed"}:
                raise SceneWorkflowConflict("baseline is already submitted or frozen")
            if self._baseline(camera_id, workflow_id).exists():
                raise SceneWorkflowConflict("baseline output already exists; reconcile it or create a new workflow")
            self._audit(camera_id, workflow_id, actor, "scene_workflow.baseline.requested")
            record.update(state="building_baseline", error=None, baseline_task_id=None)
            self._save(record)
        try:
            task_id = task_manager.submit(
                "scene_baseline", partial(self._build_baseline, camera_id=camera_id), camera_id=camera_id,
                workflow_id=workflow_id, videos=videos, annotations=annotations,
                sample_every=sample_every, max_images=max_images,
            )
        except TaskSubmissionRejected as exc:
            with self._lock(camera_id, workflow_id, wait=True):
                record = self._load(camera_id, workflow_id)
                # Only the manager's preflight rejection proves that no worker
                # owns this reservation. Restore retryability, not readiness.
                record.update(
                    state="baseline_failed", baseline_task_id=None,
                    error="Baseline task was not started; retry after task capacity is available",
                )
                self._save(record)
            raise SceneWorkflowConflict(str(exc)) from exc
        except Exception:
            with self._lock(camera_id, workflow_id, wait=True):
                record = self._load(camera_id, workflow_id)
                # Task submission has no transactional handshake with its
                # worker. An exception cannot prove that a worker never started.
                # Keep the reservation rather than admit a concurrent rebuild.
                record.update(error="Baseline task submission unconfirmed; inspect task state before retry")
                self._save(record)
            raise
        with self._lock(camera_id, workflow_id, wait=True):
            record = self._load(camera_id, workflow_id)
            record["baseline_task_id"] = task_id
            self._save(record)
        return task_id

    def _build_baseline(self, progress_callback, *, camera_id: str, workflow_id: str,
                        videos: list[Path], annotations: list[Path], sample_every: int,
                        max_images: int) -> dict[str, Any]:
        from argus.validation.scene_profile_baseline import build_scene_profile_baseline

        try:
            self._directory(camera_id, workflow_id)
            draft = SceneProfileStore(self.root).load(camera_id, workflow_id)
            record = self._load(camera_id, workflow_id)
            self._assert_policy(record)
            progress_callback(5, "严格解码并按独立 session 构建训练/校准基线")
            manifest = build_scene_profile_baseline(
                camera_id=camera_id, scene_profile_id=workflow_id, videos=videos,
                annotations=annotations, reference=draft.reference_path,
                output_dir=self._baseline(camera_id, workflow_id),
                sample_every=sample_every, max_images=max_images,
                min_training_images=100, min_calibration_images=50,
                strict_release_decoder=True, allow_alignable=False,
                scene_guard_config=SceneGuardConfig(**record["scene_guard"]),
            )
            if manifest.get("schema_version") != BASELINE_SCHEMA or manifest.get("training_ready") is not True:
                raise ValueError("strict baseline was produced but is not training-ready; inspect its rejection evidence")
            baseline = self._baseline(camera_id, workflow_id)
            with self._lock(camera_id, workflow_id, wait=True):
                record = self._load(camera_id, workflow_id)
                if manifest.get("reference", {}).get("sha256") != record["reference_sha256"]:
                    raise ValueError("baseline reference differs from the frozen scene draft")
                record.update(
                    state="baseline_ready", error=None,
                    baseline_sha256=evidence_artifact_sha256(baseline),
                    manifest_sha256=_file_sha256(baseline / "manifest.json"),
                )
                self._save(record)
            progress_callback(100, "v3 基线已冻结；仍需人工确认训练和后续发布验收")
            return {"workflow_id": workflow_id, "schema_version": BASELINE_SCHEMA, "training_ready": True}
        except BaseException as exc:
            with self._lock(camera_id, workflow_id, wait=True):
                record = self._load(camera_id, workflow_id)
                record.update(state="baseline_failed", error=f"Baseline failed ({type(exc).__name__}); inspect task diagnostics")
                self._save(record)
            raise

    def submit_training(self, camera_id: str, workflow_id: str, *, model_type: str,
                        image_size: int, actor: str) -> str:
        from argus.anomaly.scene_training import validate_scene_training_input

        with self._lock(camera_id, workflow_id):
            record = self._load(camera_id, workflow_id)
            self._assert_policy(record)
            prior_job_id = record.get("training_job_id")
            if record.get("approved") is True:
                raise SceneWorkflowConflict("approved workflows are immutable; create a new workflow for another model")
            if prior_job_id:
                prior = self.database.get_training_job(prior_job_id)
                if prior is None or prior.status not in {"failed", "rejected"}:
                    raise SceneWorkflowConflict("prior training is active, complete or unconfirmed; duplicate submission refused")
            elif record.get("state") != "baseline_ready":
                raise SceneWorkflowConflict("a frozen training-ready baseline is required")
            lineage = {
                "schema_version": "argus.scene-training-input.v1",
                **{key: record[key] for key in (
                    "workflow_id", "scene_profile_id", "baseline_sha256",
                    "manifest_sha256", "reference_sha256",
                )},
            }
            validate_scene_training_input(
                baselines_dir=self.baselines_dir, camera_id=camera_id, scene_workflow=lineage,
            )
            self._audit(camera_id, workflow_id, actor, "scene_workflow.training.requested")
            job_id = str(uuid.uuid4())
            # Freeze the ID before the DB commit. An unknown commit outcome must
            # not silently create another job on retry.
            history = list(record.get("training_job_history", []))
            if prior_job_id:
                history.append(prior_job_id)
            record.update(state="training_submitted", training_job_id=job_id, training_job_history=history, error=None)
            self._save(record)
            job = TrainingJobRecord(
                job_id=job_id, job_type="anomaly_head", camera_id=camera_id,
                zone_id="default", model_type=model_type, trigger_type="manual",
                triggered_by=actor, confirmation_required=True,
                status=TrainingJobStatus.PENDING_CONFIRMATION.value,
                dataset_version=f"scene-{workflow_id}",
                hyperparameters=json.dumps({"scene_workflow": lineage, "image_size": image_size}),
            )
            self.database.save_training_job(job)
            return job_id

    def _trained_model(self, record: dict[str, Any]):
        from argus.anomaly.artifacts import artifact_metadata
        from argus.anomaly.scene_training import (
            validate_scene_training_calibration,
            validate_scene_training_input,
        )

        camera_id = record["camera_id"]
        self._assert_policy(record)
        job_id = record.get("training_job_id")
        job = self.database.get_training_job(job_id) if job_id else None
        if (
            job is None or job.status != "complete" or not job.model_version_id
            or job.camera_id != camera_id or job.job_type != "anomaly_head"
            or job.zone_id != "default" or job.dataset_version != f"scene-{record['workflow_id']}"
            or job.confirmation_required is not True
            or not isinstance(job.confirmed_by, str) or not job.confirmed_by.strip()
            or job.confirmed_by.strip().lower() == "unknown"
            or not isinstance(job.confirmed_at, datetime)
        ):
            raise SceneWorkflowConflict("the workflow's human-confirmed training job must complete first")
        params = json.loads(job.hyperparameters or "{}")
        baseline = validate_scene_training_input(
            baselines_dir=self.baselines_dir, camera_id=camera_id,
            scene_workflow=params.get("scene_workflow"),
        )
        if baseline.lineage["workflow_id"] != record["workflow_id"]:
            raise ValueError("training job belongs to a different scene workflow")
        model = self.registry.get_by_version_id(job.model_version_id)
        if model is None or model.camera_id != camera_id or model.stage == "retired":
            raise ValueError("workflow model is unavailable or belongs to another camera")
        if model.data_hash != registry_training_data_hash(baseline.root):
            raise ValueError("registry model is not bound to the frozen scene baseline")
        training_params = json.loads(model.training_params or "{}")
        if training_params.get("scene_workflow") != baseline.lineage:
            raise ValueError("registry model has no exact scene training job lineage")
        metadata = artifact_metadata(Path(model.model_path))
        if metadata.get("calibration_status") != "ok":
            raise ValueError("successful runtime calibration is required before profile approval")
        validate_scene_training_calibration(
            model_path=model.model_path, scene_training_input=baseline,
        )
        digest = runtime_artifact_sha256(model.model_path)
        if model.model_hash != digest[:16]:
            raise ValueError("runtime model bytes changed after registration")
        return baseline, model, digest

    def approve_profile(self, camera_id: str, workflow_id: str, *, actor: str) -> str:
        with self._lock(camera_id, workflow_id):
            record = self._load(camera_id, workflow_id)
            if record.get("state") in {"selecting_profile", "selection_incomplete"}:
                raise SceneWorkflowConflict("scene selection requires administrator reconciliation before further operations")
            baseline, model, digest = self._trained_model(record)
            self._audit(camera_id, workflow_id, actor, "scene_workflow.profile_approval.requested")
            now = datetime.now(timezone.utc).isoformat()
            store = self._profile_store(camera_id, workflow_id)
            if (self.profiles_dir / camera_id / workflow_id).exists():
                # Reconcile only our exact immutable publication after a lost
                # workflow-record write. Never adopt an unrelated same-name profile.
                profile = store.load(camera_id, workflow_id)
                self._check_approved_profile(profile, record, model, digest)
            else:
                profile = store.create_from_baseline(
                    camera_id, workflow_id, baseline.root / "reference.jpg",
                    metadata={
                        "operator_approved": True, "validation_only": False,
                        "approved_by": actor, "approved_at": now,
                        "model_version_id": model.model_version_id,
                        "model_artifact_sha256": digest,
                        "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
                        "baseline_version": f"scene-{workflow_id}",
                        "baseline_sha256": record["baseline_sha256"],
                        "scene_workflow_id": workflow_id,
                        "training_job_id": record["training_job_id"],
                    },
                    activate=False,
                )
                self._check_approved_profile(profile, record, model, digest)
            # Keep the original approver/time when reconciling an already
            # published profile; a retry is not a new approval.
            record.update(
                state="profile_selected" if record.get("state") == "profile_selected" else "profile_approved", approved=True,
                approved_by=profile.metadata["approved_by"],
                approved_at=profile.metadata["approved_at"],
            )
            self._save(record)
            return profile.profile_id

    @staticmethod
    def _check_approved_profile(profile, record, model, digest) -> None:
        approved_by = profile.metadata.get("approved_by")
        approved_at = profile.metadata.get("approved_at")
        try:
            approved_time = datetime.fromisoformat(approved_at) if isinstance(approved_at, str) else None
            approval_time_valid = approved_time is not None and approved_time.utcoffset() is not None
        except (TypeError, ValueError, OverflowError):
            approval_time_valid = False
        expected = {
            "operator_approved": True, "validation_only": False,
            "model_version_id": model.model_version_id,
            "model_artifact_sha256": digest,
            "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
            "baseline_version": f"scene-{record['workflow_id']}",
            "baseline_sha256": record["baseline_sha256"],
            "scene_workflow_id": record["workflow_id"],
            "training_job_id": record["training_job_id"],
        }
        if (
            any(profile.metadata.get(key) != value for key, value in expected.items())
            or profile.metadata.get("operator_approved") is not True
            or profile.metadata.get("validation_only") is not False
            or profile.source_baseline_name != "reference.jpg"
            or profile.source_baseline_sha256 != record["reference_sha256"]
            or not isinstance(approved_by, str) or not approved_by.strip()
            or approved_by.strip().lower() == "unknown" or not approval_time_valid
        ):
            raise SceneWorkflowConflict("existing profile does not match this approved workflow")

    def activate_profile(self, camera_id: str, workflow_id: str, *, actor: str,
                         expected_active_profile_id: str | None,
                         expected_active_profile_revision: str | None) -> str:
        """Select baseline/profile files, never a runtime model or model stage.

        Both pointers are individually atomic. A partial filesystem failure is
        persisted for reconciliation; it is never reported as model activation.
        ModelReleaseService still prepares/verifies the runtime against the
        selected profile and existing production evidence gates.
        """
        with self._activation_lock(camera_id), self._lock(camera_id, workflow_id):
            store = self._profile_store(camera_id, workflow_id)
            if self._active_selection(camera_id, store) != (
                expected_active_profile_id, expected_active_profile_revision,
            ):
                raise SceneWorkflowConflict("active scene changed since confirmation; refresh and confirm the selection again")
            record = self._load(camera_id, workflow_id)
            if record.get("approved") is not True:
                raise SceneWorkflowConflict("explicit profile approval is required before scene selection")
            if record.get("state") not in {"profile_approved", "profile_selected"}:
                raise SceneWorkflowConflict("scene selection state is incomplete; administrator reconciliation is required")
            baseline, model, digest = self._trained_model(record)
            profile = store.load(camera_id, workflow_id)
            self._check_approved_profile(profile, record, model, digest)
            self._audit(camera_id, workflow_id, actor, "scene_workflow.profile_selection.requested")
            record.update(state="selecting_profile", error=None)
            self._save(record)
            try:
                marker = baseline.root.parent / "current.txt"
                _reject_links(marker)
                temporary = marker.with_name(f".current.{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("x", encoding="utf-8") as handle:
                        handle.write(baseline.root.name)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, marker)
                finally:
                    temporary.unlink(missing_ok=True)
                store.set_active(camera_id, workflow_id)
                record.update(state="profile_selected", selected_by=actor, selected_at=datetime.now(timezone.utc).isoformat())
                self._save(record)
            except BaseException:
                record.update(state="selection_incomplete", error="Scene selection incomplete; inspect both pointers before any release")
                self._save(record)
                raise
            return workflow_id

    def describe(self, record: dict[str, Any]) -> dict[str, Any]:
        camera_id, workflow_id = record["camera_id"], record["workflow_id"]
        baseline = self._baseline(camera_id, workflow_id)
        manifest = _read_json(baseline / "manifest.json") if (baseline / "manifest.json").exists() else {}
        counts = manifest.get("split_contract", {}).get("sample_counts", {})
        job_id = record.get("training_job_id")
        job = self.database.get_training_job(job_id) if job_id and self.database else None
        active = False
        if self.profiles_dir.exists():
            active = self._profile_store(camera_id).get_active_profile_id(camera_id) == workflow_id
        blockers = [
            "production 仍要求完整独立回归、现有全部质量门与可信 72 小时 soak",
            "选择活动场景只更新基线/profile 指针，不切换模型、不恢复评分、不修改 applied stage",
        ]
        if record.get("approved") is not True:
            blockers.insert(0, "先完成 v3 基线、人工确认训练、校准及独立 profile 审批")
        readiness_labels = {
            "corpus_evidence_complete": "基线来源/解码/审核证据不完整",
            "at_least_two_declared_independent_sessions": "不足两个经审核的独立正常录制 session",
            "minimum_training_images_met": "训练样本不足",
            "minimum_calibration_images_met": "独立校准样本不足",
            "minimum_accepted_time_span_met": "有效正常样本时间跨度不足",
            "full_source_grouping_has_no_diagnostic_samples": "存在仅诊断样本，不能用于正式场景训练",
        }
        checks = manifest.get("training_readiness_checks", {})
        if isinstance(checks, dict):
            for name, label in readiness_labels.items():
                if checks.get(name) is False:
                    blockers.insert(0, label)
        if record.get("scene_guard") != self.scene_guard:
            blockers.insert(0, "相机 SceneGuard 策略已变化；旧工作流禁止复用，请创建新工作流")
        if job_id and job is None:
            blockers.insert(0, "训练提交结果未知；核对数据库后再处理，禁止重复提交")
        return {
            "workflow_id": workflow_id, "profile_id": workflow_id,
            "state": record["state"], "created_by": record["created_by"], "created_at": record["created_at"],
            "reference_sha256": record["reference_sha256"],
            "baseline_schema": manifest.get("schema_version"),
            "training_ready": manifest.get("training_ready") is True and record.get("state") != "baseline_failed",
            "train_count": counts.get("train", 0), "calibration_count": counts.get("calibration", 0),
            "baseline_task_id": record.get("baseline_task_id"), "training_job_id": job_id,
            "training_status": job.status if job else None,
            "model_version_id": job.model_version_id if job else None,
            "approved": record.get("approved") is True, "active": active,
            "error": job.error if job and job.error else record.get("error"), "blockers": blockers,
        }

    def snapshot(self, camera_id: str) -> dict[str, Any]:
        _identifier(camera_id)
        camera_root = _reject_links(self.root / camera_id)
        workflows = []
        issues = []
        if camera_root.is_dir():
            for directory in sorted(camera_root.iterdir()):
                if directory.name.startswith(".") or not directory.is_dir():
                    continue
                try:
                    workflows.append(self.describe(self._load(camera_id, directory.name)))
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    # A partially published draft is not a valid workflow and
                    # must not gain training/approval authority. Keep other
                    # workflows usable, but never disguise it as empty data.
                    issues.append({
                        "workflow_id": directory.name,
                        "message": f"工作流记录缺失或损坏（{type(exc).__name__}）；需管理员核查，未授予任何操作权限",
                    })
        profiles = []
        active_profile_revision = None
        if self.profiles_dir.is_dir():
            store = self._profile_store(camera_id)
            active, active_profile_revision = self._active_selection(camera_id, store)
            profiles = [{
                "profile_id": profile.profile_id,
                "approved": profile.metadata.get("operator_approved") is True,
                "active": profile.profile_id == active,
                "model_version_id": profile.metadata.get("model_version_id"),
            } for profile in store.list_profiles(camera_id)]
        references = []
        camera_baselines = _reject_links(self.baselines_dir / camera_id)
        if camera_baselines.is_dir():
            for path in sorted(camera_baselines.glob("*/*/*")):
                if len(references) >= 100:
                    break
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                safe = _relative_file(camera_baselines, path.relative_to(camera_baselines).as_posix(), {".png", ".jpg", ".jpeg"})
                references.append({"path": safe.relative_to(camera_baselines).as_posix(), "sha256": _file_sha256(safe)})
        return {
            "camera_id": camera_id, "workflows": workflows, "issues": issues, "profiles": profiles, "references": references,
            "active_profile_revision": active_profile_revision,
            "source_root_label": str(self.sources_root),
            "requirements": {"baseline_schema": BASELINE_SCHEMA, "min_sources": 2,
                             "min_train_images": 100, "min_calibration_images": 50,
                             "production_gate_unchanged": True},
        }
