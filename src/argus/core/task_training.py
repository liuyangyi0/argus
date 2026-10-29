"""Durable, explicitly approved task training; import never starts computation.

The database is the job authority. Files are derived execution inputs/results,
not a second queue. Only the lazy child entry point imports a model SDK.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import MutableMapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from argus.capture.task_collector import controlled_path, png_image_size, safe_component
from argus.config.schema import TaskTrainingConfig, TrustedTaskTrainingBaseModelConfig
from argus.storage.models import TaskTrainingExecutionRecord
from argus.storage.task_workflows import TaskWorkflowBlocked, TaskWorkflowConflict, _actor, _canonical, _sha

RECIPE_ID = "yolov8_detection.v1"
ACTIVE_STATES = {"queued", "waiting_resources", "running", "cancelling", "reconciling"}
_PARAM_DEFAULTS = {"epochs": 10, "imgsz": 640, "batch": 4, "seed": 0, "timeout_seconds": 3600}


class _StartupTrainingTrust(MutableMapping):
    """Revocable compatibility mapping; never expands startup provenance."""

    def __init__(self, initial, lock):
        self._allowed, self._values, self._lock = dict(initial), dict(initial), lock

    def __getitem__(self, key):
        with self._lock:
            return self._values[key]

    def __iter__(self):
        with self._lock:
            return iter(tuple(self._values))

    def __len__(self):
        with self._lock:
            return len(self._values)

    def __setitem__(self, key, value):
        with self._lock:
            if key not in self._allowed or self._allowed[key] != value:
                raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_NOT_TRUSTED")
            self._values[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._values[key]

    def clear(self):
        with self._lock:
            self._values.clear()

    def copy(self):
        with self._lock:
            return dict(self._values)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        with self._lock:
            if any(key not in self._allowed or self._allowed[key] != value
                   for key, value in changes.items()):
                raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_NOT_TRUSTED")
            self._values.update(changes)


def training_params(params: dict) -> dict:
    if not isinstance(params, dict) or set(params) - set(_PARAM_DEFAULTS):
        raise ValueError("unsupported training parameter")
    value = {**_PARAM_DEFAULTS, **params}
    limits = {"epochs": (1, 1000), "imgsz": (32, 2048), "batch": (1, 64),
              "seed": (0, 2**31 - 1), "timeout_seconds": (60, 86400)}
    if any(type(value[key]) is not int or not low <= value[key] <= high for key, (low, high) in limits.items()):
        raise ValueError("invalid training parameter")
    if value["imgsz"] % 32:
        raise ValueError("imgsz must be a multiple of 32")
    return value


def artifact_sha256(path: Path, *, max_bytes=512 * 1024 * 1024) -> str:
    size = path.stat().st_size
    if not 0 < size <= max_bytes:
        raise ValueError("training artifact has invalid size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        read = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            read += len(block)
            if read > max_bytes:
                raise ValueError("training artifact exceeds its size bound")
            digest.update(block)
    if read != size:
        raise TaskWorkflowConflict("training artifact changed while reading")
    return digest.hexdigest()


class TaskTrainingService:
    def __init__(self, workflows, output_root: str | Path, models_root: str | Path,
                 *, trusted_base_models: dict[str, str] | None = None,
                 configuration: TaskTrainingConfig | None = None):
        self.workflows, self._db = workflows, workflows._db
        self.models_root = Path(models_root).resolve()
        self.output_root = Path(output_root).resolve()
        if not self.output_root.is_relative_to(self.models_root) or self.output_root == self.models_root:
            raise ValueError("training outputs need a dedicated directory inside models_root")
        # PT can contain Python pickle. A request-supplied hash proves identity,
        # not provenance. Only operator-preprovisioned, server-configured base
        # weights may cross the lazy deserialization boundary; default is deny.
        self._configuration = TaskTrainingConfig.model_validate(
            configuration.model_dump() if configuration is not None else {})
        self._configuration_identity = _canonical(self._configuration.model_dump())
        self._authority_lock = threading.RLock()
        if trusted_base_models and self._configuration.trusted_base_models:
            raise ValueError("training trust must use one startup configuration source")
        bases = list(self._configuration.trusted_base_models)
        if trusted_base_models:
            # Keep the existing server-side mapping constructor compatible.
            # Its display identities are deterministic, never absolute paths.
            bases = [TrustedTaskTrainingBaseModelConfig(
                model_id="legacy-" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:16],
                name=path.rsplit("/", 1)[-1], relative_path=path, artifact_sha256=digest,
            ) for path, digest in trusted_base_models.items()]
            TaskTrainingConfig(trusted_base_models=bases)
        self._configured_base_models = tuple(base.model_dump() for base in bases)
        self._trusted_base_models = _StartupTrainingTrust(
            {base.relative_path: base.artifact_sha256 for base in bases}, self._authority_lock)

    @property
    def trusted_base_models(self):
        return self._trusted_base_models

    def configuration_matches(self, configuration: TaskTrainingConfig) -> bool:
        """Compare metadata only; runtime wiring cannot substitute new authority."""
        if not isinstance(configuration, TaskTrainingConfig):
            return False
        try:
            current = TaskTrainingConfig.model_validate(configuration.model_dump())
        except (TypeError, ValueError):
            return False
        return _canonical(current.model_dump()) == self._configuration_identity

    def capabilities(self, *, runtime_snapshot: dict | None = None,
                     resource_snapshot: dict | None = None) -> dict:
        """Read startup authority and cached observations, never model/GPU state.

        This advertises configured provenance, not file existence, hash validity,
        resource reservation, training admission, or production release approval.
        """
        runtime_states = {"disabled", "not_started", "running", "waiting_resources", "reconciling",
                          "owner_unavailable", "stopping", "stopped", "cleanup_unconfirmed", "error"}
        runtime = {"state": "disabled", "enabled": False, "error_code": None}
        if isinstance(runtime_snapshot, dict):
            state = runtime_snapshot.get("state")
            runtime = {"state": state if isinstance(state, str) and state in runtime_states else "error",
                       "enabled": runtime_snapshot.get("enabled") is True,
                       "error_code": self._capability_code(runtime_snapshot.get("error_code"))}
        resource = {"state": "disabled", "reason_code": None, "device": "cuda:0",
                    "free_memory_mb": None, "memory_limit_enforced": False}
        if isinstance(resource_snapshot, dict):
            state = resource_snapshot.get("state")
            memory = resource_snapshot.get("free_memory_mb")
            resource.update({
                "state": state if isinstance(state, str) and state in {"disabled", "waiting_maintenance", "blocked", "ready", "leased"} else "blocked",
                "reason_code": self._capability_code(resource_snapshot.get("reason_code")),
                "free_memory_mb": memory if type(memory) is int and 0 <= memory <= 2**31 - 1 else None,
            })
        config = self._configuration
        with self._authority_lock:
            trusted_options = [{**base, "configured_trust": True} for base in self._configured_base_models
                               if self.trusted_base_models.get(base["relative_path"]) == base["artifact_sha256"]]
        return {"schema": "argus.task-training-capabilities.v1", "recipe_id": RECIPE_ID,
                "configured_enabled": config.enabled, "maintenance_approved": config.maintenance_approved,
                "resource_mode": config.resource_mode, "device": config.device,
                "minimum_free_memory_mb": config.minimum_free_memory_mb,
                "poll_interval_seconds": config.poll_interval_seconds,
                "trusted_base_models": trusted_options,
                "runtime": runtime, "resource": resource,
                "validation_only": True, "production_allowed": False}

    @staticmethod
    def _capability_code(value: object) -> str | None:
        return value if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,100}", value) else None

    def _trusted_base(self, relative_path: str, digest: str) -> dict:
        with self._authority_lock:
            if self.trusted_base_models.get(relative_path) != digest:
                raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_NOT_TRUSTED")
            return {"schema": "argus.operator-preprovisioned-training-base.v1",
                    "relative_path": relative_path, "artifact_sha256": digest}

    def _request_trust(self, request):
        submission = request["submission"]
        trust = self._trusted_base(submission["base_model_path"], submission["base_model_sha256"])
        if trust != request.get("base_model_trust"):
            raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_TRUST_CHANGED")
        return trust

    @staticmethod
    def _request(row) -> dict:
        if _sha(row.request_json) != row.request_sha256:
            raise TaskWorkflowBlocked("TRAINING_REQUEST_INTEGRITY_FAILED")
        try:
            request = json.loads(row.request_json)
            if (not isinstance(request, dict)
                    or set(request) != {"schema", "recipe_id", "submission", "package",
                                        "workflow_manifest_sha256", "base_model_trust"}
                    or request["schema"] != "argus.task-training-request.v1"
                    or request["recipe_id"] != RECIPE_ID):
                raise ValueError("invalid request schema")
            submission, package = request["submission"], request["package"]
            if (not isinstance(submission, dict)
                    or set(submission) != {"workflow_id", "expected_revision", "package_sha256",
                                          "base_model_path", "base_model_sha256", "params", "actor",
                                          "execution_approval"}
                    or not isinstance(package, dict) or package["recipe"] != "external_detection.v1"
                    or type(submission["expected_revision"]) is not int or submission["expected_revision"] < 1
                    or training_params(submission["params"]) != submission["params"]):
                raise ValueError("invalid request submission")
            actor = _actor(submission["actor"])
            identity_matches = (
                submission["workflow_id"] == row.workflow_id
                and package["binding"]["task_id"] == row.task_id
                and actor == row.created_by
                and submission["execution_approval"] == {"actor": actor, "scope": "training_execution_only"}
                and package["package_sha256"] == submission["package_sha256"]
                and request["base_model_trust"] == {
                    "schema": "argus.operator-preprovisioned-training-base.v1",
                    "relative_path": submission["base_model_path"],
                    "artifact_sha256": submission["base_model_sha256"],
                })
        except (TypeError, ValueError, KeyError) as exc:
            raise TaskWorkflowBlocked("TRAINING_REQUEST_SCHEMA_INVALID") from exc
        if not identity_matches:
            raise TaskWorkflowBlocked("TRAINING_REQUEST_IDENTITY_MISMATCH")
        return request

    def _view(self, row) -> dict:
        request = self._request(row)
        return {"job_id": row.job_id, "task_id": row.task_id, "workflow_id": row.workflow_id,
                "revision": row.revision, "status": row.status, "request_sha256": row.request_sha256,
                "recipe_id": RECIPE_ID, "binding": request["package"]["binding"],
                "dataset_id": request["package"]["dataset"]["dataset_id"],
                "package_sha256": request["package"]["package_sha256"],
                "params": request["submission"]["params"], "created_by": row.created_by,
                "result": json.loads(row.result_json) if row.result_json else None,
                "production_allowed": False, "approval_scope": "training_only"}

    def get(self, job_id: str) -> dict:
        with self._db.get_session() as session:
            row = session.get(TaskTrainingExecutionRecord, job_id)
            if row is None:
                raise KeyError("task training job does not exist")
            return self._view(row)

    def list(self, *, task_id: str | None = None) -> list[dict]:
        with self._db.get_session() as session:
            statement = select(TaskTrainingExecutionRecord).order_by(TaskTrainingExecutionRecord.created_at)
            if task_id is not None:
                statement = statement.where(TaskTrainingExecutionRecord.task_id == task_id)
            return [self._view(row) for row in session.scalars(statement)]

    def submit(self, *, workflow_id: str, expected_revision: int, expected_package_sha256: str,
               base_model_path: str, base_model_sha256: str, params: dict, actor: str,
               idempotency_key: str) -> dict:
        actor = _actor(actor)
        safe_component(workflow_id)
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("training needs an integer workflow revision")
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 128:
            raise ValueError("training needs an idempotency key")
        for digest in (expected_package_sha256, base_model_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("training needs exact SHA-256 identities")
        submission = {"workflow_id": workflow_id, "expected_revision": expected_revision,
                      "package_sha256": expected_package_sha256, "base_model_path": base_model_path,
                      "base_model_sha256": base_model_sha256, "params": training_params(params), "actor": actor,
                      "execution_approval": {"actor": actor, "scope": "training_execution_only"}}
        job_id = "training-" + _sha(_canonical({"workflow_id": workflow_id, "key": idempotency_key}))[:32]
        with self._db.get_session() as session:
            existing = session.get(TaskTrainingExecutionRecord, job_id)
            if existing is not None:
                if self._request(existing)["submission"] != submission:
                    raise TaskWorkflowConflict("training idempotency key was reused with different content")
                return self._view(existing)
        # Hashing a large approved dataset/base must not hold the database's
        # writer lock and starve live event/outbox persistence. Recheck the
        # complete workflow digest under the final short write transaction.
        source = self.workflows.get(workflow_id)
        if source["revision"] != expected_revision:
            raise TaskWorkflowConflict("workflow revision changed")
        if source["state"] != "training_export_ready":
            raise TaskWorkflowBlocked("APPROVED_TRAINING_PACKAGE_REQUIRED")
        package = self.workflows.export_training_package(workflow_id)
        if package["package_sha256"] != expected_package_sha256 or package["recipe"] != "external_detection.v1":
            raise TaskWorkflowConflict("training package identity changed")
        trust = self._trusted_base(base_model_path, base_model_sha256)
        base = controlled_path(self.models_root, base_model_path, must_exist=True)
        if base.suffix.lower() != ".pt" or artifact_sha256(base) != base_model_sha256:
            raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_IDENTITY_MISMATCH")
        with self._authority_lock, self._db.get_session() as session:
            self.workflows._begin(session)
            existing = session.get(TaskTrainingExecutionRecord, job_id)
            if existing is not None:
                if self._request(existing)["submission"] != submission:
                    raise TaskWorkflowConflict("training idempotency key was reused with different content")
                return self._view(existing)
            if self._trusted_base(base_model_path, base_model_sha256) != trust:
                raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_TRUST_CHANGED")
            record, manifest = self.workflows._record(session, workflow_id, expected_revision)
            if record.manifest_sha256 != source["manifest_sha256"] or manifest["binding"] != package["binding"]:
                raise TaskWorkflowConflict("training approval changed during preparation")
            if record.template_id != "object_detection.v1":
                raise TaskWorkflowBlocked("TRAINING_TEMPLATE_NOT_SUPPORTED")
            if record.state != "training_export_ready":
                raise TaskWorkflowBlocked("APPROVED_TRAINING_PACKAGE_REQUIRED")
            active = session.scalar(select(TaskTrainingExecutionRecord).where(
                TaskTrainingExecutionRecord.task_id == record.task_id,
                TaskTrainingExecutionRecord.status.in_(ACTIVE_STATES)))
            if active is not None:
                raise TaskWorkflowConflict("task already has an unresolved training job")
            request = {"schema": "argus.task-training-request.v1", "recipe_id": RECIPE_ID,
                       "submission": submission, "package": package,
                       "workflow_manifest_sha256": record.manifest_sha256,
                       "base_model_trust": trust}
            encoded = _canonical(request)
            row = TaskTrainingExecutionRecord(job_id=job_id, task_id=record.task_id,
                workflow_id=workflow_id, status="queued", revision=1, request_json=encoded,
                request_sha256=_sha(encoded), result_json=None, created_by=actor)
            session.add(row)
            session.flush()
            result = self._view(row)
            session.commit()
            return result

    def _approved_workflow(self, session, row, request):
        record, manifest = self.workflows._record(session, row.workflow_id)
        if (record.revision != request["submission"]["expected_revision"]
                or record.state != "training_export_ready"
                or record.manifest_sha256 != request["workflow_manifest_sha256"]
                or manifest["binding"] != request["package"]["binding"]
                or manifest["dataset_id"] != request["package"]["dataset"]["dataset_id"]):
            raise TaskWorkflowConflict("training approval or dataset changed")
        return record

    def prepare(self, job_id: str) -> dict:
        """Internal execution boundary: recheck approval and every source image."""
        with self._authority_lock, self._db.get_session() as session:
            row = session.get(TaskTrainingExecutionRecord, job_id)
            if row is None:
                raise KeyError("task training job does not exist")
            request = self._request(row)
            self._approved_workflow(session, row, request)
            self._request_trust(request)
            workflow_id, request_sha256 = row.workflow_id, row.request_sha256
        # Large file work never retains a DB reader/writer or authority lock.
        package = self.workflows.export_training_package(workflow_id)
        if package != request["package"]:
            raise TaskWorkflowConflict("training package changed")
        base = controlled_path(self.models_root, request["submission"]["base_model_path"], must_exist=True)
        if artifact_sha256(base) != request["submission"]["base_model_sha256"]:
            raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_IDENTITY_MISMATCH")
        self._request_trust(request)
        return {"schema": "argus.task-training-plan.v1", "job_id": job_id,
                "request_sha256": request_sha256, "request": request,
                "images_root": str(self.workflows.root), "models_root": str(self.models_root),
                "output_root": str(self.output_root)}

    def transition(self, job_id: str, *, expected_revision: int, from_states: set[str],
                   status: str, result: dict | None = None) -> dict:
        """Internal CAS; HTTP callers never supply arbitrary state transitions."""
        with self._authority_lock, self._db.get_session() as session:
            self.workflows._begin(session)
            row = session.get(TaskTrainingExecutionRecord, job_id)
            if row is None:
                raise KeyError("task training job does not exist")
            if row.revision != expected_revision or row.status not in from_states:
                raise TaskWorkflowConflict("training execution revision changed")
            request = self._request(row)
            if status == "complete":
                # Result preflight runs outside the writer lock. Approval and
                # execution authority must still be current at the actual CAS.
                self._approved_workflow(session, row, request)
                self._request_trust(request)
                execution = json.loads(row.result_json or "{}")
                if (not isinstance(result, dict) or not execution.get("execution_id")
                        or result.get("execution_id") != execution["execution_id"]
                        or result.get("request_sha256") != row.request_sha256
                        or result.get("job_id") != row.job_id
                        or result.get("package_sha256") != request["package"]["package_sha256"]):
                    raise TaskWorkflowConflict("training result execution identity changed")
            row.status, row.revision = status, row.revision + 1
            row.result_json = _canonical(result) if result is not None else row.result_json
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            value = self._view(row)
            session.commit()
            return value

    def cancel(self, job_id: str, *, expected_revision: int, actor: str) -> dict:
        _actor(actor)
        current = self.get(job_id)
        if current["status"] in {"cancelled", "complete", "failed"}:
            if current["revision"] != expected_revision:
                raise TaskWorkflowConflict("training execution revision changed")
            return current
        state = "cancelled" if current["status"] in {"queued", "waiting_resources"} else "cancelling"
        result = {**(current["result"] or {}), "cancelled_by": actor}
        return self.transition(job_id, expected_revision=expected_revision, from_states=ACTIVE_STATES,
                               status=state, result=result)

    def claim(self, job_id: str, *, expected_revision: int, execution_id: str | None = None) -> tuple[dict, dict]:
        plan = self.prepare(job_id)
        # Recheck the bound workflow under the same write transaction as claim.
        with self._authority_lock, self._db.get_session() as session:
            self.workflows._begin(session)
            row = session.get(TaskTrainingExecutionRecord, job_id)
            if row is None or row.revision != expected_revision or row.status not in {"queued", "waiting_resources"}:
                raise TaskWorkflowConflict("training job is no longer queued")
            if row.request_sha256 != plan["request_sha256"]:
                raise TaskWorkflowConflict("training request changed before execution")
            request = self._request(row)
            self._approved_workflow(session, row, request)
            self._request_trust(request)
            execution_id = execution_id or uuid4().hex
            if not isinstance(execution_id, str) or len(execution_id) != 32 or any(char not in "0123456789abcdef" for char in execution_id):
                raise ValueError("invalid training execution identity")
            plan["execution_id"] = execution_id
            row.status, row.revision = "running", row.revision + 1
            row.result_json = _canonical({"execution_id": execution_id, "worker_pid": None})
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            value = self._view(row)
            session.commit()
            return value, plan

    @contextmanager
    def execution_authorization(self, job_id: str, *, expected_revision: int,
                                execution_id: str, request_sha256: str):
        """Hold the final current approval/trust boundary around worker start.

        The caller must not nest another service write in this narrow block.
        No model hashing, device observation or model SDK call occurs here.
        Cancellation and trust revocation serialize before or after the start,
        rather than committing between the last check and starting a child.
        """
        with self._authority_lock, self._db.get_session() as session:
            self.workflows._begin(session)
            row = session.get(TaskTrainingExecutionRecord, job_id)
            if (row is None or row.status != "running" or row.revision != expected_revision
                    or row.request_sha256 != request_sha256):
                raise TaskWorkflowConflict("training execution authorization changed")
            execution = json.loads(row.result_json or "{}")
            if execution.get("execution_id") != execution_id or execution.get("cancelled_by"):
                raise TaskWorkflowConflict("training execution authorization changed")
            request = self._request(row)
            self._approved_workflow(session, row, request)
            self._request_trust(request)
            yield
            session.commit()


class CudaTrainingWitness:
    """Observe actual training callbacks, not the post-training CPU checkpoint."""
    def __init__(self):
        self.completed_batches = 0

    def batch_start(self, trainer):
        if not str(trainer.device).startswith("cuda") or not str(next(trainer.model.parameters()).device).startswith("cuda"):
            raise TaskWorkflowBlocked("TRAINING_BACKEND_IDENTITY_MISMATCH")

    def batch_end(self, trainer):
        self.batch_start(trainer)
        if not str(getattr(getattr(trainer, "loss", None), "device", "")).startswith("cuda"):
            raise TaskWorkflowBlocked("TRAINING_LOSS_BACKEND_IDENTITY_MISMATCH")
        self.completed_batches += 1


def execute_training_plan(plan: dict) -> dict:
    """EXPLICIT CHILD-ONLY real training entry point; never call in local tests.

    API contract: Ultralytics train/export documentation. The exact SDK is
    pinned because raw YOLOv8 ONNX output is part of Argus' adapter contract.
    No automatic dependency/model/dataset download is permitted.
    """
    import importlib.metadata
    import os
    import sys

    if (plan.get("schema") != "argus.task-training-plan.v1"
            or plan["request"].get("recipe_id") != RECIPE_ID):
        raise TaskWorkflowBlocked("TRAINING_RECIPE_NOT_SUPPORTED")
    if _sha(_canonical(plan["request"])) != plan["request_sha256"]:
        raise TaskWorkflowBlocked("TRAINING_REQUEST_INTEGRITY_FAILED")
    request = plan["request"]
    if request.get("base_model_trust") != {
        "schema": "argus.operator-preprovisioned-training-base.v1",
        "relative_path": request["submission"]["base_model_path"],
        "artifact_sha256": request["submission"]["base_model_sha256"],
    }:
        raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_NOT_TRUSTED")
    params = training_params(request["submission"]["params"])
    output_root = Path(plan["output_root"])
    run = controlled_path(output_root, safe_component(plan["job_id"]) + "/" + safe_component(plan["execution_id"]))
    run.mkdir(parents=True, exist_ok=False)
    # Must precede SDK import: settings and training output stay in this owned
    # execution directory. A missing optional dependency fails, not pip install.
    os.environ.update({"YOLO_CONFIG_DIR": str(run / "sdk-config"), "YOLO_AUTOINSTALL": "false",
                       "YOLO_OFFLINE": "true", "WANDB_MODE": "disabled", "COMET_MODE": "DISABLED"})
    def reject_network(event, _args):
        if event == "socket.connect":
            raise PermissionError("OFFLINE_TRAINING_NETWORK_DISABLED")
    sys.addaudithook(reject_network)
    if importlib.metadata.version("ultralytics") != "8.4.24":
        raise TaskWorkflowBlocked("TRAINING_SDK_VERSION_NOT_VALIDATED")
    import torch
    from ultralytics import YOLO, settings

    if not torch.cuda.is_available():
        raise TaskWorkflowBlocked("TRAINING_CUDA_REQUIRED")
    settings.update({key: False for key in (
        "sync", "hub", "wandb", "mlflow", "clearml", "comet", "neptune", "raytune", "tensorboard",
    ) if key in settings})
    source = controlled_path(Path(plan["models_root"]), request["submission"]["base_model_path"], must_exist=True)
    if artifact_sha256(source) != request["submission"]["base_model_sha256"]:
        raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_IDENTITY_MISMATCH")
    with source.open("rb") as stream:
        base_bytes = stream.read(512 * 1024 * 1024 + 1)
    if len(base_bytes) > 512 * 1024 * 1024:
        raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_TOO_LARGE")
    if hashlib.sha256(base_bytes).hexdigest() != request["submission"]["base_model_sha256"]:
        raise TaskWorkflowBlocked("TRAINING_BASE_MODEL_IDENTITY_MISMATCH")
    local_base = run / "base.pt"
    with local_base.open("xb") as stream:
        stream.write(base_bytes)
    dataset = request["package"]["dataset"]
    labels = dataset["label_schema"]["labels"]
    if not labels or dataset["label_schema"]["kind"] != "detection":
        raise TaskWorkflowBlocked("TRAINING_LABEL_SCHEMA_INVALID")
    for sample in dataset["samples"]:
        original = controlled_path(Path(plan["images_root"]), sample["relative_path"], must_exist=True)
        with original.open("rb") as stream:
            content = stream.read(32 * 1024 * 1024 + 1)
        if (len(content) > 32 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != sample["image_sha256"]
                or tuple(sample["image_size"]) != png_image_size(content)):
            raise TaskWorkflowBlocked("TRAINING_SAMPLE_IDENTITY_CHANGED")
        split = {"train": "train", "validation": "val"}.get(sample["split"])
        if split is None:
            raise TaskWorkflowBlocked("TRAINING_SPLIT_INVALID")
        image = controlled_path(run, f"dataset/images/{split}/{safe_component(sample['sample_id'])}.png")
        annotation = controlled_path(run, f"dataset/labels/{split}/{safe_component(sample['sample_id'])}.txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        annotation.parent.mkdir(parents=True, exist_ok=True)
        with image.open("xb") as stream:
            stream.write(content)
        rows = []
        for item in sample["review"]["annotation"]["objects"]:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            rows.append(f"{labels.index(item['label'])} {(x1+x2)/2:.12g} {(y1+y2)/2:.12g} {x2-x1:.12g} {y2-y1:.12g}")
        with annotation.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(rows))
    data_file = run / "dataset.yaml"
    with data_file.open("x", encoding="utf-8") as stream:
        json.dump({"path": str(run / "dataset"), "train": "images/train", "val": "images/val",
                   "names": labels}, stream)
    model = YOLO(str(local_base), task="detect")
    architecture = Path(str(model.model.yaml.get("yaml_file", ""))).name
    if architecture not in {f"yolov8{scale}.yaml" for scale in "nslmx"}:
        raise TaskWorkflowBlocked("TRAINING_BASE_ARCHITECTURE_NOT_SUPPORTED")
    witness = CudaTrainingWitness()
    model.add_callback("on_train_batch_start", witness.batch_start)
    model.add_callback("on_train_batch_end", witness.batch_end)
    model.train(data=str(data_file), epochs=params["epochs"], imgsz=params["imgsz"], batch=params["batch"],
                seed=params["seed"], device=0, workers=0, project=str(run), name="training",
                exist_ok=False, resume=False, pretrained=False, amp=False, plots=False, cache=False)
    # Ultralytics reloads the best checkpoint on CPU after training. Prove the
    # actual batches/model/loss were CUDA, rather than misreading that reload.
    if witness.completed_batches < 1:
        raise TaskWorkflowBlocked("TRAINING_BACKEND_IDENTITY_MISMATCH")
    if [model.names[index] for index in range(len(labels))] != labels or len(model.names) != len(labels):
        raise TaskWorkflowBlocked("TRAINING_EXPORTED_LABEL_SCHEMA_MISMATCH")
    exported = Path(model.export(format="onnx", dynamic=False, simplify=False, nms=False,
                                 imgsz=params["imgsz"], batch=1, device=0, opset=17))
    models_root = Path(plan["models_root"]).resolve()
    if not exported.resolve().is_relative_to(run.resolve()) or not exported.resolve().is_relative_to(models_root):
        raise TaskWorkflowBlocked("TRAINING_ARTIFACT_ESCAPED_OUTPUT_ROOT")
    # Validate static raw-format graph without loading an inference session.
    import onnx
    from argus.core.task_adapters import _reject_external_onnx_data

    with exported.open("rb") as stream:
        payload = stream.read(512 * 1024 * 1024 + 1)
    if len(payload) > 512 * 1024 * 1024:
        raise TaskWorkflowBlocked("TRAINING_ARTIFACT_TOO_LARGE")
    _reject_external_onnx_data(payload)
    graph = onnx.load_model_from_string(payload).graph
    inputs, outputs = list(graph.input), list(graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise TaskWorkflowBlocked("TRAINING_EXPORTED_FORMAT_NOT_SUPPORTED")
    def dimensions(value):
        return [dim.dim_value if dim.HasField("dim_value") else None
                for dim in value.type.tensor_type.shape.dim]
    output_shape = dimensions(outputs[0])
    if (dimensions(inputs[0]) != [1, 3, params["imgsz"], params["imgsz"]]
            or len(output_shape) != 3 or output_shape[0] != 1
            or output_shape[1] != 4 + len(labels) or not output_shape[2]
            or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
        raise TaskWorkflowBlocked("TRAINING_EXPORTED_FORMAT_NOT_SUPPORTED")
    descriptor = {"name": "Task-trained YOLOv8 detector", "adapter_id": "onnx_detection.v1",
                  "output_schema": "yolov8_detection.v1", "labels": labels,
                  "input_size": [params["imgsz"], params["imgsz"]],
                  "input_layout": "NCHW", "input_dtype": "float32", "color_order": "RGB"}
    # Candidate registration is a separate approved control-plane operation.
    # The worker returns no production stage, approval, or registry mutation.
    return {"schema": "argus.task-training-result.v1", "job_id": plan["job_id"],
            "execution_id": plan["execution_id"], "request_sha256": plan["request_sha256"],
            "package_sha256": request["package"]["package_sha256"], "provider": "torch-cuda",
            "candidate": {"relative_path": str(exported.resolve().relative_to(models_root)).replace("\\", "/"),
                          "artifact_sha256": artifact_sha256(exported), "descriptor": descriptor},
            "production_allowed": False, "stage": "candidate"}
