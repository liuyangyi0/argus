"""Conservative, single-GPU training admission. Construction/GET are inert.

The application maintenance latch excludes Argus model consumers. A bounded
nvidia-smi probe additionally rejects missing/ambiguous devices, active compute
processes and insufficient free memory. It is not an OS GPU memory hard limit or
a guarantee against another, unmanaged application starting after the probe.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import threading


_TRAINING_SLOTS_LOCK = threading.Lock()
# One manager is a host compute owner, not one independent slot per Dashboard
# or database. Keep strong identities until the admitted worker is cleaned up.
_TRAINING_SLOTS: dict[int, tuple[object, object, object]] = {}


def training_resource_lease_held(manager) -> bool:
    """Cached local ownership; never a GPU, PID or database observation."""
    with _TRAINING_SLOTS_LOCK:
        entry = _TRAINING_SLOTS.get(id(manager))
        return entry is not None and entry[0] is manager


def training_compute_exclusion_required(config, database=None):
    """Quarantine online compute across restart while old GPU ownership is unknown."""
    training = getattr(config, "task_training", None)
    if getattr(training, "enabled", False) is True and getattr(training, "maintenance_approved", False) is True:
        return True
    from argus.storage.database import Database
    if not isinstance(database, Database):
        return False
    from sqlalchemy import or_, select
    from argus.storage.models import TaskTrainingExecutionRecord
    try:
        with database.get_session() as session:
            return session.scalar(select(TaskTrainingExecutionRecord.job_id).where(
                or_(TaskTrainingExecutionRecord.status.is_(None),
                    TaskTrainingExecutionRecord.status.not_in({"queued", "waiting_resources", "complete", "failed", "cancelled"})),
            ).limit(1)) is not None
    except Exception:
        # An unreadable owner ledger is not proof that the GPU is free.
        return True


class NvidiaTrainingProbe:
    def __call__(self) -> dict:
        # The validated recipe selects physical device 0. Do not silently
        # reinterpret it through a user/session CUDA remapping.
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "0"}:
            raise RuntimeError("device mapping unavailable")
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise RuntimeError("GPU probe unavailable")

        def query(fields):
            result = subprocess.run([executable, fields, "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5, check=True)
            if len(result.stdout) > 65536:
                raise RuntimeError("GPU probe output too large")
            return list(csv.reader(io.StringIO(result.stdout.strip()))) if result.stdout.strip() else []

        rows = query("--query-gpu=index,uuid,memory.total,memory.free")
        if len(rows) != 1 or len(rows[0]) != 4:
            raise RuntimeError("only a single unambiguous GPU is supported")
        index, uuid, total, free = (value.strip() for value in rows[0])
        if index != "0" or not uuid.startswith("GPU-"):
            raise RuntimeError("GPU identity unavailable")
        total, free = int(total), int(free)
        if not 0 <= free <= total or total <= 0:
            raise RuntimeError("GPU memory unavailable")
        processes = query("--query-compute-apps=gpu_uuid,pid")
        if any(len(row) != 2 or row[0].strip() != uuid or not row[1].strip().isdigit() for row in processes):
            raise RuntimeError("GPU process observation unavailable")
        return {"device": "cuda:0", "free_memory_mb": free, "active_compute": bool(processes)}


class _TrainingResourceLease:
    def __init__(self, admission, token):
        self.admission = admission
        self.token = token
        self.released = False

    def release(self):
        with self.admission._lock:
            if self.released:
                return
            if self.admission._lease is not self:
                raise RuntimeError("training lease identity changed")
            with _TRAINING_SLOTS_LOCK:
                entry = _TRAINING_SLOTS.get(id(self.admission.manager))
                if (entry is None or entry[0] is not self.admission.manager
                        or entry[1] is not self.admission or entry[2] is not self.token):
                    raise RuntimeError("training host slot identity changed")
                del _TRAINING_SLOTS[id(self.admission.manager)]
            self.admission._lease = None
            self.released = True
            self.admission._set("ready", None)


class TaskTrainingResourceAdmission:
    def __init__(self, config, camera_manager, *, database=None, probe=None):
        self.config = config.model_copy(deep=True)
        self.manager = camera_manager
        self.database = database
        self.probe = probe or NvidiaTrainingProbe()
        self._lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._lease = None
        self._status = {"state": "disabled", "reason_code": "TRAINING_DISABLED", "device": "cuda:0",
                        "free_memory_mb": None, "memory_limit_enforced": False}
        if self.config.enabled:
            self._set("waiting_maintenance", "TRAINING_MAINTENANCE_REQUIRED")

    def _set(self, state, reason):
        with self._snapshot_lock:
            self._status.update(state=state, reason_code=reason)

    def snapshot(self):
        with self._snapshot_lock:
            value = dict(self._status)
        with _TRAINING_SLOTS_LOCK:
            entry = _TRAINING_SLOTS.get(id(self.manager))
            if entry is not None and entry[0] is self.manager and entry[1] is not self:
                value.update(state="leased", reason_code="TRAINING_RESOURCE_IN_USE")
        return value

    def _memory(self, value):
        with self._snapshot_lock:
            self._status["free_memory_mb"] = value

    def __call__(self, _job):
        with self._lock:
            if not self.config.enabled:
                self._set("disabled", "TRAINING_DISABLED")
                return None
            if self._lease is not None:
                self._set("leased", "TRAINING_RESOURCE_IN_USE")
                return None
            if not self.config.maintenance_approved:
                self._set("waiting_maintenance", "TRAINING_MAINTENANCE_REQUIRED")
                return None
            self._memory(None)
            token = object()
            with _TRAINING_SLOTS_LOCK:
                if id(self.manager) in _TRAINING_SLOTS:
                    self._set("leased", "TRAINING_RESOURCE_IN_USE")
                    return None
                _TRAINING_SLOTS[id(self.manager)] = (self.manager, self, token)
            granted = False
            try:
                enter = getattr(self.manager, "enter_training_maintenance", None)
                if not callable(enter) or enter() is not True:
                    self._set("blocked", "TRAINING_ONLINE_COMPUTE_ACTIVE")
                    return None
                if self.database is not None and self.database.list_training_jobs(status="running"):
                    self._set("blocked", "TRAINING_LEGACY_JOB_ACTIVE")
                    return None
                observed = self.probe()
                free = observed.get("free_memory_mb")
                if (observed.get("device") != "cuda:0" or type(free) is not int or free < 0
                        or type(observed.get("active_compute")) is not bool):
                    raise RuntimeError("invalid resource observation")
                self._memory(free)
                if observed["active_compute"]:
                    self._set("blocked", "TRAINING_EXTERNAL_COMPUTE_ACTIVE")
                    return None
                if free < self.config.minimum_free_memory_mb:
                    self._set("blocked", "TRAINING_INSUFFICIENT_GPU_MEMORY")
                    return None
                self._lease = _TrainingResourceLease(self, token)
                self._set("leased", None)
                granted = True
                return self._lease
            except Exception:
                self._memory(None)
                self._set("blocked", "TRAINING_RESOURCE_OBSERVATION_UNAVAILABLE")
                return None
            finally:
                if not granted:
                    # No worker was returned or started, so this provisional
                    # slot is positively unused. Do not touch a successor.
                    with _TRAINING_SLOTS_LOCK:
                        entry = _TRAINING_SLOTS.get(id(self.manager))
                        if entry is not None and entry[1] is self and entry[2] is token:
                            del _TRAINING_SLOTS[id(self.manager)]
