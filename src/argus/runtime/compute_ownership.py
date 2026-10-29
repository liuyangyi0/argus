"""Per-user, host-local exclusion for the validated single CUDA device.

This is an OS handle lock, not a PID file or an expiring database lease. Every
Argus database and process for the same user shares the default path. Import and
construction are inert; a compute consumer must explicitly acquire ownership.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading


_REGISTRY_LOCK = threading.Lock()
# Strong references are intentional. Losing the manager's last external
# reference is never evidence that its worker/model resources were cleaned up.
_OWNERS: dict[str, tuple["ComputeOwnership", object]] = {}


def _lock_file(stream):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class ComputeOwnership:
    """One manager may retain host compute until its cleanup is confirmed.

    ``acquire(owner)`` is idempotent for the exact same owner and instance, not
    a per-task reference counter. The manager releases once, only after all its
    consumers and training leases are positively clean. A failed release is
    retryable and preserves the owner registry rather than guessing cleanup.
    """

    def __init__(self, path: str | Path | None = None, *, guard_training_workers: bool = True):
        if type(guard_training_workers) is not bool:
            raise ValueError("training worker guard must be explicit")
        self.path = Path(path if path is not None else
                         Path.home() / ".argus" / "runtime" / "cuda0-compute.lock").expanduser().resolve()
        self.key = os.path.normcase(str(self.path))
        self.worker_path = self.path.with_name(self.path.name + ".training-worker")
        self._guard_training_workers = guard_training_workers
        self._stream = None
        self._worker_probe_stream = None
        self._owner = None
        self._held = False
        self._cleanup_pending = False

    def _clear(self):
        entry = _OWNERS.get(self.key)
        if entry is not None and entry[0] is self and entry[1] is self._owner:
            del _OWNERS[self.key]
        self._stream = None
        self._worker_probe_stream = None
        self._owner = None
        self._held = False
        self._cleanup_pending = False

    def _close_handle(self):
        complete = True
        # Keep the main guard until every secondary observation handle is
        # positively closed, including an ambiguous secondary lock attempt.
        for name in ("_worker_probe_stream", "_stream"):
            stream = getattr(self, name)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                if getattr(stream, "closed", False) is not True:
                    complete = False
                    if name == "_worker_probe_stream":
                        return False
                    continue
            setattr(self, name, None)
        return complete

    @staticmethod
    def _initialize_lock_stream(stream):
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()

    def _check_training_workers(self):
        """Parent death does not certify that its GPU child has exited.

        A child takes this independent handle before its parent authorizes any
        model loading. New main owners must reject it until the child exits.
        """
        self._worker_probe_stream = self.worker_path.open("a+b")
        self._initialize_lock_stream(self._worker_probe_stream)
        _lock_file(self._worker_probe_stream)
        self._worker_probe_stream.close()
        self._worker_probe_stream = None

    def acquire(self, owner) -> bool:
        if owner is None:
            raise ValueError("compute ownership requires an owner identity")
        with _REGISTRY_LOCK:
            entry = _OWNERS.get(self.key)
            if entry is not None:
                return (entry[0] is self and entry[1] is owner
                        and self._held and not self._cleanup_pending)
            self._owner = owner
            _OWNERS[self.key] = (self, owner)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = self.path.open("a+b")
                self._initialize_lock_stream(self._stream)
                _lock_file(self._stream)
                if self._guard_training_workers:
                    self._check_training_workers()
                self._held = True
                return True
            except Exception:
                if self._close_handle():
                    self._clear()
                else:
                    self._cleanup_pending = True
                return False

    def release(self, owner) -> bool:
        with _REGISTRY_LOCK:
            if self._owner is None:
                return True
            entry = _OWNERS.get(self.key)
            if self._owner is not owner or entry is None or entry[0] is not self or entry[1] is not owner:
                return False
            if not self._close_handle():
                self._cleanup_pending = True
                return False
            self._clear()
            return True

    def transfer(self, owner, successor) -> bool:
        """Bind an already-locked startup claim to its assembled manager.

        No handle is unlocked/reopened, so startup cannot race another process
        between early resource exclusion and runtime construction.
        """
        if successor is None:
            raise ValueError("compute ownership requires a successor identity")
        with _REGISTRY_LOCK:
            entry = _OWNERS.get(self.key)
            if (self._owner is not owner or not self._held or self._cleanup_pending
                    or entry is None or entry[0] is not self or entry[1] is not owner):
                return False
            self._owner = successor
            _OWNERS[self.key] = (self, successor)
            return True

    def snapshot(self) -> dict:
        """Cached observation only; no filesystem or process liveness probes."""
        with _REGISTRY_LOCK:
            return {"held": self._held, "cleanup_pending": self._cleanup_pending}
