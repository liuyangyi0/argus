"""Durable release queue execution outside the HTTP event loop.

SQLite is a single-runtime deployment: one process owns the release worker for
one database. OS locks are released by the kernel on process death; a second
worker cannot mistake an active process's commands for interrupted commands.
Each camera also has a lock shared with synchronous internal release callers.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from argus.storage.model_release_service import ModelReleaseService

logger = structlog.get_logger()


class ReleaseExecutionBusy(RuntimeError):
    """Another process or thread currently owns this execution scope."""


class ReleaseExecutionLock:
    """Non-blocking process lock, with a per-engine lock for in-memory tests."""

    _guard = threading.Lock()
    _memory_locks: dict[tuple[int, str], threading.Lock] = {}

    def __init__(self, session_factory, scope: str) -> None:
        with session_factory() as session:
            bind = session.get_bind()
            engine = getattr(bind, "engine", bind)
            database = engine.url.database
            if engine.url.get_backend_name() != "sqlite":
                raise RuntimeError("release execution requires a SQLite runtime database")
            engine_id = id(engine)
        self._path: Path | None = None
        self._file = None
        self._memory_lock: threading.Lock | None = None
        if database and database != ":memory:":
            digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
            db_path = Path(database).resolve()
            self._path = db_path.parent / f".{db_path.name}.release-{digest}.lock"
        else:
            with self._guard:
                self._memory_lock = self._memory_locks.setdefault(
                    (engine_id, scope), threading.Lock(),
                )

    def acquire(self) -> None:
        if self._memory_lock is not None:
            if not self._memory_lock.acquire(blocking=False):
                raise ReleaseExecutionBusy("release execution is already owned")
            return
        assert self._path is not None
        lock_file = self._path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if lock_file.seek(0, os.SEEK_END) == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            raise ReleaseExecutionBusy("another process owns release execution") from exc
        self._file = lock_file

    def release(self) -> None:
        if self._memory_lock is not None:
            self._memory_lock.release()
            return
        lock_file, self._file = self._file, None
        if lock_file is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


class ModelReleaseWorker:
    """One managed thread scans durable desired operations, never HTTP tasks.

    Deploy one dashboard/runtime process per SQLite database on a local
    filesystem. Startup rejects a second owner. Shutdown waits for the current
    prepare/swap/compensation transaction; forcibly killing a slow GPU loader
    requires a process restart and its pending approval will be failed during
    recovery, never automatically replayed.
    """

    def __init__(self, service: ModelReleaseService, *, poll_seconds: float = 1.0):
        self._service = service
        self._poll_seconds = poll_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lease = service.execution_lock("worker")
        self._lifecycle_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.running:
                return
            self._lease.acquire()
            try:
                # Never replay approvals after a restart. Applied rows are
                # restored by the registry-authorized runtime startup path.
                self._service.reconcile_interrupted_operations()
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._run,
                    name="model-release-worker",
                    daemon=True,
                )
                self._thread.start()
            except BaseException:
                self._lease.release()
                raise

    def notify(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._thread is None:
                return
            self._stop.set()
            self._wake.set()
            # Let the two-phase transaction finish or compensate before the
            # camera manager/database are shut down by the caller.
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._wake.clear()
                try:
                    operations = self._service.list_pending_operations()
                except Exception:
                    logger.exception("model_release.worker_queue_read_failed")
                    self._wake.wait(self._poll_seconds)
                    continue
                for operation in operations:
                    if self._stop.is_set():
                        break
                    try:
                        self._service.execute_operation(operation.operation_id)
                    except ReleaseExecutionBusy:
                        continue
                    except Exception:
                        logger.exception(
                            "model_release.worker_operation_failed",
                            operation_id=operation.operation_id,
                        )
                self._wake.wait(self._poll_seconds)
        except BaseException:
            logger.exception("model_release.worker_stopped_unexpectedly")
            try:
                self._service.reconcile_interrupted_operations()
            except Exception:
                logger.exception("model_release.worker_recovery_failed")
        finally:
            self._lease.release()
