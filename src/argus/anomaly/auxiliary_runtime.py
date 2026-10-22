"""Bounded execution for optional model SDKs.

A timed-out SDK call cannot safely be killed in a Python thread.  Keep its one
slot occupied until it actually finishes; never queue another frame behind it.
Workers return values only.  Owners must commit them after ``wait`` succeeds,
so shutdown and timed-out loads cannot publish a late ready state.
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_EXECUTOR_TYPE = concurrent.futures.ThreadPoolExecutor


class AuxiliaryUnavailableError(RuntimeError):
    """The requested optional model is not usable."""


class AuxiliaryBusyError(AuxiliaryUnavailableError):
    """A previous SDK call still owns the single execution slot."""


class AuxiliaryTimeoutError(TimeoutError):
    """An SDK call exceeded its deadline; its late result is discarded."""


@dataclass(eq=False)
class _Ticket:
    future: concurrent.futures.Future
    generation: int
    owner: object
    submitted_at: float
    expired: bool = False


class BoundedAuxiliaryRuntime:
    """One in-flight call, no pending queue, and terminal shutdown."""

    _leases_lock = threading.RLock()
    _leases: dict[str, object] = {}

    def __init__(self, name: str, runtime_key: str | None = None):
        if runtime_key is not None and (not isinstance(runtime_key, str) or not runtime_key.strip()):
            raise ValueError("runtime_key must be a nonempty string or None")
        self._name = name
        self._runtime_key = runtime_key
        self._lock = threading.RLock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._future: concurrent.futures.Future | None = None
        self._generation = 0
        self._closed = False
        self._owner = object()
        self._uncertain_submission: threading.Event | None = None
        self._cleanup_lock = threading.Lock()

    @property
    def cleanup_pending(self) -> bool:
        with self._lock:
            return self._closed and self._executor is not None

    @property
    def busy(self) -> bool:
        with self._lock:
            if self._uncertain_submission is not None and not self._uncertain_submission.is_set():
                return True
            if self._future is not None and not self._future.done():
                return True
            with self._leases_lock:
                return self._runtime_key is not None and self._runtime_key in self._leases

    @classmethod
    def _release_lease(cls, key: str | None, owner: object) -> None:
        if key is not None:
            with cls._leases_lock:
                if cls._leases.get(key) is owner:
                    del cls._leases[key]

    def submit(self, call: Callable[[], Any]) -> _Ticket:
        with self._lock:
            if self._closed:
                raise AuxiliaryUnavailableError(f"{self._name} is stopped")
            if self._uncertain_submission is not None and not self._uncertain_submission.is_set():
                raise AuxiliaryBusyError(f"{self._name} has an unknown SDK submission outcome")
            if self._future is not None and not self._future.done():
                raise AuxiliaryBusyError(f"{self._name} has an unfinished SDK call")
            lease_owner = object()
            with self._leases_lock:
                if self._runtime_key is not None:
                    if self._runtime_key in self._leases:
                        raise AuxiliaryBusyError(
                            f"{self._name} has an unfinished SDK call from a previous instance"
                        )
                    self._leases[self._runtime_key] = lease_owner
            completed = threading.Event()

            def execute():
                try:
                    return call()
                finally:
                    # submit() itself can fail after accepting work. Only this
                    # controlled completion proves such unknown work has ended.
                    completed.set()
                    self._release_lease(self._runtime_key, lease_owner)

            try:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix=self._name,
                    )
            except BaseException:
                self._release_lease(self._runtime_key, lease_owner)
                raise
            submitted_at = time.monotonic()
            try:
                future = self._executor.submit(execute)
            except BaseException as exc:
                # ThreadPoolExecutor rejects a known closed/broken executor
                # before enqueueing. Other submit errors have unknown outcome.
                known_refusal = (
                    type(self._executor) is _EXECUTOR_TYPE
                    and isinstance(exc, RuntimeError)
                    and (
                        getattr(self._executor, "_shutdown", False) is True
                        or bool(getattr(self._executor, "_broken", False))
                    )
                )
                if known_refusal or completed.is_set():
                    self._release_lease(self._runtime_key, lease_owner)
                else:
                    self._uncertain_submission = completed
                raise
            self._uncertain_submission = None
            self._future = future
            # Cancellation before execution has no wrapper completion, so the
            # Future callback also releases the lease after proven termination.
            key = self._runtime_key
            future.add_done_callback(lambda _: self._release_lease(key, lease_owner))
            return _Ticket(future, self._generation, self._owner, submitted_at)

    def wait(self, ticket: _Ticket, timeout: float) -> Any:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        with self._lock:
            if ticket.owner is not self._owner:
                raise AuxiliaryUnavailableError("SDK result belongs to another runtime")
            if ticket.expired or self._closed or ticket.generation != self._generation:
                raise AuxiliaryUnavailableError(f"{self._name} SDK result is no longer eligible")
        deadline = ticket.submitted_at + timeout
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise concurrent.futures.TimeoutError
            result = ticket.future.result(timeout=remaining)
            if time.monotonic() > deadline:
                raise concurrent.futures.TimeoutError
        except concurrent.futures.TimeoutError as exc:
            with self._lock:
                ticket.expired = True
            try:
                ticket.future.cancel()
            except Exception:
                # Cancellation failure cannot make a late result eligible or
                # release an in-flight cross-instance execution lease.
                pass
            raise AuxiliaryTimeoutError(
                f"{self._name} SDK call timed out after {timeout:g}s"
            ) from exc
        finally:
            with self._lock:
                if self._future is ticket.future and ticket.future.done():
                    self._future = None
        with self._lock:
            if ticket.expired or self._closed or ticket.generation != self._generation:
                raise AuxiliaryUnavailableError(
                    f"{self._name} stopped before its SDK result was accepted"
                )
        return result

    def shutdown(self) -> None:
        with self._cleanup_lock:
            with self._lock:
                if not self._closed:
                    self._closed = True
                    self._generation += 1
                if self._future is not None:
                    self._future.cancel()
                executor = self._executor
            if executor is not None:
                # Retain the handle on failure so a later stop can retry.
                executor.shutdown(wait=False, cancel_futures=True)
                with self._lock:
                    if self._executor is executor:
                        self._executor = None
