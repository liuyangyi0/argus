"""Stable public runtime contracts for camera inference.

The dashboard and orchestration layers should depend on the small interfaces in
this module instead of reaching into ``DetectionPipeline`` internals.  The
concrete pipeline remains intentionally hidden behind ``CameraRuntime``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from argus.alerts.grader import Alert
    from argus.core.runner import RunnerSnapshot


class FrameOutcomeKind(str, Enum):
    """Unambiguous result of offering one frame to a camera runner."""

    NO_FRAME = "no_frame"
    BLOCKED = "blocked"
    NORMAL = "normal"
    ALERT = "alert"
    FAILED = "failed"


@dataclass(frozen=True)
class FrameOutcome:
    """Immutable result of one runner iteration.

    ``Alert | None`` alone cannot distinguish a disconnected camera, a
    SceneGuard suppression, a successful normal frame, and an inference
    failure.  The explicit kind keeps those states distinct while ``alert``
    preserves the existing return value for compatibility.
    """

    kind: FrameOutcomeKind
    alert: Alert | None
    frames_captured_before: int
    frames_captured_after: int
    reason: str | None = None
    scene_guard_state: str | None = None

    @property
    def frame_captured(self) -> bool:
        return self.frames_captured_after > self.frames_captured_before


@dataclass(frozen=True)
class ModelIdentity:
    """Identity of a model applied by or proposed to a camera runtime."""

    version: str | None
    path: str | None
    artifact_sha256: str | None = None
    scene_profile_id: str | None = None
    release_revision: int | None = None


@dataclass(frozen=True)
class PreparedModelReload:
    """A reload request bound to the slot identity observed at prepare time."""

    previous: ModelIdentity
    target: ModelIdentity


@dataclass(frozen=True)
class ModelReloadResult:
    """Atomic model reload result; failure always retains ``previous``."""

    success: bool
    previous: ModelIdentity
    current: ModelIdentity
    attempted: ModelIdentity
    error: str | None = None


class RuntimeModelSlot:
    """Thread-safe owner of a camera runtime's applied model identity.

    The slot serializes prepare/commit and advances identity only after the
    detector reports literal ``True``.  A failed or stale request therefore
    cannot make status claim that an unapplied model is active.
    """

    def __init__(
        self,
        *,
        version: str | None = None,
        path: str | None = None,
        artifact_sha256: str | None = None,
        scene_profile_id: str | None = None,
        release_revision: int | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._identity = ModelIdentity(
            version=version,
            path=path,
            artifact_sha256=artifact_sha256,
            scene_profile_id=scene_profile_id,
            release_revision=release_revision,
        )

    @property
    def identity(self) -> ModelIdentity:
        with self._lock:
            return self._identity

    def adopt(
        self,
        *,
        version: str | None,
        path: str | None,
        artifact_sha256: str | None = None,
        scene_profile_id: str | None = None,
        release_revision: int | None = None,
    ) -> ModelIdentity:
        """Synchronize identity after a separately verified initialization."""
        with self._lock:
            self._identity = ModelIdentity(
                version=version,
                path=path,
                artifact_sha256=artifact_sha256,
                scene_profile_id=scene_profile_id,
                release_revision=release_revision,
            )
            return self._identity

    def prepare(
        self,
        *,
        version: str | None,
        path: str,
        artifact_sha256: str | None = None,
        scene_profile_id: str | None = None,
        release_revision: int | None = None,
    ) -> PreparedModelReload:
        """Bind a target to the currently applied identity."""
        target = ModelIdentity(
            version=version,
            path=str(path),
            artifact_sha256=artifact_sha256,
            scene_profile_id=scene_profile_id,
            release_revision=release_revision,
        )
        with self._lock:
            return PreparedModelReload(previous=self._identity, target=target)

    def apply(
        self,
        prepared: PreparedModelReload,
        reloader: Callable[[str, str | None], bool],
    ) -> ModelReloadResult:
        """Apply ``prepared`` atomically, retaining the old identity on failure."""
        with self._lock:
            if self._identity != prepared.previous:
                return ModelReloadResult(
                    success=False,
                    previous=prepared.previous,
                    current=self._identity,
                    attempted=prepared.target,
                    error="model slot changed after reload was prepared",
                )
            try:
                success = reloader(prepared.target.path or "", prepared.target.version)
            except Exception as exc:
                return ModelReloadResult(
                    success=False,
                    previous=prepared.previous,
                    current=self._identity,
                    attempted=prepared.target,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if success is not True:
                return ModelReloadResult(
                    success=False,
                    previous=prepared.previous,
                    current=self._identity,
                    attempted=prepared.target,
                    error="runtime model reload returned failure",
                )
            self._identity = prepared.target
            return ModelReloadResult(
                success=True,
                previous=prepared.previous,
                current=self._identity,
                attempted=prepared.target,
            )

    def reload(
        self,
        *,
        version: str | None,
        path: str,
        artifact_sha256: str | None = None,
        scene_profile_id: str | None = None,
        release_revision: int | None = None,
        reloader: Callable[[str, str | None], bool],
    ) -> ModelReloadResult:
        """Prepare and apply a target through one convenience operation."""
        return self.apply(
            self.prepare(
                version=version,
                path=path,
                artifact_sha256=artifact_sha256,
                scene_profile_id=scene_profile_id,
                release_revision=release_revision,
            ),
            reloader,
        )


@runtime_checkable
class CameraRuntime(Protocol):
    """Public per-camera runtime surface for orchestration and dashboards."""

    @property
    def camera_id(self) -> str:
        ...

    def latest_frame(self) -> np.ndarray | None:
        ...

    def snapshot(self) -> RunnerSnapshot | None:
        ...

    def status(self) -> dict | None:
        ...

    def scene_guard(self) -> dict | None:
        ...

    def reconnect(self) -> bool:
        ...


class CameraRuntimeAdapter:
    """Callback-backed adapter that keeps ``CameraManager`` behind a Protocol."""

    def __init__(
        self,
        *,
        camera_id: str,
        latest_frame: Callable[[], np.ndarray | None],
        snapshot: Callable[[], RunnerSnapshot | None],
        status: Callable[[], dict | None],
        scene_guard: Callable[[], dict | None],
        reconnect: Callable[[], bool],
    ) -> None:
        self._camera_id = camera_id
        self._latest_frame = latest_frame
        self._snapshot = snapshot
        self._status = status
        self._scene_guard = scene_guard
        self._reconnect = reconnect

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def latest_frame(self) -> np.ndarray | None:
        return self._latest_frame()

    def snapshot(self) -> RunnerSnapshot | None:
        return self._snapshot()

    def status(self) -> dict | None:
        return self._status()

    def scene_guard(self) -> dict | None:
        return self._scene_guard()

    def reconnect(self) -> bool:
        return self._reconnect()
