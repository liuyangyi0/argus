"""Context managers for safely switching pipeline operating modes.

Used by capture jobs (痛点 1) and training tasks (痛点 3) to put the
camera(s) into COLLECTION/TRAINING for the duration of the work, with a
hard guarantee that the previous mode is restored on exit — including
when the wrapped block raises.

Without this guard a crashing capture/training task would leave the
pipeline stuck in COLLECTION forever (the watchdog in pipeline.py is
the second line of defence, this is the first).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import structlog

from argus.core.pipeline import PipelineMode

if TYPE_CHECKING:
    from argus.capture.manager import CameraManager

logger = structlog.get_logger()


class PipelineModeGuard(AbstractContextManager["PipelineModeGuard"]):
    """Per-camera mode guard."""

    def __init__(
        self,
        camera_manager: "CameraManager",
        camera_id: str,
        target_mode: PipelineMode,
        fallback_mode: PipelineMode = PipelineMode.ACTIVE,
        reason: str = "",
    ) -> None:
        self._manager = camera_manager
        self._camera_id = camera_id
        self._target = target_mode
        self._fallback = fallback_mode
        self._reason = reason
        self._previous_mode: PipelineMode | None = None
        self._entered = False

    def __enter__(self) -> "PipelineModeGuard":
        previous_value = self._manager.get_pipeline_mode(self._camera_id)
        if previous_value is not None:
            try:
                self._previous_mode = PipelineMode(previous_value)
            except ValueError:
                self._previous_mode = self._fallback
        ok = self._manager.set_pipeline_mode(self._camera_id, self._target)
        self._entered = ok
        if ok:
            logger.info(
                "pipeline.mode_guard_enter",
                camera_id=self._camera_id,
                mode=self._target.value,
                previous=self._previous_mode.value if self._previous_mode else None,
                reason=self._reason,
            )
        else:
            logger.warning(
                "pipeline.mode_guard_enter_failed",
                camera_id=self._camera_id,
                target=self._target.value,
                reason=self._reason or "camera_not_found_or_no_pipeline",
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._entered:
            return
        restore_to = self._previous_mode or self._fallback
        try:
            self._manager.set_pipeline_mode(self._camera_id, restore_to)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "pipeline.mode_guard_restore_error",
                camera_id=self._camera_id,
                target=restore_to.value,
                error=str(e),
            )
        else:
            logger.info(
                "pipeline.mode_guard_exit",
                camera_id=self._camera_id,
                restored=restore_to.value,
                exception=exc_type.__name__ if exc_type else None,
            )


class GlobalPipelineModeGuard(AbstractContextManager["GlobalPipelineModeGuard"]):
    """Switch every camera into a target mode (training shares one GPU)."""

    def __init__(
        self,
        camera_manager: "CameraManager",
        target_mode: PipelineMode,
        fallback_mode: PipelineMode = PipelineMode.ACTIVE,
        reason: str = "",
    ) -> None:
        self._manager = camera_manager
        self._target = target_mode
        self._fallback = fallback_mode
        self._reason = reason
        self._previous: dict[str, PipelineMode] = {}
        self._entered = False

    def __enter__(self) -> "GlobalPipelineModeGuard":
        for cid in self._manager.list_camera_ids():
            value = self._manager.get_pipeline_mode(cid)
            if value is None:
                continue
            try:
                self._previous[cid] = PipelineMode(value)
            except ValueError:
                self._previous[cid] = self._fallback
            self._manager.set_pipeline_mode(cid, self._target)
        self._entered = True
        logger.info(
            "pipeline.global_mode_guard_enter",
            mode=self._target.value,
            cameras=list(self._previous.keys()),
            reason=self._reason,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._entered:
            return
        for cid, prev in self._previous.items():
            try:
                self._manager.set_pipeline_mode(cid, prev)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "pipeline.global_mode_guard_restore_error",
                    camera_id=cid,
                    error=str(e),
                )
        logger.info(
            "pipeline.global_mode_guard_exit",
            restored=len(self._previous),
            exception=exc_type.__name__ if exc_type else None,
        )
