"""Tests for PipelineModeGuard / GlobalPipelineModeGuard (PR2)."""

from unittest.mock import MagicMock

import pytest

from argus.core.pipeline import PipelineMode
from argus.core.pipeline_mode_guard import (
    GlobalPipelineModeGuard,
    PipelineModeGuard,
)


class _FakeManager:
    """Minimal CameraManager stand-in for guard tests."""

    def __init__(self, modes: dict[str, PipelineMode]):
        self.modes = dict(modes)
        self.set_calls: list[tuple[str, PipelineMode]] = []
        self.set_should_raise: dict[str, Exception] = {}

    def get_pipeline_mode(self, camera_id: str):
        m = self.modes.get(camera_id)
        return m.value if m else None

    def set_pipeline_mode(self, camera_id: str, mode: PipelineMode) -> bool:
        self.set_calls.append((camera_id, mode))
        if camera_id in self.set_should_raise:
            raise self.set_should_raise[camera_id]
        if camera_id not in self.modes:
            return False
        self.modes[camera_id] = mode
        return True

    def list_camera_ids(self):
        return list(self.modes.keys())


def test_guard_switches_into_target_then_restores_previous():
    mgr = _FakeManager({"cam_a": PipelineMode.ACTIVE})

    with PipelineModeGuard(mgr, "cam_a", PipelineMode.COLLECTION, reason="capture"):
        assert mgr.modes["cam_a"] == PipelineMode.COLLECTION

    assert mgr.modes["cam_a"] == PipelineMode.ACTIVE


def test_guard_restores_even_on_exception():
    """Critical contract: a crashing capture/training job must not leave
    the pipeline stuck in COLLECTION/TRAINING."""
    mgr = _FakeManager({"cam_a": PipelineMode.LEARNING})

    with pytest.raises(RuntimeError, match="boom"):
        with PipelineModeGuard(mgr, "cam_a", PipelineMode.COLLECTION):
            assert mgr.modes["cam_a"] == PipelineMode.COLLECTION
            raise RuntimeError("boom")

    assert mgr.modes["cam_a"] == PipelineMode.LEARNING


def test_guard_swallows_restore_errors_so_job_exception_propagates():
    """If set_pipeline_mode itself raises during restore, the guard logs
    and proceeds — original exception must still surface."""
    mgr = _FakeManager({"cam_a": PipelineMode.ACTIVE})

    def make_failing_set():
        calls = {"n": 0}

        def setter(camera_id, mode):
            calls["n"] += 1
            mgr.set_calls.append((camera_id, mode))
            if calls["n"] == 1:  # enter succeeds
                mgr.modes[camera_id] = mode
                return True
            raise RuntimeError("set_failed")  # exit raises

        return setter

    mgr.set_pipeline_mode = make_failing_set()

    with pytest.raises(RuntimeError, match="boom"):
        with PipelineModeGuard(mgr, "cam_a", PipelineMode.COLLECTION):
            raise RuntimeError("boom")

    # Two calls attempted: enter (succeeded) + exit (raised but swallowed)
    assert len(mgr.set_calls) == 2


def test_guard_no_op_when_camera_missing():
    """If camera doesn't exist set_pipeline_mode returns False; the guard
    must not attempt to restore (avoids spurious set calls on exit)."""
    mgr = _FakeManager({})

    with PipelineModeGuard(mgr, "ghost_cam", PipelineMode.COLLECTION):
        pass

    # One call attempted on enter (returned False), no exit restore
    assert mgr.set_calls == [("ghost_cam", PipelineMode.COLLECTION)]


def test_guard_unknown_previous_mode_falls_back():
    """If get_pipeline_mode returns a value outside the enum, fall back."""
    mgr = _FakeManager({"cam_a": PipelineMode.ACTIVE})
    mgr.get_pipeline_mode = lambda cid: "weird_mode" if cid == "cam_a" else None

    with PipelineModeGuard(
        mgr, "cam_a", PipelineMode.COLLECTION, fallback_mode=PipelineMode.MAINTENANCE
    ):
        assert mgr.modes["cam_a"] == PipelineMode.COLLECTION

    assert mgr.modes["cam_a"] == PipelineMode.MAINTENANCE


def test_global_guard_switches_all_then_restores_each():
    mgr = _FakeManager({
        "cam_a": PipelineMode.ACTIVE,
        "cam_b": PipelineMode.LEARNING,
        "cam_c": PipelineMode.MAINTENANCE,
    })

    with GlobalPipelineModeGuard(mgr, PipelineMode.TRAINING, reason="train_job_42"):
        assert all(m == PipelineMode.TRAINING for m in mgr.modes.values())

    assert mgr.modes == {
        "cam_a": PipelineMode.ACTIVE,
        "cam_b": PipelineMode.LEARNING,
        "cam_c": PipelineMode.MAINTENANCE,
    }


def test_global_guard_restores_on_exception():
    mgr = _FakeManager({"cam_a": PipelineMode.ACTIVE, "cam_b": PipelineMode.LEARNING})

    with pytest.raises(ValueError):
        with GlobalPipelineModeGuard(mgr, PipelineMode.TRAINING):
            raise ValueError("OOM")

    assert mgr.modes["cam_a"] == PipelineMode.ACTIVE
    assert mgr.modes["cam_b"] == PipelineMode.LEARNING


def test_global_guard_continues_even_if_one_camera_fails_to_restore():
    """One bad camera must not block the others from being restored."""
    mgr = _FakeManager({"cam_a": PipelineMode.ACTIVE, "cam_b": PipelineMode.LEARNING})
    real_set = mgr.set_pipeline_mode

    def selective_set(camera_id, mode):
        # Restore phase only: cam_a explodes, cam_b succeeds
        if camera_id == "cam_a" and mode == PipelineMode.ACTIVE:
            raise RuntimeError("cam_a restore failed")
        return real_set(camera_id, mode)

    with GlobalPipelineModeGuard(mgr, PipelineMode.TRAINING):
        mgr.set_pipeline_mode = selective_set  # swap only for exit phase

    # cam_b must still have been restored despite cam_a's error
    assert mgr.modes["cam_b"] == PipelineMode.LEARNING


def test_global_guard_with_no_cameras_is_noop():
    mgr = _FakeManager({})
    with GlobalPipelineModeGuard(mgr, PipelineMode.TRAINING):
        pass
    assert mgr.set_calls == []
