"""Explicit camera runtime and atomic model-slot contracts."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from argus.config.schema import AlertConfig
from argus.capture.manager import CameraManager, _PreparedCameraRelease
from argus.core.pipeline import DetectionPipeline
from argus.core.runner import CameraInferenceRunner
from argus.core.runtime import (
    CameraRuntime,
    FrameOutcomeKind,
    ModelIdentity,
    RuntimeModelSlot,
)


class _OutcomePipeline:
    def __init__(
        self,
        *,
        produces_frame: bool,
        detection_failed: bool = False,
        scene_usable: bool = True,
        alert: object | None = None,
        raises: Exception | None = None,
        explicit_kind: FrameOutcomeKind | None = None,
        explicit_reason: str | None = None,
    ) -> None:
        self.camera_config = SimpleNamespace(camera_id="cam_01")
        self.stats = SimpleNamespace(frames_captured=0, frames_analyzed=0)
        self.last_detection_failed = detection_failed
        self.segmenter_consecutive_failures = 0
        self.last_inference_record = None
        self._produces_frame = produces_frame
        self._scene_usable = scene_usable
        self._alert = alert
        self._raises = raises
        self.last_frame_outcome_kind = explicit_kind
        self.last_frame_outcome_reason = explicit_reason

    def run_once(self):
        if self._produces_frame:
            self.stats.frames_captured += 1
            if self._scene_usable:
                self.stats.frames_analyzed += 1
        if self._raises is not None:
            raise self._raises
        return self._alert

    def get_scene_guard_status(self) -> dict:
        return {
            "enabled": not self._scene_usable,
            "usable": self._scene_usable,
            "state": "stable" if self._scene_usable else "mismatch",
            "reason": "camera pose changed" if not self._scene_usable else "stable",
        }


@pytest.mark.parametrize(
    ("pipeline", "expected"),
    [
        (_OutcomePipeline(produces_frame=False), FrameOutcomeKind.NO_FRAME),
        (
            _OutcomePipeline(produces_frame=True, scene_usable=False),
            FrameOutcomeKind.BLOCKED,
        ),
        (
            _OutcomePipeline(produces_frame=True, detection_failed=True),
            FrameOutcomeKind.FAILED,
        ),
        (_OutcomePipeline(produces_frame=True), FrameOutcomeKind.NORMAL),
    ],
)
def test_runner_returns_explicit_non_alert_outcomes(pipeline, expected):
    runner = CameraInferenceRunner(pipeline)

    outcome = runner.run_once_outcome()

    assert outcome.kind is expected
    assert outcome.alert is None
    assert outcome.frame_captured is (expected is not FrameOutcomeKind.NO_FRAME)
    if expected in {FrameOutcomeKind.NO_FRAME, FrameOutcomeKind.BLOCKED}:
        assert runner.consecutive_failures == 0


def test_runner_alert_compatibility_uses_explicit_alert_outcome():
    alert = MagicMock()
    pipeline = _OutcomePipeline(
        produces_frame=True,
        detection_failed=True,
        scene_usable=False,
        alert=alert,
    )
    runner = CameraInferenceRunner(pipeline)

    outcome = runner.run_once_outcome()

    assert outcome.kind is FrameOutcomeKind.ALERT
    assert outcome.alert is alert

    second_alert = runner.run_once()
    assert second_alert is alert


def test_runner_converts_pipeline_exception_to_failed_outcome():
    pipeline = _OutcomePipeline(
        produces_frame=True,
        raises=RuntimeError("decoder exploded"),
    )
    runner = CameraInferenceRunner(pipeline)

    outcome = runner.run_once_outcome()

    assert outcome.kind is FrameOutcomeKind.FAILED
    assert outcome.frame_captured is True
    assert "decoder exploded" in (outcome.reason or "")
    assert runner.consecutive_failures == 1


def test_runner_preserves_pipeline_failure_that_was_caught_internally():
    pipeline = _OutcomePipeline(
        produces_frame=True,
        explicit_kind=FrameOutcomeKind.FAILED,
        explicit_reason="RuntimeError: inference worker crashed",
    )
    runner = CameraInferenceRunner(pipeline)

    outcome = runner.run_once_outcome()

    assert outcome.kind is FrameOutcomeKind.FAILED
    assert outcome.reason == "RuntimeError: inference worker crashed"
    assert runner.consecutive_failures == 1


def test_model_slot_retains_old_identity_when_reload_fails():
    slot = RuntimeModelSlot(version="v1", path="models/v1/model.pt")

    result = slot.reload(
        version="v2",
        path="models/v2/model.pt",
        reloader=lambda _path, _version: False,
    )

    assert result.success is False
    assert result.current == ModelIdentity(version="v1", path="models/v1/model.pt")
    assert slot.identity == result.current


def test_model_slot_advances_only_after_literal_success():
    slot = RuntimeModelSlot(version="v1", path="models/v1/model.pt")
    calls: list[tuple[str, str | None]] = []

    result = slot.reload(
        version="v2",
        path="models/v2/model.pt",
        reloader=lambda path, version: calls.append((path, version)) or True,
    )

    assert result.success is True
    assert calls == [("models/v2/model.pt", "v2")]
    assert slot.identity == ModelIdentity(version="v2", path="models/v2/model.pt")


def test_model_slot_carries_release_and_scene_identity_after_verified_swap():
    slot = RuntimeModelSlot(version="v1", path="models/v1/model.pt")

    result = slot.reload(
        version="v2",
        path="models/v2/model.pt",
        artifact_sha256="a" * 64,
        scene_profile_id="scene-main-v3",
        release_revision=12,
        reloader=lambda _path, _version: True,
    )

    assert result.success is True
    assert slot.identity == ModelIdentity(
        version="v2",
        path="models/v2/model.pt",
        artifact_sha256="a" * 64,
        scene_profile_id="scene-main-v3",
        release_revision=12,
    )


def test_runner_recovery_uses_current_authorized_slot_without_filesystem_scan(monkeypatch):
    pipeline = _OutcomePipeline(produces_frame=True, detection_failed=True)
    pipeline.reload_anomaly_model = MagicMock(return_value=True)
    monkeypatch.setattr(
        DetectionPipeline,
        "_find_model",
        lambda *_args, **_kwargs: pytest.fail("runner must not scan for model artifacts"),
    )
    runner = CameraInferenceRunner(
        pipeline,
        version_tag="v1",
        initial_model_path="registry/cam_01/v1/model.pt",
    )

    runner._handle_failure_threshold()

    pipeline.reload_anomaly_model.assert_called_once_with(
        "registry/cam_01/v1/model.pt",
        version_tag="v1",
    )
    assert runner.model_identity == ModelIdentity(
        version="v1",
        path="registry/cam_01/v1/model.pt",
    )


def test_runner_recovery_fails_closed_when_registry_has_no_authorized_artifact():
    pipeline = _OutcomePipeline(produces_frame=True, detection_failed=True)
    pipeline.reload_anomaly_model = MagicMock(return_value=True)
    runner = CameraInferenceRunner(
        pipeline,
        version_tag="v1",
        initial_model_path="registry/cam_01/v1/model.pt",
        authorized_model_resolver=lambda _camera_id: None,
    )

    runner._handle_failure_threshold()

    pipeline.reload_anomaly_model.assert_not_called()
    assert runner.model_identity == ModelIdentity(
        version="v1",
        path="registry/cam_01/v1/model.pt",
    )


def test_runner_adopts_complete_identity_only_after_prepared_commit():
    runner = CameraInferenceRunner(
        _OutcomePipeline(produces_frame=True),
        version_tag="v1",
        initial_model_path="registry/cam_01/v1/model.pt",
    )

    identity = runner.adopt_prepared_model_identity(
        version="v2",
        path="registry/cam_01/v2/model.pt",
        artifact_sha256="b" * 64,
        scene_profile_id="scene-main-v3",
        release_revision=4,
    )

    assert identity == ModelIdentity(
        version="v2",
        path="registry/cam_01/v2/model.pt",
        artifact_sha256="b" * 64,
        scene_profile_id="scene-main-v3",
        release_revision=4,
    )
    assert runner.model_identity is identity


def test_manager_with_registry_never_falls_back_to_filesystem_scan(monkeypatch):
    manager = CameraManager(
        cameras=[],
        alert_config=AlertConfig(),
        database=MagicMock(),
    )
    monkeypatch.setattr(manager, "_get_primary_model_record", lambda _camera_id: None)
    monkeypatch.setattr(manager, "_has_registered_model_records", lambda _camera_id: False)
    monkeypatch.setattr(
        DetectionPipeline,
        "_find_model",
        lambda *_args, **_kwargs: pytest.fail(
            "registry-backed startup must not scan model directories"
        ),
    )
    camera = SimpleNamespace(
        camera_id="cam_01",
        anomaly=SimpleNamespace(inference_backend="torch-cuda"),
    )

    assert manager._resolve_model_path(camera) is None
    manager.stop_all()


def test_manager_legacy_reload_is_always_fail_closed_and_keeps_applied_identity():
    manager = CameraManager(cameras=[], alert_config=AlertConfig())
    pipeline = _OutcomePipeline(produces_frame=True)
    pipeline.reload_anomaly_model = MagicMock(return_value=True)
    runner = CameraInferenceRunner(
        pipeline,
        version_tag="v1",
        initial_model_path="registry/cam_01/v1/model.pt",
    )
    manager._pipelines["cam_01"] = pipeline
    manager._runners["cam_01"] = runner

    first_result = manager.reload_model(
        "cam_01",
        "registry/cam_01/v2/model.pt",
        version_tag="v2",
    )
    assert first_result is False
    assert manager.get_model_slot_identity("cam_01") == ModelIdentity(
        version="v1",
        path="registry/cam_01/v1/model.pt",
    )

    second_result = manager.reload_model(
        "cam_01",
        "registry/cam_01/v3/model.pt",
        version_tag="v3",
    )
    assert second_result is False
    assert manager.get_model_slot_identity("cam_01") == ModelIdentity(
        version="v1",
        path="registry/cam_01/v1/model.pt",
    )
    pipeline.reload_anomaly_model.assert_not_called()

    manager._pipelines.clear()
    manager._runners.clear()
    manager.stop_all()


def test_camera_runtime_adapter_exposes_only_public_manager_operations():
    manager = CameraManager(cameras=[], alert_config=AlertConfig())
    manager._pipelines["cam_01"] = MagicMock()
    frame = MagicMock()
    runner_snapshot = MagicMock()
    manager.get_latest_frame = MagicMock(return_value=frame)
    manager.get_runner_snapshot = MagicMock(return_value=runner_snapshot)
    manager.get_camera_status_payload = MagicMock(return_value={"running": True})
    manager.get_scene_guard_status = MagicMock(return_value={"usable": True})
    manager.reconnect_camera = MagicMock(return_value=True)

    runtime = manager.get_runtime("cam_01")

    assert isinstance(runtime, CameraRuntime)
    assert runtime.latest_frame() is frame
    assert runtime.snapshot() is runner_snapshot
    assert runtime.status() == {"running": True}
    assert runtime.scene_guard() == {"usable": True}
    assert runtime.reconnect() is True
    assert not hasattr(runtime, "apply_release")

    manager._pipelines.clear()
    manager.stop_all()


def test_manager_exposes_dashboard_runtime_snapshots_without_runner_access():
    manager = CameraManager(cameras=[], alert_config=AlertConfig())
    detector = MagicMock()
    pipeline = MagicMock()
    pipeline.get_active_tracks_snapshot.return_value = [{"track_id": 7}]
    pipeline.get_model_statuses_snapshot.return_value = [
        {"camera_id": "cam_01", "name": "anomaly", "backend": "torch-cuda"},
    ]
    pipeline.get_runtime_model_state.return_value = {
        "primary_model_version": "v3",
        "shadow_model_version": "v4",
    }
    pipeline.get_anomaly_backend_status.return_value = {
        "actual": "torch-cuda",
        "expected": "torch-cuda",
    }
    pipeline.get_anomaly_inference_handle.return_value = detector
    pipeline.get_wall_status.return_value = {
        "current_score": 0.12,
        "score_sparkline": [0.1, 0.12],
    }
    manager._pipelines["cam_01"] = pipeline

    assert manager.get_active_tracks_snapshot("cam_01") == [{"track_id": 7}]
    assert manager.get_active_tracks_snapshot("missing") is None
    assert manager.get_model_statuses_snapshot() == [
        {"camera_id": "cam_01", "name": "anomaly", "backend": "torch-cuda"},
    ]
    assert manager.get_runtime_model_state("cam_01") == {
        "primary_model_version": "v3",
        "shadow_model_version": "v4",
    }
    assert manager.get_anomaly_backend_status("cam_01") == {
        "actual": "torch-cuda",
        "expected": "torch-cuda",
    }
    assert manager.get_anomaly_inference_handle("cam_01") is detector
    assert manager.get_wall_status("cam_01") == {
        "current_score": 0.12,
        "score_sparkline": [0.1, 0.12],
    }

    manager._pipelines.clear()
    manager.stop_all()


def test_two_phase_transaction_adopts_and_restores_complete_model_identity():
    previous = ModelIdentity(
        version="v1",
        path="registry/v1/model.pt",
        artifact_sha256="1" * 64,
        scene_profile_id="scene-v1",
        release_revision=3,
    )
    pipeline = MagicMock()
    pipeline._scene_processing_lock = threading.RLock()
    pipeline._shadow_runner_lock = threading.RLock()
    pipeline.get_shadow_runner.return_value = None
    pipeline.commit_anomaly_model_release.return_value = True
    pipeline.abort_anomaly_model_release.return_value = True
    runner = MagicMock()
    transaction = _PreparedCameraRelease(
        manager=MagicMock(),
        camera_id="cam_01",
        pipeline=pipeline,
        runner=runner,
        plan=object(),
        primary_swap=object(),
        shadow_change=False,
        previous_shadow=None,
        target_shadow=None,
        previous_identity=previous,
        target_primary_version="v2",
        target_primary_path="registry/v2/model.pt",
        target_primary_artifact_sha256="2" * 64,
        target_scene_profile_id="scene-v2",
        target_release_revision=4,
        adopt_primary_identity=True,
    )

    committed = transaction.commit()

    assert committed.success is True
    runner.adopt_prepared_model_identity.assert_called_once_with(
        version="v2",
        path="registry/v2/model.pt",
        artifact_sha256="2" * 64,
        scene_profile_id="scene-v2",
        release_revision=4,
    )

    restored = transaction.abort()

    assert restored.success is True
    runner.adopt_prepared_model_identity.assert_called_with(
        version="v1",
        path="registry/v1/model.pt",
        artifact_sha256="1" * 64,
        scene_profile_id="scene-v1",
        release_revision=3,
    )
