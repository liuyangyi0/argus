"""A partially mutated standby swap must restore identity before unlock."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from argus.capture.manager import _PreparedCameraRelease
from argus.core.pipeline import DetectionPipeline, PreparedAnomalyModelRelease
from argus.core.runtime import ModelIdentity


def _pipeline_and_swap(tmp_path, monkeypatch):
    old_path = tmp_path / "old.pt"
    new_path = tmp_path / "new.pt"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")
    old_detector, new_detector = object(), object()
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline.camera_config = SimpleNamespace(camera_id="cam-1")
    pipeline._scene_processing_lock = threading.RLock()
    pipeline._anomaly_detector = old_detector
    pipeline._anomaly_model_path = old_path
    pipeline._model_version_id = "old-version"
    pipeline._model_artifact_sha256 = "1" * 64
    pipeline._scene_guard = SimpleNamespace(
        marker="old-reference", require_reconfirmation=MagicMock(),
    )
    pipeline._scene_guard_reference_frame = object()
    pipeline._scene_guard_reference_error = None
    pipeline._scene_guard_runtime_error = None
    pipeline._scene_guard_last_usable = True
    pipeline._scene_guard_profile_id = "old-profile"
    pipeline._scene_sensitive_rois = [(1, 2, 3, 4)]
    pipeline._candidate_model_temporal = None
    pipeline._reset_scene_sensitive_state = MagicMock()
    monkeypatch.setattr(
        "argus.core.pipeline.runtime_artifact_sha256", lambda _path: "2" * 64,
    )
    swap = PreparedAnomalyModelRelease(
        owner_id=id(pipeline), target_detector=new_detector, target_path=new_path,
        target_version="new-version", target_artifact_sha256="2" * 64,
        target_scene_profile_id="new-profile", previous_detector=old_detector,
        previous_path=old_path, previous_version="old-version",
        previous_artifact_sha256="1" * 64,
    )
    return pipeline, swap


def _assert_old_identity(pipeline, swap):
    assert pipeline._anomaly_detector is swap.previous_detector
    assert pipeline._anomaly_model_path == swap.previous_path
    assert pipeline._model_version_id == swap.previous_version
    assert pipeline._model_artifact_sha256 == swap.previous_artifact_sha256
    assert pipeline._scene_guard.marker == "old-reference"
    assert pipeline._scene_guard_profile_id == "old-profile"
    assert swap.committed is False


def test_exception_after_detector_swap_restores_old_identity_and_guard(tmp_path, monkeypatch):
    pipeline, swap = _pipeline_and_swap(tmp_path, monkeypatch)
    old_reference = pipeline._scene_guard_reference_frame

    def revalidate():
        if pipeline._model_version_id == "new-version":
            pipeline._scene_guard.marker = "partially-mutated-new-reference"
            pipeline._scene_guard_reference_frame = object()
            pipeline._scene_guard_profile_id = "new-profile"
            raise RuntimeError("reference refresh failed after detector swap")

    pipeline._revalidate_scene_guard_reference = revalidate

    with pytest.raises(RuntimeError, match="after detector swap"):
        pipeline.commit_anomaly_model_release(swap)

    _assert_old_identity(pipeline, swap)
    assert pipeline._scene_guard_reference_frame is old_reference
    assert swap.rollback_error is None
    assert pipeline.abort_anomaly_model_release(swap) is True


def test_failed_guard_compensation_keeps_old_detector_but_blocks_detection(tmp_path, monkeypatch):
    pipeline, swap = _pipeline_and_swap(tmp_path, monkeypatch)
    old_reference = pipeline._scene_guard_reference_frame
    pipeline._revalidate_scene_guard_reference = MagicMock(
        side_effect=OSError("profile storage unavailable"),
    )

    with pytest.raises(OSError, match="profile storage unavailable"):
        pipeline.commit_anomaly_model_release(swap)

    _assert_old_identity(pipeline, swap)
    assert pipeline._scene_guard_reference_frame is old_reference
    assert pipeline._scene_guard_last_usable is False
    assert pipeline._scene_guard_policy_recovery_required is True
    assert "profile storage unavailable" in pipeline._scene_guard_runtime_error
    assert pipeline.abort_anomaly_model_release(swap) is False
    pipeline._reset_scene_sensitive_state.assert_called_once()


def test_exception_in_first_post_swap_reset_is_also_compensated(tmp_path, monkeypatch):
    pipeline, swap = _pipeline_and_swap(tmp_path, monkeypatch)
    pipeline._candidate_model_temporal = SimpleNamespace(
        reset=MagicMock(side_effect=[RuntimeError("temporal reset failed"), None]),
    )
    pipeline._revalidate_scene_guard_reference = MagicMock()

    with pytest.raises(RuntimeError, match="temporal reset failed"):
        pipeline.commit_anomaly_model_release(swap)

    _assert_old_identity(pipeline, swap)
    assert swap.rollback_error is None


def test_successful_swap_can_compensate_a_later_database_failure(tmp_path, monkeypatch):
    pipeline, swap = _pipeline_and_swap(tmp_path, monkeypatch)
    old_reference = pipeline._scene_guard_reference_frame

    def revalidate():
        if pipeline._model_version_id == "new-version":
            pipeline._scene_guard.marker = "new-reference"
            pipeline._scene_guard_reference_frame = object()
            pipeline._scene_guard_profile_id = "new-profile"

    pipeline._revalidate_scene_guard_reference = revalidate

    assert pipeline.commit_anomaly_model_release(swap) is True
    assert pipeline._anomaly_detector is swap.target_detector
    assert pipeline._scene_guard.marker == "new-reference"
    assert pipeline.abort_anomaly_model_release(swap) is True

    _assert_old_identity(pipeline, swap)
    assert pipeline._scene_guard_reference_frame is old_reference


def test_guard_rollback_failure_still_restores_runner_identity(tmp_path, monkeypatch):
    pipeline, swap = _pipeline_and_swap(tmp_path, monkeypatch)
    pipeline._shadow_runner_lock = threading.RLock()
    pipeline.get_shadow_runner = MagicMock(return_value=None)

    def revalidate():
        if pipeline._model_version_id == "old-version":
            raise OSError("old profile cannot be refreshed")

    pipeline._revalidate_scene_guard_reference = revalidate
    runner = MagicMock()
    old_identity = ModelIdentity(
        version="old-version", path=str(swap.previous_path),
        artifact_sha256="1" * 64, scene_profile_id="old-profile", release_revision=1,
    )
    transaction = _PreparedCameraRelease(
        manager=MagicMock(), camera_id="cam-1", pipeline=pipeline, runner=runner,
        plan=object(), primary_swap=swap, shadow_change=False,
        previous_shadow=None, target_shadow=None, previous_identity=old_identity,
        target_primary_version="new-version", target_primary_path=str(swap.target_path),
        target_primary_artifact_sha256="2" * 64, target_scene_profile_id="new-profile",
        target_release_revision=2, adopt_primary_identity=True,
    )

    assert transaction.commit().success is True
    assert transaction.abort().success is False

    _assert_old_identity(pipeline, swap)
    assert pipeline._scene_guard_last_usable is False
    runner.adopt_prepared_model_identity.assert_called_with(
        version=old_identity.version, path=old_identity.path,
        artifact_sha256=old_identity.artifact_sha256,
        scene_profile_id=old_identity.scene_profile_id,
        release_revision=old_identity.release_revision,
    )
    assert transaction.lock_held is False
    assert transaction.shadow_lock_held is False
