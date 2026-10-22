"""Integration contracts for fixed-scene gating in DetectionPipeline."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from argus.anomaly.detector import AnomalyResult
from argus.anomaly.candidate_model_evidence import (
    build_calibration_id,
    prepare_model_evidence_map,
)
from argus.anomaly.persistent_change import (
    PersistentChangeCandidate,
    PersistentChangeResult,
)
from argus.capture.camera import FrameData
from argus.config.schema import (
    AlertConfig,
    CandidateModelEvidenceConfig,
    CameraConfig,
    LowLightConfig,
    PersistentChangeConfig,
    SceneGuardConfig,
    StaticCandidateConfirmationConfig,
)
from argus.core.diagnostics import FrameDiagnostics
from argus.core.pipeline import (
    DetectionPipeline,
    PipelineMode,
    _frame_content_sha256,
)
from argus.person.detector import ObjectDetectionResult
from argus.prefilter.mog2 import PreFilterResult
from argus.preprocessing.scene_profile import (
    RUNTIME_ARTIFACT_HASH_SCHEME,
    SceneProfileStore,
    runtime_artifact_sha256,
)


def _feature_scene(seed: int = 7, *, width: int = 320, height: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    for _ in range(90):
        x = int(rng.integers(8, width - 8))
        y = int(rng.integers(8, height - 8))
        color = tuple(int(value) for value in rng.integers(80, 255, size=3))
        cv2.circle(frame, (x, y), int(rng.integers(2, 6)), color, -1)
    cv2.putText(frame, "ARGUS-CAM-01", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, (12, 12), (width - 13, height - 13), (150, 210, 80), 2)
    return frame


def _camera_config(
    *,
    scene_guard: SceneGuardConfig | None = None,
    static_confirmation: StaticCandidateConfirmationConfig | None = None,
) -> CameraConfig:
    return CameraConfig(
        camera_id="scene-cam",
        name="Scene Camera",
        source="test.mp4",
        protocol="file",
        fps_target=60,
        resolution=(320, 240),
        low_light=LowLightConfig(enabled=False),
        scene_guard=scene_guard or SceneGuardConfig(enabled=False),
        static_confirmation=(
            static_confirmation or StaticCandidateConfirmationConfig(enabled=False)
        ),
        persistent_change=PersistentChangeConfig(
            enabled=bool(
                static_confirmation is not None
                and static_confirmation.enabled
                and static_confirmation.mode == "enforce"
            )
        ),
    )


def _build_pipeline(
    camera: CameraConfig,
    *,
    model_version_id: str | None = "model-v1",
    model_path=None,
    shared_anomaly_detector=None,
) -> DetectionPipeline:
    with (
        patch("argus.core.pipeline.MOG2PreFilter"),
        patch("argus.core.pipeline.YOLOObjectDetector"),
        patch("argus.core.pipeline.AnomalibDetector"),
        patch("argus.core.pipeline.AlertGrader"),
        patch("argus.core.pipeline.CameraCapture"),
        patch("argus.core.pipeline.ZoneMaskEngine") as zone_class,
        patch("argus.core.pipeline.METRICS"),
    ):
        zone = MagicMock()
        zone.apply.side_effect = lambda frame: frame
        zone.get_include_zones.return_value = []
        zone_class.return_value = zone
        pipeline = DetectionPipeline(
            camera_config=camera,
            alert_config=AlertConfig(),
            model_version_id=model_version_id,
            model_path=model_path,
            shared_anomaly_detector=shared_anomaly_detector,
        )

    pipeline._mode = PipelineMode.ACTIVE
    pipeline._learning_start_time = None
    pipeline._camera = MagicMock()
    pipeline._camera.state = SimpleNamespace(reconnect_count=0)
    pipeline._zone_mask.apply.side_effect = lambda frame: frame
    pipeline._prefilter.process.return_value = PreFilterResult(
        has_change=False,
        change_ratio=0.0,
    )
    return pipeline


def _frame_data(frame: np.ndarray, number: int) -> FrameData:
    return FrameData(
        frame=frame,
        camera_id="scene-cam",
        timestamp=time.monotonic(),
        frame_number=number,
        resolution=(frame.shape[1], frame.shape[0]),
    )


def _trusted_profile_metadata(
    model_path,
    *,
    model_version_id: str = "model-v1",
    operator_approved: object = True,
    validation_only: object = False,
) -> dict[str, object]:
    """Build the complete authority/runtime binding used by online profiles."""
    return {
        "model_version_id": model_version_id,
        "model_artifact_sha256": runtime_artifact_sha256(model_path),
        "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
        "operator_approved": operator_approved,
        "validation_only": validation_only,
    }


def _initialize_loaded_model(pipeline: DetectionPipeline) -> None:
    pipeline._camera.connect.return_value = True
    pipeline._anomaly_detector.load.return_value = True
    pipeline._anomaly_detector.get_status.return_value = SimpleNamespace(
        minmax_broken=False
    )
    assert pipeline.initialize() is True


def test_explicit_no_model_does_not_run_secondary_filesystem_discovery() -> None:
    camera = _camera_config()
    with patch.object(DetectionPipeline, "_find_model") as discover:
        with (
            patch("argus.core.pipeline.MOG2PreFilter"),
            patch("argus.core.pipeline.YOLOObjectDetector"),
            patch("argus.core.pipeline.AnomalibDetector") as detector,
            patch("argus.core.pipeline.AlertGrader"),
            patch("argus.core.pipeline.CameraCapture"),
            patch("argus.core.pipeline.ZoneMaskEngine"),
            patch("argus.core.pipeline.METRICS"),
        ):
            DetectionPipeline(
                camera_config=camera,
                alert_config=AlertConfig(),
                model_path=None,
                auto_discover_model=False,
            )

    discover.assert_not_called()
    assert detector.call_args.kwargs["model_path"] is None


def test_missing_reference_blocks_before_any_detector_and_keeps_preview() -> None:
    camera = _camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        consecutive_matches=1,
    ))
    pipeline = _build_pipeline(camera)
    frame = _feature_scene()

    assert pipeline.process_frame(_frame_data(frame, 1)) is None

    pipeline._prefilter.process.assert_not_called()
    assert np.array_equal(pipeline.get_latest_frame(), frame)
    status = pipeline.get_scene_guard_status()
    assert status["state"] == "unknown"
    assert status["usable"] is False
    diag = pipeline.get_diagnostics_buffer().get_recent(1)[0]
    assert diag.scene_guard_state == "unknown"
    assert diag.candidate_records[-1].suppression_reason == "scene_guard:unknown"


def test_enabled_scene_guard_without_runtime_instance_fails_closed() -> None:
    pipeline = _build_pipeline(_camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        consecutive_matches=1,
    )))
    frame = _feature_scene()
    pipeline._scene_guard = None

    with patch("argus.core.pipeline.get_error_channel"):
        assert pipeline.process_frame(_frame_data(frame, 1)) is None

    pipeline._prefilter.process.assert_not_called()
    pipeline._anomaly_detector.predict.assert_not_called()
    pipeline._alert_grader.evaluate.assert_not_called()
    assert np.array_equal(pipeline.get_latest_frame(), frame)

    status = pipeline.get_scene_guard_status()
    assert status["enabled"] is True
    assert status["state"] == "blocked"
    assert status["usable"] is False
    assert status["reference_available"] is False

    diag = pipeline.get_diagnostics_buffer().get_recent(1)[0]
    assert diag.final_decision == "skipped"
    assert diag.scene_guard_state == "blocked"
    scene_stage = next(stage for stage in diag.stages if stage.stage_name == "scene_guard")
    assert scene_stage.skipped is True
    assert scene_stage.skip_reason == "guard_unavailable"
    assert diag.candidate_records[-1].decision == "blocked"
    assert diag.candidate_records[-1].suppression_reason == "scene_guard:blocked"


def test_explicitly_disabled_scene_guard_without_runtime_instance_passes() -> None:
    pipeline = _build_pipeline(_camera_config())
    frame = _feature_scene()
    diag = FrameDiagnostics(1, time.time(), "scene-cam")

    output, usable = pipeline._apply_scene_guard(
        frame,
        _frame_data(frame, 1),
        diag,
    )

    assert usable is True
    assert output is frame
    assert diag.scene_guard_state == "disabled"
    assert diag.stages == []


def test_scene_isolation_and_recovery_are_published_once_per_transition() -> None:
    pipeline = _build_pipeline(_camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        consecutive_matches=1,
    )))
    pipeline._event_bus = MagicMock()
    pipeline._degradation_publisher = MagicMock()
    reference = _feature_scene()

    with patch("argus.core.pipeline.get_error_channel"):
        pipeline.process_frame(_frame_data(reference, 1))
        pipeline.process_frame(_frame_data(reference, 2))
        assert pipeline._event_bus.publish.call_count == 1
        assert (
            pipeline._event_bus.publish.call_args.args[0]
            == "pipeline.scene_guard_blocked"
        )

        pipeline.set_scene_reference(reference, source="test:approved")
        pipeline.process_frame(_frame_data(reference, 3))

        assert pipeline._event_bus.publish.call_count == 2
        assert (
            pipeline._event_bus.publish.call_args.args[0]
            == "pipeline.scene_guard_recovered"
        )
        assert pipeline._degradation_publisher.call_count == 2


def test_guard_releases_after_safe_streak_then_blocks_bad_frame_immediately() -> None:
    camera = _camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        consecutive_matches=2,
        check_interval_frames=60,
    ))
    pipeline = _build_pipeline(camera)
    reference = _feature_scene()
    pipeline.set_scene_reference(reference, source="test:approved")

    assert pipeline.process_frame(_frame_data(reference, 1)) is None
    pipeline._prefilter.process.assert_not_called()

    assert pipeline.process_frame(_frame_data(reference, 2)) is None
    assert pipeline._prefilter.process.call_count == 1
    assert pipeline.get_scene_guard_status()["usable"] is True

    pipeline._prefilter.process.reset_mock()
    blocked = np.zeros_like(reference)
    assert pipeline.process_frame(_frame_data(blocked, 3)) is None

    pipeline._prefilter.process.assert_not_called()
    status = pipeline.get_scene_guard_status()
    assert status["state"] == "blocked"
    assert status["usable"] is False

    assert pipeline.process_frame(_frame_data(reference, 4)) is None
    pipeline._prefilter.process.assert_not_called()
    assert pipeline.get_scene_guard_status()["state"] == "calibrating"

    assert pipeline.process_frame(_frame_data(reference, 5)) is None
    assert pipeline._prefilter.process.call_count == 1
    assert pipeline.get_scene_guard_status()["usable"] is True


def test_active_profile_must_match_model_and_current_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "normal.jpg"
    assert cv2.imwrite(str(baseline_path), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"trusted runtime model")

    profiles_dir = tmp_path / "profiles"
    store = SceneProfileStore(profiles_dir)
    store.create_from_baseline(
        "scene-cam",
        "approved-v1",
        baseline_path,
        metadata=_trusted_profile_metadata(model_path),
        activate=True,
    )
    camera = _camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        profiles_dir=str(profiles_dir),
        fallback_to_baseline=False,
        consecutive_matches=1,
    ))

    matching = _build_pipeline(
        camera,
        model_version_id="model-v1",
        model_path=model_path,
    )
    # Constructor-time resolution cannot authorize an artifact that has not
    # yet been loaded and fingerprinted on both sides of the load operation.
    assert matching.get_scene_guard_status()["reference_available"] is False
    _initialize_loaded_model(matching)
    matching_status = matching.get_scene_guard_status()
    assert matching_status["reference_error"] is None
    assert matching_status["bound_model_version"] == "model-v1"
    assert matching_status["runtime_model_artifact_sha256"] == (
        runtime_artifact_sha256(model_path)
    )

    mismatching = _build_pipeline(
        camera,
        model_version_id="model-v2",
        model_path=model_path,
    )
    _initialize_loaded_model(mismatching)
    mismatch_status = mismatching.get_scene_guard_status()
    assert mismatch_status["usable"] is False
    assert "profile=model-v1, runtime=model-v2" in mismatch_status["reference_error"]


@pytest.mark.parametrize(
    ("operator_approved", "validation_only", "expected_reason"),
    [
        ("true", False, "not explicitly operator-approved"),
        (1, False, "not explicitly operator-approved"),
        (True, True, "validation-only"),
        (True, "false", "invalid validation_only"),
    ],
)
def test_online_profile_authority_requires_literal_approval_and_non_validation(
    tmp_path,
    monkeypatch,
    operator_approved,
    validation_only,
    expected_reason,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "normal.jpg"
    assert cv2.imwrite(str(baseline_path), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"trusted runtime model")
    profiles_dir = tmp_path / "profiles"
    SceneProfileStore(profiles_dir).create_from_baseline(
        "scene-cam",
        "authority-case",
        baseline_path,
        metadata=_trusted_profile_metadata(
            model_path,
            operator_approved=operator_approved,
            validation_only=validation_only,
        ),
        activate=True,
    )
    pipeline = _build_pipeline(
        _camera_config(scene_guard=SceneGuardConfig(
            enabled=True,
            profiles_dir=str(profiles_dir),
        )),
        model_version_id="model-v1",
        model_path=model_path,
    )

    _initialize_loaded_model(pipeline)

    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is False
    assert expected_reason in status["reference_error"]


def test_validation_run_must_explicitly_opt_out_of_operator_approval_gate(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "normal.jpg"
    assert cv2.imwrite(str(baseline_path), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"validation runtime model")
    profiles_dir = tmp_path / "profiles"
    SceneProfileStore(profiles_dir).create_from_baseline(
        "scene-cam",
        "validation-only",
        baseline_path,
        metadata=_trusted_profile_metadata(
            model_path,
            operator_approved=False,
            validation_only=True,
        ),
        activate=True,
    )
    pipeline = _build_pipeline(
        _camera_config(scene_guard=SceneGuardConfig(
            enabled=True,
            profiles_dir=str(profiles_dir),
            require_operator_approved_profile=False,
        )),
        model_version_id="model-v1",
        model_path=model_path,
    )

    _initialize_loaded_model(pipeline)

    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is True
    assert status["profile_id"] == "validation-only"
    assert status["reference_error"] is None


@pytest.mark.parametrize(
    ("metadata_mutation", "expected_reason"),
    [
        ("missing_digest", "no valid model_artifact_sha256"),
        ("wrong_digest", "runtime artifact mismatch"),
        ("missing_scheme", "does not use the Argus runtime artifact hash scheme"),
    ],
)
def test_online_profile_rejects_missing_or_wrong_runtime_artifact_binding(
    tmp_path,
    monkeypatch,
    metadata_mutation,
    expected_reason,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "normal.jpg"
    assert cv2.imwrite(str(baseline_path), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"trusted runtime model")
    metadata = _trusted_profile_metadata(model_path)
    if metadata_mutation == "missing_digest":
        metadata.pop("model_artifact_sha256")
    elif metadata_mutation == "wrong_digest":
        metadata["model_artifact_sha256"] = "b" * 64
    else:
        metadata.pop("model_artifact_hash_scheme")
    profiles_dir = tmp_path / "profiles"
    SceneProfileStore(profiles_dir).create_from_baseline(
        "scene-cam",
        "artifact-case",
        baseline_path,
        metadata=metadata,
        activate=True,
    )
    pipeline = _build_pipeline(
        _camera_config(scene_guard=SceneGuardConfig(
            enabled=True,
            profiles_dir=str(profiles_dir),
        )),
        model_version_id="model-v1",
        model_path=model_path,
    )

    _initialize_loaded_model(pipeline)

    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is False
    assert expected_reason in status["reference_error"]


def test_required_profile_binding_never_falls_back_to_unbound_baseline(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    assert cv2.imwrite(str(baseline_dir / "normal.jpg"), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    camera = _camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        profiles_dir=str(tmp_path / "profiles"),
        fallback_to_baseline=True,
        require_profile_model_binding=True,
        consecutive_matches=1,
    ))

    pipeline = _build_pipeline(camera, model_version_id="model-v1")

    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is False
    assert status["usable"] is False
    assert status["profile_id"] is None
    assert "no active" in status["reference_error"]
    assert "scene profile" in status["reference_error"]


def test_alignable_pose_is_diagnostic_only_by_default() -> None:
    camera = _camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        allow_alignable=False,
        consecutive_matches=1,
        stable_max_shift_px=5.0,
        alignable_max_shift_px=32.0,
    ))
    pipeline = _build_pipeline(camera)
    reference = _feature_scene()
    transform = np.float32([[1.0, 0.0, 18.0], [0.0, 1.0, 7.0]])
    shifted = cv2.warpAffine(
        reference,
        transform,
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )
    pipeline.set_scene_reference(reference, source="test:approved")

    assert pipeline.process_frame(_frame_data(shifted, 1)) is None

    pipeline._prefilter.process.assert_not_called()
    status = pipeline.get_scene_guard_status()
    assert status["state"] == "alignable"
    assert status["usable"] is False
    assert "homography reuse is disabled" in status["reason"]
    diagnostic = pipeline.get_diagnostics_buffer().get_recent(1)[0]
    scene_stage = next(
        stage for stage in diagnostic.stages if stage.stage_name == "scene_guard"
    )
    assert scene_stage.metadata["geometry_normalization_applied"] is False
    assert scene_stage.metadata["warp_border_policy"] == "not_applicable"


def test_fresh_alignable_check_warps_downstream_prefilter_and_model_input() -> None:
    from concurrent.futures import Future

    camera = _camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        allow_alignable=True,
        consecutive_matches=1,
        stable_max_shift_px=5.0,
        alignable_max_shift_px=32.0,
        check_interval_frames=600,
        check_interval_seconds=60.0,
    ))
    pipeline = _build_pipeline(camera)
    reference = _feature_scene()
    shifted = cv2.warpAffine(
        reference,
        np.float32([[1.0, 0.0, 18.0], [0.0, 1.0, 7.0]]),
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )
    pipeline.set_scene_reference(reference, source="test:approved")
    pipeline._last_heartbeat_time = 0.0
    pipeline._prefilter.process.return_value = PreFilterResult(
        has_change=True,
        change_ratio=0.1,
    )
    pipeline._object_detector.detect.return_value = ObjectDetectionResult(
        has_persons=False
    )
    pipeline._anomaly_detector.get_status.return_value = SimpleNamespace(
        mode="torch",
        model_loaded=True,
        ssim_calibrated=True,
    )
    pipeline._anomaly_detector.predict.return_value = AnomalyResult(
        anomaly_score=0.1,
        anomaly_map=None,
        is_anomalous=False,
        threshold=0.7,
    )

    def completed_future(function, *args):
        future = Future()
        future.set_result(function(*args))
        return future

    with patch.object(pipeline, "_inference_executor") as executor:
        executor.submit.side_effect = completed_future
        assert pipeline.process_frame(_frame_data(shifted, 1)) is None

    scene_result = pipeline._scene_guard_last_result
    assert scene_result is not None
    assert scene_result.state.value == "alignable"
    assert scene_result.homography is not None
    raw_aligned = cv2.warpPerspective(
        shifted,
        scene_result.homography,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    expected = pipeline._fill_scene_warp_border(
        raw_aligned,
        scene_result.homography,
        shifted.shape,
    )
    prefilter_input = pipeline._prefilter.process.call_args.args[0]
    model_input = pipeline._anomaly_detector.predict.call_args.args[0]
    np.testing.assert_array_equal(prefilter_input, expected)
    np.testing.assert_array_equal(model_input, expected)
    assert not np.array_equal(prefilter_input, shifted)

    diagnostic = pipeline.get_diagnostics_buffer().get_recent(1)[0]
    scene_stage = next(
        stage for stage in diagnostic.stages if stage.stage_name == "scene_guard"
    )
    assert scene_stage.metadata["checked"] is True
    assert scene_stage.metadata["geometry_normalization_applied"] is True
    assert scene_stage.metadata["warp_border_policy"] == (
        "approved_reference_fill_outside_source_footprint"
    )


def test_policy_recovery_counts_only_fresh_geometry_checks() -> None:
    pipeline = _build_pipeline(_camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        consecutive_matches=3,
    )))
    pipeline._scene_guard_policy_recovery_required = True
    stable = SimpleNamespace(
        state=SimpleNamespace(value="stable"),
        usable=True,
    )

    assert pipeline._scene_policy_usable(stable, checked=False) is False
    assert pipeline._scene_guard_policy_recovery_count == 0
    assert pipeline._scene_policy_usable(stable, checked=True) is False
    assert pipeline._scene_guard_policy_recovery_count == 1
    assert pipeline._scene_policy_usable(stable, checked=False) is False
    assert pipeline._scene_guard_policy_recovery_count == 1
    assert pipeline._scene_policy_usable(stable, checked=True) is False
    assert pipeline._scene_policy_usable(stable, checked=True) is True


def test_capture_reconnect_requires_fresh_geometry_streak() -> None:
    pipeline = _build_pipeline(_camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        consecutive_matches=3,
        check_interval_frames=600,
        check_interval_seconds=60.0,
    )))
    reference = _feature_scene()
    pipeline.set_scene_reference(reference, source="test:approved")

    for number in range(1, 4):
        pipeline.process_frame(_frame_data(reference, number))
    assert pipeline.get_scene_guard_status()["usable"] is True
    calls_before_reconnect = pipeline._prefilter.process.call_count

    pipeline._camera.state.reconnect_count = 1
    pipeline.process_frame(_frame_data(reference, 4))
    assert pipeline.get_scene_guard_status()["state"] == "calibrating"
    assert pipeline.get_scene_guard_status()["usable"] is False
    assert pipeline._prefilter.process.call_count == calls_before_reconnect

    pipeline.process_frame(_frame_data(reference, 5))
    assert pipeline.get_scene_guard_status()["usable"] is False
    pipeline.process_frame(_frame_data(reference, 6))
    assert pipeline.get_scene_guard_status()["state"] == "stable"
    assert pipeline.get_scene_guard_status()["usable"] is True
    assert pipeline._prefilter.process.call_count == calls_before_reconnect + 1


def test_reconnect_policy_counts_only_raw_stable_when_alignment_is_disabled() -> None:
    pipeline = _build_pipeline(_camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        allow_alignable=False,
        consecutive_matches=3,
        stable_max_shift_px=5.0,
        alignable_max_shift_px=32.0,
        check_interval_frames=600,
        check_interval_seconds=60.0,
    )))
    reference = _feature_scene()
    shifted = cv2.warpAffine(
        reference,
        np.float32([[1.0, 0.0, 16.0], [0.0, 1.0, 3.0]]),
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )
    pipeline.set_scene_reference(reference, source="test:approved")
    for number in range(1, 4):
        pipeline.process_frame(_frame_data(reference, number))
    assert pipeline.get_scene_guard_status()["usable"] is True
    calls_before_reconnect = pipeline._prefilter.process.call_count

    pipeline._camera.state.reconnect_count = 1
    pipeline.process_frame(_frame_data(shifted, 4))
    pipeline.process_frame(_frame_data(shifted, 5))
    pipeline.process_frame(_frame_data(reference, 6))

    # The algorithm has three geometrically safe observations, but only the
    # latest one is policy-valid STABLE.  Default unwarped inference stays
    # isolated until three consecutive raw STABLE checks have been observed.
    status = pipeline.get_scene_guard_status()
    assert status["state"] == "stable"
    assert status["usable"] is False
    assert status["policy_recovery_count"] == 1
    assert pipeline._prefilter.process.call_count == calls_before_reconnect

    pipeline.process_frame(_frame_data(reference, 7))
    assert pipeline.get_scene_guard_status()["usable"] is False
    pipeline.process_frame(_frame_data(reference, 8))
    recovered = pipeline.get_scene_guard_status()
    assert recovered["usable"] is True
    assert recovered["recovery_progress"] == {
        "current": 3,
        "required": 3,
        "complete": True,
    }
    assert pipeline._prefilter.process.call_count == calls_before_reconnect + 1


def test_cached_stable_scene_does_not_run_full_resolution_homography() -> None:
    pipeline = _build_pipeline(_camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        allow_alignable=False,
        consecutive_matches=1,
        check_interval_frames=60,
        check_interval_seconds=10.0,
    )))
    reference = _feature_scene()
    pipeline.set_scene_reference(reference, source="test:approved")
    pipeline.process_frame(_frame_data(reference, 1))

    with patch(
        "argus.core.pipeline.cv2.warpPerspective",
        wraps=cv2.warpPerspective,
    ) as warp:
        pipeline.process_frame(_frame_data(reference, 2))

    warp.assert_not_called()


def test_thumbnail_pose_shift_triggers_same_frame_full_geometry_check() -> None:
    pipeline = _build_pipeline(_camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        allow_alignable=False,
        consecutive_matches=1,
        stable_max_shift_px=5.0,
        alignable_max_shift_px=32.0,
        check_interval_frames=600,
        check_interval_seconds=10.0,
    )))
    reference = _feature_scene()
    pipeline.set_scene_reference(reference, source="test:approved")
    pipeline.process_frame(_frame_data(reference, 1))
    shifted = cv2.warpAffine(
        reference,
        np.float32([[1.0, 0.0, 16.0], [0.0, 1.0, 3.0]]),
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )

    assert pipeline._scene_guard is not None
    with patch.object(
        pipeline._scene_guard,
        "evaluate",
        wraps=pipeline._scene_guard.evaluate,
    ) as evaluate:
        pipeline.process_frame(_frame_data(shifted, 2))

    evaluate.assert_called_once()
    status = pipeline.get_scene_guard_status()
    assert status["usable"] is False
    assert status["quick_shift_px"] > 5.0


def test_static_confirmation_shadow_observes_and_enforce_rejects() -> None:
    confirmation = StaticCandidateConfirmationConfig(
        enabled=True,
        mode="enforce",
        sensitive_rois=[(20, 20, 76, 76)],
        sensitive_min_structure_score=0.95,
    )
    pipeline = _build_pipeline(_camera_config(
        scene_guard=SceneGuardConfig(
            enabled=True,
            require_profile_model_binding=False,
        ),
        static_confirmation=confirmation,
    ))
    reference = np.full((96, 96, 3), 80, dtype=np.uint8)
    current = reference.copy()
    current[30:66, 30:66] = 110
    pipeline._scene_guard_reference_frame = reference
    candidate = PersistentChangeCandidate(
        track_id=4,
        bbox=(30, 30, 36, 36),
        centroid=(48.0, 48.0),
        area_px=1296,
        mean_diff=30.0,
        max_diff=30.0,
        first_seen_ts=1.0,
        last_seen_ts=4.0,
        seen_frames=20,
        persistence_seconds=3.0,
        score=0.88,
    )
    persistent = PersistentChangeResult(
        ready=True,
        has_detection=True,
        score=0.88,
        candidates=[candidate],
    )
    anomaly = AnomalyResult(
        anomaly_score=0.1,
        anomaly_map=np.zeros((24, 24), dtype=np.float32),
        is_anomalous=False,
        threshold=0.7,
    )

    enforce_diag = FrameDiagnostics(1, time.time(), "scene-cam")
    enforced = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=enforce_diag,
    )
    assert enforced == []
    assert enforce_diag.candidate_records[-1].decision == "rejected"

    pipeline.camera_config.static_confirmation.mode = "shadow"
    shadow_diag = FrameDiagnostics(2, time.time(), "scene-cam")
    shadowed = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=shadow_diag,
    )
    assert shadowed == [candidate]
    assert shadow_diag.candidate_records[-1].decision == "shadow_reject"


def _model_required_fixture(*, mode: str = "shadow"):
    confirmation = StaticCandidateConfirmationConfig(
        enabled=True,
        mode=mode,
        model_evidence=CandidateModelEvidenceConfig(
            enabled=True,
            fusion_policy="model_required",
            calibration_id="placeholder",
            global_score_min=0.5,
            roi_p95_min=0.7,
            roi_context_delta_min=0.4,
            temporal_required_passes=2,
            temporal_min_gap_seconds=0.0,
            temporal_max_gap_seconds=2.0,
            temporal_iou_threshold=0.5,
        ),
    )
    scene_guard = (
        SceneGuardConfig(enabled=True, require_profile_model_binding=True)
        if mode == "enforce"
        else None
    )
    pipeline = _build_pipeline(
        _camera_config(
            scene_guard=scene_guard,
            static_confirmation=confirmation,
        )
    )
    reference = np.full((96, 96, 3), 80, dtype=np.uint8)
    current = reference.copy()
    current[30:66, 30:66] = 220
    current[34:62:4, 34:62] = 20
    current[34:62, 34:62:4] = 20
    pipeline._scene_guard_reference_frame = reference
    pipeline._scene_guard_reference_sha256 = _frame_content_sha256(reference)
    candidate = PersistentChangeCandidate(
        track_id=7,
        bbox=(30, 30, 36, 36),
        centroid=(48.0, 48.0),
        area_px=1296,
        mean_diff=80.0,
        max_diff=200.0,
        first_seen_ts=1.0,
        last_seen_ts=4.0,
        seen_frames=20,
        persistence_seconds=3.0,
        score=0.9,
        support_mask=np.ones((36, 36), dtype=bool),
    )
    persistent = PersistentChangeResult(
        ready=True,
        has_detection=True,
        score=0.9,
        candidates=[candidate],
        active_tracks=[candidate],
    )
    model_map = np.full((24, 24), 0.1, dtype=np.float32)
    model_map[7:17, 7:17] = 0.9
    anomaly = AnomalyResult(
        anomaly_score=0.8,
        anomaly_map=np.zeros((24, 24), dtype=np.float32),
        is_anomalous=True,
        threshold=0.7,
        model_score=0.9,
        model_anomaly_map=model_map,
        model_output_space="anomalib_postprocessed",
        model_output_comparable=True,
    )
    pipeline._model_artifact_sha256 = "a" * 64
    binding = pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    pipeline.camera_config.static_confirmation.model_evidence.calibration_id = (
        build_calibration_id(binding)
    )
    return pipeline, current, candidate, persistent, anomaly


def test_model_required_shadow_records_pending_then_accepts_without_changing_candidates():
    pipeline, current, candidate, persistent, anomaly = _model_required_fixture()

    first_diag = FrameDiagnostics(10, time.time(), "scene-cam")
    first = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=first_diag,
    )
    second_diag = FrameDiagnostics(11, time.time(), "scene-cam")
    second = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=second_diag,
    )

    assert first == [candidate]
    assert second == [candidate]
    assert first_diag.candidate_records[-1].decision == "shadow_pending"
    assert first_diag.candidate_records[-1].evidence["would_accept"] is False
    assert (
        first_diag.candidate_records[-1].evidence["attribution_disposition"]
        == "pending"
    )
    assert second_diag.candidate_records[-1].decision == "shadow_accept"
    evidence = second_diag.candidate_records[-1].evidence
    assert evidence["attribution_disposition"] == "accepted"
    assert evidence["model_evidence"]["status"] == "evaluated"
    assert evidence["model_evidence"]["roi_p95"] == pytest.approx(0.9)
    assert evidence["temporal_evidence"]["confirmed"] is True


def test_model_required_enforce_releases_only_on_second_consecutive_vote():
    pipeline, current, candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )

    first = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(20, time.time(), "scene-cam"),
    )
    second = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(21, time.time(), "scene-cam"),
    )

    assert first == []
    assert second == [candidate]


def test_temporal_gate_uses_processed_observations_not_source_frame_numbers():
    pipeline, current, candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    first_diag = FrameDiagnostics(0, time.time(), "scene-cam")
    first_diag.candidate_model_observation_number = 1
    second_diag = FrameDiagnostics(6, time.time(), "scene-cam")
    second_diag.candidate_model_observation_number = 2

    first = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=first_diag,
    )
    second = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=second_diag,
    )

    assert first == []
    assert second == [candidate]


def test_model_map_is_prepared_once_for_multiple_candidates():
    pipeline, current, candidate, persistent, anomaly = _model_required_fixture()
    second_candidate = replace(candidate, track_id=8)
    multiple = replace(
        persistent,
        candidates=[candidate, second_candidate],
        active_tracks=[candidate, second_candidate],
    )

    with patch(
        "argus.core.pipeline.prepare_model_evidence_map",
        wraps=prepare_model_evidence_map,
    ) as prepare:
        pipeline._confirm_persistent_change_candidates(
            current_frame=current,
            persistent_result=multiple,
            anomaly_result=anomaly,
            diag=FrameDiagnostics(25, time.time(), "scene-cam"),
        )

    prepare.assert_called_once()


def test_model_required_enforce_marks_missing_binding_unavailable_not_rejected():
    pipeline, current, _candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    pipeline._model_artifact_sha256 = None
    diag = FrameDiagnostics(26, time.time(), "scene-cam")

    result = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=diag,
    )

    assert result == []
    assert diag.candidate_records[-1].decision == "unavailable"
    assert diag.candidate_records[-1].evidence["would_accept"] is None
    assert (
        diag.candidate_records[-1].evidence["attribution_disposition"]
        == "unavailable"
    )
    assert diag.stages[-1].skip_reason == "candidate_evidence_unavailable"


def test_missing_reference_records_nullable_model_telemetry():
    pipeline, current, candidate, persistent, anomaly = _model_required_fixture()
    pipeline._scene_guard_reference_frame = None
    pipeline._scene_guard_reference_sha256 = None
    diag = FrameDiagnostics(27, time.time(), "scene-cam")

    result = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=diag,
    )

    assert result == [candidate]
    model = diag.candidate_records[-1].evidence["model_evidence"]
    assert model["status"] == "upstream_skipped"
    assert model["would_accept"] is None
    assert (
        diag.candidate_records[-1].evidence["attribution_disposition"]
        == "unavailable"
    )


def test_enforce_preserves_global_hotspot_while_first_candidate_vote_is_pending():
    pipeline, current, _candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    diag = FrameDiagnostics(28, time.time(), "scene-cam")
    pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=diag,
    )
    confirmation = diag.candidate_records[-1]
    assert confirmation.decision == "pending"
    assert confirmation.evidence["attribution_disposition"] == "pending"

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=persistent.candidates,
    )

    assert attributed is anomaly
    assert attributed.is_anomalous is True
    record = diag.candidate_records[-1]
    assert record.candidate_type == "anomaly_attribution"
    assert record.decision == "preserved"
    assert record.evidence["reason"] == "candidate_confirmation_pending"
    assert record.evidence["pending_candidate_count"] == 1


def test_missing_explicit_disposition_cannot_authorize_cleanup():
    pipeline, current, candidate, _persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    diag = FrameDiagnostics(28, time.time(), "scene-cam")
    diag.add_candidate(
        candidate_type="persistent_confirmation",
        decision="rejected",
        stage="static_confirmation",
        evidence={
            "track_id": candidate.track_id,
            "would_accept": False,
            "model_evidence": {"status": "evaluated"},
        },
    )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=[candidate],
    )

    assert attributed is anomaly
    record = diag.candidate_records[-1]
    assert record.decision == "preserved"
    assert record.evidence["reason"] == "candidate_evidence_unavailable"
    assert record.evidence["rejected_candidate_count"] == 0
    assert record.evidence["unavailable_candidate_count"] == 1


def test_enforce_preserves_global_channel_when_independent_hotspot_exists():
    pipeline, current, _candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    model_map = np.full_like(anomaly.model_anomaly_map, 0.1)
    model_map[0:3, 0:3] = 0.9
    anomaly = replace(anomaly, model_anomaly_map=model_map)
    diag = FrameDiagnostics(29, time.time(), "scene-cam")
    pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=diag,
    )
    assert (
        diag.candidate_records[-1].evidence["attribution_disposition"]
        == "rejected"
    )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=persistent.candidates,
    )

    assert attributed.is_anomalous is True
    assert diag.candidate_records[-1].decision == "preserved"
    assert diag.candidate_records[-1].evidence["reason"] == (
        "independent_hotspot_preserved"
    )


def test_simplex_signal_prevents_global_anomaly_suppression():
    pipeline, current, _candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    diag = FrameDiagnostics(30, time.time(), "scene-cam")
    pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=diag,
    )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=True,
        persistent_candidates=persistent.candidates,
    )

    assert attributed.is_anomalous is True
    assert diag.candidate_records[-1].evidence["reason"] == (
        "simplex_safety_channel_detected"
    )


def test_shadow_records_attribution_without_changing_global_anomaly():
    pipeline, current, candidate, _persistent, anomaly = _model_required_fixture()
    rejected = replace(
        candidate,
        bbox=(28, 28, 40, 40),
        support_mask=np.ones((40, 40), dtype=bool),
    )
    diag = FrameDiagnostics(31, time.time(), "scene-cam")
    diag.add_candidate(
        candidate_type="persistent_confirmation",
        decision="shadow_reject",
        stage="static_confirmation",
        evidence={
            "track_id": rejected.track_id,
            "would_accept": False,
            "attribution_disposition": "rejected",
            "model_evidence": {"status": "evaluated"},
        },
    )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=[rejected],
    )

    assert attributed.is_anomalous is True
    assert diag.candidate_records[-1].decision == "shadow_suppress"
    assert diag.candidate_records[-1].evidence["effective"] is False


def _mixed_candidate_attribution_fixture(*, mode: str):
    pipeline, current, candidate, _persistent, anomaly = _model_required_fixture(
        mode=mode
    )
    accepted = replace(
        candidate,
        track_id=7,
        bbox=(20, 20, 32, 32),
        support_mask=np.ones((32, 32), dtype=bool),
    )
    rejected = replace(
        candidate,
        track_id=8,
        bbox=(44, 20, 32, 32),
        support_mask=np.ones((32, 32), dtype=bool),
    )

    # Frame-to-map scale is exactly 4:1.  The accepted and rejected components
    # overlap at x=[11, 13), and the independent hotspot at the bottom-left is
    # outside both components.
    appearance_map = np.full((24, 24), 0.1, dtype=np.float32)
    appearance_map[5:13, 5:19] = 0.9
    appearance_map[20:22, 2:4] = 0.95
    model_map = np.full((24, 24), 0.1, dtype=np.float32)
    model_map[5:13, 5:13] = 0.9
    model_map[20:22, 2:4] = 0.95
    anomaly = replace(
        anomaly,
        anomaly_map=appearance_map,
        model_anomaly_map=model_map,
    )
    diag = FrameDiagnostics(32, time.time(), "scene-cam")
    for disposition, item, would_accept in (
        ("accepted", accepted, True),
        ("rejected", rejected, False),
    ):
        diag.add_candidate(
            candidate_type="persistent_confirmation",
            decision=disposition,
            stage="static_confirmation",
            evidence={
                "track_id": item.track_id,
                "would_accept": would_accept,
                "attribution_disposition": disposition,
                "model_evidence": {"status": "evaluated"},
            },
        )
    return pipeline, current, [accepted, rejected], anomaly, diag


@pytest.mark.parametrize(
    ("protected_disposition", "protected_would_accept", "model_status"),
    (
        ("accepted", True, "evaluated"),
        ("pending", False, "evaluated"),
        ("unavailable", None, "rate_limited"),
    ),
)
def test_protective_disposition_wins_overlap_with_rejected_support(
    protected_disposition,
    protected_would_accept,
    model_status,
):
    pipeline, current, candidate, _persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    rejected = replace(
        candidate,
        track_id=41,
        bbox=(20, 20, 32, 32),
        support_mask=np.ones((32, 32), dtype=bool),
    )
    protected = replace(
        candidate,
        track_id=42,
        # One source pixel contributes only 1/16 of a coarse target pixel.
        # Protective dispositions must still own that overlap.
        bbox=(36, 20, 1, 1),
        support_mask=np.ones((1, 1), dtype=bool),
    )
    appearance_map = np.full((24, 24), 0.1, dtype=np.float32)
    appearance_map[5:13, 5:13] = 0.9
    model_map = appearance_map.copy()
    anomaly = replace(
        anomaly,
        anomaly_map=appearance_map,
        model_anomaly_map=model_map,
    )
    diag = FrameDiagnostics(33, time.time(), "scene-cam")
    for disposition, item, would_accept, status in (
        ("rejected", rejected, False, "evaluated"),
        (
            protected_disposition,
            protected,
            protected_would_accept,
            model_status,
        ),
    ):
        diag.add_candidate(
            candidate_type="persistent_confirmation",
            decision=disposition,
            stage="static_confirmation",
            evidence={
                "track_id": item.track_id,
                "would_accept": would_accept,
                "attribution_disposition": disposition,
                "model_evidence": {"status": status},
            },
        )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=[rejected, protected],
    )

    assert attributed.is_anomalous is True
    retained = attributed.anomaly_map[5:13, 5:13]
    assert np.count_nonzero(retained) == 1
    assert attributed.anomaly_map[5, 9] == pytest.approx(0.9)
    record = diag.candidate_records[-1]
    assert record.decision == "partially_suppressed"
    assert record.evidence["overlap_hot_pixels"] == 1
    assert record.evidence[f"{protected_disposition}_candidate_count"] == 1


def test_tiny_rejected_support_cannot_claim_entire_coarse_map_pixel():
    pipeline, current, candidate, _persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    tiny_rejected = replace(
        candidate,
        track_id=43,
        bbox=(0, 0, 1, 1),
        support_mask=np.ones((1, 1), dtype=bool),
    )
    # Keep a valid 2-D model output after the evidence contract's squeeze();
    # the hot top-left cell still covers 48x48 frame pixels, so one source
    # pixel must not claim ownership of that whole coarse cell.
    coarse_map = np.array(
        [[0.9, 0.1], [0.1, 0.1]],
        dtype=np.float32,
    )
    anomaly = replace(
        anomaly,
        anomaly_map=coarse_map,
        model_anomaly_map=coarse_map,
    )
    diag = FrameDiagnostics(34, time.time(), "scene-cam")
    diag.add_candidate(
        candidate_type="persistent_confirmation",
        decision="rejected",
        stage="static_confirmation",
        evidence={
            "track_id": tiny_rejected.track_id,
            "would_accept": False,
            "attribution_disposition": "rejected",
            "model_evidence": {"status": "evaluated"},
        },
    )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=[tiny_rejected],
    )

    assert attributed is anomaly
    record = diag.candidate_records[-1]
    assert record.decision == "preserved"
    assert record.evidence["covered_hot_pixels"] == 0
    assert record.evidence["independent_hot_pixels"] == 1
    assert record.evidence["partial_removal_support_pixel_count"] == 0
    assert record.evidence["removal_occupancy_threshold"] == pytest.approx(0.95)


@pytest.mark.parametrize("remaining_peak", (0.0, 0.3))
def test_partial_cleanup_preserves_scalar_when_native_hotspot_is_independent(
    remaining_peak,
):
    pipeline, current, candidate, _persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    assert pipeline._zone_mask.get_include_zones.return_value == []
    rejected = replace(
        candidate,
        track_id=44,
        bbox=(20, 20, 32, 32),
        support_mask=np.ones((32, 32), dtype=bool),
    )
    appearance_map = np.zeros((24, 24), dtype=np.float32)
    appearance_map[5:13, 5:13] = 0.9
    if remaining_peak:
        appearance_map[20, 20] = remaining_peak
    model_map = np.zeros((24, 24), dtype=np.float32)
    model_map[5:13, 5:13] = 0.9
    # Native evidence outside the rejected component prevents whole-channel
    # suppression while the display map exercises scalar cleanup behavior.
    model_map[20, 20] = 0.8
    anomaly = replace(
        anomaly,
        anomaly_map=appearance_map,
        model_anomaly_map=model_map,
    )
    diag = FrameDiagnostics(35, time.time(), "scene-cam")
    diag.add_candidate(
        candidate_type="persistent_confirmation",
        decision="rejected",
        stage="static_confirmation",
        evidence={
            "track_id": rejected.track_id,
            "would_accept": False,
            "attribution_disposition": "rejected",
            "model_evidence": {"status": "evaluated"},
        },
    )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=[rejected],
    )

    assert attributed.anomaly_score == pytest.approx(anomaly.anomaly_score)
    assert attributed.is_anomalous is True
    assert diag.anomaly_score == pytest.approx(anomaly.anomaly_score)
    assert diag.is_anomalous is True
    assert float(np.max(attributed.anomaly_map)) == pytest.approx(
        anomaly.anomaly_score
    )
    assert attributed.anomaly_map[20, 20] == pytest.approx(
        anomaly.anomaly_score
    )
    assert np.count_nonzero(attributed.anomaly_map[5:13, 5:13]) == 0
    record = diag.candidate_records[-1]
    assert record.decision == "partially_suppressed"
    assert record.evidence["appearance_score_before"] == pytest.approx(
        anomaly.anomaly_score
    )
    assert record.evidence["appearance_score_after"] == pytest.approx(
        anomaly.anomaly_score
    )
    assert record.evidence["appearance_score_scale"] == pytest.approx(1.0)
    assert record.evidence["scalar_preserved_for_native_support"] is True
    assert record.evidence["scalar_support_reconstructed"] is True
    assert record.evidence["native_support_projected_pixel_count"] > 0


def test_enforce_removes_only_rejected_support_in_mixed_candidate_frame():
    pipeline, current, candidates, anomaly, diag = (
        _mixed_candidate_attribution_fixture(mode="enforce")
    )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=candidates,
    )

    assert attributed.is_anomalous is True
    assert attributed.anomaly_score == anomaly.anomaly_score
    assert np.allclose(attributed.anomaly_map[5:13, 5:13], 0.9)
    assert np.all(attributed.anomaly_map[5:13, 13:19] == 0.0)
    assert np.allclose(attributed.anomaly_map[20:22, 2:4], 0.95)
    record = diag.candidate_records[-1]
    assert record.decision == "partially_suppressed"
    assert record.evidence["reason"] == (
        "rejected_candidate_support_partially_removed"
    )
    assert record.evidence["partial_removal_pixel_count"] == 48
    # Reject-exclusive display evidence is removed even though no corresponding
    # native model-hot pixel crossed the attribution threshold.
    assert record.evidence["partial_removal_coverage"] == pytest.approx(0.0)
    assert record.evidence["overlap_hot_pixels"] == 16
    assert record.evidence["independent_hot_pixels"] == 4
    assert record.evidence["partial_removal_effective"] is True


def test_shadow_records_mixed_partial_removal_without_changing_map():
    pipeline, current, candidates, anomaly, diag = (
        _mixed_candidate_attribution_fixture(mode="shadow")
    )
    original_map = anomaly.anomaly_map.copy()

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=candidates,
    )

    assert attributed is anomaly
    assert np.array_equal(attributed.anomaly_map, original_map)
    record = diag.candidate_records[-1]
    assert record.decision == "shadow_partial_suppress"
    assert record.evidence["would_remove_partial"] is True
    assert record.evidence["partial_removal_effective"] is False
    assert record.evidence["effective"] is False
    assert record.evidence["partial_removal_pixel_count"] == 48


def test_concave_candidate_bbox_cannot_hide_hotspot_outside_component_support():
    pipeline, current, candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    support = np.zeros((36, 36), dtype=bool)
    support[:4, :] = True
    support[:, :4] = True
    candidate = replace(candidate, support_mask=support)
    persistent = replace(
        persistent,
        candidates=[candidate],
        active_tracks=[candidate],
    )
    model_map = np.full((24, 24), 0.1, dtype=np.float32)
    # Hotspot is inside the candidate's rectangular bbox, but in the empty
    # lower-right area of the L-shaped connected component.
    model_map[14:16, 14:16] = 0.9
    anomaly = replace(anomaly, model_anomaly_map=model_map)
    diag = FrameDiagnostics(32, time.time(), "scene-cam")
    pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=diag,
    )

    attributed = pipeline._apply_candidate_anomaly_attribution(
        anomaly,
        diag=diag,
        frame_shape=current.shape,
        simplex_detected=False,
        persistent_candidates=persistent.candidates,
    )

    assert attributed.is_anomalous is True
    attribution = diag.candidate_records[-1]
    assert attribution.decision == "preserved"
    assert attribution.evidence["reason"] == "independent_hotspot_preserved"
    assert attribution.evidence["hot_coverage"] == pytest.approx(0.0)


def test_stale_persistent_track_cannot_supply_second_vote():
    pipeline, current, candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(30, time.time(), "scene-cam"),
    )
    stale = replace(candidate, observed_this_frame=False)
    stale_result = replace(persistent, candidates=[stale], active_tracks=[stale])

    stale_diag = FrameDiagnostics(31, time.time(), "scene-cam")
    second = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=stale_result,
        anomaly_result=anomaly,
        diag=stale_diag,
    )
    third = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(32, time.time(), "scene-cam"),
    )

    assert second == []
    assert stale_diag.candidate_records[-1].evidence["would_accept"] is None
    assert third == []


def test_scene_evidence_reset_prevents_pending_vote_crossing_epochs():
    pipeline, current, _candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(40, time.time(), "scene-cam"),
    )
    original_epoch = pipeline._candidate_model_evidence_epoch

    pipeline._reset_candidate_model_evidence_state()
    result = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(41, time.time(), "scene-cam"),
    )

    assert pipeline._candidate_model_evidence_epoch == original_epoch + 1
    assert result == []


def test_mode_transition_clears_candidate_confirmation_votes():
    pipeline, current, _candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(45, time.time(), "scene-cam"),
    )
    assert pipeline._candidate_model_temporal.snapshot()

    pipeline.set_mode(PipelineMode.LEARNING)
    pipeline.set_mode(PipelineMode.ACTIVE)

    assert pipeline._candidate_model_temporal.snapshot() == []


def test_upstream_skip_is_nullable_and_clears_pending_vote():
    pipeline, current, _candidate, persistent, anomaly = _model_required_fixture(
        mode="enforce"
    )
    pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(50, time.time(), "scene-cam"),
    )
    skipped_diag = FrameDiagnostics(51, time.time(), "scene-cam")

    pipeline._record_candidate_model_evidence_skip(
        persistent_result=persistent,
        diag=skipped_diag,
        status="upstream_skipped",
        reason="person_detected",
    )
    restarted = pipeline._confirm_persistent_change_candidates(
        current_frame=current,
        persistent_result=persistent,
        anomaly_result=anomaly,
        diag=FrameDiagnostics(52, time.time(), "scene-cam"),
    )

    record = skipped_diag.candidate_records[-1]
    assert record.evidence["decision_status"] == "upstream_skipped"
    assert record.evidence["would_accept"] is None
    assert restarted == []


def test_same_model_tag_with_different_artifact_hash_changes_calibration_binding():
    pipeline, _current, _candidate, _persistent, anomaly = _model_required_fixture()
    first = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    pipeline._model_artifact_sha256 = "b" * 64
    second = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    assert pipeline._model_version_id == "model-v1"
    assert first != second


def test_calibration_binding_includes_explicit_algorithm_and_dependency_contracts():
    pipeline, _current, _candidate, _persistent, anomaly = _model_required_fixture()

    binding = pipeline._candidate_model_evidence_binding(
        anomaly.model_output_space
    )

    assert binding["schema_version"] == 2
    assert binding["output_transform"] == "argus-model-evidence-v2"
    assert binding["algorithm_contract"]["anomaly_attribution"] == (
        "connected-component-partitioned-map-overlap-v3"
    )
    assert set(binding["runtime_dependencies"]) == {
        "anomalib",
        "numpy",
        "opencv_python",
        "openvino",
        "torch",
    }


def test_candidate_evidence_policy_change_changes_calibration_binding():
    pipeline, _current, _candidate, _persistent, anomaly = _model_required_fixture()
    first = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    pipeline.camera_config.static_confirmation.model_evidence.context_ratio = 0.75
    second = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    assert first != second


def test_reference_content_change_changes_calibration_binding():
    pipeline, current, _candidate, _persistent, anomaly = _model_required_fixture()
    first = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    changed = current.copy()
    changed[0, 0] = 255
    pipeline._scene_guard_reference_sha256 = _frame_content_sha256(changed)
    second = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    assert first != second


def test_model_input_contract_change_changes_calibration_binding():
    pipeline, _current, _candidate, _persistent, anomaly = _model_required_fixture()
    first = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    pipeline.camera_config.person_filter.confidence = 0.77
    second = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )
    masked = build_calibration_id(
        pipeline._candidate_model_evidence_binding(
            anomaly.model_output_space,
            model_input_variant="person_masked",
        )
    )

    assert first != second
    assert second != masked


def test_allow_alignable_policy_change_invalidates_calibration_binding():
    pipeline, _current, _candidate, _persistent, anomaly = _model_required_fixture()
    first = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    pipeline.camera_config.scene_guard.allow_alignable = True
    second = build_calibration_id(
        pipeline._candidate_model_evidence_binding(anomaly.model_output_space)
    )

    assert first != second


def test_shared_detector_uses_its_actual_artifact_path(tmp_path):
    shared_path = tmp_path / "shared.pt"
    per_camera_path = tmp_path / "camera.pt"
    shared_path.write_bytes(b"shared")
    per_camera_path.write_bytes(b"per-camera")
    shared = MagicMock()
    shared._model_path = shared_path

    pipeline = _build_pipeline(
        _camera_config(),
        model_path=per_camera_path,
        shared_anomaly_detector=shared,
    )

    assert pipeline._anomaly_model_path == shared_path


def test_model_reload_revalidates_profile_before_frame_processing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "normal.jpg"
    assert cv2.imwrite(str(baseline_path), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    old_model_path = tmp_path / "old-model.pt"
    old_model_path.write_bytes(b"old model artifact")

    profiles_dir = tmp_path / "profiles"
    SceneProfileStore(profiles_dir).create_from_baseline(
        "scene-cam",
        "approved-v1",
        baseline_path,
        metadata=_trusted_profile_metadata(old_model_path),
        activate=True,
    )
    pipeline = _build_pipeline(
        _camera_config(scene_guard=SceneGuardConfig(
            enabled=True,
            profiles_dir=str(profiles_dir),
            fallback_to_baseline=False,
            consecutive_matches=1,
        )),
        model_version_id="model-v1",
        model_path=old_model_path,
    )
    _initialize_loaded_model(pipeline)
    assert pipeline.get_scene_guard_status()["reference_available"] is True
    pipeline._anomaly_detector.hot_reload.return_value = True
    new_model_path = tmp_path / "new-model.pt"
    new_model_path.write_bytes(b"new model artifact")

    assert pipeline.reload_anomaly_model(new_model_path, version_tag="model-v2") is True

    pipeline._anomaly_detector.hot_reload.assert_called_once()
    assert pipeline._model_version_id == "model-v2"
    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is False
    assert status["usable"] is False
    assert "profile=model-v1, runtime=model-v2" in status["reference_error"]


def test_model_reload_authorizes_only_stable_digest_bound_profile(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "normal.jpg"
    assert cv2.imwrite(str(baseline_path), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    old_model_path = tmp_path / "old-model.pt"
    new_model_path = tmp_path / "new-model.pt"
    old_model_path.write_bytes(b"old model artifact")
    new_model_path.write_bytes(b"new model artifact")
    profiles_dir = tmp_path / "profiles"
    SceneProfileStore(profiles_dir).create_from_baseline(
        "scene-cam",
        "approved-v2",
        baseline_path,
        metadata=_trusted_profile_metadata(
            new_model_path,
            model_version_id="model-v2",
        ),
        activate=True,
    )
    pipeline = _build_pipeline(
        _camera_config(scene_guard=SceneGuardConfig(
            enabled=True,
            profiles_dir=str(profiles_dir),
        )),
        model_version_id="model-v1",
        model_path=old_model_path,
    )
    _initialize_loaded_model(pipeline)
    assert pipeline.get_scene_guard_status()["reference_available"] is False
    pipeline._anomaly_detector.hot_reload.return_value = True

    assert pipeline.reload_anomaly_model(
        new_model_path,
        version_tag="model-v2",
    ) is True

    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is True
    assert status["profile_id"] == "approved-v2"
    assert status["runtime_model_artifact_sha256"] == (
        runtime_artifact_sha256(new_model_path)
    )
    assert status["bound_model_artifact_sha256"] == (
        runtime_artifact_sha256(new_model_path)
    )


def test_model_reload_never_keeps_old_hash_when_artifact_changes_during_swap():
    pipeline, _current, _candidate, _persistent, _anomaly = (
        _model_required_fixture()
    )
    pipeline._anomaly_detector.hot_reload.return_value = True
    original_epoch = pipeline._candidate_model_evidence_epoch

    with patch(
        "argus.core.pipeline._model_artifact_sha256",
        side_effect=["new-before", "new-after"],
    ):
        result = pipeline.reload_anomaly_model(
            "new-model.pt",
            version_tag="model-v2",
        )

    assert result is True
    assert pipeline._model_artifact_sha256 is None
    assert pipeline._candidate_model_evidence_epoch > original_epoch
    assert pipeline.get_scene_guard_status()["reference_available"] is False


def test_initialize_never_binds_hash_when_artifact_changes_during_load(tmp_path):
    pipeline = _build_pipeline(_camera_config())
    pipeline._anomaly_model_path = tmp_path / "model.pt"
    pipeline._anomaly_model_path.write_bytes(b"model")
    pipeline._camera.connect.return_value = True
    pipeline._anomaly_detector.get_status.return_value = SimpleNamespace(
        minmax_broken=False
    )
    pipeline._anomaly_detector.load.return_value = True

    with patch(
        "argus.core.pipeline._model_artifact_sha256",
        side_effect=["before-load", "after-load"],
    ):
        assert pipeline.initialize() is True

    assert pipeline._model_artifact_sha256 is None


def test_initialize_false_load_result_cannot_authorize_runtime_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "normal.jpg"
    assert cv2.imwrite(str(baseline_path), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    profiles_dir = tmp_path / "profiles"
    SceneProfileStore(profiles_dir).create_from_baseline(
        "scene-cam",
        "approved-v1",
        baseline_path,
        metadata=_trusted_profile_metadata(model_path),
        activate=True,
    )
    pipeline = _build_pipeline(
        _camera_config(scene_guard=SceneGuardConfig(
            enabled=True,
            profiles_dir=str(profiles_dir),
        )),
        model_version_id="model-v1",
        model_path=model_path,
    )
    pipeline._camera.connect.return_value = True
    pipeline._anomaly_detector.load.return_value = False
    pipeline._anomaly_detector.get_status.return_value = SimpleNamespace(
        minmax_broken=False
    )

    assert pipeline.initialize() is True

    assert pipeline._model_artifact_sha256 is None
    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is False
    assert status["usable"] is False
    assert "fingerprint is unavailable" in status["reference_error"]


def test_scene_bound_pipeline_rejects_model_reload_without_version() -> None:
    pipeline = _build_pipeline(_camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        fallback_to_baseline=False,
        require_profile_model_binding=True,
    )))
    pipeline._anomaly_detector.hot_reload.return_value = True

    assert pipeline.reload_anomaly_model("unregistered-model.pt") is False

    pipeline._anomaly_detector.hot_reload.assert_not_called()


def test_model_version_refresh_loads_profile_published_after_startup(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    profiles_dir = tmp_path / "profiles"
    camera = _camera_config(scene_guard=SceneGuardConfig(
        enabled=True,
        profiles_dir=str(profiles_dir),
        fallback_to_baseline=False,
        consecutive_matches=1,
    ))
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"runtime model")
    pipeline = _build_pipeline(
        camera,
        model_version_id="model-v1",
        model_path=model_path,
    )
    _initialize_loaded_model(pipeline)
    assert pipeline.get_scene_guard_status()["reference_available"] is False

    reference = _feature_scene()
    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "normal.jpg"
    assert cv2.imwrite(str(baseline_path), reference)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    SceneProfileStore(profiles_dir).create_from_baseline(
        "scene-cam",
        "approved-v1",
        baseline_path,
        metadata=_trusted_profile_metadata(model_path),
        activate=True,
    )

    pipeline.set_model_version_id("model-v1")

    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is True
    assert status["profile_id"] == "approved-v1"
    assert status["reference_error"] is None


def test_unusable_profile_revalidation_clears_previous_reference(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    profiles_dir = tmp_path / "profiles"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"runtime model")
    pipeline = _build_pipeline(
        _camera_config(scene_guard=SceneGuardConfig(
            enabled=True,
            profiles_dir=str(profiles_dir),
        )),
        model_version_id="model-v1",
        model_path=model_path,
    )
    _initialize_loaded_model(pipeline)
    pipeline.set_scene_reference(_feature_scene(), source="test:previous")
    assert pipeline.get_scene_guard_status()["reference_available"] is True

    baseline_dir = tmp_path / "data" / "baselines" / "scene-cam" / "default" / "v001"
    baseline_dir.mkdir(parents=True)
    featureless = np.full((240, 320, 3), 80, dtype=np.uint8)
    baseline_path = baseline_dir / "featureless.jpg"
    assert cv2.imwrite(str(baseline_path), featureless)
    (baseline_dir.parent / "current.txt").write_text("v001", encoding="utf-8")
    SceneProfileStore(profiles_dir).create_from_baseline(
        "scene-cam",
        "featureless-v1",
        baseline_path,
        metadata=_trusted_profile_metadata(model_path),
        activate=True,
    )

    pipeline.set_model_version_id("model-v1")

    status = pipeline.get_scene_guard_status()
    assert status["reference_available"] is False
    assert status["usable"] is False
    assert "approved scene reference is unusable" in status["reference_error"]
