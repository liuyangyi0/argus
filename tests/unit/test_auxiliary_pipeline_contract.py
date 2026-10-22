"""Auxiliary orchestration with fake heads only; no source media/model I/O."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.alerts.grader import AlertGrader, DetectionType
from argus.anomaly.detector import AnomalyResult
from argus.anomaly.water_leak import WaterLeakCandidate, WaterLeakResult
from argus.config.schema import CameraConfig, ClassifierConfig, PhysicsConfig, SegmenterConfig
from argus.core.pipeline import DetectionPipeline
from argus.person.detector import ObjectDetectionResult
from argus.prefilter.mog2 import PreFilterResult


def _module_pipeline():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline.camera_config = CameraConfig(camera_id="mock-only", name="Mock", source="offline")
    pipeline._scene_processing_lock = threading.RLock()
    pipeline._auxiliary_errors = {}
    pipeline._auxiliary_retired = {}
    pipeline._auxiliary_shutdown = False
    pipeline._auxiliary_load_blocked = set()
    pipeline._classifier = None
    pipeline._segmenter = None
    pipeline._classifier_config = ClassifierConfig(vocabulary=["wrench", "shadow"])
    pipeline._segmenter_config = SegmenterConfig()
    pipeline._physics_config = PhysicsConfig()
    pipeline._speed_estimator = None
    pipeline._camera_calibration = None
    pipeline._segmenter_consecutive_failures = 3
    return pipeline


def _component():
    component = MagicMock()
    component.load.return_value = True
    component.get_status.return_value = {
        "ready": True, "loaded": True, "state": "ready", "busy": False,
    }
    component.vocabulary = ["wrench", "shadow"]
    return component


@pytest.mark.parametrize("name", ["classifier", "segmenter"])
def test_failed_enable_retains_slot_and_can_retry(name, monkeypatch):
    pipeline = _module_pipeline()
    component = _component()
    component.load.return_value = False
    component.get_status.return_value = {"ready": False, "reason": "fake failure", "busy": True}
    module, factory = (
        ("argus.anomaly.classifier", "OpenVocabClassifier") if name == "classifier"
        else ("argus.anomaly.segmenter", "InstanceSegmenter")
    )
    create = MagicMock(return_value=component)
    monkeypatch.setattr(f"{module}.{factory}", create)
    assert pipeline.reload_module(f"{name}.enabled", True) is False
    assert getattr(pipeline, f"_{name}") is component
    assert name in pipeline._auxiliary_load_blocked
    assert pipeline._auxiliary_status(name, component)["ready"] is False
    component.load.return_value = True
    component.get_status.return_value = {"ready": True, "loaded": True, "busy": False}
    assert pipeline.reload_module(f"{name}.enabled", True) is True
    assert create.call_count == 1
    assert name not in pipeline._auxiliary_load_blocked


@pytest.mark.parametrize("name", ["classifier", "segmenter"])
def test_disable_detaches_closes_and_reports_draining(name):
    pipeline = _module_pipeline()
    component = _component()
    component.get_status.return_value = {"ready": False, "state": "stopped", "busy": True}
    setattr(pipeline, f"_{name}", component)
    assert pipeline.reload_module(f"{name}.enabled", False) is True
    component.shutdown.assert_called_once()
    assert getattr(pipeline, f"_{name}") is None
    status = pipeline._auxiliary_status(name, None)
    assert status["state"] == "draining"
    assert status["busy"] is True
    assert pipeline.reload_module(f"{name}.enabled", True) is False


def test_unknown_cleanup_failure_retains_ownership():
    pipeline = _module_pipeline()
    component = _component()
    component.shutdown.side_effect = RuntimeError("fake cleanup failure")
    pipeline._classifier = component
    assert pipeline.reload_module("classifier.enabled", False) is False
    assert pipeline._auxiliary_retired["classifier"] is component
    assert pipeline.reload_module("classifier.enabled", True) is False
    assert pipeline._auxiliary_status("classifier", None)["state"] == "failed"
    component.shutdown.side_effect = None
    component.get_status.return_value = {"state": "stopped", "busy": False}
    assert pipeline.reload_module("classifier.enabled", False) is True
    assert "classifier" not in pipeline._auxiliary_retired
    assert "classifier" not in pipeline._auxiliary_errors


def test_summary_does_not_hide_camera_with_failed_status():
    from argus.capture.manager import CameraManager

    manager = CameraManager.__new__(CameraManager)
    manager._lock = threading.RLock()
    manager._pipelines = {"ready-camera": object(), "unknown-camera": object()}
    manager._correlator = None
    manager.get_system_status_snapshots = lambda: [{
        "camera_id": "ready-camera", "components": {"classifier": {
            "state": "ready", "ready": True, "loaded": True, "degraded": False,
        }},
    }]
    summary = manager.get_runtime_component_summary()
    assert summary["total_pipelines"] == 2
    assert summary["classifier_ready"] == 1
    assert summary["classifier_failed"] == 1


def test_speed_toggle_creates_and_removes_estimator_without_camera_physics():
    pipeline = _module_pipeline()
    assert not hasattr(pipeline.camera_config, "physics")
    assert pipeline.reload_module("physics.speed_enabled", True) is True
    assert pipeline._speed_estimator is not None
    assert pipeline.reload_module("physics.speed_enabled", False) is True
    assert pipeline._speed_estimator is None


def test_stopped_pipeline_cannot_reload():
    pipeline = _module_pipeline()
    pipeline._auxiliary_shutdown = True
    assert pipeline.reload_module("physics.speed_enabled", True) is False


def test_policy_buckets_refresh_with_successful_vocabulary_push():
    pipeline = _module_pipeline()
    pipeline._classifier = _component()
    pipeline._classifier_config.low_risk_labels = ["shadow"]
    pipeline._classifier_config.suppress_labels = []
    assert pipeline.update_classifier_vocabulary(["wrench", "shadow"]) is True
    assert pipeline._classifier_low_risk == {"shadow"}
    assert pipeline._classifier_suppress == set()


@pytest.fixture
def simulate(request):
    guard = next((
        plugin for plugin in request.config.pluginmanager.get_plugins()
        if type(plugin).__name__ == "FunctionalGuard"
    ), None)
    assert guard is not None, "Run using scripts/run_local_functional_tests.py --suite auxiliary"
    return guard.simulate_frame


def _flow_pipeline(*, water=True, anomalous=False):
    # The existing constructor fixture mocks camera, YOLO and anomaly factories.
    import test_pipeline_core as fixtures

    pipeline = fixtures._build_pipeline()
    for name in (
        "_persistent_change", "_simplex", "_plume", "_water_leak", "_aligner",
        "_static_candidate_confirmer", "_candidate_model_temporal", "_fast_motion",
    ):
        setattr(pipeline, name, None)
    pipeline._physics_postproc = MagicMock()
    pipeline._physics_tracker = MagicMock()
    pipeline._physics_tracker.update.return_value = SimpleNamespace(active_tracks=[])
    pipeline._frame_quality_assessor = MagicMock()
    pipeline._frame_quality_assessor.confidence_multiplier.return_value = 1.0
    _, config = fixtures._make_configs()
    config = config.model_copy(update={
        "temporal": config.temporal.model_copy(update={
            "evidence_threshold": 0.5, "spatial_continuity_enabled": False,
        }),
    })
    pipeline._alert_grader = AlertGrader(config)
    pipeline._alert_grader.evaluate = MagicMock(wraps=pipeline._alert_grader.evaluate)
    pipeline._last_heartbeat_time = 0
    pipeline._prefilter.process.return_value = PreFilterResult(has_change=True, change_ratio=0.1)
    if water:
        candidate = WaterLeakCandidate(
            track_id=8, leak_type="pooling", bbox=(80, 90, 40, 30),
            centroid=(100.0, 105.0), area_px=1200, growth_area_px=360,
            mean_absdiff=48.0, mean_signed_delta=-42.0, first_seen_ts=1.0,
            last_seen_ts=2.5, seen_frames=6, persistence_seconds=1.5,
            vertical_extent_px=30.0, vertical_aspect=0.75, score=0.98,
        )
        water_map = np.zeros((480, 640), dtype=np.float32)
        water_map[90:120, 80:120] = 0.98
        pipeline._water_leak = MagicMock()
        pipeline._water_leak.process.return_value = WaterLeakResult(
            ready=True, has_detection=True, score=0.98, anomaly_map=water_map,
            candidates=[candidate], active_tracks=[candidate], changed_fraction=0.003,
        )
    appearance_map = np.zeros((32, 32), dtype=np.float32)
    appearance_map[8:16, 8:16] = 0.99
    anomaly = AnomalyResult(
        anomaly_score=0.99 if anomalous else 0.0,
        anomaly_map=appearance_map if anomalous else None,
        is_anomalous=anomalous, threshold=0.7,
    )
    futures = []
    for value in (ObjectDetectionResult(has_persons=False), anomaly):
        future = Future()
        future.set_result(value)
        futures.append(future)
    pipeline._inference_executor = MagicMock()
    pipeline._inference_executor.submit.side_effect = futures
    return pipeline, fixtures._make_frame_data(frame=np.full((480, 640, 3), 127, dtype=np.uint8))


def test_fast_motion_failure_does_not_drop_independent_water_alert(simulate):
    pipeline, frame = _flow_pipeline()
    pipeline._fast_motion = MagicMock()
    pipeline._fast_motion.process.side_effect = RuntimeError("fake fast head failure")
    pipeline._alert_grader.decay_unobserved = MagicMock(wraps=pipeline._alert_grader.decay_unobserved)
    result = simulate(pipeline, frame)
    assert result is not None
    assert result.detection_type == DetectionType.WATER_LEAK
    assert "fake fast head failure" in pipeline._auxiliary_errors["fast_motion"]
    diag = pipeline.get_diagnostics_buffer().get_recent(1)[0]
    assert any(r.candidate_type == "fast_motion" and r.decision == "unavailable" for r in diag.candidate_records)
    assert all(
        DetectionType.PROJECTILE.value not in (call.kwargs.get("decay_detection_types") or set())
        for call in pipeline._alert_grader.decay_unobserved.call_args_list
    )


def test_fast_handler_side_effect_failure_cannot_reenter_other_decision_path(simulate, monkeypatch):
    from argus.prefilter.fast_motion import FastMotionResult

    pipeline, frame = _flow_pipeline()
    pipeline._fast_motion = MagicMock()
    pipeline._fast_motion.process.return_value = FastMotionResult(has_detection=True, max_confidence=0.99)
    handler = MagicMock(side_effect=RuntimeError("fake post-dispatch failure"))
    monkeypatch.setattr(pipeline, "_handle_fast_motion_detection", handler)
    assert simulate(pipeline, frame) is None
    handler.assert_called_once()
    pipeline._water_leak.process.assert_not_called()
    pipeline._inference_executor.submit.assert_not_called()
    assert "fast_motion" not in pipeline._auxiliary_errors


@pytest.mark.parametrize("raw", [
    ("shadow", 0.2), ("shadow", float("nan")), ("shadow", float("inf")),
    ("shadow", "0.99"), (["shadow"], 0.99), ("shadow",), ("unknown", 0.99),
])
def test_untrusted_classifier_cannot_drop_water_or_original_appearance(simulate, raw):
    pipeline, frame = _flow_pipeline(anomalous=True)
    pipeline._classifier = MagicMock()
    pipeline._classifier.vocabulary = ["shadow", "wrench"]
    pipeline._classifier.classify_with_evidence.return_value = (
        {"label": raw[0], "confidence": raw[1], "bbox": [160, 120, 320, 240]}
        if len(raw) == 2 else raw
    )
    pipeline._classifier_min_score = 0.0
    pipeline._classifier_suppress = {"shadow"}
    pipeline._classifier_high_risk = set()
    pipeline._classifier_low_risk = set()
    result = simulate(pipeline, frame)
    assert result is not None
    alerts = pipeline._last_emitted_alerts
    assert any(a.detection_type == DetectionType.WATER_LEAK for a in alerts)
    assert any(
        call.kwargs["detection_type"] in (DetectionType.ANOMALY, DetectionType.HYBRID)
        and call.kwargs["anomaly_score"] >= 0.99
        and call.kwargs["classification_label"] is None
        for call in pipeline._alert_grader.evaluate.call_args_list
    ), [call.kwargs for call in pipeline._alert_grader.evaluate.call_args_list]
    assert all(a.classification_label is None for a in alerts)


def test_trusted_classifier_suppression_still_preserves_water(simulate):
    pipeline, frame = _flow_pipeline(anomalous=True)
    pipeline._classifier = MagicMock()
    pipeline._classifier.vocabulary = ["shadow"]
    pipeline._classifier.classify_with_evidence.return_value = {
        "label": "shadow", "confidence": 0.99, "bbox": [160, 120, 320, 240],
    }
    pipeline._classifier_min_score = 0.0
    pipeline._classifier_suppress = {"shadow"}
    pipeline._classifier_high_risk = set()
    pipeline._classifier_low_risk = set()
    result = simulate(pipeline, frame)
    assert result is not None
    assert result.detection_type == DetectionType.WATER_LEAK
    assert all(a.detection_type == DetectionType.WATER_LEAK for a in pipeline._last_emitted_alerts)


def test_classifier_exception_is_local_and_recorded(simulate):
    pipeline, frame = _flow_pipeline(anomalous=True)
    pipeline._classifier = MagicMock()
    pipeline._classifier.classify_with_evidence.side_effect = TimeoutError("fake classification timeout")
    pipeline._classifier_min_score = 0.0
    result = simulate(pipeline, frame)
    assert result is not None
    assert "fake classification timeout" in pipeline._auxiliary_errors["classifier"]
    assert pipeline._last_detection_failed is False


def test_segmenter_failure_preserves_alert_and_failure_count(simulate, monkeypatch):
    pipeline, frame = _flow_pipeline(water=False, anomalous=True)
    pipeline._segmenter = MagicMock()
    pipeline._segmenter.segment.side_effect = TimeoutError("fake segmentation timeout")
    pipeline._segmenter_min_score = 0.0
    pipeline._segmenter_max_points = 1
    monkeypatch.setattr("argus.anomaly.segmenter.extract_peak_points", lambda *a, **kw: [(1, 1)])
    result = simulate(pipeline, frame)
    assert result is not None
    assert pipeline._segmenter_consecutive_failures == 1
    assert "fake segmentation timeout" in pipeline._auxiliary_errors["segmenter"]


def test_scene_mismatch_still_blocks_all_auxiliary_calls(simulate, monkeypatch):
    from argus.core.runtime import FrameOutcomeKind

    pipeline, frame = _flow_pipeline(anomalous=True)
    pipeline._fast_motion = MagicMock()
    pipeline._classifier = MagicMock()
    pipeline._segmenter = MagicMock()
    monkeypatch.setattr(pipeline, "_apply_scene_guard", lambda f, *args: (f, False))
    monkeypatch.setattr(pipeline, "get_scene_guard_status", lambda: {
        "state": "mismatch", "usable": False, "reason": "fake wrong scene",
    })
    assert simulate(pipeline, frame) is None
    assert pipeline.last_frame_outcome_kind is FrameOutcomeKind.BLOCKED
    pipeline._fast_motion.process.assert_not_called()
    pipeline._water_leak.process.assert_not_called()
    pipeline._classifier.classify.assert_not_called()
    pipeline._classifier.classify_with_evidence.assert_not_called()
    pipeline._segmenter.segment.assert_not_called()
    pipeline._inference_executor.submit.assert_not_called()


def test_person_filter_keeps_independent_water_channel_live(simulate):
    pipeline, frame = _flow_pipeline(anomalous=True)
    yolo = Future()
    yolo.set_result(ObjectDetectionResult(has_persons=True))
    anomaly = Future()
    anomaly.set_result(AnomalyResult(anomaly_score=0.0, anomaly_map=None, is_anomalous=False, threshold=0.7))
    pipeline._inference_executor.submit.side_effect = [yolo, anomaly]
    result = simulate(pipeline, frame)
    assert result is not None
    assert result.detection_type == DetectionType.WATER_LEAK
    pipeline._water_leak.process.assert_called_once()
