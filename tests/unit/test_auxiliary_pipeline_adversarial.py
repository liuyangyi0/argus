"""Adversarial lifecycle/attribution checks with fake recognition heads only."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.anomaly.detector import AnomalyResult
from argus.core.pipeline import DetectionPipeline
from argus.core.runtime import FrameOutcomeKind
from argus.person.detector import ObjectDetectionResult

from test_auxiliary_pipeline_contract import _flow_pipeline


@pytest.fixture
def simulate(request):
    guard = next((plugin for plugin in request.config.pluginmanager.get_plugins()
                  if type(plugin).__name__ == "FunctionalGuard"), None)
    assert guard is not None, "Use the guarded local auxiliary suite"
    return guard.simulate_frame


def test_read_started_before_shutdown_cannot_process_a_late_frame():
    pipeline, frame = _flow_pipeline()
    entered, release = threading.Event(), threading.Event()
    results = []

    def read():
        entered.set()
        assert release.wait(2)
        return frame

    def run():
        try:
            results.append(pipeline.run_once())
        except Exception as exc:
            results.append(exc)

    pipeline._camera.read_latest.side_effect = read
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(1)
        pipeline.shutdown()
        release.set()
        thread.join(1)
        assert results == [None]
        assert pipeline.last_frame_outcome_kind is FrameOutcomeKind.BLOCKED
        pipeline._inference_executor.submit.assert_not_called()
        pipeline._camera.request_reconnect.assert_not_called()
    finally:
        release.set()
        thread.join(1)


def test_stopped_pipeline_does_not_read_or_reconnect():
    pipeline, _ = _flow_pipeline()
    pipeline.shutdown()
    assert pipeline.run_once() is None
    pipeline._camera.read_latest.assert_not_called()
    pipeline._camera.request_reconnect.assert_not_called()


def test_pipeline_shutdown_retains_failed_retired_cleanup_for_retry():
    pipeline, _ = _flow_pipeline()
    component = MagicMock()
    component.shutdown.side_effect = RuntimeError("fake cleanup failed")
    pipeline._auxiliary_retired["classifier"] = component
    with pytest.raises(RuntimeError, match="fake cleanup failed"):
        pipeline.shutdown()
    assert pipeline._auxiliary_retired["classifier"] is component
    pipeline._camera.stop.assert_called_once()
    component.shutdown.side_effect = None
    component.get_status.return_value = {"state": "stopped", "busy": False, "cleanup_pending": False}
    pipeline.shutdown()
    assert component.shutdown.call_count == 2
    assert "classifier" not in pipeline._auxiliary_retired


@pytest.mark.parametrize("map_kind", ["absent", "constant", "subthreshold"])
def test_unlocalized_classification_is_not_even_requested(simulate, map_kind):
    pipeline, frame = _flow_pipeline(water=False, anomalous=True)
    values = None if map_kind == "absent" else np.ones((32, 32), dtype=np.float32)
    if map_kind == "subthreshold":
        values[:] = 0
        values[8:16, 8:16] = 0.01
    yolo, appearance = Future(), Future()
    yolo.set_result(ObjectDetectionResult(has_persons=False))
    appearance.set_result(AnomalyResult(
        anomaly_score=0.99, anomaly_map=values, is_anomalous=True, threshold=0.7,
    ))
    pipeline._inference_executor.submit.side_effect = [yolo, appearance]
    pipeline._classifier = MagicMock()
    pipeline._classifier.vocabulary = ["shadow"]
    pipeline._classifier.classify_with_evidence.return_value = {
        "label": "shadow", "confidence": 0.99, "bbox": [160, 120, 320, 240],
    }
    pipeline._classifier_min_score = 0
    pipeline._classifier_suppress = {"shadow"}
    pipeline._classifier_high_risk = set()
    pipeline._classifier_low_risk = set()
    simulate(pipeline, frame)
    if map_kind in {"absent", "constant"}:
        pipeline._classifier.classify_with_evidence.assert_not_called()
    assert all(
        call.kwargs.get("classification_label") is None
        for call in pipeline._alert_grader.evaluate.call_args_list
    )
    assert pipeline._alert_grader.evaluate.called


def test_legacy_high_risk_conflict_cannot_suppress_the_tool(simulate):
    pipeline, frame = _flow_pipeline(water=False, anomalous=True)
    pipeline._classifier = MagicMock()
    pipeline._classifier.vocabulary = ["wrench"]
    pipeline._classifier.classify_with_evidence.return_value = {
        "label": "wrench", "confidence": 0.99, "bbox": [160, 120, 320, 240],
    }
    pipeline._classifier_min_score = 0
    pipeline._classifier_suppress = {"wrench"}
    pipeline._classifier_high_risk = {"wrench"}
    pipeline._classifier_low_risk = {"wrench"}
    simulate(pipeline, frame)
    assert pipeline._alert_grader.evaluate.called
    assert any(
        call.kwargs.get("classification_label") == "wrench"
        for call in pipeline._alert_grader.evaluate.call_args_list
    )


def test_stale_stationary_track_does_not_suppress_unlocalized_current_anomaly(simulate):
    pipeline, frame = _flow_pipeline(water=False, anomalous=True)
    pipeline._physics_postproc.extract_regions.return_value = []
    pipeline._physics_tracker.update.return_value = SimpleNamespace(active_tracks=[
        SimpleNamespace(track_id=9, suppressed=True, last_seen_frame=0),
    ])
    simulate(pipeline, frame)
    assert pipeline._alert_grader.evaluate.called
    diag = pipeline.get_diagnostics_buffer().get_recent(1)[0]
    assert not any(r.suppression_reason == "stationary_track_suppressed" for r in diag.candidate_records)


def test_stationary_evidence_requires_complete_current_support():
    values = np.zeros((20, 20), dtype=np.float32)
    values[5:10, 5:10] = 0.9
    region = SimpleNamespace(x=5, y=5, width=5, height=5, centroid_x=7.0, centroid_y=7.0)
    track = SimpleNamespace(suppressed=True, last_seen_frame=2, centroid_x=7.0, centroid_y=7.0)
    assert DetectionPipeline._stationary_evidence_is_current(values, [region], [track], 2)
    assert not DetectionPipeline._stationary_evidence_is_current(values, [region], [track], 3)
    values[15, 15] = 0.9
    assert not DetectionPipeline._stationary_evidence_is_current(values, [region], [track], 2)


def test_disabled_desired_state_never_uses_an_attached_old_classifier(simulate):
    pipeline, frame = _flow_pipeline(water=False, anomalous=True)
    pipeline._classifier = MagicMock()
    pipeline._classifier_config = SimpleNamespace(enabled=False)
    pipeline._classifier_min_score = 0
    simulate(pipeline, frame)
    pipeline._classifier.classify_with_evidence.assert_not_called()
    assert pipeline._alert_grader.evaluate.called
