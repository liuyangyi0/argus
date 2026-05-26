from __future__ import annotations

from unittest.mock import MagicMock

from argus.dashboard.degradation_bridge import publish_pipeline_degradation


def test_anomaly_entered_reports_model_fallback_and_broadcasts():
    ws_manager = MagicMock()
    degradation_manager = MagicMock()
    payload = {
        "type": "entered",
        "component": "anomaly",
        "camera_id": "cam_01",
        "reason": "load_failed",
    }

    publish_pipeline_degradation(
        ws_manager,
        degradation_manager,
        "system_degradation",
        payload,
    )

    degradation_manager.report.assert_called_once_with(
        category="model_fallback",
        camera_id="cam_01",
    )
    ws_manager.broadcast.assert_called_once_with("system_degradation", payload)


def test_anomaly_recovered_resolves_model_fallback_and_broadcasts():
    ws_manager = MagicMock()
    degradation_manager = MagicMock()
    payload = {
        "type": "recovered",
        "component": "anomaly",
        "camera_id": "cam_01",
    }

    publish_pipeline_degradation(
        ws_manager,
        degradation_manager,
        "system_degradation",
        payload,
    )

    degradation_manager.resolve_by_category.assert_called_once_with(
        category="model_fallback",
        camera_id="cam_01",
    )
    ws_manager.broadcast.assert_called_once_with("system_degradation", payload)


def test_non_anomaly_event_only_broadcasts():
    ws_manager = MagicMock()
    degradation_manager = MagicMock()
    payload = {"type": "entered", "component": "segmenter", "camera_id": "cam_01"}

    publish_pipeline_degradation(
        ws_manager,
        degradation_manager,
        "system_degradation",
        payload,
    )

    degradation_manager.report.assert_not_called()
    degradation_manager.resolve_by_category.assert_not_called()
    ws_manager.broadcast.assert_called_once_with("system_degradation", payload)
