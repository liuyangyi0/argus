"""Tests for the external sensor fusion store (argus.sensors.fusion)."""

from __future__ import annotations

import threading
import time

import pytest

from argus.alerts.grader import AlertGrader
from argus.config.schema import (
    AlertConfig,
    AlertSeverity,
    SeverityThresholds,
    SuppressionConfig,
    TemporalConfirmation,
    ZonePriority,
)
from argus.sensors.fusion import SensorFusion


# ── Basic behaviour ──


def test_set_get_returns_multiplier() -> None:
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("cam1", "z1", 1.5, valid_for_s=30)
    assert fusion.get_multiplier("cam1", "z1") == pytest.approx(1.5)


def test_get_unknown_returns_1() -> None:
    fusion = SensorFusion(enabled=True)
    assert fusion.get_multiplier("cam1", "z1") == 1.0


def test_expiration_returns_1() -> None:
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("cam1", "z1", 2.0, valid_for_s=0.1)
    assert fusion.get_multiplier("cam1", "z1") == pytest.approx(2.0)
    time.sleep(0.2)
    assert fusion.get_multiplier("cam1", "z1") == 1.0


def test_zone_wildcard_fallback() -> None:
    """set ("cam1", "*") should answer any zone lookup on cam1."""
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("cam1", "*", 1.4, valid_for_s=30)
    assert fusion.get_multiplier("cam1", "zone1") == pytest.approx(1.4)
    assert fusion.get_multiplier("cam1", "zone-xyz") == pytest.approx(1.4)
    # But a different camera still falls through to 1.0
    assert fusion.get_multiplier("cam2", "zone1") == 1.0


def test_global_wildcard_fallback() -> None:
    """set ("*", "*") should answer any lookup."""
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("*", "*", 0.5, valid_for_s=30)
    assert fusion.get_multiplier("any-cam", "any-zone") == pytest.approx(0.5)
    assert fusion.get_multiplier("other-cam", "z99") == pytest.approx(0.5)


def test_exact_wins_over_wildcards() -> None:
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("*", "*", 2.0, valid_for_s=30)
    fusion.set_signal("cam1", "*", 1.5, valid_for_s=30)
    fusion.set_signal("cam1", "z1", 1.2, valid_for_s=30)
    assert fusion.get_multiplier("cam1", "z1") == pytest.approx(1.2)
    assert fusion.get_multiplier("cam1", "z2") == pytest.approx(1.5)
    assert fusion.get_multiplier("camX", "z1") == pytest.approx(2.0)


# ── Bounds validation ──


def test_multiplier_zero_raises() -> None:
    fusion = SensorFusion(enabled=True)
    with pytest.raises(ValueError):
        fusion.set_signal("cam1", "z1", 0.0, valid_for_s=30)


def test_multiplier_too_large_raises() -> None:
    fusion = SensorFusion(enabled=True)
    with pytest.raises(ValueError):
        fusion.set_signal("cam1", "z1", 10.0, valid_for_s=30)


def test_multiplier_below_min_raises() -> None:
    fusion = SensorFusion(enabled=True)
    with pytest.raises(ValueError):
        fusion.set_signal("cam1", "z1", 0.05, valid_for_s=30)


def test_multiplier_boundary_values_ok() -> None:
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("cam1", "z1", 0.1, valid_for_s=30)
    assert fusion.get_multiplier("cam1", "z1") == pytest.approx(0.1)
    fusion.set_signal("cam1", "z1", 5.0, valid_for_s=30)
    assert fusion.get_multiplier("cam1", "z1") == pytest.approx(5.0)


def test_negative_valid_for_raises() -> None:
    fusion = SensorFusion(enabled=True)
    with pytest.raises(ValueError):
        fusion.set_signal("cam1", "z1", 1.5, valid_for_s=-1.0)


# ── Disabled fusion ──


def test_disabled_fusion_returns_1_regardless() -> None:
    fusion = SensorFusion(enabled=False)
    fusion.set_signal("cam1", "z1", 3.0, valid_for_s=60)
    assert fusion.get_multiplier("cam1", "z1") == 1.0
    fusion.set_signal("*", "*", 2.0, valid_for_s=60)
    assert fusion.get_multiplier("anything", "anything") == 1.0


def test_enable_toggle_respected() -> None:
    fusion = SensorFusion(enabled=False)
    fusion.set_signal("cam1", "z1", 2.5, valid_for_s=60)
    assert fusion.get_multiplier("cam1", "z1") == 1.0
    fusion.enabled = True
    assert fusion.get_multiplier("cam1", "z1") == pytest.approx(2.5)


# ── Removal / listing ──


def test_remove_signal() -> None:
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("cam1", "z1", 1.5, valid_for_s=60)
    assert fusion.remove_signal("cam1", "z1") is True
    assert fusion.get_multiplier("cam1", "z1") == 1.0
    assert fusion.remove_signal("cam1", "z1") is False


def test_active_signals_lists_only_valid() -> None:
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("cam1", "z1", 1.5, valid_for_s=30)
    fusion.set_signal("cam2", "z9", 0.8, valid_for_s=0.05)
    time.sleep(0.1)
    signals = fusion.active_signals()
    keys = {(s["camera_id"], s["zone_id"]) for s in signals}
    assert ("cam1", "z1") in keys
    assert ("cam2", "z9") not in keys


# ── Thread safety ──


def test_thread_safety_many_writers() -> None:
    """100 threads each set a unique key; readers on the main thread must
    see every committed entry without exceptions."""
    fusion = SensorFusion(enabled=True)
    errors: list[BaseException] = []

    def writer(idx: int) -> None:
        try:
            for j in range(20):
                fusion.set_signal(f"cam{idx}", f"zone{j}", 1.0 + (idx % 10) * 0.1,
                                  valid_for_s=30)
        except BaseException as exc:  # pragma: no cover — surfaces via assert
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # Every (camX, zoneY) we wrote should be retrievable
    for idx in range(100):
        for j in range(20):
            expected = 1.0 + (idx % 10) * 0.1
            assert fusion.get_multiplier(f"cam{idx}", f"zone{j}") == pytest.approx(expected)


# ── End-to-end: grader applies the fusion multiplier ──


def _grader_config() -> AlertConfig:
    """AlertConfig tuned so a single call can trigger an alert."""
    return AlertConfig(
        severity_thresholds=SeverityThresholds(info=0.3, low=0.5, medium=0.7, high=0.9),
        temporal=TemporalConfirmation(
            max_gap_seconds=10.0,
            min_spatial_overlap=0.0,
            evidence_lambda=0.80,
            evidence_threshold=0.5,
        ),
        suppression=SuppressionConfig(
            same_zone_window_seconds=10.0,
            same_camera_window_seconds=5.0,
        ),
    )


def test_grader_applies_fusion_multiplier() -> None:
    """Score 0.5 with fusion=1.5 -> adjusted_score=0.75 -> MEDIUM (medium threshold=0.7)."""
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("cam1", "z1", 1.5, valid_for_s=30)

    grader = AlertGrader(_grader_config(), sensor_fusion=fusion)
    alert = grader.evaluate(
        camera_id="cam1",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.5,
        frame_number=1,
    )
    assert alert is not None
    assert alert.severity == AlertSeverity.MEDIUM


def test_grader_without_fusion_baseline_behaviour() -> None:
    """Same input, no fusion -> adjusted_score=0.5 -> LOW severity."""
    grader = AlertGrader(_grader_config())
    alert = grader.evaluate(
        camera_id="cam1",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.5,
        frame_number=1,
    )
    assert alert is not None
    assert alert.severity == AlertSeverity.LOW


def test_grader_fusion_disabled_no_effect() -> None:
    """Even with a signal in the store, disabled fusion returns neutral 1.0."""
    fusion = SensorFusion(enabled=False)
    fusion.set_signal("cam1", "z1", 1.5, valid_for_s=30)

    grader = AlertGrader(_grader_config(), sensor_fusion=fusion)
    alert = grader.evaluate(
        camera_id="cam1",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.5,
        frame_number=1,
    )
    assert alert is not None
    assert alert.severity == AlertSeverity.LOW


def test_grader_fusion_damp_below_info_suppresses_alert() -> None:
    """fusion=0.3 on score=0.5 → adjusted=0.15 < info threshold → no alert."""
    fusion = SensorFusion(enabled=True)
    fusion.set_signal("cam1", "z1", 0.3, valid_for_s=30)

    grader = AlertGrader(_grader_config(), sensor_fusion=fusion)
    alert = grader.evaluate(
        camera_id="cam1",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.5,
        frame_number=1,
    )
    assert alert is None
