"""Tests for InferenceRecord and related enums."""

import time

from argus.core.inference_record import (
    ConformalLevel,
    FinalDecision,
    InferenceRecord,
    PrefilterDecision,
)


class TestPrefilterDecision:
    def test_all_values(self):
        assert PrefilterDecision.PASSED.value == "passed"
        assert PrefilterDecision.SKIPPED_NO_CHANGE.value == "skipped_no_change"
        assert PrefilterDecision.SKIPPED_HEARTBEAT.value == "skipped_heartbeat"
        assert PrefilterDecision.SKIPPED_LOCK.value == "skipped_lock"

    def test_from_string(self):
        assert PrefilterDecision("passed") == PrefilterDecision.PASSED


class TestConformalLevel:
    def test_all_values(self):
        assert ConformalLevel.NONE.value == "none"
        assert ConformalLevel.INFO.value == "info"
        assert ConformalLevel.LOW.value == "low"
        assert ConformalLevel.MEDIUM.value == "medium"
        assert ConformalLevel.HIGH.value == "high"


class TestFinalDecision:
    def test_all_values(self):
        assert FinalDecision.NORMAL.value == "normal"
        assert FinalDecision.INFO.value == "info"
        assert FinalDecision.ALERT.value == "alert"
        assert FinalDecision.SUPPRESSED.value == "suppressed"


class TestInferenceRecord:
    def _make_record(self, **kwargs):
        defaults = dict(
            frame_id="abc123",
            camera_id="cam-01",
            timestamp_ns=time.time_ns(),
            model_version="v1.0",
            anomaly_score=0.85,
            prefilter_result=PrefilterDecision.PASSED,
            conformal_level=ConformalLevel.MEDIUM,
            final_decision=FinalDecision.ALERT,
        )
        defaults.update(kwargs)
        return InferenceRecord(**defaults)

    def test_basic_creation(self):
        rec = self._make_record()
        assert rec.frame_id == "abc123"
        assert rec.camera_id == "cam-01"
        assert rec.anomaly_score == 0.85
        assert rec.final_decision == FinalDecision.ALERT

    def test_defaults(self):
        rec = self._make_record()
        assert rec.health_metrics == {}
        assert rec.heatmap_ref is None
        assert rec.cusum_evidence == {}
        assert rec.sam2_objects == []
        assert rec.cross_cam_corroboration == {}
        assert rec.safety_channel_result is None
        assert rec.stage_durations_ms == {}

    def test_to_dict(self):
        rec = self._make_record(
            safety_channel_result=True,
            stage_durations_ms={"anomaly": 15.3},
        )
        d = rec.to_dict()
        assert d["frame_id"] == "abc123"
        assert d["prefilter_result"] == "passed"
        assert d["conformal_level"] == "medium"
        assert d["final_decision"] == "alert"
        assert d["safety_channel_result"] is True
        assert d["stage_durations_ms"] == {"anomaly": 15.3}

    def test_roundtrip(self):
        rec = self._make_record(
            health_metrics={"blur": 120.5},
            cusum_evidence={"cam-01:zone-a": 2.5},
            safety_channel_result=False,
        )
        d = rec.to_dict()
        restored = InferenceRecord.from_dict(d)
        assert restored.frame_id == rec.frame_id
        assert restored.camera_id == rec.camera_id
        assert restored.timestamp_ns == rec.timestamp_ns
        assert restored.model_version == rec.model_version
        assert restored.health_metrics == rec.health_metrics
        assert restored.prefilter_result == rec.prefilter_result
        assert restored.anomaly_score == rec.anomaly_score
        assert restored.cusum_evidence == rec.cusum_evidence
        assert restored.conformal_level == rec.conformal_level
        assert restored.safety_channel_result == rec.safety_channel_result
        assert restored.final_decision == rec.final_decision

    def test_from_dict_defaults(self):
        minimal = {
            "frame_id": "x",
            "camera_id": "c",
            "timestamp_ns": 0,
            "model_version": "v0",
        }
        rec = InferenceRecord.from_dict(minimal)
        assert rec.prefilter_result == PrefilterDecision.PASSED
        assert rec.conformal_level == ConformalLevel.NONE
        assert rec.final_decision == FinalDecision.NORMAL
        assert rec.anomaly_score == 0.0
