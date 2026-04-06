"""Tests for DegradationStateMachine, LockState, and DegradationEvent."""

from unittest.mock import MagicMock

from argus.core.degradation import (
    DegradationEvent,
    DegradationState,
    DegradationStateMachine,
    LockState,
)


class TestLockState:
    def test_values(self):
        assert LockState.UNLOCKED.value == "unlocked"
        assert LockState.LOCKED.value == "locked"
        assert LockState.CLEARING.value == "clearing"


class TestDegradationState:
    def test_values(self):
        assert DegradationState.NOMINAL.value == "nominal"
        assert DegradationState.DEGRADED_SEGMENTER.value == "degraded_segmenter"
        assert DegradationState.DEGRADED_DETECTOR.value == "degraded_detector"
        assert DegradationState.BACKBONE_FAILED.value == "backbone_failed"
        assert DegradationState.RESTARTING.value == "restarting"


class TestDegradationStateMachine:
    def test_initial_state(self):
        sm = DegradationStateMachine("cam-01")
        assert sm.state == DegradationState.NOMINAL

    def test_valid_transition(self):
        sm = DegradationStateMachine("cam-01")
        result = sm.transition(DegradationState.RESTARTING, "test restart")
        assert result is True
        assert sm.state == DegradationState.RESTARTING

    def test_invalid_transition(self):
        sm = DegradationStateMachine("cam-01")
        # NOMINAL -> NOMINAL is a no-op (returns True)
        # but DEGRADED_SEGMENTER -> BACKBONE_FAILED is invalid
        sm.transition(DegradationState.DEGRADED_SEGMENTER, "segmenter failed")
        result = sm.transition(DegradationState.BACKBONE_FAILED, "bad transition")
        assert result is False
        assert sm.state == DegradationState.DEGRADED_SEGMENTER

    def test_noop_same_state(self):
        sm = DegradationStateMachine("cam-01")
        result = sm.transition(DegradationState.NOMINAL, "already nominal")
        assert result is True
        assert len(sm.get_recent_events()) == 0

    def test_event_recording(self):
        sm = DegradationStateMachine("cam-01")
        sm.transition(DegradationState.RESTARTING, "model reload")
        sm.transition(DegradationState.NOMINAL, "reload success")

        events = sm.get_recent_events()
        assert len(events) == 2
        assert events[0].from_state == DegradationState.NOMINAL
        assert events[0].to_state == DegradationState.RESTARTING
        assert events[0].reason == "model reload"
        assert events[0].camera_id == "cam-01"
        assert events[1].from_state == DegradationState.RESTARTING
        assert events[1].to_state == DegradationState.NOMINAL

    def test_event_is_frozen(self):
        sm = DegradationStateMachine("cam-01")
        sm.transition(DegradationState.RESTARTING, "test")
        event = sm.get_recent_events()[0]
        assert isinstance(event, DegradationEvent)
        # frozen dataclass — attribute assignment should raise
        try:
            event.reason = "modified"
            assert False, "Should have raised"
        except AttributeError:
            pass

    def test_max_events_bounded(self):
        sm = DegradationStateMachine("cam-01", max_events=3)
        # Do 4 transitions (cycle through states)
        sm.transition(DegradationState.RESTARTING, "r1")
        sm.transition(DegradationState.NOMINAL, "n1")
        sm.transition(DegradationState.RESTARTING, "r2")
        sm.transition(DegradationState.NOMINAL, "n2")

        events = sm.get_recent_events()
        assert len(events) == 3  # oldest evicted

    def test_reset(self):
        sm = DegradationStateMachine("cam-01")
        sm.transition(DegradationState.DEGRADED_DETECTOR, "detector failed")
        sm.reset()
        assert sm.state == DegradationState.NOMINAL

    def test_reset_when_already_nominal(self):
        sm = DegradationStateMachine("cam-01")
        sm.reset()
        assert sm.state == DegradationState.NOMINAL
        assert len(sm.get_recent_events()) == 0

    def test_audit_logger_called(self):
        mock_audit = MagicMock()
        sm = DegradationStateMachine("cam-01", audit_logger=mock_audit)
        sm.transition(DegradationState.RESTARTING, "test")

        mock_audit.log.assert_called_once_with(
            user="system",
            action="degradation_transition",
            target_type="camera",
            target_id="cam-01",
            detail="nominal -> restarting: test",
        )

    def test_audit_logger_exception_does_not_crash(self):
        mock_audit = MagicMock()
        mock_audit.log.side_effect = RuntimeError("db down")
        sm = DegradationStateMachine("cam-01", audit_logger=mock_audit)
        # Should not raise
        result = sm.transition(DegradationState.RESTARTING, "test")
        assert result is True
        assert sm.state == DegradationState.RESTARTING

    def test_full_degradation_cycle(self):
        """Simulate: NOMINAL -> RESTARTING -> DEGRADED_DETECTOR -> RESTARTING -> NOMINAL."""
        sm = DegradationStateMachine("cam-01")
        assert sm.transition(DegradationState.RESTARTING, "failures exceeded")
        assert sm.transition(DegradationState.DEGRADED_DETECTOR, "reload failed")
        assert sm.transition(DegradationState.RESTARTING, "retry")
        assert sm.transition(DegradationState.NOMINAL, "reload succeeded")
        assert sm.state == DegradationState.NOMINAL
        assert len(sm.get_recent_events()) == 4


class TestCusumSnapshot:
    """Test CusumSnapshot via AlertGrader.get_cusum_state()."""

    def test_cusum_snapshot_from_grader(self):
        from argus.alerts.grader import AlertGrader, CusumSnapshot
        from argus.config.schema import AlertConfig, ZonePriority

        grader = AlertGrader(AlertConfig())
        # No tracker yet
        assert grader.get_cusum_state("cam-01:zone-a") is None

        # Feed a score to create a tracker
        import numpy as np
        grader.evaluate(
            camera_id="cam-01",
            zone_id="zone-a",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.8,
            frame_number=1,
        )
        snapshot = grader.get_cusum_state("cam-01:zone-a")
        assert snapshot is not None
        assert isinstance(snapshot, CusumSnapshot)
        assert snapshot.evidence > 0
        assert snapshot.max_score > 0

    def test_get_all_cusum_states(self):
        from argus.alerts.grader import AlertGrader
        from argus.config.schema import AlertConfig, ZonePriority

        grader = AlertGrader(AlertConfig())
        grader.evaluate("cam-01", "z1", ZonePriority.STANDARD, 0.8, 1)
        grader.evaluate("cam-02", "z1", ZonePriority.STANDARD, 0.9, 1)

        states = grader.get_all_cusum_states()
        assert "cam-01:z1" in states
        assert "cam-02:z1" in states
        assert len(states) == 2
