"""Tests for the EventBus publish/subscribe system."""

import threading
import time

import pytest

from argus.core.event_bus import (
    AlertRaised,
    CameraConnected,
    CameraDisconnected,
    DriftDetected,
    Event,
    EventBus,
    FrameAnalyzed,
    ModelLoaded,
    PipelineStageCompleted,
)


class TestEventBus:
    def test_subscribe_and_publish(self, event_bus):
        received = []
        event_bus.subscribe(FrameAnalyzed, received.append)

        event = FrameAnalyzed(camera_id="cam-01", frame_number=1, anomaly_score=0.5)
        event_bus.publish(event)

        assert len(received) == 1
        assert received[0].camera_id == "cam-01"
        assert received[0].anomaly_score == 0.5

    def test_multiple_subscribers(self, event_bus):
        results_a = []
        results_b = []
        event_bus.subscribe(AlertRaised, results_a.append)
        event_bus.subscribe(AlertRaised, results_b.append)

        event_bus.publish(AlertRaised(alert_id="ALT-001", severity="high"))

        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_event_type_isolation(self, event_bus):
        frame_events = []
        alert_events = []
        event_bus.subscribe(FrameAnalyzed, frame_events.append)
        event_bus.subscribe(AlertRaised, alert_events.append)

        event_bus.publish(FrameAnalyzed(camera_id="cam-01"))
        event_bus.publish(AlertRaised(alert_id="ALT-001"))

        assert len(frame_events) == 1
        assert len(alert_events) == 1

    def test_unsubscribe(self, event_bus):
        received = []
        event_bus.subscribe(FrameAnalyzed, received.append)

        event_bus.publish(FrameAnalyzed(camera_id="cam-01"))
        assert len(received) == 1

        event_bus.unsubscribe(FrameAnalyzed, received.append)
        event_bus.publish(FrameAnalyzed(camera_id="cam-02"))
        assert len(received) == 1  # no new event

    def test_unsubscribe_nonexistent_handler(self, event_bus):
        # Should not raise
        event_bus.unsubscribe(FrameAnalyzed, lambda e: None)

    def test_handler_exception_does_not_propagate(self, event_bus):
        def bad_handler(event):
            raise RuntimeError("handler crashed")

        received = []
        event_bus.subscribe(FrameAnalyzed, bad_handler)
        event_bus.subscribe(FrameAnalyzed, received.append)

        # Should not raise, and second handler should still run
        event_bus.publish(FrameAnalyzed(camera_id="cam-01"))
        assert len(received) == 1

    def test_clear(self, event_bus):
        received = []
        event_bus.subscribe(FrameAnalyzed, received.append)
        event_bus.clear()

        event_bus.publish(FrameAnalyzed(camera_id="cam-01"))
        assert len(received) == 0

    def test_thread_safety(self, event_bus):
        received = []
        event_bus.subscribe(FrameAnalyzed, received.append)

        def publish_events():
            for i in range(100):
                event_bus.publish(FrameAnalyzed(camera_id=f"cam-{i}"))

        threads = [threading.Thread(target=publish_events) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 400

    def test_event_timestamp(self):
        before = time.time()
        event = FrameAnalyzed(camera_id="cam-01")
        after = time.time()

        assert before <= event.timestamp <= after

    def test_duplicate_subscribe_ignored(self, event_bus):
        received = []
        handler = received.append
        event_bus.subscribe(FrameAnalyzed, handler)
        event_bus.subscribe(FrameAnalyzed, handler)  # duplicate

        event_bus.publish(FrameAnalyzed(camera_id="cam-01"))
        assert len(received) == 1  # called once, not twice


    def test_has_subscribers(self, event_bus):
        assert event_bus.has_subscribers(FrameAnalyzed) is False

        event_bus.subscribe(FrameAnalyzed, lambda e: None)
        assert event_bus.has_subscribers(FrameAnalyzed) is True
        assert event_bus.has_subscribers(AlertRaised) is False


class TestDomainEvents:
    def test_frame_analyzed_defaults(self):
        event = FrameAnalyzed()
        assert event.camera_id == ""
        assert event.anomaly_score == 0.0
        assert event.mog2_skipped is False

    def test_frame_analyzed_stage_latencies_immutable(self):
        from types import MappingProxyType
        event = FrameAnalyzed(stage_latencies=MappingProxyType({"mog2": 1.5}))
        with pytest.raises(TypeError):
            event.stage_latencies["new_key"] = 2.0

    def test_alert_raised(self):
        event = AlertRaised(
            alert_id="ALT-001",
            camera_id="cam-01",
            zone_id="zone-a",
            severity="high",
            anomaly_score=0.95,
        )
        assert event.severity == "high"

    def test_camera_events(self):
        connected = CameraConnected(camera_id="cam-01")
        disconnected = CameraDisconnected(camera_id="cam-01", reason="timeout")
        assert connected.camera_id == "cam-01"
        assert disconnected.reason == "timeout"

    def test_model_loaded(self):
        event = ModelLoaded(
            camera_id="cam-01",
            model_type="patchcore",
            model_path="/models/cam01.ckpt",
            load_time_seconds=1.5,
        )
        assert event.load_time_seconds == 1.5

    def test_drift_detected(self):
        event = DriftDetected(
            camera_id="cam-01",
            ks_statistic=0.15,
            p_value=0.001,
        )
        assert event.ks_statistic == 0.15

    def test_pipeline_stage_completed(self):
        event = PipelineStageCompleted(
            camera_id="cam-01",
            stage="anomaly",
            duration_seconds=0.042,
            frame_number=100,
        )
        assert event.stage == "anomaly"
