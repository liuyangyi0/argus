"""Pure in-memory fan-out tests; no camera, decoder, or inference is used."""

import numpy as np
import pytest

from argus.capture.camera import FrameData
from argus.capture.frame_hub import FrameHub
from argus.core.frame_identity import build_frame_identity


def make_frame(number=1, *, epoch=1, session="session"):
    return FrameData(
        np.full((2, 3, 3), number, dtype=np.uint8), "cam", 12.0, number, (3, 2),
        identity=build_frame_identity(
            camera_id="cam", source_declared="mock", source_runtime="mock",
            stream_id=None, frame_index=number, source_session_id=session,
            source_epoch=epoch,
        ),
    )


def test_frame_hub_broadcast_metadata_and_pixels_are_isolated():
    hub = FrameHub("cam")
    first = hub.subscribe("first")
    second = hub.subscribe("second")
    hub.activate("session", 1)
    original = make_frame()
    assert hub.publish(original)
    original.frame[:] = 99
    a, b = first.get(0), second.get(0)
    assert a is not None and b is not None
    assert np.all(a.frame == 1)
    assert np.shares_memory(a.frame, b.frame)
    with pytest.raises(ValueError):
        a.frame[0, 0, 0] = 4
    with pytest.raises(ValueError):
        a.frame.setflags(write=True)
    a.identity.pipeline_ingest_ts = 88
    a.identity.frame_index = 99
    a.frame = a.frame.copy()
    a.frame[:] = 55
    assert b.identity.pipeline_ingest_ts is None
    assert b.identity.frame_index == 1
    assert np.all(b.frame == 1)
    assert hub.latest_frame.identity.frame_index == 1


def test_frame_hub_latest_and_ordered_have_separate_backpressure():
    hub = FrameHub("cam")
    latest = hub.subscribe("preview")
    ordered = hub.subscribe("motion", mode="ordered_window", capacity=3)
    hub.activate("session", 1)
    for number in range(1, 6):
        hub.publish(make_frame(number))
    assert latest.get(0).frame_number == 5
    assert latest.status.dropped_frames == 4
    assert [ordered.get(0).frame_number for _ in range(3)] == [3, 4, 5]
    assert ordered.status.dropped_frames == 2
    assert latest.get(0) is None
    latest.close()
    hub.publish(make_frame(6))
    assert ordered.get(0).frame_number == 6


def test_frame_hub_epoch_change_revokes_queued_and_late_frames():
    hub = FrameHub("cam")
    consumer = hub.subscribe("task")
    hub.activate("session", 1)
    hub.publish(make_frame())
    hub.activate("session", 2)
    assert consumer.get(0) is None
    assert hub.latest_frame is None
    assert not hub.publish(make_frame())
    assert hub.publish(make_frame(epoch=2))
    hub.invalidate()
    assert consumer.get(0) is None
    assert not hub.publish(make_frame(epoch=2))


def test_frame_identity_does_not_invent_source_frame_or_pts():
    identity = make_frame().identity
    assert identity.source_frame_index is None
    assert identity.pts is None
    assert identity.time_base_num is None and identity.time_base_den is None
    assert identity.frame_index_provenance == "capture_sequence"
    assert identity.to_dict()["source_session_id"] == "session"


@pytest.mark.parametrize("mode,capacity", [("bad", 1), ("latest", 0), ("ordered_window", True)])
def test_invalid_subscription_rejected(mode, capacity):
    with pytest.raises(ValueError):
        FrameHub("cam").subscribe("bad", mode=mode, capacity=capacity)


def test_duplicate_subscription_and_mismatched_identity_rejected():
    hub = FrameHub("cam")
    hub.subscribe("same")
    with pytest.raises(ValueError, match="already registered"):
        hub.subscribe("same")
    hub.activate("session", 1)
    bad = make_frame()
    bad.identity.camera_id = "other"
    assert not hub.publish(bad)


def test_old_subscription_cleanup_cannot_close_replacement():
    hub = FrameHub("cam")
    old = hub.subscribe("task")
    old.close()
    replacement = hub.subscribe("task")
    hub.unsubscribe("task", subscription=old)
    hub.activate("session", 1)
    hub.publish(make_frame())
    assert replacement.get(0) is not None
