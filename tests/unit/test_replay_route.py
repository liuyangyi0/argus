"""Tests for replay HTTP endpoints (/api/replay/*).

Covers the three UX additions:
- POST /api/replay/{alert_id}/clips (operator clip persistence)
- DELETE /api/replay/{alert_id}/clips/{index}
- GET  /api/replay/{alert_id}/reference?frame_offset_seconds=<float>

The store is wired directly onto app.state after create_app, mirroring the
production path where __main__.py constructs the store.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from argus.core.alert_ring_buffer import (
    FrameSnapshot,
    RecordingStatus,
    SolidifiedRecording,
)
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.storage.alert_recording import AlertRecordingStore
from argus.storage.database import Database


def _jpeg(w: int = 320, h: int = 240, value: int = 128) -> bytes:
    frame = np.full((h, w, 3), value, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _recording(
    alert_id: str = "ALT-RT-001",
    camera_id: str = "cam_01",
    trigger_ts: float | None = None,
    frame_count: int = 8,
    status: RecordingStatus = RecordingStatus.COMPLETE,
) -> SolidifiedRecording:
    ts = trigger_ts if trigger_ts is not None else time.time()
    frames = [
        FrameSnapshot(
            timestamp=ts - frame_count + i,
            # Shade frames distinctly but keep pixel value in uint8 range so
            # the JPEGs decode to recognisably different content — this is
            # what lets the offset test distinguish two reference frames.
            frame_jpeg=_jpeg(value=(20 + i * 7) % 220 + 10),
            anomaly_score=0.1 + i * 0.05,
            simplex_score=None,
            cusum_evidence={"cam_01:default": 0.4 + i * 0.05},
            yolo_persons=[],
            frame_number=i,
        )
        for i in range(frame_count)
    ]
    return SolidifiedRecording(
        alert_id=alert_id,
        camera_id=camera_id,
        severity="medium",
        trigger_timestamp=ts,
        trigger_frame_index=frame_count - 1,
        frames=frames,
        fps=15,
        status=status,
    )


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def store(tmp_path):
    return AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))


@pytest.fixture
def client(db, store, tmp_path):
    app = create_app(
        database=db,
        health_monitor=HealthMonitor(),
        alerts_dir=str(tmp_path / "alerts"),
    )
    app.state.recording_store = store
    return TestClient(app)


class TestReplayClipsEndpoints:
    def test_post_clip_persists_and_returns_list(self, client, store):
        store.save(_recording())

        resp = client.post(
            "/api/replay/ALT-RT-001/clips",
            json={"start_index": 1, "end_index": 5, "label": "interesting"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["clip"]["start_index"] == 1
        assert body["data"]["clip"]["end_index"] == 5
        assert body["data"]["clip"]["label"] == "interesting"
        assert len(body["data"]["clips"]) == 1

        # Signals response should surface the persisted clip after reload.
        sig_resp = client.get("/api/replay/ALT-RT-001/signals")
        assert sig_resp.status_code == 200
        sig_body = sig_resp.json()
        assert sig_body["code"] == 0
        assert sig_body["data"]["clips"][0]["label"] == "interesting"

    def test_post_clip_rejects_bad_payload(self, client, store):
        store.save(_recording())

        # end_index < start_index
        bad = client.post(
            "/api/replay/ALT-RT-001/clips",
            json={"start_index": 5, "end_index": 2},
        )
        assert bad.status_code == 400
        assert bad.json()["code"] != 0

        # non-int types
        bad2 = client.post(
            "/api/replay/ALT-RT-001/clips",
            json={"start_index": "0", "end_index": 3},
        )
        assert bad2.status_code == 400

        # out-of-range index
        over = client.post(
            "/api/replay/ALT-RT-001/clips",
            json={"start_index": 0, "end_index": 99},
        )
        assert over.status_code == 400

    def test_post_clip_unknown_alert_returns_404(self, client):
        resp = client.post(
            "/api/replay/does-not-exist/clips",
            json={"start_index": 0, "end_index": 1},
        )
        assert resp.status_code == 404

    def test_delete_clip_removes_by_index(self, client, store):
        store.save(_recording())
        store.add_clip("ALT-RT-001", 0, 2, "a")
        store.add_clip("ALT-RT-001", 3, 5, "b")

        resp = client.delete("/api/replay/ALT-RT-001/clips/0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert len(body["data"]["clips"]) == 1
        assert body["data"]["clips"][0]["label"] == "b"

        # Deleting a non-existent index yields 404.
        missing = client.delete("/api/replay/ALT-RT-001/clips/42")
        assert missing.status_code == 404


class TestReplayReferenceOffset:
    def test_reference_offset_shifts_target_frame(self, client, store, tmp_path):
        """Two different offsets should request two different in-recording frames."""
        # Reference recording — a full day before the alert so the /reference
        # handler finds it via the yesterday fallback.
        historical_ts = time.time() - 86400
        historical = _recording(
            alert_id="HISTORICAL-001",
            trigger_ts=historical_ts,
            frame_count=30,
        )
        store.save(historical)

        # The alert that will be replayed — current time, same camera.
        current = _recording(alert_id="ALT-RT-001")
        store.save(current)

        resp0 = client.get("/api/replay/ALT-RT-001/reference")
        assert resp0.status_code == 200
        body0 = resp0.json()
        assert body0["code"] == 0
        # The historical recording ended roughly at historical_ts, so the
        # trigger time itself lands near the last frame. We just assert it's
        # available and base64 non-empty.
        assert body0["data"]["available"] is True
        frame0 = body0["data"]["frame_base64"]
        assert frame0

        # A -5s offset should move the target to an earlier frame. The JPEG
        # bytes are produced by different frames in the MP4 so they will
        # differ (even with default CRF, adjacent frames differ unless fed
        # identical JPEGs — here frames are shaded with per-index values).
        resp_neg = client.get(
            "/api/replay/ALT-RT-001/reference",
            params={"frame_offset_seconds": -5.0},
        )
        assert resp_neg.status_code == 200
        body_neg = resp_neg.json()
        assert body_neg["code"] == 0
        assert body_neg["data"]["available"] is True
        frame_neg = body_neg["data"]["frame_base64"]
        assert frame_neg

        # Positive offset (post-trigger) should succeed too — availability
        # depends on whether an in-range frame exists; when the offset walks
        # past the recording end, it clamps to the last frame.
        resp_pos = client.get(
            "/api/replay/ALT-RT-001/reference",
            params={"frame_offset_seconds": 0.5},
        )
        assert resp_pos.status_code == 200
        assert resp_pos.json()["code"] == 0

        # The -5s offset should yield a visibly different frame vs the 0s one
        # (they index into different positions within the same historical MP4).
        assert frame0 != frame_neg

    def test_reference_offset_zero_matches_default(self, client, store):
        historical_ts = time.time() - 86400
        store.save(_recording(alert_id="HISTORICAL-002", trigger_ts=historical_ts))
        store.save(_recording(alert_id="ALT-RT-002"))

        r1 = client.get("/api/replay/ALT-RT-002/reference")
        r2 = client.get(
            "/api/replay/ALT-RT-002/reference",
            params={"frame_offset_seconds": 0.0},
        )
        assert r1.status_code == 200 and r2.status_code == 200
        # Base64 payloads should be identical when offset is zero.
        assert r1.json()["data"]["frame_base64"] == r2.json()["data"]["frame_base64"]
