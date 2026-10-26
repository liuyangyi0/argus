"""Tests for replay HTTP endpoints (/api/replay/*).

Covers the three UX additions:
- POST /api/replay/{alert_id}/clips (operator clip persistence)
- DELETE /api/replay/{alert_id}/clips/{index}
- GET  /api/replay/{alert_id}/reference?frame_offset_seconds=<float>

The store is wired directly onto app.state after create_app, mirroring the
production path where __main__.py constructs the store.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, AuthConfig
from argus.core.alert_ring_buffer import (
    FrameSnapshot,
    RecordingStatus,
    SolidifiedRecording,
)
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.dashboard.auth import create_session_token, hash_password
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
def replay_media(monkeypatch):
    """Replace only media encoding/extraction, never the replay/store logic.

    These HTTP tests must not select a host-dependent OpenCV/FFmpeg encoder.
    The temporary .mp4 files contain explicitly fake JSON carrying the input
    JPEGs so store publication/renaming and exact reference indices are real.
    Codec correctness belongs to separate media tests, not this mock suite.
    """
    state = SimpleNamespace(encodings=[], extractions=[], fail_at=None)

    class FakeEncoder:
        def __init__(self, path, *, fps, width, height, **_options):
            self.path = Path(path)
            self.frames = []
            state.encodings.append({"fps": fps, "width": width, "height": height})

        def write_jpeg_frame(self, jpeg):
            if state.fail_at == "write":
                raise OSError("simulated frame encoding failure")
            self.frames.append(base64.b64encode(jpeg).decode("ascii"))

        def finalize(self):
            if state.fail_at == "finalize":
                raise OSError("simulated encoder finalization failure")
            payload = json.dumps({"test_only_jpeg_frames": self.frames}).encode("ascii")
            self.path.write_bytes(payload)
            return len(payload)

    def extract_frame(path, index):
        state.extractions.append((Path(path), index))
        frames = json.loads(Path(path).read_bytes())["test_only_jpeg_frames"]
        if index < 0 or index >= len(frames):
            return None
        return base64.b64decode(frames[index], validate=True)

    monkeypatch.setattr("argus.storage.alert_recording.Mp4Encoder", FakeEncoder)
    monkeypatch.setattr("argus.storage.alert_recording.extract_frame_jpeg", extract_frame)
    # /reference imports the helper at request time, unlike store.load_frame.
    monkeypatch.setattr("argus.core.video_encoder.extract_frame_jpeg", extract_frame)
    return state


@pytest.fixture
def store(tmp_path, replay_media):
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


class TestReplayVideoRanges:
    @staticmethod
    def _serve_file(store, monkeypatch, tmp_path, payload: bytes):
        video_path = tmp_path / "recording.mp4"
        video_path.write_bytes(payload)
        monkeypatch.setattr(store, "get_video_path", lambda _alert_id: video_path)
        monkeypatch.setattr(
            store,
            "load_metadata",
            lambda _alert_id: {"status": "complete"},
        )
        return video_path

    def test_bounded_range_returns_exact_bytes(
        self, client, store, monkeypatch, tmp_path,
    ):
        payload = bytes(range(100))
        self._serve_file(store, monkeypatch, tmp_path, payload)

        response = client.get(
            "/api/replay/ALT-RANGE/video",
            headers={"Range": "bytes=10-19"},
        )

        assert response.status_code == 206
        assert response.content == payload[10:20]
        assert response.headers["content-range"] == "bytes 10-19/100"
        assert response.headers["content-length"] == "10"
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["cache-control"] == "private, max-age=3600"

    @pytest.mark.parametrize(
        ("range_header", "expected", "content_range"),
        [
            ("bytes=90-", bytes(range(90, 100)), "bytes 90-99/100"),
            ("bytes=-7", bytes(range(93, 100)), "bytes 93-99/100"),
            ("bytes=-500", bytes(range(100)), "bytes 0-99/100"),
            ("bytes=95-999", bytes(range(95, 100)), "bytes 95-99/100"),
        ],
    )
    def test_open_suffix_and_clamped_ranges(
        self,
        client,
        store,
        monkeypatch,
        tmp_path,
        range_header,
        expected,
        content_range,
    ):
        self._serve_file(store, monkeypatch, tmp_path, bytes(range(100)))

        response = client.get(
            "/api/replay/ALT-RANGE/video",
            headers={"Range": range_header},
        )

        assert response.status_code == 206
        assert response.content == expected
        assert response.headers["content-range"] == content_range

    @pytest.mark.parametrize(
        "range_header",
        [
            "bytes=100-",
            "bytes=20-10",
            "bytes=abc-def",
            "bytes=-0",
            "bytes=",
            "bytes=+1-2",
            "items=0-1",
            "bytes=0-1,4-5",
        ],
    )
    def test_invalid_or_unsatisfiable_range_returns_416_envelope(
        self,
        client,
        store,
        monkeypatch,
        tmp_path,
        range_header,
    ):
        self._serve_file(store, monkeypatch, tmp_path, bytes(range(100)))

        response = client.get(
            "/api/replay/ALT-RANGE/video",
            headers={"Range": range_header},
        )

        assert response.status_code == 416
        assert response.headers["content-range"] == "bytes */100"
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["cache-control"] == "private, no-store"
        assert response.json() == {
            "code": 41600,
            "msg": "请求的字节范围无效",
            "data": None,
        }

    def test_full_video_response_is_private_and_range_capable(
        self, client, store, monkeypatch, tmp_path,
    ):
        payload = bytes(range(32))
        self._serve_file(store, monkeypatch, tmp_path, payload)

        response = client.get("/api/replay/ALT-RANGE/video")

        assert response.status_code == 200
        assert response.content == payload
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["cache-control"] == "private, max-age=3600"

    def test_in_progress_video_ranges_are_never_cached(
        self,
        client,
        store,
        monkeypatch,
        tmp_path,
    ):
        self._serve_file(store, monkeypatch, tmp_path, bytes(range(32)))
        monkeypatch.setattr(
            store,
            "load_metadata",
            lambda _alert_id: {"status": "recording"},
        )

        response = client.get(
            "/api/replay/ALT-RANGE/video",
            headers={"Range": "bytes=0-7"},
        )

        assert response.status_code == 206
        assert response.headers["cache-control"] == "private, no-store"

    @pytest.mark.parametrize("metadata_result", [None, {}])
    def test_video_cache_fails_closed_without_complete_metadata(
        self,
        client,
        store,
        monkeypatch,
        tmp_path,
        metadata_result,
    ):
        self._serve_file(store, monkeypatch, tmp_path, bytes(range(32)))
        monkeypatch.setattr(
            store,
            "load_metadata",
            lambda _alert_id: metadata_result,
        )

        response = client.get("/api/replay/ALT-RANGE/video")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "private, no-store"

    def test_video_cache_fails_closed_when_metadata_read_raises(
        self,
        client,
        store,
        monkeypatch,
        tmp_path,
    ):
        self._serve_file(store, monkeypatch, tmp_path, bytes(range(32)))

        def fail_metadata(_alert_id):
            raise OSError("metadata unavailable")

        monkeypatch.setattr(store, "load_metadata", fail_metadata)

        response = client.get("/api/replay/ALT-RANGE/video")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "private, no-store"

    def test_range_against_empty_file_is_unsatisfiable(
        self, client, store, monkeypatch, tmp_path,
    ):
        self._serve_file(store, monkeypatch, tmp_path, b"")

        response = client.get(
            "/api/replay/ALT-RANGE/video",
            headers={"Range": "bytes=0-"},
        )

        assert response.status_code == 416
        assert response.headers["content-range"] == "bytes */0"


class TestReplayOptionalMediaErrors:
    @pytest.mark.parametrize("resource", ["frame", "heatmap"])
    def test_missing_optional_frame_uses_error_envelope(
        self, client, resource,
    ):
        response = client.get(f"/api/replay/unknown/{resource}/0")

        assert response.status_code == 404
        body = response.json()
        assert body["code"] == 40400
        assert body["data"] is None

    @pytest.mark.parametrize("resource", ["frame", "heatmap"])
    def test_negative_optional_frame_index_uses_error_envelope(
        self, client, resource,
    ):
        response = client.get(f"/api/replay/unknown/{resource}/-1")

        assert response.status_code == 400
        body = response.json()
        assert body["code"] == 40000
        assert body["data"] is None

    @pytest.mark.parametrize("resource", ["frame", "heatmap"])
    def test_unconfigured_store_uses_error_envelope(
        self, client, resource,
    ):
        client.app.state.recording_store = None

        response = client.get(f"/api/replay/unknown/{resource}/0")

        assert response.status_code == 503
        assert response.json() == {
            "code": 50300,
            "msg": "录像存储未配置",
            "data": None,
        }


class TestReplayCacheHeaders:
    def test_replay_json_success_is_never_cached(self, client, store):
        store.save(_recording())

        response = client.get("/api/replay/ALT-RT-001/metadata")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"

    def test_replay_json_error_is_never_cached(self, client):
        response = client.get("/api/replay/does-not-exist/metadata")

        assert response.status_code == 404
        assert response.headers["cache-control"] == "no-store"


class TestReplayMockMediaPersistence:
    def test_saved_metadata_signals_and_frame_indices_agree(self, store, replay_media):
        recording = _recording(trigger_ts=1_700_000_000.0, frame_count=8)
        path, size = store.save(recording)

        metadata = store.load_metadata(recording.alert_id)
        assert metadata["frame_count"] == len(recording.frames)
        assert metadata["trigger_frame_index"] == recording.trigger_frame_index
        assert metadata["start_timestamp"] == recording.frames[0].timestamp
        assert metadata["end_timestamp"] == recording.frames[-1].timestamp
        assert metadata["fps"] == recording.fps
        assert (metadata["width"], metadata["height"]) == (320, 240)
        assert replay_media.encodings == [{"fps": 15, "width": 320, "height": 240}]
        video = Path(path) / metadata["video_file"]
        assert size == video.stat().st_size
        signals = store.load_signals(recording.alert_id)
        assert signals["timestamps"] == [frame.timestamp for frame in recording.frames]
        assert signals["frame_numbers"] == [frame.frame_number for frame in recording.frames]
        for index in (0, 3, 7):
            assert store.load_frame(recording.alert_id, index) == recording.frames[index].frame_jpeg
        assert [index for _path, index in replay_media.extractions] == [0, 3, 7]

    @pytest.mark.parametrize("fail_at", ["write", "finalize"])
    def test_encoding_failure_preserves_previous_recording_and_retry(
        self, store, replay_media, fail_at,
    ):
        original = _recording(trigger_ts=1_700_000_000.0)
        path, _size = store.save(original)
        old_metadata = store.load_metadata(original.alert_id)
        old_signals = store.load_signals(original.alert_id)
        replacement = _recording(trigger_ts=original.trigger_timestamp, frame_count=3)
        replay_media.fail_at = fail_at

        with pytest.raises(OSError, match="simulated"):
            store.save(replacement)

        assert store.load_metadata(original.alert_id) == old_metadata
        assert store.load_signals(original.alert_id) == old_signals
        assert store.load_frame(original.alert_id, 7) == original.frames[7].frame_jpeg
        assert not list(Path(path).parent.glob(f".{original.alert_id}.*"))
        replay_media.fail_at = None
        store.save(replacement)
        assert store.load_metadata(original.alert_id)["frame_count"] == 3
        assert store.load_frame(original.alert_id, 2) == replacement.frames[2].frame_jpeg


def _auth_client_for_role(db, store, tmp_path, role: str) -> TestClient:
    username = f"replay-{role}"
    db.create_user(username, hash_password("replay-secret"), role, username)
    config = ArgusConfig(
        auth=AuthConfig(enabled=True, api_token="replay-test-token"),
    )
    app = create_app(
        database=db,
        health_monitor=HealthMonitor(),
        alerts_dir=str(tmp_path / "auth-alerts"),
        config=config,
    )
    app.state.recording_store = store
    client = TestClient(app)
    token = create_session_token(username, role, app.state.session_secret)
    client.cookies.set("argus_session", token)
    return client


class TestReplayRbac:
    @pytest.mark.parametrize(
        ("method", "path", "json_body"),
        [
            ("GET", "/api/replay/missing/metadata", None),
            ("GET", "/api/replay/missing/signals", None),
            ("GET", "/api/replay/missing/video", None),
            ("GET", "/api/replay/missing/frame/0", None),
            ("GET", "/api/replay/missing/heatmap/0", None),
            ("GET", "/api/replay/missing/reference", None),
            ("GET", "/api/replay/storyboard/missing", None),
            ("GET", "/api/physics/missing/trajectory", None),
            ("POST", "/api/replay/missing/pin-frame", {"index": 0}),
            (
                "POST",
                "/api/replay/missing/clips",
                {"start_index": 0, "end_index": 1},
            ),
            ("DELETE", "/api/replay/missing/clips/0", None),
        ],
    )
    def test_viewer_is_forbidden_before_resource_lookup(
        self,
        db,
        store,
        tmp_path,
        method,
        path,
        json_body,
    ):
        client = _auth_client_for_role(db, store, tmp_path, "viewer")

        response = client.request(method, path, json=json_body)

        assert response.status_code == 403
        assert response.json()["code"] == 40300

    @pytest.mark.parametrize(
        ("method", "path", "json_body", "expected_status"),
        [
            ("GET", "/api/replay/missing/metadata", None, 404),
            ("GET", "/api/replay/missing/signals", None, 404),
            ("GET", "/api/replay/missing/video", None, 404),
            ("GET", "/api/replay/missing/frame/0", None, 404),
            ("GET", "/api/replay/missing/heatmap/0", None, 404),
            ("GET", "/api/replay/missing/reference", None, 200),
            ("GET", "/api/replay/storyboard/missing", None, 404),
            ("GET", "/api/physics/missing/trajectory", None, 404),
            ("POST", "/api/replay/missing/pin-frame", {"index": 0}, 404),
            (
                "POST",
                "/api/replay/missing/clips",
                {"start_index": 0, "end_index": 1},
                404,
            ),
            ("DELETE", "/api/replay/missing/clips/0", None, 404),
        ],
    )
    def test_operator_reaches_replay_resource_handlers(
        self,
        db,
        store,
        tmp_path,
        method,
        path,
        json_body,
        expected_status,
    ):
        client = _auth_client_for_role(db, store, tmp_path, "operator")

        response = client.request(method, path, json=json_body)

        assert response.status_code == expected_status


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
        assert bad.status_code == 422
        assert bad.json()["code"] == 42200

        # non-int types
        bad2 = client.post(
            "/api/replay/ALT-RT-001/clips",
            json={"start_index": "0", "end_index": 3},
        )
        assert bad2.status_code == 422
        assert bad2.json()["code"] == 42200

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
    def test_reference_offset_shifts_target_frame(self, client, store, replay_media):
        """Two different offsets should request two different in-recording frames."""
        # Reference recording — a full day before the alert so the /reference
        # handler finds it via the yesterday fallback.
        historical_ts = 1_700_000_000.0
        historical = _recording(
            alert_id="HISTORICAL-001",
            trigger_ts=historical_ts,
            frame_count=30,
        )
        store.save(historical)

        # The alert that will be replayed — current time, same camera.
        current = _recording(alert_id="ALT-RT-001", trigger_ts=historical_ts + 86400)
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
        assert base64.b64decode(frame0) == historical.frames[29].frame_jpeg

        # A -5s offset should request the exact earlier input JPEG. Codec
        # behavior is mocked; selecting the right recording/index is not.
        resp_neg = client.get(
            "/api/replay/ALT-RT-001/reference",
            params={"frame_offset_seconds": -5.0},
        )
        assert resp_neg.status_code == 200
        body_neg = resp_neg.json()
        assert body_neg["code"] == 0
        assert body_neg["data"]["available"] is True
        frame_neg = body_neg["data"]["frame_base64"]
        assert base64.b64decode(frame_neg) == historical.frames[25].frame_jpeg

        # Positive offset (post-trigger) should succeed too — availability
        # depends on whether an in-range frame exists; when the offset walks
        # past the recording end, it clamps to the last frame.
        resp_pos = client.get(
            "/api/replay/ALT-RT-001/reference",
            params={"frame_offset_seconds": 0.5},
        )
        assert resp_pos.status_code == 200
        assert resp_pos.json()["code"] == 0

        assert [index for _path, index in replay_media.extractions] == [29, 25, 29]
        assert all(path.parent.name == "HISTORICAL-001" for path, _index in replay_media.extractions)
        # Different offsets must not accidentally return the same cached frame.
        assert frame0 != frame_neg

    def test_reference_offset_zero_matches_default(self, client, store):
        historical_ts = 1_700_000_000.0
        store.save(_recording(alert_id="HISTORICAL-002", trigger_ts=historical_ts))
        store.save(_recording(alert_id="ALT-RT-002", trigger_ts=historical_ts + 86400))

        r1 = client.get("/api/replay/ALT-RT-002/reference")
        r2 = client.get(
            "/api/replay/ALT-RT-002/reference",
            params={"frame_offset_seconds": 0.0},
        )
        assert r1.status_code == 200 and r2.status_code == 200
        # Base64 payloads should be identical when offset is zero.
        assert r1.json()["data"]["frame_base64"] == r2.json()["data"]["frame_base64"]

    def test_reference_rejects_invalid_explicit_time(self, client, store):
        store.save(_recording(alert_id="ALT-RT-TIME"))

        response = client.get(
            "/api/replay/ALT-RT-TIME/reference",
            params={"time": "25:61:00"},
        )

        assert response.status_code == 400
        assert response.json()["msg"] == "时间格式无效，请使用 HH:MM:SS"
