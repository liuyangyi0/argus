"""Smoke tests for the Cameras -> Alerts -> Replay -> Reports core API loop."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from argus.alerts.dispatcher import AlertDispatcher
from argus.alerts.grader import Alert
from argus.config.schema import AlertConfig, AlertSeverity
from argus.core.alert_ring_buffer import (
    FrameSnapshot,
    RecordingStatus,
    SolidifiedRecording,
)
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.storage.alert_recording import AlertRecordingStore
from argus.storage.database import Database
from argus.storage.models import AlertRecordingRecord


def _jpeg(w: int = 64, h: int = 48, value: int = 128) -> bytes:
    frame = np.full((h, w, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buf.tobytes()


def _recording(alert_id: str, camera_id: str, trigger_ts: float) -> SolidifiedRecording:
    frames = [
        FrameSnapshot(
            timestamp=trigger_ts - 1.5 + i * 0.5,
            frame_jpeg=_jpeg(value=80 + i * 20),
            anomaly_score=0.7 + i * 0.05,
            simplex_score=None,
            cusum_evidence={f"{camera_id}:zone_a": 1.2 + i},
            yolo_persons=[],
            frame_number=i,
            heatmap_raw=np.full((48, 64), 40 + i * 30, dtype=np.uint8),
        )
        for i in range(4)
    ]
    return SolidifiedRecording(
        alert_id=alert_id,
        camera_id=camera_id,
        severity="high",
        trigger_timestamp=trigger_ts,
        trigger_frame_index=2,
        frames=frames,
        fps=4,
        status=RecordingStatus.COMPLETE,
    )


def test_dispatch_alert_surfaces_evidence_replay_and_report_stats(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'argus.db'}")
    database.initialize()
    dispatcher: AlertDispatcher | None = None
    try:
        alerts_dir = tmp_path / "alerts"
        recording_store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))
        app = create_app(
            database=database,
            health_monitor=HealthMonitor(),
            alerts_dir=str(alerts_dir),
        )
        app.state.recording_store = recording_store
        client = TestClient(app)

        ws_events: list[tuple[str, dict]] = []
        dispatcher = AlertDispatcher(
            AlertConfig(),
            database,
            alerts_dir=alerts_dir,
            on_alert=lambda topic, payload: ws_events.append((topic, payload)),
        )

        alert_id = "ALT-core-loop-001"
        camera_id = "cam_01"
        trigger_ts = time.time()
        recording = _recording(alert_id, camera_id, trigger_ts)
        recording_path, recording_size = recording_store.save(recording)
        database.save_alert_recording(
            AlertRecordingRecord(
                alert_id=alert_id,
                camera_id=camera_id,
                severity="high",
                recording_path=recording_path,
                start_timestamp=recording.frames[0].timestamp,
                end_timestamp=recording.frames[-1].timestamp,
                trigger_timestamp=trigger_ts,
                frame_count=len(recording.frames),
                fps=recording.fps,
                file_size_bytes=recording_size,
                status=RecordingStatus.COMPLETE.value,
                width=64,
                height=48,
            )
        )

        snapshot = np.full((48, 64, 3), 130, dtype=np.uint8)
        heatmap = np.zeros((48, 64), dtype=np.float32)
        heatmap[12:36, 18:46] = 1.0
        alert = Alert(
            alert_id=alert_id,
            camera_id=camera_id,
            zone_id="zone_a",
            severity=AlertSeverity.HIGH,
            anomaly_score=0.97,
            timestamp=trigger_ts,
            frame_number=42,
            snapshot=snapshot,
            heatmap=heatmap,
            handling_policy="detail_required",
        )
        alert._solidified_recording = recording

        dispatcher.dispatch(alert)
        dispatcher.flush_db_queue()

        alert_events = [payload for topic, payload in ws_events if topic == "alerts"]
        assert len(alert_events) == 1
        ws_alert = alert_events[0]
        assert ws_alert["alert_id"] == alert_id
        assert ws_alert["has_recording"] is True
        assert ws_alert["recording_status"] == RecordingStatus.COMPLETE.value
        assert Path(ws_alert["snapshot_path"]).exists()
        assert Path(ws_alert["heatmap_path"]).exists()

        alerts_resp = client.get("/api/alerts/json")
        assert alerts_resp.status_code == 200
        alerts = alerts_resp.json()["data"]["alerts"]
        assert alerts[0]["alert_id"] == alert_id
        assert alerts[0]["has_recording"] is True
        assert alerts[0]["recording_status"] == RecordingStatus.COMPLETE.value
        assert alerts[0]["snapshot_path"]
        assert alerts[0]["heatmap_path"]

        detail_resp = client.get(f"/api/alerts/{alert_id}/detail")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()["data"]
        assert detail["has_recording"] is True
        assert detail["recording_status"] == RecordingStatus.COMPLETE.value
        assert detail["snapshot_path"] == alerts[0]["snapshot_path"]
        assert detail["heatmap_path"] == alerts[0]["heatmap_path"]

        metadata_resp = client.get(f"/api/replay/{alert_id}/metadata")
        assert metadata_resp.status_code == 200
        metadata = metadata_resp.json()["data"]
        assert metadata["alert_id"] == alert_id
        assert metadata["status"] == RecordingStatus.COMPLETE.value
        assert metadata["video_url"] == f"/api/replay/{alert_id}/video"

        signals_resp = client.get(f"/api/replay/{alert_id}/signals")
        assert signals_resp.status_code == 200
        signals = signals_resp.json()["data"]
        assert len(signals["timestamps"]) == len(recording.frames)
        assert signals["key_frames"]

        video_resp = client.get(
            f"/api/replay/{alert_id}/video",
            headers={"Range": "bytes=0-31"},
        )
        assert video_resp.status_code == 206
        assert video_resp.headers["content-range"].startswith("bytes 0-31/")

        frame_resp = client.get(f"/api/replay/{alert_id}/frame/0")
        assert frame_resp.status_code == 200
        assert frame_resp.headers["content-type"] == "image/jpeg"

        evidence_resp = client.get(f"/api/alerts/{alert_id}/evidence.zip")
        assert evidence_resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(evidence_resp.content)) as zf:
            names = set(zf.namelist())
            assert "manifest.json" in names
            assert "images/snapshot.jpg" in names
            assert "images/heatmap.jpg" in names
            assert "replay/metadata.json" in names
            assert "replay/signals.json" in names
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["alert_id"] == alert_id
            assert manifest["has_recording"] is True

        reports_resp = client.get("/api/reports/json")
        assert reports_resp.status_code == 200
        evidence = reports_resp.json()["data"]["evidence"]
        assert evidence["total_alerts"] == 1
        assert evidence["alerts_with_snapshot"] == 1
        assert evidence["alerts_with_heatmap"] == 1
        assert evidence["alerts_with_recording"] == 1
        assert evidence["evidence_complete_count"] == 1
        assert evidence["evidence_complete_rate"] == 100.0
    finally:
        if dispatcher is not None:
            dispatcher.close()
        database.close()
