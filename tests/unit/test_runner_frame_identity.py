"""CameraInferenceRunner frame identity snapshot coverage."""

from __future__ import annotations

from types import SimpleNamespace

from argus.core.degradation import LockState
from argus.core.runner import CameraInferenceRunner


def test_runner_snapshot_includes_latest_frame_identity():
    identity = {
        "camera_id": "cam_01",
        "source_declared": "0",
        "source_runtime": "rtsp://127.0.0.1:8554/cam_01",
        "stream_id": "cam_01",
        "frame_index": 99,
        "capture_wall_ts": 1710000000.0,
        "capture_monotonic_ts": 10.0,
        "pipeline_ingest_ts": 1710000000.1,
        "preview_stream_ts": None,
        "snapshot_id": None,
        "runtime_source_hash": "hash",
    }
    pipeline = SimpleNamespace(
        camera_config=SimpleNamespace(camera_id="cam_01"),
        stats=SimpleNamespace(),
        lock_state=LockState.UNLOCKED,
        last_heartbeat_time=0.0,
        get_detector_status=lambda: SimpleNamespace(
            mode="ssim_fallback",
            model_path=None,
        ),
        get_cusum_states=lambda: {},
        get_frame_identity_status=lambda: {"latest": identity},
    )
    runner = CameraInferenceRunner(pipeline)

    snapshot = runner.get_snapshot()

    assert snapshot.latest_frame_identity == identity
