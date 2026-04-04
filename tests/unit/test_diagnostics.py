"""Tests for per-frame diagnostics buffer (DET-008) and threshold evaluation (DET-005)."""

import threading

from argus.core.diagnostics import (
    DiagnosticsBuffer,
    FrameDiagnostics,
    FrameScoreRecord,
    StageResult,
)


def test_buffer_append_and_get_recent():
    buf = DiagnosticsBuffer(maxlen=10, score_maxlen=5)

    for i in range(15):
        buf.append(FrameDiagnostics(
            frame_number=i, timestamp=float(i), camera_id="cam1"
        ))

    # Buffer maxlen is 10, so oldest entries evicted
    recent = buf.get_recent(10)
    assert len(recent) == 10
    assert recent[0].frame_number == 5
    assert recent[-1].frame_number == 14


def test_buffer_get_recent_fewer_than_requested():
    buf = DiagnosticsBuffer(maxlen=100)

    for i in range(3):
        buf.append(FrameDiagnostics(
            frame_number=i, timestamp=float(i), camera_id="cam1"
        ))

    recent = buf.get_recent(50)
    assert len(recent) == 3


def test_score_buffer_independence():
    buf = DiagnosticsBuffer(maxlen=100, score_maxlen=5)

    for i in range(10):
        buf.append_score(FrameScoreRecord(
            frame_number=i, timestamp=float(i), anomaly_score=i * 0.1, was_alert=False
        ))

    scores = buf.get_scores()
    assert len(scores) == 5  # score_maxlen = 5
    assert scores[0].frame_number == 5  # oldest surviving


def test_evaluate_threshold_empty():
    buf = DiagnosticsBuffer()
    result = buf.evaluate_threshold(0.5)
    assert result["total_frames"] == 0
    assert result["would_alert_count"] == 0


def test_evaluate_threshold_counts():
    buf = DiagnosticsBuffer(score_maxlen=100)

    # Add scores: 0.0, 0.1, ..., 0.9
    for i in range(10):
        buf.append_score(FrameScoreRecord(
            frame_number=i,
            timestamp=float(i),
            anomaly_score=i * 0.1,
            was_alert=(i * 0.1) >= 0.7,  # current threshold 0.7
        ))

    result = buf.evaluate_threshold(0.5)
    assert result["total_frames"] == 10
    # Scores >= 0.5: 0.5, 0.6, 0.7, 0.8, 0.9 = 5
    assert result["would_alert_count"] == 5
    # Current alerts (threshold 0.7): 0.7, 0.8, 0.9 = 3
    assert result["current_alert_count"] == 3
    assert result["new_threshold"] == 0.5
    assert len(result["score_distribution"]) == 10


def test_evaluate_threshold_distribution():
    buf = DiagnosticsBuffer(score_maxlen=100)

    # All scores are 0.35
    for i in range(5):
        buf.append_score(FrameScoreRecord(
            frame_number=i, timestamp=float(i), anomaly_score=0.35, was_alert=False
        ))

    result = buf.evaluate_threshold(0.5)
    # All in bucket 3 (0.3-0.4)
    assert result["score_distribution"][3] == 5
    assert sum(result["score_distribution"]) == 5


def test_buffer_thread_safety():
    """Concurrent appends from multiple threads should not corrupt buffer."""
    buf = DiagnosticsBuffer(maxlen=1000)
    errors = []

    def writer(start):
        try:
            for i in range(100):
                buf.append(FrameDiagnostics(
                    frame_number=start + i,
                    timestamp=float(start + i),
                    camera_id="cam1",
                ))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(buf) == 500


def test_buffer_len():
    buf = DiagnosticsBuffer(maxlen=100)
    assert len(buf) == 0

    buf.append(FrameDiagnostics(frame_number=0, timestamp=0.0, camera_id="c"))
    assert len(buf) == 1


def test_stage_result_defaults():
    sr = StageResult(stage_name="test", duration_ms=1.5)
    assert sr.skipped is False
    assert sr.skip_reason == ""
    assert sr.metadata == {}


def test_frame_diagnostics_defaults():
    fd = FrameDiagnostics(frame_number=1, timestamp=1.0, camera_id="cam1")
    assert fd.total_duration_ms == 0.0
    assert fd.stages == []
    assert fd.anomaly_score == 0.0
    assert fd.is_anomalous is False
    assert fd.alert_emitted is False
    assert fd.pipeline_mode == "active"
