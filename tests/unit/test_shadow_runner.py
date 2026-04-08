"""Tests for the shadow inference runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.anomaly.shadow_runner import ShadowRunner, _BATCH_FLUSH_SIZE
from argus.storage.models import Base, ShadowInferenceLog


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture()
def mock_detector():
    det = MagicMock()
    result = MagicMock()
    result.anomaly_score = 0.42
    det.predict.return_value = result
    return det


@pytest.fixture()
def runner(session_factory, mock_detector):
    r = ShadowRunner(
        shadow_model_path=Path("/fake/shadow/model"),
        shadow_version_id="shadow-v1",
        production_version_id="prod-v1",
        camera_id="cam-01",
        session_factory=session_factory,
        sample_rate=3,
        threshold=0.5,
    )
    # Inject the mock detector directly (skip lazy loading)
    r._detector = mock_detector
    return r


def _blank_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestSamplingRate:

    def test_only_runs_on_nth_frame(self, runner, mock_detector):
        """Shadow should only run every sample_rate frames."""
        for _ in range(9):
            runner.run_shadow(_blank_frame(), production_score=0.3)

        # sample_rate=3, so frames 3, 6, 9 should trigger (3 calls)
        assert mock_detector.predict.call_count == 3

    def test_first_frame_not_sampled(self, runner, mock_detector):
        """Frame 1 is not a multiple of sample_rate, so no prediction."""
        runner.run_shadow(_blank_frame(), production_score=0.3)
        assert mock_detector.predict.call_count == 0

    def test_sample_rate_1_runs_every_frame(self, session_factory, mock_detector):
        r = ShadowRunner(
            shadow_model_path=Path("/fake"),
            shadow_version_id="s1",
            production_version_id="p1",
            camera_id="cam-01",
            session_factory=session_factory,
            sample_rate=1,
        )
        r._detector = mock_detector
        for _ in range(5):
            r.run_shadow(_blank_frame())
        assert mock_detector.predict.call_count == 5


class TestNoAlerts:

    def test_shadow_never_triggers_alerts(self, runner, mock_detector):
        """Shadow results must never produce alerts."""
        # Make shadow score very high (above threshold)
        high_result = MagicMock()
        high_result.anomaly_score = 0.99
        mock_detector.predict.return_value = high_result

        # Run enough frames to trigger shadow
        for _ in range(3):
            result = runner.run_shadow(_blank_frame(), production_score=0.1)
            assert result is None  # run_shadow returns None, no alert


class TestBatchFlush:

    def test_logs_buffered_until_batch_size(self, runner, session_factory):
        """Logs should be buffered and only flushed at batch size."""
        # Run enough frames for some shadow predictions but below flush threshold
        count = (_BATCH_FLUSH_SIZE - 1) * runner._sample_rate
        for _ in range(count):
            runner.run_shadow(_blank_frame(), production_score=0.3)

        # Logs should still be pending (not flushed)
        with session_factory() as session:
            db_count = session.query(ShadowInferenceLog).count()
            assert db_count == 0

        # Manual flush should write them
        runner.flush()
        with session_factory() as session:
            db_count = session.query(ShadowInferenceLog).count()
            assert db_count == _BATCH_FLUSH_SIZE - 1

    def test_auto_flush_at_batch_size(self, runner, session_factory):
        """Logs auto-flush when batch size is reached."""
        count = _BATCH_FLUSH_SIZE * runner._sample_rate
        for _ in range(count):
            runner.run_shadow(_blank_frame(), production_score=0.3)

        with session_factory() as session:
            db_count = session.query(ShadowInferenceLog).count()
            assert db_count == _BATCH_FLUSH_SIZE


class TestLogContent:

    def test_log_records_correct_data(self, runner, session_factory, mock_detector):
        result = MagicMock()
        result.anomaly_score = 0.65
        mock_detector.predict.return_value = result

        for _ in range(3):
            runner.run_shadow(
                _blank_frame(),
                production_score=0.3,
                production_alerted=False,
            )

        runner.flush()

        with session_factory() as session:
            logs = session.query(ShadowInferenceLog).all()
            assert len(logs) == 1
            log = logs[0]
            assert log.camera_id == "cam-01"
            assert log.shadow_version_id == "shadow-v1"
            assert log.production_version_id == "prod-v1"
            assert log.shadow_score == 0.65
            assert log.production_score == 0.3
            assert log.shadow_would_alert is True  # 0.65 >= 0.5
            assert log.production_alerted is False
            assert log.latency_ms is not None
            assert log.latency_ms >= 0


class TestErrorHandling:

    def test_prediction_error_does_not_crash(self, runner, mock_detector):
        """Shadow prediction errors should be silently logged."""
        mock_detector.predict.side_effect = RuntimeError("GPU OOM")

        # Should not raise
        for _ in range(6):
            runner.run_shadow(_blank_frame(), production_score=0.3)

    def test_load_failure_skips_predictions(self, session_factory):
        """If model fails to load, subsequent calls are no-ops."""
        r = ShadowRunner(
            shadow_model_path=Path("/nonexistent/model"),
            shadow_version_id="s1",
            production_version_id="p1",
            camera_id="cam-01",
            session_factory=session_factory,
            sample_rate=1,
        )
        # Force load failure (exhaust all retries)
        r._load_failures = r._max_load_retries

        for _ in range(5):
            r.run_shadow(_blank_frame())

        r.flush()
        with session_factory() as session:
            assert session.query(ShadowInferenceLog).count() == 0
