"""Tests for multi-scale sliding window anomaly detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.anomaly.detector import (
    AnomalibDetector,
    AnomalyResult,
    MultiScaleDetector,
    _TileInfo,
)


@pytest.fixture
def mock_base_detector():
    """Create a mock AnomalibDetector that returns configurable scores."""
    detector = MagicMock(spec=AnomalibDetector)
    detector.threshold = 0.7
    detector.is_loaded = True
    # Default: return low score (normal)
    detector.predict.return_value = AnomalyResult(
        anomaly_score=0.1,
        anomaly_map=np.zeros((256, 256), dtype=np.float32),
        is_anomalous=False,
        threshold=0.7,
    )
    return detector


class TestTileGeneration:
    def test_generates_tiles_for_1080p(self, mock_base_detector):
        """Should generate correct number of tiles for 1920x1080 frame."""
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)
        tiles = ms._generate_tiles(1080, 1920)

        # 512 tiles with stride 384 on 1920 width: positions 0, 384, 768, 1152, 1408 → ~5 cols
        # 512 tiles with stride 384 on 1080 height: positions 0, 384, 568 → ~3 rows
        assert len(tiles) >= 6  # At least 2x3
        assert len(tiles) <= 20  # Not too many

    def test_tiles_cover_entire_frame(self, mock_base_detector):
        """Every pixel should be covered by at least one tile."""
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)
        frame_h, frame_w = 1080, 1920
        tiles = ms._generate_tiles(frame_h, frame_w)

        # Create coverage map
        coverage = np.zeros((frame_h, frame_w), dtype=np.int32)
        for t in tiles:
            coverage[t.y : t.y + t.h, t.x : t.x + t.w] += 1

        # Every pixel must be covered at least once
        assert coverage.min() >= 1, f"Uncovered pixels found, min coverage: {coverage.min()}"

    def test_tiles_stay_in_bounds(self, mock_base_detector):
        """No tile should extend beyond frame boundaries."""
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)
        frame_h, frame_w = 1080, 1920
        tiles = ms._generate_tiles(frame_h, frame_w)

        for t in tiles:
            assert t.x >= 0
            assert t.y >= 0
            assert t.x + t.w <= frame_w, f"Tile exceeds width: x={t.x}, w={t.w}"
            assert t.y + t.h <= frame_h, f"Tile exceeds height: y={t.y}, h={t.h}"

    def test_small_frame_no_tiles(self, mock_base_detector):
        """Frame smaller than tile_size should fall back to single prediction."""
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        result = ms.predict(frame)

        # Should call base detector once (no tiling)
        mock_base_detector.predict.assert_called_once()

    def test_odd_resolution(self, mock_base_detector):
        """Should handle non-standard resolutions without error."""
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)
        tiles = ms._generate_tiles(720, 1280)
        assert len(tiles) > 0

        # Verify bounds
        for t in tiles:
            assert t.x + t.w <= 1280
            assert t.y + t.h <= 720


class TestMultiScaleScoring:
    def test_max_score_from_tiles(self, mock_base_detector):
        """Final score should be the maximum across all tiles and global."""
        call_count = 0

        def mock_predict(frame):
            nonlocal call_count
            call_count += 1
            h, w = frame.shape[:2]
            # One specific tile size gets high score (simulating a small anomaly)
            if h < 600 and w < 600:
                # This is a tile, not full frame
                if call_count == 3:  # Third tile has anomaly
                    return AnomalyResult(
                        anomaly_score=0.92,
                        anomaly_map=np.full((256, 256), 0.92, dtype=np.float32),
                        is_anomalous=True,
                        threshold=0.7,
                    )
            # Full frame and other tiles return low score
            return AnomalyResult(
                anomaly_score=0.15,
                anomaly_map=np.zeros((256, 256), dtype=np.float32),
                is_anomalous=False,
                threshold=0.7,
            )

        mock_base_detector.predict.side_effect = mock_predict
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = ms.predict(frame)

        # Final score should be the maximum (0.92 from the anomalous tile)
        assert result.anomaly_score == 0.92
        assert result.is_anomalous is True

    def test_global_score_wins_for_large_anomaly(self, mock_base_detector):
        """If full-frame score is highest, it should be used."""
        def mock_predict(frame):
            h, w = frame.shape[:2]
            if h > 600:  # Full frame
                return AnomalyResult(
                    anomaly_score=0.95,
                    anomaly_map=np.full((256, 256), 0.9, dtype=np.float32),
                    is_anomalous=True,
                    threshold=0.7,
                )
            return AnomalyResult(
                anomaly_score=0.3,
                anomaly_map=np.zeros((256, 256), dtype=np.float32),
                is_anomalous=False,
                threshold=0.7,
            )

        mock_base_detector.predict.side_effect = mock_predict
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = ms.predict(frame)

        assert result.anomaly_score == 0.95

    def test_all_normal_stays_normal(self, mock_base_detector):
        """If all tiles and global are normal, result should be normal."""
        mock_base_detector.predict.return_value = AnomalyResult(
            anomaly_score=0.1,
            anomaly_map=np.zeros((256, 256), dtype=np.float32),
            is_anomalous=False,
            threshold=0.7,
        )
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = ms.predict(frame)

        assert result.anomaly_score == 0.1
        assert result.is_anomalous is False


class TestHeatmapMerge:
    def test_merged_heatmap_full_frame_size(self, mock_base_detector):
        """Merged heatmap should match original frame dimensions."""
        mock_base_detector.predict.return_value = AnomalyResult(
            anomaly_score=0.5,
            anomaly_map=np.full((256, 256), 0.3, dtype=np.float32),
            is_anomalous=False,
            threshold=0.7,
        )
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = ms.predict(frame)

        assert result.anomaly_map is not None
        assert result.anomaly_map.shape == (1080, 1920)

    def test_merged_heatmap_none_when_no_maps(self, mock_base_detector):
        """If no tile produces a heatmap, merged should be None."""
        mock_base_detector.predict.return_value = AnomalyResult(
            anomaly_score=0.5,
            anomaly_map=None,
            is_anomalous=False,
            threshold=0.7,
        )
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = ms.predict(frame)

        assert result.anomaly_map is None

    def test_overlap_takes_maximum(self, mock_base_detector):
        """Overlapping regions should use the maximum value."""
        call_idx = [0]

        def mock_predict(frame):
            call_idx[0] += 1
            h, w = frame.shape[:2]
            if h <= 512 and call_idx[0] == 1:
                # First tile: high values
                return AnomalyResult(
                    anomaly_score=0.8,
                    anomaly_map=np.full((256, 256), 0.9, dtype=np.float32),
                    is_anomalous=True,
                    threshold=0.7,
                )
            return AnomalyResult(
                anomaly_score=0.2,
                anomaly_map=np.full((256, 256), 0.1, dtype=np.float32),
                is_anomalous=False,
                threshold=0.7,
            )

        mock_base_detector.predict.side_effect = mock_predict
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = ms.predict(frame)

        # The first tile region should have high values (~0.9)
        assert result.anomaly_map is not None
        top_left = result.anomaly_map[0:256, 0:256]
        assert top_left.max() > 0.5


class TestMultiScaleInterface:
    def test_same_interface_as_base(self, mock_base_detector):
        """MultiScaleDetector should expose same properties as AnomalibDetector."""
        ms = MultiScaleDetector(mock_base_detector, tile_size=512, tile_overlap=0.25)
        assert hasattr(ms, "predict")
        assert hasattr(ms, "is_loaded")
        assert hasattr(ms, "load")
        assert hasattr(ms, "hot_reload")
        assert hasattr(ms, "threshold")

    def test_delegates_load(self, mock_base_detector):
        """load() should delegate to base detector."""
        ms = MultiScaleDetector(mock_base_detector)
        ms.load()
        mock_base_detector.load.assert_called_once()

    def test_delegates_hot_reload(self, mock_base_detector):
        """hot_reload() should delegate to base detector."""
        from pathlib import Path

        ms = MultiScaleDetector(mock_base_detector)
        ms.hot_reload(Path("/tmp/model.bin"))
        mock_base_detector.hot_reload.assert_called_once()

    def test_disabled_by_default_in_config(self):
        """AnomalyConfig should have multiscale disabled by default."""
        from argus.config.schema import AnomalyConfig

        config = AnomalyConfig()
        assert config.enable_multiscale is False
        assert config.tile_size == 512
        assert config.tile_overlap == 0.25
