"""Tests for anomaly map post-processing (ANO-006)."""

from __future__ import annotations

import numpy as np
import pytest

from argus.core.anomaly_postprocess import AnomalyMapProcessor, AnomalyRegion


@pytest.fixture
def processor() -> AnomalyMapProcessor:
    return AnomalyMapProcessor()


class TestAnomalyRegionDataclass:
    """Tests for AnomalyRegion dataclass."""

    def test_fields(self):
        region = AnomalyRegion(
            x=10, y=20, width=50, height=30,
            area=1200, centroid_x=35.0, centroid_y=35.0,
            max_score=0.95, mean_score=0.75,
        )
        assert region.x == 10
        assert region.width == 50
        assert region.max_score == 0.95


class TestProcessZeroMap:
    """All-zero anomaly map should remain all-zero after processing."""

    def test_zero_map_stays_zero(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((100, 100), dtype=np.float32)
        result = processor.process(anomaly_map)
        assert result.shape == (100, 100)
        assert result.max() == 0.0

    def test_zero_map_dtype(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((64, 64), dtype=np.float32)
        result = processor.process(anomaly_map)
        assert result.dtype == np.float32


class TestSmallNoiseBlobsRemoved:
    """Small noise blobs should be removed by morphological operations."""

    def test_tiny_blobs_removed(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        # Add tiny 2x2 noise blobs (area=4, well below min_contour_area=100)
        anomaly_map[10:12, 10:12] = 0.8
        anomaly_map[50:52, 80:82] = 0.9
        anomaly_map[150:152, 150:152] = 0.7
        result = processor.process(anomaly_map)
        # Tiny blobs should be removed or heavily reduced
        assert result.max() < 0.1

    def test_scattered_pixels_removed(self):
        processor = AnomalyMapProcessor(min_contour_area=50)
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        # Scatter individual bright pixels
        rng = np.random.RandomState(42)
        for _ in range(20):
            y, x = rng.randint(0, 200, 2)
            anomaly_map[y, x] = 0.9
        result = processor.process(anomaly_map)
        assert result.max() < 0.1


class TestLargeRegionPreserved:
    """Large connected anomaly regions should be preserved."""

    def test_large_blob_preserved(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        # Large 30x30 region (area=900, well above min_contour_area=100)
        anomaly_map[50:80, 50:80] = 0.8
        result = processor.process(anomaly_map)
        # The center of the large region should still have significant values
        center_region = result[55:75, 55:75]
        assert center_region.max() > 0.3

    def test_large_region_shape_roughly_preserved(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[80:130, 80:130] = 0.9
        result = processor.process(anomaly_map)
        # Region should still be active in roughly the same area
        assert result[100, 100] > 0.3
        # Corners far from the region should be zero
        assert result[0, 0] == pytest.approx(0.0, abs=0.01)
        assert result[199, 199] == pytest.approx(0.0, abs=0.01)


class TestExtractRegionsCount:
    """extract_regions should find the correct number of regions."""

    def test_no_regions_in_zero_map(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((100, 100), dtype=np.float32)
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 0

    def test_single_region(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[50:80, 50:80] = 0.8
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 1

    def test_two_separate_regions(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        # Two well-separated large blobs
        anomaly_map[10:40, 10:40] = 0.9
        anomaly_map[150:180, 150:180] = 0.7
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 2

    def test_small_regions_filtered_out(self):
        processor = AnomalyMapProcessor(min_contour_area=200)
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        # Large region (900 pixels)
        anomaly_map[50:80, 50:80] = 0.8
        # Small region (25 pixels, below 200 threshold)
        anomaly_map[150:155, 150:155] = 0.9
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 1


class TestExtractRegionsBoundingBox:
    """extract_regions should return correct bounding boxes."""

    def test_bounding_box_location(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[40:70, 60:100] = 0.8
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 1
        r = regions[0]
        # Bounding box should approximately match the drawn rectangle
        assert abs(r.x - 60) <= 1
        assert abs(r.y - 40) <= 1
        assert abs(r.width - 40) <= 2
        assert abs(r.height - 30) <= 2

    def test_bounding_box_contains_region(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[20:50, 30:80] = 0.9
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 1
        r = regions[0]
        # Bounding box must fully contain the region
        assert r.x <= 30
        assert r.y <= 20
        assert r.x + r.width >= 80
        assert r.y + r.height >= 50


class TestExtractRegionsCentroids:
    """extract_regions should return approximately correct centroids."""

    def test_centroid_in_center(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[50:80, 50:80] = 0.8
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 1
        r = regions[0]
        # Centroid should be near center of the 50:80, 50:80 region = ~65, 65
        assert abs(r.centroid_x - 65.0) < 3.0
        assert abs(r.centroid_y - 65.0) < 3.0


class TestExtractRegionsScores:
    """extract_regions should return correct max_score and mean_score."""

    def test_max_score(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[50:80, 50:80] = 0.6
        anomaly_map[60:65, 60:65] = 0.95  # hot spot within region
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 1
        assert regions[0].max_score == pytest.approx(0.95, abs=0.01)

    def test_mean_score_less_than_max(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[50:80, 50:80] = 0.6
        anomaly_map[60:65, 60:65] = 0.95
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 1
        assert regions[0].mean_score < regions[0].max_score

    def test_regions_sorted_by_max_score(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[10:40, 10:40] = 0.6
        anomaly_map[150:180, 150:180] = 0.9
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 2
        assert regions[0].max_score >= regions[1].max_score


class TestMinContourAreaFiltering:
    """min_contour_area parameter should control region filtering."""

    def test_default_filters_small(self):
        processor = AnomalyMapProcessor(min_contour_area=500)
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        # Region with area ~100 (10x10), below 500 threshold
        anomaly_map[50:60, 50:60] = 0.8
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 0

    def test_low_threshold_keeps_small(self):
        processor = AnomalyMapProcessor(min_contour_area=10)
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[50:60, 50:60] = 0.8
        regions = processor.extract_regions(anomaly_map, threshold=0.5)
        assert len(regions) == 1


class TestProcessPreservesShape:
    """process() should preserve the shape of the input map."""

    def test_shape_preserved(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((120, 160), dtype=np.float32)
        anomaly_map[30:90, 40:120] = 0.7
        result = processor.process(anomaly_map)
        assert result.shape == (120, 160)

    def test_non_square_map(self, processor: AnomalyMapProcessor):
        anomaly_map = np.zeros((50, 300), dtype=np.float32)
        anomaly_map[10:40, 100:200] = 0.6
        result = processor.process(anomaly_map)
        assert result.shape == (50, 300)


class TestMorphologyEffects:
    """Morphological operations should fill gaps and smooth boundaries."""

    def test_close_fills_gap(self):
        processor = AnomalyMapProcessor(morphology_kernel=7, min_contour_area=50)
        anomaly_map = np.zeros((100, 100), dtype=np.float32)
        # Two blobs separated by a small 2-pixel gap
        anomaly_map[40:50, 30:45] = 0.8
        anomaly_map[40:50, 47:60] = 0.8
        result = processor.process(anomaly_map)
        # After morphological close, the gap should be filled
        # Check that the gap region now has some value
        gap_max = result[44:46, 45:48].max()
        assert gap_max > 0.1
