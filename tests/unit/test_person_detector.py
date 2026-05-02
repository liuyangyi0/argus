"""Unit tests for ``argus.person.detector.YOLOObjectDetector``.

These tests never load a real YOLO model — they mock ``get_shared_yolo`` and
``model.predict`` / ``model.track`` so the detector class can be exercised
end-to-end (init → ensure → detect → batch → graceful degradation) on CI
boxes without GPUs or weight downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.person.detector import (
    COCO_CLASS_NAMES,
    ObjectDetection,
    ObjectDetectionResult,
    PersonDetection,
    PersonFilterResult,
    YOLOObjectDetector,
    YOLOPersonDetector,
    get_shared_yolo,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_box(x1: int, y1: int, x2: int, y2: int, conf: float, cls_id: int,
              track_id: int | None = None) -> MagicMock:
    """Build an object that mimics ultralytics' ``Boxes`` row API.

    Each row exposes ``.xyxy[0].cpu().numpy().astype(int)`` -> ndarray[4],
    ``.conf[0]`` -> float, ``.cls[0]`` -> int, ``.id[0]`` -> int (or .id is None).
    """
    box = MagicMock()
    arr = np.array([x1, y1, x2, y2], dtype=np.int64)
    box.xyxy[0].cpu.return_value.numpy.return_value.astype.return_value = arr
    box.conf = [conf]
    box.cls = [cls_id]
    if track_id is None:
        box.id = None
    else:
        box.id = [track_id]
    return box


def _make_results(boxes: list[MagicMock]) -> list[MagicMock]:
    """Wrap a list of mock boxes into the ultralytics single-result list shape."""
    res = MagicMock()
    res.boxes = boxes if boxes else None
    return [res]


def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Solid-colour frame so blur calls don't crash but content is unimportant."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ── Backward-compat aliases ─────────────────────────────────────────────────


class TestBackwardCompatAliases:
    def test_person_detection_is_object_detection(self):
        assert PersonDetection is ObjectDetection

    def test_person_filter_result_is_object_detection_result(self):
        assert PersonFilterResult is ObjectDetectionResult

    def test_yolo_person_detector_is_yolo_object_detector(self):
        assert YOLOPersonDetector is YOLOObjectDetector


class TestCocoClassNames:
    def test_person_is_class_zero(self):
        assert COCO_CLASS_NAMES[0] == "person"

    def test_unknown_class_not_in_table(self):
        assert 999 not in COCO_CLASS_NAMES


class TestObjectDetectionDefaults:
    def test_class_defaults_to_person(self):
        det = ObjectDetection(x1=0, y1=0, x2=10, y2=10, confidence=0.9)
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.track_id is None


class TestObjectDetectionResultDefaults:
    def test_empty_defaults(self):
        result = ObjectDetectionResult()
        assert result.persons == []
        assert result.objects == []
        assert result.non_person_objects == []
        assert result.has_persons is False
        assert result.has_objects is False
        assert result.masked_frame is None
        assert result.filter_available is True


# ── get_shared_yolo registry ─────────────────────────────────────────────────


class TestSharedYoloRegistry:
    def test_returns_same_instance_on_repeat_call(self):
        """Two calls with the same name must hit the cached instance."""
        from argus.person import detector as det_mod

        sentinel = MagicMock(name="yolo_instance")

        with patch.dict(det_mod._shared_yolo_registry, {}, clear=True):
            with patch("ultralytics.YOLO", return_value=sentinel) as ultra:
                first = get_shared_yolo("dummy.pt")
                second = get_shared_yolo("dummy.pt")
            assert first is second is sentinel
            assert ultra.call_count == 1


# ── Constructor wiring ──────────────────────────────────────────────────────


class TestConstructor:
    def test_defaults(self):
        d = YOLOObjectDetector()
        assert d.confidence == 0.4
        assert d.skip_frame_on_person is False
        assert d.mask_padding == 20
        assert d.classes_to_detect == [0]  # person only
        assert d.enable_tracking is False
        assert d._available is None  # not yet attempted
        assert d.status.name == "yolo"

    def test_custom_class_list_passed_through(self):
        d = YOLOObjectDetector(classes_to_detect=[0, 14, 16])
        assert d.classes_to_detect == [0, 14, 16]
        # ModelStatus extra reflects construction-time options
        assert d.status.extra["classes"] == [0, 14, 16]
        assert d.status.extra["tracking"] is False
        assert d.status.extra["sahi"] is False

    def test_tracking_and_sahi_flags(self):
        d = YOLOObjectDetector(enable_tracking=True, sahi_enabled=True)
        assert d.enable_tracking is True
        assert d._sahi_enabled is True
        assert d.status.extra["tracking"] is True
        assert d.status.extra["sahi"] is True


# ── Graceful degradation: model unavailable ─────────────────────────────────


class TestEnsureModelGracefulFailure:
    def test_load_failure_marks_unavailable_and_records_error(self):
        d = YOLOObjectDetector()

        with patch(
            "argus.person.detector.get_shared_yolo",
            side_effect=ImportError("no ultralytics"),
        ):
            d._ensure_model()

        assert d._available is False
        assert d.status.loaded is False
        assert d.status.last_error and "ultralytics" in d.status.last_error

    def test_ensure_model_is_idempotent(self):
        d = YOLOObjectDetector()
        with patch(
            "argus.person.detector.get_shared_yolo",
            side_effect=ImportError("no ultralytics"),
        ) as gsy:
            d._ensure_model()
            d._ensure_model()
            d._ensure_model()
        assert gsy.call_count == 1, "_ensure_model must short-circuit after first attempt"

    def test_detect_returns_empty_when_model_unavailable(self):
        d = YOLOObjectDetector()
        with patch(
            "argus.person.detector.get_shared_yolo",
            side_effect=ImportError("no ultralytics"),
        ):
            result = d.detect(_make_frame())

        assert result.filter_available is False
        assert result.has_persons is False
        assert result.persons == []

    def test_detect_batch_unavailable_returns_empty_list_per_frame(self):
        d = YOLOObjectDetector()
        with patch(
            "argus.person.detector.get_shared_yolo",
            side_effect=ImportError("no ultralytics"),
        ):
            results = d.detect_batch([_make_frame(), _make_frame()])
        assert len(results) == 2
        assert all(r.filter_available is False for r in results)


# ── Happy path with mocked model ────────────────────────────────────────────


class TestDetectHappyPath:
    def test_single_person_detected_and_masked(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_results([
            _make_box(50, 50, 150, 200, conf=0.9, cls_id=0),
        ])
        d = YOLOObjectDetector(skip_frame_on_person=False)

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            result = d.detect(_make_frame())

        assert result.has_persons is True
        assert result.has_objects is True
        assert len(result.persons) == 1
        assert result.persons[0].class_name == "person"
        assert result.persons[0].confidence == pytest.approx(0.9)
        assert result.masked_frame is not None  # blur applied
        assert d.status.total_inferences == 1
        assert d.status.consecutive_failures == 0

    def test_multi_class_split_into_persons_and_others(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_results([
            _make_box(10, 10, 50, 50, 0.8, 0),    # person
            _make_box(60, 60, 100, 100, 0.7, 16),  # dog
            _make_box(120, 120, 160, 160, 0.6, 39),  # bottle
        ])
        d = YOLOObjectDetector(classes_to_detect=[0, 16, 39])

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            result = d.detect(_make_frame())

        assert result.has_persons is True
        assert len(result.persons) == 1
        assert len(result.non_person_objects) == 2
        names = sorted(o.class_name for o in result.non_person_objects)
        assert names == ["bottle", "dog"]

    def test_skip_frame_on_person_suppresses_masked_frame(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_results([
            _make_box(50, 50, 100, 100, 0.9, 0),
        ])
        d = YOLOObjectDetector(skip_frame_on_person=True)

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            result = d.detect(_make_frame())

        assert result.has_persons is True
        assert result.masked_frame is None

    def test_no_detections_yields_empty_lists(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_results([])
        d = YOLOObjectDetector()

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            result = d.detect(_make_frame())

        assert result.persons == []
        assert result.objects == []
        assert result.has_persons is False
        assert result.has_objects is False
        assert result.masked_frame is None

    def test_unknown_class_id_uses_synthetic_name(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_results([
            _make_box(0, 0, 10, 10, 0.5, 999),
        ])
        d = YOLOObjectDetector(classes_to_detect=[999])

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            result = d.detect(_make_frame())

        assert result.objects[0].class_name == "class_999"


class TestDetectTrackingPath:
    def test_tracking_enabled_uses_track_method(self):
        mock_model = MagicMock()
        mock_model.track.return_value = _make_results([
            _make_box(0, 0, 50, 50, 0.95, 0, track_id=42),
        ])
        d = YOLOObjectDetector(enable_tracking=True)

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            result = d.detect(_make_frame())

        mock_model.track.assert_called_once()
        mock_model.predict.assert_not_called()
        assert result.persons[0].track_id == 42

    def test_predict_failure_records_inference_failure_and_returns_empty(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA OOM")
        d = YOLOObjectDetector()

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            result = d.detect(_make_frame())

        assert result.filter_available is False
        assert d.status.consecutive_failures == 1
        assert d.status.total_failures == 1
        assert "CUDA OOM" in (d.status.last_error or "")


# ── Batch detection ─────────────────────────────────────────────────────────


class TestDetectBatch:
    def test_empty_input_returns_empty_list(self):
        d = YOLOObjectDetector()
        assert d.detect_batch([]) == []

    def test_tracking_falls_back_to_sequential(self):
        """Tracking is stateful per frame so batch must dispatch to detect()."""
        mock_model = MagicMock()
        mock_model.track.return_value = _make_results([])
        d = YOLOObjectDetector(enable_tracking=True)

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            results = d.detect_batch([_make_frame(), _make_frame(), _make_frame()])

        assert len(results) == 3
        assert mock_model.track.call_count == 3
        mock_model.predict.assert_not_called()

    def test_sahi_falls_back_to_sequential(self):
        d = YOLOObjectDetector(sahi_enabled=True)
        with patch("argus.person.detector.get_shared_yolo", return_value=MagicMock()):
            with patch.object(d, "detect") as detect_mock:
                detect_mock.return_value = ObjectDetectionResult()
                d.detect_batch([_make_frame(), _make_frame()])
                assert detect_mock.call_count == 2

    def test_happy_batch_path_per_frame_results(self):
        """Each frame in the batch gets its own ObjectDetectionResult."""
        mock_model = MagicMock()
        # Two results, one per frame
        res_a = MagicMock(boxes=[_make_box(0, 0, 20, 20, 0.9, 0)])
        res_b = MagicMock(boxes=[_make_box(10, 10, 30, 30, 0.85, 16)])
        mock_model.predict.return_value = [res_a, res_b]
        d = YOLOObjectDetector(classes_to_detect=[0, 16])

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            results = d.detect_batch([_make_frame(), _make_frame()])

        assert len(results) == 2
        assert results[0].has_persons is True and results[0].persons[0].class_id == 0
        assert results[1].has_persons is False
        assert results[1].non_person_objects[0].class_name == "dog"

    def test_batch_predict_failure_falls_back_to_sequential(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = [RuntimeError("batch fail"),
                                          _make_results([]),
                                          _make_results([])]
        d = YOLOObjectDetector()

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            results = d.detect_batch([_make_frame(), _make_frame()])

        # Two extra .predict calls happen during the per-frame fallback.
        assert mock_model.predict.call_count == 3
        assert len(results) == 2


class TestMaskPadding:
    def test_mask_padding_clamped_to_frame_bounds(self):
        """Padding must not run off the frame; verifies safe clamping path."""
        mock_model = MagicMock()
        # Person box right at the edge — padding would go negative without clamping
        mock_model.predict.return_value = _make_results([
            _make_box(0, 0, 10, 10, 0.9, 0),
        ])
        d = YOLOObjectDetector(mask_padding=200)

        with patch("argus.person.detector.get_shared_yolo", return_value=mock_model):
            result = d.detect(_make_frame(h=64, w=64))  # smaller than padding

        assert result.masked_frame is not None
        assert result.masked_frame.shape == (64, 64, 3)  # frame did not grow
