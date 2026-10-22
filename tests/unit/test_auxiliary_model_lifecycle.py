"""Lifecycle-only checks: every SDK boundary is fake, no model/recognition I/O."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.anomaly.auxiliary_runtime import (
    AuxiliaryBusyError,
    AuxiliaryUnavailableError,
)
from argus.anomaly.classifier import OpenVocabClassifier
from argus.anomaly.segmenter import InstanceSegmenter, SegmentationResult


@pytest.fixture(params=["classifier", "segmenter"])
def optional_model(request, monkeypatch):
    if request.param == "classifier":
        model = OpenVocabClassifier(timeout_seconds=0.05)
        factory_name = "_create_model"
    else:
        model = InstanceSegmenter(timeout_seconds=0.05)
        factory_name = "_create_predictor"
    sdk = MagicMock(name="fake_sdk")
    monkeypatch.setattr(model, factory_name, lambda *args: (sdk, "fake-device"))
    yield model, factory_name, sdk
    model.shutdown()


def _invoke(model, frame=None):
    frame = np.zeros((2, 2, 3), dtype=np.uint8) if frame is None else frame
    if isinstance(model, OpenVocabClassifier):
        return model.classify(frame)
    return model.segment(frame, [(0, 0)])


def test_ready_requires_successful_target_load(optional_model):
    model, _, _ = optional_model
    assert model.get_status()["state"] == "unloaded"
    assert model.load() is True
    status = model.get_status()
    assert status["loaded"] is True
    assert status["ready"] is True
    assert status["degraded"] is False
    assert status["device"] == "fake-device"


@pytest.mark.parametrize("kind", ["classifier", "segmenter"])
def test_fake_sdk_initialization_failure_never_attaches_partial_model(kind, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    sdk = MagicMock()
    if kind == "classifier":
        model = OpenVocabClassifier()
        sdk.set_classes.side_effect = RuntimeError("fake SDK setup failure")
        monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOWorld=lambda _: sdk))
    else:
        model = InstanceSegmenter()
        factory = MagicMock(side_effect=RuntimeError("fake SDK setup failure"))
        monkeypatch.setitem(sys.modules, "sam2", SimpleNamespace())
        monkeypatch.setitem(
            sys.modules, "sam2.sam2_image_predictor",
            SimpleNamespace(SAM2ImagePredictor=SimpleNamespace(from_pretrained=factory)),
        )
    try:
        assert model.load() is False
        assert model.get_status()["loaded"] is False
        assert model.get_status()["ready"] is False
        if kind == "classifier":
            assert model._model is None
            sdk.set_classes.side_effect = None
        else:
            assert model._predictor is None
            factory.side_effect = None
            factory.return_value = sdk
        assert model.load() is True
        assert model.get_status()["ready"] is True
    finally:
        model.shutdown()


def test_failed_load_is_retryable_but_not_retried_on_every_frame(optional_model, monkeypatch):
    model, factory_name, sdk = optional_model
    factory = MagicMock(side_effect=RuntimeError("fake load failed"))
    monkeypatch.setattr(model, factory_name, factory)
    assert model.load() is False
    assert model.get_status()["ready"] is False
    assert model.get_status()["loaded"] is False
    assert "fake load failed" in model.get_status()["reason"]
    with pytest.raises(AuxiliaryUnavailableError, match="fake load failed"):
        _invoke(model)
    assert factory.call_count == 1
    factory.side_effect = None
    factory.return_value = (sdk, "fake-device")
    assert model.load() is True
    assert model.get_status()["ready"] is True
    assert factory.call_count == 2


def test_load_timeout_has_no_queue_or_late_ready(optional_model, monkeypatch):
    model, factory_name, sdk = optional_model
    started, release = threading.Event(), threading.Event()

    def blocked_factory(*args):
        started.set()
        assert release.wait(2), "test must release the fake SDK"
        return sdk, "fake-device"

    factory = MagicMock(side_effect=blocked_factory)
    monkeypatch.setattr(model, factory_name, factory)
    try:
        assert model.load() is False
        assert started.is_set()
        future = model._runtime._future
        assert model.get_status()["busy"] is True
        assert model.load() is False
        assert factory.call_count == 1
        assert model.get_status()["ready"] is False
        release.set()
        future.result(timeout=1)
        assert model.get_status()["busy"] is False
        assert model.get_status()["loaded"] is False
        assert model.get_status()["ready"] is False
        monkeypatch.setattr(model, factory_name, lambda *args: (sdk, "fake-device"))
        assert model.load() is True
    finally:
        release.set()


def test_shutdown_rejects_late_load_commit(optional_model, monkeypatch):
    model, factory_name, sdk = optional_model
    started, release = threading.Event(), threading.Event()
    outcome = []

    def blocked_factory(*args):
        started.set()
        assert release.wait(2)
        return sdk, "fake-device"

    monkeypatch.setattr(model, factory_name, blocked_factory)
    model._timeout_seconds = 1
    caller = threading.Thread(target=lambda: outcome.append(model.load()))
    caller.start()
    try:
        assert started.wait(1)
        model.shutdown()
        release.set()
        caller.join(timeout=1)
        assert not caller.is_alive()
        assert outcome == [False]
        assert model.get_status()["state"] == "stopped"
        assert model.get_status()["loaded"] is False
        assert model.load() is False
        with pytest.raises(AuxiliaryUnavailableError, match="stopped"):
            _invoke(model)
    finally:
        release.set()
        caller.join(timeout=1)


def test_concurrent_load_does_not_construct_second_sdk(optional_model, monkeypatch):
    model, factory_name, sdk = optional_model
    started, release = threading.Event(), threading.Event()
    outcome = []

    def blocked_factory(*args):
        started.set()
        assert release.wait(2)
        return sdk, "fake-device"

    factory = MagicMock(side_effect=blocked_factory)
    monkeypatch.setattr(model, factory_name, factory)
    model._timeout_seconds = 1
    caller = threading.Thread(target=lambda: outcome.append(model.load()))
    caller.start()
    try:
        assert started.wait(1)
        assert model.get_status()["state"] == "loading"
        assert model.get_status()["ready"] is False
        assert model.load() is False
        assert factory.call_count == 1
        release.set()
        caller.join(timeout=1)
        assert outcome == [True]
    finally:
        release.set()
        caller.join(timeout=1)


@pytest.mark.parametrize("kind", ["classifier", "segmenter"])
def test_process_lease_survives_shutdown_until_old_sdk_finishes(kind, monkeypatch):
    started, release, lease_released = threading.Event(), threading.Event(), threading.Event()
    key = f"test-lifecycle-{kind}-{id(started)}"
    sdk = MagicMock()
    if kind == "classifier":
        old = OpenVocabClassifier(timeout_seconds=0.05, runtime_key=key)
        new = OpenVocabClassifier(timeout_seconds=1, runtime_key=key)
        factory_name = "_create_model"
    else:
        old = InstanceSegmenter(timeout_seconds=0.05, runtime_key=key)
        new = InstanceSegmenter(timeout_seconds=1, runtime_key=key)
        factory_name = "_create_predictor"

    def blocked_factory(*args):
        started.set()
        assert release.wait(2)
        return sdk, "fake-device"

    monkeypatch.setattr(old, factory_name, blocked_factory)
    new_factory = MagicMock(return_value=(sdk, "fake-device"))
    monkeypatch.setattr(new, factory_name, new_factory)
    try:
        assert old.load() is False
        assert started.is_set()
        old_future = old._runtime._future
        old_future.add_done_callback(lambda _: lease_released.set())
        old.shutdown()
        assert new.get_status()["busy"] is True
        for _ in range(3):
            assert new.load() is False
        assert new._runtime._executor is None
        new_factory.assert_not_called()
        release.set()
        assert lease_released.wait(1)
        assert new.load() is True
        assert old.get_status()["loaded"] is False
        assert old.get_status()["state"] == "stopped"
        assert new.get_status()["ready"] is True
    finally:
        release.set()
        old.shutdown()
        new.shutdown()


@pytest.mark.parametrize("kind", ["classifier", "segmenter"])
@pytest.mark.parametrize("succeeds", [True, False])
def test_done_sdk_future_still_keeps_owner_commit_window_exclusive(kind, succeeds, monkeypatch):
    if kind == "classifier":
        model = OpenVocabClassifier()
        sdk = MagicMock(return_value=[])
        monkeypatch.setattr(model, "_create_model", lambda _: (sdk, "fake-device"))
    else:
        model = InstanceSegmenter()
        monkeypatch.setattr(model, "_create_predictor", lambda: (object(), "fake-device"))
        monkeypatch.setattr(model, "_segment_sam2", lambda *args, **kwargs: SegmentationResult())
    assert model.load() is True
    sdk_done, release_owner = threading.Event(), threading.Event()
    original_wait = model._runtime.wait
    outcome = []

    def pause_after_sdk_done(ticket, timeout):
        result = original_wait(ticket, timeout)
        sdk_done.set()
        assert release_owner.wait(2)
        if not succeeds:
            raise RuntimeError("fake post-SDK failure")
        return result

    monkeypatch.setattr(model._runtime, "wait", pause_after_sdk_done)

    def invoke():
        try:
            outcome.append(_invoke(model))
        except Exception as exc:
            outcome.append(exc)

    caller = threading.Thread(target=invoke)
    caller.start()
    try:
        assert sdk_done.wait(1)
        assert model.get_status()["busy"] is True
        assert model.get_status()["ready"] is False
        with pytest.raises(AuxiliaryBusyError, match="unfinished SDK call"):
            _invoke(model)
        assert model.load() is False
        release_owner.set()
        caller.join(timeout=1)
        assert len(outcome) == 1
        assert model.get_status()["ready"] is succeeds
        if not succeeds:
            assert isinstance(outcome[0], RuntimeError)
    finally:
        release_owner.set()
        caller.join(timeout=1)
        model.shutdown()


def test_done_load_future_does_not_admit_prediction_before_commit(optional_model, monkeypatch):
    model, _, _ = optional_model
    sdk_done, release_owner = threading.Event(), threading.Event()
    original_wait = model._runtime.wait
    outcome = []

    def pause_after_sdk_done(ticket, timeout):
        result = original_wait(ticket, timeout)
        sdk_done.set()
        assert release_owner.wait(2)
        return result

    monkeypatch.setattr(model._runtime, "wait", pause_after_sdk_done)
    caller = threading.Thread(target=lambda: outcome.append(model.load()))
    caller.start()
    try:
        assert sdk_done.wait(1)
        assert model.get_status()["state"] == "loading"
        assert model.get_status()["ready"] is False
        with pytest.raises(AuxiliaryBusyError, match="unfinished SDK call"):
            _invoke(model)
        assert model.load() is False
        release_owner.set()
        caller.join(timeout=1)
        assert outcome == [True]
    finally:
        release_owner.set()
        caller.join(timeout=1)


def test_load_vocabulary_missing_or_invalid_file_keeps_current_labels(tmp_path):
    model = OpenVocabClassifier(vocabulary=["wrench"])
    try:
        with pytest.raises(FileNotFoundError, match="Configured classifier vocabulary"):
            model.load_vocabulary(tmp_path / "missing.json")
        invalid = tmp_path / "invalid.json"
        invalid.write_text('{"wrong": "shape"}', encoding="utf-8")
        with pytest.raises(ValueError, match="nonempty list"):
            model.load_vocabulary(invalid)
        assert model.vocabulary == ["wrench"]
    finally:
        model.shutdown()


def test_classifier_failure_differs_from_normal_no_detection(monkeypatch):
    classifier = OpenVocabClassifier()
    sdk = MagicMock(return_value=[])
    monkeypatch.setattr(classifier, "_create_model", lambda _: (sdk, "fake-device"))
    try:
        assert classifier.load() is True
        assert _invoke(classifier) is None
        sdk.side_effect = RuntimeError("fake prediction failed")
        with pytest.raises(RuntimeError, match="fake prediction failed"):
            _invoke(classifier)
        assert classifier.get_status()["degraded"] is True
        assert classifier.get_status()["ready"] is False
        sdk.side_effect = None
        assert _invoke(classifier) is None
        assert classifier.get_status()["ready"] is True
    finally:
        classifier.shutdown()


@pytest.mark.parametrize("kind", ["classifier", "segmenter"])
def test_prediction_timeout_is_bounded_and_owns_its_input(kind, monkeypatch):
    started, release = threading.Event(), threading.Event()
    observed = []
    if kind == "classifier":
        model = OpenVocabClassifier(timeout_seconds=0.05)
        sdk = MagicMock()
        monkeypatch.setattr(model, "_create_model", lambda _: (sdk, "fake-device"))

        def blocked(frame, **kwargs):
            started.set()
            assert release.wait(2)
            observed.append(int(frame[0, 0, 0]))
            return []

        sdk.side_effect = blocked
        call = sdk
    else:
        model = InstanceSegmenter(timeout_seconds=0.05)
        monkeypatch.setattr(model, "_create_predictor", lambda: (object(), "fake-device"))

        def blocked(frame, points, **kwargs):
            started.set()
            assert release.wait(2)
            observed.append(int(frame[0, 0, 0]))
            return SegmentationResult()

        call = MagicMock(side_effect=blocked)
        monkeypatch.setattr(model, "_segment_sam2", call)
    try:
        assert model.load() is True
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with pytest.raises(TimeoutError, match="timed out"):
            _invoke(model, frame)
        assert started.is_set()
        future = model._runtime._future
        frame[:] = 99
        for _ in range(3):
            with pytest.raises(AuxiliaryBusyError, match="unfinished SDK call"):
                _invoke(model)
        assert call.call_count == 1
        assert model.get_status()["ready"] is False
        release.set()
        future.result(timeout=1)
        assert observed == [0]
        # Completion of the timed-out job cannot clear its failure state.
        assert model.get_status()["ready"] is False
        _invoke(model)
        assert model.get_status()["ready"] is True
    finally:
        release.set()
        model.shutdown()


def test_segmenter_failure_is_not_an_empty_success(monkeypatch):
    model = InstanceSegmenter()
    monkeypatch.setattr(model, "_create_predictor", lambda: (object(), "fake-device"))
    fake_segment = MagicMock(side_effect=RuntimeError("fake segmentation failed"))
    monkeypatch.setattr(model, "_segment_sam2", fake_segment)
    try:
        assert model.load() is True
        with pytest.raises(RuntimeError, match="fake segmentation failed"):
            _invoke(model)
        assert model.get_status()["degraded"] is True
        fake_segment.side_effect = None
        fake_segment.return_value = SegmentationResult()
        assert _invoke(model).num_objects == 0
        assert model.get_status()["ready"] is True
    finally:
        model.shutdown()


@pytest.mark.parametrize("kind", ["classifier", "segmenter"])
def test_shutdown_rejects_prediction_result_already_running(kind, monkeypatch):
    started, release = threading.Event(), threading.Event()
    outcome = []
    if kind == "classifier":
        model = OpenVocabClassifier(timeout_seconds=1)

        def blocked(frame, **kwargs):
            started.set()
            assert release.wait(2)
            return []

        sdk = MagicMock(side_effect=blocked)
        monkeypatch.setattr(model, "_create_model", lambda _: (sdk, "fake-device"))
    else:
        model = InstanceSegmenter(timeout_seconds=1)
        monkeypatch.setattr(model, "_create_predictor", lambda: (object(), "fake-device"))

        def blocked(frame, points, **kwargs):
            started.set()
            assert release.wait(2)
            return SegmentationResult()

        monkeypatch.setattr(model, "_segment_sam2", blocked)
    assert model.load() is True

    def invoke():
        try:
            outcome.append(_invoke(model))
        except Exception as exc:
            outcome.append(exc)

    caller = threading.Thread(target=invoke)
    caller.start()
    try:
        assert started.wait(1)
        model.shutdown()
        release.set()
        caller.join(timeout=1)
        assert not caller.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], AuxiliaryUnavailableError)
        assert model.get_status()["state"] == "stopped"
        assert model.get_status()["ready"] is False
    finally:
        release.set()
        caller.join(timeout=1)
        model.shutdown()


def test_contour_fallback_requires_explicit_opt_in_and_is_visibly_degraded(monkeypatch):
    for allow_fallback in (False, True):
        model = InstanceSegmenter(allow_contour_fallback=allow_fallback)
        monkeypatch.setattr(
            model, "_create_predictor", MagicMock(side_effect=ImportError("fake missing SDK")),
        )
        fallback = MagicMock(return_value=SegmentationResult())
        monkeypatch.setattr(model, "_segment_contour_fallback", fallback)
        try:
            assert model.load() is False
            status = model.get_status()
            assert status["loaded"] is False
            assert status["ready"] is False
            assert status["degraded"] is True
            if allow_fallback:
                assert status["backend"] == "contour"
                result = _invoke(model)
                assert result.backend == "contour"
                assert result.degraded is True
            else:
                assert status["backend"] is None
                with pytest.raises(AuxiliaryUnavailableError, match="fake missing SDK"):
                    _invoke(model)
                fallback.assert_not_called()
        finally:
            model.shutdown()


def test_vocabulary_update_failure_does_not_commit_labels(monkeypatch):
    model = OpenVocabClassifier(vocabulary=["wrench"])
    sdk = MagicMock()
    monkeypatch.setattr(model, "_create_model", lambda _: (sdk, "fake-device"))
    try:
        assert model.load() is True
        sdk.set_classes.side_effect = RuntimeError("fake vocabulary failure")
        with pytest.raises(RuntimeError, match="fake vocabulary failure"):
            model.update_vocabulary(["shadow"])
        assert model.vocabulary == ["wrench"]
        assert model.get_status()["loaded"] is False
        assert model.get_status()["ready"] is False
    finally:
        model.shutdown()


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1, True, "0.99"])
def test_classifier_rejects_invalid_confidence(confidence, monkeypatch):
    model = OpenVocabClassifier()
    boxes = [SimpleNamespace(cls=0, conf=confidence)]
    sdk = MagicMock(return_value=[SimpleNamespace(boxes=boxes, names={0: "shadow"})])
    monkeypatch.setattr(model, "_create_model", lambda _: (sdk, "fake-device"))
    try:
        assert model.load() is True
        with pytest.raises(ValueError, match="invalid confidence"):
            _invoke(model)
    finally:
        model.shutdown()


def test_failed_sam2_prediction_always_resets_sdk_image(monkeypatch):
    model = InstanceSegmenter()
    sdk = MagicMock()
    sdk.predict.side_effect = RuntimeError("fake prediction failed")
    monkeypatch.setattr(model, "_create_predictor", lambda: (sdk, "fake-device"))
    try:
        assert model.load() is True
        with pytest.raises(RuntimeError, match="fake prediction failed"):
            _invoke(model)
        sdk.reset_image.assert_called_once()
    finally:
        model.shutdown()
