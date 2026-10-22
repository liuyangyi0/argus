"""Adversarial lifecycle/evidence contracts using fake SDKs and fake executors."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.anomaly import auxiliary_runtime as runtime_module
from argus.anomaly.auxiliary_runtime import (
    AuxiliaryBusyError,
    AuxiliaryTimeoutError,
    AuxiliaryUnavailableError,
    BoundedAuxiliaryRuntime,
)
from argus.anomaly.classifier import OpenVocabClassifier
from argus.anomaly.segmenter import InstanceSegmenter


class _ImmediateExecutor:
    def __init__(self, *, shutdown_failures=0):
        self.shutdown_calls = 0
        self.shutdown_failures = shutdown_failures

    def submit(self, call):
        future = Future()
        future.set_running_or_notify_cancel()
        try:
            future.set_result(call())
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("fake executor cleanup failure")


def test_deadline_starts_at_submission_not_at_wait(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(runtime_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    runtime = BoundedAuxiliaryRuntime("fake")
    runtime._executor = _ImmediateExecutor()
    try:
        ticket = runtime.submit(lambda: "already-completed-but-too-old")
        clock[0] = 3.0
        with pytest.raises(AuxiliaryTimeoutError):
            runtime.wait(ticket, 1.0)
        assert ticket.expired is True
        with pytest.raises(AuxiliaryUnavailableError, match="no longer eligible"):
            runtime.wait(ticket, 30.0)
    finally:
        runtime.shutdown()


def test_timed_out_ticket_cannot_be_accepted_by_second_wait():
    runtime = BoundedAuxiliaryRuntime("fake")
    future = Future()
    future.set_running_or_notify_cancel()
    runtime._executor = SimpleNamespace(
        submit=lambda _: future, shutdown=lambda **kwargs: None,
    )
    try:
        ticket = runtime.submit(lambda: "unused-fake-call")
        with pytest.raises(AuxiliaryTimeoutError):
            runtime.wait(ticket, 0.001)
        future.set_result("late")
        with pytest.raises(AuxiliaryUnavailableError, match="no longer eligible"):
            runtime.wait(ticket, 1.0)
    finally:
        if not future.done():
            future.set_result("cleanup")
        runtime.shutdown()


def test_foreign_runtime_cannot_accept_other_owners_ticket():
    first = BoundedAuxiliaryRuntime("first")
    second = BoundedAuxiliaryRuntime("second")
    first._executor = _ImmediateExecutor()
    try:
        ticket = first.submit(lambda: "first-result")
        with pytest.raises(AuxiliaryUnavailableError, match="another runtime"):
            second.wait(ticket, 1.0)
        assert first.wait(ticket, 1.0) == "first-result"
    finally:
        first.shutdown()
        second.shutdown()


def test_unknown_submit_failure_retains_cross_instance_lease_until_call_ends():
    started, release, completed = threading.Event(), threading.Event(), threading.Event()
    threads = []

    class AcceptedThenRaised:
        def submit(self, call):
            def run():
                try:
                    call()
                finally:
                    completed.set()

            thread = threading.Thread(target=run)
            threads.append(thread)
            thread.start()
            assert started.wait(1)
            raise RuntimeError("fake submit acknowledgement lost")

        def shutdown(self, **kwargs):
            pass

    key = f"unknown-submit-{id(started)}"
    old = BoundedAuxiliaryRuntime("old", runtime_key=key)
    new = BoundedAuxiliaryRuntime("new", runtime_key=key)
    old._executor = AcceptedThenRaised()
    new._executor = _ImmediateExecutor()

    def blocked_sdk():
        started.set()
        assert release.wait(2), "test must release the fake SDK"

    try:
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            old.submit(blocked_sdk)
        assert old.busy is True
        old.shutdown()
        assert old.busy is True
        assert new.busy is True
        with pytest.raises(AuxiliaryBusyError):
            new.submit(lambda: None)
        release.set()
        assert completed.wait(1)
        assert old.busy is False
        ticket = new.submit(lambda: "new-result")
        assert new.wait(ticket, 1) == "new-result"
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=1)
        old.shutdown()
        new.shutdown()


def test_known_closed_executor_rejection_does_not_poison_lease():
    key = "known-closed-executor"
    closed = ThreadPoolExecutor(max_workers=1)
    closed.shutdown()
    old = BoundedAuxiliaryRuntime("old", runtime_key=key)
    old._executor = closed
    new = BoundedAuxiliaryRuntime("new", runtime_key=key)
    new._executor = _ImmediateExecutor()
    call = MagicMock()
    try:
        with pytest.raises(RuntimeError, match="shutdown"):
            old.submit(call)
        call.assert_not_called()
        assert old.busy is False
        ticket = new.submit(lambda: "new-result")
        assert new.wait(ticket, 1) == "new-result"
    finally:
        old.shutdown()
        new.shutdown()


def test_known_executor_creation_failure_does_not_poison_lease(monkeypatch):
    key = "failed-executor-construction"
    old = BoundedAuxiliaryRuntime("old", runtime_key=key)
    new = BoundedAuxiliaryRuntime("new", runtime_key=key)
    new._executor = _ImmediateExecutor()
    monkeypatch.setattr(
        runtime_module.concurrent.futures, "ThreadPoolExecutor",
        MagicMock(side_effect=RuntimeError("fake constructor refusal")),
    )
    try:
        with pytest.raises(RuntimeError, match="constructor refusal"):
            old.submit(lambda: None)
        ticket = new.submit(lambda: "new-result")
        assert new.wait(ticket, 1) == "new-result"
    finally:
        old.shutdown()
        new.shutdown()


@pytest.mark.parametrize("kind", ["classifier", "segmenter"])
def test_cleanup_failure_preserves_handle_and_explicit_retry(kind, monkeypatch):
    if kind == "classifier":
        model = OpenVocabClassifier()
        monkeypatch.setattr(model, "_create_model", lambda _: (MagicMock(), "fake-device"))
    else:
        model = InstanceSegmenter()
        monkeypatch.setattr(model, "_create_predictor", lambda: (MagicMock(), "fake-device"))
    executor = _ImmediateExecutor(shutdown_failures=1)
    model._runtime._executor = executor
    assert model.load() is True
    with pytest.raises(RuntimeError, match="cleanup failure"):
        model.shutdown()
    assert model._runtime._executor is executor
    assert model.get_status()["state"] == "cleanup_failed"
    assert model.get_status()["cleanup_pending"] is True
    assert model.get_status()["ready"] is False
    model.shutdown()
    assert executor.shutdown_calls == 2
    assert model.get_status()["state"] == "stopped"
    assert model.get_status()["cleanup_pending"] is False
    assert model.get_status()["degraded"] is False
    model.shutdown()
    assert executor.shutdown_calls == 2


def _classifier_with_box(monkeypatch, raw_box):
    model = OpenVocabClassifier(vocabulary=["shadow"])
    box = SimpleNamespace(cls=0, conf=0.99)
    if raw_box is not None:
        box.xyxy = raw_box
    sdk = MagicMock(return_value=[SimpleNamespace(boxes=[box], names={0: "shadow"})])
    monkeypatch.setattr(model, "_create_model", lambda _: (sdk, "fake-device"))
    model._runtime._executor = _ImmediateExecutor()
    assert model.load() is True
    return model, sdk


def test_localized_classifier_box_is_projected_from_clipped_crop(monkeypatch):
    model, sdk = _classifier_with_box(monkeypatch, [[1.0, 2.0, 6.0, 7.0]])
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    try:
        evidence = model.classify_with_evidence(frame, bbox=(10, 5, 10, 10))
        assert evidence == {"label": "shadow", "confidence": 0.99, "bbox": [11.0, 7.0, 16.0, 12.0]}
        assert sdk.call_args.args[0].shape == (10, 10, 3)
        assert model.classify(frame, bbox=(10, 5, 10, 10)) == ("shadow", 0.99)
    finally:
        model.shutdown()


@pytest.mark.parametrize("raw_box", [
    None, [], [[0, 0, 0, 5]], [[-1, 0, 5, 5]], [[0, 0, 11, 5]],
    [[0, 0, 5, 11]], [[0, 0, float("nan"), 5]], [[0, 0, float("inf"), 5]],
    [[0, 0, "5", 5]], [[False, 0, 5, 5]], [[0, 0, 5]],
])
def test_missing_or_invalid_detector_localization_cannot_be_used_as_bounded_evidence(monkeypatch, raw_box):
    model, _ = _classifier_with_box(monkeypatch, raw_box)
    try:
        result = model.classify_with_evidence(
            np.zeros((20, 30, 3), dtype=np.uint8), bbox=(10, 5, 10, 10),
        )
        assert result == {"label": "shadow", "confidence": 0.99, "bbox": None}
    finally:
        model.shutdown()
