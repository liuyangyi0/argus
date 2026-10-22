"""Durable background releases, restart fencing and responsive HTTP behavior."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.config.schema import DashboardConfig
from argus.dashboard.app import create_app
from argus.storage.model_release_service import (
    ModelReleaseService,
    RuntimeReleaseCommit,
    RuntimeReleaseIdentity,
    RuntimeReleasePreparation,
)
from argus.storage.models import Base, ModelRecord, ModelReleaseOperation
from argus.storage.release_execution import ModelReleaseWorker, ReleaseExecutionBusy
from argus.storage.release_pipeline import ReleasePipeline


@pytest.fixture
def sessions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'releases.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        session.add(ModelRecord(
            model_version_id="test-v1",
            camera_id="cam-1",
            model_type="patchcore",
            model_hash="a" * 64,
            data_hash="b" * 64,
            stage="candidate",
            is_active=False,
        ))
        session.commit()
    yield factory
    engine.dispose()


class _Runtime:
    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.allow_finish = threading.Event()
        self.allow_finish.set()

    def prepare_model_release(self, plan):
        self.calls += 1
        self.entered.set()
        if not self.allow_finish.wait(5):
            raise RuntimeError("test runtime timed out")

        class Transaction:
            def commit(self):
                return RuntimeReleaseCommit(True, state={"applied": True})

            def abort(self):
                return RuntimeReleaseCommit(True, state={"restored": True})

            def finalize(self):
                return RuntimeReleaseCommit(True)

        return RuntimeReleasePreparation(
            True,
            identity=RuntimeReleaseIdentity(
                camera_id=plan.camera_id,
                model_version_id=plan.model_version_id,
                target_stage=plan.target_stage,
                runtime_artifact_sha256=plan.runtime_artifact_sha256,
            ),
            transaction=Transaction(),
        )


def _service(sessions, runtime, events=None):
    return ModelReleaseService(
        sessions,
        ReleasePipeline(sessions, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=runtime.prepare_model_release,
        event_publisher=(
            lambda topic, payload: events.append((topic, payload))
        ) if events is not None else None,
    )


def _wait_terminal(service, operation_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        operation = service.get_operation(operation_id)
        if operation.status != "pending":
            return operation
        time.sleep(0.01)
    raise AssertionError("release operation did not become terminal")


def test_submit_is_durable_and_does_no_runtime_work(sessions):
    runtime = _Runtime()
    service = _service(sessions, runtime)
    operation, model = service.submit_transition(
        model_version_id="test-v1",
        target_stage="shadow",
        requested_by="operator",
        idempotency_key="same-request",
    )
    assert operation.status == "pending"
    assert model.stage == "candidate"
    assert runtime.calls == 0
    assert service.get_operation(operation.operation_id).status == "pending"
    same, _ = service.submit_transition(
        model_version_id="test-v1",
        target_stage="shadow",
        requested_by="operator",
        idempotency_key="same-request",
    )
    assert same.operation_id == operation.operation_id
    applied, _ = service.execute_operation(operation.operation_id)
    assert applied.status == "applied"
    service.execute_operation(operation.operation_id)
    assert runtime.calls == 1


def test_worker_owns_queue_and_duplicate_notifications_do_not_reprepare(sessions):
    runtime = _Runtime()
    runtime.allow_finish.clear()
    events = []
    service = _service(sessions, runtime, events)
    worker = ModelReleaseWorker(service, poll_seconds=0.01)
    worker.start()
    try:
        operation, _ = service.submit_transition(
            model_version_id="test-v1",
            target_stage="shadow",
            requested_by="operator",
        )
        worker.notify()
        assert runtime.entered.wait(2)
        for _ in range(5):
            worker.notify()
        assert service.get_operation(operation.operation_id).status == "pending"
        runtime.allow_finish.set()
        assert _wait_terminal(service, operation.operation_id).status == "applied"
        worker.stop()
        assert runtime.calls == 1
        assert [payload["type"] for _, payload in events] == [
            "release_operation_pending", "release_operation_applied",
        ]
    finally:
        runtime.allow_finish.set()
        worker.stop()


def test_second_worker_cannot_recover_or_execute_an_owned_database(sessions):
    first = ModelReleaseWorker(_service(sessions, _Runtime()))
    second = ModelReleaseWorker(_service(sessions, _Runtime()))
    first.start()
    try:
        with pytest.raises(ReleaseExecutionBusy):
            second.start()
    finally:
        first.stop()
    second.start()
    second.stop()


@pytest.mark.parametrize("target_stage", ["shadow", "production"])
def test_startup_fails_interrupted_operation_without_replaying_approval(
    sessions, target_stage,
):
    runtime = _Runtime()
    events = []
    service = _service(sessions, runtime, events)
    operation, _ = service.submit_transition(
        model_version_id="test-v1",
        target_stage=target_stage,
        requested_by="operator",
        idempotency_key="old-approval",
        release_evidence_sources={"untrusted": "never read after restart"},
    )
    worker = ModelReleaseWorker(service, poll_seconds=0.01)
    worker.start()
    try:
        failed = service.get_operation(operation.operation_id)
        assert failed.status == "failed"
        assert "new authorized request" in failed.error
        assert runtime.calls == 0
        with sessions() as session:
            model = session.query(ModelRecord).one()
            assert model.stage == "candidate"
            assert model.release_revision == 0
        assert events[-1][1]["type"] == "release_operation_failed"
    finally:
        worker.stop()


def test_worker_failure_is_durable_and_broadcast(sessions):
    runtime = _Runtime()

    def reject(_plan):
        raise RuntimeError("standby failed")

    runtime.prepare_model_release = reject
    events = []
    service = _service(sessions, runtime, events)
    worker = ModelReleaseWorker(service, poll_seconds=0.01)
    worker.start()
    try:
        operation, _ = service.submit_transition(
            model_version_id="test-v1",
            target_stage="shadow",
            requested_by="operator",
        )
        worker.notify()
        failed = _wait_terminal(service, operation.operation_id)
        worker.stop()
        assert failed.status == "failed"
        assert "standby failed" in failed.error
        assert events[-1][1]["type"] == "release_operation_failed"
        with sessions() as session:
            assert session.query(ModelRecord).one().stage == "candidate"
    finally:
        worker.stop()


def test_release_http_returns_before_slow_runtime_and_other_api_remains_usable(sessions):
    runtime = _Runtime()
    runtime.allow_finish.clear()
    database = SimpleNamespace(get_session=sessions)
    app = create_app(
        database=database,
        camera_manager=runtime,
        config=SimpleNamespace(
            dashboard=DashboardConfig(go2rtc_enabled=False),
            cameras=[],
        ),
    )
    app.state._release_pipeline = ReleasePipeline(
        sessions, min_shadow_days=0, min_canary_days=0,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/models/test-v1/promote",
                json={"target_stage": "shadow", "idempotency_key": "http-once"},
            )
            assert response.status_code == 202
            data = response.json()["data"]
            assert data["operation"]["status"] == "pending"
            assert data["operation_id"] == data["operation"]["operation_id"]
            assert data["runtime_synced"] is False
            assert runtime.entered.wait(2)
            operation_url = f"/api/models/release-operations/{data['operation_id']}"
            assert client.get(operation_url).json()["data"]["operation"]["status"] == "pending"
            assert client.get("/api/models/json").status_code == 200
            runtime.allow_finish.set()
            operation = _wait_terminal(app.state.model_release_service, data["operation_id"])
            assert operation.status == "applied"
            retry = client.post(
                "/api/models/test-v1/promote",
                json={"target_stage": "shadow", "idempotency_key": "http-once"},
            )
            assert retry.json()["data"]["operation_id"] == data["operation_id"]
            assert runtime.calls == 1
    finally:
        runtime.allow_finish.set()


def test_app_shutdown_keeps_process_shared_mjpeg_executor_usable():
    from argus.dashboard.routes.cameras import _STREAM_EXECUTOR

    config = SimpleNamespace(dashboard=DashboardConfig(go2rtc_enabled=False), cameras=[])
    for _ in range(2):
        with TestClient(create_app(config=config)):
            assert _STREAM_EXECUTOR.submit(lambda: "alive").result(timeout=2) == "alive"
        assert _STREAM_EXECUTOR.submit(lambda: "still alive").result(timeout=2) == "still alive"


def test_production_evidence_build_runs_only_after_submission(sessions, tmp_path, monkeypatch):
    runtime = _Runtime()
    service = _service(sessions, runtime)
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"model weights")
    with sessions() as session:
        session.query(ModelRecord).one().model_path = str(artifact)
        session.commit()
    calls = []

    def reject_evidence(sources, **kwargs):
        calls.append((sources, kwargs))
        raise ValueError("strict v3 and signed soak evidence rejected")

    monkeypatch.setattr(
        "argus.storage.release_evidence.build_release_evidence_from_sources",
        reject_evidence,
    )
    operation, _ = service.submit_transition(
        model_version_id="test-v1",
        target_stage="production",
        requested_by="operator",
        release_evidence_sources={"missing": "evidence"},
    )
    assert operation.status == "pending"
    assert runtime.calls == 0
    assert calls == []
    failed, _ = service.execute_operation(operation.operation_id)
    assert failed.status == "failed"
    assert "strict v3 and signed soak evidence rejected" in failed.error
    assert len(calls) == 1
    assert calls[0][0] == {"missing": "evidence"}
    assert calls[0][1]["operator_id"] == "operator"
    assert calls[0][1]["operator_approved"] is True
    assert calls[0][1]["approved_at"].endswith("+00:00")
    assert runtime.calls == 0
    with sessions() as session:
        assert session.query(ModelReleaseOperation).one().status == "failed"


def test_stranded_execution_claim_becomes_failed_without_a_second_prepare(sessions):
    runtime = _Runtime()
    service = _service(sessions, runtime)
    operation, _ = service.submit_transition(
        model_version_id="test-v1", target_stage="shadow", requested_by="operator",
    )
    with sessions() as session:
        row = session.query(ModelReleaseOperation).filter_by(
            operation_id=operation.operation_id,
        ).one()
        row.runtime_state_json = '{"execution_owner":"interrupted-thread"}'
        session.commit()
    failed, model = service.execute_operation(operation.operation_id)
    assert failed.status == "failed"
    assert "new authorized request" in failed.error
    assert model.stage == "candidate"
    assert runtime.calls == 0
