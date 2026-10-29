"""Approved training and annotation roundtrips with fake workers only."""

import copy
import hashlib
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from argus.core.task_training import CudaTrainingWitness, TaskTrainingService, training_params
from argus.runtime.task_training import SpawnTrainingWorker, TaskTrainingCoordinator
from argus.storage.analysis_tasks import AnalysisTaskService
from argus.storage.database import Database
from argus.storage.models import AnalysisTaskOperationRecord
from argus.storage.task_workflows import TaskWorkflowBlocked, TaskWorkflowConflict, TaskWorkflowService
from test_task_workflows import annotation, export_ready, png_image, reviewed, seed_samples


@pytest.fixture
def environment(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'training.db'}")
    database.initialize()
    tasks = AnalysisTaskService(database)
    task = tasks.create_task("cam", "object_detection.v1", "people", {
        "rules": {"labels": ["person"]}, "collection": {"interval_seconds": 0.1},
    }, requested_by="operator", idempotency_key="create-training-test")
    service = TaskWorkflowService(database, tmp_path / "images", encoder=lambda _: png_image())
    workflow = service.create(task["task_id"], actor="operator")
    yield SimpleNamespace(db=database, tasks=tasks, task=task, service=service, workflow=workflow)
    service.close()
    database.close()


def exchange_payload(env):
    current = reviewed(env)
    envelope = env.service.export_annotation_exchange(current["workflow_id"], target="cvat")
    return {**envelope, "results": [{**envelope["samples"][0], "annotation": annotation(),
                                     "notes": "external annotation note"}]}


def test_exchange_import_never_grants_approval_preserves_notes_and_retries(environment):
    env = environment
    payload = exchange_payload(env)
    current = env.service.get(payload["workflow_id"])
    dataset = env.service.freeze(current["workflow_id"], expected_revision=current["revision"], actor="human")
    # Freeze changed the report: an old external result cannot apply to it.
    with pytest.raises(TaskWorkflowConflict):
        env.service.import_annotation_exchange(payload["workflow_id"], expected_revision=env.service.get(payload["workflow_id"])["revision"], payload=payload, actor="operator")
    payload = env.service.export_annotation_exchange(payload["workflow_id"], target="cvat")
    payload["results"] = [{**payload["samples"][0], "annotation": annotation(), "notes": "external annotation note"}]
    changed = env.service.import_annotation_exchange(payload["workflow_id"], expected_revision=payload["workflow_revision"], payload=payload, actor="operator")
    sample_id = payload["samples"][0]["sample_id"]
    review = changed["manifest"]["reviews"][sample_id]
    assert review["decision"] == "needs_review"
    assert review["notes"] == "keep this human note"
    assert changed["manifest"]["dataset_id"] is None
    assert changed["manifest"]["training"] is None
    assert env.service.get_dataset(dataset["dataset_id"])["dataset_id"] == dataset["dataset_id"]
    assert changed["manifest"]["samples"][0]["auto_annotation"]["external_notes"] == "external annotation note"
    retried = env.service.import_annotation_exchange(payload["workflow_id"], expected_revision=payload["workflow_revision"], payload=payload, actor="operator")
    assert retried["revision"] == changed["revision"]
    assert len(retried["manifest"]["review_history"]) == len(changed["manifest"]["review_history"])


@pytest.mark.parametrize("change", ["pts", "frame", "session", "sha", "labels", "report", "image", "decision", "duplicate", "bool_identity"])
def test_exchange_rejects_changed_identity_schema_or_external_approval(environment, change):
    env = environment
    payload = exchange_payload(env)
    previous = env.service.get(payload["workflow_id"])
    row = payload["results"][0]
    if change == "pts":
        row["identity"]["pts"] += 1
    elif change == "frame":
        row["identity"]["source_frame_index"] += 1
    elif change == "session":
        row["identity"]["source_session_id"] = "another-source"
    elif change == "sha":
        row["image_sha256"] = "a" * 64
    elif change == "labels":
        payload["label_schema"]["labels"] = ["another-class"]
    elif change == "report":
        payload["report_sha256"] = "a" * 64
    elif change == "image":
        row["image_size"] = [99, 88]
    elif change == "decision":
        row["decision"] = "approved"
    elif change == "bool_identity":
        row["identity"]["source_epoch"] = True
    else:
        payload["results"].append(copy.deepcopy(row))
    with pytest.raises((ValueError, TaskWorkflowConflict)):
        env.service.import_annotation_exchange(payload["workflow_id"], expected_revision=payload["workflow_revision"], payload=payload, actor="operator")
    assert env.service.get(payload["workflow_id"])["revision"] == previous["revision"]


def test_empty_exchange_rejected_and_sources_not_exported(environment):
    env = environment
    with pytest.raises(TaskWorkflowBlocked, match="NO_SAMPLES"):
        env.service.export_annotation_exchange(env.workflow["workflow_id"], target="cvat")
    current = seed_samples(env)
    exchange = env.service.export_annotation_exchange(current["workflow_id"], target="fiftyone")
    assert "source_runtime" not in exchange["samples"][0]["identity"]
    assert "relative_path" not in exchange["samples"][0]
    assert exchange["external_side_effects"] is False


def test_legacy_human_origin_cannot_approve_external_annotation(environment):
    env = environment
    current = seed_samples(env)
    exported = env.service.export_review(current["workflow_id"], target="cvat")
    sample = exported["samples"][0]
    payload = {**exported, "results": [{"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
        "origin": "human", "decision": "approved", "annotation": annotation()}]}
    with pytest.raises(TaskWorkflowBlocked, match="USE_V2"):
        env.service.import_review(current["workflow_id"], expected_revision=current["revision"], payload=payload, actor="operator")
    assert env.service.get(current["workflow_id"])["manifest"]["reviews"] == {}


def training_environment(env, tmp_path):
    current = export_ready(env)
    models = tmp_path / "models"
    models.mkdir()
    (models / "base.pt").write_bytes(b"fake-weights-no-sdk")
    service = TaskTrainingService(env.service, models / "training", models,
        trusted_base_models={"base.pt": hashlib.sha256(b"fake-weights-no-sdk").hexdigest()})
    package = env.service.export_training_package(current["workflow_id"])
    args = {"workflow_id": current["workflow_id"], "expected_revision": current["revision"],
            "expected_package_sha256": package["package_sha256"], "base_model_path": "base.pt",
            "base_model_sha256": hashlib.sha256(b"fake-weights-no-sdk").hexdigest(),
            "params": {"epochs": 1}, "actor": "operator", "idempotency_key": "submit-training"}
    return service, args


class Lease:
    def __init__(self):
        self.released = 0
        self.fail_once = False

    def release(self):
        if self.fail_once:
            self.fail_once = False
            raise OSError("fake release failure")
        self.released += 1


class FakeWorker:
    pid = 99999999

    def __init__(self):
        self.plan = None
        self.message = None
        self.cleanup = True
        self.stop_calls = 0

    def start(self, plan):
        self.plan = plan

    def poll(self):
        return self.message

    def stop(self):
        self.stop_calls += 1
        return self.cleanup


def fake_success(worker, service):
    plan = worker.plan
    output = service.output_root / plan["job_id"] / plan["execution_id"] / "candidate.onnx"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"fake-result-only-no-inference")
    return {"state": "complete", "result": {"schema": "argus.task-training-result.v1",
        "job_id": plan["job_id"], "execution_id": plan["execution_id"],
        "request_sha256": plan["request_sha256"], "package_sha256": plan["request"]["package"]["package_sha256"],
        "provider": "torch-cuda", "production_allowed": False, "stage": "candidate",
        "candidate": {"relative_path": output.relative_to(service.models_root).as_posix(),
            "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "descriptor": {"name": "Mock trained detector", "adapter_id": "onnx_detection.v1",
                "output_schema": "yolov8_detection.v1", "labels": ["person"], "input_size": [640, 640],
                "input_layout": "NCHW", "input_dtype": "float32", "color_order": "RGB"}}}}


@pytest.mark.parametrize("changed", [None, "execution_id", "artifact", "extra_field", "descriptor_input", "descriptor_color"])
def test_worker_result_has_to_match_current_execution_and_artifact(environment, tmp_path, changed):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        worker.message = fake_success(worker, service)
        if changed == "execution_id":
            worker.message["result"]["execution_id"] = "late-prior-execution"
        elif changed == "artifact":
            worker.message["result"]["candidate"]["artifact_sha256"] = "a" * 64
        elif changed == "extra_field":
            worker.message["result"]["raw_sdk_url"] = "private-local-example"
        elif changed == "descriptor_input":
            worker.message["result"]["candidate"]["descriptor"]["input_size"] = [320, 320]
        elif changed == "descriptor_color":
            worker.message["result"]["candidate"]["descriptor"]["color_order"] = "BGR"
        coordinator.tick()
        result = service.get(job["job_id"])
        assert result["status"] == ("complete" if changed is None else "failed")
        assert result["production_allowed"] is False
        assert lease.released == 1
    finally:
        coordinator.close()


def test_unknown_completion_commit_is_read_back_without_retraining(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    transition = service.transition
    raised = False
    def commit_then_lose_response(*positional, **options):
        nonlocal raised
        value = transition(*positional, **options)
        if options["status"] == "complete" and not raised:
            raised = True
            raise RuntimeError("simulated unknown successful commit")
        return value
    try:
        coordinator.tick()
        execution_id = worker.plan["execution_id"]
        worker.message = fake_success(worker, service)
        monkeypatch.setattr(service, "transition", commit_then_lose_response)
        with pytest.raises(RuntimeError, match="unknown"):
            coordinator.tick()
        assert service.get(job["job_id"])["status"] == "complete"
        coordinator.tick()
        assert worker.plan["execution_id"] == execution_id
        assert lease.released == 1
    finally:
        coordinator.close()


def test_submission_is_durable_idempotent_and_never_executes(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    assert job["status"] == "queued" and job["production_allowed"] is False
    assert service.submit(**args) == job
    restored = TaskTrainingService(environment.service, service.output_root, service.models_root)
    assert restored.get(job["job_id"]) == job
    with pytest.raises(TaskWorkflowConflict, match="idempotency"):
        service.submit(**{**args, "params": {"epochs": 2}})
    with pytest.raises(TaskWorkflowConflict, match="unresolved"):
        service.submit(**{**args, "idempotency_key": "another-training"})


def test_request_hash_does_not_make_arbitrary_pickle_trusted(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    unconfigured = TaskTrainingService(environment.service, service.output_root, service.models_root)
    with pytest.raises(TaskWorkflowBlocked, match="NOT_TRUSTED"):
        unconfigured.submit(**args)
    job = service.submit(**args)
    service.trusted_base_models.clear()
    with pytest.raises(TaskWorkflowBlocked, match="NOT_TRUSTED"):
        service.prepare(job["job_id"])


def test_resource_permission_required_before_worker_is_created(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    def forbidden_factory():
        raise AssertionError("no worker may be constructed without resource permission")
    coordinator = TaskTrainingCoordinator(service, worker_factory=forbidden_factory)
    try:
        coordinator.tick()
        assert service.get(job["job_id"])["status"] == "waiting_resources"
    finally:
        assert coordinator.close()


def test_changed_approval_before_training_is_rejected(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    current = environment.service.get(args["workflow_id"])
    sample = current["manifest"]["samples"][0]
    environment.service.review(args["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
        expected_image_sha256=sample["image_sha256"], decision="needs_review", actor="human")
    with pytest.raises(TaskWorkflowConflict):
        service.prepare(job["job_id"])


def test_cancellation_stops_worker_before_resource_release_and_retries_cleanup(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        running = service.get(job["job_id"])
        assert running["status"] == "running" and worker.plan
        cancelling = service.cancel(job["job_id"], expected_revision=running["revision"], actor="human")
        assert cancelling["status"] == "cancelling"
        worker.cleanup = False
        coordinator.tick()
        assert lease.released == 0 and service.get(job["job_id"])["status"] == "cancelling"
        worker.cleanup = True
        lease.fail_once = True
        with pytest.raises(OSError):
            coordinator.tick()
        assert service.get(job["job_id"])["status"] == "cancelled"
        assert lease.released == 0
        coordinator.tick()
        assert lease.released == 1
    finally:
        worker.cleanup = True
        coordinator.close()


def test_second_coordinator_cannot_mutate_or_take_over_live_training(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    first = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    second = TaskTrainingCoordinator(service, resource_admission=lambda _job: Lease(), worker_factory=FakeWorker)
    try:
        first.tick()
        before = service.get(job["job_id"])
        second.tick()
        assert service.get(job["job_id"]) == before
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("params", [{"device": "cpu"}, {"resume": True}, {"epochs": True}, {"imgsz": 33}, {"batch": 0}])
def test_training_parameters_are_closed(params):
    with pytest.raises(ValueError):
        training_params(params)


def test_restarted_running_job_keeps_identity_and_requires_positive_worker_exit(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    claimed, plan = service.claim(job["job_id"], expected_revision=job["revision"])
    service.transition(job["job_id"], expected_revision=claimed["revision"], from_states={"running"},
                       status="running", result={"execution_id": plan["execution_id"], "worker_pid": 99999999})
    coordinator = TaskTrainingCoordinator(service)
    try:
        monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: True)
        coordinator.tick()
        changed = service.get(job["job_id"])
        assert changed["status"] == "reconciling"
        assert changed["result"]["execution_id"] == plan["execution_id"]
        with pytest.raises(TaskWorkflowConflict):
            coordinator.reconcile_exited(job["job_id"], expected_revision=changed["revision"])
        monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: False)
        assert coordinator.reconcile_exited(job["job_id"], expected_revision=changed["revision"])["status"] == "failed"
    finally:
        coordinator.close()


def test_start_return_flag_is_not_process_cleanup_evidence():
    class FakeProcess:
        pid = 123456789
        alive = False
        terminated = False

        def start(self):
            self.alive = True
            raise OSError("fake start created process but lost acknowledgement")

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False
            self.terminated = True

        def join(self, timeout):
            pass

    worker = SpawnTrainingWorker()
    worker.process = FakeProcess()
    worker._start_attempted = True
    with pytest.raises(OSError):
        worker.process.start()  # A fake process only; never the training entry point.
    assert worker._process_started is False and worker.process.is_alive()
    assert worker.stop()
    assert worker.process.terminated and not worker.process.is_alive()


def test_unknown_start_without_os_identity_does_not_claim_cleanup():
    closed = []
    worker = SpawnTrainingWorker()
    worker._start_attempted = True
    worker.process = SimpleNamespace(pid=None)
    worker.connection = SimpleNamespace(close=lambda: closed.append("parent"))
    worker._child_connection = SimpleNamespace(close=lambda: closed.append("child"))
    assert worker.stop() is False
    assert closed == ["parent", "child"]
    assert worker.connection is None and worker._child_connection is None
    assert worker._cleaned is False


def test_unknown_start_closes_pipes_but_retains_lease_until_positive_exit(environment, tmp_path):
    closed = []
    class UnknownStartWorker(SpawnTrainingWorker):
        def prepare(self, _plan):
            # This fixture only simulates an unknown start outcome; never
            # inherit the real spawn/readiness preparation entry point.
            pass

        def start(self, plan):
            self._start_attempted = True
            self.process = SimpleNamespace(pid=None)
            self.connection = SimpleNamespace(close=lambda: closed.append("parent"))
            self._child_connection = SimpleNamespace(close=lambda: closed.append("child"))
            raise OSError("fake start outcome unknown")
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = UnknownStartWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        assert closed == ["parent", "child"]
        assert service.get(job["job_id"])["status"] == "reconciling"
        assert job["job_id"] in coordinator._owned
        assert coordinator._owner.held and lease.released == 0
        coordinator.tick()
        assert lease.released == 0 and job["job_id"] in coordinator._owned
        # An explicit fake OS observation later establishes exit. Pipe closure
        # alone never allowed the coordinator to release this training lease.
        worker.process = SimpleNamespace(pid=123456789, is_alive=lambda: False, join=lambda timeout: None)
        coordinator.tick()
        assert service.get(job["job_id"])["status"] == "failed"
        assert lease.released == 1
    finally:
        worker.process = SimpleNamespace(pid=123456789, is_alive=lambda: False, join=lambda timeout: None)
        assert coordinator.close()


def test_constructor_failure_still_closes_both_owned_pipe_ends():
    closed = []
    worker = SpawnTrainingWorker()
    worker.connection = SimpleNamespace(close=lambda: closed.append("parent"))
    worker._child_connection = SimpleNamespace(close=lambda: closed.append("child"))
    assert worker.process is None
    assert worker.stop() is True
    assert closed == ["parent", "child"]


def test_factory_failure_and_failed_lease_release_remain_owned(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    lease = Lease()
    lease.fail_once = True
    calls = []
    def failing_factory():
        calls.append("factory")
        raise OSError("fake constructor failure")
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease,
        worker_factory=failing_factory)
    try:
        with pytest.raises(OSError, match="release"):
            coordinator.tick()
        assert job["job_id"] in coordinator._pending_leases
        assert coordinator._owner.held and lease.released == 0
        coordinator.tick()
        assert calls == ["factory"]
        assert lease.released == 1
        assert service.get(job["job_id"])["status"] == "failed"
    finally:
        coordinator.close()


def test_claim_commit_lost_response_is_confirmed_not_started_without_pid_guess(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    claim = service.claim
    def committed_then_raised(*positional, **options):
        claim(*positional, **options)
        raise OSError("fake committed claim with lost response")
    monkeypatch.setattr(service, "claim", committed_then_raised)
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        result = service.get(job["job_id"])
        assert worker.plan is None
        assert result["status"] == "failed"
        assert result["result"]["worker_start_confirmed_absent"] is True
        assert lease.released == 1
        coordinator.tick()
        assert worker.plan is None and lease.released == 1
    finally:
        coordinator.close()


def test_unknown_started_worker_retains_lease_until_cleanup_retry(environment, tmp_path):
    class FailedStartWorker(FakeWorker):
        def start(self, plan):
            self.plan = plan
            self.cleanup = False
            raise OSError("fake started worker acknowledgement failure")
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FailedStartWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        assert service.get(job["job_id"])["status"] == "reconciling"
        assert lease.released == 0 and job["job_id"] in coordinator._owned
        worker.cleanup = True
        coordinator.tick()
        assert service.get(job["job_id"])["status"] == "failed"
        assert lease.released == 1
    finally:
        worker.cleanup = True
        coordinator.close()


@pytest.mark.parametrize("device,loss_device", [("cpu", "cpu"), ("cuda:0", "cpu"), ("cuda:0", "cuda:0")])
def test_cuda_witness_checks_real_batch_model_and_loss_not_reload(device, loss_device):
    witness = CudaTrainingWitness()
    trainer = SimpleNamespace(device=device, model=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=device)])),
                              loss=SimpleNamespace(device=loss_device))
    if device == "cuda:0" and loss_device == "cuda:0":
        witness.batch_start(trainer)
        witness.batch_end(trainer)
        assert witness.completed_batches == 1
    else:
        with pytest.raises(TaskWorkflowBlocked):
            witness.batch_end(trainer)
        assert witness.completed_batches == 0


def candidate_registration_environment(env, service, tmp_path):
    """Use the real app registrar and registry; artifacts remain fake bytes."""
    from argus.config.schema import ArgusConfig, StorageConfig
    from argus.dashboard.app import create_app

    app = create_app(database=env.db, config=ArgusConfig(storage=StorageConfig(
        models_dir=service.models_root, baselines_dir=tmp_path / "baselines",
    )), alerts_dir=str(tmp_path / "alerts"))
    env.service._candidate_registrar = app.state.task_workflows._candidate_registrar
    env.service._candidate_validator = app.state.task_workflows._candidate_validator
    return app


@pytest.mark.parametrize("initial_mismatch", [False, True])
def test_completed_job_candidate_roundtrip_preserves_expected_artifact_sha(environment, tmp_path, initial_mismatch):
    env = environment
    service, args = training_environment(env, tmp_path)
    app = candidate_registration_environment(env, service, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        worker.message = fake_success(worker, service)
        coordinator.tick()
        completed = service.get(job["job_id"])
        assert completed["status"] == "complete"
        candidate = completed["result"]["candidate"]
        artifact = service.models_root / candidate["relative_path"]
        original = artifact.read_bytes()
        before = env.service.get(args["workflow_id"])
        if initial_mismatch:
            artifact.write_bytes(b"changed-after-training")
            with pytest.raises(ValueError, match="content"):
                env.service.import_candidate(args["workflow_id"], expected_revision=before["revision"],
                    candidate=candidate, expected_package_sha256=args["expected_package_sha256"], actor="human")
            assert env.service.get(args["workflow_id"])["revision"] == before["revision"]
            assert env.service.get(args["workflow_id"])["state"] == "training_export_ready"
            assert app.state.analysis_task_service.list_models() == []
            artifact.write_bytes(original)
        imported = env.service.import_candidate(args["workflow_id"], expected_revision=before["revision"],
            candidate=candidate, expected_package_sha256=args["expected_package_sha256"], actor="human")
        assert imported["state"] == "candidate_registered"
        models = app.state.analysis_task_service.list_models()
        assert len(models) == 1 and models[0]["artifact_sha256"] == candidate["artifact_sha256"]
        assert imported["manifest"]["training"]["candidate"]["artifact_id"] == models[0]["model_id"]
        # A committed registration receipt remains retryable without the
        # mutable file, while later execution still requires its exact SHA.
        artifact.unlink()
        retried = env.service.import_candidate(args["workflow_id"], expected_revision=before["revision"],
            candidate=candidate, expected_package_sha256=args["expected_package_sha256"], actor="human")
        assert retried["revision"] == imported["revision"]
    finally:
        coordinator.close()
        app.state.task_workflows.close()


def test_swap_after_preflight_does_not_pollute_registration_receipt_and_same_body_retries(environment, tmp_path, monkeypatch):
    env = environment
    service, args = training_environment(env, tmp_path)
    app = candidate_registration_environment(env, service, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        worker.message = fake_success(worker, service)
        coordinator.tick()
        candidate = service.get(job["job_id"])["result"]["candidate"]
        artifact = service.models_root / candidate["relative_path"]
        original = artifact.read_bytes()
        registry = app.state.analysis_task_service
        fingerprint = registry._trusted_artifact
        calls = 0
        def swap_before_authoritative_registration(path):
            nonlocal calls
            calls += 1
            if calls == 3:  # preflight, registrar validation, registry's read
                artifact.write_bytes(b"swapped-before-registry-commit")
            return fingerprint(path)
        monkeypatch.setattr(registry, "_trusted_artifact", swap_before_authoritative_registration)
        before = env.service.get(args["workflow_id"])
        from argus.storage.analysis_tasks import AnalysisTaskConflictError
        with pytest.raises(AnalysisTaskConflictError, match="expected training result"):
            env.service.import_candidate(args["workflow_id"], expected_revision=before["revision"],
                candidate=candidate, expected_package_sha256=args["expected_package_sha256"], actor="human")
        assert env.service.get(args["workflow_id"])["state"] == "candidate_importing"
        assert registry.list_models() == []
        with env.db.get_session() as session:
            assert session.scalar(select(AnalysisTaskOperationRecord).where(
                AnalysisTaskOperationRecord.action == "register_task_model")) is None
        artifact.write_bytes(original)
        # The exact original request, including its old revision, reconciles
        # the same pending import after restoring the authorized bytes.
        imported = env.service.import_candidate(args["workflow_id"], expected_revision=before["revision"],
            candidate=candidate, expected_package_sha256=args["expected_package_sha256"], actor="human")
        assert imported["state"] == "candidate_registered"
        assert len(registry.list_models()) == 1
        assert registry.list_models()[0]["artifact_sha256"] == candidate["artifact_sha256"]
    finally:
        coordinator.close()
        app.state.task_workflows.close()


def external_candidate_environment(env, tmp_path, *, include_sha=True):
    service, args = training_environment(env, tmp_path)
    app = candidate_registration_environment(env, service, tmp_path)
    artifact = service.models_root / "external.onnx"
    artifact.write_bytes(b"opaque external candidate without model execution")
    candidate = {"relative_path": "external.onnx", "descriptor": {
        "name": "External mock", "adapter_id": "onnx_detection.v1",
        "output_schema": "yolov8_detection.v1", "labels": ["person"], "input_size": [640, 640],
    }}
    if include_sha:
        candidate["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return app, artifact, candidate, {
        "expected_revision": args["expected_revision"], "candidate": candidate,
        "expected_package_sha256": args["expected_package_sha256"], "actor": "human",
    }


def test_adversarial_registration_unknown_commit_reconciles_without_mutable_source(environment, tmp_path):
    env = environment
    app, artifact, _candidate, arguments = external_candidate_environment(env, tmp_path)
    registrar = env.service._candidate_registrar
    calls = []
    def lose_registration_ack(**kwargs):
        result = registrar(**kwargs)
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("registration committed but acknowledgement lost")
        return result
    env.service._candidate_registrar = lose_registration_ack
    try:
        with pytest.raises(OSError, match="acknowledgement"):
            env.service.import_candidate(env.workflow["workflow_id"], **arguments)
        assert len(app.state.analysis_task_service.list_models()) == 1
        artifact.unlink()
        done = env.service.import_candidate(env.workflow["workflow_id"], **{**arguments, "actor": "other-human"})
        assert done["state"] == "candidate_registered"
        assert len(app.state.analysis_task_service.list_models()) == 1
        assert calls[0]["idempotency_key"] == calls[1]["idempotency_key"]
        assert calls[0]["actor"] == calls[1]["actor"] == "human"
    finally:
        app.state.task_workflows.close()


def test_adversarial_optional_sha_import_cannot_change_artifact_on_pending_retry(environment, tmp_path):
    env = environment
    app, artifact, _candidate, arguments = external_candidate_environment(env, tmp_path, include_sha=False)
    registrar = env.service._candidate_registrar
    calls = []
    def fail_before_registration(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            raise OSError("registrar unavailable before write")
        return registrar(**kwargs)
    env.service._candidate_registrar = fail_before_registration
    try:
        with pytest.raises(OSError, match="unavailable"):
            env.service.import_candidate(env.workflow["workflow_id"], **arguments)
        artifact.write_bytes(b"different unapproved bytes at the same path")
        with pytest.raises((ValueError, TaskWorkflowConflict)):
            env.service.import_candidate(env.workflow["workflow_id"], **arguments)
        assert app.state.analysis_task_service.list_models() == []
        assert calls[0]["candidate"] == calls[1]["candidate"]
        assert calls[0]["candidate"]["artifact_sha256"] == hashlib.sha256(
            b"opaque external candidate without model execution").hexdigest()
    finally:
        app.state.task_workflows.close()


def test_adversarial_concurrent_candidate_imports_reconcile_same_final_receipt(environment):
    env = environment
    current = export_ready(env)
    package = env.service.export_training_package(current["workflow_id"])
    barrier = threading.Barrier(2)
    calls = []
    def registrar(**kwargs):
        calls.append(kwargs)
        barrier.wait(timeout=10)
        return {"artifact_id": "one-registered-candidate"}
    env.service._candidate_registrar = registrar
    arguments = {"expected_revision": current["revision"], "candidate": {"path": "mock.onnx"},
                 "expected_package_sha256": package["package_sha256"], "actor": "human"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(env.service.import_candidate, current["workflow_id"], **arguments) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert results[0] == results[1]
    assert results[0]["state"] == "candidate_registered"
    assert calls[0]["idempotency_key"] == calls[1]["idempotency_key"]


def test_adversarial_pending_retry_cannot_replace_original_expected_revision(environment):
    env = environment
    current = export_ready(env)
    package = env.service.export_training_package(current["workflow_id"])
    calls = []
    def registrar(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("unknown registrar outcome")
        return {"artifact_id": "candidate"}
    env.service._candidate_registrar = registrar
    arguments = {"expected_revision": current["revision"], "candidate": {"path": "mock.onnx"},
                 "expected_package_sha256": package["package_sha256"], "actor": "human"}
    with pytest.raises(OSError):
        env.service.import_candidate(current["workflow_id"], **arguments)
    with pytest.raises(TaskWorkflowConflict, match="revision"):
        env.service.import_candidate(current["workflow_id"], **{**arguments, "expected_revision": 999999})
    assert len(calls) == 1


def test_adversarial_old_training_approval_does_not_survive_candidate_import_intent(environment, tmp_path):
    env = environment
    service, args = training_environment(env, tmp_path)
    job = service.submit(**args)
    original = env.service.get(args["workflow_id"])
    def unknown_registrar(**_kwargs):
        raise OSError("external candidate registration outcome unknown")
    env.service._candidate_registrar = unknown_registrar
    with pytest.raises(OSError):
        env.service.import_candidate(args["workflow_id"], expected_revision=original["revision"],
            candidate={"path": "external.onnx"}, expected_package_sha256=args["expected_package_sha256"], actor="human")
    importing = env.service.get(args["workflow_id"])
    assert importing["manifest_sha256"] == original["manifest_sha256"]
    assert importing["revision"] != original["revision"]
    with pytest.raises(TaskWorkflowConflict, match="approval"):
        service.prepare(job["job_id"])


def test_adversarial_late_training_result_cannot_commit_after_approval_changes(environment, tmp_path, monkeypatch):
    env = environment
    service, args = training_environment(env, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        worker.message = fake_success(worker, service)
        prepare = service.prepare
        def change_approval_after_prepare(job_id):
            plan = prepare(job_id)
            current = env.service.get(args["workflow_id"])
            sample = current["manifest"]["samples"][0]
            env.service.review(args["workflow_id"], sample["sample_id"], expected_revision=current["revision"],
                expected_image_sha256=sample["image_sha256"], decision="needs_review", actor="human")
            return plan
        monkeypatch.setattr(service, "prepare", change_approval_after_prepare)
        coordinator.tick()
        assert service.get(job["job_id"])["status"] == "failed"
        assert lease.released == 1
    finally:
        coordinator.close()


@pytest.mark.parametrize("cleanup_confirmed", [False, True])
@pytest.mark.parametrize("finish_method", ["tick", "close"])
def test_adversarial_old_worker_result_cannot_overwrite_new_execution_identity(environment, tmp_path, cleanup_confirmed, finish_method):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        worker.message = fake_success(worker, service)
        running = service.get(job["job_id"])
        replacement = service.transition(job["job_id"], expected_revision=running["revision"], from_states={"running"},
            status="running", result={"execution_id": "b" * 32, "worker_pid": 99999998})
        worker.cleanup = cleanup_confirmed
        finish = getattr(coordinator, finish_method)
        finish()
        assert service.get(job["job_id"]) == replacement
        if not cleanup_confirmed:
            assert lease.released == 0 and job["job_id"] in coordinator._owned
            worker.cleanup = True
            finish()
        assert service.get(job["job_id"]) == replacement
        assert lease.released == 1 and job["job_id"] not in coordinator._owned
    finally:
        worker.cleanup = True
        coordinator.close()


def duplicate_training_job(service, job_id, *, new_job_id="training-second", status="queued", result=None):
    """Submit a second real approved task/workflow; all bytes remain synthetic."""
    submission = service.prepare(job_id)["request"]["submission"]
    tasks, workflows = AnalysisTaskService(service._db), service.workflows
    task = tasks.create_task("cam", "object_detection.v1", "independent queue task", {
        "rules": {"labels": ["person"]}, "collection": {"interval_seconds": 0.1},
    }, requested_by=submission["actor"], idempotency_key=f"create-{new_job_id}")
    workflow = workflows.create(task["task_id"], actor=submission["actor"])
    env = SimpleNamespace(db=service._db, tasks=tasks, task=task, service=workflows, workflow=workflow)
    ready = export_ready(env)
    package = workflows.export_training_package(ready["workflow_id"])
    queued = service.submit(workflow_id=ready["workflow_id"], expected_revision=ready["revision"],
        expected_package_sha256=package["package_sha256"], base_model_path=submission["base_model_path"],
        base_model_sha256=submission["base_model_sha256"], params=submission["params"],
        actor=submission["actor"], idempotency_key=new_job_id)
    if status == "queued" and result is None:
        return queued
    return service.transition(queued["job_id"], expected_revision=queued["revision"],
        from_states={"queued"}, status=status, result=result)


def test_training_single_slot_holds_later_jobs_without_second_resource_request(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    first = service.submit(**args)
    second = duplicate_training_job(service, first["job_id"])
    workers, leases, requests = [], [], []
    def admission(job):
        requests.append(job["job_id"])
        lease = Lease()
        leases.append(lease)
        return lease
    def factory():
        worker = FakeWorker()
        workers.append(worker)
        return worker
    coordinator = TaskTrainingCoordinator(service, resource_admission=admission, worker_factory=factory)
    try:
        coordinator.tick()
        coordinator.tick()
        assert len(workers) == len(leases) == len(requests) == 1
        assert service.get(second["job_id"])["status"] == "queued"
        workers[0].message = {"state": "failed", "error_code": "TRAINING_MOCK_DONE"}
        coordinator.tick()
        assert leases[0].released == 1
        assert len(workers) == 2 and requests == [first["job_id"], second["job_id"]]
    finally:
        assert coordinator.close()


@pytest.mark.parametrize("pid_state", [None, True, "unknown", "error"])
def test_unresolved_orphan_blocks_even_older_queue_and_retains_shutdown_owner(environment, tmp_path, monkeypatch, pid_state):
    service, args = training_environment(environment, tmp_path)
    queued = service.submit(**args)
    orphan = duplicate_training_job(service, queued["job_id"], status="running",
        result={"execution_id": "a" * 32, "worker_pid": None if pid_state is None else 123456789})
    def liveness(_pid):
        if pid_state == "error":
            raise OSError("fake OS liveness unknown")
        return None if pid_state == "unknown" else True
    monkeypatch.setattr("argus.runtime.task_training._process_alive", liveness)
    requests = []
    coordinator = TaskTrainingCoordinator(service,
        resource_admission=lambda job: requests.append(job["job_id"]), worker_factory=FakeWorker)
    try:
        coordinator.tick()
        coordinator.tick()
        assert requests == []
        assert service.get(queued["job_id"])["status"] == "queued"
        current = service.get(orphan["job_id"])
        assert current["status"] == "reconciling"
        assert current["result"]["execution_id"] == "a" * 32
        assert coordinator.close() is False and coordinator._owner.held
        with pytest.raises(TaskWorkflowConflict):
            coordinator.reconcile_exited(orphan["job_id"], expected_revision=current["revision"])
    finally:
        # Test-only evidence of exit. Missing PID cannot be reconciled by a
        # guessed PID; explicitly update that controlled fixture's identity.
        current = service.get(orphan["job_id"])
        if current["result"]["worker_pid"] is None:
            current = service.transition(orphan["job_id"], expected_revision=current["revision"],
                from_states={"reconciling"}, status="reconciling",
                result={**current["result"], "worker_pid": 123456789})
        monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: False)
        coordinator.reconcile_exited(orphan["job_id"], expected_revision=current["revision"])
        assert coordinator.close()


@pytest.mark.parametrize("original_state,expected_status", [("running", "failed"), ("cancelling", "cancelled"), ("reconciling", "failed")])
def test_restart_auto_recovers_only_confirmed_exit_and_preserves_cancellation(environment, tmp_path, monkeypatch, original_state, expected_status):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    claimed, plan = service.claim(job["job_id"], expected_revision=job["revision"])
    service.transition(job["job_id"], expected_revision=claimed["revision"], from_states={"running"},
        status=original_state, result={"execution_id": plan["execution_id"], "worker_pid": 123456789})
    monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: False)
    coordinator = TaskTrainingCoordinator(service, worker_factory=lambda: pytest.fail("orphan must never retrain"))
    try:
        coordinator.tick()
        coordinator.tick()
        result = service.get(job["job_id"])
        assert result["status"] == expected_status
        assert result["result"]["worker_exit_confirmed"] is True
        assert result["result"]["execution_id"] == plan["execution_id"]
    finally:
        assert coordinator.close()


def test_human_reconciliation_requires_current_revision_and_never_assumes_pid_reuse_is_exit(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    claimed, plan = service.claim(job["job_id"], expected_revision=job["revision"])
    current = service.transition(job["job_id"], expected_revision=claimed["revision"], from_states={"running"},
        status="reconciling", result={"execution_id": plan["execution_id"], "worker_pid": 123456789})
    probes = []
    monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda pid: probes.append(pid) or True)
    coordinator = TaskTrainingCoordinator(service)
    try:
        with pytest.raises(TaskWorkflowConflict, match="revision"):
            coordinator.reconcile_exited(job["job_id"], expected_revision=current["revision"] - 1)
        assert probes == []
        with pytest.raises(TaskWorkflowConflict, match="exit"):
            coordinator.reconcile_exited(job["job_id"], expected_revision=current["revision"])
        assert probes == [123456789]
    finally:
        monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: False)
        coordinator.reconcile_exited(job["job_id"], expected_revision=current["revision"])
        assert coordinator.close()


def test_resource_denial_has_stable_reason_and_revision_then_resumes_exactly_once(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    class Admission:
        granted = False
        def __call__(self, _job):
            return lease if self.granted else None
        def snapshot(self):
            return {"error_code": "TRAINING_MAINTENANCE_APPROVAL_REQUIRED"}
    admission = Admission()
    coordinator = TaskTrainingCoordinator(service, resource_admission=admission, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        denied = service.get(job["job_id"])
        assert denied["status"] == "waiting_resources"
        assert denied["result"] == {"error_code": "TRAINING_MAINTENANCE_APPROVAL_REQUIRED"}
        coordinator.tick()
        assert service.get(job["job_id"]) == denied
        admission.granted = True
        coordinator.tick()
        execution = worker.plan["execution_id"]
        coordinator.tick()
        assert worker.plan["execution_id"] == execution
        assert service.get(job["job_id"])["status"] == "running"
    finally:
        assert coordinator.close()


@pytest.mark.parametrize("failure", ["db", "release", "worker"])
def test_close_retries_cleanup_without_releasing_owner_on_failure(environment, tmp_path, monkeypatch, failure):
    service, args = training_environment(environment, tmp_path)
    service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    coordinator.tick()
    transition = service.transition
    if failure == "db":
        monkeypatch.setattr(service, "transition", lambda *_a, **_k: (_ for _ in ()).throw(OSError("fake commit unknown")))
    elif failure == "release":
        lease.fail_once = True
    else:
        worker.cleanup = False
    try:
        assert coordinator.close() is False
        assert coordinator._owner.held and coordinator._owned
        assert lease.released == 0
        monkeypatch.setattr(service, "transition", transition)
        worker.cleanup = True
        assert coordinator.close() is True
        assert not coordinator._owner.held and not coordinator._owned
        assert lease.released == 1
    finally:
        monkeypatch.setattr(service, "transition", transition)
        worker.cleanup = True
        coordinator.close()


def test_os_database_lock_still_excludes_owner_when_process_local_registry_is_absent(environment, tmp_path):
    from argus.runtime.task_training import _OWNERS, _OWNERS_LOCK

    service, _args = training_environment(environment, tmp_path)
    first, second = TaskTrainingCoordinator(service), TaskTrainingCoordinator(service)
    assert first.start()
    # Simulate the second interpreter's empty process-local registry, without
    # starting any subprocess. The OS byte/flock lock remains genuinely held.
    with _OWNERS_LOCK:
        retained = _OWNERS.pop(first._owner.key)
    try:
        assert second.start() is False
        assert first._owner.held and not second._owner.held
    finally:
        with _OWNERS_LOCK:
            _OWNERS[first._owner.key] = retained
        assert second.close()
        assert first.close()


def test_adversarial_channel_poll_error_becomes_cleanup_message_not_infinite_retry():
    worker = SpawnTrainingWorker()
    worker.connection = SimpleNamespace(poll=lambda: (_ for _ in ()).throw(OSError("private pipe failure")))
    worker.process = SimpleNamespace(is_alive=lambda: True)
    worker.started, worker.timeout = 0, 60
    assert worker.poll() == {"state": "failed", "error_code": "TRAINING_WORKER_CHANNEL_LOST"}


def test_adversarial_cancel_committed_after_claim_never_starts_training(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    claim = service.claim
    def claim_then_cancel(*positional, **options):
        claimed, plan = claim(*positional, **options)
        service.cancel(claimed["job_id"], expected_revision=claimed["revision"], actor="human")
        return claimed, plan
    monkeypatch.setattr(service, "claim", claim_then_cancel)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        assert worker.plan is None
        assert service.get(job["job_id"])["status"] == "cancelled"
        assert lease.released == 1
    finally:
        coordinator.close()


def test_adversarial_unknown_database_state_blocks_older_queue_and_owner_release(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    queued = service.submit(**args)
    unknown = duplicate_training_job(service, queued["job_id"], status="unrecognized_execution_state")
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        assert worker.plan is None
        assert coordinator.snapshot()["error_code"] == "TRAINING_JOB_STATE_UNKNOWN"
        assert coordinator.close() is False and coordinator._owner.held
    finally:
        current = service.get(unknown["job_id"])
        service.transition(unknown["job_id"], expected_revision=current["revision"],
            from_states={current["status"]}, status="failed", result={"error_code": "TRAINING_MOCK_CLEANED"})
        coordinator.close()


def test_adversarial_cancelling_cleanup_failure_is_visible_in_cached_snapshot(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        running = service.get(job["job_id"])
        service.cancel(job["job_id"], expected_revision=running["revision"], actor="human")
        worker.cleanup = False
        coordinator.tick()
        assert coordinator.snapshot()["error_code"] == "TRAINING_WORKER_CLEANUP_UNCONFIRMED"
        assert coordinator.snapshot()["cleanup_pending"] is True
        assert lease.released == 0 and coordinator._owner.held
    finally:
        worker.cleanup = True
        coordinator.close()


@pytest.mark.parametrize("message", [[], "private invalid IPC", {"state": "failed", "error_code": {"private": "sdk url"}}])
def test_adversarial_invalid_worker_message_fails_closed_without_leaking_or_retaining_forever(environment, tmp_path, message):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        worker.message = message
        coordinator.tick()
        result = service.get(job["job_id"])
        assert result["status"] == "failed"
        assert result["result"]["error_code"] == "TRAINING_RESULT_INVALID"
        assert lease.released == 1
    finally:
        coordinator.close()


def test_adversarial_owner_failed_close_keeps_os_lock_and_retryable_handle(environment, tmp_path):
    from argus.runtime.task_training import _OWNERS, _OWNERS_LOCK

    service, _args = training_environment(environment, tmp_path)
    first, second = TaskTrainingCoordinator(service), TaskTrainingCoordinator(service)
    assert first.start()
    stream = first._owner.stream
    class FailedClose:
        fail = True
        def __getattr__(self, name):
            return getattr(stream, name)
        def close(self):
            if self.fail:
                raise OSError("synthetic close not confirmed")
            return stream.close()
    wrapper = FailedClose()
    first._owner.stream = wrapper
    try:
        assert first.close() is False
        with _OWNERS_LOCK:
            retained = _OWNERS.pop(first._owner.key)
        try:
            assert second.start() is False  # OS handle, not registry, must still exclude it.
        finally:
            second.close()
            with _OWNERS_LOCK:
                _OWNERS[first._owner.key] = retained
        wrapper.fail = False
        assert first.close() is True
    finally:
        # Baseline failure may already have unlocked the OS handle. Cleanup
        # this test-owned descriptor without relying on its broken retry path.
        wrapper.fail = False
        if not first.close():
            stream.close()
            first._owner.held = False
            first._owner.stream = None
            with _OWNERS_LOCK:
                if _OWNERS.get(first._owner.key) is first:
                    _OWNERS.pop(first._owner.key)
        second.close()


@pytest.mark.parametrize("authorization", [None, {"state": "cancel"}, {"state": "authorize", "execution_id": "different"}])
def test_adversarial_child_never_computes_before_exact_parent_authorization(tmp_path, monkeypatch, authorization):
    from argus.runtime import task_training

    calls, messages, locks = [], [], []
    class Ownership:
        worker_path = tmp_path / "worker.lock"
        def __init__(self, *_args, **_kwargs):
            locks.append(self)
        def acquire(self, _owner):
            return True
        def release(self, _owner):
            raise AssertionError("child lock must live until OS process exit")
    class Connection:
        def poll(self, _timeout):
            return authorization is not None
        def recv(self):
            return authorization
        def send(self, value):
            messages.append(value)
        def close(self):
            pass
    monkeypatch.setattr("argus.runtime.compute_ownership.ComputeOwnership", Ownership)
    monkeypatch.setattr("argus.core.task_training.execute_training_plan", lambda plan: calls.append(plan) or {})
    monkeypatch.setattr(task_training.threading, "Thread", lambda **_kwargs: SimpleNamespace(start=lambda: None))
    task_training._training_child(Connection(), {"execution_id": "a" * 32})
    assert calls == []
    assert locks
    assert not any(message.get("state") == "complete" for message in messages)


def test_adversarial_worker_unknown_liveness_cannot_certify_cleanup():
    worker = SpawnTrainingWorker()
    worker.process = SimpleNamespace(pid=123456789, is_alive=lambda: None, join=lambda timeout: None)
    worker._start_attempted = True
    assert worker.stop() is False
    assert worker._cleaned is False


def test_child_exact_handshake_takes_secondary_before_mock_engine_and_never_explicitly_releases(tmp_path, monkeypatch):
    from argus.runtime import task_training

    order, arguments = [], []
    class Ownership:
        def __init__(self, path, **kwargs):
            arguments.append((path, kwargs))
        def acquire(self, _owner):
            order.append("lock")
            return True
        def release(self, _owner):
            pytest.fail("child OS lifetime, not SDK return, owns secondary release")
    class Connection:
        def send(self, value):
            order.append(value["state"])
        def poll(self, timeout):
            assert timeout == 5
            return True
        def recv(self):
            return {"state": "authorize", "execution_id": "a" * 32}
        def close(self):
            order.append("closed")
    monkeypatch.setattr("argus.runtime.compute_ownership.ComputeOwnership", Ownership)
    monkeypatch.setattr("argus.core.task_training.execute_training_plan", lambda _plan: order.append("mock_engine") or {})
    monkeypatch.setattr(task_training.threading, "Thread", lambda **_kwargs: SimpleNamespace(start=lambda: order.append("watcher")))
    path = tmp_path / "controlled-worker.lock"
    task_training._training_child(Connection(), {"execution_id": "a" * 32}, path)
    assert order == ["lock", "ready", "watcher", "mock_engine", "complete", "closed"]
    assert arguments == [(path, {"guard_training_workers": False})]


def test_spawn_readiness_prepares_only_controlled_lock_path_without_authorizing_or_loading_engine(tmp_path):
    sent, spawned = [], []
    execution_id = "a" * 32
    parent = SimpleNamespace(poll=lambda timeout: timeout == 5,
        recv=lambda: {"state": "ready", "execution_id": execution_id}, send=sent.append, close=lambda: None)
    child = SimpleNamespace(close=lambda: None)
    process = SimpleNamespace(pid=123456789, start=lambda: spawned.append("spawned"),
        is_alive=lambda: False, join=lambda timeout: None)
    def process_factory(**kwargs):
        spawned.append(kwargs)
        return process
    context = SimpleNamespace(Pipe=lambda **_kwargs: (parent, child), Process=process_factory)
    path = tmp_path / "controlled-worker.lock"
    worker = SpawnTrainingWorker(context=context, worker_ownership_path=path)
    plan = {"job_id": "fake", "execution_id": execution_id, "request_sha256": "b" * 64,
            "request": {"submission": {"params": {"timeout_seconds": 60}}}}
    try:
        worker.prepare(plan)
        assert sent == [] and worker._authorized is False
        assert spawned[0]["args"] == (child, plan, str(path))
        assert "worker_ownership_path" not in plan and len(spawned) == 2
    finally:
        assert worker.stop()


def test_prepare_runs_outside_writer_authorization_but_start_runs_inside(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    service.submit(**args)
    original = service.execution_authorization
    inside, order = False, []
    @contextmanager
    def observed(*positional, **kwargs):
        nonlocal inside
        with original(*positional, **kwargs):
            inside = True
            order.append("authorization")
            try:
                yield
            finally:
                inside = False
    class PreparedWorker(FakeWorker):
        def prepare(self, _plan):
            assert not inside
            order.append("prepare")
        def start(self, plan):
            assert inside
            order.append("start")
            super().start(plan)
    monkeypatch.setattr(service, "execution_authorization", observed)
    worker, lease = PreparedWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        assert order == ["prepare", "authorization", "start"]
    finally:
        assert coordinator.close()


def test_adversarial_resource_release_failure_stays_visible_after_terminal_commit(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        coordinator.tick()
        worker.message = {"state": "failed", "error_code": "TRAINING_MOCK_DONE"}
        lease.fail_once = True
        with pytest.raises(OSError):
            coordinator.tick()
        assert service.get(job["job_id"])["status"] == "failed"
        assert coordinator.snapshot()["cleanup_pending"] is True
        assert coordinator.snapshot()["error_code"] is not None
        assert coordinator._owner.held and lease.released == 0
        coordinator.tick()
        assert coordinator.snapshot()["cleanup_pending"] is False and lease.released == 1
    finally:
        coordinator.close()


def test_adversarial_received_none_is_invalid_message_not_an_idle_channel():
    worker = SpawnTrainingWorker()
    worker.connection = SimpleNamespace(poll=lambda: True, recv=lambda: None)
    worker.process = SimpleNamespace(is_alive=lambda: True)
    worker.started, worker.timeout = 0, 60
    assert worker.poll() == {"state": "failed", "error_code": "TRAINING_RESULT_INVALID"}


def test_adversarial_stop_inside_execution_authorization_never_starts_worker(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = FakeWorker(), Lease()
    coordinator = TaskTrainingCoordinator(service, resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    original = service.execution_authorization
    @contextmanager
    def stop_before_yield(*positional, **kwargs):
        with original(*positional, **kwargs):
            coordinator.request_stop()
            yield
    monkeypatch.setattr(service, "execution_authorization", stop_before_yield)
    try:
        coordinator.tick()
        assert worker.plan is None
        assert worker.stop_calls >= 1 and lease.released == 1
        current = service.get(job["job_id"])
        assert current["status"] == "failed"
        assert current["result"]["error_code"] == "TRAINING_COORDINATOR_CLOSED"
    finally:
        coordinator.close()
