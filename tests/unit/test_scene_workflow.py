"""Scene workflow state/ownership tests use fake decoders, SDKs and databases."""

from __future__ import annotations

import hashlib
import json
import stat
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus.storage import scene_workflow as workflow
from argus.anomaly.scene_training import SceneTrainingInput
from argus.config.schema import SceneGuardConfig as SceneGuardPolicy


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _Database:
    def __init__(self):
        self.audits = []
        self.jobs = {}
        self.save_calls = 0
        self.save_failure = None

    def get_session(self):
        database = self

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def add(self, value):
                database.audits.append(value)

            def commit(self):
                return None

        return Session()

    def save_training_job(self, job):
        self.save_calls += 1
        if self.save_failure == "before":
            raise RuntimeError("unknown database commit")
        self.jobs[job.job_id] = job
        if self.save_failure == "after":
            raise RuntimeError("unknown database commit")

    def get_training_job(self, job_id):
        return self.jobs.get(job_id)


@pytest.fixture
def service(tmp_path, monkeypatch):
    profiles, active = {}, {}

    class Store:
        def __init__(self, root):
            self.root = Path(root)

        def create_from_baseline(self, camera_id, profile_id, source, *, metadata, activate=False):
            key = (str(self.root), camera_id, profile_id)
            target = self.root / camera_id / profile_id
            if target.exists():
                raise FileExistsError("immutable profile exists")
            target.mkdir(parents=True)
            source = Path(source)
            raw = source.read_bytes()
            # Simulate the real store's JPEG normalization: the source identity
            # and normalized reference identity are intentionally different.
            normalized = b"normalized-reference:" + raw
            reference = target / "reference.jpg"
            reference.write_bytes(normalized)
            profile = SimpleNamespace(
                camera_id=camera_id, profile_id=profile_id, reference_path=reference,
                reference_sha256=_sha(normalized), source_baseline_name=source.name,
                source_baseline_sha256=_sha(raw), metadata=dict(metadata),
            )
            profiles[key] = profile
            if activate:
                self.set_active(camera_id, profile_id)
            return profile

        def load(self, camera_id, profile_id):
            return profiles[(str(self.root), camera_id, profile_id)]

        def set_active(self, camera_id, profile_id):
            active[(str(self.root), camera_id)] = profile_id
            return self.load(camera_id, profile_id)

        def get_active_profile_id(self, camera_id):
            return active.get((str(self.root), camera_id))

        def list_profiles(self, camera_id):
            return [value for (root, camera, _id), value in profiles.items()
                    if root == str(self.root) and camera == camera_id]

    monkeypatch.setattr(workflow, "SceneProfileStore", Store)
    database, registry = _Database(), Mock()
    baselines = tmp_path / "baselines"
    reference = baselines / "cam1" / "default" / "v001" / "normal.png"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"operator-selected-normal-reference")
    instance = workflow.SceneWorkflowService(
        baselines_dir=baselines, profiles_dir=tmp_path / "scene_profiles",
        database=database, registry=registry,
    )
    return SimpleNamespace(
        service=instance, database=database, registry=registry, reference=reference,
        profiles=profiles, active=active, store_type=Store,
    )


def _create(fixture, name="scene1", actor="creator"):
    return fixture.service.create(
        "cam1", name, reference_path="default/v001/normal.png",
        reference_sha256=_sha(fixture.reference.read_bytes()),
        normal_reference_confirmed=True, actor=actor,
    )


def _sources(fixture):
    root = fixture.service.sources_root
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in (1, 2):
        (root / f"normal-{index}.mp4").write_bytes(f"source-{index}".encode())
        (root / f"normal-{index}.json").write_text("{}", encoding="utf-8")
        sources.append({"video": f"normal-{index}.mp4", "annotations": f"normal-{index}.json"})
    return sources


def _lineage(record):
    return {
        "schema_version": "argus.scene-training-input.v1",
        **{key: record[key] for key in (
            "workflow_id", "scene_profile_id", "baseline_sha256", "manifest_sha256", "reference_sha256",
        )},
    }


def _seed_training(fixture, monkeypatch, *, name="scene1", complete=False):
    _create(fixture, name)
    service = fixture.service
    baseline = service._baseline("cam1", name)
    baseline.mkdir(parents=True)
    record = service._load("cam1", name)
    draft = fixture.store_type(service.root).load("cam1", name)
    (baseline / "reference.jpg").write_bytes(draft.reference_path.read_bytes())
    manifest = {
        "schema_version": workflow.BASELINE_SCHEMA, "training_ready": True,
        "scene_guard": deepcopy(service.scene_guard),
        "reference": {"sha256": record["reference_sha256"]},
        "split_contract": {"sample_counts": {"train": 100, "calibration": 50, "diagnostic_only": 0}},
    }
    (baseline / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record.update(
        state="baseline_ready", baseline_sha256="b" * 64,
        manifest_sha256=_sha((baseline / "manifest.json").read_bytes()),
    )
    service._save(record)

    def validate(*, baselines_dir, camera_id, scene_workflow, **kwargs):
        root = service._baseline(camera_id, scene_workflow["workflow_id"])
        return SceneTrainingInput(
            baselines_dir=Path(baselines_dir), camera_id=camera_id,
            root=root, images_dir=root / "images", image_count=150,
            training_count=100, calibration_count=50, lineage=dict(scene_workflow),
        )

    monkeypatch.setattr("argus.anomaly.scene_training.validate_scene_training_input", validate)
    monkeypatch.setattr("argus.anomaly.artifacts.artifact_metadata", lambda _path: {"calibration_status": "ok"})
    monkeypatch.setattr(workflow, "registry_training_data_hash", lambda _path: "b" * 64)
    monkeypatch.setattr(workflow, "runtime_artifact_sha256", lambda _path: "d" * 64)
    if not complete:
        return record
    job_id = service.submit_training("cam1", name, model_type="patchcore", image_size=256, actor="creator")
    job = fixture.database.jobs[job_id]
    job.status = "complete"
    job.confirmed_by = "training-reviewer"
    job.confirmed_at = datetime.now(timezone.utc)
    job.model_version_id = "trained-version"
    model_path = service.baselines_dir.parent / "models" / name / "model.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"scene runtime")
    (model_path.parent / "calibration.json").write_text(json.dumps({
        "info": 0.1, "low": 0.11, "medium": 0.12, "high": 0.13,
        "n_samples": 50, "sorted_scores": [0.1] * 50,
        "target_fprs": {"info": 0.1, "low": 0.01, "medium": 0.001, "high": 0.0001},
        "actual_fprs": {"info": 0.0, "low": 0.0, "medium": 0.0, "high": 0.0},
    }), encoding="utf-8")
    model = SimpleNamespace(
        camera_id="cam1", model_version_id="trained-version", stage="candidate",
        data_hash="b" * 64, model_path=str(model_path), model_hash="d" * 16,
        training_params=json.dumps({"scene_workflow": _lineage(service._load("cam1", name))}),
    )
    fixture.registry.get_by_version_id.return_value = model
    return job, model


def test_draft_creation_requires_actor_reference_confirmation_and_never_activates(service):
    before = service.reference.read_bytes()
    result = _create(service)
    assert result["state"] == "draft" and result["approved"] is False and result["active"] is False
    draft = service.store_type(service.service.root).load("cam1", "scene1")
    assert draft.metadata["operator_approved"] is False and draft.metadata["validation_only"] is True
    assert draft.metadata["normal_reference_confirmed_by"] == "creator"
    assert service.reference.read_bytes() == before and service.active == {}
    assert service.database.audits[0].user == "creator"
    service.registry.get_by_version_id.assert_not_called()


@pytest.mark.parametrize("actor", [None, "", "unknown", "   ", " Unknown ", 123])
def test_anonymous_draft_creation_cannot_publish_a_profile(service, actor):
    with pytest.raises(ValueError, match="authenticated"):
        _create(service, actor=actor)
    assert service.profiles == {} and service.active == {}


@pytest.mark.parametrize("relative", ["../outside.png", "/outside.png", "C:/private.png", "default\\v1\\x.png", "default/./x.png"])
def test_reference_paths_cannot_escape_or_select_arbitrary_server_files(service, relative):
    with pytest.raises(ValueError):
        service.service.create(
            "cam1", "scene1", reference_path=relative, reference_sha256="a" * 64,
            normal_reference_confirmed=True, actor="creator",
        )
    assert service.profiles == {} and service.database.audits == []


def test_changed_reference_and_duplicate_workflow_are_not_automatically_adopted(service):
    with pytest.raises(workflow.SceneWorkflowConflict, match="reference changed"):
        service.service.create(
            "cam1", "scene1", reference_path="default/v001/normal.png", reference_sha256="a" * 64,
            normal_reference_confirmed=True, actor="creator",
        )
    _create(service)
    with pytest.raises(FileExistsError):
        _create(service)
    assert service.service._load("cam1", "scene1")["created_by"] == "creator"


def test_baseline_task_uses_the_real_taskmanager_camera_metadata_calling_convention(service, monkeypatch):
    _create(service)
    sources = _sources(service)
    sources_before = {item["video"]: (service.service.sources_root / item["video"]).read_bytes() for item in sources}
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        target = kwargs["output_dir"]
        target.mkdir(parents=True)
        manifest = {
            "schema_version": workflow.BASELINE_SCHEMA, "training_ready": True,
            "scene_guard": deepcopy(service.service.scene_guard),
            "reference": {"sha256": service.service._load("cam1", "scene1")["reference_sha256"]},
        }
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr("argus.validation.scene_profile_baseline.build_scene_profile_baseline", build)

    class Tasks:
        def submit(self, kind, func, camera_id="", **kwargs):
            # TaskManager consumes camera_id as metadata, not as func kwargs.
            assert kind == "scene_baseline" and camera_id == "cam1"
            func(lambda *_args: None, **kwargs)
            return "baseline-task"

    task_id = service.service.start_baseline(
        "cam1", "scene1", sources=sources, sample_every=30, max_images=2000,
        actor="creator", task_manager=Tasks(),
    )
    record = service.service._load("cam1", "scene1")
    assert task_id == record["baseline_task_id"] == "baseline-task"
    assert record["state"] == "baseline_ready"
    assert calls[0]["camera_id"] == "cam1"
    assert calls[0]["strict_release_decoder"] is True and calls[0]["allow_alignable"] is False
    assert calls[0]["min_training_images"] == 100 and calls[0]["min_calibration_images"] == 50
    policy = calls[0]["scene_guard_config"]
    assert policy.min_matches == 40 and policy.min_inliers == 18
    assert policy.min_coverage_ratio == 0.12 and policy.min_inlier_ratio == 0.45
    assert all((service.service.sources_root / name).read_bytes() == raw for name, raw in sources_before.items())
    assert service.active == {}


@pytest.mark.parametrize("field,value", [
    ("enabled", False), ("allow_alignable", True), ("fallback_to_baseline", True),
    ("require_profile_model_binding", False), ("require_operator_approved_profile", False),
])
def test_workflow_refuses_non_strict_deployment_policy_before_creating_authority(service, field, value):
    policy = SceneGuardPolicy.model_validate({**SceneGuardPolicy().model_dump(), field: value})
    strict = workflow.SceneWorkflowService(
        baselines_dir=service.service.baselines_dir, profiles_dir=service.service.profiles_dir,
        database=service.database, registry=service.registry, scene_guard_policy=policy,
    )
    with pytest.raises(ValueError, match="strict SceneGuard"):
        strict.create(
            "cam1", "scene1", reference_path="default/v001/normal.png",
            reference_sha256=_sha(service.reference.read_bytes()), normal_reference_confirmed=True, actor="creator",
        )
    assert service.profiles == {} and service.database.audits == []


def test_changed_camera_thresholds_cannot_reuse_an_existing_workflow(service):
    _create(service)
    sources = _sources(service)
    stricter = workflow.SceneWorkflowService(
        baselines_dir=service.service.baselines_dir, profiles_dir=service.service.profiles_dir,
        database=service.database, registry=service.registry,
        scene_guard_policy=SceneGuardPolicy(min_matches=41),
    )
    tasks = Mock()
    with pytest.raises(workflow.SceneWorkflowConflict, match="policy changed"):
        stricter.start_baseline(
            "cam1", "scene1", sources=sources, sample_every=30, max_images=2000,
            actor="creator", task_manager=tasks,
        )
    tasks.submit.assert_not_called()
    assert service.service._load("cam1", "scene1")["state"] == "draft"


def test_unknown_baseline_submission_keeps_reservation_and_blocks_a_second_worker(service):
    _create(service)
    sources = _sources(service)
    tasks = Mock()
    tasks.submit.side_effect = RuntimeError("submit outcome unknown")
    with pytest.raises(RuntimeError, match="outcome unknown"):
        service.service.start_baseline(
            "cam1", "scene1", sources=sources, sample_every=30, max_images=2000,
            actor="creator", task_manager=tasks,
        )
    record = service.service._load("cam1", "scene1")
    assert record["state"] == "building_baseline" and "unconfirmed" in record["error"]
    with pytest.raises(workflow.SceneWorkflowConflict):
        service.service.start_baseline(
            "cam1", "scene1", sources=sources, sample_every=30, max_images=2000,
            actor="creator", task_manager=tasks,
        )
    tasks.submit.assert_called_once()


@pytest.mark.parametrize("max_concurrent,other_camera", [(1, "cam2"), (2, "cam1")])
def test_known_task_rejection_does_not_permanently_reserve_an_unstarted_workflow(
    service, monkeypatch, max_concurrent, other_camera,
):
    from argus.dashboard import tasks

    class HeldWorker:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(tasks, "threading", SimpleNamespace(
        Thread=HeldWorker, Lock=tasks.threading.Lock, Event=tasks.threading.Event,
    ))
    manager = tasks.TaskManager(max_concurrent=max_concurrent)
    owner = manager.submit("scene_baseline", Mock(), camera_id=other_camera)
    _create(service)
    sources = _sources(service)
    with pytest.raises(workflow.SceneWorkflowConflict) as caught:
        service.service.start_baseline(
            "cam1", "scene1", sources=sources, sample_every=30, max_images=2000,
            actor="creator", task_manager=manager,
        )
    assert isinstance(caught.value.__cause__, tasks.TaskSubmissionRejected)
    record = service.service._load("cam1", "scene1")
    assert record["state"] == "baseline_failed" and record["baseline_task_id"] is None
    assert "not started" in record["error"] and len(manager.get_all_tasks()) == 1
    assert not service.service._baseline("cam1", "scene1").exists()

    manager.get_task(owner).status = tasks.TaskStatus.COMPLETE
    retry = service.service.start_baseline(
        "cam1", "scene1", sources=sources, sample_every=30, max_images=2000,
        actor="creator", task_manager=manager,
    )
    record = service.service._load("cam1", "scene1")
    assert record["state"] == "building_baseline" and record["baseline_task_id"] == retry
    assert manager.get_task(retry).status == tasks.TaskStatus.PENDING
    assert service.active == {}


@pytest.mark.parametrize("damage", ["v2", "not-ready", "wrong-reference"])
def test_bad_baseline_result_is_not_published_as_ready_or_silently_overwritten(service, monkeypatch, damage):
    _create(service)
    sources = _sources(service)

    def build(**kwargs):
        target = kwargs["output_dir"]
        target.mkdir(parents=True)
        manifest = {
            "schema_version": "argus.scene-profile-baseline.v2" if damage == "v2" else workflow.BASELINE_SCHEMA,
            "training_ready": damage != "not-ready",
            "scene_guard": deepcopy(service.service.scene_guard),
            "reference": {"sha256": "a" * 64 if damage == "wrong-reference" else service.service._load("cam1", "scene1")["reference_sha256"]},
        }
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr("argus.validation.scene_profile_baseline.build_scene_profile_baseline", build)
    with pytest.raises(ValueError):
        service.service._build_baseline(
            lambda *_args: None, camera_id="cam1", workflow_id="scene1",
            videos=[service.service.sources_root / row["video"] for row in sources],
            annotations=[service.service.sources_root / row["annotations"] for row in sources],
            sample_every=30, max_images=2000,
        )
    assert service.service._load("cam1", "scene1")["state"] == "baseline_failed"
    tasks = Mock()
    with pytest.raises(workflow.SceneWorkflowConflict, match="output already exists"):
        service.service.start_baseline(
            "cam1", "scene1", sources=sources, sample_every=30, max_images=2000,
            actor="creator", task_manager=tasks,
        )
    tasks.submit.assert_not_called()


def test_existing_operation_lock_is_not_stolen_or_removed(service):
    _create(service)
    lock = service.service._directory("cam1", "scene1") / ".operation.lock"
    lock.write_text("original owner", encoding="utf-8")
    with pytest.raises(workflow.SceneWorkflowConflict, match="busy"):
        with service.service._lock("cam1", "scene1"):
            pytest.fail("another operation must not enter")
    assert lock.read_text(encoding="utf-8") == "original owner"


def test_waiting_metadata_writer_times_out_without_expiring_someone_elses_lock(service, monkeypatch):
    _create(service)
    lock = service.service._directory("cam1", "scene1") / ".operation.lock"
    lock.write_text("original owner", encoding="utf-8")
    ticks = iter([0.0, 0.0, 6.0])
    pause = Mock()
    monkeypatch.setattr(workflow, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=pause))
    with pytest.raises(workflow.SceneWorkflowConflict, match="busy"):
        with service.service._lock("cam1", "scene1", wait=True):
            pytest.fail("timed out owner cannot be replaced")
    pause.assert_called_once_with(0.01)
    assert lock.read_text(encoding="utf-8") == "original owner"


def test_scene_training_submission_requires_separate_human_confirmation_and_has_no_paths(service, monkeypatch):
    _seed_training(service, monkeypatch)
    job_id = service.service.submit_training("cam1", "scene1", model_type="patchcore", image_size=256, actor="creator")
    job = service.database.jobs[job_id]
    assert job.status == "pending_confirmation" and job.confirmation_required is True
    assert job.camera_id == "cam1" and job.dataset_version == "scene-scene1"
    assert job.triggered_by == "creator"
    params = json.loads(job.hyperparameters)
    assert set(params) == {"scene_workflow", "image_size"}
    assert params["scene_workflow"] == _lineage(service.service._load("cam1", "scene1"))
    assert service.active == {}
    with pytest.raises(workflow.SceneWorkflowConflict):
        service.service.submit_training("cam1", "scene1", model_type="patchcore", image_size=256, actor="creator")
    assert service.database.save_calls == 1


@pytest.mark.parametrize("phase", ["before", "after"])
def test_training_commit_unknown_reserves_one_job_id_without_duplicate_submission(service, monkeypatch, phase):
    _seed_training(service, monkeypatch)
    service.database.save_failure = phase
    with pytest.raises(RuntimeError, match="unknown database commit"):
        service.service.submit_training("cam1", "scene1", model_type="patchcore", image_size=256, actor="creator")
    saved = service.service._load("cam1", "scene1")
    assert saved["state"] == "training_submitted" and saved["training_job_id"]
    if phase == "after":
        assert set(service.database.jobs) == {saved["training_job_id"]}
    with pytest.raises(workflow.SceneWorkflowConflict):
        service.service.submit_training("cam1", "scene1", model_type="patchcore", image_size=256, actor="creator")
    assert service.database.save_calls == 1


@pytest.mark.parametrize("status", ["failed", "rejected"])
def test_known_terminal_training_can_be_explicitly_resubmitted_with_history_and_confirmation(service, monkeypatch, status):
    _seed_training(service, monkeypatch)
    first = service.service.submit_training("cam1", "scene1", model_type="patchcore", image_size=256, actor="creator")
    service.database.jobs[first].status = status
    second = service.service.submit_training("cam1", "scene1", model_type="patchcore", image_size=256, actor="retry-operator")
    record = service.service._load("cam1", "scene1")
    assert first != second and record["training_job_id"] == second
    assert record["training_job_history"] == [first]
    assert service.database.jobs[first].status == status
    assert service.database.jobs[second].status == "pending_confirmation"
    assert service.database.jobs[second].confirmation_required is True
    assert service.database.jobs[second].triggered_by == "retry-operator"


@pytest.mark.parametrize("field,value", [
    ("camera_id", "other-camera"), ("job_type", "backbone"), ("zone_id", "other-zone"),
    ("dataset_version", "scene-other"), ("confirmed_by", None), ("confirmed_by", ""),
    ("confirmed_by", "   "), ("confirmed_by", " UNKNOWN "),
    ("confirmed_at", None), ("confirmed_at", "2026-10-22T12:30:00+00:00"),
    ("confirmed_at", True), ("confirmation_required", False),
])
def test_profile_approval_refuses_foreign_or_unconfirmed_training(service, monkeypatch, field, value):
    job, _model = _seed_training(service, monkeypatch, complete=True)
    setattr(job, field, value)
    with pytest.raises((workflow.SceneWorkflowConflict, ValueError)):
        service.service.approve_profile("cam1", "scene1", actor="approver")
    assert service.active == {}
    assert not (service.service.profiles_dir / "cam1" / "scene1").exists()


@pytest.mark.parametrize("damage", ["foreign-workflow", "registry-lineage", "data-hash", "runtime-hash", "calibration"])
def test_profile_approval_rechecks_model_lineage_calibration_and_artifact_identity(service, monkeypatch, damage):
    job, model = _seed_training(service, monkeypatch, complete=True)
    if damage == "foreign-workflow":
        params = json.loads(job.hyperparameters)
        params["scene_workflow"]["workflow_id"] = "other"
        params["scene_workflow"]["scene_profile_id"] = "other"
        job.hyperparameters = json.dumps(params)
    elif damage == "registry-lineage":
        model.training_params = "{}"
    elif damage == "data-hash":
        model.data_hash = "c" * 64
    elif damage == "runtime-hash":
        model.model_hash = "c" * 16
    else:
        monkeypatch.setattr("argus.anomaly.artifacts.artifact_metadata", lambda _path: {"calibration_status": "failed"})
    with pytest.raises(ValueError):
        service.service.approve_profile("cam1", "scene1", actor="approver")
    assert not (service.service.profiles_dir / "cam1" / "scene1").exists()
    assert service.active == {} and model.stage == "candidate"


def test_profile_approval_binds_calibrated_model_but_does_not_select_it_or_change_stage(service, monkeypatch):
    _job, model = _seed_training(service, monkeypatch, complete=True)
    profile_id = service.service.approve_profile("cam1", "scene1", actor="approver")
    profile = service.store_type(service.service.profiles_dir).load("cam1", profile_id)
    record = service.service._load("cam1", "scene1")
    assert profile.metadata["approved_by"] == record["approved_by"] == "approver"
    assert datetime.fromisoformat(record["approved_at"]).tzinfo is not None
    assert profile.source_baseline_sha256 == record["reference_sha256"]
    assert profile.reference_sha256 != record["reference_sha256"]  # JPEG normalization is allowed.
    assert record["approved"] is True and record["state"] == "profile_approved"
    assert model.stage == "candidate" and service.active == {}
    assert not (service.service._baseline("cam1", "scene1").parent / "current.txt").exists()


def test_cached_service_rechecks_new_profile_root_links_before_writing(service, monkeypatch):
    _seed_training(service, monkeypatch, complete=True)
    original_lstat = Path.lstat

    def substituted_root(path, *args, **kwargs):
        if path == service.service.profiles_dir:
            return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", substituted_root)
    with pytest.raises(ValueError, match="links/reparse"):
        service.service.approve_profile("cam1", "scene1", actor="approver")
    assert service.active == {}


def test_approval_retry_reconciles_exact_profile_without_replacing_the_original_approver(service, monkeypatch):
    _seed_training(service, monkeypatch, complete=True)
    real_save = service.service._save
    fail_once = True

    def fail_after_publication(record):
        nonlocal fail_once
        if record["state"] == "profile_approved" and fail_once:
            fail_once = False
            raise OSError("workflow record commit failed")
        real_save(record)

    monkeypatch.setattr(service.service, "_save", fail_after_publication)
    with pytest.raises(OSError):
        service.service.approve_profile("cam1", "scene1", actor="original-approver")
    profile = service.store_type(service.service.profiles_dir).load("cam1", "scene1")
    original_metadata = deepcopy(profile.metadata)
    assert service.service.approve_profile("cam1", "scene1", actor="retrying-operator") == "scene1"
    assert profile.metadata == original_metadata
    record = service.service._load("cam1", "scene1")
    assert record["approved_by"] == "original-approver"
    assert record["approved_at"] == original_metadata["approved_at"]


def test_profile_selection_requires_approval_and_changes_only_profile_and_baseline_pointers(service, monkeypatch):
    _job, model = _seed_training(service, monkeypatch, complete=True)
    with pytest.raises(workflow.SceneWorkflowConflict, match="approval"):
        service.service.activate_profile(
            "cam1", "scene1", actor="operator",
            expected_active_profile_id=None, expected_active_profile_revision=None,
        )
    service.service.approve_profile("cam1", "scene1", actor="approver")
    assert service.service.activate_profile(
        "cam1", "scene1", actor="selector",
        expected_active_profile_id=None, expected_active_profile_revision=None,
    ) == "scene1"
    record = service.service._load("cam1", "scene1")
    assert record["state"] == "profile_selected" and record["selected_by"] == "selector"
    assert (service.service._baseline("cam1", "scene1").parent / "current.txt").read_text(encoding="utf-8") == "scene-scene1"
    assert service.active[(str(service.service.profiles_dir), "cam1")] == "scene1"
    assert model.stage == "candidate"


def test_partial_profile_selection_is_persisted_and_blocks_reapproval_or_blind_retry(service, monkeypatch):
    _job, model = _seed_training(service, monkeypatch, complete=True)
    service.service.approve_profile("cam1", "scene1", actor="approver")
    original = service.store_type.set_active
    monkeypatch.setattr(service.store_type, "set_active", Mock(side_effect=OSError("profile pointer write failed")))
    with pytest.raises(OSError):
        service.service.activate_profile(
            "cam1", "scene1", actor="selector",
            expected_active_profile_id=None, expected_active_profile_revision=None,
        )
    record = service.service._load("cam1", "scene1")
    assert record["state"] == "selection_incomplete" and record["error"]
    assert service.active == {} and model.stage == "candidate"
    monkeypatch.setattr(service.store_type, "set_active", original)
    with pytest.raises(workflow.SceneWorkflowConflict):
        service.service.activate_profile(
            "cam1", "scene1", actor="selector",
            expected_active_profile_id=None, expected_active_profile_revision=None,
        )
    with pytest.raises(workflow.SceneWorkflowConflict):
        service.service.approve_profile("cam1", "scene1", actor="approver")
    assert service.service._load("cam1", "scene1")["state"] == "selection_incomplete"


def test_camera_selection_lock_blocks_other_workflow_mutation_without_stealing_it(service, monkeypatch):
    _seed_training(service, monkeypatch, complete=True)
    service.service.approve_profile("cam1", "scene1", actor="approver")
    lock = service.service.root / "cam1" / ".activation.lock"
    lock.write_text("other selection", encoding="utf-8")
    with pytest.raises(workflow.SceneWorkflowConflict):
        service.service.activate_profile(
            "cam1", "scene1", actor="selector",
            expected_active_profile_id=None, expected_active_profile_revision=None,
        )
    assert lock.read_text(encoding="utf-8") == "other selection"
    assert not (service.service._baseline("cam1", "scene1").parent / "current.txt").exists()


@pytest.mark.parametrize("current_profile_id", ["newer-scene", "original-scene"])
def test_delayed_profile_selection_rejects_changed_id_or_aba_revision_before_mutation(
    service, monkeypatch, current_profile_id,
):
    _job, model = _seed_training(service, monkeypatch, complete=True)
    service.service.approve_profile("cam1", "scene1", actor="approver")
    active_key = (str(service.service.profiles_dir), "cam1")
    pointer = service.service.profiles_dir / "cam1" / "active.json"
    service.active[active_key] = "original-scene"
    pointer.write_text(json.dumps({"profile_id": "original-scene", "selected_at": "first"}), encoding="utf-8")
    store = service.store_type(service.service.profiles_dir)
    observed_id, observed_revision = service.service._active_selection("cam1", store)

    # Another selection has committed. Returning to the same ID is still a
    # different selection intent (ABA); the pointer-byte revision must catch it.
    service.active[active_key] = current_profile_id
    pointer.write_text(json.dumps({"profile_id": current_profile_id, "selected_at": "later"}), encoding="utf-8")
    pointer_before = pointer.read_bytes()
    record_before = service.service._load("cam1", "scene1")
    baseline_marker = service.service._baseline("cam1", "scene1").parent / "current.txt"
    baseline_marker.write_text("scene-existing", encoding="utf-8")
    audit, save = Mock(), Mock()
    monkeypatch.setattr(service.service, "_audit", audit)
    monkeypatch.setattr(service.service, "_save", save)
    registry_reads = service.registry.get_by_version_id.call_count

    with pytest.raises(workflow.SceneWorkflowConflict, match="changed since confirmation"):
        service.service.activate_profile(
            "cam1", "scene1", actor="delayed-selector",
            expected_active_profile_id=observed_id,
            expected_active_profile_revision=observed_revision,
        )
    audit.assert_not_called()
    save.assert_not_called()
    assert service.service._load("cam1", "scene1") == record_before
    assert pointer.read_bytes() == pointer_before and service.active[active_key] == current_profile_id
    assert baseline_marker.read_text(encoding="utf-8") == "scene-existing"
    assert service.registry.get_by_version_id.call_count == registry_reads
    assert model.stage == "candidate"


def test_active_selection_rejects_pointer_changes_during_read_instead_of_mixing_identity_and_revision(
    service, monkeypatch,
):
    pointer = service.service.profiles_dir / "cam1" / "active.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"profile_id":"old-scene"}', encoding="utf-8")
    store = service.store_type(service.service.profiles_dir)

    def read_and_replace(_camera_id):
        pointer.write_text('{"profile_id":"new-scene"}', encoding="utf-8")
        return "old-scene"

    monkeypatch.setattr(store, "get_active_profile_id", read_and_replace)
    with pytest.raises(workflow.SceneWorkflowConflict, match="changed while reading"):
        service.service._active_selection("cam1", store)


def test_broken_or_orphan_workflows_are_visible_as_issues_not_adopted_as_valid_records(service):
    _create(service)
    missing = service.service.root / "cam1" / "orphan"
    missing.mkdir()
    malformed = service.service.root / "cam1" / "broken"
    malformed.mkdir()
    (malformed / "workflow.json").write_text("not-json", encoding="utf-8")
    snapshot = service.service.snapshot("cam1")
    assert [item["workflow_id"] for item in snapshot["workflows"]] == ["scene1"]
    assert {item["workflow_id"] for item in snapshot["issues"]} == {"broken", "orphan"}
    assert all("未授予任何操作权限" in item["message"] for item in snapshot["issues"])
    assert service.active == {}


@pytest.mark.parametrize("damage", ["one-score", "false-count", "nan", "unsorted", "thresholds", "alpha"])
def test_profile_approval_revalidates_serialized_calibration_even_if_status_and_hash_claim_ok(service, monkeypatch, damage):
    _job, model = _seed_training(service, monkeypatch, complete=True)
    path = Path(model.model_path).parent / "calibration.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if damage == "one-score":
        calibration["sorted_scores"] = [0.1]
    elif damage == "false-count":
        calibration["n_samples"] = 51
    elif damage == "nan":
        calibration["sorted_scores"][0] = float("nan")
    elif damage == "unsorted":
        calibration["sorted_scores"][0] = 0.2
    elif damage == "thresholds":
        calibration["high"] = calibration["medium"]
    else:
        calibration["target_fprs"]["high"] = 0
    path.write_text(json.dumps(calibration), encoding="utf-8")
    # The fixture keeps metadata status/hash looking valid deliberately: an
    # older/custom trainer's claim must not replace full calibration evidence.
    with pytest.raises(ValueError, match="scene calibration"):
        service.service.approve_profile("cam1", "scene1", actor="approver")
    assert not (service.service.profiles_dir / "cam1" / "scene1").exists()
    assert service.service._load("cam1", "scene1")["approved"] is False
    assert service.active == {} and model.stage == "candidate"


def test_training_confirmation_accepts_database_naive_datetime_and_same_reviewer(service, monkeypatch):
    job, _model = _seed_training(service, monkeypatch, complete=True)
    job.confirmed_by = "creator"
    # SQLite DateTime may deserialize as naive. This is valid typed database
    # metadata, not the external profile's ISO-8601 approval timestamp.
    job.confirmed_at = datetime(2026, 10, 22, 12, 30)
    assert service.service.approve_profile("cam1", "scene1", actor="creator") == "scene1"


@pytest.mark.parametrize("field,value", [
    ("approved_by", ""), ("approved_by", "   "), ("approved_by", "unknown"),
    ("approved_by", " UNKNOWN "), ("approved_by", True),
    ("approved_at", ""), ("approved_at", "not-an-iso-timestamp"),
    ("approved_at", "2026-10-22T12:30:00"), ("approved_at", True),
])
def test_approval_retry_cannot_adopt_an_existing_profile_without_real_approval_identity(service, monkeypatch, field, value):
    _seed_training(service, monkeypatch, complete=True)
    service.service.approve_profile("cam1", "scene1", actor="original-approver")
    profile = service.store_type(service.service.profiles_dir).load("cam1", "scene1")
    profile.metadata[field] = value
    record = service.service._load("cam1", "scene1")
    record.update(state="training_submitted", approved=False)
    service.service._save(record)
    with pytest.raises(workflow.SceneWorkflowConflict, match="existing profile"):
        service.service.approve_profile("cam1", "scene1", actor="retrying-operator")
    saved = service.service._load("cam1", "scene1")
    assert saved["approved"] is False and saved["state"] == "training_submitted"
    assert service.active == {}
