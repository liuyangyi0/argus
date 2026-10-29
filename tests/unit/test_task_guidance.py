"""Navigation snapshots only: no models, files, devices or computation."""

from copy import deepcopy

import pytest

from argus.core.task_guidance import task_guidance
from argus.dashboard.analysis_task_contracts import TaskGuidanceResponse


def current():
    return {
        "task_id": "one", "authority": "managed", "template_id": "object_detection.v1",
        "current_revision": 3, "lifecycle": "active",
        "source": {"state": "running", "running": True, "connected": True},
        "runtime": {"state": "running", "running": True, "applied_task_revision": 3,
                    "bundle_id": "bundle-one", "mode": "shadow", "deployment_revision": 4,
                    "deployment_operation_id": "operation-four"},
        "deployment": {"authority": "managed", "desired_state": "running", "runtime_status": "running",
                       "deployment_revision": 4, "applied_deployment_revision": 4,
                       "desired_bundle_id": "bundle-one", "applied_bundle_id": "bundle-one",
                       "current_operation_id": "operation-four"},
    }


def test_current_observation_is_not_readiness_or_authorization_and_never_changes_input():
    task = current()
    before = deepcopy(task)
    result = task_guidance(task)
    TaskGuidanceResponse.model_validate(result)
    assert result["state"] == "observed_running"
    assert result["target"] == "events"
    assert result["is_authorization"] is False
    assert result["production_scope"] == "unsupported"
    assert result["training_requirement"] == "optional"
    assert task == before


@pytest.mark.parametrize(("section", "field", "value"), [
    ("runtime", "bundle_id", "other-task-bundle"),
    ("runtime", "deployment_revision", 2),
    ("runtime", "deployment_operation_id", "old-operation"),
    ("runtime", "applied_task_revision", 2),
    ("runtime", "applied_task_revision", True),
    ("runtime", "mode", "production"),
    ("runtime", "running", False),
    ("runtime", "state", "unknown"),
    ("deployment", "applied_bundle_id", "old-bundle"),
    ("deployment", "applied_deployment_revision", 3),
    ("deployment", "applied_deployment_revision", True),
    ("deployment", "runtime_status", "unknown"),
    ("deployment", "failure_code", "late-operation-failure"),
    ("source", "connected", None),
    ("source", "connected", False),
    ("source", "state", "unknown"),
    ("source", "running", False),
])
def test_partial_or_old_observations_never_become_running(section, field, value):
    task = current()
    task[section][field] = value
    result = task_guidance(task)
    assert result["state"] != "observed_running"
    assert result["is_authorization"] is False


@pytest.mark.parametrize(("state", "code", "target"), [
    ("cleanup_pending", "CLEANUP_REQUIRED", "deployment"),
    ("cleanup_failed", "CLEANUP_REQUIRED", "deployment"),
    ("draining", "CLEANUP_REQUIRED", "deployment"),
    ("sampling_insufficient", "RUNTIME_ATTENTION", "deployment"),
    ("runtime_dead", "RUNTIME_ATTENTION", "deployment"),
])
def test_faults_take_priority_over_otherwise_matching_snapshots(state, code, target):
    task = current()
    task["runtime"]["state"] = state
    result = task_guidance(task)
    assert (result["state"], result["reason_code"], result["target"]) == ("attention", code, target)


@pytest.mark.parametrize("state", ["reconcile_required", "recovering", "pending", "preparing", "waiting_scene"])
def test_pending_deployment_is_not_applied_even_if_old_worker_runs(state):
    task = current()
    task["deployment"]["runtime_status"] = state
    assert task_guidance(task)["state"] != "observed_running"


def test_events_failure_does_not_claim_no_events_is_normal():
    task = current()
    task["runtime"]["event_error"] = "EVENT_QUEUE_FAILED"
    assert task_guidance(task)["reason_code"] == "EVENT_EVIDENCE_UNAVAILABLE"


@pytest.mark.parametrize(("template", "training", "target"), [
    ("object_detection.v1", "optional", "models"),
    ("ppe_compliance_dual.v1", "optional", "models"),
    ("pose_estimation.v1", "optional", "models"),
    ("fast_motion.v1", "not_applicable", "configuration"),
])
def test_new_task_does_not_require_unnecessary_training(template, training, target):
    task = current()
    task.update(template_id=template, runtime={"state": "stopped", "running": False},
                deployment={"authority": "managed", "desired_state": "stopped"})
    result = task_guidance(task)
    assert result["state"] == "setup"
    assert result["training_requirement"] == training
    assert result["target"] == target


def test_stop_intent_does_not_hide_live_worker():
    task = current()
    task["deployment"]["desired_state"] = "stopped"
    assert task_guidance(task)["reason_code"] == "UNEXPECTED_RUNTIME"


@pytest.mark.parametrize("projection", ["not_synced", "unbound", "blocked", "stale", "authorized"])
def test_legacy_binding_is_not_production_or_scene_approval(projection):
    task = current()
    task.update(authority="legacy", template_id="legacy_anomaly.v1")
    task["deployment"] = {"authority": "legacy", "projection_status": projection}
    result = task_guidance(task)
    assert result["training_requirement"] == "legacy_workflow"
    assert result["is_authorization"] is False
    assert result["production_scope"] == "existing_legacy_gates"
    assert result["target"] == ("legacy_runtime" if projection == "authorized" else "legacy_baseline")


def test_missing_runtime_and_source_are_unknown_not_stopped_or_normal():
    task = current()
    task.update(runtime={}, source={})
    assert task_guidance(task)["state"] == "unknown"


@pytest.mark.parametrize("runtime", [{}, {"state": "unknown"}, {"state": "stopped"}])
def test_stop_intent_never_fills_missing_runtime_observations(runtime):
    task = current()
    task["deployment"]["desired_state"] = "stopped"
    task["runtime"] = runtime
    assert task_guidance(task)["reason_code"] == "STOP_UNVERIFIED"


def test_stop_intent_detects_real_managed_running_shape_without_boolean():
    task = current()
    task["deployment"]["desired_state"] = "stopped"
    del task["runtime"]["running"]
    assert task_guidance(task)["reason_code"] == "UNEXPECTED_RUNTIME"


def test_missing_source_running_observation_is_unknown_not_confirmed_unavailable():
    task = current()
    del task["source"]["running"]
    result = task_guidance(task)
    assert result["state"] == "unknown"
    assert result["reason_code"] == "SOURCE_UNVERIFIED"


@pytest.mark.parametrize("revision", [True, 3.0, "3", None])
def test_task_revision_must_itself_be_a_verified_integer(revision):
    task = current()
    task["current_revision"] = revision
    if revision is True:
        task["runtime"]["applied_task_revision"] = 1
    assert task_guidance(task)["state"] != "observed_running"


def test_managed_runtime_does_not_require_an_unprovided_running_boolean():
    task = current()
    del task["runtime"]["running"]
    assert task_guidance(task)["state"] == "observed_running"


def test_archived_task_has_no_action():
    task = current()
    task["lifecycle"] = "archived"
    result = task_guidance(task)
    assert result["state"] == "archived"
    assert result["target"] == "none"


def test_untrusted_fault_strings_do_not_leak_into_guidance():
    task = current()
    task["deployment"]["failure_code"] = "rtsp://private:password@private/path"
    result = task_guidance(task)
    assert "password" not in str(result)
    assert result["reason_code"] == "DEPLOYMENT_FAILED"
