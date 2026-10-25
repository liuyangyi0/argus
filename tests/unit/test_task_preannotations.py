"""Data-only alignment: no inference, encoding, source video or external calls."""

from copy import deepcopy

import pytest

from argus.core.task_preannotations import propose_review_annotations


def fixtures():
    binding = {"task_id": "task-1", "camera_id": "cam", "task_revision": 2,
               "task_config_sha256": "a" * 64, "template_id": "object_detection.v1"}
    identity = {"camera_id": "cam", "source_session_id": "session", "source_epoch": 2,
                "frame_index": 8, "source_frame_index": None, "pts": 40,
                "time_base_num": 1, "time_base_den": 120, "capture_wall_ts": 100.5, "capture_monotonic_ts": 5.2,
                "frame_index_provenance": "capture_sequence", "timestamp_provenance": "local_capture_clock"}
    envelope = {"schema": "argus.task-review-envelope.v1", "binding": binding, "target": "cvat",
                "workflow_id": "flow", "workflow_revision": 3, "report_sha256": "b" * 64,
                "samples": [{"sample_id": "image-1", "image_sha256": "c" * 64,
                             "identity": identity, "image_size": [200, 100]}]}
    result = {"schema_version": "argus.task-shadow-result.v1", **binding,
              "production_allowed": False, "source": deepcopy(identity), "bundle_id": "d" * 64,
              "manifest_sha256": "e" * 64, "models": {"detector": {"model_id": "model-1"}},
              "request_id": 12, "data": {"state": "observed", "detections": [
                  {"label": "person", "confidence": 0.9, "bbox_xyxy": [10, 20, 100, 80]},
              ]}}
    return envelope, result


def test_exact_identity_produces_unapproved_proposal_without_mutating_report():
    envelope, result = fixtures()
    before = deepcopy(envelope)
    proposal = propose_review_annotations(envelope, [result])["results"][0]
    assert envelope == before
    assert proposal["origin"] == "auto"
    assert "decision" not in proposal
    assert proposal["annotation"]["objects"][0]["bbox_xyxy"] == [0.05, 0.2, 0.5, 0.8]
    assert proposal["image_sha256"] == "c" * 64
    assert proposal["source"]["alignment"] == "exact_source_frame_identity.v1"


@pytest.mark.parametrize("field,value", [("source_epoch", 3), ("pts", 41), ("frame_index", 9),
                                         ("source_session_id", "other"), ("time_base_den", 60),
                                         ("capture_wall_ts", 100.6)])
def test_close_timestamps_or_reused_frame_numbers_cannot_attach_wrong_annotations(field, value):
    envelope, result = fixtures()
    result["source"][field] = value
    assert propose_review_annotations(envelope, [result])["results"] == []


def test_pose_proposal_round_trips_the_workflow_annotation_contract():
    from argus.storage.task_workflows import TaskWorkflowService

    envelope, result = fixtures()
    envelope["binding"]["template_id"] = result["template_id"] = "pose_estimation.v1"
    result["data"]["detections"][0]["keypoints"] = [
        {"index": index, "x": 50, "y": 40, "confidence": 0.9} for index in range(17)
    ]
    result["data"]["detections"][0]["keypoints"][0].update(confidence=0.2, x=-1)
    annotation = propose_review_annotations(envelope, [result])["results"][0]["annotation"]
    assert annotation["objects"][0]["keypoint_schema"] == "coco17.v1"
    assert annotation["objects"][0]["keypoints"][0] == {"index": 0, "x": 0.0, "y": 0.0, "visibility": 0}
    assert TaskWorkflowService._annotation(annotation, "pose_estimation.v1") == annotation


@pytest.mark.parametrize("change", ["task_revision", "warming_up", "bad_geometry"])
def test_task_revision_status_and_geometry_are_fail_closed(change):
    envelope, result = fixtures()
    if change == "task_revision":
        result[change] = 1
    elif change == "warming_up":
        result["data"]["state"] = "warming_up"
    else:
        result["data"]["detections"][0]["bbox_xyxy"][2] = 250
        with pytest.raises(ValueError):
            propose_review_annotations(envelope, [result])
        return
    assert propose_review_annotations(envelope, [result])["results"] == []
