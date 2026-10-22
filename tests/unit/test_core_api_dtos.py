"""OpenAPI and validation guards for the operational dashboard DTOs."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from argus.config.schema import ArgusConfig, DashboardConfig
from argus.dashboard.api_contracts import (
    AlertIdListRequest,
    CameraModeRequest,
    ReplayClipRequest,
    SceneGuardStatusResponse,
)
from argus.dashboard.app import create_app


def _client() -> TestClient:
    config = ArgusConfig(dashboard=DashboardConfig(go2rtc_enabled=False))
    return TestClient(create_app(config=config))


def _scene_guard_payload() -> dict:
    return {
        "enabled": True,
        "usable": False,
        "state": "blocked",
        "raw_state": "mismatch",
        "reason": "camera pose does not match the approved profile",
        "profile_id": "scene-main-v3",
        "bound_model_version": "model-v3",
        "bound_model_artifact_sha256": "a" * 64,
        "runtime_model_artifact_sha256": "a" * 64,
        "reference_source": "approved_profile",
        "reference_sha256": "b" * 64,
        "reference_error": None,
        "runtime_error": None,
        "check_interval_frames": 15,
        "check_interval_seconds": 1.0,
        "last_checked_frame": 42,
        "quick_match_score": 0.2,
        "quick_shift_px": 18.0,
        "reference_available": True,
        "policy_recovery_count": 1,
        "required_recovery_count": 3,
        "recovery_progress": {
            "current": 1,
            "required": 3,
            "complete": False,
        },
        "metrics": {
            "reference_available": True,
            "good_matches": 8,
            "inlier_count": 3,
            "inlier_ratio": 0.375,
            "shift_px": 18.0,
            "rotation_deg": 4.0,
            "scale_delta": 0.08,
            "estimator": "orb-homography",
        },
    }


def test_scene_guard_dto_exposes_safety_and_recovery_contract():
    status = SceneGuardStatusResponse.model_validate(_scene_guard_payload())

    assert status.usable is False
    assert status.profile_id == "scene-main-v3"
    assert status.bound_model_version == "model-v3"
    assert status.recovery_progress.current == 1
    assert status.recovery_progress.required == 3

    with pytest.raises(ValidationError):
        SceneGuardStatusResponse.model_validate({
            **_scene_guard_payload(),
            "undeclared_runtime_field": True,
        })


def test_mutation_request_models_are_closed_and_do_not_coerce_core_fields():
    assert CameraModeRequest.model_validate({"mode": "active"}).mode == "active"

    with pytest.raises(ValidationError):
        CameraModeRequest.model_validate({"mode": "active", "legacy": True})
    with pytest.raises(ValidationError):
        ReplayClipRequest.model_validate({
            "start_index": "0",
            "end_index": 3,
        })
    with pytest.raises(ValidationError):
        ReplayClipRequest.model_validate({
            "start_index": 4,
            "end_index": 3,
        })
    with pytest.raises(ValidationError):
        AlertIdListRequest.model_validate({"alert_ids": [123]})


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/cameras/cam-01/mode", {"mode": "unsupported"}),
        ("/api/alerts/alert-01/workflow", {"status": "unsupported"}),
        (
            "/api/replay/alert-01/clips",
            {"start_index": "0", "end_index": 2},
        ),
    ],
)
def test_core_mutation_validation_uses_unified_422_envelope(path, body):
    response = _client().post(path, json=body)

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == 42200
    assert payload["msg"] == "请求参数校验失败"
    assert payload["data"]["errors"]
    assert payload["data"]["request_id"] == response.headers["x-request-id"]


def test_openapi_links_core_routes_to_typed_envelopes_and_body_dtos():
    schema = _client().get("/openapi.json").json()
    components = schema["components"]["schemas"]

    assert "CameraDetailResponse" in components
    assert "SceneGuardStatusResponse" in components
    assert "ReplayMetadataResponse" in components
    assert "ReplaySignalsResponse" in components
    assert "AlertTrajectoryResponse" in components

    camera_detail = components["CameraDetailResponse"]
    assert camera_detail["additionalProperties"] is False
    assert camera_detail["properties"]["scene_guard"]
    scene_guard = components["SceneGuardStatusResponse"]
    for field in (
        "usable",
        "state",
        "reason",
        "profile_id",
        "bound_model_version",
        "bound_model_artifact_sha256",
        "recovery_progress",
    ):
        assert field in scene_guard["properties"]

    mode_operation = schema["paths"]["/api/cameras/{camera_id}/mode"]["post"]
    assert "CameraModeRequest" in json.dumps(mode_operation["requestBody"])

    typed_success_routes = (
        ("/api/cameras/{camera_id}/detail/json", "get", "CameraDetailResponse"),
        ("/api/cameras/{camera_id}/start", "post", "CameraLifecycleResponse"),
        ("/api/cameras/{camera_id}/stop", "post", "CameraLifecycleResponse"),
        ("/api/alerts/{alert_id}/workflow", "post", "AlertWorkflowTransitionResponse"),
        ("/api/replay/{alert_id}/metadata", "get", "ReplayMetadataResponse"),
        ("/api/replay/{alert_id}/signals", "get", "ReplaySignalsResponse"),
        ("/api/physics/{alert_id}/trajectory", "get", "AlertTrajectoryResponse"),
    )
    for path, method, payload_model in typed_success_routes:
        response_schema = schema["paths"][path][method]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]
        rendered = json.dumps(response_schema)
        assert "ApiEnvelope" in rendered
        assert payload_model in rendered
