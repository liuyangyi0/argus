"""Contract tests for the dashboard envelope and pure-ASGI middleware."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, DashboardConfig
from argus.dashboard.api_contracts import CameraModeResponse
from argus.dashboard.api_response import (
    API_MEDIA_PATHS,
    ApiEnvelope,
    JsonApiEnvelopeMiddleware,
    api_success,
    api_typed_success,
    install_api_envelope_openapi,
)
from argus.dashboard.app import create_app
from argus.dashboard.auth import RateLimitMiddleware, RequestIdMiddleware


def _dashboard_config() -> ArgusConfig:
    return ArgusConfig(dashboard=DashboardConfig(go2rtc_enabled=False))


def test_api_envelope_is_a_strict_typed_contract():
    envelope = ApiEnvelope[dict[str, str]].model_validate({
        "code": 0,
        "msg": "ok",
        "data": {"status": "ready"},
    })

    assert envelope.data == {"status": "ready"}
    with pytest.raises(ValueError):
        ApiEnvelope[dict].model_validate({
            "code": 0,
            "msg": "ok",
            "data": {},
            "legacy": True,
        })
    with pytest.raises(ValueError):
        ApiEnvelope[dict].model_validate({"code": 0, "msg": "ok"})


def test_typed_success_validates_payload_and_preserves_response_headers():
    response = api_typed_success(
        {
            "camera_id": "cam-01",
            "previous_mode": None,
            "pipeline_mode": "active",
        },
        CameraModeResponse,
        msg="mode changed",
        status_code=202,
        headers={"X-Argus-Contract": "validated"},
    )

    assert response.status_code == 202
    assert response.headers["x-argus-contract"] == "validated"
    assert json.loads(response.body) == {
        "code": 0,
        "msg": "mode changed",
        "data": {
            "camera_id": "cam-01",
            "previous_mode": None,
            "pipeline_mode": "active",
        },
    }

    with pytest.raises(ValueError):
        api_typed_success(
            {
                "camera_id": "cam-01",
                "previous_mode": None,
                "pipeline_mode": "active",
                "undeclared": True,
            },
            CameraModeResponse,
        )


def test_typed_success_contract_drift_returns_unified_500_with_request_id():
    app = create_app(config=_dashboard_config())

    class DriftedRecordingStore:
        @staticmethod
        def load_metadata(_alert_id):
            return {
                "alert_id": "alert-01",
                "camera_id": "cam-01",
                "severity": "medium",
                "trigger_timestamp": 10.0,
                "trigger_frame_index": 1,
                "start_timestamp": 9.0,
                "end_timestamp": 11.0,
                "frame_count": 3,
                "fps": 15,
                "linked_alert_id": None,
                "status": "complete",
                "video_file": "recording.mp4",
                "width": 1920,
                "height": 1080,
                "codec": "h264",
                "created_at": "2026-07-14T12:00:00+00:00",
                "undeclared_runtime_field": True,
            }

    app.state.recording_store = DriftedRecordingStore()

    response = TestClient(app, raise_server_exceptions=False).get(
        "/api/replay/alert-01/metadata"
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": 50000,
        "msg": "内部错误",
        "data": {"request_id": response.headers["x-request-id"]},
    }


def test_request_id_and_rate_limit_use_the_standard_envelope():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, max_requests_per_minute=1)
    app.add_middleware(RequestIdMiddleware)

    @app.post("/mutation")
    async def mutation():
        return {"ok": True}

    client = TestClient(app)
    headers = {"x-request-id": "contract-123"}

    first = client.post("/mutation", headers=headers)
    limited = client.post("/mutation", headers=headers)

    assert first.headers["x-request-id"] == "contract-123"
    assert limited.status_code == 429
    assert limited.json() == {
        "code": 42900,
        "msg": "Rate limit exceeded",
        "data": {"request_id": "contract-123"},
    }
    assert limited.headers["retry-after"] == "60"


def test_unsafe_caller_request_id_is_replaced():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/probe")
    async def probe():
        return {"ok": True}

    response = TestClient(app).get(
        "/probe",
        headers={"x-request-id": "contains spaces"},
    )

    assert response.headers["x-request-id"].startswith("req_")
    assert response.headers["x-request-id"] != "contains spaces"


def test_http_exception_and_validation_errors_use_the_standard_envelope():
    app = create_app(config=_dashboard_config())
    client = TestClient(app)

    missing = client.get("/api/not-a-route")
    invalid = client.post("/api/sensors/signal", json={})

    assert missing.status_code == 404
    assert missing.json()["code"] == 40400
    assert missing.json()["data"]["request_id"] == missing.headers["x-request-id"]

    assert invalid.status_code == 422
    invalid_body = invalid.json()
    assert invalid_body["code"] == 42200
    assert invalid_body["msg"] == "请求参数校验失败"
    assert invalid_body["data"]["errors"]
    assert invalid_body["data"]["request_id"] == invalid.headers["x-request-id"]


@pytest.mark.asyncio
async def test_unhandled_exception_is_sanitized_and_correlated():
    app = create_app(config=_dashboard_config())
    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/boom",
        "raw_path": b"/boom",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "state": {"request_id": "req_contract"},
        "app": app,
    }
    request = Request(scope)
    handler = app.exception_handlers[Exception]

    response = await handler(request, RuntimeError("secret failure detail"))
    body = json.loads(response.body)

    assert response.status_code == 500
    assert body == {
        "code": 50000,
        "msg": "内部错误",
        "data": {"request_id": "req_contract"},
    }
    assert response.headers["x-request-id"] == "req_contract"
    assert "secret failure detail" not in response.body.decode("utf-8")


def test_streaming_success_is_not_wrapped_in_json():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/stream")
    async def stream():
        return StreamingResponse(iter([b"frame"]), media_type="multipart/x-mixed-replace")

    response = TestClient(app).get("/stream")

    assert response.status_code == 200
    assert response.content == b"frame"
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert response.headers["x-request-id"].startswith("req_")


def test_raw_legacy_json_is_normalized_once_at_api_boundary():
    app = FastAPI()
    app.add_middleware(JsonApiEnvelopeMiddleware)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/api/raw-object")
    async def raw_object():
        return {"ready": True}

    @app.get("/api/raw-list")
    async def raw_list():
        return ["camera-a", "camera-b"]

    @app.get("/api/already-enveloped")
    async def already_enveloped():
        return api_success({"ready": True})

    @app.post("/api/raw-created", status_code=201)
    async def raw_created():
        return {"created": True}

    client = TestClient(app)

    assert client.get("/api/raw-object").json() == {
        "code": 0,
        "msg": "ok",
        "data": {"ready": True},
    }
    assert client.get("/api/raw-list").json() == {
        "code": 0,
        "msg": "ok",
        "data": ["camera-a", "camera-b"],
    }
    assert client.get("/api/already-enveloped").json() == {
        "code": 0,
        "msg": "ok",
        "data": {"ready": True},
    }
    created = client.post("/api/raw-created")
    assert created.status_code == 201
    assert created.json()["data"] == {"created": True}


def test_media_range_attachment_and_openapi_are_not_buffered_or_wrapped():
    app = FastAPI()
    app.add_middleware(JsonApiEnvelopeMiddleware)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/api/image")
    async def image():
        return Response(content=b"jpeg", media_type="image/jpeg")

    @app.get("/api/range")
    async def range_response():
        return Response(
            content=b"partial",
            status_code=206,
            media_type="video/mp4",
            headers={"Content-Range": "bytes 0-6/12"},
        )

    @app.get("/api/export")
    async def export():
        return JSONResponse(
            {"rows": [1]},
            headers={"Content-Disposition": 'attachment; filename="rows.json"'},
        )

    client = TestClient(app)
    found = client.get("/api/image")
    ranged = client.get("/api/range", headers={"range": "bytes=0-6"})
    attachment = client.get("/api/export")
    schema = client.get("/openapi.json")

    assert found.content == b"jpeg"
    assert found.headers["content-type"].startswith("image/jpeg")
    assert ranged.status_code == 206
    assert ranged.content == b"partial"
    assert ranged.headers["content-range"] == "bytes 0-6/12"
    assert attachment.json() == {"rows": [1]}
    assert schema.status_code == 200
    assert "openapi" in schema.json()


def test_openapi_documents_json_envelopes_and_media_exceptions():
    app = create_app(config=_dashboard_config())
    schema = TestClient(app).get("/openapi.json").json()

    envelope = schema["components"]["schemas"]["ApiEnvelope"]
    assert envelope["required"] == ["code", "msg", "data"]
    assert envelope["additionalProperties"] is False

    camera_list = schema["paths"]["/api/cameras/json"]["get"]
    assert camera_list["x-argus-json-envelope"] is True
    camera_schema = camera_list["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert camera_schema["x-argus-envelope-schema"] is True
    assert camera_schema["$ref"].startswith(
        "#/components/schemas/ApiEnvelope_CameraListResponse_"
    )
    assert "allOf" not in camera_schema

    replay_video = schema["paths"]["/api/replay/{alert_id}/video"]["get"]
    assert replay_video["x-argus-json-envelope"] is False
    video_content = replay_video["responses"]["200"]["content"]
    assert video_content == {
        "video/mp4": {"schema": {"type": "string", "format": "binary"}},
    }

    for path, path_item in schema["paths"].items():
        if not path.startswith("/api/"):
            continue
        for method in ("get", "post", "put", "patch", "delete"):
            operation = path_item.get(method)
            if operation is None:
                continue
            assert operation["x-argus-json-envelope"] is (path not in API_MEDIA_PATHS)


def test_core_model_mutations_use_closed_pydantic_request_contracts():
    app = create_app(config=_dashboard_config())
    client = TestClient(app)
    schema = app.openapi()

    expected_models = {
        ("/api/models/batch-inference", "post"): "ModelBatchInferenceRequest",
        ("/api/models/{version_id}/promote", "post"): "ModelPromoteRequest",
        ("/api/models/{version_id}/retire", "post"): "ModelRetireRequest",
        ("/api/models/{version_id}/reexport", "post"): "ModelReexportRequest",
        ("/api/models/ab-live-compare", "post"): "ModelAbLiveCompareRequest",
        ("/api/models/backbone/upgrade", "post"): "BackboneUpgradeRequest",
    }
    components = schema["components"]["schemas"]
    for (path, method), model_name in expected_models.items():
        request_schema = schema["paths"][path][method]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        refs = json.dumps(request_schema)
        assert f"#/components/schemas/{model_name}" in refs
        assert components[model_name]["additionalProperties"] is False

    invalid = client.post(
        "/api/models/unused/promote",
        json={"target_stage": "shadow", "undeclared": True},
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == 42200
    assert invalid.json()["data"]["request_id"] == invalid.headers["x-request-id"]


def test_openapi_does_not_wrap_an_explicit_generic_envelope_twice():
    app = FastAPI()

    @app.get(
        "/api/typed",
        response_model=ApiEnvelope[dict[str, str]],
    )
    async def typed_response():
        return ApiEnvelope(code=0, msg="ok", data={"state": "ready"})

    app.add_middleware(JsonApiEnvelopeMiddleware)
    install_api_envelope_openapi(app)

    schema = app.openapi()
    response_schema = schema["paths"]["/api/typed"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert response_schema["x-argus-envelope-schema"] is True
    assert response_schema["$ref"].startswith(
        "#/components/schemas/ApiEnvelope_"
    )
    assert "allOf" not in response_schema
