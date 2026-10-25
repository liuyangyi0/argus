"""Unified API response helpers — enforces consistent JSON envelope for all endpoints.

Every JSON endpoint should use these helpers instead of raw JSONResponse:

    from argus.dashboard.api_response import api_success, api_error, api_paginated

Success: {"code": 0, "msg": "ok", "data": {...}}
Error:   {"code": 40400, "msg": "告警不存在", "data": null}
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from fastapi import FastAPI


T = TypeVar("T")


class ApiEnvelope(BaseModel, Generic[T]):
    """Typed representation of the public JSON response contract.

    File/video/streaming responses are deliberately outside this contract;
    their *error* responses still use the envelope through the application
    exception handlers.
    """

    model_config = ConfigDict(extra="forbid")

    code: int
    msg: str
    data: T | None


# ── Error codes: <HTTP status> * 100 + sequence ──
# code=0 always means success.

class ErrorCode:
    """Numeric error codes.  Convention: HTTP_STATUS * 100 + seq."""

    SUCCESS = 0
    VALIDATION_ERROR = 40000
    CONFIRMATION_REQUIRED = 40001
    AUTH_REQUIRED = 40100
    FORBIDDEN = 40300
    NOT_FOUND = 40400
    METHOD_NOT_ALLOWED = 40500
    CONFLICT = 40900
    UNPROCESSABLE_ENTITY = 42200
    RATE_LIMITED = 42900
    INTERNAL_ERROR = 50000
    SERVICE_UNAVAILABLE = 50300


_STATUS_ERROR_CODES: dict[int, int] = {
    400: ErrorCode.VALIDATION_ERROR,
    401: ErrorCode.AUTH_REQUIRED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    409: ErrorCode.CONFLICT,
    422: ErrorCode.UNPROCESSABLE_ENTITY,
    429: ErrorCode.RATE_LIMITED,
    500: ErrorCode.INTERNAL_ERROR,
    503: ErrorCode.SERVICE_UNAVAILABLE,
}


# Successful bodies for these endpoints are deliberately not JSON API
# documents.  Keeping the list next to the OpenAPI normalizer makes the public
# contract explicit instead of relying on FastAPI's default JSON response class
# for functions which return a FileResponse/StreamingResponse at runtime.
API_MEDIA_RESPONSES: dict[str, tuple[str, ...]] = {
    "/api/analysis-tasks/{task_id}/workflows/{workflow_id}/samples/{sample_id}/image": ("image/png",),
    "/api/alerts/{alert_id}/evidence.zip": ("application/zip",),
    "/api/alerts/{alert_id}/image/{image_type}": ("image/jpeg",),
    "/api/alerts/export-csv": ("text/csv",),
    "/api/alerts/export-pdf": ("text/html",),
    "/api/baseline/{camera_id}/{version}/images/{filename}/content": (
        "application/octet-stream",
    ),
    "/api/cameras/{camera_id}/heatmap-stream": ("multipart/x-mixed-replace",),
    "/api/cameras/{camera_id}/snapshot": ("image/jpeg",),
    "/api/cameras/{camera_id}/stream": ("multipart/x-mixed-replace",),
    "/api/labeling/{entry_id}/image": ("image/jpeg",),
    "/api/recordings/{camera_id}/segment/{filename}/video": ("video/mp4",),
    "/api/replay/{alert_id}/frame/{index}": ("image/jpeg",),
    "/api/replay/{alert_id}/heatmap/{index}": ("image/jpeg",),
    "/api/replay/{alert_id}/video": ("video/mp4",),
    "/api/reports/compliance": ("application/pdf", "text/csv"),
    "/api/system/metrics": ("text/plain",),
    "/api/zones/snapshot/{camera_id}": ("image/jpeg",),
}
API_MEDIA_PATHS = frozenset(API_MEDIA_RESPONSES)


def error_code_for_status(status_code: int) -> int:
    """Return the stable business error code for an HTTP status."""
    return _STATUS_ERROR_CODES.get(status_code, status_code * 100)


class JsonApiEnvelopeMiddleware:
    """Enforce the public envelope at the ASGI boundary for JSON APIs.

    Legacy dashboard handlers still returning a raw ``dict`` or ``list`` are
    normalized here so the wire contract cannot drift while those handlers are
    migrated to typed DTOs.  The middleware buffers *only* ``application/json``
    responses.  Video, images, files, HTML, MJPEG, SSE and all other streaming
    content types are passed through without buffering.

    A ``Content-Disposition: attachment`` response is also always passed
    through, even if its declared media type happens to be JSON.
    """

    def __init__(self, app: ASGIApp, *, api_prefix: str = "/api") -> None:
        self.app = app
        self._api_prefix = api_prefix

    @staticmethod
    def _header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> bytes:
        name = name.lower()
        for key, value in headers:
            if key.lower() == name:
                return value.lower()
        return b""

    @classmethod
    def _is_json_content_type(cls, headers: list[tuple[bytes, bytes]]) -> bool:
        content_type = cls._header_value(headers, b"content-type")
        media_type = content_type.split(b";", 1)[0].strip()
        disposition = cls._header_value(headers, b"content-disposition")
        return media_type == b"application/json" and b"attachment" not in disposition

    @staticmethod
    def _is_envelope(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        return (
            set(payload) == {"code", "msg", "data"}
            and isinstance(payload.get("code"), int)
            and isinstance(payload.get("msg"), str)
        )

    @staticmethod
    def _request_id(scope: Scope) -> str | None:
        state = scope.get("state")
        if not isinstance(state, dict):
            return None
        request_id = state.get("request_id")
        return request_id if isinstance(request_id, str) else None

    @staticmethod
    def _error_message(status_code: int, payload: object) -> str:
        if isinstance(payload, dict):
            for key in ("msg", "detail", "error", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value
        return {
            400: "请求无效",
            401: "Authentication required",
            403: "权限不足",
            404: "资源不存在",
            405: "请求方法不支持",
            409: "资源状态冲突",
            422: "请求参数校验失败",
            429: "请求过于频繁",
            500: "内部错误",
            503: "服务不可用",
        }.get(status_code, "请求失败")

    @classmethod
    def _normalized_payload(
        cls,
        scope: Scope,
        status_code: int,
        payload: object,
    ) -> object:
        if cls._is_envelope(payload):
            return payload
        if status_code < 400:
            return {"code": ErrorCode.SUCCESS, "msg": "ok", "data": payload}

        request_id = cls._request_id(scope)
        details: object | None = payload
        if isinstance(payload, dict) and set(payload) == {"detail"}:
            details = payload["detail"]
        data: dict[str, object | None] = {"request_id": request_id}
        if details not in (None, b"", "", {}):
            data["details"] = details
        return {
            "code": error_code_for_status(status_code),
            "msg": cls._error_message(status_code, payload),
            "data": data,
        }

    @staticmethod
    def _replace_content_headers(
        headers: list[tuple[bytes, bytes]],
        content_length: int,
    ) -> list[tuple[bytes, bytes]]:
        kept = [
            (key, value)
            for key, value in headers
            if key.lower() not in {b"content-length", b"content-type"}
        ]
        kept.extend([
            (b"content-type", b"application/json"),
            (b"content-length", str(content_length).encode("ascii")),
        ])
        return kept

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or not str(scope.get("path", "")).startswith(f"{self._api_prefix}/")
            or scope.get("method") == "HEAD"
        ):
            await self.app(scope, receive, send)
            return

        response_start: Message | None = None
        buffered_body: list[bytes] = []
        normalize = False

        async def send_with_envelope(message: Message) -> None:
            nonlocal response_start, normalize
            if message["type"] == "http.response.start":
                response_start = message
                headers = list(message.get("headers", []))
                normalize = self._is_json_content_type(headers)
                if not normalize:
                    await send(message)
                return

            if message["type"] != "http.response.body" or not normalize:
                await send(message)
                return

            buffered_body.append(message.get("body", b""))
            if message.get("more_body", False):
                return

            assert response_start is not None
            raw_body = b"".join(buffered_body)
            try:
                payload: object = json.loads(raw_body) if raw_body else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = raw_body.decode("utf-8", errors="replace")
            normalized = self._normalized_payload(
                scope,
                int(response_start["status"]),
                payload,
            )
            body = json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            start = dict(response_start)
            start["headers"] = self._replace_content_headers(
                list(response_start.get("headers", [])),
                len(body),
            )
            await send(start)
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, send_with_envelope)


def install_api_envelope_openapi(app: FastAPI) -> None:
    """Make OpenAPI describe the envelope enforced at the ASGI boundary.

    FastAPI cannot infer a response model from handlers returning
    ``JSONResponse``.  This post-processor preserves each inferred payload
    schema under ``data`` and annotates the explicitly enumerated media routes
    as envelope exceptions.  It is intentionally installed after all routers.
    """

    original_openapi = app.openapi

    def is_envelope_schema(
        candidate: object,
        components: dict[str, Any],
        *,
        seen: set[str] | None = None,
    ) -> bool:
        if not isinstance(candidate, dict):
            return False
        if candidate.get("x-argus-envelope-schema") is True:
            return True

        ref = candidate.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            name = ref.rsplit("/", 1)[-1]
            visited = set() if seen is None else set(seen)
            if name in visited:
                return False
            visited.add(name)
            return is_envelope_schema(
                components.get(name),
                components,
                seen=visited,
            )

        properties = candidate.get("properties")
        required = candidate.get("required")
        return (
            isinstance(properties, dict)
            and {"code", "msg", "data"}.issubset(properties)
            and isinstance(required, list)
            and {"code", "msg", "data"}.issubset(required)
        )

    def enveloped_openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema

        schema = original_openapi()
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        components.setdefault("ApiEnvelope", {
            "type": "object",
            "additionalProperties": False,
            "required": ["code", "msg", "data"],
            "properties": {
                "code": {"type": "integer"},
                "msg": {"type": "string"},
                "data": {},
            },
        })

        for path, path_item in schema.get("paths", {}).items():
            if not path.startswith("/api/") or not isinstance(path_item, dict):
                continue
            media_types = API_MEDIA_RESPONSES.get(path)
            is_media = media_types is not None
            for method, operation in path_item.items():
                if method.lower() not in {
                    "get", "post", "put", "patch", "delete", "options", "head",
                } or not isinstance(operation, dict):
                    continue
                operation["x-argus-json-envelope"] = not is_media
                for status_code, response in operation.get("responses", {}).items():
                    if not isinstance(response, dict):
                        continue
                    if is_media and str(status_code).startswith("2"):
                        response["content"] = {
                            media_type: {
                                "schema": {"type": "string", "format": "binary"},
                            }
                            for media_type in media_types
                        }
                        continue
                    json_content = response.get("content", {}).get("application/json")
                    if not isinstance(json_content, dict):
                        continue
                    payload_schema = json_content.get("schema") or {}
                    if is_envelope_schema(payload_schema, components):
                        if isinstance(payload_schema, dict):
                            payload_schema["x-argus-envelope-schema"] = True
                        continue
                    json_content["schema"] = {
                        "x-argus-envelope-schema": True,
                        "allOf": [
                            {"$ref": "#/components/schemas/ApiEnvelope"},
                            {
                                "type": "object",
                                "properties": {
                                    "data": {
                                        "anyOf": [payload_schema, {"type": "null"}],
                                    },
                                },
                            },
                        ],
                    }

        app.openapi_schema = schema
        return schema

    app.openapi = enveloped_openapi


def api_success(
    data: Any = None,
    msg: str = "ok",
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return a successful JSON envelope: {code: 0, msg, data}."""
    body: dict[str, Any] = {"code": 0, "msg": msg, "data": data}
    return JSONResponse(body, status_code=status_code, headers=headers)


def api_typed_success(
    data: Any,
    data_model: type[BaseModel],
    msg: str = "ok",
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Validate and serialize a successful response against its public DTO.

    Returning a ``JSONResponse`` makes FastAPI skip its decorator-level
    ``response_model`` validation.  Core operational routes use this helper
    so their payload and the complete generic envelope are validated before
    bytes are emitted, while retaining the status/header behaviour of
    :func:`api_success`.
    """
    envelope_model = ApiEnvelope[data_model]
    envelope = envelope_model.model_validate({
        "code": ErrorCode.SUCCESS,
        "msg": msg,
        # Runtime snapshots legitimately contain tuples, enums and dataclasses.
        # Validate the same JSON-compatible value that FastAPI would put on the
        # wire, rather than rejecting safe Python representations before
        # response serialization.
        "data": jsonable_encoder(data),
    })
    return JSONResponse(
        envelope.model_dump(mode="json"),
        status_code=status_code,
        headers=headers,
    )


def api_error(
    code: int,
    msg: str,
    status_code: int = 400,
    data: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return an error JSON envelope: {code: >0, msg, data}."""
    return JSONResponse(
        {"code": code, "msg": msg, "data": data},
        status_code=status_code,
        headers=headers,
    )


def api_paginated(
    key: str,
    items: list,
    total: int,
    page: int,
    page_size: int,
) -> JSONResponse:
    """Return a paginated list: {code: 0, msg, data: {<key>: [...], total, page, page_size}}."""
    return api_success(data={
        key: items,
        "total": total,
        "page": page,
        "page_size": page_size,
    })


# ── Convenience shortcuts for common errors ──

def api_not_found(msg: str = "资源不存在") -> JSONResponse:
    return api_error(ErrorCode.NOT_FOUND, msg, 404)


def api_unavailable(msg: str = "服务不可用") -> JSONResponse:
    return api_error(ErrorCode.SERVICE_UNAVAILABLE, msg, 503)


def api_forbidden(msg: str = "权限不足") -> JSONResponse:
    return api_error(ErrorCode.FORBIDDEN, msg, 403)


def api_validation_error(msg: str) -> JSONResponse:
    return api_error(ErrorCode.VALIDATION_ERROR, msg, 400)


def api_conflict(msg: str, data: Any = None) -> JSONResponse:
    return api_error(ErrorCode.CONFLICT, msg, 409, data=data)


def api_internal_error(msg: str = "内部错误") -> JSONResponse:
    return api_error(ErrorCode.INTERNAL_ERROR, msg, 500)
