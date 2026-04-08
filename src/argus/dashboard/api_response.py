"""Unified API response helpers — enforces consistent JSON envelope for all endpoints.

Every JSON endpoint should use these helpers instead of raw JSONResponse:

    from argus.dashboard.api_response import api_success, api_error, api_paginated

Success: {"code": 0, "msg": "ok", "data": {...}}
Error:   {"code": 40400, "msg": "告警不存在", "data": null}
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


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
    CONFLICT = 40900
    INTERNAL_ERROR = 50000
    SERVICE_UNAVAILABLE = 50300


def api_success(
    data: dict[str, Any] | list | None = None,
    msg: str = "ok",
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return a successful JSON envelope: {code: 0, msg, data}."""
    body: dict[str, Any] = {"code": 0, "msg": msg, "data": data}
    return JSONResponse(body, status_code=status_code, headers=headers)


def api_error(
    code: int,
    msg: str,
    status_code: int = 400,
    data: Any = None,
) -> JSONResponse:
    """Return an error JSON envelope: {code: >0, msg, data}."""
    return JSONResponse({"code": code, "msg": msg, "data": data}, status_code=status_code)


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


def api_conflict(msg: str) -> JSONResponse:
    return api_error(ErrorCode.CONFLICT, msg, 409)


def api_internal_error(msg: str = "内部错误") -> JSONResponse:
    return api_error(ErrorCode.INTERNAL_ERROR, msg, 500)
