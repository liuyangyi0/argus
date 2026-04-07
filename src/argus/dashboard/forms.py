"""Form parsing helpers for dashboard routes."""

from __future__ import annotations

import json
from urllib.parse import parse_qsl

from fastapi import Request
from starlette.datastructures import FormData


async def parse_request_form(request: Request) -> FormData:
    """Parse form data, falling back to urlencoded bodies without python-multipart."""
    try:
        return await request.form()
    except AssertionError as exc:
        if "python-multipart" not in str(exc):
            raise

        content_type = request.headers.get("content-type", "")
        mime_type = content_type.split(";", 1)[0].strip().lower()
        if mime_type != "application/x-www-form-urlencoded":
            raise

        body = (await request.body()).decode("utf-8", errors="replace")
        return FormData(parse_qsl(body, keep_blank_values=True))


def htmx_toast_headers(
    message: str,
    *,
    toast_type: str = "success",
    **events: object,
) -> dict[str, str]:
    """Build ASCII-safe HTMX trigger headers for toast notifications."""
    payload: dict[str, object] = {
        "showToast": {"message": message, "type": toast_type},
    }
    payload.update(events)
    return {"HX-Trigger": json.dumps(payload)}