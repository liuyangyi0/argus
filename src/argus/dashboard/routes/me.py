"""Current-user identity endpoint.

Returns the username and role of the session attached to the request, so
the Vue frontend can render role-aware navigation / RBAC guards without
having to scrape cookies or replay /login.

The auth middleware already produces the canonical 401 response for
unauthenticated API requests, so this handler only deals with the
authenticated case.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("")
async def get_me(request: Request) -> JSONResponse:
    """Return the current session user's identity (flat shape).

    Response (200):
        {"username": str, "role": str, "display_name": str?}

    ``display_name`` is omitted when the user record has no display name
    (or when the principal is not backed by a DB row, e.g. the
    auth-disabled "system" user or the legacy X-API-Token "api" user).

    Unauthenticated callers never reach this handler — the AuthMiddleware
    short-circuits with ``{"error": "Authentication required"}`` (401).
    """
    user = getattr(request.state, "user", None) or {}
    username = user.get("username") if isinstance(user, dict) else None
    role = user.get("role") if isinstance(user, dict) else None

    # Defensive: middleware should always populate these for authenticated
    # requests, but if something upstream skipped state.user, treat it the
    # same as unauthenticated so the frontend doesn't render with a stale
    # identity.
    if not username or not role:
        return JSONResponse(
            {"error": "Authentication required"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="Argus"'},
        )

    body: dict[str, str] = {"username": username, "role": role}

    # Look up display_name from the user table when available.  Synthetic
    # principals like "system" (auth disabled) or "api" (X-API-Token) are
    # not in the user table, so .get_user() returns None and we just skip
    # the optional field.
    db = getattr(request.app.state, "db", None)
    if db is not None:
        try:
            record = db.get_user(username)
        except Exception:
            record = None
        if record is not None:
            display_name = getattr(record, "display_name", None)
            if display_name:
                body["display_name"] = display_name

    return JSONResponse(body, status_code=200)
