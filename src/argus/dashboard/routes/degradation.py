"""Degradation status API for the global degradation bar (UX v2 §5).

Provides endpoints for the frontend to query active and historical
degradation events displayed in the 48px global bar.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _get_degradation_manager(request: Request):
    """Get the GlobalDegradationManager from app state."""
    return getattr(request.app.state, "degradation_manager", None)


@router.get("/active")
def active_degradations(request: Request):
    """Return all active degradation events.

    Response: [{event_id, level, category, camera_id, title, impact,
                action, started_at}]
    Sorted by severity (severe first), then by start time.
    """
    manager = _get_degradation_manager(request)
    if manager is None:
        return JSONResponse([])
    return JSONResponse(manager.get_active())


@router.get("/history")
def degradation_history(
    request: Request,
    days: int = Query(default=7, ge=1, le=90),
):
    """Return degradation event history within the last N days.

    Response: [{event_id, level, category, camera_id, title, impact,
                action, started_at, resolved_at}]
    """
    manager = _get_degradation_manager(request)
    if manager is None:
        return JSONResponse([])
    return JSONResponse(manager.get_history(days=days))


@router.get("/summary")
def degradation_summary(request: Request):
    """Return a summary suitable for the global degradation bar.

    Response: {active_count, max_level, events: [...top 3...]}
    """
    manager = _get_degradation_manager(request)
    if manager is None:
        return JSONResponse({"active_count": 0, "max_level": None, "events": []})

    active = manager.get_active()
    max_level = manager.max_active_level
    return JSONResponse({
        "active_count": len(active),
        "max_level": max_level.value if max_level else None,
        "events": active[:3],  # Bar shows max 3, rest collapse to "+N"
    })
