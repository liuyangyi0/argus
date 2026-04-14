"""Reports and statistics routes (Chinese UI with Chart.js)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from argus.dashboard.api_response import api_success, api_unavailable

router = APIRouter()


@router.get("/json")
async def reports_json(request: Request):
    """JSON statistics API for external consumption."""
    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    total = db.get_alert_count()
    all_alerts = db.get_alerts(limit=50000)
    fp_count = sum(1 for a in all_alerts if a.false_positive)
    ack_count = sum(1 for a in all_alerts if a.acknowledged)

    return {
        "total_alerts": total,
        "by_severity": {
            "high": db.get_alert_count(severity="high"),
            "medium": db.get_alert_count(severity="medium"),
            "low": db.get_alert_count(severity="low"),
            "info": db.get_alert_count(severity="info"),
        },
        "false_positive_count": fp_count,
        "false_positive_rate": round(fp_count / total * 100, 2) if total > 0 else 0,
        "acknowledged_count": ack_count,
        "acknowledged_rate": round(ack_count / total * 100, 2) if total > 0 else 0,
    }


# ── Shared query helpers ──────────


def _compute_daily_trend(alerts, days: int):
    """Aggregate alerts by date and severity for the last N days."""
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).replace(tzinfo=None)

    daily: dict[str, dict[str, int]] = defaultdict(
        lambda: {"high": 0, "medium": 0, "low": 0, "info": 0},
    )
    for a in alerts:
        if a.timestamp and a.timestamp >= cutoff:
            key = a.timestamp.strftime("%m-%d")
            if a.severity in daily[key]:
                daily[key][a.severity] += 1

    labels, high, medium, low, info = [], [], [], [], []
    for i in range(days):
        d = now - timedelta(days=days - 1 - i)
        key = d.strftime("%m-%d")
        labels.append(key)
        counts = daily.get(key, {"high": 0, "medium": 0, "low": 0, "info": 0})
        high.append(counts["high"])
        medium.append(counts["medium"])
        low.append(counts["low"])
        info.append(counts["info"])
    return {"labels": labels, "high": high, "medium": medium, "low": low, "info": info}


def _compute_camera_dist(alerts):
    """Count alerts per camera."""
    cam_counts: dict[str, int] = defaultdict(int)
    for a in alerts:
        cam_counts[a.camera_id] += 1
    return [{"camera_id": c, "count": n} for c, n in sorted(cam_counts.items())]


def _compute_fp_trend(alerts, days: int):
    """Daily false-positive rate for the last N days."""
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).replace(tzinfo=None)

    daily_total: dict[str, int] = defaultdict(int)
    daily_fp: dict[str, int] = defaultdict(int)
    for a in alerts:
        if a.timestamp and a.timestamp >= cutoff:
            key = a.timestamp.strftime("%m-%d")
            daily_total[key] += 1
            if a.false_positive:
                daily_fp[key] += 1

    labels, rates = [], []
    for i in range(days):
        d = now - timedelta(days=days - 1 - i)
        key = d.strftime("%m-%d")
        labels.append(key)
        total = daily_total.get(key, 0)
        fp = daily_fp.get(key, 0)
        rates.append(round(fp / total * 100, 1) if total > 0 else 0)
    return {"labels": labels, "rates": rates}


# ── JSON chart data endpoints (consumed by Vue frontend) ──────────


@router.get("/daily-trend/json")
async def daily_trend_json(request: Request, days: int = Query(30, ge=7, le=90)):
    """Daily alert trend data as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    return api_success(_compute_daily_trend(db.get_alerts(limit=50000), days))


@router.get("/severity-dist/json")
async def severity_dist_json(request: Request):
    """Severity distribution as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    return api_success({
        "high": db.get_alert_count(severity="high"),
        "medium": db.get_alert_count(severity="medium"),
        "low": db.get_alert_count(severity="low"),
        "info": db.get_alert_count(severity="info"),
    })


@router.get("/camera-dist/json")
async def camera_dist_json(request: Request):
    """Alerts per camera as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    return api_success({"cameras": _compute_camera_dist(db.get_alerts(limit=50000))})


@router.get("/fp-trend/json")
async def fp_trend_json(request: Request, days: int = Query(30, ge=7, le=90)):
    """Daily false positive rate trend as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    return api_success(_compute_fp_trend(db.get_alerts(limit=50000), days))
