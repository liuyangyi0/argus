"""Training job lifecycle API routes.

Provides endpoints for creating, confirming, rejecting, and listing
training jobs. Nuclear environment requires human confirmation before
any training job executes.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from argus.storage.models import (
    AuditLog,
    TrainingJobRecord,
    TrainingJobStatus,
    TrainingJobType,
    TrainingTriggerType,
)

logger = structlog.get_logger()

router = APIRouter()


def _get_db(request: Request):
    return getattr(request.app.state, "database", None) or getattr(request.app.state, "db", None)


def _write_audit(db, user: str, action: str, target_id: str, detail: str) -> None:
    """Write an audit log entry (best-effort, failures are logged not raised)."""
    try:
        with db.get_session() as session:
            session.add(AuditLog(
                user=user,
                action=action,
                target_type="training_job",
                target_id=target_id,
                detail=detail,
            ))
            session.commit()
    except Exception as e:
        logger.warning("training_job.audit_failed", error=str(e))


def _get_pending_job(db, job_id: str):
    """Fetch a job and validate it's pending confirmation. Returns (job, error_response)."""
    job = db.get_training_job(job_id)
    if job is None:
        return None, JSONResponse({"error": f"Job not found: {job_id}"}, status_code=404)
    if job.status != TrainingJobStatus.PENDING_CONFIRMATION.value:
        return None, JSONResponse(
            {"error": f"Job is not pending confirmation (status={job.status})"},
            status_code=400,
        )
    return job, None


@router.get("/json")
async def list_training_jobs(
    request: Request,
    status: str | None = None,
    job_type: str | None = None,
    camera_id: str | None = None,
    limit: int = 50,
):
    """List training jobs with optional filters."""
    db = _get_db(request)
    if db is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    jobs = db.list_training_jobs(
        status=status, job_type=job_type, camera_id=camera_id, limit=min(limit, 200),
    )
    pending_count = db.count_pending_jobs()
    return JSONResponse({
        "jobs": [j.to_dict() for j in jobs],
        "total": len(jobs),
        "pending_count": pending_count,
    })


@router.get("/{job_id}")
async def get_training_job(request: Request, job_id: str):
    """Get training job detail."""
    db = _get_db(request)
    if db is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    job = db.get_training_job(job_id)
    if job is None:
        return JSONResponse({"error": f"Job not found: {job_id}"}, status_code=404)

    data = job.to_dict()
    for field in ("hyperparameters", "metrics", "validation_report"):
        if data.get(field):
            try:
                data[field] = json.loads(data[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return JSONResponse(data)


@router.post("/")
async def create_training_job(request: Request):
    """Create a new training job (status=pending_confirmation).

    Body: {
        job_type: "ssl_backbone" | "anomaly_head",
        camera_id?: str,     (required for anomaly_head)
        zone_id?: str,       (default: "default")
        model_type?: str,
        trigger_type?: str,  (default: "manual")
        triggered_by?: str,
        hyperparameters?: dict,
    }
    """
    db = _get_db(request)
    if db is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    valid_types = {t.value for t in TrainingJobType}
    job_type = body.get("job_type")
    if job_type not in valid_types:
        return JSONResponse(
            {"error": f"job_type must be one of {valid_types}"},
            status_code=400,
        )

    camera_id = body.get("camera_id")
    if job_type == TrainingJobType.ANOMALY_HEAD.value and not camera_id:
        return JSONResponse(
            {"error": "camera_id is required for anomaly_head jobs"},
            status_code=400,
        )

    # Validate camera_id exists in config
    if camera_id:
        config = getattr(request.app.state, "config", None)
        if config and not any(c.camera_id == camera_id for c in config.cameras):
            return JSONResponse(
                {"error": f"Camera '{camera_id}' not found in configuration"},
                status_code=400,
            )

    hyperparams = body.get("hyperparameters")
    job_id = str(uuid.uuid4())[:12]

    base_model_version = None
    if job_type == TrainingJobType.ANOMALY_HEAD.value:
        backbone = db.get_active_backbone()
        if backbone:
            base_model_version = backbone.backbone_version_id

    record = TrainingJobRecord(
        job_id=job_id,
        job_type=job_type,
        camera_id=camera_id,
        zone_id=body.get("zone_id", "default"),
        model_type=body.get("model_type"),
        trigger_type=body.get("trigger_type", TrainingTriggerType.MANUAL.value),
        triggered_by=body.get("triggered_by", "dashboard"),
        confirmation_required=True,
        status=TrainingJobStatus.PENDING_CONFIRMATION.value,
        base_model_version=base_model_version,
        dataset_version=body.get("dataset_version"),
        hyperparameters=json.dumps(hyperparams) if hyperparams else None,
    )

    db.save_training_job(record)
    logger.info(
        "training_job.created",
        job_id=job_id,
        job_type=job_type,
        camera_id=camera_id,
    )

    return JSONResponse({"status": "ok", "job_id": job_id}, status_code=201)


@router.post("/{job_id}/confirm")
async def confirm_training_job(request: Request, job_id: str):
    """Human confirms a pending training job (writes audit log)."""
    db = _get_db(request)
    if db is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    job, err = _get_pending_job(db, job_id)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        body = {}

    confirmed_by = body.get("confirmed_by", "operator")
    now = datetime.now(timezone.utc)

    # Atomic update: only transition if still pending (prevents double-confirm race)
    updated = db.update_training_job(
        job_id,
        status=TrainingJobStatus.QUEUED.value,
        confirmed_by=confirmed_by,
        confirmed_at=now,
    )
    if updated is False:
        return JSONResponse(
            {"error": "Job was already confirmed or status changed"},
            status_code=409,
        )

    _write_audit(
        db, confirmed_by, "training_job.confirm", job_id,
        f"Confirmed {job.job_type} job for {job.camera_id or 'all cameras'}",
    )

    logger.info("training_job.confirmed", job_id=job_id, confirmed_by=confirmed_by)
    return JSONResponse({"status": "ok", "job_id": job_id, "new_status": "queued"})


@router.post("/{job_id}/reject")
async def reject_training_job(request: Request, job_id: str):
    """Human rejects a pending training job."""
    db = _get_db(request)
    if db is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    job, err = _get_pending_job(db, job_id)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        body = {}

    rejected_by = body.get("rejected_by", "operator")
    reason = body.get("reason", "")

    db.update_training_job(
        job_id,
        status=TrainingJobStatus.REJECTED.value,
        error=f"Rejected by {rejected_by}: {reason}" if reason else f"Rejected by {rejected_by}",
    )

    _write_audit(
        db, rejected_by, "training_job.reject", job_id,
        reason or "No reason provided",
    )

    logger.info("training_job.rejected", job_id=job_id, rejected_by=rejected_by)
    return JSONResponse({"status": "ok", "job_id": job_id, "new_status": "rejected"})


@router.get("/backbones/json")
async def list_backbones(request: Request):
    """List backbone versions."""
    db = _get_db(request)
    if db is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    backbones = db.list_backbones()
    active_id = next(
        (b.backbone_version_id for b in backbones if b.is_active), None,
    )
    return JSONResponse({
        "backbones": [b.to_dict() for b in backbones],
        "active_version_id": active_id,
        "total": len(backbones),
    })
