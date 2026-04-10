"""Labeling queue API routes for active learning (closed-loop retraining).

Provides endpoints for operators to label uncertain frames, view queue
statistics, and trigger incremental retraining when enough labels are
accumulated.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from argus.dashboard.api_response import (
    api_internal_error,
    api_not_found,
    api_success,
    api_unavailable,
    api_validation_error,
)

logger = structlog.get_logger()

router = APIRouter()


@router.get("/queue")
def labeling_queue(
    request: Request,
    camera_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    """Get pending labeling queue entries sorted by entropy (most uncertain first)."""
    db = request.app.state.db
    if db is None:
        return api_unavailable("数据库未初始化")

    entries = db.get_labeling_queue(
        camera_id=camera_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return api_success(data={
        "entries": [e.to_dict() for e in entries],
        "count": len(entries),
    })


@router.post("/{entry_id}/label")
async def label_entry(
    request: Request,
    entry_id: int,
) -> JSONResponse:
    """Label an uncertain frame as normal or anomaly.

    Body: {"label": "normal"|"anomaly", "labeled_by": "operator_name"}
    """
    db = request.app.state.db
    if db is None:
        return api_unavailable("数据库未初始化")

    try:
        body = await request.json()
    except Exception:
        return api_validation_error("无效的请求体")

    label = body.get("label")
    if label not in ("normal", "anomaly"):
        return api_validation_error("label 必须为 'normal' 或 'anomaly'")

    labeled_by = body.get("labeled_by", "operator")

    record = db.label_entry(entry_id, label=label, labeled_by=labeled_by)
    if record is None:
        return api_not_found("标注条目不存在")

    # If labeled as "normal", copy frame to baselines for retraining
    if label == "normal" and record.frame_path:
        _copy_to_baselines(record.camera_id, record.frame_path)

    # If labeled as "anomaly", frame can be used as positive sample in future
    # (stored in labeling_queue with label='anomaly' for incremental training)

    logger.info(
        "labeling.entry_labeled",
        entry_id=entry_id,
        camera_id=record.camera_id,
        label=label,
        labeled_by=labeled_by,
    )

    return api_success(data=record.to_dict(), msg="标注成功")


@router.post("/{entry_id}/skip")
async def skip_entry(
    request: Request,
    entry_id: int,
) -> JSONResponse:
    """Skip a labeling entry (mark as skipped)."""
    db = request.app.state.db
    if db is None:
        return api_unavailable("数据库未初始化")

    from argus.storage.models import LabelingQueueRecord, LabelingQueueStatus
    from sqlalchemy import select

    with db.get_session() as session:
        record = session.scalar(
            select(LabelingQueueRecord).where(LabelingQueueRecord.id == entry_id)
        )
        if record is None:
            return api_not_found("标注条目不存在")
        record.status = LabelingQueueStatus.SKIPPED
        session.commit()

    return api_success(msg="已跳过")


@router.get("/stats")
def labeling_stats(
    request: Request,
    camera_id: str | None = Query(None),
) -> JSONResponse:
    """Get labeling queue statistics."""
    db = request.app.state.db
    if db is None:
        return api_unavailable("数据库未初始化")

    stats = db.get_labeling_stats(camera_id=camera_id)

    # Also include active learning sampler stats if available
    sampler = getattr(request.app.state, "active_learning_sampler", None)
    stats["sampler"] = sampler.get_stats() if sampler else None
    return api_success(data=stats)


@router.post("/trigger-retrain")
async def trigger_retrain(
    request: Request,
) -> JSONResponse:
    """Trigger incremental retraining using accumulated labels.

    Body: {"camera_id": "cam_01", "model_type": "patchcore"} (optional)
    """
    db = request.app.state.db
    if db is None:
        return api_unavailable("数据库未初始化")

    try:
        body = await request.json()
    except Exception:
        body = {}

    camera_id = body.get("camera_id")

    # Count available labeled entries not yet used for training
    labeled = db.get_labeled_entries(camera_id=camera_id, trained_into="")
    if len(labeled) < 5:
        return api_validation_error(
            f"标注数量不足: {len(labeled)} 条 (需要 >= 5 条未训练标注)"
        )

    # Collect normal-labeled frames as supplementary baselines
    normal_entries = [e for e in labeled if e.label == "normal"]
    anomaly_entries = [e for e in labeled if e.label == "anomaly"]

    # Copy normal frames to baseline directory for retraining
    copied = 0
    for entry in normal_entries:
        if entry.frame_path and Path(entry.frame_path).exists():
            _copy_to_baselines(entry.camera_id, entry.frame_path)
            copied += 1

    # Create a training job if training_jobs infrastructure is available
    from argus.storage.models import TrainingJobRecord, TrainingJobStatus, TrainingTriggerType
    import uuid

    job_id = f"al-{uuid.uuid4().hex[:12]}"
    model_type = body.get("model_type", "patchcore")
    target_camera = camera_id or (labeled[0].camera_id if labeled else "unknown")

    triggered_by = body.get("triggered_by", "active_learning")

    job = TrainingJobRecord(
        job_id=job_id,
        job_type="anomaly_head",
        camera_id=target_camera,
        model_type=model_type,
        trigger_type=TrainingTriggerType.DRIFT_SUGGESTED,
        triggered_by=triggered_by,
        confirmation_required=True,
        status=TrainingJobStatus.PENDING_CONFIRMATION,
    )
    db.save_training_job(job)

    # Mark labeled entries as pending training (will be finalized after training completes)
    entry_ids = [e.id for e in labeled]
    db.mark_labeling_trained(entry_ids, trained_into=job_id)

    logger.info(
        "labeling.retrain_triggered",
        job_id=job_id,
        camera_id=target_camera,
        normal_count=len(normal_entries),
        anomaly_count=len(anomaly_entries),
        copied_baselines=copied,
    )

    return api_success(data={
        "job_id": job_id,
        "camera_id": target_camera,
        "normal_count": len(normal_entries),
        "anomaly_count": len(anomaly_entries),
        "copied_baselines": copied,
        "message": f"已创建训练任务 {job_id}，需要工程师确认后执行",
    })


@router.get("/{entry_id}/image", response_model=None)
def labeling_image(
    request: Request,
    entry_id: int,
):
    """Serve the uncertain frame image for labeling UI."""
    db = request.app.state.db
    if db is None:
        return api_unavailable("数据库未初始化")

    from argus.storage.models import LabelingQueueRecord
    from sqlalchemy import select

    with db.get_session() as session:
        record = session.scalar(
            select(LabelingQueueRecord).where(LabelingQueueRecord.id == entry_id)
        )
        if record is None:
            return api_not_found("标注条目不存在")
        frame_path = record.frame_path

    if not frame_path or not Path(frame_path).exists():
        return api_not_found("图片文件不存在")

    return FileResponse(frame_path, media_type="image/jpeg")


def _copy_to_baselines(camera_id: str, frame_path: str) -> bool:
    """Copy a labeled frame to the baselines directory for retraining."""
    src = Path(frame_path)
    if not src.exists():
        return False

    baselines_dir = Path("data/baselines") / camera_id / "default"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    dst = baselines_dir / f"al_{src.name}"
    if dst.exists():
        return False

    try:
        shutil.copy2(str(src), str(dst))
        logger.debug(
            "labeling.frame_copied_to_baselines",
            camera_id=camera_id,
            src=str(src),
            dst=str(dst),
        )
        return True
    except Exception:
        logger.warning(
            "labeling.copy_failed",
            camera_id=camera_id,
            src=str(src),
            exc_info=True,
        )
        return False
