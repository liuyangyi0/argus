"""Data backup and restore API routes (Chinese UI)."""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from argus.dashboard.forms import htmx_toast_headers, parse_request_form

logger = structlog.get_logger()

router = APIRouter()


@router.get("/list/json")
async def backup_list_json(request: Request):
    """List backups as JSON for Vue frontend."""
    from argus.dashboard.api_response import api_success, api_unavailable

    backup_manager = getattr(request.app.state, "backup_manager", None)
    if not backup_manager:
        return api_unavailable("备份管理器不可用")

    backups = backup_manager.list_backups()
    return api_success({"backups": backups})


@router.post("/create")
async def create_backup(request: Request):
    """Create a new backup (submitted as background task)."""
    backup_manager = getattr(request.app.state, "backup_manager", None)
    task_manager = getattr(request.app.state, "task_manager", None)

    if not backup_manager:
        return JSONResponse(
            {"error": "备份管理器不可用"},
            status_code=503,
            headers=htmx_toast_headers("备份管理器不可用", toast_type="error"),
        )

    # Parse include_models from form
    form = await parse_request_form(request)
    include_models = form.get("include_models") == "true"

    def _do_backup(progress_callback, include_models: bool = False):
        progress_callback(10, "正在备份数据库...")
        result = backup_manager.create_backup(include_models=include_models)
        progress_callback(100, f"备份完成，大小 {result.get('size_mb', 0)} MB")
        return result

    if task_manager:
        try:
            task_id = task_manager.submit(
                "backup_create",
                _do_backup,
                include_models=include_models,
            )
            logger.info("backup.task_submitted", task_id=task_id)
            return JSONResponse(
                {"status": "ok", "task_id": task_id},
                headers={
                    "HX-Trigger": json.dumps({
                        "showToast": {"message": "备份任务已提交", "type": "success"},
                    }),
                },
            )
        except RuntimeError as e:
            return JSONResponse(
                {"error": str(e)},
                status_code=429,
                headers=htmx_toast_headers(str(e), toast_type="error"),
            )
    else:
        # Fallback: run synchronously
        try:
            result = backup_manager.create_backup(include_models=include_models)
            return JSONResponse(
                {"status": "ok", **result},
                headers={
                    "HX-Trigger": json.dumps({
                        "showToast": {
                            "message": f"备份完成，大小 {result.get('size_mb', 0)} MB",
                            "type": "success",
                        },
                        "backupCreated": {},
                    }),
                },
            )
        except Exception as e:
            logger.error("backup.create_failed", error=str(e))
            return JSONResponse(
                {"error": str(e)},
                status_code=500,
                headers=htmx_toast_headers(f"备份失败: {e}", toast_type="error"),
            )


@router.post("/restore")
async def restore_backup(request: Request):
    """Restore database from a named backup."""
    backup_manager = getattr(request.app.state, "backup_manager", None)
    if not backup_manager:
        return JSONResponse(
            {"error": "备份管理器不可用"},
            status_code=503,
            headers=htmx_toast_headers("备份管理器不可用", toast_type="error"),
        )

    form = await parse_request_form(request)
    backup_name = form.get("backup_name", "").strip()
    if not backup_name:
        return JSONResponse({"error": "未指定备份名称"}, status_code=400)

    # Validate name is a plain directory name (no path traversal)
    if "/" in backup_name or "\\" in backup_name or ".." in backup_name:
        return JSONResponse({"error": "备份名称无效"}, status_code=400)

    success = await asyncio.to_thread(backup_manager.restore_database, backup_name)
    if success:
        logger.info("restore.ok", backup=backup_name)
        return JSONResponse(
            {"status": "ok"},
            headers={
                "HX-Trigger": json.dumps({
                    "showToast": {
                        "message": f"数据库已从 {backup_name} 恢复，建议重启服务",
                        "type": "success",
                    },
                }),
            },
        )
    else:
        return JSONResponse(
            {"error": "恢复失败，请查看日志"},
            status_code=500,
            headers=htmx_toast_headers("数据库恢复失败", toast_type="error"),
        )


@router.delete("/{backup_name}")
def delete_backup(request: Request, backup_name: str):
    """Delete a specific backup."""
    backup_manager = getattr(request.app.state, "backup_manager", None)
    if not backup_manager:
        return JSONResponse({"error": "备份管理器不可用"}, status_code=503)

    # Validate name
    if "/" in backup_name or "\\" in backup_name or ".." in backup_name:
        return JSONResponse({"error": "备份名称无效"}, status_code=400)

    success = backup_manager.delete_backup(backup_name)
    if success:
        return HTMLResponse(
            "",
            headers={
                "HX-Trigger": json.dumps({
                    "showToast": {"message": "备份已删除", "type": "success"},
                    "backupDeleted": {},
                }),
            },
        )
    return JSONResponse({"error": "备份不存在"}, status_code=404)


