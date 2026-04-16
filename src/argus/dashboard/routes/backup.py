"""Data backup and restore API routes (Chinese UI)."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Request

from argus.dashboard.api_response import api_success, api_error, api_unavailable
from argus.dashboard.forms import parse_request_form

logger = structlog.get_logger()

router = APIRouter()


@router.get("/list/json")
async def backup_list_json(request: Request):
    """List backups as JSON for Vue frontend."""
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
        return api_unavailable("备份管理器不可用")

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
            return api_success({"task_id": task_id})
        except RuntimeError as e:
            return api_error(42900, str(e), status_code=429)
    else:
        # Fallback: run synchronously
        try:
            result = backup_manager.create_backup(include_models=include_models)
            return api_success(result)
        except Exception as e:
            logger.error("backup.create_failed", error=str(e))
            return api_error(50000, f"备份失败: {e}", status_code=500)


@router.post("/restore")
async def restore_backup(request: Request):
    """Restore database from a named backup."""
    backup_manager = getattr(request.app.state, "backup_manager", None)
    if not backup_manager:
        return api_unavailable("备份管理器不可用")

    form = await parse_request_form(request)
    backup_name = form.get("backup_name", "").strip()
    if not backup_name:
        return api_error(40000, "未指定备份名称")

    # Validate name is a plain directory name (no path traversal)
    if "/" in backup_name or "\\" in backup_name or ".." in backup_name:
        return api_error(40000, "备份名称无效")

    success = await asyncio.to_thread(backup_manager.restore_database, backup_name)
    if success:
        logger.info("restore.ok", backup=backup_name)
        return api_success({"backup_name": backup_name, "restored": True})
    else:
        return api_error(50000, "恢复失败，请查看日志", status_code=500)


@router.delete("/{backup_name}")
def delete_backup(request: Request, backup_name: str):
    """Delete a specific backup."""
    backup_manager = getattr(request.app.state, "backup_manager", None)
    if not backup_manager:
        return api_unavailable("备份管理器不可用")

    # Validate name
    if "/" in backup_name or "\\" in backup_name or ".." in backup_name:
        return api_error(40000, "备份名称无效")

    success = backup_manager.delete_backup(backup_name)
    if success:
        return api_success({"backup_name": backup_name, "deleted": True})
    from argus.dashboard.api_response import api_not_found
    return api_not_found("备份不存在")


