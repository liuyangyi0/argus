"""Background task monitoring API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from argus.dashboard.api_response import api_success, api_unavailable, api_validation_error

router = APIRouter()


@router.get("/json")
def tasks_list_json(request: Request):
    """JSON API: active tasks with progress."""
    task_manager = getattr(request.app.state, "task_manager", None)
    if not task_manager:
        return api_success({"tasks": []})

    tasks = task_manager.get_active_tasks()
    return api_success({"tasks": [
        {
            "task_id": t.task_id,
            "task_type": t.task_type,
            "camera_id": t.camera_id,
            "status": t.status.value,
            "progress": t.progress,
            "message": t.message,
            "error": t.error,
            "result": t.result if isinstance(t.result, dict) else None,
        }
        for t in tasks
    ]})


@router.delete("/{task_id}")
async def dismiss_task(request: Request, task_id: str):
    """Dismiss a completed/failed task."""
    task_manager = getattr(request.app.state, "task_manager", None)
    if not task_manager:
        return api_unavailable("不可用")

    if task_manager.dismiss(task_id):
        return HTMLResponse("")
    return api_validation_error("无法清除运行中的任务")
