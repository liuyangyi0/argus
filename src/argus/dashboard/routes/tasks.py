"""Background task monitoring API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from argus.dashboard.components import empty_state, progress_bar
from argus.dashboard.tasks import TaskStatus

router = APIRouter()

_TYPE_LABELS = {
    "baseline_capture": "基线采集",
    "model_training": "模型训练",
    "model_export": "模型导出",
    "fp_export": "误报导出",
}


@router.get("", response_class=HTMLResponse)
async def tasks_list(request: Request):
    """Active tasks list with progress."""
    task_manager = getattr(request.app.state, "task_manager", None)
    if not task_manager:
        return HTMLResponse(empty_state("任务管理器不可用"))

    tasks = task_manager.get_active_tasks()
    if not tasks:
        return HTMLResponse(
            '<div data-ws-topic="tasks" data-ws-refresh-url="/api/tasks"'
            ' hx-get="/api/tasks" hx-trigger="every 30s" hx-swap="outerHTML">'
            + empty_state("暂无活跃任务")
            + "</div>"
        )

    html = ('<div data-ws-topic="tasks" data-ws-refresh-url="/api/tasks"'
            ' hx-get="/api/tasks" hx-trigger="every 30s" hx-swap="outerHTML">')
    for t in tasks:
        html += _render_task_card(t)
    html += "</div>"
    return HTMLResponse(html)


@router.get("/{task_id}", response_class=HTMLResponse)
async def task_detail(request: Request, task_id: str):
    """Single task progress - HTMX polls this for live updates."""
    task_manager = getattr(request.app.state, "task_manager", None)
    if not task_manager:
        return HTMLResponse("")

    task = task_manager.get_task(task_id)
    if not task:
        return HTMLResponse("")

    return HTMLResponse(_render_task_card(task))


@router.delete("/{task_id}")
async def dismiss_task(request: Request, task_id: str):
    """Dismiss a completed/failed task."""
    task_manager = getattr(request.app.state, "task_manager", None)
    if not task_manager:
        return JSONResponse({"error": "不可用"}, status_code=503)

    if task_manager.dismiss(task_id):
        return HTMLResponse("")
    return JSONResponse({"error": "无法清除运行中的任务"}, status_code=400)


def _render_task_card(task) -> str:
    """Render a single task card with progress bar."""
    type_label = _TYPE_LABELS.get(task.task_type, task.task_type)
    status_cls = f"task-status-{task.status.value}"

    status_labels = {
        TaskStatus.PENDING: "等待中",
        TaskStatus.RUNNING: "运行中",
        TaskStatus.COMPLETE: "已完成",
        TaskStatus.FAILED: "失败",
    }
    status_text = status_labels.get(task.status, task.status.value)

    # Only poll when running
    poll_attr = ""
    if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        poll_attr = (
            f'data-ws-topic="tasks" data-ws-refresh-url="/api/tasks/{task.task_id}" '
            f'hx-get="/api/tasks/{task.task_id}" hx-trigger="every 30s" hx-swap="outerHTML"'
        )

    dismiss_btn = ""
    if task.status in (TaskStatus.COMPLETE, TaskStatus.FAILED):
        dismiss_btn = (
            f'<button class="btn btn-ghost btn-sm" '
            f'hx-delete="/api/tasks/{task.task_id}" hx-swap="outerHTML"'
            f'>关闭</button>'
        )

    camera_text = f" — {task.camera_id}" if task.camera_id else ""

    error_html = ""
    if task.error:
        error_html = f'<div style="color:#f44336;font-size:12px;margin-top:4px;">{task.error}</div>'

    bar = progress_bar(task.progress, task.status.value)

    return f"""
    <div class="task-card" id="task-{task.task_id}" {poll_attr}>
        <div class="task-header">
            <div>
                <span style="font-weight:500;">{type_label}{camera_text}</span>
                <span class="task-type" style="margin-left:8px;">
                    <span class="{status_cls}">{status_text}</span> — {task.progress}%
                </span>
            </div>
            {dismiss_btn}
        </div>
        {bar}
        <div class="task-message">{task.message}</div>
        {error_html}
    </div>"""
