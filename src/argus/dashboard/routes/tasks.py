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


@router.get("/json")
async def tasks_list_json(request: Request):
    """JSON API: active tasks with progress."""
    task_manager = getattr(request.app.state, "task_manager", None)
    if not task_manager:
        return JSONResponse({"tasks": []})

    tasks = task_manager.get_active_tasks()
    return JSONResponse({"tasks": [
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

    # Capture stats block for completed baseline captures (CAP-006)
    stats_html = ""
    if (
        task.status == TaskStatus.COMPLETE
        and task.task_type == "baseline_capture"
        and isinstance(task.result, dict)
        and "stats" in task.result
    ):
        stats_html = _render_capture_stats(task.result["stats"])

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
        {stats_html}
    </div>"""


def _render_capture_stats(stats: dict) -> str:
    """Render baseline capture statistics block (CAP-006)."""
    total = stats.get("total_grabbed", 0)
    accepted = stats.get("accepted", 0)
    rejected = stats.get("total_rejected", 0)
    bmin = stats.get("brightness_min", 0)
    bmax = stats.get("brightness_max", 0)

    # Filter breakdown
    reasons = []
    for key, label in [
        ("rejected_blur", "模糊"),
        ("rejected_exposure", "曝光"),
        ("rejected_duplicate", "重复"),
        ("rejected_person", "人员"),
        ("rejected_encoder", "编码错误"),
    ]:
        val = stats.get(key, 0)
        if val > 0:
            reasons.append(f"{label} {val}")

    reason_text = " | ".join(reasons) if reasons else "无"

    return f"""
    <div style="margin-top:8px;padding:8px 12px;background:#1a1d27;border-radius:6px;font-size:12px;color:#8890a0;">
        <div style="margin-bottom:4px;">
            <span style="color:#e0e0e0;">采集统计:</span>
            抓取 {total} 帧 | 保留 <span style="color:#4fc3f7;">{accepted}</span> 帧 | 过滤 {rejected} 帧
        </div>
        <div style="margin-bottom:4px;">
            <span style="color:#e0e0e0;">过滤原因:</span> {reason_text}
        </div>
        <div>
            <span style="color:#e0e0e0;">亮度范围:</span> {bmin:.0f} - {bmax:.0f}
        </div>
    </div>"""
