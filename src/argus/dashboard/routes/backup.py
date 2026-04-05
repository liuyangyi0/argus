"""Data backup and restore API routes (Chinese UI)."""

from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from argus.dashboard.components import (
    confirm_button,
    empty_state,
    page_header,
)

logger = structlog.get_logger()

router = APIRouter()


@router.get("", response_class=HTMLResponse)
async def backup_page(request: Request):
    """Backup management page."""
    backup_manager = getattr(request.app.state, "backup_manager", None)

    header = page_header(
        "数据备份",
        "数据库与配置的备份与恢复",
        '<button class="btn btn-primary" hx-post="/api/backup/create" '
        'hx-swap="none" hx-include="#backup-form">立即备份</button>',
    )

    create_form = """
    <div class="card">
        <h3>创建备份</h3>
        <form id="backup-form">
            <div class="form-group" style="display:flex;align-items:center;gap:10px;">
                <input type="checkbox" name="include_models" id="include_models"
                       value="true" style="width:auto;margin:0;">
                <label for="include_models" class="form-label" style="margin:0;cursor:pointer;">
                    包含模型文件（体积较大）
                </label>
            </div>
        </form>
        <p style="color:#8890a0;font-size:13px;margin-top:8px;">
            备份内容：SQLite 数据库（在线热备份）、配置文件，可选模型文件。
        </p>
    </div>"""

    backups_html = _render_backup_table(backup_manager)

    return HTMLResponse(f"""
    {header}
    {create_form}
    <div id="backup-list"
         hx-get="/api/backup/list"
         hx-trigger="load"
         hx-swap="innerHTML">
        {backups_html}
    </div>""")


@router.get("/list", response_class=HTMLResponse)
async def backup_list(request: Request):
    """Render the backup list table (used for HTMX refresh)."""
    backup_manager = getattr(request.app.state, "backup_manager", None)
    return HTMLResponse(_render_backup_table(backup_manager))


@router.post("/create")
async def create_backup(request: Request):
    """Create a new backup (submitted as background task)."""
    backup_manager = getattr(request.app.state, "backup_manager", None)
    task_manager = getattr(request.app.state, "task_manager", None)

    if not backup_manager:
        return JSONResponse(
            {"error": "备份管理器不可用"},
            status_code=503,
            headers={"HX-Trigger": '{"showToast": {"message": "备份管理器不可用", "type": "error"}}'},
        )

    # Parse include_models from form
    form = await request.form()
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
                headers={"HX-Trigger": f'{{"showToast": {{"message": "{e}", "type": "error"}}}}'},
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
                headers={"HX-Trigger": f'{{"showToast": {{"message": "备份失败: {e}", "type": "error"}}}}'},
            )


@router.post("/restore")
async def restore_backup(request: Request):
    """Restore database from a named backup."""
    backup_manager = getattr(request.app.state, "backup_manager", None)
    if not backup_manager:
        return JSONResponse(
            {"error": "备份管理器不可用"},
            status_code=503,
            headers={"HX-Trigger": '{"showToast": {"message": "备份管理器不可用", "type": "error"}}'},
        )

    form = await request.form()
    backup_name = form.get("backup_name", "").strip()
    if not backup_name:
        return JSONResponse({"error": "未指定备份名称"}, status_code=400)

    # Validate name is a plain directory name (no path traversal)
    if "/" in backup_name or "\\" in backup_name or ".." in backup_name:
        return JSONResponse({"error": "备份名称无效"}, status_code=400)

    success = backup_manager.restore_database(backup_name)
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
            headers={"HX-Trigger": '{"showToast": {"message": "数据库恢复失败", "type": "error"}}'},
        )


@router.delete("/{backup_name}")
async def delete_backup(request: Request, backup_name: str):
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


def _render_backup_table(backup_manager) -> str:
    """Render the backup list as an HTML table."""
    if not backup_manager:
        return empty_state("备份管理器不可用")

    backups = backup_manager.list_backups()
    if not backups:
        return empty_state("暂无备份", "点击「立即备份」创建第一个备份")

    rows = ""
    for b in backups:
        db_badge = _yes_no_badge(b["has_db"])
        cfg_badge = _yes_no_badge(b["has_configs"])
        model_badge = _yes_no_badge(b["has_models"])

        restore_btn = (
            f'<button class="btn btn-warning btn-sm" '
            f'hx-post="/api/backup/restore" '
            f'hx-vals=\'{{"backup_name": "{b["name"]}}}\' '
            f'hx-swap="none" '
            f'hx-confirm="确定要从此备份恢复数据库吗？当前数据库将被覆盖（系统会保留一份安全副本），'
            f'建议恢复后重启服务。">'
            f'恢复</button>'
        )
        bname = b["name"]
        delete_btn = (
            f'<button class="btn btn-danger btn-sm" '
            f'hx-delete="/api/backup/{bname}" '
            f'hx-target="closest tr" '
            f'hx-swap="outerHTML" '
            f'hx-confirm="确定删除备份 {bname} 吗？此操作不可撤销。">'
            f'删除</button>'
        )

        rows += f"""
        <tr>
            <td style="font-family:monospace;font-size:13px;">{b["created"]}</td>
            <td>{b["size_mb"]} MB</td>
            <td style="text-align:center;">{db_badge}</td>
            <td style="text-align:center;">{cfg_badge}</td>
            <td style="text-align:center;">{model_badge}</td>
            <td>
                <div class="flex gap-8">
                    {restore_btn}
                    {delete_btn}
                </div>
            </td>
        </tr>"""

    return f"""
    <div class="card">
        <h3>备份列表</h3>
        <table>
            <thead>
                <tr>
                    <th>创建时间</th>
                    <th>大小</th>
                    <th style="text-align:center;">数据库</th>
                    <th style="text-align:center;">配置</th>
                    <th style="text-align:center;">模型</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>"""


def _yes_no_badge(value: bool) -> str:
    if value:
        return '<span style="color:#4caf50;font-weight:600;">✓</span>'
    return '<span style="color:#616161;">—</span>'
