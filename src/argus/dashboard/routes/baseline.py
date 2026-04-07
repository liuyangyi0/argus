"""Baseline and model management routes (Chinese UI)."""

from __future__ import annotations

import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from argus.capture.baseline_job import run_baseline_capture_job, BaselineCaptureJobConfig
from argus.config.schema import BaselineCaptureConfig
from argus.dashboard.components import (
    confirm_button,
    empty_state,
    form_checkbox,
    form_group,
    form_select,
    page_header,
    progress_bar,
    stat_card,
    tab_bar,
)
from argus.dashboard.forms import htmx_toast_headers, parse_request_form
from argus.dashboard.model_runtime import activate_model_version, find_registered_model_by_path

logger = structlog.get_logger()

router = APIRouter()

_DEFAULT_ZONE_ID = "default"

_GRADE_COLORS = {
    "A": "var(--status-ok)",
    "B": "var(--status-ok-text)",
    "C": "var(--status-warn)",
    "F": "var(--status-critical)",
}

_TABS = [
    ("capture", "基线采集", "/api/baseline/capture"),
    ("browse", "基线浏览", "/api/baseline/list"),
    ("train", "模型训练", "/api/baseline/train"),
    ("models", "模型管理", "/api/baseline/models"),
    ("history", "训练报告", "/api/baseline/training-history"),
]


def _camera_baseline_root(
    baselines_dir: Path,
    camera_id: str,
    zone_id: str = _DEFAULT_ZONE_ID,
) -> Path:
    """Return the default-zone root for a camera, with legacy flat-layout fallback."""
    zone_root = baselines_dir / camera_id / zone_id
    if zone_root.is_dir():
        return zone_root
    return baselines_dir / camera_id


def _iter_camera_baseline_versions(
    baselines_dir: Path,
    camera_id: str,
    zone_id: str = _DEFAULT_ZONE_ID,
):
    """Yield version directories for one camera's default baseline zone."""
    base_dir = _camera_baseline_root(baselines_dir, camera_id, zone_id)
    if not base_dir.exists():
        return
    for version_dir in sorted(base_dir.iterdir()):
        if version_dir.is_dir():
            yield version_dir


def _resolve_baseline_version_dir(
    baselines_dir: Path,
    camera_id: str,
    version: str,
    zone_id: str = _DEFAULT_ZONE_ID,
) -> Path:
    """Resolve a version directory for the default zone, including legacy flat paths."""
    return _camera_baseline_root(baselines_dir, camera_id, zone_id) / version


def _read_capture_meta(version_dir: Path) -> dict:
    """Load capture metadata if present."""
    import json as _json

    meta_path = version_dir / "capture_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return _json.loads(meta_path.read_text())
    except (ValueError, OSError):
        return {}


def _count_version_images(version_dir: Path) -> int:
    """Count images stored in a baseline version directory."""
    return len(list(version_dir.glob("*.png"))) + len(list(version_dir.glob("*.jpg")))


def _reset_current_marker_after_delete(base_dir: Path, deleted_version: str) -> None:
    """Update current.txt when the referenced version has been deleted."""
    marker = base_dir / "current.txt"
    if not marker.exists():
        return
    current_version = marker.read_text().strip()
    if current_version != deleted_version:
        return

    remaining_versions = sorted([path for path in base_dir.iterdir() if path.is_dir()])
    if remaining_versions:
        marker.write_text(remaining_versions[-1].name)
        return

    marker.unlink(missing_ok=True)


@router.get("", response_class=HTMLResponse)
async def baseline_main(request: Request):
    """Model management page with tabs."""
    is_htmx = request.headers.get("HX-Request") == "true"
    header = "" if is_htmx else page_header("模型", "采集基线、训练检测模型、部署到摄像头")
    tabs = tab_bar(_TABS, "capture")
    return HTMLResponse(f"""
    {header}
    {tabs}
    <div hx-get="/api/baseline/capture" hx-trigger="load" hx-target="#tab-content" hx-swap="innerHTML"></div>""")


@router.get("/capture", response_class=HTMLResponse)
async def capture_form(request: Request):
    """Baseline capture form."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse(empty_state("摄像头管理器不可用"))

    statuses = camera_manager.get_status()
    cam_options = [(c.camera_id, f"{c.camera_id} — {c.name}") for c in statuses if c.connected]

    if not cam_options:
        return HTMLResponse(empty_state("没有在线的摄像头", "请先启动摄像头"))

    cam_select = form_select("选择摄像头", "camera_id", cam_options)
    count_input = form_group("采集帧数", "count", value="100", input_type="number",
                              hint="建议至少 100 帧，覆盖不同光照条件", min_val="10", max_val="1000")
    interval_input = form_group("采集间隔（秒）", "interval", value="2.0", input_type="number",
                                 hint="帧之间的时间间隔", min_val="0.5", max_val="60", step="0.5")
    session_label_select = form_select("采集场景", "session_label", [
        ("daytime", "白天/日间"),
        ("night", "夜间"),
        ("maintenance", "检修期间"),
        ("custom", "自定义"),
    ], hint="选择当前采集的场景类型，建议覆盖多种场景以提高模型鲁棒性")

    # Coverage hint: check which session types already exist
    coverage_hint = ""
    config = request.app.state.config
    if config:
        baselines_dir = Path(config.storage.baselines_dir)
        existing_sessions: set[str] = set()
        for cam_dir in (sorted(baselines_dir.iterdir()) if baselines_dir.exists() else []):
            if cam_dir.is_dir():
                for ver_dir in _iter_camera_baseline_versions(baselines_dir, cam_dir.name):
                    session_label = str(_read_capture_meta(ver_dir).get("session_label", "")).lower()
                    if session_label in {"daytime", "night", "maintenance"}:
                        existing_sessions.add(session_label)
        if existing_sessions and len(existing_sessions) < 3:
            missing = {"daytime", "night", "maintenance"} - existing_sessions
            labels_map = {"daytime": "白天/日间", "night": "夜间", "maintenance": "检修期间"}
            missing_labels = "、".join(labels_map[m] for m in sorted(missing))
            coverage_hint = (
                f'<div class="card" style="border-left:3px solid #ffb74d;padding:12px;margin-bottom:12px;">'
                f'<span style="color:#ffb74d;">提示:</span> 尚未采集以下场景: {missing_labels}。'
                f'建议覆盖多种场景以提高检测准确率。</div>'
            )

    # Show active capture tasks or stats for completed ones
    task_html = ""
    task_manager = getattr(request.app.state, "task_manager", None)
    if task_manager:
        active = [t for t in task_manager.get_active_tasks()
                  if t.task_type == "baseline_capture" and t.status.value in ("pending", "running")]
        if active:
            task_html = '<div class="card" style="border-left:3px solid #4fc3f7;"><h3>正在采集</h3>'
            task_html += f'<div hx-get="/api/tasks/{active[0].task_id}" data-ws-topic="tasks" data-ws-refresh-url="/api/tasks/{active[0].task_id}" hx-trigger="every 30s" hx-swap="outerHTML"></div>'
            task_html += '</div>'
        else:
            # Show stats for most recently completed capture
            completed = [t for t in task_manager.get_active_tasks()
                         if t.task_type == "baseline_capture" and t.status.value == "complete"]
            if completed:
                task_html = (
                    '<div hx-get="/api/baseline/capture/stats" hx-trigger="load" hx-swap="innerHTML"></div>'
                )

    return HTMLResponse(f"""
    {task_html}
    {coverage_hint}
    <div class="card">
        <h3>开始基线采集</h3>
        <p style="color:#8890a0;font-size:13px;margin-bottom:16px;">
            从在线摄像头采集"正常"场景的参考图片，用于训练异常检测模型。
            确保采集期间场景中没有异物。
        </p>
        <form hx-post="/api/baseline/job" hx-target="#tab-content" hx-swap="innerHTML">
            {cam_select}
            {session_label_select}
            <div class="form-row">
                {count_input}
                {interval_input}
            </div>
            <div class="form-actions">
                <button type="submit" class="btn btn-primary">开始采集</button>
            </div>
        </form>
    </div>""")


@router.get("/list", response_class=HTMLResponse)
def baseline_list(request: Request):
    """List baselines by camera with version info."""
    config = request.app.state.config
    if not config:
        return HTMLResponse(empty_state("配置不可用"))

    baselines_dir = Path(config.storage.baselines_dir)
    html = ""
    for cam_dir in sorted(baselines_dir.iterdir()) if baselines_dir.exists() else []:
        if not cam_dir.is_dir():
            continue
        camera_id = cam_dir.name
        versions = []
        for ver_dir in _iter_camera_baseline_versions(baselines_dir, camera_id):
            versions.append((ver_dir.name, _count_version_images(ver_dir)))

        if not versions:
            continue

        rows = ""
        for ver_name, img_count in versions:
            rows += f"""
            <tr>
                <td>{ver_name}</td>
                <td>{img_count} 张</td>
                <td>
                    <button class="btn btn-ghost btn-sm"
                        hx-get="/api/baseline/{camera_id}/images?version={ver_name}"
                        hx-target="#tab-content" hx-swap="innerHTML">查看</button>
                </td>
            </tr>"""

        html += f"""
        <div class="card">
            <h3>摄像头: {camera_id}</h3>
            <table>
                <thead><tr><th>版本</th><th>图片数量</th><th>操作</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>"""

    if not html:
        html = empty_state("暂无基线数据", "请先在「基线采集」标签页中采集基线图片")

    return HTMLResponse(html)


@router.get("/list/json")
def baseline_list_json(request: Request):
    """JSON API: list baselines by camera with version info."""
    config = request.app.state.config
    if not config:
        return JSONResponse({"error": "配置不可用"}, status_code=503)

    baselines_dir = Path(config.storage.baselines_dir)
    if not baselines_dir.exists():
        return JSONResponse({"baselines": []})

    baselines = []
    for cam_dir in sorted(baselines_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        camera_id = cam_dir.name
        for ver_dir in _iter_camera_baseline_versions(baselines_dir, camera_id):
            image_count = _count_version_images(ver_dir)
            if not image_count:
                continue
            meta = _read_capture_meta(ver_dir)
            lifecycle = _get_lifecycle(request)
            version_state = None
            if lifecycle:
                ver_rec = lifecycle.get_version(camera_id, _DEFAULT_ZONE_ID, ver_dir.name)
                version_state = ver_rec.state if ver_rec else None
            baselines.append({
                "camera_id": camera_id,
                "version": ver_dir.name,
                "image_count": image_count,
                "session_label": meta.get("session_label", ""),
                "status": "ready",
                "state": version_state,
            })

    return JSONResponse({"baselines": baselines})


@router.get("/models/json")
async def models_list_json(request: Request):
    """JSON API: list trained models found in data/models/."""
    config = request.app.state.config
    if not config:
        return JSONResponse({"error": "配置不可用"}, status_code=503)

    models_dir = Path(config.storage.models_dir)
    if not models_dir.exists():
        return JSONResponse({"models": []})

    result = []
    for cam_dir in sorted(models_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        camera_id = cam_dir.name

        for pattern, fmt in [("model.xml", "openvino"), ("model.onnx", "onnx"),
                              ("model.pt", "pytorch"), ("model.ckpt", "checkpoint")]:
            for model_file in sorted(cam_dir.rglob(pattern),
                                     key=lambda p: p.stat().st_mtime, reverse=True):
                mtime = time.strftime("%Y-%m-%dT%H:%M:%S",
                                      time.localtime(model_file.stat().st_mtime))
                size_mb = model_file.stat().st_size / (1024 * 1024)
                rel_path = str(model_file.relative_to(Path(".")))
                result.append({
                    "camera_id": camera_id,
                    "format": fmt,
                    "size_mb": round(size_mb, 1),
                    "trained_at": mtime,
                    "model_path": rel_path,
                })

    return JSONResponse({"models": result})


@router.get("/training-history/json")
async def training_history_json(request: Request):
    """JSON API: training history with quality grades."""
    database = getattr(request.app.state, "database", None)
    if not database:
        return JSONResponse({"records": []})

    camera_id = request.query_params.get("camera_id")
    records = database.get_training_history(camera_id=camera_id, limit=50)
    return JSONResponse({"records": [r.to_dict() for r in records]})


@router.get("/{camera_id}/images", response_class=HTMLResponse)
def baseline_images(request: Request, camera_id: str):
    """Show baseline image thumbnails for a camera version."""
    config = request.app.state.config
    version = request.query_params.get("version", "default")
    baselines_dir = _resolve_baseline_version_dir(Path(config.storage.baselines_dir), camera_id, version)

    if not baselines_dir.exists():
        return HTMLResponse(empty_state("未找到基线图片"))

    images = sorted(baselines_dir.glob("*.png")) + sorted(baselines_dir.glob("*.jpg"))

    back_btn = (
        '<button class="btn btn-ghost btn-sm mb-16" '
        'hx-get="/api/baseline/list" hx-target="#tab-content" hx-swap="innerHTML">'
        '← 返回列表</button>'
    )

    if not images:
        return HTMLResponse(back_btn + empty_state("该版本暂无图片"))

    grid = '<div class="thumb-grid">'
    for img in images[:100]:  # Limit to 100 thumbnails
        grid += (
            f'<div class="thumb-item" style="display:inline-block;position:relative;margin:4px;">'
            f'<img src="/api/baseline/{camera_id}/image/{img.name}?version={version}" '
            f'alt="{img.name}" title="{img.name}">'
            f'<button class="btn btn-ghost btn-sm" style="position:absolute;top:2px;right:2px;'
            f'background:rgba(0,0,0,0.6);color:#ff5252;font-size:14px;padding:2px 6px;border-radius:4px;" '
            f'hx-delete="/api/baseline/{camera_id}/image/{img.name}?version={version}" '
            f'hx-target="#tab-content" hx-swap="innerHTML" '
            f'hx-confirm="确定删除此图片？">&times;</button>'
            f'</div>'
        )
    grid += '</div>'

    total = len(images)
    shown = min(total, 100)

    return HTMLResponse(f"""
    {back_btn}
    <div class="card">
        <h3>{camera_id} / {version} — {total} 张图片{f" (显示前 {shown} 张)" if total > 100 else ""}</h3>
        {grid}
    </div>""")


@router.get("/{camera_id}/image/{filename}")
def baseline_image(request: Request, camera_id: str, filename: str):
    """Serve a single baseline image."""
    config = request.app.state.config
    version = request.query_params.get("version", "default")
    baselines_dir = Path(config.storage.baselines_dir)
    version_dir = _resolve_baseline_version_dir(baselines_dir, camera_id, version)

    # Path safety
    img_path = (version_dir / filename).resolve()
    if not str(img_path).startswith(str(baselines_dir.resolve())):
        return Response(status_code=400)

    if not img_path.exists():
        return Response(status_code=404)

    # Return as thumbnail (resized)
    frame = cv2.imread(str(img_path))
    if frame is None:
        return Response(status_code=500)

    h, w = frame.shape[:2]
    if w > 300:
        scale = 300 / w
        frame = cv2.resize(frame, (300, int(h * scale)))

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.get("/capture/stats", response_class=HTMLResponse)
async def capture_stats_report(request: Request):
    """Display capture statistics after a capture task completes."""
    task_manager = getattr(request.app.state, "task_manager", None)
    if not task_manager:
        return HTMLResponse("")

    # Find the most recently completed capture task
    completed = [
        t for t in task_manager.get_active_tasks()
        if t.task_type == "baseline_capture" and t.status.value == "complete" and t.result
    ]
    if not completed:
        return HTMLResponse("")

    task = completed[-1]
    result = task.result
    stats = result.get("stats", {})

    if not stats:
        return HTMLResponse("")

    retention_pct = f"{stats.get('retention_rate', 0) * 100:.0f}%"
    duration = stats.get("capture_duration_seconds", 0)
    duration_str = f"{duration:.0f}s" if duration < 60 else f"{duration / 60:.1f}min"
    brightness_min = stats.get("brightness_min", 0)
    brightness_max = stats.get("brightness_max", 0)

    cards = (
        stat_card("采集帧数", str(stats.get("captured_frames", 0)), "#4fc3f7")
        + stat_card("总帧数", str(stats.get("total_frames", 0)), "#e0e0e0")
        + stat_card("过滤帧数", str(stats.get("filtered_frames", 0)), "#ffb74d")
        + stat_card("保留率", retention_pct, "#66bb6a" if stats.get("retention_rate", 0) > 0.8 else "#ff7043")
        + stat_card("耗时", duration_str, "#e0e0e0")
        + stat_card("亮度范围", f"{brightness_min:.0f}-{brightness_max:.0f}", "#e0e0e0")
    )

    filter_html = ""
    filter_reasons = stats.get("filter_reasons", {})
    if filter_reasons:
        items = "".join(f"<li>{reason}: {cnt} 帧</li>" for reason, cnt in filter_reasons.items())
        filter_html = f'<div style="margin-top:8px;"><strong>过滤原因:</strong><ul>{items}</ul></div>'

    return HTMLResponse(f"""
    <div class="card" style="border-left:3px solid #66bb6a;">
        <h3>采集统计报告</h3>
        <div class="stat-row" style="display:flex;gap:12px;flex-wrap:wrap;">
            {cards}
        </div>
        {filter_html}
    </div>""")


@router.delete("/{camera_id}/image/{filename}")
async def delete_baseline_image(request: Request, camera_id: str, filename: str):
    """Delete a single baseline image."""
    config = request.app.state.config
    if not config:
        return JSONResponse({"error": "配置不可用"}, status_code=503)

    version = request.query_params.get("version", "default")
    baselines_dir = Path(config.storage.baselines_dir)
    version_dir = _resolve_baseline_version_dir(baselines_dir, camera_id, version)

    # Path safety: validate the resolved path is under baselines directory
    img_path = (version_dir / filename).resolve()
    if not str(img_path).startswith(str(baselines_dir.resolve())):
        return Response(status_code=400)

    if not img_path.exists():
        return Response(status_code=404)

    img_path.unlink()
    logger.info("baseline.image_deleted", camera_id=camera_id, filename=filename, version=version)

    # Return refreshed thumbnail list via HTMX
    return HTMLResponse(
        f'<div hx-get="/api/baseline/{camera_id}/images?version={version}" '
        f'hx-trigger="load" hx-target="#tab-content" hx-swap="innerHTML"></div>',
        headers=htmx_toast_headers("图片已删除"),
    )


@router.delete("/version")
async def delete_baseline_version(request: Request):
    """Delete an entire baseline version directory and its lifecycle record."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", _DEFAULT_ZONE_ID)
    version = data.get("version", "")
    user = data.get("user", "operator")

    if not camera_id:
        camera_id = request.query_params.get("camera_id", "")
    if not version:
        version = request.query_params.get("version", "")
    if zone_id == _DEFAULT_ZONE_ID:
        zone_id = request.query_params.get("zone_id", zone_id)
    if user == "operator":
        user = request.query_params.get("user", user)

    if not camera_id or not version:
        return JSONResponse({"error": "camera_id and version are required"}, status_code=400)

    config = request.app.state.config
    if not config:
        return JSONResponse({"error": "配置不可用"}, status_code=503)

    baselines_dir = Path(config.storage.baselines_dir)
    base_dir = _camera_baseline_root(baselines_dir, camera_id, zone_id)
    version_dir = _resolve_baseline_version_dir(baselines_dir, camera_id, version, zone_id)
    if not version_dir.exists() or not version_dir.is_dir():
        return JSONResponse({"error": "基线版本不存在"}, status_code=404)

    lifecycle = _get_lifecycle(request)
    if lifecycle is not None:
        version_record = lifecycle.get_version(camera_id, zone_id, version)
        if version_record is not None and version_record.state == "active":
            return JSONResponse({"error": "生产中的基线版本不能直接删除，请先退役"}, status_code=400)

    shutil.rmtree(version_dir)
    _reset_current_marker_after_delete(base_dir, version)

    if lifecycle is not None:
        client_ip = request.client.host if request.client else ""
        lifecycle.delete_version(camera_id, zone_id, version, user=user, ip_address=client_ip)

    logger.info(
        "baseline.version_dir_deleted",
        camera_id=camera_id,
        zone_id=zone_id,
        version=version,
        user=user,
    )
    return JSONResponse({"status": "ok", "camera_id": camera_id, "version": version})


@router.post("/version/delete")
async def delete_baseline_version_post(request: Request):
    """POST alias for version deletion to avoid DELETE-body compatibility issues."""
    return await delete_baseline_version(request)


@router.get("/train", response_class=HTMLResponse)
async def train_form(request: Request):
    """Model training form."""
    config = request.app.state.config
    camera_manager = request.app.state.camera_manager

    if not config or not camera_manager:
        return HTMLResponse(empty_state("服务不可用"))

    # List cameras that have baselines
    baselines_dir = Path(config.storage.baselines_dir)
    from argus.anomaly.baseline import BaselineManager

    baseline_manager = BaselineManager(baselines_dir=str(baselines_dir))
    cam_options = []
    for cam_dir in sorted(baselines_dir.iterdir()) if baselines_dir.exists() else []:
        if cam_dir.is_dir():
            baseline_dir = baseline_manager.get_baseline_dir(cam_dir.name, _DEFAULT_ZONE_ID)
            img_count = baseline_manager.count_images(cam_dir.name, _DEFAULT_ZONE_ID)
            if img_count >= 10:
                cam_options.append(
                    (cam_dir.name, f"{cam_dir.name} / {baseline_dir.name} ({img_count} 张基线图片)")
                )

    if not cam_options:
        return HTMLResponse(empty_state(
            "暂无可用于训练的基线数据",
            "需要至少 10 张基线图片。请先在「基线采集」标签页中采集基线。"
        ))

    # Active training tasks
    task_html = ""
    task_manager = getattr(request.app.state, "task_manager", None)
    if task_manager:
        active = [t for t in task_manager.get_active_tasks()
                  if t.task_type == "model_training" and t.status.value in ("pending", "running")]
        if active:
            task_html = '<div class="card" style="border-left:3px solid var(--status-info);"><h3>正在训练</h3>'
            task_html += f'<div hx-get="/api/tasks/{active[0].task_id}" data-ws-topic="tasks" data-ws-refresh-url="/api/tasks/{active[0].task_id}" hx-trigger="every 30s" hx-swap="outerHTML"></div>'
            task_html += '</div>'

    cam_select = form_select("选择摄像头", "camera_id", cam_options)
    model_select = form_select("模型类型", "model_type", [
        ("patchcore", "PatchCore（推荐，速度快）"),
        ("efficient_ad", "EfficientAD（精度高）"),
    ], selected="patchcore")
    export_select = form_select("导出格式", "export_format", [
        ("openvino", "OpenVINO（推荐，推理最快）"),
        ("onnx", "ONNX"),
        ("none", "不导出"),
    ], selected="openvino")
    quantization_select = form_select("量化精度", "quantization", [
        ("fp16", "FP16（半精度，默认）"),
        ("fp32", "FP32（全精度）"),
        ("int8", "INT8（量化，CPU推理最快）"),
    ], selected="fp16")

    return HTMLResponse(f"""
    {task_html}
    <div class="card">
        <h3>训练异常检测模型</h3>
        <p style="color:var(--text-secondary);font-size:var(--text-sm);margin-bottom:var(--space-4);">
            使用基线图片训练异常检测模型。训练完成后可直接部署到摄像头。
            训练耗时通常为 5-15 分钟，取决于图片数量和模型类型。
        </p>
        <form hx-post="/api/baseline/train" hx-target="#tab-content" hx-swap="innerHTML">
            {cam_select}
            <div class="form-row">
                {model_select}
                {export_select}
            </div>
            <div class="form-row">
                {quantization_select}
            </div>
            <div class="form-actions">
                <button type="submit" class="btn btn-primary">开始训练</button>
            </div>
        </form>
    </div>""")


@router.post("/train")
async def start_training(request: Request):
    """Start a model training background task."""
    task_manager = getattr(request.app.state, "task_manager", None)
    config = request.app.state.config

    if not task_manager or not config:
        return JSONResponse({"error": "服务不可用"}, status_code=503)

    form = await parse_request_form(request)
    camera_id = form.get("camera_id", "")
    model_type = form.get("model_type", "patchcore")
    export_format = form.get("export_format", "openvino")
    quantization = form.get("quantization", "fp16")
    resume_from = form.get("resume_from", "") or None
    skip_validation = form.get("skip_baseline_validation", "").lower() in ("true", "1", "yes")

    if not camera_id:
        return JSONResponse({"error": "请选择摄像头"}, status_code=400)

    # Validate quantization value
    if quantization not in ("fp32", "fp16", "int8"):
        quantization = "fp16"

    # Get database if available
    database = getattr(request.app.state, "database", None)
    database_url = None
    if database:
        database_url = database._database_url

    # Resolve anomaly config for the selected camera
    anomaly_config = None
    cam_cfg = next((c for c in config.cameras if c.camera_id == camera_id), None)
    if cam_cfg is not None:
        anomaly_config = cam_cfg.anomaly

    def _run_training(progress_callback, **kwargs):
        return _train_model_task(
            progress_callback,
            camera_id=camera_id,
            **kwargs,
        )

    try:
        task_manager.submit(
            "model_training",
            _run_training,
            camera_id=camera_id,
            baselines_dir=str(config.storage.baselines_dir),
            models_dir=str(config.storage.models_dir),
            model_type=model_type,
            export_format=export_format,
            quantization=quantization,
            database_url=database_url,
            anomaly_config=anomaly_config,
            resume_from=resume_from,
            skip_baseline_validation=skip_validation,
        )
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return HTMLResponse(
        '<div hx-get="/api/baseline/train" hx-trigger="load" hx-swap="innerHTML"></div>',
        headers=htmx_toast_headers("模型训练已启动"),
    )


@router.get("/models", response_class=HTMLResponse)
async def models_list(request: Request):
    """List trained models found in data/models/."""
    config = request.app.state.config
    if not config:
        return HTMLResponse(empty_state("配置不可用"))

    models_dir = Path(config.storage.models_dir)
    if not models_dir.exists():
        return HTMLResponse(empty_state("暂无已训练的模型", "请先在「模型训练」标签页中训练模型"))

    rows = ""
    for cam_dir in sorted(models_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        camera_id = cam_dir.name

        # Find model files
        for pattern, fmt in [("model.xml", "OpenVINO"), ("model.onnx", "ONNX"),
                              ("model.pt", "PyTorch"), ("model.ckpt", "Checkpoint")]:
            for model_file in sorted(cam_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(model_file.stat().st_mtime))
                size_mb = model_file.stat().st_size / (1024 * 1024)
                rel_path = str(model_file.relative_to(Path(".")))

                rows += f"""
                <tr>
                    <td>{camera_id}</td>
                    <td>{fmt}</td>
                    <td>{size_mb:.1f} MB</td>
                    <td>{mtime}</td>
                    <td class="mono" style="font-size:var(--text-xs);color:var(--text-tertiary);">{rel_path}</td>
                    <td>
                        <button class="btn btn-primary btn-sm"
                            hx-post="/api/baseline/deploy"
                            hx-vals='{{"camera_id": "{camera_id}", "model_path": "{rel_path}"}}'
                            hx-swap="none"
                            hx-confirm="确定将此模型部署到 {camera_id}？">
                            部署
                        </button>
                    </td>
                </tr>"""

    if not rows:
        return HTMLResponse(empty_state("暂无已训练的模型"))

    return HTMLResponse(f"""
    <div class="card">
        <h3>已训练模型</h3>
        <table>
            <thead><tr>
                <th>摄像头</th><th>格式</th><th>大小</th>
                <th>训练时间</th><th>路径</th><th>操作</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""")


@router.post("/deploy")
async def deploy_model(request: Request):
    """Deploy a trained model to a camera (hot-reload)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "不可用"}, status_code=503)

    data = await request.json()
    camera_id = data.get("camera_id", "")
    model_path = data.get("model_path", "")

    if not camera_id or not model_path:
        return JSONResponse({"error": "缺少参数"}, status_code=400)

    # Path safety
    models_dir = Path("data/models").resolve()
    resolved = Path(model_path).resolve()
    if not str(resolved).startswith(str(models_dir)):
        return JSONResponse({"error": "模型路径无效"}, status_code=400)
    if not resolved.exists():
        return JSONResponse({"error": "模型文件不存在"}, status_code=404)

    if camera_id not in camera_manager._pipelines:
        return JSONResponse({"error": f"摄像头 {camera_id} 不存在"}, status_code=404)

    # If model_path is a directory, find the actual model file inside it
    if resolved.is_dir():
        from argus.core.pipeline import DetectionPipeline

        model_file = DetectionPipeline._find_model(camera_id)
        if model_file is None:
            # Fallback: search inside the provided directory directly
            for pattern in ("model.xml", "model.pt", "model.ckpt"):
                matches = list(resolved.rglob(pattern))
                if matches:
                    model_file = matches[0]
                    break
        if model_file is None:
            return JSONResponse({"error": "模型目录中未找到模型文件"}, status_code=404)
        resolved = model_file

    registry_record = find_registered_model_by_path(
        request,
        resolved,
        camera_id=camera_id,
    )
    if registry_record is not None:
        _, success = activate_model_version(
            request,
            registry_record.model_version_id,
            triggered_by="dashboard",
        )
    else:
        success = camera_manager.reload_model(camera_id, str(resolved))

    if success:
        audit = getattr(request.app.state, "audit_logger", None)
        client_ip = request.client.host if request.client else ""
        if audit:
            audit.log(
                user="operator",
                action="deploy_model",
                target_type="camera",
                target_id=camera_id,
                detail=model_path,
                ip_address=client_ip,
            )
        return JSONResponse(
            {
                "status": "ok",
                "model_version_id": registry_record.model_version_id if registry_record else None,
            },
            headers=htmx_toast_headers("模型已部署"),
        )
    return JSONResponse({"error": "模型部署失败"}, status_code=500)


# ── Advanced Baseline Capture Job Endpoints ──


@router.post("/job")
async def start_capture_job(request: Request):
    """Start an advanced baseline capture job with strategy selection."""
    app = request.app
    camera_manager = app.state.camera_manager
    task_manager = app.state.task_manager
    config = app.state.config
    if not camera_manager or not task_manager or not config:
        return JSONResponse({"error": "服务不可用"}, status_code=503)

    form = await parse_request_form(request)
    camera_id = form.get("camera_id", "")
    session_label = form.get("session_label", "daytime")

    if not camera_id:
        return JSONResponse({"error": "请选择摄像头"}, status_code=400)

    if "count" in form or "interval" in form:
        target_frames = int(form.get("count", 100))
        interval = float(form.get("interval", 2.0))
        duration_hours = max(target_frames * interval / 3600, 1 / 3600)
        sampling_strategy = "uniform"
        diversity_threshold = config.baseline_capture.diversity_threshold
        frames_per_period = config.baseline_capture.frames_per_period
    else:
        target_frames = int(form.get("target_frames", 1000))
        duration_hours = float(form.get("duration_hours", 24.0))
        sampling_strategy = form.get("sampling_strategy", config.baseline_capture.default_strategy)
        diversity_threshold = float(
            form.get("diversity_threshold", config.baseline_capture.diversity_threshold)
        )
        frames_per_period = int(
            form.get("frames_per_period", config.baseline_capture.frames_per_period)
        )

    job_config = BaselineCaptureJobConfig(
        camera_id=camera_id,
        session_label=session_label,
        target_frames=target_frames,
        duration_hours=duration_hours,
        sampling_strategy=sampling_strategy,
        storage_path=str(config.storage.baselines_dir),
        quality_config=config.capture_quality,
        pause_on_anomaly_lock=config.baseline_capture.pause_on_anomaly_lock,
        diversity_threshold=diversity_threshold,
        dino_backbone=config.baseline_capture.dino_backbone,
        dino_image_size=config.baseline_capture.dino_image_size,
        active_sleep_min_seconds=config.baseline_capture.active_sleep_min_seconds,
        active_cpu_threads=config.baseline_capture.active_cpu_threads,
        schedule_periods=dict(config.baseline_capture.schedule_periods),
        frames_per_period=frames_per_period,
        post_capture_review=config.baseline_capture.post_capture_review,
        review_flag_percentile=config.baseline_capture.review_flag_percentile,
        models_dir=str(config.storage.models_dir),
        exports_dir=str(config.models.anomalib_export_dir),
    )

    pause_ev = threading.Event()
    pause_ev.set()  # not paused
    abort_ev = threading.Event()
    lifecycle = getattr(app.state, "baseline_lifecycle", None)

    try:
        task_id = task_manager.submit(
            "baseline_capture",
            run_baseline_capture_job,
            camera_id=camera_id,
            job_config=job_config,
            camera_manager=camera_manager,
            pause_event=pause_ev,
            abort_event=abort_ev,
            lifecycle=lifecycle,
        )
        # Sync events so TaskManager.pause_task/abort_task work
        task_info = task_manager.get_task(task_id)
        if task_info:
            task_info.pause_event = pause_ev
            task_info.abort_event = abort_ev
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=409)

    return JSONResponse({"task_id": task_id, "status": "submitted"})


@router.post("/capture")
async def start_capture_legacy(request: Request):
    """Backward-compatible alias for quick capture clients still posting to /capture."""
    return await start_capture_job(request)


@router.post("/job/{task_id}/pause")
async def pause_capture_job(task_id: str, request: Request):
    """Pause a running baseline capture job."""
    task_manager = request.app.state.task_manager
    if task_manager.pause_task(task_id):
        return JSONResponse({"task_id": task_id, "status": "paused"})
    return JSONResponse({"error": "Task not found or not running"}, status_code=404)


@router.post("/job/{task_id}/resume")
async def resume_capture_job(task_id: str, request: Request):
    """Resume a paused baseline capture job."""
    task_manager = request.app.state.task_manager
    if task_manager.resume_task(task_id):
        return JSONResponse({"task_id": task_id, "status": "resumed"})
    return JSONResponse({"error": "Task not found or not paused"}, status_code=404)


@router.post("/job/{task_id}/abort")
async def abort_capture_job(task_id: str, request: Request):
    """Abort a running or paused baseline capture job."""
    task_manager = request.app.state.task_manager
    if task_manager.abort_task(task_id):
        return JSONResponse({"task_id": task_id, "status": "aborting"})
    return JSONResponse({"error": "Task not found or not active"}, status_code=404)


@router.get("/job/{task_id}")
async def get_capture_job_status(task_id: str, request: Request):
    """Get the status of a baseline capture job."""
    task_manager = request.app.state.task_manager
    task = task_manager.get_task(task_id)
    if task is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({
        "task_id": task.task_id,
        "task_type": task.task_type,
        "camera_id": task.camera_id,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
        "error": task.error,
        "result": task.result if task.status.value in ("complete", "aborted", "failed") else None,
    })


# ── Helpers ──


_SESSION_LABELS = {"day": "白天", "night": "夜间", "maintenance": "检修", "other": "其他"}


def _build_coverage_hints(baselines_dir: Path, camera_ids: list[str]) -> str:
    """Build coverage hint HTML showing which session labels exist per camera (CAP-008)."""
    import json

    if not baselines_dir.exists():
        return ""

    hints = []
    for cam_id in camera_ids:
        cam_dir = baselines_dir / cam_id / "default"
        if not cam_dir.exists():
            continue

        existing_labels: dict[str, int] = {}
        for version_dir in sorted(cam_dir.iterdir()):
            meta_path = version_dir / "capture_meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                label = meta.get("session_label", "")
                accepted = meta.get("stats", {}).get("accepted", 0)
                if label:
                    existing_labels[label] = existing_labels.get(label, 0) + accepted
            except (json.JSONDecodeError, OSError):
                continue

        if not existing_labels:
            continue

        items = []
        for key, display in _SESSION_LABELS.items():
            if key in existing_labels:
                items.append(
                    f'<span style="color:#4caf50;">&#10003; {display} ({existing_labels[key]}帧)</span>'
                )
            else:
                items.append(f'<span style="color:#f44336;">&#10007; 缺少{display}</span>')

        hints.append(
            f'<div style="margin-bottom:4px;"><strong>{cam_id}:</strong> '
            + " &nbsp; ".join(items)
            + "</div>"
        )

    if not hints:
        return ""

    return (
        '<div class="card" style="border-left:3px solid #ff9800;margin-bottom:12px;">'
        '<h3 style="font-size:14px;">采集覆盖性检查</h3>'
        '<div style="font-size:12px;color:#8890a0;">'
        + "".join(hints)
        + "</div></div>"
    )



# ── Training history & reports (TRN-004/007) ──


@router.get("/training-history", response_class=HTMLResponse)
async def training_history(request: Request):
    """Show training history with quality grades."""
    database = getattr(request.app.state, "database", None)
    if not database:
        return HTMLResponse(empty_state("数据库不可用"))

    camera_id = request.query_params.get("camera_id")
    records = database.get_training_history(camera_id=camera_id, limit=50)

    if not records:
        return HTMLResponse(empty_state("暂无训练记录", "完成一次模型训练后，训练报告将显示在此处"))



    rows = ""
    for r in records:
        grade_color = _GRADE_COLORS.get(r.quality_grade, "#888")
        grade_badge = (
            f'<span style="color:{grade_color};font-weight:bold;font-size:16px;">'
            f'{r.quality_grade or "—"}</span>'
        )
        threshold_str = f"{r.threshold_recommended:.3f}" if r.threshold_recommended else "—"
        trained_at = r.trained_at.strftime("%Y-%m-%d %H:%M") if r.trained_at else "—"
        status_str = "完成" if r.status == "complete" else "失败"
        status_color = "#4caf50" if r.status == "complete" else "#f44336"

        rows += f"""
        <tr>
            <td>{r.camera_id}</td>
            <td>{r.model_type}</td>
            <td>{r.baseline_count}</td>
            <td>{r.train_count}/{r.val_count}</td>
            <td>{grade_badge}</td>
            <td>{threshold_str}</td>
            <td style="color:{status_color};">{status_str}</td>
            <td>{r.duration_seconds:.0f}s</td>
            <td>{trained_at}</td>
            <td>
                <button class="btn btn-ghost btn-sm"
                    hx-get="/api/baseline/training-report/{r.id}"
                    hx-target="#tab-content" hx-swap="innerHTML">
                    详情
                </button>
            </td>
        </tr>"""

    return HTMLResponse(f"""
    <div class="card">
        <h3>训练历史</h3>
        <table>
            <thead><tr>
                <th>摄像头</th><th>模型</th><th>基线数</th>
                <th>训练/验证</th><th>质量</th><th>推荐阈值</th>
                <th>状态</th><th>耗时</th><th>时间</th><th>操作</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""")


def _metric_bar(label: str, value: float, threshold: float, max_val: float = 1.0, lower_is_better: bool = False) -> str:
    """Render a metric with progress bar visualization."""
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    passed = (value <= threshold) if lower_is_better else (value >= threshold)
    color = "var(--status-ok)" if passed else "var(--status-critical)"
    icon = "&#10003;" if passed else "&#10007;"
    return f"""
    <div style="margin-bottom:var(--space-3);">
        <div class="flex-between" style="font-size:var(--text-sm);margin-bottom:var(--space-1);">
            <span style="color:var(--text-secondary);">{label}</span>
            <span style="color:{color};font-weight:var(--font-semibold);">{icon} {value:.4f}</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width:{pct:.0f}%;background:{color};"></div>
        </div>
    </div>"""


@router.get("/training-report/{record_id}", response_class=HTMLResponse)
async def training_report(request: Request, record_id: int):
    """Show detailed training report with metric bars and validation results."""
    database = getattr(request.app.state, "database", None)
    if not database:
        return HTMLResponse(empty_state("数据库不可用"))

    record = database.get_training_record(record_id)
    if not record:
        return HTMLResponse(empty_state("训练记录不存在"))

    back_btn = (
        '<button class="btn btn-ghost btn-sm mb-16" '
        'hx-get="/api/baseline/training-history" hx-target="#tab-content" hx-swap="innerHTML">'
        '&larr; 返回列表</button>'
    )

    grade_color = _GRADE_COLORS.get(record.quality_grade, "var(--text-tertiary)")

    # Pre-validation section
    pre_val_html = ""
    if record.corruption_rate is not None:
        pass_icon = "&#10003;" if record.pre_validation_passed else "&#10007;"
        pass_color = "var(--status-ok-text)" if record.pre_validation_passed else "var(--status-critical-text)"
        pre_val_html = f"""
        <div class="card">
            <h3>基线质量验证</h3>
            <table>
                <tr><td style="color:var(--text-secondary);">验证结果</td>
                    <td style="color:{pass_color};">{pass_icon} {"通过" if record.pre_validation_passed else "失败"}</td></tr>
                <tr><td style="color:var(--text-secondary);">损坏率</td><td>{record.corruption_rate:.1%}</td></tr>
                <tr><td style="color:var(--text-secondary);">近似重复率</td><td>{record.near_duplicate_rate:.1%}</td></tr>
                <tr><td style="color:var(--text-secondary);">亮度标准差</td><td>{record.brightness_std:.1f}</td></tr>
            </table>
        </div>"""

    # Score distribution with metric bars
    score_html = ""
    if record.val_score_mean is not None:
        score_html = f"""
        <div class="card">
            <h3>验证集分数分布</h3>
            {_metric_bar("均值 (越低越好)", record.val_score_mean, 0.3, max_val=1.0, lower_is_better=True)}
            {_metric_bar("标准差 (越低越好)", record.val_score_std, 0.1, max_val=0.5, lower_is_better=True)}
            {_metric_bar("最大值", record.val_score_max, 0.5, max_val=1.0, lower_is_better=True)}
            {_metric_bar("P95", record.val_score_p95, 0.4, max_val=1.0, lower_is_better=True)}
        </div>"""

    # Output validation section
    output_html = ""
    if record.checkpoint_valid is not None:
        def _check(val, label_ok, label_fail):
            if val:
                return f'<span style="color:var(--status-ok-text);">&#10003; {label_ok}</span>'
            return f'<span style="color:var(--status-critical-text);">&#10007; {label_fail}</span>'

        latency_str = f"{record.inference_latency_ms:.1f}ms" if record.inference_latency_ms else "—"
        output_html = f"""
        <div class="card">
            <h3>输出验证</h3>
            <table>
                <tr><td style="color:var(--text-secondary);">模型文件</td><td>{_check(record.checkpoint_valid, "有效", "无效")}</td></tr>
                <tr><td style="color:var(--text-secondary);">导出文件</td><td>{_check(record.export_valid, "有效", "无效/未导出")}</td></tr>
                <tr><td style="color:var(--text-secondary);">冒烟测试</td><td>{_check(record.smoke_test_passed, "通过", "失败")}</td></tr>
                <tr><td style="color:var(--text-secondary);">推理延迟</td><td>{latency_str}</td></tr>
            </table>
        </div>"""

    trained_at = record.trained_at.strftime("%Y-%m-%d %H:%M:%S") if record.trained_at else "—"
    threshold_str = f"{record.threshold_recommended:.4f}" if record.threshold_recommended else "—"
    error_html = (
        f'<div style="color:var(--status-critical-text);margin-top:var(--space-3);">'
        f'<strong>错误:</strong> {record.error}</div>'
        if record.error else ""
    )

    return HTMLResponse(f"""
    {back_btn}
    <div class="card" style="border-left:4px solid {grade_color};">
        <h3>训练报告 #{record.id}</h3>
        <div style="display:flex;align-items:center;gap:var(--space-5);margin-bottom:var(--space-4);">
            <span style="font-size:var(--text-4xl);font-weight:var(--font-semibold);color:{grade_color};line-height:1;">
                {record.quality_grade or "—"}
            </span>
            <div style="font-size:var(--text-sm);">
                <div><strong>摄像头:</strong> {record.camera_id}</div>
                <div><strong>模型:</strong> {record.model_type} | {record.export_format or "未导出"}</div>
                <div><strong>数据:</strong> {record.baseline_count} 基线 &rarr; {record.train_count} 训练 + {record.val_count} 验证</div>
                <div><strong>推荐阈值:</strong> <span class="mono">{threshold_str}</span></div>
                <div><strong>耗时:</strong> {record.duration_seconds:.0f} 秒 | {trained_at}</div>
            </div>
        </div>
        {error_html}
    </div>
    {pre_val_html}
    {score_html}
    {output_html}
    """)


@router.post("/compare")
async def compare_models_route(request: Request):
    """Compare two trained models (TRN-008)."""
    database = getattr(request.app.state, "database", None)
    if not database:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    data = await request.json()
    old_record_id = data.get("old_record_id")
    new_record_id = data.get("new_record_id")

    if not old_record_id or not new_record_id:
        return JSONResponse({"error": "缺少参数"}, status_code=400)

    old_record = database.get_training_record(int(old_record_id))
    new_record = database.get_training_record(int(new_record_id))

    if not old_record or not new_record:
        return JSONResponse({"error": "训练记录不存在"}, status_code=404)

    if not old_record.model_path or not new_record.model_path:
        return JSONResponse({"error": "模型路径不存在"}, status_code=400)

    # Find a validation set (prefer new record's baseline)
    config = request.app.state.config
    baselines_dir = Path(config.storage.baselines_dir)
    val_dir = baselines_dir / new_record.camera_id / new_record.zone_id
    if not val_dir.exists():
        return JSONResponse({"error": "验证集目录不存在"}, status_code=400)

    try:
        from argus.anomaly.baseline import BaselineManager
        from argus.anomaly.trainer import ModelTrainer

        bm = BaselineManager(baselines_dir=str(baselines_dir))
        trainer = ModelTrainer(baseline_manager=bm)

        # Find model files
        old_model = trainer._find_best_model_file(Path(old_record.model_path))
        new_model = trainer._find_best_model_file(Path(new_record.model_path))

        if not old_model or not new_model:
            return JSONResponse({"error": "无法找到模型文件"}, status_code=400)

        result = trainer.compare_models(old_model, new_model, val_dir)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# ── Background task functions ──


def _train_model_task(
    progress_callback, *, baselines_dir, models_dir, camera_id, model_type,
    export_format, quantization="fp16", database_url=None, anomaly_config=None,
    resume_from=None, skip_baseline_validation=False,
):
    """Train an anomaly detection model with full validation pipeline.

    Uses ModelTrainer with progress callback for UI updates.
    Saves TrainingRecord to database when available.
    """
    progress_callback(5, "正在验证基线数据...")

    try:
        from argus.anomaly.baseline import BaselineManager
        from argus.anomaly.trainer import ModelTrainer, TrainingStatus
    except ImportError:
        raise RuntimeError("Anomalib 未安装，无法训练模型")

    bm = BaselineManager(baselines_dir=baselines_dir)

    trainer = ModelTrainer(
        baseline_manager=bm,
        models_dir=models_dir,
        exports_dir=str(Path(models_dir).parent / "exports"),
    )

    # Train with progress callback
    fmt = export_format if export_format != "none" else None
    result = trainer.train(
        camera_id=camera_id,
        zone_id="default",
        model_type=model_type,
        export_format=fmt,
        quantization=quantization,
        progress_callback=progress_callback,
        anomaly_config=anomaly_config,
        resume_from=resume_from,
        skip_baseline_validation=skip_baseline_validation,
    )

    if result.status == TrainingStatus.FAILED:
        raise RuntimeError(result.error or "训练失败（未知原因）")

    # Save training record to database
    if database_url:
        try:
            from argus.storage.database import Database
            from argus.storage.model_registry import ModelRegistry
            from argus.storage.models import TrainingRecord

            db = Database(database_url=database_url)
            db.initialize()

            # Get baseline version
            baseline_dir = bm.get_baseline_dir(camera_id, "default")
            baseline_version = baseline_dir.name

            pre_val = result.pre_validation or {}
            output_val = result.output_validation or {}
            val_stats = result.val_stats or {}
            quality = result.quality_report

            record = TrainingRecord(
                camera_id=camera_id,
                zone_id="default",
                model_type=model_type,
                export_format=fmt,
                baseline_version=baseline_version,
                baseline_count=result.image_count,
                train_count=result.train_count,
                val_count=result.val_count,
                pre_validation_passed=pre_val.get("passed", True),
                corruption_rate=pre_val.get("corruption_rate"),
                near_duplicate_rate=pre_val.get("near_duplicate_rate"),
                brightness_std=pre_val.get("brightness_std"),
                val_score_mean=val_stats.get("mean"),
                val_score_std=val_stats.get("std"),
                val_score_max=val_stats.get("max"),
                val_score_p95=val_stats.get("p95"),
                quality_grade=quality.grade if quality else None,
                threshold_recommended=result.threshold_recommended,
                model_path=result.model_path,
                export_path=str(Path(models_dir).parent / "exports" / camera_id / "default") if fmt else None,
                checkpoint_valid=output_val.get("checkpoint_valid"),
                export_valid=output_val.get("export_valid"),
                smoke_test_passed=output_val.get("smoke_test_passed"),
                inference_latency_ms=output_val.get("inference_latency_ms"),
                status=result.status.value,
                error=result.error,
                duration_seconds=result.duration_seconds,
                trained_at=datetime.now(timezone.utc),
            )
            db.save_training_record(record)

            if result.model_path and not result.model_version_id:
                registry = ModelRegistry(session_factory=db.get_session)
                result.model_version_id = registry.register(
                    model_path=result.model_path,
                    baseline_dir=baseline_dir,
                    camera_id=camera_id,
                    model_type=model_type,
                    training_params={
                        "export_format": fmt,
                        "quantization": quantization,
                        "train_count": result.train_count,
                        "val_count": result.val_count,
                        "quality_grade": result.quality_report.grade if result.quality_report else None,
                    },
                )
            db.close()
        except Exception as e:
            logger.warning("training.record_save_failed", error=str(e))

    grade_str = f" | 质量: {result.quality_report.grade}" if result.quality_report else ""
    threshold_str = f" | 推荐阈值: {result.threshold_recommended:.3f}" if result.threshold_recommended else ""
    progress_callback(100, f"训练完成 — 耗时 {result.duration_seconds:.0f} 秒{grade_str}{threshold_str}")

    return {
        "model_path": result.model_path,
        "model_version_id": result.model_version_id,
        "status": result.status.value,
        "duration": result.duration_seconds,
        "image_count": result.image_count,
        "train_count": result.train_count,
        "val_count": result.val_count,
        "quality_grade": result.quality_report.grade if result.quality_report else None,
        "threshold_recommended": result.threshold_recommended,
    }


@router.post("/optimize", response_class=HTMLResponse)
async def optimize_baseline(request: Request):
    """Optimize baseline by selecting most diverse subset (A4-2)."""
    form = await parse_request_form(request)
    camera_id = form.get("camera_id", "")
    target_ratio = float(form.get("target_ratio", 0.2))

    app = request.app
    baseline_mgr = getattr(app.state, "baseline_manager", None)
    if baseline_mgr is None:
        return HTMLResponse('<div class="error">基线管理器未初始化</div>')

    baseline_dir = baseline_mgr.get_baseline_dir(camera_id, "default")
    all_images = sorted(
        list(baseline_dir.glob("*.png")) + list(baseline_dir.glob("*.jpg"))
    )
    if not all_images:
        return HTMLResponse('<div class="error">未找到基线图片</div>')

    target_count = max(30, int(len(all_images) * target_ratio))
    selected = baseline_mgr.diversity_select(baseline_dir, target_count)
    selected_set = set(selected)

    # Move unselected to backup directory
    backup_dir = baseline_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    moved = 0
    for img_path in all_images:
        if img_path not in selected_set:
            import shutil
            shutil.move(str(img_path), str(backup_dir / img_path.name))
            moved += 1

    return HTMLResponse(
        f'<div class="success">优化完成: 保留 {len(selected)} 张, '
        f'移除 {moved} 张到 backup/</div>'
    )


def _get_baseline_manager(request: Request):
    """Get or create a BaselineManager from app state or config."""
    baseline_mgr = getattr(request.app.state, "baseline_manager", None)
    if baseline_mgr is not None:
        return baseline_mgr
    config = request.app.state.config
    if config:
        from argus.anomaly.baseline import BaselineManager
        return BaselineManager(baselines_dir=str(config.storage.baselines_dir))
    return None


@router.get("/optimize/preview")
async def optimize_preview(
    request: Request,
    camera_id: str = "",
    zone_id: str = "default",
    target_ratio: float = 0.2,
):
    """Preview how many images would be kept/moved by optimization."""
    if not camera_id:
        return JSONResponse({"error": "camera_id is required"}, status_code=400)

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return JSONResponse({"error": "基线管理器不可用"}, status_code=503)

    baseline_dir = baseline_mgr.get_baseline_dir(camera_id, zone_id)
    all_images = sorted(
        list(baseline_dir.glob("*.png")) + list(baseline_dir.glob("*.jpg"))
    )
    total = len(all_images)
    if total == 0:
        return JSONResponse({"total": 0, "keep": 0, "move": 0})

    target_count = max(30, int(total * target_ratio))
    keep = min(target_count, total)
    return JSONResponse({"total": total, "keep": keep, "move": total - keep})


@router.post("/optimize/json")
async def optimize_baseline_json(request: Request):
    """JSON API: Optimize baseline by selecting most diverse subset (A4-2).

    Params (JSON body):
        camera_id: str (required)
        zone_id: str (default "default")
        target_ratio: float (default 0.2)

    Returns: {"selected": N, "moved": M, "backup_dir": "..."}
    """
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    target_ratio = float(data.get("target_ratio", 0.2))

    if not camera_id:
        return JSONResponse({"error": "camera_id is required"}, status_code=400)

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return JSONResponse({"error": "基线管理器不可用"}, status_code=503)

    baseline_dir = baseline_mgr.get_baseline_dir(camera_id, zone_id)
    all_images = sorted(
        list(baseline_dir.glob("*.png")) + list(baseline_dir.glob("*.jpg"))
    )
    if not all_images:
        return JSONResponse({"error": "未找到基线图片"}, status_code=404)

    target_count = max(30, int(len(all_images) * target_ratio))
    selected = baseline_mgr.diversity_select(baseline_dir, target_count)
    selected_set = set(selected)

    # Move unselected to backup directory
    backup_dir = baseline_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    moved = 0
    for img_path in all_images:
        if img_path not in selected_set:
            shutil.move(str(img_path), str(backup_dir / img_path.name))
            moved += 1

    logger.info(
        "baseline.optimized",
        camera_id=camera_id,
        zone_id=zone_id,
        selected=len(selected),
        moved=moved,
    )

    # Audit trail
    audit = getattr(request.app.state, "audit_logger", None)
    if audit:
        client_ip = request.client.host if request.client else ""
        audit.log(
            user="operator",
            action="optimize_baseline",
            target_type="baseline",
            target_id=f"{camera_id}/{zone_id}",
            detail=f"保留 {len(selected)} 张, 移除 {moved} 张",
            ip_address=client_ip,
        )

    return JSONResponse({
        "selected": len(selected),
        "moved": moved,
        "backup_dir": str(backup_dir),
    })


# ── Camera Group Endpoints ──


@router.get("/groups/json")
async def camera_groups_json(request: Request):
    """JSON API: list configured camera groups with baseline status."""
    config = request.app.state.config
    if not config:
        return JSONResponse({"groups": []})

    groups = []
    baseline_mgr = _get_baseline_manager(request)
    for grp in getattr(config, "camera_groups", []):
        image_count = 0
        current_version = None
        if baseline_mgr:
            group_dir = baseline_mgr.get_group_baseline_dir(grp.group_id, grp.zone_id)
            if group_dir.is_dir():
                image_count = (
                    len(list(group_dir.glob("*.png")))
                    + len(list(group_dir.glob("*.jpg")))
                )
                current_version = group_dir.name
        groups.append({
            "group_id": grp.group_id,
            "name": grp.name,
            "camera_ids": grp.camera_ids,
            "zone_id": grp.zone_id,
            "image_count": image_count,
            "current_version": current_version,
        })
    return JSONResponse({"groups": groups})


@router.post("/groups/merge")
async def merge_group_baseline(request: Request):
    """Merge baselines from member cameras into a group baseline version."""
    data = await request.json()
    group_id = data.get("group_id", "")
    zone_id = data.get("zone_id", "default")
    target_count = data.get("target_count")

    if not group_id:
        return JSONResponse({"error": "group_id is required"}, status_code=400)

    config = request.app.state.config
    if not config:
        return JSONResponse({"error": "配置不可用"}, status_code=503)

    group_cfg = next(
        (g for g in getattr(config, "camera_groups", []) if g.group_id == group_id),
        None,
    )
    if group_cfg is None:
        return JSONResponse({"error": f"Group {group_id} not found"}, status_code=404)

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return JSONResponse({"error": "基线管理器不可用"}, status_code=503)

    version_dir = baseline_mgr.merge_camera_baselines(
        group_id=group_id,
        camera_ids=group_cfg.camera_ids,
        zone_id=zone_id,
        target_count=int(target_count) if target_count else None,
    )

    image_count = (
        len(list(version_dir.glob("*.png")))
        + len(list(version_dir.glob("*.jpg")))
    )

    return JSONResponse({
        "status": "ok",
        "group_id": group_id,
        "version": version_dir.name,
        "image_count": image_count,
    })


# ── False Positive Merge Endpoint ──


@router.post("/merge-fp")
async def merge_false_positives(request: Request):
    """Merge false positive candidate pool into a new baseline version (Draft state)."""
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    max_fp_images = data.get("max_fp_images")

    if not camera_id:
        return JSONResponse({"error": "camera_id is required"}, status_code=400)

    feedback_mgr = getattr(request.app.state, "feedback_manager", None)
    if feedback_mgr is None:
        return JSONResponse({"error": "反馈���理器未初始化"}, status_code=503)

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return JSONResponse({"error": "基线管理器不可用"}, status_code=503)

    result = feedback_mgr.merge_fp_into_baseline(
        camera_id=camera_id,
        zone_id=zone_id,
        baseline_manager=baseline_mgr,
        max_fp_images=int(max_fp_images) if max_fp_images else None,
    )

    if "error" in result:
        return JSONResponse(result, status_code=400)

    return JSONResponse({"status": "ok", **result})


# ── Baseline Lifecycle Endpoints ──


def _get_lifecycle(request: Request):
    """Get BaselineLifecycle from app state."""
    return getattr(request.app.state, "baseline_lifecycle", None)


@router.get("/versions/json")
async def baseline_versions_json(request: Request):
    """JSON API: list baseline versions with lifecycle state."""
    camera_id = request.query_params.get("camera_id", "")
    zone_id = request.query_params.get("zone_id", "default")
    if not camera_id:
        return JSONResponse({"error": "camera_id is required"}, status_code=400)

    lifecycle = _get_lifecycle(request)
    if lifecycle is None:
        return JSONResponse({"error": "生命周期管理器未初始化"}, status_code=503)

    versions = lifecycle.get_versions(camera_id, zone_id)
    return JSONResponse({"versions": [v.to_dict() for v in versions]})


@router.post("/verify")
async def verify_baseline(request: Request):
    """Verify a baseline version (Draft -> Verified)."""
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    version = data.get("version", "")
    verified_by = data.get("verified_by", "")
    verified_by_secondary = data.get("verified_by_secondary")

    if not camera_id or not version or not verified_by:
        return JSONResponse(
            {"error": "camera_id, version, verified_by are required"}, status_code=400
        )

    lifecycle = _get_lifecycle(request)
    if lifecycle is None:
        return JSONResponse({"error": "生命周期管理器未初始化"}, status_code=503)

    try:
        client_ip = request.client.host if request.client else ""
        record = lifecycle.verify(
            camera_id, zone_id, version,
            verified_by=verified_by,
            verified_by_secondary=verified_by_secondary,
            ip_address=client_ip,
        )
        return JSONResponse({"status": "ok", "version": record.to_dict()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.post("/activate-baseline")
async def activate_baseline(request: Request):
    """Activate a baseline version (Verified -> Active). Auto-retires previous."""
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    version = data.get("version", "")
    user = data.get("user", "operator")

    if not camera_id or not version:
        return JSONResponse(
            {"error": "camera_id and version are required"}, status_code=400
        )

    lifecycle = _get_lifecycle(request)
    if lifecycle is None:
        return JSONResponse({"error": "生命周期管理器未初始化"}, status_code=503)

    try:
        client_ip = request.client.host if request.client else ""
        record = lifecycle.activate(
            camera_id, zone_id, version, user=user, ip_address=client_ip,
        )
        return JSONResponse({"status": "ok", "version": record.to_dict()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.post("/retire")
async def retire_baseline(request: Request):
    """Retire a baseline version (Active -> Retired)."""
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    version = data.get("version", "")
    user = data.get("user", "operator")
    reason = data.get("reason", "")

    if not camera_id or not version:
        return JSONResponse(
            {"error": "camera_id and version are required"}, status_code=400
        )

    lifecycle = _get_lifecycle(request)
    if lifecycle is None:
        return JSONResponse({"error": "生命周期管理器未初始化"}, status_code=503)

    try:
        client_ip = request.client.host if request.client else ""
        record = lifecycle.retire(
            camera_id, zone_id, version,
            user=user, reason=reason, ip_address=client_ip,
        )
        return JSONResponse({"status": "ok", "version": record.to_dict()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
