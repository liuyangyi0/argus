"""Baseline and model management routes (Chinese UI)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from argus.dashboard.components import (
    confirm_button,
    empty_state,
    form_group,
    form_select,
    page_header,
    progress_bar,
    tab_bar,
)

logger = structlog.get_logger()

router = APIRouter()

_TABS = [
    ("capture", "基线采集", "/api/baseline/capture"),
    ("browse", "基线浏览", "/api/baseline/list"),
    ("train", "模型训练", "/api/baseline/train"),
    ("models", "模型管理", "/api/baseline/models"),
]


@router.get("", response_class=HTMLResponse)
async def baseline_main(request: Request):
    """Baseline & model management page with tabs."""
    header = page_header("基线与模型管理", "采集基线图片、训练检测模型、部署到摄像头")
    tabs = tab_bar(_TABS, "capture")
    # Load first tab content on page load
    return HTMLResponse(f"""
    {header}
    {tabs}
    <script>
        // Auto-load first tab
        document.addEventListener('DOMContentLoaded', function() {{
            htmx.ajax('GET', '/api/baseline/capture', {{target: '#tab-content', swap: 'innerHTML'}});
        }});
    </script>""")


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

    # Show active capture tasks
    task_html = ""
    task_manager = getattr(request.app.state, "task_manager", None)
    if task_manager:
        active = [t for t in task_manager.get_active_tasks()
                  if t.task_type == "baseline_capture" and t.status.value in ("pending", "running")]
        if active:
            task_html = '<div class="card" style="border-left:3px solid #4fc3f7;"><h3>正在采集</h3>'
            task_html += f'<div hx-get="/api/tasks/{active[0].task_id}" hx-trigger="every 1s" hx-swap="outerHTML"></div>'
            task_html += '</div>'

    return HTMLResponse(f"""
    {task_html}
    <div class="card">
        <h3>开始基线采集</h3>
        <p style="color:#8890a0;font-size:13px;margin-bottom:16px;">
            从在线摄像头采集"正常"场景的参考图片，用于训练异常检测模型。
            确保采集期间场景中没有异物。
        </p>
        <form hx-post="/api/baseline/capture" hx-target="#tab-content" hx-swap="innerHTML">
            {cam_select}
            <div class="form-row">
                {count_input}
                {interval_input}
            </div>
            <div class="form-actions">
                <button type="submit" class="btn btn-primary">开始采集</button>
            </div>
        </form>
    </div>""")


@router.post("/capture")
async def start_capture(request: Request):
    """Start a baseline capture background task."""
    task_manager = getattr(request.app.state, "task_manager", None)
    camera_manager = request.app.state.camera_manager
    config = request.app.state.config

    if not task_manager or not camera_manager or not config:
        return JSONResponse({"error": "服务不可用"}, status_code=503)

    form = await request.form()
    camera_id = form.get("camera_id", "")
    count = int(form.get("count", 100))
    interval = float(form.get("interval", 2.0))

    if not camera_id:
        return JSONResponse({"error": "请选择摄像头"}, status_code=400)

    baselines_dir = str(config.storage.baselines_dir)

    try:
        task_id = task_manager.submit(
            "baseline_capture",
            _capture_baseline_task,
            camera_id=camera_id,
            camera_manager=camera_manager,
            output_dir=baselines_dir,
            count=count,
            interval=interval,
        )
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Return the capture tab with task status
    return HTMLResponse(
        f'<div hx-get="/api/baseline/capture" hx-trigger="load" hx-swap="innerHTML"></div>',
        headers={"HX-Trigger": '{"showToast": {"message": "基线采集已启动", "type": "success"}}'},
    )


@router.get("/list", response_class=HTMLResponse)
async def baseline_list(request: Request):
    """List baselines by camera with version info."""
    config = request.app.state.config
    if not config:
        return HTMLResponse(empty_state("配置不可用"))

    baselines_dir = Path(config.storage.baselines_dir)
    if not baselines_dir.exists():
        return HTMLResponse(empty_state("暂无基线数据", "请先采集基线图片"))

    html = ""
    for cam_dir in sorted(baselines_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        camera_id = cam_dir.name
        versions = []
        for ver_dir in sorted(cam_dir.iterdir()):
            if not ver_dir.is_dir():
                continue
            images = list(ver_dir.glob("*.png")) + list(ver_dir.glob("*.jpg"))
            versions.append((ver_dir.name, len(images)))

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


@router.get("/{camera_id}/images", response_class=HTMLResponse)
async def baseline_images(request: Request, camera_id: str):
    """Show baseline image thumbnails for a camera version."""
    config = request.app.state.config
    version = request.query_params.get("version", "default")
    baselines_dir = Path(config.storage.baselines_dir) / camera_id / version

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
            f'<img src="/api/baseline/{camera_id}/image/{img.name}?version={version}" '
            f'alt="{img.name}" title="{img.name}">'
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
async def baseline_image(request: Request, camera_id: str, filename: str):
    """Serve a single baseline image."""
    config = request.app.state.config
    version = request.query_params.get("version", "default")
    baselines_dir = Path(config.storage.baselines_dir)

    # Path safety
    img_path = (baselines_dir / camera_id / version / filename).resolve()
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


@router.get("/train", response_class=HTMLResponse)
async def train_form(request: Request):
    """Model training form."""
    config = request.app.state.config
    camera_manager = request.app.state.camera_manager

    if not config or not camera_manager:
        return HTMLResponse(empty_state("服务不可用"))

    # List cameras that have baselines
    baselines_dir = Path(config.storage.baselines_dir)
    cam_options = []
    for cam_dir in sorted(baselines_dir.iterdir()) if baselines_dir.exists() else []:
        if cam_dir.is_dir():
            img_count = len(list(cam_dir.rglob("*.png"))) + len(list(cam_dir.rglob("*.jpg")))
            if img_count >= 10:
                cam_options.append((cam_dir.name, f"{cam_dir.name} ({img_count} 张基线图片)"))

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
            task_html = '<div class="card" style="border-left:3px solid #4fc3f7;"><h3>正在训练</h3>'
            task_html += f'<div hx-get="/api/tasks/{active[0].task_id}" hx-trigger="every 1s" hx-swap="outerHTML"></div>'
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

    return HTMLResponse(f"""
    {task_html}
    <div class="card">
        <h3>训练异常检测模型</h3>
        <p style="color:#8890a0;font-size:13px;margin-bottom:16px;">
            使用基线图片训练异常检测模型。训练完成后可直接部署到摄像头。
            训练耗时通常为 5-15 分钟，取决于图片数量和模型类型。
        </p>
        <form hx-post="/api/baseline/train" hx-target="#tab-content" hx-swap="innerHTML">
            {cam_select}
            <div class="form-row">
                {model_select}
                {export_select}
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

    form = await request.form()
    camera_id = form.get("camera_id", "")
    model_type = form.get("model_type", "patchcore")
    export_format = form.get("export_format", "openvino")

    if not camera_id:
        return JSONResponse({"error": "请选择摄像头"}, status_code=400)

    try:
        task_manager.submit(
            "model_training",
            _train_model_task,
            camera_id=camera_id,
            baselines_dir=str(config.storage.baselines_dir),
            models_dir=str(config.storage.models_dir),
            model_type=model_type,
            export_format=export_format,
        )
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return HTMLResponse(
        '<div hx-get="/api/baseline/train" hx-trigger="load" hx-swap="innerHTML"></div>',
        headers={"HX-Trigger": '{"showToast": {"message": "模型训练已启动", "type": "success"}}'},
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
                    <td style="font-size:12px;color:#8890a0;">{rel_path}</td>
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

    pipeline = camera_manager._pipelines.get(camera_id)
    if not pipeline:
        return JSONResponse({"error": f"摄像头 {camera_id} 不存在"}, status_code=404)

    success = pipeline.reload_anomaly_model(str(resolved))

    if success:
        return JSONResponse(
            {"status": "ok"},
            headers={"HX-Trigger": '{"showToast": {"message": "模型已部署", "type": "success"}}'},
        )
    return JSONResponse({"error": "模型部署失败"}, status_code=500)


# ── Background task functions ──


def _capture_baseline_task(
    progress_callback, *, camera_manager, camera_id, output_dir, count, interval
):
    """Capture raw (unmasked) baseline frames from a running camera.

    Uses get_raw_frame() to get frames BEFORE zone masking is applied,
    ensuring training data is clean of zone mask artifacts (CRIT-05).
    Uses BaselineManager-compatible directory structure (HIGH-07).
    """
    from argus.anomaly.baseline import BaselineManager

    bm = BaselineManager(baselines_dir=output_dir)
    version_dir = bm.create_new_version(camera_id, "default")

    captured = 0
    skipped_none = 0
    for i in range(count):
        # Use raw frame (before zone mask) for training data
        frame = camera_manager.get_raw_frame(camera_id)
        if frame is None:
            # Fallback to latest processed frame if raw not available
            frame = camera_manager.get_latest_frame(camera_id)

        if frame is not None:
            path = version_dir / f"baseline_{captured:05d}.png"
            cv2.imwrite(str(path), frame)
            captured += 1
        else:
            skipped_none += 1

        progress_callback(
            int((i + 1) / count * 100),
            f"已采集 {captured}/{count} 帧 (跳过 {skipped_none})",
        )
        time.sleep(interval)

    # Set as current version
    if captured > 0:
        bm.set_current_version(camera_id, "default", version_dir.name)

    logger.info(
        "baseline.capture_complete",
        camera_id=camera_id, captured=captured, skipped=skipped_none, total=count,
    )
    return {"captured": captured, "skipped": skipped_none, "total": count, "output_dir": str(version_dir)}


def _train_model_task(
    progress_callback, *, baselines_dir, models_dir, camera_id, model_type, export_format
):
    """Train an anomaly detection model using the real ModelTrainer API (CRIT-01 fix).

    Uses BaselineManager + ModelTrainer with correct constructor signatures.
    """
    progress_callback(5, "正在验证基线数据...")

    try:
        from argus.anomaly.baseline import BaselineManager
        from argus.anomaly.trainer import ModelTrainer, TrainingStatus
    except ImportError:
        raise RuntimeError("Anomalib 未安装，无法训练模型")

    bm = BaselineManager(baselines_dir=baselines_dir)
    image_count = bm.count_images(camera_id, "default")

    if image_count < 30:
        raise ValueError(f"基线图片不足: 需要至少 30 张，实际 {image_count} 张")

    progress_callback(10, f"找到 {image_count} 张基线图片，正在训练 {model_type} 模型...")

    trainer = ModelTrainer(
        baseline_manager=bm,
        models_dir=models_dir,
        exports_dir=str(Path(models_dir).parent / "exports"),
    )

    # Train (blocking call)
    fmt = export_format if export_format != "none" else None
    result = trainer.train(
        camera_id=camera_id,
        zone_id="default",
        model_type=model_type,
        export_format=fmt,
    )

    if result.status == TrainingStatus.FAILED:
        raise RuntimeError(result.error or "训练失败（未知原因）")

    progress_callback(100, f"训练完成 — 耗时 {result.duration_seconds:.0f} 秒")
    return {
        "model_path": result.model_path,
        "status": result.status.value,
        "duration": result.duration_seconds,
        "image_count": result.image_count,
    }
