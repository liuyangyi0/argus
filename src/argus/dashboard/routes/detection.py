"""Detection debug view and diagnostics API routes (DET-003).

Provides a real-time debug panel showing per-frame pipeline processing
results, detector status, sensitivity preview, and pipeline mode control.
"""

from __future__ import annotations

from dataclasses import asdict

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from argus.dashboard.components import (
    empty_state,
    form_select,
    page_header,
    progress_bar,
    stat_card,
    status_badge,
)

logger = structlog.get_logger()

router = APIRouter()


@router.get("", response_class=HTMLResponse)
async def detection_debug_page(request: Request):
    """Detection debug view main page with camera selector."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse(empty_state("系统未启动"))

    statuses = camera_manager.get_status()
    if not statuses:
        return HTMLResponse(empty_state("无摄像头配置"))

    # Camera selector dropdown
    options = [(s.camera_id, f"{s.camera_id} - {s.name}") for s in statuses]
    first_cam = statuses[0].camera_id

    header = page_header("检测调试", "实时查看管线各阶段处理结果")

    return HTMLResponse(f"""
    {header}
    <div class="flex gap-16 mb-16">
        <div class="form-group" style="min-width:300px;">
            <label class="form-label">选择摄像头</label>
            <select id="debug-camera-select" class="form-select"
                    onchange="updateDebugCamera(this.value)">
                {''.join(f'<option value="{v}">{t}</option>' for v, t in options)}
            </select>
        </div>
    </div>

    <div class="grid-2 gap-16">
        <!-- Left: Detector status + Mode control -->
        <div>
            <div id="detector-status"
                 hx-get="/api/detection/detector-status/{first_cam}"
                 hx-trigger="load, every 3s"
                 hx-swap="innerHTML">
            </div>
            <div id="mode-control" class="mt-16"
                 hx-get="/api/detection/mode-control/{first_cam}"
                 hx-trigger="load, every 5s"
                 hx-swap="innerHTML">
            </div>
        </div>

        <!-- Right: Sensitivity preview -->
        <div>
            <div id="sensitivity-preview"
                 hx-get="/api/detection/sensitivity-preview/{first_cam}"
                 hx-trigger="load"
                 hx-swap="innerHTML">
            </div>
        </div>
    </div>

    <!-- Frame log table -->
    <div class="mt-16">
        <div id="frame-log"
             hx-get="/api/detection/frame-log/{first_cam}"
             hx-trigger="load, every 3s"
             hx-swap="innerHTML">
        </div>
    </div>

    <script>
    function updateDebugCamera(camId) {{
        // Update all HTMX targets with new camera
        const targets = [
            ['detector-status', '/api/detection/detector-status/' + camId],
            ['mode-control', '/api/detection/mode-control/' + camId],
            ['sensitivity-preview', '/api/detection/sensitivity-preview/' + camId],
            ['frame-log', '/api/detection/frame-log/' + camId],
        ];
        targets.forEach(([id, url]) => {{
            const el = document.getElementById(id);
            if (el) {{
                el.setAttribute('hx-get', url);
                htmx.ajax('GET', url, {{target: '#' + id, swap: 'innerHTML'}});
            }}
        }});
    }}
    </script>
    """)


@router.get("/detector-status/{camera_id}", response_class=HTMLResponse)
async def detector_status(request: Request, camera_id: str):
    """Detector status card showing model info and calibration progress (DET-004)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse("")

    status = camera_manager.get_detector_status(camera_id)
    if not status:
        return HTMLResponse(empty_state("摄像头未运行"))

    mode_label = "Anomalib 模型" if status["mode"] == "anomalib" else "SSIM 回退模式"
    mode_color = "#4caf50" if status["mode"] == "anomalib" else "#ff9800"

    model_info = ""
    if status["model_path"]:
        model_info = f'<div class="form-hint">模型路径: {status["model_path"]}</div>'

    calibration_html = ""
    if status["mode"] == "ssim_fallback":
        pct = int(status["ssim_calibration_progress"] * 100)
        cal_status = "complete" if status["ssim_calibrated"] else "running"
        calibration_html = f"""
        <div class="mt-8">
            <div class="form-label">SSIM 校准进度</div>
            {progress_bar(pct, cal_status)}
            <div class="form-hint">
                {'校准完成' if status["ssim_calibrated"] else f'校准中 ({pct}%)'}
                {f' | 噪声基底: {status["ssim_noise_floor"]:.4f}' if status["ssim_noise_floor"] is not None else ''}
            </div>
        </div>"""

    # Learning progress
    learning = camera_manager.get_learning_progress(camera_id)
    learning_html = ""
    if learning and learning.get("active"):
        lpct = int(learning["progress"] * 100)
        learning_html = f"""
        <div class="mt-8">
            <div class="form-label">学习模式进度</div>
            {progress_bar(lpct, "running")}
            <div class="form-hint">
                已运行 {learning["elapsed_seconds"]:.0f}s / {learning["total_seconds"]:.0f}s
            </div>
        </div>"""

    return HTMLResponse(f"""
    <div class="card">
        <h3>检测器状态</h3>
        <div class="flex gap-16 mt-8">
            {stat_card("检测模式", mode_label, mode_color)}
            {stat_card("异常阈值", f'{status["threshold"]:.2f}', "#4fc3f7")}
        </div>
        {model_info}
        {calibration_html}
        {learning_html}
    </div>""")


@router.get("/mode-control/{camera_id}", response_class=HTMLResponse)
async def mode_control(request: Request, camera_id: str):
    """Pipeline mode selector (DET-006)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse("")

    current_mode = camera_manager.get_pipeline_mode(camera_id)
    if current_mode is None:
        return HTMLResponse(empty_state("摄像头未运行"))

    modes = [
        ("active", "正常检测 (ACTIVE)"),
        ("maintenance", "维护模式 (MAINTENANCE)"),
        ("learning", "学习模式 (LEARNING)"),
    ]

    mode_descriptions = {
        "active": "正常检测与告警",
        "maintenance": "MOG2 冻结，检测继续，防止背景吸收",
        "learning": "全管线运行但抑制告警",
    }

    return HTMLResponse(f"""
    <div class="card">
        <h3>管线模式</h3>
        <div class="form-group mt-8">
            <select class="form-select"
                    hx-post="/api/detection/set-mode/{camera_id}"
                    hx-swap="none"
                    name="mode">
                {''.join(
                    f'<option value="{v}"{" selected" if v == current_mode else ""}>{t}</option>'
                    for v, t in modes
                )}
            </select>
            <div class="form-hint">{mode_descriptions.get(current_mode, "")}</div>
        </div>
    </div>""")


@router.post("/set-mode/{camera_id}")
async def set_mode(request: Request, camera_id: str):
    """Set pipeline operating mode (DET-006)."""
    from argus.core.pipeline import PipelineMode

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse("", status_code=404)

    form = await request.form()
    mode_str = form.get("mode", "active")
    try:
        mode = PipelineMode(mode_str)
    except ValueError:
        return HTMLResponse("无效模式", status_code=400)

    success = camera_manager.set_pipeline_mode(camera_id, mode)
    if not success:
        return HTMLResponse("摄像头未运行", status_code=404)

    return HTMLResponse(
        "",
        headers={"HX-Trigger": '{"showToast": "模式已切换"}'},
    )


@router.get("/sensitivity-preview/{camera_id}", response_class=HTMLResponse)
async def sensitivity_preview(request: Request, camera_id: str):
    """Preview how a new threshold would affect alert counts (DET-005)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse("")

    threshold = float(request.query_params.get("threshold", "0.7"))
    result = camera_manager.evaluate_threshold(camera_id, threshold)

    if result is None or result["total_frames"] == 0:
        return HTMLResponse(f"""
        <div class="card">
            <h3>灵敏度预览</h3>
            <div class="form-group mt-8">
                <label class="form-label">预览阈值</label>
                <div class="flex gap-8">
                    <input type="number" class="form-input" name="threshold"
                           value="{threshold}" min="0.1" max="1.0" step="0.05"
                           style="max-width:120px;">
                    <button class="btn btn-ghost btn-sm"
                            hx-get="/api/detection/sensitivity-preview/{camera_id}"
                            hx-include="[name='threshold']"
                            hx-target="#sensitivity-preview"
                            hx-swap="innerHTML">预览</button>
                </div>
            </div>
            <div class="empty-state"><div class="message">暂无分数数据（需要更多帧处理）</div></div>
        </div>""")

    total = result["total_frames"]
    would = result["would_alert_count"]
    current = result["current_alert_count"]
    dist = result["score_distribution"]

    # Build simple bar chart
    max_count = max(dist) if dist else 1
    bars_html = ""
    for i, count in enumerate(dist):
        height = int((count / max_count) * 60) if max_count > 0 else 0
        label = f"{i / 10:.1f}"
        is_above = (i / 10) >= threshold
        color = "#f44336" if is_above else "#4caf50"
        bars_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;flex:1;">
            <div style="height:60px;display:flex;align-items:flex-end;">
                <div style="width:100%;height:{height}px;background:{color};border-radius:2px 2px 0 0;min-width:12px;"></div>
            </div>
            <div style="font-size:10px;color:#8890a0;margin-top:2px;">{label}</div>
        </div>"""

    return HTMLResponse(f"""
    <div class="card">
        <h3>灵敏度预览</h3>
        <div class="form-group mt-8">
            <label class="form-label">预览阈值</label>
            <div class="flex gap-8">
                <input type="number" class="form-input" name="threshold"
                       value="{threshold}" min="0.1" max="1.0" step="0.05"
                       style="max-width:120px;">
                <button class="btn btn-ghost btn-sm"
                        hx-get="/api/detection/sensitivity-preview/{camera_id}"
                        hx-include="[name='threshold']"
                        hx-target="#sensitivity-preview"
                        hx-swap="innerHTML">预览</button>
            </div>
        </div>
        <div class="flex gap-16 mt-8">
            {stat_card("统计帧数", str(total), "#4fc3f7")}
            {stat_card("当前告警", str(current), "#ff9800")}
            {stat_card("新阈值告警", str(would), "#f44336" if would > current else "#4caf50")}
        </div>
        <div class="mt-8">
            <div class="form-label">分数分布</div>
            <div style="display:flex;gap:2px;padding:8px 0;align-items:flex-end;">
                {bars_html}
            </div>
        </div>
    </div>""")


@router.get("/frame-log/{camera_id}", response_class=HTMLResponse)
async def frame_log_panel(request: Request, camera_id: str):
    """Frame-level log panel with per-stage timing (DET-003/008)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse("")

    diagnostics = camera_manager.get_diagnostics(camera_id, n=30)
    if diagnostics is None or len(diagnostics) == 0:
        return HTMLResponse(f"""
        <div class="card">
            <h3>帧级日志</h3>
            {empty_state("暂无诊断数据", "等待管线处理帧...")}
        </div>""")

    # Build table rows (most recent first)
    rows_html = ""
    for diag in reversed(diagnostics):
        # Stage timing bars
        stage_bars = ""
        for stage in diag.stages:
            colors = {
                "zone_mask": "#4fc3f7",
                "mog2": "#ff9800",
                "person": "#9c27b0",
                "anomaly": "#f44336",
            }
            color = colors.get(stage.stage_name, "#888")
            width = min(int(stage.duration_ms * 2), 100)  # Scale: 1ms = 2px, max 100px
            label = stage.stage_name.replace("_", " ")
            if stage.skipped:
                stage_bars += (
                    f'<span style="display:inline-block;padding:1px 4px;font-size:10px;'
                    f'color:#8890a0;border:1px solid #333;border-radius:2px;margin-right:2px;">'
                    f'{label}: skip</span>'
                )
            else:
                stage_bars += (
                    f'<span style="display:inline-block;height:14px;width:{max(width, 4)}px;'
                    f'background:{color};border-radius:2px;margin-right:2px;" '
                    f'title="{label}: {stage.duration_ms:.1f}ms"></span>'
                )

        score_display = f"{diag.anomaly_score:.3f}" if diag.anomaly_score > 0 else "-"
        alert_badge = '<span class="badge badge-high">!</span>' if diag.alert_emitted else ""
        anomalous_mark = (
            '<span style="color:#f44336;">*</span>' if diag.is_anomalous else ""
        )

        mode_badge = ""
        if diag.pipeline_mode != "active":
            mode_color = "#ff9800" if diag.pipeline_mode == "maintenance" else "#4fc3f7"
            mode_badge = (
                f'<span style="font-size:10px;color:{mode_color};">'
                f'{diag.pipeline_mode[:4]}</span>'
            )

        rows_html += f"""
        <tr>
            <td style="font-family:monospace;font-size:12px;">{diag.frame_number}</td>
            <td>{diag.total_duration_ms:.1f}ms</td>
            <td>{stage_bars}</td>
            <td>{score_display}{anomalous_mark}</td>
            <td>{alert_badge} {mode_badge}</td>
        </tr>"""

    # Stage legend
    legend = """
    <div style="display:flex;gap:12px;font-size:11px;color:#8890a0;margin-top:8px;">
        <span><span style="display:inline-block;width:10px;height:10px;background:#4fc3f7;border-radius:2px;"></span> Zone Mask</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#ff9800;border-radius:2px;"></span> MOG2</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#9c27b0;border-radius:2px;"></span> Person</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#f44336;border-radius:2px;"></span> Anomaly</span>
    </div>"""

    return HTMLResponse(f"""
    <div class="card">
        <h3>帧级日志 <span style="font-size:12px;color:#8890a0;">(最近 {len(diagnostics)} 帧)</span></h3>
        <table style="width:100%;margin-top:8px;">
            <thead>
                <tr>
                    <th style="width:80px;">帧号</th>
                    <th style="width:80px;">耗时</th>
                    <th>阶段耗时</th>
                    <th style="width:80px;">分数</th>
                    <th style="width:60px;">状态</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        {legend}
    </div>""")
