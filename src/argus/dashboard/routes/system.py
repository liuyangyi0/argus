"""System status, overview, and administration routes (Chinese UI).

Consolidates: overview, system info, config, backup, audit, reports, users
into a single tabbed interface.
"""

from __future__ import annotations

import shutil

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from argus.dashboard.components import (
    page_header,
    stat_card,
    status_badge,
    status_banner,
    status_dot,
    tab_bar,
)

logger = structlog.get_logger()

router = APIRouter()

# Sub-tabs for the /system page
_SYSTEM_TABS = [
    ("overview", "系统概览", "/api/system/overview"),
    ("details", "系统详情", "/api/system/details"),
    ("config", "配置管理", "/api/config"),
    ("backup", "数据备份", "/api/backup"),
    ("audit", "审计日志", "/api/audit"),
    ("reports", "报表统计", "/api/reports"),
    ("users", "用户管理", "/api/users"),
]


@router.get("", response_class=HTMLResponse)
async def system_page(request: Request):
    """System administration page — unified tab interface."""
    is_htmx = request.headers.get("HX-Request") == "true"
    header = "" if is_htmx else page_header("系统", "管理配置、备份、审计和用户")
    tabs = tab_bar(_SYSTEM_TABS, "overview")
    return HTMLResponse(f"""
    {header}
    {tabs}
    <div hx-get="/api/system/overview" hx-trigger="load" hx-target="#tab-content" hx-swap="innerHTML"></div>""")


@router.get("/overview", response_class=HTMLResponse)
def overview(request: Request):
    """Dashboard overview with key metrics, camera grid, and pending actions."""
    health_monitor = request.app.state.health_monitor
    camera_manager = request.app.state.camera_manager
    db = request.app.state.db

    # Gather metrics
    total_alerts = 0
    high_alerts = 0
    medium_alerts = 0
    cameras_status = []
    system_status = "offline"
    uptime = "0s"

    if health_monitor:
        h = health_monitor.get_health()
        total_alerts = h.total_alerts
        system_status = h.status.value
        uptime = _format_uptime(h.uptime_seconds)

    if db:
        try:
            high_alerts = db.get_alert_count(severity="high")
            medium_alerts = db.get_alert_count(severity="medium")
        except Exception:
            pass

    if camera_manager:
        cameras_status = camera_manager.get_status()

    connected = sum(1 for c in cameras_status if c.connected)
    total_cams = len(cameras_status)

    # Disk usage
    disk_text = "N/A"
    try:
        usage = shutil.disk_usage("data")
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        disk_text = f"{free_gb:.1f}/{total_gb:.0f}GB"
    except OSError:
        pass

    # System status banner
    status_messages = {
        "healthy": "系统运行正常 — 所有摄像头已连接",
        "degraded": f"系统降级运行 — {connected}/{total_cams} 台摄像头在线",
        "unhealthy": "系统异常 — 请检查摄像头连接",
    }
    banner_msg = status_messages.get(system_status, "系统状态未知")
    banner_html = status_banner(system_status, banner_msg)

    # Anomaly lock status
    lock_html = ""
    if camera_manager:
        locked_cams = []
        for cam in cameras_status:
            pipeline = camera_manager._pipelines.get(cam.camera_id)
            if pipeline and getattr(pipeline, "_locked", False):
                locked_cams.append(cam.camera_id)
        if locked_cams:
            lock_items = "".join(
                f'<div class="flex-between" style="padding:var(--space-2) 0;">'
                f'<span>{status_dot("degraded")}{cam_id} — 异常区域已锁定</span>'
                f'<button class="btn btn-warning btn-sm" '
                f'hx-post="/api/config/clear-lock/{cam_id}" hx-swap="outerHTML">'
                f'解除锁定</button></div>'
                for cam_id in locked_cams
            )
            lock_html = f"""
            <div class="card" style="border-left:3px solid var(--status-warn);">
                <h3>异常锁定</h3>
                {lock_items}
            </div>"""

    # Stat cards
    alert_color = "var(--status-critical)" if high_alerts > 0 else "var(--status-ok)"
    stats_html = f"""
    <div class="grid-4 mb-16">
        {stat_card("摄像头在线", f"{connected}/{total_cams}", "var(--status-info-text)")}
        {stat_card("累计告警", str(total_alerts), alert_color)}
        {stat_card("可用磁盘", disk_text)}
        {stat_card("运行时间", uptime)}
    </div>"""

    # ── Pending actions — the "what to do right now" section ──
    pending_items = []

    # Pending alerts (HIGH/MEDIUM, reuse counts from above)
    pending_alert_count = high_alerts + medium_alerts
    if pending_alert_count > 0:
        pending_items.append(
            f'<div class="flex-between" style="padding:var(--space-3) 0;'
            f'border-bottom:1px solid var(--border-subtle);">'
            f'<span style="color:var(--status-critical-text);">'
            f'待处理告警 <strong>{pending_alert_count}</strong> 条'
            f'</span>'
            f'<a href="/alerts" class="btn btn-sm btn-primary">处理</a>'
            f'</div>'
        )

    # Running tasks
    task_manager = getattr(request.app.state, "task_manager", None)
    if task_manager:
        active = task_manager.get_active_tasks()
        running = [t for t in active if t.status.value in ("pending", "running")]
        if running:
            pending_items.append(
                f'<div class="flex-between" style="padding:var(--space-3) 0;'
                f'border-bottom:1px solid var(--border-subtle);">'
                f'<span style="color:var(--status-info-text);">'
                f'运行中的任务 <strong>{len(running)}</strong> 个'
                f'</span>'
                f'<a href="/models" class="btn btn-sm btn-ghost">查看</a>'
                f'</div>'
            )

    # Offline cameras
    offline = [c for c in cameras_status if not c.connected]
    if offline:
        cam_ids = ", ".join(c.camera_id for c in offline[:3])
        pending_items.append(
            f'<div class="flex-between" style="padding:var(--space-3) 0;'
            f'border-bottom:1px solid var(--border-subtle);">'
            f'<span style="color:var(--status-warn-text);">'
            f'离线摄像头 <strong>{len(offline)}</strong> 台 ({cam_ids})'
            f'</span>'
            f'<a href="/cameras" class="btn btn-sm btn-ghost">查看</a>'
            f'</div>'
        )

    pending_html = ""
    if pending_items:
        pending_html = f"""
        <div class="card" style="border-left:3px solid var(--status-info);">
            <h3>待办事项</h3>
            {"".join(pending_items)}
        </div>"""
    else:
        pending_html = """
        <div class="card" style="border-left:3px solid var(--status-ok);">
            <h3>待办事项</h3>
            <p style="color:var(--text-tertiary);padding:var(--space-2) 0;">暂无待办 — 一切正常</p>
        </div>"""

    # Camera grid
    camera_cards = ""
    for cam in cameras_status:
        status_text = "在线" if cam.connected else "离线"
        frames = cam.stats.frames_captured if cam.stats else 0
        latency = f"{cam.stats.avg_latency_ms:.1f}ms" if cam.stats else "N/A"
        alerts_count = cam.stats.alerts_emitted if cam.stats else 0

        preview_img = (
            f'<img src="/api/cameras/{cam.camera_id}/snapshot?t='
            + "'+Date.now()+'"
            + f'" style="width:100%;height:100%;object-fit:cover;" alt="{cam.name}" '
            f'onerror="this.style.display=\'none\'" />'
        )

        camera_cards += f"""
        <div class="camera-card camera-list-card"
             hx-get="/api/cameras/{cam.camera_id}/detail" hx-target="#content" hx-swap="innerHTML"
             hx-push-url="/cameras?cam={cam.camera_id}">
            <div class="header">
                <span>{status_dot("connected" if cam.connected else "offline")}{cam.camera_id} — {cam.name}</span>
                <span style="font-size:var(--text-xs);color:var(--text-secondary);">{status_text}</span>
            </div>
            <div class="preview" style="position:relative;">
                {preview_img if cam.connected else '<span style="color:var(--text-tertiary);">离线</span>'}
            </div>
            <div style="padding:var(--space-2) var(--space-3);font-size:var(--text-xs);color:var(--text-secondary);">
                帧数: {frames} | 延迟: {latency} | 告警: {alerts_count}
            </div>
        </div>"""

    cameras_html = f"""
    <div class="card">
        <h3>摄像头概览</h3>
        <div class="grid-2">{camera_cards if camera_cards else '<p style="color:var(--text-tertiary);">暂无摄像头</p>'}</div>
    </div>"""

    # Recent alerts
    recent_alerts_html = ""
    if db:
        try:
            recent = db.get_alerts(limit=10)
            if recent:
                rows = ""
                for a in recent:
                    ts = a.timestamp.strftime("%H:%M:%S") if a.timestamp else ""
                    rows += f"""
                    <tr>
                        <td class="mono" style="font-size:var(--text-xs);">{a.alert_id[:20]}</td>
                        <td>{ts}</td>
                        <td>{a.camera_id}</td>
                        <td>{a.zone_id}</td>
                        <td>{status_badge(a.severity)}</td>
                        <td>{a.anomaly_score:.3f}</td>
                    </tr>"""

                recent_alerts_html = f"""
                <div class="card">
                    <h3>最近告警</h3>
                    <table>
                        <thead><tr>
                            <th>告警ID</th><th>时间</th><th>摄像头</th>
                            <th>区域</th><th>严重度</th><th>分数</th>
                        </tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                    <div style="text-align:center;margin-top:var(--space-3);">
                        <a href="/alerts" class="btn btn-ghost btn-sm">查看全部告警</a>
                    </div>
                </div>"""
        except Exception:
            pass

    return HTMLResponse(f"""
    <div data-ws-topic="health" data-ws-refresh-url="/api/system/overview"
         hx-get="/api/system/overview" hx-trigger="every 30s" hx-swap="outerHTML">
        {banner_html}
        {lock_html}
        {stats_html}
        {pending_html}
        <div class="grid-2 mb-16">
            <div>{cameras_html}</div>
            <div>{recent_alerts_html}</div>
        </div>
    </div>""")


@router.get("/health")
def health(request: Request):
    """JSON health endpoint for monitoring tools."""
    health_monitor = request.app.state.health_monitor
    if not health_monitor:
        return {"status": "unknown"}

    h = health_monitor.get_health()
    return {
        "status": h.status.value,
        "uptime_seconds": round(h.uptime_seconds, 1),
        "total_alerts": h.total_alerts,
        "cameras": [
            {
                "camera_id": c.camera_id,
                "connected": c.connected,
                "frames_captured": c.frames_captured,
                "avg_latency_ms": round(c.avg_latency_ms, 1),
            }
            for c in h.cameras
        ],
        "platform": h.platform,
        "python_version": h.python_version,
    }


@router.get("/details", response_class=HTMLResponse)
async def system_details(request: Request):
    """Detailed system information sub-tab."""
    health_monitor = request.app.state.health_monitor
    if not health_monitor:
        return HTMLResponse('<div class="empty-state"><div class="message">健康监控不可用</div></div>')

    h = health_monitor.get_health()

    camera_rows = ""
    for c in h.cameras:
        status_text = "已连接" if c.connected else "已断开"
        camera_rows += f"""
        <tr>
            <td>{status_dot("connected" if c.connected else "offline")}{c.camera_id}</td>
            <td>{status_text}</td>
            <td>{c.frames_captured}</td>
            <td>{c.avg_latency_ms:.1f}ms</td>
            <td>{c.reconnect_count}</td>
            <td style="color:var(--text-tertiary);">{c.error or "—"}</td>
        </tr>"""

    return HTMLResponse(f"""
    <div>
        <div class="card">
            <h3>运行状态</h3>
            <table>
                <tr><td style="color:var(--text-secondary);width:120px;">操作系统</td><td>{h.platform}</td></tr>
                <tr><td style="color:var(--text-secondary);">Python 版本</td><td>{h.python_version}</td></tr>
                <tr><td style="color:var(--text-secondary);">运行时间</td><td>{_format_uptime(h.uptime_seconds)}</td></tr>
                <tr><td style="color:var(--text-secondary);">系统状态</td><td>{h.status.value.upper()}</td></tr>
                <tr><td style="color:var(--text-secondary);">累计告警</td><td>{h.total_alerts}</td></tr>
            </table>
        </div>

        <div class="card">
            <h3>摄像头详情</h3>
            <table>
                <thead><tr>
                    <th>摄像头</th><th>状态</th><th>帧数</th>
                    <th>延迟</th><th>重连次数</th><th>错误信息</th>
                </tr></thead>
                <tbody>{camera_rows if camera_rows else '<tr><td colspan="6" style="color:var(--text-tertiary);">暂无摄像头</td></tr>'}</tbody>
            </table>
        </div>
    </div>""")


def _format_uptime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}秒"
    if seconds < 3600:
        return f"{seconds / 60:.0f}分钟"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.1f}小时"
    return f"{hours / 24:.1f}天"
