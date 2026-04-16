"""Reports and statistics routes (Chinese UI with Chart.js)."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from argus.dashboard.api_response import api_error, api_success, api_unavailable

logger = structlog.get_logger()

router = APIRouter()


@router.get("/json")
async def reports_json(request: Request):
    """JSON statistics API for external consumption."""
    db = request.app.state.db
    if not db:
        return JSONResponse({"error": "数据库不可用"}, status_code=503)

    total = db.get_alert_count()
    all_alerts = db.get_alerts(limit=50000)
    fp_count = sum(1 for a in all_alerts if a.false_positive)
    ack_count = sum(1 for a in all_alerts if a.acknowledged)

    return api_success({
        "total_alerts": total,
        "by_severity": {
            "high": db.get_alert_count(severity="high"),
            "medium": db.get_alert_count(severity="medium"),
            "low": db.get_alert_count(severity="low"),
            "info": db.get_alert_count(severity="info"),
        },
        "false_positive_count": fp_count,
        "false_positive_rate": round(fp_count / total * 100, 2) if total > 0 else 0,
        "acknowledged_count": ack_count,
        "acknowledged_rate": round(ack_count / total * 100, 2) if total > 0 else 0,
    })


# ── Shared query helpers ──────────


def _compute_daily_trend(alerts, days: int):
    """Aggregate alerts by date and severity for the last N days."""
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).replace(tzinfo=None)

    daily: dict[str, dict[str, int]] = defaultdict(
        lambda: {"high": 0, "medium": 0, "low": 0, "info": 0},
    )
    for a in alerts:
        if a.timestamp and a.timestamp >= cutoff:
            key = a.timestamp.strftime("%m-%d")
            if a.severity in daily[key]:
                daily[key][a.severity] += 1

    labels, high, medium, low, info = [], [], [], [], []
    for i in range(days):
        d = now - timedelta(days=days - 1 - i)
        key = d.strftime("%m-%d")
        labels.append(key)
        counts = daily.get(key, {"high": 0, "medium": 0, "low": 0, "info": 0})
        high.append(counts["high"])
        medium.append(counts["medium"])
        low.append(counts["low"])
        info.append(counts["info"])
    return {"labels": labels, "high": high, "medium": medium, "low": low, "info": info}


def _compute_camera_dist(alerts):
    """Count alerts per camera."""
    cam_counts: dict[str, int] = defaultdict(int)
    for a in alerts:
        cam_counts[a.camera_id] += 1
    return [{"camera_id": c, "count": n} for c, n in sorted(cam_counts.items())]


def _compute_fp_trend(alerts, days: int):
    """Daily false-positive rate for the last N days."""
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).replace(tzinfo=None)

    daily_total: dict[str, int] = defaultdict(int)
    daily_fp: dict[str, int] = defaultdict(int)
    for a in alerts:
        if a.timestamp and a.timestamp >= cutoff:
            key = a.timestamp.strftime("%m-%d")
            daily_total[key] += 1
            if a.false_positive:
                daily_fp[key] += 1

    labels, rates = [], []
    for i in range(days):
        d = now - timedelta(days=days - 1 - i)
        key = d.strftime("%m-%d")
        labels.append(key)
        total = daily_total.get(key, 0)
        fp = daily_fp.get(key, 0)
        rates.append(round(fp / total * 100, 1) if total > 0 else 0)
    return {"labels": labels, "rates": rates}


# ── JSON chart data endpoints (consumed by Vue frontend) ──────────


@router.get("/daily-trend/json")
async def daily_trend_json(request: Request, days: int = Query(30, ge=7, le=90)):
    """Daily alert trend data as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    return api_success(_compute_daily_trend(db.get_alerts(limit=50000), days))


@router.get("/severity-dist/json")
async def severity_dist_json(request: Request):
    """Severity distribution as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    return api_success({
        "high": db.get_alert_count(severity="high"),
        "medium": db.get_alert_count(severity="medium"),
        "low": db.get_alert_count(severity="low"),
        "info": db.get_alert_count(severity="info"),
    })


@router.get("/camera-dist/json")
async def camera_dist_json(request: Request):
    """Alerts per camera as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    return api_success({"cameras": _compute_camera_dist(db.get_alerts(limit=50000))})


@router.get("/fp-trend/json")
async def fp_trend_json(request: Request, days: int = Query(30, ge=7, le=90)):
    """Daily false positive rate trend as JSON."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")
    return api_success(_compute_fp_trend(db.get_alerts(limit=50000), days))


# ── Compliance report generation ──────────


def _generate_compliance_data(
    db,
    audit_logger,
    health_monitor,
    camera_manager,
    days: int,
) -> dict:
    """Collect all data needed for a compliance report."""
    from argus.storage.model_registry import ModelRegistry

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    # SQLite timestamps are stored without tzinfo
    since_naive = since.replace(tzinfo=None)

    node_id = "argus-node-01"

    # ── System summary ──
    camera_count = 0
    camera_ids: list[str] = []
    if camera_manager:
        camera_configs = getattr(camera_manager, "_cameras", None) or []
        camera_count = len(camera_configs)
        camera_ids = [c.camera_id for c in camera_configs]
    elif health_monitor:
        health = health_monitor.get_health()
        camera_count = len(health.cameras)
        camera_ids = [c.camera_id for c in health.cameras]

    uptime_hours = 0.0
    if health_monitor:
        health = health_monitor.get_health()
        uptime_hours = round(health.uptime_seconds / 3600, 2)

    total_alerts = db.get_alert_count(since=since)

    # ── Alert statistics ──
    severity_counts = {
        sev: db.get_alert_count(severity=sev, since=since)
        for sev in ("high", "medium", "low", "info")
    }

    all_alerts = db.get_alerts(limit=50000)
    period_alerts = [a for a in all_alerts if a.timestamp and a.timestamp >= since_naive]
    fp_count = sum(1 for a in period_alerts if a.false_positive)
    ack_count = sum(1 for a in period_alerts if a.acknowledged)
    fp_rate = round(fp_count / len(period_alerts) * 100, 2) if period_alerts else 0
    ack_rate = round(ack_count / len(period_alerts) * 100, 2) if period_alerts else 0

    # ── Daily trend ──
    daily_trend = _compute_daily_trend(all_alerts, days)

    # ── Model status ──
    model_status: list[dict] = []
    if db._session_factory:
        try:
            registry = ModelRegistry(db._session_factory)
            for cam_id in camera_ids:
                active = registry.get_active(cam_id)
                latest_training = db.get_latest_training(cam_id)
                model_status.append({
                    "camera_id": cam_id,
                    "active_model": active.model_version_id if active else "无",
                    "model_type": active.model_type if active else "-",
                    "stage": active.stage if active else "-",
                    "last_training": (
                        latest_training.trained_at.strftime("%Y-%m-%d %H:%M")
                        if latest_training and latest_training.trained_at
                        else "-"
                    ),
                    "quality_grade": (
                        latest_training.quality_grade if latest_training else "-"
                    ),
                })
        except Exception:
            logger.debug("compliance.model_status_failed", exc_info=True)

    # ── Audit trail summary ──
    audit_summary: dict | None = None
    if audit_logger:
        try:
            total_actions = audit_logger.count_logs()
            # Get recent logs to compute by-type and by-user
            recent_logs = audit_logger.get_logs(limit=10000)
            by_type: dict[str, int] = defaultdict(int)
            by_user: dict[str, int] = defaultdict(int)
            for entry in recent_logs:
                by_type[entry.action] += 1
                by_user[entry.user] += 1
            audit_summary = {
                "total_actions": total_actions,
                "by_type": dict(by_type),
                "by_user": dict(by_user),
            }
        except Exception:
            logger.debug("compliance.audit_summary_failed", exc_info=True)

    # ── Camera health ──
    camera_health: list[dict] = []
    if health_monitor:
        health = health_monitor.get_health()
        for cam in health.cameras:
            camera_health.append({
                "camera_id": cam.camera_id,
                "connected": cam.connected,
                "frames_captured": cam.frames_captured,
                "avg_latency_ms": round(cam.avg_latency_ms, 1),
                "reconnect_count": cam.reconnect_count,
                "error": cam.error or "",
            })

    return {
        "header": {
            "title": "Argus 异物检测系统 — 合规报告",
            "node_id": node_id,
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "period_days": days,
            "period_start": since.strftime("%Y-%m-%d"),
            "period_end": now.strftime("%Y-%m-%d"),
        },
        "system_summary": {
            "total_cameras": camera_count,
            "uptime_hours": uptime_hours,
            "total_alerts": total_alerts,
        },
        "alert_stats": {
            "by_severity": severity_counts,
            "false_positive_count": fp_count,
            "false_positive_rate": fp_rate,
            "acknowledged_count": ack_count,
            "acknowledged_rate": ack_rate,
        },
        "daily_trend": daily_trend,
        "model_status": model_status,
        "audit_summary": audit_summary,
        "camera_health": camera_health,
    }


def _render_csv(data: dict) -> str:
    """Render compliance data as a multi-section CSV string."""
    buf = io.StringIO()
    w = csv.writer(buf)

    h = data["header"]
    w.writerow([h["title"]])
    w.writerow(["节点", h["node_id"]])
    w.writerow(["生成时间", h["generated_at"]])
    w.writerow(["报告周期", f'{h["period_start"]} ~ {h["period_end"]} ({h["period_days"]}天)'])
    w.writerow([])

    # System summary
    ss = data["system_summary"]
    w.writerow(["## 系统概览"])
    w.writerow(["摄像头数量", ss["total_cameras"]])
    w.writerow(["运行时间(小时)", ss["uptime_hours"]])
    w.writerow(["期间告警总数", ss["total_alerts"]])
    w.writerow([])

    # Alert stats
    a = data["alert_stats"]
    w.writerow(["## 告警统计"])
    w.writerow(["严重度", "数量"])
    for sev, count in a["by_severity"].items():
        label = {"high": "高", "medium": "中", "low": "低", "info": "提示"}.get(sev, sev)
        w.writerow([label, count])
    w.writerow(["误报数", a["false_positive_count"]])
    w.writerow(["误报率(%)", a["false_positive_rate"]])
    w.writerow(["已确认数", a["acknowledged_count"]])
    w.writerow(["确认率(%)", a["acknowledged_rate"]])
    w.writerow([])

    # Daily trend table
    dt = data["daily_trend"]
    w.writerow(["## 每日告警趋势"])
    w.writerow(["日期", "高", "中", "低", "提示"])
    for i, label in enumerate(dt["labels"]):
        w.writerow([label, dt["high"][i], dt["medium"][i], dt["low"][i], dt["info"][i]])
    w.writerow([])

    # Model status
    if data["model_status"]:
        w.writerow(["## 模型状态"])
        w.writerow(["摄像头", "活跃模型", "模型类型", "阶段", "最近训练", "质量等级"])
        for m in data["model_status"]:
            w.writerow([
                m["camera_id"], m["active_model"], m["model_type"],
                m["stage"], m["last_training"], m["quality_grade"],
            ])
        w.writerow([])

    # Audit summary
    if data["audit_summary"]:
        au = data["audit_summary"]
        w.writerow(["## 审计摘要"])
        w.writerow(["操作总数", au["total_actions"]])
        w.writerow([])
        w.writerow(["操作类型", "次数"])
        for action, count in sorted(au["by_type"].items(), key=lambda x: -x[1]):
            w.writerow([action, count])
        w.writerow([])
        w.writerow(["操作人", "次数"])
        for user, count in sorted(au["by_user"].items(), key=lambda x: -x[1]):
            w.writerow([user, count])
        w.writerow([])

    # Camera health
    if data["camera_health"]:
        w.writerow(["## 摄像头健康状态"])
        w.writerow(["摄像头", "连接", "已采帧数", "平均延迟(ms)", "重连次数", "错误"])
        for ch in data["camera_health"]:
            w.writerow([
                ch["camera_id"],
                "是" if ch["connected"] else "否",
                ch["frames_captured"],
                ch["avg_latency_ms"],
                ch["reconnect_count"],
                ch["error"],
            ])

    return buf.getvalue()


def _render_pdf(data: dict) -> bytes:
    """Render compliance data as a PDF using reportlab.

    Raises ImportError if reportlab is not installed.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm, bottomMargin=15 * mm)
    styles = getSampleStyleSheet()

    # Try to register a CJK font for Chinese text
    _font_name = "Helvetica"
    try:
        import pathlib
        # Common CJK font paths on various OS
        _cjk_candidates = [
            pathlib.Path("C:/Windows/Fonts/msyh.ttc"),
            pathlib.Path("C:/Windows/Fonts/simhei.ttf"),
            pathlib.Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            pathlib.Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            pathlib.Path("/System/Library/Fonts/PingFang.ttc"),
        ]
        for fp in _cjk_candidates:
            if fp.exists():
                pdfmetrics.registerFont(TTFont("CJK", str(fp)))
                _font_name = "CJK"
                break
    except Exception:
        pass  # Fall back to Helvetica — Chinese glyphs may not render

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName=_font_name, fontSize=16
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"], fontName=_font_name, fontSize=12
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["Normal"], fontName=_font_name, fontSize=9
    )

    elements: list = []

    def _add_heading(text: str):
        elements.append(Spacer(1, 6 * mm))
        elements.append(Paragraph(text, heading_style))
        elements.append(Spacer(1, 2 * mm))

    def _add_table(headers: list[str], rows: list[list], col_widths=None):
        tdata = [headers] + rows
        t = Table(tdata, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), _font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(t)

    # Header
    h = data["header"]
    elements.append(Paragraph(h["title"], title_style))
    elements.append(Paragraph(
        f'节点: {h["node_id"]} | 生成: {h["generated_at"]} | '
        f'周期: {h["period_start"]} ~ {h["period_end"]} ({h["period_days"]}天)',
        body_style,
    ))

    # System summary
    ss = data["system_summary"]
    _add_heading("系统概览")
    _add_table(
        ["指标", "值"],
        [
            ["摄像头数量", str(ss["total_cameras"])],
            ["运行时间(小时)", str(ss["uptime_hours"])],
            ["期间告警总数", str(ss["total_alerts"])],
        ],
    )

    # Alert stats
    a = data["alert_stats"]
    _add_heading("告警统计")
    sev_labels = {"high": "高", "medium": "中", "low": "低", "info": "提示"}
    sev_rows = [[sev_labels.get(s, s), str(c)] for s, c in a["by_severity"].items()]
    sev_rows.append(["误报率", f'{a["false_positive_rate"]}%'])
    sev_rows.append(["确认率", f'{a["acknowledged_rate"]}%'])
    _add_table(["严重度 / 指标", "数量 / 值"], sev_rows)

    # Daily trend (show up to 31 days in the PDF table)
    dt = data["daily_trend"]
    _add_heading("每日告警趋势")
    trend_rows = []
    for i, lbl in enumerate(dt["labels"]):
        trend_rows.append([lbl, str(dt["high"][i]), str(dt["medium"][i]),
                           str(dt["low"][i]), str(dt["info"][i])])
    _add_table(["日期", "高", "中", "低", "提示"], trend_rows)

    # Model status
    if data["model_status"]:
        _add_heading("模型状态")
        m_rows = [
            [m["camera_id"], m["active_model"], m["model_type"],
             m["stage"], m["last_training"], m["quality_grade"]]
            for m in data["model_status"]
        ]
        _add_table(["摄像头", "活跃模型", "类型", "阶段", "最近训练", "等级"], m_rows)

    # Audit summary
    if data["audit_summary"]:
        au = data["audit_summary"]
        _add_heading("审计摘要")
        elements.append(Paragraph(f'操作总数: {au["total_actions"]}', body_style))
        if au["by_type"]:
            _add_table(
                ["操作类型", "次数"],
                [[act, str(cnt)] for act, cnt in sorted(au["by_type"].items(), key=lambda x: -x[1])],
            )
        if au["by_user"]:
            elements.append(Spacer(1, 2 * mm))
            _add_table(
                ["操作人", "次数"],
                [[usr, str(cnt)] for usr, cnt in sorted(au["by_user"].items(), key=lambda x: -x[1])],
            )

    # Camera health
    if data["camera_health"]:
        _add_heading("摄像头健康状态")
        ch_rows = [
            [ch["camera_id"],
             "是" if ch["connected"] else "否",
             str(ch["frames_captured"]),
             str(ch["avg_latency_ms"]),
             str(ch["reconnect_count"]),
             ch["error"] or "-"]
            for ch in data["camera_health"]
        ]
        _add_table(["摄像头", "连接", "已采帧数", "延迟(ms)", "重连", "错误"], ch_rows)

    doc.build(elements)
    return buf.getvalue()


@router.get("/compliance")
async def compliance_report(
    request: Request,
    days: int = Query(30, ge=7, le=365),
    format: str = Query("csv"),
):
    """Generate a downloadable compliance report (CSV or PDF)."""
    db = request.app.state.db
    if not db:
        return api_unavailable("数据库不可用")

    audit_logger = getattr(request.app.state, "audit_logger", None)
    health_monitor = getattr(request.app.state, "health_monitor", None)
    camera_manager = getattr(request.app.state, "camera_manager", None)

    data = _generate_compliance_data(db, audit_logger, health_monitor, camera_manager, days)

    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if format == "pdf":
        try:
            pdf_bytes = _render_pdf(data)
        except ImportError:
            return api_error(
                40000,
                "请安装 reportlab: pip install reportlab",
                status_code=400,
            )
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="compliance_{now_str}.pdf"',
            },
        )

    # Default: CSV
    csv_text = _render_csv(data)
    # Use UTF-8 BOM so Excel opens Chinese correctly
    csv_bytes = b"\xef\xbb\xbf" + csv_text.encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="compliance_{now_str}.csv"',
        },
    )
