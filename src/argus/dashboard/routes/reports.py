"""Reports and statistics routes (Chinese UI with Chart.js)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from argus.dashboard.components import empty_state, page_header, stat_card

router = APIRouter()


@router.get("", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Reports overview page with statistics and charts."""
    db = request.app.state.db
    if not db:
        return HTMLResponse(empty_state("数据库不可用"))

    # Compute stats
    total = db.get_alert_count()
    high = db.get_alert_count(severity="high")
    medium = db.get_alert_count(severity="medium")
    low = db.get_alert_count(severity="low")
    info = db.get_alert_count(severity="info")

    # False positive rate
    all_alerts = db.get_alerts(limit=10000)
    fp_count = sum(1 for a in all_alerts if a.false_positive)
    fp_rate = f"{fp_count / total * 100:.1f}%" if total > 0 else "0%"

    # Acknowledged rate
    ack_count = sum(1 for a in all_alerts if a.acknowledged)
    ack_rate = f"{ack_count / total * 100:.1f}%" if total > 0 else "0%"

    is_htmx = request.headers.get("HX-Request") == "true"
    header = "" if is_htmx else page_header("报表统计", "告警趋势、误报率和系统运行统计")

    stats_html = f"""
    <div class="grid-4 mb-16">
        {stat_card("告警总数", str(total))}
        {stat_card("高严重度", str(high), "#d50000")}
        {stat_card("误报率", fp_rate, "#ff9800")}
        {stat_card("确认率", ack_rate, "#4caf50")}
    </div>"""

    # Daily trend chart container (loaded via HTMX)
    chart_html = f"""
    <div class="card">
        <h3>每日告警趋势（最近 30 天）</h3>
        <div id="trend-chart-container"
             hx-get="/api/reports/daily-trend"
             hx-trigger="load"
             hx-swap="innerHTML">
            <div class="empty-state"><div class="spinner"></div></div>
        </div>
    </div>

    <div class="grid-2">
        <div class="card">
            <h3>按严重度分布</h3>
            <div id="severity-chart-container"
                 hx-get="/api/reports/severity-dist"
                 hx-trigger="load"
                 hx-swap="innerHTML">
                <div class="empty-state"><div class="spinner"></div></div>
            </div>
        </div>
        <div class="card">
            <h3>按摄像头分布</h3>
            <div id="camera-chart-container"
                 hx-get="/api/reports/camera-dist"
                 hx-trigger="load"
                 hx-swap="innerHTML">
                <div class="empty-state"><div class="spinner"></div></div>
            </div>
        </div>
    </div>

    <div class="card">
        <h3>误报率趋势（最近 30 天）</h3>
        <div id="fp-chart-container"
             hx-get="/api/reports/fp-trend"
             hx-trigger="load"
             hx-swap="innerHTML">
            <div class="empty-state"><div class="spinner"></div></div>
        </div>
    </div>"""

    return HTMLResponse(f"""
    {header}
    {stats_html}
    {chart_html}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>""")


@router.get("/daily-trend", response_class=HTMLResponse)
async def daily_trend_chart(request: Request, days: int = Query(30, ge=7, le=90)):
    """Render daily alert trend chart using Chart.js."""
    db = request.app.state.db
    if not db:
        return HTMLResponse(empty_state("数据库不可用"))

    alerts = db.get_alerts(limit=50000)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # Aggregate by date and severity
    daily: dict[str, dict[str, int]] = defaultdict(lambda: {"high": 0, "medium": 0, "low": 0, "info": 0})
    for a in alerts:
        if a.timestamp and a.timestamp >= cutoff.replace(tzinfo=None):
            date_key = a.timestamp.strftime("%m-%d")
            if a.severity in daily[date_key]:
                daily[date_key][a.severity] += 1

    # Generate sorted date labels for last N days
    labels = []
    high_data = []
    medium_data = []
    low_data = []
    info_data = []
    for i in range(days):
        d = now - timedelta(days=days - 1 - i)
        key = d.strftime("%m-%d")
        labels.append(key)
        counts = daily.get(key, {"high": 0, "medium": 0, "low": 0, "info": 0})
        high_data.append(counts["high"])
        medium_data.append(counts["medium"])
        low_data.append(counts["low"])
        info_data.append(counts["info"])

    chart_id = "dailyTrendChart"
    return HTMLResponse(f"""
    <canvas id="{chart_id}" height="200"></canvas>
    <script>
    (function() {{
        var ctx = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {labels},
                datasets: [
                    {{ label: '高', data: {high_data}, backgroundColor: '#d50000', stack: 's' }},
                    {{ label: '中', data: {medium_data}, backgroundColor: '#b71c1c', stack: 's' }},
                    {{ label: '低', data: {low_data}, backgroundColor: '#e65100', stack: 's' }},
                    {{ label: '提示', data: {info_data}, backgroundColor: '#0d47a1', stack: 's' }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: '#8890a0' }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#8890a0' }}, grid: {{ color: '#2a2d37' }} }},
                    y: {{ ticks: {{ color: '#8890a0' }}, grid: {{ color: '#2a2d37' }}, beginAtZero: true }}
                }}
            }}
        }});
    }})();
    </script>""")


@router.get("/severity-dist", response_class=HTMLResponse)
async def severity_distribution(request: Request):
    """Severity distribution doughnut chart."""
    db = request.app.state.db
    if not db:
        return HTMLResponse(empty_state("数据库不可用"))

    high = db.get_alert_count(severity="high")
    medium = db.get_alert_count(severity="medium")
    low = db.get_alert_count(severity="low")
    info = db.get_alert_count(severity="info")

    chart_id = "severityChart"
    return HTMLResponse(f"""
    <canvas id="{chart_id}" height="200"></canvas>
    <script>
    (function() {{
        var ctx = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: ['高', '中', '低', '提示'],
                datasets: [{{
                    data: [{high}, {medium}, {low}, {info}],
                    backgroundColor: ['#d50000', '#b71c1c', '#e65100', '#0d47a1']
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#8890a0' }} }} }}
            }}
        }});
    }})();
    </script>""")


@router.get("/camera-dist", response_class=HTMLResponse)
async def camera_distribution(request: Request):
    """Alerts per camera bar chart."""
    db = request.app.state.db
    if not db:
        return HTMLResponse(empty_state("数据库不可用"))

    alerts = db.get_alerts(limit=50000)
    cam_counts: dict[str, int] = defaultdict(int)
    for a in alerts:
        cam_counts[a.camera_id] += 1

    if not cam_counts:
        return HTMLResponse(empty_state("暂无数据"))

    cameras = sorted(cam_counts.keys())
    counts = [cam_counts[c] for c in cameras]

    chart_id = "cameraChart"
    return HTMLResponse(f"""
    <canvas id="{chart_id}" height="200"></canvas>
    <script>
    (function() {{
        var ctx = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {cameras},
                datasets: [{{ label: '告警数', data: {counts}, backgroundColor: '#4fc3f7' }}]
            }},
            options: {{
                responsive: true,
                indexAxis: 'y',
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#8890a0' }}, grid: {{ color: '#2a2d37' }}, beginAtZero: true }},
                    y: {{ ticks: {{ color: '#8890a0' }}, grid: {{ color: '#2a2d37' }} }}
                }}
            }}
        }});
    }})();
    </script>""")


@router.get("/fp-trend", response_class=HTMLResponse)
async def false_positive_trend(request: Request, days: int = Query(30, ge=7, le=90)):
    """Daily false positive rate trend line chart."""
    db = request.app.state.db
    if not db:
        return HTMLResponse(empty_state("数据库不可用"))

    alerts = db.get_alerts(limit=50000)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    daily_total: dict[str, int] = defaultdict(int)
    daily_fp: dict[str, int] = defaultdict(int)
    for a in alerts:
        if a.timestamp and a.timestamp >= cutoff.replace(tzinfo=None):
            key = a.timestamp.strftime("%m-%d")
            daily_total[key] += 1
            if a.false_positive:
                daily_fp[key] += 1

    labels = []
    rates = []
    for i in range(days):
        d = now - timedelta(days=days - 1 - i)
        key = d.strftime("%m-%d")
        labels.append(key)
        total = daily_total.get(key, 0)
        fp = daily_fp.get(key, 0)
        rate = round(fp / total * 100, 1) if total > 0 else 0
        rates.append(rate)

    chart_id = "fpTrendChart"
    return HTMLResponse(f"""
    <canvas id="{chart_id}" height="150"></canvas>
    <script>
    (function() {{
        var ctx = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: '误报率 (%)',
                    data: {rates},
                    borderColor: '#ff9800',
                    backgroundColor: 'rgba(255,152,0,0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: '#8890a0' }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#8890a0' }}, grid: {{ color: '#2a2d37' }} }},
                    y: {{ ticks: {{ color: '#8890a0', callback: function(v){{ return v + '%'; }} }}, grid: {{ color: '#2a2d37' }}, beginAtZero: true, max: 100 }}
                }}
            }}
        }});
    }})();
    </script>""")


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

    return {
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
    }
