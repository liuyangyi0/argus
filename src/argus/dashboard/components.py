"""Reusable HTML component functions for the Argus dashboard.

All text output is in Chinese. Components return HTML strings that
can be embedded in route handler responses.
"""

from __future__ import annotations


def page_header(title: str, subtitle: str = "", actions_html: str = "") -> str:
    """Page header with title, optional subtitle, and action buttons."""
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    actions = f'<div class="flex gap-12">{actions_html}</div>' if actions_html else ""
    return f"""
    <div class="page-header">
        <div><h2>{title}</h2>{sub}</div>
        {actions}
    </div>"""


def stat_card(label: str, value: str, color: str = "#e0e0e0") -> str:
    """Metric stat card with large value and small label."""
    return f"""
    <div class="card stat">
        <div class="value" style="color:{color};">{value}</div>
        <div class="label">{label}</div>
    </div>"""


def status_badge(severity: str) -> str:
    """Severity badge with color coding."""
    labels = {"info": "提示", "low": "低", "medium": "中", "high": "高"}
    label = labels.get(severity, severity)
    return f'<span class="badge badge-{severity}">{label}</span>'


def status_dot(status: str) -> str:
    """Small colored status dot."""
    cls_map = {
        "healthy": "status-healthy",
        "degraded": "status-degraded",
        "unhealthy": "status-unhealthy",
        "offline": "status-offline",
        "connected": "status-healthy",
        "disconnected": "status-offline",
    }
    cls = cls_map.get(status, "status-offline")
    return f'<span class="status-dot {cls}"></span>'


def status_banner(status: str, message: str) -> str:
    """Full-width status banner for system health."""
    icons = {"healthy": "●", "degraded": "◐", "unhealthy": "○"}
    icon = icons.get(status, "○")
    return f"""
    <div class="status-banner {status}">
        <span style="font-size:18px;">{icon}</span>
        <span>{message}</span>
    </div>"""


def form_group(
    label: str,
    name: str,
    value: str = "",
    input_type: str = "text",
    hint: str = "",
    placeholder: str = "",
    required: bool = False,
    min_val: str = "",
    max_val: str = "",
    step: str = "",
) -> str:
    """Form input group with label and optional hint."""
    attrs = f'name="{name}" id="{name}" value="{value}" class="form-input"'
    if placeholder:
        attrs += f' placeholder="{placeholder}"'
    if required:
        attrs += " required"
    if min_val:
        attrs += f' min="{min_val}"'
    if max_val:
        attrs += f' max="{max_val}"'
    if step:
        attrs += f' step="{step}"'
    hint_html = f'<div class="form-hint">{hint}</div>' if hint else ""
    return f"""
    <div class="form-group">
        <label class="form-label" for="{name}">{label}</label>
        <input type="{input_type}" {attrs}>
        {hint_html}
    </div>"""


def form_select(label: str, name: str, options: list[tuple[str, str]], selected: str = "", hint: str = "") -> str:
    """Form select dropdown. Options are (value, display_text) tuples."""
    opts_html = ""
    for val, text in options:
        sel = ' selected' if val == selected else ''
        opts_html += f'<option value="{val}"{sel}>{text}</option>'
    hint_html = f'<div class="form-hint">{hint}</div>' if hint else ""
    return f"""
    <div class="form-group">
        <label class="form-label" for="{name}">{label}</label>
        <select name="{name}" id="{name}" class="form-select">{opts_html}</select>
        {hint_html}
    </div>"""


def progress_bar(progress: int, status: str = "running") -> str:
    """Animated progress bar. Status: running, success, error."""
    fill_cls = "progress-fill"
    if status == "complete":
        fill_cls += " success"
    elif status == "failed":
        fill_cls += " error"
    return f"""
    <div class="progress-bar">
        <div class="{fill_cls}" style="width:{progress}%;"></div>
    </div>"""


def empty_state(message: str, hint: str = "") -> str:
    """Empty state placeholder."""
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    return f"""
    <div class="empty-state">
        <div class="message">{message}</div>
        {hint_html}
    </div>"""


def confirm_button(
    text: str,
    url: str,
    method: str = "post",
    target: str = "",
    swap: str = "outerHTML",
    css_class: str = "btn btn-primary btn-sm",
    confirm_msg: str = "",
) -> str:
    """HTMX-powered button with optional confirm dialog."""
    hx_method = f'hx-{method}="{url}"'
    hx_target = f'hx-target="{target}"' if target else ""
    hx_confirm = f'hx-confirm="{confirm_msg}"' if confirm_msg else ""
    return (
        f'<button class="{css_class}" {hx_method} hx-swap="{swap}" '
        f'{hx_target} {hx_confirm}>{text}</button>'
    )


def tab_bar(tabs: list[tuple[str, str, str]], active_tab: str) -> str:
    """Tab navigation bar. Tabs are (id, label, url) tuples."""
    items = ""
    for tab_id, label, url in tabs:
        active = " active" if tab_id == active_tab else ""
        items += (
            f'<button class="tab-item{active}" '
            f'hx-get="{url}" hx-target="#tab-content" hx-swap="innerHTML">'
            f'{label}</button>'
        )
    return f'<div class="tab-bar">{items}</div><div id="tab-content"></div>'


def camera_select(cameras: list, selected_id: str = "") -> str:
    """Camera selection dropdown from camera status list."""
    options = [(c.camera_id, f"{c.camera_id} - {c.name}") for c in cameras]
    return form_select("选择摄像头", "camera_id", options, selected=selected_id)
