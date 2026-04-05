"""Minimal stubs for legacy HTML component functions.

These are kept only to prevent ImportError in route files that still
contain HTMX HTML-rendering code. The Vue SPA frontend does not use these.
"""


def page_header(title: str, subtitle: str = "", actions_html: str = "") -> str:
    return f"<h2>{title}</h2>"


def stat_card(label: str, value: str, color: str = "") -> str:
    return f"<div>{label}: {value}</div>"


def status_badge(severity: str) -> str:
    return f"<span>{severity}</span>"


def status_dot(status: str) -> str:
    return ""


def status_banner(status: str, message: str) -> str:
    return f"<div>{message}</div>"


def form_group(label: str, name: str, **kwargs) -> str:
    return f'<input name="{name}" />'


def form_select(label: str, name: str, options=None, **kwargs) -> str:
    return f'<select name="{name}"></select>'


def form_checkbox(label: str, name: str, **kwargs) -> str:
    return f'<input type="checkbox" name="{name}" />'


def progress_bar(progress: int = 0, status: str = "") -> str:
    return f"<div>{progress}%</div>"


def empty_state(message: str = "", hint: str = "", **kwargs) -> str:
    return f"<div>{message}</div>"


def confirm_button(text: str, url: str, **kwargs) -> str:
    return f"<button>{text}</button>"


def tab_bar(tabs=None, active_tab: str = "") -> str:
    return "<div></div>"


def camera_select(cameras=None, selected_id: str = "") -> str:
    return "<select></select>"


def pipeline_stepper(steps=None) -> str:
    return "<div></div>"
