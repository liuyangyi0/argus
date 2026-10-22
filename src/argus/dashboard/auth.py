"""Authentication middleware and utilities for the Argus dashboard.

Supports:
- Session-based cookie auth (primary)
- X-API-Token header (legacy API automation)

Public endpoints (no auth required):
- /login, /logout
- /api/system/health  (Docker healthcheck)
- /metrics            (Prometheus scraping)
- /static/*
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlsplit

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from argus.config.schema import AuthConfig
from argus.dashboard.api_response import ErrorCode, api_error
from argus.dashboard.forms import parse_request_form

if TYPE_CHECKING:
    from argus.storage.database import Database

logger = structlog.get_logger()

# Endpoints that bypass authentication entirely
_PUBLIC_PATHS = frozenset({
    "/api/system/health",
    "/metrics",
    "/login",
    "/logout",
})

_SESSION_COOKIE = "argus_session"


def _safe_next_url(value: str) -> str:
    """Return a same-origin absolute path suitable for login redirects."""
    if not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or "\r" in value or "\n" in value:
        return "/"
    return value


def _login_url(next_url: str, *, error: str | None = None) -> str:
    params = {"next": _safe_next_url(next_url)}
    if error:
        params["error"] = error
    return f"/login?{urlencode(params)}"


def _set_session_cookie(
    request: Request,
    response: RedirectResponse,
    token: str,
) -> None:
    """Set one consistently scoped session cookie from application policy."""
    max_age = int(getattr(request.app.state, "session_max_age", 28800))
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=max_age,
        path="/",
    )


# ── Password utilities ──

def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex() + ":" + key.hex()


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against its stored hash. Constant-time safe."""
    try:
        salt_hex, key_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return secrets.compare_digest(key.hex(), key_hex)
    except Exception:
        return False


# ── Session token utilities ──

def create_session_token(username: str, role: str, secret: str) -> str:
    """Create a signed, base64-encoded session token."""
    data = json.dumps({"u": username, "r": role, "t": int(time.time())})
    sig = hmac.new(secret.encode(), data.encode(), "sha256").hexdigest()
    return base64.urlsafe_b64encode(f"{data}|{sig}".encode()).decode()


def verify_session_token(token: str, secret: str, max_age: int = 28800) -> dict | None:
    """Verify and decode a session token. Returns user dict or None."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        data_str, sig = decoded.rsplit("|", 1)
        expected_sig = hmac.new(
            secret.encode(),
            data_str.encode(),
            "sha256",
        ).hexdigest()
        if not secrets.compare_digest(sig, expected_sig):
            return None
        data = json.loads(data_str)
        if time.time() - data.get("t", 0) > max_age:
            return None
        return {"username": data["u"], "role": data["r"]}
    except Exception:
        return None


# ── Role helper ──

def require_role(request: Request, *roles: str) -> bool:
    """Check if current user has one of the required roles. Returns True if authorized."""
    user = getattr(request.state, "user", None)
    if user is None:
        return False
    return user.get("role", "viewer") in roles


def current_username(request: Request) -> str:
    """Return the logged-in username for audit logs.

    Resolves from ``request.state.user`` (populated by the auth middleware).
    Returns ``"unknown"`` for anonymous / token-less calls so audit trails
    never contain hardcoded ``"operator"`` placeholders.
    """
    user = getattr(request.state, "user", None) or {}
    if isinstance(user, dict):
        username = user.get("username")
        if isinstance(username, str) and username:
            return username
    return "unknown"


# ── UX v2 §11.1: Permission matrix (4-tier RBAC) ──

PERMISSION_MAP: dict[str, list[str]] = {
    "viewer": [
        "read_alerts", "read_cameras", "read_system", "read_models",
        "read_degradation",
    ],
    "operator": [
        "read_alerts", "read_cameras", "read_system", "read_models",
        "read_degradation",
        "handle_alerts", "shift_handoff", "mute_audio",
        "view_replay", "pin_frame",
    ],
    "engineer": [
        "read_alerts", "read_cameras", "read_system", "read_models",
        "read_degradation",
        "handle_alerts", "shift_handoff", "mute_audio",
        "view_replay", "pin_frame",
        "edit_zones", "edit_thresholds", "edit_config",
        "manage_baselines", "manage_training", "manage_models",
        "rollback_model",
    ],
    "admin": ["*"],  # wildcard: all permissions
}

# Valid roles (includes 'engineer' added in UX v2)
VALID_ROLES = frozenset(PERMISSION_MAP.keys())


def refresh_session_identity(
    user_info: dict,
    database: Database | None,
) -> dict[str, str] | None:
    """Resolve a signed session against the current active database user.

    The cookie role is only a signed login-time claim. When a database is
    available, authorization always uses the current row so role downgrades,
    disable and deletion take effect without waiting for the 8-hour cookie to
    expire. Database failures fail closed.
    """
    username = user_info.get("username")
    role = user_info.get("role")
    if not isinstance(username, str) or not username:
        return None
    if database is None:
        if not isinstance(role, str) or role not in VALID_ROLES:
            return None
        return {"username": username, "role": role}
    try:
        record = database.get_user(username)
    except Exception:
        logger.warning("auth.session_identity_lookup_failed", username=username)
        return None
    current_role = getattr(record, "role", None) if record is not None else None
    if (
        record is None
        or not bool(getattr(record, "active", False))
        or current_role not in VALID_ROLES
    ):
        return None
    return {"username": username, "role": str(current_role)}


def has_permission(request: Request, permission: str) -> bool:
    """Check if the current user has a specific permission.

    Uses the PERMISSION_MAP to resolve role → permissions.
    Admin role has wildcard access to all permissions.
    """
    user = getattr(request.state, "user", None)
    if user is None:
        return False
    role = user.get("role", "viewer")
    perms = PERMISSION_MAP.get(role, [])
    return "*" in perms or permission in perms


def require_permission(request: Request, permission: str) -> bool:
    """Check permission and return True if authorized.

    For use in route handlers:
        if not require_permission(request, "handle_alerts"):
            return api_error(ErrorCode.FORBIDDEN, "权限不足", status_code=403)
    """
    return has_permission(request, permission)


def get_denied_response(request: Request, permission: str) -> dict:
    """Build a standardized 403 response body for permission denial.

    Returns the standard envelope body suitable for a JSON response.
    """
    user = getattr(request.state, "user", None)
    role = user.get("role", "unknown") if user else "anonymous"
    return {
        "code": ErrorCode.FORBIDDEN,
        "msg": "权限不足",
        "data": {
            "required_permission": permission,
            "your_role": role,
            "message": f"角色 '{role}' 没有 '{permission}' 权限",
        },
    }


# ── Auth Middleware (pure ASGI) ──
#
# IMPORTANT: These middleware classes are implemented as pure ASGI middleware
# instead of Starlette's BaseHTTPMiddleware.  BaseHTTPMiddleware buffers the
# entire body of StreamingResponse through an internal asyncio queue, which
# blocks the event loop for long-lived MJPEG streams and prevents other
# requests from being processed.
#
# See: https://github.com/encode/starlette/discussions/1729
#      https://github.com/encode/starlette/issues/1012


class RequestIdMiddleware:
    """Attach a safe request ID without buffering streaming responses.

    A caller-provided ``X-Request-ID`` is retained when it contains only
    printable identifier characters and is at most 128 characters. Otherwise
    a cryptographically random ID is generated. The value is available as
    ``request.state.request_id`` and returned in the response header.
    """

    _HEADER = b"x-request-id"
    _ALLOWED = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
    )

    def __init__(self, app: ASGIApp):
        self.app = app

    @classmethod
    def _resolve_request_id(cls, scope: Scope) -> str:
        for name, raw_value in scope.get("headers", []):
            if name.lower() != cls._HEADER:
                continue
            try:
                candidate = raw_value.decode("ascii")
            except UnicodeDecodeError:
                break
            valid_length = 1 <= len(candidate) <= 128
            if valid_length and all(char in cls._ALLOWED for char in candidate):
                return candidate
            break
        return f"req_{secrets.token_hex(16)}"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._resolve_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = [
                    header
                    for header in message.get("headers", [])
                    if header[0].lower() != self._HEADER
                ]
                headers.append((self._HEADER, request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class AuthMiddleware:
    """Session + API-token authentication middleware (pure ASGI).

    Checks in order:
    1. Session cookie (browser users)
    2. X-API-Token header (legacy API automation)

    Unauthenticated browser requests are redirected to /login.
    Unauthenticated API requests receive a 401 JSON response.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: AuthConfig,
        session_secret: str,
        database: Database | None = None,
    ):
        self.app = app
        self._enabled = config.enabled
        self._token_hash = (
            hashlib.sha256(config.api_token.encode()).hexdigest()
            if config.api_token else ""
        )
        self._secret = session_secret
        self._max_age = config.session_timeout_minutes * 60
        self._database = database

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # WebSocket connections are authenticated separately in the WS
        # endpoint handler (via verify_ws_token), not here.
        if scope["type"] == "websocket":
            if not self._enabled:
                scope.setdefault("state", {})["user"] = {"username": "system", "role": "admin"}
            await self.app(scope, receive, send)
            return

        # HTTP requests
        request = Request(scope)

        if not self._enabled:
            request.state.user = {"username": "system", "role": "admin"}
            await self.app(scope, receive, send)
            return

        path = request.url.path

        # Static files and public paths bypass auth
        if path.startswith("/static/") or path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Try session cookie first
        session_token = request.cookies.get(_SESSION_COOKIE)
        if session_token:
            user_info = verify_session_token(session_token, self._secret, self._max_age)
            if user_info:
                current_identity = refresh_session_identity(user_info, self._database)
                if current_identity is not None:
                    request.state.user = current_identity
                    await self.app(scope, receive, send)
                    return

        # Try X-API-Token header (legacy)
        api_token = request.headers.get("x-api-token", "")
        if api_token and self._verify_api_token(api_token):
            request.state.user = {"username": "api", "role": "admin"}
            await self.app(scope, receive, send)
            return

        # Try HTTP Basic Auth (legacy)
        if not api_token:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Basic "):
                try:
                    from base64 import b64decode as _b64decode
                    decoded = _b64decode(auth_header[6:]).decode("utf-8")
                    _, _, token = decoded.partition(":")
                    if token and self._verify_api_token(token):
                        request.state.user = {"username": "api", "role": "admin"}
                        await self.app(scope, receive, send)
                        return
                except Exception:
                    logger.debug("auth.api_token_verify_failed", exc_info=True)

        # Not authenticated — redirect browsers to /login, return JSON for API
        accepts_html = "text/html" in request.headers.get("accept", "")
        is_htmx = request.headers.get("hx-request") == "true"

        # API callers always receive a machine-readable envelope even if a
        # browser sends ``Accept: text/html``.
        if path.startswith("/api/"):
            accepts_html = False

        if accepts_html and not is_htmx:
            query = request.url.query
            next_url = f"{path}?{query}" if query else path
            redirect_url = _login_url(next_url)
            response = RedirectResponse(url=redirect_url, status_code=302)
        else:
            response = api_error(
                ErrorCode.AUTH_REQUIRED,
                "Authentication required",
                status_code=401,
                data={"request_id": getattr(request.state, "request_id", None)},
                headers={"WWW-Authenticate": 'Bearer realm="Argus"'},
            )
        await response(scope, receive, send)

    def _verify_api_token(self, token: str) -> bool:
        """Constant-time token comparison to prevent timing attacks."""
        if not self._token_hash:
            return False
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return secrets.compare_digest(token_hash, self._token_hash)


# ── Rate Limit Middleware (pure ASGI) ──

class RateLimitMiddleware:
    """Simple in-memory rate limiter for POST endpoints (pure ASGI).

    Limits requests per IP per minute. Uses a sliding window counter.
    """

    def __init__(self, app: ASGIApp, max_requests_per_minute: int = 60):
        self.app = app
        self._max_rpm = max_requests_per_minute
        self._counters: dict[str, list[float]] = defaultdict(list)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        if method not in ("POST", "DELETE"):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        now = time.monotonic()
        window = 60.0

        # Prune old entries
        entries = [t for t in self._counters[client_ip] if now - t < window]
        if not entries:
            self._counters.pop(client_ip, None)
        else:
            self._counters[client_ip] = entries

        if len(entries) >= self._max_rpm:
            logger.warning("rate_limit.exceeded", client_ip=client_ip, count=len(entries))
            request_id = scope.get("state", {}).get("request_id")
            response = api_error(
                ErrorCode.RATE_LIMITED,
                "Rate limit exceeded",
                status_code=429,
                data={"request_id": request_id},
                headers={"Retry-After": "60"},
            )
            await response(scope, receive, send)
            return

        self._counters[client_ip].append(now)
        await self.app(scope, receive, send)


def verify_token(token: str, config: AuthConfig) -> bool:
    """Verify an API token against the configured token.

    Shared by both HTTP auth middleware and WebSocket auth.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not config.enabled:
        return True
    if not config.api_token or not token:
        return False
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expected_hash = hashlib.sha256(config.api_token.encode()).hexdigest()
    return secrets.compare_digest(token_hash, expected_hash)


# ── Security Headers Middleware (pure ASGI) ──

class SecurityHeadersMiddleware:
    """Adds security headers to all responses (pure ASGI)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # Static assets manage their own Cache-Control. Replay media responses
        # set an explicit private policy themselves; replay JSON and all replay
        # errors must inherit the fail-closed no-store default here.
        is_cacheable = path.startswith("/static/")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"x-xss-protection", b"1; mode=block"),
                ]
                has_cache_control = any(
                    name.lower() == b"cache-control" for name, _ in headers
                )
                if not is_cacheable and not has_cache_control:
                    extra.append((b"cache-control", b"no-store"))
                message["headers"] = headers + extra
            await send(message)

        await self.app(scope, receive, send_with_headers)


# ── Login / Logout routes ──

auth_router = APIRouter()


@auth_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login form."""
    error = request.query_params.get("error", "")
    next_url = _safe_next_url(request.query_params.get("next", "/"))
    error_html = (
        f'<div class="login-error" role="alert">{escape(error)}</div>'
        if error else ""
    )
    return HTMLResponse(_LOGIN_PAGE_HTML.format(
        error_html=error_html,
        next_url=escape(next_url, quote=True),
    ))


@auth_router.post("/login")
async def login_handler(request: Request):
    """Handle login form submission."""
    form = await parse_request_form(request)
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    next_url = _safe_next_url(str(form.get("next", "/")))

    db = getattr(request.app.state, "db", None)
    secret = getattr(request.app.state, "session_secret", "default-secret")
    auth_enabled = getattr(request.app.state, "auth_enabled", True)

    if not auth_enabled:
        # Auth disabled — accept any login
        token = create_session_token(username or "guest", "admin", secret)
        response = RedirectResponse(url=next_url, status_code=302)
        _set_session_cookie(request, response, token)
        return response

    if not db or not username or not password:
        return RedirectResponse(
            url=_login_url(next_url, error="用户名或密码不能为空"),
            status_code=302,
        )

    user = db.get_user(username)
    if user is None or not user.active or not verify_password(password, user.password_hash):
        logger.warning("auth.login_failed", username=username)
        return RedirectResponse(
            url=_login_url(next_url, error="用户名或密码错误"),
            status_code=302,
        )

    # Update last_login timestamp
    db.update_user(username, last_login=datetime.now(timezone.utc))

    token = create_session_token(user.username, user.role, secret)
    response = RedirectResponse(url=next_url, status_code=302)
    _set_session_cookie(request, response, token)
    logger.info("auth.login_success", username=username, role=user.role)
    return response


@auth_router.get("/logout", include_in_schema=False)
async def logout_get():
    """Keep the SPA catch-all from turning an unsafe logout GET into 200."""
    return api_error(
        ErrorCode.METHOD_NOT_ALLOWED,
        "Method Not Allowed",
        status_code=405,
        headers={"Allow": "POST"},
    )


@auth_router.post("/logout")
async def logout(request: Request):
    """Clear session cookie and redirect to /login."""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(
        _SESSION_COOKIE,
        path="/",
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="lax",
    )
    return response


# ── Login page HTML ──

_LOGIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Argus - 登录</title>
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{ margin: 0; min-height: 100%; }}
        body {{
            color: #172033;
            background: #f4f7fb;
            font-family: Inter, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
        }}
        .login-container {{
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
            background:
                radial-gradient(circle at 18% 12%, rgba(35, 99, 235, 0.12), transparent 34%),
                #f4f7fb;
        }}
        .login-card {{
            background: #ffffff;
            border: 1px solid #d9e2ef;
            border-radius: 14px;
            padding: 40px 36px;
            width: 360px;
            max-width: 100%;
            box-shadow: 0 18px 48px rgba(30, 55, 90, 0.12);
        }}
        .login-logo {{
            font-size: 28px;
            font-weight: 700;
            color: #1859c9;
            letter-spacing: 4px;
            text-align: center;
            margin-bottom: 6px;
        }}
        .login-subtitle {{
            text-align: center;
            color: #66758b;
            font-size: 13px;
            margin-bottom: 32px;
        }}
        .login-error {{
            background: #fff1f0;
            border: 1px solid #ffccc7;
            color: #a8071a;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 13px;
            margin-bottom: 16px;
            text-align: center;
        }}
        .form-group {{ margin-bottom: 18px; }}
        .form-label {{
            display: block;
            margin-bottom: 7px;
            color: #344054;
            font-size: 13px;
            font-weight: 600;
        }}
        .form-input {{
            width: 100%;
            min-height: 42px;
            padding: 9px 12px;
            color: #172033;
            background: #ffffff;
            border: 1px solid #b8c5d6;
            border-radius: 7px;
            font: inherit;
            outline: none;
        }}
        .form-input:focus {{
            border-color: #2363eb;
            box-shadow: 0 0 0 3px rgba(35, 99, 235, 0.16);
        }}
        .login-btn {{
            width: 100%;
            min-height: 42px;
            padding: 10px 14px;
            margin-top: 8px;
            color: #ffffff;
            background: #1859c9;
            border: 1px solid #1859c9;
            border-radius: 7px;
            font: inherit;
            font-weight: 600;
            cursor: pointer;
        }}
        .login-btn:hover {{ background: #1249aa; }}
        .login-btn:focus-visible {{
            outline: 3px solid rgba(35, 99, 235, 0.32);
            outline-offset: 2px;
        }}
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-card">
            <div class="login-logo">ARGUS</div>
            <div class="login-subtitle">视觉异常检测系统</div>
            {error_html}
            <form method="post" action="/login">
                <input type="hidden" name="next" value="{next_url}">
                <div class="form-group">
                    <label class="form-label">用户名</label>
                    <input type="text" name="username" class="form-input"
                           placeholder="请输入用户名" autofocus autocomplete="username" required>
                </div>
                <div class="form-group">
                    <label class="form-label">密码</label>
                    <input type="password" name="password" class="form-input"
                           placeholder="请输入密码" autocomplete="current-password" required>
                </div>
                <button type="submit" class="btn btn-primary login-btn">登录</button>
            </form>
        </div>
    </div>
</body>
</html>"""
