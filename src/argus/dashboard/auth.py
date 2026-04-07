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
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from argus.config.schema import AuthConfig
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
    sig = hmac.new(secret.encode(), data.encode(), "sha256").hexdigest()[:16]
    return base64.urlsafe_b64encode(f"{data}|{sig}".encode()).decode()


def verify_session_token(token: str, secret: str, max_age: int = 28800) -> dict | None:
    """Verify and decode a session token. Returns user dict or None."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        data_str, sig = decoded.rsplit("|", 1)
        expected_sig = hmac.new(secret.encode(), data_str.encode(), "sha256").hexdigest()[:16]
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
            return JSONResponse({"error": "权限不足", ...}, status_code=403)
    """
    return has_permission(request, permission)


def get_denied_response(request: Request, permission: str) -> dict:
    """Build a standardized 403 response body for permission denial.

    Returns dict suitable for JSONResponse(body, status_code=403).
    """
    user = getattr(request.state, "user", None)
    role = user.get("role", "unknown") if user else "anonymous"
    return {
        "error": "权限不足",
        "required_permission": permission,
        "your_role": role,
        "message": f"角色 '{role}' 没有 '{permission}' 权限",
    }


# ── Auth Middleware ──

class AuthMiddleware(BaseHTTPMiddleware):
    """Session + API-token authentication middleware.

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
        super().__init__(app)
        self._enabled = config.enabled
        self._token_hash = (
            hashlib.sha256(config.api_token.encode()).hexdigest()
            if config.api_token else ""
        )
        self._secret = session_secret
        self._max_age = config.session_timeout_minutes * 60
        self._database = database

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled:
            # Auth disabled: set a default admin user so role checks still work
            request.state.user = {"username": "system", "role": "admin"}
            return await call_next(request)

        path = request.url.path

        # Static files are always public
        if path.startswith("/static/"):
            return await call_next(request)

        # Public paths
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Try session cookie first
        session_token = request.cookies.get(_SESSION_COOKIE)
        if session_token:
            user_info = verify_session_token(session_token, self._secret, self._max_age)
            if user_info:
                request.state.user = user_info
                return await call_next(request)

        # Try X-API-Token header (legacy)
        api_token = request.headers.get("x-api-token", "")
        if api_token and self._verify_api_token(api_token):
            request.state.user = {"username": "api", "role": "admin"}
            return await call_next(request)

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
                        return await call_next(request)
                except Exception:
                    pass

        # Not authenticated — redirect browsers to /login, return JSON for API
        accepts_html = "text/html" in request.headers.get("accept", "")
        is_htmx = request.headers.get("hx-request") == "true"

        if accepts_html and not is_htmx:
            redirect_url = f"/login?next={request.url.path}"
            return RedirectResponse(url=redirect_url, status_code=302)

        return JSONResponse(
            {"error": "Authentication required"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="Argus"'},
        )

    def _verify_api_token(self, token: str) -> bool:
        """Constant-time token comparison to prevent timing attacks."""
        if not self._token_hash:
            return False
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return secrets.compare_digest(token_hash, self._token_hash)


# ── Rate Limit Middleware ──

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter for POST endpoints.

    Limits requests per IP per minute. Uses a sliding window counter.
    """

    def __init__(self, app: ASGIApp, max_requests_per_minute: int = 60):
        super().__init__(app)
        self._max_rpm = max_requests_per_minute
        self._counters: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only rate-limit POST/DELETE
        if request.method not in ("POST", "DELETE"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
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
            return JSONResponse(
                {"error": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )

        self._counters[client_ip].append(now)
        return await call_next(request)


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


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Cache-Control"] = "no-store"
        return response


# ── Login / Logout routes ──

auth_router = APIRouter()


@auth_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login form."""
    error = request.query_params.get("error", "")
    next_url = request.query_params.get("next", "/")
    error_html = (
        f'<div class="login-error">{error}</div>' if error else ""
    )
    return HTMLResponse(_LOGIN_PAGE_HTML.format(
        error_html=error_html,
        next_url=next_url,
    ))


@auth_router.post("/login")
async def login_handler(request: Request):
    """Handle login form submission."""
    form = await parse_request_form(request)
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    next_url = str(form.get("next", "/"))

    # Sanitize next_url to prevent open redirect
    if not next_url.startswith("/"):
        next_url = "/"

    db = getattr(request.app.state, "db", None)
    secret = getattr(request.app.state, "session_secret", "default-secret")
    auth_enabled = getattr(request.app.state, "auth_enabled", True)

    if not auth_enabled:
        # Auth disabled — accept any login
        token = create_session_token(username or "guest", "admin", secret)
        response = RedirectResponse(url=next_url, status_code=302)
        response.set_cookie(
            _SESSION_COOKIE, token,
            httponly=True, samesite="lax", max_age=28800,
        )
        return response

    if not db or not username or not password:
        return RedirectResponse(
            url=f"/login?error=用户名或密码不能为空&next={next_url}",
            status_code=302,
        )

    user = db.get_user(username)
    if user is None or not user.active or not verify_password(password, user.password_hash):
        logger.warning("auth.login_failed", username=username)
        return RedirectResponse(
            url=f"/login?error=用户名或密码错误&next={next_url}",
            status_code=302,
        )

    # Update last_login timestamp
    db.update_user(username, last_login=datetime.now(timezone.utc))

    token = create_session_token(user.username, user.role, secret)
    response = RedirectResponse(url=next_url, status_code=302)
    response.set_cookie(
        _SESSION_COOKIE, token,
        httponly=True, samesite="lax", max_age=28800,
    )
    logger.info("auth.login_success", username=username, role=user.role)
    return response


@auth_router.get("/logout")
async def logout(request: Request):
    """Clear session cookie and redirect to /login."""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(_SESSION_COOKIE)
    return response


# ── Login page HTML ──

_LOGIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Argus - 登录</title>
    <link rel="stylesheet" href="/static/css/argus.css">
    <style>
        .login-container {{
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0d0f17;
        }}
        .login-card {{
            background: #161822;
            border: 1px solid #2a2d37;
            border-radius: 10px;
            padding: 40px 36px;
            width: 360px;
            max-width: 95vw;
        }}
        .login-logo {{
            font-size: 28px;
            font-weight: 700;
            color: #4fc3f7;
            letter-spacing: 4px;
            text-align: center;
            margin-bottom: 6px;
        }}
        .login-subtitle {{
            text-align: center;
            color: #8890a0;
            font-size: 13px;
            margin-bottom: 32px;
        }}
        .login-error {{
            background: rgba(244,67,54,0.12);
            border: 1px solid #f44336;
            color: #f44336;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 13px;
            margin-bottom: 16px;
            text-align: center;
        }}
        .login-btn {{
            width: 100%;
            padding: 10px;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-card">
            <div class="login-logo">ARGUS</div>
            <div class="login-subtitle">核电站异物检测系统</div>
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
