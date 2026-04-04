"""Authentication middleware for the Argus dashboard.

Provides HTTP Basic Auth with a shared API token. Designed for nuclear plant
environments where simplicity and reliability matter more than OAuth complexity.

Public endpoints (no auth required):
- /api/system/health  (Docker healthcheck)
- /metrics            (Prometheus scraping)
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import defaultdict

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from argus.config.schema import AuthConfig

logger = structlog.get_logger()

# Endpoints that bypass authentication
_PUBLIC_PATHS = frozenset({
    "/api/system/health",
    "/metrics",
})


class AuthMiddleware(BaseHTTPMiddleware):
    """Token-based authentication middleware.

    Accepts the API token via:
    - HTTP Basic Auth (username ignored, password = token)
    - X-API-Token header
    """

    def __init__(self, app: ASGIApp, config: AuthConfig):
        super().__init__(app)
        self._enabled = config.enabled
        self._token_hash = hashlib.sha256(config.api_token.encode()).hexdigest() if config.api_token else ""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled:
            return await call_next(request)

        # Allow public endpoints
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Check X-API-Token header
        token = request.headers.get("x-api-token", "")

        # Check HTTP Basic Auth
        if not token:
            from base64 import b64decode
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Basic "):
                try:
                    decoded = b64decode(auth_header[6:]).decode("utf-8")
                    # username:password — we only care about the password
                    _, _, token = decoded.partition(":")
                except Exception:
                    pass

        if not token or not self._verify_token(token):
            return JSONResponse(
                {"error": "Authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Argus"'},
            )

        return await call_next(request)

    def _verify_token(self, token: str) -> bool:
        """Constant-time token comparison to prevent timing attacks."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return secrets.compare_digest(token_hash, self._token_hash)


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
        entries = self._counters[client_ip]
        self._counters[client_ip] = [t for t in entries if now - t < window]
        entries = self._counters[client_ip]

        if len(entries) >= self._max_rpm:
            logger.warning("rate_limit.exceeded", client_ip=client_ip, count=len(entries))
            return JSONResponse(
                {"error": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )

        entries.append(now)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Cache-Control"] = "no-store"
        return response
