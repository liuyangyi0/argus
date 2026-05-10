"""Lint guard: forbid Starlette ``BaseHTTPMiddleware`` in src/argus/.

Why: ``BaseHTTPMiddleware`` buffers the entire body of a ``StreamingResponse``
through an internal asyncio queue, blocking the event loop for long-lived
MJPEG streams and starving other requests. All middleware must be written
as pure ASGI (``__call__(scope, receive, send)``).

References inside comments/docstrings are fine — we only forbid actual import
or subclassing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "argus"

# Forbidden patterns: import statement or class declaration that subclasses it.
_FORBIDDEN = [
    re.compile(r"^\s*from\s+starlette\.middleware\.base\s+import\s+.*BaseHTTPMiddleware"),
    re.compile(r"^\s*from\s+starlette\.middleware\s+import\s+.*\bbase\b"),
    re.compile(r"^\s*class\s+\w+\s*\([^)]*BaseHTTPMiddleware[^)]*\)\s*:"),
]


def _strip_comment(line: str) -> str:
    """Remove trailing # ... comment from a Python source line."""
    in_str = None
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if in_str:
            if c == "\\":
                out.append(c)
                i += 1
                if i < len(line):
                    out.append(line[i])
                    i += 1
                continue
            if c == in_str:
                in_str = None
            out.append(c)
        else:
            if c in ("'", '"'):
                in_str = c
                out.append(c)
            elif c == "#":
                break
            else:
                out.append(c)
        i += 1
    return "".join(out)


def _iter_python_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_base_http_middleware_usage() -> None:
    offenders: list[str] = []
    for path in _iter_python_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, raw in enumerate(text.splitlines(), start=1):
            code = _strip_comment(raw)
            if "BaseHTTPMiddleware" not in code:
                continue
            for pat in _FORBIDDEN:
                if pat.search(code):
                    offenders.append(f"{path.relative_to(SRC_ROOT.parents[1])}:{lineno}: {raw.strip()}")
                    break

    if offenders:
        msg = (
            "Starlette BaseHTTPMiddleware is forbidden — it buffers StreamingResponse and "
            "blocks MJPEG streams. Reimplement as pure ASGI middleware "
            "(__call__(scope, receive, send)). Offenders:\n  "
            + "\n  ".join(offenders)
        )
        pytest.fail(msg)
