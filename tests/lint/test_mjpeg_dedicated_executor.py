"""Lint guard: MJPEG encoding paths must use the dedicated thread pool.

Why: ``cv2.imencode`` takes 5-15 ms per frame; on a long-lived MJPEG stream
that's hundreds of jobs per second. If those jobs land on the asyncio default
thread pool they starve regular API requests sharing it. The streaming code
must dispatch through the dedicated ``_STREAM_EXECUTOR`` instead.

This test enforces two specific patterns:

1. Module ``src/argus/dashboard/routes/cameras.py`` must declare a
   ``_STREAM_EXECUTOR`` symbol (a ``ThreadPoolExecutor``).
2. Across all of ``src/argus/``, no source line may combine
   ``asyncio.to_thread(`` with ``cv2.imencode`` (within the same call) — that
   is the exact anti-pattern documented in CLAUDE.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "argus"
CAMERAS_ROUTE = SRC_ROOT / "dashboard" / "routes" / "cameras.py"


def _iter_python_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_stream_executor_declared_in_cameras_route() -> None:
    """The dedicated ThreadPoolExecutor for MJPEG must exist in the route file."""
    assert CAMERAS_ROUTE.is_file(), f"missing {CAMERAS_ROUTE}"
    text = CAMERAS_ROUTE.read_text(encoding="utf-8")
    assert re.search(
        r"^_STREAM_EXECUTOR\s*=\s*.*ThreadPoolExecutor", text, re.MULTILINE
    ), "_STREAM_EXECUTOR ThreadPoolExecutor must be declared at module level in cameras.py"


def test_no_asyncio_to_thread_with_imencode() -> None:
    """`asyncio.to_thread(cv2.imencode, ...)` is the documented anti-pattern."""
    pattern = re.compile(r"asyncio\.to_thread\s*\([^)]*imencode")
    offenders: list[str] = []
    for path in _iter_python_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(
                    f"{path.relative_to(SRC_ROOT.parents[1])}:{lineno}: {line.strip()}"
                )
    if offenders:
        pytest.fail(
            "Forbidden: asyncio.to_thread(cv2.imencode, ...) starves the default "
            "thread pool under MJPEG streaming load. Use a dedicated "
            "ThreadPoolExecutor (see _STREAM_EXECUTOR in dashboard/routes/cameras.py).\n  "
            + "\n  ".join(offenders)
        )
