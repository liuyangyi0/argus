"""OpenAPI-derived frontend contracts must be checked in and deterministic."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "web" / "src" / "types" / "generated-openapi.ts"


def test_generated_openapi_types_are_current_and_never_use_any():
    result = subprocess.run(
        [sys.executable, "scripts/generate_openapi_types.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    source = GENERATED.read_text(encoding="utf-8")
    assert not re.search(r"\bany\b", source), (
        "Generated contracts must use safe `unknown` for unconstrained schemas"
    )
    for prefix in (
        "alerts",
        "cameras",
        "config",
        "models",
        "physics",
        "replay",
        "system",
    ):
        assert f"/api/{prefix}" in source
