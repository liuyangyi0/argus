"""Core browser contracts must not fall back to unbounded TypeScript any."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_core_frontend_contracts_do_not_use_typescript_any():
    targets = [
        path
        for path in (ROOT / "web" / "src" / "api").rglob("*.ts")
        if "__tests__" not in path.parts
    ]
    targets.extend([
        ROOT / "web" / "src" / "composables" / "useWebSocket.ts",
        ROOT / "web" / "src" / "composables" / "useReplayController.ts",
        ROOT / "web" / "src" / "composables" / "useCanvasCompositor.ts",
        ROOT / "web" / "src" / "components" / "replay" / "ReplaySignalTracks.vue",
        ROOT / "web" / "src" / "types" / "api.ts",
    ])

    violations: list[str] = []
    for path in targets:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bany\b", line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")

    assert not violations, (
        "Use a DTO, generic, type guard, or safe `unknown` at API/WS boundaries:\n"
        + "\n".join(violations)
    )
