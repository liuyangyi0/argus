"""Lint guard: dashboard routes must NOT hardcode audit user as 'operator'.

Reasoning: audit logs are useless for forensics if every action is attributed
to a placeholder username. Routes should pull the actual logged-in user from
``request.state.user`` via ``argus.dashboard.auth.current_username(request)``.
This guard runs every commit so the regression is caught immediately.
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTES_DIR = Path(__file__).parent.parent.parent / "src" / "argus" / "dashboard"

# Match user="operator" and submitted_by="operator" with single or double quotes.
_HARDCODE_PATTERNS = [
    re.compile(r'user\s*=\s*[\'"]operator[\'"]'),
    re.compile(r'submitted_by\s*=\s*[\'"]operator[\'"]'),
]


def test_no_hardcoded_operator_in_audit_calls():
    offending: list[str] = []
    for py in ROUTES_DIR.rglob("*.py"):
        # Self-skip: the regex literals in this very file would otherwise
        # be matched if the lint suite copied this file under dashboard/.
        # The lint file lives under tests/, not dashboard/, so no real
        # collision — but be defensive in case folks restructure.
        if py.name == "test_no_hardcoded_audit_operator.py":
            continue
        text = py.read_text(encoding="utf-8")
        for pattern in _HARDCODE_PATTERNS:
            for m in pattern.finditer(text):
                line_no = text[: m.start()].count("\n") + 1
                offending.append(
                    f"{py.relative_to(ROUTES_DIR.parent.parent.parent)}:{line_no}"
                    f"  {m.group(0)}"
                )

    assert not offending, (
        "Routes must not hardcode audit username as 'operator'. "
        "Use current_username(request) from argus.dashboard.auth so the "
        "actual logged-in user is recorded. Offending sites:\n  "
        + "\n  ".join(offending)
    )
