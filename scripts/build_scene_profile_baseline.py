#!/usr/bin/env python3
"""Compatibility CLI for the shared fail-closed scene baseline builder.

The reusable implementation lives in ``argus.validation.scene_profile_baseline``.
Importing this historical entry point returns that same module so existing
callers and dependency-injection hooks cannot accidentally diverge from the
Dashboard implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from argus.validation import scene_profile_baseline as _implementation
except ModuleNotFoundError as exc:
    if exc.name != "argus":
        raise
    # Support direct execution from a checkout without an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from argus.validation import scene_profile_baseline as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    # Preserve imports of the historical API, including attributes monkeypatched
    # by callers: function globals must always be the canonical implementation.
    sys.modules[__name__] = _implementation
