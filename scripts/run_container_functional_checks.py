"""Validate installed-wheel delivery before the source-based contract tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    scripts = Path(__file__).resolve().parent
    result = subprocess.call([sys.executable, "-I", str(scripts / "check_installed_dashboard.py")])
    if result:
        return result
    return subprocess.call([
        sys.executable, str(scripts / "run_local_functional_tests.py"),
        "--suite", "core", "--filter",
        "config_loader or api_contract or startup_security or health_readiness_api",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
