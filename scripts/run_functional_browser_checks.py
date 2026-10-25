"""Start and stop the isolated no-compute Dashboard around browser validation."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--project")
    parser.add_argument("--grep")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    base_url = f"http://127.0.0.1:{args.port}"
    output = ROOT / ".local-validation"
    output.mkdir(exist_ok=True)
    run_id = secrets.token_urlsafe(32)
    with (output / "functional-server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/run_local_functional_server.py"),
             "--port", str(args.port)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "ARGUS_FUNCTIONAL_RUN_ID": run_id},
        )
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Isolated Dashboard exited; see functional-server.log")
                try:
                    with urlopen(base_url + "/__local-functional__/identity", timeout=1) as response:
                        identity = json.load(response)
                    # Windows venv launchers may have a different PID from Python.
                    # A per-start identity still rejects an existing server on this port.
                    if not isinstance(identity, dict) or identity.get("run_id") != run_id:
                        raise RuntimeError("Port belongs to another Dashboard process")
                    break
                except (URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise RuntimeError("Isolated Dashboard startup timed out")
            command = [sys.executable, str(ROOT / "scripts/run_local_frontend_tests.py"),
                       "e2e", "--base-url", base_url, "--output-name", "playwright-ci"]
            if args.project:
                command.extend(["--project", args.project])
            if args.grep:
                command.extend(["--grep", args.grep])
            return subprocess.call(command, cwd=ROOT)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
