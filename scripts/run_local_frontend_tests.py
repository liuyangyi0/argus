"""Run frontend-only validation with workspace-local Windows temporary files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("unit", "types", "build", "e2e"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--grep", help="Select a browser test by title")
    parser.add_argument("--project", help="Select one configured browser viewport")
    parser.add_argument("--output-name", default="playwright")
    args = parser.parse_args()
    if args.task == "e2e":
        url = urlsplit(args.base_url)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment):
            parser.error("E2E requires the isolated loopback functional server")
        with urlopen(args.base_url.rstrip("/") + "/__local-functional__/identity", timeout=5) as response:
            identity = json.load(response)
        if (not isinstance(identity, dict)
                or identity.get("schema_version") != "argus.local-functional-server.v1"
                or any(identity.get(key) is not True for key in (
                    "isolated_database", "training_disabled", "inference_disabled",
                ))):
            parser.error("Refusing E2E writes against a non-isolated Dashboard")
    node = shutil.which("node")
    if not node:
        parser.error("Node.js is required")
    output = ROOT / ".local-validation"
    temporary = output / "temp"
    temporary.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(temporary), TMP=str(temporary))
    web = ROOT / "web"

    def run(relative: str, *options: str) -> int:
        return subprocess.call([node, str(web / relative), *options], cwd=web, env=env)

    if args.task == "unit":
        return run("node_modules/vitest/vitest.mjs", "run", "--maxWorkers=2",
                   "--reporter=default", "--reporter=json", f"--outputFile={output / 'vitest.json'}")
    if args.task in {"types", "build"}:
        status = run("node_modules/vue-tsc/bin/vue-tsc.js", "-b")
        if status or args.task == "types":
            return status
        return run("node_modules/vite/bin/vite.js", "build")
    env["ARGUS_E2E_BASE_URL"] = args.base_url
    if not args.output_name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in args.output_name):
        parser.error("output-name must be a simple lowercase artifact directory name")
    env["ARGUS_E2E_OUTPUT_DIR"] = str(output / args.output_name)
    env["ARGUS_E2E_USERNAME"] = "local-admin"
    env["ARGUS_E2E_PASSWORD"] = "Argus-local-only!42"
    env["ARGUS_E2E_VIEWER_USERNAME"] = "local-viewer"
    env["ARGUS_E2E_VIEWER_PASSWORD"] = "Argus-local-only!42"
    env["ARGUS_E2E_CAMERA_ID"] = "local-camera"
    options = ["test", "e2e/local-functional.spec.ts"]
    if args.grep:
        options.extend(["--grep", args.grep])
    if args.project:
        options.extend(["--project", args.project])
    return run("node_modules/@playwright/test/cli.js", *options)


if __name__ == "__main__":
    raise SystemExit(main())
