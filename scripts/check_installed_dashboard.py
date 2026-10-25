"""Smoke-check the installed wheel and built UI without model/media execution.

Run with ``python -I scripts/check_installed_dashboard.py --frontend-dir web/dist``
after installing a non-editable wheel into the ordinary-function environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


class ScriptAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            src = dict(attrs).get("src", "")
            if src.startswith("/assets/"):
                self.paths.append(urlsplit(src).path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-dir", type=Path, default=Path("/app/web/dist"))
    args = parser.parse_args()
    frontend = args.frontend_dir.resolve(strict=True)
    guard_spec = importlib.util.spec_from_file_location(
        "functional_guard", Path(__file__).with_name("run_local_functional_tests.py"),
    )
    guard_module = importlib.util.module_from_spec(guard_spec)
    guard_spec.loader.exec_module(guard_module)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.meta_path.insert(0, guard_module.NoModelEngines())
    installed = importlib.metadata.distribution("argus").locate_file("argus").resolve()
    spec = importlib.util.find_spec("argus")
    if spec is None or Path(spec.origin).resolve().parent != installed:
        raise RuntimeError("Dashboard smoke must import the installed wheel, not the source checkout")
    guard = guard_module.FunctionalGuard()
    guard.pytest_configure(None)

    from fastapi.testclient import TestClient

    from argus.config.schema import ArgusConfig, DashboardConfig, StorageConfig
    from argus.dashboard.app import create_app

    index = (frontend / "index.html").read_bytes()
    assets = ScriptAssets()
    assets.feed(index.decode("utf-8"))
    if not assets.paths:
        raise RuntimeError("Built frontend has no local JavaScript assets")
    with tempfile.TemporaryDirectory(prefix="argus-wheel-smoke-") as directory:
        root = Path(directory)
        storage = StorageConfig(**{
            name: root / name.removesuffix("_dir")
            for name in StorageConfig.model_fields if name.endswith("_dir")
        }, database_url=f"sqlite:///{(root / 'unused.db').as_posix()}")
        config = ArgusConfig(
            dashboard=DashboardConfig(go2rtc_enabled=False, frontend_dist_dir=frontend),
            storage=storage,
        )
        # No database, cameras, runner, or release worker is supplied.
        with TestClient(create_app(config=config)) as client:
            for path in ("/", "/models", "/system"):
                response = client.get(path)
                assert response.status_code == 200, path
                assert response.content == index, path
                assert "text/html" in response.headers["content-type"]
            for path in assets.paths:
                response = client.get(path)
                assert response.status_code == 200, path
                assert response.content == (frontend / path.lstrip("/")).read_bytes(), path
            assert client.get("/api/system/health").status_code == 200
            assert client.get("/api/not-a-real-endpoint").status_code == 404
    if guard.violations:
        raise RuntimeError(f"Unexpected compute/media execution: {guard.violations}")
    print("Installed-wheel Dashboard smoke passed: SPA, built JavaScript, API routing; no compute/media")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
