"""Isolated loopback Dashboard for ordinary browser tests, never inference.

Only a temporary database/config/data tree is used. No scheduler, runner,
trainer, go2rtc, camera, or source video is started. Test credentials have no
relationship to the user's real database.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from run_local_functional_tests import FunctionalGuard, NoModelEngines

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.meta_path.insert(0, NoModelEngines())
    guard = FunctionalGuard()
    guard.pytest_configure(None)

    import uvicorn

    from argus.capture.manager import CameraManager
    from argus.config.loader import save_config
    from argus.config.schema import ArgusConfig, AuthConfig, CameraConfig, DashboardConfig, StorageConfig
    from argus.core.health import HealthMonitor
    from argus.dashboard.api_response import api_unavailable
    from argus.dashboard.app import create_app
    from argus.dashboard.auth import hash_password
    from argus.storage.database import Database
    from starlette.responses import JSONResponse

    output = ROOT / ".local-validation"
    output.mkdir(exist_ok=True)
    data = Path(tempfile.mkdtemp(prefix="dashboard-", dir=output))
    storage = StorageConfig(**{
        name: data / name.removesuffix("_dir")
        for name in StorageConfig.model_fields if name.endswith("_dir")
    }, database_url=f"sqlite:///{(data / 'functional.db').as_posix()}")
    for name in StorageConfig.model_fields:
        if name.endswith("_dir"):
            getattr(storage, name).mkdir(parents=True, exist_ok=True)
    config = ArgusConfig(
        auth=AuthConfig(enabled=True, api_token="isolated-local-functional-tests-only"),
        dashboard=DashboardConfig(host="127.0.0.1", port=args.port, go2rtc_enabled=False),
        storage=storage,
        cameras=[CameraConfig(camera_id="local-camera", name="本机普通功能测试相机（未连接）",
                              protocol="file", source="disabled-no-source-video")],
    )
    config.cameras[0].scene_guard.profiles_dir = str(data / "scene_profiles")
    config_path = data / "config.yaml"
    save_config(config, config_path)
    db = Database(storage.database_url)
    db.initialize()
    for username, role in (("local-admin", "admin"), ("local-viewer", "viewer")):
        db.create_user(username, hash_password("Argus-local-only!42"), role, username)
    db.save_alert(
        alert_id="local-functional-alert", camera_id="local-camera", zone_id="default",
        timestamp=datetime.now(timezone.utc), severity="medium", anomaly_score=0.8,
        detection_type="test_fixture", evidence_summary="人工构造的普通功能测试记录，不是识别结果",
    )
    manager = CameraManager(config.cameras, config.alerts, database=db, baselines_dir=storage.baselines_dir)
    app = create_app(database=db, camera_manager=manager, health_monitor=HealthMonitor(),
                     config=config, config_path=str(config_path), alerts_dir=str(storage.alerts_dir))

    class NoComputeOrMedia:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                path = scope["path"]
                if path == "/__local-functional__/identity":
                    await JSONResponse({
                        "schema_version": "argus.local-functional-server.v1",
                        "pid": os.getpid(), "isolated_database": True,
                        "run_id": os.environ.get("ARGUS_FUNCTIONAL_RUN_ID"),
                        "training_disabled": True, "inference_disabled": True,
                    })(scope, receive, send)
                    return
                if any(part in path for part in ("/stream", "/snapshot", "/webrtc", "/mse")) or (
                    scope["method"] not in {"GET", "HEAD", "OPTIONS"}
                    and path.startswith(("/api/models", "/api/baseline", "/api/scene-workflow", "/api/cameras"))
                ):
                    await api_unavailable("本机普通功能测试：真实媒体、模型执行及摄像头控制已禁用")(scope, receive, send)
                    return
            await self.app(scope, receive, send)

    app.add_middleware(NoComputeOrMedia)
    print(f"Isolated functional Dashboard: http://127.0.0.1:{args.port}", flush=True)
    print(f"Temporary data: {data}", flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        db.close()


if __name__ == "__main__":
    main()
