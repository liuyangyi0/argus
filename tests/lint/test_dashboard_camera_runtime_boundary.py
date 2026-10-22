"""Keep Dashboard routes behind CameraManager's public runtime contract."""

from __future__ import annotations

from pathlib import Path


ROUTE_DIR = Path(__file__).resolve().parents[2] / "src" / "argus" / "dashboard" / "routes"
ROUTES = [
    ROUTE_DIR / "cameras.py",
    ROUTE_DIR / "config.py",
    ROUTE_DIR / "models.py",
    ROUTE_DIR / "physics.py",
    ROUTE_DIR / "system.py",
    ROUTE_DIR / "zones.py",
]
RUNTIME_CONTRACT = (
    Path(__file__).resolve().parents[2] / "src" / "argus" / "core" / "runtime.py"
)
BOOTSTRAP = Path(__file__).resolve().parents[2] / "src" / "argus" / "__main__.py"

BANNED_ACCESS = {
    "camera_manager._cameras": "use get_camera_config",
    "manager._cameras": "use get_camera_config",
    "camera_manager._pipelines": "use a CameraManager snapshot method",
    "manager._pipelines": "use a CameraManager snapshot method",
    "camera_manager._runners": "use a CameraManager snapshot method",
    "manager._runners": "use a CameraManager snapshot method",
    "runner._pipeline": "do not unwrap CameraInferenceRunner in Dashboard code",
    "pipeline._anomaly_detector": "use get_anomaly_inference_handle/status snapshot",
    "pipeline._physics_tracker": "use get_active_tracks_snapshot",
    "tracker._tracks": "use get_active_tracks_snapshot",
    "camera_manager.get_pipeline(": "add a purpose-specific manager operation",
    "camera_manager._correlator": "use get_runtime_component_summary",
}


def test_dashboard_camera_routes_do_not_read_runtime_private_fields() -> None:
    violations: list[str] = []
    for path in ROUTES:
        source = path.read_text(encoding="utf-8")
        for token, replacement in BANNED_ACCESS.items():
            if token in source:
                violations.append(f"{path.name}: {token!r}; {replacement}")

    assert not violations, "Dashboard runtime boundary violations:\n" + "\n".join(violations)


def test_public_camera_runtime_does_not_expose_release_application() -> None:
    source = RUNTIME_CONTRACT.read_text(encoding="utf-8")
    assert "def apply_release(" not in source, (
        "CameraRuntime is a Dashboard/orchestration view; all model publication "
        "must enter through ModelReleaseService's prepared transaction"
    )


def test_application_bootstrap_does_not_read_manager_runtime_collections() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "manager._pipelines" not in source
    assert "manager._runners" not in source
