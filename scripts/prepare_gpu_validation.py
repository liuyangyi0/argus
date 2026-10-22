#!/usr/bin/env python3
"""Prepare an isolated, fail-closed Argus runtime on gpu-validation.

The script intentionally creates a validation-only scene profile and a canary
registry row.  It never writes to the source checkout, the production database,
or a production model stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from argus.config.loader import load_config  # noqa: E402
from argus.anomaly.detector import AnomalibDetector  # noqa: E402
from argus.preprocessing.scene_profile import (  # noqa: E402
    RUNTIME_ARTIFACT_HASH_SCHEME,
    SceneProfileStore,
    runtime_artifact_sha256,
)
from argus.preprocessing.scene_guard import (  # noqa: E402
    SceneGuard,
    SceneGuardConfig as SceneGuardAlgorithmConfig,
    SceneState,
)
from argus.storage.database import Database  # noqa: E402
from argus.storage.model_registry import ModelRegistry  # noqa: E402
from argus.storage.model_release_service import (  # noqa: E402
    ModelReleaseService,
    RuntimeReleaseCommit,
    RuntimeReleaseIdentity,
    RuntimeReleasePreparation,
)
from argus.storage.models import ModelStage  # noqa: E402
from argus.storage.release_pipeline import ReleasePipeline  # noqa: E402


DEFAULT_EXEC_SOURCE = (
    "exec:ffmpeg -hide_banner -loglevel warning -f v4l2 "
    "-input_format mjpeg -video_size 1920x1080 -framerate 60 "
    "-i /dev/video0 -an -c:v h264_nvenc -preset ll -tune ull "
    "-rc constqp -qp 24 -pix_fmt yuv420p -f rtsp "
    "-rtsp_transport tcp {output}"
    "#killsignal=sigkill#video=h264#starttimeout=20"
)
def _production_checkout() -> Path:
    """Resolve the runtime account, not caller-supplied HOME, on Linux."""
    if os.name == "posix":
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir) / "argus"
    return Path.home() / "argus"


PRODUCTION_CHECKOUT = _production_checkout()


class _ColdStartRuntimeTransaction:
    """Hold a CUDA-loaded standby until the release CAS commits."""

    def __init__(self, detector: AnomalibDetector) -> None:
        self._detector: AnomalibDetector | None = detector
        self._committed = False

    def commit(self) -> RuntimeReleaseCommit:
        if self._detector is None:
            return RuntimeReleaseCommit(False, error="cold-start standby was released")
        self._committed = True
        return RuntimeReleaseCommit(True, state={"cold_start_cuda_loaded": True})

    def abort(self) -> RuntimeReleaseCommit:
        self._release()
        return RuntimeReleaseCommit(True, state={"cold_start_cuda_released": True})

    def finalize(self) -> RuntimeReleaseCommit:
        self._release()
        return RuntimeReleaseCommit(True, state={"cold_start_cuda_released": True})

    def _release(self) -> None:
        detector = self._detector
        self._detector = None
        if detector is not None:
            setattr(detector, "_engine", None)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _cold_start_runtime_prepare(plan) -> RuntimeReleasePreparation:
    if not plan.model_path or not plan.runtime_artifact_sha256:
        return RuntimeReleasePreparation(
            success=False,
            error="cold-start release has no exact runtime artifact identity",
        )
    try:
        before_sha256 = runtime_artifact_sha256(plan.model_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return RuntimeReleasePreparation(
            success=False,
            error=f"cold-start artifact fingerprint failed: {exc}",
        )
    if before_sha256 != plan.runtime_artifact_sha256:
        return RuntimeReleasePreparation(
            success=False,
            error="cold-start artifact changed before CUDA load",
        )
    detector = AnomalibDetector(
        model_path=plan.model_path,
        inference_backend="torch-cuda",
        camera_id=plan.camera_id,
    )
    try:
        loaded = detector.load()
    except Exception as exc:
        return RuntimeReleasePreparation(
            success=False,
            error=f"torch-cuda cold-start load failed: {type(exc).__name__}: {exc}",
        )
    if loaded is not True or detector.status.loaded is not True:
        return RuntimeReleasePreparation(
            success=False,
            error="torch-cuda cold-start load did not produce a loaded model",
        )
    transaction = _ColdStartRuntimeTransaction(detector)
    if detector.status.backend != "torch-cuda":
        transaction.abort()
        return RuntimeReleasePreparation(
            success=False,
            error=f"forbidden backend fallback: {detector.status.backend}",
        )
    if detector.status.model_path != str(plan.model_path):
        transaction.abort()
        return RuntimeReleasePreparation(
            success=False,
            error="CUDA detector loaded a different model path",
        )
    try:
        after_sha256 = runtime_artifact_sha256(plan.model_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        transaction.abort()
        return RuntimeReleasePreparation(
            success=False,
            error=f"post-load artifact fingerprint failed: {exc}",
        )
    if after_sha256 != before_sha256:
        transaction.abort()
        return RuntimeReleasePreparation(
            success=False,
            error="cold-start artifact changed during CUDA load",
        )
    return RuntimeReleasePreparation(
        success=True,
        identity=RuntimeReleaseIdentity(
            camera_id=plan.camera_id,
            model_version_id=plan.model_version_id,
            target_stage=plan.target_stage,
            runtime_artifact_sha256=plan.runtime_artifact_sha256,
        ),
        transaction=transaction,
        state={
            "backend": "torch-cuda",
            "cold_start": True,
            "bootstrap_verified": True,
        },
    )


def _profile_bound_runtime_prepare(
    store: SceneProfileStore,
    plan,
) -> RuntimeReleasePreparation:
    if plan.target_stage == ModelStage.PRODUCTION.value:
        return RuntimeReleasePreparation(
            False,
            error="validation bootstrap can never prepare a production release",
        )
    profile = store.get_active(plan.camera_id)
    if profile is None:
        return RuntimeReleasePreparation(False, error="active validation profile missing")
    metadata = profile.metadata
    if metadata.get("validation_only") is not True:
        return RuntimeReleasePreparation(
            False,
            error="cold-start bootstrap accepts validation-only profiles exclusively",
        )
    if metadata.get("operator_approved") is not False:
        return RuntimeReleasePreparation(
            False,
            error="validation profile must not claim production operator approval",
        )
    if metadata.get("model_version_id") != plan.model_version_id:
        return RuntimeReleasePreparation(False, error="scene profile/model mismatch")
    if metadata.get("model_artifact_sha256") != plan.runtime_artifact_sha256:
        return RuntimeReleasePreparation(False, error="scene profile/artifact mismatch")
    return _cold_start_runtime_prepare(plan)


def _absolute(path: Path, *, label: str, file: bool | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if file is True and not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    if file is False and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _executable_path(path: Path, *, label: str) -> Path:
    """Make the parent absolute without dereferencing a venv interpreter link."""
    expanded = path.expanduser()
    candidate = expanded if expanded.is_absolute() else Path.cwd() / expanded
    preserved = candidate.parent.resolve() / candidate.name
    if not preserved.is_file():
        raise ValueError(f"{label} must be a file: {preserved}")
    return preserved


def _state_path(state_dir: Path, *parts: str) -> Path:
    path = state_dir.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scene_reference(reference_path: Path, scene_config: Any) -> None:
    """Fail before registry bootstrap when the supplied profile cannot be trusted."""
    frame = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("baseline reference cannot be decoded")

    algorithm_config = SceneGuardAlgorithmConfig(
        process_width=scene_config.process_width,
        feature_type=scene_config.feature_type,
        min_matches=scene_config.min_matches,
        min_inliers=scene_config.min_inliers,
        min_inlier_ratio=scene_config.min_inlier_ratio,
        max_reprojection_error_px=scene_config.max_reprojection_error_px,
        min_coverage_ratio=scene_config.min_coverage_ratio,
        min_overlap_ratio=scene_config.min_overlap_ratio,
        stable_max_shift_px=scene_config.stable_max_shift_px,
        stable_max_corner_displacement_px=(
            scene_config.stable_max_corner_displacement_px
        ),
        stable_max_rotation_deg=scene_config.stable_max_rotation_deg,
        stable_max_scale_delta=scene_config.stable_max_scale_delta,
        alignable_max_shift_px=scene_config.alignable_max_shift_px,
        alignable_max_rotation_deg=scene_config.alignable_max_rotation_deg,
        alignable_max_scale_delta=scene_config.alignable_max_scale_delta,
        consecutive_matches=scene_config.consecutive_matches,
        return_aligned_frame=False,
    )
    try:
        guard = SceneGuard(reference_frame=frame, config=algorithm_config)
    except (ValueError, RuntimeError, cv2.error) as exc:
        raise ValueError(
            f"baseline reference cannot authorize SceneGuard: {exc}"
        ) from exc

    result = None
    for _ in range(algorithm_config.consecutive_matches):
        result = guard.evaluate(frame)
    if result is None or result.state is not SceneState.STABLE or not result.usable:
        reason = "no evaluation result" if result is None else result.reason
        raise ValueError(
            "baseline reference cannot reach a stable SceneGuard state: " + reason
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_isolated_config(config: Any, state_dir: Path) -> None:
    expected_db = state_dir / "data" / "db" / "argus-validation.db"
    if config.storage.database_url != f"sqlite:///{expected_db.as_posix()}":
        raise ValueError("effective database URL escaped the validation state")

    write_paths = [
        config.storage.baselines_dir,
        config.storage.models_dir,
        config.storage.exports_dir,
        config.storage.backbones_dir,
        config.storage.foe_objects_dir,
        config.storage.model_packages_dir,
        config.storage.alerts_dir,
        config.storage.inference_records_dir,
        config.logging.log_dir,
        config.feedback.validation_dir,
        config.continuous_recording.output_dir,
        config.dashboard.go2rtc_config_dir,
    ]
    for camera in config.cameras:
        write_paths.extend([
            camera.scene_guard.profiles_dir,
            camera.ring_buffer.archive_dir,
        ])
    for raw_path in write_paths:
        if raw_path is None:
            raise ValueError("effective validation write path is missing")
        candidate = Path(raw_path).expanduser()
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (state_dir / candidate).resolve()
        )
        if not _is_within(resolved, state_dir):
            raise ValueError(f"effective write path escaped validation state: {resolved}")

    binary = Path(config.dashboard.go2rtc_binary or "").resolve()
    if not _is_within(binary, state_dir):
        raise ValueError("effective go2rtc binary escaped validation state")


def _camera_from_base(raw: dict[str, Any], camera_id: str) -> dict[str, Any]:
    cameras = raw.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("base config must contain at least one camera")
    source = next(
        (
            item
            for item in cameras
            if isinstance(item, dict) and item.get("camera_id") == camera_id
        ),
        cameras[0],
    )
    if not isinstance(source, dict):
        raise ValueError("base camera config is invalid")
    return copy.deepcopy(source)


def _build_config(
    raw: dict[str, Any],
    *,
    state_dir: Path,
    source_root: Path,
    camera_id: str,
    go2rtc_bin: Path,
    argus_port: int,
    go2rtc_api_port: int,
    go2rtc_rtsp_port: int,
    go2rtc_webrtc_port: int,
    exec_source: str,
    api_token: str,
) -> dict[str, Any]:
    config = copy.deepcopy(raw)
    config["node_id"] = "gpu-validation-isolated-validation"

    camera = _camera_from_base(config, camera_id)
    camera.update({
        "camera_id": camera_id,
        "name": "OBSBOT Meet 2 — isolated validation",
        "source": "/dev/video0",
        "protocol": "usb",
        "fps_target": 60,
        "resolution": [1920, 1080],
    })
    usb = camera.setdefault("usb", {})
    usb.update({
        "device_name": "OBSBOT Meet 2 StreamCamera",
        "preferred_backend": "v4l2",
        "pixel_format": "mjpeg",
        "min_runtime_fps": 50.0,
        "go2rtc_source": exec_source,
    })
    anomaly = camera.setdefault("anomaly", {})
    anomaly["inference_backend"] = "torch-cuda"
    anomaly["model_type"] = "patchcore"
    anomaly.update({
        "min_shadow_days": 0,
        "min_shadow_samples": 0,
        "max_shadow_alert_rate_delta": 1.0,
        "max_shadow_score_divergence": 1.0,
        "max_shadow_latency_ms": 0.0,
        # Production remains gated even in the isolated validation registry.
        "min_canary_days": max(7, int(anomaly.get("min_canary_days", 7))),
    })
    scene_guard = camera.setdefault("scene_guard", {})
    scene_guard.update({
        "enabled": True,
        "profiles_dir": str(state_dir / "scene_profiles"),
        "reference_image": None,
        "fallback_to_baseline": False,
        "allow_alignable": False,
        "require_profile_model_binding": True,
        # The isolated profile is technical validation evidence, not production
        # operator approval. Production keeps this gate enabled.
        "require_operator_approved_profile": False,
    })
    ring_buffer = camera.setdefault("ring_buffer", {})
    ring_buffer["archive_dir"] = str(state_dir / "data" / "recordings")
    person_filter = camera.setdefault("person_filter", {})
    person_filter["model_name"] = str(source_root / "yolo11n.pt")
    config["cameras"] = [camera]

    dashboard = config.setdefault("dashboard", {})
    dashboard.update({
        "host": "127.0.0.1",
        "port": argus_port,
        "debug": False,
        "go2rtc_enabled": True,
        "go2rtc_api_port": go2rtc_api_port,
        "go2rtc_rtsp_port": go2rtc_rtsp_port,
        "go2rtc_webrtc_port": go2rtc_webrtc_port,
        "go2rtc_webrtc_listen_host": "127.0.0.1",
        "go2rtc_binary": str(go2rtc_bin),
        "go2rtc_config_dir": str(state_dir / "go2rtc"),
        "go2rtc_replace_stale_process": False,
    })
    config["auth"] = {
        "enabled": True,
        "api_token": api_token,
        "session_timeout_minutes": 120,
    }

    db_path = state_dir / "data" / "db" / "argus-validation.db"
    storage = config.setdefault("storage", {})
    storage.update({
        "database_url": f"sqlite:///{db_path.as_posix()}",
        "baselines_dir": str(state_dir / "data" / "baselines"),
        "models_dir": str(state_dir / "data" / "models"),
        "exports_dir": str(state_dir / "data" / "exports"),
        "backbones_dir": str(state_dir / "data" / "backbones"),
        "foe_objects_dir": str(state_dir / "data" / "foe_objects"),
        "model_packages_dir": str(state_dir / "data" / "model_packages"),
        "alerts_dir": str(state_dir / "data" / "alerts"),
        "inference_records_dir": str(state_dir / "data" / "inference_records"),
    })
    models = config.setdefault("models", {})
    models.update({
        "yolo_path": str(source_root / "yolo11n.pt"),
        "anomalib_model_dir": str(state_dir / "data" / "models"),
        "anomalib_export_dir": str(state_dir / "data" / "exports"),
    })
    config.setdefault("logging", {})["log_dir"] = str(state_dir / "logs" / "app")
    config.setdefault("feedback", {})["validation_dir"] = str(
        state_dir / "data" / "validation"
    )
    for section in ("classifier", "segmenter", "cross_camera", "retraining"):
        config.setdefault(section, {})["enabled"] = False
    continuous = config.setdefault("continuous_recording", {})
    continuous.update({
        "enabled": False,
        "archive_enabled": False,
        "archive_path": None,
        "output_dir": str(state_dir / "data" / "continuous_recordings"),
    })
    return config


def _write_env(
    path: Path,
    *,
    source_root: Path,
    state_dir: Path,
    config_path: Path,
    python_path: Path,
    go2rtc_bin: Path,
    source_manifest: Path,
    camera_id: str,
    argus_port: int,
    go2rtc_api_port: int,
    go2rtc_rtsp_port: int,
    go2rtc_webrtc_port: int,
    api_token: str,
    api_token_file: Path,
    model_version_id: str,
    model_stage: str,
    runtime_artifact_sha256_value: str,
    scene_profile_id: str,
    release_revision: int,
    admin_username: str,
    admin_password: str,
    viewer_username: str,
    viewer_password: str,
) -> None:
    values: dict[str, str | int] = {
        "ARGUS_ROOT": str(source_root),
        "ARGUS_STATE_DIR": str(state_dir),
        "ARGUS_CONFIG": str(config_path),
        "ARGUS_PYTHON": str(python_path),
        "GO2RTC_BIN": str(go2rtc_bin),
        "GO2RTC_CONFIG_DIR": str(state_dir / "go2rtc"),
        "SOURCE_SHA256_MANIFEST": str(source_manifest),
        "ARTIFACT_SHA256_MANIFEST": str(state_dir / "artifacts.sha256"),
        "CAMERA_ID": camera_id,
        "ARGUS_PORT": argus_port,
        "GO2RTC_API_PORT": go2rtc_api_port,
        "GO2RTC_RTSP_PORT": go2rtc_rtsp_port,
        "GO2RTC_WEBRTC_PORT": go2rtc_webrtc_port,
        "CAMERA_DEVICE": "/dev/video0",
        "STARTUP_TIMEOUT_SECONDS": 180,
        "ARGUS_API_TOKEN": api_token,
        "ARGUS_API_TOKEN_FILE": str(api_token_file),
        "MODEL_VERSION_ID": model_version_id,
        "MODEL_STAGE": model_stage,
        "RUNTIME_ARTIFACT_SHA256": runtime_artifact_sha256_value,
        "SCENE_PROFILE_ID": scene_profile_id,
        "RELEASE_REVISION": release_revision,
        "ARGUS_E2E_USERNAME": admin_username,
        "ARGUS_E2E_PASSWORD": admin_password,
        "ARGUS_E2E_VIEWER_USERNAME": viewer_username,
        "ARGUS_E2E_VIEWER_PASSWORD": viewer_password,
    }
    path.write_text(
        "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()),
        encoding="utf-8",
        newline="\n",
    )
    path.chmod(0o600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--forbidden-root",
        type=Path,
        default=PRODUCTION_CHECKOUT,
        help="Production checkout that must never contain source or validation state",
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--go2rtc-bin", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    parser.add_argument("--camera-id", default="validation_camera_01")
    parser.add_argument("--profile-id")
    parser.add_argument("--argus-port", type=int, default=18081)
    parser.add_argument("--go2rtc-api-port", type=int, default=11984)
    parser.add_argument("--go2rtc-rtsp-port", type=int, default=18554)
    parser.add_argument("--go2rtc-webrtc-port", type=int, default=18555)
    parser.add_argument("--exec-source", default=DEFAULT_EXEC_SOURCE)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    ambient_overrides = sorted(
        name for name in os.environ if name.startswith("ARGUS__")
    )
    if ambient_overrides:
        raise SystemExit(
            "refusing ambient Argus config overrides: "
            + ", ".join(ambient_overrides)
        )
    source_root = _absolute(args.source_root, label="source root", file=False)
    forbidden_root = args.forbidden_root.expanduser().resolve()
    base_config = _absolute(args.base_config, label="base config", file=True)
    source_manifest = _absolute(
        args.source_manifest, label="source manifest", file=True
    )
    python_path = _executable_path(args.python, label="Python interpreter")
    source_go2rtc_bin = _absolute(args.go2rtc_bin, label="go2rtc binary", file=True)
    source_model_artifact = _absolute(
        args.model_artifact, label="model artifact", file=True
    )
    baseline_reference = _absolute(
        args.baseline_reference, label="baseline reference", file=True
    )
    state_dir = args.state_dir.expanduser().resolve()
    protected_roots = {forbidden_root, PRODUCTION_CHECKOUT}
    if any(
        source_root == root or _is_within(source_root, root)
        for root in protected_roots
    ):
        raise SystemExit("source-root must be an isolated validation snapshot")
    if _is_within(state_dir, source_root) or any(
        state_dir == root or _is_within(state_dir, root)
        for root in protected_roots
    ):
        raise SystemExit(
            "state-dir must be outside both the source snapshot and production checkout"
        )
    if not _is_within(base_config, source_root):
        raise SystemExit("base-config must come from the immutable source snapshot")
    if not _is_within(source_manifest, state_dir):
        raise SystemExit("source-manifest must live below the validation state")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    config_path = state_dir / "argus-validation.yaml"
    env_path = state_dir / "validation.env"
    api_token_path = state_dir / "api-token"
    db_path = state_dir / "data" / "db" / "argus-validation.db"
    for owned in (config_path, env_path, api_token_path, db_path):
        if owned.exists():
            raise SystemExit(
                f"refusing to reuse validation state containing {owned}; "
                "choose a new timestamped state directory"
            )

    copied_go2rtc_bin = state_dir / "bin" / "go2rtc"
    copied_go2rtc_bin.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_go2rtc_bin, copied_go2rtc_bin)
    copied_go2rtc_bin.chmod(0o755)

    copied_model_dir = state_dir / "data" / "authorized_models" / "bootstrap"
    shutil.copytree(source_model_artifact.parent, copied_model_dir)
    model_artifact = copied_model_dir / source_model_artifact.name
    if not model_artifact.is_file():
        raise SystemExit("isolated model copy is missing the requested artifact")
    source_runtime_artifact_sha256 = runtime_artifact_sha256(
        source_model_artifact
    )
    copied_runtime_artifact_sha256 = runtime_artifact_sha256(model_artifact)
    if copied_runtime_artifact_sha256 != source_runtime_artifact_sha256:
        raise SystemExit(
            "isolated runtime artifact SHA256 mismatch (model or sidecar changed)"
        )

    api_token = secrets.token_urlsafe(36)
    admin_username = "validation-admin"
    admin_password = secrets.token_urlsafe(24)
    viewer_username = "validation-viewer"
    viewer_password = secrets.token_urlsafe(24)

    raw = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit("base config must decode to an object")
    config = _build_config(
        raw,
        state_dir=state_dir,
        source_root=source_root,
        camera_id=args.camera_id,
        go2rtc_bin=copied_go2rtc_bin,
        argus_port=args.argus_port,
        go2rtc_api_port=args.go2rtc_api_port,
        go2rtc_rtsp_port=args.go2rtc_rtsp_port,
        go2rtc_webrtc_port=args.go2rtc_webrtc_port,
        exec_source=args.exec_source,
        api_token=api_token,
    )
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    config_path.chmod(0o600)
    validated = load_config(config_path)
    _assert_isolated_config(validated, state_dir)
    if validated.cameras[0].anomaly.inference_backend != "torch-cuda":
        raise SystemExit("validation config did not preserve torch-cuda")
    try:
        _validate_scene_reference(
            baseline_reference,
            validated.cameras[0].scene_guard,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    baseline_version = "validation_reference"
    baseline_dir = (
        state_dir / "data" / "baselines" / args.camera_id / "default" / baseline_version
    )
    baseline_dir.mkdir(parents=True, exist_ok=False)
    baseline_copy = baseline_dir / baseline_reference.name
    shutil.copy2(baseline_reference, baseline_copy)
    (baseline_dir.parent / "current.txt").write_text(
        baseline_version + "\n", encoding="utf-8", newline="\n"
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    database = Database(database_url=validated.storage.database_url)
    database.initialize()
    from argus.dashboard.auth import hash_password

    database.create_user(
        admin_username,
        hash_password(admin_password),
        role="admin",
        display_name="Validation Admin",
    )
    database.create_user(
        viewer_username,
        hash_password(viewer_password),
        role="viewer",
        display_name="Validation Viewer",
    )
    db_path.chmod(0o600)
    registry = ModelRegistry(session_factory=database.get_session)
    model_version_id = registry.register(
        model_path=model_artifact,
        baseline_dir=baseline_dir,
        camera_id=args.camera_id,
        model_type="patchcore",
        training_params={
            "authority": "isolated-validation-only",
            "production_approved": False,
        },
    )

    profile_id = args.profile_id or (
        "validation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    artifact_sha256 = copied_runtime_artifact_sha256
    store = SceneProfileStore(state_dir / "scene_profiles")
    profile = store.create_from_baseline(
        args.camera_id,
        profile_id,
        baseline_copy,
        metadata={
            "operator_approved": False,
            "validation_only": True,
            "model_version_id": model_version_id,
            "model_artifact_sha256": artifact_sha256,
            "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
            "baseline_version": baseline_version,
            "authority": "isolated-validation-only",
        },
        activate=False,
    )
    store.set_active(args.camera_id, profile.profile_id)

    release_pipeline = ReleasePipeline(
        session_factory=database.get_session,
        min_shadow_days=0,
        min_canary_days=7,
        min_shadow_samples=0,
        max_shadow_alert_rate_delta=1.0,
        max_shadow_score_divergence=1.0,
        max_shadow_latency_ms=0.0,
    )
    release_service = ModelReleaseService(
        database.get_session,
        release_pipeline,
        runtime_prepare=lambda plan: _profile_bound_runtime_prepare(store, plan),
    )
    shadow_operation, _ = release_service.request_transition(
        model_version_id=model_version_id,
        target_stage=ModelStage.SHADOW.value,
        requested_by="gpu-validation-bootstrap",
        reason="isolated validation fixture; not production approval",
        idempotency_key=f"validation:{model_version_id}:shadow",
    )
    if shadow_operation.status != "applied":
        raise SystemExit(
            f"validation shadow release failed: {shadow_operation.error or 'unknown'}"
        )
    canary_operation, canary_record = release_service.request_transition(
        model_version_id=model_version_id,
        target_stage=ModelStage.CANARY.value,
        requested_by="gpu-validation-bootstrap",
        reason="isolated CUDA/SceneGuard validation fixture",
        canary_camera_id=args.camera_id,
        idempotency_key=f"validation:{model_version_id}:canary",
    )
    if canary_operation.status != "applied" or canary_record.stage != ModelStage.CANARY.value:
        raise SystemExit(
            f"validation canary release failed: {canary_operation.error or 'unknown'}"
        )

    api_token_path.write_text(api_token + "\n", encoding="utf-8", newline="\n")
    api_token_path.chmod(0o600)

    _write_env(
        env_path,
        source_root=source_root,
        state_dir=state_dir,
        config_path=config_path,
        python_path=python_path,
        go2rtc_bin=copied_go2rtc_bin,
        source_manifest=source_manifest,
        camera_id=args.camera_id,
        argus_port=args.argus_port,
        go2rtc_api_port=args.go2rtc_api_port,
        go2rtc_rtsp_port=args.go2rtc_rtsp_port,
        go2rtc_webrtc_port=args.go2rtc_webrtc_port,
        api_token=api_token,
        api_token_file=api_token_path,
        model_version_id=model_version_id,
        model_stage=ModelStage.CANARY.value,
        runtime_artifact_sha256_value=artifact_sha256,
        scene_profile_id=profile.profile_id,
        release_revision=int(canary_record.release_revision),
        admin_username=admin_username,
        admin_password=admin_password,
        viewer_username=viewer_username,
        viewer_password=viewer_password,
    )

    artifact_manifest = state_dir / "artifacts.sha256"
    artifact_files = [
        source_manifest,
        config_path,
        env_path,
        api_token_path,
        copied_go2rtc_bin,
        baseline_copy,
        baseline_dir.parent / "current.txt",
    ]
    artifact_files.extend(
        item for item in sorted(copied_model_dir.rglob("*")) if item.is_file()
    )
    scene_profiles_dir = state_dir / "scene_profiles"
    artifact_files.extend(
        item for item in sorted(scene_profiles_dir.rglob("*")) if item.is_file()
    )
    outside_state = [
        item for item in artifact_files if not _is_within(item.resolve(), state_dir)
    ]
    if outside_state:
        raise SystemExit(
            "artifact manifest input escaped validation state: "
            + ", ".join(str(item) for item in outside_state)
        )
    artifact_manifest.write_text(
        "".join(
            f"{_sha256(item)}  {item.relative_to(state_dir).as_posix()}\n"
            for item in sorted(set(artifact_files))
        ),
        encoding="utf-8",
        newline="\n",
    )
    artifact_manifest.chmod(0o600)
    print(json.dumps({
        "authority": "isolated-validation-only",
        "production_stage": False,
        "state_dir": str(state_dir),
        "config": str(config_path),
        "env": str(env_path),
        "api_token_file": str(api_token_path),
        "artifact_manifest": str(artifact_manifest),
        "camera_id": args.camera_id,
        "model_version_id": model_version_id,
        "model_stage": ModelStage.CANARY.value,
        "release_revision": int(canary_record.release_revision),
        "model_artifact_sha256": artifact_sha256,
        "scene_profile_id": profile.profile_id,
        "operator_approved": False,
        "validation_only": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
