from __future__ import annotations

from pathlib import Path
import shlex

import cv2
import numpy as np
import pytest

from argus.config.schema import ArgusConfig, CameraConfig
from scripts.prepare_gpu_validation import (
    DEFAULT_EXEC_SOURCE,
    _assert_isolated_config,
    _build_config,
    _executable_path,
    _validate_scene_reference,
    _write_env,
)


def test_validation_config_is_isolated_authenticated_and_never_production(tmp_path: Path):
    source_root = tmp_path / "source"
    state_dir = tmp_path / "state"
    source_root.mkdir()
    raw = ArgusConfig(
        cameras=[CameraConfig(
            camera_id="validation_camera_01",
            name="OBSBOT",
            source="rtsp://127.0.0.1:8554/validation_camera_01",
            protocol="rtsp",
        )],
    ).model_dump(mode="json")

    configured = _build_config(
        raw,
        state_dir=state_dir,
        source_root=source_root,
        camera_id="validation_camera_01",
        go2rtc_bin=state_dir / "bin" / "go2rtc",
        argus_port=18081,
        go2rtc_api_port=11984,
        go2rtc_rtsp_port=18554,
        go2rtc_webrtc_port=18555,
        exec_source=DEFAULT_EXEC_SOURCE,
        api_token="validation-secret",
    )

    camera = configured["cameras"][0]
    assert camera["source"] == "/dev/video0"
    assert camera["protocol"] == "usb"
    assert camera["usb"]["go2rtc_source"].startswith("exec:ffmpeg")
    assert "h264_nvenc" in camera["usb"]["go2rtc_source"]
    assert camera["anomaly"]["inference_backend"] == "torch-cuda"
    assert camera["anomaly"]["min_shadow_days"] == 0
    assert camera["anomaly"]["min_shadow_samples"] == 0
    assert camera["anomaly"]["min_canary_days"] >= 7
    assert camera["scene_guard"]["require_profile_model_binding"] is True
    assert camera["scene_guard"]["require_operator_approved_profile"] is False
    assert configured["dashboard"]["go2rtc_replace_stale_process"] is False
    assert configured["dashboard"]["go2rtc_webrtc_listen_host"] == "127.0.0.1"
    assert configured["auth"] == {
        "enabled": True,
        "api_token": "validation-secret",
        "session_timeout_minutes": 120,
    }
    assert state_dir.as_posix() in configured["storage"]["database_url"]
    _assert_isolated_config(ArgusConfig.model_validate(configured), state_dir)

    configured["storage"]["database_url"] = "sqlite:////srv/argus/data/db/argus.db"
    with pytest.raises(ValueError, match="database URL escaped"):
        _assert_isolated_config(ArgusConfig.model_validate(configured), state_dir)


def test_venv_python_path_keeps_the_terminal_symlink(tmp_path: Path):
    target = tmp_path / "python-real"
    target.write_bytes(b"python")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    link = venv_bin / "python"
    link.symlink_to(target)

    preserved = _executable_path(link, label="Python interpreter")

    assert preserved == link
    assert preserved.is_symlink()


def test_validation_env_binds_exact_release_identity_and_token_file(tmp_path: Path):
    env_path = tmp_path / "validation.env"
    token_path = tmp_path / "api-token"

    _write_env(
        env_path,
        source_root=tmp_path / "source",
        state_dir=tmp_path,
        config_path=tmp_path / "argus.yaml",
        python_path=tmp_path / "venv" / "bin" / "python",
        go2rtc_bin=tmp_path / "bin" / "go2rtc",
        source_manifest=tmp_path / "source.sha256",
        camera_id="cam_01",
        argus_port=18081,
        go2rtc_api_port=11984,
        go2rtc_rtsp_port=18554,
        go2rtc_webrtc_port=18555,
        api_token="secret",
        api_token_file=token_path,
        model_version_id="model-v1",
        model_stage="canary",
        runtime_artifact_sha256_value="a" * 64,
        scene_profile_id="profile-v1",
        release_revision=2,
        admin_username="admin",
        admin_password="admin-secret",
        viewer_username="viewer",
        viewer_password="viewer-secret",
    )

    values = dict(
        shlex.split(line)[0].split("=", 1)
        for line in env_path.read_text(encoding="utf-8").splitlines()
    )
    assert values["ARGUS_API_TOKEN_FILE"] == str(token_path)
    assert values["MODEL_VERSION_ID"] == "model-v1"
    assert values["MODEL_STAGE"] == "canary"
    assert values["RUNTIME_ARTIFACT_SHA256"] == "a" * 64
    assert values["SCENE_PROFILE_ID"] == "profile-v1"
    assert values["RELEASE_REVISION"] == "2"


def test_validation_rejects_featureless_scene_reference(tmp_path: Path) -> None:
    reference = tmp_path / "green-first-frame.jpg"
    assert cv2.imwrite(
        str(reference),
        np.full((1080, 1920, 3), (0, 128, 0), dtype=np.uint8),
    )
    scene_config = CameraConfig(
        camera_id="cam_01",
        name="Camera",
        source="0",
        protocol="usb",
    ).scene_guard

    with pytest.raises(ValueError, match="cannot authorize SceneGuard"):
        _validate_scene_reference(reference, scene_config)


def test_validation_accepts_reference_that_self_confirms_stable(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "textured-reference.jpg"
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y in range(0, image.shape[0], 40):
        for x in range(0, image.shape[1], 40):
            colour = (
                (245, 245, 245)
                if (x // 40 + y // 40) % 2
                else (20, 20, 20)
            )
            cv2.rectangle(image, (x, y), (x + 28, y + 28), colour, -1)
            cv2.circle(image, (x + 20, y + 20), 7, (30, 140, 240), -1)
    assert cv2.imwrite(str(reference), image)
    scene_config = CameraConfig(
        camera_id="cam_01",
        name="Camera",
        source="0",
        protocol="usb",
    ).scene_guard

    _validate_scene_reference(reference, scene_config)
