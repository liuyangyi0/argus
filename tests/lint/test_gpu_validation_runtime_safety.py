"""Static safety gates for the isolated gpu-validation lifecycle script.

The real CUDA/camera validation still runs only on gpu-validation.  These checks keep
future edits from silently reintroducing name/port-owner kills, weak PID files,
or an unauthenticated health gate.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "gpu_validation_runtime.sh"
)


@pytest.fixture(scope="module")
def runtime_script() -> str:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def test_runtime_script_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash syntax validation runs on the gpu-validation Linux gate")
    result = subprocess.run(
        [bash, "-n"],
        input=SCRIPT.read_text(encoding="utf-8").encode("utf-8"),
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_runtime_lifecycle_is_locked_and_group_owned(runtime_script: str) -> None:
    assert 'getent passwd "$(id -u)"' in runtime_script
    assert 'readonly PRODUCTION_CHECKOUT="$(realpath -- "$RUNTIME_HOME/argus")"' in runtime_script
    assert runtime_script.index("readonly PRODUCTION_CHECKOUT=") < runtime_script.index('source "$ENV_FILE"')
    assert 'ARGUS_STATE_DIR" == "$ENV_ANCHOR_STATE' in runtime_script
    for fixed_suffix in (
        "/validation.env",
        "/argus-validation.yaml",
        "/source.sha256",
        "/artifacts.sha256",
        "/bin/go2rtc",
        "/api-token",
    ):
        assert fixed_suffix in runtime_script
    assert "flock -n 9" in runtime_script
    assert "setsid --fork bash -c" in runtime_script
    for field in ("boot_id", "pid", "starttime", "pgid", "sid", "owner_token"):
        assert f'printf "{field}=' in runtime_script
    assert 'kill -TERM -- "-$pgid"' in runtime_script
    assert 'kill -KILL -- "-$pgid"' in runtime_script

    forbidden = re.compile(r"(?m)^\s*(?:sudo\s+)?(?:pkill|killall|fuser)\b")
    assert forbidden.search(runtime_script) is None


def test_runtime_uses_strict_source_inventory(runtime_script: str) -> None:
    assert 'verify_manifest "$ARGUS_ROOT" "$SOURCE_SHA256_MANIFEST" 1' in runtime_script
    assert "strict manifest file-set mismatch" in runtime_script
    assert "unmanifested=" in runtime_script
    assert "SHA256 mismatch for" in runtime_script


def test_runtime_health_binds_auth_network_and_model_identity(
    runtime_script: str,
) -> None:
    assert '--header "X-API-Token:' not in runtime_script
    assert "--config -" in runtime_script
    assert 'ARGUS__DASHBOARD__GO2RTC_WEBRTC_LISTEN_HOST=127.0.0.1' in runtime_script
    assert 'assert_loopback_listener udp "$GO2RTC_WEBRTC_PORT"' in runtime_script
    assert '"http://127.0.0.1:${ARGUS_PORT}/api/models/json"' in runtime_script
    assert '[[ "$unauth_status" == "401" ]]' in runtime_script
    for variable in (
        "MODEL_VERSION_ID",
        "MODEL_STAGE",
        "RUNTIME_ARTIFACT_SHA256",
        "SCENE_PROFILE_ID",
        "RELEASE_REVISION",
        "ARGUS_API_TOKEN_FILE",
    ):
        assert variable in runtime_script
    assert "expected_config = {" in runtime_script
    assert 'record.get("runtime_state") != "applied"' in runtime_script
    assert 'scene_guard.get("bound_model_version") != model_version' in runtime_script


def test_safety_gate_failure_keeps_preview_runtime_for_diagnosis(
    runtime_script: str,
) -> None:
    timeout_block = runtime_script.split(
        'if (( SECONDS >= deadline )); then\n      echo "$health_output"',
        1,
    )[1].split("\n    fi", 1)[0]
    assert "SAFETY-BLOCKED" in timeout_block
    assert "STARTED_THIS_INVOCATION=0" in timeout_block
    assert "startup_failure" not in timeout_block


def test_startup_waits_for_lazy_go2rtc_camera_owner(runtime_script: str) -> None:
    """The API may be ready before go2rtc launches its lazy exec source."""
    wait_block = runtime_script.split("wait_camera_owner() {", 1)[1].split(
        "\n}",
        1,
    )[0]
    assert "STARTUP_TIMEOUT_SECONDS" in wait_block
    assert "until verify_camera_owner" in wait_block
    assert "sleep 1" in wait_block

    start_block = runtime_script.split("start_runtime() {", 1)[1].split(
        "\n}",
        1,
    )[0]
    assert "wait_camera_owner ||" in start_block
    assert "Camera ownership did not converge" in start_block
