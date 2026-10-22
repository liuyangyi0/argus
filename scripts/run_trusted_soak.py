#!/usr/bin/env python3
"""Collect and attest an Argus production soak without trusting mutable summaries.

The run directory and signing key deliberately live outside the source checkout.
``start`` creates an immutable run contract, ``collect`` appends one health sample
(or runs continuously), ``status`` replays the complete hash chain, and
``finalize`` refuses to sign unless at least 72 scheduled hours are complete.

This command is intentionally fail closed.  A missed collection slot, wall-clock
rollback, SceneGuard block, model identity change, non-CUDA backend, unhealthy
Argus response, broken NVIDIA probe, or telemetry mutation permanently makes the
run unsuitable for production evidence.  Start a new run after correcting the
fault; do not edit the JSONL file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

try:
    from argus.preprocessing.scene_profile import runtime_artifact_sha256
    from argus.storage.release_evidence import (
        MIN_SOAK_HOURS,
        SOAK_REPORT_SCHEMA,
        canonical_json,
        canonical_sha256,
        evidence_artifact_sha256,
    )
except ModuleNotFoundError:  # direct execution from a source checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from argus.preprocessing.scene_profile import runtime_artifact_sha256
    from argus.storage.release_evidence import (
        MIN_SOAK_HOURS,
        SOAK_REPORT_SCHEMA,
        canonical_json,
        canonical_sha256,
        evidence_artifact_sha256,
    )

try:
    from scripts.attest_soak_report import _protected_file, attest_report
except ModuleNotFoundError:  # direct execution as ``python scripts/...``
    from attest_soak_report import _protected_file, attest_report


RUN_SCHEMA = "argus.trusted-soak-run.v1"
SAMPLE_SCHEMA = "argus.trusted-soak-sample.v1"
PRODUCER_NAME = "argus-trusted-soak"
PRODUCER_VERSION = "1"
DEFAULT_INTERVAL_SECONDS = 60.0
DEFAULT_MAX_LATENESS_SECONDS = 5.0
DEFAULT_CLOCK_SKEW_SECONDS = 5.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0
MAX_SAMPLE_INTERVAL_SECONDS = 60.0
MAX_ALLOWED_LATENESS_SECONDS = 5.0
MAX_ALLOWED_CLOCK_SKEW_SECONDS = 5.0
MAX_REQUEST_TIMEOUT_SECONDS = 10.0
_SHA256_FIELDS = {
    "runtime_artifact_sha256",
    "config_sha256",
    "scene_profile_sha256",
    "baseline_sha256",
    "corpus_sha256",
    "regression_report_sha256",
}
_SUBJECT_FIELDS = {
    "model_version_id",
    "camera_id",
    "registry_model_hash",
    *_SHA256_FIELDS,
}
_REPO_ROOT = Path(__file__).resolve().parents[1]


class Clock(Protocol):
    def time_ns(self) -> int: ...

    def monotonic_ns(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def time_ns(self) -> int:
        return time.time_ns()

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


SYSTEM_CLOCK = SystemClock()


class FetchJson(Protocol):
    def __call__(
        self,
        url: str,
        *,
        token: str | None,
        timeout: float,
    ) -> Mapping[str, Any]: ...


class CommandRunner(Protocol):
    def __call__(self, command: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class AuditResult:
    records: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]
    reboot_count: int
    max_lateness_seconds: float

    @property
    def valid(self) -> bool:
        return not self.reasons


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _producer_build_sha256() -> str:
    """Bind the collector, signer, verifier, and runtime hash implementation."""

    sources = (
        Path(__file__).resolve(),
        _REPO_ROOT / "scripts" / "attest_soak_report.py",
        _REPO_ROOT / "src" / "argus" / "storage" / "release_evidence.py",
        _REPO_ROOT / "src" / "argus" / "preprocessing" / "scene_profile.py",
    )
    digest = hashlib.sha256()
    for path in sources:
        if not path.is_file():
            raise ValueError(f"trusted soak producer source is missing: {path.name}")
        label = path.relative_to(_REPO_ROOT).as_posix().encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _iso_utc(wall_time_ns: int) -> str:
    return datetime.fromtimestamp(
        wall_time_ns / 1_000_000_000,
        tz=timezone.utc,
    ).isoformat().replace("+00:00", "Z")


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return False
    return True


def _secure_external_directory(raw_path: Path, *, create: bool) -> Path:
    expanded = raw_path.expanduser().absolute()
    if expanded.is_symlink():
        raise ValueError("soak state path must not be a symlink")
    path = expanded.resolve()
    if _inside_repo(path):
        raise ValueError("soak state directory must be outside the source checkout")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("soak state path must be a real directory")
    if os.name == "posix":
        metadata = path.stat()
        if metadata.st_mode & 0o077:
            raise ValueError("soak state directory must not allow group/other access")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise ValueError("soak state directory must be owned by the current user")
    return path


def _secure_external_file(raw_path: Path, *, label: str) -> Path:
    expanded = raw_path.expanduser().absolute()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    path = expanded.resolve(strict=True)
    if _inside_repo(path):
        raise ValueError(f"{label} must be outside the source checkout")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a real file")
    _protected_file(path)
    return path


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            os.chmod(path, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


@contextmanager
def _exclusive_lock(state_dir: Path, *, name: str = ".lock"):
    """Hold the run lock for one mutation or an entire collection loop."""

    lock_path = state_dir / name
    handle = lock_path.open("a+b")
    if os.name == "posix":
        os.chmod(lock_path, 0o600)
    if os.name == "posix":
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("another soak command is already using this run") from exc
    else:  # pragma: no cover - deployment is Linux; retained for developer tooling
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            handle.close()
            raise RuntimeError("another soak command is already using this run") from exc
    try:
        yield
    finally:
        if os.name == "posix":
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:  # pragma: no cover
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _validate_subject(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _SUBJECT_FIELDS:
        raise ValueError(
            "soak subject must contain exactly model_version_id, camera_id and "
            "the registry/runtime/evaluated-artifact bindings"
        )
    subject: dict[str, str] = {}
    for field in ("model_version_id", "camera_id", "registry_model_hash"):
        item = value.get(field)
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(f"soak subject {field} must be a non-empty string")
        subject[field] = item
    for field in _SHA256_FIELDS:
        item = value.get(field)
        if (
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError(f"soak subject {field} must be a lowercase SHA-256")
        subject[field] = item
    return subject


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
    ):
        raise ValueError(
            "base URL must be a loopback http(s) origin without credentials/query"
        )
    return value.rstrip("/")


def prepare_subject(
    *,
    model_version_id: str,
    camera_id: str,
    registry_model_hash: str,
    model_path: Path,
    config_path: Path,
    scene_profile_path: Path,
    baseline_path: Path,
    corpus_path: Path,
    regression_report_path: Path,
    output: Path,
) -> dict[str, str]:
    """Build the exact subject from the same immutable release artifacts."""

    regression_report = _read_mapping(
        regression_report_path.expanduser().resolve(strict=True),
        label="regression report",
    )
    actual_digests = {
        "config_sha256": evidence_artifact_sha256(config_path),
        "scene_profile_sha256": evidence_artifact_sha256(scene_profile_path),
        "baseline_sha256": evidence_artifact_sha256(baseline_path),
        "corpus_sha256": evidence_artifact_sha256(corpus_path),
    }
    runtime_digest = runtime_artifact_sha256(model_path)
    suite = regression_report.get("suite")
    if not isinstance(suite, Mapping):
        raise ValueError("regression report suite is missing")
    evaluated = suite.get("evaluated_artifacts")
    expected_evaluated = {"camera_id": camera_id, **actual_digests}
    if evaluated != expected_evaluated:
        raise ValueError(
            "regression report evaluated_artifacts do not match the declared sources"
        )
    expected_runtime_identity = [{
        "version_id": model_version_id,
        "artifact_sha256": runtime_digest,
    }]
    if suite.get("runtime_model_identity_keys") != expected_runtime_identity:
        raise ValueError(
            "regression report runtime identity does not match the declared model"
        )
    subject = _validate_subject({
        "model_version_id": model_version_id,
        "camera_id": camera_id,
        "registry_model_hash": registry_model_hash,
        "runtime_artifact_sha256": runtime_digest,
        **actual_digests,
        "regression_report_sha256": canonical_sha256(regression_report),
    })
    output_path = output.expanduser().absolute()
    if output_path.is_symlink():
        raise ValueError("soak subject output must not be a symlink")
    output_path = output_path.resolve()
    if _inside_repo(output_path):
        raise ValueError("soak subject output must be outside the source checkout")
    if output_path.exists():
        raise ValueError("soak subject output already exists; refusing to overwrite it")
    _secure_external_directory(output_path.parent, create=True)
    _write_json_atomic(output_path, subject)
    return subject


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        value = ""
    return value or f"platform-{sys.platform}"


def start_run(
    state_dir: Path,
    *,
    subject: Mapping[str, Any],
    base_url: str,
    sample_interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    max_lateness_seconds: float = DEFAULT_MAX_LATENESS_SECONDS,
    clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    token_file: Path | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> dict[str, Any]:
    directory = _secure_external_directory(state_dir, create=True)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in (
            sample_interval_seconds,
            max_lateness_seconds,
            clock_skew_seconds,
            request_timeout_seconds,
        )
    ):
        raise ValueError("soak timing values must be finite and positive")
    if float(max_lateness_seconds) >= float(sample_interval_seconds):
        raise ValueError("max lateness must be smaller than the sample interval")
    if float(sample_interval_seconds) > MAX_SAMPLE_INTERVAL_SECONDS:
        raise ValueError(
            f"sample interval must not exceed {MAX_SAMPLE_INTERVAL_SECONDS:g} seconds"
        )
    if float(max_lateness_seconds) > MAX_ALLOWED_LATENESS_SECONDS:
        raise ValueError(
            f"max lateness must not exceed {MAX_ALLOWED_LATENESS_SECONDS:g} seconds"
        )
    if float(clock_skew_seconds) > MAX_ALLOWED_CLOCK_SKEW_SECONDS:
        raise ValueError(
            f"clock skew tolerance must not exceed "
            f"{MAX_ALLOWED_CLOCK_SKEW_SECONDS:g} seconds"
        )
    if float(request_timeout_seconds) > MAX_REQUEST_TIMEOUT_SECONDS:
        raise ValueError(
            f"request timeout must not exceed {MAX_REQUEST_TIMEOUT_SECONDS:g} seconds"
        )
    protected_token = None
    if token_file is not None:
        protected_token = str(_secure_external_file(token_file, label="API token file"))

    with _exclusive_lock(directory):
        run_path = directory / "run.json"
        telemetry_path = directory / "telemetry.jsonl"
        if run_path.exists() or telemetry_path.exists():
            raise ValueError("soak state already exists; use a new empty directory")
        wall_ns = clock.time_ns()
        monotonic_ns = clock.monotonic_ns()
        run = {
            "schema_version": RUN_SCHEMA,
            "run_id": str(uuid.uuid4()),
            "created_at": _iso_utc(wall_ns),
            "start_wall_time_ns": wall_ns,
            "start_monotonic_ns": monotonic_ns,
            "start_boot_id": _boot_id(),
            "subject": _validate_subject(subject),
            "base_url": _validate_base_url(base_url),
            "sample_interval_seconds": float(sample_interval_seconds),
            "max_lateness_seconds": float(max_lateness_seconds),
            "clock_skew_seconds": float(clock_skew_seconds),
            "request_timeout_seconds": float(request_timeout_seconds),
            "token_file": protected_token,
            "producer": {
                "name": PRODUCER_NAME,
                "version": PRODUCER_VERSION,
                "build_sha256": _producer_build_sha256(),
            },
        }
        run["contract_sha256"] = _sha256_bytes(canonical_json(run).encode("utf-8"))
        _write_json_atomic(run_path, run)
        descriptor = os.open(
            telemetry_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(descriptor)
        return run


def _load_run(state_dir: Path) -> tuple[Path, dict[str, Any]]:
    directory = _secure_external_directory(state_dir, create=False)
    run = _read_mapping(directory / "run.json", label="soak run")
    required = {
        "schema_version",
        "run_id",
        "created_at",
        "start_wall_time_ns",
        "start_monotonic_ns",
        "start_boot_id",
        "subject",
        "base_url",
        "sample_interval_seconds",
        "max_lateness_seconds",
        "clock_skew_seconds",
        "request_timeout_seconds",
        "token_file",
        "producer",
        "contract_sha256",
    }
    if set(run) != required or run.get("schema_version") != RUN_SCHEMA:
        raise ValueError("soak run contract has an unsupported shape")
    try:
        parsed_run_id = uuid.UUID(str(run.get("run_id")))
    except ValueError as exc:
        raise ValueError("soak run ID is invalid") from exc
    if str(parsed_run_id) != run.get("run_id"):
        raise ValueError("soak run ID is not canonical")
    start_wall = run.get("start_wall_time_ns")
    start_monotonic = run.get("start_monotonic_ns")
    if (
        not isinstance(start_wall, int)
        or isinstance(start_wall, bool)
        or not isinstance(start_monotonic, int)
        or isinstance(start_monotonic, bool)
        or start_wall <= 0
        or start_monotonic < 0
        or run.get("created_at") != _iso_utc(start_wall)
        or not isinstance(run.get("start_boot_id"), str)
        or not run.get("start_boot_id")
    ):
        raise ValueError("soak run initial clock evidence is invalid")
    _validate_subject(run.get("subject"))
    _validate_base_url(str(run.get("base_url", "")))
    interval = run.get("sample_interval_seconds")
    lateness = run.get("max_lateness_seconds")
    clock_skew = run.get("clock_skew_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or not 0 < float(interval) <= MAX_SAMPLE_INTERVAL_SECONDS
        or isinstance(lateness, bool)
        or not isinstance(lateness, (int, float))
        or not 0 < float(lateness) <= MAX_ALLOWED_LATENESS_SECONDS
        or float(lateness) >= float(interval)
        or isinstance(clock_skew, bool)
        or not isinstance(clock_skew, (int, float))
        or not 0 < float(clock_skew) <= MAX_ALLOWED_CLOCK_SKEW_SECONDS
    ):
        raise ValueError("soak run timing contract is outside production limits")
    unsigned = dict(run)
    contract_digest = unsigned.pop("contract_sha256", None)
    if contract_digest != _sha256_bytes(canonical_json(unsigned).encode("utf-8")):
        raise ValueError("soak run contract digest is invalid")
    current_build = _producer_build_sha256()
    producer = run.get("producer")
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {"name", "version", "build_sha256"}
        or producer.get("name") != PRODUCER_NAME
        or producer.get("version") != PRODUCER_VERSION
        or producer.get("build_sha256") != current_build
    ):
        raise ValueError("trusted soak producer changed after the run was started")
    timeout = run.get("request_timeout_seconds")
    token_file = run.get("token_file")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= MAX_REQUEST_TIMEOUT_SECONDS
        or (token_file is not None and not isinstance(token_file, str))
    ):
        raise ValueError("soak run request/authentication contract is invalid")
    return directory, run


def _read_token(run: Mapping[str, Any]) -> str | None:
    raw_path = run.get("token_file")
    if raw_path is None:
        return None
    if not isinstance(raw_path, str):
        raise ValueError("soak token file binding is invalid")
    path = _secure_external_file(Path(raw_path), label="API token file")
    token = path.read_text(encoding="utf-8").strip()
    if not token or "\n" in token or "\r" in token:
        raise ValueError("API token file must contain exactly one non-empty token")
    return token


def _fetch_json(
    url: str,
    *,
    token: str | None,
    timeout: float,
) -> Mapping[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["X-API-Token"] = token
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured origin
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"request failed: {type(exc).__name__}") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("code") != 0
        or set(payload) != {"code", "msg", "data"}
        or not isinstance(payload.get("data"), Mapping)
    ):
        raise ValueError("Argus API returned a non-success envelope")
    return payload["data"]


def _probe_cuda(
    *,
    command_runner: CommandRunner | None = None,
    timeout: float,
) -> dict[str, Any]:
    runner = command_runner or subprocess.run
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "gpus": [], "error": type(exc).__name__}
    if completed.returncode != 0:
        return {
            "available": False,
            "gpus": [],
            "error": f"exit-{completed.returncode}",
        }
    gpus = []
    for line in str(completed.stdout).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not all(fields):
            return {"available": False, "gpus": [], "error": "invalid-output"}
        gpus.append({"uuid": fields[0], "name": fields[1], "driver": fields[2]})
    return {"available": bool(gpus), "gpus": gpus, "error": None if gpus else "no-gpu"}


def _bounded_reason(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text[:300]


def _unwrap_scene_guard(camera: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = camera.get("scene_guard")
    if isinstance(direct, Mapping):
        return direct
    runtime = camera.get("runtime")
    nested = runtime.get("scene_guard") if isinstance(runtime, Mapping) else None
    return nested if isinstance(nested, Mapping) else {}


def _collect_observations(
    run: Mapping[str, Any],
    *,
    fetch_json: FetchJson | None,
    command_runner: CommandRunner | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    fetch = fetch_json or _fetch_json
    base_url = str(run["base_url"])
    subject = _validate_subject(run["subject"])
    camera_id = subject["camera_id"]
    token = _read_token(run)
    timeout = float(run["request_timeout_seconds"])
    reasons: list[str] = []

    def request(path: str, label: str) -> Mapping[str, Any]:
        try:
            return fetch(f"{base_url}{path}", token=token, timeout=timeout)
        except Exception as exc:  # external probe failures become durable evidence
            reasons.append(f"{label}: {_bounded_reason(exc)}")
            return {}

    health_data = request("/api/system/health", "health request failed")
    camera_data = request(
        f"/api/cameras/{quote(camera_id, safe='')}/detail/json",
        "camera request failed",
    )
    models_data = request("/api/models/status", "model request failed")
    scene = _unwrap_scene_guard(camera_data)

    health = {
        "status": health_data.get("status"),
        "camera_connected": next(
            (
                item.get("connected")
                for item in health_data.get("cameras", [])
                if isinstance(item, Mapping) and item.get("camera_id") == camera_id
            ),
            None,
        ),
    }
    if health["status"] != "healthy":
        reasons.append("Argus health is not healthy")
    if health["camera_connected"] is not True:
        reasons.append("health monitor does not report the camera connected")

    camera = {
        "camera_id": camera_data.get("camera_id"),
        "connected": camera_data.get("connected"),
        "running": camera_data.get("running"),
        "last_frame_index": camera_data.get("last_frame_index"),
        "source_consistency": camera_data.get("source_consistency"),
        "scene_guard": {
            "enabled": scene.get("enabled"),
            "usable": scene.get("usable"),
            "state": scene.get("state"),
            "profile_id": scene.get("profile_id") or scene.get("profile"),
            "reference_sha256": scene.get("reference_sha256"),
            "last_checked_frame": scene.get("last_checked_frame"),
            "bound_model_version": scene.get("bound_model_version"),
            "bound_model_artifact_sha256": scene.get(
                "bound_model_artifact_sha256"
            ),
            "runtime_model_artifact_sha256": scene.get(
                "runtime_model_artifact_sha256"
            ),
        },
    }
    if camera["camera_id"] != camera_id:
        reasons.append("camera detail identity does not match the soak subject")
    if camera["connected"] is not True or camera["running"] is not True:
        reasons.append("camera is not connected and running")
    if (
        not isinstance(camera["last_frame_index"], int)
        or isinstance(camera["last_frame_index"], bool)
        or camera["last_frame_index"] < 0
    ):
        reasons.append("camera has no current frame identity")
    if camera["source_consistency"] not in {"direct", "go2rtc_redirect"}:
        reasons.append("camera declared/runtime source is inconsistent")
    scene_summary = camera["scene_guard"]
    if scene_summary["enabled"] is not True or scene_summary["usable"] is not True:
        reasons.append("SceneGuard is disabled or blocking inference")
    if not scene_summary["profile_id"] or not scene_summary["reference_sha256"]:
        reasons.append("SceneGuard profile/reference identity is incomplete")
    if (
        not isinstance(scene_summary["last_checked_frame"], int)
        or isinstance(scene_summary["last_checked_frame"], bool)
        or scene_summary["last_checked_frame"] < 0
    ):
        reasons.append("SceneGuard has no current verification frame")
    if scene_summary["bound_model_version"] != subject["model_version_id"]:
        reasons.append("SceneGuard model version does not match the soak subject")
    for field in (
        "bound_model_artifact_sha256",
        "runtime_model_artifact_sha256",
    ):
        if scene_summary[field] != subject["runtime_artifact_sha256"]:
            reasons.append(f"SceneGuard {field} does not match the soak subject")

    raw_models = models_data.get("models")
    models = raw_models if isinstance(raw_models, list) else []
    anomaly = next(
        (
            item
            for item in models
            if isinstance(item, Mapping)
            and item.get("camera_id") == camera_id
            and item.get("name") == "anomaly"
        ),
        {},
    )
    extra = anomaly.get("extra") if isinstance(anomaly.get("extra"), Mapping) else {}
    model = {
        "name": anomaly.get("name"),
        "loaded": anomaly.get("loaded"),
        "backend": anomaly.get("backend"),
        "expected_backend": extra.get("expected_backend"),
        "consecutive_failures": anomaly.get("consecutive_failures"),
    }
    if model["loaded"] is not True:
        reasons.append("anomaly model is not loaded")
    if model["backend"] != "torch-cuda" or model["expected_backend"] != "torch-cuda":
        reasons.append("anomaly model is not strictly configured and running on torch-cuda")
    if model["consecutive_failures"] != 0:
        reasons.append("anomaly model has consecutive inference failures")

    cuda = _probe_cuda(command_runner=command_runner, timeout=timeout)
    if cuda["available"] is not True:
        reasons.append("nvidia-smi did not report an available CUDA GPU")

    identity = {
        "camera_id": camera_id,
        "model_version_id": scene_summary["bound_model_version"],
        "runtime_artifact_sha256": scene_summary[
            "runtime_model_artifact_sha256"
        ],
        "scene_profile_id": scene_summary["profile_id"],
        "scene_reference_sha256": scene_summary["reference_sha256"],
        "anomaly_backend": model["backend"],
        "gpus": sorted(
            (
                {
                    "uuid": item.get("uuid"),
                    "name": item.get("name"),
                    "driver": item.get("driver"),
                }
                for item in cuda["gpus"]
                if isinstance(item, Mapping)
            ),
            key=lambda item: (str(item["uuid"]), str(item["name"])),
        ),
    }
    observations = {
        "health": health,
        "camera": camera,
        "model": model,
        "cuda": cuda,
    }
    return observations, identity, reasons


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    encoded = canonical_json(record).encode("utf-8") + b"\n"
    # CRT text descriptors translate LF to CRLF on Windows, invalidating our
    # canonical JSONL bytes and hash chain. Never relax the LF-only auditor.
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _observation_safety_reasons(
    run: Mapping[str, Any],
    record: Mapping[str, Any],
) -> list[str]:
    """Re-evaluate the safety facts instead of trusting ``assessment``."""

    subject = _validate_subject(run["subject"])
    observations = record.get("observations")
    identity = record.get("runtime_identity")
    if not isinstance(observations, Mapping) or not isinstance(identity, Mapping):
        return ["observations/runtime identity are malformed"]
    health = observations.get("health")
    camera = observations.get("camera")
    model = observations.get("model")
    cuda = observations.get("cuda")
    if not all(
        isinstance(item, Mapping) for item in (health, camera, model, cuda)
    ):
        return ["one or more safety observations are malformed"]
    assert isinstance(health, Mapping)
    assert isinstance(camera, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(cuda, Mapping)
    scene = camera.get("scene_guard")
    if not isinstance(scene, Mapping):
        return ["SceneGuard observation is malformed"]
    reasons: list[str] = []
    if health.get("status") != "healthy" or health.get("camera_connected") is not True:
        reasons.append("health observation is not eligible")
    if (
        camera.get("camera_id") != subject["camera_id"]
        or camera.get("connected") is not True
        or camera.get("running") is not True
        or not isinstance(camera.get("last_frame_index"), int)
        or isinstance(camera.get("last_frame_index"), bool)
        or camera.get("last_frame_index") < 0
        or camera.get("source_consistency")
        not in {"direct", "go2rtc_redirect"}
    ):
        reasons.append("camera observation is not eligible")
    if (
        scene.get("enabled") is not True
        or scene.get("usable") is not True
        or not scene.get("profile_id")
        or not scene.get("reference_sha256")
        or not isinstance(scene.get("last_checked_frame"), int)
        or isinstance(scene.get("last_checked_frame"), bool)
        or scene.get("last_checked_frame") < 0
        or scene.get("bound_model_version") != subject["model_version_id"]
        or scene.get("bound_model_artifact_sha256")
        != subject["runtime_artifact_sha256"]
        or scene.get("runtime_model_artifact_sha256")
        != subject["runtime_artifact_sha256"]
    ):
        reasons.append("SceneGuard observation is not eligible")
    if (
        model.get("name") != "anomaly"
        or model.get("loaded") is not True
        or model.get("backend") != "torch-cuda"
        or model.get("expected_backend") != "torch-cuda"
        or model.get("consecutive_failures") != 0
    ):
        reasons.append("model observation is not eligible")
    gpus = cuda.get("gpus")
    if cuda.get("available") is not True or not isinstance(gpus, list) or not gpus:
        reasons.append("CUDA observation is not eligible")
        gpus = []
    expected_identity = {
        "camera_id": subject["camera_id"],
        "model_version_id": scene.get("bound_model_version"),
        "runtime_artifact_sha256": scene.get("runtime_model_artifact_sha256"),
        "scene_profile_id": scene.get("profile_id"),
        "scene_reference_sha256": scene.get("reference_sha256"),
        "anomaly_backend": model.get("backend"),
        "gpus": sorted(
            (
                {
                    "uuid": item.get("uuid"),
                    "name": item.get("name"),
                    "driver": item.get("driver"),
                }
                for item in gpus
                if isinstance(item, Mapping)
            ),
            key=lambda item: (str(item["uuid"]), str(item["name"])),
        ),
    }
    if dict(identity) != expected_identity:
        reasons.append("runtime identity is not derived from the safety observations")
    return reasons


def audit_run(state_dir: Path) -> AuditResult:
    directory, run = _load_run(state_dir)
    telemetry_path = directory / "telemetry.jsonl"
    reasons: list[str] = []
    records: list[dict[str, Any]] = []
    previous_hash: str | None = None
    previous_timing: Mapping[str, Any] | None = {
        "wall_time_ns": int(run["start_wall_time_ns"]),
        "monotonic_ns": int(run["start_monotonic_ns"]),
        "boot_id": run["start_boot_id"],
    }
    previous_observations: Mapping[str, Any] | None = None
    first_identity: object | None = None
    reboot_count = 0
    max_lateness_ns = 0
    interval_ns = round(float(run["sample_interval_seconds"]) * 1_000_000_000)
    lateness_limit_ns = round(float(run["max_lateness_seconds"]) * 1_000_000_000)
    skew_limit_ns = round(float(run["clock_skew_seconds"]) * 1_000_000_000)
    start_wall_ns = int(run["start_wall_time_ns"])

    try:
        lines = telemetry_path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        return AuditResult((), (f"telemetry is unavailable: {exc}",), 0, 0.0)
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
            reasons.append(f"telemetry line {line_number} is not a complete LF record")
            continue
        encoded = raw_line[:-1]
        try:
            parsed = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            reasons.append(f"telemetry line {line_number} is not valid JSON")
            continue
        if not isinstance(parsed, dict) or canonical_json(parsed).encode("utf-8") != encoded:
            reasons.append(f"telemetry line {line_number} is not canonical JSON")
            continue
        record = parsed
        records.append(record)
        expected_record_fields = {
            "schema_version",
            "run_id",
            "run_contract_sha256",
            "sequence",
            "previous_record_sha256",
            "timing",
            "runtime_identity",
            "observations",
            "assessment",
            "record_sha256",
        }
        if set(record) != expected_record_fields:
            reasons.append(f"telemetry line {line_number} record shape is invalid")
        stored_hash = record.get("record_sha256")
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        actual_hash = _sha256_bytes(canonical_json(unsigned).encode("utf-8"))
        if stored_hash != actual_hash:
            reasons.append(f"telemetry line {line_number} record hash is invalid")
        if record.get("schema_version") != SAMPLE_SCHEMA:
            reasons.append(f"telemetry line {line_number} schema is invalid")
        if record.get("run_id") != run["run_id"]:
            reasons.append(f"telemetry line {line_number} belongs to another run")
        if record.get("run_contract_sha256") != run["contract_sha256"]:
            reasons.append(f"telemetry line {line_number} run contract binding is invalid")
        if record.get("sequence") != line_number:
            reasons.append(f"telemetry line {line_number} sequence is invalid")
        if record.get("previous_record_sha256") != previous_hash:
            reasons.append(f"telemetry line {line_number} hash-chain link is invalid")
        previous_hash = stored_hash if isinstance(stored_hash, str) else None

        timing = record.get("timing")
        if not isinstance(timing, Mapping):
            reasons.append(f"telemetry line {line_number} timing is missing")
            continue
        due_ns = start_wall_ns + line_number * interval_ns
        wall_ns = timing.get("wall_time_ns")
        monotonic_ns = timing.get("monotonic_ns")
        lateness_ns = timing.get("lateness_ns")
        boot_id = timing.get("boot_id")
        expected_timing_fields = {
            "observed_at",
            "wall_time_ns",
            "monotonic_ns",
            "boot_id",
            "boot_wall_estimate_ns",
            "due_wall_time_ns",
            "lateness_ns",
        }
        if set(timing) != expected_timing_fields:
            reasons.append(f"telemetry line {line_number} timing shape is invalid")
        if not isinstance(boot_id, str) or not boot_id:
            reasons.append(f"telemetry line {line_number} boot identity is invalid")
        if (
            not isinstance(wall_ns, int)
            or isinstance(wall_ns, bool)
            or not isinstance(monotonic_ns, int)
            or isinstance(monotonic_ns, bool)
            or not isinstance(lateness_ns, int)
            or isinstance(lateness_ns, bool)
            or timing.get("due_wall_time_ns") != due_ns
            or lateness_ns != wall_ns - due_ns
            or wall_ns < due_ns
        ):
            reasons.append(f"telemetry line {line_number} scheduled timing is invalid")
        else:
            max_lateness_ns = max(max_lateness_ns, lateness_ns)
            if lateness_ns > lateness_limit_ns:
                reasons.append(f"telemetry line {line_number} missed its collection slot")
        boot_wall_ns = timing.get("boot_wall_estimate_ns")
        clock_values_are_ints = (
            isinstance(wall_ns, int)
            and not isinstance(wall_ns, bool)
            and isinstance(monotonic_ns, int)
            and not isinstance(monotonic_ns, bool)
        )
        if (
            not isinstance(boot_wall_ns, int)
            or isinstance(boot_wall_ns, bool)
            or not clock_values_are_ints
            or boot_wall_ns != wall_ns - monotonic_ns
        ):
            reasons.append(f"telemetry line {line_number} boot clock evidence is invalid")
        if clock_values_are_ints and timing.get("observed_at") != _iso_utc(wall_ns):
            reasons.append(f"telemetry line {line_number} observed_at is invalid")

        if previous_timing is not None:
            previous_wall = previous_timing.get("wall_time_ns")
            previous_monotonic = previous_timing.get("monotonic_ns")
            previous_boot = previous_timing.get("boot_id")
            if isinstance(previous_wall, int) and isinstance(wall_ns, int):
                if wall_ns <= previous_wall:
                    reasons.append(f"telemetry line {line_number} wall clock rolled back")
                if boot_id == previous_boot:
                    if (
                        not isinstance(previous_monotonic, int)
                        or not isinstance(monotonic_ns, int)
                        or monotonic_ns <= previous_monotonic
                    ):
                        reasons.append(
                            f"telemetry line {line_number} monotonic clock did not advance"
                        )
                    elif abs(
                        (wall_ns - previous_wall)
                        - (monotonic_ns - previous_monotonic)
                    ) > skew_limit_ns:
                        reasons.append(
                            f"telemetry line {line_number} wall/monotonic clocks diverged"
                        )
                else:
                    reboot_count += 1
                    if isinstance(boot_wall_ns, int) and boot_wall_ns < previous_wall:
                        reasons.append(
                            f"telemetry line {line_number} reboot chronology is impossible"
                        )
        assessment = record.get("assessment")
        if (
            not isinstance(assessment, Mapping)
            or set(assessment) != {"eligible", "reasons"}
            or assessment.get("eligible") is not True
            or assessment.get("reasons") != []
        ):
            reasons.append(f"telemetry line {line_number} contains a failed safety sample")
        for observation_reason in _observation_safety_reasons(run, record):
            reasons.append(
                f"telemetry line {line_number} failed recomputation: "
                f"{observation_reason}"
            )
        observations = record.get("observations")
        if (
            previous_observations is not None
            and previous_timing is not None
            and isinstance(observations, Mapping)
            and boot_id == previous_timing.get("boot_id")
        ):
            previous_camera = previous_observations.get("camera")
            current_camera = observations.get("camera")
            if isinstance(previous_camera, Mapping) and isinstance(
                current_camera,
                Mapping,
            ):
                previous_frame = previous_camera.get("last_frame_index")
                current_frame = current_camera.get("last_frame_index")
                if (
                    isinstance(previous_frame, int)
                    and not isinstance(previous_frame, bool)
                    and isinstance(current_frame, int)
                    and not isinstance(current_frame, bool)
                    and current_frame <= previous_frame
                ):
                    reasons.append(
                        f"telemetry line {line_number} camera frame did not advance"
                    )
                previous_scene = previous_camera.get("scene_guard")
                current_scene = current_camera.get("scene_guard")
                if isinstance(previous_scene, Mapping) and isinstance(
                    current_scene,
                    Mapping,
                ):
                    previous_check = previous_scene.get("last_checked_frame")
                    current_check = current_scene.get("last_checked_frame")
                    if (
                        isinstance(previous_check, int)
                        and not isinstance(previous_check, bool)
                        and isinstance(current_check, int)
                        and not isinstance(current_check, bool)
                        and current_check <= previous_check
                    ):
                        reasons.append(
                            f"telemetry line {line_number} SceneGuard check did not advance"
                        )
        identity = record.get("runtime_identity")
        if first_identity is None:
            first_identity = identity
        elif identity != first_identity:
            reasons.append(f"telemetry line {line_number} runtime identity changed")
        if isinstance(observations, Mapping):
            previous_observations = observations
        previous_timing = timing

    return AuditResult(
        tuple(records),
        tuple(dict.fromkeys(reasons)),
        reboot_count,
        max_lateness_ns / 1_000_000_000,
    )


def _collect_once_locked(
    directory: Path,
    run: Mapping[str, Any],
    *,
    clock: Clock,
    fetch_json: FetchJson | None,
    command_runner: CommandRunner | None,
) -> dict[str, Any]:
    if (directory / "finalized.json").exists():
        raise ValueError("soak run is already finalized and cannot accept more samples")
    audit = audit_run(directory)
    if not audit.valid:
        raise ValueError("existing soak telemetry failed audit: " + "; ".join(audit.reasons))
    sequence = len(audit.records) + 1
    interval_ns = round(float(run["sample_interval_seconds"]) * 1_000_000_000)
    due_ns = int(run["start_wall_time_ns"]) + sequence * interval_ns
    wall_ns = clock.time_ns()
    if wall_ns < due_ns:
        raise ValueError(
            f"collection slot is not due for {(due_ns - wall_ns) / 1e9:.3f} seconds"
        )
    monotonic_ns = clock.monotonic_ns()
    observations, identity, sample_reasons = _collect_observations(
        run,
        fetch_json=fetch_json,
        command_runner=command_runner,
    )
    lateness_ns = wall_ns - due_ns
    if lateness_ns > round(float(run["max_lateness_seconds"]) * 1_000_000_000):
        sample_reasons.append("scheduled collection slot was missed")
    if audit.records and identity != audit.records[0].get("runtime_identity"):
        sample_reasons.append("runtime identity changed during the soak")
    timing = {
        "observed_at": _iso_utc(wall_ns),
        "wall_time_ns": wall_ns,
        "monotonic_ns": monotonic_ns,
        "boot_id": _boot_id(),
        "boot_wall_estimate_ns": wall_ns - monotonic_ns,
        "due_wall_time_ns": due_ns,
        "lateness_ns": lateness_ns,
    }
    previous_hash = audit.records[-1]["record_sha256"] if audit.records else None
    record: dict[str, Any] = {
        "schema_version": SAMPLE_SCHEMA,
        "run_id": run["run_id"],
        "run_contract_sha256": run["contract_sha256"],
        "sequence": sequence,
        "previous_record_sha256": previous_hash,
        "timing": timing,
        "runtime_identity": identity,
        "observations": observations,
        "assessment": {
            "eligible": not sample_reasons,
            "reasons": list(dict.fromkeys(sample_reasons)),
        },
    }
    record["record_sha256"] = _sha256_bytes(
        canonical_json(record).encode("utf-8")
    )
    _append_record(directory / "telemetry.jsonl", record)
    return record


def collect_once(
    state_dir: Path,
    *,
    clock: Clock = SYSTEM_CLOCK,
    fetch_json: FetchJson | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    directory, run = _load_run(state_dir)
    with _exclusive_lock(directory):
        return _collect_once_locked(
            directory,
            run,
            clock=clock,
            fetch_json=fetch_json,
            command_runner=command_runner,
        )


def collect_loop(
    state_dir: Path,
    *,
    clock: Clock = SYSTEM_CLOCK,
    fetch_json: FetchJson | None = None,
    command_runner: CommandRunner | None = None,
) -> None:
    directory, run = _load_run(state_dir)
    ready_announced = False
    with _exclusive_lock(directory, name=".collector.lock"):
        while True:
            if (directory / "finalized.json").exists():
                return
            audit = audit_run(directory)
            if not audit.valid:
                raise ValueError(
                    "existing soak telemetry failed audit: " + "; ".join(audit.reasons)
                )
            interval_ns = round(float(run["sample_interval_seconds"]) * 1_000_000_000)
            due_ns = int(run["start_wall_time_ns"]) + (
                len(audit.records) + 1
            ) * interval_ns
            remaining = (due_ns - clock.time_ns()) / 1_000_000_000
            if remaining > 0:
                clock.sleep(remaining)
            if (directory / "finalized.json").exists():
                return
            try:
                record = collect_once(
                    directory,
                    clock=clock,
                    fetch_json=fetch_json,
                    command_runner=command_runner,
                )
            except ValueError:
                if (directory / "finalized.json").exists():
                    return
                raise
            print(
                json.dumps(
                    {
                        "sequence": record["sequence"],
                        "eligible": record["assessment"]["eligible"],
                        "observed_at": record["timing"]["observed_at"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            completed_hours = (
                int(record["sequence"])
                * float(run["sample_interval_seconds"])
                / 3600.0
            )
            if completed_hours >= MIN_SOAK_HOURS and not ready_announced:
                print(
                    json.dumps(
                        {
                            "complete": True,
                            "scheduled_duration_hours": completed_hours,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                ready_announced = True


def status_run(state_dir: Path) -> dict[str, Any]:
    directory, run = _load_run(state_dir)
    audit = audit_run(directory)
    interval = float(run["sample_interval_seconds"])
    scheduled_hours = len(audit.records) * interval / 3600.0
    return {
        "run_id": run["run_id"],
        "started_at": run["created_at"],
        "sample_interval_seconds": interval,
        "observed_sample_count": len(audit.records),
        "scheduled_duration_hours": scheduled_hours,
        "minimum_duration_hours": MIN_SOAK_HOURS,
        "remaining_hours": max(0.0, MIN_SOAK_HOURS - scheduled_hours),
        "audit_valid": audit.valid,
        "reboot_count": audit.reboot_count,
        "max_lateness_seconds": audit.max_lateness_seconds,
        "reasons": list(audit.reasons),
        "finalized": (directory / "finalized.json").is_file(),
        "telemetry_path": str(directory / "telemetry.jsonl"),
    }


def finalize_run(
    state_dir: Path,
    *,
    trust_store: Path,
    key_id: str,
    output: Path,
    clock: Clock = SYSTEM_CLOCK,
) -> dict[str, Any]:
    directory, run = _load_run(state_dir)
    trust_path = _secure_external_file(trust_store, label="soak trust store")
    if not key_id or key_id != key_id.strip():
        raise ValueError("key ID must be a non-empty string")
    with _exclusive_lock(directory):
        finalized_path = directory / "finalized.json"
        if finalized_path.exists():
            raise ValueError("soak run is already finalized")
        audit = audit_run(directory)
        if not audit.valid:
            raise ValueError("soak telemetry is not eligible: " + "; ".join(audit.reasons))
        count = len(audit.records)
        interval = float(run["sample_interval_seconds"])
        duration_hours = count * interval / 3600.0
        if duration_hours < MIN_SOAK_HOURS:
            raise ValueError(
                f"soak is incomplete: {duration_hours:.6f}h < {MIN_SOAK_HOURS:.1f}h"
            )
        ended_wall_ns = int(run["start_wall_time_ns"]) + round(
            count * interval * 1_000_000_000
        )
        generated_wall_ns = clock.time_ns()
        if generated_wall_ns < ended_wall_ns:
            raise ValueError("finalization wall clock precedes the completed soak window")
        last_wall_ns = int(audit.records[-1]["timing"]["wall_time_ns"])
        if generated_wall_ns < last_wall_ns:
            raise ValueError("finalization wall clock rolled back after the last sample")
        freshness_limit_ns = round(
            (interval + float(run["max_lateness_seconds"])) * 1_000_000_000
        )
        if generated_wall_ns - last_wall_ns > freshness_limit_ns:
            raise ValueError(
                "finalization is too far from the last eligible sample; "
                "the post-soak interval is no longer covered"
            )
        telemetry_path = directory / "telemetry.jsonl"
        report = {
            "schema_version": SOAK_REPORT_SCHEMA,
            "producer": dict(run["producer"]),
            "generated_at": _iso_utc(generated_wall_ns),
            "subject": _validate_subject(run["subject"]),
            "window": {
                "started_at": _iso_utc(int(run["start_wall_time_ns"])),
                "ended_at": _iso_utc(ended_wall_ns),
                "duration_hours": duration_hours,
            },
            "telemetry": {
                "raw_telemetry_sha256": _file_sha256(telemetry_path),
                "sample_interval_seconds": interval,
                "expected_sample_count": count,
                "observed_sample_count": count,
                "gap_count": 0,
                # Scheduled-slot coverage is exact.  Per-sample scheduler
                # lateness and both clocks remain visible in the raw JSONL.
                "max_gap_seconds": interval,
                "continuity_complete": True,
            },
            "result": {"passed": True},
        }
        signed = attest_report(
            report,
            _read_mapping(trust_path, label="soak trust store"),
            key_id=key_id,
        )
        output_path = output.expanduser().resolve()
        managed_paths = {
            directory / "run.json",
            directory / "telemetry.jsonl",
            directory / ".lock",
            directory / ".collector.lock",
            finalized_path,
            trust_path,
        }
        if output_path in managed_paths:
            raise ValueError("signed report output collides with protected soak state")
        if output_path.exists():
            raise ValueError("signed report output already exists; refusing to overwrite it")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_path, signed)
        _write_json_atomic(
            finalized_path,
            {
                "run_id": run["run_id"],
                "finalized_at": _iso_utc(generated_wall_ns),
                "report_path": str(output_path),
                "report_sha256": _file_sha256(output_path),
                "telemetry_sha256": report["telemetry"][
                    "raw_telemetry_sha256"
                ],
            },
        )
        return signed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-subject",
        help="hash the immutable release inputs into an external subject file",
    )
    prepare.add_argument("--model-version-id", required=True)
    prepare.add_argument("--camera-id", required=True)
    prepare.add_argument("--registry-model-hash", required=True)
    prepare.add_argument("--model-path", required=True, type=Path)
    prepare.add_argument("--config-path", required=True, type=Path)
    prepare.add_argument("--scene-profile-path", required=True, type=Path)
    prepare.add_argument("--baseline-path", required=True, type=Path)
    prepare.add_argument("--corpus-path", required=True, type=Path)
    prepare.add_argument("--regression-report", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    start = subparsers.add_parser("start", help="create a new external soak run")
    start.add_argument("--state-dir", required=True, type=Path)
    start.add_argument("--subject", required=True, type=Path)
    start.add_argument("--base-url", default="http://127.0.0.1:18080")
    start.add_argument("--sample-interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    start.add_argument("--max-lateness", type=float, default=DEFAULT_MAX_LATENESS_SECONDS)
    start.add_argument("--clock-skew", type=float, default=DEFAULT_CLOCK_SKEW_SECONDS)
    start.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    start.add_argument("--token-file", type=Path)

    collect = subparsers.add_parser("collect", help="append samples to an existing run")
    collect.add_argument("--state-dir", required=True, type=Path)
    collect.add_argument("--loop", action="store_true")

    status = subparsers.add_parser("status", help="replay and audit the telemetry chain")
    status.add_argument("--state-dir", required=True, type=Path)

    finalize = subparsers.add_parser("finalize", help="verify, sign, and write a v2 report")
    finalize.add_argument("--state-dir", required=True, type=Path)
    finalize.add_argument("--trust-store", required=True, type=Path)
    finalize.add_argument("--key-id", required=True)
    finalize.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-subject":
            result = prepare_subject(
                model_version_id=args.model_version_id,
                camera_id=args.camera_id,
                registry_model_hash=args.registry_model_hash,
                model_path=args.model_path,
                config_path=args.config_path,
                scene_profile_path=args.scene_profile_path,
                baseline_path=args.baseline_path,
                corpus_path=args.corpus_path,
                regression_report_path=args.regression_report,
                output=args.output,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output.expanduser().resolve()),
                        "subject_sha256": _sha256_bytes(
                            canonical_json(result).encode("utf-8")
                        ),
                    }
                )
            )
        elif args.command == "start":
            subject = _read_mapping(
                args.subject.expanduser().resolve(strict=True),
                label="soak subject",
            )
            result = start_run(
                args.state_dir,
                subject=subject,
                base_url=args.base_url,
                sample_interval_seconds=args.sample_interval,
                max_lateness_seconds=args.max_lateness,
                clock_skew_seconds=args.clock_skew,
                request_timeout_seconds=args.request_timeout,
                token_file=args.token_file,
            )
            print(json.dumps({"run_id": result["run_id"], "state_dir": str(args.state_dir)}))
        elif args.command == "collect":
            if args.loop:
                collect_loop(args.state_dir)
            else:
                record = collect_once(args.state_dir)
                print(json.dumps(record["assessment"], ensure_ascii=False))
        elif args.command == "status":
            print(json.dumps(status_run(args.state_dir), ensure_ascii=False, indent=2))
        elif args.command == "finalize":
            finalize_run(
                args.state_dir,
                trust_store=args.trust_store,
                key_id=args.key_id,
                output=args.output,
            )
            print(args.output.expanduser().resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"trusted soak refused: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
