from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

import argus.storage.release_evidence as release_evidence
import scripts.run_trusted_soak as trusted_soak
from argus.preprocessing.scene_profile import runtime_artifact_sha256
from argus.storage.release_evidence import (
    SOAK_ATTESTATION_ALGORITHM,
    SOAK_REPORT_SCHEMA,
    SOAK_TRUST_STORE_SCHEMA,
    _validate_soak_report_source,
    canonical_sha256,
    evidence_artifact_sha256,
)
from scripts.run_trusted_soak import (
    audit_run,
    collect_once,
    finalize_run,
    prepare_subject,
    start_run,
    status_run,
)


@dataclass
class FakeClock:
    wall_ns: int = 1_800_000_000_000_000_000
    mono_ns: int = 100_000_000_000

    def time_ns(self) -> int:
        return self.wall_ns

    def monotonic_ns(self) -> int:
        return self.mono_ns

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float, *, monotonic_seconds: float | None = None) -> None:
        self.wall_ns += round(seconds * 1_000_000_000)
        self.mono_ns += round(
            (seconds if monotonic_seconds is None else monotonic_seconds)
            * 1_000_000_000
        )


class Completed:
    returncode = 0
    stdout = "GPU-uuid-1, NVIDIA GeForce RTX 3060, 555.42\n"
    stderr = ""


class FakeHTTPResponse:
    status = 200

    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _subject() -> dict[str, str]:
    return {
        "model_version_id": "model-v3",
        "camera_id": "obsbot",
        "registry_model_hash": "1" * 64,
        "runtime_artifact_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "scene_profile_sha256": "4" * 64,
        "baseline_sha256": "5" * 64,
        "corpus_sha256": "6" * 64,
        "regression_report_sha256": "7" * 64,
    }


def _fetcher(*, profile_id: str = "profile-1"):
    subject = _subject()
    frame_index = 0

    def fetch(url: str, *, token: str | None, timeout: float):
        nonlocal frame_index
        assert token is None
        assert timeout == 10.0
        if url.endswith("/api/system/health"):
            return {
                "status": "healthy",
                "cameras": [{"camera_id": "obsbot", "connected": True}],
            }
        if url.endswith("/api/cameras/obsbot/detail/json"):
            frame_index += 100
            return {
                "camera_id": "obsbot",
                "connected": True,
                "running": True,
                "last_frame_index": frame_index,
                "source_consistency": "go2rtc_redirect",
                "runtime": {
                    "scene_guard": {
                        "enabled": True,
                        "usable": True,
                        "state": "matching",
                        "profile_id": profile_id,
                        "reference_sha256": "8" * 64,
                        "last_checked_frame": frame_index,
                        "bound_model_version": subject["model_version_id"],
                        "bound_model_artifact_sha256": subject[
                            "runtime_artifact_sha256"
                        ],
                        "runtime_model_artifact_sha256": subject[
                            "runtime_artifact_sha256"
                        ],
                    }
                },
            }
        if url.endswith("/api/models/status"):
            return {
                "models": [
                    {
                        "name": "anomaly",
                        "camera_id": "obsbot",
                        "loaded": True,
                        "backend": "torch-cuda",
                        "consecutive_failures": 0,
                        "extra": {"expected_backend": "torch-cuda"},
                    }
                ]
            }
        raise AssertionError(url)

    return fetch


def _command_runner(command: list[str], **kwargs):
    assert command[0] == "nvidia-smi"
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False
    return Completed()


def _trust_store(path: Path) -> Path:
    value = {
        "schema_version": SOAK_TRUST_STORE_SCHEMA,
        "keys": [
            {
                "key_id": "soak-key-1",
                "algorithm": SOAK_ATTESTATION_ALGORITHM,
                "secret_base64": base64.b64encode(b"s" * 32).decode("ascii"),
                "enabled": True,
            }
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    return path


def test_http_probe_unwraps_the_api_envelope(monkeypatch):
    seen = {}

    def fake_urlopen(request, *, timeout):
        seen["url"] = request.full_url
        # urllib normalizes the request header key while preserving its value.
        seen["api_token"] = request.get_header("X-api-token")
        seen["timeout"] = timeout
        return FakeHTTPResponse({
            "code": 0,
            "msg": "ok",
            "data": {"status": "healthy", "cameras": []},
        })

    monkeypatch.setattr(trusted_soak, "urlopen", fake_urlopen)

    data = trusted_soak._fetch_json(
        "http://127.0.0.1:18080/api/system/health",
        token="protected-token",
        timeout=3.0,
    )

    assert data == {"status": "healthy", "cameras": []}
    assert seen == {
        "url": "http://127.0.0.1:18080/api/system/health",
        "api_token": "protected-token",
        "timeout": 3.0,
    }


def test_http_probe_rejects_a_business_error_envelope(monkeypatch):
    monkeypatch.setattr(
        trusted_soak,
        "urlopen",
        lambda request, *, timeout: FakeHTTPResponse({
            "code": 50300,
            "msg": "unavailable",
            "data": None,
        }),
    )

    with pytest.raises(ValueError, match="non-success envelope"):
        trusted_soak._fetch_json(
            "http://127.0.0.1:18080/api/models/status",
            token=None,
            timeout=3.0,
        )


def test_release_validator_rejects_an_hourly_sampling_contract(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_bytes(b"raw\n")
    report = {
        "schema_version": SOAK_REPORT_SCHEMA,
        "producer": {
            "name": "test",
            "version": "1",
            "build_sha256": "a" * 64,
        },
        "generated_at": "2026-07-04T00:00:00Z",
        "subject": _subject(),
        "window": {
            "started_at": "2026-07-01T00:00:00Z",
            "ended_at": "2026-07-04T00:00:00Z",
            "duration_hours": 72.0,
        },
        "telemetry": {
            "raw_telemetry_sha256": hashlib.sha256(b"raw\n").hexdigest(),
            "sample_interval_seconds": 3600.0,
            "expected_sample_count": 72,
            "observed_sample_count": 72,
            "gap_count": 0,
            "max_gap_seconds": 3600.0,
            "continuity_complete": True,
        },
        "result": {"passed": True},
        "attestation": {
            "algorithm": SOAK_ATTESTATION_ALGORITHM,
            "key_id": "not-reached",
            "signature": "not-reached",
        },
    }

    with pytest.raises(ValueError, match="continuity/sample evidence"):
        _validate_soak_report_source(
            report,
            expected_subject=_subject(),
            telemetry_path=telemetry,
            trust_store_path=tmp_path / "not-reached.json",
        )


def test_start_rejects_a_sampling_interval_above_one_minute(tmp_path):
    with pytest.raises(ValueError, match="must not exceed 60 seconds"):
        start_run(
            tmp_path / "soak-state",
            subject=_subject(),
            base_url="http://127.0.0.1:18080",
            sample_interval_seconds=61,
            max_lateness_seconds=5,
            clock=FakeClock(),
        )


def test_prepare_subject_hashes_the_exact_regression_inputs(tmp_path):
    model = tmp_path / "model.pt"
    config = tmp_path / "runtime.yaml"
    profile = tmp_path / "profile"
    baseline = tmp_path / "baseline"
    corpus = tmp_path / "corpus"
    model.write_bytes(b"model")
    config.write_text("camera: obsbot\n", encoding="utf-8")
    for directory, payload in (
        (profile, b"profile"),
        (baseline, b"baseline"),
        (corpus, b"corpus"),
    ):
        directory.mkdir()
        (directory / "evidence.bin").write_bytes(payload)
    evaluated = {
        "camera_id": "obsbot",
        "config_sha256": evidence_artifact_sha256(config),
        "scene_profile_sha256": evidence_artifact_sha256(profile),
        "baseline_sha256": evidence_artifact_sha256(baseline),
        "corpus_sha256": evidence_artifact_sha256(corpus),
    }
    regression = {
        "status": "ok",
        "suite": {
            "evaluated_artifacts": evaluated,
            "runtime_model_identity_keys": [{
                "version_id": "model-v3",
                "artifact_sha256": runtime_artifact_sha256(model),
            }],
        },
    }
    regression_path = tmp_path / "regression.json"
    regression_path.write_text(json.dumps(regression), encoding="utf-8")
    output = tmp_path / "external" / "soak-subject.json"

    subject = prepare_subject(
        model_version_id="model-v3",
        camera_id="obsbot",
        registry_model_hash="registry-hash-16",
        model_path=model,
        config_path=config,
        scene_profile_path=profile,
        baseline_path=baseline,
        corpus_path=corpus,
        regression_report_path=regression_path,
        output=output,
    )

    assert subject["runtime_artifact_sha256"] == runtime_artifact_sha256(model)
    assert subject["regression_report_sha256"] == canonical_sha256(regression)
    assert json.loads(output.read_text(encoding="utf-8")) == subject


def test_collect_and_finalize_72h_with_mocked_time_http_and_subprocess(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(trusted_soak, "MAX_SAMPLE_INTERVAL_SECONDS", 3600.0)
    monkeypatch.setattr(
        release_evidence,
        "MAX_SOAK_SAMPLE_INTERVAL_SECONDS",
        3600.0,
    )
    clock = FakeClock()
    state_dir = tmp_path / "soak-state"
    start_run(
        state_dir,
        subject=_subject(),
        base_url="http://127.0.0.1:18080",
        sample_interval_seconds=3600,
        max_lateness_seconds=5,
        clock=clock,
    )

    fetcher = _fetcher()
    for _ in range(72):
        clock.advance(3600)
        sample = collect_once(
            state_dir,
            clock=clock,
            fetch_json=fetcher,
            command_runner=_command_runner,
        )
        assert sample["assessment"] == {"eligible": True, "reasons": []}

    status = status_run(state_dir)
    assert status["audit_valid"] is True
    assert status["scheduled_duration_hours"] == 72.0
    assert status["remaining_hours"] == 0.0

    trust_store = _trust_store(tmp_path / "soak-trust.json")
    output = tmp_path / "signed-report.json"
    report = finalize_run(
        state_dir,
        trust_store=trust_store,
        key_id="soak-key-1",
        output=output,
        clock=clock,
    )

    assert report["schema_version"] == SOAK_REPORT_SCHEMA
    assert report["window"]["duration_hours"] == 72.0
    assert report["telemetry"]["observed_sample_count"] == 72
    assert report["attestation"]["key_id"] == "soak-key-1"
    with pytest.raises(ValueError, match="already finalized"):
        collect_once(
            state_dir,
            clock=clock,
            fetch_json=fetcher,
            command_runner=_command_runner,
        )
    passed, duration = _validate_soak_report_source(
        report,
        expected_subject=_subject(),
        telemetry_path=state_dir / "telemetry.jsonl",
        trust_store_path=trust_store,
    )
    assert passed is True
    assert duration == 72.0


def test_append_record_writes_exact_canonical_lf_bytes(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.touch()
    record = {"message": "line one\nline two", "sample": 1}
    trusted_soak._append_record(path, record)
    assert path.read_bytes() == trusted_soak.canonical_json(record).encode("utf-8") + b"\n"
    assert b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize("tampering", ["content", "crlf"])
def test_audit_rejects_raw_telemetry_tampering(tmp_path, tampering):
    clock = FakeClock()
    state_dir = tmp_path / "soak-state"
    start_run(
        state_dir,
        subject=_subject(),
        base_url="http://127.0.0.1:18080",
        sample_interval_seconds=60,
        max_lateness_seconds=5,
        clock=clock,
    )
    clock.advance(60)
    collect_once(
        state_dir,
        clock=clock,
        fetch_json=_fetcher(),
        command_runner=_command_runner,
    )
    telemetry = state_dir / "telemetry.jsonl"
    original = telemetry.read_bytes()
    telemetry.write_bytes(
        original.replace(b"healthy", b"degrade") if tampering == "content"
        else original.replace(b"\n", b"\r\n")
    )

    audit = audit_run(state_dir)

    assert audit.valid is False
    expected = "record hash is invalid" if tampering == "content" else "not a complete LF record"
    assert any(expected in reason for reason in audit.reasons)


def test_late_slot_is_durable_and_permanently_ineligible(tmp_path):
    clock = FakeClock()
    state_dir = tmp_path / "soak-state"
    start_run(
        state_dir,
        subject=_subject(),
        base_url="http://127.0.0.1:18080",
        sample_interval_seconds=60,
        max_lateness_seconds=5,
        clock=clock,
    )
    clock.advance(66)
    sample = collect_once(
        state_dir,
        clock=clock,
        fetch_json=_fetcher(),
        command_runner=_command_runner,
    )

    assert sample["assessment"]["eligible"] is False
    assert "scheduled collection slot was missed" in sample["assessment"]["reasons"]
    audit = audit_run(state_dir)
    assert audit.valid is False
    assert any("missed its collection slot" in reason for reason in audit.reasons)


def test_runtime_identity_change_and_monotonic_divergence_are_rejected(tmp_path):
    clock = FakeClock()
    state_dir = tmp_path / "soak-state"
    start_run(
        state_dir,
        subject=_subject(),
        base_url="http://127.0.0.1:18080",
        sample_interval_seconds=60,
        max_lateness_seconds=5,
        clock_skew_seconds=2,
        clock=clock,
    )
    clock.advance(60)
    collect_once(
        state_dir,
        clock=clock,
        fetch_json=_fetcher(profile_id="profile-1"),
        command_runner=_command_runner,
    )
    clock.advance(60, monotonic_seconds=70)
    second = collect_once(
        state_dir,
        clock=clock,
        fetch_json=_fetcher(profile_id="profile-2"),
        command_runner=_command_runner,
    )

    assert second["assessment"]["eligible"] is False
    assert "runtime identity changed during the soak" in second["assessment"]["reasons"]
    audit = audit_run(state_dir)
    assert audit.valid is False
    assert any("runtime identity changed" in reason for reason in audit.reasons)
    assert any("wall/monotonic clocks diverged" in reason for reason in audit.reasons)


def test_finalize_refuses_before_72_hours(tmp_path):
    clock = FakeClock()
    state_dir = tmp_path / "soak-state"
    start_run(
        state_dir,
        subject=_subject(),
        base_url="http://127.0.0.1:18080",
        sample_interval_seconds=60,
        max_lateness_seconds=5,
        clock=clock,
    )
    clock.advance(60)
    collect_once(
        state_dir,
        clock=clock,
        fetch_json=_fetcher(),
        command_runner=_command_runner,
    )

    with pytest.raises(ValueError, match="soak is incomplete"):
        finalize_run(
            state_dir,
            trust_store=_trust_store(tmp_path / "soak-trust.json"),
            key_id="soak-key-1",
            output=tmp_path / "signed-report.json",
            clock=clock,
        )
