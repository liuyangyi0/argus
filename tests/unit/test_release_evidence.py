"""Pure contract tests for production release evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from argus.storage.release_evidence import (
    SOAK_ATTESTATION_ALGORITHM,
    SOAK_REPORT_SCHEMA,
    SOAK_TRUST_STORE_SCHEMA,
    _validate_soak_report_source,
    build_release_evidence,
    build_release_evidence_from_sources,
    canonical_json,
    canonical_training_artifact_root,
    evidence_artifact_sha256,
    registry_training_data_hash,
    sign_soak_report_hmac,
    validate_baseline_manifest_source,
    validate_release_evidence,
)


def _validate(evidence: object, *, model_path, model_version_id: str, model_hash: str):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    return validate_release_evidence(
        evidence,
        model_version_id=model_version_id,
        camera_id="cam_01",
        registry_model_hash=model_hash,
        runtime_artifact_sha256=runtime_artifact_sha256(model_path),
    )


def test_valid_evidence_is_canonical_and_digest_stable(
    tmp_path,
    release_evidence_factory,
):
    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    evidence = release_evidence_factory(
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
    )

    result = _validate(
        evidence,
        model_path=model,
        model_version_id="model-v1",
        model_hash="registry-hash",
    )

    assert result.valid is True
    assert result.reasons == ()
    assert result.canonical_json == canonical_json(evidence)
    assert result.sha256 is not None and len(result.sha256) == 64
    reordered = dict(reversed(list(evidence.items())))
    assert canonical_json(reordered) == result.canonical_json


def test_evidence_artifact_hash_binds_relative_paths_sizes_and_bytes(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("one", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("two", encoding="utf-8")

    initial = evidence_artifact_sha256(root)
    assert evidence_artifact_sha256(root) == initial

    (nested / "b.txt").write_text("changed", encoding="utf-8")
    assert evidence_artifact_sha256(root) != initial


def test_registry_hash_uses_v2_artifact_root_but_preserves_legacy_scope(
    tmp_path,
    release_baseline_factory,
):
    baseline = release_baseline_factory(root=tmp_path / "v2-baseline")
    assert registry_training_data_hash(baseline / "images") == (
        registry_training_data_hash(baseline)
    )
    initial = registry_training_data_hash(baseline / "images")
    with (baseline / "manifest.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert registry_training_data_hash(baseline / "images") != initial

    legacy_parent = tmp_path / "legacy"
    legacy_images = legacy_parent / "images"
    legacy_images.mkdir(parents=True)
    (legacy_images / "a.png").write_bytes(b"legacy")
    legacy_hash = registry_training_data_hash(legacy_images)
    (legacy_parent / "unrelated.txt").write_text("outside", encoding="utf-8")
    assert registry_training_data_hash(legacy_images) == legacy_hash


def test_soak_v1_passed_duration_shorthand_is_rejected(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root shape is invalid"):
        _validate_soak_report_source(
            {
                "schema_version": "argus.soak-report.v1",
                "subject": {},
                "passed": True,
                "duration_hours": 72.0,
            },
            expected_subject={},
            telemetry_path=telemetry,
        )


def _valid_soak_report(telemetry_sha256: str) -> dict:
    return {
        "schema_version": SOAK_REPORT_SCHEMA,
        "producer": {
            "name": "trusted-producer",
            "version": "1.0",
            "build_sha256": "1" * 64,
        },
        "generated_at": "2026-07-14T00:00:01+00:00",
        "subject": {"bound": "subject"},
        "window": {
            "started_at": "2026-07-11T00:00:00+00:00",
            "ended_at": "2026-07-14T00:00:00+00:00",
            "duration_hours": 72.0,
        },
        "telemetry": {
            "raw_telemetry_sha256": telemetry_sha256,
            "sample_interval_seconds": 60.0,
            "expected_sample_count": 4320,
            "observed_sample_count": 4320,
            "gap_count": 0,
            "max_gap_seconds": 60.0,
            "continuity_complete": True,
        },
        "result": {"passed": True},
        "attestation": {
            "algorithm": SOAK_ATTESTATION_ALGORITHM,
            "key_id": "soak-test-key",
            "signature": "pending",
        },
    }


def _trust_store(tmp_path: Path, secret: bytes, *, key_id="soak-test-key") -> Path:
    path = tmp_path / "soak-trust-store.json"
    path.write_text(json.dumps({
        "schema_version": SOAK_TRUST_STORE_SCHEMA,
        "keys": [{
            "key_id": key_id,
            "algorithm": SOAK_ATTESTATION_ALGORITHM,
            "secret_base64": base64.b64encode(secret).decode("ascii"),
            "enabled": True,
        }],
    }), encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    return path


def _upgrade_baseline_to_release_v3(root: Path) -> dict:
    """Turn the shared offline fixture into complete strict V3 evidence."""

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "argus.scene-profile-baseline.v3"
    decoder_dir = root / "decoder"
    decoder_dir.mkdir()
    records = [
        {
            "frame_index": 0,
            "pts": 0,
            "pts_time_seconds": 0.0,
            "time_base": "1/1000",
            "duration_pts": 100,
            "duration_seconds": 0.1,
        },
        {
            "frame_index": 1,
            "pts": 100,
            "pts_time_seconds": 0.1,
            "time_base": "1/1000",
            "duration_pts": 150,
            "duration_seconds": 0.15,
        },
        {
            "frame_index": 2,
            "pts": 250,
            "pts_time_seconds": 0.25,
            "time_base": "1/1000",
            "duration_pts": None,
            "duration_seconds": None,
        },
    ]
    samples_by_source = {
        int(sample["source_index"]): sample for sample in manifest["samples"]
    }
    for source_index, source in enumerate(manifest["sources"]):
        timestamp_path = decoder_dir / f"source_{source_index:06d}.frames.jsonl"
        timestamp_path.write_text(
            "".join(canonical_json(record) + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )
        source_sha256 = source["sha256"]
        source.update({
            "fps": 8.0,
            "reported_frame_count": 3,
            "decoded_frame_count": 3,
            "decode_backend": "ffprobe+ffmpeg-rawvideo",
            "container_reported_frame_count": 3,
            "opencv_read_terminated": False,
            "source_read_error_detectable": True,
            "vfr_timestamps_preserved": True,
            "frame_time_basis": "pts_times_stream_time_base",
            "accepted_images": 1,
            "first_accepted_frame_index": source_index + 1,
            "last_accepted_frame_index": source_index + 1,
            "accepted_time_span_seconds": 0.0,
            "strict_decoder_evidence": {
                "schema_version": "argus.strict-video-decoder-evidence.v1",
                "backend": "ffprobe+ffmpeg-rawvideo",
                "strict_release_decoder": True,
                "source_sha256_before": source_sha256,
                "source_sha256_after": source_sha256,
                "source_sha256_matches": True,
                "stream_index": 0,
                "width": 16,
                "height": 16,
                "pixel_format": "bgr24",
                "time_base": "1/1000",
                "timestamp_frame_count": 3,
                "pts_strictly_increasing": True,
                "variable_frame_rate_detected": True,
                "vfr_detection_basis": "distinct_adjacent_pts_deltas",
                "distinct_pts_deltas": [100, 150],
                "nominal_frame_rate": 8.0,
                "container_declared_frame_count": 3,
                "container_declared_frame_count_matches": True,
                "ffprobe": {
                    "returncode": 0,
                    "stderr_empty": True,
                    "decoded_frame_inventory_count": 3,
                },
                "ffmpeg": {
                    "returncode": 0,
                    "stderr_empty": True,
                    "decoded_frame_count": 3,
                    "decoded_frame_count_matches_ffprobe": True,
                    "partial_raw_frame_detected": False,
                    "clean_eof": True,
                    "xerror_enabled": True,
                },
                "decode_error_distinguishable_from_clean_eof": True,
                "variable_frame_rate_timestamps_preserved": True,
                "complete_decode_verified": True,
                "source_exhausted": True,
                "truncated": False,
                "sufficient_for_model_release": True,
            },
            "frame_timestamp_evidence": {
                "schema_version": "argus.strict-video-frame-timestamps.v1",
                "path": f"decoder/source_{source_index:06d}.frames.jsonl",
                "sha256": hashlib.sha256(timestamp_path.read_bytes()).hexdigest(),
                "frame_count": 3,
                "time_base": "1/1000",
                "record_format": "one_json_object_per_decoded_frame",
                "coordinates": "zero_based_decoded_presentation_order",
            },
        })
        sample = samples_by_source[source_index]
        record = records[source_index + 1]
        sample.update({
            "source_path": source["path"],
            "frame_index": record["frame_index"],
            "time_seconds": record["pts_time_seconds"],
            "pts": record["pts"],
            "time_base": record["time_base"],
        })
    manifest["decoder_evidence"] = {
        "backend": "ffprobe+ffmpeg-rawvideo",
        "strict_release_decoder_requested": True,
        "all_sources_strictly_audited": True,
        "source_count": len(manifest["sources"]),
        "read_termination_signal": (
            "ffmpeg_clean_zero_exit_after_complete_raw_frame_stream"
        ),
        "decoded_frame_count_cross_check": (
            "ffmpeg_raw_frames_equal_ffprobe_show_frames_inventory"
        ),
        "source_sha256_checked_before_and_after_decode": True,
        "decode_error_distinguishable_from_clean_eof": True,
        "variable_frame_rate_timestamps_preserved": True,
        "time_basis": "per_frame_pts_times_stream_time_base",
        "timestamp_evidence_format": (
            "sha256_bound_jsonl_one_record_per_decoded_frame"
        ),
        "sufficient_for_model_release": True,
    }
    manifest["selection"] = {
        "total_frames_read": 3 * len(manifest["sources"]),
        "accepted_images": len(manifest["samples"]),
        "truncated_by_max_images": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _sign_sources_soak(
    sources: dict[str, str],
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = Path(sources["soak_report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    secret = b"v3-release-test-secret-material!!"
    report["attestation"] = {
        "algorithm": SOAK_ATTESTATION_ALGORITHM,
        "key_id": "soak-test-key",
        "signature": "pending",
    }
    report["attestation"]["signature"] = sign_soak_report_hmac(
        report,
        secret=secret,
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    store = _trust_store(tmp_path, secret)
    monkeypatch.setenv("ARGUS_SOAK_TRUST_STORE", str(store))


def test_public_baseline_verifier_preserves_v2_offline_contract(
    tmp_path,
    release_baseline_factory,
):
    from scripts.run_regression_suite import _validate_baseline_artifact_v2

    baseline = release_baseline_factory(root=tmp_path / "baseline")

    manifest, source_hashes, source_session_ids = (
        validate_baseline_manifest_source(baseline)
    )
    regression_result = _validate_baseline_artifact_v2(
        baseline,
        camera_id="cam_01",
        scene_profile_id="release-profile-v1",
    )

    assert manifest["schema_version"] == "argus.scene-profile-baseline.v2"
    assert source_hashes == {
        source["sha256"] for source in manifest["sources"]
    }
    assert source_session_ids == {"session-a", "session-b"}
    assert regression_result == (manifest, source_hashes, source_session_ids)
    assert manifest["decoder_evidence"]["sufficient_for_model_release"] is False


def test_regression_gate_accepts_complete_strict_v3_baseline(
    tmp_path,
    release_baseline_factory,
):
    from scripts.run_regression_suite import _validate_baseline_artifact_v2

    baseline = release_baseline_factory(root=tmp_path / "baseline")
    expected = _upgrade_baseline_to_release_v3(baseline)

    manifest, source_hashes, source_session_ids = (
        _validate_baseline_artifact_v2(
            baseline,
            camera_id="cam_01",
            scene_profile_id="release-profile-v1",
        )
    )

    assert manifest["schema_version"] == "argus.scene-profile-baseline.v3"
    assert manifest["decoder_evidence"]["sufficient_for_model_release"] is True
    assert source_hashes == {source["sha256"] for source in expected["sources"]}
    assert source_session_ids == {"session-a", "session-b"}


def test_regression_gate_rejects_tampered_v3_decoder_evidence(
    tmp_path,
    release_baseline_factory,
):
    from scripts.run_regression_suite import _validate_baseline_artifact_v2

    baseline = release_baseline_factory(root=tmp_path / "baseline")
    manifest = _upgrade_baseline_to_release_v3(baseline)
    manifest["sources"][0]["strict_decoder_evidence"]["ffmpeg"][
        "returncode"
    ] = 1
    (baseline / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ffmpeg audit"):
        _validate_baseline_artifact_v2(
            baseline,
            camera_id="cam_01",
            scene_profile_id="release-profile-v1",
        )


def test_structural_soak_v2_still_needs_configured_trust_store(
    tmp_path,
    monkeypatch,
):
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text("{}\n", encoding="utf-8")
    telemetry_sha256 = hashlib.sha256(telemetry.read_bytes()).hexdigest()
    report = _valid_soak_report(telemetry_sha256)
    monkeypatch.delenv("ARGUS_SOAK_TRUST_STORE", raising=False)

    with pytest.raises(ValueError, match="ARGUS_SOAK_TRUST_STORE"):
        _validate_soak_report_source(
            report,
            expected_subject={"bound": "subject"},
            telemetry_path=telemetry,
        )


def test_trusted_soak_attestation_verifies_and_tampering_fails(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text("{}\n", encoding="utf-8")
    report = _valid_soak_report(hashlib.sha256(telemetry.read_bytes()).hexdigest())
    secret = b"s" * 32
    report["attestation"]["signature"] = sign_soak_report_hmac(
        report,
        secret=secret,
    )
    store = _trust_store(tmp_path, secret)

    assert _validate_soak_report_source(
        report,
        expected_subject={"bound": "subject"},
        telemetry_path=telemetry,
        trust_store_path=store,
    ) == (True, 72.0)

    report["producer"]["version"] = "tampered"
    with pytest.raises(ValueError, match="signature verification failed"):
        _validate_soak_report_source(
            report,
            expected_subject={"bound": "subject"},
            telemetry_path=telemetry,
            trust_store_path=store,
        )


def test_soak_attestation_rejects_unknown_key(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text("{}\n", encoding="utf-8")
    report = _valid_soak_report(hashlib.sha256(telemetry.read_bytes()).hexdigest())
    secret = b"s" * 32
    report["attestation"]["signature"] = sign_soak_report_hmac(
        report,
        secret=secret,
    )
    store = _trust_store(tmp_path, secret, key_id="different-key")

    with pytest.raises(ValueError, match="unknown or ambiguous"):
        _validate_soak_report_source(
            report,
            expected_subject={"bound": "subject"},
            telemetry_path=telemetry,
            trust_store_path=store,
        )


def test_source_builder_rejects_v2_baseline_decoder_as_release_evidence(
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    baseline = release_baseline_factory(root=tmp_path / "baseline")
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
    )

    with pytest.raises(ValueError, match="decoder evidence is insufficient"):
        build_release_evidence_from_sources(
            sources,
            model_version_id="model-v1",
            camera_id="cam_01",
            registry_model_hash="registry-hash",
            registry_data_hash=registry_training_data_hash(baseline),
            runtime_artifact_digest=runtime_artifact_sha256(model),
            operator_id="authenticated-operator",
            operator_approved=True,
            approved_at="2026-07-14T00:00:00+00:00",
        )


def test_source_builder_accepts_complete_hash_bound_v3_decoder_evidence(
    tmp_path,
    monkeypatch,
    release_baseline_factory,
    release_evidence_sources_factory,
):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    baseline = release_baseline_factory(root=tmp_path / "baseline")
    _upgrade_baseline_to_release_v3(baseline)
    assert canonical_training_artifact_root(baseline / "images") == baseline
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
    )
    _sign_sources_soak(sources, tmp_path=tmp_path, monkeypatch=monkeypatch)

    evidence = build_release_evidence_from_sources(
        sources,
        model_version_id="model-v1",
        camera_id="cam_01",
        registry_model_hash="registry-hash",
        registry_data_hash=registry_training_data_hash(baseline),
        runtime_artifact_digest=runtime_artifact_sha256(model),
        operator_id="authenticated-operator",
        operator_approved=True,
        approved_at="2026-07-14T00:00:00+00:00",
    )

    assert evidence["subject"]["model_version_id"] == "model-v1"
    assert evidence["validation"]["report_status_ok"] is True


@pytest.mark.parametrize(
    ("tamper", "reason"),
    [
        ("source_hash_after", "not release-grade"),
        ("ffmpeg_returncode", "ffmpeg audit"),
        ("timestamp_bytes", "checksum"),
        ("timestamp_semantics", "PTS/time_base"),
        ("timestamp_count", "frame count"),
        ("vfr_flag", "VFR/PTS"),
        ("sample_pts", "PTS timestamp"),
        ("aggregate_bool", "v3 decoder evidence"),
    ],
)
def test_source_builder_rejects_tampered_v3_decoder_lineage(
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
    tamper,
    reason,
):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    baseline = release_baseline_factory(root=tmp_path / "baseline")
    manifest = _upgrade_baseline_to_release_v3(baseline)
    if tamper == "source_hash_after":
        manifest["sources"][0]["strict_decoder_evidence"][
            "source_sha256_after"
        ] = "0" * 64
    elif tamper == "ffmpeg_returncode":
        manifest["sources"][0]["strict_decoder_evidence"]["ffmpeg"][
            "returncode"
        ] = 1
    elif tamper == "timestamp_bytes":
        timestamp_path = baseline / manifest["sources"][0][
            "frame_timestamp_evidence"
        ]["path"]
        timestamp_path.write_text(
            timestamp_path.read_text(encoding="utf-8") + "{}\n",
            encoding="utf-8",
        )
    elif tamper == "timestamp_semantics":
        timestamp_evidence = manifest["sources"][0][
            "frame_timestamp_evidence"
        ]
        timestamp_path = baseline / timestamp_evidence["path"]
        records = [
            json.loads(line)
            for line in timestamp_path.read_text(encoding="utf-8").splitlines()
        ]
        records[1]["pts_time_seconds"] = 0.2
        timestamp_path.write_text(
            "".join(canonical_json(record) + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )
        timestamp_evidence["sha256"] = hashlib.sha256(
            timestamp_path.read_bytes()
        ).hexdigest()
    elif tamper == "timestamp_count":
        manifest["sources"][0]["frame_timestamp_evidence"]["frame_count"] = 2
    elif tamper == "vfr_flag":
        manifest["sources"][0]["strict_decoder_evidence"][
            "variable_frame_rate_detected"
        ] = False
    elif tamper == "sample_pts":
        manifest["samples"][0]["pts"] += 1
    elif tamper == "aggregate_bool":
        manifest["decoder_evidence"]["source_count"] = True
    if tamper != "timestamp_bytes":
        (baseline / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
    )

    with pytest.raises(ValueError, match=reason):
        build_release_evidence_from_sources(
            sources,
            model_version_id="model-v1",
            camera_id="cam_01",
            registry_model_hash="registry-hash",
            registry_data_hash=registry_training_data_hash(baseline),
            runtime_artifact_digest=runtime_artifact_sha256(model),
            operator_id="authenticated-operator",
            operator_approved=True,
            approved_at="2026-07-14T00:00:00+00:00",
        )


def test_source_builder_rejects_symlinked_v3_timestamp_inventory(
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
):
    baseline = release_baseline_factory(root=tmp_path / "baseline")
    manifest = _upgrade_baseline_to_release_v3(baseline)
    timestamp_path = baseline / manifest["sources"][0][
        "frame_timestamp_evidence"
    ]["path"]
    external = tmp_path / "external.frames.jsonl"
    timestamp_path.replace(external)
    timestamp_path.symlink_to(external)
    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")

    with pytest.raises(ValueError, match="symlink"):
        release_evidence_sources_factory(
            root=tmp_path / "sources",
            model_path=model,
            model_version_id="model-v1",
            registry_model_hash="registry-hash",
            baseline_path=baseline,
        )


def test_source_builder_rejects_legacy_v1_baseline_manifest(
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    baseline = release_baseline_factory(root=tmp_path / "baseline")
    manifest_path = baseline / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "argus.scene-profile-baseline.v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
    )

    with pytest.raises(ValueError, match="argus.scene-profile-baseline.v2"):
        build_release_evidence_from_sources(
            sources,
            model_version_id="model-v1",
            camera_id="cam_01",
            registry_model_hash="registry-hash",
            registry_data_hash=registry_training_data_hash(baseline),
            runtime_artifact_digest=runtime_artifact_sha256(model),
            operator_id="operator",
            operator_approved=True,
            approved_at="2026-07-14T00:00:00+00:00",
        )


def test_source_builder_rejects_cross_camera_and_stale_config(
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    baseline = release_baseline_factory(root=tmp_path / "baseline")
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
        camera_id="other-camera",
    )
    kwargs = {
        "model_version_id": "model-v1",
        "camera_id": "cam_01",
        "registry_model_hash": "registry-hash",
        "registry_data_hash": registry_training_data_hash(baseline),
        "runtime_artifact_digest": runtime_artifact_sha256(model),
        "operator_id": "operator",
        "operator_approved": True,
        "approved_at": "2026-07-14T00:00:00+00:00",
    }

    with pytest.raises(ValueError, match="camera_id does not match"):
        build_release_evidence_from_sources(sources, **kwargs)

    matching = release_evidence_sources_factory(
        root=tmp_path / "matching-sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
    )
    config_path = matching["config_path"]
    with open(config_path, "a", encoding="utf-8") as handle:
        handle.write("changed: true\n")

    with pytest.raises(ValueError, match="config_sha256 does not match"):
        build_release_evidence_from_sources(matching, **kwargs)


def test_source_builder_rejects_inactive_profile_even_when_profile_hash_matches(
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    baseline = release_baseline_factory(root=tmp_path / "baseline")
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
    )
    profile_path = Path(sources["scene_profile_path"])
    (profile_path.parent / "active.json").write_text(
        json.dumps({
            "schema_version": 1,
            "camera_id": "cam_01",
            "profile_id": "different-profile",
            "updated_at": "2026-07-14T00:00:00+00:00",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not the camera's active profile"):
        build_release_evidence_from_sources(
            sources,
            model_version_id="model-v1",
            camera_id="cam_01",
            registry_model_hash="registry-hash",
            registry_data_hash=registry_training_data_hash(baseline),
            runtime_artifact_digest=runtime_artifact_sha256(model),
            operator_id="operator",
            operator_approved=True,
            approved_at="2026-07-14T00:00:00+00:00",
        )


def test_source_builder_rejects_training_and_regression_source_overlap(
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    evaluated_hash = hashlib.sha256(b"independent-regression-video").hexdigest()
    baseline = release_baseline_factory(
        root=tmp_path / "baseline",
        training_source_sha256=evaluated_hash,
    )
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
    )

    with pytest.raises(ValueError, match="not an independent holdout"):
        build_release_evidence_from_sources(
            sources,
            model_version_id="model-v1",
            camera_id="cam_01",
            registry_model_hash="registry-hash",
            registry_data_hash=registry_training_data_hash(baseline),
            runtime_artifact_digest=runtime_artifact_sha256(model),
            operator_id="operator",
            operator_approved=True,
            approved_at="2026-07-14T00:00:00+00:00",
        )


def test_source_builder_rejects_baseline_not_bound_to_registry(
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    baseline = release_baseline_factory(root=tmp_path / "baseline")
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources",
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
        baseline_path=baseline,
    )

    with pytest.raises(ValueError, match="registry training data hash"):
        build_release_evidence_from_sources(
            sources,
            model_version_id="model-v1",
            camera_id="cam_01",
            registry_model_hash="registry-hash",
            registry_data_hash="0" * 16,
            runtime_artifact_digest=runtime_artifact_sha256(model),
            operator_id="operator",
            operator_approved=True,
            approved_at="2026-07-14T00:00:00+00:00",
        )


@pytest.mark.parametrize(
    ("path", "bad_value", "reason_fragment"),
    [
        (("operator_approval", "operator_approved"), 1, "literal boolean true"),
        (("validation", "report_status_ok"), False, "literal boolean true"),
        (("validation", "pipeline_evaluation"), False, "literal boolean true"),
        (("validation", "production_parity"), "true", "literal boolean true"),
        (
            ("validation", "trusted_scene_reference"),
            False,
            "literal boolean true",
        ),
        (
            ("validation", "scene_profile_operator_approval"),
            False,
            "literal boolean true",
        ),
        (
            ("validation", "runtime_model_identity"),
            False,
            "literal boolean true",
        ),
        (
            ("validation", "positive_annotation_coverage"),
            False,
            "literal boolean true",
        ),
        (
            ("validation", "candidate_confirmation"),
            False,
            "literal boolean true",
        ),
        (
            ("validation", "corpus", "complete"),
            False,
            "literal boolean true",
        ),
        (
            ("validation", "corpus", "independent_holdout"),
            False,
            "literal boolean true",
        ),
        (
            ("validation", "corpus", "source_inventory_verified"),
            False,
            "literal boolean true",
        ),
        (
            ("validation", "corpus", "normal_hours"),
            1.999,
            "normal corpus is too short",
        ),
        (
            ("validation", "corpus", "static_foreign_event_count"),
            19,
            "static corpus is too small",
        ),
        (("validation", "scene_guard", "coverage"), 0.989, "below 0.99"),
        (
            ("validation", "metrics", "false_positive_events_per_hour"),
            2.001,
            "between 0 and 2",
        ),
        (
            ("validation", "metrics", "event_recall"),
            0.949,
            "overall event recall",
        ),
        (
            ("validation", "metrics", "static_foreign_recall"),
            0.949,
            "between 0.95 and 1",
        ),
        (
            ("validation", "metrics", "positive_annotation_coverage"),
            0.989,
            "positive annotation coverage",
        ),
        (
            ("validation", "metrics", "delay_p95_sec"),
            10.001,
            "between 0 and 10s",
        ),
        (("soak", "passed"), False, "literal boolean true"),
        (("soak", "duration_hours"), 71.999, "soak is too short"),
    ],
)
def test_each_hard_gate_fails_closed(
    tmp_path,
    release_evidence_factory,
    path,
    bad_value,
    reason_fragment,
):
    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    evidence = release_evidence_factory(
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
    )
    target = evidence
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value

    result = _validate(
        evidence,
        model_path=model,
        model_version_id="model-v1",
        model_hash="registry-hash",
    )

    assert result.valid is False
    assert reason_fragment in "; ".join(result.reasons)


def test_approved_true_shorthand_and_binding_mismatch_are_rejected(
    tmp_path,
    release_evidence_factory,
):
    model = tmp_path / "model.pt"
    model.write_bytes(b"runtime")
    shorthand = _validate(
        {"approved": True},
        model_path=model,
        model_version_id="model-v1",
        model_hash="registry-hash",
    )
    assert shorthand.valid is False
    assert "unsupported fields: approved" in "; ".join(shorthand.reasons)

    evidence = release_evidence_factory(
        model_path=model,
        model_version_id="different-version",
        registry_model_hash="registry-hash",
    )
    mismatch = _validate(
        evidence,
        model_path=model,
        model_version_id="model-v1",
        model_hash="registry-hash",
    )
    assert mismatch.valid is False
    assert "model_version_id does not match registry" in "; ".join(
        mismatch.reasons
    )

    digest_mismatch_evidence = release_evidence_factory(
        model_path=model,
        model_version_id="model-v1",
        registry_model_hash="registry-hash",
    )
    digest_mismatch_evidence["artifacts"]["runtime_artifact_sha256"] = "b" * 64
    digest_mismatch = _validate(
        digest_mismatch_evidence,
        model_path=model,
        model_version_id="model-v1",
        model_hash="registry-hash",
    )
    assert digest_mismatch.valid is False
    assert "runtime artifact digest does not match model_path" in "; ".join(
        digest_mismatch.reasons
    )


def test_builder_keeps_incomplete_current_report_no_go():
    report = {
        "status": "ok",
        "evaluation_mode": "pipeline",
        "suite": {
            "acceptance": {
                "corpus_completeness": "pass",
                "independent_holdout": "fail",
                "scene_profile_operator_approval": "fail",
                "production_parity": "pass",
                "trusted_scene_reference": "pass",
                "runtime_model_identity": "pass",
                "positive_annotation_coverage": "pass",
                "candidate_confirmation": "pass",
            },
            "corpus_complete": True,
            "independent_holdout_evidence_complete": False,
            "source_inventory_evidence_complete": False,
            "normal_hours": 0.5,
            "static_foreign_event_count": 10,
            "positive_annotation_coverage_evidence_complete": True,
            "scene_guard_evidence_complete": True,
            "scene_guard_usable_coverage_ratio": 0.995,
            "false_positive_events_per_hour": 5.5879,
            "event_recall": 0.2,
            "static_foreign_event_recall": 0.2,
            "positive_annotation_coverage_ratio": 0.99,
            "delay_p95_sec": 7.7,
            "scene_profile_operator_approval_evidence_complete": False,
            "production_parity_evidence_complete": True,
            "trusted_scene_reference_evidence_complete": True,
            "runtime_model_identity_evidence_complete": True,
            "candidate_contract_evidence_complete": True,
            "runtime_model_identity_keys": [
                {"version_id": "model-v1", "artifact_sha256": "a" * 64}
            ],
        },
    }

    evidence = build_release_evidence(
        deepcopy(report),
        model_version_id="model-v1",
        camera_id="cam_01",
        registry_model_hash="registry-hash",
        config_sha256="1" * 64,
        scene_profile_sha256="2" * 64,
        baseline_sha256="3" * 64,
        corpus_sha256="4" * 64,
        operator_id="operator",
        operator_approved=False,
        approved_at="2026-07-14T00:00:00+00:00",
        soak_passed=False,
        soak_duration_hours=0.0,
        soak_report_sha256="5" * 64,
    )

    assert evidence["operator_approval"]["operator_approved"] is False
    assert evidence["validation"]["scene_profile_operator_approval"] is False
    assert evidence["validation"]["corpus"]["independent_holdout"] is False
    assert evidence["validation"]["corpus"]["source_inventory_verified"] is False
    assert evidence["validation"]["metrics"]["false_positive_events_per_hour"] == 5.5879
    assert evidence["soak"]["passed"] is False
