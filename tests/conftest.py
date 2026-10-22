"""Shared test fixtures."""

# ruff: noqa: E402

import hashlib
import json
import sys
from pathlib import Path

# Ensure the worktree's src/ is first on sys.path so an editable install
# from another worktree doesn't shadow the code under test.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pytest

from argus.config.schema import (
    AlertConfig,
    AnomalyConfig,
    CameraConfig,
    MOG2Config,
    PersonFilterConfig,
    SceneGuardConfig,
)
from argus.core.event_bus import EventBus
from argus.preprocessing.scene_profile import runtime_artifact_sha256
from argus.storage.release_evidence import (
    RELEASE_EVIDENCE_SCHEMA,
    SOAK_REPORT_SCHEMA,
    canonical_sha256,
    evidence_artifact_sha256,
)


@pytest.fixture
def running_release_client():
    """Create clients with the real release-worker startup/shutdown lifecycle."""
    from contextlib import ExitStack

    from fastapi.testclient import TestClient

    with ExitStack() as stack:
        yield lambda app: stack.enter_context(TestClient(app))


@pytest.fixture
def await_release_operation():
    """Separate HTTP acceptance from durable runtime/application completion."""
    import time

    def wait(client, response, *, status="applied"):
        assert response.status_code == 202, response.text
        data = response.json()["data"]
        operation_id = data["operation"]["operation_id"]
        service = client.app.state.model_release_service
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            operation = service.get_operation(operation_id)
            if operation.status != "pending":
                assert operation.status == status, operation.to_dict()
                models = client.get("/api/models/json").json()["data"]["models"]
                model = next(
                    row for row in models
                    if row["model_version_id"] == operation.model_version_id
                )
                return {
                    **data,
                    "operation": operation.to_dict(),
                    "model": model,
                    "runtime_synced": operation.status == "applied",
                    "runtime_state": operation.to_dict()["runtime_state"],
                }
            time.sleep(0.01)
        raise AssertionError(f"release operation {operation_id} did not complete")

    return wait


@pytest.fixture
def blank_frame():
    """A 480x640 black frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def noisy_frame():
    """A 480x640 frame with random noise (simulates change)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_with_anomaly_patch(blank_frame):
    """A black frame with a saturated red rectangle (simulates anomaly).

    Pure white (BGR=255,255,255) is HSV saturation=0 and gets filtered out by
    the prefilter's chromaticity-based shadow suppression. Use a saturated
    color so the patch survives shadow filtering.
    """
    frame = blank_frame.copy()
    frame[100:200, 150:300] = (0, 0, 255)
    return frame


@pytest.fixture
def camera_config():
    """A test camera config using file protocol."""
    return CameraConfig(
        camera_id="test_cam",
        name="Test Camera",
        source="test_video.mp4",
        protocol="file",
        fps_target=30,
        scene_guard=SceneGuardConfig(enabled=False),
        mog2=MOG2Config(
            history=50,
            var_threshold=25.0,
            detect_shadows=True,
            change_pct_threshold=0.005,
        ),
        person_filter=PersonFilterConfig(
            confidence=0.4,
            skip_frame_on_person=False,
            model_name="yolo11n.pt",
        ),
        anomaly=AnomalyConfig(
            model_type="patchcore",
            threshold=0.7,
            image_size=(256, 256),
        ),
    )


@pytest.fixture
def alert_config():
    """A test alert config."""
    return AlertConfig()


@pytest.fixture
def event_bus():
    """A fresh EventBus instance, cleared after test."""
    bus = EventBus()
    yield bus
    bus.clear()


@pytest.fixture
def frame_sequence(blank_frame, noisy_frame):
    """A sequence of 10 frames alternating blank and noisy."""
    return [blank_frame if i % 2 == 0 else noisy_frame for i in range(10)]


@pytest.fixture
def release_evidence_factory():
    """Build structurally valid production evidence for release-gate tests."""

    def _build(
        *,
        model_path: str | Path,
        model_version_id: str,
        registry_model_hash: str,
        camera_id: str = "cam_01",
    ) -> dict:
        return {
            "schema_version": RELEASE_EVIDENCE_SCHEMA,
            "subject": {
                "model_version_id": model_version_id,
                "camera_id": camera_id,
                "registry_model_hash": registry_model_hash,
            },
            "artifacts": {
                "runtime_artifact_sha256": runtime_artifact_sha256(model_path),
                "config_sha256": "1" * 64,
                "scene_profile_sha256": "2" * 64,
                "baseline_sha256": "3" * 64,
                "corpus_sha256": "4" * 64,
                "regression_report_sha256": "5" * 64,
            },
            "operator_approval": {
                "operator_approved": True,
                "operator_id": "test-operator",
                "approved_at": "2026-07-14T00:00:00+00:00",
            },
            "validation": {
                "report_status_ok": True,
                "pipeline_evaluation": True,
                "production_parity": True,
                "trusted_scene_reference": True,
                "scene_profile_operator_approval": True,
                "runtime_model_identity": True,
                "positive_annotation_coverage": True,
                "candidate_confirmation": True,
                "corpus": {
                    "complete": True,
                    "independent_holdout": True,
                    "source_inventory_verified": True,
                    "normal_hours": 2.0,
                    "static_foreign_event_count": 20,
                },
                "scene_guard": {
                    "evidence_complete": True,
                    "coverage": 0.99,
                },
                "metrics": {
                    "false_positive_events_per_hour": 2.0,
                    "event_recall": 0.95,
                    "static_foreign_recall": 0.95,
                    "positive_annotation_coverage": 0.99,
                    "delay_p95_sec": 10.0,
                },
            },
            "soak": {
                "passed": True,
                "duration_hours": 72.0,
                "report_sha256": "6" * 64,
            },
        }

    return _build


@pytest.fixture
def release_baseline_factory():
    """Create one immutable baseline tree with auditable training lineage."""

    def _build(
        *,
        root: Path,
        training_source_sha256: str | None = None,
        camera_id: str = "cam_01",
        scene_profile_id: str = "release-profile-v1",
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        reference = root / "reference.png"
        reference.write_bytes(b"release-baseline-reference")
        sample = root / "images" / "sample_000000.png"
        sample.parent.mkdir(exist_ok=True)
        sample.write_bytes(b"release-baseline-sample")
        calibration_sample = root / "images" / "sample_000001.png"
        calibration_sample.write_bytes(b"release-calibration-sample")
        training_source_sha256 = training_source_sha256 or hashlib.sha256(
            b"independent-training-source"
        ).hexdigest()
        calibration_source_sha256 = hashlib.sha256(
            b"independent-calibration-source"
        ).hexdigest()
        annotation_sha256 = hashlib.sha256(b"reviewed-annotation").hexdigest()
        review_common = {
            "complete": True,
            "reasons": [],
            "schema_version": "argus.normal-video-review.v1",
            "camera_id": camera_id,
            "camera_id_matches": True,
            "review_complete": True,
            "reviewed_by": "test-reviewer",
            "reviewed_at": "2026-07-14T00:00:00+00:00",
            "reviewed_at_has_timezone": True,
        }
        manifest = {
            "schema_version": "argus.scene-profile-baseline.v2",
            "camera_id": camera_id,
            "scene_profile_id": scene_profile_id,
            "validation_only": True,
            "release_eligible": False,
            "operator_approved": False,
            "active_profile_modified": False,
            "reference_auto_selected": False,
            "dataset_status": "offline_training_input_ready",
            "training_ready": True,
            "training_readiness_scope": "offline_training_input_only",
            "training_readiness_checks": {
                "corpus_evidence_complete": True,
                "at_least_two_declared_independent_sessions": True,
                "minimum_training_images_met": True,
                "minimum_calibration_images_met": True,
                "minimum_accepted_time_span_met": True,
                "full_source_grouping_has_no_diagnostic_samples": True,
            },
            "corpus_evidence_complete": True,
            "source_inventory_evidence_complete": True,
            "annotation_exclusion_evidence_complete": True,
            "selection_evidence_complete": True,
            "decoder_evidence": {
                "backend": "opencv.VideoCapture",
                "reported_frame_count_required_to_match": True,
                "decode_error_distinguishable_from_clean_eof": False,
                "variable_frame_rate_timestamps_preserved": False,
                "sufficient_for_model_release": False,
            },
            "split_contract": {
                "random_frame_split_allowed": False,
                "declared_session_ids": ["session-a", "session-b"],
                "declared_independent_session_count": 2,
                "minimum_distinct_sessions_for_training_ready": 2,
                "single_session_policy": "diagnostic_only",
                "source_counts": {
                    "train": 1,
                    "calibration": 1,
                    "diagnostic_only": 0,
                },
                "sample_counts": {
                    "train": 1,
                    "calibration": 1,
                    "diagnostic_only": 0,
                },
            },
            "reference": {
                "source_path": "immutable-training-reference.png",
                "copied_path": reference.name,
                "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "width": 16,
                "height": 16,
            },
            "sources": [
                {
                    "source_index": 0,
                    "path": "immutable-training-source.mp4",
                    "sha256": training_source_sha256,
                    "annotation_sha256": annotation_sha256,
                    "annotation_exclusions_auditable": True,
                    "source_exhausted": True,
                    "opencv_read_terminated": True,
                    "reported_frame_count_matches": True,
                    "source_read_error": None,
                    "truncated_by_max_images": False,
                    "split_role": "train",
                    "annotation_review_evidence": {
                        **review_common,
                        "source_video_sha256": training_source_sha256,
                        "source_video_sha256_matches": True,
                        "source_session_id": "session-a",
                    },
                },
                {
                    "source_index": 1,
                    "path": "immutable-calibration-source.mp4",
                    "sha256": calibration_source_sha256,
                    "annotation_sha256": annotation_sha256,
                    "annotation_exclusions_auditable": True,
                    "source_exhausted": True,
                    "opencv_read_terminated": True,
                    "reported_frame_count_matches": True,
                    "source_read_error": None,
                    "truncated_by_max_images": False,
                    "split_role": "calibration",
                    "annotation_review_evidence": {
                        **review_common,
                        "source_video_sha256": calibration_source_sha256,
                        "source_video_sha256_matches": True,
                        "source_session_id": "session-b",
                    },
                },
            ],
            "samples": [
                {
                    "sample_index": 0,
                    "path": sample.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                    "source_index": 0,
                    "source_sha256": training_source_sha256,
                    "split_role": "train",
                },
                {
                    "sample_index": 1,
                    "path": calibration_sample.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(
                        calibration_sample.read_bytes()
                    ).hexdigest(),
                    "source_index": 1,
                    "source_sha256": calibration_source_sha256,
                    "split_role": "calibration",
                },
            ],
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return root

    return _build


@pytest.fixture
def release_evidence_sources_factory():
    """Write server-side source artifacts for production promotion API tests."""

    def _build(
        *,
        root: Path,
        model_path: str | Path,
        model_version_id: str,
        registry_model_hash: str,
        baseline_path: str | Path,
        camera_id: str = "cam_01",
    ) -> dict[str, str]:
        root.mkdir(parents=True, exist_ok=True)
        config = root / "runtime.yaml"
        config.write_text(f"camera: {camera_id}\n", encoding="utf-8")
        baseline = Path(baseline_path).resolve(strict=True)
        baseline_manifest = json.loads(
            (baseline / "manifest.json").read_text(encoding="utf-8")
        )
        baseline_reference = baseline_manifest["reference"]

        profile_id = baseline_manifest["scene_profile_id"]
        scene_profile = root / "scene-profiles" / camera_id / profile_id
        scene_profile.mkdir(parents=True)
        profile_reference = scene_profile / "reference.jpg"
        profile_reference.write_bytes(b"normalized-release-scene-reference")
        profile_reference_sha256 = hashlib.sha256(
            profile_reference.read_bytes()
        ).hexdigest()
        runtime_digest = runtime_artifact_sha256(model_path)
        (scene_profile / "metadata.json").write_text(
            json.dumps({
                "schema_version": 1,
                "camera_id": camera_id,
                "profile_id": profile_id,
                "created_at": "2026-07-14T00:00:00+00:00",
                "reference": {
                    "filename": profile_reference.name,
                    "sha256": profile_reference_sha256,
                    "width": 16,
                    "height": 16,
                },
                "source": {
                    "kind": "existing_baseline",
                    "filename": Path(baseline_reference["copied_path"]).name,
                    "sha256": baseline_reference["sha256"],
                },
                "metadata": {
                    "operator_approved": True,
                    "validation_only": False,
                    "model_version_id": model_version_id,
                    "model_artifact_sha256": runtime_digest,
                    "model_artifact_hash_scheme": "argus-runtime-artifact-v1",
                },
            }, sort_keys=True),
            encoding="utf-8",
        )
        (scene_profile.parent / "active.json").write_text(
            json.dumps({
                "schema_version": 1,
                "camera_id": camera_id,
                "profile_id": profile_id,
                "updated_at": "2026-07-14T00:00:00+00:00",
            }, sort_keys=True),
            encoding="utf-8",
        )

        corpus = root / "corpus"
        clips_dir = corpus / "clips"
        annotations_dir = corpus / "annotations"
        clips_dir.mkdir(parents=True)
        annotations_dir.mkdir()
        video = clips_dir / "release-case.mp4"
        video.write_bytes(b"independent-regression-video")
        annotation = annotations_dir / "release-case.json"
        annotation.write_text('{"positive_ranges": []}\n', encoding="utf-8")
        video_relative = video.relative_to(corpus).as_posix()
        annotation_relative = annotation.relative_to(corpus).as_posix()
        (corpus / "manifest.yaml").write_text(
            "default_camera_id: " + camera_id + "\n"
            "clips:\n"
            "  - id: release-case\n"
            "    kind: normal\n"
            "    source_session_id: regression-session-z\n"
            "    video: " + video_relative + "\n"
            "    annotations: " + annotation_relative + "\n",
            encoding="utf-8",
        )
        video_sha256 = hashlib.sha256(video.read_bytes()).hexdigest()
        annotation_sha256 = hashlib.sha256(annotation.read_bytes()).hexdigest()
        evaluated_artifacts = {
            "camera_id": camera_id,
            "config_sha256": evidence_artifact_sha256(config),
            "scene_profile_sha256": evidence_artifact_sha256(scene_profile),
            "baseline_sha256": evidence_artifact_sha256(baseline),
            "corpus_sha256": evidence_artifact_sha256(corpus),
        }
        evaluated_scene_profile = {
            "camera_id": camera_id,
            "profile_id": profile_id,
            "reference_sha256": profile_reference_sha256,
            "source_baseline_sha256": baseline_reference["sha256"],
            "model_version_id": model_version_id,
            "model_artifact_sha256": runtime_digest,
        }
        evaluated_source_inventory = [{
            "clip_id": "release-case",
            "camera_id": camera_id,
            "source_session_id": "regression-session-z",
            "video_relative_path": video_relative,
            "source_video_sha256": video_sha256,
            "annotation_relative_path": annotation_relative,
            "annotation_sha256": annotation_sha256,
        }]

        report = {
            "status": "ok",
            "evaluation_mode": "pipeline",
            "fp_gap_seconds": 60.0,
            "warmup_seconds": 30.0,
            "min_scene_guard_coverage": 0.99,
            "min_effective_eval_fps": 5.0,
            "min_projectile_eval_fps": 59.0,
            "suite": {
                "acceptance": {
                    "corpus_completeness": "pass",
                    "independent_holdout": "pass",
                    "scene_profile_operator_approval": "pass",
                    "production_parity": "pass",
                    "trusted_scene_reference": "pass",
                    "runtime_model_identity": "pass",
                    "positive_annotation_coverage": "pass",
                    "candidate_confirmation": "pass",
                },
                "corpus_complete": True,
                "independent_holdout_evidence_complete": True,
                "source_inventory_evidence_complete": True,
                "kind_specific_sampling_evidence_complete": True,
                "positive_events_by_kind": {
                    "static_foreign": 20,
                    "projectile": 20,
                },
                "positive_event_recall_by_kind": {
                    "static_foreign": 0.95,
                    "projectile": 0.95,
                },
                "normal_hours": 2.0,
                "static_foreign_event_count": 20,
                "positive_annotation_coverage_evidence_complete": True,
                "scene_guard_all_frame_evidence_complete": True,
                "scene_guard_all_frame_usable_coverage_ratio": 0.99,
                "false_positive_events_per_hour": 2.0,
                "event_recall": 0.95,
                "static_foreign_event_recall": 0.95,
                "positive_annotation_coverage_ratio": 0.99,
                "delay_p95_sec": 10.0,
                "scene_profile_operator_approval_evidence_complete": True,
                "production_parity_evidence_complete": True,
                "trusted_scene_reference_evidence_complete": True,
                "runtime_model_identity_evidence_complete": True,
                "candidate_contract_evidence_complete": True,
                "evaluated_artifacts": evaluated_artifacts,
                "evaluated_source_inventory": evaluated_source_inventory,
                "evaluated_scene_profile": evaluated_scene_profile,
                "evaluated_artifact_binding_evidence_complete": True,
                "source_inventory": {
                    "source_video_sha256": [video_sha256],
                },
                "runtime_model_identity_keys": [{
                    "version_id": model_version_id,
                    "artifact_sha256": runtime_digest,
                }],
            },
        }
        regression_report = root / "regression-report.json"
        regression_report.write_text(
            json.dumps(report, sort_keys=True),
            encoding="utf-8",
        )
        soak_report = root / "soak-report.json"
        soak_telemetry = root / "soak-telemetry.jsonl"
        soak_telemetry.write_text(
            '{"timestamp":"2026-07-11T00:00:00+00:00","healthy":true}\n',
            encoding="utf-8",
        )
        soak_subject = {
            "model_version_id": model_version_id,
            "camera_id": camera_id,
            "registry_model_hash": registry_model_hash,
            "runtime_artifact_sha256": runtime_digest,
            **{
                key: value
                for key, value in evaluated_artifacts.items()
                if key != "camera_id"
            },
            "regression_report_sha256": canonical_sha256(report),
        }
        soak_report.write_text(
            json.dumps({
                "schema_version": SOAK_REPORT_SCHEMA,
                "producer": {
                    "name": "untrusted-test-soak-producer",
                    "version": "0.0-test",
                    "build_sha256": "7" * 64,
                },
                "generated_at": "2026-07-14T00:00:01+00:00",
                "subject": soak_subject,
                "window": {
                    "started_at": "2026-07-11T00:00:00+00:00",
                    "ended_at": "2026-07-14T00:00:00+00:00",
                    "duration_hours": 72.0,
                },
                "telemetry": {
                    "raw_telemetry_sha256": hashlib.sha256(
                        soak_telemetry.read_bytes()
                    ).hexdigest(),
                    "sample_interval_seconds": 60.0,
                    "expected_sample_count": 4320,
                    "observed_sample_count": 4320,
                    "gap_count": 0,
                    "max_gap_seconds": 60.0,
                    "continuity_complete": True,
                },
                "result": {"passed": True},
                "attestation": {
                    "algorithm": "unverified-test-only",
                    "key_id": "test-key",
                    "signature": "not-trusted",
                },
            }),
            encoding="utf-8",
        )
        return {
            "regression_report_path": str(regression_report),
            "config_path": str(config),
            "scene_profile_path": str(scene_profile),
            "baseline_path": str(baseline),
            "corpus_path": str(corpus),
            "soak_report_path": str(soak_report),
            "soak_telemetry_path": str(soak_telemetry),
        }

    return _build
