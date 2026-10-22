from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from argus.config.schema import (
    AlertSeverity,
    ArgusConfig,
    CameraConfig,
    StaticCandidateConfirmationConfig,
)
from argus.core.diagnostics import DiagnosticsBuffer, FrameDiagnostics
from argus.core.pipeline import PipelineMode, _frame_content_sha256
from argus.preprocessing.scene_profile import (
    RUNTIME_ARTIFACT_HASH_SCHEME,
    SceneProfileStore,
)
from argus.validation.pipeline_video_eval import (
    VirtualVideoClock,
    _clip_annotations,
    _summarize_candidate_model_evidence,
    _summarize_prospective_candidate_fusion,
    _summarize_scene_guard,
    create_pipeline,
    evaluate_pipeline_video,
)


def _write_video(path: Path, frames: int = 8, fps: float = 2.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (16, 12),
    )
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((12, 16, 3), index * 10, dtype=np.uint8))
    writer.release()


class _Stats:
    def __init__(self) -> None:
        self.frames = 0

    def snapshot(self):
        return {"captured": self.frames, "alerts": 1}


class _FakePipeline:
    def __init__(self) -> None:
        self.mode = PipelineMode.LEARNING
        self.stats = _Stats()
        self.diagnostics = DiagnosticsBuffer()
        self.scene_references: list[dict] = []
        self._model_version_id = "model-v1"
        self._anomaly_model_path = None
        self._model_artifact_sha256 = None
        self._scene_guard_reference_frame = None
        self.camera_config = SimpleNamespace(
            camera_id="cam",
            static_confirmation=StaticCandidateConfirmationConfig(),
        )
        self._offline_evaluation_context = {
            "sensor_fusion_enabled": False,
            "synchronized_sensor_corpus": False,
            "cross_camera_enabled": False,
            "synchronized_cross_camera_corpus": False,
        }

    def initialize(self) -> bool:
        return True

    def set_mode(self, mode: PipelineMode):
        self.mode = mode

    def set_scene_reference(self, frame, *, source, profile_id, metadata):
        self.scene_references.append({
            "frame": frame.copy(),
            "source": source,
            "profile_id": profile_id,
            "metadata": dict(metadata),
        })

    def process_frame(self, frame_data):
        self.stats.frames += 1
        should_alert = self.mode == PipelineMode.ACTIVE and frame_data.frame_number == 4
        diag = FrameDiagnostics(
            frame_number=frame_data.frame_number,
            timestamp=frame_data.timestamp,
            camera_id=frame_data.camera_id,
            anomaly_score=0.9 if should_alert else 0.1,
            final_decision="alert" if should_alert else "normal",
        )
        self.diagnostics.append(diag)
        if not should_alert:
            return None
        return SimpleNamespace(
            alert_id="a1",
            camera_id=frame_data.camera_id,
            zone_id="default",
            frame_number=frame_data.frame_number,
            severity=AlertSeverity.MEDIUM,
            anomaly_score=0.9,
            detection_type="persistent_change",
            category="static_foreign",
            detected_objects=[],
            evidence_summary="persistent change",
            evidence_json={},
            event_group_id="event-1",
        )

    def get_diagnostics_buffer(self):
        return self.diagnostics

    def get_scene_guard_status(self):
        blocked = self.stats.frames in {3, 4}
        return {
            "enabled": True,
            "state": "moved" if blocked else "stable",
            "raw_state": "moved" if blocked else "stable",
            "usable": not blocked,
            "reason": "test pose",
            "profile_id": "pose-front-v1",
            "bound_model_version": "model-v1",
            "reference_source": None,
            "reference_sha256": None,
            "reference_available": False,
        }

    def get_detector_status(self):
        return SimpleNamespace(
            mode="ssim_fallback",
            model_path=None,
            model_loaded=False,
        )

    def shutdown(self):
        return None


class _ReleaseReadyPipeline(_FakePipeline):
    def __init__(
        self,
        *,
        model_path: Path,
        reference_frame: np.ndarray,
        profile_id: str = "pose-front-v1",
        bound_model_version: str | None = "model-v1",
        detector_loaded: bool = True,
        sensor_fusion_enabled: bool = False,
        operator_approved: bool | None = True,
    ) -> None:
        super().__init__()
        self._anomaly_model_path = model_path
        self._model_artifact_sha256 = "a" * 64
        self._scene_guard_reference_frame = reference_frame.copy()
        self._profile_id = profile_id
        self._bound_model_version = bound_model_version
        self._detector_loaded = detector_loaded
        profiles_dir = model_path.parent / "scene-profiles"
        baseline = model_path.parent / f"{profile_id}-baseline.png"
        assert cv2.imwrite(str(baseline), reference_frame)
        metadata: dict[str, object] = {}
        if bound_model_version is not None:
            metadata["model_version_id"] = bound_model_version
        metadata["model_artifact_sha256"] = self._model_artifact_sha256
        metadata["model_artifact_hash_scheme"] = RUNTIME_ARTIFACT_HASH_SCHEME
        metadata["validation_only"] = False
        if operator_approved is not None:
            metadata["operator_approved"] = operator_approved
        SceneProfileStore(profiles_dir).create_from_baseline(
            "cam",
            profile_id,
            baseline,
            metadata=metadata,
            activate=True,
        )
        self.camera_config.scene_guard = SimpleNamespace(
            profiles_dir=str(profiles_dir),
        )
        self._offline_evaluation_context = {
            "sensor_fusion_enabled": sensor_fusion_enabled,
            "synchronized_sensor_corpus": False,
            "cross_camera_enabled": False,
            "synchronized_cross_camera_corpus": False,
        }

    def get_scene_guard_status(self):
        return super().get_scene_guard_status() | {
            "profile_id": self._profile_id,
            "bound_model_version": self._bound_model_version,
            "reference_source": f"profile:{self._profile_id}",
            "reference_sha256": _frame_content_sha256(
                self._scene_guard_reference_frame
            ),
            "reference_available": True,
        }

    def get_detector_status(self):
        return SimpleNamespace(
            mode="anomalib" if self._detector_loaded else "ssim_fallback",
            model_path=str(self._anomaly_model_path),
            model_loaded=self._detector_loaded,
        )


class _SwallowingPipeline(_FakePipeline):
    """Simulate live-pipeline exception handling returning without a diagnostic."""

    def process_frame(self, frame_data):
        if frame_data.frame_number == 3:
            self.stats.frames += 1
            return None
        return super().process_frame(frame_data)


class _MultiAlertPipeline(_FakePipeline):
    """Expose two independently emitted zone alerts for one processed frame."""

    def process_frame(self, frame_data):
        primary = super().process_frame(frame_data)
        self._last_emitted_alerts = []
        if primary is None:
            return None
        secondary = SimpleNamespace(**vars(primary))
        secondary.alert_id = "a2"
        secondary.zone_id = "water_zone"
        secondary.anomaly_score = 0.88
        secondary.detection_type = "water_leak"
        secondary.category = "water_leak"
        self._last_emitted_alerts = [primary, secondary]
        return primary


def test_virtual_clock_advances_video_time() -> None:
    clock = VirtualVideoClock(epoch=1000.0, monotonic_origin=500.0)
    clock.set(12.5)
    assert clock.monotonic() == 512.5
    assert clock.time() == 1012.5


def test_create_pipeline_rejects_unknown_camera_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "argus.validation.pipeline_video_eval.load_config",
        lambda _path: SimpleNamespace(
            cameras=[SimpleNamespace(camera_id="known-camera")],
        ),
    )

    with pytest.raises(ValueError, match="unknown-camera.*known-camera"):
        create_pipeline(
            config_path=tmp_path / "config.yaml",
            camera_id="unknown-camera",
            video_path=tmp_path / "case.avi",
            model_path=None,
        )


def test_create_pipeline_passes_production_heads_and_disables_discovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = ArgusConfig(cameras=[CameraConfig(
        camera_id="cam",
        name="Camera",
        source="rtsp://production",
    )])
    captured: dict = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "argus.validation.pipeline_video_eval.load_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        "argus.validation.pipeline_video_eval.DetectionPipeline",
        fake_pipeline,
    )

    pipeline = create_pipeline(
        config_path=tmp_path / "config.yaml",
        camera_id="cam",
        video_path=tmp_path / "case.avi",
        model_path=None,
    )

    assert captured["classifier_config"] is config.classifier
    assert captured["segmenter_config"] is config.segmenter
    assert captured["physics_config"] is config.physics
    assert captured["imaging_config"] is config.imaging
    assert captured["auto_discover_model"] is False
    assert pipeline._offline_evaluation_context == {  # noqa: SLF001
        "sensor_fusion_enabled": False,
        "synchronized_sensor_corpus": False,
        "cross_camera_enabled": False,
        "synchronized_cross_camera_corpus": False,
    }


def test_clip_annotations_excludes_events_beyond_partial_replay() -> None:
    annotations = {
        "path": "labels.json",
        "positive_ranges": [
            {"start_frame": 2, "end_frame": 6, "label": 1},
            {"start_frame": 10, "end_frame": 12, "label": 1},
        ],
        "ignore_ranges": [{"start_frame": 4, "end_frame": 9, "label": -1}],
    }

    clipped = _clip_annotations(annotations, evaluated_frames=5)

    assert clipped["positive_ranges"] == [
        {"start_frame": 2, "end_frame": 4, "label": 1},
    ]
    assert clipped["ignore_ranges"] == [
        {"start_frame": 4, "end_frame": 4, "label": -1},
    ]


def test_scene_guard_summary_distinguishes_disabled_from_protected_usable() -> None:
    summary = _summarize_scene_guard(
        [{
            "scene_guard": {
                "enabled": False,
                "state": "disabled",
                "usable": True,
            },
        }],
        sample_period_seconds=0.2,
    )

    assert summary["usable_frames"] == 1
    assert summary["enabled_frames"] == 0
    assert summary["disabled_frames"] == 1
    assert summary["geometry_normalized_frames"] == 0
    assert summary["alignable_without_normalization_frames"] == 0
    assert summary["warp_border_policy_counts"] == {"not_reported": 1}


def test_scene_guard_summary_audits_alignable_geometry_normalization() -> None:
    summary = _summarize_scene_guard(
        [
            {
                "scene_guard": {
                    "enabled": True,
                    "state": "alignable",
                    "usable": True,
                    "geometry_normalization_applied": True,
                    "warp_border_policy": (
                        "approved_reference_fill_outside_source_footprint"
                    ),
                },
            },
            {
                "scene_guard": {
                    "enabled": True,
                    "state": "alignable",
                    "usable": True,
                    "geometry_normalization_applied": False,
                    "warp_border_policy": "not_reported",
                },
            },
        ],
        sample_period_seconds=1.0,
    )

    assert summary["geometry_normalized_frames"] == 1
    assert summary["alignable_without_normalization_frames"] == 1
    assert summary["warp_border_policy_counts"] == {
        "approved_reference_fill_outside_source_footprint": 1,
        "not_reported": 1,
    }


def test_candidate_model_summary_excludes_unavailable_from_reject_denominator() -> None:
    def record(status, would_accept):
        metrics = (
            {
                "model_score": 0.8,
                "roi_p95": 0.7,
                "roi_context_delta": 0.4,
            }
            if status in {"evaluated", "uncalibrated"}
            else {}
        )
        return {
            "candidate_type": "persistent_confirmation",
            "evidence": {
                "model_evidence": {
                    "status": status,
                    "would_accept": would_accept,
                    "output_space": "anomalib_postprocessed",
                    "calibration_id": "cal-1",
                    **metrics,
                }
            },
        }

    stale = record("evaluated", True)
    stale["evidence"]["observed_this_frame"] = False
    summary = _summarize_candidate_model_evidence([
        {"candidate_records": [record("evaluated", True)]},
        {"candidate_records": [record("evaluated", False)]},
        {"candidate_records": [record("rate_limited", None)]},
        {"candidate_records": [record("inference_error", None)]},
        {"candidate_records": [record("unavailable", None)]},
        {"candidate_records": [record("upstream_skipped", None)]},
        {"candidate_records": [record("uncalibrated", None)]},
        {
            "score_eligible": False,
            "candidate_records": [record("evaluated", True)],
        },
        {"score_eligible": True, "candidate_records": [stale]},
    ])

    assert summary["observed_candidates"] == 7
    assert summary["all_observed_candidates"] == 9
    assert summary["excluded_candidate_observations"] == 1
    assert summary["not_observed_candidate_records"] == 1
    assert summary["evaluated_candidates"] == 2
    assert summary["eligible_candidates"] == 6
    assert summary["telemetry_coverage_ratio"] == pytest.approx(0.5)
    assert summary["map_extraction_eligible_candidates"] == 6
    assert summary["map_extracted_candidates"] == 3
    assert summary["map_extraction_coverage_ratio"] == pytest.approx(0.5)
    assert summary["calibrated_evaluation_eligible_candidates"] == 3
    assert summary["calibrated_evaluated_candidates"] == 2
    assert summary["calibrated_evaluation_coverage_ratio"] == pytest.approx(2 / 3)
    assert summary["accepted_candidates"] == 1
    assert summary["rejected_candidates"] == 1
    assert summary["rule_accept_rate"] == pytest.approx(0.5)
    assert summary["nullable_contract_violations"] == 0


def test_prospective_summary_uses_outer_fusion_and_temporal_decision() -> None:
    def record(would_accept, *, streak, confirmed, status="evaluated"):
        return {
            "candidate_type": "persistent_confirmation",
            "evidence": {
                "decision_status": status,
                "would_accept": would_accept,
                "mode": "enforce",
                "fusion_policy": "model_required",
                "temporal_evidence": {
                    "status": status,
                    "status_reason": "temporal_confirmed" if confirmed else "pending",
                    "streak": streak,
                    "confirmed": confirmed,
                },
            },
        }

    accepted = record(True, streak=2, confirmed=True)
    pending = record(False, streak=1, confirmed=False)
    rejected = record(False, streak=0, confirmed=False)
    unavailable = record(None, streak=0, confirmed=False, status="upstream_skipped")
    would_suppress_global = {
        "candidate_type": "anomaly_attribution",
        "evidence": {
            "would_suppress": True,
            "effective": False,
            "reason": "all_hot_support_attributed_to_rejected_candidates",
            "hot_coverage": 0.9,
        },
    }
    would_preserve_global = {
        "candidate_type": "anomaly_attribution",
        "evidence": {
            "would_suppress": False,
            "effective": False,
            "reason": "independent_hotspot_preserved",
            "hot_coverage": 0.2,
        },
    }
    summary = _summarize_prospective_candidate_fusion([
        {
            "score_eligible": True,
            "candidate_records": [accepted, pending, would_suppress_global],
        },
        {
            "score_eligible": True,
            "candidate_records": [rejected, unavailable, would_preserve_global],
        },
        {
            "score_eligible": False,
            "candidate_records": [accepted, would_suppress_global],
        },
    ])

    assert summary["observed_candidates"] == 4
    assert summary["all_observed_candidates"] == 5
    assert summary["evaluated_candidates"] == 3
    assert summary["accepted_candidates"] == 1
    assert summary["pending_candidates"] == 1
    assert summary["rejected_candidates"] == 1
    assert summary["unavailable_candidates"] == 1
    assert summary["accept_rate"] == pytest.approx(1 / 3)
    assert summary["confirmed_candidates"] == 1
    assert summary["mode_counts"] == {"enforce": 4}
    assert summary["fusion_policy_counts"] == {"model_required": 4}
    attribution = summary["anomaly_attribution"]
    assert attribution["observed_frames"] == 2
    assert attribution["excluded_frame_observations"] == 1
    assert attribution["would_suppress_frames"] == 1
    assert attribution["would_preserve_frames"] == 1
    assert attribution["suppress_rate"] == pytest.approx(0.5)
    assert attribution["hot_coverage_summary"]["mean"] == pytest.approx(0.55)


def test_pipeline_video_eval_uses_final_alerts_and_warmup(tmp_path: Path) -> None:
    video = tmp_path / "case.avi"
    annotation = tmp_path / "case.labels.json"
    _write_video(video)
    annotation.write_text(json.dumps({
        "positive_ranges": [{"start_frame": 4, "end_frame": 5, "label": "static"}],
        "ignore_ranges": [],
    }))
    pipeline = _FakePipeline()

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        annotation_path=annotation,
        sample_every=1,
        warmup_seconds=1.0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    by_frame = {record["frame_index"]: record for record in result["frames"]}
    assert result["evaluation_mode"] == "pipeline"
    assert result["schema_version"] == "argus.pipeline-video-eval.v1"
    assert result["evaluation_origin"] == "fresh"
    assert result["source_video_sha256"] == hashlib.sha256(
        video.read_bytes()
    ).hexdigest()
    assert result["annotation_sha256"] == hashlib.sha256(
        annotation.read_bytes()
    ).hexdigest()
    assert result["source_artifacts_stable"] is True
    assert result["annotation_inventory"] == {
        "raw_range_counts": {"positive": 1, "ignore": 0},
        "parsed_range_counts": {"positive": 1, "ignore": 0},
        "clipped_range_counts": {"positive": 1, "ignore": 0},
        "invalid_range_count": 0,
        "out_of_bounds_range_count": 0,
        "loader_normalized_range_count": 0,
        "clipped_or_dropped_range_count": 0,
        "all_ranges_within_source": True,
    }
    assert result["alerts"][0]["detection_type"] == "persistent_change"
    assert by_frame[4]["decision"] == "tp"
    assert by_frame[4]["predicted"] == 1
    assert by_frame[0]["pipeline_mode"] == "learning"
    assert by_frame[0]["decision"] == "excluded"
    assert by_frame[0]["score_eligible"] is False
    assert by_frame[0]["exclusion_reason"] == "warmup"
    assert by_frame[2]["pipeline_mode"] == "active"
    assert by_frame[2]["score_eligible"] is True
    assert result["failed_frames"] == []
    assert result["source_fps_available"] is True
    assert result["source_total_frames_known"] is True
    assert result["source_exhausted"] is True
    assert result["truncated_by_max_frames"] is False
    assert result["source_read_error"] is False
    assert result["max_frames"] is None
    assert result["total_frames"] == result["source_total_frames"] == 8
    assert result["processed_frames"] == 8
    assert result["sample_every"] == 1
    assert result["effective_eval_fps"] == pytest.approx(2.0)
    assert result["expected_processed_frames"] == 8
    assert result["scored_frames"] == 6
    assert result["warmup_frames"] == 2
    assert result["excluded_frames"] == 2
    assert result["scene_reference"]["mode"] == "video_frame"
    assert result["scene_reference"]["frame_index"] == 0
    assert result["scene_reference"]["profile_id"] == "pose-front-v1"
    assert result["scene_reference"]["bound_model_version"] == "model-v1"
    assert result["scene_reference"]["release_approved"] is False
    assert result["model_identity"]["release_approved"] is False
    assert result["production_parity"]["supported"] is True
    assert result["release_contract"]["approved"] is False
    assert result["candidate_confirmation_contract"]["available"] is True
    assert result["candidate_confirmation_contract"]["enabled"] is False
    assert pipeline.scene_references[0]["source"] == "offline-eval:video-frame:0"
    assert pipeline.scene_references[0]["profile_id"] == "pose-front-v1"
    assert pipeline.scene_references[0]["metadata"]["offline_evaluation"] is True
    assert pipeline.scene_references[0]["metadata"]["model_version_id"] == "model-v1"
    assert pipeline.scene_references[0]["frame"].shape == (12, 16, 3)
    assert result["scene_guard_summary"] == {
        "state_counts": {"moved": 2, "stable": 4},
        "observed_frames": 6,
        "enabled_frames": 6,
        "disabled_frames": 0,
        "usable_frames": 4,
        "blocked_frames": 2,
        "geometry_normalized_frames": 0,
        "alignable_without_normalization_frames": 0,
        "warp_border_policy_counts": {"not_reported": 6},
        "usable_coverage_ratio": 2 / 3,
        "blocked_seconds_estimate": 1.0,
        "longest_blocked_seconds_estimate": 1.0,
        "availability_transitions": 1,
    }
    assert result["scene_guard_summary_score_eligible"] == result["scene_guard_summary"]
    assert result["scene_guard_summary_all"]["observed_frames"] == 8
    assert result["scene_guard_summary_all"]["usable_coverage_ratio"] == 0.75


def test_pipeline_video_eval_serializes_all_zone_alerts(tmp_path: Path) -> None:
    video = tmp_path / "case.avi"
    _write_video(video)
    pipeline = _MultiAlertPipeline()

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        sample_every=1,
        warmup_seconds=1.0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    frame = next(item for item in result["frames"] if item["frame_index"] == 4)
    assert frame["predicted"] == 1
    assert frame["alert_count"] == 2
    assert frame["alert_ids"] == ["a1", "a2"]
    assert frame["detection_types"] == ["persistent_change", "water_leak"]
    alerts = [item for item in result["alerts"] if item["frame_number"] == 4]
    assert {(item["alert_id"], item["zone_id"]) for item in alerts} == {
        ("a1", "default"),
        ("a2", "water_zone"),
    }


def test_pipeline_video_eval_rejects_annotation_toctou_mutation(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    annotation = tmp_path / "case.labels.json"
    _write_video(video, frames=2)
    annotation.write_text(
        json.dumps({"positive_ranges": [], "ignore_ranges": []}),
        encoding="utf-8",
    )
    pipeline = _FakePipeline()
    original_process_frame = pipeline.process_frame
    mutated = False

    def mutating_process_frame(frame_data):
        nonlocal mutated
        if not mutated:
            annotation.write_text(
                json.dumps({
                    "positive_ranges": [
                        {"start_frame": 0, "end_frame": 0},
                    ],
                    "ignore_ranges": [],
                }),
                encoding="utf-8",
            )
            mutated = True
        return original_process_frame(frame_data)

    pipeline.process_frame = mutating_process_frame  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="annotation changed"):
        evaluate_pipeline_video(
            video,
            config_path=tmp_path / "unused.yaml",
            camera_id="cam",
            annotation_path=annotation,
            warmup_seconds=0.0,
            pipeline_factory=lambda **_kwargs: pipeline,
        )


def test_pipeline_video_eval_rehashes_source_video_at_eof(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import argus.validation.pipeline_video_eval as evaluator

    video = tmp_path / "case.avi"
    annotation = tmp_path / "case.labels.json"
    _write_video(video, frames=2)
    annotation.write_text(
        json.dumps({"positive_ranges": [], "ignore_ranges": []}),
        encoding="utf-8",
    )
    original_hash = evaluator._stream_sha256
    video_hash_calls = 0

    def changing_video_hash(path: Path) -> str:
        nonlocal video_hash_calls
        digest = original_hash(path)
        if Path(path) == video:
            video_hash_calls += 1
            if video_hash_calls == 2:
                return "0" * 64
        return digest

    monkeypatch.setattr(evaluator, "_stream_sha256", changing_video_hash)

    with pytest.raises(RuntimeError, match="source video changed"):
        evaluate_pipeline_video(
            video,
            config_path=tmp_path / "unused.yaml",
            camera_id="cam",
            annotation_path=annotation,
            warmup_seconds=0.0,
            pipeline_factory=lambda **_kwargs: _FakePipeline(),
        )


@pytest.mark.parametrize(
    "positive_range",
    [
        {"start_frame": -1, "end_frame": 1},
        {"start_frame": 3, "end_frame": 2},
        {"start_frame": 1},
        {"start_frame": 1, "end_frame": 8},
        {"start_frame": "1", "end_frame": 2},
    ],
)
def test_pipeline_video_eval_audits_invalid_or_out_of_bounds_annotations(
    tmp_path: Path,
    positive_range,
) -> None:
    video = tmp_path / "case.avi"
    annotation = tmp_path / "case.labels.json"
    _write_video(video, frames=8)
    annotation.write_text(
        json.dumps({
            "positive_ranges": [positive_range],
            "ignore_ranges": [],
        }),
        encoding="utf-8",
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        annotation_path=annotation,
        warmup_seconds=0.0,
        pipeline_factory=lambda **_kwargs: _FakePipeline(),
    )

    assert result["annotation_inventory"]["all_ranges_within_source"] is False


def test_pipeline_video_eval_uses_supplied_reference_image(tmp_path: Path) -> None:
    video = tmp_path / "case.avi"
    reference = tmp_path / "approved-reference.png"
    _write_video(video, frames=2)
    cv2.imwrite(str(reference), np.full((10, 14, 3), 77, dtype=np.uint8))
    pipeline = _FakePipeline()

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        max_frames=1,
        warmup_seconds=0,
        scene_reference_path=reference,
        scene_profile_id="approved-profile-v2",
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    loaded = pipeline.scene_references[0]
    assert result["scene_reference"]["mode"] == "image"
    assert result["scene_reference"]["path"] == str(reference)
    assert result["scene_reference"]["profile_id"] == "approved-profile-v2"
    assert result["scene_reference"]["configured_profile_id"] == "pose-front-v1"
    assert result["scene_reference"]["profile_id_matches_configured"] is False
    assert result["scene_reference"]["release_approved"] is False
    assert result["source_exhausted"] is False
    assert result["truncated_by_max_frames"] is True
    assert result["source_read_error"] is False
    assert result["max_frames"] == 1
    assert result["total_frames"] == 1
    assert loaded["profile_id"] == "approved-profile-v2"
    assert loaded["frame"].shape == (10, 14, 3)
    assert np.all(loaded["frame"] == 77)


def test_release_requires_active_profile_and_explicit_matching_model_identity(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    model = tmp_path / "model.pt"
    _write_video(video, frames=1)
    reference_frame = np.full((12, 16, 3), 31, dtype=np.uint8)
    pipeline = _ReleaseReadyPipeline(
        model_path=model,
        reference_frame=reference_frame,
    )
    pipeline.camera_config.static_confirmation = (
        StaticCandidateConfirmationConfig(enabled=True, mode="shadow")
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        model_path=model,
        model_version_id="model-v1",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    reference = result["scene_reference"]
    assert reference["mode"] == "active_profile"
    assert reference["configured_profile_initialized"] is True
    assert reference["profile_id_matches_configured"] is True
    assert reference["matches_configured_reference"] is True
    assert reference["technical_reference_match"] is True
    assert reference["operator_approved"] is True
    assert reference["operator_authorization"]["metadata_field_present"] is True
    assert reference["operator_authorization"]["reason"] is None
    assert reference["reference_approved"] is True
    assert reference["release_approved"] is True
    identity = result["model_identity"]
    assert identity["requested"]["explicit_path"] is True
    assert identity["requested"]["explicit_version"] is True
    assert identity["actual"]["detector_loaded"] is True
    assert identity["actual"]["artifact_sha256"] == "a" * 64
    assert identity["normalized_path_match"] is True
    assert identity["version_match"] is True
    assert identity["release_approved"] is True
    assert result["production_parity"]["supported"] is True
    assert result["release_contract"] == {
        "approved": True,
        "reference_approved": True,
        "operator_approved": True,
        "operator_approval_reason": None,
        "model_identity_approved": True,
        "production_parity_supported": True,
        "reasons": [],
    }
    assert result["candidate_confirmation_contract"]["available"] is True
    assert result["candidate_confirmation_contract"]["enabled"] is True
    assert result["candidate_confirmation_contract"]["mode"] == "shadow"


@pytest.mark.parametrize(
    ("operator_approved", "reason"),
    [
        (False, "scene_profile_operator_approval_not_granted"),
        (None, "scene_profile_operator_approval_missing"),
    ],
)
def test_active_validation_profile_cannot_authorize_production_release(
    tmp_path: Path,
    operator_approved: bool | None,
    reason: str,
) -> None:
    video = tmp_path / "case.avi"
    model = tmp_path / "model.pt"
    _write_video(video, frames=1)
    pipeline = _ReleaseReadyPipeline(
        model_path=model,
        reference_frame=np.full((12, 16, 3), 33, dtype=np.uint8),
        operator_approved=operator_approved,
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        model_path=model,
        model_version_id="model-v1",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    reference = result["scene_reference"]
    assert reference["technical_reference_match"] is True
    assert reference["operator_approved"] is False
    assert reference["reference_approved"] is False
    assert reference["operator_authorization"]["reason"] == reason
    assert result["model_identity"]["release_approved"] is True
    assert result["release_contract"]["approved"] is False
    assert result["release_contract"]["operator_approved"] is False
    assert result["release_contract"]["operator_approval_reason"] == reason
    assert result["release_contract"]["reasons"] == [reason]


def test_runtime_model_identity_is_captured_after_pipeline_initialize(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    requested_model = tmp_path / "initialized.pt"
    preinitialize_model = tmp_path / "constructor.pt"
    _write_video(video, frames=1)
    pipeline = _ReleaseReadyPipeline(
        model_path=preinitialize_model,
        reference_frame=np.full((12, 16, 3), 36, dtype=np.uint8),
    )

    def initialize():
        pipeline._anomaly_model_path = requested_model
        pipeline._model_artifact_sha256 = "b" * 64
        return True

    pipeline.initialize = initialize  # type: ignore[method-assign]
    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        model_path=requested_model,
        model_version_id="model-v1",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    identity = result["model_identity"]
    assert identity["actual"]["pipeline_path"] == str(requested_model)
    assert identity["actual"]["detector_path"] == str(requested_model)
    assert identity["actual"]["artifact_sha256"] == "b" * 64
    assert identity["normalized_path_match"] is True
    assert identity["release_approved"] is True


def test_configured_reference_hash_must_match_initialized_profile_frame(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    model = tmp_path / "model.pt"
    _write_video(video, frames=1)
    pipeline = _ReleaseReadyPipeline(
        model_path=model,
        reference_frame=np.full((12, 16, 3), 38, dtype=np.uint8),
    )
    original_status = pipeline.get_scene_guard_status

    def wrong_hash_status():
        return original_status() | {"reference_sha256": "0" * 64}

    pipeline.get_scene_guard_status = wrong_hash_status  # type: ignore[method-assign]
    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        model_path=model,
        model_version_id="model-v1",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    reference = result["scene_reference"]
    assert reference["configured_profile_initialized"] is False
    assert reference["matches_configured_reference"] is False
    assert reference["reference_approved"] is False
    assert result["release_contract"]["approved"] is False


def test_arbitrary_profile_id_cannot_approve_exact_configured_reference(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    model = tmp_path / "model.pt"
    _write_video(video, frames=1)
    pipeline = _ReleaseReadyPipeline(
        model_path=model,
        reference_frame=np.full((12, 16, 3), 42, dtype=np.uint8),
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        model_path=model,
        model_version_id="model-v1",
        scene_profile_id="caller-nominated-profile",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    reference = result["scene_reference"]
    assert reference["matches_configured_reference"] is True
    assert reference["profile_id_matches_configured"] is False
    assert reference["reference_approved"] is False
    assert result["release_contract"]["approved"] is False


def test_profile_bound_model_is_not_synthesized_from_runtime(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    model = tmp_path / "model.pt"
    _write_video(video, frames=1)
    pipeline = _ReleaseReadyPipeline(
        model_path=model,
        reference_frame=np.full((12, 16, 3), 52, dtype=np.uint8),
        bound_model_version=None,
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        model_path=model,
        model_version_id="model-v1",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    assert result["scene_reference"]["bound_model_version"] is None
    assert (
        result["scene_reference"]["profile_bound_model_matches_runtime"]
        is False
    )
    assert result["scene_reference"]["reference_approved"] is False
    assert result["release_contract"]["approved"] is False


def test_release_rejects_missing_explicit_model_and_sensor_parity_gap(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    actual_model = tmp_path / "actual.pt"
    _write_video(video, frames=1)
    pipeline = _ReleaseReadyPipeline(
        model_path=actual_model,
        reference_frame=np.full((12, 16, 3), 62, dtype=np.uint8),
        sensor_fusion_enabled=True,
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    assert result["model_identity"]["requested"]["explicit_path"] is False
    assert result["model_identity"]["requested"]["explicit_version"] is False
    assert result["model_identity"]["release_approved"] is False
    assert result["production_parity"] == {
        "supported": False,
        "sensor_fusion_enabled": True,
        "synchronized_sensor_corpus": False,
        "cross_camera_enabled": False,
        "synchronized_cross_camera_corpus": False,
        "reason": "sensor_fusion_enabled_without_synchronized_sensor_corpus",
        "reasons": ["sensor_fusion_enabled_without_synchronized_sensor_corpus"],
    }
    assert result["release_contract"]["approved"] is False
    assert "model_identity_not_approved" in result["release_contract"]["reasons"]
    assert (
        "sensor_fusion_enabled_without_synchronized_sensor_corpus"
        in result["release_contract"]["reasons"]
    )
    assert result["scene_reference"]["release_approved"] is False


def test_release_rejects_requested_and_actual_model_path_mismatch(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    requested_model = tmp_path / "requested.pt"
    actual_model = tmp_path / "actual.pt"
    _write_video(video, frames=1)
    pipeline = _ReleaseReadyPipeline(
        model_path=actual_model,
        reference_frame=np.full((12, 16, 3), 72, dtype=np.uint8),
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        model_path=requested_model,
        model_version_id="model-v1",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    identity = result["model_identity"]
    assert identity["actual"]["pipeline_path"] == str(actual_model)
    assert identity["actual"]["detector_path"] == str(actual_model)
    assert identity["normalized_path_match"] is False
    assert identity["release_approved"] is False
    assert result["release_contract"]["approved"] is False


def test_release_rejects_cross_camera_policy_without_synchronized_replay(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    model = tmp_path / "model.pt"
    _write_video(video, frames=1)
    pipeline = _ReleaseReadyPipeline(
        model_path=model,
        reference_frame=np.full((12, 16, 3), 76, dtype=np.uint8),
    )
    pipeline._offline_evaluation_context.update({  # noqa: SLF001
        "cross_camera_enabled": True,
        "synchronized_cross_camera_corpus": False,
    })

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        model_path=model,
        model_version_id="model-v1",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    parity = result["production_parity"]
    assert parity["supported"] is False
    assert parity["cross_camera_enabled"] is True
    assert parity["reason"] == (
        "cross_camera_enabled_without_synchronized_multi_camera_corpus"
    )
    assert result["release_contract"]["approved"] is False


def test_pipeline_video_eval_records_scene_profile_model_mismatch_for_diagnostics(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    _write_video(video, frames=1)
    pipeline = _FakePipeline()
    original_status = pipeline.get_scene_guard_status

    def mismatched_status():
        return original_status() | {"bound_model_version": "old-model"}

    pipeline.get_scene_guard_status = mismatched_status  # type: ignore[method-assign]

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    assert result["scene_reference"]["bound_model_version"] == "old-model"
    assert (
        result["scene_reference"]["profile_bound_model_matches_runtime"]
        is False
    )
    assert result["scene_reference"]["release_approved"] is False
    assert len(pipeline.scene_references) == 1


def test_pipeline_video_eval_rejects_stale_diagnostic_after_swallowed_error(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    _write_video(video, frames=5)

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: _SwallowingPipeline(),
    )

    assert result["failed_frames"] == [{
        "frame_index": 3,
        "error": "pipeline produced no current-frame diagnostic (latest=2)",
    }]
    assert 3 not in {record["frame_index"] for record in result["frames"]}
    assert result["scene_guard_summary"]["observed_frames"] == 4


def test_ignore_ranges_skip_scoring_but_preserve_pipeline_sequence(
    tmp_path: Path,
) -> None:
    video = tmp_path / "case.avi"
    annotation = tmp_path / "case.labels.json"
    _write_video(video, frames=5)
    annotation.write_text(json.dumps({
        "positive_ranges": [],
        "ignore_ranges": [{"start_frame": 2, "end_frame": 2, "label": -1}],
    }))
    pipeline = _FakePipeline()

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        annotation_path=annotation,
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    assert pipeline.stats.frames == 5
    assert result["processed_frames"] == 5
    assert result["scored_frames"] == 4
    assert result["ignored_frames"] == 1
    by_frame = {record["frame_index"]: record for record in result["frames"]}
    assert by_frame[2]["score_eligible"] is False
    assert by_frame[2]["exclusion_reason"] == "annotation_ignore"
    assert by_frame[2]["decision"] == "excluded"
    assert result["scene_guard_summary"]["observed_frames"] == 4
    assert result["scene_guard_summary_all"]["observed_frames"] == 5


def test_pipeline_video_eval_rejects_negative_reference_frame(tmp_path: Path) -> None:
    video = tmp_path / "case.avi"
    _write_video(video, frames=2)

    with pytest.raises(ValueError, match="scene_reference_frame_index"):
        evaluate_pipeline_video(
            video,
            config_path=tmp_path / "unused.yaml",
            camera_id="cam",
            scene_reference_frame_index=-1,
            pipeline_factory=lambda **_kwargs: _FakePipeline(),
        )


@pytest.mark.parametrize("max_frames", [0, -1])
def test_pipeline_video_eval_rejects_nonpositive_max_frames(
    tmp_path: Path,
    max_frames: int,
) -> None:
    video = tmp_path / "case.avi"
    _write_video(video, frames=2)

    with pytest.raises(ValueError, match="max_frames must be >= 1"):
        evaluate_pipeline_video(
            video,
            config_path=tmp_path / "unused.yaml",
            camera_id="cam",
            max_frames=max_frames,
            pipeline_factory=lambda **_kwargs: _FakePipeline(),
        )


def test_pipeline_video_eval_marks_unknown_source_count_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "case.avi"
    _write_video(video, frames=8)
    original_capture = cv2.VideoCapture

    class UnknownCountCapture:
        def __init__(self, *args, **kwargs) -> None:
            self._capture = original_capture(*args, **kwargs)

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 0.0
            return float(self._capture.get(property_id))

        def __getattr__(self, name: str):
            return getattr(self._capture, name)

    monkeypatch.setattr(
        "argus.validation.pipeline_video_eval.cv2.VideoCapture",
        UnknownCountCapture,
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: _FakePipeline(),
    )

    assert result["source_total_frames"] == 0
    assert result["source_total_frames_known"] is False
    assert result["source_exhausted"] is True
    assert result["truncated_by_max_frames"] is False
    assert result["source_read_error"] is False
    assert result["total_frames"] == 8


def test_pipeline_video_eval_marks_early_capture_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "case.avi"
    _write_video(video, frames=8)
    original_capture = cv2.VideoCapture

    class InflatedCountCapture:
        def __init__(self, *args, **kwargs) -> None:
            self._capture = original_capture(*args, **kwargs)

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 10.0
            return float(self._capture.get(property_id))

        def __getattr__(self, name: str):
            return getattr(self._capture, name)

    monkeypatch.setattr(
        "argus.validation.pipeline_video_eval.cv2.VideoCapture",
        InflatedCountCapture,
    )

    result = evaluate_pipeline_video(
        video,
        config_path=tmp_path / "unused.yaml",
        camera_id="cam",
        warmup_seconds=0,
        pipeline_factory=lambda **_kwargs: _FakePipeline(),
    )

    assert result["source_total_frames"] == 10
    assert result["source_total_frames_known"] is True
    assert result["source_exhausted"] is False
    assert result["truncated_by_max_frames"] is False
    assert result["source_read_error"] is True
    assert result["total_frames"] == 8
