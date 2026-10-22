"""Pure report-contract tests; no decoder, Torch or FiftyOne is required."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from argus.integrations import scene_comparison
from argus.integrations.contracts import (
    FrameTimestamp,
    ReviewBundle,
    ReviewClip,
    file_sha256,
)
from argus.integrations.scene_comparison import (
    load_scene_comparisons,
    require_verified_scene_comparisons,
)


def _decision(state: str = "stable", *, raw_state: str | None = None) -> dict:
    return {
        "status": "evaluated", "state": state, "raw_state": raw_state or state,
        "reason": "diagnostic pose estimate", "metrics": {"inliers": 25},
        "elapsed_ms": 2.0, "matches_stable_pose": state == "stable",
    }


def _failed() -> dict:
    return {
        "status": "error", "state": "unknown", "raw_state": "unknown",
        "reason": "optional matcher unavailable", "metrics": None,
        "elapsed_ms": None, "matches_stable_pose": False,
        "matcher": {"error_code": "matcher_unavailable"},
    }


def _evidence(clip: ReviewClip) -> dict:
    return {
        "schema_version": "argus.strict-video-decoder-evidence.v1",
        "backend": "ffprobe+ffmpeg-rawvideo", "strict_release_decoder": True,
        "source_sha256_before": clip.video_sha256,
        "source_sha256_after": clip.video_sha256, "source_sha256_matches": True,
        "stream_index": 0, "width": clip.width, "height": clip.height,
        "pixel_format": "bgr24", "time_base": "1/10",
        "timestamp_frame_count": clip.frame_count, "pts_strictly_increasing": True,
        "nominal_frame_rate": clip.fps,
        "container_declared_frame_count": clip.frame_count,
        "container_declared_frame_count_matches": True,
        "ffprobe": {
            "returncode": 0, "stderr_empty": True,
            "decoded_frame_inventory_count": clip.frame_count,
        },
        "ffmpeg": {
            "returncode": 0, "stderr_empty": True, "decoded_frame_count": clip.frame_count,
            "decoded_frame_count_matches_ffprobe": True, "partial_raw_frame_detected": False,
            "clean_eof": True, "xerror_enabled": True,
        },
        "decode_error_distinguishable_from_clean_eof": True,
        "variable_frame_rate_timestamps_preserved": True,
        "complete_decode_verified": True, "source_exhausted": True,
        "truncated": False,
    }


@pytest.fixture
def bundle(tmp_path: Path) -> ReviewBundle:
    video = tmp_path / "relocated-video.mp4"
    video.write_bytes(b"a content-bound source, no decoding in these contract tests")
    clip = ReviewClip(
        clip_id="clip-one", camera_id="camera-one", video_path=str(video),
        video_sha256=file_sha256(video), fps=5.0, frame_count=4, width=640, height=480,
        timestamps=[FrameTimestamp(
            frame_index=index, pts=10 + index * 2, time_base_num=1, time_base_den=10,
            timestamp_sec=(10 + index * 2) / 10,
        ) for index in range(4)],
    )
    return ReviewBundle(
        bundle_id="bundle-one", suite_id="suite-one", manifest_sha256="a" * 64,
        clips=[clip],
    )


@pytest.fixture
def payload(bundle: ReviewBundle) -> dict:
    clip = bundle.clips[0]
    return {
        "schema_version": "argus.scene-guard-comparison.v1", "status": "complete",
        "diagnostic_only": True, "gating_effect": "none", "release_authority": False,
        "source_inputs_stable": True, "production_pipeline_parity_claimed": False,
        "baseline_scope": "standalone_scene_guard_full_check",
        "reference": {
            "camera_id": clip.camera_id, "profile_id": "approved-profile",
            "reference_sha256": "b" * 64, "operator_approved": True,
            "model_version_id_declared": "model-one",
            "runtime_model_binding_verified": False,
        },
        "inputs": {
            "video": {"path": "/original-machine/recording.mp4", "sha256": clip.video_sha256},
            "config": {"path": "/unavailable/config.yaml", "sha256": "c" * 64},
            "extractor_weights": {"path": "/unavailable/aliked.pth", "sha256": "d" * 64},
            "matcher_weights": {"path": "/unavailable/lightglue.pth", "sha256": "e" * 64},
        },
        "effective_config_sha256": "f" * 64,
        "decoder_evidence": {"baseline": _evidence(clip), "lightglue": _evidence(clip)},
        "sample_every": 1, "max_frames": None, "comparison_truncated": False,
        "errors": [],
        "summary": {"disagreement_frame_indices": [], "comparable_pairs": 9999},
        "algorithm_thresholds": {"feature_type": "ORB", "consecutive_matches": 3},
        "lightglue_options": {"extractor": "ALIKED", "device": "cuda:0"},
        "metric_semantics": {"raw_match_counts_comparable": False},
        "frames": [{
            "frame_index": index, "pts": stamp.pts, "time_base": "1/10",
            "timestamp_sec": stamp.timestamp_sec, "relative_timestamp_sec": index / 5,
            "frame_sha256": str(index) * 64, "baseline": _decision(), "lightglue": _decision(),
        } for index, stamp in enumerate(clip.timestamps)],
    }


def _write(tmp_path: Path, payload: dict, name: str = "comparison.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return path


def test_recomputes_disagreements_and_keeps_diagnostics_not_ground_truth(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload["frames"][1]["lightglue"] = _decision("moved")
    payload["frames"][2]["baseline"] = _decision("calibrating", raw_state="stable")
    payload["frames"][2]["lightglue"] = _decision("calibrating", raw_state="alignable")
    payload["frames"][1]["state_disagreement"] = False
    reports = load_scene_comparisons(bundle, [_write(tmp_path, payload)])
    review = reports["clip-one"]
    assert review.frame_rows[1]["review_category"] == "confirmed_disagreement"
    assert review.frame_rows[1]["state_disagreement"] is True
    assert review.frame_rows[2]["review_category"] == "raw_disagreement"
    assert review.summary["state_disagreement_count"] == 1
    assert review.summary["raw_state_disagreement_count"] == 2
    assert review.summary["comparable_pairs"] == 4
    assert "disagreement_frame_indices" not in review.summary
    assert review.summary["pose_ground_truth_available"] is False
    assert review.summary["gating_effect"] == "none"
    assert "ground_truth" not in review.frame_rows[1]
    require_verified_scene_comparisons(reports, bundle=bundle)


def test_missing_weights_profiles_and_relocated_video_path_are_not_loaded(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    report = load_scene_comparisons(bundle, [_write(tmp_path, payload)])["clip-one"]
    assert report.profile_id == "approved-profile"
    assert report.summary["inputs"]["video"]["path"] != bundle.clips[0].video_path


def test_partial_prefix_uses_full_source_eof_and_exact_sampling_grid(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload.update(status="partial", sample_every=2, max_frames=3, comparison_truncated=True)
    payload["frames"] = [payload["frames"][0], payload["frames"][2]]
    review = load_scene_comparisons(bundle, [_write(tmp_path, payload)])["clip-one"]
    assert [row["frame_index"] for row in review.frame_rows] == [0, 2]
    assert review.summary["comparison_truncated"] is True
    assert review.report_status == "partial"


def test_sparse_vfr_frames_keep_negative_pts_origin_and_exact_relative_time(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    # Frame 2 is 75 ms after the original first frame, not 2 / nominal_fps.
    # Its index must remain 2 rather than becoming sampled-row index 1.
    pts = [-90000, -88999, -87750, -86000]
    timestamps = [FrameTimestamp(
        frame_index=index, pts=value, time_base_num=1, time_base_den=30000,
        timestamp_sec=float(Fraction(value, 30000)),
    ) for index, value in enumerate(pts)]
    clip = bundle.clips[0].model_copy(update={"timestamps": timestamps, "fps": 30.0})
    bundle = bundle.model_copy(update={"clips": [clip]})
    for evidence in payload["decoder_evidence"].values():
        evidence.update(time_base="1/30000", nominal_frame_rate=30.0)
    for row, stamp in zip(payload["frames"], timestamps):
        row.update(
            pts=stamp.pts, time_base="1/30000", timestamp_sec=stamp.timestamp_sec,
            relative_timestamp_sec=float(Fraction(stamp.pts - pts[0], 30000)),
        )
    payload["sample_every"] = 2
    payload["frames"] = payload["frames"][::2]
    reviews = load_scene_comparisons(bundle, [_write(tmp_path, payload)])
    rows = reviews[clip.clip_id].frame_rows
    assert [row["frame_index"] for row in rows] == [0, 2]
    assert rows[0]["pts"] == -90000
    assert rows[0]["timestamp_sec"] == -3.0
    assert rows[1]["pts"] == -87750
    assert rows[1]["relative_timestamp_sec"] == 0.075
    assert rows[1]["relative_timestamp_sec"] != 2 / clip.fps
    require_verified_scene_comparisons(reviews, bundle=bundle)


def test_matcher_failure_is_error_not_agreement_or_normal_disagreement(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload["status"] = "error"
    payload["errors"] = [{"stage": "lightglue_initialization", "error": "unavailable"}]
    payload["decoder_evidence"].pop("lightglue")
    for row in payload["frames"]:
        row["lightglue"] = _failed()
    review = load_scene_comparisons(bundle, [_write(tmp_path, payload)])["clip-one"]
    assert review.summary["comparable_pairs"] == 0
    assert review.summary["error_frames"] == 4
    assert review.summary["state_disagreement_count"] == 0
    for row in review.frame_rows:
        assert row["comparison_error"] is True
        assert row["state_disagreement"] is None
        assert row["raw_state_disagreement"] is None
        assert row["error_algorithms"] == ["lightglue"]
        assert row["review_category"] == "error"


@pytest.mark.parametrize("key,value", [
    ("schema_version", "other-schema"), ("diagnostic_only", False),
    ("gating_effect", "allow"), ("release_authority", True),
    ("source_inputs_stable", False), ("production_pipeline_parity_claimed", True),
    ("baseline_scope", "production_runtime"), ("status", "unknown"),
])
def test_rejects_authority_or_unstable_source_claims(
    tmp_path: Path, bundle: ReviewBundle, payload: dict, key: str, value: object,
) -> None:
    payload[key] = value
    with pytest.raises(ValueError):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


@pytest.mark.parametrize("key,value", [
    ("width", 641), ("height", 481), ("timestamp_frame_count", 3),
    ("source_sha256_after", "9" * 64), ("source_sha256_matches", False),
    ("complete_decode_verified", False), ("source_exhausted", False),
    ("truncated", True), ("time_base", "1/5"), ("nominal_frame_rate", 6.0),
])
def test_rejects_unproven_decoder_identity(
    tmp_path: Path, bundle: ReviewBundle, payload: dict, key: str, value: object,
) -> None:
    payload["decoder_evidence"]["baseline"][key] = value
    with pytest.raises(ValueError):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "reordered", "pts", "time_base", "timestamp",
    "relative_time", "off_grid",
])
def test_rejects_frame_identity_and_sampling_drift(
    tmp_path: Path, bundle: ReviewBundle, payload: dict, mutation: str,
) -> None:
    rows = payload["frames"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "pts":
        rows[1]["pts"] += 1
    elif mutation == "time_base":
        rows[1]["time_base"] = "2/20"
    elif mutation == "timestamp":
        rows[1]["timestamp_sec"] += 0.01
    elif mutation == "relative_time":
        rows[1]["relative_timestamp_sec"] += 0.01
    else:
        payload["sample_every"] = 2
        payload["frames"] = [rows[0], rows[1]]
    with pytest.raises(ValueError):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


def test_rejects_missing_baseline_eof_even_for_partial_report(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload["status"] = "partial"
    payload["decoder_evidence"].pop("baseline")
    with pytest.raises(ValueError):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


def test_rejects_complete_comparison_without_second_clean_eof(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload["decoder_evidence"].pop("lightglue")
    with pytest.raises(ValueError):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


def test_rejects_error_row_claiming_stable_pose(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload["status"] = "partial"
    payload["frames"][0]["lightglue"] = _failed()
    payload["frames"][0]["lightglue"]["state"] = "stable"
    with pytest.raises(ValueError):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


def test_matcher_error_code_cannot_be_hidden_in_an_evaluated_state(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload["frames"][0]["lightglue"]["matcher"] = {"error_code": "matcher_failed"}
    with pytest.raises(ValueError, match="matcher failures"):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


@pytest.mark.parametrize("binding,value", [
    ("scene_profile_id", "different-profile"), ("model_version_id", "different-model"),
])
def test_explicit_profile_or_model_binding_cannot_be_silently_mismatched(
    tmp_path: Path, bundle: ReviewBundle, payload: dict, binding: str, value: str,
) -> None:
    clip = bundle.clips[0].model_copy(update={binding: value})
    bound_bundle = bundle.model_copy(update={"clips": [clip]})
    with pytest.raises(ValueError, match="explicit .* binding"):
        load_scene_comparisons(bound_bundle, [_write(tmp_path, payload)])


def test_rejects_ambiguous_clip_binding_and_duplicate_reports(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    report = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="multiple"):
        load_scene_comparisons(bundle, [report, report])
    duplicate_clip = bundle.clips[0].model_copy(update={"clip_id": "another-clip"})
    ambiguous_bundle = bundle.model_copy(update={"clips": [bundle.clips[0], duplicate_clip]})
    with pytest.raises(ValueError, match="exactly one"):
        load_scene_comparisons(ambiguous_bundle, [report])
    payload["reference"]["camera_id"] = "wrong-camera"
    with pytest.raises(ValueError, match="exactly one"):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


def test_report_change_and_nested_snapshot_changes_are_detected(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    path = _write(tmp_path, payload)
    reports = load_scene_comparisons(bundle, [path])
    reports["clip-one"].frame_rows[0]["baseline"]["reason"] = "changed in memory"
    with pytest.raises(ValueError, match="modified"):
        require_verified_scene_comparisons(reports, bundle=bundle)
    reports = load_scene_comparisons(bundle, [path])
    payload["frames"][0]["baseline"]["reason"] = "changed on disk"
    _write(tmp_path, payload)
    with pytest.raises(ValueError, match="changed after preflight"):
        require_verified_scene_comparisons(reports, bundle=bundle)


def test_hand_constructed_review_cannot_attach_authentic_report_hash_to_fake_rows(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    reports = load_scene_comparisons(bundle, [_write(tmp_path, payload)])
    review = reports["clip-one"]
    forged = copy.deepcopy(review.frame_rows[0])
    forged["state_disagreement"] = True
    reports["clip-one"] = replace(review, frame_rows=(forged, *review.frame_rows[1:]))
    with pytest.raises(ValueError, match="modified"):
        require_verified_scene_comparisons(reports, bundle=bundle)


@pytest.mark.parametrize("keys,replacement", [
    (("frame_index",), 0.0),
    (("pts",), 10.0),
    (("timestamp_sec",), 1),
    (("comparable",), 1),
    (("comparison_error",), 0),
    (("state_disagreement",), 0),
    (("baseline", "elapsed_ms"), 2),
    (("baseline", "matches_stable_pose"), 1),
    (("baseline", "metrics", "inliers"), 25.0),
])
def test_snapshot_recheck_rejects_equal_valued_json_type_changes(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
    keys: tuple[str, ...], replacement: object,
) -> None:
    reports = load_scene_comparisons(bundle, [_write(tmp_path, payload)])
    target = reports["clip-one"].frame_rows[0]
    for key in keys[:-1]:
        target = target[key]
    # These replacements compare equal in Python despite changing the
    # hash-bound validated JSON types. Numeric/bool coercion is not a review.
    assert target[keys[-1]] == replacement
    target[keys[-1]] = replacement
    with pytest.raises(ValueError, match="modified"):
        require_verified_scene_comparisons(reports, bundle=bundle)


def test_snapshot_recheck_also_protects_derived_summary_json_types(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    reports = load_scene_comparisons(bundle, [_write(tmp_path, payload)])
    reports["clip-one"].summary["source_inputs_stable"] = 1
    with pytest.raises(ValueError, match="modified"):
        require_verified_scene_comparisons(reports, bundle=bundle)


def test_bound_review_cannot_be_reused_for_another_bundle_clip(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    reviews = load_scene_comparisons(bundle, [_write(tmp_path, payload)])
    other_clip = bundle.clips[0].model_copy(update={"camera_id": "different-camera"})
    other_bundle = bundle.model_copy(update={"clips": [other_clip]})
    with pytest.raises(ValueError, match="different review bundle"):
        require_verified_scene_comparisons(reviews, bundle=other_bundle)
    with pytest.raises(ValueError, match="mapping key"):
        require_verified_scene_comparisons({"other-id": reviews["clip-one"]}, bundle=bundle)


def test_rejects_duplicate_json_keys_and_nonfinite_numeric_overflow(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    path = tmp_path / "comparison.json"
    path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_scene_comparisons(bundle, [path])
    raw = json.dumps(payload).replace('"elapsed_ms": 2.0', '"elapsed_ms": 1e999', 1)
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_scene_comparisons(bundle, [path])


def test_bounded_rows_and_summary_prevent_unbounded_sample_documents(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload["frames"][0]["lightglue"]["matcher"] = {"oversized": "x" * (129 * 1024)}
    with pytest.raises(ValueError, match="frame diagnostics exceed"):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])
    payload["frames"][0]["lightglue"].pop("matcher")
    payload["algorithm_thresholds"] = {"oversized": "x" * (65 * 1024)}
    with pytest.raises(ValueError, match="summary diagnostics exceed"):
        load_scene_comparisons(bundle, [_write(tmp_path, payload)])


def test_report_error_list_is_compacted_with_explicit_truncation(
    tmp_path: Path, bundle: ReviewBundle, payload: dict,
) -> None:
    payload["status"] = "partial"
    payload["errors"] = [{"stage": "diagnostic", "error": "x" * 2048} for _ in range(100)]
    summary = load_scene_comparisons(bundle, [_write(tmp_path, payload)])["clip-one"].summary
    assert summary["report_error_count"] == 100
    assert summary["report_errors_truncated"] is True
    assert len(summary["report_errors"]) == 16
    assert all(len(error["error"]) == 1024 for error in summary["report_errors"])


def test_detects_report_bytes_change_during_load(
    tmp_path: Path, bundle: ReviewBundle, payload: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path, payload)
    original = file_sha256(path)
    calls = 0

    def changed_hash(_path: Path) -> str:
        nonlocal calls
        calls += 1
        return original if calls == 1 else "0" * 64

    monkeypatch.setattr(scene_comparison, "file_sha256", changed_hash)
    with pytest.raises(ValueError, match="changed while loading"):
        load_scene_comparisons(bundle, [path])
