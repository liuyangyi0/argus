"""Offline orchestration regressions; no Torch, video decoder or model is run."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from argus.preprocessing.scene_guard import SceneGuardResult, SceneMatchMetrics, SceneState
from argus.preprocessing.scene_profile import RUNTIME_ARTIFACT_HASH_SCHEME, SceneProfileStore
from argus.validation import scene_guard_comparison as comparison


def _result(state=SceneState.STABLE):
    return SceneGuardResult(state, state, "fixture scene geometry", SceneMatchMetrics())


@pytest.fixture
def experiment(monkeypatch, tmp_path):
    paths = {name: tmp_path / name for name in ("config", "video", "extractor", "matcher")}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    metadata = tmp_path / "metadata.json"
    reference = tmp_path / "reference.jpg"
    active = tmp_path / "active.json"
    metadata.write_bytes(b"metadata")
    reference.write_bytes(b"reference")
    profile_files = [metadata, reference, active]
    profile_hashes = {
        str(path): comparison.file_sha256(path) if path.exists() else None
        for path in profile_files
    }
    policy = SimpleNamespace(
        enabled=True, allow_alignable=False, require_profile_model_binding=True,
        require_operator_approved_profile=True,
        profiles_dir=str(tmp_path / "profiles"),
        model_dump=lambda: {"process_width": 640, "consecutive_matches": 4},
    )
    camera = SimpleNamespace(camera_id="cam", scene_guard=policy)
    config = SimpleNamespace(cameras=[camera], model_dump=lambda **_kw: {"fixture": "config"})
    monkeypatch.setattr(comparison, "load_config", lambda _path: config)
    monkeypatch.setattr(comparison, "_profile_reference", lambda *_args: (
        np.full((64, 64, 3), 100, np.uint8),
        {"source_file_sha256": profile_hashes, "runtime_model_binding_verified": False},
        profile_files,
    ))
    events = []
    decoders = []

    class Decoder:
        def __init__(self, _path, *, expected_sha256):
            self.consumed = 0
            self.index = len(decoders)
            decoders.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            for index in range(4):
                self.consumed += 1
                yield SimpleNamespace(
                    frame_index=index,
                    frame=np.full((64, 64, 3), 100 + index, np.uint8),
                    timestamp=SimpleNamespace(pts=index, pts_time_seconds=index / 30, time_base="1/30"),
                )
            events.append(f"eof-{self.index}")

        def evidence(self):
            assert self.consumed == 4
            return {"timestamp_frame_count": 4, "source_exhausted": True}

    class Baseline:
        def __init__(self, _reference, *, config):
            assert config.process_width == 640
            assert config.consecutive_matches == 4

        def evaluate(self, frame):
            events.append(f"baseline-{int(frame[0, 0, 0])}")
            frame[:] = 0  # Must not alter the decoded source or the other pass.
            return _result()

        def require_reconfirmation(self):
            events.append("baseline-reset")

    class Shadow:
        last_match_metadata = {"device": "fixture-device"}

        def evaluate(self, frame):
            events.append(f"shadow-{int(frame[0, 0, 0])}")
            return _result()

        def require_reconfirmation(self):
            events.append("shadow-reset")

    shadow = Shadow()

    def make_shadow(_reference, config, **_kwargs):
        assert events[-1] == "eof-0"
        assert config.process_width == 640
        events.append("initialize-lightglue")
        return shadow

    monkeypatch.setattr(comparison, "StrictVideoDecoder", Decoder)
    monkeypatch.setattr(comparison, "SceneGuard", Baseline)
    monkeypatch.setattr(comparison, "_make_shadow", make_shadow)
    kwargs = dict(
        config_path=paths["config"], camera_id="cam", video_path=paths["video"],
        profile_id="approved", extractor_weights=paths["extractor"],
        extractor_sha256=comparison.file_sha256(paths["extractor"]),
        matcher_weights=paths["matcher"], matcher_sha256=comparison.file_sha256(paths["matcher"]),
    )
    return SimpleNamespace(
        kwargs=kwargs, paths=paths, events=events, decoders=decoders, shadow=shadow,
        make_shadow=make_shadow, profile_files=profile_files,
    )


def test_two_complete_passes_share_source_frames_without_gate_authority(experiment):
    report = comparison.run_scene_guard_comparison(**experiment.kwargs)
    assert report["status"] == "complete"
    assert [decoder.consumed for decoder in experiment.decoders] == [4, 4]
    assert experiment.events.index("eof-0") < experiment.events.index("initialize-lightglue")
    assert [row["frame_index"] for row in report["frames"]] == [0, 1, 2, 3]
    assert "shadow-100" in experiment.events
    assert report["summary"]["comparable_pairs"] == 4
    assert report["summary"]["state_agreement_fraction"] == 1
    assert report["gating_effect"] == "none"
    assert report["release_authority"] is False
    assert report["production_pipeline_parity_claimed"] is False
    assert report["reference"]["runtime_model_binding_verified"] is False
    assert not any("usable" in row[key] for row in report["frames"] for key in ("baseline", "lightglue"))


def test_prefix_is_partial_but_both_decoders_still_consume_eof(experiment):
    report = comparison.run_scene_guard_comparison(**experiment.kwargs, max_frames=2)
    assert report["status"] == "partial"
    assert report["comparison_truncated"] is True
    assert len(report["frames"]) == 2
    assert [decoder.consumed for decoder in experiment.decoders] == [4, 4]


def test_sampling_retains_exact_original_frame_indices_and_pts(experiment):
    report = comparison.run_scene_guard_comparison(**experiment.kwargs, sample_every=2)
    assert report["status"] == "complete"
    assert [(row["frame_index"], row["pts"]) for row in report["frames"]] == [(0, 0), (2, 2)]
    assert report["frames"][1]["relative_timestamp_sec"] == pytest.approx(2 / 30)


def test_contained_matcher_exception_is_not_counted_as_valid_agreement(experiment):
    experiment.shadow.last_match_metadata = {"error_code": "inference_failed", "error_type": "RuntimeError"}
    report = comparison.run_scene_guard_comparison(**experiment.kwargs)
    assert report["status"] == "error"
    assert report["summary"]["lightglue_errors"] == 4
    assert report["summary"]["comparable_pairs"] == 0
    assert report["summary"]["state_agreement_fraction"] is None
    assert report["frames"][0]["lightglue"]["state"] == "unknown"
    assert report["frames"][0]["lightglue"]["matcher"]["error_code"] == "inference_failed"


def test_normal_unknown_is_evaluated_geometry_not_inference_failure(experiment, monkeypatch):
    monkeypatch.setattr(experiment.shadow, "evaluate", lambda _frame: _result(SceneState.UNKNOWN))
    report = comparison.run_scene_guard_comparison(**experiment.kwargs)
    assert report["status"] == "complete"
    assert report["summary"]["lightglue_errors"] == 0
    assert report["summary"]["disagreement_frame_indices"] == [0, 1, 2, 3]
    assert report["summary"]["accuracy_improvement_established"] is False


def test_dependency_failure_is_preserved_in_error_report(experiment, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise ImportError("optional matcher is missing")

    monkeypatch.setattr(comparison, "_make_shadow", unavailable)
    report = comparison.run_scene_guard_comparison(**experiment.kwargs)
    assert report["status"] == "error"
    assert report["summary"]["baseline_errors"] == 0
    assert report["summary"]["lightglue_errors"] == 4
    assert len(experiment.decoders) == 1
    assert report["errors"][0]["stage"] == "lightglue_initialization"


def test_baseline_reference_feature_failure_does_not_prevent_learned_pass(experiment, monkeypatch):
    def no_features(*_args, **_kwargs):
        raise ValueError("approved reference has insufficient ORB features")

    monkeypatch.setattr(comparison, "SceneGuard", no_features)
    report = comparison.run_scene_guard_comparison(**experiment.kwargs)
    assert report["status"] == "error"
    assert report["summary"]["baseline_errors"] == 4
    assert report["summary"]["lightglue_errors"] == 0
    assert len(experiment.decoders) == 2


@pytest.mark.parametrize("change", ["delete", "directory", "replace", "create-active"])
def test_source_changes_never_discard_completed_diagnostic_rows(experiment, monkeypatch, change):
    def changed(*args, **kwargs):
        result = experiment.make_shadow(*args, **kwargs)
        path = experiment.paths["matcher"]
        if change in {"delete", "directory"}:
            path.unlink()
            if change == "directory":
                path.mkdir()
        elif change == "replace":
            path.write_bytes(b"changed weights")
        else:
            experiment.profile_files[-1].write_bytes(b"new active pointer")
        return result

    monkeypatch.setattr(comparison, "_make_shadow", changed)
    report = comparison.run_scene_guard_comparison(**experiment.kwargs)
    assert report["status"] == "partial"
    assert report["source_inputs_stable"] is False
    assert report["summary"]["comparable_pairs"] == 4
    assert any(error["stage"] == "input_identity" for error in report["errors"])


def _report(source):
    return {
        "schema_version": comparison.COMPARISON_SCHEMA, "diagnostic_only": True,
        "gating_effect": "none", "release_authority": False, "protected_paths": [str(source)],
        "protected_directories": [str(source.parent / "profiles")],
    }


def test_report_writer_never_replaces_profile_source_even_with_overwrite(tmp_path):
    source = tmp_path / "metadata.json"
    source.write_bytes(b"approved profile")
    with pytest.raises(ValueError, match="source input"):
        comparison.write_comparison_report(_report(source), source, overwrite=True)
    assert source.read_bytes() == b"approved profile"


def test_report_writer_rejects_hardlink_to_profile_source(tmp_path):
    source, output = tmp_path / "reference.jpg", tmp_path / "report.json"
    source.write_bytes(b"approved reference")
    try:
        os.link(source, output)
    except OSError:
        pytest.skip("filesystem does not support hardlinks")
    with pytest.raises(ValueError, match="hard link"):
        comparison.write_comparison_report(_report(source), output, overwrite=True)
    assert source.read_bytes() == output.read_bytes() == b"approved reference"


def test_report_writer_is_no_clobber_by_default(tmp_path):
    source, output = tmp_path / "config.yaml", tmp_path / "report.json"
    source.write_bytes(b"config")
    output.write_bytes(b"previous report")
    with pytest.raises(FileExistsError):
        comparison.write_comparison_report(_report(source), output)
    assert output.read_bytes() == b"previous report"


@pytest.fixture
def approved_profile(tmp_path):
    baseline = tmp_path / "baseline.png"
    encoded, content = cv2.imencode(".png", np.full((64, 64, 3), 100, np.uint8))
    assert encoded
    baseline.write_bytes(content.tobytes())
    store = SceneProfileStore(tmp_path / "profiles")
    profile = store.create_from_baseline("cam", "approved", baseline, metadata={
        "operator_approved": True, "validation_only": False,
        "model_version_id": "version-1", "model_artifact_sha256": "a" * 64,
        "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
    })
    camera = SimpleNamespace(camera_id="cam", scene_guard=SimpleNamespace(profiles_dir=str(store.root)))
    return camera, profile


def test_profile_binding_is_declared_only_and_pointer_is_not_activated(approved_profile):
    camera, profile = approved_profile
    frame, info, files = comparison._profile_reference(camera, "approved")
    assert frame.shape == (64, 64, 3)
    assert info["operator_approved"] is True
    assert info["runtime_model_binding_verified"] is False
    assert info["selected_profile_is_active"] is False
    assert not files[-1].exists()
    assert info["source_file_sha256"][str(profile.reference_path)] == hashlib.sha256(
        profile.reference_path.read_bytes()
    ).hexdigest()


def test_approval_revoked_during_profile_load_cannot_be_reported_stable(approved_profile, monkeypatch):
    camera, profile = approved_profile
    original = SceneProfileStore.load

    def revoke(store, *args):
        result = original(store, *args)
        path = profile.reference_path.parent / "metadata.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metadata"]["operator_approved"] = False
        path.write_text(json.dumps(payload), encoding="utf-8")
        return result

    monkeypatch.setattr(SceneProfileStore, "load", revoke)
    with pytest.raises(ValueError, match="changed while loading"):
        comparison._profile_reference(camera, "approved")
