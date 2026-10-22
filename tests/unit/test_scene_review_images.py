"""Controlled VFR image contract tests using fake frames/decoder only."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import zlib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus.integrations import scene_review_images as images
from argus.integrations.contracts import FrameTimestamp, ReviewBundle, ReviewClip, file_sha256
from argus.integrations.scene_comparison import load_scene_comparisons


def _decision(state: str = "stable", *, raw: str | None = None) -> dict:
    return {
        "status": "evaluated", "state": state, "raw_state": raw or state,
        "reason": "diagnostic only", "metrics": {"inliers": 12},
        "elapsed_ms": 1.0, "matches_stable_pose": state == "stable",
    }


def _tensor_digest(raw: bytes, width: int = 2, height: int = 1) -> str:
    digest = hashlib.sha256()
    digest.update(b"uint8")
    digest.update(json.dumps([height, width, 3]).encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _evidence(clip: ReviewClip) -> dict:
    return {
        "schema_version": "argus.strict-video-decoder-evidence.v1",
        "backend": "ffprobe+ffmpeg-rawvideo", "strict_release_decoder": True,
        "source_sha256_before": clip.video_sha256, "source_sha256_after": clip.video_sha256,
        "source_sha256_matches": True, "stream_index": 0, "width": clip.width,
        "height": clip.height, "pixel_format": "bgr24", "time_base": "1/30000",
        "timestamp_frame_count": clip.frame_count, "pts_strictly_increasing": True,
        "variable_frame_rate_detected": True, "distinct_pts_deltas": [1001, 1249, 1750],
        "nominal_frame_rate": clip.fps, "container_declared_frame_count": clip.frame_count,
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
        "variable_frame_rate_timestamps_preserved": True, "complete_decode_verified": True,
        "source_exhausted": True, "truncated": False, "sufficient_for_model_release": True,
    }


class _Frame:
    dtype = "uint8"
    shape = (1, 2, 3)

    def __init__(self, raw: bytes):
        self.raw = raw

    def tobytes(self, *, order: str) -> bytes:
        assert order == "C"
        return self.raw


class _Decoder:
    def __init__(self, clip: ReviewClip, frames: list[bytes], consumed: list[int]):
        self.clip = clip
        self.frames = frames
        self.consumed = consumed
        self.complete = False
        self.bad_pts = False
        self.fail_eof = False
        self.timestamp_changes: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            assert self.complete, "extractor must exhaust the original source"
        return False

    def __iter__(self):
        for index, raw in enumerate(self.frames):
            stamp = self.clip.timestamps[index]
            self.consumed.append(index)
            timestamp = SimpleNamespace(
                frame_index=index, pts=stamp.pts + int(self.bad_pts and index == 2),
                time_base="1/30000", pts_time_seconds=float(Fraction(stamp.pts, 30000)),
            )
            if index == 0:
                for name, value in self.timestamp_changes.items():
                    setattr(timestamp, name, value)
            yield SimpleNamespace(
                frame_index=index, frame=_Frame(raw), timestamp=timestamp,
            )
        if self.fail_eof:
            raise ValueError("injected strict decoder EOF failure")
        self.complete = True

    def evidence(self):
        assert self.complete
        return _evidence(self.clip)


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video = tmp_path / "original-vfr.mp4"
    video.write_bytes(b"source bytes are never modified or decoded in this test")
    pts = [-90000, -88999, -87750, -86000]
    clip = ReviewClip(
        clip_id="clip/unsafe-path", camera_id="camera-one", video_path=str(video),
        video_sha256=file_sha256(video), fps=30.0, frame_count=4, width=2, height=1,
        timestamps=[FrameTimestamp(
            frame_index=index, pts=value, time_base_num=1, time_base_den=30000,
            timestamp_sec=float(Fraction(value, 30000)),
        ) for index, value in enumerate(pts)],
    )
    bundle = ReviewBundle(
        bundle_id="bundle-one", suite_id="suite-one", manifest_sha256="a" * 64, clips=[clip],
    )
    frames = [bytes([index, 30, 200, 44, 99, 100 + index]) for index in range(4)]
    profile_root = tmp_path / "protected-profiles"
    profile_root.mkdir()
    report_path = tmp_path / "comparison.json"
    payload = {
        "schema_version": "argus.scene-guard-comparison.v1", "status": "complete",
        "diagnostic_only": True, "gating_effect": "none", "release_authority": False,
        "source_inputs_stable": True, "production_pipeline_parity_claimed": False,
        "baseline_scope": "standalone_scene_guard_full_check",
        "reference": {
            "camera_id": clip.camera_id, "profile_id": "approved-profile",
            "reference_sha256": "b" * 64, "operator_approved": True,
            "runtime_model_binding_verified": False,
        },
        "inputs": {
            "video": {"path": str(video), "sha256": clip.video_sha256},
            "config": {"path": str(tmp_path / "config.yaml"), "sha256": "c" * 64},
            "extractor_weights": {"path": "missing/aliked.pth", "sha256": "d" * 64},
            "matcher_weights": {"path": "missing/lightglue.pth", "sha256": "e" * 64},
        },
        "effective_config_sha256": "f" * 64,
        "protected_paths": [str(video), str(report_path)],
        "protected_directories": [str(profile_root)],
        "decoder_evidence": {"baseline": _evidence(clip), "lightglue": _evidence(clip)},
        "sample_every": 1, "max_frames": None, "comparison_truncated": False,
        "errors": [], "algorithm_thresholds": {}, "lightglue_options": {},
        "metric_semantics": {},
        "frames": [{
            "frame_index": index, "pts": stamp.pts, "time_base": "1/30000",
            "timestamp_sec": stamp.timestamp_sec,
            "relative_timestamp_sec": float(Fraction(stamp.pts - pts[0], 30000)),
            "frame_sha256": _tensor_digest(raw), "baseline": _decision(),
            "lightglue": _decision("moved") if index == 2 else _decision(),
        } for index, (stamp, raw) in enumerate(zip(clip.timestamps, frames))],
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    reviews = load_scene_comparisons(bundle, [report_path])
    consumed: list[int] = []
    decoder = _Decoder(clip, frames, consumed)
    monkeypatch.setattr(images, "_decoder", lambda source, digest: decoder)
    return SimpleNamespace(
        bundle=bundle, clip=clip, reviews=reviews, frames=frames, decoder=decoder,
        consumed=consumed, report=report_path, payload=payload, video=video,
        profile_root=profile_root, output=tmp_path / "new-image-review",
    )


def _prepare(case, **kwargs) -> Path:
    return images.prepare_scene_review_images(
        case.bundle, scene_comparisons=case.reviews, output_dir=case.output, **kwargs,
    )


def _load(case, manifest: Path) -> dict:
    return images.load_scene_review_images(
        manifest, bundle=case.bundle, scene_comparisons=case.reviews,
    )


def _rewrite_manifest(path: Path, payload: dict) -> None:
    payload["artifact_id"] = hashlib.sha256(images._json_bytes({
        key: value for key, value in payload.items() if key != "artifact_id"
    })).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_vfr_frame_pixels_index_pts_and_summary_are_source_bound(case) -> None:
    video_sha, report_sha = file_sha256(case.video), file_sha256(case.report)
    manifest = _prepare(case)
    assert case.consumed == [0, 1, 2, 3]
    data = _load(case, manifest)
    row = data["images"][0]
    assert row["frame_index"] == 2 and row["frame_number"] == 3
    assert row["pts_exact"] == "-87750" and row["time_base_exact"] == "1/30000"
    assert row["relative_timestamp_sec"] == 0.075 != 2 / case.clip.fps
    assert data["reports"][case.clip.clip_id]["summary"] == case.reviews[case.clip.clip_id].summary
    png = (manifest.parent / row["image_path"]).read_bytes()
    assert images._png_bgr(png, width=2, height=1) == case.frames[2]
    assert data["diagnostic_only"] is True and data["gating_effect"] == "none"
    assert data["release_authority"] is False and data["ground_truth_authority"] is False
    assert file_sha256(case.video) == video_sha and file_sha256(case.report) == report_sha
    assert not (manifest.parent / ".argus-incomplete").exists()


def test_all_compared_preserves_order_without_clip_id_path_injection(case) -> None:
    data = _load(case, _prepare(case, selection="all-compared"))
    assert [row["frame_index"] for row in data["images"]] == [0, 1, 2, 3]
    for row in data["images"]:
        assert row["image_path"].startswith("images/")
        assert "clip/unsafe-path" not in row["image_path"]
        assert row["report_sha256"] == case.reviews[case.clip.clip_id].report_sha256


def test_sparse_comparison_preserves_original_index_and_decodes_all_source_frames(case) -> None:
    case.payload["sample_every"] = 2
    case.payload["frames"] = case.payload["frames"][::2]
    case.report.write_text(json.dumps(case.payload), encoding="utf-8")
    case.reviews = load_scene_comparisons(case.bundle, [case.report])
    data = _load(case, _prepare(case))
    assert case.consumed == [0, 1, 2, 3]
    assert [row["frame_index"] for row in data["images"]] == [2]
    assert data["images"][0]["frame_number"] == 3
    assert data["images"][0]["pts_exact"] == "-87750"


def test_prefix_sparse_comparison_still_verifies_full_source_eof(case) -> None:
    case.payload.update(
        status="partial", sample_every=2, max_frames=3, comparison_truncated=True,
    )
    case.payload["frames"] = case.payload["frames"][:3:2]
    case.report.write_text(json.dumps(case.payload), encoding="utf-8")
    case.reviews = load_scene_comparisons(case.bundle, [case.report])

    data = _load(case, _prepare(case))

    assert case.consumed == [0, 1, 2, 3]
    assert [row["frame_index"] for row in data["images"]] == [2]
    summary = data["reports"][case.clip.clip_id]["summary"]
    assert summary["comparison_truncated"] is True
    assert summary["max_frames"] == 3 and summary["sampled_frames"] == 2
    assert data["decoder_evidence"][case.clip.clip_id]["ffmpeg"]["decoded_frame_count"] == 4


def test_attached_clip_without_selected_images_still_reaches_full_eof(case, monkeypatch) -> None:
    other = case.clip.model_copy(update={"clip_id": "other-clip", "camera_id": "camera-two"})
    case.bundle = case.bundle.model_copy(update={"clips": [case.clip, other]})
    other_report = case.report.with_name("other-comparison.json")
    payload = copy.deepcopy(case.payload)
    payload["reference"]["camera_id"] = other.camera_id
    payload["protected_paths"].append(str(other_report))
    for row in payload["frames"]:
        row["lightglue"] = _decision()  # No selected disagreements for this clip.
    other_report.write_text(json.dumps(payload), encoding="utf-8")
    case.reviews = load_scene_comparisons(case.bundle, [case.report, other_report])
    other_consumed: list[int] = []
    other_decoder = _Decoder(other, case.frames, other_consumed)
    decoders = iter([case.decoder, other_decoder])
    monkeypatch.setattr(images, "_decoder", lambda source, digest: next(decoders))

    data = _load(case, _prepare(case))

    assert case.consumed == other_consumed == [0, 1, 2, 3]
    assert {row["clip_id"] for row in data["images"]} == {case.clip.clip_id}
    assert set(data["reports"]) == {case.clip.clip_id, other.clip_id}
    assert data["decoder_evidence"][other.clip_id]["ffmpeg"]["clean_eof"] is True


def test_default_selection_includes_raw_only_disagreements_and_algorithm_errors(case) -> None:
    case.payload["status"] = "partial"
    case.payload["frames"][1]["baseline"] = _decision("calibrating", raw="stable")
    case.payload["frames"][1]["lightglue"] = _decision("calibrating", raw="alignable")
    case.payload["frames"][3]["lightglue"] = {
        "status": "error", "state": "unknown", "raw_state": "unknown",
        "reason": "matcher failed", "metrics": None, "elapsed_ms": None,
        "matches_stable_pose": False, "matcher": {"error_code": "matcher_failed"},
    }
    case.report.write_text(json.dumps(case.payload), encoding="utf-8")
    case.reviews = load_scene_comparisons(case.bundle, [case.report])
    rows = _load(case, _prepare(case))["images"]
    assert [row["frame_index"] for row in rows] == [1, 2, 3]
    assert [row["comparison"]["review_category"] for row in rows] == [
        "raw_disagreement", "confirmed_disagreement", "error",
    ]
    assert rows[0]["comparison"]["state_disagreement"] is False
    assert rows[0]["comparison"]["raw_state_disagreement"] is True
    assert rows[2]["comparison"]["comparison_error"] is True
    assert rows[2]["comparison"]["state_disagreement"] is None


def test_large_pts_strings_and_fraction_relative_times_do_not_lose_integer_identity(case) -> None:
    values = [2**60, 2**60 + 1, 2**60 + 3, 2**60 + 8]
    stamps = [FrameTimestamp(
        frame_index=index, pts=value, time_base_num=1, time_base_den=30000,
        timestamp_sec=float(Fraction(value, 30000)),
    ) for index, value in enumerate(values)]
    clip = case.clip.model_copy(update={"timestamps": stamps})
    case.bundle = case.bundle.model_copy(update={"clips": [clip]})
    case.clip = clip
    case.decoder.clip = clip
    for row, stamp in zip(case.payload["frames"], stamps):
        row.update(
            pts=stamp.pts, timestamp_sec=float(Fraction(stamp.pts, 30000)),
            relative_timestamp_sec=float(Fraction(stamp.pts - values[0], 30000)),
        )
    case.report.write_text(json.dumps(case.payload), encoding="utf-8")
    case.reviews = load_scene_comparisons(case.bundle, [case.report])
    row = _load(case, _prepare(case))["images"][0]
    assert row["pts_exact"] == str(2**60 + 3)
    assert row["relative_timestamp_sec"] == float(Fraction(3, 30000))
    assert row["timestamp_sec"] - stamps[0].timestamp_sec == 0.0
    assert row["relative_timestamp_sec"] > 0


def test_image_loader_never_constructs_a_decoder(case, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _prepare(case)

    def forbidden(*args, **kwargs):
        raise AssertionError("artifact loader must not decode source video")

    monkeypatch.setattr(images, "_decoder", forbidden)
    assert len(_load(case, manifest)["images"]) == 1


@pytest.mark.parametrize("failure", ["tensor", "pts", "eof"])
def test_extraction_failure_remains_unimportable(case, failure: str) -> None:
    if failure == "tensor":
        case.decoder.frames[2] = b"\x00" * 6
    elif failure == "pts":
        case.decoder.bad_pts = True
    else:
        case.decoder.fail_eof = True
    with pytest.raises(ValueError):
        _prepare(case)
    assert (case.output / ".argus-incomplete").is_file()
    assert not (case.output / "manifest.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        _load(case, case.output / "manifest.json")


@pytest.mark.parametrize("field,value", [
    ("frame_index", False), ("pts", -89999),
    ("time_base", "2/60000"), ("pts_time_seconds", -2.999),
])
def test_timestamp_corruption_in_unselected_frame_blocks_publication(case, field, value) -> None:
    # The default selection only exports frame 2, but frame 0 still binds the
    # complete original timestamp sequence and cannot be skipped during checks.
    case.decoder.timestamp_changes[field] = value
    with pytest.raises(ValueError, match="PTS/time base"):
        _prepare(case)
    assert (case.output / ".argus-incomplete").is_file()
    assert not (case.output / "manifest.json").exists()


@pytest.mark.parametrize("source", ["video", "report"])
def test_input_changed_after_decoder_eof_cannot_publish_manifest(case, monkeypatch, source) -> None:
    evidence = case.decoder.evidence

    def change_after_eof():
        proof = evidence()
        if source == "video":
            case.video.write_bytes(b"injected source change after strict decoding")
        else:
            case.payload["frames"][2]["lightglue"]["reason"] = "injected report refresh"
            case.report.write_text(json.dumps(case.payload), encoding="utf-8")
        return proof

    monkeypatch.setattr(case.decoder, "evidence", change_after_eof)
    with pytest.raises(ValueError, match="changed"):
        _prepare(case)
    assert case.consumed == [0, 1, 2, 3]
    assert (case.output / ".argus-incomplete").is_file()
    assert not (case.output / "manifest.json").exists()


def test_count_and_metadata_budget_fail_before_output_creation(case) -> None:
    with pytest.raises(ValueError, match="max_images"):
        _prepare(case, selection="all-compared", max_images=1)
    assert not case.output.exists()
    with pytest.raises(ValueError, match="metadata alone"):
        _prepare(case, max_output_bytes=1)
    assert not case.output.exists() and not case.consumed


def test_actual_pretty_manifest_is_included_in_output_budget(case) -> None:
    manifest = _prepare(case)
    data = _load(case, manifest)
    png_size = sum((manifest.parent / row["image_path"]).stat().st_size for row in data["images"])
    final_bytes = manifest.stat().st_size + png_size
    case.output = case.output.with_name("second-image-review")
    case.decoder.complete = False
    with pytest.raises(ValueError, match="byte|bytes"):
        _prepare(case, max_output_bytes=final_bytes - 1)
    assert not (case.output / "manifest.json").exists()


@pytest.mark.parametrize("destination", ["exists", "profile", "source", "parent_missing"])
def test_output_must_be_new_and_not_overlap_sources(case, destination: str) -> None:
    if destination == "exists":
        case.output.mkdir()
    elif destination == "profile":
        case.output = case.profile_root / "new-child"
    elif destination == "source":
        case.output = case.video
    else:
        case.output = case.output / "missing-parent" / "child"
    with pytest.raises((ValueError, FileExistsError)):
        _prepare(case)
    assert not case.consumed


def test_explicit_protected_paths_apply_to_output_subtrees(case) -> None:
    with pytest.raises(ValueError, match="protected"):
        _prepare(case, protected_paths=[case.output.parent])


def test_report_updates_require_a_new_artifact(case) -> None:
    manifest = _prepare(case)
    case.payload["frames"][2]["lightglue"]["reason"] = "updated diagnostic report"
    case.report.write_text(json.dumps(case.payload), encoding="utf-8")
    case.reviews = load_scene_comparisons(case.bundle, [case.report])
    with pytest.raises(ValueError, match="current bundle/reports"):
        _load(case, manifest)


def test_joint_png_hash_and_manifest_digest_rewrite_cannot_forge_pixels(case) -> None:
    manifest = _prepare(case)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    row = data["images"][0]
    png = images._png_bytes(b"\x00" * 6, width=2, height=1)
    (manifest.parent / row["image_path"]).write_bytes(png)
    row["image_sha256"] = hashlib.sha256(png).hexdigest()
    _rewrite_manifest(manifest, data)
    with pytest.raises(ValueError, match="pixels differ"):
        _load(case, manifest)


@pytest.mark.parametrize("mutation", [
    "frame", "pts", "path", "missing", "duplicate", "authority", "evidence",
])
def test_loader_rederives_selection_and_authority(case, mutation: str) -> None:
    manifest = _prepare(case)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "frame":
        data["images"][0]["frame_index"] = 0
    elif mutation == "pts":
        data["images"][0]["pts_exact"] = "-87749"
    elif mutation == "path":
        data["images"][0]["image_path"] = "../original-vfr.mp4"
    elif mutation == "missing":
        data["images"].clear()
    elif mutation == "duplicate":
        data["images"].append(copy.deepcopy(data["images"][0]))
    elif mutation == "authority":
        data["release_authority"] = True
    else:
        data["decoder_evidence"][case.clip.clip_id]["ffmpeg"]["clean_eof"] = False
    _rewrite_manifest(manifest, data)
    with pytest.raises(ValueError):
        _load(case, manifest)


@pytest.mark.parametrize("mutation", ["index", "number", "seconds", "comparison", "authority"])
def test_joint_manifest_redigest_cannot_hide_equal_valued_json_type_changes(case, mutation) -> None:
    manifest = _prepare(case, selection="all-compared")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    row = payload["images"][0]
    if mutation == "index":
        row["frame_index"] = False  # Python compares False == 0.
    elif mutation == "number":
        row["frame_number"] = True  # Python compares True == 1.
    elif mutation == "seconds":
        row["timestamp_sec"] = -3  # Original exact display value is -3.0.
    elif mutation == "comparison":
        row["comparison"]["comparable"] = 1
    else:
        payload["release_authority"] = 0
    _rewrite_manifest(manifest, payload)
    with pytest.raises(ValueError, match="disagrees|current bundle/reports"):
        _load(case, manifest)


def test_loader_rejects_hardlinked_images(case) -> None:
    manifest = _prepare(case)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    image = manifest.parent / data["images"][0]["image_path"]
    try:
        os.link(image, manifest.parent / "image-alias.png")
    except OSError:
        pytest.skip("filesystem does not support hard links")
    with pytest.raises(ValueError, match="hard links"):
        _load(case, manifest)


def test_output_parent_symlink_is_rejected(case, tmp_path: Path) -> None:
    target, link = tmp_path / "real-parent", tmp_path / "linked-parent"
    target.mkdir()
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not permitted by this host")
    case.output = link / "new-review"
    with pytest.raises(ValueError, match="symlink/reparse"):
        _prepare(case)
    assert not (target / "new-review").exists()


def test_png_rejects_extra_streams_filters_chunks_and_zip_bombs() -> None:
    ihdr = images._png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
    ending = images._png_chunk(b"IEND", b"")
    good = zlib.compress(b"\x00" + b"\x01" * 6)
    candidates = [
        images._PNG_SIGNATURE + ihdr + images._png_chunk(b"IDAT", good + good) + ending,
        images._PNG_SIGNATURE + ihdr
        + images._png_chunk(b"IDAT", zlib.compress(b"\x01" * 7)) + ending,
        images._PNG_SIGNATURE + ihdr
        + images._png_chunk(b"IDAT", zlib.compress(b"\x00" * 100_000)) + ending,
        images._PNG_SIGNATURE + ihdr + images._png_chunk(b"IDAT", good) + ending + b"tail",
        images._PNG_SIGNATURE + ihdr + images._png_chunk(b"tEXt", b"unexpected")
        + images._png_chunk(b"IDAT", good) + ending,
    ]
    for png in candidates:
        with pytest.raises(ValueError):
            images._png_bgr(png, width=2, height=1)


def test_report_guard_read_is_bounded(case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "_MAX_MANIFEST_BYTES", 32)
    with pytest.raises(ValueError, match="bounded protected-input"):
        images._report_guards(case.reviews)
    assert not case.output.exists()


def test_no_disagreements_requires_explicit_all_compared(case) -> None:
    case.payload["frames"][2]["lightglue"] = _decision()
    case.report.write_text(json.dumps(case.payload), encoding="utf-8")
    case.reviews = load_scene_comparisons(case.bundle, [case.report])
    with pytest.raises(ValueError, match="all-compared"):
        _prepare(case)
    assert not case.output.exists()
