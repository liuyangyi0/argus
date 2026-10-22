"""Controlled, source-bound still-image review for variable-frame-rate video.

This offline-only path never seeks by nominal FPS or changes its source video.
Only ``prepare_scene_review_images`` imports the strict decoder. Loading an
existing artifact uses the standard library to verify our deliberately narrow
lossless PNG format, original BGR tensor identity, PTS and report bindings.
Nothing in this module approves a profile, publishes a model or grants a gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
import zlib
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from argus.integrations.contracts import (
    ReviewBundle,
    ReviewClip,
    require_verified_media,
    review_source_paths,
    write_json,
)
from argus.integrations.scene_comparison import (
    SceneComparisonReview,
    require_verified_scene_comparisons,
)


_SCHEMA = "argus.scene-review-images.v1"
_INCOMPLETE = ".argus-incomplete"
_INCOMPLETE_BYTES = b"argus.scene-review-images.v1\n"
_MAX_IMAGES = 10_000
_MAX_OUTPUT_BYTES = 10 * 1024**3
_MAX_MANIFEST_BYTES = 512 * 1024**2
_MAX_RAW_IMAGE_BYTES = 128 * 1024**2
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SELECTIONS = {"disagreements", "all-compared"}


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("image review must contain finite UTF-8 JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in image review: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite image review number")
    return parsed


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite image review constant: {value}")


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicates,
            parse_float=_finite_float, parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("image review must be valid UTF-8 JSON") from exc


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _check_chain(path: Path, *, must_exist: bool = True) -> None:
    """Reject symlinks and Windows junction/reparse points before resolution."""
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            if must_exist:
                raise ValueError(f"image review path does not exist: {component}") from None
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400,
        ):
            raise ValueError(f"image review cannot use symlink/reparse paths: {component}")


def _read_owned_file(path: Path, maximum: int) -> bytes:
    _check_chain(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
        raise ValueError("image review files must be bounded regular files without hard links")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("image review file changed while opening")
        raw = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
    _check_chain(path)
    current = path.lstat()
    def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
        return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_nlink

    if (
        len(raw) > maximum or identity(info) != identity(after)
        or identity(after) != identity(current)
    ):
        raise ValueError("image review file changed while reading or exceeded its limit")
    return raw


def _exclusive_write(path: Path, data: bytes) -> None:
    _check_chain(path.parent)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _check_chain(path)
    if path.stat().st_nlink != 1:
        raise ValueError("image review output acquired an unexpected hard link")


def _check_bounds(max_images: int, max_output_bytes: int) -> None:
    if type(max_images) is not int or not 1 <= max_images <= _MAX_IMAGES:
        raise ValueError(f"max_images must be an integer in [1, {_MAX_IMAGES}]")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= _MAX_OUTPUT_BYTES:
        raise ValueError(f"max_output_bytes must be an integer in [1, {_MAX_OUTPUT_BYTES}]")


def _dimensions(clip: ReviewClip) -> int:
    size = clip.width * clip.height * 3
    if size > _MAX_RAW_IMAGE_BYTES:
        raise ValueError("source dimensions exceed the controlled still-image pixel budget")
    return size


def _tensor_sha256(bgr: bytes, *, width: int, height: int) -> str:
    # This is byte-for-byte the offline comparison's _frame_sha256 contract:
    # uint8 dtype, default-spaced JSON dimensions, contiguous original BGR.
    digest = hashlib.sha256()
    digest.update(b"uint8")
    digest.update(json.dumps([height, width, 3]).encode("ascii"))
    digest.update(bgr)
    return digest.hexdigest()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", zlib.crc32(kind + payload) & 0xFFFFFFFF,
    )


def _png_bytes(bgr: bytes, *, width: int, height: int) -> bytes:
    rgb = bytearray(bgr)
    rgb[0::3], rgb[2::3] = bgr[2::3], bgr[0::3]
    stride = width * 3
    scanlines = b"".join(
        b"\x00" + rgb[offset:offset + stride] for offset in range(0, len(rgb), stride)
    )
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def _png_bgr(raw: bytes, *, width: int, height: int) -> bytes:
    """Accept only this module's RGB8, filter-0, one-IDAT lossless format."""
    if not raw.startswith(_PNG_SIGNATURE):
        raise ValueError("image review requires its controlled PNG format")
    position = len(_PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    for expected_kind in (b"IHDR", b"IDAT", b"IEND"):
        if position + 12 > len(raw):
            raise ValueError("incomplete controlled PNG")
        size = struct.unpack_from(">I", raw, position)[0]
        kind = raw[position + 4:position + 8]
        end = position + 8 + size
        if kind != expected_kind or end + 4 > len(raw):
            raise ValueError("unexpected chunks in controlled PNG")
        data = raw[position + 8:end]
        crc = struct.unpack_from(">I", raw, end)[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != crc:
            raise ValueError("controlled PNG chunk CRC mismatch")
        chunks.append((kind, data))
        position = end + 4
    if position != len(raw) or chunks[2][1] or chunks[0][1] != struct.pack(
        ">IIBBBBB", width, height, 8, 2, 0, 0, 0,
    ):
        raise ValueError("controlled PNG dimensions, format or trailing data disagree")
    expected = (width * 3 + 1) * height
    decoder = zlib.decompressobj()
    try:
        scanlines = decoder.decompress(chunks[1][1], expected + 1)
    except zlib.error as exc:
        raise ValueError("controlled PNG contains invalid compressed pixels") from exc
    if (
        len(scanlines) != expected or not decoder.eof or decoder.unconsumed_tail
        or decoder.unused_data
    ):
        raise ValueError("controlled PNG pixel budget, EOF or compressed stream disagrees")
    stride = width * 3 + 1
    if any(scanlines[offset] != 0 for offset in range(0, expected, stride)):
        raise ValueError("controlled PNG supports only filter-0 scanlines")
    rgb = b"".join(
        scanlines[offset + 1:offset + stride] for offset in range(0, expected, stride)
    )
    bgr = bytearray(rgb)
    bgr[0::3], bgr[2::3] = rgb[2::3], rgb[0::3]
    return bytes(bgr)


def _image_path(clip_id: str, index: int) -> str:
    # User-controlled identities never become filesystem path components.
    return f"images/{_sha256(clip_id.encode('utf-8'))}/{index:012d}.png"


def _expected(
    bundle: ReviewBundle, reviews: Mapping[str, SceneComparisonReview], selection: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(selection, str) or selection not in _SELECTIONS:
        raise ValueError("selection must be disagreements or all-compared")
    if not reviews:
        raise ValueError("still-image review requires at least one scene comparison report")
    reports: dict[str, Any] = {}
    images: list[dict[str, Any]] = []
    metadata_bytes = 0
    for clip in bundle.clips:
        review = reviews.get(clip.clip_id)
        if review is None:
            continue
        _dimensions(clip)
        reports[clip.clip_id] = {
            "report_path": str(review.report_path), "report_sha256": review.report_sha256,
            "report_status": review.report_status, "profile_id": review.profile_id,
            "summary": json.loads(_json_bytes(review.summary)),
        }
        metadata_bytes += len(_json_bytes(reports[clip.clip_id]))
        if metadata_bytes > _MAX_MANIFEST_BYTES:
            raise ValueError("still-image metadata exceeds its hard manifest budget")
        origin = clip.timestamps[0]
        first = origin.pts * Fraction(origin.time_base_num, origin.time_base_den)
        for row in review.frame_rows:
            if selection == "disagreements" and not any(row[key] is True for key in (
                "state_disagreement", "raw_state_disagreement", "comparison_error",
            )):
                continue
            index = row["frame_index"]
            stamp = clip.timestamps[index]
            time_base = Fraction(stamp.time_base_num, stamp.time_base_den)
            exact = stamp.pts * time_base
            image = {
                "clip_id": clip.clip_id, "camera_id": clip.camera_id,
                "source_video_sha256": clip.video_sha256,
                "frame_index": index, "frame_number": index + 1,
                "pts_exact": str(stamp.pts),
                "time_base_exact": f"{time_base.numerator}/{time_base.denominator}",
                "timestamp_sec": float(exact), "relative_timestamp_sec": float(exact - first),
                "frame_sha256": row["frame_sha256"],
                "image_path": _image_path(clip.clip_id, index),
                "report_sha256": review.report_sha256,
                "comparison": json.loads(_json_bytes(row)),
            }
            metadata_bytes += len(_json_bytes(image))
            if metadata_bytes > _MAX_MANIFEST_BYTES:
                raise ValueError("still-image metadata exceeds its hard manifest budget")
            images.append(image)
            if len(images) > _MAX_IMAGES:
                raise ValueError("selected frames exceed the hard still-image count budget")
    if not images:
        raise ValueError("no frames match the still-image selection; consider all-compared")
    base = {
        "schema_version": _SCHEMA, "diagnostic_only": True, "gating_effect": "none",
        "release_authority": False, "ground_truth_authority": False,
        "bundle_id": bundle.bundle_id, "suite_id": bundle.suite_id,
        "bundle_sha256": _sha256(_json_bytes(bundle.model_dump(mode="json"))),
        "selection": selection, "reports": reports,
    }
    return base, images


def _report_guards(reviews: Mapping[str, SceneComparisonReview]) -> tuple[set[Path], set[Path]]:
    files: set[Path] = set()
    directories: set[Path] = set()
    for review in reviews.values():
        with review.report_path.open("rb") as handle:
            raw = handle.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_MANIFEST_BYTES:
            raise ValueError("scene comparison exceeds its bounded protected-input read")
        if _sha256(raw) != review.report_sha256:
            raise ValueError("scene comparison changed while reading protected inputs")
        payload = _parse_json(raw)
        for key, destination in (
            ("protected_paths", files), ("protected_directories", directories),
        ):
            entries = payload.get(key)
            if not isinstance(entries, list) or not entries or any(
                not isinstance(entry, str) or not entry for entry in entries
            ):
                raise ValueError(f"still-image preparation requires comparison {key}")
            destination.update(_absolute(entry).resolve() for entry in entries)
        files.add(review.report_path.resolve())
    return files, directories


def _check_output(
    output: Path, bundle: ReviewBundle, reviews: Mapping[str, SceneComparisonReview],
    protected_paths: Iterable[str | Path],
) -> None:
    _check_chain(output.parent)
    if not output.parent.is_dir():
        raise ValueError("still-image output parent must already be a directory")
    if os.path.lexists(output):
        raise FileExistsError(f"still-image output directory must be new: {output}")
    files, directories = _report_guards(reviews)
    files.update(review_source_paths(bundle))
    files.update(_absolute(path).resolve() for path in protected_paths)
    for protected in files | directories:
        if (
            output == protected or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError("still-image output overlaps protected source/profile paths")


def _validate_evidence(evidence: Any, clip: ReviewClip) -> None:
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != "argus.strict-video-decoder-evidence.v1"
    ):
        raise ValueError("still-image extraction requires strict decoder evidence")
    if (
        evidence.get("backend") != "ffprobe+ffmpeg-rawvideo"
        or evidence.get("pixel_format") != "bgr24"
    ):
        raise ValueError("still-image decoder backend/pixel format disagrees")
    for key in (
        "strict_release_decoder", "source_sha256_matches", "pts_strictly_increasing",
        "container_declared_frame_count_matches", "decode_error_distinguishable_from_clean_eof",
        "variable_frame_rate_timestamps_preserved", "complete_decode_verified", "source_exhausted",
    ):
        if evidence.get(key) is not True:
            raise ValueError(f"still-image decoder did not prove {key}")
    if evidence.get("truncated") is not False:
        raise ValueError("still-image extraction must decode through clean EOF")
    for key in ("source_sha256_before", "source_sha256_after"):
        if evidence.get(key) != clip.video_sha256:
            raise ValueError("still-image decoder source hash disagrees")
    for key, value in (
        ("width", clip.width), ("height", clip.height),
        ("timestamp_frame_count", clip.frame_count),
    ):
        if type(evidence.get(key)) is not int or evidence[key] != value:
            raise ValueError("still-image decoder dimensions/frame count disagree")
    first = clip.timestamps[0]
    time_base = Fraction(first.time_base_num, first.time_base_den)
    if evidence.get("time_base") != f"{time_base.numerator}/{time_base.denominator}":
        raise ValueError("still-image decoder time base disagrees")
    if type(evidence.get("stream_index")) is not int or evidence["stream_index"] < 0:
        raise ValueError("still-image decoder stream index is invalid")
    nominal = evidence.get("nominal_frame_rate")
    if (
        type(nominal) not in (int, float) or not math.isfinite(nominal)
        or not math.isclose(nominal, clip.fps, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("still-image decoder nominal frame rate disagrees")
    count = evidence.get("container_declared_frame_count")
    if count is not None and (type(count) is not int or count != clip.frame_count):
        raise ValueError("still-image decoder declared frame count disagrees")
    for process, count_key in (
        ("ffprobe", "decoded_frame_inventory_count"), ("ffmpeg", "decoded_frame_count"),
    ):
        details = evidence.get(process)
        if (
            not isinstance(details, dict) or type(details.get("returncode")) is not int
            or details["returncode"] != 0 or details.get("stderr_empty") is not True
        ):
            raise ValueError("still-image decoder process did not finish cleanly")
        if type(details.get(count_key)) is not int or details[count_key] != clip.frame_count:
            raise ValueError("still-image decoder process frame count disagrees")
    for key in ("decoded_frame_count_matches_ffprobe", "clean_eof", "xerror_enabled"):
        if evidence["ffmpeg"].get(key) is not True:
            raise ValueError("still-image decoder did not prove clean full-source EOF")
    if evidence["ffmpeg"].get("partial_raw_frame_detected") is not False:
        raise ValueError("still-image decoder observed a partial frame")


def _decoder(source: str, digest: str) -> Any:
    from argus.validation.strict_video_decoder import StrictVideoDecoder

    return StrictVideoDecoder(source, expected_sha256=digest)


def prepare_scene_review_images(
    bundle: ReviewBundle, *, scene_comparisons: Mapping[str, SceneComparisonReview],
    output_dir: str | Path, selection: str = "disagreements", max_images: int = _MAX_IMAGES,
    max_output_bytes: int = _MAX_OUTPUT_BYTES, protected_paths: Iterable[str | Path] = (),
) -> Path:
    """Explicitly extract original compared frames; never overwrite or truncate.

    A failed attempt remains visibly incomplete and cannot be imported. Retry
    into a different new directory; this function never recursively removes
    files and never rewrites existing artifacts or any source input.
    """
    _check_bounds(max_images, max_output_bytes)
    bundle = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    reviews = dict(scene_comparisons)
    require_verified_media(bundle)
    require_verified_scene_comparisons(reviews, bundle=bundle)
    base, expected = _expected(bundle, reviews, selection)
    if len(expected) > max_images:
        raise ValueError("selected frames exceed max_images; selection is never silently truncated")
    minimum_manifest = {
        **base, "images": [{**row, "image_sha256": "0" * 64} for row in expected],
        "decoder_evidence": {}, "artifact_id": "0" * 64,
    }
    minimum_bytes = len(json.dumps(
        minimum_manifest, ensure_ascii=False, indent=2, allow_nan=False,
    ).encode("utf-8")) + 1
    if (
        minimum_bytes > _MAX_MANIFEST_BYTES
        or minimum_bytes + len(_INCOMPLETE_BYTES) > max_output_bytes
    ):
        raise ValueError("still-image metadata alone exceeds its manifest/output byte budget")
    output = _absolute(output_dir)
    _check_output(output, bundle, reviews, protected_paths)
    output.mkdir(exist_ok=False)
    _check_chain(output)
    _exclusive_write(output / _INCOMPLETE, _INCOMPLETE_BYTES)
    (output / "images").mkdir(exist_ok=False)
    expected_by_clip: dict[str, dict[int, dict[str, Any]]] = {}
    for row in expected:
        expected_by_clip.setdefault(row["clip_id"], {})[row["frame_index"]] = row
    images: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    total_bytes = 0
    for clip in bundle.clips:
        if clip.clip_id not in reviews:
            continue
        selected = expected_by_clip.get(clip.clip_id, {})
        seen: set[int] = set()
        count = 0
        with _decoder(clip.video_path, clip.video_sha256) as decoder:
            for decoded in decoder:
                index = decoded.frame_index
                if type(index) is not int or index != count or index >= clip.frame_count:
                    raise ValueError("strict source frame order differs from the review bundle")
                count += 1
                original = clip.timestamps[index]
                tick = Fraction(original.time_base_num, original.time_base_den)
                if (
                    type(decoded.timestamp.frame_index) is not int
                    or decoded.timestamp.frame_index != index
                    or type(decoded.timestamp.pts) is not int
                    or decoded.timestamp.pts != original.pts
                    or decoded.timestamp.time_base != f"{tick.numerator}/{tick.denominator}"
                    or decoded.timestamp.pts_time_seconds != float(original.pts * tick)
                ):
                    raise ValueError(
                        "strict source frame PTS/time base differs from the review bundle"
                    )
                if index not in selected:
                    continue
                row = selected[index]
                frame = decoded.frame
                if (
                    str(frame.dtype) != "uint8"
                    or tuple(frame.shape) != (clip.height, clip.width, 3)
                ):
                    raise ValueError("strict decoded source frame is not original-size BGR uint8")
                bgr = frame.tobytes(order="C")
                if len(bgr) != _dimensions(clip) or _tensor_sha256(
                    bgr, width=clip.width, height=clip.height,
                ) != row["frame_sha256"]:
                    raise ValueError(
                        "decoded original frame tensor SHA256 differs from comparison report"
                    )
                png = _png_bytes(bgr, width=clip.width, height=clip.height)
                total_bytes += len(png)
                if total_bytes + minimum_bytes + len(_INCOMPLETE_BYTES) > max_output_bytes:
                    raise ValueError(
                        "still-image bytes exceed max_output_bytes; extraction is not truncated"
                    )
                destination = output / row["image_path"]
                if not destination.parent.exists():
                    destination.parent.mkdir(exist_ok=False)
                _exclusive_write(destination, png)
                images.append({**row, "image_sha256": _sha256(png)})
                seen.add(index)
        if count != clip.frame_count or seen != set(selected):
            raise ValueError("still-image extraction did not consume all source frames/selections")
        proof = decoder.evidence()
        _validate_evidence(proof, clip)
        evidence[clip.clip_id] = proof
    if len(images) != len(expected):
        raise ValueError("still-image extraction omitted selected frames")
    require_verified_media(bundle)
    require_verified_scene_comparisons(reviews, bundle=bundle)
    manifest = {**base, "decoder_evidence": evidence, "images": images}
    manifest["artifact_id"] = _sha256(_json_bytes(manifest))
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False,
    ).encode("utf-8") + b"\n"
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("still-image manifest exceeds its hard size budget")
    if total_bytes + len(manifest_bytes) + len(_INCOMPLETE_BYTES) > max_output_bytes:
        raise ValueError("images and manifest exceed max_output_bytes")
    _check_chain(output)
    # No-clobber atomic publication; the marker continues to block imports
    # until every source/report check and the final manifest write succeeds.
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    (output / _INCOMPLETE).unlink()
    return manifest_path


def load_scene_review_images(
    manifest_path: str | Path, *, bundle: ReviewBundle,
    scene_comparisons: Mapping[str, SceneComparisonReview],
) -> dict[str, Any]:
    """Verify an immutable artifact without importing a decoder, CV or SDK."""
    path = _absolute(manifest_path)
    if path.name != "manifest.json" or os.path.lexists(path.parent / _INCOMPLETE):
        raise ValueError("still-image artifact is incomplete or not its manifest.json")
    bundle = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    reviews = dict(scene_comparisons)
    require_verified_media(bundle)
    require_verified_scene_comparisons(reviews, bundle=bundle)
    raw = _read_owned_file(path, _MAX_MANIFEST_BYTES)
    payload = _parse_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("still-image manifest must be an object")
    base, expected = _expected(bundle, reviews, payload.get("selection"))
    if set(payload) != set(base) | {"artifact_id", "decoder_evidence", "images"}:
        raise ValueError("still-image manifest has unexpected or missing fields")
    if any(_json_bytes(payload.get(key)) != _json_bytes(value) for key, value in base.items()):
        raise ValueError("still-image manifest no longer matches the current bundle/reports")
    if not _is_digest(payload["artifact_id"]) or payload["artifact_id"] != _sha256(_json_bytes({
        key: value for key, value in payload.items() if key != "artifact_id"
    })):
        raise ValueError("still-image artifact content digest disagrees")
    evidence = payload["decoder_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(reviews):
        raise ValueError("still-image decoder evidence must cover exactly the attached reports")
    clips = {clip.clip_id: clip for clip in bundle.clips}
    for clip_id, proof in evidence.items():
        _validate_evidence(proof, clips[clip_id])
    images = payload["images"]
    if not isinstance(images, list) or len(images) != len(expected):
        raise ValueError("still-image selections are missing, duplicated or truncated")
    total_bytes = len(raw)
    for actual, row in zip(images, expected):
        if not isinstance(actual, dict) or set(actual) != set(row) | {"image_sha256"}:
            raise ValueError("still-image row has unexpected or missing fields")
        if any(_json_bytes(actual.get(key)) != _json_bytes(value) for key, value in row.items()):
            raise ValueError("still-image row disagrees with original frame/PTS/report identity")
        relative = PurePosixPath(actual["image_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in actual["image_path"]:
            raise ValueError("still-image path must remain within the artifact root")
        clip = clips[actual["clip_id"]]
        maximum = _dimensions(clip) + clip.height + 1024 * 1024
        png = _read_owned_file(path.parent.joinpath(*relative.parts), maximum)
        total_bytes += len(png)
        if total_bytes > _MAX_OUTPUT_BYTES:
            raise ValueError("still-image artifact exceeds its hard byte budget")
        if not _is_digest(actual["image_sha256"]) or _sha256(png) != actual["image_sha256"]:
            raise ValueError("still-image PNG SHA256 differs from its manifest")
        bgr = _png_bgr(png, width=clip.width, height=clip.height)
        if _tensor_sha256(bgr, width=clip.width, height=clip.height) != row["frame_sha256"]:
            raise ValueError("still-image pixels differ from the source-bound comparison frame")
    require_verified_media(bundle)
    require_verified_scene_comparisons(reviews, bundle=bundle)
    if (
        os.path.lexists(path.parent / _INCOMPLETE)
        or _read_owned_file(path, _MAX_MANIFEST_BYTES) != raw
    ):
        raise ValueError("still-image manifest changed during verification")
    return payload


__all__ = ["prepare_scene_review_images", "load_scene_review_images"]
