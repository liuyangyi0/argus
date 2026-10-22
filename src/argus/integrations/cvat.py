"""Optional, full-video CVAT review bridge. No production state is modified.

The local receipt is the authoritative upload-to-task mapping. CVAT does not expose
a checksum of the original uploaded video, so the receipt binds the locally
verified bytes to the created task, immutable upload metadata and task creation
date. Renaming/replacing a task or changing its frame selection invalidates it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .contracts import (
    ReviewAnnotations,
    ReviewBox,
    ReviewBundle,
    ReviewClip,
    ReviewRange,
    file_sha256,
    require_verified_media,
    review_source_paths,
)

_RECEIPT_SCHEMA = "argus.cvat-receipt.v2"
_EVENT_ATTRIBUTE = "argus_event_id"
_IGNORE_REASON_ATTRIBUTE = "argus_ignore_reason"
_SOURCE_GEOMETRY_ATTRIBUTE = "argus_source_geometry"


class CVATIntegrationError(ValueError):
    """A review operation failed without changing Argus production state."""


class _MissingCVATDependency(RuntimeError):
    pass


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if callable(getattr(value, "to_dict", None)):
        return value.to_dict()
    raise CVATIntegrationError("Unexpected CVAT response object")


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value))


def _integer(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise CVATIntegrationError(f"CVAT {field} must be an integer")
    return value


def _frame(value: Any, clip: ReviewClip) -> int:
    index = _integer(value, field="frame")
    if not 0 <= index < clip.frame_count:
        raise CVATIntegrationError(f"CVAT frame outside clip {clip.clip_id}")
    return index


def _server_url(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise CVATIntegrationError("CVAT URL must be HTTP(S), without embedded credentials/query")
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


@contextmanager
def _client_session(
    url: str,
    *,
    username: str | None,
    password: str | None,
    token: str | None,
    organization: str | None,
) -> Iterator[tuple[Any, Any, Any]]:
    # Import only when explicitly invoking CVAT. Ordinary Argus startup stays light.
    try:
        from cvat_sdk import Client, Config, models
        from cvat_sdk.core.client import AccessTokenCredentials
        from cvat_sdk.core.proxies.tasks import ResourceType
    except ImportError:
        raise _MissingCVATDependency(
            'CVAT integration requires: pip install -e ".[cvat]"'
        ) from None

    token = (
        token
        or os.getenv("ARGUS_CVAT_TOKEN")
        or os.getenv("ARGUS_CVAT_ACCESS_TOKEN")
        or os.getenv("CVAT_ACCESS_TOKEN")
    )
    username = username or os.getenv("ARGUS_CVAT_USERNAME")
    password = password or os.getenv("ARGUS_CVAT_PASSWORD")
    if token and (username or password):
        raise CVATIntegrationError("Configure either a CVAT access token or username/password")
    if not token and not (username and password):
        raise CVATIntegrationError("Set ARGUS_CVAT_TOKEN or ARGUS_CVAT_USERNAME/PASSWORD")
    # Never suppress SDK/server API incompatibility or certificate validation.
    with Client(url, config=Config(allow_unsupported_server=False, verify_ssl=True)) as client:
        client.login(AccessTokenCredentials(token) if token else (username, password))
        client.organization_slug = organization or ""
        yield client, models, ResourceType


def _source_identity(bundle: ReviewBundle, clip: ReviewClip) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        "suite_id": bundle.suite_id,
        "manifest_sha256": bundle.manifest_sha256,
        "clip_id": clip.clip_id,
        "camera_id": clip.camera_id,
        "source_session_id": clip.source_session_id,
        "video_sha256": clip.video_sha256,
        "video_name": Path(clip.video_path).name,
        "frame_count": clip.frame_count,
        "width": clip.width,
        "height": clip.height,
        "fps": clip.fps,
        "split": clip.split,
        "scene_profile_id": clip.scene_profile_id,
        "model_version_id": clip.model_version_id,
        "source_annotations_sha256": hashlib.sha256(
            json.dumps(
                clip.annotations.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "timestamps_sha256": hashlib.sha256(
            json.dumps(
                [item.model_dump(mode="json") for item in clip.timestamps],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _identity_name(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "Argus-review-" + hashlib.sha256(encoded).hexdigest()


def _label_names(clip: ReviewClip) -> list[str]:
    names = {"anomaly", "ignore"}
    names.update(item.label for item in clip.annotations.positive_ranges)
    names.update(item.label for item in clip.annotations.boxes)
    if "ignore" in {item.label for item in clip.annotations.positive_ranges}:
        raise CVATIntegrationError("The 'ignore' label cannot denote a positive range")
    if "ignore" in {item.label for item in clip.annotations.boxes}:
        raise CVATIntegrationError("Ignore must be a full-frame ignore range, not a rectangle")
    return sorted(names)


def _label_specs(names: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "type": "tag" if name == "ignore" else "any",
            "attributes": [
                {
                    "name": attr_name,
                    "mutable": False,
                    "input_type": "text",
                    "default_value": "",
                    "values": [],
                }
                for attr_name in (
                    [_EVENT_ATTRIBUTE, _IGNORE_REASON_ATTRIBUTE]
                    if name == "ignore"
                    else [_EVENT_ATTRIBUTE, _SOURCE_GEOMETRY_ATTRIBUTE]
                )
            ],
        }
        for name in names
    ]


def _preflight_source_annotations(clip: ReviewClip) -> None:
    specs = _label_specs(_label_names(clip))
    for label_id, label in enumerate(specs, 1):
        label["id"] = label_id
        for attr_id, attribute in enumerate(label["attributes"], label_id * 100):
            attribute["id"] = attr_id
    _seed_annotations(clip, specs)


def _label_map(labels: Sequence[Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    for raw in labels:
        item = _mapping(raw)
        label_id = _integer(item.get("id"), field="label id")
        name = item.get("name")
        if (
            label_id <= 0 or not isinstance(name, str) or not name
            or name in names or label_id in result
        ):
            raise CVATIntegrationError("Invalid/duplicate CVAT label schema")
        attribute_ids: set[int] = set()
        attribute_names: set[str] = set()
        for raw_attribute in item.get("attributes", []):
            attribute = _mapping(raw_attribute)
            attribute_id = _integer(attribute.get("id"), field="attribute id")
            attribute_name = attribute.get("name")
            if (
                attribute_id <= 0 or attribute_id in attribute_ids
                or not isinstance(attribute_name, str) or not attribute_name
                or attribute_name in attribute_names
            ):
                raise CVATIntegrationError("Invalid/duplicate CVAT attribute schema")
            attribute_ids.add(attribute_id)
            attribute_names.add(attribute_name)
        names.add(name)
        result[label_id] = item
    return result


def _label_for(item: dict[str, Any], labels: dict[int, dict[str, Any]]) -> dict[str, Any]:
    label_id = _integer(item.get("label_id"), field="annotation label id")
    if label_id not in labels:
        raise CVATIntegrationError("CVAT annotation references an unknown label")
    return labels[label_id]


def _attribute_value(item: dict[str, Any], label: dict[str, Any], name: str) -> str | None:
    schema = {
        attribute["id"]
        for raw in label.get("attributes", [])
        if (attribute := _mapping(raw)).get("name") == name
    }
    values = [
        str(attribute.get("value", ""))
        for raw in item.get("attributes", [])
        if (attribute := _mapping(raw)).get("spec_id") in schema
    ]
    if len(values) > 1:
        raise CVATIntegrationError("Duplicate CVAT review identity/reason attributes")
    return (values[0] or None) if values else None


def _event_id(item: dict[str, Any], label: dict[str, Any]) -> str | None:
    return _attribute_value(item, label, _EVENT_ATTRIBUTE)


def _label_signature(labels: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": label_id,
            "name": item["name"],
            "type": _enum(item.get("type", "any")),
            "attributes": sorted(
                [
                    {
                        "id": attr["id"], "name": attr["name"],
                        "mutable": attr.get("mutable"),
                        "input_type": _enum(attr.get("input_type")),
                        "default_value": attr.get("default_value"),
                        "values": attr.get("values"),
                    }
                    for raw in item.get("attributes", [])
                    if (attr := _mapping(raw))
                ],
                key=lambda attr: attr["id"],
            ),
        }
        for label_id, item in sorted(_label_map(labels).items())
    ]


def _rectangle(item: dict[str, Any], clip: ReviewClip) -> tuple[float, float, float, float]:
    if _enum(item.get("type")) != "rectangle" or item.get("elements"):
        raise CVATIntegrationError("Only axis-aligned CVAT rectangles are supported")
    rotation = item.get("rotation", 0)
    if not isinstance(rotation, (int, float)) or not math.isfinite(rotation) or rotation % 360:
        raise CVATIntegrationError("Rotated CVAT rectangles are not supported")
    points = item.get("points")
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise CVATIntegrationError("CVAT rectangle needs exactly four coordinates")
    if any(type(value) not in {int, float} or not math.isfinite(value) for value in points):
        raise CVATIntegrationError("CVAT rectangle coordinates must be finite numbers")
    left, top, right, bottom = map(float, points)
    if not (0 <= left < right <= clip.width and 0 <= top < bottom <= clip.height):
        raise CVATIntegrationError("CVAT rectangle coordinates are outside the original video")
    return left, top, right, bottom


def _ranges(frames: dict[tuple[str, str | None], set[int]]) -> list[ReviewRange]:
    result: list[ReviewRange] = []
    for (label, event_id), indices in sorted(frames.items(), key=lambda pair: str(pair[0])):
        ordered = sorted(indices)
        if not ordered:
            continue
        start = previous = ordered[0]
        for index in ordered[1:]:
            if index == previous + 1:
                previous = index
                continue
            result.append(ReviewRange(
                start_frame=start, end_frame=previous, label=label, event_id=event_id,
            ))
            start = previous = index
        result.append(ReviewRange(
            start_frame=start, end_frame=previous, label=label, event_id=event_id,
        ))
    return result


def annotations_from_cvat(
    clip: ReviewClip,
    annotations: Any,
    labels: Sequence[Any],
    *,
    reviewed_by: str,
    completed: bool,
    confirm_normal: bool = False,
) -> ReviewAnnotations:
    """Convert zero-based CVAT tags/shapes/tracks, rejecting ambiguous annotation loss.

    Seeded source geometry is marked explicitly: on a source label/frame with
    frame tags, its boxes do not introduce extra events. Newly drawn geometry
    remains an independent event even when another event already tags that frame.
    Removing an object box does not remove a pre-existing frame-tag event; a human
    reviewer must edit the frame tags as well when rejecting that event.

    Rectangle keyframes interpolate linearly. An outside keyframe excludes its own
    frame and subsequent frames until a visible keyframe. A final visible keyframe
    holds until the last frame, matching CVAT video-track semantics.
    """
    if not isinstance(reviewed_by, str) or not reviewed_by.strip():
        raise CVATIntegrationError("An explicit reviewed_by identity is required")
    if not completed:
        raise CVATIntegrationError("CVAT task must be completed before final review import")
    data = _mapping(annotations)
    if any(value for key, value in data.items() if key not in {
        "version", "tags", "shapes", "tracks",
    }):
        raise CVATIntegrationError(
            "Unsupported CVAT annotation collection; intervals cannot be silently discarded"
        )
    label_ids = _label_map(labels)
    geometry_event_origins: dict[str, bool] = {}
    for collection in ("shapes", "tracks"):
        for raw in data.get(collection, []):
            item = _mapping(raw)
            label = _label_for(item, label_ids)
            event_id = _event_id(item, label)
            if event_id is None:
                continue
            source_geometry = (
                _attribute_value(item, label, _SOURCE_GEOMETRY_ATTRIBUTE) == "true"
            )
            if event_id in geometry_event_origins and (
                source_geometry or geometry_event_origins[event_id]
            ):
                raise CVATIntegrationError(
                    "Cloned source geometry needs a new event ID and a cleared "
                    "argus_source_geometry marker; the same object must remain "
                    "in its original track"
                )
            geometry_event_origins[event_id] = source_geometry
    explicit_event_ids = {
        event_id
        for collection in ("tags", "shapes", "tracks")
        for raw in data.get(collection, [])
        if (event_id := _event_id(_mapping(raw), _label_for(_mapping(raw), label_ids)))
    }

    def anonymous_event_id(kind: str, identity: int) -> str:
        candidate = f"cvat-{kind}-{identity}"
        while candidate in explicit_event_ids:
            candidate += "-generated"
        explicit_event_ids.add(candidate)
        return candidate

    positives: dict[tuple[str, str | None], set[int]] = defaultdict(set)
    ignored: dict[tuple[str, str | None], set[int]] = defaultdict(set)
    boxes: list[ReviewBox] = []
    tag_coverage: dict[str, set[int]] = defaultdict(set)

    def record(
        frame: int, label: dict[str, Any], item: dict[str, Any], event_id: str | None,
        *, geometry: bool = False,
    ) -> None:
        if label["name"] == "ignore":
            reason = _attribute_value(item, label, _IGNORE_REASON_ATTRIBUTE) or "ignore"
            ignored[reason, event_id].add(frame)
        elif (
            not geometry
            or _attribute_value(item, label, _SOURCE_GEOMETRY_ATTRIBUTE) != "true"
            or frame not in tag_coverage[label["name"]]
        ):
            positives[label["name"], event_id].add(frame)

    for raw in data.get("tags", []):
        item = _mapping(raw)
        label = _label_for(item, label_ids)
        frame = _frame(item.get("frame"), clip)
        record(frame, label, item, _event_id(item, label))
        if label["name"] != "ignore":
            tag_coverage[label["name"]].add(frame)

    for raw in data.get("shapes", []):
        item = _mapping(raw)
        label = _label_for(item, label_ids)
        if label["name"] == "ignore":
            raise CVATIntegrationError("Ignore annotations must be full-frame tags, not shapes")
        frame = _frame(item.get("frame"), clip)
        points = _rectangle(item, clip)
        outside = item.get("outside", False)
        if type(outside) is not bool:
            raise CVATIntegrationError("CVAT outside flags must be booleans")
        if outside:
            raise CVATIntegrationError("Outside shapes must belong to a CVAT track")
        event_id = _event_id(item, label)
        if event_id is None:
            # Separate same-frame rectangles must not merge into one event.
            shape_id = item.get("id")
            if shape_id is not None:
                shape_id = _integer(shape_id, field="shape id")
            event_id = anonymous_event_id(
                "shape", shape_id if shape_id is not None else len(boxes),
            )
        record(frame, label, item, event_id, geometry=True)
        if label["name"] != "ignore":
            boxes.append(ReviewBox(
                frame_index=frame, label=label["name"], xtl=points[0], ytl=points[1],
                xbr=points[2], ybr=points[3], track_id=None,
            ))

    seen_tracks: set[int] = set()
    for raw in data.get("tracks", []):
        track = _mapping(raw)
        label = _label_for(track, label_ids)
        if label["name"] == "ignore":
            raise CVATIntegrationError("Ignore annotations must be full-frame tags, not tracks")
        if track.get("elements"):
            raise CVATIntegrationError("Nested/skeleton CVAT tracks are not supported")
        track_id = _integer(track.get("id"), field="track id")
        if track_id < 0 or track_id in seen_tracks:
            raise CVATIntegrationError("Invalid/duplicate CVAT track identity")
        seen_tracks.add(track_id)
        event_id = _event_id(track, label) or anonymous_event_id("track", track_id)
        keyframes: list[tuple[int, bool, tuple[float, float, float, float]]] = []
        for raw_shape in track.get("shapes", []):
            shape = _mapping(raw_shape)
            outside = shape.get("outside", False)
            if type(outside) is not bool:
                raise CVATIntegrationError("CVAT outside flags must be booleans")
            keyframes.append((_frame(shape.get("frame"), clip), outside, _rectangle(shape, clip)))
        keyframes.sort(key=lambda item: item[0])
        if not keyframes or len({item[0] for item in keyframes}) != len(keyframes):
            raise CVATIntegrationError("CVAT track has missing/duplicate keyframes")
        if _frame(track.get("frame"), clip) != keyframes[0][0]:
            raise CVATIntegrationError("CVAT track start disagrees with its first keyframe")
        for position, (start, outside, left_points) in enumerate(keyframes):
            if outside:
                continue
            next_key = keyframes[position + 1] if position + 1 < len(keyframes) else None
            stop = next_key[0] if next_key else clip.frame_count
            for frame in range(start, stop):
                fraction = (frame - start) / (stop - start) if next_key else 0.0
                points = tuple(
                    a + (b - a) * fraction
                    for a, b in zip(left_points, next_key[2] if next_key else left_points)
                )
                record(frame, label, track, event_id, geometry=True)
                if label["name"] != "ignore":
                    boxes.append(ReviewBox(
                        frame_index=frame, label=label["name"], xtl=points[0], ytl=points[1],
                        xbr=points[2], ybr=points[3], track_id=track_id,
                    ))

    if not positives and not ignored and not confirm_normal:
        raise CVATIntegrationError(
            "Empty CVAT annotations are unknown; explicitly confirm this clip as normal"
        )
    if confirm_normal and (positives or ignored or boxes):
        raise CVATIntegrationError(
            "A confirmed-normal clip cannot contain positive or ignore annotations"
        )
    return ReviewAnnotations(
        positive_ranges=_ranges(positives),
        ignore_ranges=_ranges(ignored),
        boxes=sorted(boxes, key=lambda item: (
            item.frame_index, item.track_id if item.track_id is not None else -1, item.label,
        )),
        reviewed=True,
        reviewed_by=reviewed_by.strip(),
    )


def _seed_annotations(clip: ReviewClip, labels: Sequence[Any]) -> dict[str, Any]:
    by_name = {item["name"]: item for item in _label_map(labels).values()}

    def attributes(
        label: str, event_id: str | None, ignore_reason: str | None = None,
        *, source_geometry: bool = False,
    ) -> list[dict[str, Any]]:
        result = []
        for name, value in (
            (_EVENT_ATTRIBUTE, event_id), (_IGNORE_REASON_ATTRIBUTE, ignore_reason),
            (_SOURCE_GEOMETRY_ATTRIBUTE, "true" if source_geometry else None),
        ):
            if value is None:
                continue
            attr = next((
                item for raw in by_name[label].get("attributes", [])
                if (item := _mapping(raw)).get("name") == name
            ), None)
            if attr is None:
                raise CVATIntegrationError(
                    "CVAT task is missing a review identity/reason attribute"
                )
            result.append({"spec_id": attr["id"], "value": value})
        return result

    tags = []
    used_ids = {
        item.event_id
        for item in [*clip.annotations.positive_ranges, *clip.annotations.ignore_ranges]
        if item.event_id is not None
    }
    for ignored_range, ranges in (
        (False, clip.annotations.positive_ranges), (True, clip.annotations.ignore_ranges),
    ):
        for index, item in enumerate(ranges):
            name = "ignore" if ignored_range else item.label
            event_id = item.event_id
            if event_id is None:
                kind = "ignore" if ignored_range else "positive"
                event_id = f"argus-source-{kind}-range-{index}"
                while event_id in used_ids:
                    event_id += "-generated"
                used_ids.add(event_id)
            for frame in range(item.start_frame, item.end_frame + 1):
                tags.append({
                    "frame": frame,
                    "label_id": by_name[name]["id"],
                    "attributes": attributes(
                        name, event_id, item.label if ignored_range else None,
                    ),
                    "source": "auto",
                })
    shapes = []
    tracked: dict[int, list[ReviewBox]] = defaultdict(list)

    def rectangle(
        item: ReviewBox, *, frame: int | None = None, outside: bool = False,
    ) -> dict[str, Any]:
        return {
            "frame": item.frame_index if frame is None else frame,
            "type": "rectangle",
            "points": [item.xtl, item.ytl, item.xbr, item.ybr],
            "occluded": False,
            "outside": outside,
            "z_order": 0,
            "rotation": 0.0,
            "attributes": [],
        }

    for box_index, item in enumerate(clip.annotations.boxes):
        if item.track_id is not None:
            tracked[item.track_id].append(item)
        else:
            shape = rectangle(item)
            shape.update(
                label_id=by_name[item.label]["id"], source="auto",
                attributes=attributes(
                    item.label, f"argus-source-shape-{box_index}", source_geometry=True,
                ),
            )
            shapes.append(shape)
    tracks = []
    for track_id, items in sorted(tracked.items()):
        items.sort(key=lambda item: item.frame_index)
        if (
            len({item.label for item in items}) != 1
            or len({item.frame_index for item in items}) != len(items)
        ):
            raise CVATIntegrationError("Source track has mixed labels or duplicate frame boxes")
        keyframes = []
        for position, item in enumerate(items):
            keyframes.append(rectangle(item))
            next_frame = (
                items[position + 1].frame_index if position + 1 < len(items)
                else clip.frame_count
            )
            if next_frame > item.frame_index + 1 and item.frame_index + 1 < clip.frame_count:
                keyframes.append(rectangle(item, frame=item.frame_index + 1, outside=True))
        tracks.append({
            "frame": items[0].frame_index,
            "label_id": by_name[items[0].label]["id"],
            "attributes": attributes(
                items[0].label, f"argus-track-{track_id}", source_geometry=True,
            ),
            "source": "auto",
            "shapes": keyframes,
        })
    # Explicit outside keyframes prevent inventing visibility across annotation gaps.
    return {"version": 0, "tags": tags, "shapes": shapes, "tracks": tracks}


def _meta_snapshot(task: Any, clip: ReviewClip, *, expected_name: str) -> dict[str, Any]:
    task.fetch()
    if task.name != expected_name or _enum(task.mode) != "interpolation":
        raise CVATIntegrationError("CVAT task identity/mode differs from its upload receipt")
    if _enum(getattr(task, "dimension", "2d")) != "2d":
        raise CVATIntegrationError("CVAT task must be a 2D full-video task")
    meta = _mapping(task.get_meta())
    if (
        _integer(meta.get("size"), field="size") != clip.frame_count
        or _integer(task.size, field="task size") != clip.frame_count
    ):
        raise CVATIntegrationError("CVAT frame count differs from verified video")
    if meta.get("start_frame") != 0 or meta.get("stop_frame") != clip.frame_count - 1:
        raise CVATIntegrationError("CVAT must contain the complete zero-based video frame range")
    if meta.get("frame_filter", "") not in {"", "step=1"} or meta.get("deleted_frames"):
        raise CVATIntegrationError("CVAT frame sampling/deletion is not supported")
    included = meta.get("included_frames")
    if included is not None and included != list(range(clip.frame_count)):
        raise CVATIntegrationError("CVAT frame selection differs from the complete video")
    frames = [_mapping(item) for item in meta.get("frames", [])]
    # CVAT video metadata exposes one source FrameMeta (not an image-list task).
    if len(frames) != 1:
        raise CVATIntegrationError("CVAT source must be exactly one complete video")
    source = frames[0]
    if (
        source.get("name") != Path(clip.video_path).name
        or source.get("width") != clip.width
        or source.get("height") != clip.height
    ):
        raise CVATIntegrationError("CVAT source filename/dimensions differ from verified video")
    created = getattr(task, "created_date", None)
    if created is None:
        raise CVATIntegrationError("CVAT task is missing its creation identity")
    return {
        "task_name": task.name,
        "task_created_date": str(created),
        "size": meta["size"],
        "start_frame": meta["start_frame"],
        "stop_frame": meta["stop_frame"],
        "frame_filter": meta.get("frame_filter", ""),
        "source": {key: source[key] for key in ("name", "width", "height")},
    }


@contextmanager
def _receipt_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise CVATIntegrationError(
            "CVAT receipt is locked; reconcile an interrupted operation before retrying"
        ) from None
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(str(os.getpid()))
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_receipt(
    path: Path, bundle: ReviewBundle, url: str, organization: str | None,
) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise CVATIntegrationError("Cannot read a valid CVAT upload receipt") from None
    expected = (
        _RECEIPT_SCHEMA, bundle.bundle_id, bundle.suite_id,
        bundle.manifest_sha256, url, organization or "",
    )
    if not isinstance(receipt, dict) or tuple(receipt.get(key) for key in (
        "schema_version", "bundle_id", "suite_id", "manifest_sha256", "server_url", "organization",
    )) != expected:
        raise CVATIntegrationError(
            "CVAT receipt belongs to a different server/organization/source bundle"
        )
    if (
        not isinstance(receipt.get("clips"), dict)
        or set(receipt["clips"]) - {item.clip_id for item in bundle.clips}
    ):
        raise CVATIntegrationError("CVAT receipt contains unknown clip bindings")
    task_ids: set[int] = set()
    for clip in bundle.clips:
        entry = receipt["clips"].get(clip.clip_id)
        if entry is None:
            continue
        if not isinstance(entry, dict) or entry.get("source") != _source_identity(bundle, clip):
            raise CVATIntegrationError("CVAT receipt source/timestamp/split identity mismatch")
        task_id = entry.get("task_id")
        if task_id is not None:
            task_id = _integer(task_id, field="receipt task id")
            if task_id <= 0 or task_id in task_ids:
                raise CVATIntegrationError("CVAT receipt binds duplicate/invalid task IDs")
            task_ids.add(task_id)
    return receipt


def _receipt_destination(receipt_path: str | Path, bundle: ReviewBundle) -> Path:
    path = Path(receipt_path).resolve()
    protected = review_source_paths(bundle)
    if path in protected or path.with_name(path.name + ".lock") in protected:
        raise CVATIntegrationError(
            "CVAT receipt/lock path cannot replace review source media/labels"
        )
    return path


def _validate_bound_task(task: Any, entry: dict[str, Any], clip: ReviewClip) -> list[Any]:
    if _integer(task.id, field="task id") != entry.get("task_id"):
        raise CVATIntegrationError("CVAT task ID differs from its receipt")
    snapshot = _meta_snapshot(task, clip, expected_name=_identity_name(entry["source"]))
    if entry.get("metadata") != snapshot:
        raise CVATIntegrationError("CVAT source metadata differs from its upload receipt")
    labels = task.get_labels()
    if sorted(item["name"] for item in _label_map(labels).values()) != entry.get("label_names"):
        raise CVATIntegrationError("CVAT task label schema changed; refuse ambiguous import")
    if _label_signature(labels) != entry.get("label_schema"):
        raise CVATIntegrationError(
            "CVAT label IDs/review attributes differ from the upload receipt"
        )
    return labels


def export_bundle_to_cvat(
    bundle: ReviewBundle,
    *,
    url: str,
    receipt_path: str | Path,
    username: str | None = None,
    password: str | None = None,
    token: str | None = None,
    organization: str | None = None,
) -> dict[str, Any]:
    """Upload one video per task, saving every known remote identity before upload.

    Repeating a completed export validates existing tasks without changing labels.
    Unknown remote outcomes remain in the receipt and require reconciliation; the
    bridge never blindly creates duplicate tasks or overwrites human annotations.
    """
    bundle = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    require_verified_media(bundle)
    for clip in bundle.clips:
        _preflight_source_annotations(clip)
    url = _server_url(url)
    path = _receipt_destination(receipt_path, bundle)
    with _receipt_lock(path):
        if path.exists():
            receipt = _load_receipt(path, bundle, url, organization)
        else:
            receipt = {
                "schema_version": _RECEIPT_SCHEMA,
                "bundle_id": bundle.bundle_id,
                "suite_id": bundle.suite_id,
                "manifest_sha256": bundle.manifest_sha256,
                "server_url": url,
                "organization": organization or "",
                "clips": {},
            }
            _write_receipt(path, receipt)
        try:
            with _client_session(
                url, username=username, password=password, token=token, organization=organization,
            ) as (client, models, resource_type):
                for clip in bundle.clips:
                    entry = receipt["clips"].get(clip.clip_id)
                    if entry is not None:
                        if entry.get("status") != "ready":
                            raise CVATIntegrationError(
                                f"CVAT upload for {clip.clip_id} has an uncertain outcome; "
                                "reconcile receipt before retrying"
                            )
                        _validate_bound_task(client.tasks.retrieve(entry["task_id"]), entry, clip)
                        continue
                    identity = _source_identity(bundle, clip)
                    names = _label_names(clip)
                    entry = {
                        "source": identity, "status": "creating", "task_id": None,
                        "label_names": names,
                    }
                    receipt["clips"][clip.clip_id] = entry
                    _write_receipt(path, receipt)
                    spec = models.TaskWriteRequest(
                        name=_identity_name(identity), labels=_label_specs(names)
                    )
                    task = client.tasks.create(spec=spec)
                    entry.update(
                        task_id=_integer(task.id, field="created task id"), status="uploading",
                    )
                    _write_receipt(path, receipt)
                    task.upload_data(
                        resources=[clip.video_path], resource_type=resource_type.LOCAL,
                        params={
                            "start_frame": 0, "stop_frame": clip.frame_count - 1,
                            "frame_step": 1, "image_quality": 100,
                        },
                        wait_for_completion=True,
                    )
                    if file_sha256(clip.video_path) != clip.video_sha256:
                        raise CVATIntegrationError(
                            "Video changed during CVAT upload; task cannot be bound as ready"
                        )
                    entry["metadata"] = _meta_snapshot(
                        task, clip, expected_name=_identity_name(identity),
                    )
                    entry["status"] = "seeding"
                    _write_receipt(path, receipt)
                    labels = task.get_labels()
                    if sorted(item["name"] for item in _label_map(labels).values()) != names:
                        raise CVATIntegrationError(
                            "Created CVAT task label schema differs from requested labels"
                        )
                    entry["label_schema"] = _label_signature(labels)
                    seed = _seed_annotations(clip, labels)
                    if seed["tags"] or seed["shapes"] or seed["tracks"]:
                        task.set_annotations(models.LabeledDataRequest(**seed))
                    entry["status"] = "ready"
                    _write_receipt(path, receipt)
        except (CVATIntegrationError, _MissingCVATDependency):
            raise
        except Exception:
            raise CVATIntegrationError(
                "CVAT export failed; keep the receipt and reconcile recorded task IDs "
                "before retrying"
            ) from None
    return receipt


def import_bundle_from_cvat(
    bundle: ReviewBundle,
    *,
    url: str,
    receipt_path: str | Path,
    reviewed_by: str,
    username: str | None = None,
    password: str | None = None,
    token: str | None = None,
    organization: str | None = None,
    confirmed_normal_clip_ids: Collection[str] = (),
) -> ReviewBundle:
    """Return a reviewed bundle; never mutate the original bundle or production DB.

    annotation_path/annotation_sha256 continue identifying the original manifest's
    source labels. They are provenance, not checksums of the new review annotations.
    """
    bundle = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    require_verified_media(bundle)
    if not isinstance(reviewed_by, str) or not reviewed_by.strip():
        raise CVATIntegrationError("An explicit reviewed_by identity is required")
    normal_ids = set(confirmed_normal_clip_ids)
    if normal_ids - {item.clip_id for item in bundle.clips}:
        raise CVATIntegrationError("Normal confirmation contains unknown clip IDs")
    url = _server_url(url)
    path = _receipt_destination(receipt_path, bundle)
    with _receipt_lock(path):
        receipt = _load_receipt(path, bundle, url, organization)
        if set(receipt["clips"]) != {item.clip_id for item in bundle.clips}:
            raise CVATIntegrationError("CVAT receipt does not cover every source clip")
        clips = []
        try:
            with _client_session(
                url, username=username, password=password, token=token, organization=organization,
            ) as (client, _models, _resource_type):
                for clip in bundle.clips:
                    entry = receipt["clips"][clip.clip_id]
                    if entry.get("status") != "ready":
                        raise CVATIntegrationError("CVAT upload receipt is not ready")
                    task = client.tasks.retrieve(entry["task_id"])
                    labels = _validate_bound_task(task, entry, clip)
                    if _enum(task.status) != "completed":
                        raise CVATIntegrationError(
                            "CVAT task must be completed before final review import"
                        )
                    updated_date = getattr(task, "updated_date", None)
                    if updated_date is None:
                        raise CVATIntegrationError(
                            "CVAT task is missing its annotation revision timestamp"
                        )
                    raw_annotations = task.get_annotations()
                    _validate_bound_task(task, entry, clip)
                    if str(task.updated_date) != str(updated_date):
                        raise CVATIntegrationError(
                            "CVAT task changed during review import; retry a stable completed task"
                        )
                    annotations = annotations_from_cvat(
                        clip, raw_annotations, labels, reviewed_by=reviewed_by,
                        completed=_enum(task.status) == "completed",
                        confirm_normal=clip.clip_id in normal_ids,
                    )
                    clips.append(clip.model_copy(update={"annotations": annotations}))
        except (CVATIntegrationError, _MissingCVATDependency):
            raise
        except Exception:
            raise CVATIntegrationError(
                "CVAT import failed; no reviewed bundle was written"
            ) from None
    require_verified_media(bundle)
    return ReviewBundle.model_validate(
        bundle.model_copy(update={"clips": clips}).model_dump(mode="json")
    )
