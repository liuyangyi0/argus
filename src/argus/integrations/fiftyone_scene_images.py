"""Optional FiftyOne *image* review of controlled SceneGuard frame artifacts.

This offline importer never seeks a video, infers a model, changes ground truth,
or approves a scene profile. Each sample displays one lossless PNG whose pixels
have already been bound to an original frame and exact PTS by the artifact
loader. Importing this module does not import the optional FiftyOne SDK.

Image datasets have separate ownership from the native-video review adapter.
Managed observations may be refreshed, but manual fields are never assigned,
including ``None``: FiftyOne saves dirty fields from cached sample documents,
so assigning a cached missing value could erase a concurrently saved review.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import socket
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

from argus.integrations.contracts import ReviewBundle, ReviewClip, file_sha256
from argus.integrations.fiftyone import _dataset_import_lock
from argus.integrations.scene_comparison import SceneComparisonReview
from argus.integrations.scene_review_images import load_scene_review_images

_OWNER = "argus.scene-review-images.fiftyone.v1"
_INFO_KEY = "argus_scene_images"
_SAMPLE_SCHEMA_REVISION = 1
_MAX_INFO_REPORT_BYTES = 1024 * 1024
_SOURCE_REGISTRY_KEY = "source_registry"
_SOURCE_REGISTRY_REQUIRED_KEY = "source_registry_required"
_SOURCE_REGISTRY_SCHEMA = "argus.scene-review-image-sources.v2"
_MAX_SOURCE_REGISTRY_BYTES = 4 * 1024 * 1024
_MAX_SDK_CONFIG_BYTES = 1024 * 1024
_SDK_ATTEMPTED = False
_SDK_LOCK = threading.Lock()
_LOCK_COLLECTION_NAME = "argus_scene_image_import_locks"
_DATASET_PROJECTION = {
    "_id": 1, "name": 1, "info": 1, "media_type": 1, "version": 1,
}
_SOURCE_FIELDS = frozenset({
    "video_sha256", "camera_id", "media_identity", "source_session_id", "split",
})
_ClipSource = tuple[str, str, str | None, str, str]
_HUMAN_STATES = ("same_pose", "moved", "uncertain", "unreviewable")
_HUMAN_FIELDS = (
    "argus_scene_human_state", "argus_scene_human_note",
    "argus_scene_reviewed_report_sha256",
)
_STRING_FIELDS = (
    "argus_owner", "argus_scene_image_key", "argus_suite_id", "argus_bundle_id",
    "argus_clip_id", "argus_camera_id", "argus_source_session_id", "argus_split",
    "argus_video_sha256", "argus_media_identity", "argus_manifest_sha256",
    "argus_image_manifest_sha256", "argus_image_artifact_id", "argus_image_sha256",
    "argus_import_state", "argus_import_generation", "argus_pts_exact",
    "argus_time_base_exact", "argus_frame_sha256", "argus_scene_frame_sha256",
    "argus_scene_report_sha256", "argus_scene_comparison_path",
    "argus_scene_comparison_status", "argus_scene_comparison_profile_id",
    "argus_scene_comparison_summary_json", "argus_scene_comparison_json",
    "argus_scene_review_category", "argus_scene_error_algorithms_json",
    "argus_scene_gating_effect", "argus_scene_baseline_status",
    "argus_scene_baseline_state", "argus_scene_baseline_raw_state",
    "argus_scene_baseline_reason", "argus_scene_baseline_metrics_json",
    "argus_scene_lightglue_status", "argus_scene_lightglue_state",
    "argus_scene_lightglue_raw_state", "argus_scene_lightglue_reason",
    "argus_scene_lightglue_metrics_json", "argus_scene_lightglue_matcher_json",
)
_BOOLEAN_FIELDS = (
    "argus_scene_has_comparison", "argus_scene_comparable", "argus_scene_disagreement",
    "argus_scene_raw_disagreement", "argus_scene_comparison_error",
    "argus_scene_diagnostic_only", "argus_scene_release_authority",
    "argus_scene_ground_truth_authority",
)
_FLOAT_FIELDS = (
    "argus_timestamp_sec", "argus_relative_timestamp_sec",
    "argus_scene_baseline_elapsed_ms", "argus_scene_lightglue_elapsed_ms",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _preflight_sdk_config() -> None:
    """Refuse configured initialization hooks before the optional SDK runs.

    ETA ignores empty environment variables, so ``FIFTYONE_MODULE_PATH=""``
    does not override a nonempty ``module_path`` in FiftyOne's JSON config.
    Inspect the same path as ``fiftyone.core.config.locate_config`` without
    importing it. Do not replace the config: that could silently select another
    database, and it may contain credentials that must not be copied or logged.
    """
    config_path = os.environ.get(
        "FIFTYONE_CONFIG_PATH",
        os.path.join(os.path.expanduser("~"), ".fiftyone", "config.json"),
    )
    try:
        # Match SDK load_config: an absent config uses the SDK defaults. An
        # explicitly supplied path is used verbatim, not expanduser/resolved.
        if not os.path.isfile(config_path):
            return
        with open(config_path, "rb") as handle:
            raw = handle.read(_MAX_SDK_CONFIG_BYTES + 1)
        if len(raw) > _MAX_SDK_CONFIG_BYTES:
            raise ValueError("SDK config exceeds the controlled metadata budget")
        config = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as exc:
        raise RuntimeError(
            "image review cannot safely inspect FiftyOne config before SDK initialization; "
            "use a valid hook-free config in a fresh independent scene-images-import CLI process"
        ) from exc
    if not isinstance(config, dict):
        raise RuntimeError(
            "image review requires a FiftyOne JSON config object with no module hooks"
        )
    module_path = config.get("module_path")
    if module_path is not None and (not isinstance(module_path, list) or module_path):
        raise RuntimeError(
            "image review refuses FiftyOne config module hooks before SDK initialization; "
            "use a hook-free config in a fresh independent scene-images-import CLI process"
        )


def _load_review_sdk() -> tuple[Any, Any]:
    """Initialize one isolated importer, without inheriting SDK singletons.

    FiftyOne's cached ``load_dataset`` can save dirty dataset metadata before
    returning it. Disabling the cache after prior SDK use is insufficient:
    existing cached samples can still be returned. Require a fresh process,
    disable caches and automatic migrations before the very first SDK import,
    and never attempt another import in this process, including after failure.
    The returned dataset may still be launched in the same CLI process.
    """
    global _SDK_ATTEMPTED
    with _SDK_LOCK:
        if _SDK_ATTEMPTED:
            raise RuntimeError(
                "image review requires a fresh independent scene-images-import CLI process; "
                "FiftyOne SDK was already loaded or attempted"
            )
        _SDK_ATTEMPTED = True
        if any(
            name == "fiftyone" or name.startswith("fiftyone.")
            for name in tuple(sys.modules)
        ):
            raise RuntimeError(
                "image review requires a fresh independent scene-images-import CLI process; "
                "FiftyOne SDK was already loaded or attempted"
            )
        # These process-wide settings are intentional and are not temporarily
        # toggled around a cached SDK. A failed initialization also requires a
        # new process so partially initialized modules cannot be reused.
        os.environ["FIFTYONE_SINGLETON_CACHE"] = "false"
        os.environ["FIFTYONE_DATABASE_ADMIN"] = "false"
        # Clear any environment-provided hooks, then inspect file-provided hooks
        # before import. ETA treats an empty environment value as unspecified.
        os.environ["FIFTYONE_MODULE_PATH"] = ""
        _preflight_sdk_config()
        fo = importlib.import_module("fiftyone")
        config = getattr(fo, "config", None)
        module_path = getattr(config, "module_path", None)
        if (
            getattr(config, "singleton_cache", None) is not False
            or getattr(config, "database_admin", None) is not False
            or (
                module_path is not None
                and (not isinstance(module_path, (list, tuple)) or len(module_path) != 0)
            )
        ):
            raise RuntimeError(
                "image review requires FiftyOne with singleton_cache=False and "
                "database_admin=False and no module hooks in a fresh independent "
                "scene-images-import CLI process"
            )
        database = importlib.import_module("fiftyone.core.odm.database")
        connection = database.get_db_conn()
        datasets = connection.datasets
        for handle in (connection, datasets):
            preference = getattr(handle, "read_preference", None)
            mode = getattr(preference, "mode", None)
            concern = getattr(handle, "write_concern", None)
            if (
                type(mode) is not int or mode != 0
                or getattr(concern, "acknowledged", None) is not True
                or getattr(getattr(handle, "read_concern", None), "level", None) != "local"
            ):
                # Changing only our raw collection would leave SDK loads/saves
                # using the URI's unsafe defaults. Even PRIMARY majority reads
                # can omit an acknowledged w=1 write while replicas lag. An
                # unspecified concern may inherit a server-wide majority
                # default, so it cannot establish fresh identity/sample reads.
                # Refuse the whole connection instead of overriding raw reads.
                raise RuntimeError(
                    "image review requires PRIMARY reads and acknowledged writes on the "
                    "FiftyOne database and datasets collection, with explicit local read concern; "
                    "configure the database URI with readConcernLevel=local and retry in a "
                    "fresh independent scene-images-import CLI process"
                )
        return fo, datasets


def _has_simple_collation(metadata: dict[str, Any]) -> bool:
    """Require byte-exact names and owner tokens, never locale comparisons."""
    if "collation" not in metadata:
        return True
    collation = metadata["collation"]
    return (
        isinstance(collation, dict) and set(collation) == {"locale"}
        and collation["locale"] == "simple"
    )


def _lock_collection_metadata(database: Any) -> dict[str, Any] | None:
    """Read one exact collection description without assuming it is ordinary."""
    cursor = database.list_collections(filter={"name": _LOCK_COLLECTION_NAME})
    rows: list[Any] = []
    try:
        for row in cursor:
            rows.append(row)
            if len(rows) == 2:
                break
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()
    if len(rows) > 1:
        raise ValueError("database import lock collection metadata is ambiguous")
    if not rows:
        return None
    if not isinstance(rows[0], dict):
        raise ValueError("database import lock collection metadata is invalid")
    return rows[0]


def _validated_lock_collection(datasets: Any) -> Any:
    """Require a normal, globally unique, non-expiring lock storage contract.

    Merely inserting the same ``_id`` is not sufficient: time-series collections
    have no default unique ID index, capped/TTL storage can evict a live lock,
    and sharded collections need not enforce ID uniqueness across shards. Never
    repair, clear or rebuild an unknown collection to make it appear compatible.
    Concurrent collection/index administration remains outside this protocol.
    """
    try:
        database = datasets.database
        for handle in (database, datasets):
            mode = getattr(getattr(handle, "read_preference", None), "mode", None)
            if (
                type(mode) is not int or mode != 0
                or getattr(getattr(handle, "write_concern", None), "acknowledged", None) is not True
                or getattr(getattr(handle, "read_concern", None), "level", None) != "local"
            ):
                raise ValueError(
                    "lock storage requires PRIMARY reads, explicit local read concern "
                    "and acknowledged writes"
                )
        if getattr(getattr(database, "client", None), "is_mongos", None) is not False:
            raise ValueError("database import locks do not support mongos/sharded deployments")
        metadata = _lock_collection_metadata(database)
        if metadata is None:
            try:
                database.create_collection(_LOCK_COLLECTION_NAME)
            except Exception as exc:
                # PyMongo may detect a concurrent creator before issuing the
                # command, or the server may answer NamespaceExists (48).
                namespace_exists = type(getattr(exc, "code", None)) is int and exc.code == 48
                collection_exists = (
                    type(exc).__name__ == "CollectionInvalid"
                    and type(exc).__module__ == "pymongo.errors"
                )
                if not namespace_exists and not collection_exists:
                    raise
            # Creation or an existence race is never evidence of a safe shape.
            metadata = _lock_collection_metadata(database)
        if (
            not isinstance(metadata, dict)
            or metadata.get("name") != _LOCK_COLLECTION_NAME
            or metadata.get("type") != "collection"
            or not isinstance(metadata.get("options"), dict)
        ):
            raise ValueError("database import locks require an ordinary collection")
        options = metadata["options"]
        if not _has_simple_collation(options):
            # A delete without an operation-level collation inherits the
            # collection default. Numeric/case-insensitive comparison must not
            # turn a different token into the apparent current owner.
            raise ValueError("database import locks require simple binary collection collation")
        if (
            options.get("capped", False) is not False
            or options.get("autoIndexId", True) is not True
            or any(name in options for name in (
                "timeseries", "viewOn", "pipeline", "clusteredIndex", "expireAfterSeconds",
            ))
        ):
            raise ValueError("database import lock storage must not evict or expire live locks")
        locks = database.get_collection(_LOCK_COLLECTION_NAME)
        mode = getattr(getattr(locks, "read_preference", None), "mode", None)
        if (
            type(mode) is not int or mode != 0
            or getattr(getattr(locks, "write_concern", None), "acknowledged", None) is not True
            or getattr(getattr(locks, "read_concern", None), "level", None) != "local"
        ):
            raise ValueError(
                "lock collection requires PRIMARY reads, explicit local read concern "
                "and acknowledged writes"
            )
        indexes = locks.index_information()
        if not isinstance(indexes, dict) or any(
            not isinstance(spec, dict) or "expireAfterSeconds" in spec
            or not _has_simple_collation(spec)
            for spec in indexes.values()
        ):
            raise ValueError(
                "database import lock indexes must be inspectable, binary and have no TTL"
            )
        identity = indexes.get("_id_")
        key = identity.get("key") if isinstance(identity, dict) else None
        if (
            not isinstance(key, (list, tuple)) or len(key) != 1
            or not isinstance(key[0], (list, tuple)) or len(key[0]) != 2
            or key[0][0] != "_id" or type(key[0][1]) is not int or key[0][1] != 1
            or identity.get("unique", True) is not True
            or identity.get("sparse", False) is not False
            or identity.get("hidden", False) is not False
            or "partialFilterExpression" in identity
        ):
            raise ValueError("database import locks require the ordinary unique _id_ index")
        return locks
    except Exception as exc:
        raise RuntimeError(
            "image review cannot verify a safe database import lock collection contract; "
            "use an ordinary non-expiring collection on a non-mongos dedicated review database, "
            "configure readConcernLevel=local and simple binary collation, "
            "and have an administrator inspect its metadata and indexes without clearing live locks"
        ) from exc


def _uncertain_database_failure(error: BaseException) -> BaseException | None:
    """Find an opaque DB/transport failure without importing the optional SDK.

    A client-side exception cannot prove that a server-side write has stopped.
    SDK wrappers can hide the driver error in either exception chain, or inside
    a group, so inspect all of them. Unknown inspection/interruption outcomes
    are conservative too; a diagnostic must never accidentally release a lock.
    """
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        try:
            if (
                not isinstance(current, Exception)
                or isinstance(current, (ConnectionError, TimeoutError))
                or getattr(current, "timeout", None) is True
                or any(
                    cls.__module__.split(".", 1)[0] in {"pymongo", "mongoengine"}
                    for cls in type(current).__mro__
                )
            ):
                return current
            pending.extend(
                chained for chained in (current.__cause__, current.__context__)
                if isinstance(chained, BaseException)
            )
            if isinstance(current, BaseExceptionGroup):
                pending.extend(reversed(current.exceptions))
        except BaseException:
            return current
    return None


class _ImportLease:
    """Remember uncertain writes, including errors swallowed by compensation."""

    def __init__(self) -> None:
        self.uncertain_failure: BaseException | None = None

    @property
    def uncertain(self) -> bool:
        return self.uncertain_failure is not None

    def record_failure(self, error: BaseException) -> bool:
        if self.uncertain_failure is not None:
            return True
        # Fail closed before running diagnostics: even allocating the
        # classifier's work queue or an interruption can fail. Keep the original
        # failure if classification itself cannot establish a definite result.
        self.uncertain_failure = error
        try:
            uncertain = _uncertain_database_failure(error)
        except BaseException:
            return True
        self.uncertain_failure = uncertain
        return self.uncertain


_SAFE_IMPORT_ERROR_MESSAGES = frozenset({
    "image artifact changed while importing",
    "original-frame pixels changed; use a new image dataset",
    "clip source/frame mapping changed; use a new image dataset",
    "image dataset exact original-frame PTS was changed",
    "image dataset original-frame number was changed",
    "image dataset sample source is absent from the durable source registry",
    "image dataset has an incomplete/degraded v2 source registry; "
    "historical samples cannot repair the durable source registry",
    "refusing unrelated/native-video dataset; use a separate image dataset name",
    "image dataset belongs to another review suite",
    "image dataset is bound to an importer on another host",
    "image review cannot write to a native-video dataset",
    "image dataset has an invalid source registry requirement marker",
    "image dataset has a legacy/incomplete source registry; automatic legacy migration is "
    "unsafe; use a new image dataset or a separately verified complete-history migration",
    "image dataset has a legacy/incomplete v1 source registry; use a new image dataset or "
    "a separately verified complete-history migration",
    "image dataset has an invalid versioned source registry",
    "image source registry must contain only finite JSON values",
    "image source registry exceeds the controlled metadata budget",
    "image dataset has an invalid source registry clip identity",
    "image dataset source registry has unexpected or missing fields",
    "image dataset has invalid source registry field values",
    "image dataset contains conflicting clip source identities",
    "image dataset contains conflicting clip capture-session identities",
    "image dataset contains conflicting assigned clip splits",
    "image dataset has invalid persisted identity metadata",
    "image dataset SDK version differs; automatic migration is disabled; "
    "use a separately migrated dataset",
    "image dataset persisted identity changed while loading; retry in a "
    "fresh independent scene-images-import CLI process",
    "database import lock release was not confirmed",
})


def _import_error_summary(error: BaseException) -> str:
    """Expose types and known fixed Argus diagnostics, not arbitrary DB details."""
    name = type(error).__name__[:80]
    # Never call an SDK exception's custom __str__: it can contain credentials,
    # be arbitrarily large, or raise while we are trying to preserve the lock.
    if type(error) in {ValueError, RuntimeError} and len(error.args) == 1:
        message = error.args[0]
        if type(message) is str and message in _SAFE_IMPORT_ERROR_MESSAGES:
            return f"{name}: {message}"
    return name


class ImportLockFailure(RuntimeError):
    """A failed/uncertain lock outcome that preserves the primary import cause."""

    def __init__(
        self, message: str, *, import_error: BaseException | None = None,
        cleanup_error: BaseException | None = None,
        uncertain_error: BaseException | None = None,
    ) -> None:
        self.import_error = import_error
        self.cleanup_error = cleanup_error
        self.uncertain_error = uncertain_error
        parts = [message]
        for label, error in (
            ("import error", import_error), ("cleanup error", cleanup_error),
            ("uncertain error", uncertain_error),
        ):
            if error is not None:
                parts.append(f"{label}={_import_error_summary(error)}")
        super().__init__("; ".join(parts))


@contextmanager
def _database_import_lock(datasets: Any, dataset_name: str) -> Iterator[_ImportLease]:
    """Serialize importers across users/temp directories, without expiring locks.

    A dataset's hostname marker and a process-local filesystem lock cannot
    coordinate two users with different temporary directories. MongoDB's unique
    ``_id`` is the acquisition boundary. An uncertain acknowledgement must not
    grant ownership, and no later importer may automatically steal the record.
    """
    token = uuid.uuid4().hex
    recovery = (
        "confirm no importer or related server-side operation is active, then have an "
        "administrator recover the lock "
        "in the dedicated review database before retrying in a fresh CLI process"
    )
    # A bad collection contract is not an attempted acquisition. Preserve its
    # precise preflight failure instead of suggesting an unknown write occurred.
    locks = _validated_lock_collection(datasets)
    try:
        acquired = locks.insert_one({
            "_id": dataset_name, "token": token, "host": socket.gethostname(),
            "pid": os.getpid(), "created_at": datetime.now(timezone.utc),
        })
        if getattr(acquired, "acknowledged", None) is not True:
            raise RuntimeError("database import lock acquisition was not acknowledged")
    except BaseException as exc:
        # This includes duplicate keys and acknowledgements lost after commit.
        # Even an apparently stale lock is never reclaimed automatically.
        raise RuntimeError(
            "image dataset database import lock could not be acquired or its outcome is "
            "uncertain; " + recovery
        ) from exc
    lease = _ImportLease()
    import_error: BaseException | None = None
    try:
        yield lease
    except BaseException as exc:
        import_error = exc
        lease.record_failure(exc)
        raise
    finally:
        if lease.uncertain:
            # Deleting a mutex on a new connection cannot cancel an earlier
            # timed-out write. Keep the record while that write may still finish.
            raise ImportLockFailure(
                "image dataset database import lock retained after an uncertain database "
                "operation; " + recovery,
                import_error=import_error, uncertain_error=lease.uncertain_failure,
            ) from import_error
        try:
            released = locks.delete_one({"_id": dataset_name, "token": token})
            deleted_count = getattr(released, "deleted_count", None) if (
                getattr(released, "acknowledged", None) is True
            ) else None
            if (
                getattr(released, "acknowledged", None) is not True
                or type(deleted_count) is not int or deleted_count != 1
            ):
                raise RuntimeError("database import lock release was not confirmed")
        except BaseException as exc:
            raise ImportLockFailure(
                "image dataset database import lock release is unconfirmed; " + recovery,
                import_error=import_error, cleanup_error=exc,
            ) from exc


def _valid_identity(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _check_source_registry_budget(registry: Any) -> None:
    try:
        size = len(_json(registry).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("image source registry must contain only finite JSON values") from exc
    if size > _MAX_SOURCE_REGISTRY_BYTES:
        raise ValueError("image source registry exceeds the controlled metadata budget")


def _source_registry(sources: dict[str, _ClipSource]) -> dict[str, Any]:
    registry = {
        "schema_version": _SOURCE_REGISTRY_SCHEMA,
        "clips": {clip_id: {
            "video_sha256": video_sha, "camera_id": camera,
            "media_identity": identity, "source_session_id": session, "split": split,
        } for clip_id, (video_sha, camera, session, split, identity) in sources.items()},
    }
    _check_source_registry_budget(registry)
    return registry


def _load_source_registry(info: dict[str, Any]) -> dict[str, _ClipSource]:
    if (
        _SOURCE_REGISTRY_REQUIRED_KEY in info
        and info[_SOURCE_REGISTRY_REQUIRED_KEY] is not True
    ):
        raise ValueError("image dataset has an invalid source registry requirement marker")
    if (
        _SOURCE_REGISTRY_KEY not in info
        or info.get(_SOURCE_REGISTRY_REQUIRED_KEY) is not True
    ):
        # Samples cannot establish complete historical ownership: a previous
        # bundle may have included zero-image, reportless or now-absent clips.
        # Never turn this partial evidence into an apparently complete ledger.
        raise ValueError(
            "image dataset has a legacy/incomplete source registry; automatic legacy "
            "migration is unsafe; use a new image dataset or a separately verified "
            "complete-history migration"
        )
    registry = info[_SOURCE_REGISTRY_KEY]
    _check_source_registry_budget(registry)
    if (
        isinstance(registry, dict)
        and registry.get("schema_version") == "argus.scene-review-image-sources.v1"
    ):
        # V1 could have been synthesized from incomplete sample history by
        # the old automatic migration. Its shape cannot prove provenance.
        raise ValueError(
            "image dataset has a legacy/incomplete v1 source registry; use a new image "
            "dataset or a separately verified complete-history migration"
        )
    if (
        not isinstance(registry, dict) or set(registry) != {"schema_version", "clips"}
        or registry["schema_version"] != _SOURCE_REGISTRY_SCHEMA
        or not isinstance(registry["clips"], dict)
        or not registry["clips"]
    ):
        raise ValueError("image dataset has an invalid versioned source registry")
    sources: dict[str, _ClipSource] = {}
    for clip_id, source in registry["clips"].items():
        if not _valid_identity(clip_id) or not isinstance(source, dict):
            raise ValueError("image dataset has an invalid source registry clip identity")
        if set(source) != _SOURCE_FIELDS:
            raise ValueError("image dataset source registry has unexpected or missing fields")
        video_sha = source["video_sha256"]
        camera = source["camera_id"]
        identity = source["media_identity"]
        session = source["source_session_id"]
        split = source["split"]
        if (
            not _valid_sha256(video_sha) or not _valid_sha256(identity)
            or not _valid_identity(camera)
            or (session is not None and not _valid_identity(session))
            or not isinstance(split, str)
            or split not in {"unassigned", "train", "validation", "test"}
            or (split != "unassigned" and session is None)
        ):
            raise ValueError("image dataset has invalid source registry field values")
        sources[clip_id] = (video_sha, camera, session, split, identity)
    return sources


def _image_key(camera_id: str, clip_id: str, video_sha256: str, frame_index: int) -> str:
    return _digest({
        "camera_id": camera_id, "clip_id": clip_id,
        "video_sha256": video_sha256, "frame_index": frame_index,
    })


def _media_identity(clip: ReviewClip) -> str:
    # Nominal FPS is deliberately not used to seek or identify image frames.
    # The source bytes, geometry and exact original PTS inventory are binding.
    return _digest({
        "video_sha256": clip.video_sha256, "frame_count": clip.frame_count,
        "width": clip.width, "height": clip.height,
        "timestamps": [stamp.model_dump(mode="json") for stamp in clip.timestamps],
    })


def _ensure_schema(dataset: Any, fo: Any) -> None:
    for name in (*_STRING_FIELDS, *_HUMAN_FIELDS):
        dataset.add_sample_field(name, fo.StringField)
    for name in _BOOLEAN_FIELDS:
        dataset.add_sample_field(name, fo.BooleanField)
    for name in _FLOAT_FIELDS:
        dataset.add_sample_field(name, fo.FloatField)
    for name in ("argus_frame_index", "argus_frame_number"):
        dataset.add_sample_field(name, fo.IntField)
    dataset.create_index("argus_scene_image_key", unique=True)


def _image_fields(
    bundle: ReviewBundle, clip: ReviewClip, image: dict[str, Any],
    report: dict[str, Any], manifest: dict[str, Any], *,
    manifest_sha256: str, generation: str, media_identity: str, summary_json: str,
) -> dict[str, Any]:
    """Return only importer-owned observations, never labels or manual review."""
    row = image["comparison"]
    fields = {
        "argus_owner": _OWNER,
        "argus_scene_image_key": _image_key(
            clip.camera_id, clip.clip_id, clip.video_sha256, image["frame_index"],
        ),
        "argus_suite_id": bundle.suite_id, "argus_bundle_id": bundle.bundle_id,
        "argus_clip_id": clip.clip_id, "argus_camera_id": clip.camera_id,
        "argus_source_session_id": clip.source_session_id, "argus_split": clip.split,
        "argus_video_sha256": clip.video_sha256, "argus_media_identity": media_identity,
        "argus_manifest_sha256": bundle.manifest_sha256,
        "argus_image_manifest_sha256": manifest_sha256,
        "argus_image_artifact_id": manifest["artifact_id"],
        "argus_image_sha256": image["image_sha256"],
        "argus_import_state": "pending", "argus_import_generation": generation,
        "argus_frame_index": image["frame_index"],
        # This is original-source metadata, not FiftyOne's sample/frame number.
        "argus_frame_number": image["frame_number"],
        "argus_pts_exact": image["pts_exact"],
        "argus_time_base_exact": image["time_base_exact"],
        "argus_timestamp_sec": image["timestamp_sec"],
        "argus_relative_timestamp_sec": image["relative_timestamp_sec"],
        "argus_frame_sha256": image["frame_sha256"],
        "argus_scene_frame_sha256": image["frame_sha256"],
        "argus_scene_report_sha256": image["report_sha256"],
        "argus_scene_comparison_path": report["report_path"],
        "argus_scene_comparison_status": report["report_status"],
        "argus_scene_comparison_profile_id": report["profile_id"],
        "argus_scene_comparison_summary_json": summary_json,
        "argus_scene_comparison_json": _json(row),
        "argus_scene_has_comparison": True,
        "argus_scene_comparable": row["comparable"],
        "argus_scene_disagreement": row["state_disagreement"],
        "argus_scene_raw_disagreement": row["raw_state_disagreement"],
        "argus_scene_comparison_error": row["comparison_error"],
        "argus_scene_review_category": row["review_category"],
        "argus_scene_error_algorithms_json": _json(row["error_algorithms"]),
        "argus_scene_diagnostic_only": True, "argus_scene_gating_effect": "none",
        "argus_scene_release_authority": False, "argus_scene_ground_truth_authority": False,
    }
    for algorithm in ("baseline", "lightglue"):
        decision = row[algorithm]
        prefix = f"argus_scene_{algorithm}_"
        for key in ("status", "state", "raw_state", "reason", "elapsed_ms"):
            fields[prefix + key] = decision.get(key)
        metrics = decision.get("metrics")
        fields[prefix + "metrics_json"] = _json(metrics) if metrics is not None else None
    matcher = row["lightglue"].get("matcher")
    fields["argus_scene_lightglue_matcher_json"] = _json(matcher) if matcher is not None else None
    return fields


def _save_views(dataset: Any, fo: Any, generation: str) -> None:
    current = dataset.match({
        "argus_import_state": "complete", "argus_import_generation": generation,
    })
    dataset.save_view("argus-current", current, overwrite=True)
    for name, field in (
        ("argus-scene-disagreements", "argus_scene_disagreement"),
        ("argus-scene-raw-disagreements", "argus_scene_raw_disagreement"),
        ("argus-scene-errors", "argus_scene_comparison_error"),
    ):
        dataset.save_view(name, current.match(fo.ViewField(field) == True), overwrite=True)  # noqa: E712

    def text_or_empty(name: str) -> Any:
        field = fo.ViewField(name)
        return field.exists().if_else(field, "")

    reviewed_hash = text_or_empty("argus_scene_reviewed_report_sha256")
    report_hash = text_or_empty("argus_scene_report_sha256")
    same_report = reviewed_hash == report_hash
    has_comparison = fo.ViewField("argus_scene_has_comparison") == True  # noqa: E712
    reviewed = (
        ~(report_hash == "") & same_report
        & text_or_empty("argus_scene_human_state").is_in(list(_HUMAN_STATES))
    )
    # Sample-level expressions stay live when a reviewer saves notes/state.
    # Imported booleans must never cache or approve the human decision.
    for name, expression in (
        ("argus-scene-unreviewed", has_comparison & ~reviewed),
        ("argus-scene-reviewed", has_comparison & reviewed),
        ("argus-scene-stale-review", has_comparison & ~(reviewed_hash == "") & ~same_report),
    ):
        dataset.save_view(name, current.match(expression), overwrite=True)


def _merge_clip_source(
    previous: _ClipSource | None, current: _ClipSource,
) -> _ClipSource:
    """Fold historical partial selections without rewriting their samples.

    A source can acquire its first capture session and assigned split. Frames
    omitted by that import retain the unassigned ancestor, including after a
    failed import. Only these missing ancestors are compatible: two concrete
    sessions or assigned splits must never be merged into a preferred value.
    """
    if previous is None:
        return current
    video_sha, camera, session, split, identity = previous
    new_sha, new_camera, new_session, new_split, new_identity = current
    if (video_sha, camera, identity) != (new_sha, new_camera, new_identity):
        raise ValueError("image dataset contains conflicting clip source identities")
    if session is not None and new_session is not None and session != new_session:
        raise ValueError("image dataset contains conflicting clip capture-session identities")
    if split != "unassigned" and new_split != "unassigned" and split != new_split:
        raise ValueError("image dataset contains conflicting assigned clip splits")
    return (
        video_sha, camera, session if session is not None else new_session,
        split if split != "unassigned" else new_split, identity,
    )


def _existing_samples(
    dataset: Any, bundle: ReviewBundle, media_identities: dict[str, str],
    *, stored_sources: dict[str, _ClipSource],
) -> tuple[dict[str, Any], dict[str, _ClipSource]]:
    """Refuse mixed ownership and mutations to a source's original frame map."""
    existing: dict[str, Any] = {}
    sources = dict(stored_sources)
    clips = {clip.clip_id: clip for clip in bundle.clips}
    for sample in dataset.iter_samples():
        if sample.get_field("argus_owner") != _OWNER:
            raise ValueError("image dataset contains a sample not owned by Argus image review")
        if sample.get_field("argus_suite_id") != bundle.suite_id:
            raise ValueError("image dataset contains a sample from another review suite")
        clip_id = sample.get_field("argus_clip_id")
        camera_id = sample.get_field("argus_camera_id")
        video_sha = sample.get_field("argus_video_sha256")
        index = sample.get_field("argus_frame_index")
        if (
            not _valid_identity(clip_id) or not _valid_identity(camera_id)
            or not _valid_sha256(video_sha)
            or type(index) is not int or index < 0
        ):
            raise ValueError("image dataset contains an invalid original-frame identity")
        key = _image_key(camera_id, clip_id, video_sha, index)
        if sample.get_field("argus_scene_image_key") != key or key in existing:
            raise ValueError("image dataset contains conflicting or duplicate frame identities")
        frame_number = sample.get_field("argus_frame_number")
        if type(frame_number) is not int or frame_number != index + 1:
            raise ValueError("image dataset original-frame number was changed")
        identity = sample.get_field("argus_media_identity")
        split = sample.get_field("argus_split")
        session = sample.get_field("argus_source_session_id")
        if not _valid_sha256(identity):
            raise ValueError("image dataset has an invalid immutable media identity")
        if not isinstance(split, str) or split not in {"unassigned", "train", "validation", "test"}:
            raise ValueError("image dataset has an invalid split")
        if session is not None and (
            not _valid_identity(session)
        ):
            raise ValueError("image dataset has an invalid capture-session identity")
        if split != "unassigned" and not session:
            raise ValueError("assigned image sample has no capture-session identity")
        source = (video_sha, camera_id, session, split, identity)
        if clip_id not in stored_sources:
            raise ValueError(
                "image dataset sample source is absent from the durable source registry"
            )
        registered = stored_sources[clip_id]
        if _merge_clip_source(registered, source) != registered:
            # The ledger is committed before any sample assignment. Historical
            # samples may retain missing ancestors, but cannot establish a
            # concrete binding that this purportedly complete ledger lacks.
            raise ValueError(
                "image dataset has an incomplete/degraded v2 source registry; "
                "historical samples cannot repair the durable source registry"
            )
        clip = clips.get(clip_id)
        if clip is not None:
            if (
                camera_id != clip.camera_id or video_sha != clip.video_sha256
                or identity != media_identities[clip_id] or index >= clip.frame_count
            ):
                raise ValueError("clip source/frame mapping changed; use a new image dataset")
            stamp = clip.timestamps[index]
            tick = Fraction(stamp.time_base_num, stamp.time_base_den)
            time_base = f"{tick.numerator}/{tick.denominator}"
            if (
                sample.get_field("argus_pts_exact") != str(stamp.pts)
                or sample.get_field("argus_time_base_exact") != time_base
            ):
                raise ValueError("image dataset exact original-frame PTS was changed")
        existing[key] = sample
    for clip in bundle.clips:
        previous = sources.get(clip.clip_id)
        if previous is not None:
            video, camera, session, split, identity = previous
            if (video, camera, identity) != (
                clip.video_sha256, clip.camera_id, media_identities[clip.clip_id],
            ):
                raise ValueError("clip source/frame mapping changed; use a new image dataset")
            if split != "unassigned" and split != clip.split:
                raise ValueError("assigned clip split is immutable; use a new image dataset")
            if session is not None and session != clip.source_session_id:
                raise ValueError("a clip capture-session identity is immutable")
        # Current identity replaces only the in-memory aggregate. Nonselected
        # samples and their manual notes retain their original historical data.
        sources[clip.clip_id] = (
            clip.video_sha256, clip.camera_id, clip.source_session_id,
            clip.split, media_identities[clip.clip_id],
        )
    assignments: dict[tuple[str, str], set[str]] = {}
    for video_sha, _camera, session, split, _identity in sources.values():
        keys = [("video", video_sha)]
        if session:
            keys.append(("session", session))
        for key in keys:
            assignments.setdefault(key, set()).add(split)
    if any(len(splits) > 1 for splits in assignments.values()):
        raise ValueError("historical video/capture session crosses image dataset splits")
    return existing, sources


def _require_owned_metadata(
    metadata: Any, media_type: Any, bundle: ReviewBundle, host: str,
) -> dict[str, Any]:
    """Validate ownership without loading or refreshing an SDK dataset."""
    info = metadata.get(_INFO_KEY) if isinstance(metadata, dict) else None
    if not isinstance(info, dict) or info.get("owner") != _OWNER:
        raise ValueError("refusing unrelated/native-video dataset; use a separate image dataset name")
    if info.get("suite_id") != bundle.suite_id:
        raise ValueError("image dataset belongs to another review suite")
    if (info.get("importing_host") or info.get("host")) != host:
        raise RuntimeError("image dataset is bound to an importer on another host")
    if media_type not in (None, "image"):
        raise ValueError("image review cannot write to a native-video dataset")
    return info


def _require_owned_dataset(dataset: Any, bundle: ReviewBundle, host: str) -> dict[str, Any]:
    return _require_owned_metadata(dataset.info, dataset.media_type, bundle, host)


def _persisted_identity(snapshot: Any, dataset_name: str) -> tuple[Any, str, str, Any, str]:
    """Freeze the raw metadata fields that a fresh SDK load may not change."""
    if (
        not isinstance(snapshot, dict) or snapshot.get("_id") is None
        or snapshot.get("name") != dataset_name
        or not isinstance(snapshot.get("version"), str) or not snapshot["version"]
        or not isinstance(snapshot.get("info"), dict)
    ):
        raise ValueError("image dataset has invalid persisted identity metadata")
    return (
        snapshot["_id"], snapshot["name"], _json(snapshot["info"]),
        snapshot.get("media_type"), snapshot["version"],
    )


def import_scene_review_images(
    bundle: ReviewBundle, *, manifest_path: str | Path,
    scene_comparisons: dict[str, SceneComparisonReview], dataset_name: str,
    persistent: bool = True,
) -> Any:
    """Upsert a separately owned image dataset after full artifact verification.

    Each import revalidates the original videos, reports, PNG bytes and decoded
    pixel digests before importing FiftyOne, then once more before publication.
    Report refreshes keep stable source-frame identities and manual notes, but
    only an explicit human decision bound to the *current* report hash counts
    as reviewed. Old/nonselected samples remain historical, excluded from all
    current-generation saved views. No source video or artifact is written.
    Invoke once in a fresh independent ``scene-images-import`` CLI process;
    an already imported or previously attempted FiftyOne SDK is refused.
    """
    if not isinstance(dataset_name, str) or not dataset_name or dataset_name != dataset_name.strip():
        raise ValueError("dataset_name must be nonempty without surrounding whitespace")
    if not isinstance(scene_comparisons, dict):
        raise ValueError("scene_comparisons must be a clip-ID mapping")
    bundle = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    comparisons = dict(scene_comparisons)
    # Do not resolve aliases here: the loader must see and reject symlink or
    # reparse components in the caller's original path before accepting it.
    path = Path(os.path.abspath(Path(manifest_path).expanduser()))
    manifest = load_scene_review_images(path, bundle=bundle, scene_comparisons=comparisons)
    manifest_snapshot = _digest(manifest)
    manifest_sha = file_sha256(path)
    generation = _digest({
        "sample_schema_revision": _SAMPLE_SCHEMA_REVISION,
        "manifest_sha256": manifest_sha, "artifact_id": manifest["artifact_id"],
    })
    clips = {clip.clip_id: clip for clip in bundle.clips}
    media_identities = {clip.clip_id: _media_identity(clip) for clip in bundle.clips}
    # Account for every current clip, not only clips with selected images or a
    # comparison report. Zero-selection sources must also retain assignments.
    current_registry = _source_registry({clip.clip_id: (
        clip.video_sha256, clip.camera_id, clip.source_session_id,
        clip.split, media_identities[clip.clip_id],
    ) for clip in bundle.clips})
    summaries = {
        clip_id: _json(report["summary"]) for clip_id, report in manifest["reports"].items()
    }
    report_info = {clip_id: {
        key: report[key] for key in ("report_sha256", "report_status", "profile_id")
    } for clip_id, report in manifest["reports"].items()}
    if len(_json(report_info).encode("utf-8")) > _MAX_INFO_REPORT_BYTES:
        raise ValueError("too many reports for one image dataset; split the review bundle")
    # Materialize/serialize all managed observations before any optional SDK/DB
    # action. Loader validation is authoritative; no native-video FPS gate or
    # video player API is invoked on this independent image-only path.
    prepared: list[tuple[dict[str, Any], str]] = []
    for image in manifest["images"]:
        fields = _image_fields(
            bundle, clips[image["clip_id"]], image, manifest["reports"][image["clip_id"]],
            manifest, manifest_sha256=manifest_sha, generation=generation,
            media_identity=media_identities[image["clip_id"]],
            summary_json=summaries[image["clip_id"]],
        )
        prepared.append((image, fields["argus_scene_image_key"]))
    # Retain only each original row plus its small key. The potentially large
    # report summary is cached once per clip, not once per selected image.
    if prepared:
        del fields
    fo, datasets = _load_review_sdk()
    host = socket.gethostname()
    with (
        _dataset_import_lock(dataset_name),
        _database_import_lock(datasets, dataset_name) as lease,
    ):
        snapshot = datasets.find_one({"name": dataset_name}, _DATASET_PROJECTION)
        if snapshot is not None:
            if not isinstance(snapshot, dict):
                raise ValueError("image dataset has invalid persisted identity metadata")
            # Inspect persisted ownership before load_dataset can migrate or
            # otherwise save any target metadata. Never trust a cached marker.
            info = _require_owned_metadata(
                snapshot.get("info"), snapshot.get("media_type"), bundle, host,
            )
            _load_source_registry(info)
            identity_before = _persisted_identity(snapshot, dataset_name)
            if (
                not isinstance(getattr(fo, "__version__", None), str)
                or snapshot["version"] != fo.__version__
            ):
                raise ValueError(
                    "image dataset SDK version differs; automatic migration is disabled; "
                    "use a separately migrated dataset"
                )
            dataset = fo.load_dataset(dataset_name)
            snapshot_after = datasets.find_one({"name": dataset_name}, _DATASET_PROJECTION)
            try:
                identity_after = _persisted_identity(snapshot_after, dataset_name)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "image dataset persisted identity changed while loading; retry in a "
                    "fresh independent scene-images-import CLI process"
                ) from exc
            if (
                identity_after != identity_before
                or dataset.name != dataset_name
                or dataset.version != identity_before[4]
                or dataset.media_type != identity_before[3]
                or _json(dataset.info) != identity_before[2]
            ):
                raise RuntimeError(
                    "image dataset persisted identity changed while loading; retry in a "
                    "fresh independent scene-images-import CLI process"
                )
            info = _require_owned_dataset(dataset, bundle, host)
        else:
            dataset = fo.Dataset(name=dataset_name, persistent=persistent)
            info = {
                "owner": _OWNER, "suite_id": bundle.suite_id, "host": host,
                "importing_host": host, "state": "failed",
                "error": "initial image import not completed",
                # Commit the full first-bundle identity ledger together with
                # ownership, even if no sample can subsequently be written.
                _SOURCE_REGISTRY_KEY: current_registry,
                _SOURCE_REGISTRY_REQUIRED_KEY: True,
            }
            dataset.info[_INFO_KEY] = info
            dataset.save()
        existing, sources = _existing_samples(
            dataset, bundle, media_identities, stored_sources=_load_source_registry(info),
        )
        source_registry = _source_registry(sources)
        for image, key in prepared:
            sample = existing.get(key)
            if sample is not None and sample.get_field("argus_frame_sha256") != image["frame_sha256"]:
                raise ValueError("original-frame pixels changed; use a new image dataset")
        dataset.info[_INFO_KEY] = {
            **info, "state": "importing", "host": host, "importing_host": host,
            "bundle_id": bundle.bundle_id, "pending_generation": generation, "error": None,
            # Persist assignments with the importing state before any sample
            # writes. Failure cannot release a source for a different split.
            _SOURCE_REGISTRY_KEY: source_registry,
            _SOURCE_REGISTRY_REQUIRED_KEY: True,
        }
        dataset.save()
        touched: list[Any] = []
        try:
            dataset.media_type = "image"
            _ensure_schema(dataset, fo)
            for image, key in prepared:
                fields = _image_fields(
                    bundle, clips[image["clip_id"]], image,
                    manifest["reports"][image["clip_id"]], manifest,
                    manifest_sha256=manifest_sha, generation=generation,
                    media_identity=media_identities[image["clip_id"]],
                    summary_json=summaries[image["clip_id"]],
                )
                sample = existing.get(key)
                is_new = sample is None
                if is_new:
                    sample = fo.Sample(filepath=str(path.parent / image["image_path"]))
                touched.append(sample)
                if not is_new:
                    sample["argus_import_state"] = "pending"
                    sample.save()
                    sample.filepath = str(path.parent / image["image_path"])
                for name, value in fields.items():
                    sample[name] = value
                clip = clips[image["clip_id"]]
                sample.metadata = fo.ImageMetadata(width=clip.width, height=clip.height, num_channels=3)
                if is_new:
                    dataset.add_sample(sample)
                else:
                    sample.save()
            verified = load_scene_review_images(path, bundle=bundle, scene_comparisons=comparisons)
            if _digest(verified) != manifest_snapshot or file_sha256(path) != manifest_sha:
                raise ValueError("image artifact changed while importing")
            for sample in touched:
                sample["argus_import_state"] = "complete"
                sample.save()
            _save_views(dataset, fo, generation)
            dataset.info[_INFO_KEY] = {
                **dataset.info[_INFO_KEY], "state": "complete", "generation": generation,
                "pending_generation": None, "manifest_path": str(path),
                "manifest_sha256": manifest_sha, "artifact_id": manifest["artifact_id"],
                "image_count": len(prepared), "report_count": len(report_info),
                "reports": report_info, "selection": manifest["selection"],
                "report_summary_location": "artifact manifest and argus_scene_comparison_summary_json",
                "diagnostic_only": True, "gating_effect": "none",
                "release_authority": False, "ground_truth_authority": False,
            }
            dataset.save()
        except Exception as exc:
            if lease.record_failure(exc):
                # No compensation can prove an earlier opaque SDK write was
                # cancelled. Stop issuing writes and let the lease retain its
                # mutex until the server-side outcome is reconciled.
                raise
            for sample in touched:
                try:
                    sample["argus_import_state"] = "pending"
                    sample.save()
                except BaseException as compensation_error:
                    if lease.record_failure(compensation_error):
                        break
            if not lease.uncertain:
                try:
                    dataset.info[_INFO_KEY] = {
                        **dataset.info[_INFO_KEY], "state": "failed",
                        "error": _import_error_summary(exc),
                    }
                    dataset.save()
                except BaseException as compensation_error:
                    lease.record_failure(compensation_error)
                    # Keep the primary failure. The lease also remembers a
                    # swallowed uncertain compensation and must retain its lock.
            raise
    return dataset


__all__ = ["import_scene_review_images"]
