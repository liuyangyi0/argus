"""Frame identity metadata shared across capture, pipeline, and evidence."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass
class FrameIdentity:
    """Identifies one captured frame across runtime surfaces."""

    camera_id: str
    source_declared: str
    source_runtime: str
    stream_id: str | None
    frame_index: int
    capture_wall_ts: float
    capture_monotonic_ts: float
    pipeline_ingest_ts: float | None = None
    preview_stream_ts: float | None = None
    snapshot_id: str | None = None

    def with_updates(self, **updates: Any) -> "FrameIdentity":
        """Return a copy with selected fields replaced."""
        return replace(self, **updates)

    @property
    def runtime_source_hash(self) -> str:
        """Short stable hash for overlays and low-leak diagnostics."""
        return hashlib.sha1(self.source_runtime.encode("utf-8")).hexdigest()[:10]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe primitives."""
        data = asdict(self)
        data["runtime_source_hash"] = self.runtime_source_hash
        return data


def build_frame_identity(
    *,
    camera_id: str,
    source_declared: str,
    source_runtime: str,
    stream_id: str | None,
    frame_index: int,
    capture_wall_ts: float | None = None,
    capture_monotonic_ts: float | None = None,
) -> FrameIdentity:
    """Create a frame identity with current timestamps by default."""
    return FrameIdentity(
        camera_id=camera_id,
        source_declared=source_declared,
        source_runtime=source_runtime,
        stream_id=stream_id,
        frame_index=frame_index,
        capture_wall_ts=time.time() if capture_wall_ts is None else capture_wall_ts,
        capture_monotonic_ts=(
            time.monotonic()
            if capture_monotonic_ts is None
            else capture_monotonic_ts
        ),
    )
