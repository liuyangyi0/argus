"""Dataset selection model for multi-version training (痛点 2).

Lets the user pick any combination of baseline captures
(camera × zone × version, optionally tagged by session_label) and have
the trainer merge them into a single training set.

Serialized as JSON in TrainingJobRecord.dataset_selection so the
frontend can post the user's picks and the scheduler can persist
auto-retraining strategies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class DatasetSelectionItem:
    """One baseline directory the user wants to include."""

    camera_id: str
    zone_id: str
    version: str
    session_label: str | None = None

    def relative_path(self) -> Path:
        """Return ``<camera_id>/<zone_id>/<version>``."""
        return Path(self.camera_id) / self.zone_id / self.version

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "version": self.version,
            **({"session_label": self.session_label} if self.session_label else {}),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "DatasetSelectionItem":
        try:
            camera_id = str(raw["camera_id"]).strip()
            zone_id = str(raw.get("zone_id", "default")).strip() or "default"
            version = str(raw["version"]).strip()
        except KeyError as e:
            raise ValueError(f"DatasetSelectionItem missing required field: {e.args[0]}")
        if not camera_id or not version:
            raise ValueError("DatasetSelectionItem requires non-empty camera_id and version")
        session_label = raw.get("session_label") or None
        return cls(
            camera_id=camera_id,
            zone_id=zone_id,
            version=version,
            session_label=session_label,
        )


@dataclass
class DatasetSelection:
    """Ordered set of baseline directories merged into one training run."""

    items: list[DatasetSelectionItem] = field(default_factory=list)
    total_frames: int | None = None  # informational; computed by frontend

    def __post_init__(self) -> None:
        # Reject empty selection — caller should fall back to single-version
        # path instead of constructing an empty DatasetSelection.
        if not self.items:
            raise ValueError("DatasetSelection requires at least one item")
        # Multi-camera training is not supported by ModelTrainer.train (which
        # is per-camera): enforce single camera_id here so callers fail fast.
        camera_ids = {item.camera_id for item in self.items}
        if len(camera_ids) != 1:
            raise ValueError(
                f"DatasetSelection cannot mix cameras (got {sorted(camera_ids)})"
            )

    @property
    def camera_id(self) -> str:
        return self.items[0].camera_id

    def iter_dirs(self, baselines_root: Path) -> Iterator[Path]:
        """Yield absolute paths for each selection item under ``baselines_root``."""
        for item in self.items:
            yield baselines_root / item.relative_path()

    def version_summary(self) -> str:
        """Short human label for logs/audit ("v001+v003 / 1234 frames")."""
        versions = "+".join(item.version for item in self.items)
        if self.total_frames is not None:
            return f"{versions} / {self.total_frames} frames"
        return versions

    def to_json(self) -> str:
        payload = {
            "items": [item.to_dict() for item in self.items],
        }
        if self.total_frames is not None:
            payload["total_frames"] = self.total_frames
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | None) -> "DatasetSelection | None":
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"DatasetSelection JSON parse error: {e}")
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, raw: dict) -> "DatasetSelection":
        if not isinstance(raw, dict):
            raise ValueError("DatasetSelection payload must be a dict")
        items_raw = raw.get("items")
        if not isinstance(items_raw, Iterable) or isinstance(items_raw, (str, bytes)):
            raise ValueError("DatasetSelection 'items' must be a list")
        items = [DatasetSelectionItem.from_dict(it) for it in items_raw]
        total_frames = raw.get("total_frames")
        if total_frames is not None:
            try:
                total_frames = int(total_frames)
            except (TypeError, ValueError):
                raise ValueError("DatasetSelection 'total_frames' must be an int")
            if total_frames < 0:
                raise ValueError("DatasetSelection 'total_frames' must be >= 0")
        return cls(items=items, total_frames=total_frames)
