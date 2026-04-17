"""Generic external-sensor fusion layer.

Holds short-lived ``(camera_id, zone_id) -> (multiplier, valid_until_ts)``
entries. The alert grader consults ``get_multiplier`` during severity
evaluation; external signal sources push entries via ``set_signal``.

The store is intentionally sensor-agnostic: a temperature probe, a
vibration sensor, a radiation monitor, or a lidar distance feed can all
share the same API. Callers encode their own interpretation into the
scalar multiplier (``1.0`` = neutral, ``>1`` = escalate, ``<1`` = damp).

Wildcards:
- ``(camera_id, "*")`` applies to every zone on a camera.
- ``("*", "*")`` applies globally.
``get_multiplier`` falls back through exact → camera-wide → global, and
returns ``1.0`` when nothing matches, the entry has expired, or fusion is
disabled.
"""

from __future__ import annotations

import threading
import time
from typing import Iterable

_MIN_MULTIPLIER = 0.1
_MAX_MULTIPLIER = 5.0
_DEFAULT_VALID_FOR_S = 60.0


class SensorFusion:
    """Thread-safe TTL store mapping ``(camera_id, zone_id)`` to multipliers."""

    def __init__(
        self,
        enabled: bool = False,
        default_valid_for_s: float = _DEFAULT_VALID_FOR_S,
    ) -> None:
        self.enabled = enabled
        self._default_valid_for_s = float(default_valid_for_s)
        self._signals: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = threading.Lock()

    # ── Public API ──

    def set_signal(
        self,
        camera_id: str,
        zone_id: str,
        multiplier: float,
        valid_for_s: float | None = None,
    ) -> None:
        """Upsert a signal. ``multiplier`` must lie in ``[0.1, 5.0]``.

        ``valid_for_s`` defaults to ``default_valid_for_s`` (60s unless
        otherwise configured). ``valid_for_s=0`` is allowed — the entry is
        stored but will be treated as expired on the very next read.
        """
        mult = float(multiplier)
        if not (_MIN_MULTIPLIER <= mult <= _MAX_MULTIPLIER):
            raise ValueError(
                f"multiplier must be in [{_MIN_MULTIPLIER}, {_MAX_MULTIPLIER}], "
                f"got {multiplier!r}"
            )
        ttl = self._default_valid_for_s if valid_for_s is None else float(valid_for_s)
        if ttl < 0:
            raise ValueError(f"valid_for_s must be non-negative, got {valid_for_s!r}")
        expires_at = time.monotonic() + ttl
        with self._lock:
            self._signals[(camera_id, zone_id)] = (mult, expires_at)

    def get_multiplier(
        self,
        camera_id: str,
        zone_id: str,
        now: float | None = None,
    ) -> float:
        """Look up the effective multiplier for a ``(camera_id, zone_id)`` pair.

        Returns ``1.0`` if fusion is disabled, the entry is missing, or
        the entry has expired. Falls back from exact match to camera-wide
        (``(camera_id, "*")``) to global (``("*", "*")``).
        """
        if not self.enabled:
            return 1.0
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            for key in (
                (camera_id, zone_id),
                (camera_id, "*"),
                ("*", "*"),
            ):
                entry = self._signals.get(key)
                if entry is None:
                    continue
                mult, expires_at = entry
                if current < expires_at:
                    return mult
                # Expired — drop lazily so the store self-heals without a sweeper thread
                self._signals.pop(key, None)
        return 1.0

    def remove_signal(self, camera_id: str, zone_id: str) -> bool:
        """Remove a single entry. Returns ``True`` if something was removed."""
        with self._lock:
            return self._signals.pop((camera_id, zone_id), None) is not None

    def active_signals(self, now: float | None = None) -> list[dict[str, object]]:
        """Return a snapshot of still-valid signals.

        Shape: ``[{camera_id, zone_id, multiplier, remaining_s}, ...]``.
        Expired entries are pruned in-place as a side effect.
        """
        current = time.monotonic() if now is None else float(now)
        out: list[dict[str, object]] = []
        with self._lock:
            expired: list[tuple[str, str]] = []
            for (cam, zone), (mult, expires_at) in self._signals.items():
                remaining = expires_at - current
                if remaining <= 0:
                    expired.append((cam, zone))
                    continue
                out.append({
                    "camera_id": cam,
                    "zone_id": zone,
                    "multiplier": mult,
                    "remaining_s": round(remaining, 3),
                })
            for key in expired:
                self._signals.pop(key, None)
        return out

    def clear(self, keys: Iterable[tuple[str, str]] | None = None) -> None:
        """Remove all signals, or just the subset in ``keys``."""
        with self._lock:
            if keys is None:
                self._signals.clear()
            else:
                for key in keys:
                    self._signals.pop(key, None)
