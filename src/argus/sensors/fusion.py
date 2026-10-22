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

from dataclasses import dataclass, field
import threading
import time
from typing import Iterable

_MIN_MULTIPLIER = 0.1
_MAX_MULTIPLIER = 5.0
_DEFAULT_VALID_FOR_S = 60.0


@dataclass(frozen=True)
class SensorFusionDecision:
    """Rule output for a raw external sensor reading."""

    sensor_type: str
    value: float
    unit: str | None
    multiplier: float
    severity: str
    reason: str
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.multiplier != 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "sensor_type": self.sensor_type,
            "value": self.value,
            "unit": self.unit,
            "multiplier": self.multiplier,
            "severity": self.severity,
            "reason": self.reason,
            "active": self.active,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class SensorFusionHead:
    """Map typed sensor readings to bounded fusion multipliers."""

    gas_warning_ppm: float = 10.0
    gas_danger_ppm: float = 50.0
    pressure_drop_warning_kpa: float = 5.0
    pressure_drop_danger_kpa: float = 15.0
    humidity_warning_rh: float = 85.0
    humidity_danger_rh: float = 95.0
    temperature_warning_c: float = 70.0
    temperature_danger_c: float = 90.0
    warning_multiplier: float = 1.5
    danger_multiplier: float = 2.5

    @classmethod
    def from_config(cls, config: object | None) -> "SensorFusionHead":
        if config is None:
            return cls()
        return cls(
            gas_warning_ppm=float(getattr(config, "gas_warning_ppm", 10.0)),
            gas_danger_ppm=float(getattr(config, "gas_danger_ppm", 50.0)),
            pressure_drop_warning_kpa=float(
                getattr(config, "pressure_drop_warning_kpa", 5.0),
            ),
            pressure_drop_danger_kpa=float(
                getattr(config, "pressure_drop_danger_kpa", 15.0),
            ),
            humidity_warning_rh=float(getattr(config, "humidity_warning_rh", 85.0)),
            humidity_danger_rh=float(getattr(config, "humidity_danger_rh", 95.0)),
            temperature_warning_c=float(
                getattr(config, "temperature_warning_c", 70.0),
            ),
            temperature_danger_c=float(getattr(config, "temperature_danger_c", 90.0)),
            warning_multiplier=float(getattr(config, "warning_multiplier", 1.5)),
            danger_multiplier=float(getattr(config, "danger_multiplier", 2.5)),
        )

    def evaluate(
        self,
        sensor_type: str,
        value: float,
        unit: str | None = None,
    ) -> SensorFusionDecision:
        """Evaluate one raw sensor reading.

        Supported readings are intentionally explicit:
        gas concentration in ppm, positive pressure drop in kPa, relative
        humidity in percent, and temperature in Celsius.
        """
        normalized = sensor_type.strip().lower().replace("-", "_")
        value_f = float(value)
        if normalized in {
            "gas",
            "gas_ppm",
            "combustible_gas",
            "voc",
            "h2s",
            "co",
        }:
            return self._threshold_decision(
                sensor_type=normalized,
                value=value_f,
                unit=unit or "ppm",
                warning=self.gas_warning_ppm,
                danger=self.gas_danger_ppm,
                metric="gas_ppm",
            )
        if normalized in {
            "pressure_drop",
            "pressure_drop_kpa",
            "pressure_delta",
            "pressure_delta_kpa",
        }:
            drop = abs(value_f) if "delta" in normalized else value_f
            return self._threshold_decision(
                sensor_type=normalized,
                value=drop,
                unit=unit or "kPa",
                warning=self.pressure_drop_warning_kpa,
                danger=self.pressure_drop_danger_kpa,
                metric="pressure_drop_kpa",
            )
        if normalized in {"humidity", "relative_humidity", "rh"}:
            return self._threshold_decision(
                sensor_type=normalized,
                value=value_f,
                unit=unit or "%RH",
                warning=self.humidity_warning_rh,
                danger=self.humidity_danger_rh,
                metric="relative_humidity",
            )
        if normalized in {"temperature", "temperature_c", "temp_c"}:
            return self._threshold_decision(
                sensor_type=normalized,
                value=value_f,
                unit=unit or "C",
                warning=self.temperature_warning_c,
                danger=self.temperature_danger_c,
                metric="temperature_c",
            )
        return SensorFusionDecision(
            sensor_type=normalized,
            value=value_f,
            unit=unit,
            multiplier=1.0,
            severity="neutral",
            reason="unsupported_sensor_type",
            evidence={"supported": False},
        )

    def _threshold_decision(
        self,
        *,
        sensor_type: str,
        value: float,
        unit: str,
        warning: float,
        danger: float,
        metric: str,
    ) -> SensorFusionDecision:
        if value >= danger:
            severity = "danger"
            multiplier = self.danger_multiplier
            threshold = danger
        elif value >= warning:
            severity = "warning"
            multiplier = self.warning_multiplier
            threshold = warning
        else:
            severity = "neutral"
            multiplier = 1.0
            threshold = warning
        return SensorFusionDecision(
            sensor_type=sensor_type,
            value=value,
            unit=unit,
            multiplier=multiplier,
            severity=severity,
            reason=f"{metric}_{severity}",
            evidence={
                "metric": metric,
                "threshold": threshold,
                "warning_threshold": warning,
                "danger_threshold": danger,
            },
        )


@dataclass(frozen=True)
class _SignalEntry:
    multiplier: float
    expires_at: float
    metadata: dict[str, object] = field(default_factory=dict)


class SensorFusion:
    """Thread-safe TTL store mapping ``(camera_id, zone_id)`` to multipliers."""

    def __init__(
        self,
        enabled: bool = False,
        default_valid_for_s: float = _DEFAULT_VALID_FOR_S,
        head: SensorFusionHead | None = None,
    ) -> None:
        self.enabled = enabled
        self._default_valid_for_s = float(default_valid_for_s)
        self._head = head or SensorFusionHead()
        self._signals: dict[tuple[str, str], _SignalEntry] = {}
        self._lock = threading.Lock()

    # ── Public API ──

    def set_signal(
        self,
        camera_id: str,
        zone_id: str,
        multiplier: float,
        valid_for_s: float | None = None,
        metadata: dict[str, object] | None = None,
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
            self._signals[(camera_id, zone_id)] = _SignalEntry(
                multiplier=mult,
                expires_at=expires_at,
                metadata=dict(metadata or {}),
            )

    def set_reading(
        self,
        camera_id: str,
        zone_id: str,
        sensor_type: str,
        value: float,
        *,
        unit: str | None = None,
        source: str | None = None,
        valid_for_s: float | None = None,
    ) -> SensorFusionDecision:
        """Evaluate a typed sensor reading and store an active fusion signal."""
        decision = self._head.evaluate(sensor_type, value, unit)
        if not decision.active:
            self.remove_signal(camera_id, zone_id)
            return decision
        metadata = {
            "source": source,
            "sensor_type": decision.sensor_type,
            "severity": decision.severity,
            "reason": decision.reason,
            "value": decision.value,
            "unit": decision.unit,
            "evidence": decision.evidence,
        }
        self.set_signal(
            camera_id,
            zone_id,
            decision.multiplier,
            valid_for_s=valid_for_s,
            metadata=metadata,
        )
        return decision

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
                if current < entry.expires_at:
                    return entry.multiplier
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
            for (cam, zone), entry in self._signals.items():
                remaining = entry.expires_at - current
                if remaining <= 0:
                    expired.append((cam, zone))
                    continue
                payload: dict[str, object] = {
                    "camera_id": cam,
                    "zone_id": zone,
                    "multiplier": entry.multiplier,
                    "remaining_s": round(remaining, 3),
                }
                payload.update(entry.metadata)
                out.append(payload)
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
