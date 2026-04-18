"""End-to-end integration test: Alert.model_version_id must survive dispatch to the DB.

Plan ref: e2e-mainline-v1 item 1.4 — "告警 model_version_id 贯穿 smoke test".
Prevents the "model swapped but alert still tagged with old version" data-lineage
hole that would make post-hoc audits ("which model produced this alert?")
impossible.

Covers:
- AlertRecord row persists the model_version_id that was set on the Alert
- Switching the active model via the registry causes subsequent alerts to pick
  up the new version id (no stale state)
- Alerts with no model_version_id (e.g. SSIM / simplex fallback path) stay NULL
- Database.save_alert directly accepts and persists the field (API regression guard)

Uses the dispatcher's synchronous ``_dispatch_database`` path so the background
db-worker thread isn't required — the assertion target is the pipeline
    Alert → AlertDispatcher → Database.save_alert → AlertRecord
and the queue is orthogonal.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from argus.alerts.dispatcher import AlertDispatcher
from argus.alerts.grader import Alert, AlertSeverity
from argus.config.schema import AlertConfig
from argus.storage.database import Database
from argus.storage.model_registry import ModelRegistry


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    db = Database(f"sqlite:///{tmp_path / 'alerts.db'}")
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def dispatcher(tmp_db: Database, tmp_path: Path) -> AlertDispatcher:
    """Dispatcher with background threads disabled for deterministic tests."""
    d = AlertDispatcher(
        config=AlertConfig(),
        database=tmp_db,
        alerts_dir=str(tmp_path / "alerts"),
    )
    yield d
    d.close()


def _register_and_activate(
    registry: ModelRegistry,
    *,
    camera_id: str,
    model_path: Path,
    baseline_dir: Path,
) -> str:
    """Register a candidate, then activate it. Returns the version id."""
    model_version_id = registry.register(
        model_path=str(model_path),
        baseline_dir=str(baseline_dir),
        camera_id=camera_id,
        model_type="patchcore",
        training_params={"image_size": 256, "export_format": "openvino"},
    )
    registry.activate(model_version_id, allow_bypass=True)
    return model_version_id


def _make_alert(
    alert_id: str,
    *,
    camera_id: str = "cam_01",
    model_version_id: str | None = None,
) -> Alert:
    return Alert(
        alert_id=alert_id,
        camera_id=camera_id,
        zone_id="default",
        severity=AlertSeverity.MEDIUM,
        anomaly_score=0.75,
        timestamp=time.time(),
        frame_number=0,
        model_version_id=model_version_id,
    )


def test_alert_carries_active_model_version_to_db(
    tmp_db: Database, dispatcher: AlertDispatcher, tmp_path: Path
):
    """Dispatcher must forward Alert.model_version_id into the AlertRecord row."""
    registry = ModelRegistry(session_factory=tmp_db.get_session)
    v1 = _register_and_activate(
        registry,
        camera_id="cam_01",
        model_path=tmp_path / "m_v1.xml",
        baseline_dir=tmp_path / "baseline_v1",
    )

    dispatcher._dispatch_database(_make_alert("A-1", model_version_id=v1), None, None)

    rows = tmp_db.get_alerts(limit=10)
    assert len(rows) == 1
    assert rows[0].alert_id == "A-1"
    assert rows[0].model_version_id == v1


def test_model_switch_preserves_historical_lineage(
    tmp_db: Database, dispatcher: AlertDispatcher, tmp_path: Path
):
    """After swapping the active model, old alerts keep v1, new alerts carry v2."""
    registry = ModelRegistry(session_factory=tmp_db.get_session)
    v1 = _register_and_activate(
        registry,
        camera_id="cam_01",
        model_path=tmp_path / "m_v1.xml",
        baseline_dir=tmp_path / "baseline_v1",
    )
    dispatcher._dispatch_database(_make_alert("A-1", model_version_id=v1), None, None)

    v2 = _register_and_activate(
        registry,
        camera_id="cam_01",
        model_path=tmp_path / "m_v2.xml",
        baseline_dir=tmp_path / "baseline_v2",
    )
    assert v2 != v1
    dispatcher._dispatch_database(_make_alert("A-2", model_version_id=v2), None, None)

    by_id = {r.alert_id: r for r in tmp_db.get_alerts(limit=10)}
    assert by_id["A-1"].model_version_id == v1
    assert by_id["A-2"].model_version_id == v2


def test_alert_without_model_version_stays_null(
    tmp_db: Database, dispatcher: AlertDispatcher
):
    """Simplex / SSIM fallback path: no model version → DB row stays NULL."""
    dispatcher._dispatch_database(_make_alert("A-FB", model_version_id=None), None, None)

    rows = tmp_db.get_alerts(limit=10)
    assert len(rows) == 1
    assert rows[0].model_version_id is None


def test_save_alert_persists_model_version_id_directly(tmp_db: Database):
    """DB layer guard: save_alert accepts and persists model_version_id."""
    tmp_db.save_alert(
        alert_id="A-DB",
        timestamp=datetime.now(timezone.utc),
        camera_id="cam_01",
        zone_id="default",
        severity="medium",
        anomaly_score=0.5,
        model_version_id="v-direct-test",
    )
    rows = tmp_db.get_alerts(limit=1)
    assert rows[0].model_version_id == "v-direct-test"
