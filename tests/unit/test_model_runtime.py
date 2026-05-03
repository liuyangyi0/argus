"""Tests for model_runtime activate/rollback (P1 fix 2026-05).

Activation must respect the candidate→shadow→canary→production gate
unless the caller explicitly opts in to ``allow_bypass=True``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from argus.dashboard.model_runtime import activate_model_version


class _FakeRequest:
    def __init__(self, registry, camera_manager=None):
        self.app = SimpleNamespace(state=SimpleNamespace(
            db=SimpleNamespace(get_session=MagicMock()),
            camera_manager=camera_manager,
        ))


@pytest.fixture
def registry_factory():
    """Patch get_registry to return a controllable registry mock."""
    with patch("argus.dashboard.model_runtime.get_registry") as get:
        def _make(record=None, activate_side_effect=None):
            registry = MagicMock()
            if activate_side_effect is not None:
                registry.activate.side_effect = activate_side_effect
            registry.get_by_version_id.return_value = record
            get.return_value = registry
            return registry
        yield _make


def _record(version="v1", camera_id="cam_01", model_path=None):
    rec = MagicMock()
    rec.model_version_id = version
    rec.camera_id = camera_id
    rec.model_path = model_path or ""
    return rec


class TestActivateModelVersion:
    def test_default_does_not_bypass_stage_gate(self, registry_factory):
        registry = registry_factory(record=_record())
        req = _FakeRequest(registry)

        activate_model_version(req, "v1", triggered_by="dashboard")

        # P1 fix: allow_bypass defaults to False.
        registry.activate.assert_called_once_with(
            "v1", triggered_by="dashboard", allow_bypass=False,
        )

    def test_explicit_bypass_passes_through(self, registry_factory):
        """Admin emergency path: caller opts in explicitly."""
        registry = registry_factory(record=_record())
        req = _FakeRequest(registry)

        activate_model_version(
            req, "v1", triggered_by="admin_force", allow_bypass=True,
        )

        registry.activate.assert_called_once_with(
            "v1", triggered_by="admin_force", allow_bypass=True,
        )

    def test_stage_gate_rejection_propagates_value_error(self, registry_factory):
        """Routes (e.g. /models/{id}/activate) catch this and surface a 400
        so the dashboard can prompt the user to walk the release pipeline."""
        registry_factory(
            record=_record(),
            activate_side_effect=ValueError(
                "Cannot activate model at stage 'candidate'."
            ),
        )
        req = _FakeRequest(MagicMock())

        with pytest.raises(ValueError, match="candidate"):
            activate_model_version(req, "v1")

    def test_database_unavailable_raises(self):
        with patch("argus.dashboard.model_runtime.get_registry", return_value=None):
            req = _FakeRequest(MagicMock())
            with pytest.raises(ValueError, match="Database not available"):
                activate_model_version(req, "v1")
