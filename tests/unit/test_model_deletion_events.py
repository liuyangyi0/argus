"""Deletion notifications follow the committed registry mutation, never preflight."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from starlette.requests import Request

from argus.dashboard.routes.models import delete_model
from argus.storage.model_registry import ModelDeletionConflictError


def _request(ws_manager):
    return Request({
        "type": "http", "method": "DELETE", "path": "/api/models/candidate",
        "headers": [], "app": SimpleNamespace(state=SimpleNamespace(ws_manager=ws_manager)),
        "state": {"user": {"username": "operator", "role": "engineer"}},
    })


def _registry():
    registry = Mock()
    registry.get_by_version_id.return_value = SimpleNamespace(
        camera_id="cam1", stage="candidate", is_active=False, release_revision=7,
    )
    return registry


def test_model_deleted_event_is_published_only_after_registry_commit_returns():
    registry, ws_manager = _registry(), Mock()
    order = []
    result = {"deleted": "candidate", "artifacts_retained": True, "garbage_collection": "deferred"}

    def commit(*args, **kwargs):
        order.append("committed")
        return result

    def publish(topic, data, **metadata):
        assert order == ["committed"]
        order.append("published")
        assert topic == "models"
        assert data == {
            "type": "model.deleted", "event": "model.deleted", "deleted": True,
            "model_version_id": "candidate", "camera_id": "cam1", "previous_stage": "candidate",
            "previous_release_revision": 7, "artifacts_retained": True,
            "garbage_collection": "deferred",
        }
        assert metadata == {"event_type": "model.deleted", "resource_id": "candidate"}
        return True

    registry.delete_model.side_effect = commit
    ws_manager.broadcast.side_effect = publish
    with patch("argus.dashboard.routes.models._get_registry", return_value=registry):
        response = delete_model(_request(ws_manager), "candidate")

    assert json.loads(response.body)["data"] == result
    assert order == ["committed", "published"]
    registry.delete_model.assert_called_once_with(
        "candidate", requested_by="operator", expected_revision=7,
    )


@pytest.mark.parametrize("failure,status", [
    (ModelDeletionConflictError("pending release"), 409),
    (ValueError("not found"), 404),
])
def test_rejected_model_deletion_does_not_broadcast(failure, status):
    registry, ws_manager = _registry(), Mock()
    registry.delete_model.side_effect = failure
    with patch("argus.dashboard.routes.models._get_registry", return_value=registry):
        response = delete_model(_request(ws_manager), "candidate")
    assert response.status_code == status
    ws_manager.broadcast.assert_not_called()


def test_database_commit_failure_never_publishes_a_model_tombstone():
    registry, ws_manager = _registry(), Mock()
    registry.delete_model.side_effect = RuntimeError("commit failed")
    with (
        patch("argus.dashboard.routes.models._get_registry", return_value=registry),
        pytest.raises(RuntimeError, match="commit failed"),
    ):
        delete_model(_request(ws_manager), "candidate")
    ws_manager.broadcast.assert_not_called()


@pytest.mark.parametrize("failure", [False, RuntimeError("queue unavailable")])
def test_event_transport_failure_does_not_turn_committed_deletion_into_a_retry(failure):
    registry, ws_manager = _registry(), Mock()
    result = {"deleted": "candidate", "artifacts_retained": True, "garbage_collection": "deferred"}
    registry.delete_model.return_value = result
    if isinstance(failure, Exception):
        ws_manager.broadcast.side_effect = failure
    else:
        ws_manager.broadcast.return_value = failure
    with patch("argus.dashboard.routes.models._get_registry", return_value=registry):
        response = delete_model(_request(ws_manager), "candidate")
    assert response.status_code == 200
    assert json.loads(response.body)["data"] == result
    ws_manager.broadcast.assert_called_once()
