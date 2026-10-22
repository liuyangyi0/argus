"""Model snapshots restore durable commands without widening operation RBAC."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from starlette.requests import Request

from argus.dashboard.routes.models import list_models


def _request(role):
    return Request({
        "type": "http", "method": "GET", "path": "/api/models/json",
        "headers": [], "app": SimpleNamespace(state=SimpleNamespace()),
        "state": {"user": {"username": "reader", "role": role}},
    })


def _model(version):
    return SimpleNamespace(
        model_version_id=version,
        to_dict=lambda: {"model_version_id": version, "stage": "candidate"},
    )


def test_model_list_restores_latest_operation_for_every_stage_in_one_lookup():
    registry = Mock()
    registry.list_models.return_value = [_model("a"), _model("b")]
    operation = {
        "operation_id": "pending-shadow", "model_version_id": "a",
        "target_stage": "shadow", "status": "pending",
    }
    service = Mock()
    service.get_latest_operations.return_value = {
        "a": SimpleNamespace(to_dict=lambda: operation),
    }
    with (
        patch("argus.dashboard.routes.models._get_registry", return_value=registry),
        patch("argus.dashboard.routes.models._get_release_service", return_value=service),
        patch("argus.dashboard.routes.models._model_runtime_state", return_value="waiting"),
    ):
        response = list_models(_request("engineer"))
    body = json.loads(response.body)["data"]
    assert body["models"][0]["latest_release_operation"] == operation
    assert body["models"][1]["latest_release_operation"] is None
    service.get_latest_operations.assert_called_once_with(["a", "b"])


def test_read_only_model_viewer_does_not_gain_release_operation_details():
    registry = Mock()
    registry.list_models.return_value = [_model("a")]
    with (
        patch("argus.dashboard.routes.models._get_registry", return_value=registry),
        patch("argus.dashboard.routes.models._get_release_service") as get_service,
        patch("argus.dashboard.routes.models._model_runtime_state", return_value="waiting"),
    ):
        response = list_models(_request("viewer"))
    assert json.loads(response.body)["data"]["models"][0]["latest_release_operation"] is None
    get_service.assert_not_called()


def test_operation_lookup_failure_is_not_misrepresented_as_no_pending_command():
    registry = Mock()
    registry.list_models.return_value = [_model("a")]
    service = Mock()
    service.get_latest_operations.side_effect = RuntimeError("database unavailable")
    with (
        patch("argus.dashboard.routes.models._get_registry", return_value=registry),
        patch("argus.dashboard.routes.models._get_release_service", return_value=service),
    ):
        response = list_models(_request("engineer"))
    assert response.status_code == 500
    assert json.loads(response.body)["code"] != 0
