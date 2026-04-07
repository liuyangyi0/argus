"""Tests for the 4-tier RBAC permission matrix (UX v2 §11.1)."""

import pytest

from argus.dashboard.auth import (
    PERMISSION_MAP,
    VALID_ROLES,
    get_denied_response,
    has_permission,
)


class _FakeRequest:
    """Minimal request mock for permission tests."""

    def __init__(self, role: str | None = None):
        self.state = type("State", (), {})()
        if role is not None:
            self.state.user = {"username": "test", "role": role}
        else:
            self.state.user = None


class TestPermissionMap:
    def test_four_roles_exist(self):
        assert VALID_ROLES == {"viewer", "operator", "engineer", "admin"}

    def test_viewer_can_read(self):
        req = _FakeRequest("viewer")
        assert has_permission(req, "read_alerts") is True
        assert has_permission(req, "read_cameras") is True
        assert has_permission(req, "read_system") is True

    def test_viewer_cannot_write(self):
        req = _FakeRequest("viewer")
        assert has_permission(req, "handle_alerts") is False
        assert has_permission(req, "edit_zones") is False
        assert has_permission(req, "manage_models") is False

    def test_operator_can_handle_alerts(self):
        req = _FakeRequest("operator")
        assert has_permission(req, "handle_alerts") is True
        assert has_permission(req, "shift_handoff") is True
        assert has_permission(req, "mute_audio") is True
        assert has_permission(req, "view_replay") is True

    def test_operator_cannot_manage(self):
        req = _FakeRequest("operator")
        assert has_permission(req, "edit_zones") is False
        assert has_permission(req, "manage_training") is False
        assert has_permission(req, "manage_models") is False

    def test_engineer_has_management_permissions(self):
        req = _FakeRequest("engineer")
        assert has_permission(req, "handle_alerts") is True
        assert has_permission(req, "edit_zones") is True
        assert has_permission(req, "edit_thresholds") is True
        assert has_permission(req, "manage_baselines") is True
        assert has_permission(req, "manage_training") is True
        assert has_permission(req, "manage_models") is True
        assert has_permission(req, "rollback_model") is True

    def test_admin_has_wildcard(self):
        req = _FakeRequest("admin")
        assert has_permission(req, "handle_alerts") is True
        assert has_permission(req, "manage_models") is True
        assert has_permission(req, "any_random_permission") is True

    def test_no_user_denied(self):
        req = _FakeRequest(None)
        assert has_permission(req, "read_alerts") is False

    def test_unknown_role_denied(self):
        req = _FakeRequest("superuser")
        assert has_permission(req, "read_alerts") is False


class TestDeniedResponse:
    def test_denied_response_format(self):
        req = _FakeRequest("viewer")
        resp = get_denied_response(req, "edit_zones")
        assert resp["error"] == "权限不足"
        assert resp["required_permission"] == "edit_zones"
        assert resp["your_role"] == "viewer"
        assert "viewer" in resp["message"]
