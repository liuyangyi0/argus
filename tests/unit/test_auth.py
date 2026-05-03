"""Tests for the dashboard authentication system."""

import time

import pytest

from argus.dashboard.auth import (
    PERMISSION_MAP,
    create_session_token,
    current_username,
    has_permission,
    hash_password,
    require_role,
    verify_password,
    verify_session_token,
)


# ── Password hashing ──


class TestPasswordHashing:
    def test_hash_and_verify(self):
        """Hashed password should verify correctly."""
        pw = "correct-horse-battery-staple"
        hashed = hash_password(pw)
        assert verify_password(pw, hashed) is True

    def test_wrong_password_fails(self):
        """Wrong password should not verify."""
        hashed = hash_password("real-password")
        assert verify_password("wrong-password", hashed) is False

    def test_different_hashes_for_same_password(self):
        """Each call should produce a different hash (random salt)."""
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2
        # Both should still verify
        assert verify_password("same-password", h1) is True
        assert verify_password("same-password", h2) is True

    def test_empty_password(self):
        """Empty password should hash and verify like any other string."""
        hashed = hash_password("")
        assert verify_password("", hashed) is True
        assert verify_password("notempty", hashed) is False

    def test_malformed_stored_hash_returns_false(self):
        """Garbage stored hash should not raise, just return False."""
        assert verify_password("pw", "not-a-valid-hash") is False
        assert verify_password("pw", "") is False
        assert verify_password("pw", "::::") is False


# ── Session tokens ──


class TestSessionTokens:
    SECRET = "test-secret-key-12345"

    def test_create_and_verify_token(self):
        """Token round-trip: create then verify returns user dict."""
        token = create_session_token("alice", "operator", self.SECRET)
        result = verify_session_token(token, self.SECRET)
        assert result is not None
        assert result["username"] == "alice"
        assert result["role"] == "operator"

    def test_expired_token_rejected(self):
        """Token older than max_age should be rejected."""
        token = create_session_token("bob", "viewer", self.SECRET)
        # Verify with max_age=0 so it's immediately expired
        result = verify_session_token(token, self.SECRET, max_age=0)
        assert result is None

    def test_tampered_token_rejected(self):
        """Modified token payload should fail signature check."""
        token = create_session_token("charlie", "admin", self.SECRET)
        # Flip a character in the middle of the base64
        chars = list(token)
        mid = len(chars) // 2
        chars[mid] = "A" if chars[mid] != "A" else "B"
        tampered = "".join(chars)
        result = verify_session_token(tampered, self.SECRET)
        assert result is None

    def test_wrong_secret_rejected(self):
        """Token verified with a different secret should fail."""
        token = create_session_token("dave", "engineer", self.SECRET)
        result = verify_session_token(token, "wrong-secret")
        assert result is None

    def test_garbage_token_returns_none(self):
        """Completely invalid token string should return None."""
        assert verify_session_token("garbage!!!", self.SECRET) is None
        assert verify_session_token("", self.SECRET) is None

    def test_large_max_age_accepts_recent_token(self):
        """Token within max_age window should be accepted."""
        token = create_session_token("eve", "viewer", self.SECRET)
        result = verify_session_token(token, self.SECRET, max_age=86400)
        assert result is not None
        assert result["username"] == "eve"


# ── Permission system (RBAC) ──


class TestPermissionMap:
    def test_viewer_has_read_permissions(self):
        """Viewer role should have all read_* permissions."""
        perms = PERMISSION_MAP["viewer"]
        assert "read_alerts" in perms
        assert "read_cameras" in perms
        assert "read_system" in perms
        assert "read_models" in perms
        assert "read_degradation" in perms

    def test_viewer_cannot_handle_alerts(self):
        """Viewer should not have write/action permissions."""
        perms = PERMISSION_MAP["viewer"]
        assert "handle_alerts" not in perms
        assert "edit_zones" not in perms
        assert "edit_config" not in perms

    def test_operator_inherits_viewer(self):
        """Operator should include all viewer permissions plus extras."""
        viewer_perms = set(PERMISSION_MAP["viewer"])
        operator_perms = set(PERMISSION_MAP["operator"])
        assert viewer_perms.issubset(operator_perms)
        # Operator extras
        assert "handle_alerts" in operator_perms
        assert "shift_handoff" in operator_perms

    def test_engineer_inherits_operator(self):
        """Engineer should include all operator permissions plus extras."""
        operator_perms = set(PERMISSION_MAP["operator"])
        engineer_perms = set(PERMISSION_MAP["engineer"])
        assert operator_perms.issubset(engineer_perms)
        assert "edit_zones" in engineer_perms
        assert "manage_training" in engineer_perms

    def test_admin_has_wildcard(self):
        """Admin role should have wildcard '*' permission."""
        assert "*" in PERMISSION_MAP["admin"]

    def test_four_roles_exist(self):
        """Exactly four roles should be defined."""
        assert set(PERMISSION_MAP.keys()) == {"viewer", "operator", "engineer", "admin"}


# ── has_permission / require_role helpers ──


class _FakeRequest:
    """Minimal request-like object for testing permission helpers."""

    def __init__(self, user: dict | None):
        self.state = type("State", (), {"user": user})()


class TestHasPermission:
    def test_admin_has_any_permission(self):
        """Admin (wildcard) should pass any permission check."""
        req = _FakeRequest({"username": "root", "role": "admin"})
        assert has_permission(req, "read_alerts") is True
        assert has_permission(req, "edit_config") is True
        assert has_permission(req, "nonexistent_perm") is True

    def test_viewer_has_read_permission(self):
        req = _FakeRequest({"username": "v", "role": "viewer"})
        assert has_permission(req, "read_alerts") is True

    def test_viewer_lacks_write_permission(self):
        req = _FakeRequest({"username": "v", "role": "viewer"})
        assert has_permission(req, "edit_zones") is False

    def test_no_user_returns_false(self):
        req = _FakeRequest(None)
        assert has_permission(req, "read_alerts") is False

    def test_unknown_role_returns_false(self):
        req = _FakeRequest({"username": "x", "role": "superuser"})
        assert has_permission(req, "read_alerts") is False


class TestRequireRole:
    def test_matching_role(self):
        req = _FakeRequest({"username": "a", "role": "operator"})
        assert require_role(req, "operator", "admin") is True

    def test_non_matching_role(self):
        req = _FakeRequest({"username": "a", "role": "viewer"})
        assert require_role(req, "operator", "admin") is False

    def test_no_user_returns_false(self):
        req = _FakeRequest(None)
        assert require_role(req, "admin") is False


class TestCurrentUsername:
    """Audit logs must read username from request.state.user, never hardcode."""

    def test_returns_username_when_logged_in(self):
        req = _FakeRequest({"username": "alice", "role": "operator"})
        assert current_username(req) == "alice"

    def test_returns_unknown_for_anonymous(self):
        req = _FakeRequest(None)
        assert current_username(req) == "unknown"

    def test_returns_unknown_when_username_missing(self):
        req = _FakeRequest({"role": "operator"})
        assert current_username(req) == "unknown"

    def test_returns_unknown_when_username_empty(self):
        req = _FakeRequest({"username": "", "role": "operator"})
        assert current_username(req) == "unknown"

    def test_returns_unknown_when_state_user_is_not_dict(self):
        req = _FakeRequest("some_string_garbage")  # robustness against bad middleware
        assert current_username(req) == "unknown"
