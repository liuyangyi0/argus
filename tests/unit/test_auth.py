"""Tests for the dashboard authentication system."""


from base64 import b64encode
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, AuthConfig
from argus.dashboard.app import _resolve_session_secret, create_app
from argus.dashboard.auth import (
    PERMISSION_MAP,
    AuthMiddleware,
    _safe_next_url,
    create_session_token,
    current_username,
    has_permission,
    hash_password,
    require_human_permission,
    require_role,
    verify_password,
    verify_session_token,
)
from argus.storage.database import Database


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


def _human_approval_client(*, enabled=True, username="reviewer", role="engineer", database=True):
    """Exercise the real auth middleware without starting runtime services."""
    app = FastAPI()
    db = Mock()
    db.get_user.return_value = SimpleNamespace(active=True, role=role)
    app.state.database = db if database else None
    app.add_middleware(
        AuthMiddleware, config=AuthConfig(enabled=enabled, api_token="shared-test-token"),
        session_secret="human-test-secret", database=app.state.database,
    )

    @app.post("/api/review-probe")
    async def probe(request: Request):
        return {
            "allowed": require_human_permission(request, "manage_training"),
            "method": request.state.auth_method,
            "username": current_username(request),
        }

    client = TestClient(app)
    session = create_session_token(username, role, "human-test-secret")
    return client, db, session


@pytest.mark.parametrize("method", ["header", "basic"])
def test_shared_credentials_cannot_be_used_as_human_approval(method):
    client, _db, _session = _human_approval_client()
    headers = {"X-API-Token": "shared-test-token"}
    if method == "basic":
        encoded = b64encode(b"reviewer:shared-test-token").decode()
        headers = {"Authorization": f"Basic {encoded}"}
    # A caller-supplied name/source must not change the server auth provenance.
    headers["X-Auth-Method"] = "session"
    response = client.post("/api/review-probe", headers=headers, json={"username": "reviewer"})
    assert response.status_code == 200
    assert response.json() == {"allowed": False, "method": "api_token", "username": "api"}


def test_auth_disabled_is_not_human_approval_even_with_a_signed_cookie():
    client, _db, session = _human_approval_client(enabled=False)
    client.cookies.set("argus_session", session)
    response = client.post("/api/review-probe")
    assert response.json() == {"allowed": False, "method": "disabled", "username": "system"}


@pytest.mark.parametrize("username", ["reviewer", "api", "system"])
@pytest.mark.parametrize("role", ["admin", "engineer"])
def test_human_approval_accepts_actual_accounts_not_username_heuristics(username, role):
    client, db, session = _human_approval_client(username=username, role=role)
    client.cookies.set("argus_session", session)
    response = client.post("/api/review-probe")
    assert response.json() == {"allowed": True, "method": "session", "username": username}
    db.get_user.assert_called_with(username)


def test_human_approval_requires_a_database_backed_identity():
    client, _db, session = _human_approval_client(database=False)
    client.cookies.set("argus_session", session)
    assert client.post("/api/review-probe").json()["allowed"] is False


def test_human_approval_rechecks_account_changes_after_middleware_authentication():
    client, db, session = _human_approval_client(role="admin")
    client.cookies.set("argus_session", session)
    db.get_user.side_effect = [
        SimpleNamespace(active=True, role="admin"),
        SimpleNamespace(active=True, role="viewer"),
    ]
    assert client.post("/api/review-probe").json()["allowed"] is False


@pytest.mark.parametrize("record", [None, SimpleNamespace(active=False, role="admin")])
def test_disabled_or_deleted_account_cannot_review(record):
    client, db, session = _human_approval_client(role="admin")
    client.cookies.set("argus_session", session)
    db.get_user.return_value = record
    assert client.post("/api/review-probe").status_code == 401


def test_human_approval_fails_closed_when_account_lookup_fails_after_authentication():
    client, db, session = _human_approval_client(role="admin")
    client.cookies.set("argus_session", session)
    db.get_user.side_effect = [SimpleNamespace(active=True, role="admin"), RuntimeError("db unavailable")]
    assert client.post("/api/review-probe").json()["allowed"] is False


# ── /api/me endpoint ──


@pytest.fixture
def _me_db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'me.db'}")
    database.initialize()
    yield database
    database.close()


def _build_auth_enabled_client(database: Database) -> tuple[TestClient, str]:
    """Build a TestClient with auth enabled and return (client, session_secret)."""
    config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="me-test-token"))
    app = create_app(database=database, config=config)
    client = TestClient(app)
    return client, app.state.session_secret


def test_session_secret_cannot_be_derived_from_shared_api_token(monkeypatch):
    monkeypatch.delenv("ARGUS_SESSION_SECRET", raising=False)
    config = AuthConfig(enabled=True, api_token="stable-api-token")

    first = _resolve_session_secret(config)
    second = _resolve_session_secret(config)

    assert first != second
    assert first != config.api_token
    assert len(first) == 64
    legacy_derived = hashlib.sha256(
        b"argus.dashboard.session.v1\0" + config.api_token.encode(),
    ).hexdigest()
    assert first != legacy_derived

    monkeypatch.setenv("ARGUS_SESSION_SECRET", "independent-session-material")
    overridden = _resolve_session_secret(config)
    assert overridden != first
    assert overridden != "independent-session-material"
    assert overridden == _resolve_session_secret(config)


@pytest.mark.parametrize("value", ["shared-test-token", "共享测试凭证"])
def test_session_secret_rejects_the_same_value_as_shared_api_token(monkeypatch, value):
    config = AuthConfig(enabled=True, api_token=value)
    monkeypatch.setenv("ARGUS_SESSION_SECRET", f" {value} ")
    with pytest.raises(ValueError, match="independent"):
        _resolve_session_secret(config)


def test_independent_unicode_session_secret_does_not_crash_comparison(monkeypatch):
    config = AuthConfig(enabled=True, api_token="共享测试凭证")
    monkeypatch.setenv("ARGUS_SESSION_SECRET", "独立的测试会话密钥")
    assert len(_resolve_session_secret(config)) == 64


def test_api_credential_cannot_forge_a_personal_account_cookie(_me_db, monkeypatch):
    monkeypatch.delenv("ARGUS_SESSION_SECRET", raising=False)
    _me_db.create_user("reviewer", hash_password("reviewer-password"), "engineer", None)
    client, _actual_secret = _build_auth_enabled_client(_me_db)
    # Reproduce the old derivation using only the shared API credential. This
    # must no longer produce a session accepted as a named account.
    derived_from_api_token = hashlib.sha256(
        b"argus.dashboard.session.v1\0me-test-token",
    ).hexdigest()
    client.cookies.set("argus_session", create_session_token(
        "reviewer", "engineer", derived_from_api_token,
    ))
    assert client.get("/api/me").status_code == 401


@pytest.mark.parametrize("identity", ["api_token", "basic_token", "disabled", "engineer", "inactive"])
@pytest.mark.parametrize(
    ("method", "url", "payload"),
    [
        ("POST", "/api/users/json", {"username": "forged-reviewer", "password": "new-secret", "role": "admin"}),
        ("PUT", "/api/users/reviewer/json", {"password": "new-secret", "role": "admin"}),
        ("DELETE", "/api/users/reviewer/json", None),
        ("POST", "/api/users/reviewer/toggle-active/json", None),
    ],
)
def test_personal_identity_mutations_reject_shared_or_invalid_authority(
    _me_db, identity, method, url, payload,
):
    victim = _me_db.create_user("reviewer", hash_password("original-secret"), "viewer", None)
    original_hash = victim.password_hash
    role = "engineer" if identity == "engineer" else "admin"
    _me_db.create_user("acting-user", hash_password("actor-secret"), role, None)
    config = ArgusConfig(auth=AuthConfig(enabled=identity != "disabled", api_token="me-test-token"))
    app = create_app(database=_me_db, config=config)
    client = TestClient(app)
    if identity == "api_token":
        client.headers["X-API-Token"] = "me-test-token"
    elif identity == "basic_token":
        client.headers["Authorization"] = "Basic " + b64encode(b"machine:me-test-token").decode()
    else:
        client.cookies.set("argus_session", create_session_token(
            "acting-user", role, app.state.session_secret,
        ))
        if identity == "inactive":
            _me_db.update_user("acting-user", active=False)

    response = client.request(method, url, json=payload)

    assert response.status_code == (401 if identity == "inactive" else 403)
    assert _me_db.get_user("forged-reviewer") is None
    unchanged = _me_db.get_user("reviewer")
    assert unchanged is not None
    assert unchanged.password_hash == original_hash
    assert unchanged.role == "viewer"
    assert unchanged.active is True
    if identity in {"api_token", "basic_token", "disabled"}:
        # Preserve the existing read-only administration API authorization.
        assert client.get("/api/users/json").status_code == 200


class TestMeEndpoint:
    """GET /api/me — current-session identity probe used by the SPA."""

    def test_unauthenticated_returns_401(self, _me_db):
        """No session cookie → middleware returns the canonical 401 envelope."""
        client, _ = _build_auth_enabled_client(_me_db)
        # Force the API branch (not the HTML redirect branch) of the middleware.
        resp = client.get("/api/me", headers={"accept": "application/json"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["code"] == 40100
        assert body["msg"] == "Authentication required"
        assert body["data"]["request_id"] == resp.headers["x-request-id"]
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["cache-control"] == "no-store"

    def test_api_request_never_redirects_to_html_login(self, _me_db):
        client, _ = _build_auth_enabled_client(_me_db)

        resp = client.get("/api/me", headers={"accept": "text/html"})

        assert resp.status_code == 401
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["code"] == 40100

    def test_authenticated_returns_username_and_role(self, _me_db):
        """Valid session cookie → 200 with enveloped user identity."""
        _me_db.create_user(
            "alice", hash_password("pw-alice"), "operator", "Alice Operator",
        )
        client, secret = _build_auth_enabled_client(_me_db)
        token = create_session_token("alice", "operator", secret)
        client.cookies.set("argus_session", token)

        resp = client.get("/api/me")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["username"] == "alice"
        assert body["role"] == "operator"
        assert body["display_name"] == "Alice Operator"
        # Must not leak sensitive fields.
        assert "password_hash" not in body
        assert "session_token" not in body
        assert "api_token" not in body

    def test_authenticated_without_display_name_omits_field(self, _me_db):
        """Users with no display_name omit it inside the response envelope."""
        _me_db.create_user("bob", hash_password("pw-bob"), "viewer", None)
        client, secret = _build_auth_enabled_client(_me_db)
        token = create_session_token("bob", "viewer", secret)
        client.cookies.set("argus_session", token)

        resp = client.get("/api/me")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body == {"username": "bob", "role": "viewer"}

    def test_session_uses_current_database_role_and_fails_closed_when_disabled(
        self,
        _me_db,
    ):
        _me_db.create_user("carol", hash_password("pw-carol"), "admin", None)
        client, secret = _build_auth_enabled_client(_me_db)
        client.cookies.set(
            "argus_session",
            create_session_token("carol", "admin", secret),
        )

        _me_db.update_user("carol", role="viewer")
        current = client.get("/api/me")
        assert current.status_code == 200
        assert current.json()["data"]["role"] == "viewer"

        _me_db.update_user("carol", active=False)
        disabled = client.get("/api/me")
        assert disabled.status_code == 401
        assert disabled.json()["code"] == 40100


class TestLoginPage:
    def test_rate_limit_response_keeps_security_headers(self, _me_db):
        client, _ = _build_auth_enabled_client(_me_db)

        response = None
        for _ in range(61):
            response = client.post(
                "/login",
                data={"username": "", "password": ""},
                follow_redirects=False,
            )

        assert response is not None
        assert response.status_code == 429
        assert response.json()["code"] == 42900
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["cache-control"] == "no-store"

    def test_login_page_is_self_contained_and_escapes_query_values(self, _me_db):
        client, _ = _build_auth_enabled_client(_me_db)

        resp = client.get(
            "/login",
            params={
                "error": '<script>alert("xss")</script>',
                "next": '/cameras?filter="online"',
            },
        )

        assert resp.status_code == 200
        assert "/static/css/argus.css" not in resp.text
        assert "<script>alert" not in resp.text
        assert "&lt;script&gt;alert" in resp.text
        assert 'value="/cameras?filter=&quot;online&quot;"' in resp.text
        assert 'name="username"' in resp.text
        assert 'name="password"' in resp.text

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("/overview", "/overview"),
            ("https://example.invalid", "/"),
            ("//example.invalid", "/"),
            ("/\\example.invalid", "/"),
            ("/overview\r\nX-Test: injected", "/"),
        ],
    )
    def test_login_next_url_remains_same_origin(self, value, expected):
        assert _safe_next_url(value) == expected

    def test_auth_redirect_and_login_preserve_the_full_same_origin_url(self, _me_db):
        _me_db.create_user(
            "alice",
            hash_password("pw-alice"),
            "operator",
            "Alice Operator",
        )
        client, _ = _build_auth_enabled_client(_me_db)

        protected = client.get(
            "/alerts?camera=cam-01&status=new",
            headers={"accept": "text/html"},
            follow_redirects=False,
        )

        assert protected.status_code == 302
        assert protected.headers["x-content-type-options"] == "nosniff"
        assert protected.headers["x-frame-options"] == "DENY"
        assert protected.headers["cache-control"] == "no-store"
        login_query = parse_qs(urlsplit(protected.headers["location"]).query)
        assert login_query == {"next": ["/alerts?camera=cam-01&status=new"]}

        logged_in = client.post(
            "/login",
            data={
                "username": "alice",
                "password": "pw-alice",
                "next": login_query["next"][0],
            },
            follow_redirects=False,
        )

        assert logged_in.status_code == 302
        assert logged_in.headers["location"] == "/alerts?camera=cam-01&status=new"
        login_cookie = logged_in.headers["set-cookie"]
        assert "argus_session=" in login_cookie
        assert "HttpOnly" in login_cookie
        assert "SameSite=lax" in login_cookie
        assert "Path=/" in login_cookie
        assert "Max-Age=28800" in login_cookie

    def test_logout_requires_post_and_deletes_the_session_cookie(self, _me_db):
        client, secret = _build_auth_enabled_client(_me_db)
        client.cookies.set(
            "argus_session",
            create_session_token("alice", "operator", secret),
        )

        get_response = client.get("/logout", follow_redirects=False)
        assert get_response.status_code == 405
        assert get_response.headers["allow"] == "POST"
        assert get_response.json() == {
            "code": 40500,
            "msg": "Method Not Allowed",
            "data": None,
        }
        response = client.post("/logout", follow_redirects=False)

        assert response.status_code == 302
        assert response.headers["location"] == "/login"
        cookie = response.headers["set-cookie"]
        assert "argus_session=" in cookie
        assert "Max-Age=0" in cookie
        assert "Path=/" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie
