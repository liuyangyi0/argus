from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.smoke_dashboard_routes import (
    BROWSER_ROUTE_EXPECTATIONS,
    CORE_DASHBOARD_ROUTES,
    DashboardSmokeFailure,
    _check_browser_dom_routes,
    _dump_dom_with_browser,
    _find_headless_browser,
    _wait_for_camera,
    parse_args,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    def get(self, path, timeout=None):  # noqa: ARG002
        return _FakeResponse(self._payload)


def test_core_dashboard_routes_cover_operator_loop():
    assert {
        "/cameras",
        "/alerts",
        "/replay/ALT-smoke",
        "/replay/ALT-smoke/storyboard",
        "/models/baseline",
        "/models/training",
        "/models/registry",
        "/system/overview",
        "/system/config",
        "/system/degradation",
        "/reports",
    }.issubset(set(CORE_DASHBOARD_ROUTES))


def test_browser_dom_expectations_cover_every_core_route():
    assert set(BROWSER_ROUTE_EXPECTATIONS) == set(CORE_DASHBOARD_ROUTES)
    assert all(markers for markers in BROWSER_ROUTE_EXPECTATIONS.values())


def test_parse_args_accepts_dashboard_smoke_overrides():
    args = parse_args([
        "--port",
        "18080",
        "--camera",
        "c",
        "--camera-id",
        "c",
        "--timeout",
        "3.5",
        "--min-frames",
        "2",
        "--browser",
        "off",
        "--keep-work-dir",
    ])

    assert args.port == 18080
    assert args.camera == "c"
    assert args.camera_id == "c"
    assert args.timeout == 3.5
    assert args.min_frames == 2
    assert args.browser == "off"
    assert args.keep_work_dir is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--port", "-1"],
        ["--port", "65536"],
        ["--timeout", "0"],
        ["--min-frames", "0"],
        ["--browser-timeout", "0"],
        ["--browser-virtual-time-ms", "0"],
    ],
)
def test_parse_args_rejects_invalid_dashboard_smoke_values(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_find_headless_browser_rejects_missing_explicit_path(tmp_path):
    assert _find_headless_browser(str(tmp_path / "missing.exe")) is None


def test_browser_dom_check_can_be_disabled(tmp_path):
    args = parse_args(["--browser", "off"])

    result = _check_browser_dom_routes(args, "http://127.0.0.1:1", tmp_path)

    assert result == {"status": "off", "routes_checked": []}


def test_browser_dom_required_fails_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.smoke_dashboard_routes._find_headless_browser", lambda explicit_path=None: None)
    args = parse_args(["--browser", "required"])

    with pytest.raises(DashboardSmokeFailure, match="browser required"):
        _check_browser_dom_routes(args, "http://127.0.0.1:1", tmp_path)


def test_browser_dom_dump_retries_with_fresh_profile(monkeypatch, tmp_path):
    calls = []

    def fake_cdp(**kwargs):
        calls.append(kwargs["user_data_dir"])
        if len(calls) == 1:
            raise TimeoutError("target not ready")
        return "<html>ARGUS</html>"

    monkeypatch.setattr("scripts.smoke_dashboard_routes._dump_dom_with_cdp", fake_cdp)

    result = _dump_dom_with_browser(
        browser_path="chrome.exe",
        url="http://127.0.0.1:1/cameras",
        user_data_dir=tmp_path / "profile",
        timeout_s=30,
        virtual_time_ms=1000,
    )

    assert result == "<html>ARGUS</html>"
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert calls[0].parent == tmp_path / "profile"
    assert calls[1].parent == tmp_path / "profile"


def test_wait_for_camera_returns_connected_running_camera():
    payload = {
        "data": {
            "cameras": [
                {
                    "camera_id": "c",
                    "connected": True,
                    "running": True,
                    "stats": {"frames_captured": 5},
                }
            ]
        }
    }
    process = SimpleNamespace(poll=lambda: None)

    row = _wait_for_camera(
        _FakeClient(payload),
        camera_id="c",
        timeout_s=0.1,
        min_frames=5,
        process=process,
    )

    assert row["camera_id"] == "c"


def test_wait_for_camera_fails_when_argus_exits_early():
    process = SimpleNamespace(poll=lambda: 1, returncode=1)

    with pytest.raises(DashboardSmokeFailure, match="exited early"):
        _wait_for_camera(
            _FakeClient({"data": {"cameras": []}}),
            camera_id="c",
            timeout_s=0.1,
            min_frames=5,
            process=process,
        )
