from __future__ import annotations

from types import SimpleNamespace

import argus.__main__ as argus_main


def _config(host: str, *, auth_enabled: bool = False):
    return SimpleNamespace(
        dashboard=SimpleNamespace(host=host, port=8080),
        auth=SimpleNamespace(enabled=auth_enabled),
    )


def test_dashboard_external_bind_warning_rule_treats_loopback_as_safe():
    for host in ("127.0.0.1", "127.1.2.3", "localhost", "::1", "[::1]"):
        assert argus_main._dashboard_exposed_without_auth(_config(host)) is False


def test_dashboard_external_bind_warning_rule_flags_external_hosts():
    for host in ("0.0.0.0", "::", "192.0.2.10", "10.0.0.5", "argus.local"):
        assert argus_main._dashboard_exposed_without_auth(_config(host)) is True


def test_dashboard_external_bind_warning_rule_skips_when_auth_enabled():
    for host in ("0.0.0.0", "::", "192.0.2.10", "argus.local"):
        assert (
            argus_main._dashboard_exposed_without_auth(
                _config(host, auth_enabled=True),
            )
            is False
        )


def test_warn_if_dashboard_exposed_without_auth_logs_only_for_unsafe_bind(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class _Logger:
        def warning(self, event: str, **kwargs):
            calls.append((event, kwargs))

    monkeypatch.setattr(argus_main, "logger", _Logger())

    argus_main._warn_if_dashboard_exposed_without_auth(_config("127.0.0.1"))
    assert calls == []

    argus_main._warn_if_dashboard_exposed_without_auth(_config("0.0.0.0"))
    assert len(calls) == 1
    assert calls[0][0] == "dashboard.auth_disabled_external_bind"
    assert calls[0][1]["host"] == "0.0.0.0"
