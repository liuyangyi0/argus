"""SSH helper configuration checks with no network or actual key access."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts import setup_remote_sshkey as helper


@pytest.fixture(autouse=True)
def no_implicit_credentials(monkeypatch):
    for key in ("HOST", "USER", "PUBKEY", "PASSWORD"):
        monkeypatch.delenv("ARGUS_REMOTE_SSH_" + key, raising=False)
    client = MagicMock()
    monkeypatch.setattr(helper.paramiko, "SSHClient", client)
    return client


def test_missing_configuration_never_reads_key_or_connects(no_implicit_credentials, monkeypatch):
    read = MagicMock(side_effect=AssertionError("must not read implicit key"))
    monkeypatch.setattr(helper.Path, "read_text", read)
    assert helper.main() == 2
    read.assert_not_called()
    no_implicit_credentials.assert_not_called()


def test_invalid_public_key_never_connects(tmp_path, monkeypatch, no_implicit_credentials):
    key = tmp_path / "public.pub"
    key.write_text("not-a-public-key", encoding="utf-8")
    monkeypatch.setenv("ARGUS_REMOTE_SSH_HOST", "gpu-host.example")
    monkeypatch.setenv("ARGUS_REMOTE_SSH_USER", "operator")
    monkeypatch.setenv("ARGUS_REMOTE_SSH_PUBKEY", str(key))
    assert helper.main() == 2
    no_implicit_credentials.assert_not_called()


def test_known_hosts_required_and_client_closed_on_failure(tmp_path, monkeypatch, no_implicit_credentials):
    key = tmp_path / "public.pub"
    key.write_text("ssh-ed25519 cHVibGljLWtleS1maXh0dXJl fixture", encoding="utf-8")
    monkeypatch.setenv("ARGUS_REMOTE_SSH_HOST", "gpu-host.example")
    monkeypatch.setenv("ARGUS_REMOTE_SSH_USER", "operator")
    monkeypatch.setenv("ARGUS_REMOTE_SSH_PUBKEY", str(key))
    monkeypatch.setenv("ARGUS_REMOTE_SSH_PASSWORD", "mock-password")
    client = no_implicit_credentials.return_value
    client.connect.side_effect = RuntimeError("mock connection failure")
    with pytest.raises(RuntimeError, match="mock connection failure"):
        helper.main()
    client.load_system_host_keys.assert_called_once()
    assert isinstance(client.set_missing_host_key_policy.call_args.args[0], helper.paramiko.RejectPolicy)
    assert client.connect.call_args.kwargs["look_for_keys"] is False
    assert client.connect.call_args.kwargs["allow_agent"] is False
    client.close.assert_called_once()
