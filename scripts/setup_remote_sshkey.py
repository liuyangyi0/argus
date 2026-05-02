"""One-off: copy local pubkey to remote ~/.ssh/authorized_keys via password SSH.

Uses paramiko (pure Python) so it bypasses Git Bash /dev/tty + sshpass-win32 bugs.

Usage:
    # Pass password via env var (CI/scripted):
    ARGUS_REMOTE_SSH_PASSWORD=... python scripts/setup_remote_sshkey.py

    # Or interactively (will prompt):
    python scripts/setup_remote_sshkey.py

Set ARGUS_REMOTE_SSH_HOST, ARGUS_REMOTE_SSH_USER and ARGUS_REMOTE_SSH_PUBKEY
explicitly. The host must already be trusted in the user's known_hosts file.
"""
from __future__ import annotations

import getpass
import base64
import binascii
import os
import sys
from pathlib import Path

import paramiko

def run(client: paramiko.SSHClient, cmd: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd)
    rc = stdout.channel.recv_exit_status()
    return rc, stdout.read().decode(), stderr.read().decode()


def main() -> int:
    host = os.environ.get("ARGUS_REMOTE_SSH_HOST", "").strip()
    user = os.environ.get("ARGUS_REMOTE_SSH_USER", "").strip()
    public_key_path = os.environ.get("ARGUS_REMOTE_SSH_PUBKEY", "").strip()
    if not host or not user or not public_key_path:
        print("[err] set ARGUS_REMOTE_SSH_HOST, ARGUS_REMOTE_SSH_USER and ARGUS_REMOTE_SSH_PUBKEY", file=sys.stderr)
        return 2
    try:
        pubkey = Path(public_key_path).expanduser().read_text(encoding="utf-8").strip()
        parts = pubkey.split()
        if "\n" in pubkey or "\r" in pubkey or len(parts) < 2:
            raise ValueError("expected one public-key line")
        if parts[0] not in {"ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521"}:
            raise ValueError("unsupported public-key type")
        if not base64.b64decode(parts[1], validate=True):
            raise ValueError("empty public-key payload")
    except (OSError, UnicodeError, ValueError, binascii.Error):
        print("[err] provide a readable, single-line SSH public key", file=sys.stderr)
        return 2

    password = os.environ.get("ARGUS_REMOTE_SSH_PASSWORD")
    if not password:
        password = getpass.getpass(f"Password for {user}@{host}: ")
    if not password:
        print("[err] no password supplied (set ARGUS_REMOTE_SSH_PASSWORD or enter at prompt)", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        return _install_public_key(client, host, user, password, pubkey)
    finally:
        client.close()


def _install_public_key(client, host, user, password, pubkey) -> int:
    client.connect(host, username=user, password=password, timeout=10,
                   allow_agent=False, look_for_keys=False)
    print("[conn] authenticated to a known host")

    rc, out, err = run(client, "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys")
    if rc != 0:
        print(f"[err] mkdir: {err}", file=sys.stderr)
        return 1

    rc, out, _ = run(client, "cat ~/.ssh/authorized_keys")
    if pubkey in out:
        print("[remote] key already present, skipping append")
    else:
        escaped = pubkey.replace("'", "'\\''")
        rc, _, err = run(client, f"echo '{escaped}' >> ~/.ssh/authorized_keys")
        if rc != 0:
            print(f"[err] append: {err}", file=sys.stderr)
            return 1
        print("[remote] key appended to authorized_keys")

    rc, out, _ = run(client, "hostname && whoami && wc -l ~/.ssh/authorized_keys")
    print(f"[remote] verify:\n{out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
