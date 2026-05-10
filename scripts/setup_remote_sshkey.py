"""One-off: copy local pubkey to remote ~/.ssh/authorized_keys via password SSH.

Uses paramiko (pure Python) so it bypasses Git Bash /dev/tty + sshpass-win32 bugs.

Usage:
    # Pass password via env var (CI/scripted):
    ARGUS_REMOTE_SSH_PASSWORD=... python scripts/setup_remote_sshkey.py

    # Or interactively (will prompt):
    python scripts/setup_remote_sshkey.py

Host/user can also be overridden via ARGUS_REMOTE_SSH_HOST / ARGUS_REMOTE_SSH_USER.
"""
from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("ARGUS_REMOTE_SSH_HOST", "192.168.66.222")
USER = os.environ.get("ARGUS_REMOTE_SSH_USER", "whp")
PUBKEY_PATH = Path.home() / ".ssh" / "id_ed25519_argus.pub"


def run(client: paramiko.SSHClient, cmd: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd)
    rc = stdout.channel.recv_exit_status()
    return rc, stdout.read().decode(), stderr.read().decode()


def main() -> int:
    pubkey = PUBKEY_PATH.read_text().strip()
    print(f"[local] pubkey: {pubkey[:60]}...")

    password = os.environ.get("ARGUS_REMOTE_SSH_PASSWORD")
    if not password:
        password = getpass.getpass(f"Password for {USER}@{HOST}: ")
    if not password:
        print("[err] no password supplied (set ARGUS_REMOTE_SSH_PASSWORD or enter at prompt)", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[conn] {USER}@{HOST} with password...")
    client.connect(HOST, username=USER, password=password, timeout=10,
                   allow_agent=False, look_for_keys=False)
    print("[conn] ok")

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

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
