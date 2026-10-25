"""Keep removed deployment identities and sector labels out of public files.

Policy values are encoded to avoid republishing the removed text in this guard.
Only working files are checked: old local clones require a separate history audit.
"""
from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {
    "restricted_sector_zh": bytes.fromhex("e6a0b8e794b5"),
    "restricted_sector_en": bytes.fromhex("6e75636c656172"),
    "private_host_alias": bytes.fromhex("776870323232"),
    "private_deployment_subnet": bytes.fromhex("3139322e3136382e36362e"),
    "private_account_path": bytes.fromhex("2f686f6d652f776870"),
    "private_key_filename": bytes.fromhex("69645f656432353531395f6172677573"),
    "private_windows_validation_host": bytes.fromhex("6e6c792d6465736b746f70"),
    "private_windows_validation_address": bytes.fromhex("3139322e3136382e332e313232"),
}
PRIVATE_KEY_BLOCK = re.compile(
    rb"-----BEGIN "
    rb"(?P<kind>(?:(?:OPENSSH|RSA|EC|DSA|ENCRYPTED) )?PRIVATE KEY)-----\r?\n"
    rb"[A-Za-z0-9+/=\r\n]{64,}-----END (?P=kind)-----"
)
MEDIA_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".png", ".jpg", ".jpeg", ".gif"}


def test_public_files_do_not_reintroduce_private_deployment_details():
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
    )
    problems = []
    for relative in set(output.split(b"\0")) - {b""}:
        name = relative.decode("utf-8")
        path = ROOT / name
        if not path.exists() and not path.is_symlink():
            continue
        for label, value in FORBIDDEN.items():
            if value in relative.lower():
                problems.append(f"{name}: filename:{label}")
        if path.suffix.lower() in MEDIA_SUFFIXES:
            continue
        data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
        if b"\0" in data:
            continue
        for label, value in FORBIDDEN.items():
            if value in data.lower():
                problems.append(f"{name}: content:{label}")
        if PRIVATE_KEY_BLOCK.search(data):
            problems.append(f"{name}: private-key block")
    assert not problems, "Public-file privacy guard failed (contents omitted):\n" + "\n".join(sorted(problems))


@pytest.mark.skipif(os.environ.get("ARGUS_PUBLICATION_HISTORY_CHECK") != "1", reason="History audit requires a cleaned publication clone")
def test_public_history_does_not_reintroduce_removed_data():
    objects = subprocess.check_output(
        ["git", "rev-list", "--objects", "--all"], cwd=ROOT,
    ).splitlines()
    problems = []
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for item in objects:
            object_id, _, name = item.partition(b" ")
            for label, value in FORBIDDEN.items():
                if value in name.lower():
                    problems.append(f"{object_id.decode()}: filename:{label}")
            process.stdin.write(object_id + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().split()
            assert len(header) == 3, "Invalid git object header"
            size = int(header[2])
            data = process.stdout.read(size)
            assert len(data) == size and process.stdout.read(1) == b"\n", "Truncated git object"
            for label, value in FORBIDDEN.items():
                if value in data.lower():
                    problems.append(f"{object_id.decode()}: {header[1].decode()}:{label}")
            if PRIVATE_KEY_BLOCK.search(data):
                problems.append(f"{object_id.decode()}: private-key block")
    finally:
        process.stdin.close()
        process.stdout.close()
        assert process.wait(timeout=10) == 0
    assert not problems, "Public-history privacy guard failed (contents omitted):\n" + "\n".join(sorted(problems))
