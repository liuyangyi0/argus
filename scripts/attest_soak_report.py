#!/usr/bin/env python3
"""Attach a protected-key HMAC attestation to an Argus soak report.

The key file is intentionally external to the repository.  On Linux it must
be owned by the current service user and inaccessible to group/other users.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import stat
import sys
from pathlib import Path

try:
    from argus.storage.release_evidence import (
        SOAK_ATTESTATION_ALGORITHM,
        SOAK_REPORT_SCHEMA,
        SOAK_TRUST_STORE_SCHEMA,
        sign_soak_report_hmac,
    )
except ModuleNotFoundError:  # direct execution from a source checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from argus.storage.release_evidence import (
        SOAK_ATTESTATION_ALGORITHM,
        SOAK_REPORT_SCHEMA,
        SOAK_TRUST_STORE_SCHEMA,
        sign_soak_report_hmac,
    )


def _read_mapping(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _protected_file(path: Path) -> None:
    if os.name != "posix":
        return
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("trust store must not allow group/other access")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise ValueError("trust store must be owned by the current user")


def attest_report(report: dict, trust_store: dict, *, key_id: str) -> dict:
    if report.get("schema_version") != SOAK_REPORT_SCHEMA:
        raise ValueError(f"report must use {SOAK_REPORT_SCHEMA}")
    if trust_store.get("schema_version") != SOAK_TRUST_STORE_SCHEMA:
        raise ValueError("unsupported trust store schema")
    keys = trust_store.get("keys")
    if not isinstance(keys, list):
        raise ValueError("trust store keys must be an array")
    matches = [
        item
        for item in keys
        if isinstance(item, dict) and item.get("key_id") == key_id
    ]
    if len(matches) != 1:
        raise ValueError("key_id is unknown or ambiguous")
    key = matches[0]
    if (
        key.get("enabled") is not True
        or key.get("algorithm") != SOAK_ATTESTATION_ALGORITHM
    ):
        raise ValueError("attestation key is disabled or unsupported")
    try:
        secret = base64.b64decode(key["secret_base64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("attestation key material is invalid") from exc

    signed = dict(report)
    signed["attestation"] = {
        "algorithm": SOAK_ATTESTATION_ALGORITHM,
        "key_id": key_id,
        "signature": "pending",
    }
    signed["attestation"]["signature"] = sign_soak_report_hmac(
        signed,
        secret=secret,
    )
    return signed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--trust-store", required=True, type=Path)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    report_path = args.report.expanduser().resolve(strict=True)
    trust_path = args.trust_store.expanduser().resolve(strict=True)
    _protected_file(trust_path)
    signed = attest_report(
        _read_mapping(report_path, "soak report"),
        _read_mapping(trust_path, "soak trust store"),
        key_id=args.key_id,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(signed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
