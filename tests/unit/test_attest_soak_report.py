from __future__ import annotations

import base64

from argus.storage.release_evidence import (
    SOAK_ATTESTATION_ALGORITHM,
    SOAK_REPORT_SCHEMA,
    SOAK_TRUST_STORE_SCHEMA,
    sign_soak_report_hmac,
)
from scripts.attest_soak_report import attest_report


def test_attest_report_uses_selected_enabled_key():
    secret = b"k" * 32
    report = {
        "schema_version": SOAK_REPORT_SCHEMA,
        "subject": {"model": "v1"},
    }
    store = {
        "schema_version": SOAK_TRUST_STORE_SCHEMA,
        "keys": [{
            "key_id": "key-1",
            "algorithm": SOAK_ATTESTATION_ALGORITHM,
            "secret_base64": base64.b64encode(secret).decode("ascii"),
            "enabled": True,
        }],
    }

    signed = attest_report(report, store, key_id="key-1")

    assert signed["attestation"]["signature"] == sign_soak_report_hmac(
        signed,
        secret=secret,
    )
    assert "attestation" not in report
