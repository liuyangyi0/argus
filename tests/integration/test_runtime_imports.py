"""Smoke tests guarding ``argus`` package import surface and CLI startup.

These exist as the safety net for Stage-3 refactors of ``__main__.py`` and
``trainer.py``: any rewrite that breaks module loading or strips a
public symbol fails one of these long before integration tests would.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


# ── CLI smoke test ──────────────────────────────────────────────────────────


def test_cli_help_exits_zero() -> None:
    """`python -m argus --help` must succeed; that proves __main__.py imports."""
    result = subprocess.run(
        [sys.executable, "-m", "argus", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"argus --help failed with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Argus" in result.stdout
    assert "--config" in result.stdout
    assert "--no-dashboard" in result.stdout


# ── Public symbol surface (Stage 3 backwards-compat guard) ──────────────────


@pytest.mark.parametrize(
    "module_path,symbol",
    [
        # argus.anomaly.trainer must keep these importable for routes/, tests/, jobs/
        ("argus.anomaly.trainer", "ModelTrainer"),
        ("argus.anomaly.trainer", "TrainingStatus"),
        ("argus.anomaly.trainer", "TrainingResult"),
        ("argus.anomaly.trainer", "QualityReport"),
        ("argus.anomaly.trainer", "MODEL_INFO"),
        # _list_images is a private util but training_validator.py imports it
        ("argus.anomaly.trainer", "_list_images"),
        # Trainer's OpenVINO static-shape guard must remain accessible
        # even after the export module is split out
        ("argus.anomaly.trainer", "ModelTrainer"),
    ],
)
def test_public_symbol_importable(module_path: str, symbol: str) -> None:
    mod = importlib.import_module(module_path)
    assert hasattr(mod, symbol), f"{module_path} no longer exports {symbol!r}"


def test_model_trainer_keeps_export_guard_attached() -> None:
    """``_assert_openvino_static`` must remain reachable on ModelTrainer.

    External callers (and unit tests) reach it via
    ``ModelTrainer._assert_openvino_static``; the Stage-3 split into a helper
    module must preserve that path even though the implementation moves.
    """
    from argus.anomaly.trainer import ModelTrainer

    assert hasattr(ModelTrainer, "_assert_openvino_static"), (
        "ModelTrainer._assert_openvino_static disappeared — Stage-3 export "
        "split must re-mount the helper as a static method."
    )
    assert callable(ModelTrainer._assert_openvino_static)


def test_runtime_storage_modules_importable() -> None:
    """Storage layer must not break under Stage-3 __main__.py reshuffles."""
    for path in [
        "argus.storage.database",
        "argus.storage.audit",
        "argus.storage.backup",
        "argus.storage.alert_recording",
        "argus.storage.inference_buffer",
        "argus.storage.inference_store",
    ]:
        importlib.import_module(path)
