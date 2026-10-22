# model_discovery.py

"""Helpers for locating inference-ready anomaly model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

INFERENCE_MODEL_PATTERNS = ("model.xml", "model.pt")
_INFERENCE_MODEL_SUFFIXES = {".xml", ".pt"}
InferenceBackend = Literal["auto", "torch-cuda", "openvino", "torch-cpu"]


def _patterns_for_backend(backend: InferenceBackend = "auto") -> tuple[str, ...]:
    if backend == "openvino":
        return ("model.xml",)
    if backend in {"torch-cuda", "torch-cpu"}:
        return ("model.pt",)
    return INFERENCE_MODEL_PATTERNS


def _suffixes_for_backend(backend: InferenceBackend = "auto") -> set[str]:
    return {Path(pattern).suffix for pattern in _patterns_for_backend(backend)}


def _latest_match(search_dir: Path, backend: InferenceBackend = "auto") -> Path | None:
    for pattern in _patterns_for_backend(backend):
        matches = sorted(
            search_dir.rglob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0].resolve()
    return None


def find_runtime_model_in_dir(
    base: Path,
    camera_id: str,
    backend: InferenceBackend = "auto",
) -> Path | None:
    """Find the best inference-ready model for a camera from a directory."""
    search_dirs: list[Path] = []
    exports_dir = Path("data/exports") / camera_id
    if exports_dir.exists():
        search_dirs.append(exports_dir)
    if base.exists():
        search_dirs.append(base)

    seen: set[Path] = set()
    for search_dir in search_dirs:
        resolved_dir = search_dir.resolve()
        if resolved_dir in seen:
            continue
        seen.add(resolved_dir)
        match = _latest_match(search_dir, backend)
        if match is not None:
            return match
    return None


def find_runtime_model(camera_id: str, backend: InferenceBackend = "auto") -> Path | None:
    """Auto-discover the latest inference-ready model for a camera."""
    for search_dir in (Path("data/exports") / camera_id, Path("data/models") / camera_id):
        if not search_dir.exists():
            continue
        match = _latest_match(search_dir, backend)
        if match is not None:
            return match
    return None


def resolve_runtime_model_path(
    model_path: str | Path,
    camera_id: str,
    backend: InferenceBackend = "auto",
) -> Path | None:
    """Resolve a requested path to a deployable runtime artifact."""
    candidate = Path(model_path)
    allowed_suffixes = _suffixes_for_backend(backend)
    if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes:
        return candidate.resolve()
    if candidate.is_file() and candidate.suffix.lower() in _INFERENCE_MODEL_SUFFIXES:
        # An explicitly requested runtime artifact of the wrong backend must
        # fail closed.  Non-runtime files such as Lightning checkpoints may
        # still resolve to their separately exported deployment artifact.
        return None

    search_dirs: list[Path] = []
    exports_dir = Path("data/exports") / camera_id
    if exports_dir.exists():
        search_dirs.append(exports_dir)

    if candidate.is_dir():
        search_dirs.append(candidate)
    elif candidate.exists():
        search_dirs.append(candidate.parent)

    seen: set[Path] = set()
    for search_dir in search_dirs:
        resolved_dir = search_dir.resolve()
        if resolved_dir in seen or not search_dir.exists():
            continue
        seen.add(resolved_dir)
        match = _latest_match(search_dir, backend)
        if match is not None:
            return match

    return find_runtime_model(camera_id, backend)


def find_all_models(camera_id: str) -> list[Path]:
    """Discover all inference-ready model files for a camera.

    Searches data/exports/{camera_id} and data/models/{camera_id} for
    .pt and .xml files. Returns deduplicated list sorted by mtime (newest first).
    Used by DetectorEnsemble to find multiple models for multi-model fusion.
    """
    seen: set[Path] = set()
    results: list[tuple[float, Path]] = []

    for search_dir in (Path("data/exports") / camera_id, Path("data/models") / camera_id):
        if not search_dir.exists():
            continue
        for suffix in (".xml", ".pt"):
            for match in search_dir.rglob(f"*{suffix}"):
                resolved = match.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    results.append((match.stat().st_mtime, resolved))

    # Sort newest first
    results.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in results]
