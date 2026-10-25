"""Check lock coverage and prevent model engines in ordinary-function CI."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
MODEL_ENGINES = {"torch", "torchvision", "ultralytics", "anomalib", "openvino",
                 "onnxruntime", "onnxruntime-gpu", "lightglue", "sam2", "fiftyone", "cvat-sdk"}


def read_lock(path: Path) -> dict[str, list[Requirement]]:
    locked: dict[str, list[Requirement]] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        declaration, *hashes = line.split(" --hash=")
        requirement = Requirement(declaration.strip())
        pins = list(requirement.specifier)
        if (len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version
                or requirement.url or not hashes
                or any(not re.fullmatch(r"sha256:[0-9a-f]{64}", item.strip()) for item in hashes)):
            raise ValueError(f"{path.name}: {requirement.name} is not exactly hash-locked")
        locked.setdefault(canonicalize_name(requirement.name), []).append(requirement)
    if not locked:
        raise ValueError(f"Empty dependency lock: {path}")
    return locked


def main() -> int:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = read_lock(ROOT / "requirements/runtime-linux-x86_64.txt")
    functional = read_lock(ROOT / "requirements/functional.txt")
    build = read_lock(ROOT / "requirements/build.txt")
    for declaration in project["project"]["dependencies"]:
        requirement = Requirement(declaration)
        entries = runtime.get(canonicalize_name(requirement.name), [])
        if not entries or any(next(iter(entry.specifier)).version not in requirement.specifier
                              for entry in entries):
            raise ValueError(f"Runtime lock is stale for {requirement.name}")
    for declaration in project["build-system"]["requires"]:
        requirement = Requirement(declaration)
        entries = build.get(canonicalize_name(requirement.name), [])
        if not entries or any(next(iter(entry.specifier)).version not in requirement.specifier
                              for entry in entries):
            raise ValueError(f"Build lock is stale for {requirement.name}")
    if MODEL_ENGINES.intersection(functional):
        raise ValueError("Ordinary-function lock contains a model or external review engine")
    for source, locks in (("functional.in", (functional,)), ("build.in", (build,)),
                          ("runtime-constraints.txt", (runtime, functional))):
        for line in (ROOT / "requirements" / source).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith(("#", "-")):
                continue
            requirement = Requirement(line)
            entries = [entry for locked in locks
                       for entry in locked.get(canonicalize_name(requirement.name), [])]
            if source in {"functional.in", "build.in"} and not entries:
                raise ValueError(f"{source} lock misses {requirement.name}")
            if any(next(iter(entry.specifier)).version not in requirement.specifier
                   for entry in entries):
                raise ValueError(f"Dependency lock is stale for {source}: {requirement.name}")
    manifest = json.loads((ROOT / "web/package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "web/package-lock.json").read_text(encoding="utf-8"))
    root_package = lock["packages"][""]
    for field in ("dependencies", "devDependencies"):
        if root_package.get(field, {}) != manifest.get(field, {}):
            raise ValueError(f"Frontend lock is stale for {field}")
    print(f"Dependency locks checked: {len(runtime)} runtime, {len(functional)} functional, "
          f"{len(build)} build packages; frontend declarations match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
