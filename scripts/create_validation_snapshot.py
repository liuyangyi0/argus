"""Create a source-only validation snapshot, excluding secrets and runtime data.

The manifest identifies the dirty working files, not a Git commit. A hardware
test must verify every entry before using the snapshot. No network operations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src", "scripts", "tests", "configs", "requirements", "docs", "web/src", "web/public", "web/e2e", "web/dist")
FILES = ("pyproject.toml", "README.md", "LICENSE", "AGENTS.md", "web/package.json", "web/package-lock.json",
         "web/index.html", "web/vite.config.ts", "web/vitest.config.ts", "web/playwright.config.ts",
         "web/tsconfig.json", "web/tsconfig.app.json", "web/tsconfig.node.json")
EXTENSIONS = {".py", ".ts", ".vue", ".js", ".mjs", ".cjs", ".json", ".yaml", ".yml", ".toml",
              ".txt", ".in", ".md", ".html", ".css", ".svg", ".png", ".jpg", ".jpeg", ".ico", ".woff", ".woff2"}
EXCLUDED = {"__pycache__", "node_modules", ".git", ".venv", ".local-validation", ".pytest_cache", ".ruff_cache"}


def assert_plain_path(path: Path) -> None:
    """Never follow a source link/junction, including parent directories."""
    for part in (path, *path.parents):
        if part == ROOT:
            break
        if part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400:
            raise RuntimeError("Snapshot must not traverse links")
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Snapshot source escaped the repository")


def paths() -> list[Path]:
    result = set()
    for name in DIRECTORIES:
        folder = ROOT / name
        if folder.is_dir():
            for path in folder.rglob("*"):
                relative = path.relative_to(ROOT)
                if (not path.is_file() or any(part in EXCLUDED or part.startswith(".") for part in relative.parts)
                        or path.suffix.lower() not in EXTENSIONS):
                    continue
                assert_plain_path(path)
                result.add(path)
    result.update(ROOT / name for name in FILES if (ROOT / name).is_file())
    for path in result:
        assert_plain_path(path)
    return sorted(result, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / ".local-validation").resolve()) or output.suffix != ".zip":
        parser.error("Snapshot output must be a private .local-validation/*.zip")
    if output.exists():
        parser.error("Refusing to overwrite an existing validation snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = paths()
    entries = []
    fingerprints = {}
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            before = path.stat()
            content = path.read_bytes()
            after = path.stat()
            fingerprint = (before.st_size, before.st_mtime_ns, before.st_ino)
            if fingerprint != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise RuntimeError("Working file changed while snapshotting")
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, content)
            entries.append({"path": relative, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            fingerprints[path] = fingerprint
        if paths() != selected or any(
            fingerprints[path] != (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
            for path in selected
        ):
            raise RuntimeError("Working tree changed before snapshot completion")
        canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
        manifest = {"schema": "argus.validation-source-snapshot.v1", "files": entries,
                    "source_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "git_commit_claimed": False, "runtime_data_included": False}
        archive.writestr("_argus_snapshot_manifest.json", json.dumps(manifest, sort_keys=True))
    with output.open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({"path": str(output), "file_count": len(entries),
                      "source_sha256": manifest["source_sha256"],
                      "archive_sha256": archive_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
