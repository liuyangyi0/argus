"""Compare the existing SceneGuard with LightGlue in an isolated offline process.

This command records diagnostic disagreements. It never changes a production
SceneGuard decision, approves a profile, or releases a model. Optional model
dependencies are imported only after explicit command invocation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("must be a 64-character hexadecimal SHA256")
    return value.lower()


def _keypoints(value: str) -> int:
    number = _positive_int(value)
    if not 32 <= number <= 8192:
        raise argparse.ArgumentTypeError("must be in [32, 8192]")
    return number


def _device(value: str) -> str:
    if value == "cpu" or re.fullmatch(r"cuda(?::[0-9]+)?", value):
        return value
    raise argparse.ArgumentTypeError("must be cpu, cuda, or cuda:<device index>")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--video", required=True, type=Path, help="Local prerecorded video")
    parser.add_argument(
        "--profile-id", required=True,
        help="Existing operator-approved, model-bound profile in the configured profiles_dir",
    )
    parser.add_argument("--extractor-weights", required=True, type=Path)
    parser.add_argument("--extractor-sha256", required=True, type=_sha256)
    parser.add_argument("--matcher-weights", required=True, type=Path)
    parser.add_argument("--matcher-sha256", required=True, type=_sha256)
    parser.add_argument("--device", default="cuda:0", type=_device)
    parser.add_argument("--max-keypoints", default=2048, type=_keypoints)
    parser.add_argument(
        "--sample-every", default=1, type=_positive_int,
        help="Compare every Nth decoded frame; sampling is recorded in the report",
    )
    parser.add_argument("--max-frames", type=_positive_int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace this report file only")
    return parser.parse_args(argv)


def _check_output(args: argparse.Namespace) -> None:
    output = args.output
    resolved = output.resolve()
    for source in (args.config, args.video, args.extractor_weights, args.matcher_weights):
        if resolved == source.resolve():
            raise ValueError(f"output must not replace an input: {output}")
        if output.exists() and source.exists() and output.samefile(source):
            raise ValueError(f"output must not replace an input: {output}")
    if output.is_symlink():
        raise ValueError(f"output must not be a symbolic link: {output}")
    if output.exists() and (not args.overwrite or not output.is_file()):
        raise FileExistsError(f"output exists; use --overwrite for report files only: {output}")
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError("overwrite is restricted to existing SceneGuard comparison reports") from exc
        if not (
            isinstance(existing, dict)
            and existing.get("schema_version") == "argus.scene-guard-comparison.v1"
            and existing.get("diagnostic_only") is True
            and existing.get("gating_effect") == "none"
            and existing.get("release_authority") is False
        ):
            raise ValueError("overwrite is restricted to existing SceneGuard comparison reports")


def run(args: argparse.Namespace) -> int:
    _check_output(args)
    from argus.validation.scene_guard_comparison import (
        run_scene_guard_comparison,
        write_comparison_report,
    )

    report = run_scene_guard_comparison(
        config_path=args.config,
        camera_id=args.camera_id,
        video_path=args.video,
        profile_id=args.profile_id,
        extractor_weights=args.extractor_weights,
        extractor_sha256=args.extractor_sha256,
        matcher_weights=args.matcher_weights,
        matcher_sha256=args.matcher_sha256,
        device=args.device,
        max_keypoints=args.max_keypoints,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
    )
    write_comparison_report(report, args.output, overwrite=args.overwrite)
    status = report.get("status", "unknown")
    print(f"SceneGuard comparison report written to {args.output}; status={status}")
    if status != "complete":
        print("Comparison is incomplete; inspect the report errors and coverage.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"SceneGuard comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
