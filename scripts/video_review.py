"""Prepare and exchange offline video review bundles, reports and review images.

The optional SDKs are imported only by the command that uses them. Credentials
come from ARGUS_CVAT_USERNAME, ARGUS_CVAT_PASSWORD and ARGUS_CVAT_TOKEN; they
are never accepted as command-line arguments or stored in the review bundle.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


_SECRET_ENV_NAMES = (
    "ARGUS_CVAT_USERNAME", "ARGUS_CVAT_PASSWORD", "ARGUS_CVAT_TOKEN",
    "ARGUS_CVAT_ACCESS_TOKEN", "CVAT_ACCESS_TOKEN",
    "FIFTYONE_DATABASE_URI",
)
_SECRET_ARGUMENTS = {"--token", "--password", "--access-token", "--api-key"}
_MONGODB_URI = re.compile(r"mongodb(?:\+srv)?://\S+", re.IGNORECASE)


def _redact(message: str, extra_secrets: tuple[str, ...] = ()) -> str:
    # Configuration-file credentials may not exist in this process's env. Mask
    # the entire MongoDB URI, including query secrets, before replacements that
    # might otherwise remove part of its scheme and make it unrecognizable.
    message = _MONGODB_URI.sub("[redacted]", message)
    # Longest first: replacing a short username must not expose a token suffix
    # when that username happens to be a substring of the access token.
    secrets = {os.environ.get(key, "") for key in _SECRET_ENV_NAMES}
    secrets.update(extra_secrets)
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    return message


class _ReviewArgumentParser(argparse.ArgumentParser):
    """Reject secret arguments without echoing their values into stderr."""

    def parse_known_args(self, args=None, namespace=None):
        values = list(sys.argv[1:] if args is None else args)
        secrets = []
        for index, value in enumerate(values):
            flag, separator, inline = value.partition("=")
            if flag not in _SECRET_ARGUMENTS:
                continue
            if separator:
                secrets.append(inline)
            elif index + 1 < len(values):
                secrets.append(values[index + 1])
        self._argument_secrets = tuple(secrets)
        return super().parse_known_args(values, namespace)

    def error(self, message):
        super().error(_redact(message, getattr(self, "_argument_secrets", ())))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _ReviewArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Prepare an immutable-source review bundle")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--evaluation-dir", type=Path)
    prepare.add_argument("--split-file", type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--overwrite", action="store_true", help="Replace this output file only")

    push = commands.add_parser("cvat-push", help="Upload videos and annotation suggestions")
    _add_cvat_arguments(push)
    push.add_argument("--receipt", required=True, type=Path)

    pull = commands.add_parser("cvat-pull", help="Import completed, human-reviewed annotations")
    _add_cvat_arguments(pull)
    pull.add_argument("--receipt", required=True, type=Path)
    pull.add_argument("--reviewed-by", required=True)
    pull.add_argument("--confirm-normal-clip", action="append", default=[], metavar="CLIP_ID")
    pull.add_argument("--output", required=True, type=Path)
    pull.add_argument("--overwrite", action="store_true", help="Replace this output file only")

    labels = commands.add_parser(
        "export-labels", help="Export reviewed labels and regression manifest",
    )
    labels.add_argument("--bundle", required=True, type=Path)
    labels.add_argument("--output-dir", required=True, type=Path)
    labels.add_argument("--split", choices=["train", "validation", "test", "unassigned"])
    labels.add_argument(
        "--overwrite", action="store_true", help="Replace generated output files only",
    )

    fiftyone = commands.add_parser(
        "fiftyone-import", help="Create a FiftyOne dataset from a bundle",
    )
    fiftyone.add_argument("--bundle", required=True, type=Path)
    fiftyone.add_argument("--dataset-name", required=True)
    fiftyone.add_argument(
        "--scene-comparison", action="append", default=[], type=Path, metavar="REPORT",
        help="Attach a SceneGuard/LightGlue diagnostic report; repeat for distinct clips",
    )
    fiftyone.add_argument(
        "--launch", action="store_true", help="Open the app and wait for it to close",
    )

    images = commands.add_parser(
        "scene-images-prepare",
        help="Extract hash-verified source frames into a new diagnostic image directory",
    )
    _add_scene_image_arguments(images)
    images.add_argument("--output-dir", required=True, type=Path)
    images.add_argument(
        "--selection", choices=["disagreements", "all-compared"], default="disagreements",
        help="Disagreements includes raw-state disagreements and algorithm errors",
    )
    images.add_argument(
        "--max-images", type=_positive_int, default=10000,
        help="Positive selection limit; hard maximum 10000 (excess fails, never truncates)",
    )
    images.add_argument(
        "--max-output-bytes", type=_positive_int, default=10737418240,
        help="Positive artifact byte budget; hard maximum 10737418240 (10 GiB)",
    )

    image_import = commands.add_parser(
        "scene-images-import", help="Import verified review PNGs as a FiftyOne image dataset",
    )
    _add_scene_image_arguments(image_import)
    image_import.add_argument("--manifest", required=True, type=Path)
    image_import.add_argument("--dataset-name", required=True)
    image_import.add_argument(
        "--launch", action="store_true", help="Open the unreviewed image view and wait",
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _add_scene_image_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--scene-comparison", required=True, action="append", type=Path, metavar="REPORT",
        help="Validated diagnostic report; repeat for distinct clips",
    )


def _add_cvat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--url", required=True, help="CVAT origin, without credentials or query string",
    )
    parser.add_argument("--organization", help="CVAT organization slug, when applicable")


def _check_output(path: Path, *, overwrite: bool, inputs: list[Path | None]) -> None:
    resolved = path.resolve()
    for source in inputs:
        if source is not None and resolved == source.resolve():
            raise ValueError(f"output must not replace an input: {path}")
    if path.is_symlink():
        raise ValueError(f"output must not be a symbolic link: {path}")
    if path.exists() and (not overwrite or not path.is_file()):
        raise FileExistsError(f"output exists; use --overwrite for output files only: {path}")


def _bundle_sources(bundle) -> list[Path]:
    sources: list[Path] = []
    if getattr(bundle, "manifest_path", None):
        sources.append(Path(bundle.manifest_path))
    for clip in getattr(bundle, "clips", []):
        for field in ("video_path", "scene_reference_path", "annotation_path", "evaluation_path"):
            value = getattr(clip, field, None)
            if value:
                sources.append(Path(value))
    return sources


def _cvat_credentials(url: str) -> dict[str, str | None]:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "--url must be an HTTP(S) CVAT origin without credentials, query or fragment",
        )
    username = os.environ.get("ARGUS_CVAT_USERNAME") or None
    password = os.environ.get("ARGUS_CVAT_PASSWORD") or None
    token = os.environ.get("ARGUS_CVAT_TOKEN") or None
    if not token and not (username and password):
        raise ValueError("set ARGUS_CVAT_TOKEN or both ARGUS_CVAT_USERNAME and ARGUS_CVAT_PASSWORD")
    if token and (username or password):
        raise ValueError("use either ARGUS_CVAT_TOKEN or username/password, not both")
    return {"username": username, "password": password, "token": token}


def run(args: argparse.Namespace) -> None:
    from argus.integrations.contracts import load_bundle, save_bundle

    if args.command == "prepare":
        _check_output(
            args.output,
            overwrite=args.overwrite,
            inputs=[args.manifest, args.split_file],
        )
        from argus.integrations.corpus import prepare_review_bundle

        bundle = prepare_review_bundle(
            args.manifest,
            evaluation_dir=args.evaluation_dir,
            split_file=args.split_file,
        )
        protected = [args.manifest, args.split_file, *_bundle_sources(bundle)]
        if args.evaluation_dir is not None:
            for clip in bundle.clips:
                protected.extend([
                    args.evaluation_dir / f"{clip.clip_id}.json",
                    args.evaluation_dir / "video_eval" / f"{clip.clip_id}.json",
                ])
        _check_output(args.output, overwrite=args.overwrite, inputs=protected)
        save_bundle(bundle, args.output, overwrite=args.overwrite)
        print(f"Review bundle written to {args.output}")
        return

    bundle = load_bundle(args.bundle)
    if args.command == "cvat-push":
        if args.receipt.resolve() == args.bundle.resolve():
            raise ValueError("CVAT receipt must not replace the input bundle")
        from argus.integrations.cvat import export_bundle_to_cvat

        export_bundle_to_cvat(
            bundle,
            url=args.url,
            receipt_path=args.receipt,
            organization=args.organization,
            **_cvat_credentials(args.url),
        )
        print(f"CVAT tasks recorded in {args.receipt}")
    elif args.command == "cvat-pull":
        resolved_receipt = args.receipt.resolve()
        receipt_lock = resolved_receipt.with_name(resolved_receipt.name + ".lock")
        _check_output(
            args.output,
            overwrite=args.overwrite,
            inputs=[args.bundle, args.receipt, receipt_lock, *_bundle_sources(bundle)],
        )
        from argus.integrations.cvat import import_bundle_from_cvat

        reviewed = import_bundle_from_cvat(
            bundle,
            url=args.url,
            receipt_path=args.receipt,
            reviewed_by=args.reviewed_by,
            organization=args.organization,
            confirmed_normal_clip_ids=args.confirm_normal_clip,
            **_cvat_credentials(args.url),
        )
        save_bundle(reviewed, args.output, overwrite=args.overwrite)
        print(f"Reviewed bundle written to {args.output}")
    elif args.command == "export-labels":
        from argus.integrations.corpus import export_review_labels

        if (args.output_dir / "manifest.json").resolve() == args.bundle.resolve():
            raise ValueError("regression export must not replace its input review bundle")
        manifest = export_review_labels(
            bundle, args.output_dir, overwrite=args.overwrite, split=args.split,
            protected_paths=[args.bundle],
        )
        print(f"Regression manifest written to {manifest}")
    elif args.command == "scene-images-prepare":
        from argus.integrations.scene_comparison import load_scene_comparisons

        comparisons = load_scene_comparisons(bundle, args.scene_comparison)
        from argus.integrations.scene_review_images import prepare_scene_review_images

        manifest = prepare_scene_review_images(
            bundle,
            scene_comparisons=comparisons,
            output_dir=args.output_dir,
            selection=args.selection,
            max_images=args.max_images,
            max_output_bytes=args.max_output_bytes,
            protected_paths=[args.bundle],
        )
        print(f"Scene review image manifest written to {manifest}")
        print("Diagnostic images only; no source video, profile or SceneGuard gate was changed.")
    elif args.command == "scene-images-import":
        from argus.integrations.scene_comparison import load_scene_comparisons

        comparisons = load_scene_comparisons(bundle, args.scene_comparison)
        from argus.integrations.fiftyone_scene_images import import_scene_review_images

        dataset = import_scene_review_images(
            bundle,
            manifest_path=args.manifest,
            scene_comparisons=comparisons,
            dataset_name=args.dataset_name,
            persistent=True,
        )
        print(f"FiftyOne diagnostic image dataset imported: {args.dataset_name}")
        print(
            "Launch view: argus-scene-unreviewed. Review source frame index, exact PTS and "
            "report SHA; an empty view is not a pass or release approval."
        )
        if args.launch:
            import fiftyone as fo

            session = fo.launch_app(view=dataset.load_saved_view("argus-scene-unreviewed"))
            session.wait()
    elif args.command == "fiftyone-import":
        scene_paths = getattr(args, "scene_comparison", [])
        scene_arguments = {}
        if scene_paths:
            # Validate source identity, frame numbers and exact timestamps before
            # importing the optional SDK or making any database changes.
            from argus.integrations.scene_comparison import load_scene_comparisons

            scene_arguments["scene_comparisons"] = load_scene_comparisons(bundle, scene_paths)
        from argus.integrations.fiftyone import import_bundle_to_fiftyone

        dataset = import_bundle_to_fiftyone(
            bundle, dataset_name=args.dataset_name, persistent=True, **scene_arguments,
        )
        print(f"FiftyOne dataset created: {args.dataset_name}")
        if scene_paths:
            print(
                f"Scene comparison reports attached: {len(scene_paths)}; launch view: "
                "argus-scene-disagreements. Also inspect argus-scene-raw-disagreements "
                "and argus-scene-errors; an empty disagreement view is not a pass."
            )
        if args.launch:
            import fiftyone as fo

            initial_view = "argus-scene-disagreements" if scene_paths else "argus-current"
            session = fo.launch_app(view=dataset.load_saved_view(initial_view))
            session.wait()


def _safe_error(exc: Exception) -> str:
    try:
        message = str(exc)
    except BaseException:
        # Some SDK exceptions have a broken or side-effecting __str__. Do not
        # call it again via repr/formatting while handling the original failure.
        return f"{type(exc).__name__[:80]} (details unavailable)"
    return _redact(message)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", "") or ""
        if missing == "cvat_sdk" or missing.startswith("cvat_sdk."):
            hint = 'python -m pip install "cvat-sdk>=2.79,<2.80"'
        elif missing == "fiftyone" or missing.startswith("fiftyone."):
            hint = 'python -m pip install "fiftyone>=1.8,<2" (use a separate review venv)'
        elif missing == "av" or missing.startswith("av."):
            hint = 'python -m pip install "av>=12.0.0"'
        elif missing == "argus" or missing.startswith("argus."):
            hint = "set PYTHONPATH to this repository's src directory, or install Argus"
        else:
            hint = "install the missing dependency in the review environment (see runbook)"
        print(f"ERROR: optional dependency unavailable; {hint}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {_safe_error(exc)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
