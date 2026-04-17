"""Offline precision/recall/F1/AUROC/PR-AUC evaluation on a regression set.

Usage:
    # Explicit dirs
    python scripts/evaluate_regression.py \
        --model data/models/latest/model.xml \
        --positive-dir tests/regression/anomaly \
        --negative-dir tests/regression/normal

    # From standard project layout (auto-resolves camera dirs)
    python scripts/evaluate_regression.py \
        --model data/models/latest/model.xml \
        --camera-id c

    # Sweep threshold and find optimal F1
    python scripts/evaluate_regression.py \
        --model data/models/latest/model.xml \
        --camera-id c \
        --sweep

Outputs a markdown report to stdout (or --output path). Exits non-zero if F1
drops below --min-f1, letting this script double as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _tabulate_pr_curve(curve: dict, n_points: int = 10) -> str:
    precisions = curve["precisions"]
    recalls = curve["recalls"]
    thresholds = curve["thresholds"]
    if not precisions:
        return "(empty curve)"
    step = max(1, len(precisions) // n_points)
    rows = ["| threshold | precision | recall |", "|---|---|---|"]
    for i in range(0, len(precisions), step):
        rows.append(
            f"| {thresholds[i]:.4f} | {precisions[i]:.4f} | {recalls[i]:.4f} |"
        )
    return "\n".join(rows)


def build_report(
    model_path: Path,
    positive_dir: Path,
    negative_dir: Path,
    metrics: dict,
    pr_curve: dict,
    threshold: float,
    optimal_threshold: float | None = None,
) -> str:
    cm = metrics["confusion_matrix"]
    lines = [
        f"# Regression Evaluation Report",
        "",
        f"- **Model**: `{model_path}`",
        f"- **Positive dir**: `{positive_dir}` ({metrics['n_positive']} images)",
        f"- **Negative dir**: `{negative_dir}` ({metrics['n_negative']} images)",
        f"- **Threshold**: {threshold:.4f}",
    ]
    if optimal_threshold is not None:
        lines.append(f"- **Optimal F1 threshold (sweep)**: {optimal_threshold:.4f}")
    lines += [
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| AUROC | {metrics['auroc']:.4f} |",
        f"| PR-AUC | {metrics['pr_auc']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "| | Pred anomaly | Pred normal |",
        "|---|---|---|",
        f"| **GT anomaly** | TP = {cm['tp']} | FN = {cm['fn']} |",
        f"| **GT normal**  | FP = {cm['fp']} | TN = {cm['tn']} |",
        "",
        "## PR Curve (sampled)",
        "",
        _tabulate_pr_curve(pr_curve),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model file or dir")
    parser.add_argument("--positive-dir", type=Path, help="Directory of anomaly images (label=1)")
    parser.add_argument("--negative-dir", type=Path, help="Directory of normal images (label=0)")
    parser.add_argument("--camera-id", type=str, help="Resolve pos/neg from standard project layout")
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Data root for --camera-id layout (default: data)")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sweep", action="store_true",
                        help="Also search for optimal F1 threshold")
    parser.add_argument("--min-f1", type=float, default=0.0,
                        help="CI gate: exit non-zero if F1 < this value")
    parser.add_argument("--output", type=Path, help="Write markdown report to this file")
    parser.add_argument("--json", dest="json_output", type=Path,
                        help="Also dump metrics + scores as JSON here")
    args = parser.parse_args()

    # Resolve dirs
    if args.camera_id:
        positive_dir = args.data_root / "validation" / args.camera_id / "confirmed"
        negative_dir = args.data_root / "baselines" / args.camera_id / "false_positives"
    else:
        if args.positive_dir is None or args.negative_dir is None:
            parser.error("Must provide --camera-id OR both --positive-dir and --negative-dir")
        positive_dir = args.positive_dir
        negative_dir = args.negative_dir

    if not positive_dir.is_dir():
        print(f"ERROR: positive_dir does not exist: {positive_dir}", file=sys.stderr)
        return 2
    if not negative_dir.is_dir():
        print(f"ERROR: negative_dir does not exist: {negative_dir}", file=sys.stderr)
        return 2

    from argus.anomaly.detector import AnomalibDetector
    from argus.anomaly.metrics import compute_pr_curve, find_optimal_threshold
    from argus.validation.recall_test import evaluate_metrics

    detector = AnomalibDetector(
        model_path=args.model,
        threshold=args.threshold,
        image_size=(args.image_size, args.image_size),
    )
    if not detector.load():
        print(f"ERROR: failed to load model: {args.model}", file=sys.stderr)
        return 3

    metrics = evaluate_metrics(
        detector,
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        threshold=args.threshold,
    )
    if metrics is None:
        print("ERROR: insufficient samples for evaluation", file=sys.stderr)
        return 4

    import numpy as np
    y_true = np.array(metrics["labels"])
    y_scores = np.array(metrics["scores"])
    pr_curve = compute_pr_curve(y_true, y_scores)

    optimal_t = None
    if args.sweep:
        optimal_t = find_optimal_threshold(y_true, y_scores, target="f1")

    report = build_report(
        model_path=args.model,
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        metrics=metrics,
        pr_curve=pr_curve,
        threshold=args.threshold,
        optimal_threshold=optimal_t,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in metrics.items()}
        payload["pr_curve"] = pr_curve
        if optimal_t is not None:
            payload["optimal_threshold"] = optimal_t
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if metrics["f1"] < args.min_f1:
        print(
            f"FAIL: F1 {metrics['f1']:.4f} < min_f1 {args.min_f1}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
