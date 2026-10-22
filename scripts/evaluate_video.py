"""Offline frame-level evaluation for a labeled video.

Example:
    python scripts/evaluate_video.py \\
        --video data/validation/cam_01/leak_test.avi \\
        --annotations data/validation/cam_01/leak_test.labels.json \\
        --model data/models/latest/model.xml \\
        --export-dir data/review/leak_test \\
        --output-json data/reports/leak_test.video_eval.json

The annotation file uses inclusive frame ranges:
    {
      "positive_ranges": [{"start_frame": 120, "end_frame": 180}],
      "ignore_ranges": [{"start_sec": 40.0, "end_sec": 42.5}]
    }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_report(result: dict) -> str:
    metrics = result.get("metrics")
    lines = [
        "# Offline Video Evaluation",
        "",
        f"- **Video**: `{result['video_path']}`",
        f"- **Annotations**: `{result.get('annotation_path') or '-'}`",
        f"- **Threshold**: {result['threshold']:.4f}",
        f"- **Sample every**: {result['sample_every']} frame(s)",
        f"- **Scored frames**: {result['scored_frames']}",
        f"- **Ignored frames**: {result['ignored_frames']}",
        f"- **Failed frames**: {len(result['failed_frames'])}",
        f"- **False positives**: {len(result['false_positive_frames'])}",
        f"- **False negatives**: {len(result['false_negative_frames'])}",
    ]
    if result.get("optimal_threshold") is not None:
        lines.append(f"- **Optimal F1 threshold**: {result['optimal_threshold']:.4f}")

    if metrics is None:
        lines += ["", "No labels were provided; metrics were not computed."]
        return "\n".join(lines)

    cm = metrics["confusion_matrix"]
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
        f"| **GT normal** | FP = {cm['fp']} | TN = {cm['tn']} |",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--model", type=Path, help="Model file or directory; omitted uses fallback detector")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--export-dir", type=Path, help="Write FP/FN review frames here")
    parser.add_argument("--output-json", type=Path, help="Write full JSON result here")
    parser.add_argument("--output-md", type=Path, help="Write markdown summary here")
    parser.add_argument("--sweep", action="store_true", help="Also compute best-F1 threshold")
    parser.add_argument("--min-f1", type=float, default=0.0, help="Exit 1 if labeled F1 is below this")
    args = parser.parse_args()

    from argus.anomaly.detector import AnomalibDetector
    from argus.validation.video_eval import evaluate_video, write_video_evaluation_report

    detector = AnomalibDetector(
        model_path=args.model,
        threshold=args.threshold,
        image_size=(args.image_size, args.image_size),
    )
    if not detector.load() and args.model is not None:
        print(f"ERROR: failed to load detector for model: {args.model}", file=sys.stderr)
        return 3

    result = evaluate_video(
        detector,
        args.video,
        annotation_path=args.annotations,
        threshold=args.threshold,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        export_dir=args.export_dir,
        sweep=args.sweep,
    )
    report = build_report(result)

    if args.output_json:
        write_video_evaluation_report(result, args.output_json)
        print(f"JSON report written to {args.output_json}")

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
        print(f"Markdown report written to {args.output_md}")
    else:
        print(report)

    metrics = result.get("metrics")
    if metrics is None:
        return 0
    if float(metrics["f1"]) < args.min_f1:
        print(
            f"FAIL: F1 {metrics['f1']:.4f} < min_f1 {args.min_f1}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
