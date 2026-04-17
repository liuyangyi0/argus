"""Build an evaluation set (y_true + y_scores) from production runtime data.

Joins AlertRecord + FeedbackRecord to extract:
    y_score  = AlertRecord.anomaly_score
    y_true   = 1 if FeedbackRecord.feedback_type == 'confirmed'
               0 if FeedbackRecord.feedback_type == 'false_positive'
(Entries with type 'uncertain' are skipped.)

Computes Precision / Recall / F1 / AUROC / PR-AUC on this derived set.
Optionally copies the snapshot images into a `tests/regression/{anomaly,normal}/`
style layout so production feedback automatically grows the regression corpus.

Usage:
    # Print markdown summary over last 30 days, all cameras
    python scripts/build_eval_set_from_inference_records.py --days 30

    # Filter by camera, write JSON and markdown
    python scripts/build_eval_set_from_inference_records.py \
        --camera-id c --days 60 \
        --output reports/eval_from_feedback.md \
        --json-out reports/eval_from_feedback.json

    # Also materialize a regression set from snapshots
    python scripts/build_eval_set_from_inference_records.py \
        --days 30 --copy-to tests/regression/from_feedback

Exits non-zero if F1 falls below --min-f1 (CI gate).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--database-url", default="sqlite:///data/db/argus.db")
    parser.add_argument("--camera-id", type=str, help="Filter by camera")
    parser.add_argument("--days", type=int, default=30, help="Look-back window (days)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Decision threshold for P/R/F1 (AUROC/PR-AUC are threshold-free)")
    parser.add_argument("--output", type=Path, help="Write markdown report here")
    parser.add_argument("--json-out", type=Path, help="Write raw scores+labels JSON here")
    parser.add_argument("--copy-to", type=Path,
                        help="Copy snapshot images into {anomaly,normal}/ subdirs here")
    parser.add_argument("--min-f1", type=float, default=0.0,
                        help="CI gate: exit non-zero if F1 < this value")
    args = parser.parse_args()

    # Defer imports so --help is fast even without the project env
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from argus.storage.models import AlertRecord, FeedbackRecord

    engine = create_engine(args.database_url)
    Session = sessionmaker(bind=engine)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    with Session() as session:
        # FeedbackRecord with concrete verdict
        fb_stmt = (
            select(FeedbackRecord)
            .where(FeedbackRecord.feedback_type.in_(["confirmed", "false_positive"]))
            .where(FeedbackRecord.alert_id.isnot(None))
        )
        if args.camera_id:
            fb_stmt = fb_stmt.where(FeedbackRecord.camera_id == args.camera_id)
        feedbacks = list(session.scalars(fb_stmt).all())

        alert_ids = [f.alert_id for f in feedbacks if f.alert_id]
        if not alert_ids:
            print("No feedback with alert_id found in window.", file=sys.stderr)
            return 2

        alerts_stmt = (
            select(AlertRecord)
            .where(AlertRecord.alert_id.in_(alert_ids))
            .where(AlertRecord.timestamp >= cutoff)
        )
        alerts = {a.alert_id: a for a in session.scalars(alerts_stmt).all()}

    # Build y_true / y_scores aligned
    rows: list[dict] = []
    for fb in feedbacks:
        alert = alerts.get(fb.alert_id)
        if alert is None:
            continue
        y = 1 if fb.feedback_type == "confirmed" else 0
        rows.append({
            "alert_id": fb.alert_id,
            "camera_id": alert.camera_id,
            "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
            "score": float(alert.anomaly_score),
            "label": y,
            "snapshot_path": alert.snapshot_path,
            "feedback_type": fb.feedback_type,
        })

    if len(rows) < 10:
        print(
            f"Only {len(rows)} (confirmed + false_positive) feedback entries with a "
            f"linked alert in the last {args.days} days — not enough to evaluate.",
            file=sys.stderr,
        )
        return 3

    import numpy as np

    from argus.anomaly.metrics import (
        compute_pr_auc,
        compute_pr_curve,
        evaluate_at_threshold,
        find_optimal_threshold,
    )

    y_scores = np.array([r["score"] for r in rows], dtype=np.float64)
    y_true = np.array([r["label"] for r in rows], dtype=np.int64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        print(
            f"Degenerate set: n_positive={n_pos}, n_negative={n_neg} — need ≥1 of each.",
            file=sys.stderr,
        )
        return 4

    metrics = evaluate_at_threshold(y_true, y_scores, args.threshold)
    optimal_t = find_optimal_threshold(y_true, y_scores, target="f1")
    pr_auc = compute_pr_auc(y_true, y_scores)  # same as metrics["pr_auc"], kept for symmetry
    pr_curve = compute_pr_curve(y_true, y_scores)

    # Markdown report
    cm = metrics["confusion_matrix"]
    md_lines = [
        "# Evaluation from Production Feedback",
        "",
        f"- **Source DB**: `{args.database_url}`",
        f"- **Window**: last {args.days} days",
        f"- **Camera filter**: {args.camera_id or '(all)'}",
        f"- **Feedback rows used**: {len(rows)} (positives={n_pos}, negatives={n_neg})",
        f"- **Threshold**: {args.threshold:.4f}",
        f"- **Optimal F1 threshold (sweep)**: {optimal_t:.4f}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| AUROC | {metrics['auroc']:.4f} |",
        f"| PR-AUC | {pr_auc:.4f} |",
        "",
        "## Confusion Matrix (@ threshold)",
        "",
        "| | Pred anomaly | Pred normal |",
        "|---|---|---|",
        f"| **GT anomaly** | TP = {cm['tp']} | FN = {cm['fn']} |",
        f"| **GT normal**  | FP = {cm['fp']} | TN = {cm['tn']} |",
    ]
    report = "\n".join(md_lines)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Markdown report → {args.output}")
    else:
        print(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": metrics,
            "pr_curve": pr_curve,
            "optimal_f1_threshold": optimal_t,
            "pr_auc": pr_auc,
            "rows": rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"JSON payload → {args.json_out}", file=sys.stderr)

    if args.copy_to:
        pos_dir = args.copy_to / "anomaly"
        neg_dir = args.copy_to / "normal"
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)
        copied_pos = 0
        copied_neg = 0
        for r in rows:
            src = r.get("snapshot_path")
            if not src:
                continue
            src_path = Path(src)
            if not src_path.is_file():
                continue
            dest_dir = pos_dir if r["label"] == 1 else neg_dir
            dest = dest_dir / f"{r['alert_id']}{src_path.suffix}"
            try:
                shutil.copyfile(src_path, dest)
                if r["label"] == 1:
                    copied_pos += 1
                else:
                    copied_neg += 1
            except Exception as e:
                print(f"copy failed: {src_path} → {dest}: {e}", file=sys.stderr)
        print(
            f"Copied to {args.copy_to}: {copied_pos} positive, {copied_neg} negative "
            f"(skipped missing snapshots)",
            file=sys.stderr,
        )

    if metrics["f1"] < args.min_f1:
        print(
            f"FAIL: F1 {metrics['f1']:.4f} < min_f1 {args.min_f1}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
