# Argus Regression Corpus

This directory is the Phase 1 home for real-site video regression material.
Keep large videos local or on NAS; keep manifests and labels in git when they
do not expose sensitive site data.

Start with `HANDOFF.md` when resuming the 2026-06-13 real-video corpus. It
records the source host, restore commands, commit IDs, and first baseline
numbers.

## Required Layout

```text
data/regression/
  README.md
  HANDOFF.md                    # where to resume the 2026-06-13 corpus work
  manifest.yaml                  # real suite, created from manifest.example.yaml
  manifest.example.yaml          # tracked template
  capture_notes.md               # capture details for the current real corpus
  annotations/
    *.labels.json                # optional, tracked when safe
  clips/                         # ignored: raw videos
  reports/                       # ignored: generated baseline reports
  review/                        # ignored: exported FP/FN frames
```

## Minimum Capture Set

Negative clips:

- At least 30 minutes per target scene.
- Must cover lights on/off, people walking through, reflections, low light,
  slight camera vibration, and daylight changes.
- If the full clip is normal, annotations can be omitted; the suite runner will
  create a temporary all-normal label file. Add `ignore_ranges` only for invalid
  periods such as camera handling or blocked lens.

Positive clips:

- Static foreign object: at least 10 clips, and at least 20 annotated events
  before declaring acceptance coverage.
- Projectile / fast small object: at least 10 clips if this requirement is kept.
- Use inclusive frame ranges in the same schema accepted by
  `scripts/evaluate_video.py`.

## Annotation Schema

```json
{
  "positive_ranges": [
    {
      "start_frame": 120,
      "end_frame": 300,
      "label": "static_foreign"
    }
  ],
  "ignore_ranges": [
    { "start_sec": 40.0, "end_sec": 42.5, "label": "camera_adjustment" }
  ]
}
```

`start_frame` / `end_frame` are inclusive. `start_sec` / `end_sec` are accepted
aliases and are converted using the video FPS.

## Running the Suite

```powershell
python scripts/run_regression_suite.py `
  --manifest data/regression/manifest.yaml `
  --model data/models/latest/model.xml `
  --output-dir data/regression/reports/current
```

Useful modes:

- `--dry-run`: validate the manifest and report clip inventory without loading a model.
- `--from-eval-json <file>`: summarize existing `evaluate_video.py` JSON output.
- `--allow-missing`: keep missing clips in the report instead of failing fast.

The runner reports event-level false positives per hour and event-level recall.
Frame-level FP/FN from `evaluate_video.py` are only intermediate data.
