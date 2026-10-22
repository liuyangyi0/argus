# Argus Regression Handoff - 2026-06-13

This file is the durable entry point for the real-video regression corpus
captured on 2026-06-13. Raw videos and generated reports are intentionally
ignored by git; this file records where they are, how to restore them, and what
baseline numbers were observed.

## Commits

- Phase 0 clean baseline: `2110d43 Freeze detection quality baseline`
- Phase 1 regression suite and tracked corpus index: `ca96c94 Add real video regression suite`
- Handoff and resume notes: `75150f1 Document regression corpus handoff`
- Detection convergence plan status: `dfb9bcc Update detection convergence plan status`
- Preferred future capture mode: `d622335 Record preferred regression capture framerate`
- Phase 2 GPU baseline results: `30a1e4a Record phase 2 GPU baseline results`
- Static label tightening: `3e739c3 Tighten static regression labels`

Use `ca96c94` or later when running `scripts/run_regression_suite.py`; earlier
commits do not include the suite runner, manifest, labels, or this handoff.

Related plan:

- `docs/design/recovery-plan-2026-06-12.md`

## Corpus Locations

Tracked files:

- `data/regression/manifest.yaml`
- `data/regression/capture_notes.md`
- `data/regression/annotations/pos_static_foreign_10events_wide_20260613_103042.labels.json`
- `data/regression/annotations/pos_projectile_10events_wide_20260613_105301.labels.json`

Ignored local videos:

- `data/regression/clips/negative/neg_scene_001_30min_20260613_095718.mp4`
- `data/regression/clips/positive/pos_static_foreign_10events_wide_20260613_103042.mp4`
- `data/regression/clips/positive/pos_projectile_10events_wide_20260613_105301.mp4`

Remote source copy:

- Host: `gpu-validation`
- Directory: `/srv/argus-regression-capture`
- Camera: OBSBOT Meet 2, `/dev/video0`, MJPEG 1920x1080 at 30 FPS
- Preferred future capture mode: MJPEG 1920x1080 at 60 FPS. This was verified
  on `gpu-validation` on 2026-06-13. Requests for 120 FPS are reduced by the driver to
  60 FPS; 4K is limited to 30 FPS; YUYV is limited to 30 FPS.

Restore videos from the test machine if the local ignored copies are missing:

```powershell
New-Item -ItemType Directory -Force data\regression\clips\negative | Out-Null
New-Item -ItemType Directory -Force data\regression\clips\positive | Out-Null
scp gpu-validation:/srv/argus-regression-capture/clips/negative/neg_scene_001_30min_20260613_095718.mp4 data/regression/clips/negative/
scp gpu-validation:/srv/argus-regression-capture/clips/positive/pos_static_foreign_10events_wide_20260613_103042.mp4 data/regression/clips/positive/
scp gpu-validation:/srv/argus-regression-capture/clips/positive/pos_projectile_10events_wide_20260613_105301.mp4 data/regression/clips/positive/
```

## Captured Clips

| Clip | Kind | Duration | Frames | Notes |
|---|---|---:|---:|---|
| `neg_scene_001_30min_20260613_095718` | negative | 1800.000s | 54000 | lights on/off, people, reflection, low light, slight vibration, stable periods |
| `pos_static_foreign_10events_wide_20260613_103042` | static foreign object | 1199.966667s | 35999 | 10 coarse positive windows; operator reported small second-level timing error |
| `pos_projectile_10events_wide_20260613_105301` | projectile / fast motion | 779.933333s | 23398 | 10 coarse positive windows; requirement is still provisional |

The positive labels are coarse second-based windows. Before using them as a
strict frame-level gate, visually review and tighten boundaries.

The clips captured on 2026-06-13 are 1920x1080 at 30 FPS. New regression clips,
especially projectile / fast-motion clips, should use MJPEG 1920x1080 at 60 FPS.

## Verification Commands

Validate inventory without inference:

```powershell
.\.venv\Scripts\python.exe scripts\run_regression_suite.py `
  --manifest data\regression\manifest.yaml `
  --dry-run `
  --output-dir data\regression\reports\capture-dry-run
```

Verify video metadata:

```powershell
ffprobe -v error -select_streams v:0 `
  -show_entries stream=width,height,r_frame_rate,duration,nb_frames `
  -of json data\regression\clips\negative\neg_scene_001_30min_20260613_095718.mp4
```

Run the first fallback baseline, sampled at 1 FPS:

```powershell
.\.venv\Scripts\python.exe scripts\run_regression_suite.py `
  --manifest data\regression\manifest.yaml `
  --sample-every 30 `
  --sweep `
  --output-dir data\regression\reports\fallback-ssim-sample30
```

Re-summarize existing per-video JSON without rerunning video inference:

```powershell
.\.venv\Scripts\python.exe scripts\run_regression_suite.py `
  --from-eval-json data\regression\reports\fallback-ssim-sample30\video_eval\neg_scene_001_30min_20260613_095718.json `
  --from-eval-json data\regression\reports\fallback-ssim-sample30\video_eval\pos_static_foreign_10events_wide_20260613_103042.json `
  --from-eval-json data\regression\reports\fallback-ssim-sample30\video_eval\pos_projectile_10events_wide_20260613_105301.json `
  --output-dir data\regression\reports\fallback-ssim-sample30-ca96c94
```

## Initial Baseline Numbers

Report path, if still present locally:

- `data/regression/reports/fallback-ssim-sample30-ca96c94/regression_report.md`

Configuration:

- Commit reported by re-summary: `ca96c94`
- Detector: SSIM fallback, no model path
- Threshold: `0.7`
- Sampling: `sample_every=30` at 30 FPS, so approximately 1 FPS
- FP grouping gap: `60s`

Suite result:

| Metric | Target | Actual | Status |
|---|---:|---:|---|
| False positives/hour | <= 2.00 | 11.5044 | fail |
| Event recall | >= 0.95 | 0.5000 | fail |
| P95 detection delay | <= 10.00s | 43.0000s | fail |

Per-clip result:

| Clip | FP events | FP/hour | Recall | P95 delay |
|---|---:|---:|---:|---:|
| `neg_scene_001_30min_20260613_095718` | 1 | 2.0000 | - | - |
| `pos_static_foreign_10events_wide_20260613_103042` | 7 | 33.6164 | 0.5000 | 43.0000s |
| `pos_projectile_10events_wide_20260613_105301` | 2 | 12.4224 | 0.5000 | 10.6000s |

Channel attribution in this fallback run was all `anomaly`, because no
pipeline inference-record candidate events were supplied.

## Next Decisions

- Confirm whether projectile / fast small object remains an acceptance
  requirement. If not, remove or separate it from go/no-go metrics.
- Select the real deployed model path and rerun the suite with `--model`.
- Visually review positive windows and refine labels before enforcing strict
  frame-level metrics.
- When pipeline inference records are available, add `candidate_events` paths
  to `manifest.yaml` so FP attribution can distinguish `fast_motion`,
  `simplex`, `persistent_change`, `ssim_fallback`, and `anomaly`.

## Phase 2 Test-Machine Run - 2026-06-13

Test machine rebuild:

- Host: `gpu-validation`
- Old dirty repository backup: `/srv/argus.backup-20260613-115329`
- New repository: `/srv/argus`
- New repository commit: `d622335`
- New repository branch: `main`
- New repository status after rebuild and dry-run: clean
- Reused virtualenv: `/srv/argus/.venv`
- GPU verification: `torch_cuda=True`, `NVIDIA GeForce RTX 3060`
- Regression videos in the new repository are symlinks to
  `/srv/argus-regression-capture`; large videos were not copied into git.

Remote verification commands:

```bash
cd /srv/argus
git status --short
git rev-parse --short HEAD
.venv/bin/python - <<'PY'
import torch, anomalib, openvino, cv2
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
.venv/bin/python scripts/run_regression_suite.py \
  --manifest data/regression/manifest.yaml \
  --dry-run \
  --output-dir data/regression/reports/remote-dry-run
```

Observed verification:

- `git status --short`: clean
- `git rev-parse --short HEAD`: `d622335`
- Dry-run report status: `ok`, 3 clips listed, no missing inputs
- Existing captured clips are 1920x1080 at 30 FPS. Future captures should use
  MJPEG 1920x1080 at 60 FPS.

Generated label review images:

- Remote: `/srv/argus/data/regression/review/label-window-review-20260613`
- Remote: `/srv/argus/data/regression/review/label-window-motion-review-20260613`
- Local ignored copies: `data/regression/review/`

Training input:

- Baseline source: `neg_scene_001_30min_20260613_095718`
- Baseline directory: `/srv/argus/data/baselines/validation_camera_01/default/v001`
- Extracted images: 300
- Person filtering: YOLO loaded on CUDA; no accepted candidate was rejected as
  person in this extraction pass.
- Baseline validation was intentionally skipped for training. The fixed-camera
  negative clip is visually stable enough that the generic quality gate reported
  `brightness_std=0.9273` and `near_duplicate_rate=0.98`, which is expected for
  this kind of normal scene and not a useful blocker for the first PatchCore
  baseline.

PatchCore training:

- Remote training report:
  `/srv/argus/data/regression/reports/phase2_patchcore_20260613/training_result.json`
- Local ignored copy:
  `data/regression/reports/phase2_patchcore_20260613/training_result.json`
- Model output:
  `/srv/argus/data/models/phase2_patchcore_20260613/validation_camera_01/default`
- OpenVINO export:
  `/srv/argus/data/exports/phase2_patchcore_20260613/validation_camera_01/default/weights/openvino/model.xml`
- OpenVINO static-shape guard: passed
- Training status: complete
- Training duration: 83.124s
- Train/val split: 240 / 60
- Quality grade: B
- Recommended threshold: `0.5298`
- Validation score p95: `0.4565`
- Output validation: checkpoint valid, export valid, smoke test passed
- Calibration status: ok

Static-object main-model regression:

- Report:
  `/srv/argus/data/regression/reports/phase2_patchcore_static_sample30_t05298/regression_report.md`
- Local ignored copy:
  `data/regression/reports/phase2_patchcore_static_sample30_t05298/regression_report.md`
- Scope: negative clip + static positive clip only
- Model: PatchCore OpenVINO export above
- Threshold: `0.5298`
- Sampling: `sample_every=30`

| Metric | Actual | Status |
|---|---:|---|
| False positives/hour | 11.2957 | fail |
| Static event recall | 1.0000 | pass |
| P95 detection delay | 42.5500s | fail |

Per-clip:

| Clip | FP events | FP/hour | Recall | P95 delay |
|---|---:|---:|---:|---:|
| `neg_scene_001_30min_20260613_095718` | 4 | 8.0000 | - | - |
| `pos_static_foreign_10events_wide_20260613_103042` | 4 | 19.2094 | 1.0000 | 42.5500s |

Interpretation: the first real PatchCore model can hit all coarse static
events, but it does not yet reduce FP/hour enough. The high delay is not yet a
reliable model-quality signal because the static labels are still 45-second
coarse windows and need visual tightening.

Static-object main-model regression after tightened labels:

- Label commit: `3e739c3`
- Label source: 1 FPS visual review of stable-object windows, excluding obvious
  operator placement/removal timing.
- Remote report:
  `/srv/argus/data/regression/reports/phase2_patchcore_static_tight_labels_manifest_sample30_t05298/regression_report.md`
- Local ignored copy:
  `data/regression/reports/phase2_patchcore_static_tight_labels_manifest_sample30_t05298/regression_report.md`
- Scope: negative clip + static positive clip only
- Model: same PatchCore OpenVINO export above
- Threshold: `0.5298`
- Sampling: `sample_every=30`

| Metric | Actual | Status |
|---|---:|---|
| False positives/hour | 8.1768 | fail |
| Static event recall | 0.2000 | fail |
| P95 detection delay | 0.0000s | pass |

Per-clip:

| Clip | FP events | FP/hour | Recall | P95 delay |
|---|---:|---:|---:|---:|
| `neg_scene_001_30min_20260613_095718` | 4 | 8.0000 | - | - |
| `pos_static_foreign_10events_wide_20260613_103042` | 2 | 8.5548 | 0.2000 | 0.0000s |

Important report note: the earlier report path
`phase2_patchcore_static_tight_labels_sample30_t05298` was summarized from
`--from-eval-json` after `evaluate_video.py` had run the negative clip without
an annotation file. That left negative frames as `unlabeled`, so negative false
positives were not counted. Do not use that report for FP/hour.

Per-event stable-window result at threshold `0.5298`:

| Event | Window (s) | Hit | Max score |
|---|---:|---|---:|
| `static_foreign_event_01` | 140-164 | no | 0.3310 |
| `static_foreign_event_02` | 225-270 | yes | 0.6691 |
| `static_foreign_event_03` | 340-372 | no | 0.3376 |
| `static_foreign_event_04` | 444-477 | no | 0.3218 |
| `static_foreign_event_05` | 541-582 | no | 0.3410 |
| `static_foreign_event_06` | 653-690 | yes | 0.7286 |
| `static_foreign_event_07` | 771-795 | no | 0.3172 |
| `static_foreign_event_08` | 860-900 | no | 0.3186 |
| `static_foreign_event_09` | 960-1001 | no | 0.3474 |
| `static_foreign_event_10` | 1069-1110 | no | 0.3441 |

Interpretation: the coarse-window recall of `1.0000` was not a stable-object
quality signal. After tightening the labels, the current PatchCore model only
hits 2 of 10 stable static-object events. Use this tightened-label report as
the current main-model static baseline. Do not use the coarse-window recall for
go/no-go decisions.

Root-cause investigation for low stable-object scores:

- The OpenVINO export used by regression did not have `calibration.json` in the
  detector search path. Training saved calibration at
  `/srv/argus/data/models/phase2_patchcore_20260613/validation_camera_01/default/calibration.json`,
  but the evaluated model path was
  `/srv/argus/data/exports/phase2_patchcore_20260613/validation_camera_01/default/weights/openvino/model.xml`.
  Detector logs showed `anomaly.no_calibration` for the uncalibrated run.
- The recommended threshold `0.5298` came from validation-normal scores
  (`max=0.5045`, then `max * 1.05`). Most stable small-object events score only
  `0.317-0.347` raw, which is inside the normal validation distribution rather
  than above it. At that threshold they are intentionally rejected.
- Training input quality was weak for calibration: the baseline quality gate
  reported `near_duplicate_rate=0.98` and `brightness_std=0.9273`, then was
  skipped for the first Phase 2 baseline. Calibration used only 60 validation
  normal scores from this narrow split, so it is not representative of the full
  negative regression clip with light changes, people, reflections, and slight
  vibration.
- Object scale is a real contributor. The missed stable objects are small after
  the fixed `256x256` OpenVINO resize: visual diff estimates put most missed
  objects at roughly `7-25 px` wide and `31-37 px` high in model input space.
  The only continuously raw-threshold hit is event 06, a much larger dark object
  around `40x112 px` in model input space. Event 02 only exceeds threshold at
  the first second of the window and may still be contaminated by operator
  motion.

Calibration experiment:

- For diagnosis only, `calibration.json` was copied into the OpenVINO directory
  on `gpu-validation` so the detector could load it:
  `/srv/argus/data/exports/phase2_patchcore_20260613/validation_camera_01/default/weights/openvino/calibration.json`
- Remote report:
  `/srv/argus/data/regression/reports/phase2_patchcore_static_tight_labels_calibrated_sample30_t05298/regression_report.md`
- Local ignored copy:
  `data/regression/reports/phase2_patchcore_static_tight_labels_calibrated_sample30_t05298/regression_report.md`

| Metric | Actual | Status |
|---|---:|---|
| False positives/hour | 4.0884 | fail |
| Static event recall | 1.0000 | pass |
| P95 detection delay | 13.5500s | fail |

Interpretation: loading calibration proves the model has enough signal to hit
all static events at an event level, but the current calibration/threshold
still fails FP/hour and delay. The next fix should not be "just copy
calibration and ship"; it should make calibration/export reproducible and then
retrain or recalibrate from a broader negative set before selecting thresholds
on the regression suite.

Fast-motion projectile regression:

- Report:
  `/srv/argus/data/regression/reports/phase2_fast_motion_projectile_default/regression_report.md`
- Local ignored copy:
  `data/regression/reports/phase2_fast_motion_projectile_default/regression_report.md`
- Detector: `FastMotionDetector` defaults
- Input FPS: 30 FPS. This is below `FastMotionConfig.min_runtime_fps=50` for
  low-miss fast-motion mode.

| Metric | Actual |
|---|---:|
| False positives/hour | 31.7700 |
| Projectile event recall | 0.5000 |
| P95 detection delay | 18.4533s |

Per-clip:

| Clip | Detection frames | FP events | FP/hour | Recall | P95 delay |
|---|---:|---:|---:|---:|---:|
| `neg_scene_001_30min_20260613_095718` | 178 | 14 | 28.0000 | - | - |
| `pos_projectile_10events_wide_20260613_105301` | 62 | 7 | 43.4783 | 0.5000 | 18.4533s |

Interpretation: default fast-motion does not meet projectile acceptance on the
current 30 FPS clips. Keep projectile as a separate detection line, but recapture
projectile material at MJPEG 1920x1080 60 FPS before using it as a strict
acceptance gate.

Phase 2 v002 existing-data PatchCore run:

- Date: 2026-06-13
- Commit: `6716876`
- Source negative clip:
  `/srv/argus/data/regression/clips/negative/neg_scene_001_30min_20260613_095718.mp4`
- Full 1 FPS baseline:
  `/srv/argus/data/baselines/validation_camera_01/default/v002_existing_1fps`
- Full 1 FPS training attempt:
  `/srv/argus/data/regression/reports/phase2_patchcore_v002_existing_20260613_152941/training_result.json`
- Result: failed after training with CUDA OOM during validation/export. It used
  1800 images, split 1440 / 360. Pre-validation also failed because the normal
  scene is intentionally stable (`brightness_std=0.9869`,
  `near_duplicate_rate=99.1%`).

Trainable subset:

- Subset baseline:
  `/srv/argus/data/baselines/validation_camera_01/default/v002_existing_subset450`
- Selection: 450 frames from the 1 FPS baseline, mixing time coverage, pHash
  unique frames, brightness/texture extremes, and transition windows.
- Training report:
  `/srv/argus/data/regression/reports/phase2_patchcore_v002_subset_20260613_153551/training_result.json`
- Model directory:
  `/srv/argus/data/models/phase2_patchcore_v002_subset_20260613_153551/validation_camera_01/default`
- OpenVINO export:
  `/srv/argus/data/exports/phase2_patchcore_v002_subset_20260613_153551/validation_camera_01/default/weights/openvino/model.xml`
- Calibration copied by export fix:
  `/srv/argus/data/exports/phase2_patchcore_v002_subset_20260613_153551/validation_camera_01/default/weights/openvino/calibration.json`
- Training status: complete
- Duration: 160.08s
- Train/val split: 360 / 90
- Quality grade: F
- Recommended threshold: `0.7866`
- Calibration status: ok
- Note: generic pre-validation still failed for a fixed, stable scene
  (`brightness_std=1.6817`, `near_duplicate_rate=96.4444%`). This is recorded,
  not hidden.

Static regression for v002 subset model:

| Threshold | FP/hour | Static recall | P95 delay | Report |
|---:|---:|---:|---:|---|
| `0.7866` | 8.1768 | 0.2000 | 0.0000s | `/srv/argus/data/regression/reports/phase2_patchcore_v002_subset_20260613_153551_static_tight_sample30_t07866/regression_report.md` |
| `0.5000` | 2.7256 | 1.0000 | 4.5500s | `/srv/argus/data/regression/reports/phase2_patchcore_v002_subset_20260613_153551_static_tight_sample30_t05000/regression_report.md` |

Threshold sweep from the v002 regression JSON:

| Threshold | FP/hour | Static recall | P95 delay |
|---:|---:|---:|---:|
| `0.50` | 2.7256 | 1.0000 | 4.55s |
| `0.55` | 4.0884 | 1.0000 | 4.55s |
| `0.60` | 4.0884 | 1.0000 | 6.65s |
| `0.65` | 4.0884 | 1.0000 | 11.85s |
| `0.70` | 5.4512 | 1.0000 | 11.85s |
| `0.73` | 8.1768 | 1.0000 | 20.40s |
| `0.74` | 9.5396 | 0.9000 | - |
| `0.75` | 14.9907 | 0.8000 | - |
| `0.76` | 9.5396 | 0.3000 | - |
| `0.7866` | 8.1768 | 0.2000 | 0.00s |

Interpretation: v002 is much better than the first calibrated model for static
recall and delay at threshold `0.5`, but still misses the Phase 0 FP target
(`2.7256/h` vs `<=2/h`). Treat it as an inspectable candidate, not acceptance
passing.

Live test-machine service for visual inspection:

- Host: `gpu-validation` / `192.0.2.10`
- URL: `http://192.0.2.10:18080/`
- PID file:
  `/srv/argus/data/logs/phase2_v002_argus.pid`
- Log:
  `/srv/argus/data/logs/phase2_v002_argus.out`
- Runtime config:
  `/srv/argus/data/runtime/phase2_v002_static_model.yaml`
- External go2rtc config:
  `/srv/argus/data/runtime/go2rtc_exec_obsbot.yaml`
- go2rtc binary:
  `/srv/argus/data/runtime/go2rtc/v1.9.14/go2rtc`
- Runtime model symlink:
  `/srv/argus/data/exports/validation_camera_01 -> /srv/argus/data/exports/phase2_patchcore_v002_subset_20260613_153551/validation_camera_01`
- Camera mode: OBSBOT Meet 2, USB MJPEG 1920x1080 at 60 FPS through external
  go2rtc RTSP (`rtsp://127.0.0.1:8554/validation_camera_01`).
- go2rtc note: v1.9.14 starts correctly, but its built-in `ffmpeg:device`
  USB source rejects Ubuntu ffmpeg `4.4.2` with `unsupported version`. The
  live run therefore uses an `exec:ffmpeg ... {output}` source with
  `h264_nvenc`; `ffprobe` verified H.264 1920x1080 at 60 FPS.
- Observed status after go2rtc RTSP startup: reported FPS 60.0, actual FPS
  60.0, backend `ffmpeg`, pixel format `h264`, OpenVINO model loaded,
  calibration loaded with 90 samples, anomaly inference failures 0.
- Config intentionally disables `simplex`, `persistent_change`, and
  `fast_motion` so this demo shows the PatchCore static-object line only.
- The service initially entered learning mode, then was switched to active with
  `POST /api/cameras/validation_camera_01/mode {"mode":"active"}` so live
  inspection does not need to wait 10 minutes.

Immediate next step:

- Use the live v002 service to visually inspect static-object behavior on the
  test machine. It is currently PatchCore-only and should be set to active mode
  after restart.
- If this go2rtc RTSP path is promoted beyond inspection, codify the USB
  `exec:ffmpeg` source or upgrade the test machine ffmpeg; do not rely on the
  go2rtc built-in `ffmpeg:device` source with Ubuntu ffmpeg `4.4.2`.
- Continue static-object convergence from the v002 threshold `0.5` baseline:
  recall and delay are acceptable, but FP/hour is still above target
  (`2.7256/h` vs `<=2/h`).
- Rebuild or extend the normal training/calibration set with more coverage of
  light changes, reflections, people near the edge, and slight vibration.
  Do not skip the baseline quality warning for the next candidate without
  recording why.
- Keep projectile metrics separate from static-object go/no-go. Re-record
  projectile clips at MJPEG 1920x1080 60 FPS, then re-run fast-motion regression
  before tuning.
