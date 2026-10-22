# Argus Phase 1 Capture Notes - gpu-validation - 2026-06-13

Camera: OBSBOT Meet 2 on `/dev/video0`, MJPEG 1920x1080 at 30 FPS for this
capture session.
Staging directory: `/srv/argus-regression-capture`.

Future capture default: MJPEG 1920x1080 at 60 FPS. This mode was verified on
`gpu-validation` on 2026-06-13; 120 FPS requests are reduced to 60 FPS, 4K is reduced to
30 FPS, and YUYV is limited to 30 FPS.

## Completed Clips

- `clips/negative/neg_scene_001_30min_20260613_095718.mp4`
  - Started: 2026-06-13T09:57:18+08:00
  - Duration: 1800s, 54000 frames
  - Coverage: lights on/off, people walk-through, reflections, low light, slight vibration, stable periods

- `clips/positive/pos_static_foreign_10events_wide_20260613_103042.mp4`
  - Started: 2026-06-13T10:30:42+08:00
  - Duration: 1199.966667s, 35999 frames
  - Labels: `annotations/pos_static_foreign_10events_wide_20260613_103042.labels.json`
  - Planned positive windows, seconds from clip start: 120-165, 225-270, 330-375, 435-480, 540-585, 645-690, 750-795, 855-900, 960-1005, 1065-1110
  - Timing note: operator reported small second-level timing error. Treat current labels as coarse windows until visual review.

- `clips/positive/pos_projectile_10events_wide_20260613_105301.mp4`
  - Started: 2026-06-13T10:53:01+08:00
  - Duration: 779.933333s, 23398 frames
  - Labels: `annotations/pos_projectile_10events_wide_20260613_105301.labels.json`
  - Planned positive windows, seconds from clip start: 90-110, 155-175, 220-240, 285-305, 350-370, 415-435, 480-500, 545-565, 610-630, 675-695
  - Timing note: coarse windows; operator timing may be off by a few seconds.

## Discarded Clips

- `discarded/pos_static_foreign_10events_20260613_102815.*`: stopped because time windows were too close / unclear.

## Resume Pointer

See `data/regression/HANDOFF.md` for commit IDs, restore commands, and first
baseline numbers.
