"""Create a deterministic local video source for Argus development.

The generated video starts with a stable baseline scene, then introduces a
high-contrast object that settles in place. Use it when a workstation has no
USB/RTSP camera:

    python scripts/create_dev_video.py --output data/dev/demo_camera.avi

Then set a camera to:

    protocol: file
    source: data/dev/demo_camera.avi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from argus.runtime.dev_video import create_dev_video
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from argus.runtime.dev_video import create_dev_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local Argus development video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dev/demo_camera.avi"),
        help="Output video path (default: data/dev/demo_camera.avi)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--anomaly-start-s", type=float, default=6.0)
    parser.add_argument(
        "--motion",
        choices=("settle", "moving"),
        default="settle",
        help="Anomaly motion pattern: settle triggers stable foreign-object alerts; moving is for tracking demos.",
    )
    args = parser.parse_args()

    meta = create_dev_video(
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
        anomaly_start_s=args.anomaly_start_s,
        motion=args.motion,
    )
    print(
        "Created {output} ({width}x{height}, {fps} fps, {frames} frames, "
        "anomaly starts at frame {anomaly_start_frame}, motion={motion})".format(**meta)
    )


if __name__ == "__main__":
    main()
