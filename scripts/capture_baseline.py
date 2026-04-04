"""Capture baseline (normal) images from a camera for Anomalib training.

Usage:
    python scripts/capture_baseline.py --source rtsp://192.168.1.100:554/stream1 \
        --output data/baselines/cam_01 --count 100 --interval 2.0

This captures N frames at regular intervals and saves them as the "normal"
reference images. These are then used to train PatchCore/EfficientAD models.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Capture baseline images from a camera")
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Camera source: RTSP URL, USB device index (e.g., '0'), or video file path",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for baseline images",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="Number of frames to capture (default: 100)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between captures (default: 2.0)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="Resize images to WxH (e.g., --resize 256 256)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine source type
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        return

    print(f"Capturing {args.count} baseline frames from: {args.source}")
    print(f"Output directory: {output_dir}")
    print(f"Interval: {args.interval}s")
    if args.resize:
        print(f"Resize to: {args.resize[0]}x{args.resize[1]}")

    captured = 0
    try:
        while captured < args.count:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame, retrying...")
                time.sleep(0.5)
                continue

            if args.resize:
                frame = cv2.resize(frame, tuple(args.resize))

            filename = output_dir / f"baseline_{captured:05d}.png"
            cv2.imwrite(str(filename), frame)
            captured += 1

            print(f"\r  Captured {captured}/{args.count}", end="", flush=True)

            if captured < args.count:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Captured {captured} frames.")
    finally:
        cap.release()

    print(f"\nDone. {captured} baseline images saved to {output_dir}")
    print(f"\nNext step: Train the anomaly model with:")
    print(f"  python scripts/train_model.py --data {output_dir}")


if __name__ == "__main__":
    main()
