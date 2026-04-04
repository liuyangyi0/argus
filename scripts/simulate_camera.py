"""Simulate a camera feed by replaying a video file through the pipeline.

Useful for testing and development without a real camera.

Usage:
    python scripts/simulate_camera.py --video data/demo.mp4 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys

import cv2
import structlog

from argus.config.loader import load_config
from argus.config.schema import AlertSeverity
from argus.core.pipeline import DetectionPipeline

_SEVERITY_COLORS = {
    AlertSeverity.INFO: "\033[36m",
    AlertSeverity.LOW: "\033[33m",
    AlertSeverity.MEDIUM: "\033[91m",
    AlertSeverity.HIGH: "\033[31;1m",
}
_RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Simulate camera feed from video file")
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Config file")
    parser.add_argument("--display", "-d", action="store_true", help="Show video window")
    parser.add_argument("--loop", action="store_true", help="Loop video indefinitely")
    args = parser.parse_args()

    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer(colors=True)],
        wrapper_class=structlog.make_filtering_bound_logger(structlog.logging.INFO),
    )

    config = load_config(args.config)

    # Override first camera source with the video file
    if not config.cameras:
        print("Error: No cameras configured", file=sys.stderr)
        sys.exit(1)

    cam_config = config.cameras[0].model_copy(
        update={"source": args.video, "protocol": "file"}
    )

    def on_alert(alert):
        color = _SEVERITY_COLORS.get(alert.severity, "")
        print(
            f"{color}[ALERT] {alert.alert_id} | "
            f"{alert.severity.value.upper():6s} | "
            f"Score: {alert.anomaly_score:.3f}"
            f"{_RESET}"
        )

    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=config.alerts,
        on_alert=on_alert,
    )

    if not pipeline.initialize():
        print("Error: Failed to initialize pipeline", file=sys.stderr)
        sys.exit(1)

    print(f"Simulating camera feed from: {args.video}")
    print("Press 'q' to quit" if args.display else "Press Ctrl+C to stop")

    try:
        while True:
            alert = pipeline.run_once()

            if args.display:
                # Show the latest frame in a window
                cap = pipeline._camera._cap
                if cap is not None:
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow("Argus Simulation", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

            # Check if video ended
            if not pipeline._camera.state.connected:
                if args.loop:
                    pipeline._camera.connect()
                else:
                    print("Video ended.")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.shutdown()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
