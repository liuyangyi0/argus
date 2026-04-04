"""Export a trained Anomalib model to OpenVINO or ONNX format.

Usage:
    python scripts/export_model.py \
        --checkpoint data/models/cam_01/weights.ckpt \
        --output data/exports/cam_01 \
        --format openvino
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export Anomalib model for deployment")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--format", "-f",
        choices=["openvino", "onnx"],
        default="openvino",
        help="Export format (default: openvino)",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Input image size (default: 256)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    output_dir = Path(args.output)

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from anomalib.deploy import ExportMode, export
    except ImportError:
        print("Error: anomalib is not installed. Run: pip install anomalib", file=sys.stderr)
        sys.exit(1)

    export_mode = ExportMode.OPENVINO if args.format == "openvino" else ExportMode.ONNX

    print(f"Exporting model")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Format:     {args.format}")
    print(f"  Output:     {output_dir}")
    print(f"  Image size: {args.image_size}x{args.image_size}")

    export(
        model_path=str(ckpt_path),
        export_mode=export_mode,
        export_root=str(output_dir),
        input_size=(args.image_size, args.image_size),
    )

    print(f"\nExport complete: {output_dir}")


if __name__ == "__main__":
    main()
