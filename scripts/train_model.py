"""Train an Anomalib anomaly detection model on baseline images.

Usage:
    python scripts/train_model.py \
        --data data/baselines/cam_01 \
        --output data/models/cam_01 \
        --model patchcore \
        --image-size 256

Supported models:
    patchcore    - Best accuracy, moderate speed (recommended)
    efficient_ad - Fastest inference, good for edge deployment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train Anomalib anomaly detection model")
    parser.add_argument("--data", "-d", required=True, help="Directory of normal baseline images")
    parser.add_argument("--output", "-o", required=True, help="Output directory for trained model")
    parser.add_argument(
        "--model", "-m",
        choices=["patchcore", "efficient_ad"],
        default="patchcore",
        help="Model type (default: patchcore)",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Input image size (default: 256)")
    parser.add_argument("--export", choices=["openvino", "onnx"], help="Export format after training")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    images = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
    if len(images) < 10:
        print(f"Error: Need at least 10 baseline images, found {len(images)}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training {args.model} model")
    print(f"  Data:       {data_dir} ({len(images)} images)")
    print(f"  Output:     {output_dir}")
    print(f"  Image size: {args.image_size}x{args.image_size}")

    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd, Patchcore
    except ImportError:
        print("Error: anomalib is not installed. Run: pip install anomalib", file=sys.stderr)
        sys.exit(1)

    # Create dataset - Anomalib Folder dataset expects a "normal" subfolder
    datamodule = Folder(
        name="baseline",
        root=str(data_dir.parent),
        normal_dir=data_dir.name,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,  # Windows compatibility
    )

    # Select model
    if args.model == "patchcore":
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=0.1,
        )
    else:
        model = EfficientAd()

    # Train
    engine = Engine(
        default_root_dir=str(output_dir),
        max_epochs=1,  # PatchCore only needs 1 epoch (feature extraction)
    )

    print("\nTraining started...")
    engine.fit(model=model, datamodule=datamodule)
    print("Training complete.")

    # Export if requested
    if args.export:
        print(f"\nExporting to {args.export} format...")
        engine.export(
            model=model,
            export_mode=args.export,
        )
        print(f"Model exported to {output_dir}")

    print(f"\nModel saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Copy model to your deployment target")
    print(f"  2. Update configs/default.yaml with the model path")
    print(f"  3. Restart argus: python -m argus --config configs/default.yaml")


if __name__ == "__main__":
    main()
