"""Generate synthetic anomaly images by compositing FOE objects onto normal baselines.

Usage:
    python scripts/generate_synthetic.py \
        --baseline-dir data/baselines/cam_01/default/v003 \
        --objects-dir data/foe_objects/ \
        --output-dir data/synthetic/ \
        --count 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.validation.synthetic import generate_synthetic


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic anomaly images by compositing FOE objects onto baselines"
    )
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Directory with normal baseline images")
    parser.add_argument("--objects-dir", type=Path, required=True,
                        help="Directory with FOE object images (PNG with alpha or white bg)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for synthetic images + masks")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of synthetic images to generate (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--min-scale", type=float, default=0.3,
                        help="Minimum scale factor for FOE objects (default: 0.3)")
    parser.add_argument("--max-scale", type=float, default=2.0,
                        help="Maximum scale factor for FOE objects (default: 2.0)")
    parser.add_argument("--blur-sigma", type=float, default=2.0,
                        help="Gaussian blur sigma for edge blending (default: 2.0)")
    args = parser.parse_args()

    generate_synthetic(
        baseline_dir=args.baseline_dir,
        objects_dir=args.objects_dir,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        blur_sigma=args.blur_sigma,
        progress=True,
    )


if __name__ == "__main__":
    main()
