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
import random
from pathlib import Path

import cv2
import numpy as np


def generate_synthetic(
    baseline_dir: Path,
    objects_dir: Path,
    output_dir: Path,
    count: int = 500,
    seed: int = 42,
) -> int:
    """Generate synthetic anomaly images.

    Args:
        baseline_dir: Directory with normal baseline images.
        objects_dir: Directory with FOE object images (white background, pre-cropped).
        output_dir: Output directory for synthetic images + GT masks.
        count: Number of synthetic images to generate.
        seed: Random seed for reproducibility.

    Returns:
        Number of images actually generated.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    baseline_images = sorted(
        list(baseline_dir.glob("*.png")) + list(baseline_dir.glob("*.jpg"))
    )
    object_images = sorted(
        list(objects_dir.glob("*.png")) + list(objects_dir.glob("*.jpg"))
    )

    if not baseline_images:
        print(f"No baseline images found in {baseline_dir}")
        return 0
    if not object_images:
        print(f"No FOE object images found in {objects_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    generated = 0
    for i in range(count):
        # Random baseline and object
        bg_path = random.choice(baseline_images)
        obj_path = random.choice(object_images)

        bg = cv2.imread(str(bg_path))
        obj = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
        if bg is None or obj is None:
            continue

        h, w = bg.shape[:2]

        # Random scale (10-40% of frame width)
        scale = rng.uniform(0.1, 0.4)
        obj_w = int(w * scale)
        obj_h = int(obj.shape[0] * obj_w / max(obj.shape[1], 1))
        if obj_h <= 0 or obj_w <= 0:
            continue
        obj_resized = cv2.resize(obj, (obj_w, obj_h))

        # Random position
        max_x = max(1, w - obj_w)
        max_y = max(1, h - obj_h)
        px = int(rng.integers(0, max_x))
        py = int(rng.integers(0, max_y))

        # Create mask from object (white bg → invert)
        if obj_resized.shape[2] == 4:
            # Has alpha channel
            mask = obj_resized[:, :, 3]
        else:
            # White background detection
            gray = cv2.cvtColor(obj_resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Composite
        result = bg.copy()
        roi = result[py:py+obj_h, px:px+obj_w]
        mask_3ch = cv2.merge([mask, mask, mask])
        mask_inv = cv2.bitwise_not(mask_3ch)
        bg_part = cv2.bitwise_and(roi, mask_inv)
        fg_part = cv2.bitwise_and(obj_resized[:, :, :3], mask_3ch)
        result[py:py+obj_h, px:px+obj_w] = cv2.add(bg_part, fg_part)

        # Brightness matching (match mean of placement area)
        bg_mean = bg[py:py+obj_h, px:px+obj_w].mean()
        obj_mean = fg_part.mean()
        if obj_mean > 0:
            factor = bg_mean / obj_mean
            factor = np.clip(factor, 0.5, 2.0)

        # Ground truth mask (full frame size)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        gt_mask[py:py+obj_h, px:px+obj_w] = mask

        # Save
        cv2.imwrite(str(output_dir / f"synthetic_{i:05d}.png"), result)
        cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), gt_mask)
        generated += 1

    print(f"Generated {generated}/{count} synthetic images in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic anomaly images")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_synthetic(
        args.baseline_dir, args.objects_dir, args.output_dir,
        args.count, args.seed,
    )


if __name__ == "__main__":
    main()
