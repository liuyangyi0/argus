"""Synthetic anomaly image generation by compositing FOE objects onto normal baselines.

Core logic for generating test data used in recall evaluation.
FOE = Foreign Object/Event.
"""

from __future__ import annotations

import random
from pathlib import Path

import structlog
import cv2
import numpy as np

logger = structlog.get_logger()


def _random_perspective_transform(
    obj: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a random perspective warp to an FOE object image.

    Randomly shifts each corner by up to 10% of the image dimensions
    to simulate viewing angle variation.
    """
    h, w = obj.shape[:2]
    margin_x = max(1, int(w * 0.10))
    margin_y = max(1, int(h * 0.10))

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [rng.integers(0, margin_x), rng.integers(0, margin_y)],
        [w - rng.integers(0, margin_x), rng.integers(0, margin_y)],
        [w - rng.integers(0, margin_x), h - rng.integers(0, margin_y)],
        [rng.integers(0, margin_x), h - rng.integers(0, margin_y)],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    flags = cv2.INTER_LINEAR
    border_value = (0, 0, 0, 0) if obj.shape[2] == 4 else (0, 0, 0)
    result = cv2.warpPerspective(
        obj, M, (w, h), flags=flags,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )
    return result


def _match_brightness(
    fg_region: np.ndarray,
    bg_region: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Adjust foreground brightness/contrast to match the background region.

    Only considers pixels where mask > 0 for the mean calculation.
    Returns the adjusted foreground region.
    """
    if mask.sum() == 0:
        return fg_region

    mask_bool = mask > 0

    # Compute mean brightness in LAB space for perceptual accuracy
    bg_lab = cv2.cvtColor(bg_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg_lab = cv2.cvtColor(fg_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_l = bg_lab[:, :, 0][mask_bool]
    fg_l = fg_lab[:, :, 0][mask_bool]

    if len(fg_l) == 0 or fg_l.mean() == 0:
        return fg_region

    bg_mean = float(bg_l.mean())
    fg_mean = float(fg_l.mean())

    if fg_mean > 0:
        factor = bg_mean / fg_mean
        factor = np.clip(factor, 0.5, 2.0)
        fg_lab[:, :, 0] = np.clip(fg_lab[:, :, 0] * factor, 0, 255)

    result = cv2.cvtColor(fg_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def _blur_mask_edges(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian blur to mask edges for seamless blending.

    Returns a float mask in [0, 1] range with soft edges.
    """
    if sigma <= 0:
        return mask.astype(np.float32) / 255.0

    ksize = int(sigma * 4) | 1  # ensure odd kernel
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigma)
    max_val = blurred.max()
    if max_val > 0:
        blurred = blurred / max_val
    return blurred


def _extract_mask(obj: np.ndarray) -> np.ndarray:
    """Extract a binary foreground mask from an FOE object image.

    Handles both alpha-channel PNGs and white-background images.
    """
    if obj.shape[2] == 4:
        return obj[:, :, 3]
    else:
        gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        return mask


def generate_synthetic(
    baseline_dir: Path,
    objects_dir: Path,
    output_dir: Path,
    count: int = 500,
    seed: int = 42,
    min_scale: float = 0.3,
    max_scale: float = 2.0,
    blur_sigma: float = 2.0,
    edge_margin: float = 0.05,
    progress: bool = False,
) -> int:
    """Generate synthetic anomaly images.

    Args:
        baseline_dir: Directory with normal baseline images.
        objects_dir: Directory with FOE object images (white background or alpha).
        output_dir: Output directory for synthetic images + GT masks.
        count: Number of synthetic images to generate.
        seed: Random seed for reproducibility.
        min_scale: Minimum scale factor for FOE objects relative to original size.
        max_scale: Maximum scale factor for FOE objects relative to original size.
        blur_sigma: Gaussian sigma for edge blending (0 = no blur).
        edge_margin: Fraction of frame to avoid at edges (0.05 = 5%).
        progress: Show tqdm progress bar if available.

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

    # Progress bar (optional tqdm)
    iterator = range(count)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Generating synthetic images", unit="img")
        except ImportError:
            logger.debug("synthetic.tqdm_not_available", exc_info=True)

    generated = 0
    for i in iterator:
        # Random baseline and object
        bg_path = random.choice(baseline_images)
        obj_path = random.choice(object_images)

        bg = cv2.imread(str(bg_path))
        obj = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
        if bg is None or obj is None:
            continue

        h, w = bg.shape[:2]

        # Random scale factor (relative to object's original size)
        scale = rng.uniform(min_scale, max_scale)
        obj_w = max(1, int(obj.shape[1] * scale))
        obj_h = max(1, int(obj.shape[0] * scale))

        # Clamp to fit within frame
        obj_w = min(obj_w, int(w * 0.8))
        obj_h = min(obj_h, int(h * 0.8))
        if obj_h <= 2 or obj_w <= 2:
            continue

        obj_resized = cv2.resize(obj, (obj_w, obj_h))

        # Apply random perspective transform
        obj_resized = _random_perspective_transform(obj_resized, rng)

        # Extract mask from object
        mask = _extract_mask(obj_resized)

        # Random position avoiding edges
        margin_x = int(w * edge_margin)
        margin_y = int(h * edge_margin)
        max_x = max(1, w - obj_w - margin_x)
        max_y = max(1, h - obj_h - margin_y)
        px = int(rng.integers(margin_x, max(margin_x + 1, max_x)))
        py = int(rng.integers(margin_y, max(margin_y + 1, max_y)))

        # Ensure we don't exceed frame bounds
        if py + obj_h > h or px + obj_w > w:
            continue

        # Match brightness of FOE object to target background region
        bg_region = bg[py:py + obj_h, px:px + obj_w]
        fg_rgb = obj_resized[:, :, :3]
        fg_adjusted = _match_brightness(fg_rgb, bg_region, mask)

        # Soft mask for seamless blending
        soft_mask = _blur_mask_edges(mask, blur_sigma)
        soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)

        # Alpha blend composite
        result = bg.copy()
        roi = result[py:py + obj_h, px:px + obj_w].astype(np.float32)
        fg_float = fg_adjusted.astype(np.float32)
        blended = roi * (1.0 - soft_mask_3ch) + fg_float * soft_mask_3ch
        result[py:py + obj_h, px:px + obj_w] = np.clip(blended, 0, 255).astype(np.uint8)

        # Ground truth binary mask (full frame size, ensure 2D)
        mask_2d = mask.squeeze() if mask.ndim > 2 else mask
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        gt_mask[py:py + obj_h, px:px + obj_w] = mask_2d

        # Save
        cv2.imwrite(str(output_dir / f"synthetic_{i:05d}.png"), result)
        cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), gt_mask)
        generated += 1

    print(f"Generated {generated}/{count} synthetic images in {output_dir}")
    return generated
