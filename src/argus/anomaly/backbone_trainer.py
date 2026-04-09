"""DINOv2 SSL continue-pretraining (Level 1 of two-level training architecture).

Fine-tunes a DINOv2-ViT backbone on pooled baseline images from all cameras
to learn domain-specific visual features for the plant environment.

This runs infrequently (monthly/quarterly) and produces a shared backbone
checkpoint that all per-camera anomaly heads (Level 2) build upon.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = structlog.get_logger()


@dataclass
class BackboneTrainingResult:
    """Result of a backbone SSL fine-tuning run."""

    success: bool
    checkpoint_path: str | None = None
    checkpoint_hash: str = ""
    dataset_hash: str = ""
    total_images: int = 0
    epochs_completed: int = 0
    final_loss: float = 0.0
    duration_seconds: float = 0.0
    error: str | None = None


class _BaselineImageDataset(Dataset):
    """Simple dataset that loads and augments baseline images for SSL training."""

    def __init__(self, image_paths: list[Path], image_size: int = 224):
        self._paths = image_paths
        self._size = image_size

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        path = self._paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(
                "backbone_dataset.imread_failed",
                path=str(path),
                idx=idx,
            )
            return None  # filtered by collate_fn

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._size, self._size))

        # Normalize to [0, 1] then to ImageNet stats
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor


class _SSLHead(nn.Module):
    """Simple self-supervised projection head for DINO-style training.

    Projects CLS token to a lower-dimensional space for contrastive loss.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 256, hidden_dim: int = 2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.mlp(x), dim=-1)


def _collect_baseline_images(
    camera_ids: list[str],
    baselines_dir: Path,
    max_per_camera: int = 5000,
    max_total: int = 50000,
) -> list[Path]:
    """Collect baseline images from all cameras for pooled training."""
    all_paths: list[Path] = []
    for cam_id in camera_ids:
        cam_dir = baselines_dir / cam_id
        if not cam_dir.exists():
            continue
        # Search recursively for images
        images = sorted(
            list(cam_dir.rglob("*.png")) + list(cam_dir.rglob("*.jpg"))
        )
        all_paths.extend(images[:max_per_camera])

    # Enforce total image cap to prevent OOM
    if len(all_paths) > max_total:
        logger.warning(
            "backbone_trainer.images_capped",
            collected=len(all_paths),
            max_total=max_total,
        )
        all_paths = all_paths[:max_total]

    logger.info(
        "backbone_trainer.images_collected",
        total=len(all_paths),
        cameras=len(camera_ids),
    )
    return all_paths


def _compute_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _compute_dataset_hash(image_paths: list[Path]) -> str:
    """Compute a deterministic hash of the dataset (paths + mtime + count).

    Includes file modification times so that re-captured baselines at the
    same paths produce a different hash.
    """
    h = hashlib.sha256()
    h.update(str(len(image_paths)).encode())
    for p in sorted(image_paths)[:100]:  # Sample first 100 for speed
        h.update(str(p).encode())
        try:
            h.update(str(p.stat().st_mtime_ns).encode())
        except OSError:
            pass
    return h.hexdigest()[:16]


class BackboneTrainer:
    """Fine-tunes DINOv2 backbone on plant baseline images."""

    def __init__(self, output_dir: Path | str = "data/backbones"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        camera_ids: list[str],
        baselines_dir: Path,
        backbone_type: str = "dinov2_vitb14",
        image_size: int = 224,
        epochs: int = 5,
        lr: float = 1e-5,
        batch_size: int = 16,
        progress_callback: callable | None = None,
    ) -> BackboneTrainingResult:
        """Fine-tune DINOv2 backbone on pooled baseline images.

        Args:
            camera_ids: List of camera IDs to pool baselines from.
            baselines_dir: Root directory containing per-camera baselines.
            backbone_type: DINOv2 variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14).
            image_size: Input image size for training.
            epochs: Number of training epochs.
            lr: Learning rate for AdamW optimizer.
            batch_size: Training batch size.
            progress_callback: Optional (pct, msg) callback.

        Returns:
            BackboneTrainingResult with checkpoint path and metadata.
        """
        start = time.monotonic()

        def _progress(pct: int, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        _progress(5, "收集基线图片...")

        # Collect all baseline images
        image_paths = _collect_baseline_images(camera_ids, baselines_dir)
        if len(image_paths) < 50:
            return BackboneTrainingResult(
                success=False,
                error=f"基线图片不足: {len(image_paths)} (需要 >= 50)",
                total_images=len(image_paths),
                duration_seconds=time.monotonic() - start,
            )

        dataset_hash = _compute_dataset_hash(image_paths)

        _progress(10, f"加载 {backbone_type} 预训练权重...")

        # Load pretrained DINOv2
        try:
            backbone = torch.hub.load(
                "facebookresearch/dinov2", backbone_type, pretrained=True,
            )
        except Exception as e:
            return BackboneTrainingResult(
                success=False,
                error=f"Failed to load {backbone_type}: {e}",
                duration_seconds=time.monotonic() - start,
            )

        backbone.train()

        # Determine feature dimension from backbone
        embed_dim = backbone.embed_dim  # 384 for vits14, 768 for vitb14, 1024 for vitl14
        ssl_head = _SSLHead(in_dim=embed_dim, out_dim=256)

        # Create dataset and dataloader
        dataset = _BaselineImageDataset(image_paths, image_size=image_size)

        def _skip_none_collate(batch: list[torch.Tensor | None]) -> torch.Tensor | None:
            """Filter out None entries (failed reads) and stack the rest."""
            valid = [t for t in batch if t is not None]
            if not valid:
                return None
            return torch.stack(valid)

        # Windows spawn-based multiprocessing can cause issues; use 0 workers there
        num_workers = 0 if sys.platform == "win32" else 2
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=_skip_none_collate,
        )

        # Abort if corruption rate exceeds 10% (sample-based check to avoid
        # reading every image twice — DataLoader will read them during training)
        sample_size = min(200, len(dataset))
        sample_indices = list(range(sample_size))  # first N images
        corrupt_count = sum(1 for i in sample_indices if dataset[i] is None)
        if corrupt_count > 0:
            corrupt_rate = corrupt_count / sample_size
            logger.warning(
                "backbone_trainer.corrupt_images",
                corrupt=corrupt_count,
                sampled=sample_size,
                total=len(dataset),
                rate=f"{corrupt_rate:.1%}",
            )
            if corrupt_rate > 0.10:
                return BackboneTrainingResult(
                    success=False,
                    error=f"图像损坏率过高: {corrupt_count}/{sample_size} 抽样 ({corrupt_rate:.1%} > 10%)",
                    total_images=len(image_paths),
                    duration_seconds=time.monotonic() - start,
                )

        # Optimizer: only fine-tune last few transformer blocks + SSL head
        # Freeze early layers for efficiency
        for name, param in backbone.named_parameters():
            param.requires_grad = False
        # Unfreeze last 2 transformer blocks
        for name, param in backbone.named_parameters():
            if any(f"blocks.{i}." in name for i in range(max(0, backbone.n_blocks - 2), backbone.n_blocks)):
                param.requires_grad = True

        optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, backbone.parameters()))
            + list(ssl_head.parameters()),
            lr=lr,
            weight_decay=0.05,
        )

        _progress(15, f"开始训练 ({epochs} epochs, {len(image_paths)} 张图片)...")

        # Training loop: self-supervised contrastive loss
        # Two augmented views of same image should have similar features
        total_batches = len(dataloader) * epochs
        batch_count = 0
        final_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                if batch is None:
                    continue  # entire batch was corrupt — skip
                # Create two augmented views (simple: horizontal flip)
                view1 = batch
                view2 = torch.flip(batch, dims=[3])  # Horizontal flip

                # Forward through backbone
                feat1 = backbone(view1)  # CLS token features
                feat2 = backbone(view2)

                # Project through SSL head
                z1 = ssl_head(feat1)
                z2 = ssl_head(feat2)

                # Cosine similarity loss (simplified DINO)
                loss = 2 - 2 * (z1 * z2).sum(dim=-1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                batch_count += 1

                if batch_count % 10 == 0 and total_batches > 0:
                    pct = 15 + int(80 * batch_count / total_batches)
                    _progress(pct, f"Epoch {epoch + 1}/{epochs} — loss: {loss.item():.4f}")

            avg_loss = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss
            logger.info(
                "backbone_trainer.epoch",
                epoch=epoch + 1,
                avg_loss=round(avg_loss, 4),
                batches=n_batches,
            )

        # Save checkpoint (backbone weights only, no SSL head)
        _progress(95, "保存骨干 checkpoint...")

        ts = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_name = f"{backbone_type}_{ts}.pth"
        checkpoint_path = self._output_dir / checkpoint_name

        # Save only backbone state dict (encoder weights)
        torch.save(backbone.state_dict(), str(checkpoint_path))
        checkpoint_hash = _compute_hash(checkpoint_path)

        duration = time.monotonic() - start
        _progress(100, f"骨干训练完成 — {duration:.0f}s")

        logger.info(
            "backbone_trainer.complete",
            checkpoint=str(checkpoint_path),
            hash=checkpoint_hash,
            images=len(image_paths),
            epochs=epochs,
            final_loss=round(final_loss, 4),
            duration=f"{duration:.0f}s",
        )

        return BackboneTrainingResult(
            success=True,
            checkpoint_path=str(checkpoint_path),
            checkpoint_hash=checkpoint_hash,
            dataset_hash=dataset_hash,
            total_images=len(image_paths),
            epochs_completed=epochs,
            final_loss=final_loss,
            duration_seconds=duration,
        )
