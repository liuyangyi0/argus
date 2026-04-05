"""Tests for model training pipeline (Phase 2: TRN-001 through TRN-008)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.anomaly.baseline import BaselineManager
from argus.anomaly.trainer import ModelTrainer, QualityReport, TrainingStatus


@pytest.fixture
def bm(tmp_path):
    return BaselineManager(baselines_dir=tmp_path / "baselines")


@pytest.fixture
def trainer(bm, tmp_path):
    return ModelTrainer(
        baseline_manager=bm,
        models_dir=tmp_path / "models",
        exports_dir=tmp_path / "exports",
    )


def _create_baseline_images(bm, camera_id="cam_01", zone_id="default", count=50, varied=True):
    """Helper: create N baseline images with optional brightness variation."""
    version_dir = bm.create_new_version(camera_id, zone_id)
    rng = np.random.default_rng(42)

    for i in range(count):
        if varied:
            # Vary brightness to pass diversity check
            brightness = 50 + int(150 * (i / count))
            frame = np.full((100, 100, 3), brightness, dtype=np.uint8)
            # Add some unique noise per frame
            noise = rng.integers(-20, 20, (100, 100, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        cv2.imwrite(str(version_dir / f"baseline_{i:05d}.png"), frame)

    bm.set_current_version(camera_id, zone_id, version_dir.name)
    return version_dir


# ── TRN-001: Pre-training Validation ──


class TestPreValidation:
    def test_insufficient_count(self, trainer, bm):
        """Should fail when fewer than 30 images."""
        _create_baseline_images(bm, count=15)
        baseline_dir = bm.get_baseline_dir("cam_01")
        result = trainer._validate_baseline_quality(baseline_dir)
        assert not result["passed"]
        assert any("不足" in e for e in result["errors"])

    def test_high_corruption(self, trainer, bm, tmp_path):
        """Should fail when corruption rate >= 10%."""
        version_dir = bm.create_new_version("cam_01", "default")

        # Create 30 images, 5 corrupted (16.7%)
        for i in range(25):
            frame = np.full((100, 100, 3), 50 + i * 5, dtype=np.uint8)
            cv2.imwrite(str(version_dir / f"baseline_{i:05d}.png"), frame)

        # Create 5 corrupted files (not valid images)
        for i in range(25, 30):
            (version_dir / f"baseline_{i:05d}.png").write_text("not an image")

        bm.set_current_version("cam_01", "default", version_dir.name)
        baseline_dir = bm.get_baseline_dir("cam_01")
        result = trainer._validate_baseline_quality(baseline_dir)
        assert not result["passed"]
        assert result["corruption_rate"] >= 0.1
        assert any("损坏率" in e for e in result["errors"])

    def test_high_near_duplicates(self, trainer, bm):
        """Should fail when near-duplicate rate >= 80%."""
        version_dir = bm.create_new_version("cam_01", "default")

        # Create 30 identical images
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        for i in range(30):
            cv2.imwrite(str(version_dir / f"baseline_{i:05d}.png"), frame)

        bm.set_current_version("cam_01", "default", version_dir.name)
        baseline_dir = bm.get_baseline_dir("cam_01")
        result = trainer._validate_baseline_quality(baseline_dir)
        assert not result["passed"]
        assert result["near_duplicate_rate"] >= 0.8
        assert any("重复" in e for e in result["errors"])

    def test_low_brightness_diversity(self, trainer, bm):
        """Should fail when brightness std <= 2.0."""
        version_dir = bm.create_new_version("cam_01", "default")

        # Create 30 images with nearly identical brightness but unique patterns
        rng = np.random.default_rng(42)
        for i in range(30):
            # Brightness centered at 128, std < 2
            brightness = 128
            frame = np.full((100, 100, 3), brightness, dtype=np.uint8)
            # Add distinct patterns to avoid duplicate detection, but keep mean brightness same
            pattern = rng.integers(0, 2, (100, 100, 3), dtype=np.uint8)
            frame = frame + pattern  # tiny variation, mean stays ~128
            cv2.imwrite(str(version_dir / f"baseline_{i:05d}.png"), frame)

        bm.set_current_version("cam_01", "default", version_dir.name)
        baseline_dir = bm.get_baseline_dir("cam_01")
        result = trainer._validate_baseline_quality(baseline_dir)
        assert not result["passed"]
        assert result["brightness_std"] <= 2.0
        assert any("亮度" in e for e in result["errors"])

    def test_pass_good_baselines(self, trainer, bm):
        """Should pass with diverse, high-quality baselines."""
        _create_baseline_images(bm, count=50, varied=True)
        baseline_dir = bm.get_baseline_dir("cam_01")
        result = trainer._validate_baseline_quality(baseline_dir)
        assert result["passed"]
        assert result["corruption_rate"] < 0.1
        assert result["near_duplicate_rate"] < 0.8
        assert result["brightness_std"] > 2.0
        assert len(result["errors"]) == 0


# ── TRN-002: Train/Val Split ──


class TestTrainValSplit:
    def test_split_ratio(self, trainer, bm, tmp_path):
        """Should split 80/20."""
        _create_baseline_images(bm, count=50)
        baseline_dir = bm.get_baseline_dir("cam_01")
        output_dir = tmp_path / "models" / "cam_01" / "default"
        output_dir.mkdir(parents=True)

        train_dir, val_dir, train_count, val_count = trainer._split_train_val(
            baseline_dir, output_dir
        )
        assert train_count == 40  # 80% of 50
        assert val_count == 10  # 20% of 50

        # Verify files exist
        train_images = list((train_dir / "normal").glob("*.png"))
        val_images = list((val_dir / "normal").glob("*.png"))
        assert len(train_images) == 40
        assert len(val_images) == 10

    def test_split_deterministic(self, trainer, bm, tmp_path):
        """Same seed should produce identical splits."""
        _create_baseline_images(bm, count=50)
        baseline_dir = bm.get_baseline_dir("cam_01")

        out1 = tmp_path / "split1"
        out1.mkdir()
        _, _, _, _ = trainer._split_train_val(baseline_dir, out1, seed=42)
        train1 = sorted(f.name for f in (out1 / "_split" / "train" / "normal").glob("*.png"))

        out2 = tmp_path / "split2"
        out2.mkdir()
        _, _, _, _ = trainer._split_train_val(baseline_dir, out2, seed=42)
        train2 = sorted(f.name for f in (out2 / "_split" / "train" / "normal").glob("*.png"))

        assert train1 == train2

    def test_split_different_seeds(self, trainer, bm, tmp_path):
        """Different seeds should produce different splits."""
        _create_baseline_images(bm, count=50)
        baseline_dir = bm.get_baseline_dir("cam_01")

        out1 = tmp_path / "split1"
        out1.mkdir()
        _, _, _, _ = trainer._split_train_val(baseline_dir, out1, seed=42)
        train1 = sorted(f.name for f in (out1 / "_split" / "train" / "normal").glob("*.png"))

        out2 = tmp_path / "split2"
        out2.mkdir()
        _, _, _, _ = trainer._split_train_val(baseline_dir, out2, seed=99)
        train2 = sorted(f.name for f in (out2 / "_split" / "train" / "normal").glob("*.png"))

        assert train1 != train2


# ── TRN-005: Threshold Recommendation ──


class TestThresholdRecommendation:
    def test_basic_calculation(self):
        """Threshold should be max(mean + 2.5*std, max * 1.05)."""
        stats = {"scores": [0.1, 0.2, 0.15], "mean": 0.15, "std": 0.05, "max": 0.2}
        threshold = ModelTrainer._recommend_threshold(stats)
        expected = max(0.15 + 2.5 * 0.05, 0.2 * 1.05)  # max(0.275, 0.21) = 0.275
        assert abs(threshold - round(expected, 4)) < 0.001

    def test_cap_at_095(self):
        """Threshold should not exceed 0.95."""
        stats = {"scores": [0.8, 0.9, 0.95], "mean": 0.88, "std": 0.08, "max": 0.95}
        threshold = ModelTrainer._recommend_threshold(stats)
        assert threshold <= 0.95

    def test_minimum_threshold(self):
        """Threshold should not go below 0.1."""
        stats = {"scores": [0.01, 0.02], "mean": 0.015, "std": 0.005, "max": 0.02}
        threshold = ModelTrainer._recommend_threshold(stats)
        assert threshold >= 0.1

    def test_empty_scores_default(self):
        """Should return default 0.7 when no validation data."""
        stats = {"scores": [], "mean": 0.0, "std": 0.0, "max": 0.0}
        threshold = ModelTrainer._recommend_threshold(stats)
        assert threshold == 0.7


# ── TRN-004: Quality Report ──


class TestQualityReport:
    def test_grade_a(self):
        """Should grade A for excellent model."""
        stats = {"scores": [0.1], "mean": 0.1, "std": 0.05, "max": 0.2, "p95": 0.18}
        report = ModelTrainer._generate_quality_report(stats, threshold_recommended=0.5)
        assert report.grade == "A"
        assert any("优秀" in s for s in report.suggestions)

    def test_grade_b(self):
        """Should grade B for good model."""
        stats = {"scores": [0.3], "mean": 0.35, "std": 0.1, "max": 0.45, "p95": 0.42}
        report = ModelTrainer._generate_quality_report(stats, threshold_recommended=0.6)
        assert report.grade == "B"

    def test_grade_c(self):
        """Should grade C for acceptable model."""
        stats = {"scores": [0.4], "mean": 0.45, "std": 0.2, "max": 0.55, "p95": 0.52}
        report = ModelTrainer._generate_quality_report(stats, threshold_recommended=0.7)
        assert report.grade == "C"
        assert any("重新采集" in s for s in report.suggestions)

    def test_grade_f(self):
        """Should grade F for poor model."""
        stats = {"scores": [0.8], "mean": 0.6, "std": 0.2, "max": 0.85, "p95": 0.82}
        report = ModelTrainer._generate_quality_report(stats, threshold_recommended=0.7)
        assert report.grade == "F"

    def test_empty_scores(self):
        """Should grade F with suggestion for empty validation set."""
        stats = {"scores": [], "mean": 0.0, "std": 0.0, "max": 0.0}
        report = ModelTrainer._generate_quality_report(stats, threshold_recommended=0.7)
        assert report.grade == "F"
        assert any("验证集为空" in s for s in report.suggestions)


# ── TRN-007: Training Record ──


class TestTrainingRecord:
    def test_save_and_query(self, tmp_path):
        """Should save and retrieve training records."""
        from argus.storage.database import Database
        from argus.storage.models import TrainingRecord

        db = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
        db.initialize()

        record = TrainingRecord(
            camera_id="cam_01",
            zone_id="default",
            model_type="patchcore",
            export_format="openvino",
            baseline_version="v001",
            baseline_count=100,
            train_count=80,
            val_count=20,
            pre_validation_passed=True,
            corruption_rate=0.02,
            near_duplicate_rate=0.15,
            brightness_std=25.0,
            val_score_mean=0.12,
            val_score_std=0.05,
            val_score_max=0.28,
            val_score_p95=0.22,
            quality_grade="A",
            threshold_recommended=0.35,
            model_path="/models/cam_01/default",
            checkpoint_valid=True,
            export_valid=True,
            smoke_test_passed=True,
            inference_latency_ms=45.2,
            status="complete",
            duration_seconds=120.5,
            trained_at=datetime.utcnow(),
        )

        saved = db.save_training_record(record)
        assert saved.id is not None

        # Query by camera
        history = db.get_training_history(camera_id="cam_01")
        assert len(history) == 1
        assert history[0].quality_grade == "A"

        # Get latest
        latest = db.get_latest_training("cam_01", "default")
        assert latest is not None
        assert latest.model_type == "patchcore"

        # Get by ID
        fetched = db.get_training_record(saved.id)
        assert fetched is not None
        assert fetched.threshold_recommended == 0.35

        db.close()

    def test_to_dict(self, tmp_path):
        """TrainingRecord.to_dict should return all fields."""
        from argus.storage.database import Database
        from argus.storage.models import TrainingRecord

        db = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
        db.initialize()

        record = TrainingRecord(
            camera_id="cam_01",
            zone_id="default",
            model_type="patchcore",
            baseline_version="v001",
            baseline_count=50,
            train_count=40,
            val_count=10,
            pre_validation_passed=True,
            status="complete",
            duration_seconds=60.0,
            trained_at=datetime.utcnow(),
        )
        saved = db.save_training_record(record)
        d = saved.to_dict()

        assert d["camera_id"] == "cam_01"
        assert d["status"] == "complete"
        assert "trained_at" in d
        assert "created_at" in d

        db.close()

    def test_history_ordering(self, tmp_path):
        """Training history should be ordered by trained_at descending."""
        from argus.storage.database import Database
        from argus.storage.models import TrainingRecord

        db = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
        db.initialize()

        for i in range(3):
            record = TrainingRecord(
                camera_id="cam_01",
                zone_id="default",
                model_type="patchcore",
                baseline_version=f"v{i+1:03d}",
                baseline_count=50,
                train_count=40,
                val_count=10,
                pre_validation_passed=True,
                quality_grade=["C", "B", "A"][i],
                status="complete",
                duration_seconds=60.0,
                trained_at=datetime(2026, 1, i + 1),
            )
            db.save_training_record(record)

        history = db.get_training_history(camera_id="cam_01")
        assert len(history) == 3
        # Most recent first
        assert history[0].quality_grade == "A"
        assert history[2].quality_grade == "C"

        db.close()


# ── TRN-006: Output Validation ──


class TestOutputValidation:
    def test_missing_checkpoint(self, trainer, tmp_path):
        """Should report invalid when no checkpoint exists."""
        output_dir = tmp_path / "models" / "cam_01"
        output_dir.mkdir(parents=True)
        val_dir = tmp_path / "val"
        val_dir.mkdir()

        result, _ = trainer._validate_output(output_dir, None, val_dir)
        assert not result["checkpoint_valid"]
        assert any("未找到" in e for e in result["errors"])

    def test_empty_checkpoint(self, trainer, tmp_path):
        """Should report invalid for zero-size checkpoint."""
        output_dir = tmp_path / "models" / "cam_01"
        output_dir.mkdir(parents=True)
        (output_dir / "model.ckpt").write_text("")  # empty file
        val_dir = tmp_path / "val"
        val_dir.mkdir()

        result, _ = trainer._validate_output(output_dir, None, val_dir)
        assert not result["checkpoint_valid"]
        assert any("大小为 0" in e for e in result["errors"])

    def test_valid_checkpoint_no_export(self, trainer, tmp_path):
        """Should report valid checkpoint with no export."""
        output_dir = tmp_path / "models" / "cam_01"
        output_dir.mkdir(parents=True)
        (output_dir / "model.ckpt").write_bytes(b"fake model data" * 100)
        val_dir = tmp_path / "val"
        val_dir.mkdir()

        result, _ = trainer._validate_output(output_dir, None, val_dir)
        assert result["checkpoint_valid"]


# ── Integration: train() with insufficient baselines ──


class TestTrainIntegration:
    def test_train_insufficient_baselines(self, trainer, bm):
        """Should fail early with insufficient baselines."""
        _create_baseline_images(bm, count=10)
        result = trainer.train("cam_01", model_type="patchcore")
        assert result.status == TrainingStatus.FAILED
        assert "不足" in result.error

    def test_train_pre_validation_fails(self, trainer, bm):
        """Should fail if pre-validation detects quality issues."""
        # Create 30 identical images (high duplicate rate)
        version_dir = bm.create_new_version("cam_01", "default")
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        for i in range(30):
            cv2.imwrite(str(version_dir / f"baseline_{i:05d}.png"), frame)
        bm.set_current_version("cam_01", "default", version_dir.name)

        result = trainer.train("cam_01", model_type="patchcore")
        assert result.status == TrainingStatus.FAILED
        assert result.pre_validation is not None
        assert not result.pre_validation["passed"]


# ── Utility: _find_best_model_file ──


class TestFindBestModelFile:
    def test_prefers_xml(self, tmp_path):
        """Should prefer .xml over .ckpt."""
        (tmp_path / "model.ckpt").write_bytes(b"ckpt")
        (tmp_path / "model.xml").write_bytes(b"xml")
        result = ModelTrainer._find_best_model_file(tmp_path)
        assert result.suffix == ".xml"

    def test_finds_nested(self, tmp_path):
        """Should find model files in subdirectories."""
        nested = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
        nested.mkdir(parents=True)
        (nested / "model.ckpt").write_bytes(b"ckpt")
        result = ModelTrainer._find_best_model_file(tmp_path)
        assert result is not None
        assert result.name == "model.ckpt"

    def test_returns_none_empty_dir(self, tmp_path):
        """Should return None when no model files exist."""
        result = ModelTrainer._find_best_model_file(tmp_path)
        assert result is None
