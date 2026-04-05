# Argus v0.2 检测算法升级 — 任务分解

> 每个任务独立可执行，包含完整的上下文、输入边界、输出验收标准。
> 使用 Dinomaly2 作为检测骨干。

---

## 总览

```
Phase A: 基础强化（零新模型依赖）
  A1  CUSUM 时序累积 .............. 4 个子任务
  A2  Conformal Prediction 校准 ... 5 个子任务
  A3  Simplex 双通道 .............. 4 个子任务
  A4  主动学习基线 ................ 3 个子任务

Phase B: Dinomaly2 骨干升级
  B1  Dinomaly2 配置与训练集成 .... 6 个子任务
  B2  INT8 量化导出 ............... 4 个子任务

Phase C: 运维智能
  C1  摄像头健康层 ................ 6 个子任务
  C2  KS 漂移监控 ................ 4 个子任务
  C3  跨摄像头关联 ................ 5 个子任务
  C4  最小 MLOps .................. 4 个子任务

Phase D: 高级检测
  D1  开放词汇检测 OVD ........... 4 个子任务
  D2  SAM 2 实例分割 ............. 4 个子任务
  D3  合成异常数据 ................ 4 个子任务
```

---

## Phase A: 基础强化

---

### A1: CUSUM 时序证据累积 ✅ 已修改完

#### A1-1: 修改 _AnomalyTracker 数据结构 ✅ 已修改完

**目标**: 将硬计数器 `consecutive_count` 替换为浮点证据 `evidence`。

**当前状态**:
- 文件: `src/argus/alerts/grader.py:40-48`
- `_AnomalyTracker` 有 `consecutive_count: int = 0`
- `evaluate()` 在 L121 做 `tracker.consecutive_count += 1`
- L150 判断 `tracker.consecutive_count < self._config.temporal.min_consecutive_frames`

**修改过程**:

1. 打开 `src/argus/alerts/grader.py`
2. 修改 `_AnomalyTracker` dataclass (L40-48):
   ```python
   @dataclass
   class _AnomalyTracker:
       """Tracks anomaly evidence for a specific zone using exponential accumulation."""
       evidence: float = 0.0           # 替换 consecutive_count
       first_seen: float = 0.0
       last_seen: float = 0.0
       max_score: float = 0.0
       prev_anomaly_mask: np.ndarray | None = None
   ```

**边界**:
- 只改 dataclass 定义，不改 evaluate() 逻辑（A1-2 做）
- 不删除 `consecutive_count`——先加 `evidence` 字段，A1-2 中再移除旧字段
- 不改 config（A1-3 做）

**验收**: `_AnomalyTracker` 同时有 `evidence` 和 `consecutive_count`（过渡态），代码无语法错误。

---

#### A1-2: 重写 evaluate() 时序逻辑 ✅ 已修改完

**目标**: 用指数衰减证据替换硬计数。

**当前状态**:
- `evaluate()` 在 L115-157 做时序确认
- L119: gap > max_gap_seconds → reset count to 1
- L125: count += 1
- L150: count < min_consecutive_frames → return None

**修改过程**:

1. 在 `evaluate()` 中替换 L115-157 的时序逻辑块:

```python
# Step 3: Temporal evidence accumulation
tracker = self._trackers[zone_key]
gap = now - tracker.last_seen if tracker.last_seen > 0 else 0

# Gap timeout: reset evidence entirely
if gap > self._config.temporal.max_gap_seconds:
    tracker.evidence = 0.0
    tracker.first_seen = now
    tracker.max_score = adjusted_score

tracker.last_seen = now

# Spatial continuity check (保持不变, L130-148)
# ... 如果 IoU < min_overlap:
#     tracker.evidence = 0.0  # 替换原来的 tracker.consecutive_count = 1
#     tracker.max_score = adjusted_score

# Exponential evidence accumulation
lam = self._config.temporal.evidence_lambda
if severity is not None:  # score >= info threshold
    tracker.evidence = lam * tracker.evidence + adjusted_score
else:
    tracker.evidence *= lam  # pure decay for sub-threshold frames

tracker.max_score = max(tracker.max_score, adjusted_score)

# Check if accumulated evidence exceeds threshold
if tracker.evidence < self._config.temporal.evidence_threshold:
    logger.debug(
        "grader.accumulating_evidence",
        zone=zone_key,
        evidence=round(tracker.evidence, 3),
        threshold=self._config.temporal.evidence_threshold,
    )
    return None
```

2. 注意：Step 2 (severity mapping, L103-113) 中当 severity is None 时，原来直接 reset tracker 并 return None。现在改为：让 evidence 做纯衰减（不 return），但如果 evidence 也低于阈值，后面自然 return None。

**关键改动点**:
- L107: `self._trackers[zone_key] = _AnomalyTracker()` → 改为仅衰减：
  ```python
  if severity is None:
      tracker = self._trackers[zone_key]
      tracker.evidence *= self._config.temporal.evidence_lambda
      if tracker.evidence < 0.01:  # 完全衰减后清理
          self._trackers[zone_key] = _AnomalyTracker()
      return None
  ```
- L146: `tracker.consecutive_count = 1` → `tracker.evidence = 0.0`
- 删除所有 `consecutive_count` 引用

**边界**:
- 不改 Step 1 (zone multiplier)、Step 4 (suppression)、Step 5 (emit alert)
- 空间 IoU 检查逻辑保持不变，只是重置 evidence 而不是 count
- `max_score` 逻辑不变

**验收**:
- 弱持续信号（连续 10 帧 score=0.55）能累积到触发
- 瞬态高分（1 帧 score=0.95 后归零）不触发
- gap 超时后 evidence 归零
- 空间 IoU 不匹配时 evidence 归零

---

#### A1-3: 添加 CUSUM 配置参数 ✅ 已修改完

**目标**: 在 `TemporalConfirmation` 中添加 `evidence_lambda` 和 `evidence_threshold`。

**当前状态**:
- 文件: `src/argus/config/schema.py:146-154`
- `TemporalConfirmation` 有 `min_consecutive_frames`, `max_gap_seconds`, `min_spatial_overlap`

**修改过程**:

1. 修改 `TemporalConfirmation` (schema.py:146-154):
   ```python
   class TemporalConfirmation(BaseModel):
       """Require anomaly persistence before alerting."""
       # Legacy hard threshold (保留用于回退)
       min_consecutive_frames: int = Field(default=3, ge=1, le=30)
       max_gap_seconds: float = Field(default=10.0, ge=1.0, le=120.0)
       min_spatial_overlap: float = Field(
           default=0.3, ge=0.0, le=1.0,
           description="Min IoU between consecutive anomaly heatmaps (0=disabled)",
       )
       # CUSUM evidence accumulation
       evidence_lambda: float = Field(
           default=0.95, ge=0.80, le=0.99,
           description="Exponential decay factor for evidence accumulation (0.8-0.99)",
       )
       evidence_threshold: float = Field(
           default=3.0, ge=0.5, le=20.0,
           description="Accumulated evidence threshold to trigger alert (0.5-20.0)",
       )
   ```

2. 更新 `configs/default.yaml` 中 `alerts.temporal` 部分加入新参数。

**边界**:
- 不改其他 config 类
- `min_consecutive_frames` 保留但不再被 evaluate() 使用（A1-2 已替换为 evidence）

**验收**: `ArgusConfig` 能正确解析含 `evidence_lambda` 和 `evidence_threshold` 的 YAML。

---

#### A1-4: 补充 CUSUM 单元测试 ✅ 已修改完

**目标**: 验证 CUSUM 行为的所有关键场景。

**当前状态**: `tests/unit/test_alert_grader.py` 有 12+ 现有测试。

**新增测试**:

```python
class TestCUSUMEvidence:
    """CUSUM 证据累积测试。"""

    def test_weak_persistent_signal_triggers(self):
        """连续 15 帧 score=0.55 → evidence 累积到阈值 → 触发告警。"""
        # INFO 阈值 = 0.50，所以 0.55 每帧贡献 0.55
        # E = 0 + 0.55, then 0.95*0.55 + 0.55, ...
        # 约 7-8 帧后 evidence 应超过 3.0

    def test_transient_spike_decays(self):
        """1 帧 score=0.95，后续 score=0.0 → evidence 快速衰减 → 不触发。"""

    def test_gap_timeout_resets_evidence(self):
        """连续 5 帧有信号，gap > max_gap_seconds，然后 1 帧 → evidence 重新从 0 开始。"""

    def test_spatial_mismatch_resets_evidence(self):
        """IoU < min_spatial_overlap → evidence 归零。"""

    def test_steady_medium_signal(self):
        """连续帧 score=0.72 → evidence 稳步累积 → 约 5 帧触发。"""

    def test_below_info_threshold_decays(self):
        """score < 0.50 的帧 → evidence 纯衰减。"""

    def test_mixed_signal_accumulates(self):
        """交替 score=0.6 和 score=0.0 → evidence 缓慢累积但可能不触发。"""
```

**边界**:
- 只加新测试，不改现有测试（它们可能需要适配 evidence 逻辑，如果有直接依赖 consecutive_count 的断言需要更新）
- 检查现有测试 `test_temporal_confirmation_required` 是否需要适配

**验收**: `pytest tests/unit/test_alert_grader.py -v` 全部通过。

---

### A2: Conformal Prediction 分数校准 ✅ 已修改完

#### A2-1: 创建 ConformalCalibrator 核心类 ✅ 已修改完

**目标**: 实现 conformal prediction 的分位数校准逻辑。

**新建文件**: `src/argus/alerts/calibration.py`

**实现过程**:

```python
"""Conformal prediction score calibration for statistically guaranteed FPR.

Given a set of anomaly scores from known-normal frames, computes thresholds
that guarantee a target false positive rate using distribution-free quantiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CalibrationResult:
    """Calibrated thresholds with metadata."""
    info_threshold: float
    low_threshold: float
    medium_threshold: float
    high_threshold: float
    n_calibration_samples: int
    target_fprs: dict[str, float]


class ConformalCalibrator:
    """Distribution-free threshold calibration using conformal prediction.

    Given normal-frame scores, find the score threshold that guarantees
    P(score > threshold | normal) <= target_fpr.

    The key property: for n calibration samples, the (1-alpha) quantile
    provides a coverage guarantee of at least (1-alpha) with finite-sample
    validity, regardless of the score distribution.
    """

    def calibrate(
        self,
        normal_scores: np.ndarray,
        target_fprs: dict[str, float] | None = None,
    ) -> CalibrationResult:
        """Compute calibrated thresholds from normal-frame scores.

        Args:
            normal_scores: 1D array of anomaly scores from known-normal frames.
                          Must have at least 50 samples for meaningful calibration.
            target_fprs: Target FPR per severity level.
                        Defaults: info=0.10, low=0.01, medium=0.001, high=0.0001

        Returns:
            CalibrationResult with four calibrated thresholds.

        Raises:
            ValueError: If fewer than 50 scores provided.
        """
        if len(normal_scores) < 50:
            raise ValueError(
                f"Need >= 50 calibration scores, got {len(normal_scores)}. "
                "Collect more baseline frames."
            )

        if target_fprs is None:
            target_fprs = {
                "info": 0.10,    # 10% FPR → 约 5/min @ 5 FPS (before temporal)
                "low": 0.01,     # 1% FPR
                "medium": 0.001, # 0.1% FPR
                "high": 0.0001,  # 0.01% FPR → 约 1/day @ 5 FPS
            }

        scores = np.sort(normal_scores)
        n = len(scores)

        # Conformal quantile: for target FPR alpha, use ceil((1-alpha)*(n+1))/n
        # This gives finite-sample coverage guarantee >= (1-alpha)
        thresholds = {}
        for level, alpha in target_fprs.items():
            quantile_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
            quantile_idx = min(quantile_idx, n - 1)  # clamp to valid range
            thresholds[level] = float(scores[quantile_idx])

        # Ensure ordering: info <= low <= medium <= high
        ordered = ["info", "low", "medium", "high"]
        for i in range(1, len(ordered)):
            if thresholds[ordered[i]] < thresholds[ordered[i - 1]]:
                thresholds[ordered[i]] = thresholds[ordered[i - 1]] + 0.01

        result = CalibrationResult(
            info_threshold=thresholds["info"],
            low_threshold=thresholds["low"],
            medium_threshold=thresholds["medium"],
            high_threshold=thresholds["high"],
            n_calibration_samples=n,
            target_fprs=target_fprs,
        )

        logger.info(
            "calibration.complete",
            n_samples=n,
            thresholds={k: round(v, 4) for k, v in thresholds.items()},
        )
        return result

    def save(self, result: CalibrationResult, path: Path) -> None:
        """Save calibration to JSON file alongside model."""
        data = {
            "info": result.info_threshold,
            "low": result.low_threshold,
            "medium": result.medium_threshold,
            "high": result.high_threshold,
            "n_samples": result.n_calibration_samples,
            "target_fprs": result.target_fprs,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> CalibrationResult | None:
        """Load calibration from JSON file. Returns None if not found."""
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return CalibrationResult(
            info_threshold=data["info"],
            low_threshold=data["low"],
            medium_threshold=data["medium"],
            high_threshold=data["high"],
            n_calibration_samples=data["n_samples"],
            target_fprs=data["target_fprs"],
        )
```

**边界**:
- 只创建 calibration.py，不改其他文件
- 不依赖 scipy（纯 numpy 实现）
- 不集成到 grader/trainer（A2-2 和 A2-3 做）

**验收**: `from argus.alerts.calibration import ConformalCalibrator` 无报错；手动测试 `calibrate(np.random.randn(1000))` 返回合理阈值。

---

#### A2-2: 添加校准配置 ✅ 已修改完

**目标**: 在 schema.py 中添加 `CalibrationConfig`。

**修改文件**: `src/argus/config/schema.py`

**修改过程**:

1. 在 `AlertConfig` 之前（约 L182）添加:
   ```python
   class CalibrationConfig(BaseModel):
       """Conformal prediction score calibration."""
       enabled: bool = Field(
           default=False,
           description="Use calibrated thresholds instead of manual severity_thresholds",
       )
       target_fpr_info: float = Field(default=0.10, ge=0.001, le=0.5)
       target_fpr_low: float = Field(default=0.01, ge=0.0001, le=0.1)
       target_fpr_medium: float = Field(default=0.001, ge=0.00001, le=0.01)
       target_fpr_high: float = Field(default=0.0001, ge=0.000001, le=0.001)
       min_calibration_samples: int = Field(default=50, ge=20, le=1000)
   ```

2. 在 `AlertConfig` (L182) 中添加:
   ```python
   calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
   ```

**边界**:
- 默认 `enabled=False`，不影响现有行为
- 不修改 `SeverityThresholds`（它保持作为手动回退）

**验收**: 配置文件加入 `calibration.enabled: true` 后能正确加载。

---

#### A2-3: 集成校准到训练流程 ✅ 已修改完

**目标**: 训练完成后自动跑校准，存储阈值。

**修改文件**: `src/argus/anomaly/trainer.py`

**修改过程**:

1. 在 `trainer.py` 的 `train()` 方法末尾（训练成功后、return 之前）添加校准步骤:

```python
# After successful training, run calibration if enabled
from argus.alerts.calibration import ConformalCalibrator

# Use 20% of baseline images as calibration set
all_images = sorted(baseline_dir.glob("*.png")) + sorted(baseline_dir.glob("*.jpg"))
n_cal = max(50, len(all_images) // 5)  # at least 50, or 20%
cal_images = all_images[-n_cal:]  # use last N (most recent)

# Run inference on calibration set to get score distribution
cal_scores = []
for img_path in cal_images:
    img = cv2.imread(str(img_path))
    if img is not None:
        result = detector.predict(img)
        cal_scores.append(result.anomaly_score)

if len(cal_scores) >= 50:
    calibrator = ConformalCalibrator()
    cal_result = calibrator.calibrate(np.array(cal_scores))
    cal_path = model_output_dir / "calibration.json"
    calibrator.save(cal_result, cal_path)
    logger.info("trainer.calibration_saved", path=str(cal_path))
```

**边界**:
- 校准是 best-effort：如果校准集不够 50 张，跳过并 log warning
- 校准文件保存在模型目录旁边（与模型版本绑定）
- 不修改 grader.py（A2-4 做）

**验收**: 训练一个 PatchCore 模型后，`data/models/{camera}/calibration.json` 被创建。

---

#### A2-4: Grader 加载校准阈值 ✅ 已修改完

**目标**: `AlertGrader` 初始化时尝试加载校准阈值，成功则覆盖手动阈值。

**修改文件**: `src/argus/alerts/grader.py`

**修改过程**:

1. 给 `AlertGrader.__init__` 添加可选参数 `calibration_path: Path | None = None`
2. 如果 calibration_path 存在且 config.calibration.enabled:
   ```python
   from argus.alerts.calibration import ConformalCalibrator

   if calibration_path and self._config.calibration.enabled:
       calibrator = ConformalCalibrator()
       cal = calibrator.load(calibration_path)
       if cal:
           self._config.severity_thresholds = SeverityThresholds(
               info=cal.info_threshold,
               low=cal.low_threshold,
               medium=cal.medium_threshold,
               high=cal.high_threshold,
           )
           logger.info("grader.using_calibrated_thresholds", thresholds=cal)
       else:
           logger.warning("grader.calibration_not_found", path=str(calibration_path))
   ```

**边界**:
- 回退到手动阈值（当校准不可用时）
- 不修改 `_score_to_severity()` 逻辑（它已经用 `self._config.severity_thresholds`）
- 需要在 `pipeline.py` 创建 `AlertGrader` 时传入 calibration_path

**验收**: 有 calibration.json 时使用校准阈值；无文件时回退到默认 0.50/0.70/0.85/0.95。

---

#### A2-5: 校准单元测试 ✅ 已修改完

**新建文件**: `tests/unit/test_calibration.py`

**测试用例**:
```python
def test_calibrate_with_normal_distribution():
    """正态分布 scores → 阈值递增。"""

def test_calibrate_minimum_samples():
    """< 50 samples → ValueError。"""

def test_threshold_ordering_guaranteed():
    """任何 score 分布 → info <= low <= medium <= high。"""

def test_save_and_load_roundtrip():
    """save → load → 阈值一致。"""

def test_load_missing_file():
    """文件不存在 → 返回 None。"""

def test_calibrate_with_uniform_scores():
    """均匀分布 [0, 0.3] → 阈值全在 0.3 附近。"""
```

**验收**: `pytest tests/unit/test_calibration.py -v` 全通过。

---

### A3: Simplex 双通道安全架构 ✅ 已修改完

#### A3-1: 创建 SimplexDetector 核心类 ✅ 已修改完

**目标**: 纯 OpenCV 的帧差分检测器，无 ML 依赖。

**新建文件**: `src/argus/prefilter/simple_detector.py`

**实现过程**:

```python
"""Simplex safety channel: formally verifiable frame-difference detector.

This detector uses only classical CV primitives (absdiff, threshold,
morphology, contour analysis) with no learned parameters. It provides
a detection floor that works even when the ML channel fails.

The invariant it guarantees:
  "Any object larger than min_area_px pixels that remains stationary
   for longer than min_static_seconds will be detected."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class StaticRegion:
    """A connected component that has been stationary."""
    centroid: tuple[int, int]
    area_px: int
    first_seen: float
    last_seen: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass
class SimplexResult:
    """Result from simplex detector."""
    has_detection: bool = False
    static_regions: list[StaticRegion] = field(default_factory=list)
    max_static_seconds: float = 0.0


class SimplexDetector:
    """Frame-difference based detector for the safety channel.

    Algorithm:
    1. cv2.absdiff(current, reference) → diff
    2. cv2.cvtColor(diff, GRAY) if color
    3. cv2.GaussianBlur(diff, (5,5)) → suppress noise
    4. cv2.threshold(diff, diff_threshold, 255, BINARY) → mask
    5. cv2.morphologyEx(mask, OPEN, kernel) → remove small noise
    6. cv2.morphologyEx(mask, CLOSE, kernel) → fill holes
    7. cv2.findContours → connected components
    8. Filter by min_area_px
    9. Track centroids: if a region's centroid stays within
       match_radius_px for > min_static_seconds → detection
    """

    def __init__(
        self,
        diff_threshold: int = 30,
        min_area_px: int = 500,
        min_static_seconds: float = 30.0,
        morph_kernel_size: int = 5,
        match_radius_px: int = 50,
    ):
        self._diff_threshold = diff_threshold
        self._min_area_px = min_area_px
        self._min_static_seconds = min_static_seconds
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self._match_radius = match_radius_px
        self._reference: np.ndarray | None = None
        self._tracked: list[StaticRegion] = []

    def set_reference(self, frame: np.ndarray) -> None:
        """Set the reference frame (from baseline)."""
        self._reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        self._tracked.clear()

    def detect(self, frame: np.ndarray) -> SimplexResult:
        """Run simplex detection on a single frame.

        Returns SimplexResult. Latency: <2ms on typical hardware.
        """
        if self._reference is None:
            return SimplexResult()

        now = time.monotonic()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Step 1-2: Frame difference
        diff = cv2.absdiff(gray, self._reference)

        # Step 3: Noise suppression
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # Step 4: Binary threshold
        _, mask = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)

        # Step 5-6: Morphology
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # Step 7: Connected components
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 8: Filter by area and extract centroids
        current_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self._min_area_px:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                current_regions.append((cx, cy, area, (x, y, w, h)))

        # Step 9: Match with tracked regions
        matched_indices = set()
        for cx, cy, area, bbox in current_regions:
            best_match = None
            best_dist = float("inf")
            for i, tracked in enumerate(self._tracked):
                if i in matched_indices:
                    continue
                dist = ((cx - tracked.centroid[0]) ** 2 + (cy - tracked.centroid[1]) ** 2) ** 0.5
                if dist < self._match_radius and dist < best_dist:
                    best_match = i
                    best_dist = dist

            if best_match is not None:
                # Update existing tracked region
                self._tracked[best_match].centroid = (cx, cy)
                self._tracked[best_match].area_px = area
                self._tracked[best_match].last_seen = now
                self._tracked[best_match].bbox = bbox
                matched_indices.add(best_match)
            else:
                # New region
                self._tracked.append(StaticRegion(
                    centroid=(cx, cy),
                    area_px=area,
                    first_seen=now,
                    last_seen=now,
                    bbox=bbox,
                ))

        # Prune regions not seen in this frame (gone)
        self._tracked = [
            t for i, t in enumerate(self._tracked)
            if i in matched_indices or (now - t.last_seen) < 2.0  # 2s grace
        ]

        # Check for static detections
        static = [
            t for t in self._tracked
            if (now - t.first_seen) >= self._min_static_seconds
        ]

        max_static = max((now - t.first_seen for t in self._tracked), default=0.0)

        return SimplexResult(
            has_detection=len(static) > 0,
            static_regions=static,
            max_static_seconds=max_static,
        )

    def reset(self) -> None:
        """Reset tracked regions (e.g., after model reload)."""
        self._tracked.clear()
```

**边界**:
- 纯 OpenCV + numpy，零 ML 依赖
- 只检测"大于 X 像素、静止超过 Y 秒"的物体
- 不做评分，只做二元检测 (has_detection)
- 不集成到 pipeline（A3-3 做）

**验收**: 传入白色背景参考帧 + 含黑色矩形的测试帧 → `has_detection=True`（等待 min_static_seconds 后）。

---

#### A3-2: Simplex 配置 ✅ 已修改完

**修改文件**: `src/argus/config/schema.py`

**在 `MOG2Config` 之后添加**:
```python
class SimplexConfig(BaseModel):
    """Simplex safety channel: formally verifiable frame-difference detector."""
    enabled: bool = Field(default=True, description="Enable simplex parallel detection channel")
    diff_threshold: int = Field(default=30, ge=10, le=100)
    min_area_px: int = Field(default=500, ge=100, le=50000)
    min_static_seconds: float = Field(default=30.0, ge=5.0, le=600.0)
    morph_kernel_size: int = Field(default=5, ge=3, le=15)
    match_radius_px: int = Field(default=50, ge=10, le=200)
```

**在 `CameraConfig` 中添加字段**:
```python
simplex: SimplexConfig = Field(default_factory=SimplexConfig)
```

**验收**: YAML 配置加入 `simplex.enabled: true` 后正确加载。

---

#### A3-3: 集成 Simplex 到 Pipeline ✅ 已修改完

**修改文件**: `src/argus/core/pipeline.py`

**修改过程**:

1. 在 `DetectionPipeline.__init__` 中创建 `SimplexDetector`:
   ```python
   from argus.prefilter.simple_detector import SimplexDetector

   if camera_config.simplex.enabled:
       self._simplex = SimplexDetector(
           diff_threshold=camera_config.simplex.diff_threshold,
           min_area_px=camera_config.simplex.min_area_px,
           min_static_seconds=camera_config.simplex.min_static_seconds,
           morph_kernel_size=camera_config.simplex.morph_kernel_size,
           match_radius_px=camera_config.simplex.match_radius_px,
       )
   else:
       self._simplex = None
   ```

2. 在 `process_frame()` 中，异常检测之后、alert grading 之前，运行 simplex:
   ```python
   # Run simplex in parallel
   simplex_result = None
   if self._simplex is not None:
       simplex_result = self._simplex.detect(masked_frame)

   # Merge: if simplex detects but ML doesn't, emit INFO-level alert
   if simplex_result and simplex_result.has_detection and (anomaly_result is None or not anomaly_result.is_anomalous):
       # Simplex-only detection → emit at INFO with simplex flag
       logger.info("pipeline.simplex_only_detection", regions=len(simplex_result.static_regions))
       # Create a synthetic anomaly_result with minimum score
       # ... (or directly create an Alert with severity=INFO and source="simplex")
   ```

3. 设置 simplex reference frame：在基线加载或模型部署时调用 `self._simplex.set_reference(first_baseline_frame)`

**边界**:
- Simplex 与 ML 通道取 OR 逻辑
- Simplex 检测不影响 ML 分数
- Simplex 结果在 Alert 中标记来源 (`source: "ml" | "simplex" | "both"`)

**验收**: ML 模型加载失败时，simplex 仍能检测大面积静止物体。

---

#### A3-4: Simplex 单元测试 ✅ 已修改完

**新建文件**: `tests/unit/test_simplex.py`

**测试用例**:
```python
def test_detect_static_object():
    """参考帧纯灰 → 当前帧有黑色矩形 → 等待 30s 后检测到。"""

def test_no_detection_on_reference():
    """参考帧 == 当前帧 → 无检测。"""

def test_small_object_filtered():
    """面积 < min_area_px → 不检测。"""

def test_moving_object_not_static():
    """物体每帧移动 → centroid 不匹配 → 不触发静止检测。"""

def test_reset_clears_tracking():
    """reset() 后 tracked regions 清空。"""

def test_reference_frame_required():
    """未设置 reference → 返回空结果。"""
```

**验收**: `pytest tests/unit/test_simplex.py -v` 全通过。

---

### A4: 主动学习基线采集 ✅ 已修改完

#### A4-1: 实现多样性选择算法 ✅ 已修改完

**修改文件**: `src/argus/anomaly/baseline.py`

**在 `BaselineManager` 类中添加方法**:

```python
def diversity_select(
    self,
    image_dir: Path,
    target_count: int,
    feature_size: tuple[int, int] = (64, 64),
) -> list[Path]:
    """Select the most diverse subset of images using k-center greedy.

    Uses color histogram features in LAB space for perceptual diversity.
    Returns paths of selected images, sorted by selection order.

    Algorithm (Sener & Savarese, ICLR 2018 simplified):
    1. Compute feature vector for each image (LAB color histogram)
    2. Start with first image as seed
    3. Iteratively add the image farthest from all already-selected images
    4. Stop at target_count
    """
    import cv2

    image_paths = sorted(
        list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    )

    if len(image_paths) <= target_count:
        return image_paths  # nothing to select

    # Step 1: Compute features (LAB color histograms, 3 channels x 32 bins = 96-dim)
    features = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            features.append(np.zeros(96))
            continue
        img = cv2.resize(img, feature_size)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist = []
        for ch in range(3):
            h = cv2.calcHist([lab], [ch], None, [32], [0, 256])
            h = h.flatten() / (h.sum() + 1e-8)  # normalize
            hist.append(h)
        features.append(np.concatenate(hist))

    features = np.array(features)  # (N, 96)

    # Step 2-4: K-center greedy
    n = len(features)
    selected = [0]  # start with first image
    min_distances = np.full(n, np.inf)

    for _ in range(target_count - 1):
        # Update min distances to selected set
        last = features[selected[-1]]
        dists = np.linalg.norm(features - last, axis=1)
        min_distances = np.minimum(min_distances, dists)
        min_distances[selected] = -1  # exclude already selected

        # Select farthest point
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)

    return [image_paths[i] for i in sorted(selected)]
```

**边界**:
- 不删除任何图片（只返回选择列表）
- 不修改 BaselineManager 的其他方法
- 特征计算在 CPU 上 ~5ms/帧

**验收**: 100 张图片选 20 张 → 返回 20 个 Path；返回的图片视觉上比均匀间隔更多样。

---

#### A4-2: Dashboard 集成优化按钮 ✅ 已修改完

**修改文件**: `src/argus/dashboard/routes/baseline.py`

**在基线采集完成后的页面添加"优化基线"按钮**:

1. 在采集完成的响应 HTML 中加入:
   ```html
   <button hx-post="/api/baseline/optimize"
           hx-vals='{"camera_id": "...", "target_ratio": 0.2}'
           hx-target="#optimize-result">
     优化基线（保留最多样的 20%）
   </button>
   ```

2. 新增路由 `POST /api/baseline/optimize`:
   ```python
   @router.post("/optimize")
   async def optimize_baseline(request: Request):
       camera_id = form.get("camera_id")
       target_ratio = float(form.get("target_ratio", 0.2))
       baseline_dir = baseline_mgr.get_baseline_dir(camera_id, "default")
       all_images = list(baseline_dir.glob("*.png"))
       target_count = max(30, int(len(all_images) * target_ratio))
       selected = baseline_mgr.diversity_select(baseline_dir, target_count)
       # Move unselected to backup directory
       ...
   ```

**边界**:
- 不自动执行，需要操作员点击
- 被排除的图片移到 backup 子目录（不删除）
- 最少保留 30 张

**验收**: 采集 200 帧 → 点击优化 → 保留 40 帧，其余移到 backup/。

---

#### A4-3: 基线多样性测试 ✅ 已修改完

**修改文件**: `tests/unit/test_baseline.py`

**新增测试**:
```python
def test_diversity_select_reduces_count():
    """100 张图 → target=20 → 返回 20 个 Path。"""

def test_diversity_select_with_fewer_images():
    """图片数 < target → 返回全部。"""

def test_diversity_select_returns_sorted_paths():
    """返回的 Path 列表按文件名排序。"""
```

---

## Phase B: Dinomaly2 骨干升级

---

### B1: Dinomaly2 配置与训练集成 ✅ 已修改完

#### B1-1: 验证 Anomalib 中 Dinomaly2 可用性 ✅ 已修改完

**目标**: 确认当前 anomalib 版本是否包含 Dinomaly/Dinomaly2，如果不包含则确定升级路径。

**过程**:

```bash
# 在项目虚拟环境中运行
pip show anomalib  # 查看当前版本

# 尝试导入 Dinomaly
python -c "from anomalib.models import Dinomaly; print('Dinomaly available')"

# 如果失败，检查 Anomalib 2.1+ 是否有
pip install --dry-run "anomalib>=2.1.0,<3.0.0"

# 查看可用模型列表
python -c "from anomalib.models import get_model; help(get_model)"
```

**决策点**:
- 如果 Dinomaly 在 anomalib>=2.1.0 中可用 → 升级 pyproject.toml 并继续 B1-2
- 如果不可用 → 需要从 GitHub 安装最新 anomalib 或手动集成 Dinomaly2 代码
- Dinomaly2 需要 DINOv2 backbone → 确认 `timm` 或 `torch.hub` 是否有 `dinov2_vitb14`

**边界**: 只做验证，不改代码。

**验收**: 明确 Dinomaly2 的集成路径（anomalib 内置 or 手动集成）。

---

#### B1-2: 添加 Dinomaly2 配置 ✅ 已修改完

**修改文件**: `src/argus/config/schema.py`

**修改过程**:

1. 修改 `AnomalyConfig.model_type` Literal (L78):
   ```python
   model_type: Literal["patchcore", "efficient_ad", "anomalydino", "dinomaly2"] = "patchcore"
   ```

2. 在 `AnomalyConfig` 中添加 Dinomaly2 特定参数:
   ```python
   # Dinomaly2 parameters (only used when model_type="dinomaly2")
   dinomaly_backbone: str = Field(
       default="dinov2_vitb14",
       description="DINOv2 backbone variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)",
   )
   dinomaly_encoder_layers: list[int] = Field(
       default=[2, 5, 8, 11],
       description="Intermediate ViT layers to extract features from",
   )
   dinomaly_few_shot_images: int = Field(
       default=8, ge=1, le=100,
       description="Minimum baseline images for few-shot mode (Dinomaly2 supports 8-shot)",
   )
   dinomaly_multi_class: bool = Field(
       default=False,
       description="Use unified multi-class model for all cameras (shared backbone)",
   )
   ```

**边界**:
- 默认 `model_type` 仍为 `"patchcore"`
- 新参数只在 `model_type="dinomaly2"` 时生效
- 不改 SSIM 回退参数

**验收**: YAML 中 `anomaly.model_type: dinomaly2` + `anomaly.dinomaly_backbone: dinov2_vitb14` 正确解析。

---

#### B1-3: 实现 Dinomaly2 训练路径 ✅ 已修改完

**修改文件**: `src/argus/anomaly/trainer.py`

**当前状态**: `_train_anomalib()` 方法在创建模型时只处理 PatchCore 和 EfficientAD:
```python
if model_type == "patchcore":
    model = Patchcore(backbone=..., layers_to_extract=...)
elif model_type == "efficient_ad":
    model = EfficientAd(...)
```

**修改过程**:

1. 添加 Dinomaly2 分支:
   ```python
   elif model_type == "dinomaly2":
       # Dinomaly2 uses DINOv2 backbone with reconstruction-based anomaly detection
       from anomalib.models import Dinomaly  # or custom import path

       model = Dinomaly(
           backbone=anomaly_config.dinomaly_backbone,
           layers=anomaly_config.dinomaly_encoder_layers,
       )
       # Dinomaly2 训练参数
       max_epochs = 1  # Dinomaly2 类似 PatchCore，是 feature-extraction based
       # 注意：具体 API 取决于 B1-1 的验证结果
   ```

2. 处理 few-shot 模式：如果基线图片数 < 30 但 >= `dinomaly_few_shot_images`:
   ```python
   n_images = baseline_mgr.count_images(camera_id, zone_id)
   if n_images < 30 and model_type == "dinomaly2":
       if n_images >= anomaly_config.dinomaly_few_shot_images:
           logger.info("trainer.few_shot_mode", n_images=n_images)
           # Proceed with few-shot training
       else:
           return TrainingResult(status="failed", error=f"Need >= {anomaly_config.dinomaly_few_shot_images} images for few-shot")
   ```

**边界**:
- 不改 PatchCore / EfficientAD 路径
- API 细节取决于 B1-1 验证
- 如果 Anomalib 没有内置 Dinomaly，这里需要手动构建模型

**验收**: `model_type="dinomaly2"` + 50 张基线 → 训练成功 → 模型可加载推理。

---

#### B1-4: 验证 Dinomaly2 推理兼容性 ✅ 已修改完

**修改文件**: `src/argus/anomaly/detector.py`

**目标**: 确保 Dinomaly2 模型通过现有 OpenVINO/Torch inferencer 正确加载和推理。

**过程**:

1. 用 B1-3 训练出的模型测试加载:
   ```python
   detector = AnomalibDetector(
       model_path="data/exports/cam_01/model.xml",  # OpenVINO
       threshold=0.7,
       image_size=(256, 256),
   )
   detector.load()
   result = detector.predict(test_frame)
   # 验证 result.anomaly_score 在 [0, 1] 范围
   # 验证 result.anomaly_map 形状正确
   ```

2. 如果 Dinomaly2 输出格式与 PatchCore 不同，在 `predict()` 中添加模型类型判断:
   ```python
   if self._model_type == "dinomaly2":
       # Dinomaly2 可能返回不同的 score 范围或 heatmap 格式
       # 做必要的归一化
       ...
   ```

**边界**:
- 不改 SSIM 回退路径
- 如果 OpenVINO 导出不支持 Dinomaly2 的某些算子，回退到 Torch 推理

**验收**: `detector.predict(frame)` 返回有效的 `AnomalyResult`，分数在 [0,1] 范围。

---

#### B1-5: 多摄像头共享模型（可选） ✅ 已修改完

**目标**: Dinomaly2 的多类统一模型允许多台摄像头共享一个模型实例。

**修改文件**: `src/argus/capture/manager.py`

**当前状态**: 每个 `DetectionPipeline` 在 `__init__` 中独立创建 `AnomalibDetector`。

**修改过程**:

1. 在 `CameraManager` 中添加共享模型管理:
   ```python
   class CameraManager:
       def __init__(self, ...):
           ...
           self._shared_detector: AnomalibDetector | None = None

       def _get_detector(self, camera_config: CameraConfig) -> AnomalibDetector:
           if camera_config.anomaly.dinomaly_multi_class and self._shared_detector:
               return self._shared_detector
           # Otherwise create per-camera detector
           return AnomalibDetector(...)
   ```

2. 修改 `DetectionPipeline` 接受外部传入的 detector 而不是自己创建。

**边界**:
- 只在 `dinomaly_multi_class=True` 时共享
- 线程安全：推理时用 Lock 或者确保 inferencer 是线程安全的
- PatchCore/EfficientAD 仍然一机一模型

**验收**: 2 台摄像头配置 `dinomaly_multi_class: true` → 共享一个 detector 实例 → 两台都能正常推理。

---

#### B1-6: Dinomaly2 集成测试 ✅ 已修改完

**修改文件**: `tests/unit/test_config.py` + 新建 `tests/unit/test_dinomaly.py`

**测试用例**:
```python
# test_config.py 补充
def test_dinomaly2_config_valid():
    """model_type=dinomaly2 + backbone + layers 正确解析。"""

def test_dinomaly2_few_shot_minimum():
    """dinomaly_few_shot_images=8 + 只有 5 张图 → 报错。"""

# test_dinomaly.py (如果可以 mock)
def test_dinomaly2_training_smoke():
    """用 10 张 256x256 灰度图训练 → 不崩溃。"""

def test_dinomaly2_predict_returns_valid_result():
    """推理返回 score in [0,1] + anomaly_map shape 正确。"""
```

---

### B2: INT8 量化导出 ✅ 已修改完

#### B2-1: 添加量化配置 ✅ 已修改完

**修改文件**: `src/argus/config/schema.py`

**在 `AnomalyConfig` 中添加**:
```python
quantization: Literal["fp32", "fp16", "int8"] = Field(
    default="fp16",
    description="Model precision for inference (fp32=full, fp16=half, int8=quantized)",
)
quantization_calibration_images: int = Field(
    default=100, ge=50, le=1000,
    description="Number of baseline images used for INT8 calibration",
)
```

**验收**: YAML 中 `anomaly.quantization: int8` 正确解析。

---

#### B2-2: 实现 INT8 PTQ 量化流程 ✅ 已修改完

**修改文件**: `scripts/export_model.py` + `src/argus/anomaly/trainer.py`

**过程**:

1. 在 `trainer.py` 的导出步骤后添加量化:
   ```python
   if anomaly_config.quantization == "int8":
       try:
           import nncf  # Neural Network Compression Framework

           # Load calibration images
           cal_images = load_calibration_images(
               baseline_dir,
               count=anomaly_config.quantization_calibration_images,
               image_size=anomaly_config.image_size,
           )

           # Create calibration dataset
           cal_dataset = nncf.Dataset(cal_images, transform_fn)

           # Quantize OpenVINO model
           import openvino as ov
           core = ov.Core()
           model = core.read_model(str(export_path / "model.xml"))
           quantized = nncf.quantize(model, cal_dataset, preset=nncf.QuantizationPreset.MIXED)
           ov.save_model(quantized, str(export_path / "model_int8.xml"))

       except ImportError:
           logger.warning("trainer.nncf_not_available", msg="Install nncf for INT8 quantization")
   ```

2. 在 `scripts/export_model.py` 中添加 `--quantize int8` CLI 选项。

**边界**:
- `nncf` 是可选依赖（`pip install nncf`）
- 如果 nncf 不可用，回退到 fp16
- 量化后自动做精度回归检查（A2 的校准分数对比）

**验收**: 导出 INT8 模型 → 文件大小约为 FP32 的 1/4 → 推理延迟降低 2-3x。

---

#### B2-3: 量化精度验证 ✅ 已修改完

**修改文件**: `src/argus/anomaly/trainer.py`

**在量化完成后添加精度回归检查**:

```python
# Run both FP32 and INT8 on same calibration set
fp32_scores = run_inference(fp32_model, cal_images)
int8_scores = run_inference(int8_model, cal_images)

# Compare
score_diff = np.abs(fp32_scores - int8_scores)
max_diff = score_diff.max()
mean_diff = score_diff.mean()

if max_diff > 0.1:  # 10% max score deviation
    logger.warning(
        "trainer.quantization_degradation",
        max_diff=max_diff,
        mean_diff=mean_diff,
        msg="INT8 quantization may have degraded accuracy. Consider FP16.",
    )
```

**验收**: 量化后 max_diff < 0.05（5%）对于 PatchCore；Dinomaly2 ViT 可能需要 mixed precision。

---

#### B2-4: 量化测试 ✅ 已修改完

**新建文件**: `tests/unit/test_quantization.py`

```python
def test_quantization_config_values():
    """fp32/fp16/int8 都是合法值。"""

def test_int8_requires_calibration_images():
    """quantization=int8 + calibration_images < 50 → 验证报错。"""
```

---

## Phase C: 运维智能

---

### C1: 摄像头健康层 ✅ 已修改完

#### C1-1: ✅ 创建 CameraHealthAnalyzer 核心类

**新建文件**: `src/argus/capture/health.py`

**实现过程**:

```python
"""Camera hardware health monitoring.

Detects 5 classes of camera issues that would otherwise cause
systematic false positives or missed detections:
1. Frame freeze (decoder hang, TCP still connected)
2. Lens contamination / fogging
3. Mechanical displacement
4. Flash / arc welding suppression
5. Auto-gain / exposure drift
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class HealthCheckResult:
    """Result of camera health analysis."""
    is_frozen: bool = False
    sharpness_score: float = 0.0      # Laplacian variance, higher = sharper
    sharpness_baseline: float = 0.0   # calibrated at startup
    displacement_px: float = 0.0      # cumulative displacement from reference
    is_flash: bool = False            # sudden brightness spike
    brightness_mean: float = 0.0
    gain_drift_pct: float = 0.0       # % change from baseline brightness
    suppress_detection: bool = False   # if True, skip anomaly detection this frame
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class CameraHealthAnalyzer:
    """Analyzes per-frame camera health metrics.

    Each check costs <1ms. Total overhead: ~3-5ms per frame.
    """

    def __init__(
        self,
        freeze_window: int = 10,        # frames to check for freeze
        freeze_hash_threshold: float = 0.01,  # pHash variance threshold
        sharpness_drop_pct: float = 0.3, # 30% drop from baseline = contamination
        displacement_threshold_px: float = 20.0,  # cumulative px shift
        flash_sigma: float = 3.0,        # brightness spike threshold in std devs
        brightness_window: int = 300,    # frames for rolling brightness stats (60s @ 5fps)
        gain_drift_threshold_pct: float = 20.0,  # % change in 1h
    ):
        self._freeze_window = freeze_window
        self._freeze_hash_threshold = freeze_hash_threshold
        self._sharpness_drop_pct = sharpness_drop_pct
        self._displacement_threshold = displacement_threshold_px
        self._flash_sigma = flash_sigma
        self._gain_drift_threshold = gain_drift_threshold_pct

        # Rolling buffers
        self._frame_hashes: deque[int] = deque(maxlen=freeze_window)
        self._brightness_history: deque[float] = deque(maxlen=brightness_window)

        # Baselines (calibrated during first N frames)
        self._sharpness_baseline: float | None = None
        self._brightness_baseline: float | None = None
        self._reference_gray: np.ndarray | None = None
        self._cumulative_dx: float = 0.0
        self._cumulative_dy: float = 0.0
        self._prev_gray: np.ndarray | None = None
        self._calibration_count = 0
        self._calibration_sharpness: list[float] = []
        self._calibration_brightness: list[float] = []

    def analyze(self, frame: np.ndarray) -> HealthCheckResult:
        """Run all health checks on a single frame. <5ms total."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        result = HealthCheckResult()

        # 1. Frame freeze detection (pHash variance)
        frame_hash = self._compute_phash(gray)
        self._frame_hashes.append(frame_hash)
        if len(self._frame_hashes) >= self._freeze_window:
            hash_variance = np.var([h for h in self._frame_hashes])
            result.is_frozen = hash_variance < self._freeze_hash_threshold
            if result.is_frozen:
                result.warnings.append("frame_frozen")
                result.suppress_detection = True

        # 2. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        result.sharpness_score = sharpness

        # 3. Brightness
        brightness = float(gray.mean())
        result.brightness_mean = brightness
        self._brightness_history.append(brightness)

        # Calibration phase (first 30 frames)
        if self._calibration_count < 30:
            self._calibration_sharpness.append(sharpness)
            self._calibration_brightness.append(brightness)
            self._calibration_count += 1
            if self._calibration_count == 30:
                self._sharpness_baseline = np.median(self._calibration_sharpness)
                self._brightness_baseline = np.median(self._calibration_brightness)
                self._reference_gray = gray.copy()
                logger.info(
                    "camera_health.calibrated",
                    sharpness=round(self._sharpness_baseline, 1),
                    brightness=round(self._brightness_baseline, 1),
                )
            self._prev_gray = gray.copy()
            return result

        result.sharpness_baseline = self._sharpness_baseline

        # 2b. Lens contamination check
        if sharpness < self._sharpness_baseline * (1 - self._sharpness_drop_pct):
            result.warnings.append("lens_contamination")

        # 3b. Flash detection
        if len(self._brightness_history) >= 10:
            recent = list(self._brightness_history)[-60:]  # last 12 seconds
            mean_b = np.mean(recent[:-1]) if len(recent) > 1 else brightness
            std_b = np.std(recent[:-1]) if len(recent) > 1 else 1.0
            if std_b > 0 and abs(brightness - mean_b) > self._flash_sigma * std_b:
                result.is_flash = True
                result.warnings.append("flash_detected")
                result.suppress_detection = True

        # 3c. Gain drift
        if self._brightness_baseline and self._brightness_baseline > 0:
            drift = abs(brightness - self._brightness_baseline) / self._brightness_baseline * 100
            result.gain_drift_pct = drift
            if drift > self._gain_drift_threshold:
                result.warnings.append("gain_drift")

        # 4. Mechanical displacement (phase correlation)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            try:
                shift, _ = cv2.phaseCorrelate(
                    self._prev_gray.astype(np.float64),
                    gray.astype(np.float64),
                )
                dx, dy = shift
                if abs(dx) < 5 and abs(dy) < 5:  # ignore large motion
                    self._cumulative_dx += dx
                    self._cumulative_dy += dy
                displacement = (self._cumulative_dx**2 + self._cumulative_dy**2) ** 0.5
                result.displacement_px = displacement
                if displacement > self._displacement_threshold:
                    result.warnings.append("mechanical_displacement")
            except cv2.error:
                pass

        self._prev_gray = gray.copy()
        return result

    def _compute_phash(self, gray: np.ndarray) -> int:
        """Compute simple perceptual hash (8x8 DCT-based)."""
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        mean_val = resized.mean()
        return int(np.packbits((resized > mean_val).flatten())[0])

    def reset(self) -> None:
        """Reset all baselines (e.g., after camera recalibration)."""
        self._frame_hashes.clear()
        self._brightness_history.clear()
        self._sharpness_baseline = None
        self._brightness_baseline = None
        self._reference_gray = None
        self._cumulative_dx = 0.0
        self._cumulative_dy = 0.0
        self._prev_gray = None
        self._calibration_count = 0
        self._calibration_sharpness.clear()
        self._calibration_brightness.clear()
```

**边界**:
- 只用 OpenCV 和 numpy
- 每个检查独立，可单独开关
- `suppress_detection` 是给 pipeline 的建议，不强制

**验收**: 传入冻结帧序列 → `is_frozen=True`；传入模糊帧 → `sharpness_score` 远低于 baseline。

---

#### C1-2: ✅ 健康层配置

**修改文件**: `src/argus/config/schema.py`

**添加**:
```python
class CameraHealthConfig(BaseModel):
    """Camera hardware health monitoring."""
    enabled: bool = Field(default=True)
    freeze_detection: bool = Field(default=True)
    lens_contamination_detection: bool = Field(default=True)
    displacement_detection: bool = Field(default=True)
    flash_suppression: bool = Field(default=True)
    gain_drift_detection: bool = Field(default=True)
    freeze_window_frames: int = Field(default=10, ge=5, le=30)
    sharpness_drop_pct: float = Field(default=0.3, ge=0.1, le=0.8)
    displacement_threshold_px: float = Field(default=20.0, ge=5.0, le=100.0)
    flash_sigma: float = Field(default=3.0, ge=2.0, le=5.0)
    gain_drift_threshold_pct: float = Field(default=20.0, ge=5.0, le=50.0)
```

**在 `CameraConfig` 中添加**:
```python
health: CameraHealthConfig = Field(default_factory=CameraHealthConfig)
```

---

#### C1-3: ✅ 集成到 Pipeline

**修改文件**: `src/argus/core/pipeline.py`

**在 `process_frame()` 最前面（Zone Masking 之前）插入**:

```python
# Stage -1: Camera health check
if self._health_analyzer:
    health = self._health_analyzer.analyze(frame_data.frame)
    if health.suppress_detection:
        logger.info("pipeline.health_suppressed", reason=health.warnings)
        self._stats.frames_skipped_health += 1
        return None  # skip this frame entirely
    if health.warnings:
        logger.warning("pipeline.health_warnings", warnings=health.warnings)
```

**验收**: 闪光帧 → 跳过检测；冻结帧 → 跳过并告警。

---

#### C1-4: ✅ HealthMonitor 扩展

**修改文件**: `src/argus/core/health.py`

**在 `CameraHealth` dataclass 中添加**:
```python
sharpness_score: float = 0.0
displacement_px: float = 0.0
is_frozen: bool = False
gain_drift_pct: float = 0.0
health_warnings: list[str] = field(default_factory=list)
```

---

#### C1-5: ✅ Dashboard 健康指标显示

**修改文件**: `src/argus/dashboard/routes/cameras.py`

**在摄像头状态卡片中添加健康指标**:
- 锐度条形图（绿/黄/红）
- 位移累积值
- 冻结/闪光/漂移状态标签
- 工具提示说明每个指标含义

---

#### C1-6: ✅ 健康层测试

**新建文件**: `tests/unit/test_camera_health.py`

```python
def test_frozen_frame_detection():
    """10 帧完全相同 → is_frozen=True。"""

def test_normal_frames_not_frozen():
    """10 帧有轻微差异 → is_frozen=False。"""

def test_flash_detection():
    """正常帧序列中插入一帧全白 → is_flash=True。"""

def test_sharpness_calibration():
    """前 30 帧后 sharpness_baseline 被设置。"""

def test_blur_detection():
    """sharpness 降低 50% → warnings 包含 lens_contamination。"""

def test_displacement_accumulation():
    """每帧平移 1px → 20 帧后 displacement_px ≈ 20。"""
```

---

### C2: KS 漂移监控 ✅ 已修改完

#### C2-1: ✅ 创建 DriftDetector 核心类

**新建文件**: `src/argus/anomaly/drift.py`

```python
"""Model drift detection using Kolmogorov-Smirnov test.

Compares rolling window of recent anomaly scores against the reference
distribution from training/calibration time. Alerts when distributions
diverge significantly.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class DriftStatus:
    """Current drift detection status."""
    is_drifted: bool = False
    ks_statistic: float = 0.0
    p_value: float = 1.0
    reference_mean: float = 0.0
    current_mean: float = 0.0
    samples_collected: int = 0
    last_check_time: float = 0.0


class DriftDetector:
    """Detects score distribution drift using KS test.

    Usage:
        detector = DriftDetector(reference_scores)
        for each frame:
            detector.update(anomaly_score)
        status = detector.get_status()
    """

    def __init__(
        self,
        reference_scores: np.ndarray | None = None,
        window_size: int = 5000,       # ~16 min @ 5fps
        check_interval: int = 500,     # check every 500 scores
        ks_threshold: float = 0.1,     # KS statistic threshold
        p_value_threshold: float = 0.01,
    ):
        self._reference = np.sort(reference_scores) if reference_scores is not None else None
        self._window: deque[float] = deque(maxlen=window_size)
        self._check_interval = check_interval
        self._ks_threshold = ks_threshold
        self._p_value_threshold = p_value_threshold
        self._count_since_check = 0
        self._status = DriftStatus()

    def set_reference(self, scores: np.ndarray) -> None:
        """Set reference distribution (from calibration)."""
        self._reference = np.sort(scores)
        self._status.reference_mean = float(scores.mean())

    def update(self, score: float) -> None:
        """Feed a new anomaly score. Periodically runs KS test."""
        self._window.append(score)
        self._count_since_check += 1

        if self._count_since_check >= self._check_interval and len(self._window) >= 100:
            self._run_check()
            self._count_since_check = 0

    def _run_check(self) -> None:
        """Run KS test between reference and current window."""
        if self._reference is None or len(self._window) < 100:
            return

        current = np.array(self._window)
        ks_stat, p_value = self._ks_2samp(self._reference, current)

        self._status = DriftStatus(
            is_drifted=(ks_stat > self._ks_threshold and p_value < self._p_value_threshold),
            ks_statistic=float(ks_stat),
            p_value=float(p_value),
            reference_mean=float(self._reference.mean()),
            current_mean=float(current.mean()),
            samples_collected=len(self._window),
            last_check_time=time.time(),
        )

        if self._status.is_drifted:
            logger.warning(
                "drift.detected",
                ks=round(ks_stat, 4),
                p=round(p_value, 6),
                ref_mean=round(self._status.reference_mean, 4),
                cur_mean=round(self._status.current_mean, 4),
            )

    @staticmethod
    def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        """Pure numpy KS two-sample test (no scipy dependency).

        Returns (ks_statistic, approximate_p_value).
        """
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        all_values = np.concatenate([a_sorted, b_sorted])
        all_values.sort()

        cdf_a = np.searchsorted(a_sorted, all_values, side="right") / len(a)
        cdf_b = np.searchsorted(b_sorted, all_values, side="right") / len(b)
        ks_stat = float(np.max(np.abs(cdf_a - cdf_b)))

        # Approximate p-value using asymptotic formula
        n = len(a) * len(b) / (len(a) + len(b))
        # Kolmogorov distribution approximation
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * ks_stat
        if lam < 0.001:
            p_value = 1.0
        else:
            # First few terms of the series
            p_value = 2 * sum(
                (-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam)
                for k in range(1, 6)
            )
            p_value = max(0.0, min(1.0, p_value))

        return ks_stat, p_value

    def get_status(self) -> DriftStatus:
        return self._status
```

**边界**:
- 纯 numpy 实现，不依赖 scipy
- 只做监控（Dashboard 显示），不自动重训
- `_ks_2samp` 的 p-value 是近似值（Kolmogorov 渐近公式）

**验收**: 传入均值为 0.1 的参考分布 + 均值为 0.3 的当前窗口 → `is_drifted=True`。

---

#### C2-2: ✅ 漂移配置

**修改文件**: `src/argus/config/schema.py`

```python
class DriftConfig(BaseModel):
    """Score distribution drift monitoring."""
    enabled: bool = Field(default=True)
    window_size: int = Field(default=5000, ge=500, le=100000)
    check_interval: int = Field(default=500, ge=100, le=5000)
    ks_threshold: float = Field(default=0.1, ge=0.01, le=0.5)
    p_value_threshold: float = Field(default=0.01, ge=0.001, le=0.1)
```

**在 `CameraConfig` 中添加**:
```python
drift: DriftConfig = Field(default_factory=DriftConfig)
```

---

#### C2-3: ✅ 集成到 Pipeline

**修改文件**: `src/argus/core/pipeline.py`

在 anomaly detection 后喂分数:
```python
if self._drift_detector:
    self._drift_detector.update(anomaly_result.anomaly_score)
```

---

#### C2-4: ✅ 漂移测试

**新建文件**: `tests/unit/test_drift.py`

```python
def test_no_drift_same_distribution():
    """参考和当前来自同一分布 → is_drifted=False。"""

def test_drift_detected_shifted_distribution():
    """参考 mean=0.1, 当前 mean=0.4 → is_drifted=True。"""

def test_ks_2samp_known_values():
    """和 scipy.stats.ks_2samp 对比结果一致（误差 < 0.01）。"""

def test_insufficient_samples_no_check():
    """< 100 samples → 不运行检查。"""
```

---

### C3: 跨摄像头关联 ✅ 已修改完

#### C3-1: ✅ 创建 CrossCameraCorrelator

**新建文件**: `src/argus/core/correlation.py`

```python
"""Cross-camera anomaly correlation.

When cameras have overlapping fields of view, an anomaly seen by one
camera but not corroborated by another is likely a false positive
(e.g., lens-specific contamination, single-camera glare).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CameraOverlapPair:
    """A pair of cameras with overlapping fields of view."""
    camera_a: str
    camera_b: str
    # 3x3 homography matrix: projects points from camera_a to camera_b
    homography: list[list[float]]


@dataclass
class CorrelationResult:
    """Result of cross-camera correlation check."""
    corroborated: bool = True  # default: no correlation data = assume true
    partner_camera: str | None = None
    partner_score_at_location: float = 0.0


class CrossCameraCorrelator:
    """Correlates anomalies across overlapping camera pairs.

    Usage:
    1. Configure camera pairs with homography matrices
    2. Feed anomaly results from each camera
    3. Query correlation before emitting alert
    """

    def __init__(self, pairs: list[CameraOverlapPair]):
        self._pairs: dict[str, list[tuple[str, np.ndarray]]] = {}
        for pair in pairs:
            H = np.array(pair.homography, dtype=np.float64)
            self._pairs.setdefault(pair.camera_a, []).append((pair.camera_b, H))
            # Inverse for the other direction
            H_inv = np.linalg.inv(H)
            self._pairs.setdefault(pair.camera_b, []).append((pair.camera_a, H_inv))

        # Recent anomaly maps per camera (last 5 seconds)
        self._recent_maps: dict[str, tuple[float, np.ndarray]] = {}

    def update(self, camera_id: str, anomaly_map: np.ndarray | None, timestamp: float) -> None:
        """Store the latest anomaly map for a camera."""
        if anomaly_map is not None:
            self._recent_maps[camera_id] = (timestamp, anomaly_map)

    def check(
        self,
        camera_id: str,
        anomaly_location: tuple[int, int],  # (x, y) in camera frame
        timestamp: float,
        max_age_seconds: float = 5.0,
    ) -> CorrelationResult:
        """Check if an anomaly is corroborated by partner cameras.

        Args:
            camera_id: Camera that detected the anomaly
            anomaly_location: (x, y) pixel location in that camera's frame
            timestamp: Detection timestamp
            max_age_seconds: Max age of partner data to consider

        Returns:
            CorrelationResult with corroborated=True if any partner
            camera shows elevated score at the projected location.
        """
        partners = self._pairs.get(camera_id, [])
        if not partners:
            return CorrelationResult()  # no partners, assume corroborated

        src_point = np.array([[anomaly_location]], dtype=np.float64)

        for partner_id, H in partners:
            partner_data = self._recent_maps.get(partner_id)
            if partner_data is None:
                continue

            partner_time, partner_map = partner_data
            if timestamp - partner_time > max_age_seconds:
                continue

            # Project anomaly location to partner camera
            dst_point = cv2.perspectiveTransform(src_point, H)
            px, py = int(dst_point[0, 0, 0]), int(dst_point[0, 0, 1])

            # Check if projected point is within partner map bounds
            h, w = partner_map.shape[:2]
            if 0 <= px < w and 0 <= py < h:
                # Check score in a neighborhood (±25px)
                r = 25
                y1, y2 = max(0, py - r), min(h, py + r)
                x1, x2 = max(0, px - r), min(w, px + r)
                region_score = float(partner_map[y1:y2, x1:x2].max())

                if region_score > 0.3:  # partner also sees something
                    return CorrelationResult(
                        corroborated=True,
                        partner_camera=partner_id,
                        partner_score_at_location=region_score,
                    )

        # No partner corroboration found
        return CorrelationResult(
            corroborated=False,
            partner_camera=partners[0][0] if partners else None,
            partner_score_at_location=0.0,
        )

    def prune_stale(self, max_age: float = 30.0) -> None:
        """Remove stale anomaly maps."""
        now = time.time()
        stale = [k for k, (t, _) in self._recent_maps.items() if now - t > max_age]
        for k in stale:
            del self._recent_maps[k]
```

**边界**:
- 需要预先标定 homography（手动或通过对应点计算）
- 无 partner 的摄像头默认 `corroborated=True`（不降级）
- Partner 数据超过 5 秒不使用

**验收**: 两台模拟摄像头，cam_a 检测到异常 → 投影到 cam_b → cam_b 无对应信号 → `corroborated=False`。

---

#### C3-2: ✅ 跨摄像头配置

**修改文件**: `src/argus/config/schema.py`

```python
class CameraOverlapConfig(BaseModel):
    """Overlap between two cameras for cross-correlation."""
    camera_a: str
    camera_b: str
    homography: list[list[float]] = Field(
        description="3x3 homography matrix projecting from camera_a to camera_b",
    )

class CrossCameraConfig(BaseModel):
    """Cross-camera anomaly correlation."""
    enabled: bool = Field(default=False)
    overlap_pairs: list[CameraOverlapConfig] = Field(default_factory=list)
    uncorroborated_severity_downgrade: int = Field(
        default=1, ge=0, le=2,
        description="Downgrade severity by N levels when uncorroborated (0=disabled)",
    )
```

**在 `ArgusConfig` 中添加**:
```python
cross_camera: CrossCameraConfig = Field(default_factory=CrossCameraConfig)
```

---

#### C3-3: ✅ 集成到 CameraManager

**修改文件**: `src/argus/capture/manager.py`

在 `_alert_handler` 中插入 correlation check:
```python
if self._correlator and alert:
    # Extract anomaly location from heatmap peak
    if alert.heatmap is not None:
        peak_y, peak_x = np.unravel_index(alert.heatmap.argmax(), alert.heatmap.shape)
        corr = self._correlator.check(alert.camera_id, (peak_x, peak_y), alert.timestamp)
        if not corr.corroborated:
            # Downgrade severity
            ...
```

---

#### C3-4: ✅ Alert 加 corroborated 字段

**修改文件**: `src/argus/alerts/grader.py`

```python
@dataclass
class Alert:
    ...
    corroborated: bool = True  # NEW
    correlation_partner: str | None = None  # NEW
```

---

#### C3-5: ✅ 跨摄像头测试

**新建文件**: `tests/unit/test_correlation.py`

```python
def test_corroborated_when_partner_agrees():
    """两台摄像头同一位置都有高分 → corroborated=True。"""

def test_uncorroborated_when_partner_clean():
    """cam_a 异常但 cam_b 对应位置正常 → corroborated=False。"""

def test_no_partners_defaults_to_corroborated():
    """无 overlap pair → corroborated=True。"""

def test_stale_partner_data_ignored():
    """partner 数据超过 5s → 不使用。"""

def test_homography_projection_correct():
    """已知 H 矩阵 → 投影坐标正确。"""
```

---

### C4: 最小可行 MLOps

#### C4-1: ✅ 创建 ModelRecord ORM 和 Registry

**修改文件**: `src/argus/storage/models.py` + NEW `src/argus/storage/model_registry.py`

在 models.py 加:
```python
class ModelRecord(Base):
    __tablename__ = "models"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id = mapped_column(String(128), unique=True, nullable=False, index=True)
    camera_id = mapped_column(String(50), nullable=False, index=True)
    model_type = mapped_column(String(30), nullable=False)
    model_hash = mapped_column(String(64), nullable=False)  # SHA256 of model file
    data_hash = mapped_column(String(64), nullable=False)   # SHA256 of baseline manifest
    code_version = mapped_column(String(40))                 # git commit hash
    training_params = mapped_column(Text)                    # JSON string
    calibration_thresholds = mapped_column(Text)             # JSON string
    created_at = mapped_column(DateTime, server_default=func.now())
    is_active = mapped_column(Boolean, default=False)
```

在 model_registry.py:
```python
class ModelRegistry:
    def register(self, model_path, baseline_dir, camera_id, model_type, training_params) -> str:
        """Register a new model version. Returns model_version_id."""

    def get_active(self, camera_id) -> ModelRecord | None:
        """Get the active model for a camera."""

    def activate(self, model_version_id) -> None:
        """Set a model as active (deactivates others for same camera)."""
```

---

#### C4-2: ✅ 训练后自动注册

**修改文件**: `src/argus/anomaly/trainer.py`

训练成功后调用 `registry.register()`。

---

#### C4-3: ✅ Alert 带 model_version_id

**修改文件**: `src/argus/alerts/grader.py`

`Alert` dataclass 加 `model_version_id: str | None = None`，在 pipeline 创建 grader 时传入。

---

#### C4-4: ✅ MLOps 测试

```python
def test_register_and_activate_model():
def test_alert_carries_model_version():
def test_model_hash_changes_on_retrain():
```

---

## Phase D: 高级检测

---

### D1: 开放词汇检测 OVD ✅ 已修改完

#### D1-1: ✅ 创建 OpenVocabClassifier

**新建文件**: `src/argus/anomaly/classifier.py`

使用 ultralytics YOLO-World（已在 `ultralytics>=8.3.0` 中）:

```python
from ultralytics import YOLOWorld

class OpenVocabClassifier:
    def __init__(self, model_name="yolov8s-worldv2.pt", vocabulary=None):
        self._model = YOLOWorld(model_name)
        if vocabulary:
            self._model.set_classes(vocabulary)

    def classify(self, frame, bbox=None) -> tuple[str, float] | None:
        """Classify anomaly region. Returns (label, confidence) or None."""
        if bbox:
            x, y, w, h = bbox
            crop = frame[y:y+h, x:x+w]
        else:
            crop = frame
        results = self._model(crop, verbose=False)
        if results[0].boxes and len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes[0].cls)
            conf = float(results[0].boxes[0].conf)
            label = results[0].names[cls_id]
            return label, conf
        return None
```

**FOE 词表** (可配置):
```python
FOE_VOCAB = [
    "wrench", "bolt", "nut", "screwdriver", "hammer",
    "rag", "glove", "plastic bag", "tape", "wire",
    "insulation", "debris", "paint chip",
    "insect", "bird", "shadow", "reflection",
]
```

---

#### D1-2: ✅ OVD 配置

```python
class ClassifierConfig(BaseModel):
    enabled: bool = False
    model_name: str = "yolov8s-worldv2.pt"
    vocabulary: list[str] = Field(default_factory=lambda: [...])
    min_anomaly_score_to_classify: float = Field(default=0.5)
    high_risk_labels: list[str] = Field(default_factory=lambda: ["wrench", "bolt", ...])
    low_risk_labels: list[str] = Field(default_factory=lambda: ["insect", "shadow", ...])
```

---

#### D1-3: ✅ Pipeline 集成

在 anomaly detection 后、alert grading 前:
```python
if classifier and anomaly_result.anomaly_score >= classifier_config.min_anomaly_score_to_classify:
    classification = classifier.classify(frame, anomaly_bbox)
    # Attach to alert, adjust severity based on risk category
```

---

#### D1-4: ✅ OVD 测试

```python
def test_classify_returns_label_and_confidence():
def test_empty_crop_returns_none():
def test_high_risk_label_escalates():
def test_low_risk_label_suppresses():
```

---

### D2: SAM 2 实例分割 ✅ 已修改完

#### D2-1: ✅ 创建 InstanceSegmenter

**新建文件**: `src/argus/anomaly/segmenter.py`

```python
class InstanceSegmenter:
    """SAM 2 based instance segmentation for anomaly regions."""

    def __init__(self, model_size="small"):
        # Load SAM 2 model
        ...

    def segment(self, frame, prompt_points) -> list[SegmentedObject]:
        """Given anomaly peak points, segment objects."""
        ...
```

#### D2-2: ✅ 配置 + D2-3: Pipeline 集成 + D2-4: 测试

（结构同 D1，细节取决于 SAM 2 的 Python API 和 Jetson 兼容性。）

---

### D3: 合成异常数据 ✅ 已修改完

#### D3-1: ✅ 合成管线脚本

**新建文件**: `scripts/generate_synthetic.py`

```python
"""Generate synthetic anomaly images by compositing FOE objects onto normal baselines.

Usage:
    python scripts/generate_synthetic.py \
        --baseline-dir data/baselines/cam_01/default/v003 \
        --objects-dir data/foe_objects/ \
        --output-dir data/synthetic/ \
        --count 500
"""
```

过程:
1. 加载正常基线帧
2. 加载 FOE 物体照片（白背景已抠图）
3. 随机选择基线帧 + 物体
4. 透视变换 + 尺寸缩放 + 高斯模糊边缘 + 亮度匹配
5. 粘贴到随机位置
6. 保存合成图 + GT mask

---

#### D3-2: ✅ Recall 评估工具

**新建文件**: `src/argus/validation/recall_test.py`

```python
def evaluate_recall(detector, synthetic_dir) -> dict:
    """Run detector on synthetic anomaly images and compute recall."""
    tp, fn = 0, 0
    for img_path, mask_path in load_synthetic_pairs(synthetic_dir):
        result = detector.predict(img)
        if result.anomaly_score >= threshold:
            tp += 1
        else:
            fn += 1
    return {"recall": tp / (tp + fn), "tp": tp, "fn": fn}
```

---

#### D3-3: ✅ 配置 + D3-4: 测试

```python
def test_synthetic_generation_produces_valid_images():
def test_recall_evaluation_runs():
```

---

## 实施顺序总结

```
Week 1:  A1 (CUSUM) + A2 (Conformal) + A3 (Simplex) + A4 (主动BL)
Week 2:  A 阶段测试验证 + 集成测试
Week 3:  B1-1~B1-4 (Dinomaly2 验证+配置+训练+推理)
Week 4:  B1-5~B1-6 (共享模型+测试) + B2 (INT8)
Week 3-4 并行: C1 (健康层) + C2 (漂移)
Week 5:  C3 (跨摄像头) + C4 (MLOps)
Week 6:  D1 (OVD) + D3 (合成数据)
Week 7:  D2 (SAM 2) + 全系统集成测试
```
