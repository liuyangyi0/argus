# Argus 开发路线图

> 基于 v0.1.0 深度代码审计 | 更新于 2026-04-04
>
> 核心原则：**先把「采集 → 训练 → 识别」三步做扎实，再做外围功能。**

---

## 已完成功能统计

| 阶段 | 总任务 | 已完成 | 完成率 | 状态 |
|------|--------|--------|--------|------|
| 阶段 0 — 致命缺陷修复 | 18 | 18 | 100% | DONE |
| 阶段 1 — 基线采集增强 | 8 | 8 | 100% | DONE |
| 阶段 2 — 模型训练增强 | 8 | 8 | 100% | DONE |
| 阶段 3 — 识别增强 | 10 | 10 | 100% | DONE |
| 阶段 3.5 — YOLO 增强 | 6 | 5 | 83% | IN PROGRESS |
| 阶段 3.6 — 异常检测增强 | 7 | 6 | 86% | IN PROGRESS |
| 外围功能（已实现部分） | 2 | 2 | 100% | DONE |
| **合计** | **59** | **57** | **97%** | — |

测试覆盖：**322+ tests passing**

---

## 一、阶段 0 — 致命缺陷修复 ✅ DONE (18/18)

### 0.1 时间戳系统修复 ✅

| 任务 | 改动 | 状态 |
|------|------|------|
| CRIT-03: `AlertGrader.evaluate()` 中 `timestamp` 改用 `time.time()` | `grader.py:175` | ✅ |
| CRIT-04: `_date_folder()` 用传入的 timestamp 参数 | `dispatcher.py:335` | ✅ |
| 全面排查所有 `time.monotonic()` 被当作 epoch 时间的位置 | 全项目 grep | ✅ |

### 0.2 训练流程修复 ✅

| 任务 | 改动 | 状态 |
|------|------|------|
| CRIT-01: 重写 `_train_model_task`，对齐 `ModelTrainer` 的真实 API | `dashboard/routes/baseline.py` | ✅ |
| CRIT-02: 实现真正的 `Engine.export()`，导出 OpenVINO/ONNX | `anomaly/trainer.py` | ✅ |
| HIGH-04: EfficientAD 设 `max_epochs=70`，PatchCore 保持 `1` | `anomaly/trainer.py` | ✅ |
| HIGH-08: 训练完成后检查输出文件存在且 >0 且可加载 | `anomaly/trainer.py` | ✅ |
| HIGH-12: 统一 Anomalib API 参数，锁定版本 | `trainer.py` + `pyproject.toml` | ✅ |

### 0.3 基线采集修复 ✅

| 任务 | 改动 | 状态 |
|------|------|------|
| CRIT-05: 新增 `pipeline.get_raw_frame()` 返回原始帧 | `core/pipeline.py` + `capture/manager.py` | ✅ |
| HIGH-07: `_capture_baseline_task` 使用 `BaselineManager.create_new_version()` | `dashboard/routes/baseline.py` | ✅ |
| 文件命名用 `captured` 计数器而非循环 `i` | 同上 | ✅ |

### 0.4 检测管线修复 ✅

| 任务 | 改动 | 状态 |
|------|------|------|
| HIGH-02: 锁定期间冻结 MOG2 学习率 | `core/pipeline.py` + `prefilter/mog2.py` | ✅ |
| HIGH-03: `_locked` 读写用 `threading.Lock` 保护 | `core/pipeline.py` | ✅ |
| HIGH-05: Zone mask 后保存原始帧副本供采集使用 | `core/pipeline.py` | ✅ |
| HIGH-06: `_evaluate_zones` 遍历所有 zone，返回最高优先级的告警 | `core/pipeline.py` | ✅ |

### 0.5 分发线程修复 ✅

| 任务 | 改动 | 状态 |
|------|------|------|
| HIGH-01: `while True` 内部 `try/except` + 指数退避重试 | `alerts/dispatcher.py` | ✅ |
| HIGH-13: 关闭时 drain 队列，等待清空或超时 | `alerts/dispatcher.py` | ✅ |
| HIGH-11: baseline `current.txt` 原子写入 | `anomaly/baseline.py` | ✅ |
| HIGH-09: 重连移到独立线程或设超时上限 | `capture/camera.py` | ✅ |
| HIGH-10: SSIM Sigmoid 参数标定 | `anomaly/detector.py` | ✅ |

---

## 二、阶段 1 — 基线采集增强 ✅ DONE (8/8)

实现文件：`frame_filter.py` + `baseline.py`

| ID | 功能 | 状态 |
|----|------|------|
| CAP-002 | 模糊帧过滤（Laplacian 方差法） | ✅ |
| CAP-003 | 有人帧过滤（复用 YOLOPersonDetector） | ✅ |
| CAP-004 | 过曝/过暗过滤 | ✅ |
| CAP-005 | 帧间去重（SSIM >= 0.98） | ✅ |
| CAP-006 | 采集统计报告 | ✅ |
| CAP-007 | 单张删除 | ✅ |
| CAP-008 | 多时段采集提示 | ✅ |
| CAP-009 | 编码器错误帧检测（信息熵 < 3.0） | ✅ |

---

## 三、阶段 2 — 模型训练增强 ✅ DONE (8/8)

实现文件：`quality.py` + `model_compare.py` + `trainer.py` + `database.py`

| ID | 功能 | 状态 |
|----|------|------|
| TRN-001 | 训练前验证（数量/损坏率/重复率/亮度） | ✅ |
| TRN-002 | 自动切分验证集（80/20, seed=42） | ✅ |
| TRN-003 | 训练后自动验证（验证集推理 + 分数分布） | ✅ |
| TRN-004 | 质量报告（A/B/C/F 等级） | ✅ |
| TRN-005 | 阈值推荐 | ✅ |
| TRN-006 | 输出验证（checkpoint 完整性） | ✅ |
| TRN-007 | 训练历史记录（TrainingRecord ORM） | ✅ |
| TRN-008 | 模型 A/B 对比（model_compare.py） | ✅ |

---

## 四、阶段 3 — 识别增强 ✅ DONE (10/10)

| ID | 功能 | 状态 |
|----|------|------|
| DET-003 | 检测调试视图（FrameDiagnostics） | ✅ |
| DET-004 | SSIM/模型状态显示 | ✅ |
| DET-005 | 灵敏度调节预览 | ✅ |
| DET-006 | 检测暂停/恢复（三模式） | ✅ |
| DET-007 | 告警快照标注（红色矩形框 + 分数标签） | ✅ |
| DET-008 | 帧级日志面板 | ✅ |
| DET-009 | 分发断路器（三态） | ✅ |
| DET-010 | 学习模式 | ✅ |
| DET-011 | 人员遮罩改进（高斯模糊） | ✅ |
| DET-012 | YOLO 模型共享 | ✅ |

---

## 五、阶段 3.5 — YOLO 增强 (5/6)

| ID | 功能 | 状态 |
|----|------|------|
| YOLO-001 | Multi-class COCO 检测 | ✅ |
| YOLO-002 | 目标追踪（BoT-SORT） | ✅ |
| YOLO-003 | 重构为 YOLOObjectDetector | ✅ |
| YOLO-004 | YOLO+Anomalib 混合管线 | ✅ |
| YOLO-005 | 语义告警上下文 | ✅ |
| YOLO-006 | 自定义 YOLO 训练基础设施 | 待定（需标注数据） |

---

## 六、阶段 3.6 — 异常检测增强 (6/7)

| ID | 功能 | 实现文件 | 状态 |
|----|------|----------|------|
| ANO-001 | 图像质量评估 | `frame_quality.py` | ✅ |
| ANO-002 | 时序异常追踪 | `temporal_tracker.py` | ✅ |
| ANO-003 | 自适应阈值 | `adaptive_threshold.py` | ✅ |
| ANO-004 | 模型多样性（FastFlow/Padim） | anomaly module | ✅ |
| ANO-005 | 模型集成 | anomaly module | ✅ |
| ANO-006 | 异常图后处理 | `anomaly_postprocess.py` | ✅ |
| ANO-007 | INT8 量化 | — | 待实现 |

---

## 七、外围功能

### 已实现 ✅

| 功能 | 说明 |
|------|------|
| 告警工作流状态机 | NEW -> ACKNOWLEDGED -> INVESTIGATING -> RESOLVED -> CLOSED |
| 浏览器音频告警 | Web Audio API, 3 个严重等级 |

### 待实现

| 优先级 | 领域 | 关键功能 |
|--------|------|----------|
| P1 | 用户权限 | 多用户 RBAC、登录页、会话管理 |
| P1 | 审计合规 | 操作审计表、告警证据链 |
| P2 | 报表统计 | 日报/周报、PDF 导出、误报率趋势 |
| P2 | 摄像头诊断 | 图像质量评分、偏移检测、遮挡检测 |
| P2 | 备份恢复 | SQLite 自动备份、配置回滚 |
| P2 | DCS 集成 | Modbus TCP、OPC-UA |
| P3 | 多节点管理 | 中心服务器、告警汇聚、模型分发 |
| P3 | 软件授权 | License Key、功能分级 |
| P3 | 性能监控 | Prometheus、资源面板 |
| P3 | 国际化 | i18n 框架、英文翻译 |

---

## 八、实施进度

```
阶段 0 — 致命缺陷修复        ██████████ 100% ✅
阶段 1 — 基线采集增强        ██████████ 100% ✅
阶段 2 — 模型训练增强        ██████████ 100% ✅
阶段 3 — 识别增强            ██████████ 100% ✅
阶段 3.5 — YOLO 增强         ████████░░  83%
阶段 3.6 — 异常检测增强      ████████░░  86%
外围功能                      █░░░░░░░░░  ~10%
```

### 依赖关系

```
阶段 0 (缺陷修复) ✅
     │
     ├──→ 阶段 1 (采集增强) ✅
     │         │
     │         ▼
     └──→ 阶段 2 (训练增强) ✅ ──→ 阶段 3 (识别增强) ✅
                                        │
                              ┌─────────┼─────────┐
                              ▼         ▼         ▼
                        阶段 3.5    阶段 3.6    外围功能
                       (YOLO增强)  (异常增强)
```

---

## 九、剩余工作

| ID | 功能 | 阻塞原因 | 优先级 |
|----|------|----------|--------|
| YOLO-006 | 自定义 YOLO 训练基础设施 | 需要现场标注数据 | P1 |
| ANO-007 | INT8 量化 | 待排期 | P2 |

---

## 十、关键测试标准

| 阶段 | 验证标准 | 状态 |
|------|----------|------|
| 阶段 0 | `pytest` 全通过 + Dashboard 完整流程不崩溃 | ✅ |
| 阶段 1 | 采集 200 帧，保留率 50-80%，无模糊/有人/重复帧 | ✅ |
| 阶段 2 | 训练完有质量报告 A/B 级，推荐阈值合理 | ✅ |
| 阶段 3 | 调试视图可看到每帧处理结果，灵敏度调节有即时反馈 | ✅ |
| 全局 | 322+ tests passing | ✅ |
