# Argus 商业化路线图

> 基于 v0.1.0 深度代码审计 | 2026-04-04
>
> 核心原则：**先把「采集 → 训练 → 识别」三步做扎实，再做外围功能。**

---

## 〇、问题总览

### 致命级（不修就不能用）

| ID | 问题 | 位置 | 影响 |
|----|------|------|------|
| CRIT-01 | Dashboard 训练函数传错参数，运行时 TypeError 崩溃 | `dashboard/routes/baseline.py:510-514` | 网页训练功能完全不可用 |
| CRIT-02 | trainer.py 导出逻辑是空壳，`Engine.export()` 从未调用 | `anomaly/trainer.py:136-143` | 选了导出格式实际什么都没导出 |
| CRIT-03 | `alert.timestamp` 用 `time.monotonic()` 而非 UNIX 时间戳 | `alerts/grader.py:175` + `alerts/dispatcher.py:97` | 数据库/Webhook/邮件中所有时间戳完全错误 |
| CRIT-04 | `_date_folder()` 忽略传入 timestamp | `alerts/dispatcher.py:335` | 告警快照存入错误日期目录 |
| CRIT-05 | 基线采集存的是掩码后的帧（排除区域被涂黑） | `dashboard/routes/baseline.py:476` | 训练数据含人工伪影，模型学到错误特征 |

### 严重级（核心逻辑错误）

| ID | 问题 | 位置 | 影响 |
|----|------|------|------|
| HIGH-01 | Email/Webhook 线程遇异常永久死亡，不恢复 | `alerts/dispatcher.py:157,190` | 第一次网络故障后告警通道永久失效 |
| HIGH-02 | 锁定期间 MOG2 仍在学习（learningRate 未冻结） | `core/pipeline.py:227-229` | 异常物体被 MOG2 吸收进背景，锁定机制失效 |
| HIGH-03 | `_locked` 字段无线程同步保护 | `core/pipeline.py:142-145` | 操作员清除锁定时与检测线程竞态 |
| HIGH-04 | EfficientAD 只训练 1 epoch（需 70-100） | `anomaly/trainer.py:194-197` | 选 EfficientAD 等于没训练 |
| HIGH-05 | Zone mask 在 MOG2 前施加 | `core/pipeline.py:215` | 热更新区域后 MOG2 需 100 秒重新收敛 |
| HIGH-06 | 多区域评估只返回第一个告警 | `core/pipeline.py:359-371` | 可能跳过 CRITICAL 区域的高优先级告警 |
| HIGH-07 | 基线采集绕过 BaselineManager 版本管理 | `dashboard/routes/baseline.py:471` | 多次采集覆盖旧数据，无法回滚 |
| HIGH-08 | 训练器不验证输出文件是否生成 | `anomaly/trainer.py:145-161` | 静默训练失败，状态显示"完成" |
| HIGH-09 | 摄像头重连阻塞整条管线线程 | `capture/camera.py:157-192` | 网络抖动时检测中断最长 60 秒 |
| HIGH-10 | SSIM Sigmoid 参数未标定 | `anomaly/detector.py:257-258` | SSIM 回退模式分数不可靠 |
| HIGH-11 | baseline `current.txt` 非原子写入 | `anomaly/baseline.py:75-79` | 并发训练可损坏版本文件 |
| HIGH-12 | Anomalib API 参数与版本不一致 | `trainer.py` vs `train_model.py` | `layers_to_extract` vs `layers`，必有一处运行时报错 |
| HIGH-13 | daemon 线程退出时静默丢弃队列中的告警 | `alerts/dispatcher.py:57-62` | 核安全场景不可接受的数据丢失 |

---

## 一、阶段 0 — 致命缺陷修复（让核心流程能跑通）

### 0.1 时间戳系统修复

| 任务 | 改动 | 复杂度 |
|------|------|--------|
| CRIT-03: `AlertGrader.evaluate()` 中 `timestamp` 改用 `time.time()` | `grader.py:175` | S |
| CRIT-04: `_date_folder()` 用传入的 timestamp 参数 | `dispatcher.py:335` | S |
| 全面排查所有 `time.monotonic()` 被当作 epoch 时间的位置 | 全项目 grep | S |

### 0.2 训练流程修复

| 任务 | 改动 | 复杂度 |
|------|------|--------|
| CRIT-01: 重写 `_train_model_task`，对齐 `ModelTrainer` 的真实 API | `dashboard/routes/baseline.py:492-528` | M |
| CRIT-02: 实现真正的 `Engine.export()`，导出 OpenVINO/ONNX | `anomaly/trainer.py:136-143` | M |
| HIGH-04: EfficientAD 设 `max_epochs=70`，PatchCore 保持 `1` | `anomaly/trainer.py:194-197` | S |
| HIGH-08: 训练完成后检查输出文件存在且 >0 且可加载 | `anomaly/trainer.py` 末尾 | S |
| HIGH-12: 统一 Anomalib API 参数，锁定版本 `>=2.0.0,<3.0.0` | `trainer.py` + `pyproject.toml` | M |

### 0.3 基线采集修复

| 任务 | 改动 | 复杂度 |
|------|------|--------|
| CRIT-05: 新增 `pipeline.get_raw_frame()` 返回原始帧 | `core/pipeline.py` + `capture/manager.py` | S |
| HIGH-07: `_capture_baseline_task` 使用 `BaselineManager.create_new_version()` | `dashboard/routes/baseline.py:467-489` | S |
| 文件命名用 `captured` 计数器而非循环 `i` | 同上 | S |

### 0.4 检测管线修复

| 任务 | 改动 | 复杂度 |
|------|------|--------|
| HIGH-02: 锁定期间 `_prefilter.process(frame, learning_rate_override=0.0)` | `core/pipeline.py:227-229` + `prefilter/mog2.py` | M |
| HIGH-03: `_locked` 读写用 `threading.Lock` 保护 | `core/pipeline.py` | S |
| HIGH-05: Zone mask 后保存原始帧副本供采集使用（不改 MOG2 顺序，但在锁定期间冻结 MOG2 学习率来缓解） | `core/pipeline.py` | M |
| HIGH-06: `_evaluate_zones` 遍历所有 zone，返回最高优先级的告警 | `core/pipeline.py:359-371` | S |

### 0.5 分发线程修复

| 任务 | 改动 | 复杂度 |
|------|------|--------|
| HIGH-01: `while True` 内部 `try/except Exception` + 指数退避重试，不 break | `alerts/dispatcher.py:157,190` | M |
| HIGH-13: 关闭时 drain 队列：等待队列清空或超时（5 秒），再退出 | `alerts/dispatcher.py` | M |
| HIGH-11: baseline `current.txt` 原子写入（临时文件 + rename） | `anomaly/baseline.py:75-79` | S |
| HIGH-09: 重连移到独立线程或设超时上限 | `capture/camera.py` | L |

---

## 二、阶段 1 — 基线采集增强（让采集数据质量可靠）

| ID | 功能 | 方案 | 复杂度 |
|----|------|------|--------|
| CAP-002 | 模糊帧过滤 | Laplacian 方差法，默认阈值 100，自适应（中位数 30%） | S |
| CAP-003 | 有人帧过滤 | 复用 `YOLOPersonDetector`，confidence 降至 0.3 | S |
| CAP-004 | 过曝/过暗过滤 | 灰度均值 30-225 + 饱和像素占比 <30% + std >10 | S |
| CAP-005 | 帧间去重 | SSIM 相似度 >= 0.98 则跳过，256x256 缩小后计算 | S |
| CAP-006 | 采集统计报告 | 报告采集/过滤/保留帧数、过滤原因分布、亮度范围 | S |
| CAP-007 | 单张删除 | 缩略图页增加 `hx-delete` 按钮 | S |
| CAP-008 | 多时段采集提示 | 采集会话标签（白天/夜间/检修），覆盖性检查提示 | S |
| CAP-009 | 编码器错误帧检测 | 信息熵 < 3.0 则丢弃（全灰/全绿/花屏帧） | S |

---

## 三、阶段 2 — 模型训练增强（让训练结果可评估）

| ID | 功能 | 方案 | 复杂度 |
|----|------|------|--------|
| TRN-001 | 训练前验证 | 数量≥30、损坏率<10%、近似重复率<80%、亮度 std>2 | M |
| TRN-002 | 自动切分验证集 | 80/20 切分，固定 seed=42，物理复制到 train/val 目录 | M |
| TRN-003 | 训练后自动验证 | 验证集跑推理，收集分数分布（均值/std/max/P95） | M |
| TRN-004 | 质量报告 | 质量等级 A/B/C/F + 分数分布 + 建议 | M |
| TRN-005 | 阈值推荐 | `threshold = max(mean + 2.5*std, max_score * 1.05)`，上限 0.95 | M |
| TRN-006 | 输出验证 | checkpoint 存在/可加载 + 导出文件完整 + 冒烟推理测试 | S |
| TRN-007 | 训练历史记录 | 新增 `TrainingRecord` ORM 模型，记录每次训练参数和结果 | M |
| TRN-008 | 模型 A/B 对比 | 新旧模型在同一验证集上跑推理，对比分数分布和延迟 | L |

---

## 四、阶段 3 — 识别增强（让检测可调试可运维）

| ID | 功能 | 方案 | 复杂度 |
|----|------|------|--------|
| DET-003 | 检测调试视图 | 每 stage 中间结果缓存到 `FrameDiagnostics`，前端分步显示 | L |
| DET-004 | SSIM/模型状态显示 | `AnomalibDetector.get_status()` 返回当前模式/模型路径/校准进度 | S |
| DET-005 | 灵敏度调节预览 | 缓存最近 300 帧分数，新阈值下重新评估，返回预计告警数 | M |
| DET-006 | 检测暂停/恢复 | 三模式：ACTIVE / MAINTENANCE（冻结 MOG2）/ LEARNING（不告警） | M |
| DET-007 | 告警快照标注 | `cv2.findContours` 提取异常区域 → 红色矩形框 + 分数标签 | S |
| DET-008 | 帧级日志面板 | `deque(maxlen=1500)` 环形缓冲，记录每帧处理结果 | M |
| DET-009 | 分发断路器 | 三态断路器（closed/open/half-open），open 时持久化到 fallback 文件 | M |
| DET-010 | 学习模式 | 首次启动自动进入，持续 `max(history/fps*3, 600)` 秒，期间不告警 | M |
| DET-011 | 人员遮罩改进 | 纯黑填充改为高斯模糊 | S |
| DET-012 | YOLO 模型共享 | 多管线共享单个 YOLO 实例 | M |

---

## 五、外围功能（核心做好后再做）

| 优先级 | 领域 | 关键功能 |
|--------|------|----------|
| P1 | 用户权限 | 多用户 RBAC、登录页、会话管理 |
| P1 | 审计合规 | 操作审计表、告警证据链 |
| P1 | 告警工作流 | 生命周期状态机、超时升级、交接班 |
| P1 | 声光提醒 | 浏览器音频告警、标题栏闪烁 |
| P2 | 报表统计 | 日报/周报、PDF 导出、误报率趋势 |
| P2 | 摄像头诊断 | 图像质量评分、偏移检测、遮挡检测 |
| P2 | 备份恢复 | SQLite 自动备份、配置回滚 |
| P2 | DCS 集成 | Modbus TCP、OPC-UA |
| P3 | 多节点管理 | 中心服务器、告警汇聚、模型分发 |
| P3 | 软件授权 | License Key、功能分级 |
| P3 | 性能监控 | Prometheus、资源面板 |
| P3 | 国际化 | i18n 框架、英文翻译 |

---

## 六、实施计划

```
阶段 0 — 致命缺陷修复（1.5 周）
  ├── 0.1 时间戳修复          [2h]
  ├── 0.2 训练流程修复         [3d]
  ├── 0.3 基线采集修复         [1d]
  ├── 0.4 检测管线修复         [2d]
  └── 0.5 分发线程修复         [2d]

阶段 1 — 基线采集增强（1 周）
  └── CAP-002~009             [5d]

阶段 2 — 模型训练增强（2 周）
  └── TRN-001~008             [8d]

阶段 3 — 识别增强（2 周）
  └── DET-003~012             [8d]

阶段 4+ — 外围功能（按需）
  └── 用户权限 → 审计 → 告警工作流 → ...
```

### 依赖关系

```
CRIT-01~05 ──→ 阶段 1 (采集增强)
     │              │
     │              ▼
     └──────→ 阶段 2 (训练增强) ──→ 阶段 3 (识别增强)
                                        │
                                        ▼
                                   阶段 4+ (外围)
```

### 关键测试

| 阶段 | 验证标准 |
|------|----------|
| 阶段 0 | `pytest` 全通过 + Dashboard 上跑一次完整流程不崩溃 |
| 阶段 1 | 采集 200 帧，保留率 50-80%，无模糊/有人/重复帧 |
| 阶段 2 | 训练完有质量报告 A/B 级，推荐阈值合理（正常帧分数远低于阈值） |
| 阶段 3 | 调试视图可看到每帧处理结果，灵敏度调节有即时反馈 |
