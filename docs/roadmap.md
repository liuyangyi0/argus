# Argus 商业化路线图

> 基于 v0.1.0 深度代码审计 | 2026-04-04 | 更新于 2026-04-04
>
> 核心原则：**先把「采集 → 训练 → 识别」三步做扎实，再做外围功能。**
>
> 当前进度：**阶段 3 已完成** ✅ | 阶段 0 残留 bug 待修复

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

## 四、阶段 3 — 识别增强（让检测可调试可运维）✅ 已完成

| ID | 功能 | 方案 | 状态 |
|----|------|------|------|
| DET-003 | 检测调试视图 | `FrameDiagnostics` + `/detection` 页面 + 帧级日志表 | ✅ |
| DET-004 | SSIM/模型状态显示 | `DetectorStatus` dataclass + `get_status()` + Dashboard 面板 | ✅ |
| DET-005 | 灵敏度调节预览 | 300 帧分数缓存 + `evaluate_threshold()` + 分布直方图 | ✅ |
| DET-006 | 检测暂停/恢复 | `PipelineMode` (ACTIVE/MAINTENANCE/LEARNING) + 模式选择器 | ✅ |
| DET-007 | 告警快照标注 | `cv2.findContours` → 红色矩形框 + 分数标签 | ✅ |
| DET-008 | 帧级日志面板 | `DiagnosticsBuffer` deque(maxlen=1500) + 阶段耗时可视化 | ✅ |
| DET-009 | 分发断路器 | `CircuitBreaker` 三态机 + JSON 持久化 | ✅ |
| DET-010 | 学习模式 | 首次启动自动 LEARNING，持续 max(history/fps*3, 600)s | ✅ |
| DET-011 | 人员遮罩改进 | 纯黑填充 → 高斯模糊 | ✅ |
| DET-012 | YOLO 模型共享 | `get_shared_yolo()` 注册表 + `shared_model` 参数 | ✅ |

---

## 五、阶段 3.5 — YOLO 多类别检测增强

> YOLO11n 当前仅用于人员过滤（COCO class 0）。扩展为多类别检测器 + 追踪，
> 构建 YOLO + Anomalib 双流混合管线，可减少 20-40% 误报。

| ID | 功能 | 方案 | 复杂度 |
|----|------|------|--------|
| YOLO-001 | 多类别 COCO 检测 | 扩展 `classes=[0]` → `[0,14,16,17,26,30,34,39]`（人/鸟/猫/狗/背包/手提包/行李箱/瓶子），配置化 `classes_to_detect` | S |
| YOLO-002 | 目标追踪 | `model.track(persist=True)` 启用 BoT-SORT，添加 `track_id` 到检测结果，消除闪烁告警 | S |
| YOLO-003 | 重构为通用对象检测器 | `YOLOPersonDetector` → `YOLOObjectDetector`，`PersonDetection` → `ObjectDetection` (+ class_id, class_name, track_id) | M |
| YOLO-004 | YOLO+Anomalib 混合管线 | YOLO 高置信度检测 → 即时语义告警；Anomalib → 未知异常兜底；双流独立评分 | M |
| YOLO-005 | 告警语义化 | Alert 增加 `detection_type: Literal["object","anomaly","hybrid"]` + `object_class`，Dashboard 显示物体类别图标 | S |
| YOLO-006 | 自定义 YOLO 训练基础设施 | Dashboard FOE 标注界面 + `scripts/train_yolo_foe.py` + 模型版本管理 | L |

---

## 六、阶段 3.6 — 异常检测增强

> 提升检测质量、降低误报、加快推理速度。

| ID | 功能 | 方案 | 复杂度 |
|----|------|------|--------|
| ANO-001 | 图像质量评估 | Laplacian 模糊检测 + 灰度直方图曝光检测 + Shannon 信息熵（<3.0丢弃），低质量帧降低置信度 | S |
| ANO-002 | 时序异常追踪 | 跟踪异常区域质心跨帧持续性，同位置持续 N 帧 → 升级严重性，移动 → 降低置信度 | M |
| ANO-003 | 自适应阈值 | 300 帧滑动窗口 EWMA/P95 自动调节阈值，范围钳制 [base±0.1]，防止漂移 | S |
| ANO-004 | 模型多样性 | 新增 FastFlow（实时推理）、Padim（轻量边缘部署），trainer.py + schema.py 扩展 | M |
| ANO-005 | 模型集成 | 同时运行 2-3 个异常模型，投票/均值/加权融合分数，降低误报 30-50% | M |
| ANO-006 | 异常图后处理 | 形态学开闭运算 + 轮廓面积过滤（<100px 噪声移除），锐化热力图边界 | S |
| ANO-007 | OpenVINO INT8 量化 | 训练后 INT8 量化导出，推理加速 2-3x，精度损失 <2% 验证 | M |

---

## 七、外围功能（核心做好后再做）

| 优先级 | 领域 | 关键功能 |
|--------|------|----------|
| P1 | 安全加固 | TLS/HTTPS（Traefik 反代）、SQLite 加密（sqlcipher）、审计日志表 |
| P1 | 用户权限 | JWT 认证、多用户 RBAC（admin/operator/viewer）、会话管理 |
| P1 | 告警工作流 | 生命周期状态机（new→assigned→resolved→closed）、超时升级、交接班 |
| P1 | 声光提醒 | 浏览器音频告警、标题栏闪烁、WebSocket 实时推送替代 HTMX 轮询 |
| P1 | 可观测性 | Prometheus `/metrics` 端点、Grafana 模板、告警率/延迟/连接状态指标 |
| P2 | 报表统计 | 日报/周报、PDF 导出、误报率趋势 |
| P2 | 摄像头诊断 | 图像质量评分、偏移检测、遮挡检测 |
| P2 | 备份恢复 | SQLite 自动备份、配置回滚、S3/MinIO 对象存储 |
| P2 | DCS 集成 | Syslog 转发（快速）→ Modbus TCP → OPC-UA → MQTT |
| P2 | Dashboard UX | 暗色/亮色主题、移动端响应式、键盘快捷键、自定义 Widget 布局 |
| P3 | 多节点管理 | 中心服务器、告警汇聚、模型分发、PostgreSQL 迁移 |
| P3 | 高可用 | Kubernetes Helm Chart、主被动切换、状态持久化检查点 |
| P3 | 性能优化 | GPU 推理路径（CUDA/TensorRT）、批量推理、GStreamer 硬件解码 |
| P3 | 软件授权 | License Key、功能分级 |
| P3 | 国际化 | i18n 框架、英文翻译 |

---

## 八、实施计划

```
阶段 0 — 致命缺陷修复（1.5 周）        ← 当前最高优先级
  ├── 0.1 时间戳修复          [2h]
  ├── 0.2 训练流程修复         [3d]
  ├── 0.3 基线采集修复         [1d]
  ├── 0.4 检测管线修复         [2d]
  └── 0.5 分发线程修复         [2d]

阶段 1 — 基线采集增强（1 周）
  └── CAP-002~009             [5d]

阶段 2 — 模型训练增强（2 周）
  └── TRN-001~008             [8d]

阶段 3 — 识别增强（2 周）             ✅ 已完成
  └── DET-003~012             [done]

阶段 3.5 — YOLO 多类别增强（1 周）
  ├── YOLO-001~003 (重构+多类别+追踪)  [3d]
  └── YOLO-004~005 (混合管线+语义化)   [2d]

阶段 3.6 — 异常检测增强（1.5 周）
  ├── ANO-001~003 (质量+时序+自适应)   [4d]
  └── ANO-006~007 (后处理+量化)        [3d]

阶段 4+ — 外围功能（按需）
  └── 安全加固 → 可观测性 → DCS 集成 → ...
```

### 依赖关系

```
阶段 0 (bug修复) ──→ 阶段 1 (采集增强) ──→ 阶段 2 (训练增强)
       │                                          │
       └──→ 阶段 3.5 (YOLO增强)                   │
                │                                  ▼
                └──→ 阶段 3.6 (异常检测增强) ──→ 阶段 4+ (外围)
```

### 关键测试

| 阶段 | 验证标准 |
|------|----------|
| 阶段 0 | `pytest` 全通过 + Dashboard 上跑一次完整流程不崩溃 |
| 阶段 1 | 采集 200 帧，保留率 50-80%，无模糊/有人/重复帧 |
| 阶段 2 | 训练完有质量报告 A/B 级，推荐阈值合理（正常帧分数远低于阈值） |
| 阶段 3 | ✅ 调试视图可看到每帧处理结果，灵敏度调节有即时反馈 |
| 阶段 3.5 | YOLO 检测到工具/动物等物体时显示类别标签，追踪 ID 跨帧稳定 |
| 阶段 3.6 | 模糊帧自动降低置信度，同位置异常持续 5 帧后严重性升级 |
