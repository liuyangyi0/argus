# Argus 系统架构文档

> 版本 0.2.0 | 最后更新: 2026-04-05

## 1. 项目简介

Argus（阿耳戈斯）是面向核电站关键区域的异物检测（FOE）视觉系统。通过固定摄像头 + 深度学习异常检测，实现 7x24 小时不间断监控，自动发现并分级报警墙皮脱落、保温层掉落、遗留工具等异常物体。

**部署模式：** 边缘节点（Intel NUC / NVIDIA Jetson），每节点 2-4 路摄像头，核电站内网环境。

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Argus 边缘节点                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ 摄像头 1  │   │ 摄像头 2  │   │ 摄像头 3  │   │ 摄像头 4  │        │
│  │ (RTSP)   │   │ (RTSP)   │   │ (USB)    │   │ (File)   │        │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘        │
│       │              │              │              │               │
│       ▼              ▼              ▼              ▼               │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              CameraManager（线程池）                      │       │
│  │  每个摄像头一个独立 DetectionPipeline 线程                │       │
│  └─────────────────────────┬───────────────────────────────┘       │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              DetectionPipeline（每摄像头）                │       │
│  │                                                         │       │
│  │  Stage 0: ZoneMaskEngine（区域掩码）                     │       │
│  │       ↓                                                 │       │
│  │  Stage 1: MOG2PreFilter（背景减除 + 心跳旁路）            │       │
│  │       ↓ (有变化 or 心跳 or 锁定)                         │       │
│  │  Stage 2: YOLOObjectDetector（人员遮罩 + 多类目标检测）   │       │
│  │       ↓ (无人 or 已遮罩)                                │       │
│  │  Stage 3: Anomalib + Simplex 双通道并行                  │       │
│  │       ├── Anomalib 深度异常检测                          │       │
│  │       └── Simplex 安全通道（帧差 + 静态参考）             │       │
│  │       ↓ (双通道融合)                                    │       │
│  │  Stage 4: OVD 分类（YOLO-World 开放词汇检测，可选）       │       │
│  │       ↓                                                 │       │
│  │  Stage 5: SAM 2 实例分割（可选）                         │       │
│  │       ↓                                                 │       │
│  │  Stage 5.5: AnomalyPostprocess + TemporalTracker        │       │
│  │       ↓                                                 │       │
│  │  AlertGrader（分级 + CUSUM 时间确认 + 去重抑制）          │       │
│  └─────────────────────────┬───────────────────────────────┘       │
│                            │                                       │
│                            ├───→ KS DriftDetector（漂移监控）       │
│                            │     异常分数 → 参考/检验窗口对比       │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │         CrossCameraCorrelator（跨摄像头关联）             │       │
│  │         多摄像头异常事件时空关联 + 置信度提升              │       │
│  └─────────────────────────┬───────────────────────────────┘       │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              AlertDispatcher（多通道分发 + 断路器）       │       │
│  │                                                         │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │       │
│  │  │ SQLite   │  │ Webhook  │  │  邮件    │              │       │
│  │  │ 数据库   │  │ (后台线程) │  │(后台线程) │              │       │
│  │  └──────────┘  └──────────┘  └──────────┘              │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │   FastAPI 后端 + Vue 3 + Ant Design Vue SPA + WebSocket  │       │
│  │                                                         │       │
│  │  总览 | 摄像头 | 摄像头详情 | 告警中心 | 模型管理        │       │
│  │  系统设置 | 用户管理 | 审计日志 | 报表 | 备份             │       │
│  │                                                         │       │
│  │  + MJPEG 实时视频流                                     │       │
│  │  + WebSocket 实时告警推送                                │       │
│  │  + 后台任务管理（基线采集/模型训练）                      │       │
│  │  + RBAC 用户权限 + 审计日志 + 限流 + 安全头              │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────┐              │
│  │TaskScheduler│  │HealthMonitor │  │ TaskManager   │              │
│  │(定时维护)    │  │(健康监控)     │  │(后台任务队列)  │              │
│  └────────────┘  └──────────────┘  └───────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 线程模型

```
主线程 ─── main loop（健康检查 1s 轮询）
  │
  ├── 摄像头线程 1 ─── DetectionPipeline.run_once() 循环
  ├── 摄像头线程 2 ─── DetectionPipeline.run_once() 循环
  ├── ...
  ├── Dashboard 线程 ─── Uvicorn (FastAPI)
  ├── WebSocket 线程 ─── 实时推送（告警/状态/帧数据）
  ├── Scheduler 线程 ─── APScheduler（定时清理/磁盘检查）
  ├── Webhook 线程 ─── 后台 HTTP POST 队列（含断路器）
  └── Email 线程 ─── 后台 SMTP 发送队列
```

### 2.3 数据流

```
摄像头帧 → 区域掩码 → MOG2 判断变化？
                         │
                    ┌────┤
                    │    │
               无变化    有变化 / 心跳 / 锁定
                │        │
              跳过帧     YOLO 多类目标检测（人员遮罩 + 语义上下文）
                         │
                    ┌────┤
                    │    │
               有人跳帧  无人/已遮罩
                │        │
              跳过帧     ┌── Anomalib 深度异常检测（Dinomaly2/PatchCore/EfficientAD/...）
                         │   + 可选模型集成（ensemble）
                         ├── Simplex 安全通道（帧差 + 静态参考）
                         └── 双通道融合 → 异常分数
                              │
                              ├───→ DriftDetector（KS 检验漂移监控）
                              │
                              ▼
                         OVD 分类（YOLO-World，可选：对异常区域分类）
                              │
                              ▼
                         SAM 2 实例分割（可选：异常区域精细分割）
                              │
                              ▼
                         异常图后处理 + 时序追踪
                              │
                    ┌─────────┤
                    │         │
               正常           异常 (score > threshold)
                │             │
              更新锁定        AlertGrader 评估（自适应阈值）
                状态           │
                    ┌─────────┤
                    │         │
               未达确认        达到 CUSUM 确认条件
               阈值           (evidence > threshold + IoU > 0.3)
                │              │
              等待下帧         CrossCameraCorrelator 跨摄像头关联校验
                               │
                    ┌──────────┤
                    │          │
               抑制窗口内      窗口外
                │              │
              丢弃            生成 Alert → 分发（DB + Webhook + 邮件 + WebSocket）
```

---

## 3. 包结构

```
src/argus/
├── __init__.py
├── __main__.py              # CLI 入口，子系统编排
│
├── config/                  # 配置系统
│   ├── schema.py            # Pydantic 配置模型（全参数验证）
│   └── loader.py            # YAML 加载/保存（原子写入）
│
├── capture/                 # 视频采集
│   ├── camera.py            # 单摄像头驱动（RTSP/USB/文件，自动重连）
│   ├── manager.py           # 多摄像头线程池
│   ├── quality.py           # 采集质量检查
│   └── frame_filter.py      # 帧过滤（模糊/过曝/重复/编码错误）
│
├── prefilter/               # 预筛
│   ├── mog2.py              # MOG2 背景减除 + 相位相关防抖 + 辐射去噪
│   └── simple_detector.py   # Simplex 安全通道（帧差 + 静态参考检测）
│
├── person/                  # 目标检测
│   └── detector.py          # YOLO 多类目标检测 + BoT-SORT 追踪
│
├── anomaly/                 # 异常检测
│   ├── detector.py          # Anomalib 推理 + 多尺度 + SSIM 回退
│   ├── baseline.py          # 基线版本管理
│   ├── trainer.py           # 模型训练编排（含验证/质量评估/导出）
│   ├── quality.py           # 训练数据质量验证
│   ├── ensemble.py          # 多模型集成
│   ├── model_compare.py     # 模型 A/B 对比
│   ├── classifier.py        # YOLO-World OVD 开放词汇分类
│   ├── segmenter.py         # SAM 2 实例分割
│   └── drift.py             # KS 漂移监控（参考/检验窗口）
│
├── core/                    # 核心编排
│   ├── pipeline.py          # 多阶段检测管线（三模式：ACTIVE/MAINTENANCE/LEARNING）
│   ├── zone_mask.py         # 多边形区域掩码引擎
│   ├── health.py            # 系统健康监控
│   ├── scheduler.py         # 定时维护任务（APScheduler）
│   ├── diagnostics.py       # 帧级检测诊断
│   ├── temporal_tracker.py  # 时序异常追踪
│   ├── adaptive_threshold.py # 自适应阈值
│   ├── anomaly_postprocess.py # 异常图后处理
│   ├── frame_quality.py     # 运行时图像质量评估
│   ├── circuit_breaker.py   # 分发断路器（三态）
│   └── correlation.py       # 跨摄像头异常关联
│
├── alerts/                  # 告警系统
│   ├── grader.py            # 分级 + CUSUM 时间确认 + 去重抑制
│   ├── dispatcher.py        # 多通道分发（DB/Webhook/邮件）+ 断路器
│   ├── feedback.py          # 误报反馈闭环
│   └── calibration.py       # Conformal prediction 校准
│
├── storage/                 # 持久化
│   ├── database.py          # SQLite + WAL（告警/训练/用户/模型存储）
│   ├── models.py            # ORM（AlertRecord, BaselineRecord, TrainingRecord, AuditLog, User, ModelRecord）
│   ├── audit.py             # 操作审计日志
│   ├── backup.py            # 数据库备份/恢复
│   └── model_registry.py    # 模型版本管理（注册/部署/回滚）
│
├── validation/              # 验证与评估
│   ├── synthetic.py         # 合成异常生成
│   └── recall_test.py       # Recall 评估
│
└── dashboard/               # Web UI
    ├── app.py               # FastAPI 应用工厂（服务 Vue SPA）
    ├── auth.py              # RBAC 认证 + 限流 + 安全头
    ├── tasks.py             # 后台任务管理器（线程池）
    ├── websocket.py         # WebSocket 连接管理（主题订阅）
    └── routes/
        ├── system.py        # 总览 + 健康 API
        ├── cameras.py       # 摄像头管理
        ├── baseline.py      # 基线采集 + 模型训练 + 部署 + 训练历史
        ├── zones.py         # 检测区域编辑器
        ├── alerts.py        # 告警中心（含工作流状态机）
        ├── detection.py     # 检测调试视图 + 灵敏度预览
        ├── config.py        # 系统设置
        ├── tasks.py         # 后台任务进度 API
        ├── audit.py         # 审计日志页面
        ├── users.py         # 用户管理（RBAC）
        ├── reports.py       # 统计报表（日报/周报/趋势）
        ├── backup.py        # 数据库备份/恢复
        └── models.py        # 模型管理 API（注册/部署/回滚）

web/                         # Vue 3 SPA 前端
├── src/
│   ├── App.vue
│   ├── main.ts
│   ├── api/index.ts
│   ├── composables/useWebSocket.ts
│   ├── router/index.ts
│   └── views/
│       ├── Overview.vue     # 总览
│       ├── Cameras.vue      # 摄像头列表
│       ├── CameraDetail.vue # 摄像头详情
│       ├── Alerts.vue       # 告警中心
│       ├── Models.vue       # 模型管理
│       └── System.vue       # 系统设置
└── dist/                    # 生产构建输出
```

---

## 4. 关键配置参数

### 4.1 配置层级

```yaml
ArgusConfig                    # 顶层
├── node_id: str
├── cameras: list[CameraConfig]
│   ├── camera_id, name, source, protocol, fps_target, resolution
│   ├── mog2: MOG2Config
│   ├── person_filter: PersonFilterConfig
│   ├── anomaly: AnomalyConfig    # dinomaly2 参数, enable_calibration, quantization
│   ├── simplex: SimplexConfig    # Simplex 安全通道配置
│   ├── classifier: ClassifierConfig  # YOLO-World OVD 配置
│   ├── segmenter: SegmenterConfig    # SAM 2 分割配置
│   └── capture_quality: CaptureQualityConfig
├── alerts: AlertConfig
│   ├── severity_thresholds
│   ├── temporal (CUSUM evidence)
│   ├── suppression
│   ├── webhook
│   └── email
├── drift: DriftConfig             # KS 漂移监控配置
├── cross_camera: CrossCameraConfig # 跨摄像头关联配置
├── auth: AuthConfig               # RBAC 用户认证
├── dashboard: DashboardConfig
├── storage: StorageConfig
├── models: ModelsConfig
└── logging: LoggingConfig
```

### 4.2 核心参数默认值

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| MOG2 历史帧数 | 500 | 10-5000 | 背景建模所用帧数 |
| 变化检测阈值 | 0.5% | 0.01%-50% | 低于此比例视为无变化 |
| 心跳帧间隔 | 150 | 10-3000 | 强制全帧检测间隔 |
| 锁定触发分数 | 0.85 | 0.5-0.99 | 异常锁定触发阈值 |
| 锁定解除帧数 | 10 | 1-100 | 连续正常帧数解除锁定 |
| CUSUM evidence_lambda | 0.95 | 0.5-1.0 | CUSUM 证据衰减系数 |
| CUSUM evidence_threshold | 3.0 | 0.5-20.0 | CUSUM 确认触发阈值 |
| 确认最大间隔 | 10s | 1-120s | 帧间最大时间间隔 |
| 空间重叠(IoU) | 0.3 | 0-1.0 | 连续帧异常区域重叠率 |
| 同区域抑制 | 300s | 10-3600s | 同区域告警去重窗口 |
| 告警保留 | 90 天 | 7-3650 天 | 自动清理周期 |
| Drift reference_window | 500 | 50-5000 | KS 检验参考窗口大小 |
| Drift test_window | 100 | 20-1000 | KS 检验检验窗口大小 |
| Drift p_value_threshold | 0.01 | 0.001-0.1 | 漂移判定 p 值阈值 |
| Simplex diff_threshold | 30 | 5-100 | 帧差绝对值阈值 |
| Simplex min_static_seconds | 30 | 5-300 | 最小静态参考更新间隔(秒) |

---

## 5. 代码统计

| 类别 | 文件数 | 说明 |
|------|--------|------|
| 核心源码 (`src/argus/`) | 60 | 含 __init__.py |
| Dashboard 路由 | 12 | 不含 __init__.py |
| Vue 前端 (`web/src/`) | 12 | .vue + .ts |
| 单元测试 | 33 | |
| 集成测试 | 8 | |
| 工具脚本 | 4 | |
| 配置/文档 | 6 | |

测试覆盖：**595 个测试函数**（单元 ~554 + 集成 41）

---

## 6. 技术栈

| 层 | 技术 | 版本 |
|---|------|------|
| 视频采集 | OpenCV | >= 4.10 |
| 目标检测 | Ultralytics YOLO11n (多类 + BoT-SORT) | >= 8.3 |
| 异常检测 | Anomalib (Dinomaly2/PatchCore/EfficientAD/FastFlow/PaDiM)，DINOv2 backbone | >= 2.0 |
| OVD 分类 | YOLO-World（开放词汇检测） | - |
| 模型集成 | 多模型 Ensemble | 内置 |
| 推理加速 | OpenVINO | >= 2024.0 |
| INT8 量化 | NNCF | >= 2.0 |
| Web 框架 | FastAPI + WebSocket | >= 0.115 |
| 前端 | Vue 3 + Ant Design Vue + Vite | Vue 3.5 |
| 数据库 | SQLAlchemy + SQLite (WAL) | >= 2.0 |
| 配置 | Pydantic v2 + YAML | >= 2.0 |
| 日志 | structlog（JSON 文件 + 控制台） | >= 24.0 |
| 容器 | Docker + docker-compose | - |
| 语言 | Python | >= 3.11 |

---

## 7. 部署架构

```
docker-compose.yml
├── argus-edge (单容器)
│   ├── Port: 8080 (Dashboard + WebSocket)
│   ├── Volumes:
│   │   ├── ./data/baselines → /app/data/baselines
│   │   ├── ./data/models    → /app/data/models
│   │   ├── ./data/exports   → /app/data/exports
│   │   ├── ./data/db        → /app/data/db
│   │   ├── ./data/alerts    → /app/data/alerts
│   │   └── ./data/logs      → /app/data/logs
│   ├── Resources: 4 CPU, 4GB RAM
│   ├── Healthcheck: GET /api/system/health (30s interval)
│   └── Restart: unless-stopped
```

---

## 8. ORM 模型

| 表 | 类 | 用途 |
|----|-----|------|
| alerts | AlertRecord | 告警记录（含工作流状态机） |
| baselines | BaselineRecord | 基线图片索引 |
| training_records | TrainingRecord | 训练历史（参数/指标/质量等级） |
| audit_logs | AuditLog | 操作审计（用户/动作/目标/详情） |
| users | User | RBAC 用户（admin/operator/viewer） |
| models | ModelRecord | 模型版本管理（注册/部署/回滚/指标） |
