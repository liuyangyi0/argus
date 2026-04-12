# Argus 系统架构文档

> 最后更新：2026-04-11

本文档描述当前仓库已经落地的系统结构，而不是历史规划版本中的理想形态。

## 1. 系统定位

Argus 是面向固定摄像头场景的边缘侧视觉异常检测系统。当前仓库由两个主要应用组成：

- Python 后端：负责采集、检测、告警、模型与数据管理。
- Vue 前端：负责监控、运维、回放、模型与基线操作。

典型部署为单节点运行，使用 SQLite 存储业务数据，本地文件系统存储基线、模型、录制和推理记录。

## 2. 顶层架构

```text
摄像头 / 视频源
  -> Capture Layer
  -> Detection Pipeline
  -> Alerting + Persistence
  -> Dashboard API + WebSocket
  -> Vue Dashboard

辅助子系统
  -> TaskScheduler / TaskManager
  -> go2rtc
  -> Database / ModelRegistry
  -> AlertRecordingStore / InferenceRecordStore
```

## 3. 启动流程

当前启动入口为 `src/argus/__main__.py`，整体顺序如下：

1. 解析命令行参数并加载 YAML 配置。
2. 初始化结构化日志、数据库、审计与指标。
3. 初始化告警分发、推理记录存储和告警录像存储。
4. 如果启用 go2rtc，则优先启动 go2rtc 并为摄像头准备流代理。
5. 创建 CameraManager、HealthMonitor、TaskScheduler、TaskManager。
6. 创建 FastAPI 应用并挂载 API、WebSocket 与前端静态资源。
7. 启动摄像头检测线程与 Dashboard 服务。

这个顺序很重要，因为 USB 摄像头在当前实现中优先交给 go2rtc 独占，再由检测管线消费重定向后的 RTSP 流。

## 4. 检测管线

每个摄像头拥有独立运行上下文，核心处理链如下：

```text
Frame
  -> Zone Mask
  -> MOG2 Prefilter
  -> Person Filter
  -> Anomaly Detector
  -> Simplex Safety Channel
  -> Postprocess / Temporal Tracker
  -> AlertGrader
  -> Dispatcher / Storage
```

### 4.1 Capture Layer

- 支持 RTSP、USB、文件输入。
- CameraManager 负责多摄像头生命周期管理。
- 采集层包含重连、帧率控制、质量过滤、健康监控。

### 4.2 Zone Mask

- 在检测早期应用 include / exclude 区域限制。
- 区域配置来自 YAML 和 Dashboard API。

### 4.3 MOG2 Prefilter

- 负责前景变化筛选，减少主检测链的算力消耗。
- 支持去噪、相位相关稳定、heartbeat 强制放行和锁定旁路。

### 4.4 Person Filter

- 基于 Ultralytics YOLO 的人员或指定类别过滤。
- 可选择遮罩或跳帧策略。
- 跟踪能力和类别列表由配置控制。

### 4.5 Anomaly Detector

- 当前代码路径支持 PatchCore、EfficientAD、Dinomaly 相关配置与模型管理流程。
- 无模型场景下保留冷启动与兜底逻辑。
- 训练、导出、发布、比较等能力位于 `src/argus/anomaly/`。

### 4.6 Simplex Safety Channel

- 作为独立的传统 CV 安全通道，与主异常检测结果并行存在。
- 主要职责是提供简单、稳定、低依赖的异常候选信号。

### 4.7 Postprocess 与 Temporal Logic

- 包含异常图后处理、时序跟踪、自适应阈值、CUSUM 证据累积等逻辑。
- 漂移检测、降级检测和推理记录也在这一层附近完成。

### 4.8 Alerting

- AlertGrader 负责按分数、区域优先级和时序证据生成告警等级。
- AlertDispatcher 负责落库与对外分发。
- 环形缓冲和录像存储负责生成回放素材。

## 5. 数据与存储

### 5.1 数据库

- 当前主存储为 SQLite。
- ORM 模型位于 `src/argus/storage/models.py`。
- 数据库存储告警、用户、审计、训练作业、模型记录等结构化数据。

### 5.2 文件存储

默认数据目录在 `data/`，当前主要子目录包括：

- `alerts/`：告警相关数据。
- `baselines/`：基线图像与版本目录。
- `models/`：导出的异常检测模型。
- `backbones/`：骨干网络产物。
- `recordings/`：回放视频。
- `inference_records/`：逐帧推理记录。
- `logs/`：日志输出。
- `backups/`：数据库备份文件。

### 5.3 运行期存储组件

- `Database`：SQLite 访问与迁移辅助。
- `ModelRegistry`：模型版本与激活状态管理。
- `InferenceRecordStore`：逐帧推理记录落盘。
- `AlertRecordingStore`：告警回放素材归档与修复。

## 6. Dashboard 与 API

FastAPI 应用位于 `src/argus/dashboard/app.py`，当前职责包括：

- 认证、会话、限流和安全响应头。
- `/api/*` JSON API。
- `/ws` WebSocket 实时推送。
- 前端构建产物静态托管。

当前已装配的 API 主题包括：

- cameras
- alerts
- zones
- config
- detection
- system
- tasks
- audit
- backup
- users
- reports
- models
- replay
- degradation
- training-jobs
- streaming
- labeling
- baseline

## 7. 前端结构

前端位于 `web/`，使用 Vue 3、TypeScript、Vite、Pinia、Ant Design Vue。

当前主路由为：

- `/overview`
- `/cameras`
- `/cameras/:id`
- `/alerts`
- `/models`
- `/system`

其中：

- 模型页通过 Tab 聚合基线、训练、模型发布、A/B 对比、标注和阈值预览。
- 系统页通过 Tab 聚合系统概览、配置管理、用户、审计、降级历史、音频配置、备份与清理。

## 8. 后台任务与调度

- `TaskScheduler` 负责维护定时任务和训练作业处理。
- `TaskManager` 负责前端可见的任务状态。
- 训练作业通过 `training_jobs` API 创建、确认和查询，当前实现要求人工确认后再执行。

## 9. 流媒体路径

go2rtc 是当前视频访问链条中的关键组件：

- 浏览器优先使用 go2rtc 提供的流协议。
- 检测链可通过 go2rtc 重定向后的 RTSP 地址消费视频。
- go2rtc 不可用时，Dashboard 会退回到 MJPEG 相关路径。

## 10. 可选能力与默认状态

以下能力在代码与配置中存在，但默认配置通常关闭：

- 开放词汇分类器
- SAM2 分割
- 跨摄像头关联
- 自动重训练

因此它们应被视为“可选扩展能力”，而不是默认运行路径。

## 11. 当前实现边界

为避免文档继续漂移，建议按以下原则理解仓库：

- 以代码中已注册的路由、已存在的前端页面和默认配置为准。
- 对于配置项已存在但默认关闭的模块，只描述为“支持接入”或“可选”。
- 不再使用未经核实的模块数量、测试数量、页面数量等宣传式统计。
