# Argus 系统架构文档

> 最后更新：2026-05-04

本文档描述当前仓库已经落地的系统结构、运行时关系和主要边界，内容以实际代码路径为准，而不是历史规划中的理想版本。

## 1. 系统定位

Argus 是面向固定摄像头场景的单节点边缘视觉异常检测系统，当前仓库包含两个主应用和一组长期运行的后台子系统：

- Python 后端：负责采集、检测、告警、训练作业、存储和 API。
- Vue 前端：负责监控、运维、回放、模型与基线操作。
- 后台子系统：负责任务调度、流媒体代理、回放归档、推理记录、主动学习和降级管理。

当前默认部署形态是单机运行：业务元数据存储在 SQLite，本地文件系统保存基线、模型、录像、推理记录和备份。

## 2. 运行时拓扑

当前进程内的主要运行关系如下：

```text
argus CLI 进程
  -> 配置加载与日志初始化
  -> Database / Audit / Metrics
  -> AlertDispatcher
  -> InferenceRecordStore
  -> AlertRecordingStore
  -> go2rtc
  -> EventBus + ActiveLearningSampler
  -> CameraManager
     -> 每路摄像头一个 DetectionPipeline / Runner
  -> TaskScheduler
  -> FastAPI App
     -> REST API
     -> WebSocket ConnectionManager
     -> Vue SPA 静态托管
```

这里的关键点不是“模块有哪些”，而是“哪些对象在运行期长期持有状态”：

- `CameraManager` 持有所有摄像头的生命周期与状态。
- 每路 `DetectionPipeline` 持有本摄像头的检测链、统计信息、锁定状态和诊断缓冲。
- `TaskScheduler` 持有周期性维护任务与训练作业处理任务。
- FastAPI `app.state` 持有数据库、摄像头管理器、健康监视器、任务管理器、go2rtc、降级管理器和回放存储等共享对象。

## 3. 启动与关闭流程

当前启动入口为 `src/argus/__main__.py`，实际顺序大致如下：

1. 解析命令行参数并加载 YAML 配置。
2. 初始化结构化日志、旋转文件日志和运行指标。
3. 初始化 `Database` 并在首次启动时创建默认管理员。
4. 初始化 `AlertDispatcher`、`InferenceRecordStore`、`AlertRecordingStore`。
5. 如果启用 go2rtc，则优先启动 go2rtc，并对 USB 摄像头做源地址重定向。
6. 初始化 `EventBus` 与 `ActiveLearningSampler`。
7. 创建 `CameraManager`，将数据库、录制存储、分类器/分割器配置、跨摄像头配置等注入检测链。
8. 创建 FastAPI 应用，把共享对象挂到 `app.state`。
9. 启动 `TaskScheduler`，注册维护任务与训练作业处理任务。
10. 启动全部摄像头检测线程。
11. 主线程进入健康轮询、指标刷新和 go2rtc 存活检查循环。

关闭时则按相反方向收缩：先关闭 go2rtc，再停止调度器、摄像头、记录存储、告警分发和数据库连接。

这个顺序很重要，因为当前实现依赖 go2rtc 尽早接管 USB 摄像头，随后检测链和浏览器都消费代理后的流地址。

## 4. 核心检测架构

每路摄像头由独立的检测运行上下文承载，避免单路异常拖垮全局。主链路如下：

```text
Frame Source
  -> CameraCapture
  -> Zone Mask
  -> MOG2 Prefilter
  -> YOLO Person / Object Filter
  -> Anomaly Detector
  -> Simplex Safety Channel
  -> Postprocess / Temporal Tracker
  -> AlertGrader
  -> AlertDispatcher + Storage
```

### 4.1 Capture Layer

- 支持 RTSP、USB、视频文件输入。
- `CameraCapture` 负责单路采集、重连、帧率控制与基础统计。
- `CameraManager` 统一负责多摄像头启动、停止、状态汇总和共享模型资源注入。

### 4.2 Zone Mask

- `ZoneMaskEngine` 在检测早期应用 include / exclude 区域裁剪。
- 区域配置同时来自 YAML 和 Dashboard API。
- 多 include zone 会被独立评估，最后再汇总到告警分级逻辑。

### 4.3 Prefilter

- 当前主预筛是 `MOG2PreFilter`。
- 设计目标是降低主检测器吞吐压力，而不是替代主检测器。
- 当前实现包含 heartbeat 全帧旁路、稳定化和锁定区域旁路等机制，用来缓解静态异物被背景吸收的问题。

### 4.4 Person / Object Filter

- 当前过滤器基于 Ultralytics YOLO。
- 既可做人过滤，也支持按类别过滤。
- 支持 tracking 与 SAHI 切片推理等可选路径，是否启用由摄像头配置控制。

### 4.5 Anomaly Detector

- 当前代码路径覆盖 PatchCore、EfficientAD、Dinomaly 相关模型流程。
- 检测器支持运行时模型自动发现、共享检测器、多尺度模式和校准选项。
- 当模型不可用或需要降级时，检测链会进入兜底或恢复流程，而不是简单崩溃退出。

### 4.6 Simplex Safety Channel

- Simplex 是独立于主异常模型的传统 CV 安全通道。
- 它与主检测结果并行存在，用来提供低依赖、低复杂度的兜底候选信号。
- 当前默认配置启用该通道，因此它属于默认运行路径的一部分。

### 4.7 Postprocess / Temporal Logic

- 包括异常图后处理、时序跟踪、自适应阈值、CUSUM 证据累积、锁定与清除逻辑。
- `TemporalTracker`、`adaptive_threshold`、`anomaly_postprocess`、`correlation` 等模块位于这一层。
- `frame_quality`、`diagnostics`、`circuit_breaker` 等支撑模块也在这里与主链相邻。

### 4.8 Alerting

- `AlertGrader` 负责把分数、时序证据、区域优先级和检测类型转换为最终告警等级。
- `AlertDispatcher` 负责落库和对外分发。
- 告警图像、录像、工作流状态和反馈队列都围绕这一阶段沉淀。

## 5. 事件、反馈与主动学习

当前实现不只是“检测后直接告警”，还包含一条事件驱动的反馈回路：

```text
DetectionPipeline
  -> EventBus.publish(FrameAnalyzed / AlertRaised / ...)
  -> ActiveLearningSampler
  -> Labeling Queue / data/active_learning
  -> Training Jobs
  -> New Model / Baseline Update
```

### 5.1 EventBus

- `EventBus` 用于在同步检测链和其他子系统之间传递事件。
- 主要用途是降低检测链与主动学习、回放、诊断之间的直接耦合。

### 5.2 Active Learning

- `ActiveLearningSampler` 监听 `FrameAnalyzed` 事件。
- 它基于异常分数和熵估计挑选不确定帧，保存到主动学习目录，并落库到标注队列。
- 这条链路把“检测结果”转为“待人工标注样本”，是当前训练闭环的重要补充。

### 5.3 Feedback 与 Labeling

- Dashboard 暴露标注队列相关 API 与模型页标签入口。
- 反馈管理器挂在 `app.state.feedback_manager`，供反馈队列与基线目录协同使用。
- 这部分能力已经落地，但更多体现为后端能力和模型页标签，而不是独立主路由。

## 6. 健康检查、降级与可恢复性

### 6.1 HealthMonitor

- `HealthMonitor` 汇总摄像头连接状态、帧数、延迟、告警数和进程运行时间。
- 主线程持续更新健康信息，并同步刷新 Prometheus 指标。
- 系统页中的“系统概览”和摄像头健康信息依赖这一层数据。

### 6.2 降级状态机

- `DegradationStateMachine` 负责单摄像头推理运行状态切换。
- 状态包括 nominal、segmenter 降级、detector 降级、backbone 失败和 restarting。
- 非法状态切换会被拒绝并记录日志。

### 6.3 全局降级管理

- `GlobalDegradationManager` 负责生成面向操作员的全局降级事件和中文文案。
- Dashboard 顶部降级条与历史记录依赖该层。
- 这意味着降级在当前系统中不是“内部日志概念”，而是用户可见的第一类运行状态。

### 6.4 熔断与兜底

- 对外告警分发链带有熔断思路，避免异常外部依赖拖垮主流程。
- 模型、流媒体和分割器等子系统都有降级路径，目标是“功能降级但服务继续存活”。

## 7. Dashboard 与 API 架构

FastAPI 应用位于 `src/argus/dashboard/app.py`，当前承担三类职责：

- HTTP API：配置、摄像头、告警、模型、训练作业、回放、系统管理等。
- WebSocket：向前端广播 health、cameras、alerts、tasks、degradation 等主题消息。
- SPA 托管：生产模式下直接托管 `web/dist`。

### 7.1 App State 注入

共享对象通过 `app.state` 暴露给路由和中间件，主要包括：

- `db` / `database`
- `camera_manager`
- `health_monitor`
- `task_manager`
- `ws_manager`
- `go2rtc`
- `baseline_lifecycle`
- `feedback_manager`
- `degradation_manager`
- `recording_store`

这使得 Dashboard 层本质上是“HTTP/WebSocket 门面”，而不是独立业务核心。

### 7.2 API 主题

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

### 7.3 WebSocket

- `ConnectionManager` 使用 janus 队列桥接同步线程和异步 WebSocket 客户端。
- 广播主题目前包括 health、cameras、alerts、tasks、wall、degradation、heatmap。
- 这层设计的目标是让摄像头线程、告警分发器和后台任务可以线程安全地向前端推送状态，而不直接依赖事件循环。

### 7.4 中间件与安全

- 当前中间件包括安全头、限流和认证。
- 认证状态同时约束普通 API 和 WebSocket。
- 项目明确避免使用会缓冲流式响应的 `BaseHTTPMiddleware`，相关中间件均应保持纯 ASGI 风格。

## 8. 前端架构

前端位于 `web/`，使用 Vue 3、TypeScript、Vite、Pinia、Ant Design Vue。

当前主路由为：

- `/overview`
- `/cameras`
- `/cameras/:id`
- `/alerts`
- `/reports`
- `/models`
- `/system`

另有：

- `/training` 重定向到 `/models?tab=training`，不是独立页面。
- `/replay/:alertId` 和 `/replay/:alertId/storyboard` 是从告警跳转的参数化子路由，由 `views/ReplayView.vue` 和 `views/StoryboardReplay.vue` 渲染，不在侧栏出现。

### 8.1 页面职责

- Overview：系统总览、状态聚合、摄像头概况。
- Cameras / CameraDetail：摄像头清单、播放、诊断和控制。
- Alerts：告警列表、筛选、工作流操作和回放详情。
- Reports：报表与统计分析。
- Models：基线、训练与评估、模型发布、A/B 对比、标注队列、阈值预览。
- System：系统概览、配置管理、备份、审计、降级历史、音频告警、用户管理、存储清理。
- ReplayView / StoryboardReplay：从告警进入的录像回放视图，分别是单机位回放和多机位故事板回放。

### 8.2 前后端边界

- 前端只消费后端已经稳定提供的 API 与 WebSocket 主题。
- “后端有 API”不等于“前端有独立主页面”。
- 训练、标注、配置、审计等能力当前主要表现为模型页或系统页下的标签聚合。

## 9. 数据与存储

### 9.1 结构化存储

- 主数据库为 SQLite。
- ORM 模型位于 `src/argus/storage/models.py`。
- 数据库存储告警、用户、审计、训练作业、模型记录、标注队列等结构化数据。

### 9.2 文件存储

默认数据目录位于 `data/`，主要包括：

- `alerts/`：告警图像与相关数据。
- `recordings/`：回放视频归档。
- `inference_records/`：逐帧推理记录。
- `baselines/`：基线图像与版本目录。
- `models/`：已导出模型。
- `backbones/`：骨干网络训练产物。
- `exports/`：导出产物。
- `logs/`：日志文件。
- `backups/`：数据库与配置备份。

### 9.3 运行期存储组件

- `Database`：数据库访问与迁移辅助。
- `ModelRegistry`：模型版本、激活状态和查询入口。
- `InferenceRecordStore`：逐帧推理结果落盘。
- `AlertRecordingStore`：录像归档、修复与回放素材管理。
- `BackupManager`：备份创建与保留。

## 10. 调度、训练与后台维护

### 10.1 TaskScheduler

- 当前调度器优先使用 APScheduler；缺失时会退化为禁用状态。
- 调度器承载系统维护任务，而不是直接承载摄像头实时检测。

### 10.2 维护任务

当前已经注册的周期任务包括：

- stale camera 检查
- 旧告警清理
- 推理记录清理
- 磁盘空间检查
- 训练作业处理
- 自动备份
- 条件启用的自动重训练

此外，代码中已经定义了 backbone 重训练检查任务工厂，但当前启动链尚未注册它。

### 10.3 训练作业模型

- 训练作业先通过 API 创建为待确认状态。
- 运营或工程人员确认后，调度器定期拉取 queued 作业并执行。
- 这意味着训练链路在架构上是“异步后台任务”，而不是同步请求内执行。

## 11. 流媒体路径

go2rtc 是当前视频访问链条中的关键节点：

- 浏览器优先走 go2rtc 提供的 WebRTC、MSE、HLS 能力。
- 对 USB 摄像头，go2rtc 会先独占设备，再重定向成 RTSP 给检测链消费。
- Dashboard-only 模式下，FastAPI 生命周期也可启动 go2rtc。
- 当 go2rtc 不可用时，前端会回退到 MJPEG 相关路径。

因此流媒体子系统不是单纯的“播放器配套”，而是同时影响浏览器访问和检测链视频来源。

## 12. 可选能力与默认状态

以下能力在代码或配置中存在，但默认配置通常关闭，应视为可选扩展能力：

- 开放词汇分类器
- 分割器
- 跨摄像头关联
- 自动重训练

以下能力属于默认运行路径的一部分：

- go2rtc 集成
- Simplex 安全通道
- 健康检查
- 漂移检测
- 环形缓冲与录像回放链路

## 13. 当前边界与维护原则

为避免文档再次漂移，后续维护应遵循以下原则：

- 以 `__main__.py` 的启动顺序和 `create_app()` 的装配结果为准。
- 以实际注册的 API、WebSocket 主题和前端主路由为准。
- 对默认关闭的模块只描述为“可选”或“支持接入”。
- 对反馈、回放、训练、标注等能力，要明确区分“后端能力”“标签页入口”和“独立主页面”。
- 不再使用未经核实的模块数量、测试数量或宣传式功能统计。

## 14. 2026-04 后增量

本节记录 `2026-04-12` 之后合入主干、影响整体架构边界的关键改动。后续如再做大改，仍应在这里追加摘要并同步更新对应正文小节，避免读者只读前 13 节就拿到过期心智。

### 14.1 安全 / RBAC（commit `91dfc2f`）

- 调度器新增 stage gate：之前 `auto_retrain` 路径会以 `allow_bypass=True` 跳过 release pipeline，现在已与人工激活共用同一套 stage gate（见 `src/argus/storage/model_registry.py` 的 `activate` 校验段）。
- 7 个原本宽松的端点补齐 RBAC 校验（覆盖系统配置、模型管理、训练任务等敏感动作），未授权调用会被中间件拒绝。
- 17 处 audit 写入不再硬编码 `triggered_by`，改为从认证身份解析，方便审计回放。

参考阅读：`docs/enable_checklists/auto_retraining.md` §1 / §6.3 已同步更新对应描述。

### 14.2 模型发布管线（commit `a68db43`）

- Release pipeline 接入 WebSocket：阶段流转事件实时推送到前端，Models 页可观察 shadow → canary → production 的进度。
- `activate` 端点不再有 bypass 路径，所有激活动作都必须先通过 stage gate。
- 训练阶段不再热载候选模型：训练完成的产物先注册为 `CANDIDATE`，等通过 release pipeline 才会被任何 pipeline 采用，避免“训练即上线”的污染风险。

### 14.3 训练管线（commit `da9453e`）

- `dataset_strategy` 多版本策略落地：单次重训练任务可以指定数据集版本组合（例如“当前 v3 baseline + 上次 FP 增量”），实现见 `src/argus/core/scheduler.py` 中 `_resolve_dataset_selection_for_retrain`。
- 默认配置中分类器仍保持关闭（与 README/requirements 一致），新增训练任务不会因配置缺省而把 D1 误开。
