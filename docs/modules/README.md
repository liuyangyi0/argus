# Argus 模块地图

> 背景索引：`docs/source-index.md`

Argus 的后端按“运行装配、采集、检测、告警、模型、存储、Dashboard 门面”拆分，前端按“路由视图、组件、组合式逻辑、API 客户端、状态 store”拆分。当前默认形态是单节点边缘部署，SQLite 和本地文件系统是主要持久化边界。

## 后端模块

| 模块 | 主要路径 | 职责 | 典型入口 |
| --- | --- | --- | --- |
| 运行装配 | `src/argus/__main__.py`、`src/argus/runtime/` | CLI、日志、开发视频、训练任务 wiring、进程生命周期 | `python -m argus --config configs/default.yaml` |
| 配置 | `src/argus/config/`、`configs/default.yaml` | YAML 加载、环境变量覆盖、Pydantic schema、默认运行参数 | `load_config()` |
| 摄像头采集 | `src/argus/capture/`、`src/argus/camera/` | RTSP/USB/file/GigE 输入、运行规划、采集质量、基线采样 | `CameraManager.start_all()` |
| 流媒体 | `src/argus/streaming/` | go2rtc 生命周期、流注册、预览 URL、USB/GigE 运行源改写 | `StreamRegistry.reconcile()` |
| 检测核心 | `src/argus/core/`、`src/argus/prefilter/`、`src/argus/person/` | 检测管线、MOG2、fast motion、人员/物体过滤、时序、降级、指标 | `DetectionPipeline.process_frame()` |
| 告警 | `src/argus/alerts/` | 告警分级、事件聚合、分发、反馈、校准 | `AlertGrader.grade()`、`AlertDispatcher.dispatch()` |
| 异常模型 | `src/argus/anomaly/` | anomalib 推理、基线、训练、模型对比、分类器/分割器、影子模型 | `AnomalibDetector.predict()`、`TrainingJobExecutor.process_queued_jobs()` |
| 存储 | `src/argus/storage/` | SQLite、ORM、轻量自动迁移、模型注册、录像归档、备份、审计 | `Database.initialize()` |
| Dashboard | `src/argus/dashboard/` | FastAPI 应用、HTTP API、WebSocket、认证、前端静态托管 | `create_app()` |
| 物理/成像/传感 | `src/argus/physics/`、`src/argus/imaging/`、`src/argus/sensors/` | 轨迹、速度、标定、多模态成像、外部传感器融合 | API route 或 pipeline optional stage |
| 验证工具 | `src/argus/validation/`、`scripts/` | 合成数据、召回评估、smoke 测试、导出/训练脚本 | `scripts/smoke_*.py` |

## 前端模块

| 模块 | 主要路径 | 职责 |
| --- | --- | --- |
| 路由与布局 | `web/src/router/`、`web/src/layouts/` | 主导航、鉴权守卫、嵌套路由、旧 `?tab=` 跳转兼容 |
| 页面 | `web/src/views/` | Overview、Cameras、Alerts、Reports、Models、System、Replay 等视图 |
| 组件 | `web/src/components/` | 视频墙、回放、告警详情、模型表格、系统面板、校准向导 |
| API 客户端 | `web/src/api/` | `/api/*` HTTP 封装、统一响应解包、错误归一化 |
| 状态与组合式逻辑 | `web/src/stores/`、`web/src/composables/` | 用户状态、告警状态、WebSocket、go2rtc 播放、回放控制 |
| 流播放服务 | `web/src/services/streaming/` | 播放器工厂和流管理，优先 go2rtc，失败时回退 |

## 主数据流

```text
CameraConfig
  -> StreamRegistry / CameraOrchestrator
  -> CameraManager
  -> CameraInferenceRunner
  -> DetectionPipeline
  -> AlertGrader
  -> AlertDispatcher
  -> Database + AlertRecordingStore
  -> FastAPI API / WebSocket
  -> Vue Dashboard
```

## 维护边界

- `src/argus/dashboard/routes/*` 不应承载核心业务状态；它们主要读取 `app.state` 上注入的服务。
- `CameraManager` 是多摄像头生命周期边界；单路检测细节属于 `DetectionPipeline` 和 `CameraInferenceRunner`。
- `ModelRegistry` 和 `ReleasePipeline` 是模型上线状态边界；训练完成不等于上线。
- `Database._AUTO_MIGRATIONS` 是 SQLite 旧库兼容边界；改 ORM 字段必须同步检查。
- go2rtc 同时影响浏览器播放和检测链运行源，不能只当作前端播放器辅助模块。
