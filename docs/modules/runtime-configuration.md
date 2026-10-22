# 运行装配与配置

> 关键源码：`src/argus/__main__.py`、`src/argus/config/schema.py`、`src/argus/config/loader.py`、`configs/default.yaml`

本模块说明 Argus 进程如何启动、配置如何进入各子系统、哪些能力默认启用，以及后台任务如何挂载。

## CLI 入口

运行入口是：

```powershell
python -m argus --config configs/default.yaml
```

`src/argus/__main__.py` 中的 `main()` 负责：

- 解析命令行参数：`--config`、`--camera`、`--verbose`、`--no-dashboard`、`--dev-video`、`--dev-video-motion`、`--dev-fast-training`。
- 加载 YAML 配置并初始化结构化日志、文件日志、GPU/CUDA 环境日志和 Prometheus 指标。
- 初始化 SQLite、默认管理员、推理写缓冲、基线管理器、训练器、告警分发器、回放录像存储、go2rtc、事件总线和主动学习采样器。
- 创建 `CameraManager`、`TaskManager`、`TaskScheduler` 和 FastAPI app。
- 启动 go2rtc、注册/改写摄像头源、启动摄像头线程和 Dashboard 线程。
- 主线程持续刷新健康状态、指标和 go2rtc 存活状态，退出时按相反方向关闭资源。

## 配置加载

配置入口是 `load_config()`：

- `src/argus/config/loader.py` 读取 YAML，递归合并 layered config，并做路径分隔符归一化。
- `src/argus/config/schema.py` 定义全部 Pydantic 模型，包含摄像头、告警、Dashboard、存储、模型、分类器、分割器、跨摄像头、训练、成像、物理、连续录像、传感融合等配置。
- `configs/default.yaml` 是当前默认配置，不等同于所有功能都启用。

新增配置时应同时更新：

- `schema.py` 的模型字段和默认值。
- `configs/default.yaml`。
- 使用该配置的 route 或 pipeline 热更新逻辑。
- 相关测试，尤其是配置 API、模块开关、pipeline 行为测试。

## 默认能力

默认配置下通常启用：

- USB 示例摄像头。
- Dashboard 只绑定 `127.0.0.1`，认证默认关闭，用于本机开发和 smoke 测试。
- go2rtc 集成。
- Simplex 安全通道。
- MOG2 预筛、alignment、fast motion、低光照处理。
- 健康检查、漂移检测、环形缓冲与告警录像回放。
- 人工确认后的训练作业模型。

默认配置下通常关闭：

- 开放词汇分类器。
- 分割器。
- 跨摄像头关联。
- 自动重训练。

文档和演示应明确区分“代码已有”“配置可开启”和“默认运行路径”。

## 外部访问安全

默认配置不是远程暴露配置。需要从局域网或生产环境访问 Dashboard 时，必须显式开启认证并改为外部绑定：

```powershell
$env:ARGUS__AUTH__ENABLED='true'
$env:ARGUS__AUTH__API_TOKEN='<long random token>'
$env:ARGUS__DASHBOARD__HOST='0.0.0.0'
python -m argus --config configs/default.yaml
```

新数据库首次启动会创建 `admin/admin` bootstrap 用户。外部暴露前必须先在本机登录并修改或替换该管理员账号。启动链会在 Dashboard 绑定非 loopback 地址且认证关闭时输出非阻断 warning，提醒该配置只适合受控开发环境。

## 开发模式

`--dev-video` 会调用 `_apply_dev_video_source()`：

- 生成 `data/dev/demo_camera.avi`。
- 将本次进程中的目标摄像头源改为 `file`。
- 关闭 Dashboard go2rtc 集成，避免本地文件流仍走摄像头代理。

`--dev-fast-training` 会把正常 `ModelTrainer` 替换为确定性开发训练器，方便 smoke 测试走通训练/导出/UI 链路。它不是生产训练路径。

## 调度任务

后台调度由 `src/argus/core/scheduler.py` 的 `TaskScheduler` 承载。当前任务类型包括：

- stale camera 检查。
- 旧告警和录像清理。
- 推理记录清理。
- 磁盘空间检查。
- 训练作业处理。
- 自动备份。
- 条件启用的自动重训练。
- 条件启用的 backbone 重训练检查。

训练作业不会在 HTTP 请求内同步执行。API 先创建待确认作业，确认后调度器定期取 queued 作业执行。

## App State 注入

`create_app()` 把运行期对象挂到 `app.state`，主要包括：

- `db` / `database`
- `camera_manager`
- `health_monitor`
- `task_manager`
- `ws_manager`
- `go2rtc`
- `stream_registry`
- `camera_orchestrator`
- `baseline_lifecycle`
- `baseline_manager`
- `model_trainer`
- `feedback_manager`
- `degradation_manager`
- `recording_store`
- `sensor_fusion`

Dashboard route 应通过这些对象完成操作，不应重新创建数据库、摄像头管理器或训练器。

## 相关测试

- `tests/unit/test_config.py`
- `tests/unit/test_config_loader.py`
- `tests/unit/test_config_api.py`
- `tests/unit/test_scheduler.py`
- `tests/unit/test_task_pause.py`
- `tests/unit/test_training_job_wiring.py`
- `tests/integration/test_runtime_imports.py`
