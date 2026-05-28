# Argus

Argus 是一个面向固定摄像头场景的视觉异常检测系统。仓库当前实现聚焦于单节点边缘部署：后端负责视频采集、检测管线、告警与模型生命周期，前端提供 Vue 3 仪表盘用于监控、回放、基线采集和训练管理。

## 当前实现概览

- 多源采集：支持 RTSP、USB、视频文件输入，带重连与健康检查。
- 分阶段检测：区域掩码、MOG2 预筛、人员过滤、异常检测、安全通道、后处理、告警分级。
- 双通道异常判断：主异常检测通道配合 Simplex 安全通道，降低单点失效风险。
- 告警闭环：SQLite 持久化、Webhook / 邮件 / WebSocket 分发、工作流状态管理、回放取证。
- 模型与基线：基线版本管理、训练作业、模型注册、发布阶段切换、批量推理。
- Dashboard：Vue 3 + Ant Design Vue 单页应用，提供总览、摄像头、告警、模型、系统管理等页面。

## 当前系统架构

```text
Camera Source (RTSP / USB / File)
  -> CameraManager
  -> DetectionPipeline
     -> Zone Mask
     -> MOG2 Prefilter
     -> Person Filter
     -> Anomaly Detector
     -> Simplex Safety Channel
     -> Postprocess / Temporal Logic
     -> AlertGrader
     -> AlertDispatcher + Storage

FastAPI Backend
  -> JSON API
  -> WebSocket
  -> Replay / Streaming / Task APIs

Vue Dashboard
  -> Overview
  -> Cameras / Camera Detail
  -> Alerts
  -> Models
  -> System
```

## 与旧文档相比的校正

- 本仓库当前前端主路由为 7 个：总览、摄像头、摄像头详情、告警、报表、模型、系统。另有两个从告警跳转的参数化子路由（`/replay/:alertId` 和 `/replay/:alertId/storyboard`），由 ReplayView 和 StoryboardReplay 视图渲染，不在侧栏出现。
- 用户、审计、备份等能力目前主要以后端 API 和系统页子标签形式暴露；训练作业入口位于模型页标签中，而不是独立前端路由。
- YOLO-World 分类、SAM2 分割、跨摄像头关联、自动重训练等能力在配置层和后端代码中已预留，但默认配置下通常关闭，不应视为默认启用能力。
- 项目版本信息以代码为准：Python 包版本当前为 0.1.0，运行时指标与 FastAPI 应用内部版本使用 0.2.0。

## 快速开始

### 后端

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m argus --config configs/default.yaml
```

默认情况下 Dashboard 运行在 http://127.0.0.1:8080。
如果本机没有 USB / RTSP 摄像头，可用同一份默认配置启动可复现的本地视频源：

```powershell
python -m argus --config configs/default.yaml --dev-video
```

这会生成 `data/dev/demo_camera.avi`，并在本次进程中把摄像头切到 `file` 输入，便于验证 Cameras → Alerts → Replay → Models → System → Reports 闭环。
需要在本机快速验证训练/导出 UI 链路时，可额外加 `--dev-fast-training`；它会生成可被 OpenVINO 加载的确定性开发模型工件，不会替代正常 anomalib 训练。

### 前端开发模式

```powershell
cd web
npm install
npm run dev
```

### 测试

```powershell
python -m pytest tests/ -v

# 核心业务闭环：Cameras -> Alerts -> Replay -> Models -> System -> Reports
python scripts/smoke_core_loop.py --timeout 75 --recording-timeout 45

# 后端实际托管的 Dashboard 路由与媒体入口
python scripts/smoke_dashboard_routes.py --timeout 75

# 真实业务数据 + 浏览器页面：Camera detail -> Alerts -> Replay -> Models -> System -> Reports
python scripts/smoke_dashboard_business_flow.py --timeout 90 --recording-timeout 90 --browser required

# 同一条业务路径也可指向真实 USB/RTSP 源；先 preflight，确认摄像头/go2rtc 能读到帧
# USB preflight 会在 Windows 上附带系统识别到的 Camera 设备清单，便于定位设备枚举问题
python scripts/smoke_dashboard_business_flow.py --preflight --camera-source 0 --camera-protocol usb --require-go2rtc
python scripts/smoke_dashboard_business_flow.py --preflight --camera-source 0 --camera-protocol usb --camera-resolution 1920,1080 --usb-device-name "OBSBOT Meet 2 StreamCamera" --require-go2rtc --preflight-measure-seconds 30

# 无 RTSP 硬件时，可用本地 go2rtc 夹具把生成视频发布成 RTSP，再验证 RTSP + go2rtc 路径
python scripts/smoke_dashboard_business_flow.py --preflight --rtsp-fixture --require-go2rtc
python scripts/smoke_dashboard_business_flow.py --rtsp-fixture --require-go2rtc --browser required

# preflight 通过后再跑完整业务闭环；activation-delay 留给操作员放入测试目标
python scripts/smoke_dashboard_business_flow.py --camera-source 0 --camera-protocol usb --require-go2rtc --activation-delay 10 --browser required
python scripts/smoke_dashboard_business_flow.py --camera-source rtsp://user:pass@host/stream --camera-protocol rtsp --require-go2rtc --activation-delay 10 --browser required

# 真实 anomalib 训练链路（比 dev-fast 慢；会先在临时目录生成 baseline）
python scripts/smoke_dashboard_business_flow.py --training-mode normal --training-timeout 420 --browser off

# Ubuntu/GPU 测试机 CUDA 验收：实际执行 Torch CUDA、OpenCV CUDA CLAHE/MOG2，可选串联真实训练 smoke
python scripts/smoke_cuda.py
python scripts/smoke_cuda.py --business-smoke --business-training-timeout 420
```

`smoke_dashboard_routes.py` 在本机找到 Chrome / Edge / Chromium 时，会额外以 headless 浏览器执行前端 JS 并检查核心页面 DOM；需要强制浏览器检查时可加 `--browser required`。
`smoke_dashboard_business_flow.py` 会派生临时配置和数据目录，启动真实 Dashboard 服务，通过 WebSocket 验证告警实时推送，生成 Replay 证据，并默认用 `--dev-fast-training` 完成可复现的训练任务、导出、模型发布 / 回滚 API 和浏览器页面检查。需要验证真实训练时改用 `--training-mode normal`；需要验证硬件时先跑 `--preflight --camera-source ...`，USB preflight 会输出 Windows 摄像头设备清单、OpenCV 运行时包信息和排查提示，完整业务运行会同时检查 snapshot JPEG、Streaming/go2rtc 合同和业务页面 DOM。没有 RTSP 设备时可加 `--rtsp-fixture`，脚本会启动一个本地 go2rtc，把生成视频以 RTSP 发布，再让 Argus 以真实 RTSP 输入和独立 go2rtc 浏览器代理路径消费它。
`smoke_cuda.py` 面向 Ubuntu/GPU 测试机，默认要求 PyTorch CUDA 与 OpenCV CUDA 都可用，并实际执行矩阵乘、CLAHE 和 MOG2；CPU-only Windows 开发机只做诊断时可加 `--allow-missing-cuda`。

## 默认配置要点

默认配置文件位于 `configs/default.yaml`，当前主要特征如下：

- 默认示例摄像头使用 USB 输入。
- Dashboard 默认开启 go2rtc 集成。
- 本地无硬件时可用 `--dev-video` 显式切到可循环的视频文件输入，不修改 `configs/default.yaml`。
- Simplex 安全通道、健康检查、漂移检测、环形缓冲默认开启。
- 分类器、分割器、跨摄像头、自动重训练默认关闭。

## 目录结构

```text
src/argus/
  alerts/         告警分级、分发、反馈、校准
  anomaly/        异常检测、基线、训练、模型生命周期
  capture/        摄像头采集、质量过滤、基线采样
  config/         YAML + Pydantic 配置模型
  contracts/      数据契约与边界校验
  core/           检测编排、阈值、时序、健康、降级、指标
  dashboard/      FastAPI 应用、认证、中间件、API 路由
  imaging/        采集后处理、偏振与多模态融合
  person/         人员检测与过滤
  physics/        相机标定、跨摄像头几何、轨迹与速度估计
  prefilter/      MOG2 与 Simplex 预筛相关能力
  preprocessing/  帧对齐与稳定化预处理
  runtime/        日志初始化、训练任务热加载等运行时装配
  sensors/        多传感器融合接入
  storage/        SQLite、ORM、备份、模型注册、推理记录
  streaming/      go2rtc 集成与流媒体代理
  validation/     合成验证与评估工具

web/
  src/views/      7 个主路由视图 + 2 个参数化子路由视图（ReplayView / StoryboardReplay）
  src/components/ 模型管理、图表与页面组件
  src/composables/ WebSocket、模型状态、流播放逻辑

tests/
  integration/    集成测试
  unit/           单元测试
```

## 运行时组件

- FastAPI 后端负责 API、认证、WebSocket、配置重载和静态资源托管。
- CameraManager 为每个摄像头维护独立运行上下文，避免单路故障拖垮全局。
- TaskScheduler 负责周期维护任务与训练作业处理；TaskManager 负责前端可见的长任务状态与进度管理。
- Database、ModelRegistry、InferenceRecordStore、AlertRecordingStore 提供持久化能力。
- go2rtc 用于浏览器播放和 USB / RTSP 流重定向；不可用时回退到 MJPEG 路径。

## 当前前端页面

- Overview：系统总览、状态、摄像头概况。
- Cameras：摄像头列表与控制。
- CameraDetail：单摄像头详情、画面和诊断。
- Alerts：告警列表、筛选、工作流处理。
- Reports：报表与统计分析。
- Models：基线、训练、模型、A/B 对比、标注、阈值预览。
- System：系统概览、配置管理、用户、审计、降级历史、音频配置、备份与清理。
- ReplayView / StoryboardReplay：从告警跳转进入的单机位 / 多机位录像回放，参数化子路由，不在侧栏。

> 注意：本节描述当前实现（v1）。完整 UX v2（值班台 Video Wall）设计见 `docs/argus_ux_v2_enhancements_5.md`，实现仍在分阶段进行。

## 远程测试机

```bash
ssh whp222
```

- 私钥：`~/.ssh/id_ed25519_argus`
- 目标：`whp@192.168.66.222`

## 相关文档

- `docs/architecture.md`：当前实现架构与运行边界。
- `docs/requirements.md`：当前产品目标与已落地范围说明。
- `CLAUDE.md`：本仓库开发环境、命令和踩坑记录。
- `web/README.md`：前端子项目开发说明。
