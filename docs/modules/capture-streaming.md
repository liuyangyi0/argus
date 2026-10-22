# 摄像头采集与流媒体

> 关键源码：`src/argus/capture/`、`src/argus/camera/`、`src/argus/streaming/`、`src/argus/dashboard/routes/cameras.py`、`src/argus/dashboard/routes/streaming.py`

本模块覆盖摄像头输入如何进入检测链，以及浏览器播放如何与检测链共享同一路视频源。

## 职责划分

| 层 | 主要类/函数 | 职责 |
| --- | --- | --- |
| 输入适配 | `CameraCapture`、`CaptureFactory`、`GigECapture`、`CaptureAdapter` | 打开 RTSP/USB/file/GigE，读取帧，处理重连和基础状态 |
| 多摄像头运行 | `CameraManager` | 管理每路 `CameraInferenceRunner`，线程生命周期、状态聚合、模型热切换、跨摄像头关联 |
| 运行规划 | `CameraRuntimePlanner`、`CameraOrchestrator` | 根据摄像头配置和流注册结果生成检测链实际读取的运行源 |
| 流代理 | `Go2RTCManager`、`StreamRegistry`、`PreviewGateway` | 启动 go2rtc，注册 stream，生成 WebRTC/MSE/HLS/MJPEG URL |
| Dashboard 控制 | `/api/cameras/*`、`/api/streaming/*` | 增删改摄像头、启动停止、快照、预览流信息、手动注册 stream |

## 运行规划边界

```text
CameraConfig
  -> CameraRuntimePlanner
  -> CameraRuntimePlan
  -> StreamRegistry / PreviewGateway / CaptureFactory / CameraOrchestrator
```

`CameraConfig.source` 和 `CameraConfig.protocol` 是用户持久化意图，不能被 go2rtc relay URL 覆盖。运行时如果需要把 USB 设备、预览地址或检测输入改写成 relay，应写入 `CameraRuntimePlan`，由 orchestrator 和 capture factory 消费。

- `argus.camera.planner` 根据原始配置生成不可变运行计划。
- `argus.streaming.stream_registry` 持有期望的 go2rtc streams，并在启动或重启后做 reconcile。
- `argus.streaming.preview_gateway` 负责浏览器预览 URL、snapshot JPEG、MJPEG latest-frame、heatmap stream 和在线摄像头探测。
- `argus.capture.factory` 根据运行计划创建采集适配器；pipeline 内仍保留无注入 adapter 时的兼容分支。
- `argus.camera.orchestrator` 负责把 stream 注册、摄像头生命周期调用和检测链输入串起来。

## 运行路径

常规 RTSP：

```text
CameraConfig(protocol=rtsp, source=rtsp://...)
  -> StreamRegistry.ensure_registered()
  -> go2rtc stream
  -> browser playback via go2rtc
  -> DetectionPipeline reads original or resolved RTSP source
```

USB 默认路径：

```text
CameraConfig(protocol=usb, source=0)
  -> usb_to_go2rtc_source()
  -> go2rtc opens physical device
  -> go2rtc exposes rtsp://127.0.0.1:<port>/<stream>
  -> CameraOrchestrator rewrites runtime source to RTSP
  -> DetectionPipeline reads go2rtc RTSP
  -> browser playback also uses go2rtc
```

文件开发路径：

```text
--dev-video
  -> generated AVI file
  -> protocol=file
  -> go2rtc disabled for that process
  -> DetectionPipeline reads local file
```

协议默认行为：

- USB：go2rtc 持有物理设备，检测链读取 go2rtc RTSP relay。
- RTSP：检测链读取原始 RTSP URL，浏览器预览通过 go2rtc relay。
- GigE：检测链通过 GigE SDK 读取，仍支持可选 exec preview。
- File：检测链读取本地文件，预览使用 pipeline latest frame。

## go2rtc 的双重角色

go2rtc 不是单纯的浏览器播放器配套。对 USB 摄像头，它通常先打开物理设备，再把它重定向为 RTSP 给检测链读取；这样浏览器和后端不会同时抢占同一个 USB 设备。

因此新增摄像头协议时必须检查：

- `src/argus/streaming/stream_registry.py` 是否能从配置生成 `StreamDeclaration`。
- `src/argus/streaming/go2rtc_manager.py` 是否能生成 go2rtc source。
- `src/argus/camera/orchestrator.py` 是否会把运行源改写给检测链。
- `src/argus/__main__.py` 的启动阶段是否在检测链打开摄像头之前完成注册。

## CameraManager 生命周期

`CameraManager` 是多摄像头边界：

- 每路摄像头对应一个 `CameraInferenceRunner`。
- 每个 runner 包装一个 `DetectionPipeline`。
- 摄像头循环运行在独立线程中，避免单路异常影响其它摄像头。
- 共享资源包括 YOLO 模型、推理线程池、ring buffer JPEG 编码线程池、可选的共享异常检测器。
- 状态通过 `get_status()`、`get_runner_snapshot()`、`get_wall_status()`、`get_backpressure_stats()` 暴露给 API 和 WebSocket。

## 采集质量与基线采样

`capture/quality.py` 负责基线采集阶段的画质过滤：

- 模糊检测。
- 亮度/曝光范围。
- 饱和像素比例。
- 熵阈值。
- SSIM 去重。
- 可选人员检测过滤。

`capture/baseline_job.py` 负责长任务形式的基线采集。采样策略位于 `capture/samplers.py`：

- `UniformSampler`：均匀采样。
- `ActiveSampler`：基于模型/异常分数挑选。
- `ScheduledSampler`：按时间表采样。

采集任务通过 `TaskManager` 向前端报告进度，并支持暂停、恢复、终止。

## Dashboard 相关 API

摄像头 API 位于 `/api/cameras`：

- `GET /api/cameras/json`：摄像头列表和状态。
- `GET /api/cameras/{camera_id}/detail/json`：单路详情、诊断、播放信息。
- `POST /api/cameras` / `PUT /api/cameras/{camera_id}` / `DELETE /api/cameras/{camera_id}`：配置管理。
- `POST /api/cameras/{camera_id}/start` / `stop`：单路启停。
- `POST /api/cameras/{camera_id}/mode`：切换检测模式。
- `GET /api/cameras/{camera_id}/snapshot`：快照。
- `GET /api/cameras/{camera_id}/stream`：MJPEG 回退流。
- `GET /api/cameras/wall/status`：视频墙聚合状态。

流媒体 API 位于 `/api/streaming`：

- `GET /api/streaming`：当前注册 stream。
- `GET /api/streaming/{camera_id}`：单路播放信息。
- `POST /api/streaming/{camera_id}/register`：手动注册或刷新 stream。

## 注意事项

- MJPEG 编码必须使用摄像头路由中的 `_STREAM_EXECUTOR` 专用线程池，不要把 `cv2.imencode()` 丢到默认线程池。
- BaseHTTPMiddleware 会缓冲流式响应，项目禁止使用。
- USB 摄像头真实硬件排查应先跑 preflight smoke，再跑完整业务闭环。
- go2rtc 不可用时应退化为 MJPEG 或不可用状态，而不是阻塞普通 API。
- 当前仍保留 `CameraManager.source_resolver`、`CameraSourceResolution`、`go2rtc_manager` 兼容函数，以及 pipeline 在没有注入 capture adapter 时的协议分支；清理这些 shim 前必须确认所有调用方都改为消费 `CameraRuntimePlan`、`StreamRegistry` 和 `CaptureFactory`。

## 相关测试

- `tests/unit/test_camera.py`
- `tests/integration/test_camera_manager.py`
- `tests/integration/test_camera_orchestrator.py`
- `tests/unit/test_go2rtc_manager.py`
- `tests/unit/test_stream_registry_contract.py`
- `tests/unit/test_preview_gateway_contract.py`
- `tests/unit/test_dashboard_streaming_contract.py`
- `tests/lint/test_mjpeg_dedicated_executor.py`
