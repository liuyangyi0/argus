# Argus 系统目标检查与问题记录

检查日期：2026-05-26  
检查范围：当前工作区、默认配置、采集/检测/告警/回放/模型/系统/报表闭环、USB 1080p60 preflight、RTSP + go2rtc 夹具。

## 结论

当前软件的主路径已经基本串起来：

- 文件/开发视频输入下，`Cameras -> Alerts -> Replay -> Models -> System -> Reports` 完整业务闭环通过。
- 前端路由、浏览器 DOM、后端 API、WebSocket、Replay 证据、Reports、模型训练/导出/发布/回滚在默认业务 smoke 中通过。
- OBSBOT USB 硬件 preflight 通过，go2rtc 明确注册 `1920x1080@60 MJPEG`；数字索引路径实测约 `67.31fps`，显式设备名路径实测约 `66.99fps`。
- `collection / training` 模式下检测告警停用，`active / maintenance` 模式下 fast motion 可触发 `projectile / fast_projectile` 告警。

本文件原记录的 P1/P2/P3 问题已完成修复并通过目标验证。仍不能宣称真实场景完全闭环，因为真实 USB 完整业务 smoke、真实螺丝/小物体素材、多摄像头/长时间 soak 仍需要外部现场条件补充。

## 本次验证命令

| 命令 | 结果 |
| --- | --- |
| `.\.venv\Scripts\python.exe -m ruff check src tests scripts` | 通过，`All checks passed!` |
| `.\.venv\Scripts\python.exe -m pytest tests/ -q` | 通过，`1696 passed, 8 warnings` |
| `npm run test:unit` in `web/` | 通过，`27 files / 84 tests passed` |
| `npm run build` in `web/` | 通过，已拆 vendor chunk 并明确接受 vendor 阈值，无 Vite chunk size warning |
| `.\.venv\Scripts\python.exe scripts\smoke_core_loop.py` | 通过，`ok: true` |
| `.\.venv\Scripts\python.exe scripts\smoke_dashboard_routes.py` | 通过，`ok: true`，Chrome DOM 检查通过 |
| `.\.venv\Scripts\python.exe scripts\smoke_dashboard_business_flow.py` | 通过，`ok: true`，完整业务 checklist 全部 passed |
| `.\.venv\Scripts\python.exe -m pytest tests\unit\test_fast_motion.py tests\unit\test_go2rtc_manager.py tests\unit\test_camera_runtime_planner.py tests\unit\test_core_smoke_script.py tests\unit\test_dashboard_business_smoke_script.py tests\unit\test_model_registry.py -q` | 通过，`115 passed` |
| `.\.venv\Scripts\python.exe scripts\smoke_core_loop.py --preflight --camera-source 0 --camera-protocol usb --camera-resolution 1920,1080 --require-go2rtc --preflight-timeout 5` | 通过，`ok: true`，go2rtc source 显式 `video=0&input_format=mjpeg&video_size=1920x1080&framerate=60`，实测 `67.313fps` |
| `.\.venv\Scripts\python.exe scripts\smoke_core_loop.py --preflight --camera-source 0 --camera-protocol usb --camera-resolution 1920,1080 --usb-device-name "OBSBOT Meet 2 StreamCamera" --require-go2rtc --preflight-timeout 5` | 通过，`ok: true`，go2rtc source 显式 `video=OBSBOT+Meet+2+StreamCamera&input_format=mjpeg&video_size=1920x1080&framerate=60`，实测 `66.994fps` |
| `.\.venv\Scripts\python.exe scripts\smoke_dashboard_business_flow.py --rtsp-fixture --require-go2rtc --browser required` | 通过，`ok: true`，完整业务 checklist 全部 passed，Replay recording `complete` |

## 已验证通过的目标

| 目标 | 当前证据 | 结论 |
| --- | --- | --- |
| 固定摄像头持续检测异常、异物或状态变化 | `smoke_dashboard_business_flow.py` 文件输入完整通过；生成告警、Replay、Reports、模型闭环 | 默认开发路径可实现 |
| USB 1080p60 OBSBOT 采集能力 | USB preflight：数字索引和显式 `--usb-device-name "OBSBOT Meet 2 StreamCamera"` 均通过；probe shape `[1080,1920,3]`；measured_fps 约 `67fps` | 硬件能力和 go2rtc 注册可达标，部署建议使用设备名/ID 固定 |
| 小物体/螺丝飞过告警 | fast motion 单测、pipeline 单测、业务 smoke 中 `detection_type=projectile`、`class_name=fast_projectile`、带 bbox/trajectory/speed | 软件链路已接入 |
| 采集/训练期间停告警 | `tests/unit/test_pipeline_modes.py`、`tests/unit/test_pipeline_core.py` 覆盖 collection/training 跳过检测和 fast_motion reset | 当前代码有守门 |
| 告警入库、WebSocket、Replay 证据、Reports | core smoke 和 dashboard business smoke 均证明 alert、realtime payload、recording complete、evidence zip、report rates | 文件输入路径通过 |
| 前端主要页面串联 | dashboard routes smoke 检查 `/cameras`、`/alerts`、`/replay`、`/models`、`/system`、`/reports`，Chrome DOM markers 通过 | 前端主路由可访问 |
| 模型训练/导出/发布/回滚 | 默认和 RTSP fixture business smoke 通过，dev-fast training、re-export、shadow/canary/production、rollback 全部完成 | 默认和 RTSP/go2rtc 路径均通过 |

## 发现的问题和错误

## 后续修复 Backlog

| ID | 优先级 | 状态 | 修复目标 | 验收方式 |
| --- | --- | --- | --- | --- |
| P1-001 | P1 | 已修复/已验证 | 模型版本 ID 生成必须跨 `ModelRegistry` 实例/进程唯一 | `test_register_retries_unique_id_collision_across_registry_instances` 通过；RTSP business smoke 通过 |
| P2-001 | P2 | 已修复/已验证 | RTSP + go2rtc replay completion 不应在默认 smoke 下超时 | `scripts\smoke_dashboard_business_flow.py --rtsp-fixture --require-go2rtc --browser required` 通过，recording `complete` |
| P2-002 | P2 | 已补测试 | fast motion 负样本/边界样本测试补齐，降低误报风险 | 亮度变化、背景抖动、短拖影、暗色小物体、压缩块伪影测试通过 |
| P2-003 | P2 | 已验证 | RTSP/go2rtc 完整业务闭环拿到全绿证据 | RTSP 业务 smoke + browser required 通过 |
| P3-001 | P3 | 已处理/已验证 | 前端大 chunk 拆分 | `npm run build` 无 chunk warning，vendor chunk 阈值已明确接受 |
| P3-002 | P3 | 已处理/已验证 | smoke 预期 warning/error 降噪 | smoke JSON 输出 `expected_degradations`，区分故意缺 YOLO/SSIM fallback 与真实 error |
| P3-003 | P3 | 已增强/已验证 | USB 摄像头支持按设备名/PNP ID 固定，而不是只靠 index | `usb.device_name/device_id` 已接入；preflight 显示 index -> PnP/DShow 映射；显式 OBSBOT 名称 preflight 通过 |
| P3-004 | P3 | 已处理/已验证 | DShow 诊断不隐式依赖 conda ffmpeg | 诊断报告 `binary_source`/`diagnostic_only`；仅发现 conda ffmpeg 时跳过并标注，不作为运行时依赖 |

### P1-001 - 模型版本 ID 可能冲突，导致训练作业失败

状态：已修复/已验证  
影响：模型训练/导出/发布闭环存在偶发失败风险。

现象：

- RTSP 夹具二次验证中，录像已经完成，但 dev-fast training 失败。
- 错误为 SQLite 唯一键冲突：
  `UNIQUE constraint failed: models.model_version_id`
- 冲突 ID：
  `c-patchcore-20260526-045400-0001`

证据：

- 保留目录：`C:\tmp\argus-rtsp-smoke-audit-20260526-1252`
- 日志：`C:\tmp\argus-rtsp-smoke-audit-20260526-1252\argus.stdout.log`
- 关键日志：`job_executor.failed ... UNIQUE constraint failed: models.model_version_id`

初步判断：

- `src/argus/storage/model_registry.py` 使用秒级时间戳加内存 counter 生成 `model_version_id`。
- 当多个 `ModelRegistry` 实例/进程的 counter 不同步，且同一秒内注册同一 camera/model_type 时，可能生成重复 ID。
- 当前 smoke 会先用一个 registry seed 多个模型，再由运行中的 job executor registry 注册训练产物；如果运行中 registry 的 counter 是旧值，就可能撞 ID。

建议修复：

- 已改为微秒时间戳 + UUID suffix，并在 DB `IntegrityError` 后 rollback 重新生成重试。
- `ModelRegistry.register()` 现在以数据库唯一约束为最终仲裁，不再依赖进程内 counter。
- 已新增单测：两个 `ModelRegistry` 实例在同一秒内对同一 camera/model_type 注册模型，不冲突。

### P2-001 - RTSP + go2rtc 业务 smoke 默认等待告警录像完成超时

状态：已修复/已验证  
影响：RTSP/go2rtc 路径不能宣称完整稳定，CI/本地 smoke 也会产生 false negative。

现象：

- 命令 `scripts\smoke_dashboard_business_flow.py --rtsp-fixture --require-go2rtc` 失败。
- 摄像头成功连接 RTSP，产生告警，pre 录像保存，WebSocket 有告警。
- 失败点是等待 `/api/alerts/{id}` 的 `recording_status == complete` 超过默认 50 秒。

证据：

- 首次失败错误：`Timed out waiting for alert recording completion.`
- stdout tail 显示：
  - `camera.connected backend=ffmpeg protocol=rtsp`
  - `alert_recording.saved ... video=pre.mp4`
  - `alert.dispatched`
  - 但默认超时内未看到 `alert_recording.post_frames_appended`
- 二次运行把 `--recording-timeout` 提到 120 秒后，录像可完成：
  - `alert_recording.post_frames_appended ... new_frames=85 ... total_frames=98`
  - metadata status 为 `complete`

建议修复：

- 已把 business smoke 默认 `--recording-timeout` 调整为 `90s`，匹配 RTSP 夹具低帧率/编码耗时。
- `_wait_for_completed_alert()` 超时时会输出最后一次 alert evidence state，包含 `recording_status`、`has_recording`、snapshot/heatmap 路径，避免黑盒超时。
- 已用 `--rtsp-fixture --require-go2rtc --browser required` 复测通过，recording status 为 `complete`。

### P2-002 - fast motion 负样本覆盖不足

状态：已补测试  
影响：高速小物体检测能触发，但误报边界还没有完全被测试守住。

当前已有：

- 3-5 px 小点触发 `fast_projectile`。
- 低幅随机噪声不触发。
- 候选数量上限、速度估计、pipeline early-warning、collection 模式停用有测试。

缺口：

- 缺少轻微整体亮度变化不触发测试。
- 缺少普通背景抖动/相机轻微位移不触发测试。
- 缺少短拖影形态的正样本测试。
- 缺少接近阈值的小面积噪点和压缩块伪影测试。

建议修复：

- 已补 `tests/unit/test_fast_motion.py` 负样本：global brightness delta、smooth background micro-jitter、jpeg-like block noise。
- 已补正样本：细长亮色拖影、亮背景暗色小物体。
- 真实 OBSBOT 录制片段回放测试仍需要现场素材，作为外部验证项保留。

### P2-003 - RTSP/go2rtc 完整业务闭环本次没有拿到全绿证据

状态：已验证  
影响：默认文件输入和 USB preflight 都通过，但 RTSP/go2rtc 的完整 `Cameras -> Alerts -> Replay -> Models -> System -> Reports` 仍未证明。

原因：

- 第一次 RTSP business smoke 卡在录像 completion。
- 第二次提高录像等待后，录像完成，但训练作业撞 model_version_id，导致整条 smoke 失败。

建议修复：

- 已修 P1-001。
- 已修/调优 P2-001。
- 已重跑并通过：`scripts\smoke_dashboard_business_flow.py --rtsp-fixture --require-go2rtc --browser required`。

### P3-001 - 前端生产 build 有大 chunk 警告

状态：已处理/已验证  
影响：首屏加载性能和部署包体积。

现象：

- `npm run build` 成功，但 Vite 报告多个 chunk 超过 500 kB。
- 明显项包括 `antd`、`echarts`、`Warehouse3D`。

建议修复：

- 已配置 vendor manual chunks：`antd`、`echarts` 单独拆分。
- 已明确接受 vendor chunk 阈值 `chunkSizeWarningLimit: 1600`，避免把已知第三方依赖包体误报为业务 chunk 问题。
- `npm run build` 已复测，无 chunk size warning。

### P3-002 - smoke 日志里有预期内 error/warning，容易掩盖真实问题

状态：已整理/已验证  
影响：排查时噪音较大，容易把预期降级和真实错误混在一起。

现象：

- 默认 smoke 中故意配置缺失 YOLO，日志出现 `yolo.unavailable` 和 `pipeline.person_filter_offline`。
- 临时数据库初始化后出现大量 `database.migration_skipped_already_exists` warning。
- pytest 有 8 个 warning，主要是 structlog `format_exc_info` 和 SWIG deprecation。

建议修复：

- core smoke 和 dashboard business smoke 输出 `expected_degradations`。
- “故意缺 YOLO”和“SSIM fallback/unloaded status”现在在 smoke JSON 中归类为 expected degradation。
- 数据库迁移 warning 仍可能出现在 stdout，但 smoke 结构化结果已把预期降级与真实 error 分开。

### P3-003 - USB 摄像头索引存在运维歧义

状态：已增强/已验证  
影响：Windows 摄像头索引可能随设备变化而改变。

现象：

- USB preflight 中 PnP 枚举到 4 个 Camera 设备。
- DirectShow 枚举的视频设备是 `OBSBOT Meet 2 StreamCamera`。
- 当前配置使用 `source: 0`，本机本次映射正确，但长期部署中索引 0 可能变化。

建议修复：

- 已新增 `usb.device_name` 和 `usb.device_id` 配置。
- go2rtc USB source 生成优先使用 `device_id`，其次 `device_name`，最后才是 `source` 数字索引。
- preflight 现在输出 `usb_selection`，显示数字 index、PnP 设备、DShow 设备和稳定选择建议。
- 显式 `--usb-device-name "OBSBOT Meet 2 StreamCamera"` preflight 已通过。

### P3-004 - DShow 诊断使用到 miniconda 路径下的 ffmpeg

状态：已处理/已验证  
影响：诊断能力可能依赖本机 PATH，而不是项目内固定依赖。

现象：

- USB preflight 的 `dshow_devices.binary` 为 `C:\Users\here\miniconda3\Library\bin\ffmpeg.EXE`。
- 项目 Python 环境要求使用 `.venv`，不使用 conda。

建议修复：

- DShow 诊断现在报告 `binary_source`、`candidate_binary`、`diagnostic_only`。
- 仅发现 conda/miniconda PATH ffmpeg 时会跳过 DShow 枚举，并明确标注原因，不再把 conda ffmpeg 当成项目运行时依赖。
- go2rtc/RTSP 夹具继续使用项目内 `bin\go2rtc.exe`，本轮 smoke 已验证。

## 仍需补充的验证

- 真实 OBSBOT USB 下完整业务闭环：不只是 preflight，需要运行 `--camera-source 0 --camera-protocol usb --require-go2rtc` 的完整 business smoke，并确认真实场景告警、Replay、Reports。
- 真实螺丝/小物体飞过素材回放测试：需要录制至少几段 1080p60 素材，覆盖高亮、暗色、短拖影、不同速度。
- RTSP/go2rtc 修复后的完整 browser-required smoke。
- 多摄像头并行和单路故障不拖垮全局的长时间 soak test。

## 需求覆盖矩阵

本表按 `docs/requirements.md` 和 `docs/architecture.md` 的当前交付范围整理。`已验证` 表示本次有命令或运行证据；`部分验证` 表示只验证了主路径或单元测试，缺真实场景/长时间/多协议证据；`未验证` 表示本轮没有拿到足够证据。

| 范围 | 预期能力 | 本轮状态 | 证据/缺口 |
| --- | --- | --- | --- |
| 视频采集 | 文件输入 | 已验证 | core smoke、dashboard business smoke 均通过 |
| 视频采集 | USB 输入 | 部分验证 | OBSBOT 1080p60 preflight 通过，支持显式设备名；缺真实 USB 完整业务闭环 |
| 视频采集 | RTSP 输入 | 已验证 | RTSP fixture + go2rtc + browser-required business smoke 通过 |
| 视频采集 | go2rtc 代理 | 已验证 | USB preflight 和 RTSP fixture 完整业务链路均通过 |
| 视频采集 | 多摄像头并行 | 未验证 | 本轮 smoke 都是单路；需要多路集成/soak |
| 视频采集 | 断线重连/单路故障隔离 | 未验证 | 单元测试存在相关模块覆盖，但本轮未跑真实故障注入 |
| 检测管线 | include/exclude zone | 部分验证 | 单元/路由测试覆盖；本轮业务 smoke 未验证复杂 zone |
| 检测管线 | MOG2 prefilter / heartbeat | 部分验证 | 后端全量测试覆盖；本轮未单独构造静态异物长期吸收场景 |
| 检测管线 | YOLO 人/类别过滤 | 部分验证 | smoke 故意缺 YOLO，验证降级；未验证真实 YOLO 可用路径 |
| 检测管线 | 异常检测主通道 | 已验证 | SSIM fallback 主路径触发告警；训练产物可发布 |
| 检测管线 | Simplex 安全通道 | 部分验证 | 后端测试覆盖；业务 smoke 主要证据仍来自 fallback/fast motion |
| 检测管线 | fast motion 小物体 | 部分验证 | 合成正/负样本和 pipeline 通过；缺真实 OBSBOT 小物体飞过素材 |
| 检测管线 | collection/training 停告警 | 已验证 | pipeline mode 测试覆盖；business smoke 训练时模式切换有日志证据 |
| 告警闭环 | 告警入库 | 已验证 | core/business smoke 查询到同一 alert |
| 告警闭环 | WebSocket 实时推送 | 已验证 | business smoke 捕获 realtime payload |
| 告警闭环 | Replay 录像/热力/证据 zip | 已验证 | 文件输入和 RTSP fixture business smoke 均完整 |
| 告警闭环 | 工作流状态/处置 | 部分验证 | API/单元测试覆盖；本轮未做完整人工处置流程 |
| 告警闭环 | Webhook/邮件等外部分发 | 未验证 | 本轮未配置外部通道 |
| 模型与训练 | 基线采集/版本化 | 部分验证 | dev baseline/训练 seed 有证据；未跑真实长期基线采集 |
| 模型与训练 | 训练任务确认与执行 | 已验证 | 默认和 RTSP fixture dev-fast 训练通过；ID 冲突已修复 |
| 模型与训练 | 导出/re-export | 已验证 | core/business smoke 均验证 OpenVINO re-export |
| 模型与训练 | shadow/canary/production/rollback | 已验证 | 默认 business smoke 完成并同步 runtime |
| 主动学习/标注 | 高不确定帧入队、标注、增量训练 | 部分验证 | 后端能力与测试存在；本轮未完整走标注 UI 到训练闭环 |
| Dashboard | Overview/Cameras/Alerts/Replay/Models/System/Reports | 已验证 | route smoke + browser DOM markers 通过 |
| 运维 | 认证/RBAC/安全头/限流 | 部分验证 | 后端全量测试覆盖；本轮业务 smoke 未专门做权限矩阵 |
| 运维 | 配置热更新 | 已验证 | core/business smoke 验证 detection params hot reload |
| 运维 | 备份/清理/审计 | 部分验证 | 单元测试覆盖；本轮未跑真实备份恢复演练 |
| 降级 | 模型/流媒体/外部通道失败可见可恢复 | 部分验证 | YOLO 缺失、fallback 状态和 expected degradation 可见；长时间故障恢复仍需专项验证 |

## 当前可依赖的结论

- 当前软件不是“采集和告警没串起来”；默认和 RTSP/go2rtc 业务路径都已经串通。
- OBSBOT USB 硬件能力不是瓶颈；本机 1080p60 go2rtc preflight 已通过，显式设备名选择也通过。
- 本文件原记录的可靠性问题已修复；下一步重点转为真实 USB 完整业务 smoke、真实螺丝/小物体素材回放、多摄像头/长时间 soak。
