# 多分析任务平台：P3—P6 实现与使用说明

日期：2026-10-23。承接 [P0—P2](multi-task-platform-p0-p2.md)。本文记录可核实的代码能力和交付限制，不是模型质量或生产验收报告。

后续对抗审查及修复见 [P3—P6 审查记录](../audits/multi-task-platform-p3-p6-adversarial-2026-10-23.md)，其中的最终回归记录晚于本文下方首轮实现验证。

源生命周期、持久化部署、重启恢复、历史配置恢复及多角色选择的后续实现见 [部署补齐记录](multi-task-deployment-completion-2026-10-24.md)。下文保留首轮交付边界和当时测试结果；涉及启动和部署的现行写入口以该补齐记录为准。

## 当前交付边界

同一个相机源可保留原异物任务，并配置多个独立旁路任务。采集源只打开一次；新任务不竞争读取设备、不修改源视频、不接入旧告警分发器，也不改变现有 SceneGuard 和 `ModelReleaseService` 的生产权限。

| 阶段 | 本轮实现 | 仍不代表 |
| --- | --- | --- |
| P3 任务与执行包 | 任务增改归档、不可变配置版本、候选模型登记、camera scope、SHA 冻结及幂等收据 | 任意模型/插件上传即执行 |
| P4 独立执行 | 固定适配器、spawn 子进程、有界 IPC、加载/推理超时、资源准入、失败清理重试 | GPU 吞吐验收或操作系统级内存硬隔离 |
| P5 数据工作流 | 周期采集、精确帧血缘、图片审核、预标注、离线交换、不可变数据集 | 自动标注等于人工批准，或已上传 CVAT/FiftyOne |
| P6 训练衔接 | 配方、人工批准、外部训练清单、候选回登记、生产 readiness | 内置目标检测/姿态训练器，或新的生产发布入口 |

所有新模板固定 `production_ready=false`、`production_allowed=false`。旁路运行成功、候选登记成功、训练清单导出成功都不是生产授权。原异物任务仍通过原发布门和 SceneGuard；不得以旁路功能替换其基线、校准或签名要求。

## 模板与模型格式

| 模板 | 模型角色与约束 |
| --- | --- |
| `object_detection.v1` | `detector`，`onnx_detection.v1`，YOLOv8 raw 输出 |
| `pose_estimation.v1` | `pose`，`onnx_pose.v1`，单 person 类、COCO-17 |
| `ppe_compliance.v1` | `detector`，必须显式含 person / helmet / bare_head；缺少安全帽不是裸头证据；歧义不判违规 |
| `ppe_compliance_dual.v1` | `person_detector` 与 `head_detector` 两个固定角色，分别要求 person 和 helmet / bare_head；全部角色准备后才运行，仍仅旁路 |
| `fast_motion.v1` | 无模型权重，固定 CPU 算法；有序窗口，默认目标 60 FPS，实际采样不足即阻断 |
| `legacy_anomaly.v1` | 原异物链兼容投影，不允许通过新 CRUD 改写或发布 |

模型只允许已存在于受控 `models_root` 内的 ONNX 文件。登记读取实际 SHA；冻结和解析再次核验，不按目录自行发现权重。外部 tensor 文件、自定义算子域、动态输入输出及非声明格式拒绝执行。支持单张静态 NCHW / float32 / RGB 输入和单个 raw 输出；不是任意 ONNX 的通用加载器。

检测模型描述示例（`input_size` 为 H、W，labels 顺序必须与实际输出一致）：

```json
{
  "name": "Object detector",
  "adapter_id": "onnx_detection.v1",
  "output_schema": "yolov8_detection.v1",
  "labels": ["person", "foreign_object"],
  "input_size": [640, 640]
}
```

执行依赖是可选的 `task-onnx-cpu` 或 `task-onnx-cuda`，部署环境只选一种 ORT 分发，不能同时安装。要求 ORT ≥ 1.23.2，构造时禁回退；CUDA 还禁止节点落到 CPU，允许 ORT 的隐式 CPU provider 注册不等于允许 CPU 执行。依据为 [ORT 构造实现](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/python/onnxruntime_inference_collection.py) 和 [会话初始化实现](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/session/inference_session.cc)。GPU 依赖、驱动和真实图兼容性仍需在目标机器验证并锁定；本轮没有安装或运行这些引擎，现有容器依赖锁未包含新的可选 ORT 分发。

CPU/CUDA 必须在任务 `resources.device` 中显式声明。CUDA 不可用、依赖缺失、加载失败均明确受阻，不自动下载或退回 CPU、OpenVINO、SSIM。快速运动算法仅支持 CPU，要求 CUDA 会被拒绝。

## 使用流程

入口：相机详情 → 分析任务 → 多任务工作台（管理员/工程师）。viewer 不显示写入工作台，服务端独立鉴权。

1. 选择模板并创建命名任务。默认配置由模板给出，进阶 JSON 拒绝未知字段。
2. 对模型型任务，人工登记受控 ONNX 文件，选择兼容模型；固定快速运动算法无需权重。
3. 冻结当前执行包。修改配置后必须重新冻结；旧包不能按新版本启动。
4. 在有合适硬件的环境独立启动视频源，再提交任务持久化部署。`202` 表示意图已登记，不代表执行包已应用；以部署观察状态为准。本机普通验证不执行这一步的真实计算。
5. 查看独立任务状态和最近旁路结果；停止/重试清理只停止该任务，不关闭相机预览。
6. 新建绑定当前任务版本的数据流程，启动周期采集；或设置 `collection.enabled=true`，随该执行会话自动建立幂等数据流程。
7. 停止采集并对账，逐帧审核，冻结数据集，提出外部训练并由人工批准导出。
8. 外部训练产物回到受控 models 目录，按交接 package SHA 核验并登记候选，再创建新执行包做旁路对比。生产仍为 NO-GO。

默认禁止自动采集；开启示例：

```json
{
  "collection": {
    "enabled": true,
    "policy": "periodic",
    "interval_seconds": 10.0,
    "max_samples": 1000,
    "max_storage_mb": 1024
  }
}
```

首轮仅执行 periodic；2026-10-28 已补 `object_detection.v1` 的 events 策略，只收集冻结游标后的同执行/同源已提交事件关键帧，见 [事件入样与训练候选](object-detection-lifecycle-v1.md#验证态闭环补齐事件入样与训练候选2026-10-28)。这不是事件前后录像；disagreement / low_confidence 仍明确阻断，不静默替代为周期采集。DB 流程状态与 `collection_runtime` 分开显示；首次写入结果未知、清单落盘失败和清理失败均保留所有权供对账，不能换一个流程身份假装重试成功。

## 帧身份、审核和训练血缘

采样来自同一 FrameHub 的独立订阅。只写受控衍生 PNG，保留 camera、source session、epoch、原始帧号、PTS/time base 和 provenance；没有原始时间证据时保持缺失，不能用计数器补造。图片内容/结构、SHA 和清单一致性均需核验。

预标注只接受最近有界结果中的精确同帧、同任务版本/配置和同源身份，不按“最近时间”猜匹配；没有匹配则明确不可用。快速运动结果不冒充通用检测标注。所有自动建议均为 `needs_review`，不继承旧人工批准。

人审提交携带流程 revision、sample ID 和图片 SHA。页面刷新不能把旧表单自动套上新 revision；版本变化保留草稿但要求明确重载。原有备注和审核历史保留。审核、标注回传或报告变化会使旧 dataset / training approval 失效。

冻结需要每个样本完成人工复核；批准样本至少有两个独立源会话，训练集与验证集按会话隔离。旧数据缺少 manifest SHA 时不补造，不可直接用于训练批准。空 label schema、错误配方、跨任务数据、过期批准或 package SHA 不符均拒绝。

新任务的 CVAT/FiftyOne 交换是任务图片 envelope（schema、binding、样本 SHA、标注及摘要），不是对两者 SDK 上传成功的声明，也不能直接当作其原生模型训练格式。外部程序需按明确 schema 转换并回传；既有视频复核工具独立保留，不把视频帧号套到任务图片上。

训练配方目前仅 `external_detection.v1`（目标检测/PPE）、`external_pose.v1`（姿态）；固定快速运动算法没有可登记的训练权重，不列为可训练模板，历史不兼容提案也不得新批准导出。禁止自动调用旧 anomalib 训练器替代。人工批准的是外部训练交接；清单导出不表示训练已开始/完成。候选登记也不修改旧 registry 的 active/production 状态。

## 运行时和故障边界

- 任务配置、执行包、worker execution、source session/epoch 都是独立身份。更新、归档、相机重启和停止后的迟到结果不能覆盖当前状态。
- `scene.policy=require_profile` 当前复用正在运行的 legacy SceneGuard 的实际处理证据，绑定源 session/epoch、profile、参考图和模型摘要。消费暂停、重连后未重验、证据过期或观测忙时明确阻断；预览帧身份不能充当证据。新任务只接受 stable/raw stable，尚未对原始输入应用 homography，因此不借用 legacy 的 alignable 放行。新任务不写旧 SceneGuard 的批准、阈值或恢复状态。
- 结果只进入有界内存旁路列表，不进入生产告警。结果提交使用非阻塞 DB 权威栅栏；锁忙可拒收，不能阻塞相机源。
- 默认最多两个新 worker，总声明预算 4096 MB。`memory_limit_mb` 是准入/缓冲预算，不是 OS 内存硬限额，状态明确 `memory_limit_enforced=false`。
- 清理须确认子进程退出、共享内存释放和任务注销；失败保留 lease、task identity 和 cleanup_pending，不允许同 ID 无界再启动。
- 有序窗口丢帧和不满足快速运动 FPS 时持续阻断，需显式停止/重启；不得把丢失时序的“无检测”当正常。
- REST 读取不加载模型、不启动设备、不上传外部服务；图片使用同源鉴权和 private no-store。SDK/路径原始异常不返回客户端。

## 普通功能验证与未验事项

只使用受保护的本地 ordinary/mock 入口，不运行真实训练、识别、相机采集或源视频：

```powershell
.venv\Scripts\python.exe scripts\run_local_functional_tests.py --suite platform
.venv\Scripts\python.exe scripts\run_local_frontend_tests.py unit
.venv\Scripts\python.exe scripts\run_local_frontend_tests.py build
.venv\Scripts\python.exe scripts\generate_openapi_types.py --check
.venv\Scripts\python.exe scripts\run_functional_browser_checks.py --grep managed
```

新增浏览器验收使用临时数据库和禁止计算的服务，覆盖 1920×1080、1440×900、1024×768 及手机尺寸的任务创建、执行包、数据流程和 viewer 权限，不把这些结果当真实推理通过。

### 2026-10-23 验证记录

| 范围 | 可核实结果 |
| --- | --- |
| 核心普通回归 `--suite core`（本轮较早运行） | 1297 passed |
| 最终平台专项 `--suite platform` | 320 passed |
| Dashboard 普通回归 `--suite dashboard` | 244 passed；3 条既有 structlog 格式警告 |
| 最终静态/迁移守门 `--suite lint` | 74 passed，1 skipped；跳过项不计为通过 |
| 最终前端 Vitest 全套 | 49 个文件、293 项通过 |
| 最终前端类型检查与构建 | 通过 |
| 新任务四视口浏览器专项 | 8 passed；覆盖创建、冻结执行包、数据流程、生产 NO-GO 和 viewer 拒绝 |
| 四视口完整普通浏览器回归 | 初跑 61 passed、6 项不适用视口 skipped、1 项平板焦点失败；该失败随后已修复 |
| 修复后的平板完整普通浏览器回归 | 17 passed，包含原失败的真实 Tab / Shift+Tab / Escape 及角色权限用例 |
| OpenAPI 生成类型一致性、改动 Python Ruff、依赖锁一致性、`git diff --check` | 通过 |

平板失败来自 Drawer 动画完成回调重置已经位于弹层内的键盘焦点；修复保留现有焦点，并阻止旧关闭动画在重新打开后抢焦点，没有放宽 E2E 断言。修复后重跑的是完整平板项目，不声称又执行了一次四视口全套。

构建与浏览器曾并行访问 `dist`，出现 Windows 文件占用和资源缺失；已中止受影响轮次，停止服务后顺序重建，再运行上述回归。原无界前端并发入口也曾发生导入超时，最终使用受控入口的两个 worker 全套通过。相关环境失败不计为功能通过。

验证仅允许普通业务、临时 SQLite 和模拟识别头/worker，不包含全量训练、识别或真实视频测试集。本轮没有提交或推送代码，没有修改源视频或放宽旧发布门。

真实多模型吞吐、显存占用、CUDA 恢复、机位移动和重连实测、模型 recall/FP/delay、外部训练端到端、签名评估及可信 soak 均待目标机器验收。新模板的生产评估/签名/发布协议和专属训练执行器尚未实现；在这些能力完成前，本轮只交付旁路和数据工作流，不宣称 P3—P6 完整生产闭环。
