# Windows GPU 多任务隔离验证

该流程只能在明确授权的 Windows NVIDIA GPU 测试机执行。本机普通验证不运行这些脚本。不包含源视频读取、训练、生产审批或业务识别效果验收。

## 隔离环境与制品

- 使用 Python 3.11 独立 venv、独立源码快照、数据库、模型及结果目录，不覆盖现有部署。
- Windows 须安装微软官方 x64 VC++ v14 可再发行运行库；先验证安装器 Authenticode 签名。缺失时 Torch 可能以 `WinError 126 / c10.dll` 失败，不能误判为 CUDA 不可用并切换 CPU。参考 [微软官方运行库下载](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。
- 连接凭据保存在仓库外的本机私有文件；不放入 Git、文档、快照、结果或远端脚本。
- `scripts/create_validation_snapshot.py` 生成逐文件 SHA256 清单。目标机先核对归档 SHA、清单源摘要、全部文件大小与 SHA，并拒绝额外文件或逃逸路径，再执行快照代码。
- 官方运行包下载可以续传；未完成大小及官方 SHA 核验的 `.part` 文件不得安装。
- Windows 验证组合为 Torch 2.10.0 / torchvision 0.25.0 的 cu128 分发、ONNX 1.17.0、ORT GPU 1.23.2。不得同时安装 `onnxruntime` CPU 分发；NVIDIA 部署不得静默降级。
- 必须先完成 `pip check`，记录实际依赖版本。安装包存在、GPU 被驱动枚举到或者 provider 名称可见，都不能单独算 CUDA 执行通过。

项目依赖按 `requirements/runtime-constraints.txt` 约束安装。完整 Torch CUDA 轮子安装并校验后，再安装项目与 `task-onnx-cuda` 可选依赖，避免解析器先下载 CPU Torch。安装本身不运行训练或模型识别。

### 浏览器地址与执行位置

通过 SSH 本地端口转发访问测试机时，浏览器地址可以是 `http://127.0.0.1:18808`：本机回环端口只是隧道入口，HTTP 请求转发给测试机回环地址上的服务，不代表本机在训练或推理。页面渲染在本机，摄像头采集、模型加载、训练、校准和识别仍应在测试机进程执行。

验证时分别记录隧道监听进程、远端服务 PID、实际解释器/源码快照及远端 CUDA 执行日志。不能只凭页面能打开、端口号或 GPU 被枚举到就认定执行位置与后端正确。无需为消除地址误解而把测试服务改成对公网或整个局域网监听。

Windows SSH 启动的受控服务还需独立的持久日志：把标准输出/错误写到验证目录，避免原 SSH 输出通道关闭后日志写入失败影响请求。保留进程所有权、限时退出和完整清理机制；不能把普通后台启动成功当成服务会跨 SSH 会话存活的证据。

## 真实运行时验证

在目标机的快照工作目录，使用目标 venv 的 Python：

```powershell
python scripts/run_gpu_task_runtime_smoke.py --execute --output-root C:\ArgusValidation\Results\gpu-runtime-new
python scripts/run_gpu_task_restart_smoke.py --execute --output-root C:\ArgusValidation\Results\gpu-process-restart-new
```

示例输出目录必须是已有父目录下的全新绝对路径，脚本拒绝覆盖。脚本生成自己的动态像素相关 ONNX 计算图和内存源，不下载识别权重，不打开摄像头或源视频。新任务始终为 shadow，`production_allowed=false`、`alerts_emitted=false`。

### 执行与任务级验证

`run_gpu_task_runtime_smoke.py` 要求真实 ORT profile 中的 MatMul/Gemm 等算子使用 CUDA，禁止构造时和节点级 CPU 回退。实际 spawn worker 也必须显示 CUDA provider，不能借父进程预加载或者测试脚本改 PATH 才运行。

覆盖双任务同帧与结果身份隔离、停止一个任务保留另一个、同执行包重新部署、完整执行包升级与显式回滚、数据库连接及协调器重建、已停止意图不复活、共享内存和所有权清理。

这里的协调器重建不是进程重启；进程级证据由第二个脚本单独产生。

### 进程级恢复与故障隔离

`run_gpu_task_restart_smoke.py` 使用三个独立控制进程，以及生产 `IsolatedTaskWorker` 的真实 CUDA 子进程：

1. 两个任务 applied 后，非协作终止测试控制进程，观察 worker 是否自行因管道断开退出。
2. 新控制进程重新打开同一数据库，只恢复已授权执行包；源会话和 worker execution 身份必须变化。
3. 终止一个任务的真实 GPU worker，要求它恢复授权版本，而另一个任务的 worker 身份保持不变。
4. 恢复后升级与回滚任务 A，任务 B 不重启。
5. 持久化停止 A 后再重启控制进程，A 不得复活，B 恢复。

测试专用 Windows Job Object 只容纳自身控制进程及后代，作为超时清理兜底。观察 controller crash 后旧 worker 自行退出时保持 Job 打开，不能靠关闭 Job 杀掉它们再宣称生产退出机制通过。测试不修改系统服务、其他用户进程或安全策略。

相关接口依据为 [Microsoft Job 分配和嵌套规则](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject)。Windows CUDA DLL 发现使用 [ORT 官方预加载接口](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls)，不永久改 PATH，不从模型目录加载自定义 DLL。

## 结果判定

只接受进程真实退出码、`result.json` 中每项检查、未验证项、CUDA profile、实际 worker 身份及数据库收据相互一致的证据。超时、依赖缺失、清理未确认或者结果文件缺失都不能算通过。

这些检查不测量异物、安全帽、姿态或快速运动任务的 recall、FP/hour、P95 delay，也不是整机断电、GPU 驱动重置或 72 小时 soak。SceneGuard、baseline v3、校准、人工审批和可信 soak 门禁保持不变；生产仍须独立验收。

## 2026-10-25 首轮硬件记录

- RTX 3060 上 Torch `2.10.0+cu128` / CUDA 12.8 张量运算通过，ORT GPU `1.23.2` 的动态 MatMul 等 profile 仅包含 CUDA 算子执行；CPU provider 注册本身不作为执行证据。
- 运行时 8 项、控制进程及 worker 故障恢复 6 项全部通过，未验证项为空。完整依赖安装和 `pip check` 通过，没有安装 CPU ORT 分发。
- 完整 SDK 安装后的摄像头源 6 项复查通过，双订阅 12 帧一致、关闭一个后另一个继续；无模型 SDK 导入、无图片或视频保存，最终清理确认。
- 修复了缺 VC++ 运行库的环境问题；普通/模拟平台回归 484 项通过。
- 修正验证夹具的单帧等待竞态：结果读取与单帧提交可能争用非阻塞 DB 栅栏，现使用有界持续供帧，并仍要求各任务返回同一原始帧，不拼接不同时刻的结果。生产结果栅栏和安全门未改变。

原始 profile、成功与首次失败回执保存在本地 Git 忽略的 `.local-validation/gpu-hardware-20261025/`，不上传连接材料。以上仅为硬件管道验证，不是业务模型或生产验收。

## 新相机离线流程验证记录（2026-10-25）

这项验证与上面的合成计算图 smoke 分开：使用真实固定办公室摄像头，分别录制正常训练、校准和留出会话；录像只读，不随机拆相邻帧，也不混入留出录像训练。正常性、测试训练启动和精确模型/场景绑定均有明确用户授权，通过原有账号认证和审批接口完成，没有关闭审批或修改任务状态来跳过检查。

- strict v3 基线提供 179 张训练图、179 张独立校准图；真实 PatchCore 使用 `torch-cuda` 训练、导出 `.pt`、完整校准并登记候选。静态场景适配保留原始质量指标与窄工况警告，不将重复像素当成损坏，也不将其当成独立样本证明。
- 新一轮候选离线验证完整只读解码 10,815 帧，每 60 帧抽样，共 181 帧；前 2 帧被既有 SceneGuard 恢复门拦截，后 179 帧完成 CUDA 模型评分。原视频和模型制品 SHA 前后一致，未分发外部告警。
- 该离线验证保留初始化的学习态，并在 SceneGuard 实际放行后调用已加载模型评分；不是实时 legacy 主链发布、最终告警效果或生产采样能力验收。模型仍为 candidate，质量评级 F，精度暂未接受，生产 readiness 仍为 NO-GO。
- 同一真实 USB 源上的预训练目标检测与姿态任务使用独立 CUDA worker，在 shadow 模式验证独立启停、不同真实权重的执行包切换、精确回滚和后端进程重启恢复；停止意图不会因重启被复活。它们不替代新训练异常模型的实时部署验证。
- 执行包切换显式停止目标任务，以遵守现有双 worker 资源准入上限；没有把该检查称为零中断热更新，也未增加 GPU 并发上限。

原始录像、模型、连接材料、审批回执和详细结果保存在私有验证目录，不进入 Git。这里仅记录工程流程通过的范围；完整业务回归、物理移机、异常事件、长时负例和可信签名 soak 仍需另行验收，生产门禁保持不变。
