# Argus

固定摄像头的视觉异常监测与证据复核工作台。

Argus 将视频接入、场景校验、异常检测、告警处置和模型管理放在同一套边缘应用中。它关注的不只是“画面里有什么”，还包括“相机是否仍在原来的位置”“当前模型是否适用于这个场景”“这条告警能否通过录像和记录复核”。

项目采用 Python / FastAPI 后端和 Vue 3 / TypeScript 前端，面向单节点、多摄像头部署。适用于固定视角下的异常变化与异物监测研究，不是任意场景通用识别器，也不替代人工巡检或独立安全保护系统。

> 当前处于工程验证阶段。普通功能和模拟测试通过，不等于真实模型精度、GPU 性能、外部标注服务或生产发布已经验收通过。

## 能做什么

| 能力 | 当前范围与边界 |
| --- | --- |
| 视频接入与预览 | RTSP、USB 和视频文件输入；通过 go2rtc 与浏览器播放链路提供预览、重连和运行状态。 |
| 场景与机位校验 | SceneGuard 关联参考场景、基线和模型；机位不匹配时保留预览，阻断受保护检测链的评分与告警放行。选择新场景不等于自动批准或切换模型。 |
| 异常检测 | 主异常模型结合区域、运动、时序证据和辅助检测通道；具体可检测对象、误报率与延迟必须用目标场景的独立视频验收。 |
| 告警处置与取证 | 告警查询、确认与误报处理，关联录像、关键帧、轨迹和诊断信号，支持实时状态更新与审计。 |
| 基线与模型管理 | 场景草稿、基线构建、人工确认训练、独立校准、候选登记、受控发布与回滚；不以训练完成代替发布批准。 |
| 视频人工复核 | 独立 CLI 对接 CVAT / FiftyOne，保留源摘要、原始帧号和时间映射；可将 LightGlue 旁路分歧报告送入复核。 |

分类器、分割器、跨摄像头关联和自动重训练属于可选能力，默认配置下关闭。LightGlue 只用于离线旁路比较，不参与现有 SceneGuard 放行或恢复。VFR 视频的受控图片复核保留原始帧号、PTS、图片和报告摘要，不改写源视频。

这些集成的真实 SDK、数据库、播放器对齐和故障恢复仍需在独立复核环境验收，不能仅凭模拟测试认定可交付。

## 工作方式

```text
视频源 → 采集与质量检查 → 场景安全门 → 检测与时序判断 → 告警与证据
                             ↑                        ↓
                      场景 / 基线 / 模型绑定      Dashboard 处置与回放

离线视频 → CVAT 人工标注 / FiftyOne 复核 → 独立回归报告 → 发布证据
                   LightGlue 仅作旁路对比，不提供发布许可
```

Web 工作台提供总览、摄像头与详情、告警、报表、模型和系统管理；回放从告警进入。3D 页面标记为演示，不属于已完成的现场空间管理能力。

## 开发环境

- Python 3.11+；仓库开发环境使用 `.venv`，不要求 conda。
- Node.js / npm 用于前端开发；具体依赖以 [web/package.json](web/package.json) 为准。
- 实际运行还需要按配置准备视频源、模型工件及相应推理后端。NVIDIA GPU 部署显式使用 `torch-cuda` 与 `.pt` 工件，加载失败不能静默改走 CPU / OpenVINO / SSIM。
- 模型权重、视频素材、部署凭据和运行数据库不作为公开源码的一部分提交。

在 PowerShell 中安装后端开发依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

前端开发：

```powershell
cd web
npm install
npm run dev
```

在具备所需视频源与推理环境的机器上启动完整应用：

```powershell
python -m argus --config configs/default.yaml
```

默认 Dashboard 地址为 `http://127.0.0.1:8080`。后端托管前端时，先在 `web/` 执行 `npm run build`。默认配置不是可直接对外暴露的生产配置。

## 无 GPU 的普通功能验证

只验证 API、权限、数据库、任务状态、前端和模拟业务时，使用受控入口。它们阻断真实训练、模型识别和相机 / 源视频读写；不要用全量 pytest 或完整应用启动代替这组命令。

```powershell
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite lint
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite core
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite dashboard
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite review
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite auxiliary
.\.venv\Scripts\python.exe scripts/run_local_frontend_tests.py unit
.\.venv\Scripts\python.exe scripts/run_local_frontend_tests.py build
```

隔离浏览器测试服务与 E2E 的启动方法见 [普通功能验证说明](docs/runbooks/video-review-integrations.md#本机普通功能验证禁训练和识别)。真实模型、相机、CUDA、完整视频回归和长稳测试应在专用环境另行执行。

## 发布与安全边界

- 固定机位是模型适用性的前提。移动、旋转、变焦或更换场景后，必须重新核对场景与模型绑定，不能直接沿用旧模型的验收结论。
- production 保持严格 NO-GO，直到该候选的全部发布门通过。严格 baseline v3 解码证据、独立数据 / session、完整校准、模型与 profile 身份、独立人工审批、完整回归和可信的至少 72 小时 soak 均不能省略。
- 当前发布指标包括 SceneGuard coverage ≥ 99%、FP ≤ 2/h、overall / static / projectile recall ≥ 95%、P95 delay ≤ 10 秒；完整约束以 [验收契约](docs/acceptance.md#当前-production-发布契约沿用严格门禁不作能力分档) 和服务端校验为准。这些是门槛，不是已达成结果。
- 默认配置仅绑定回环地址且关闭认证。远程访问前必须开启认证、设置独立强随机 token、替换首次启动的默认管理员密码，并配置 TLS 与网络访问控制。
- 真实主机、账号、地址和密钥只保存在仓库外私有配置。历史清理后的旧克隆不得直接合并或推回远端，见 [公开仓库信息卫生](docs/runbooks/publication-hygiene.md)。

## 代码与文档

```text
src/argus/       视频采集、检测、场景工作流、API、存储与流媒体
web/src/         Vue 工作台、播放器、状态管理与前后端契约
scripts/         普通功能验证、离线复核、回归与运维入口
configs/         配置示例，不存放真实部署身份
tests/           单元、模拟集成与静态守门测试
docs/            架构、验收约束、设计与操作手册
```

- [文档入口](docs/index.md) · [架构](docs/architecture.md) · [API 参考](docs/api-reference.md)
- [CVAT / FiftyOne 复核](docs/runbooks/video-review-integrations.md) · [LightGlue 旁路对比](docs/runbooks/scene-guard-lightglue-shadow.md)
- [开发约定](AGENTS.md) · [前端开发](web/README.md) · [检测质量验收](docs/acceptance.md)
