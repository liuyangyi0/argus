# Argus

面向固定摄像头的边缘视觉异常检测与告警系统。

Argus 将多路视频接入、场景与机位校验、异常检测、告警处置和录像回放整合到同一套 Web 工作台。操作员可以实时查看画面、确认告警并追溯现场证据；工程人员可以维护正常场景基线、训练与校准任务，以及每路摄像头的模型版本。

系统采用 Python / FastAPI 后端和 Vue 3 / TypeScript 前端，面向单节点、多摄像头部署。业务元数据使用 SQLite，基线、模型和录像等资料保存在本地文件系统。

项目持续开发中。检测效果依赖固定机位、画质、目标尺寸、输入帧率及训练数据，需要用目标场景的独立素材验证。

## 核心能力

- **视频管理**：接入 RTSP、USB 摄像头和视频文件，提供实时预览、启停控制、断线重连与健康状态；通过 go2rtc 衔接采集与浏览器播放。
- **场景校验**：SceneGuard 检查当前画面与已批准场景是否匹配；不匹配时保留预览，暂停受保护检测流程并显示原因。
- **异常检测**：结合正常场景基线、异常模型、运动分析和时序证据，检测持续变化并形成告警候选；支持 include / exclude 区域。
- **告警与回放**：支持筛选、确认、误报反馈和实时更新，按告警类型和配置关联可用的录像、快照、热力图、轨迹及诊断信号，方便复核告警来源。
- **基线与模型**：管理场景、基线版本、训练确认、独立校准、模型登记、分阶段发布与回滚。训练完成不等于自动上线。
- **权限与运维**：提供角色权限、审计、运行状态、故障与降级提示、配置管理、备份和清理入口。

### 检测通道

| 通道 | 关注的画面变化 | 默认配置 |
| --- | --- | --- |
| 静态异物 / 持续场景变化 | 新增物体、持续占用及相对正常场景的变化 | 开启 |
| 快速运动目标 | 快速经过画面的小物体及其运动轨迹候选 | 开启 |
| 可见水渍 / 滴漏 | 湿斑、积水、滴落或流痕等可见迹象 | 关闭，按场景启用 |
| 可见羽流 | 烟尘、蒸汽、粉尘等可见扩散形态 | 关闭，按场景启用 |

“开启”指配置状态，不代表视频源、模型或检测质量已经就绪。快速运动目标对帧率、分辨率和运动模糊有额外要求；水渍与羽流分析不提供液体成分或无色气体识别。具体参数见 [默认配置](configs/default.yaml)，适用边界见 [检测质量验收](docs/acceptance.md)。

开放词汇分类、分割、跨摄像头关联和自动重训练也是可选模块，默认关闭。

## 业务流程

```text
接入视频 → 建立正常场景 → 构建基线 → 训练与校准 → 验证并发布 → 监控与告警复核
```

视频可以先接入预览。启用受保护的检测流程，还需要完成场景审批与匹配模型绑定；相机移动、旋转或变焦后，应重新检查场景适用性。工程人员可从模型页的基线管理入口进入 [场景、基线、训练与发布流程](docs/runbooks/video-review-integrations.md#dashboard-场景基线训练与发布工作流)。

正式发布由服务端检查独立回归、人工审批和长稳运行证据；离线复核结果或训练成功不能代替发布许可。详细门禁见 [验收契约](docs/acceptance.md)。

## 快速开始（开发环境）

### 1. 安装并构建

需要 Python 3.11+、Node.js 22.12+（22.x）或 24+，以及 npm。以下命令以 PowerShell 为例，均从仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

npm --prefix web install
npm --prefix web run build
```

前端构建产物位于 `web/dist`，完成构建后后端才能托管完整 Vue 工作台。CVAT、FiftyOne 和 LightGlue 不属于这组默认安装依赖。

### 2. 配置并启动

检查 [configs/default.yaml](configs/default.yaml)，按实际设备调整摄像头的 `source` 和 `protocol`；默认示例为 USB 设备 `0`。默认 USB 链路依赖 go2rtc，还需要准备对应的设备访问权限与流媒体环境。

默认 SceneGuard 要求经批准的场景与模型绑定。首次启动可先在工作台完成配置；服务启动或预览可用不表示检测已就绪，应同时检查场景、模型和运行状态。NVIDIA GPU 部署须显式配置 `torch-cuda` 与 `.pt` 工件，不能把 OpenVINO 工件当作 CUDA 模型。

```powershell
python -m argus --config configs/default.yaml
```

打开 [本机工作台](http://127.0.0.1:8080)。默认仅绑定回环地址，认证关闭；空数据库首次初始化账号为 `admin/admin`。对外访问前必须更换默认密码、开启认证、设置强随机 API token，并配置 TLS 与网络访问控制。

配置支持进程环境变量覆盖，例如 `ARGUS__AUTH__ENABLED`、`ARGUS__AUTH__API_TOKEN`。变量清单见 [.env.example](.env.example)；Argus 不会自动加载 `.env` 文件，需要由终端或服务管理器显式导入。

### 3. 前端开发模式（可选）

保持后端运行，在仓库根目录另开终端：

```powershell
npm --prefix web run dev -- --host 127.0.0.1
```

打开 `http://127.0.0.1:5173`。Vite 将 API 与 WebSocket 请求代理到默认后端 `127.0.0.1:8080`；开发服务不要直接暴露到公网。

## 可选离线工具

这些工具通过独立 CLI 使用，不随 Dashboard 启动连接外部服务：

| 工具 | 用途与边界 |
| --- | --- |
| CVAT | 上传视频标注任务、人工复核与回导标签，再导出独立回归素材。 |
| FiftyOne | 对照预测与标签、筛选问题样本、记录复核笔记；笔记不会自动回写为验收真值。 |
| LightGlue | 离线对比 SceneGuard 与局部特征匹配结果，定位分歧帧；不参与现有 SceneGuard 放行或恢复。 |

复核数据保留源摘要、原始帧号和时间映射；不适合原生播放器对齐的 VFR 视频可使用受控逐帧图片复核，不改写源视频。真实外部服务、播放器对齐与故障恢复仍需在复核环境验收。

安装和操作见 [CVAT / FiftyOne 手册](docs/runbooks/video-review-integrations.md) 与 [LightGlue 旁路对比](docs/runbooks/scene-guard-lightglue-shadow.md)。

## 开发与验证

只验证普通 API、权限、数据库、前端及模拟业务时，可使用不运行真实训练和识别的受控测试入口：

```powershell
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite core
.\.venv\Scripts\python.exe scripts/run_local_frontend_tests.py unit
.\.venv\Scripts\python.exe scripts/run_local_frontend_tests.py build
```

完整普通功能清单和隔离浏览器验证见 [开发验证说明](docs/runbooks/video-review-integrations.md#本机普通功能验证禁训练和识别)。真实训练、识别、GPU 性能、完整视频回归和长稳运行需另行验证；模拟测试通过不代表这些项目已通过。

## 代码与文档

```text
src/argus/       后端：采集、检测、场景工作流、API、存储与流媒体
web/src/         前端：监控页面、播放器、状态管理与接口契约
scripts/         离线复核、回归、开发验证与运维入口
configs/         配置示例
tests/           单元、集成与静态守门测试
docs/            架构、设计、验收约束与操作手册
```

- [文档入口](docs/index.md) · [系统架构](docs/architecture.md) · [API 参考](docs/api-reference.md)
- [前端开发](web/README.md) · [开发约定](AGENTS.md) · [配置与敏感信息管理](docs/runbooks/publication-hygiene.md)
