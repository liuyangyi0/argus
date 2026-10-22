# Argus 文档入口

> 最后更新：2026-07-13

本文档定义 Argus 仓库的文档结构。后续写模块细节时，先读 `docs/source-index.md` 的压缩源码索引，再按模块打开源码和对应文档，避免一次性把全仓库上下文塞满。

## 文档结构

| 文档 | 作用 | 维护时机 |
| --- | --- | --- |
| `README.md` | 项目是什么、快速启动、测试入口、主要目录 | 启动方式、默认能力或用户入口变化时 |
| `docs/architecture.md` | 运行时拓扑、数据流、关键边界、默认/可选能力 | 启动链、管线拓扑、跨模块数据流变化时 |
| `docs/acceptance.md` | 检测质量验收数字、误报/召回计数口径和物理边界 | 调整检测质量目标、回归库结论或部署边界时 |
| `docs/design/recovery-plan-2026-06-12.md` | 检测质量收敛计划、P0/P1 当前状态和后续 P2-P4 路线 | 继续检测质量收敛工作、调整阶段状态或复盘计划时 |
| `docs/design/camera-scene-guard-and-video-recognition-2026-07-13.md` | 摄像机移动防护、视频案例生产管线回放、算法路线与 GO/NO-GO 门槛 | 调整 SceneGuard、scene profile、视频回归或候选确认方案时 |
| `docs/runbooks/gpu-validation-runtime.md` | gpu-validation 测试机、OBSBOT、go2rtc exec+NVENC、Argus 启停、模型切换与回归命令 | 测试机运行状态、摄像头档位、隧道、GPU/OpenVINO 切换变化时 |
| `data/regression/HANDOFF.md` | 真实回归素材的恢复入口、远端路径、commit、基线数字、推荐采集帧率和下一步 | 下次接着跑回归库、恢复视频、查首版基线或录新素材时 |
| `docs/source-index.md` | 自动提取的目录树、Python 类/函数签名、FastAPI 路由、前端路由/API 客户端 | 改动源码结构或 API 后运行生成脚本 |
| `docs/api-reference.md` | Dashboard HTTP API、WebSocket 主题、响应约定 | 新增、删除或改变 `/api/*` / `/ws` 行为时 |
| `docs/modules/README.md` | 后端与前端模块地图 | 新增顶层包、页面域或重要子系统时 |
| `docs/modules/runtime-configuration.md` | CLI、配置、启动装配、调度任务、默认配置边界 | `__main__.py`、配置 schema、默认 YAML 或调度装配变化时 |
| `docs/modules/capture-streaming.md` | 摄像头采集、运行规划、go2rtc、预览/播放路径 | 新协议、新采集适配器或流媒体路径变化时 |
| `docs/modules/detection-pipeline.md` | 单路检测链、模式、降级、告警生成与诊断 | `core/pipeline.py`、`runner.py`、告警判定或检测阶段变化时 |
| `docs/modules/models-training.md` | 基线、训练、模型注册、发布管线、标注闭环 | 训练作业、模型阶段、基线生命周期或标注逻辑变化时 |
| `docs/modules/storage-operations.md` | SQLite、ORM、自动迁移、录像/推理记录、备份与清理 | ORM 字段、数据目录、保留策略、迁移逻辑变化时 |
| `docs/modules/dashboard-frontend.md` | FastAPI 门面、认证、WebSocket、Vue 路由、前端 API 层 | 后端路由、app state、前端路由或页面职责变化时 |
| `docs/requirements.md` | 当前产品目标和交付范围 | 产品范围、默认/非默认能力表述变化时 |
| `docs/enable_checklists/*` | 默认关闭能力的启用检查清单 | 分类器、分割器、跨摄像头、自动重训练等能力启用路径变化时 |
| `docs/design/*` | 设计提案和未完全落地的 UX / 产品方案 | 方案被采纳、废弃或拆分到当前文档时 |
| `docs/audits/*` | 历史审计、集成检查和一次性调查记录 | 需要追溯历史判断或复盘时 |
| `docs/archive/*` | 不参与当前维护的历史产物或静态导出 | 仅作追溯，默认不作为当前实现依据 |

## 推荐阅读顺序

1. 新人理解项目：`README.md` -> `docs/architecture.md` -> `docs/modules/README.md`。
2. 改后端业务：先读 `docs/source-index.md` 对应包签名，再读相关模块文档。
3. 改 Dashboard API：读 `docs/modules/dashboard-frontend.md` 和 `docs/api-reference.md`，再打开对应 `src/argus/dashboard/routes/*.py`。
4. 改检测链：读 `docs/modules/detection-pipeline.md`，再打开 `src/argus/core/pipeline.py`、`src/argus/core/runner.py`、`src/argus/capture/manager.py`。
5. 改模型训练/发布：读 `docs/modules/models-training.md`，再打开 `src/argus/anomaly/job_executor.py`、`src/argus/anomaly/trainer.py`、`src/argus/storage/release_pipeline.py`。
6. 改存储字段：读 `docs/modules/storage-operations.md`，同时检查 `src/argus/storage/models.py` 和 `src/argus/storage/database.py` 的 `_AUTO_MIGRATIONS`。

## 自动索引

运行：

```powershell
python scripts/extract_source_index.py --output docs/source-index.md
```

索引内容包括：

- `src/argus/`、`web/src/`、`configs/`、`scripts/`、`tests/`、`docs/` 的压缩目录树。
- 每个 Python 文件的顶层类、函数和类方法签名。
- 从 `create_app()` 装配结果和 route decorators 提取的 FastAPI 路由。
- 从 Vue router 和 `web/src/api/*.ts` 提取的前端路由/API 客户端函数。

该索引不是架构文档本身，只是写文档和做代码导航的轻量上下文。
