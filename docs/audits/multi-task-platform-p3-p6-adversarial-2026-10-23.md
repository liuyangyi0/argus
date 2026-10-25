# P3—P6 多任务改造对抗审查与修复

日期：2026-10-23。三路 agent 分别审查运行时、场景/存储权威和采集/人审，主 agent 审查路由及前端，并交叉复核集成边界。工作树原有修改保留，未提交或推送。

## 已确认并修复的问题

| 优先级 | 问题与影响 | 修复边界 |
| --- | --- | --- |
| P1 | 执行包解析时尚无 worker，stop 可能遗漏后续启动；旧 stop 也可能清理后来启动的新执行 | 每次启动有独立取消令牌；停止仅清理原 execution，未知/失败清理继续持有 lease |
| P1 | 配置/归档的运行状态预检查与 start 之间有 TOCTOU | `mutate_if_stopped` 与启停共用控制锁，从清理确认到 DB 提交保持串行 |
| P1 | 单个全局控制锁使一个任务长时间加载拖延其他任务的停止确认 | 按 task 管理控制锁；全局短锁只维护执行登记和资源预算；关闭先取消全部 pending/worker，再逐任务确认清理 |
| P1 | 已提交操作响应丢失后，后续运行状态会挡住同键重试 | SELECT-only 收据核对动作、actor、完整请求；未命中才进入写门，BUSY 时再查一次已提交收据，不借此放行新写 |
| P1 | 新任务场景观测可能继承 legacy 的宽松验证 profile | managed 独立核验人工审批、审批身份/带时区时间、非 validation、模型版本/完整 SHA/hash scheme；不修改 legacy 的几何判断和恢复政策 |
| P1 | 摄像头已声明停止但尚未拆资源时，新任务仍可能注册或提交 | managed 读取、最终场景 CAS 和任务注册均拒绝 stopping/retiring，并复核消费线程身份与存活 |
| P1 | 自然失败/重启后旧自动采集器可能遗留；人工复用 workflow 后可能被旧自动 stop 停止 | 自动执行保留精确 collection_id；定向清理旧 execution 的实际采集实例，workflow 复用不转移所有权 |
| P1 | 终态采集收据已提交，但清理失败不能正常重试；新 claim 提交响应丢失又可能被旧终态映射卡住 | 对本进程确权的终态 collector 原位重试清理；区分 DB claim、实际 worker 和最终收据；关闭订阅失败仍尝试 join，未确认不能释放所有权 |
| P2 | worker readiness/provider、布尔 request_id、畸形 result.data 未严格拒绝 | 校验精确后端、真正 int 请求 ID 及模板允许的结果状态；畸形响应进入失败与确认清理，成功后清除过期错误码 |
| P2 | 姿态预标注缺 COCO-17 schema，所有建议被严格工作流拒绝 | 生成完整 `keypoint_schema=coco17.v1`；保持精确同帧匹配和自动建议待人工复核 |
| P2 | 固定快速运动模板被列为可训练，但无模型角色，产物永远不能登记 | 从外部检测配方移除该模板；旧提案也不能新批准或导出，API 只列当前任务兼容配方，未知任务返回 404 |
| P2 | 候选登记结果未知后换人重试会改变幂等请求 actor | 保留首次登记 actor 与请求身份，对账原登记，不伪造新审批或重复候选 |
| P2 | SQLite 重读丢失 UTC tzinfo，使历史收据与当前读取时间不一致 | 任务序列化统一恢复 UTC，不更改原时间点 |
| P2 | 前端启动等待时停止按钮也被禁用，迟到启动响应可能覆盖停止提示 | 允许等待中的启动被停止；独立操作代次拒绝迟到响应，保持跨相机/任务隔离 |
| P2 | 审核图片加载失败时仍可批准；跨相机仍保留旧候选路径草稿 | 当前图片成功加载前禁止新批准；晚到旧图片不能解锁新样本；切相机清空候选草稿 |

相关实现：

- 运行时：`core/task_execution.py`、`capture/task_execution_manager.py`。
- 场景：`capture/manager.py`、`core/pipeline.py` 的 managed 专用观测/CAS。
- 权威与对账：`storage/analysis_tasks.py`、`dashboard/routes/task_platform.py`。
- 数据闭环：`runtime/task_automation.py`、`storage/task_workflows.py`、`capture/task_collector.py`、`core/task_preannotations.py`。
- 前端：`TaskWorkbench.vue`、`TaskDatasetPanel.vue` 及普通浏览器用例。

## 人审与历史状态复核

归档或配置变更后，旧流程可历史读取及安全清理，但不能继续冻结、批准/导出训练或登记候选。自动预标注仍为待复核；新的报告/标注会撤销当前训练批准，保留人工备注、审核历史和旧冻结数据集，不把历史批准当作新报告的批准。

前端显示当前人工决定与自动建议。后台刷新后旧草稿不会套用新 revision；图片未就绪/失败也不会自动改变服务端历史审核决定，只阻止本次新批准。

旧 profile 缺人工审批身份、时间或完整模型绑定时，新任务 `require_profile` 明确 NO-GO；必须按原服务端审批流程处理，不能手填历史审批。旧 SceneGuard 的放行、阈值与恢复规则不变。

## 验证口径

只运行 `run_local_functional_tests.py` 的受保护普通/mock 套件，以及前端单元、类型/构建和隔离浏览器。浏览器人审图片用例使用合成 SVG 传输，验证页面加载/失败和版本交互；不冒充真实 PNG/SHA、摄像头或识别验收，后端图片校验另由模拟样本回归覆盖。

各 agent 已完成定向回归，以下为集成验证结果（定向用例包含在对应套件中，不重复累加）：

| 验证 | 结果 |
| --- | --- |
| 多任务平台完整普通/mock 回归（`platform`，最终冻结代码） | 390 passed |
| 原场景及发布相关普通/mock 回归（`core --filter scene`） | 347 passed，950 deselected |
| 前端完整单元回归 | 49 个文件，297 passed |
| 多任务真实浏览器普通功能（1920 / 1440 / 1024 / 手机视口） | 12 passed；不启动模型或采集源视频 |
| 静态约束及 ORM 自动迁移守门 | 74 passed，1 skipped |
| 本轮变更 Python Ruff、前端类型检查和构建、OpenAPI 类型一致性、`git diff --check` | 通过 |

测试没有修改源视频，没有加载真实模型或运行训练/识别，没有新增生产发布路径。工作区代码尚未提交或推送。

## 仍未验收的边界

- 新任务继续 shadow-only，production NO-GO，原异物行为和发布门禁保留。
- 真实 CUDA 并行吞吐、显存、相机移动/重连质量及 GPU 故障恢复尚未测试。
- 同一任务的启停和配置写入保持串行；不同任务不再共用加载/停止长操作锁。实际进程退出和 GPU 驱动故障下的停止时延仍待目标机器验收。
- 声明内存预算不是 OS 硬隔离；外部训练交接不是内置检测/姿态训练器，离线清单不是 SDK 已上传。
