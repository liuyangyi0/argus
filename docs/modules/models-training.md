# 基线、训练与模型发布

> 关键源码：`src/argus/anomaly/`、`src/argus/capture/baseline_job.py`、`src/argus/storage/model_registry.py`、`src/argus/storage/release_pipeline.py`、`src/argus/dashboard/routes/models.py`、`src/argus/dashboard/routes/training_jobs.py`、`src/argus/dashboard/routes/baseline.py`

本模块描述从基线采集到模型训练、注册、发布、回滚和标注闭环的完整路径。

## 基线生命周期

基线数据保存在 `storage.baselines_dir`，默认 `data/baselines`。相关对象：

- `BaselineManager`：读写基线目录。
- `BaselineLifecycle`：管理基线版本状态。
- `BaselineVersionRecord`：数据库中的版本状态。
- `run_baseline_capture_job()`：长任务采集入口。

基线版本状态：

```text
draft -> verified -> active -> retired
```

采集集合可以按 camera、zone、version 管理，支持激活、退役、删除、图片上传/删除、集合合并、误报样本合并和优化预览。

## 训练作业

训练作业不是同步 API：

```text
POST /api/training-jobs/
  -> create pending job
POST /api/training-jobs/{job_id}/confirm
  -> queued
TaskScheduler
  -> TrainingJobExecutor.process_queued_jobs()
  -> running / completed / failed
```

`TrainingJobExecutor` 执行流程：

1. 恢复超时卡在 running 的旧作业。
2. 校验超参数。
3. 根据 job type 执行异常头训练或 SSL backbone 训练。
4. 使用 `GlobalPipelineModeGuard` 临时切换摄像头到 `training` 模式。
5. 调用 `ModelTrainer` 或 `BackboneTrainer`。
6. 使用 `TrainingValidator` 做输出校验。
7. 使用 `ModelPackager` 生成模型包。
8. 写 `TrainingJobRecord`、`TrainingRecord`，并在 `ModelRegistry` 注册模型。

`ModelTrainer` 封装 anomalib 训练和导出。OpenVINO IR 必须保持静态形状：导出时需要固定 `input_size=(H, W)`，并禁用动态 axes。

## 多版本数据集

`DatasetSelection` 支持一个训练任务引用多个 baseline version。`ModelTrainer` 内部通过 `_DatasetMerger` 将多个目录合并为临时训练根目录：

- 优先 symlink。
- Windows 普通用户态不支持时回退 copy。
- 文件名前缀包含 camera、zone、version，避免同名冲突。
- 退出时清理临时目录。

自动重训练也会通过 scheduler 中的 `_resolve_dataset_selection_for_retrain()` 解析数据集策略。

## 模型注册

`ModelRegistry.register()` 记录：

- `model_version_id`
- camera id
- model type
- model hash
- data hash
- code version
- training params
- component type
- backbone ref
- model path
- 初始 stage：`candidate`

模型记录是上线状态的事实来源之一，但运行时是否已经热加载还要看发布 operation 的
`applied` 状态和 API 返回的 `runtime_state`，不能仅凭数据库 stage 判断。

## 发布管线

外部发布请求统一交给 `ModelReleaseService` 持久化命令；`ReleasePipeline` 负责阶段与证据规则：

```text
candidate -> shadow -> canary -> production -> retired
shadow -> candidate / retired
canary -> shadow / retired
```

规则：

- 所有 stage transition 都要求明确的 `triggered_by`，并写 `ModelVersionEvent` 与 `AuditLog`。
- `canary` 需要指定 `canary_camera_id`，且必须与模型 camera id 一致。
- `production` promotion 会退役旧 production 和旧 canary。
- shadow/canary 最短停留天数由配置控制。
- transition 写库失败必须 rollback，不能留下半成品状态。
- transition 成功后可通过 WebSocket 向前端发布模型发布事件。

训练完成只生成 candidate，不直接上线。

## 受控回滚

回滚是独立命令，不开放通用的 `retired -> production` 晋级，也不恢复旧激活接口的绕过能力。
只有同一摄像头曾经完成两阶段生产发布、当前已退役且不活跃的版本可以作为目标。
仅有历史 stage 字段、手工迁移记录或缺少原始证据的旧模型均不具备资格。

操作流程：

1. 使用个人账号登录，在模型页对历史版本执行回滚检查。
2. 核对目标 profile 与当前配置绑定、原始 artifact、baseline v3 严格解码证据、
   可信 soak 签名及其配置/语料摘要。系统不会自动切换或批准场景 profile。
3. 确认理由后提交。请求冻结当前生产版本、源/目标 release revision 和历史证据身份，
   返回 `202 + operation_id`；同一摄像头已有待完成发布或 canary 时拒绝。
4. worker 再次核验，执行 runtime prepare/swap 和数据库 CAS；成功才更新 applied stage，
   旧生产版本退役。失败沿用原补偿链，补偿失败保持失败/阻断，不伪报回滚成功。
5. 换模后仍需 SceneGuard 连续确认；预览可用不等于评分和告警已恢复。

网络超时后保留原幂等键与请求重试，先查 operation，不能把超时理解为回滚失败。
数据库版本变化、证据变更或 profile 不符时重新检查，禁止跳过现有 production 门禁。

## Shadow / Canary 运行时

`CameraManager` 会根据 registry 记录创建或更新：

- primary model：生产或 canary 阶段参与主检测。
- shadow runner：旁路评分，不触发告警。

Models 页显示应以 API 返回的 runtime state 为准，不能只看 DB stage。

## 标注与主动学习

主动学习路径：

```text
DetectionPipeline
  -> EventBus.publish(FrameAnalyzed)
  -> ActiveLearningSampler
  -> LabelingQueueRecord
  -> /api/labeling/*
  -> trigger retrain
```

标注队列支持：

- 查询 queue。
- 标注 normal/anomaly。
- skip。
- 查看统计。
- 访问图片。
- 触发增量训练任务。

误报、确认异常和不确定反馈还会进入 feedback 相关存储，用于后续训练数据选择。

## 可选模型能力

默认关闭但代码存在：

- 开放词汇分类器：`OpenVocabClassifier`，配置在 `classifier`。
- 分割器：`segmenter.py`，通常用于异常区域实例分割增强。
- 跨摄像头 corroboration：`CrossCameraCorrelator` 和 config 中的 overlap pairs。
- 自动重训练：scheduler 条件注册。

启用这些能力前先看 `docs/enable_checklists/`，并补对应 smoke 或 targeted tests。

## Dashboard API

主要 API：

- `/api/baseline/*`：基线集合、版本、采集任务、优化、合并、图片。
- `/api/training-jobs/*`：训练作业创建、确认、拒绝、查询、source alerts、backbone 列表。
- `/api/models/*`：模型列表、激活、回滚、promote、retire、reexport、recalibrate、A/B 对比、shadow report、threshold preview。
- `/api/labeling/*`：标注队列。

详见 `docs/api-reference.md`。

## 相关测试

- `tests/unit/test_trainer.py`
- `tests/unit/test_trainer_export.py`
- `tests/unit/test_training_job_wiring.py`
- `tests/unit/test_training_history.py`
- `tests/unit/test_model_registry.py`
- `tests/unit/test_release_pipeline.py`
- `tests/unit/test_release_api_policy.py`
- `tests/unit/test_labeling_lineage.py`
- `tests/unit/test_feedback_queue.py`
- `tests/unit/test_model_runtime.py`
- `tests/unit/test_models_business_flow.py`
