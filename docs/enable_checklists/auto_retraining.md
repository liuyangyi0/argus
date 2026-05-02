# 自动重训练 启用 Checklist

> 适用范围：Argus `src/argus/anomaly/` 下的异常检测模型自动重训练链路。
> 默认在 `configs/default.yaml` 中关闭（`retraining.enabled: false`）。
> 本 checklist 描述当前实现，不引入新触发器或新调度策略。

---

## 1. 模块定位

自动重训练负责在新基线图片累积到一定数量后，**自动触发**单摄像头/摄像机组的异常检测头（anomaly head）重训练，并按质量等级阈值决定是否自动激活新模型版本；同时按周期检查共享 SSL backbone 是否到期，触发 backbone 重训练任务（默认仍需人工确认）。

**默认关闭的原因**（按代码注释）：

- 安全/审计要求：`RetrainingConfig.require_human_confirmation` 默认为 `True`，调度器创建的 backbone 任务会以 `pending_confirmation` 状态等待人工放行（`src/argus/core/scheduler.py:296-310`）。
- 资源占用：训练阶段会主动把 YOLO 等推理模型迁出 GPU（`ModelTrainer._free_gpu_for_training`，`src/argus/anomaly/trainer.py:181-218`），训练期间 GPU 推理会受影响。
- 数据质量风险：基线/标注质量不达标时，自动激活的新模型可能反向劣化检测效果。

**何时决定重训练**（实现见 `create_retraining_task`，`src/argus/core/scheduler.py:327-530`）：

1. 调度器按 `retraining.interval_hours` 间隔（默认 24h）周期性扫描全部已配置摄像头。
2. 对每个摄像头，比较 `BaselineManager.count_images(camera_id, "default")` 与 ModelRegistry 里上一版本记录的 `image_count`。
3. 当“当前总数”和“相对上次训练新增数”同时 `≥ retraining.min_new_baselines`（默认 20）时，调用 `ModelTrainer.train(...)` 训练新模型。
4. 训练完成后若 `auto_deploy=True` 且质量等级 `≥ auto_deploy_min_grade`，调用 `ModelRegistry.activate(..., allow_bypass=True)`。

---

## 2. 触发条件

当前实现里 **共三类触发器**，互不耦合：

### 2.1 计划重训练（scheduled，单摄像头/摄像机组）

- 注册位置：`create_retraining_task`（`src/argus/core/scheduler.py:327`）。
- 周期：`retraining.interval_hours` 小时（默认 24，范围 1–168）。
- 判定逻辑（每个摄像头独立判定）：
  - `current_count = baseline_manager.count_images(camera_id, "default")`
  - `last_image_count = model_registry.list_models(camera_id=camera_id)[0].image_count`
  - 必要条件：`current_count >= min_new_baselines` 且 `current_count - last_image_count >= min_new_baselines`
- 摄像机组路径：`baseline_manager.get_group_baseline_dir(group_id, zone_id)` 下 `*.png + *.jpg` 数 `>= min_new_baselines` 时触发，复用首个成员摄像头的 `anomaly` 配置。

### 2.2 计划 backbone 重训练（scheduled，全局共享）

- 注册位置：`create_backbone_retraining_task`（`src/argus/core/scheduler.py:260`）。
- 周期：固定 24h 检查一次。
- 判定逻辑：`active_backbone.created_at` 距今天数 `>= retraining.backbone_retrain_interval_days`（默认 30，范围 7–180）。
- 行为：仅写入一条 `TrainingJobRecord(status=PENDING_CONFIRMATION, job_type=ssl_backbone, trigger_type=scheduled)`，**不会自动执行**，等待运维在 Models 页训练任务列表中确认/拒绝。

### 2.3 人工/漂移建议重训练（pending_confirmation 队列）

- 调度器另注册 `create_job_processing_task`（`src/argus/core/scheduler.py:239`）每分钟扫一次 `training_jobs` 表里 `status=QUEUED` 的任务，由 `TrainingJobExecutor.process_queued_jobs()` 执行。
- 触发来源：
  - `manual`：dashboard 训练任务弹窗。
  - `drift_suggested`：标注队列里漂移规则触发，写入位置 `src/argus/dashboard/routes/labeling.py:197`。
  - `scheduled`：上述 backbone 检查产生。
- 注意：该队列处理任务无论 `retraining.enabled` 是否打开都会注册（参见 `src/argus/__main__.py:367` 调用 `_register_training_job_processing`），它消费的是“已经在数据库里 `QUEUED`”的任务。`retraining.enabled` 只影响 2.1 / 2.2 两个 scheduled 入口。

---

## 3. 数据闭环依赖

### 3.1 训练数据来源

- 全部读取自 `BaselineManager`：
  - 单摄像头：`baselines_dir/{camera_id}/default/v{NNN}/baseline_*.png|jpg`（`current.txt` 标记当前版本，`baseline.py:44-64`）。
  - 摄像机组：`baselines_dir/groups/{group_id}/{zone_id}/`（无版本子目录）。
- **生命周期 gate**：`ModelTrainer.train` 会检查 baseline 是否处于 `Verified` 或 `Active` 状态，否则训练直接失败（`trainer.py:278-292`）。
- 误报反馈如果配置了 `feedback.auto_baseline_on_fp=true`（默认开），FP 抓图会被并入 baseline 目录，`create_retraining_task` 计数会因此增长。

### 3.2 是否需要先开 active learning？

不强制。结论拆分如下：

- **不依赖**：当前 `create_retraining_task` 只看磁盘上 baseline 图片数量，与 `ActiveLearningSampler` 的 `RetrainingTriggered` 事件**没有订阅链路**（grep `RetrainingTriggered` 结果只有发布者本身，`src/argus/core/active_learning.py:313`）。
- **建议同时开启**：active learning 把高熵帧推入标注队列（`data/active_learning/`），运营人员在 Labeling 页确认/打标后会写入 `data/baselines/{cam}/false_positives` 或 `data/validation/{cam}/confirmed`，间接为下次自动重训练补充样本。

### 3.3 标注数据最小数量要求

- **训练侧硬下限**：`MIN_BASELINE_IMAGES = 30`（`trainer.py:28`）。低于此数量，`ModelTrainer.train` 会走 few-shot 分支或直接 `FAILED`。
- **触发下限**：`retraining.min_new_baselines`（默认 20，范围 5–500）。建议保持 ≥ 30，与训练侧硬下限对齐。
- **真实标签评估下限**：`val_precision/recall/f1/auroc/pr_auc` 仅在 `data/validation/{cam}/confirmed` 与 `data/baselines/{cam}/false_positives` 各自 ≥10 张时才会写入（`storage/models.py:299-316`）。否则训练记录里这些字段为空，质量等级仅基于 normal-holdout 的 `val_score_*`。

---

## 4. 配置项

来自 `src/argus/config/schema.py:929-987` 的 `RetrainingConfig`，对应 `configs/default.yaml:252-264` 的 `retraining:` 段。

| 字段 | 默认 | 合法范围 | 说明 / 调优建议 |
|---|---|---|---|
| `enabled` | `false` | bool | 总开关。**只有这一项控制 scheduled 入口；queued 任务消费始终启用**。|
| `interval_hours` | `24` | 1–168 | 计划重训练扫描周期。生产环境建议 24h；开发联调可降到 1h 加快验证。|
| `min_new_baselines` | `20` | 5–500 | 触发阈值；建议 ≥ 30 与训练硬下限对齐。摄像机组共用此值。|
| `auto_deploy` | `false` | bool | 训练成功且达到质量门槛后是否自动 `ModelRegistry.activate(..., allow_bypass=True)`。生产慎开。|
| `auto_deploy_min_grade` | `B` | A/B/C/F | 自动激活的最低质量等级。等级排序：A=4 > B=3 > C=2 > F=1（`scheduler.py:236`）。|
| `backbone_retrain_interval_days` | `30` | 7–180 | Backbone 任务到期判定（仅入队待确认）。|
| `backbone_type` | `dinov2_vitb14` | `dinov2_vits14/vitb14/vitl14` | Backbone 变体。`vitl14` 显存占用最高。|
| `validation_auroc_threshold` | `0.99` | 0.90–1.0 | 训练验证 AUROC 阈值（`TrainingValidator` 使用）。|
| `validation_recall_threshold` | `0.95` | 0.80–1.0 | 合成异常的 recall 阈值。|
| `historical_replay_days` | `30` | 7–90 | 历史告警回放窗口。|
| `require_human_confirmation` | `true` | bool | Backbone 任务总是要人工确认，本字段保留为业务标志位。|
| `confirmation_timeout_hours` | `72` | 1–168 | Pending 任务过期阈值（待业务侧消费）。|

---

## 5. 启用步骤

按顺序机械执行，每一步附带预期结果。

1. **确认前置依赖**
   - `pip install apscheduler` 已安装（缺失则 `scheduler.py:46-52` 会打 `scheduler.apscheduler_not_available` 警告并放弃所有计划任务）。
   - `data/baselines/{camera_id}/default/v{NNN}/` 至少有一个版本，且 `BaselineLifecycle` 中状态为 `Verified` 或 `Active`。
   - 已成功手动训练过一次，使 `models` 表里有一条 `image_count` 记录（否则首轮 `last_image_count=0`，新增数等于全量，可能立刻触发）。

2. **修改 `configs/default.yaml`**：
   ```yaml
   retraining:
     enabled: true
     interval_hours: 24
     min_new_baselines: 30           # ≥ 训练硬下限
     auto_deploy: false              # 首次启用建议保持 false
     auto_deploy_min_grade: B
     backbone_retrain_interval_days: 30
   ```

3. **重启后端**：`python -m argus --config configs/default.yaml`。
   配置目前**不支持热加载**，必须重启进程。

4. **校验调度任务已注册**（见第 9 节 自检命令）。

5. **观察首轮运行**：等到 `interval_hours` 周期到达，或临时把 `interval_hours` 设小做联调；查看日志关键字 `retraining.triggered` / `retraining.complete`。

6. **打开 auto_deploy 前的灰度建议**：在 `auto_deploy=false` 状态下连续观察 ≥ 3 轮重训练的 `quality_grade` 分布，确认稳定 ≥ B 后再打开自动激活。

---

## 6. 训练管控

### 6.1 主链路是否暂停

- **不暂停**，但训练期间 GPU 推理性能会下降：
  - `ModelTrainer._free_gpu_for_training()` 把所有 `_shared_yolo_registry` 中的 YOLO 模型 `.to("cpu")`，并 `torch.cuda.empty_cache()`（`trainer.py:182-218`）。
  - 训练完 `_restore_gpu_after_training()` 把它们再迁回 `cuda:0`（`trainer.py:220-233`）。
- **训练完成后是热加载，不需要重启**：`register_training_job_processing` 注入了 `_on_model_trained` 回调，调用 `pipeline.reload_anomaly_model(model_path)`（`runtime/training_job_wiring.py:46-56`）。回调在 daemon 线程里跑（`job_executor.py:432-440`）。
- 但 **scheduled 路径**（`create_retraining_task`）目前 **没有热加载回调**，它直接调 `ModelTrainer.train` + `ModelRegistry.activate`，需要 pipeline 自己读 ModelRegistry 才能在下一轮加载新模型。如需即时生效，建议走 queued 任务路径而非 scheduled 路径。

### 6.2 失败与回退

- **训练失败**：`TrainingResult.status != COMPLETE` 时，`scheduler.py:466-471` 仅记 `retraining.camera_failed` 日志并继续下一个摄像头，**不会自动回退**当前激活模型。
- **打包失败**：`TrainingJobExecutor._execute_head` 会把刚注册的 CANDIDATE 通过 `ModelRegistry.retire()` 回收，避免 ghost 版本（`job_executor.py:362-395`）。
- **质量不达标**：`auto_deploy=true` 但 `grade < auto_deploy_min_grade` 时，仅记 `retraining.skip_deploy` 日志，旧模型保持 active。
- **stale RUNNING 任务**：`TrainingJobExecutor.recover_stale_jobs(max_running_hours=6.0)` 每分钟扫描一次，把 RUNNING 超过 6h 的任务标记 FAILED（`job_executor.py:118-145`）。
- **手动回退**：`ModelRegistry.rollback(camera_id)` 可以回到上一个 PRODUCTION 版本（`storage/model_registry.py:386`）。需运维主动调用，无自动 rollback 逻辑。

### 6.3 模型版本管理

- 自动激活路径：`scheduler.py:438-451` 调 `ModelRegistry.register(...)` 拿到 `version_id`，立即 `activate(version_id, allow_bypass=True)`。`allow_bypass=True` 跳过 shadow → canary → production 的 release pipeline 校验（`storage/model_registry.py:235-298`）。
- Queued 任务路径：注册阶段创建 `CANDIDATE`，需要业务在 Models 页通过“晋升”按钮走 promote 流程。

---

## 7. UI 入口

- **Models 页 → 训练与评估 tab**（`web/src/views/Models.vue:66`）：
  - 子 tab `训练历史`：`TrainingHistory.vue` 调 `getTrainingHistory()`，渲染 `training_records` 表。所有自动重训练完成都会在这里出现。
  - 子 tab `训练任务`（`TrainingJobsList.vue`）：渲染 `training_jobs` 表，`trigger_type` 列展示 `manual / drift_suggested / scheduled` 图标，自动 backbone 任务会以 `pending_confirmation` 状态出现并提供“确认/拒绝”按钮。
- **下一次预定的重训练时间**：当前 UI **没有专门面板**展示。可以通过日志关键字 `scheduler.retraining_registered` / `scheduler.backbone_check_registered` 推算（启动时间 + `interval_hours`）。
- **激活的模型版本**：Models 页 → 模型管理 tab → `ModelTable.vue`，显示当前激活版本和 stage。

---

## 8. 性能影响

> 实测数字依赖具体硬件/模型组合，下方给出参考值与测量方法，未实测处标 TBD。

| 指标 | 参考值 / TBD | 测量方法 |
|---|---|---|
| 训练期间 GPU 显存占用 | TBD（依模型）；PatchCore/PaDiM 约 2–4 GB，Dinomaly+ViT-L 8 GB+ | `nvidia-smi -l 2` 或日志 `trainer.gpu_memory_freed` 中 `free_mb`/`total_mb` |
| 训练期间 CPU 占用 | TBD | `top -p <pid>` 或 `htop`，关注训练子进程 |
| 单摄像头训练耗时 | PatchCore/PaDiM 约 30–120 s（CPU 也能跑）；EfficientAD/FastFlow 70 epochs 数分钟–数十分钟；Dinomaly 视 backbone 大小 1–10 min | 训练完成后查 `training_records.duration_seconds` 或日志 `retraining.complete` |
| 主链路 FPS 退化 | TBD（YOLO 离 GPU 后回到 CPU/小模型，跌幅大；建议训练前后各采 30 s FPS 对比） | dashboard 摄像头详情页或 `/api/health`，对比训练开始前/中/后各采 30 s |
| 热加载停顿 | 单摄像头 < 200 ms（仅切换 anomaly 头） | 日志 `training.model_hot_reloaded` 时间戳与下一帧检测日志间隔 |

**强烈建议**：在生产环境启用前，必须按上述方法实测，并把数字回填本表（避免后人盲目跟随默认值）。

---

## 9. 自检命令

> Windows PowerShell 假设已 `.\.venv\Scripts\Activate.ps1`；Linux/macOS 用对应命令。

### 9.1 验证调度任务已注册

启动日志中应出现下列关键字（缺一表示未注册成功）：

- `scheduler.started` `backend=apscheduler`
- `scheduler.retraining_registered interval_hours=<N> auto_deploy=<bool> min_grade=<X>`
- `scheduler.backbone_check_registered interval_days=<N>`
- `scheduler.job_processing_registered`

抓取最近 200 行日志：

```powershell
Get-Content data\logs\argus.log -Tail 200 | Select-String "scheduler\."
```

或在 Python REPL 里检查（需替换 db 路径）：

```python
from argus.storage.database import Database
db = Database("sqlite:///data/argus.db")
print(db.list_training_jobs(limit=5))
```

### 9.2 期望日志关键字（按时间顺序）

- `retraining.skip_insufficient` / `retraining.skip_no_new`：未达阈值，正常。
- `retraining.triggered camera_id=... new_images=... total_images=...`：触发训练。
- `job_executor.started job_id=...`（仅 queued 路径）。
- `retraining.complete camera_id=... grade=... status=...`。
- `retraining.auto_deployed camera_id=... version_id=... grade=...`（仅 auto_deploy 命中）。
- `retraining.skip_deploy`（grade 低于阈值时）。
- `training.model_hot_reloaded camera_id=... model_path=...`（仅 queued 路径）。
- `scheduler.backbone_job_created job_id=...`（backbone 到期）。

### 9.3 数据库判定

新一行 `training_records` 出现：

```powershell
sqlite3 data\argus.db "SELECT id, camera_id, model_type, quality_grade, val_score_mean, val_score_p95, created_at FROM training_records ORDER BY id DESC LIMIT 5;"
```

最新模型版本：

```powershell
sqlite3 data\argus.db "SELECT model_version_id, camera_id, stage, is_active, image_count, quality_grade, created_at FROM models ORDER BY created_at DESC LIMIT 5;"
```

待确认 backbone 任务：

```powershell
sqlite3 data\argus.db "SELECT job_id, job_type, trigger_type, status, created_at FROM training_jobs WHERE status='pending_confirmation' ORDER BY created_at DESC;"
```

---

## 10. FAQ

**Q1：什么场景下不要开自动重训练？**
- 单摄像头基线图 < 30 张：训练硬下限不满足，每轮都会失败。
- 数据漂移剧烈但缺少标注：scheduled 路径只看数量不看质量，可能反向劣化。
- 仅 CPU、且部署了 EfficientAD/FastFlow：训练耗时长，会持续抢资源；建议手动训练或夜间窗口手动触发。
- 多摄像头共享一张 GPU 且推理 FPS 已逼近 SLA：YOLO 迁出/迁回会出现可见的检测延迟尖刺。

**Q2：开了 `enabled=true` 但没看到任何训练发生？**
- 确认 `apscheduler` 已安装（无则不会注册任何计划任务）。
- 检查 baseline lifecycle 状态：未到 Verified/Active 时 `ModelTrainer.train` 直接失败（看日志 `retraining.camera_failed`）。
- 阈值没满足：日志会出现 `retraining.skip_insufficient` 或 `retraining.skip_no_new`。
- 首次启用且没有任何旧模型：scheduler 会以 `last_image_count=0` 计算新增，可能立刻触发，反而要担心“立即跑训练”而不是“没跑”。

**Q3：scheduled 训练完成了，但 pipeline 还在用旧模型？**
- 当前 `create_retraining_task` 路径**没有挂热加载回调**。要想立即生效，请走 dashboard 训练任务（queued 路径），它通过 `register_training_job_processing` 的 `_on_model_trained` 钩子调用 `pipeline.reload_anomaly_model(...)`。或者重启进程让 pipeline 从 ModelRegistry 重新加载 active 版本。

**Q4：自动激活的版本质量不行，怎么回退？**
- 调用 `ModelRegistry.rollback(camera_id, triggered_by="ops")`（`storage/model_registry.py:386`）回到上一 PRODUCTION 版本；或在 Models 页“模型管理”里手动激活历史版本。当前**没有自动 rollback 逻辑**。

**Q5：backbone 任务一直停在 pending_confirmation 怎么办？**
- 在 Models 页 → 训练任务列表里点“确认”进入 QUEUED；下一次 `process_training_jobs`（每分钟）会被 `TrainingJobExecutor` 拾取执行。`require_human_confirmation` 是设计上的安全闸，不要在没有评审流程的情况下绕过。
