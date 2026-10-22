# 存储与运维

> 关键源码：`src/argus/storage/`、`src/argus/core/alert_ring_buffer.py`、`src/argus/core/video_encoder.py`、`src/argus/core/scheduler.py`

Argus 当前使用 SQLite 保存结构化业务数据，使用本地文件系统保存图像、录像、基线、模型、导出产物、日志和备份。

## 数据目录

默认配置中的主要目录：

| 目录 | 用途 |
| --- | --- |
| `data/db/argus.db` | SQLite 主数据库 |
| `data/alerts` | 告警快照、热力图和导出素材 |
| `data/recordings` | 告警录像和 replay 元数据 |
| `data/inference_records` | 文件形式的推理审计记录 |
| `data/baselines` | 基线图片和版本目录 |
| `data/models` | 训练模型产物 |
| `data/exports` | OpenVINO/ONNX 等导出产物 |
| `data/backbones` | SSL backbone 产物 |
| `data/model_packages` | 模型包和模型卡 |
| `data/logs` | 运行日志 |
| `data/backups` | 数据库/配置备份 |

## ORM 模型

`src/argus/storage/models.py` 定义主要表：

- `AlertRecord`：告警、工作流、分组、物理轨迹、分类、分割、跨摄像头、模型版本 lineage，以及 P1 诊断证据字段。
- `BaselineRecord` / `BaselineVersionRecord`：基线图片和版本生命周期。
- `TrainingRecord` / `TrainingJobRecord`：训练历史和异步训练作业。
- `ModelRecord` / `ModelVersionEvent` / `ShadowInferenceLog`：模型版本、阶段流转、shadow 评分。
- `InferenceRecord`：非 NORMAL 帧的 DB 推理记录。
- `FeedbackRecord` / `LabelingQueueRecord`：反馈和标注队列。
- `AlertRecordingRecord`：告警录像元数据。
- `AuditLog`：审计日志。
- `User`、`Region`、`BackboneRecord` 等运维数据。

多数 ORM 对象带 `to_dict()`，供 API route 返回 JSON。

## 数据库初始化与迁移

`Database.initialize()` 做：

1. 创建 SQLite 目录。
2. 建 engine 和 session factory。
3. `Base.metadata.create_all()`。
4. SQLite WAL、busy timeout、synchronous、cache、temp store pragma。
5. `PRAGMA quick_check`。
6. `_auto_migrate()` 增加旧库缺失列。
7. `_drop_deprecated_tables()` 删除代码已退役的表。

轻量迁移由模块级 `_AUTO_MIGRATIONS` 管理。新增 ORM 字段时必须同步检查这里，否则已有数据库会缺列。P1 诊断证据相关列包含 `detection_type`、`evidence_summary`、`evidence_json` 和 `suppression_reason`；其中 `evidence_json` 以 TEXT JSON 存储，由 `AlertRecord.to_dict()` 解析回对象。

守门测试：

- `tests/lint/test_orm_auto_migrate.py`

## 告警存储

告警链路：

```text
AlertDispatcher
  -> save snapshot / heatmap
  -> Database.save_alert()
  -> WebSocket / webhook / email
  -> AlertRecordingStore metadata
```

`Database.save_alert()` 接受的字段很多，包含基础告警、事件分组、速度/轨迹、分类、跨摄像头、分割、类别、模型版本 id 和诊断证据字段。P1 证据字段的流向是 `Alert.evidence_summary` / `Alert.evidence_json` / `Alert.suppression_reason` -> `Database.save_alert()` -> `AlertRecord` -> API/WebSocket/前端。新增告警字段时要同步：

- `Alert` dataclass 或相关 payload。
- `AlertRecord` ORM。
- `_AUTO_MIGRATIONS`。
- `Database.save_alert()`。
- route 序列化和前端类型。
- replay/report/export 逻辑。

## 录像与 Replay

`AlertFrameBuffer` 在 pipeline 中维护告警前后帧。告警成立后生成 `SolidifiedRecording`，交给 `AlertRecordingStore`：

- 保存 MP4。
- 保存 metadata。
- 保存信号时间线。
- 保存帧和热力图索引。
- 支持 post frames 追加。
- 支持 replay frame、heatmap、reference、pin frame、clip。
- 支持 moov reindex 和旧录像修复。

Replay API 基于 `AlertRecordingStore` 和 DB alert record 组合出前端需要的素材。

## 推理记录

推理记录有两条路径：

- 文件路径：`InferenceRecordStore` 保存非 NORMAL 帧和对应原始帧。
- DB 路径：`InferenceBuffer` write-behind 批量写入 `inference_records` 表。

当前 runner 默认只持久化 `INFO`、`INFO_CANDIDATE`、`SKIPPED_HEARTBEAT` 和 `ALERT` 决策，避免安静摄像头产生高频 `NORMAL` 或 `DEBUG_NORMAL_SAMPLE` 行。如果后续评估需要负样本，应显式调整 buffer 策略和保留策略。

## 备份、保留与清理

- `BackupManager` 创建、恢复、列出、删除备份，并执行轮转。
- `Database.delete_old_alerts()` 删除旧告警，可返回关联录像路径。
- `AlertRecordingStore.cleanup_old()` 删除旧录像目录。
- `create_maintenance_tasks()` 注册旧告警清理、推理记录清理和磁盘检查。
- `ContinuousRecordingManager` 管理持续录像片段，独立于告警 replay 录像。

## 审计

`AuditLogger` 负责写用户行为和关键状态变更。敏感动作应写入真实用户名，不应硬编码 operator。相关守门测试：

- `tests/lint/test_no_hardcoded_audit_operator.py`

## 相关测试

- `tests/unit/test_database.py`
- `tests/unit/test_storage_recorder.py`
- `tests/unit/test_alert_recording.py`
- `tests/unit/test_inference_record.py`
- `tests/unit/test_inference_record_candidate.py`
- `tests/unit/test_database_alert_evidence_migration.py`
- `tests/unit/test_inference_buffer.py`
- `tests/unit/test_backup_route.py`
- `tests/unit/test_reports_route.py`
- `tests/unit/test_replay_route.py`
- `tests/lint/test_orm_auto_migrate.py`
